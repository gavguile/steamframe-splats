"""S7: the train scene rendered with gsplat in the Steam Frame.

You start where training photo 00002 was taken, facing the same way, at about
life size (see scene_frame.py). Each eye is rendered with gsplat from the live
OpenXR pose and field of view, then written into the GL swapchain through
CUDA-GL interop.

OpenGL images have their origin at the bottom-left, and the runtime reads GL
swapchain images that way. gsplat, like the earlier test renderers, produces
row 0 at the top, so rows are flipped on the way into the swapchain. Without
the flip the headset shows the image upside down, and the scene shears when
you turn your head.

--snapshot-every N saves the left-eye frame (as rendered, before the flip) to
data/snapshots/ every N seconds. That is the app's output, not the
compositor's final image.
"""

import argparse
import ctypes
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import xr
from cuda.bindings import runtime as cudart
from OpenGL import GL
from PIL import Image

from s4_spike import cuda_check
from xr_render import render_xr
from scene_frame import load_colmap_frame
from splat_io import load_gaussians
from xr_camera import intrinsics_from_fov, pose_to_matrix, viewmat_from_pose
from xr_test_pattern import ACTIVE_STATES, DEFAULT_RUNTIME, TestPattern

log = logging.getLogger("viewer")

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "data" / "tnt-train" / "point_cloud.ply"
COLMAP = ROOT / "data" / "tnt-train" / "colmap"
SNAPSHOTS = ROOT / "data" / "snapshots"


def to_swapchain_rgba(img, out):
    """img: (H, W, 3) float in [0, 1], row 0 at the top. Writes GL row order (row 0 at the bottom)."""
    out[..., :3] = (img.clamp(0, 1) * 255).round().to(torch.uint8).flip(0)
    return out


def write_interop(resource, buf, width, height):
    cuda_check(cudart.cudaGraphicsMapResources(1, resource, 0))
    array = cuda_check(cudart.cudaGraphicsSubResourceGetMappedArray(resource, 0, 0))
    cuda_check(cudart.cudaMemcpy2DToArray(
        array, 0, 0, buf.data_ptr(), width * 4, width * 4, height,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice))
    cuda_check(cudart.cudaGraphicsUnmapResources(1, resource, 0))
    torch.cuda.synchronize()
    GL.glFinish()


class Viewer(TestPattern):
    color_formats = (GL.GL_SRGB8_ALPHA8,)

    def __init__(self, render_size, start_view, snapshot_every, near_m):
        super().__init__()
        self.near_m = near_m
        self.render_size = render_size
        self.start_view = start_view
        self.snapshot_every = snapshot_every
        self.resources = []
        self.bufs = []

    def setup(self):
        self.g = load_gaussians(SCENE, device="cuda")
        self.frame = load_colmap_frame(COLMAP, self.start_view)
        log.info("scene: %d Gaussians | start view %s | %.3f m per unit | camera height %.3f units",
                 len(self.g), self.start_view, self.frame.metres_per_unit, self.frame.camera_height_units)
        n = self.render_size
        for _ in range(2):
            b = torch.zeros((n, n, 4), dtype=torch.uint8, device="cuda")
            b[..., 3] = 255
            self.bufs.append(b)
        for eye in range(2):
            self.resources.append([
                cuda_check(cudart.cudaGraphicsGLRegisterImage(
                    img.image, GL.GL_TEXTURE_2D,
                    cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard))
                for img in self.images[eye]])
        self.near_units = max(self.near_m / self.frame.metres_per_unit, 0.01)
        log.info("near cut %.2f m (%.3f units): splats closer to the eye are not drawn; sorted by distance", self.near_m, self.near_units)
        torch.cuda.synchronize()

    def eye_camera(self, view):
        q, p, f = view.pose.orientation, view.pose.position, view.fov
        view_local = viewmat_from_pose(pose_to_matrix((q.x, q.y, q.z, q.w), (p.x, p.y, p.z)))
        view_world = self.frame.view_world(view_local)
        k = intrinsics_from_fov(f.angle_left, f.angle_right, f.angle_up, f.angle_down,
                                self.render_size, self.render_size)
        return (torch.tensor(view_world, dtype=torch.float32, device="cuda"),
                torch.tensor(k, dtype=torch.float32, device="cuda"))

    def run(self, seconds, log_interval):
        n = self.render_size
        deadline = time.monotonic() + seconds
        next_log = time.monotonic() + log_interval
        next_snap = time.monotonic() + (self.snapshot_every or 0)
        frames, render_ms = 0, []
        SNAPSHOTS.mkdir(parents=True, exist_ok=True)
        log.info("running for %.0f s at %d x %d per eye", seconds, n, n)
        while time.monotonic() < deadline and not self.exit_requested:
            self.poll_events()
            if self.exit_requested:
                break
            if self.state not in ACTIVE_STATES:
                time.sleep(0.05)
                continue
            try:
                frame_state = xr.wait_frame(self.session, xr.FrameWaitInfo())
            except xr.ResultException:
                continue
            xr.begin_frame(self.session, None)
            layer = None
            if frame_state.should_render:
                _, views = xr.locate_views(
                    self.session,
                    xr.ViewLocateInfo(xr.ViewConfigurationType.PRIMARY_STEREO,
                                      frame_state.predicted_display_time, self.space))
                t0 = time.perf_counter()
                for eye in range(2):
                    view_w, k = self.eye_camera(views[eye])
                    img = render_xr(self.g, view_w, k, n, n, near_distance=self.near_units)
                    if eye == 0 and self.snapshot_every and time.monotonic() >= next_snap:
                        path = SNAPSHOTS / f"left_{time.strftime('%H%M%S')}.png"
                        Image.fromarray((img.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()).save(path)
                        log.info("snapshot %s", path)
                        next_snap = time.monotonic() + self.snapshot_every
                    to_swapchain_rgba(img, self.bufs[eye])
                    sc = self.swapchains[eye]
                    index = xr.acquire_swapchain_image(sc, xr.SwapchainImageAcquireInfo())
                    xr.wait_swapchain_image(sc, xr.SwapchainImageWaitInfo(xr.INFINITE_DURATION))
                    write_interop(self.resources[eye][index], self.bufs[eye], n, n)
                    xr.release_swapchain_image(sc, xr.SwapchainImageReleaseInfo())
                render_ms.append((time.perf_counter() - t0) * 1000)
                layer = self.build_layer(views)
            end = xr.FrameEndInfo(frame_state.predicted_display_time, xr.EnvironmentBlendMode.OPAQUE)
            if layer is not None:
                end.layers = [ctypes.byref(layer)]
            xr.end_frame(self.session, end)
            frames += 1
            now = time.monotonic()
            if now >= next_log:
                ms = f"{np.mean(render_ms):.1f}" if render_ms else "n/a"
                log.info("frames %d in %.0f s | stereo render+write ms %s | state %s",
                         frames, log_interval, ms, self.state.name)
                frames, render_ms = 0, []
                next_log = now + log_interval

    def close(self):
        for regs in self.resources:
            for res in regs:
                cudart.cudaGraphicsUnregisterResource(res)
        self.resources = []
        super().close()


def main(argv=None):
    p = argparse.ArgumentParser(description="View the train scene in the Steam Frame (gsplat).")
    p.add_argument("--seconds", type=float, default=180.0)
    p.add_argument("--size", type=int, default=2160, help="render size per eye")
    p.add_argument("--start-view", default="00002.jpg", help="training photo whose pose is the start pose")
    p.add_argument("--snapshot-every", type=float, default=0.0, help="seconds between left-eye snapshots (0 = off)")
    p.add_argument("--log-interval", type=float, default=2.0)
    p.add_argument("--near-m", type=float, default=0.75,
                   help="hide splats closer than this to the eye, in metres (removes floaters at the start pose)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    runtime = os.environ.get("XR_RUNTIME_JSON", str(DEFAULT_RUNTIME))
    if not Path(runtime).is_file():
        raise SystemExit(f"OpenXR runtime manifest not found: {runtime}")
    os.environ["XR_RUNTIME_JSON"] = runtime

    app = Viewer(args.size, args.start_view, args.snapshot_every, args.near_m)
    try:
        app.create_instance()
        app.prepare_system()
        app.width = app.height = args.size
        app.open_window()
        app.create_session()
        app.create_swapchains()
        app.setup()
        app.run(args.seconds, args.log_interval)
    except xr.exception.ResultException as exc:
        raise SystemExit(f"OpenXR error: {exc}. Start SteamVR and wait for the headset to show active.") from exc
    finally:
        app.close()


if __name__ == "__main__":
    main()
