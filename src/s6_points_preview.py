"""S6: points and axes preview in the headset, drawn with torch (no gsplat).

Each Gaussian center of the train scene is drawn as one point in its DC color.
A 1 m reference cube (white) and axis lines (red X, green Y, blue Z) sit in the
same local space. Their size is in metres. The scene is scaled to metres with
METRES_PER_UNIT, and COLMAP's y-down, z-forward frame is flipped to y-up.

Purpose: check orientation, scale, and eye placement before gsplat is in the
loop. Head pose and field of view come from OpenXR, converted by xr_camera.py.
Frames are written into the swapchain through CUDA-GL interop, as in S4.

Runs on this PC. Nothing runs on the headset.
"""

import argparse
import ctypes
import logging
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import xr
from cuda.bindings import runtime as cudart
from OpenGL import GL

from s4_spike import cuda_check
from splat_io import load_gaussians
from xr_camera import intrinsics_from_fov, pose_to_matrix, viewmat_from_pose
from xr_test_pattern import ACTIVE_STATES, DEFAULT_RUNTIME, TestPattern, heading_degrees

log = logging.getLogger("s6_points_preview")

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "data" / "tnt-train" / "point_cloud.ply"
RENDER = 2160                      # per-eye swapchain size, as planned for the 50 Hz path
METRES_PER_UNIT = 0.1              # scene units to metres (not metric in the source data)
KEEP_RADIUS_UNITS = 12.0           # scene units kept around the median (about 1.2 m at 0.1 m/unit)
SCENE_OFFSET_M = (0.0, 1.0, -3.0)  # scene centre in local metres: 3 m ahead, 1 m up
CUBE_CORNER_M = (1.2, 0.0, -1.0)   # 1 m reference cube: x 1.2..2.2, y 0..1, z -1..-2
AXIS_ORIGIN_M = (0.0, 0.0, -1.0)   # axis marker: 1 m ahead of the start pose
AXIS_LEN_M = 0.6
SH_C0 = 0.28209479177387814


def scene_points(path, device, keep_radius_units=KEEP_RADIUS_UNITS):
    """Scene centres in local metres, within `keep_radius_units` of the median.

    The trained scene has a long halo of outlier Gaussians. Half of them lie within
    2.8 units of the median, but 10% lie beyond 62 units. A percentile cut still
    kept that halo, which spread the preview over about 19 m.
    """
    g = load_gaussians(path, device=device)
    center = torch.median(g.means, dim=0).values
    dist = torch.linalg.norm(g.means - center, dim=1)
    keep = dist <= keep_radius_units
    means = g.means[keep]
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device))  # COLMAP y-down, z-forward -> y-up
    local = (means - center) @ flip.T * METRES_PER_UNIT + torch.tensor(SCENE_OFFSET_M, device=device)
    rgb = (g.colors[keep, 0, :] * SH_C0 + 0.5).clamp(0, 1)
    return local, (rgb * 255).round().to(torch.uint8)


def marker_points(n=600):
    c0 = np.array(CUBE_CORNER_M, dtype=np.float32)
    corners = [c0 + np.array([sx, sy, -sz], dtype=np.float32) for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)]
    pts, cols = [], []
    for a, b in combinations(corners, 2):
        if abs(np.abs(a - b).sum() - 1.0) < 1e-6:  # cube edges differ in exactly one coordinate by 1 m
            t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
            pts.append(a + (b - a) * t)
            cols.append(np.tile([255, 255, 255], (n, 1)))
    # OpenXR axes: +X right (red), +Y up (green), +Z toward the user (blue).
    origin = np.array(AXIS_ORIGIN_M, dtype=np.float32)
    axes = [(np.array([1, 0, 0], np.float32), (255, 60, 60)),
            (np.array([0, 1, 0], np.float32), (60, 255, 60)),
            (np.array([0, 0, 1], np.float32), (80, 120, 255))]
    for direction, color in axes:
        t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
        pts.append(origin + direction * AXIS_LEN_M * t)
        cols.append(np.tile(np.array(color, np.uint8), (n, 1)))
    return np.concatenate(pts), np.concatenate(cols).astype(np.uint8)


@torch.no_grad()
def draw_points(buf, pts, rgb, view_cv, k, width, height, radius=0):
    """Draw points into an (H, W, 4) uint8 buffer with a depth buffer.

    Each pixel keeps the nearest point. Writing in unsorted order is not enough:
    when several points share a pixel, an unordered scatter picks an arbitrary
    winner, which made the first preview shimmer when the head moved.
    """
    buf.zero_()
    buf[..., 3] = 255
    pc = pts @ view_cv[:3, :3].T + view_cv[:3, 3]
    z = pc[:, 2]
    keep = z > 0.05
    pc, z, col = pc[keep], z[keep], rgb[keep]
    u = torch.floor(k[0, 0] * pc[:, 0] / z + k[0, 2]).long()
    v = torch.floor(k[1, 1] * pc[:, 1] / z + k[1, 2]).long()
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z, col = u[inside], v[inside], z[inside], col[inside]
    if radius:
        # Footprint: each point also covers a (2r+1) x (2r+1) block of pixels.
        offsets = [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)]
        u = torch.cat([u + dx for dx, _ in offsets])
        v = torch.cat([v + dy for _, dy in offsets])
        z = z.repeat(len(offsets))
        col = col.repeat(len(offsets), 1)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, z, col = u[inside], v[inside], z[inside], col[inside]
    flat_idx = v * width + u
    nearest = torch.full((width * height,), float("inf"), dtype=z.dtype, device=z.device)
    nearest.scatter_reduce_(0, flat_idx, z, reduce="amin", include_self=True)
    winner = z == nearest[flat_idx]
    buf.view(-1, 4)[flat_idx[winner], :3] = col[winner]


def write_interop(resource, pat, width, height):
    cuda_check(cudart.cudaGraphicsMapResources(1, resource, 0))
    array = cuda_check(cudart.cudaGraphicsSubResourceGetMappedArray(resource, 0, 0))
    cuda_check(cudart.cudaMemcpy2DToArray(
        array, 0, 0, pat.data_ptr(), width * 4, width * 4, height,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice))
    cuda_check(cudart.cudaGraphicsUnmapResources(1, resource, 0))
    torch.cuda.synchronize()
    GL.glFinish()


class PointsPreview(TestPattern):
    color_formats = (GL.GL_SRGB8_ALPHA8,)

    def __init__(self, scene_path, footprint=0):
        super().__init__()
        self.scene_path = scene_path
        self.footprint = footprint
        self.resources = []
        self.bufs = []
        self.pts = None
        self.rgb = None

    def setup(self):
        pts_s, rgb_s = scene_points(self.scene_path, "cuda")
        pts_m, rgb_m = marker_points()
        self.pts = torch.cat([pts_s, torch.from_numpy(pts_m).to("cuda")])
        self.rgb = torch.cat([rgb_s, torch.from_numpy(rgb_m).to("cuda")])
        log.info("points: %d scene centres + %d markers", len(pts_s), len(pts_m))
        self.bufs = [torch.zeros((RENDER, RENDER, 4), dtype=torch.uint8, device="cuda") for _ in range(2)]
        for eye in range(2):
            self.resources.append([
                cuda_check(cudart.cudaGraphicsGLRegisterImage(
                    img.image, GL.GL_TEXTURE_2D,
                    cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard))
                for img in self.images[eye]])
        torch.cuda.synchronize()

    def run(self, seconds, log_interval):
        deadline = time.monotonic() + seconds
        next_log = time.monotonic() + log_interval
        frames = 0
        draw_ms, write_ms = [], []
        log.info("running for %.0f s", seconds)
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
            heading = None
            if frame_state.should_render:
                view_state, views = xr.locate_views(
                    self.session,
                    xr.ViewLocateInfo(xr.ViewConfigurationType.PRIMARY_STEREO,
                                      frame_state.predicted_display_time, self.space))
                heading = heading_degrees(views[0].pose.orientation)
                for eye in range(2):
                    q, p, f = views[eye].pose.orientation, views[eye].pose.position, views[eye].fov
                    cam = pose_to_matrix((q.x, q.y, q.z, q.w), (p.x, p.y, p.z))
                    view_cv = torch.tensor(viewmat_from_pose(cam), dtype=torch.float32, device="cuda")
                    k = torch.tensor(intrinsics_from_fov(f.angle_left, f.angle_right, f.angle_up, f.angle_down,
                                                         RENDER, RENDER), dtype=torch.float32, device="cuda")
                    t0 = time.perf_counter()
                    draw_points(self.bufs[eye], self.pts, self.rgb, view_cv, k, RENDER, RENDER,
                                radius=self.footprint)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    index = xr.acquire_swapchain_image(self.swapchains[eye], xr.SwapchainImageAcquireInfo())
                    xr.wait_swapchain_image(self.swapchains[eye], xr.SwapchainImageWaitInfo(xr.INFINITE_DURATION))
                    write_interop(self.resources[eye][index], self.bufs[eye].flip(0).contiguous(), RENDER, RENDER)  # GL rows: bottom first
                    xr.release_swapchain_image(self.swapchains[eye], xr.SwapchainImageReleaseInfo())
                    t2 = time.perf_counter()
                    draw_ms.append((t1 - t0) * 1000)
                    write_ms.append((t2 - t1) * 1000)
                layer = self.build_layer(views)
            end = xr.FrameEndInfo(frame_state.predicted_display_time, xr.EnvironmentBlendMode.OPAQUE)
            if layer is not None:
                end.layers = [ctypes.byref(layer)]
            xr.end_frame(self.session, end)
            frames += 1
            now = time.monotonic()
            if now >= next_log:
                h = "n/a" if heading is None else f"{heading:+.1f} deg"
                log.info("heading %s | frames %d | draw ms %.2f | write ms %.2f | state %s",
                         h, frames, np.mean(draw_ms) if draw_ms else float("nan"),
                         np.mean(write_ms) if write_ms else float("nan"), self.state.name)
                frames = 0
                draw_ms.clear()
                write_ms.clear()
                next_log = now + log_interval

    def close(self):
        for regs in self.resources:
            for res in regs:
                cudart.cudaGraphicsUnregisterResource(res)
        self.resources = []
        super().close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="S6: points and axes preview in the headset.")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--log-interval", type=float, default=2.0)
    parser.add_argument("--footprint", type=int, default=0,
                        help="draw each point as a (2r+1)^2 block; 0 = one pixel per point")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log.info("footprint radius %d", args.footprint)
    runtime = os.environ.get("XR_RUNTIME_JSON", str(DEFAULT_RUNTIME))
    if not Path(runtime).is_file():
        raise SystemExit(f"OpenXR runtime manifest not found: {runtime}")
    os.environ["XR_RUNTIME_JSON"] = runtime

    app = PointsPreview(SCENE, footprint=args.footprint)
    try:
        app.create_instance()
        app.prepare_system()
        app.width = app.height = RENDER
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
