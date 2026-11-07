"""S4 spike: write a per-frame GPU pattern into the OpenXR swapchain images.

Two write paths, same pattern, same swapchains:

  interop  CUDA writes straight into the GL swapchain texture (CUDA-GL interop).
  copy     GPU -> CPU, then glTexSubImage2D into the same texture.

Each frame's pattern has a horizontal gradient tinted per eye (left red, right
blue), a moving white bar, and a 24-bit frame counter across the top rows.
With --verify-every N, the counter is read back from the texture before the
image is released. A mismatch means a stale or misplaced write.

Timing covers the write only: map/copy/unmap or upload, plus the sync that
must finish before release. Pattern generation runs before the timer starts.

Runs on this PC. Nothing runs on the headset.
"""

import argparse
import ctypes
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import xr
from cuda.bindings import runtime as cudart
from OpenGL import GL

from xr_test_pattern import ACTIVE_STATES, DEFAULT_RUNTIME, TestPattern

log = logging.getLogger("s4_spike")

COUNTER_BITS = 24
CELL = 16  # pixels per counter bit
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "s4"


def cuda_check(result):
    """cuda-python returns (error, value...) tuples. Raise on any error."""
    err = result[0] if isinstance(result, tuple) else result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return result[1] if isinstance(result, tuple) and len(result) > 1 else None


def counter_strip(frame):
    """RGBA strip (CELL rows x COUNTER_BITS*CELL columns). The R channel holds each bit."""
    strip = np.zeros((CELL, COUNTER_BITS * CELL, 4), dtype=np.uint8)
    strip[..., 3] = 255
    for bit in range(COUNTER_BITS):
        strip[:, bit * CELL:(bit + 1) * CELL, 0] = 255 if (frame >> bit) & 1 else 0
    return strip


def decode_counter(row):
    """row: (COUNTER_BITS * CELL, 4) uint8, read from the middle of the counter strip."""
    value = 0
    for bit in range(COUNTER_BITS):
        if row[bit * CELL + CELL // 2, 0] >= 128:
            value |= 1 << bit
    return value


class Spike(TestPattern):
    # SteamVR offers only sRGB color formats for GL swapchains (checked on the
    # S4 run). The counter uses only byte values 0 and 255, which sRGB encoding
    # leaves unchanged, so readback still decodes correctly.
    color_formats = (GL.GL_SRGB8_ALPHA8,)

    def __init__(self, mode, verify_every):
        super().__init__()
        if mode not in ("interop", "copy"):
            raise ValueError(mode)
        self.mode = mode
        self.verify_every = verify_every
        self.frame = 0
        self.bases = []
        self.resources = []  # interop: [eye][image index] -> cudaGraphicsResource
        self.read_fbo = None
        self.write_ms = [[], []]  # per eye, milliseconds per write
        self.verify_ok = 0
        self.verify_bad = []

    # --- GPU setup ---------------------------------------------------------

    def setup_gpu(self):
        device = torch.device("cuda", 0)
        w, h = self.width, self.height
        ramp_x = torch.linspace(0, 255, w, device=device).to(torch.uint8)
        ramp_y = torch.linspace(0, 255, h, device=device).to(torch.uint8)
        for eye in range(2):
            base = torch.zeros((h, w, 4), dtype=torch.uint8, device=device)
            base[..., 0] = ramp_x[None, :] if eye == 0 else 0
            base[..., 1] = ramp_y[:, None]
            base[..., 2] = ramp_x[None, :] if eye == 1 else 0
            base[..., 3] = 255
            self.bases.append(base)
        if self.mode == "interop":
            for eye in range(2):
                regs = []
                for img in self.images[eye]:
                    res = cuda_check(cudart.cudaGraphicsGLRegisterImage(
                        img.image, GL.GL_TEXTURE_2D,
                        cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard))
                    regs.append(res)
                self.resources.append(regs)
        self.read_fbo = GL.glGenFramebuffers(1)
        torch.cuda.synchronize()

    def make_frame_pattern(self, eye):
        device = self.bases[eye].device
        pat = self.bases[eye].clone()
        bar_x = (self.frame * 8) % self.width
        pat[:, bar_x:bar_x + 40, :3] = 255
        strip = torch.from_numpy(counter_strip(self.frame & 0xFFFFFF)).to(device)
        pat[:CELL, :COUNTER_BITS * CELL] = strip
        torch.cuda.synchronize()  # pattern is ready before the timer starts
        return pat

    # --- the two write paths -----------------------------------------------

    def write_interop(self, resource, pat):
        cuda_check(cudart.cudaGraphicsMapResources(1, resource, 0))
        array = cuda_check(cudart.cudaGraphicsSubResourceGetMappedArray(resource, 0, 0))
        cuda_check(cudart.cudaMemcpy2DToArray(
            array, 0, 0, pat.data_ptr(), self.width * 4, self.width * 4, self.height,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice))
        cuda_check(cudart.cudaGraphicsUnmapResources(1, resource, 0))
        torch.cuda.synchronize()
        GL.glFinish()

    def write_copy(self, tex, pat):
        host = pat.cpu().numpy()  # device -> host (pageable memory)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                           GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, host)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glFinish()

    def read_counter(self, tex):
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.read_fbo)
        GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, tex, 0)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        data = GL.glReadPixels(0, CELL // 2, COUNTER_BITS * CELL, 1, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
        if isinstance(data, (bytes, bytearray)):
            arr = np.frombuffer(data, dtype=np.uint8)
        else:
            arr = np.asarray(data, dtype=np.uint8)
        return decode_counter(arr.reshape(COUNTER_BITS * CELL, 4))

    def render_spike(self):
        for eye in range(2):
            sc = self.swapchains[eye]
            index = xr.acquire_swapchain_image(sc, xr.SwapchainImageAcquireInfo())
            xr.wait_swapchain_image(sc, xr.SwapchainImageWaitInfo(xr.INFINITE_DURATION))
            pat = self.make_frame_pattern(eye)
            tex = self.images[eye][index].image
            t0 = time.perf_counter()
            if self.mode == "interop":
                self.write_interop(self.resources[eye][index], pat)
            else:
                self.write_copy(tex, pat)
            self.write_ms[eye].append((time.perf_counter() - t0) * 1000.0)
            if self.verify_every and self.frame % self.verify_every == 0:
                got = self.read_counter(tex)
                expected = self.frame & 0xFFFFFF
                if got == expected:
                    self.verify_ok += 1
                else:
                    self.verify_bad.append({"frame": self.frame, "eye": eye,
                                            "expected": expected, "got": got})
            xr.release_swapchain_image(sc, xr.SwapchainImageReleaseInfo())

    # --- loop ----------------------------------------------------------------

    def run_loop(self, seconds, log_interval):
        deadline = time.monotonic() + seconds
        next_log = time.monotonic() + log_interval
        xr_frames = 0
        log.info("mode=%s verify_every=%d running for %.0f s", self.mode, self.verify_every, seconds)
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
                self.render_spike()
                layer = self.build_layer(views)
            end = xr.FrameEndInfo(frame_state.predicted_display_time, xr.EnvironmentBlendMode.OPAQUE)
            if layer is not None:
                end.layers = [ctypes.byref(layer)]
            xr.end_frame(self.session, end)
            self.frame += 1
            xr_frames += 1

            now = time.monotonic()
            if now >= next_log:
                left = self.write_ms[0][-1] if self.write_ms[0] else float("nan")
                right = self.write_ms[1][-1] if self.write_ms[1] else float("nan")
                log.info("frame %d | state %s | last write ms left %.3f right %.3f | verify ok %d bad %d",
                         self.frame, self.state.name, left, right, self.verify_ok, len(self.verify_bad))
                next_log = now + log_interval
        return xr_frames

    def summary(self, xr_frames, seconds):
        def stats(samples):
            if not samples:
                return None
            a = np.asarray(samples)
            return {"n": int(a.size), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
                    "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}
        per_frame = [l + r for l, r in zip(self.write_ms[0], self.write_ms[1])]
        return {
            "mode": self.mode,
            "seconds": seconds,
            "xr_frames_with_render": len(self.write_ms[0]),
            "xr_frames_total": xr_frames,
            "size": [self.width, self.height],
            "format": "GL_SRGB8_ALPHA8",
            "write_ms_per_eye": stats(self.write_ms[0] + self.write_ms[1]),
            "write_ms_per_frame_both_eyes": stats(per_frame),
            "verify_every": self.verify_every,
            "verify_ok": self.verify_ok,
            "verify_bad": len(self.verify_bad),
            "verify_bad_examples": self.verify_bad[:5],
        }

    def close(self):
        for regs in self.resources:
            for res in regs:
                cudart.cudaGraphicsUnregisterResource(res)
        self.resources = []
        if self.read_fbo is not None:
            GL.glDeleteFramebuffers(1, [self.read_fbo])
            self.read_fbo = None
        super().close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="S4: CUDA interop vs copy into swapchain images.")
    parser.add_argument("--mode", choices=("interop", "copy"), required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--verify-every", type=int, default=0,
                        help="read back the frame counter every N frames (0 = off, for timing)")
    parser.add_argument("--log-interval", type=float, default=2.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    runtime = os.environ.get("XR_RUNTIME_JSON", str(DEFAULT_RUNTIME))
    if not Path(runtime).is_file():
        raise SystemExit(f"OpenXR runtime manifest not found: {runtime}")
    os.environ["XR_RUNTIME_JSON"] = runtime

    app = Spike(args.mode, args.verify_every)
    try:
        app.create_instance()
        app.prepare_system()
        app.open_window()
        app.create_session()
        app.create_swapchains()
        app.setup_gpu()
        xr_frames = app.run_loop(args.seconds, args.log_interval)
        result = app.summary(xr_frames, args.seconds)
    except xr.exception.ResultException as exc:
        raise SystemExit(f"OpenXR error: {exc}. Start SteamVR and wait for the headset to show active.") from exc
    finally:
        app.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.mode}-verify{args.verify_every}.json"
    out.write_text(json.dumps(result, indent=2))
    log.info("summary written to %s", out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
