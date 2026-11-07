"""S3 test pattern: one solid color per eye, through OpenXR on the Steam Frame.

Left eye is red, right eye is blue. Every interval the app logs the head
heading: degrees from straight ahead, positive when the head turns right.
That lets the head-tracking direction be checked against the real motion.

Runs on this PC. Nothing runs on the headset. SteamVR must already be running
with the headset active; see README.md.
"""

import argparse
import ctypes
import logging
import math
import os
import time
from pathlib import Path

# pyglfw loads its Wayland build whenever XDG_SESSION_TYPE is wayland. The GLX
# path used by the Xlib binding needs the X11 build, so choose it before import.
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import glfw  # noqa: E402
import xr  # noqa: E402
from OpenGL import GL, GLX  # noqa: E402

DEFAULT_RUNTIME = Path.home() / ".local/share/Steam/steamapps/common/SteamVR/steamxr_linux64.json"
EYE_COLORS = ((0.9, 0.1, 0.1, 1.0), (0.1, 0.2, 0.9, 1.0))  # left, right
COLOR_FORMATS = (GL.GL_SRGB8_ALPHA8, GL.GL_RGBA8)  # first one the runtime offers wins
ACTIVE_STATES = (
    xr.SessionState.READY,
    xr.SessionState.SYNCHRONIZED,
    xr.SessionState.VISIBLE,
    xr.SessionState.FOCUSED,
)

log = logging.getLogger("xr_test_pattern")


def heading_degrees(q):
    """Angle of the view's forward direction from straight ahead, in degrees.

    OpenXR is right-handed, Y up, looking down -Z. Forward is (0, 0, -1) rotated
    by the head orientation. A right turn moves forward toward +X, which gives a
    positive result.
    """
    ux, uy, uz, w = q.x, q.y, q.z, q.w
    fx, fy, fz = 0.0, 0.0, -1.0
    # v' = f + w*t + u x t, with t = 2 * (u x f)
    tx = 2.0 * (uy * fz - uz * fy)
    ty = 2.0 * (uz * fx - ux * fz)
    tz = 2.0 * (ux * fy - uy * fx)
    vx = fx + w * tx + (uy * tz - uz * ty)
    vz = fz + w * tz + (ux * ty - uy * tx)
    return math.degrees(math.atan2(vx, -vz))


def identity_pose():
    return xr.Posef(xr.Quaternionf(0, 0, 0, 1), xr.Vector3f(0, 0, 0))


class TestPattern:
    color_formats = COLOR_FORMATS  # subclasses may override

    def __init__(self):
        self.instance = None
        self.system_id = None
        self.window = None
        self.session = None
        self.space = None
        self.binding = None
        self.swapchains = []
        self.images = []
        self.fbo = None
        self.state = xr.SessionState.UNKNOWN
        self.exit_requested = False
        self.width = 0
        self.height = 0

    # --- setup -------------------------------------------------------------

    def create_instance(self):
        extensions = xr.enumerate_instance_extension_properties()
        if xr.KHR_OPENGL_ENABLE_EXTENSION_NAME not in extensions:
            raise RuntimeError("runtime does not offer XR_KHR_opengl_enable")
        app_info = xr.ApplicationInfo(
            application_name="steamframe-test-pattern",
            application_version=0,
            engine_name="pyopenxr",
            engine_version=xr.PYOPENXR_CURRENT_API_VERSION,
            api_version=xr.Version(1, 0, xr.XR_VERSION_PATCH),
        )
        ici = xr.InstanceCreateInfo(
            application_info=app_info,
            enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
        )
        self.instance = xr.create_instance(ici)

    def prepare_system(self):
        self.system_id = xr.get_system(
            self.instance, xr.SystemGetInfo(xr.FormFactor.HEAD_MOUNTED_DISPLAY))
        views = xr.enumerate_view_configuration_views(
            self.instance, self.system_id, xr.ViewConfigurationType.PRIMARY_STEREO)
        if len(views) != 2:
            raise RuntimeError(f"expected 2 views, got {len(views)}")
        self.width = views[0].recommended_image_rect_width
        self.height = views[0].recommended_image_rect_height
        log.info("recommended view size: %d x %d per eye", self.width, self.height)

        # The OpenGL graphics requirements call must happen before session creation.
        get_reqs = ctypes.cast(
            xr.get_instance_proc_addr(self.instance, "xrGetOpenGLGraphicsRequirementsKHR"),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR,
        )
        reqs = xr.GraphicsRequirementsOpenGLKHR()
        result = xr.exception.check_result(
            xr.Result(get_reqs(self.instance, self.system_id, ctypes.byref(reqs))))
        if result.is_exception():
            raise result

    def open_window(self):
        # GLX (used by the Xlib binding) needs X11. This session is Wayland, so
        # GLFW would otherwise pick its Wayland backend.
        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        glfw.window_hint(glfw.VISIBLE, False)
        glfw.window_hint(glfw.DOUBLEBUFFER, False)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(64, 64, "steamframe test pattern", None, None)
        if self.window is None:
            raise RuntimeError("could not create the GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # the desktop window must not pace the headset

    def create_session(self):
        self.binding = xr.GraphicsBindingOpenGLXlibKHR()
        self.binding.x_display = GLX.glXGetCurrentDisplay()
        self.binding.glx_context = GLX.glXGetCurrentContext()
        self.binding.glx_drawable = GLX.glXGetCurrentDrawable()
        next_ptr = ctypes.cast(ctypes.pointer(self.binding), ctypes.c_void_p)
        sci = xr.SessionCreateInfo(xr.SessionCreateFlags.NONE, self.system_id, next=next_ptr)
        self.session = xr.create_session(self.instance, sci)
        self.space = xr.create_reference_space(
            self.session,
            xr.ReferenceSpaceCreateInfo(xr.ReferenceSpaceType.LOCAL, identity_pose()),
        )

    def create_swapchains(self):
        offered = xr.enumerate_swapchain_formats(self.session)
        fmt = next((f for f in self.color_formats if f in offered), None)
        if fmt is None:
            raise RuntimeError(f"no supported color format among offered: {list(offered)}")
        for _ in range(2):
            ci = xr.SwapchainCreateInfo()
            ci.usage_flags = xr.SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT
            ci.format = fmt
            ci.sample_count = 1
            ci.width = self.width
            ci.height = self.height
            ci.face_count = 1
            ci.array_size = 1
            ci.mip_count = 1
            sc = xr.create_swapchain(self.session, ci)
            self.swapchains.append(sc)
            self.images.append(xr.enumerate_swapchain_images(sc, xr.SwapchainImageOpenGLKHR))
        self.fbo = GL.glGenFramebuffers(1)
        log.info("swapchains: 2 x %d x %d, format 0x%X", self.width, self.height, fmt)

    # --- session events ----------------------------------------------------

    def poll_events(self):
        while True:
            try:
                buf = xr.poll_event(self.instance)
            except xr.EventUnavailable:
                return
            if xr.StructureType(buf.type) != xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                continue
            ev = ctypes.cast(
                ctypes.byref(buf), ctypes.POINTER(xr.EventDataSessionStateChanged)).contents
            self.state = xr.SessionState(ev.state)
            log.info("session state -> %s", self.state.name)
            if self.state == xr.SessionState.READY:
                xr.begin_session(
                    self.session, xr.SessionBeginInfo(xr.ViewConfigurationType.PRIMARY_STEREO))
            elif self.state == xr.SessionState.STOPPING:
                xr.end_session(self.session)
                self.exit_requested = True
            elif self.state == xr.SessionState.EXITING:
                self.exit_requested = True

    # --- frame -------------------------------------------------------------

    def render_eyes(self):
        for eye in range(2):
            sc = self.swapchains[eye]
            index = xr.acquire_swapchain_image(sc, xr.SwapchainImageAcquireInfo())
            xr.wait_swapchain_image(sc, xr.SwapchainImageWaitInfo(xr.INFINITE_DURATION))
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D,
                self.images[eye][index].image, 0)
            GL.glViewport(0, 0, self.width, self.height)
            GL.glClearColor(*EYE_COLORS[eye])
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            xr.release_swapchain_image(sc, xr.SwapchainImageReleaseInfo())
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def build_layer(self, views):
        layer_views = (xr.CompositionLayerProjectionView * 2)(
            xr.CompositionLayerProjectionView(), xr.CompositionLayerProjectionView())
        for eye in range(2):
            lv = layer_views[eye]
            lv.pose = views[eye].pose
            lv.fov = views[eye].fov
            lv.sub_image.swapchain = self.swapchains[eye]
            lv.sub_image.image_rect.extent = xr.Extent2Di(self.width, self.height)
            lv.sub_image.image_array_index = 0
        layer = xr.CompositionLayerProjection(views=layer_views)
        layer.space = self.space
        layer.view_count = 2
        return layer

    def run(self, seconds, log_interval):
        deadline = time.monotonic() + seconds
        next_log = time.monotonic() + log_interval
        frames = 0
        last_heading = None
        last_valid = False
        log.info("running for %.0f s; turn your head to check heading signs", seconds)
        while time.monotonic() < deadline and not self.exit_requested:
            glfw.poll_events()
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
                view_state, views = xr.locate_views(
                    self.session,
                    xr.ViewLocateInfo(
                        xr.ViewConfigurationType.PRIMARY_STEREO,
                        frame_state.predicted_display_time,
                        self.space,
                    ),
                )
                last_valid = bool(view_state.view_state_flags & xr.ViewStateFlags.ORIENTATION_VALID_BIT)
                last_heading = heading_degrees(views[0].pose.orientation)
                glfw.make_context_current(self.window)
                self.render_eyes()
                layer = self.build_layer(views)
            end = xr.FrameEndInfo(frame_state.predicted_display_time, xr.EnvironmentBlendMode.OPAQUE)
            if layer is not None:
                end.layers = [ctypes.byref(layer)]
            xr.end_frame(self.session, end)
            frames += 1

            now = time.monotonic()
            if now >= next_log:
                heading = "n/a" if last_heading is None else f"{last_heading:+.1f} deg"
                log.info("heading %s (positive = right) | orientation valid %s | frames %d | state %s",
                         heading, last_valid, frames, self.state.name)
                frames = 0
                next_log = now + log_interval

    def close(self):
        for sc in self.swapchains:
            xr.destroy_swapchain(sc)
        if self.session is not None:
            xr.destroy_session(self.session)
        if self.instance is not None:
            xr.destroy_instance(self.instance)
        if self.window is not None:
            glfw.destroy_window(self.window)
            glfw.terminate()


def main(argv=None):
    parser = argparse.ArgumentParser(description="One solid color per eye through OpenXR.")
    parser.add_argument("--seconds", type=float, default=60.0, help="how long to run")
    parser.add_argument("--log-interval", type=float, default=1.0, help="seconds between log lines")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    runtime = os.environ.get("XR_RUNTIME_JSON", str(DEFAULT_RUNTIME))
    if not Path(runtime).is_file():
        raise SystemExit(f"OpenXR runtime manifest not found: {runtime}")
    # Point the loader at SteamVR for this process only. Nothing system-wide changes.
    os.environ["XR_RUNTIME_JSON"] = runtime

    app = TestPattern()
    try:
        app.create_instance()
        app.prepare_system()
        app.open_window()
        app.create_session()
        app.create_swapchains()
        app.run(args.seconds, args.log_interval)
    except xr.exception.ResultException as exc:
        raise SystemExit(
            f"OpenXR error: {exc}. If this is XR_ERROR_RUNTIME_FAILURE, SteamVR is probably "
            "not running with the headset active. Start SteamVR first and wait for it to show "
            "the headset as active.") from exc
    finally:
        app.close()


if __name__ == "__main__":
    main()
