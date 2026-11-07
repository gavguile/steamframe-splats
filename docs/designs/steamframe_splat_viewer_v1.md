# Steam Frame Splat Viewer, Version 1 (as built)

*   **Author:** gavguile
*   **Status:** Approved (record of version 1 as built)
*   **Date:** 2026-11-07

--------------------------------------------------------------------------------

## Overview

This document records version 1 of `steamframe-splats` as it was built and
tested. It is a design record, not a proposal. Every goal below lists the result
that was measured or observed in the headset.

Version 1 is a Python program that runs on a Linux PC with an RTX 3070 Ti. It
renders one trained 3D Gaussian Splatting scene (the Tanks and Temples "train"
scene, 534,812 splats) separately for each eye with gsplat, from the live head
pose. It writes each image straight into the OpenXR swapchain on the GPU. SteamVR
streams the frames over Wi-Fi to a Valve Steam Frame. The user looks around by
moving their head and flies through the scene with the controllers. No code
runs on the headset.

The project is a pipecleaner: a short end-to-end build whose purpose is to find
the real blockers on a new device. The full stage-by-stage record, with the
evidence behind each result, is in [concepts_brief.md](../../concepts_brief.md).

## The Problem

The author had used the Steam Frame only for games, had done no development on
it, and wanted to look at Gaussian splat scenes in it. Nothing on this machine
had ever sent frames to the headset. The questions were:

1.  Can an image rendered on this PC, drawn separately for each eye from the
    live head pose, reach the Steam Frame through SteamVR on Linux and stay fixed
    in the room as the head moves?
2.  Can a trained splat scene be rendered fast enough for that, on this GPU,
    using an established library rather than a renderer written from scratch?

The solution: stream from the PC, through SteamVR's OpenXR runtime. Render with
gsplat (PyTorch and CUDA). Hand frames to OpenXR through CUDA-GL interop. Start
the viewer at a real training-camera position, scaled to roughly life size.

## Goals

Each goal states the criterion and the result.

*   **G1. Frames reach the headset and stay fixed in the room.** Met.
    SteamVR's own OpenXR sample and then this app ran on the headset. After the
    swapchain row-order fix, the author reports the sky is up and the scene stays
    fixed when the head moves.
*   **G2. The render is correct, not just plausible.** Met. The scene was
    rendered from the original training camera poses and compared with the
    training photos: 23.2 to 25.7 dB on training views, and 22.9 dB on a held-out
    view. A mirrored render scores about 9 dB, so the check catches convention
    errors. 25 automated tests cover the loader, the camera conversions, the row
    order, the scene placement, the renderer and the locomotion maths. The
    brief's S7 exit also asked for the headset view to be compared with the
    offline render at the same pose. That was not done. Dropped by author for a
    pipecleaner. The headset evidence is the author's visual check.
*   **G3. Look around with the head, fly with the controllers.** Met. Left
    stick flies where the head looks. Right stick snap-turns 30 degrees and moves
    up and down. Grip gives 3x speed. A resets to the start pose. SteamVR selects
    the Steam Frame controller profile (`/interaction_profiles/valve/frame_controller`).
*   **G4. A usable frame rate.** Partly met. The original target was the
    headset's 90 Hz. Measured: 45 frames per second at 1322 x 1322 per eye (the
    default), and 30 frames per second at 2160. The author accepted the lower
    rate. 90 Hz is not reachable by lowering resolution: below 1322 the stereo
    frame stays at about 11.5 ms.
*   **G5. Reproducible from the public repo.** Met, except the headset part. A
    fresh clone installs with `uv sync`. The fetch scripts download every data
    file and verify it by SHA-256. All tests pass, and the photo comparison
    reproduces exactly. The headset run from the clone was not repeated.
*   **Dropped criteria.** The brief also proposed a 15-minute stereo session,
    a 5-minute free-fly session with no dropped frames, and a dropped-frame
    threshold (D9). Dropped by author: not needed for a pipecleaner. The longest
    headset run was 4 minutes, with no errors.
*   **G6. Public, licensed, and honest about authorship.** Met. Public GitHub
    repo `gavguile/steamframe-splats`, Apache-2.0. The README discloses AI use.
    Commits name only the author.

## Non-Goals

*   Capturing, training, or editing splat scenes. Version 1 views one existing
    scene.
*   Any code on the headset, including an on-device build or a page in the
    headset's browser.
*   A web-based viewer, or reusing an existing splat viewer (decision D1).
*   Windows or macOS support.
*   Committing scene data to the repository. Scripts fetch it instead.
*   90 Hz rendering, foveated rendering, and per-frame overhead work. These are
    candidates for a later iteration.
*   Improving the scene's visual quality. The background is coarse in the
    trained scene itself, and only a better capture would fix that.

## Open Questions

*   **Q1. Scene license.** The trained scene's Hugging Face card declares no
    license. The camera-pose dataset card says MIT. The upstream Tanks and Temples
    terms have not been checked. The repo contains no scene data, only download
    scripts, so any risk falls on whoever runs them.
*   **Q2. Scale.** 2.21 m per scene unit is estimated from an assumed 1.6 m
    camera height and an estimated ground level. It has not been checked against
    a real measurement such as the locomotive's known length.
*   **Q3. Latency.** Motion-to-photon latency over Wi-Fi was never measured. The
    only evidence is the author's report that the scene stays fixed.

## Critical User Journeys (CUJs)

### CUJ 1: First run from a fresh clone

*   **Given**: Linux, an NVIDIA GPU with the CUDA toolkit, SteamVR with the
    Steam Frame streaming, uv installed.
*   **When**: The user runs `uv sync`, `scripts/fetch_scene.sh`,
    `scripts/fetch_reference.sh`, then `uv run python src/viewer.py`.
*   **Then**: gsplat compiles its kernels on first use (about 100 s). In the
    headset, the user stands where training photo 00002 was taken, facing
    locomotive 713, with the sky up. The log shows about 90 frames per 2 s while
    the session is focused.

### CUJ 2: Look and fly

*   **Given**: The viewer is running and the headset is worn.
*   **When**: The user turns their head, then uses the sticks, grip, and A.
*   **Then**: The scene stays fixed under head motion. Overlapping splats do not
    reorder on pure head turns. Movement follows the head direction at 1.5 m/s
    (4.5 m/s with grip). Snap turns rotate around the head. A returns to the
    start pose.

### CUJ 3: Verify the render without the headset

*   **Given**: The data is fetched and CUDA is available.
*   **When**: The user runs `uv run pytest` and `uv run python src/s5_reference_check.py`.
*   **Then**: All tests pass. The photo comparison reports about 24.4 dB on
    training views and 22.9 dB on the held-out view, and passes the 20 dB bar.

## Technical Plan (Architecture)

### Data flow

```text
 data/tnt-train/point_cloud.ply ──► splat_io.load_gaussians ──► Gaussians on GPU (scene units, COLMAP axes)
 data/tnt-train/colmap/*.bin    ──► scene_frame.load_colmap_frame ──► SceneFrame (up, start pose, metres per unit)

 every frame (viewer.Viewer.run):
   xrWaitFrame ─► xrLocateViews (per-eye pose + asymmetric FOV, LOCAL space, metres)
       │
       ├─► xr_input.Controls.poll ─► locomotion.Rig.update  (rig = where the user is in the scene)
       │
       └─► for each eye:
             head pose ──► rig.matrix() @ head ──► xr_camera.viewmat_from_pose   (OpenXR -> OpenCV axes)
                       ──► SceneFrame.view_world                                (metres -> scene units, rigid)
             FOV angles ─► xr_camera.intrinsics_from_fov                        (asymmetric K)
             xr_render.render_xr (gsplat pipeline, splats sorted by distance from the eye)
             viewer.to_swapchain_rgba  (float -> uint8 RGBA, rows flipped for OpenGL)
             CUDA-GL interop copy into the acquired swapchain texture, glFinish, release
   xrEndFrame with a projection layer carrying the LOCAL-space head poses
       │
       ▼
 SteamVR compositor ─► encoder ─► Wi-Fi dongle ─► Steam Frame
```

### Components

| Module | Responsibility |
| --- | --- |
| `splat_io.py` | Reads binary INRIA-layout 3DGS PLY files. Applies the activations (exp scale, sigmoid opacity, normalized quaternions). Reorders SH coefficients from channel-major. |
| `scene_frame.py` | Places the scene in the headset's space. Up is the average up of all training cameras. The start pose is a training photo's position, levelled. Scale comes from an assumed 1.6 m camera height. Composes a head view matrix into a rigid world view matrix, so the splats never move and no SH rotation is needed. |
| `xr_camera.py` | Converts an OpenXR pose (camera to local, -Z forward, Y up) to an OpenCV view matrix (`diag(1, -1, -1) @ inverse(pose)`). Converts the four FOV angles to an asymmetric pinhole K. |
| `xr_render.py` | gsplat's own pipeline (projection, SH colour, tile intersection, rasterization) with two changes: splats are sorted by Euclidean distance from the eye instead of view depth, and splats closer than the near distance are culled by distance. Returns black for zero splats (gsplat aborts on zero). |
| `locomotion.py` | The rig: a yaw and position between OpenXR's local space and the virtual scene. Head-directed flight, a deadzone, snap turns around the head, boost and reset. |
| `xr_input.py` | One action set: a thumbstick vector per hand, grip as boost, and A as reset. Bindings are suggested for the Frame, Index and Touch profiles separately, so one rejection does not block the others. Reports the profile SteamVR chose. |
| `viewer.py` | The application. Owns the OpenXR session (through the `TestPattern` base class), the per-eye CUDA-GL registrations, the frame loop, snapshots and logging. |
| `xr_test_pattern.py` | The OpenXR and GLX setup shared by all headset apps: instance (with optional extensions), the glfw X11 window for the GLX context, session, swapchains, the event loop. Also the S3 test app. |

### Key decisions and why

*   **Render on the PC and stream through SteamVR.** Nothing is built for the
    headset. Iteration is edit and restart. Measured cost: none observed beyond
    the author's "a bit janky on Linux" Wi-Fi link. Latency was not measured.
*   **OpenXR through pyopenxr, with the OpenGL/GLX binding.** Vulkan2 also
    works in SteamVR's sample. GL was chosen because CUDA-GL interop is direct.
    glfw must load its X11 build (`PYGLFW_LIBRARY_VARIANT=x11`) under a Wayland
    session, because GLX needs X11.
*   **The runtime is selected per process.** `XR_RUNTIME_JSON` points the loader
    at SteamVR's manifest for this process only. Nothing system-wide changes.
*   **CUDA-GL interop for the frame path.** 0.44 ms per eye against 7.57 ms for
    a GPU-to-CPU-to-GL copy (S4). Each swapchain image is registered once, then
    mapped, copied and unmapped per frame.
*   **Rows are flipped before the swapchain.** OpenGL images have their origin
    at the bottom-left, and the runtime reads GL swapchain images that way.
    Without the flip every frame was upside down, which went unnoticed until S6.
*   **sRGB swapchain format.** SteamVR offers only `GL_SRGB8_ALPHA8` for GL
    colour swapchains. The splat colours are trained on sRGB photos, so the
    bytes are written unchanged.
*   **Distance sort instead of depth sort.** View depth changes on a head turn,
    so overlapping splats reorder and pop. Distance does not change. Measured
    rotation consistency: 23.0 dB with depth, 27.1 dB with distance. The cost is
    a dependency on gsplat's private `gsplat.cuda._wrapper` module.
*   **The splats stay in scene units.** The head pose is converted into the
    scene, not the scene into the head's space. View-dependent colour stays
    correct with no SH rotation, and nothing is copied per frame except the
    near cut.
*   **1322 x 1322 per eye.** Half the runtime's recommended 2644. A stereo frame
    takes 12 to 15 ms, which fits every second 90 Hz slot: 45 frames per second.

### Frame timing (measured)

Per eye, at 2160 px, full scene, GPU time: projection 0.1 ms, SH colour 0.4 ms,
tile intersection and sort 2.3 ms, rasterization 5.8 ms. There are 5.2 million
splat-tile pairs. Pixel coverage and tile overlap drive the cost, not the splat
count. On top of the GPU time, each eye pays for a full GPU synchronisation,
the colour conversion, the row flip, the interop copy and a `glFinish`. Below
1322 px the stereo frame stays at about 11.5 ms, so 90 Hz needs those fixed
costs cut, not fewer pixels.

### Known structural weaknesses

*   `viewer.py` depends on stage modules: `TestPattern` (from the S3 test app) is
    its base class, and `cuda_check` comes from `s4_spike.py`. The interop write
    function exists twice (`viewer.py` and `s6_points_preview.py`).
*   `xr_render.py` imports `gsplat.cuda._wrapper`, which is gsplat's private
    module. `pyproject.toml` allows `gsplat>=1.5.3`; only `uv.lock` pins 1.5.3.
    A gsplat upgrade can break the viewer without warning.
*   The session handling covers state changes and exit, but nothing was tested
    for a SteamVR restart or a Wi-Fi drop mid-session.
*   The render loop synchronises the GPU several times per eye. That is the
    largest remaining performance cost that does not depend on the scene.

## Alternatives Considered

Some alternatives were rejected by a decision of the author, and others by a
measurement. Each entry says which. Several engine options were never
evaluated: they fell away when the author chose the Python stack. Those entries
say so, rather than claim a technical verdict that was never reached.

### Platform and reuse

#### Alternative A: Reuse an existing splat viewer, or a web viewer (WebXR)

*   **Pros**: Could replace most of the build. A WebXR page in the PC's
    browser would need no CUDA.
*   **Cons**: Less control. A page in the headset's browser would run code
    on the headset and render on its GPU. Linux browser support for WebXR through
    SteamVR was never checked.
*   **Reason for Rejection**: Rejected by author: the project is for fun and
    for full control, so it stays standalone and not web-based, even though
    other projects already do this (decision D1).

#### Alternative B: A game engine (Unity, Unreal, Godot 4)

*   **Pros**: The engine owns the OpenXR swapchain, which removes the
    CUDA-to-swapchain boundary. Mature input and locomotion.
*   **Cons**: Splat rendering is a third-party plugin or addon in each. Unity
    had no supported Linux editor according to the brief (not checked). The
    project and build weight is large for a viewer whose core is one render call.
*   **Reason for Rejection**: Not evaluated. Dropped when the author chose the
    Python stack and asked for the most established splat tooling. gsplat is a
    library, not an engine.

#### Alternative C: Build for the headset itself (on-device)

*   **Pros**: No streaming, so no Wi-Fi latency or encoding.
*   **Cons**: Needs the headset's toolchain and graphics API, neither of which
    was verified. Rendering would move to the headset's much smaller GPU.
*   **Reason for Rejection**: Rejected by author: out of scope for version 1.
    It may become a later project.

#### Alternative D: A C++ host with its own rasterizer

*   **Pros**: No Python binding risk and no PyTorch dependency. Full control
    over the frame loop and synchronisation.
*   **Cons**: Writing a correct sorted, alpha-blended Gaussian rasterizer with
    SH colour is the hard part, and changing language does not remove it.
*   **Reason for Rejection**: Kept as the fallback if the Python path failed at
    the swapchain or missed the budget. It did not fail at the swapchain (S3,
    S4). The budget miss was solved by lowering resolution.

#### Alternative E: The INRIA reference rasterizer

*   **Pros**: The renderer the scene was trained with.
*   **Cons**: Non-commercial research license, according to the brief; the
    LICENSE file was not read. Code copied into an Apache-2.0 public repo would
    carry that license.
*   **Reason for Rejection**: Rejected by author on license grounds. gsplat is
    Apache-2.0 (checked on GitHub).

### Runtime and frame path

#### Alternative F: OpenVR, or Monado, instead of SteamVR's OpenXR runtime

*   **Pros**: OpenVR needs no runtime selection. Monado is open source.
*   **Cons**: OpenVR is SteamVR's legacy API. The Steam Frame streams through
    SteamVR, and Monado was not expected to provide that path (not checked).
*   **Reason for Rejection**: Not needed. SteamVR's OpenXR runtime worked once
    the headset was active (S2), selected per process with `XR_RUNTIME_JSON`.

#### Alternative G: Register SteamVR as the system-wide OpenXR runtime

*   **Pros**: Any OpenXR app finds SteamVR without an environment variable.
*   **Cons**: It changes every OpenXR app on the machine, and needs an undo.
*   **Reason for Rejection**: Not needed. The per-process `XR_RUNTIME_JSON`
    override was enough, and it leaves the system untouched.

#### Alternative H: Vulkan binding instead of OpenGL

*   **Pros**: Works in SteamVR's `helloxr` sample. Modern API.
*   **Cons**: CUDA would need Vulkan external memory instead of the direct
    CUDA-GL interop.
*   **Reason for Rejection**: OpenGL through GLX worked (S3), and CUDA-GL interop
    met the budget (S4). Vulkan was not needed.

#### Alternative I: Copy frames GPU to CPU to GL

*   **Pros**: Simplest code. No interop.
*   **Cons**: Two transfers per frame.
*   **Reason for Rejection**: Rejected on measurement: 7.57 ms per eye against
    0.44 ms for interop at 2644 px (S4). It alone exceeds the 90 Hz budget.

### Rendering choices made during the build

#### Alternative J: gsplat's default depth sort

*   **Pros**: Unmodified library call. No dependency on a private module.
*   **Cons**: Splats reorder when the head turns. The author reported this as
    visible popping in the headset.
*   **Reason for Rejection**: Rejected on measurement and the author's report:
    rotation consistency 23.0 dB against 27.1 dB with the distance sort, at
    equal speed. The author confirmed popping was better.

#### Alternative K: Fewer splats (lower LOD levels) to reach the frame budget

*   **Pros**: Large speed-up was expected.
*   **Cons**: LOD4 (224k) scored 19.0 dB on the held-out view, below the
    20 dB bar. LOD3 and LOD2 were unusable (12.3 and 9.2 dB). LOD2 still took
    11.6 ms per stereo frame, because its splats are large.
*   **Reason for Rejection**: Rejected on measurement. Quality fell before speed
    improved enough.

#### Alternative L: Full or near-full render size (2644 or 2160 per eye)

*   **Pros**: Sharper image.
*   **Cons**: 26.1 ms (2644) and about 23 to 26 ms in the headset (2160) per
    stereo frame. That gives 30 frames per second.
*   **Reason for Rejection**: Rejected by author: "just lower the resolution
    significantly". 1322 px gives 45 frames per second. 2160 stays available as
    `--size 2160`.

#### Alternative M: Remove floaters near the training cameras, or a larger near cut

*   **Pros**: The start pose is a training-camera position, and training leaves
    stray splats in front of each camera.
*   **Cons**: Deleting splats near all training cameras made the ground worse.
    A near cut of 2.5 m deleted a real locomotive about 2 m from the start pose.
*   **Reason for Rejection**: Rejected on rendered comparisons. The remaining
    near-field smear is real but poorly reconstructed geometry. The viewer keeps
    a 0.75 m distance-based cut, and free-fly lets the user move away from it.

#### Alternative N: A points-only preview (S6) as the headset check

*   **Pros**: No gsplat in the loop, so coordinate errors are isolated.
*   **Cons**: Sparse points flicker under head motion. Its own pixel-scatter
    bug and the upside-down swapchain both looked like "the scene skews", so the
    two were hard to tell apart.
*   **Reason for Rejection**: Superseded by the S7 splat viewer after the row
    order fix. The file remains as a debugging tool.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Performance**: 45 frames per second at 1322 px per eye (headset log, 90
    frames per 2 s). Stereo render and write: 12 to 15 ms. Peak GPU memory for
    the full scene: about 630 MiB at 2644 px, measured offline. The first run
    adds about 100 s while gsplat compiles its CUDA kernels.
*   **Security and privacy**: No network access at runtime. The fetch scripts
    download pinned files over HTTPS from Hugging Face and refuse any file whose
    size or SHA-256 does not match. Snapshots and logs stay on the local disk. The
    public repo was scanned for tokens, Steam IDs, device serials and home paths
    before every push.
*   **System impact**: No system-wide OpenXR registration. The only system
    change was adding `/opt/cuda/bin` to the author's zsh and fish start-up
    files, which the author approved.
*   **Rollout and rollback**: A single-user tool. Rollback is `git checkout` of
    an earlier commit. `--size 2160` restores the sharper, slower setting.

--------------------------------------------------------------------------------

### Additional Resources

*   [concepts_brief.md](../../concepts_brief.md): the multi-lens review, the stage
    plan, and the evidence for every stage result (S0 to S9).
*   [README.md](../../README.md): setup, controls and options.
*   Scene: `laphisboy/3DGS-FLoD_pretrained_models` (Hugging Face), revision
    `5fb96747...`. Camera poses and photos: `nvs-bench/tanksandtemples`, revision
    `f5c95295...`.
*   gsplat 1.5.3 (nerfstudio-project, Apache-2.0). pyopenxr 1.1.6501
    (Apache-2.0). Pattern for the GLX session: the maintainer's
    `pyopenxr_examples/xr_examples/gl_example.py`.

### Future Improvements

*   **Cut the per-frame overhead**: render both eyes before writing either, sync
    once per frame instead of several times per eye, and build camera tensors
    without a host round trip. This is the only route to 90 Hz found so far.
*   **Foveated rendering**: SteamVR offers `XR_EXT_eye_gaze_interaction` for
    this headset. Render the periphery at a lower resolution.
*   **A better-captured scene**, with a checked license.
*   **Pin gsplat exactly** in `pyproject.toml`, or vendor the four pipeline
    calls, so a gsplat upgrade cannot break `xr_render.py` silently.
*   **Separate the app from the stage tools**: move the OpenXR session setup and
    `cuda_check` out of `xr_test_pattern.py` and `s4_spike.py` into a shared
    module, and keep one interop write function.
*   **Measure latency** with a phone slow-motion camera, and test SteamVR
    restarts and Wi-Fi drops mid-session.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

This is a map of the files as built (commit `9db3fcc` and later). Every file
exists. No changes are proposed here; the Future Improvements list above names
the candidates.

### Target Files (as built)

#### Application

##### 1. `src/viewer.py` (Exists)

*   **Type / Scope**: Python application, the entry point.
*   **Role**: Class `Viewer(TestPattern)`. `setup()` creates the controls,
    loads the scene and the `SceneFrame`, allocates one RGBA buffer per eye, and
    registers every swapchain image with CUDA. `eye_camera(view)` builds the
    rigid world view matrix and the asymmetric K for one eye, including the rig.
    `run()` is the frame loop: poll events, wait frame, locate views, poll
    input, update the rig, render, convert, flip, interop write, end frame. It
    also logs and saves snapshots. `to_swapchain_rgba()` converts and flips
    rows. Options: `--seconds`, `--size` (1322), `--start-view` (00002.jpg),
    `--near-m` (0.75), `--snapshot-every`, `--log-interval`. Enables
    `XR_VALVE_frame_controller_interaction` when offered.

##### 2. `src/xr_render.py` (Exists)

*   **Type / Scope**: Python module, the renderer.
*   **Role**: `render_xr(g, viewmat, k, width, height, near_distance=0.0,
    sort="distance") -> (H, W, 3) float`. gsplat's classic pipeline for one
    camera, unpacked: `fully_fused_projection`, `spherical_harmonics`,
    `isect_tiles` with distance or depth as the key, `isect_offset_encode`,
    `rasterize_to_pixels`. Culls splats by distance before projection. Returns
    black for zero splats.

##### 3. `src/scene_frame.py` (Exists)

*   **Type / Scope**: Python module, scene placement.
*   **Role**: `estimate_frame(cam_from_world_by_name, points_world, start_name,
    camera_height_m=1.6) -> SceneFrame`. `SceneFrame.view_world(view_local_cv)`
    composes a local view (metres) into a rigid world view (scene units).
    `load_colmap_frame(colmap_dir, start_name)` reads the COLMAP model with
    pycolmap.

##### 4. `src/xr_camera.py` (Exists)

*   **Type / Scope**: Python module, camera conversions.
*   **Role**: `pose_to_matrix(quat_xyzw, position)`, `viewmat_from_pose(cam_to_local)`
    (OpenXR to OpenCV), `intrinsics_from_fov(left, right, up, down, w, h)`,
    `project(viewmat, k, points)` (tests only).

##### 5. `src/locomotion.py` (Exists)

*   **Type / Scope**: Python module, free-fly maths.
*   **Role**: Class `Rig` with `matrix()`, `head_virtual(head_local)`,
    `update(head_local, left_stick, right_stick, boost, dt)`,
    `snap_turn(angle, pivot)`, `reset()`. Constants: 1.5 m/s, 3x boost,
    30-degree snap, 0.15 deadzone, snap arms at 0.7 and re-arms below 0.3.

##### 6. `src/xr_input.py` (Exists)

*   **Type / Scope**: Python module, OpenXR actions.
*   **Role**: Class `Controls(instance, session)`. Action set `flying` with
    `stick` (vector2f per hand), `boost` (squeeze value), `reset` (right A).
    Suggests bindings per profile: Frame, Index, Touch. `poll()` returns
    `(left_stick, right_stick, boost, reset)`, or `None` when the session is not
    focused. `current_profile()` returns the profile SteamVR chose.

##### 7. `src/splat_io.py` (Exists)

*   **Type / Scope**: Python module, the scene loader.
*   **Role**: Dataclass `Gaussians` (means, quats, scales, opacities, colors
    `(N, 16, 3)`, sh_degree). `read_ply`, `load_gaussians(path, device)`, and
    `write_ply` (tests only).

##### 8. `src/xr_test_pattern.py` (Exists)

*   **Type / Scope**: Python module and S3 test app.
*   **Role**: Class `TestPattern`, the OpenXR and GLX base for all headset apps:
    `create_instance` (with `optional_extensions`), `prepare_system`,
    `open_window` (glfw X11 build, hidden window), `create_session` (Xlib
    binding, LOCAL space), `create_swapchains` (one per eye, first offered
    format from `color_formats`), `poll_events`, `build_layer`, `close`. Also
    `heading_degrees` and the solid-colour S3 app.

#### Stage tools (kept for debugging and evidence)

##### 9. `src/s4_spike.py` (Exists)

*   **Type / Scope**: Python app, the S4 interop and copy benchmark.
*   **Role**: Writes a counter pattern through interop or copy and reads it back.
    Also provides `cuda_check`, which `viewer.py` imports (see the structural
    weaknesses).

##### 10. `src/s5_offline_render.py` (Exists)

*   **Type / Scope**: Python app, offline renders and timing.
*   **Role**: `render()` (plain gsplat), `look_at_w2c`, `intrinsics`,
    `colmap_square_view`. Modes: `--explore`, `--render`, `--timing`.

##### 11. `src/s5_reference_check.py` (Exists)

*   **Type / Scope**: Python app, the photo comparison.
*   **Role**: Renders four COLMAP views (three training, one held out). Reports
    PSNR against the photos and a mirrored control. Pass bar: 20 dB.

##### 12. `src/s6_points_preview.py` (Exists)

*   **Type / Scope**: Python app, the S6 point preview (superseded).
*   **Role**: `scene_points`, `marker_points`, `draw_points` (depth-buffered,
    optional footprint). Keeps its own interop write.

#### Data scripts

##### 13. `scripts/fetch_scene.sh` (Exists)

*   **Type / Scope**: Bash. **Role**: Downloads the LOD 5 train PLY at a pinned
    revision into `data/tnt-train/` and verifies its size and SHA-256.

##### 14. `scripts/fetch_reference.sh` (Exists)

*   **Type / Scope**: Bash. **Role**: Downloads the COLMAP model (cameras,
    images, points), four training photos and LOD 2 to 4, all pinned and
    SHA-256 verified. The viewer needs the COLMAP files for its start pose.

#### Tests

##### 15. `tests/test_splat_io.py` (Exists)

*   2 tests: PLY round trip; channel-major SH layout.

##### 16. `tests/test_xr_camera.py` (Exists)

*   6 tests: centre projection; symmetric and asymmetric FOV edges; right turn;
    behind-the-head depth; gsplat honours the asymmetric principal point (GPU).

##### 17. `tests/test_render.py` (Exists)

*   5 tests (GPU): DC colour round trip; +x is image right; +y is image down;
    `render_xr` with the depth sort matches gsplat; the near distance hides a
    close splat.

##### 18. `tests/test_scene_frame.py` (Exists)

*   4 tests: up and scale recovered on a synthetic scene; the start view is
    level and faces the photo direction; head motion moves in metres; swapchain
    rows are flipped for GL.

##### 19. `tests/test_locomotion.py` (Exists)

*   5 tests: forward flies where the head looks; looking down descends; vertical
    and boost; deadzone; snap turn once per flick, in place.

##### 20. `tests/test_points_preview.py` (Exists)

*   3 tests: the nearer point wins; footprint fills a block; drawing is
    deterministic.

#### Project files

##### 21. `pyproject.toml` and `uv.lock` (Exist)

*   Python 3.14 or later. The S3 stack is pinned exactly (pyopenxr 1.1.6501,
    PyOpenGL 3.1.10, glfw 2.11.0, numpy 2.5.4). torch, cuda-python 13.4.4,
    gsplat, pillow and pycolmap are runtime dependencies; pytest is dev. The
    lock pins every installed version.

##### 22. `.gitignore` (Exists)

*   Keeps `data/`, scene formats (`*.ply`, `*.splat`, `*.spz` and others),
    `.venv/`, caches and `*.log` out of git.

##### 23. `README.md`, `LICENSE`, `concepts_brief.md` (Exist)

*   Setup, controls and AI disclosure; Apache-2.0; the stage-by-stage evidence.

##### 24. `docs/designs/steamframe_splat_viewer_v1.md` (Create on finalization)

*   This document.

### Verification Plan

*   **Build**: `uv sync`. gsplat compiles its kernels on first use and needs
    `nvcc`.
*   **Data**: `scripts/fetch_scene.sh && scripts/fetch_reference.sh`. Each
    script exits non-zero on any size or SHA-256 mismatch.
*   **Unit tests**: `uv run pytest`. 25 tests; 6 need CUDA.
*   **Render correctness**: `uv run python src/s5_reference_check.py`. Expect
    about 24.4 dB on training views and 22.9 dB on the held-out view, and a pass.
*   **Static analysis and linting**: none configured. Not part of version 1.
*   **Test coverage requirement**: none set. The tests target the conventions
    that failed or could fail silently: axes, FOV, row order, SH layout, sort
    key, scene placement.
*   **Headset (manual)**: with SteamVR active, run `uv run python src/viewer.py`.
    Check: sky up; the scene stays fixed under head motion; the sticks, grip and A
    work; the log shows about 90 frames per 2 s while focused, and
    `controller profile in use: ... valve/frame_controller`.

--------------------------------------------------------------------------------

*Drafted with EGM*
