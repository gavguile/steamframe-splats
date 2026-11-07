# Steam Frame Splat Viewer, Version 1 (as built)

*   **Author:** gavguile
*   **Status:** Final (record of version 1 as built), 2026-11-07
*   **Date:** 2026-11-07

--------------------------------------------------------------------------------

## Overview

This document records version 1 of `steamframe-splats` as it was built and
tested. It is a design record, not a proposal. Every goal below gives its result
and says whether it was measured offline, observed in the headset, or checked
in the repo.

Version 1 is a Python program that runs on a Linux PC with an RTX 3070 Ti. It
renders one trained 3D Gaussian Splatting scene separately for each eye, with
gsplat, from the live head pose. The scene is the Tanks and Temples "train"
scene with 534,812 splats. It is the full-detail level (LOD 5) of a pretrained
model that is published at five levels of detail. The program writes each image
straight into the OpenXR swapchain on the GPU. SteamVR streams the frames over
Wi-Fi, through a USB Wi-Fi dongle, to a Valve Steam Frame. The user looks around
by moving their head and flies through the scene with the controllers. No code
runs on the headset.

The project is a pipecleaner: a short end-to-end build whose purpose is to find
the real blockers on a new device. The evidence is in
[concepts_brief.md](../../concepts_brief.md). The brief's "Decisions and
results" entries are the evidence. Its planning sections are the original plan,
and they are superseded wherever they differ from this record.

The record names stages and decisions from the brief:

*   S0: desk research
*   S1: device check
*   S2: SteamVR's `helloxr` sample
*   S3: OpenXR test pattern
*   S4: CUDA-to-swapchain spike
*   S5: offline gsplat render
*   S6: points preview
*   S7: splat viewer in the headset
*   S8: controller free-fly
*   S9: fresh-clone check

Dn are the brief's numbered decisions.

## The Problem

The author had used the Steam Frame only for games and had done no development
on it. They wanted to look at Gaussian splat scenes in it. No program the author
wrote had ever sent frames to the headset. The questions were:

1.  Can an image rendered on this PC, drawn separately for each eye from the
    live head pose, reach the Steam Frame through SteamVR on Linux and stay fixed
    in the room as the head moves?
2.  Can a trained splat scene be rendered fast enough for that, on this GPU,
    using an established library rather than a renderer written from scratch?

The solution:

*   Stream from the PC, through SteamVR's OpenXR runtime.
*   Render with gsplat (PyTorch and CUDA).
*   Hand frames to OpenXR through CUDA-GL interop.
*   Start the viewer at a real training-camera position, scaled to roughly life
    size.

## Goals

Each goal states its result. Results live here. Other sections refer to them.

*   **G1. Frames reach the headset and stay fixed in the room.** Met. SteamVR's
    own OpenXR sample, and then this app, ran on the PC and displayed in the
    headset. The rows were upside down until a fix (see Architecture, Key
    decisions). After the fix, the author reports that the sky is up and the
    scene stays fixed when the head moves.
*   **G2. The render is correct.** Partly met. The evidence covers each part of
    the render path:
    1.  **Loader, COLMAP convention and gsplat.** The scene was rendered through
        stock gsplat from the original training camera poses and compared with
        the photos by PSNR. Training views scored 23.2 to 25.7 dB (mean 24.4).
        The held-out view scored 22.9 dB. The pass bar was 20 dB, chosen by the
        author. A left-right mirrored render scores about 9 dB, so the metric
        detects a mirror error.
    2.  **The viewer's renderer, `render_xr`.** It matches gsplat exactly when it
        uses gsplat's depth sort. A unit test checks this on synthetic splats.
    3.  **The distance sort** that the viewer ships with. It is backed by
        rotation consistency: the PSNR between a view and the same view turned
        8 degrees and warped back, over the area where the two overlap, at four
        angles. Depth sort scored 23.0 dB and distance sort 27.1 dB. The viewer
        path was never compared with the photos.
    4.  **The head-pose chain.** It is backed by unit tests of `xr_camera` and
        `scene_frame`, and by the author's headset check. Eye order was checked
        in S3 (left red, right blue).

    25 automated tests cover the loader, the camera conversions, the row order,
    the scene placement, the renderer, the locomotion maths and the S6 points
    preview. 6 of them need CUDA.
*   **G3. Look around with the head, fly with the controllers.** Met (author
    report and logs). The left stick flies where the head looks. The right stick
    snap-turns 30 degrees and moves up and down. Either grip gives 3x speed. A on
    the right controller resets to the start pose. SteamVR selects the Steam Frame
    controller profile (`/interaction_profiles/valve/frame_controller`).
*   **G4. A usable frame rate.** Partly met. The budget at 90 Hz is 11.1 ms per
    stereo frame. A frame that misses its slot waits for the next one, so the
    rate drops to 45 or 30 frames per second. The original target was 90 Hz.
    Measured in the headset: 45 frames per second at 1322 x 1322 per eye (the
    default), and 30 at 2160. The author accepted 45. Frame timing below shows
    why a lower resolution alone does not reach 90 Hz.
*   **G5. Reproducible from the public repo.** Met, except the headset part. A
    fresh clone at commit `3e08ed9` (before the 1322 default) installed with
    `uv sync`. The fetch scripts downloaded every data file and verified it by
    SHA-256. All tests passed. The photo comparison gave 24.43 dB on training
    views and 22.93 dB on the held-out view, the same as on the development
    copy. The headset run from the clone was not repeated.
*   **G6. Public, licensed, AI use disclosed.** Met. Public GitHub repo
    `gavguile/steamframe-splats`, Apache-2.0. The README discloses AI use.
    Commits name only the author.

**Dropped criteria.** The author dropped these brief criteria as unnecessary for
a pipecleaner:

*   a 15-minute stereo session
*   a 5-minute free-fly session with no dropped frames
*   a dropped-frame threshold (D9)
*   the S7 comparison of the headset view with the offline render at the same
    pose

The longest headset run was 4 minutes, with no errors.

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
    trained scene itself. A better capture would help more than any renderer
    change.

## Open Questions

*   **Q1. Scene license.** The trained scene's Hugging Face card declares no
    license. The camera-pose dataset card says MIT. The upstream Tanks and Temples
    terms have not been checked. The repo distributes no scene data. The license
    terms for users who download it are unchecked; check the upstream terms
    first.
*   **Q2. Scale.** 2.21 m per scene unit is an estimate. It comes from an assumed
    1.6 m camera height and an estimated ground level. It has not been checked
    against a real measurement, such as the locomotive's known length.
*   **Q3. Latency.** Motion-to-photon latency over Wi-Fi was never measured. The
    only evidence is the author's report that the scene stays fixed.
*   **Q4. Reprojection.** Does SteamVR reproject head rotation into the missed
    90 Hz slots at 45 frames per second? Not verified. 45 frames per second is
    accepted on the author's visual check only. Movement driven by the
    controller is not reprojected, so fast flight may judder.

## Critical User Journeys (CUJs)

### CUJ 1: First run from a fresh clone

*   **Given**: The README's prerequisites are in place:
    *   Linux with an NVIDIA GPU
    *   Python 3.14 and uv
    *   CUDA 13 with `nvcc` on PATH
    *   SteamVR already running, with the Steam Frame active through its Wi-Fi
        dongle
*   **When**: The user runs the commands in the Verification Plan (build, data,
    headset).
*   **Then**:
    *   gsplat compiles its kernels on first use (about 100 s in this project's
        runs).
    *   In the headset, the user stands where training photo 00002 was taken,
        with the sky up.
    *   The viewer exits after 180 s unless `--seconds` is given.
    *   If SteamVR was not running, startup fails with
        `XR_ERROR_RUNTIME_FAILURE`.

### CUJ 2: Look and fly

*   **Given**: The viewer is running and the headset is worn.
*   **When**: The user turns their head, then uses the sticks, the grip, and A.
*   **Then**:
    *   The scene stays fixed under head motion.
    *   Splats reorder much less on head turns than with depth sorting (see G2).
    *   Movement follows the head direction at 1.5 m/s, or 4.5 m/s with grip.
    *   Snap turns rotate around the left eye, about 3 cm from the head centre.
    *   A returns the user to the start pose.

### CUJ 3: Verify the render without the headset

*   **Given**: The data is fetched and CUDA is available.
*   **When**: The user runs the unit tests and the reference check from the
    Verification Plan.
*   **Then**: The results match G2 and G5.

--------------------------------------------------------------------------------

## Architecture (as built)

### Data flow

```text
 data/tnt-train/point_cloud.ply ──► splat_io.load_gaussians ──► Gaussians on GPU (scene units, COLMAP axes)
 data/tnt-train/colmap/*.bin    ──► scene_frame.load_colmap_frame ──► SceneFrame (up, start pose, metres per unit)

 each frame in viewer.Viewer.run (rendered only while the session is active and should_render is true):
   xrWaitFrame ─► xrLocateViews (per-eye pose + asymmetric FOV, LOCAL space, metres)
       │
       ├─► xr_input.Controls.poll ─► locomotion.Rig.update(left-eye pose used as the head)
       │                               (rig = where the user is in the scene)
       └─► for each eye:
             eye pose ──► rig.matrix() @ eye ──► xr_camera.viewmat_from_pose    (OpenXR -> OpenCV axes)
                      ──► SceneFrame.view_world                                 (metres -> scene units, rigid)
             FOV angles ─► xr_camera.intrinsics_from_fov                        (asymmetric K)
             xr_render.render_xr (copies the splats outside the near cut, sorts by distance from the eye)
             viewer.to_swapchain_rgba  (float -> uint8 RGBA, rows flipped for OpenGL)
             CUDA-GL interop copy into the acquired swapchain texture, glFinish, release
   xrEndFrame with a projection layer of the per-eye LOCAL-space poses and FOVs
              (deliberately without the rig, so the compositor reprojects against real head motion)
       │
       ▼
 SteamVR compositor ─► encoder ─► USB Wi-Fi dongle ─► Steam Frame
```

### Components

| Module | Responsibility |
| --- | --- |
| `splat_io.py` | Reads binary INRIA-layout 3DGS PLY files. Applies the activations: exp scale, sigmoid opacity, normalized quaternions. Reorders SH coefficients from channel-major. |
| `scene_frame.py` | Places the scene in the headset's space. Up is the average up of all training cameras. The start pose is a training photo's position, levelled. Scale comes from an assumed 1.6 m camera height. Composes a head view matrix into a rigid world view matrix, so the splats never move and no SH rotation is needed. |
| `xr_camera.py` | Converts an OpenXR pose (camera to local, -Z forward, Y up) to an OpenCV view matrix (`diag(1, -1, -1) @ inverse(pose)`). Converts the four FOV angles to an asymmetric pinhole K. |
| `xr_render.py` | gsplat's own five-step pipeline, with two changes. Splats are sorted by Euclidean distance from the eye instead of view depth. Splats closer than the near distance are dropped, also by distance. Returns black when no splats remain, because gsplat aborts on zero splats. |
| `locomotion.py` | The rig: a yaw and a position between OpenXR's local space and the virtual scene. Head-directed flight, a deadzone, snap turns, boost and reset. |
| `xr_input.py` | One action set: a thumbstick vector per hand, grip as boost, and A as reset. Bindings are suggested separately for the Frame, Index and Touch profiles, so one rejection does not block the others. Reports the profile SteamVR chose. |
| `viewer.py` | The application. Owns the OpenXR session (through the `TestPattern` base class), the per-eye CUDA-GL registrations, the frame loop, snapshots and logging. |
| `xr_test_pattern.py` | The OpenXR and GLX setup shared by all headset apps: the instance (with optional extensions), the glfw X11 window for the GLX context, the session, swapchains and the event loop. Also the S3 test app. |
| Stage tools: `s4_spike.py`, `s5_offline_render.py`, `s5_reference_check.py`, `s6_points_preview.py` | The S4 interop benchmark, the offline renderer and timing, the photo comparison, and the S6 points preview. The viewer and the tests still import from them (see Known risks and weaknesses). |
| `scripts/fetch_scene.sh`, `scripts/fetch_reference.sh` | Download the scene and the reference data at pinned revisions, and verify each file by SHA-256. |

### Key decisions and why

*   **Render on the PC and stream through SteamVR.** Nothing is built for the
    headset. Iteration is edit and restart. Cost: not measured. The author
    describes the Linux Wi-Fi link as somewhat uneven (author report). Latency
    is Q3.
*   **OpenXR through pyopenxr, with the OpenGL/GLX binding.** Vulkan2 also
    works in SteamVR's sample. GL was chosen because CUDA-GL interop is direct.
*   **The app finds the runtime and the glfw build itself.** It sets
    `XR_RUNTIME_JSON` for its own process, defaulting to the standard Steam
    library path (`DEFAULT_RUNTIME` in `xr_test_pattern.py`). A user with
    SteamVR installed elsewhere sets the variable. Nothing system-wide changes.
    glfw's X11 build is selected the same way (`PYGLFW_LIBRARY_VARIANT=x11`
    through `setdefault`), because GLX needs X11 under a Wayland session.
*   **CUDA-GL interop for the frame path.** 0.44 ms per eye, against 7.57 ms for
    a GPU-to-CPU-to-GL copy, at 2644 px (S4). Each swapchain image is registered
    once, then mapped, copied and unmapped per frame.
*   **Rows are flipped before the swapchain.** OpenGL images have their origin
    at the bottom-left, and the runtime reads GL swapchain images that way.
    Without the flip, every frame was upside down. That went unnoticed until S6,
    because the S3 solid colours and the S4 vertical bar look the same either way.
*   **sRGB swapchain format.** SteamVR offers only sRGB colour formats for GL
    swapchains, and the app uses `GL_SRGB8_ALPHA8`. The splat colours were trained
    on sRGB photos, so the bytes are written unchanged.
*   **Distance sort instead of depth sort.** View depth changes when the head
    turns, so overlapping splats swap order and pop. Distance from the eye does
    not change under a rotation about the eye. Results are in G2.
    `xr_render.py` imports the five pipeline functions from
    `gsplat.cuda._wrapper`. gsplat's public package exports the same five.
*   **The splats stay in scene units.** The head pose is converted into the
    scene, not the scene into the head's space. View-dependent colour stays
    correct with no SH rotation. The near cut still copies all splats outside
    the cut, for each eye, every frame.
*   **1322 x 1322 per eye.** This is half the width of the runtime's recommended
    2644, so a quarter of the pixels. The runtime scales the image up. Result in
    G4. `--size 2160` restores the sharper, slower setting.

### Frame timing (measured)

*   **Budget.** At 90 Hz the budget is 11.1 ms per stereo frame (G4).
*   **GPU time per eye** at 2160 px for the full scene:

    | Stage | Time |
    | --- | --- |
    | Projection | 0.1 ms |
    | SH colour | 0.4 ms |
    | Tile intersection and sort | 2.3 ms |
    | Rasterization | 5.8 ms |

    There are 5.2 million splat-tile pairs. Pixel coverage and tile overlap
    drive the cost, not the splat count.
*   **Stereo frame in the headset** (render and write): 23 to 26 ms at 2160, and
    12 to 15 ms at 1322.
*   **Offline floor.** Offline, render plus conversion stays at about 11.5 ms at
    1080 and 900 px. That run had no interop write. The cause of the floor was
    not profiled.
*   **Candidate fixed costs:**
    *   the per-eye near-cut copy of all splats, which also forces a host sync
    *   the per-eye `torch.linalg.inv`
    *   building the camera tensors from numpy
    *   separate kernel launches
    *   the GPU syncs and `glFinish` per eye
*   **Batching both eyes.** Rendering both eyes in one gsplat call was tried at
    2160 px and gave no gain.

### Known risks and weaknesses (as built)

*   **Startup:** a cold SteamVR start fails with `XR_ERROR_RUNTIME_FAILURE`.
    Start SteamVR first.
*   **Errors mid-session:** a `wait_frame` error is retried at once, with no
    back-off. Other OpenXR errors end the program with a message. Nothing
    recovers without a restart. SteamVR restarts and Wi-Fi drops mid-session
    were not tested.
*   **CUDA version:** the CUDA toolkit's major version must match torch's
    (CUDA 13), or gsplat's first-run build fails.
*   **Near cut:** the 0.75 m near cut also hides real close geometry.
*   **gsplat upgrades:** an upgrade (for example `uv lock --upgrade`) can change
    the five gsplat functions' signatures, with no error at install time.
    `pyproject.toml` allows `gsplat>=1.5.3`; `uv.lock` pins 1.5.3.
*   **Stage modules:** `viewer.py` depends on stage modules. Its base class,
    `TestPattern`, comes from the S3 test app, and `cuda_check` comes from
    `s4_spike.py`. The tests and the reference check depend on stage tools:
    `test_render.py` and `s5_reference_check.py` import `s5_offline_render.py`,
    and `test_points_preview.py` imports `s6_points_preview.py`.
    `test_scene_frame.py` imports `viewer`, so the unit tests load the whole
    OpenXR, OpenGL and CUDA stack.
*   **Duplicate code:** the interop write function exists three times: in
    `viewer.py`, in `s6_points_preview.py`, and as `Spike.write_interop` in
    `s4_spike.py`.

## Alternatives Considered

Each entry says whether it was rejected by a decision of the author or by a
measurement. Options that were never evaluated are listed at the end, one line
each.

### Alternative A: Reuse an existing splat viewer, or a web viewer (WebXR)

*   **Pros**: Could replace most of the build. A WebXR page in the PC's
    browser would need no CUDA.
*   **Cons**: The author would not own the renderer or the frame loop. A page in
    the headset's browser would run code on the headset. Linux browser support
    for WebXR through SteamVR was never checked.
*   **Reason for Rejection**: Rejected by author. The project is for fun and for
    full control, so it stays standalone and not web-based (decision D1). Other
    projects may already do this; none was checked.

### Alternative D: A C++ host with its own rasterizer

*   **Pros**: No Python binding risk and no PyTorch dependency. Full control
    over the frame loop and synchronisation.
*   **Cons**: Writing a correct sorted, alpha-blended Gaussian rasterizer with
    SH colour is the hard part, and changing language does not remove it.
*   **Reason for Rejection**: Not needed for version 1. The Python path worked at
    the swapchain (S3, S4). It misses the 90 Hz budget, and the author accepted
    45 frames per second instead (G4). Whether a C++ host would remove the fixed
    per-frame costs is not known.

### Alternative E: The INRIA reference rasterizer

*   **Pros**: It is the renderer the scene was trained with.
*   **Cons**: The brief reports a non-commercial research license; the LICENSE
    file was not read. Code copied into an Apache-2.0 public repo would carry
    that license.
*   **Reason for Rejection**: Rejected by author, on the license as reported in
    the brief (not confirmed). gsplat is Apache-2.0, as checked on GitHub.

### Alternative I: Copy frames from GPU to CPU to GL

*   **Pros**: The simplest code. No interop.
*   **Cons**: Two transfers per frame.
*   **Reason for Rejection**: Rejected on measurement. 7.57 ms per eye against
    0.44 ms for interop at 2644 px (S4). That alone exceeds the 90 Hz budget.

### Alternative J: gsplat's default depth sort

*   **Pros**: An unmodified library call.
*   **Cons**: Splats reorder when the head turns. The author reported this as
    visible popping in the headset.
*   **Reason for Rejection**: Rejected on measurement and on the author's
    report. The rotation-consistency score is lower (G2), at equal speed. The
    author confirmed popping was better with the distance sort.

### Alternative K: Fewer splats (lower LOD levels) to meet the frame budget

*   **Pros**: A large speed-up was expected.
*   **Cons**: At 2644 px, LOD4 (224k splats) scored 19.0 dB on the held-out
    view, below the 20 dB bar. LOD3 and LOD2 were unusable (12.3 and 9.2 dB).
    LOD2 still took 11.6 ms per stereo frame, because its splats are large.
*   **Reason for Rejection**: Rejected on measurement. Quality fell before speed
    improved enough.

### Alternative L: Full or near-full render size (2644 or 2160 per eye)

*   **Pros**: A sharper image.
*   **Cons**: Offline, 26.1 ms per stereo frame at 2644. In the headset, 23 to
    26 ms at 2160, which gives 30 frames per second. 2644 never ran in the
    headset.
*   **Reason for Rejection**: Rejected by author ("just lower the resolution
    significantly"; see the brief's "Render size lowered" entry). Result in G4.

### Alternative M: Remove floaters near the training cameras, or use a larger near cut

*   **Pros**: Would remove the stray splats that training leaves in front of
    each camera, including at the start pose.
*   **Cons**: Deleting splats near all training cameras made the ground worse.
    A near cut of 2.5 m deleted a real locomotive about 2 m from the start pose.
*   **Reason for Rejection**: Rejected on rendered comparisons. The remaining
    near-field smear is real but poorly reconstructed geometry. The viewer keeps
    a 0.75 m cut, measured by distance from the eye, and free-fly lets the user
    move away from it.

### Alternative N: A points-only preview (S6) as the headset check

*   **Pros**: No gsplat in the loop, so coordinate errors are isolated.
*   **Cons**: Its original pixel-scatter bug and the upside-down swapchain both
    looked like "the scene skews", so the two were hard to tell apart. With
    single-pixel points the scene looks sparse. The file keeps its own placement
    (0.1 m per unit, 3 m ahead), not the S7 start pose.
*   **Reason for Rejection**: Superseded by the S7 splat viewer after the
    row-order fix. The file remains as a debugging tool.

### Not evaluated, or not needed

*   **Unity, Unreal, Godot 4.** Not evaluated. They fell away when the author
    chose the Python stack and the most established splat tooling.
*   **An on-device build.** Rejected by author: out of scope for version 1. It
    may become a later project.
*   **OpenVR, or the Monado runtime.** Not needed. SteamVR's OpenXR runtime
    worked once the headset was active (S2).
*   **Registering SteamVR as the system-wide OpenXR runtime.** Not needed. The
    per-process `XR_RUNTIME_JSON` was enough, and it leaves the system untouched.
*   **The Vulkan binding.** Not needed. OpenGL through GLX worked (S3), and
    CUDA-GL interop met its part of the budget (S4).

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Performance**: Results are in G4 and Frame timing. Peak GPU memory for
    the full scene with stock gsplat, measured offline: about 630 MiB at
    2644 px, and 336 MiB at 1322 px.
*   **Security and privacy**: The viewer makes no network calls. Frames do
    stream over Wi-Fi, through SteamVR. The fetch scripts download pinned files
    over HTTPS from Hugging Face, and refuse any file whose size or SHA-256 does
    not match. Snapshots and logs stay on the local disk. The public repo was
    scanned for tokens, Steam IDs, device serials and home paths before every
    push.
*   **System impact**: No system-wide OpenXR registration. Besides installing
    the CUDA toolkit, the only system change was adding `/opt/cuda/bin` to the
    author's zsh and fish start-up files, so that gsplat's first-run kernel
    build finds `nvcc`. The author approved that change.

## Additional Resources

*   [concepts_brief.md](../../concepts_brief.md): the multi-lens review, the
    stage plan, and the evidence for every stage result (S0 to S9).
*   [README.md](../../README.md): prerequisites, setup, controls and options.
*   Scene: `laphisboy/3DGS-FLoD_pretrained_models` (Hugging Face). Camera poses
    and photos: `nvs-bench/tanksandtemples`. The full revisions and SHA-256
    values are pinned in `scripts/fetch_scene.sh` and
    `scripts/fetch_reference.sh`.
*   gsplat 1.5.3 (nerfstudio-project, Apache-2.0). pyopenxr 1.1.6501
    (Apache-2.0). The GLX session follows the maintainer's
    `pyopenxr_examples/xr_examples/gl_example.py`.

## Future Improvements

*   **Profile the fixed per-frame costs before choosing a fix.** This is the
    first step toward 90 Hz.
*   **Foveated rendering.** SteamVR listed `XR_EXT_eye_gaze_interaction` among
    its extensions in the S2 run.
*   **A better-captured scene**, with a checked license.
*   **Import the five functions from `gsplat`**, and pin `gsplat==1.5.3` in
    `pyproject.toml`.
*   **Separate the app from the stage tools.** Move the session setup and
    `cuda_check` into a shared module, and keep one interop write function.
*   **Measure latency**, check reprojection (Q4), and test SteamVR restarts and
    Wi-Fi drops mid-session.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

This is a file map of the code at commit `9db3fcc`. Every file exists. No
changes are proposed; candidates are in Future Improvements. Signatures,
constants and algorithms are documented in the code. They are not repeated
here.

### Target Files (as built)

| File | Role |
| --- | --- |
| `src/viewer.py` | Entry point and frame loop (see Components). Options: `--seconds` (180), `--size` (1322), `--start-view` (`00002.jpg`), `--near-m` (0.75), `--snapshot-every` (0, off), `--log-interval` (2). Enables `XR_VALVE_frame_controller_interaction` when offered. |
| `src/xr_render.py` | `render_xr`: gsplat pipeline with a choice of sort key and the near cut. |
| `src/scene_frame.py` | `estimate_frame`, `load_colmap_frame`, `SceneFrame.view_world`. |
| `src/xr_camera.py` | Pose-to-matrix, OpenXR-to-OpenCV view, FOV-to-K, and `project` (used by tests). |
| `src/locomotion.py` | The `Rig` class. |
| `src/xr_input.py` | The `Controls` class. |
| `src/splat_io.py` | The `Gaussians` dataclass, `load_gaussians`, and `write_ply` (used by tests). |
| `src/xr_test_pattern.py` | The `TestPattern` base class and the S3 test app. |
| `src/s4_spike.py` | S4 benchmark; provides `cuda_check` to `viewer.py`. |
| `src/s5_offline_render.py` | Offline render and timing; provides `render` to tests and the reference check. |
| `src/s5_reference_check.py` | Photo comparison: three training views (00002, 00004, 00006) and one held-out view (00009), with a mirrored control. |
| `src/s6_points_preview.py` | S6 points preview, superseded. |
| `scripts/fetch_scene.sh` | Scene download, pinned and verified. |
| `scripts/fetch_reference.sh` | COLMAP model (needed by the viewer), four photos, and LOD 2 to 4. Pinned and verified. |
| `tests/test_splat_io.py` | 2 tests: PLY round trip; SH channel order. |
| `tests/test_xr_camera.py` | 6 tests: projections, asymmetric FOV, head turns; 1 needs CUDA (gsplat principal point). |
| `tests/test_render.py` | 5 tests, all CUDA: colour, image axes, `render_xr` against gsplat, near cut. |
| `tests/test_scene_frame.py` | 4 tests: up, scale and start view on a synthetic scene; head motion in metres; GL row flip (imports `viewer`). |
| `tests/test_locomotion.py` | 5 tests: flight direction, descent, vertical and boost, deadzone, snap turn. |
| `tests/test_points_preview.py` | 3 tests: depth buffer and determinism of the S6 point drawing. |
| `pyproject.toml`, `uv.lock` | Python 3.14 or later. The S3 stack is pinned exactly. torch, cuda-python 13.4.4, gsplat, pillow and pycolmap are runtime dependencies; pytest is a dev dependency. The lock pins every version. |
| `README.md`, `LICENSE`, `concepts_brief.md`, `.gitignore` | Setup and AI disclosure; Apache-2.0; stage evidence; data and caches kept out of git. |

### Verification Plan

*   **Build**: `uv sync`. gsplat compiles its kernels on first use and needs
    `nvcc`.
*   **Data**: `scripts/fetch_scene.sh && scripts/fetch_reference.sh`. Each
    script exits non-zero on any size or SHA-256 mismatch.
*   **Unit tests**: `uv run pytest`. 25 tests; 6 need CUDA and are skipped
    without it.
*   **Render correctness**: `uv run python src/s5_reference_check.py`. It writes
    `data/s5/reference_check_full.json` and a side-by-side image. It passes when
    the training-view mean and the held-out view are both at least 20 dB.
    Expected values are in G2.
*   **Test coverage requirement**: none set. The tests target the conventions
    that failed or could fail silently: axes, FOV, row order, SH layout, sort
    key and scene placement.
*   **Headset (manual)**: with SteamVR running and the headset active, run
    `uv run python src/viewer.py --seconds 300`. Pass when:
    *   each CUJ 2 outcome is observed in the headset;
    *   while the session is `FOCUSED`, the log shows lines starting
        `frames 89`, `frames 90` or `frames 91` `in 2 s`;
    *   the log shows `controller profile in use: left:
        /interaction_profiles/valve/frame_controller; right:
        /interaction_profiles/valve/frame_controller`.

--------------------------------------------------------------------------------

*Drafted with EGM*
