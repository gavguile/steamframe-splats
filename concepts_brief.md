# Concept Brief: Steam Frame Splat Viewer (pipecleaner)

- **Feature name:** Steam Frame Splat Viewer (pipecleaner)
- **Author:** gavguile
- **Date:** 2026-11-07
- **Mode:** BRAINSTORM
- **Artifact:** concepts_brief.md
- **Status:** Revised after the first hardware test (helloxr). The section below supersedes the open items it names.

## Reviewer Status

| Reviewer | Status |
| --- | --- |
| egm-design (Skeptical Architect) | completed |
| egm-codebase-details (Technical Accuracy and Reuse) | completed |
| egm-zoom-out (Simplicity and Scope) | completed |
| egm-comprehension (Naive Reader) | completed |
| egm-implementation-readiness | completed |
| egm-structural | completed |
| egm-inventive-catalyst | completed |

All seven reviewers returned a report on the first run. None needed a retry.

**Evidence labels used below.**
- **Verified:** read in a local file during this run.
- **Unverified:** not checked. Reviewers had no web access and could not run commands.
- **Constraint:** given in the brief context as a fact or a limit.
- **User report:** the user's own account, not checked here.
- **Author proposal:** the concept's own choice, to be tested.

Where a reviewer made a local-file claim that this run did not re-check, the text says "reviewer check."

## Decisions and results since the review

**Decided by the author**
- **D1 (existing viewers): closed.** The project stays standalone and native. No web-based viewer. Existing projects are not evaluated for reuse.
- **D10 (repo license): Apache-2.0.**
- **Repo:** public, owner `gavguile`, name `steamframe-splats`.
- **Path:** Python first. The PC renders and SteamVR streams the frames. Nothing runs on the headset.

**S1 result (device confirmation): passed.** The logs identify the headset as the Steam Frame ("Autodetected Headset, using Steam Frame profile"). The Wi-Fi dongle is a USB WLAN adapter (ID 28de:2432).

**S2 result (`helloxr`): passed.** Evidence: the run log and the author's report.
- Command: `helloxr -g Vulkan2 -v`, with `XR_RUNTIME_JSON` set to SteamVR's `steamxr_linux64.json` for that process only. No system-wide OpenXR registration was made, so no undo was needed. This deviates from the S2 wording, which asked for registration.
- Runtime: SteamVR/OpenXR 2.18.4.
- Recommended view size: 2644 x 2644 per view, primary stereo.
- Swapchain formats offered: 43, 50, 109, 106, 97, 64, 126, 124, 129, 130 (Vulkan format numbers as logged; not yet mapped to names). The color format chosen was 43 and the depth format was 126.
- Reference spaces: three (LOCAL and STAGE named in the log).
- Session states seen: IDLE, READY, SYNCHRONIZED, VISIBLE, FOCUSED (12:33:01), VISIBLE (12:33:07).
- Tracking: orientation and position both reported as available at system level.
- Controller: a "Right Hand Frame Controller OpenXR Grip" pose action is bound. The interaction profile path is not in the log. Still open for S8.
- Image and head behavior: the author reports that the sample image displays and works. Head-lock was not measured.

**D2 (graphics binding), revised.** Vulkan2 works end to end in `helloxr`. OpenGL through GLX also works: the S3 app renders into GL swapchain images at about 90 Hz (log and author report). The frame-transport choice (interop or copy) stays open until S4.

**S0 results (desk work, no installs of the project's stack).**
- **D5 (scene): decided.** Tanks and Temples "train", LOD 5 (full detail). Source: Hugging Face model repo `laphisboy/3DGS-FLoD_pretrained_models`, revision `5fb96747023120cff1dcbdf21c2ae6f2ae7b2182`, file `tnt/train/point_cloud/lod_5_iteration_30000/point_cloud.ply`. Verified: 534,812 Gaussians, SH degree 3 (from the training config), standard binary 3DGS PLY (62 floats per Gaussian), 132,634,907 bytes, SHA-256 `507cc39c5bde73766d3173aa7065f53aec3487d20d9748327d58b72a118fa80f`. Stored at `data/tnt-train/` (gitignored). Fetched by `scripts/fetch_scene.sh`.
- **Scene license: open.** The repo card declares no license. The scene comes from Tanks and Temples, whose terms have not been checked. The repo does not contain the file, so the risk stays with each user who downloads it. Check the upstream terms before any public use.
- **gsplat:** Apache-2.0 (verified on GitHub). PyPI 1.5.3 ships only a pure-Python wheel and a source tarball, so its CUDA kernels are built on the user's machine. Building needs `nvcc`. Checked after the author installed the toolkit: CUDA 13.4.3 at `/opt/cuda`, nvcc release 13.4 (V13.4.92), and a test kernel compiled and ran with correct output. The toolkit is not on the shell `PATH`, so builds must set `PATH` and `CUDA_HOME`. The driver supports CUDA 13.4. torch 2.15 depends on CUDA 13 packages (verified from PyPI metadata), which matches the toolkit's major version.
- **pyopenxr:** Apache-2.0 (the LICENSE file says so, verified; GitHub's license field says "Other"). PyPI 1.1.6501, requires Python 3.10 or later. Wheel tags not checked.
- **Python (D4, provisional):** PyPI lists torch 2.15.0 with Linux wheels for Python 3.14 (verified from metadata). Whether gsplat and pyopenxr install cleanly on 3.14 is decided by the test install in S3.
- **Reproducibility:** `scripts/fetch_scene.sh` pins the revision and checks size and SHA-256. It was run against the existing file (verified: accepted). Its download branch has not run yet; the same URL was fetched manually.

**S3 result (Python test pattern): passed, with the exit amended below.**
- Stack: `src/xr_test_pattern.py` on pyopenxr 1.1.6501, PyOpenGL, and glfw, through the OpenGL/GLX binding. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. This replaces the earlier `requirements.txt`.
- glfw's Wayland build cannot use the X11 platform that GLX needs. The app selects glfw's X11 build before import (`PYGLFW_LIBRARY_VARIANT=x11`).
- Measured: recommended view 2644 x 2644 per eye. Swapchains use GL_SRGB8_ALPHA8. About 90 frames per second are submitted while the session is visible.
- Checked by the author: left eye red, right eye blue.
- Heading responds to head turns in both directions. During large sweeps it ranged from about -101 to +93 degrees. The sign convention is checked by unit tests on known rotations. The run shows the expected sign pattern, but no turn was timed.
- Known gap: while the session is not visible, the heading holds its last value. Fix before S6.

**S4 result (CUDA-to-swapchain spike): passed. D2 = OpenGL through GLX. D3 = CUDA-GL interop.**
- Setup: 2644 x 2644 per eye. SteamVR offers only sRGB color formats for GL swapchains (`GL_SRGB8_ALPHA8`); plain `GL_RGBA8` is not offered. The first attempt failed at setup for this reason.
- Same pattern on both paths: a gradient tinted per eye, a moving bar, and a 24-bit frame counter in the top rows.
- Interop path: CUDA writes into the GL swapchain texture (register once, then map, device-to-device copy, unmap, sync, `glFinish` before release).
- Copy path: GPU to CPU (pageable memory), then `glTexSubImage2D`.
- Write-only timing, 45 s runs under SteamVR with the headset visible:
  - Interop: 3,753 rendered frames. Per eye, mean 0.44 ms (p95 0.50, max 2.69). Both eyes per frame, mean 0.89 ms (p95 0.97).
  - Copy: 1,400 rendered frames. Per eye, mean 7.57 ms (p95 8.61, max 13.39). Both eyes per frame, mean 15.13 ms (p95 16.27).
- Budget: 11.1 ms per frame at 90 Hz for both eyes plus everything else. The copy path exceeds it on writes alone. Interop uses about 8%.
- Correctness: during the headset run, the counter was read back from the texture before release on every frame: 2,286 checks, 0 mismatches. Offline (hidden GLX window, no OpenXR): 600 of 600 writes correct on each path.
- Not tested: pinned host memory for the copy path. Interop is cheaper by a wide margin and avoids the host round trip, so the choice does not depend on it.
- Visual check (author report): one bar per eye, moving smoothly. No tearing was reported. The counter readback shows our writes land before release. It does not show what the compositor displays.
- Pattern generation was excluded from these timings. It runs in the real frame loop and needs its own line in the S8 budget.

**S5 result (offline gsplat render, no headset): the budget is missed. Author decision needed.**
- Setup: gsplat 1.5.3 on torch 2.15 (CUDA 13.2 runtime), RTX 3070 Ti. Scene: Tanks and Temples train, LOD 5, 534,812 Gaussians, SH degree 3. The loader reads the INRIA layout (checked by a synthetic round trip and by a channel-major SH test). Tests (5) pass: DC color round trip, +x is image right, +y is image down.
- Reference: the scene's own training photos, rendered from their COLMAP poses (the same world frame and convention as the renderer). Control: the same render mirrored left-right.
  - Full scene, training views 00002 / 00004 / 00006: 25.7 / 24.4 / 23.2 dB. Held-out view 00009: 22.9 dB. Mirrored control: about 9 dB. The bar is 20 dB, my choice, not a published standard.
  - LOD4 (224,078): held-out 19.0 dB, fails the bar. LOD3 (131,944): 12.3 dB. LOD2 (27,052): 9.2 dB. Unusable.
- Timing, stereo frame (both eyes), 30 to 50 iterations after warmup:

| Render per eye | Gaussians | Stereo ms, mean (p95) | Peak VRAM |
| --- | --- | --- | --- |
| 2644 | 534,812 | 26.1 (27.2) | 628 MiB |
| 2160 | 534,812 | 19.8 (26.5) | 493 MiB |
| 1322 | 534,812 | 13.2 (20.0) | 336 MiB |
| 2644 | 224,078 (LOD4) | 20.3 (21.9) | 465 MiB |
| 2644 | 131,944 (LOD3) | 18.2 (19.7) | 405 MiB |
| 2644 | 27,052 (LOD2) | 11.6 (13.2) | 335 MiB |

- Budget: 11.1 ms per frame at 90 Hz for both eyes, plus everything else. The full scene at 2160 is 19.8 ms, about 1.8 times the budget. No measured setting reaches 11.1 ms at a useful quality.
- Levers tried:
  - Render size: 1322 per eye is still 13.2 ms.
  - SH degree 0 instead of 3: 13.0 ms against 13.6 ms per eye at 2644. Negligible.
  - LOD: quality falls off before the timing gets near the budget.
  - Batching both eyes in one gsplat call, packed and unpacked: 19.1 to 19.3 ms at 2160. No gain.
- Caveats: scene scale is not metric. SteamVR was running during the timings, and its compositor may share the GPU. The p95 spikes at 2160 and 2644 suggest that. Not separated.
- Decision for the author (the brief's stop rule): the stop rule is "if S4 and S5 cannot meet the budget at 2160 per eye with a scene the author accepts, write up the measurements and stop before S7." Options:
  1. Stop here and write up the measurements.
  2. Accept a lower app frame rate, about 50 Hz at 2160. Test whether SteamVR's reprojection makes it acceptable in the headset. The reprojection behavior for this stream is unverified.
  3. Investigate a faster renderer: profile gsplat, or write a renderer that suits this scene. Large effort. Not started.
- Fixed reference data: `scripts/fetch_reference.sh` pins the COLMAP files, four training photos, and the LOD levels by SHA-256. The LOD files and the photos are under `data/`, which is gitignored.

## Overview

### What is wrong, missing, or risky (read first)

1. **Blocker. No path from this PC to the headset is proven.** Milestone 0 shows that SteamVR sees the headset. It does not show that an app can create an OpenXR session, submit frames, and have them streamed. No OpenXR runtime is registered: `~/.config/openxr/1/` does not exist (Verified). SteamVR runs on Wayland with a DRM lease on `/dev/dri/card1` (Verified, `vrcompositor.txt` lines 15-18). Which graphics binding SteamVR offers under Wayland is Unverified. The first test is SteamVR's own `helloxr` sample, which is present (Verified, `steamapps/common/SteamVR/bin/linux64/helloxr`), run before any Python code.

2. **Blocker. Four candidates could replace most of the build and have not been checked.** They are: existing open-source splat viewers that target OpenXR or SteamVR; a WebXR page run in this PC's browser; JavaScript splat renderers through WebXR; and Godot with a splat addon. The concept lists them as "not yet checked." If one of them runs on Linux with SteamVR and has a compatible license, most of the plan is unnecessary. This must be settled before any code is written (decision D1).

3. **Major. The frame transfer order is backwards.** The concept copies every frame from GPU to CPU, then copies it back into the swapchain. At the panel size of 2160 x 2160 per eye, 8-bit RGBA is about 18.7 MB per eye. For both eyes at 90 Hz that is about 3.4 GB/s in each direction (arithmetic, not timed). Float32 output would be several times larger. CUDA-GL interop, or CUDA into Vulkan external memory, should be timed first. The CPU copy stays as a fallback.

4. **Major. The render size is undecided, and the encoder width may waste part of it.** SteamVR recommends a 2644 x 2644 per-eye render target at 90 Hz, clamped to 1.5x total area (Verified, `vrcompositor.txt` lines 938-941). The logs also set an encode target width of 1152 (Verified, `driver_vrlink.txt` line 106). If the stream is encoded near 1152 wide, GPU time spent above that is discarded. What the setting controls is Unverified.

5. **Major. The frame budget has no measured numbers.** The budget is 11.1 ms per frame at 90 Hz and 8.3 ms at 120 Hz, for both eyes together. It must also cover transfer, the Python loop, and OpenXR waits. The compositor and encoder share the same GPU. No splat-count target, latency threshold, or VRAM ceiling is given. The compositor log shows 120 Hz recommended at 11:57 and 90 Hz at 12:01 and 12:05 (Verified), so the refresh rate changes between sessions.

6. **Major. The coordinate, projection, and splat-data conversions are not named.** OpenXR is right-handed, Y up, looking down -Z. Trained scenes often use a Y-down, Z-forward (COLMAP/OpenCV) convention (Unverified for the chosen scene). OpenXR gives an asymmetric field of view per eye, so a symmetric intrinsics matrix is wrong. Trained PLY files store log scale, logit opacity, and spherical-harmonic coefficients in a fixed field order, and the loader must convert them. A missed conversion looks like a mirrored, upside-down, or washed-out image, which is easy to mistake for a streaming fault.

7. **Major. The scene is not chosen.** No Hugging Face repo, file, revision, or license is named for "train" or "table." Train is commonly associated with Tanks and Temples data, whose terms may be non-commercial (Unverified). Derived splats may inherit those terms.

8. **Major. The Python stack is unverified.** Only `/usr/bin/python3.14` is installed (Verified). The user reports no PyTorch, gsplat, or pyopenxr is installed. A reviewer check found no CUDA toolkit under `/usr/local` or `/opt`. Whether PyTorch, gsplat, and pyopenxr publish wheels for Python 3.14 is Unverified.

9. **Major. The milestone order tests the riskiest seam last.** The CUDA-to-swapchain handoff first appears at the stereo milestone. The desktop-window milestone never touches OpenXR. Only milestone 0 has a stated test, and that test rests on user report and logs.

10. **Major. Registering a runtime changes every OpenXR app on this PC.** Writing the active runtime file routes all of them to SteamVR. No undo is written down. Record the prior state (file absent) and the undo before registering.

11. **Major. Session handling and observability are missing.** No plan covers OpenXR session states, such as the headset being removed, the PC sleeping, a Wi-Fi drop, or a SteamVR restart. No per-frame timing is logged. Without these, a black screen or a stutter can only be traced by reading SteamVR logs by hand.

12. **Major. The controller binding is unknown.** The logs show "Steam Frame Controllers" (Verified, `controller.txt`) and an internal SteamVR profile name (reviewer check). The OpenXR interaction profile string the app must bind is Unverified.

13. **Major. Repo and licensing are not set up.** The directory is not a git repo. No repo license is chosen. gsplat's Apache-2.0 license is Unverified. GitHub's 100 MiB per-file limit is from memory (Unverified). A `.gitignore` must exist before the first commit.

14. **Nit.** The Wi-Fi dongle's end of the link (PC side or headset side) and its band are not recorded. "Custom code on the headset" is undefined. "Engine" is used for a library. "Pipecleaner" is undefined.

### Correction to the given facts

The brief context says the SteamVR logs "do not name the device." The logs do name it (Verified):
- `driver_vrlink.txt` lines 70 and 99 give the model as "Deckard MP," vendor Valve.
- `driver_vrlink.txt` line 105 reads "Autodetected Headset, using Steam Frame profile."
- `vrhw_usb.vdf` line 10 lists a USB device with model "Wifi Dongle."
- `controller.txt` lists "Steam Frame Controllers."

Treat the device as identified in the logs. The logs do not show which end of the link the dongle sits on, and they do not show the Wi-Fi band.

### The proposal, as given

A Python program on this Linux PC creates an OpenXR session through SteamVR. It renders one trained splat scene into both eyes with gsplat on the RTX 3070 Ti and submits the frames. SteamVR encodes and streams them over Wi-Fi to the Steam Frame. Head pose and controller state come back through OpenXR. Version 1 is free-fly with a controller and a head-tracked look. Nothing runs on the headset.

### The pipecleaner question

Can an image rendered on this PC, drawn separately for each eye from the headset's live head pose, reach the Steam Frame through SteamVR and stay fixed in the room as the head moves?

Splat quality and speed are a second question. Answer the first with a plain stereo test pattern before answering the second.

### Facts checked in this run

Verified:
- GPU: RTX 3070 Ti. NVIDIA kernel module version 615.78.08 (`/proc/driver/nvidia/version`). The 8 GB VRAM figure is from the spec, not checked.
- Headset model "Deckard MP" with the Steam Frame profile (`driver_vrlink.txt` lines 70, 99, 105).
- Refresh rate per session: 120 Hz recommended at 11:57, 90 Hz at 12:01 and 12:05 (`vrcompositor.txt` lines 123, 938, 1175).
- Panel 2160 x 2160 per eye; recommended render target 2644 x 2644 at 90 Hz (`vrcompositor.txt` lines 938-941).
- Encode target width 1152 (`driver_vrlink.txt` line 106).
- SteamVR session on Wayland with a DRM lease (`vrcompositor.txt` lines 15-18, 830-833, 1067-1068).
- No OpenXR active runtime file. `~/.config/openxr/1/` does not exist.
- SteamVR ships `helloxr` and `vrpathreg` in `steamapps/common/SteamVR/bin/linux64`.
- Only Python 3.14 is in `/usr/bin`.

Not checked: `nvidia-smi` output, adb, scene sizes and licenses, SteamVR's OpenXR graphics extensions under Wayland, the stream codec, and the Wi-Fi band.

## Analysis

### 1. The frame path into the headset

There are four ways to fill the OpenXR swapchain, listed from most to least expensive:

a. **Copy GPU to CPU, then CPU to swapchain** (the concept's default). Two transfers per frame in each direction. The simplest code and the most expensive. Keep it as a fallback.

b. **CUDA-GL interop.** Register each runtime swapchain image once, then do a device-to-device copy each frame. This needs an OpenGL binding that SteamVR accepts under Wayland (Unverified). It also needs an explicit sync so the compositor never reads a half-written frame. The reviewers lean toward OpenGL because CUDA-GL interop is built for it. That lean is not checked against SteamVR.

c. **CUDA into Vulkan external memory, then a GPU copy into the swapchain.** This needs a Vulkan binding. A reviewer check found an external-memory extension on SteamVR's compositor device (`vrcompositor.txt` lines 74-75). Whether an app can use it is Unverified.

d. **A renderer that writes straight into the swapchain** (GL or Vulkan compute, or a game engine). This removes the CUDA boundary altogether. It needs a different renderer.

Test (a) and (b) at 2160 x 2160 per eye with CUDA events, with SteamVR running. Choose the graphics binding after `helloxr` lists what SteamVR offers.

### 2. Frame budget and render size

- The budget is 11.1 ms per frame at 90 Hz and 8.3 ms at 120 Hz. Both eyes, the transfer, Python, and OpenXR waits all share it.
- The GPU is shared with SteamVR's compositor, the video encoder (whether it uses NVENC is Unverified), and the desktop.
- Storage: a standard trained PLY stores 62 float32 values per Gaussian, 248 bytes each. That is about 250 MB per million Gaussians (layout from memory, Unverified for the chosen file). Storage probably does not bind before compute does.
- Measure with a fixed camera path through the scene, not one view. The number of visible Gaussians changes with the view.
- Render size: start at 2160 per eye. Consider 2644 only after the encode-width question is answered.

### 3. Coordinates and splat conversions

Each of these must be right before stereo is tested:

- **Pose direction.** OpenXR supplies camera-to-world poses. gsplat's expected input is world-to-camera (reviewer check; gsplat's API is Unverified). A missed inverse makes the scene follow the head.
- **Scene axes.** A Y-down scene placed in OpenXR's Y-up space needs a fixed rotation, typically 180 degrees about X. A single-axis negation gives a mirror image that looks almost right, which is harder to spot. The chosen scene's convention is Unverified.
- **Projection.** Each eye's field of view is asymmetric. Derive the intrinsics from the four angles and check them against a known pose.
- **Activations and ordering.** Scale is stored as a log and opacity as a logit. Quaternion order and spherical-harmonic coefficient layout must match gsplat's loader. A wrong order gives wrong colors and no error.
- **Eye order.** Left and right must match the swapchain image order, or the eyes swap.

Test with an axis-marker scene (one colored object at the origin) and a small synthetic PLY with known splats. Both must pass before stereo work begins.

### 4. Controller and free-fly

- A head-tracked look needs only the per-eye views that OpenXR already supplies.
- Free-fly should move an app-owned world offset, not the reference space. The reference space choice (LOCAL or STAGE) is undecided.
- The compositor can reproject for head rotation if reprojection is active for this stream (Unverified). It cannot correct controller-driven translation, so the app must render the moved world itself.
- Render at the predicted display time from `xrWaitFrame` (standard OpenXR practice, from memory).
- Free-fly is not needed to answer the pipecleaner question. The user's scope puts it in version 1, so it stays, but it is staged last (decision D11).
- Locomotion rules are undecided: speed, smooth or snap turn, vertical movement, and the world-up convention (decision D7).

### 5. Runtime registration and the Python environment

- **Registration.** Point the active runtime file at SteamVR's manifest, `steamxr_linux64.json` (present, Verified), or set it through SteamVR's own option (location Unverified). Record the prior state and the undo before doing either.
- **Loaders.** Two OpenXR loader libraries are on disk (reviewer check). Which one a Python binding loads is Unverified.
- **Python.** Only 3.14 is installed. Whether torch, gsplat, and pyopenxr have 3.14 wheels is Unverified. Reviewers suggested a uv-managed 3.12 or 3.13 venv as the lower-risk default. Nothing was checked. Decide by a test install (decision D4).
- **CUDA.** The driver is 615.78.08 (Verified). The CUDA version it supports is not checked, because `nvidia-smi` was not run. Pick a torch wheel built for a CUDA version at or below it.
- **gsplat build.** gsplat may compile CUDA kernels at install or first use, which needs `nvcc`. This is Unverified for the pinned version.
- **pyopenxr.** Its maturity is Unverified. Before relying on it, check that it exposes graphics requirements, swapchain image enumeration, and session creation.

### 6. Scene, license, and repo

- Name the Hugging Face repo, file, and revision. Read the license on each dataset card. Record the Gaussian count and file size.
- The repo holds the download script only. Scene files stay outside it. Pin the revision and a SHA-256 in the script.
- Add `.gitignore` rules for `data/`, `*.ply`, `*.npz`, `*.splat`, `*.ksplat`, `.venv/`, and `__pycache__/` before the first commit.
- The README needs the AI-use disclosure, license notes, the version table, and the runtime undo step.
- Read gsplat's LICENSE and NOTICE at the pinned commit. Confirm that no code is copied from the INRIA repository.

### 7. Sessions and observability

- **Per-run log.** Scene path, Gaussian count, and SH degree at load. Per-stage GPU time from CUDA events. Wall time from `xrWaitFrame` to `xrEndFrame`. Frames where `shouldRender` is false. Every session-state change. XR result codes by name. Peak GPU memory. The SteamVR session start time or ID, so frames can be matched to SteamVR logs.
- **Session states.** Skip rendering when the session is not visible. Exit cleanly on EXITING.
- **Test.** Remove the headset, sleep the PC, and restart SteamVR. Each must leave the app in a known state.
- **Reviewer check, not re-run.** The logs show headset drops that line up with SteamVR restarts (`vrwebhelper_pairing` and five `vrmonitor` startups). Confirm with the author before treating these as link faults.

### 8. Assumptions tested

| Assumption | Status | Test (staging step) |
| --- | --- | --- |
| pyopenxr is mature enough to build on | Unverified | Release history check; run its Linux example against SteamVR (S3) |
| gsplat is Apache-2.0 and suitable | License Unverified; suitability untested | Read LICENSE and NOTICE at the pinned commit; render test (S5) |
| Wi-Fi latency is acceptable | Untested; no threshold | Phone slow-motion camera with a fixed marker; author sets the threshold after measuring (S7) |
| Scene licenses allow a download link | Unverified | Read each dataset card (S0) |
| The repo holds the script, not scene files | Sound in principle; not yet enforced | `.gitignore` and a fresh-clone test (S9) |
| Python 3.14 versus a pinned venv | Pinned venv is the right shape; version Unverified | Test install (S0) |

### 9. Terms and wording (Nit)

- Define "custom code on the headset." Under a plain reading, a JavaScript page the author writes and the headset browser runs counts as custom code.
- "Engine" is used for what is a library (gsplat). Say "library" or "renderer."
- Define "pipecleaner": a short end-to-end build to find the real blockers.
- "Completed streaming session": the logs show encoding activity (reviewer check). Say "ran" unless the end is confirmed.
- Tag every fact with one of the evidence labels above.

## Goals vs Non-Goals

**Goals (as given):**
- View one standard trained splat scene in stereo on the Steam Frame, rendered on this PC and streamed by SteamVR over Wi-Fi.
- Head-tracked look.
- Controller free-fly.
- A public repo with a download script and an AI-use disclosure.

**Goals added for measurement:**
- Stereo frame time and peak VRAM recorded for a named scene at a named render scale.
- Motion-to-photon latency measured before the author sets a target.
- A session length target with no session loss. The author sets the length (a 15-minute stereo session and a 5-minute free-fly session are suggested).
- A dropped-frame threshold set by the author (decision D9).

**Non-goals (as given):**
- Capture, editing, multiplayer, and an on-device build.

**Non-goals (added):**
- Committing scene files to the repo.
- C code in version 1, unless the Python path fails at the swapchain.
- Training or fitting splats, scene conversion tools, or more than one scene format.
- Windows or macOS support.
- Performance work beyond what the measurements require.
- Any new code on the headset. The boundary is defined in decision D8.

## Alternatives Weighed

The "stated reason" column gives the concept's own reasons, with a technical restatement where one is possible. No source links were available in this run, so no rejection is attributed to a named person.

| Alternative | Status | Technical reason | Check to settle it |
| --- | --- | --- | --- |
| Unity | Rejected for this machine | The concept states there is no supported Linux editor. Unity's current Linux status is Unverified. Using it would need a Windows or macOS machine. | Link Unity's system requirements page |
| Unreal Engine | Deferred | A large C++ and Blueprint project for a viewer whose core is one render call. Splat support is a third-party plugin whose OpenXR-on-Linux status is Unverified. | Find a splat plugin; check its license, last commit, Linux support, and OpenXR support |
| Godot 4 | Unassessed; first fallback | The engine owns the swapchain, so the CUDA-to-swapchain boundary goes away. The cost is a splat renderer written as a compute pipeline, or an addon of unknown quality. The concept's stated reason ("third-party") is a maintenance reason, not a technical one. | Check Godot's OpenXR path under SteamVR on Linux; find and check an addon |
| C++ OpenXR host with its own rasterizer | Rejected for version 1 | The hard part is depth-sorting and alpha-blending many anisotropic Gaussians, and evaluating spherical harmonics. Changing the host language does not remove that work. It would remove the Python binding risk and the CUDA boundary. | Revisit if the Python path fails at the swapchain or misses budget |
| INRIA reference rasterizer (diff-gaussian-rasterization) | Rejected | The concept states a non-commercial research license. Copying its code into a public repo would carry that license. It is a training-era CUDA extension, so the CUDA boundary would remain. | Link the LICENSE file in the INRIA gaussian-splatting repository and read it; the author decides whether a public hobby repo falls under it |
| On-device build | Deferred | Needs the headset's toolchain and graphics API, neither of which is verified. It moves rendering to the headset GPU. The scope excludes it. | Revisit as a later project |
| WebXR page run by the headset browser | Rejected for version 1 as the main path; keep as a probe | Puts author JavaScript on the headset, which conflicts with the no-custom-code rule under a plain reading. Renders on the headset GPU, not this PC. Its value is a zero-install test of whether the headset browser runs immersive WebXR. | Zero-install probe (S1b). Headset WebXR support is Unverified |
| WebXR page run in this PC's browser, session through SteamVR's OpenXR runtime | Unassessed | If Linux Chromium can start an immersive-vr session backed by SteamVR, rendering stays on this PC and needs no CUDA. WebGL 2 lacks compute shaders, so sorting is harder. WebGPU has compute. | Check `navigator.xr.isSessionSupported('immersive-vr')` in this PC's browser |
| Existing open-source splat viewers that target OpenXR or SteamVR | Unassessed; highest priority | Could remove most of the build. Names recalled from memory, not checked: aras-p's UnityGaussianSplatting (Unity), PlayCanvas, antimatter15/splat, GaussianSplats3D, Spark. | Search GitHub for OpenXR or SteamVR splat viewers; check license, last commit, Linux support, stereo, and PLY loading |
| JavaScript splat renderers through WebXR (PlayCanvas, Three.js-based) | Unassessed | Same runtime question as the PC-browser WebXR row. Browser memory limits for large scenes are Unverified. | Run one scene in this PC's browser |
| Godot with a community splat addon | Unassessed | Same as the Godot row. | Find and check the addon |
| OpenVR API instead of OpenXR | Unassessed; fallback | Avoids runtime registration, because it goes through SteamVR's own API. It is a legacy API. The Python binding's maturity and Steam Frame support are Unverified. | Use only if registration cannot be fixed |
| Monado (open OpenXR runtime) | Rejected | The Steam Frame streaming path is SteamVR's. Monado is not expected to provide it (Unverified). | None unless the author wants to revisit |
| Copy-to-CPU swapchain path | Rejected as default; kept as fallback | Two transfers per frame. About 3.4 GB/s each way at 2160 per eye, RGBA8, 90 Hz (arithmetic). Float32 is several times more. | Time it against interop (S4) |
| Python with a GL or Vulkan compute renderer (no CUDA) | Unassessed | Removes PyTorch, the CUDA toolkit, and the CUDA-to-swapchain boundary. The cost is writing a correct sorted, alpha-blended rasterizer. | Fallback if the S4 spike fails |
| Gaussian centers drawn as colored points (Python or C++) | Not rejected; recommended first content step | Small job: a PLY reader and point drawing. It checks orientation, scale, and eye placement. It cannot judge splat quality. | Use in S6 |

## Key Decisions & Staging for Next Steps

### Open decisions

- **D1. Go/no-go on existing viewers.** Check the four unassessed candidates before any code. If a viewer works on Linux with SteamVR and has a compatible license, decide before S2 whether this project becomes a learning exercise or stops. Owner: author. Blocks S2.
- **D2. Graphics binding: decided, OpenGL through GLX** (S3 and S4). Vulkan2 also works in `helloxr`, but nothing in the plan needs it.
- **D3. Frame transport: decided, CUDA-GL interop** (S4 timings: 0.44 ms per eye against 7.57 ms). The copy path stays only as a comparison.
- **D4. Python version and venv tooling.** Decided by a test install in S0.
- **D5. Scene** (repo, file, revision, license, Gaussian count). Decided in S0.
- **D6. Render scale.** Start at 2160 per eye. Consider 2644 only after the encode-width question is answered.
- **D7. Locomotion rules.** Speed, smooth or snap turn, vertical movement, world-up convention. Owner: author.
- **D8. Definition of "custom code on the headset."** Decide whether browser-hosted JavaScript the author writes counts.
- **D9. Success numbers.** Frame time, motion-to-photon latency, session length, VRAM ceiling, dropped-frame threshold. Set after the first measurements.
- **D10. Repo license.** Blocks the public push. Owner: author.
- **D11. Free-fly in version 1.** Keep it, as the user's scope says, but stage it last. Owner: author to confirm.

### Provisional decisions (carry forward, not final)

- Python for version 1, pending D1.
- SteamVR as the OpenXR runtime, pending S2.
- The PC renders. The headset runs nothing custom.
- Scene files stay out of the repo. The download script goes in the repo.
- No on-device build.

### Staging

- **S0. Desk work, no installs.** Exit: a written go/no-go and decisions D1, D4, and D5. Tasks: check the four unassessed candidates; read the chosen scene's license and gsplat's LICENSE and NOTICE; check torch, gsplat, and pyopenxr wheel tags for candidate Python versions; run `nvidia-smi` for the CUDA version; write the runtime registration undo.
- **S1. Device confirmation** (existing milestone 0). Exit: a SteamVR device-view screenshot showing the Steam Frame profile, the SteamVR build, and the dongle model. The logs already support this (Verified). Record which end of the link the dongle sits on.
- **S1b. Optional zero-install probe.** Run a WebXR immersive-vr page in the headset browser and in this PC's browser. Exit: a world-locked cube appears in the headset, or the browser's failure is recorded.
- **S2. Register SteamVR as the OpenXR runtime, after writing the undo.** Run `helloxr` before any Python. Exit: a sample image in the headset with head tracking working. Record the graphics bindings, recommended view size, swapchain formats, and the controller interaction profile string.
- **S3. Python OpenXR test pattern in a pinned venv.** Exit (amended after the first run): each eye shows its own solid color; the runtime reports valid orientation; the logged heading responds to head turns with the expected sign. Head-lock moves to S6, because a solid color fills the view and cannot show whether the scene stays fixed in the room. Fallback: the C++ host from the SDK samples.
- **S4. CUDA-to-swapchain spike.** CUDA writes a moving pattern with a frame counter and per-eye markers into the swapchain at the runtime size. Time both the interop path and the copy path. Exit: no tearing and no stale frames; milliseconds per frame recorded for both paths. Decide D2 and D3 here.
- **S5. Offline splat render.** Test the loader on a synthetic PLY with known splats. Render the chosen scene to PNG from a fixed camera. Check for NaN values. Time a stereo frame and log peak VRAM. Exit: the PNG matches a named reference render within a stated tolerance; a timing table exists; the author sets the go/no-go against the budget.
- **S6. Points preview and axis test in the headset.** Draw Gaussian centers as colored points, with an axis-marker scene. Exit: the scene appears in both eyes with correct orientation and scale against the reference object. This isolates coordinate bugs before gsplat is in the loop.
- **S7. Stereo in the headset with gsplat.** Exit: both eyes are correct and not swapped; the headset view matches the desktop render at the same pose; a 15-minute session runs with no session loss.
- **S8. Controller free-fly.** Exit: the interaction profile string is confirmed from the runtime; button input moves the world offset, not the view; a scripted test distance is logged; a 5-minute session at the target refresh runs with no dropped frames (threshold from D9).
- **S9. Repo.** Run `git init`. Add the `.gitignore` before the first commit. Add the fetch script with repo ID, revision, and SHA-256. Write the README with the version table, AI disclosure, license notes, and the registration undo. Exit: a fresh clone reproduces S7 in a clean venv.

### Pivot and stop rules

- **If S2 fails:** the problem is the runtime, not the renderer. Check SteamVR's Linux OpenXR support and the registration path. The OpenVR route is the fallback. Do not start S3 until S2 passes.
- **If S4 shows neither path fits the budget at the runtime size:** lower the render scale first, then reduce splat count or SH degree, then consider a non-CUDA renderer or a C++ host.
- **If S5 misses the budget at a useful scene size:** lower the SH degree or splat count before changing the renderer.
- **If D1 finds an existing viewer that meets version 1:** decide before S2 whether to continue as a learning exercise or stop.
- **Proposed stop rule, for the author to confirm:** if S4 and S5 cannot meet the budget at 2160 per eye with a scene the author accepts, write up the measurements and stop before S7.

### Still unverified (needs a check before reliance)

- SteamVR's OpenXR graphics bindings and swapchain formats under Wayland.
- Whether SteamVR reprojects for this stream, and what the encode-width setting controls.
- The Steam Frame's refresh rates beyond 90 and 120 Hz, its controller interaction profile, and its browser's WebXR support.
- pyopenxr's maturity and which OpenXR loader it loads.
- gsplat's license text, API, camera convention, and build requirements.
- Wheel availability for torch, gsplat, and pyopenxr on Python 3.14 and on any other candidate Python.
- The CUDA version the driver supports (`nvidia-smi` not run).
- The Hugging Face repos, revisions, licenses, and file sizes for "train" and "table."
- GitHub's per-file size limit.
- The existence and XR support of the open-source viewers named above.
- The stream codec, the Wi-Fi band, and the dongle's end of the link.
- The cause of the headset drops in the logs.

**S5 decision (author):** accept a lower frame rate and continue. Measured later in the headset: about 30 frames per second (see S7).

**S6 result (points and axes preview): replaced by S7 after two bugs were found.**
- Bug 1, every headset stage since S3: OpenGL swapchain images have their origin at the bottom-left, and all renderers wrote the top row first. The headset showed every frame upside down. The S3 solid colours and the S4 vertical bar look the same either way, so it went unnoticed. In S6 it showed as a scene that sheared and would not stay fixed when the head moved (author report). Fix: rows are flipped on the way into the swapchain (`viewer.to_swapchain_rgba`, with a test). Evidence: the S4 readback found the counter, written to buffer row 0, at GL row 0.
- Bug 2: the point preview wrote points with an unordered scatter, so which point won a shared pixel was undefined (12,145 pixels differed between two identical draws). Fixed with a depth buffer and a test.
- Placement: the preview put a shrunken scene 3 m away (author: "very far away"). Replaced by the S7 start pose.

**S7 result (gsplat in the headset): passed (author report and logs).**
- Start pose: training photo 00002's camera position, facing the same way, horizon level (`scene_frame.py`). Up is the average up of all 301 training cameras. Scale is 2.21 m per scene unit, from an assumed camera height of 1.6 m and a ground height estimated from the sparse points. This is an estimate, not a measurement. Tests cover the composition of the head pose with the scene frame.
- Author check after the row fix: sky up, scene stays fixed when the head moves.
- Splats are sorted by distance from the eye instead of view depth (`xr_render.py`, gsplat's pipeline with the sort key swapped). With depth as the key it matches gsplat exactly (test). Rotation consistency, measured as an 8-degree turn warped back over the overlap: 23.0 dB with depth, 27.1 dB with distance, at no speed cost. Author: popping is better.
- Splats closer than 0.75 m to the eye are not drawn (distance-based). The remaining near-field smear is real but poorly reconstructed geometry: the ground and a locomotive about 2 m to the right of the start pose. A larger cut deletes that locomotive. Culling splats near the training cameras made the ground worse.
- Frame rate: about 30 frames per second. A stereo frame at 2160 per eye takes 23 to 26 ms including the write, and frame pacing rounds that to every third 90 Hz slot.
- Scene quality: only the train is sharp. The background is coarse in the trained scene itself (author: "fine for testing").

**S8 result (controller free-fly): passed (author report and logs).**
- Left stick flies where the head looks. Right stick snap-turns 30 degrees and moves up and down. Either grip gives 3x speed (1.5 m/s base). A returns to the start pose (`locomotion.py`, with tests).
- Controller profile: the Steam Frame profile `/interaction_profiles/valve/frame_controller` is only accepted with the `XR_VALVE_frame_controller_interaction` extension enabled. In the S8 run it was not enabled, and SteamVR served the controllers through `/interaction_profiles/oculus/touch_controller`. The extension is now enabled, and the Frame bindings are accepted (checked). Which profile SteamVR picks with it on has not been checked in the headset yet.
- Run: 4 minutes, no errors, 42 distinct positions logged, reset used twice.
