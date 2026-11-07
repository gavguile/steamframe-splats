# steamframe-splats

A pipecleaner project: look at a standard trained Gaussian splat scene in a Valve Steam Frame, with the scene rendered on a PC and streamed to the headset through SteamVR.

## Status

Working pipecleaner. The train scene renders with gsplat on this PC, streams through SteamVR to the Steam Frame, and you can fly through it with the controllers. It renders at 1322 x 1322 per eye by default on an RTX 3070 Ti (see the brief for frame rates). The scene is sharp on the train and coarse elsewhere, which is how it was trained.

## Scope (version 1)

- One standard trained splat scene, rendered on the PC.
- Head-tracked look.
- Controller free-fly.
- No code runs on the headset.

Out of scope: capturing or editing splats, multiplayer, and on-headset builds.

## Run the viewer

Needs Linux, an NVIDIA GPU with the CUDA toolkit (`nvcc`; gsplat compiles its kernels on first use), SteamVR with the Steam Frame active, and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
scripts/fetch_scene.sh        # the trained train scene (133 MB)
scripts/fetch_reference.sh    # camera poses (used for the start pose) and test photos
uv run python src/viewer.py --seconds 300
```

Controls:

| Input | Action |
| --- | --- |
| Left stick | Fly where you look (forward/back, sideways) |
| Right stick left/right | Snap turn 30 degrees |
| Right stick up/down | Move straight up or down |
| Either grip (hold) | 3x speed |
| A (right controller) | Back to the start position |

Options: `--size` (render size per eye, default 1322; 2160 is sharper but runs at about 30 frames per second), `--near-m` (hide splats closer than this, default 0.75 m), `--snapshot-every N` (save the left-eye frame to `data/snapshots/`).

You start where one of the training photos was taken, at roughly life size. The scale is estimated from the camera height (2.2 m per scene unit), not measured.

## Run the test pattern

Needs SteamVR installed, the Steam Frame active in SteamVR, and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run python src/xr_test_pattern.py --seconds 60
```

Start SteamVR first and wait until the headset shows as active. If the app starts while SteamVR is cold, it fails with `XR_ERROR_RUNTIME_FAILURE`. The app points the OpenXR loader at SteamVR for its own process only, so no system-wide OpenXR setting changes.

## Scene data

The scene files are not in this repo. Fetch the train scene with:

```sh
scripts/fetch_scene.sh
```

It downloads one pinned file into `data/` and checks its size and SHA-256. The source repo declares no license, and the scene comes from Tanks and Temples. Check those terms before any public use.

## Planning

[`concepts_brief.md`](concepts_brief.md) holds the planning brief: the risks, the staged plan, and the decisions taken so far.

## AI disclosure

This project was built with AI assistance (Claude Code, Anthropic). The author directs the work, runs the tests on their own hardware, and makes the decisions. The AI is not listed as an author.

## License

Apache License 2.0. See [LICENSE](LICENSE).
