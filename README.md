# steamframe-splats

A pipecleaner project: look at a standard trained Gaussian splat scene in a Valve Steam Frame, with the scene rendered on a PC and streamed to the headset through SteamVR.

## Status

Early. SteamVR's OpenXR sample (`helloxr`) runs on the Steam Frame. A test app that shows a solid color in each eye also runs on the headset. The splat viewer is not built yet.

## Scope (version 1)

- One standard trained splat scene, rendered on the PC.
- Head-tracked look.
- Controller free-fly.
- No code runs on the headset.

Out of scope: capturing or editing splats, multiplayer, and on-headset builds.

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
