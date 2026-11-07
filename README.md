# steamframe-splats

A pipecleaner project: look at a standard trained Gaussian splat scene in a Valve Steam Frame, with the scene rendered on a PC and streamed to the headset through SteamVR.

## Status

Early. The first hardware test passed: SteamVR's OpenXR sample (`helloxr`) runs on the Steam Frame. The splat viewer is not built yet.

## Scope (version 1)

- One standard trained splat scene, rendered on the PC.
- Head-tracked look.
- Controller free-fly.
- No code runs on the headset.

Out of scope: capturing or editing splats, multiplayer, and on-headset builds.

## Planning

[`concepts_brief.md`](concepts_brief.md) holds the planning brief: the risks, the staged plan, and the decisions taken so far.

## AI disclosure

This project was built with AI assistance (Claude Code, Anthropic). The author directs the work, runs the tests on their own hardware, and makes the decisions. The AI is not listed as an author.

## License

Apache License 2.0. See [LICENSE](LICENSE).
