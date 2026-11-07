#!/usr/bin/env bash
# Download the "train" scene: a 3D Gaussian Splatting model of the Tanks and
# Temples train scene, LOD 5 (full detail), 534,812 Gaussians, SH degree 3.
#
# Source: Hugging Face model repo laphisboy/3DGS-FLoD_pretrained_models
# (the card declares no license; the scene comes from Tanks and Temples).
# The file is pinned to one revision and checked by size and SHA-256.
# Scene files are never committed to this repo (see .gitignore).
set -euo pipefail

REPO="laphisboy/3DGS-FLoD_pretrained_models"
REVISION="5fb96747023120cff1dcbdf21c2ae6f2ae7b2182"
IN_REPO="tnt/train/point_cloud/lod_5_iteration_30000/point_cloud.ply"
EXPECTED_BYTES=132634907
EXPECTED_SHA256="507cc39c5bde73766d3173aa7065f53aec3487d20d9748327d58b72a118fa80f"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/tnt-train/point_cloud.ply"
URL="https://huggingface.co/${REPO}/resolve/${REVISION}/${IN_REPO}"

check() {
  [ -f "$1" ] \
    && [ "$(stat -c %s "$1")" -eq "$EXPECTED_BYTES" ] \
    && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$EXPECTED_SHA256" ]
}

if check "$DEST"; then
  echo "already present and verified: $DEST"
  exit 0
fi

mkdir -p "$(dirname "$DEST")"
TMP="$(mktemp "$(dirname "$DEST")/.download.XXXXXX")"
trap 'rm -f "$TMP"' EXIT

echo "downloading $URL"
curl -fL --retry 3 -o "$TMP" "$URL"

if ! check "$TMP"; then
  echo "verification failed: size or SHA-256 does not match the pinned values" >&2
  exit 1
fi

mv "$TMP" "$DEST"
trap - EXIT
echo "verified and saved: $DEST"
