#!/usr/bin/env bash
# Fetch the reference data for S5: COLMAP poses and training photos for the train scene,
# and the lower-detail LOD levels of the same scene (for the timing and quality sweep).
# Everything is pinned to one revision and verified by size and SHA-256.
# None of these files are committed (data/ is gitignored).
#
# Sources:
#   nvs-bench/tanksandtemples (dataset, card license MIT; Tanks and Temples source terms not checked)
#   laphisboy/3DGS-FLoD_pretrained_models (model repo, card declares no license)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/tnt-train"
NVS_REV="f5c9529539ab3e93dc33362f84034c6704bd7256"
FLOD_REV="5fb96747023120cff1dcbdf21c2ae6f2ae7b2182"

# dest | url | expected bytes | expected sha256
ENTRIES=(
  "colmap/cameras.bin|https://huggingface.co/datasets/nvs-bench/tanksandtemples/resolve/${NVS_REV}/train/sparse/0/cameras.bin|64|25d6f6bc51d126c680a5f5598bae17722d9b29030eb0b9b0b242406f508b9cc0"
  "colmap/images.bin|https://huggingface.co/datasets/nvs-bench/tanksandtemples/resolve/${NVS_REV}/train/sparse/0/images.bin|73111698|14e48a5bd9afc87ce7f46fbf093852a4ff2185b4b24d79bdcd3ad58db7a101e0"
  "colmap/points3D.bin|https://huggingface.co/datasets/nvs-bench/tanksandtemples/resolve/${NVS_REV}/train/sparse/0/points3D.bin|19486218|e54fc1d534ed7cc248442dbb567de1094ecca5c28c61a8c96986cc95766d706f"
  "gt/00002.jpg|https://huggingface.co/datasets/nvs-bench/tanksandtemples/resolve/${NVS_REV}/train/images/00002.jpg|383531|f243a2de5a551a8aaba07cfff321857af2ab7b4b95c58e5c3eae8913cee639c5"
  "gt/00004.jpg|https://huggingface.co/datasets/nvs-bench/tanksandtemples/resolve/${NVS_REV}/train/images/00004.jpg|416407|7fd3beb07312c7907f1f09f83382730d6262a819c5010503ee3e4cd7db3b615d"
  "gt/00006.jpg|https://huggingface.co/datasets/nvs-bench/tanksandtemples/resolve/${NVS_REV}/train/images/00006.jpg|424664|692a9db46ae89895b722fd3f5818a9844393f16c47fb0c3fa63f292f725d594b"
  "gt/00009.jpg|https://huggingface.co/datasets/nvs-bench/tanksandtemples/resolve/${NVS_REV}/train/images/00009.jpg|422816|f10a73567d35d884218f5fcd62923d072b57e843e97fe54719cee8544e4252e6"
  "lods/lod2.ply|https://huggingface.co/laphisboy/3DGS-FLoD_pretrained_models/resolve/${FLOD_REV}/tnt/train/point_cloud/lod_2_iteration_15000/point_cloud.ply|6710426|ba237e0a38df0a1cf6d06789e6bc9686b3a6efb5419eb82cc0b096f59a56648a"
  "lods/lod3.ply|https://huggingface.co/laphisboy/3DGS-FLoD_pretrained_models/resolve/${FLOD_REV}/tnt/train/point_cloud/lod_3_iteration_20000/point_cloud.ply|32723643|df602aacfaa4f519bb4dcd45d3b495a6060db3f0dc2217d035d090d24554115a"
  "lods/lod4.ply|https://huggingface.co/laphisboy/3DGS-FLoD_pretrained_models/resolve/${FLOD_REV}/tnt/train/point_cloud/lod_4_iteration_25000/point_cloud.ply|55572875|99bfc7b2ae346d3087b78143852d5d41d5d1201ba4545d09f8d7faa0d36d74a6"
)

verify() {
  local path="$1" bytes="$2" sha="$3"
  [ -f "$path" ] \
    && [ "$(stat -c %s "$path")" -eq "$bytes" ] \
    && [ "$(sha256sum "$path" | cut -d' ' -f1)" = "$sha" ]
}

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r rel url bytes sha <<< "$entry"
  dest="$DATA/$rel"
  if verify "$dest" "$bytes" "$sha"; then
    echo "ok (already verified): $rel"
    continue
  fi
  mkdir -p "$(dirname "$dest")"
  tmp="$(mktemp "$(dirname "$dest")/.download.XXXXXX")"
  curl -fL --retry 3 -o "$tmp" "$url"
  if verify "$tmp" "$bytes" "$sha"; then
    mv "$tmp" "$dest"
    echo "downloaded and verified: $rel"
  else
    rm -f "$tmp"
    echo "verification failed for $rel (size or SHA-256 mismatch)" >&2
    exit 1
  fi
done
echo "all reference files verified"
