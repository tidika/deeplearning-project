#!/usr/bin/env bash
# Split archive/Images into several smaller zips for slow manual uploads.
# Usage: bash scripts/zip_images_in_chunks.sh [images_dir] [files_per_zip]
# Example: bash scripts/zip_images_in_chunks.sh archive/Images 4000
# Output: upload_chunks/archive_Images_part01.zip, ...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMG_DIR="${1:-${ROOT}/archive/Images}"
CHUNK="${2:-4000}"
OUT_DIR="${ROOT}/upload_chunks"

if [[ ! -d "$IMG_DIR" ]]; then
  echo "Not a directory: $IMG_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
cd "$ROOT"

mapfile -t FILES < <(find "$IMG_DIR" -maxdepth 1 -type f -name '*.jpg' | sort)
total=${#FILES[@]}
if [[ "$total" -eq 0 ]]; then
  echo "No .jpg under $IMG_DIR" >&2
  exit 1
fi

echo "Found $total jpg files. Chunk size: $CHUNK -> $OUT_DIR/"
part=1
count=0
batch_rel=()
for f in "${FILES[@]}"; do
  rel="${f#"$ROOT"/}"
  batch_rel+=("$rel")
  count=$((count + 1))
  if [[ "$count" -eq "$CHUNK" ]]; then
    out="${OUT_DIR}/archive_Images_part$(printf '%02d' "$part").zip"
    rm -f "$out"
    zip -q -r "$out" "${batch_rel[@]}"
    echo "  -> $(basename "$out") ($count files)"
    part=$((part + 1))
    count=0
    batch_rel=()
  fi
done

if [[ ${#batch_rel[@]} -gt 0 ]]; then
  out="${OUT_DIR}/archive_Images_part$(printf '%02d' "$part").zip"
  rm -f "$out"
  zip -q -r "$out" "${batch_rel[@]}"
  echo "  -> $(basename "$out") (${#batch_rel[@]} files)"
fi

echo ""
echo "Upload each part separately. Also upload archive/captions.txt (small)."
echo "On server: unzip all parts in project root (same folder as archive/) so paths stay archive/Images/*.jpg"
