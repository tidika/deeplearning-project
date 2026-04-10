#!/usr/bin/env bash
# Pack training outputs for report / handoff (checkpoints, logs, samples).
# Uses `zip` if available; otherwise Python (no `zip` binary required).
# Usage: bash scripts/package_training_results.sh [output.zip]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-${ROOT}/transformer_training_results.zip}"
cd "$ROOT"
if [[ ! -d artifacts/transformer_caption ]]; then
  echo "No artifacts/transformer_caption — train first or adjust path." >&2
  exit 1
fi
rm -f "$OUT"

if command -v zip >/dev/null 2>&1; then
  zip -r "$OUT" \
    artifacts/transformer_caption/ \
    docs/transformer_model_design.md \
    configs/transformer_caption.example.json \
    configs/data_archive.example.json
  if [[ -f REPORT_NOTES.txt ]]; then
    zip "$OUT" REPORT_NOTES.txt
  fi
else
  echo "zip not found; using Python (scripts/make_zip.py)..." >&2
  if [[ -f REPORT_NOTES.txt ]]; then
    python3 "$ROOT/scripts/make_zip.py" "$OUT" \
      artifacts/transformer_caption \
      docs/transformer_model_design.md \
      configs/transformer_caption.example.json \
      configs/data_archive.example.json \
      REPORT_NOTES.txt
  else
    python3 "$ROOT/scripts/make_zip.py" "$OUT" \
      artifacts/transformer_caption \
      docs/transformer_model_design.md \
      configs/transformer_caption.example.json \
      configs/data_archive.example.json
  fi
fi

echo ""
echo "Created: $OUT"
echo "Optional: add REPORT_NOTES.txt at repo root and re-run this script."
