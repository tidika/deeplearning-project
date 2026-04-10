#!/usr/bin/env bash
# Pack Python project files for GPU server (excludes datasets and large artifacts).
# Uses `zip` if available; otherwise Python (scripts/make_zip.py).
# Usage: bash scripts/package_code_for_server.sh [output.zip]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-${ROOT}/deeplearning-project_code_for_server.zip}"
cd "$ROOT"
rm -f "$OUT"

if command -v zip >/dev/null 2>&1; then
  zip -r "$OUT" \
    src/ \
    scripts/ \
    configs/ \
    tests/ \
    metadata/splits/ \
    metadata/vocab/ \
    docs/ \
    .github/ \
    requirements.txt \
    README.md \
    -x "*__pycache__*" -x "*.pyc" -x "*.DS_Store"
else
  echo "zip not found; using Python (scripts/make_zip.py)..." >&2
  python3 "$ROOT/scripts/make_zip.py" "$OUT" \
    src \
    scripts \
    configs \
    tests \
    metadata/splits \
    metadata/vocab \
    docs \
    .github \
    requirements.txt \
    README.md
fi
echo ""
echo "Created: $OUT"
echo "NOT included (copy to server separately): archive/ or data/ images, archive/captions.txt"
echo "On server: unzip, place Flickr30k under archive/ as before, pip install -r requirements.txt, then train."
