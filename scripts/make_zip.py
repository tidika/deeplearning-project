#!/usr/bin/env python3
"""Create a zip from paths (files or dirs) relative to cwd. Skips __pycache__, *.pyc, .DS_Store."""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def _skip(path: Path) -> bool:
    if "__pycache__" in path.parts:
        return True
    if path.suffix == ".pyc":
        return True
    if path.name == ".DS_Store":
        return True
    return False


def _add(zf: zipfile.ZipFile, path: Path, root: Path) -> None:
    if not path.exists():
        print(f"warning: missing {path}", file=sys.stderr)
        return
    if path.is_file():
        if not _skip(path):
            zf.write(path, path.relative_to(root).as_posix())
        return
    for f in path.rglob("*"):
        if f.is_file() and not _skip(f):
            zf.write(f, f.relative_to(root).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Output .zip path")
    parser.add_argument("paths", nargs="+", help="Files or directories to include")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for raw in args.paths:
            p = Path(raw)
            if not p.is_absolute():
                p = root / p
            p = p.resolve()
            _add(zf, p, root)

    print(f"Created {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
