"""Build references JSON (image id -> list of reference captions) for evaluate_captions.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import load_caption_map, load_split_file
from src.project_config import get_config_value, load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/data_archive.example.json",
        help="Shared data config (captions_path, splits_dir).",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val", help="Which split file to use.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: artifacts/transformer_caption/references_<split>.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_project_config(REPO_ROOT / args.config)

    captions_path = REPO_ROOT / get_config_value(cfg, "captions_path", "data/captions.txt")
    splits_dir = Path(get_config_value(cfg, "splits_dir", "metadata/splits"))
    split_path = REPO_ROOT / splits_dir / f"{args.split}.txt"

    caption_map = load_caption_map(captions_path)
    image_ids = load_split_file(split_path)

    references: dict[str, list[str]] = {}
    missing: list[str] = []
    for image_id in image_ids:
        caps = caption_map.get(image_id)
        if not caps:
            missing.append(image_id)
        else:
            references[image_id] = list(caps)

    if missing:
        raise ValueError(
            f"{len(missing)} images in split have no captions (e.g. {missing[:3]}). "
            "Check captions_path and split file."
        )

    out = args.output
    if out is None:
        out = REPO_ROOT / "artifacts" / "transformer_caption" / f"references_{args.split}.json"
    else:
        out = REPO_ROOT / out

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(references, handle, indent=2, ensure_ascii=True)

    print(f"Wrote {len(references)} images to {out.relative_to(REPO_ROOT)}")
    print(f"Example: first id has {len(next(iter(references.values())))} reference captions.")


if __name__ == "__main__":
    main()
