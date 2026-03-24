"""Generate deterministic image-level train/val/test splits for Flickr30k."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import build_image_splits, load_caption_map, save_split_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--captions-path",
        default="data/captions.txt",
        help="Path to Flickr30k image-caption CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="metadata/splits",
        help="Directory where train/val/test split files will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument(
        "--summary-path",
        default="metadata/splits/split_summary.json",
        help="Optional JSON path for split statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    caption_map = load_caption_map(REPO_ROOT / args.captions_path)
    image_ids = sorted(caption_map.keys())

    splits = build_image_splits(
        image_ids=image_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    save_split_files(splits, REPO_ROOT / args.output_dir)

    caption_counts = [len(captions) for captions in caption_map.values()]
    unique_caption_counts = sorted(set(caption_counts))
    summary = {
        "captions_path": args.captions_path,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "num_images": len(image_ids),
        "num_captions": sum(caption_counts),
        "captions_per_image": {
            "unique_values": unique_caption_counts,
            "min": min(caption_counts),
            "max": max(caption_counts),
        },
        "split_sizes": {split_name: len(split_ids) for split_name, split_ids in splits.items()},
    }

    summary_path = REPO_ROOT / args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(f"Saved splits to {Path(args.output_dir)}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
