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
from src.project_config import DEFAULT_CONFIG_PATH, get_config_value, load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the shared data config JSON.")
    parser.add_argument("--captions-path", default=None, help="Path to Flickr30k image-caption CSV file.")
    parser.add_argument("--output-dir", default=None, help="Directory where train/val/test split files will be written.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for split generation.")
    parser.add_argument("--train-ratio", type=float, default=None, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test split ratio.")
    parser.add_argument("--summary-path", default=None, help="Optional JSON path for split statistics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(REPO_ROOT / args.config)

    captions_path = args.captions_path or get_config_value(config, "captions_path", "data/captions.txt")
    output_dir = args.output_dir or get_config_value(config, "splits_dir", "metadata/splits")
    seed = args.seed if args.seed is not None else int(get_config_value(config, "seed", 42))
    train_ratio = args.train_ratio if args.train_ratio is not None else float(get_config_value(config, "train_ratio", 0.8))
    val_ratio = args.val_ratio if args.val_ratio is not None else float(get_config_value(config, "val_ratio", 0.1))
    test_ratio = args.test_ratio if args.test_ratio is not None else float(get_config_value(config, "test_ratio", 0.1))
    summary_path = args.summary_path or str(Path(output_dir) / "split_summary.json")

    caption_map = load_caption_map(REPO_ROOT / captions_path)
    image_ids = sorted(caption_map.keys())

    splits = build_image_splits(
        image_ids=image_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    save_split_files(splits, REPO_ROOT / output_dir)

    caption_counts = [len(captions) for captions in caption_map.values()]
    unique_caption_counts = sorted(set(caption_counts))
    summary = {
        "config_path": args.config,
        "captions_path": captions_path,
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
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

    resolved_summary_path = REPO_ROOT / summary_path
    resolved_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print(f"Saved splits to {Path(output_dir)}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
