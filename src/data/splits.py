"""Train / val / test split generation and I/O for Flickr30k."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_caption_map(captions_path: str | Path) -> Dict[str, List[str]]:
    """Load a Flickr30k captions file into {image_id: [caption, ...]}."""
    captions_path = Path(captions_path)
    caption_map: Dict[str, List[str]] = defaultdict(list)

    with captions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", maxsplit=2)
            if len(parts) < 2:
                continue
            image_id = parts[0].strip().strip('"')
            # Skip header row
            if image_id == "image_name":
                continue
            if len(parts) == 3:
                caption = parts[2].strip().rstrip(",").strip('"')
            else:
                # Malformed line: try to extract caption after the comment number
                remainder = parts[1].strip().rstrip(",")
                # Strip leading digits (comment number) and whitespace
                caption = remainder.lstrip("0123456789").strip().strip('"')
            if not image_id or not caption:
                continue
            caption_map[image_id].append(caption)

    return dict(caption_map)


def load_split_file(split_path: str | Path) -> List[str]:
    """Read a split text file (one image id per line)."""
    split_path = Path(split_path)
    with split_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def build_image_splits(
    image_ids: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Deterministically split image ids into train / val / test."""
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    ids = sorted(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    train_end = int(round(n * train_ratio))
    val_end = train_end + int(round(n * val_ratio))

    return {
        "train": ids[:train_end],
        "val": ids[train_end:val_end],
        "test": ids[val_end:],
    }


def save_split_files(splits: Dict[str, List[str]], output_dir: str | Path) -> None:
    """Write each split to a text file (one image id per line)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_ids in splits.items():
        path = output_dir / f"{split_name}.txt"
        with path.open("w", encoding="utf-8") as handle:
            for image_id in split_ids:
                handle.write(f"{image_id}\n")
