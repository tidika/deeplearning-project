"""Train/val/test splits and Flickr30k caption file loading."""

from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence


def _load_caption_map_csv(path: Path) -> Dict[str, List[str]]:
    """Kaggle-style CSV: columns image,caption (header optional). Handles quotes and commas in captions."""
    caption_map: MutableMapping[str, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(handle, dialect)
        rows = list(reader)
    if not rows:
        return {}

    header_lower = [cell.strip().lower() for cell in rows[0]]
    start = 0
    if "image" in header_lower and ("caption" in header_lower or "comment" in header_lower):
        start = 1
    image_col = 0
    caption_col = 1
    if start == 1:
        header_lower = [cell.strip().lower() for cell in rows[0]]
        if "image" in header_lower:
            image_col = header_lower.index("image")
        if "caption" in header_lower:
            caption_col = header_lower.index("caption")
        elif "comment" in header_lower:
            caption_col = header_lower.index("comment")

    for row in rows[start:]:
        if len(row) < 2:
            continue
        image_key = row[image_col].split("#", 1)[0].strip()
        caption = (row[caption_col] or "").strip()
        if not image_key:
            continue
        if caption:
            caption_map[image_key].append(caption)
    return {key: list(values) for key, values in caption_map.items()}


def load_caption_map(captions_path: str | Path) -> Dict[str, List[str]]:
    """Load captions: tab `image#n<TAB>caption`, comma CSV (Kaggle), or `image#n,caption` per line."""
    path = Path(captions_path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv") or "results" in path.name.lower():
        return _load_caption_map_csv(path)

    caption_map: MutableMapping[str, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline()
        handle.seek(0)
        if first_line and "image" in first_line.lower() and ("caption" in first_line.lower() or first_line.strip().startswith("image,")):
            return _load_caption_map_csv(path)

        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                left, caption = line.split("\t", 1)
            else:
                match = re.match(r"^([^,]+),([\s\S]*)$", line)
                if not match:
                    raise ValueError(f"Unrecognized caption line format: {line[:80]!r}")
                left, caption = match.group(1), match.group(2)
            image_key = left.split("#", 1)[0].strip()
            cap = caption.strip()
            if cap:
                caption_map[image_key].append(cap)
    return {key: list(values) for key, values in caption_map.items()}


def load_split_file(split_path: str | Path) -> List[str]:
    path = Path(split_path)
    image_ids: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if name:
                image_ids.append(name)
    return image_ids


def build_image_splits(
    image_ids: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 0,
) -> Dict[str, List[str]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative")

    ids = list(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    n_test = n - n_train - n_val
    if n_test < 0:
        raise ValueError("Invalid split sizes")

    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def save_split_files(splits: Mapping[str, Sequence[str]], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        if name not in splits:
            raise KeyError(f"Missing split {name}")
        path = out / f"{name}.txt"
        with path.open("w", encoding="utf-8") as handle:
            for image_id in splits[name]:
                handle.write(f"{image_id}\n")
