"""Build a shared Flickr30k vocabulary from the training split only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import load_caption_map, load_split_file
from src.data.vocab import Vocabulary
from src.project_config import DEFAULT_CONFIG_PATH, get_config_value, load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the shared data config JSON.")
    parser.add_argument("--captions-path", default=None, help="Path to Flickr30k image-caption CSV file.")
    parser.add_argument("--train-split", default=None, help="Path to the training split file used for vocabulary construction.")
    parser.add_argument("--output-path", default=None, help="Path where the vocabulary JSON should be written.")
    parser.add_argument("--stats-path", default=None, help="Path where vocabulary statistics should be written.")
    parser.add_argument("--min-word-freq", type=int, default=None, help="Minimum token frequency required to enter the vocabulary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(REPO_ROOT / args.config)

    captions_path = args.captions_path or get_config_value(config, "captions_path", "data/captions.txt")
    splits_dir = Path(get_config_value(config, "splits_dir", "metadata/splits"))
    vocab_dir = Path(get_config_value(config, "vocab_dir", "metadata/vocab"))
    train_split = args.train_split or str(splits_dir / "train.txt")
    output_path = args.output_path or str(vocab_dir / "flickr30k_vocab.json")
    stats_path = args.stats_path or str(vocab_dir / "flickr30k_vocab_stats.json")
    min_word_freq = args.min_word_freq if args.min_word_freq is not None else int(get_config_value(config, "min_word_freq", 5))
    max_caption_length = int(get_config_value(config, "max_caption_length", 40))

    caption_map = load_caption_map(REPO_ROOT / captions_path)
    train_image_ids = load_split_file(REPO_ROOT / train_split)

    train_captions = []
    missing_images = []
    for image_id in train_image_ids:
        captions = caption_map.get(image_id)
        if not captions:
            missing_images.append(image_id)
            continue
        train_captions.extend(captions)

    if missing_images:
        raise ValueError(
            f"Found {len(missing_images)} training images missing captions. "
            f"First few: {missing_images[:5]}"
        )

    vocab = Vocabulary.build(train_captions, min_word_freq=min_word_freq)
    resolved_output_path = REPO_ROOT / output_path
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(resolved_output_path)

    stats = {
        "config_path": args.config,
        "captions_path": captions_path,
        "train_split": train_split,
        "min_word_freq": min_word_freq,
        "max_caption_length": max_caption_length,
        "num_train_images": len(train_image_ids),
        "num_train_captions": len(train_captions),
        "vocab_size": len(vocab.word2idx),
        "special_tokens": [token for token, idx in sorted(vocab.word2idx.items(), key=lambda item: item[1])[:4]],
    }

    resolved_stats_path = REPO_ROOT / stats_path
    resolved_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=True)

    print(f"Saved vocabulary to {Path(output_path)}")
    print(json.dumps(stats, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
