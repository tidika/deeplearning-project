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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--captions-path",
        default="data/captions.txt",
        help="Path to Flickr30k image-caption CSV file.",
    )
    parser.add_argument(
        "--train-split",
        default="metadata/splits/train.txt",
        help="Path to the training split file used for vocabulary construction.",
    )
    parser.add_argument(
        "--output-path",
        default="metadata/vocab/flickr30k_vocab.json",
        help="Path where the vocabulary JSON should be written.",
    )
    parser.add_argument(
        "--stats-path",
        default="metadata/vocab/flickr30k_vocab_stats.json",
        help="Path where vocabulary statistics should be written.",
    )
    parser.add_argument(
        "--min-word-freq",
        type=int,
        default=5,
        help="Minimum token frequency required to enter the vocabulary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    caption_map = load_caption_map(REPO_ROOT / args.captions_path)
    train_image_ids = load_split_file(REPO_ROOT / args.train_split)

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

    vocab = Vocabulary.build(train_captions, min_word_freq=args.min_word_freq)
    output_path = REPO_ROOT / args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(output_path)

    stats = {
        "captions_path": args.captions_path,
        "train_split": args.train_split,
        "min_word_freq": args.min_word_freq,
        "num_train_images": len(train_image_ids),
        "num_train_captions": len(train_captions),
        "vocab_size": len(vocab.word2idx),
        "special_tokens": [token for token, idx in sorted(vocab.word2idx.items(), key=lambda item: item[1])[:4]],
    }

    stats_path = REPO_ROOT / args.stats_path
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=True)

    print(f"Saved vocabulary to {Path(args.output_path)}")
    print(json.dumps(stats, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
