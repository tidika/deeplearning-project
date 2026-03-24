"""Load a split and vocabulary, build a DataLoader, and print one batch summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.constants import DEFAULT_MAX_CAPTION_LENGTH
from src.data.dataset import (
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IMAGE_STD,
    create_caption_dataloader,
)
from src.data.splits import load_split_file
from src.data.vocab import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions-path", default="data/captions.txt", help="Path to the Flickr30k captions CSV.")
    parser.add_argument("--images-dir", default="data/Images", help="Directory containing the Flickr30k images.")
    parser.add_argument("--split-path", default="metadata/splits/train.txt", help="Path to a saved split file.")
    parser.add_argument("--vocab-path", default="metadata/vocab/flickr30k_vocab.json", help="Path to a saved vocabulary JSON.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size used for the verification DataLoader.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--max-caption-length", type=int, default=DEFAULT_MAX_CAPTION_LENGTH, help="Maximum encoded caption length.")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE[0], help="Square image size used before batching.")
    parser.add_argument("--image-mean", type=float, nargs=3, default=DEFAULT_IMAGE_MEAN, help="RGB channel means used for normalization.")
    parser.add_argument("--image-std", type=float, nargs=3, default=DEFAULT_IMAGE_STD, help="RGB channel standard deviations used for normalization.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the split before batching.")
    return parser.parse_args()


def tensor_shape(value: object) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def main() -> None:
    args = parse_args()

    split_image_ids = load_split_file(REPO_ROOT / args.split_path)
    vocab = Vocabulary.load(REPO_ROOT / args.vocab_path)
    dataloader = create_caption_dataloader(
        captions_path=REPO_ROOT / args.captions_path,
        images_dir=REPO_ROOT / args.images_dir,
        split_image_ids=split_image_ids,
        vocab=vocab,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        max_caption_length=args.max_caption_length,
        image_size=args.image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
    )

    dataset = dataloader.dataset
    batch = next(iter(dataloader))

    summary = {
        "split_path": args.split_path,
        "vocab_path": args.vocab_path,
        "num_split_images": len(split_image_ids),
        "dataset_size": len(dataset),
        "vocab_size": len(vocab.word2idx),
        "image_size": [args.image_size, args.image_size],
        "image_mean": [float(value) for value in args.image_mean],
        "image_std": [float(value) for value in args.image_std],
        "batch_size": len(batch["image_ids"]),
        "images_shape": tensor_shape(batch["images"]),
        "caption_ids_shape": tensor_shape(batch["caption_ids"]),
        "target_ids_shape": tensor_shape(batch["target_ids"]),
        "lengths_shape": tensor_shape(batch["lengths"]),
        "padding_mask_shape": tensor_shape(batch["padding_mask"]),
        "first_image_id": batch["image_ids"][0],
        "first_caption_text": batch["caption_texts"][0],
        "first_caption_ids_prefix": batch["caption_ids"][0][:10].tolist(),
        "first_target_ids_prefix": batch["target_ids"][0][:10].tolist(),
        "batch_image_min": float(batch["images"].min().item()),
        "batch_image_max": float(batch["images"].max().item()),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
