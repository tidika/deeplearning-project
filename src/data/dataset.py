"""Flickr30k caption dataset and DataLoader factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.data.constants import DEFAULT_MAX_CAPTION_LENGTH, END_TOKEN, PAD_TOKEN, START_TOKEN
from src.data.preprocess import tokenize_caption, truncate_tokens
from src.data.splits import load_caption_map
from src.data.vocab import Vocabulary

DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_IMAGE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class CaptionDataset(Dataset):
    """One sample per (image, caption) pair."""

    def __init__(
        self,
        caption_map: Dict[str, List[str]],
        images_dir: Path,
        split_image_ids: List[str],
        vocab: Vocabulary,
        max_caption_length: int = DEFAULT_MAX_CAPTION_LENGTH,
        transform: Any = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.max_caption_length = max_caption_length
        self.transform = transform

        self.samples: List[Tuple[str, str]] = []
        for image_id in split_image_ids:
            for caption in caption_map.get(image_id, []):
                self.samples.append((image_id, caption))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_id, caption_text = self.samples[index]
        image_path = self.images_dir / image_id
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        tokens = tokenize_caption(caption_text)
        tokens = truncate_tokens(tokens, self.max_caption_length)

        start_id = self.vocab.word2idx[START_TOKEN]
        end_id = self.vocab.word2idx[END_TOKEN]
        token_ids = self.vocab.encode(tokens)

        # caption_ids: <start> tokens <end>
        # target_ids:  tokens <end> <pad>  (shifted by one)
        caption_ids = [start_id] + token_ids + [end_id]
        target_ids = token_ids + [end_id] + [end_id]
        # Keep them the same length
        target_ids = target_ids[: len(caption_ids)]
        length = len(caption_ids)

        return {
            "image_id": image_id,
            "image_path": str(image_path),
            "image": image,
            "caption_text": caption_text,
            "caption_ids": caption_ids,
            "target_ids": target_ids,
            "length": length,
        }


def _collate_captions(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length captions into a single batch tensor."""
    pad_id = batch[0]["caption_ids"][0]  # won't be used; we grab from vocab below
    # We need the pad token id — it's always 0 by convention.
    pad_id = 0

    max_len = max(sample["length"] for sample in batch)
    batch_size = len(batch)

    images = torch.stack([sample["image"] for sample in batch])
    caption_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    target_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    image_ids: List[str] = []
    caption_texts: List[str] = []

    for i, sample in enumerate(batch):
        length = sample["length"]
        caption_ids[i, :length] = torch.tensor(sample["caption_ids"], dtype=torch.long)
        target_ids[i, :length] = torch.tensor(sample["target_ids"], dtype=torch.long)
        padding_mask[i, :length] = 1
        lengths[i] = length
        image_ids.append(sample["image_id"])
        caption_texts.append(sample["caption_text"])

    return {
        "images": images,
        "image_ids": image_ids,
        "caption_texts": caption_texts,
        "caption_ids": caption_ids,
        "target_ids": target_ids,
        "padding_mask": padding_mask,
        "lengths": lengths,
    }


def create_caption_dataloader(
    captions_path: str | Path,
    images_dir: str | Path,
    split_image_ids: List[str],
    vocab: Vocabulary,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    max_caption_length: int = DEFAULT_MAX_CAPTION_LENGTH,
    image_size: int | Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    image_mean: Tuple[float, float, float] = DEFAULT_IMAGE_MEAN,
    image_std: Tuple[float, float, float] = DEFAULT_IMAGE_STD,
) -> DataLoader:
    """Build a DataLoader for a given split."""
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=image_mean, std=image_std),
    ])

    caption_map = load_caption_map(captions_path)

    dataset = CaptionDataset(
        caption_map=caption_map,
        images_dir=Path(images_dir),
        split_image_ids=split_image_ids,
        vocab=vocab,
        max_caption_length=max_caption_length,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_captions,
    )
