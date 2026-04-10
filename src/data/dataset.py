"""PyTorch Dataset and DataLoader for Flickr30k captioning."""

from __future__ import annotations

import functools
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.data.constants import DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_STD, END_TOKEN, PAD_TOKEN
from src.data.splits import load_caption_map
from src.data.vocab import Vocabulary


def _build_transform(image_size: int | Tuple[int, int], mean: Tuple[float, float, float], std: Tuple[float, float, float]):
    import torchvision.transforms as transforms

    if isinstance(image_size, int):
        size_hw = (image_size, image_size)
    else:
        size_hw = (int(image_size[0]), int(image_size[1]))
    return transforms.Compose(
        [
            transforms.Resize(size_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


class FlickrCaptionDataset(Dataset):
    """One row per (image, caption) pair; training samples a random caption per image each epoch."""

    def __init__(
        self,
        captions_path: Path,
        images_dir: Path,
        split_image_ids: Sequence[str],
        vocab: Vocabulary,
        max_caption_length: int,
        image_size: int | Tuple[int, int],
        image_mean: Tuple[float, float, float],
        image_std: Tuple[float, float, float],
        random_caption: bool,
        seed: int,
    ) -> None:
        self.caption_map = load_caption_map(captions_path)
        self.images_dir = images_dir
        self.image_ids = list(split_image_ids)
        self.vocab = vocab
        self.max_caption_length = max_caption_length
        self.transform = _build_transform(image_size, image_mean, image_std)
        self.random_caption = random_caption
        self._rng = random.Random(seed)

        missing = [i for i in self.image_ids if i not in self.caption_map]
        if missing:
            raise ValueError(f"{len(missing)} split images missing from caption map (e.g. {missing[:3]})")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_id = self.image_ids[index]
        captions = self.caption_map[image_id]
        if self.random_caption:
            caption_text = self._rng.choice(captions)
        else:
            caption_text = captions[0]

        image_path = self.images_dir / image_id
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        tensor_image = self.transform(image)

        encoded = self.vocab.encode(caption_text, self.max_caption_length)
        if len(encoded) > self.max_caption_length:
            encoded = encoded[: self.max_caption_length]
            encoded[-1] = self.vocab.word2idx[END_TOKEN]

        pad_id = self.vocab.word2idx[PAD_TOKEN]
        caption_ids = torch.tensor(encoded, dtype=torch.long)
        target_ids = torch.empty_like(caption_ids)
        if caption_ids.numel() >= 2:
            target_ids[:-1] = caption_ids[1:]
            target_ids[-1] = pad_id
        else:
            target_ids.fill_(pad_id)

        length = int(caption_ids.numel())
        padding_mask = torch.zeros(length, dtype=torch.bool)
        return {
            "image_id": image_id,
            "image": tensor_image,
            "caption_text": caption_text,
            "caption_ids": caption_ids,
            "target_ids": target_ids,
            "length": length,
            "padding_mask": padding_mask,
        }


def _collate_batch(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    image_ids = [item["image_id"] for item in batch]
    caption_texts = [item["caption_text"] for item in batch]

    caption_ids = pad_sequence([item["caption_ids"] for item in batch], batch_first=True, padding_value=pad_id)
    target_ids = pad_sequence([item["target_ids"] for item in batch], batch_first=True, padding_value=pad_id)

    max_len = caption_ids.size(1)
    padding_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        lengths[i] = item["length"]
        padding_mask[i, : item["length"]] = True

    return {
        "image_ids": image_ids,
        "images": images,
        "caption_texts": caption_texts,
        "caption_ids": caption_ids,
        "target_ids": target_ids,
        "lengths": lengths,
        "padding_mask": padding_mask,
    }


def create_caption_dataloader(
    captions_path: str | Path,
    images_dir: str | Path,
    split_image_ids: Sequence[str],
    vocab: Vocabulary,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_caption_length: int,
    image_size: int | Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    image_mean: Tuple[float, float, float] = DEFAULT_IMAGE_MEAN,
    image_std: Tuple[float, float, float] = DEFAULT_IMAGE_STD,
    random_caption: bool | None = None,
    seed: int = 42,
) -> DataLoader:
    if random_caption is None:
        random_caption = shuffle

    pad_id = vocab.word2idx[PAD_TOKEN]
    dataset = FlickrCaptionDataset(
        captions_path=Path(captions_path),
        images_dir=Path(images_dir),
        split_image_ids=split_image_ids,
        vocab=vocab,
        max_caption_length=max_caption_length,
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
        random_caption=random_caption,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=functools.partial(_collate_batch, pad_id=pad_id),
    )
