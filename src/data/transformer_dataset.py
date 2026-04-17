"""Dataset and DataLoader for pre-extracted Transformer spatial features.

Expects features pre-extracted by scripts/extract_transformer_features.py
and saved as:
    data/transformer_features/{image_id}.pt  →  shape (49, 2048)

The 49 spatial locations come from ResNet-50's final convolutional layer
(layer4), which outputs a 7×7 feature map with 2048 channels per location.

Only images whose feature file exists are included in the dataset, so
training can start even if extraction is partially complete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.constants import DEFAULT_MAX_CAPTION_LENGTH, END_TOKEN, START_TOKEN
from src.data.preprocess import tokenize_caption, truncate_tokens
from src.data.vocab import Vocabulary


class TransformerCaptionDataset(Dataset):
    """One sample per (image, caption) pair using pre-extracted ResNet features.

    Mirrors BUTDCaptionDataset but loads spatial grid features instead of
    region features, keeping the rest of the pipeline identical so that
    BUTD and Transformer models are evaluated on the same data.
    """

    def __init__(
        self,
        caption_map: Dict[str, List[str]],
        features_dir: str | Path,
        split_image_ids: List[str],
        vocab: Vocabulary,
        max_caption_length: int = DEFAULT_MAX_CAPTION_LENGTH,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.vocab = vocab
        self.max_caption_length = max_caption_length

        self.samples: List[tuple[str, str]] = []
        missing = 0
        for image_id in split_image_ids:
            feat_path = self.features_dir / f"{image_id}.pt"
            if not feat_path.exists():
                missing += 1
                continue
            for caption in caption_map.get(image_id, []):
                self.samples.append((image_id, caption))

        if missing:
            print(
                f"[TransformerCaptionDataset] {missing}/{len(split_image_ids)} images "
                "have no feature file and were skipped. Run "
                "scripts/extract_transformer_features.py to generate missing features."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_id, caption_text = self.samples[index]

        # Load pre-extracted spatial features: (49, 2048)
        feat_path = self.features_dir / f"{image_id}.pt"
        features = torch.load(feat_path, map_location="cpu", weights_only=True)

        tokens = tokenize_caption(caption_text)
        tokens = truncate_tokens(tokens, self.max_caption_length)

        start_id = self.vocab.word2idx[START_TOKEN]
        end_id = self.vocab.word2idx[END_TOKEN]
        token_ids = self.vocab.encode(tokens)

        # caption_ids: <start> w1 w2 ... wN <end>
        # target_ids:  w1 w2 ... wN <end> <end>   (shifted by 1, same length)
        caption_ids = [start_id] + token_ids + [end_id]
        target_ids = token_ids + [end_id] + [end_id]
        target_ids = target_ids[: len(caption_ids)]
        length = len(caption_ids)

        return {
            "image_id": image_id,
            "features": features,
            "caption_text": caption_text,
            "caption_ids": caption_ids,
            "target_ids": target_ids,
            "length": length,
        }


def _collate_transformer(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length captions; features are already fixed-size."""
    pad_id = 0  # PAD_TOKEN is always index 0 per vocab convention
    max_len = max(s["length"] for s in batch)
    B = len(batch)

    features = torch.stack([s["features"] for s in batch])  # (B, 49, 2048)
    caption_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    target_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    padding_mask = torch.zeros(B, max_len, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    image_ids: List[str] = []
    caption_texts: List[str] = []

    for i, s in enumerate(batch):
        L = s["length"]
        caption_ids[i, :L] = torch.tensor(s["caption_ids"], dtype=torch.long)
        target_ids[i, :L] = torch.tensor(s["target_ids"], dtype=torch.long)
        padding_mask[i, :L] = 1
        lengths[i] = L
        image_ids.append(s["image_id"])
        caption_texts.append(s["caption_text"])

    return {
        "features": features,
        "image_ids": image_ids,
        "caption_texts": caption_texts,
        "caption_ids": caption_ids,
        "target_ids": target_ids,
        "padding_mask": padding_mask,
        "lengths": lengths,
    }


def create_transformer_dataloader(
    caption_map: Dict[str, List[str]],
    features_dir: str | Path,
    split_image_ids: List[str],
    vocab: Vocabulary,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    max_caption_length: int = DEFAULT_MAX_CAPTION_LENGTH,
) -> DataLoader:
    """Build a DataLoader over pre-extracted Transformer features for one split."""
    dataset = TransformerCaptionDataset(
        caption_map=caption_map,
        features_dir=features_dir,
        split_image_ids=split_image_ids,
        vocab=vocab,
        max_caption_length=max_caption_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_transformer,
    )
