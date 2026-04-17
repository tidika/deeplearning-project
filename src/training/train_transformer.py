"""Training and evaluation utilities for the Transformer captioning model.

Provides three importable functions used by notebooks/train_transformer.ipynb
and notebooks/colab_transformer.ipynb:
    train_one_epoch   — one teacher-forced pass over the training DataLoader
    evaluate_loss     — validation loss over the val DataLoader
    generate_captions — greedy autoregressive decoding, returns
                        Dict[image_id, predicted_caption]

The API is intentionally identical to src/training/train_butd.py so that
the results_analysis notebook can call both with the same interface.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.constants import END_TOKEN, START_TOKEN
from src.data.vocab import Vocabulary
from src.models.transformer_model import TransformerCaptionModel


def train_one_epoch(
    model: TransformerCaptionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Run one epoch of teacher-forced training.

    Args:
        model:     TransformerCaptionModel (moved to device by the caller).
        loader:    DataLoader from create_transformer_dataloader.
        optimizer: Any torch optimizer.
        criterion: Loss function; typically CrossEntropyLoss(ignore_index=0).
        device:    Compute device.
        grad_clip: Max gradient norm. Transformers are sensitive to large
                   gradients; 1.0 is a safer default than the 5.0 used for LSTMs.

    Returns:
        Average cross-entropy loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="  train", unit="batch", leave=False)
    for batch in pbar:
        features = batch["features"].to(device)       # (B, 49, 2048)
        caption_ids = batch["caption_ids"].to(device)  # (B, T)
        target_ids = batch["target_ids"].to(device)    # (B, T)

        logits = model(features, caption_ids)  # (B, T, vocab_size)

        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), target_ids.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_loss(
    model: TransformerCaptionModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute mean cross-entropy loss over a DataLoader (no gradient).

    Args:
        model:     TransformerCaptionModel.
        loader:    Validation DataLoader.
        criterion: Same criterion used during training.
        device:    Compute device.

    Returns:
        Average loss over all batches.
    """
    model.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc="    val", unit="batch", leave=False)
    for batch in pbar:
        features = batch["features"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        logits = model(features, caption_ids)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), target_ids.view(B * T))
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def generate_captions(
    model: TransformerCaptionModel,
    loader: DataLoader,
    vocab: Vocabulary,
    device: torch.device,
    max_length: int = 40,
) -> Dict[str, str]:
    """Generate one caption per unique image via greedy decoding.

    Iterates the DataLoader and generates a caption for each image. When
    the same image_id appears multiple times (5 captions per image in
    Flickr30k), only the first prediction is kept.

    Args:
        model:      TransformerCaptionModel.
        loader:     DataLoader (shuffle=False recommended for eval).
        vocab:      Vocabulary with word2idx / idx2word.
        device:     Compute device.
        max_length: Maximum number of tokens to generate per caption.

    Returns:
        Dict mapping image_id → predicted caption string.
    """
    model.eval()
    start_id = vocab.word2idx[START_TOKEN]
    end_id = vocab.word2idx[END_TOKEN]

    predictions: Dict[str, str] = {}

    for batch in loader:
        image_ids = batch["image_ids"]
        features = batch["features"].to(device)

        # Skip images already processed
        new_mask = [i for i, iid in enumerate(image_ids) if iid not in predictions]
        if not new_mask:
            continue

        new_features = features[new_mask]
        new_ids = [image_ids[i] for i in new_mask]

        generated = model.generate(new_features, start_id, end_id, max_length)

        for iid, token_ids in zip(new_ids, generated.tolist()):
            if end_id in token_ids:
                token_ids = token_ids[: token_ids.index(end_id)]
            words = vocab.decode(token_ids)
            predictions[iid] = " ".join(words)

    return predictions
