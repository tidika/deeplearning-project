"""Tests for the CNN + Transformer captioner (no dataset, no pretrained weight download)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.cnn_transformer_captioner import CNNTransformerCaptioner


class TransformerModelTests(unittest.TestCase):
    def test_forward_and_generate_shapes(self) -> None:
        torch.manual_seed(0)
        model = CNNTransformerCaptioner(
            vocab_size=1000,
            d_model=128,
            nhead=4,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            max_caption_len=16,
            freeze_backbone=True,
            use_pretrained_encoder=False,
        )
        model.eval()
        images = torch.randn(2, 3, 224, 224)
        caption_ids = torch.tensor([[1, 5, 6, 2, 0, 0], [1, 7, 2, 0, 0, 0]])
        with torch.no_grad():
            logits = model(images, caption_ids)
        self.assertEqual(tuple(logits.shape), (2, 6, 1000))

        with torch.no_grad():
            ids = model.generate(images, start_id=1, end_id=2, max_len=12)
        self.assertEqual(ids.dim(), 2)
        self.assertEqual(ids.size(0), 2)
        self.assertTrue(ids.size(1) >= 1)


if __name__ == "__main__":
    unittest.main()
