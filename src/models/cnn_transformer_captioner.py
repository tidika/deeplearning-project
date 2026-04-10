"""CNN encoder (ResNet) + Transformer decoder for image captioning."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class CNNTransformerCaptioner(nn.Module):
    """
    Spatial grid features from a frozen or fine-tuned ResNet backbone,
    projected to `d_model` and consumed as `memory` by a Transformer decoder.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_caption_len: int = 40,
        freeze_backbone: bool = False,
        use_pretrained_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_caption_len = max_caption_len
        self.pad_id = 0

        weights = ResNet50_Weights.IMAGENET1K_V1 if use_pretrained_encoder else None
        backbone = models.resnet50(weights=weights)
        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.image_proj = nn.Linear(2048, d_model)
        self.image_pos = nn.Parameter(torch.randn(1, 49, d_model) * 0.02)

        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_caption_len, d_model) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.logits = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.word_emb.weight, std=0.02)
        nn.init.normal_(self.logits.weight, std=0.02)
        nn.init.zeros_(self.logits.bias)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Return memory tensor (batch, num_patches, d_model)."""
        feat = self.backbone(images)
        b, c, h, w = feat.shape
        feat = feat.flatten(2).transpose(1, 2)
        memory = self.image_proj(feat) + self.image_pos
        return memory * math.sqrt(self.d_model)

    def forward(
        self,
        images: torch.Tensor,
        caption_ids: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Teacher forcing: `caption_ids` are full padded sequences including <start>...<end>.
        Returns logits for positions matching `caption_ids` (same length).
        """
        memory = self.encode_images(images)
        tgt_emb = self.word_emb(caption_ids) * math.sqrt(self.d_model)
        seq_len = caption_ids.size(1)
        tgt_emb = tgt_emb + self.pos_emb[:, :seq_len, :]
        tgt_emb = self.dropout(tgt_emb)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=images.device, dtype=torch.bool),
            diagonal=1,
        )
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = caption_ids == self.pad_id

        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,
        )
        return self.logits(out)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        start_id: int,
        end_id: int,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Greedy decoding; returns token ids including <start> ... <end>."""
        self.eval()
        max_len = max_len or self.max_caption_len
        batch = images.size(0)
        device = images.device
        memory = self.encode_images(images)

        ids = torch.full((batch, 1), start_id, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_emb = self.word_emb(ids) * math.sqrt(self.d_model)
            seq_len = ids.size(1)
            tgt_emb = tgt_emb + self.pos_emb[:, :seq_len, :]
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            out = self.decoder(tgt=self.dropout(tgt_emb), memory=memory, tgt_mask=causal_mask)
            next_logits = self.logits(out[:, -1, :])
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if bool((next_id.squeeze(-1) == end_id).all()):
                break
        return ids
