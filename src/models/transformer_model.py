"""Transformer-based image captioning model.

Implements a ResNet-50 encoder + Transformer decoder architecture for
image captioning, used as the second model in the BUTD vs Transformer
comparison study.

Visual pipeline:
    Image → ResNet-50 (frozen) → 7×7 spatial grid → flatten → 49 region
    vectors of 2048-dim each.  Features are pre-extracted offline by
    scripts/extract_transformer_features.py.

Caption pipeline:
    Tokens → Embedding + Positional Encoding → Transformer Decoder
    (masked self-attention over text + cross-attention over image features)
    → word logits.

Reference:
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to token embeddings.

    Injects information about the position of each token in the sequence
    so the transformer can distinguish word order.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, embed_dim) token embeddings.
        Returns:
            (B, T, embed_dim) embeddings with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Visual Feature Projection
# ---------------------------------------------------------------------------
class VisualProjection(nn.Module):
    """Projects spatial ResNet features into the transformer's embed_dim space.

    ResNet-50 outputs 2048-dim features per spatial location. This module
    linearly projects them to embed_dim so they can be used as cross-attention
    keys/values in the transformer decoder.
    """

    def __init__(self, feature_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, num_patches, feature_dim) spatial image features.
        Returns:
            (B, num_patches, embed_dim) projected features.
        """
        return self.norm(self.proj(features))


# ---------------------------------------------------------------------------
# Transformer Caption Model
# ---------------------------------------------------------------------------
class TransformerCaptionModel(nn.Module):
    """Transformer decoder captioning model with cross-attention over spatial
    ResNet-50 features.

    Architecture:
        - Visual projection: (B, 49, 2048) → (B, 49, embed_dim)
        - Token embedding + sinusoidal positional encoding
        - N transformer decoder layers, each with:
            * Masked multi-head self-attention (causal, over text tokens)
            * Multi-head cross-attention (over projected image features)
            * Feed-forward network
        - Linear output projection → vocab logits

    All hyperparameters are configurable so ablation experiments
    (changing num_layers, num_heads, embed_dim, etc.) are straightforward.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 2048,
        feature_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        """
        Args:
            vocab_size:   Size of the output vocabulary.
            embed_dim:    Token embedding / model dimension (d_model).
            num_heads:    Number of attention heads in each layer.
            num_layers:   Number of transformer decoder layers.
            ff_dim:       Hidden dimension of the feed-forward sub-layer.
            feature_dim:  Dimension of input spatial features (2048 for ResNet-50).
            dropout:      Dropout rate applied throughout the model.
            max_seq_len:  Maximum caption length for positional encoding.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # --- Visual side ---
        self.visual_proj = VisualProjection(feature_dim, embed_dim)

        # --- Text side ---
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout, max_seq_len)

        # --- Transformer decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,  # (B, T, D) convention throughout
            norm_first=True,   # Pre-LN for more stable training
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        # --- Output projection ---
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform init for embedding and output projection."""
        nn.init.xavier_uniform_(self.token_embed.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def _causal_mask(self, size: int, device: torch.device) -> Tensor:
        """Upper-triangular mask that prevents attending to future tokens.

        Returns a (size, size) bool tensor where True means 'ignore this
        position' (PyTorch's convention for tgt_mask).
        """
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(self, features: Tensor, caption_ids: Tensor) -> Tensor:
        """Teacher-forced forward pass.

        Args:
            features:    (B, num_patches, feature_dim) pre-extracted spatial features.
            caption_ids: (B, T) token ids starting with <start>.

        Returns:
            logits: (B, T, vocab_size) unnormalized word scores.
        """
        B, T = caption_ids.shape

        # Project visual features into embed_dim space
        memory = self.visual_proj(features)  # (B, num_patches, embed_dim)

        # Embed tokens and add positional encoding
        tgt = self.pos_encoding(
            self.token_embed(caption_ids) * math.sqrt(self.embed_dim)
        )  # (B, T, embed_dim)

        # Causal mask — prevents each position from attending to future tokens
        tgt_mask = self._causal_mask(T, caption_ids.device)

        # Key padding mask — True where caption_ids == 0 (PAD token)
        tgt_key_padding_mask = caption_ids == 0  # (B, T)

        # Transformer decoder: self-attn over text + cross-attn over image
        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, T, embed_dim)

        return self.fc_out(out)  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        features: Tensor,
        start_id: int,
        end_id: int,
        max_length: int = 40,
    ) -> Tensor:
        """Greedy autoregressive decoding.

        Args:
            features:   (B, num_patches, feature_dim) spatial image features.
            start_id:   Index of the <start> token.
            end_id:     Index of the <end> token.
            max_length: Maximum tokens to generate per caption.

        Returns:
            (B, L) tensor of generated token ids (after <start>,
            includes <end> if generated within max_length).
        """
        self.eval()
        B = features.shape[0]
        device = features.device

        # Project visual features once
        memory = self.visual_proj(features)  # (B, num_patches, embed_dim)

        # Start with <start> token for every item in the batch
        generated = torch.full((B, 1), start_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_length):
            T = generated.shape[1]
            tgt = self.pos_encoding(
                self.token_embed(generated) * math.sqrt(self.embed_dim)
            )
            tgt_mask = self._causal_mask(T, device)

            out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            # Take logits at the last position only
            next_token = self.fc_out(out[:, -1, :]).argmax(dim=-1)  # (B,)

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == end_id)
            if finished.all():
                break

        # Return only the generated tokens (strip leading <start>)
        return generated[:, 1:]  # (B, L)
