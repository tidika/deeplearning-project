"""Bottom-Up Top-Down attention model for image captioning.

Implements the two-layer LSTM decoder from:
  Anderson et al., "Bottom-Up and Top-Down Attention for Image Captioning
  and Visual Question Answering", CVPR 2018.

Assumes region features have been pre-extracted by
scripts/extract_butd_features.py and stored as
  data/butd_features/{image_id}.pt  →  shape (num_regions, feature_dim)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class AdditiveAttention(nn.Module):
    """Soft additive (Bahdanau-style) attention over K visual regions.

    Computes a single attended feature vector as a weighted sum of region
    features, where weights are derived from the alignment between each
    region and the current top-down LSTM hidden state.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        # Project region features and hidden state into a shared attention space
        self.feat_proj = nn.Linear(feature_dim, attention_dim, bias=False)
        self.h_proj = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, features: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features: (B, K, feature_dim) — region feature bank.
            h:        (B, hidden_dim)      — top-down LSTM hidden state.

        Returns:
            attended: (B, feature_dim) attention-weighted feature sum.
            alpha:    (B, K)           attention weights (sum to 1 per sample).
        """
        feat_att = self.feat_proj(features)            # (B, K, att_dim)
        h_att = self.h_proj(h).unsqueeze(1)            # (B, 1, att_dim)
        energy = self.v(torch.tanh(feat_att + h_att))  # (B, K, 1)
        alpha = torch.softmax(energy.squeeze(-1), dim=1)          # (B, K)
        attended = (alpha.unsqueeze(-1) * features).sum(dim=1)    # (B, feature_dim)
        return attended, alpha


class BUTDCaptionModel(nn.Module):
    """Two-layer LSTM captioning decoder with bottom-up top-down attention.

    Layer 1 — Attention LSTM:
        input  = concat(h2_{t-1}, mean_features, word_emb_{t-1})
        output = h1_t  (used as attention query)

    Layer 2 — Language LSTM:
        input  = concat(attended_feat, h1_t)
        output = h2_t  (used to predict the next word)

    Training uses teacher forcing: caption_ids (including <start>) are fed
    as inputs at every step and logits are returned for all positions.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        feature_dim: int = 1024,
        attention_dim: int = 512,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # LSTM 1: top-down attention LSTM
        # input: concat(h2_prev [hidden_dim], mean_feat [feature_dim], word_emb [embed_dim])
        self.lstm1 = nn.LSTMCell(
            input_size=hidden_dim + feature_dim + embed_dim,
            hidden_size=hidden_dim,
        )

        self.attention = AdditiveAttention(feature_dim, hidden_dim, attention_dim)

        # LSTM 2: language LSTM
        # input: concat(attended_feat [feature_dim], h1 [hidden_dim])
        self.lstm2 = nn.LSTMCell(
            input_size=feature_dim + hidden_dim,
            hidden_size=hidden_dim,
        )

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Linear projections to initialize hidden/cell states from mean features
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

    def _init_hidden(
        self, mean_feat: Tensor
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """Compute initial (h, c) for both LSTMs from mean-pooled features."""
        h = torch.tanh(self.init_h(mean_feat))  # (B, hidden_dim)
        c = torch.tanh(self.init_c(mean_feat))  # (B, hidden_dim)
        return (h, c), (h.clone(), c.clone())

    def forward(self, features: Tensor, caption_ids: Tensor) -> Tensor:
        """Teacher-forced forward pass.

        Args:
            features:    (B, K, feature_dim) pre-extracted region features.
            caption_ids: (B, T) token ids; first column is always <start>.

        Returns:
            logits: (B, T, vocab_size) unnormalized word scores.
        """
        B, _K, _D = features.shape
        T = caption_ids.shape[1]

        mean_feat = features.mean(dim=1)  # (B, feature_dim) — global context
        (h1, c1), (h2, c2) = self._init_hidden(mean_feat)

        word_embs = self.dropout(self.embed(caption_ids))  # (B, T, embed_dim)
        logits: List[Tensor] = []

        for t in range(T):
            word_emb = word_embs[:, t, :]  # (B, embed_dim)

            # --- Attention LSTM ---
            lstm1_in = torch.cat([h2, mean_feat, word_emb], dim=1)
            h1, c1 = self.lstm1(lstm1_in, (h1, c1))

            # --- Attention over regions ---
            attended, _ = self.attention(features, h1)

            # --- Language LSTM ---
            lstm2_in = torch.cat([attended, h1], dim=1)
            h2, c2 = self.lstm2(lstm2_in, (h2, c2))

            logits.append(self.fc_out(self.dropout(h2)))  # (B, vocab_size)

        return torch.stack(logits, dim=1)  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        features: Tensor,
        start_id: int,
        end_id: int,
        max_length: int = 40,
    ) -> Tensor:
        """Greedy decoding from pre-extracted features.

        Args:
            features:   (B, K, feature_dim).
            start_id:   Index of the <start> token.
            end_id:     Index of the <end> token.
            max_length: Maximum number of tokens to generate.

        Returns:
            (B, L) tensor of generated token ids (starts after <start>,
            includes <end> if generated within max_length).
        """
        self.eval()
        B = features.shape[0]
        device = features.device

        mean_feat = features.mean(dim=1)
        (h1, c1), (h2, c2) = self._init_hidden(mean_feat)

        curr = torch.full((B,), start_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        tokens: List[Tensor] = []

        for _ in range(max_length):
            word_emb = self.embed(curr)  # no dropout at inference

            lstm1_in = torch.cat([h2, mean_feat, word_emb], dim=1)
            h1, c1 = self.lstm1(lstm1_in, (h1, c1))

            attended, _ = self.attention(features, h1)

            lstm2_in = torch.cat([attended, h1], dim=1)
            h2, c2 = self.lstm2(lstm2_in, (h2, c2))

            curr = self.fc_out(h2).argmax(dim=-1)  # (B,)
            tokens.append(curr.clone())

            finished = finished | (curr == end_id)
            if finished.all():
                break

        return torch.stack(tokens, dim=1)  # (B, L)
