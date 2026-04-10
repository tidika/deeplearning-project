"""Caption normalization and tokenization (shared across models)."""

from __future__ import annotations

import re
from typing import List

_PUNCT = '.,!?;:"()'


def normalize_caption(text: str) -> str:
    cleaned = text.strip().lower()
    for ch in _PUNCT:
        cleaned = cleaned.replace(ch, f" {ch} ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize_caption(text: str) -> List[str]:
    return normalize_caption(text).split()


def truncate_tokens(tokens: List[str], max_length: int) -> List[str]:
    """Keep content tokens so that [<start>] + tokens + [<end>] fits in max_length."""
    max_content = max(0, max_length - 2)
    return tokens[:max_content]
