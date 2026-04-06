"""Caption preprocessing: normalization, tokenization, and truncation."""

from __future__ import annotations

import re
from typing import List


# Punctuation marks that get surrounded with spaces before tokenization.
_PUNCTUATION_RE = re.compile(r"([.,!?;:\"()\)])")


def normalize_caption(raw: str) -> str:
    """Lowercase, space-pad punctuation, and collapse whitespace."""
    text = raw.strip().lower()
    text = _PUNCTUATION_RE.sub(r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_caption(raw: str) -> List[str]:
    """Normalize then split on whitespace."""
    return normalize_caption(raw).split()


def truncate_tokens(tokens: List[str], max_length: int) -> List[str]:
    """Truncate token list so that *with* <start> and <end> it fits max_length.

    The caller is expected to add boundary tokens after truncation, so we
    reserve 2 slots for them.
    """
    return tokens[: max_length - 2]
