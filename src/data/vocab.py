"""Vocabulary construction and serialization."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from src.data.constants import END_TOKEN, PAD_TOKEN, START_TOKEN, UNK_TOKEN
from src.data.preprocess import tokenize_caption


class Vocabulary:
    """Word-to-index mapping with fixed special tokens."""

    def __init__(self, word2idx: Dict[str, int]) -> None:
        self.word2idx = word2idx
        self.idx2word: Dict[int, str] = {idx: word for word, idx in word2idx.items()}

    @classmethod
    def build(cls, captions: List[str], min_word_freq: int = 5) -> "Vocabulary":
        """Build vocabulary from raw caption strings."""
        counter: Counter[str] = Counter()
        for caption in captions:
            tokens = tokenize_caption(caption)
            counter.update(tokens)

        word2idx: Dict[str, int] = {}
        # Fixed special token ids: pad=0, start=1, end=2, unk=3
        for token in [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]:
            word2idx[token] = len(word2idx)

        for word, freq in sorted(counter.items()):
            if freq >= min_word_freq and word not in word2idx:
                word2idx[word] = len(word2idx)

        return cls(word2idx)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert a list of tokens to a list of indices."""
        unk_id = self.word2idx[UNK_TOKEN]
        return [self.word2idx.get(token, unk_id) for token in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        """Convert a list of indices back to tokens."""
        return [self.idx2word.get(idx, UNK_TOKEN) for idx in ids]

    def save(self, path: str | Path) -> None:
        """Save vocabulary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.word2idx, handle, indent=2, ensure_ascii=True)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load vocabulary from a JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            word2idx = json.load(handle)
        return cls(word2idx)
