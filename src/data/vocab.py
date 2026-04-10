"""Vocabulary built from training captions only."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from src.data.constants import END_TOKEN, PAD_TOKEN, START_TOKEN, UNK_TOKEN
from src.data.preprocess import normalize_caption, tokenize_caption, truncate_tokens


class Vocabulary:
    def __init__(self, word2idx: Dict[str, int]) -> None:
        self.word2idx = dict(word2idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def __len__(self) -> int:
        return len(self.word2idx)

    @classmethod
    def build(cls, caption_texts: Iterable[str], min_word_freq: int) -> "Vocabulary":
        counter: Counter[str] = Counter()
        for text in caption_texts:
            tokens = tokenize_caption(normalize_caption(text))
            counter.update(tokens)

        words = [word for word, count in counter.items() if count >= min_word_freq]
        words.sort()

        word2idx: Dict[str, int] = {
            PAD_TOKEN: 0,
            START_TOKEN: 1,
            END_TOKEN: 2,
            UNK_TOKEN: 3,
        }
        for word in words:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
        return cls(word2idx)

    def encode(self, text: str, max_caption_length: int) -> List[int]:
        tokens = truncate_tokens(tokenize_caption(normalize_caption(text)), max_caption_length)
        unk = self.word2idx[UNK_TOKEN]
        body = [self.word2idx.get(token, unk) for token in tokens]
        return [self.word2idx[START_TOKEN], *body, self.word2idx[END_TOKEN]]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        pad_id = self.word2idx[PAD_TOKEN]
        start_id = self.word2idx[START_TOKEN]
        end_id = self.word2idx[END_TOKEN]
        pieces: List[str] = []
        for idx in ids:
            i = int(idx)
            if i == pad_id:
                continue
            if i == end_id:
                break
            if skip_special and i == start_id:
                continue
            word = self.idx2word.get(i, UNK_TOKEN)
            if word not in (PAD_TOKEN, START_TOKEN, END_TOKEN):
                pieces.append(word)
        return " ".join(pieces)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"word2idx": self.word2idx}, handle, indent=2, ensure_ascii=True)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        word2idx = raw["word2idx"]
        return cls(word2idx)
