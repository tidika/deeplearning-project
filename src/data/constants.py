"""Shared constants for the data pipeline."""

from __future__ import annotations

PAD_TOKEN: str = "<pad>"
START_TOKEN: str = "<start>"
END_TOKEN: str = "<end>"
UNK_TOKEN: str = "<unk>"

SPECIAL_TOKENS: list[str] = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

DEFAULT_MAX_CAPTION_LENGTH: int = 40
