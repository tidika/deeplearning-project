"""Shared constants for the Flickr30k captioning data pipeline."""

from __future__ import annotations

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

DEFAULT_MAX_CAPTION_LENGTH = 40
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)
