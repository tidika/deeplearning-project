"""Lightweight tests that do not require the full local Flickr30k dataset."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess import normalize_caption, tokenize_caption, truncate_tokens
from src.data.splits import build_image_splits
from src.eval.metrics import evaluate_captions
from src.project_config import get_image_normalization, get_image_size, load_project_config


class LightweightChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_project_config(REPO_ROOT / "configs/data_config.example.json")
        with (REPO_ROOT / "configs/references.example.json").open("r", encoding="utf-8") as handle:
            cls.references = json.load(handle)
        with (REPO_ROOT / "configs/predictions.example.json").open("r", encoding="utf-8") as handle:
            cls.predictions = json.load(handle)

    def test_config_defaults_are_readable(self) -> None:
        self.assertEqual(self.config["captions_path"], "data/captions.txt")
        self.assertEqual(get_image_size(self.config), (224, 224))
        mean, std = get_image_normalization(self.config)
        self.assertEqual(mean, (0.485, 0.456, 0.406))
        self.assertEqual(std, (0.229, 0.224, 0.225))

    def test_preprocessing_rules_are_stable(self) -> None:
        raw = '  A Dog,  RUNS! (Fast)  '
        self.assertEqual(normalize_caption(raw), 'a dog , runs ! ( fast )')
        self.assertEqual(tokenize_caption(raw), ['a', 'dog', ',', 'runs', '!', '(', 'fast', ')'])
        self.assertEqual(truncate_tokens(['a', 'b', 'c', 'd'], max_length=4), ['a', 'b'])

    def test_invalid_split_ratios_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_image_splits(["a.jpg", "b.jpg"], train_ratio=-0.1, val_ratio=0.6, test_ratio=0.5)

    def test_evaluation_requires_exact_id_match(self) -> None:
        partial_predictions = dict(self.predictions)
        partial_predictions.pop(next(iter(partial_predictions)))
        with self.assertRaises(ValueError):
            evaluate_captions(self.references, partial_predictions, require_exact_match=True)


if __name__ == "__main__":
    unittest.main()
