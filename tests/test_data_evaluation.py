"""Lightweight automated tests for the shared data and evaluation pipeline."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import DEFAULT_IMAGE_SIZE, create_caption_dataloader
from src.data.preprocess import normalize_caption, tokenize_caption
from src.data.splits import load_caption_map, load_split_file
from src.data.vocab import Vocabulary
from src.eval.metrics import evaluate_captions


class DataEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.captions_path = REPO_ROOT / "data/captions.txt"
        cls.images_dir = REPO_ROOT / "data/Images"
        cls.train_split_path = REPO_ROOT / "metadata/splits/train.txt"
        cls.val_split_path = REPO_ROOT / "metadata/splits/val.txt"
        cls.test_split_path = REPO_ROOT / "metadata/splits/test.txt"
        cls.vocab_path = REPO_ROOT / "metadata/vocab/flickr30k_vocab.json"
        cls.vocab_stats_path = REPO_ROOT / "metadata/vocab/flickr30k_vocab_stats.json"
        cls.references_path = REPO_ROOT / "configs/references.example.json"
        cls.predictions_path = REPO_ROOT / "configs/predictions.example.json"

        cls.caption_map = load_caption_map(cls.captions_path)
        cls.train_ids = load_split_file(cls.train_split_path)
        cls.val_ids = load_split_file(cls.val_split_path)
        cls.test_ids = load_split_file(cls.test_split_path)
        cls.vocab = Vocabulary.load(cls.vocab_path)
        with cls.vocab_stats_path.open("r", encoding="utf-8") as handle:
            cls.vocab_stats = json.load(handle)
        with cls.references_path.open("r", encoding="utf-8") as handle:
            cls.references = json.load(handle)
        with cls.predictions_path.open("r", encoding="utf-8") as handle:
            cls.predictions = json.load(handle)

    def test_preprocessing_rules_are_stable(self) -> None:
        raw = '  A Dog,  RUNS! (Fast)  '
        self.assertEqual(normalize_caption(raw), 'a dog , runs ! ( fast )')
        self.assertEqual(tokenize_caption(raw), ['a', 'dog', ',', 'runs', '!', '(', 'fast', ')'])

    def test_special_token_ids_are_fixed(self) -> None:
        self.assertEqual(self.vocab.word2idx['<pad>'], 0)
        self.assertEqual(self.vocab.word2idx['<start>'], 1)
        self.assertEqual(self.vocab.word2idx['<end>'], 2)
        self.assertEqual(self.vocab.word2idx['<unk>'], 3)

    def test_splits_are_disjoint(self) -> None:
        train_ids = set(self.train_ids)
        val_ids = set(self.val_ids)
        test_ids = set(self.test_ids)
        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)

    def test_vocab_is_built_from_training_split_only(self) -> None:
        train_captions = []
        for image_id in self.train_ids:
            train_captions.extend(self.caption_map[image_id])

        rebuilt_vocab = Vocabulary.build(
            train_captions,
            min_word_freq=int(self.vocab_stats['min_word_freq']),
        )
        self.assertEqual(rebuilt_vocab.word2idx, self.vocab.word2idx)

    def test_dataloader_outputs_expected_shapes(self) -> None:
        subset_ids = self.train_ids[:4]
        dataloader = create_caption_dataloader(
            captions_path=self.captions_path,
            images_dir=self.images_dir,
            split_image_ids=subset_ids,
            vocab=self.vocab,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            max_caption_length=40,
            image_size=DEFAULT_IMAGE_SIZE,
        )
        batch = next(iter(dataloader))

        self.assertEqual(tuple(batch['images'].shape), (4, 3, 224, 224))
        self.assertEqual(tuple(batch['caption_ids'].shape), tuple(batch['target_ids'].shape))
        self.assertEqual(tuple(batch['caption_ids'].shape), tuple(batch['padding_mask'].shape))
        self.assertEqual(tuple(batch['lengths'].shape), (4,))

    def test_evaluation_returns_expected_metrics(self) -> None:
        metrics = evaluate_captions(self.references, self.predictions)
        self.assertEqual(set(metrics.keys()), {'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor'})
        for value in metrics.values():
            self.assertGreaterEqual(float(value), 0.0)


if __name__ == '__main__':
    unittest.main()
