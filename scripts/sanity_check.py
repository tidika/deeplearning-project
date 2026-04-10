"""Run one combined sanity check over splits, vocabulary, dataloader, and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.constants import END_TOKEN, PAD_TOKEN, START_TOKEN, UNK_TOKEN
from src.data.dataset import DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_STD, create_caption_dataloader
from src.data.splits import load_caption_map, load_split_file
from src.data.vocab import Vocabulary
from src.eval.metrics import evaluate_captions
from src.project_config import (
    DEFAULT_CONFIG_PATH,
    get_config_value,
    get_image_normalization,
    get_image_size,
    load_project_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--captions-path", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--test-split", default=None)
    parser.add_argument("--vocab-path", default=None)
    parser.add_argument("--vocab-stats-path", default=None)
    parser.add_argument("--references-path", default=None)
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--image-mean", type=float, nargs=3, default=None)
    parser.add_argument("--image-std", type=float, nargs=3, default=None)
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _check(name: str, passed: bool, **details: Any) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def _tensor_shape(value: Any) -> List[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def main() -> None:
    args = parse_args()
    config = load_project_config(REPO_ROOT / args.config)

    captions_path = REPO_ROOT / (args.captions_path or get_config_value(config, "captions_path", "data/captions.txt"))
    images_dir = REPO_ROOT / (args.images_dir or get_config_value(config, "images_dir", "data/Images"))
    splits_dir = Path(get_config_value(config, "splits_dir", "metadata/splits"))
    vocab_dir = Path(get_config_value(config, "vocab_dir", "metadata/vocab"))
    train_split_path = REPO_ROOT / (args.train_split or str(splits_dir / "train.txt"))
    val_split_path = REPO_ROOT / (args.val_split or str(splits_dir / "val.txt"))
    test_split_path = REPO_ROOT / (args.test_split or str(splits_dir / "test.txt"))
    vocab_path = REPO_ROOT / (args.vocab_path or str(vocab_dir / "flickr30k_vocab.json"))
    vocab_stats_path = REPO_ROOT / (args.vocab_stats_path or str(vocab_dir / "flickr30k_vocab_stats.json"))
    references_path = REPO_ROOT / (args.references_path or get_config_value(config, "references_example_path", "configs/references.example.json"))
    predictions_path = REPO_ROOT / (args.predictions_path or get_config_value(config, "predictions_example_path", "configs/predictions.example.json"))

    config_image_size = get_image_size(config, default=DEFAULT_IMAGE_SIZE)
    image_size = args.image_size if args.image_size is not None else int(config_image_size[0])
    config_mean, config_std = get_image_normalization(config, default_mean=DEFAULT_IMAGE_MEAN, default_std=DEFAULT_IMAGE_STD)
    image_mean = tuple(args.image_mean) if args.image_mean is not None else config_mean
    image_std = tuple(args.image_std) if args.image_std is not None else config_std

    checks: List[Dict[str, Any]] = []

    caption_map = load_caption_map(captions_path)
    image_files = {path.name for path in images_dir.glob("*.jpg")}
    caption_image_ids = set(caption_map.keys())
    caption_counts = [len(captions) for captions in caption_map.values()]

    checks.append(_check("dataset_alignment", image_files == caption_image_ids, num_image_files=len(image_files), num_caption_image_ids=len(caption_image_ids), image_only_count=len(image_files - caption_image_ids), caption_only_count=len(caption_image_ids - image_files)))
    checks.append(_check("captions_per_image", len(set(caption_counts)) == 1 and caption_counts[0] == 5, unique_counts=sorted(set(caption_counts)), min=min(caption_counts), max=max(caption_counts)))

    train_ids = set(load_split_file(train_split_path))
    val_ids = set(load_split_file(val_split_path))
    test_ids = set(load_split_file(test_split_path))
    all_split_ids = train_ids | val_ids | test_ids

    checks.append(_check("split_disjointness", not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids), train_val_overlap=len(train_ids & val_ids), train_test_overlap=len(train_ids & test_ids), val_test_overlap=len(val_ids & test_ids)))
    checks.append(_check("split_coverage", all_split_ids == caption_image_ids, total_unique_split_ids=len(all_split_ids), caption_image_ids=len(caption_image_ids), missing_from_splits=len(caption_image_ids - all_split_ids), missing_from_captions=len(all_split_ids - caption_image_ids)))
    checks.append(_check("split_image_files_exist", train_ids.issubset(image_files) and val_ids.issubset(image_files) and test_ids.issubset(image_files), missing_train_images=len(train_ids - image_files), missing_val_images=len(val_ids - image_files), missing_test_images=len(test_ids - image_files)))

    vocab = Vocabulary.load(vocab_path)
    special_token_ids = {PAD_TOKEN: vocab.word2idx.get(PAD_TOKEN), START_TOKEN: vocab.word2idx.get(START_TOKEN), END_TOKEN: vocab.word2idx.get(END_TOKEN), UNK_TOKEN: vocab.word2idx.get(UNK_TOKEN)}
    checks.append(_check("vocab_special_tokens", special_token_ids == {PAD_TOKEN: 0, START_TOKEN: 1, END_TOKEN: 2, UNK_TOKEN: 3}, token_ids=special_token_ids, vocab_size=len(vocab.word2idx)))

    train_captions: List[str] = []
    for image_id in sorted(train_ids):
        train_captions.extend(caption_map[image_id])

    vocab_stats = _load_json(vocab_stats_path)
    min_word_freq = int(vocab_stats["min_word_freq"])
    max_caption_length = int(get_config_value(config, "max_caption_length", vocab_stats.get("max_caption_length", 40)))
    rebuilt_vocab = Vocabulary.build(train_captions, min_word_freq=min_word_freq)
    checks.append(_check("vocab_matches_training_split", rebuilt_vocab.word2idx == vocab.word2idx, num_train_images=len(train_ids), num_train_captions=len(train_captions), stats_num_train_images=int(vocab_stats["num_train_images"]), stats_num_train_captions=int(vocab_stats["num_train_captions"]), min_word_freq=min_word_freq))

    dataloader = create_caption_dataloader(captions_path=captions_path, images_dir=images_dir, split_image_ids=sorted(train_ids), vocab=vocab, batch_size=args.batch_size, shuffle=False, num_workers=0, max_caption_length=max_caption_length, image_size=image_size, image_mean=image_mean, image_std=image_std)
    batch = next(iter(dataloader))
    image_shape = _tensor_shape(batch["images"])
    caption_shape = _tensor_shape(batch["caption_ids"])
    target_shape = _tensor_shape(batch["target_ids"])
    mask_shape = _tensor_shape(batch["padding_mask"])
    lengths_shape = _tensor_shape(batch["lengths"])
    dataloader_passed = image_shape is not None and caption_shape is not None and target_shape is not None and mask_shape is not None and lengths_shape is not None and image_shape[0] == args.batch_size and image_shape[1] == 3 and caption_shape == target_shape == mask_shape and lengths_shape == [args.batch_size]
    checks.append(_check("dataloader_batch", dataloader_passed, images_shape=image_shape, caption_ids_shape=caption_shape, target_ids_shape=target_shape, padding_mask_shape=mask_shape, lengths_shape=lengths_shape, batch_image_min=float(batch["images"].min().item()), batch_image_max=float(batch["images"].max().item()), first_image_id=batch["image_ids"][0]))

    references = _load_json(references_path)
    predictions = _load_json(predictions_path)
    metrics = evaluate_captions(references, predictions)
    checks.append(_check("evaluation_pipeline", all(metric_name in metrics for metric_name in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor"]), metrics={key: round(float(value), 6) for key, value in metrics.items()}, num_reference_ids=len(references), num_prediction_ids=len(predictions), num_common_ids=len(set(references) & set(predictions))))

    summary = {"config_path": args.config, "python_executable": sys.executable, "overall_passed": all(check["passed"] for check in checks), "checks": checks}
    print(json.dumps(summary, indent=2, ensure_ascii=True))

    if args.output_path:
        output_path = REPO_ROOT / args.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=True)

    if not summary["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
