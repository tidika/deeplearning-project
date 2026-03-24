"""Evaluate generated captions from JSON files using the shared metric pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import evaluate_captions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--references-path",
        required=True,
        help="Path to JSON mapping image ids to a list of reference captions.",
    )
    parser.add_argument(
        "--predictions-path",
        required=True,
        help="Path to JSON mapping image ids to a generated caption string.",
    )
    parser.add_argument(
        "--output-path",
        default="",
        help="Optional path to save the evaluation summary JSON.",
    )
    return parser.parse_args()


def _load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_references(raw) -> Dict[str, List[str]]:
    if not isinstance(raw, dict):
        raise TypeError("references JSON must be an object mapping image ids to lists of captions")

    references: Dict[str, List[str]] = {}
    for image_id, captions in raw.items():
        if not isinstance(image_id, str):
            raise TypeError("reference image ids must be strings")
        if not isinstance(captions, list) or not captions:
            raise TypeError(f"references[{image_id!r}] must be a non-empty list of strings")
        if not all(isinstance(caption, str) for caption in captions):
            raise TypeError(f"references[{image_id!r}] must contain only strings")
        references[image_id] = captions
    return references


def _validate_predictions(raw) -> Dict[str, str]:
    if not isinstance(raw, dict):
        raise TypeError("predictions JSON must be an object mapping image ids to caption strings")

    predictions: Dict[str, str] = {}
    for image_id, caption in raw.items():
        if not isinstance(image_id, str):
            raise TypeError("prediction image ids must be strings")
        if not isinstance(caption, str):
            raise TypeError(f"predictions[{image_id!r}] must be a string")
        predictions[image_id] = caption
    return predictions


def main() -> None:
    args = parse_args()

    references = _validate_references(_load_json(REPO_ROOT / args.references_path))
    predictions = _validate_predictions(_load_json(REPO_ROOT / args.predictions_path))

    common_ids = sorted(set(references) & set(predictions))
    if not common_ids:
        raise ValueError("references and predictions do not share any image ids")

    missing_predictions = sorted(set(references) - set(predictions))
    extra_predictions = sorted(set(predictions) - set(references))
    results = evaluate_captions(references, predictions, require_exact_match=True)
    summary = {
        "references_path": args.references_path,
        "predictions_path": args.predictions_path,
        "num_reference_ids": len(references),
        "num_prediction_ids": len(predictions),
        "num_common_ids": len(common_ids),
        "num_missing_predictions": len(missing_predictions),
        "num_extra_predictions": len(extra_predictions),
        "metrics": {metric_name: round(float(metric_value), 6) for metric_name, metric_value in results.items()},
    }

    print(json.dumps(summary, indent=2, ensure_ascii=True))

    if args.output_path:
        output_path = REPO_ROOT / args.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
