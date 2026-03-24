"""BLEU and METEOR scoring utilities for generated captions."""

from __future__ import annotations

from typing import Dict, List


def _tokenize_for_metrics(text: str) -> List[str]:
    return text.strip().lower().split()


def compute_bleu_scores(
    references: Dict[str, List[str]],
    predictions: Dict[str, str],
) -> Dict[str, float]:
    """Compute BLEU-1 through BLEU-4 with a shared evaluation contract."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    except ImportError as exc:  # pragma: no cover - dependency may be installed later
        raise ImportError(
            "nltk is required to compute BLEU scores. Install it with `py -3 -m pip install nltk`."
        ) from exc

    common_ids = sorted(set(references) & set(predictions))
    if not common_ids:
        raise ValueError("references and predictions must share at least one image id")

    refs = [[_tokenize_for_metrics(caption) for caption in references[image_id]] for image_id in common_ids]
    hyps = [_tokenize_for_metrics(predictions[image_id]) for image_id in common_ids]
    smooth = SmoothingFunction().method1

    return {
        "bleu_1": corpus_bleu(refs, hyps, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smooth),
        "bleu_2": corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=smooth),
        "bleu_3": corpus_bleu(refs, hyps, weights=(1 / 3, 1 / 3, 1 / 3, 0.0), smoothing_function=smooth),
        "bleu_4": corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
    }


def compute_meteor_score(
    references: Dict[str, List[str]],
    predictions: Dict[str, str],
) -> float:
    """Compute the mean METEOR score over aligned image ids."""
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError as exc:  # pragma: no cover - dependency may be installed later
        raise ImportError(
            "nltk is required to compute METEOR. Install it with `py -3 -m pip install nltk`."
        ) from exc

    common_ids = sorted(set(references) & set(predictions))
    if not common_ids:
        raise ValueError("references and predictions must share at least one image id")

    scores = []
    try:
        for image_id in common_ids:
            reference_tokens = [_tokenize_for_metrics(caption) for caption in references[image_id]]
            prediction_tokens = _tokenize_for_metrics(predictions[image_id])
            scores.append(meteor_score(reference_tokens, prediction_tokens))
    except LookupError as exc:  # pragma: no cover - depends on local nltk data
        raise LookupError(
            "METEOR requires NLTK data resources. Try `py -3 -m nltk.downloader wordnet omw-1.4`."
        ) from exc

    return sum(scores) / len(scores)


def evaluate_captions(
    references: Dict[str, List[str]],
    predictions: Dict[str, str],
) -> Dict[str, float]:
    """Compute all shared captioning metrics in one place."""
    results = compute_bleu_scores(references, predictions)
    results["meteor"] = compute_meteor_score(references, predictions)
    return results
