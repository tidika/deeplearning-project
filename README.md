# Deep Learning Project: Image Captioning Comparison

This project compares two image captioning approaches on Flickr30k:

- a CNN + Transformer decoder baseline
- a Bottom-Up / Top-Down style object-attention model

The comparison is designed to use the same data pipeline, vocabulary, image preprocessing, and evaluation setup for both models.

## Project Layout

```text
deeplearning-project/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- data_config.example.json
|   |-- predictions.example.json
|   `-- references.example.json
|-- data/
|   |-- Images/
|   `-- captions.txt
|-- metadata/
|   |-- sanity_check_summary.json
|   |-- splits/
|   |   |-- README.md
|   |   |-- split_summary.json
|   |   |-- test.txt
|   |   |-- train.txt
|   |   `-- val.txt
|   `-- vocab/
|       |-- README.md
|       |-- flickr30k_vocab.json
|       `-- flickr30k_vocab_stats.json
|-- scripts/
|   |-- build_vocab.py
|   |-- evaluate_captions.py
|   |-- generate_splits.py
|   |-- inspect_dataloader.py
|   |-- run_checks.py
|   `-- sanity_check.py
|-- tests/
|   |-- test_data_evaluation.py
|   `-- test_lightweight_checks.py
|-- .github/
|   `-- workflows/
|       `-- ci.yml
`-- src/
    |-- data/
    |   |-- __init__.py
    |   |-- constants.py
    |   |-- dataset.py
    |   |-- preprocess.py
    |   |-- splits.py
    |   `-- vocab.py
    |-- eval/
    |   |-- __init__.py
    |   `-- metrics.py
    `-- project_config.py
```

Note: `data/` is ignored by Git in this repo, so committed split files should live under `metadata/splits/`, not under `data/`.

## Shared Configuration

The data and evaluation scripts use [data_config.example.json](configs/data_config.example.json) as the shared source of default paths and settings.

Config-backed settings include:

- captions path
- image directory
- split output directory
- vocabulary output directory
- train / val / test ratios
- random seed
- `min_word_freq`
- `max_caption_length`
- image size
- image normalization
- example evaluation file paths

All major scripts still allow CLI overrides, but their defaults come from the shared config file.

## Shared Interface

Dataset sample fields:

- `image_id`
- `image_path`
- `image`
- `caption_text`
- `caption_ids`
- `target_ids`
- `length`

Evaluation inputs:

- references: `dict[str, list[str]]`
- predictions: `dict[str, str]`

## Caption Preprocessing Rules

The shared caption preprocessing pipeline is fixed so both models consume the same tokenized data:

- trim leading and trailing whitespace
- lowercase all caption text
- surround these punctuation marks with spaces before tokenization:
  `. , ! ? ; : " ( )`
- collapse repeated whitespace to a single space
- tokenize by whitespace after normalization
- add `<start>` and `<end>` after tokenization when building model inputs
- reserve `<pad>`, `<start>`, `<end>`, and `<unk>` with fixed ids `0, 1, 2, 3`
- build the vocabulary from the training split only
- use `min_word_freq = 5`
- use `max_caption_length = 40`, where the limit includes both boundary tokens

## Image Preprocessing Defaults

Unless a model owner provides a custom image transform, the shared loader will:

- convert images to RGB
- resize images to `224 x 224`
- scale pixel values to `[0, 1]`
- normalize channels with ImageNet statistics

Default normalization:

- mean: `[0.485, 0.456, 0.406]`
- std: `[0.229, 0.224, 0.225]`

## Evaluation Contract

The shared evaluation pipeline computes corpus-level metrics over the exact set of image ids present in both files. By default, references and predictions must contain the same image ids; partial prediction files are treated as invalid rather than silently scoring only the overlap.

Metrics:

- BLEU-1
- BLEU-2
- BLEU-3
- BLEU-4
- METEOR

References JSON must map each image id to a non-empty list of reference captions:

```json
{
  "1000092795.jpg": [
    "Two young guys with shaggy hair look at their hands while hanging out in the yard .",
    "Two friends enjoy time spent together ."
  ]
}
```

Predictions JSON must map each image id to exactly one generated caption string:

```json
{
  "1000092795.jpg": "Two young men stand together in a yard ."
}
```

Templates:

- [references.example.json](configs/references.example.json)
- [predictions.example.json](configs/predictions.example.json)

## Quick Start

1. Install dependencies:

```powershell
py -3 -m pip install -r requirements.txt
```

2. Put Flickr30k images under [data/Images](data/Images).
3. Keep captions in [data/captions.txt](data/captions.txt).
4. Generate fixed splits:

```powershell
py -3 scripts/generate_splits.py --config configs/data_config.example.json
```

5. Build the shared vocabulary from the training split:

```powershell
py -3 scripts/build_vocab.py --config configs/data_config.example.json
```

6. Inspect one batch from the shared DataLoader:

```powershell
py -3 scripts/inspect_dataloader.py --config configs/data_config.example.json
```

7. Evaluate captions from JSON files:

```powershell
py -3 scripts/evaluate_captions.py --config configs/data_config.example.json
```

8. Run the full local verification suite:

```powershell
py -3 scripts/run_checks.py --config configs/data_config.example.json
```

## Validation

Full local verification:

```powershell
py -3 scripts/run_checks.py --config configs/data_config.example.json
```

Direct sanity check only:

```powershell
py -3 scripts/sanity_check.py --config configs/data_config.example.json
```

Full dataset-dependent unit tests:

```powershell
py -3 -m unittest tests.test_data_evaluation
```

Lightweight tests that do not require the local Flickr30k dataset:

```powershell
py -3 -m unittest tests.test_lightweight_checks
```

## CI

A minimal GitHub Actions workflow is provided at [ci.yml](.github/workflows/ci.yml). It runs lightweight checks that do not depend on the local Flickr30k dataset:

- dependency installation
- NLTK resource download
- Python compile checks
- `tests.test_lightweight_checks`

## Key Files

- Shared config: [data_config.example.json](configs/data_config.example.json)
- Split files: [metadata/splits](metadata/splits)
- Vocabulary files: [metadata/vocab](metadata/vocab)
- Shared dataset loader: [dataset.py](src/data/dataset.py)
- Shared evaluation metrics: [metrics.py](src/eval/metrics.py)
- Local verification entrypoint: [run_checks.py](scripts/run_checks.py)
- Sanity-check output: [sanity_check_summary.json](metadata/sanity_check_summary.json)
- Full dataset-dependent tests: [test_data_evaluation.py](tests/test_data_evaluation.py)
- Lightweight CI tests: [test_lightweight_checks.py](tests/test_lightweight_checks.py)

## Dependencies

METEOR may also require NLTK data resources:

```powershell
py -3 -m nltk.downloader wordnet omw-1.4
```
