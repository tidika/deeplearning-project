# Deep Learning Project: Image Captioning Comparison

This project compares two image captioning approaches on Flickr30k:

- a **CNN + Transformer decoder** baseline (implemented in this repo)
- a Bottom-Up / Top-Down style object-attention model

The comparison is designed to use the same data pipeline, vocabulary, image preprocessing, and evaluation setup for both models.

## Project Layout

```text
deeplearning-project/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- data_config.example.json
|   |-- data_archive.example.json    # optional: captions/images under archive/
|   |-- transformer_caption.example.json
|   |-- predictions.example.json
|   `-- references.example.json
|-- archive/                         # optional local path for Flickr30k (gitignored if present)
|-- metadata/
|   |-- splits/   (train / val / test lists)
|   `-- vocab/
|-- scripts/
|   |-- train_transformer_caption.py
|   |-- generate_captions_transformer.py
|   |-- build_references_json.py
|   |-- evaluate_captions.py
|   |-- build_vocab.py, generate_splits.py, ...
|   |-- package_code_for_server.sh, package_training_results.sh, make_zip.py
|   `-- ...
|-- src/
|   |-- data/     (dataset, vocab, splits, preprocessing)
|   |-- eval/     (BLEU / METEOR)
|   `-- models/   (CNN + Transformer captioner)
|-- tests/
|-- docs/         (design notes, server setup, insights report)
`-- artifacts/    # training outputs (gitignored)
```

Note: `data/` and `archive/` are typically **gitignored** when they contain large datasets. Committed **splits** and **vocabulary** live under `metadata/`.

## CNN + Transformer (baseline)

- **Model:** `src/models/cnn_transformer_captioner.py` — ResNet-50 spatial features + `nn.TransformerDecoder`.
- **Train:** `python scripts/train_transformer_caption.py --config configs/transformer_caption.example.json`
- **Config:** Point `data_config` to `configs/data_archive.example.json` if images live under `archive/` (see that file for paths).
- **Greedy inference:** `python scripts/generate_captions_transformer.py --config configs/transformer_caption.example.json --split val`
- **References JSON for eval:** `python scripts/build_references_json.py --config configs/data_archive.example.json --split val`
- **Metrics:** `python scripts/evaluate_captions.py --references-path ... --predictions-path ...`

**Results & analysis:** see [docs/TRANSFORMER_INSIGHTS_REPORT.md](docs/TRANSFORMER_INSIGHTS_REPORT.md) (metrics, issues, artifacts).

**Server / training notes:** [docs/server_training_and_report.md](docs/server_training_and_report.md).

## Shared Configuration

The data and evaluation scripts use [data_config.example.json](configs/data_config.example.json) (or [data_archive.example.json](configs/data_archive.example.json) for `archive/`) as the shared source of default paths and settings.

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

```bash
python3 -m pip install -r requirements.txt
```

2. Place Flickr30k under `data/` **or** `archive/` (see `configs/data_config.example.json` or `configs/data_archive.example.json`).
3. Generate fixed splits and vocabulary (example for default config):

```bash
python3 scripts/generate_splits.py --config configs/data_config.example.json
python3 scripts/build_vocab.py --config configs/data_config.example.json
```

4. Inspect one batch:

```bash
python3 scripts/inspect_dataloader.py --config configs/data_config.example.json
```

5. Evaluate captions from JSON files:

```bash
python3 scripts/evaluate_captions.py --config configs/data_config.example.json
```

6. Run lightweight tests:

```bash
python3 -m unittest tests.test_lightweight_checks tests.test_transformer_model
```

## Validation

Full local verification (requires local Flickr30k under expected paths):

```bash
python3 scripts/run_checks.py --config configs/data_config.example.json
```

Full dataset-dependent unit tests:

```bash
python3 -m unittest tests.test_data_evaluation
```

## CI

Workflow: [.github/workflows/ci.yml](.github/workflows/ci.yml) — lightweight checks, `tests.test_lightweight_checks`, `tests.test_transformer_model`.

## Key Files

- Shared config: [data_config.example.json](configs/data_config.example.json)
- Transformer training config: [transformer_caption.example.json](configs/transformer_caption.example.json)
- Split files: [metadata/splits](metadata/splits)
- Vocabulary files: [metadata/vocab](metadata/vocab)
- Shared dataset loader: [src/data/dataset.py](src/data/dataset.py)
- Shared evaluation metrics: [src/eval/metrics.py](src/eval/metrics.py)
- Insights (metrics & issues): [docs/TRANSFORMER_INSIGHTS_REPORT.md](docs/TRANSFORMER_INSIGHTS_REPORT.md)

## Dependencies

METEOR requires NLTK data:

```bash
python3 -m nltk.downloader wordnet omw-1.4
```

PyTorch: install a **torch / torchvision** build that matches your **CUDA** version if using GPU ([pytorch.org](https://pytorch.org)).

## Git: what not to commit

Large or machine-local paths are listed in `.gitignore`, including:

- `artifacts/` (checkpoints, logs)
- `archive/` and downloaded `.zip` bundles
- `.venv/`

Keep **checkpoints** and **training zips** outside the repository; store evaluation **JSON summaries** or link to shared storage if required for grading.
