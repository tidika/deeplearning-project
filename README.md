# Deep Learning Project: Image Captioning Comparison

This project compares two image captioning approaches on Flickr30k:

- a CNN + Transformer decoder baseline
- a Bottom-Up / Top-Down style object-attention model

The comparison is designed to use the same data pipeline, vocabulary, image preprocessing, and evaluation setup for both models.

## Project Layout

```text
deeplearning-project/
|-- README.md
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
|   `-- sanity_check.py
`-- src/
    |-- data/
    |   |-- __init__.py
    |   |-- constants.py
    |   |-- dataset.py
    |   |-- preprocess.py
    |   |-- splits.py
    |   `-- vocab.py
    `-- eval/
        |-- __init__.py
        `-- metrics.py
```

Note: `data/` is ignored by Git in this repo, so committed split files should live under `metadata/splits/`, not under `data/`.

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

## Image Preprocessing Defaults

Unless a model owner provides a custom image transform, the shared loader will:

- convert images to RGB
- resize images to `224 x 224`
- scale pixel values to `[0, 1]`
- normalize channels with ImageNet statistics

Default normalization:

- mean: `[0.485, 0.456, 0.406]`
- std: `[0.229, 0.224, 0.225]`

## Evaluation JSON Format

References JSON must map each image id to a list of reference captions:

```json
{
  "1000092795.jpg": [
    "Two young guys with shaggy hair look at their hands while hanging out in the yard .",
    "Two friends enjoy time spent together ."
  ]
}
```

Predictions JSON must map each image id to one generated caption string:

```json
{
  "1000092795.jpg": "Two young men stand together in a yard ."
}
```

Templates:

- [references.example.json](configs/references.example.json)
- [predictions.example.json](configs/predictions.example.json)

## Quick Start

1. Put Flickr30k images under [data/Images](data/Images).
2. Keep captions in [data/captions.txt](data/captions.txt).
3. Generate fixed splits:

```powershell
py -3 scripts/generate_splits.py
```

4. Build the shared vocabulary from the training split:

```powershell
py -3 scripts/build_vocab.py --min-word-freq 5
```

5. Inspect one batch from the shared DataLoader:

```powershell
py -3 scripts/inspect_dataloader.py --split-path metadata/splits/train.txt --batch-size 4
```

6. Evaluate captions from JSON files:

```powershell
py -3 scripts/evaluate_captions.py --references-path configs/references.example.json --predictions-path configs/predictions.example.json
```

7. Run the full sanity check:

```powershell
py -3 scripts/sanity_check.py
```

## Key Files

- Split files: [metadata/splits](metadata/splits)
- Vocabulary files: [metadata/vocab](metadata/vocab)
- Shared dataset loader: [dataset.py](src/data/dataset.py)
- Shared evaluation metrics: [metrics.py](src/eval/metrics.py)
- Sanity-check output: [sanity_check_summary.json](metadata/sanity_check_summary.json)

## Dependencies

BLEU and METEOR require `nltk`:

```powershell
py -3 -m pip install nltk
```

METEOR may also require NLTK data resources:

```powershell
py -3 -m nltk.downloader wordnet omw-1.4
```

## Notes For The Report

The `Data + Evaluation` section should cover:

- dataset summary
- image split methodology
- caption preprocessing pipeline
- vocabulary rules and thresholds
- padding and special token handling
- image resizing and normalization choices
- evaluation metrics and why they were chosen
- any known data quality issues or limitations
