# Deep Learning Project: Image Captioning Comparison

This project compares two image captioning approaches on Flickr30k:

- a CNN + Transformer decoder baseline
- a Bottom-Up / Top-Down style object-attention model

The shared goal is to compare both models under the same data pipeline, vocabulary, and evaluation setup.

## Current Focus

The first milestone is `Data + Evaluation`. This part is the common foundation for both model tracks and should be completed before model training begins.

Primary responsibilities:

- organize Flickr30k image-caption data
- create fixed train / val / test image splits
- define caption preprocessing rules
- build a shared vocabulary with special tokens
- expose a reusable dataset / dataloader interface
- evaluate generated captions with BLEU and METEOR

## Recommended Project Layout

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
|   |-- splits/
|   |   `-- README.md
|   `-- vocab/
|       `-- README.md
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

## Data + Evaluation TODO

Short-term checklist for the owner of this section:

- [ ] confirm the final Flickr30k image root and caption file path
- [ ] verify each image maps to five captions
- [ ] create image-level train / val / test splits with a fixed random seed
- [ ] save split file lists under `metadata/splits/`
- [ ] finalize caption normalization and tokenization rules
- [ ] set `min_word_freq` and `max_caption_length`
- [ ] build the vocabulary from the training split only
- [ ] reserve `<pad>`, `<start>`, `<end>`, and `<unk>`
- [ ] implement the shared dataset and collate function
- [ ] define the exact evaluation API for model teams
- [ ] run BLEU-1 to BLEU-4 and METEOR on generated captions
- [ ] document all preprocessing and evaluation decisions for the report

## Shared Interface Contract

To keep both models comparable, both teams should consume the same outputs from the data pipeline.

Expected dataset sample fields:

- `image_id`
- `image_path`
- `image`
- `caption_text`
- `caption_ids`
- `target_ids`
- `length`

Expected evaluation inputs:

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

You can use [references.example.json](deeplearning-project/configs/references.example.json) and [predictions.example.json](deeplearning-project/configs/predictions.example.json) as templates.

## Quick Start

1. Put Flickr30k images in [data/Images](deeplearning-project/data/Images).
2. Keep captions in [data/captions.txt](deeplearning-project/data/captions.txt).
3. Run [generate_splits.py](deeplearning-project/scripts/generate_splits.py) to write fixed split files.
4. Run [build_vocab.py](deeplearning-project/scripts/build_vocab.py) to build the shared training-only vocabulary.
5. Run [inspect_dataloader.py](deeplearning-project/scripts/inspect_dataloader.py) to verify split loading, vocab loading, and batch construction.
6. Run [evaluate_captions.py](deeplearning-project/scripts/evaluate_captions.py) to evaluate model outputs with the shared metric pipeline.
7. Run [sanity_check.py](deeplearning-project/scripts/sanity_check.py) to validate the full Data + Evaluation stack in one pass.
8. Store committed shared split files under [metadata/splits](deeplearning-project/metadata/splits).
9. Store committed shared vocabulary files under [metadata/vocab](deeplearning-project/metadata/vocab).
10. Use [src/data/dataset.py](deeplearning-project/src/data/dataset.py) as the common dataset entry point.
11. Use [src/eval/metrics.py](deeplearning-project/src/eval/metrics.py) for caption evaluation.

## Script Usage

Generate deterministic image splits:

```powershell
py -3 scripts/generate_splits.py
```

Build the shared vocabulary from the training split only:

```powershell
py -3 scripts/build_vocab.py --min-word-freq 5
```

Inspect one batch from the shared DataLoader:

```powershell
py -3 scripts/inspect_dataloader.py --split-path metadata/splits/train.txt --batch-size 4
```

Evaluate captions from JSON files:

```powershell
py -3 scripts/evaluate_captions.py --references-path configs/references.example.json --predictions-path configs/predictions.example.json
```

Run the full sanity check:

```powershell
py -3 scripts/sanity_check.py
```

Note: BLEU and METEOR require `nltk`. Install it with:

```powershell
py -3 -m pip install nltk
```

METEOR may also require NLTK data resources:

```powershell
py -3 -m nltk.downloader wordnet omw-1.4
```

## Notes For The Report

The `Data + Evaluation` section should eventually cover:

- dataset summary
- image split methodology
- caption preprocessing pipeline
- vocabulary rules and thresholds
- padding and special token handling
- image resizing and normalization choices
- evaluation metrics and why they were chosen
- any known data quality issues or limitations
