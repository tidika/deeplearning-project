# Image Captioning: BUTD vs Transformer

A comparative study of two image captioning architectures on the Flickr30k dataset.

| Model | Visual Features | Decoder | Attention |
|---|---|---|---|
| **BUTD** | 36 object regions (Faster R-CNN, 1024-dim) | 2-layer LSTM | Additive (Bahdanau) |
| **Transformer** | 49 spatial patches (ResNet-50, 2048-dim) | Transformer decoder | Multi-head cross-attention |

## Results

Evaluated on the Flickr30k validation split (3,178 images):

| Metric | BUTD | Transformer |
|---|---|---|
| BLEU-1 | 0.6318 | **0.6686** |
| BLEU-2 | 0.4175 | **0.4611** |
| BLEU-3 | 0.2768 | **0.3183** |
| BLEU-4 | 0.1854 | **0.2216** |
| METEOR | 0.3920 | **0.4230** |

The Transformer outperforms BUTD on all five metrics, with a 19.6% relative improvement on BLEU-4.

## Project Layout

```text
deeplearning-project/
|-- README.md
|-- requirements.txt
|-- configs/
|   |-- butd_val_predictions.json
|   |-- transformer_val_predictions.json
|   |-- final_results.json
|   |-- data_config.example.json
|   |-- predictions.example.json
|   `-- references.example.json
|-- data/                          <- not committed (download separately)
|   |-- Images/                    <- Flickr30k .jpg files
|   |-- captions.txt
|   |-- butd_features/             <- pre-extracted Faster R-CNN features
|   `-- transformer_features/      <- pre-extracted ResNet-50 features
|-- metadata/
|   |-- sanity_check_summary.json
|   |-- splits/
|   |   |-- train.txt
|   |   |-- val.txt
|   |   `-- test.txt
|   `-- vocab/
|       |-- flickr30k_vocab.json
|       `-- flickr30k_vocab_stats.json
|-- notebooks/
|   |-- colab_butd.ipynb           <- train BUTD model on Colab
|   |-- colab_transformer.ipynb    <- train Transformer model on Colab
|   |-- colab_results_analysis.ipynb <- run full evaluation on Colab
|   `-- results_analysis.ipynb     <- local evaluation notebook
|-- scripts/
|   |-- build_vocab.py
|   |-- extract_butd_features.py
|   |-- extract_transformer_features.py
|   |-- evaluate_captions.py
|   |-- generate_splits.py
|   |-- inspect_dataloader.py
|   |-- run_checks.py
|   `-- sanity_check.py
|-- src/
|   |-- data/
|   |   |-- butd_dataset.py
|   |   |-- transformer_dataset.py
|   |   |-- constants.py
|   |   |-- dataset.py
|   |   |-- preprocess.py
|   |   |-- splits.py
|   |   `-- vocab.py
|   |-- eval/
|   |   `-- metrics.py
|   |-- models/
|   |   |-- butd_model.py
|   |   `-- transformer_model.py
|   `-- training/
|       |-- train_butd.py
|       `-- train_transformer.py
|-- tests/
|   |-- test_data_evaluation.py
|   `-- test_lightweight_checks.py
`-- .github/
    `-- workflows/
        `-- ci.yml
```

## Reproducing the Results

### 1. Dataset

Download the Flickr30k dataset from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr30k) and place the files as follows:

```
data/Images/     <- all .jpg image files
data/captions.txt
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
```

### 3. Generate splits and vocabulary

```bash
python scripts/generate_splits.py --config configs/data_config.example.json
python scripts/build_vocab.py --config configs/data_config.example.json
```

### 4. Train the models (Google Colab recommended)

Upload the project folder to Google Drive, then open and run the Colab notebooks in order:

| Notebook | Purpose |
|---|---|
| `notebooks/colab_butd.ipynb` | Extract BUTD features and train the BUTD model |
| `notebooks/colab_transformer.ipynb` | Extract Transformer features and train the Transformer model |
| `notebooks/colab_results_analysis.ipynb` | Run full evaluation and generate all result figures |

> **Note:** Set runtime to T4 GPU before running. Both models were trained for 20 epochs. Feature extraction for BUTD takes approximately 2-3 hours; Transformer feature extraction takes approximately 2-3 minutes.

### 5. Evaluate locally

```bash
python scripts/evaluate_captions.py --config configs/data_config.example.json
```

## Data Pipeline

Both models share the same data pipeline to ensure a fair comparison:

- **Split**: 80/10/10 train/val/test, fixed seed (seed=42)
- **Vocabulary**: built from training split only, `min_word_freq=5`, vocab size=7,013
- **Special tokens**: `<pad>=0`, `<start>=1`, `<end>=2`, `<unk>=3`
- **Caption preprocessing**: lowercase, punctuation space-padded, whitespace normalized
- **Max caption length**: 40 tokens (including boundary tokens)

## Model Details

### BUTD Model

- **Encoder**: Faster R-CNN (ResNet-50 + FPN, pretrained on COCO) — top-36 object regions, 1024-dim each
- **Decoder**: 2-layer LSTM (Attention LSTM + Language LSTM)
- **Attention**: Additive (Bahdanau-style) over 36 object regions
- **Parameters**: 18,470,245
- **Training**: Adam (lr=4e-4), StepLR decay (step=5, γ=0.5), dropout=0.5, grad clip=5.0

### Transformer Model

- **Encoder**: ResNet-50 (frozen) — 7×7 spatial feature map, 2048-dim projected to 512-dim
- **Decoder**: 3 TransformerDecoderLayer blocks (masked self-attention + cross-attention)
- **Attention**: Multi-head cross-attention (8 heads) over 49 spatial locations
- **Parameters**: 20,851,557
- **Training**: Adam (lr=1e-4), StepLR decay (step=5, γ=0.5), dropout=0.1, grad clip=1.0

## Evaluation

Both models are evaluated using corpus-level BLEU-1/2/3/4 and METEOR on the validation split. Each image has 5 human-written reference captions.

```bash
python scripts/evaluate_captions.py --config configs/data_config.example.json
```

## Validation and CI

```bash
# Full local verification
python scripts/run_checks.py --config configs/data_config.example.json

# Lightweight tests (no dataset required)
python -m unittest tests.test_lightweight_checks

# Full dataset-dependent tests
python -m unittest tests.test_data_evaluation
```

A GitHub Actions workflow at [ci.yml](.github/workflows/ci.yml) runs lightweight checks on every push.

## Dependencies

- Python 3.10+
- torch==2.10.0
- torchvision
- numpy==2.3.3
- Pillow==11.3.0
- nltk==3.9.3

## Team

- Tochukwu Idika (Team Lead, BUTD model)
- Junhui Chen (Data pipeline and evaluation)
- Po Ting Lee (Transformer model)
- Randy Kang (Experiments and analysis)

Georgia Institute of Technology — Deep Learning (CS 7643), Spring 2026
