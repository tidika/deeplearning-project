# CNN + Transformer Captioner — Design Notes (for report)

This document records the main design choices and known issues for the **Transformer** branch (Section 3: CNN encoder + Transformer decoder).

## Commands

After placing Flickr30k under `data/` and building splits + vocabulary (see the root `README.md`):

```bash
# Train (edit `configs/transformer_caption.example.json` for epochs, batch size, device)
python scripts/train_transformer_caption.py --config configs/transformer_caption.example.json

# Greedy captions on val/test → JSON for shared evaluation
python scripts/generate_captions_transformer.py --config configs/transformer_caption.example.json --split val
python scripts/evaluate_captions.py --config configs/data_config.example.json \
  --references-path path/to/references_val.json \
  --predictions-path artifacts/transformer_caption/predictions_val.json
```

Build reference JSONs from the val/test splits with `scripts/` helpers if your team adds them, or assemble `dict[image_id, list[str]]` from `captions.txt` to match the evaluation contract.

## Architecture

- **Visual encoder**: ResNet-50 backbone pretrained on ImageNet (`torchvision.models.resnet50`), **global average pooling and FC removed**, so the output is a **spatial grid** of shape \(7 \times 7\) at 224×224 input, with 2048 channels per location.
- **Feature projection**: A linear layer maps each spatial vector from 2048 to `d_model` (default 512). **Learned positional embeddings** are added for the 49 image tokens.
- **Text side**: Token embeddings + **learned positional embeddings** for caption positions (max length aligned with `max_caption_length` in the shared data config).
- **Decoder**: PyTorch `nn.TransformerDecoder` with `TransformerDecoderLayer` (**pre-norm**, GELU, dropout). Each layer has **causal (self-attention) mask** on captions and **cross-attention** from caption tokens to image tokens (implemented by `TransformerDecoder` over `memory`).

## Training

- **Objective**: Categorical cross-entropy on next-token prediction, with `ignore_index` set to `<pad>` (id 0).
- **Targets**: `target_ids` are a **one-step shift** of `caption_ids` (standard language modeling alignment).
- **Optimization**: AdamW on parameters with `requires_grad=True`; when `freeze_backbone` is true, only projection + decoder + embeddings train (faster, less GPU memory).
- **Regularization**: Dropout on decoder; optional weight decay from config.

## Checkpoints and outputs

- Each epoch saves `checkpoint_epoch_XXX.pt` and `checkpoint_latest.pt` under `checkpoint_dir` (see `configs/transformer_caption.example.json`).
- `training_log.jsonl` appends one JSON object per epoch (train/val loss, wall time).
- `sample_captions_val.json` holds a few **validation** greedy samples for qualitative review.

## Evaluation

- Run `scripts/generate_captions_transformer.py` to produce a **predictions JSON** mapping image id → one string, then use the shared `scripts/evaluate_captions.py` with the same reference format as the rest of the project (BLEU / METEOR).

## Issues and limitations

- **Greedy decoding** only in the provided scripts; beam search can improve BLEU at extra cost.
- **Single caption per image** per step in the DataLoader (random caption on train, first caption on val/test) to match common practice; all five references are still used at evaluation time if you build references JSON accordingly.
- **Frozen ResNet** is a strong baseline for speed; unfreezing or using a smaller learning rate on the backbone may improve quality but increases training time and overfitting risk on Flickr30k.
- **Download**: First run with `use_pretrained_encoder: true` downloads ImageNet weights for ResNet-50 (torchvision cache).

## Reproducibility

- Seeding: Python `random`, `torch`, and CUDA (if available) are set from `seed` in the transformer config.
- For full reproducibility on GPU, some operations remain nondeterministic unless you enable deterministic algorithms in PyTorch (not enabled by default here).
