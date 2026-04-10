# CNN + Transformer Captioning — Insights Report

*Generated for project documentation; metrics from validation / test evaluation on Flickr30k (full `archive/` pipeline).*

## 1. Summary

We trained a **CNN + Transformer decoder** captioning model: **ResNet-50** (ImageNet) outputs **7×7 spatial features** (49 tokens), projected to `d_model`, consumed by a **PyTorch `TransformerDecoder`** with **causal self-attention** and **cross-attention** to image tokens. Training used **teacher forcing** and **token-level cross-entropy**; inference used **greedy decoding**.

## 2. Quantitative Results

Corpus-level metrics (shared `evaluate_captions.py`; references built from `archive/captions.txt` + split files; predictions from `checkpoint_latest.pt`).

| Split | Images | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
|-------|--------|--------|--------|--------|--------|--------|
| **Val** | 3178 | 0.592 | 0.375 | 0.240 | 0.158 | 0.363 |
| **Test** | 3179 | 0.587 | 0.367 | 0.233 | 0.153 | 0.358 |

Test scores are slightly lower than validation but in the same range, which is **expected** (held-out images, sampling noise) and suggests **no severe overfitting** to the val split alone.

## 3. Training Observations

- **Loss:** Training loss decreased over 10 epochs (final train ~2.79, val ~3.23 in the logged run); validation tracked training without a large divergence in the final epoch.
- **Throughput:** When data lived on **NFS**, the first batches were **I/O bound**; `num_workers=0` avoided Docker `/dev/shm` limits but limited loader parallelism.
- **GPU:** After aligning **PyTorch CUDA build** with the cluster driver, **CUDA** was used (**NVIDIA RTX A5000**). Initial driver / wheel mismatch caused fallback to CPU until resolved.

## 4. Issues & Mitigations

| Issue | Mitigation |
|--------|------------|
| `/dev/shm` / shared memory errors with DataLoader workers | Set **`num_workers`** to **0** in config (or use larger shm in Kubernetes). |
| Slow first epoch on NFS | Expected; optional copy of `archive/Images` to **local disk** on the node. |
| METEOR requires NLTK data | `python3 -m nltk.downloader wordnet omw-1.4` |
| No `zip` binary on server | Use **`scripts/make_zip.py`** or **`package_training_results.sh`** (falls back to Python). |

## 5. Artifacts (for report / reproducibility)

Under `artifacts/transformer_caption/` (also bundled in `transformer_training_results.zip` locally):

- `checkpoint_epoch_*.pt`, `checkpoint_latest.pt`
- `training_log.jsonl`
- `sample_captions_val.json`
- `references_{val,test}.json`, `predictions_{val,test}.json`, `eval_{val,test}_summary.json` (when generated)

**Checkpoints are large** — store outside Git (see README); keep JSON summaries in the repo if desired.

## 6. Limitations & Future Work

- **Greedy decoding** only; **beam search** may improve BLEU.
- **Single caption per image per step** in the DataLoader (train: random; val/test: first caption); evaluation still uses **five references** per image.
- **Frozen ResNet** option trades speed / memory for possible accuracy gains from partial fine-tuning.

## 7. References (code & docs)

- Model: `src/models/cnn_transformer_captioner.py`
- Training: `scripts/train_transformer_caption.py`
- Evaluation: `scripts/evaluate_captions.py`, `scripts/build_references_json.py`, `scripts/generate_captions_transformer.py`
- Design notes: `docs/transformer_model_design.md`
- English report section draft: `docs/report_section3_transformer_EN.md`
