# 3. Transformer Model

*Paste or adapt the following into your report. Figures and tables can reference paths under `artifacts/transformer_caption/` after training.*

---

## 3. Transformer Model

This section describes the **CNN + Transformer decoder** image captioning model trained on Flickr30k. The model follows a standard encoder–decoder design: a **convolutional backbone** extracts **spatial visual features** from the input image; a **Transformer decoder** then generates the caption word by word using **causal (masked) self-attention** over text tokens and **cross-attention** to the visual features.

### 3.1 Visual Encoder and Spatial Features

We use a **pretrained ResNet-50** (ImageNet weights via `torchvision`) as the CNN encoder. The final global pooling layer and the classification head are **removed**, so the network outputs a **spatial feature map** of shape \(7 \times 7 \times 2048\) for an input resolution of \(224 \times 224\). Each of the **49 spatial locations** is treated as one visual token. A linear layer projects these vectors to the model dimension \(d_{\text{model}}\) (e.g., 512). **Learned positional embeddings** are added for these 49 image tokens so the decoder can distinguish different spatial positions.

*Design decision:* Freezing the ResNet backbone (optional in config) reduces GPU memory and training time while the decoder and projection layers adapt to the captioning task.

### 3.2 Transformer Decoder

The text side uses **token embeddings** and **learned positional embeddings** for caption positions (up to the shared maximum caption length, including `<start>` and `<end>`). The core is a stack of **`TransformerDecoderLayer`** blocks (PyTorch), each with:

- **Masked self-attention** on caption tokens: a **causal mask** prevents positions from attending to future tokens, which is required for **autoregressive** training and generation.
- **Cross-attention**: queries come from the caption representation; keys and values come from the **image feature sequence**. This lets each word attend to relevant regions of the spatial grid.
- Feed-forward sublayers, residual connections, pre-norm, GELU activation, and dropout, as in common Transformer implementations.

The output of the decoder is projected to **vocabulary size** with a linear layer. Training minimizes **token-level cross-entropy** for next-token prediction, with padding positions ignored.

### 3.3 Training and Testing

**Training** uses **teacher forcing**: the decoder receives the ground-truth token sequence (including `<start>` … `<end>`), and the loss is computed on **shifted targets** (standard language-model alignment). Optimization uses **AdamW** with learning rate and weight decay from the shared JSON config.

**Testing / evaluation** uses **greedy decoding** by default: starting from `<start>`, the model repeatedly appends the argmax token until `<end>` or a length limit. Generated captions are written to a **predictions JSON** (image id → string). **Corpus-level metrics** (e.g., BLEU-1–4, METEOR) are computed with the shared evaluation script against reference captions on the validation or test split.

### 3.4 Checkpoints and Sample Captions

During training we **save checkpoints** after each epoch (`checkpoint_epoch_XXX.pt` and `checkpoint_latest.pt`) containing model weights, optimizer state, and config for reproducibility. We also save **`training_log.jsonl`** (train/validation loss per epoch). For qualitative analysis, **sample captions** on a subset of validation images are stored as JSON (e.g., `sample_captions_val.json`).

### 3.5 Design Decisions and Issues

| Topic | Decision / issue |
|--------|------------------|
| Backbone | ResNet-50 for strong ImageNet features; spatial map instead of a single vector to support cross-attention. |
| Decoder | Off-the-shelf `nn.TransformerDecoder` implements masked self-attention and cross-attention in a well-tested form. |
| Decoding | Greedy search is simple and fast; **beam search** is not enabled in the default script (possible improvement for BLEU). |
| Data | Vocabulary built from the **training split only**; shared preprocessing with the other model in the project. |
| Compute | Large batch size or full fine-tuning of the CNN may cause **OOM** on small GPUs; reducing batch size or keeping the backbone frozen mitigates this. |
| Manual upload | Full image folders are large; uploading **code + metadata** as one zip and **images** separately (or split archives) may be necessary on bandwidth-limited portals. |

*Replace bracketed examples with your measured losses, metric numbers, and 1–2 concrete failure cases from `sample_captions_val.json` after experiments.*

For **manual-only upload** workflows and splitting large `archive/` folders, see `docs/server_training_and_report.md`.
