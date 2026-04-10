"""Train the CNN + Transformer captioning model on Flickr30k."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.constants import END_TOKEN, START_TOKEN
from src.data.dataset import create_caption_dataloader
from src.data.splits import load_split_file
from src.data.vocab import Vocabulary
from src.models.cnn_transformer_captioner import CNNTransformerCaptioner
from src.project_config import get_config_value, get_image_normalization, get_image_size, load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transformer_caption.example.json", help="Transformer training config JSON.")
    parser.add_argument("--data-config", default=None, help="Override shared data config path.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. Use 0 in Docker/K8s if you see /dev/shm or 'No space left on device' errors.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: CNNTransformerCaptioner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_id: int,
    grad_clip: float,
    train: bool,
    desc: str = "",
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[misc, assignment]

    batches = loader
    if tqdm is not None:
        batches = tqdm(loader, desc=desc, leave=True, dynamic_ncols=True, mininterval=0.5)

    print(
        f"[train] {desc}: reading first batch from disk (on NFS this can take minutes; bar stays at 0% until then)...",
        flush=True,
    )

    for batch in batches:
        images = batch["images"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(images, caption_ids)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=pad_id,
                reduction="sum",
            )

        if train:
            loss_sum.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        n = int((target_ids != pad_id).sum().item())
        batch_loss = float(loss_sum.item()) / max(n, 1)
        total_loss += float(loss_sum.item())
        total_tokens += max(n, 1)

        if tqdm is not None and hasattr(batches, "set_postfix"):
            batches.set_postfix(loss=f"{batch_loss:.4f}")

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def _sample_captions(
    model: CNNTransformerCaptioner,
    loader: DataLoader,
    vocab: Vocabulary,
    device: torch.device,
    max_batches: int,
) -> List[Dict[str, str]]:
    model.eval()
    start_id = vocab.word2idx[START_TOKEN]
    end_id = vocab.word2idx[END_TOKEN]
    rows: List[Dict[str, str]] = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        images = batch["images"].to(device)
        ids_out = model.generate(images, start_id=start_id, end_id=end_id)
        for i, image_id in enumerate(batch["image_ids"]):
            text = vocab.decode(ids_out[i].tolist())
            rows.append({"image_id": image_id, "caption": text})
    return rows


def main() -> None:
    args = parse_args()
    cfg_path = REPO_ROOT / args.config
    cfg = _load_json(cfg_path)

    data_config_path = REPO_ROOT / (args.data_config or cfg["data_config"])
    data_cfg = load_project_config(data_config_path)

    captions_path = REPO_ROOT / get_config_value(data_cfg, "captions_path", "data/captions.txt")
    images_dir = REPO_ROOT / get_config_value(data_cfg, "images_dir", "data/Images")
    max_caption_length = int(cfg.get("max_caption_length") or get_config_value(data_cfg, "max_caption_length", 40))

    vocab_path = REPO_ROOT / cfg["vocab_path"]
    train_split = REPO_ROOT / cfg["train_split"]
    val_split = REPO_ROOT / cfg["val_split"]

    epochs = int(args.epochs if args.epochs is not None else cfg["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else cfg["batch_size"])
    num_workers = int(args.num_workers) if args.num_workers is not None else int(cfg["num_workers"])
    if num_workers > 0:
        print(
            f"[train] num_workers={num_workers} — if you see /dev/shm or shared memory errors (Docker/K8s), set num_workers to 0 in config.",
            flush=True,
        )
    lr = float(cfg["learning_rate"])
    weight_decay = float(cfg["weight_decay"])
    grad_clip = float(cfg["grad_clip"])
    seed = int(cfg["seed"])

    device_str = args.device or cfg.get("device") or "cpu"
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.", file=sys.stderr)
        print(
            "Hint: PyTorch CUDA build must match your NVIDIA driver. See https://pytorch.org/get-started/locally/",
            file=sys.stderr,
        )
        device = torch.device("cpu")

    print(
        f"[train] device={device} | cuda_available={torch.cuda.is_available()}"
        + (f" | gpu={torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""),
        flush=True,
    )

    _set_seed(seed)

    vocab = Vocabulary.load(vocab_path)
    pad_id = vocab.word2idx["<pad>"]
    vocab_size = len(vocab.word2idx)

    image_size = get_image_size(data_cfg)
    image_mean, image_std = get_image_normalization(data_cfg)

    train_ids = load_split_file(train_split)
    val_ids = load_split_file(val_split)

    train_loader = create_caption_dataloader(
        captions_path=captions_path,
        images_dir=images_dir,
        split_image_ids=train_ids,
        vocab=vocab,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        max_caption_length=max_caption_length,
        image_size=int(image_size[0]),
        image_mean=image_mean,
        image_std=image_std,
        random_caption=True,
        seed=seed,
    )
    val_loader = create_caption_dataloader(
        captions_path=captions_path,
        images_dir=images_dir,
        split_image_ids=val_ids,
        vocab=vocab,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        max_caption_length=max_caption_length,
        image_size=int(image_size[0]),
        image_mean=image_mean,
        image_std=image_std,
        random_caption=False,
        seed=seed,
    )

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    print(
        f"[train] train_batches={n_train_batches} val_batches={n_val_batches} batch_size={batch_size} epochs={epochs}",
        flush=True,
    )

    model = CNNTransformerCaptioner(
        vocab_size=vocab_size,
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_decoder_layers=int(cfg["num_decoder_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
        max_caption_len=max_caption_length,
        freeze_backbone=bool(cfg["freeze_backbone"]),
        use_pretrained_encoder=bool(cfg.get("use_pretrained_encoder", True)),
    ).to(device)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)

    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = REPO_ROOT / resume_path
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1

    checkpoint_dir = REPO_ROOT / cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = REPO_ROOT / cfg["log_path"]
    if log_path.exists():
        log_path.unlink()

    sample_path = REPO_ROOT / cfg["sample_captions_path"]
    eval_batches = max(1, (int(cfg.get("eval_images_for_samples", 8)) + batch_size - 1) // batch_size)

    print(
        "[train] Tip: images on slow network storage (NFS) = long wait before tqdm moves. "
        "Copying data to local disk speeds this up.",
        flush=True,
    )

    for epoch in range(start_epoch, start_epoch + epochs):
        ep_num = epoch - start_epoch + 1
        print(f"[train] === epoch {ep_num}/{epochs} (checkpoint index {epoch}) ===", flush=True)
        t0 = time.perf_counter()
        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            pad_id,
            grad_clip,
            train=True,
            desc=f"train e{epoch}",
        )
        val_loss = _run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            pad_id,
            grad_clip,
            train=False,
            desc=f"val e{epoch}",
        )
        elapsed = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "seconds": round(elapsed, 3),
        }
        print(json.dumps(row, ensure_ascii=True), flush=True)
        _append_jsonl(log_path, row)

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "data_config_path": str(data_config_path),
        }
        torch.save(state, checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(state, checkpoint_dir / "checkpoint_latest.pt")

        samples = _sample_captions(model, val_loader, vocab, device, max_batches=eval_batches)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_path.open("w", encoding="utf-8") as handle:
            json.dump(samples, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
