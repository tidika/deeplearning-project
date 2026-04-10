"""Generate captions with a trained CNN + Transformer checkpoint (JSON predictions file)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.constants import START_TOKEN
from src.data.dataset import create_caption_dataloader
from src.data.splits import load_split_file
from src.data.vocab import Vocabulary
from src.models.cnn_transformer_captioner import CNNTransformerCaptioner
from src.project_config import get_config_value, get_image_normalization, get_image_size, load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transformer_caption.example.json", help="Transformer config JSON used for training.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (defaults to checkpoint_latest.pt).")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="Which split file to run.")
    parser.add_argument("--output", default=None, help="Output predictions JSON path.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = REPO_ROOT / args.config
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg: Dict[str, Any] = json.load(handle)

    data_config_path = REPO_ROOT / cfg["data_config"]
    data_cfg = load_project_config(data_config_path)

    captions_path = REPO_ROOT / get_config_value(data_cfg, "captions_path", "data/captions.txt")
    images_dir = REPO_ROOT / get_config_value(data_cfg, "images_dir", "data/Images")
    max_caption_length = int(cfg.get("max_caption_length") or get_config_value(data_cfg, "max_caption_length", 40))

    split_key = f"{args.split}_split"
    split_path = REPO_ROOT / cfg[split_key]
    vocab_path = REPO_ROOT / cfg["vocab_path"]

    checkpoint_path = REPO_ROOT / (args.checkpoint or Path(cfg["checkpoint_dir"]) / "checkpoint_latest.pt")
    if args.output:
        output_path = REPO_ROOT / args.output
    else:
        key = "predictions_val_path" if args.split == "val" else "predictions_test_path"
        output_path = REPO_ROOT / cfg[key]

    batch_size = int(args.batch_size or cfg["batch_size"])
    device_str = args.device or cfg.get("device") or "cpu"
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    vocab = Vocabulary.load(vocab_path)
    vocab_size = len(vocab.word2idx)
    start_id = vocab.word2idx[START_TOKEN]
    end_id = vocab.word2idx["<end>"]

    image_size = get_image_size(data_cfg)
    image_mean, image_std = get_image_normalization(data_cfg)

    split_ids = load_split_file(split_path)
    loader = create_caption_dataloader(
        captions_path=captions_path,
        images_dir=images_dir,
        split_image_ids=split_ids,
        vocab=vocab,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        max_caption_length=max_caption_length,
        image_size=int(image_size[0]),
        image_mean=image_mean,
        image_std=image_std,
        random_caption=False,
        seed=int(cfg["seed"]),
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    stored_cfg = checkpoint.get("config", cfg)

    model = CNNTransformerCaptioner(
        vocab_size=vocab_size,
        d_model=int(stored_cfg["d_model"]),
        nhead=int(stored_cfg["nhead"]),
        num_decoder_layers=int(stored_cfg["num_decoder_layers"]),
        dim_feedforward=int(stored_cfg["dim_feedforward"]),
        dropout=float(stored_cfg["dropout"]),
        max_caption_len=max_caption_length,
        freeze_backbone=bool(stored_cfg["freeze_backbone"]),
        use_pretrained_encoder=bool(stored_cfg.get("use_pretrained_encoder", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions: Dict[str, str] = {}
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            ids_out = model.generate(images, start_id=start_id, end_id=end_id)
            for i, image_id in enumerate(batch["image_ids"]):
                predictions[image_id] = vocab.decode(ids_out[i].tolist())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2, ensure_ascii=True)

    print(f"Wrote {len(predictions)} captions to {output_path}")


if __name__ == "__main__":
    main()
