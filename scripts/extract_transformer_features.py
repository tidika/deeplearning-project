"""Offline ResNet-50 spatial feature extraction for Transformer captioning.

Extracts spatial grid features from every Flickr30k image using a pretrained
ResNet-50 backbone (torchvision) and saves each image's features as a .pt file:

    data/transformer_features/{image_id}.pt  →  float32 tensor, shape (49, 2048)

The 49 spatial locations come from ResNet-50's final convolutional layer (layer4)
which outputs a 7×7 = 49 position grid with 2048 channels per location. These
serve as the memory (keys/values) for the transformer decoder's cross-attention.

Unlike BUTD's Faster R-CNN features (object regions), these are dense spatial
grid features — each of the 49 positions corresponds to a fixed patch of the
image rather than a detected object. This is the key architectural difference
being studied in the comparison.

Usage
-----
From the project root:

    python scripts/extract_transformer_features.py

Optional flags:
    --images-dir   DIR   Path to image folder   (default: data/Images)
    --output-dir   DIR   Where to save .pt files (default: data/transformer_features)
    --splits-dir   DIR   Split txt files dir     (default: metadata/splits)
    --batch-size   INT   Images per GPU batch    (default: 32)
    --device       STR   "cuda", "cpu", or "auto" (default: auto)

The script is idempotent: images whose output file already exists are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ---------------------------------------------------------------------------
# Project root on path for src.* imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.splits import load_split_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPATIAL_SIZE = 7        # ResNet-50 layer4 output: 7×7
NUM_PATCHES = SPATIAL_SIZE * SPATIAL_SIZE  # 49
FEATURE_DIM = 2048      # ResNet-50 layer4 channels


# ---------------------------------------------------------------------------
# ResNet-50 spatial feature extractor
# ---------------------------------------------------------------------------
class _ResNetExtractor(nn.Module):
    """Extracts (49, 2048) spatial features using ResNet-50's layer4 output."""

    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as exc:
            raise ImportError(
                "torchvision is required. Install with: pip install torchvision"
            ) from exc

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Keep everything up to and including layer4; drop avgpool and fc
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """
        Args:
            images: (B, 3, 224, 224) ImageNet-normalised tensors.
        Returns:
            (B, 49, 2048) spatial feature tensor on CPU.
        """
        self.eval()
        feat = self.features(images)           # (B, 2048, 7, 7)
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1)        # (B, 7, 7, 2048)
        feat = feat.reshape(B, H * W, C)       # (B, 49, 2048)
        return feat.cpu()


# ---------------------------------------------------------------------------
# Image dataset (no captions needed)
# ---------------------------------------------------------------------------
_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class _ImageDataset(Dataset):
    """Returns (image_id, normalised_tensor) for each pending image."""

    def __init__(self, image_ids: list[str], images_dir: Path) -> None:
        try:
            available = set(os.listdir(images_dir))
        except OSError:
            available = set(image_ids)
        self.samples = [iid for iid in image_ids if iid in available]
        self.images_dir = images_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[str, Tensor | None]:
        iid = self.samples[idx]
        try:
            img = Image.open(self.images_dir / iid).convert("RGB")
            return iid, _TRANSFORM(img)
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: skipping {iid} ({exc})")
            return iid, None


# ---------------------------------------------------------------------------
# Collate — filters out images that failed to load
# ---------------------------------------------------------------------------
def _safe_collate(batch: list) -> tuple[list[str], Tensor]:
    valid = [(iid, img) for iid, img in batch if img is not None]
    if not valid:
        return [], torch.zeros(0)
    ids, imgs = zip(*valid)
    return list(ids), torch.stack(imgs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-extract ResNet-50 spatial features for Transformer training."
    )
    parser.add_argument("--images-dir", default="data/Images")
    parser.add_argument("--output-dir", default="data/transformer_features")
    parser.add_argument("--splits-dir", default="metadata/splits")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    images_dir = _PROJECT_ROOT / args.images_dir
    output_dir = _PROJECT_ROOT / args.output_dir
    splits_dir = _PROJECT_ROOT / args.splits_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Collect all image IDs
    all_ids: list[str] = []
    for split in ("train", "val", "test"):
        split_file = splits_dir / f"{split}.txt"
        if split_file.exists():
            all_ids.extend(load_split_file(split_file))
    all_ids = sorted(set(all_ids))

    pending = [iid for iid in all_ids if not (output_dir / f"{iid}.pt").exists()]
    print(
        f"Total images: {len(all_ids)}, "
        f"already done: {len(all_ids) - len(pending)}, "
        f"to extract: {len(pending)}"
    )
    if not pending:
        print("All features already extracted. Nothing to do.")
        return

    # Build extractor
    print("Loading pretrained ResNet-50 …")
    extractor = _ResNetExtractor().to(device).eval()
    print("Model loaded.")

    dataset = _ImageDataset(pending, images_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=_safe_collate,
    )

    total_saved = 0
    total_batches = len(loader)

    for batch_idx, (image_ids, images) in enumerate(loader, 1):
        if len(image_ids) == 0:
            continue  # entire batch had unreadable images
        try:
            images = images.to(device)
            features = extractor(images)  # (B, 49, 2048) on CPU
        except RuntimeError as exc:
            if "out of memory" in str(exc):
                torch.cuda.empty_cache()
                print(f"  [batch {batch_idx}] OOM — retrying one image at a time …")
                for iid, img in zip(image_ids, images.unbind(0)):
                    try:
                        feat = extractor(img.unsqueeze(0).to(device))
                        torch.save(feat[0].contiguous(), output_dir / f"{iid}.pt")
                        total_saved += 1
                    except RuntimeError:
                        torch.cuda.empty_cache()
                        print(f"    skipped {iid} (still OOM at batch size 1)")
                continue
            print(f"  [batch {batch_idx}] failed: {exc} — skipping")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  [batch {batch_idx}] failed: {exc} — skipping")
            continue

        for iid, feat in zip(image_ids, features):
            torch.save(feat.contiguous(), output_dir / f"{iid}.pt")
            total_saved += 1

        if device.type == "cuda":
            torch.cuda.empty_cache()

        if batch_idx % 100 == 0 or batch_idx == total_batches:
            pct = 100.0 * batch_idx / total_batches
            print(f"  [{batch_idx}/{total_batches}]  {pct:.1f}%  saved {total_saved} features")

    print(f"\nDone. Saved {total_saved} feature files to {output_dir}")


if __name__ == "__main__":
    main()
