"""Offline Faster R-CNN feature extraction for BUTD image captioning.

Extracts top-K region features from every Flickr30k image using a
pretrained Faster R-CNN (ResNet-50 + FPN backbone, torchvision) and saves
each image's features as a .pt file:

    data/butd_features/{image_id}.pt  →  float32 tensor, shape (K, 1024)

The 1024-dim features come from the detector's TwoMLPHead (box_head), which
is the last layer before the final class/box prediction heads. Proposals are
ordered by objectness score (highest first), so the top K are the most
salient detected regions.

Usage
-----
From the project root:

    python scripts/extract_butd_features.py

Optional flags:
    --images-dir   DIR   Path to image folder  (default: data/Images)
    --output-dir   DIR   Where to save .pt files (default: data/butd_features)
    --splits-dir   DIR   Split txt files dir    (default: metadata/splits)
    --num-regions  INT   Regions per image       (default: 36)
    --batch-size   INT   Images per GPU batch    (default: 8)
    --device       STR   "cuda", "cpu", or "auto" (default: auto)

The script is idempotent: images whose output file already exists are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ---------------------------------------------------------------------------
# Ensure the project root is on the path so src.* imports work when the
# script is run directly from any directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.splits import load_split_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_REGIONS = 36
FEATURE_DIM = 1024  # TwoMLPHead output dim in fasterrcnn_resnet50_fpn


# ---------------------------------------------------------------------------
# Lightweight image-only dataset (no caption loading needed)
# ---------------------------------------------------------------------------
class _ImageDataset(Dataset):
    """Returns (image_id, raw_tensor_in_0_1) for each image in image_ids."""

    def __init__(self, image_ids: list[str], images_dir: Path) -> None:
        # List the directory once instead of calling .exists() per file.
        # This avoids 30k individual API calls on Google Drive which causes
        # [Errno 5] Input/output error due to rate limiting.
        try:
            available = set(os.listdir(images_dir))
        except OSError:
            available = set(image_ids)  # fallback: assume all present
        self.samples = [iid for iid in image_ids if iid in available]
        self.images_dir = images_dir
        self._to_tensor = T.ToTensor()  # converts PIL [0,255] → float [0,1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[str, Tensor]:
        iid = self.samples[idx]
        img = Image.open(self.images_dir / iid).convert("RGB")
        return iid, self._to_tensor(img)


def _collate_images(batch: list[tuple[str, Tensor]]) -> tuple[list[str], list[Tensor]]:
    """Keep images as a list (variable spatial sizes for the FRCNN transform)."""
    ids, tensors = zip(*batch)
    return list(ids), list(tensors)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_batch(
    images: list[Tensor],
    detector: torch.nn.Module,
    num_regions: int,
    feature_dim: int,
    device: torch.device,
) -> Tensor:
    """Extract region features for a batch of images.

    Runs the Faster R-CNN pipeline up to (and including) the box_head, then
    takes the top-``num_regions`` proposals per image (already sorted by
    objectness score from the RPN).

    Args:
        images:      List of (3, H, W) float tensors in [0, 1]. Variable
                     spatial sizes are fine — the detector transform handles it.
        detector:    Pretrained Faster R-CNN in eval mode.
        num_regions: How many regions to keep per image.
        feature_dim: Expected output dim of box_head (1024 for ResNet-50 FPN).
        device:      Compute device.

    Returns:
        (B, num_regions, feature_dim) float32 tensor on CPU.
    """
    images = [img.to(device) for img in images]

    # Step 1: detector's own transform (normalises + batches variable-size imgs)
    image_list, _ = detector.transform(images, None)

    # Step 2: ResNet-50 + FPN backbone → multi-scale feature maps
    backbone_features = detector.backbone(image_list.tensors)

    # Step 3: Region Proposal Network → top proposals per image
    proposals, _ = detector.rpn(image_list, backbone_features, None)
    # proposals: List[Tensor], each (N_i, 4), sorted by objectness descending

    # Step 4: ROI Align → (total_proposals, 256, 7, 7)
    pooled = detector.roi_heads.box_roi_pool(
        backbone_features, proposals, image_list.image_sizes
    )

    # Step 5: TwoMLPHead → (total_proposals, 1024)
    box_features = detector.roi_heads.box_head(pooled)

    # Step 6: split by image and keep top num_regions
    counts = [p.shape[0] for p in proposals]
    per_image = list(box_features.cpu().split(counts, dim=0))

    result: list[Tensor] = []
    for feats in per_image:
        n = feats.shape[0]
        if n >= num_regions:
            result.append(feats[:num_regions])
        else:
            pad = torch.zeros(num_regions - n, feature_dim)
            result.append(torch.cat([feats, pad], dim=0))

    return torch.stack(result, dim=0)  # (B, num_regions, feature_dim)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-extract Faster R-CNN region features for BUTD training."
    )
    parser.add_argument("--images-dir", default="data/Images")
    parser.add_argument("--output-dir", default="data/butd_features")
    parser.add_argument("--splits-dir", default="metadata/splits")
    parser.add_argument("--num-regions", type=int, default=NUM_REGIONS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Resolve paths relative to project root
    images_dir = _PROJECT_ROOT / args.images_dir
    output_dir = _PROJECT_ROOT / args.output_dir
    splits_dir = _PROJECT_ROOT / args.splits_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Collect all image IDs across splits
    all_ids: list[str] = []
    for split in ("train", "val", "test"):
        split_file = splits_dir / f"{split}.txt"
        if split_file.exists():
            all_ids.extend(load_split_file(split_file))
    all_ids = sorted(set(all_ids))

    # Filter to only those not yet extracted
    pending = [iid for iid in all_ids if not (output_dir / f"{iid}.pt").exists()]
    print(f"Total images: {len(all_ids)}, already done: {len(all_ids) - len(pending)}, to extract: {len(pending)}")
    if not pending:
        print("All features already extracted. Nothing to do.")
        return

    # Load pretrained Faster R-CNN
    try:
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights,
            fasterrcnn_resnet50_fpn,
        )
    except ImportError:
        print(
            "ERROR: torchvision is required. Install with:\n"
            "  pip install torchvision"
        )
        sys.exit(1)

    print("Loading pretrained Faster R-CNN …")
    detector = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    detector.eval()
    detector.to(device)
    for p in detector.parameters():
        p.requires_grad_(False)
    print("Model loaded.")

    # Build dataset and loader over pending images
    dataset = _ImageDataset(pending, images_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_images,
    )

    total_saved = 0
    total_batches = len(loader)

    for batch_idx, (image_ids, images) in enumerate(loader, 1):
        try:
            features = extract_batch(
                images, detector, args.num_regions, FEATURE_DIM, device
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc):
                # Free fragmented GPU memory and retry one image at a time
                torch.cuda.empty_cache()
                print(f"  [batch {batch_idx}] OOM — retrying one image at a time …")
                for iid, img in zip(image_ids, images):
                    try:
                        feat = extract_batch([img], detector, args.num_regions, FEATURE_DIM, device)
                        torch.save(feat[0].contiguous(), output_dir / f"{iid}.pt")
                        total_saved += 1
                    except RuntimeError:
                        torch.cuda.empty_cache()
                        print(f"    skipped {iid} (still OOM at batch size 1)")
                continue
            print(f"  [batch {batch_idx}] extraction failed: {exc} — skipping")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  [batch {batch_idx}] extraction failed: {exc} — skipping")
            continue

        for iid, feat in zip(image_ids, features):
            torch.save(feat.contiguous(), output_dir / f"{iid}.pt")
            total_saved += 1

        # Free reserved-but-unallocated GPU memory between batches
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if batch_idx % 50 == 0 or batch_idx == total_batches:
            pct = 100.0 * batch_idx / total_batches
            print(f"  [{batch_idx}/{total_batches}]  {pct:.1f}%  saved {total_saved} features")

    print(f"\nDone. Saved {total_saved} feature files to {output_dir}")


if __name__ == "__main__":
    main()
