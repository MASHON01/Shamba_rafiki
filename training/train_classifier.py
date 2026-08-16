#!/usr/bin/env python3
"""
Train the leaf-disease image classifier and export it to ONNX.

    python training/train_classifier.py --data training/data/plant_images --epochs 12

Transfer-learns MobileNetV3-small (ImageNet-pretrained) on a folder of
leaf photos and exports models/plant_classifier.onnx plus a
plant_classifier.labels.json class map. Preprocessing here matches the
serve-time transform in backend/app/vision/preprocess.py exactly.
Heavy deps (torch, torchvision) are imported lazily and only needed
when this runs; the app and kiosk never import this file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config.constants import (  # noqa: E402
    CLASSIFIER_INPUT_SIZE,
    CLASSIFIER_LABELS_SUFFIX,
    CLASSIFIER_NORM_MEAN,
    CLASSIFIER_NORM_STD,
    KNOWN_CROPS,
    METADATA_UNKNOWN,
)

_CROP_TOKEN_CANON: dict[str, str] = {
    surface.lower(): canonical
    for canonical, surfaces in KNOWN_CROPS.items()
    for surface in surfaces
}
_CROP_TOKEN_CANON.setdefault("corn", "maize")
_CROP_TOKEN_CANON.setdefault("corn_(maize)", "maize")


def _canon_crop(token: str) -> str:
    key = token.strip().lower().replace(" ", "_")
    if key in _CROP_TOKEN_CANON:
        return _CROP_TOKEN_CANON[key]
    first = key.split("_")[0].split("(")[0]
    return _CROP_TOKEN_CANON.get(first, METADATA_UNKNOWN)


def _clean_condition(token: str) -> str:
    text = token.replace("_", " ").strip()
    if not text:
        return METADATA_UNKNOWN
    if text.lower() in {"healthy", "health"}:
        return "healthy"
    words = [w if (w.isupper() and len(w) <= 4) else w.capitalize() for w in text.split()]
    return " ".join(words)


def parse_class_name(folder_name: str) -> dict[str, str]:
    raw = folder_name.strip()
    if "___" in raw:
        crop_tok, cond_tok = raw.split("___", 1)
    else:
        crop_tok, cond_tok = "", raw
    return {
        "crop": _canon_crop(crop_tok),
        "condition": _clean_condition(cond_tok),
        "label": raw,
    }


def build_label_map(class_names: list[str]) -> list[dict[str, str]]:
    return [parse_class_name(name) for name in class_names]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train + export the leaf-disease classifier.")
    parser.add_argument("--data", required=True, help="ImageFolder root (train/[val/] inside).")
    parser.add_argument("--out", default="models/plant_classifier.onnx", help="ONNX output path.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.15, help="Used if no val/ dir.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Subset, random_split
        from torchvision import datasets, models, transforms
        from torchvision.models import MobileNet_V3_Small_Weights
    except ImportError as exc:
        print(f"This script needs torch + torchvision: pip install torch torchvision ({exc})",
              file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = CLASSIFIER_INPUT_SIZE

    normalize = transforms.Normalize(mean=list(CLASSIFIER_NORM_MEAN), std=list(CLASSIFIER_NORM_STD))
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(), normalize]
    )

    data_root = Path(args.data)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.is_dir():
        train_dir = data_root

    full_train = datasets.ImageFolder(str(train_dir), transform=train_tf)
    class_names = full_train.classes

    if val_dir.is_dir():
        train_ds = full_train
        val_ds = datasets.ImageFolder(str(val_dir), transform=eval_tf)
    else:
        n_val = max(1, int(len(full_train) * args.val_split))
        n_train = len(full_train) - n_val
        gen = torch.Generator().manual_seed(args.seed)
        train_ds, val_subset = random_split(full_train, [n_train, n_val], generator=gen)
        val_base = datasets.ImageFolder(str(train_dir), transform=eval_tf)
        val_ds = Subset(val_base, val_subset.indices)

    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train {len(train_ds)}  Val {len(val_ds)}  Device {device}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                correct += (model(images).argmax(dim=1) == targets).sum().item()
                total += targets.size(0)
        acc = correct / max(1, total)
        print(f"epoch {epoch:>2}/{args.epochs}  loss={running/len(train_ds):.4f}  val_acc={acc:.3f}")
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val accuracy: {best_acc:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval().to("cpu")
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model, dummy, str(out_path), input_names=["input"], output_names=["logits"], opset_version=13
    )

    labels = build_label_map(class_names)
    sidecar = out_path.with_name(out_path.stem + CLASSIFIER_LABELS_SUFFIX)
    sidecar.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote:\n  {out_path}\n  {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
