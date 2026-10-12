#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.baselines.panoramic_cnn import (
    build_official_panoramic_cnn,
    resize_spherical_density,
    spherical_area_tensor,
)
from sphererate.data.early_view import read_manifest
from sphererate.losses import weighted_cc, weighted_kld
from sphererate.utils import ensure_dir


class StaticFinalDataset(Dataset):
    def __init__(self, root: Path, split: str, height: int, width: int) -> None:
        self.rows = sorted(
            [row for row in read_manifest(root / "manifest.jsonl") if row["split"] == split],
            key=lambda row: str(row["image_id"]),
        )
        self.height, self.width = int(height), int(width)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        with Image.open(row["image"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(rgb).permute(2, 0, 1)
        image = F.interpolate(image[None], size=(self.height, self.width), mode="bilinear", align_corners=False)[0]
        with np.load(row["target"]) as data:
            final = torch.from_numpy(np.asarray(data["final"], dtype=np.float32))
        return {
            "image": image,
            "target": resize_spherical_density(final, self.height, self.width),
            "image_id": str(row["image_id"]),
        }


def _source_commit(source: Path) -> str:
    head = source / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    value = head.read_text().strip()
    if value.startswith("ref:"):
        ref = source / ".git" / value.split(" ", 1)[1]
        return ref.read_text().strip() if ref.exists() else value
    return value


def _atomic_torch_save(state: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, area: torch.Tensor) -> float:
    model.eval()
    values = []
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        prediction = F.softplus(model(image))
        values.extend(weighted_kld(prediction, target, area).detach().cpu().reshape(-1).tolist())
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train published Panoramic-CNN as a leak-free static baseline.")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--run-dir", default="runs/eh6_panoramic_static_seed20260711")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-root", default="data/processed/early_horizon_salient360_v1")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    root = Path(args.data_root)
    train_set = StaticFinalDataset(root, "train", args.height, args.width)
    val_set = StaticFinalDataset(root, "val", args.height, args.width)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = build_official_panoramic_cnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    area = spherical_area_tensor(args.height, args.width, device).unsqueeze(0)
    output = ensure_dir(args.run_dir)
    checkpoint_dir = ensure_dir(output / "checkpoints")
    best = float("inf")
    history = []
    start_epoch = 1
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"
    if args.resume:
        candidates = [last_path, best_path]
        previous = None
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                previous = torch.load(candidate, map_location=device)
                break
            except (RuntimeError, EOFError):
                continue
        if previous is None:
            raise RuntimeError(f"Cannot resume from readable checkpoint in {checkpoint_dir}")
        model.load_state_dict(previous["model"])
        if "optimizer" in previous:
            optimizer.load_state_dict(previous["optimizer"])
        start_epoch = int(previous["epoch"]) + 1
        if best_path.exists():
            best = float(torch.load(best_path, map_location="cpu")["val_kld_sph"])
        history_path = output / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())
    source = Path("third_party/Panoramic-CNN-360-Saliency")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            prediction = F.softplus(model(image))
            loss = weighted_kld(prediction, target, area) + 0.5 * (1.0 - weighted_cc(prediction, target, area))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_kld = validate(model, val_loader, device, area)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_kld_sph": val_kld}
        history.append(row)
        print(json.dumps(row), flush=True)
        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "val_kld_sph": val_kld,
            "source": str(source),
            "source_commit": _source_commit(source),
            "role": "published_static_360_final_repeat",
            "optimizer": optimizer.state_dict(),
        }
        _atomic_torch_save(state, checkpoint_dir / "last.pt")
        if val_kld < best:
            best = val_kld
            _atomic_torch_save(state, checkpoint_dir / "best.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(best_path.read_bytes()).hexdigest()
    (output / "provenance.json").write_text(
        json.dumps({"checkpoint": str(best_path), "sha256": digest, "best_val_kld_sph": best}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
