#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.losses import nss, weighted_cc, weighted_kld
from sphererate.models.registry import build_model
from sphererate.utils import ensure_dir, seed_everything


def _model_config(seed: int, processed_name: str) -> dict:
    standard_v3 = "standard_v3" in str(processed_name)
    sitzmann_standard_v2 = "sitzmann_official" in str(processed_name) and "standard_v2" in str(processed_name)
    return {
        "project": {"name": "kernel_v2_static", "seed": int(seed)},
        "paths": {"processed_root": "data/processed", "runs_root": "runs", "reports_root": "reports"},
        "data": {
            "dataset": "salient360_early_horizon_v2",
            "processed_name": str(processed_name),
            "image_size": [256, 512],
            "num_bins": 6,
            "renderer": "sphere_chord",
            "sigma_deg": 2.0,
        },
        "train": {
            "task": "saliency",
            "model": "semantic_static_aux",
            "device": "cuda",
            "epochs": 30,
            "batch_size": 4,
            "num_workers": 2,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "amp": True,
            "selection_metric": "kld_sph",
            "selection_mode": "min",
            "augment_roll_degrees": [0, 60, 120, 180, 240, 300],
        },
        "model": {
            "in_channels": 8,
            "base_channels": 32,
            "hidden_channels": 64,
            "num_bins": 6,
            "semantic_backbone": "resnet18",
            "semantic_layer": "layer3",
            "semantic_pretrained": True,
        },
        "loss": {"final_kl": 1.0, "final_cc": 1.0, "final_nss": 0.1, "temporal": 0.0},
        "protocol": {
            "target_version": 3 if standard_v3 else 2,
            "split": (
                "train59_val13_test13"
                if standard_v3
                else "train16_val3_test3"
                if sitzmann_standard_v2
                else "historical_nested_split"
            ),
            "fit_split": "train",
            "selection_split": "val",
            "report_split": "test" if (standard_v3 or sitzmann_standard_v2) else None,
            "test_access": (
                "ordinary_repeatable_evaluation_never_model_selection"
                if (standard_v3 or sitzmann_standard_v2)
                else "historically_forbidden"
            ),
        },
    }


def _roll_batch(batch: dict[str, torch.Tensor | list[str]], generator: torch.Generator) -> None:
    image = batch["image"]
    final = batch["final"]
    fixation = batch["fixation"]
    assert isinstance(image, torch.Tensor) and isinstance(final, torch.Tensor) and isinstance(fixation, torch.Tensor)
    degree_steps = torch.randint(0, 6, (image.shape[0],), generator=generator)
    for index, step in enumerate(degree_steps.tolist()):
        shift = int(round(step * image.shape[-1] / 6.0))
        image[index] = torch.roll(image[index], shifts=shift, dims=-1)
        final[index] = torch.roll(final[index], shifts=shift, dims=-1)
        fixation[index] = torch.roll(fixation[index], shifts=shift, dims=-1)


@torch.no_grad()
def _validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    klds, ccs = [], []
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["final"].to(device)
        area = batch["area_weights"].to(device)
        prediction = model(image)["final"]
        klds.append(float(weighted_kld(prediction, target, area)))
        ccs.append(float(weighted_cc(prediction, target, area)))
    return {"kld_sph": float(np.mean(klds)), "cc_sph": float(np.mean(ccs))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the frozen strong static model on kernel-v2 final maps.")
    parser.add_argument("--processed-name", default="early_horizon_salient360_spherical2deg_v2")
    parser.add_argument("--run-dir", default="runs/kernel_v2_static_final_seed20260711")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    random.seed(args.seed)
    config = _model_config(args.seed, args.processed_name)
    config["train"]["epochs"] = int(args.epochs)
    config["train"]["device"] = str(args.device)
    root = Path(config["paths"]["processed_root"]) / args.processed_name
    train_set = EarlyHorizonDataset(root, split="train", target_mode="correct")
    val_set = EarlyHorizonDataset(root, split="val", target_mode="correct")
    loader_generator = torch.Generator().manual_seed(args.seed)
    roll_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        generator=loader_generator,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device(args.device)
    model = build_model("semantic_static_aux", config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    run_dir = ensure_dir(args.run_dir)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    best = float("inf")
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            _roll_batch(batch, roll_generator)
            image = batch["image"].to(device, non_blocking=True)
            target = batch["final"].to(device, non_blocking=True)
            fixation = batch["fixation"].to(device, non_blocking=True)
            area = batch["area_weights"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(image)["final"]
                loss = weighted_kld(prediction, target, area) + (1.0 - weighted_cc(prediction, target, area)) - 0.1 * nss(prediction, fixation)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation = _validate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": validation}
        history.append(record)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "val": validation,
            "selection_metric": "kld_sph",
            "selection_score": validation["kld_sph"],
        }
        torch.save(state, checkpoint_dir / "last.pt")
        if validation["kld_sph"] < best:
            best = validation["kld_sph"]
            torch.save(state, checkpoint_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(
            f"epoch={epoch:02d} train={record['train_loss']:.6f} val_kld={validation['kld_sph']:.6f} "
            f"val_cc={validation['cc_sph']:.6f} best={best:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
