#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.losses import weighted_cc, weighted_kld
from sphererate.models.registry import build_model
from sphererate.utils import ensure_dir, seed_everything


def _config(seed: int, processed_name: str) -> dict:
    standard_v3 = "standard_v3" in str(processed_name)
    return {
        "project": {"name": "kernel_v2_tempsal_erp", "seed": int(seed)},
        "paths": {"processed_root": "data/processed", "runs_root": "runs"},
        "data": {
            "processed_name": str(processed_name),
            "image_size": [256, 512],
            "num_bins": 4,
            "horizons_s": [0.5, 1.0, 3.0, 5.0],
            "renderer": "sphere_chord",
            "sigma_deg": 2.0,
        },
        "train": {
            "model": "semantic_tempsal_temporal",
            "epochs": 30,
            "batch_size": 4,
            "num_workers": 2,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "amp": True,
            "selection_metric": "early_macro_kld_sph",
            "outer_test": (
                "ordinary_repeatable_evaluation_never_model_selection"
                if standard_v3
                else "historically_forbidden"
            ),
        },
        "model": {
            "in_channels": 8,
            "base_channels": 32,
            "hidden_channels": 64,
            "num_bins": 4,
            "learned_accumulation": True,
            "semantic_backbone": "resnet18",
            "semantic_layer": "layer3",
            "semantic_pretrained": True,
        },
        "loss": {"cumulative_kld": 1.0, "cumulative_cc": 0.5},
        "baseline": {
            "name": "TempSAL-ERP cumulative adaptation",
            "input": "ERP RGB + fixed geometry; no start template or gaze history",
        },
        "protocol": {
            "target_version": 3 if standard_v3 else 2,
            "split": "train59_val13_test13" if standard_v3 else "historical_nested_split",
            "fit_split": "train",
            "selection_split": "val",
            "report_split": "test" if standard_v3 else None,
        },
    }


@torch.no_grad()
def _validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    rows = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        area = batch["area_weights"].to(device, non_blocking=True)
        prediction = model(image)["cumulative"]
        rows.append(
            {
                "kld": float(weighted_kld(prediction[:, :3], target[:, :3], area)),
                "cc": float(weighted_cc(prediction[:, :3], target[:, :3], area)),
            }
        )
    return {
        "early_macro_kld_sph": float(np.mean([row["kld"] for row in rows])),
        "early_macro_cc_sph": float(np.mean([row["cc"] for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a leak-free TempSAL-style ERP cumulative-map baseline on v2.")
    parser.add_argument("--processed-name", default="early_horizon_salient360_spherical2deg_v2")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    seed_everything(args.seed)
    config = _config(args.seed, args.processed_name)
    config["train"]["epochs"] = int(args.epochs)
    config["train"]["device"] = str(args.device)
    root = Path(config["paths"]["processed_root"]) / args.processed_name
    train_set = EarlyHorizonDataset(root, split="train", target_mode="correct")
    val_set = EarlyHorizonDataset(root, split="val", target_mode="correct")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        generator=generator,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device(args.device)
    model = build_model("semantic_tempsal_temporal", config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    run_dir = ensure_dir(args.run_dir or f"runs/kernel_v2_tempsal_erp_seed{args.seed}")
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    parameters = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    best = float("inf")
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            area = batch["area_weights"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(image)["cumulative"]
                loss = weighted_kld(prediction, target, area) + 0.5 * (1.0 - weighted_cc(prediction, target, area))
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
            "parameters": parameters,
        }
        torch.save(state, checkpoint_dir / "last.pt")
        if validation["early_macro_kld_sph"] < best:
            best = validation["early_macro_kld_sph"]
            torch.save(state, checkpoint_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(
            f"epoch={epoch:02d} train={record['train_loss']:.6f} "
            f"val_kld={validation['early_macro_kld_sph']:.6f} "
            f"val_cc={validation['early_macro_cc_sph']:.6f} best={best:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
