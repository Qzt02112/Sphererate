#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.path.expandvars("${SCANDMM_ROOT}/baselines/third_party/Pathformer3D"))
sys.path.insert(0, str(SOURCE))

import datasets.dataloder as official_data
from modules.mdn3d import mixture_probability
from modules.transformer import Transformer


CONFIG: dict[str, Any] = {
    "image_input_resize": (128, 256),
    "action_map_size": (128, 256),
    "feature_dim": 448,
    "patch_size": (8, 8),
    "d_model": 128,
    "d_ff": 128,
    "d_k": 64,
    "d_v": 64,
    "enc_n_layers": 4,
    "dec_n_layers": 4,
    "n_heads": 8,
    "dropout": 0.0,
    "num_gauss": 5,
    "MDN_hidden_num": 16,
    "postion_method": "fixed",
    "max_length": 20,
    "replace_encoder": False,
    "lr": 5e-6,
    "weight_decay": 0.0,
    "warmup_epochs": 10,
    "milestones_after_warmup": (10, 20, 30, 40),
    "gamma": 0.5,
    "start_pitch_deg": 0.0,
    "start_yaw_deg": 90.0,
}


def _seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _model() -> Transformer:
    height, width = CONFIG["image_input_resize"]
    patch_h, patch_w = CONFIG["patch_size"]
    return Transformer(
        image_input_resize=CONFIG["image_input_resize"],
        feature_dim=CONFIG["feature_dim"],
        patch_size=CONFIG["patch_size"],
        num_patch_h=height // patch_h,
        num_patch_w=width // patch_w,
        d_model=CONFIG["d_model"],
        d_k=CONFIG["d_k"],
        d_v=CONFIG["d_v"],
        n_heads=CONFIG["n_heads"],
        d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"],
        enc_n_layers=CONFIG["enc_n_layers"],
        postion_method=CONFIG["postion_method"],
        max_length=CONFIG["max_length"],
        dec_n_layers=CONFIG["dec_n_layers"],
        MDN_hidden_num=CONFIG["MDN_hidden_num"],
        num_gauss=CONFIG["num_gauss"],
        action_map_size=CONFIG["action_map_size"],
        replace_encoder=CONFIG["replace_encoder"],
    )


def _start_xyz(device: torch.device) -> torch.Tensor:
    lat = math.radians(float(CONFIG["start_pitch_deg"]))
    lon = math.radians(float(CONFIG["start_yaw_deg"]))
    return torch.tensor(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
        dtype=torch.float32,
        device=device,
    )


def _lr_multiplier(scheduler_epoch: int) -> float:
    warmup = int(CONFIG["warmup_epochs"])
    if scheduler_epoch <= warmup:
        return float(scheduler_epoch) / float(warmup)
    after = scheduler_epoch - warmup
    decay_count = sum(after >= int(value) for value in CONFIG["milestones_after_warmup"])
    return float(CONFIG["gamma"]) ** decay_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain official Pathformer3D on the frozen kernel-v2 split.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=40, help="Official final epoch index (inclusive).")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, help="Smoke-test limit; omit for formal runs.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--dataset-json",
        default="data/processed/kernel_v2_matched_scanpaths/pathformer3d/dataset.json",
    )
    parser.add_argument("--start-pitch-deg", type=float, default=0.0)
    parser.add_argument("--start-yaw-deg", type=float, default=90.0)
    parser.add_argument("--expected-train", type=int)
    parser.add_argument("--expected-val", type=int)
    parser.add_argument("--expected-gate", type=int)
    parser.add_argument("--expected-test", type=int)
    args = parser.parse_args()
    if args.seed not in {20260710, 20260711, 20260712}:
        raise ValueError("Seed is outside the frozen matched-baseline protocol")
    dataset_json = (ROOT / args.dataset_json).resolve()
    data = json.loads(dataset_json.read_text(encoding="utf-8"))
    expected = {
        "train": int(args.expected_train) if args.expected_train is not None else 1840,
        "val": int(args.expected_val) if args.expected_val is not None else 9,
    }
    if "gate" in data or args.expected_gate is not None:
        expected["gate"] = int(args.expected_gate) if args.expected_gate is not None else 13
    if "test" in data or args.expected_test is not None:
        expected["test"] = int(args.expected_test) if args.expected_test is not None else 13
    observed = {key: len(data.get(key, [])) for key in expected}
    if observed != expected:
        raise ValueError(f"Matched Pathformer3D dataset counts changed: {observed} != {expected}")

    CONFIG["start_pitch_deg"] = float(args.start_pitch_deg)
    CONFIG["start_yaw_deg"] = float(args.start_yaw_deg)
    _seed_all(args.seed)
    device = torch.device(args.device)
    run_dir = ROOT / args.run_dir
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    official_data.paths["matched_data_dict_path"] = str(dataset_json)
    os.chdir(ROOT)
    dataset = official_data.ScanPath360Dataset(
        phase="train",
        dataset_name="matched",
        image_input_resize=CONFIG["image_input_resize"],
        patch_size=CONFIG["patch_size"],
        max_length=CONFIG["max_length"],
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
        generator=generator,
    )
    model = _model().to(device)
    optimizer = AdamW(model.parameters(), lr=float(CONFIG["lr"]), weight_decay=float(CONFIG["weight_decay"]))
    scheduler = LambdaLR(optimizer, lr_lambda=_lr_multiplier)
    start_epoch = 0
    existing = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if existing:
        state = torch.load(existing[-1], map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1

    source_commit = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    provenance = {
        "method": "Pathformer3D",
        "source": str(SOURCE),
        "source_commit": source_commit,
        "dataset_json": str(dataset_json),
        "dataset_json_sha256": _sha256(dataset_json),
        "split_counts": {key: len(data[key]) for key in expected},
        "test_entries_used_for_training": 0,
        "seed": args.seed,
        "official_config": CONFIG,
        "compatibility": {
            "mmcv_replaced_by_argparse": True,
            "warmup_scheduler_reimplemented_with_pytorch_lambda": True,
            "declared_start_xyz_replaces_zero_BOS": True,
            "probability_floor": 1e-12,
        },
        "smoke_max_batches": args.max_batches,
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log_path = run_dir / "train.jsonl"
    start_vector = _start_xyz(device)
    for epoch in range(start_epoch, int(args.epochs) + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        losses: list[float] = []
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            images = batch["imgs"].to(device, non_blocking=True)
            scanpath = batch["scanpath"].float().to(device, non_blocking=True)
            dec_inputs = batch["dec_inputs"].float().to(device, non_blocking=True)
            dec_inputs[:, 0] = start_vector
            dec_masks = batch["dec_masks"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pis, mus, sigmas, rhos = model(images, dec_inputs, dec_masks)
            probabilities = mixture_probability(
                pis, mus, sigmas, rhos, scanpath.unsqueeze(-1)
            ).squeeze(-1)
            valid = torch.arange(CONFIG["max_length"], device=device)[None] < batch[
                "valid_len"
            ].to(device)[:, None]
            selected = probabilities[valid]
            if not torch.isfinite(selected).all():
                raise FloatingPointError(f"Non-finite Pathformer3D probability at epoch {epoch}")
            loss = -torch.log(selected.clamp_min(1e-12)).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if args.max_batches is not None and batch_index + 1 >= int(args.max_batches):
                break
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_nll": float(np.mean(losses)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": float(time.perf_counter() - epoch_start),
            "peak_cuda_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else None
            ),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if epoch % int(args.checkpoint_every) == 0 or epoch == int(args.epochs):
            torch.save(
                {
                    "epoch": epoch,
                    "seed": args.seed,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": CONFIG,
                    "source_commit": source_commit,
                    "dataset_json_sha256": provenance["dataset_json_sha256"],
                },
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
            )


if __name__ == "__main__":
    main()
