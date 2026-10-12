#!/usr/bin/env python
"""Retrain official HAT on the standard Salient360 train/val/test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.path.expandvars("${SCANDMM_ROOT}/baselines/third_party/HAT"))
sys.path.insert(0, str(SOURCE))

from common.data import SPTrans_Human_Gaze
from common.losses import focal_loss
from common.utils import preprocess_fixations, transform_fixations
from hat.models import HumanAttnTransformer


STANDARD_SEEDS = {20260710, 20260711, 20260712}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def hat_config() -> SimpleNamespace:
    return SimpleNamespace(
        im_w=512,
        im_h=256,
        patch_num=[32, 16],
        patch_size=[16, 16],
        patch_count=512,
        max_traj_length=21,
        target_fix_map_sigma=16,
        TAP="FV",
        pad_idx=0,
        eos_idx=1,
        special_symbols=["<pad>"],
        backbone_config=str(SOURCE / "configs/resnet50.yaml"),
        pixel_decoder="MSD",
        use_DCB_target=False,
        cortical_magnify=False,
        cover_ratio=0.5,
        saliency_pred=False,
        target_map_pred=False,
        use_action_map=False,
    )


def build_model(pa: SimpleNamespace) -> HumanAttnTransformer:
    previous = Path.cwd()
    os.chdir(SOURCE)
    try:
        model = HumanAttnTransformer(
            pa,
            num_decoder_layers=6,
            hidden_dim=384,
            nhead=4,
            ntask=1,
            tgt_vocab_size=pa.patch_count + len(pa.special_symbols),
            num_output_layers=3,
            separate_fix_arch=True,
            train_encoder=False,
            train_pixel_decoder=True,
            use_dino=False,
            dropout=0.0,
            dim_feedforward=512,
            parallel_arch=True,
            dorsal_source=["P1"],
            num_encoder_layers=3,
            transfer_learning_setting="none",
            project_queries=False,
            is_pretraining=False,
            output_feature_map_name="P4",
        )
    finally:
        os.chdir(previous)
    return model


def _human_scanpaths(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, record in enumerate(data["trajectories"]["train"]):
        points = np.asarray(record["points_yx"], dtype=np.float32)
        if points.shape != (21, 2):
            raise ValueError(f"Unexpected HAT training trajectory shape: {points.shape}")
        rows.append(
            {
                "name": f"{record['image_id']}.png",
                "task": "none",
                "condition": "freeview",
                "X": (points[:, 1] * 512.0).clip(0.0, 511.0).tolist(),
                "Y": (points[:, 0] * 256.0).clip(0.0, 255.0).tolist(),
                "T": [0.0] + [1.0] * 20,
                "subject": int(index + 1),
                "length": 21,
                "split": "train",
            }
        )
    return rows


def _dataset(dataset_path: Path, pa: SimpleNamespace) -> tuple[SPTrans_Human_Gaze, int]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    audit = data["audit"]
    if set(audit["image_counts"]) not in ({"train", "val", "gate"}, {"train", "val", "test"}):
        raise ValueError("HAT image split must contain train, val, and one evaluation split")
    if set(audit["trajectory_counts"]) != {"train", "val"}:
        raise ValueError("HAT training trajectories must contain train and val only")
    scanpaths = _human_scanpaths(data)
    labels = preprocess_fixations(
        scanpaths,
        pa.patch_size,
        pa.patch_num,
        pa.im_h,
        pa.im_w,
        has_stop=True,
        sample_scanpath=False,
        discretize_fix=False,
        remove_return_fixations=False,
        is_coco_dataset=False,
    )
    # The declared start is observed, never predicted.  The official generic
    # preprocessing also emits an empty-history target for the first point;
    # remove only that record and retain all official subsequent/stop labels.
    labels = [label for label in labels if len(label[3]) > 0]
    term_count = sum(bool(label[5]) for label in labels)
    train_path_count = int(audit["trajectory_counts"]["train"])
    expected_labels = train_path_count * 21
    if len(labels) != expected_labels or term_count != train_path_count:
        raise ValueError(
            f"Unexpected HAT labels: total={len(labels)}, term={term_count}, "
            f"expected={expected_labels}/{train_path_count}"
        )
    transform = transforms.Compose(
        [
            transforms.Resize((pa.im_h, pa.im_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = SPTrans_Human_Gaze(
        str(dataset_path.parent),
        labels,
        {},
        None,
        pa,
        transform,
        {"none": 0},
        blur_action=True,
        coco_annos=None,
    )
    return dataset, term_count


def _losses(
    model: torch.nn.Module,
    batch: dict[str, Any],
    pa: SimpleNamespace,
    device: torch.device,
    term_loss: torch.nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    image = batch["true_state"].to(device, non_blocking=True)
    task_ids = batch["task_id"].to(device, non_blocking=True)
    is_last = batch["is_last"].to(device, non_blocking=True)
    inp_seq, inp_seq_high = transform_fixations(
        batch["normalized_fixations"],
        batch["is_padding"],
        pa,
        False,
        return_highres=True,
    )
    inp_seq = inp_seq.to(device, non_blocking=True)
    inp_seq_high = inp_seq_high.to(device, non_blocking=True)
    logits = model(image, inp_seq, inp_seq == pa.pad_idx, inp_seq_high, task_ids)
    prediction = logits["pred_fixation_map"]
    if prediction.ndim == 4:
        prediction = prediction[torch.arange(image.shape[0], device=device), task_ids]
    prediction = torch.sigmoid(prediction)
    if prediction.shape[-2:] != (pa.im_h, pa.im_w):
        prediction = F.interpolate(
            prediction.unsqueeze(1), size=(pa.im_h, pa.im_w), mode="bilinear", align_corners=False
        ).squeeze(1)
    non_terminal = ~is_last
    if torch.any(non_terminal):
        map_loss = focal_loss(
            prediction[non_terminal],
            batch["target_fix_map"].to(device, non_blocking=True)[non_terminal],
            alpha=1,
            beta=4,
            weights=batch["IOR_weight_map"].to(device, non_blocking=True)[non_terminal],
        )
    else:
        map_loss = prediction.sum() * 0.0
    termination = logits["pred_termination"]
    if termination.ndim == 2:
        termination = termination[torch.arange(image.shape[0], device=device), task_ids]
    stop_loss = term_loss(termination, is_last.float())
    total = map_loss + stop_loss
    return total, {
        "next_fix_pred": float(map_loss.detach().cpu()),
        "term_pred": float(stop_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain official HAT on the standard matched split.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--dataset-json",
        default="data/processed/kernel_v2_matched_scanpaths/hat/dataset.json",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=5000,
        help="Matched 5k training budget; the seed-20260710 val curve plateaued at 3k--4k.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.seed not in STANDARD_SEEDS:
        raise ValueError("Seed is outside the standard matched-baseline protocol")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Matched HAT training requires CUDA")
    _seed_all(args.seed)
    pa = hat_config()
    dataset_path = (ROOT / args.dataset_json).resolve()
    dataset, term_count = _dataset(dataset_path, pa)
    dataset_payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True,
        persistent_workers=int(args.workers) > 0,
        generator=generator,
    )
    model = build_model(pa).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
    term_pos_weight = (len(dataset) / float(term_count)) - 1.0
    term_loss = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(term_pos_weight, dtype=torch.float32, device=device)
    )

    run_dir = (ROOT / args.run_dir).resolve()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    epoch = 0
    existing = sorted(checkpoint_dir.glob("step_*.pt"))
    if existing:
        state = torch.load(existing[-1], map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
        epoch = int(state["epoch"])

    source_commit = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    fallback_path = (
        SOURCE / "hat/pixel_decoder/ops/functions/ms_deform_attn_func.py"
    )
    model_path = SOURCE / "hat/models.py"
    provenance = {
        "method": "HAT-ERP-declared-start-v1",
        "source": str(SOURCE),
        "source_commit": source_commit,
        "dataset_json": str(dataset_path),
        "dataset_json_sha256": _sha256(dataset_path),
        "split_counts": dataset_payload["audit"]["image_counts"],
        "trajectory_counts": dataset_payload["audit"]["trajectory_counts"],
        "test_entries_used_for_training": 0,
        "seed": args.seed,
        "training": {
            "max_iters": int(args.max_iters),
            "batch_size": int(args.batch_size),
            "optimizer": "AdamW",
            "lr": 1e-4,
            "losses": ["official focal next-fixation", "official termination BCE"],
            "backbone": "official frozen Mask2Former R50 initialization",
            "pixel_decoder": "official trainable MSDeformAttn decoder",
            "budget_decision": "5k fixed after seed-20260710 val convergence audit; checkpoints every 1k",
        },
        "conditioning_semantics": {
            "declared_start_is_first_history_token": True,
            "declared_start_is_not_a_supervised_target": True,
            "first_supervised_target": "first_one_second_gaze",
            "subsequent_history_is_autoregressive": True,
        },
        "compatibility": {
            "detectron2_cpu_extension_build": True,
            "official_pytorch_msdeformattn_fallback": True,
            "fallback_import_patch_sha256": _sha256(fallback_path),
            "backbone_key_compatibility_patch_sha256": _sha256(model_path),
        },
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    log_path = run_dir / "train.jsonl"
    while step < int(args.max_iters):
        model.train()
        epoch_start = time.perf_counter()
        epoch_losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            total, record = _losses(model, batch, pa, device, term_loss)
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite HAT loss at step {step}")
            total.backward()
            optimizer.step()
            step += 1
            epoch_losses.append(record["total"])
            if step % 20 == 0 or step == 1:
                print(
                    json.dumps(
                        {"step": step, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **record}
                    ),
                    flush=True,
                )
            if step % int(args.checkpoint_every) == 0 or step == int(args.max_iters):
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "seed": args.seed,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": vars(pa),
                        "source_commit": source_commit,
                        "dataset_json_sha256": provenance["dataset_json_sha256"],
                    },
                    checkpoint_dir / f"step_{step:06d}.pt",
                )
            if step >= int(args.max_iters):
                break
        scheduler.step()
        epoch_record = {
            "epoch": epoch,
            "end_step": step,
            "mean_total_loss": float(np.mean(epoch_losses)),
            "seconds": float(time.perf_counter() - epoch_start),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "peak_cuda_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024**2))
                if device.type == "cuda"
                else None
            ),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(epoch_record) + "\n")
        epoch += 1


if __name__ == "__main__":
    main()
