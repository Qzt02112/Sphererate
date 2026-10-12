from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from sphererate.config import save_config
from sphererate.data.dataset import ProcessedNextFixationDataset, ProcessedSaliencyDataset
from sphererate.evaluation.metrics import mean_scores, saliency_scores
from sphererate.evaluation.nextfix import next_fixation_scores
from sphererate.evaluation.process import process_scores
from sphererate.losses import saliency_training_loss
from sphererate.models.registry import build_model
from sphererate.utils import ensure_dir, seed_everything


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(name)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _row_dataset(row: dict[str, Any]) -> str:
    return str(row.get("dataset") or "unknown")


def _dataset_balanced_sampler(dataset: ProcessedSaliencyDataset | ProcessedNextFixationDataset, seed: int) -> WeightedRandomSampler:
    counts = Counter(_row_dataset(row) for row in dataset.rows)
    weights = torch.as_tensor([1.0 / counts[_row_dataset(row)] for row in dataset.rows], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


def _seed_from_config(config: dict[str, Any], name: str, fallback: int | None = None) -> int:
    project_seed = int(config.get("project", {}).get("seed", 20260708)) if fallback is None else int(fallback)
    train_cfg = config.get("train", {})
    return int(train_cfg.get(name, config.get("project", {}).get(name, project_seed)))


def _loader(config: dict[str, Any], split: str, processed_root: Path, augment: bool) -> DataLoader:
    train_cfg = config["train"]
    data_seed = _seed_from_config(config, "data_seed")
    sampler_seed = _seed_from_config(config, "sampler_seed")
    max_key = "max_train_samples" if split == "train" else "max_val_samples"
    image_id_files = train_cfg.get("image_id_files") or {}
    image_id_file = train_cfg.get(f"{split}_image_ids") or image_id_files.get(split)
    sample_id_files = train_cfg.get("sample_id_files") or train_cfg.get("scanpath_sample_id_files") or {}
    sample_id_file = train_cfg.get(f"{split}_sample_ids") or sample_id_files.get(split)
    task = str(train_cfg.get("task", "saliency")).lower()
    if task in {"next_fixation", "next-fixation", "scanpath_next"}:
        negative_prefix_mode = train_cfg.get("scanpath_negative_prefix_modes", train_cfg.get("scanpath_negative_prefix_mode"))
        dataset = ProcessedNextFixationDataset(
            processed_root,
            split=split,
            prefix_length=int(train_cfg.get("scanpath_prefix_length", 3)),
            prefix_mode=str(train_cfg.get("scanpath_prefix_mode", "normal")),
            prefix_encoding=str(train_cfg.get("scanpath_prefix_encoding", "collapsed")),
            negative_prefix_mode=negative_prefix_mode,
            augment_roll=augment,
            augment_roll_degrees=train_cfg.get("augment_roll_degrees"),
            max_samples=train_cfg.get(max_key),
            image_id_file=image_id_file,
            sample_id_file=sample_id_file,
            sigma_px=float(config.get("data", {}).get("sigma_px", 10.0)),
            area_weighted=bool(config.get("data", {}).get("area_weighted", True)),
            shuffle_seed=data_seed,
            start_geometry=bool(train_cfg.get("scanpath_start_geometry", False)),
        )
    else:
        dataset = ProcessedSaliencyDataset(
            processed_root,
            split=split,
            augment_roll=augment,
            augment_roll_degrees=train_cfg.get("augment_roll_degrees"),
            max_samples=train_cfg.get(max_key),
            image_id_file=image_id_file,
            shuffle_temporal_labels=bool(train_cfg.get("shuffle_temporal_labels", False)),
            shuffle_seed=data_seed,
            temporal_target_mode=str(train_cfg.get("temporal_target_mode", "correct")),
            condition_prefix_bins=int(train_cfg.get("condition_prefix_bins", 0) or 0),
            condition_prefix_step_channel=bool(train_cfg.get("condition_prefix_step_channel", False)),
            condition_startpoint_channels=bool(train_cfg.get("condition_startpoint_channels", False)),
            startpoint_condition_mode=str(train_cfg.get("startpoint_condition_mode", "actual")),
            startpoint_prior_sigma_px=float(train_cfg.get("startpoint_prior_sigma_px", config.get("data", {}).get("sigma_px", 10.0))),
            startpoint_random_seed=int(train_cfg.get("startpoint_random_seed", 20260710)),
        )
    sampler = None
    shuffle = split == "train"
    if split == "train" and bool(train_cfg.get("dataset_balanced_sampling", False)):
        sampler = _dataset_balanced_sampler(dataset, sampler_seed)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=True,
    )


def _batch_datasets(batch: dict[str, Any], size: int) -> list[str]:
    names = batch.get("dataset")
    if names is None:
        return ["unknown"] * size
    if isinstance(names, str):
        return [names] * size
    return [str(name) for name in names]


def _add_dataset_metrics(
    summary: dict[str, float],
    rows_by_dataset: dict[str, list[dict[str, float]]],
    prefix: str = "",
) -> None:
    dataset_scores = {
        name: mean_scores(rows)
        for name, rows in sorted(rows_by_dataset.items())
        if rows
    }
    for name, scores in dataset_scores.items():
        for key, value in scores.items():
            summary[f"dataset/{name}/{prefix}{key}"] = value
    macro = mean_scores(list(dataset_scores.values()))
    for key, value in macro.items():
        summary[f"macro_{prefix}{key}"] = value
    if not prefix:
        summary["num_datasets"] = float(len(dataset_scores))


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, config: dict[str, Any]) -> dict[str, float]:
    model.eval()
    train_cfg = config.get("train", {})
    validate_process = bool(train_cfg.get("validate_process_metrics", False))
    validate_nextfix = str(train_cfg.get("task", "saliency")).lower() in {"next_fixation", "next-fixation", "scanpath_next"}
    loss_rows: list[dict[str, float]] = []
    metric_rows: list[dict[str, float]] = []
    metric_rows_by_dataset: dict[str, list[dict[str, float]]] = defaultdict(list)
    process_rows: list[dict[str, float]] = []
    process_rows_by_dataset: dict[str, list[dict[str, float]]] = defaultdict(list)
    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(batch["image"])
        _, logs = saliency_training_loss(outputs, batch, config)
        loss_rows.append(logs)
        pred = outputs["final"].detach().cpu().numpy()
        gt = batch["final"].detach().cpu().numpy()
        fixation = batch["fixation"].detach().cpu().numpy()
        area = batch["area_weights"].detach().cpu().numpy()
        dataset_names = _batch_datasets(batch, pred.shape[0])
        for i in range(pred.shape[0]):
            scores = saliency_scores(pred[i, 0], gt[i, 0], fixation[i, 0], area[i, 0]).as_dict()
            if validate_nextfix:
                scores.update(next_fixation_scores(pred[i, 0], fixation[i, 0]))
            metric_rows.append(scores)
            metric_rows_by_dataset[dataset_names[i]].append(scores)
        if validate_process:
            temporal_pred = outputs["temporal"].detach().cpu().numpy()
            cumulative_pred = outputs["cumulative"].detach().cpu().numpy()
            temporal_gt = batch["temporal"].detach().cpu().numpy()
            cumulative_gt = batch["cumulative"].detach().cpu().numpy()
            image_ids = batch["image_id"]
            for i in range(pred.shape[0]):
                proc = process_scores(
                    {"temporal": temporal_pred[i], "cumulative": cumulative_pred[i]},
                    {
                        "temporal": temporal_gt[i],
                        "cumulative": cumulative_gt[i],
                        "area_weights": area[i, 0],
                    },
                    str(image_ids[i]),
                )
                process_rows.append(proc)
                process_rows_by_dataset[dataset_names[i]].append(proc)
    losses = mean_scores(loss_rows)
    metrics = mean_scores(metric_rows)
    summary = {**{f"loss/{k}": v for k, v in losses.items()}, **metrics}
    _add_dataset_metrics(summary, metric_rows_by_dataset)
    if validate_process:
        process_metrics = mean_scores(process_rows)
        summary.update({f"process/{k}": v for k, v in process_metrics.items()})
        _add_dataset_metrics(summary, process_rows_by_dataset, prefix="process_")
    return summary


@torch.no_grad()
def save_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: str | Path,
    show_progress: bool = True,
) -> None:
    model.eval()
    out_dir = ensure_dir(out_dir)
    for batch in tqdm(loader, desc=f"Saving predictions to {out_dir}", disable=not show_progress):
        batch = _move_batch(batch, device)
        outputs = model(batch["image"])
        temporal = outputs["temporal"].detach().cpu().numpy()
        cumulative = outputs["cumulative"].detach().cpu().numpy()
        final = outputs["final"].detach().cpu().numpy()
        image_ids = batch["image_id"]
        for i, image_id in enumerate(image_ids):
            np.savez_compressed(
                out_dir / f"{image_id}.npz",
                final=final[i, 0].astype(np.float32),
                temporal=temporal[i].astype(np.float32),
                cumulative=cumulative[i].astype(np.float32),
            )


def train_model(config: dict[str, Any], model_name: str | None = None, device_name: str | None = None) -> Path:
    seed_everything(int(config["project"].get("seed", 20260708)))
    train_cfg = config["train"]
    if model_name:
        train_cfg["model"] = model_name
    if device_name:
        train_cfg["device"] = device_name
    model_name = train_cfg.get("model", "dytsaldmm")
    device = _device(train_cfg.get("device", "cuda"))

    processed_root = Path(config["paths"]["processed_root"]) / config["data"]["processed_name"]
    if not (processed_root / "manifest.jsonl").exists():
        raise FileNotFoundError(f"Missing processed dataset manifest: {processed_root / 'manifest.jsonl'}")

    experiment = train_cfg.get("experiment") or f"{model_name}_{config['data']['dataset']}"
    run_dir = ensure_dir(Path(config["paths"]["runs_root"]) / experiment)
    ensure_dir(run_dir / "checkpoints")
    save_config(config, run_dir / "config.yaml")

    train_loader = _loader(config, "train", processed_root, augment=bool(train_cfg.get("augment_roll", True)))
    val_loader = _loader(config, "val", processed_root, augment=False)

    model = build_model(model_name, config).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    selection_metric = str(train_cfg.get("selection_metric", "kld_sph"))
    selection_mode = str(train_cfg.get("selection_mode", "min")).lower()
    if selection_mode not in {"min", "max"}:
        raise ValueError(f"train.selection_mode must be 'min' or 'max', got {selection_mode!r}")
    best_score = float("inf") if selection_mode == "min" else -float("inf")
    show_progress = bool(train_cfg.get("progress_bar", True))
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(train_cfg.get("epochs", 3)) + 1):
        model.train()
        epoch_logs: list[dict[str, float]] = []
        pbar = tqdm(train_loader, desc=f"{experiment} epoch {epoch}", disable=not show_progress)
        for batch in pbar:
            batch = _move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch["image"])
                if "negative_images" in batch and float(config.get("loss", {}).get("order_rank", 0.0)) > 0.0:
                    b, n, c, h, w = batch["negative_images"].shape
                    neg_out = model(batch["negative_images"].reshape(b * n, c, h, w))["final"]
                    outputs["negative_finals"] = neg_out.reshape(b, n, *neg_out.shape[1:])
                elif "negative_image" in batch and float(config.get("loss", {}).get("order_rank", 0.0)) > 0.0:
                    outputs["negative_final"] = model(batch["negative_image"])["final"]
                loss, logs = saliency_training_loss(outputs, batch, config)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_logs.append(logs)
            pbar.set_postfix(loss=f"{logs['loss']:.4f}", kld=f"{logs['final_kl']:.4f}")
        train_summary = mean_scores(epoch_logs)
        val_summary = validate(model, val_loader, device, config)
        row = {"epoch": epoch, "train": train_summary, "val": val_summary}
        history.append(row)
        with open(run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        fallback = val_summary.get("kld_sph", val_summary.get("kld", float("inf")))
        selected_score = float(val_summary.get(selection_metric, fallback))
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "config": config,
            "val": val_summary,
            "selection_metric": selection_metric,
            "selection_score": selected_score,
        }
        torch.save(ckpt, run_dir / "checkpoints" / "last.pt")
        improved = selected_score < best_score if selection_mode == "min" else selected_score > best_score
        if improved:
            best_score = selected_score
            torch.save(ckpt, run_dir / "checkpoints" / "best.pt")
        if bool(train_cfg.get("log_epoch_details", True)):
            print(f"[epoch {epoch}] train={train_summary} val={val_summary}")
        else:
            print(
                f"[epoch {epoch}] "
                f"train_loss={train_summary.get('loss', float('nan')):.6f} "
                f"val_kld_sph={val_summary.get('kld_sph', float('nan')):.6f} "
                f"val_sim_sph={val_summary.get('sim_sph', float('nan')):.6f} "
                f"macro_kld_sph={val_summary.get('macro_kld_sph', float('nan')):.6f} "
                f"selected={selected_score:.6f} best={best_score:.6f}"
            )

    best_path = run_dir / "checkpoints" / "best.pt"
    if bool(train_cfg.get("save_predictions", True)) and best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])
        save_predictions(model, val_loader, device, run_dir / "predictions" / "val", show_progress=show_progress)
    return run_dir


def load_model_for_inference(checkpoint: str | Path, device_name: str = "cuda") -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    device = _device(device_name)
    state = torch.load(checkpoint, map_location=device)
    config = state["config"]
    model_name = config["train"].get("model", "dytsaldmm")
    model = build_model(model_name, config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, config, device
