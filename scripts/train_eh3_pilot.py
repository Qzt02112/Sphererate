#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.config import load_config, save_config
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.tile_prefetch import gaze_mass_coverage, rank_tiles
from sphererate.losses import (
    raw_dwell_frontier_rank_loss,
    raw_dwell_soft_topk_coverage_loss,
    raw_dwell_consensus_rank_loss,
    raw_dwell_tile_kl_loss,
    semantic_log_ratio_loss,
    tile_mass_kl_loss,
    tile_pairwise_rank_loss,
    weighted_cc,
    weighted_kld,
    weighted_sim,
)
from sphererate.models.early_horizon import TemplateTimeGatedResidualNet
from sphererate.models.spherical_dwell_rate import SphericalDwellRateNet
from sphererate.utils import ensure_dir, seed_everything


def _device(name: str) -> torch.device:
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def _model(config: dict[str, Any], mode: str) -> nn.Module:
    cfg = config["model"]
    if str(cfg.get("family", "")).lower() == "spherical_dwell_rate":
        return SphericalDwellRateNet(
            in_channels=int(cfg["in_channels"]),
            base_channels=int(cfg["base_channels"]),
            hidden_channels=int(cfg["hidden_channels"]),
            horizons_s=config["data"]["horizons_s"],
            semantic_backbone=str(cfg["semantic_backbone"]),
            semantic_layer=str(cfg["semantic_layer"]),
            semantic_pretrained=bool(cfg["semantic_pretrained"]),
            semantic_alpha_init=cfg.get("semantic_alpha_init", [0.05, 0.20, 0.65, 0.90]),
            reachability_beta_init=cfg.get("reachability_beta_init", [16.0, 4.0, 0.45, 0.10]),
            reachability_mode=str(cfg.get("reachability_mode", "isotropic")),
            reachability_north_beta_init=cfg.get("reachability_north_beta_init"),
            reachability_offset_deg_init=cfg.get("reachability_offset_deg_init"),
            maximum_reachability_offset_deg=float(
                cfg.get("maximum_reachability_offset_deg", 30.0)
            ),
            interval_residual_scale=float(cfg.get("interval_residual_scale", 2.0)),
            use_interval_residual=bool(cfg.get("use_interval_residual", True)),
            use_start_cost=bool(cfg.get("use_start_cost", True)),
            accumulation_mode=str(cfg.get("accumulation_mode", "cumulative_rate")),
        )
    return TemplateTimeGatedResidualNet(
        in_channels=int(cfg["in_channels"]),
        base_channels=int(cfg["base_channels"]),
        hidden_channels=int(cfg["hidden_channels"]),
        horizons_s=config["data"]["horizons_s"],
        conditioning_mode=mode,
        semantic_backbone=str(cfg["semantic_backbone"]),
        semantic_layer=str(cfg["semantic_layer"]),
        semantic_pretrained=bool(cfg["semantic_pretrained"]),
        residual_fusion=str(cfg.get("residual_fusion", "monotonic_gate")),
        residual_bound=float(cfg.get("residual_bound", 4.0)),
        semantic_alpha_init=cfg.get("semantic_alpha_init"),
        tile_rows=int(cfg.get("tile_rows", 8)),
        tile_cols=int(cfg.get("tile_cols", 16)),
        tile_hidden_channels=int(cfg.get("tile_hidden_channels", 16)),
        tile_residual_scale=float(cfg.get("tile_residual_scale", 0.5)),
        pixel_hidden_channels=int(cfg.get("pixel_hidden_channels", 16)),
        pixel_residual_scale=float(cfg.get("pixel_residual_scale", 0.5)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initialize_semantic_anchor(
    model: nn.Module,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    model_cfg = config["model"]
    raw_path = model_cfg.get("semantic_anchor_checkpoint")
    if raw_path is None:
        return None
    if model.residual_fusion not in {"semantic_anchor", "spherical_dwell_rate"}:
        raise ValueError(
            "semantic_anchor_checkpoint requires semantic_anchor or spherical_dwell_rate"
        )
    path = Path(str(raw_path))
    if not path.is_file():
        raise FileNotFoundError(path)
    source = torch.load(path, map_location="cpu")
    source_state = source["model"]
    target_state = model.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    for target_key, target_value in target_state.items():
        source_key: str | None = None
        if target_key.startswith("features."):
            source_key = target_key
        elif target_key.startswith("semantic_block."):
            source_key = (
                target_key
                if target_key in source_state
                else target_key.replace("semantic_block.", "final_head.0.", 1)
            )
        elif target_key.startswith("semantic_out."):
            source_key = (
                target_key
                if target_key in source_state
                else target_key.replace("semantic_out.", "final_head.1.", 1)
            )
        if source_key is None:
            continue
        if source_key not in source_state:
            raise KeyError(f"Missing semantic-anchor source tensor: {source_key}")
        if source_state[source_key].shape != target_value.shape:
            raise ValueError(
                f"Semantic-anchor shape mismatch: {source_key} {tuple(source_state[source_key].shape)} "
                f"!= {target_key} {tuple(target_value.shape)}"
            )
        transferred[target_key] = source_state[source_key]
    expected = [
        key
        for key in target_state
        if key.startswith(("features.", "semantic_block.", "semantic_out."))
    ]
    if set(transferred) != set(expected):
        missing = sorted(set(expected) - set(transferred))
        raise RuntimeError(f"Incomplete semantic-anchor transfer: {missing}")
    model.load_state_dict(transferred, strict=False)
    if bool(model_cfg.get("freeze_semantic_anchor_features", False)):
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return {
        "type": "matched_static_semantic_anchor",
        "checkpoint": str(path),
        "sha256": _sha256(path),
        "source_epoch": int(source.get("epoch", -1)),
        "source_val": source.get("val"),
        "transferred_tensor_count": len(transferred),
        "features_frozen": bool(model_cfg.get("freeze_semantic_anchor_features", False)),
    }


def _loss(
    output: dict[str, torch.Tensor],
    template: torch.Tensor,
    target: torch.Tensor,
    final_target: torch.Tensor,
    raw_dwell_tiles: torch.Tensor,
    topk_consensus: torch.Tensor | None,
    area: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = output["cumulative"]
    kld = weighted_kld(prediction, target, area)
    correlation = weighted_cc(prediction, target, area)
    train_cfg = config["train"]
    ratio_weight = float(train_cfg.get("semantic_ratio_weight", 0.0))
    tile_kl_weight = float(train_cfg.get("tile_kl_weight", 0.0))
    tile_rank_weight = float(train_cfg.get("tile_rank_weight", 0.0))
    sim_weight = float(train_cfg.get("sim_weight", 0.0))
    raw_tile_kl_weight = float(train_cfg.get("raw_tile_kl_weight", 0.0))
    raw_frontier_rank_weight = float(train_cfg.get("raw_frontier_rank_weight", 0.0))
    soft_topk_coverage_weight = float(train_cfg.get("soft_topk_coverage_weight", 0.0))
    consensus_rank_weight = float(train_cfg.get("consensus_rank_weight", 0.0))
    semantic_anchor_weight = float(train_cfg.get("semantic_anchor_weight", 0.0))
    temporal_anchor_weight = float(train_cfg.get("temporal_anchor_weight", 0.0))
    zero = prediction.new_zeros(())
    similarity = weighted_sim(prediction, target, area) if sim_weight else zero
    ratio = (
        semantic_log_ratio_loss(
            output["effective_residual_logits"],
            template,
            target,
            area,
            clip=float(train_cfg.get("semantic_ratio_clip", 4.0)),
        )
        if ratio_weight
        else zero
    )
    tile_kl = (
        tile_mass_kl_loss(
            prediction,
            target,
            area,
            tile_rows=int(train_cfg.get("tile_rows", 8)),
            tile_cols=int(train_cfg.get("tile_cols", 16)),
        )
        if tile_kl_weight
        else zero
    )
    tile_rank = (
        tile_pairwise_rank_loss(
            prediction,
            target,
            area,
            tile_rows=int(train_cfg.get("tile_rows", 8)),
            tile_cols=int(train_cfg.get("tile_cols", 16)),
            budgets_percent=tuple(
                float(value) for value in train_cfg.get("coverage_selection_budgets", [5.0, 10.0])
            ),
            temperature=float(train_cfg.get("tile_rank_temperature", 0.25)),
        )
        if tile_rank_weight
        else zero
    )
    horizon_weights = torch.as_tensor(
        train_cfg.get("tile_horizon_weights", [1.0] * prediction.shape[1]),
        dtype=prediction.dtype,
        device=prediction.device,
    )
    raw_tile_kl = (
        raw_dwell_tile_kl_loss(
            prediction,
            raw_dwell_tiles,
            area,
            tile_rows=int(train_cfg.get("tile_rows", 8)),
            tile_cols=int(train_cfg.get("tile_cols", 16)),
            horizon_weights=horizon_weights,
        )
        if raw_tile_kl_weight
        else zero
    )
    raw_frontier_rank = (
        raw_dwell_frontier_rank_loss(
            prediction,
            raw_dwell_tiles,
            area,
            tile_rows=int(train_cfg.get("tile_rows", 8)),
            tile_cols=int(train_cfg.get("tile_cols", 16)),
            budgets_percent=tuple(
                float(value)
                for value in train_cfg.get("coverage_selection_budgets", [5.0, 10.0, 20.0])
            ),
            frontier_width=int(train_cfg.get("raw_frontier_width", 4)),
            temperature=float(train_cfg.get("raw_frontier_temperature", 0.25)),
            horizon_weights=horizon_weights,
        )
        if raw_frontier_rank_weight
        else zero
    )
    soft_topk_coverage = (
        raw_dwell_soft_topk_coverage_loss(
            prediction,
            raw_dwell_tiles,
            area,
            tile_rows=int(train_cfg.get("tile_rows", 8)),
            tile_cols=int(train_cfg.get("tile_cols", 16)),
            budgets_percent=tuple(
                float(value)
                for value in train_cfg.get("coverage_selection_budgets", [5.0, 10.0, 20.0])
            ),
            epsilon=float(train_cfg.get("soft_topk_epsilon", 0.15)),
            iterations=int(train_cfg.get("soft_topk_iterations", 30)),
            horizon_weights=horizon_weights,
        )
        if soft_topk_coverage_weight
        else zero
    )
    consensus_rank = (
        raw_dwell_consensus_rank_loss(
            prediction,
            topk_consensus,
            area,
            tile_rows=int(train_cfg.get("tile_rows", 8)),
            tile_cols=int(train_cfg.get("tile_cols", 16)),
            temperature=float(train_cfg.get("consensus_rank_temperature", 0.25)),
            minimum_difference=float(train_cfg.get("consensus_minimum_difference", 0.1)),
            horizon_weights=horizon_weights,
        )
        if consensus_rank_weight and topk_consensus is not None
        else zero
    )
    if semantic_anchor_weight:
        if "semantic_map" not in output:
            raise ValueError("semantic_anchor_weight requires model output['semantic_map']")
        semantic_anchor_kld = weighted_kld(output["semantic_map"], final_target, area)
        semantic_anchor_cc = weighted_cc(output["semantic_map"], final_target, area)
    else:
        semantic_anchor_kld = zero
        semantic_anchor_cc = zero
    if temporal_anchor_weight:
        if "temporal_semantic_maps" not in output:
            raise ValueError("temporal_anchor_weight requires temporal_semantic_maps")
        temporal_anchor_kld = weighted_kld(output["temporal_semantic_maps"], target, area)
        temporal_anchor_cc = weighted_cc(output["temporal_semantic_maps"], target, area)
        temporal_anchor_sim = weighted_sim(output["temporal_semantic_maps"], target, area)
    else:
        temporal_anchor_kld = zero
        temporal_anchor_cc = zero
        temporal_anchor_sim = zero
    total = (
        float(train_cfg["kld_weight"]) * kld
        + float(train_cfg["cc_weight"]) * (1.0 - correlation)
        + sim_weight * (1.0 - similarity)
        + ratio_weight * ratio
        + tile_kl_weight * tile_kl
        + tile_rank_weight * tile_rank
        + raw_tile_kl_weight * raw_tile_kl
        + raw_frontier_rank_weight * raw_frontier_rank
        + soft_topk_coverage_weight * soft_topk_coverage
        + consensus_rank_weight * consensus_rank
        + semantic_anchor_weight
        * (
            semantic_anchor_kld
            + float(train_cfg.get("semantic_anchor_cc_weight", 0.5))
            * (1.0 - semantic_anchor_cc)
        )
        + temporal_anchor_weight
        * (
            temporal_anchor_kld
            + float(train_cfg.get("temporal_anchor_cc_weight", 0.5))
            * (1.0 - temporal_anchor_cc)
            + float(train_cfg.get("temporal_anchor_sim_weight", 0.25))
            * (1.0 - temporal_anchor_sim)
        )
    )
    return total, {
        "loss": float(total.detach()),
        "kld": float(kld.detach()),
        "cc": float(correlation.detach()),
        "sim": float(similarity.detach()),
        "semantic_ratio": float(ratio.detach()),
        "tile_kl": float(tile_kl.detach()),
        "tile_rank": float(tile_rank.detach()),
        "raw_tile_kl": float(raw_tile_kl.detach()),
        "raw_frontier_rank": float(raw_frontier_rank.detach()),
        "soft_topk_coverage": float(soft_topk_coverage.detach()),
        "consensus_rank": float(consensus_rank.detach()),
        "semantic_anchor_kld": float(semantic_anchor_kld.detach()),
        "semantic_anchor_cc": float(semantic_anchor_cc.detach()),
        "temporal_anchor_kld": float(temporal_anchor_kld.detach()),
        "temporal_anchor_cc": float(temporal_anchor_cc.detach()),
        "temporal_anchor_sim": float(temporal_anchor_sim.detach()),
    }


@torch.no_grad()
def _validate(
    model: TemplateTimeGatedResidualNet,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    horizon_values = [float(value) for value in config["data"]["horizons_s"]]
    primary = {float(value) for value in config["data"]["primary_horizons_s"]}
    rows: list[dict[str, float]] = []
    coverage_rows: list[dict[str, float]] = []
    gate_values: list[np.ndarray] = []
    train_cfg = config["train"]
    tile_rows = int(train_cfg.get("tile_rows", 8))
    tile_cols = int(train_cfg.get("tile_cols", 16))
    budgets = tuple(float(value) for value in train_cfg.get("coverage_budgets", [5.0, 10.0, 20.0]))
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        template = batch["template"].to(device, non_blocking=True)
        area = batch["area_weights"].to(device, non_blocking=True)
        output = model(
            image,
            template,
            start_geometry=batch["start_geometry"].to(device, non_blocking=True),
        )
        gate_values.append(output["gates"].detach().cpu().numpy())
        predictions_np = output["cumulative"].detach().float().cpu().numpy()
        area_np = area.detach().float().cpu().numpy()
        for sample_index in range(image.shape[0]):
            for index, horizon in enumerate(horizon_values):
                prediction_h = output["cumulative"][
                    sample_index : sample_index + 1, index : index + 1
                ]
                target_h = target[sample_index : sample_index + 1, index : index + 1]
                area_h = area[sample_index : sample_index + 1]
                rows.append(
                    {
                        "horizon_s": horizon,
                        "kld_sph": float(weighted_kld(prediction_h, target_h, area_h)),
                        "sim_sph": float(weighted_sim(prediction_h, target_h, area_h)),
                        "cc_sph": float(weighted_cc(prediction_h, target_h, area_h)),
                    }
                )
        for sample_index, fixations_path in enumerate(batch["fixations_path"]):
            with np.load(str(fixations_path)) as fixation_data:
                fixations = np.asarray(fixation_data["fixations"], dtype=np.float32)
            observer_ids = np.unique(fixations[:, 3]).astype(int).tolist()
            for horizon_index, horizon in enumerate(horizon_values):
                for budget in budgets:
                    selected = rank_tiles(
                        predictions_np[sample_index, horizon_index],
                        area_np[sample_index, 0],
                        tile_rows=tile_rows,
                        tile_cols=tile_cols,
                        budget_percent=budget,
                    )
                    coverage_rows.append(
                        {
                            "horizon_s": horizon,
                            "budget_percent": budget,
                            "coverage": gaze_mass_coverage(
                                fixations,
                                observer_ids=observer_ids,
                                horizon_s=horizon,
                                selected_tile_ids=selected,
                                tile_rows=tile_rows,
                                tile_cols=tile_cols,
                            ),
                        }
                    )
    per_horizon = {
        f"{horizon:g}": {
            metric: float(np.mean([row[metric] for row in rows if row["horizon_s"] == horizon]))
            for metric in ("kld_sph", "sim_sph", "cc_sph")
        }
        for horizon in horizon_values
    }
    for horizon in horizon_values:
        for budget in budgets:
            per_horizon[f"{horizon:g}"][f"coverage_{budget:g}"] = float(
                np.mean(
                    [
                        row["coverage"]
                        for row in coverage_rows
                        if row["horizon_s"] == horizon and row["budget_percent"] == budget
                    ]
                )
            )
    macro_rows = [row for row in rows if row["horizon_s"] in primary]
    result = {
        "early_macro_kld_sph": float(np.mean([row["kld_sph"] for row in macro_rows])),
        "early_macro_sim_sph": float(np.mean([row["sim_sph"] for row in macro_rows])),
        "early_macro_cc_sph": float(np.mean([row["cc_sph"] for row in macro_rows])),
        "per_horizon": per_horizon,
        "gates": np.mean(np.concatenate(gate_values, axis=0), axis=0).tolist(),
    }
    for budget in budgets:
        result[f"early_macro_coverage_{budget:g}"] = float(
            np.mean(
                [
                    row["coverage"]
                    for row in coverage_rows
                    if row["horizon_s"] in primary and row["budget_percent"] == budget
                ]
            )
        )
        result[f"all_macro_coverage_{budget:g}"] = float(
            np.mean(
                [
                    row["coverage"]
                    for row in coverage_rows
                    if row["budget_percent"] == budget
                ]
            )
        )
    selection_budgets = tuple(
        float(value) for value in train_cfg.get("coverage_selection_budgets", [5.0, 10.0])
    )
    result["coverage_selection_mean"] = float(
        np.mean([result[f"early_macro_coverage_{budget:g}"] for budget in selection_budgets])
    )
    reference = train_cfg.get("selection_reference")
    if reference:
        margins = []
        for budget in selection_budgets:
            key = f"coverage_{budget:g}"
            margins.append(
                result[f"early_macro_coverage_{budget:g}"]
                - float(reference["early_macro"][key])
            )
            margins.append(
                result["per_horizon"]["5"][key]
                - float(reference["horizon_5"][key])
            )
        result["coverage_margins_vs_reference"] = margins
        result["worst_coverage_margin_vs_reference"] = float(min(margins))
        endpoint_margins = list(margins)
        endpoint_margins.extend(
            [
                float(reference["early_macro"]["kld_sph"])
                - result["early_macro_kld_sph"],
                result["early_macro_sim_sph"]
                - float(reference["early_macro"]["sim_sph"]),
                result["early_macro_cc_sph"]
                - float(reference["early_macro"]["cc_sph"]),
                float(reference["horizon_5"]["kld_sph"])
                - result["per_horizon"]["5"]["kld_sph"],
                result["per_horizon"]["5"]["sim_sph"]
                - float(reference["horizon_5"]["sim_sph"]),
                result["per_horizon"]["5"]["cc_sph"]
                - float(reference["horizon_5"]["cc_sph"]),
            ]
        )
        result["endpoint_margins_vs_reference"] = endpoint_margins
        result["worst_endpoint_margin_vs_reference"] = float(min(endpoint_margins))
    return result


@torch.no_grad()
def _save_predictions(
    model: TemplateTimeGatedResidualNet,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> None:
    ensure_dir(out_dir)
    model.eval()
    for batch in loader:
        output = model(
            batch["image"].to(device, non_blocking=True),
            batch["template"].to(device, non_blocking=True),
            start_geometry=batch["start_geometry"].to(device, non_blocking=True),
        )["cumulative"].detach().cpu().numpy()
        for index, image_id in enumerate(batch["image_id"]):
            np.savez_compressed(out_dir / f"{image_id}.npz", cumulative=output[index].astype(np.float32))


def train_run(
    config: dict[str, Any],
    run: dict[str, Any],
    *,
    smoke: bool,
    init_checkpoint: str | Path | None = None,
) -> Path:
    # Per-run overrides make matched ablations explicit in the saved checkpoint
    # while keeping one frozen base configuration and one batch invocation.
    config = json.loads(json.dumps(config))
    for section in ("model", "train"):
        overrides = run.get(f"{section}_overrides", {})
        if not isinstance(overrides, dict):
            raise TypeError(f"{section}_overrides must be a mapping")
        config[section].update(overrides)
    seed = int(config["train"]["seed"])
    seed_everything(seed)
    device = _device(str(config["train"]["device"]))
    processed_root = Path(config["paths"]["processed_root"]) / config["paths"]["processed_name"]
    tile_rows = int(config["train"].get("tile_rows", 8))
    tile_cols = int(config["train"].get("tile_cols", 16))
    consensus_root = config["paths"].get("consensus_root")
    train_dataset: Any = EarlyHorizonDataset(
        processed_root,
        split="train",
        target_mode=str(run["target_mode"]),
        shuffle_seed=int(config["train"]["data_seed"]),
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        consensus_root=consensus_root,
    )
    val_dataset: Any = EarlyHorizonDataset(
        processed_root,
        split="val",
        target_mode="correct",
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        consensus_root=None,
    )
    epochs = 1 if smoke else int(config["train"]["epochs"])
    if smoke:
        train_dataset = Subset(train_dataset, range(min(4, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(2, len(val_dataset))))
    generator = torch.Generator().manual_seed(int(config["train"]["sampler_seed"])
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0 if smoke else int(config["train"]["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=0 if smoke else int(config["train"]["num_workers"]),
        pin_memory=True,
    )
    model = _model(config, str(run["conditioning_mode"]))
    initialization = _initialize_semantic_anchor(model, config)
    model = model.to(device)
    if init_checkpoint is not None:
        if initialization is not None:
            raise ValueError("Full init_checkpoint and semantic_anchor_checkpoint are mutually exclusive")
        init_path = Path(init_checkpoint)
        state = torch.load(init_path, map_location=device)
        allow_partial = bool(config["model"].get("allow_partial_init", False))
        incompatible = model.load_state_dict(state["model"], strict=not allow_partial)
        if allow_partial:
            allowed_prefixes = tuple(
                str(value)
                for value in config["model"].get(
                    "partial_init_allowed_prefixes", ["tile_reduce.", "tile_codes", "tile_out."]
                )
            )
            invalid_missing = [
                key for key in incompatible.missing_keys if not key.startswith(allowed_prefixes)
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Invalid partial initialization: missing={invalid_missing}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
        if bool(config["model"].get("initialize_temporal_anchor_from_static", False)):
            if not hasattr(model, "temporal_semantic_out"):
                raise ValueError("temporal-anchor initialization requires temporal_semantic_out")
            with torch.no_grad():
                repeats = int(model.temporal_semantic_out.conv.weight.shape[0])
                model.temporal_semantic_out.conv.weight.copy_(
                    model.semantic_out.conv.weight.repeat(repeats, 1, 1, 1)
                )
                model.temporal_semantic_out.conv.bias.copy_(
                    model.semantic_out.conv.bias.repeat(repeats)
                )
        initialization = {
            "checkpoint": str(init_path),
            "source_epoch": int(state["epoch"]),
            "source_run": str(init_path.parents[1]),
        }
    if bool(config["model"].get("freeze_semantic_anchor_features", False)):
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    if bool(config["model"].get("freeze_semantic_anchor_map", False)):
        for module_name in ("semantic_block", "semantic_out"):
            module = getattr(model, module_name)
            for parameter in module.parameters():
                parameter.requires_grad = False
    parameters = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    use_amp = bool(config["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    run_name = f"{run['name']}_smoke" if smoke else str(run["name"])
    run_dir = ensure_dir(Path(config["paths"]["runs_root"]) / run_name)
    ensure_dir(run_dir / "checkpoints")
    run_config = json.loads(json.dumps(config))
    run_config["run"] = dict(run)
    run_config["initialization"] = initialization
    run_config["train"]["epochs_effective"] = epochs
    save_config(run_config, run_dir / "config.yaml")
    selection_metric = str(config["train"].get("selection_metric", "early_macro_kld_sph"))
    coverage_selection = selection_metric == "coverage_5_10_with_kld_guard"
    v11_selection = selection_metric == "v11_worst_coverage_margin_with_density_guard"
    v11_endpoint_selection = selection_metric == "v11_worst_endpoint_margin_with_density_guard"
    maximize_selection = coverage_selection or v11_selection or v11_endpoint_selection
    best = float("-inf") if maximize_selection else float("inf")
    best_feasible = False
    fallback_kld = float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        if bool(config["model"].get("freeze_semantic_anchor_features", False)):
            model.features.eval()
        logs: list[dict[str, float]] = []
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            final_target = batch["final"].to(device, non_blocking=True)
            raw_dwell_tiles = batch["raw_dwell_tiles"].to(device, non_blocking=True)
            topk_consensus = (
                batch["raw_topk_consensus"].to(device, non_blocking=True)
                if "raw_topk_consensus" in batch
                else None
            )
            template = batch["template"].to(device, non_blocking=True)
            area = batch["area_weights"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(
                    image,
                    template,
                    start_geometry=batch["start_geometry"].to(device, non_blocking=True),
                )
                loss, current = _loss(
                    output,
                    template,
                    target,
                    final_target,
                    raw_dwell_tiles,
                    topk_consensus,
                    area,
                    config,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            logs.append(current)
        validation = _validate(model, val_loader, device, config)
        row = {
            "epoch": epoch,
            "train": {name: float(np.mean([item[name] for item in logs])) for name in logs[0]},
            "val": validation,
        }
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        val_kld = float(validation["early_macro_kld_sph"])
        if v11_endpoint_selection:
            score = float(validation["worst_endpoint_margin_vs_reference"])
        elif v11_selection:
            score = float(validation["worst_coverage_margin_vs_reference"])
        elif coverage_selection:
            score = float(validation["coverage_selection_mean"])
        else:
            score = val_kld
        guard = float(config["train"].get("kld_guard_max", float("inf")))
        feasible = val_kld <= guard
        if v11_selection or v11_endpoint_selection:
            density_guard = config["train"].get("density_guard", {})
            early = density_guard.get("early_macro", {})
            late = density_guard.get("horizon_5", {})
            feasible = (
                val_kld <= float(early.get("kld_max", float("inf")))
                and float(validation["early_macro_cc_sph"])
                >= float(early.get("cc_min", float("-inf")))
                and float(validation["per_horizon"]["5"]["kld_sph"])
                <= float(late.get("kld_max", float("inf")))
                and float(validation["per_horizon"]["5"]["cc_sph"])
                >= float(late.get("cc_min", float("-inf")))
            )
        selection = {
            "metric": selection_metric,
            "score": score,
            "kld_guard_max": guard if coverage_selection else None,
            "density_guard": (
                config["train"].get("density_guard")
                if (v11_selection or v11_endpoint_selection)
                else None
            ),
            "guard_pass": feasible if maximize_selection else None,
        }
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": run_config,
            "run": dict(run),
            "val": validation,
            "parameters": parameters,
            "initialization": initialization,
            "selection": selection,
        }
        torch.save(checkpoint, run_dir / "checkpoints" / "last.pt")
        improved = False
        if maximize_selection:
            if feasible and (not best_feasible or score > best):
                best = score
                best_feasible = True
                improved = True
            elif not best_feasible and val_kld < fallback_kld:
                fallback_kld = val_kld
                improved = True
        elif score < best:
            best = score
            improved = True
        if improved:
            torch.save(checkpoint, run_dir / "checkpoints" / "best.pt")
        best_label = (
            f"{best:.6f}"
            if (not maximize_selection or best_feasible)
            else f"fallback_kld={fallback_kld:.6f}"
        )
        print(
            f"[{run['role']}] epoch={epoch:02d} train={row['train']['loss']:.6f} "
            f"val_early_kld={val_kld:.6f} val_early_cc={validation['early_macro_cc_sph']:.6f} "
            f"val_cov_select={score:.6f} "
            f"guard={'pass' if feasible else 'fail'} best={best_label}",
            flush=True,
        )

    best_state = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(best_state["model"])
    _save_predictions(model, val_loader, device, run_dir / "predictions" / "val")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen EH3 matched pilot runs.")
    parser.add_argument("--config", default="configs/eh3_template_residual_256x512.yaml")
    parser.add_argument("--batch", default="configs/batch_eh3_pilot.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--role", help="Optional single role")
    parser.add_argument("--seed", type=int, help="Override all training/data/sampler seeds and run-name seed")
    parser.add_argument("--init-checkpoint", help="Optional matched SphereHorizon checkpoint used to initialize training")
    args = parser.parse_args()
    config = load_config(args.config)
    batch = load_config(args.batch)
    allowed_outer_status = {
        "sealed_not_materialized",
        "previously_used_under_v1_targets",
        "ordinary_repeatable_evaluation_never_model_selection",
    }
    if batch["outer_test_status"] not in allowed_outer_status:
        raise ValueError("Unknown test-access protocol")
    selected = [dict(run) for run in batch["runs"] if args.role is None or run["role"] == args.role]
    if not selected:
        raise ValueError("No matching EH3 runs")
    if args.seed is not None:
        seed = int(args.seed)
        config["project"]["seed"] = seed
        for key in ("seed", "data_seed", "sampler_seed"):
            config["train"][key] = seed
        for run in selected:
            run["name"] = re.sub(r"seed\d+", f"seed{seed}", str(run["name"]))
    outputs = [
        str(train_run(config, run, smoke=bool(args.smoke), init_checkpoint=args.init_checkpoint))
        for run in selected
    ]
    print(json.dumps({"runs": outputs}, indent=2))


if __name__ == "__main__":
    main()
