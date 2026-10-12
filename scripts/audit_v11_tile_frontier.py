#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.config import load_config
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.tile_prefetch import tile_probability_mass
from sphererate.models.early_horizon import TemplateTimeGatedResidualNet
from sphererate.models.registry import build_model
from sphererate.utils import ensure_dir, seed_everything


def _spherehorizon_from_checkpoint(checkpoint: dict[str, Any]) -> TemplateTimeGatedResidualNet:
    config = checkpoint["config"]
    model_cfg = config["model"]
    model = TemplateTimeGatedResidualNet(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        hidden_channels=int(model_cfg["hidden_channels"]),
        horizons_s=config["data"]["horizons_s"],
        conditioning_mode=str(checkpoint.get("run", {}).get("conditioning_mode", "time")),
        semantic_backbone=str(model_cfg["semantic_backbone"]),
        semantic_layer=str(model_cfg["semantic_layer"]),
        semantic_pretrained=False,
        residual_fusion=str(model_cfg.get("residual_fusion", "monotonic_gate")),
        residual_bound=float(model_cfg.get("residual_bound", 4.0)),
        semantic_alpha_init=model_cfg.get("semantic_alpha_init"),
    )
    model.load_state_dict(checkpoint["model"])
    return model


def _load_method(spec: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, str]:
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    family = str(spec["family"])
    if family == "spherehorizon":
        model = _spherehorizon_from_checkpoint(checkpoint)
    elif family == "registry":
        model = build_model(str(spec["registry_name"]), checkpoint["config"])
        model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError(f"Unknown method family: {family}")
    model = model.to(device).eval()
    return model, family


def _order(scores: np.ndarray) -> np.ndarray:
    flat = np.asarray(scores, dtype=np.float64).ravel()
    ids = np.arange(flat.size, dtype=np.int64)
    return np.lexsort((ids, -flat)).astype(np.int64)


def _frontier_row(
    *,
    method: str,
    content_id: str,
    horizon_s: float,
    budget_percent: float,
    prediction_mass: np.ndarray,
    raw_target: np.ndarray,
) -> dict[str, Any]:
    pred = np.asarray(prediction_mass, dtype=np.float64).ravel()
    target = np.asarray(raw_target, dtype=np.float64).ravel()
    target /= target.sum()
    count = min(len(pred), max(1, int(math.ceil(len(pred) * float(budget_percent) / 100.0))))
    pred_order = _order(pred)
    target_order = _order(target)
    pred_top = pred_order[:count]
    target_top = target_order[:count]
    candidate_top = pred_order[: min(len(pred), 2 * count)]
    target_top_set = set(target_top.tolist())
    pred_top_set = set(pred_top.tolist())
    candidate_set = set(candidate_top.tolist())
    overlap = target_top_set & pred_top_set
    recoverable = target_top_set & candidate_set
    oracle_mass = float(target[target_top].sum())
    coverage = float(target[pred_top].sum())
    return {
        "method": method,
        "content_id": content_id,
        "horizon_s": float(horizon_s),
        "budget_percent": float(budget_percent),
        "tile_count": int(count),
        "coverage": coverage,
        "oracle_coverage": oracle_mass,
        "oracle_gap": oracle_mass - coverage,
        "gt_top_tile_overlap": len(overlap) / float(count),
        "gt_top_mass_recall_at_2b": float(target[list(recoverable)].sum() / max(oracle_mass, 1e-12)),
        "cutoff_margin": float(pred[pred_order[count - 1]] - pred[pred_order[count]])
        if count < len(pred)
        else 0.0,
        "frontier_swaps": int(count - len(overlap)),
    }


@torch.no_grad()
def _audit_method(
    *,
    name: str,
    family: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    horizons: list[float],
    budgets: list[float],
    tile_rows: int,
    tile_cols: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        if family == "spherehorizon":
            prediction = model(
                image, batch["template"].to(device, non_blocking=True)
            )["cumulative"]
        else:
            prediction = model(image)["cumulative"]
        prediction = prediction.detach().float().cpu().numpy()
        area = batch["area_weights"].numpy()
        raw_target = batch["raw_dwell_tiles"].numpy()
        for sample_index, content_id in enumerate(batch["canonical_content_id"]):
            for horizon_index, horizon in enumerate(horizons):
                mass = tile_probability_mass(
                    prediction[sample_index, horizon_index],
                    area[sample_index, 0],
                    tile_rows=tile_rows,
                    tile_cols=tile_cols,
                )
                for budget in budgets:
                    rows.append(
                        _frontier_row(
                            method=name,
                            content_id=str(content_id),
                            horizon_s=horizon,
                            budget_percent=budget,
                            prediction_mass=mass,
                            raw_target=raw_target[sample_index, horizon_index],
                        )
                    )
    return rows


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    horizons: list[float],
    primary: set[float],
    budgets: list[float],
) -> dict[str, Any]:
    metrics = (
        "coverage",
        "oracle_coverage",
        "oracle_gap",
        "gt_top_tile_overlap",
        "gt_top_mass_recall_at_2b",
        "cutoff_margin",
        "frontier_swaps",
    )
    result: dict[str, Any] = {"per_horizon": {}, "early_macro": {}}
    for budget in budgets:
        budget_rows = [row for row in rows if row["budget_percent"] == budget]
        result["per_horizon"][f"{budget:g}"] = {}
        for horizon in horizons:
            current = [row for row in budget_rows if row["horizon_s"] == horizon]
            result["per_horizon"][f"{budget:g}"][f"{horizon:g}"] = {
                metric: float(np.mean([row[metric] for row in current])) for metric in metrics
            }
        early = [row for row in budget_rows if row["horizon_s"] in primary]
        result["early_macro"][f"{budget:g}"] = {
            metric: float(np.mean([row[metric] for row in early])) for metric in metrics
        }
    return result


def _summary(result: dict[str, Any]) -> str:
    lines = [
        "# V11 R1: Raw-dwell Tile Frontier Audit",
        "",
        "No training was performed. Only the Salient360 validation split was loaded; gate, test, and Sitzmann were forbidden.",
        "",
        "## Early-macro cutoff audit (0.5/1/3 s)",
        "",
        "| Method | Budget | Coverage | Oracle gap | GT overlap | GT-top mass recall@2B | Swaps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in result["methods"].items():
        for budget in ("5", "10", "20"):
            row = values["early_macro"][budget]
            lines.append(
                f"| {method} | {budget}% | {row['coverage']:.4f} | {row['oracle_gap']:.4f} | "
                f"{row['gt_top_tile_overlap']:.3f} | {row['gt_top_mass_recall_at_2b']:.3f} | "
                f"{row['frontier_swaps']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Late-horizon diagnosis (5 s)",
            "",
            "| Method | Budget | Coverage | Oracle gap | GT overlap | GT-top mass recall@2B |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, values in result["methods"].items():
        for budget in ("5", "10", "20"):
            row = values["per_horizon"][budget]["5"]
            lines.append(
                f"| {method} | {budget}% | {row['coverage']:.4f} | {row['oracle_gap']:.4f} | "
                f"{row['gt_top_tile_overlap']:.3f} | {row['gt_top_mass_recall_at_2b']:.3f} |"
            )
    lines.extend(
        [
            "",
            "`GT-top mass recall@2B` measures how much mass of the oracle top-B set is already present in the model's top-2B candidate pool. High recall with low Coverage supports a lightweight re-ranking correction; low recall indicates missing content candidates.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw-dwell tile decision frontiers without training.")
    parser.add_argument("--config", default="configs/v11_r1_frontier_audit.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    split = str(config["protocol"]["split"])
    if split != "val":
        raise PermissionError("V11 R1 is frozen to the validation split")
    seed_everything(int(config["project"]["seed"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tile_rows, tile_cols = (int(value) for value in config["protocol"]["tile_grid"])
    dataset = EarlyHorizonDataset(
        config["paths"]["processed_root"],
        split=split,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
    horizons = [float(value) for value in config["protocol"]["horizons_s"]]
    primary = {float(value) for value in config["protocol"]["primary_horizons_s"]}
    budgets = [float(value) for value in config["protocol"]["budgets_percent"]]
    all_rows: list[dict[str, Any]] = []
    methods: dict[str, Any] = {}
    for spec in config["methods"]:
        model, family = _load_method(spec, device)
        rows = _audit_method(
            name=str(spec["name"]),
            family=family,
            model=model,
            loader=loader,
            device=device,
            horizons=horizons,
            budgets=budgets,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
        )
        all_rows.extend(rows)
        methods[str(spec["name"])] = _aggregate(
            rows, horizons=horizons, primary=primary, budgets=budgets
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "protocol": config["protocol"],
        "content_count": len(dataset),
        "methods": methods,
        "rows": all_rows,
    }
    report_root = ensure_dir(config["paths"]["report_root"])
    (report_root / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (report_root / "summary.md").write_text(_summary(result), encoding="utf-8")
    print(report_root / "summary.md")


if __name__ == "__main__":
    main()
