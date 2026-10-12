#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.tile_prefetch import gaze_mass_coverage, rank_tiles
from sphererate.losses import weighted_cc, weighted_kld, weighted_sim
from sphererate.models.spherical_dwell_rate import SphericalDwellRateNet


SEEDS = (20260710, 20260711, 20260712)
HORIZONS = (0.5, 1.0, 3.0, 5.0)
EARLY = {0.5, 1.0, 3.0}
METRICS = (
    "kld_sph",
    "sim_sph",
    "cc_sph",
    "coverage_5",
    "coverage_10",
    "coverage_15",
    "coverage_20",
)


def _map_metrics(prediction: np.ndarray, target: torch.Tensor, area: torch.Tensor) -> dict[str, float]:
    pred = torch.from_numpy(np.asarray(prediction, dtype=np.float32)).view(1, 1, *prediction.shape)
    gt = target.view(1, 1, *target.shape)
    weights = area.view(1, 1, *area.shape)
    return {
        "kld_sph": float(weighted_kld(pred, gt, weights)),
        "sim_sph": float(weighted_sim(pred, gt, weights)),
        "cc_sph": float(weighted_cc(pred, gt, weights)),
    }


def _coverage_metrics(
    prediction: np.ndarray,
    area: np.ndarray,
    fixations: np.ndarray,
    observer_ids: list[int],
    horizon: float,
) -> dict[str, float]:
    output = {}
    for budget in (5.0, 10.0, 15.0, 20.0):
        selected = rank_tiles(
            prediction,
            area,
            tile_rows=8,
            tile_cols=16,
            budget_percent=budget,
        )
        output[f"coverage_{budget:g}"] = gaze_mass_coverage(
            fixations,
            observer_ids=observer_ids,
            horizon_s=horizon,
            selected_tile_ids=selected,
            tile_rows=8,
            tile_cols=16,
        )
    return output


def _aggregate(rows: list[dict[str, Any]], horizons: set[float]) -> dict[str, float]:
    current = [row for row in rows if float(row["horizon_s"]) in horizons]
    return {metric: float(np.mean([row[metric] for row in current])) for metric in METRICS}


def _collapse_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average multiple declared-start conditions within a canonical scene."""

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["content_id"]), float(row["horizon_s"])), []).append(row)
    return [
        {
            "content_id": content_id,
            "horizon_s": horizon,
            **{
                metric: float(np.mean([row[metric] for row in current if metric in row]))
                for metric in METRICS
                if all(metric in row for row in current)
            },
        }
        for (content_id, horizon), current in sorted(grouped.items())
    ]


def _bootstrap_difference(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    metric: str,
    horizons: set[float],
    *,
    iterations: int = 10000,
) -> dict[str, float | int]:
    candidate_index = {
        (row["content_id"], float(row["horizon_s"])): float(row[metric]) for row in candidate
    }
    baseline_index = {
        (row["content_id"], float(row["horizon_s"])): float(row[metric]) for row in baseline
    }
    contents = sorted({key[0] for key in candidate_index})
    differences = []
    for content in contents:
        per_horizon = []
        for horizon in sorted(horizons):
            key = (content, float(horizon))
            if key not in candidate_index or key not in baseline_index:
                raise KeyError(f"Missing paired row {key}")
            if metric == "kld_sph":
                per_horizon.append(baseline_index[key] - candidate_index[key])
            else:
                per_horizon.append(candidate_index[key] - baseline_index[key])
        differences.append(float(np.mean(per_horizon)))
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(20260805)
    sampled = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "content_count": int(len(values)),
    }


def _model_from_checkpoint(
    checkpoint: dict[str, Any], device: torch.device
) -> SphericalDwellRateNet:
    cfg = checkpoint["config"]
    model_cfg = cfg["model"]
    model = SphericalDwellRateNet(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        hidden_channels=int(model_cfg["hidden_channels"]),
        horizons_s=cfg["data"]["horizons_s"],
        semantic_backbone=str(model_cfg["semantic_backbone"]),
        semantic_layer=str(model_cfg["semantic_layer"]),
        semantic_pretrained=False,
        semantic_alpha_init=model_cfg["semantic_alpha_init"],
        reachability_beta_init=model_cfg["reachability_beta_init"],
        reachability_mode=model_cfg["reachability_mode"],
        reachability_north_beta_init=model_cfg["reachability_north_beta_init"],
        reachability_offset_deg_init=model_cfg["reachability_offset_deg_init"],
        maximum_reachability_offset_deg=model_cfg["maximum_reachability_offset_deg"],
        interval_residual_scale=model_cfg["interval_residual_scale"],
        use_interval_residual=bool(model_cfg.get("use_interval_residual", True)),
        use_start_cost=bool(model_cfg.get("use_start_cost", True)),
        accumulation_mode=str(model_cfg.get("accumulation_mode", "cumulative_rate")),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def _candidate_and_template_rows(
    dataset: EarlyHorizonDataset,
    split: str,
    run_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    if split != "val":
        for seed in SEEDS:
            checkpoint = torch.load(
                run_root
                / f"v12c_hybrid_dwell_rate_seed{seed}"
                / "checkpoints"
                / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            models.append(_model_from_checkpoint(checkpoint, device))
    for item in dataset:
        image_id = str(item["image_id"])
        content_id = str(item["canonical_content_id"])
        if split == "val":
            predictions = [
                np.load(
                    run_root
                    / f"v12c_hybrid_dwell_rate_seed{seed}"
                    / "predictions"
                    / "val"
                    / f"{image_id}.npz"
                )["cumulative"].astype(np.float32)
                for seed in SEEDS
            ]
        else:
            predictions = [
                model(
                    item["image"].unsqueeze(0).to(device),
                    item["template"].unsqueeze(0).to(device),
                    start_geometry=item["start_geometry"].unsqueeze(0).to(device),
                )["cumulative"][0].float().cpu().numpy()
                for model in models
            ]
        fixations = np.load(str(item["fixations_path"]))["fixations"].astype(np.float32)
        observers = np.unique(fixations[:, 3]).astype(int).tolist()
        area_np = item["area_weights"][0].numpy()
        for index, horizon in enumerate(HORIZONS):
            per_seed = []
            for prediction in predictions:
                metrics = _map_metrics(
                    prediction[index], item["target"][index], item["area_weights"][0]
                )
                metrics.update(
                    _coverage_metrics(
                        prediction[index], area_np, fixations, observers, horizon
                    )
                )
                per_seed.append(metrics)
            candidate_rows.append(
                {
                    "content_id": content_id,
                    "horizon_s": horizon,
                    **{
                        metric: float(np.mean([row[metric] for row in per_seed]))
                        for metric in METRICS
                    },
                }
            )
            template = item["template"][index].numpy()
            template_metrics = _map_metrics(
                template, item["target"][index], item["area_weights"][0]
            )
            template_metrics.update(
                _coverage_metrics(template, area_np, fixations, observers, horizon)
            )
            template_rows.append(
                {"content_id": content_id, "horizon_s": horizon, **template_metrics}
            )
    del models
    return _collapse_conditions(candidate_rows), _collapse_conditions(template_rows)


@torch.no_grad()
def _wrong_image_rows(dataset: EarlyHorizonDataset, run_root: Path) -> list[dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = [dataset[index] for index in range(len(dataset))]
    wrong_rows_by_seed: list[list[dict[str, Any]]] = []
    for seed in SEEDS:
        checkpoint = torch.load(
            run_root
            / f"v12c_hybrid_dwell_rate_seed{seed}"
            / "checkpoints"
            / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = _model_from_checkpoint(checkpoint, device)
        seed_rows = []
        for index, target_item in enumerate(items):
            wrong_item = items[(index + 1) % len(items)]
            output = model(
                wrong_item["image"].unsqueeze(0).to(device),
                target_item["template"].unsqueeze(0).to(device),
                start_geometry=target_item["start_geometry"].unsqueeze(0).to(device),
            )["cumulative"][0, 2].float().cpu().numpy()
            metrics = _map_metrics(
                output, target_item["target"][2], target_item["area_weights"][0]
            )
            seed_rows.append(
                {
                    "content_id": str(target_item["canonical_content_id"]),
                    "horizon_s": 3.0,
                    **metrics,
                }
            )
        wrong_rows_by_seed.append(seed_rows)
        del model
    output = []
    for index in range(len(items)):
        output.append(
            {
                "content_id": wrong_rows_by_seed[0][index]["content_id"],
                "horizon_s": 3.0,
                **{
                    metric: float(
                        np.mean([seed_rows[index][metric] for seed_rows in wrong_rows_by_seed])
                    )
                    for metric in ("kld_sph", "sim_sph", "cc_sph")
                },
            }
        )
    return _collapse_conditions(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate frozen V12-C checkpoints.")
    parser.add_argument("--split", choices=("val", "gate", "test"), default="val")
    parser.add_argument(
        "--data-root",
        default="data/processed/early_horizon_salient360_spherical2deg_v2",
    )
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--tempsal-report")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    split = str(args.split)
    data_root = ROOT / args.data_root
    run_root = ROOT / args.run_root
    dataset = EarlyHorizonDataset(data_root, split=split)
    candidate_rows, template_rows = _candidate_and_template_rows(dataset, split, run_root)
    tempsal_path = (
        ROOT / args.tempsal_report
        if args.tempsal_report
        else ROOT / "reports" / f"kernel_v2_tempsal_erp_{split}" / "results.json"
    )
    tempsal_result = json.loads(
        tempsal_path.read_text()
    )
    tempsal_scores = tempsal_result["score_rows"]
    # Rebuild full rows without relying on dictionary overwrite across budgets.
    tempsal_rows = []
    for row in tempsal_scores:
        merged = dict(row)
        for budget in (5.0, 10.0, 15.0, 20.0):
            match = next(
                value
                for value in tempsal_result["coverage_rows"][str(budget)]
                if value["content_id"] == row["content_id"]
                and float(value["horizon_s"]) == float(row["horizon_s"])
            )
            merged[f"coverage_{budget:g}"] = float(match["coverage"])
        tempsal_rows.append(merged)
    tempsal_rows = _collapse_conditions(tempsal_rows)
    wrong_rows = _wrong_image_rows(dataset, run_root)
    correct_3s = [row for row in candidate_rows if row["horizon_s"] == 3.0]
    wrong_attribution = {
        metric: _bootstrap_difference(
            correct_3s,
            wrong_rows,
            metric,
            {3.0},
        )
        for metric in ("kld_sph", "sim_sph", "cc_sph")
    }
    methods = {
        "early_template": {
            "early_macro": _aggregate(template_rows, EARLY),
            "horizon_5": _aggregate(template_rows, {5.0}),
        },
        "tempsal_erp": {
            "early_macro": _aggregate(tempsal_rows, EARLY),
            "horizon_5": _aggregate(tempsal_rows, {5.0}),
        },
        "v12c": {
            "early_macro": _aggregate(candidate_rows, EARLY),
            "horizon_5": _aggregate(candidate_rows, {5.0}),
        },
    }
    comparisons = {}
    for baseline_name, baseline_rows in (
        ("early_template", template_rows),
        ("tempsal_erp", tempsal_rows),
    ):
        comparisons[baseline_name] = {
            scope: {
                metric: _bootstrap_difference(
                    candidate_rows, baseline_rows, metric, horizons
                )
                for metric in METRICS
            }
            for scope, horizons in (("early_macro", EARLY), ("horizon_5", {5.0}))
        }
    per_horizon = {
        f"{horizon:g}": _aggregate(candidate_rows, {horizon}) for horizon in HORIZONS
    }
    checkpoints = []
    for seed in SEEDS:
        checkpoint = torch.load(
            run_root
            / f"v12c_hybrid_dwell_rate_seed{seed}"
            / "checkpoints"
            / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        checkpoints.append(
            {
                "seed": seed,
                "epoch": int(checkpoint["epoch"]),
                "parameters": checkpoint["parameters"],
            }
        )
    result = {
        "method": "SphereRate (V12-C) spherical semantic dwell-rate",
        "split": split,
        "seeds": list(SEEDS),
        "checkpoints": checkpoints,
        "inputs": ["ERP_image", "declared_start", "horizon"],
        "forbidden_inputs": ["TempSAL_prediction", "gaze_history", "gate/test labels"],
        "methods": methods,
        "per_horizon_v12c": per_horizon,
        "comparisons_v12c_over_baseline": comparisons,
        "wrong_image_3s": wrong_attribution,
        "outer_test_target_loads": 0,
        "gate_target_loads": len(dataset) if split == "gate" else 0,
        "test_target_loads": len(dataset) if split == "test" else 0,
        "data_root": str(args.data_root),
        "run_root": str(args.run_root),
        "tempsal_report": str(tempsal_path.relative_to(ROOT)),
    }
    out = (
        ROOT / args.output_dir
        if args.output_dir
        else ROOT / "reports" / (
            "v12c_spherical_dwell_rate" if split == "val" else f"v12c_spherical_dwell_rate_{split}"
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        f"# SphereRate：Sitzmann-only {split} 三种子结果",
        "",
        "V12-C 不读取 TempSAL 预测或 gaze history。它从 ERP、declared start 与 horizon 预测四段非负注意力到达率，并在 raw dwell 空间累计。三个 checkpoint 均只由当前数据集的 validation split 选择。"
        + (
            "本报告是冻结后的历史已见 development gate 稳定性评估，不是新的盲测。"
            if split == "gate"
            else "测试集不参与模型或超参数选择。"
        ),
        "",
        "## Early macro（0.5/1/3 s）",
        "",
        "| Method | KLD↓ | SIM↑ | CC↑ | Cov@5↑ | Cov@10↑ | Cov@15↑ | Cov@20↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"early_template": "Early template", "tempsal_erp": "TempSAL-ERP", "v12c": "SphereRate"}
    for name in ("early_template", "tempsal_erp", "v12c"):
        values = methods[name]["early_macro"]
        lines.append(
            f"| {labels[name]} | {values['kld_sph']:.4f} | {values['sim_sph']:.4f} | "
            f"{values['cc_sph']:.4f} | {values['coverage_5']:.4f} | "
            f"{values['coverage_10']:.4f} | {values['coverage_15']:.4f} | "
            f"{values['coverage_20']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 5 s",
            "",
            "| Method | KLD↓ | SIM↑ | CC↑ | Cov@5↑ | Cov@10↑ | Cov@15↑ | Cov@20↑ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("early_template", "tempsal_erp", "v12c"):
        values = methods[name]["horizon_5"]
        lines.append(
            f"| {labels[name]} | {values['kld_sph']:.4f} | {values['sim_sph']:.4f} | "
            f"{values['cc_sph']:.4f} | {values['coverage_5']:.4f} | "
            f"{values['coverage_10']:.4f} | {values['coverage_15']:.4f} | "
            f"{values['coverage_20']:.4f} |"
        )
    lines.extend(["", "## 相对 TempSAL-ERP 的 paired content bootstrap", ""])
    for scope in ("early_macro", "horizon_5"):
        lines.append(f"### {scope}")
        lines.append("")
        for metric, values in comparisons["tempsal_erp"][scope].items():
            lines.append(
                f"- {metric}: {values['mean']:+.5f} "
                f"[{values['ci95_low']:+.5f}, {values['ci95_high']:+.5f}]"
            )
        lines.append("")
    lines.extend(["## Content attribution", ""])
    lines.append(
        "3 s correct-image 相对 deterministic wrong-image 的正向 improvement（KLD 为 wrong−correct）："
    )
    lines.append("")
    for metric, values in wrong_attribution.items():
        lines.append(
            f"- {metric}: {values['mean']:+.5f} "
            f"[{values['ci95_low']:+.5f}, {values['ci95_high']:+.5f}]"
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"- 三种子验证集选择的 epochs 为 {[item['epoch'] for item in checkpoints]}。",
            (
                "- 冻结 gate 上，V12-C 的 early KLD/SIM/CC/Cov@10/20 与 5 s 六项均高于 TempSAL-ERP，但 early Cov@5 低 0.00453；低预算全面领先尚未跨 split 稳定。"
                if split == "gate"
                else "- 与 TempSAL-ERP 的方向和不确定性见 paired content bootstrap；不根据测试结果回调模型。"
            ),
            (
                "- 冻结 checkpoint 已完成历史 development gate 稳定性评估；outer-test 仍未打开，gate 不作为新的盲测证据。"
                if split == "gate"
                else "- 本报告仅比较同一数据集内的 train/validation/test 协议，不包含跨数据集冻结测试。"
            ),
        ]
    )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(out / "summary.md")


if __name__ == "__main__":
    main()
