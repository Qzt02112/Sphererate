#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.aggregate_v12c_dwell_rate import _model_from_checkpoint
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.tile_prefetch import gaze_mass_coverage, rank_tiles
from sphererate.losses import weighted_cc, weighted_kld, weighted_sim
from sphererate.models.registry import build_model


SEEDS = (20260710, 20260711, 20260712)
HORIZONS = (0.5, 1.0, 3.0, 5.0)
EARLY = {0.5, 1.0, 3.0}
BUDGETS = (5.0, 10.0, 15.0, 20.0)
MAP_METRICS = ("kld_sph", "sim_sph", "cc_sph")
METRICS = MAP_METRICS + tuple(f"coverage_{value:g}" for value in BUDGETS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(
    prediction: np.ndarray,
    target: torch.Tensor,
    area: torch.Tensor,
    fixations: np.ndarray,
    observer_ids: list[int],
    horizon: float,
) -> dict[str, float]:
    pred = torch.from_numpy(np.asarray(prediction, dtype=np.float32)).view(1, 1, *prediction.shape)
    gt = target.view(1, 1, *target.shape)
    weights = area.view(1, 1, *area.shape)
    result = {
        "kld_sph": float(weighted_kld(pred, gt, weights)),
        "sim_sph": float(weighted_sim(pred, gt, weights)),
        "cc_sph": float(weighted_cc(pred, gt, weights)),
    }
    area_np = area.numpy()
    for budget in BUDGETS:
        selected = rank_tiles(
            prediction,
            area_np,
            tile_rows=8,
            tile_cols=16,
            budget_percent=budget,
        )
        result[f"coverage_{budget:g}"] = gaze_mass_coverage(
            fixations,
            observer_ids=observer_ids,
            horizon_s=horizon,
            selected_tile_ids=selected,
            tile_rows=8,
            tile_cols=16,
        )
    return result


def _aggregate(rows: list[dict[str, Any]], horizons: set[float]) -> dict[str, float]:
    selected = [row for row in rows if float(row["horizon_s"]) in horizons]
    return {metric: float(np.mean([float(row[metric]) for row in selected])) for metric in METRICS}


def _seed_summary(rows: list[dict[str, Any]], horizons: set[float]) -> dict[str, Any]:
    by_seed = {
        str(seed): _aggregate([row for row in rows if int(row["seed"]) == seed], horizons)
        for seed in SEEDS
    }
    return {
        "per_seed": by_seed,
        "mean": {metric: float(np.mean([by_seed[str(seed)][metric] for seed in SEEDS])) for metric in METRICS},
        "std": {metric: float(np.std([by_seed[str(seed)][metric] for seed in SEEDS], ddof=1)) for metric in METRICS},
    }


def _paired_bootstrap(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    metric: str,
    horizons: set[float],
    *,
    iterations: int = 10000,
) -> dict[str, float | int]:
    def averaged(rows: list[dict[str, Any]]) -> dict[tuple[str, float], float]:
        buckets: dict[tuple[str, float], list[float]] = {}
        for row in rows:
            if float(row["horizon_s"]) not in horizons:
                continue
            key = (str(row["content_id"]), float(row["horizon_s"]))
            buckets.setdefault(key, []).append(float(row[metric]))
        return {key: float(np.mean(values)) for key, values in buckets.items()}

    left, right = averaged(candidate), averaged(baseline)
    contents = sorted({key[0] for key in left})
    differences = []
    for content in contents:
        values = []
        for horizon in sorted(horizons):
            key = (content, horizon)
            improvement = right[key] - left[key] if metric == "kld_sph" else left[key] - right[key]
            values.append(improvement)
        differences.append(float(np.mean(values)))
    values_np = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(20260805)
    bootstrap = values_np[rng.integers(0, len(values_np), size=(iterations, len(values_np)))].mean(axis=1)
    return {
        "mean_improvement": float(values_np.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "content_count": len(contents),
    }


@torch.no_grad()
def _evaluate(
    dataset: EarlyHorizonDataset,
    runs_root: Path,
    device: torch.device,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    v12_models: dict[int, torch.nn.Module] = {}
    tempsal_models: dict[int, torch.nn.Module] = {}
    provenance: dict[str, list[dict[str, Any]]] = {"v12c": [], "tempsal_erp": []}
    for seed in SEEDS:
        v12_path = runs_root / f"v12c_hybrid_dwell_rate_seed{seed}" / "checkpoints" / "best.pt"
        v12_state = torch.load(v12_path, map_location="cpu", weights_only=False)
        v12_models[seed] = _model_from_checkpoint(v12_state, device)
        provenance["v12c"].append(
            {
                "seed": seed,
                "epoch": int(v12_state["epoch"]),
                "checkpoint": str(v12_path),
                "sha256": _sha256(v12_path),
                "parameters": v12_state["parameters"],
                "selection": v12_state["selection"],
                "val": v12_state["val"],
            }
        )
        temp_path = runs_root / f"tempsal_erp_seed{seed}" / "checkpoints" / "best.pt"
        temp_state = torch.load(temp_path, map_location="cpu", weights_only=False)
        temp_model = build_model("semantic_tempsal_temporal", temp_state["config"]).to(device)
        temp_model.load_state_dict(temp_state["model"])
        tempsal_models[seed] = temp_model.eval()
        provenance["tempsal_erp"].append(
            {
                "seed": seed,
                "epoch": int(temp_state["epoch"]),
                "checkpoint": str(temp_path),
                "sha256": _sha256(temp_path),
                "parameters": temp_state["parameters"],
                "val": temp_state["val"],
            }
        )

    rows: dict[str, list[dict[str, Any]]] = {"v12c": [], "tempsal_erp": [], "early_template": []}
    for item in dataset:
        image = item["image"].unsqueeze(0).to(device)
        template_input = item["template"].unsqueeze(0).to(device)
        start = item["start_geometry"].unsqueeze(0).to(device)
        predictions = {
            "v12c": {
                seed: v12_models[seed](image, template_input, start_geometry=start)["cumulative"][0]
                .float()
                .cpu()
                .numpy()
                for seed in SEEDS
            },
            "tempsal_erp": {
                seed: tempsal_models[seed](image)["cumulative"][0].float().cpu().numpy()
                for seed in SEEDS
            },
        }
        with np.load(str(item["fixations_path"])) as fixation_file:
            fixations = np.asarray(fixation_file["fixations"], dtype=np.float32)
        observers = np.unique(fixations[:, 3]).astype(int).tolist()
        common = {
            "image_id": str(item["image_id"]),
            "content_id": str(item["canonical_content_id"]),
        }
        for index, horizon in enumerate(HORIZONS):
            for method in ("v12c", "tempsal_erp"):
                for seed in SEEDS:
                    rows[method].append(
                        {
                            **common,
                            "seed": seed,
                            "horizon_s": horizon,
                            **_metrics(
                                predictions[method][seed][index],
                                item["target"][index],
                                item["area_weights"][0],
                                fixations,
                                observers,
                                horizon,
                            ),
                        }
                    )
            rows["early_template"].append(
                {
                    **common,
                    "seed": -1,
                    "horizon_s": horizon,
                    **_metrics(
                        item["template"][index].numpy(),
                        item["target"][index],
                        item["area_weights"][0],
                        fixations,
                        observers,
                        horizon,
                    ),
                }
            )
    return rows, provenance


def _write_summary(result: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Standard-Split Primary Benchmark — {result['split']}",
        "",
        "All trainable models fit 59 contents and use the same 13-content validation set for checkpoint selection. The unchanged 13-content test set is used only for reporting.",
        "",
    ]
    for scope in ("early_macro", "horizon_5"):
        title = "Early macro (0.5/1/3 s)" if scope == "early_macro" else "5 s"
        lines.extend(
            [
                f"## {title}",
                "",
                "| Method | KLD↓ | SIM↑ | CC↑ | Cov@5↑ | Cov@10↑ | Cov@15↑ | Cov@20↑ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, label in (("early_template", "Early Template"), ("tempsal_erp", "TempSAL-ERP"), ("v12c", "V12-C")):
            values = result["methods"][method][scope]
            if "mean" in values:
                mean, std = values["mean"], values["std"]
                formatted = {metric: f"{mean[metric]:.4f}±{std[metric]:.4f}" for metric in METRICS}
            else:
                formatted = {metric: f"{values[metric]:.4f}" for metric in METRICS}
            lines.append(
                f"| {label} | {formatted['kld_sph']} | {formatted['sim_sph']} | {formatted['cc_sph']} | "
                f"{formatted['coverage_5']} | {formatted['coverage_10']} | {formatted['coverage_15']} | "
                f"{formatted['coverage_20']} |"
            )
        lines.append("")
    lines.extend(["## V12-C improvement over TempSAL-ERP", ""])
    for scope in ("early_macro", "horizon_5"):
        lines.append(f"### {scope}")
        lines.append("")
        for metric, values in result["comparisons"]["tempsal_erp"][scope].items():
            lines.append(
                f"- {metric}: {values['mean_improvement']:+.5f} "
                f"[{values['ci95_low']:+.5f}, {values['ci95_high']:+.5f}]"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate V12-C, TempSAL-ERP, and Early Template on the standard split.")
    parser.add_argument("--data-root", default="data/processed/early_horizon_salient360_spherical2deg_standard_v3")
    parser.add_argument("--runs-root", default="runs/standard_split_v3")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    dataset = EarlyHorizonDataset(data_root, split=args.split)
    rows, provenance = _evaluate(dataset, Path(args.runs_root), torch.device(args.device))
    methods: dict[str, Any] = {}
    for method in ("v12c", "tempsal_erp"):
        methods[method] = {
            "early_macro": _seed_summary(rows[method], EARLY),
            "horizon_5": _seed_summary(rows[method], {5.0}),
            "per_horizon": {f"{h:g}": _seed_summary(rows[method], {h}) for h in HORIZONS},
        }
    methods["early_template"] = {
        "early_macro": _aggregate(rows["early_template"], EARLY),
        "horizon_5": _aggregate(rows["early_template"], {5.0}),
        "per_horizon": {f"{h:g}": _aggregate(rows["early_template"], {h}) for h in HORIZONS},
    }
    comparisons = {
        baseline: {
            scope: {
                metric: _paired_bootstrap(rows["v12c"], rows[baseline], metric, horizons)
                for metric in METRICS
            }
            for scope, horizons in (("early_macro", EARLY), ("horizon_5", {5.0}))
        }
        for baseline in ("tempsal_erp", "early_template")
    }
    result = {
        "protocol": "salient360_standard_train59_val13_test13",
        "split": args.split,
        "fit_split": "train",
        "selection_split": "val",
        "report_split": "test",
        "test_used_for_selection": False,
        "seeds": list(SEEDS),
        "methods": methods,
        "comparisons": comparisons,
        "provenance": provenance,
        "data_manifest": {"path": str(data_root / 'manifest.jsonl'), "sha256": _sha256(data_root / 'manifest.jsonl')},
    }
    output = Path(args.output_dir or f"reports/standard_split_primary_{args.split}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fieldnames = ["method", "image_id", "content_id", "seed", "horizon_s", *METRICS]
    with (output / "per_content.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, method_rows in rows.items():
            writer.writerows({"method": method, **row} for row in method_rows)
    _write_summary(result, output / "summary.md")
    print(json.dumps({"report": str(output / 'summary.md')}, indent=2))


if __name__ == "__main__":
    main()
