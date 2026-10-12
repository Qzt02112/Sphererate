#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.config import apply_overrides, load_config
from sphererate.evaluation.evaluate import evaluate_prediction_dir
from sphererate.train import _loader, _move_batch, load_model_for_inference
from sphererate.utils import ensure_dir, read_jsonl


METHOD_BY_NAME = {
    "marginal_proposed": "proposed",
    "temporal_aux": "ordinary",
    "random_condition": "random",
}
PRIMARY_METRICS = ("kld_sph", "sim_sph", "cc_sph")


def validate_gate_c_split(split: str) -> str:
    split = str(split).lower()
    if split != "val":
        raise ValueError("Gate C is validation only; the frozen test split cannot be evaluated before method freeze")
    return split


def _read_metric_rows(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            str(row["image_id"]): {
                key: float(value)
                for key, value in row.items()
                if key not in {"dataset", "image_id", "condition"}
            }
            for row in csv.DictReader(handle)
        }


def grouped_bootstrap_improvement(
    candidate: dict[str, dict[str, float]],
    control: dict[str, dict[str, float]],
    group_by_image: dict[str, str],
    metric: str,
    iterations: int = 10000,
    seed: int = 20260710,
) -> dict[str, float | int]:
    if set(candidate) != set(control):
        raise ValueError("Candidate and control per-image rows are not paired")
    oriented_by_group: dict[str, list[float]] = defaultdict(list)
    for image_id in sorted(candidate):
        if image_id not in group_by_image:
            raise KeyError(f"Missing content group for {image_id}")
        if metric.startswith("kld"):
            improvement = control[image_id][metric] - candidate[image_id][metric]
        else:
            improvement = candidate[image_id][metric] - control[image_id][metric]
        oriented_by_group[group_by_image[image_id]].append(float(improvement))
    values = np.asarray([np.mean(items) for items in oriented_by_group.values()], dtype=np.float64)
    if not len(values):
        raise ValueError("No paired groups available for bootstrap")
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        draw = rng.integers(0, len(values), size=len(values))
        samples[index] = values[draw].mean()
    return {
        "oriented_improvement_mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "num_groups": int(len(values)),
    }


def gate_c_seed_decision(summaries: dict[str, dict[str, Any]], added_parameter_fraction: float) -> dict[str, Any]:
    proposed = summaries["proposed"]
    failed_controls: list[str] = []
    details: dict[str, dict[str, bool]] = {}
    for control_name in ("static", "ordinary", "random"):
        control = summaries[control_name]
        checks = {
            "mixed_kld_better": proposed["scores"]["kld_sph"] < control["scores"]["kld_sph"],
            "macro_kld_better": proposed["macro_scores"]["kld_sph"] < control["macro_scores"]["kld_sph"],
            "mixed_sim_no_regression": proposed["scores"]["sim_sph"] >= control["scores"]["sim_sph"],
            "mixed_cc_no_regression": proposed["scores"]["cc_sph"] >= control["scores"]["cc_sph"],
        }
        details[control_name] = checks
        if not all(checks.values()):
            failed_controls.append(control_name)
    parameter_budget_pass = float(added_parameter_fraction) <= 0.10
    return {
        "pass": not failed_controls and parameter_budget_pass,
        "failed_controls": failed_controls,
        "parameter_budget_pass": parameter_budget_pass,
        "added_parameter_fraction": float(added_parameter_fraction),
        "checks": details,
    }


def _method_from_run_name(name: str) -> str | None:
    for fragment, method in METHOD_BY_NAME.items():
        if fragment in name:
            return method
    return None


@torch.no_grad()
def latent_diagnostics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    residual_means: list[float] = []
    pairwise_distances: list[float] = []
    entropies: list[float] = []
    effective_counts: list[float] = []
    max_weights: list[float] = []
    weight_sum: torch.Tensor | None = None
    num_images = 0
    off_max_abs = 0.0
    original_gate = model.residual_gate.detach().clone()

    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(batch["image"])
        residual = outputs["residual_logits"].detach().float()
        maps = outputs["condition_maps"].detach().float()
        weights = outputs["weights"].detach().float()
        area = batch["area_weights"].float()
        batch_size, condition_count = weights.shape
        num_images += batch_size
        residual_means.extend(residual.abs().mean(dim=(1, 2, 3)).cpu().tolist())

        maps_sph = maps / (maps * area).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
        for left in range(condition_count):
            for right in range(left + 1, condition_count):
                pair_sim = (torch.minimum(maps_sph[:, left], maps_sph[:, right]) * area[:, 0]).sum(dim=(-2, -1))
                pairwise_distances.extend((1.0 - pair_sim).cpu().tolist())

        entropy = -(weights.clamp_min(1e-7) * weights.clamp_min(1e-7).log()).sum(dim=-1)
        entropies.extend((entropy / math.log(condition_count)).cpu().tolist())
        effective_counts.extend(torch.exp(entropy).cpu().tolist())
        max_weights.extend(weights.max(dim=-1).values.cpu().tolist())
        current_sum = weights.sum(dim=0).cpu()
        weight_sum = current_sum if weight_sum is None else weight_sum + current_sum

        model.residual_gate.fill_(-100.0)
        disabled = model(batch["image"])
        off_max_abs = max(
            off_max_abs,
            float((disabled["final"] - disabled["teacher"]).abs().max().detach().cpu()),
        )
        model.residual_gate.copy_(original_gate)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    teacher_parameters = sum(parameter.numel() for parameter in model.teacher.parameters())
    stage_codes = model.stage_codes.detach().float().cpu()
    code_distance = torch.pdist(stage_codes).mean().item() if stage_codes.shape[0] > 1 else 0.0
    return {
        "num_images": int(num_images),
        "mean_residual_abs": float(np.mean(residual_means)),
        "residual_gate_value": float(torch.sigmoid(original_gate).cpu()),
        "mean_pairwise_condition_one_minus_sim_sph": float(np.mean(pairwise_distances)),
        "mean_normalized_selector_entropy": float(np.mean(entropies)),
        "mean_effective_anchor_count": float(np.mean(effective_counts)),
        "mean_max_selector_weight": float(np.mean(max_weights)),
        "mean_selector_weights": (weight_sum / float(num_images)).tolist() if weight_sum is not None else [],
        "mean_pairwise_stage_code_l2": float(code_distance),
        "residual_off_max_abs_error": float(off_max_abs),
        "trainable_added_parameters": int(trainable),
        "teacher_parameters": int(teacher_parameters),
        "added_parameter_fraction": float(trainable / teacher_parameters),
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def evaluate_seed(
    seed: int,
    runs: dict[str, dict[str, Any]],
    output_root: Path,
    device_name: str,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    if set(runs) != {"proposed", "ordinary", "random"}:
        raise ValueError(f"Seed {seed} is missing a Gate-C method: {sorted(runs)}")
    seed_root = ensure_dir(output_root / f"seed{seed}")
    summaries: dict[str, dict[str, Any]] = {}
    metric_rows: dict[str, dict[str, dict[str, float]]] = {}
    proposed_model = None
    proposed_cfg = None
    proposed_device = None
    processed_root = None

    for method, run in runs.items():
        checkpoint = Path(run["checkpoint"])
        model, cfg, device = load_model_for_inference(checkpoint, device_name)
        current_root = Path(cfg["paths"]["processed_root"]) / cfg["data"]["processed_name"]
        processed_root = current_root if processed_root is None else processed_root
        if current_root != processed_root:
            raise ValueError("Gate-C runs do not use the same processed dataset")
        if int(cfg["model"]["in_channels"]) != 8 or cfg["train"].get("condition_startpoint_channels", False):
            raise ValueError(f"{run['name']} is not image-only P0 inference")
        pred_dir = checkpoint.parents[1] / "predictions" / "val"
        summaries[method] = evaluate_prediction_dir(
            processed_root,
            pred_dir,
            split="val",
            output_dir=seed_root / method,
        )
        metric_rows[method] = _read_metric_rows(seed_root / method / "val_per_image.csv")
        if method == "proposed":
            proposed_model, proposed_cfg, proposed_device = model, cfg, device

    assert proposed_model is not None and proposed_cfg is not None and proposed_device is not None and processed_root is not None
    static_checkpoint = Path(proposed_cfg["model"]["teacher_checkpoint"])
    static_pred_dir = static_checkpoint.parents[1] / "predictions" / "val"
    summaries["static"] = evaluate_prediction_dir(
        processed_root,
        static_pred_dir,
        split="val",
        output_dir=seed_root / "static",
    )
    metric_rows["static"] = _read_metric_rows(seed_root / "static" / "val_per_image.csv")

    manifest_rows = [row for row in read_jsonl(processed_root / "manifest.jsonl") if row["split"] == "val"]
    group_by_image = {str(row["image_id"]): str(row["group_id"]) for row in manifest_rows}
    paired = {
        control: {
            metric: grouped_bootstrap_improvement(
                metric_rows["proposed"],
                metric_rows[control],
                group_by_image,
                metric,
                iterations=bootstrap_iterations,
                seed=seed,
            )
            for metric in PRIMARY_METRICS
        }
        for control in ("static", "ordinary", "random")
    }
    loader = _loader(proposed_cfg, "val", processed_root, augment=False)
    diagnostics = latent_diagnostics(proposed_model, loader, proposed_device)
    decision = gate_c_seed_decision(summaries, diagnostics["added_parameter_fraction"])
    result = {
        "protocol": "P0_image_only",
        "gate": "C",
        "split": "val",
        "seed": int(seed),
        "num_images": len(manifest_rows),
        "num_groups": len(set(group_by_image.values())),
        "runs": {method: run["name"] for method, run in runs.items()},
        "static_checkpoint": str(static_checkpoint),
        "summaries": summaries,
        "paired_proposed_improvement": paired,
        "latent_diagnostics": diagnostics,
        "decision": decision,
        "test_status": "frozen_not_evaluated",
    }
    (seed_root / "gate_c_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for method in ("proposed", "static", "ordinary", "random"):
        aggregate[method] = {
            scope: {
                metric: _mean_std([float(result["summaries"][method][scope][metric]) for result in results])
                for metric in PRIMARY_METRICS
            }
            for scope in ("scores", "macro_scores")
        }
    return {
        "protocol": "P0_image_only",
        "gate": "C",
        "split": "val",
        "seeds": [result["seed"] for result in results],
        "aggregate": aggregate,
        "per_seed_decisions": {str(result["seed"]): result["decision"] for result in results},
        "pass": len(results) == 3 and all(result["decision"]["pass"] for result in results),
        "test_status": "frozen_not_evaluated",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/multi360_256x512.yaml")
    parser.add_argument("--batch", default="configs/batch_gate_c_marginalization_3seed.yaml")
    parser.add_argument("--output-root", default="reports/gate_c_marginalization")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()
    validate_gate_c_split(args.split)

    base = load_config(args.config)
    batch = load_config(args.batch)
    requested_seeds = set(args.seeds or [])
    runs_by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in batch.get("runs", []):
        method = _method_from_run_name(str(run["name"]))
        if method is None:
            continue
        cfg = apply_overrides(base, run.get("overrides", {}))
        seed = int(cfg["project"]["seed"])
        if requested_seeds and seed not in requested_seeds:
            continue
        checkpoint = ROOT / cfg["paths"]["runs_root"] / str(run["name"]) / "checkpoints" / "best.pt"
        if checkpoint.exists():
            runs_by_seed[seed][method] = {"name": str(run["name"]), "checkpoint": str(checkpoint)}

    output_root = ensure_dir(args.output_root)
    results = [
        evaluate_seed(seed, runs_by_seed[seed], output_root, args.device, int(args.bootstrap_iterations))
        for seed in sorted(runs_by_seed)
        if set(runs_by_seed[seed]) == {"proposed", "ordinary", "random"}
    ]
    if not results:
        raise FileNotFoundError("No complete Gate-C seed triplet with best checkpoints was found")
    aggregate = aggregate_results(results)
    (output_root / "gate_c_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

