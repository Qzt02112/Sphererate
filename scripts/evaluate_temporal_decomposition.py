#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.config import load_config
from sphererate.data.maps import erp_gaussian_map
from sphererate.evaluation.content_interaction import grouped_bootstrap_improvement, log_density_mix
from sphererate.evaluation.metrics import cc, kld, mean_scores, saliency_scores, sim
from sphererate.evaluation.temporal_factorization import (
    accumulate_temporal_maps,
    align_to_start,
    area_normalize,
    centered_log_residual,
    deterministic_observer_split,
    restore_from_start,
    select_monotonic_alphas,
    weighted_cosine,
)
from sphererate.utils import ensure_dir, read_jsonl


FINAL_METRICS = ("kld_sph", "sim_sph", "cc_sph", "nss", "auc_judd")


def _load_prediction(root: Path, split: str, image_id: str) -> np.ndarray:
    with np.load(root / split / f"{image_id}.npz") as data:
        return np.asarray(data["final"], dtype=np.float32).squeeze()


def _load_target(row: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(row["target"]) as data:
        return {
            key: np.asarray(data[key], dtype=np.float32).squeeze()
            for key in ("temporal", "cumulative", "final", "fixation", "area_weights")
        }


def _load_fixations(row: dict[str, Any]) -> np.ndarray:
    with np.load(row["fixations"]) as data:
        return np.asarray(data["fixations"], dtype=np.float32)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _provenance_audit(
    rows_by_split: dict[str, list[dict[str, Any]]],
    num_bins: int,
    expected_observers: int,
    expected_samples: int,
    expected_per_bin: int,
    bin_weights: Sequence[float],
) -> dict[str, Any]:
    shape_failures = 0
    observer_failures = 0
    trajectory_failures = 0
    bin_count_failures = 0
    temporal_mass_errors: list[float] = []
    final_klds: list[float] = []
    final_sims: list[float] = []
    opened_splits: set[str] = set()
    for split, rows in rows_by_split.items():
        for row in rows:
            opened_splits.add(split)
            target = _load_target(row)
            temporal = target["temporal"]
            area = target["area_weights"]
            if temporal.shape != (num_bins, area.shape[0], area.shape[1]):
                shape_failures += 1
            temporal_mass_errors.extend(
                [abs(float((current * area).sum()) - 1.0) for current in temporal]
            )
            reconstructed = accumulate_temporal_maps(temporal, bin_weights, area)
            final_klds.append(kld(reconstructed, target["final"], area))
            final_sims.append(sim(reconstructed, target["final"], area))
            fixations = _load_fixations(row)
            observer_ids = np.unique(fixations[:, 3].astype(np.int64))
            if len(observer_ids) != expected_observers:
                observer_failures += 1
            for observer_id in observer_ids:
                trajectory = fixations[fixations[:, 3].astype(np.int64) == observer_id]
                if len(trajectory) != expected_samples:
                    trajectory_failures += 1
                counts = np.bincount(trajectory[:, 5].astype(np.int64), minlength=num_bins)
                if counts.tolist() != [expected_per_bin] * num_bins:
                    bin_count_failures += 1
    return {
        "opened_splits": sorted(opened_splits),
        "test_target_opened": False,
        "row_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "shape_failures": shape_failures,
        "observer_count_failures": observer_failures,
        "trajectory_length_failures": trajectory_failures,
        "per_observer_bin_count_failures": bin_count_failures,
        "max_temporal_area_mass_error": float(max(temporal_mass_errors, default=0.0)),
        "equal_weight_final_reconstruction": {
            "mean_kld_sph": float(np.mean(final_klds)),
            "max_kld_sph": float(np.max(final_klds)),
            "mean_sim_sph": float(np.mean(final_sims)),
            "min_sim_sph": float(np.min(final_sims)),
        },
        "pass": not any((shape_failures, observer_failures, trajectory_failures, bin_count_failures)),
    }


def _build_train_templates(
    rows: list[dict[str, Any]],
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    sums: np.ndarray | None = None
    area: np.ndarray | None = None
    for row in rows:
        target = _load_target(row)
        if sums is None:
            sums = np.zeros_like(target["temporal"], dtype=np.float64)
            area = target["area_weights"]
        sums += align_to_start(target["temporal"], float(row["start_lon_deg"]))
    if sums is None or area is None:
        raise ValueError("No train rows available for temporal templates")
    templates = np.stack(
        [area_normalize(sums[index] / len(rows), area) for index in range(num_bins)],
        axis=0,
    )
    return templates, area


def _bootstrap_grouped_values(
    values_by_image: dict[str, float],
    groups_by_image: dict[str, str],
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    by_group: dict[str, list[float]] = {}
    for image_id, value in values_by_image.items():
        by_group.setdefault(groups_by_image[image_id], []).append(float(value))
    values = np.asarray([np.mean(by_group[key]) for key in sorted(by_group)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draw = rng.integers(0, len(values), len(values))
        samples[index] = float(values[draw].mean())
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "num_groups": int(len(values)),
    }


def _contribution_curves(
    rows: list[dict[str, Any]],
    templates: np.ndarray,
    static_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        target = _load_target(row)
        area = target["area_weights"]
        restored = restore_from_start(templates, float(row["start_lon_deg"]))
        static = _load_prediction(static_root, str(row["split"]), str(row["image_id"]))
        for bin_index in range(templates.shape[0]):
            gt = target["temporal"][bin_index]
            records.append(
                {
                    "image_id": row["image_id"],
                    "base_image_id": row["base_image_id"],
                    "condition": row["condition"],
                    "bin": bin_index,
                    "start_kld_sph": kld(restored[bin_index], gt, area),
                    "start_sim_sph": sim(restored[bin_index], gt, area),
                    "start_cc_sph": cc(restored[bin_index], gt, area),
                    "content_kld_sph": kld(static, gt, area),
                    "content_sim_sph": sim(static, gt, area),
                    "content_cc_sph": cc(static, gt, area),
                }
            )
    summaries: dict[str, Any] = {}
    for bin_index in range(templates.shape[0]):
        current = [record for record in records if int(record["bin"]) == bin_index]
        summaries[str(bin_index)] = {
            key: float(np.mean([float(record[key]) for record in current]))
            for key in (
                "start_kld_sph",
                "start_sim_sph",
                "start_cc_sph",
                "content_kld_sph",
                "content_sim_sph",
                "content_cc_sph",
            )
        }
    by_image: dict[str, dict[int, dict[str, Any]]] = {}
    groups: dict[str, str] = {}
    for record in records:
        image_id = str(record["image_id"])
        by_image.setdefault(image_id, {})[int(record["bin"])] = record
        groups[image_id] = str(record["base_image_id"])
    trend_values = {
        "start_kld_decline": {
            image_id: bins[5]["start_kld_sph"] - bins[0]["start_kld_sph"] for image_id, bins in by_image.items()
        },
        "start_sim_decline": {
            image_id: bins[0]["start_sim_sph"] - bins[5]["start_sim_sph"] for image_id, bins in by_image.items()
        },
        "content_kld_increase": {
            image_id: bins[0]["content_kld_sph"] - bins[5]["content_kld_sph"] for image_id, bins in by_image.items()
        },
        "content_sim_increase": {
            image_id: bins[5]["content_sim_sph"] - bins[0]["content_sim_sph"] for image_id, bins in by_image.items()
        },
    }
    return records, {"per_bin": summaries, "trends": (trend_values, groups)}


def _alpha_costs(
    rows: list[dict[str, Any]],
    templates: np.ndarray,
    static_root: Path,
    alpha_grid: Sequence[float],
    device_name: str,
) -> np.ndarray:
    device = torch.device(device_name if not device_name.startswith("cuda") or torch.cuda.is_available() else "cpu")
    alpha = torch.as_tensor(alpha_grid, dtype=torch.float32, device=device)[:, None, None, None]
    costs = torch.zeros((templates.shape[0], len(alpha_grid)), dtype=torch.float64, device=device)
    eps = 1e-7
    for row in rows:
        target = _load_target(row)
        restored = restore_from_start(templates, float(row["start_lon_deg"]))
        static = _load_prediction(static_root, str(row["split"]), str(row["image_id"]))
        area = torch.from_numpy(target["area_weights"]).to(device=device, dtype=torch.float32)
        gt = torch.from_numpy(target["temporal"]).to(device=device, dtype=torch.float32)
        log_template = torch.log(torch.from_numpy(restored).to(device=device).clamp_min(eps))
        log_static = torch.log(torch.from_numpy(static).to(device=device).clamp_min(eps))[None]
        logits = alpha * log_template[None] + (1.0 - alpha) * log_static[None]
        logits = logits - logits.amax(dim=(-2, -1), keepdim=True)
        pred = torch.exp(logits)
        pred = pred / (pred * area[None, None]).sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
        gt_n = gt / (gt * area[None]).sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
        current = (area[None, None] * gt_n[None] * torch.log((gt_n[None] + eps) / (pred + eps))).sum(dim=(-2, -1))
        costs += current.transpose(0, 1).to(torch.float64)
    return (costs / len(rows)).detach().cpu().numpy()


def _render_half_temporal(
    fixations: np.ndarray,
    observer_ids: Sequence[int],
    num_bins: int,
    area: np.ndarray,
    sigma_px: float,
) -> np.ndarray:
    selected = fixations[np.isin(fixations[:, 3].astype(np.int64), np.asarray(observer_ids, dtype=np.int64))]
    height, width = area.shape
    maps = []
    for bin_index in range(num_bins):
        points = selected[selected[:, 5].astype(np.int64) == bin_index]
        maps.append(
            erp_gaussian_map(
                points[:, 0],
                points[:, 1],
                height,
                width,
                sigma_px=sigma_px,
                weights=points[:, 4],
                area_weights=area,
                area_weighted=True,
            )
        )
    return np.stack(maps, axis=0)


def _fixed_sector(longitude: float) -> int:
    value = (float(longitude) + 180.0) % 360.0 - 180.0
    return 0 if value < -60.0 else 1 if value < 60.0 else 2


def _reliability_analysis(
    rows: list[dict[str, Any]],
    templates: np.ndarray,
    num_bins: int,
    observer_seed: int,
    different_seed: int,
    sigma_px: float,
    iterations: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_target = _load_target(rows[0])
    height, width = sample_target["area_weights"].shape
    left_residuals = np.empty((len(rows), num_bins, height, width), dtype=np.float32)
    right_residuals = np.empty_like(left_residuals)
    areas: list[np.ndarray] = []
    for row_index, row in enumerate(rows):
        target = _load_target(row)
        area = target["area_weights"]
        fixations = _load_fixations(row)
        observers = np.unique(fixations[:, 3].astype(np.int64)).tolist()
        left_ids, right_ids = deterministic_observer_split(observers, str(row["image_id"]), observer_seed)
        left_maps = _render_half_temporal(fixations, left_ids, num_bins, area, sigma_px)
        right_maps = _render_half_temporal(fixations, right_ids, num_bins, area, sigma_px)
        restored = restore_from_start(templates, float(row["start_lon_deg"]))
        for bin_index in range(num_bins):
            left = centered_log_residual(left_maps[bin_index], restored[bin_index], area)
            right = centered_log_residual(right_maps[bin_index], restored[bin_index], area)
            left_residuals[row_index, bin_index] = align_to_start(left, float(row["start_lon_deg"]))
            right_residuals[row_index, bin_index] = align_to_start(right, float(row["start_lon_deg"]))
        areas.append(area)
    donor_indices: list[int] = []
    for index, row in enumerate(rows):
        candidates = [
            candidate
            for candidate, other in enumerate(rows)
            if candidate != index
            and str(other["group_id"]) != str(row["group_id"])
            and _fixed_sector(other["start_lon_deg"]) == _fixed_sector(row["start_lon_deg"])
        ]
        digest = hashlib.sha256(f"{different_seed}:{row['image_id']}".encode("utf-8")).digest()
        donor_indices.append(candidates[int.from_bytes(digest[:8], "big") % len(candidates)])
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for bin_index in range(num_bins):
            same = weighted_cosine(left_residuals[row_index, bin_index], right_residuals[row_index, bin_index], areas[row_index])
            donor = donor_indices[row_index]
            different = weighted_cosine(left_residuals[row_index, bin_index], right_residuals[donor, bin_index], areas[row_index])
            records.append(
                {
                    "image_id": row["image_id"],
                    "base_image_id": row["base_image_id"],
                    "bin": bin_index,
                    "same_image_cosine": same,
                    "different_image_cosine": different,
                    "oriented_reliability": same - different,
                    "donor_image_id": rows[donor]["image_id"],
                }
            )
    groups = {str(row["image_id"]): str(row["base_image_id"]) for row in rows}
    summaries: dict[str, Any] = {}
    for bin_index in range(num_bins):
        current = [record for record in records if int(record["bin"]) == bin_index]
        values = {str(record["image_id"]): float(record["oriented_reliability"]) for record in current}
        summaries[str(bin_index)] = {
            "same_image_cosine": float(np.mean([record["same_image_cosine"] for record in current])),
            "different_image_cosine": float(np.mean([record["different_image_cosine"] for record in current])),
            "oriented_reliability": _bootstrap_grouped_values(values, groups, iterations, bootstrap_seed),
        }
    return records, summaries


def _final_method_scores(
    rows: list[dict[str, Any]],
    templates: np.ndarray,
    static_root: Path,
    start_root: Path,
    wrong_mapping: dict[str, str],
    alphas: Sequence[float],
    constant_alpha: float,
    bin_weights: Sequence[float],
    shuffle_seed: int,
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]], list[float]]:
    rng = np.random.default_rng(shuffle_seed)
    shuffled = list(np.asarray(alphas)[rng.permutation(len(alphas))].astype(float))
    alpha_sets = {
        "monotonic": list(alphas),
        "reversed": list(reversed(alphas)),
        "shuffled": shuffled,
        "temporal_constant": [constant_alpha] * len(alphas),
    }
    methods: dict[str, dict[str, dict[str, float]]] = {name: {} for name in [
        "monotonic", "reversed", "shuffled", "temporal_constant", "wrong_image",
        "start_only", "static", "r0_constant_final", "temporal_start_template",
    ]}
    for row in rows:
        image_id = str(row["image_id"])
        target = _load_target(row)
        area = target["area_weights"]
        restored = restore_from_start(templates, float(row["start_lon_deg"]))
        static = _load_prediction(static_root, "val", image_id)
        start_final = _load_prediction(start_root, "val", image_id)
        for name, current_alphas in alpha_sets.items():
            temporal = np.stack(
                [log_density_mix(restored[index], static, current_alphas[index], area) for index in range(len(alphas))],
                axis=0,
            )
            final = accumulate_temporal_maps(temporal, bin_weights, area)
            methods[name][image_id] = saliency_scores(final, target["final"], target["fixation"], area).as_dict()
        donor_static = _load_prediction(static_root, "val", wrong_mapping[image_id])
        wrong_temporal = np.stack(
            [log_density_mix(restored[index], donor_static, alphas[index], area) for index in range(len(alphas))], axis=0
        )
        methods["wrong_image"][image_id] = saliency_scores(
            accumulate_temporal_maps(wrong_temporal, bin_weights, area), target["final"], target["fixation"], area
        ).as_dict()
        methods["start_only"][image_id] = saliency_scores(
            start_final, target["final"], target["fixation"], area
        ).as_dict()
        methods["static"][image_id] = saliency_scores(static, target["final"], target["fixation"], area).as_dict()
        methods["r0_constant_final"][image_id] = saliency_scores(
            log_density_mix(start_final, static, constant_alpha, area), target["final"], target["fixation"], area
        ).as_dict()
        methods["temporal_start_template"][image_id] = saliency_scores(
            accumulate_temporal_maps(restored, bin_weights, area), target["final"], target["fixation"], area
        ).as_dict()
    summaries = {name: mean_scores(list(scores.values())) for name, scores in methods.items()}
    return methods, summaries, shuffled


def _paired(
    candidate: dict[str, dict[str, float]],
    control: dict[str, dict[str, float]],
    groups: dict[str, str],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return {
        metric: grouped_bootstrap_improvement(candidate, control, groups, metric, iterations, seed)
        for metric in FINAL_METRICS
    }


def _read_wrong_mapping(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["target_image_id"]: row["donor_image_id"] for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate T0 temporal decomposition evaluation")
    parser.add_argument("--config", default="configs/gate_t0_temporal_decomposition.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["network_training"] != "forbidden" or config["test_status"] != "frozen_not_evaluated":
        raise ValueError("T0 must forbid network training and keep test frozen")
    paths = config["paths"]
    temporal_cfg = config["temporal"]
    reliability_cfg = config["reliability"]
    stats_cfg = config["statistics"]
    output_root = ensure_dir(paths["output_root"])
    processed_root = Path(paths["conditioned_root"])
    all_rows = read_jsonl(processed_root / "manifest.jsonl")
    rows_by_split = {
        split: sorted([row for row in all_rows if row["split"] == split], key=lambda row: row["image_id"])
        for split in ("train", "val")
    }
    num_bins = int(temporal_cfg["num_bins"])
    bin_weights = [float(value) for value in temporal_cfg["fixed_bin_weights"]]
    iterations = int(stats_cfg["bootstrap_iterations"])
    bootstrap_seed = int(stats_cfg["bootstrap_seed"])
    provenance = _provenance_audit(
        rows_by_split,
        num_bins,
        int(reliability_cfg["expected_observers"]),
        int(reliability_cfg["expected_samples_per_observer"]),
        int(reliability_cfg["expected_samples_per_bin_per_observer"]),
        bin_weights,
    )
    templates, area = _build_train_templates(rows_by_split["train"], num_bins)
    np.savez_compressed(output_root / "train_start_temporal_templates.npz", templates=templates, area_weights=area)
    static_root = Path(paths["static_predictions_root"])
    start_root = Path(paths["start_only_predictions_root"])
    per_bin_records, curve_data = _contribution_curves(rows_by_split["val"], templates, static_root)
    trend_values, trend_groups = curve_data["trends"]
    trends = {
        name: _bootstrap_grouped_values(values, trend_groups, iterations, bootstrap_seed)
        for name, values in trend_values.items()
    }
    _write_csv(output_root / "per_bin_metrics.csv", per_bin_records)
    alpha_grid = [float(value) for value in temporal_cfg["alpha_grid"]]
    alpha_costs = _alpha_costs(
        rows_by_split["train"], templates, static_root, alpha_grid, str(config["runtime"]["alpha_device"])
    )
    selected_alphas = select_monotonic_alphas(alpha_costs, alpha_grid)
    reliability_records, reliability = _reliability_analysis(
        rows_by_split["val"],
        templates,
        num_bins,
        int(reliability_cfg["observer_split_seed"]),
        int(reliability_cfg["different_image_seed"]),
        float(reliability_cfg["sigma_px"]),
        iterations,
        bootstrap_seed,
    )
    _write_csv(output_root / "split_half_reliability.csv", reliability_records)
    methods, final_summaries, shuffled_alphas = _final_method_scores(
        rows_by_split["val"],
        templates,
        static_root,
        start_root,
        _read_wrong_mapping(Path(paths["wrong_image_mapping"])),
        selected_alphas,
        float(temporal_cfg["constant_alpha"]),
        bin_weights,
        int(temporal_cfg["shuffled_alpha_seed"]),
    )
    groups = {str(row["image_id"]): str(row["base_image_id"]) for row in rows_by_split["val"]}
    comparisons = {
        "monotonic_vs_start_only": _paired(methods["monotonic"], methods["start_only"], groups, iterations, bootstrap_seed),
        "monotonic_vs_temporal_start_template": _paired(
            methods["monotonic"], methods["temporal_start_template"], groups, iterations, bootstrap_seed
        ),
        "monotonic_vs_reversed": _paired(methods["monotonic"], methods["reversed"], groups, iterations, bootstrap_seed),
        "monotonic_vs_shuffled": _paired(methods["monotonic"], methods["shuffled"], groups, iterations, bootstrap_seed),
        "monotonic_vs_wrong_image": _paired(methods["monotonic"], methods["wrong_image"], groups, iterations, bootstrap_seed),
        "monotonic_vs_r0_constant": _paired(methods["monotonic"], methods["r0_constant_final"], groups, iterations, bootstrap_seed),
    }
    start_decline = all(trends[name]["ci95_low"] > 0.0 for name in ("start_kld_decline", "start_sim_decline"))
    content_increase = all(trends[name]["ci95_low"] > 0.0 for name in ("content_kld_increase", "content_sim_increase"))
    late_reliability = reliability[str(num_bins - 1)]["oriented_reliability"]["ci95_low"] > 0.0
    alpha_drop = float(selected_alphas[0] - selected_alphas[-1])
    nonconstant_alpha = alpha_drop >= float(temporal_cfg["minimum_alpha_drop"])
    final_vs_start = comparisons["monotonic_vs_start_only"]
    final_primary = final_vs_start["kld_sph"]["ci95_low"] > 0.0 and final_vs_start["sim_sph"]["ci95_low"] > 0.0
    cc_no_regression = final_vs_start["cc_sph"]["ci95_high"] >= 0.0
    order_pass = all(
        comparisons[name][metric]["ci95_low"] > 0.0
        for name in ("monotonic_vs_reversed", "monotonic_vs_shuffled")
        for metric in ("kld_sph", "sim_sph")
    )
    wrong_pass = all(
        comparisons["monotonic_vs_wrong_image"][metric]["ci95_low"] > 0.0
        for metric in ("kld_sph", "sim_sph")
    )
    temporal_template_comparison = comparisons["monotonic_vs_temporal_start_template"]
    beats_temporal_start_template = all(
        temporal_template_comparison[metric]["ci95_low"] > 0.0 for metric in ("kld_sph", "sim_sph")
    )
    decision_components = {
        "provenance_pass": bool(provenance["pass"]),
        "start_explanation_declines": bool(start_decline),
        "content_explanation_increases": bool(content_increase),
        "late_residual_reliability": bool(late_reliability),
        "nonconstant_monotonic_alpha": bool(nonconstant_alpha),
        "final_primary_positive_ci": bool(final_primary),
        "final_cc_no_significant_regression": bool(cc_no_regression),
        "correct_order_beats_reversed_and_shuffled": bool(order_pass),
        "actual_image_beats_wrong_image": bool(wrong_pass),
    }
    gate_pass = all(decision_components.values())
    result = {
        "protocol": config["protocol"],
        "gate": "T0",
        "decision": "PASS" if gate_pass else "FAIL",
        "network_training_performed": False,
        "provenance": provenance,
        "contribution_curves": {"per_bin": curve_data["per_bin"], "trends": trends},
        "alpha_selection": {
            "grid": alpha_grid,
            "mean_train_kld_costs_by_bin_and_alpha": alpha_costs.tolist(),
            "selected_monotonic_alphas": selected_alphas,
            "alpha_drop": alpha_drop,
            "reversed_alphas": list(reversed(selected_alphas)),
            "shuffled_alphas": shuffled_alphas,
            "fixed_bin_weights": bin_weights,
        },
        "split_half_reliability": reliability,
        "final_summaries": final_summaries,
        "paired_improvement": comparisons,
        "decision_components": decision_components,
        "additional_diagnostics": {
            "factorization_beats_temporal_start_template_on_both_primary_metrics": bool(
                beats_temporal_start_template
            )
        },
        "gate_pass": gate_pass,
        "test_status": "frozen_not_evaluated",
    }
    final_rows: list[dict[str, Any]] = []
    for image_id in sorted(methods["monotonic"]):
        row: dict[str, Any] = {"image_id": image_id, "base_image_id": groups[image_id]}
        for method_name, scores_by_image in methods.items():
            for metric, value in scores_by_image[image_id].items():
                row[f"{method_name}/{metric}"] = value
        final_rows.append(row)
    _write_csv(output_root / "final_per_image.csv", final_rows)
    (output_root / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
