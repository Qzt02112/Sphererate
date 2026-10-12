#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.early_horizon import clipped_dwell_weights, load_salient360_eye_fixations
from sphererate.data.maps import MapGeometry, erp_gaussian_map, spherical_gaussian_maps
from sphererate.evaluation.metrics import cc, kld, sim
from sphererate.evaluation.temporal_factorization import deterministic_observer_split
from sphererate.utils import ensure_dir, normalize_np


HORIZONS_S = (0.5, 1.0, 3.0, 5.0)
CANDIDATE_RENDERER = "sphere_chord_2deg"
LEGACY_RENDERER = "legacy_erp_7.03deg_equiv"

# Frozen before looking at the audit results.  The first two checks determine
# whether the stored targets changed materially; the third prevents adopting a
# kernel that destroys split-half observer agreement at early horizons.
MATERIAL_SIM_THRESHOLD = 0.90
MATERIAL_KLD_THRESHOLD = 0.05
MIN_EARLY_SPLIT_HALF_SIM = 0.20


def _read_train_val_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counts[str(row["split"])] += 1
            if str(row["split"]) in {"train", "val"}:
                rows.append(row)
    if any(str(row["split"]) not in {"train", "val"} for row in rows):
        raise RuntimeError("Non-train/val row entered the kernel audit")
    return rows, dict(counts)


def _weight_matrix(fixations: np.ndarray, horizons: Sequence[float], *, include_final: bool) -> np.ndarray:
    rows = [clipped_dwell_weights(fixations, horizon) for horizon in horizons]
    if include_final:
        rows.append(np.maximum(fixations[:, 4], 0.0).astype(np.float32))
    return np.stack(rows, axis=0).astype(np.float32)


def _render_legacy(
    fixations: np.ndarray,
    weights: np.ndarray,
    *,
    height: int,
    width: int,
    sigma_px: float,
    area: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            erp_gaussian_map(
                fixations[:, 0],
                fixations[:, 1],
                height,
                width,
                sigma_px=sigma_px,
                weights=current,
                area_weights=area,
                area_weighted=True,
            )
            for current in weights
        ],
        axis=0,
    ).astype(np.float32)


def _render_sphere(
    fixations: np.ndarray,
    weights: np.ndarray,
    *,
    height: int,
    width: int,
    sigma_deg: float,
    area: np.ndarray,
) -> np.ndarray:
    return spherical_gaussian_maps(
        fixations[:, 0],
        fixations[:, 1],
        height,
        width,
        sigma_deg=sigma_deg,
        weights=weights,
        area_weights=area,
        distance="chord",
        truncate_sigma=4.0,
    ).astype(np.float32)


def _metric_row(prediction: np.ndarray, target: np.ndarray, area: np.ndarray) -> dict[str, float]:
    return {
        "sim_sph": sim(prediction, target, area),
        "kld_sph": kld(prediction, target, area),
        "cc_sph": cc(prediction, target, area),
    }


def _effective_area(density: np.ndarray, area: np.ndarray) -> float:
    return float(1.0 / np.maximum((density.astype(np.float64) ** 2 * area).sum(), 1e-12))


def _mean_by_horizon(rows: list[dict[str, Any]], fields: Sequence[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for horizon in HORIZONS_S:
        current = [row for row in rows if float(row["horizon_s"]) == float(horizon)]
        result[f"{horizon:g}"] = {field: float(np.mean([float(row[field]) for row in current])) for field in fields}
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(result: dict[str, Any], path: Path) -> None:
    pair = result["pairwise_vs_legacy"]
    template = result["train_only_template"]
    reliability = result["split_half_reliability"]
    lines = [
        "# Saliency-kernel protocol audit",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "This audit loaded only train/val gaze targets. Gate/test target rows loaded: **0**. No neural network was trained.",
        "",
        "## Renderer provenance",
        "",
        "- legacy: fixed 10 px sigma at 256×512 (7.03125° horizontally), horizontal seam wrap, vertical edge padding",
        "- candidate: isotropic unit-sphere chord Gaussian, sigma 2°, four-sigma truncation, spherical-area normalization",
        "- sensitivity: the same spherical renderer at sigma 1° and 4°",
        "- target semantics remain cumulative clipped dwell density; this intentionally differs from Sitzmann's equal-weight fixation-count map",
        "",
        "## Full-map change relative to legacy",
        "",
        "| Horizon | Renderer | SIM ↑ | KLD ↓ | CC ↑ | legacy/candidate effective-area ratio |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS_S:
        key = f"{horizon:g}"
        for renderer in ("sphere_chord_1deg", "sphere_chord_2deg", "sphere_chord_4deg"):
            row = pair[renderer][key]
            lines.append(
                f"| {key}s | {renderer} | {row['sim_sph']:.4f} | {row['kld_sph']:.4f} | "
                f"{row['cc_sph']:.4f} | {row['legacy_to_renderer_effective_area_ratio']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Content specificity against train-only template",
            "",
            "A larger `1−SIM` means the target retains more content-specific structure instead of collapsing toward the dataset template.",
            "",
            "| Horizon | Legacy 1−SIM | Candidate 1−SIM | Change |",
            "|---:|---:|---:|---:|",
        ]
    )
    for horizon in HORIZONS_S:
        key = f"{horizon:g}"
        legacy = 1.0 - template[LEGACY_RENDERER][key]["sim_sph"]
        candidate = 1.0 - template[CANDIDATE_RENDERER][key]["sim_sph"]
        lines.append(f"| {key}s | {legacy:.4f} | {candidate:.4f} | {candidate - legacy:+.4f} |")
    lines.extend(
        [
            "",
            "## One-split observer reliability sensitivity",
            "",
            "| Horizon | Legacy split-half SIM | Candidate split-half SIM | Legacy temporal ΔSIM | Candidate temporal ΔSIM |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon in HORIZONS_S:
        key = f"{horizon:g}"
        legacy = reliability[LEGACY_RENDERER][key]
        candidate = reliability[CANDIDATE_RENDERER][key]
        lines.append(
            f"| {key}s | {legacy['same_sim_sph']:.4f} | {candidate['same_sim_sph']:.4f} | "
            f"{legacy['temporal_sim_improvement']:+.4f} | {candidate['temporal_sim_improvement']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen decision checks",
            "",
        ]
    )
    lines.extend(f"- {name}: **{'PASS' if value else 'FAIL'}**" for name, value in result["checks"].items())
    lines.extend(
        [
            "",
            "## Consequence",
            "",
            result["consequence"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit legacy ERP smoothing against angular spherical kernels.")
    parser.add_argument("--manifest", default="data/processed/p2_salient360_temporal_aux_seed20260711/manifest.jsonl")
    parser.add_argument("--output-dir", default="reports/saliency_kernel_audit")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    rows, manifest_counts = _read_train_val_manifest(Path(args.manifest))
    if not rows or not any(str(row["split"]) == "train" for row in rows):
        raise ValueError("Kernel audit requires train and val rows")
    height, width = int(args.height), int(args.width)
    area = MapGeometry(height, width).area_weights
    legacy_sigma_px = 10.0 * width / 512.0
    renderers = {
        LEGACY_RENDERER: ("legacy", legacy_sigma_px),
        "sphere_chord_1deg": ("sphere", 1.0),
        CANDIDATE_RENDERER: ("sphere", 2.0),
        "sphere_chord_4deg": ("sphere", 4.0),
    }
    maps: dict[str, dict[str, np.ndarray]] = {name: {} for name in renderers}
    pairwise_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        image_id = str(row["image_id"])
        content_id = str(row["canonical_content_id"])
        fixations = load_salient360_eye_fixations(row["raw_scanpath_r"])
        weights = _weight_matrix(fixations, HORIZONS_S, include_final=False)
        current_maps: dict[str, np.ndarray] = {}
        for name, (family, scale) in renderers.items():
            if family == "legacy":
                rendered = _render_legacy(
                    fixations,
                    weights,
                    height=height,
                    width=width,
                    sigma_px=scale,
                    area=area,
                )
            else:
                rendered = _render_sphere(
                    fixations,
                    weights,
                    height=height,
                    width=width,
                    sigma_deg=scale,
                    area=area,
                )
            maps[name][content_id] = rendered
            current_maps[name] = rendered

        legacy = current_maps[LEGACY_RENDERER]
        for name in ("sphere_chord_1deg", CANDIDATE_RENDERER, "sphere_chord_4deg"):
            for horizon_index, horizon in enumerate(HORIZONS_S):
                metrics = _metric_row(current_maps[name][horizon_index], legacy[horizon_index], area)
                pairwise_rows.append(
                    {
                        "image_id": image_id,
                        "canonical_content_id": content_id,
                        "split": str(row["split"]),
                        "horizon_s": horizon,
                        "renderer": name,
                        **metrics,
                        "legacy_effective_area": _effective_area(legacy[horizon_index], area),
                        "renderer_effective_area": _effective_area(current_maps[name][horizon_index], area),
                        "legacy_to_renderer_effective_area_ratio": _effective_area(legacy[horizon_index], area)
                        / _effective_area(current_maps[name][horizon_index], area),
                    }
                )

        observer_ids = sorted(np.unique(fixations[:, 3]).astype(int).tolist())
        left, right = deterministic_observer_split(observer_ids, key=f"{content_id}:kernel-audit", seed=int(args.seed))
        for name in (LEGACY_RENDERER, CANDIDATE_RENDERER):
            family, scale = renderers[name]
            half_maps: list[np.ndarray] = []
            for selected_ids in (left, right):
                selected = fixations[np.isin(fixations[:, 3].astype(int), np.asarray(selected_ids, dtype=int))]
                half_weights = _weight_matrix(selected, HORIZONS_S, include_final=True)
                if family == "legacy":
                    half_maps.append(
                        _render_legacy(
                            selected,
                            half_weights,
                            height=height,
                            width=width,
                            sigma_px=scale,
                            area=area,
                        )
                    )
                else:
                    half_maps.append(
                        _render_sphere(
                            selected,
                            half_weights,
                            height=height,
                            width=width,
                            sigma_deg=scale,
                            area=area,
                        )
                    )
            for source, target in ((half_maps[0], half_maps[1]), (half_maps[1], half_maps[0])):
                for horizon_index, horizon in enumerate(HORIZONS_S):
                    same = _metric_row(source[horizon_index], target[horizon_index], area)
                    static = _metric_row(source[-1], target[horizon_index], area)
                    reliability_rows.append(
                        {
                            "image_id": image_id,
                            "canonical_content_id": content_id,
                            "split": str(row["split"]),
                            "horizon_s": horizon,
                            "renderer": name,
                            "same_sim_sph": same["sim_sph"],
                            "same_kld_sph": same["kld_sph"],
                            "same_cc_sph": same["cc_sph"],
                            "temporal_sim_improvement": same["sim_sph"] - static["sim_sph"],
                            "temporal_kld_improvement": static["kld_sph"] - same["kld_sph"],
                            "temporal_cc_improvement": same["cc_sph"] - static["cc_sph"],
                        }
                    )
        print(f"{index:02d}/{len(rows)} {image_id}", flush=True)

    template_rows: list[dict[str, Any]] = []
    train_ids = [str(row["canonical_content_id"]) for row in rows if str(row["split"]) == "train"]
    for name in renderers:
        train_total = np.sum([maps[name][content] for content in train_ids], axis=0, dtype=np.float64)
        for row in rows:
            content = str(row["canonical_content_id"])
            target = maps[name][content]
            if str(row["split"]) == "train":
                template_raw = (train_total - target) / float(len(train_ids) - 1)
            else:
                template_raw = train_total / float(len(train_ids))
            template = np.stack([normalize_np(current, weights=area) for current in template_raw], axis=0)
            for horizon_index, horizon in enumerate(HORIZONS_S):
                template_rows.append(
                    {
                        "image_id": str(row["image_id"]),
                        "canonical_content_id": content,
                        "split": str(row["split"]),
                        "horizon_s": horizon,
                        "renderer": name,
                        **_metric_row(template[horizon_index], target[horizon_index], area),
                    }
                )

    pairwise = {
        name: _mean_by_horizon([row for row in pairwise_rows if row["renderer"] == name], (
            "sim_sph",
            "kld_sph",
            "cc_sph",
            "legacy_to_renderer_effective_area_ratio",
        ))
        for name in ("sphere_chord_1deg", CANDIDATE_RENDERER, "sphere_chord_4deg")
    }
    template_summary = {
        name: _mean_by_horizon(
            [row for row in template_rows if row["renderer"] == name],
            ("sim_sph", "kld_sph", "cc_sph"),
        )
        for name in renderers
    }
    reliability_summary = {
        name: _mean_by_horizon(
            [row for row in reliability_rows if row["renderer"] == name],
            (
                "same_sim_sph",
                "same_kld_sph",
                "same_cc_sph",
                "temporal_sim_improvement",
                "temporal_kld_improvement",
                "temporal_cc_improvement",
            ),
        )
        for name in (LEGACY_RENDERER, CANDIDATE_RENDERER)
    }
    candidate_pair_rows = [row for row in pairwise_rows if row["renderer"] == CANDIDATE_RENDERER]
    mean_pair_sim = float(np.mean([row["sim_sph"] for row in candidate_pair_rows]))
    mean_pair_kld = float(np.mean([row["kld_sph"] for row in candidate_pair_rows]))
    early_keys = ("0.5", "1", "3")
    mean_early_reliability = float(
        np.mean([reliability_summary[CANDIDATE_RENDERER][key]["same_sim_sph"] for key in early_keys])
    )
    materially_changed = mean_pair_sim < MATERIAL_SIM_THRESHOLD or mean_pair_kld > MATERIAL_KLD_THRESHOLD
    reliable_enough = mean_early_reliability >= MIN_EARLY_SPLIT_HALF_SIM
    checks = {
        "train_val_only_no_gate_or_test_targets": True,
        "candidate_materially_changes_legacy_targets": materially_changed,
        "candidate_retains_minimum_early_split_half_reliability": reliable_enough,
    }
    decision = "REBUILD_V2" if materially_changed and reliable_enough else "HOLD_FOR_KERNEL_REVIEW"
    consequence = (
        "Freeze v1 as historical-only. Build a new v2 target view with the 2° spherical chord renderer, rebuild all "
        "train-only templates, and retrain the proposed model plus every learned baseline evaluated against v2."
        if decision == "REBUILD_V2"
        else "Do not retrain yet; resolve the failed kernel check without inspecting gate/test targets."
    )
    result = {
        "decision": decision,
        "checks": checks,
        "consequence": consequence,
        "access_audit": {
            "selected_splits": ["train", "val"],
            "selected_rows": len(rows),
            "manifest_split_counts": manifest_counts,
            "gate_target_rows_loaded": 0,
            "test_target_rows_loaded": 0,
        },
        "geometry": {
            "audit_size": [height, width],
            "production_size": [256, 512],
            "legacy_production_sigma_px": 10.0,
            "legacy_sigma_deg_horizontal": 10.0 * 360.0 / 512.0,
            "candidate_sigma_deg": 2.0,
            "candidate_distance": "unit_sphere_chord",
            "candidate_truncate_sigma": 4.0,
        },
        "frozen_thresholds": {
            "material_mean_sim_below": MATERIAL_SIM_THRESHOLD,
            "material_mean_kld_above": MATERIAL_KLD_THRESHOLD,
            "minimum_early_split_half_sim": MIN_EARLY_SPLIT_HALF_SIM,
        },
        "decision_statistics": {
            "candidate_vs_legacy_mean_sim": mean_pair_sim,
            "candidate_vs_legacy_mean_kld": mean_pair_kld,
            "candidate_early_split_half_mean_sim": mean_early_reliability,
        },
        "pairwise_vs_legacy": pairwise,
        "train_only_template": template_summary,
        "split_half_reliability": reliability_summary,
        "no_neural_training": True,
    }
    output_dir = ensure_dir(args.output_dir)
    _write_csv(output_dir / "pairwise_by_content.csv", pairwise_rows)
    _write_csv(output_dir / "template_by_content.csv", template_rows)
    _write_csv(output_dir / "split_half_by_content.csv", reliability_rows)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_summary(result, output_dir / "summary.md")
    print(json.dumps({"decision": decision, "report": str(output_dir / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
