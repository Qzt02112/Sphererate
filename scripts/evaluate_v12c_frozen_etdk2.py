#!/usr/bin/env python
"""Frozen V12-C evaluation on real multi-start ET-DK2 sessions.

ET-DK2 contains one observer per session, so density scores are explicitly
reported as individual-session diagnostics.  Raw-dwell tile coverage is the
primary external endpoint.  Only the current validation content groups are
read; ET-DK2 test rows remain metadata-only and Salient360 outer-test remains
sealed.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.data.dataset import coord_channels
from sphererate.data.early_dataset import _declared_start_geometry
from sphererate.data.early_horizon import clipped_dwell_weights
from sphererate.data.maps import MapGeometry, spherical_gaussian_maps
from sphererate.evaluation.temporal_factorization import (
    normalize_longitude_deg,
    reference_alignment_pixel_shift,
)
from sphererate.models.registry import build_model
from scripts.aggregate_v12c_dwell_rate import (
    EARLY,
    HORIZONS,
    METRICS,
    SEEDS,
    _aggregate,
    _bootstrap_difference,
    _coverage_metrics,
    _map_metrics,
    _model_from_checkpoint,
)


CANONICAL_START_YAW_DEG = 90.0
MAP_METRICS = ("kld_sph", "sim_sph", "cc_sph")


def _read_rows(
    path: Path, salient360_manifest: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_roles = {
        str(row["image_id"]): str(row["split"])
        for row in (
            json.loads(line)
            for line in salient360_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    selected = []
    test_metadata = 0
    excluded_by_source_role: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") != "etdk2":
                continue
            if row.get("split") == "test":
                test_metadata += 1
            elif row.get("split") == "val":
                source_role = source_roles.get(str(row.get("source_image_id")), "missing")
                # ET-DK2's local split was historically generated independently
                # and contains both Salient360 training and sealed-outer content.
                # Retain only source val/gate images: this is condition/protocol
                # transfer, not a fresh cross-content test.
                if source_role in {"val", "gate"}:
                    selected.append(dict(row, salient360_source_split=source_role))
                else:
                    excluded_by_source_role[source_role] += 1
    return selected, {
        "selected_val_sessions": len(selected),
        "selected_val_contents": len({row["canonical_content_id"] for row in selected}),
        "etdk2_test_metadata_rows": test_metadata,
        "etdk2_test_target_rows_loaded": 0,
        "salient360_outer_test_target_rows_loaded": 0,
        "excluded_etdk2_val_sessions_by_salient360_source_role": dict(
            sorted(excluded_by_source_role.items())
        ),
        "selected_salient360_source_roles": sorted(
            {str(row["salient360_source_split"]) for row in selected}
        ),
    }


def _rgb(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).float()


def _aligned_input(rgb: torch.Tensor, start_yaw_deg: float) -> tuple[torch.Tensor, int]:
    shift = reference_alignment_pixel_shift(
        float(start_yaw_deg), CANONICAL_START_YAW_DEG, int(rgb.shape[-1])
    )
    aligned = torch.roll(rgb, shifts=shift, dims=-1)
    return torch.cat([aligned, coord_channels(rgb.shape[-2], rgb.shape[-1])], dim=0), shift


def _target_maps(fixations: np.ndarray, area: np.ndarray) -> np.ndarray:
    weights = np.stack(
        [clipped_dwell_weights(fixations, horizon) for horizon in HORIZONS], axis=0
    )
    return spherical_gaussian_maps(
        fixations[:, 0],
        fixations[:, 1],
        height=area.shape[0],
        width=area.shape[1],
        sigma_deg=2.0,
        weights=weights,
        area_weights=area,
        distance="chord",
        truncate_sigma=4.0,
    ).astype(np.float32)


def _content_average(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["content_id"]), float(row["horizon_s"]))].append(row)
    return [
        {
            "content_id": content,
            "horizon_s": horizon,
            **{
                metric: float(np.mean([float(value[metric]) for value in current]))
                for metric in METRICS
                if metric in current[0]
            },
        }
        for (content, horizon), current in sorted(grouped.items())
    ]


def _mean_prediction_metrics(
    predictions: list[np.ndarray],
    target: np.ndarray,
    area: np.ndarray,
    fixations: np.ndarray,
    horizon: float,
) -> dict[str, float]:
    observers = np.unique(fixations[:, 3]).astype(int).tolist()
    area_t = torch.from_numpy(area)
    values = []
    for prediction in predictions:
        current = _map_metrics(prediction, torch.from_numpy(target), area_t)
        current.update(_coverage_metrics(prediction, area, fixations, observers, horizon))
        values.append(current)
    return {
        metric: float(np.mean([row[metric] for row in values])) for metric in METRICS
    }


def _format(values: dict[str, float]) -> str:
    return " / ".join(f"{values[name]:.4f}" for name in METRICS)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/processed/etdk2_salient360shared_256x512_k6_session_v1/manifest.jsonl",
    )
    parser.add_argument(
        "--template-root",
        default="data/processed/early_horizon_salient360_spherical2deg_v2/baselines/early_template",
    )
    parser.add_argument(
        "--salient360-manifest",
        default="data/processed/early_horizon_salient360_spherical2deg_v2/manifest.jsonl",
    )
    parser.add_argument("--output-dir", default="reports/v12c_frozen_transfer_etdk2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = ROOT / args.manifest
    rows, access = _read_rows(manifest, ROOT / args.salient360_manifest)
    if not rows or access["selected_val_contents"] < 2:
        raise ValueError(f"Insufficient ET-DK2 validation data: {access}")
    contents = sorted({str(row["canonical_content_id"]) for row in rows})
    source_ids = {
        content: str(next(row["source_image_id"] for row in rows if row["canonical_content_id"] == content))
        for content in contents
    }
    images = {
        content: _rgb(next(row["image"] for row in rows if row["canonical_content_id"] == content))
        for content in contents
    }
    templates = {}
    for content in contents:
        with np.load(ROOT / args.template_root / f"{source_ids[content]}.npz") as data:
            templates[content] = np.asarray(data["cumulative"], dtype=np.float32)
    wrong_content = {
        content: contents[(index + 1) % len(contents)] for index, content in enumerate(contents)
    }

    device = torch.device(args.device)
    v12_models = []
    tempsal_models = []
    checkpoints: dict[str, list[dict[str, Any]]] = {"v12c": [], "tempsal_erp": []}
    for seed in SEEDS:
        v12_path = ROOT / "runs" / f"v12c_hybrid_dwell_rate_seed{seed}" / "checkpoints" / "best.pt"
        v12_state = torch.load(v12_path, map_location="cpu", weights_only=False)
        v12_models.append(_model_from_checkpoint(v12_state, device))
        checkpoints["v12c"].append(
            {
                "seed": int(seed),
                "path": str(v12_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(v12_path.read_bytes()).hexdigest(),
                "epoch": int(v12_state["epoch"]),
            }
        )
        tempsal_path = ROOT / "runs" / f"kernel_v2_tempsal_erp_seed{seed}" / "checkpoints" / "best.pt"
        tempsal_state = torch.load(tempsal_path, map_location="cpu", weights_only=False)
        tempsal = build_model("semantic_tempsal_temporal", tempsal_state["config"]).to(device)
        tempsal.load_state_dict(tempsal_state["model"])
        tempsal.eval()
        tempsal_models.append(tempsal)
        checkpoints["tempsal_erp"].append(
            {
                "seed": int(seed),
                "path": str(tempsal_path.relative_to(ROOT)),
                "sha256": hashlib.sha256(tempsal_path.read_bytes()).hexdigest(),
                "epoch": int(tempsal_state["epoch"]),
            }
        )

    geometry = MapGeometry(256, 512)
    area = geometry.area_weights.astype(np.float32)
    canonical_start = torch.from_numpy(
        _declared_start_geometry(
            {"model_declared_start_yaw_deg": CANONICAL_START_YAW_DEG}, 256, 512
        )
    ).unsqueeze(0).to(device)
    sessions_by_condition: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sessions_by_condition[
            (str(row["canonical_content_id"]), float(row["start_erp_longitude_deg"]))
        ].append(row)

    session_rows: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
            "v12c",
            "tempsal_erp",
            "source_train_only_template",
            "wrong_start",
            "wrong_image",
        )
    }
    for condition_index, ((content, start), current_sessions) in enumerate(
        sorted(sessions_by_condition.items())
    ):
        aligned, shift = _aligned_input(images[content], start)
        aligned = aligned.unsqueeze(0).to(device)
        v12_predictions = [
            torch.roll(
                model(aligned, None, start_geometry=canonical_start)["cumulative"][0],
                shifts=-shift,
                dims=-1,
            ).float().cpu().numpy()
            for model in v12_models
        ]
        tempsal_predictions = [
            torch.roll(model(aligned)["cumulative"][0], shifts=-shift, dims=-1)
            .float()
            .cpu()
            .numpy()
            for model in tempsal_models
        ]

        wrong_yaw = normalize_longitude_deg(start + 180.0)
        wrong_aligned, wrong_shift = _aligned_input(images[content], wrong_yaw)
        wrong_aligned = wrong_aligned.unsqueeze(0).to(device)
        wrong_start_predictions = [
            torch.roll(
                model(wrong_aligned, None, start_geometry=canonical_start)["cumulative"][0],
                shifts=-wrong_shift,
                dims=-1,
            ).float().cpu().numpy()
            for model in v12_models
        ]

        wrong_image_aligned, wrong_image_shift = _aligned_input(
            images[wrong_content[content]], start
        )
        wrong_image_aligned = wrong_image_aligned.unsqueeze(0).to(device)
        wrong_image_predictions = [
            torch.roll(
                model(wrong_image_aligned, None, start_geometry=canonical_start)["cumulative"][0],
                shifts=-wrong_image_shift,
                dims=-1,
            ).float().cpu().numpy()
            for model in v12_models
        ]
        template_prediction = np.roll(templates[content], shift=-shift, axis=-1)

        for row in current_sessions:
            with np.load(ROOT / row["fixations"]) as data:
                fixations = np.asarray(data["fixations"], dtype=np.float32)
            targets = _target_maps(fixations, area)
            predictions = {
                "v12c": v12_predictions,
                "tempsal_erp": tempsal_predictions,
                "source_train_only_template": [template_prediction],
                "wrong_start": wrong_start_predictions,
                "wrong_image": wrong_image_predictions,
            }
            for horizon_index, horizon in enumerate(HORIZONS):
                for method, maps in predictions.items():
                    metrics = _mean_prediction_metrics(
                        [value[horizon_index] for value in maps],
                        targets[horizon_index],
                        area,
                        fixations,
                        horizon,
                    )
                    session_rows[method].append(
                        {
                            "content_id": content,
                            "session_id": str(row["image_id"]),
                            "horizon_s": horizon,
                            **metrics,
                        }
                    )
        print(
            f"condition {condition_index + 1}/{len(sessions_by_condition)}: "
            f"{content} start={start:g}",
            flush=True,
        )

    content_rows = {name: _content_average(values) for name, values in session_rows.items()}
    methods = {
        name: {
            "early_macro": _aggregate(values, EARLY),
            "horizon_5": _aggregate(values, {5.0}),
        }
        for name, values in content_rows.items()
        if name in {"v12c", "tempsal_erp", "source_train_only_template"}
    }
    comparisons = {
        baseline_name: {
            scope: {
                metric: _bootstrap_difference(
                    content_rows["v12c"], baseline_rows, metric, horizons
                )
                for metric in METRICS
            }
            for scope, horizons in (("early_macro", EARLY), ("horizon_5", {5.0}))
        }
        for baseline_name, baseline_rows in (
            ("tempsal_erp", content_rows["tempsal_erp"]),
            ("source_train_only_template", content_rows["source_train_only_template"]),
        )
    }
    attribution = {
        control: {
            metric: _bootstrap_difference(
                [row for row in content_rows["v12c"] if row["horizon_s"] == 3.0],
                [row for row in content_rows[control] if row["horizon_s"] == 3.0],
                metric,
                {3.0},
            )
            for metric in MAP_METRICS
        }
        for control in ("wrong_start", "wrong_image")
    }
    result = {
        "protocol": "v12c_frozen_salient360_to_etdk2_multistart_v1",
        "evaluation_role": "external_start_and_collection_protocol_transfer_on_source_val_gate_contents",
        "density_target_note": "one observer per session; spherical 2-degree targets rebuilt from clipped raw dwell only for evaluation",
        "primary_external_endpoint": "raw_dwell_tile_coverage",
        "methods": methods,
        "comparisons": comparisons,
        "attribution_3s": attribution,
        "per_horizon_v12c": {
            f"{horizon:g}": _aggregate(content_rows["v12c"], {horizon})
            for horizon in HORIZONS
        },
        "checkpoints": checkpoints,
        "access_audit": access,
        "data_audit": {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "condition_count": len(sessions_by_condition),
            "start_longitudes_deg": sorted({float(key[1]) for key in sessions_by_condition}),
            "target_domain_calibration": False,
            "checkpoint_selection_after_external_evaluation": False,
            "cross_content_generalization_claim": False,
        },
        "content_rows": content_rows,
    }
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# V12-C Frozen Transfer: Salient360 → ET-DK2",
        "",
        "三个 V12-C checkpoint 与三个 TempSAL checkpoint 均按 Salient360 val 预先选择并完全冻结。ET-DK2 只做坐标对齐和评测，不校准任何模型参数。当前 ET-DK2 local split 与 Salient360 主 split 不一致，因此已排除 Salient360 train 与 sealed outer contents；这里只检验跨起点/跨采集协议，不主张独立跨内容泛化。ET-DK2 每个 session 只有一名 observer，因此 density 是 individual-session diagnostic；raw-dwell Coverage 是主要外部指标。",
        "",
        f"评测 `{access['selected_val_sessions']}` sessions、`{access['selected_val_contents']}` contents、`{len(sessions_by_condition)}` content-start conditions；ET-DK2 test target loads=`0`。",
        "",
        "指标顺序：`KLD↓ / SIM↑ / CC↑ / Cov@5↑ / Cov@10↑ / Cov@20↑`。",
        "",
        "| Method | Early macro | 5 s |",
        "|---|---|---|",
    ]
    for name, values in methods.items():
        lines.append(
            f"| {name} | {_format(values['early_macro'])} | {_format(values['horizon_5'])} |"
        )
    lines.extend(["", "## V12-C − TempSAL（正值表示 V12-C 更好）", ""])
    for scope in ("early_macro", "horizon_5"):
        lines.append(f"### {scope}")
        lines.append("")
        for metric, values in comparisons["tempsal_erp"][scope].items():
            lines.append(
                f"- {metric}: {values['mean']:+.5f} "
                f"[{values['ci95_low']:+.5f}, {values['ci95_high']:+.5f}]"
            )
        lines.append("")
    lines.extend(["## 3 s attribution", ""])
    for control, values_by_metric in attribution.items():
        lines.append(f"### {control}")
        lines.append("")
        for metric, values in values_by_metric.items():
            lines.append(
                f"- {metric}: {values['mean']:+.5f} "
                f"[{values['ci95_low']:+.5f}, {values['ci95_high']:+.5f}]"
            )
        lines.append("")
    lines.extend(
        [
            "## Integrity",
            "",
            "- target-domain calibration: **False**",
            "- checkpoint reselection: **False**",
            f"- excluded ET-DK2 val sessions by source role: `{access['excluded_etdk2_val_sessions_by_salient360_source_role']}`",
            f"- ET-DK2 test target loads: `{access['etdk2_test_target_rows_loaded']}`",
            f"- Salient360 outer-test target loads: `{access['salient360_outer_test_target_rows_loaded']}`",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md"), "methods": methods}, indent=2))


if __name__ == "__main__":
    main()
