#!/usr/bin/env python
"""Frozen Salient360-trained V12-C evaluation on official Sitzmann VR.

The evaluator uses the already materialized coordinate-only cross view.  No
Sitzmann train/validation target, template, or statistic is used by V12-C.
Published-test scenes are historically seen development evidence, not a fresh
blind test.
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
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.data.early_dataset import EarlyHorizonDataset
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


MAP_METRICS = ("kld_sph", "sim_sph", "cc_sph")


def _scene_average(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _seed_average(rows_by_seed: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for rows in rows_by_seed:
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


def _merge_tempsal(path: Path) -> list[dict[str, Any]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for score in result["score_rows"]:
        merged = dict(score)
        for budget in (5.0, 10.0, 20.0):
            match = next(
                row
                for row in result["coverage_rows"][str(budget)]
                if row["content_id"] == score["content_id"]
                and float(row["horizon_s"]) == float(score["horizon_s"])
            )
            merged[f"coverage_{budget:g}"] = float(match["coverage"])
        rows.append(merged)
    return rows


def _format(values: dict[str, float]) -> str:
    return " / ".join(f"{values[name]:.4f}" for name in METRICS)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/processed/cross_view_sitzmann_at_salient360_start_v1",
    )
    parser.add_argument(
        "--tempsal-results",
        default="reports/cross_tempsal_salient360_to_sitzmann_gate/results.json",
    )
    parser.add_argument(
        "--output-dir", default="reports/v12c_frozen_transfer_sitzmann"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_root = ROOT / args.data_root
    provenance = json.loads(
        (data_root / "cross_view_provenance.json").read_text(encoding="utf-8")
    )
    dataset = EarlyHorizonDataset(data_root, split="gate", target_mode="correct")
    items = [dataset[index] for index in range(len(dataset))]
    if len(items) != 12 or len({str(item["canonical_content_id"]) for item in items}) != 3:
        raise ValueError("Expected 12 Sitzmann conditions from three published-test scenes")
    if provenance["model_source"] != "salient360" or int(provenance["shift_px"]) != 128:
        raise ValueError("Sitzmann cross view is not the frozen Salient360-source view")

    device = torch.device(args.device)
    models = []
    checkpoints = []
    for seed in SEEDS:
        path = ROOT / "runs" / f"v12c_hybrid_dwell_rate_seed{seed}" / "checkpoints" / "best.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        models.append(_model_from_checkpoint(state, device))
        checkpoints.append(
            {
                "seed": int(seed),
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "epoch": int(state["epoch"]),
            }
        )

    by_scene_view = {
        (str(item["canonical_content_id"]), str(item["image_id"]).rsplit("_vp", 1)[-1]): item
        for item in items
    }
    scenes = sorted({key[0] for key in by_scene_view})
    wrong_scene = {scene: scenes[(index + 1) % len(scenes)] for index, scene in enumerate(scenes)}
    per_seed_rows: list[list[dict[str, Any]]] = []
    per_seed_wrong: list[list[dict[str, Any]]] = []
    template_conditions: list[dict[str, Any]] = []

    for model in models:
        seed_conditions: list[dict[str, Any]] = []
        seed_wrong_conditions: list[dict[str, Any]] = []
        for item in items:
            content = str(item["canonical_content_id"])
            view = str(item["image_id"]).rsplit("_vp", 1)[-1]
            prediction = model(
                item["image"].unsqueeze(0).to(device),
                None,
                start_geometry=item["start_geometry"].unsqueeze(0).to(device),
            )["cumulative"][0].float().cpu().numpy()
            wrong_item = by_scene_view[(wrong_scene[content], view)]
            wrong_prediction = model(
                wrong_item["image"].unsqueeze(0).to(device),
                None,
                start_geometry=item["start_geometry"].unsqueeze(0).to(device),
            )["cumulative"][0, 2].float().cpu().numpy()
            with np.load(str(item["fixations_path"])) as data:
                fixations = np.asarray(data["fixations"], dtype=np.float32)
            observers = np.unique(fixations[:, 3]).astype(int).tolist()
            area = item["area_weights"][0]
            area_np = area.numpy()
            for horizon_index, horizon in enumerate(HORIZONS):
                metrics = _map_metrics(prediction[horizon_index], item["target"][horizon_index], area)
                metrics.update(
                    _coverage_metrics(
                        prediction[horizon_index], area_np, fixations, observers, horizon
                    )
                )
                seed_conditions.append(
                    {"content_id": content, "horizon_s": horizon, **metrics}
                )
                if not template_conditions:
                    # Filled after seed evaluation to avoid duplicating rows.
                    pass
            wrong_metrics = _map_metrics(wrong_prediction, item["target"][2], area)
            seed_wrong_conditions.append(
                {"content_id": content, "horizon_s": 3.0, **wrong_metrics}
            )
        per_seed_rows.append(_scene_average(seed_conditions))
        per_seed_wrong.append(_scene_average(seed_wrong_conditions))

    for item in items:
        content = str(item["canonical_content_id"])
        with np.load(str(item["fixations_path"])) as data:
            fixations = np.asarray(data["fixations"], dtype=np.float32)
        observers = np.unique(fixations[:, 3]).astype(int).tolist()
        area = item["area_weights"][0]
        area_np = area.numpy()
        for horizon_index, horizon in enumerate(HORIZONS):
            template = item["template"][horizon_index].numpy()
            metrics = _map_metrics(template, item["target"][horizon_index], area)
            metrics.update(_coverage_metrics(template, area_np, fixations, observers, horizon))
            template_conditions.append(
                {"content_id": content, "horizon_s": horizon, **metrics}
            )

    candidate_rows = _seed_average(per_seed_rows)
    wrong_rows = _seed_average(per_seed_wrong)
    template_rows = _scene_average(template_conditions)
    tempsal_rows = _merge_tempsal(ROOT / args.tempsal_results)

    methods = {
        "target_train_only_template": {
            "early_macro": _aggregate(template_rows, EARLY),
            "horizon_5": _aggregate(template_rows, {5.0}),
        },
        "tempsal_erp_salient360_frozen": {
            "early_macro": _aggregate(tempsal_rows, EARLY),
            "horizon_5": _aggregate(tempsal_rows, {5.0}),
        },
        "v12c_salient360_frozen": {
            "early_macro": _aggregate(candidate_rows, EARLY),
            "horizon_5": _aggregate(candidate_rows, {5.0}),
        },
    }
    comparisons = {
        name: {
            scope: {
                metric: _bootstrap_difference(candidate_rows, baseline, metric, horizons)
                for metric in METRICS
            }
            for scope, horizons in (("early_macro", EARLY), ("horizon_5", {5.0}))
        }
        for name, baseline in (
            ("target_train_only_template", template_rows),
            ("tempsal_erp_salient360_frozen", tempsal_rows),
        )
    }
    correct_3s = [row for row in candidate_rows if float(row["horizon_s"]) == 3.0]
    attribution = {
        metric: _bootstrap_difference(correct_3s, wrong_rows, metric, {3.0})
        for metric in MAP_METRICS
    }
    manifest_hash = hashlib.sha256((data_root / "manifest.jsonl").read_bytes()).hexdigest()
    result = {
        "protocol": "v12c_frozen_salient360_to_sitzmann_v1",
        "evaluation_role": "historically_seen_official_published_test_scenes",
        "methods": methods,
        "comparisons": comparisons,
        "wrong_image_3s": attribution,
        "per_horizon_v12c": {
            f"{horizon:g}": _aggregate(candidate_rows, {horizon}) for horizon in HORIZONS
        },
        "checkpoints": checkpoints,
        "data_audit": {
            "condition_count": len(items),
            "scene_count": len(scenes),
            "coordinate_shift_px": int(provenance["shift_px"]),
            "model_start_yaw_deg": float(provenance["external_scanpath_start_yaw_deg"]),
            "manifest_sha256_matches_provenance": manifest_hash
            == provenance["output_manifest_sha256"],
            "target_domain_train_or_val_loaded_by_v12c": 0,
            "target_domain_calibration": False,
            "outer_test_loaded": False,
        },
        "candidate_rows": candidate_rows,
        "template_rows": template_rows,
        "tempsal_rows": tempsal_rows,
    }
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# V12-C Frozen Transfer: Salient360 → Sitzmann",
        "",
        "三个 V12-C checkpoint 完全冻结；只做预先定义的 +128 px 坐标对齐。Sitzmann train/val target、template 和统计均未进入 V12-C。三个 published-test scenes 已被历史工作查看，因此这里是 external development evidence，不是新盲测。",
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
        for metric, values in comparisons["tempsal_erp_salient360_frozen"][scope].items():
            lines.append(
                f"- {metric}: {values['mean']:+.5f} "
                f"[{values['ci95_low']:+.5f}, {values['ci95_high']:+.5f}]"
            )
        lines.append("")
    lines.extend(["## 3 s content attribution", ""])
    for metric, values in attribution.items():
        lines.append(
            f"- {metric}: {values['mean']:+.5f} "
            f"[{values['ci95_low']:+.5f}, {values['ci95_high']:+.5f}]"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- conditions/scenes: `{len(items)}/{len(scenes)}`",
            f"- manifest hash matches frozen cross-view provenance: **{result['data_audit']['manifest_sha256_matches_provenance']}**",
            "- target-domain calibration: **False**",
            "- target-domain train/val target loads by V12-C: `0`",
            "- outer-test loads: `0`",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md"), "methods": methods}, indent=2))


if __name__ == "__main__":
    main()
