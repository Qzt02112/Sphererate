#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.data.maps import MapGeometry
from sphererate.evaluation.content_interaction import grouped_bootstrap_improvement
from sphererate.evaluation.late_saliency import (
    build_session_window_map,
    deterministic_wrong_content_map,
    estimate_start_template,
    fuse_log_density,
    restore_start_template,
)
from sphererate.evaluation.metrics import cc, kld, mean_scores, normalize_density, sim
from sphererate.evaluation.r1_late import (
    content_oracle_metadata,
    decide_r1,
    select_fusion_alpha,
    validate_r1_splits,
    wrong_start_longitude,
)
from sphererate.utils import ensure_dir, read_jsonl


METRICS = ("kld_sph", "sim_sph", "cc_sph")


def _load_npz_array(path: str | Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data[key], dtype=np.float32).squeeze()


def _load_content_maps(rows: list[dict[str, Any]], area: np.ndarray) -> dict[str, np.ndarray]:
    maps: dict[str, np.ndarray] = {}
    for row in rows:
        content = str(row["canonical_content_id"])
        maps[content] = normalize_density(_load_npz_array(row["target"], "final"), area)
    return maps


def _load_targets(
    rows: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    geometry: MapGeometry,
    sigma_px: float,
) -> dict[str, dict[str, np.ndarray]]:
    targets = {str(window["name"]): {} for window in windows}
    for row in tqdm(rows, desc="R1 session targets"):
        sample_id = str(row["image_id"])
        fixations = _load_npz_array(row["fixations"], "fixations")
        for window in windows:
            name = str(window["name"])
            if float(window["start_s"]) == 0.0 and float(window["end_s"]) >= float(row["duration_s"]):
                target = _load_npz_array(row["target"], "final")
            else:
                target = build_session_window_map(
                    fixations,
                    start_s=float(window["start_s"]),
                    end_s=float(window["end_s"]),
                    geometry=geometry,
                    sigma_px=float(sigma_px),
                )
            targets[name][sample_id] = normalize_density(target, geometry.area_weights)
    return targets


def _scores(prediction: np.ndarray, target: np.ndarray, area: np.ndarray) -> dict[str, float]:
    return {
        "kld_sph": kld(prediction, target, area),
        "sim_sph": sim(prediction, target, area),
        "cc_sph": cc(prediction, target, area),
    }


def _method_summary(
    scores: dict[str, dict[str, float]],
    row_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    session_mean = mean_scores(list(scores.values()))
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for sample_id, values in scores.items():
        grouped[str(row_by_id[sample_id]["canonical_content_id"])].append(values)
    per_content = {content: mean_scores(values) for content, values in sorted(grouped.items())}
    return {
        "session_mean": session_mean,
        "content_mean": mean_scores(list(per_content.values())),
        "per_content": per_content,
    }


def _evaluate_window(
    *,
    name: str,
    rows: list[dict[str, Any]],
    targets: dict[str, np.ndarray],
    template: np.ndarray,
    content_maps: dict[str, np.ndarray],
    alpha: float,
    area: np.ndarray,
    wrong_content_seed: int,
) -> tuple[dict[str, dict[str, dict[str, float]]], list[dict[str, Any]], dict[str, Any]]:
    contents = sorted({str(row["canonical_content_id"]) for row in rows})
    wrong_content = deterministic_wrong_content_map(contents, seed=int(wrong_content_seed))
    methods: dict[str, dict[str, dict[str, float]]] = {
        method: {}
        for method in (
            "start_only_actual",
            "start_only_wrong180",
            "content_oracle",
            "fusion_actual",
            "fusion_wrong_image",
            "fusion_wrong_start",
        )
    }
    per_session: list[dict[str, Any]] = []
    groups: dict[str, str] = {}
    row_by_id = {str(row["image_id"]): row for row in rows}

    for row in rows:
        sample_id = str(row["image_id"])
        content = str(row["canonical_content_id"])
        longitude = float(row["start_erp_longitude_deg"])
        wrong_longitude = wrong_start_longitude(longitude)
        target = targets[sample_id]
        start_actual = restore_start_template(template, longitude)
        start_wrong = restore_start_template(template, wrong_longitude)
        content_actual = content_maps[content]
        content_wrong = content_maps[wrong_content[content]]
        predictions = {
            "start_only_actual": start_actual,
            "start_only_wrong180": start_wrong,
            "content_oracle": content_actual,
            "fusion_actual": fuse_log_density(start_actual, content_actual, alpha, area),
            "fusion_wrong_image": fuse_log_density(start_actual, content_wrong, alpha, area),
            "fusion_wrong_start": fuse_log_density(start_wrong, content_actual, alpha, area),
        }
        groups[sample_id] = content
        for method, prediction in predictions.items():
            values = _scores(prediction, target, area)
            methods[method][sample_id] = values
            per_session.append(
                {
                    "window": name,
                    "method": method,
                    "image_id": sample_id,
                    "canonical_content_id": content,
                    "subject_id": row["subject_id"],
                    "start_index": row["start_index"],
                    "start_longitude_declared_deg": row["start_longitude_deg"],
                    "start_erp_longitude_deg": longitude,
                    **values,
                }
            )

    summaries = {method: _method_summary(values, row_by_id) for method, values in methods.items()}
    return methods, per_session, {"summaries": summaries, "groups": groups, "wrong_content": wrong_content}


def _paired(
    candidate: dict[str, dict[str, float]],
    control: dict[str, dict[str, float]],
    groups: dict[str, str],
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    return {
        metric: grouped_bootstrap_improvement(
            candidate,
            control,
            groups,
            metric,
            iterations=int(iterations),
            seed=int(seed),
        )
        for metric in METRICS
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    decision = summary["decision"]
    late = summary["windows"][summary["primary_window"]]
    lines = [
        "# Gate R1: Initial-Start-Conditioned Late Saliency",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        f"Primary target: `{summary['primary_window']}`",
        f"Train-selected alpha: `{summary['selected_alpha']:.2f}`",
        "",
        "> Salient360 maps are a non-deployable content oracle used only to test whether image-specific signal exists.",
        "",
        "## Frozen validation results",
        "",
        "| Method | KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ |",
        "|---|---:|---:|---:|",
    ]
    for method, data in late["summaries"].items():
        scores = data["content_mean"]
        lines.append(
            f"| {method} | {scores['kld_sph']:.6f} | {scores['sim_sph']:.6f} | {scores['cc_sph']:.6f} |"
        )
    lines.extend(["", "## Preregistered comparisons", ""])
    for comparison, metrics in summary["comparisons"].items():
        lines.append(f"### {comparison}")
        lines.append("")
        for metric, stats in metrics.items():
            lines.append(
                f"- {metric}: improvement={stats['oriented_improvement_mean']:.6f}, "
                f"95% CI [{stats['ci95_low']:.6f}, {stats['ci95_high']:.6f}]"
            )
        lines.append("")
    lines.extend(["## Gate checks", ""])
    for key, value in decision["checks"].items():
        lines.append(f"- {key}: **{'PASS' if value else 'FAIL'}**")
    lines.extend(
        [
            "",
            "No network was trained. Alpha and the start template were selected/estimated on train only; val was frozen evaluation; test was not read.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    validate_r1_splits(data_cfg["selection_split"], data_cfg["evaluation_split"])
    view_root = Path(data_cfg["view_root"])
    session_rows = read_jsonl(view_root / data_cfg["session_manifest"])
    content_rows = read_jsonl(view_root / data_cfg["content_manifest"])
    train_rows = [row for row in session_rows if row["split"] == "train"]
    val_rows = [row for row in session_rows if row["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError("R1 requires non-empty train and val sessions")

    h, w = (int(value) for value in data_cfg["image_size"])
    geometry = MapGeometry(h, w)
    area = geometry.area_weights
    windows = [config["targets"]["primary"], config["targets"]["diagnostic"]]
    all_rows = train_rows + val_rows
    targets = _load_targets(all_rows, windows, geometry, float(data_cfg["sigma_px"]))
    content_maps = _load_content_maps(content_rows, area)
    missing = sorted({str(row["canonical_content_id"]) for row in all_rows} - set(content_maps))
    if missing:
        raise KeyError(f"Missing Salient360 content oracle maps: {missing}")

    templates = {
        str(window["name"]): estimate_start_template(
            [
                (targets[str(window["name"])][str(row["image_id"])], float(row["start_erp_longitude_deg"]))
                for row in train_rows
            ],
            area,
        )
        for window in windows
    }
    primary_name = str(config["targets"]["primary"]["name"])
    alpha_samples = [
        (
            targets[primary_name][str(row["image_id"])],
            restore_start_template(templates[primary_name], float(row["start_erp_longitude_deg"])),
            content_maps[str(row["canonical_content_id"])],
        )
        for row in train_rows
    ]
    selected_alpha, alpha_grid_scores = select_fusion_alpha(
        alpha_samples,
        alpha_grid=config["fusion"]["alpha_grid"],
        area_weights=area,
    )

    output_windows: dict[str, Any] = {}
    all_per_session: list[dict[str, Any]] = []
    window_methods: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    window_groups: dict[str, dict[str, str]] = {}
    for window in windows:
        name = str(window["name"])
        methods, per_session, details = _evaluate_window(
            name=name,
            rows=val_rows,
            targets=targets[name],
            template=templates[name],
            content_maps=content_maps,
            alpha=selected_alpha,
            area=area,
            wrong_content_seed=int(config["controls"]["wrong_content_seed"]),
        )
        window_methods[name] = methods
        window_groups[name] = details.pop("groups")
        output_windows[name] = details
        all_per_session.extend(per_session)

    stats_cfg = config["statistics"]
    methods = window_methods[primary_name]
    groups = window_groups[primary_name]
    comparisons = {
        "fusion_vs_start": _paired(
            methods["fusion_actual"], methods["start_only_actual"], groups,
            int(stats_cfg["bootstrap_iterations"]), int(stats_cfg["bootstrap_seed"]),
        ),
        "fusion_vs_wrong_image": _paired(
            methods["fusion_actual"], methods["fusion_wrong_image"], groups,
            int(stats_cfg["bootstrap_iterations"]), int(stats_cfg["bootstrap_seed"]),
        ),
        "start_vs_wrong180": _paired(
            methods["start_only_actual"], methods["start_only_wrong180"], groups,
            int(stats_cfg["bootstrap_iterations"]), int(stats_cfg["bootstrap_seed"]),
        ),
        "fusion_vs_wrong_start": _paired(
            methods["fusion_actual"], methods["fusion_wrong_start"], groups,
            int(stats_cfg["bootstrap_iterations"]), int(stats_cfg["bootstrap_seed"]),
        ),
    }
    decision = decide_r1(selected_alpha=selected_alpha, comparisons=comparisons)
    summary = {
        "protocol": config["protocol"],
        "gate": config["gate"],
        "primary_window": primary_name,
        "selection_split": data_cfg["selection_split"],
        "evaluation_split": data_cfg["evaluation_split"],
        "num_train_sessions": len(train_rows),
        "num_val_sessions": len(val_rows),
        "num_train_contents": len({row["canonical_content_id"] for row in train_rows}),
        "num_val_contents": len({row["canonical_content_id"] for row in val_rows}),
        "content_oracle": content_oracle_metadata(),
        "selected_alpha": selected_alpha,
        "alpha_grid_train_kld_sph": alpha_grid_scores,
        "windows": output_windows,
        "comparisons": comparisons,
        "decision": decision,
        "test_used": False,
        "network_training": False,
    }

    out_root = ensure_dir(config["output"]["root"])
    with (out_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    _write_csv(all_per_session, out_root / "per_session.csv")
    _write_markdown(summary, out_root / "summary.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate R1 start-conditioned late saliency without training.")
    parser.add_argument("--config", default="configs/gate_r1_start_conditioned_late.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    summary = evaluate(config)
    print(json.dumps({
        "decision": summary["decision"]["decision"],
        "selected_alpha": summary["selected_alpha"],
        "report": str(Path(config["output"]["root"]) / "summary.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
