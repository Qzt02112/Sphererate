#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
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
from sphererate.evaluation.early_model import compare_coverage_rows
from sphererate.evaluation.temporal_factorization import normalize_longitude_deg, reference_alignment_pixel_shift
from sphererate.evaluation.tile_prefetch import gaze_mass_coverage, rank_tiles
from sphererate.data.maps import MapGeometry
from sphererate.utils import ensure_dir
from scripts.evaluate_eh3_gate import _build_model_from_state


SEEDS = (20260710, 20260711, 20260712)
HORIZONS = (0.5, 1.0, 3.0, 5.0)
PRIMARY = {0.5, 1.0, 3.0}
SALIENT360_CANONICAL_START_YAW_DEG = 90.0


def _read_etdk2_val(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected = []
    test_metadata = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") != "etdk2":
                continue
            if row.get("split") == "test":
                test_metadata += 1
                continue
            if row.get("split") == "val":
                selected.append(row)
    return selected, {
        "selected_val_sessions": len(selected),
        "selected_val_contents": len({row["canonical_content_id"] for row in selected}),
        "outer_test_metadata_rows": test_metadata,
        "outer_test_target_rows_loaded": 0,
    }


def _load_rgb(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).float()


def _aligned_input(rgb: torch.Tensor, longitude_deg: float) -> tuple[torch.Tensor, int]:
    # Salient360 training images were viewed from the ERP image center, which
    # maps to +90 degrees (x_norm≈0.75) in the released scanpath convention.
    # Rotate an arbitrary ET-DK2 start into that exact training reference.
    shift = reference_alignment_pixel_shift(
        float(longitude_deg), SALIENT360_CANONICAL_START_YAW_DEG, rgb.shape[-1]
    )
    aligned = torch.roll(rgb, shifts=shift, dims=-1)
    return torch.cat([aligned, coord_channels(rgb.shape[-2], rgb.shape[-1])], dim=0), shift


@torch.no_grad()
def _predict_conditions(
    checkpoint: Path,
    conditions: list[tuple[str, float]],
    images: dict[str, torch.Tensor],
    templates: dict[str, torch.Tensor],
    wrong_image_map: dict[str, str],
    device: torch.device,
) -> dict[str, dict[tuple[str, float], np.ndarray]]:
    state = torch.load(checkpoint, map_location=device)
    model = _build_model_from_state(state, device)
    output: dict[str, dict[tuple[str, float], np.ndarray]] = {
        "candidate_correct": {},
        "wrong_start": {},
        "wrong_image": {},
        "early_template": {},
    }
    for content, start in conditions:
        template = templates[content].unsqueeze(0).to(device)
        correct_input, correct_shift = _aligned_input(images[content], start)
        prediction = model(correct_input.unsqueeze(0).to(device), template)["cumulative"][0]
        output["candidate_correct"][(content, start)] = torch.roll(
            prediction, shifts=-correct_shift, dims=-1
        ).cpu().numpy().astype(np.float32)

        wrong_start = normalize_longitude_deg(start + 180.0)
        wrong_input, wrong_shift = _aligned_input(images[content], wrong_start)
        wrong_prediction = model(wrong_input.unsqueeze(0).to(device), template)["cumulative"][0]
        output["wrong_start"][(content, start)] = torch.roll(
            wrong_prediction, shifts=-wrong_shift, dims=-1
        ).cpu().numpy().astype(np.float32)

        wrong_content = wrong_image_map[content]
        wrong_image_input, image_shift = _aligned_input(images[wrong_content], start)
        image_prediction = model(wrong_image_input.unsqueeze(0).to(device), template)["cumulative"][0]
        output["wrong_image"][(content, start)] = torch.roll(
            image_prediction, shifts=-image_shift, dims=-1
        ).cpu().numpy().astype(np.float32)

        output["early_template"][(content, start)] = torch.roll(
            templates[content], shifts=-correct_shift, dims=-1
        ).numpy().astype(np.float32)
    return output


def _session_coverage_rows(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[tuple[str, float], np.ndarray]],
) -> dict[str, list[dict[str, Any]]]:
    area = MapGeometry(256, 512).area_weights
    output = {method: [] for method in predictions}
    for row in rows:
        content = str(row["canonical_content_id"])
        start = float(row["start_erp_longitude_deg"])
        with np.load(row["fixations"]) as data:
            fixations = np.asarray(data["fixations"], dtype=np.float32)
        observer_ids = np.unique(fixations[:, 3]).astype(int).tolist()
        for method, values in predictions.items():
            maps = values[(content, start)]
            for index, horizon in enumerate(HORIZONS):
                selected = rank_tiles(
                    maps[index], area, tile_rows=8, tile_cols=16, budget_percent=10.0
                )
                output[method].append(
                    {
                        "content_id": content,
                        "session_id": str(row["image_id"]),
                        "subject_id": str(row["subject_id"]),
                        "start_erp_longitude_deg": start,
                        "horizon_s": horizon,
                        "coverage": gaze_mass_coverage(
                            fixations,
                            observer_ids=observer_ids,
                            horizon_s=horizon,
                            selected_tile_ids=selected,
                            tile_rows=8,
                            tile_cols=16,
                        ),
                    }
                )
    return output


def _content_average(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        groups[(str(row["content_id"]), float(row["horizon_s"]))].append(float(row["coverage"]))
    return [
        {"content_id": content, "horizon_s": horizon, "coverage": float(np.mean(values))}
        for (content, horizon), values in sorted(groups.items())
    ]


def _seed_average(rows_by_seed: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for rows in rows_by_seed:
        for row in rows:
            groups[(str(row["content_id"]), float(row["horizon_s"]))].append(float(row["coverage"]))
    return [
        {"content_id": content, "horizon_s": horizon, "coverage": float(np.mean(values))}
        for (content, horizon), values in sorted(groups.items())
    ]


def _format_ci(result: dict[str, Any]) -> str:
    return f"{result['mean']:.6f} [{result['ci95_low']:.6f}, {result['ci95_high']:.6f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EH4 on ET-DK2 real multi-start sessions.")
    parser.add_argument(
        "--manifest",
        default="data/processed/etdk2_salient360shared_256x512_k6_session_v1/manifest.jsonl",
    )
    parser.add_argument("--output-dir", default="reports/kernel_v2_etdk2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--template-root",
        default="data/processed/early_horizon_salient360_spherical2deg_v2/baselines/early_template",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default="runs/kernel_v2_candidate_correct_seed{seed}/checkpoints/best.pt",
    )
    args = parser.parse_args()
    rows, access = _read_etdk2_val(Path(args.manifest))
    if len(rows) != 126 or access["selected_val_contents"] != 7:
        raise ValueError(f"Expected 126 ET-DK2 val sessions / 7 contents, got {access}")
    contents = sorted({str(row["canonical_content_id"]) for row in rows})
    source_image = {content: next(row["source_image_id"] for row in rows if row["canonical_content_id"] == content) for content in contents}
    images = {
        content: _load_rgb(next(row["image"] for row in rows if row["canonical_content_id"] == content))
        for content in contents
    }
    templates = {}
    for content in contents:
        image_id = source_image[content]
        with np.load(Path(args.template_root) / f"{image_id}.npz") as data:
            templates[content] = torch.from_numpy(np.asarray(data["cumulative"], dtype=np.float32))
    wrong_image_map = {content: contents[(index + 1) % len(contents)] for index, content in enumerate(contents)}
    conditions = sorted({(str(row["canonical_content_id"]), float(row["start_erp_longitude_deg"])) for row in rows})
    device = torch.device(args.device)
    seed_results: dict[str, Any] = {}
    seed_content_rows: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    session_rows_seed11: dict[str, list[dict[str, Any]]] | None = None
    for seed in SEEDS:
        predictions = _predict_conditions(
            Path(str(args.checkpoint_pattern).format(seed=seed)),
            conditions,
            images,
            templates,
            wrong_image_map,
            device,
        )
        session_rows = _session_coverage_rows(rows, predictions)
        if seed == 20260711:
            session_rows_seed11 = session_rows
        content_rows = {method: _content_average(values) for method, values in session_rows.items()}
        comparisons = {
            control: compare_coverage_rows(
                content_rows["candidate_correct"],
                content_rows[control],
                iterations=10000,
                seed=20260711,
            )
            for control in ("early_template", "wrong_start", "wrong_image")
        }
        seed_results[str(seed)] = comparisons
        for method, values in content_rows.items():
            seed_content_rows[method].append(values)

    averaged = {method: _seed_average(values) for method, values in seed_content_rows.items()}
    aggregate = {
        control: compare_coverage_rows(
            averaged["candidate_correct"], averaged[control], iterations=10000, seed=20260711
        )
        for control in ("early_template", "wrong_start", "wrong_image")
    }
    checks = {
        "candidate_vs_template_ci_positive": aggregate["early_template"]["ci95_low"] > 0,
        "correct_vs_wrong_start_ci_positive": aggregate["wrong_start"]["ci95_low"] > 0,
        "correct_vs_wrong_image_ci_positive": aggregate["wrong_image"]["ci95_low"] > 0,
        "all_seed_directions_positive": all(
            result[control]["mean"] > 0
            for result in seed_results.values()
            for control in ("early_template", "wrong_start", "wrong_image")
        ),
        "outer_test_not_used": access["outer_test_target_rows_loaded"] == 0,
    }
    result = {
        "gate": "EH4-ETDK2-real-multi-start",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "access_audit": access,
        "conditions": len(conditions),
        "start_longitudes_deg": sorted({condition[1] for condition in conditions}),
        "salient360_canonical_start_yaw_deg": SALIENT360_CANONICAL_START_YAW_DEG,
        "wrong_image_map": wrong_image_map,
        "per_seed": seed_results,
        "seed_average": aggregate,
        "bootstrap_group": "canonical_content_id",
        "outer_test_used": False,
    }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if session_rows_seed11 is not None:
        import csv
        flat = [dict(row, method=method) for method, values in session_rows_seed11.items() for row in values]
        with (output / "per_session_seed20260711.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    lines = [
        "# EH4: ET-DK2 Real Multi-Start Validation",
        "",
        f"Decision: **{result['decision']}**",
        "",
        f"Evaluated `{len(rows)}` sessions from `{access['selected_val_contents']}` content groups and `{len(result['start_longitudes_deg'])}` declared starts. Outer-test targets loaded: `0`.",
        "",
        "## Per-seed early-macro 10% coverage improvements",
        "",
        "| Seed | vs early template | vs wrong start | vs wrong image |",
        "|---:|---:|---:|---:|",
    ]
    for seed, comparisons in seed_results.items():
        lines.append(
            f"| {seed} | {_format_ci(comparisons['early_template'])} | "
            f"{_format_ci(comparisons['wrong_start'])} | {_format_ci(comparisons['wrong_image'])} |"
        )
    lines.extend(
        [
            "",
            "## Seed-averaged content-bootstrap evidence",
            "",
            f"- candidate vs early template: {_format_ci(aggregate['early_template'])}",
            f"- correct vs wrong start: {_format_ci(aggregate['wrong_start'])}",
            f"- correct vs wrong image: {_format_ci(aggregate['wrong_image'])}",
            "",
            "## Checks",
            "",
        ]
    )
    lines.extend(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in checks.items())
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "report": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
