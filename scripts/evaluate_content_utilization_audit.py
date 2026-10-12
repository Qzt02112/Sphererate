#!/usr/bin/env python
"""Audit frozen SphereHorizon image-content utilization without training or tuning."""

from __future__ import annotations

import argparse
import csv
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
from sphererate.evaluation.early_model import compare_method_rows
from sphererate.evaluation.early_reliability import bootstrap_mean_ci
from sphererate.utils import ensure_dir
from scripts.evaluate_eh3_gate import _predict_learned, _score_methods


HORIZONS = (0.5, 1.0, 3.0, 5.0)
SEED = 20260711


def _by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
    return {(str(row["content_id"]), float(row["horizon_s"])): row for row in rows}


def _gate_values(checkpoint: Path) -> list[float]:
    state = torch.load(checkpoint, map_location="cpu")
    increments = torch.nn.functional.softplus(state["model"]["gate_deltas"].float())
    gates = torch.cumsum(increments, dim=0) / increments.sum().clamp_min(1e-7)
    return [float(value) for value in gates]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/content_utilization_audit")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = ensure_dir(args.output_dir)
    data_dir = ensure_dir(output / "data")
    dataset = EarlyHorizonDataset(
        "data/processed/early_horizon_salient360_v1",
        split="test",
        target_mode="correct",
        allow_outer_test=True,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    if len(samples) != 13:
        raise ValueError(f"Expected 13 frozen test contents, found {len(samples)}")

    checkpoint = Path(f"runs/eh3_candidate_correct_seed{SEED}/checkpoints/best.pt")
    device = torch.device(args.device)
    correct, provenance = _predict_learned(checkpoint, samples, device)
    wrong_image, wrong_provenance = _predict_learned(
        checkpoint, samples, device, counterfactual="wrong_image"
    )
    if provenance["sha256"] != wrong_provenance["sha256"]:
        raise RuntimeError("Correct and wrong-image inference did not use the same checkpoint")
    template = {
        str(sample["image_id"]): sample["template"].numpy().astype(np.float32)
        for sample in samples
    }
    scores, _ = _score_methods(
        {"correct": correct, "template": template, "wrong_image": wrong_image}, samples
    )
    correct_rows = _by_key(scores["correct"])
    template_rows = _by_key(scores["template"])
    wrong_rows = _by_key(scores["wrong_image"])

    comparison_template_seed11 = compare_method_rows(scores["correct"], scores["template"])
    comparison_wrong = compare_method_rows(scores["correct"], scores["wrong_image"])
    frozen_eh5 = json.loads(Path("reports/eh5_outer_test/results.json").read_text())
    comparison_template_seed_average = frozen_eh5["seed_average"]["map_vs_template"]
    recorded_wrong = frozen_eh5["attribution_seed20260711"]["wrong_image"]["map"]
    for horizon in HORIZONS:
        key = f"{horizon:g}"
        observed = comparison_wrong["per_horizon"][key]["kld_sph"]["mean"]
        expected = recorded_wrong["per_horizon"][key]["kld_sph"]["mean"]
        if abs(float(observed) - float(expected)) > 1e-8:
            raise RuntimeError(f"Wrong-image reproduction mismatch at {horizon:g}s")

    audit_rows = []
    l1_by_horizon: dict[str, dict[str, Any]] = {}
    fractions: dict[str, dict[str, float]] = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        l1_values: dict[str, float] = {}
        better_template = 0
        better_wrong = 0
        for sample_index, sample in enumerate(samples):
            image_id = str(sample["image_id"])
            content = str(sample["canonical_content_id"])
            area = sample["area_weights"].numpy()[0]
            key = (content, horizon)
            l1 = float((np.abs(correct[image_id][horizon_index] - wrong_image[image_id][horizon_index]) * area).sum())
            l1_values[content] = l1
            gain_template = float(template_rows[key]["kld_sph"] - correct_rows[key]["kld_sph"])
            gain_wrong = float(wrong_rows[key]["kld_sph"] - correct_rows[key]["kld_sph"])
            better_template += int(gain_template > 0.0)
            better_wrong += int(gain_wrong > 0.0)
            audit_rows.append(
                {
                    "image_id": image_id,
                    "content_id": content,
                    "wrong_image_id": str(samples[(sample_index + 1) % len(samples)]["image_id"]),
                    "horizon_s": horizon,
                    "kld_gain_vs_template": gain_template,
                    "kld_gain_vs_wrong_image": gain_wrong,
                    "spherical_l1_correct_vs_wrong": l1,
                }
            )
        l1_by_horizon[f"{horizon:g}"] = bootstrap_mean_ci(
            l1_values, iterations=10000, seed=SEED
        )
        fractions[f"{horizon:g}"] = {
            "correct_better_than_template": better_template / len(samples),
            "correct_better_than_wrong_image": better_wrong / len(samples),
        }

    with (output / "per_content.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    three_second = sorted(
        [row for row in audit_rows if float(row["horizon_s"]) == 3.0],
        key=lambda row: (float(row["kld_gain_vs_wrong_image"]), str(row["image_id"])),
    )
    selected = {
        "weakest": three_second[0],
        "median": three_second[len(three_second) // 2],
        "strongest": three_second[-1],
    }
    index_by_id = {str(sample["image_id"]): index for index, sample in enumerate(samples)}
    selection_rows = []
    for role, row in selected.items():
        image_id = str(row["image_id"])
        sample_index = index_by_id[image_id]
        wrong_index = (sample_index + 1) % len(samples)
        sample = samples[sample_index]
        wrong_sample = samples[wrong_index]
        rgb = np.clip(sample["image"][:3].numpy().transpose(1, 2, 0), 0.0, 1.0)
        wrong_rgb = np.clip(wrong_sample["image"][:3].numpy().transpose(1, 2, 0), 0.0, 1.0)
        np.savez_compressed(
            data_dir / f"{role}_{image_id}.npz",
            rgb=rgb.astype(np.float32),
            wrong_rgb=wrong_rgb.astype(np.float32),
            gt=sample["target"].numpy().astype(np.float32),
            template=template[image_id],
            correct=correct[image_id],
            wrong_image=wrong_image[image_id],
            horizons_s=sample["horizons_s"].numpy().astype(np.float32),
        )
        selection_rows.append(
            {
                "role": role,
                "image_id": image_id,
                "wrong_image_id": str(wrong_sample["image_id"]),
                "selection_metric": "3s correct-vs-wrong-image KLD improvement",
                "selection_value": float(row["kld_gain_vs_wrong_image"]),
                "candidate_vs_template_3s": float(row["kld_gain_vs_template"]),
            }
        )

    result = {
        "audit": "frozen content-utilization diagnostic",
        "training_performed": False,
        "model_or_checkpoint_changed": False,
        "post_test_tuning_performed": False,
        "split": "frozen test",
        "num_contents": len(samples),
        "checkpoint": provenance,
        "gates": dict(zip([f"{value:g}" for value in HORIZONS], _gate_values(checkpoint))),
        "candidate_vs_template_seed_average": comparison_template_seed_average,
        "candidate_vs_template_seed20260711": comparison_template_seed11,
        "candidate_vs_wrong_image_seed20260711": comparison_wrong,
        "spherical_l1_correct_vs_wrong_image": l1_by_horizon,
        "content_win_fractions": fractions,
        "wrong_image_policy": "next frozen-test image; target, template, time and start remain fixed",
        "selection": selection_rows,
        "interpretation_lock": (
            "Image dependence is distinct from image usefulness: a wrong image may change the output "
            "even when the correct-image prediction does not beat the template."
        ),
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (data_dir / "selection.json").write_text(
        json.dumps(selection_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"results": str(output / "results.json"), "selection": selection_rows}, indent=2))


if __name__ == "__main__":
    main()
