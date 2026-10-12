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

from sphererate.config import load_config
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.data.maps import MapGeometry
from sphererate.evaluation.early_model import (
    compare_coverage_rows,
    compare_method_rows,
    decide_eh3,
)
from sphererate.evaluation.metrics import cc, kld, sim
from sphererate.evaluation.tile_prefetch import gaze_mass_coverage, rank_tiles
from sphererate.models.early_horizon import TemplateTimeGatedResidualNet
from sphererate.train import load_model_for_inference
from sphererate.utils import ensure_dir


METHOD_ORDER = (
    "early_template",
    "static_final_repeat",
    "candidate_correct",
    "candidate_shuffled_label",
    "capacity_non_temporal",
    "wrong_time",
    "wrong_start",
    "wrong_image",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_model_from_state(state: dict[str, Any], device: torch.device) -> TemplateTimeGatedResidualNet:
    config = state["config"]
    model_cfg = config["model"]
    model = TemplateTimeGatedResidualNet(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        hidden_channels=int(model_cfg["hidden_channels"]),
        horizons_s=config["data"]["horizons_s"],
        conditioning_mode=str(state["run"]["conditioning_mode"]),
        semantic_backbone=str(model_cfg["semantic_backbone"]),
        semantic_layer=str(model_cfg["semantic_layer"]),
        semantic_pretrained=bool(model_cfg["semantic_pretrained"]),
        residual_fusion=str(model_cfg.get("residual_fusion", "monotonic_gate")),
        residual_bound=float(model_cfg.get("residual_bound", 4.0)),
        semantic_alpha_init=model_cfg.get("semantic_alpha_init"),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def _predict_learned(
    checkpoint: Path,
    samples: list[dict[str, Any]],
    device: torch.device,
    *,
    counterfactual: str | None = None,
    batch_size: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    state = torch.load(checkpoint, map_location=device)
    model = _build_model_from_state(state, device)
    predictions: dict[str, np.ndarray] = {}
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    for start in range(0, len(samples), batch_size):
        indices = list(range(start, min(start + batch_size, len(samples))))
        image_indices = indices
        if counterfactual == "wrong_image":
            image_indices = [(index + 1) % len(samples) for index in indices]
        image = torch.stack([samples[index]["image"] for index in image_indices]).to(device)
        template = torch.stack([samples[index]["template"] for index in indices]).to(device)
        kwargs: dict[str, Any] = {}
        restore_shift = 0
        if counterfactual == "wrong_time":
            permutation_device = permutation.to(device)
            template = template[:, permutation_device]
            kwargs["horizons_s"] = model.horizons_s[permutation_device]
        elif counterfactual == "wrong_start":
            restore_shift = image.shape[-1] // 2
            image = torch.roll(image, shifts=-restore_shift, dims=-1)
        output = model(image, template, **kwargs)["cumulative"]
        if counterfactual == "wrong_start":
            output = torch.roll(output, shifts=restore_shift, dims=-1)
        output_np = output.detach().cpu().numpy()
        for local, index in enumerate(indices):
            predictions[str(samples[index]["image_id"])] = output_np[local].astype(np.float32)
    return predictions, {
        "checkpoint": str(checkpoint),
        "sha256": _sha256(checkpoint),
        "parameters": state["parameters"],
        "epoch": int(state["epoch"]),
        "val": state["val"],
    }


@torch.no_grad()
def _predict_static(
    checkpoint: Path,
    samples: list[dict[str, Any]],
    device_name: str,
    *,
    batch_size: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model, _, device = load_model_for_inference(checkpoint, device_name=device_name)
    predictions: dict[str, np.ndarray] = {}
    for start in range(0, len(samples), batch_size):
        current = samples[start : start + batch_size]
        image = torch.stack([sample["image"] for sample in current]).to(device)
        final = model(image)["final"].detach().cpu().numpy()
        for index, sample in enumerate(current):
            predictions[str(sample["image_id"])] = np.repeat(final[index], 4, axis=0).astype(np.float32)
    return predictions, {"checkpoint": str(checkpoint), "sha256": _sha256(checkpoint)}


def _score_methods(
    predictions: dict[str, dict[str, np.ndarray]],
    samples: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    score_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in predictions}
    coverage_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in predictions}
    for sample in samples:
        image_id = str(sample["image_id"])
        content = str(sample["canonical_content_id"])
        target = sample["target"].numpy()
        area = sample["area_weights"].numpy()[0]
        horizons = sample["horizons_s"].numpy()
        with np.load(str(sample["fixations_path"])) as fixation_data:
            fixations = np.asarray(fixation_data["fixations"], dtype=np.float32)
        observer_ids = np.unique(fixations[:, 3]).astype(int).tolist()
        for method, method_predictions in predictions.items():
            prediction = method_predictions[image_id]
            for index, horizon in enumerate(horizons):
                score_rows[method].append(
                    {
                        "method": method,
                        "image_id": image_id,
                        "content_id": content,
                        "horizon_s": float(horizon),
                        "kld_sph": kld(prediction[index], target[index], area),
                        "sim_sph": sim(prediction[index], target[index], area),
                        "cc_sph": cc(prediction[index], target[index], area),
                    }
                )
                selected = rank_tiles(
                    prediction[index],
                    area,
                    tile_rows=8,
                    tile_cols=16,
                    budget_percent=10.0,
                )
                coverage_rows[method].append(
                    {
                        "method": method,
                        "image_id": image_id,
                        "content_id": content,
                        "horizon_s": float(horizon),
                        "coverage": gaze_mass_coverage(
                            fixations,
                            observer_ids=observer_ids,
                            horizon_s=float(horizon),
                            selected_tile_ids=selected,
                            tile_rows=8,
                            tile_cols=16,
                        ),
                    }
                )
    return score_rows, coverage_rows


def _method_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    primary = [row for row in rows if float(row["horizon_s"]) in {0.5, 1.0, 3.0}]
    return {metric: float(np.mean([row[metric] for row in primary])) for metric in ("kld_sph", "sim_sph", "cc_sph")}


def _format_ci(result: dict[str, Any]) -> str:
    return f"{result['mean']:.6f} [{result['ci95_low']:.6f}, {result['ci95_high']:.6f}]"


def _write_summary(result: dict[str, Any], path: Path) -> None:
    outer_note = (
        "Development gate is historically seen. V2 test target files were not loaded in this run; "
        "the outer-test contents were already used under v1 and are not a fresh confirmatory set."
        if result.get("outer_test_status") == "previously_used_under_v1_targets"
        else "Development gate is historically seen and is not presented as confirmatory evidence. Outer-test remains sealed."
    )
    lines = [
        "# Gate EH3: Minimal Template-Residual Pilot",
        "",
        f"Decision: **{result['decision']}**",
        "",
        outer_note,
        "",
        "## Early-macro map metrics",
        "",
        "| Method | KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ |",
        "|---|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        score = result["method_means"][method]
        lines.append(f"| {method} | {score['kld_sph']:.6f} | {score['sim_sph']:.6f} | {score['cc_sph']:.6f} |")
    lines.extend(["", "## Candidate improvements", ""])
    for control, comparison in result["comparisons"].items():
        macro = comparison["early_macro"]
        lines.extend(
            [
                f"### vs {control}",
                "",
                f"- KLD: {_format_ci(macro['kld_sph'])}",
                f"- SIM: {_format_ci(macro['sim_sph'])}",
                f"- CC: {_format_ci(macro['cc_sph'])}",
                "",
            ]
        )
    lines.extend(["## 10% tile coverage improvements", ""])
    for control, comparison in result["coverage_comparisons"].items():
        lines.append(f"- candidate vs {control}: {_format_ci(comparison)}")
    lines.extend(["", "## Gate checks", ""])
    lines.extend(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in result["checks"].items())
    lines.extend(
        [
            "",
            f"Matched learned-run parameter counts: **{result['parameter_match']}**.",
            f"Outer-test used: **{result['outer_test_used']}**.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-shot EH3 development-gate decision.")
    parser.add_argument("--config", default="configs/eh3_template_residual_256x512.yaml")
    parser.add_argument("--batch", default="configs/batch_eh3_pilot.yaml")
    parser.add_argument("--output-dir", default="reports/gate_eh3_template_residual")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    batch = load_config(args.batch)
    allowed_outer_status = {"sealed_not_materialized", "previously_used_under_v1_targets"}
    if batch["decision_split"] != "gate" or batch["outer_test_status"] not in allowed_outer_status:
        raise ValueError("EH3 gate/test status mismatch")
    processed_root = Path(config["paths"]["processed_root"]) / config["paths"]["processed_name"]
    dataset = EarlyHorizonDataset(processed_root, split="gate", target_mode="correct")
    if len(dataset) != 13:
        raise ValueError(f"EH3 development gate must contain 13 contents, found {len(dataset)}")
    samples = [dataset[index] for index in range(len(dataset))]
    device = torch.device(args.device)
    run_by_role = {str(run["role"]): run for run in batch["runs"]}
    learned_predictions: dict[str, dict[str, np.ndarray]] = {}
    provenance: dict[str, Any] = {}
    for role in ("candidate_correct", "candidate_shuffled_label", "capacity_non_temporal"):
        checkpoint = Path(config["paths"]["runs_root"]) / run_by_role[role]["name"] / "checkpoints" / "best.pt"
        learned_predictions[role], provenance[role] = _predict_learned(checkpoint, samples, device)
    candidate_checkpoint = Path(provenance["candidate_correct"]["checkpoint"])
    for counterfactual in ("wrong_time", "wrong_start", "wrong_image"):
        learned_predictions[counterfactual], _ = _predict_learned(
            candidate_checkpoint,
            samples,
            device,
            counterfactual=counterfactual,
        )
    predictions: dict[str, dict[str, np.ndarray]] = dict(learned_predictions)
    predictions["early_template"] = {
        str(sample["image_id"]): sample["template"].numpy().astype(np.float32) for sample in samples
    }
    static_checkpoint = Path(
        config.get("baseline", {}).get(
            "static_checkpoint",
            "runs/p2_final_only_seed20260711/checkpoints/best.pt",
        )
    )
    predictions["static_final_repeat"], provenance["static_final_repeat"] = _predict_static(
        static_checkpoint, samples, args.device
    )
    score_rows, coverage_rows = _score_methods(predictions, samples)
    comparisons = {
        control: compare_method_rows(
            score_rows["candidate_correct"],
            score_rows[control],
            iterations=10000,
            seed=20260711,
        )
        for control in METHOD_ORDER
        if control != "candidate_correct"
    }
    coverage_comparisons = {
        control: compare_coverage_rows(
            coverage_rows["candidate_correct"],
            coverage_rows[control],
            iterations=10000,
            seed=20260711,
        )
        for control in ("static_final_repeat", "early_template")
    }
    parameter_values = {
        (item["parameters"]["total"], item["parameters"]["trainable"])
        for role, item in provenance.items()
        if role in {"candidate_correct", "candidate_shuffled_label", "capacity_non_temporal"}
    }
    parameter_match = len(parameter_values) == 1
    decision = decide_eh3(
        comparisons,
        coverage_comparisons=coverage_comparisons,
        parameter_match=parameter_match,
        outer_test_used=False,
    )
    result = {
        "gate": "EH3-single-seed-template-residual",
        **decision,
        "method_means": {method: _method_means(score_rows[method]) for method in METHOD_ORDER},
        "comparisons": comparisons,
        "coverage_comparisons": coverage_comparisons,
        "provenance": provenance,
        "parameter_match": parameter_match,
        "development_gate_fresh": False,
        "outer_test_status": str(batch["outer_test_status"]),
        "outer_test_used": False,
        "no_alignment_control": "structurally identical on Salient360 because declared start yaw is 0 for every sample; deferred to ET-DK2",
    }
    output_dir = ensure_dir(args.output_dir)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    flat_scores = [row for method in METHOD_ORDER for row in score_rows[method]]
    with (output_dir / "per_content.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_scores[0]))
        writer.writeheader()
        writer.writerows(flat_scores)
    flat_coverage = [row for method in METHOD_ORDER for row in coverage_rows[method]]
    with (output_dir / "per_content_coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_coverage[0]))
        writer.writeheader()
        writer.writerows(flat_coverage)
    _write_summary(result, output_dir / "summary.md")
    print(json.dumps({"decision": result["decision"], "report": str(output_dir / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
