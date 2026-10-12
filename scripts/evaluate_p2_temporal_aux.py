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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.config import load_config
from sphererate.data.dataset import ProcessedSaliencyDataset
from sphererate.evaluation.metrics import mean_scores, saliency_scores
from sphererate.evaluation.p2_temporal_aux import p2_gate_decision
from sphererate.train import load_model_for_inference
from sphererate.utils import ensure_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _evaluate_run(
    checkpoint: Path,
    dataset: ProcessedSaliencyDataset,
    prediction_dir: Path,
    device_name: str,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    model, _, device = load_model_for_inference(checkpoint, device_name=device_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    ensure_dir(prediction_dir)
    scores: dict[str, dict[str, float]] = {}
    for batch in loader:
        prediction = model(batch["image"].to(device, non_blocking=True))["final"].detach().cpu().numpy()
        gt = batch["final"].numpy()
        fixation = batch["fixation"].numpy()
        area = batch["area_weights"].numpy()
        for index, image_id in enumerate(batch["image_id"]):
            image_id = str(image_id)
            current = saliency_scores(
                prediction[index, 0], gt[index, 0], fixation[index, 0], area[index, 0]
            ).as_dict()
            scores[image_id] = current
            np.savez_compressed(prediction_dir / f"{image_id}.npz", final=prediction[index, 0].astype(np.float32))
    counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    return scores, counts


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Gate P2: Image-Only Temporal Auxiliary",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "Inference input: `ERP RGB + ERP geometry` only. The auxiliary head is training-only and does not feed final predictions.",
        "",
        "## Frozen gate results",
        "",
        "| Method | KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ |",
        "|---|---:|---:|---:|",
    ]
    for role in ("final_only", "temporal_correct", "temporal_shuffled", "capacity_final_repeat"):
        score = summary["method_means"][role]
        lines.append(f"| {role} | {score['kld_sph']:.6f} | {score['sim_sph']:.6f} | {score['cc_sph']:.6f} |")
    lines.extend(["", "## Preregistered comparisons", ""])
    for name, metrics in summary["comparisons"].items():
        lines.extend([f"### {name}", ""])
        for metric, result in metrics.items():
            lines.append(
                f"- {metric}: improvement={result['oriented_improvement_mean']:.6f}, "
                f"95% CI [{result['ci95_low']:.6f}, {result['ci95_high']:.6f}]"
            )
        lines.append("")
    lines.extend(["## Gate checks", ""])
    lines.extend(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in summary["checks"].items())
    lines.extend(
        [
            "",
            f"All methods have identical parameter counts: **{summary['parameter_match']}**.",
            "Gate contains 13 outer-validation contents; dev selected checkpoints; outer test was not read.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen P2 temporal-auxiliary gate.")
    parser.add_argument("--config", default="configs/p2_salient360_temporal_aux_256x512.yaml")
    parser.add_argument("--batch", default="configs/batch_p2_temporal_aux_pilot.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="reports/gate_p2_temporal_aux")
    args = parser.parse_args()

    base = load_config(args.config)
    batch = load_config(args.batch)
    if batch.get("decision_split") != "gate" or batch.get("test_status") != "frozen_not_evaluated":
        raise ValueError("P2 batch must use the frozen gate and keep test unopened")
    processed_root = Path(base["paths"]["processed_root"]) / base["data"]["processed_name"]
    dataset = ProcessedSaliencyDataset(processed_root, split="gate", augment_roll=False)
    if len(dataset) != 13:
        raise ValueError(f"Frozen P2 gate must contain 13 contents, found {len(dataset)}")
    groups = {str(row["image_id"]): str(row["canonical_content_id"]) for row in dataset.rows}

    all_scores: dict[str, dict[str, dict[str, float]]] = {}
    parameters: dict[str, dict[str, int]] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for run in batch["runs"]:
        role = str(run["role"])
        run_name = str(run["name"])
        checkpoint = Path(base["paths"]["runs_root"]) / run_name / "checkpoints" / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing P2 checkpoint: {checkpoint}")
        scores, counts = _evaluate_run(
            checkpoint,
            dataset,
            Path(base["paths"]["runs_root"]) / run_name / "predictions" / "gate",
            args.device,
            int(base["train"]["batch_size"]),
            int(base["train"]["num_workers"]),
        )
        all_scores[role] = scores
        parameters[role] = counts
        checkpoints[role] = {"path": str(checkpoint), "sha256": _sha256(checkpoint)}

    result = p2_gate_decision(
        all_scores,
        groups,
        iterations=int(batch["bootstrap_resamples"]),
        seed=int(batch["bootstrap_seed"]),
    )
    parameter_match = len({(item["total"], item["trainable"]) for item in parameters.values()}) == 1
    result["checks"]["identical_parameter_counts"] = parameter_match
    if not parameter_match:
        result["decision"] = "FAIL"
    result.update(
        {
            "gate": "P2-pilot",
            "split": "gate",
            "num_contents": len(dataset),
            "method_means": {role: mean_scores(list(scores.values())) for role, scores in all_scores.items()},
            "parameters": parameters,
            "parameter_match": parameter_match,
            "checkpoints": checkpoints,
            "test_used": False,
        }
    )

    output_dir = ensure_dir(args.output_dir)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    rows = []
    for role, scores in all_scores.items():
        for image_id, metrics in scores.items():
            rows.append({"method": role, "image_id": image_id, "canonical_content_id": groups[image_id], **metrics})
    with (output_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(result, output_dir / "summary.md")
    print(json.dumps({"decision": result["decision"], "report": str(output_dir / 'summary.md')}, indent=2))


if __name__ == "__main__":
    main()

