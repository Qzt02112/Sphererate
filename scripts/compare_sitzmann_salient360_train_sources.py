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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.metrics import cc, kld, sim
from sphererate.evaluation.tile_prefetch import gaze_mass_coverage, rank_tiles
from sphererate.models.early_horizon import TemplateTimeGatedResidualNet
from sphererate.utils import ensure_dir


HORIZONS = (0.5, 1.0, 3.0, 5.0)
EARLY = {0.5, 1.0, 3.0}
COVERAGE_BUDGETS = (5.0, 10.0, 20.0)
METRICS = ("kld", "sim", "cc", *(f"coverage_{budget:g}" for budget in COVERAGE_BUDGETS))
START_X = {
    "sitzmann": 0.5,
    "salient360": 0.75,
    "salient360_to_sitzmann_finetune": 0.5,
}


def _model_from_checkpoint(path: Path, device: torch.device) -> tuple[TemplateTimeGatedResidualNet, dict[str, Any]]:
    state = torch.load(path, map_location=device)
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
    return model, state


def _source_template(root: Path) -> torch.Tensor:
    dataset = EarlyHorizonDataset(root, split="val", target_mode="correct")
    return dataset[0]["template"].clone()


def _move_sample_to_start(sample: dict[str, Any], *, shift_px: int) -> tuple[torch.Tensor, torch.Tensor]:
    image = sample["image"].clone()
    if shift_px:
        # RGB and saliency move together. ERP coordinate channels describe the
        # destination grid and therefore must be recomputed/left unrolled.
        image[:3] = torch.roll(image[:3], shifts=int(shift_px), dims=-1)
    target = torch.roll(sample["target"], shifts=int(shift_px), dims=-1) if shift_px else sample["target"]
    return image, target


def _move_fixations_to_start(fixations: np.ndarray, *, shift_px: int, width: int = 512) -> np.ndarray:
    moved = np.asarray(fixations, dtype=np.float32).copy()
    if shift_px:
        moved[:, 0] = np.mod(moved[:, 0] + float(shift_px) / float(width), 1.0)
    return moved


def _score(
    *,
    train_source: str,
    eval_dataset: str,
    checkpoint: Path | None,
    dataset: EarlyHorizonDataset,
    source_template: torch.Tensor,
    template_policy: str,
    device: torch.device,
    seed: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    model = None
    state = None
    if checkpoint is not None:
        model, state = _model_from_checkpoint(checkpoint, device)
    shift = int(round((START_X[train_source] - START_X[eval_dataset]) * 512.0))
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            image, target = _move_sample_to_start(sample, shift_px=shift)
            if model is None:
                prediction = source_template
                method = "source_template"
            else:
                prediction = model(image[None].to(device), source_template[None].to(device))["cumulative"][0].cpu()
                method = "spherehorizon"
            area = sample["area_weights"].numpy()[0]
            target_np = target.numpy()
            prediction_np = prediction.numpy()
            with np.load(str(sample["fixations_path"])) as fixation_data:
                fixations = _move_fixations_to_start(
                    np.asarray(fixation_data["fixations"], dtype=np.float32),
                    shift_px=shift,
                    width=prediction_np.shape[-1],
                )
            observers = np.unique(fixations[:, 3]).astype(int).tolist()
            for horizon_index, horizon in enumerate(HORIZONS):
                coverage = {}
                for budget in COVERAGE_BUDGETS:
                    selected = rank_tiles(
                        prediction_np[horizon_index],
                        area,
                        tile_rows=8,
                        tile_cols=16,
                        budget_percent=budget,
                    )
                    coverage[f"coverage_{budget:g}"] = gaze_mass_coverage(
                        fixations,
                        observer_ids=observers,
                        horizon_s=float(horizon),
                        selected_tile_ids=selected,
                        tile_rows=8,
                        tile_cols=16,
                    )
                records.append(
                    {
                        "train_source": train_source,
                        "eval_dataset": eval_dataset,
                        "method": method,
                        "template_policy": str(template_policy),
                        "seed": "template" if seed is None else int(seed),
                        "scene_id": str(sample["canonical_content_id"]),
                        "image_id": str(sample["image_id"]),
                        "horizon_s": horizon,
                        "kld": kld(prediction_np[horizon_index], target_np[horizon_index], area),
                        "sim": sim(prediction_np[horizon_index], target_np[horizon_index], area),
                        "cc": cc(prediction_np[horizon_index], target_np[horizon_index], area),
                        **coverage,
                    }
                )
    provenance = None
    if state is not None:
        provenance = {
            "checkpoint": str(checkpoint),
            "epoch": int(state["epoch"]),
            "validation": state["val"],
        }
    return records, provenance


def _scene_macro(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scene_id"]), float(row["horizon_s"]))].append(row)
    scene_horizon = []
    for (scene, horizon), current in grouped.items():
        scene_horizon.append(
            {
                "scene": scene,
                "horizon_s": horizon,
                **{
                    metric: float(np.mean([float(row[metric]) for row in current]))
                    for metric in METRICS
                },
            }
        )
    per_horizon = {}
    for horizon in HORIZONS:
        current = [row for row in scene_horizon if row["horizon_s"] == horizon]
        per_horizon[f"{horizon:g}"] = {
            metric: float(np.mean([row[metric] for row in current])) for metric in METRICS
        }
    early = [row for row in scene_horizon if row["horizon_s"] in EARLY]
    return {
        "scene_count": len({row["scene"] for row in scene_horizon}),
        "early_macro": {
            metric: float(np.mean([row[metric] for row in early])) for metric in METRICS
        },
        "per_horizon": per_horizon,
    }


def _mean_seed_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seed_count": len(summaries),
        "early_macro": {
            metric: {
                "mean": float(np.mean([summary["early_macro"][metric] for summary in summaries])),
                "std": float(np.std([summary["early_macro"][metric] for summary in summaries], ddof=1)),
                "per_seed": [float(summary["early_macro"][metric]) for summary in summaries],
            }
            for metric in METRICS
        },
    }


def _write_summary(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Controlled Train-Source Comparison",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "All SphereHorizon scratch runs use the same architecture, ImageNet initialization, loss, optimizer, 30-epoch budget, and three seeds. The bounded fine-tune uses the matched Salient360 checkpoint, 15 epochs, and a lower learning rate. Maps are horizontally rotated only for frozen cross-dataset evaluation so that the declared start occupies the training source's coordinate; RGB and target are moved together while ERP coordinate channels remain tied to the destination grid.",
        "",
        "## In-domain gate",
        "",
        "| Train/eval dataset | Template KLD | Model KLD ↓ | Template CC | Model CC ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in ("salient360", "sitzmann"):
        item = result["comparisons"][f"{dataset}_to_{dataset}"]
        lines.append(
            f"| {dataset} | {item['template']['early_macro']['kld']:.4f} | "
            f"{item['model']['early_macro']['kld']['mean']:.4f}±{item['model']['early_macro']['kld']['std']:.4f} | "
            f"{item['template']['early_macro']['cc']:.4f} | "
            f"{item['model']['early_macro']['cc']['mean']:.4f}±{item['model']['early_macro']['cc']['std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen cross-dataset gate",
            "",
            "| Train source → evaluation | Source-template KLD | Model KLD ↓ | Source-template CC | Model CC ↑ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in ("salient360_to_sitzmann", "sitzmann_to_salient360"):
        item = result["comparisons"][key]
        lines.append(
            f"| {key.replace('_to_', ' → ')} | {item['template']['early_macro']['kld']:.4f} | "
            f"{item['model']['early_macro']['kld']['mean']:.4f}±{item['model']['early_macro']['kld']['std']:.4f} | "
            f"{item['template']['early_macro']['cc']:.4f} | "
            f"{item['model']['early_macro']['cc']['mean']:.4f}±{item['model']['early_macro']['cc']['std']:.4f} |"
        )
    fine = result["finetune"]["sitzmann_gate"]
    scratch = result["comparisons"]["sitzmann_to_sitzmann"]["model"]["early_macro"]
    frozen = result["target_template_diagnostic"]["salient360_to_sitzmann"]["model"]["early_macro"]
    template = result["comparisons"]["sitzmann_to_sitzmann"]["template"]["early_macro"]
    lines.extend(
        [
            "",
            "## Sitzmann adaptation ablation",
            "",
            "| Initialization/adaptation | KLD ↓ | SIM ↑ | CC ↑ |",
            "|---|---:|---:|---:|",
            f"| Sitzmann train-only template | {template['kld']:.4f} | {template['sim']:.4f} | {template['cc']:.4f} |",
            f"| Salient360 frozen residual | {frozen['kld']['mean']:.4f}±{frozen['kld']['std']:.4f} | {frozen['sim']['mean']:.4f}±{frozen['sim']['std']:.4f} | {frozen['cc']['mean']:.4f}±{frozen['cc']['std']:.4f} |",
            f"| Sitzmann from scratch | {scratch['kld']['mean']:.4f}±{scratch['kld']['std']:.4f} | {scratch['sim']['mean']:.4f}±{scratch['sim']['std']:.4f} | {scratch['cc']['mean']:.4f}±{scratch['cc']['std']:.4f} |",
            f"| Salient360 → Sitzmann fine-tune | {fine['early_macro']['kld']['mean']:.4f}±{fine['early_macro']['kld']['std']:.4f} | {fine['early_macro']['sim']['mean']:.4f}±{fine['early_macro']['sim']['std']:.4f} | {fine['early_macro']['cc']['mean']:.4f}±{fine['early_macro']['cc']['std']:.4f} |",
        ]
    )
    lines.extend(
        [
            "",
            "## Cross-dataset residual transfer with target train-only template",
            "",
            "This diagnostic holds the evaluation-domain population prior fixed and changes only which dataset trained the content residual.",
            "",
            "| Train source → evaluation | Target-template KLD | Model KLD ↓ | Target-template CC | Model CC ↑ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in ("salient360_to_sitzmann", "sitzmann_to_salient360"):
        item = result["target_template_diagnostic"][key]
        lines.append(
            f"| {key.replace('_to_', ' → ')} | {item['template']['early_macro']['kld']:.4f} | "
            f"{item['model']['early_macro']['kld']['mean']:.4f}±{item['model']['early_macro']['kld']['std']:.4f} | "
            f"{item['template']['early_macro']['cc']:.4f} | "
            f"{item['model']['early_macro']['cc']['mean']:.4f}±{item['model']['early_macro']['cc']['std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            *[f"- {line}" for line in result["interpretation"]],
            "",
            "## Experiment consequence",
            "",
            result["consequence"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled Sitzmann-vs-Salient360 training-source comparison.")
    parser.add_argument("--sitzmann-root", default="data/processed/sitzmann_official_256x512_spherical2deg_v1")
    parser.add_argument("--salient360-root", default="data/processed/early_horizon_salient360_spherical2deg_v2")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", default="reports/sitzmann_train_source_comparison")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    roots = {"sitzmann": Path(args.sitzmann_root), "salient360": Path(args.salient360_root)}
    datasets = {
        name: EarlyHorizonDataset(root, split="gate", target_mode="correct") for name, root in roots.items()
    }
    templates = {name: _source_template(root) for name, root in roots.items()}
    seeds = (20260710, 20260711, 20260712)
    checkpoints = {
        "salient360": {
            seed: Path(args.runs_root) / f"kernel_v2_candidate_correct_seed{seed}" / "checkpoints" / "best.pt"
            for seed in seeds
        },
        "sitzmann": {
            seed: Path(args.runs_root)
            / f"sitzmann_official_candidate_correct_seed{seed}"
            / "checkpoints"
            / "best.pt"
            for seed in seeds
        },
    }
    finetune_checkpoints = {
        seed: Path(args.runs_root) / f"salient360_to_sitzmann_finetune_seed{seed}" / "checkpoints" / "best.pt"
        for seed in seeds
    }
    missing = [str(path) for source in checkpoints.values() for path in source.values() if not path.is_file()]
    missing.extend(str(path) for path in finetune_checkpoints.values() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    all_rows: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    target_template_diagnostic: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for train_source in ("salient360", "sitzmann"):
        for eval_dataset in ("salient360", "sitzmann"):
            key = f"{train_source}_to_{eval_dataset}"
            template_rows, _ = _score(
                train_source=train_source,
                eval_dataset=eval_dataset,
                checkpoint=None,
                dataset=datasets[eval_dataset],
                source_template=templates[train_source],
                template_policy="source_domain",
                device=device,
                seed=None,
            )
            all_rows.extend(template_rows)
            seed_summaries = []
            provenance[key] = []
            for seed in seeds:
                rows, current_provenance = _score(
                    train_source=train_source,
                    eval_dataset=eval_dataset,
                    checkpoint=checkpoints[train_source][seed],
                    dataset=datasets[eval_dataset],
                    source_template=templates[train_source],
                    template_policy="source_domain",
                    device=device,
                    seed=seed,
                )
                all_rows.extend(rows)
                seed_summaries.append(_scene_macro(rows))
                provenance[key].append(current_provenance)
                print(f"{key} seed={seed}", flush=True)
            comparisons[key] = {
                "template": _scene_macro(template_rows),
                "model": _mean_seed_summaries(seed_summaries),
            }
            if train_source != eval_dataset:
                shift = int(round((START_X[train_source] - START_X[eval_dataset]) * 512.0))
                aligned_target_template = torch.roll(templates[eval_dataset], shifts=shift, dims=-1)
                target_template_rows, _ = _score(
                    train_source=train_source,
                    eval_dataset=eval_dataset,
                    checkpoint=None,
                    dataset=datasets[eval_dataset],
                    source_template=aligned_target_template,
                    template_policy="target_domain_train_only",
                    device=device,
                    seed=None,
                )
                all_rows.extend(target_template_rows)
                target_seed_summaries = []
                for seed in seeds:
                    rows, _ = _score(
                        train_source=train_source,
                        eval_dataset=eval_dataset,
                        checkpoint=checkpoints[train_source][seed],
                        dataset=datasets[eval_dataset],
                        source_template=aligned_target_template,
                        template_policy="target_domain_train_only",
                        device=device,
                        seed=seed,
                    )
                    all_rows.extend(rows)
                    target_seed_summaries.append(_scene_macro(rows))
                target_template_diagnostic[key] = {
                    "template": _scene_macro(target_template_rows),
                    "model": _mean_seed_summaries(target_seed_summaries),
                }

    finetune: dict[str, Any] = {}
    for eval_dataset in ("sitzmann", "salient360"):
        train_source = "salient360_to_sitzmann_finetune"
        shift = int(round((START_X[train_source] - START_X[eval_dataset]) * 512.0))
        aligned_target_template = torch.roll(templates[eval_dataset], shifts=shift, dims=-1)
        summaries = []
        for seed in seeds:
            rows, current_provenance = _score(
                train_source=train_source,
                eval_dataset=eval_dataset,
                checkpoint=finetune_checkpoints[seed],
                dataset=datasets[eval_dataset],
                source_template=aligned_target_template,
                template_policy="target_domain_train_only",
                device=device,
                seed=seed,
            )
            all_rows.extend(rows)
            summaries.append(_scene_macro(rows))
            provenance.setdefault(f"finetune_to_{eval_dataset}", []).append(current_provenance)
        finetune[f"{eval_dataset}_gate"] = _mean_seed_summaries(summaries)

    sal_in = comparisons["salient360_to_salient360"]
    sitz_in = comparisons["sitzmann_to_sitzmann"]
    sitz_gain_kld = sitz_in["template"]["early_macro"]["kld"] - sitz_in["model"]["early_macro"]["kld"]["mean"]
    sitz_gain_cc = sitz_in["model"]["early_macro"]["cc"]["mean"] - sitz_in["template"]["early_macro"]["cc"]
    sal_gain_kld = sal_in["template"]["early_macro"]["kld"] - sal_in["model"]["early_macro"]["kld"]["mean"]
    sal_gain_cc = sal_in["model"]["early_macro"]["cc"]["mean"] - sal_in["template"]["early_macro"]["cc"]
    replace = sitz_gain_kld > 0 and sitz_gain_cc > 0 and sitz_gain_kld > sal_gain_kld and sitz_gain_cc > sal_gain_cc
    result = {
        "decision": "REPLACE_PRIMARY_WITH_SITZMANN" if replace else "KEEP_SALIENT360_PRIMARY; SITZMANN_COMPLEMENT/FINE-TUNE",
        "comparisons": comparisons,
        "target_template_diagnostic": target_template_diagnostic,
        "finetune": finetune,
        "in_domain_model_gain": {
            "salient360": {"kld_reduction": sal_gain_kld, "cc_increase": sal_gain_cc},
            "sitzmann": {"kld_reduction": sitz_gain_kld, "cc_increase": sitz_gain_cc},
        },
        "interpretation": [
            "In-domain absolute scores are not directly comparable across datasets; the meaningful quantity is model improvement over that dataset's train-only template.",
            "Sitzmann validation selects epochs 3–7 and then degrades, which is the expected overfitting signature of 16 unique training scenes.",
            "Frozen whole-system transfer uses the source-domain template; the target-template diagnostic separately isolates whether the learned content residual generalizes.",
            "The fine-tune ablation tests whether Sitzmann is best used after broader Salient360 training rather than as the sole source.",
        ],
        "consequence": (
            "The initial controlled comparison did not support replacing Salient360. The subsequently authorized full K6 suite confirmed Salient360 as the primary content-diverse benchmark and Sitzmann as a secondary in-domain/frozen-transfer track."
            if not replace
            else "Retrain the trainable matched baselines on Sitzmann and retain Salient360 as frozen external evaluation."
        ),
        "seeds": list(seeds),
        "start_alignment": {
            "sitzmann_x_norm": START_X["sitzmann"],
            "salient360_x_norm": START_X["salient360"],
            "cross_dataset_policy": "roll RGB and target to source start; keep destination ERP coordinate channels",
        },
        "provenance": provenance,
    }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "per_scene_horizon.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    _write_summary(result, output / "summary.md")
    print(json.dumps({"decision": result["decision"], "summary": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
