#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.tile_prefetch import budget_tile_count, tile_probability_mass
from sphererate.losses import weighted_cc, weighted_kld, weighted_sim
from sphererate.models.registry import build_model
from sphererate.utils import ensure_dir, seed_everything


@torch.no_grad()
def _collect(
    model: torch.nn.Module,
    dataset: EarlyHorizonDataset,
    device: torch.device,
) -> list[dict[str, Any]]:
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
    samples = []
    for batch in loader:
        prediction = model(batch["image"].to(device, non_blocking=True))["cumulative"]
        for index, content_id in enumerate(batch["canonical_content_id"]):
            samples.append(
                {
                    "content_id": str(content_id),
                    "prediction": prediction[index].detach().float().cpu(),
                    "template": batch["template"][index].float(),
                    "target": batch["target"][index].float(),
                    "area": batch["area_weights"][index].float(),
                    "raw_tiles": batch["raw_dwell_tiles"][index].numpy(),
                }
            )
    return samples


def _spherical_softmax(logits: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    stable = logits - logits.amax(dim=(-2, -1), keepdim=True)
    density = torch.exp(stable)
    return density / (density * area).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def _evaluate_alpha(
    samples: list[dict[str, Any]],
    *,
    horizon_index: int,
    alpha: float,
) -> dict[str, float]:
    rows = _evaluate_alpha_rows(samples, horizon_index=horizon_index, alpha=alpha)
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key != "content_id"
    }


def _evaluate_alpha_rows(
    samples: list[dict[str, Any]],
    *,
    horizon_index: int,
    alpha: float,
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        template = sample["template"][horizon_index : horizon_index + 1]
        baseline = sample["prediction"][horizon_index : horizon_index + 1]
        area = sample["area"]
        logits = (1.0 - float(alpha)) * template.clamp_min(1e-8).log()
        logits = logits + float(alpha) * baseline.clamp_min(1e-8).log()
        prediction = _spherical_softmax(logits, area)
        target = sample["target"][horizon_index : horizon_index + 1]
        row = {
            "content_id": sample["content_id"],
            "kld_sph": float(weighted_kld(prediction[None], target[None], area[None])),
            "sim_sph": float(weighted_sim(prediction[None], target[None], area[None])),
            "cc_sph": float(weighted_cc(prediction[None], target[None], area[None])),
        }
        mass = tile_probability_mass(
            prediction[0].numpy(), area[0].numpy(), tile_rows=8, tile_cols=16
        ).ravel()
        order = np.lexsort((np.arange(mass.size), -mass))
        raw = sample["raw_tiles"][horizon_index].ravel()
        for budget in (5.0, 10.0, 20.0):
            count = budget_tile_count(8, 16, budget)
            row[f"coverage_{budget:g}"] = float(raw[order[:count]].sum())
        rows.append(row)
    return rows


def _select_alpha(sweep: dict[str, dict[str, float]]) -> str:
    baseline = sweep["1.00"]
    feasible = []
    for alpha, values in sweep.items():
        if (
            values["kld_sph"] <= baseline["kld_sph"]
            and values["sim_sph"] >= baseline["sim_sph"]
            and values["cc_sph"] >= baseline["cc_sph"]
            and all(
                values[f"coverage_{budget:g}"]
                >= baseline[f"coverage_{budget:g}"]
                for budget in (5.0, 10.0, 20.0)
            )
        ):
            coverage_gain = np.mean(
                [
                    values[f"coverage_{budget:g}"] - baseline[f"coverage_{budget:g}"]
                    for budget in (5.0, 10.0, 20.0)
                ]
            )
            density_gain = (
                baseline["kld_sph"] - values["kld_sph"]
                + values["sim_sph"] - baseline["sim_sph"]
                + values["cc_sph"] - baseline["cc_sph"]
            )
            feasible.append((float(coverage_gain), float(density_gain), float(alpha), alpha))
    return max(feasible, default=(0.0, 0.0, 1.0, "1.00"))[-1]


def _select_early_alpha_vector(
    sweeps: dict[str, dict[str, dict[str, float]]]
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    horizons = ("0.5", "1", "3")
    metrics = ("kld_sph", "sim_sph", "cc_sph", "coverage_5", "coverage_10", "coverage_20")

    def macro(keys: tuple[str, str, str]) -> dict[str, float]:
        return {
            metric: float(
                np.mean([sweeps[horizon][alpha][metric] for horizon, alpha in zip(horizons, keys)])
            )
            for metric in metrics
        }

    baseline_keys = ("1.00", "1.00", "1.00")
    baseline = macro(baseline_keys)
    best = (float("-inf"), float("-inf"), baseline_keys)
    # The 0.5-s endpoint is reset-dominated and already saturated; adapting it
    # creates cutoff swaps without application headroom.  The attachable prior
    # is therefore activated only after the immediate-reset horizon.
    for keys in itertools.product(("1.00",), sweeps["1"], sweeps["3"]):
        values = macro(keys)
        gains = [
            baseline["kld_sph"] - values["kld_sph"],
            values["sim_sph"] - baseline["sim_sph"],
            values["cc_sph"] - baseline["cc_sph"],
            values["coverage_5"] - baseline["coverage_5"],
            values["coverage_10"] - baseline["coverage_10"],
            values["coverage_20"] - baseline["coverage_20"],
        ]
        if min(gains[:3]) < -1e-12 or min(gains[3:]) < -2e-4:
            continue
        normalized = [
            gains[0] / 0.05,
            gains[1] / 0.01,
            gains[2] / 0.01,
            gains[3] / 0.01,
            gains[4] / 0.01,
            gains[5] / 0.01,
        ]
        candidate = (float(np.mean(normalized)), float(min(normalized)), keys)
        if candidate > best:
            best = candidate
    selected = {horizon: float(alpha) for horizon, alpha in zip(horizons, best[2])}
    return selected, {"baseline": baseline, "adapted": macro(best[2])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only start-prior adapter for TempSAL-ERP.")
    parser.add_argument(
        "--checkpoint", default="runs/kernel_v2_tempsal_erp_seed20260711/checkpoints/best.pt"
    )
    parser.add_argument(
        "--processed-root", default="data/processed/early_horizon_salient360_spherical2deg_v2"
    )
    parser.add_argument("--report-root", default="reports/v11g_tempsal_prior_adapter")
    parser.add_argument("--evaluation-split", choices=("val", "gate"), default="val")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    seed_everything(20260805)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model("semantic_tempsal_temporal", checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    train = _collect(
        model, EarlyHorizonDataset(args.processed_root, split="train"), device
    )
    evaluation = _collect(
        model, EarlyHorizonDataset(args.processed_root, split=args.evaluation_split), device
    )
    alphas = np.linspace(0.0, 1.0, 21)
    train_sweeps = {}
    for horizon_index, horizon in enumerate((0.5, 1.0, 3.0, 5.0)):
        sweep = {
            f"{alpha:.2f}": _evaluate_alpha(
                train, horizon_index=horizon_index, alpha=float(alpha)
            )
            for alpha in alphas
        }
        train_sweeps[f"{horizon:g}"] = sweep
    selected, train_early_macro = _select_early_alpha_vector(train_sweeps)
    selected["5"] = float(_select_alpha(train_sweeps["5"]))
    val_results = {}
    for horizon_index, horizon in enumerate((0.5, 1.0, 3.0, 5.0)):
        alpha = selected[f"{horizon:g}"]
        val_results[f"{horizon:g}"] = {
            "adapted": _evaluate_alpha(evaluation, horizon_index=horizon_index, alpha=float(alpha)),
            "tempsal": _evaluate_alpha(evaluation, horizon_index=horizon_index, alpha=1.0),
        }
    metrics = ("kld_sph", "sim_sph", "cc_sph", "coverage_5", "coverage_10", "coverage_20")
    early = ("0.5", "1", "3")
    early_macro = {
        name: {
            metric: float(np.mean([val_results[horizon][name][metric] for horizon in early]))
            for metric in metrics
        }
        for name in ("adapted", "tempsal")
    }
    paired_rows = []
    for horizon_index, horizon in enumerate((0.5, 1.0, 3.0, 5.0)):
        adapted_rows = _evaluate_alpha_rows(
            evaluation, horizon_index=horizon_index, alpha=selected[f"{horizon:g}"]
        )
        tempsal_rows = _evaluate_alpha_rows(
            evaluation, horizon_index=horizon_index, alpha=1.0
        )
        for adapted_row, tempsal_row in zip(adapted_rows, tempsal_rows):
            if adapted_row["content_id"] != tempsal_row["content_id"]:
                raise RuntimeError("paired content order mismatch")
            paired_rows.append(
                {
                    "content_id": adapted_row["content_id"],
                    "horizon_s": float(horizon),
                    "adapted": adapted_row,
                    "tempsal": tempsal_row,
                }
            )
    result = {
        "protocol": {
            "alpha_selection_split": "train",
            "evaluation_split": args.evaluation_split,
            "forbidden_splits": ["test", "sitzmann"],
            "alpha_grid": [float(value) for value in alphas],
        },
        "selected_alphas": selected,
        "train_early_macro_selection": train_early_macro,
        "early_macro": early_macro,
        "per_horizon": val_results,
        "paired_rows": paired_rows,
        "train_sweeps": train_sweeps,
    }
    out = ensure_dir(args.report_root)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# V11-G TempSAL-ERP Start-Prior Adapter",
        "",
        f"No neural training. Mixing coefficients were selected on Salient360 train only; `{args.evaluation_split}` was untouched until evaluation.",
        "",
        f"Selected alphas: `{selected}` (1.0 is the unchanged TempSAL-ERP).",
        "",
        f"## {args.evaluation_split} early macro (0.5/1/3 s)",
        "",
        "| Method | KLD | SIM | CC | Cov@5 | Cov@10 | Cov@20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("tempsal", "adapted"):
        values = early_macro[name]
        lines.append(
            f"| {name} | {values['kld_sph']:.4f} | {values['sim_sph']:.4f} | "
            f"{values['cc_sph']:.4f} | {values['coverage_5']:.4f} | "
            f"{values['coverage_10']:.4f} | {values['coverage_20']:.4f} |"
        )
    lines.extend(["", f"## {args.evaluation_split} per horizon", "", "| t | alpha | method | KLD | SIM | CC | Cov@5 | Cov@10 | Cov@20 |", "|---:|---:|---|---:|---:|---:|---:|---:|---:|"])
    for horizon, values in val_results.items():
        for name in ("tempsal", "adapted"):
            row = values[name]
            lines.append(
                f"| {horizon} | {selected[horizon]:.2f} | {name} | {row['kld_sph']:.4f} | "
                f"{row['sim_sph']:.4f} | {row['cc_sph']:.4f} | {row['coverage_5']:.4f} | "
                f"{row['coverage_10']:.4f} | {row['coverage_20']:.4f} |"
            )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "summary.md")


if __name__ == "__main__":
    main()
