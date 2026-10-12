#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.data.early_horizon import build_cumulative_density
from sphererate.evaluation.metrics import kld
from sphererate.utils import ensure_dir
from scripts.evaluate_eh3_gate import _predict_learned


HORIZON = 3.0
HORIZON_INDEX = 2
BLUE = "#1769AA"
ORANGE = "#D55E00"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.linewidth": 0.55,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _fixations(sample: dict) -> np.ndarray:
    with np.load(str(sample["fixations_path"])) as data:
        return np.asarray(data["fixations"], dtype=np.float32)


def _legacy_target(sample: dict) -> np.ndarray:
    fixations = _fixations(sample)
    observers = np.unique(fixations[:, 3]).astype(int).tolist()
    return build_cumulative_density(
        fixations,
        observer_ids=observers,
        horizon_s=HORIZON,
        height=256,
        width=512,
        sigma_px=10.0,
        renderer="legacy_erp",
    )


def _overlay_density(ax: plt.Axes, rgb: np.ndarray, density: np.ndarray, vmin: float, vmax: float) -> None:
    ax.imshow(rgb, aspect="equal")
    log_density = np.log10(np.maximum(density, 1e-8))
    alpha = np.clip((log_density - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    ax.imshow(log_density, cmap="magma", vmin=vmin, vmax=vmax, alpha=0.15 + 0.65 * alpha, aspect="equal")
    ax.set_box_aspect(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a kernel-v2 gate-only content-attribution diagnostic.")
    parser.add_argument("--output-dir", default="reports/kernel_v2_content_audit")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    _style()
    output = ensure_dir(args.output_dir)
    dataset = EarlyHorizonDataset(
        "data/processed/early_horizon_salient360_spherical2deg_v2",
        split="gate",
        target_mode="correct",
    )
    samples = [dataset[index] for index in range(len(dataset))]
    checkpoint = Path("runs/kernel_v2_candidate_correct_seed20260711/checkpoints/best.pt")
    device = torch.device(args.device)
    correct, provenance = _predict_learned(checkpoint, samples, device)
    wrong, _ = _predict_learned(checkpoint, samples, device, counterfactual="wrong_image")
    records = []
    for sample in samples:
        image_id = str(sample["image_id"])
        area = sample["area_weights"].numpy()[0]
        target = sample["target"].numpy()[HORIZON_INDEX]
        template = sample["template"].numpy()[HORIZON_INDEX]
        prediction = correct[image_id][HORIZON_INDEX]
        wrong_prediction = wrong[image_id][HORIZON_INDEX]
        records.append(
            {
                "image_id": image_id,
                "gain_vs_template": kld(template, target, area) - kld(prediction, target, area),
                "gain_vs_wrong": kld(wrong_prediction, target, area) - kld(prediction, target, area),
            }
        )
    ordered = sorted(records, key=lambda row: (row["gain_vs_wrong"], row["image_id"]))
    selected_records = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    roles = ("weakest", "median", "strongest")
    by_id = {str(sample["image_id"]): sample for sample in samples}
    selected = []
    density_values = []
    for role, record in zip(roles, selected_records):
        sample = by_id[record["image_id"]]
        image_id = str(sample["image_id"])
        payload = {
            "role": role,
            "record": record,
            "sample": sample,
            "rgb": np.clip(sample["image"][:3].numpy().transpose(1, 2, 0), 0.0, 1.0),
            "legacy": _legacy_target(sample),
            "target": sample["target"].numpy()[HORIZON_INDEX],
            "template": sample["template"].numpy()[HORIZON_INDEX],
            "correct": correct[image_id][HORIZON_INDEX],
            "wrong": wrong[image_id][HORIZON_INDEX],
        }
        selected.append(payload)
        for key in ("legacy", "target", "template", "correct", "wrong"):
            density_values.append(np.log10(np.maximum(payload[key], 1e-8)).ravel())
    density_flat = np.concatenate(density_values)
    vmin = float(np.quantile(density_flat, 0.03))
    vmax = float(np.quantile(density_flat, 0.995))
    residuals = [np.log(item["correct"] + 1e-8) - np.log(item["template"] + 1e-8) for item in selected]
    residual_limit = float(np.quantile(np.abs(np.concatenate([value.ravel() for value in residuals])), 0.985))

    fig = plt.figure(figsize=(11.4, 4.25), constrained_layout=True)
    grid = fig.add_gridspec(3, 7)
    titles = (
        "ERP + observed gaze",
        "Legacy GT\nfixed 10 px",
        "Corrected GT\nspherical 2°",
        "Train-only template",
        "Correct-image pred.",
        "Wrong-image pred.",
        "Content residual\nvs template",
    )
    heat_axes, residual_axes = [], []
    heat_image = residual_image = None
    for row_index, (payload, residual) in enumerate(zip(selected, residuals)):
        fixations = _fixations(payload["sample"])
        starts = fixations[:, 2]
        current = fixations[starts < HORIZON]
        for col_index, title in enumerate(titles):
            ax = fig.add_subplot(grid[row_index, col_index])
            if row_index == 0:
                ax.set_title(title, fontweight="bold" if col_index in {0, 2, 4, 6} else "normal")
            if col_index == 0:
                ax.imshow(payload["rgb"], aspect="equal")
                ax.set_box_aspect(0.5)
                if len(current):
                    index = np.linspace(0, len(current) - 1, min(220, len(current)), dtype=int)
                    ax.scatter(
                        np.mod(current[index, 0], 1.0) * 512,
                        np.clip(current[index, 1], 0.0, 1.0) * 256,
                        s=2.2,
                        c="white",
                        edgecolors="black",
                        linewidths=0.15,
                        alpha=0.62,
                    )
                ax.axvline(0.75 * 512, color="#00B4D8", linestyle="--", linewidth=0.75)
                ax.set_ylabel(
                    f"{payload['role']}\n{payload['record']['image_id']}\n"
                    f"ΔKLDwrong={payload['record']['gain_vs_wrong']:+.3f}",
                    rotation=0,
                    labelpad=31,
                    va="center",
                    fontweight="bold",
                )
            elif col_index in {1, 2, 3}:
                key = {1: "legacy", 2: "target", 3: "template"}[col_index]
                heat_image = ax.imshow(
                    np.log10(np.maximum(payload[key], 1e-8)),
                    cmap="magma",
                    vmin=vmin,
                    vmax=vmax,
                    aspect="equal",
                )
                ax.set_box_aspect(0.5)
                heat_axes.append(ax)
            elif col_index in {4, 5}:
                _overlay_density(ax, payload["rgb"], payload["correct" if col_index == 4 else "wrong"], vmin, vmax)
                if col_index == 4:
                    for spine in ax.spines.values():
                        spine.set_color(BLUE)
                        spine.set_linewidth(1.0)
            else:
                ax.imshow(payload["rgb"], aspect="equal")
                strength = np.clip(np.abs(residual) / max(residual_limit, 1e-8), 0.0, 1.0)
                residual_image = ax.imshow(
                    residual,
                    cmap="coolwarm",
                    norm=Normalize(-residual_limit, residual_limit),
                    alpha=0.12 + 0.72 * strength,
                    aspect="equal",
                )
                ax.set_box_aspect(0.5)
                residual_axes.append(ax)
            ax.set_xticks([])
            ax.set_yticks([])
    assert heat_image is not None and residual_image is not None
    heat_bar = fig.colorbar(heat_image, ax=heat_axes, location="bottom", shrink=0.48, pad=0.035, aspect=35)
    heat_bar.set_label("log10 spherical density (shared scale)")
    residual_bar = fig.colorbar(residual_image, ax=residual_axes, location="bottom", shrink=0.8, pad=0.035, aspect=18)
    residual_bar.set_label("signed log-density residual")
    fig.suptitle(
        "Correcting the kernel reveals scene-linked probability reallocation at 3 s",
        fontsize=9.4,
        fontweight="bold",
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"kernel_v2_content_attribution_3s.{suffix}", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "split": "historically_seen_gate",
        "outer_test_used": False,
        "horizon_s": HORIZON,
        "checkpoint": provenance,
        "selection": [dict(role=role, **record) for role, record in zip(roles, selected_records)],
        "figure": str(output / "kernel_v2_content_attribution_3s.png"),
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Kernel-v2 content attribution",
        "",
        "![Kernel-v2 content attribution](kernel_v2_content_attribution_3s.png)",
        "",
        "Gate-only diagnostic; no v2 test target was loaded. The dashed cyan line marks the declared canonical start longitude.",
        "",
        "Legacy 10 px targets visibly merge fixation clusters into a broad start-centred field. The corrected 2° spherical targets preserve separated scene-linked modes. Correct/wrong-image predictions and the signed residual show where RGB evidence reallocates probability relative to the train-only template.",
    ]
    (output / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"figure": summary["figure"], "index": str(output / "index.md")}, indent=2))


if __name__ == "__main__":
    main()
