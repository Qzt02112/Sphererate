#!/usr/bin/env python
"""Visualize real ET-DK2 multi-start behavior without training or tuning."""

from __future__ import annotations

import json
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.data.dataset import coord_channels
from sphererate.data.early_horizon import build_cumulative_density
from sphererate.evaluation.temporal_factorization import reference_alignment_pixel_shift
from scripts.evaluate_eh3_gate import _build_model_from_state


MANIFEST = ROOT / "data/processed/etdk2_salient360shared_256x512_k6_session_v1/manifest.jsonl"
CHECKPOINT = ROOT / "runs/kernel_v2_candidate_correct_seed20260711/checkpoints/best.pt"
TEMPLATE_ROOT = ROOT / "data/processed/early_horizon_salient360_spherical2deg_v2/baselines/early_template"
OUTPUT_ROOT = ROOT / "reports/kernel_v2_etdk2"
DATA_FILE = OUTPUT_ROOT / "eh4_multistart_data.npz"
ALL_STARTS = (-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0)
DISPLAY_STARTS = (-180.0, -90.0, 0.0, 90.0)
HORIZONS = (0.5, 1.0, 3.0, 5.0)
DISPLAY_HORIZONS = (1.0, 3.0)
CANONICAL_START = 90.0
HEIGHT, WIDTH = 256, 512


def _style() -> None:
    global mpl, plt
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.65,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _read_rows() -> list[dict]:
    rows = []
    with MANIFEST.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") == "etdk2" and row.get("split") == "val":
                rows.append(row)
    return rows


def _select_content(rows: list[dict]) -> tuple[str, list[dict]]:
    """Select transparently: first content ID with all eight declared starts."""
    by_content: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_content[str(row["canonical_content_id"])].append(row)
    eligible = [
        content
        for content, current in by_content.items()
        if set(float(row["start_erp_longitude_deg"]) for row in current) == set(ALL_STARTS)
    ]
    if not eligible:
        raise RuntimeError("No ET-DK2 validation content covers all eight starts")
    content = sorted(eligible)[0]
    return content, by_content[content]


def _load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _observed_maps(rows: list[dict]) -> tuple[dict[float, dict[float, np.ndarray]], dict[float, int]]:
    maps: dict[float, dict[float, np.ndarray]] = {}
    counts: dict[float, int] = {}
    for start in DISPLAY_STARTS:
        selected = [row for row in rows if float(row["start_erp_longitude_deg"]) == start]
        arrays = []
        for session_index, row in enumerate(selected):
            with np.load(row["fixations"]) as data:
                fixation = np.asarray(data["fixations"], dtype=np.float32).copy()
            fixation[:, 3] = float(session_index)
            arrays.append(fixation)
        combined = np.concatenate(arrays, axis=0)
        observer_ids = range(len(selected))
        maps[start] = {
            horizon: build_cumulative_density(
                combined,
                observer_ids=observer_ids,
                horizon_s=horizon,
                height=HEIGHT,
                width=WIDTH,
                renderer="sphere_chord",
                sigma_deg=2.0,
            )
            for horizon in DISPLAY_HORIZONS
        }
        counts[start] = len(selected)
    return maps, counts


@torch.no_grad()
def _predict(rgb: np.ndarray, template: np.ndarray) -> dict[float, dict[float, np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(CHECKPOINT, map_location=device)
    model = _build_model_from_state(state, device)
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    geometry = coord_channels(HEIGHT, WIDTH)
    template_tensor = torch.from_numpy(template).unsqueeze(0).to(device)
    output: dict[float, dict[float, np.ndarray]] = {}
    for start in DISPLAY_STARTS:
        shift = reference_alignment_pixel_shift(start, CANONICAL_START, WIDTH)
        aligned = torch.roll(rgb_tensor, shifts=shift, dims=-1)
        model_input = torch.cat([aligned, geometry], dim=0).unsqueeze(0).to(device)
        cumulative = model(model_input, template_tensor)["cumulative"][0]
        restored = torch.roll(cumulative, shifts=-shift, dims=-1).cpu().numpy().astype(np.float32)
        output[start] = {
            horizon: restored[HORIZONS.index(horizon)] for horizon in DISPLAY_HORIZONS
        }
    return output


def _start_x(start: float) -> float:
    return (((float(start) + 180.0) / 360.0) % 1.0) * WIDTH


def _draw_start(ax: plt.Axes, start: float, *, on_rgb: bool) -> None:
    x = _start_x(start)
    color = "#00A6D6"
    if on_rgb:
        ax.axvline(x, color=color, linewidth=1.15, linestyle="--", alpha=0.95)
        ax.scatter([x], [HEIGHT / 2], s=22, facecolor="white", edgecolor=color, linewidth=1.1, zorder=4)
    else:
        ax.scatter([x], [5], marker="v", s=18, facecolor=color, edgecolor="white", linewidth=0.35, zorder=4)


def _plot(
    content: str,
    rgb: np.ndarray,
    observed: dict[float, dict[float, np.ndarray]],
    predictions: dict[float, dict[float, np.ndarray]],
    counts: dict[float, int],
) -> Path:
    values = []
    for source in (observed, predictions):
        for by_horizon in source.values():
            for current in by_horizon.values():
                values.append(np.log10(np.maximum(current, 1e-8)).ravel())
    flat = np.concatenate(values)
    vmin = float(np.quantile(flat, 0.04))
    vmax = float(np.quantile(flat, 0.997))

    fig = plt.figure(figsize=(9.1, 4.65), constrained_layout=True)
    grid = fig.add_gridspec(4, 5, width_ratios=[1.08, 1, 1, 1, 1])
    titles = ("ERP + declared start", "Observed 1 s", "SphereHorizon 1 s", "Observed 3 s", "SphereHorizon 3 s")
    heat_axes: list[plt.Axes] = []
    last = None
    for row_index, start in enumerate(DISPLAY_STARTS):
        for col_index in range(5):
            ax = fig.add_subplot(grid[row_index, col_index])
            if row_index == 0:
                ax.set_title(titles[col_index], fontweight="bold" if col_index in {0, 2, 4} else "normal")
            if col_index == 0:
                ax.imshow(rgb, aspect="equal")
                ax.set_box_aspect(0.5)
                _draw_start(ax, start, on_rgb=True)
                ax.set_ylabel(f"yaw {start:+g}°\n(n={counts[start]})", rotation=0, labelpad=30, va="center", fontweight="bold")
            else:
                source = observed if col_index in {1, 3} else predictions
                horizon = 1.0 if col_index in {1, 2} else 3.0
                last = ax.imshow(
                    np.log10(np.maximum(source[start][horizon], 1e-8)),
                    cmap="magma",
                    vmin=vmin,
                    vmax=vmax,
                    aspect="equal",
                )
                ax.set_box_aspect(0.5)
                _draw_start(ax, start, on_rgb=False)
                heat_axes.append(ax)
                for spine in ax.spines.values():
                    spine.set_linewidth(1.0 if col_index in {2, 4} else 0.45)
                    spine.set_edgecolor("#1769AA" if col_index in {2, 4} else "#9CA3AF")
            ax.set_xticks([])
            ax.set_yticks([])

    assert last is not None
    colorbar = fig.colorbar(last, ax=heat_axes, location="bottom", shrink=0.52, pad=0.045, aspect=38)
    colorbar.set_label("log$_{10}$ spherical density (shared scale)")
    image_id = content.split(":", 1)[-1]
    fig.suptitle(
        f"Same panorama, different declared starts — ET-DK2 {image_id}",
        fontsize=9.4,
        fontweight="bold",
    )
    stem = OUTPUT_ROOT / f"eh4_multistart_{image_id}"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "prepare", "plot"), default="all")
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.mode in {"all", "prepare"}:
        rows = _read_rows()
        content, content_rows = _select_content(rows)
        source_id = str(content_rows[0]["source_image_id"])
        rgb = _load_rgb(content_rows[0]["image"])
        with np.load(TEMPLATE_ROOT / f"{source_id}.npz") as data:
            template = np.asarray(data["cumulative"], dtype=np.float32)
        observed, counts = _observed_maps(content_rows)
        predictions = _predict(rgb, template)
        np.savez_compressed(
            DATA_FILE,
            content_id=np.asarray(content),
            rgb=rgb.astype(np.float32),
            starts=np.asarray(DISPLAY_STARTS, dtype=np.float32),
            horizons=np.asarray(DISPLAY_HORIZONS, dtype=np.float32),
            counts=np.asarray([counts[start] for start in DISPLAY_STARTS], dtype=np.int32),
            observed=np.stack(
                [[observed[start][horizon] for horizon in DISPLAY_HORIZONS] for start in DISPLAY_STARTS]
            ).astype(np.float32),
            predictions=np.stack(
                [[predictions[start][horizon] for horizon in DISPLAY_HORIZONS] for start in DISPLAY_STARTS]
            ).astype(np.float32),
        )
    else:
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"Run --mode prepare first: {DATA_FILE}")
        with np.load(DATA_FILE) as data:
            content = str(data["content_id"].item())
            rgb = np.asarray(data["rgb"], dtype=np.float32)
            starts = [float(value) for value in data["starts"]]
            horizons = [float(value) for value in data["horizons"]]
            count_values = [int(value) for value in data["counts"]]
            observed_array = np.asarray(data["observed"], dtype=np.float32)
            prediction_array = np.asarray(data["predictions"], dtype=np.float32)
        counts = {start: count for start, count in zip(starts, count_values)}
        observed = {
            start: {horizon: observed_array[i, j] for j, horizon in enumerate(horizons)}
            for i, start in enumerate(starts)
        }
        predictions = {
            start: {horizon: prediction_array[i, j] for j, horizon in enumerate(horizons)}
            for i, start in enumerate(starts)
        }

    if args.mode == "prepare":
        print(json.dumps({"data": str(DATA_FILE), "training_performed": False}, indent=2))
        return

    _style()
    stem = _plot(content, rgb, observed, predictions, counts)
    audit = {
        "role": "EH4 multi-start mechanism visualization",
        "selection": "lexicographically first validation content covering all eight ET-DK2 starts",
        "content_id": content,
        "available_starts_deg": list(ALL_STARTS),
        "displayed_starts_deg": list(DISPLAY_STARTS),
        "displayed_horizons_s": list(DISPLAY_HORIZONS),
        "sessions_per_displayed_start": {str(k): v for k, v in counts.items()},
        "checkpoint_seed": 20260711,
        "training_performed": False,
        "warning": "Observed maps pool only the listed sessions and are descriptive, not stable population ground truth.",
    }
    (stem.with_name(stem.name + "_audit.json")).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"figure": str(stem.with_suffix('.png')), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
