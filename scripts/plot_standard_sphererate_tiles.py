#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.aggregate_v12c_dwell_rate import _model_from_checkpoint
from sphererate.baselines.scanpath_time import second_sampled_scanpaths_to_cumulative_maps
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.tile_prefetch import gaze_mass_coverage, rank_tiles
from sphererate.models.registry import build_model


DATA_ROOT = ROOT / "data/processed/early_horizon_salient360_spherical2deg_standard_v3"
RUNS_ROOT = ROOT / "runs/standard_split_v3"
PRIMARY_ROWS = ROOT / "reports/standard_split_primary_test/per_content.csv"
OUTPUT = ROOT / "reports/standard_split_tile_decisions"
PAPER_FIGURE = ROOT / "paper/figures/sphererate_tile_decisions_3s.pdf"
QUAL_SEED = 20260711
HORIZON_INDEX = 2
HORIZON_S = 3.0
TILE_ROWS = 8
TILE_COLS = 16
BUDGET = 10.0
METHODS = ("GT ranking", "Early template", "TempSAL-ERP", "SphereRate", "ScanDMM", "HAT")
# Column titles only.  The first column is the in-sample oracle and is labelled "GT" on the
# figure; the "GT ranking" key is kept everywhere else (map dict, results.json) unchanged.
TITLES = {"GT ranking": "GT"}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.4,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 500,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def _example_ids() -> list[tuple[str, str, float]]:
    buckets: dict[tuple[str, str], list[float]] = {}
    with PRIMARY_ROWS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                float(row["horizon_s"]) != HORIZON_S
                or row["method"] not in {"v12c", "tempsal_erp"}
                or int(row["seed"]) != QUAL_SEED
            ):
                continue
            buckets.setdefault((row["method"], row["image_id"]), []).append(float(row["coverage_10"]))
    ids = sorted({image_id for _, image_id in buckets})
    differences = [
        (
            image_id,
            float(np.mean(buckets[("v12c", image_id)]))
            - float(np.mean(buckets[("tempsal_erp", image_id)])),
        )
        for image_id in ids
    ]
    ordered = sorted(differences, key=lambda item: item[1])
    best = ordered[-1]
    failure = ordered[0]
    median = min(ordered, key=lambda item: abs(item[1]))
    return [(best[0], "largest gain", best[1]), (median[0], "near tie", median[1]), (failure[0], "failure", failure[1])]


def _tile_rect(tile_id: int, height: int, width: int) -> tuple[float, float, float, float]:
    row, col = divmod(tile_id, TILE_COLS)
    return col * width / TILE_COLS, row * height / TILE_ROWS, width / TILE_COLS, height / TILE_ROWS


def _draw(
    axis: plt.Axes,
    rgb: np.ndarray,
    density: np.ndarray,
    selected: set[int],
    oracle: set[int],
    coverage: float,
    start_xy: tuple[float, float],
) -> None:
    luminance = np.tensordot(rgb[..., :3], np.asarray([0.2126, 0.7152, 0.0722]), axes=1)
    muted = np.clip(0.18 * rgb + 0.82 * luminance[..., None], 0.0, 1.0)
    muted = np.clip(0.55 * muted + 0.45, 0.0, 1.0)
    low, high = np.quantile(density, [0.01, 0.995])
    heat = np.clip((density - low) / max(high - low, 1e-12), 0.0, 1.0)
    axis.imshow(muted, aspect="equal")
    axis.imshow(heat, cmap="turbo", vmin=0.0, vmax=1.0, alpha=0.23, aspect="equal")
    height, width = density.shape
    for tile_id in sorted(oracle):
        x, y, w, h = _tile_rect(tile_id, height, width)
        axis.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#D62728", linestyle=(0, (2, 1.4)), linewidth=0.8))
    for tile_id in sorted(selected):
        x, y, w, h = _tile_rect(tile_id, height, width)
        color = "#0072B2" if tile_id in oracle else "#E69F00"
        axis.add_patch(Rectangle((x, y), w, h, facecolor=mpl.colors.to_rgba(color, 0.34), edgecolor=color, linewidth=0.9))
    for row in range(1, TILE_ROWS):
        axis.axhline(row * height / TILE_ROWS, color="white", linewidth=0.28, alpha=0.95)
    for col in range(1, TILE_COLS):
        axis.axvline(col * width / TILE_COLS, color="white", linewidth=0.28, alpha=0.95)
    axis.scatter([start_xy[0]], [start_xy[1]], marker="+", s=14, c="#00D5FF", linewidths=1.1)
    axis.text(
        0.014,
        0.035,
        f"Cov. {coverage:.3f} | hit {len(selected & oracle)}/13",
        transform=axis.transAxes,
        fontsize=4.8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.58, "pad": 1.0, "edgecolor": "none"},
    )
    axis.set_box_aspect(0.5)
    axis.set_xticks([])
    axis.set_yticks([])


def _scanpath_map(run: str, image_id: str, sample: dict[str, Any]) -> np.ndarray:
    paths = np.load(RUNS_ROOT / run / "scanpaths" / f"{image_id}.npy")
    return second_sampled_scanpaths_to_cumulative_maps(
        paths,
        horizons_s=sample["horizons_s"].numpy().tolist(),
        height=int(sample["target"].shape[-2]),
        width=int(sample["target"].shape[-1]),
        renderer="sphere_chord",
        sigma_deg=2.0,
        coordinate_order="yx",
    )[HORIZON_INDEX]


def _coverage(sample: dict[str, Any], selected: set[int]) -> float:
    with np.load(str(sample["fixations_path"])) as data:
        fixations = np.asarray(data["fixations"], dtype=np.float32)
    return gaze_mass_coverage(
        fixations,
        observer_ids=np.unique(fixations[:, 3]).astype(int).tolist(),
        horizon_s=HORIZON_S,
        selected_tile_ids=selected,
        tile_rows=TILE_ROWS,
        tile_cols=TILE_COLS,
    )


@torch.no_grad()
def main() -> None:
    _style()
    examples = _example_ids()
    dataset = EarlyHorizonDataset(DATA_ROOT, split="test")
    samples = {str(item["image_id"]): item for item in dataset}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v12_state = torch.load(
        RUNS_ROOT / f"v12c_hybrid_dwell_rate_seed{QUAL_SEED}/checkpoints/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    v12_model = _model_from_checkpoint(v12_state, device)
    temp_state = torch.load(
        RUNS_ROOT / f"tempsal_erp_seed{QUAL_SEED}/checkpoints/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    tempsal_model = build_model("semantic_tempsal_temporal", temp_state["config"]).to(device)
    tempsal_model.load_state_dict(temp_state["model"])
    tempsal_model.eval()

    figure, axes = plt.subplots(3, len(METHODS), figsize=(7.10, 2.72), squeeze=False)
    # No row labels are drawn, so the figure needs almost no left margin.  The band between
    # top and bottom is sized to 3 panels + 2 x 6 pt gaps: hspace=0 drops the slot padding and
    # set_box_aspect centres each 37.57 pt panel in its slot, leaving exactly the two 6 pt gaps.
    figure.subplots_adjust(left=0.004, right=0.995, top=0.875, bottom=0.2076, wspace=0.045, hspace=0.0)
    records: list[dict[str, Any]] = []
    for row_index, (image_id, role, delta) in enumerate(examples):
        sample = samples[image_id]
        image = sample["image"].unsqueeze(0).to(device)
        template_input = sample["template"].unsqueeze(0).to(device)
        start_input = sample["start_geometry"].unsqueeze(0).to(device)
        sphere = v12_model(image, template_input, start_geometry=start_input)["cumulative"][
            0, HORIZON_INDEX
        ].float().cpu().numpy()
        tempsal = tempsal_model(image)["cumulative"][0, HORIZON_INDEX].float().cpu().numpy()
        maps = {
            "GT ranking": sample["target"][HORIZON_INDEX].numpy(),
            "Early template": sample["template"][HORIZON_INDEX].numpy(),
            "TempSAL-ERP": tempsal,
            "SphereRate": sphere,
            "ScanDMM": _scanpath_map(f"scandmm_declared_start_seed20260711_test", image_id, sample),
            "HAT": _scanpath_map(f"hat_seed20260711_test", image_id, sample),
        }
        area = sample["area_weights"][0].numpy()
        selections = {
            method: set(rank_tiles(values, area, tile_rows=TILE_ROWS, tile_cols=TILE_COLS, budget_percent=BUDGET).tolist())
            for method, values in maps.items()
        }
        oracle = selections["GT ranking"]
        rgb = np.clip(sample["image"][:3].numpy().transpose(1, 2, 0), 0.0, 1.0)
        geometry = sample["start_geometry"][1].numpy()
        start_y, start_x = np.unravel_index(np.argmax(geometry), geometry.shape)
        for col_index, method in enumerate(METHODS):
            coverage = _coverage(sample, selections[method])
            _draw(
                axes[row_index, col_index],
                rgb,
                maps[method],
                selections[method],
                oracle,
                coverage,
                (float(start_x), float(start_y)),
            )
            records.append(
                {
                    "image_id": image_id,
                    "role": role,
                    "method": method,
                    "coverage": coverage,
                    "oracle_overlap": len(selections[method] & oracle),
                }
            )
            if row_index == 0:
                axes[row_index, col_index].set_title(
                    TITLES.get(method, method),
                    fontsize=6.2,
                    pad=2.0,
                    fontweight="bold" if method in {"GT ranking", "SphereRate"} else "normal",
                    color="#1769AA" if method == "SphereRate" else "black",
                )
            # The per-row label (image_id / role / delta) is intentionally not drawn;
            # it stays available in results.json under "examples".
            if method == "SphereRate":
                for spine in axes[row_index, col_index].spines.values():
                    spine.set_color("#1769AA")
                    spine.set_linewidth(0.9)

    figure.text(
        0.50,
        0.182,
        "dashed red: GT top-10%  |  blue: selected GT tile  |  orange: selected outside GT",
        ha="center",
        fontsize=5.6,
        color="#4F5963",
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(OUTPUT / f"sphererate_tile_decisions_3s.{suffix}", bbox_inches="tight", pad_inches=0.02)
    figure.savefig(PAPER_FIGURE, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    result = {
        "split": "standard_test",
        "horizon_s": HORIZON_S,
        "tile_grid": [TILE_ROWS, TILE_COLS],
        "budget_percent": BUDGET,
        "examples": [dict(image_id=image_id, role=role, coverage_delta_vs_tempsal=delta) for image_id, role, delta in examples],
        "example_rule": "largest, closest-to-zero, and smallest SphereRate-minus-TempSAL Cov@10 difference at 3 s",
        "qualitative_model_seed": QUAL_SEED,
        "scanpath_seed": QUAL_SEED,
        "scanpaths_per_content": 200,
        "records": records,
    }
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "index.md").write_text(
        "# Standard-test 3-s tile decisions\n\n"
        "![Tile decisions](sphererate_tile_decisions_3s.png)\n\n"
        "Rows are selected by a deterministic best/tie/failure rule from all 13 test contents.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
