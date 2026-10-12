#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from sphererate.models.early_horizon import TemplateTimeGatedResidualNet
from sphererate.utils import ensure_dir, seed_everything


def _model(checkpoint: dict[str, Any]) -> TemplateTimeGatedResidualNet:
    config = checkpoint["config"]
    cfg = config["model"]
    model = TemplateTimeGatedResidualNet(
        in_channels=int(cfg["in_channels"]),
        base_channels=int(cfg["base_channels"]),
        hidden_channels=int(cfg["hidden_channels"]),
        horizons_s=config["data"]["horizons_s"],
        conditioning_mode=str(checkpoint.get("run", {}).get("conditioning_mode", "time")),
        semantic_backbone=str(cfg["semantic_backbone"]),
        semantic_layer=str(cfg["semantic_layer"]),
        semantic_pretrained=False,
        residual_fusion=str(cfg.get("residual_fusion", "semantic_anchor")),
        semantic_alpha_init=cfg.get("semantic_alpha_init"),
        tile_rows=int(cfg.get("tile_rows", 8)),
        tile_cols=int(cfg.get("tile_cols", 16)),
        tile_hidden_channels=int(cfg.get("tile_hidden_channels", 16)),
        tile_residual_scale=float(cfg.get("tile_residual_scale", 0.5)),
    )
    model.load_state_dict(checkpoint["model"])
    return model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep the 5-s template-to-semantic mixing alpha.")
    parser.add_argument(
        "--checkpoint", default="runs/v11a_raw_dwell_s2_seed20260711/checkpoints/best.pt"
    )
    parser.add_argument("--processed-root", default="data/processed/early_horizon_salient360_spherical2deg_v2")
    parser.add_argument("--report-root", default="reports/v11c_alpha_sweep")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(20260805)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _model(checkpoint).to(device).eval()
    dataset = EarlyHorizonDataset(args.processed_root, split="val", tile_rows=8, tile_cols=16)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
    alphas = np.linspace(0.0, 1.0, 21)
    rows: dict[str, list[dict[str, float]]] = {f"{alpha:.2f}": [] for alpha in alphas}
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        template = batch["template"].to(device, non_blocking=True)[:, 3:4]
        target = batch["target"].to(device, non_blocking=True)[:, 3:4]
        area = batch["area_weights"].to(device, non_blocking=True)
        semantic = model(image, batch["template"].to(device, non_blocking=True))["semantic_map"]
        raw_target = batch["raw_dwell_tiles"][:, 3].numpy()
        for alpha in alphas:
            logits = (1.0 - float(alpha)) * template.clamp_min(1e-8).log()
            logits = logits + float(alpha) * semantic.clamp_min(1e-8).log()
            prediction = model._spherical_softmax(logits)
            prediction_np = prediction.detach().float().cpu().numpy()
            area_np = area.detach().float().cpu().numpy()
            for sample_index in range(prediction.shape[0]):
                pred_h = prediction[sample_index : sample_index + 1]
                target_h = target[sample_index : sample_index + 1]
                area_h = area[sample_index : sample_index + 1]
                row = {
                    "kld_sph": float(weighted_kld(pred_h, target_h, area_h)),
                    "sim_sph": float(weighted_sim(pred_h, target_h, area_h)),
                    "cc_sph": float(weighted_cc(pred_h, target_h, area_h)),
                }
                mass = tile_probability_mass(
                    prediction_np[sample_index, 0],
                    area_np[sample_index, 0],
                    tile_rows=8,
                    tile_cols=16,
                ).ravel()
                order = np.lexsort((np.arange(mass.size), -mass))
                for budget in (5.0, 10.0, 20.0):
                    count = budget_tile_count(8, 16, budget)
                    row[f"coverage_{budget:g}"] = float(
                        raw_target[sample_index].ravel()[order[:count]].sum()
                    )
                rows[f"{alpha:.2f}"].append(row)
    metrics = ("kld_sph", "sim_sph", "cc_sph", "coverage_5", "coverage_10", "coverage_20")
    aggregate = {
        alpha: {metric: float(np.mean([row[metric] for row in current])) for metric in metrics}
        for alpha, current in rows.items()
    }
    reference = {
        "kld_sph": 1.2125685038389984,
        "sim_sph": 0.4184489437827358,
        "cc_sph": 0.4417222020340404,
        "coverage_5": 0.3251118997329180,
        "coverage_10": 0.4677827616099049,
        "coverage_20": 0.6670177291374980,
    }
    for values in aggregate.values():
        values["worst_coverage_margin"] = min(
            values[f"coverage_{budget:g}"] - reference[f"coverage_{budget:g}"]
            for budget in (5.0, 10.0, 20.0)
        )
        values["density_all_better"] = bool(
            values["kld_sph"] < reference["kld_sph"]
            and values["sim_sph"] > reference["sim_sph"]
            and values["cc_sph"] > reference["cc_sph"]
        )
    best_alpha = max(aggregate, key=lambda key: aggregate[key]["worst_coverage_margin"])
    result = {
        "protocol": {
            "split": "val",
            "horizon_s": 5.0,
            "checkpoint": args.checkpoint,
            "forbidden_splits": ["gate", "test", "sitzmann"],
        },
        "reference": reference,
        "learned_alpha": float(model.semantic_alphas()[-1].detach().cpu()),
        "best_alpha_by_worst_coverage_margin": best_alpha,
        "sweep": aggregate,
    }
    out = ensure_dir(args.report_root)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# V11-C 5-s Template–Semantic Alpha Sweep",
        "",
        "No training; Salient360 validation only.",
        "",
        f"- learned alpha: `{result['learned_alpha']:.4f}`",
        f"- best alpha by worst Coverage margin: `{best_alpha}`",
        "",
        "| alpha | KLD | SIM | CC | Cov@5 | Cov@10 | Cov@20 | worst margin | all density better |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for alpha, values in aggregate.items():
        if alpha in {"0.00", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90", "1.00", best_alpha}:
            lines.append(
                f"| {alpha} | {values['kld_sph']:.4f} | {values['sim_sph']:.4f} | "
                f"{values['cc_sph']:.4f} | {values['coverage_5']:.4f} | "
                f"{values['coverage_10']:.4f} | {values['coverage_20']:.4f} | "
                f"{values['worst_coverage_margin']:+.4f} | "
                f"{'yes' if values['density_all_better'] else 'no'} |"
            )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "summary.md")


if __name__ == "__main__":
    main()
