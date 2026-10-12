#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.baselines.scanpath_time import second_sampled_scanpaths_to_cumulative_maps
from sphererate.baselines.volume_time import temporal_volume_to_cumulative_maps
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.metrics import kld
from sphererate.utils import ensure_dir
from scripts.evaluate_eh3_gate import _predict_learned
from scripts.evaluate_eh6_panoramic_static import _predict as _predict_panoramic
from scripts.evaluate_eh6_scandmm import _average_rows, _score


def _load_external_rows(path: str) -> dict[str, Any]:
    current = json.loads(Path(path).read_text())
    return {
        "scores": current["score_rows"],
        "coverage": current["coverage_rows"],
        "tier": current["tier"],
    }


def main() -> None:
    output = ensure_dir("reports/eh6_visualizations/data")
    dataset = EarlyHorizonDataset(
        "data/processed/early_horizon_salient360_v1",
        split="test",
        target_mode="correct",
        allow_outer_test=True,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    budgets = (5.0, 10.0, 20.0)
    device = torch.device("cuda")

    candidate_scores_by_seed = []
    candidate_coverage_by_budget = {str(value): [] for value in budgets}
    candidate_seed11: dict[str, np.ndarray] | None = None
    for seed in (20260710, 20260711, 20260712):
        predictions, _ = _predict_learned(
            Path(f"runs/eh3_candidate_correct_seed{seed}/checkpoints/best.pt"), samples, device
        )
        if seed == 20260711:
            candidate_seed11 = predictions
        scores, coverage = _score(predictions, samples, budgets)
        candidate_scores_by_seed.append(scores)
        for budget in budgets:
            candidate_coverage_by_budget[str(budget)].append(coverage[str(budget)])
    assert candidate_seed11 is not None
    candidate_scores = _average_rows(candidate_scores_by_seed, ("kld_sph", "sim_sph", "cc_sph"))
    candidate_coverage = {
        budget: _average_rows(rows, ("coverage",)) for budget, rows in candidate_coverage_by_budget.items()
    }

    template_predictions = {str(sample["image_id"]): sample["template"].numpy() for sample in samples}
    template_scores, template_coverage = _score(template_predictions, samples, budgets)
    methods = {
        "SphereHorizon": {
            "scores": candidate_scores,
            "coverage": candidate_coverage,
            "tier": "proposed",
        },
        "Early template": {
            "scores": template_scores,
            "coverage": template_coverage,
            "tier": "internal_nonlearned",
        },
        "ScanDMM": _load_external_rows("reports/eh6_scandmm_declaredstart_test/results.json"),
        "Panoramic-CNN": _load_external_rows("reports/eh6_panoramic_static_test/results.json"),
        "SaltiNet": _load_external_rows("reports/eh6_saltinet_pretrained_test/results.json"),
    }
    (output / "metric_rows.json").write_text(
        json.dumps({"methods": methods, "horizons_s": [0.5, 1.0, 3.0, 5.0]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    scan_predictions = {}
    scan_root = Path("runs/eh6_scandmm_declaredstart_seed20260711_test_200/scanpaths")
    for sample in samples:
        image_id = str(sample["image_id"])
        scan_predictions[image_id] = second_sampled_scanpaths_to_cumulative_maps(
            np.load(scan_root / f"{image_id}.npy"),
            horizons_s=sample["horizons_s"].numpy().tolist(),
            height=int(sample["target"].shape[-2]),
            width=int(sample["target"].shape[-1]),
            sigma_px=10.0,
            coordinate_order="yx",
        )
    panoramic_predictions = _predict_panoramic(
        Path("runs/eh6_panoramic_static_seed20260711/checkpoints/best.pt"), samples, device
    )
    saltinet_predictions = {}
    saltinet_root = Path("runs/eh6_saltinet_pretrained_test/volumes")
    for sample in samples:
        image_id = str(sample["image_id"])
        with np.load(saltinet_root / f"{image_id}.npz") as data:
            volume = np.asarray(data["volume"], dtype=np.float32)
        saltinet_predictions[image_id] = temporal_volume_to_cumulative_maps(
            volume,
            horizons_s=sample["horizons_s"].numpy().tolist(),
            duration_s=25.0,
            height=int(sample["target"].shape[-2]),
            width=int(sample["target"].shape[-1]),
        )

    ranking = []
    for sample in samples:
        image_id = str(sample["image_id"])
        target = sample["target"].numpy()
        area = sample["area_weights"].numpy()[0]
        values = []
        for index in range(3):
            values.append(
                kld(scan_predictions[image_id][index], target[index], area)
                - kld(candidate_seed11[image_id][index], target[index], area)
            )
        ranking.append((float(np.mean(values)), image_id))
    ranking.sort()
    selected = {
        "worst": ranking[0][1],
        "median": ranking[len(ranking) // 2][1],
        "best": ranking[-1][1],
    }
    selection_rows = []
    by_id = {str(sample["image_id"]): sample for sample in samples}
    for role, image_id in selected.items():
        sample = by_id[image_id]
        rgb = np.clip(sample["image"][:3].numpy().transpose(1, 2, 0), 0.0, 1.0)
        np.savez_compressed(
            output / f"{role}_{image_id}.npz",
            rgb=rgb.astype(np.float32),
            gt=sample["target"].numpy().astype(np.float32),
            template=template_predictions[image_id].astype(np.float32),
            spherehorizon=candidate_seed11[image_id].astype(np.float32),
            scandmm=scan_predictions[image_id].astype(np.float32),
            panoramic=panoramic_predictions[image_id].astype(np.float32),
            saltinet=saltinet_predictions[image_id].astype(np.float32),
            horizons_s=sample["horizons_s"].numpy().astype(np.float32),
        )
        improvement = next(value for value, current in ranking if current == image_id)
        selection_rows.append({"role": role, "image_id": image_id, "sphere_vs_scandmm_kld_improvement": improvement})
    (output / "selection.json").write_text(
        json.dumps(selection_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "selected": selection_rows}, indent=2))


if __name__ == "__main__":
    main()
