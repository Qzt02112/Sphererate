from __future__ import annotations

import json

import numpy as np
import torch

from sphererate.evaluation.process import evaluate_process_prediction_dir
from sphererate.train import validate


class _FixedProcessModel(torch.nn.Module):
    def __init__(self, temporal: torch.Tensor, cumulative: torch.Tensor, final: torch.Tensor) -> None:
        super().__init__()
        self._temporal = temporal
        self._cumulative = cumulative
        self._final = final

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = image.shape[0]
        return {
            "temporal": self._temporal.repeat(batch_size, 1, 1, 1),
            "cumulative": self._cumulative.repeat(batch_size, 1, 1, 1),
            "final": self._final.repeat(batch_size, 1, 1, 1),
        }


def test_process_evaluation_tracks_temporal_variation(tmp_path):
    processed = tmp_path / "processed"
    targets = processed / "targets"
    preds = tmp_path / "preds"
    reports = tmp_path / "reports"
    targets.mkdir(parents=True)
    preds.mkdir()

    area = np.ones((4, 4), dtype=np.float32)
    temporal_gt = np.zeros((2, 4, 4), dtype=np.float32)
    temporal_gt[0, 1, 1] = 1.0
    temporal_gt[1, 2, 2] = 1.0
    cumulative_gt = np.stack([temporal_gt[0], temporal_gt.sum(axis=0)], axis=0)

    temporal_pred = np.repeat(temporal_gt[:1], 2, axis=0)
    cumulative_pred = np.repeat(cumulative_gt[:1], 2, axis=0)

    target_path = targets / "P1.npz"
    np.savez(
        target_path,
        temporal=temporal_gt,
        cumulative=cumulative_gt,
        final=temporal_gt[-1],
        fixation=(temporal_gt[-1] > 0).astype(np.float32),
        area_weights=area,
    )
    np.savez(preds / "P1.npz", final=temporal_pred[-1], temporal=temporal_pred, cumulative=cumulative_pred)

    with open(processed / "manifest.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"image_id": "P1", "split": "val", "target": str(target_path)}) + "\n")

    summary = evaluate_process_prediction_dir(processed, preds, split="val", output_dir=reports)

    scores = summary["scores"]
    assert scores["temporal_step_tv_gt"] > 0.0
    assert scores["temporal_step_tv_pred"] == 0.0
    assert scores["temporal_step_tv_abs_gap"] > 0.0
    assert (reports / "val_summary.json").exists()
    assert (reports / "val_per_image.csv").exists()


def test_validation_can_report_process_metrics():
    area = torch.ones(1, 1, 4, 4)
    temporal_gt = torch.zeros(1, 2, 4, 4)
    temporal_gt[:, 0, 1, 1] = 1.0
    temporal_gt[:, 1, 2, 2] = 1.0
    cumulative_gt = torch.stack([temporal_gt[:, 0], temporal_gt.sum(dim=1)], dim=1)
    final_gt = temporal_gt[:, 1:2]
    batch = {
        "image": torch.zeros(1, 8, 4, 4),
        "temporal": temporal_gt,
        "cumulative": cumulative_gt,
        "final": final_gt,
        "fixation": final_gt,
        "area_weights": area,
        "image_id": ["P1"],
        "dataset": ["demo"],
    }
    model = _FixedProcessModel(
        temporal_gt[:, :1].repeat(1, 2, 1, 1),
        cumulative_gt[:, :1].repeat(1, 2, 1, 1),
        final_gt,
    )

    summary = validate(model, [batch], torch.device("cpu"), {"train": {"validate_process_metrics": True}})

    assert summary["process/temporal_step_tv_gt"] > 0.0
    assert summary["process/temporal_step_tv_pred"] == 0.0
    assert summary["macro_process_temporal_kld_sph"] >= 0.0
    assert summary["dataset/demo/process_temporal_step_tv_abs_gap"] > 0.0
