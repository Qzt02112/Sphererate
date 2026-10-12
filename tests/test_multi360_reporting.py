from __future__ import annotations

import json

import numpy as np

from sphererate.data.dataset import ProcessedSaliencyDataset
from sphererate.evaluation.evaluate import evaluate_prediction_dir


def test_max_samples_are_balanced_across_datasets(tmp_path) -> None:
    rows = []
    for dataset_name, count in [("jufe", 4), ("salient360", 3), ("sitzmann", 3)]:
        for idx in range(count):
            rows.append(
                {
                    "dataset": dataset_name,
                    "image_id": f"{dataset_name}_{idx}",
                    "split": "train",
                    "image": str(tmp_path / f"{dataset_name}_{idx}.png"),
                    "target": str(tmp_path / f"{dataset_name}_{idx}.npz"),
                }
            )
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    dataset = ProcessedSaliencyDataset(tmp_path, split="train", max_samples=6)
    selected = [row["dataset"] for row in dataset.rows]

    assert selected.count("jufe") == 2
    assert selected.count("salient360") == 2
    assert selected.count("sitzmann") == 2


def test_evaluate_prediction_dir_reports_dataset_and_macro_scores(tmp_path) -> None:
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    manifest_rows = []
    for dataset_name, col in [("jufe", 1), ("salient360", 2)]:
        image_id = f"{dataset_name}_0"
        final = np.zeros((4, 4), dtype=np.float32)
        final[1, col] = 1.0
        target_path = tmp_path / f"{image_id}.npz"
        np.savez_compressed(
            target_path,
            temporal=np.stack([final, final], axis=0),
            cumulative=np.stack([final, final], axis=0),
            final=final,
            fixation=final,
            area_weights=np.ones_like(final, dtype=np.float32),
        )
        np.savez_compressed(pred_dir / f"{image_id}.npz", final=final)
        manifest_rows.append(
            {
                "dataset": dataset_name,
                "image_id": image_id,
                "split": "test",
                "image": str(tmp_path / f"{image_id}.png"),
                "target": str(target_path),
            }
        )
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in manifest_rows) + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "report"
    summary = evaluate_prediction_dir(tmp_path, pred_dir, split="test", output_dir=out_dir)

    assert summary["num_images"] == 2
    assert sorted(summary["datasets"]) == ["jufe", "salient360"]
    assert "kld_sph" in summary["macro_scores"]
    csv_text = (out_dir / "test_per_image.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("dataset,image_id,")
