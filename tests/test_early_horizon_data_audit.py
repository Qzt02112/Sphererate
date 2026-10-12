from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sphererate.data.early_audit import audit_early_horizon_view
from sphererate.data.early_view import build_train_only_baselines


def _write_target(path: Path, value: float) -> None:
    area = np.ones((2, 4), dtype=np.float32)
    density = np.full((2, 4), value, dtype=np.float32)
    density /= float((density * area).sum())
    np.savez_compressed(
        path,
        cumulative=np.stack([density, density]),
        final=density,
        fixation=np.zeros_like(density),
        area_weights=area,
        horizons_s=np.array([0.5, 1.0], dtype=np.float32),
    )


def test_audit_and_train_only_baselines_never_materialize_test_predictions(tmp_path: Path) -> None:
    root = tmp_path / "view"
    (root / "targets").mkdir(parents=True)
    rows = []
    for index, split in enumerate(("train", "train", "val", "test"), start=1):
        target = root / "targets" / f"P{index}.npz"
        if split != "test":
            _write_target(target, float(index))
        rows.append(
            {
                "image_id": f"P{index}",
                "canonical_content_id": f"salient360:P{index}",
                "split": split,
                "target": str(target) if split != "test" else None,
                "fixations": None,
                "target_sealed": split == "test",
            }
        )
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    audit = audit_early_horizon_view(root, expected_horizons_s=(0.5, 1.0))
    baseline_root = build_train_only_baselines(root, tmp_path / "baselines")

    assert audit["zero_leakage"] is True
    assert audit["outer_test_target_files_loaded"] == 0
    assert audit["invalid_spherical_integrals"] == 0
    assert not (baseline_root / "early_template" / "P4.npz").exists()
    metadata = json.loads((baseline_root / "metadata.json").read_text())
    assert metadata["test_prediction_count"] == 0
    assert metadata["train_template_policy"] == "leave_one_content_out"


def test_leave_one_content_out_removes_every_condition_of_held_out_scene(tmp_path: Path) -> None:
    root = tmp_path / "multi_condition_view"
    (root / "targets").mkdir(parents=True)
    rows = []
    patterns = {
        "A0": ("scene:A", 0),
        "A1": ("scene:A", 1),
        "B0": ("scene:B", 4),
        "B1": ("scene:B", 5),
    }
    expected_b = np.zeros((2, 4), dtype=np.float32)
    expected_b.flat[[4, 5]] = 0.5
    for image_id, (content, hot_index) in patterns.items():
        density = np.zeros((2, 4), dtype=np.float32)
        density.flat[hot_index] = 1.0
        target = root / "targets" / f"{image_id}.npz"
        np.savez_compressed(
            target,
            cumulative=np.stack([density, density]),
            final=density,
            fixation=np.zeros_like(density),
            area_weights=np.ones_like(density),
            horizons_s=np.array([0.5, 1.0], dtype=np.float32),
        )
        rows.append(
            {
                "image_id": image_id,
                "canonical_content_id": content,
                "split": "train",
                "target": str(target),
                "fixations": None,
                "target_sealed": False,
            }
        )
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    baseline_root = build_train_only_baselines(root, tmp_path / "multi_condition_baselines")

    for image_id in ("A0", "A1"):
        with np.load(baseline_root / "early_template" / f"{image_id}.npz") as data:
            np.testing.assert_allclose(data["cumulative"][0], expected_b)
            assert int(data["source_count"]) == 2
    metadata = json.loads((baseline_root / "metadata.json").read_text())
    assert metadata["train_content_count"] == 2
    assert metadata["train_condition_count"] == 4
    assert metadata["multi_condition_policy"] == "remove_all_rows_of_the_held_out_canonical_content"
