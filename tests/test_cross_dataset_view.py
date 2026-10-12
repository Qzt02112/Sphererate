from pathlib import Path

import json
import numpy as np
from PIL import Image

from sphererate.data.cross_dataset_view import (
    build_cross_dataset_view,
    roll_erp_array,
    roll_normalized_x,
)
from sphererate.data.early_dataset import _declared_start_geometry
from scripts.compare_sitzmann_salient360_train_sources import _move_fixations_to_start


def test_roll_normalized_x_is_invertible() -> None:
    values = np.asarray([0.0, 0.1, 0.75, 0.999], dtype=np.float32)
    shifted = roll_normalized_x(values, 128, 512)
    restored = roll_normalized_x(shifted, -128, 512)
    np.testing.assert_allclose(restored, values, atol=1e-7)


def test_coverage_fixations_follow_the_same_cross_dataset_roll() -> None:
    fixations = np.asarray(
        [[0.75, 0.5, 0.0, 1.0, 0.5, 1.0], [0.05, 0.5, 0.0, 1.0, 0.5, 2.0]],
        dtype=np.float32,
    )
    shifted = _move_fixations_to_start(fixations, shift_px=-128, width=512)
    np.testing.assert_allclose(shifted[:, 0], [0.5, 0.8], atol=1e-7)
    restored = _move_fixations_to_start(shifted, shift_px=128, width=512)
    np.testing.assert_allclose(restored, fixations, atol=1e-7)


def test_roll_erp_array_is_invertible() -> None:
    values = np.arange(2 * 4 * 8).reshape(2, 4, 8)
    np.testing.assert_array_equal(roll_erp_array(roll_erp_array(values, 3), -3), values)


def test_build_view_rolls_rgb_targets_fixations_and_preserves_ids(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "targets").mkdir()
    (source / "fixations").mkdir()
    (source / "baselines/early_template").mkdir(parents=True)
    rgb = np.zeros((2, 4, 3), dtype=np.uint8)
    rgb[:, 0] = 255
    Image.fromarray(rgb).save(source / "images/a.png")
    cumulative = np.zeros((1, 2, 4), dtype=np.float32)
    cumulative[:, :, 0] = 1.0
    np.savez_compressed(
        source / "targets/a.npz",
        cumulative=cumulative,
        final=cumulative[0],
        fixation=cumulative[0],
        area_weights=np.ones((2, 4), dtype=np.float32),
        horizons_s=np.asarray([1.0], dtype=np.float32),
    )
    fixation_rows = np.asarray([[0.0, 0.5, 0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    np.savez_compressed(source / "fixations/a.npz", fixations=fixation_rows)
    np.savez_compressed(
        source / "baselines/early_template/a.npz",
        cumulative=cumulative,
        horizons_s=np.asarray([1.0], dtype=np.float32),
    )
    row = {
        "image_id": "a",
        "canonical_content_id": "scene:a",
        "split": "gate",
        "image": str(source / "images/a.png"),
        "target": str(source / "targets/a.npz"),
        "fixations": str(source / "fixations/a.npz"),
        "target_sealed": False,
    }
    (source / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    output = tmp_path / "output"
    result = build_cross_dataset_view(
        source,
        output,
        shift_px=1,
        model_source="other",
        target_start_x_norm=0.75,
        external_scanpath_start_yaw_deg=90.0,
        workspace_root=tmp_path,
    )
    transformed = json.loads((output / "manifest.jsonl").read_text())
    assert transformed["canonical_content_id"] == "scene:a"
    with Image.open(tmp_path / transformed["image"]) as image:
        assert np.asarray(image)[0, 1, 0] == 255
    with np.load(tmp_path / transformed["target"]) as data:
        assert data["cumulative"][0, 0, 1] == 1.0
        np.testing.assert_allclose(data["cumulative"].sum(), cumulative.sum())
    with np.load(tmp_path / transformed["fixations"]) as data:
        np.testing.assert_allclose(data["fixations"][0, 0], 0.25)
    assert result["sealed_target_loads"] == 0
    assert transformed["model_declared_start_yaw_deg"] == 90.0


def test_cross_view_start_geometry_prefers_model_coordinate_metadata() -> None:
    geometry = _declared_start_geometry(
        {
            "declared_start_yaw_deg": 180.0,
            "external_scanpath_start_yaw_deg": 90.0,
        },
        height=16,
        width=32,
    )
    # +90 degrees is x=0.75 under the model's [-pi, pi) longitude grid.
    minimum_x = int(np.argmin(geometry[0, 8]))
    assert minimum_x in {23, 24}
