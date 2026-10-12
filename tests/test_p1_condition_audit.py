from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("audit_p1_condition_data", root / "scripts/audit_p1_condition_data.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair_rows(image_path: str) -> list[dict]:
    common = {
        "source_image_id": "100_len1_st_1",
        "base_image_id": "jufe_100_len1_st_1",
        "split": "train",
        "group_id": "jufe:100",
        "image": image_path,
    }
    return [
        {
            **common,
            "image_id": "jufe_100_len1_st_1__start_good",
            "condition": "good",
            "counterpart_condition": "bad",
            "start_lon_deg": -30.0,
            "start_lat_deg": 0.0,
            "counterpart_start_lon_deg": 140.0,
            "counterpart_start_lat_deg": 0.0,
        },
        {
            **common,
            "image_id": "jufe_100_len1_st_1__start_bad",
            "condition": "bad",
            "counterpart_condition": "good",
            "start_lon_deg": 140.0,
            "start_lat_deg": 0.0,
            "counterpart_start_lon_deg": -30.0,
            "counterpart_start_lat_deg": 0.0,
        },
    ]


def test_circular_distance_wraps_longitude() -> None:
    module = _module()
    assert module.circular_distance_deg(-179.0, 179.0) == 2.0
    assert module.circular_distance_deg(10.0, 40.0) == 30.0


def test_condition_pair_audit_checks_raw_coordinate_provenance(tmp_path) -> None:
    module = _module()
    image = tmp_path / "same.png"
    image.write_bytes(b"same image bytes")
    rows = _pair_rows(str(image))
    result = module.audit_conditioned_rows(
        rows,
        raw_startpoints={"100_len1_st_1.png": (-30.0, 140.0)},
        check_artifacts=False,
    )
    assert result["pair_count"] == 1
    assert result["complete_pair_count"] == 1
    assert result["coordinate_provenance_mismatches"] == 0
    assert result["pair_split_group_mismatches"] == 0
    assert result["pair_image_mismatches"] == 0

    rows[0]["start_lon_deg"] = -20.0
    bad = module.audit_conditioned_rows(
        rows,
        raw_startpoints={"100_len1_st_1.png": (-30.0, 140.0)},
        check_artifacts=False,
    )
    assert bad["coordinate_provenance_mismatches"] == 1


def test_coordinate_summary_reports_unseen_val_coordinates() -> None:
    module = _module()
    rows = [
        {"split": "train", "condition": "good", "start_lon_deg": 0.0, "start_lat_deg": 0.0},
        {"split": "train", "condition": "bad", "start_lon_deg": 180.0, "start_lat_deg": 0.0},
        {"split": "val", "condition": "good", "start_lon_deg": 10.0, "start_lat_deg": 0.0},
        {"split": "val", "condition": "bad", "start_lon_deg": -170.0, "start_lat_deg": 0.0},
    ]
    summary, coordinate_rows = module.condition_coordinate_summary(rows)
    assert summary["all_latitudes_zero"] is True
    assert summary["splits"]["val"]["unseen_exact_coordinate_fraction"] == 1.0
    assert {row["nearest_train_longitude_distance_deg"] for row in coordinate_rows if row["split"] == "val"} == {10.0}


def test_fixed_longitude_sector_wraps_boundaries() -> None:
    module = _module()
    assert module.fixed_longitude_sector(-180.0) == 0
    assert module.fixed_longitude_sector(-60.0) == 1
    assert module.fixed_longitude_sector(59.9) == 1
    assert module.fixed_longitude_sector(60.0) == 2
    assert module.fixed_longitude_sector(180.0) == 0


def test_jufe_target_audit_measures_conditioning_fixation_share(tmp_path) -> None:
    import numpy as np

    module = _module()
    fixation_path = tmp_path / "fixations.npz"
    # Two observers, three fixations each. Both first fixations are at longitude 0, latitude 0.
    fixations = np.asarray(
        [
            [0.5, 0.5, 0.0, 0.0, 1.0, 0.0],
            [0.6, 0.5, 1.0, 0.0, 1.0, 1.0],
            [0.7, 0.5, 2.0, 0.0, 1.0, 2.0],
            [0.5, 0.5, 0.0, 1.0, 1.0, 0.0],
            [0.4, 0.5, 1.0, 1.0, 1.0, 1.0],
            [0.3, 0.5, 2.0, 1.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(fixation_path, fixations=fixations)
    result = module.jufe_condition_target_audit(
        [
            {
                "split": "train",
                "start_lon_deg": 0.0,
                "start_lat_deg": 0.0,
                "fixations": str(fixation_path),
            }
        ]
    )
    assert result["opened_splits"] == ["train"]
    assert result["conditioning_fixation_mass_fraction_mean"] == 1.0 / 3.0
    assert result["first_fixation_within_5deg_fraction"] == 1.0


def test_good_bad_label_does_not_change_runtime_input_when_coordinate_is_fixed(tmp_path) -> None:
    import json

    import numpy as np
    import torch
    from PIL import Image

    from sphererate.data.dataset import ProcessedSaliencyDataset

    height, width = 8, 16
    image_path = tmp_path / "same.png"
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(image_path)
    target_path = tmp_path / "target.npz"
    density = np.ones((height, width), dtype=np.float32)
    np.savez_compressed(
        target_path,
        temporal=np.stack([density, density]),
        cumulative=np.stack([density, density]),
        final=density,
        fixation=density,
        area_weights=density,
    )
    common = {
        "split": "train",
        "image": str(image_path),
        "target": str(target_path),
        "start_lon_deg": 37.0,
        "start_lat_deg": 0.0,
    }
    rows = [
        {**common, "image_id": "same_good", "condition": "good"},
        {**common, "image_id": "same_bad", "condition": "bad"},
    ]
    manifest = "".join(json.dumps(row) + "\n" for row in rows)
    (tmp_path / "manifest.jsonl").write_text(manifest, encoding="utf-8")

    dataset = ProcessedSaliencyDataset(
        tmp_path,
        split="train",
        augment_roll=False,
        condition_startpoint_channels=True,
        startpoint_condition_mode="actual",
        startpoint_prior_sigma_px=1.0,
    )
    first, second = dataset[0], dataset[1]

    assert torch.equal(first["image"], second["image"])
    assert "condition" not in first and "condition" not in second
