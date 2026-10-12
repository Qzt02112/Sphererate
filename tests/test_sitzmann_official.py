from __future__ import annotations

import numpy as np

from sphererate.data.sitzmann_official import (
    OFFICIAL_SCANPATH_TEST_SCENES,
    _scene_split_map,
    align_fixations_to_reference,
    alignment_shift_px,
    make_content_split,
)


def test_viewpoint_alignment_moves_each_declared_start_to_erp_center() -> None:
    for viewpoint, x_norm in enumerate((0.0, 0.25, 0.5, 0.75)):
        fixations = np.asarray([[x_norm, 0.5, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        aligned = align_fixations_to_reference(fixations, viewpoint, width=512)
        assert aligned[0, 0] == np.float32(0.5)


def test_alignment_shifts_match_quarter_erp_rolls() -> None:
    assert [alignment_shift_px(viewpoint, 512) for viewpoint in range(4)] == [256, 128, 0, -128]


def test_content_split_keeps_published_test_scenes_as_nonfresh_gate() -> None:
    split = make_content_split(range(22), seed=20260713, val_count=3)
    assert split["gate"] == list(OFFICIAL_SCANPATH_TEST_SCENES)
    assert split["val"] == [3, 18, 20]
    assert len(split["train"]) == 16
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]).isdisjoint(split["gate"])
    assert set(split["val"]).isdisjoint(split["gate"])


def test_scene_split_map_accepts_standard_test_role() -> None:
    split = {"train": [1, 2], "val": [3], "test": [0]}
    assert _scene_split_map(split) == {1: "train", 2: "train", 3: "val", 0: "test"}


def test_scene_split_map_rejects_ambiguous_report_roles() -> None:
    split = {"train": [1], "val": [2], "gate": [3], "test": [4]}
    try:
        _scene_split_map(split)
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("Expected an ambiguous report-role error")
