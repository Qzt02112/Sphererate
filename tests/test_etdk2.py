from __future__ import annotations

import numpy as np
import pytest

from sphererate.data.etdk2 import (
    START_LONGITUDES_DEG,
    erp_start_longitude_deg,
    iter_viewing_sessions,
    normalize_session,
    start_longitude_deg,
)


def _row(
    *,
    image: int,
    gray: bool,
    gray_idx: int,
    time: float,
    subject: int = 2,
    gaze_x: float = 0.5,
    gaze_y: float = 0.5,
) -> dict[str, str]:
    values = {
        "SUB_ID": str(subject),
        "IMG_INDEX": str(image),
        "IS_GRAY": str(gray),
        "GRAY_IDX": str(gray_idx),
        "UNITY_TIMESTAMP": str(time),
        "SMI_TIMESTAMP": str(int(time * 1e9)),
        "GIW_TEXTURE_X": str(gaze_x),
        "GIW_TEXTURE_Y": str(gaze_y),
        "HIW_TEXTURE_X": "0.5",
        "HIW_TEXTURE_Y": "0.5",
    }
    for prefix in ("GIW", "HIW", "GAZE_DIR", "HEAD_POS", "HEAD_EUL"):
        for axis, value in zip("XYZ", (1.0, 0.0, 0.0)):
            values[f"{prefix}_{axis}"] = str(value)
    return values


def test_start_longitude_mapping_matches_official_readme() -> None:
    assert START_LONGITUDES_DEG == (-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0)
    assert [start_longitude_deg(index) for index in range(8)] == list(START_LONGITUDES_DEG)
    assert [erp_start_longitude_deg(index) for index in range(8)] == [
        135.0, 90.0, 45.0, 0.0, -45.0, -90.0, -135.0, -180.0
    ]
    with pytest.raises(ValueError, match="gray index"):
        start_longitude_deg(8)


def test_iter_sessions_attaches_preceding_gray_orientation_and_keeps_repeated_trials() -> None:
    rows = [
        _row(image=-1, gray=True, gray_idx=3, time=0.0),
        _row(image=0, gray=False, gray_idx=3, time=1.0),
        _row(image=0, gray=False, gray_idx=3, time=2.0),
        _row(image=0, gray=True, gray_idx=5, time=3.0),
        _row(image=1, gray=False, gray_idx=5, time=4.0),
        _row(image=1, gray=True, gray_idx=1, time=5.0),
        _row(image=0, gray=False, gray_idx=1, time=6.0),
    ]

    sessions = list(iter_viewing_sessions(rows, {0: "P2", 1: "P3"}))

    assert [session.stimulus_id for session in sessions] == ["P2", "P3", "P2"]
    assert [session.trial_index for session in sessions] == [0, 1, 2]
    assert [session.start_index for session in sessions] == [3, 5, 1]
    assert [session.start_longitude_deg for session in sessions] == [0.0, 90.0, -90.0]
    assert [session.start_erp_longitude_deg for session in sessions] == [0.0, -90.0, 90.0]
    assert all(session.subject_id == "S0002" for session in sessions)


def test_iter_sessions_excludes_unavailable_construction_stimuli() -> None:
    rows = [
        _row(image=-1, gray=True, gray_idx=0, time=0.0),
        _row(image=40, gray=False, gray_idx=0, time=1.0),
        _row(image=40, gray=False, gray_idx=0, time=2.0),
    ]

    assert list(iter_viewing_sessions(rows, {40: "C0"}, shared_only=True)) == []
    assert len(list(iter_viewing_sessions(rows, {40: "C0"}, shared_only=False))) == 1


def test_iter_sessions_rejects_gray_stimulus_start_mismatch() -> None:
    rows = [
        _row(image=-1, gray=True, gray_idx=2, time=0.0),
        _row(image=0, gray=False, gray_idx=3, time=1.0),
    ]

    with pytest.raises(ValueError, match="start mismatch"):
        list(iter_viewing_sessions(rows, {0: "P2"}))


def test_normalize_session_rebases_time_and_marks_invalid_coordinates() -> None:
    rows = [
        _row(image=0, gray=False, gray_idx=3, time=10.0, gaze_x=0.25, gaze_y=0.75),
        _row(image=0, gray=False, gray_idx=3, time=10.5, gaze_x=1.2, gaze_y=0.50),
    ]

    trace = normalize_session(rows)

    np.testing.assert_allclose(trace["time_s"], [0.0, 0.5])
    np.testing.assert_array_equal(trace["gaze_valid"], [True, False])
    np.testing.assert_array_equal(trace["head_valid"], [True, True])
    np.testing.assert_allclose(trace["gaze_erp_xy"][0], [0.25, 0.25])
    np.testing.assert_allclose(trace["head_erp_xy"][0], [0.5, 0.5])
    assert trace["gaze_world_xyz"].shape == (2, 3)
    assert trace["head_world_xyz"].shape == (2, 3)


def test_normalize_session_rejects_non_monotonic_timestamps() -> None:
    rows = [
        _row(image=0, gray=False, gray_idx=3, time=10.0),
        _row(image=0, gray=False, gray_idx=3, time=9.0),
    ]

    with pytest.raises(ValueError, match="monotonic"):
        normalize_session(rows)
