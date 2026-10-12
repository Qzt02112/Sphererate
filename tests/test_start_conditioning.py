from __future__ import annotations

import numpy as np

from sphererate.data.dataset import _startpoint_relative_channels, _startpoint_row


def test_random_startpoint_is_deterministic_and_seeded() -> None:
    row = {
        "image_id": "jufe_100_len1_st_1__start_good",
        "start_lon_deg": -30.0,
        "start_lat_deg": 0.0,
        "counterpart_start_lon_deg": 140.0,
        "counterpart_start_lat_deg": 0.0,
    }

    first = _startpoint_row(row, "random", random_seed=17)
    repeated = _startpoint_row(row, "random", random_seed=17)
    other_seed = _startpoint_row(row, "random", random_seed=18)

    assert first["start_lon_deg"] == repeated["start_lon_deg"]
    assert first["start_lon_deg"] != other_seed["start_lon_deg"]
    assert -180.0 <= first["start_lon_deg"] < 180.0
    assert first["start_lat_deg"] == 0.0
    assert row["start_lon_deg"] == -30.0


def test_swapped_startpoint_uses_counterpart_without_mutating_row() -> None:
    row = {
        "image_id": "sample",
        "start_lon_deg": -30.0,
        "start_lat_deg": 0.0,
        "counterpart_start_lon_deg": 140.0,
        "counterpart_start_lat_deg": 5.0,
    }

    swapped = _startpoint_row(row, "swapped")

    assert swapped["start_lon_deg"] == 140.0
    assert swapped["start_lat_deg"] == 5.0
    assert row["start_lon_deg"] == -30.0


def test_relative_start_channels_wrap_longitude_and_remain_finite_at_poles() -> None:
    area = np.ones((8, 16), dtype=np.float32)
    left = _startpoint_relative_channels(
        {"image_id": "left", "start_lon_deg": -180.0, "start_lat_deg": 0.0}, 8, 16, area, 1.0
    )
    right = _startpoint_relative_channels(
        {"image_id": "right", "start_lon_deg": 180.0, "start_lat_deg": 0.0}, 8, 16, area, 1.0
    )
    north = _startpoint_relative_channels(
        {"image_id": "north", "start_lon_deg": 0.0, "start_lat_deg": 90.0}, 8, 16, area, 1.0
    )
    south = _startpoint_relative_channels(
        {"image_id": "south", "start_lon_deg": 0.0, "start_lat_deg": -90.0}, 8, 16, area, 1.0
    )

    assert np.allclose(left, right, atol=1e-6)
    assert np.isfinite(north).all()
    assert np.isfinite(south).all()
