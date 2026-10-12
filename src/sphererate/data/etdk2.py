from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping

import numpy as np


START_LONGITUDES_DEG = (-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0)


@dataclass(frozen=True)
class ETDK2Session:
    subject_id: str
    stimulus_index: int
    stimulus_id: str
    trial_index: int
    start_index: int
    start_longitude_deg: float
    start_erp_longitude_deg: float
    start_latitude_deg: float
    raw_row_start: int
    raw_row_end: int
    rows: tuple[dict[str, str], ...]


def start_longitude_deg(gray_idx: int) -> float:
    index = int(gray_idx)
    if index < 0 or index >= len(START_LONGITUDES_DEG):
        raise ValueError(f"Invalid ET-DK2 gray index: {gray_idx}")
    return START_LONGITUDES_DEG[index]


def erp_start_longitude_deg(gray_idx: int) -> float:
    """Convert the official horizontal direction to ERP texture-X longitude."""
    declared = start_longitude_deg(gray_idx)
    return float(((-declared + 180.0) % 360.0) - 180.0)


def _as_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _subject_id(value: object) -> str:
    return f"S{int(float(str(value))):04d}"


def _make_session(
    rows: list[dict[str, str]],
    image_index: Mapping[int, str],
    trial_index: int,
    raw_row_start: int,
    raw_row_end: int,
    expected_start_index: int | None,
) -> ETDK2Session:
    first = rows[0]
    subjects = {_subject_id(row["SUB_ID"]) for row in rows}
    stimulus_indices = {int(float(row["IMG_INDEX"])) for row in rows}
    starts = {int(float(row["GRAY_IDX"])) for row in rows}
    if len(subjects) != 1 or len(stimulus_indices) != 1 or len(starts) != 1:
        raise ValueError(
            "ET-DK2 session is not internally constant: "
            f"subjects={subjects}, stimuli={stimulus_indices}, starts={starts}"
        )
    stimulus_index = next(iter(stimulus_indices))
    start_index = next(iter(starts))
    if expected_start_index is not None and start_index != expected_start_index:
        raise ValueError(
            f"ET-DK2 gray/stimulus start mismatch at trial {trial_index}: "
            f"gray={expected_start_index}, stimulus={start_index}"
        )
    if stimulus_index not in image_index:
        raise ValueError(f"Unknown ET-DK2 IMG_INDEX={stimulus_index}")
    return ETDK2Session(
        subject_id=next(iter(subjects)),
        stimulus_index=stimulus_index,
        stimulus_id=str(image_index[stimulus_index]),
        trial_index=int(trial_index),
        start_index=start_index,
        start_longitude_deg=start_longitude_deg(start_index),
        start_erp_longitude_deg=erp_start_longitude_deg(start_index),
        start_latitude_deg=0.0,
        raw_row_start=int(raw_row_start),
        raw_row_end=int(raw_row_end),
        rows=tuple(dict(row) for row in rows),
    )


def iter_viewing_sessions(
    rows: Iterable[Mapping[str, object]],
    image_index: Mapping[int, str],
    shared_only: bool = True,
) -> Iterator[ETDK2Session]:
    current: list[dict[str, str]] = []
    current_key: tuple[str, int, int] | None = None
    current_start_row = 0
    current_expected_start: int | None = None
    pending_gray_start: int | None = None
    trial_index = 0
    last_row_index = -1

    def materialize(raw_row_end: int) -> ETDK2Session | None:
        nonlocal trial_index
        if not current:
            return None
        session = _make_session(
            current,
            image_index=image_index,
            trial_index=trial_index,
            raw_row_start=current_start_row,
            raw_row_end=raw_row_end,
            expected_start_index=current_expected_start,
        )
        trial_index += 1
        if shared_only and not session.stimulus_id.startswith("P"):
            return None
        return session

    for row_index, raw in enumerate(rows):
        last_row_index = row_index
        row = {str(key): "" if value is None else str(value) for key, value in raw.items()}
        if _as_bool(row["IS_GRAY"]):
            session = materialize(row_index - 1)
            if session is not None:
                yield session
            current = []
            current_key = None
            current_expected_start = None
            pending_gray_start = int(float(row["GRAY_IDX"]))
            start_longitude_deg(pending_gray_start)
            continue

        key = (
            _subject_id(row["SUB_ID"]),
            int(float(row["IMG_INDEX"])),
            int(float(row["GRAY_IDX"])),
        )
        if current_key is not None and key != current_key:
            session = materialize(row_index - 1)
            if session is not None:
                yield session
            current = []
            current_key = None
            current_expected_start = None

        if not current:
            current_key = key
            current_start_row = row_index
            current_expected_start = pending_gray_start
            pending_gray_start = None
        current.append(row)

    session = materialize(last_row_index)
    if session is not None:
        yield session


def _column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(name, "nan")))
        except (TypeError, ValueError):
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def _xyz(rows: list[dict[str, str]], prefix: str) -> np.ndarray:
    return np.stack([_column(rows, f"{prefix}_{axis}") for axis in "XYZ"], axis=1).astype(np.float32)


def unity_texture_to_erp_xy(texture_xy: np.ndarray) -> np.ndarray:
    """Convert Unity bottom-left UV coordinates to top-left ERP array coordinates."""
    erp = np.asarray(texture_xy, dtype=np.float32).copy()
    if erp.ndim != 2 or erp.shape[1] != 2:
        raise ValueError(f"Expected texture coordinates [N,2], got {erp.shape}")
    erp[:, 1] = 1.0 - erp[:, 1]
    return erp


def normalize_session(rows: Iterable[Mapping[str, object]]) -> dict[str, np.ndarray]:
    materialized = [
        {str(key): "" if value is None else str(value) for key, value in row.items()}
        for row in rows
    ]
    if not materialized:
        raise ValueError("Cannot normalize an empty ET-DK2 session")

    unity = _column(materialized, "UNITY_TIMESTAMP")
    if not np.isfinite(unity).all():
        raise ValueError("ET-DK2 session has invalid Unity timestamps")
    if np.any(np.diff(unity) < -1e-9):
        raise ValueError("ET-DK2 session timestamps are not monotonic")
    time_s = unity - unity[0]

    gaze_xy = np.stack(
        [_column(materialized, "GIW_TEXTURE_X"), _column(materialized, "GIW_TEXTURE_Y")], axis=1
    ).astype(np.float32)
    head_xy = np.stack(
        [_column(materialized, "HIW_TEXTURE_X"), _column(materialized, "HIW_TEXTURE_Y")], axis=1
    ).astype(np.float32)
    gaze_erp_xy = unity_texture_to_erp_xy(gaze_xy)
    head_erp_xy = unity_texture_to_erp_xy(head_xy)
    gaze_valid = np.isfinite(gaze_xy).all(axis=1) & (gaze_xy >= 0.0).all(axis=1) & (gaze_xy <= 1.0).all(axis=1)
    head_valid = np.isfinite(head_xy).all(axis=1) & (head_xy >= 0.0).all(axis=1) & (head_xy <= 1.0).all(axis=1)

    return {
        "time_s": time_s.astype(np.float64),
        "unity_timestamp_s": unity.astype(np.float64),
        "smi_timestamp_ns": _column(materialized, "SMI_TIMESTAMP").astype(np.float64),
        "gaze_texture_xy": gaze_xy,
        "head_texture_xy": head_xy,
        "gaze_erp_xy": gaze_erp_xy,
        "head_erp_xy": head_erp_xy,
        "gaze_valid": gaze_valid.astype(bool),
        "head_valid": head_valid.astype(bool),
        "gaze_world_xyz": _xyz(materialized, "GIW"),
        "head_world_xyz": _xyz(materialized, "HIW"),
        "gaze_head_xyz": _xyz(materialized, "GAZE_DIR"),
        "head_position_xyz": _xyz(materialized, "HEAD_POS"),
        "head_euler_xyz": _xyz(materialized, "HEAD_EUL"),
    }
