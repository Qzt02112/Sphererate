from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image


SPATIAL_TARGET_KEYS = frozenset({"cumulative", "final", "fixation", "area_weights"})


def roll_normalized_x(x: np.ndarray, shift_px: int, width: int) -> np.ndarray:
    """Circularly move normalized ERP x coordinates by an integer pixel shift."""

    if width <= 0:
        raise ValueError("width must be positive")
    return np.mod(np.asarray(x, dtype=np.float64) + float(shift_px) / float(width), 1.0)


def roll_erp_array(values: np.ndarray, shift_px: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError(f"ERP array needs at least two dimensions, got {array.shape}")
    return np.roll(array, shift=int(shift_px), axis=-1)


def roll_target_archive(source: Path, destination: Path, shift_px: int) -> None:
    with np.load(source, allow_pickle=False) as data:
        values = {
            key: roll_erp_array(data[key], shift_px) if key in SPATIAL_TARGET_KEYS else np.asarray(data[key])
            for key in data.files
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **values)


def roll_fixation_archive(source: Path, destination: Path, shift_px: int, width: int) -> None:
    with np.load(source, allow_pickle=False) as data:
        values = {key: np.asarray(data[key]) for key in data.files}
    fixations = np.asarray(values["fixations"]).copy()
    if fixations.ndim != 2 or fixations.shape[1] < 2:
        raise ValueError(f"Invalid fixation array {fixations.shape} in {source}")
    fixations[:, 0] = roll_normalized_x(fixations[:, 0], shift_px, width).astype(fixations.dtype)
    values["fixations"] = fixations
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **values)


def roll_scanpath_archive(source: Path, destination: Path, shift_px: int, width: int) -> None:
    with np.load(source, allow_pickle=False) as data:
        values = {key: np.asarray(data[key]) for key in data.files}
    scanpaths = np.asarray(values["scanpaths_yx"]).copy()
    if scanpaths.ndim != 3 or scanpaths.shape[-1] != 2:
        raise ValueError(f"Invalid scanpath array {scanpaths.shape} in {source}")
    scanpaths[..., 1] = roll_normalized_x(scanpaths[..., 1], shift_px, width).astype(scanpaths.dtype)
    values["scanpaths_yx"] = scanpaths
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **values)


def build_cross_dataset_view(
    source_root: str | Path,
    output_root: str | Path,
    *,
    shift_px: int,
    model_source: str,
    target_start_x_norm: float,
    external_scanpath_start_yaw_deg: float,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize a coordinate-only ERP view for frozen cross-domain inference.

    RGB, fixation coordinates, target maps, and display-independent templates are
    rolled together. Split labels and canonical scene identities are immutable.
    Sealed rows are copied as metadata only; their absent targets are never read.
    """

    workspace = Path(workspace_root or Path.cwd()).resolve()
    source = _resolve(source_root, workspace)
    output = _resolve(output_root, workspace)
    manifest = source / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"No rows in {manifest}")

    width: int | None = None
    transformed: list[dict[str, Any]] = []
    sealed_target_loads = 0
    for row in rows:
        image_id = str(row["image_id"])
        image_source = _resolve(row["image"], workspace)
        with Image.open(image_source) as image:
            rgb = np.asarray(image.convert("RGB"))
        if width is None:
            width = int(rgb.shape[1])
        if rgb.shape[1] != width:
            raise ValueError(f"Mixed ERP widths: {rgb.shape[1]} and {width}")
        image_destination = output / "images" / f"{image_id}.png"
        image_destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.roll(rgb, shift=int(shift_px), axis=1)).save(image_destination)

        current = dict(row)
        current["image"] = _workspace_relative(image_destination, workspace)
        current["cross_view"] = True
        current["cross_view_model_source"] = str(model_source)
        current["cross_view_shift_px"] = int(shift_px)
        current["model_start_x_norm"] = float(target_start_x_norm)
        current["external_scanpath_start_yaw_deg"] = float(external_scanpath_start_yaw_deg)
        current["model_declared_start_yaw_deg"] = float(external_scanpath_start_yaw_deg)

        target_path = row.get("target")
        fixation_path = row.get("fixations")
        if bool(row.get("target_sealed")) or target_path is None or fixation_path is None:
            current["target"] = None
            current["fixations"] = None
        else:
            target_destination = output / "targets" / f"{image_id}.npz"
            fixation_destination = output / "fixations" / f"{image_id}.npz"
            roll_target_archive(_resolve(target_path, workspace), target_destination, shift_px)
            roll_fixation_archive(_resolve(fixation_path, workspace), fixation_destination, shift_px, width)
            current["target"] = _workspace_relative(target_destination, workspace)
            current["fixations"] = _workspace_relative(fixation_destination, workspace)

            if row.get("scanpaths_1hz"):
                scanpath_destination = output / "scanpaths" / f"{image_id}.npz"
                roll_scanpath_archive(
                    _resolve(row["scanpaths_1hz"], workspace), scanpath_destination, shift_px, width
                )
                current["scanpaths_1hz"] = _workspace_relative(scanpath_destination, workspace)

            template_source = source / "baselines" / "early_template" / f"{image_id}.npz"
            if template_source.is_file():
                template_destination = output / "baselines" / "early_template" / f"{image_id}.npz"
                roll_target_archive(template_source, template_destination, shift_px)
        transformed.append(current)

    output.mkdir(parents=True, exist_ok=True)
    manifest_output = output / "manifest.jsonl"
    manifest_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in transformed),
        encoding="utf-8",
    )
    for name in ("dataset_info.json",):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, output / name)
    metadata_source = source / "baselines" / "metadata.json"
    if metadata_source.is_file():
        metadata_destination = output / "baselines" / "metadata.json"
        metadata_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metadata_source, metadata_destination)

    result = {
        "protocol": "coordinate_only_cross_dataset_view_v1",
        "source_root": _workspace_relative(source, workspace),
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "output_root": _workspace_relative(output, workspace),
        "output_manifest_sha256": hashlib.sha256(manifest_output.read_bytes()).hexdigest(),
        "row_count": len(transformed),
        "split_counts": {
            split: sum(row["split"] == split for row in transformed)
            for split in sorted({str(row["split"]) for row in transformed})
        },
        "shift_px": int(shift_px),
        "width": int(width or 0),
        "model_source": str(model_source),
        "target_start_x_norm": float(target_start_x_norm),
        "external_scanpath_start_yaw_deg": float(external_scanpath_start_yaw_deg),
        "sealed_target_loads": sealed_target_loads,
        "changes_split_membership": False,
        "changes_canonical_content_id": False,
    }
    (output / "cross_view_provenance.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def _resolve(path: str | Path, workspace: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (workspace / value).resolve()


def _workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace))
    except ValueError:
        return str(path.resolve())
