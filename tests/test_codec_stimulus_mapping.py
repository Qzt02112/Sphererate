from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.prepare_salient360_codec_stimuli import (
    alignment_audit,
    build_manifest,
    discover_originals,
    image_id_from_path,
)


def test_image_id_parser_is_strict_about_p_tokens() -> None:
    assert image_id_from_path(Path("P001.jpg")) == "P1"
    assert image_id_from_path(Path("stimulus_P42_final.png")) == "P42"
    assert image_id_from_path(Path("map42.png")) is None


def test_discovery_rejects_ambiguous_source_images(tmp_path: Path) -> None:
    Image.new("RGB", (1024, 512)).save(tmp_path / "P1.jpg")
    Image.new("RGB", (1024, 512)).save(tmp_path / "P01.png")
    with pytest.raises(RuntimeError, match="Ambiguous"):
        discover_originals(tmp_path)


def test_manifest_preserves_split_and_original_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "P7.jpg"
    Image.new("RGB", (1024, 512), "red").save(path)
    rows = build_manifest(
        {"P7": path},
        [{"image_id": "P7", "split": "test"}],
    )
    assert rows[0]["canonical_content_id"] == "salient360:P7"
    assert rows[0]["split"] == "test"
    assert rows[0]["width"] == 1024
    assert rows[0]["height"] == 512


def test_manifest_rejects_processed_resolution_as_codec_source(tmp_path: Path) -> None:
    path = tmp_path / "P7.png"
    Image.new("RGB", (512, 256)).save(path)
    with pytest.raises(RuntimeError, match="not an original-resolution"):
        build_manifest({"P7": path}, [{"image_id": "P7", "split": "test"}])


def test_alignment_audit_detects_zero_shift(tmp_path: Path) -> None:
    array = np.tile(np.arange(1024, dtype=np.uint16), (512, 1))
    original = tmp_path / "P7.png"
    processed = tmp_path / "processed.png"
    Image.fromarray((array % 256).astype(np.uint8)).save(original)
    Image.open(original).resize((512, 256), Image.Resampling.LANCZOS).save(processed)
    shift, ncc = alignment_audit(original, processed)
    assert shift == 0
    assert ncc > 0.999
