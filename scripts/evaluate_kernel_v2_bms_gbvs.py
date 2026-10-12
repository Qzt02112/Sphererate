#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.early_model import compare_coverage_rows, compare_method_rows
from sphererate.utils import ensure_dir
from scripts.evaluate_eh6_scandmm import _macro, _score


SOURCE = ROOT / "third_party/GBVS360-BMS360-ProSal"
BINARY = SOURCE / "bin/salient"
MODEL_ID = {"gbvs360": 2, "bms360": 6}
OFFICIAL_SHAPE = (1024, 2048)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_state() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(
        ["git", "-C", str(SOURCE), "diff", "--", "Salient360/main.cpp"], text=True
    )
    return {
        "repository": "Telecommunication-Telemedia-Assessment/GBVS360-BMS360-ProSal",
        "commit": commit,
        "compatibility_patch": "expose authors' existing non-submission general-model CLI",
        "compatibility_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "binary_sha256": _sha256(BINARY),
    }


def _asset_workdir(run_root: Path) -> Path:
    workdir = ensure_dir(run_root / "official_assets")
    data_link = workdir / "data"
    if not data_link.exists():
        data_link.symlink_to(SOURCE / "data", target_is_directory=True)
    for name in (
        "haarcascade_frontalface_alt.xml",
        "haarcascade_profileface.xml",
        "haarcascade_eye_tree_eyeglasses.xml",
    ):
        link = workdir / name
        if not link.exists():
            link.symlink_to(SOURCE / "data" / name)
    return workdir


def _predict_one(
    method: str,
    image_path: Path,
    cache_path: Path,
    height: int,
    width: int,
    asset_workdir: Path,
) -> np.ndarray:
    if cache_path.exists():
        prediction = np.load(cache_path)
        if prediction.shape != (height, width):
            raise ValueError(f"Bad cached shape in {cache_path}: {prediction.shape}")
        return np.asarray(prediction, dtype=np.float32)

    raw_path = cache_path.with_suffix(".bin")
    command = [
        str(BINARY),
        "--general-model",
        str(MODEL_ID[method]),
        "--input-file",
        str(image_path.resolve()),
        "--output-file",
        str(raw_path.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=asset_workdir,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0 or not raw_path.exists():
        raise RuntimeError(
            f"{method} failed for {image_path} (exit={completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if "cannot open" in (completed.stdout + completed.stderr).lower():
        raise RuntimeError(f"{method} reported a missing official asset for {image_path}")
    raw = np.fromfile(raw_path, dtype=np.float32)
    expected = OFFICIAL_SHAPE[0] * OFFICIAL_SHAPE[1]
    if raw.size != expected or not np.isfinite(raw).all():
        raise ValueError(f"Invalid official binary map {raw_path}: count={raw.size}, expected={expected}")
    prediction = cv2.resize(
        raw.reshape(OFFICIAL_SHAPE),
        (int(width), int(height)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    prediction = np.maximum(prediction, 0.0)
    np.save(cache_path, prediction)
    raw_path.unlink()
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate official deterministic BMS360/GBVS360 on kernel-v2.")
    parser.add_argument("--method", choices=tuple(MODEL_ID), required=True)
    parser.add_argument("--split", choices=("val", "gate", "test"), default="val")
    parser.add_argument(
        "--data-root", default="data/processed/early_horizon_salient360_spherical2deg_v2"
    )
    parser.add_argument("--run-root", default="runs/kernel_v2_bms_gbvs")
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    if not BINARY.is_file():
        raise FileNotFoundError(f"Build the official binary first: {ROOT / 'scripts/build_bms360_gbvs360.sh'}")

    dataset = EarlyHorizonDataset(args.data_root, split=args.split, target_mode="correct")
    samples = [dataset[index] for index in range(len(dataset))]
    expected = int(
        args.expected_count if args.expected_count is not None else {"val": 9, "gate": 13, "test": 13}[args.split]
    )
    if len(samples) != expected:
        raise ValueError(f"Frozen split mismatch: {args.split} has {len(samples)}, expected {expected}")

    run_root = Path(args.run_root)
    cache_dir = ensure_dir(run_root / args.method / args.split / "maps")
    asset_workdir = _asset_workdir(run_root)
    predictions: dict[str, np.ndarray] = {}
    for sample, row in zip(samples, dataset.rows, strict=True):
        image_id = str(sample["image_id"])
        height, width = (int(value) for value in sample["target"].shape[-2:])
        final = _predict_one(
            args.method,
            Path(str(row["image"])),
            cache_dir / f"{image_id}.npy",
            height,
            width,
            asset_workdir,
        )
        predictions[image_id] = np.repeat(final[None], len(sample["horizons_s"]), axis=0)
        print(f"[{args.method} {args.split}] {image_id}", flush=True)

    budgets = (5.0, 10.0, 15.0, 20.0)
    scores, coverages = _score(predictions, samples, budgets)
    template = {str(sample["image_id"]): sample["template"].numpy() for sample in samples}
    template_scores, template_coverages = _score(template, samples, budgets)
    result = {
        "baseline": args.method.upper(),
        "tier": "deterministic_static_360",
        "split": args.split,
        "training": "not applicable; deterministic published method",
        "time_semantics": "one static content map repeated over all horizons",
        "official_model_id": MODEL_ID[args.method],
        "official_output_shape": list(OFFICIAL_SHAPE),
        "official_defaults": {
            "equatorial_prior": True,
            "postprocess_erosion_kernel": 32,
            "gbvs360_channels": "DIO" if args.method == "gbvs360" else None,
        },
        "source": _source_state(),
        "content_ids": sorted(predictions),
        "method_early_macro": _macro(scores, ("kld_sph", "sim_sph", "cc_sph")),
        "coverage_early_macro": {
            str(budget): _macro(coverages[str(budget)], ("coverage",))["coverage"]
            for budget in budgets
        },
        "score_rows": scores,
        "coverage_rows": coverages,
        "comparisons": {
            "map_vs_early_template": compare_method_rows(
                scores, template_scores, iterations=10000, seed=20260711
            ),
            "coverage_vs_early_template": {
                str(budget): compare_coverage_rows(
                    coverages[str(budget)],
                    template_coverages[str(budget)],
                    iterations=10000,
                    seed=20260711,
                )
                for budget in budgets
            },
        },
    }
    output = ensure_dir(
        args.output_dir or f"reports/kernel_v2_{args.method}_{args.split}"
    )
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metric = result["method_early_macro"]
    coverage = result["coverage_early_macro"]
    lines = [
        f"# Kernel-v2 {args.method.upper()} — {args.split}",
        "",
        "Tier: **published deterministic 360° static baseline**; no fitting or split-specific parameter selection.",
        "",
        "| KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ | Cov@5 | Cov@10 | Cov@15 | Cov@20 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {metric['kld_sph']:.6f} | {metric['sim_sph']:.6f} | {metric['cc_sph']:.6f} | "
        f"{coverage['5.0']:.6f} | {coverage['10.0']:.6f} | {coverage['15.0']:.6f} | "
        f"{coverage['20.0']:.6f} |",
        "",
        "The official 2048×1024 float output is resized to the frozen target grid. Display smoothing is not used.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {
        "method": args.method,
        "split": args.split,
        "content_count": len(predictions),
        "test_evaluated": bool(args.split == "test"),
        "test_used_for_selection": False,
        "source": result["source"],
    }
    (cache_dir.parent / "inference_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
