#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.models.registry import build_model
from sphererate.utils import ensure_dir


METHODS = (
    "spherehorizon",
    "tempsal_erp",
    "panoramic_cnn",
    "deepgaze_msdb",
    "pathformer3d",
    "scandmm",
    "bms360",
    "gbvs360",
)


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _measure_prepared_cuda(
    function: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            function()
            torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
        peak = int(torch.cuda.max_memory_allocated(device))
    return {
        "mean": float(statistics.fmean(timings)),
        "std": float(statistics.pstdev(timings)),
        "median": float(statistics.median(timings)),
        "p90": _percentile(timings, 90.0),
        "min": float(min(timings)),
        "max": float(max(timings)),
    }, peak


def _sample_and_image_path() -> tuple[dict[str, Any], Path]:
    dataset = EarlyHorizonDataset(
        ROOT / "data/processed/early_horizon_salient360_spherical2deg_v2",
        split="gate",
        target_mode="correct",
    )
    return dataset[0], Path(str(dataset.rows[0]["image"]))


def _sample() -> dict[str, Any]:
    return _sample_and_image_path()[0]


def _map_method(method: str, device: torch.device, warmup: int, repeats: int) -> dict[str, Any]:
    sample = _sample()
    image = sample["image"].unsqueeze(0).to(device)
    if method == "spherehorizon":
        from scripts.evaluate_eh3_gate import _build_model_from_state

        checkpoint = ROOT / "runs/kernel_v2_candidate_correct_seed20260711/checkpoints/best.pt"
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model = _build_model_from_state(state, device)
        template = sample["template"].unsqueeze(0).to(device)
        function = lambda: model(image, template)["cumulative"]
        output = "4 cumulative maps at 256x512"
        parameters = state["parameters"]
    elif method == "tempsal_erp":
        checkpoint = ROOT / "runs/kernel_v2_tempsal_erp_seed20260711/checkpoints/best.pt"
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model = build_model("semantic_tempsal_temporal", state["config"]).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        function = lambda: model(image)["cumulative"]
        output = "4 cumulative maps at 256x512"
        parameters = state["parameters"]
    elif method == "deepgaze_msdb":
        third_party = ROOT / "third_party/deepgaze-msdb"
        sys.path.insert(0, str(third_party))
        from deepgaze_pytorch import DeepGazeMSDB

        checkpoint = Path.home() / ".cache/torch/hub/checkpoints/deepgazemsdb.pth"
        model = DeepGazeMSDB(pretrained=True).to(device)
        # The official feature branch currently returns None from train()/eval().
        model.eval()
        native = (image[:, :3] * 255.0).to(torch.uint8)
        centerbias = torch.zeros(native.shape[0], native.shape[-2], native.shape[-1], device=device)
        shifts = (0, native.shape[-1] // 4, native.shape[-1] // 2, 3 * native.shape[-1] // 4)

        def function() -> torch.Tensor:
            outputs = []
            for shift in shifts:
                current = torch.roll(native, shifts=shift, dims=-1)
                prediction = model(current, centerbias, pixel_per_dva=30.0, dataset=None).exp()
                outputs.append(torch.roll(prediction, shifts=-shift, dims=-1))
            return torch.stack(outputs).mean(dim=0)

        output = "1 final map at 256x512 from four ERP yaw rolls; repetition is free"
        parameters = {
            "total": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable": 2,
            "note": "official network frozen; two external train-only calibration scalars",
        }
    else:
        from torch.nn import functional as F
        from sphererate.baselines.panoramic_cnn import build_official_panoramic_cnn

        checkpoint = ROOT / "runs/kernel_v2_panoramic_static_seed20260711/checkpoints/best.pt"
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model = build_official_panoramic_cnn().to(device)
        model.load_state_dict(state["model"])
        model.eval()
        native = torch.nn.functional.interpolate(
            image[:, :3], size=(int(state["height"]), int(state["width"])),
            mode="bilinear", align_corners=False,
        )
        function = lambda: F.softplus(model(native))
        output = "1 final map at native 128x256; repetition to 4 horizons is free"
        count = int(sum(parameter.numel() for parameter in model.parameters()))
        parameters = {"total": count, "trainable": count}
    latency, peak = _measure_prepared_cuda(
        function, warmup=warmup, repeats=repeats, device=device
    )
    return {
        "method": method,
        "tier": "native_map_model",
        "checkpoint": str(checkpoint),
        "parameters": parameters,
        "prepared_input_shape": list(native.shape) if method in {"panoramic_cnn", "deepgaze_msdb"} else list(image.shape),
        "output_workload": output,
        "latency_ms": latency,
        "peak_allocated_bytes": peak,
        "timing_scope": "prepared GPU tensor to model output; batch=1; float32; excludes data loading and metrics",
    }


def _pathformer(device: torch.device, warmup: int, repeats: int) -> dict[str, Any]:
    from scripts.select_kernel_v2_pathformer3d import (
        _load_model,
        _official_images,
        _sample_scanpaths,
    )

    selection = json.loads(
        (ROOT / "runs/kernel_v2_pathformer3d_retrained_seed20260711/selection.json").read_text()
    )
    checkpoint = ROOT / selection["selected"]["checkpoint"]
    model = _load_model(checkpoint, device)
    images = _official_images(
        ROOT / "data/processed/kernel_v2_matched_scanpaths/pathformer3d/dataset.json",
        "gate",
    )
    image_id = sorted(images)[0]
    image = images[image_id]
    function = lambda: _sample_scanpaths(model, image, 200, device)
    latency, peak = _measure_prepared_cuda(
        function, warmup=warmup, repeats=repeats, device=device
    )
    count = int(sum(parameter.numel() for parameter in model.parameters()))
    return {
        "method": "pathformer3d",
        "tier": "autoregressive_scanpath_model",
        "checkpoint": str(checkpoint),
        "parameters": {"total": count, "trainable": count},
        "prepared_input_shape": list(image.shape),
        "output_workload": "200 scanpaths x 20 one-second gaze points",
        "latency_ms": latency,
        "peak_allocated_bytes": peak,
        "timing_scope": "prepared GPU image to normalized scanpaths; includes autoregressive decoding and GPU-to-CPU coordinate conversion; excludes map rendering",
    }


def _scandmm(device: torch.device, warmup: int, repeats: int) -> dict[str, Any]:
    import pyro
    from pyro.infer import Predictive
    from scripts.run_scandmm_inference import _install_dummy_matplotlib

    _install_dummy_matplotlib()
    source = Path(os.path.expandvars("${SCANDMM_ROOT}/baselines/third_party/ScanDMM"))
    sys.path.insert(0, str(source))
    from inference import Inference  # type: ignore
    from models import DMM  # type: ignore
    from suppor_lib import image_process, sphere2xyz  # type: ignore
    from sphererate.baselines.scandmm_declared_start import build_declared_start_dmm

    selection = json.loads(
        (
            ROOT
            / "runs/kernel_v2_scandmm_declared_start_seed20260711/selection.json"
        ).read_text()
    )
    checkpoint = ROOT / selection["best"]["checkpoint"]
    pyro.clear_param_store()
    DMM = build_declared_start_dmm(DMM)
    model = DMM(use_cuda=True)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    model.eval()
    runner = Inference(model, "", 200, 20, "", False)
    sample, image_path = _sample_and_image_path()
    # EarlyHorizonDataset exposes the resized image tensor; official ScanDMM expects
    # its own 240x320 preprocessing, so use the manifest image only during setup.
    image_tensor = image_process(str(image_path)).unsqueeze(0).repeat([200, 1, 1, 1]).to(device)
    lat_lon = torch.tensor([[0.0, 90.0]], dtype=torch.float32).repeat(200, 1)
    start = sphere2xyz(lat_lon).unsqueeze(1).to(torch.float32)
    scanpaths = start.repeat([1, 20, 1]).to(device)
    mask = torch.ones([200, 20], device=device)
    predictive = Predictive(model.model, num_samples=1)

    def function() -> np.ndarray:
        samples = predictive(
            scanpaths=scanpaths,
            scanpaths_reversed=None,
            mask=mask,
            scanpath_lengths=None,
            images=image_tensor,
            predict=True,
            initial_gaze=start[:, 0, :],
        )
        return runner.summary(samples).cpu().numpy()

    latency, peak = _measure_prepared_cuda(
        function, warmup=warmup, repeats=repeats, device=device
    )
    count = int(sum(parameter.numel() for parameter in model.parameters()))
    return {
        "method": "scandmm",
        "tier": "autoregressive_scanpath_model",
        "checkpoint": str(checkpoint),
        "parameters": {"total": count, "trainable": count},
        "prepared_input_shape": list(image_tensor.shape),
        "output_workload": "200 scanpaths x 20 one-second gaze points",
        "latency_ms": latency,
        "peak_allocated_bytes": peak,
        "timing_scope": "official-preprocessed GPU image batch and declared start to normalized scanpaths; includes Pyro generation and coordinate conversion; excludes image preprocessing and map rendering",
        "initialization_semantics": "independent_declared_tensor",
    }


def _traditional(method: str, warmup: int, repeats: int) -> dict[str, Any]:
    from scripts.evaluate_kernel_v2_bms_gbvs import BINARY, MODEL_ID, _asset_workdir

    _, image_path = _sample_and_image_path()
    image_path = image_path.resolve()
    asset_workdir = _asset_workdir(ROOT / "runs/kernel_v2_bms_gbvs")
    timings = []
    with tempfile.TemporaryDirectory(prefix=f"{method}_efficiency_") as temporary:
        output = Path(temporary) / "map.bin"
        command = [
            str(BINARY), "--general-model", str(MODEL_ID[method]),
            "--input-file", str(image_path), "--output-file", str(output),
        ]
        for index in range(warmup + repeats):
            if output.exists():
                output.unlink()
            start = time.perf_counter()
            completed = subprocess.run(
                command, cwd=asset_workdir, capture_output=True, text=True, timeout=300
            )
            elapsed = (time.perf_counter() - start) * 1000.0
            if completed.returncode != 0 or not output.exists():
                raise RuntimeError(f"{method} efficiency run failed: {completed.stderr}")
            if index >= warmup:
                timings.append(elapsed)
    return {
        "method": method,
        "tier": "deterministic_cpu_saliency",
        "checkpoint": None,
        "parameters": None,
        "prepared_input_shape": "official image file",
        "output_workload": "1 final saliency map at 1024x2048",
        "latency_ms": {
            "mean": float(statistics.fmean(timings)),
            "std": float(statistics.pstdev(timings)),
            "median": float(statistics.median(timings)),
            "p90": _percentile(timings, 90.0),
            "min": float(min(timings)),
            "max": float(max(timings)),
        },
        "peak_allocated_bytes": None,
        "timing_scope": "CPU subprocess wall time from image file to official 1024x2048 float output; excludes resize and metrics",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark matched kernel-v2 baselines without conflating output workloads.")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--output-dir", default="reports/kernel_v2_matched_efficiency")
    args = parser.parse_args()
    scanpath = args.method in {"pathformer3d", "scandmm"}
    traditional = args.method in {"bms360", "gbvs360"}
    warmup = int(args.warmup if args.warmup is not None else (1 if scanpath or traditional else 10))
    repeats = int(args.repeats if args.repeats is not None else (5 if scanpath or traditional else 50))
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")
    if traditional:
        result = _traditional(args.method, warmup, repeats)
    else:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Neural matched-efficiency benchmark is frozen to CUDA")
        if args.method in {"spherehorizon", "tempsal_erp", "panoramic_cnn", "deepgaze_msdb"}:
            result = _map_method(args.method, device, warmup, repeats)
        elif args.method == "pathformer3d":
            result = _pathformer(device, warmup, repeats)
        else:
            result = _scandmm(device, warmup, repeats)
        result["gpu_name"] = torch.cuda.get_device_name(device)
    result.update(
        {
            "warmup": warmup,
            "repeats": repeats,
            "precision": "float32",
            "benchmark_date": "2026-07-12",
            "test_target_loads": 0,
        }
    )
    output = ensure_dir(args.output_dir) / f"{args.method}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
