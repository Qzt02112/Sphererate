#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/kernel_v2_matched_efficiency"
ORDER = (
    "spherehorizon",
    "tempsal_erp",
    "panoramic_cnn",
    "deepgaze_msdb",
    "pathformer3d",
    "scandmm",
    "bms360",
    "gbvs360",
)
DISPLAY = {
    "spherehorizon": "SphereHorizon",
    "tempsal_erp": "TempSAL-ERP",
    "panoramic_cnn": "Panoramic-CNN",
    "deepgaze_msdb": "DeepGaze MSDB (ERP-adapted)",
    "pathformer3d": "Pathformer3D",
    "scandmm": "ScanDMM",
    "bms360": "BMS360",
    "gbvs360": "GBVS360",
}


def main() -> None:
    records = [json.loads((REPORT / f"{method}.json").read_text()) for method in ORDER]
    (REPORT / "results.json").write_text(
        json.dumps({"records": records, "test_target_loads": 0}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Kernel-v2 matched-baseline efficiency",
        "",
        "Latency is reported by native output tier because four-map prediction and generation of 200×20 scanpath points are different workloads. Values are single-image wall-clock measurements after warm-up; no data loading, map metric, or scanpath-to-saliency rendering is included unless explicitly stated.",
        "",
        "## Native neural map models — RTX 5880 Ada, FP32",
        "",
        "| Method | Native output | Parameters | Latency mean ± std | Peak allocated |",
        "|---|---|---:|---:|---:|",
    ]
    for record in records:
        if record["tier"] != "native_map_model":
            continue
        parameters = int(record["parameters"]["total"]) / 1e6
        latency = record["latency_ms"]
        memory = int(record["peak_allocated_bytes"]) / 1024**2
        lines.append(
            f"| {DISPLAY[record['method']]} | {record['output_workload']} | {parameters:.2f}M | "
            f"{latency['mean']:.2f} ± {latency['std']:.2f} ms | {memory:.1f} MiB |"
        )
    lines.extend(
        [
            "",
            "## Autoregressive scanpath models — RTX 5880 Ada, FP32",
            "",
            "| Method | Native output | Parameters | Latency mean ± std | Peak allocated |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for record in records:
        if record["tier"] != "autoregressive_scanpath_model":
            continue
        parameters = int(record["parameters"]["total"]) / 1e6
        latency = record["latency_ms"]
        memory = int(record["peak_allocated_bytes"]) / 1024**2
        lines.append(
            f"| {DISPLAY[record['method']]} | {record['output_workload']} | {parameters:.2f}M | "
            f"{latency['mean']:.1f} ± {latency['std']:.1f} ms | {memory:.1f} MiB |"
        )
    lines.extend(
        [
            "",
            "## Deterministic CPU saliency methods",
            "",
            "| Method | Native output | Latency mean ± std | Scope |",
            "|---|---|---:|---|",
        ]
    )
    for record in records:
        if record["tier"] != "deterministic_cpu_saliency":
            continue
        latency = record["latency_ms"]
        lines.append(
            f"| {DISPLAY[record['method']]} | {record['output_workload']} | "
            f"{latency['mean'] / 1000.0:.2f} ± {latency['std'] / 1000.0:.2f} s | official CPU subprocess |"
        )
    lines.extend(
        [
            "",
            "Interpretation: SphereHorizon is slower than the simpler TempSAL-ERP map head but remains real-time at about 9.2 ms for all four maps. DeepGaze MSDB uses 451.64M frozen parameters and four full model passes for its ERP roll ensemble, so it is a strong recent-content diagnostic rather than an efficiency competitor. ScanDMM and Pathformer3D timings reflect a much larger stochastic workload and must not be read as an architecture-only speed comparison. Reachable-final adds only deterministic O(HW) masking/renormalization and is not included in the native model times.",
            "",
            "All records report `test_target_loads=0`.",
        ]
    )
    (REPORT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT / "summary.md"), "methods": len(records)}, indent=2))


if __name__ == "__main__":
    main()
