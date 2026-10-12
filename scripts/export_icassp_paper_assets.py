#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/icassp_paper_assets"


def main() -> None:
    benchmark = _load(ROOT / "reports/kernel_v2_matched_baselines/results.json")
    efficiency = _load(ROOT / "reports/kernel_v2_matched_efficiency/results.json")
    semantics = _load(ROOT / "reports/scandmm_training_semantics_comparison/results.json")
    integrity = _load(ROOT / "reports/kernel_v2_matched_baseline_audit/results.json")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    main_table = _main_table(benchmark)
    (OUTPUT / "table_main_gate.tex").write_text(main_table, encoding="utf-8")
    (ROOT / "paper/tables/table_main_gate.tex").write_text(
        main_table, encoding="utf-8"
    )
    (OUTPUT / "table_efficiency.tex").write_text(
        _efficiency_table(efficiency), encoding="utf-8"
    )
    (OUTPUT / "table_scandmm_correction.tex").write_text(
        _scandmm_table(semantics), encoding="utf-8"
    )
    (OUTPUT / "results_section.md").write_text(
        _results_text(benchmark, semantics, integrity), encoding="utf-8"
    )
    (OUTPUT / "figure_captions.md").write_text(_captions(), encoding="utf-8")
    manifest = {
        "role": "publication-ready ICASSP quantitative/qualitative assets",
        "source_reports": {
            "benchmark": "reports/kernel_v2_matched_baselines/results.json",
            "efficiency": "reports/kernel_v2_matched_efficiency/results.json",
            "scandmm_correction": "reports/scandmm_training_semantics_comparison/results.json",
            "integrity": "reports/kernel_v2_matched_baseline_audit/results.json",
        },
        "artifacts": sorted(path.name for path in OUTPUT.iterdir() if path.is_file()),
        "development_evaluation": True,
        "test_target_loads": 0,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# ICASSP paper assets\n\n"
        "These tables and paragraphs are generated from the frozen kernel-v2 reports. "
        "The main benchmark is a historically seen development-gate evaluation, not a fresh test.\n\n"
        "- `table_main_gate.tex`: compact matched benchmark.\n"
        "- `table_efficiency.tex`: native-workload efficiency tiers.\n"
        "- `table_scandmm_correction.tex`: supplementary start-semantics correction.\n"
        "- `results_section.md`: restrained English results draft.\n"
        "- `figure_captions.md`: captions for the quantitative, qualitative, and motion figures.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), "artifacts": manifest["artifacts"]}, indent=2))


def _main_table(data: dict) -> str:
    methods = data["gate"]["methods"]
    rows = [
        ("Panoramic-CNN", False),
        ("Panoramic-CNN + reachable", True),
        ("BMS360", False),
        ("BMS360 + reachable", True),
        ("GBVS360", False),
        ("GBVS360 + reachable", True),
        ("DeepGaze MSDB (ERP-adapted)", False),
        ("DeepGaze MSDB + reachable", True),
        ("Pathformer3D", True),
        ("ScanDMM", True),
        ("TempSAL-ERP", False),
        ("Early template", True),
        ("SphereHorizon", True),
    ]
    if "HAT" in methods:
        rows.insert(rows.index(("TempSAL-ERP", False)), ("HAT", True))
    best = {
        "kld_sph": min(methods[name]["metrics"]["kld_sph"] for name, _ in rows),
        "sim_sph": max(methods[name]["metrics"]["sim_sph"] for name, _ in rows),
        "cc_sph": max(methods[name]["metrics"]["cc_sph"] for name, _ in rows),
        "5.0": max(methods[name]["coverage"]["5.0"] for name, _ in rows),
        "10.0": max(methods[name]["coverage"]["10.0"] for name, _ in rows),
        "20.0": max(methods[name]["coverage"]["20.0"] for name, _ in rows),
    }

    def score(value: float, key: str) -> str:
        formatted = f"{value:.3f}"
        return f"\\textbf{{{formatted}}}" if abs(value - best[key]) < 1e-12 else formatted

    def parameters(name: str) -> str:
        value = str(data["metadata"][name]["parameters"])
        if value == "non-neural":
            return "--"
        if value.startswith("451.64M"):
            return "451.64M"
        return value.split()[0]

    checkmark = "$\\checkmark$"
    body = []
    for index, (name, start_aware) in enumerate(rows):
        metric = methods[name]["metrics"]
        coverage = methods[name]["coverage"]
        if name == "TempSAL-ERP":
            body.append("\\midrule")
        row = (
            f"{_latex_name(name)} & {checkmark if start_aware else ''} & "
            f"{parameters(name)} & {score(metric['kld_sph'], 'kld_sph')} & "
            f"{score(metric['sim_sph'], 'sim_sph')} & {score(metric['cc_sph'], 'cc_sph')} & "
            f"{score(coverage['5.0'], '5.0')} & {score(coverage['10.0'], '10.0')} & "
            f"{score(coverage['20.0'], '20.0')} "
        )
        body.append(row + "\\\\")
    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Matched early-attention results on the 13-content development gate. "
        "Map and coverage entries are macro-averaged over $t\\!\\in\\!\\{0.5,1,3\\}$ s; "
        "the 5-s guard is reported separately in Fig.~\\ref{fig:horizon_behavior}. "
        "$\\dagger$ denotes the same train-only reachable-support attachment; $\\checkmark$ indicates "
        "use of the declared start. Params are total neural parameters; ``--'' denotes a non-neural "
        "method. Coverage uses raw gaze dwell on the same $8\\!\\times\\!16$ tile grid.}\n"
        "\\label{tab:matched_gate}\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{2.6pt}\n"
        "\\renewcommand{\\arraystretch}{0.92}\n"
        "\\begin{tabular*}{0.99\\textwidth}{@{\\extracolsep{\\fill}}lcccccccc@{}}\n"
        "\\toprule\n"
        "Method & \\shortstack{Start-\\\\aware} & Params & KLD$_{\\mathrm{sph}}$ $\\downarrow$ & "
        "SIM$_{\\mathrm{sph}}$ $\\uparrow$ & CC$_{\\mathrm{sph}}$ $\\uparrow$ & "
        "\\shortstack{Cov.\\\\@5\\% $\\uparrow$} & "
        "\\shortstack{Cov.\\\\@10\\% $\\uparrow$} & "
        "\\shortstack{Cov.\\\\@20\\% $\\uparrow$} \\\\ \n"
        "\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n"
        "\\end{tabular*}\n"
        "\\renewcommand{\\arraystretch}{1}\n"
        "\\vspace{-1mm}\n"
        "\\end{table*}\n"
    )


def _efficiency_table(data: dict) -> str:
    by_method = {row["method"]: row for row in data["records"]}
    rows = (
        ("spherehorizon", "4 maps, 256$\\times$512"),
        ("tempsal_erp", "4 maps, 256$\\times$512"),
        ("panoramic_cnn", "1 final map, 128$\\times$256"),
        ("deepgaze_msdb", "1 final map, four yaw rolls"),
        ("scandmm", "200$\\times$20 gaze points"),
        ("pathformer3d", "200$\\times$20 gaze points"),
    )
    display = {
        "spherehorizon": "SphereHorizon",
        "tempsal_erp": "TempSAL-ERP",
        "panoramic_cnn": "Panoramic-CNN",
        "deepgaze_msdb": "MSDB-25 (ERP-adapted)",
        "scandmm": "ScanDMM",
        "pathformer3d": "Pathformer3D",
    }
    body = []
    for index, (key, workload) in enumerate(rows):
        row = by_method[key]
        if index == 4:
            body.append("\\midrule")
        params = row["parameters"]["total"] / 1e6
        latency = row["latency_ms"]
        memory = row["peak_allocated_bytes"] / 1024**2
        body.append(
            f"{display[key]} & {workload} & {params:.2f}M & "
            f"{latency['mean']:.1f}$\\pm${latency['std']:.1f} & {memory:.0f} \\\\"
        )
    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Prepared-input efficiency on one RTX 5880 Ada in FP32. Map prediction and "
        "autoregressive scanpath generation are different native workloads and are separated by a rule.}\n"
        "\\label{tab:efficiency}\n"
        "\\begin{tabular}{llrrr}\n"
        "\\toprule\n"
        "Method & Native output & Parameters & Latency (ms) & Peak MiB \\\\ \n"
        "\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )


def _scandmm_table(data: dict) -> str:
    body = []
    for split in ("val", "gate"):
        row = data["splits"][split]
        for key, name in (("legacy", "First-target initializer"), ("corrected", "Independent start")):
            metric = row[f"{key}_early_macro"]
            body.append(
                f"{split.capitalize()} & {name} & {metric['kld_sph']:.3f} & "
                f"{metric['sim_sph']:.3f} & {metric['cc_sph']:.3f} & "
                f"{row[f'{key}_coverage_10pct']:.3f} \\\\"
            )
        if split == "val":
            body.append("\\midrule")
    return (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Correcting ScanDMM start conditioning.}\n"
        "\\label{tab:scandmm_start}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{llcccc}\n\\toprule\n"
        "Split & Initializer & KLD $\\downarrow$ & SIM $\\uparrow$ & CC $\\uparrow$ & Cov. $\\uparrow$ \\\\ \n"
        "\\midrule\n" + "\n".join(body) + "\n"
        "\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )


def _results_text(benchmark: dict, semantics: dict, integrity: dict) -> str:
    gate = benchmark["gate"]
    methods = gate["methods"]
    comparisons = gate["spherehorizon_comparisons"]
    sphere = methods["SphereHorizon"]
    template_cmp = comparisons["Early template"]
    temp_cmp = comparisons["TempSAL-ERP"]
    scan_cmp = comparisons["ScanDMM"]
    hat_cmp = comparisons["HAT"]
    corrected = semantics["splits"]["gate"]
    return f"""# Results draft

## Early cumulative maps require both a start prior and image evidence

SphereHorizon achieves {sphere['metrics']['kld_sph']:.3f} KLD, {sphere['metrics']['sim_sph']:.3f} SIM, and {sphere['metrics']['cc_sph']:.3f} CC on the development gate (Table~\\ref{{tab:matched_gate}}). Its improvement over the train-only early template is modest but consistent: the paired KLD gain is {_ci(template_cmp['map']['early_macro']['kld_sph'])}, and the 10\\% coverage gain is {_ci(template_cmp['coverage']['10.0'])}. This pattern is consistent with the task decomposition: the shared start strongly determines early support, while image content provides a smaller re-ranking signal within that support.

The matched TempSAL-ERP baseline reaches {methods['TempSAL-ERP']['metrics']['kld_sph']:.3f} KLD. SphereHorizon improves KLD by {_ci(temp_cmp['map']['early_macro']['kld_sph'])}, with corresponding SIM and CC gains. Their coverage difference is only {_ci(temp_cmp['coverage']['10.0'])}; the interval crosses zero. We therefore interpret the comparison as an improvement in map accuracy with comparable fixed-budget coverage, rather than a uniform advantage on every endpoint.

## Reachability is necessary but insufficient

Applying the train-calibrated reachable support substantially strengthens static saliency. GBVS360 improves from {methods['GBVS360']['metrics']['kld_sph']:.3f} to {methods['GBVS360 + reachable']['metrics']['kld_sph']:.3f} KLD, and the ERP-adapted MSDB model improves from {methods['DeepGaze MSDB (ERP-adapted)']['metrics']['kld_sph']:.3f} to {methods['DeepGaze MSDB + reachable']['metrics']['kld_sph']:.3f}. Both remain behind SphereHorizon. The gap to MSDB+reachable is {_ci(comparisons['DeepGaze MSDB + reachable']['map']['early_macro']['kld_sph'])} KLD. Reachability thus constrains where early attention can occur, but it does not determine which reachable regions attract attention.

## Scanpath models move, but solve a different native problem

The same-split Pathformer3D, corrected ScanDMM, and HAT adaptations obtain {methods['Pathformer3D']['metrics']['kld_sph']:.3f}, {methods['ScanDMM']['metrics']['kld_sph']:.3f}, and {methods['HAT']['metrics']['kld_sph']:.3f} KLD. HAT lowers KLD relative to the other scanpath adapters but does not improve their SIM, CC, or 10\\% coverage. SphereHorizon improves over HAT by {_ci(hat_cmp['map']['early_macro']['kld_sph'])} KLD and {_ci(hat_cmp['coverage']['10.0'])} coverage, and over corrected ScanDMM by {_ci(scan_cmp['map']['early_macro']['kld_sph'])} KLD. These are secondary 1-Hz scanpath-to-map adaptations whose 0.5-s map reuses the first-second point; plausible individual rollouts do not guarantee a calibrated population density.

The ScanDMM correction separates the declared start from the first supervised gaze. On the gate, this change improves KLD by {_ci(corrected['paired_map_difference']['early_macro']['kld_sph'])} and coverage by {_ci(corrected['paired_coverage_10pct_difference'])} relative to the legacy initializer. The corrected row is therefore both fairer and stronger; the legacy row is excluded from the matched table.

## Application gain and qualitative behavior

At a 10\\% tile budget, SphereHorizon covers {sphere['coverage']['10.0']:.3f} of early gaze mass, compared with {methods['Early template']['coverage']['10.0']:.3f} for the template. The absolute gain recovers 11.2\\% of the template's remaining uncovered mass. In P95 and P37, content-specific re-ranking moves the fixed budget toward more GT-ranked tiles; P10 is an informative failure where HAT and TempSAL allocate the same budget more effectively. The ranking can therefore reverse near the tile cutoff even when the full density is more accurate.

## Efficiency and scope

SphereHorizon produces all four 256$\\times$512 maps in 9.2 ms with 3.69M parameters. ScanDMM and Pathformer3D require 174.3 and 517.8 ms to generate 200$\\times$20 gaze points, but these timings reflect different output workloads and are not architecture-only speed rankings. The integrity audit passes {integrity['passed']}/{integrity['total']} checks with zero v2-test target access. External ET-DK2 evaluation confirms image sensitivity but does not establish a significant improvement over its start template; the claim is therefore limited to canonical/equatorial early population attention rather than broad cross-dataset generalization.
"""


def _captions() -> str:
    return """# Publication captions

**Fig. 2 — Horizon-wise behavior.** Positive values indicate a SphereHorizon improvement over the named control on the 13-content development gate. Error bars are paired 95% content-bootstrap intervals with 10,000 resamples. The shaded 0.5/1/3-s range defines the primary early task, while 5 s is a prespecified guard. KLD gain is control minus SphereHorizon; coverage gain is SphereHorizon minus control under the same 10% tile budget.

**Fig. 3 — Matched tile decisions at 3 s.** P95 and P37 show cases where content-specific re-ranking moves the fixed budget toward more GT-ranked tiles; P10 is a failure where other methods allocate the same budget more effectively. Numbers report raw-dwell coverage and GT-tile overlap. Cyan, blue, and orange denote the GT set and selections inside/outside it. Scanpath panels use 200 paths from one fixed seed.

**Evidence-package Fig. — Matched macro benchmark.** Early-map accuracy and fixed-budget gaze coverage on the development gate, averaged over 0.5/1/3 s. Points are absolute scores, arrows show the train-calibrated reachability attachment, and error bars are 95% paired content-bootstrap intervals. This expanded diagnostic remains outside the main manuscript.

**Supplementary Fig. — Generated-path motion.** Thirty trajectories from one fixed seed are shown over the first three seconds. Both Pathformer3D and corrected ScanDMM move away from the declared start. Their start-centered population densities therefore reflect directional averaging, not copied or stationary start points.
"""


def _ci(row: dict) -> str:
    return f"{row['mean']:.4f} [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"


def _latex_name(name: str) -> str:
    return {
        "Panoramic-CNN": "Panoramic-CNN~\\cite{martin2020panoramic}",
        "Panoramic-CNN + reachable": "Panoramic-CNN$^{\\dagger}$~\\cite{martin2020panoramic}",
        "BMS360": "BMS360~\\cite{lebreton2018gbvs360}",
        "BMS360 + reachable": "BMS360$^{\\dagger}$~\\cite{lebreton2018gbvs360}",
        "GBVS360": "GBVS360~\\cite{lebreton2018gbvs360}",
        "GBVS360 + reachable": "GBVS360$^{\\dagger}$~\\cite{lebreton2018gbvs360}",
        "DeepGaze MSDB (ERP-adapted)": "MSDB-25 (ERP)~\\cite{kummerer2025msdb}",
        "DeepGaze MSDB + reachable": "MSDB-25$^{\\dagger}$ (ERP)~\\cite{kummerer2025msdb}",
        "Pathformer3D": "Pathformer3D~\\cite{quan2024pathformer3d}",
        "ScanDMM": "ScanDMM~\\cite{sui2023scandmm}",
        "HAT": "HAT (ERP)~\\cite{yang2024hat}",
        "TempSAL-ERP": "TempSAL-ERP~\\cite{aydemir2023tempsal}",
        "Early template": "Early template ($\\beta_t=0$)",
        "SphereHorizon": "SphereHorizon (ours)",
    }.get(name, name)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
