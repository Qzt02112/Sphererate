# SphereRate Mainline Progress

更新时间：2026-07-13

> Execution is governed by [RESEARCH_PROTOCOL.md](../RESEARCH_PROTOCOL.md). Update this table only after a Gate decision.

## Current Gate

**Kernel-v2 benchmark and official Sitzmann re-entry completed**

```text
old final-map Gate C FAIL
-> early cumulative task + spherical-kernel correction
-> matched development benchmark PASS
-> official Sitzmann train-source audit: keep Salient360 primary
```

## Progress Table

| Work item | State | Evidence / blocker | Next action |
|---|---|---|---|
| Canonical mainline | **Locked** | Start-aligned early cumulative saliency; no gaze history | Keep all docs aligned |
| Group-disjoint split | **Done** | Clean audit: 363 groups, 0 leaks; legacy audit: 214 leaks | Use only `...groupdisjoint_v1` |
| Outer test | **Historical / not fresh** | Used under invalid v1 target; v2 reuse can only be post-hoc | Never use for tuning |
| CC evaluation/loss | **Done** | NumPy/PyTorch self-CC = 1; affine/constant/gradient tests pass | Recompute reports and retrain affected models |
| Prior baselines | **Done** | Clean val: train-average mixed `KLD_sph=0.3718`, `SIM_sph=0.6689`; 184/184 IDs; train-only provenance | Use as locked lower bounds for clean static |
| Strong static | **Locked** | Final-only natural-sampling 3-seed: mixed KLD `0.3554±0.0072`, macro `0.4094±0.0055`; Gate A PASS | Use unchanged as Gate B teacher/control |
| JUFE condition analysis | Done | Same-image good/bad start maps differ | Retained motivation |
| V7 P1 conditioned oracle | **Gate B PASS** | Clean 3-seed actual `KLD_sph=0.2488±0.0099`; static `0.3636`, swapped `0.4566`, random `0.4006`; all paired CIs positive | Freeze as P1 mechanism evidence only |
| V8 P0 marginalization | **Gate C FAIL / NO-GO** | Proposed mixed KLD 0.3497 but macro 0.4220; random condition is stronger; anchors collapse | Do not expand to 3 seeds |
| Temporal process diagnostics | Historical | Existing branches move weakly or collapse; final gains unsupported | Use only as supporting analysis |
| External baselines | **Done on kernel-v2 dev gate** | TempSAL, static, DeepGaze MSDB, ScanDMM, Pathformer3D | Keep frozen |
| Official Sitzmann | **Done** | 22 scenes, four starts, 16/3/3 split; Salient360 remains primary | External/secondary only |
| Next-fixation/scanpath branches | Archived | Outside P0 task | Do not revive as mainline |

## Gate Checklist

## Pre-Gate Smoke Note

2026-07-10 legacy-split smoke:

- Historical result: `jufe_start_conditioned_256x512_k6_order` contained 2064 rows (good/bad each 1032; train/val/test = 1402/248/414). This generated root was removed on 2026-07-13 after the JUFE route was closed.
- Report: `reports/jufe_start_conditioned_smoke/summary.md`.
- Full-val `KLD_sph`: image-only semantic-static `0.4248`, actual-start residual `0.2608`, swapped-start residual `0.2430`.
- Decision: fail condition-correctness control. The residual branch helps, but correct startpoint use is not proven because swapped start is better.
- Next action: add paired good/bad condition-contrastive loss before formal Gate B training.

### Gate A — protocol repair

- [x] JUFE content-disjoint split: 180/39/39 groups.
- [x] Official Sitzmann scene-disjoint split: 16/3/3 scenes; four conditions stay together.
- [x] Salient360 frozen image-disjoint split: 59/13/13 images.
- [x] Split manifest contains group-level assignments; conditioned good/bad data must inherit them on rebuild.
- [x] `CC(x,x)=1` and affine-invariance tests pass.
- [x] Spherical normalization, horizontal-wrap and yaw-roll tests pass.
- [x] Uniform/equator/train-average predictions regenerated and evaluated on clean val.
- [x] Clean strong-static baseline rerun for 3 seeds and locked.
- [x] New final-test manifest frozen and not used for model evaluation.

### Gate B — conditioned signal

- [x] V7 actual condition evaluated on clean val for 3 seeds.
- [x] V7 actual beats static and swapped/random on condition-specific metrics with paired CIs.
- [x] Efficiency, condition sensitivity and residual magnitude reported.

### Gate C — image-only contribution

- [x] V8 image-only marginalization implemented with one shared backbone.
- [x] One-seed static, ordinary temporal auxiliary, and random-condition pilot controls complete.
- [x] Pilot decision: FAIL; therefore random partitions and three-seed expansion are not authorized.
- [x] Final test remains unopened.

### Gate D — paper comparison

- [x] TempSAL-style same-protocol baseline.
- [x] ScanDMM-to-saliency matched post-processing.
- [x] Strong static 360° baseline.
- [x] Main table, ablation, efficiency, and qualitative figure complete.

## Historical Reports

Existing `reports/f2_*` and `reports/multi360_*` artifacts are retained as `pre_protocol_diagnostic`. They explain the protocol revision but are not copied into the paper main table.

## Maintenance Rule

After each experiment batch, record:

1. `claim_tested`;
2. Gate and decision threshold;
3. locked baseline and matched control;
4. val result with seeds/CI;
5. pass/fail decision and the single next action.
