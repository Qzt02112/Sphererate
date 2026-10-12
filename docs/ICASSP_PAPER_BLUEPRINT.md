# SphereHorizon ICASSP paper blueprint

更新时间：2026-07-12

> **Draft assembled.** 完整 provisional ICASSP 稿件见 [paper/main.tex](../paper/main.tex)；双源引用审计见 [paper/LITERATURE_VERIFICATION.md](../paper/LITERATURE_VERIFICATION.md)，五页版式审核见 [paper/DRAFT_AUDIT.md](../paper/DRAFT_AUDIT.md)。当前为官方 2026 kit 的临时版式，待 2027 kit 与匿名规则发布后迁移。

## Recommended title

> **SphereHorizon: Start-Aligned Early Attention Forecasting for 360° Images**

Alternative, more task-forward title:

> **Beyond Final Saliency: Early Attention Forecasting for 360° Tile Prefetching**

Do not use “general temporal saliency”, “arbitrary initial orientation”, “scanpath prediction”, “personalized attention”, “final-map improvement” or “state of the art” in the title.

## One-sentence contribution

We define and validate start-aligned early cumulative saliency for 360° images, then show that a lightweight image-conditioned residual improves map accuracy and tile-prefetch coverage beyond temporal-map, scanpath-to-map, static-final and reachability-only controls without using gaze history.

## Five-sentence abstract skeleton

1. Existing 360° saliency models mostly predict a final attention map, which cannot represent the strong start-dependent evolution of attention during the first seconds of viewing.
2. We formulate start-aligned early cumulative saliency at `0.5/1/3/5 s`, construct spherical dwell-weighted targets, and audit whether these targets retain sufficient observer agreement and content specificity.
3. Based on the observed transition from a generic start prior to content-dependent attention, SphereHorizon combines a train-only early template with a time-gated image correction and produces all four maps in about `9.2 ms` without gaze history.
4. On a content-disjoint Salient360 development benchmark, SphereHorizon improves spherical KLD/SIM/CC over a matched TempSAL adaptation and remains substantially better than reachability-filtered static saliency and same-split ScanDMM/Pathformer3D adaptations.
5. At a 10% tile budget, the model improves absolute gaze coverage over the early template by `0.0173`, recovering `11.2%` of its remaining miss mass, while external ET-DK2 results delimit the claim to canonical/equatorial early attention rather than universal cross-dataset generalization.

The abstract must label the main split as development evaluation unless a final submission policy explicitly defines another term. Add verified citations only after the related-work pass; never create BibTeX from memory.

## Story order

```text
final saliency is temporally mismatched to early tile prefetch
    ↓
early cumulative maps are meaningful but strongly start-biased
    ↓
reachability is necessary and surprisingly strong
    ↓
reachability alone cannot perform content-specific re-ranking
    ↓
a minimal template + time-gated image correction is sufficient
    ↓
map accuracy improves; application gain is modest but measurable
```

This order prevents the paper from being read as “another temporal branch”.

## Main contributions

1. **Task and target.** A start-aligned early cumulative population-saliency formulation with real-time clipped-dwell targets and spherical 2° rendering, together with observer reliability and leakage audits.
2. **Minimal model.** A train-only early template plus time-gated image-conditioned log-density correction that explicitly models the template-to-content transition without recurrent rollout or gaze history.
3. **Application and controls.** Fixed-budget tile-prefetch evaluation, a train-calibrated reachable-final control, and same-split temporal/static/scanpath baselines showing that support constraints alone do not solve content-specific early attention.

Do not list “continuous time encoding” as a contribution because the capacity control does not support it.

## Section outline

### 1. Introduction

- Paragraph 1: 360° streaming must choose tiles before users explore the panorama; final saliency answers the wrong temporal question.
- Paragraph 2: early prediction is difficult because all viewers share a reset point, creating a strong generic prior that can hide whether the model uses image content.
- Paragraph 3: define cumulative maps and preview the empirical finding—reachability captures much of coarse coverage but not content re-ranking.
- Paragraph 4: introduce SphereHorizon as the minimal model implied by that finding.
- Paragraph 5: give the three contribution bullets above.

### 2. Related work

Organize by assumptions, not paper chronology:

1. static 360° saliency and ERP-aware modeling;
2. temporal saliency volumes/maps, especially SaltiNet and TempSAL;
3. 360° scanpath generation, including ScanDMM and Pathformer3D;
4. viewport/tile prefetch driven by attention prediction.

Each paragraph must end with the assumption gap: final map vs early cumulative target, history-free population map vs scanpath rollout, or image semantics vs pure reachability. All citations remain `[CITATION NEEDED]` until programmatically verified.

### 3. Problem and target construction

- Define legal input `(I, g_0, t)` and output `C_t`.
- Explain yaw alignment and the canonical `+90°` Salient360 convention.
- Define clipped-dwell accumulation and 2° spherical chord Gaussian.
- State spherical normalization and metrics.
- Explain primary `0.5/1/3 s` macro and 5 s guard.
- Define 8×16 tile grid and gaze-mass coverage.

### 4. SphereHorizon

- Present the implemented log-density form `C_t = Softmax_sph(log T_t + β_t Δ_t(I,g_0))`.
- Explain train-only/LOO template construction.
- Explain shared visual trunk, bounded residual and horizon conditioning.
- State 3.69M total / 0.91M trainable parameters.
- Emphasize no gaze history, first fixation, observer ID or autoregressive rollout.

### 5. Experimental protocol

- Data roles: Salient360 primary, ET-DK2 external diagnostic, JUFE historical-only.
- Split roles: 50 train, 9 val, 13 historically seen development gate, 13 outer contents already used under v1.
- Baseline tiers: native map, reachability-filtered static, secondary scanpath-to-map.
- Start fairness: ScanDMM native initial 3D interface; Pathformer declared-start decoder token.
- Statistics: canonical-content paired bootstrap, 10,000 resamples, three seeds.
- Audit: zero v2-test target loads and `134/134 PASS`; corrected ScanDMM uses independent declared-start conditioning in model/guide. Label DeepGaze MSDB as `ICCV 2025, ERP-adapted`, not native spherical SOTA.

### 6. Results

#### 6.1 Early-map accuracy and tile coverage

Lead with the matched benchmark and [quantitative figure](../reports/kernel_v2_matched_quantitative/index.md). State map-metric wins over TempSAL; state that TempSAL coverage is comparable rather than significantly worse.

#### 6.2 Is reachability sufficient?

Use the raw→reachable arrows and the GBVS360+reachable gap. The key sentence is: “Reachability constrains where attention can occur, whereas the image residual re-ranks which reachable regions attract attention.”

#### 6.3 Content attribution and qualitative cases

Use P55/P10/P97 from [qualitative figure](../reports/kernel_v2_matched_qualitative/index.md). Do not call all examples successes: P55 shows template/reachability can be nearly sufficient, P10 shows re-ranking, P97 is a difficult content case.

#### 6.4 Efficiency

Report native map models together; report scanpath workload separately. Use “prepared-input four-map forward” rather than end-to-end system latency.

#### 6.5 External diagnostic and limitations

State ET-DK2 correct-vs-wrong evidence and candidate-vs-template failure in the results, not only in limitations.

### 7. Conclusion

Restate the task insight: early 360° attention combines a strong start-dependent support constraint with weaker but necessary content re-ranking. Future work should address arbitrary starting latitude, stronger cross-dataset population targets and online personalization only when history is available.

## Figure and table placement

| Order | Artifact | Purpose |
|---:|---|---|
| Fig. 1 | [spherehorizon_overview.pdf](../reports/spherehorizon_overview/spherehorizon_overview.pdf) | Final-map mismatch, start alignment, log-density residual and tile output |
| Fig. 2 | [horizon_behavior.pdf](../reports/kernel_v2_horizon_behavior/horizon_behavior.pdf) | Reveal when content improves map fidelity and fixed-budget tile selection, including the 5-s failure guard |
| Fig. 3 | [matched_qualitative_3s.pdf](../reports/kernel_v2_matched_qualitative/matched_qualitative_3s.pdf) | Show matched tile decisions, successful re-ranking cases and an informative failure |
| Evidence package | [matched_gate_benchmark.pdf](../reports/kernel_v2_matched_quantitative/matched_gate_benchmark.pdf) | Paired-CI quantitative visualization retained outside the page-limited main draft |
| Table 1 | matched benchmark | KLD/SIM/CC/coverage by tier |
| Table 2 | compact efficiency + parameters | Operational cost without cross-workload ranking |
| Appendix | source audit, ET-DK2, kernel sensitivity and expanded budgets | Reproducibility and limitations |

## Main-table rules

- Separate native temporal/map methods from scanpath-to-map adaptations with a horizontal rule or distinct subtables.
- Report early template, SphereHorizon, TempSAL-ERP, strongest reachable-final method, Pathformer3D and ScanDMM in the compact main table.
- Put raw BMS/GBVS/Panoramic rows in the expanded table or appendix, while Fig. 2c preserves their reachable effect.
- Bold only actual best values; do not decorate non-significant differences as wins.
- Add input columns: image, start, horizon, history; and a time-semantics footnote for scanpath models.

## Claims that must appear as limitations

- Salient360 has one canonical start; arbitrary 3D initial orientation is untested.
- The v2 main split is a historically seen development gate, not a fresh blind test.
- ET-DK2 does not establish stable improvement over the early template.
- The primary result is limited to 0.5/1/3 s; 5 s is a guard.
- TempSAL-ERP has comparable 10% coverage despite weaker map accuracy.
- Scanpath baselines use a first-second approximation at 0.5 s.
- The target captures group attention, not individual trajectories.

## Current submission-readiness decision

The complete provisional draft is assembled and visually audited. The strongest ICASSP framing remains an insight/task/application paper supported by a minimal method, not an architecture-first SOTA paper. Current reviewer trajectory is borderline-to-weak-accept if the narrow claims, reachability analysis, development-gate disclosure and ET-DK2 failure remain visible. A frozen v2 outer-test run may add completeness but cannot create a fresh-test claim because those contents were already inspected under v1.
