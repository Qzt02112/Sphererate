# SphereRate / SphereHorizon experiment log

更新时间：2026-08-06

## Contribution in one sentence

SphereRate reframes 360° saliency as start-aligned early cumulative population attention and, without gaze history, improves density accuracy and medium-budget attention-weighted HEVC reconstruction; the tightest 5% real-byte allocation remains worse than TempSAL-ERP.

## Current task

```text
input  = ERP RGB + ERP geometry + declared equatorial yaw + horizon t
output = cumulative population saliency C_t, t ∈ {0.5, 1, 3, 5} s
use    = rank 8×16 ERP tiles under 5%/10%/15%/20% budgets
primary= equal-weight 0.5/1/3/5 s early macro, 5/10/15/20% budgets
```

Target renderer：unit-sphere chord Gaussian, `σ=2°`, `4σ` support, clipped-dwell weighting and spherical-area normalization. Primary eye is right; left-eye results are sensitivity only. Legal inference excludes gaze history, first fixation and observer identity.

## Experiments

### E16 — Real-codec application + matched mechanism ablation — COMPLETE

- **Codec data:** official Salient360 `Image_Stimuli.7z`, verified MD5 `ac1a471e058177705ea96120af19ff2c`; 85 unique content mappings and 13 standard-test native ERPs. Test original/processed alignment has zero circular shift and NCC `.996--.999`.
- **Coding setup:** native-resolution ERP, `8x16` independently decodable tiles, ffmpeg/libx265 HEVC-Intra YUV420, QP42/QP22; `3328` bitstreams encoded, decoded, byte-counted, and dimension-audited.
- **Equal-byte result:** SphereRate Gaze-PSNR vs TempSAL at 5/10/15/20% is `-.253/+.070/+.170/+.136 dB`; only the 5% coverage loss is statistically confirmed. Versus Early template, 10/15/20% gains are `+.342/+.509/+.392 dB`, all paired CIs positive.
- **Mechanism result:** removing start cost catastrophically degrades all metrics; removing rate accumulation degrades SIM/CC/Cov.@10/20 but slightly improves Cov.@5; removing tile losses has only tiny KLD/SIM/CC/Cov.@10 effects and neutral effects elsewhere.
- **Decision:** retain start cost and raw-mass accumulation as method innovations; describe tile loss as a small deployment regularizer. Real application claim is restricted to medium budgets; 5% is an explicit failure, not hidden.
- **Evidence:** [codec](../reports/codec_tile_prefetch_test/summary.md), [ablation](../reports/sphererate_ablation_test/summary.md), [paper PDF](../output/pdf/SphereHorizon_ICASSP_draft.pdf).

### E15 — Standard 59/13/13 rerun — COMPLETE

- **协议修订：** 原 train+dev 合并为 59-content train，原 13-content gate 改作 val，原 13-content test 保持 test。所有 learned methods 只在 train 拟合、按 val 选 checkpoint；test 可重复评估，但不参与任何选择。
- **统一口径：** spherical 2° target、0.5/1/3-s early macro、5-s guard、8×16 tiles、Cov.@5/10/15/20、三种子和 content-grouped bootstrap。
- **数据审计：** 85 行严格为 59/13/13；train IDs 等于旧 train+dev，val IDs 等于旧 gate，test IDs 不变；train template 对训练内容 LOO，对 val/test 仅聚合 59 train；审计 PASS。
- **主模型 test：** V12-C SphereRate 的 early KLD/SIM/CC/Cov.@5/10/15/20 为 `1.0736/.4702/.6151/.7313/.8484/.9059/.9344`；TempSAL-ERP 为 `1.1106/.4477/.6067/.7325/.8404/.9003/.9316`。唯一反向项是 Cov.@5 `-0.0012`。
- **5 s test：** V12-C 为 `1.0672/.4461/.4921/.3444/.5151/.6408/.7183`，七项均优于 TempSAL。
- **完整对比：** Early Template、TempSAL-ERP、Panoramic-CNN、BMS360、GBVS360、DeepGaze MSDB、reachable controls、Pathformer3D、ScanDMM 和 HAT 已全部完成。ScanDMM 的 val-selected test early KLD/SIM/CC/Cov.@5/10/15/20 为 `1.8747/.3924/.4790/.7070/.8149/.8685/.8955`；HAT 为 `1.8885/.3251/.4451/.7036/.8128/.8769/.9057`，均低于 V12-C。
- **完整性：** unified benchmark 无缺失行；标准协议审计 PASS；全量测试 `251 passed`。
- **Evidence:** [data audit](../reports/standard_split_data_audit/summary.md), [primary test](../reports/standard_split_primary_test/summary.md), [full benchmark](../reports/standard_split_full_benchmark_test/summary.md), [protocol audit](../reports/standard_split_protocol_audit/summary.md).

以下 E1--E14 保留为历史决策链；其中 `50/9/gate/outer`、sealed、once-only 和 frozen-test 访问规则均不再约束 E15。

### E1 — Saliency-kernel audit

- **Claim tested:** the legacy 10 px ERP kernel changes the scientific target and suppresses content-specific structure.
- **Setup:** 59 train/val contents only; compare legacy fixed-pixel kernel with 1°/2°/4° spherical kernels; no neural training and zero gate/test target access.
- **Key result:** legacy vs 2° target mean `SIM=0.605`, while legacy effective area is about `2.0–2.6×` larger. Switching to 2° increases target `1−SIM` from the train-only template by about `0.22–0.29`.
- **Decision:** rebuild all targets and retrain all learned methods; v1 numbers are historical only.
- **Evidence:** [kernel audit](../reports/saliency_kernel_audit/summary.md).

### E2 — Kernel-v2 data and leakage audit

- **Claim tested:** the corrected targets and templates are reproducible without content leakage.
- **Setup:** `50 train / 9 val / 13 development-gate / 13 sealed outer-test`; group-disjoint content roles; train-content templates use leave-one-out aggregation.
- **Key result:** 72 non-test targets rebuilt; independent reconstruction and spherical normalization pass; v2 test targets remain sealed.
- **Evidence:** [data audit](../reports/kernel_v2_data_audit/summary.md).

### E3 — Minimal start-template residual model

- **Claim tested:** image content adds predictable early-map information beyond the train-only start/time template.
- **Setup:** shared visual trunk plus bounded horizon-conditioned residual; no recurrent, transformer, diffusion or rollout structure; three seeds.
- **Key result on gate vs template:** KLD `+0.0432 [0.0219,0.0626]`, SIM `+0.0152 [0.0072,0.0223]`, CC `+0.0175 [0.0078,0.0260]`, 10% coverage `+0.0173 [0.0077,0.0269]`.
- **Interpretation:** the improvement is consistent but modest; architecture novelty is not the paper's central contribution.
- **Evidence:** [three-seed robustness](../reports/kernel_v2_robustness/summary.md).

### E4 — Attribution and capacity controls

- **Claim tested:** improvements depend on time, start and image content rather than parameter count or a universal template.
- **Setup:** shuffled label/time, wrong start, deterministic wrong image and fixed-code capacity control.
- **Key result:** wrong-time/start/image controls pass; candidate vs capacity-matched fixed code is not significant.
- **Interpretation:** the evidence supports content use and the early-task formulation, but not a claim that continuous time encoding is superior.
- **Evidence:** [kernel-v2 benchmark](../reports/kernel_v2_benchmark/summary.md), [content attribution](../reports/kernel_v2_content_audit/index.md).

### E5 — Train-only reachable-final control

- **Claim tested:** whether the task can be solved by filtering a final saliency map to the region physically reachable from the declared start.
- **Setup:** right-eye clipped-dwell q95 great-circle radii from 50 train contents only; frozen radii `39.39°/44.42°/97.84°/125.03°`; hard gate applied to Panoramic-CNN/BMS360/GBVS360.
- **Key result:** GBVS360 improves from gate KLD `2.7008` to `1.6939`, but SphereHorizon remains better by KLD `0.6323 [0.5221,0.7542]`, SIM `0.1281 [0.1051,0.1548]` and CC `0.2055 [0.1677,0.2531]`.
- **Surprising finding:** reachability explains much of the coverage benefit, but not accurate content re-ranking.
- **Evidence:** [reachable-final](../reports/kernel_v2_reachable_final/summary.md).

### E6 — Matched public baselines

- **Claim tested:** SphereHorizon remains competitive against task-matched temporal maps and published scanpath/static methods under the same split.
- **Setup:** TempSAL-ERP, Panoramic-CNN, BMS360, GBVS360, official DeepGaze MSDB (ICCV 2025) with an explicit ERP adapter, and three-seed ScanDMM, Pathformer3D, and HAT; conditionable methods receive the declared start through their native model/history interface; all learned/calibrated methods use the same 50/9 split.
- **Gate results:** SphereHorizon `1.0616/0.4763/0.6163/0.8626` KLD/SIM/CC/10%-coverage; TempSAL `1.1287/0.4496/0.5762/0.8604`; DeepGaze MSDB calibrated+reachable `1.5191/0.3558/0.4570/0.8263`; Pathformer `1.9143/0.3576/0.4311/0.8215`; corrected ScanDMM `1.8030/0.3949/0.4819/0.8389`; HAT `1.7243/0.3531/0.4674/0.8314`.
- **Statistical reading:** map accuracy improves significantly over TempSAL; 10% coverage vs TempSAL is not significant. SphereHorizon improves KLD/SIM/CC/coverage over HAT by `0.6627/0.1232/0.1489/0.0312`, with all paired 95% intervals above zero. HAT's lower KLD than the other scanpath adapters does not transfer to SIM, CC, or coverage, exposing a trajectory-to-population calibration mismatch rather than a simple model-capacity ranking.
- **Evidence:** [matched benchmark](../reports/kernel_v2_matched_baselines/summary.md), [integrity audit](../reports/kernel_v2_matched_baseline_audit/summary.md).

### E6b — Recent-model and scanpath-sampling audits

- **Recent source audit:** among 2025–2026 candidates, DeepGaze MSDB is the only current strong fixation-density model with official code/checkpoint that can be adapted without changing the task. IMRE is the best native-ODI match but lacks public code; CASP/SalViT360-AV require video/audio; BLIP2OISal predicts distortion-aware regions rather than free-viewing fixation density.
- **MSDB adapter:** official weights frozen; four yaw rolls; two train-only calibration scalars; raw and reachable both retained. The planar-model ppd caveat is explicit.
- **Sampling decision:** use 200 paths from fixed seed 20260711 for representative population-map figures, 30 for individual trajectory motion, and three-seed metric averaging for formal results. A 600-path ensemble is a stability reference, not a representative single output.
- **Evidence:** [recent-model audit](../reports/recent_saliency_2025_2026_source_audit/summary.md), [MSDB gate](../reports/kernel_v2_deepgaze_msdb_gate/summary.md), [sample-count audit](../reports/kernel_v2_scanpath_sample_count_audit/summary.md).

### E6c — ScanDMM declared-start training correction

- **Problem:** official ScanDMM uses the first one-second gaze both to initialize `z_prev` and supervise `obs_x_1`; inference can accept a declared start, but the old same-split training row was not independently start-conditioned.
- **Correction:** a state-dict-compatible adapter passes a separate `(0°, +90°)` gaze tensor to both model and guide, while retaining the original 20 supervised future gaze points.
- **Semantic audit:** PASS. State keys match official ScanDMM, `obs_x_1` remains unchanged, and changing only the start changes the first latent distribution. The train/val first target is a median `15.01°/12.30°` from the declared start.
- **Status:** complete. Three frozen seeds selected epochs `399/374/349` on val only; 200-path val/gate evaluation is complete. After adding HAT, the unified matched-suite integrity audit is `188/188 PASS`. The old ScanDMM row remains diagnostic only.
- **Evidence:** [declared-start audit](../reports/scandmm_declared_start_semantics_audit/summary.md).

### E7 — Application effect size

- **Claim tested:** whether the small absolute coverage gain is meaningful relative to the remaining uncovered gaze mass.
- **Setup:** derived ratio `(coverage_ours−coverage_control)/(1−coverage_control)` with the same content-paired 10,000-resample bootstrap.
- **Key result:** vs early template, absolute coverage `+0.0173` corresponds to `11.2% [5.0%,17.2%]` remaining miss-mass reduction.
- **Boundary:** derived interpretation only; it does not replace the preregistered absolute coverage endpoint.
- **Evidence:** [miss reduction](../reports/kernel_v2_miss_reduction/summary.md).

### E8 — Efficiency

- **Claim tested:** whether the minimal map predictor is operationally lightweight.
- **Setup:** RTX 5880 Ada FP32 prepared-input wall-clock; native output workload retained for each tier.
- **Key result:** SphereHorizon produces four 256×512 maps in `9.20±0.04 ms`, peak `468.1 MiB`, with 3.69M total/0.91M trainable parameters. TempSAL takes `6.69±0.02 ms`. DeepGaze MSDB's four-yaw ERP ensemble takes `1173.3±13.5 ms` and 451.64M frozen parameters. Corrected ScanDMM/Pathformer generate 200×20 scanpath points in `174.3/517.8 ms`; different output tiers are not used for direct architecture-only speed ranking.
- **Evidence:** [efficiency](../reports/kernel_v2_matched_efficiency/summary.md).

### E9 — External multi-start diagnostic

- **Claim tested:** whether content use and template improvement remain stable across real multi-start ET-DK2 sessions.
- **Setup:** 126 sessions, 7 content groups, 8 equatorial starts; content-cluster bootstrap.
- **Key result:** correct vs wrong start `+0.7829 [0.7633,0.7998]`; correct vs wrong image `+0.0112 [0.0002,0.0225]`; candidate vs template `+0.0100 [-0.0043,0.0238]`.
- **Decision:** condition sensitivity and image use are supported; stable cross-dataset improvement is not.
- **Evidence:** [ET-DK2](../reports/kernel_v2_etdk2/summary.md).

### E10 — Sitzmann frozen transfer — INVALIDATED / REMOVED

- **Reason:** the local ScanDMM-derived pickle discarded the official per-trial `viewpoint_idx`. Inferring three pseudo-start sectors from first-second gaze is not equivalent to the dataset's four assigned starting viewports.
- **Additional data issue:** the processed training view contains rotation-derived rows whose provenance is insufficient for a start-aware frozen benchmark.
- **Decision:** all previous E10 scores are void and must not be cited. The evaluation script, report, combined manifest, and derived Sitzmann processed root were removed on 2026-07-13.
- **Re-entry condition:** fulfilled by E11; E10 itself remains invalid and must never be cited.

### E11 — Official Sitzmann reconstruction and train-source decision

- **Question:** does the official four-start dataset fit the task well enough to replace Salient360 as the primary training set?
- **Reconstruction:** read only `data/vr/cubemap_*.pck` from `sitzmann_raw_data.zip`; exclude browser/seated; retain trial subject, timestamps and `viewpoint_idx`; align yaw `{0°,90°,180°,270°}` to ERP center; build 2° spherical clipped-dwell targets. The output has 22 scenes, 88 scene-start cells, 780 trials and 774 cell-level physical observers after merging six repeated-subject trials.
- **Split:** 16 train / 3 val / 3 published-test gate scenes. The four conditions of a scene stay together; published test is not fresh.
- **Data finding:** Sitzmann has stronger controlled early agreement in CC (`0.5213` vs Salient360 `0.4427`) but worse split-half KLD (`3.8857` vs `3.4002`). Its 544 train observers are divided across 64 targets with a median of 8 observers, while Salient360 provides 1,857 observers across 50 targets with a median of 36.5 and 50 unique training scenes.
- **Controlled three-seed result:** in-domain Sitzmann scratch improves template KLD `1.3273→1.1880±0.0057`, but CC only `0.7459→0.7493±0.0041` and validation degrades after epochs 3–7. With the target train-only template held fixed, Salient360 residual transfers to Sitzmann (`1.2263±0.0064 / 0.7518±0.0003` KLD/CC), whereas Sitzmann residual provides little transfer to Salient360 (`1.0987±0.0046 / 0.5997±0.0013`).
- **Adaptation:** Salient360→Sitzmann fine-tuning reaches `1.1965±0.0027 / 0.7569±0.0013`, trading slightly worse KLD than scratch for higher CC and retaining Salient360 performance.
- **Preliminary decision at E11:** keep Salient360 as primary training source and use Sitzmann as official real-multi-start external evidence. The conservative “do not rerun the complete suite” clause was later superseded by the user's explicit authorization of E12; the primary-source conclusion itself remained unchanged.
- **Evidence:** [training suitability](../reports/sitzmann_training_suitability/summary.md), [controlled train-source comparison](../reports/sitzmann_train_source_comparison/summary.md).

### E12 — Full Sitzmann-only suite and bidirectional frozen transfer

- **Question:** does E11's source decision survive a complete matched suite, including scanpath models that natively receive the declared start?
- **Protocol:** train SphereHorizon, TempSAL, Panoramic-CNN, ScanDMM and Pathformer3D with three seeds on Sitzmann-only; select checkpoints only on source validation; evaluate all learned models in the four source→target cells. Coordinate-aligned cross views rotate RGB, targets, templates and scanpaths together. BMS360, GBVS360, DeepGaze MSDB and source-calibrated reachable controls use the same target protocol.
- **Dense trajectory correction:** official `gaze_lat_lon/timestamp` is sampled at integer `0..29 s`; all 780/780 trials are preserved, with 547/117/116 train/val/gate trials and zero repaired bins. Fixation intervals remain the source of cumulative saliency-map GT.
- **Map-model result:** SphereHorizon KLD for `Sal→Sal / Sal→Sitz / Sitz→Sitz / Sitz→Sal` is `1.0616 / 1.6306 / 1.1880 / 1.8157`; TempSAL is `1.1287 / 1.6538 / 1.1445 / 1.9349`. Sitzmann-only training therefore helps these task-matched models in-domain, but transfers poorly back to Salient360.
- **Scanpath result:** Salient360-trained ScanDMM and Pathformer3D outperform their Sitzmann-only counterparts even on Sitzmann (`2.4317` vs `2.7480`, and `2.5343` vs `3.9220` KLD). The declared-start/output audit passes, so this is not caused by omitting the start interface. It is consistent with Sitzmann's limited scene diversity and with a mismatch between dense-trajectory supervision and fixation-derived cumulative targets.
- **Controls:** on Sitzmann, raw BMS/GBVS give about `3.33/3.19` KLD; reachable variants improve to about `2.01`, and reachable DeepGaze reaches `1.9326`, but all remain behind the direct early-map models.
- **Application metric:** on the frozen 8×16 grid, Sitzmann in-domain `Cov@5/10/20%` is `0.8309/0.9047/0.9737` for SphereHorizon and `0.8459/0.9150/0.9674` for TempSAL. Coverage and density fidelity therefore give complementary rankings; the application claim is competitive coverage and clear improvement over ungated static-final maps, not uniform dominance over TempSAL at every budget.
- **Decision:** retain Salient360 as the primary content-diverse benchmark. Promote Sitzmann from a diagnostic-only role to a secondary in-domain matched track plus bidirectional frozen-transfer evidence. Do not replace Salient360 with Sitzmann as the sole source.
- **Integrity:** zero content leakage, source-val-only selection, 780/780 dense trials, canonical scene aggregation, finite metrics and sealed target loads=`0`; all checks pass.
- **Evidence:** [full benchmark](../reports/sitzmann_cross_dataset_benchmark/summary.md), [machine-readable audit](../reports/sitzmann_cross_dataset_benchmark/integrity_audit.json).

### E13 — Semantic re-ranking and low-budget Coverage optimization

- **Question:** can stronger image semantics increase both the gap over the early template and Cov@5/10 relative to TempSAL without adding a complex temporal model?
- **Protocol:** Salient360 train-only design and val-only selection; fixed 8×16 grid and 5%/10% primary budgets; KLD guard `1.10999`; no gate, v2 outer-test, or Sitzmann target access.
- **S0:** same-image GT-final oracle passes at 3 s, so semantic re-ranking headroom exists; wrong-image and frozen DeepGaze controls separate correspondence from generic saliency.
- **S1:** removing the early scalar suppression and adding log-ratio/tile losses yields val `KLD=1.1192`, `Cov@5/10=0.7437/0.8339`. It concentrates the top tiles but fails KLD, 10%, and attribution gates.
- **S2:** a transferred matched-static semantic anchor, frozen encoder, 74k-parameter shallow head, and four monotonic alphas yields `KLD/CC=1.0722/0.6178`, `Cov@5/10=0.7303/0.8441`. Correct-minus-wrong-image at 3 s is significant for KLD/CC and Cov@10, but Cov@5 remains below both references.
- **Headroom stop:** train-selected S1/S2 product fusion reaches `KLD=1.1045`, `Cov@5/10=0.7459/0.8369`; the two budgets still cannot pass jointly. No third head or additional weight search is authorized.
- **Metric correction:** EH3 validation previously averaged unequal batch means. KLD/CC now use per-content aggregation; Coverage was already per content. The correction changes S2 map fidelity to PASS but not its Cov@5 failure.
- **Decision:** retain V10. Use S2 only as an ablation showing that stronger semantic density and attribution do not automatically solve the lowest-budget tile-ranking endpoint.
- **Evidence:** [final decision](../reports/semantic_reranking_final/summary.md).

### E14 — V12-C frozen external transfer

- **Question:** do the standalone V12-C gains over TempSAL survive without any target-domain calibration?
- **Freeze:** the exact three epoch-1 checkpoints selected on Salient360 val were hashed before use. Sitzmann and ET-DK2 evaluators share the same hashes; no checkpoint, geometry, loss, tile grid, budget or target prior was changed.
- **Sitzmann:** on the three historically seen official published-test scenes, V12-C early KLD/SIM/CC/Cov@5/10/20 is `1.7485/.2626/.4586/.8209/.9018/.9615`, below frozen TempSAL `1.6538/.2873/.4808/.8514/.9107/.9713`. Correct-vs-wrong-image remains significant, so image use transfers but ranking quality does not.
- **ET-DK2 protocol correction:** the local ET-DK2 split is not synchronized with Salient360. The evaluator excludes 36 source-train sessions and 18 sealed-outer sessions, retaining only 54 sessions from P13/P3/P76. This is a cross-start/collection-protocol diagnostic, not independent cross-content evidence.
- **ET-DK2 result:** early CC improves by `+.0160`, but Cov@5/10/20 changes by `-.0174/-.0031/-.0074`; 5 s Cov@20 improves `+.0174`. Wrong-start attribution is strong while wrong-image is unstable.
- **Decision:** frozen cross-dataset dominance over TempSAL fails. Retain the Salient360 in-domain claim and report external results as a limitation. External feedback tuning is forbidden.
- **Integrity:** all checkpoint hashes match; target-domain calibration, ET-DK2 test loads and Salient360 outer loads are zero.
- **Evidence:** [overall](../reports/v12c_frozen_transfer/summary.md), [integrity](../reports/v12c_frozen_transfer/integrity_audit.md).

### E15 — SphereRate Sitzmann-only standard train/val/test

- **Question:** can same-domain Sitzmann training remove the frozen-transfer deficit without using a cross-dataset comparison?
- **Protocol:** official standing-VR data only; `16 train / 3 val / 3 test` canonical scenes and four aligned starts per scene. A coordinate audit corrected the model-space start from the historical `180°` label to ERP-centre `0°`. Static and semantic anchors are trained on the 16 scenes; three SphereRate checkpoints are selected only on the three validation scenes.
- **Train-only calibration:** clipped dwell gives first-interval east/north standard deviations `5.845°/6.909°` and second-interval `18.601°/16.364°`; validation/test target loads during calibration are zero.
- **Test result:** early SphereRate KLD/SIM/CC is `1.2098/.4253/.6493` versus TempSAL `1.1445/.4937/.7078`. Cov.@5/10/15/20 is `.8486/.9203/.9553/.9774` versus `.8459/.9150/.9524/.9674`. At 5 s, SphereRate improves KLD `1.7844→1.6598` and all four Coverage point estimates, while SIM/CC remain lower.
- **Attribution:** 3 s correct-image over wrong-image KLD/SIM/CC gains are `+.5879/+.0637/+.0955`, with all content-bootstrap intervals positive. The density deficit is not explained by ignoring image content.
- **Decision:** same-domain training repairs much of the transfer loss and improves tile ranking, but does not establish overall saliency superiority. Retain as a supplementary in-domain result; stop active cross-dataset comparison and do not replace the Salient360 main benchmark.
- **Evidence:** [Sitzmann-only report](../reports/sitzmann_standard_v2/summary.md).

## Figure inventory

| Artifact | Role in paper | Main takeaway |
|---|---|---|
| [matched_gate_benchmark.pdf](../reports/kernel_v2_matched_quantitative/matched_gate_benchmark.pdf) | Main quantitative figure | Reachability is strong but insufficient; SphereHorizon improves map accuracy and coverage over the selected controls |
| [matched_qualitative_3s.pdf](../reports/kernel_v2_matched_qualitative/matched_qualitative_3s.pdf) | Main qualitative or supplementary figure | Static reachability limits support, while the image-conditioned model re-ranks content regions |
| [kernel_v2_content_attribution_3s_display_smoothed_10px.png](../reports/kernel_v2_content_audit/kernel_v2_content_attribution_3s_display_smoothed_10px.png) | Attribution supplement | Correct/wrong-image residuals show non-zero image dependence; smoothing is display-only |

## Table inventory

| Table | Source | Placement |
|---|---|---|
| Main matched benchmark | [summary](../reports/kernel_v2_matched_baselines/summary.md) | Main paper |
| Efficiency by native workload | [summary](../reports/kernel_v2_matched_efficiency/summary.md) | Main or compact supplementary table |
| Reachability raw vs gated | [summary](../reports/kernel_v2_reachable_final/summary.md) | Main ablation/control table |
| Per-horizon and per-budget results | machine-readable result JSONs | Appendix |
| Source/split/start audit | [188/188 PASS](../reports/kernel_v2_matched_baseline_audit/summary.md) | Appendix/reproducibility |

## Failed or closed branches

- Temporal supervision does not improve final saliency under shuffled controls (`P2 FAIL`).
- JUFE conditioned final saliency is dominated by a learned start-only template; content fusion is not stable (`P1/R0 FAIL`).
- ET-DK2 does not establish candidate-over-template external improvement under kernel-v2.
- Continuous horizon encoding is not significantly better than fixed horizon codes.
- 5 s is not a supported primary improvement horizon.
- SalDiff-DTM has no verified official source; the local hash-ID reproduction cannot generalize to content-disjoint images.
- SaltiNet official weights have unresolved Salient360 content overlap and remain diagnostic only.
- Semantic re-ranking S1/S2 cannot jointly pass KLD, Cov@5, Cov@10, and attribution on val; architecture expansion is closed.

## Open decision

V12-C frozen external transfer is complete and does not establish overall superiority to TempSAL. The outer split remains permanently closed because it was already read under invalid v1 targets. Remaining authorized work is application accounting, paper integration and explicitly preregistered ablations; any future domain-robust training requires a new legal split and cannot tune on the reported external results.
