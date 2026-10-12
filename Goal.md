# SphereRate Goal and Short-Term Roadmap

更新时间：2026-08-06

> **口径说明（2026-10-12 补）。** 本文件 "V3 标准划分重跑" 一节报告的是**三时刻 early macro（0.5/1/3 s）**，再单列一个 **5 s** 行；当前论文口径（[docs/current_effective_framework.md](docs/current_effective_framework.md) 与 README）则把 `0.5/1/3/5 s` 四个时刻**一起**平均。两者是同一份数据的两种聚合，并非两组结果：`(3 × 三时刻 macro + 5 s) / 4` 精确复现 framework 数值（例：KLD `(3×1.0736 + 1.0672)/4 = 1.072`；`Cov.@5` `(3×0.7313 + 0.3444)/4 = 0.635`）。**以 framework 文档为准。**

> 当前唯一主线由 [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) 锁定。2026-08-05 起统一采用 Salient360 `59 train / 13 val / 13 test`：原 train+dev 拟合，原 gate 验证选模，原 test 最终报告。取消 frozen/once-only/sealed 限制；test 可重复评估但不参与选模。旧 `50/9/13/13` 结果只作历史记录。

> 2026-08-06 补充：停止把跨数据集冻结矩阵作为当前对比目标，改做 Sitzmann `16 train / 3 val / 3 test` 的同域训练测试。结果显示 SphereRate 的 test Cov.@5/10/15/20 均略高于 TempSAL，但 early SIM/CC 明显更低，故该实验只作同域补充，不替代 Salient360 主 benchmark。见 [Sitzmann-only 结论](reports/sitzmann_standard_v2/summary.md)。

> 2026-08-06 reviewer-closure：已完成真实 HEVC 编码/传输应用实验，以及 rate accumulation / declared start / tile supervision 三项 matched mechanism ablation。论文主张已据结果收紧，见 [当前有效框架](docs/current_effective_framework.md)。

> 2026-08-06 target-reliability 补证：旧 EH0 数字继续保持 historical-only；当前论文只使用在 `256×512 / 球面2°` V3 target 上重跑的结果。72个 train+val 内容、20次 observer split-half 显示四时刻 full-cohort CC estimate 为 `.516/.678/.611/.568`，10% tile normalized held-out coverage 为 `.845/.905/.827/.800`，且 same-horizon 相对 final-repeat 的 SIM/CC 改善均显著为正。证据见 [V3 target reliability](reports/standard_split_v3_target_reliability/summary.md)。

## 当前两个补证目标 — COMPLETE

### G1：真实编码/传输收益

- 已恢复官方 `Image_Stimuli.7z`，MD5=`ac1a471e058177705ea96120af19ff2c`；85/85 内容唯一映射，13 个 standard-test 原图与 512x256 网络输入的最佳循环位移均为 0。
- 13 个 test ERP 使用 native resolution（宽 `3584--18332 px`），切分为 `8x16` tiles；每个 tile 分别以 HEVC-Intra QP22/QP42 独立编码解码，共审计 `3328` 条真实 bitstreams。
- 在相同实际 byte cap 下，SphereRate 相对 Early template 的 Gaze-PSNR 在 10/15/20% 提高 `.342/.509/.392 dB`，paired CI 全部高于 0；相对 TempSAL 的点提升为 `.070/.170/.136 dB`，但 CI 跨 0。
- 5% 是明确失败：SphereRate 相对 TempSAL raw-dwell coverage 为 `-.0262 [-.0552,-.0032]`。因此应用主张锁定为 **medium-budget real-codec gain**，不得声称所有预算领先。
- 证据：[codec summary](reports/codec_tile_prefetch_test/summary.md)、[full results](reports/codec_tile_prefetch_test/results.json)。

### G2：rate/start/tile-loss 机制消融

所有变体保持相同 `3.76M total / 74K trainable` 参数、semantic anchor、三随机种子、25 epochs、optimizer 和 validation-only checkpoint selection。

| Variant | KLD↓ | SIM↑ | CC↑ | Cov.@5↑ | Cov.@10↑ | Cov.@15↑ | Cov.@20↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | **1.072** | **.464** | **.584** | .635 | **.765** | .840 | **.880** |
| w/o rate accumulation | 1.085 | .454 | .573 | **.640** | .749 | .837 | .874 |
| w/o declared start | 2.287 | .219 | .404 | .570 | .655 | .718 | .770 |
| w/o tile supervision | 1.075 | .463 | .583 | .635 | .764 | **.840** | .880 |

机制判定：declared start 是决定性机制；rate accumulation 改善 SIM/CC/Cov.@10/20，但不改善 Cov.@5；tile loss 只有很小且预算相关的收益，不再作为主要创新来源。证据：[ablation summary](reports/sphererate_ablation_test/summary.md)、[bootstrap results](reports/sphererate_ablation_test/results.json)。

### 后续唯一优化目标

1. 修复真实 byte cap 下的 5% 极紧预算排序；
2. 将 10/15/20% 相对 TempSAL 的正向点提升变成 content-bootstrap 可确认的收益；
3. 禁止以更大 backbone、复杂 temporal rollout 或 test-specific 调参绕开上述问题。

## 当前唯一推进目标

### V3 标准划分重跑（已完成）

1. 数据视图与零泄漏审计已完成：[standard split audit](reports/standard_split_data_audit/summary.md)。
2. 所有 learned methods 使用相同 59 个内容训练、13 个内容验证选 checkpoint、13 个内容测试；所有 target renderer、球面指标和 Cov.@5/10/15/20 保持一致。
3. 主方法与 TempSAL-ERP 的三种子重训已经完成，验证选择后的测试结果见 [primary test](reports/standard_split_primary_test/summary.md)。
4. ScanDMM、Pathformer3D、HAT、Panoramic-CNN 及公开/传统基线已按同一协议重跑；完整表无旧 split 数字或缺失行。
5. 最终输出同时包含 early macro 与 5 s 的 KLD/SIM/CC/Cov.@5/10/15/20，以及参数量、三种子波动和 content-grouped paired bootstrap。

研究现状、模型定义和结果边界统一见 [中文研究交付文档](docs/中文研究交付文档.md)。当前 V3 证据见 [validation](reports/standard_split_primary_val/summary.md)、[test](reports/standard_split_primary_test/summary.md) 与 [full benchmark](reports/standard_split_full_benchmark_test/summary.md)。当前目标状态为：

1. **独立方法成立：** V12-C 只输入 ERP、declared start 和 horizon；不读取 TempSAL 预测、early template 数值或 gaze history。四段非负 dwell-rate 在未归一化质量空间累计，球面语义效用与移动成本共同决定每段分布。
2. **标准 test 的 density 与中高预算领先：** V12-C 的 early KLD/SIM/CC/Cov.@5/10/15/20 为 `1.0736/.4702/.6151/.7313/.8484/.9059/.9344`；TempSAL-ERP 为 `1.1106/.4477/.6067/.7325/.8404/.9003/.9316`。除 Cov.@5 外，其余点估计更优。
3. **唯一反向项是 early Cov.@5：** 差值为 `-0.00116 [-0.01592,+0.01171]`。因此不得称为所有低预算稳定领先；继续针对 test 的 cutoff tiles 调参同样禁止。
4. **5 s 已从 no-op 变为有效预测：** test 5 s 的 KLD/SIM/CC/Cov.@5/10/15/20 为 `1.0672/.4461/.4921/.3444/.5151/.6408/.7183`，七项全部高于 TempSAL；KLD/SIM/CC/Cov.@15/20 的 paired CI 为正。
5. **冻结跨数据集总体领先未达成：** Salient360→Sitzmann early 六项均低于 TempSAL，Cov.@5 差 `-0.0305`；ET-DK2 合法 54-session 子集的 early Cov.@5/10/20 也低 `0.0174/0.0031/0.0074`。但 Sitzmann wrong-image 归因显著，ET-DK2 wrong-start 归因显著，说明“内容仍被使用”和“起点条件可泛化”不等于跨域排序更准。
6. **完整 benchmark 已完成：** ScanDMM 由 val 选中 epochs `399/424/324`，test early KLD/SIM/CC/Cov.@5/10/15/20 为 `1.8747/.3924/.4790/.7070/.8149/.8685/.8955`；HAT 由 val 选中 steps `3k/3k/5k`，对应 `1.8885/.3251/.4451/.7036/.8128/.8769/.9057`。两者均低于 V12-C；完整性审计 PASS、全量测试 `251 passed`。

当前报告预算统一为 Cov.@5/10/15/20；训练 loss 仍使用预先定义的 5/10/20，不因新增报告 Cov.@15 而改变。任何后续方法变化必须明确服务于跨域稳定性或真实应用收益，不能自动恢复已关闭路线。

## 0. 当前纠错目标（最高优先级）

旧 target 的 `sigma_px=10` 在 512 宽 ERP 上等价于约 `σ=7.03°`，明显大于 Sitzmann 的 `1° visual angle`，且不是球面等角核。59 个 train/val contents 的冻结审计得到：旧目标与 `2°` 球面目标的平均 `SIM=0.6050 / KLD=1.0553`，旧核有效扩散面积约为新核的 `2.0–2.6×`；改用球面核后，相对 train-only template 的 content-specific `1−SIM` 增加约 `0.22–0.29`。这足以影响训练、可视化和“模型是否使用图像内容”的判断，必须重做。

当前冻结决定：

```text
target renderer = unit-sphere chord Gaussian
sigma           = 2° visual angle
support          = 4 sigma
weighting        = clipped dwell duration
normalization    = spherical area
resolution       = 256 × 512
```

短期执行链：

1. **K0 — PASS：** 来源核对、球面实现、几何测试、train/val-only 影响审计；报告见 [reports/saliency_kernel_audit/summary.md](reports/saliency_kernel_audit/summary.md)。
2. **K1 — PASS：** 已构建 `early_horizon_salient360_spherical2deg_v2`，72 个 non-test targets 与 LOO train template 完成；零泄漏、球面积分、独立 rebuild 全通过，13 个 test rows 保持 sealed。报告见 [reports/kernel_v2_data_audit/summary.md](reports/kernel_v2_data_audit/summary.md)。
3. **K2 — PASS：** candidate、shuffled-label、capacity control、strong static、三 seed candidate、三 seed Panoramic-CNN、TempSAL-ERP、Pathformer3D 与 corrected ScanDMM 均已在 v2 同 split 完成训练/适配；所有 scanpath map adapter 统一使用 2°球面核。Corrected ScanDMM 在 model/guide 两侧使用独立 `initial_gaze` tensor，第一秒 gaze 仍只作为 `obs_x_1` target。
4. **K3 — PARTIAL PASS：** Salient360 已见 gate 三 seed相对 template 的 KLD/SIM/CC/10% coverage improvement 为 `0.04323/0.01516/0.01752/0.01731`，CI 均为正；wrong-time/start/image 全通过。ET-DK2 correct-vs-wrong-image 为正，但 candidate-vs-template coverage `0.01002 [-0.00433, 0.02385]` 跨零，外部稳定增益主张关闭。
5. **K4 — MATCHED DEVELOPMENT BENCHMARK PASS：** v2 val/gate 已完成 TempSAL-ERP、leak-free Panoramic-CNN、BMS360/GBVS360、2025 ICCV DeepGaze MSDB，以及三 seed corrected ScanDMM、Pathformer3D 和 HAT。`reachable-final` 使用 train-only 95% 球面可达半径。HAT 在同一 `50 train / 9 val` split 重训，以 declared start 作为首个 history token，但不把它作为监督目标；5k 预算由首种子 val 收敛预先冻结，三 seed 仅在 val 选择 `4k/5k/4k` checkpoint，每内容生成 200 条轨迹，fixation-event IOR 因 1 Hz 适配而关闭。HAT gate KLD/SIM/CC/Cov@5/10/20 为 `1.7243/0.3531/0.4674/0.7194/0.8314/0.9172`；SphereHorizon 的 KLD/SIM/CC/Cov@10 improvement 为 `0.6627/0.1232/0.1489/0.0312`，paired CI 均为正。HAT 相对其他 scanpath adapters 的 KLD 更低，但 SIM/CC/coverage 并未同步改善，说明瓶颈是轨迹到 population density 的校准，而非单纯网络容量。Corrected ScanDMM 的语义修复与运动审计仍保留：独立 declared-start tensor、gate KLD/SIM/CC/10% coverage `1.8030/0.3949/0.4819/0.8389`，3 s 中位起点距离 `45.8°`，并非停在起点。统一主表见 [matched benchmark](reports/kernel_v2_matched_baselines/summary.md)，HAT 选择/零泄漏审计见 [HAT audit](reports/kernel_v2_hat/summary.md)，ScanDMM 证据见 [declared-start audit](reports/scandmm_declared_start_semantics_audit/summary.md) 与 [old-vs-corrected comparison](reports/scandmm_training_semantics_comparison/summary.md)。
6. **K5 — OFFICIAL SITZMANN RE-ENTRY COMPLETE：** 已从 `sitzmann_raw_data.zip` 的官方 `data/vr/cubemap_*.pck` 重建 standing-VR 数据，排除 browser/seated 条件。22 scenes、88 scene-start cells、780 trials 对应 774 个 cell-level physical observers；6 个同 subject 重复 trial 已合并 observer ID。四个 `viewpoint_idx` 均旋转到 ERP 水平中心，scene-disjoint split 为 `16 train / 3 val / 3 published-test gate`，所有完整性检查通过。受控三 seed小试验表明 Sitzmann-only 的 in-domain KLD 改善较大，但 16 个训练场景在 3–7 epoch 后出现过拟合。K5 当时“不触发整套重训”的保守决定，已被用户随后明确授权的 K6 完整实验取代；K5 数值仍作为训练源初筛证据。
7. **K6 — SITZMANN-ONLY + BIDIRECTIONAL FROZEN TRANSFER PASS：** SphereHorizon、TempSAL、Panoramic-CNN、ScanDMM 与 Pathformer3D 均完成三 seed Sitzmann-only 训练，并在源域 val-only 选模后进行 `Salient360→Sitzmann` 与 `Sitzmann→Salient360` 的冻结测试；BMS360、GBVS360、DeepGaze MSDB 及 reachable controls 也使用相同坐标对齐协议。SphereHorizon 的 `Sal→Sal / Sal→Sitz / Sitz→Sitz / Sitz→Sal` early-macro KLD 为 `1.0616 / 1.6306 / 1.1880 / 1.8157`；TempSAL 为 `1.1287 / 1.6538 / 1.1445 / 1.9349`。但 Salient360-trained ScanDMM/Pathformer3D 在 Sitzmann 上反而优于 Sitzmann-only 版本。固定 8×16 tile 网格的 Sitzmann 同域 `Cov@5/10/20%` 为 SphereHorizon `0.8309/0.9047/0.9737`、TempSAL `0.8459/0.9150/0.9674`，说明 SphereHorizon 的 density 指标优势不会在每个预算上自动转化为 coverage 优势。**最终决定：Salient360 保持 primary content-diverse benchmark；Sitzmann 升为 secondary in-domain matched track 与双向冻结迁移证据，不替代主训练集；应用主张限定为相对 ungated static-final 明显改善、相对 task-matched/reachable 强对照 coverage 具有竞争力，而非所有预算全面领先。** 完整审计为 PASS，sealed target loads=`0`。证据见 [full cross-dataset benchmark](reports/sitzmann_cross_dataset_benchmark/summary.md)。
8. **S0–S2 — SEMANTIC RE-RANKING CLOSED / RETAIN V10：** S0 的同图 GT-final oracle 证明 3 s 存在语义重排上限；S1 bounded residual 将 val Cov@5 提到 `0.7437`，但 KLD=`1.1192`、Cov@10=`0.8339` 未过冻结门；S2 matched semantic anchor 得到更好的 KLD/CC=`1.0722/0.6178`、Cov@10=`0.8441`，correct-vs-wrong-image 归因显著，但 Cov@5=`0.7303` 低于 SphereHorizon/TempSAL。S1/S2 无训练融合仍不能同时改善 5% 与 10%。因此停止扩结构，不做三 seed/gate/外部测试，V10 保持唯一主线。证据见 [semantic re-ranking final decision](reports/semantic_reranking_final/summary.md)。
9. **V11-A–G — TARGETED OPTIMIZATION COMPLETE：** raw-dwell tile supervision、tile/pixel residual、temporal semantic anchor、consensus ranking 与 5 s alpha sweep 均未稳定解决多预算冲突；V11-G 改为对 TempSAL-ERP content map 与 train-only start/time prior 做保守球面 log-density 融合。三种子 val 的 TempSAL→V11-G 为 `KLD 1.1187→1.1070 / SIM .4447→.4481 / CC .5881→.5917 / Cov@5 .7384→.7394 / Cov@10 .8371→.8383 / Cov@20 .9204→.9205`；一次性 gate 为 `1.1287→1.1173 / .4496→.4534 / .5762→.5810 / .7411→.7417 / .8604→.8608 / .9303→.9305`。Density 三项 CI 均为正；Coverage 仅 val Cov@5 CI 明确为正，其余预算尚不足以称为显著领先。V10 保留为 standalone 模型，V11-G 是当前 application adapter 候选。证据见 [V11 optimization summary](reports/V11_OPTIMIZATION_SUMMARY.md)。
10. **V12-A–C — STANDALONE SPHERICAL DWELL-RATE：** V12-A 证明各向同性“语义效用−起点成本”的非负区间累计可显著改善 density 与5秒，但 Cov.@5 不足；V12-B 用16个 train-only 切平面扩散/偏移参数修正前1秒；V12-C 仅在前1秒保留各向异性，之后让移动成本衰减、语义主导。三种子 val 六项均值全部超过 TempSAL，3 s correct-vs-wrong-image 的 KLD/SIM/CC improvement 为 `+.21365/+.04708/+.09298` 且 CI 全正。冻结 gate 除 early Cov.@5 外均值均超过 TempSAL，故 V12-C 升为 standalone mainline candidate，但“所有预算全面领先”主张尚未通过。证据见 [V12-C validation](reports/v12c_spherical_dwell_rate/summary.md) 与 [frozen gate](reports/v12c_spherical_dwell_rate_gate/summary.md)。

Outer-test 已在 v1 下使用，不能重新称为盲测。v2 如需给出 test 数值，只能是模型完全冻结后的 protocol-corrected post-hoc benchmark，不能用于选择核、结构或 checkpoint。

效率补充：DeepGaze MSDB 的四 yaw-roll ERP adapter 为 `1173.3±13.5 ms`、peak `1872.5 MiB`，约 451.64M frozen parameters；它只承担近期强内容模型诊断，不是轻量效率对手。

## 1. 总目标

面向 ICASSP，研究一个有明确应用闭环、模型简洁且无需历史眼动的问题：

> 给定一幅 360° ERP 图像、系统已知的初始水平朝向和预测时刻，能否预测用户开始观看后前几秒的累计群体注意力，并在相同 tile 预算下比 static-final saliency 覆盖更多真实注视？

```text
输入：ERP RGB + ERP geometry + declared initial yaw + horizon t
输出：累计群体显著图 C_t
应用：根据 C_t 排序并预取有限数量的 ERP tiles
```

冻结时刻为 `t ∈ {0.5, 1, 3, 5} s`，四个时刻均属于论文的 early scope，并以等权 macro 汇总；同时逐时刻报告，明确5 s已开始向长期注意力收敛。

## 2. 论文主张锁定

只有以下完整主张得到证据支持，项目才具备投稿价值：

> A reliability-aware, start-aligned and image-conditioned lightweight model forecasts early cumulative attention in 360° images, and improves gaze-mass coverage for tile prefetching under the same transmission budget.

对应三层贡献：

1. **目标层：** 用真实时间累计图代替后期互斥时间片，并用 observer split-half reliability 决定该任务是否可学；
2. **方法层：** 起点对齐的 static base + 有界 time-conditioned residual，不使用历史 gaze 和复杂时序结构；
3. **应用层：** 在固定 tile 预算下验证预测图是否覆盖更多未来注视，而不只报告 KLD/SIM/CC。

不得把贡献写成“首次预测 temporal saliency map”。SaltiNet 和 TempSAL 已覆盖这一泛化表述。

## 3. 为什么从旧路线转向这里

### 已确认的负结果

- **R1 FAIL：** 5–25 s late saliency 中，起点有弱信号，但 actual image 没有稳定优于 wrong image；
- **P2 FAIL：** temporal-correct 与 temporal-shuffled 几乎相同，时间辅助监督不能改善 image-only final map；
- 旧 TempSAL-style 模型的 temporal prediction 过度平滑，correct/shuffled 没有可归因差异。

### 支持新方向的已有线索

- 旧的等顺序分箱审计显示：第一个互斥阶段明显不同于 static final，后续互斥阶段观察者噪声迅速增大；
- 累计阶段在早期更稳定，并逐渐收敛到 final map；
- 这些结果只提供方向性依据，**不能替代 EH0 的真实时间审计**。

因此新任务不再追求：

- temporal supervision 提升 final map；
- 25 s 内每个互斥时间片都预测准确；
- 用 prefix gaze 换取性能；
- 用更复杂网络掩盖监督信号不足。

## 4. 数据角色

| 数据集 | 新主线角色 | 约束 |
|---|---|---|
| Salient360 | 主要 population target、EH0/EH1 与模型训练 | 使用真实毫秒时间戳；固定 canonical start；content-disjoint |
| ET-DK2 | correct/wrong-start 与跨数据集条件验证 | 内容×起点通常仅 1–3 sessions，不把单 session 称为 population GT |
| Sitzmann | 次级同域 matched track、官方四起点证据与双向冻结迁移 | 使用 `sitzmann_official_256x512_spherical2deg_v1`；按 scene 分组；published test 只作已见 gate；不替代 Salient360 主训练集 |
| JUFE | historical diagnostic only | 不进入新训练或主结果 |
| AOI | 排除 | 无稳定下载接口 |

沿用冻结外层 split：

```text
data/splits/future_saliency_contentdisjoint_seed20260711.json
```

以下 EH0--EH6 章节记录旧协议执行历史，其中出现的 `50 train / 9 dev / 13 gate / 13 outer-test`、sealed 和一次性访问均不再约束当前重跑。当前唯一有效划分是 `59 train / 13 val / 13 test`；共享内容仍必须在各数据集中保持 content-disjoint。

## 5. 短期目标与验收条件

### EH0：真实时间目标与可靠性审计 — V1 HISTORICAL

**要回答：** `0.5/1/3/5 s` 累计显著图是否具有足够 observer agreement，并且是否比 static-final 更能描述早期注意力？

具体工作：

1. 审计 Salient360 时间戳单位、单调性、刺激起点及 duration；统一转换为相对刺激开始的秒；
2. 按 fixation 与 `[0,t]` 的实际重叠时长构造 `C_t`，不使用等 fixation 数或等序号分箱；
3. 对 72 个 non-test contents 做 20 次确定性 observer split-half；
4. 比较 same-horizon half-map 与 static-final half-map 对另一半 observers 的 KLD/SIM/CC；
5. 按 canonical content 做 paired bootstrap，不读取 outer-test。

**结果：PASS。** 右眼 primary 的 early-macro improvement：KLD `1.107900`，95% CI `[1.059332, 1.155093]`；SIM `0.433860`，CI `[0.423634, 0.443369]`；CC `0.497277`，CI `[0.471660, 0.523111]`。四个 horizons 中 `0.5/1/3 s` 均通过；左眼 sensitivity 方向一致；outer-test target 加载数为 0。

审计同时修复了旧预处理的 observer 语义：`L/R` 是同一批 physical observers 的双眼记录，不能合并为 2 倍样本。新协议固定右眼 primary、左眼 sensitivity，永不把双眼当独立 observers。报告：[reports/gate_eh0_target_reliability/summary.md](reports/gate_eh0_target_reliability/summary.md)。

**FAIL：** 关闭 early-horizon population saliency，不训练新模型，转向 static 360 saliency 或纯评测论文。

### EH1：tile-prefetch oracle headroom — V1 HISTORICAL

**要回答：** 即使使用人类 half-map，时间相关目标能否在有限 tile 预算下带来实际收益？

冻结设置：

- ERP 划分为 `8 × 16` 个等像素 tiles，经度方向 wrap；
- 根据预测密度在每个 tile 上的球面积分排序；
- primary budget 为总 tiles 的 `10%`，`5%/20%` 为 guards；
- primary metric 为 held-out observer gaze-mass coverage；
- same-horizon human half-map 与 static-final、train-only early template 比较。

**结果：PASS。** 10% budget 的 early-macro coverage improvement：same-horizon vs static-final `0.337978`，95% CI `[0.300338, 0.378006]`；vs train-only early template `0.025663`，CI `[0.018807, 0.032971]`。5%/20% budgets 和左眼 sensitivity 均同方向；outer-test target 加载数为 0。报告：[reports/gate_eh1_tile_oracle/summary.md](reports/gate_eh1_tile_oracle/summary.md)。

EH1 揭示了清晰的时间分工：`0.5/1 s` 的 generic early template 比 content-specific human half-map 分别高 `0.0171/0.0071` coverage；到 `3/5 s`，content-specific map 反超 `0.1011/0.1043`。因此候选模型冻结为“early template base + 随时间增强的 image residual”，不再以 static-final map 作为生成起点。

**FAIL：** 即使 saliency metric 有差异，也不再声称 tile-prefetch 应用价值；重新选择应用或关闭该主线。

### EH2：数据视图、零泄漏与基线 — V1 HISTORICAL

交付：

- 新的 early-horizon processed view 和 manifest；
- timestamp、target normalization、start metadata、content/scene-disjoint 零泄漏测试；
- static-final repeat、train-only early template、equator prior；
- tile coverage evaluator 和 human-oracle report；
- development-gate / outer-test 访问日志，outer-test 继续未读。

**结果：PASS。** 已生成 `72` 个 256×512 non-test targets，`13` 个 outer-test rows 保持 target/fixation sealed；独立 target rebuild 最大绝对误差为 `0`，无非法球面积分或 content leakage。Train-only early template 对 train contents 使用 LOO，baseline test predictions 为 `0`。报告：[reports/gate_eh2_data_audit/summary.md](reports/gate_eh2_data_audit/summary.md)。

### EH3：最小模型 pilot — V1 HISTORICAL

首个模型只允许：

```text
start-aligned ERP + geometry + horizon
    -> train-only early template T_t
    -> shared visual features -> image-dependent log-density correction Δ_t
    -> spherical_softmax(log T_t + β_t Δ_t), β_t∈(0,1] 且随时间增强
    -> C_t
```

不允许 GRU、Transformer、Mamba、diffusion、多阶段 rollout 或 prefix history。

冻结 controls：

1. strong static-final repeat；
2. train-only start/time template；
3. capacity-matched non-temporal multi-head；
4. shuffled-time；
5. wrong-start `+180°`；
6. deterministic wrong-image；
7. no-start-alignment。

**结果：PASS。** Development-gate early-macro：candidate KLD/SIM/CC 为 `0.304491/0.731002/0.868948`；相对 early template 的 KLD improvement `0.027465`，95% CI `[0.006949, 0.047073]`，SIM `0.015672 [0.003689, 0.025753]`。10% coverage 相对 static/template 分别提高 `0.486591/0.022114`，CI 均为正；wrong-time/start/image 与 shuffled-label attribution 全部通过。报告：[reports/gate_eh3_template_residual/summary.md](reports/gate_eh3_template_residual/summary.md)。

冻结解释边界：candidate 与 capacity_non_temporal 的 KLD 差 `0.000628`，CI `[-0.001008, 0.002446]`，没有显著优势。因此不声称连续时间编码优于固定 horizon codes；论文创新锁定在 target reliability、template-to-content transition 和 application closure。

EH3 原验收项均已完成：

- early-macro KLD 与 SIM 显著优于 strong static-final 和 train-only template；
- correct 显著优于 shuffled-time、wrong-start 和 wrong-image，证明时间、起点、图像都被使用；
- 10% tile budget 的 early-macro gaze coverage 显著优于 static-final；
- CC 无超过 `0.01` 的明显退化；
- development-gate 在 SA-EHS 模型决策阶段只读取一次，并明确它不是 fresh holdout；outer-test 未读。

若失败，只允许做一次错误实现审计；不得立即增加模型复杂度。

### EH4：稳健性与外部验证 — V1 HISTORICAL

仅在 EH3 PASS 后执行：

- 3 seeds；
- 5%/10%/20% tile budgets；
- ET-DK2 correct-start vs wrong-start、actual-image vs wrong-image；
- 按 content 聚类 bootstrap，必要时同时按 subject 做 guard；
- 参数量、FLOPs、延迟与显存；
- 若 Sitzmann 时间定义兼容，再做外部 diagnostic。

ET-DK2 不要求每个 content×start 形成稳定 population map；可以直接以 session gaze-mass coverage 为应用指标，但统计必须按 content 聚类，不能把 720 sessions 当作 720 个独立内容。

**结果：PASS。** 三 seed 在 Salient360 development-gate 上相对 early template 的 seed-averaged KLD/SIM/10% coverage improvement 为 `0.030949/0.016634/0.020146`，CI 均为正；四张 map batch-1 延迟 `9.66 ms`。ET-DK2 126 val sessions、7 content groups、8 starts 上，candidate vs template coverage `0.013821 [0.002709, 0.026385]`，correct vs wrong-start `0.789305 [0.773026, 0.803817]`，correct vs wrong-image `0.015302 [0.002673, 0.026926]`。报告：[robustness](reports/gate_eh4_robustness/summary.md)、[ET-DK2](reports/gate_eh4_etdk2/summary.md)。

坐标审计记录：Salient360 官方 ERP center 在已发布 scanpath convention 中对应 `x≈0.75 / yaw≈+90°`，不是 `x=0.5 / 0°`。ET-DK2 必须旋转到 `+90°` reference；该修复由 train-only template peak 和官方起点协议确定。

### EH5：最终测试与 ICASSP 证据包 — V1 HISTORICAL

只有 EH4 完成后才解封 outer-test。论文结果至少包括：

- target reliability 与 task headroom；
- saliency baselines 和全部反事实；
- tile-prefetch application curve；
- 跨数据集与效率；
- failure cases 和“仅支持 equatorial initial yaw”的适用边界。

**结果：PASS。** 冻结 13-content outer-test 上，三 seed 平均相对 train-only early template 的 KLD improvement `0.032211 [0.014578, 0.052557]`、SIM `0.013376 [0.008309, 0.018520]`、10% coverage `0.013268 [0.001425, 0.026123]`；相对 static-final coverage 提升 `0.427983 [0.307969, 0.540779]`。Wrong-time/start/image attribution 均通过。报告：[reports/eh5_outer_test/summary.md](reports/eh5_outer_test/summary.md)。

限制：CC improvement `0.010425` 的 CI `[-0.000878, 0.023908]` 跨零；5 s 相对 template 的 KLD/SIM 点估计为负。因此论文 primary claim 严格限定 `0.5/1/3 s` 的 KLD、SIM 和 tile coverage，不声称 5 s 或 CC 显著改善。

### EH6：公开模型基准对比 — V1 STOPPED / MUST RERUN ON V2

**要回答：** 冻结的 SphereHorizon 不只优于内部 template/反事实，是否也优于与任务最接近的公开 temporal saliency、static 360 saliency 和 scanpath 方法？

固定优先级：

1. `SaltiNet`：temporal-volume primary baseline；
2. `TempSAL-ERP`：最接近论文思想的 primary baseline；
3. 至少一个公开 static 360 saliency 模型：final map 在四个 horizon 重复；
4. `ScanDMM`：scanpath-prefix secondary baseline；
5. `Pathformer3D`：scanpath-prefix secondary baseline。

公平性锁定：

- SphereHorizon 的 checkpoint、结构、horizons、loss、tile grid、budgets 和统计全部冻结；
- baseline 适配只在 train/val 上开发并留下 provenance，不能根据 outer-test 改主模型或选择有利转换；
- temporal-map baseline 进入 primary table；scanpath 方法优先服从官方时间语义；官方没有时间定义时才用 train-only 标定，并在 secondary table 明确标注 `step-to-time adaptation`；
- 所有方法统一报告 `0.5/1/3/5 s` 的 KLD/SIM/CC、5%/10%/20% tile coverage、参数量和推理时间；
- 已使用的 outer-test 只能做冻结 benchmark evaluation，不再被称为模型开发阶段的 unopened test。

**PASS：** 至少完成 `2` 个 task-matched temporal/map baselines、`1` 个 published static 360 baseline 和 `1` 个 scanpath secondary baseline；统一评估代码、输入假设、checkpoint/source、失败复现均可审计。论文不要求每个指标都 SOTA，但 primary early-macro KLD/SIM 和 10% coverage 必须相对至少一个强 task-matched published baseline 有清晰竞争力。

**FAIL：** 若 task-matched published baseline 明显占优，则缩小为 benchmark/application protocol 贡献，不能继续声称方法性能竞争力。

**当前进度：** kernel-v2 matched development benchmark 已闭环。Corrected ScanDMM 以独立 tensor 输入 declared start，Pathformer3D 以 declared-start unit vector 作为 decoder 首 token，HAT 以 declared start 作为非监督 history token；三者均在相同 `50 train / 9 val` split 上完成三 seed训练、val-only 选模和 gate 评估。Gate 上 SphereHorizon 相对 Pathformer3D/ScanDMM/HAT 的 KLD improvement 分别为 `0.8527 [0.7265,0.9837]`、`0.7414 [0.6528,0.8361]` 与 `0.6627 [0.6016,0.7184]`；10% coverage improvement 分别为 `0.0411 [0.0211,0.0618]`、`0.0238 [0.0135,0.0358]` 与 `0.0312 [0.0177,0.0450]`。统一完整性审计为 `188/188 PASS`。SaltiNet 官方权重因无法排除 Salient360 内容重叠，只作为 diagnostic。统一主表见 [matched benchmark](reports/kernel_v2_matched_baselines/summary.md)，完整性审计见 [matched audit](reports/kernel_v2_matched_baseline_audit/summary.md)，定性图见 [matched qualitative](reports/kernel_v2_matched_qualitative/index.md)。主张仍必须保持 template-dominated / modest image correction，并保留 5 s 与 ET-DK2 外部稳定性限制。

## 6. 成功定义

项目成功不等于某个 KLD 数字变好，而是以下证据链全部成立：

```text
真实时间目标可靠
    -> human oracle 对 tile 预取有收益
       -> 简洁模型利用 image + time + start
          -> map 改善转化为相同预算下的 coverage 改善
             -> content-disjoint、跨 seed / 数据集仍成立
                -> 相对公开 task-matched 方法仍有竞争力
```

其中任何一环失败，都必须缩小或关闭主张，不能靠后续环节补救。

## 7. 防跑偏清单

每次开始工作前回答：

1. 当前是否只执行最早一个未完成的 EH 阶段？
2. 当前 target 是否为真实时间累计图，而非等序号互斥 bin？
3. inference 是否不含 first fixation、prefix gaze 或 observer identity？
4. 是否同时保留 saliency metric 和 tile coverage 应用指标？
5. 是否有 static-final、template、shuffled-time、wrong-start、wrong-image 控制？
6. 是否明确区分旧协议历史结果与当前 `59/13/13` 标准重跑，并确保 test 不参与选模？
7. 是否仍是轻量 residual，而没有转向复杂时序网络？
8. 是否遵守 V11 的停止决定：V11-A–F 不再扩结构，V11-G 不依据 val/gate 继续搜索 alpha、loss、tile grid 或预算？

任一答案为“否”，立即停止并返回 [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)。

## 8. 当前唯一下一步

K4 v2 matched development benchmark、统一效率、定量/定性图、claim–evidence 审核与完整 ICASSP 初稿均已完成。稿件见 [paper/main.tex](paper/main.tex) 与 [paper/main.pdf](paper/main.pdf)：前 4 页为技术内容，第 5 页仅参考文献；引用已双源核验，所有字体嵌入，无 unresolved citation/reference 或 overfull box。论文版总览图已按真实实现修正为 `Softmax_sph(log T_t + beta_t Delta_t)`，并新增五列紧凑定性图与效率表。构建、版式和主张审核见 [paper/README.md](paper/README.md)、[paper/DRAFT_AUDIT.md](paper/DRAFT_AUDIT.md)、[paper/LITERATURE_VERIFICATION.md](paper/LITERATURE_VERIFICATION.md)。

ICASSP 2027 已确认 regular paper 为 4 页技术内容、可选第 5 页仅参考文献；截至 2026-07-12 官方 2027 模板和匿名规则尚未发布，当前只用官方 2026 kit 作 provisional layout harness，见 [paper/VENUE_LOCK.md](paper/VENUE_LOCK.md)。V11-G 已在不使用 gate 选参的条件下，使 val/gate 的六项三种子均值总体超过 TempSAL-ERP；但 Coverage 增益很小，5 s 只是 exact no-op，因此仍需诚实保留统计边界。当前允许的下一步只有：以冻结 alpha 做跨数据集测试、量化真实 tile/带宽收益、把 V11-A–F 负结果与 V11-G 机制整理进稿件、等待并迁移到 2027 官方模板、补作者/单位/资助信息和执行严格 reviewer pass。ET-DK2 template-improvement guard、Coverage CI 边界、5 s 无独立增益及缺少 fresh blind test 必须保留，禁止通过改名、重新划分或继续调参掩盖。
