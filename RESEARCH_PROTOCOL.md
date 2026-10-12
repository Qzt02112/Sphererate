# SphereRate Early-Horizon Research Protocol

更新时间：2026-08-06

**Goal:** 构建并严格验证一个无需历史眼动的轻量 360° early-horizon saliency 方法，用预测的前几秒累计注意力提升固定预算下的 tile-prefetch coverage。

**Architecture:** 当前 standalone candidate 为 V12-C SphereRate。共享语义干线从 ERP 提取内容效用；declared start 在球面切平面中提供前1秒各向异性移动成本，之后成本衰减并让语义主导。模型预测四段非负 interval dwell-rate，在未归一化质量空间按真实时长累计，最后才做球面积归一化。训练加入 Cov.@5/10/20 对齐的可微 top-k regret；推理不读取 TempSAL、early-template 数值或历史 gaze。V10/V11-G 分别降为旧 standalone 与 application-adapter 对照。

**Tech stack:** Python、NumPy、PyTorch、现有 `sphererate` 数据与球面指标模块、pytest、content-grouped bootstrap。

> **唯一权威协议。** 自 2026-08-05 起采用常规三分法：原 `train+dev` 合并为训练集，原 `gate` 改作验证集，原 `test` 保持测试集，即 `59 train / 13 val / 13 test`。所有可学习模型只在 train 拟合、只按 val 选择 checkpoint/超参数，并报告该选择在 test 上的结果。test 可重复评估，不再使用“冻结、一次访问、sealed outer-test”限制；但不得根据 test 结果反向选择模型。此前基于 `50/9/13/13` 角色的结果均标为历史证据，不进入新主表。

> **Sitzmann-only 补充协议（2026-08-06）。** 当前不再推进跨数据集冻结比较。Sitzmann 使用 scene-disjoint `16 train / 3 val / 3 test`：train 拟合同域 anchor、切平面尺度和 SphereRate，val 选择 checkpoint，test 只报告。四个 declared starts 先在 canonical scene 内平均，不能作为独立场景扩大样本量。该补充实验不改变 Salient360 主 benchmark，完整结果见 [reports/sitzmann_standard_v2/summary.md](reports/sitzmann_standard_v2/summary.md)。

> **2026-07-11 kernel correction：v1 结果已冻结为 historical-only。** 旧目标使用 `sigma_px=10`（512 宽时等价于水平 `σ=7.03125°`）的固定 ERP 像素核。该核没有可靠的视觉角依据，并在极区改变实际角尺度。Train/val-only 审计表明，旧目标与 `σ=2°` 球面核目标的平均 SIM/KLD 为 `0.604963/1.055303`，旧核有效扩散面积约为新核的 `2.0–2.6×`，且旧核显著压低 content specificity。因此 EH0–EH6 的旧数值不得再作为 v2 论文证据。当前 `K0–K6` 已完成：球面核审计、v2 数据、matched retraining、归因复评、公开 baseline、官方 Sitzmann 重建，以及 Sitzmann-only/双向冻结迁移矩阵。正式核审计见 [reports/saliency_kernel_audit/summary.md](reports/saliency_kernel_audit/summary.md)。

> **2026-08-06 V3 target-reliability 补证。** 已在当前 `256×512 / sphere_chord / σ=2°` 目标上重跑72个 train+val 内容、每内容20次 physical-observer split-half、A→B/B→A 双向审计，标准 test fixation/target 加载数为0。四个 horizons 的 split-half CC 为 `.356/.522/.453/.408`，Spearman--Brown full-cohort estimate 为 `.516/.678/.611/.568`；same-horizon 相对 final-repeat 的 SIM/CC improvement 在四个 horizons 上均为正且95% CI不跨0。10% raw-dwell tile 的 held-out normalized oracle coverage 为 `.845/.905/.827/.800`。KLD 在3/5 s因窄核 half-map 的 sparse-support penalty 反向，故不得用单一 KLD 判定标签可靠性。正式报告见 [reports/standard_split_v3_target_reliability/summary.md](reports/standard_split_v3_target_reliability/summary.md)。

## 0. Kernel-Correction Lock（当前最高优先级）

### K0：来源、几何与 train/val-only 影响审计 — PASS

- Sitzmann 论文明确使用 `σ=1° visual angle`；公开 notebook 以纬度相关的水平 sigma 近似 ERP 畸变；
- Salient360 官方 toolbox 在单位球面上以 fixation/pixel 的 3D chord distance 构造 Gaussian，默认 `σ=2°`；
- v2 采用官方 Salient360 风格的 `unit-sphere chord Gaussian, σ=2°, truncate=4σ`，`1°/4°` 只作敏感性，不允许按模型结果搜索；
- v2 保留当前应用定义所需的 clipped dwell weighting，因此它是 cumulative dwell density，而不是 Sitzmann 的 equal-fixation-count static map；
- 经度 seam、极点、纬度角尺度与球面积分测试必须全部通过。

来源：[Sitzmann et al. paper](https://arxiv.org/abs/1612.04335v2)、[Sitzmann code](https://github.com/vsitzmann/vr-saliency)、[Salient360 toolbox](https://github.com/David-Ef/salient360Toolbox)。

### K1–K4：修复后的唯一合法执行链

1. **K1 v2 data：** 新建 `early_horizon_salient360_spherical2deg_v2`，不得覆盖 v1；重建 cumulative/final maps、train-only LOO template、integrity/leakage report；outer-test 在此阶段保持 sealed metadata。
2. **K2 matched retraining：** 保持网络、horizons、loss、优化器和 controls 不变，重训 candidate、shuffled-label、capacity control 和所有 learned baselines；不以旧 test 结果调参。
3. **K3 re-evaluation：** 先在 val/已见 gate 上重跑 map、wrong-time/start/image、content-utilization 与 tile coverage。v2 若不能稳定优于 template，不增加复杂结构，直接缩小或关闭方法主张。
4. **K4 baseline rerun：** 所有从 fixation/scanpath 生成 map 的 baseline 必须统一用 v2 renderer；直接输出 saliency density 的模型只做球面积归一化，不额外卷积。能接收 initial gaze 的公开模型必须在**训练与推理两阶段**通过原生/decoder 或可审计的 state-dict-compatible adapter 显式输入每个样本的 declared start，不能用第一秒 GT gaze 代替起点，也不能只把数据筛成 fixed-start 后声称已利用起点。首个未来 gaze 必须继续作为 `t=1` target，禁止通过 prepend declared start 改变监督长度或把起点作为额外 target。另冻结 `reachable-final` 强对照：用 train-only 右眼 clipped-dwell 的 95% 球面角距离包络硬裁剪 static final map，同时保留未裁剪原始行。禁止把 v1 与 v2 数值放入同一公平比较表。

2026-08-05 起本段旧 outer-test 约束被标准 `train/val/test` 协议取代。测试集不参与模型选择，但允许为排错、复现和统一 benchmark 重复运行；所有报告必须保留 validation-selected checkpoint 的来源。

---

## 1. Active Task Lock

### SA-EHS：Start-Aligned Early-Horizon Saliency

```text
(ERP RGB, ERP geometry, declared initial yaw, horizon t)
    -> cumulative population saliency C_t
    -> rank ERP tiles
    -> early gaze-mass coverage under a fixed tile budget
```

冻结预测时刻：

```text
t = {0.5, 1, 3, 5} seconds after stimulus onset
primary = equal-weight macro over {0.5, 1, 3, 5} s
```

这里的 temporal 信息是一个部署时已知的预测时刻查询，不是历史 gaze。初始朝向必须来自显示系统或数据集声明值，不能用 first fixation 代替。

### 唯一允许的论文主张

> A reliability-aware, start-aligned and image-conditioned lightweight model forecasts early cumulative attention in 360° images, and the improved maps cover more future gaze under the same tile-prefetch budget.

该主张必须同时通过 target reliability、human-oracle application headroom、matched model controls 和 application coverage 四层证据。只改善 saliency metric 或只改善 coverage 均不够。

## 2. Historical Evidence and Closed Routes

### R1：initial-start-conditioned 5–25 s late saliency — FAIL / CLOSED

- train-selected fusion `alpha=0.90`；
- fusion vs start 的 KLD improvement 为 `0.009883`，95% CI `[-0.002581, 0.025828]`；
- actual image 没有稳定优于 wrong image；
- 起点仅表现为弱信号。

正式报告：[reports/gate_r1_start_conditioned_late/summary.md](reports/gate_r1_start_conditioned_late/summary.md)。

### P2：image-only temporal auxiliary for final saliency — FAIL / CLOSED

| Method | KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ |
|---|---:|---:|---:|
| final_only | 0.201785 | 0.755163 | 0.734828 |
| temporal_correct | 0.204531 | 0.753521 | 0.734099 |
| temporal_shuffled | 0.204520 | 0.753916 | 0.730955 |
| capacity_final_repeat | 0.205777 | 0.753209 | 0.725570 |

Correct temporal 与 shuffled temporal 几乎相同，不能声称 temporal supervision 为 final map 提供了独立信息。

正式报告：[reports/gate_p2_temporal_aux/summary.md](reports/gate_p2_temporal_aux/summary.md)。

### 旧 temporal target 的边界

旧等序号分箱审计表明：早期累计注意力可能可靠，晚期互斥时间片不可靠。但旧结果不是按真实秒数构造，只能用于提出 SA-EHS，不能作为新任务的正式结果。EH0 必须重新验证。

### 永久关闭或降为 diagnostic 的方向

1. temporal auxiliary 提升 final map：关闭；
2. 5–25 s late map：关闭；
3. prefix-conditioned future saliency：oracle/diagnostic only；
4. 逐个晚期互斥 map：关闭；
5. GRU、Transformer、Mamba、diffusion、复杂 rollout：SA-EHS pilot 禁止；
6. “首次 temporal saliency”表述：禁止，已有 SaltiNet/TempSAL。

## 3. Legal Inputs and Deployment Scope

### 合法输入

- ERP RGB；
- 固定 ERP geometry channels；
- 系统声明的初始 ERP longitude / yaw；
- 预测时刻 `t ∈ {0.5, 1, 3, 5} s`；
- train-only 学习参数与公开预训练视觉权重。

### 禁止输入

- first fixation、prefix fixation、gaze/head trajectory；
- observer identity、session outcome 或未来 viewport；
- GT temporal/final saliency map；
- dev/gate/test target 统计生成的 prior；
- 使用 GT first fixation 推断“初始朝向”。

### 当前适用边界

- 初始条件先限定为 **equatorial yaw**，不声称任意 pitch/roll；
- 预测 population-level early attention，不预测个体 scanpath；
- 输入是静态 360° 图像，不引入音频或视频运动；
- 主要应用是 session 开始、尚无历史行为时的冷启动 tile prefetch。

## 4. Frozen Target Definition

### 4.1 时间零点与单位

- `t=0` 定义为全景刺激开始显示；
- Salient360 原始 timestamp/duration 先审计单位，预期毫秒，统一转换为秒；
- 每个 observer 的 timestamp 必须非降序，且有效观看区间必须覆盖冻结 horizons；
- 不允许按 fixation 数、序号或每人轨迹比例替代真实时间。

### 4.2 累计显著图

对 horizon `t`，observer `o` 的第 `i` 个 fixation 对 `[0,t]` 的有效 dwell 权重为：

```text
w_i(t) = max(0, min(end_i, t) - max(start_i, 0))
```

使用单位球面 chord-distance Gaussian 生成 density：`σ=2°`，截断半径 `4σ`，再以 cosine-latitude 球面积权重归一化。Population target `C_t` 聚合指定 observer 集合在 `[0,t]` 内的有效 dwell。跨越 horizon 边界的 fixation 只能计入重叠部分，不能整段纳入。固定 ERP 像素 sigma、纬度无关的水平卷积和 latitude wrap 均不得用于 v2 正式 target。

ET-DK2 为高频 gaze samples，按 `[0,t]` 内有效采样时长积分，不把每个 sample 误称为 fixation。

### 4.3 目标约束

- `C_t` 是累计、重新归一化的概率密度，不要求归一化后逐像素单调；
- `0.5/1/3/5 s` 均属于当前论文的 early targets，并以等权 macro 汇总；
- 仍需逐时刻报告，不能用 macro 掩盖5 s向 final distribution 收敛的现象；
- final map 只作为 static baseline，不是优化目标；
- kernel 以视觉角定义，冻结为 `sphere_chord / sigma_deg=2.0 / truncate=4σ`；正式分辨率 `256×512`，area normalization 开启；K2 以后不得搜索。

## 5. Data Roles and Split Lock

### 5.1 Salient360：主要 population benchmark

- 85 个 canonical contents；
- observer-level scanpaths 带真实 timestamp/duration；
- canonical initial yaw 必须由公开数据协议或元数据验证，禁止用首注视估计；
- 用于 EH0/EH1、模型 train/dev/gate/test。

### 5.2 ET-DK2：起点条件与跨数据集验证

- 720 sessions、40 个与 Salient360 共享的 contents、18 subjects、8 个 equatorial starts；
- content×start 通常只有 1–3 sessions，**不得把单 session 或稀疏 content×start map 称为稳定 population ground truth**；
- EH3 之前不用于候选模型拟合；
- EH4 以 session gaze-mass coverage、correct/wrong-start 和 actual/wrong-image 为主要用途；
- 统计按 canonical content 聚类，subject 作为额外 guard。

### 5.3 Sitzmann、JUFE、AOI

- Sitzmann：已从 `data/sitzmann_raw_data.zip` 中的官方 standing-VR `data/vr/cubemap_*.pck` 重建；browser 与 seated-VR 不进入当前任务；
- 每个 trial 保留 `experiment_id`、`viewpoint_idx∈{0,1,2,3}`、真实 fixation 起止时间和 duration；同一 subject 在同一 scene-start cell 的重复 trial 共用 observer ID，不能重复计为 independent observers；
- 四个官方起点按 `(0°,90°,180°,270°)` 映射，并将图像、fixation 和 target 同步旋转到 ERP 水平中心 `x=0.5`；禁止用 first fixation 反推起点；
- 重建视图含 `22 scenes / 88 scene-start cells / 780 trials / 774 cell-level observers`。Map GT 的 train supervision 为 `16 scenes / 64 cells / 544 cell-level physical observers`，中位每 target `8` 人；scanpath 模型读取官方 dense `gaze_lat_lon/timestamp` 后得到 `547` 个 train trials，精确采样 `0..29 s` 的 1 Hz 序列，780/780 trials 均保留且无 bin repair。场景多样性仍显著小于 Salient360，因此不替代其 primary training 角色；
- 旧 ScanDMM 转换和此前 pseudo-start frozen-transfer 结果永久无效；
- JUFE：historical diagnostic only；
- AOI：无稳定下载接口，排除。

### 5.3.1 Primary training-source lock

受控三 seed 对照固定相同 SphereHorizon 架构、loss、optimizer 和 scratch budget。Sitzmann-only 在其 gate 上相对 train-only template 的 early-macro KLD 从 `1.3273` 降至 `1.1880±0.0057`，但 CC 仅从 `0.7459` 升至 `0.7493±0.0041`，并在 3–7 epoch 后出现明显验证退化。固定 target-domain template 后，Salient360 residual 在 Sitzmann 上达到 `1.2263±0.0064 / 0.7518±0.0003` KLD/CC；Sitzmann residual 迁移到 Salient360 仅达到 `1.0987±0.0046 / 0.5997±0.0013`，改善很小。Salient360→Sitzmann fine-tune 为 `1.1965±0.0027 / 0.7569±0.0013`，表现为 KLD 与 CC 的折中，不改变 primary-source 决定。

K5 的上述小规模对照只用于决定是否值得投入完整计算。用户随后明确授权 K6：对 SphereHorizon、TempSAL、Panoramic-CNN、ScanDMM、Pathformer3D 做三 seed Sitzmann-only 训练，并做双向冻结跨训练集测试。所有 checkpoint 均只由 source validation 选择；跨域只做已声明起点的经度坐标对齐，不做 target adaptation。结果显示 map models 可从 Sitzmann 同域监督获益，但 Salient360-trained ScanDMM/Pathformer3D 在 Sitzmann target 上仍优于各自 Sitzmann-only 版本；而在 Salient360 target 上，五个 learned methods 均以 Salient360 training 更好。固定 `8×16` 网格的 `5%/10%/20%` gaze-mass coverage 也已统一补算：SphereHorizon 与 TempSAL 在两数据集上整体接近，且不同预算互有胜负，因此禁止把 density-metric 优势写成“所有 tile budget 均领先”。

最终冻结决定：

1. Salient360 保持论文主训练集和 matched benchmark 数据源；
2. Sitzmann 用于次级同域 matched track、官方四起点证据和双向 frozen-transfer 分析；
3. K6 的 Sitzmann-only baseline suite 已完成；不得基于 target gate 继续调结构、训练预算、adapter 或 reachability radius；
4. Sitzmann published scanpath test scenes不是 fresh holdout，不作为确认性 test；
5. 证据入口：[training suitability](reports/sitzmann_training_suitability/summary.md)、[train-source comparison](reports/sitzmann_train_source_comparison/summary.md) 与 [full cross-dataset benchmark](reports/sitzmann_cross_dataset_benchmark/summary.md)。

### 5.4 Standard content-disjoint split（2026-08-05 起生效）

唯一主划分文件：

```text
data/splits/salient360_standard_train59_val13_test13_seed20260711.json
```

Sitzmann 独立 scene split：

```text
data/splits/sitzmann_official_contentdisjoint_seed20260713.json
train = 16 scenes / 64 start-conditioned rows
dev   =  3 scenes / 12 start-conditioned rows
gate  =  3 published scanpath-test scenes / 12 rows; historically defined, not fresh
```

Salient360 角色：

```text
train 59 -> 原 train+dev；parameter fitting
val   13 -> 原 gate；checkpoint/超参数选择
test  13 -> 原 test；最终性能报告，可重复评估
```

要求：

- Salient360 与 ET-DK2 共享的 `canonical_content_id` 必须在同一 split；
- Sitzmann rotations 必须按 scene 同组；
- observer split 只用于同一 content 内的 reliability/oracle，不替代 content split；
- val 可以用于早停、checkpoint 选择和预先声明的模型比较；
- test 不参与上述选择；取消一次性访问与 sealed target 机制；
- 主表中的 learned baselines 必须使用相同 59 个 train contents 拟合，并按相同 13 个 val contents 选模；
- 同时报告三种子均值/标准差和 content-grouped paired bootstrap；单次 test 点估计不得用于追加调参。

## 6. EH0 — Real-Time Target Reliability Gate

> 本节及后续 EH0--EH6、K/V 历史结果记录当时的执行过程。文中的 gate、outer-test、sealed、frozen 与一次性访问措辞均由 5.4 的标准三分法取代；这些旧数字不得进入当前主表。

**状态：PASS。未进行新训练。**

### 实现范围

- Create: `src/sphererate/data/early_horizon.py`
- Create: `src/sphererate/evaluation/early_reliability.py`
- Create: `scripts/audit_early_horizon_targets.py`
- Create: `tests/test_early_horizon_targets.py`
- Create: `tests/test_early_reliability.py`
- Create: `reports/gate_eh0_target_reliability/summary.md`

优先复用：

- `src/sphererate/data/salient360.py` 的 observer/timestamp 读取；
- `src/sphererate/data/maps.py` 的 seam-aware map construction；
- `src/sphererate/evaluation/metrics.py` 的修复后 spherical KLD/SIM/CC；
- `src/sphererate/evaluation/temporal_factorization.py` 的 deterministic observer split 思路。

### 必须审计

1. timestamp/duration 单位和范围；
2. 每 observer 时间单调性、首末时间和 horizon coverage；
3. 左右眼 observer ID 是否独立且无重复；
4. `[0,t]` 截断 dwell 的边界测试；
5. ERP seam、极区和球面积归一化；
6. Salient360 canonical initial yaw 的公开 provenance；
7. non-test 与 outer-test 访问隔离。

### Reliability comparison

对 72 个 non-test contents 各做 20 次确定性 observer half split，split seed 从 `20260711` 顺序派生。先在每个 content 内平均 20 次及 A/B 双向结果，再按 content 统计。用 A-half 生成：

- same-horizon `C_t^A`；
- static-final `C_final^A`。

两者都预测 B-half 的 `C_t^B`，交换 A/B 后取平均。Improvement 统一为正值更好：

```text
KLD improvement = KLD(static-final, target) - KLD(same-horizon, target)
SIM improvement = SIM(same-horizon, target) - SIM(static-final, target)
CC  improvement = CC(same-horizon, target)  - CC(static-final, target)
```

### EH0 PASS

按 `canonical_content_id` paired bootstrap 10,000 次，seed `20260711`：

1. `{0.5,1,3 s}` early-macro 的 KLD 和 SIM improvement 95% CI lower bound 均 `> 0`；
2. 三个 early horizons 中至少两个在 KLD、SIM 上各自 lower bound `> 0`；
3. 5 s 的 CC improvement CI lower bound 不小于 `-0.01`；
4. canonical initial yaw provenance 已验证；
5. outer-test 未读取。

冻结结果：右眼 primary 的 early-macro KLD improvement `1.107900`，95% CI `[1.059332, 1.155093]`；SIM `0.433860`，CI `[0.423634, 0.443369]`；CC `0.497277`，CI `[0.471660, 0.523111]`。`0.5/1/3 s` 均为正，5 s CC guard 通过，左眼 sensitivity 同方向，outer-test target 加载数为 0。EH0 判定 **PASS**。

Observer 审计还发现旧 processed view 把 `L/R` 双眼分别编号，导致同一 physical observer 被重复计算。SA-EHS 从 EH0 起固定：右眼为 primary、左眼仅作 sensitivity、双眼永不 pooled as independent observers。

正式报告：[reports/gate_eh0_target_reliability/summary.md](reports/gate_eh0_target_reliability/summary.md)。

## 7. EH1 — Tile-Prefetch Oracle Gate

**状态：PASS。未进行新训练。**

### 冻结应用设置

- ERP tile grid：`8 latitude × 16 longitude`；
- 等像素 tile，longitude seam wrap；
- tile score：预测 density 在 tile 内的 spherical probability mass；
- budgets：总 tile 数的 `5% / 10% / 20%`，四舍五入规则在测试中冻结；
- primary budget：`10%`；
- primary application metric：held-out observer cumulative gaze-mass coverage；
- guard：5%/20% coverage curve、极区/缝合线样例。

### 实现范围

- Create: `src/sphererate/evaluation/tile_prefetch.py`
- Create: `scripts/evaluate_eh1_tile_oracle.py`
- Create: `tests/test_tile_prefetch.py`
- Create: `reports/gate_eh1_tile_oracle/summary.md`

### Oracle controls

用 A-half 生成 tile ranking，预测 B-half：

1. same-horizon human map；
2. static-final human map；
3. train-only early template：在 declared-start frame 按 horizon 聚合；Salient360 中即 canonical-start template；
4. equator/uniform prior。

### EH1 PASS

1. primary 10% budget 下，same-horizon 相对 static-final 的 early-macro coverage improvement CI lower bound `> 0`；
2. `0.5/1/3 s` 至少两个 horizon 单独 lower bound `> 0`；
3. same-horizon 相对 train-only early template 的 early-macro coverage improvement CI lower bound 也必须 `> 0`；
4. outer-test 未读取。

冻结结果：10% budget 的 early-macro same-horizon vs static-final coverage improvement `0.337978`，95% CI `[0.300338, 0.378006]`；vs train-only early template `0.025663`，CI `[0.018807, 0.032971]`。5%/20% 与左眼 sensitivity 同方向，outer-test target 加载数为 0。EH1 判定 **PASS**。

分 horizon 结果显示：在 `0.5/1 s`，generic template 比 content-specific half-map 分别高 `0.017074/0.007067`；在 `3/5 s`，content-specific half-map 反超 `0.101130/0.104270`。因此 EH3 模型必须从 `T_t` 出发，用随时间增强的 image residual 建模 template-to-content transition。

正式报告：[reports/gate_eh1_tile_oracle/summary.md](reports/gate_eh1_tile_oracle/summary.md)。

## 8. EH2 — Data View, Leakage Audit and Non-Learned Baselines

**状态：PASS。未进行候选神经网络训练。**

### 实现范围

- Create: `src/sphererate/data/early_view.py`
- Create: `src/sphererate/data/early_audit.py`
- Create: `scripts/build_early_horizon_data_view.py`
- Create: `scripts/audit_early_horizon_data.py`
- Create: `configs/early_horizon_256x512.yaml`
- Create: `tests/test_early_horizon_data_view.py`
- Create: `tests/test_early_horizon_data_audit.py`
- Output: `data/processed/early_horizon_salient360_etdk2_v1`
- Output: `reports/gate_eh2_data_audit/summary.md`

### 必须冻结的 baseline

- `static_final_repeat`：强 static model 的 final prediction 重复到各 horizon；
- `early_template`：train-only、在 declared-start frame 按 horizon 聚合；
- `equator_prior` 与 `uniform_prior`；
- `capacity_non_temporal` 留到 EH3 matched training。

### EH2 PASS

- manifest 每行包含 canonical content、split、真实 horizon、declared start 与 target provenance；
- content/scene leakage 为 0；
- train-only priors 不读取 dev/gate/test target；
- 独立 target 实现逐像素/积分一致；
- 所有 map spherical integral 为 1；
- outer-test 访问计数为 0。

冻结结果：72 个 non-test targets 已生成，13 个 outer-test rows 只保留 sealed metadata；独立 target rebuild 最大绝对误差 `0`；非法球面积分、horizon 文件和 content leakage 均为 `0`；train-only baseline test predictions 为 `0`。EH2 判定 **PASS**。

正式报告：[reports/gate_eh2_data_audit/summary.md](reports/gate_eh2_data_audit/summary.md)。

## 9. EH3 — Minimal Model Pilot

**状态：PASS。Single-seed matched pilot 已完成。**

### 冻结候选模型

```text
ERP RGB + geometry + horizon t
    -> yaw-align using declared start
    -> train-only early density template T_t
    -> shared visual trunk -> shallow log-density correction Δ_t
    -> spherical_softmax(log T_t + β_t * Δ_t), β_t in (0,1] and monotonic
    -> restore to original ERP yaw
```

约束：

- `β_t` 有界且随 horizon 单调；`Δ_t` 本身不宣称有界；
- temporal part 只能是 time embedding + 浅层 modulation/residual；
- 不使用 recurrent state、self-attention temporal stack 或 autoregressive rollout；
- 模型输出四个累计 maps，但 primary checkpoint 只按 dev early-macro KLD 选择；
- final map 不参与主损失；
- pilot 只跑一个冻结 seed，EH3 PASS 前禁止 3-seed。

### Matched runs

| Run | 变化 | 目的 |
|---|---|---|
| `static_final_repeat` | 强 static final 重复 | 主要 static reference |
| `early_template` | train-only declared-start-frame prior | time/start prior control |
| `capacity_non_temporal` | 同参数量四个固定 heads，无 time query | capacity control |
| `candidate_correct` | correct image/time/start | candidate |
| `candidate_shuffled_label` | 每图固定打乱 horizon label | matched temporal-supervision control |
| `candidate_no_alignment` | 同预算但不做 start yaw alignment | geometry mechanism |

所有 learned runs 必须同 seed、初始化策略、batch order、optimizer、augmentation、epoch budget 与 checkpoint rule。

同一个 `candidate_correct` checkpoint 还必须做三个**不重新训练**的 inference counterfactual：

- `wrong_time`：使用冻结的 horizon permutation；
- `wrong_start`：declared yaw 固定加 `180°`；
- `wrong_image`：split 内确定性错配 ERP，target/time/start 保持不变。

Wrong-image 不得跨 split，且错配表在训练前冻结。

### EH3 Decision Gate

只允许一次读取 13 个 gate contents。Candidate 必须同时满足：

1. early-macro KLD、SIM 相对 `static_final_repeat` 的 CI lower bound `> 0`；
2. early-macro KLD、SIM 相对 `early_template` 的 CI lower bound `> 0`；
3. correct 相对 shuffled-label learned control，以及 wrong-time、wrong-start、wrong-image inference counterfactuals 的 primary KLD 或 SIM 至少一项 lower bound `> 0`，且另一项点估计同方向；
4. 10% tile budget coverage 相对 static-final 的 early-macro CI lower bound `> 0`；
5. candidate CC 相对主要 controls 的 CI lower bound均不小于 `-0.01`；
6. 参数量匹配、outer-test 未读；development-gate 明确标注为 historically seen，而非 confirmatory evidence。

如果 correct 不能优于 wrong-image，论文不能声称 image-conditioned；如果不能优于 wrong-start，不能声称 start-conditioned；如果不能优于 wrong-time/shuffled-label，不能声称 time-conditioned。不得以 application gain 掩盖 attribution failure。

冻结结果：candidate early-macro KLD/SIM/CC 为 `0.304491/0.731002/0.868948`；相对 early template 的 KLD improvement `0.027465`，95% CI `[0.006949, 0.047073]`，SIM `0.015672 [0.003689, 0.025753]`；10% coverage 相对 template 提高 `0.022114 [0.010291, 0.034840]`。Wrong-time/start/image 和 shuffled-label attribution 全部通过，参数完全匹配，outer-test 未读。EH3 判定 **PASS**。

Candidate 与 capacity_non_temporal 没有显著差异：KLD improvement `0.000628`，CI `[-0.001008, 0.002446]`。因此禁止把贡献表述为“连续时间编码优于固定 horizon heads”。Salient360 wrong-start 的巨大退化建立在统一 canonical start 上，只是机制检查；真实 start generalization 必须由 EH4 ET-DK2 验证。

正式报告：[reports/gate_eh3_template_residual/summary.md](reports/gate_eh3_template_residual/summary.md)。

## 10. Metrics and Statistics

### Saliency metrics

- Primary：`KLD_sph`、`SIM_sph`；
- Guard：`CC_sph`；
- 全部基于修复后的球面积权重；
- 先逐 content、逐 horizon 计算，再做 early macro，禁止把像素或 sessions 当独立样本；禁止对不等大小的 batch mean 再等权平均。9-content val 必须是 9 个 content 等权，不能让最后一个单样本 batch 获得 4 倍权重。

### Application metrics

- Primary：10% budget cumulative gaze-mass coverage；
- Guards：5%/20% coverage、per-horizon curve；
- 每个 tile 按传输单元计一次预算，ranking score 使用 spherical mass；
- tie-breaking 必须确定性并写入测试。

### Bootstrap

```yaml
group: canonical_content_id
iterations: 10000
seed: 20260711
ci: percentile_95
improvement_orientation: positive_is_better
```

ET-DK2 的 session-level 分析仍以 content 为 primary cluster；subject-cluster 或双向 cluster 作为 guard。不得把同一图像的多个 start/session 当独立内容扩大显著性。

## 11. EH4 — Robustness and External Validation

仅在 EH3 PASS 后执行：

1. seeds `20260710/20260711/20260712`；
2. ET-DK2 correct vs wrong-start、actual vs wrong-image；
3. tile budgets 5%/10%/20%；
4. resolution、tile seam 与极区 guards；
5. 参数量、FLOPs、单图四 horizon 延迟、显存；
6. 如果时间定义兼容，增加 Sitzmann external diagnostic；
7. 失败案例按场景类型和起点附近/远处目标分类。

EH4 不允许引入新主干结构。只允许复现实验、效率测量和预注册的外部验证。

冻结结果：三 seed 的 Salient360 map 与 10% coverage improvement 均稳定；seed-average 相对 template 的 KLD/SIM/coverage improvement 为 `0.030949/0.016634/0.020146`，CI 均为正。ET-DK2 126 sessions、7 contents、8 starts 上，candidate vs template `0.013821 [0.002709, 0.026385]`，correct vs wrong-start `0.789305 [0.773026, 0.803817]`，correct vs wrong-image `0.015302 [0.002673, 0.026926]`。EH4 判定 **PASS**。

ET-DK2 首次运行暴露并修复了固定 frame offset：Salient360 official ERP center 对应 released scanpath `x≈0.75 / yaw≈+90°`。修复前把 start 对齐到 `0°` 会使正确模板错位 90°；该实现错误不作为研究结果。修复由坐标约定唯一确定，重新运行未改变模型或指标。

正式报告：[robustness](reports/gate_eh4_robustness/summary.md)、[ET-DK2](reports/gate_eh4_etdk2/summary.md)。

## 12. EH5 — Final Test and Paper Evidence

只有 EH4 完成且主张未改变，才允许一次性解封 outer-test。Test 后不得回到 dev 调参再重测。

论文最小证据包：

1. 真实时间 target reliability；
2. human-oracle application headroom；
3. strong static、template 与 matched controls；
4. shuffled-time、wrong-start、wrong-image attribution；
5. tile-prefetch budget curve；
6. 3-seed、content bootstrap、ET-DK2 external condition test；
7. 效率与适用边界。

允许的标题方向：

> **SphereHorizon: Reliability-Aware Early Attention Forecasting for 360° Images**

不允许在标题或摘要中声称：首次 temporal saliency、任意 3D initial orientation、个性化预测、完整 scanpath prediction 或改善 final saliency。

冻结 outer-test 结果：三 seed 平均相对 train-only early template 的 KLD improvement `0.032211 [0.014578, 0.052557]`、SIM `0.013376 [0.008309, 0.018520]`、10% coverage `0.013268 [0.001425, 0.026123]`；相对 static-final coverage `0.427983 [0.307969, 0.540779]`。Wrong-time/start/image attribution 均通过。EH5 判定 **PASS**，outer-test 已使用且永久禁止 post-test tuning/retest。

CC improvement CI 轻微跨零，5 s 相对 early template 不改善；两者必须作为限制报告。正式报告：[reports/eh5_outer_test/summary.md](reports/eh5_outer_test/summary.md)。

## 13. EH6 — Frozen External Baseline Benchmark

**状态：MATCHED DEVELOPMENT BENCHMARK PASS。** v2 的 val/gate 已完成同 split、显式 declared-start、零 v2-test target 访问的公开方法比较。该状态只表示 baseline matrix 闭环，不把历史 outer-test 恢复为新的盲测，也不等于论文与效率证据已经全部定稿。

### 13.1 Baseline tiers

Primary task-matched table 至少包括：

1. `SaltiNet` temporal saliency volume；
2. faithful `TempSAL-ERP` temporal/map adaptation；
3. 至少一个 published static 360 saliency model，将同一 final map 重复到四个 horizons；
4. 至少一个 published static final 的 `reachable-final` 版本，用起点与时间可达域构成强简单对照；
5. SphereHorizon、train-only early template、内部 strong static-final repeat。

Secondary scanpath-to-application table 至少包括：

1. `ScanDMM`；
2. `Pathformer3D`，若官方实现无法在冻结环境复现，必须给出可审计 blocker，不能用内部替身冒充。

### 13.2 Frozen fairness rules

- SphereHorizon 三个 checkpoints 及其 SHA256 永久冻结；不得重训、重选或改输出后处理；
- baseline adapter、checkpoint 选择和所有 step/time 映射只使用 source、train 和 val；outer-test 结果不得参与任何选择；
- 公开 pretrained checkpoint 若训练数据与 Salient360 outer-test content 重叠，必须披露并单列，不得作为无泄漏主比较；
- 直接输出 temporal map/volume 的方法进入 primary table；scanpath 方法只进入 secondary table；
- ScanDMM、Pathformer3D 等具备初始注视接口的方法必须显式接收 declared start；Salient360 当前声明值均为 `(pitch=0°, yaw=+90°)` 只是数据条件一致，不等于允许省略模型条件输入。禁止以 GT first fixation 替代 start；
- scanpath 时间语义优先服从官方源码：例如 ScanDMM 预处理每秒采一个 gaze point，使用 `1 step = 1 s`，其中 `0.5 s` 只能标为 first-second approximation；只有官方没有时间定义的 fixation-step 方法才允许用 Salient360 train-only physical-observer schedule 标定，并明确标记 `step-to-time adaptation`；同时报告 step-matched sensitivity；
- `reachable-final` 的主设置固定为 train 50 contents、右眼、clipped-dwell 加权 95% great-circle distance；冻结半径为 `39.3918°/44.4163°/97.8417°/125.0349°`（0.5/1/3/5 s）。quantile、半径和 hard/soft gate 不得根据 val/gate 结果选择；原始 final-repeat 与裁剪版必须同时报告；
- 统一使用真实时间累计 target、ERP spherical KLD/SIM/CC、`8×16` tiles、5%/10%/20% gaze-mass coverage；primary 为 `0.5/1/3 s` macro 和 10% budget；
- stochastic 方法至少三 seeds 或足够的公开采样数；统计仍按 canonical content paired bootstrap；
- 同表报告输入假设（image/start/history）、训练数据、参数量、单图延迟和复现状态。

### 13.3 Execution order and gate

执行顺序固定为：

```text
EH6-B0 protocol + source/environment audit
-> EH6-B1 SaltiNet
-> EH6-B2 ScanDMM
-> EH6-B3 Pathformer3D
-> EH6-B4 TempSAL-ERP
-> EH6-B5 published static 360 baseline
-> EH6-B6 frozen unified table
```

可因明确的环境 blocker 调换 B1–B5 的运行先后，但不得跳过 baseline 类别。EH6 PASS 要求：两个 task-matched temporal/map baselines、一个 published static 360 baseline、一个 scanpath secondary baseline均有统一结果；无法复现项有 source、依赖、权重和错误日志。性能不设“必须全面 SOTA”的事后门槛，但论文主张必须随结果缩放。

Outer-test 已在 EH5 使用一次。EH6 的 test evaluation 只能称为 **frozen post-hoc benchmark evaluation**；其合法性来自主方法完全冻结和 baseline protocol 预注册，而不是重新把它包装成 unopened test。

### 13.4 Frozen progress record

- ScanDMM：必须使用模型已有的 initial 3D gaze-point interface，输入 declared `(pitch=0°, yaw=+90°)`；官方随机起点仅为历史 diagnostic；
- ScanDMM frozen test：KLD/SIM/CC `1.2524/0.5536/0.7313`，10% coverage `0.8177`；SphereHorizon 10% coverage `0.8315`，paired improvement `0.0138 [0.0024, 0.0255]`；
- Panoramic-CNN：published static 360 architecture 已在冻结 50/9 split 上从头训练三 seeds，test 结果 `2.0433/0.2202/0.1849`，10% coverage `0.3486`；
- BMS360/GBVS360：官方确定性源码已完成 val/gate；其 final map 经 `reachable-final` 后 gate 分别达到 KLD/SIM/CC/10% coverage `1.7297/0.3431/0.3992/0.8115` 与 `1.6939/0.3483/0.4108/0.8106`，但仍明显弱于 SphereHorizon 的 `1.0616/0.4763/0.6163/0.8626`，说明起点可达域本身不能解决任务；
- DeepGaze MSDB（ICCV 2025）：官方 CLIP+DINOv2 权重冻结，四 yaw-roll ERP adapter 后只在 50-train 拟合两个标量；gate calibrated raw 为 `2.5021/0.1483/0.2283/0.4640`，加同一 reachable 后为 `1.5191/0.3558/0.4570/0.8263`。该行证明近期强 2D fixation content model 加运动约束仍不足，但必须标注 `ERP-adapted`；`reference_ppd=30` 不是物理 HMD ppd。IMRE (TIP 2025) 因没有公开官方代码/checkpoint 只列 source blocker；
- Pathformer3D：同 50/9 split 三 seed、val-only 选模、每 content 200 条 scanpath；gate KLD/SIM/CC/10% coverage 为 `1.9143/0.3576/0.4311/0.8215`；decoder 首 token 为 declared-start unit vector；
- Corrected ScanDMM-v3：同 50/9 split 三 seed、500 epochs、val-only 选择 epoch `399/374/349`、每 content 200 条 scanpath；gate KLD/SIM/CC/10% coverage 为 `1.8030/0.3949/0.4819/0.8389`；独立 declared-start tensor 同时传入 model/guide，`obs_x_1` 仍为第一秒 target；
- 统一效率：RTX 5880 Ada FP32 prepared-input wall-clock 下，SphereHorizon/TempSAL-ERP 四张 map 分别为 `9.20/6.69 ms`；Pathformer3D/corrected ScanDMM 的 `200×20` scanpath 输出分别为 `517.8/174.3 ms`，因 workload 不同只分 tier 报告；
- 应用解释：SphereHorizon 相对 early template 的 10% coverage 绝对增益 `0.0173`，对应剩余 gaze-miss mass reduction `11.2% [5.0%,17.2%]`；该量是冻结主指标的派生解释，不是替换 endpoint；
- SaltiNet official checkpoint：论文披露其在 36/40 Salient360 training images 上 fine-tune，但仓库没有图像 ID，因此只能作为 overlap-disclosed diagnostic；
- 当前统一报告：[reports/kernel_v2_matched_baselines/summary.md](reports/kernel_v2_matched_baselines/summary.md)；完整性审计 [reports/kernel_v2_matched_baseline_audit/summary.md](reports/kernel_v2_matched_baseline_audit/summary.md) 为 `134/134 PASS`、无 caveat。旧 ScanDMM first-target initializer 只保留在 [语义比较](reports/scandmm_training_semantics_comparison/summary.md)，不得进入主表。定性 population map 只展示固定 central seed 的 200 paths；30 paths 用于轨迹运动，600 paths 仅作为三-seed Monte Carlo 稳定性参考。轨迹运动与采样数审计分别见 [motion audit](reports/kernel_v2_scanpath_motion_audit/summary.md) 和 [sample-count audit](reports/kernel_v2_scanpath_sample_count_audit/summary.md)。

### 13.5 Semantic re-ranking optimization lock — CLOSED

用户授权的 S0→S1→S2 只允许在 Salient360 train/val 内改善两个冻结目标：相对 TempSAL 的 5%/10% Coverage，以及相对 early template 的可归因图像语义增益。结果如下：

- S0 oracle PASS，证明 3 s 存在同图语义重排 headroom；
- S1 bounded residual 只改善 Cov@5，KLD/Cov@10/归因门失败；
- matched S2 semantic anchor 的 KLD/CC、Cov@10 和 correct-vs-wrong-image 归因通过，但 Cov@5 失败；
- S1/S2 train-only 选择的无训练融合仍不能同时超过 5% 与 10% 参考。

因此该分支判定 `STOP_NEW_ARCHITECTURE_EXPANSION / RETAIN_V10`。禁止继续搜索 loss 权重、alpha、tile grid、budget 或新增第三个 head；也禁止为失败候选运行三 seed、gate、Sitzmann 或 v2 outer-test。S2 只可作为“更好的 density/语义归因不自动转化为最低预算 tile ranking”的消融与 limitation。完整证据见 [reports/semantic_reranking_final/summary.md](reports/semantic_reranking_final/summary.md)。

### 13.6 V11 targeted TempSAL optimization lock — COMPLETE

用户在 S0–S2 关闭后重新明确授权了一个独立、受限目标：在固定 Salient360 split、`8×16` grid 和 Cov@5/10/20 下，针对 TempSAL-ERP 做定向优化，同时保持方法简单。该授权只覆盖 V11 审计与开发，不恢复复杂时序网络，也不允许重新使用 outer-test。

V11-A–F 依次检验了 raw-dwell tile supervision、SIM、多预算选模、tiny tile/pixel residual、temporal semantic anchor、split-half consensus rank 与 5 s template–semantic alpha。它们改善了部分 density 或单个预算，但没有稳定解决 Cov@5/10/20 与 5 s 的联合冲突，故全部关闭，不再扩 head 或 loss。

V11-G 的冻结定义为：

```text
C_t = Softmax_sph((1-alpha_t) log P_t + alpha_t log S_t)
P_t = train-only start/time prior
S_t = same-split TempSAL-ERP content prediction
```

协议约束：

1. `alpha_t` 只由 train contents 选择；val 只用于开发报告，gate 只打开一次且不反馈调参；
2. `0.5 s` 固定 `alpha=1`；`5 s` train-only 规则回退为 `alpha=1`，因此是 TempSAL exact no-op；
3. 三种子 train-only alpha 为 `{1,.85,1,1}`、`{1,.90,1,1}` 与 `{1,1,1,1}`，顺序对应 `0.5/1/3/5 s`；
4. 不增加神经网络参数，不使用 first fixation、prefix gaze、observer identity、val/gate prior 或 target statistics；
5. 不得根据 V11-G 的 val/gate 结果继续搜索 alpha grid、loss、horizon、tile budget、template 或 TempSAL checkpoint。

三种子 val 上，TempSAL→V11-G 的 KLD/SIM/CC/Cov@5/10/20 为 `1.1187→1.1070 / .4447→.4481 / .5881→.5917 / .7384→.7394 / .8371→.8383 / .9204→.9205`。一次性 gate 为 `1.1287→1.1173 / .4496→.4534 / .5762→.5810 / .7411→.7417 / .8604→.8608 / .9303→.9305`。两者六项均值均提高；map-density 三项 paired CI 均严格为正。Coverage 中只有 val Cov@5 CI 明确为正，其余 CI 接近或跨 0，因此只能声称 point-estimate overall improvement，禁止写成所有 Coverage 预算显著领先。

模型角色冻结为：**V10 = standalone scientific model；V11-G = candidate application adapter**。V11-G 达到当前工程目标，但其结构本身不足以承担“新 backbone”创新主张；论文贡献仍需由 early-attention task、start-aware product-of-experts 解释、严格无泄漏协议和 tile-prefetch 应用闭环共同支撑。完整证据：[reports/V11_OPTIMIZATION_SUMMARY.md](reports/V11_OPTIMIZATION_SUMMARY.md)。

### 13.7 V12 standalone spherical dwell-rate lock — CURRENT

用户在指出 V11-G 复用 TempSAL、创新性不足后，明确授权转向独立方法。V12 不恢复 prefix gaze、复杂 scanpath 生成或 temporal-final fusion；它直接在群体累计 dwell 任务上建立如下机制：

```text
A(I,p)                         = image semantic utility
C_i(p,s0)                     = spherical gaze-shift cost from declared start
R_i(p)                        = Softmax_sph(alpha_i A_i - C_i), R_i >= 0
D_t(p)                        = sum_{i:tau_i<=t} delta_tau_i R_i(p)
P_t(p)                        = spherical_normalize(D_t)
```

其中累计发生在未归一化 dwell mass `D_t` 上；只在计算 saliency density 时归一化。V12-C 在 `0–1 s` 使用由 train raw dwell 拟合的16个切平面横纵扩散/偏移参数，之后移动成本衰减并由图像语义主导。一个统一 Tile score 同时接受 Cov.@5/10/20 的 entropy-regularized differentiable top-k regret，不生成预算特定的三张 map。

执行和冻结记录：

1. V12-A 各向同性版本证明 density 与5秒可改善，但 Cov.@5 失败；
2. V12-B 在所有时段使用各向异性，修复 early Cov.@5，却把晚期内容分布误当作生理成本；
3. V12-C 只在前1秒保留 train-only 各向异性，结构和 loss 在 seed `20260711` 后冻结；seed `20260710/12` 仅复验；
4. 三个 seed 均由 val 选择 epoch 1。四分区 canonical-content leakage 为0，三个 run 只生成9个 val predictions；template-input no-op 单测通过；完整测试 `246 passed`；
5. Val early KLD/SIM/CC/Cov.@5/10/20 为 `1.0670/.4526/.6002/.7399/.8463/.9206`，六项均值高于 TempSAL；5秒六项也全部提高；
6. 冻结 historical gate early 为 `1.0667/.4672/.6109/.7365/.8636/.9322`：KLD/SIM/CC/Cov.@10/20 提高，但 Cov.@5 相对 TempSAL 为 `-0.00453 [-0.02330,+0.01439]`；5秒六项均提高；
7. Val/gate 3秒 correct-vs-wrong-image 的 KLD/SIM/CC attribution CI 全正，排除“只从起点扩散”的解释；
8. Gate 已用于稳定性评估但未用于任何选择；禁止根据该结果继续调结构、loss、球面校准、tile grid、预算或 checkpoint policy。Outer-test 仍不打开。

当前角色由本节更新为：**V12-C = standalone mainline candidate；V10 =旧 standalone reference；V11-G = TempSAL-dependent adapter reference**。允许的主张是 density、Cov.@10/20 和5秒的稳定改善，以及 val 上全部点估计改善；禁止声称 early Cov.@5 跨 split 全面领先。证据：[validation](reports/v12c_spherical_dwell_rate/summary.md)、[frozen gate](reports/v12c_spherical_dwell_rate_gate/summary.md)、[protocol audit](reports/v12c_spherical_dwell_rate/protocol_audit.md)。

### 13.8 V12-C frozen external transfer — COMPLETE / DOMINANCE FAIL

三个 epoch-1 checkpoint 的 SHA-256 在 Sitzmann 与 ET-DK2 evaluator 间完全一致；任何外部结果产生后均未重新选择 checkpoint，也未拟合 target-domain prior、reachability、loss、soft-topk、tile grid 或 budget。

1. Salient360→Sitzmann 使用预先物化的 `+128 px` coordinate-only view，将 official four-start conditions 对齐到 source-model yaw `90°`。V12-C early KLD/SIM/CC/Cov.@5/10/20 为 `1.7485/.2626/.4586/.8209/.9018/.9615`，TempSAL 为 `1.6538/.2873/.4808/.8514/.9107/.9713`；除 Cov.@10 CI 跨0外，其余差值 CI 严格为负。5秒只有 KLD/Cov.@10 点估计为正且 CI 跨0。
2. Sitzmann 3秒 correct-vs-wrong-image KLD/SIM/CC gain 为 `+.3320/+.0364/+.0748` 且 CI 全正。因此跨域失败不是模型完全忽略 content，而是 source semantic ranking/calibration 未超过 TempSAL。
3. ET-DK2 local split 与 Salient360 主 split 并未同步。Formal evaluator 排除了 source-train 的36 sessions与 sealed-outer P33 的18 sessions，只读取 source val/gate 的 P76/P13/P3 共54 sessions；ET-DK2 test targets和Salient360 outer targets均为0 loads。该结果只能称为 cross-start/collection-protocol diagnostic，禁止称为独立 cross-content test。
4. ET-DK2 early V12-C 相对 TempSAL 的 CC 为 `+.0160` 且 CI 为正，但 Cov.@5/10/20 为 `-.0174/-.0031/-.0074`，其中 Cov.@20 CI 严格为负。5秒 Cov.@20 为 `+.0174` 且 CI 为正，其余互有胜负。
5. ET-DK2 wrong-start 会使 KLD/SIM/CC 恶化 `2.9232/.1134/.2505` 且 CI 全正，wrong-image 则无稳定差异。结论是 explicit start dynamics 可跨 yaw/协议泛化，content re-ranking 尚不能。

最终判定：**FROZEN CROSS-DATASET DOMINANCE FAIL / RETAIN SALIENT360 IN-DOMAIN CLAIM**。外部结果只能作为机制边界和 limitation；禁止反馈调参。证据：[overall report](reports/v12c_frozen_transfer/summary.md)、[Sitzmann](reports/v12c_frozen_transfer_sitzmann/summary.md)、[ET-DK2](reports/v12c_frozen_transfer_etdk2/summary.md)、[integrity](reports/v12c_frozen_transfer/integrity_audit.md)。

## 14. Reproducibility Record

```yaml
protocol: start_aligned_early_horizon_saliency
protocol_version: 3_standard_train59_val13_test13
status: standard_split_rerun_complete_protocol_audit_pass
horizons_s: [0.5, 1.0, 3.0, 5.0]
primary_horizons_s: [0.5, 1.0, 3.0]
legal_inference_input: ERP_RGB + ERP_geometry + declared_initial_yaw + horizon
forbidden_inference_input: gaze_history + first_fixation + observer_id
primary_dataset: Salient360_population_scanpaths
condition_dataset: ET-DK2_sessions
split: data/splits/salient360_standard_train59_val13_test13_seed20260711.json
counts: {train: 59, val: 13, test: 13}
selection_split: val
report_split: test
test_repeatable: true
test_used_for_selection: false
target_renderer: unit_sphere_chord_gaussian
target_sigma_deg: 2.0
target_kernel_truncate_sigma: 4.0
legacy_target_renderer: fixed_erp_sigma10px_historical_only
tile_grid: [8, 16]
tile_budgets_percent: [5, 10, 15, 20]
primary_tile_budget_percent: 10
bootstrap_group: canonical_content_id
bootstrap_seed: 20260711
bootstrap_resamples: 10000
test_evaluated: true
test_driven_selection_authorized: false
semantic_reranking_decision: V11_A_to_F_closed_no_joint_multi_budget_solution
standalone_model: V12C_SphereRate
candidate_application_adapter: V11_G_TempSAL_train_only_start_time_prior_product
v11_gate_reused_for_tuning: false
v11_point_estimate_status: all_KLD_SIM_CC_Cov5_Cov10_Cov20_means_improve_on_val_and_gate
v11_statistical_boundary: density_CIs_positive_coverage_CIs_mostly_inconclusive_5s_exact_noop
v12c_role: current_standalone_mainline_candidate
v12c_inputs: ERP_image + declared_start + horizon
v12c_forbidden_inputs: TempSAL_prediction + early_template_values + gaze_history
v12c_val_status: all_KLD_SIM_CC_Cov5_Cov10_Cov20_means_improve
v12c_gate_status: KLD_SIM_CC_Cov10_Cov20_and_all_5s_means_improve_early_Cov5_below_TempSAL
v12c_gate_reused_for_tuning: false
v12c_frozen_external_status: Sitzmann_and_ETDK2_overall_TempSAL_dominance_fail
v12c_external_feedback_tuning: forbidden
etdk2_formal_external_subset: 54_sessions_3_source_val_gate_contents_train_and_outer_excluded
standard_baseline_benchmark: complete_no_missing_rows
standard_protocol_audit: PASS
test_suite: 251_passed
manuscript_status: provisional_icassp_draft_compiled_4_technical_plus_1_reference_page
venue_template_status: official_2027_kit_not_yet_public_using_2026_layout_harness
```

## 15. Session Checklist

1. 当前是否只做最早一个未完成的 kernel-correction 阶段？
2. EH0/EH1 结束前是否保持零新训练？
3. target 是否按真实秒数和截断 dwell 构造？
4. inference 是否完全不含 gaze history 或 first fixation？
5. 是否同时检验 map accuracy、attribution controls 和 tile coverage？
6. ET-DK2 是否按 session/稀疏条件诚实统计，而没有伪装成稳定 population GT？
7. 是否按 canonical content 聚类并保持 shared contents 同 split？
8. 是否明确标注旧协议历史，并确认当前 test 不参与 checkpoint、结构或超参数选择？
9. 是否没有搜索 horizons、tile grid、budget 或复杂结构以追逐结果？
10. 是否只在 train/val 冻结 baseline adapter，并把 scanpath 对比明确标成 secondary？
11. 是否避免在 EH6 PASS 前宣称外部竞争性证据完整？
12. 当前 target 是否明确记录 `sphere_chord / 2° / 4σ`，且没有混用 v1 checkpoint 或 map？
13. V11-G 是否保持 train-only alpha、gate 零调参，并明确区分“六项均值提高”和“Coverage 显著提高”？
14. V12-C 是否保持非负 raw-mass accumulation、无 TempSAL/template inference，并明确保留 gate early Cov.@5 缺口？
15. 是否明确记录 V12-C frozen external dominance FAIL，并排除 ET-DK2 source-train/outer overlap，拒绝用外部结果反馈调参？

任一答案为“否”，立即停止并回到本协议。

## 16. Current Authorized Action

K0/K1/K2 已完成，K3 在 Salient360 三 seed上 PASS、在 ET-DK2 template-improvement guard 上 FAIL。K4 matched development benchmark 已完成 TempSAL-ERP、Panoramic-CNN、BMS360/GBVS360、DeepGaze MSDB 2025 ERP adapter、reachable-final、corrected ScanDMM 与 Pathformer3D，完整性审计 `134/134 PASS`。K5 官方 Sitzmann standing-VR 重建完成；K6 又完成全部 learned baselines 的三 seed Sitzmann-only 训练、传统/近期 controls 与双向 frozen-transfer 矩阵。K6 完整性审计 PASS、sealed target loads=`0`。最终结论是 **Salient360 primary content-diverse benchmark + Sitzmann secondary in-domain/frozen-transfer track**，不是 Sitzmann-only 替换。统一主表/图、效率元数据、双源引用与 claim–evidence 审核已经组装成 [完整 provisional ICASSP 稿件](paper/main.tex)：4 页技术内容加第 5 页参考文献，编译与逐页渲染检查通过。官方 ICASSP 2027 kit/匿名规则尚未发布，当前 2026 style 仅作 layout harness。

V12-C 已替代 V11-G 成为 standalone mainline candidate。它在 val 的全部六项点估计、gate 的 density/Cov.@10/20 和 val/gate 的5秒均超过 TempSAL，并通过 wrong-image attribution；但 gate early Cov.@5 低 `0.00453` 且 CI 跨0，低预算全面领先尚未成立。

冻结 external transfer 已完成且总体相对 TempSAL **FAIL**：Sitzmann early 六项均低，ET-DK2 early Coverage 三项均值也低；仅起点归因和部分5秒指标能跨域。当前只授权：保持 V12-C checkpoint/结构/loss/校准完全冻结，做真实 tile/带宽收益、必要的已定义消融和论文证据更新；迁移到未来发布的2027官方模板；补全作者/单位/资助信息并执行严格 reviewer pass。禁止依据 Sitzmann、ET-DK2、Salient360 val 或已见 gate 继续搜索训练源、kernel、alpha、loss、soft-topk temperature、球面校准、模型结构、horizon、tile grid 或 budget；禁止把域内点估计改善写成跨数据集或所有 Coverage 全面领先。若未来解决跨域 content re-ranking，必须另立预注册 multi-domain/domain-robust 训练协议和合法未见 content split。Outer-test 不再打开。v1 checkpoints、targets、figures 和 EH0–EH6 数值全部只保留为历史诊断，不得进入 v2 主表。
