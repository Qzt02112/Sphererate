# Future-Saliency Data Preparation

更新时间：2026-07-14

> 当前 primary content-diverse 训练数据仍为 Salient360；官方重建的 Sitzmann 同时承担 secondary in-domain matched track 与双向 frozen-transfer 验证。ET-DK2 提供多起点外部应用证据。旧 ScanDMM-derived Sitzmann、JUFE 和旧 Multi360 合并视图均不得恢复。权威边界见 [RESEARCH_PROTOCOL.md](../RESEARCH_PROTOCOL.md)。

## 当前数据角色

| Dataset | Role | Rows | Canonical contents | Notes |
|---|---|---:|---:|---|
| ET-DK2 | external multi-start sessions | 720 | 40 | 18 subjects，8 starts，只保留公开 `P#` |
| Salient360 | primary early-horizon population target | 85 | 85 | 与 ET-DK2 同名 `P#` 严格同 split |
| Sitzmann | secondary in-domain matched track / bidirectional frozen transfer | 88 | 22 | 780 trials；四类官方 `viewpoint_idx`；按 scene 分组 |

AOI 已排除；JUFE 为 diagnostic-only。

Sitzmann 官方原始归档保留为 `data/sitzmann_raw_data.zip`，SHA-256 为 `26b221416e85ccc03cdecc033e54120da953968a8a9c54a6aec09bd155a4df9f`。当前重建直接读取官方 standing-VR pickle，而不是旧 ScanDMM 转换或从 first fixation 推断起点。

## Official Sitzmann standing-VR reconstruction

```bash
python scripts/prepare_sitzmann_official.py
```

输出：

```text
data/interim/sitzmann_official_vr_v1
data/processed/sitzmann_official_256x512_spherical2deg_v1
data/splits/sitzmann_official_contentdisjoint_seed20260713.json
```

重建规则：

- 只读取 `data/vr/cubemap_0000.pck` 至 `cubemap_0021.pck`，明确排除 browser 与 seated-VR；
- 解析 780 个 trials、117 个全局 subject IDs；同 subject 在同一 scene-start cell 的 6 个重复 trials 共用 observer ID，因此得到 774 个 cell-level physical observers；
- saliency-map GT 继续由官方 fixation 起止时间与 clipped dwell 构建；scanpath models 则读取官方 dense `gaze_lat_lon` 与 `timestamp`，在最近真实采样处提取 `0..29 s` 的 1 Hz 序列，不能用 fixation interval 伪造 trajectory；
- `viewpoint_idx={0,1,2,3}` 对应 yaw `{0°,90°,180°,270°}`；图像、fixation 与 target 同步旋转，使 declared start 落在 ERP 水平中心 `x=0.5`；
- target 使用与主线一致的 clipped dwell、`sphere_chord / σ=2° / 4σ` 和球面积归一化；
- 22 scenes 严格分为 `16 train / 3 val / 3 published-test gate`；同一 scene 的四个起点永不跨 split；published test 不是 fresh holdout；
- train-only template 对 train scene 采用 leave-one-scene-out，并同时移除该 scene 的四个条件，避免 rotation leakage。

完整性结果：88/88 targets 可读，四起点齐全、scene leakage 为 0、非法球面积分为 0；dense scanpath 为 780/780 trials，train/val/gate=`547/117/116`，dropped=`0`、repaired bins=`0`。完整矩阵结论是 `Salient360 primary content-diverse benchmark + Sitzmann secondary in-domain/frozen-transfer track`，见 [data suitability](../reports/sitzmann_training_suitability/summary.md)、[controlled comparison](../reports/sitzmann_train_source_comparison/summary.md) 和 [full cross-dataset benchmark](../reports/sitzmann_cross_dataset_benchmark/summary.md)。

## 官方 ET-DK2 raw data

来源：Zenodo v3 record `4642612`，CC-BY-4.0。

Raw root：

```text
${SCANDMM_ROOT}/data/raw/etdk2
```

下载与校验：

```bash
python scripts/download_etdk2.py \
  --out-root ${SCANDMM_ROOT}/data/raw/etdk2
```

下载器校验 Zenodo 官方 MD5，并记录：

```text
${SCANDMM_ROOT}/data/raw/etdk2/source_record.json
```

官方数据实际包含：

- 18个 participant CSV；
- 每人50个有效25秒试次；
- 40个公开 Salient360 `P#`；
- 10个未公开 construction `C#`；
- 每人一个单行、0秒的日志毛刺。

预处理排除180个 `C#` sessions和18个毛刺，保留720个公开 sessions。

数据限制：每个公开内容有18个sessions，但8类start并非逐图完全平衡。每图覆盖6–8类start；320个content-start cells中35个为空，65个非空cells只有1个observer。因此后续D1不能直接按image-start cell构造稳定population target。

## ET-DK2 session preparation

```bash
python scripts/prepare_dataset.py --config configs/etdk2_256x512.yaml
```

输出：

```text
data/processed/etdk2_salient360shared_256x512_k6_session_v1
```

每个 session manifest row 包含：

- `canonical_content_id`、`subject_id`、`trial_index`；
- `start_index`、start longitude/latitude；
- duration、sample counts、raw row range；
- image、trace、target、fixations和raw provenance。

Trace NPZ 保留 gaze/head 2D和3D量；target NPZ保存可重建的六段 temporal/cumulative/full maps。正式 future target 要在 D1 锁定 `τ` 后重新从 trace/fixations 确定性生成。

## Current canonical-content split

当前工具直接读取各自 manifest，不再经过已删除的三数据集合并视图：

```text
data/processed/early_horizon_salient360_spherical2deg_v2/manifest.jsonl
data/processed/etdk2_salient360shared_256x512_k6_session_v1/manifest.jsonl
data/splits/future_saliency_contentdisjoint_seed20260711.json
```

ET-DK2 `P#` 与 Salient360 `P#` 共用 `salient360:P#` 和同一 split。当前 v2 主线的正式完整性审计见 `reports/kernel_v2_matched_baseline_audit/summary.md`；外部多起点诊断见 `reports/kernel_v2_etdk2/summary.md`。

## Reproduction tests

```bash
pytest -q \
  tests/test_etdk2_download.py \
  tests/test_etdk2.py \
  tests/test_etdk2_prepare.py \
  tests/test_group_disjoint_splits.py \
  tests/test_future_saliency_data_view.py \
  tests/test_future_saliency_data_audit.py
```

## Removed historical views

JUFE、旧 Multi360、P2、旧 Sitzmann 和三数据集合并视图已于 2026-07-13 删除。报告与构建代码仅用于解释历史决策；不得从这些路线启动新训练。清理清单见 [workspace_cleanup_20260713.md](workspace_cleanup_20260713.md)。
