# SphereRate

[English](README.md) | **简体中文**

**SphereRate（V12-C）** 的参考实现：面向 360° ERP 图像的**起始感知早段累积显著度预测**，
也是论文中全部训练、消融、基准对比与真实编码实验所使用的代码。

> **状态说明。** 这是**研究代码发布**，并非官方论文工程。对应论文**正在投稿评审中
> （ICASSP）**，下方结果取自当前稿件。逐样本预测缓存、数据集与训练好的权重**不包含**
> 在本仓库内（见[数据](#数据)与[复现论文数值](#复现论文数值)）。

---

## 任务定义

```text
输入 = ERP RGB + ERP 几何 + 系统声明的赤道起始位置 + 查询时刻 t
输出 = 累积人群注意分布 C_t，t 属于 {0.5, 1, 3, 5} 秒
用途 = 在 5 / 10 / 15 / 20 % 预算下对 8x16 个 ERP 瓦片排序
```

四个时刻都位于**早段观看窗口**内。模型刻意**不**使用 gaze prefix、真实首个注视点、
观察者身份、历史 scanpath、TempSAL 预测或 early-template 数值。唯一的条件信号是图像、
球面几何、系统声明的起始位置和查询时刻。

## 方法

SphereRate 预测球面上随时间变化的**非负区间停留率**，在未归一化的质量空间累加，
最后统一归一化：

```text
ERP RGB + 几何
    -> 冻结的同划分语义锚点
    -> 图像相关的语义效用 U_i(I)

系统声明的起始位置
    -> 切平面 east/north 几何
    -> 随时刻变化的移动代价 R_i(x; s)

语义效用 - 起始代价
    -> 非负区间停留率 r_i(x)
    -> 原始质量 M_k = sum_{i<=k} dt_i * r_i
    -> 球面归一化 C_k
    -> 同一张图同时用于密度预测与瓦片排序
```

$$
r_i(x)=\operatorname{Softmax}_{\mathrm{sph}}\bigl(\alpha_i[U(x;I)+E_i(x;I)]-R_i(x;s)\bigr),
\qquad
M_k(x)=\sum_{i=1}^{k}\Delta t_i\, r_i(x),
\qquad
C_k=\frac{M_k}{\int_{\mathbb S^2}M_k\,dA}.
$$

设计要点：

- **起始代价**回答"当前时段内该区域能否被合理到达"；
- **图像语义**回答"在可达范围内，哪里值得看"；
- **非负停留率**在质量空间累加，因此累积过程永远不会减去已看过的停留；
- **多预算瓦片损失**只对齐同一张输出图在部署预算下的排序，不增加预算专用 head；
- 总参数 `3,755,545`，其中可训练 `74,264`。

固定的 target 渲染协议：

```text
target renderer = 单位球弦长高斯
sigma           = 2 度视角
support         = 4 sigma
weighting       = 截断的停留时长
normalization   = 球面积
resolution      = 256 x 512
```

## 实验结果

四个时刻的 early macro，三个随机种子，内容不交叉的 Salient360 标准划分
`59 train / 13 val / 13 test`（仅用 `val` 选择 checkpoint）：

| 方法 | KLD↓ | SIM↑ | CC↑ | Cov.@5↑ | Cov.@10↑ | Cov.@15↑ | Cov.@20↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Early template | 1.231 | .431 | .533 | .596 | .721 | .803 | .848 |
| TempSAL-ERP | 1.131 | .441 | .568 | .633 | .748 | .828 | .872 |
| **SphereRate** | **1.072** | **.464** | **.584** | **.635** | **.765** | **.840** | **.880** |

机制消融（参数量、语义锚点、25 epochs、优化器、采样顺序与 val-only 选模全部对齐，
每行只关闭一个机制）：

| 变体 | KLD↓ | SIM↑ | CC↑ | Cov.@5↑ | Cov.@10↑ | Cov.@15↑ | Cov.@20↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **完整** | **1.072** | **.464** | **.584** | .635 | **.765** | .840 | **.880** |
| 去掉 rate accumulation | 1.085 | .454 | .573 | **.640** | .749 | .837 | .874 |
| 去掉 declared-start 代价 | 2.287 | .219 | .404 | .570 | .655 | .718 | .770 |
| 去掉 tile supervision | 1.075 | .463 | .583 | .635 | .764 | **.840** | .880 |

真实 HEVC 瓦片预取应用实验（原生分辨率 Salient360 ERP，8×16 独立瓦片，
`13 × 128 × 2 = 3328` 条码流，实测 byte cap 下的 gaze-PSNR / gaze coverage）：

| 方法 | 5% G-PSNR / Cov. | 10% | 15% | 20% |
|---|---:|---:|---:|---:|
| Early template | 33.002 / .472 | 34.969 / .655 | 35.886 / .737 | 36.539 / .794 |
| TempSAL-ERP | **33.470 / .525** | 35.241 / .687 | 36.225 / .765 | 36.794 / .818 |
| **SphereRate** | 33.217 / .499 | **35.311 / .696** | **36.395 / .779** | **36.931 / .829** |
| GT-attention oracle | 33.970 / .605 | 36.082 / .793 | 37.018 / .863 | 37.493 / .904 |

## 结论边界（如实声明）

以下边界是结果的一部分，不是事后补充的免责声明。**请勿**把本方法引用为在所有指标上
全面优于基线。

- **declared start 是决定性机制。** 去掉它所有指标都会崩塌。
- **early `Cov.@5` 不是显著优势。** 四个 paired macro coverage 置信区间均跨 0；
  提升主要出现在 `3 s` 和 `5 s`，而 `0.5 s` 仍由 start template 主导。
- **rate accumulation** 改善 SIM / CC / Cov.@10 / Cov.@20，但**不**改善 Cov.@5。
- **tile supervision** 只有很小且随预算变化的收益，**不是**主要创新来源。
- **真实编码只在中等预算领先。** SphereRate 在 10 / 15 / 20 % 相对 early template 提升
  `.342 / .509 / .392 dB`，但 **5% 预算明确失败**（比 TempSAL 少 `.026` 的 gaze
  coverage，CI `[-.055, -.003]`）。
- **冻结跨数据集迁移并不优于 TempSAL。** 在 Sitzmann 和合法的 ET-DK2 子集上 early
  指标更低；Salient360 同域基准才是主要证据。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `src/sphererate/models/` | SphereRate 模型、球面层、UNet backbone |
| `src/sphererate/data/` | 数据视图、划分、ERP/几何预处理 |
| `src/sphererate/evaluation/` | 球面指标、瓦片排序、编码预取 |
| `src/sphererate/baselines/` | baseline runner 与适配器 |
| `scripts/` | 数据 / 训练 / 评估 / 审计 / 绘图入口 |
| `configs/` | 主线与 baseline 配置（当前有效的是 `configs/standard_split/`） |
| `tests/` | 单元与契约测试 |
| `docs/` | 协议、框架、指标与版本说明 |
| `data/splits/`、`data/manifests/` | 冻结的划分定义与数据清单 |

## 安装

需要 **Python ≥ 3.10**，训练需要 CUDA。可编辑安装在较旧的解释器上需要先升级
`pip` / `setuptools`：

```bash
git clone https://github.com/Qzt02112/Sphererate.git
cd Sphererate
python -m pip install -U pip setuptools   # pip < 23 / setuptools < 68 时必须先执行
python -m pip install -e .
```

依赖：`numpy`、`Pillow>=9.1`、`PyYAML`、`requests`、`torch`、`torchvision`、`tqdm`。

### 路径配置

部分配置与辅助脚本把机器相关的根目录写成 `${...}` 占位符。它们会从环境变量展开，
所以请用环境变量而不是逐个改文件：

```bash
export SCANDMM_ROOT=/path/to/ScanDMM     # 原始数据 + 第三方 baseline 源码
export TEMPSAL_ROOT=/path/to/SphereRate  # 本项目在你机器上的根目录
```

未设置时占位符会原样保留为 `${SCANDMM_ROOT}` 字符串，方便你一眼看出哪里没配，
而不会静默指向错误的路径。

## 数据

三个数据集都**可公开下载**，本仓库不重新分发其中任何一个。

| 数据集 | 作用 | 来源 |
|---|---|---|
| **Salient360!** | 主要 early-horizon 人群目标 | <https://www.interdigital.com/data_sets/salient-360-dataset> |
| **Sitzmann VR saliency** | 次要同域 track / 冻结迁移 | <https://github.com/vsitzmann/vr-saliency> |
| **ET-DK2** | 外部多起点 sessions（CC-BY-4.0） | Zenodo record `10.5281/zenodo.4642612` |

重建处理后的数据视图（路径可在 `configs/` 中配置）：

```bash
# Salient360：标准 59/13/13 内容不交叉划分，256x512，球面 2 度 target
python scripts/build_standard_salient360_v3.py --config configs/standard_split/v12c_hybrid_dwell_rate.yaml

# Sitzmann：官方 standing-VR 重建（需要 data/sitzmann_raw_data.zip）
python scripts/prepare_sitzmann_official.py --archive data/sitzmann_raw_data.zip

# ET-DK2：Zenodo 下载器会校验官方 MD5，然后预处理 720 个公开 session
python scripts/download_etdk2.py --out-root "${SCANDMM_ROOT}/data/raw/etdk2"
python scripts/prepare_dataset.py --config configs/etdk2_256x512.yaml
```

预期的处理后目录（克隆后请调整 YAML 中的根路径）：

```text
data/processed/early_horizon_salient360_spherical2deg_standard_v3/
data/processed/salient360_256x512_k6_order/
data/processed/standard_split_v3_matched_scanpaths/
data/processed/sitzmann_official_256x512_spherical2deg_standard_v2/
data/interim/salient360_codec_originals/
```

## 训练与评估

三个固定种子（`20260710 / 20260711 / 20260712`）；只依据 `val` 选择 checkpoint。
`test` 可重复评估，但**从不**参与选模。

```bash
# 主线 SphereRate / V12-C，三种子
python scripts/batch_train.py \
  --config configs/standard_split/v12c_hybrid_dwell_rate.yaml \
  --batch  configs/standard_split/batch_v12c_hybrid_dwell_rate.yaml \
  --device cuda

# 机制消融
python scripts/batch_train.py --config configs/standard_split/v12c_hybrid_dwell_rate.yaml \
  --batch configs/standard_split/batch_v12c_ablation.yaml --device cuda

# 汇总主要 test 数值
python scripts/aggregate_standard_split_primary.py

# 真实 HEVC 瓦片预取应用
python scripts/evaluate_codec_tile_prefetch.py --config configs/standard_split/codec_application.yaml

# 对齐后的外部 baseline（ScanDMM / Pathformer3D / HAT / Panoramic-CNN 等）
python scripts/batch_train.py --config configs/standard_split/v12c_hybrid_dwell_rate.yaml \
  --batch configs/standard_split/matched_baselines.yaml --device cuda
```

指标定义见 [docs/metrics.md](docs/metrics.md)：球面 KLD / SIM / CC，以及瓦片的
`Cov.@5/10/15/20`。

## 测试

```bash
python -m pip install pytest
python -m pytest -q
```

最小检出下的预期结果：`241 passed, 8 skipped`。跳过的都是需要导入上游 baseline 源码
（ScanDMM / Pathformer3D / HAT）的适配器契约测试——这些源码**没有**内置于本仓库。
把 `SCANDMM_ROOT` 与 `TEMPSAL_ROOT` 指向含有 `third_party/` 的检出目录即可运行它们。

## 复现论文数值

本仓库包含**代码、配置、固定划分与数据清单**。要精确复现表格，还需要上面三个数据集，
然后重新训练与评估。论文使用的冻结证据（validation 选出的 checkpoint，以及各报告
summary 与 paired-bootstrap 结果）位于**完整交付包**中，因体积与数据集条款限制
刻意未提交到本仓库。

## 第三方 baseline

对比方法**不在本仓库重新分发**。此处仅列出出处以便致谢；请从各自仓库获取，
并遵守各自的许可证。

| Baseline | 类别 | 来源 |
|---|---|---|
| TempSAL | temporal saliency | <https://github.com/IVRL/Tempsal> |
| Panoramic-CNN-360-Saliency | static 360 saliency | <https://github.com/DaniMS-ZGZ/Panoramic-CNN-360-Saliency> |
| GBVS360 / BMS360 / ProSal | traditional saliency | <https://github.com/Telecommunication-Telemedia-Assessment/GBVS360-BMS360-ProSal> |
| DeepGaze MSDB | static saliency | <https://github.com/matthias-k/deepgaze/tree/feature-deepgaze-msdb> |
| ScanDMM | scanpath-to-saliency | <https://github.com/xiangjieSui/ScanDMM> |
| Pathformer3D | scanpath-to-saliency | <https://github.com/lsztzp/Pathformer3D> |
| HAT | transformer scanpath-to-saliency | <https://github.com/cvlab-stonybrook/HAT> |
| 360-aware saliency | traditional saliency | <https://github.com/MikhailStartsev/360_aware_saliency> |
| SalBiNet360 | static 360 saliency | <https://github.com/githubcbob/SalBiNet360> |

感谢以上所有作者公开其代码与数据。

## 引用

论文正在评审中，公开后会补充 BibTeX；在此之前请引用本仓库。

## 联系方式

问题与 bug 反馈：**271218869@qq.com**

## 许可证

以 [MIT License](LICENSE) 发布 —— © 2026 Qiu Zhaotian。该许可仅覆盖本仓库中的代码；
第三方 baseline 仍遵循各自的许可证。
