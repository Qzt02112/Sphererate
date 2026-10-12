# 工作区清理交付说明（2026-08-07）

## 结论

本次采用**可逆归档**，没有删除研究数据、checkpoint、报告或源码。所有退出当前主线的内容均按原相对路径移动到：

```text
temp/workspace_cleanup_20260807/
```

当前活动工作区只围绕以下研究范围组织：

1. Salient360 标准 `59 train / 13 val / 13 test`、球面 `2°` target；
2. SphereRate / V12-C 三种子主模型及机制消融；
3. TempSAL-ERP、Panoramic-CNN、BMS360、GBVS360、DeepGaze MSDB、Pathformer3D、ScanDMM、HAT 的标准划分对比；
4. 真实 HEVC tile-prefetch 应用实验；
5. Sitzmann scene-disjoint 同域训练与测试；
6. 当前论文、关键证据、源码、配置与活动测试。

## 活动目录

- `data/processed/early_horizon_salient360_spherical2deg_standard_v3/`
- `data/processed/standard_split_v3_matched_scanpaths/`
- `data/processed/sitzmann_official_256x512_spherical2deg_standard_v2/`
- `data/processed/salient360_256x512_k6_order/`
- `data/raw/salient360_official/`
- `data/interim/salient360_tile_hevc_qp22_qp42/`
- `runs/standard_split_v3/`
- `runs/sitzmann_standard_v2/`
- `reports/standard_split_*`
- `reports/sphererate_*`
- `reports/codec_tile_prefetch_test/`
- `reports/sitzmann_standard_v2/`
- `paper/`、`src/`、`scripts/`、`configs/`、`tests/`、`third_party/`

## 已归档内容

- 旧 Salient360 v2 split、kernel-v2 数据视图和 ET-DK2；
- 双向跨数据集 view/scanpath 与对应运行和报告；
- V11、旧 V12、semantic re-ranking 及旧 kernel-v2 运行；
- 未被 validation 选中的 Pathformer3D、ScanDMM、HAT 周期 checkpoint；
- 重复 `last.pt`、smoke run、中断运行和 selection sweep 中间产物；
- 已被当前图表替代的旧论文 figure/table；
- baseline 虚拟环境、Python/pytest 缓存、LaTeX 中间文件和旧预览；
- 依赖上述历史产物的 4 个旧测试文件。

归档总量约 **13.66 GB**。活动目录变化约为：

- `runs/`：12 GB → 2.8 GB；
- `reports/`：151 MB → 16 MB；
- `third_party/`：2.2 GB → 106 MB（源码保留，环境归档）；
- `data/`：9.3 GB → 7.2 GB；
- `tmp/`：214 MB → 空目录。

## 验证结果

- 标准 `59/13/13` 协议审计：`PASS`, `complete=true`；
- 当前图1可从保留的 seed-20260711 checkpoint 重新生成；
- 论文从无 `.aux` 的干净状态编译成功：12 页；
- 最终 PDF：`output/pdf/SphereHorizon_ICASSP_draft.pdf`；
- 活动测试集：**251 passed**；
- Pathformer3D、ScanDMM、HAT 三种子的 validation-selected checkpoint 均存在。

## 恢复方式

归档目录完整复刻原相对路径。恢复单个目录示例：

```bash
mkdir -p data/processed
mv temp/workspace_cleanup_20260807/data/processed/early_horizon_salient360_spherical2deg_v2 data/processed/
```

如需无损复制恢复全部归档内容、同时保留归档副本：

```bash
rsync -a \
  --exclude='_ARCHIVE_README.md' \
  --exclude='post_validation/' \
  temp/workspace_cleanup_20260807/ ./
```

清理规则记录在 `scripts/archive_workspace_20260807.sh`。该脚本为日期锁定的一次性操作；归档完成后再次执行会直接退出。
