# 工作区保留与清理清单（2026-07-15）

## 原则

本次清理以“论文和后续两项研究目标能直接继续”为边界。保留不可替代的原始数据、主线处理数据、selection-audited checkpoint、可复核的 JSON/Markdown 证据、论文源码和测试；删除可重建环境、缓存、预测副本、周期性中间 checkpoint、smoke run，以及已有冻结汇总替代的旧路线产物。

删除操作由 `scripts/cleanup_workspace_20260715.sh` 明确列出，不使用通配式删除整个 `data/`、`runs/` 或 `reports/`。清理前总占用约 20 GiB，其中 `data/` 7,021,936,548 B、`runs/` 10,325,290,878 B、`reports/` 325,632,277 B、`archive/` 741,218,211 B、`third_party/` 2,155,302,319 B。

## 必须保留

### 1. 原始数据和主线处理数据

- `data/sitzmann_raw_data.zip`：Sitzmann 官方原始包；SHA-256 `26b221416e85ccc03cdecc033e54120da953968a8a9c54a6aec09bd155a4df9f`。
- `data/interim/sitzmann_official_vr_v1/`：从官方包解析出的最小中间表示。
- `data/processed/early_horizon_salient360_spherical2deg_v2/`：Salient360 主线 early cumulative target 与 split。
- `data/processed/salient360_256x512_k6_order/`：主线 ERP 图像与轨迹来源。
- `data/processed/sitzmann_official_256x512_spherical2deg_v1/`：Sitzmann 官方重建后的四起点数据。
- `data/processed/kernel_v2_matched_scanpaths/`、`data/processed/sitzmann_official_matched_scanpaths_v1/`：ScanDMM/Pathformer3D/HAT 的匹配输入。
- `data/processed/cross_view_*_v1/`、`data/processed/cross_scanpaths_*_v1/`：双向冻结迁移的坐标对齐视图和轨迹。
- `data/processed/etdk2_salient360shared_256x512_k6_session_v1/`：当前唯一保留的非等纬起点/跨会话外部诊断集，暂不删除，避免无法从本地重建。

### 2. 主线模型 checkpoint

- SphereHorizon 三种子：
  - `runs/kernel_v2_candidate_correct_seed20260710/checkpoints/best.pt` — `3589ba916dd4b93b8c93ffcc88bf327c996d014ba2b368335f190ae73cefecbb`
  - `runs/kernel_v2_candidate_correct_seed20260711/checkpoints/best.pt` — `7cdca7ee15be12dbcd8198760cc9eaf9c76ac463b04d927cd793a591d995edb5`
  - `runs/kernel_v2_candidate_correct_seed20260712/checkpoints/best.pt` — `3695c3de2db78e165742af373b1bed17df794dbe56c4db362c9f9d095e0d054b`
- TempSAL-ERP、Panoramic-CNN、Pathformer3D、ScanDMM 三种子的 validation-selected checkpoint。
- HAT：seed 20260710/11/12 分别保留 `step_004000.pt`、`step_005000.pt`、`step_004000.pt`；选择依据为 `reports/kernel_v2_hat/selection_audit.json`。
- Sitzmann-only 与 Salient360$\rightarrow$Sitzmann fine-tune：每种子保留一个 validation-selected/best checkpoint。
- 语义重排序后续入口：保留 S1、S2、S2-matched-anchor 的非 smoke `best.pt` 与训练元数据；它们是下一轮 Cov.@5/10/15 优化的可复用起点，但不作为当前论文结论。

### 3. 结果证据与论文

- 主结果：`reports/kernel_v2_matched_baselines/`、`reports/kernel_v2_matched_quantitative/`、`reports/kernel_v2_horizon_behavior/`。
- 跨数据集：`reports/sitzmann_train_source_comparison/`、`reports/sitzmann_cross_dataset_benchmark/`；后者 `results.json` SHA-256 为 `0eca8b9b969cdde567cf8e6bdb0b91ba8c2a076c5091fb32c2a69a8995518ead`。
- 机制与审计：所有仍由 `Goal.md`、`RESEARCH_PROTOCOL.md`、`MAINLINE_MODEL_VERSIONS.md`、论文或测试直接引用的 `kernel_v2_*`、`semantic_reranking_*`、Sitzmann rebuild/source comparison 和 source audit 报告。
- 论文：`paper/main.tex`、`paper/references.bib`、`paper/tables/`、`paper/figures/`，以及交付 PDF。
- 实现与验证：`src/`、`scripts/`、`configs/`、`tests/`、`docs/`。

## 明确删除

- `archive/off_mainline_nextfix_20260710/`：已封存且不在当前主线的旧预测 CSV。
- `third_party/.envs/`：可由环境说明重建，不属于模型源码。
- `data/processed/early_horizon_salient360_v1/` 与 `data/processed/salient360_128x256_k6_order/`：分别被 spherical-2deg v2 和 256x512 主线替代。
- HAT、Sitzmann-only ScanDMM/Pathformer3D 的未选中周期 checkpoint；全部 smoke run；所有可重新生成的 `predictions/`；有 `best.pt` 时的重复 `last.pt`。
- closed-route/旧 kernel 可视化、JUFE/P1/final-map 试验、旧 EH/F0/F1/F2 报告；selection sweep 仅保留最终选中条目和汇总。
- Python/pytest/LaTeX 临时缓存与历史日志。论文编译产物会在验证阶段重新生成。

## 清理后状态

清理脚本已执行。工作区总占用从约 20 GiB 降至 8.8 GiB；关键目录当前为：`data/` 6,666,846,688 B、`runs/` 2,393,404,304 B、`reports/` 137,778,383 B、`third_party/` 107,873,304 B，空 `archive/` 仅保留目录本身。按清理前后五个主要目录计算，共释放 11,263,473,458 B（约 10.49 GiB）。

删除项不可由 Git 恢复（该目录不是 Git 仓库），但删除范围均为可再生项或已有冻结证据替代项；Sitzmann 原始包和三种子 SphereHorizon checksum 未变化。

最终验证：

- `pytest -q`：236 passed；pytest 显式使用项目根目录与 `src/`，不再依赖残留模块缓存。
- `paper/main.tex`：清理后强制重编译成功，6 页，无 overfull、undefined reference 或 LaTeX warning。
- 新增 `paper/tables/table_sitzmann_transfer.tex`，Salient360$\rightarrow$Sitzmann 冻结结果已进入正文；最终 PDF 为 `output/pdf/SphereHorizon_ICASSP_draft.pdf`。
- Sitzmann 原始包与三种子 SphereHorizon `best.pt` 均重新执行 SHA-256 检查并通过。
