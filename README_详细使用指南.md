# 舰船分类项目详细使用指南（仅新分层流程）

本指南只保留当前分层后的脚本，不再使用旧兼容入口。

## 1. 当前代码流程（与你理解一致）

1. 数据集划分：将原始数据划分为 `train/`、`val/`、`test/`。
2. 四模型训练：分别训练 `resnet`、`vgg`、`vit`、`vit_fusion`。
3. 鲁棒性评估：在 3 个 seed 和 4 个场景（`clean` + `light/medium/heavy`）下评估。
4. 斜率排序与绘图：计算 ranking，输出两张鲁棒性图。
5. 可视化分析：导出样本注意力图/热力图（Grad-CAM、ViT rollout）。

## 2. 核心脚本（新分层）

- 数据处理
  - `utils/split_dataset.py`

- 训练
  - `train/train_resnet.py`
  - `train/train_vgg.py`
  - `train/train_vit.py`
  - `train/train_vit_fusion.py`

- 鲁棒性实验
  - `utils/robustness/occlusion_suite.py`
  - `utils/robustness/slope_ranking.py`

- 可视化
  - `utils/visualization/robustness_plots.py`
  - `utils/analysis.py`（`visuals` 子命令）
  - `utils/generate_visuals.py`

- 一键流程
  - `utils/pipelines/full_robustness_pipeline.py`

## 3. 输出目录规范

- 评估结果：`outputs/evaluation/<run_name>/`
- 注意力图：`outputs/visualizations/attention/<run_name>/`
- 鲁棒性：`outputs/robustness/<experiment_name>/`
  - `runs/`
  - `metrics/`
  - `ranking/`
  - `plots/`

## 4. 环境准备

```bash
cd /mnt/e/aircas/dht/graduation
conda activate one
python -m pip install torch torchvision scikit-learn matplotlib opencv-python pillow tqdm pandas openpyxl
```

## 5. 按步骤执行

### Step 1. 划分数据集

```bash
conda activate one
python utils/split_dataset.py --dataset_subdir dataset/ship_fine --source FGSCR
```

### Step 2. 训练四个模型
# clean训练（不加训练遮挡）
```bash
conda activate one
cd graduation
python train/train_resnet.py --train_occlusion_mode none --train_occlusion_p 0.0
python train/train_vgg.py --train_occlusion_mode none --train_occlusion_p 0.0
python train/train_vit.py --train_occlusion_mode none --train_occlusion_p 0.0
python train/train_vit_fusion.py --train_occlusion_mode none --train_occlusion_p 0.0
```

说明：

- 上述命令即论文主实验的“clean 训练”设置。
- 若需严格可复现实验，请给四个训练脚本都加同一个 `--seed`（例如 `--seed 42`）。

### Step 3. 运行评估以生成预测 CSV / 报告（用于可视化与后续分析）

在生成注意力图或运行鲁棒性实验前，需要先对模型做一次评估以生成预测 CSV（`preds_{model}.csv`）、分类报告与混淆矩阵等。可直接使用评估入口：

```bash
conda activate one
# 直接调用评估脚本（推荐）
python utils/evaluate_models.py --models vgg --dataset_subdir dataset/ship_fine --test_split test --output_subdir outputs/evaluation/default_eval

# 或使用统一的分析入口（同样会调用 evaluate_models）
python utils/analysis.py eval --models vit --output_subdir outputs/evaluation/default_eval
```

输出示例文件（位于 `outputs/evaluation/default_eval/`）：

- `preds_vit.csv` — 预测表（用于 `utils/analysis.py visuals` / `utils/generate_visuals.py`）
- `report_vit.txt` — 分类报告和汇总指标
- `per_class_vit.csv` — 每类指标（recall/overall_acc/support）
- `class_indices_eval.json` — 类索引映射（index -> class name）
- `confmat_vit.png` — 混淆矩阵图片
- `per_class_vit.png` — 每类 recall 柱状图

注意：如果你在运行 `utils/analysis.py visuals` 时看到 `FileNotFoundError: Predictions CSV not found`，请先运行本步骤以生成对应的 `preds_{model}.csv`，或确认 `--csv` 路径是指向存在的文件（可使用项目根相对路径）。

### Step 4. 跑鲁棒性评估（clean + 遮挡测试）

```bash
conda activate one
# 主实验：clean测试 + light/medium/heavy遮挡测试
python utils/robustness/occlusion_suite.py --include_clean --seeds 42 123 3407 --occlusion_mode mixed --occlusion_levels light medium heavy --experiment_name keypart_experiments
```

说明：

- 该步骤不会重新训练模型，只会读取 `weights/` 下训练好的权重进行测试。
- 目前遮挡填充值已采用通道均值（`IMAGENET_MEAN`），避免纯黑遮挡带来的非物理高频边缘。

### Step 5. 计算斜率并绘制两张图

```bash
conda activate one
python utils/robustness/slope_ranking.py --occlusion_mode mixed --experiment_name keypart_experiments
python utils/visualization/robustness_plots.py --experiment_name keypart_experiments
```

说明：

- `slope_ranking.py` 已使用真实遮挡比例 `clean=0.00, light=0.10, medium=0.20, heavy=0.35` 计算斜率。
- 同时输出 AUPC / AUPC(norm) 指标，更适合非线性退化曲线比较。

生成图文件：
- `outputs/robustness/keypart_experiments/plots/robustness_composite_ranking.png`
- `outputs/robustness/keypart_experiments/plots/robustness_ranking_visualization.png`

### Step 6. 样本注意力/热力图导出

```bash
conda activate one
python utils/analysis.py visuals --model resnet --csv outputs/evaluation/default_eval/preds_resnet.csv --n 3 --out_dir outputs/visualizations/attention/default_eval
```

## 6. 一键版（可选）

```bash
conda activate one
python utils/pipelines/full_robustness_pipeline.py --include_clean --experiment_name keypart_experiments
```

## 7. 说明
 - `robustness_composite_ranking.png` 现在使用 [0,1] 区间归一化（最差为 0，最好为 1）。

## 8. 论文主实验指令（可直接复制）

### 8.1 Clean 训练（四模型）

```bash
conda activate one
python train/train_resnet.py --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42
python train/train_vgg.py --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42
python train/train_vit.py --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42
python train/train_vit_fusion.py --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42
```

### 8.2 遮挡测试（3 seeds, clean+light+medium+heavy）

```bash
conda activate one
python utils/robustness/occlusion_suite.py \
  --include_clean \
  --models resnet vgg vit vit_fusion \
  --seeds 42 123 3407 \
  --occlusion_mode mixed \
  --occlusion_levels light medium heavy \
  --occlusion_p 1.0 \
  --experiment_name keypart_experiments

python utils/robustness/slope_ranking.py \
  --occlusion_mode mixed \
  --experiment_name keypart_experiments

python utils/visualization/robustness_plots.py \
  --experiment_name keypart_experiments

# 可选：按自定义权重合成 AUPC/斜率/下降量（默认 aupc:0.5, slope:0.25, drop:0.25）
python utils/visualization/robustness_plots.py --experiment_name keypart_experiments --aupc_weight 0.5 --slope_weight 0.25 --drop_weight 0.25

# 可选：零侵入整理论文结果（仅复制/重命名，不重跑训练与评估）
# 1) 先生成主文汇总图（含 fig03~fig06，图中文字默认中文）
python utils/visualization/paper_summary_plots.py --experiment_name keypart_experiments --occlusion_mode mixed --seed_for_confmat 42

# 2) 再打包到 main_text / appendix 分层目录
python utils/pipelines/package_paper_results.py --experiment_name keypart_experiments --tag main --include_full_runs --overwrite
```

生成后的主文图文件（位于 `outputs/paper_results/keypart_experiments_main/main_text/figures/`）：

- `fig01_robustness_composite_ranking.png`
- `fig02_robustness_metric_panels.png`
- `fig03_macro_balanced_curves_mean_std.png`
- `fig04_per_class_recall_drop_heatmap_idx.png`
- `fig05_seed_variance_curves.png`
- `fig06_confmat_clean_vs_heavy_top2.png`

对应主文表（位于 `outputs/paper_results/keypart_experiments_main/main_text/tables/`）：

- `tab01_summary_keypart_metrics_agg.csv/.json`
- `tab02_robustness_slope_ranking.csv/.json`
- `tab03_curve_mean_std.csv`
- `tab04_per_class_recall_drop.csv`
- `tab05_seed_curve_points.csv`

## 9. 论文图的推荐解读写法（可直接用于正文）

### fig01 综合鲁棒性排序图

- 作用：给出总体结论，回答“谁最稳健”。
- 结论句模板：
  - `在综合得分下，模型A显著优于模型B/C，说明其在多种遮挡强度下具有更稳定的判别能力。`

### fig02 指标分解图（斜率/下降量等）

- 作用：解释 fig01 的来源，避免“黑箱式综合分”。
- 结论句模板：
  - `模型A在|slope|和clean→heavy下降量上均更小，表明其退化速度更缓。`

### fig03 均值±标准差曲线图

- 作用：展示“随遮挡比例增加，性能如何变化”，并体现跨 seed 的离散程度。
- 结论句模板：
  - `随着遮挡比例从0.00增加到0.35，各模型macro-F1与balanced-accuracy整体下降；模型A曲线始终位于上方且误差带较窄。`

### fig04 各类别 Recall 下降热图

- 作用：展示类别层面的脆弱性分布，回答“哪些类最容易受遮挡影响”。
- 结论句模板：
  - `在heavy遮挡下，若干类别出现集中退化（暖色更明显），提示这些类别对关键部位遮挡更敏感。`

### fig05 seed 细线 + 均值粗线图

- 作用：专门回答“结果是否稳定、是否偶然”。
- 结论句模板：
  - `各seed曲线与均值曲线趋势一致，说明退化规律稳定可复现；模型A在不同seed下波动较小。`

### fig06 clean/heavy 混淆矩阵对照图（Top-2）

- 作用：展示错误类型迁移，回答“错成了什么类”。
- 结论句模板：
  - `与clean相比，heavy场景下混淆主要集中在若干相近舰型之间，模型A的非对角增强幅度小于模型B。`

## 10. 写作注意点

- 不要只报综合分：正文至少同时给 fig01 + fig03 + fig06（总体、趋势、错误结构）。
- 热图建议放主文，完整 per-class/confmat 全量图放附录（`appendix/full_runs`）。
- 若提到稳定性，建议同时引用 `tab03_curve_mean_std.csv` 与 `tab05_seed_curve_points.csv`。
