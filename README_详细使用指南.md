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
cd /mnt/e/aircas/dht/graduation_fail
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

```bash
conda activate one
python train/train_resnet.py
python train/train_vgg.py
python train/train_vit.py
python train/train_vit_fusion.py
```

### Step 3. 运行评估以生成预测 CSV / 报告（用于可视化与后续分析）

在生成注意力图或运行鲁棒性实验前，需要先对模型做一次评估以生成预测 CSV（`preds_{model}.csv`）、分类报告与混淆矩阵等。可直接使用评估入口：

```bash
conda activate one
# 直接调用评估脚本（推荐）
python utils/evaluate_models.py --models vit --dataset_subdir dataset/ship_fine --test_split test --output_subdir outputs/evaluation/default_eval

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

### Step 4. 跑鲁棒性评估（3 seed + 4 场景）

```bash
conda activate one
python utils/robustness/occlusion_suite.py --include_clean --seeds 42 123 3407 --occlusion_mode mixed --occlusion_levels light medium heavy --experiment_name keypart_experiments
```

### Step 4. 计算斜率并绘制两张图

```bash
conda activate one
python utils/robustness/slope_ranking.py --occlusion_mode mixed --experiment_name keypart_experiments
python utils/visualization/robustness_plots.py --experiment_name keypart_experiments
```

生成图文件：
- `outputs/robustness/keypart_experiments/plots/robustness_composite_ranking.png`
- `outputs/robustness/keypart_experiments/plots/robustness_ranking_visualization.png`

### Step 5. 样本注意力/热力图导出

```bash
conda activate one
python utils/analysis.py visuals --model vit --csv outputs/evaluation/default_eval/preds_vit.csv --n 5 --out_dir outputs/visualizations/attention/default_eval
```

## 6. 一键版（可选）

```bash
conda activate one
python utils/pipelines/full_robustness_pipeline.py --include_clean --experiment_name keypart_experiments
```

## 7. 说明

- 本仓库已去除旧兼容入口，只保留新分层脚本。
- `robustness_composite_ranking.png` 使用带下限归一化，最差模型不会显示为 0，避免误读。
