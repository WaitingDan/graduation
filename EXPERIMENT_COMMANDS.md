# 实验命令清单（只写非默认参数）

在项目根目录执行：

```bash
cd /mnt/e/aircas/dht/graduation
```

## 1) 训练（公平对比配置）

> 下面命令仅包含**非默认参数**。默认值不重复写（如 `--batch_size 32` 不再出现）。

```bash
python train/train_resnet.py --label_smoothing 0.1 --train_occlusion_mode mixed --train_occlusion_level medium --train_occlusion_p 0.4
python train/train_vgg.py --label_smoothing 0.1 --train_occlusion_mode mixed --train_occlusion_level medium --train_occlusion_p 0.4
python train/train_vit.py --label_smoothing 0.1 --train_occlusion_mode mixed --train_occlusion_level medium --train_occlusion_p 0.4
python train/train_vit_fusion.py --label_smoothing 0.1 --loss_w_global 0.2 --loss_w_local 0.2 --loss_w_fusion 0.6 --dropout 0.3 --train_occlusion_mode mixed --train_occlusion_level medium --train_occlusion_p 0.4
```

## 2) 常规评估

```bash
python utils/analysis.py eval --models resnet vgg vit vit_fusion --seed 42
```

## 3) 单次重遮挡评估（快速看鲁棒性）

```bash
python utils/evaluate_models.py --models resnet vgg vit vit_fusion --eval_occlusion_mode mixed --eval_occlusion_level heavy --eval_occlusion_p 1.0 --output_subdir outputs/fair_eval_heavy --seed 42
```

## 4) Key-part 完整流程（核心）

### Step A: 跑多场景多种子评估

```bash
python utils/keypart_experiments/run_occlusion_suite.py --include_clean --output_subdir outputs/keypart_experiments
```

### Step B: 计算下降斜率并排序

```bash
python utils/keypart_experiments/analyze_robustness_slope.py --occlusion_mode mixed
```

### Step C: 出图

```bash
python utils/visualize_robustness_ranking.py
```

### Step D: 导出论文草稿（可选）

```bash
python utils/keypart_experiments/export_report.py --out_md outputs/keypart_experiments/chapter4_draft.md
```

## 5) 关键脚本参数含义（你关心的）

### 训练常用参数

- `--lr`：学习率（Learning Rate），控制每次参数更新步长。
- `--weight_decay`：权重衰减（L2 正则），抑制过拟合。
- `--label_smoothing`：标签平滑系数，降低过度自信。
- `--accum_steps`：梯度累积步数，用于等效增大 batch（省显存）。
- `--train_occlusion_mode`：训练遮挡类型（`none/block/stripe/mixed`）。
- `--train_occlusion_level`：遮挡强度（`light/medium/heavy`）。
- `--train_occlusion_p`：训练样本应用遮挡的概率（0~1）。

### `run_occlusion_suite.py` 常用参数

- `--include_clean`：是否加入 clean 场景。
- `--models`：参与评估模型列表。
- `--seeds`：多随机种子列表（用于均值/方差统计）。
- `--occlusion_mode`：遮挡模式（默认 `mixed`）。
- `--occlusion_levels`：遮挡等级列表（默认 `light medium heavy`）。
- `--occlusion_p`：评估时遮挡概率（默认 1.0）。
- `--output_subdir`：结果输出根目录。
- `--no_csv`：不写 CSV，仅写 JSON。

### `analyze_robustness_slope.py` 常用参数

- `--agg_csv` / `--agg_json`：输入聚合文件（默认读取 `outputs/keypart_experiments/summary_keypart_metrics_agg.*`）。
- `--occlusion_mode`：指定按哪种遮挡模式计算斜率。
- `--out_csv` / `--out_json` / `--out_md`：输出排名文件路径。
- `--no_csv`：不输出 CSV。

## 6) 结果文件位置

- `run_occlusion_suite.py` 输出：
  - `outputs/keypart_experiments/summary_keypart_metrics.json`
  - `outputs/keypart_experiments/summary_keypart_metrics_agg.json`
  - 以及 `runs/<scenario>/seed_<seed>/` 下的明细报告
- `analyze_robustness_slope.py` 输出：
  - `outputs/keypart_experiments/robustness_slope_ranking.json`
  - `outputs/keypart_experiments/robustness_slope_ranking.md`
- `visualize_robustness_ranking.py` 输出：
  - `outputs/keypart_experiments/robustness_ranking_visualization.png`
  - `outputs/keypart_experiments/robustness_composite_ranking.png`
