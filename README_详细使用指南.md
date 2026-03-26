# 舰船分类项目详细使用指南（完整版）

本版目标：
- 命令只写非默认参数（默认参数不重复）
- 每个关键脚本给出用途、流程、参数解释
- 补全 `run_occlusion_suite.py` 和 `analyze_robustness_slope.py` 的完整用法

---

## 0. 环境与目录

```bash
cd /mnt/e/aircas/dht/graduation
python -m pip install torch torchvision scikit-learn matplotlib opencv-python pillow tqdm pandas openpyxl
```

---

## 1. 训练流程

### 1.1 最小训练（全部默认参数）

```bash
python train/train_resnet.py
python train/train_vgg.py
python train/train_vit.py
python train/train_vit_fusion.py
```

### 1.2 公平对比训练（推荐）

```bash
python train/train_resnet.py --label_smoothing 0.1 --train_occlusion_mode mixed --train_occlusion_level medium --train_occlusion_p 0.4
python train/train_vgg.py --label_smoothing 0.1 --train_occlusion_mode mixed --train_occlusion_level medium --train_occlusion_p 0.4
python train/train_vit.py --label_smoothing 0.1 --train_occlusion_mode mixed --train_occlusion_level medium --train_occlusion_p 0.4
python train/train_vit_fusion.py --label_smoothing 0.1 --loss_w_global 0.2 --loss_w_local 0.2 --loss_w_fusion 0.6 --dropout 0.3 --train_occlusion_mode mixed --train_occlusion_level medium --train_occlusion_p 0.4
```

> `vit_fusion` 当前默认只使用基础融合逻辑，不包含额外裁剪抖动/注意力平滑参数。

### 1.3 训练参数含义（四脚本通用）

- `--lr`：学习率，越大收敛越快但不稳定风险更高。
- `--weight_decay`：L2 正则强度，越大越抗过拟合但可能欠拟合。
- `--label_smoothing`：标签平滑，降低过度自信，提升泛化。
- `--accum_steps`：梯度累积步数，用小显存模拟大 batch。
- `--no_amp`：关闭混合精度训练。
- `--no_pretrained`：关闭预训练权重。
- `--deterministic`：更强可复现（可能变慢）。
- `--train_occlusion_mode`：训练遮挡类型（`none/block/stripe/mixed`）。
- `--train_occlusion_level`：训练遮挡强度（`light/medium/heavy`）。
- `--train_occlusion_p`：训练时应用遮挡的概率（0~1）。

### 1.4 `vit_fusion` 额外参数

- `--loss_w_global` / `--loss_w_local` / `--loss_w_fusion`：三路损失权重。
- `--dropout`：融合分类头 dropout。
- `--crop_size`：局部分支裁剪尺寸。
- `--topk_patches`：由注意力选取的 patch 数量。
- `--rollout_layers`：注意力 rollout 层数。
- `--freeze_encoder_layers`：冻结前 N 层编码器。

---

## 2. 评估流程

### 2.1 统一入口（推荐）

```bash
python utils/analysis.py eval --models resnet vgg vit vit_fusion --seed 42
```

### 2.2 指定遮挡评估（快速鲁棒性检查）

```bash
python utils/evaluate_models.py --models resnet vgg vit vit_fusion --eval_occlusion_mode mixed --eval_occlusion_level heavy --eval_occlusion_p 1.0 --output_subdir outputs/fair_eval_heavy --seed 42
```

### 2.3 `analysis.py` 子命令说明

- `eval`：跑评估并输出 `preds/report/per_class/confmat`。
- `visuals`：根据 `preds_*.csv` 生成可视化。
- `pipeline`：先 `eval` 再按需 `visuals`。

---

## 3. Key-part 鲁棒性完整流程（必读）

### Step 1：批量生成各场景指标

```bash
python utils/keypart_experiments/run_occlusion_suite.py --include_clean --output_subdir outputs/keypart_experiments
```

这一步会对：
- 场景：`clean` + `mixed_light/mixed_medium/mixed_heavy`
- 模型：默认 `resnet vgg vit vit_fusion`
- 种子：默认 `42 123 3407`

输出：
- `summary_keypart_metrics.json`
- `summary_keypart_metrics_agg.json`
- 各场景明细：`runs/<scenario>/seed_<seed>/...`

### Step 2：计算鲁棒性下降斜率并排序

```bash
python utils/keypart_experiments/analyze_robustness_slope.py --occlusion_mode mixed
```

输出：
- `robustness_slope_ranking.json`
- `robustness_slope_ranking.md`

### Step 3：可视化排序结果

```bash
python utils/visualize_robustness_ranking.py
```

输出：
- `robustness_ranking_visualization.png`
- `robustness_composite_ranking.png`

### Step 4：导出论文草稿（可选）

```bash
python utils/keypart_experiments/export_report.py --out_md outputs/keypart_experiments/chapter4_draft.md
```

---

## 4. Key-part 脚本参数详解

### 4.1 `run_occlusion_suite.py`

- `--include_clean`：加入 clean 场景。
- `--models`：模型列表。
- `--seeds`：随机种子列表。
- `--occlusion_mode`：遮挡模式（`block/stripe/mixed`）。
- `--occlusion_levels`：遮挡等级列表。
- `--occlusion_p`：遮挡概率。
- `--output_subdir`：输出目录。
- `--no_csv`：不输出 CSV，只输出 JSON。

### 4.2 `analyze_robustness_slope.py`

- `--agg_csv` / `--agg_json`：输入聚合指标文件。
- `--occlusion_mode`：按哪种遮挡模式计算斜率。
- `--out_csv` / `--out_json` / `--out_md`：输出路径。
- `--no_csv`：不保存 ranking CSV。

斜率解释：
- 斜率越大（越接近 0，下降越慢）→ 鲁棒性越好。

---

## 5. 其它脚本用法（补全）

### 5.1 热力图导出（单图/多图）

```bash
python utils/keypart_experiments/export_occlusion_heatmaps.py --images dataset/ship_fine/test/001.Nimitz-class_aircraft_carrier/P0031.bmp --include_clean
```

常用参数：
- `--models`：默认四模型。
- `--occlusion_mode` / `--occlusion_levels`：遮挡设置。
- `--fusion_rollout_blur` / `--fusion_rollout_gamma` / `--fusion_rollout_mix`：fusion 热力图控制。

### 5.2 ViT attention rollout

```bash
python utils/vit_attention_rollout.py --image dataset/ship_fine/test/001.Nimitz-class_aircraft_carrier/P0031.bmp
```

### 5.3 Grad-CAM

单图：

```bash
python utils/gradcam_cnn_models.py --image dataset/ship_fine/test/001.Nimitz-class_aircraft_carrier/P0031.bmp --model resnet
```

批量（从 preds CSV）：

```bash
python utils/gradcam_cnn_models.py --csv outputs/preds_resnet.csv --model resnet --wrong_only
```

### 5.4 从预测 CSV 生成样本可视化

```bash
python utils/generate_visuals.py --model resnet --csv outputs/preds_resnet.csv
```

### 5.5 数据集划分

```bash
python utils/split_dataset.py --source FGSCR
```

### 5.6 兼容入口（历史 pipeline）

```bash
python utils/run_pipeline.py --models resnet vgg vit --visuals
```

---

## 6. 常见问题

- 显存不足：减小 `--batch_size` 或增大 `--accum_steps`。
- 想完全复现：固定 `--seed` 并加 `--deterministic`。
- 想做从头训练：加 `--no_pretrained`。
- 想禁用 AMP：加 `--no_amp`。
