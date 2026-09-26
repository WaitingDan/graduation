# 舰船分类项目详细使用指南

这份指南面向第一次接触项目的读者，目标是让你能按步骤跑通：数据划分、模型训练、测试评估、鲁棒性分析、热力图可视化，以及最终导出代码。

## 1. 先看最短流程

如果你只想先跑通项目，按下面顺序做就够了。

1. 准备环境。
2. 划分数据集到 `train/`、`val/`、`test/`。
3. 训练模型。
4. 做测试评估。
5. 生成热力图和鲁棒性图。
6. 导出代码文本文件。

## 2. 环境准备

进入项目根目录后执行：

```bash
cd /mnt/e/aircas/dht/graduation
conda activate one
python -m pip install torch torchvision timm matplotlib scikit-learn opencv-python pillow tqdm pandas pytorch_grad_cam
```

如果网络较慢，可以加镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com
```

## 3. 数据集怎么准备

原始数据在 `dataset/ship_fine/FGSCR/` 下。先把它切分成训练集、验证集、测试集。

```bash
python utils/split_dataset.py --dataset_subdir dataset/ship_fine --source FGSCR
```

执行后，你应该能看到：

- `dataset/ship_fine/train`
- `dataset/ship_fine/val`
- `dataset/ship_fine/test`

如果你已经手动分好了目录，可以跳过这一步。

## 4. 训练模型

项目里有 4 个训练脚本：

- `train/train_resnet.py`
- `train/train_vgg.py`
- `train/train_vit.py`
- `train/train_vit_fusion.py`

### 4.1 先做一个小测试

建议先跑 2 个 epoch，确认环境没问题：

```bash
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --epochs 2 --batch_size 8
```

### 4.2 正式训练示例

```bash
python train/train_resnet.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 32 --seed 42
python train/train_vgg.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 32 --seed 42
python train/train_vit.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 32 --seed 42
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 32 --seed 42 --attn_rollout_layers 4 --attn_temperature 1.0 --part_gate_init 0.0 --part_dropout_p 0.1 --fusion_modules abc --weight_name ablation/vit_fusion_abc_best.pth
```

说明：

- `vit_fusion` 是主模型。
- `--fusion_modules abc` 表示 A、B、C 三个模块都启用。
- 如果你只做消融，可用 `a`、`ab`、`abc` 不同组合。

训练完成后，常见输出有：

- `weights/*.pth`
- `outputs/*_loss_curve.png`
- `outputs/*_accuracy_curve.png`
- `class_indices.json`

## 5. 测试评估怎么跑

单模型评估示例：

```bash
python utils/evaluate_models.py --models vit_fusion --dataset_subdir dataset/ship_fine --test_split test --seed 42 --vit_fusion_weight weights/ablation/vit_fusion_abc_best.pth --output_subdir outputs/evaluation/fusion_eval
```

也可以评估其他模型：

```bash
python utils/evaluate_models.py --models vit --dataset_subdir dataset/ship_fine --test_split test --seed 42 --output_subdir outputs/evaluation/vit_eval
python utils/evaluate_models.py --models resnet --dataset_subdir dataset/ship_fine --test_split test --seed 42 --output_subdir outputs/evaluation/resnet_eval
python utils/evaluate_models.py --models vgg --dataset_subdir dataset/ship_fine --test_split test --seed 42 --output_subdir outputs/evaluation/vgg_eval
```

这一步会输出：

- 预测 CSV
- 分类报告
- 混淆矩阵
- 每类指标

## 6. 为什么 fusion 的热力图会比 vit 差

如果你以前看到 `vit_fusion` 热力图明显不如 `vit`，常见原因有两个：

1. 可视化预处理不一致。
2. 权重文件和 `.meta.json` 没有正确加载。

我已经修正了相关脚本：

- `utils/vit_attention_rollout.py`
- `utils/vit_fusion_rollout.py`

现在两者都统一使用 ImageNet 的标准归一化，并且会检查图片是否成功读取。这样更适合做对比。

### 6.1 单图热力图

```bash
python utils/vit_attention_rollout.py --image path/to/img.jpg --weights weights/vit_best.pth --output outputs/visualizations/attention/vit_rollout.png
python utils/vit_fusion_rollout.py --image path/to/img.jpg --weights weights/ablation/vit_fusion_abc_best.pth --output outputs/visualizations/attention/vit_fusion_rollout.png
```

### 6.2 批量可视化

先生成预测结果，再画热力图：

```bash
python utils/analysis.py visuals --model vit --csv outputs/evaluation/vit_eval/preds_vit.csv --n 3 --out_dir outputs/visualizations/attention/vit_eval
python utils/analysis.py visuals --model vit_fusion --csv outputs/evaluation/fusion_eval/preds_vit_fusion.csv --weights weights/ablation/vit_fusion_abc_best.pth --n 3 --out_dir outputs/visualizations/attention/fusion_eval
```

## 7. 鲁棒性实验与消融实验

这一节给出两组可直接运行的命令：

1. 鲁棒性主实验：`vit` vs `vit_fusion`（结果展示名可写为 AG-ViT）。
2. 消融实验：`baseline / a / ab / abc` 四个变体对比。

注意：

- CLI 参数里模型名仍使用 `vit_fusion`（兼容现有代码）。
- 结果图中的 fusion 展示名已可改为 `AG-ViT`（你前面做过代码修改）。

### 7.1 鲁棒性主实验（pairwise）

```bash
python utils/compare_module_effectiveness.py \
  --baseline_model vit \
  --target_model vit_fusion \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seeds 42 123 3407 \
  --occlusion_mode mixed \
  --levels clean light medium heavy \
  --occlusion_p 1.0 \
  --output_subdir outputs/evaluation/module_effectiveness \
  --experiment_name vit_vs_fusion_main \
  --table_metrics accuracy macro_recall macro_f1 balanced_accuracy top3_accuracy map_macro_ovr map_micro_ovr
```

主要输出目录：

- `outputs/evaluation/module_effectiveness/vit_vs_fusion_main/`
- 关键文件：`raw_runs.csv`、`target_vs_baseline_deltas.csv`、`module_effectiveness_table.md`、`module_effectiveness_report.md`

### 7.2 模块消融实验（variant mode）

```bash
python utils/compare_module_effectiveness.py \
  --variant_models vit vit_fusion vit_fusion vit_fusion \
  --variant_labels baseline a ab abc \
  --variant_weights \
    weights/vit_best.pth \
    weights/ablation/vit_fusion_a_best.pth \
    weights/ablation/vit_fusion_ab_best.pth \
    weights/ablation/vit_fusion_abc_best.pth \
  --variant_fusion_modules none a ab abc \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seeds 42 123 3407 \
  --occlusion_mode mixed \
  --levels clean light medium heavy \
  --occlusion_p 1.0 \
  --plot_metrics accuracy macro_f1 balanced_accuracy \
  --output_subdir outputs/evaluation/module_effectiveness \
  --experiment_name vit_ablation_chain
```

主要输出目录：

- `outputs/evaluation/module_effectiveness/vit_ablation_chain/`
- 关键文件：`ablation_raw_runs.csv`、`ablation_summary_mean_std_ci95.csv`、`ablation_vs_baseline_deltas.csv`
- 关键图：`ablation_variant_radar_clean.png`、`ablation_variant_radar_mixed_light.png`、`ablation_variant_radar_mixed_medium.png`、`ablation_variant_radar_mixed_heavy.png`

### 7.3 一次性连续运行（先鲁棒性，再消融）

```bash
python utils/compare_module_effectiveness.py \
  --baseline_model vit \
  --target_model vit_fusion \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seeds 42 123 3407 \
  --occlusion_mode mixed \
  --levels clean light medium heavy \
  --occlusion_p 1.0 \
  --output_subdir outputs/evaluation/module_effectiveness \
  --experiment_name vit_vs_fusion_main

python utils/compare_module_effectiveness.py \
  --variant_models vit vit_fusion vit_fusion vit_fusion \
  --variant_labels baseline a ab abc \
  --variant_weights \
    weights/vit_best.pth \
    weights/ablation/vit_fusion_a_best.pth \
    weights/ablation/vit_fusion_ab_best.pth \
    weights/ablation/vit_fusion_abc_best.pth \
  --variant_fusion_modules none a ab abc \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seeds 42 123 3407 \
  --occlusion_mode mixed \
  --levels clean light medium heavy \
  --occlusion_p 1.0 \
  --plot_metrics accuracy macro_f1 balanced_accuracy \
  --output_subdir outputs/evaluation/module_effectiveness \
  --experiment_name vit_ablation_chain
```

### 7.4 四模型鲁棒性联合实验（推荐：occlusion_suite）

四模型鲁棒性主流程建议使用 `utils/robustness/occlusion_suite.py`，它会按场景（clean / mixed_light / mixed_medium / mixed_heavy）和多 seed 自动汇总。

```bash
python utils/robustness/occlusion_suite.py \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --models resnet vgg vit vit_fusion \
  --seeds 42 123 3407 \
  --include_clean \
  --occlusion_mode mixed \
  --occlusion_levels light medium heavy \
  --occlusion_p 1.0 \
  --vit_fusion_weight weights/ablation/vit_fusion_abc_best.pth \
  --experiment_name robustness_4models

python utils/robustness/slope_ranking.py \
  --experiment_name robustness_4models \
  --occlusion_mode mixed

python utils/robustness/significance_test.py \
  --experiment_name robustness_4models

python utils/visualization/robustness_plots.py \
  --experiment_name robustness_4models
```

主要输出目录：

- `outputs/robustness/robustness_4models/metrics/`
- `outputs/robustness/robustness_4models/ranking/`
- `outputs/robustness/robustness_4models/plots/`

关键文件：

- `summary_keypart_metrics.csv`
- `summary_keypart_metrics_agg.csv`
- `robustness_ranking.csv`（由 `slope_ranking.py` 生成）

说明：

- CLI 中模型名仍是 `vit_fusion`；展示名可以在图表层使用 `AG-ViT`。
- `--seeds` 控制评估随机性（遮挡位置/形状），不会自动切换到 seed 对应权重；如需多 seed 权重对比，请显式提供对应权重路径策略。

### 7.5 论文摘要图脚本：`paper_summary_plots.py`

说明：该脚本是“零入侵”工具，只读取 `occlusion_suite` / `slope_ranking` 的输出并生成论文/报告友好的汇总图（曲线、热图、混淆矩阵拼图等）。它不会重新评估或训练模型。

示例使用（在项目根目录执行）：

```bash
python utils/visualization/paper_summary_plots.py --experiment_name robustness_4models --occlusion_mode mixed --models resnet vgg vit vit_fusion
```

可选参数摘要：
- `--output_subdir`：自定义输出根目录（默认使用 `outputs/robustness/<experiment_name>`）
- `--experiment_name`：鲁棒性实验名（默认 `keypart_experiments`）
- `--occlusion_mode`：`block|stripe|mixed`（用于匹配生成的 runs 目录）
- `--models`：顺序列表（用于绘图和热图列顺序）
- `--seed_for_confmat`：用于选取 light/medium/heavy 混淆矩阵拼图的共同 seed
- `--font_family`：可选中文字体家族，用于保证中文标签的展示

前置条件（必须满足）：

- 已运行并生成鲁棒性输出（通常由 `utils/robustness/occlusion_suite.py` 产生），并且按 experiment 存放在 `outputs/robustness/<experiment_name>/` 下；脚本会读取下列文件：
  - `metrics/summary_keypart_metrics.csv`（汇总指标，脚本主要以此为输入）
  - `runs/<scenario>/seed_*/per_class_<model>.csv`（用于构建 per-class recall drop 热图）
  - `runs/<scenario>/seed_*/confmat_<model>.png`（用于生成 light/medium/heavy 的 2×2 混淆矩阵拼图）
  - `ranking/robustness_slope_ranking.csv` 或 `ranking/robustness_slope_ranking.json`（用于选取 top 模型）

- 需要 Python 依赖（建议与项目中环境一致）：
  - `matplotlib`, `numpy`, `pandas`, `Pillow` (PIL), `opencv-python`
  - 如果需要中文显示，建议系统安装 Noto/SimHei/Source Han 等 CJK 字体，或通过 `--font_family` 指定可用字体

- 运行顺序建议：
  1. `utils/robustness/occlusion_suite.py`（生成 runs/metrics）
  2. `utils/robustness/slope_ranking.py`（生成 ranking）
  3. `python utils/visualization/paper_summary_plots.py ...`（生成论文图）

注意：脚本内部会把 `vit_fusion` 的展示名映射为 `AG-ViT`（不改变权重或文件名），因此图中会显示 `AG-ViT`，但其它命令行工具仍然使用 `vit_fusion` 作为模型标识以保持向后兼容。

## 8. 输出文件放在哪

常见目录如下：

- `outputs/evaluation/...`：评估结果
- `outputs/visualizations/...`：热力图
- `outputs/robustness/...`：鲁棒性排序和图表
- `weights/...`：模型权重

如果你要找模块对比结果，通常看：

- `raw_runs.csv`
- `target_vs_baseline_deltas.csv`
- `module_effectiveness_table.md`
- `module_effectiveness_report.md`

### 8.1 模型效率报告脚本：`report_efficiency.py`

说明：这个脚本只做模型效率统计，不做训练和测试。它会基于模型结构报告参数量、FLOPs、推理延迟和吞吐量，适合在写论文“模型复杂度/推理效率”小节时直接引用。

示例使用（在项目根目录执行）：

```bash
python utils/report_efficiency.py --models resnet vgg vit vit_fusion
```

如果你想显式指定输入尺寸、批大小或设备，可以这样写：

```bash
python utils/report_efficiency.py \
  --models resnet vgg vit vit_fusion \
  --img_size 224 \
  --batch_size 32 \
  --warmup 20 \
  --iters 100 \
  --device auto \
  --pretrained
```

可选参数摘要：

- `--models`：要统计的模型，支持 `resnet`、`vgg`、`vit`、`vit_fusion`
- `--num_classes`：类别数，默认会从 `class_indices.json` 推断
- `--img_size`：输入图像尺寸，默认 `224`
- `--batch_size`：推理批大小，默认 `1`
- `--warmup`：延迟统计前的预热次数
- `--iters`：正式计时次数
- `--device`：`auto|cpu|cuda`
- `--half`：在 CUDA 上使用 FP16 统计延迟
- `--pretrained`：使用预训练骨干初始化模型
- `--fusion_topk`、`--fusion_use_part_self_attention`、`--fusion_part_gate_init`、`--fusion_part_dropout_p`：`vit_fusion` 的结构参数
- `--output_csv`：导出 CSV 路径，默认 `outputs/evaluation/efficiency/efficiency_report.csv`
- `--output_md`：导出 Markdown 报告路径，默认 `outputs/evaluation/efficiency/efficiency_report.md`

脚本输出的核心指标包括：

- 参数量 `params_total_m`、可训练参数量 `params_trainable_m`
- `FLOPs` 及其估算后端 `flops_method`
- 延迟均值、标准差、中位数、`P90`
- 吞吐量 `throughput_fps`

说明：

- 如果 `thop`、`fvcore` 都不可用，脚本会退回到 `torch.profiler`；三者都失败时，`FLOPs` 会显示为 `N/A`。
- `vit_fusion` 在报告里仍使用命令行模型名，但你可以在论文表格中把它写成 `AG-ViT`。
- 这个脚本默认统计的是模型结构效率，不会读取训练好的分类权重；如果你只是想比较网络架构本身，这种方式最合适。

主要输出目录：

- `outputs/evaluation/efficiency/efficiency_report.csv`
- `outputs/evaluation/efficiency/efficiency_report.md`

Markdown 报告适合直接复制到论文附录或实验记录里，CSV 适合后续再画表或做对比分析。

## 9. 第三章关键技术分析实验

这一组命令对应你论文第三章的可视化实验模块。默认会使用 `weights/vit_best.pth`，单图实验默认使用样本 [dataset/ship_fine/FGSCR/001.Nimitz-class_aircraft_carrier/P0002.bmp](/mnt/e/aircas/dht/graduation/dataset/ship_fine/FGSCR/001.Nimitz-class_aircraft_carrier/P0002.bmp)，因此通常不需要再手动传图片和权重。

### 9.1 实验1：Attention Rollout 可视化

```bash
python analysis/chapter3/experiment1_rollout.py
python analysis/chapter3/experiment1_rollout.py --compare_layers 4 8 12 --output_dir outputs/paper/chapter3/rollout_compare_4_8_12
python analysis/chapter3/experiment1_rollout.py --dual_output_mode off
python analysis/chapter3/batch_rollout.py --count 8
```

说明：

- `experiment1_rollout.py` 默认 `rollout_layers=8`。
- 默认开启双输出模式：会同时保存原始图（`heatmap.png`、`overlay.png`）和论文增强图（`heatmap_paper.png`、`overlay_paper.png`）。
- 使用 `--compare_layers 4 8 12` 可在同一目录下生成分层对比结果（`layer_4`、`layer_8`、`layer_12`）。

### 9.2 实验2：多层 Attention 演化分析

```bash
python analysis/chapter3/experiment2_layer_evolution.py
```

### 9.3 实验3：Soft Mask 生成实验

```bash
python analysis/chapter3/experiment3_soft_mask.py
```

### 9.4 一键运行第三章全部实验

```bash
python analysis/chapter3/run_all.py
```

说明：

- 以上脚本只包含对 vit 基线有意义的实验 1、2、3。
- 若你想改成别的测试图片，可以直接传 `--image`。
- 若你想覆盖默认权重，也可以显式传 `--weights`。

## 10. 导出代码文本文件

为了上交代码，我已经生成了一个完整文本文件：

- [all_code_models_train_utils.txt](all_code_models_train_utils.txt)

这个文件已经按 `models`、`train`、`utils` 下所有 `.py` 文件逐个拼接完成，没有占位符，也没有重复段落。

## 11. 你可以直接照抄的命令

如果你不想看说明，只想直接跑，推荐这三条：

```bash
python utils/split_dataset.py --dataset_subdir dataset/ship_fine --source FGSCR
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 32 --seed 42 --attn_rollout_layers 4 --attn_temperature 1.0 --part_gate_init 0.0 --part_dropout_p 0.1 --fusion_modules abc --weight_name ablation/vit_fusion_abc_best.pth
python utils/evaluate_models.py --models vit_fusion --dataset_subdir dataset/ship_fine --test_split test --seed 42 --vit_fusion_weight weights/ablation/vit_fusion_abc_best.pth --output_subdir outputs/evaluation/fusion_eval
```

如果你需要，我下一步可以把这份指南再压缩成“答辩版 1 页提纲”，或者改成更像论文附录的格式。