# 舰船分类项目详细使用指南（完整版）

本指南面向第一次接触这个项目的读者，目标是把“数据怎么划分、模型怎么训练、鲁棒性怎么评估、结果怎么解释、论文里怎么写”全部讲清楚。

如果你只想先跑通实验，可以直接按第 4 节的步骤顺序执行；如果你想理解每个脚本到底在做什么，再看后面的说明。

## 1. 流程总览

1. 划分数据集（train/val/test）
  - 这一步的作用是把原始船舶图片拆成训练集、验证集和测试集，避免模型训练时看到测试集内容。
  - 训练集用于更新模型参数，验证集用于选最优权重，测试集只在最后做正式评估。
2. 训练四个模型（resnet、vgg、vit、vit_fusion）
  - 这一步会分别训练四种网络，得到各自的权重文件。
  - 其中 `vit_fusion` 是你的主模型，当前为单分支 ViT 的轻量残差融合版。
3. 运行鲁棒性评估（clean + light/medium/heavy）
  - 这一步会在无遮挡、轻度遮挡、中度遮挡和重度遮挡四种场景下测试模型。
  - 重点不是只看 clean 精度，而是看模型在遮挡加重后是否掉得更慢。
4. 生成排名与图表（slope、AUPC、综合图）
  - 这一步会把不同模型的退化曲线转换成可比较的数值，比如斜率、AUPC、clean->heavy 下降量。
  - 你在论文里看到的“鲁棒性排序图”就是这里生成的。
5. 显著性检验（fusion vs top1，配对检验）
  - 这一步回答“fusion 的提升是不是偶然的”。
  - 如果显著性检验通过，说明 fusion 和 top1 的差异更可信。
6. 导出可解释性可视化（可选）
  - 这一步会生成 Grad-CAM、ViT rollout、fusion attention 可视化，帮助解释模型为什么这么预测。

## 2. 核心脚本

- `utils/split_dataset.py`
  - 把原始数据拆成 train/val/test。
  - 如果你的原始数据还没有划分，这个脚本是第一步。
- `train/train_resnet.py`、`train/train_vgg.py`、`train/train_vit.py`、`train/train_vit_fusion.py`
  - 分别训练四个基础模型和主模型。
  - 这些脚本默认会保存最优权重到 `weights/`，并画训练曲线到 `outputs/`。
- `utils/robustness/occlusion_suite.py`
  - 在 clean 和不同遮挡强度下批量评估模型。
  - 它不会重新训练，只会读取已有权重。
- `utils/robustness/slope_ranking.py`
  - 把不同遮挡强度下的结果转成斜率、AUPC、下降量和排名。
  - 这里得到的 CSV/JSON 是鲁棒性排序图的直接输入。
- `utils/visualization/robustness_plots.py`
  - 读取排名文件，绘制综合排名图和指标分解图。
  - 这是论文主文里最常用的鲁棒性图来源。
- `utils/visualization/paper_summary_plots.py`
  - 把已有输出整理成论文友好的图表和表格。
  - 适合最终出稿时做“零侵入整理”。
- `utils/report_efficiency.py`
  - 统计 FLOPs、参数量和推理时延。
  - 适合论文里补一个“复杂度/部署开销分析”。
- `utils/compare_module_effectiveness.py`
  - 对比 baseline 与目标模块模型在 clean/遮挡场景下的指标变化（含均值、方差与增量）。
  - 会额外生成可直接阅读的表格文件，便于论文中展示“加模块前后”的收益。
- `utils/pipelines/full_robustness_pipeline.py`
  - 一键跑训练/评估/绘图的总入口。
  - 适合批量复现实验，但不适合第一次学习流程时直接跳着用。

补充说明：
- 本指南优先覆盖“可直接运行的入口脚本”。
- 像 `utils/common.py`、`utils/metrics.py`、`models/vit_model.py` 这类基础模块属于被入口脚本调用的实现组件，通常不单独写成操作步骤。

## 3. 环境准备

建议先确认你正在使用项目指定的 Python 环境，再安装依赖。这个项目里大部分脚本都基于 PyTorch、torchvision、timm、matplotlib、sklearn、opencv 和 tqdm。

```bash
cd /mnt/e/aircas/dht/graduation
conda activate one
python -m pip install torch torchvision scikit-learn matplotlib opencv-python pillow tqdm pandas openpyxl
# 可选：若希望更稳定地统计 FLOPs，建议安装
python -m pip install thop fvcore
python -m pip install timm
```

### timm 预训练权重与镜像
- 当前基准实现已统一为 `timm`：
  - `resnet` 对应 `resnet18`
  - `vgg` 对应 `vgg13`
  - `vit` 对应 `vit_small_patch16_224`
- timm 的预训练权重默认通过 HuggingFace 下载；在网络受限环境下推荐设置镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com
```
- 若下载失败，可在训练时加 `--no_pretrained` 使用随机初始化，或先在能联网的机器下载并把权重放到本地缓存。

### 关于“公平性”（统一实现库）
- 为了严格可比，建议统一预训练来源与实现库。当前项目默认是“全 timm”配置，能减少实现差异带来的额外变量。
- 若你后续引入 `torchvision` 版本，请把它作为单独对照组，并在论文中明确说明实现来源与权重来源。

如果你打算运行可解释性脚本，还建议确认下面这些包可用：
- `pytorch_grad_cam`
- `numpy`
- `pillow`

如果这些包缺失，先安装再跑脚本，避免中途报错。

## 4. 标准执行流程

### Step 1. 划分数据集

这一步会把原始数据整理成 `train/`、`val/`、`test/` 三个目录。
如果你已经手动划分好了，就不需要重复执行。

```bash
conda activate one
python utils/split_dataset.py --dataset_subdir dataset/ship_fine --source FGSCR
```

补充说明：
- `--dataset_subdir dataset/ship_fine` 表示数据根目录。
- `--source FGSCR` 表示原始数据来源目录名。
- 运行结束后，你应该能在 `dataset/ship_fine/train`、`val`、`test` 下看到类别文件夹。

### Step 2. 训练四个模型（clean 训练）

clean 训练的意思是：训练时不人为加遮挡，模型先学会最基础的分类能力。
这一阶段得到的模型通常是后续鲁棒性实验的起点。

```bash
conda activate one
#4月27训练所用指令
python train/train_resnet.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 32 --lr 2e-4 --weight_decay 1e-4 --label_smoothing 0.1 --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42
python train/train_vgg.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 32 --lr 2e-4 --weight_decay 1e-4 --label_smoothing 0.1 --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42
python train/train_vit.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 32 --lr 2e-4 --weight_decay 1e-4 --label_smoothing 0.1 --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 32 --lr_backbone 2e-5 --lr_head 2e-4 --weight_decay 1e-4 --label_smoothing 0.1 --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42 --attn_rollout_layers 4 --attn_temperature 1.0 --part_gate_init 0.0 --part_dropout_p 0.1 --attn_sparse_weight 0.005 --fusion_modules abc --weight_name ablation/vit_fusion_abc_best.pth
```

说明：
-- 为保证可复现，建议训练脚本统一使用同一个 seed。
- 当前默认骨干：`resnet=resnet18 (timm)`、`vgg=vgg13 (timm)`、`vit=vit_small_patch16_224 (timm)`。
- 这个版本不再依赖 `token drop`、`consistency loss` 或 `input stem`；主实验建议直接按上面的推荐命令训练。
- `vit_fusion` 的推荐公平配置是 clean 训练 + 单分支关键 token 建模，默认无遮挡训练；如果你要做“更强鲁棒版”对照，再单独加遮挡增强。
- `vit_fusion` 常用参数：`--attn_rollout_layers`、`--attn_temperature`、`--part_gate_init`、`--part_dropout_p`、`--attn_sparse_weight`。`--topk_patches` 仅保留为旧权重兼容参数，当前 soft pooling 不再依赖它筛选 token。`--use_part_self_attention` 只是兼容旧命令的开关，对当前轻量版主流程不再起作用。
- `vit_fusion` 新增 `--fusion_modules` 用于控制模块开关：`a`（中层注意力重加权）、`b`（局部分支精炼）、`c`（融合门控）。例如 `--fusion_modules ab` 表示仅启用 A/B。
- 当前推荐配置下，`attn_rollout_layers=4`、`attn_temperature=1.0`、`part_gate_init=0.0`、`part_dropout_p=0.1` 和 `attn_sparse_weight=0.005` 是稳健起点。
- `vit_fusion` 当前主流程不依赖局部自注意力模块；`--use_part_self_attention` 仅作为旧命令兼容参数保留。
- 如果你只是想复现主实验，直接用这些显式命令即可；如果你要做结构对比，再在此基础上加消融参数。

训练结束后，一般会得到：
- `weights/<model>_best.pth`：验证集最优权重
- `outputs/<model>_loss_curve.png`：训练/验证损失曲线
- `outputs/<model>_accuracy_curve.png`：训练/验证准确率曲线
- `class_indices.json`：类别索引映射

### Step 3. 评估单个模型（逐个看结果）

如果你只想先确认某一个模型的测试集结果，下面这几组可以直接照抄。`utils/evaluate_models.py` 的 `--models` 参数一次只写一个模型名时，就是单模型评估。

```bash
conda activate one
python utils/evaluate_models.py \
  --models vit_fusion \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seed 42 \
  --vit_fusion_weight weights/ablation/vit_fusion_abc_best.pth \
  --output_subdir outputs/evaluation/fusion_eval

python utils/evaluate_models.py \
  --models vit \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seed 42 \
  --output_subdir outputs/evaluation/vit_eval

python utils/evaluate_models.py \
  --models resnet \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seed 42 \
  --output_subdir outputs/evaluation/resnet_eval

python utils/evaluate_models.py \
  --models vgg \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seed 42 \
  --output_subdir outputs/evaluation/vgg_eval
```

说明：
- 这一步会输出预测 CSV、分类报告、混淆矩阵和每类指标。
- `--vit_fusion_weight` 用来指定 fusion 权重，避免和默认路径冲突。
如果你的权重文件旁边有 `.meta.json`，脚本会自动按其中参数恢复模型结构；这对 `vit_fusion` 是可选的，主要用于保存训练时的超参数记录。

### Step 4. 运行鲁棒性评估与排序

这一步会在 clean 和不同遮挡强度下批量测试模型，用来比较“同一模型在不同遮挡条件下退化得有多快”。

```bash
conda activate one
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 32 --lr_backbone 2e-5 --lr_head 2e-4 --weight_decay 1e-4 --label_smoothing 0.1 --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42 --attn_rollout_layers 4 --attn_temperature 1.0 --part_gate_init 0.0 --part_dropout_p 0.1 --attn_sparse_weight 0.005 --fusion_modules abc

python utils/robustness/occlusion_suite.py \
  --models resnet vgg vit vit_fusion  \
  --vit_fusion_weight weights/ablation/vit_fusion_abc_best.pth \
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
```

说明：
- clean 场景默认开启，所以不需要额外加 `--include_clean`。
- 如果你只想看遮挡测试，可以显式加 `--no_include_clean`。
- 该阶段不会重新训练模型，只会读取 `weights/` 里的权重文件。
- `vit_fusion` 的默认评估权重现在指向 `weights/ablation/vit_fusion_abc_best.pth`，如果你想临时切换别的 fusion 权重，可以显式传 `--vit_fusion_weight`。
- 输出的聚合指标里包含 95% CI 字段，便于你在论文里说明不同 seed 下的波动范围。

鲁棒性结果一般会输出三类东西：
1. 原始测试结果：每个 seed、每个场景的 F1 和 balanced accuracy。
2. 汇总结果：均值、标准差、95% CI。
3. 排名结果：斜率、AUPC、clean->heavy 下降量和综合排序图。

### Step 5. 显著性检验与绘图

如果你已经跑完鲁棒性评估，就用这一组生成论文里最常用的分析图和统计结论。

```bash
conda activate one
python utils/robustness/significance_test.py \
  --experiment_name keypart_experiments \
  --fusion_model vit_fusion

python utils/visualization/robustness_plots.py \
  --experiment_name keypart_experiments
```

说明：
- 显著性检验脚本默认比较 `vit_fusion` 与排名 top1 的模型。
- 生成的图表和报告会保存到 `outputs/robustness/<experiment_name>/` 下面对应目录。

### Step 5.5 模块有效性对比（新增）

如果你需要回答“加了某个模块到底有没有用”，建议单独运行这个脚本，它会输出 baseline 与目标模型在各场景下的逐项指标差值。

```bash
conda activate one
python utils/compare_module_effectiveness.py \
  --baseline_model vit \
  --target_model vit_fusion \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seeds 42 123 3407 \
  --occlusion_mode mixed \
  --levels clean light medium heavy \
  --occlusion_p 1.0 \
  --experiment_name vit_vs_fusion_main
```

如果要自定义表格指标列，可增加 `--table_metrics`：

```bash
python utils/compare_module_effectiveness.py \
  --baseline_model vit \
  --target_model vit_fusion \
  --seeds 42 123 3407 \
  --table_metrics accuracy macro_recall macro_f1 balanced_accuracy top3_accuracy map_macro_ovr map_micro_ovr \
  --experiment_name vit_vs_fusion_main
```

说明：
- 默认输出目录为 `outputs/evaluation/module_effectiveness/<experiment_name>/`。
- 核心输出包括：
  - `raw_runs.csv`：每个 seed、每个场景的原始指标。
  - `target_vs_baseline_deltas.csv`：逐运行的 baseline/target/delta 对比。
  - `delta_summary_by_scenario.csv`：按场景聚合的 delta 汇总。
  - `module_effectiveness_table.csv`：面向论文表格的均值对比表。
  - `module_effectiveness_table.md`：可直接阅读和粘贴的 Markdown 表格。
  - `module_effectiveness_report.md`：自动生成的文字结论报告。

如果你要做模块消融链路（baseline、A、AB、ABC），推荐先各训练一次并固定权重命名：

```bash
conda activate one
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 32 --lr_backbone 2e-5 --lr_head 2e-4 --weight_decay 1e-4 --label_smoothing 0.1 --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42 --attn_rollout_layers 4 --attn_temperature 1.0 --part_gate_init 0.0 --part_dropout_p 0.1 --attn_sparse_weight 0.005 --fusion_modules a --weight_name ablation/vit_fusion_a_best.pth
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 32 --lr_backbone 2e-5 --lr_head 2e-4 --weight_decay 1e-4 --label_smoothing 0.1 --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42 --attn_rollout_layers 4 --attn_temperature 1.0 --part_gate_init 0.0 --part_dropout_p 0.1 --attn_sparse_weight 0.005 --fusion_modules ab --weight_name ablation/vit_fusion_ab_best.pth
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 32 --lr_backbone 2e-5 --lr_head 2e-4 --weight_decay 1e-4 --label_smoothing 0.1 --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42 --attn_rollout_layers 4 --attn_temperature 1.0 --part_gate_init 0.0 --part_dropout_p 0.1 --attn_sparse_weight 0.005 --fusion_modules abc --weight_name ablation/vit_fusion_abc_best.pth
```

然后用同一批权重文件，在三个 seed 下做评估随机性消融（不要求为每个 seed 重新训练）：

```bash
conda activate one
python utils/compare_module_effectiveness.py \
  --variant_models vit vit_fusion vit_fusion vit_fusion \
  --variant_labels baseline a ab abc \
  --variant_weights weights/vit_best.pth weights/ablation/vit_fusion_a_best.pth weights/ablation/vit_fusion_ab_best.pth weights/ablation/vit_fusion_abc_best.pth \
  --variant_fusion_modules none a ab abc \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seeds 42 123 3407 \
  --occlusion_mode mixed \
  --levels clean light medium heavy \
  --occlusion_p 1.0 \
  --plot_metrics accuracy macro_f1 balanced_accuracy \
  --experiment_name vit_ablation_chain
```

说明补充：
- 这里的 `--seeds 42 123 3407` 控制的是评估阶段随机性（遮挡位置/形状等），不自动切换为不同 seed 训练得到的权重。
- 因此一个固定权重文件可以用于三个 seed 的消融评估。
- 如果你要做“训练随机性 + 评估随机性”联合分析，再额外准备按 seed 区分命名的多组权重。

多变体模式新增输出：
- `ablation_raw_runs.csv`：每个 seed、每个场景、每个变体的原始指标。
- `ablation_summary_mean_std_ci95.csv`：按变体和场景聚合后的均值、标准差和 95% CI。
- `ablation_vs_baseline_deltas.csv`：每个变体相对 baseline 的增量对比。
- `ablation_variant_means.png`：多指标柱状对比图（按场景分组，按变体并列）。

### Step 6. 效率评估与可解释性

这一节用于补充“模型好不好用”和“为什么这样预测”，不影响主实验指标。

```bash
conda activate one
python utils/report_efficiency.py \
  --models resnet vgg vit vit_fusion \
  --img_size 224 \
  --batch_size 1 \
  --warmup 20 \
  --iters 100 \
  --device auto

python utils/evaluate_models.py \
  --models vit_fusion \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seed 42 \
  --vit_fusion_weight weights/ablation/vit_fusion_abc_best.pth \
  --output_subdir outputs/evaluation/fusion_eval

python utils/analysis.py visuals \
  --model vit_fusion \
  --csv outputs/evaluation/fusion_eval/preds_vit_fusion.csv \
  --n 3 \
  --out_dir outputs/visualizations/attention/default_eval
```

单图 rollout：

```bash
python utils/vit_fusion_rollout.py \
  --image path/to/img.jpg \
  --weights weights/ablation/vit_fusion_abc_best.pth \
  --rollout_layers 3 \
  --output outputs/visualizations/attention/vit_fusion_rollout.png
```

说明：
- `report_efficiency.py` 会输出 CSV 和 Markdown 报告到 `outputs/evaluation/efficiency/`。
- 若在 GPU 上测时延，建议固定环境并多次运行取平均；如需测试半精度可加 `--half`。
- 对 `vit_fusion`，当前结构为单分支关键 token 建模，FLOPs 与时延统计直接按完整前向路径执行。
- `analysis.py visuals` 会根据预测结果挑选样本并生成可视化图。
- `vit_fusion_rollout.py` 更适合答辩时展示单张图的 attention rollout；默认读取权重里保存的 `attn_rollout_layers`，也可以用 `--rollout_layers` 临时覆盖。

## 7. 一键流程（可选）

如果你已经熟悉前面步骤，可以直接用一键流程把训练、评估、排序和绘图串起来。
它适合做完整复现，但不适合初学者第一次逐步理解每一步。

```bash
conda activate one
python utils/pipelines/full_robustness_pipeline.py --experiment_name keypart_experiments
```

说明：
- 一键流程默认包含 clean。
- 如果你只想看遮挡场景，可以加 `--no_include_clean`。
- 一键流程默认会执行：鲁棒性评估、slope ranking、显著性检验、鲁棒性绘图。
- 如果你已经有现成结果，也可以跳过训练阶段，只保留评估与绘图。

关于一键流程 (`utils/pipelines/full_robustness_pipeline.py`)：

- 该脚本是流水线入口，用于将训练、评估与可视化串联成一次可重复运行的实验。
- 它的逻辑是：先训练你指定的模型，再在多个遮挡场景下评估，最后把结果整理成排名和图表。
- 适合“我要一口气把整套实验跑完”的场景。

典型流程：
1.（可选）按 `--train_models` 训练指定模型。
2. 调用 `utils/robustness/occlusion_suite.py` 做遮挡评估。
3. 调用 `utils/robustness/slope_ranking.py` 生成排名。
4. 调用 `utils/visualization/robustness_plots.py` 画综合图。

主要常用参数：
- `--train_models`：列出要训练的模型，比如 `resnet vgg vit vit_fusion`。
- `--eval_models`：指定用于评估的模型列表，默认四模型都跑。
- `--seeds`：控制随机种子数量。
- `--occlusion_levels`、`--occlusion_mode`、`--occlusion_p`：控制遮挡类型和强度。
- `--quick`：快速模式，适合先确认流程是否能跑通。

输出：
- 脚本会把结果写入 `outputs/robustness/<experiment_name>/` 下的 `runs/`、`metrics/`、`ranking/`、`plots/`、`reports/` 等目录。
- 运行结束后会打印实验根路径，方便你继续查看输出。

适用场景：
- 需要一次性跑完训练→评估→绘图的复现实验。
- 如果你只想做单步操作，比如只评估或只绘图，也可以直接调用对应脚本。

## 8. 输出目录说明

这个目录最容易让初学者迷路，所以单独解释一下。
你可以把它理解成“实验结果仓库”：训练、评估、排名、图表都会在这里按固定规则保存。

- 鲁棒性主目录：outputs/robustness/keypart_experiments
- 关键子目录：
  - runs：每个场景与 seed 的原始评估输出
  - metrics：汇总指标（summary 与 agg）
  - ranking：斜率/AUPC/排名表
  - reports：显著性检验报告
  - plots：综合排序图与指标分解图

常用结果文件：
- metrics/summary_keypart_metrics.csv
- metrics/summary_keypart_metrics_agg.csv
- ranking/robustness_slope_ranking.csv
- reports/significance_fusion_vs_top1.md
- reports/significance_fusion_vs_top1.json
- plots/robustness_composite_ranking.png
- plots/robustness_ranking_visualization.png
- efficiency/efficiency_report.csv
- efficiency/efficiency_report.md

怎么读这些文件：
- `summary_keypart_metrics.csv`：每一次真实评估的逐条结果，信息最细。
- `summary_keypart_metrics_agg.csv`：按场景和模型聚合后的结果，论文里更常用。
- `robustness_slope_ranking.csv`：根据退化趋势计算出来的排名表。
- `robustness_composite_ranking.png`：综合排序图，适合放主文。
- `robustness_ranking_visualization.png`：四个子图拆开看，便于解释哪个指标拖后腿。
- `efficiency_report.csv/md`：效率统计结果，适合做复杂度分析表。

## 9. Fusion 快速消融（单分支版本）

如果你只想优化 `vit_fusion`，建议围绕“关键 token 建模强度”做小步迭代，而不是再引入双分支结构。

```bash
conda activate one

# 方案 A：默认（推荐起点）
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --epochs 50 --batch_size 32 --lr_backbone 2e-5 --lr_head 2e-4 --weight_decay 1e-4 --label_smoothing 0.1 --train_occlusion_mode mixed --train_occlusion_level medium --train_occlusion_p 0.4 --seed 42 --attn_rollout_layers 4 --attn_temperature 1.0 --part_gate_init 0.0 --part_dropout_p 0.1 --attn_sparse_weight 0.005 --weight_name vit_fusion_default_best.pth

# 方案 B：更稳的 attention 聚合（增加 rollout 层数）
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --epochs 50 --batch_size 32 --lr_backbone 2e-5 --lr_head 2e-4 --weight_decay 1e-4 --label_smoothing 0.1 --train_occlusion_mode mixed --train_occlusion_level medium --train_occlusion_p 0.4 --seed 42 --attn_rollout_layers 6 --attn_temperature 1.0 --part_gate_init 0.0 --part_dropout_p 0.1 --attn_sparse_weight 0.005 --weight_name vit_fusion_rollout6_best.pth

# 方案 C：启用关键 token 内部自注意力
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --epochs 50 --batch_size 32 --lr_backbone 2e-5 --lr_head 2e-4 --weight_decay 1e-4 --label_smoothing 0.1 --train_occlusion_mode mixed --train_occlusion_level medium --train_occlusion_p 0.4 --seed 42 --use_part_self_attention --attn_rollout_layers 4 --attn_temperature 1.0 --part_gate_init 0.0 --part_dropout_p 0.1 --attn_sparse_weight 0.005 --weight_name vit_fusion_selfattn_best.pth

# 方案 D：更强鲁棒正则（提高 dropout 和稀疏约束）
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --epochs 50 --batch_size 32 --lr_backbone 2e-5 --lr_head 2e-4 --weight_decay 1e-4 --label_smoothing 0.1 --train_occlusion_mode mixed --train_occlusion_level heavy --train_occlusion_p 0.5 --seed 42 --attn_rollout_layers 4 --attn_temperature 1.0 --part_gate_init 0.0 --part_dropout_p 0.2 --attn_sparse_weight 0.01 --weight_name vit_fusion_robust_best.pth
```

建议比较顺序：
1. 先看 `mixed_medium` 与 `mixed_heavy` 的 `macro_f1_mean`、`balanced_accuracy_mean`。
2. 再看 ranking 中的 slope 和 AUPC。
3. 最后看综合分图，判断总体稳定性。

如果某个版本 clean 很高但 heavy 跌幅大，通常说明它对少数关键 token 依赖过强，需要提高 `part_dropout_p`、适度增大 `attn_rollout_layers`，或降低 `attn_temperature` 来让 soft pooling 更集中。

## 10. 论文图与写作建议（详细）

建议主文放入的图（并给出生成指令）：

1. 数据分布图（类别样本数柱状图）
```bash
python scripts/plot_class_counts.py
```

2. 遮挡示意图（原图/条带/块状/混合）
```bash
python scripts/visualize_occlusions.py
```

3. 鲁棒性趋势曲线（F1 / 平衡准确率 均值±标准差）
```bash
python utils/visualization/paper_summary_plots.py --experiment_name keypart_experiments
```
输出：paper_fig_macro_balanced_curves.png

4. 综合鲁棒性排序 + 指标拆解图（主结论图）
```bash
python utils/visualization/robustness_plots.py --experiment_name keypart_experiments
```
输出：robustness_ranking_visualization.png、robustness_composite_ranking.png

5. 各类别 recall 下降热图（clean -> heavy）
```bash
python utils/visualization/paper_summary_plots.py --experiment_name keypart_experiments
```
输出：paper_fig_per_class_recall_drop_heatmap.png

6. 混淆矩阵 2×2 拼图（seed=42，light/medium/heavy 三张）
```bash
python utils/visualization/paper_summary_plots.py --experiment_name keypart_experiments --seed_for_confmat 42
```
输出：
- paper_fig_confmat_grid_mixed_light_seed42.png
- paper_fig_confmat_grid_mixed_medium_seed42.png
- paper_fig_confmat_grid_mixed_heavy_seed42.png

7. 可解释性可视化（批量热力图 + 单图 rollout）

7.1 批量热力图（主入口，推荐优先使用）

```bash
# 先分别导出各模型的预测 CSV（如果你还没有对应结果）
python utils/evaluate_models.py \
  --models resnet \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seed 42 \
  --output_subdir outputs/evaluation/resnet_eval
python utils/evaluate_models.py \
  --models vgg \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seed 42 \
  --output_subdir outputs/evaluation/vgg_eval
python utils/evaluate_models.py \
  --models vit \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seed 42 \
  --output_subdir outputs/evaluation/vit_eval
python utils/evaluate_models.py \
  --models vit_fusion \
  --dataset_subdir dataset/ship_fine \
  --test_split test \
  --seed 42 \
  --vit_fusion_weight weights/ablation/vit_fusion_abc_best.pth \
  --output_subdir outputs/evaluation/fusion_eval

# ResNet / VGG Grad-CAM（批量）
python utils/gradcam_cnn_models.py --image path/to/img.jpg --model resnet --weights weights/resnet_best.pth --out_dir outputs/visualizations/attention
python utils/gradcam_cnn_models.py --image path/to/img.jpg --model vgg --weights weights/vgg_best.pth --out_dir outputs/visualizations/attention

# ViT 批量可视化
python utils/analysis.py visuals \
  --model vit \
  --csv outputs/evaluation/vit_eval/preds_vit.csv \
  --n 3 \
  --out_dir outputs/visualizations/attention/vit_eval

# ViT-Fusion 批量可视化
python utils/analysis.py visuals \
  --model vit_fusion \
  --csv outputs/evaluation/fusion_eval/preds_vit_fusion.csv \
  --weights weights/ablation/vit_fusion_abc_best.pth \
  --n 3 \
  --out_dir outputs/visualizations/attention/default_eval

# 如果你想把四个模型都各自出一组图，建议分别执行上面的四段命令
```

7.2 单图 rollout（补充入口，适合调试与答辩展示）

```bash
# ViT rollout
python utils/vit_attention_rollout.py --image dataset/ship_fine/test/018.Whitby_Island-class_dock_landing_ship/P5565.bmp --weights weights/vit_best.pth --output outputs/visualizations/attention/vit_rollout.png

# ViT-Fusion rollout（注意权重使用 fusion 主实验权重）
python utils/vit_fusion_rollout.py --image dataset/ship_fine/test/018.Whitby_Island-class_dock_landing_ship/P5565.bmp --weights weights/ablation/vit_fusion_abc_best.pth --output outputs/visualizations/attention/vit_fusion_rollout.png
```

说明：
- `analysis.py visuals` 是常规批量热力图入口，适合从预测 CSV 中自动挑选多个样本做对比。
- `ResNet / VGG` 用 `gradcam_cnn_models.py`，`ViT` 用 `analysis.py visuals --model vit`，`ViT-Fusion` 用 `analysis.py visuals --model vit_fusion --weights weights/ablation/vit_fusion_abc_best.pth`，这样四个模型会走各自正确的可视化路径。
- `vit_attention_rollout.py` 和 `vit_fusion_rollout.py` 更适合单图调试或答辩展示。

8. 效率/部署开销表（复杂度对比表）
```bash
python utils/report_efficiency.py --models resnet vgg vit vit_fusion --img_size 224 --batch_size 1 --warmup 20 --iters 100 --device auto
```
输出：outputs/evaluation/efficiency/efficiency_report.md

配色说明（所有图一致）：
- vit: 蓝色
- vit_fusion(ABC): 红色
- resnet: 橙色
- vgg: 绿色
- fusion(A)、fusion(AB): 使用红色同色调但更浅的区分色（用于消融图）

写作注意：
- 不要只报告综合分，需同时报告原始指标（macro_f1、balanced_accuracy、slope、AUPC）。
- 综合分是实验内相对归一化结果（现已在图中显示分量拆解），最差模型可能为 0，但不代表原始性能为 0。
- 建议同时引用 CI 与显著性检验结果，避免仅凭均值差异下结论。

## 11. 论文中放置“效率指标”建议

建议新增一个小节：`Efficiency Analysis`（或“复杂度与部署开销分析”）。

推荐放置位置：
1. 主文实验章节中，放在“鲁棒性结果”之后、“可视化分析”之前。
2. 表格建议命名为 `Table X: Model Efficiency Comparison`，列出 `Params(M)`、`FLOPs(G)`、`Latency(ms)`、`FPS`。
3. 正文里用 2-3 句总结“性能-鲁棒性-开销”的 trade-off（例如 fusion 在鲁棒性提升下带来的额外计算成本）。

若主文篇幅紧张：
1. 主文仅保留一个精简效率表（四模型对比）。
2. 将不同 batch size、FP32/FP16、CPU/GPU 详细数据放附录。

