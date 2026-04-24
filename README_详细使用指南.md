# 舰船分类项目详细使用指南（完整版）

本指南面向第一次接触这个项目的读者，目标是把“数据怎么划分、模型怎么训练、鲁棒性怎么评估、结果怎么解释、论文里怎么写”全部讲清楚。

如果你只想先跑通实验，可以直接按第 4 节的步骤顺序执行；如果你想理解每个脚本到底在做什么，再看后面的说明。

## 1. 流程总览

1. 划分数据集（train/val/test）
  - 这一步的作用是把原始船舶图片拆成训练集、验证集和测试集，避免模型训练时看到测试集内容。
  - 训练集用于更新模型参数，验证集用于选最优权重，测试集只在最后做正式评估。
2. 训练四个模型（resnet、vgg、vit、vit_fusion）
  - 这一步会分别训练四种网络，得到各自的权重文件。
  - 其中 `vit_fusion` 是你的主模型，它同时包含全局分支和局部分支。
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
  - 分别训练四个模型。
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
- `utils/pipelines/full_robustness_pipeline.py`
  - 一键跑训练/评估/绘图的总入口。
  - 适合批量复现实验，但不适合第一次学习流程时直接跳着用。

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
python train/train_resnet.py --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42
python train/train_vgg.py --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42
python train/train_vit.py --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42
python train/train_vit_fusion.py --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42
```

说明：
- 为保证可复现，建议四个训练脚本统一使用同一个 seed。
- 当前默认骨干：`resnet=resnet18 (timm)`、`vgg=vgg13 (timm)`、`vit=vit_small_patch16_224 (timm)`。
- `vit_fusion` 常用消融参数：`--topk_patches`、`--use_cross_attention`、`--use_local_self_attention`、`--local_gate_init`、`--share_backbone`。
- 当前默认配置下，`topk_patches=3` 是“兼顾局部信息与噪声控制”的折中值。
- 如果你只是想复现主实验，直接用默认命令即可；如果你要做结构对比，再去加消融参数。

训练结束后，一般会得到：
- `weights/<model>_best.pth`：验证集最优权重
- `outputs/<model>_loss_curve.png`：训练/验证损失曲线
- `outputs/<model>_accuracy_curve.png`：训练/验证准确率曲线
- `class_indices.json`：类别索引映射

### Step 3. 运行鲁棒性评估与绘图

这一步会读取 `weights/` 下的权重，在多个遮挡场景下做测试。
它的目标不是训练新模型，而是比较“同一模型在不同遮挡条件下退化得有多快”。

```bash
conda activate one
python utils/robustness/occlusion_suite.py \
  --models resnet vgg vit vit_fusion \
  --seeds 42\
  --occlusion_mode mixed \
  --occlusion_levels light medium heavy \
  --occlusion_p 1.0 \
  --experiment_name keypart_experiments

python utils/robustness/slope_ranking.py \
  --occlusion_mode mixed \
  --experiment_name keypart_experiments

python utils/visualization/robustness_plots.py \
  --experiment_name keypart_experiments

python utils/robustness/significance_test.py \
  --experiment_name keypart_experiments
```

说明：
- clean 场景现在默认开启，所以你不需要额外加 `--include_clean`。
- 如果你想只看遮挡测试，可以显式加 `--no_include_clean`。
- 该阶段不会重新训练模型，只会读取 `weights/` 里的权重文件。
- 输出的聚合指标里包含 95% CI 字段，便于你在论文里说明不同 seed 下的波动范围。
- 显著性检验脚本默认比较 `vit_fusion` 与排名 top1 的模型，结果更适合写成论文里的“显著优于/无显著差异”。

鲁棒性结果一般会输出三类东西：
1. 原始测试结果：每个 seed、每个场景的 macro-F1 和 balanced accuracy。
2. 汇总结果：均值、标准差、95% CI。
3. 排名结果：斜率、AUPC、clean->heavy 下降量和综合排序图。

### Step 4. backbone有多大，是不是300多M？

这一节是回答“模型好不好用”时必须补上的部分。鲁棒性强不代表模型一定轻量，效率指标可以帮助读者理解你为鲁棒性付出了多少额外开销。

```bash
conda activate one
python utils/report_efficiency.py \
  --models resnet vgg vit vit_fusion \
  --img_size 224 \
  --batch_size 1 \
  --warmup 20 \
  --iters 100 \
  --device auto
```

说明：
- 默认会输出 CSV 和 Markdown 报告到 `outputs/evaluation/efficiency/`。
- 若在 GPU 上测时延，建议固定环境并多次运行取平均；如需测试半精度可加 `--half`。
- 对 `vit_fusion`，脚本默认在 FLOPs 统计时关闭 attention guidance（仅用于提升统计兼容性），不影响时延统计路径。
- 如果环境里没有 `thop` 或 `fvcore`，脚本会自动尝试其它 FLOPs 统计后端，不会直接中断。
- 时延统计建议在同一张 GPU、同一批次大小下比较，否则不同环境的结果不可直接横向对比。

### Step 5. 样本可视化（可选）

如果你想解释“模型为什么把这张图判成这个类别”，就跑这一步。
这里的输出更偏向解释性分析，不影响主实验指标。

```bash
conda activate one
python utils/evaluate_models.py --models vit_fusion --dataset_subdir dataset/ship_fine --test_split test --output_subdir outputs/evaluation/default_eval
python utils/analysis.py visuals --model vit_fusion --csv outputs/evaluation/default_eval/preds_vit_fusion.csv --n 3 --out_dir outputs/visualizations/attention/default_eval
```

单图 rollout：

```bash
python utils/vit_fusion_rollout.py --image path/to/img.jpg --weights weights/vit_fusion_best.pth --output outputs/visualizations/attention/vit_fusion_rollout.png
```

输出解释：
- `evaluate_models.py` 会先生成预测 CSV、分类报告、混淆矩阵等基础文件。
- `analysis.py visuals` 会根据预测结果挑选样本并生成可视化图。
- `vit_fusion_rollout.py` 则是针对单张图片做 attention rollout，更适合答辩时展示。

## 5. 一键流程（可选）

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

## 6. 输出目录说明

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

## 7. Fusion 快速消融（只重训 vit_fusion）

如果你只想优化 fusion，而不想重训 resnet/vgg/vit，这一节最有用。
它的目的不是让你一次性找到“绝对最优”，而是快速判断哪类改动更可能改善鲁棒性。

```bash
conda activate one

# 方案 A：默认（推荐基线）
python train/train_vit_fusion.py --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42 --topk_patches 3 --weight_name vit_fusion_default_best.pth

# 方案 B：启用 cross-attn
python train/train_vit_fusion.py --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42 --topk_patches 3 --use_cross_attention --weight_name vit_fusion_cross_best.pth

# 方案 C：topk=2
python train/train_vit_fusion.py --train_occlusion_mode none --train_occlusion_p 0.0 --seed 42 --topk_patches 2 --use_cross_attention --weight_name vit_fusion_topk2_best.pth

# 方案 D：topk=3,use_local_self
python train/train_vit_fusion.py \
  --use_local_self_attention \
  --ablate_cross_attention
```
建议比较顺序：
1. 先看 `mixed_medium` 与 `mixed_heavy` 的 `macro_f1_mean`、`balanced_accuracy_mean`。
2. 再看 ranking 中的 slope 和 AUPC。
3. 最后看综合分图，判断总体稳定性。

如果你发现某个版本在 clean 上很强，但 heavy 下掉得很厉害，说明它“过拟合干净样本”而不是“真的更鲁棒”。

### 7.1 结合 rank2 的经验，当前 fusion 为什么可能不如旧版

从 `rank2.txt` 里的版本看，旧版 fusion 的几个关键点和当前代码不一样：

- `topk_patches=2`，而不是现在的 `3`。
  - 这意味着旧版局部分支只看两个最关键区域，噪声更少。
- `local_gate_init=1.0`，这会让局部分支更积极参与融合。
  - 旧版更愿意让局部分支参与融合，当前代码更偏向让全局分支主导。
- 旧版训练里局部/融合损失的权重没有压得这么低。
  - 这会让局部分支在训练中真正学到有用信息，而不是只做“微弱修饰”。

因此，rank2 结果更好的一个核心原因通常是：它在“利用局部信息”和“抑制局部噪声”之间找到了更平衡的位置。

### 7.2 当前代码的优先优化建议

如果你想把当前代码往 rank2 的效果靠，可以优先试下面三件事：

1. 把 `topk_patches` 从 `3` 改回 `2`。
   - 这是最直接的改动，通常能减少局部分支噪声。
2. 把 `local_gate_init` 调大一些，比如回到 `1.0` 附近。
  - 这样局部分支会更早参与融合，通常更接近 rank2 的行为。
3. 把局部分支的损失权重提高一点，不要让融合损失完全盖住局部学习。
   - 当前代码更偏“全局主导”，对遮挡任务来说可能过于保守。

如果还想继续加强，可以再做两类实验：
- 只在后半程打开 cross-attn，而不是全程默认开启。
- 用“轻度训练遮挡”替代纯 clean 训练，让模型提前适应 occlusion 分布。

这三项通常比盲目继续加复杂模块更有效。

## 8. 论文图与写作建议（精简）

建议主文至少包含：
1. 综合排序图（fig01）：给出总体结论
2. 均值±标准差曲线图（fig03）：展示退化趋势与稳定性
3. clean/heavy 混淆矩阵对照（fig06）：解释错误迁移

写作注意：
- 不要只报告综合分，需同时报告原始指标（macro_f1、balanced_accuracy、slope、AUPC）。
- 综合分是实验内相对归一化结果（现已在图中显示分量拆解），最差模型可能为 0，但不代表原始性能为 0。
- 建议同时引用 CI 与显著性检验结果，避免仅凭均值差异下结论。

## 9. 论文中放置“效率指标”建议

建议新增一个小节：`Efficiency Analysis`（或“复杂度与部署开销分析”）。

推荐放置位置：
1. 主文实验章节中，放在“鲁棒性结果”之后、“可视化分析”之前。
2. 表格建议命名为 `Table X: Model Efficiency Comparison`，列出 `Params(M)`、`FLOPs(G)`、`Latency(ms)`、`FPS`。
3. 正文里用 2-3 句总结“性能-鲁棒性-开销”的 trade-off（例如 fusion 在鲁棒性提升下带来的额外计算成本）。

若主文篇幅紧张：
1. 主文仅保留一个精简效率表（四模型对比）。
2. 将不同 batch size、FP32/FP16、CPU/GPU 详细数据放附录。

