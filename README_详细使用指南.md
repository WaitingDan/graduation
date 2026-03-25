# 舰船分类项目 - 详细使用指南（小白一步一步版）

本指南面向深度学习初学者，目标是让你按照顺序直接复制命令就能跑通。所有命令都写成单行，不使用换行续写符。

---

## 0. 先看结论（最重要）

- 推荐统一入口：utils/analysis.py
- 训练脚本：train/train_resnet.py、train/train_vgg.py、train/train_vit.py、train/train_vit_fusion.py
- 关键部件遮挡实验脚本：utils/keypart_experiments/run_occlusion_suite.py
- Windows 建议：评估和训练都用 --num_workers 0（更稳）

---

## 1. 环境准备（首次一次）

### 1.1 进入项目目录

```powershell
cd D:\GraduationProjectThree
```

### 1.2 激活 conda 环境

```powershell
conda activate base
```

### 1.3 安装依赖

```powershell
pip install torch torchvision scikit-learn matplotlib opencv-python pillow tqdm pandas openpyxl -i https://pypi.tsinghua.edu.cn/simple
```

---

## 2. 推荐最小流程（先跑通再扩展）

### 第1步：训练一个模型（ResNet）

```powershell
python train/train_resnet.py
```

### 第2步：评估该模型

```powershell
python utils/analysis.py eval --models resnet
```

### 第3步：生成可视化

```powershell
python utils/analysis.py visuals --model resnet --csv outputs/preds_resnet.csv --n 3 --out_dir outputs/visuals_resnet
```

---

## 3. 四模型完整对比流程（毕业设计常用）

### 3.1 训练四个模型

```powershell
python train/train_resnet.py
python train/train_vgg.py
python train/train_vit.py 
python train/train_vit_fusion.py 
```

### 3.2 统一评估

```powershell
python utils/analysis.py eval --models resnet vgg vit vit_fusion
```

### 3.3 统一可视化（可选）

```powershell
python utils/analysis.py pipeline --models resnet vgg vit vit_fusion --visuals
```

---

## 4. 关键部件遮挡鲁棒性实验（重点）

## 4.1 批量运行 clean + 遮挡等级（多种子）

标准模式（保存 CSV + JSON）：

```powershell
python utils/keypart_experiments/run_occlusion_suite.py --include_clean
```

导出用于论文的批量遮挡图与各模型热力图（一次生成 clean/light/medium/heavy 并输出 Grad-CAM / rollout / fusion attention）：

```powershell
python utils/keypart_experiments/export_occlusion_heatmaps.py --images dataset/ship_fine/test/001.Nimitz-class_aircraft_carrier/P0031.bmp --models resnet vgg vit vit_fusion --include_clean --occlusion_mode mixed --occlusion_levels light medium heavy --out_dir outputs/keypart_experiments/occlusion_heatmaps --seed 42
```

精简模式（不保存 CSV，只保存 JSON）：

```powershell
python utils/keypart_experiments/run_occlusion_suite.py --include_clean --no_csv
```

## 4.2 计算“性能下降斜率排序”

推荐命令（CSV 不在时会自动读 JSON）：

```powershell
python utils/keypart_experiments/analyze_robustness_slope.py
```

## 4.3 画排序图（更直观）

```powershell
python utils/visualize_robustness_ranking.py
```

会输出：
- outputs/keypart_experiments/robustness_ranking_visualization.png
- outputs/keypart_experiments/robustness_composite_ranking.png

## 4.4 导出论文章节草稿

推荐命令（CSV 不在时自动读 JSON）：

```powershell
python utils/keypart_experiments/export_report.py --out_md outputs/keypart_experiments/chapter4_draft.md
```

---

## 5. 当前推荐：仅训练 + 批量评估

本项目当前流程不再依赖 `predict/` 与 `test_images/`。建议统一使用以下批量命令：

```powershell
python utils/analysis.py eval --models resnet vgg vit vit_fusion
python utils/analysis.py pipeline --models resnet vgg vit vit_fusion --visuals
```

---

## 6. 常见问题

### 6.1 CUDA 显存不足

```powershell
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --batch_size 4 --freeze_encoder_layers 6 --accum_steps 2
```

### 6.2 Windows 多进程报错

把命令中的 `--num_workers` 改成 `0`（例如 `python utils/analysis.py eval --models resnet --num_workers 0`）。

### 6.3 权重文件找不到

```powershell
dir weights
```

---

## 7. 常用产物说明（看结果去哪里）

- 权重：weights/*.pth
- 常规评估：outputs/preds_*.csv、outputs/report_*.txt、outputs/per_class_*.csv、outputs/confmat_*.png
- 关键部件实验：outputs/keypart_experiments/
- 鲁棒性图表：outputs/keypart_experiments/robustness_ranking_visualization.png、outputs/keypart_experiments/robustness_composite_ranking.png

---

## 8. 一键查看帮助

```powershell
python utils/analysis.py --help
python utils/analysis.py eval --help
python utils/analysis.py visuals --help
python utils/analysis.py pipeline --help
python train/train_resnet.py --help
python train/train_vgg.py --help
python train/train_vit.py --help
python train/train_vit_fusion.py --help
python utils/keypart_experiments/run_occlusion_suite.py --help
python utils/keypart_experiments/analyze_robustness_slope.py --help
python utils/keypart_experiments/export_report.py --help
python utils/keypart_experiments/export_occlusion_heatmaps.py --help
python utils/generate_visuals.py --help
python utils/gradcam_cnn_models.py --help
python utils/run_pipeline.py --help
python utils/split_dataset.py --help
python utils/visualize_robustness_ranking.py
python utils/vit_attention_rollout.py --image 你的图片路径.jpg --weights weights/vit_best.pth --output outputs/vit_attention_rollout.png
```

---

## 脚本速览（简要说明）

- `utils/generate_visuals.py`：从 `preds_*.csv` 中选择代表性样本并为指定模型生成 Grad-CAM / rollout 可视化，适合快速导出论文用图候选集。
- `utils/gradcam_cnn_models.py`：对 `resnet` / `vgg` 提供单图或批量 Grad-CAM 导出功能（支持从 preds CSV 批量导出）。
- `utils/run_pipeline.py`：简洁的流水线入口，等价于 `utils/analysis.py pipeline`，便于在脚本/调度器中调用。
- `utils/split_dataset.py`：按比例把数据集分为 `train/val/test`，便于快速重建实验集划分。
- `utils/keypart_experiments/export_occlusion_heatmaps.py`：论文图专用脚本，一键为指定图片生成 clean/light/medium/heavy 遮挡图并导出各模型热力图（Grad-CAM / rollout / fusion attention）。


## 9. 各脚本常用参数作用速查（重点）

说明：下面只列“常用且你经常会改”的参数。其余参数建议先用默认值。

### 9.1 训练脚本（`train/train_resnet.py`、`train/train_vgg.py`、`train/train_vit.py`、`train/train_vit_fusion.py`）

- `--batch_size`：每次送入模型的图片数量。越大越快但更占显存。
- `--lr`：学习率。越大收敛更快但不稳定风险更高；越小更稳但更慢。
- `--epochs`：最多训练轮数。
- `--early_stop_patience`：早停耐心值。连续多少个 epoch 指标无提升就提前停止。
- `--num_workers`：DataLoader 子进程数。Windows 推荐 0；Linux 可增大提速。
- `--seed`：随机种子，控制复现实验。
- `--weight_name`：保存到 `weights/` 的权重文件名。
- `--dataset_subdir`：数据集根目录（默认 `dataset/ship_fine`）。
- `--train_occlusion_mode` / `--train_occlusion_level` / `--train_occlusion_p`：训练时数据增强遮挡策略。

ViT-Fusion 额外常用：

- `--crop_size`：局部分支裁剪窗口大小。
- `--topk_patches`：从全局注意力中取多少个关键 patch 来决定局部区域。
- `--dropout`：融合分类头 dropout 比例。
- `--accum_steps`：梯度累积步数，显存不够时可增大。
- `--freeze_encoder_layers`：冻结前 N 层编码器，降低训练开销与过拟合风险。

### 9.2 统一入口（`utils/analysis.py`）

`eval` 子命令常用：

- `--models`：要评估的模型列表（如 `resnet vgg vit vit_fusion`）。
- `--batch_size`：评估 batch 大小。
- `--num_workers`：读取数据并行度。
- `--eval_occlusion_mode` / `--eval_occlusion_level` / `--eval_occlusion_p`：评估时注入遮挡（做鲁棒性测试）。
- `--output_subdir`：输出目录。
- `--file_suffix`：输出文件后缀，便于多组实验区分。
- `--seed`：评估随机种子。

`visuals` 子命令常用：

- `--model`：可视化对应模型。
- `--csv`：输入预测文件（如 `outputs/preds_resnet.csv`）。
- `--n`：每类（高置信正确/低置信错误/高置信错误）抽样数量。
- `--out_dir`：可视化图片输出目录。

`pipeline` 子命令常用：

- `--models`：评估并可视化哪些模型。
- `--visuals`：是否在评估后自动生成可视化。

### 9.3 关键部件实验（`utils/keypart_experiments/run_occlusion_suite.py`）

- `--include_clean`：是否包含 clean（无遮挡）场景。
- `--occlusion_mode`：遮挡类型（`block` / `stripe` / `mixed`）。
- `--occlusion_levels`：遮挡等级（`light` / `medium` / `heavy`）。
- `--occlusion_p`：对样本施加遮挡的概率（实验里常设 `1.0`）。
- `--seeds`：多随机种子重复，便于统计均值和方差。
- `--models`：参与评估的模型。
- `--no_csv`：只保存 JSON，不保存 CSV。
- `--output_subdir`：实验结果根目录。

### 9.4 斜率排序与报告导出

`utils/keypart_experiments/analyze_robustness_slope.py`：

- `--agg_csv` / `--agg_json`：输入聚合指标。
- `--occlusion_mode`：指定计算哪种遮挡模式的斜率。
- `--out_csv` / `--out_json` / `--out_md`：排序结果输出路径。
- `--no_csv`：不输出 CSV，仅输出 JSON/MD。

`utils/keypart_experiments/export_report.py`：

- `--summary_csv` / `--summary_json`：输入汇总指标。
- `--out_md`：导出的论文章节草稿路径（必填）。

### 9.5 热力图脚本

`utils/vit_attention_rollout.py`：

- `--image`：输入图片路径（必填）。
- `--weights`：ViT 权重路径。
- `--output`：输出热力图路径。
- `--gamma`：热力图对比度调节（<1 扩散更多区域，>1 更聚焦）。
- `--alpha`：热力图与原图融合比例。

`utils/keypart_experiments/export_occlusion_heatmaps.py`：

- `--images`：要生成遮挡与热力图的图片列表（必填）。
- `--models`：要可视化的模型列表。
- `--include_clean`：是否同时导出 clean 场景。
- `--occlusion_mode` / `--occlusion_levels`：遮挡模式与等级。
- `--fusion_rollout_mix`：Fusion 可视化中 rollout 占比（越大越“全局多区域”）。
- `--fusion_rollout_blur`：Fusion 热力图平滑核大小（越大越平滑）。
- `--fusion_rollout_gamma`：Fusion 热力图对比度（<1 更扩散，>1 更聚焦）。
- `--out_dir`：导出目录。

---

## 10. 当前代码口径说明（2026-03 更新）

- 已移除 `predict/` 与 `test_images/`，主流程仅保留训练与批量评估。
- `train/train_vit.py` 当前仅训练标准 ViT，不再包含 local-global 可选分支。
- 如需 ViT attention rollout，可使用 `utils/vit_attention_rollout.py`，并必须显式传入 `--image`。
