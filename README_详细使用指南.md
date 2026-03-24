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
python train/train_resnet.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 8 --num_workers 0 --weight_name resnet_best.pth --seed 42
```

### 第2步：评估该模型

```powershell
python utils/analysis.py eval --models resnet --dataset_subdir dataset/ship_fine --test_split test --batch_size 8 --num_workers 0
```

### 第3步：生成可视化

```powershell
python utils/analysis.py visuals --model resnet --csv outputs/preds_resnet.csv --n 5 --out_dir outputs/visuals_resnet
```

---

## 3. 四模型完整对比流程（毕业设计常用）

### 3.1 训练四个模型

```powershell
python train/train_resnet.py --dataset_subdir dataset/ship_fine --batch_size 8 --num_workers 0 --epochs 30 --lr 1e-4 --weight_name resnet_best.pth --seed 42
python train/train_vgg.py --dataset_subdir dataset/ship_fine --batch_size 8 --num_workers 0 --epochs 30 --lr 1e-4 --weight_name vgg_best.pth --seed 42
python train/train_vit.py --dataset_subdir dataset/ship_fine --batch_size 8 --num_workers 0 --epochs 30 --lr 1e-4 --weight_name vit_best.pth --seed 42
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --batch_size 8 --num_workers 0 --epochs 30 --lr 1e-4 --crop_size 112 --topk_patches 5 --dropout 0.2 --weight_name vit_fusion_best.pth --seed 42
```

### 3.2 统一评估

```powershell
python utils/analysis.py eval --models resnet vgg vit vit_fusion --dataset_subdir dataset/ship_fine --test_split test --batch_size 8 --num_workers 0 --eval_occlusion_mode none --eval_occlusion_level light --eval_occlusion_p 0.0 --output_subdir outputs --file_suffix ""
```

### 3.3 统一可视化（可选）

```powershell
python utils/analysis.py pipeline --models resnet vgg vit vit_fusion --visuals --n 3 --out_dir outputs/visuals --dataset_subdir dataset/ship_fine --test_split test --batch_size 8 --num_workers 0
```

---

## 4. 关键部件遮挡鲁棒性实验（重点）

## 4.1 批量运行 clean + 遮挡等级（多种子）

标准模式（保存 CSV + JSON）：

```powershell
python utils/keypart_experiments/run_occlusion_suite.py --dataset_subdir dataset/ship_fine --models resnet vgg vit vit_fusion --seeds 42 123 3407 --include_clean --occlusion_mode mixed --occlusion_levels light medium heavy --occlusion_p 1.0 --output_subdir outputs/keypart_experiments
```

精简模式（不保存 CSV，只保存 JSON）：

```powershell
python utils/keypart_experiments/run_occlusion_suite.py --dataset_subdir dataset/ship_fine --models resnet vgg vit vit_fusion --seeds 42 123 3407 --include_clean --occlusion_mode mixed --occlusion_levels light medium heavy --occlusion_p 1.0 --output_subdir outputs/keypart_experiments --no_csv
```

## 4.2 计算“性能下降斜率排序”

推荐命令（CSV 不在时会自动读 JSON）：

```powershell
python utils/keypart_experiments/analyze_robustness_slope.py --agg_csv outputs/keypart_experiments/summary_keypart_metrics_agg.csv --agg_json outputs/keypart_experiments/summary_keypart_metrics_agg.json --occlusion_mode mixed --out_json outputs/keypart_experiments/robustness_slope_ranking.json --out_md outputs/keypart_experiments/robustness_slope_ranking.md --no_csv
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
python utils/keypart_experiments/export_report.py --summary_csv outputs/keypart_experiments/summary_keypart_metrics.csv --summary_json outputs/keypart_experiments/summary_keypart_metrics.json --out_md outputs/keypart_experiments/chapter4_draft.md
```

---

## 5. 当前推荐：仅训练 + 批量评估

本项目当前流程不再依赖 `predict/` 与 `test_images/`。建议统一使用以下批量命令：

```powershell
python utils/analysis.py eval --models resnet vgg vit vit_fusion --dataset_subdir dataset/ship_fine --test_split test --batch_size 8 --num_workers 0
python utils/analysis.py pipeline --models resnet vgg vit vit_fusion --visuals --n 3 --out_dir outputs/visuals --dataset_subdir dataset/ship_fine --test_split test --batch_size 8 --num_workers 0
```

---

## 6. 常见问题

### 6.1 CUDA 显存不足

```powershell
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --batch_size 4 --freeze_encoder_layers 6 --accum_steps 2
```

### 6.2 Windows 多进程报错

把命令中的 --num_workers 改成 0。

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
python utils/visualize_robustness_ranking.py
```
