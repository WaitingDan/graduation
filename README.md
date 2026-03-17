# 舰船分类（Ship Classification）

本项目支持 `resnet` / `vgg` / `vit` 三种模型的训练、评估与可视化。

## 目录说明

- `train/`：分模型训练入口
- `predict/`：单图推理脚本
- `utils/`：评估与可视化工具（推荐统一入口 `utils/analysis.py`）
- `models/`：模型构建函数
- `outputs/`：评估与可视化输出目录

## 环境依赖

```bash
pip install torch torchvision scikit-learn matplotlib opencv-python pillow tqdm pytorch-grad-cam pandas openpyxl
```

## 常用命令

### 训练

```bash
python train/train_resnet.py
python train/train_vgg.py
python train/train_vit.py
```

### 评估

```bash
python utils/analysis.py eval --models resnet vgg vit
```

### 可视化

```bash
python utils/analysis.py visuals --model resnet --csv outputs/preds_resnet.csv --n 3 --out_dir outputs/visuals
```

### 一键流水线

```bash
python utils/analysis.py pipeline --models resnet --visuals --n 3 --out_dir outputs/visuals
```

## 输出文件

- `outputs/preds_<model>.csv`
- `outputs/report_<model>.txt`
- `outputs/confmat_<model>.png`
- `outputs/per_class_<model>.csv` / `outputs/per_class_<model>.xlsx` / `outputs/per_class_<model>.png`
- `outputs/visuals/visuals_manifest.csv`

## 说明

- `class_indices.json` 会在数据划分或训练阶段自动更新。
- 建议将 `outputs/`、`.vscode/`、`.idea/` 加入 `.gitignore`。
- 如需清理输出（Windows PowerShell）：`Remove-Item -Recurse -Force .\outputs\*`
