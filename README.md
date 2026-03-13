# 舰船分类项目使用说明书

本项目支持 3 个模型（`resnet` / `vgg` / `vit`）的训练、评估与可解释性可视化（Grad-CAM / ViT Attention Rollout）。

## 1. 项目目标

- 训练 10 类舰船分类模型
- 输出评估结果（预测 CSV、分类报告、混淆矩阵）
- 基于评估结果自动筛选样本并生成热力图可视化

---

## 2. 目录说明（核心）

- `train/`：训练脚本（当前为分模型脚本）
- `predict/`：单张图片预测脚本（当前为分模型脚本）
- `utils/analysis.py`：**统一入口（推荐）**
- `utils/evaluate_models.py`：批量评估并保存结果
- `utils/generate_visuals.py`：从预测 CSV 选样并批量可视化
- `utils/gradcam_cnn_models.py`：CNN（ResNet/VGG）Grad-CAM
- `utils/vit_attention_rollout.py`：ViT attention rollout
- `weights/`：模型权重
- `outputs/`：评估与可视化输出目录（自动创建）

---

## 3. 环境准备

你当前使用的是 Conda 环境（如 `labelme`）。建议在项目根目录执行命令。

安装依赖（如未安装）：

```bash
pip install torch torchvision timm scikit-learn matplotlib opencv-python pillow tqdm pytorch-grad-cam
```

> 如果你使用 CUDA，请确保安装与显卡/驱动匹配的 PyTorch 版本。

---

## 4. 推荐用法（统一入口）

统一入口脚本：`utils/analysis.py`

### 4.1 仅评估（生成 CSV + 报告 + 混淆矩阵）

```bash
python utils/analysis.py eval --models resnet vgg vit
```

### 4.2 仅可视化（从指定 CSV 生成热力图）

```bash
python utils/analysis.py visuals --model resnet --csv outputs/preds_resnet.csv --n 3 --out_dir outputs/visuals
```

参数说明：
- `--model`：`resnet` / `vgg` / `vit`
- `--csv`：对应模型的预测 CSV
- `--n`：每类选样数量（高置信正确 / 低置信错误 / 高置信错误，各 `n` 张）
- `--out_dir`：可视化输出目录

### 4.3 一键流水线（先评估，再可视化）

```bash
python utils/analysis.py pipeline --models resnet --visuals --n 3 --out_dir outputs/visuals
```

---

## 5. 输出文件说明

运行评估后会在 `outputs/` 下生成：

- `preds_<model>.csv`：逐样本预测结果
  - 字段：`filepath,true_idx,true_name,pred_idx,pred_name,prob`
- `report_<model>.txt`：分类报告（Precision/Recall/F1）
- `confmat_<model>.png`：混淆矩阵图

运行可视化后会在 `outputs/visuals/` 下生成：

- 热力图文件（文件名含标签与置信度）
  - 示例：`high_conf_correct_P0051_t0_p0_0.998.png`
- `visuals_manifest.csv`
  - 字段：`tag,filepath,saved_visual,true_idx,pred_idx,prob`
  - `tag` 含义：
    - `high_conf_correct`
    - `low_conf_incorrect`
    - `high_conf_incorrect`

---

## 6. 训练与单图预测（当前保留）

### 6.1 训练

```bash
python train/train_resnet.py
python train/train_vgg.py
python train/train_vit.py
```

训练输出：
- `weights/<model>_best.pth`
- `outputs/<model>_loss_curve.png`
- `outputs/<model>_accuracy_curve.png`

### 6.2 单图预测

```bash
python predict/predict_resnet.py
python predict/predict_vgg.py
python predict/predict_vit.py
```

> 这些脚本默认读取 `test_images/test_ship_01.jpg`。如需改图片，请修改脚本里的 `image_path`。

---

## 7. 常见问题

### 7.1 `ModuleNotFoundError: No module named 'utils'`

请在项目根目录运行命令，或使用绝对路径运行脚本：

```bash
python d:/GraduationProjectThree/utils/analysis.py --help
```

### 7.2 运行很慢或显存不足

- 降低 batch size（训练/评估脚本中）
- 仅跑单模型：`--models resnet`
- 先做 `eval`，再小规模做 `visuals --n 1`

### 7.3 找不到权重文件

确保以下文件存在：
- `weights/resnet_best.pth`
- `weights/vgg_best.pth`
- `weights/vit_best.pth`

如果缺失，请先训练对应模型。

---

## 8. 推荐日常流程（最简）

1) 评估：

```bash
python utils/analysis.py eval --models resnet vgg vit
```

2) 生成热力图（例如 ResNet）：

```bash
python utils/analysis.py visuals --model resnet --csv outputs/preds_resnet.csv --n 3 --out_dir outputs/visuals
```

3) 写论文时使用：
- `outputs/report_*.txt`
- `outputs/confmat_*.png`
- `outputs/visuals/*.png`
- `outputs/visuals/visuals_manifest.csv`

---

## 9. 兼容入口（仍可用）

- `utils/run_pipeline.py`：兼容旧命令，内部已转到统一逻辑。

示例：

```bash
python utils/run_pipeline.py --models resnet --visuals --n 3
```

建议优先使用 `utils/analysis.py`。
