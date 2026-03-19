# 舰船分类项目 - 详细使用指南（小白一步一步版）

本指南专门给初学者：告诉你**先运行什么、后运行什么**，并尽量覆盖项目里几乎所有可直接运行的脚本。

---

## 0. 先看这个（最重要）

- 推荐主入口是 `utils/analysis.py`（评估 / 可视化 / 流水线）。
- 训练脚本共 4 个：`train_resnet.py`、`train_vgg.py`、`train_vit.py`、`train_vit_fusion.py`。
- 推理脚本共 4 个：其中 `predict_vit_fusion.py` 支持命令行参数，另外 3 个是“直接运行版（内置路径）”。
- Windows 用户建议训练与评估都使用 `--num_workers 0`（训练脚本默认已是 0；`analysis.py` 默认是 2，建议手动改成 0）。

---

## 1. 环境准备（首次只做一次）

### 1.1 打开终端并进入项目目录
```powershell
cd D:\GraduationProjectThree
```

### 1.2 激活/创建 conda 环境
```powershell
conda activate labelme
```

如果报错“环境不存在”，执行：
```powershell
conda create -n labelme python=3.9 -y
conda activate labelme
```

### 1.3 安装依赖
```powershell
pip install torch torchvision scikit-learn matplotlib opencv-python pillow tqdm pandas openpyxl -i https://pypi.tsinghua.edu.cn/simple
```

---

## 2. 推荐运行顺序（建议照抄）

### 第1步：数据划分（如果你还没有 train/val/test）
```powershell
python utils/split_dataset.py ^
  --dataset_subdir dataset/ship_fine ^
  --source FGSCR ^
  --val_rate 0.2 ^
  --test_rate 0.1
```

如果你已经有 `dataset/ship_fine/train`、`val`、`test`，这一步跳过。

### 第2步：训练模型（先从 ResNet 开始）
```powershell
python train/train_resnet.py ^
  --dataset_subdir dataset/ship_fine ^
  --epochs 30 ^
  --batch_size 8 ^
  --num_workers 0 ^
  --weight_name resnet_best.pth ^
  --seed 42
```

### 第3步：评估模型（推荐统一入口）
```powershell
python utils/analysis.py eval --models vit --dataset_subdir dataset/ship_fine --test_split test --batch_size 8 --num_workers 0

python utils/analysis.py eval --models vit_fusion --dataset_subdir dataset/ship_fine --test_split test --batch_size 8 --num_workers 0
```

### 第4步：生成可视化（论文常用）
```powershell
python utils/analysis.py visuals --model vit --csv outputs/outputs_vit/preds_vit.csv --n 3 --out_dir outputs/outputs_vit/visuals_vit

python utils/analysis.py visuals --model vit_fusion --csv outputs/outputs_vit_fusion/preds_vit_fusion.csv --n 3 --out_dir outputs/outputs_vit_fusion/visuals_vit_fusion
```

### 第5步：单图预测（演示）
```powershell
python predict/predict_resnet.py
```

---

## 3. 全部训练脚本命令（4个）

### 3.1 训练 ResNet
```powershell
python train/train_resnet.py ^
  --dataset_subdir dataset/ship_fine ^
  --batch_size 8 ^
  --num_workers 0 ^
  --epochs 30 ^
  --lr 1e-4 ^
  --weight_name resnet_best.pth ^
  --seed 42 ^
  --early_stop_patience 10
```

### 3.2 训练 VGG
```powershell
python train/train_vgg.py ^
  --dataset_subdir dataset/ship_fine ^
  --batch_size 8 ^
  --num_workers 0 ^
  --epochs 30 ^
  --lr 1e-4 ^
  --weight_name vgg_best.pth ^
  --seed 42 ^
  --early_stop_patience 10
```

### 3.3 训练 ViT（支持 local-global 分支）
```powershell
python train/train_vit.py ^
  --dataset_subdir dataset/ship_fine ^
  --batch_size 8 ^
  --num_workers 0 ^
  --epochs 30 ^
  --lr 1e-4 ^
  --weight_name vit_best.pth ^
  --seed 42
```

启用 local-global 分支：
```powershell
python train/train_vit.py ^
  --dataset_subdir dataset/ship_fine ^
  --use_local_global ^
  --top_ratio 0.2 ^
  --weight_name vit_best.pth
```

### 3.4 训练 ViT Fusion（推荐毕业设计重点模型）
```powershell
python train/train_vit_fusion.py ^
  --dataset_subdir dataset/ship_fine ^
  --batch_size 8 ^
  --num_workers 0 ^
  --epochs 30 ^
  --lr 1e-4 ^
  --crop_size 112 ^
  --topk_patches 5 ^
  --dropout 0.2 ^
  --weight_name vit_fusion_best.pth ^
  --seed 42 ^
  --early_stop_patience 10
```

显存紧张时：
```powershell
python train/train_vit_fusion.py ^
  --dataset_subdir dataset/ship_fine ^
  --batch_size 4 ^
  --freeze_encoder_layers 6 ^
  --accum_steps 2
```

---

## 4. 评估与可视化脚本命令（工具链）

## 4.1 推荐：统一入口 `utils/analysis.py`

### A) 评估
```powershell
python utils/analysis.py eval ^
  --models resnet vgg vit vit_fusion ^
  --dataset_subdir dataset/ship_fine ^
  --test_split test ^
  --batch_size 8 ^
  --num_workers 0
```

### B) 可视化
```powershell
python utils/analysis.py visuals ^
  --model vit ^
  --csv outputs/preds_vit.csv ^
  --n 3 ^
  --out_dir outputs/visuals_vit
```

### C) 一条命令跑流水线（先评估再可视化）
```powershell
python utils/analysis.py pipeline ^
  --models resnet vgg vit vit_fusion ^
  --visuals ^
  --n 3 ^
  --out_dir outputs/visuals ^
  --dataset_subdir dataset/ship_fine ^
  --test_split test ^
  --batch_size 8 ^
  --num_workers 0
```

## 4.2 直接使用工具脚本（不走 analysis）

### A) 直接评估
```powershell
python utils/evaluate_models.py ^
  --models resnet vgg vit vit_fusion ^
  --dataset_subdir dataset/ship_fine ^
  --test_split test ^
  --batch_size 8 ^
  --num_workers 0
```

### B) 从预测CSV生成可视化
注意：该脚本目前 `--model` 只支持 `resnet` / `vgg` / `vit`。
```powershell
python utils/generate_visuals.py ^
  --model resnet ^
  --csv outputs/preds_resnet.csv ^
  --n 5 ^
  --out_dir outputs/visuals_resnet
```

### C) Grad-CAM（单图）
```powershell
python utils/gradcam_cnn_models.py ^
  --image test_images/test_ship_01.jpg ^
  --model resnet ^
  --weights weights/resnet_best.pth ^
  --out_dir outputs/gradcam
```

### D) Grad-CAM（从CSV批量导出）
```powershell
python utils/gradcam_cnn_models.py ^
  --csv outputs/preds_resnet.csv ^
  --model resnet ^
  --weights weights/resnet_best.pth ^
  --out_dir outputs/gradcam_resnet ^
  --wrong_only ^
  --max_samples 20
```

### E) ViT Attention Rollout（单图热力图）
```powershell
python utils/vit_attention_rollout.py ^
  --image test_images/test_ship_02.jpg ^
  --weights weights/vit_best.pth ^
  --output outputs/vit_attention_rollout.png ^
  --gamma 1.0 ^
  --alpha 0.6
```

### F) 兼容旧版流水线脚本
注意：`utils/run_pipeline.py` 目前 `--models` 只支持 `resnet` / `vgg` / `vit`。
```powershell
python utils/run_pipeline.py ^
  --models resnet vgg vit ^
  --visuals ^
  --n 3 ^
  --out_dir outputs/visuals
```

---

## 5. 预测脚本命令（4个）

## 5.1 无参数版（路径写在脚本里，直接运行）

```powershell
python predict/predict_resnet.py
python predict/predict_vgg.py
python predict/predict_vit.py
```

这 3 个脚本默认读取：
- 图片：`test_images/test_ship_01.jpg`
- 权重：`weights/<model>_best.pth`
- 类别：`class_indices.json`

## 5.2 参数版（推荐）`predict_vit_fusion.py`

```powershell
python predict/predict_vit_fusion.py ^
  --image test_images/test_ship_02.jpg ^
  --weights weights/vit_fusion_best.pth ^
  --class_indices class_indices.json ^
  --img_size 224 ^
  --crop_size 112 ^
  --topk_patches 5 ^
  --dropout 0.2
```

需要先加载 torchvision 预训练初始化时，再加 `--pretrained_init`。

---

## 6. 一键命令模板（最常用）

### 模板A：只做一个模型（ResNet）
```powershell
cd D:\GraduationProjectThree
conda activate labelme

python train/train_resnet.py --dataset_subdir dataset/ship_fine --epochs 30 --batch_size 8 --num_workers 0 --weight_name resnet_best.pth
python utils/analysis.py eval --models resnet --dataset_subdir dataset/ship_fine --batch_size 8 --num_workers 0
python utils/analysis.py visuals --model resnet --csv outputs/preds_resnet.csv --n 5 --out_dir outputs/visuals_resnet
```

### 模板B：四模型对比（毕业设计常用）
```powershell
python train/train_resnet.py --dataset_subdir dataset/ship_fine --weight_name resnet_best.pth
python train/train_vgg.py --dataset_subdir dataset/ship_fine --weight_name vgg_best.pth
python train/train_vit.py --dataset_subdir dataset/ship_fine --weight_name vit_best.pth
python train/train_vit_fusion.py --dataset_subdir dataset/ship_fine --weight_name vit_fusion_best.pth

python utils/analysis.py eval --models resnet vgg vit vit_fusion --dataset_subdir dataset/ship_fine --batch_size 8 --num_workers 0
```

---

## 7. 结果文件看哪里

- 权重：`weights/*.pth`
- 评估输出：`outputs/preds_*.csv`、`outputs/report_*.txt`、`outputs/per_class_*.csv`、`outputs/confmat_*.png`
- 可视化输出：`outputs/visuals*`、`outputs/gradcam*`、`outputs/vit_attention_rollout.png`

---

## 8. 常见报错快速处理

### 8.1 CUDA 显存不足
```powershell
python train/train_vit_fusion.py --batch_size 4 --freeze_encoder_layers 6 --accum_steps 2
```

### 8.2 Windows 多进程报错
把相关命令里的 `--num_workers` 改成 `0`。

### 8.3 权重找不到
```powershell
dir weights
```
确认文件名与命令中的 `--weight_name` / `--weights` 一致。

---

## 9. 查看每个脚本的帮助

```powershell
python utils/analysis.py --help
python utils/analysis.py eval --help
python utils/analysis.py visuals --help
python utils/split_dataset.py --help
python train/train_resnet.py --help
python train/train_vgg.py --help
python train/train_vit.py --help
python train/train_vit_fusion.py --help
python utils/evaluate_models.py --help
python utils/generate_visuals.py --help
python utils/gradcam_cnn_models.py --help
python utils/vit_attention_rollout.py --help
python utils/run_pipeline.py --help
python predict/predict_vit_fusion.py --help
```

如果你想要，我下一步可以再给你做一个“只需复制粘贴的 7 天实验计划版命令清单（每天跑什么）”。
