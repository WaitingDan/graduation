# 实验命令小抄（论文复现版）

下面是一组用于论文复现的常用命令，直接在项目根目录运行（`/mnt/e/aircas/dht/graduation`）。只包含你最常用且非默认才需要显式写出的参数。

1) 训练 ResNet（默认参数）

```bash
python train/train_resnet.py
```

2) 训练 VGG（默认参数）

```bash
python train/train_vgg.py
```

3) 训练 ViT（微调示例 — 减小 batch 与 lr）

```bash
python train/train_vit.py --batch_size 16 --lr 2e-4
```

4) 训练 ViT-Fusion（常用示例）

```bash
python train/train_vit_fusion.py --batch_size 16 --lr 2e-4
```

5) 单模型评估（ResNet）

```bash
python utils/analysis.py eval --models resnet
```

6) 四模型统一评估

```bash
python utils/analysis.py eval --models resnet vgg vit vit_fusion
```

7) 评估并生成可视化（pipeline）

```bash
python utils/analysis.py pipeline --models resnet vgg vit vit_fusion --visuals
```

8) 关键部件遮挡鲁棒性实验（包括 clean）

```bash
python utils/keypart_experiments/run_occlusion_suite.py --include_clean
```

9) 导出论文用遮挡 + 热力图（单张图示例）

```bash
python utils/keypart_experiments/export_occlusion_heatmaps.py --images dataset/ship_fine/test/001.Nimitz-class_aircraft_carrier/P0031.bmp --models resnet vgg vit vit_fusion --include_clean --occlusion_mode mixed --occlusion_levels light medium heavy --out_dir outputs/keypart_experiments/occlusion_heatmaps --seed 42
```

10) 计算性能下降斜率并排序

```bash
python utils/keypart_experiments/analyze_robustness_slope.py
```

11) 导出论文章节草稿（从实验汇总生成 Markdown）

```bash
python utils/keypart_experiments/export_report.py --out_md outputs/keypart_experiments/chapter4_draft.md
```

12) 批量从 preds CSV 生成 Grad-CAM 可视化（示例）

```bash
python utils/gradcam_cnn_models.py --csv outputs/preds_resnet.csv --model resnet --max_samples 20
```

13) ViT attention rollout（单张图）

```bash
python utils/vit_attention_rollout.py --image path/to/image.jpg --weights weights/vit_best.pth
```

说明：如果你希望一个更短的 "最终论文命令集"（只保留最终 8-12 条真正用于结果的命令），告诉我我会生成一份只包含你要提交到论文的命令清单。
