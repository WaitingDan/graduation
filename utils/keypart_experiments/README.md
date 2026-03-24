# Key-part Occlusion Experiments

当前说明：
- 本实验流程默认评估模型为 `resnet` / `vgg` / `vit` / `vit_fusion`。
- 项目当前不包含 `predict/` 与 `test_images/` 目录，建议统一走批量评估流程。
- `train/train_vit.py` 训练的是标准 ViT；融合模型请使用 `train/train_vit_fusion.py`。

## 1) 批量评估（标准模式，保存 CSV + JSON）

```bash
python utils/keypart_experiments/run_occlusion_suite.py --dataset_subdir dataset/ship_fine --models resnet vgg vit vit_fusion --seeds 42 123 3407 --include_clean --occlusion_mode mixed --occlusion_levels light medium heavy --occlusion_p 1.0 --output_subdir outputs/keypart_experiments
```

输出：
- outputs/keypart_experiments/summary_keypart_metrics.csv
- outputs/keypart_experiments/summary_keypart_metrics.json
- outputs/keypart_experiments/summary_keypart_metrics_agg.csv
- outputs/keypart_experiments/summary_keypart_metrics_agg.json
- outputs/keypart_experiments/runs/...

## 2) 批量评估（精简模式，不保存 CSV）

```bash
python utils/keypart_experiments/run_occlusion_suite.py --dataset_subdir dataset/ship_fine --models resnet vgg vit vit_fusion --seeds 42 123 3407 --include_clean --occlusion_mode mixed --occlusion_levels light medium heavy --occlusion_p 1.0 --output_subdir outputs/keypart_experiments --no_csv
```

## 3) 导出章节草稿（CSV 不在时自动读取 JSON）

```bash
python utils/keypart_experiments/export_report.py --summary_csv outputs/keypart_experiments/summary_keypart_metrics.csv --summary_json outputs/keypart_experiments/summary_keypart_metrics.json --out_md outputs/keypart_experiments/chapter4_draft.md
```

## 4) 计算下降斜率并排序（CSV 不在时自动读取 agg JSON）

```bash
python utils/keypart_experiments/analyze_robustness_slope.py --agg_csv outputs/keypart_experiments/summary_keypart_metrics_agg.csv --agg_json outputs/keypart_experiments/summary_keypart_metrics_agg.json --occlusion_mode mixed --out_json outputs/keypart_experiments/robustness_slope_ranking.json --out_md outputs/keypart_experiments/robustness_slope_ranking.md --no_csv
```

说明：
- slope_per_level 越接近 0（数值越大）表示下降越慢、鲁棒性越强。
- drop_clean_to_heavy 越小表示从 clean 到 heavy 的整体退化越小。

## 5) 可视化排序结果

```bash
python utils/visualize_robustness_ranking.py
```

说明：脚本会优先读取 outputs/keypart_experiments/robustness_slope_ranking.json；若不存在再读取同名 CSV。

## 6) 输出目录说明（避免重复存放）

- `run_occlusion_suite.py` 的默认输出根目录是 `outputs/keypart_experiments`。
- 每次评估的明细文件（`preds_*`、`report_*`、`per_class_*`、`confmat_*`）会写入：
	- `outputs/keypart_experiments/runs/<scenario>/seed_<seed>/`
- 因此在 keypart 实验流程中，不需要再额外把同一批模型结果重复放到 `outputs/` 根目录。
- `outputs/` 根目录下的结果通常用于通用评估流程（如 `utils/analysis.py eval`），与 keypart 实验可并行存在但不是必须。
