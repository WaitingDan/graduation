# Key-part Occlusion Experiments（完整用法）

本目录用于“关键部件遮挡鲁棒性”实验，推荐和四模型统一训练结果配套使用。

## 1. 标准完整流程

### 1.1 生成场景指标（多模型 × 多种子 × 多遮挡等级）

```bash
python utils/keypart_experiments/run_occlusion_suite.py --include_clean --output_subdir outputs/keypart_experiments
```

说明：
- 默认模型：`resnet vgg vit vit_fusion`
- 默认种子：`42 123 3407`
- 默认遮挡：`mixed`，等级 `light medium heavy`

### 1.2 计算下降斜率并排序

```bash
python utils/keypart_experiments/analyze_robustness_slope.py --occlusion_mode mixed
```

### 1.3 生成可视化

```bash
python utils/visualize_robustness_ranking.py
```

### 1.4 导出论文草稿（可选）

```bash
python utils/keypart_experiments/export_report.py --out_md outputs/keypart_experiments/chapter4_draft.md
```

---

## 2. `run_occlusion_suite.py` 参数说明

- `--dataset_subdir`：数据集目录。
- `--test_split`：测试子目录名。
- `--models`：模型列表。
- `--seeds`：随机种子列表。
- `--include_clean`：是否加入 clean 场景。
- `--occlusion_mode`：遮挡模式（`block/stripe/mixed`）。
- `--occlusion_levels`：遮挡等级（可多选）。
- `--occlusion_p`：遮挡概率。
- `--output_subdir`：输出根目录。
- `--no_csv`：只输出 JSON。

### 常见命令

只跑 single-seed：

```bash
python utils/keypart_experiments/run_occlusion_suite.py --include_clean --seeds 42
```

只跑 heavy：

```bash
python utils/keypart_experiments/run_occlusion_suite.py --include_clean --occlusion_levels heavy
```

---

## 3. `analyze_robustness_slope.py` 参数说明

- `--agg_csv` / `--agg_json`：输入聚合指标。
- `--occlusion_mode`：要分析的遮挡模式。
- `--out_csv` / `--out_json` / `--out_md`：输出排名文件。
- `--no_csv`：不写 CSV。

说明：
- 脚本会优先读取 `agg_csv`，若不存在再读 `agg_json`。
- 排名依据是“随遮挡等级增加时性能下降斜率”。

---

## 4. 输出结构

- 汇总：
  - `summary_keypart_metrics.json`
  - `summary_keypart_metrics_agg.json`
- 排名：
  - `robustness_slope_ranking.json`
  - `robustness_slope_ranking.md`
- 明细：
  - `runs/<scenario>/seed_<seed>/preds_*.csv`
  - `runs/<scenario>/seed_<seed>/report_*.txt`

---

## 5. 相关脚本

- `export_occlusion_heatmaps.py`：导出遮挡图和热力图。
- `export_report.py`：导出论文章节草稿。
- `utils/visualize_robustness_ranking.py`：读取 ranking 文件并出图。
