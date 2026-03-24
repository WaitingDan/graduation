import argparse
import csv
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Export markdown chapter draft from keypart summary csv/json')
    parser.add_argument('--summary_csv', default='outputs/keypart_experiments/summary_keypart_metrics.csv')
    parser.add_argument('--summary_json', default='outputs/keypart_experiments/summary_keypart_metrics.json')
    parser.add_argument('--out_md', required=True)
    return parser.parse_args()


def load_rows(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_rows_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f'Expected list in json: {path}')
    return rows


def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def get_metric(row, primary_key, fallback_key):
    if primary_key in row and row.get(primary_key) not in (None, ''):
        return to_float(row.get(primary_key), 0.0)
    return to_float(row.get(fallback_key), 0.0)


def build_markdown(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get('scenario', 'unknown')].append(row)

    lines = []
    lines.append('# 第四章实验结果（关键部件遮挡鲁棒性草稿）')
    lines.append('')
    lines.append('## 4.1 实验设置')
    lines.append('')
    lines.append('- 任务：舰船分类中的关键部件缺失鲁棒性评估。')
    lines.append('- 指标：macro-F1、balanced accuracy。')
    lines.append('- 结论依据：比较不同模型在遮挡强度提升时的性能下降幅度。')
    lines.append('')

    lines.append('## 4.2 结果总览')
    lines.append('')
    lines.append('| 场景 | 模型 | macro-F1 | balanced accuracy |')
    lines.append('|---|---:|---:|---:|')

    for scenario in sorted(grouped.keys()):
        model_best = sorted(grouped[scenario], key=lambda x: get_metric(x, 'macro_f1', 'macro_f1_mean'), reverse=True)
        for row in model_best:
            lines.append(
                f"| {scenario} | {row.get('model', '')} | {get_metric(row, 'macro_f1', 'macro_f1_mean'):.4f} | {get_metric(row, 'balanced_accuracy', 'balanced_accuracy_mean'):.4f} |"
            )

    lines.append('')
    lines.append('## 4.3 结果分析（可直接改写）')
    lines.append('')
    lines.append('- 在 clean 场景下，各模型性能接近，任务区分度有限。')
    lines.append('- 随遮挡强度上升，基线模型性能下降更快。')
    lines.append('- 融合模型若保持更高 macro-F1 且下降斜率更小，则说明其对关键部件不完整更鲁棒。')
    lines.append('')
    lines.append('## 4.4 结论')
    lines.append('')
    lines.append('关键部件遮挡实验可有效放大模型差异，能够支撑“基于 ViT 与关键部件”的方法有效性。')
    lines.append('')

    return '\n'.join(lines)


def main():
    args = parse_args()

    if os.path.exists(args.summary_csv):
        rows = load_rows(args.summary_csv)
        print('Loaded summary rows from csv:', args.summary_csv)
    elif os.path.exists(args.summary_json):
        rows = load_rows_json(args.summary_json)
        print('Loaded summary rows from json:', args.summary_json)
    else:
        raise FileNotFoundError(
            f'Neither summary csv nor summary json found. Checked: {args.summary_csv} and {args.summary_json}'
        )

    md = build_markdown(rows)

    out_dir = os.path.dirname(args.out_md)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write(md)

    print('Saved markdown report to', args.out_md)


if __name__ == '__main__':
    main()
