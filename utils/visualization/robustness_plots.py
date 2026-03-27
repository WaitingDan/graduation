#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)



def get_robustness_layout(output_subdir=None, experiment_name='keypart_experiments'):
    root = output_subdir or os.path.join('outputs', 'robustness', experiment_name)
    return {
        'root': root,
        'runs': os.path.join(root, 'runs'),
        'metrics': os.path.join(root, 'metrics'),
        'ranking': os.path.join(root, 'ranking'),
        'plots': os.path.join(root, 'plots'),
        'reports': os.path.join(root, 'reports'),
        'visuals': os.path.join(root, 'visuals'),
    }


def ensure_layout_dirs(root_dir, layout):
    for rel in layout.values():
        os.makedirs(os.path.join(root_dir, rel), exist_ok=True)


MODEL_COLORS = {
    'vit': '#1f77b4',
    'resnet': '#ff7f0e',
    'vit_fusion': '#2ca02c',
    'vgg': '#d62728',
}


def load_ranking_data(csv_file):
    data = {
        'model': [],
        'macro_f1_slope': [],
        'balanced_acc_slope': [],
        'macro_f1_drop': [],
        'balanced_acc_drop': []
    }

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['model'].append(row['model'])
            data['macro_f1_slope'].append(float(row['macro_f1_slope_per_level']))
            data['balanced_acc_slope'].append(float(row['balanced_accuracy_slope_per_level']))
            data['macro_f1_drop'].append(float(row['macro_f1_drop_clean_to_heavy']))
            data['balanced_acc_drop'].append(float(row['balanced_accuracy_drop_clean_to_heavy']))

    return data


def load_ranking_data_from_json(json_file):
    data = {
        'model': [],
        'macro_f1_slope': [],
        'balanced_acc_slope': [],
        'macro_f1_drop': [],
        'balanced_acc_drop': []
    }

    with open(json_file, 'r', encoding='utf-8') as f:
        rows = json.load(f)

    for row in rows:
        data['model'].append(row['model'])
        data['macro_f1_slope'].append(float(row['macro_f1_slope_per_level']))
        data['balanced_acc_slope'].append(float(row['balanced_accuracy_slope_per_level']))
        data['macro_f1_drop'].append(float(row['macro_f1_drop_clean_to_heavy']))
        data['balanced_acc_drop'].append(float(row['balanced_accuracy_drop_clean_to_heavy']))

    return data


def _model_colors(models):
    return [MODEL_COLORS.get(m, '#7f7f7f') for m in models]


def _normalize_smaller_better(values, floor=0.08):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    vmax = float(np.max(arr))
    vmin = float(np.min(arr))
    if np.isclose(vmax, vmin):
        return np.ones_like(arr)
    norm = (vmax - arr) / (vmax - vmin)
    return floor + (1.0 - floor) * norm


def _sorted_by_composite(data):
    import numpy as np

    abs_f1_slopes = np.abs(np.asarray(data['macro_f1_slope']))
    abs_acc_slopes = np.abs(np.asarray(data['balanced_acc_slope']))
    f1_drop = np.asarray(data['macro_f1_drop'])
    acc_drop = np.asarray(data['balanced_acc_drop'])

    f1_score = _normalize_smaller_better(abs_f1_slopes)
    acc_score = _normalize_smaller_better(abs_acc_slopes)
    f1_drop_score = _normalize_smaller_better(f1_drop)
    acc_drop_score = _normalize_smaller_better(acc_drop)

    composite = (f1_score + acc_score + f1_drop_score + acc_drop_score) / 4.0
    order = np.argsort(composite)[::-1]

    sorted_models = [data['model'][i] for i in order]
    sorted_scores = composite[order]

    details = {
        'f1_score': f1_score,
        'acc_score': acc_score,
        'f1_drop_score': f1_drop_score,
        'acc_drop_score': acc_drop_score,
        'composite': composite,
    }
    return order, sorted_models, sorted_scores, details


def plot_robustness_ranking(data, output_dir):
    import numpy as np
    import matplotlib.pyplot as plt

    order, sorted_models, _, _ = _sorted_by_composite(data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Robustness metric comparison', fontsize=16, fontweight='bold', y=0.98)

    chart_items = [
        ('|macro-F1 slope| (smaller is better)', np.abs(np.asarray(data['macro_f1_slope']))),
        ('|balanced-acc slope| (smaller is better)', np.abs(np.asarray(data['balanced_acc_slope']))),
        ('macro-F1 drop clean->heavy (smaller is better)', np.asarray(data['macro_f1_drop'])),
        ('balanced-acc drop clean->heavy (smaller is better)', np.asarray(data['balanced_acc_drop'])),
    ]

    for ax, (title, values) in zip(axes.flatten(), chart_items):
        sorted_values = values[order]
        bars = ax.barh(sorted_models, sorted_values, color=_model_colors(sorted_models), edgecolor='black', linewidth=0.8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.25, linestyle='--')
        margin = (float(np.max(sorted_values)) if len(sorted_values) else 1.0) * 0.05
        for i, (bar, val) in enumerate(zip(bars, sorted_values)):
            ax.text(val + margin, i, f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = os.path.join(output_dir, 'robustness_ranking_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print('Saved ranking visualization to', output_file)
    return output_file


def plot_robustness_summary(data, output_dir):
    import matplotlib.pyplot as plt

    _, sorted_models, sorted_scores, _ = _sorted_by_composite(data)

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(sorted_models, sorted_scores, color=_model_colors(sorted_models), alpha=0.9, edgecolor='black', linewidth=1.0)

    ax.set_xlabel('Composite robustness score', fontsize=12, fontweight='bold')
    ax.set_title('Composite robustness ranking (higher is better)', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.05])
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (bar, score) in enumerate(zip(bars, sorted_scores), start=1):
        ax.text(score + 0.015, i - 1, f'Rank {i}: {score:.4f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    output_file = os.path.join(output_dir, 'robustness_composite_ranking.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print('Saved composite ranking plot to', output_file)

    print('\n' + '=' * 50)
    print('Composite robustness ranking (best -> worst):')
    print('=' * 50)
    for i, (model, score) in enumerate(zip(sorted_models, sorted_scores), 1):
        print(f'{i}. {model:12s} - Composite score: {score:.4f}')
    print('=' * 50 + '\n')

    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_subdir', default=None, help='optional custom output root')
    parser.add_argument('--experiment_name', default='keypart_experiments', help='used when output_subdir is not provided')
    parser.add_argument('--ranking_csv', default=None)
    parser.add_argument('--ranking_json', default=None)
    args = parser.parse_args()

    layout = get_robustness_layout(output_subdir=args.output_subdir, experiment_name=args.experiment_name)
    ensure_layout_dirs(ROOT_DIR, layout)

    output_dir = os.path.join(ROOT_DIR, layout['plots'])
    csv_file = args.ranking_csv or os.path.join(ROOT_DIR, layout['ranking'], 'robustness_slope_ranking.csv')
    json_file = args.ranking_json or os.path.join(ROOT_DIR, layout['ranking'], 'robustness_slope_ranking.json')

    if os.path.exists(json_file):
        print('Reading ranking data from json:', json_file)
        data = load_ranking_data_from_json(json_file)
    elif os.path.exists(csv_file):
        print('Reading ranking data from csv:', csv_file)
        data = load_ranking_data(csv_file)
    else:
        raise FileNotFoundError(f'Ranking file not found. Checked: {json_file} and {csv_file}')

    plot_robustness_ranking(data, output_dir)
    plot_robustness_summary(data, output_dir)


if __name__ == '__main__':
    main()
