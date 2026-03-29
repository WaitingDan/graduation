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


def configure_matplotlib_cjk_font(preferred_font=None):
    import subprocess
    import matplotlib
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties

    if preferred_font:
        matplotlib.rcParams['font.sans-serif'] = [preferred_font, 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        print('Using user-specified font:', preferred_font)
        return preferred_font

    preferred_candidates = [
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'Noto Sans SC',
        'Source Han Sans CN',
        'Source Han Sans SC',
        'WenQuanYi Zen Hei',
        'WenQuanYi Micro Hei',
        'Microsoft YaHei',
        'SimHei',
        'PingFang SC',
        'Heiti SC',
        'Arial Unicode MS',
    ]

    available_names = {f.name for f in font_manager.fontManager.ttflist}
    selected = None
    for name in preferred_candidates:
        if name in available_names:
            selected = name
            break

    # Fallback: pick any likely CJK font family by name pattern.
    if selected is None:
        patterns = ('Noto Sans CJK', 'Source Han', 'WenQuanYi', 'YaHei', 'SimHei', 'PingFang', 'Heiti')
        for name in sorted(available_names):
            if any(pat in name for pat in patterns):
                selected = name
                break

    if selected is not None:
        matplotlib.rcParams['font.sans-serif'] = [selected, 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        print('Using detected CJK font:', selected)
        return selected

    # Fallback for environments where matplotlib cache misses system fonts.
    try:
        cmd = ['fc-list', ':lang=zh', 'file', 'family']
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        font_file = None
        for line in out.splitlines():
            parts = line.split(':', 1)
            if parts and parts[0].strip().lower().endswith(('.ttf', '.ttc', '.otf')):
                font_file = parts[0].strip()
                break

        if font_file and os.path.exists(font_file):
            font_manager.fontManager.addfont(font_file)
            loaded_name = FontProperties(fname=font_file).get_name()
            matplotlib.rcParams['font.sans-serif'] = [loaded_name, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            print('Using CJK font via fc-list:', loaded_name, 'from', font_file)
            return loaded_name
    except Exception:
        pass

    # Keep default if no CJK font is found; explain how to fix.
    print('Warning: no CJK font detected. Chinese labels may appear as boxes.')
    print('Hint: install one of these fonts, e.g. Noto Sans CJK / WenQuanYi.')
    matplotlib.rcParams['axes.unicode_minus'] = False
    return None


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
            # optional fields if present in ranking csv/json
            if 'macro_f1_aupc_norm' in row:
                data.setdefault('macro_f1_aupc_norm', []).append(float(row.get('macro_f1_aupc_norm', 0.0)))
            if 'balanced_accuracy_aupc_norm' in row:
                data.setdefault('balanced_acc_aupc_norm', []).append(float(row.get('balanced_accuracy_aupc_norm', 0.0)))

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
        data.setdefault('macro_f1_aupc_norm', []).append(float(row.get('macro_f1_aupc_norm', 0.0)))
        data.setdefault('balanced_acc_aupc_norm', []).append(float(row.get('balanced_accuracy_aupc_norm', 0.0)))

    return data


def _model_colors(models):
    return [MODEL_COLORS.get(m, '#7f7f7f') for m in models]


def _normalize_smaller_better(values, floor=0.0):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    vmax = float(np.max(arr))
    vmin = float(np.min(arr))
    if np.isclose(vmax, vmin):
        return np.ones_like(arr)
    norm = (vmax - arr) / (vmax - vmin)
    return floor + (1.0 - floor) * norm


def _normalize_bigger_better(values, floor=0.0):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    vmax = float(np.max(arr))
    vmin = float(np.min(arr))
    if np.isclose(vmax, vmin):
        return np.ones_like(arr)
    norm = (arr - vmin) / (vmax - vmin)
    return floor + (1.0 - floor) * norm


def _sorted_by_composite(data, weights=None):
    """Compute composite score with optional weights.

    weights: dict with keys 'aupc', 'slope', 'drop' summing to 1.0.
    """
    import numpy as np

    if weights is None:
        weights = {'aupc': 0.5, 'slope': 0.25, 'drop': 0.25}

    abs_f1_slopes = np.abs(np.asarray(data['macro_f1_slope']))
    abs_acc_slopes = np.abs(np.asarray(data['balanced_acc_slope']))
    f1_drop = np.asarray(data['macro_f1_drop'])
    acc_drop = np.asarray(data['balanced_acc_drop'])

    # read-aupc if present, else fallback to zeros
    f1_aupc_norm = np.asarray(data.get('macro_f1_aupc_norm') or [0.0] * len(data['model']), dtype=float)
    acc_aupc_norm = np.asarray(data.get('balanced_acc_aupc_norm') or [0.0] * len(data['model']), dtype=float)

    # normalize components to [0,1]
    aupc_score = _normalize_bigger_better((f1_aupc_norm + acc_aupc_norm) / 2.0)
    slope_score = (_normalize_smaller_better(abs_f1_slopes) + _normalize_smaller_better(abs_acc_slopes)) / 2.0
    drop_score = (_normalize_smaller_better(f1_drop) + _normalize_smaller_better(acc_drop)) / 2.0

    composite = weights.get('aupc', 0.5) * aupc_score + weights.get('slope', 0.25) * slope_score + weights.get('drop', 0.25) * drop_score

    order = np.argsort(composite)[::-1]

    sorted_models = [data['model'][i] for i in order]
    sorted_scores = composite[order]

    details = {
        'aupc_score': aupc_score,
        'slope_score': slope_score,
        'drop_score': drop_score,
        'composite': composite,
    }
    return order, sorted_models, sorted_scores, details


def plot_robustness_ranking(data, output_dir):
    import numpy as np
    import matplotlib.pyplot as plt

    order, sorted_models, _, _ = _sorted_by_composite(data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('鲁棒性指标比较', fontsize=16, fontweight='bold', y=0.98)

    chart_items = [
        ('|macro-F1 斜率|（越小越好）', np.abs(np.asarray(data['macro_f1_slope']))),
        ('|balanced-acc 斜率|（越小越好）', np.abs(np.asarray(data['balanced_acc_slope']))),
        ('macro-F1 从 clean 到 heavy 的下降量（越小越好）', np.asarray(data['macro_f1_drop'])),
        ('balanced-acc 从 clean 到 heavy 的下降量（越小越好）', np.asarray(data['balanced_acc_drop'])),
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

    # default weights: aupc 0.5, slope 0.25, drop 0.25
    weights = getattr(plot_robustness_summary, 'weights', None)
    _, sorted_models, sorted_scores, _ = _sorted_by_composite(data, weights=weights)

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(sorted_models, sorted_scores, color=_model_colors(sorted_models), alpha=0.9, edgecolor='black', linewidth=1.0)

    ax.set_xlabel('综合鲁棒性得分', fontsize=12, fontweight='bold')
    ax.set_title('综合鲁棒性排序（越大越好）', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.05])
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (bar, score) in enumerate(zip(bars, sorted_scores), start=1):
        ax.text(score + 0.015, i - 1, f'排名 {i}: {score:.4f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    output_file = os.path.join(output_dir, 'robustness_composite_ranking.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print('Saved composite ranking plot to', output_file)

    print('\n' + '=' * 50)
    print('综合鲁棒性排序（从最好到最差）:')
    print('=' * 50)
    for i, (model, score) in enumerate(zip(sorted_models, sorted_scores), 1):
        print(f'{i}. {model:12s} - 综合得分: {score:.4f}')
    print('=' * 50 + '\n')

    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_subdir', default=None, help='optional custom output root')
    parser.add_argument('--experiment_name', default='keypart_experiments', help='used when output_subdir is not provided')
    parser.add_argument('--ranking_csv', default=None)
    parser.add_argument('--ranking_json', default=None)
    parser.add_argument('--aupc_weight', default=None, help='weight for AUPC component (bigger better)')
    parser.add_argument('--slope_weight', default=None, help='weight for slope component (smaller better)')
    parser.add_argument('--drop_weight', default=None, help='weight for drop component (smaller better)')
    parser.add_argument('--font_family', default=None, help='optional matplotlib font family for Chinese text, e.g. SimHei')
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

    configure_matplotlib_cjk_font(preferred_font=args.font_family)

    # support optional weighting via CLI
    weights = None
    if hasattr(args, 'aupc_weight') and args.aupc_weight is not None:
        a = float(args.aupc_weight)
        s = float(args.slope_weight or 0.0)
        d = float(args.drop_weight or 0.0)
        total = a + s + d
        if total > 0:
            weights = {'aupc': a / total, 'slope': s / total, 'drop': d / total}

    # inject weights for plotting function
    if weights is not None:
        plot_robustness_summary.weights = weights

    plot_robustness_ranking(data, output_dir)
    plot_robustness_summary(data, output_dir)


if __name__ == '__main__':
    main()
