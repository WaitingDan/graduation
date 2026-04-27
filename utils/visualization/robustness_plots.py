#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
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


# A more publication-friendly, colorblind-safe palette. ``vit_fusion`` uses
# a distinct accent color and will be further highlighted in the plotting
# code (thicker edge / full opacity) so it stands out in figures.
MODEL_COLORS = {
    'vit': '#4C78A8',        # muted blue
    'resnet': '#F58518',     # warm orange
    'vit_fusion': '#7E2F8E', # accent purple (highlighted)
    'vgg': '#54A24B',        # muted green
    'vit_two_road': '#E45756', # dual-branch red
    # 'agvit': '#17BECF',      # cyan (removed)
}


PAPER_FONT_SIZE_PT = 10.5


def _pick_font_family(candidates, available_names):
    for name in candidates:
        if name in available_names:
            return name
    return None


def _contains_cjk(text):
    if not text:
        return False
    return re.search(r'[\u4e00-\u9fff]', str(text)) is not None


def _apply_mixed_font_rules(fig, zh_font, en_font, size_pt=PAPER_FONT_SIZE_PT):
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text

    zh_prop = FontProperties(family=zh_font, size=size_pt)
    en_prop = FontProperties(family=en_font, size=size_pt)

    for obj in fig.findobj(match=lambda x: isinstance(x, Text)):
        txt = obj.get_text()
        if _contains_cjk(txt):
            obj.set_fontproperties(zh_prop)
        else:
            obj.set_fontproperties(en_prop)


def configure_matplotlib_cjk_font(preferred_font=None):
    import subprocess
    import matplotlib
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties

    available_names = {f.name for f in font_manager.fontManager.ttflist}
    en_candidates = ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif']
    zh_candidates = [
        preferred_font,
        'SimSun',
        'Songti SC',
        'STSong',
        'Noto Serif CJK SC',
        'Noto Serif CJK JP',
        'Noto Serif CJK TC',
        'Noto Sans CJK SC',
        'Source Han Serif SC',
        'AR PL UMing CN',
    ]
    zh_candidates = [x for x in zh_candidates if x]

    en_font = _pick_font_family(en_candidates, available_names) or 'DejaVu Serif'

    if preferred_font:
        zh_font = preferred_font
        matplotlib.rcParams['font.family'] = [en_font]
        matplotlib.rcParams['font.size'] = PAPER_FONT_SIZE_PT
        matplotlib.rcParams['axes.unicode_minus'] = False
        print('Using user-specified Chinese font:', preferred_font)
        print('Using English/number font:', en_font)
        return zh_font, en_font

    selected = _pick_font_family(zh_candidates, available_names)

    # Fallback: pick any likely CJK font family by name pattern.
    if selected is None:
        patterns = ('SimSun', 'Songti', 'STSong', 'Noto CJK', 'Source Han', 'WenQuanYi', 'YaHei', 'SimHei')
        for name in sorted(available_names):
            if any(pat in name for pat in patterns):
                selected = name
                break

    if selected is not None:
        matplotlib.rcParams['font.family'] = [en_font]
        matplotlib.rcParams['font.size'] = PAPER_FONT_SIZE_PT
        matplotlib.rcParams['axes.unicode_minus'] = False
        print('Using detected Chinese font:', selected)
        print('Using English/number font:', en_font)
        return selected, en_font

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
            matplotlib.rcParams['font.family'] = [en_font]
            matplotlib.rcParams['font.size'] = PAPER_FONT_SIZE_PT
            matplotlib.rcParams['axes.unicode_minus'] = False
            print('Using Chinese font via fc-list:', loaded_name, 'from', font_file)
            print('Using English/number font:', en_font)
            return loaded_name, en_font
    except Exception:
        pass

    # Keep default if no CJK font is found; explain how to fix.
    print('Warning: no CJK font detected. Chinese labels may appear as boxes.')
    print('Hint: install Songti/SimSun/Noto Serif CJK SC for publication Chinese labels.')
    matplotlib.rcParams['font.family'] = [en_font]
    matplotlib.rcParams['font.size'] = PAPER_FONT_SIZE_PT
    matplotlib.rcParams['axes.unicode_minus'] = False
    return None, en_font


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

    order, sorted_models, _, details = _sorted_by_composite(data)

    # prepare AUPC (normalized) average as an additional panel
    f1_aupc = np.asarray(data.get('macro_f1_aupc_norm') or [0.0] * len(data['model']), dtype=float)
    acc_aupc = np.asarray(data.get('balanced_acc_aupc_norm') or [0.0] * len(data['model']), dtype=float)
    aupc_avg = (f1_aupc + acc_aupc) / 2.0

    # create a complete 3x2 grid (6 panels)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('鲁棒性指标比较', fontsize=PAPER_FONT_SIZE_PT, fontweight='bold', y=0.985)

    composite_raw = np.asarray(details['composite'])

    chart_items = [
        ('|macro-F1 斜率|（越小越好）', np.abs(np.asarray(data['macro_f1_slope']))),
        ('|balanced-acc 斜率|（越小越好）', np.abs(np.asarray(data['balanced_acc_slope']))),
        ('macro-F1 从 clean 到 heavy 的下降量（越小越好）', np.asarray(data['macro_f1_drop'])),
        ('balanced-acc 从 clean 到 heavy 的下降量（越小越好）', np.asarray(data['balanced_acc_drop'])),
        ('综合鲁棒性相对指数（越大越好）', composite_raw),
        ('AUPC(norm)（越大越好）', aupc_avg),
    ]

    flat_axes = axes.flatten()
    for ax, (title, values) in zip(flat_axes, chart_items):
        sorted_values = values[order]
        bars = ax.barh(sorted_models, sorted_values, color=_model_colors(sorted_models), edgecolor='black', linewidth=0.8)
        # emphasize fusion model visually: thicker edge + full opacity
        for i, (bar, model_name) in enumerate(zip(bars, sorted_models)):
            if model_name == 'vit_fusion':
                bar.set_edgecolor('#222222')
                bar.set_linewidth(1.6)
                try:
                    bar.set_alpha(1.0)
                except Exception:
                    pass
        ax.invert_yaxis()
        ax.set_title(title, fontsize=PAPER_FONT_SIZE_PT, fontweight='bold')
        ax.grid(axis='x', alpha=0.25, linestyle='--')
        x_max = float(np.max(sorted_values)) if len(sorted_values) else 1.0
        x_pad = max(0.03, x_max * 0.18)
        ax.set_xlim([0.0, x_max + x_pad])
        for i, (bar, val) in enumerate(zip(bars, sorted_values)):
            text_x = min(val + x_pad * 0.2, x_max + x_pad * 0.92)
            ax.text(text_x, i, f'{val:.4f}', va='center', fontsize=PAPER_FONT_SIZE_PT)

    zh_font = getattr(plot_robustness_ranking, 'zh_font', None)
    en_font = getattr(plot_robustness_ranking, 'en_font', 'Times New Roman')
    if zh_font:
        _apply_mixed_font_rules(fig, zh_font=zh_font, en_font=en_font, size_pt=PAPER_FONT_SIZE_PT)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_file = os.path.join(output_dir, 'robustness_ranking_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print('Saved ranking visualization to', output_file)
    return output_file


def plot_robustness_summary(data, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    # default weights: aupc 0.5, slope 0.25, drop 0.25
    weights = getattr(plot_robustness_summary, 'weights', None)
    if weights is None:
        weights = {'aupc': 0.5, 'slope': 0.25, 'drop': 0.25}
    order, sorted_models, sorted_scores, details = _sorted_by_composite(data, weights=weights)

    aupc_component = weights.get('aupc', 0.5) * details['aupc_score'][order]
    slope_component = weights.get('slope', 0.25) * details['slope_score'][order]
    drop_component = weights.get('drop', 0.25) * details['drop_score'][order]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    ax, ax_comp = axes

    bars = ax.barh(sorted_models, sorted_scores, color=_model_colors(sorted_models), alpha=0.9, edgecolor='black', linewidth=1.0)
    # highlight fusion in summary plot
    for i, (bar, model_name) in enumerate(zip(bars, sorted_models)):
        if model_name == 'vit_fusion':
            bar.set_edgecolor('#222222')
            bar.set_linewidth(1.8)
            try:
                bar.set_alpha(1.0)
            except Exception:
                pass

    ax.set_xlabel('相对鲁棒性指数（实验内归一化）', fontsize=PAPER_FONT_SIZE_PT, fontweight='bold')
    ax.set_title('综合鲁棒性相对排序（越大越好）', fontsize=PAPER_FONT_SIZE_PT, fontweight='bold')
    ax.set_xlim([0, 1.16])
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (bar, score) in enumerate(zip(bars, sorted_scores), start=1):
        label = f'排名 {i}: {score:.4f}'
        if score >= 0.30:
            text_x = score - 0.02
            ax.text(text_x, i - 1, label, va='center', ha='right', color='white', fontsize=PAPER_FONT_SIZE_PT, fontweight='bold')
        else:
            text_x = min(score + 0.02, 1.12)
            ax.text(text_x, i - 1, label, va='center', ha='left', color='black', fontsize=PAPER_FONT_SIZE_PT, fontweight='bold')

    y = np.arange(len(sorted_models))
    ax_comp.barh(y, aupc_component, label='AUPC分量', color='#4c78a8', edgecolor='black', linewidth=0.6)
    ax_comp.barh(y, slope_component, left=aupc_component, label='Slope分量', color='#f58518', edgecolor='black', linewidth=0.6)
    ax_comp.barh(y, drop_component, left=aupc_component + slope_component, label='Drop分量', color='#54a24b', edgecolor='black', linewidth=0.6)
    ax_comp.set_title('综合分构成拆解', fontsize=PAPER_FONT_SIZE_PT, fontweight='bold')
    ax_comp.set_xlabel('加权分量和（实验内相对）', fontsize=PAPER_FONT_SIZE_PT, fontweight='bold')
    ax_comp.set_xlim([0, 1.16])
    ax_comp.grid(axis='x', alpha=0.3, linestyle='--')
    ax_comp.set_yticks(y)
    ax_comp.set_yticklabels(sorted_models)
    ax_comp.legend(loc='lower right', fontsize=PAPER_FONT_SIZE_PT)

    fig.text(
        0.5,
        0.01,
        '注：该指数为实验内相对归一化结果，适合同一次实验内模型比较；不建议跨实验做绝对数值比较。',
        ha='center',
        fontsize=PAPER_FONT_SIZE_PT,
    )

    zh_font = getattr(plot_robustness_summary, 'zh_font', None)
    en_font = getattr(plot_robustness_summary, 'en_font', 'Times New Roman')
    if zh_font:
        _apply_mixed_font_rules(fig, zh_font=zh_font, en_font=en_font, size_pt=PAPER_FONT_SIZE_PT)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    output_file = os.path.join(output_dir, 'robustness_composite_ranking.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print('Saved composite ranking plot to', output_file)

    print('\n' + '=' * 50)
    print('综合鲁棒性相对排序（从最好到最差）:')
    print('=' * 50)
    for i, (model, score) in enumerate(zip(sorted_models, sorted_scores), 1):
        print(f'{i}. {model:12s} - 相对指数: {score:.4f}')
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

    zh_font, en_font = configure_matplotlib_cjk_font(preferred_font=args.font_family)

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

    plot_robustness_ranking.zh_font = zh_font
    plot_robustness_ranking.en_font = en_font
    plot_robustness_summary.zh_font = zh_font
    plot_robustness_summary.en_font = en_font

    plot_robustness_ranking(data, output_dir)
    plot_robustness_summary(data, output_dir)


if __name__ == '__main__':
    main()
