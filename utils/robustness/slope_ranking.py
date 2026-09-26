import argparse
import csv
import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


INTENSITY_MAP = {
    'clean': 0.0,
    'light': 0.10,
    'medium': 0.20,
    'heavy': 0.35,
}

from utils.output_layout import get_robustness_layout, ensure_layout_dirs


def parse_args():
    parser = argparse.ArgumentParser(description='Compute robustness degradation slope and ranking for models')
    parser.add_argument('--output_subdir', default=None, help='optional custom output root')
    parser.add_argument('--experiment_name', default='keypart_experiments', help='used when output_subdir is not provided')
    parser.add_argument('--agg_csv', default=None)
    parser.add_argument('--agg_json', default=None)
    parser.add_argument('--occlusion_mode', default='mixed', choices=['block', 'stripe', 'mixed'])
    parser.add_argument('--out_csv', default=None)
    parser.add_argument('--out_json', default=None)
    parser.add_argument('--out_md', default=None)
    parser.add_argument('--no_csv', action='store_true', help='do not save ranking csv file')
    return parser.parse_args()


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def read_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def linear_slope(xs, ys):
    if len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return 0.0
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return num / den


def area_under_curve(xs, ys, normalize=True):
    if len(xs) < 2:
        return 0.0

    points = sorted(zip(xs, ys), key=lambda item: item[0])
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
        dx = x2 - x1
        if dx <= 0:
            continue
        area += 0.5 * (y1 + y2) * dx

    if not normalize:
        return area

    span = points[-1][0] - points[0][0]
    if span <= 0:
        return 0.0
    return area / span


def pick_metric_by_model(rows, model, scenario, metric_key):
    for row in rows:
        if row.get('model') == model and row.get('scenario') == scenario:
            return to_float(row.get(metric_key, 0.0))
    return None


def build_model_ranking(rows, occlusion_mode):
    level_seq = ['light', 'medium', 'heavy']
    scenarios = [(f'{occlusion_mode}_{lv}', INTENSITY_MAP[lv]) for lv in level_seq]

    models = sorted(set(row.get('model') for row in rows if row.get('model')))
    ranking = []

    for model in models:
        clean_macro = pick_metric_by_model(rows, model, 'clean', 'macro_f1_mean')
        clean_bal = pick_metric_by_model(rows, model, 'clean', 'balanced_accuracy_mean')

        macro_x = []
        macro_y = []
        bal_x = []
        bal_y = []

        if clean_macro is not None:
            macro_x.append(INTENSITY_MAP['clean'])
            macro_y.append(clean_macro)
        if clean_bal is not None:
            bal_x.append(INTENSITY_MAP['clean'])
            bal_y.append(clean_bal)

        for scenario, intensity in scenarios:
            macro_val = pick_metric_by_model(rows, model, scenario, 'macro_f1_mean')
            bal_val = pick_metric_by_model(rows, model, scenario, 'balanced_accuracy_mean')
            if macro_val is not None:
                macro_x.append(intensity)
                macro_y.append(macro_val)
            if bal_val is not None:
                bal_x.append(intensity)
                bal_y.append(bal_val)

        macro_slope = linear_slope(macro_x, macro_y)
        bal_slope = linear_slope(bal_x, bal_y)
        macro_aupc = area_under_curve(macro_x, macro_y, normalize=False)
        bal_aupc = area_under_curve(bal_x, bal_y, normalize=False)
        macro_aupc_norm = area_under_curve(macro_x, macro_y, normalize=True)
        bal_aupc_norm = area_under_curve(bal_x, bal_y, normalize=True)

        heavy_scenario = f'{occlusion_mode}_heavy'
        heavy_macro = pick_metric_by_model(rows, model, heavy_scenario, 'macro_f1_mean')
        heavy_bal = pick_metric_by_model(rows, model, heavy_scenario, 'balanced_accuracy_mean')

        drop_macro = (clean_macro - heavy_macro) if (clean_macro is not None and heavy_macro is not None) else None
        drop_bal = (clean_bal - heavy_bal) if (clean_bal is not None and heavy_bal is not None) else None

        ranking.append({
            'model': model,
            'macro_f1_slope_per_ratio': macro_slope,
            'balanced_accuracy_slope_per_ratio': bal_slope,
            # Backward-compatible field names used by existing plotting scripts.
            'macro_f1_slope_per_level': macro_slope,
            'balanced_accuracy_slope_per_level': bal_slope,
            'macro_f1_aupc': macro_aupc,
            'balanced_accuracy_aupc': bal_aupc,
            'macro_f1_aupc_norm': macro_aupc_norm,
            'balanced_accuracy_aupc_norm': bal_aupc_norm,
            'macro_f1_drop_clean_to_heavy': drop_macro,
            'balanced_accuracy_drop_clean_to_heavy': drop_bal,
        })

    ranking.sort(
        key=lambda x: (
            x['macro_f1_aupc_norm'],
            x['balanced_accuracy_aupc_norm'],
            x['macro_f1_slope_per_level'],
            x['balanced_accuracy_slope_per_level']
        ),
        reverse=True,
    )

    for idx, row in enumerate(ranking, start=1):
        row['rank'] = idx

    return ranking


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        'rank', 'model',
        'macro_f1_slope_per_ratio', 'balanced_accuracy_slope_per_ratio',
        'macro_f1_slope_per_level', 'balanced_accuracy_slope_per_level',
        'macro_f1_aupc', 'balanced_accuracy_aupc',
        'macro_f1_aupc_norm', 'balanced_accuracy_aupc_norm',
        'macro_f1_drop_clean_to_heavy', 'balanced_accuracy_drop_clean_to_heavy'
    ]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_md(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append('# Robustness ranking (physical ratio calibrated)')
    lines.append('')
    lines.append('| Rank | Model | F1 slope/ratio | bal-acc slope/ratio | F1 AUPC(norm) | bal-acc AUPC(norm) | F1(clean->heavy drop) | bal-acc(clean->heavy drop) |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|')

    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['model']} | {to_float(row['macro_f1_slope_per_ratio']):.6f} | {to_float(row['balanced_accuracy_slope_per_ratio']):.6f} | {to_float(row['macro_f1_aupc_norm']):.6f} | {to_float(row['balanced_accuracy_aupc_norm']):.6f} | {to_float(row['macro_f1_drop_clean_to_heavy']):.6f} | {to_float(row['balanced_accuracy_drop_clean_to_heavy']):.6f} |"
        )

    lines.append('')
    lines.append('Note: x-axis uses physical occlusion ratios {clean:0.00, light:0.10, medium:0.20, heavy:0.35}.')
    lines.append('Note: larger slope value (closer to 0) and larger AUPC(norm) indicate better robustness.')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    args = parse_args()

    layout = get_robustness_layout(output_subdir=args.output_subdir, experiment_name=args.experiment_name)
    ensure_layout_dirs(ROOT_DIR, layout)

    agg_csv = args.agg_csv or os.path.join(ROOT_DIR, layout['metrics'], 'summary_keypart_metrics_agg.csv')
    agg_json = args.agg_json or os.path.join(ROOT_DIR, layout['metrics'], 'summary_keypart_metrics_agg.json')

    out_csv = args.out_csv or os.path.join(ROOT_DIR, layout['ranking'], 'robustness_slope_ranking.csv')
    out_json = args.out_json or os.path.join(ROOT_DIR, layout['ranking'], 'robustness_slope_ranking.json')
    out_md = args.out_md or os.path.join(ROOT_DIR, layout['ranking'], 'robustness_slope_ranking.md')

    if os.path.exists(agg_csv):
        rows = read_csv(agg_csv)
        print('Loaded aggregated metrics from csv:', agg_csv)
    elif os.path.exists(agg_json):
        rows = read_json(agg_json)
        print('Loaded aggregated metrics from json:', agg_json)
    else:
        raise FileNotFoundError(
            f'Neither agg csv nor agg json found. Checked: {agg_csv} and {agg_json}'
        )

    ranking = build_model_ranking(rows, args.occlusion_mode)

    if not args.no_csv:
        write_csv(out_csv, ranking)
    write_json(out_json, ranking)
    write_md(out_md, ranking)

    if not args.no_csv:
        print('Saved slope ranking csv to', out_csv)
    print('Saved slope ranking json to', out_json)
    print('Saved slope ranking markdown to', out_md)


if __name__ == '__main__':
    main()
