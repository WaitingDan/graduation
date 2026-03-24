import argparse
import csv
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Compute robustness degradation slope and ranking for models')
    parser.add_argument('--agg_csv', default='outputs/keypart_experiments/summary_keypart_metrics_agg.csv')
    parser.add_argument('--agg_json', default='outputs/keypart_experiments/summary_keypart_metrics_agg.json')
    parser.add_argument('--occlusion_mode', default='mixed', choices=['block', 'stripe', 'mixed'])
    parser.add_argument('--out_csv', default='outputs/keypart_experiments/robustness_slope_ranking.csv')
    parser.add_argument('--out_json', default='outputs/keypart_experiments/robustness_slope_ranking.json')
    parser.add_argument('--out_md', default='outputs/keypart_experiments/robustness_slope_ranking.md')
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


def pick_metric_by_model(rows, model, scenario, metric_key):
    for row in rows:
        if row.get('model') == model and row.get('scenario') == scenario:
            return to_float(row.get(metric_key, 0.0))
    return None


def build_model_ranking(rows, occlusion_mode):
    level_seq = ['light', 'medium', 'heavy']
    scenarios = [f'{occlusion_mode}_{lv}' for lv in level_seq]

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
            macro_x.append(0)
            macro_y.append(clean_macro)
        if clean_bal is not None:
            bal_x.append(0)
            bal_y.append(clean_bal)

        for i, scenario in enumerate(scenarios, start=1):
            macro_val = pick_metric_by_model(rows, model, scenario, 'macro_f1_mean')
            bal_val = pick_metric_by_model(rows, model, scenario, 'balanced_accuracy_mean')
            if macro_val is not None:
                macro_x.append(i)
                macro_y.append(macro_val)
            if bal_val is not None:
                bal_x.append(i)
                bal_y.append(bal_val)

        macro_slope = linear_slope(macro_x, macro_y)
        bal_slope = linear_slope(bal_x, bal_y)

        heavy_scenario = f'{occlusion_mode}_heavy'
        heavy_macro = pick_metric_by_model(rows, model, heavy_scenario, 'macro_f1_mean')
        heavy_bal = pick_metric_by_model(rows, model, heavy_scenario, 'balanced_accuracy_mean')

        drop_macro = (clean_macro - heavy_macro) if (clean_macro is not None and heavy_macro is not None) else None
        drop_bal = (clean_bal - heavy_bal) if (clean_bal is not None and heavy_bal is not None) else None

        ranking.append({
            'model': model,
            'macro_f1_slope_per_level': macro_slope,
            'balanced_accuracy_slope_per_level': bal_slope,
            'macro_f1_drop_clean_to_heavy': drop_macro,
            'balanced_accuracy_drop_clean_to_heavy': drop_bal,
        })

    ranking.sort(
        key=lambda x: (
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
        'rank', 'model', 'macro_f1_slope_per_level', 'balanced_accuracy_slope_per_level',
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
    lines.append('# 鲁棒性下降斜率排序')
    lines.append('')
    lines.append('| 排名 | 模型 | macro-F1 斜率/级 | bal-acc 斜率/级 | macro-F1(clean→heavy下降) | bal-acc(clean→heavy下降) |')
    lines.append('|---:|---|---:|---:|---:|---:|')

    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['model']} | {to_float(row['macro_f1_slope_per_level']):.6f} | {to_float(row['balanced_accuracy_slope_per_level']):.6f} | {to_float(row['macro_f1_drop_clean_to_heavy']):.6f} | {to_float(row['balanced_accuracy_drop_clean_to_heavy']):.6f} |"
        )

    lines.append('')
    lines.append('说明：斜率越大（越接近0，下降越慢）代表鲁棒性越强。')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    args = parse_args()

    if os.path.exists(args.agg_csv):
        rows = read_csv(args.agg_csv)
        print('Loaded aggregated metrics from csv:', args.agg_csv)
    elif os.path.exists(args.agg_json):
        rows = read_json(args.agg_json)
        print('Loaded aggregated metrics from json:', args.agg_json)
    else:
        raise FileNotFoundError(
            f'Neither agg csv nor agg json found. Checked: {args.agg_csv} and {args.agg_json}'
        )

    ranking = build_model_ranking(rows, args.occlusion_mode)

    if not args.no_csv:
        write_csv(args.out_csv, ranking)
    write_json(args.out_json, ranking)
    write_md(args.out_md, ranking)

    if not args.no_csv:
        print('Saved slope ranking csv to', args.out_csv)
    print('Saved slope ranking json to', args.out_json)
    print('Saved slope ranking markdown to', args.out_md)


if __name__ == '__main__':
    main()
