import argparse
import csv
import json
import math
import os
import random
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.output_layout import get_robustness_layout, ensure_layout_dirs


def parse_args():
    parser = argparse.ArgumentParser(description='Paired significance test for fusion vs top1 model')
    parser.add_argument('--output_subdir', default=None, help='optional custom output root')
    parser.add_argument('--experiment_name', default='keypart_experiments', help='used when output_subdir is not provided')
    parser.add_argument('--summary_csv', default=None)
    parser.add_argument('--ranking_csv', default=None)
    parser.add_argument('--fusion_model', default='vit_fusion')
    parser.add_argument('--top1_model', default=None, help='if provided, overrides top1 model from ranking csv')
    parser.add_argument('--scenario_filter', nargs='*', default=None, help='optional scenario whitelist')
    parser.add_argument('--out_json', default=None)
    parser.add_argument('--out_md', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_perm', type=int, default=10000, help='number of sign-flip permutations for two-sided p-value')
    return parser.parse_args()


def read_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def pick_top1_model(ranking_rows):
    if not ranking_rows:
        raise RuntimeError('ranking csv is empty')
    ranked = sorted(ranking_rows, key=lambda r: int(r.get('rank', 999999)))
    top1 = ranked[0].get('model')
    if not top1:
        raise RuntimeError('failed to determine top1 model from ranking csv')
    return top1


def paired_diffs(summary_rows, model_a, model_b, metric_key, scenario_filter=None):
    # key by (scenario, seed)
    map_a = {}
    map_b = {}

    for row in summary_rows:
        scenario = row.get('scenario')
        if scenario_filter and scenario not in scenario_filter:
            continue
        seed = str(row.get('seed'))
        key = (scenario, seed)
        model = row.get('model')
        if model == model_a:
            map_a[key] = to_float(row.get(metric_key))
        elif model == model_b:
            map_b[key] = to_float(row.get(metric_key))

    keys = sorted(set(map_a.keys()) & set(map_b.keys()))
    diffs = [map_a[k] - map_b[k] for k in keys]
    return keys, diffs


def permutation_test_sign_flip(diffs, n_perm=10000, seed=42):
    # two-sided sign-flip test on paired differences
    if not diffs:
        return {
            'mean_diff': 0.0,
            'p_value_two_sided': 1.0,
            'n_pairs': 0,
        }

    obs = abs(sum(diffs) / len(diffs))
    n = len(diffs)
    rng = random.Random(seed)

    if n <= 18:
        # exact sign enumeration when feasible
        total = 1 << n
        ge = 0
        for mask in range(total):
            s = 0.0
            for i, d in enumerate(diffs):
                sign = 1.0 if ((mask >> i) & 1) else -1.0
                s += sign * d
            stat = abs(s / n)
            if stat >= obs - 1e-12:
                ge += 1
        p = ge / float(total)
    else:
        ge = 0
        trials = max(1000, int(n_perm))
        for _ in range(trials):
            s = 0.0
            for d in diffs:
                s += d if rng.random() < 0.5 else -d
            stat = abs(s / n)
            if stat >= obs - 1e-12:
                ge += 1
        p = ge / float(trials)

    return {
        'mean_diff': sum(diffs) / len(diffs),
        'p_value_two_sided': p,
        'n_pairs': n,
    }


def ci95_mean(diffs):
    if not diffs:
        return 0.0, 0.0
    n = len(diffs)
    m = sum(diffs) / n
    if n <= 1:
        return m, m
    var = sum((x - m) ** 2 for x in diffs) / (n - 1)
    se = math.sqrt(max(0.0, var)) / math.sqrt(float(n))
    d = 1.96 * se
    return m - d, m + d


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_md(path, result):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append('# Significance test: fusion vs top1')
    lines.append('')
    lines.append(f"- fusion model: {result['fusion_model']}")
    lines.append(f"- top1 model: {result['top1_model']}")
    lines.append(f"- paired keys: scenario+seed, n={result['n_pairs']}")
    lines.append('')
    lines.append('| Metric | mean_diff (fusion-top1) | CI95 low | CI95 high | permutation p(two-sided) |')
    lines.append('|---|---:|---:|---:|---:|')
    for metric in result['metrics']:
        lines.append(
            f"| {metric['metric']} | {metric['mean_diff']:.6f} | {metric['ci95_low']:.6f} | {metric['ci95_high']:.6f} | {metric['p_value_two_sided']:.6f} |"
        )
    lines.append('')
    lines.append('Interpretation: p<0.05 indicates a statistically significant difference under paired sign-flip test.')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    args = parse_args()

    layout = get_robustness_layout(output_subdir=args.output_subdir, experiment_name=args.experiment_name)
    ensure_layout_dirs(ROOT_DIR, layout)

    summary_csv = args.summary_csv or os.path.join(ROOT_DIR, layout['metrics'], 'summary_keypart_metrics.csv')
    ranking_csv = args.ranking_csv or os.path.join(ROOT_DIR, layout['ranking'], 'robustness_slope_ranking.csv')

    out_json = args.out_json or os.path.join(ROOT_DIR, layout['reports'], 'significance_fusion_vs_top1.json')
    out_md = args.out_md or os.path.join(ROOT_DIR, layout['reports'], 'significance_fusion_vs_top1.md')

    summary_rows = read_csv(summary_csv)
    ranking_rows = read_csv(ranking_csv)

    top1 = args.top1_model or pick_top1_model(ranking_rows)
    fusion = args.fusion_model

    metrics = []
    all_keys = None
    for metric_key in ('macro_f1', 'balanced_accuracy'):
        keys, diffs = paired_diffs(
            summary_rows,
            model_a=fusion,
            model_b=top1,
            metric_key=metric_key,
            scenario_filter=set(args.scenario_filter) if args.scenario_filter else None,
        )
        ci_low, ci_high = ci95_mean(diffs)
        perm = permutation_test_sign_flip(diffs, n_perm=args.n_perm, seed=args.seed)
        metrics.append({
            'metric': metric_key,
            'mean_diff': perm['mean_diff'],
            'ci95_low': ci_low,
            'ci95_high': ci_high,
            'p_value_two_sided': perm['p_value_two_sided'],
        })
        if all_keys is None:
            all_keys = keys

    result = {
        'fusion_model': fusion,
        'top1_model': top1,
        'n_pairs': len(all_keys or []),
        'paired_keys': [{'scenario': k[0], 'seed': k[1]} for k in (all_keys or [])],
        'metrics': metrics,
    }

    write_json(out_json, result)
    write_md(out_md, result)

    print('Saved significance json to', out_json)
    print('Saved significance markdown to', out_md)


if __name__ == '__main__':
    main()
