import argparse
import csv
import json
import os
import sys
from statistics import mean, pstdev

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.output_layout import get_robustness_layout, ensure_layout_dirs


ROW_FIELDS = [
    'scenario', 'model', 'seed', 'macro_f1', 'balanced_accuracy',
    'dataset_subdir', 'test_split', 'eval_occlusion_mode', 'eval_occlusion_level',
    'eval_occlusion_p', 'run_output_subdir'
]

AGG_FIELDS = [
    'scenario', 'model', 'runs', 'macro_f1_mean', 'macro_f1_std',
    'balanced_accuracy_mean', 'balanced_accuracy_std'
]


def parse_args():
    parser = argparse.ArgumentParser(description='Run key-part occlusion robustness suite')
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine')
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--models', nargs='+', choices=['resnet', 'vgg', 'vit', 'vit_fusion'], default=['resnet', 'vgg', 'vit', 'vit_fusion'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 3407])
    parser.add_argument('--include_clean', action='store_true', help='include clean test set evaluation')
    parser.add_argument('--occlusion_mode', choices=['block', 'stripe', 'mixed'], default='mixed')
    parser.add_argument('--occlusion_levels', nargs='+', choices=['light', 'medium', 'heavy'], default=['light', 'medium', 'heavy'])
    parser.add_argument('--occlusion_p', type=float, default=1.0)
    parser.add_argument('--output_subdir', default=None, help='optional custom output root')
    parser.add_argument('--experiment_name', default='keypart_experiments', help='used when output_subdir is not provided')
    parser.add_argument('--no_csv', action='store_true', help='do not save summary/agg csv files')
    return parser.parse_args()


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        key = (row['scenario'], row['model'])
        grouped.setdefault(key, []).append(row)

    agg = []
    for (scenario, model), items in grouped.items():
        macro_vals = [float(x['macro_f1']) for x in items]
        bal_vals = [float(x['balanced_accuracy']) for x in items]
        agg.append({
            'scenario': scenario,
            'model': model,
            'runs': len(items),
            'macro_f1_mean': mean(macro_vals),
            'macro_f1_std': pstdev(macro_vals) if len(macro_vals) > 1 else 0.0,
            'balanced_accuracy_mean': mean(bal_vals),
            'balanced_accuracy_std': pstdev(bal_vals) if len(bal_vals) > 1 else 0.0,
        })
    agg.sort(key=lambda x: (x['scenario'], x['model']))
    return agg


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_row(item, scenario_name, seed, args, mode, level, occ_p, run_out_subdir):
    return {
        'scenario': scenario_name,
        'model': item['model'],
        'seed': seed,
        'macro_f1': item['macro_f1'],
        'balanced_accuracy': item['balanced_accuracy'],
        'dataset_subdir': args.dataset_subdir,
        'test_split': args.test_split,
        'eval_occlusion_mode': mode,
        'eval_occlusion_level': level,
        'eval_occlusion_p': occ_p,
        'run_output_subdir': run_out_subdir,
    }


def build_output_paths(root_dir, layout):
    return {
        'summary_csv': os.path.join(root_dir, layout['metrics'], 'summary_keypart_metrics.csv'),
        'summary_json': os.path.join(root_dir, layout['metrics'], 'summary_keypart_metrics.json'),
        'agg_csv': os.path.join(root_dir, layout['metrics'], 'summary_keypart_metrics_agg.csv'),
        'agg_json': os.path.join(root_dir, layout['metrics'], 'summary_keypart_metrics_agg.json'),
    }


def main():
    args = parse_args()
    from utils.evaluate_models import run_evaluation

    layout = get_robustness_layout(output_subdir=args.output_subdir, experiment_name=args.experiment_name)
    ensure_layout_dirs(ROOT_DIR, layout)

    rows = []

    scenarios = []
    if args.include_clean:
        scenarios.append(('clean', 'none', 'light', 0.0))
    for level in args.occlusion_levels:
        scenarios.append((f'{args.occlusion_mode}_{level}', args.occlusion_mode, level, args.occlusion_p))

    for scenario_name, mode, level, occ_p in scenarios:
        for seed in args.seeds:
            run_out_subdir = os.path.join(layout['runs'], scenario_name, f'seed_{seed}')
            metrics = run_evaluation(
                selected_models=args.models,
                dataset_subdir=args.dataset_subdir,
                test_split=args.test_split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                eval_occlusion_mode=mode,
                eval_occlusion_level=level,
                eval_occlusion_p=occ_p,
                output_subdir=run_out_subdir,
                file_suffix='',
                seed=seed,
            )
            for item in metrics:
                rows.append(build_row(item, scenario_name, seed, args, mode, level, occ_p, run_out_subdir))

    output_paths = build_output_paths(ROOT_DIR, layout)
    if not args.no_csv:
        write_csv(output_paths['summary_csv'], rows, ROW_FIELDS)

    with open(output_paths['summary_json'], 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    agg_rows = aggregate_rows(rows)
    if not args.no_csv:
        write_csv(output_paths['agg_csv'], agg_rows, AGG_FIELDS)
    with open(output_paths['agg_json'], 'w', encoding='utf-8') as f:
        json.dump(agg_rows, f, ensure_ascii=False, indent=2)

    if not args.no_csv:
        print('Saved summary csv to', output_paths['summary_csv'])
    print('Saved summary json to', output_paths['summary_json'])
    if not args.no_csv:
        print('Saved aggregated csv to', output_paths['agg_csv'])
    print('Saved aggregated json to', output_paths['agg_json'])
    print('Experiment root:', os.path.join(ROOT_DIR, layout['root']))


if __name__ == '__main__':
    main()
