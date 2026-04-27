#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.output_layout import get_robustness_layout, ensure_layout_dirs


def run_cmd(cmd, cwd=ROOT_DIR):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description='One-click training + robustness evaluation pipeline')
    parser.add_argument('--train_models', nargs='*', choices=['resnet', 'vgg', 'vit', 'vit_fusion', 'vit_two_road'], default=[],
                            help='models to train before robustness evaluation (default: none)')
    parser.add_argument('--eval_models', nargs='+', choices=['resnet', 'vgg', 'vit', 'vit_fusion', 'vit_two_road'],
                            default=['resnet', 'vgg', 'vit', 'vit_fusion', 'vit_two_road'],
                        help='models included in occlusion robustness suite')
    parser.add_argument('--output_subdir', default=None, help='optional custom output root')
    parser.add_argument('--experiment_name', default='keypart_experiments', help='used when output_subdir is not provided')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 3407])
    clean_group = parser.add_mutually_exclusive_group()
    clean_group.add_argument('--include_clean', dest='include_clean', action='store_true', help='include clean scenario in suite (default: enabled)')
    clean_group.add_argument('--no_include_clean', dest='include_clean', action='store_false', help='disable clean scenario in suite')
    parser.set_defaults(include_clean=True)
    parser.add_argument('--occlusion_mode', choices=['block', 'stripe', 'mixed'], default='mixed')
    parser.add_argument('--occlusion_levels', nargs='+', choices=['light', 'medium', 'heavy'], default=['light', 'medium', 'heavy'])
    parser.add_argument('--occlusion_p', type=float, default=1.0)
    parser.add_argument('--quick', action='store_true',
                        help='quick benchmark: seeds=[42], occlusion_levels=[heavy], include_clean=True')
    parser.add_argument('--train_label_smoothing', type=float, default=0.1,
                        help='label smoothing used when --train_models is set (default: 0.1)')
    parser.add_argument('--train_occlusion_mode', choices=['none', 'block', 'stripe', 'mixed'], default='mixed',
                        help='train-time occlusion mode used when --train_models is set (default: mixed)')
    parser.add_argument('--train_occlusion_level', choices=['light', 'medium', 'heavy'], default='medium',
                        help='train-time occlusion level used when --train_models is set (default: medium)')
    parser.add_argument('--train_occlusion_p', type=float, default=0.4,
                        help='probability of train-time occlusion used when --train_models is set (default: 0.4)')
    parser.add_argument('--skip_analyze', action='store_true')
    parser.add_argument('--skip_significance', action='store_true')
    parser.add_argument('--skip_visualize', action='store_true')
    return parser.parse_args()


def run_training(train_models, args):
    robust_train_args = [
        '--label_smoothing', str(args.train_label_smoothing),
        '--train_occlusion_mode', args.train_occlusion_mode,
        '--train_occlusion_level', args.train_occlusion_level,
        '--train_occlusion_p', str(args.train_occlusion_p),
    ]

    for model_name in train_models:
        if model_name == 'resnet':
            run_cmd([sys.executable, 'train/train_resnet.py', *robust_train_args])
        elif model_name == 'vgg':
            run_cmd([sys.executable, 'train/train_vgg.py', *robust_train_args])
        elif model_name == 'vit':
            run_cmd([sys.executable, 'train/train_vit.py', *robust_train_args])
        elif model_name == 'vit_fusion':
            run_cmd([sys.executable, 'train/train_vit_fusion.py', *robust_train_args])


def main():
    args = parse_args()

    seeds = args.seeds
    occlusion_levels = args.occlusion_levels
    include_clean = args.include_clean

    if args.quick:
        seeds = [42]
        occlusion_levels = ['heavy']
        include_clean = True

    layout = get_robustness_layout(output_subdir=args.output_subdir, experiment_name=args.experiment_name)
    ensure_layout_dirs(ROOT_DIR, layout)

    if args.train_models:
        run_training(args.train_models, args)

    suite_cmd = [
        sys.executable,
        'utils/robustness/occlusion_suite.py',
        '--output_subdir', layout['root'],
        '--models', *args.eval_models,
        '--seeds', *[str(seed) for seed in seeds],
        '--occlusion_mode', args.occlusion_mode,
        '--occlusion_levels', *occlusion_levels,
        '--occlusion_p', str(args.occlusion_p),
    ]
    if include_clean:
        suite_cmd.append('--include_clean')
    run_cmd(suite_cmd)

    if not args.skip_analyze:
        run_cmd([
            sys.executable,
            'utils/robustness/slope_ranking.py',
            '--output_subdir', layout['root'],
            '--occlusion_mode', args.occlusion_mode,
        ])

    if (not args.skip_analyze) and (not args.skip_significance):
        run_cmd([
            sys.executable,
            'utils/robustness/significance_test.py',
            '--output_subdir', layout['root'],
            '--experiment_name', args.experiment_name,
        ])

    if not args.skip_visualize:
        run_cmd([
            sys.executable,
            'utils/visualization/robustness_plots.py',
            '--output_subdir', layout['root'],
        ])

    print('\nPipeline finished.')
    print('Experiment root:', os.path.join(ROOT_DIR, layout['root']))
    print('Metrics:', os.path.join(ROOT_DIR, layout['metrics']))
    print('Ranking:', os.path.join(ROOT_DIR, layout['ranking']))
    print('Plots:', os.path.join(ROOT_DIR, layout['plots']))


if __name__ == '__main__':
    main()
