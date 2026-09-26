import argparse
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analysis.chapter3.common import get_device, DEFAULT_VIT_WEIGHT
from analysis.chapter3.experiment1_rollout import batch_generate


def parse_args():
    parser = argparse.ArgumentParser(description='Batch rollout runner for Chapter 3 Experiment 1')
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine')
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--weights', default=DEFAULT_VIT_WEIGHT)
    parser.add_argument('--output_root', default=os.path.join('outputs', 'paper', 'chapter3', 'rollout', 'batch'))
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rollout_layers', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.55)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    results = batch_generate(
        dataset_subdir=args.dataset_subdir,
        split=args.test_split,
        weight_path=args.weights,
        output_root=args.output_root,
        count=args.count,
        seed=args.seed,
        rollout_layers=args.rollout_layers,
        alpha=args.alpha,
        dpi=args.dpi,
        device=get_device(args.device),
    )
    print(f'Saved {len(results)} rollout samples to {args.output_root}')


if __name__ == '__main__':
    main()
