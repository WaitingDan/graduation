import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.evaluate_models import run_evaluation
from utils.generate_visuals import run_visuals


def cmd_eval(args):
    run_evaluation(
        selected_models=args.models,
        dataset_subdir=args.dataset_subdir,
        test_split=args.test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_occlusion_mode=args.eval_occlusion_mode,
        eval_occlusion_level=args.eval_occlusion_level,
        eval_occlusion_p=args.eval_occlusion_p,
        output_subdir=args.output_subdir,
        file_suffix=args.file_suffix,
        seed=args.seed,
    )


def cmd_visuals(args):
    run_visuals(model=args.model, csv_path=args.csv, n=args.n, out_dir=args.out_dir, weights=args.weights)


def cmd_pipeline(args):
    models = args.models if args.models else ['resnet']
    run_evaluation(
        selected_models=models,
        dataset_subdir=args.dataset_subdir,
        test_split=args.test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_occlusion_mode=args.eval_occlusion_mode,
        eval_occlusion_level=args.eval_occlusion_level,
        eval_occlusion_p=args.eval_occlusion_p,
        output_subdir=args.output_subdir,
        file_suffix=args.file_suffix,
        seed=args.seed,
    )
    if args.visuals:
        for m in models:
            csv_path = os.path.join(ROOT_DIR, args.output_subdir, f'preds_{m}{args.file_suffix}.csv')
            if not os.path.exists(csv_path):
                print(f'Skip {m}: csv not found -> {csv_path}')
                continue
            run_visuals(model=m, csv_path=csv_path, n=args.n, out_dir=args.out_dir)


def build_parser():
    parser = argparse.ArgumentParser(description='Unified analysis entrypoint (eval/visuals/pipeline)')
    sub = parser.add_subparsers(dest='command', required=True)

    p_eval = sub.add_parser('eval', help='Run evaluation and save preds/report/confusion matrix')
    p_eval.add_argument('--models', nargs='+', choices=['resnet', 'vgg', 'vit', 'vit_fusion'], default=None)
    p_eval.add_argument('--dataset_subdir', default='dataset/ship_fine')
    p_eval.add_argument('--test_split', default='test')
    p_eval.add_argument('--batch_size', type=int, default=32)
    p_eval.add_argument('--num_workers', type=int, default=2)
    p_eval.add_argument('--eval_occlusion_mode', choices=['none', 'block', 'stripe', 'mixed'], default='none')
    p_eval.add_argument('--eval_occlusion_level', choices=['light', 'medium', 'heavy'], default='light')
    p_eval.add_argument('--eval_occlusion_p', type=float, default=0.0)
    p_eval.add_argument('--output_subdir', default='outputs/evaluation/default_eval')
    p_eval.add_argument('--file_suffix', default='')
    p_eval.add_argument('--seed', type=int, default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_vis = sub.add_parser('visuals', help='Generate visuals from preds csv')
    p_vis.add_argument('--model', choices=['resnet', 'vgg', 'vit', 'vit_fusion'], required=True)
    p_vis.add_argument('--csv', required=True)
    p_vis.add_argument('--n', type=int, default=3)
    p_vis.add_argument('--out_dir', default=os.path.join(ROOT_DIR, 'outputs', 'visualizations', 'attention', 'default_eval'))
    p_vis.add_argument('--weights', default=None, help='Optional model weights for vit / vit_fusion visuals')
    p_vis.set_defaults(func=cmd_visuals)

    p_pipe = sub.add_parser('pipeline', help='Run eval then optional visuals')
    p_pipe.add_argument('--models', nargs='+', choices=['resnet', 'vgg', 'vit', 'vit_fusion'], default=['resnet'])
    p_pipe.add_argument('--visuals', action='store_true')
    p_pipe.add_argument('--n', type=int, default=3)
    p_pipe.add_argument('--out_dir', default=os.path.join(ROOT_DIR, 'outputs', 'visualizations', 'attention', 'default_eval'))
    p_pipe.add_argument('--dataset_subdir', default='dataset/ship_fine')
    p_pipe.add_argument('--test_split', default='test')
    p_pipe.add_argument('--batch_size', type=int, default=32)
    p_pipe.add_argument('--num_workers', type=int, default=2)
    p_pipe.add_argument('--eval_occlusion_mode', choices=['none', 'block', 'stripe', 'mixed'], default='none')
    p_pipe.add_argument('--eval_occlusion_level', choices=['light', 'medium', 'heavy'], default='light')
    p_pipe.add_argument('--eval_occlusion_p', type=float, default=0.0)
    p_pipe.add_argument('--output_subdir', default='outputs/evaluation/default_eval')
    p_pipe.add_argument('--file_suffix', default='')
    p_pipe.add_argument('--seed', type=int, default=None)
    p_pipe.set_defaults(func=cmd_pipeline)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
