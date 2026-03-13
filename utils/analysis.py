import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.evaluate_models import run_evaluation
from utils.generate_visuals import run_visuals


def cmd_eval(args):
    run_evaluation(selected_models=args.models)


def cmd_visuals(args):
    run_visuals(model=args.model, csv_path=args.csv, n=args.n, out_dir=args.out_dir)


def cmd_pipeline(args):
    models = args.models if args.models else ['resnet']
    run_evaluation(selected_models=models)
    if args.visuals:
        for m in models:
            csv_path = os.path.join(ROOT_DIR, 'outputs', f'preds_{m}.csv')
            if not os.path.exists(csv_path):
                print(f'Skip {m}: csv not found -> {csv_path}')
                continue
            run_visuals(model=m, csv_path=csv_path, n=args.n, out_dir=args.out_dir)


def build_parser():
    parser = argparse.ArgumentParser(description='Unified analysis entrypoint (eval/visuals/pipeline)')
    sub = parser.add_subparsers(dest='command', required=True)

    p_eval = sub.add_parser('eval', help='Run evaluation and save preds/report/confusion matrix')
    p_eval.add_argument('--models', nargs='+', choices=['resnet', 'vgg', 'vit'], default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_vis = sub.add_parser('visuals', help='Generate visuals from preds csv')
    p_vis.add_argument('--model', choices=['resnet', 'vgg', 'vit'], required=True)
    p_vis.add_argument('--csv', required=True)
    p_vis.add_argument('--n', type=int, default=3)
    p_vis.add_argument('--out_dir', default=os.path.join(ROOT_DIR, 'outputs', 'visuals'))
    p_vis.set_defaults(func=cmd_visuals)

    p_pipe = sub.add_parser('pipeline', help='Run eval then optional visuals')
    p_pipe.add_argument('--models', nargs='+', choices=['resnet', 'vgg', 'vit'], default=['resnet'])
    p_pipe.add_argument('--visuals', action='store_true')
    p_pipe.add_argument('--n', type=int, default=3)
    p_pipe.add_argument('--out_dir', default=os.path.join(ROOT_DIR, 'outputs', 'visuals'))
    p_pipe.set_defaults(func=cmd_pipeline)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
