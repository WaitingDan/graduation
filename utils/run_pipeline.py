import argparse
import os
import sys

# ensure project root is importable so `from utils import ...` works when running as script
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# backward-compatible wrapper for unified analysis entrypoint
from utils.analysis import cmd_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=['resnet','vgg','vit'], default=['resnet'])
    parser.add_argument('--visuals', action='store_true', help='Generate GradCAM/attention visuals after evaluation')
    parser.add_argument('--n', type=int, default=3, help='number of examples per group for visuals')
    parser.add_argument('--out_dir', default=os.path.join(ROOT_DIR, 'outputs', 'visuals'))
    args = parser.parse_args()
    cmd_pipeline(args)
    print('Pipeline completed. Outputs in outputs/')


if __name__ == '__main__':
    main()
