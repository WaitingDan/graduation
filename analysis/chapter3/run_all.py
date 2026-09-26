import argparse
import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAPTER3_DIR = os.path.dirname(os.path.abspath(__file__))
if CHAPTER3_DIR not in sys.path:
    sys.path.insert(0, CHAPTER3_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analysis.chapter3.common import get_device, load_test_dataset, resolve_vit_weight, sample_random_test_images, resolve_default_sample_image
from analysis.chapter3.experiment1_rollout import batch_generate, create_rollout_figures
from analysis.chapter3.experiment2_layer_evolution import save_layer_evolution
from analysis.chapter3.experiment3_soft_mask import save_soft_mask


def parse_args():
    parser = argparse.ArgumentParser(description='Run all Chapter 3 analysis experiments')
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine')
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_rollout_count', type=int, default=6)
    parser.add_argument('--device', default=None)
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    weight_path = resolve_vit_weight(args.weights)
    dataset = load_test_dataset(dataset_subdir=args.dataset_subdir, split=args.test_split)
    image_path = resolve_default_sample_image()
    if not os.path.exists(image_path):
        image_path = sample_random_test_images(dataset, count=1, seed=args.seed)[0]

    create_rollout_figures(
        image_path=image_path,
        weight_path=weight_path,
        output_dir=os.path.join('outputs', 'paper', 'chapter3', 'rollout'),
        dpi=args.dpi,
        device=device,
    )
    save_layer_evolution(
        image_path=image_path,
        weight_path=weight_path,
        output_dir=os.path.join('outputs', 'paper', 'chapter3', 'layer_evolution'),
        dpi=args.dpi,
        device=device,
    )
    save_soft_mask(
        image_path=image_path,
        weight_path=weight_path,
        output_dir=os.path.join('outputs', 'paper', 'chapter3', 'soft_mask'),
        dpi=args.dpi,
        device=device,
    )

    if args.batch_rollout_count and args.batch_rollout_count > 0:
        batch_generate(
            dataset_subdir=args.dataset_subdir,
            split=args.test_split,
            weight_path=weight_path,
            output_root=os.path.join('outputs', 'paper', 'chapter3', 'rollout_batch'),
            count=args.batch_rollout_count,
            seed=args.seed,
            dpi=args.dpi,
            device=device,
        )

    print('Finished all Chapter 3 experiments.')


if __name__ == '__main__':
    main()
