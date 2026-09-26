import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAPTER3_DIR = os.path.dirname(os.path.abspath(__file__))
if CHAPTER3_DIR not in sys.path:
    sys.path.insert(0, CHAPTER3_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common import (
    build_vit_model,
    DEFAULT_VIT_WEIGHT,
    DEFAULT_SAMPLE_IMAGE,
    ensure_dir,
    get_device,
    image_to_tensor,
    load_class_names,
    load_test_dataset,
    min_max_normalize,
    read_rgb_image,
    resize_attention_map,
    save_figure,
    sample_random_test_images,
    set_paper_style,
    get_rollout_score,
    reshape_attention_vector,
)


def create_rollout_figures(image_path, weight_path, output_dir, rollout_layers=8, alpha=0.55, dpi=300, device=None):
    return create_rollout_dual_output(
        image_path=image_path,
        weight_path=weight_path,
        output_dir=output_dir,
        rollout_layers=rollout_layers,
        alpha=alpha,
        dpi=dpi,
        device=device,
        dual_output=True,
    )


def _enhance_attention_map_for_paper(attention_map):
    """Improve visual readability without changing model inference results."""
    attn = np.asarray(attention_map, dtype=np.float32)
    low = float(np.quantile(attn, 0.65))
    high = float(np.quantile(attn, 0.995))
    if high - low < 1e-8:
        return np.zeros_like(attn, dtype=np.float32)

    attn = np.clip(attn, low, high)
    attn = (attn - low) / (high - low)
    attn = np.power(attn, 0.75)
    attn = cv2.GaussianBlur(attn, (5, 5), 0)
    return min_max_normalize(attn)


def create_rollout_dual_output(
    image_path,
    weight_path,
    output_dir,
    rollout_layers=8,
    alpha=0.55,
    dpi=300,
    device=None,
    dual_output=True,
):
    set_paper_style()
    device = get_device(device)
    num_classes = len(load_class_names())
    model, _, _ = build_vit_model(num_classes=num_classes, weight_path=weight_path, device=device)

    image_rgb = read_rgb_image(image_path, image_size=224)
    input_tensor = image_to_tensor(image_rgb).to(device)

    with torch.no_grad():
        rollout = get_rollout_score(model, input_tensor, rollout_layers=rollout_layers)[0]

    attention_grid = reshape_attention_vector(rollout)
    attention_grid = min_max_normalize(attention_grid)
    attention_map = resize_attention_map(attention_grid, size=224)

    output_dir = ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.imshow(image_rgb)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    save_figure(fig, os.path.join(output_dir, 'original.png'), dpi=dpi)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    im = ax.imshow(attention_map, cmap='jet', vmin=0.0, vmax=1.0)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('注意力强度')
    fig.tight_layout(pad=0.1)
    save_figure(fig, os.path.join(output_dir, 'heatmap.png'), dpi=dpi)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.imshow(image_rgb)
    ax.imshow(attention_map, cmap='jet', alpha=alpha, vmin=0.0, vmax=1.0)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    save_figure(fig, os.path.join(output_dir, 'overlay.png'), dpi=dpi)

    if dual_output:
        paper_attention_map = _enhance_attention_map_for_paper(attention_map)

        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        im = ax.imshow(paper_attention_map, cmap='jet', vmin=0.0, vmax=1.0)
        ax.set_axis_off()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('注意力强度')
        fig.tight_layout(pad=0.1)
        save_figure(fig, os.path.join(output_dir, 'heatmap_paper.png'), dpi=dpi)

        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.imshow(image_rgb)
        ax.imshow(paper_attention_map, cmap='jet', alpha=alpha, vmin=0.0, vmax=1.0)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        save_figure(fig, os.path.join(output_dir, 'overlay_paper.png'), dpi=dpi)

    return {
        'image_path': image_path,
        'output_dir': output_dir,
        'rollout_layers': rollout_layers,
        'dual_output': dual_output,
    }


def create_rollout_layer_comparison(
    image_path,
    weight_path,
    output_dir,
    compare_layers,
    alpha=0.55,
    dpi=300,
    device=None,
    dual_output=True,
):
    results = []
    for layer_count in compare_layers:
        layer_dir = os.path.join(output_dir, f'layer_{int(layer_count)}')
        result = create_rollout_dual_output(
            image_path=image_path,
            weight_path=weight_path,
            output_dir=layer_dir,
            rollout_layers=int(layer_count),
            alpha=alpha,
            dpi=dpi,
            device=device,
            dual_output=dual_output,
        )
        results.append(result)
    return results


def batch_generate(
    dataset_subdir='dataset/ship_fine',
    split='test',
    weight_path=None,
    output_root=None,
    count=8,
    seed=42,
    rollout_layers=8,
    alpha=0.55,
    dpi=300,
    device=None,
    compare_layers=None,
    dual_output=True,
):
    set_paper_style()
    dataset = load_test_dataset(dataset_subdir=dataset_subdir, split=split)
    if output_root is None:
        output_root = os.path.join('outputs', 'paper', 'chapter3', 'rollout', 'batch')

    image_paths = sample_random_test_images(dataset, count=count, seed=seed)
    results = []
    for index, image_path in enumerate(image_paths, start=1):
        sample_dir = os.path.join(output_root, f'sample_{index:02d}')
        if compare_layers:
            results.extend(
                create_rollout_layer_comparison(
                    image_path=image_path,
                    weight_path=weight_path,
                    output_dir=sample_dir,
                    compare_layers=compare_layers,
                    alpha=alpha,
                    dpi=dpi,
                    device=device,
                    dual_output=dual_output,
                )
            )
        else:
            result = create_rollout_dual_output(
                image_path=image_path,
                weight_path=weight_path,
                output_dir=sample_dir,
                rollout_layers=rollout_layers,
                alpha=alpha,
                dpi=dpi,
                device=device,
                dual_output=dual_output,
            )
            results.append(result)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Chapter 3 Experiment 1: Attention Rollout visualization')
    parser.add_argument('--image', default=DEFAULT_SAMPLE_IMAGE, help='single image path from test set')
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine')
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--weights', default=DEFAULT_VIT_WEIGHT, help='vit checkpoint path')
    parser.add_argument('--output_dir', default=os.path.join('outputs', 'paper', 'chapter3', 'rollout'))
    parser.add_argument('--rollout_layers', type=int, default=8)
    parser.add_argument('--compare_layers', nargs='+', type=int, default=None, help='e.g. --compare_layers 4 8 12')
    parser.add_argument('--dual_output_mode', choices=['on', 'off'], default='on', help='on: save raw + paper-enhanced outputs')
    parser.add_argument('--alpha', type=float, default=0.55)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--device', default=None)
    parser.add_argument('--batch_count', type=int, default=0, help='generate multiple random samples when > 0')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    dual_output = args.dual_output_mode == 'on'

    if args.batch_count and args.batch_count > 0:
        results = batch_generate(
            dataset_subdir=args.dataset_subdir,
            split=args.test_split,
            weight_path=args.weights,
            output_root=args.output_dir,
            count=args.batch_count,
            seed=args.seed,
            rollout_layers=args.rollout_layers,
            alpha=args.alpha,
            dpi=args.dpi,
            device=device,
            compare_layers=args.compare_layers,
            dual_output=dual_output,
        )
        print(f'Saved {len(results)} rollout outputs to {args.output_dir}')
        return

    if not args.image:
        dataset = load_test_dataset(dataset_subdir=args.dataset_subdir, split=args.test_split)
        args.image = sample_random_test_images(dataset, count=1, seed=args.seed)[0]

    if args.compare_layers:
        results = create_rollout_layer_comparison(
            image_path=args.image,
            weight_path=args.weights,
            output_dir=args.output_dir,
            compare_layers=args.compare_layers,
            alpha=args.alpha,
            dpi=args.dpi,
            device=device,
            dual_output=dual_output,
        )
        print('Saved rollout layer comparison outputs to:', args.output_dir)
        print('Compared layers:', [r['rollout_layers'] for r in results])
        source_image_path = args.image
    else:
        result = create_rollout_dual_output(
            image_path=args.image,
            weight_path=args.weights,
            output_dir=args.output_dir,
            rollout_layers=args.rollout_layers,
            alpha=args.alpha,
            dpi=args.dpi,
            device=device,
            dual_output=dual_output,
        )
        print('Saved rollout figures to:', result['output_dir'])
        print('Rollout layers:', result['rollout_layers'])
        print('Dual output mode:', result['dual_output'])
        source_image_path = result['image_path']

    print('Source image:', source_image_path)


if __name__ == '__main__':
    main()
