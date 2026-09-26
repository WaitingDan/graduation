import argparse
import os
import sys

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
    enhance_attention_map,
    ensure_dir,
    get_device,
    get_rollout_score,
    image_to_tensor,
    load_class_names,
    load_test_dataset,
    min_max_normalize,
    read_rgb_image,
    resize_attention_map,
    save_figure,
    sample_random_test_images,
    set_paper_style,
    reshape_attention_vector,
)


def _make_soft_mask(attention_map, gamma=1.8, floor=0.08):
    mask = np.power(np.asarray(attention_map, dtype=np.float32), float(gamma)).astype(np.float32)
    mask = min_max_normalize(mask)
    floor = float(max(0.0, min(0.9, floor)))
    return floor + (1.0 - floor) * mask


def save_soft_mask(
    image_path,
    weight_path,
    output_dir,
    gamma=1.8,
    rollout_layers=8,
    dpi=300,
    device=None,
    paper_enhance=True,
    enhance_low_q=0.60,
    enhance_high_q=0.995,
    enhance_gamma=0.80,
    enhance_blur=5,
    mask_floor=0.08,
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
    attention_map_raw = resize_attention_map(attention_grid, size=224)
    attention_map = (
        enhance_attention_map(
            attention_map_raw,
            low_q=enhance_low_q,
            high_q=enhance_high_q,
            gamma=enhance_gamma,
            blur_kernel=enhance_blur,
        )
        if paper_enhance
        else attention_map_raw
    )
    soft_mask = _make_soft_mask(attention_map, gamma=gamma, floor=mask_floor)
    masked_image = (image_rgb.astype(np.float32) * soft_mask[..., None]).clip(0, 255).astype(np.uint8)

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
    save_figure(fig, os.path.join(output_dir, 'attention.png'), dpi=dpi)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    im = ax.imshow(soft_mask, cmap='gray', vmin=0.0, vmax=1.0)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('软掩码')
    fig.tight_layout(pad=0.1)
    save_figure(fig, os.path.join(output_dir, 'soft_mask.png'), dpi=dpi)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.imshow(masked_image)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    save_figure(fig, os.path.join(output_dir, 'masked_image.png'), dpi=dpi)

    if paper_enhance:
        raw_soft_mask = _make_soft_mask(attention_map_raw, gamma=gamma, floor=mask_floor)
        raw_masked_image = (image_rgb.astype(np.float32) * raw_soft_mask[..., None]).clip(0, 255).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        im = ax.imshow(attention_map_raw, cmap='jet', vmin=0.0, vmax=1.0)
        ax.set_axis_off()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('注意力强度')
        fig.tight_layout(pad=0.1)
        save_figure(fig, os.path.join(output_dir, 'attention_raw.png'), dpi=dpi)

        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        im = ax.imshow(raw_soft_mask, cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_axis_off()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('软掩码')
        fig.tight_layout(pad=0.1)
        save_figure(fig, os.path.join(output_dir, 'soft_mask_raw.png'), dpi=dpi)

        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.imshow(raw_masked_image)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        save_figure(fig, os.path.join(output_dir, 'masked_image_raw.png'), dpi=dpi)

    return {
        'image_path': image_path,
        'output_dir': output_dir,
        'gamma': gamma,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Chapter 3 Experiment 3: soft mask generation')
    parser.add_argument('--image', default=DEFAULT_SAMPLE_IMAGE, help='single image path from test set')
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine')
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--weights', default=DEFAULT_VIT_WEIGHT)
    parser.add_argument('--output_dir', default=os.path.join('outputs', 'paper', 'chapter3', 'soft_mask'))
    parser.add_argument('--gamma', type=float, default=1.8)
    parser.add_argument('--rollout_layers', type=int, default=8)
    parser.add_argument('--paper_enhance', choices=['on', 'off'], default='on')
    parser.add_argument('--enhance_low_q', type=float, default=0.60)
    parser.add_argument('--enhance_high_q', type=float, default=0.995)
    parser.add_argument('--enhance_gamma', type=float, default=0.80)
    parser.add_argument('--enhance_blur', type=int, default=5)
    parser.add_argument('--mask_floor', type=float, default=0.08)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    if not args.image:
        dataset = load_test_dataset(dataset_subdir=args.dataset_subdir, split=args.test_split)
        args.image = sample_random_test_images(dataset, count=1, seed=args.seed)[0]

    result = save_soft_mask(
        image_path=args.image,
        weight_path=args.weights,
        output_dir=args.output_dir,
        gamma=args.gamma,
        rollout_layers=args.rollout_layers,
        dpi=args.dpi,
        device=device,
        paper_enhance=args.paper_enhance == 'on',
        enhance_low_q=args.enhance_low_q,
        enhance_high_q=args.enhance_high_q,
        enhance_gamma=args.enhance_gamma,
        enhance_blur=args.enhance_blur,
        mask_floor=args.mask_floor,
    )
    print('Saved soft mask figures to:', result['output_dir'])
    print('Source image:', result['image_path'])


if __name__ == '__main__':
    main()
