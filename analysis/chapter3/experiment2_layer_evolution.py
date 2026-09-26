import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


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
    extract_layer_attentions,
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
)


def _cls_vector_to_map(attention_matrix):
    cls_vector = attention_matrix[0, 1:].detach().cpu().numpy().astype('float32')
    side = int(round(cls_vector.shape[0] ** 0.5))
    if side * side != cls_vector.shape[0]:
        side = int((cls_vector.shape[0] ** 0.5) + 0.999999)
        pad = side * side - cls_vector.shape[0]
        if pad > 0:
            cls_vector = np.concatenate([cls_vector, np.zeros(pad, dtype=cls_vector.dtype)])
    grid = cls_vector.reshape(side, side)
    return resize_attention_map(min_max_normalize(grid), 224)


def save_layer_evolution(
    image_path,
    weight_path,
    output_dir,
    layer_ids=(4, 8, 12),
    alpha=0.55,
    dpi=300,
    device=None,
    paper_enhance=True,
    enhance_low_q=0.60,
    enhance_high_q=0.995,
    enhance_gamma=0.80,
    enhance_blur=5,
):
    set_paper_style()
    device = get_device(device)
    num_classes = len(load_class_names())
    model, _, _ = build_vit_model(num_classes=num_classes, weight_path=weight_path, device=device)

    image_rgb = read_rgb_image(image_path, image_size=224)
    input_tensor = image_to_tensor(image_rgb).to(device)

    selected = extract_layer_attentions(model, input_tensor, layer_indices=tuple(layer_ids))
    output_dir = ensure_dir(output_dir)

    for layer_id in layer_ids:
        attn = selected.get(layer_id, None)
        if attn is None:
            raise RuntimeError(f'Failed to capture attention for layer {layer_id}')

        cls_map_raw = _cls_vector_to_map(attn.mean(dim=1)[0])
        cls_map = (
            enhance_attention_map(
                cls_map_raw,
                low_q=enhance_low_q,
                high_q=enhance_high_q,
                gamma=enhance_gamma,
                blur_kernel=enhance_blur,
            )
            if paper_enhance
            else cls_map_raw
        )

        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        im = ax.imshow(cls_map, cmap='jet', vmin=0.0, vmax=1.0)
        ax.set_axis_off()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label('注意力强度')
        fig.tight_layout(pad=0.1)
        save_figure(fig, os.path.join(output_dir, f'layer{layer_id}.png'), dpi=dpi)

        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.imshow(image_rgb)
        ax.imshow(cls_map, cmap='jet', alpha=alpha, vmin=0.0, vmax=1.0)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        save_figure(fig, os.path.join(output_dir, f'overlay_layer{layer_id}.png'), dpi=dpi)

        if paper_enhance:
            fig, ax = plt.subplots(figsize=(4.8, 4.8))
            im = ax.imshow(cls_map_raw, cmap='jet', vmin=0.0, vmax=1.0)
            ax.set_axis_off()
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label('注意力强度')
            fig.tight_layout(pad=0.1)
            save_figure(fig, os.path.join(output_dir, f'layer{layer_id}_raw.png'), dpi=dpi)

            fig, ax = plt.subplots(figsize=(4.8, 4.8))
            ax.imshow(image_rgb)
            ax.imshow(cls_map_raw, cmap='jet', alpha=alpha, vmin=0.0, vmax=1.0)
            ax.set_axis_off()
            fig.tight_layout(pad=0)
            save_figure(fig, os.path.join(output_dir, f'overlay_layer{layer_id}_raw.png'), dpi=dpi)

    return {
        'image_path': image_path,
        'output_dir': output_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Chapter 3 Experiment 2: layer-wise attention evolution')
    parser.add_argument('--image', default=DEFAULT_SAMPLE_IMAGE, help='single image path from test set')
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine')
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--weights', default=DEFAULT_VIT_WEIGHT)
    parser.add_argument('--output_dir', default=os.path.join('outputs', 'paper', 'chapter3', 'layer_evolution'))
    parser.add_argument('--layers', nargs='+', type=int, default=[4, 8, 12])
    parser.add_argument('--alpha', type=float, default=0.55)
    parser.add_argument('--paper_enhance', choices=['on', 'off'], default='on')
    parser.add_argument('--enhance_low_q', type=float, default=0.60)
    parser.add_argument('--enhance_high_q', type=float, default=0.995)
    parser.add_argument('--enhance_gamma', type=float, default=0.80)
    parser.add_argument('--enhance_blur', type=int, default=5)
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

    result = save_layer_evolution(
        image_path=args.image,
        weight_path=args.weights,
        output_dir=args.output_dir,
        layer_ids=tuple(args.layers),
        alpha=args.alpha,
        dpi=args.dpi,
        device=device,
        paper_enhance=args.paper_enhance == 'on',
        enhance_low_q=args.enhance_low_q,
        enhance_high_q=args.enhance_high_q,
        enhance_gamma=args.enhance_gamma,
        enhance_blur=args.enhance_blur,
    )
    print('Saved layer evolution figures to:', result['output_dir'])
    print('Source image:', result['image_path'])


if __name__ == '__main__':
    main()
