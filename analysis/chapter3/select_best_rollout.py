import argparse
import csv
import math
import os
import shutil
import sys

import numpy as np
import torch


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAPTER3_DIR = os.path.dirname(os.path.abspath(__file__))
if CHAPTER3_DIR not in sys.path:
    sys.path.insert(0, CHAPTER3_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common import (
    DEFAULT_VIT_WEIGHT,
    build_vit_model,
    get_device,
    get_rollout_score,
    image_to_tensor,
    load_class_names,
    load_test_dataset,
    min_max_normalize,
    read_rgb_image,
    reshape_attention_vector,
    sample_random_test_images,
)


def score_attention(attn):
    """Score attention quality for paper visualization selection.

    Higher score means more concentrated and cleaner focus with less edge bias.
    """
    v = attn.reshape(-1).astype(np.float64)
    mean_v = float(v.mean()) + 1e-8

    q90 = float(np.quantile(v, 0.90))
    top = v[v >= q90]
    top_mean = float(top.mean()) if top.size else mean_v
    peak = min((top_mean / mean_v) / 6.0, 1.0)

    p = v / (v.sum() + 1e-12)
    ent = -float(np.sum(p * np.log(p + 1e-12))) / math.log(len(p))
    compact = 1.0 - ent

    h, w = attn.shape
    m = max(1, int(min(h, w) * 0.12))
    edge_mask = np.zeros_like(attn, dtype=bool)
    edge_mask[:m, :] = True
    edge_mask[-m:, :] = True
    edge_mask[:, :m] = True
    edge_mask[:, -m:] = True
    edge_mean = float(attn[edge_mask].mean()) + 1e-8
    edge_ratio = edge_mean / (float(attn.mean()) + 1e-8)
    edge_clean = max(0.0, min(1.0, 1.0 - min(edge_ratio, 1.0)))

    score = 0.55 * peak + 0.30 * compact + 0.15 * edge_clean
    return score, peak, compact, edge_clean


def parse_args():
    parser = argparse.ArgumentParser(description='Select best Experiment-1 rollout samples for paper figures')
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine')
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--weights', default=DEFAULT_VIT_WEIGHT)
    parser.add_argument('--batch_dir', default=os.path.join('outputs', 'paper', 'chapter3', 'rollout', 'batch40'))
    parser.add_argument('--select_dir', default=os.path.join('outputs', 'paper', 'chapter3', 'rollout', 'paper_top6'))
    parser.add_argument('--count', type=int, default=40, help='must match the batch sampling count')
    parser.add_argument('--seed', type=int, default=42, help='must match the batch sampling seed')
    parser.add_argument('--topk', type=int, default=6)
    parser.add_argument('--rollout_layers', type=int, default=4)
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.select_dir, exist_ok=True)

    dataset = load_test_dataset(dataset_subdir=args.dataset_subdir, split=args.test_split)
    image_paths = sample_random_test_images(dataset, count=args.count, seed=args.seed)

    device = get_device(args.device)
    num_classes = len(load_class_names())
    model, _, _ = build_vit_model(num_classes=num_classes, weight_path=args.weights, device=device)

    rows = []
    for idx, image_path in enumerate(image_paths, start=1):
        sample_dir = os.path.join(args.batch_dir, f'sample_{idx:02d}')
        if not os.path.isdir(sample_dir):
            continue

        image_rgb = read_rgb_image(image_path, image_size=224)
        x = image_to_tensor(image_rgb).to(device)
        with torch.no_grad():
            rollout = get_rollout_score(model, x, rollout_layers=args.rollout_layers)[0]

        attn = min_max_normalize(reshape_attention_vector(rollout))
        score, peak, compact, edge_clean = score_attention(attn)

        rows.append({
            'sample_id': idx,
            'sample_dir': f'sample_{idx:02d}',
            'image_path': image_path,
            'score': score,
            'peak': peak,
            'compact': compact,
            'edge_clean': edge_clean,
        })

    if not rows:
        raise RuntimeError(f'No valid sample folders found under: {args.batch_dir}')

    rows.sort(key=lambda x: x['score'], reverse=True)

    for name in os.listdir(args.select_dir):
        path = os.path.join(args.select_dir, name)
        if os.path.isdir(path) and name.startswith('rank_'):
            shutil.rmtree(path)

    topk = min(args.topk, len(rows))
    for rank, row in enumerate(rows[:topk], start=1):
        src = os.path.join(args.batch_dir, row['sample_dir'])
        dst = os.path.join(args.select_dir, f'rank_{rank:02d}_{row["sample_dir"]}')
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    csv_path = os.path.join(args.select_dir, 'ranking.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['rank', 'sample_id', 'sample_dir', 'score', 'peak', 'compact', 'edge_clean', 'image_path'],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({
                'rank': rank,
                'sample_id': row['sample_id'],
                'sample_dir': row['sample_dir'],
                'score': f"{row['score']:.6f}",
                'peak': f"{row['peak']:.6f}",
                'compact': f"{row['compact']:.6f}",
                'edge_clean': f"{row['edge_clean']:.6f}",
                'image_path': row['image_path'],
            })

    print(f'Wrote ranking: {csv_path}')
    print(f'Selected top {topk} samples to: {args.select_dir}')
    for rank, row in enumerate(rows[:topk], start=1):
        print(f"{rank}. {row['sample_dir']} | score={row['score']:.4f} | image={row['image_path']}")


if __name__ == '__main__':
    main()
