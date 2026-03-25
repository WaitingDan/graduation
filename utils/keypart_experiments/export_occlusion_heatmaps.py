import argparse
import csv
import json
import math
import os
import random
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.common import OCCLUSION_LEVEL_TO_RATIO
from utils.gradcam_cnn_models import generate_gradcam
from utils.vit_attention_rollout import generate_vit_rollout
from models.vit_fusion_model import create_vit_global_local


def _infer_num_classes():
    try:
        class_path = os.path.join(ROOT_DIR, 'class_indices.json')
        with open(class_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return len(data)
    except Exception:
        return 10


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _load_rgb(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f'Image not found or unreadable: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _apply_block_occlusion(img_rgb, ratio, rng, fill_value=0):
    h, w = img_rgb.shape[:2]
    target_area = max(1, int(h * w * ratio))
    block_h = max(1, int(math.sqrt(target_area)))
    block_w = max(1, int(target_area / max(1, block_h)))
    block_h = min(h, block_h)
    block_w = min(w, block_w)
    y1 = rng.randint(0, max(0, h - block_h))
    x1 = rng.randint(0, max(0, w - block_w))

    out = img_rgb.copy()
    out[y1:y1 + block_h, x1:x1 + block_w, :] = fill_value
    return out


def _apply_stripe_occlusion(img_rgb, ratio, rng, fill_value=0):
    h, w = img_rgb.shape[:2]
    out = img_rgb.copy()
    if rng.random() < 0.5:
        stripe_h = max(1, int(h * ratio))
        stripe_h = min(h, stripe_h)
        y1 = rng.randint(0, max(0, h - stripe_h))
        out[y1:y1 + stripe_h, :, :] = fill_value
    else:
        stripe_w = max(1, int(w * ratio))
        stripe_w = min(w, stripe_w)
        x1 = rng.randint(0, max(0, w - stripe_w))
        out[:, x1:x1 + stripe_w, :] = fill_value
    return out


def generate_occluded_variant(image_rgb, mode, level, seed):
    ratio = OCCLUSION_LEVEL_TO_RATIO.get(level, OCCLUSION_LEVEL_TO_RATIO['light'])
    rng = random.Random(seed)

    if mode == 'none':
        return image_rgb.copy()
    if mode == 'block':
        return _apply_block_occlusion(image_rgb, ratio, rng=rng)
    if mode == 'stripe':
        return _apply_stripe_occlusion(image_rgb, ratio, rng=rng)
    if mode == 'mixed':
        if rng.random() < 0.5:
            return _apply_block_occlusion(image_rgb, ratio, rng=rng)
        return _apply_stripe_occlusion(image_rgb, ratio, rng=rng)
    raise ValueError(f'Unsupported occlusion mode: {mode}')


def save_rgb(path, image_rgb):
    _ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, image_rgb[:, :, ::-1])


def generate_vit_fusion_attention(
    image_path,
    weight_path=None,
    output_path=None,
    num_classes=None,
    alpha=0.6,
    rollout_blur=7,
    rollout_gamma=0.9,
    rollout_mix=0.85,
    device=None,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if num_classes is None:
        num_classes = _infer_num_classes()
    if weight_path is None:
        weight_path = os.path.join(ROOT_DIR, 'weights', 'vit_fusion_best.pth')
    if output_path is None:
        output_path = os.path.join(ROOT_DIR, 'outputs', 'vit_fusion_attention.png')

    image = _load_rgb(image_path)
    image = cv2.resize(image, (224, 224))

    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0).to(device)

    model = create_vit_global_local(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        patch_attention = model._extract_last_attention(model.global_model, input_tensor)

        vit = model.global_model
        tokens = vit._process_input(input_tensor)
        batch_size = tokens.shape[0]
        cls_token = vit.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)

        encoder = vit.encoder
        tokens = encoder.dropout(tokens)

        attentions = []
        for layer in encoder.layers:
            norm_tokens = layer.ln_1(tokens)
            attn_module = layer.self_attention
            qkv = torch.nn.functional.linear(norm_tokens, attn_module.in_proj_weight, attn_module.in_proj_bias)

            q, k, _ = qkv.chunk(3, dim=-1)
            head_dim = attn_module.head_dim
            num_heads = attn_module.num_heads
            scale = 1.0 / math.sqrt(head_dim)

            q = q.view(batch_size, q.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
            k = k.view(batch_size, k.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)

            attn_score = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn_prob = torch.softmax(attn_score, dim=-1)
            attentions.append(attn_prob)

            tokens = layer(tokens)

        eye = torch.eye(attentions[0].size(-1), device=device).unsqueeze(0)
        rollout = eye
        for attn in attentions:
            attn_mean = attn.mean(dim=1)
            attn_mean = attn_mean + eye
            attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
            rollout = torch.matmul(attn_mean, rollout)

        rollout_patch = rollout[:, 0, 1:]

    patch_last = patch_attention[0]
    patch_rollout = rollout_patch[0]
    mix = float(max(0.0, min(1.0, rollout_mix)))
    patch_fused = mix * patch_rollout + (1.0 - mix) * patch_last

    mask = patch_fused.detach().cpu().numpy()
    size = int(math.sqrt(mask.shape[0]))
    if size * size != mask.shape[0]:
        raise RuntimeError(f'Invalid vit_fusion patch attention size: {mask.shape[0]}')
    mask = mask.reshape(size, size)
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LINEAR)

    blur = int(max(0, rollout_blur))
    if blur % 2 == 0:
        blur += 1
    if blur >= 3:
        mask = cv2.GaussianBlur(mask, (blur, blur), sigmaX=0)

    eps = 1e-8
    mn = mask.min()
    mx = mask.max()
    if mx - mn < eps:
        mask = np.zeros_like(mask, dtype=np.float32)
    else:
        mask = (mask - mn) / (mx - mn)

    try:
        mask = np.power(mask, float(rollout_gamma))
    except Exception:
        pass

    heatmap = cv2.applyColorMap(np.uint8(mask * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    image_float = image.astype(np.float32) / 255.0
    heat_float = heatmap.astype(np.float32) / 255.0
    alpha = float(max(0.0, min(1.0, alpha)))
    overlay = alpha * heat_float + (1.0 - alpha) * image_float
    overlay = overlay / (overlay.max() + eps)
    out = (overlay * 255.0).astype(np.uint8)

    _ensure_dir(os.path.dirname(output_path))
    cv2.imwrite(output_path, out[:, :, ::-1])
    return output_path


def _normalize_models(models):
    supported = {'resnet', 'vgg', 'vit', 'vit_fusion'}
    result = []
    for m in models:
        if m not in supported:
            raise ValueError(f'Unsupported model: {m}')
        result.append(m)
    return result


def run_export(
    image_paths,
    models,
    occlusion_mode='mixed',
    occlusion_levels=None,
    include_clean=True,
    out_dir='outputs/keypart_experiments/occlusion_heatmaps',
    seed=42,
    fusion_rollout_blur=7,
    fusion_rollout_gamma=0.9,
    fusion_rollout_mix=0.85,
):
    if occlusion_levels is None:
        occlusion_levels = ['light', 'medium', 'heavy']

    models = _normalize_models(models)
    out_root = os.path.join(ROOT_DIR, out_dir)
    _ensure_dir(out_root)

    records = []
    for image_path in image_paths:
        image_abs = image_path if os.path.isabs(image_path) else os.path.join(ROOT_DIR, image_path)
        image_rgb = _load_rgb(image_abs)
        base = os.path.splitext(os.path.basename(image_abs))[0]

        scenarios = []
        if include_clean:
            scenarios.append(('clean', 'none', 'light'))
        for level in occlusion_levels:
            scenarios.append((f'{occlusion_mode}_{level}', occlusion_mode, level))

        for idx, (scenario_name, mode, level) in enumerate(scenarios):
            scenario_dir = os.path.join(out_root, base, scenario_name)
            _ensure_dir(scenario_dir)

            occ_image = generate_occluded_variant(
                image_rgb=image_rgb,
                mode=mode,
                level=level,
                seed=seed + idx,
            )

            occ_img_path = os.path.join(scenario_dir, f'{base}_{scenario_name}.png')
            save_rgb(occ_img_path, occ_image)

            for model_name in models:
                try:
                    if model_name in ('resnet', 'vgg'):
                        heat_path = generate_gradcam(
                            image_path=occ_img_path,
                            model_name=model_name,
                            weight_path=None,
                            out_dir=scenario_dir,
                            num_classes=_infer_num_classes(),
                        )
                    elif model_name == 'vit':
                        heat_path = generate_vit_rollout(
                            image_path=occ_img_path,
                            weight_path=None,
                            output_path=os.path.join(scenario_dir, f'vit_rollout_{base}_{scenario_name}.png'),
                            num_classes=_infer_num_classes(),
                        )
                    else:
                        heat_path = generate_vit_fusion_attention(
                            image_path=occ_img_path,
                            weight_path=None,
                            output_path=os.path.join(scenario_dir, f'vit_fusion_attention_{base}_{scenario_name}.png'),
                            num_classes=_infer_num_classes(),
                            rollout_blur=fusion_rollout_blur,
                            rollout_gamma=fusion_rollout_gamma,
                            rollout_mix=fusion_rollout_mix,
                        )

                    records.append({
                        'source_image': image_abs,
                        'scenario': scenario_name,
                        'occlusion_mode': mode,
                        'occlusion_level': level,
                        'model': model_name,
                        'occluded_image': occ_img_path,
                        'heatmap_image': heat_path,
                        'status': 'ok',
                        'error': '',
                    })
                except Exception as e:
                    records.append({
                        'source_image': image_abs,
                        'scenario': scenario_name,
                        'occlusion_mode': mode,
                        'occlusion_level': level,
                        'model': model_name,
                        'occluded_image': occ_img_path,
                        'heatmap_image': '',
                        'status': 'failed',
                        'error': str(e),
                    })

    manifest_path = os.path.join(out_root, 'occlusion_heatmaps_manifest.csv')
    with open(manifest_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'source_image', 'scenario', 'occlusion_mode', 'occlusion_level',
                'model', 'occluded_image', 'heatmap_image', 'status', 'error',
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(description='Export occluded images + model heatmaps for paper figures')
    parser.add_argument('--images', nargs='+', required=True, help='one or more image paths (absolute or project-relative)')
    parser.add_argument('--models', nargs='+', choices=['resnet', 'vgg', 'vit', 'vit_fusion'], default=['resnet', 'vgg', 'vit', 'vit_fusion'])
    parser.add_argument('--occlusion_mode', choices=['block', 'stripe', 'mixed'], default='mixed')
    parser.add_argument('--occlusion_levels', nargs='+', choices=['light', 'medium', 'heavy'], default=['light', 'medium', 'heavy'])
    parser.add_argument('--include_clean', action='store_true')
    parser.add_argument('--out_dir', default='outputs/keypart_experiments/occlusion_heatmaps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fusion_rollout_blur', type=int, default=7, help='Gaussian blur kernel size for vit_fusion map (odd number is best)')
    parser.add_argument('--fusion_rollout_gamma', type=float, default=0.9, help='Gamma for vit_fusion map contrast (<1 broadens attention)')
    parser.add_argument('--fusion_rollout_mix', type=float, default=0.85, help='Blend ratio: rollout vs last-layer attention for vit_fusion')
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = run_export(
        image_paths=args.images,
        models=args.models,
        occlusion_mode=args.occlusion_mode,
        occlusion_levels=args.occlusion_levels,
        include_clean=args.include_clean,
        out_dir=args.out_dir,
        seed=args.seed,
        fusion_rollout_blur=args.fusion_rollout_blur,
        fusion_rollout_gamma=args.fusion_rollout_gamma,
        fusion_rollout_mix=args.fusion_rollout_mix,
    )
    print('Saved manifest to', manifest)


if __name__ == '__main__':
    main()
