import argparse
import os
import cv2
import json
import numpy as np
import torch
from torchvision import transforms

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, ROOT_DIR)

from utils.common import IMAGENET_MEAN, IMAGENET_STD
from models.vit_fusion_model import ViTFusionModel, get_attention_map


def generate_fusion_rollout(image_path, weight_path=None, output_path=None, num_classes=None, gamma=1.0, alpha=0.6, rollout_layers=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # infer num_classes
    if num_classes is None:
        try:
            cls_path = os.path.join(ROOT_DIR, 'class_indices.json')
            with open(cls_path, 'r', encoding='utf-8') as f:
                cls = json.load(f)
            num_classes = len(cls)
        except Exception:
            num_classes = 10

    if weight_path is None:
        weight_path = os.path.join(ROOT_DIR, 'weights', 'ablation', 'vit_fusion_abc_best.pth')

    if output_path is None:
        output_path = os.path.join(ROOT_DIR, 'outputs', 'vit_fusion_attention.png')

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {image_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    meta = {}
    meta_path = None
    if weight_path is not None:
        meta_path = weight_path + '.meta.json'
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

    model = ViTFusionModel(
        num_classes=num_classes,
        pretrained=False,
        topk=int(meta.get('topk_patches', 6)),
        attn_rollout_layers=int(meta.get('attn_rollout_layers', 4)),
        use_part_self_attention=bool(meta.get('use_part_self_attention', meta.get('use_local_self_attention', False))),
        part_gate_init=float(meta.get('part_gate_init', meta.get('local_gate_init', 0.0))),
        part_dropout_p=float(meta.get('part_dropout_p', 0.1)),
        attn_temperature=float(meta.get('attn_temperature', 1.0)),
        entropy_penalty=float(meta.get('entropy_penalty', 0.5)),
        enable_module_a=bool(meta.get('enable_module_a', True)),
        enable_module_b=bool(meta.get('enable_module_b', True)),
        enable_module_c=bool(meta.get('enable_module_c', True)),
        split_layer=int(meta.get('split_layer', 8)),
        mid_reweight_temperature=float(meta.get('mid_reweight_temperature', 1.0)),
    )
    state = torch.load(weight_path, map_location=device)
    # support various checkpoint formats
    state_dict = None
    if isinstance(state, dict):
        for key in ('state_dict', 'model_state_dict', 'model', 'state'):
            if key in state and isinstance(state[key], dict):
                state_dict = state[key]
                break
        if state_dict is None:
            # maybe the dict already is a state dict (string keys -> tensors)
            # heuristics: check for e.g. 'backbone' or 'classifier' keys or any tensor value
            sample_keys = list(state.keys())[:5]
            if any(isinstance(state[k], torch.Tensor) for k in sample_keys):
                state_dict = state
    else:
        state_dict = state

    if state_dict is None:
        raise RuntimeError(f'Unable to locate state_dict in checkpoint: {weight_path}')

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # try non-strict load to be tolerant to minor key mismatches
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    # IMPORTANT: for fusion, visualize the rollout_score that actually participates in
    # mid-layer reweighting / pooling (not the plain ViT backbone attention).
    with torch.no_grad():
        if hasattr(model, '_forward_with_attn'):
            old_layers = None
            if rollout_layers is not None:
                old_layers = getattr(model, 'attn_rollout_layers', None)
                model.attn_rollout_layers = int(rollout_layers)
            _final_tokens, rollout_score = model._forward_with_attn(input_tensor)
            if old_layers is not None:
                model.attn_rollout_layers = old_layers
            attn_map = rollout_score
        else:
            selected_rollout_layers = int(meta.get('attn_rollout_layers', 4)) if rollout_layers is None else int(rollout_layers)
            attn_map = get_attention_map(model.global_model, input_tensor, rollout_layers=selected_rollout_layers)

    # same processing as vit_attention_rollout
    mask = attn_map[0]
    n = int(mask.shape[0])
    # compute square size; if not perfect square, pad with zeros to next square
    side = int(round(n ** 0.5))
    if side * side != n:
        side = int(np.ceil(n ** 0.5))
        pad = side * side - n
        m = mask.detach().cpu().numpy()
        if pad > 0:
            m = np.concatenate([m, np.zeros(pad, dtype=m.dtype)])
        mask = m.reshape(side, side)
    else:
        mask = mask.detach().reshape(side, side).cpu().numpy()

    # resize mask to image size
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LINEAR)

    eps = 1e-8
    mn = mask.min()
    mx = mask.max()
    if mx - mn < eps:
        mask = np.zeros_like(mask, dtype=np.float32)
    else:
        mask = (mask - mn) / (mx - mn)

    try:
        mask = np.power(mask, float(gamma))
    except Exception:
        pass

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_float = img.astype(np.float32) / 255.0
    alpha = float(max(0.0, min(1.0, alpha)))
    overlay = alpha * (heatmap.astype(np.float32) / 255.0) + (1.0 - alpha) * img_float
    overlay = overlay / (overlay.max() + eps)
    out = (overlay * 255.0).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, out[:, :, ::-1])
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--weights', default=os.path.join(ROOT_DIR, 'weights', 'vit_fusion_best.pth'))
    parser.add_argument('--output', default=os.path.join(ROOT_DIR, 'outputs', 'visualizations', 'attention', 'vit_fusion_rollout.png'))
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--rollout_layers', type=int, default=None, help='Override the number of last ViT layers averaged for attention rollout')
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    path = generate_fusion_rollout(
        image_path=args.image,
        weight_path=args.weights,
        output_path=args.output,
        num_classes=args.num_classes,
        gamma=args.gamma,
        alpha=args.alpha,
        rollout_layers=args.rollout_layers,
        device=device,
    )
    print('Saved:', path)


if __name__ == '__main__':
    main()
