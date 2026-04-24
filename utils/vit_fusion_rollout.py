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

from models.vit_fusion_model import ViTFusionModel, get_attention_map


def generate_fusion_rollout(image_path, weight_path=None, output_path=None, num_classes=None, gamma=1.0, alpha=0.6, device=None):
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
        weight_path = os.path.join(ROOT_DIR, 'weights', 'vit_fusion_best.pth')

    if output_path is None:
        output_path = os.path.join(ROOT_DIR, 'outputs', 'vit_fusion_attention.png')

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {image_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0).to(device)

    model = ViTFusionModel(num_classes=num_classes, pretrained=False)
    state = torch.load(weight_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # try loading by matching keys (support common save formats)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            # attempt strict=False to load partial weights
            model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    # get attention map from global_model
    attn_map = get_attention_map(model.global_model, input_tensor)

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
        device=device,
    )
    print('Saved:', path)


if __name__ == '__main__':
    main()
