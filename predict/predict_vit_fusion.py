import argparse
import json
import os
import sys

import torch
from PIL import Image


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.vit_fusion_model import create_vit_global_local
from utils.common import get_device, build_default_transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=os.path.join(ROOT_DIR, 'test_images', 'test_ship_02.jpg'))
    parser.add_argument('--weights', default=os.path.join(ROOT_DIR, 'weights', 'vit_fusion_best.pth'))
    parser.add_argument('--class_indices', default=os.path.join(ROOT_DIR, 'class_indices.json'))
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=112)
    parser.add_argument('--topk_patches', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--pretrained_init', action='store_true', help='use torchvision official pretrained weights before loading checkpoint')
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def load_class_names(class_indices_path):
    with open(class_indices_path, 'r', encoding='utf-8') as f:
        class_dict = json.load(f)
    return [class_dict[str(i)] for i in range(len(class_dict))]


def main():
    args = parse_args()

    class_names = load_class_names(args.class_indices)
    num_classes = len(class_names)

    device = torch.device(args.device) if args.device else get_device()
    print('Using device:', device)

    model = create_vit_global_local(
        num_classes=num_classes,
        pretrained=args.pretrained_init,
        crop_size=args.crop_size,
        out_size=args.img_size,
        topk_patches=args.topk_patches,
        dropout=args.dropout,
    )
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    transform = build_default_transforms(img_size=args.img_size)['val']

    image = Image.open(args.image).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        if isinstance(outputs, (tuple, list)):
            logits = outputs[-1]
        else:
            logits = outputs
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        pred_prob = float(probs[0, pred_idx].item())

    print('预测类别索引:', pred_idx)
    print('预测类别名称:', class_names[pred_idx])
    print('预测置信度:', f'{pred_prob:.4f}')


if __name__ == '__main__':
    main()