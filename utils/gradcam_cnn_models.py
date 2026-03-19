import argparse
import os
import sys
import json
import csv
import cv2
import numpy as np
import torch
from torchvision import transforms

# ensure project root is importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.resnet_model import create_resnet
from models.vgg_model import create_vgg


def _select_target_layers(model, model_name):
    if model_name == 'vgg':
        conv_layers = [m for m in model.features if isinstance(m, torch.nn.Conv2d)]
        if len(conv_layers) >= 2:
            return [conv_layers[-1], conv_layers[-2]]
        return [conv_layers[-1]]

    layers = [model.layer4[-1]]
    if len(model.layer3) > 0:
        layers.append(model.layer3[-1])
    return layers


def generate_gradcam(image_path, model_name, weight_path=None, out_dir=None, num_classes=None, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 如果未提供 num_classes，则尝试从 class_indices.json 中推断
    if num_classes is None:
        try:
            cls_path = os.path.join(ROOT_DIR, 'class_indices.json')
            with open(cls_path, 'r', encoding='utf-8') as f:
                cls = json.load(f)
            num_classes = len(cls)
        except Exception:
            num_classes = 10

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    rgb_img = img.astype(np.float32) / 255

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)

    if model_name == 'vgg':
        model = create_vgg(num_classes)
    else:
        model = create_resnet(num_classes)

    if weight_path is None:
        weight_path = os.path.join(ROOT_DIR, 'weights', f'{model_name}_best.pth')

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    target_layers = _select_target_layers(model, model_name)

    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    if out_dir is None:
        out_dir = os.path.join(ROOT_DIR, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(out_dir, f'gradcam_{model_name}_{base}.png')
    cv2.imwrite(save_path, visualization[:, :, ::-1])
    return save_path


def generate_gradcam_from_csv(
    csv_path,
    model_name,
    weight_path=None,
    out_dir=None,
    num_classes=None,
    device=None,
    wrong_only=True,
    max_samples=20,
):
    if out_dir is None:
        out_dir = os.path.join(ROOT_DIR, 'outputs', f'gradcam_{model_name}')
    os.makedirs(out_dir, exist_ok=True)

    saved = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_idx = int(row['true_idx'])
            pred_idx = int(row['pred_idx'])

            if wrong_only and true_idx == pred_idx:
                continue

            image_path = row['filepath']
            if not os.path.exists(image_path):
                continue

            try:
                save_path = generate_gradcam(
                    image_path=image_path,
                    model_name=model_name,
                    weight_path=weight_path,
                    out_dir=out_dir,
                    num_classes=num_classes,
                    device=device,
                )
                saved.append(save_path)
                if len(saved) >= max_samples:
                    break
            except Exception as ex:
                print(f'Failed on {image_path}: {ex}')

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None, help='path to image')
    parser.add_argument('--model', choices=['resnet', 'vgg'], default='resnet')
    parser.add_argument('--weights', default=None, help='path to model weights')
    parser.add_argument('--out_dir', default=os.path.join(ROOT_DIR, 'outputs'))
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--csv', default=None, help='preds csv path for batch gradcam export')
    parser.add_argument('--wrong_only', action='store_true', help='only export wrong predictions when --csv is set')
    parser.add_argument('--max_samples', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.csv is not None:
        saved = generate_gradcam_from_csv(
            csv_path=args.csv,
            model_name=args.model,
            weight_path=args.weights,
            out_dir=args.out_dir,
            num_classes=args.num_classes,
            device=device,
            wrong_only=args.wrong_only,
            max_samples=args.max_samples,
        )
        print(f'Saved {len(saved)} files into: {args.out_dir}')
    else:
        if args.image is None:
            raise ValueError('--image is required when --csv is not provided')
        path = generate_gradcam(args.image, args.model, weight_path=args.weights, out_dir=args.out_dir, num_classes=args.num_classes, device=device)
        print('Saved:', path)


if __name__ == '__main__':
    main()