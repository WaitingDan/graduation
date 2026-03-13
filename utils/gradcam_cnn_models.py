import argparse
import os
import sys
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


def generate_gradcam(image_path, model_name, weight_path=None, out_dir=None, num_classes=10, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if model_name == 'vgg':
        target_layer = model.features[-1]
    else:
        target_layer = model.layer4[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    if out_dir is None:
        out_dir = os.path.join(ROOT_DIR, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(out_dir, f'gradcam_{model_name}_{base}.png')
    cv2.imwrite(save_path, visualization[:, :, ::-1])
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='path to image')
    parser.add_argument('--model', choices=['resnet', 'vgg'], default='resnet')
    parser.add_argument('--weights', default=None, help='path to model weights')
    parser.add_argument('--out_dir', default=os.path.join(ROOT_DIR, 'outputs'))
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    path = generate_gradcam(args.image, args.model, weight_path=args.weights, out_dir=args.out_dir, num_classes=args.num_classes, device=device)
    print('Saved:', path)


if __name__ == '__main__':
    main()