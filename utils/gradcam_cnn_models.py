import torch
import cv2
import numpy as np
import os
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys

# make project root importable before importing local models
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.resnet_model import create_resnet
from models.vgg_model import create_vgg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image_path = os.path.join(ROOT_DIR, "test_images/test_ship_02.jpg")

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))

rgb_img = img.astype(np.float32) / 255

transform = transforms.Compose([
    transforms.ToTensor()
])

input_tensor = transform(img).unsqueeze(0).to(device)

num_classes = 10

models = {
    "vgg": {
        "model": create_vgg(num_classes),
        "weight": os.path.join(ROOT_DIR, "weights/vgg_best.pth")
    },
    "resnet": {
        "model": create_resnet(num_classes),
        "weight": os.path.join(ROOT_DIR, "weights/resnet_best.pth")
    }
}

for name in models:

    print("Generating GradCAM for", name)

    model = models[name]["model"]
    weight = models[name]["weight"]

    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    model.eval()

    if name == "vgg":
        target_layer = model.features[-1]

    else:
        target_layer = model.layer4[-1]

    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
    )

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    save_path = os.path.join(ROOT_DIR, f"gradcam_{name}.png")

    cv2.imwrite(save_path, visualization[:, :, ::-1])

    print("Saved:", save_path)