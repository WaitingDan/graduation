import json
import os

import torch
from torchvision import transforms


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_default_transforms():
    return {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }


def write_class_indices(class_names, output_path):
    class_dict = {str(i): name for i, name in enumerate(class_names)}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(class_dict, f, indent=4, ensure_ascii=False)
