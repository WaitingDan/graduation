import json

import torch
from torchvision import transforms
import random


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


OCCLUSION_LEVEL_TO_RATIO = {
    'light': 0.10,
    'medium': 0.20,
    'heavy': 0.35,
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StripeOcclusion:
    def __init__(self, p=0.0, level='light', fill=0.0):
        self.p = float(max(0.0, min(1.0, p)))
        self.level = level
        self.fill = fill

    def __call__(self, tensor):
        if self.p <= 0 or random.random() > self.p:
            return tensor

        ratio = OCCLUSION_LEVEL_TO_RATIO.get(self.level, OCCLUSION_LEVEL_TO_RATIO['light'])
        _, height, width = tensor.shape

        if random.random() < 0.5:
            stripe_h = max(1, int(height * ratio))
            y1 = random.randint(0, max(0, height - stripe_h))
            tensor[:, y1:y1 + stripe_h, :] = self.fill
        else:
            stripe_w = max(1, int(width * ratio))
            x1 = random.randint(0, max(0, width - stripe_w))
            tensor[:, :, x1:x1 + stripe_w] = self.fill
        return tensor


class MixedOcclusion:
    def __init__(self, p=0.0, level='light', fill=0.0):
        self.p = float(max(0.0, min(1.0, p)))
        self.level = level
        ratio = OCCLUSION_LEVEL_TO_RATIO.get(level, OCCLUSION_LEVEL_TO_RATIO['light'])
        min_scale = max(0.02, ratio * 0.7)
        max_scale = min(0.60, ratio * 1.3)
        if min_scale > max_scale:
            min_scale, max_scale = max_scale, min_scale
        self.block = transforms.RandomErasing(
            p=1.0,
            scale=(min_scale, max_scale),
            ratio=(0.6, 1.7),
            value=fill,
        )
        self.stripe = StripeOcclusion(p=1.0, level=level, fill=fill)

    def __call__(self, tensor):
        if self.p <= 0 or random.random() > self.p:
            return tensor
        if random.random() < 0.5:
            return self.block(tensor)
        return self.stripe(tensor)


def build_occlusion_transform(mode='none', level='light', p=0.0, fill=0.0):
    mode = (mode or 'none').lower()
    if mode == 'none' or p <= 0:
        return None

    ratio = OCCLUSION_LEVEL_TO_RATIO.get(level, OCCLUSION_LEVEL_TO_RATIO['light'])

    if mode == 'block':
        min_scale = max(0.02, ratio * 0.7)
        max_scale = min(0.60, ratio * 1.3)
        if min_scale > max_scale:
            min_scale, max_scale = max_scale, min_scale
        return transforms.RandomErasing(
            p=float(max(0.0, min(1.0, p))),
            scale=(min_scale, max_scale),
            ratio=(0.6, 1.7),
            value=fill,
        )

    if mode == 'stripe':
        return StripeOcclusion(p=p, level=level, fill=fill)

    if mode == 'mixed':
        return MixedOcclusion(p=p, level=level, fill=fill)

    raise ValueError(f'Unsupported occlusion mode: {mode}')


def build_default_transforms(
    img_size=224,
    train_occlusion_mode='none',
    train_occlusion_level='light',
    train_occlusion_p=0.0,
    val_occlusion_mode='none',
    val_occlusion_level='light',
    val_occlusion_p=0.0,
):
    train_ops = [
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
    ]

    train_occ = build_occlusion_transform(
        mode=train_occlusion_mode,
        level=train_occlusion_level,
        p=train_occlusion_p,
        fill=0.0,
    )
    if train_occ is not None:
        train_ops.append(train_occ)
    train_ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    val_ops = [
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ]

    val_occ = build_occlusion_transform(
        mode=val_occlusion_mode,
        level=val_occlusion_level,
        p=val_occlusion_p,
        fill=0.0,
    )
    if val_occ is not None:
        val_ops.append(val_occ)
    val_ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    return {
        "train": transforms.Compose(train_ops),
        "val": transforms.Compose(val_ops),
    }


def write_class_indices(class_names, output_path):
    class_dict = {str(i): name for i, name in enumerate(class_names)}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(class_dict, f, indent=4, ensure_ascii=False)
