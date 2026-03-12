import timm
import torch.nn as nn


def create_vit(num_classes):

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True
    )

    in_features = model.head.in_features

    model.head = nn.Linear(in_features, num_classes)

    return model