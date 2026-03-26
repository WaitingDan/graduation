import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def create_resnet(num_classes, pretrained=True):

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model