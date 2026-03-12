import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def create_resnet(num_classes):

    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model