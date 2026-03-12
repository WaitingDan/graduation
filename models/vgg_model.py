from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn

def create_vgg(num_classes):

    model = vgg16(weights=VGG16_Weights.DEFAULT)

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model