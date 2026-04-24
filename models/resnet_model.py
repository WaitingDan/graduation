import os
import torch.nn as nn

# Set mirror before importing timm/huggingface_hub so endpoint is picked up.
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HUGGINGFACE_HUB_ENDPOINT', 'https://hf-mirror.com')

import timm

def create_resnet(num_classes, pretrained=True):
    """
    Create a ResNet-18 model from timm.

    Model: resnet18
    Pretrained weights: ImageNet-1K (when pretrained=True)
    """
    try:
        model = timm.create_model('resnet18', pretrained=pretrained)
    except Exception as e:
        if pretrained:
            raise RuntimeError(
                'Failed to load timm pretrained resnet18 weights. '
                'Check network/mirror access or run with --no_pretrained.'
            ) from e
        raise

    if hasattr(model, 'reset_classifier'):
        model.reset_classifier(num_classes=num_classes)
    elif hasattr(model, 'fc') and hasattr(model.fc, 'in_features'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError('Failed to replace timm resnet18 classifier head')

    return model