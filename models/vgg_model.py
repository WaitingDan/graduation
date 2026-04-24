import os
import torch.nn as nn

# Set mirror before importing timm/huggingface_hub so endpoint is picked up.
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HUGGINGFACE_HUB_ENDPOINT', 'https://hf-mirror.com')

import timm


def create_vgg(num_classes, pretrained=True):
    """
    Create a VGG-13 model from timm.

    Model: vgg13
    Pretrained weights: ImageNet-1K (when pretrained=True)
    """
    try:
        model = timm.create_model('vgg13', pretrained=pretrained)
    except Exception as e:
        if pretrained:
            raise RuntimeError(
                'Failed to load timm pretrained vgg13 weights. '
                'Check network/mirror access or run with --no_pretrained.'
            ) from e
        raise

    if hasattr(model, 'reset_classifier'):
        model.reset_classifier(num_classes=num_classes)
    elif hasattr(model, 'get_classifier'):
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Linear):
            in_features = classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError('Unexpected timm vgg13 classifier type')
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        if len(model.classifier) == 0 or not hasattr(model.classifier[-1], 'in_features'):
            raise RuntimeError('Failed to replace timm vgg13 classifier head')
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError('Failed to replace timm vgg13 classifier head')

    return model
