import torch.nn as nn


def create_vit(num_classes, pretrained=True):
    """
    Create a ViT model using torchvision pretrained weights.
    """
    model = None
    # try torchvision first
    try:
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
    except Exception:
        raise RuntimeError('torchvision ViT model not available; please install a compatible torchvision version')

    # replace classifier head (handle different model attribute names)
    if hasattr(model, 'head') and hasattr(model.head, 'in_features'):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'heads') and hasattr(model.heads, 'head') and hasattr(model.heads.head, 'in_features'):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        # last resort: try to find a linear attribute
        replaced = False
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                parent = name.rsplit('.', 1)[0] if '.' in name else ''
                if parent:
                    parts = parent.split('.')
                    attr = model
                    for p in parts[:-1]:
                        attr = getattr(attr, p)
                    last = parts[-1]
                    setattr(attr, last, nn.Linear(module.in_features, num_classes))
                    replaced = True
                    break
        if not replaced:
            raise RuntimeError('Failed to replace ViT classifier head')

    return model