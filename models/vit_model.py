import torch
import torch.nn as nn
import os
import timm


def create_vit(num_classes, pretrained=True):
    """
    Create a ViT-S/16 model from timm.

    Model: vit_small_patch16_224
    Pretrained weights: ImageNet-1K (when pretrained=True)
    """
    # Use mirror endpoint when available to improve pretrained weight download stability.
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

    try:
        model = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
    except Exception as e:
        if pretrained:
            raise RuntimeError(
                'Failed to load timm pretrained vit_small_patch16_224 weights. '
                'Check network/mirror access or run with --no_pretrained.'
            ) from e
        raise

    # Ensure attention map extraction path can capture attention probabilities.
    if hasattr(model, 'blocks'):
        for blk in model.blocks:
            if hasattr(blk, 'attn') and hasattr(blk.attn, 'fused_attn'):
                blk.attn.fused_attn = False

    # replace classifier head (handle different model attribute names)
    if hasattr(model, 'reset_classifier'):
        model.reset_classifier(num_classes=num_classes)
    elif hasattr(model, 'head') and hasattr(model.head, 'in_features'):
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
# Note: Attention-guided ViT (agvit) implementation removed per project decision.
# If you need to restore it later, retrieve it from version control history.