import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


class LocalGlobalViT(nn.Module):
    def __init__(self, num_classes=3, top_ratio=0.2):
        super().__init__()

        self.global_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.local_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        self.hidden_dim = self.global_model.hidden_dim
        self.top_ratio = float(top_ratio)

        self.classifier = nn.Linear(self.hidden_dim * 2, num_classes)

    def _forward_features(self, model, x):
        x = model._process_input(x)
        batch_size = x.shape[0]
        cls_token = model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = model.encoder(x)
        return x[:, 0], x[:, 1:]

    def _build_local_mask(self, patch_tokens, image_size):
        batch_size, num_patches, _ = patch_tokens.shape
        patch_size = int(num_patches ** 0.5)

        scores = patch_tokens.norm(dim=-1)
        keep = max(1, int(num_patches * self.top_ratio))
        topk_idx = torch.topk(scores, k=keep, dim=1).indices

        mask = torch.zeros(batch_size, num_patches, device=patch_tokens.device, dtype=patch_tokens.dtype)
        mask.scatter_(1, topk_idx, 1.0)

        mask = mask.reshape(batch_size, 1, patch_size, patch_size)
        mask = F.interpolate(mask, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return mask

    def forward(self, x):
        image_size = x.shape[-1]

        global_cls, patch_tokens = self._forward_features(self.global_model, x)

        local_mask = self._build_local_mask(patch_tokens, image_size)
        local_x = x * local_mask

        local_cls, _ = self._forward_features(self.local_model, local_x)

        feat = torch.cat([global_cls, local_cls], dim=1)
        return self.classifier(feat)


def create_vit_local_global(num_classes, top_ratio=0.2):
    return LocalGlobalViT(num_classes=num_classes, top_ratio=top_ratio)
