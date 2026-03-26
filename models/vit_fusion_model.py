import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16
from utils.common import IMAGENET_MEAN, IMAGENET_STD


class AttentionCrop(nn.Module):
    def __init__(self, crop_size=112, out_size=224, topk_patches=5):
        super().__init__()
        self.crop_size = int(crop_size)
        self.out_size = int(out_size)
        self.topk_patches = int(topk_patches)

    def forward(self, images, patch_attention):
        batch_size, _, height, width = images.shape
        patch_count = patch_attention.shape[1]
        patch_grid = int(math.sqrt(patch_count))

        if patch_grid * patch_grid != patch_count:
            raise ValueError(f"Invalid patch attention shape: {patch_attention.shape}")

        crops = []
        for index in range(batch_size):
            attn_vec = patch_attention[index]
            keep = min(max(1, self.topk_patches), attn_vec.numel())

            topk_index = torch.topk(attn_vec, k=keep, dim=0).indices
            ys = topk_index // patch_grid
            xs = topk_index % patch_grid

            center_y = ((ys.float() + 0.5) / patch_grid * height).mean().item()
            center_x = ((xs.float() + 0.5) / patch_grid * width).mean().item()

            y1 = int(max(0, min(height - 1, center_y - self.crop_size / 2)))
            x1 = int(max(0, min(width - 1, center_x - self.crop_size / 2)))
            y2 = min(height, y1 + self.crop_size)
            x2 = min(width, x1 + self.crop_size)

            if y2 <= y1:
                y2 = min(height, y1 + 1)
            if x2 <= x1:
                x2 = min(width, x1 + 1)

            crop = images[index:index + 1, :, y1:y2, x1:x2]
            crop = F.interpolate(crop, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
            crops.append(crop)

        return torch.cat(crops, dim=0)


class ViTGlobalLocalFusion(nn.Module):
    def __init__(self, num_classes, pretrained=False, crop_size=112, out_size=224, topk_patches=5, dropout=0.2, rollout_layers=None):
        super().__init__()

        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.global_model = vit_b_16(weights=weights)
        self.local_model = vit_b_16(weights=weights)

        self.feature_dim = self.global_model.heads.head.in_features
        self.global_model.heads.head = nn.Identity()
        self.local_model.heads.head = nn.Identity()

        self.cropper = AttentionCrop(crop_size=crop_size, out_size=out_size, topk_patches=topk_patches)
        self.global_head = nn.Linear(self.feature_dim, num_classes)
        self.local_head = nn.Linear(self.feature_dim, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim * 2, num_classes),
        )

    def _forward_tokens(self, model, images):
        # torchvision ViT's `_process_input` expects images in [0,1] range
        # while our training transforms normalize images using ImageNet mean/std.
        # If images are already normalized, undo normalization before calling
        # `_process_input` to avoid double-normalization.
        imgs = images
        try:
            mean_check = float(imgs.mean().item())
        except Exception:
            mean_check = 0.0

        if mean_check < 0.3:
            # likely normalized (mean around ~0); unnormalize
            mean = torch.tensor(IMAGENET_MEAN, device=imgs.device).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=imgs.device).view(1, 3, 1, 1)
            imgs = imgs * std + mean

        tokens = model._process_input(imgs)
        batch_size = tokens.shape[0]
        cls_token = model.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        return model.encoder(tokens)

    def _extract_last_attention(self, model, images):
        # See comment in `_forward_tokens` about unnormalizing before calling
        # `_process_input`.
        imgs = images
        try:
            mean_check = float(imgs.mean().item())
        except Exception:
            mean_check = 0.0

        if mean_check < 0.3:
            mean = torch.tensor(IMAGENET_MEAN, device=imgs.device).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=imgs.device).view(1, 3, 1, 1)
            imgs = imgs * std + mean

        tokens = model._process_input(imgs)
        batch_size = tokens.shape[0]
        cls_token = model.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)

        encoder = model.encoder
        tokens = encoder.dropout(tokens)

        layers = list(encoder.layers)
        for layer in layers[:-1]:
            tokens = layer(tokens)

        last_layer = layers[-1]
        norm_tokens = last_layer.ln_1(tokens)

        attn_module = last_layer.self_attention
        qkv = F.linear(norm_tokens, attn_module.in_proj_weight, attn_module.in_proj_bias)

        q, k, _ = qkv.chunk(3, dim=-1)
        head_dim = attn_module.head_dim
        num_heads = attn_module.num_heads
        scale = 1.0 / math.sqrt(head_dim)

        q = q.view(batch_size, q.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, k.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)

        attn_score = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_prob = torch.softmax(attn_score, dim=-1)

        cls_to_patch = attn_prob[:, :, 0, 1:].mean(dim=1)
        return cls_to_patch

    def forward(self, images):
        global_tokens = self._forward_tokens(self.global_model, images)
        global_feature = global_tokens[:, 0]
        global_logits = self.global_head(global_feature)

        patch_attention = self._extract_last_attention(self.global_model, images)
        local_images = self.cropper(images, patch_attention.detach())

        local_tokens = self._forward_tokens(self.local_model, local_images)
        local_feature = local_tokens[:, 0]
        local_logits = self.local_head(local_feature)

        fused_feature = torch.cat([global_feature, local_feature], dim=1)
        fusion_logits = self.classifier(fused_feature)
        return global_logits, local_logits, fusion_logits


class ViTFusion(ViTGlobalLocalFusion):
    def __init__(self, num_classes, pretrained=False, crop_size=112, out_size=224, topk_patches=5, dropout=0.2, rollout_layers=None):
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            crop_size=crop_size,
            out_size=out_size,
            topk_patches=topk_patches,
            dropout=dropout,
            rollout_layers=rollout_layers,
        )


def create_vit_global_local(num_classes, pretrained=False, crop_size=112, out_size=224, topk_patches=5, dropout=0.2, rollout_layers=None):
    return ViTGlobalLocalFusion(
        num_classes=num_classes,
        pretrained=pretrained,
        crop_size=crop_size,
        out_size=out_size,
        topk_patches=topk_patches,
        dropout=dropout,
        rollout_layers=rollout_layers,
    )
