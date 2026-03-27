import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vit_model import create_vit


class MultiPartAttentionSoftMask:
    def __init__(self, topk=2, patch_size=16):
        self.topk = int(topk)
        self.patch_size = int(patch_size)

    def __call__(self, images, attn_map):
        bsz, _, height, width = images.shape
        grid_h = max(1, height // self.patch_size)
        grid_w = max(1, width // self.patch_size)
        total_grid = max(1, grid_h * grid_w)
        topk = max(1, min(self.topk, attn_map.shape[1]))

        local_views = []
        attn_map = torch.softmax(attn_map, dim=-1)

        for b in range(bsz):
            attn = attn_map[b]
            topk_vals, topk_idx = torch.topk(attn, topk)

            part_views = []
            for score, idx in zip(topk_vals, topk_idx):
                idx_int = int(idx.item()) % total_grid
                row = idx_int // grid_w
                col = idx_int % grid_w

                patch_mask = torch.zeros((1, 1, grid_h, grid_w), device=images.device, dtype=images.dtype)
                patch_mask[0, 0, row, col] = 1.0
                soft_mask = F.interpolate(patch_mask, size=(height, width), mode='bilinear', align_corners=False)
                soft_mask = soft_mask / soft_mask.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

                # Keep a small amount of global context to avoid information collapse.
                weighted = images[b : b + 1] * (0.2 + 0.8 * soft_mask * score)
                part_views.append(weighted)

            local_views.append(torch.cat(part_views, dim=0))

        return torch.stack(local_views, dim=0)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, global_feat, local_feat):
        query = global_feat.unsqueeze(1)
        key = local_feat
        value = local_feat

        out, _ = self.attn(query, key, value)
        out = self.norm(out + query)
        return out.squeeze(1)


def get_feature_dim(model):
    if hasattr(model, 'num_features'):
        return int(model.num_features)
    if hasattr(model, 'head') and hasattr(model.head, 'in_features'):
        return int(model.head.in_features)
    if hasattr(model, 'heads') and hasattr(model.heads, 'head') and hasattr(model.heads.head, 'in_features'):
        return int(model.heads.head.in_features)
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
        return int(model.classifier.in_features)
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            return int(m.in_features)
    raise RuntimeError('Failed to determine feature dimension for model')


def set_feature_extractor_head(model):
    if hasattr(model, 'head') and isinstance(model.head, nn.Module):
        model.head = nn.Identity()
        return
    if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
        model.heads.head = nn.Identity()
        return
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
        model.classifier = nn.Identity()
        return


def get_attention_map(model, x, rollout_layers=None):
    # Try torchvision-style encoder rollout if available
    if hasattr(model, 'encoder') and hasattr(model, '_process_input'):
        tokens = model._process_input(x)
        batch_size = tokens.shape[0]
        encoder = model.encoder
        tokens = encoder.dropout(tokens)

        all_layers = list(encoder.layers)
        if rollout_layers is None or int(getattr(rollout_layers, '')) == 0:
            selected_start = 0
        else:
            selected_start = max(0, len(all_layers) - int(rollout_layers))

        attentions = []
        for layer_index, layer in enumerate(all_layers):
            norm_tokens = layer.ln_1(tokens)
            attn_module = getattr(layer, 'self_attention', None) or getattr(layer, 'attention', None)
            if attn_module is None:
                tokens = layer(tokens)
                continue

            try:
                qkv = F.linear(norm_tokens, attn_module.in_proj_weight, attn_module.in_proj_bias)
                q, k, _ = qkv.chunk(3, dim=-1)
                head_dim = attn_module.head_dim
                num_heads = attn_module.num_heads
                scale = 1.0 / (head_dim ** 0.5)

                q = q.view(batch_size, q.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
                k = k.view(batch_size, k.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
                attn_score = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_prob = torch.softmax(attn_score, dim=-1)
            except Exception:
                tokens = layer(tokens)
                continue

            if layer_index >= selected_start:
                attentions.append(attn_prob)

            tokens = layer(tokens)

        if not attentions:
            raise RuntimeError('No attention layers available for rollout.')

        token_count = attentions[0].size(-1)
        eye = torch.eye(token_count, device=x.device).unsqueeze(0)
        rollout = eye.expand(batch_size, token_count, token_count)
        for attn in attentions:
            attn_mean = attn.mean(dim=1)
            attn_mean = attn_mean + eye
            attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            rollout = torch.matmul(attn_mean, rollout)

        cls_to_patch = rollout[:, 0, 1:]
        return cls_to_patch

    # Fallback: try timm-style blocks hook
    if hasattr(model, 'blocks') and model.blocks:
        captured = []

        def hook(_module, _inputs, output):
            captured.append(output)

        handle = model.blocks[-1].attn.attn_drop.register_forward_hook(hook)
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                _ = model(x)
        finally:
            handle.remove()
            if was_training:
                model.train()

        if not captured:
            raise RuntimeError('Failed to capture attention map from global model.')

        attn = captured[0]
        if attn.dim() == 4:
            attn_map = attn.mean(dim=1)[:, 0, 1:]
        elif attn.dim() == 3:
            if attn.shape[1] == attn.shape[2]:
                attn_map = attn[:, 0, 1:]
            else:
                attn_map = attn[:, 1:]
        else:
            raise RuntimeError(f'Unexpected attention shape: {tuple(attn.shape)}')

        return attn_map

    raise RuntimeError('Model type not supported for attention extraction')


class ViTFusionModel(nn.Module):
    def __init__(
        self,
        num_classes=10,
        topk=2,
        pretrained=True,
        patch_size=16,
        use_local_branch=True,
        use_attention_guidance=True,
        use_cross_attention=True,
        local_gate_init=1.0,
    ):
        super().__init__()

        self.use_local_branch = bool(use_local_branch)
        self.use_attention_guidance = bool(use_attention_guidance)
        self.use_cross_attention = bool(use_cross_attention)

        self.global_model = create_vit(num_classes=num_classes, pretrained=pretrained)
        set_feature_extractor_head(self.global_model)

        self.local_model = create_vit(num_classes=num_classes, pretrained=pretrained)
        set_feature_extractor_head(self.local_model)

        self.embed_dim = get_feature_dim(self.global_model)
        self.localizer = MultiPartAttentionSoftMask(topk=topk, patch_size=patch_size)
        self.fusion = CrossAttentionFusion(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        self.local_gate = nn.Parameter(torch.tensor(float(local_gate_init)))

    def forward(self, x, attn_map=None):
        if attn_map is None and self.use_local_branch and self.use_attention_guidance:
            attn_map = get_attention_map(self.global_model, x)

        global_feat = self.global_model(x)

        if not self.use_local_branch:
            fused_feat = global_feat
            logits = self.classifier(fused_feat)
            aux = {
                'local_gate': torch.tensor(0.0, device=x.device, dtype=global_feat.dtype),
                'local_branch_enabled': False,
            }
            return logits, global_feat, None, aux

        if self.use_attention_guidance and attn_map is not None:
            local_inputs = self.localizer(x, attn_map)
        else:
            local_inputs = x.unsqueeze(1).repeat(1, self.localizer.topk, 1, 1, 1)

        bsz, k_parts, ch, height, width = local_inputs.shape
        local_inputs = local_inputs.view(bsz * k_parts, ch, height, width)
        local_feat = self.local_model(local_inputs)
        local_feat = local_feat.view(bsz, k_parts, -1)

        gate = torch.sigmoid(self.local_gate)
        if self.use_cross_attention:
            local_enhanced = self.fusion(global_feat, local_feat)
        else:
            local_enhanced = local_feat.mean(dim=1)

        fused_feat = (1.0 - gate) * global_feat + gate * local_enhanced
        out = self.classifier(fused_feat)
        aux = {
            'local_gate': gate,
            'local_branch_enabled': True,
        }
        return out, global_feat, local_feat, aux


# Backward-compatible aliases used by other scripts.
class ViTFusion(ViTFusionModel):
    def __init__(
        self,
        num_classes,
        pretrained=False,
        crop_size=112,
        out_size=224,
        topk_patches=2,
        dropout=0.2,
        rollout_layers=4,
        attn_dropout_p=0.2,
        local_gate_init=1.0,
    ):
        del crop_size, out_size, dropout, rollout_layers, attn_dropout_p
        super().__init__(
            num_classes=num_classes,
            topk=topk_patches,
            pretrained=pretrained,
            local_gate_init=local_gate_init,
        )


def create_vit_global_local(
    num_classes,
    pretrained=False,
    crop_size=112,
    out_size=224,
    topk_patches=2,
    dropout=0.2,
    rollout_layers=4,
    attn_dropout_p=0.2,
    local_gate_init=1.0,
):
    del crop_size, out_size, dropout, rollout_layers, attn_dropout_p
    return ViTFusionModel(
        num_classes=num_classes,
        topk=topk_patches,
        pretrained=pretrained,
        local_gate_init=local_gate_init,
    )
