"""
vit_fusion_model.py

架构修正版，核心变化：
1. 单次前向同时拿到注意力与 token，避免两次前向带来的不一致与额外开销。
2. 在中间层后做注意力引导的 patch 重加权，再送入后半段 block。
3. 保留遮挡感知池化与自适应融合门控，并做特征空间对齐。
4. 保留 get_attention_map 兼容可视化脚本。
"""

import math
import torch
import torch.nn as nn
from models.vit_model import create_vit


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def get_feature_dim(model):
    if hasattr(model, 'num_features'):
        return int(model.num_features)
    if hasattr(model, 'head') and hasattr(model.head, 'in_features'):
        return int(model.head.in_features)
    if hasattr(model, 'heads') and hasattr(model.heads, 'head') \
            and hasattr(model.heads.head, 'in_features'):
        return int(model.heads.head.in_features)
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'in_features'):
        return int(model.classifier.in_features)
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            return int(module.in_features)
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


def _prepare_vit_tokens(model, x):
    """手动走 patch embed + position embed，返回初始 token 序列。"""
    tokens = model.patch_embed(x)
    if hasattr(model, '_pos_embed'):
        tokens = model._pos_embed(tokens)
    else:
        cls_token = getattr(model, 'cls_token', None)
        if cls_token is not None:
            dist_token = getattr(model, 'dist_token', None)
            cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
            if dist_token is not None:
                tokens = torch.cat(
                    (cls_tokens, dist_token.expand(tokens.shape[0], -1, -1), tokens),
                    dim=1,
                )
            else:
                tokens = torch.cat((cls_tokens, tokens), dim=1)
        pos_embed = getattr(model, 'pos_embed', None)
        if pos_embed is not None:
            tokens = tokens + pos_embed
        pos_drop = getattr(model, 'pos_drop', None)
        if pos_drop is not None:
            tokens = pos_drop(tokens)

    if hasattr(model, 'patch_drop'):
        tokens = model.patch_drop(tokens)
    if hasattr(model, 'norm_pre'):
        tokens = model.norm_pre(tokens)
    return tokens


def _get_block_attention(block, tokens):
    """从一个 ViT block 提取注意力权重矩阵，不改变 tokens。"""
    attn_module = getattr(block, 'attn', None)
    if attn_module is None or not hasattr(attn_module, 'qkv'):
        return None

    norm_tokens = block.norm1(tokens) if hasattr(block, 'norm1') else tokens
    qkv = attn_module.qkv(norm_tokens)
    batch_size, token_count, _ = qkv.shape
    num_heads = int(getattr(attn_module, 'num_heads', 0))
    if num_heads <= 0:
        return None

    head_dim = qkv.shape[-1] // (3 * num_heads)
    q, k = qkv.reshape(batch_size, token_count, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)[:2]
    scale = float(getattr(attn_module, 'scale', head_dim ** -0.5))
    return (q * scale @ k.transpose(-2, -1)).softmax(dim=-1)


def _rollout_from_attentions(attentions, device):
    """对注意力矩阵列表做 rollout，返回 cls-to-patch [B, N_patch]。"""
    if not attentions:
        return None

    token_count = attentions[0].shape[-1]
    eye = torch.eye(token_count, device=device).unsqueeze(0)
    rollout = eye.expand(attentions[0].shape[0], -1, -1).clone()

    for attn in attentions:
        attn_mean = attn.mean(dim=1)
        attn_aug = attn_mean + eye
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        rollout = torch.bmm(attn_aug, rollout)

    return rollout[:, 0, 1:]


def get_attention_map(model, x, rollout_layers=None):
    """兼容可视化脚本的接口，返回 cls-to-patch rollout 注意力 [B, N]。"""
    layers = int(rollout_layers) if rollout_layers is not None else 4
    tokens = _prepare_vit_tokens(model, x)
    attentions = []

    with torch.no_grad():
        for block in model.blocks:
            attn = _get_block_attention(block, tokens)
            if attn is not None:
                attentions.append(attn)
            tokens = block(tokens)

    if not attentions:
        raise RuntimeError('Failed to capture attention maps')
    return _rollout_from_attentions(attentions[-layers:], x.device)


def get_attention_rollout(model, x, rollout_layers=4):
    """向后兼容旧函数名。"""
    return get_attention_map(model, x, rollout_layers=rollout_layers)


# ---------------------------------------------------------------------------
# 子模块
# ---------------------------------------------------------------------------

class MidLayerAttnReweight(nn.Module):
    """
    中间层注意力引导 patch token 重加权。

    残差形式：out = x + gamma * (weighted - x)
    gamma 初始化为 0，训练初期等价 identity。
    """

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = float(max(0.1, temperature))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.score_proj = nn.Sequential(
            nn.Linear(1, 4),
            nn.GELU(),
            nn.Linear(4, 1),
        )

        # 初始化为近似恒等映射
        nn.init.zeros_(self.score_proj[0].weight)
        nn.init.ones_(self.score_proj[0].bias)
        nn.init.zeros_(self.score_proj[2].weight)
        nn.init.zeros_(self.score_proj[2].bias)

    def forward(self, patch_tokens, rollout_score):
        s_min = rollout_score.min(dim=1, keepdim=True).values
        s_max = rollout_score.max(dim=1, keepdim=True).values
        score_norm = (rollout_score - s_min) / (s_max - s_min + 1e-6)

        score_proj = self.score_proj(score_norm.unsqueeze(-1)).squeeze(-1)
        weight = torch.softmax(score_proj / self.temperature, dim=-1)
        weighted = patch_tokens * weight.unsqueeze(-1)
        return patch_tokens + self.gamma * (weighted - patch_tokens)


class OcclusionAwareTokenPool(nn.Module):
    """遮挡感知 Top-K 池化。"""

    def __init__(self, topk=6, entropy_penalty=0.5):
        super().__init__()
        self.topk = int(topk)
        self.entropy_penalty = float(entropy_penalty)

    def forward(self, patch_tokens, attn_score):
        batch, n_patch, dim = patch_tokens.shape
        topk = min(self.topk, n_patch)

        # Per-token entropy contribution keeps the penalty effective after normalization.
        token_uncertainty = -(attn_score * torch.log(attn_score.clamp_min(1e-8)))
        token_uncertainty = token_uncertainty / math.log(max(2, n_patch))
        token_confidence = (1.0 - self.entropy_penalty * token_uncertainty).clamp(0.1, 1.0)

        weight = attn_score * token_confidence
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        local_feat = (patch_tokens * weight.unsqueeze(-1)).sum(dim=1)

        topk_idx = weight.topk(topk, dim=-1).indices
        topk_tokens = torch.gather(
            patch_tokens,
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, dim),
        )
        return local_feat, topk_tokens


class LocalRefiner(nn.Module):
    """局部向量残差精炼。"""

    def __init__(self, dim, hidden_ratio=0.5):
        super().__init__()
        hidden = max(32, int(dim * hidden_ratio))
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma * self.mlp(self.norm(x))


class FusionGate(nn.Module):
    """融合门控，先做拼接归一化后输出 [B, 1]。"""

    def __init__(self, dim, hidden_ratio=0.25):
        super().__init__()
        hidden = max(16, int(dim * hidden_ratio))
        self.norm = nn.LayerNorm(dim * 2)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, global_feat, local_feat):
        fused = self.norm(torch.cat([global_feat, local_feat], dim=-1))
        return torch.sigmoid(self.net(fused))


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------

class ViTFusionModel(nn.Module):
    """AG-ViT: Attention-Guided ViT for Ship Recognition."""

    def __init__(
        self,
        num_classes=10,
        topk=6,
        pretrained=True,
        attn_rollout_layers=4,
        split_layer=8,
        part_gate_init=0.0,
        part_dropout_p=0.1,
        attn_temperature=1.0,
        entropy_penalty=0.5,
        mid_reweight_temperature=1.0,
        enable_module_a=True,
        enable_module_b=True,
        enable_module_c=True,
        # 兼容旧接口
        use_part_self_attention=None,
        use_local_self_attention=None,
        local_gate_init=None,
        **_unused,
    ):
        super().__init__()

        del use_part_self_attention, use_local_self_attention
        if local_gate_init is not None:
            part_gate_init = float(local_gate_init)

        self.topk = max(1, int(topk))
        self.attn_rollout_layers = max(1, int(attn_rollout_layers))
        self.part_dropout_p = float(max(0.0, min(0.9, part_dropout_p)))
        self.attn_temperature = float(max(0.1, attn_temperature))
        self.enable_module_a = bool(enable_module_a)
        self.enable_module_b = bool(enable_module_b)
        self.enable_module_c = bool(enable_module_c)

        self.backbone = create_vit(num_classes=num_classes, pretrained=pretrained)
        set_feature_extractor_head(self.backbone)
        self.global_model = self.backbone

        self.embed_dim = get_feature_dim(self.backbone)
        n_blocks = len(self.backbone.blocks)
        # split_layer 表示在第 split_layer 层之后插入重加权。
        self.split_layer = max(1, min(int(split_layer), n_blocks - 1))

        self.mid_reweight = MidLayerAttnReweight(temperature=mid_reweight_temperature)
        self.occ_pool = OcclusionAwareTokenPool(topk=self.topk, entropy_penalty=entropy_penalty)
        self.local_refiner = LocalRefiner(self.embed_dim, hidden_ratio=0.5)
        self.fusion_gate = FusionGate(self.embed_dim, hidden_ratio=0.25)

        self.part_gate = nn.Parameter(torch.tensor(float(part_gate_init)))
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        self.local_classifier = nn.Linear(self.embed_dim, num_classes)

    def _normalize_score(self, score):
        s_min = score.min(dim=1, keepdim=True).values
        s_max = score.max(dim=1, keepdim=True).values
        norm = (score - s_min) / (s_max - s_min + 1e-6)
        return torch.softmax(norm / self.attn_temperature, dim=-1)

    def _forward_with_attn(self, x):
        """
        单次遍历 backbone：
        1) 前半段收集注意力并 rollout。
        2) 在 split_layer 后对 patch token 做重加权。
        3) 继续后半段 block，输出最终 token。
        """
        tokens = _prepare_vit_tokens(self.backbone, x)
        first_half_attentions = []
        rollout_score = None

        for idx, block in enumerate(self.backbone.blocks):
            if idx < self.split_layer:
                attn = _get_block_attention(block, tokens)
                if attn is not None:
                    first_half_attentions.append(attn)

            tokens = block(tokens)

            # 在 split_layer 层后插入注意力引导重加权
            if idx == self.split_layer - 1:
                layers = min(self.attn_rollout_layers, len(first_half_attentions))
                rollout_score = _rollout_from_attentions(first_half_attentions[-layers:], x.device)

                if rollout_score is None:
                    n_patch = tokens.shape[1] - 1
                    rollout_score = torch.ones(tokens.shape[0], n_patch, device=x.device) / n_patch

                cls_token = tokens[:, :1, :]
                patch_tokens = tokens[:, 1:, :]
                if self.enable_module_a:
                    patch_tokens = self.mid_reweight(patch_tokens, rollout_score)
                tokens = torch.cat([cls_token, patch_tokens], dim=1)

        if hasattr(self.backbone, 'norm'):
            tokens = self.backbone.norm(tokens)

        if rollout_score is None:
            # 极端情况下 split 前未取到可用注意力，fallback 为均匀分布
            n_patch = tokens.shape[1] - 1
            rollout_score = torch.ones(tokens.shape[0], n_patch, device=x.device) / n_patch

        return tokens, rollout_score

    def backbone_parameters(self):
        yield from self.backbone.parameters()

    def head_parameters(self):
        for module in (
            self.mid_reweight,
            self.occ_pool,
            self.local_refiner,
            self.fusion_gate,
            self.classifier,
            self.local_classifier,
        ):
            yield from module.parameters()
        yield self.part_gate

    def forward(self, x, attn_map=None):
        # 保留 attn_map 入参兼容性，但默认使用单次前向内部生成的 rollout。
        final_tokens, rollout_score = self._forward_with_attn(x)

        if attn_map is not None:
            # 外部传入时仅用于池化分支，保持向后兼容。
            rollout_score = attn_map

        cls_token = final_tokens[:, 0, :]
        patch_count = rollout_score.shape[1]
        patch_tokens = final_tokens[:, -patch_count:, :]

        global_feat = cls_token
        score = self._normalize_score(rollout_score)

        if self.training and self.part_dropout_p > 0.0:
            mask = (torch.rand_like(score) > self.part_dropout_p).float()
            score = score * mask
            score = score / score.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        local_feat, topk_tokens = self.occ_pool(patch_tokens, score)
        if self.enable_module_b:
            local_feat = self.local_refiner(local_feat)

        global_gate = torch.sigmoid(self.part_gate)
        if self.enable_module_c:
            content_gate = self.fusion_gate(global_feat, local_feat)
        else:
            content_gate = torch.ones_like(global_gate)
        gate = global_gate * content_gate

        fused_feat = global_feat + gate * (local_feat - global_feat)
        logits = self.classifier(fused_feat)

        entropy = -(score * torch.log(score.clamp_min(1e-8))).sum(dim=-1)
        aux = {
            'part_gate': gate,
            'attn_entropy': entropy.mean(),
            'local_feat': local_feat,
            'local_gate': gate,
            'local_branch_enabled': True,
            'attn_temperature': self.attn_temperature,
            'enable_module_a': self.enable_module_a,
            'enable_module_b': self.enable_module_b,
            'enable_module_c': self.enable_module_c,
        }
        return logits, global_feat, topk_tokens, aux


class ViTFusion(ViTFusionModel):
    def __init__(
        self,
        num_classes,
        pretrained=False,
        topk_patches=6,
        attn_rollout_layers=4,
        local_gate_init=0.0,
        part_dropout_p=0.1,
        attn_temperature=1.0,
        entropy_penalty=0.5,
        split_layer=8,
        mid_reweight_temperature=1.0,
        **_unused,
    ):
        super().__init__(
            num_classes=num_classes,
            topk=topk_patches,
            pretrained=pretrained,
            attn_rollout_layers=attn_rollout_layers,
            part_gate_init=local_gate_init,
            part_dropout_p=part_dropout_p,
            attn_temperature=attn_temperature,
            entropy_penalty=entropy_penalty,
            split_layer=split_layer,
            mid_reweight_temperature=mid_reweight_temperature,
        )


def create_vit_global_local(num_classes, pretrained=False, **kwargs):
    return ViTFusionModel(num_classes=num_classes, pretrained=pretrained, **kwargs)
