import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class StructViT(nn.Module):
    def __init__(self, base_model, num_patches, alpha=0.5, num_classes=10):
        super().__init__()
        self.base = base_model
        self.alpha = alpha

        # learnable relation matrix for patches (exclude cls token)
        self.num_patches = num_patches
        self.A = nn.Parameter(torch.zeros(num_patches, num_patches))

        # replace base head with identity and use our own head
        in_features = self.base.head.in_features
        self.base.head = nn.Identity()
        self.head = nn.Linear(in_features, num_classes)

        # keep normalization if present
        self.norm = getattr(self.base, 'norm', None)

    def forward(self, x):
        B = x.shape[0]

        # patch embed
        x = self.base.patch_embed(x)  # (B, N, C)

        # cls token
        cls_token = self.base.cls_token.expand(B, -1, -1)  # (B,1,C)
        x = torch.cat((cls_token, x), dim=1)  # (B, N+1, C)

        # add pos embed
        if hasattr(self.base, 'pos_embed'):
            # pos_embed shape (1, N+1, C)
            x = x + self.base.pos_embed

        x = self.base.pos_drop(x)

        # pass through transformer blocks
        for blk in self.base.blocks:
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        # separate cls and patch tokens
        cls, tokens = x[:, 0:1, :], x[:, 1:, :]  # tokens: (B, N, C)

        # apply learnable relation
        A = F.softmax(self.A, dim=-1)  # (N, N)
        transformed = torch.matmul(A.unsqueeze(0), tokens)  # (B, N, C)
        tokens = tokens + self.alpha * transformed

        # concat back
        x = torch.cat((cls, tokens), dim=1)

        # classification by cls token
        out = x[:, 0]
        out = self.head(out)

        return out


def create_struct_vit(num_classes, alpha=0.5):
    """Create a Vision Transformer with a lightweight learnable patch-relation module.

    This prototype attaches a learnable relation matrix A (N_patches x N_patches)
    that mixes patch tokens after the transformer's blocks and before the head.

    """
    base = timm.create_model("vit_base_patch16_224", pretrained=True)

    # try to read number of patches from patch_embed
    try:
        num_patches = int(base.patch_embed.num_patches)
    except Exception:
        # fallback to 14*14 for patch16@224
        num_patches = 14 * 14

    model = StructViT(base, num_patches=num_patches, alpha=alpha, num_classes=num_classes)
    return model
