import argparse
import os
import sys
import json
import cv2
import numpy as np
import torch
import math
import torch.nn.functional as F
from torchvision import transforms
from utils.common import IMAGENET_MEAN, IMAGENET_STD

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ensure project root is importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from models.vit_model import create_vit


def generate_vit_rollout(
    image_path,
    weight_path=None,
    output_path=None,
    num_classes=None,
    gamma=1.0,
    alpha=0.6,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 如果未提供 num_classes，则尝试从 class_indices.json 中推断
    if num_classes is None:
        try:
            cls_path = os.path.join(ROOT_DIR, 'class_indices.json')
            with open(cls_path, 'r', encoding='utf-8') as f:
                cls = json.load(f)
            num_classes = len(cls)
        except Exception:
            num_classes = 10

    if weight_path is None:
        weight_path = os.path.join(ROOT_DIR, 'weights/vit_best.pth')

    if output_path is None:
        output_path = os.path.join(ROOT_DIR, 'outputs', 'vit_attention_rollout.png')

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {image_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)

    model = create_vit(num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    # capture attention matrices
    attentions = []

    def hook_fn(module, hook_input, hook_output):
        # try to extract the input tensor (could be different shapes depending on implementation)
        if len(hook_input) == 0:
            return
        x = hook_input[0]
        if not isinstance(x, torch.Tensor):
            return

        try:
            # Case 1: qkv-style attention modules expose a `qkv` method/attribute
            if hasattr(module, 'qkv') and hasattr(module, 'num_heads'):
                # expected shape: (batch, tokens, channels)
                if x.dim() != 3:
                    return
                bsz, tokens, channels = x.shape
                qkv = module.qkv(x)
                head_dim = channels // module.num_heads
                qkv = qkv.reshape(bsz, tokens, 3, module.num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q, k, _v = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * module.scale
                attn = attn.softmax(dim=-1)
                attentions.append(attn.detach())
                return

            # Case 2: torch.nn.MultiheadAttention or similar (using in_proj_weight / in_proj_bias)
            # hook_input for MultiheadAttention is often (query, key, value, ...)
            if isinstance(module, torch.nn.MultiheadAttention) or hasattr(module, 'in_proj_weight'):
                # pick the first tensor-like input as the sequence
                seq = x
                # support both (batch, tokens, channels) and (tokens, batch, channels)
                if seq.dim() == 3:
                    if getattr(module, 'batch_first', True):
                        bsz, tokens, channels = seq.shape
                        flat = seq.reshape(bsz * tokens, channels)
                        proj_w = module.in_proj_weight
                        proj_b = module.in_proj_bias if hasattr(module, 'in_proj_bias') else None
                        qkv = F.linear(flat, proj_w, proj_b)
                        qkv = qkv.view(bsz, tokens, 3, channels)
                        q = qkv[:, :, 0, :]
                        k = qkv[:, :, 1, :]
                    else:
                        tokens, bsz, channels = seq.shape
                        flat = seq.reshape(tokens * bsz, channels)
                        proj_w = module.in_proj_weight
                        proj_b = module.in_proj_bias if hasattr(module, 'in_proj_bias') else None
                        qkv = F.linear(flat, proj_w, proj_b)
                        qkv = qkv.view(tokens, bsz, 3, channels)
                        q = qkv[:, :, 0, :].permute(1, 0, 2)
                        k = qkv[:, :, 1, :].permute(1, 0, 2)

                    # reshape heads: (batch, heads, tokens, head_dim)
                    num_heads = module.num_heads if hasattr(module, 'num_heads') else getattr(module, 'head_dim', 1)
                    head_dim = channels // num_heads
                    q = q.reshape(bsz, tokens, num_heads, head_dim).permute(0, 2, 1, 3)
                    k = k.reshape(bsz, tokens, num_heads, head_dim).permute(0, 2, 1, 3)

                    scale = 1.0 / math.sqrt(head_dim)
                    attn = (q @ k.transpose(-2, -1)) * scale
                    attn = attn.softmax(dim=-1)
                    attentions.append(attn.detach())
                    return

        except Exception:
            return

    handles = []
    # prefer block list style: model.blocks with blk.attn
    candidates = []
    if hasattr(model, 'blocks'):
        for blk in model.blocks:
            att = getattr(blk, 'attn', None)
            if att is not None and hasattr(att, 'register_forward_hook'):
                handles.append(att.register_forward_hook(hook_fn))
                candidates.append((f'blocks.{len(candidates)}', att.__class__.__name__))
            else:
                # if the block itself looks like an attention module
                if all(hasattr(blk, a) for a in ('qkv', 'num_heads', 'scale')):
                    handles.append(blk.register_forward_hook(hook_fn))
                    candidates.append((f'blocks.{len(candidates)}', blk.__class__.__name__))

    # fallback: scan all submodules and hook attention-like modules (robust for torchvision variants)
    if len(handles) == 0:
        for name, module in model.named_modules():
            mclass = module.__class__.__name__.lower()
            attrs = set(dir(module))
            has_qkv = 'qkv' in attrs
            has_q_and_k = 'q' in attrs and 'k' in attrs
            has_num_heads = 'num_heads' in attrs or 'num_attention_heads' in attrs
            name_hint = ('attn' in name.lower()) or ('attention' in name.lower()) or ('multihead' in mclass) or ('selfattention' in mclass) or ('attention' in mclass)

            if (has_qkv or has_q_and_k) and (has_num_heads or 'scale' in attrs):
                try:
                    handles.append(module.register_forward_hook(hook_fn))
                    candidates.append((name, module.__class__.__name__))
                except Exception:
                    pass
            elif name_hint and hasattr(module, 'register_forward_hook'):
                try:
                    handles.append(module.register_forward_hook(hook_fn))
                    candidates.append((name, module.__class__.__name__))
                except Exception:
                    pass

    if len(handles) == 0:
        # build short hint list to aid debugging
        hints = []
        for name, module in model.named_modules():
            mclass = module.__class__.__name__
            if (('attn' in name.lower()) or ('attention' in name.lower()) or 'qkv' in dir(module) or 'num_heads' in dir(module) or 'scale' in dir(module) or 'multihead' in mclass.lower()):
                hints.append(f"{name} ({mclass})")
        hint_msg = ', '.join(hints[:40]) if len(hints) > 0 else 'none'
        raise RuntimeError(f'No attention modules found to hook into. Candidate modules: {hint_msg}')

    with torch.no_grad():
        _ = model(input_tensor)

    for h in handles:
        h.remove()

    if len(attentions) == 0:
        raise RuntimeError("No attention maps captured")

    # rollout
    result = torch.eye(attentions[0].size(-1)).to(device)
    for attn in attentions:
        attn = attn.mean(dim=1)
        attn = attn + torch.eye(attn.size(-1)).to(device)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        result = torch.matmul(attn, result)

    mask = result[0, 0, 1:]
    size = int(mask.shape[0] ** 0.5)
    mask = mask.reshape(size, size).cpu().numpy()
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_LINEAR)

    eps = 1e-8
    mn = mask.min()
    mx = mask.max()
    if mx - mn < eps:
        mask = np.zeros_like(mask, dtype=np.float32)
    else:
        mask = (mask - mn) / (mx - mn)

    try:
        mask = np.power(mask, float(gamma))
    except Exception:
        pass

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_float = img.astype(np.float32) / 255.0
    alpha = float(max(0.0, min(1.0, alpha)))
    overlay = alpha * (heatmap.astype(np.float32) / 255.0) + (1.0 - alpha) * img_float
    overlay = overlay / (overlay.max() + eps)

    out = (overlay * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, out[:, :, ::-1])
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='path to input image')
    parser.add_argument('--weights', default=os.path.join(ROOT_DIR, 'weights/vit_best.pth'))
    parser.add_argument('--output', default=os.path.join(ROOT_DIR, 'outputs', 'vit_attention_rollout.png'))
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    path = generate_vit_rollout(
        image_path=args.image,
        weight_path=args.weights,
        output_path=args.output,
        num_classes=args.num_classes,
        gamma=args.gamma,
        alpha=args.alpha,
        device=device,
    )
    print("Saved:", path)


if __name__ == '__main__':
    main()

