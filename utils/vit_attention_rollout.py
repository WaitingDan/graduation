import torch
import cv2
import numpy as np
import os
import sys
import argparse
from torchvision import transforms

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ensure project root is importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from models.vit_model import create_vit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=os.path.join(ROOT_DIR, 'test_images/test_ship_02.jpg'))
    parser.add_argument('--weights', default=os.path.join(ROOT_DIR, 'weights/vit_best.pth'))
    parser.add_argument('--output', default=os.path.join(ROOT_DIR, 'vit_attention_rollout.png'))
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma correction for mask (default 1.0)')
    parser.add_argument('--alpha', type=float, default=0.6, help='overlay alpha for heatmap (0-1)')
    parser.add_argument('--device', default=None, help='cpu or cuda device string')
    return parser.parse_args()


args = parse_args()

device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

image_path = args.image

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))

transform = transforms.Compose([
    transforms.ToTensor()
])

input_tensor = transform(img).unsqueeze(0).to(device)

num_classes = args.num_classes

model = create_vit(num_classes)

weight_path = args.weights

model.load_state_dict(torch.load(weight_path, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 捕获 attention matrix
# -----------------------------

attentions = []

def hook_fn(module, input, output):

    # output shape: (B, tokens, embed_dim)
    # Recompute attention weights from module parameters (works for timm Attention)
    try:
        x = input[0]
    except Exception:
        return

    # x: (B, N, C)
    B, N, C = x.shape

    # qkv linear -> (B, N, 3*C)
    qkv = module.qkv(x)

    # reshape and permute to (3, B, heads, N, head_dim)
    head_dim = C // module.num_heads
    qkv = qkv.reshape(B, N, 3, module.num_heads, head_dim).permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]

    # q, k: (B, heads, N, head_dim)
    attn = (q @ k.transpose(-2, -1)) * module.scale

    attn = attn.softmax(dim=-1)

    attentions.append(attn.detach())

handles = []

for blk in model.blocks:

    handle = blk.attn.register_forward_hook(hook_fn)
    handles.append(handle)

with torch.no_grad():
    _ = model(input_tensor)

for h in handles:
    h.remove()

if len(attentions) == 0:
    raise RuntimeError("No attention maps captured")

# -----------------------------
# Attention Rollout
# -----------------------------

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

# 归一化，防止除以 0
eps = 1e-8
mn = mask.min()
mx = mask.max()
if mx - mn < eps:
    mask = np.zeros_like(mask, dtype=np.float32)
else:
    mask = (mask - mn) / (mx - mn)

# gamma 校正
if hasattr(args, 'gamma') and args.gamma is not None:
    try:
        mask = np.power(mask, float(args.gamma))
    except Exception:
        pass

heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

img_float = img.astype(np.float32) / 255.0

# alpha 混合
alpha = args.alpha if hasattr(args, 'alpha') else 0.6
alpha = float(max(0.0, min(1.0, alpha)))

overlay = alpha * (heatmap.astype(np.float32) / 255.0) + (1.0 - alpha) * img_float
overlay = overlay / (overlay.max() + eps)

# 转为 uint8 再保存
out = (overlay * 255.0).astype(np.uint8)
save_path = args.output if hasattr(args, 'output') else os.path.join(ROOT_DIR, "vit_attention_rollout.png")
cv2.imwrite(save_path, out[:, :, ::-1])

print("Saved:", save_path)

