import argparse
import os
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ensure project root is importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from models.vit_model import create_vit


def generate_vit_rollout(
    image_path,
    weight_path=None,
    output_path=None,
    num_classes=10,
    gamma=1.0,
    alpha=0.6,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if weight_path is None:
        weight_path = os.path.join(ROOT_DIR, 'weights/vit_best.pth')

    if output_path is None:
        output_path = os.path.join(ROOT_DIR, 'outputs', 'vit_attention_rollout.png')

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    input_tensor = transform(img).unsqueeze(0).to(device)

    model = create_vit(num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    # capture attention matrices
    attentions = []

    def hook_fn(module, hook_input, hook_output):
        try:
            x = hook_input[0]
        except Exception:
            return

        bsz, tokens, channels = x.shape
        qkv = module.qkv(x)
        head_dim = channels // module.num_heads
        qkv = qkv.reshape(bsz, tokens, 3, module.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, _v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        attentions.append(attn.detach())

    handles = []
    for blk in model.blocks:
        handles.append(blk.attn.register_forward_hook(hook_fn))

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
    parser.add_argument('--image', default=os.path.join(ROOT_DIR, 'test_images/test_ship_02.jpg'))
    parser.add_argument('--weights', default=os.path.join(ROOT_DIR, 'weights/vit_best.pth'))
    parser.add_argument('--output', default=os.path.join(ROOT_DIR, 'outputs', 'vit_attention_rollout.png'))
    parser.add_argument('--num_classes', type=int, default=10)
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

