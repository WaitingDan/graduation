import torch
import torch.nn.functional as F

def attention_crop(images, attn_maps, crop_size=112, out_size=224):
    """
    images: [B, C, H, W]
    attn_maps: [B, 1, H, W]
    """

    B, C, H, W = images.shape
    crops = []

    for i in range(B):
        attn = attn_maps[i, 0]

        # 找最大响应点
        max_idx = int(torch.argmax(attn).item())
        y = max_idx // W
        x = max_idx % W

        # 裁剪区域
        y1 = max(0, y - crop_size // 2)
        y2 = min(H, y + crop_size // 2)
        x1 = max(0, x - crop_size // 2)
        x2 = min(W, x + crop_size // 2)

        crop = images[i:i+1, :, y1:y2, x1:x2]
        crop = F.interpolate(crop, size=(out_size, out_size), mode='bilinear', align_corners=False)

        crops.append(crop)

    return torch.cat(crops, dim=0)