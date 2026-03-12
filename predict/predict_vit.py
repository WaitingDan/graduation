import os
import sys
import json
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ==============================
# 解决模块路径
# ==============================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from models.vit_model import create_vit


# ==============================
# 设备
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("使用设备:", device)


# ==============================
# 路径配置
# ==============================

image_path = os.path.join(ROOT_DIR, "test_images/test_ship_01.jpg")

model_path = os.path.join(ROOT_DIR, "weights", "vit_best.pth")


class_json = os.path.join(ROOT_DIR, "class_indices.json")


# ==============================
# 检查文件
# ==============================

assert os.path.exists(image_path), f"找不到图片: {image_path}"
assert os.path.exists(model_path), f"找不到模型: {model_path}"
assert os.path.exists(class_json), f"找不到类别文件: {class_json}"


# ==============================
# 读取类别
# ==============================

with open(class_json, "r", encoding="utf-8") as f:
    class_dict = json.load(f)

num_classes = len(class_dict)

print("检测到类别数量:", num_classes)


# ==============================
# 创建模型
# ==============================

model = create_vit(num_classes)#vit


model.load_state_dict(torch.load(model_path, map_location=device))

model.to(device)

model.eval()


# ==============================
# 图像预处理
# ==============================

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


# ==============================
# 读取图片
# ==============================

img = Image.open(image_path).convert("RGB")

plt.imshow(img)
plt.axis("off")


img_tensor = transform(img)

img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)


# ==============================
# 预测
# ==============================

with torch.no_grad():

    outputs = model(img_tensor)

    probs = torch.softmax(outputs.squeeze(), dim=0)


# ==============================
# 获取预测结果
# ==============================

pred_idx = torch.argmax(probs).item()

pred_class = class_dict[str(pred_idx)]

pred_prob = probs[pred_idx].item()

title = f"{pred_class} ({pred_prob:.3f})"

plt.title(title)


# ==============================
# 输出概率
# ==============================

print("\nPrediction Results:\n")

for i in range(len(probs)):

    print(f"{class_dict[str(i)]:30s}: {probs[i].item():.4f}")


plt.show()