import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys

# make project root importable early so local imports work
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.metrics import evaluate_model

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from models.vit_model import create_vit
from models.resnet_model import create_resnet
from models.vgg_model import create_vgg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_path = os.path.join(ROOT_DIR, "dataset/ship_cls/test")


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(test_path, transform)
def main():
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    class_names = dataset.classes
    num_classes = len(class_names)

    models = {

        "vgg": {
            "model": create_vgg(num_classes),
            "weight": os.path.join(ROOT_DIR, "weights/vgg_best.pth")
        },

        "resnet": {
            "model": create_resnet(num_classes),
            "weight": os.path.join(ROOT_DIR, "weights/resnet_best.pth")
        },

        "vit": {
            "model": create_vit(num_classes),
            "weight": os.path.join(ROOT_DIR, "weights/vit_best.pth")
        }
    }

    for name in models:

        print("\n==============================")
        print("Testing Model:", name)
        print("==============================")

        model = models[name]["model"]
        weight_path = models[name]["weight"]

        model.load_state_dict(torch.load(weight_path, map_location=device))

        model.to(device)
        model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():

            for images, labels in loader:
                images = images.to(device)

                outputs = model(images)

                preds = torch.argmax(outputs, dim=1)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())

        cm = evaluate_model(y_true, y_pred, class_names)

        print("\nConfusion Matrix\n", cm)


if __name__ == "__main__":
    main()