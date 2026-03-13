import torch
import os
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys

# make project root importable early so local imports work
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.metrics import evaluate_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

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


def run_evaluation(selected_models=None):
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

    model_names = selected_models if selected_models else list(models.keys())

    for name in model_names:

        if name not in models:
            continue

        model = models[name]["model"]
        weight_path = models[name]["weight"]

        model.load_state_dict(torch.load(weight_path, map_location=device))

        model.to(device)
        model.eval()

        y_true = []
        y_pred = []
        probs_all = []

        with torch.no_grad():

            for images, labels in loader:
                images = images.to(device)

                outputs = model(images)

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.tolist())
                probs_all.extend(probs.tolist())

        # 获取分类报告（文本）和混淆矩阵
        cm, report = evaluate_model(y_true, y_pred, class_names)

        # 保存预测 CSV 和分类报告
        try:
            out_dir = os.path.join(ROOT_DIR, 'outputs')
            os.makedirs(out_dir, exist_ok=True)
            # preds csv
            csv_path = os.path.join(out_dir, f'preds_{name}.csv')
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write('filepath,true_idx,true_name,pred_idx,pred_name,prob\n')
                for (path, _), t, p, prob_row in zip(dataset.samples, y_true, y_pred, probs_all):
                    f.write(f'{path},{t},{class_names[t]},{p},{class_names[p]},{prob_row[p]:.6f}\n')
            # report txt
            rpt_path = os.path.join(out_dir, f'report_{name}.txt')
            with open(rpt_path, 'w', encoding='utf-8') as f:
                f.write(report)
        except Exception as e:
            print('Failed to save preds/report:', e)

        # 保存混淆矩阵为图片
        try:
            out_dir = os.path.join(ROOT_DIR, 'outputs')
            os.makedirs(out_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
                   xticklabels=class_names, yticklabels=class_names,
                   ylabel='True label', xlabel='Predicted label',
                   title=f'Confusion Matrix: {name}')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                            color='white' if cm[i, j] > thresh else 'black')
            plt.tight_layout()
            fig_path = os.path.join(out_dir, f'confmat_{name}.png')
            fig.savefig(fig_path)
            plt.close(fig)
            print('Saved confusion matrix to', fig_path)
        except Exception as e:
            print('Failed to save confusion matrix image:', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=['resnet', 'vgg', 'vit'], default=None)
    args = parser.parse_args()
    run_evaluation(selected_models=args.models)


if __name__ == "__main__":
    main()