import torch
import os
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets
import sys

# make project root importable early so local imports work
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.metrics import evaluate_model
from utils.common import build_default_transforms
import matplotlib.pyplot as plt
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from models.vit_model import create_vit
from models.resnet_model import create_resnet
from models.vgg_model import create_vgg
from models.vit_fusion_model import create_vit_global_local

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pick_first_existing(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def run_evaluation(selected_models=None, dataset_subdir='dataset/ship_fine', test_split='test', batch_size=32, num_workers=2):
    test_path = os.path.join(ROOT_DIR, dataset_subdir, test_split)
    transform = build_default_transforms()["val"]
    dataset = datasets.ImageFolder(test_path, transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    class_names = dataset.classes
    num_classes = len(class_names)

    models = {

        "vgg": {
            "model": create_vgg(num_classes),
            "weight": _pick_first_existing(
                os.path.join(ROOT_DIR, "weights/vgg_best.pth"),
                os.path.join(ROOT_DIR, "weights/vgg_best_10.pth"),
                os.path.join(ROOT_DIR, "weights/vgg_best_42.pth"),
            )
        },

        "resnet": {
            "model": create_resnet(num_classes),
            "weight": _pick_first_existing(
                os.path.join(ROOT_DIR, "weights/resnet_best.pth"),
                os.path.join(ROOT_DIR, "weights/resnet_best_10.pth"),
                os.path.join(ROOT_DIR, "weights/resnet_best_42.pth"),
            )
        },

        "vit": {
            "model": create_vit(num_classes),
            "weight": _pick_first_existing(
                os.path.join(ROOT_DIR, "weights/vit_best.pth"),
                os.path.join(ROOT_DIR, "weights/vit_best_10.pth"),
                os.path.join(ROOT_DIR, "weights/vit_best_42.pth"),
            )
        },

        "vit_fusion": {
            "model": create_vit_global_local(num_classes=num_classes, pretrained=False),
            "weight": os.path.join(ROOT_DIR, "weights/vit_fusion_best.pth")
        }
    }

    model_names = selected_models if selected_models else list(models.keys())

    for name in model_names:

        if name not in models:
            continue

        model = models[name]["model"]
        weight_path = models[name]["weight"]

        if not os.path.exists(weight_path):
            print(f'Skip {name}: weight not found -> {weight_path}')
            continue

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
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[-1]

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.tolist())
                probs_all.extend(probs.tolist())

        # 获取分类报告（文本）、混淆矩阵，以及每类指标
        cm, report, per_class_recall, per_class_overall_acc, support, macro_f1, balanced_acc = evaluate_model(y_true, y_pred, class_names)

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
            # report txt（在原文本分类报告后追加每类准确率）
            rpt_path = os.path.join(out_dir, f'report_{name}.txt')
            with open(rpt_path, 'w', encoding='utf-8') as f:
                f.write(report)
                f.write('\n\nSummary metrics:\n')
                f.write(f'macro_f1={macro_f1:.4f}\n')
                f.write(f'balanced_accuracy={balanced_acc:.4f}\n')
                f.write('\n\nPer-class recall (TP / true_samples) and overall per-class accuracy:\n')
                for i, cls in enumerate(class_names):
                    f.write(f"{i}: {cls}  recall={per_class_recall[i]:.4f}  overall_acc={per_class_overall_acc[i]:.4f}  support={int(support[i])}\n")

            # per-class CSV
            per_csv = os.path.join(out_dir, f'per_class_{name}.csv')
            with open(per_csv, 'w', encoding='utf-8') as f:
                f.write('class_idx,class_name,recall,overall_acc,support\n')
                for i, cls in enumerate(class_names):
                    f.write(f"{i},{cls},{per_class_recall[i]:.6f},{per_class_overall_acc[i]:.6f},{int(support[i])}\n")

            print(f'{name}: macro_f1={macro_f1:.4f}, balanced_accuracy={balanced_acc:.4f}')

            # 尝试导出为 Excel（如果安装了 pandas）
            try:
                import pandas as pd
                df = pd.DataFrame({
                    'class_idx': list(range(len(class_names))),
                    'class_name': class_names,
                    'recall': [float(x) for x in per_class_recall],
                    'overall_acc': [float(x) for x in per_class_overall_acc],
                    'support': [int(x) for x in support]
                })
                excel_path = os.path.join(out_dir, f'per_class_{name}.xlsx')
                df.to_excel(excel_path, index=False)
                print('Saved per-class Excel to', excel_path)
            except Exception as e:
                print('Could not save Excel (pandas missing or error):', e)

            # 绘制每类 recall 的柱状图并保存为图片，便于查看
            try:
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                indices = range(len(class_names))
                ax2.bar(indices, per_class_recall, color='tab:blue')
                ax2.set_xticks(indices)
                ax2.set_xticklabels(class_names, rotation=45, ha='right')
                ax2.set_ylabel('Recall')
                ax2.set_xlabel('Class')
                ax2.set_title(f'Per-class Recall: {name}')
                plt.tight_layout()
                img_path = os.path.join(out_dir, f'per_class_{name}.png')
                fig2.savefig(img_path)
                plt.close(fig2)
                print('Saved per-class bar chart to', img_path)
            except Exception as e:
                print('Failed to save per-class bar chart:', e)
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
    parser.add_argument('--models', nargs='+', choices=['resnet', 'vgg', 'vit', 'vit_fusion'], default=None)
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine')
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    run_evaluation(
        selected_models=args.models,
        dataset_subdir=args.dataset_subdir,
        test_split=args.test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()