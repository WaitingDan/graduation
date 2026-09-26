import torch
import os
import argparse
import random
import json
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import sys

# make project root importable early so local imports work
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.metrics import evaluate_model
from utils.common import build_default_transforms
import matplotlib.pyplot as plt

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from models.vit_model import create_vit
from models.resnet_model import create_resnet
from models.vgg_model import create_vgg
from models.vit_fusion_model import create_vit_global_local, ViTFusionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def _pick_first_existing(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def _load_weight_meta(weight_path):
    meta_path = weight_path + '.meta.json'
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, 'r', encoding='utf-8') as mf:
            return json.load(mf)
    except Exception:
        return {}


def _resolve_weight_path(model_name, seed=None, explicit_path=None):
    """Resolve checkpoint path with optional explicit override and seed-specific fallback."""
    if explicit_path:
        return explicit_path

    seed_suffix = f'_{int(seed)}' if seed is not None else None

    if model_name == 'resnet':
        candidates = []
        if seed_suffix is not None:
            candidates.append(os.path.join(ROOT_DIR, f'weights/resnet_best{seed_suffix}.pth'))
        candidates.extend([
            os.path.join(ROOT_DIR, 'weights/resnet_best.pth'),
            os.path.join(ROOT_DIR, 'weights/resnet_best_10.pth'),
            os.path.join(ROOT_DIR, 'weights/resnet_best_42.pth'),
        ])
        return _pick_first_existing(*candidates)

    if model_name == 'vgg':
        candidates = []
        if seed_suffix is not None:
            candidates.append(os.path.join(ROOT_DIR, f'weights/vgg_best{seed_suffix}.pth'))
        candidates.extend([
            os.path.join(ROOT_DIR, 'weights/vgg_best.pth'),
            os.path.join(ROOT_DIR, 'weights/vgg_best_10.pth'),
            os.path.join(ROOT_DIR, 'weights/vgg_best_42.pth'),
        ])
        return _pick_first_existing(*candidates)

    if model_name == 'vit':
        candidates = []
        if seed_suffix is not None:
            candidates.append(os.path.join(ROOT_DIR, f'weights/vit_best{seed_suffix}.pth'))
        candidates.extend([
            os.path.join(ROOT_DIR, 'weights/vit_best.pth'),
            os.path.join(ROOT_DIR, 'weights/vit_best_10.pth'),
            os.path.join(ROOT_DIR, 'weights/vit_best_42.pth'),
        ])
        return _pick_first_existing(*candidates)

    if model_name == 'vit_fusion':
        candidates = []
        if seed_suffix is not None:
            candidates.append(os.path.join(ROOT_DIR, f'weights/ablation/vit_fusion_abc_best{seed_suffix}.pth'))
        candidates.append(os.path.join(ROOT_DIR, 'weights/ablation/vit_fusion_abc_best.pth'))
        return _pick_first_existing(*candidates)

    raise ValueError(f'Unknown model for weight resolution: {model_name}')


def _build_vit_fusion_model(num_classes, weight_path=None, fusion_modules_override=None):
    meta = _load_weight_meta(weight_path) if weight_path is not None else {}

    architecture = str(meta.get('architecture', 'single_branch_part_aware'))
    topk = int(meta.get('topk_patches', 3))
    attn_rollout_layers = int(meta.get('attn_rollout_layers', 3))
    part_gate_init = float(meta.get('part_gate_init', meta.get('local_gate_init', 1.0)))
    part_dropout_p = float(meta.get('part_dropout_p', 0.15))
    use_part_self_attention = bool(meta.get('use_part_self_attention', meta.get('use_local_self_attention', False)))
    attn_temperature = float(meta.get('attn_temperature', 0.7))
    share_backbone = bool(meta.get('share_backbone', False))
    use_cross_attention = bool(meta.get('use_cross_attention', False))
    fusion_modules = str(meta.get('fusion_modules', 'abc')).lower()
    if fusion_modules_override is not None:
        # Ablation mode should strictly follow CLI override instead of saved meta flags.
        fusion_modules = str(fusion_modules_override).lower()
        enable_module_a = 'a' in fusion_modules
        enable_module_b = 'b' in fusion_modules
        enable_module_c = 'c' in fusion_modules
    else:
        enable_module_a = bool(meta.get('enable_module_a', 'a' in fusion_modules))
        enable_module_b = bool(meta.get('enable_module_b', 'b' in fusion_modules))
        enable_module_c = bool(meta.get('enable_module_c', 'c' in fusion_modules))

    if architecture in ('single_branch_part_aware', 'vit_fusion_fixed'):
        return ViTFusionModel(
            num_classes=num_classes,
            topk=topk,
            pretrained=False,
            attn_rollout_layers=attn_rollout_layers,
            use_part_self_attention=use_part_self_attention,
            part_gate_init=part_gate_init,
            part_dropout_p=part_dropout_p,
            attn_temperature=attn_temperature,
            enable_module_a=enable_module_a,
            enable_module_b=enable_module_b,
            enable_module_c=enable_module_c,
        )

    # Fallback to the backward-compatible constructor for older checkpoints.
    return create_vit_global_local(
        num_classes=num_classes,
        pretrained=False,
        topk_patches=topk,
        attn_rollout_layers=attn_rollout_layers,
        use_cross_attention=use_cross_attention,
        use_local_self_attention=use_part_self_attention,
        local_gate_init=part_gate_init,
        part_dropout_p=part_dropout_p,
        share_backbone=share_backbone,
    )


def run_evaluation(
    selected_models=None,
    dataset_subdir='dataset/ship_fine',
    test_split='test',
    batch_size=32,
    num_workers=2,
    eval_occlusion_mode='none',
    eval_occlusion_level='light',
    eval_occlusion_p=0.0,
    output_subdir='outputs/evaluation/default_eval',
    file_suffix='',
    seed=None,
    vit_fusion_weight=None,
):
    set_seed(seed)

    test_path = os.path.join(ROOT_DIR, dataset_subdir, test_split)
    transform = build_default_transforms(
        val_occlusion_mode=eval_occlusion_mode,
        val_occlusion_level=eval_occlusion_level,
        val_occlusion_p=eval_occlusion_p,
    )["val"]
    dataset = datasets.ImageFolder(test_path, transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    class_names = dataset.classes
    num_classes = len(class_names)
    class_index_map = {str(i): cls for i, cls in enumerate(class_names)}

    try:
        out_dir = os.path.join(ROOT_DIR, output_subdir)
        os.makedirs(out_dir, exist_ok=True)
        class_map_path = os.path.join(out_dir, 'class_indices_eval.json')
        with open(class_map_path, 'w', encoding='utf-8') as f:
            json.dump(class_index_map, f, ensure_ascii=False, indent=2)
        print('Saved class index mapping to', class_map_path)
    except Exception as e:
        print('Failed to save class index mapping:', e)

    models = {

        "vgg": {
            "model": create_vgg(num_classes),
            "weight": _resolve_weight_path('vgg', seed=seed)
        },

        "resnet": {
            "model": create_resnet(num_classes),
            "weight": _resolve_weight_path('resnet', seed=seed)
        },

        "vit": {
            "model": create_vit(num_classes),
            "weight": _resolve_weight_path('vit', seed=seed)
        },

        "vit_fusion": {
            # instantiate vit_fusion according to saved checkpoint metadata when available
            "model": None,
            "weight": _resolve_weight_path('vit_fusion', seed=seed, explicit_path=vit_fusion_weight)
        },
    }

    model_names = selected_models if selected_models else list(models.keys())

    all_metrics = []

    for name in model_names:

        if name not in models:
            continue

        # display label used in plot titles/annotations (map vit_fusion -> AG-ViT)
        display_label = 'AG-ViT' if name == 'vit_fusion' else name

        # prepare model instance for evaluation; for vit_fusion try to honor saved metadata
        weight_path = models[name]["weight"]
        model = models[name]["model"]
        if name == 'vit_fusion':
            model = _build_vit_fusion_model(num_classes=num_classes, weight_path=weight_path)

        if not os.path.exists(weight_path):
            print(f'Skip {name}: weight not found -> {weight_path}')
            continue

        print(f'Evaluating {name} with weight: {weight_path}')

        state_dict = torch.load(weight_path, map_location=device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict, strict=False)

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
                    selected = None
                    for item in outputs:
                        if isinstance(item, torch.Tensor) and item.ndim == 2 and item.size(1) == num_classes:
                            selected = item
                            break
                    outputs = selected if selected is not None else outputs[0]

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.tolist())
                probs_all.extend(probs.tolist())

        # 获取分类报告（文本）、混淆矩阵，以及每类指标
        cm, report, per_class_recall, per_class_overall_acc, support, macro_f1, balanced_acc = evaluate_model(y_true, y_pred, class_names)

        # 保存预测 CSV 和分类报告
        try:
            out_dir = os.path.join(ROOT_DIR, output_subdir)
            os.makedirs(out_dir, exist_ok=True)
            # preds csv
            csv_path = os.path.join(out_dir, f'preds_{name}{file_suffix}.csv')
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write('filepath,true_idx,true_name,pred_idx,pred_name,prob\n')
                for (path, _), t, p, prob_row in zip(dataset.samples, y_true, y_pred, probs_all):
                    f.write(f'{path},{t},{class_names[t]},{p},{class_names[p]},{prob_row[p]:.6f}\n')
            # report txt（在原文本分类报告后追加每类准确率）
            rpt_path = os.path.join(out_dir, f'report_{name}{file_suffix}.txt')
            with open(rpt_path, 'w', encoding='utf-8') as f:
                f.write(report)
                f.write('\n\nSummary metrics:\n')
                f.write(f'macro_f1={macro_f1:.4f}\n')
                f.write(f'balanced_accuracy={balanced_acc:.4f}\n')
                f.write('\n\nPer-class recall (TP / true_samples) and overall per-class accuracy:\n')
                for i, cls in enumerate(class_names):
                    f.write(f"{i}: {cls}  recall={per_class_recall[i]:.4f}  overall_acc={per_class_overall_acc[i]:.4f}  support={int(support[i])}\n")

            # per-class CSV
            per_csv = os.path.join(out_dir, f'per_class_{name}{file_suffix}.csv')
            with open(per_csv, 'w', encoding='utf-8') as f:
                f.write('class_idx,class_name,recall,overall_acc,support\n')
                for i, cls in enumerate(class_names):
                    f.write(f"{i},{cls},{per_class_recall[i]:.6f},{per_class_overall_acc[i]:.6f},{int(support[i])}\n")

            print(f'{name}: macro_f1={macro_f1:.4f}, balanced_accuracy={balanced_acc:.4f}')

            all_metrics.append({
                'model': name,
                'macro_f1': float(macro_f1),
                'balanced_accuracy': float(balanced_acc),
                'dataset_subdir': dataset_subdir,
                'test_split': test_split,
                'eval_occlusion_mode': eval_occlusion_mode,
                'eval_occlusion_level': eval_occlusion_level,
                'eval_occlusion_p': float(eval_occlusion_p),
                'seed': seed,
                'output_subdir': output_subdir,
            })

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
                excel_path = os.path.join(out_dir, f'per_class_{name}{file_suffix}.xlsx')
                df.to_excel(excel_path, index=False)
                print('Saved per-class Excel to', excel_path)
            except Exception as e:
                print('Could not save Excel (pandas missing or error):', e)

            # 绘制每类 recall 的柱状图并保存为图片，便于查看
            try:
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                indices = list(range(len(class_names)))
                ax2.bar(indices, per_class_recall, color='tab:blue')
                ax2.set_xticks(indices)
                ax2.set_xticklabels([str(i) for i in indices], rotation=0)
                ax2.set_ylabel('Recall')
                ax2.set_xlabel('Class Index')
                display_label = 'AG-ViT' if name == 'vit_fusion' else name
                ax2.set_title(f'Per-class Recall: {display_label} (indexed labels)')
                plt.tight_layout()
                img_path = os.path.join(out_dir, f'per_class_{name}{file_suffix}.png')
                fig2.savefig(img_path)
                plt.close(fig2)
                print('Saved per-class bar chart to', img_path)
            except Exception as e:
                print('Failed to save per-class bar chart:', e)
        except Exception as e:
            print('Failed to save preds/report:', e)

        # 保存混淆矩阵为图片
        try:
            out_dir = os.path.join(ROOT_DIR, output_subdir)
            os.makedirs(out_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            # Use numeric indices as tick labels to avoid overcrowding;
            # full mapping is saved to `class_indices_eval.json`.
            indices = list(range(len(class_names)))
            ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
                   xticklabels=indices, yticklabels=indices,
                   ylabel='True label (index)', xlabel='Predicted label (index)',
                   title=f'Confusion Matrix: {display_label}')
            plt.setp(ax.get_xticklabels(), rotation=0)
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                            color='white' if cm[i, j] > thresh else 'black')
            plt.tight_layout()
            fig_path = os.path.join(out_dir, f'confmat_{name}{file_suffix}.png')
            fig.savefig(fig_path)
            plt.close(fig)
            print('Saved confusion matrix to', fig_path)
        except Exception as e:
            print('Failed to save confusion matrix image:', e)

    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=['resnet', 'vgg', 'vit', 'vit_fusion'], default=None)
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine')
    parser.add_argument('--test_split', default='test')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--eval_occlusion_mode', choices=['none', 'block', 'stripe', 'mixed'], default='none')
    parser.add_argument('--eval_occlusion_level', choices=['light', 'medium', 'heavy'], default='light')
    parser.add_argument('--eval_occlusion_p', type=float, default=0.0)
    parser.add_argument('--output_subdir', default='outputs/evaluation/default_eval')
    parser.add_argument('--file_suffix', default='')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--vit_fusion_weight', default=None, help='optional explicit checkpoint path for vit_fusion')
    args = parser.parse_args()
    run_evaluation(
        selected_models=args.models,
        dataset_subdir=args.dataset_subdir,
        test_split=args.test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_occlusion_mode=args.eval_occlusion_mode,
        eval_occlusion_level=args.eval_occlusion_level,
        eval_occlusion_p=args.eval_occlusion_p,
        output_subdir=args.output_subdir,
        file_suffix=args.file_suffix,
        seed=args.seed,
        vit_fusion_weight=args.vit_fusion_weight,
    )


if __name__ == "__main__":
    main()