"""
Train ViT example:

python train/train_vit.py \
    --dataset_subdir dataset/ship_fine \
    --epochs 30 \
    --batch_size 32 \
    --lr 4e-4 \
    --weight_name vit_best.pth

Notes:
- 默认超参与其它训练脚本保持一致：epochs=30, batch_size=32, lr=4e-4。
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ======================
# 项目根目录
# ======================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.plot_results import plot_curve
from utils.common import get_device, build_default_transforms, write_class_indices

from models.vit_model import create_vit


def create_grad_scaler(enabled):
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(enabled):
    return torch.cuda.amp.autocast(enabled=enabled)


def set_seed(seed: int, deterministic: bool = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def get_warmup_lr_scale(epoch, warmup_epochs=3):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine', help='dataset subdir under project root')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for fine-tuning')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='learning rate warmup epochs')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping max_norm (0 to disable)')
    parser.add_argument('--deterministic', action='store_true', help='enable deterministic cudnn mode')
    parser.add_argument('--weight_name', default='vit_best.pth', help='output weight file name under weights/')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--early_stop_patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--train_occlusion_mode', choices=['none', 'block', 'stripe', 'mixed'], default='none')
    parser.add_argument('--train_occlusion_level', choices=['light', 'medium', 'heavy'], default='medium')
    parser.add_argument('--train_occlusion_p', type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.accum_steps < 1:
        raise ValueError('--accum_steps must be >= 1')

    set_seed(args.seed, deterministic=args.deterministic)
    print(f'Random seed set to: {args.seed}')

    device = get_device()
    print("Using device:", device)
    use_amp = (device.type == 'cuda') and (not args.no_amp)
    print('Use AMP:', use_amp)

    # ======================
    # 数据路径
    # ======================

    train_dir = os.path.join(ROOT_DIR, args.dataset_subdir, "train")
    val_dir = os.path.join(ROOT_DIR, args.dataset_subdir, "val")

    # ======================
    # 数据增强
    # ======================

    data_transform = build_default_transforms(
        train_occlusion_mode=args.train_occlusion_mode,
        train_occlusion_level=args.train_occlusion_level,
        train_occlusion_p=args.train_occlusion_p,
    )

    # ======================
    # 数据集
    # ======================

    train_dataset = datasets.ImageFolder(train_dir, data_transform["train"])
    val_dataset = datasets.ImageFolder(val_dir, data_transform["val"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)

    # ======================
    # 保存类别索引
    # ======================

    write_class_indices(class_names, os.path.join(ROOT_DIR, "class_indices.json"))

    # ======================
    # 模型
    # ======================

    model = create_vit(num_classes, pretrained=not args.no_pretrained)
    print("Model: ViT")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = create_grad_scaler(enabled=use_amp)

    # ======================
    # 训练参数
    # ======================

    epochs = args.epochs
    best_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0

    weight_path = os.path.join(ROOT_DIR, "weights", args.weight_name)
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    # ======================
    # 训练循环
    # ======================

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        # Warmup: linearly scale LR for the first args.warmup_epochs
        lr_scale = get_warmup_lr_scale(epoch, getattr(args, 'warmup_epochs', 0))
        if epoch < getattr(args, 'warmup_epochs', 0):
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * lr_scale

        # ---------- train ----------

        model.train()

        train_bar = tqdm(train_loader, file=sys.stdout)

        running_loss = 0
        correct = 0
        total = 0

        optimizer.zero_grad(set_to_none=True)

        for step, (images, labels) in enumerate(train_bar, start=1):

            images = images.to(device)
            labels = labels.to(device)

            with autocast_ctx(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss_for_backward = loss / args.accum_steps
            scaler.scale(loss_for_backward).backward()

            # 梯度裁剪（如果启用）
            if getattr(args, 'grad_clip', 0) and args.grad_clip > 0:
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            if (step % args.accum_steps == 0) or (step == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # ---------- validation ----------

        model.eval()

        val_bar = tqdm(val_loader, file=sys.stdout)

        val_loss_sum = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_bar:

                images = images.to(device)
                labels = labels.to(device)

                with autocast_ctx(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss_sum += loss.item()

                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_loss = val_loss_sum / len(val_loader)
        val_acc = correct / total

        # ---------- 输出 ----------

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Train Acc : {train_acc:.4f}")
        print(f"Val Acc   : {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        # ---------- 保存模型 ----------

        if val_acc > best_acc:

            best_acc = val_acc
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), weight_path)

            print("Saved Best Model")
        else:
            early_stop_counter += 1
            print(f"Early Stop Counter: {early_stop_counter}/{args.early_stop_patience}")
            if early_stop_counter >= args.early_stop_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

    print("\nTraining Finished")
    print("Best Val Accuracy:", best_acc)

    # ======================
    # 绘制曲线
    # ======================

    plot_curve(train_loss_list, val_loss_list, "loss", "vit")
    plot_curve(train_acc_list, val_acc_list, "accuracy", "vit")


if __name__ == "__main__":
    main()