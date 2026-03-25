"""
Train ResNet example:

python train/train_resnet.py \
    --dataset_subdir dataset/ship_fine \
    --epochs 15 \
    --batch_size 32 \
    --lr 4e-4 \
    --weight_name resnet_best.pth

Notes:
- 为了保证与其它模型的公平比较，默认使用 epochs=30, batch_size=32, lr=4e-4。
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
 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.plot_results import plot_curve
from utils.common import get_device, build_default_transforms, write_class_indices

from models.resnet_model import create_resnet


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine', help='dataset subdir under project root')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=4e-4, help='learning rate for fine-tuning')
    parser.add_argument('--weight_name', default='resnet_best.pth', help='output weight file name under weights/')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--train_occlusion_mode', choices=['none', 'block', 'stripe', 'mixed'], default='none')
    parser.add_argument('--train_occlusion_level', choices=['light', 'medium', 'heavy'], default='light')
    parser.add_argument('--train_occlusion_p', type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    print(f'Random seed set to: {args.seed}')

    device = get_device()
    print("Using device:", device)

    train_dir = os.path.join(ROOT_DIR, args.dataset_subdir, "train")
    val_dir = os.path.join(ROOT_DIR, args.dataset_subdir, "val")

    transform = build_default_transforms(
        train_occlusion_mode=args.train_occlusion_mode,
        train_occlusion_level=args.train_occlusion_level,
        train_occlusion_p=args.train_occlusion_p,
    )

    train_dataset = datasets.ImageFolder(train_dir, transform["train"])
    val_dataset = datasets.ImageFolder(val_dir, transform["val"])

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    write_class_indices(class_names, os.path.join(ROOT_DIR, "class_indices.json"))

    model = create_resnet(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

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

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        model.train()

        train_bar = tqdm(train_loader, file=sys.stdout)

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_bar:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        model.eval()

        val_bar = tqdm(val_loader, file=sys.stdout)

        val_loss_sum = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_bar:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss_sum += loss.item()

                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_loss = val_loss_sum / len(val_loader)
        val_acc = correct / total

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Train Acc : {train_acc:.4f}")
        print(f"Val Acc   : {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

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

    plot_curve(train_loss_list, val_loss_list, "loss", "resnet")
    plot_curve(train_acc_list, val_acc_list, "accuracy", "resnet")


if __name__ == "__main__":
    main()