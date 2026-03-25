import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.vit_fusion_model import create_vit_global_local
from utils.common import build_default_transforms, get_device, write_class_indices
from utils.plot_results import plot_curve


def create_grad_scaler(enabled):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(enabled):
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda', enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine', help='dataset subdir under project root')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size tuned for ~24GB GPU')
    parser.add_argument('--num_workers', type=int, default=0, help='num workers (use 0 on Windows to avoid multiprocessing issues)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--accum_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--crop_size', type=int, default=112)
    parser.add_argument('--topk_patches', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--loss_w_global', type=float, default=0.3)
    parser.add_argument('--loss_w_local', type=float, default=0.3)
    parser.add_argument('--loss_w_fusion', type=float, default=0.4)
    parser.add_argument('--no_amp', action='store_true', help='disable mixed precision training')
    parser.add_argument('--no_pretrained', action='store_true', help='disable torchvision official pretrained weights')
    parser.add_argument('--weight_name', default='vit_fusion_best.pth', help='output weight file name under weights/')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='early stopping patience (epochs)')
    parser.add_argument('--freeze_encoder_layers', type=int, default=0, help='number of ViT encoder layers to freeze (0=no freeze)')
    parser.add_argument('--train_occlusion_mode', choices=['none', 'block', 'stripe', 'mixed'], default='none')
    parser.add_argument('--train_occlusion_level', choices=['light', 'medium', 'heavy'], default='light')
    parser.add_argument('--train_occlusion_p', type=float, default=0.0)
    return parser.parse_args()


def set_seed(seed):
    """Fix random seed for reproducibility (best effort)."""
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Note: Full deterministic behavior requires CUBLAS_WORKSPACE_CONFIG env var
    # For now, we keep allow_tf32=True for memory efficiency


def main():
    args = parse_args()

    if args.accum_steps < 1:
        raise ValueError('--accum_steps must be >= 1')
    
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f'Random seed set to: {args.seed}')

    device = get_device()
    print('Using device:', device)
    use_amp = (device.type == 'cuda') and (not args.no_amp)
    print('Use AMP:', use_amp)

    train_dir = os.path.join(ROOT_DIR, args.dataset_subdir, 'train')
    val_dir = os.path.join(ROOT_DIR, args.dataset_subdir, 'val')

    transform = build_default_transforms(
        img_size=224,
        train_occlusion_mode=args.train_occlusion_mode,
        train_occlusion_level=args.train_occlusion_level,
        train_occlusion_p=args.train_occlusion_p,
    )
    train_dataset = datasets.ImageFolder(train_dir, transform['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform['val'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print('Classes:', class_names)
    write_class_indices(class_names, os.path.join(ROOT_DIR, 'class_indices.json'))

    model = create_vit_global_local(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        crop_size=args.crop_size,
        out_size=224,
        topk_patches=args.topk_patches,
        dropout=args.dropout,
    ).to(device)
    
    # Optionally freeze encoder layers to reduce memory and speed up training
    if args.freeze_encoder_layers > 0:
        num_total_layers = 12
        freeze_count = min(args.freeze_encoder_layers, num_total_layers)
        for layer in model.global_model.encoder.layers[:freeze_count]:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in model.local_model.encoder.layers[:freeze_count]:
            for param in layer.parameters():
                param.requires_grad = False
        print(f'Froze {freeze_count} encoder layers in both branches')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = create_grad_scaler(enabled=use_amp)

    best_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0
    weight_path = os.path.join(ROOT_DIR, 'weights', args.weight_name)
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')

        model.train()
        running_loss = 0.0
        running_loss_g = 0.0
        running_loss_l = 0.0
        running_loss_f = 0.0
        correct = 0
        total = 0

        optimizer.zero_grad(set_to_none=True)

        for step, (images, labels) in enumerate(tqdm(train_loader, file=sys.stdout), start=1):
            images = images.to(device)
            labels = labels.to(device)

            with autocast_ctx(enabled=use_amp):
                global_logits, local_logits, fusion_logits = model(images)
                loss_g = criterion(global_logits, labels)
                loss_l = criterion(local_logits, labels)
                loss_f = criterion(fusion_logits, labels)

                loss = (
                    args.loss_w_global * loss_g
                    + args.loss_w_local * loss_l
                    + args.loss_w_fusion * loss_f
                )

            loss_for_backward = loss / args.accum_steps
            scaler.scale(loss_for_backward).backward()

            if (step % args.accum_steps == 0) or (step == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            running_loss_g += loss_g.item()
            running_loss_l += loss_l.item()
            running_loss_f += loss_f.item()
            preds = torch.argmax(fusion_logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / max(1, len(train_loader))
        train_loss_g = running_loss_g / max(1, len(train_loader))
        train_loss_l = running_loss_l / max(1, len(train_loader))
        train_loss_f = running_loss_f / max(1, len(train_loader))
        train_acc = correct / max(1, total)

        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, file=sys.stdout):
                images = images.to(device)
                labels = labels.to(device)

                with autocast_ctx(enabled=use_amp):
                    global_logits, local_logits, fusion_logits = model(images)
                    loss_g = criterion(global_logits, labels)
                    loss_l = criterion(local_logits, labels)
                    loss_f = criterion(fusion_logits, labels)
                    loss = (
                        args.loss_w_global * loss_g
                        + args.loss_w_local * loss_l
                        + args.loss_w_fusion * loss_f
                    )

                val_loss_sum += loss.item()
                preds = torch.argmax(fusion_logits, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_loss = val_loss_sum / max(1, len(val_loader))
        val_acc = correct / max(1, total)

        print(f'Train Loss: {train_loss:.4f}')
        print(f'Train Loss G/L/F: {train_loss_g:.4f} / {train_loss_l:.4f} / {train_loss_f:.4f}')
        print(f'Val Loss  : {val_loss:.4f}')
        print(f'Train Acc : {train_acc:.4f}')
        print(f'Val Acc   : {val_acc:.4f}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), weight_path)
            print(f'Saved Best Model (Best Val Acc: {best_acc:.4f})')
        else:
            early_stop_counter += 1
            print(f'Early Stop Counter: {early_stop_counter}/{args.early_stop_patience}')
            if early_stop_counter >= args.early_stop_patience:
                print(f'\nEarly stopping triggered at epoch {epoch + 1} (Best epoch: {best_epoch + 1})')
                break

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

    print('\nTraining Finished')
    print('Best Val Accuracy:', best_acc)

    plot_curve(train_loss_list, val_loss_list, 'loss', 'vit_fusion')
    plot_curve(train_acc_list, val_acc_list, 'accuracy', 'vit_fusion')


if __name__ == '__main__':
    main()
