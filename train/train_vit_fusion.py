import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.vit_fusion_model import ViTFusionModel, get_attention_map
from utils.common import build_default_transforms, get_device, write_class_indices
from utils.plot_results import plot_curve


def create_grad_scaler(enabled):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(enabled):
    try:
        return torch.amp.autocast('cuda', enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine', help='dataset subdir under project root')
    parser.add_argument('--batch_size', type=int, default=32, help='default aligned with other models for fair comparison')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr_backbone', type=float, default=4e-4)
    parser.add_argument('--lr_head', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--topk_patches', type=int, default=2, help='Top-K local crops, recommended 2')
    parser.add_argument('--local_gate_init', type=float, default=1.0, help='initial value of adaptive local gate')
    parser.add_argument('--ablate_local', action='store_true', help='disable local branch for ablation')
    parser.add_argument('--ablate_attention_guidance', action='store_true', help='disable attention-guided localizer')
    parser.add_argument('--ablate_cross_attention', action='store_true', help='disable cross-attention fusion block')
    parser.add_argument('--no_amp', action='store_true', help='disable mixed precision training')
    parser.add_argument('--no_pretrained', action='store_true', help='disable pretrained timm weights')
    parser.add_argument('--weight_name', default='vit_fusion_best.pth', help='output weight file name under weights/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--train_occlusion_mode', choices=['none', 'block', 'stripe', 'mixed'], default='none')
    parser.add_argument('--train_occlusion_level', choices=['light', 'medium', 'heavy'], default='light')
    parser.add_argument('--train_occlusion_p', type=float, default=0.0)
    return parser.parse_args()


def set_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def dynamic_loss_weights(epoch):
    if epoch < 10:
        return 0.6, 0.3, 0.1
    return 0.2, 0.2, 0.6


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, use_amp, scaler):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    w_g, w_l, w_f = dynamic_loss_weights(epoch)
    loop = tqdm(loader, file=sys.stdout)
    gate_running = 0.0
    gate_count = 0

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        attn_map = None
        if model.use_local_branch and model.use_attention_guidance:
            with torch.no_grad():
                attn_map = get_attention_map(model.global_model, images)

        with autocast_ctx(enabled=use_amp):
            outputs, global_feat, local_feat, aux = model(images, attn_map)

            loss_f = criterion(outputs, labels)
            logits_g = model.classifier(global_feat)
            loss_g = criterion(logits_g, labels)

            if local_feat is not None:
                bsz, k_parts, feat_dim = local_feat.shape
                logits_l = model.classifier(local_feat.view(bsz * k_parts, feat_dim))
                loss_l = criterion(logits_l, labels.repeat_interleave(k_parts))
            else:
                loss_l = loss_f.new_zeros(())

            if local_feat is not None:
                w_g_eff, w_l_eff, w_f_eff = w_g, w_l, w_f
            else:
                denom = max(1e-6, (w_g + w_f))
                w_g_eff, w_l_eff, w_f_eff = w_g / denom, 0.0, w_f / denom

            loss = w_g_eff * loss_g + w_l_eff * loss_l + w_f_eff * loss_f

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        preds = torch.argmax(outputs, dim=1)
        total_correct += int((preds == labels).sum().item())
        total_count += int(labels.size(0))

        gate_val = aux.get('local_gate', None)
        if gate_val is not None:
            gate_running += float(gate_val.detach().mean().item())
            gate_count += 1

        loop.set_description(f'Epoch [{epoch + 1}]')
        gate_info = gate_running / gate_count if gate_count > 0 else 0.0
        loop.set_postfix(loss=f'{loss.item():.4f}', w=f'({w_g_eff:.1f},{w_l_eff:.1f},{w_f_eff:.1f})', gate=f'{gate_info:.3f}')

    avg_loss = total_loss / max(1, len(loader))
    avg_acc = total_correct / max(1, total_count)
    return avg_loss, avg_acc


def validate(model, loader, criterion, device, epoch, use_amp):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    w_g, w_l, w_f = dynamic_loss_weights(epoch)

    with torch.no_grad():
        for images, labels in tqdm(loader, file=sys.stdout):
            images = images.to(device)
            labels = labels.to(device)

            attn_map = None
            if model.use_local_branch and model.use_attention_guidance:
                attn_map = get_attention_map(model.global_model, images)

            with autocast_ctx(enabled=use_amp):
                outputs, global_feat, local_feat, _aux = model(images, attn_map)

                loss_f = criterion(outputs, labels)
                logits_g = model.classifier(global_feat)
                loss_g = criterion(logits_g, labels)

                if local_feat is not None:
                    bsz, k_parts, feat_dim = local_feat.shape
                    logits_l = model.classifier(local_feat.view(bsz * k_parts, feat_dim))
                    loss_l = criterion(logits_l, labels.repeat_interleave(k_parts))
                else:
                    loss_l = loss_f.new_zeros(())

                if local_feat is not None:
                    w_g_eff, w_l_eff, w_f_eff = w_g, w_l, w_f
                else:
                    denom = max(1e-6, (w_g + w_f))
                    w_g_eff, w_l_eff, w_f_eff = w_g / denom, 0.0, w_f / denom

                loss = w_g_eff * loss_g + w_l_eff * loss_l + w_f_eff * loss_f

            total_loss += float(loss.item())
            preds = torch.argmax(outputs, dim=1)
            total_correct += int((preds == labels).sum().item())
            total_count += int(labels.size(0))

    avg_loss = total_loss / max(1, len(loader))
    avg_acc = total_correct / max(1, total_count)
    return avg_loss, avg_acc


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    device = get_device()
    use_amp = (device.type == 'cuda') and (not args.no_amp)

    print(f'Random seed: {args.seed}')
    print('Using device:', device)
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

    model = ViTFusionModel(
        num_classes=num_classes,
        topk=args.topk_patches,
        pretrained=not args.no_pretrained,
        use_local_branch=not args.ablate_local,
        use_attention_guidance=not args.ablate_attention_guidance,
        use_cross_attention=not args.ablate_cross_attention,
        local_gate_init=args.local_gate_init,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        [
            {'params': model.global_model.parameters(), 'lr': args.lr_backbone},
            {'params': model.local_model.parameters(), 'lr': args.lr_backbone},
            {'params': model.fusion.parameters(), 'lr': args.lr_head},
            {'params': model.classifier.parameters(), 'lr': args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = create_grad_scaler(enabled=use_amp)

    best_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    weight_path = os.path.join(ROOT_DIR, 'weights', args.weight_name)
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, use_amp, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, use_amp)

        scheduler.step()

        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss  : {val_loss:.4f}')
        print(f'Train Acc : {train_acc:.4f}')
        print(f'Val Acc   : {val_acc:.4f}')
        print(f'Backbone LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Head LR    : {optimizer.param_groups[2]["lr"]:.6f}')

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
                print(f'Early stopping triggered at epoch {epoch + 1} (Best epoch: {best_epoch + 1})')
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
