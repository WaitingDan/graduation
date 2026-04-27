"""
train_vit_fusion.py  —  修复版训练脚本

核心改动（相较原版）：
  1. 损失权重调度    → 去除突变式 dynamic_loss_weights，改用线性渐变，避免收敛震荡
  2. part_tokens 维度 → 与修复后的 ViTFusionModel 返回的 [B, K, D] 对齐
  3. 局部损失        → 对 Top-K patch token 各自分类后取均值，语义明确
  4. 熵正则化        → attn_sparse_weight 默认调低至 0.005，避免过度抑制有效注意力
  5. 梯度裁剪        → 添加 max_norm=1.0 的梯度裁剪，稳定训练
  6. Warmup 学习率   → 前 3 个 epoch 线性 warmup，缓解预训练权重被破坏
  7. 超参默认值      → 与修复后的模型默认值对齐

用法（与原版完全兼容）：
  python train/train_vit_fusion.py \
      --dataset_subdir dataset/ship_fine \
      --epochs 50 \
      --batch_size 32 \
      --lr_backbone 2e-5 \
      --lr_head 2e-4 \
      --weight_name vit_fusion_fixed.pth
"""

import argparse
import json
import math
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

from models.vit_fusion_model import ViTFusionModel
from utils.common import build_default_transforms, get_device, write_class_indices
from utils.plot_results import plot_curve


# ---------------------------------------------------------------------------
# AMP 兼容
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def set_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


# ---------------------------------------------------------------------------
# 修复一：线性渐变损失权重，替换突变式调度
# ---------------------------------------------------------------------------

def get_loss_weights(epoch, total_epochs, warmup_epochs=3):
    """
    线性渐变策略：
      - warmup 阶段：w_g 高，让 global 分支先收敛
      - 渐变阶段：w_g 下降，w_f 上升，逐渐过渡到融合损失主导
      - 稳定阶段：w_g=0.25, w_l=0.25, w_f=0.50

    避免了原版 epoch=10 时权重突变导致的收敛震荡。
    """
    if epoch < warmup_epochs:
        # warmup：global 分支主导，让骨干快速适应
        return 0.6, 0.2, 0.2

    progress = min((epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs), 1.0)
    w_g = 0.6 - 0.35 * progress   # 0.60 → 0.25
    w_l = 0.2 + 0.05 * progress   # 0.20 → 0.25
    w_f = 0.2 + 0.30 * progress   # 0.20 → 0.50
    return w_g, w_l, w_f


def get_warmup_lr_scale(epoch, warmup_epochs=3):
    """前 warmup_epochs 个 epoch 线性升温。"""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='ViT Fusion Model Training (Fixed)')
    parser.add_argument('--dataset_subdir', default='dataset/ship_fine')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30,
                        help='默认 30，与其它训练脚本保持一致')
    # 修复：backbone 和 head 使用不同的、更合理的学习率
    parser.add_argument('--lr_backbone', type=float, default=2e-5,
                        help='骨干微调学习率（建议比 head 小 10x）')
    parser.add_argument('--lr_head', type=float, default=2e-4,
                        help='新增 head 模块学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='小数据集建议开启 label smoothing')
    # 模型超参（与修复后的 ViTFusionModel 默认值对齐）
    parser.add_argument('--topk_patches', type=int, default=6,
                        help='Top-K patch token 数量（建议 6）')
    parser.add_argument('--attn_rollout_layers', type=int, default=4,
                        help='Rollout 使用的最后 N 层')
    parser.add_argument('--part_gate_init', type=float, default=0.0,
                        help='gate 初始化（修复：从 -1.0 改为 0.0）')
    parser.add_argument('--part_dropout_p', type=float, default=0.1,
                        help='注意力 dropout 概率')
    parser.add_argument('--attn_temperature', type=float, default=1.0,
                        help='注意力 softmax 温度（修复：从 0.7 改为 1.0）')
    parser.add_argument('--entropy_penalty', type=float, default=0.5,
                        help='遮挡感知熵惩罚系数')
    # 修复：熵正则化权重调低，避免过度抑制有效注意力
    parser.add_argument('--attn_sparse_weight', type=float, default=0.005,
                        help='注意力熵正则化权重（修复：从 0.02 降至 0.005）')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='学习率 warmup epoch 数')
    # 训练控制
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪 max_norm（0 表示不裁剪）')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--weight_name', default='vit_fusion_best.pth')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=15,
                        help='建议 15，给 local 分支充足收敛时间')
    # 遮挡数据增强
    parser.add_argument('--train_occlusion_mode',
                        choices=['none', 'block', 'stripe', 'mixed'], default='none',
                        help='遮挡增强模式（默认无遮挡训练）')
    parser.add_argument('--train_occlusion_level',
                        choices=['light', 'medium', 'heavy'], default='medium')
    parser.add_argument('--train_occlusion_p', type=float, default=0.0,
                        help='遮挡增强概率（默认 0.0）')
    parser.add_argument('--fusion_modules', type=str, default='abc',
                        help='Fusion 模块开关，默认 abc（全开）；可选 a、ab、abc、bc 等组合')
    # 兼容旧参数
    parser.add_argument('--use_part_self_attention', action='store_true', default=False)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 训练 one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, criterion, device,
    epoch, total_epochs, use_amp, scaler,
    attn_sparse_weight, grad_clip, warmup_epochs,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    w_g, w_l, w_f = get_loss_weights(epoch, total_epochs, warmup_epochs)
    loop = tqdm(loader, file=sys.stdout)
    gate_running = 0.0
    gate_count = 0

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(enabled=use_amp):
            outputs, global_feat, part_tokens, aux = model(images)

            # 融合输出损失
            loss_f = criterion(outputs, labels)

            # 全局特征辅助损失
            logits_g = model.classifier(global_feat)
            loss_g = criterion(logits_g, labels)

            # 局部分支监督改为 pooled local feature + 独立 local head，避免干扰主分类头。
            local_feat = aux.get('local_feat', None)
            if local_feat is None and part_tokens is not None and part_tokens.dim() == 3:
                local_feat = part_tokens.mean(dim=1)

            if local_feat is not None:
                local_classifier = getattr(model, 'local_classifier', model.classifier)
                logits_l = local_classifier(local_feat)
                loss_l = criterion(logits_l, labels)
            else:
                loss_l = loss_f.new_zeros(())

            # 注意力熵正则化（保持分布不过于尖锐）
            attn_ent = aux.get('attn_entropy', loss_f.new_zeros(()))
            sparse_reg = float(attn_sparse_weight) * attn_ent

            loss = w_g * loss_g + w_l * loss_l + w_f * loss_f + sparse_reg

        scaler.scale(loss).backward()

        # 梯度裁剪
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        preds = torch.argmax(outputs, dim=1)
        total_correct += int((preds == labels).sum().item())
        total_count += int(labels.size(0))

        gate_val = aux.get('part_gate', None)
        if gate_val is not None:
            gate_running += float(gate_val.detach().mean().item())
            gate_count += 1

        loop.set_description(f'Epoch [{epoch + 1}]')
        gate_info = gate_running / gate_count if gate_count > 0 else 0.0
        loop.set_postfix(
            loss=f'{loss.item():.4f}',
            w=f'({w_g:.2f},{w_l:.2f},{w_f:.2f})',
            gate=f'{gate_info:.3f}',
        )

    return total_loss / max(1, len(loader)), total_correct / max(1, total_count)


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

def validate(
    model, loader, criterion, device,
    epoch, total_epochs, use_amp, attn_sparse_weight, warmup_epochs,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    w_g, w_l, w_f = get_loss_weights(epoch, total_epochs, warmup_epochs)

    with torch.no_grad():
        for images, labels in tqdm(loader, file=sys.stdout):
            images = images.to(device)
            labels = labels.to(device)

            with autocast_ctx(enabled=use_amp):
                outputs, global_feat, part_tokens, aux = model(images)

                loss_f = criterion(outputs, labels)
                logits_g = model.classifier(global_feat)
                loss_g = criterion(logits_g, labels)

                local_feat = aux.get('local_feat', None)
                if local_feat is None and part_tokens is not None and part_tokens.dim() == 3:
                    local_feat = part_tokens.mean(dim=1)

                if local_feat is not None:
                    local_classifier = getattr(model, 'local_classifier', model.classifier)
                    logits_l = local_classifier(local_feat)
                    loss_l = criterion(logits_l, labels)
                else:
                    loss_l = loss_f.new_zeros(())

                attn_ent = aux.get('attn_entropy', loss_f.new_zeros(()))
                sparse_reg = float(attn_sparse_weight) * attn_ent
                loss = w_g * loss_g + w_l * loss_l + w_f * loss_f + sparse_reg

            total_loss += float(loss.item())
            preds = torch.argmax(outputs, dim=1)
            total_correct += int((preds == labels).sum().item())
            total_count += int(labels.size(0))

    return total_loss / max(1, len(loader)), total_correct / max(1, total_count)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    modules = set(str(args.fusion_modules).lower())
    enable_module_a = 'a' in modules
    enable_module_b = 'b' in modules
    enable_module_c = 'c' in modules

    device = get_device()
    use_amp = (device.type == 'cuda') and (not args.no_amp)

    print(f'Random seed: {args.seed}')
    print(f'Device: {device}  |  AMP: {use_amp}')
    print(f'Rollout layers: {args.attn_rollout_layers}  |  Top-K: {args.topk_patches}')
    print(f'Gate init: {args.part_gate_init}  |  Temperature: {args.attn_temperature}')
    print(f'Fusion modules: A={enable_module_a} B={enable_module_b} C={enable_module_c}')

    # 数据路径
    train_dir = os.path.join(ROOT_DIR, args.dataset_subdir, 'train')
    val_dir = os.path.join(ROOT_DIR, args.dataset_subdir, 'val')

    # 数据增强（遮挡增强建议开启）
    transform = build_default_transforms(
        img_size=224,
        train_occlusion_mode=args.train_occlusion_mode,
        train_occlusion_level=args.train_occlusion_level,
        train_occlusion_p=args.train_occlusion_p,
    )

    train_dataset = datasets.ImageFolder(train_dir, transform['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform['val'])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f'Classes ({num_classes}): {class_names}')
    write_class_indices(class_names, os.path.join(ROOT_DIR, 'class_indices.json'))

    # 模型
    model = ViTFusionModel(
        num_classes=num_classes,
        topk=args.topk_patches,
        pretrained=not args.no_pretrained,
        attn_rollout_layers=args.attn_rollout_layers,
        part_gate_init=args.part_gate_init,
        part_dropout_p=args.part_dropout_p,
        attn_temperature=args.attn_temperature,
        entropy_penalty=args.entropy_penalty,
        enable_module_a=enable_module_a,
        enable_module_b=enable_module_b,
        enable_module_c=enable_module_c,
    ).to(device)

    # 分组优化器：backbone 小学习率，head 大学习率
    backbone_params = list(model.backbone_parameters())
    backbone_param_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.head_parameters()
                   if id(p) not in backbone_param_ids]

    base_lr_backbone = args.lr_backbone
    base_lr_head = args.lr_head

    optimizer = optim.AdamW(
        [
            {'params': backbone_params, 'lr': base_lr_backbone},
            {'params': head_params, 'lr': base_lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    # Cosine 退火调度（warmup 在训练循环里手动处理）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    scaler = create_grad_scaler(enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 训练状态
    best_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []

    weight_path = os.path.join(ROOT_DIR, 'weights', args.weight_name)
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')

        # Warmup：手动缩放学习率
        lr_scale = get_warmup_lr_scale(epoch, args.warmup_epochs)
        if epoch < args.warmup_epochs:
            for i, pg in enumerate(optimizer.param_groups):
                base = base_lr_backbone if i == 0 else base_lr_head
                pg['lr'] = base * lr_scale

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, args.epochs, use_amp, scaler,
            args.attn_sparse_weight, args.grad_clip, args.warmup_epochs,
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
            epoch, args.epochs, use_amp, args.attn_sparse_weight, args.warmup_epochs,
        )

        # Warmup 结束后才启动 scheduler
        if epoch >= args.warmup_epochs:
            scheduler.step()

        print(f'Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}')
        print(f'Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.4f}')
        print(f'Backbone LR: {optimizer.param_groups[0]["lr"]:.2e}  '
              f'Head LR: {optimizer.param_groups[1]["lr"]:.2e}')

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(model.state_dict(), weight_path)
            meta = {
                'architecture': 'vit_fusion_fixed',
                'topk_patches': args.topk_patches,
                'attn_rollout_layers': args.attn_rollout_layers,
                'part_gate_init': args.part_gate_init,
                'part_dropout_p': args.part_dropout_p,
                'attn_temperature': args.attn_temperature,
                'entropy_penalty': args.entropy_penalty,
                'attn_sparse_weight': args.attn_sparse_weight,
                'fusion_modules': ''.join(sorted(modules)),
                'enable_module_a': enable_module_a,
                'enable_module_b': enable_module_b,
                'enable_module_c': enable_module_c,
                'best_val_acc': float(best_acc),
                'best_epoch': best_epoch,
            }
            with open(weight_path + '.meta.json', 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)
            print(f'>>> Saved best model (Val Acc: {best_acc:.4f})')
        else:
            early_stop_counter += 1
            print(f'Early stop counter: {early_stop_counter}/{args.early_stop_patience}')
            if early_stop_counter >= args.early_stop_patience:
                print(f'Early stopping at epoch {epoch + 1} '
                      f'(best epoch: {best_epoch + 1})')
                break

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

    print(f'\nTraining finished. Best val acc: {best_acc:.4f} (epoch {best_epoch + 1})')
    plot_curve(train_loss_list, val_loss_list, 'loss', 'vit_fusion_fixed')
    plot_curve(train_acc_list, val_acc_list, 'accuracy', 'vit_fusion_fixed')


if __name__ == '__main__':
    main()