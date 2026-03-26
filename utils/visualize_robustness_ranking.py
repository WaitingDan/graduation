#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化鲁棒性下降斜率排序结果
"""

import os
import csv
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
from matplotlib.ft2font import FT2Font
import numpy as np

# 设置中文字体
def _font_supports_chinese(font_path, sample_text='中文鲁棒性'):
    """检查字体文件是否覆盖给定中文字符。"""
    try:
        cmap = FT2Font(font_path).get_charmap()
    except Exception:
        return False

    for ch in sample_text:
        if ord(ch) not in cmap:
            return False
    return True


def _find_system_chinese_font():
    """尝试在系统字体中寻找常见的中文字体，返回 (字体名称, 字体路径) 或 None。"""
    preferred_keywords = [
        'noto sans cjk', 'noto serif cjk', 'source han', 'simhei',
        'microsoft yahei', 'wenquanyi', 'wqy', 'pingfang', 'ar pl', '思源', '苹方'
    ]
    fallback_keywords = ['cjk', 'sc', 'tc', 'jp', 'kr', 'hei', 'song', 'fang']

    candidates = []
    for fpath in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
        if not _font_supports_chinese(fpath):
            continue
        try:
            name = fm.FontProperties(fname=fpath).get_name()
        except Exception:
            continue

        lname = name.lower()
        for kw in preferred_keywords:
            if kw in lname:
                return name, fpath

        for kw in fallback_keywords:
            if kw in lname:
                candidates.append((name, fpath))
                break

    if candidates:
        return sorted(candidates, key=lambda item: item[0])[0]

    # 兜底: 返回任意一个可显示中文的字体
    for fpath in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
        if _font_supports_chinese(fpath):
            try:
                return fm.FontProperties(fname=fpath).get_name(), fpath
            except Exception:
                continue

    return None

# 优先使用系统中的中文字体，若找不到则保留默认并在运行时提示用户安装中文字体
chinese_font_info = _find_system_chinese_font()
if chinese_font_info:
    chinese_font, chinese_font_path = chinese_font_info
    try:
        fm.fontManager.addfont(chinese_font_path)
    except Exception:
        pass
    rcParams['font.family'] = [chinese_font]
    rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
else:
    chinese_font = None
    rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("警告: 未检测到系统中文字体，中文可能显示为方块。请安装中文字体，例如: sudo apt install fonts-noto-cjk 或 fonts-wqy-zenhei")

rcParams['axes.unicode_minus'] = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_ranking_data(csv_file):
    """从CSV读取排序数据"""
    data = {
        'model': [],
        'macro_f1_slope': [],
        'balanced_acc_slope': [],
        'macro_f1_drop': [],
        'balanced_acc_drop': []
    }
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['model'].append(row['model'])
            data['macro_f1_slope'].append(float(row['macro_f1_slope_per_level']))
            data['balanced_acc_slope'].append(float(row['balanced_accuracy_slope_per_level']))
            data['macro_f1_drop'].append(float(row['macro_f1_drop_clean_to_heavy']))
            data['balanced_acc_drop'].append(float(row['balanced_accuracy_drop_clean_to_heavy']))
    
    return data


def load_ranking_data_from_json(json_file):
    """从JSON读取排序数据"""
    data = {
        'model': [],
        'macro_f1_slope': [],
        'balanced_acc_slope': [],
        'macro_f1_drop': [],
        'balanced_acc_drop': []
    }

    with open(json_file, 'r', encoding='utf-8') as f:
        rows = json.load(f)

    for row in rows:
        data['model'].append(row['model'])
        data['macro_f1_slope'].append(float(row['macro_f1_slope_per_level']))
        data['balanced_acc_slope'].append(float(row['balanced_accuracy_slope_per_level']))
        data['macro_f1_drop'].append(float(row['macro_f1_drop_clean_to_heavy']))
        data['balanced_acc_drop'].append(float(row['balanced_accuracy_drop_clean_to_heavy']))

    return data


def plot_robustness_ranking(data, output_dir):
    """绘制鲁棒性排序对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('模型鲁棒性下降斜率排序分析', fontsize=16, fontweight='bold', y=0.995)
    
    models = data['model']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. macro-F1 斜率 (绝对值,越大越差)
    ax = axes[0, 0]
    abs_slopes = np.abs(data['macro_f1_slope'])
    bars = ax.barh(models, abs_slopes, color=colors)
    ax.set_xlabel('|macro-F1 斜率| (绝对值)', fontsize=11)
    ax.set_title('Macro-F1 下降斜率\n(斜率绝对值越小越好)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, abs_slopes)):
        ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=10)
    
    # 2. balanced-accuracy 斜率
    ax = axes[0, 1]
    abs_slopes = np.abs(data['balanced_acc_slope'])
    bars = ax.barh(models, abs_slopes, color=colors)
    ax.set_xlabel('|balanced-acc 斜率| (绝对值)', fontsize=11)
    ax.set_title('Balanced-Accuracy 下降斜率\n(斜率绝对值越小越好)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, abs_slopes)):
        ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=10)
    
    # 3. macro-F1 下降量
    ax = axes[1, 0]
    bars = ax.barh(models, data['macro_f1_drop'], color=colors)
    ax.set_xlabel('Macro-F1 下降量', fontsize=11)
    ax.set_title('Clean 到 Heavy: Macro-F1 下降\n(下降越小越好)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, data['macro_f1_drop'])):
        ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=10)
    
    # 4. balanced-accuracy 下降量
    ax = axes[1, 1]
    bars = ax.barh(models, data['balanced_acc_drop'], color=colors)
    ax.set_xlabel('Balanced-Accuracy 下降量', fontsize=11)
    ax.set_title('Clean 到 Heavy: Balanced-Accuracy 下降\n(下降越小越好)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, data['balanced_acc_drop'])):
        ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'robustness_ranking_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_file}")
    
    return output_file


def plot_robustness_summary(data, output_dir):
    """绘制鲁棒性综合评分排序图"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = data['model']
    
    # 归一化处理: 每个指标都归一化到 [0, 1]
    # 对于斜率,转换为绝对值,然后越小越好所以用 (max - val) / max
    
    # 标准化macro-F1斜率 (绝对值越小越好)
    abs_f1_slopes = np.abs(data['macro_f1_slope'])
    f1_score = (np.max(abs_f1_slopes) - abs_f1_slopes) / np.max(abs_f1_slopes)
    
    # 标准化balanced-acc斜率 (绝对值越小越好)
    abs_acc_slopes = np.abs(data['balanced_acc_slope'])
    acc_score = (np.max(abs_acc_slopes) - abs_acc_slopes) / np.max(abs_acc_slopes)
    
    # 标准化macro-F1下降量 (越小越好)
    f1_drop = np.array(data['macro_f1_drop'])
    f1_drop_score = (np.max(f1_drop) - f1_drop) / np.max(f1_drop)
    
    # 标准化balanced-acc下降量 (越小越好)
    acc_drop = np.array(data['balanced_acc_drop'])
    acc_drop_score = (np.max(acc_drop) - acc_drop) / np.max(acc_drop)
    
    # 综合评分 (四个指标平均)
    composite_score = (f1_score + acc_score + f1_drop_score + acc_drop_score) / 4
    
    # 排序
    sorted_indices = np.argsort(composite_score)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_scores = composite_score[sorted_indices]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors_sorted = [colors[i] for i in sorted_indices]
    
    bars = ax.barh(sorted_models, sorted_scores, color=colors_sorted, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('综合鲁棒性评分 (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('模型鲁棒性综合排序\n(越高越好,基于4个指标平均)', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.05])
    ax.invert_yaxis()
    
    # 添加数值标签和排名
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax.text(score + 0.02, i, f'Rank {i+1}: {score:.3f}', va='center', fontsize=11, fontweight='bold')
    
    # 添加网格
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'robustness_composite_ranking.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 综合评分图已保存: {output_file}")
    
    # 打印排序结果
    print("\n" + "="*50)
    print("鲁棒性综合排序 (从最好到最差):")
    print("="*50)
    for i, (model, score) in enumerate(zip(sorted_models, sorted_scores), 1):
        print(f"{i}. {model:12s} - 综合评分: {score:.4f}")
    print("="*50 + "\n")
    
    return output_file


def main():
    csv_file = os.path.join(ROOT_DIR, 'outputs', 'keypart_experiments', 'robustness_slope_ranking.csv')
    json_file = os.path.join(ROOT_DIR, 'outputs', 'keypart_experiments', 'robustness_slope_ranking.json')
    output_dir = os.path.join(ROOT_DIR, 'outputs', 'keypart_experiments')

    if os.path.exists(json_file):
        print(f"读取数据: {json_file}")
        data = load_ranking_data_from_json(json_file)
    elif os.path.exists(csv_file):
        print(f"读取数据: {csv_file}")
        data = load_ranking_data(csv_file)
    else:
        print(f"错误: 找不到文件 {json_file} 或 {csv_file}")
        return

    print("\n生成可视化图表...")
    plot_robustness_ranking(data, output_dir)
    plot_robustness_summary(data, output_dir)
    
    print("\n✓ 所有图表生成完成!")


if __name__ == '__main__':
    main()
