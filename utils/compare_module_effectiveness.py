import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from statistics import mean, pstdev

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from models.resnet_model import create_resnet
from models.vgg_model import create_vgg
from models.vit_model import create_vit
from utils.common import build_default_transforms
from utils.evaluate_models import (
    _build_vit_fusion_model,
    _load_weight_meta,
    _resolve_weight_path,
    device,
)

SCENARIO_LEVELS = {
    "clean": None,
    "light": "light",
    "medium": "medium",
    "heavy": "heavy",
}

MODEL_CHOICES = ["resnet", "vgg", "vit", "vit_fusion"]


# Unified palette for all plots (CV-paper style)
MODEL_COLORS = {
    "vit": "#1f77b4",        # blue
    "vit_fusion": "#d62728", # red
    "resnet": "#ff7f0e",     # orange
    "vgg": "#2ca02c",        # green
}

# Ablation variants: same red tone with clear separation
VARIANT_COLORS = {
    "baseline": "#1f77b4",  # baseline (usually vit)
    "a": "#2ca02c",
    "ab": "#ff7f0e",
    "abc": "#d62728",
    "+a": "#2ca02c",
    "+a+b": "#ff7f0e",
    "+a+b+c": "#d62728",
}


def _label_color(label):
    key = str(label).strip().lower()
    if key in VARIANT_COLORS:
        return VARIANT_COLORS[key]
    if key in MODEL_COLORS:
        return MODEL_COLORS[key]
    return "#7f7f7f"


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def resolve_model_weight(model_name, seed, explicit_path):
    return _resolve_weight_path(model_name, seed=seed, explicit_path=explicit_path)


def build_model(model_name, num_classes, weight_path, fusion_modules=None):
    if model_name == "resnet":
        return create_resnet(num_classes)
    if model_name == "vgg":
        return create_vgg(num_classes)
    if model_name == "vit":
        return create_vit(num_classes)
    if model_name == "vit_fusion":
        return _build_vit_fusion_model(
            num_classes=num_classes,
            weight_path=weight_path,
            fusion_modules_override=fusion_modules,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def load_checkpoint(model, weight_path):
    state_dict = torch.load(weight_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict, strict=False)


def make_loader(dataset_subdir, test_split, batch_size, num_workers, occlusion_mode, occlusion_level, occlusion_p):
    test_path = os.path.join(ROOT_DIR, dataset_subdir, test_split)
    transform = build_default_transforms(
        val_occlusion_mode=occlusion_mode,
        val_occlusion_level=occlusion_level,
        val_occlusion_p=occlusion_p,
    )["val"]
    dataset = datasets.ImageFolder(test_path, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataset, loader


def evaluate_once(model_name, weight_path, loader, num_classes, fusion_modules=None):
    model = build_model(
        model_name,
        num_classes=num_classes,
        weight_path=weight_path,
        fusion_modules=fusion_modules,
    )
    load_checkpoint(model, weight_path)
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

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
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
    }

    k = min(3, num_classes)
    try:
        metrics[f"top{k}_accuracy"] = float(top_k_accuracy_score(y_true, y_prob, k=k, labels=np.arange(num_classes)))
    except Exception:
        metrics[f"top{k}_accuracy"] = float("nan")

    # mAP@0.5 is for detection tasks; for classification we use OvR AP-based mAP.
    try:
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        if y_true_bin.ndim == 1:
            y_true_bin = y_true_bin[:, None]
        metrics["map_macro_ovr"] = float(average_precision_score(y_true_bin, y_prob, average="macro"))
        metrics["map_micro_ovr"] = float(average_precision_score(y_true_bin, y_prob, average="micro"))
    except Exception:
        metrics["map_macro_ovr"] = float("nan")
        metrics["map_micro_ovr"] = float("nan")

    return metrics


def ci95(values):
    if not values:
        return float("nan"), float("nan")
    m = mean(values)
    if len(values) == 1:
        return float(m), float(m)
    std = pstdev(values)
    margin = 1.96 * std / (len(values) ** 0.5)
    return float(m - margin), float(m + margin)


def aggregate(rows, metric_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["scenario"])].append(row)

    summary = []
    for (model, scenario), items in grouped.items():
        out = {
            "model": model,
            "scenario": scenario,
            "runs": len(items),
        }
        for k in metric_keys:
            vals = [float(x[k]) for x in items if not np.isnan(float(x[k]))]
            out[f"{k}_mean"] = float(mean(vals)) if vals else float("nan")
            out[f"{k}_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
            low, high = ci95(vals)
            out[f"{k}_ci95_low"] = low
            out[f"{k}_ci95_high"] = high
        summary.append(out)

    summary.sort(key=lambda x: (x["scenario"], x["model"]))
    return summary


def compare_target_vs_baseline(rows, baseline_model, target_model, metric_keys):
    index = {}
    for row in rows:
        index[(row["model"], row["scenario"], row["seed"])] = row

    comparisons = []
    for key in sorted(index.keys()):
        model, scenario, seed = key
        if model != target_model:
            continue
        base_key = (baseline_model, scenario, seed)
        if base_key not in index:
            continue

        base_row = index[base_key]
        tgt_row = row
        out = {
            "scenario": scenario,
            "seed": seed,
            "baseline_model": baseline_model,
            "target_model": target_model,
        }
        for k in metric_keys:
            b = float(base_row[k])
            t = float(tgt_row[k])
            out[f"baseline_{k}"] = b
            out[f"target_{k}"] = t
            out[f"delta_{k}"] = t - b
            out[f"delta_{k}_pct"] = ((t - b) / b * 100.0) if b != 0 else float("nan")
        comparisons.append(out)

    return comparisons


def summarize_delta(comparisons, metric_keys):
    by_scenario = defaultdict(list)
    for row in comparisons:
        by_scenario[row["scenario"]].append(row)

    scenario_summary = []
    for scenario, items in sorted(by_scenario.items()):
        out = {"scenario": scenario, "runs": len(items)}
        for k in metric_keys:
            vals = [float(x[f"delta_{k}"]) for x in items if not np.isnan(float(x[f"delta_{k}"]))]
            out[f"delta_{k}_mean"] = float(mean(vals)) if vals else float("nan")
            out[f"delta_{k}_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
            out[f"wins_{k}"] = int(sum(1 for v in vals if v > 0))
        scenario_summary.append(out)

    overall = {"scenario": "overall", "runs": len(comparisons)}
    for k in metric_keys:
        vals = [float(x[f"delta_{k}"]) for x in comparisons if not np.isnan(float(x[f"delta_{k}"]))]
        overall[f"delta_{k}_mean"] = float(mean(vals)) if vals else float("nan")
        overall[f"delta_{k}_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0
        overall[f"wins_{k}"] = int(sum(1 for v in vals if v > 0))

    return scenario_summary, overall


def summarize_comparison_table(comparisons, metric_keys):
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[row["scenario"]].append(row)

    table_rows = []
    for scenario, items in sorted(grouped.items()):
        out = {"scenario": scenario, "runs": len(items)}
        for k in metric_keys:
            baseline_vals = [float(x[f"baseline_{k}"]) for x in items if not np.isnan(float(x[f"baseline_{k}"]))]
            target_vals = [float(x[f"target_{k}"]) for x in items if not np.isnan(float(x[f"target_{k}"]))]
            delta_vals = [float(x[f"delta_{k}"]) for x in items if not np.isnan(float(x[f"delta_{k}"]))]
            out[f"baseline_{k}_mean"] = float(mean(baseline_vals)) if baseline_vals else float("nan")
            out[f"target_{k}_mean"] = float(mean(target_vals)) if target_vals else float("nan")
            out[f"delta_{k}_mean"] = float(mean(delta_vals)) if delta_vals else float("nan")
        table_rows.append(out)

    overall = {"scenario": "overall", "runs": len(comparisons)}
    for k in metric_keys:
        baseline_vals = [float(x[f"baseline_{k}"]) for x in comparisons if not np.isnan(float(x[f"baseline_{k}"]))]
        target_vals = [float(x[f"target_{k}"]) for x in comparisons if not np.isnan(float(x[f"target_{k}"]))]
        delta_vals = [float(x[f"delta_{k}"]) for x in comparisons if not np.isnan(float(x[f"delta_{k}"]))]
        overall[f"baseline_{k}_mean"] = float(mean(baseline_vals)) if baseline_vals else float("nan")
        overall[f"target_{k}_mean"] = float(mean(target_vals)) if target_vals else float("nan")
        overall[f"delta_{k}_mean"] = float(mean(delta_vals)) if delta_vals else float("nan")

    return table_rows, overall


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = []
    keys = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = sorted(keys)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_value(value, digits=4):
    if value is None:
        return "-"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if np.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_markdown_table(path, title, rows, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [f"# {title}", ""]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines.append(header)
    lines.append(separator)
    for row in rows:
        values = [str(row.get(columns[0], ""))]
        for col in columns[1:]:
            values.append(_fmt_value(row.get(col)))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def make_report(args, metric_keys, scenario_summary, overall, report_path):
    lines = []
    lines.append("# Module Effectiveness Report")
    lines.append("")
    lines.append(f"- baseline_model: {args.baseline_model}")
    lines.append(f"- target_model: {args.target_model}")
    lines.append(f"- seeds: {args.seeds}")
    lines.append(f"- scenarios: {args.levels}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This project is classification, not detection.")
    lines.append("- mAP@0.5 (IoU-based box metric) is not applicable here.")
    lines.append("- We report AP-based OvR mAP: map_macro_ovr and map_micro_ovr.")
    lines.append("")
    lines.append("## Delta Summary (target - baseline)")
    lines.append("")

    for row in scenario_summary + [overall]:
        lines.append(f"### {row['scenario']} (runs={row['runs']})")
        for k in metric_keys:
            dm = row.get(f"delta_{k}_mean", float("nan"))
            ds = row.get(f"delta_{k}_std", float("nan"))
            wins = row.get(f"wins_{k}", 0)
            lines.append(f"- {k}: mean_delta={dm:.6f}, std={ds:.6f}, wins={wins}")
        lines.append("")

    core = ["accuracy", "macro_recall", "macro_f1", "balanced_accuracy"]
    core_deltas = [overall.get(f"delta_{k}_mean", float("nan")) for k in core]
    effective = all((not np.isnan(x)) and (x > 0) for x in core_deltas)
    lines.append("## Decision")
    if effective:
        lines.append("- Effective: YES (all core metrics improved on average).")
    else:
        lines.append("- Effective: NO (at least one core metric did not improve on average).")
    lines.append("")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _scenario_cfg(args, level):
    if level == "clean":
        return "clean", "none", "light", 0.0
    return f"{args.occlusion_mode}_{level}", args.occlusion_mode, SCENARIO_LEVELS[level], args.occlusion_p


def _resolve_variant_weight(model_name, seed, explicit_path, variant_label):
    if explicit_path:
        if "{seed}" in explicit_path or "{variant}" in explicit_path:
            return explicit_path.format(seed=seed, variant=variant_label)
        return explicit_path
    return resolve_model_weight(model_name, seed=seed, explicit_path=None)


def _parse_variants(args):
    if not args.variant_models:
        return None

    n = len(args.variant_models)
    labels = args.variant_labels or [f"variant_{i+1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("--variant_labels length must match --variant_models")

    weights = args.variant_weights or [None] * n
    if len(weights) != n:
        raise ValueError("--variant_weights length must match --variant_models")

    modules = args.variant_fusion_modules or [None] * n
    if len(modules) != n:
        raise ValueError("--variant_fusion_modules length must match --variant_models")

    variants = []
    for i in range(n):
        variants.append({
            "label": labels[i],
            "model": args.variant_models[i],
            "weight": weights[i],
            "fusion_modules": modules[i],
        })
    return variants


def _plot_variant_means(summary_rows, plot_metrics, output_path, scenario_order):
    labels = sorted({row["model"] for row in summary_rows})
    scenarios = [s for s in scenario_order if s in {row["scenario"] for row in summary_rows}]
    if not labels or not scenarios or not plot_metrics:
        return

    index = {(r["model"], r["scenario"]): r for r in summary_rows}
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(6 * len(plot_metrics), 4), squeeze=False)
    axes = axes[0]

    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(labels))

    for ax, metric in zip(axes, plot_metrics):
        for i, label in enumerate(labels):
            vals = []
            for sc in scenarios:
                row = index.get((label, sc), {})
                vals.append(float(row.get(f"{metric}_mean", np.nan)))
            color = _label_color(label)
            ax.bar(
                x + (i - (len(labels) - 1) / 2) * width,
                vals,
                width=width,
                label=label,
                color=color,
            )

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=min(4, len(labels)))
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_variant_radar(summary_rows, radar_metrics, output_path):
    """Draw a radar chart comparing variants.

    summary_rows: list of dicts produced by `aggregate`, each with keys like
      'model', 'scenario', '<metric>_mean'
    radar_metrics: list of metric keys to plot (strings, without _mean)
    """
    # collect labels (variants)
    labels = sorted({row["model"] for row in summary_rows})
    if not labels or not radar_metrics:
        return

    # compute overall mean across scenarios for each label
    index = defaultdict(list)
    for r in summary_rows:
        label = r["model"]
        for m in radar_metrics:
            val = r.get(f"{m}_mean", float('nan'))
            try:
                v = float(val)
            except Exception:
                v = float('nan')
            index[(label, m)].append(v)

    data = {}
    for label in labels:
        vals = []
        for m in radar_metrics:
            arr = [v for v in index.get((label, m), []) if not np.isnan(v)]
            mean_v = float(np.mean(arr)) if arr else float('nan')
            # clamp to [0,1]
            if np.isnan(mean_v):
                mean_v = 0.0
            mean_v = max(0.0, min(1.0, mean_v))
            vals.append(mean_v)
        data[label] = vals

    # radar setup
    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # draw one polygon per label
    for label in labels:
        vals = data[label]
        values = vals + vals[:1]
        color = _label_color(label)
        ax.plot(angles, values, linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    # set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0.0, 1.0)
    ax.set_title('Ablation Variants Radar (mean across seeds & scenarios)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _plot_variant_radar_per_scenario(summary_rows, radar_metrics, output_dir, scenario_order):
    """Draw radar charts per scenario and save one file per scenario.

    summary_rows: list of dicts with keys 'model','scenario','<metric>_mean'
    radar_metrics: list of metric keys to plot (no '_mean')
    output_dir: directory to save per-scenario radar images
    scenario_order: list of scenario names to plot (e.g., ['clean','mixed_light',...])
    """
    labels = sorted({row["model"] for row in summary_rows})
    if not labels or not radar_metrics:
        return

    # index by (model, scenario) -> mean values
    index = {(r["model"], r["scenario"]): r for r in summary_rows}

    os.makedirs(output_dir, exist_ok=True)

    # ensure matplotlib can render Chinese by reusing project's font helper
    try:
        from utils.visualization.robustness_plots import configure_matplotlib_cjk_font, _apply_mixed_font_rules
        zh_font, en_font = configure_matplotlib_cjk_font(preferred_font=None)
    except Exception:
        zh_font = None
        en_font = None

    for scenario in scenario_order:
        # collect values per label for this scenario
        data = {}
        for label in labels:
            vals = []
            row = index.get((label, scenario), {})
            for m in radar_metrics:
                v = row.get(f"{m}_mean", float('nan'))
                try:
                    v = float(v)
                except Exception:
                    v = float('nan')
                if np.isnan(v):
                    v = 0.0
                v = max(0.0, min(1.0, v))
                vals.append(v)
            data[label] = vals

        # radar setup
        # mapping of preferred angles for each metric
        angle_map = {
            'macro_f1': np.pi / 2.0,        # top
            'accuracy': np.pi,              # left
            'map_macro_ovr': 3.0 * np.pi / 2.0,  # bottom
            'macro_recall': 0.0,            # right (0 == 2pi)
        }

        # produce sorted metric order by increasing angle to ensure convex polygon
        metric_angle_pairs = [(m, float(angle_map.get(m, 0.0))) for m in radar_metrics]
        metric_angle_pairs.sort(key=lambda x: x[1])
        sorted_metrics = [m for m, _ in metric_angle_pairs]
        angles = [a for _, a in metric_angle_pairs]
        angles += angles[:1]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)

        # determine legend/display ordering: baseline first, then +A, +A+B, +A+B+C
        baseline_label = None
        for lab in labels:
            if lab.lower() == 'baseline':
                baseline_label = lab
                break
        if baseline_label is None:
            baseline_label = labels[0]
        others = [l for l in labels if l != baseline_label]
        others_sorted = sorted(others, key=lambda x: len(x))
        ordered_labels = [baseline_label] + others_sorted

        def display_label_from_variant(lab):
            s = lab.lower()
            if lab == baseline_label or s == 'baseline':
                return 'baseline'
            if s in ('a', '+a'):
                return '+A'
            if s in ('ab', 'a+b', 'a_b'):
                return '+A+B'
            if s in ('abc', 'a+b+c', 'a_b_c'):
                return '+A+B+C'
            return lab

        # plot each variant in order
        for label in ordered_labels:
            vals = [data[label][radar_metrics.index(m)] for m in sorted_metrics]
            values = vals + vals[:1]
            color = _label_color(label)
            ax.plot(angles, values, linewidth=2, label=display_label_from_variant(label), color=color)
            ax.fill(angles, values, alpha=0.12, color=color)

        # set custom tick labels according to sorted_metrics
        display_names = {
            'accuracy': 'Accuracy',
            'macro_recall': 'Recall',
            'macro_f1': 'F1',
            'map_macro_ovr': 'mAP',
        }
        xtick_labels = [display_names.get(m, m) for m in sorted_metrics]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(xtick_labels)
        # push axis labels outward so left/right labels do not overlap the outer ring
        ax.tick_params(axis='x', pad=18)

        # radial ticks: 0.85..1.00, hide center label
        ax.set_yticks([0.85, 0.90, 0.95, 1.00])
        ax.set_yticklabels(['0.85', '0.90', '0.95', '1.00'])
        ax.set_ylim(0.84, 1.0)

        # Chinese title per scenario
        title_map = {
            'clean': '无遮挡',
            'mixed_light': '轻度遮挡',
            'mixed_medium': '中度遮挡',
            'mixed_heavy': '重度遮挡',
        }
        pretty_scenario = title_map.get(scenario, scenario)
        ax.set_title(f'消融实验雷达图 - {pretty_scenario}', y=1.16)

        # ensure Chinese font applied if available
        try:
            if zh_font:
                _apply_mixed_font_rules(fig, zh_font=zh_font, en_font=en_font, size_pt=10)
        except Exception:
            pass

        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05))

        out_file = os.path.join(output_dir, f"ablation_variant_radar_{scenario}.png")
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close(fig)

        # draw a three-line table image (booktabs-like)
        table_cols = [display_names.get(m, m) for m in sorted_metrics]
        table_rows = []
        for lab in ordered_labels:
            vals = [data[lab][radar_metrics.index(m)] for m in sorted_metrics]
            table_rows.append([f"{v:.4f}" for v in vals])

        fig, ax = plt.subplots(figsize=(6, 1.2 + 0.5 * len(ordered_labels)))
        ax.axis('off')
        the_table = ax.table(cellText=table_rows, colLabels=table_cols, rowLabels=[display_label_from_variant(l) for l in ordered_labels], loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(1, 1.2)

        # draw three horizontal lines: top, after header, bottom
        cells = the_table.get_celld()
        ys = [cell.get_y() for cell in cells.values()]
        y_min = min(ys)
        y_max = max(ys)
        # header bottom: find min y among header cells (row 0)
        header_ys = [cell.get_y() for key, cell in cells.items() if key[0] == 0]
        header_bottom = min(header_ys) if header_ys else y_max - 0.1

        # convert table coords to axis coords and draw lines
        ax.plot([0, 1], [y_max + 0.0, y_max + 0.0], transform=ax.transAxes, color='black', linewidth=1.2)
        ax.plot([0, 1], [header_bottom, header_bottom], transform=ax.transAxes, color='black', linewidth=0.8)
        ax.plot([0, 1], [y_min - 0.0, y_min - 0.0], transform=ax.transAxes, color='black', linewidth=1.2)

        try:
            if zh_font:
                _apply_mixed_font_rules(fig, zh_font=zh_font, en_font=en_font, size_pt=10)
        except Exception:
            pass

        out_table = os.path.join(output_dir, f"ablation_variant_table_{scenario}.png")
        fig.savefig(out_table, dpi=200, bbox_inches='tight')
        plt.close(fig)


def _run_variant_mode(args, metric_keys, output_root):
    variants = _parse_variants(args)
    if not variants:
        raise ValueError("Variant mode requires --variant_models")

    raw_rows = []

    for seed in args.seeds:
        set_seed(seed)
        for level in args.levels:
            scenario, occ_mode, occ_level, occ_p = _scenario_cfg(args, level)
            dataset, loader = make_loader(
                dataset_subdir=args.dataset_subdir,
                test_split=args.test_split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                occlusion_mode=occ_mode,
                occlusion_level=occ_level,
                occlusion_p=occ_p,
            )
            num_classes = len(dataset.classes)

            for variant in variants:
                weight_path = _resolve_variant_weight(
                    model_name=variant["model"],
                    seed=seed,
                    explicit_path=variant["weight"],
                    variant_label=variant["label"],
                )
                if not os.path.exists(weight_path):
                    raise FileNotFoundError(f"Variant weight not found: {weight_path}")

                metrics = evaluate_once(
                    variant["model"],
                    weight_path,
                    loader,
                    num_classes,
                    fusion_modules=variant["fusion_modules"],
                )
                raw_rows.append({
                    "model": variant["label"],
                    "seed": seed,
                    "scenario": scenario,
                    "fusion_modules": variant["fusion_modules"] or "",
                    **metrics,
                })
                print(
                    f"seed={seed} scenario={scenario} variant={variant['label']} "
                    f"macro_f1={metrics['macro_f1']:.4f}"
                )

    summary_rows = aggregate(raw_rows, metric_keys=metric_keys)
    write_csv(os.path.join(output_root, "ablation_raw_runs.csv"), raw_rows)
    write_csv(os.path.join(output_root, "ablation_summary_mean_std_ci95.csv"), summary_rows)

    baseline_label = variants[0]["label"]
    delta_rows = []
    index = {(r["model"], r["scenario"], r["seed"]): r for r in raw_rows}
    labels = [v["label"] for v in variants]
    for label in labels[1:]:
        for seed in args.seeds:
            for level in args.levels:
                scenario, _, _, _ = _scenario_cfg(args, level)
                base = index.get((baseline_label, scenario, seed))
                tgt = index.get((label, scenario, seed))
                if base is None or tgt is None:
                    continue
                row = {
                    "scenario": scenario,
                    "seed": seed,
                    "baseline_model": baseline_label,
                    "target_model": label,
                }
                for k in metric_keys:
                    b = float(base[k])
                    t = float(tgt[k])
                    row[f"baseline_{k}"] = b
                    row[f"target_{k}"] = t
                    row[f"delta_{k}"] = t - b
                    row[f"delta_{k}_pct"] = ((t - b) / b * 100.0) if b != 0 else float("nan")
                delta_rows.append(row)

    write_csv(os.path.join(output_root, "ablation_vs_baseline_deltas.csv"), delta_rows)

    # additionally draw radar comparing core metrics averaged across seeds and scenarios
    radar_metrics = ["accuracy", "macro_recall", "macro_f1", "map_macro_ovr"]
    scenario_order = [_scenario_cfg(args, lv)[0] for lv in args.levels]
    _plot_variant_radar_per_scenario(
        summary_rows,
        radar_metrics=radar_metrics,
        output_dir=os.path.join(output_root),
        scenario_order=scenario_order,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", default="vit", choices=MODEL_CHOICES)
    parser.add_argument("--target_model", default=None, choices=MODEL_CHOICES)
    parser.add_argument("--baseline_weight", default=None)
    parser.add_argument("--target_weight", default=None)

    parser.add_argument("--dataset_subdir", default="dataset/ship_fine")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--occlusion_mode", choices=["none", "block", "stripe", "mixed"], default="mixed")
    parser.add_argument("--levels", nargs="+", choices=["clean", "light", "medium", "heavy"], default=["clean", "light", "medium", "heavy"])
    parser.add_argument("--occlusion_p", type=float, default=1.0)

    parser.add_argument("--variant_models", nargs="+", choices=MODEL_CHOICES, default=None)
    parser.add_argument("--variant_labels", nargs="+", default=None)
    parser.add_argument("--variant_weights", nargs="+", default=None)
    parser.add_argument("--variant_fusion_modules", nargs="+", default=None)

    parser.add_argument("--output_subdir", default="outputs/evaluation/module_effectiveness")
    parser.add_argument("--experiment_name", default="vit_vs_module")
    parser.add_argument(
        "--table_metrics",
        nargs="+",
        default=["accuracy", "macro_recall", "macro_f1", "balanced_accuracy", "top3_accuracy", "map_macro_ovr", "map_micro_ovr"],
        help="Metrics to include in the generated comparison table",
    )
    parser.add_argument(
        "--plot_metrics",
        nargs="+",
        default=["accuracy", "macro_f1", "balanced_accuracy"],
        help="Metrics to draw in the variant ablation plot",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    metric_keys = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "top1_accuracy",
        "top3_accuracy",
        "map_macro_ovr",
        "map_micro_ovr",
    ]

    output_root = os.path.join(ROOT_DIR, args.output_subdir, args.experiment_name)
    os.makedirs(output_root, exist_ok=True)

    if args.variant_models:
        _run_variant_mode(args, metric_keys, output_root)
        with open(os.path.join(output_root, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        print("Saved outputs to:", output_root)
        return

    if args.target_model is None:
        raise ValueError("Pairwise mode requires --target_model, or use --variant_models for ablation mode")

    raw_rows = []

    for seed in args.seeds:
        set_seed(seed)

        baseline_weight = resolve_model_weight(args.baseline_model, seed=seed, explicit_path=args.baseline_weight)
        target_weight = resolve_model_weight(args.target_model, seed=seed, explicit_path=args.target_weight)

        if not os.path.exists(baseline_weight):
            raise FileNotFoundError(f"Baseline weight not found: {baseline_weight}")
        if not os.path.exists(target_weight):
            raise FileNotFoundError(f"Target weight not found: {target_weight}")

        for level in args.levels:
            scenario, occ_mode, occ_level, occ_p = _scenario_cfg(args, level)

            dataset, loader = make_loader(
                dataset_subdir=args.dataset_subdir,
                test_split=args.test_split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                occlusion_mode=occ_mode,
                occlusion_level=occ_level,
                occlusion_p=occ_p,
            )
            num_classes = len(dataset.classes)

            base_metrics = evaluate_once(args.baseline_model, baseline_weight, loader, num_classes)
            tgt_metrics = evaluate_once(args.target_model, target_weight, loader, num_classes)

            raw_rows.append({
                "model": args.baseline_model,
                "seed": seed,
                "scenario": scenario,
                **base_metrics,
            })
            raw_rows.append({
                "model": args.target_model,
                "seed": seed,
                "scenario": scenario,
                **tgt_metrics,
            })

            print(
                f"seed={seed} scenario={scenario} "
                f"{args.baseline_model}.macro_f1={base_metrics['macro_f1']:.4f} "
                f"{args.target_model}.macro_f1={tgt_metrics['macro_f1']:.4f}"
            )

    summary_rows = aggregate(raw_rows, metric_keys=metric_keys)
    compare_rows = compare_target_vs_baseline(
        raw_rows,
        baseline_model=args.baseline_model,
        target_model=args.target_model,
        metric_keys=metric_keys,
    )
    scenario_summary, overall = summarize_delta(compare_rows, metric_keys=metric_keys)
    table_rows, table_overall = summarize_comparison_table(compare_rows, metric_keys=args.table_metrics)

    write_csv(os.path.join(output_root, "raw_runs.csv"), raw_rows)
    write_csv(os.path.join(output_root, "summary_mean_std_ci95.csv"), summary_rows)
    write_csv(os.path.join(output_root, "target_vs_baseline_deltas.csv"), compare_rows)
    write_csv(os.path.join(output_root, "delta_summary_by_scenario.csv"), scenario_summary + [overall])
    write_csv(os.path.join(output_root, "module_effectiveness_table.csv"), table_rows + [table_overall])
    write_markdown_table(
        os.path.join(output_root, "module_effectiveness_table.md"),
        title="Module Effectiveness Table",
        rows=table_rows + [table_overall],
        columns=["scenario", "runs"] + [f"baseline_{k}_mean" for k in args.table_metrics] + [f"target_{k}_mean" for k in args.table_metrics] + [f"delta_{k}_mean" for k in args.table_metrics],
    )

    with open(os.path.join(output_root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    report_path = os.path.join(output_root, "module_effectiveness_report.md")
    make_report(args, metric_keys, scenario_summary, overall, report_path)

    print("Saved outputs to:", output_root)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
