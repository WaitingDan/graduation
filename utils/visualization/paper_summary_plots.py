#!/usr/bin/env python3

"""Generate paper-friendly summary plots from existing robustness outputs.

Zero-intrusion: no retraining/evaluation, only reads files already produced by
occlusion_suite/slope_ranking and writes extra plots/tables.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.output_layout import get_robustness_layout, ensure_layout_dirs


INTENSITY_MAP = {
    "clean": 0.00,
    "light": 0.10,
    "medium": 0.20,
    "heavy": 0.35,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper summary plots from robustness outputs")
    parser.add_argument("--output_subdir", default=None, help="optional custom robustness root")
    parser.add_argument("--experiment_name", default="keypart_experiments", help="robustness experiment name")
    parser.add_argument("--occlusion_mode", choices=["block", "stripe", "mixed"], default="mixed")
    parser.add_argument("--models", nargs="+", default=["resnet", "vgg", "vit", "vit_fusion", "vit_two_road"])
    parser.add_argument("--seed_for_confmat", type=int, default=42, help="seed used in clean/heavy confusion matrix comparison")
    parser.add_argument("--font_family", default=None, help="optional matplotlib font family for Chinese labels")
    return parser.parse_args()


def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _mean_std(vals):
    import numpy as np

    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def read_summary_rows(summary_csv):
    with open(summary_csv, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_curve_agg(rows, models, occlusion_mode):
    scenarios = ["clean", f"{occlusion_mode}_light", f"{occlusion_mode}_medium", f"{occlusion_mode}_heavy"]
    scenario_to_x = {
        "clean": INTENSITY_MAP["clean"],
        f"{occlusion_mode}_light": INTENSITY_MAP["light"],
        f"{occlusion_mode}_medium": INTENSITY_MAP["medium"],
        f"{occlusion_mode}_heavy": INTENSITY_MAP["heavy"],
    }

    bucket = defaultdict(lambda: defaultdict(lambda: {"macro": [], "bal": []}))
    for r in rows:
        m = r.get("model")
        s = r.get("scenario")
        if m in models and s in scenarios:
            bucket[m][s]["macro"].append(_to_float(r.get("macro_f1")))
            bucket[m][s]["bal"].append(_to_float(r.get("balanced_accuracy")))

    out = []
    for m in models:
        for s in scenarios:
            macro_mean, macro_std = _mean_std(bucket[m][s]["macro"])
            bal_mean, bal_std = _mean_std(bucket[m][s]["bal"])
            out.append(
                {
                    "model": m,
                    "scenario": s,
                    "x_ratio": scenario_to_x[s],
                    "macro_f1_mean": macro_mean,
                    "macro_f1_std": macro_std,
                    "balanced_accuracy_mean": bal_mean,
                    "balanced_accuracy_std": bal_std,
                }
            )
    return out


def write_curve_agg_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "model",
        "scenario",
        "x_ratio",
        "macro_f1_mean",
        "macro_f1_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def configure_matplotlib_cjk_font(preferred_font=None):
    import subprocess
    import matplotlib
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties

    if preferred_font:
        matplotlib.rcParams["font.sans-serif"] = [preferred_font, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        return preferred_font

    preferred_candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans SC",
        "Source Han Sans CN",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}

    selected = None
    for name in preferred_candidates:
        if name in available:
            selected = name
            break

    if selected is None:
        patterns = ("Noto Sans CJK", "Source Han", "WenQuanYi", "YaHei", "SimHei", "PingFang", "Heiti")
        for name in sorted(available):
            if any(p in name for p in patterns):
                selected = name
                break

    if selected is not None:
        matplotlib.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        return selected

    # Fallback for environments where matplotlib cache misses system fonts.
    try:
        out = subprocess.check_output(["fc-list", ":lang=zh", "file", "family"], text=True, stderr=subprocess.STDOUT)
        font_file = None
        for line in out.splitlines():
            parts = line.split(":", 1)
            if parts and parts[0].strip().lower().endswith((".ttf", ".ttc", ".otf")):
                font_file = parts[0].strip()
                break
        if font_file and os.path.exists(font_file):
            font_manager.fontManager.addfont(font_file)
            loaded_name = FontProperties(fname=font_file).get_name()
            matplotlib.rcParams["font.sans-serif"] = [loaded_name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return loaded_name
    except Exception:
        pass

    matplotlib.rcParams["axes.unicode_minus"] = False
    return None


def plot_curve_with_errorbars(curve_rows, out_png, models):
    import numpy as np
    import matplotlib.pyplot as plt

    model_colors = {
        "vit": "#1f77b4",
        "resnet": "#ff7f0e",
        "vit_fusion": "#2ca02c",
        "vgg": "#d62728",
        "vit_two_road": "#9467bd",
    }

    by_model = defaultdict(list)
    for r in curve_rows:
        by_model[r["model"]].append(r)

    for m in by_model:
        by_model[m] = sorted(by_model[m], key=lambda x: x["x_ratio"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    ax1, ax2 = axes

    for m in models:
        rows = by_model.get(m, [])
        if not rows:
            continue
        xs = np.asarray([r["x_ratio"] for r in rows], dtype=float)
        macro_mean = np.asarray([r["macro_f1_mean"] for r in rows], dtype=float)
        macro_std = np.asarray([r["macro_f1_std"] for r in rows], dtype=float)
        bal_mean = np.asarray([r["balanced_accuracy_mean"] for r in rows], dtype=float)
        bal_std = np.asarray([r["balanced_accuracy_std"] for r in rows], dtype=float)

        color = model_colors.get(m, "#7f7f7f")
        ax1.errorbar(xs, macro_mean, yerr=macro_std, marker="o", capsize=3, label=m, color=color)
        ax2.errorbar(xs, bal_mean, yerr=bal_std, marker="o", capsize=3, label=m, color=color)

    for ax in axes:
        ax.set_xlabel("遮挡比例")
        ax.set_xticks([0.0, 0.10, 0.20, 0.35])
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_ylim(0.0, 1.03)

    ax1.set_title("macro-F1 随遮挡比例变化（均值±标准差）")
    ax1.set_ylabel("macro-F1")
    ax2.set_title("balanced-accuracy 随遮挡比例变化（均值±标准差）")
    ax2.set_ylabel("balanced-accuracy")
    ax2.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _read_per_class_csv(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row.get("class_idx"))
            out[idx] = _to_float(row.get("recall"))
    return out


def build_class_drop_matrix(runs_root, models, occlusion_mode):
    import numpy as np

    clean_root = os.path.join(runs_root, "clean")
    heavy_root = os.path.join(runs_root, f"{occlusion_mode}_heavy")
    if not os.path.isdir(clean_root) or not os.path.isdir(heavy_root):
        raise FileNotFoundError(
            f"Missing runs for clean/heavy: {clean_root} / {heavy_root}"
        )

    seeds = []
    for d in os.listdir(clean_root):
        if d.startswith("seed_") and os.path.isdir(os.path.join(clean_root, d)):
            if os.path.isdir(os.path.join(heavy_root, d)):
                seeds.append(d)
    seeds = sorted(seeds)
    if not seeds:
        raise RuntimeError("No shared seeds found between clean and heavy scenarios")

    # infer class indices from first available file
    class_indices = None
    for m in models:
        probe = os.path.join(clean_root, seeds[0], f"per_class_{m}.csv")
        if os.path.exists(probe):
            d = _read_per_class_csv(probe)
            class_indices = sorted(d.keys())
            break
    if class_indices is None:
        raise FileNotFoundError("No per_class_<model>.csv found under clean runs")

    mat = np.zeros((len(class_indices), len(models)), dtype=float)

    for j, m in enumerate(models):
        per_seed_drops = []
        for sd in seeds:
            clean_csv = os.path.join(clean_root, sd, f"per_class_{m}.csv")
            heavy_csv = os.path.join(heavy_root, sd, f"per_class_{m}.csv")
            if not (os.path.exists(clean_csv) and os.path.exists(heavy_csv)):
                continue
            clean_rec = _read_per_class_csv(clean_csv)
            heavy_rec = _read_per_class_csv(heavy_csv)
            drops = []
            for idx in class_indices:
                drops.append(clean_rec.get(idx, 0.0) - heavy_rec.get(idx, 0.0))
            per_seed_drops.append(drops)

        if per_seed_drops:
            arr = np.asarray(per_seed_drops, dtype=float)
            mat[:, j] = arr.mean(axis=0)

    return class_indices, seeds, mat


def write_class_drop_csv(path, class_indices, models, mat):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        fields = ["class_idx"] + [f"drop_{m}" for m in models]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, c in enumerate(class_indices):
            row = {"class_idx": c}
            for j, m in enumerate(models):
                row[f"drop_{m}"] = float(mat[i, j])
            w.writerow(row)


def plot_class_drop_heatmap(class_indices, models, mat, out_png):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Recall 下降量（clean - heavy）")

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=0)
    ax.set_yticks(np.arange(len(class_indices)))
    ax.set_yticklabels([str(x) for x in class_indices])
    ax.set_xlabel("模型")
    ax.set_ylabel("类别索引")
    ax.set_title("各类别 Recall 下降热图（按类别索引）")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_seed_curve_rows(rows, models, occlusion_mode):
    scenarios = ["clean", f"{occlusion_mode}_light", f"{occlusion_mode}_medium", f"{occlusion_mode}_heavy"]
    scenario_to_x = {
        "clean": INTENSITY_MAP["clean"],
        f"{occlusion_mode}_light": INTENSITY_MAP["light"],
        f"{occlusion_mode}_medium": INTENSITY_MAP["medium"],
        f"{occlusion_mode}_heavy": INTENSITY_MAP["heavy"],
    }

    out = []
    for r in rows:
        m = r.get("model")
        s = r.get("scenario")
        if m not in models or s not in scenarios:
            continue
        seed = r.get("seed")
        out.append(
            {
                "model": m,
                "seed": seed,
                "scenario": s,
                "x_ratio": scenario_to_x[s],
                "macro_f1": _to_float(r.get("macro_f1")),
                "balanced_accuracy": _to_float(r.get("balanced_accuracy")),
            }
        )
    return out


def write_seed_curve_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["model", "seed", "scenario", "x_ratio", "macro_f1", "balanced_accuracy"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_seed_variance_panels(seed_rows, models, out_png):
    import numpy as np
    import matplotlib.pyplot as plt

    model_colors = {
        "vit": "#1f77b4",
        "resnet": "#ff7f0e",
        "vit_fusion": "#2ca02c",
        "vgg": "#d62728",
        "vit_two_road": "#9467bd",
    }

    grouped = defaultdict(lambda: defaultdict(list))
    for r in seed_rows:
        grouped[r["model"]][str(r["seed"])].append(r)

    fig, axes = plt.subplots(len(models), 2, figsize=(12, 2.8 * max(1, len(models))), sharex=True)
    if len(models) == 1:
        axes = np.asarray([axes])

    for i, m in enumerate(models):
        ax_macro = axes[i, 0]
        ax_bal = axes[i, 1]
        color = model_colors.get(m, "#7f7f7f")

        seed_data = grouped.get(m, {})
        all_macro = []
        all_bal = []
        all_x = []

        for seed, rows_m in sorted(seed_data.items(), key=lambda x: x[0]):
            rows_sorted = sorted(rows_m, key=lambda x: x["x_ratio"])
            xs = np.asarray([r["x_ratio"] for r in rows_sorted], dtype=float)
            macro = np.asarray([r["macro_f1"] for r in rows_sorted], dtype=float)
            bal = np.asarray([r["balanced_accuracy"] for r in rows_sorted], dtype=float)

            ax_macro.plot(xs, macro, marker="o", alpha=0.35, color=color)
            ax_bal.plot(xs, bal, marker="o", alpha=0.35, color=color)

            all_x.append(xs)
            all_macro.append(macro)
            all_bal.append(bal)

        if all_macro:
            base_x = all_x[0]
            macro_mean = np.mean(np.vstack(all_macro), axis=0)
            bal_mean = np.mean(np.vstack(all_bal), axis=0)
            ax_macro.plot(base_x, macro_mean, marker="o", linewidth=2.2, color=color, label=f"{m} 均值")
            ax_bal.plot(base_x, bal_mean, marker="o", linewidth=2.2, color=color, label=f"{m} 均值")

        ax_macro.set_ylabel(f"{m}\nmacro-F1")
        ax_bal.set_ylabel("balanced-accuracy")
        ax_macro.grid(alpha=0.25, linestyle="--")
        ax_bal.grid(alpha=0.25, linestyle="--")
        ax_macro.set_ylim(0.0, 1.03)
        ax_bal.set_ylim(0.0, 1.03)

    for ax in axes[-1, :]:
        ax.set_xlabel("遮挡比例")
        ax.set_xticks([0.0, 0.10, 0.20, 0.35])

    axes[0, 0].set_title("不同 seed 下 macro-F1 变化（细线）与均值曲线（粗线）")
    axes[0, 1].set_title("不同 seed 下 balanced-accuracy 变化（细线）与均值曲线（粗线）")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _pick_top_models_from_ranking(ranking_csv, fallback_models):
    if not os.path.exists(ranking_csv):
        return fallback_models[:2]

    rows = []
    with open(ranking_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if not rows:
        return fallback_models[:2]

    try:
        rows = sorted(rows, key=lambda x: int(x.get("rank", 9999)))
    except Exception:
        pass
    models = [r.get("model") for r in rows if r.get("model")]
    if len(models) >= 2:
        return models[:2]
    return fallback_models[:2]


def _find_available_seed(clean_root, heavy_root, preferred_seed):
    preferred = f"seed_{preferred_seed}"
    if os.path.isdir(os.path.join(clean_root, preferred)) and os.path.isdir(os.path.join(heavy_root, preferred)):
        return preferred

    candidates = []
    for d in os.listdir(clean_root):
        if d.startswith("seed_") and os.path.isdir(os.path.join(clean_root, d)) and os.path.isdir(os.path.join(heavy_root, d)):
            candidates.append(d)
    candidates = sorted(candidates)
    if not candidates:
        raise RuntimeError("未找到 clean 与 heavy 共同存在的 seed 目录")
    return candidates[0]


def plot_confmat_comparison_panel(runs_root, occlusion_mode, models, preferred_seed, out_png):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    clean_root = os.path.join(runs_root, "clean")
    heavy_root = os.path.join(runs_root, f"{occlusion_mode}_heavy")
    if not os.path.isdir(clean_root) or not os.path.isdir(heavy_root):
        raise FileNotFoundError(f"Missing clean/heavy runs: {clean_root} / {heavy_root}")

    seed_dir = _find_available_seed(clean_root, heavy_root, preferred_seed)
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 4.8 * max(1, len(models))))
    if len(models) == 1:
        axes = [axes]

    for i, m in enumerate(models):
        clean_png = os.path.join(clean_root, seed_dir, f"confmat_{m}.png")
        heavy_png = os.path.join(heavy_root, seed_dir, f"confmat_{m}.png")
        if not os.path.exists(clean_png) or not os.path.exists(heavy_png):
            raise FileNotFoundError(f"Missing confusion matrix png for model={m}, seed={seed_dir}")

        ax_clean, ax_heavy = axes[i]
        ax_clean.imshow(mpimg.imread(clean_png))
        ax_heavy.imshow(mpimg.imread(heavy_png))
        ax_clean.axis("off")
        ax_heavy.axis("off")
        ax_clean.set_title(f"{m} | clean | {seed_dir}")
        ax_heavy.set_title(f"{m} | {occlusion_mode}_heavy | {seed_dir}")

    fig.suptitle("clean 与 heavy 场景混淆矩阵对照（Top-2 模型）", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return seed_dir


def main():
    args = parse_args()

    selected_font = configure_matplotlib_cjk_font(preferred_font=args.font_family)
    if selected_font is not None:
        print("Using CJK font:", selected_font)
    else:
        print("Warning: no CJK font detected, Chinese labels may fallback incorrectly")

    layout = get_robustness_layout(output_subdir=args.output_subdir, experiment_name=args.experiment_name)
    ensure_layout_dirs(ROOT_DIR, layout)

    summary_csv = os.path.join(ROOT_DIR, layout["metrics"], "summary_keypart_metrics.csv")
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Missing summary metrics csv: {summary_csv}")

    # Figure A: scenario curves with seed variance
    rows = read_summary_rows(summary_csv)
    curve_rows = build_curve_agg(rows, args.models, args.occlusion_mode)

    curve_csv = os.path.join(ROOT_DIR, layout["plots"], "paper_curve_mean_std.csv")
    curve_png = os.path.join(ROOT_DIR, layout["plots"], "paper_fig_macro_balanced_curves.png")
    write_curve_agg_csv(curve_csv, curve_rows)
    plot_curve_with_errorbars(curve_rows, curve_png, args.models)

    # Figure C: seed-level curves + mean trend
    seed_curve_rows = build_seed_curve_rows(rows, args.models, args.occlusion_mode)
    seed_curve_csv = os.path.join(ROOT_DIR, layout["plots"], "paper_seed_curve_points.csv")
    seed_curve_png = os.path.join(ROOT_DIR, layout["plots"], "paper_fig_seed_variance_curves.png")
    write_seed_curve_csv(seed_curve_csv, seed_curve_rows)
    plot_seed_variance_panels(seed_curve_rows, args.models, seed_curve_png)

    # Figure B: per-class recall drop heatmap (clean->heavy)
    class_indices, seeds, mat = build_class_drop_matrix(
        runs_root=os.path.join(ROOT_DIR, layout["runs"]),
        models=args.models,
        occlusion_mode=args.occlusion_mode,
    )
    heatmap_csv = os.path.join(ROOT_DIR, layout["plots"], "paper_per_class_recall_drop.csv")
    heatmap_png = os.path.join(ROOT_DIR, layout["plots"], "paper_fig_per_class_recall_drop_heatmap.png")
    write_class_drop_csv(heatmap_csv, class_indices, args.models, mat)
    plot_class_drop_heatmap(class_indices, args.models, mat, heatmap_png)

    # Figure D: clean vs heavy confusion matrix comparison panel for top-2 models
    ranking_csv = os.path.join(ROOT_DIR, layout["ranking"], "robustness_slope_ranking.csv")
    top2_models = _pick_top_models_from_ranking(ranking_csv, args.models)
    confmat_panel_png = os.path.join(ROOT_DIR, layout["plots"], "paper_fig_confmat_clean_vs_heavy_top2.png")
    used_seed_dir = plot_confmat_comparison_panel(
        runs_root=os.path.join(ROOT_DIR, layout["runs"]),
        occlusion_mode=args.occlusion_mode,
        models=top2_models,
        preferred_seed=args.seed_for_confmat,
        out_png=confmat_panel_png,
    )

    print("Saved:", curve_png)
    print("Saved:", heatmap_png)
    print("Saved:", seed_curve_png)
    print("Saved:", confmat_panel_png)
    print("Saved:", curve_csv)
    print("Saved:", heatmap_csv)
    print("Saved:", seed_curve_csv)
    print("Shared seeds used for heatmap:", ", ".join(seeds))
    print("Seed used for confusion matrix panel:", used_seed_dir)
    print("Top-2 models used for confusion matrix panel:", ", ".join(top2_models))


if __name__ == "__main__":
    main()
