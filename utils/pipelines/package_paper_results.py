"""
Zero-intrusion paper result packer.

This script does not rerun training/evaluation. It only copies existing
robustness outputs into a clean, paper-friendly directory with stable names.

Example:
python utils/pipelines/package_paper_results.py \
  --experiment_name keypart_experiments \
  --tag main
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.output_layout import get_robustness_layout


def parse_args():
    parser = argparse.ArgumentParser(description="Package existing outputs into paper-friendly structure")
    parser.add_argument("--output_subdir", default=None, help="optional custom robustness root")
    parser.add_argument("--experiment_name", default="keypart_experiments", help="robustness experiment name")
    parser.add_argument("--paper_root", default="outputs/paper_results", help="target paper results root")
    parser.add_argument("--tag", default="", help="custom package tag, e.g. main / ablation1")
    parser.add_argument("--include_full_runs", action="store_true", help="copy detailed run-level diagnostics into appendix")
    parser.add_argument("--overwrite", action="store_true", help="overwrite package directory if exists")
    return parser.parse_args()


def safe_copy(src_abs, dst_abs):
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    shutil.copy2(src_abs, dst_abs)


def build_mapping(layout):
    """Map source relative path -> target relative path in package."""
    return {
        os.path.join(layout["plots"], "robustness_composite_ranking.png"):
            os.path.join("main_text", "figures", "fig01_robustness_composite_ranking.png"),
        os.path.join(layout["plots"], "robustness_ranking_visualization.png"):
            os.path.join("main_text", "figures", "fig02_robustness_metric_panels.png"),
        os.path.join(layout["plots"], "paper_fig_macro_balanced_curves.png"):
            os.path.join("main_text", "figures", "fig03_macro_balanced_curves_mean_std.png"),
        os.path.join(layout["plots"], "paper_fig_per_class_recall_drop_heatmap.png"):
            os.path.join("main_text", "figures", "fig04_per_class_recall_drop_heatmap_idx.png"),
        os.path.join(layout["plots"], "paper_fig_seed_variance_curves.png"):
            os.path.join("main_text", "figures", "fig05_seed_variance_curves.png"),
        os.path.join(layout["plots"], "paper_fig_confmat_clean_vs_heavy_top2.png"):
            os.path.join("main_text", "figures", "fig06_confmat_clean_vs_heavy_top2.png"),

        os.path.join(layout["metrics"], "summary_keypart_metrics_agg.csv"):
            os.path.join("main_text", "tables", "tab01_summary_keypart_metrics_agg.csv"),
        os.path.join(layout["metrics"], "summary_keypart_metrics_agg.json"):
            os.path.join("main_text", "tables", "tab01_summary_keypart_metrics_agg.json"),
        os.path.join(layout["ranking"], "robustness_slope_ranking.csv"):
            os.path.join("main_text", "tables", "tab02_robustness_slope_ranking.csv"),
        os.path.join(layout["ranking"], "robustness_slope_ranking.json"):
            os.path.join("main_text", "tables", "tab02_robustness_slope_ranking.json"),
        os.path.join(layout["plots"], "paper_curve_mean_std.csv"):
            os.path.join("main_text", "tables", "tab03_curve_mean_std.csv"),
        os.path.join(layout["plots"], "paper_per_class_recall_drop.csv"):
            os.path.join("main_text", "tables", "tab04_per_class_recall_drop.csv"),
        os.path.join(layout["plots"], "paper_seed_curve_points.csv"):
            os.path.join("main_text", "tables", "tab05_seed_curve_points.csv"),

        os.path.join(layout["metrics"], "summary_keypart_metrics.csv"):
            os.path.join("appendix", "core", "app01_summary_keypart_metrics.csv"),
        os.path.join(layout["metrics"], "summary_keypart_metrics.json"):
            os.path.join("appendix", "core", "app01_summary_keypart_metrics.json"),
        os.path.join(layout["ranking"], "robustness_slope_ranking.md"):
            os.path.join("appendix", "core", "app02_robustness_slope_ranking.md"),
    }


def should_copy_run_detail(filename):
    if filename == "class_indices_eval.json":
        return True
    prefixes = ("confmat_", "per_class_", "report_")
    return filename.startswith(prefixes)


def copy_full_run_details(layout, package_root_abs):
    runs_abs = os.path.join(ROOT_DIR, layout["runs"])
    target_root = os.path.join(package_root_abs, "appendix", "full_runs")
    copied = []

    if not os.path.isdir(runs_abs):
        return copied

    for root, _, files in os.walk(runs_abs):
        for fn in files:
            if not should_copy_run_detail(fn):
                continue
            src_abs = os.path.join(root, fn)
            rel_from_runs = os.path.relpath(src_abs, runs_abs)
            dst_abs = os.path.join(target_root, rel_from_runs)
            safe_copy(src_abs, dst_abs)
            copied.append({
                "source": os.path.join(layout["runs"], rel_from_runs).replace("\\", "/"),
                "target": os.path.join("appendix", "full_runs", rel_from_runs).replace("\\", "/"),
            })

    return copied


def write_manifest(manifest_path, package_name, experiment_root_rel, copied_items, missing_items):
    lines = []
    lines.append("# Paper Result Package Manifest")
    lines.append("")
    lines.append(f"- package: {package_name}")
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- source_experiment_root: {experiment_root_rel}")
    lines.append("")
    lines.append("## Figure/Table Semantics")
    lines.append("")
    lines.append("- main_text/figures: figures used in the main paper body.")
    lines.append("- main_text/tables: core metric/ranking tables used in the main paper body.")
    lines.append("- appendix/core: supplementary global summaries and markdown report.")
    lines.append("- appendix/full_runs: run-level diagnostics (confmat/per_class/report/class index).")
    lines.append("")
    lines.append("## Copied Files")
    lines.append("")
    if copied_items:
        for item in copied_items:
            lines.append(f"- {item['target']} <- {item['source']}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("## Missing Source Files")
    lines.append("")
    if missing_items:
        for miss in missing_items:
            lines.append(f"- {miss}")
    else:
        lines.append("- (none)")

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    layout = get_robustness_layout(output_subdir=args.output_subdir, experiment_name=args.experiment_name)
    mapping = build_mapping(layout)

    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"{args.experiment_name}_{args.tag}" if args.tag else f"{args.experiment_name}_{now_tag}"

    paper_root_abs = os.path.join(ROOT_DIR, args.paper_root)
    package_root_abs = os.path.join(paper_root_abs, package_name)

    if os.path.exists(package_root_abs):
        if args.overwrite:
            shutil.rmtree(package_root_abs)
        else:
            raise FileExistsError(
                f"Package dir already exists: {package_root_abs}. Use --overwrite or change --tag."
            )

    os.makedirs(package_root_abs, exist_ok=True)
    os.makedirs(os.path.join(package_root_abs, "main_text", "figures"), exist_ok=True)
    os.makedirs(os.path.join(package_root_abs, "main_text", "tables"), exist_ok=True)
    os.makedirs(os.path.join(package_root_abs, "appendix", "core"), exist_ok=True)
    os.makedirs(os.path.join(package_root_abs, "appendix", "full_runs"), exist_ok=True)

    copied = []
    missing = []

    for src_rel, dst_rel in mapping.items():
        src_abs = os.path.join(ROOT_DIR, src_rel)
        dst_abs = os.path.join(package_root_abs, dst_rel)
        if os.path.exists(src_abs):
            safe_copy(src_abs, dst_abs)
            copied.append({"source": src_rel, "target": dst_rel})
        else:
            missing.append(src_rel)

    if args.include_full_runs:
        copied.extend(copy_full_run_details(layout, package_root_abs))

    manifest_md = os.path.join(package_root_abs, "manifest.md")
    write_manifest(
        manifest_path=manifest_md,
        package_name=package_name,
        experiment_root_rel=layout["root"],
        copied_items=copied,
        missing_items=missing,
    )

    manifest_json = os.path.join(package_root_abs, "manifest.json")
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "package": package_name,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source_experiment_root": layout["root"],
                "copied": copied,
                "missing": missing,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Paper package created:", os.path.join(args.paper_root, package_name))
    print("Copied files:", len(copied), "Missing files:", len(missing))
    print("Manifest:", os.path.join(args.paper_root, package_name, "manifest.md"))


if __name__ == "__main__":
    main()
