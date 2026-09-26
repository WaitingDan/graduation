import argparse
import csv
import importlib
import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import sys
# Ensure project root is on sys.path when running the script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_model import create_resnet
from models.vgg_model import create_vgg
from models.vit_model import create_vit
from models.vit_fusion_model import ViTFusionModel


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class EfficiencyResult:
    model: str
    params_total_m: float
    params_trainable_m: float
    flops_g: Optional[float]
    flops_method: str
    flops_note: str
    latency_ms_mean: float
    latency_ms_std: float
    latency_ms_p50: float
    latency_ms_p90: float
    throughput_fps: float
    device: str
    dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report FLOPs, parameter count, and inference latency.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet", "vgg", "vit", "vit_fusion"],
        choices=["resnet", "vgg", "vit", "vit_fusion"],
        help="Models to profile.",
    )
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes. If omitted, infer from class_indices.json.")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for profiling.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations for latency.")
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations for latency.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Profiling device.")
    parser.add_argument("--half", action="store_true", help="Use FP16 for latency on CUDA.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="Build models with pretrained weights.")

    # ViT-Fusion options (single-branch key-part modeling)
    parser.add_argument("--fusion_topk", type=int, default=3)
    fusion_attn_group = parser.add_mutually_exclusive_group()
    fusion_attn_group.add_argument("--fusion_use_part_self_attention", dest="fusion_use_part_self_attention", action="store_true")
    fusion_attn_group.add_argument("--fusion_no_part_self_attention", dest="fusion_use_part_self_attention", action="store_false")
    parser.set_defaults(fusion_use_part_self_attention=True)
    parser.add_argument("--fusion_part_gate_init", type=float, default=1.0)
    parser.add_argument("--fusion_part_dropout_p", type=float, default=0.15)

    parser.add_argument("--output_csv", default=os.path.join(ROOT_DIR, "outputs", "evaluation", "efficiency", "efficiency_report.csv"))
    parser.add_argument("--output_md", default=os.path.join(ROOT_DIR, "outputs", "evaluation", "efficiency", "efficiency_report.md"))
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_num_classes(default_value: int = 10) -> int:
    class_indices_path = os.path.join(ROOT_DIR, "class_indices.json")
    if not os.path.exists(class_indices_path):
        return default_value
    try:
        with open(class_indices_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return len(data)
    except Exception:
        pass
    return default_value


def build_model(model_name: str, num_classes: int, args: argparse.Namespace) -> nn.Module:
    if model_name == "resnet":
        return create_resnet(num_classes=num_classes, pretrained=args.pretrained)
    if model_name == "vgg":
        return create_vgg(num_classes=num_classes, pretrained=args.pretrained)
    if model_name == "vit":
        return create_vit(num_classes=num_classes, pretrained=args.pretrained)
    if model_name == "vit_fusion":
        return ViTFusionModel(
            num_classes=num_classes,
            topk=args.fusion_topk,
            pretrained=args.pretrained,
            use_part_self_attention=args.fusion_use_part_self_attention,
            part_gate_init=args.fusion_part_gate_init,
            part_dropout_p=args.fusion_part_dropout_p,
        )
    raise ValueError(f"Unsupported model: {model_name}")


class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def count_parameters(model: nn.Module) -> Tuple[float, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6


def _estimate_flops_thop(model: nn.Module, dummy: torch.Tensor) -> Tuple[float, str]:
    profile = importlib.import_module("thop").profile

    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    flops = float(macs) * 2.0  # common convention: FLOPs ~= 2 * MACs
    return flops, "thop(2*MACs)"


def _estimate_flops_fvcore(model: nn.Module, dummy: torch.Tensor) -> Tuple[float, str]:
    FlopCountAnalysis = importlib.import_module("fvcore.nn").FlopCountAnalysis

    flops = float(FlopCountAnalysis(model, dummy).total())
    return flops, "fvcore"


def _estimate_flops_torch_profiler(model: nn.Module, dummy: torch.Tensor, device: torch.device) -> Tuple[float, str]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.no_grad():
        with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            _ = model(dummy)

    total_flops = 0.0
    for event in prof.key_averages():
        flops = getattr(event, "flops", 0)
        if flops:
            total_flops += float(flops)

    if total_flops <= 0.0:
        raise RuntimeError("torch.profiler did not return FLOPs.")
    return total_flops, "torch.profiler"


def estimate_flops(
    model_name: str,
    model: nn.Module,
    dummy: torch.Tensor,
    device: torch.device,
    disable_attention_guidance_for_flops: bool,
) -> Tuple[Optional[float], str, str]:
    wrapped = LogitsOnlyWrapper(model)
    wrapped.eval()

    del disable_attention_guidance_for_flops
    note = ""

    estimators = [
        _estimate_flops_thop,
        _estimate_flops_fvcore,
        lambda m, x: _estimate_flops_torch_profiler(m, x, device),
    ]

    for estimator in estimators:
        try:
            flops, method = estimator(wrapped, dummy)
            return flops / 1e9, method, note
        except Exception:
            continue

    return None, "unavailable", (note + "; no FLOPs backend succeeded").strip("; ")


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * q
    low = int(idx)
    high = min(low + 1, len(sorted_vals) - 1)
    frac = idx - low
    return sorted_vals[low] * (1.0 - frac) + sorted_vals[high] * frac


def measure_latency(
    model: nn.Module,
    dummy: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
    use_half: bool,
) -> Tuple[float, float, float, float, float]:
    wrapped = LogitsOnlyWrapper(model)
    wrapped.eval()

    latencies_ms: List[float] = []

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            if use_half and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = wrapped(dummy)
            else:
                _ = wrapped(dummy)
        synchronize_if_needed(device)

        for _ in range(max(1, iters)):
            t0 = time.perf_counter()
            if use_half and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = wrapped(dummy)
            else:
                _ = wrapped(dummy)
            synchronize_if_needed(device)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(statistics.mean(latencies_ms))
    std_ms = float(statistics.pstdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0
    p50_ms = percentile(latencies_ms, 0.50)
    p90_ms = percentile(latencies_ms, 0.90)
    throughput_fps = (dummy.shape[0] * 1000.0) / max(1e-9, mean_ms)
    return mean_ms, std_ms, p50_ms, p90_ms, throughput_fps


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_csv(results: List[EfficiencyResult], path: str) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "model",
        "params_total_m",
        "params_trainable_m",
        "flops_g",
        "flops_method",
        "flops_note",
        "latency_ms_mean",
        "latency_ms_std",
        "latency_ms_p50",
        "latency_ms_p90",
        "throughput_fps",
        "device",
        "dtype",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)


def save_markdown(results: List[EfficiencyResult], path: str) -> None:
    ensure_parent_dir(path)
    lines = []
    lines.append("# Efficiency Report")
    lines.append("")
    lines.append("| Model | Params (M) | Trainable (M) | FLOPs (G) | FLOPs Method | Latency Mean (ms) | Latency P50 (ms) | Latency P90 (ms) | Throughput (FPS) | Device | DType |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|")

    for r in results:
        flops_text = "N/A" if r.flops_g is None else f"{r.flops_g:.3f}"
        lines.append(
            "| {model} | {params_total:.3f} | {params_trainable:.3f} | {flops} | {method} | {mean:.3f} | {p50:.3f} | {p90:.3f} | {fps:.2f} | {device} | {dtype} |".format(
                model=r.model,
                params_total=r.params_total_m,
                params_trainable=r.params_trainable_m,
                flops=flops_text,
                method=r.flops_method,
                mean=r.latency_ms_mean,
                p50=r.latency_ms_p50,
                p90=r.latency_ms_p90,
                fps=r.throughput_fps,
                device=r.device,
                dtype=r.dtype,
            )
        )

    lines.append("")
    lines.append("Notes:")
    for r in results:
        if r.flops_note:
            lines.append(f"- {r.model}: {r.flops_note}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def print_summary(results: List[EfficiencyResult]) -> None:
    print("\n" + "=" * 90)
    print("Efficiency Summary")
    print("=" * 90)
    for r in results:
        flops_text = "N/A" if r.flops_g is None else f"{r.flops_g:.3f} G"
        print(
            f"{r.model:10s} | Params: {r.params_total_m:.3f} M | FLOPs: {flops_text:>8s} | "
            f"Latency: {r.latency_ms_mean:.3f} ms (p50 {r.latency_ms_p50:.3f}, p90 {r.latency_ms_p90:.3f}) | "
            f"FPS: {r.throughput_fps:.2f} | {r.device}/{r.dtype}"
        )
    print("=" * 90 + "\n")


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args.device)
    num_classes = args.num_classes if args.num_classes is not None else infer_num_classes(default_value=10)

    dtype_name = "fp16" if (args.half and device.type == "cuda") else "fp32"

    results: List[EfficiencyResult] = []

    for model_name in args.models:
        model = build_model(model_name, num_classes=num_classes, args=args).to(device)
        model.eval()

        if args.half and device.type == "cuda":
            model.half()
            dummy = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device, dtype=torch.float16)
        else:
            dummy = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device, dtype=torch.float32)

        params_total_m, params_trainable_m = count_parameters(model)
        flops_g, flops_method, flops_note = estimate_flops(
            model_name=model_name,
            model=model,
            dummy=dummy,
            device=device,
            disable_attention_guidance_for_flops=False,
        )

        latency_ms_mean, latency_ms_std, latency_ms_p50, latency_ms_p90, throughput_fps = measure_latency(
            model=model,
            dummy=dummy,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            use_half=args.half,
        )

        results.append(
            EfficiencyResult(
                model=model_name,
                params_total_m=params_total_m,
                params_trainable_m=params_trainable_m,
                flops_g=flops_g,
                flops_method=flops_method,
                flops_note=flops_note,
                latency_ms_mean=latency_ms_mean,
                latency_ms_std=latency_ms_std,
                latency_ms_p50=latency_ms_p50,
                latency_ms_p90=latency_ms_p90,
                throughput_fps=throughput_fps,
                device=device.type,
                dtype=dtype_name,
            )
        )

    save_csv(results, args.output_csv)
    save_markdown(results, args.output_md)
    print_summary(results)
    print(f"CSV saved to: {args.output_csv}")
    print(f"Markdown saved to: {args.output_md}")


if __name__ == "__main__":
    main()
