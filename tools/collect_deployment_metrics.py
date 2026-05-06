#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import get_model, list_models
from utils.model_stats import build_deployment_report, count_trainable_params, parse_board_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect deployment metrics for a SISR checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pth) file")
    parser.add_argument("--tile", nargs=2, type=int, required=True, metavar=("H", "W"))
    parser.add_argument("--arch", choices=list_models(), default=None)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--tflite-float32", default=None)
    parser.add_argument("--tflite-int8", default=None)
    parser.add_argument("--c-header", default=None)
    parser.add_argument("--board-log", default=None)
    parser.add_argument("--voltage", type=float, default=None)
    parser.add_argument("--idle-current-ma", type=float, default=None)
    parser.add_argument("--inference-current-ma", type=float, default=None)
    parser.add_argument("--macs", type=int, default=None)
    parser.add_argument(
        "--profile-macs",
        action="store_true",
        help="Profile MACs with thop for the requested tile.",
    )
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--quality-samples", type=int, default=8)
    parser.add_argument("--output", required=True, help="Output JSON report path")
    parser.add_argument("--csv-output", default=None)
    return parser.parse_args()


def extract_checkpoint_metadata(checkpoint: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(checkpoint, dict):
        return {}, {}
    return checkpoint, checkpoint.get("config", {}) if isinstance(checkpoint.get("config"), dict) else {}


def resolve_arch_scale(
    args: argparse.Namespace, checkpoint: Any, config: dict[str, Any]
) -> tuple[str, int, bool]:
    arch = args.arch
    if arch is None and isinstance(checkpoint, dict):
        arch = checkpoint.get("arch")
    if arch is None:
        arch = config.get("arch")
    if arch is None:
        raise ValueError("Unable to determine checkpoint architecture. Pass --arch explicitly.")

    scale = args.scale
    if scale is None and isinstance(checkpoint, dict):
        scale = checkpoint.get("scale")
    if scale is None:
        scale = config.get("scale")
    if scale is None:
        raise ValueError("Unable to determine checkpoint scale. Pass --scale explicitly.")

    grayscale = bool(config.get("grayscale", False))
    return str(arch), int(scale), grayscale


def extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("state_dict"), dict):
            return checkpoint["state_dict"]
        if isinstance(checkpoint.get("model_state_dict"), dict):
            return checkpoint["model_state_dict"]
    return {}


def power_inputs_from_args(args: argparse.Namespace) -> dict[str, float] | None:
    power_values = [args.voltage, args.idle_current_ma, args.inference_current_ma]
    if all(v is None for v in power_values):
        return None
    if any(v is None for v in power_values):
        raise ValueError("Power metrics require all three power arguments: voltage, idle-current-ma, inference-current-ma")
    return {
        "voltage_v": float(args.voltage),
        "idle_current_ma": float(args.idle_current_ma),
        "inference_current_ma": float(args.inference_current_ma),
    }


def profile_macs(model: Any, tile_h: int, tile_w: int, grayscale: bool) -> int:
    try:
        from thop import profile
    except ImportError as exc:
        raise RuntimeError("MAC profiling requires thop. Install thop or pass --macs explicitly.") from exc

    channels = 1 if grayscale else 3
    was_training = model.training
    if was_training:
        model.eval()
    dummy = torch.zeros(1, channels, tile_h, tile_w)
    try:
        with torch.no_grad():
            macs, _params = profile(model, inputs=(dummy,), verbose=False)
    finally:
        if was_training:
            model.train()
    return int(macs)


def append_csv_row(csv_path: Path, report: dict[str, Any]) -> None:
    row = {
        "arch": report["model"]["arch"],
        "scale": report["model"]["scale"],
        "input_tile": "x".join(map(str, report["model"]["input_tile"])),
        "output_tile": "x".join(map(str, report["model"]["output_tile"])),
        "params": report["model"]["params"],
        "checkpoint_kb": report["artifacts"].get("checkpoint_kb"),
        "inference_ms": report.get("runtime", {}).get("inference_ms"),
        "macs": report.get("compute", {}).get("macs"),
        "mops": report.get("compute", {}).get("mops"),
        "mops_per_watt": report.get("power", {}).get("mops_per_watt"),
        "energy_per_inference_mj": report.get("power", {}).get("energy_per_inference_mj"),
    }

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_checkpoint(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


def compute_quality_metrics(
    model: Any,
    val_dir: str,
    tile_h: int,
    tile_w: int,
    scale: int,
    grayscale: bool,
    max_samples: int,
) -> dict[str, float]:
    import cv2
    import numpy as np
    from utils.metrics import calculate_psnr, calculate_ssim

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    val_path = Path(val_dir)
    if not val_path.exists() or not val_path.is_dir():
        raise ValueError("Validation directory not found")
    image_paths = sorted(
        path
        for path in val_path.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not image_paths:
        raise ValueError(f"No validation images found in {val_dir}")

    crop_h = tile_h * scale
    crop_w = tile_w * scale
    total = {
        "psnr": 0.0,
        "ssim": 0.0,
        "bicubic_psnr": 0.0,
        "bicubic_ssim": 0.0,
    }
    count = 0
    sample_count = max_samples

    model.eval()
    with torch.no_grad():
        for path in image_paths[:sample_count]:
            flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            img = cv2.imread(str(path), flag)
            if img is None:
                continue
            if not grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            if h < crop_h or w < crop_w:
                img = cv2.resize(img, (max(w, crop_w), max(h, crop_h)), interpolation=cv2.INTER_CUBIC)
                h, w = img.shape[:2]

            hr_img = img[:crop_h, :crop_w]
            lr_img = cv2.resize(hr_img, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
            bicubic_img = cv2.resize(lr_img, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)

            if grayscale:
                lr_t = (
                    torch.from_numpy(np.ascontiguousarray(lr_img))
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )
                hr_t = torch.from_numpy(np.ascontiguousarray(hr_img)).unsqueeze(0).float() / 255.0
                bicubic_t = torch.from_numpy(np.ascontiguousarray(bicubic_img)).unsqueeze(0).float() / 255.0
            else:
                lr_t = (
                    torch.from_numpy(np.ascontiguousarray(lr_img))
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )
                hr_t = (
                    torch.from_numpy(np.ascontiguousarray(hr_img))
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
                bicubic_t = (
                    torch.from_numpy(np.ascontiguousarray(bicubic_img))
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )

            sr_t = torch.clamp(model(lr_t), 0.0, 1.0).squeeze(0)

            total["psnr"] += calculate_psnr(sr_t, hr_t)
            total["ssim"] += calculate_ssim(sr_t, hr_t)
            total["bicubic_psnr"] += calculate_psnr(bicubic_t, hr_t)
            total["bicubic_ssim"] += calculate_ssim(bicubic_t, hr_t)
            count += 1

    if count == 0:
        raise ValueError(f"No readable validation images found in {val_dir}")

    quality = {key: value / count for key, value in total.items()}
    quality["psnr_gain"] = quality["psnr"] - quality["bicubic_psnr"]
    quality["ssim_gain"] = quality["ssim"] - quality["bicubic_ssim"]
    return quality


def main() -> None:
    args = parse_args()
    if args.quality_samples <= 0:
        raise ValueError("quality-samples must be > 0")
    if args.macs is not None and args.profile_macs:
        raise ValueError("Use either --macs or --profile-macs, not both.")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    _, config = extract_checkpoint_metadata(checkpoint)
    arch, scale, grayscale = resolve_arch_scale(args, checkpoint, config)

    device = torch.device("cpu")
    model = get_model(arch, scale=scale, device=device, num_channels=1 if grayscale else 3)

    state_dict = extract_state_dict(checkpoint)
    if state_dict:
        model.load_state_dict(state_dict)

    params = count_trainable_params(model)
    runtime = None
    if args.board_log:
        board_log_path = Path(args.board_log)
        if not board_log_path.exists() or not board_log_path.is_file():
            raise FileNotFoundError(f"Board log not found: {board_log_path}")
        runtime = parse_board_log(board_log_path)
    tile_h, tile_w = args.tile
    quality: dict[str, float] | None = None
    if args.val_dir is not None:
        quality = compute_quality_metrics(
            model=model,
            val_dir=args.val_dir,
            tile_h=tile_h,
            tile_w=tile_w,
            scale=scale,
            grayscale=grayscale,
            max_samples=args.quality_samples,
        )
    macs = args.macs
    if args.profile_macs:
        macs = profile_macs(model, tile_h=tile_h, tile_w=tile_w, grayscale=grayscale)

    report = build_deployment_report(
        arch=arch,
        scale=scale,
        tile_h=tile_h,
        tile_w=tile_w,
        params=params,
        checkpoint_path=checkpoint_path,
        tflite_float32_path=args.tflite_float32,
        tflite_int8_path=args.tflite_int8,
        c_header_path=args.c_header,
        quality=quality,
        runtime=runtime,
        macs=macs,
        power_inputs=power_inputs_from_args(args),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.csv_output:
        append_csv_row(Path(args.csv_output), report)

    print(f"Wrote deployment metrics to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
