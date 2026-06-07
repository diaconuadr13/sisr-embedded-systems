from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import get_model
from utils.device import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run trained SISR checkpoint on one real image.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--arch", default=None)
    p.add_argument("--scale", type=int, default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--measure", type=int, default=100)
    return p.parse_args()


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"], raw
    if isinstance(raw, dict):
        return raw, {}
    raise ValueError(f"Unsupported checkpoint format: {path}")


def checkpoint_config(meta: dict[str, Any], ckpt_path: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    config_json = ckpt_path.parent / "config.json"
    if config_json.exists():
        cfg.update(json.loads(config_json.read_text(encoding="utf-8")))
    if isinstance(meta.get("config"), dict):
        cfg.update(meta["config"])
    for key in ("arch", "scale", "model_name"):
        if key in meta and meta[key] is not None:
            cfg.setdefault(key, meta[key])
    return cfg


def read_image(path: Path, grayscale: bool) -> tuple[np.ndarray, torch.Tensor]:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    if grayscale:
        tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float().div(255.0)
        return image, tensor.unsqueeze(0)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)
    return rgb, tensor.unsqueeze(0)


def write_output(path: Path, sr: torch.Tensor, grayscale: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sr.detach().float().cpu().clamp(0.0, 1.0).squeeze(0)
    if grayscale:
        image = (out.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
    else:
        rgb = (out.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image)


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    input_path = Path(args.input)
    output_path = Path(args.output)

    state_dict, meta = load_checkpoint(ckpt_path)
    cfg = checkpoint_config(meta, ckpt_path)
    arch = args.arch or cfg.get("arch")
    scale = args.scale or cfg.get("scale")
    if arch is None or scale is None:
        raise ValueError("Could not infer arch/scale; pass --arch and --scale explicitly.")
    grayscale = bool(cfg.get("grayscale", False))

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = get_model(str(arch), scale=int(scale), device=device, num_channels=1 if grayscale else 3)
    model.load_state_dict(state_dict)
    model.eval()

    image, lr = read_image(input_path, grayscale)
    lr = lr.to(device, non_blocking=True)
    amp_active = args.amp and device.type == "cuda"

    with torch.inference_mode():
        for _ in range(max(args.warmup, 0)):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_active):
                sr = model(lr)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        times = []
        for _ in range(max(args.measure, 1)):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_active):
                sr = model(lr)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)

    write_output(output_path, sr, grayscale)
    avg_ms = float(np.mean(times) * 1000.0)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "checkpoint": str(ckpt_path),
        "model_name": cfg.get("model_name"),
        "arch": str(arch),
        "scale": int(scale),
        "grayscale": grayscale,
        "device": str(device),
        "amp_used": amp_active,
        "source_size": [int(image.shape[1]), int(image.shape[0])],
        "output_size": [int(sr.shape[-1]), int(sr.shape[-2])],
        "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "warmup_iters": int(args.warmup),
        "measure_iters": int(args.measure),
        "avg_inference_ms": avg_ms,
        "estimated_fps": (1000.0 / avg_ms) if avg_ms > 0 else 0.0,
    }
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
