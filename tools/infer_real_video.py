"""Run VSR inference on a raw video file without HR ground truth.

The script treats the input video as low-resolution real-world footage and
writes a super-resolved MP4 using the temporal window expected by the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from fractions import Fraction
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
    p = argparse.ArgumentParser(description="Run trained SISR/VSR checkpoint on a real MP4.")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pth or a state_dict checkpoint.")
    p.add_argument("--input", required=True, help="Input video path.")
    p.add_argument("--output", required=True, help="Output SR MP4 path.")
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--arch", default=None)
    p.add_argument("--scale", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--hidden-channels", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--grayscale", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--max-frames", type=int, default=0, help="Optional cap for smoke tests.")
    p.add_argument("--codec", default="mp4v", help="OpenCV fourcc, e.g. mp4v or avc1.")
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--summary", default=None, help="Optional JSON summary path.")
    return p.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"], raw
    if isinstance(raw, dict):
        return raw, {}
    raise ValueError(f"Unsupported checkpoint format: {path}")


def checkpoint_config(meta: dict[str, Any], ckpt_path: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if isinstance(meta.get("config"), dict):
        cfg.update(meta["config"])
    config_json = ckpt_path.parent / "config.json"
    if config_json.exists():
        cfg = {**json.loads(config_json.read_text(encoding="utf-8")), **cfg}
    for key in ("arch", "scale", "model_name"):
        if key in meta and meta[key] is not None:
            cfg.setdefault(key, meta[key])
    return cfg


def fps_from_capture(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        return float(fps)
    return 30.0


def frame_to_model_array(frame_bgr: np.ndarray, grayscale: bool) -> np.ndarray:
    if grayscale:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def window_to_tensor(frames: deque[np.ndarray], grayscale: bool, device: torch.device) -> torch.Tensor:
    arr = np.stack(list(frames), axis=0)
    if grayscale:
        tensor = torch.from_numpy(arr).float().div(255.0).unsqueeze(1)
    else:
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(255.0)
    return tensor.unsqueeze(0).to(device, non_blocking=True)


def tensor_to_bgr(tensor: torch.Tensor, grayscale: bool) -> np.ndarray:
    out = tensor.detach().float().cpu().clamp(0.0, 1.0).squeeze(0)
    if grayscale:
        gray = (out.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgb = (out.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_next_model_frame(cap: cv2.VideoCapture, grayscale: bool) -> tuple[bool, np.ndarray | None]:
    ok, frame = cap.read()
    if not ok:
        return False, None
    return True, frame_to_model_array(frame, grayscale)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    ckpt_path = Path(args.checkpoint)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    state_dict, meta = load_checkpoint(ckpt_path, device)
    cfg = checkpoint_config(meta, ckpt_path)
    arch = args.arch or cfg.get("arch")
    scale = args.scale or cfg.get("scale")
    if arch is None or scale is None:
        raise ValueError("Could not infer arch/scale; pass --arch and --scale explicitly.")
    num_frames = args.num_frames or cfg.get("num_frames", 1)
    hidden_channels = args.hidden_channels if args.hidden_channels is not None else cfg.get("hidden_channels")
    num_blocks = args.num_blocks if args.num_blocks is not None else cfg.get("num_blocks")
    grayscale = bool(cfg.get("grayscale", True) if args.grayscale is None else args.grayscale)

    model = get_model(
        str(arch),
        scale=int(scale),
        device=device,
        num_channels=1 if grayscale else 3,
        num_frames=int(num_frames),
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
    )
    model.load_state_dict(state_dict)
    model.eval()

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames = total_frames_raw if args.max_frames <= 0 else min(total_frames_raw, args.max_frames)
    fps = fps_from_capture(cap)
    out_size = (src_w * int(scale), src_h * int(scale))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*args.codec[:4]),
        fps,
        out_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer: {output_path}")

    radius = int(num_frames) // 2
    ok, first = read_next_model_frame(cap, grayscale)
    if not ok or first is None:
        raise RuntimeError(f"Input video has no readable frames: {input_path}")

    window: deque[np.ndarray] = deque(maxlen=int(num_frames))
    for _ in range(radius + 1):
        window.append(first)

    frames_read = 1
    last_frame = first
    while len(window) < int(num_frames):
        if args.max_frames > 0 and frames_read >= args.max_frames:
            window.append(last_frame)
            continue
        ok, frame = read_next_model_frame(cap, grayscale)
        if ok and frame is not None:
            window.append(frame)
            last_frame = frame
            frames_read += 1
        else:
            window.append(last_frame)

    amp_active = args.amp and device.type == "cuda"
    inference_times: list[float] = []
    emitted = 0
    started = time.perf_counter()

    with torch.inference_mode():
        while total_frames <= 0 or emitted < total_frames:
            model_input = window_to_tensor(window, grayscale, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_active):
                sr = model(model_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_times.append(time.perf_counter() - infer_start)

            out_bgr = tensor_to_bgr(sr, grayscale)
            if out_bgr.shape[1] != out_size[0] or out_bgr.shape[0] != out_size[1]:
                out_bgr = cv2.resize(out_bgr, out_size, interpolation=cv2.INTER_CUBIC)
            writer.write(out_bgr)
            emitted += 1

            if args.progress_every > 0 and (emitted == 1 or emitted % args.progress_every == 0):
                elapsed = time.perf_counter() - started
                avg_ms = 1000.0 * sum(inference_times) / len(inference_times)
                fps_eff = emitted / elapsed if elapsed > 0 else 0.0
                total_msg = str(total_frames) if total_frames > 0 else "?"
                print(
                    f"frames={emitted}/{total_msg} avg_infer_ms={avg_ms:.2f} "
                    f"wall_fps={fps_eff:.2f}",
                    flush=True,
                )

            if total_frames > 0 and emitted >= total_frames:
                break

            if args.max_frames > 0 and frames_read >= args.max_frames:
                window.append(last_frame)
                continue
            ok, frame = read_next_model_frame(cap, grayscale)
            if ok and frame is not None:
                window.append(frame)
                last_frame = frame
                frames_read += 1
            elif total_frames <= 0:
                # Unknown total: emit the trailing replicated frames, then stop.
                if emitted >= frames_read + radius:
                    break
                window.append(last_frame)
            else:
                window.append(last_frame)

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - started
    avg_ms = 1000.0 * sum(inference_times) / len(inference_times) if inference_times else 0.0
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "checkpoint": str(ckpt_path),
        "model_name": cfg.get("model_name"),
        "arch": str(arch),
        "scale": int(scale),
        "num_frames": int(num_frames),
        "hidden_channels": hidden_channels,
        "num_blocks": num_blocks,
        "grayscale": grayscale,
        "device": str(device),
        "amp_used": amp_active,
        "source_size": [src_w, src_h],
        "output_size": [out_size[0], out_size[1]],
        "source_fps": float(fps),
        "frames_written": emitted,
        "source_frame_count": total_frames_raw,
        "elapsed_seconds": elapsed,
        "avg_inference_ms_per_frame": avg_ms,
        "effective_wall_fps": emitted / elapsed if elapsed > 0 else 0.0,
        "codec": args.codec[:4],
        "input_fps_fraction": str(Fraction(fps).limit_denominator(1001)),
    }
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
