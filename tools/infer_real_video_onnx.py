"""Run real-video inference with an exported ONNX VSR model.

Expected model contract:
    input:  (1, T, C, H, W)
    output: (1, C, H*scale, W*scale)

The exported DJI models are static-shape, so this script resizes source frames
to the ONNX input H/W if needed and writes video at the ONNX output H/W.
"""
from __future__ import annotations

import argparse
import json
import ctypes
import os
import time
from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from utils.argparse_compat import add_boolean_optional_argument


def preload_cuda_dependencies() -> None:
    """Expose CUDA/cuDNN wheel libraries before CUDAExecutionProvider is created."""
    site_packages = Path(__file__).resolve().parents[1] / ".disertatie" / "lib" / "python3.12" / "site-packages"
    trt_lib_dir = site_packages / "tensorrt_libs"
    if trt_lib_dir.exists():
        os.environ["LD_LIBRARY_PATH"] = f"{trt_lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        for name in (
            "libnvinfer.so.10",
            "libnvinfer_plugin.so.10",
            "libnvonnxparser.so.10",
        ):
            path = trt_lib_dir / name
            if path.exists():
                try:
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
            return
        except Exception:
            pass
    try:
        import torch  # noqa: F401
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ONNX VSR inference on a real MP4.")
    p.add_argument("--model", required=True, help="Path to exported .onnx model.")
    p.add_argument("--input", required=True, help="Input video path.")
    p.add_argument("--output", required=True, help="Output SR MP4 path.")
    p.add_argument("--provider", default="auto", help="auto, cpu, cuda, tensorrt, or exact ORT provider name.")
    add_boolean_optional_argument(p, "--trt-fp16", default=False)
    add_boolean_optional_argument(p, "--trt-int8", default=False)
    add_boolean_optional_argument(p, "--trt-cache", default=True)
    p.add_argument("--trt-cache-dir", default="reports/onnx/trt_cache")
    p.add_argument("--max-frames", type=int, default=0, help="Optional cap for smoke tests.")
    p.add_argument("--codec", default="mp4v", help="OpenCV fourcc, e.g. mp4v or avc1.")
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--summary", default=None, help="Optional JSON summary path.")
    return p.parse_args()


def select_providers(args: argparse.Namespace) -> list[Any]:
    available = ort.get_available_providers()
    aliases = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }
    trt_options = {
        "trt_engine_cache_enable": "True" if args.trt_cache else "False",
        "trt_engine_cache_path": str(Path(args.trt_cache_dir)),
        "trt_fp16_enable": "True" if args.trt_fp16 else "False",
        "trt_int8_enable": "True" if args.trt_int8 else "False",
    }
    if args.provider == "auto":
        preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        providers: list[Any] = []
        for item in preferred:
            if item not in available:
                continue
            providers.append((item, trt_options) if item == "TensorrtExecutionProvider" else item)
        return providers
    requested = aliases.get(args.provider.lower(), args.provider)
    if requested not in available:
        raise RuntimeError(f"Requested provider {requested!r} is not available. Available: {available}")
    providers: list[Any] = [(requested, trt_options)] if requested == "TensorrtExecutionProvider" else [requested]
    if requested != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers


def static_dim(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Model input/output dimension {name!r} must be static; got {value!r}")
    return value


def fps_from_capture(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        return float(fps)
    return 30.0


def frame_to_model_array(frame_bgr: np.ndarray, channels: int, input_size: tuple[int, int]) -> np.ndarray:
    width, height = input_size
    if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    if channels == 1:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if channels == 3:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported channel count: {channels}")


def read_next_model_frame(
    cap: cv2.VideoCapture,
    channels: int,
    input_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    ok, frame = cap.read()
    if not ok:
        return False, None
    return True, frame_to_model_array(frame, channels, input_size)


def window_to_numpy(frames: deque[np.ndarray], channels: int) -> np.ndarray:
    arr = np.stack(list(frames), axis=0).astype(np.float32) / 255.0
    if channels == 1:
        arr = arr[:, None, :, :]
    else:
        arr = arr.transpose(0, 3, 1, 2)
    return arr[None, :, :, :, :]


def output_to_bgr(output: np.ndarray, channels: int) -> np.ndarray:
    out = np.clip(output[0], 0.0, 1.0)
    if channels == 1:
        gray = (out[0] * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgb = (out.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    preload_cuda_dependencies()
    Path(args.trt_cache_dir).mkdir(parents=True, exist_ok=True)
    providers = select_providers(args)
    session = ort.InferenceSession(str(model_path), providers=providers)
    actual_providers = session.get_providers()
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    input_shape = input_meta.shape
    output_shape = output_meta.shape
    num_frames = static_dim(input_shape[1], "T")
    channels = static_dim(input_shape[2], "C")
    input_h = static_dim(input_shape[3], "H")
    input_w = static_dim(input_shape[4], "W")
    output_h = static_dim(output_shape[2], "out_H")
    output_w = static_dim(output_shape[3], "out_W")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames = total_frames_raw if args.max_frames <= 0 else min(total_frames_raw, args.max_frames)
    fps = fps_from_capture(cap)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*args.codec[:4]),
        fps,
        (output_w, output_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer: {output_path}")

    radius = num_frames // 2
    input_size = (input_w, input_h)
    ok, first = read_next_model_frame(cap, channels, input_size)
    if not ok or first is None:
        raise RuntimeError(f"Input video has no readable frames: {input_path}")

    window: deque[np.ndarray] = deque(maxlen=num_frames)
    for _ in range(radius + 1):
        window.append(first)

    frames_read = 1
    last_frame = first
    while len(window) < num_frames:
        if args.max_frames > 0 and frames_read >= args.max_frames:
            window.append(last_frame)
            continue
        ok, frame = read_next_model_frame(cap, channels, input_size)
        if ok and frame is not None:
            window.append(frame)
            last_frame = frame
            frames_read += 1
        else:
            window.append(last_frame)

    inference_times: list[float] = []
    emitted = 0
    started = time.perf_counter()

    while total_frames <= 0 or emitted < total_frames:
        model_input = window_to_numpy(window, channels)
        infer_start = time.perf_counter()
        (sr,) = session.run([output_meta.name], {input_meta.name: model_input})
        inference_times.append(time.perf_counter() - infer_start)

        writer.write(output_to_bgr(sr, channels))
        emitted += 1

        if args.progress_every > 0 and (emitted == 1 or emitted % args.progress_every == 0):
            elapsed = time.perf_counter() - started
            avg_ms = 1000.0 * sum(inference_times) / len(inference_times)
            fps_eff = emitted / elapsed if elapsed > 0 else 0.0
            total_msg = str(total_frames) if total_frames > 0 else "?"
            print(
                f"frames={emitted}/{total_msg} avg_infer_ms={avg_ms:.2f} "
                f"wall_fps={fps_eff:.2f} provider={actual_providers[0]}",
                flush=True,
            )

        if total_frames > 0 and emitted >= total_frames:
            break

        if args.max_frames > 0 and frames_read >= args.max_frames:
            window.append(last_frame)
            continue
        ok, frame = read_next_model_frame(cap, channels, input_size)
        if ok and frame is not None:
            window.append(frame)
            last_frame = frame
            frames_read += 1
        elif total_frames <= 0:
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
        "model": str(model_path),
        "requested_provider": args.provider,
        "providers": actual_providers,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "source_size": [source_w, source_h],
        "model_input_size": [input_w, input_h],
        "output_size": [output_w, output_h],
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
