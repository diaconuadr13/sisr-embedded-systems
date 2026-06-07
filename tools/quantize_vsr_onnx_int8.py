"""Post-training INT8 quantization for exported VSR ONNX models.

This uses static calibration from a real video and writes a QDQ-format ONNX
model. QDQ is the preferred representation for TensorRT INT8 execution via
ONNX Runtime's TensorRTExecutionProvider.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from utils.argparse_compat import add_boolean_optional_argument
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantize a VSR ONNX model to INT8 using video calibration.")
    p.add_argument("--model", required=True, help="Input FP32 ONNX model.")
    p.add_argument("--video", required=True, help="Calibration video.")
    p.add_argument("--output", required=True, help="Output INT8/QDQ ONNX model.")
    p.add_argument("--samples", type=int, default=128, help="Number of calibration windows.")
    p.add_argument("--stride", type=int, default=8, help="Read every Nth window for calibration.")
    p.add_argument("--activation-type", choices=["qint8", "quint8"], default="qint8")
    p.add_argument("--weight-type", choices=["qint8", "quint8"], default="qint8")
    p.add_argument("--method", choices=["minmax", "entropy", "percentile"], default="minmax")
    add_boolean_optional_argument(p, "--per-channel", default=True)
    add_boolean_optional_argument(p, "--extra-options", default=True)
    return p.parse_args()


def static_dim(value: Any, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Model input dimension {name!r} must be static; got {value!r}")
    return value


def model_io(model_path: Path) -> tuple[str, list[int], list[int]]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    input_shape = [static_dim(v, f"input[{i}]") for i, v in enumerate(inp.shape)]
    output_shape = [static_dim(v, f"output[{i}]") for i, v in enumerate(out.shape)]
    return inp.name, input_shape, output_shape


def frame_to_model_array(frame_bgr: np.ndarray, channels: int, width: int, height: int) -> np.ndarray:
    if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    if channels == 1:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if channels == 3:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported channel count: {channels}")


def window_to_numpy(frames: deque[np.ndarray], channels: int) -> np.ndarray:
    arr = np.stack(list(frames), axis=0).astype(np.float32) / 255.0
    if channels == 1:
        arr = arr[:, None, :, :]
    else:
        arr = arr.transpose(0, 3, 1, 2)
    return arr[None, :, :, :, :]


class VideoWindowCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        video_path: Path,
        input_name: str,
        input_shape: list[int],
        samples: int,
        stride: int,
    ) -> None:
        self.input_name = input_name
        self.input_shape = input_shape
        self.samples = samples
        self.stride = max(1, stride)
        self.items = self._load(video_path)
        self.index = 0

    def _load(self, video_path: Path) -> list[dict[str, np.ndarray]]:
        _batch, num_frames, channels, height, width = self.input_shape
        radius = num_frames // 2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open calibration video: {video_path}")

        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Calibration video has no readable frames: {video_path}")
        first = frame_to_model_array(frame, channels, width, height)

        window: deque[np.ndarray] = deque(maxlen=num_frames)
        for _ in range(radius + 1):
            window.append(first)
        last = first
        while len(window) < num_frames:
            ok, frame = cap.read()
            if ok:
                last = frame_to_model_array(frame, channels, width, height)
            window.append(last)

        items: list[dict[str, np.ndarray]] = []
        seen = 0
        while len(items) < self.samples:
            if seen % self.stride == 0:
                items.append({self.input_name: window_to_numpy(window, channels)})
            seen += 1
            for _ in range(self.stride):
                ok, frame = cap.read()
                if ok:
                    last = frame_to_model_array(frame, channels, width, height)
                    window.append(last)
                else:
                    cap.release()
                    return items
        cap.release()
        return items

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.items):
            return None
        item = self.items[self.index]
        self.index += 1
        return item

    def rewind(self) -> None:
        self.index = 0


def quant_type(name: str) -> QuantType:
    return QuantType.QInt8 if name == "qint8" else QuantType.QUInt8


def calibration_method(name: str) -> CalibrationMethod:
    return {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }[name]


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_name, input_shape, output_shape = model_io(model_path)
    reader = VideoWindowCalibrationReader(
        video_path=video_path,
        input_name=input_name,
        input_shape=input_shape,
        samples=args.samples,
        stride=args.stride,
    )
    if not reader.items:
        raise RuntimeError("No calibration samples were collected.")

    extra_options: dict[str, Any] = {}
    if args.extra_options:
        extra_options = {
            "ActivationSymmetric": args.activation_type == "qint8",
            "WeightSymmetric": args.weight_type == "qint8",
            "CalibTensorRangeSymmetric": args.activation_type == "qint8",
        }

    quantize_static(
        model_input=str(model_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=quant_type(args.activation_type),
        weight_type=quant_type(args.weight_type),
        calibrate_method=calibration_method(args.method),
        per_channel=args.per_channel,
        extra_options=extra_options,
    )
    onnx.checker.check_model(onnx.load(str(output_path)))

    summary = {
        "input_model": str(model_path),
        "output_model": str(output_path),
        "calibration_video": str(video_path),
        "input_name": input_name,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "samples_requested": args.samples,
        "samples_used": len(reader.items),
        "stride": args.stride,
        "quant_format": "QDQ",
        "activation_type": args.activation_type,
        "weight_type": args.weight_type,
        "method": args.method,
        "per_channel": args.per_channel,
        "size_mb": output_path.stat().st_size / (1024 * 1024),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
