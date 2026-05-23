import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from models import get_model, list_models
from utils.device import configure_runtime, resolve_device
from utils.video_io import is_video_file, list_frame_paths, make_temporal_window, read_video_frames, save_frames, write_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run video super-resolution on a video file or frame directory.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--arch", type=str, default=None, choices=list_models())
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--hidden_channels", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--save_frames", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frame_by_frame", action="store_true")
    return parser.parse_args()


def autocast_context(use_amp: bool) -> Any:
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def load_frames(input_path: Path, max_frames: int | None) -> tuple[list[np.ndarray], float]:
    if input_path.is_dir():
        paths = list_frame_paths(input_path)
        if max_frames is not None:
            paths = paths[:max_frames]
        if not paths:
            raise FileNotFoundError(f"No image frames found in: {input_path}")
        frames = []
        for path in paths:
            img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(f"Failed to read frame: {path}")
            frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return frames, 30.0
    if is_video_file(input_path):
        return read_video_frames(input_path, max_frames=max_frames)
    raise ValueError(f"Input must be a video file or directory of frames: {input_path}")


def metadata_from_checkpoint(weights_path: Path, checkpoint: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if isinstance(checkpoint, dict):
        for key in ("arch", "scale", "num_frames", "hidden_channels", "grayscale", "config"):
            if key in checkpoint:
                metadata[key] = checkpoint[key]
    config_path = weights_path.with_name("config.json")
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metadata.setdefault("config", config)
        for key in ("arch", "scale", "num_frames", "hidden_channels", "grayscale"):
            metadata.setdefault(key, config.get(key))
    return metadata


def frame_to_tensor(frame_rgb: np.ndarray, grayscale: bool) -> torch.Tensor:
    if grayscale:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        return torch.from_numpy(np.ascontiguousarray(gray)).unsqueeze(0).float() / 255.0
    return torch.from_numpy(np.ascontiguousarray(frame_rgb)).permute(2, 0, 1).float() / 255.0


def tensor_to_frame(sr: torch.Tensor, grayscale: bool) -> np.ndarray:
    sr = torch.clamp(sr.detach().cpu(), 0.0, 1.0)
    if grayscale:
        gray = (sr.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return (sr.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def make_video_input(window: list[np.ndarray], grayscale: bool) -> torch.Tensor:
    return torch.cat([frame_to_tensor(frame, grayscale) for frame in window], dim=0)


def infer_batches(model: torch.nn.Module, inputs: list[torch.Tensor], device: torch.device, use_amp: bool, batch_size: int) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.stack(inputs[start:start + batch_size], dim=0).to(device)
            with autocast_context(use_amp):
                sr = torch.clamp(model(batch), 0.0, 1.0)
            outputs.extend([item.cpu() for item in sr])
    return outputs


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    checkpoint = torch.load(weights_path, map_location="cpu")
    metadata = metadata_from_checkpoint(weights_path, checkpoint)
    config = metadata.get("config") or {}

    arch = args.arch or metadata.get("arch") or config.get("arch")
    if arch is None:
        raise ValueError(f"Unable to determine architecture. Pass --arch. Available: {', '.join(list_models())}")
    scale = int(args.scale or metadata.get("scale") or config.get("scale") or 2)
    num_frames = int(args.num_frames or metadata.get("num_frames") or config.get("num_frames") or 3)
    hidden_channels = int(args.hidden_channels or metadata.get("hidden_channels") or config.get("hidden_channels") or 32)
    grayscale = bool(args.grayscale or metadata.get("grayscale") or config.get("grayscale", False))
    channels = 1 if grayscale else 3

    device = resolve_device(args.device)
    configure_runtime(device)
    use_amp = args.amp and device.type == "cuda"

    model_kwargs = {}
    is_video_arch = str(arch).lower().startswith("video")
    if is_video_arch:
        model_kwargs = {"num_frames": num_frames, "hidden_channels": hidden_channels}
    model = get_model(arch, scale=scale, device=device, num_channels=channels, **model_kwargs)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    frames, source_fps = load_frames(Path(args.input), args.max_frames)
    out_fps = float(args.fps or source_fps or 30.0)

    inputs: list[torch.Tensor] = []
    for idx, frame in enumerate(frames):
        if is_video_arch and not args.frame_by_frame:
            window = make_temporal_window(frames, idx, num_frames)
            inputs.append(make_video_input(window, grayscale))
        elif is_video_arch and args.frame_by_frame:
            window = [frame] * num_frames
            inputs.append(make_video_input(window, grayscale))
        else:
            inputs.append(frame_to_tensor(frame, grayscale))

    sr_tensors = infer_batches(model, inputs, device, use_amp, max(1, int(args.batch_size)))
    sr_frames = [tensor_to_frame(sr, grayscale) for sr in sr_tensors]

    output_path = Path(args.output)
    if output_path.suffix.lower() in {".mp4", ".avi", ".mov"}:
        write_video(output_path, sr_frames, fps=out_fps, codec=args.codec)
    else:
        save_frames(output_path, sr_frames)
    if args.save_frames:
        save_frames(Path(args.save_frames), sr_frames)

    print(f"[video_infer] wrote {len(sr_frames)} frames to {output_path}")


if __name__ == "__main__":
    main()
