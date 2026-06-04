"""Export trained VSR checkpoints to ONNX.

The exported model input is NCTHW-like for this project:
    input:  (B, T, C, H, W)
    output: (B, C, H*scale, W*scale)

By default the export uses static H/W, which is the friendliest shape for
TensorRT FP16/INT8 engine building.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import get_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a VSR checkpoint to ONNX.")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pth or a state_dict checkpoint.")
    p.add_argument("--output", required=True, help="Destination .onnx path.")
    p.add_argument("--height", type=int, default=512, help="Input LR frame height.")
    p.add_argument("--width", type=int, default=640, help="Input LR frame width.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--arch", default=None)
    p.add_argument("--scale", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--hidden-channels", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--grayscale", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--dynamic-hw", action="store_true", help="Export dynamic H/W axes instead of fixed size.")
    p.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
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


def build_model(args: argparse.Namespace, cfg: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
    arch = args.arch or cfg.get("arch")
    scale = args.scale or cfg.get("scale")
    if arch is None or scale is None:
        raise ValueError("Could not infer arch/scale; pass --arch and --scale explicitly.")

    grayscale = bool(cfg.get("grayscale", True) if args.grayscale is None else args.grayscale)
    model = get_model(
        str(arch),
        scale=int(scale),
        device=torch.device("cpu"),
        num_channels=1 if grayscale else 3,
        num_frames=int(args.num_frames or cfg.get("num_frames", 1)),
        hidden_channels=args.hidden_channels if args.hidden_channels is not None else cfg.get("hidden_channels"),
        num_blocks=args.num_blocks if args.num_blocks is not None else cfg.get("num_blocks"),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(args: argparse.Namespace) -> dict[str, Any]:
    ckpt_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict, meta = load_checkpoint(ckpt_path)
    cfg = checkpoint_config(meta, ckpt_path)
    model = build_model(args, cfg, state_dict)

    arch = args.arch or cfg.get("arch")
    scale = int(args.scale or cfg.get("scale"))
    num_frames = int(args.num_frames or cfg.get("num_frames", 1))
    grayscale = bool(cfg.get("grayscale", True) if args.grayscale is None else args.grayscale)
    channels = 1 if grayscale else 3
    dummy = torch.randn(args.batch_size, num_frames, channels, args.height, args.width, dtype=torch.float32)

    dynamic_axes = None
    if args.dynamic_hw:
        dynamic_axes = {
            "lr_frames": {0: "batch", 3: "height", 4: "width"},
            "sr_frame": {0: "batch", 2: "height_x_scale", 3: "width_x_scale"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["lr_frames"],
            output_names=["sr_frame"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify

            onnx_model = onnx.load(str(output_path))
            simplified, ok = simplify(onnx_model)
            if ok:
                onnx.save(simplified, str(output_path))
            else:
                print("ONNX simplifier reported check=False; keeping unsimplified model.", file=sys.stderr)
        except Exception as exc:  # pragma: no cover - optional tooling
            print(f"ONNX simplification skipped: {exc}", file=sys.stderr)

    import onnx

    model_proto = onnx.load(str(output_path))
    if args.validate:
        onnx.checker.check_model(model_proto)
    actual_opsets = {item.domain or "ai.onnx": item.version for item in model_proto.opset_import}

    summary = {
        "checkpoint": str(ckpt_path),
        "output": str(output_path),
        "model_name": cfg.get("model_name"),
        "arch": str(arch),
        "scale": scale,
        "num_frames": num_frames,
        "grayscale": grayscale,
        "input_shape": [args.batch_size, num_frames, channels, args.height, args.width],
        "output_shape": [args.batch_size, channels, args.height * scale, args.width * scale],
        "requested_opset": args.opset,
        "opsets": actual_opsets,
        "dynamic_hw": bool(args.dynamic_hw),
        "simplified": bool(args.simplify),
        "size_mb": output_path.stat().st_size / (1024 * 1024),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    export_onnx(args)


if __name__ == "__main__":
    main()
