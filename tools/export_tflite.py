#!/usr/bin/env python3
"""Export trained .pth checkpoints to TFLite format for ESP32 / TFLite Micro.

Usage:
    python tools/export_tflite.py <arch> <pth_path> <out_dir> [options]

    arch: espcn_light | fsrcnn | srcnn

Options:
    --val_dir PATH      HR images for INT8 calibration (default: data/val/DIV2K_valid_HR)
    --tile H W          LR tile size for fixed input shape (default: 64 64)
    --int8              Also produce INT8 quantized model
    --no_c_array        Skip generating C header files

Note on SRCNN:
    SRCNN applies bicubic upsampling inside forward(). TFLite has no bicubic resize op,
    so the exported model strips that step — the input must be pre-upsampled by the host
    before feeding into the TFLite model. The --tile size refers to the LR dimensions;
    the actual model input will be tile*scale (HR dimensions).

Requirements:
    pip install litert-torch
    pip install tensorflow   # only needed for --int8
"""

import sys
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")

ARCHS = ["espcn_light", "fsrcnn", "srcnn"]
SCALE = 2


class SRCNNNoUpsample(nn.Module):
    """SRCNN with bicubic pre-upsampling stripped out.

    TFLite has no bicubic resize op. The caller must upscale the LR image
    with bicubic interpolation before passing it to this model.
    """
    def __init__(self, srcnn):
        super().__init__()
        self.layers = srcnn.layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def load_model(arch: str, pth_path: str) -> nn.Module:
    if arch == "espcn_light":
        from models.espcn_light import ESPCNLight
        model = ESPCNLight(scale_factor=SCALE, num_channels=1)
    elif arch == "fsrcnn":
        from models.fsrcnn import FSRCNN
        model = FSRCNN(scale_factor=SCALE, num_channels=1)
    elif arch == "srcnn":
        from models.srcnn import SRCNN
        model = SRCNN(scale_factor=SCALE, num_channels=1)
    else:
        sys.exit(f"Unknown arch '{arch}'. Choose from: {ARCHS}")

    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    if arch == "srcnn":
        model = SRCNNNoUpsample(model)

    return model


def _input_shape(arch: str, tile_h: int, tile_w: int):
    """Return (H, W) of the model's expected input tensor."""
    if arch == "srcnn":
        return tile_h * SCALE, tile_w * SCALE
    return tile_h, tile_w


def _sample_inputs_from_val(val_dir: str, arch: str, tile_h: int, tile_w: int,
                             n_samples: int = 100):
    """Yield (1,1,H,W) float32 numpy arrays from val images for calibration."""
    in_h, in_w = _input_shape(arch, tile_h, tile_w)
    val_path = Path(val_dir)
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    images = sorted(p for p in val_path.iterdir()
                    if p.is_file() and p.suffix.lower() in exts)[:n_samples]

    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape

        # Ensure image is large enough for the HR crop
        crop_h, crop_w = tile_h * SCALE, tile_w * SCALE
        if h < crop_h or w < crop_w:
            img = cv2.resize(img, (max(w, crop_w), max(h, crop_h)),
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape

        top = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
        left = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0
        hr_crop = img[top:top + crop_h, left:left + crop_w]

        if arch == "srcnn":
            # Simulate host bicubic upsample: LR → bicubic → HR size
            lr = cv2.resize(hr_crop, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
            inp = cv2.resize(lr, (in_w, in_h), interpolation=cv2.INTER_CUBIC)
        else:
            inp = cv2.resize(hr_crop, (in_w, in_h), interpolation=cv2.INTER_CUBIC)

        inp = inp.astype(np.float32) / 255.0
        yield inp[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)


def export_float32(model: nn.Module, sample_input: torch.Tensor, out_path: Path):
    import litert_torch
    edge_model = litert_torch.convert(model.eval(), (sample_input,))
    edge_model.export(str(out_path))
    kb = out_path.stat().st_size / 1024
    print(f"  float32 TFLite → {out_path.name}  ({kb:.1f} KB)")


def export_int8(model: nn.Module, sample_input: torch.Tensor,
                val_dir: str, arch: str, tile_h: int, tile_w: int,
                out_path: Path):
    """INT8 PTQ via PyTorch 2 Export + XNNPack quantizer, then litert-torch convert."""
    try:
        import litert_torch
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPackQuantizer,
            get_symmetric_quantization_config,
        )
    except ImportError as e:
        print(f"  [SKIP] INT8 requires litert-torch + PyTorch 2.x: {e}")
        return

    print("  Preparing model for INT8 quantization...")
    quantizer = XNNPackQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())

    exported = torch.export.export(model, (sample_input,))
    m = prepare_pt2e(exported.module(), quantizer)

    print(f"  Calibrating on {val_dir} ...")
    n = 0
    for batch in _sample_inputs_from_val(val_dir, arch, tile_h, tile_w):
        m(torch.from_numpy(batch))
        n += 1
    print(f"  Calibrated with {n} samples.")

    m = convert_pt2e(m)

    edge_model = litert_torch.convert(m, (sample_input,))
    edge_model.export(str(out_path))
    kb = out_path.stat().st_size / 1024
    print(f"  INT8 TFLite    → {out_path.name}  ({kb:.1f} KB)")


def generate_c_array(tflite_path: Path, out_h_path: Path, var_name: str):
    data = tflite_path.read_bytes()
    lines = [
        f"/* Auto-generated from {tflite_path.name} — do not edit */",
        "#pragma once",
        "#include <stddef.h>",
        "",
        f"const unsigned char {var_name}[] = {{",
    ]
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.append("};")
    lines.append(f"const size_t {var_name}_len = {len(data)};")
    out_h_path.write_text("\n".join(lines) + "\n")
    print(f"  C array        → {out_h_path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arch", choices=ARCHS)
    parser.add_argument("pth_path", help="Path to trained .pth checkpoint")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--val_dir", default="data/val/DIV2K_valid_HR",
                        help="HR val images for INT8 calibration")
    parser.add_argument("--tile", nargs=2, type=int, default=[64, 64],
                        metavar=("H", "W"), help="LR tile size (default: 64 64)")
    parser.add_argument("--int8", action="store_true",
                        help="Also export INT8 quantized model")
    parser.add_argument("--no_c_array", action="store_true",
                        help="Skip generating C header files")
    args = parser.parse_args()

    try:
        import litert_torch  # noqa: F401
    except ImportError:
        sys.exit(
            "litert_torch not installed.\n"
            "  pip install litert-torch\n"
            "See: https://github.com/google-ai-edge/litert-torch"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_h, tile_w = args.tile
    in_h, in_w = _input_shape(args.arch, tile_h, tile_w)

    print(f"\nArch:       {args.arch}")
    print(f"Checkpoint: {args.pth_path}")
    print(f"Tile (LR):  {tile_h}×{tile_w}  →  model input: {in_h}×{in_w}")
    if args.arch == "srcnn":
        print("  Note: SRCNN bicubic upsample stripped — host must pre-upscale input")

    model = load_model(args.arch, args.pth_path)
    sample_input = torch.zeros(1, 1, in_h, in_w)

    # Float32
    float_out = out_dir / f"{args.arch}_float32.tflite"
    print("\n[float32]")
    export_float32(model, sample_input, float_out)
    if not args.no_c_array:
        var = f"{args.arch}_float32_tflite"
        generate_c_array(float_out, out_dir / f"{args.arch}_float32_data.h", var)

    # INT8
    if args.int8:
        print("\n[int8]")
        if not Path(args.val_dir).exists():
            print(f"  [WARN] val_dir not found: {args.val_dir} — skipping INT8")
        else:
            int8_out = out_dir / f"{args.arch}_int8.tflite"
            export_int8(model, sample_input, args.val_dir, args.arch,
                        tile_h, tile_w, int8_out)
            if not args.no_c_array and int8_out.exists():
                var = f"{args.arch}_int8_tflite"
                generate_c_array(int8_out, out_dir / f"{args.arch}_int8_data.h", var)

    print("\nDone.")


if __name__ == "__main__":
    main()
