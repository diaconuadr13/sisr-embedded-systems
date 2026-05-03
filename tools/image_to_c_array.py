#!/usr/bin/env python3
"""Convert a grayscale image patch to a C header for ESP32 firmware testing.

Usage:
    python tools/image_to_c_array.py <img_path> <out_h> --size H W [--arch srcnn]

Example:
    python tools/image_to_c_array.py data/val/DIV2K_valid_HR/0801.png \
        esp32_firmware/test_image.h --size 64 64
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path")
    parser.add_argument("out_h")
    parser.add_argument("--size", nargs=2, type=int, default=[64, 64], metavar=("H", "W"))
    parser.add_argument("--arch", default="espcn_light",
                        choices=["espcn_light", "fsrcnn", "srcnn", "edsr_tiny"],
                        help="Target arch — SRCNN input is bicubic pre-upsampled")
    parser.add_argument("--center", action="store_true",
                        help="Crop from image center instead of top-left")
    parser.add_argument("--offset", nargs=2, type=int, default=None,
                        metavar=("Y", "X"),
                        help="Top-left corner of crop in HR pixels (overrides --center)")
    args = parser.parse_args()

    scale = 2
    tile_h, tile_w = args.size

    img = cv2.imread(args.img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"Cannot read: {args.img_path}")

    # Validate tile dimensions
    if tile_h < 4 or tile_w < 4:
        print(f"  [WARN] very small tile ({tile_w}x{tile_h}) may cause model inference issues")
    if tile_h != tile_w:
        print(f"  [WARN] non-square tile ({tile_w}x{tile_h}) — ensure model supports rectangular input")

    hr_h, hr_w = tile_h * scale, tile_w * scale
    h, w = img.shape
    if h < hr_h or w < hr_w:
        sys.exit(f"Image too small ({w}×{h}), need at least {hr_w}×{hr_h}")

    if args.offset is not None:
        y0, x0 = args.offset
    elif args.center:
        y0 = (h - hr_h) // 2
        x0 = (w - hr_w) // 2
    else:
        y0, x0 = 0, 0

    y0 = max(0, min(y0, h - hr_h))
    x0 = max(0, min(x0, w - hr_w))
    hr_patch = img[y0:y0 + hr_h, x0:x0 + hr_w]

    # LR: downsample HR patch
    lr_patch = cv2.resize(hr_patch, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)

    if args.arch == "srcnn":
        # SRCNN expects bicubic pre-upsampled input (model has no upsample op)
        inp = cv2.resize(lr_patch, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        in_h, in_w = hr_h, hr_w
    else:
        inp = lr_patch
        in_h, in_w = tile_h, tile_w

    inp_f32 = inp.astype(np.float32) / 255.0  # [0,1] float32, shape (in_h, in_w)
    flat = inp_f32.flatten()

    out = Path(args.out_h)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"/* Auto-generated test input — {Path(args.img_path).name} cropped to {in_w}x{in_h} grayscale */",
        "#pragma once",
        "#include <stddef.h>",
        "",
        f"static const int TEST_INPUT_H = {in_h};",
        f"static const int TEST_INPUT_W = {in_w};",
        f"static const int TEST_INPUT_LEN = {len(flat)};",
        "",
        "static const float test_input_data[] = {",
    ]
    for i in range(0, len(flat), 8):
        chunk = flat[i:i + 8]
        lines.append("  " + ", ".join(f"{v:.6f}f" for v in chunk) + ",")
    lines.append("};")

    # Also save the HR patch for PSNR calculation on PC
    hr_f32 = hr_patch.astype(np.float32) / 255.0
    hr_flat = hr_f32.flatten()
    lines += [
        "",
        f"static const int TEST_HR_H = {hr_h};",
        f"static const int TEST_HR_W = {hr_w};",
        f"static const int TEST_HR_LEN = {len(hr_flat)};",
        "",
        "static const float test_hr_data[] = {",
    ]
    for i in range(0, len(hr_flat), 8):
        chunk = hr_flat[i:i + 8]
        lines.append("  " + ", ".join(f"{v:.6f}f" for v in chunk) + ",")
    lines.append("};")

    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}  ({in_w}×{in_h} input, {hr_w}×{hr_h} HR reference)")


if __name__ == "__main__":
    main()
