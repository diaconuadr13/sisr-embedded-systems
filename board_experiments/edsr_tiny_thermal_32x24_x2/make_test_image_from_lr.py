#!/usr/bin/env python3
"""Generate a board test_image.h from an already-low-resolution grayscale image."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def format_float_array(values: np.ndarray) -> list[str]:
    lines = []
    for i in range(0, len(values), 8):
        chunk = values[i:i + 8]
        lines.append("  " + ", ".join(f"{v:.6f}f" for v in chunk) + ",")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path")
    parser.add_argument("out_h")
    parser.add_argument("--size", nargs=2, type=int, default=[24, 32],
                        metavar=("H", "W"))
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    in_h, in_w = args.size
    scale = args.scale
    hr_h, hr_w = in_h * scale, in_w * scale

    img = cv2.imread(args.img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Cannot read: {args.img_path}")

    if img.shape != (in_h, in_w):
        img = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_AREA)

    hr_ref = cv2.resize(img, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
    inp_flat = (img.astype(np.float32) / 255.0).flatten()
    hr_flat = (hr_ref.astype(np.float32) / 255.0).flatten()

    out = Path(args.out_h)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"/* Auto-generated test input from LR image {Path(args.img_path).name} */",
        "#pragma once",
        "#include <stddef.h>",
        "",
        f"static const int TEST_INPUT_H = {in_h};",
        f"static const int TEST_INPUT_W = {in_w};",
        f"static const int TEST_INPUT_LEN = {len(inp_flat)};",
        "",
        "static const float test_input_data[] = {",
        *format_float_array(inp_flat),
        "};",
        "",
        f"static const int TEST_HR_H = {hr_h};",
        f"static const int TEST_HR_W = {hr_w};",
        f"static const int TEST_HR_LEN = {len(hr_flat)};",
        "",
        "static const float test_hr_data[] = {",
        *format_float_array(hr_flat),
        "};",
        "",
    ]
    out.write_text("\n".join(lines))
    print(f"Wrote {out} ({in_w}x{in_h} input, {hr_w}x{hr_h} bicubic reference)")


if __name__ == "__main__":
    main()
