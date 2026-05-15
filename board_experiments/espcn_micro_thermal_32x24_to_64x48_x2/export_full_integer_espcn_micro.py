#!/usr/bin/env python3
"""Export ESPCN_Micro through Keras as a fully-integer TFLite Micro model."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import torch


def pt2tf(weight: torch.Tensor) -> np.ndarray:
    return weight.detach().cpu().numpy().transpose(2, 3, 1, 0)


def build_model(state_dict: dict, tile_h: int, tile_w: int, scale: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(tile_h, tile_w, 1), batch_size=1, name="input")
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="conv0")(inp)
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu", name="conv1")(x)
    x = tf.keras.layers.Conv2D(scale * scale, 3, padding="same", name="conv2")(x)
    out = tf.keras.layers.Lambda(lambda t: tf.nn.depth_to_space(t, scale),
                                 name="depth_to_space")(x)
    model = tf.keras.Model(inp, out)
    model.get_layer("conv0").set_weights([
        pt2tf(state_dict["feature_extractor.0.weight"]),
        state_dict["feature_extractor.0.bias"].detach().cpu().numpy(),
    ])
    model.get_layer("conv1").set_weights([
        pt2tf(state_dict["feature_extractor.2.weight"]),
        state_dict["feature_extractor.2.bias"].detach().cpu().numpy(),
    ])
    model.get_layer("conv2").set_weights([
        pt2tf(state_dict["feature_extractor.4.weight"]),
        state_dict["feature_extractor.4.bias"].detach().cpu().numpy(),
    ])
    return model


def representative_dataset(val_dir: Path, tile_h: int, tile_w: int, scale: int):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = sorted(p for p in val_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in exts)[:100]
    crop_h, crop_w = tile_h * scale, tile_w * scale
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        if h < crop_h or w < crop_w:
            img = cv2.resize(img, (max(w, crop_w), max(h, crop_h)),
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape
        top = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
        left = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0
        hr = img[top:top + crop_h, left:left + crop_w]
        lr = cv2.resize(hr, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
        arr = (lr.astype(np.float32) / 255.0)[np.newaxis, :, :, np.newaxis]
        yield [arr]


def generate_c_array(tflite_path: Path, out_h_path: Path, var_name: str) -> None:
    data = tflite_path.read_bytes()
    lines = ["#pragma once", "#include <cstddef>", "#include <cstdint>", ""]
    lines.append(f"alignas(16) const unsigned char {var_name}[] = {{")
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    lines.append("};")
    lines.append(f"const size_t {var_name}_len = {len(data)};")
    out_h_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument("--val_dir", default="data/val/DIV2K_valid_HR")
    parser.add_argument("--tile", nargs=2, type=int, default=[24, 32])
    args = parser.parse_args()

    tile_h, tile_w = args.tile
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = state["state_dict"] if "state_dict" in state else state
    scale = int(state.get("scale", 2)) if isinstance(state, dict) else 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(state_dict, tile_h, tile_w, scale)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        Path(args.val_dir), tile_h, tile_w, scale)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite = converter.convert()
    out_tflite = out_dir / "espcn_micro_int8.tflite"
    out_tflite.write_bytes(tflite)
    generate_c_array(out_tflite, out_dir / "espcn_micro_int8_data.h",
                     "espcn_micro_int8_tflite")
    print(f"Wrote {out_tflite} ({out_tflite.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
