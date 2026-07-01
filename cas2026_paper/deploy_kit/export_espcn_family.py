#!/usr/bin/env python3
"""Export the ESPCN family (Micro / Light / Full) to full-integer INT8 TFLite for TFLM.

Generalises the proven board_experiments/.../export_full_integer_espcn_micro.py to the
three channel widths. Same topology (3 conv + depth_to_space); only widths, conv0 kernel,
and activation differ:
    micro : 16-> 8 ->4,  k=3,3,3, ReLU  (already deployed; here for parity)
    light : 32->16 ->4,  k=5,3,3, ReLU
    full  : 64->32 ->4,  k=5,3,3, Tanh   (Tanh is NOT fused -> firmware must AddTanh())

RUN ON YOUR MACHINE in the .disertatie env (this sandbox has no torch/TF/PyPI):
    .disertatie/bin/python cas2026_paper/deploy_kit/export_espcn_family.py light \
        --runs-dir ../runs --val-dir data/val/InfraredThermal32x24 \
        --out board_experiments/espcn_light_thermal_32x24_to_64x48_x2

Produces: <out>/espcn_<variant>_int8.tflite  and  espcn_<variant>_int8_data.h
"""
import argparse
from pathlib import Path
import cv2, numpy as np, tensorflow as tf, torch

VARIANTS = {
    # widths [c0,c1]; conv0 kernel; activation; default best-PSNR thermal checkpoint
    "micro": dict(chans=[16, 8],  k0=3, act="relu",
                  ckpt="ESPCN_Micro_thermal_gray_x2/InfraredThermal32x24/exp_20260510_200842/best_model.pth"),
    "light": dict(chans=[32, 16], k0=5, act="relu",
                  ckpt="ESPCN_Light_thermal_gray_x2/InfraredThermal32x24/exp_20260510_194053/best_model.pth"),
    "full":  dict(chans=[64, 32], k0=5, act="tanh",
                  ckpt="ESPCN_thermal_gray_x2/InfraredThermal32x24/exp_20260510_200719/best_model.pth"),
}

def pt2tf(w): return w.detach().cpu().numpy().transpose(2, 3, 1, 0)

def build_model(sd, H, W, scale, chans, k0, act):
    inp = tf.keras.Input(shape=(H, W, 1), batch_size=1, name="input")
    x = tf.keras.layers.Conv2D(chans[0], k0, padding="same", activation=act, name="conv0")(inp)
    x = tf.keras.layers.Conv2D(chans[1], 3, padding="same", activation=act, name="conv1")(x)
    x = tf.keras.layers.Conv2D(scale * scale, 3, padding="same", name="conv2")(x)
    out = tf.keras.layers.Lambda(lambda t: tf.nn.depth_to_space(t, scale), name="depth_to_space")(x)
    m = tf.keras.Model(inp, out)
    for keras_name, pt_idx in [("conv0", 0), ("conv1", 2), ("conv2", 4)]:
        m.get_layer(keras_name).set_weights([
            pt2tf(sd[f"feature_extractor.{pt_idx}.weight"]),
            sd[f"feature_extractor.{pt_idx}.bias"].detach().cpu().numpy()])
    return m

def rep_dataset(val_dir, H, W, scale):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = sorted(p for p in Path(val_dir).iterdir() if p.suffix.lower() in exts)[:100]
    ch, cw = H * scale, W * scale
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        h, w = img.shape
        if h < ch or w < cw:
            img = cv2.resize(img, (max(w, cw), max(h, ch)), interpolation=cv2.INTER_CUBIC); h, w = img.shape
        top = np.random.randint(0, h - ch + 1) if h > ch else 0
        left = np.random.randint(0, w - cw + 1) if w > cw else 0
        lr = cv2.resize(img[top:top+ch, left:left+cw], (W, H), interpolation=cv2.INTER_CUBIC)
        yield [(lr.astype(np.float32) / 255.0)[np.newaxis, :, :, np.newaxis]]

def c_array(tflite_path, out_h, var):
    data = Path(tflite_path).read_bytes()
    out = ["#pragma once", "#include <cstddef>", "#include <cstdint>", "",
           f"alignas(16) const unsigned char {var}[] = {{"]
    for i in range(0, len(data), 16):
        out.append("  " + ", ".join(f"0x{b:02x}" for b in data[i:i+16]) + ",")
    out += ["};", f"const size_t {var}_len = {len(data)};", ""]
    Path(out_h).write_text("\n".join(out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=list(VARIANTS))
    ap.add_argument("--runs-dir", default="../runs")
    ap.add_argument("--checkpoint", default=None, help="override the default best checkpoint")
    ap.add_argument("--val-dir", default="data/val/InfraredThermal32x24",
                    help="thermal HR images for INT8 calibration")
    ap.add_argument("--tile", nargs=2, type=int, default=[24, 32], help="LR tile H W")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    v = VARIANTS[a.variant]
    ckpt = a.checkpoint or str(Path(a.runs_dir) / v["ckpt"])
    H, W = a.tile
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    scale = int(state.get("scale", 2)) if isinstance(state, dict) else 2
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(sd, H, W, scale, v["chans"], v["k0"], v["act"])
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: rep_dataset(a.val_dir, H, W, scale)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()

    name = f"espcn_{a.variant}"
    tfl_path = out_dir / f"{name}_int8.tflite"; tfl_path.write_bytes(tfl)
    c_array(tfl_path, out_dir / f"{name}_int8_data.h", f"{name}_int8_tflite")
    print(f"[{a.variant}] wrote {tfl_path}  ({tfl_path.stat().st_size/1024:.1f} KB flash)")
    print(f"    checkpoint: {ckpt}")
    print(f"    firmware: reuse the ESPCN_Micro sketch; set var '{name}_int8_tflite',"
          f" include '{name}_int8_data.h'"
          + (";  ADD resolver.AddTanh() and bump MicroMutableOpResolver<3>" if v['act']=='tanh'
             else "  (ReLU is fused; ops stay CONV_2D + DEPTH_TO_SPACE)"))

if __name__ == "__main__":
    main()
