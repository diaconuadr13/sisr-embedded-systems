#!/usr/bin/env python3
"""Full-integer INT8 export for the ESP32/Pico board sketches — TF-native (Keras) path.

Why this exists: the onnx2tf `--int8` path in tools/export_tflite.py produced HYBRID
(dynamic-range) models that esp-nn TFLite Micro rejects, and this onnx2tf build emits no
SavedModel to re-quantize. This builds each net directly in Keras, ports the PyTorch
weights, and quantizes with the proven TF converter to a true full-integer int8 model
(int8 weights AND activations, int8 I/O) — the same recipe deployed ESPCN_Micro used.

Safety net: before quantizing, it runs the Keras float32 model and the original PyTorch
model on the same input and asserts they match. A mis-ported layer fails HERE (on the
export machine), not silently as a bad on-board PSNR.

    .disertatie/bin/python cas2026_paper/deploy_kit/export_board_int8.py <arch> <ckpt> <out_dir> \
        --val-dir data/val/InfraredThermal32x24 --tile 24 32

<arch> ∈ {espcn_light, espcn, fsrcnn}. Writes <out_dir>/<arch>_int8.tflite and
<out_dir>/<arch>_int8_data.h with C symbol <arch>_int8_tflite (matches the .ino).
"""
import argparse
from pathlib import Path
import cv2, numpy as np, tensorflow as tf, torch

L = tf.keras.layers

# ESPCN family config: [c0,c1] widths, conv0 kernel, activation.
ESPCN = {
    "espcn_light": dict(chans=[32, 16], k0=5, act="relu"),
    "espcn":       dict(chans=[64, 32], k0=5, act="tanh"),
}


def pt2tf(w):
    """PyTorch conv/deconv weight -> TF/Keras kernel. For Conv2d [O,I,kh,kw]->[kh,kw,I,O];
    for ConvTranspose2d [I,O,kh,kw]->[kh,kw,O,I]. Same axis permutation for both."""
    return w.detach().cpu().numpy().transpose(2, 3, 1, 0)


def build_espcn(sd, H, W, scale, cfg):
    inp = tf.keras.Input(shape=(H, W, 1), batch_size=1, name="input")
    x = L.Conv2D(cfg["chans"][0], cfg["k0"], padding="same", activation=cfg["act"], name="conv0")(inp)
    x = L.Conv2D(cfg["chans"][1], 3, padding="same", activation=cfg["act"], name="conv1")(x)
    x = L.Conv2D(scale * scale, 3, padding="same", name="conv2")(x)
    out = L.Lambda(lambda t: tf.nn.depth_to_space(t, scale), name="depth_to_space")(x)
    m = tf.keras.Model(inp, out)
    for kname, idx in [("conv0", 0), ("conv1", 2), ("conv2", 4)]:
        m.get_layer(kname).set_weights([
            pt2tf(sd[f"feature_extractor.{idx}.weight"]),
            sd[f"feature_extractor.{idx}.bias"].detach().cpu().numpy()])
    return m


def build_fsrcnn(sd, H, W, scale):
    """FSRCNN d=56,s=12,m=4. Deconv is ConvTranspose2d(k9,s2,p4,op1); reproduced exactly in
    TF as Conv2DTranspose('valid') -> Cropping2D((4,3),(4,3)) (removes P from the start and
    P-OP from the end of the (i-1)*s+k 'valid' output)."""
    def prelu(name):
        return L.PReLU(shared_axes=[1, 2], name=name)  # per-channel alpha

    inp = tf.keras.Input(shape=(H, W, 1), batch_size=1, name="input")
    x = L.Conv2D(56, 5, padding="same", name="fe")(inp);      x = prelu("fe_p")(x)
    x = L.Conv2D(12, 1, padding="same", name="shrink")(x);    x = prelu("shrink_p")(x)
    for i in range(4):
        x = L.Conv2D(12, 3, padding="same", name=f"map{i}")(x); x = prelu(f"map{i}_p")(x)
    x = L.Conv2D(56, 1, padding="same", name="expand")(x);    x = prelu("expand_p")(x)
    x = L.Conv2DTranspose(1, 9, strides=scale, padding="valid", name="deconv")(x)
    out = L.Cropping2D(cropping=((4, 3), (4, 3)), name="deconv_crop")(x)
    m = tf.keras.Model(inp, out)

    def set_conv(layer, w, b):
        m.get_layer(layer).set_weights([pt2tf(w), b.detach().cpu().numpy()])
    def set_prelu(layer, w):
        C = w.numel()
        m.get_layer(layer).set_weights([w.detach().cpu().numpy().reshape(1, 1, C)])

    set_conv("fe", sd["feature_extraction.0.weight"], sd["feature_extraction.0.bias"])
    set_prelu("fe_p", sd["feature_extraction.1.weight"])
    set_conv("shrink", sd["shrinking.0.weight"], sd["shrinking.0.bias"])
    set_prelu("shrink_p", sd["shrinking.1.weight"])
    for i in range(4):
        set_conv(f"map{i}", sd[f"mapping.{2*i}.weight"], sd[f"mapping.{2*i}.bias"])
        set_prelu(f"map{i}_p", sd[f"mapping.{2*i+1}.weight"])
    set_conv("expand", sd["expanding.0.weight"], sd["expanding.0.bias"])
    set_prelu("expand_p", sd["expanding.1.weight"])
    set_conv("deconv", sd["deconv.weight"], sd["deconv.bias"])
    return m


def load_torch(arch, ckpt, num_channels=1):
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    scale = int(state.get("scale", 2)) if isinstance(state, dict) else 2
    if arch == "espcn_light":
        from models.espcn_light import ESPCNLight as M
    elif arch == "espcn":
        from models.espcn import ESPCN as M
    elif arch == "fsrcnn":
        from models.fsrcnn import FSRCNN as M
    else:
        raise SystemExit(f"unknown arch {arch}")
    tm = M(scale_factor=scale, num_channels=num_channels)
    tm.load_state_dict(sd)
    tm.eval()
    return tm, sd, scale


def rep_dataset(val_dir, H, W, scale):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = sorted(p for p in Path(val_dir).iterdir() if p.suffix.lower() in exts)[:100]
    ch, cw = H * scale, W * scale
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        if h < ch or w < cw:
            img = cv2.resize(img, (max(w, cw), max(h, ch)), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape
        top = np.random.randint(0, h - ch + 1) if h > ch else 0
        left = np.random.randint(0, w - cw + 1) if w > cw else 0
        lr = cv2.resize(img[top:top+ch, left:left+cw], (W, H), interpolation=cv2.INTER_CUBIC)
        yield [(lr.astype(np.float32) / 255.0)[np.newaxis, :, :, np.newaxis]]


def c_array(tfl_path, out_h, var):
    data = Path(tfl_path).read_bytes()
    lines = ["#pragma once", "#include <cstddef>", "#include <cstdint>", "",
             f"alignas(16) const unsigned char {var}[] = {{"]
    for i in range(0, len(data), 16):
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in data[i:i+16]) + ",")
    lines += ["};", f"const size_t {var}_len = {len(data)};", ""]
    Path(out_h).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", choices=["espcn_light", "espcn", "fsrcnn"])
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--val-dir", default="data/val/InfraredThermal32x24")
    ap.add_argument("--tile", nargs=2, type=int, default=[24, 32], metavar=("H", "W"))
    a = ap.parse_args()
    H, W = a.tile
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    tmodel, sd, scale = load_torch(a.arch, a.ckpt)
    print(f"[{a.arch}] scale={scale}  tile(LR)={H}x{W}")
    if a.arch in ESPCN:
        kmodel = build_espcn(sd, H, W, scale, ESPCN[a.arch])
    else:
        kmodel = build_fsrcnn(sd, H, W, scale)

    # --- parity check: Keras float32 must match PyTorch, else a layer was mis-ported ---
    xt = np.random.rand(1, 1, H, W).astype(np.float32)
    with torch.no_grad():
        yt = tmodel(torch.from_numpy(xt)).numpy()               # NCHW
    yk = np.asarray(kmodel(xt.transpose(0, 2, 3, 1))).transpose(0, 3, 1, 2)  # -> NCHW
    if yt.shape != yk.shape:
        raise SystemExit(f"PARITY: shape mismatch torch{yt.shape} vs keras{yk.shape}")
    dmax = float(np.abs(yt - yk).max())
    print(f"[{a.arch}] parity max|Δ(torch,keras)| = {dmax:.3e}  (out shape {yt.shape})")
    if dmax > 1e-3:
        raise SystemExit(f"PARITY FAILED ({dmax:.3e} > 1e-3) — Keras port does not match PyTorch. "
                         "Do NOT flash; report this number.")
    print(f"[{a.arch}] parity OK")

    # --- full-integer int8 (int8 I/O) ---
    conv = tf.lite.TFLiteConverter.from_keras_model(kmodel)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: rep_dataset(a.val_dir, H, W, scale)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()

    tfl_path = out_dir / f"{a.arch}_int8.tflite"
    tfl_path.write_bytes(tfl)
    interp = tf.lite.Interpreter(model_content=tfl)
    in_dt = interp.get_input_details()[0]["dtype"].__name__
    out_dt = interp.get_output_details()[0]["dtype"].__name__
    print(f"[{a.arch}] INT8 tflite -> {tfl_path.name}  ({len(tfl)/1024:.1f} KB)  IO: in={in_dt} out={out_dt}")
    if in_dt != "int8" or out_dt != "int8":
        raise SystemExit(f"not full-integer (in={in_dt}, out={out_dt}); TFLite Micro would reject it.")
    c_array(tfl_path, out_dir / f"{a.arch}_int8_data.h", f"{a.arch}_int8_tflite")
    print(f"[{a.arch}] header      -> {a.arch}_int8_data.h  (symbol {a.arch}_int8_tflite)")


if __name__ == "__main__":
    main()
