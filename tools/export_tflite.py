#!/usr/bin/env python3
"""Export trained .pth checkpoints to TFLite format for ESP32 / TFLite Micro.

Usage:
    python tools/export_tflite.py <arch> <pth_path> <out_dir> [options]

    arch: espcn_light | fsrcnn | srcnn | edsr_tiny

Options:
    --val_dir PATH      HR images for INT8 calibration (default: data/val/DIV2K_valid_HR)
    --tile H W          LR tile size for fixed input shape (default: 64 64)
    --int8              Also produce INT8 quantized model
    --no_c_array        Skip generating C header files

Conversion backends:
    espcn_light  — onnx2tf (PyTorch → ONNX → TFLite). litert-torch decomposes
                   PixelShuffle into 6D transposes unsupported by TFLite Micro;
                   onnx2tf maps it to DEPTH_TO_SPACE instead. Output is NHWC,
                   float32 IO, INT8 weights+activations.
    edsr_tiny    — Keras (PyTorch weights → Keras → TF TFLiteConverter). res_scale
                   absorbed into conv weights. Fully fused INT8; no hybrid tensors.
    fsrcnn       — litert_torch (PT2E dynamic quantizer). Outputs float32 TFLite
                   (INT8 not yet tested for this architecture).
    srcnn        — litert_torch. Outputs float32 TFLite only (INT8 not yet tested).

Note on SRCNN:
    SRCNN applies bicubic upsampling inside forward(). TFLite has no bicubic resize op,
    so the exported model strips that step — the input must be pre-upsampled by the host
    before feeding into the TFLite model. The --tile size refers to the LR dimensions;
    the actual model input will be tile*scale (HR dimensions).

Requirements:
    pip install litert-torch              # fsrcnn, srcnn
    pip install onnx onnxruntime onnx2tf   # espcn_light
    pip install tensorflow                 # --int8 with espcn_light
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

ARCHS = ["espcn_light", "fsrcnn", "srcnn", "edsr_tiny"]


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


def _detect_channels(state_dict: dict) -> int:
    for key, tensor in state_dict.items():
        if tensor.ndim == 4:
            return tensor.shape[1]
    return 3


def load_model(arch: str, pth_path: str):
    """Load checkpoint, detect channels/scale, return (model, scale, num_channels)."""
    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    num_channels = _detect_channels(state_dict)
    scale = int(state.get("scale", 2)) if isinstance(state, dict) else 2

    if arch == "espcn_light":
        from models.espcn_light import ESPCNLight
        model = ESPCNLight(scale_factor=scale, num_channels=num_channels)
    elif arch == "fsrcnn":
        from models.fsrcnn import FSRCNN
        model = FSRCNN(scale_factor=scale, num_channels=num_channels)
    elif arch == "srcnn":
        from models.srcnn import SRCNN
        model = SRCNN(scale_factor=scale, num_channels=num_channels)
    elif arch == "edsr_tiny":
        from models.edsr_tiny import EDSRTiny
        model = EDSRTiny(scale_factor=scale, num_channels=num_channels)
    else:
        sys.exit(f"Unknown arch '{arch}'. Choose from: {ARCHS}")

    model.load_state_dict(state_dict)
    model.eval()

    if arch == "srcnn":
        model = SRCNNNoUpsample(model)

    return model, scale, num_channels


def _input_shape(arch: str, tile_h: int, tile_w: int, scale: int):
    """Return (H, W) of the model's expected input tensor."""
    if arch == "srcnn":
        return tile_h * scale, tile_w * scale
    return tile_h, tile_w


def _sample_inputs_from_val(val_dir: str, arch: str, tile_h: int, tile_w: int,
                             scale: int, n_samples: int = 100):
    """Yield (1,1,H,W) float32 numpy arrays from val images for calibration."""
    in_h, in_w = _input_shape(arch, tile_h, tile_w, scale)
    val_path = Path(val_dir)
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    images = sorted(p for p in val_path.iterdir()
                    if p.is_file() and p.suffix.lower() in exts)[:n_samples]

    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape

        crop_h, crop_w = tile_h * scale, tile_w * scale
        if h < crop_h or w < crop_w:
            img = cv2.resize(img, (max(w, crop_w), max(h, crop_h)),
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape

        top = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
        left = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0
        hr_crop = img[top:top + crop_h, left:left + crop_w]

        if arch == "srcnn":
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


def _build_keras_srcnn(model: nn.Module, in_h: int, in_w: int, num_channels: int = 1):
    """Reconstruct SRCNN as a Keras model with fixed input shape and PyTorch weights."""
    import tensorflow as tf
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]

    def pt2tf(w):
        return w.detach().numpy().transpose(2, 3, 1, 0)

    inp = tf.keras.Input(shape=(in_h, in_w, num_channels), batch_size=1, name="input")
    x = tf.keras.layers.Conv2D(convs[0].out_channels, convs[0].kernel_size[0],
                                padding="same", activation="relu", name="conv1")(inp)
    x = tf.keras.layers.Conv2D(convs[1].out_channels, convs[1].kernel_size[0],
                                padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.Conv2D(convs[2].out_channels, convs[2].kernel_size[0],
                                padding="same", name="conv3")(x)
    keras_model = tf.keras.Model(inp, x)
    keras_model.get_layer("conv1").set_weights(
        [pt2tf(convs[0].weight), convs[0].bias.detach().numpy()])
    keras_model.get_layer("conv2").set_weights(
        [pt2tf(convs[1].weight), convs[1].bias.detach().numpy()])
    keras_model.get_layer("conv3").set_weights(
        [pt2tf(convs[2].weight), convs[2].bias.detach().numpy()])
    return keras_model


def _build_keras_edsr_tiny(torch_model: nn.Module, tile_h: int, tile_w: int,
                            scale: int = 2, num_feats: int = 32, num_blocks: int = 8,
                            num_channels: int = 1):
    """Build a Keras EDSR_Tiny with weights transferred from a PyTorch checkpoint.

    res_scale=0.1 is absorbed into the second conv weight/bias of each ResBlock
    to eliminate the runtime scalar Mul op (which the TF int8 quantizer can't fuse).
    """
    import tensorflow as tf

    sd = {k: v.detach() for k, v in torch_model.state_dict().items()}

    def pt2tf(w):  # PyTorch OIHW → Keras HWIO
        return w.numpy().transpose(2, 3, 1, 0)

    inp = tf.keras.Input(shape=(tile_h, tile_w, num_channels), batch_size=1, name="input")

    head = tf.keras.layers.Conv2D(num_feats, 3, padding="same", name="head")(inp)

    x = head
    for i in range(num_blocks):
        res = x
        x = tf.keras.layers.Conv2D(num_feats, 3, padding="same", activation="relu",
                                   name=f"res{i}_c1")(x)
        x = tf.keras.layers.Conv2D(num_feats, 3, padding="same", name=f"res{i}_c2")(x)
        x = tf.keras.layers.Add(name=f"res{i}_add")([res, x])

    x = tf.keras.layers.Conv2D(num_feats, 3, padding="same", name="body_end")(x)
    x = tf.keras.layers.Add(name="global_add")([head, x])

    x = tf.keras.layers.Conv2D(num_channels * scale * scale, 3, padding="same",
                                name="upsample_conv")(x)
    x = tf.keras.layers.Lambda(lambda t: tf.nn.depth_to_space(t, scale),
                                name="pixel_shuffle")(x)

    keras_model = tf.keras.Model(inputs=inp, outputs=x)

    RES_SCALE = 0.1
    keras_model.get_layer("head").set_weights(
        [pt2tf(sd["head.weight"]), sd["head.bias"].numpy()])
    for i in range(num_blocks):
        keras_model.get_layer(f"res{i}_c1").set_weights([
            pt2tf(sd[f"body.{i}.block.0.weight"]),
            sd[f"body.{i}.block.0.bias"].numpy(),
        ])
        # Bake res_scale into second conv so no runtime Mul op is needed
        keras_model.get_layer(f"res{i}_c2").set_weights([
            pt2tf(sd[f"body.{i}.block.2.weight"]) * RES_SCALE,
            sd[f"body.{i}.block.2.bias"].numpy() * RES_SCALE,
        ])
    keras_model.get_layer("body_end").set_weights(
        [pt2tf(sd["body_end.weight"]), sd["body_end.bias"].numpy()])
    keras_model.get_layer("upsample_conv").set_weights(
        [pt2tf(sd["upsample.0.weight"]), sd["upsample.0.bias"].numpy()])

    return keras_model


def export_edsr_tiny_via_keras(model: nn.Module, arch: str, out_dir: Path,
                                do_int8: bool, val_dir: str,
                                tile_h: int, tile_w: int, no_c_array: bool,
                                scale: int, num_channels: int):
    """Export EDSR_Tiny to TFLite via Keras + TF converter.

    onnx2tf int8 quantization leaves Add/DepthToSpace activations as float32
    (hybrid model) which espressif TFLite Micro rejects. The Keras path produces
    fully fused int8 ops with QUANTIZE/DEQUANTIZE only at the IO boundary.
    """
    import tensorflow as tf
    import numpy as np

    keras_model = _build_keras_edsr_tiny(model, tile_h, tile_w, num_channels=num_channels)

    rng = np.random.default_rng(42)
    sample_np = rng.random((1, tile_h, tile_w, num_channels)).astype(np.float32)
    with torch.no_grad():
        pt_out = model(torch.from_numpy(
            sample_np.transpose(0, 3, 1, 2))).numpy().transpose(0, 2, 3, 1)
    keras_out = keras_model(sample_np).numpy()
    diff = float(np.abs(pt_out - keras_out).max())
    label = "OK" if diff < 1e-3 else "WARN — check weight transfer"
    print(f"  PyTorch/Keras sanity diff: {diff:.2e}  ({label})")

    print("\n[float32]")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    f32_bytes = converter.convert()
    f32_path = out_dir / f"{arch}_float32.tflite"
    f32_path.write_bytes(f32_bytes)
    print(f"  float32 TFLite → {f32_path.name}  ({f32_path.stat().st_size / 1024:.1f} KB)")
    if not no_c_array:
        generate_c_array(f32_path, out_dir / f"{arch}_float32_data.h",
                         f"{arch}_float32_tflite")

    if do_int8:
        print("\n[int8]")
        batches = list(_sample_inputs_from_val(val_dir, arch, tile_h, tile_w, scale))
        if not batches:
            print(f"  [WARN] no calibration images in {val_dir} — skipping INT8")
            return
        calib_nhwc = [b.transpose(0, 2, 3, 1).astype(np.float32) for b in batches]
        print(f"  Calibrating with {len(calib_nhwc)} samples.")

        def representative_dataset():
            for b in calib_nhwc:
                yield [b]

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        int8_bytes = converter.convert()
        int8_path = out_dir / f"{arch}_int8.tflite"
        int8_path.write_bytes(int8_bytes)
        print(f"  INT8 TFLite    → {int8_path.name}  ({int8_path.stat().st_size / 1024:.1f} KB)")
        if not no_c_array:
            generate_c_array(int8_path, out_dir / f"{arch}_int8_data.h",
                             f"{arch}_int8_tflite")


def export_srcnn_via_keras(model: nn.Module, arch: str, out_dir: Path,
                           do_int8: bool, val_dir: str,
                           tile_h: int, tile_w: int, no_c_array: bool,
                           scale: int, num_channels: int):
    """Export SRCNN to TFLite via Keras + TF converter.

    litert-torch decomposes SRCNN into DEPTHWISE_CONV_2D + TRANSPOSEs and
    produces fake-quantized (QUANT/DEQUANT pairs) rather than fused int8 ops.
    The Keras path goes PyTorch weights → Keras model → TFLite converter,
    which correctly produces fully fused int8 CONV_2D ops with int32 biases.
    """
    import tensorflow as tf
    import numpy as np

    in_h, in_w = _input_shape(arch, tile_h, tile_w, scale)
    keras_model = _build_keras_srcnn(model, in_h, in_w, num_channels)

    print("\n[float32]")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    f32_bytes = converter.convert()
    f32_path = out_dir / f"{arch}_float32.tflite"
    f32_path.write_bytes(f32_bytes)
    print(f"  float32 TFLite → {f32_path.name}  ({f32_path.stat().st_size / 1024:.1f} KB)")
    if not no_c_array:
        generate_c_array(f32_path, out_dir / f"{arch}_float32_data.h", f"{arch}_float32_tflite")

    if do_int8:
        print("\n[int8]")
        batches = list(_sample_inputs_from_val(val_dir, arch, tile_h, tile_w, scale))
        if not batches:
            print(f"  [WARN] no calibration images in {val_dir} — skipping INT8")
            return
        calib_nhwc = [b.transpose(0, 2, 3, 1).astype(np.float32) for b in batches]
        print(f"  Calibrating with {len(calib_nhwc)} samples.")

        def representative_dataset():
            for b in calib_nhwc:
                yield [b]

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        int8_bytes = converter.convert()
        int8_path = out_dir / f"{arch}_int8.tflite"
        int8_path.write_bytes(int8_bytes)
        print(f"  INT8 TFLite    → {int8_path.name}  ({int8_path.stat().st_size / 1024:.1f} KB)")
        if not no_c_array:
            generate_c_array(int8_path, out_dir / f"{arch}_int8_data.h", f"{arch}_int8_tflite")


def export_int8(model: nn.Module, sample_input: torch.Tensor,
                val_dir: str, arch: str, tile_h: int, tile_w: int,
                out_path: Path, scale: int):
    """INT8 PTQ via litert_torch PT2EQuantizer + torchao calibration."""
    try:
        import litert_torch
        from litert_torch.quantize import pt2e_quantizer as ltq
        from torchao.quantization.pt2e import quantize_pt2e
    except ImportError as e:
        print(f"  [SKIP] INT8 requires litert-torch + torchao: {e}")
        return

    print("  Preparing model for INT8 quantization...")
    quantizer = ltq.PT2EQuantizer().set_global(
        ltq.get_symmetric_quantization_config(is_per_channel=False)
    )

    exported = torch.export.export(model, (sample_input,))
    m = quantize_pt2e.prepare_pt2e(exported.module(), quantizer)

    print(f"  Calibrating on {val_dir} ...")
    n = 0
    for batch in _sample_inputs_from_val(val_dir, arch, tile_h, tile_w, scale):
        m(torch.from_numpy(batch))
        n += 1
    print(f"  Calibrated with {n} samples.")

    m = quantize_pt2e.convert_pt2e(m, fold_quantize=False)

    edge_model = litert_torch.convert(m, (sample_input,))
    edge_model.export(str(out_path))
    kb = out_path.stat().st_size / 1024
    print(f"  INT8 TFLite    → {out_path.name}  ({kb:.1f} KB)")


def export_via_onnx2tf(model: nn.Module, sample_input: torch.Tensor,
                       arch: str, out_dir: Path, do_int8: bool,
                       val_dir: str, tile_h: int, tile_w: int, no_c_array: bool,
                       scale: int):
    """Export to TFLite via PyTorch → ONNX → onnx2tf.

    Used for models with PixelShuffle (e.g. ESPCN_Light). litert-torch
    decomposes PixelShuffle into 6D transposes that TFLite Micro can't handle;
    onnx2tf maps it to a single DEPTH_TO_SPACE op. Output is NHWC, float32 IO.
    """
    try:
        import onnx  # noqa: F401
    except ImportError:
        sys.exit("onnx2tf path requires: pip install onnx onnxruntime onnx2tf")

    import tempfile, shutil, subprocess, numpy as np

    tmp = Path(tempfile.mkdtemp())
    onnx_path = tmp / f"{arch}.onnx"
    try:
        torch.onnx.export(model, sample_input, str(onnx_path), opset_version=13,
                          input_names=["input"], output_names=["output"])
    except Exception as e:
        raise RuntimeError(f"torch.onnx.export failed for {arch}: {e}") from e
    if not onnx_path.exists() or onnx_path.stat().st_size == 0:
        raise RuntimeError(f"ONNX export produced empty file for {arch}")

    def cleanup():
        shutil.rmtree(tmp, ignore_errors=True)

    calib_path = None
    if do_int8 and Path(val_dir).exists():
        print(f"  Calibrating on {val_dir} ...")
        batches = list(_sample_inputs_from_val(val_dir, arch, tile_h, tile_w, scale))
        if not batches:
            print(f"  [WARN] no calibration images found in {val_dir} — producing float32 model only")
        else:
            calib = np.concatenate(batches, axis=0)  # [N, 1, H, W] — ONNX NCHW format
            print(f"  Calibrated with {len(batches)} samples.")
            calib_path = tmp / "calib.npy"
            np.save(str(calib_path), calib)

    cmd = [sys.executable, "-m", "onnx2tf", "-i", str(onnx_path), "-o", str(tmp)]
    if do_int8 and calib_path:
        cmd += ["-oiqt", "-qt", "per-channel",
                "-cind", "input", str(calib_path), "[[0.0]]", "[[1.0]]"]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  onnx2tf failed:\n{r.stderr[-800:]}")
        return
    float32_produced = False
    int8_produced = False
    err = None
    try:
        print("\n[float32]")
        src = tmp / f"{arch}_float32.tflite"
        if not src.exists():
            # onnx2tf 3.x renames: look for alternative
            candidates = list(tmp.glob("*_float32*.tflite")) + list(tmp.glob("*.tflite"))
            src = candidates[0] if candidates else None
        if src and src.exists() and src.stat().st_size > 0:
            dst = out_dir / f"{arch}_float32.tflite"
            shutil.copy2(str(src), str(dst))
            print(f"  float32 TFLite → {dst.name}  ({dst.stat().st_size / 1024:.1f} KB)")
            float32_produced = True
            if not no_c_array:
                generate_c_array(dst, out_dir / f"{arch}_float32_data.h",
                                 f"{arch}_float32_tflite")
        else:
            raise RuntimeError(
                f"onnx2tf did not produce a valid float32 TFLite model.\n"
                f"  Expected: {src}\n"
                f"  Contents of {tmp}: {list(tmp.iterdir()) if tmp.exists() else 'missing'}\n"
                f"  Check onnx2tf logs above for errors."
            )

        if do_int8:
            print("\n[int8]")
            # full_integer_quant: all tensors INT8 (required by espressif TFLite Micro —
            # it rejects hybrid models where weights are INT8 but activations are float32)
            src_int = tmp / f"{arch}_full_integer_quant.tflite"
            if not src_int.exists():
                candidates = [f for f in list(tmp.glob("*.tflite")) if str(f) != str(src)]
                src_int = candidates[0] if candidates else None
            if src_int and src_int.exists() and src_int.stat().st_size > 0:
                dst = out_dir / f"{arch}_int8.tflite"
                shutil.copy2(str(src_int), str(dst))
                print(f"  INT8 TFLite    → {dst.name}  ({dst.stat().st_size / 1024:.1f} KB)")
                int8_produced = True
                if not no_c_array:
                    generate_c_array(dst, out_dir / f"{arch}_int8_data.h",
                                     f"{arch}_int8_tflite")
            else:
                raise RuntimeError(
                    f"onnx2tf did not produce an INT8 TFLite model.\n"
                    f"  Expected: {src_int}\n"
                    f"  Contents of {tmp}: {list(tmp.iterdir())}\n"
                    f"  Check onnx2tf logs above (calibration data must have >=1 images)."
                )
    except Exception as e:
        err = e
    finally:
        cleanup()
    if err:
        raise err
    if not float32_produced or (do_int8 and not int8_produced):
        sys.exit("FATAL: onnx2tf produced corrupt or empty TFLite models. "
                 "Check onnx2tf logs above.")


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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_h, tile_w = args.tile
    model, scale, num_channels = load_model(args.arch, args.pth_path)
    in_h, in_w = _input_shape(args.arch, tile_h, tile_w, scale)

    print(f"\nArch:       {args.arch}")
    print(f"Checkpoint: {args.pth_path}")
    print(f"Channels:   {num_channels}")
    print(f"Scale:      {scale}")
    print(f"Tile (LR):  {tile_h}×{tile_w}  →  model input: {in_h}×{in_w}")
    if args.arch == "srcnn":
        print("  Note: SRCNN bicubic upsample stripped — host must pre-upscale input")

    sample_input = torch.zeros(1, num_channels, in_h, in_w)

    if args.arch == "espcn_light":
        # onnx2tf path: handles PixelShuffle → DEPTH_TO_SPACE correctly
        export_via_onnx2tf(model, sample_input, args.arch, out_dir,
                           args.int8, args.val_dir, tile_h, tile_w, args.no_c_array,
                           scale)
    elif args.arch == "edsr_tiny":
        # Keras path: onnx2tf int8 produces hybrid models (espressif rejects them)
        export_edsr_tiny_via_keras(model, args.arch, out_dir,
                                   args.int8, args.val_dir, tile_h, tile_w, args.no_c_array,
                                   scale, num_channels)
    elif args.arch == "srcnn":
        # Keras path: litert-torch produces fake-quant / wrong ops for SRCNN;
        # Keras→TF converter yields properly fused int8 CONV_2D with int32 biases.
        export_srcnn_via_keras(model, args.arch, out_dir, args.int8,
                               args.val_dir, tile_h, tile_w, args.no_c_array,
                               scale, num_channels)
    else:
        try:
            import litert_torch  # noqa: F401
        except ImportError:
            sys.exit(
                "litert_torch not installed.\n"
                "  pip install litert-torch\n"
                "See: https://github.com/google-ai-edge/litert-torch"
            )

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
                            tile_h, tile_w, int8_out, scale)
                if not args.no_c_array and int8_out.exists():
                    var = f"{args.arch}_int8_tflite"
                    generate_c_array(int8_out, out_dir / f"{args.arch}_int8_data.h", var)

    print("\nDone.")


if __name__ == "__main__":
    main()
