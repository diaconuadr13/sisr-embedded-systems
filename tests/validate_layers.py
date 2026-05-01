#!/usr/bin/env python3
"""Validates C primitives against PyTorch reference outputs."""
import subprocess, struct, sys, os, tempfile
import numpy as np
import torch
import torch.nn.functional as F

BINARY = "./sisr_test"

def write_f32(path, arr):
    np.array(arr, dtype=np.float32).tofile(path)

def read_f32(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)

def run_c(*args):
    result = subprocess.run([BINARY] + [str(a) for a in args],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"C binary failed:\n{result.stderr}")

def test_conv2d():
    in_c, in_h, in_w = 3, 8, 8
    out_c, kH, kW, padding, groups = 16, 3, 3, 1, 1

    x = torch.randn(1, in_c, in_h, in_w)
    w = torch.randn(out_c, in_c // groups, kH, kW)
    b = torch.randn(out_c)
    ref = F.conv2d(x, w, b, padding=padding, groups=groups).squeeze(0)

    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, "input.bin")
        wgt = os.path.join(d, "weight.bin")
        bia = os.path.join(d, "bias.bin")
        out = os.path.join(d, "output.bin")
        write_f32(inp, x.squeeze(0).numpy())
        write_f32(wgt, w.numpy())
        write_f32(bia, b.numpy())
        run_c("conv2d", in_c, in_h, in_w, out_c, kH, kW,
              padding, groups, 1, inp, wgt, bia, out)
        c_out = read_f32(out, (out_c, in_h, in_w))

    diff = np.abs(c_out - ref.numpy()).max()
    assert diff < 1e-4, f"conv2d max diff {diff:.2e} exceeds threshold"
    print(f"  PASS conv2d (max diff {diff:.2e})")

    # Also test no-bias case
    ref_nobias = F.conv2d(x, w, None, padding=padding, groups=groups).squeeze(0)
    with tempfile.TemporaryDirectory() as d:
        inp2 = os.path.join(d, "input.bin")
        wgt2 = os.path.join(d, "weight.bin")
        out2 = os.path.join(d, "output.bin")
        write_f32(inp2, x.squeeze(0).numpy())
        write_f32(wgt2, w.numpy())
        run_c("conv2d", in_c, in_h, in_w, out_c, kH, kW,
              padding, groups, 0, inp2, wgt2, out2)
        c_out2 = read_f32(out2, (out_c, in_h, in_w))
    diff2 = np.abs(c_out2 - ref_nobias.numpy()).max()
    assert diff2 < 1e-4, f"conv2d no-bias max diff {diff2:.2e}"
    print(f"  PASS conv2d no-bias (max diff {diff2:.2e})")

def test_grouped_conv2d():
    in_c, in_h, in_w = 64, 8, 8
    out_c, kH, kW, padding, groups = 64, 3, 3, 1, 4

    x = torch.randn(1, in_c, in_h, in_w)
    w = torch.randn(out_c, in_c // groups, kH, kW)
    b = torch.randn(out_c)
    ref = F.conv2d(x, w, b, padding=padding, groups=groups).squeeze(0)

    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, "input.bin")
        wgt = os.path.join(d, "weight.bin")
        bia = os.path.join(d, "bias.bin")
        out = os.path.join(d, "output.bin")
        write_f32(inp, x.squeeze(0).numpy())
        write_f32(wgt, w.numpy())
        write_f32(bia, b.numpy())
        run_c("conv2d", in_c, in_h, in_w, out_c, kH, kW,
              padding, groups, 1, inp, wgt, bia, out)
        c_out = read_f32(out, (out_c, in_h, in_w))

    diff = np.abs(c_out - ref.numpy()).max()
    assert diff < 1e-4, f"grouped_conv2d max diff {diff:.2e} exceeds threshold"
    print(f"  PASS grouped_conv2d (max diff {diff:.2e})")

def test_conv2d_nonsquare():
    in_c, in_h, in_w = 4, 6, 10
    out_c, kH, kW, padding, groups = 8, 3, 3, 1, 1
    x = torch.randn(1, in_c, in_h, in_w)
    w = torch.randn(out_c, in_c // groups, kH, kW)
    b = torch.randn(out_c)
    ref = F.conv2d(x, w, b, padding=padding, groups=groups).squeeze(0)
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, "input.bin")
        wgt = os.path.join(d, "weight.bin")
        bia = os.path.join(d, "bias.bin")
        out = os.path.join(d, "output.bin")
        write_f32(inp, x.squeeze(0).numpy())
        write_f32(wgt, w.numpy())
        write_f32(bia, b.numpy())
        run_c("conv2d", in_c, in_h, in_w, out_c, kH, kW,
              padding, groups, 1, inp, wgt, bia, out)
        c_out = read_f32(out, (out_c, in_h, in_w))
    diff = np.abs(c_out - ref.numpy()).max()
    assert diff < 1e-4, f"conv2d nonsquare max diff {diff:.2e}"
    print(f"  PASS conv2d nonsquare (max diff {diff:.2e})")

def test_pixel_shuffle():
    channels, h, w, scale = 4, 8, 8, 2
    # PyTorch PixelShuffle expects [1, C*r*r, H, W]
    x = torch.randn(1, channels * scale * scale, h, w)
    ref = torch.nn.PixelShuffle(scale)(x).squeeze(0)

    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, "input.bin")
        out = os.path.join(d, "output.bin")
        write_f32(inp, x.squeeze(0).numpy())
        run_c("pixel_shuffle", channels * scale * scale, h, w, scale, inp, out)
        c_out = read_f32(out, (channels, h * scale, w * scale))

    diff = np.abs(c_out - ref.numpy()).max()
    assert diff < 1e-5, f"pixel_shuffle max diff {diff:.2e}"
    print(f"  PASS pixel_shuffle (max diff {diff:.2e})")

def test_pixel_shuffle_nonsquare():
    channels, h, w, scale = 4, 6, 10, 2
    x = torch.randn(1, channels * scale * scale, h, w)
    ref = torch.nn.PixelShuffle(scale)(x).squeeze(0)
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, "input.bin")
        out = os.path.join(d, "output.bin")
        write_f32(inp, x.squeeze(0).numpy())
        run_c("pixel_shuffle", channels * scale * scale, h, w, scale, inp, out)
        c_out = read_f32(out, (channels, h * scale, w * scale))
    diff = np.abs(c_out - ref.numpy()).max()
    assert diff < 1e-5, f"pixel_shuffle nonsquare max diff {diff:.2e}"
    print(f"  PASS pixel_shuffle nonsquare (max diff {diff:.2e})")

def test_tanh():
    x = np.random.randn(64).astype(np.float32)
    ref = np.tanh(x)
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, "input.bin"); out = os.path.join(d, "output.bin")
        write_f32(inp, x)
        run_c("tanh", len(x), inp, out)
        c_out = read_f32(out, (len(x),))
    diff = np.abs(c_out - ref).max()
    assert diff < 1e-6, f"tanh max diff {diff:.2e}"
    print(f"  PASS tanh (max diff {diff:.2e})")

def test_leaky_relu():
    x = np.random.randn(64).astype(np.float32)
    neg_slope = 0.1
    ref = np.where(x >= 0, x, neg_slope * x)
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, "input.bin"); out = os.path.join(d, "output.bin")
        write_f32(inp, x)
        run_c("leaky_relu", len(x), neg_slope, inp, out)
        c_out = read_f32(out, (len(x),))
    diff = np.abs(c_out - ref).max()
    assert diff < 1e-6, f"leaky_relu max diff {diff:.2e}"
    print(f"  PASS leaky_relu (max diff {diff:.2e})")

def test_weights_parser():
    """Export untrained ESPCNLight, check .sisr header via Python (no C needed here)."""
    import struct
    sys.path.insert(0, ".")
    import torch
    from models.espcn_light import ESPCNLight
    with tempfile.TemporaryDirectory() as d:
        pth  = os.path.join(d, "model.pth")
        sisr = os.path.join(d, "model.sisr")
        m = ESPCNLight(scale_factor=2, num_channels=1)
        torch.save(m.state_dict(), pth)
        ret = subprocess.run(
            ["python3", "tools/export_weights.py", "espcn_light", pth, sisr],
            capture_output=True, text=True)
        assert ret.returncode == 0, ret.stderr
        with open(sisr, "rb") as f:
            magic = f.read(4)
            assert magic == b"SISR", f"Bad magic {magic}"
            f.read(1)  # version
            num_layers = struct.unpack("<H", f.read(2))[0]
        assert num_layers == 4, f"Expected 4 layers, got {num_layers}"
    print(f"  PASS weights_parser (4 layers in ESPCNLight .sisr)")


def test_espcn_light():
    sys.path.insert(0, ".")
    import torch
    from models.espcn_light import ESPCNLight

    tile_h, tile_w = 16, 16
    model = ESPCNLight(scale_factor=2, num_channels=1)
    model.eval()

    x = torch.randn(1, 1, tile_h, tile_w)
    with torch.no_grad():
        ref = model(x).squeeze(0).numpy()  # [1, 32, 32]

    with tempfile.TemporaryDirectory() as d:
        pth  = os.path.join(d, "model.pth")
        sisr = os.path.join(d, "model.sisr")
        inp  = os.path.join(d, "input.bin")
        out  = os.path.join(d, "output.bin")

        torch.save(model.state_dict(), pth)
        subprocess.run(["python3", "tools/export_weights.py",
                        "espcn_light", pth, sisr], check=True, capture_output=True)

        write_f32(inp, x.squeeze(0).numpy())  # [1, H, W]
        run_c("espcn_light", tile_w, tile_h, inp, sisr, out)
        c_out = read_f32(out, (1, tile_h * 2, tile_w * 2))

    diff = np.abs(c_out - ref).max()
    assert diff < 1e-3, f"espcn_light max diff {diff:.2e}"
    print(f"  PASS espcn_light full model (max diff {diff:.2e})")


def test_carn_m():
    sys.path.insert(0, ".")
    import torch
    from models.carn_m import CARNM

    tile_h, tile_w = 8, 8
    model = CARNM(scale_factor=2, num_channels=1)
    model.eval()

    x = torch.randn(1, 1, tile_h, tile_w)
    with torch.no_grad():
        ref = model(x).squeeze(0).numpy()

    with tempfile.TemporaryDirectory() as d:
        pth  = os.path.join(d, "model.pth")
        sisr = os.path.join(d, "model.sisr")
        inp  = os.path.join(d, "input.bin")
        out  = os.path.join(d, "output.bin")

        torch.save(model.state_dict(), pth)
        subprocess.run(["python3", "tools/export_weights.py",
                        "carn_m", pth, sisr], check=True, capture_output=True)

        write_f32(inp, x.squeeze(0).numpy())
        run_c("carn_m", tile_w, tile_h, inp, sisr, out)
        c_out = read_f32(out, (1, tile_h * 2, tile_w * 2))

    diff = np.abs(c_out - ref).max()
    assert diff < 1e-3, f"carn_m max diff {diff:.2e}"
    print(f"  PASS carn_m full model (max diff {diff:.2e})")


if __name__ == "__main__":
    print("Running conv2d tests...")
    test_conv2d()
    test_grouped_conv2d()
    test_conv2d_nonsquare()
    print("Running pixel_shuffle tests...")
    test_pixel_shuffle()
    test_pixel_shuffle_nonsquare()
    print("Running activation tests...")
    test_tanh()
    test_leaky_relu()
    print("Running weight parser tests...")
    test_weights_parser()
    print("Running model tests...")
    test_espcn_light()
    test_carn_m()
    print("All tests passed.")
