#!/usr/bin/env python3
"""Export a trained .pth checkpoint to .sisr binary weight format.

Usage:
    python tools/export_weights.py <arch> <pth_path> <out_path>
    arch: espcn_light | carn_m
"""
import sys, struct
import numpy as np
import torch

sys.path.insert(0, ".")  # allow importing models/

MAGIC   = b"SISR"
VERSION = 1
CONV2D  = 0
PIXSHUFFLE = 2


def write_layer_conv(f, weight, bias, groups=1):
    out_c, in_c_per_group, kH, kW = weight.shape
    in_c = in_c_per_group * groups
    has_bias = 1 if bias is not None else 0
    f.write(struct.pack("<B", CONV2D))
    f.write(struct.pack("<H", out_c))
    f.write(struct.pack("<H", in_c))
    f.write(struct.pack("<B", kH))
    f.write(struct.pack("<B", kW))
    f.write(struct.pack("<B", groups))
    f.write(struct.pack("<B", has_bias))
    f.write(weight.detach().numpy().astype(np.float32).tobytes())
    if has_bias:
        f.write(bias.detach().numpy().astype(np.float32).tobytes())


def write_layer_pixshuffle(f, scale):
    f.write(struct.pack("<B", PIXSHUFFLE))
    f.write(struct.pack("<H", scale))   # out_channels repurposed = scale
    f.write(struct.pack("<H", 0))       # in_channels = 0
    f.write(struct.pack("<B", 0))       # kH = 0
    f.write(struct.pack("<B", 0))       # kW = 0
    f.write(struct.pack("<B", 1))       # groups = 1 (avoid div-by-zero in parser)
    f.write(struct.pack("<B", 0))       # has_bias = 0


def collect_espcn_light(model):
    """Returns list of (weight, bias, groups) tuples or ('pixel_shuffle', scale)."""
    layers = []
    fe = model.feature_extractor
    # Indices 0, 2, 4 are Conv2d; indices 1, 3 are Tanh (no weights)
    for idx in [0, 2, 4]:
        conv = fe[idx]
        layers.append((conv.weight.detach(), conv.bias.detach() if conv.bias is not None else None, 1))
    layers.append(('pixel_shuffle', 2))
    return layers


def collect_carn_m(model):
    """Returns ordered list matching what carn_m.c expects to read sequentially."""
    layers = []
    # head conv
    layers.append((model.head.weight.detach(), model.head.bias.detach(), 1))
    # 3 cascade units: each has shared_block.conv1, shared_block.conv2, fuse[0..2]
    for unit in model.units:
        sb = unit.shared_block
        layers.append((sb.conv1.weight.detach(), sb.conv1.bias.detach(), sb.conv1.groups))
        layers.append((sb.conv2.weight.detach(), sb.conv2.bias.detach(), sb.conv2.groups))
        for fuse_conv in unit.fuse:
            layers.append((fuse_conv.weight.detach(), fuse_conv.bias.detach(), 1))
    # global unit fuse convs
    for uf in model.unit_fuse:
        layers.append((uf.weight.detach(), uf.bias.detach(), 1))
    # upsample conv
    up_conv = model.upsample[0]
    layers.append((up_conv.weight.detach(), up_conv.bias.detach(), 1))
    # pixel shuffle
    layers.append(('pixel_shuffle', 2))
    return layers


def export(arch, pth_path, out_path):
    if arch == "espcn_light":
        from models.espcn_light import ESPCNLight
        model = ESPCNLight(scale_factor=2, num_channels=1)
        collect_fn = collect_espcn_light
    elif arch == "carn_m":
        from models.carn_m import CARNM
        model = CARNM(scale_factor=2, num_channels=1)
        collect_fn = collect_carn_m
    else:
        sys.exit(f"Unknown arch: {arch}")

    state = torch.load(pth_path, map_location="cpu")
    # Handle both raw state_dict and wrapped checkpoints
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    layers = collect_fn(model)

    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<B", VERSION))
        f.write(struct.pack("<H", len(layers)))
        for layer in layers:
            if layer[0] == 'pixel_shuffle':
                write_layer_pixshuffle(f, layer[1])
            else:
                weight, bias, groups = layer
                write_layer_conv(f, weight, bias, groups)

    print(f"Exported {len(layers)} layers to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: export_weights.py <arch> <pth_path> <out_path>")
    export(sys.argv[1], sys.argv[2], sys.argv[3])
