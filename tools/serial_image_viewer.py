#!/usr/bin/env python3
"""
Receive SISR image frames from ESP32/Pico over serial and display LR/SR/HR.

Protocol emitted by the board (after inference):
    SISR_IMG_START
    SISR:LR:<H>:<W>:<px0>,<px1>,...
    SISR:SR:<H>:<W>:<px0>,<px1>,...
    SISR:HR:<H>:<W>:<px0>,<px1>,...
    SISR_IMG_END

Usage:
    python tools/serial_image_viewer.py --port /dev/ttyUSB0
    python tools/serial_image_viewer.py --port COM3 --baud 115200 --save out.png
    python tools/serial_image_viewer.py --port /dev/ttyACM0 --continuous
"""

import argparse
import sys
import re
import time

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="SISR serial image viewer")
    p.add_argument("--port", required=True, help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    p.add_argument("--baud", type=int, default=115200, help="Baud rate (default 115200)")
    p.add_argument("--save", metavar="PATH", help="Save figure to PNG instead of showing interactively")
    p.add_argument("--timeout", type=int, default=60, help="Serial read timeout in seconds (default 60)")
    p.add_argument("--continuous", action="store_true", help="Keep reading and updating display after each frame")
    return p.parse_args()


def read_frame(ser):
    """Read one SISR_IMG_START...SISR_IMG_END frame. Returns dict {tag: np.array} or None on error."""
    frames = {}
    in_frame = False

    while True:
        try:
            raw = ser.readline()
        except OSError as exc:
            if exc.errno == 6:  # Device not configured — USB CDC not ready yet
                time.sleep(0.5)
                continue
            print(f"[viewer] Serial read error: {exc}", file=sys.stderr)
            return None
        except Exception as exc:
            print(f"[viewer] Serial read error: {exc}", file=sys.stderr)
            return None

        if not raw:
            print("[viewer] Timeout — no data received.", file=sys.stderr)
            return None

        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        if line == "SISR_IMG_START":
            in_frame = True
            frames = {}
            print("[viewer] Frame started", flush=True)
            continue

        if line == "SISR_IMG_END":
            if in_frame:
                print(f"[viewer] Frame complete. Got: {list(frames.keys())}", flush=True)
                return frames
            continue

        if in_frame and line.startswith("SISR:"):
            # SISR:TAG:H:W:px0,px1,...
            m = re.match(r"SISR:(LR|SR|HR):(\d+):(\d+):(.*)", line)
            if m:
                tag = m.group(1)
                h, w = int(m.group(2)), int(m.group(3))
                try:
                    pixels = np.array([float(x) for x in m.group(4).split(",")], dtype=np.float32)
                    if len(pixels) == h * w:
                        frames[tag] = pixels.reshape(h, w)
                        print(f"[viewer] {tag}  {h}x{w}", flush=True)
                    else:
                        print(f"[viewer] WARN: {tag} pixel count {len(pixels)} != {h*w}", flush=True)
                except ValueError as exc:
                    print(f"[viewer] WARN: parse error for {tag}: {exc}", flush=True)
            continue

        # Echo board logs that are not image data
        print(f"  [board] {line}")


def render_frame(frames, save_path=None):
    """Display LR/SR/HR side-by-side. LR is shown at the same axes size but pixelated."""
    order = [("LR", "LR Input"), ("SR", "SR Output"), ("HR", "HR Reference")]
    present = [(k, t) for k, t in order if k in frames]
    if not present:
        print("[viewer] No image data to display.", file=sys.stderr)
        return

    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    # Use HR (or SR) dims for display extent so all images map to the same spatial region.
    ref = frames.get("HR", frames.get("SR", list(frames.values())[0]))
    ref_h, ref_w = ref.shape

    for ax, (key, title) in zip(axes, present):
        img = frames[key]
        h, w = img.shape
        # extent keeps all images visually comparable at the same spatial scale
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0,
                  interpolation="nearest",
                  extent=[0, ref_w, ref_h, 0],
                  aspect="auto")
        ax.set_title(f"{title}\n({w}×{h})")
        ax.axis("off")

    fig.suptitle("SISR Board Inference — Serial Image Viewer", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viewer] Saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    args = parse_args()

    try:
        import serial
    except ImportError:
        print("ERROR: pyserial not installed. Run: pip install pyserial", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to {args.port} at {args.baud} baud...", flush=True)
    try:
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
    except serial.SerialException as exc:
        print(f"ERROR: Cannot open {args.port}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.continuous:
            frame_idx = 0
            while True:
                frames = read_frame(ser)
                if frames is None:
                    break
                save = args.save.replace(".", f"_{frame_idx}.") if args.save else None
                render_frame(frames, save_path=save)
                frame_idx += 1
        else:
            frames = read_frame(ser)
            if not frames:
                print("[viewer] No complete frame received.", file=sys.stderr)
                sys.exit(1)
            render_frame(frames, save_path=args.save)
    finally:
        ser.close()


if __name__ == "__main__":
    main()
