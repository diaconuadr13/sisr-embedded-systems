#!/usr/bin/env bash
# Turnkey: export INT8 -> flash ESP32 -> capture serial log, for one model.
# Usage (repo root, on your Mac):
#     VAL=data/val/DIV2K_valid_HR bash cas2026_paper/deploy_kit/run_model_esp32.sh espcn_light
# Requires: arduino-cli (+ esp32 core & TFLM lib already set up), your python env with torch/TF/onnx2tf.
set -euo pipefail

ARCH="${1:?usage: run_model_esp32.sh <espcn_light|espcn|fsrcnn>}"
PYTHON="${PYTHON:-.disertatie/bin/python}"
VAL="${VAL:-data/val/DIV2K_valid_HR}"
FQBN="${FQBN:-esp32:esp32:esp32}"          # override if your board differs

declare -A CKPT=(
 [espcn_light]="runs/ESPCN_Light_thermal_gray_x2/InfraredThermal32x24/exp_20260510_194053/best_model.pth"
 [espcn]="runs/ESPCN_thermal_gray_x2/InfraredThermal32x24/exp_20260510_200719/best_model.pth"
 [fsrcnn]="runs/FSRCNN_thermal_gray_x2/InfraredThermal32x24/exp_20260510_200920/best_model.pth"
)
BASE="board_experiments/${ARCH}_thermal_32x24_to_64x48_x2"
SKETCH="${BASE}/esp32_${ARCH}"
LOGDIR="board_results/esp32_dev_module/${ARCH}_32x24_to_64x48"
LOG="${LOGDIR}/board_log.txt"
mkdir -p "$LOGDIR"

echo "== [1/4] export INT8 (${ARCH}) =="
$PYTHON tools/export_tflite.py "$ARCH" "${CKPT[$ARCH]}" "${BASE}/model" \
        --int8 --tile 24 32 --val_dir "$VAL"
cp "${BASE}/model/${ARCH}_int8_data.h" "${SKETCH}/"
ls -l "${BASE}/model/${ARCH}_int8.tflite"

echo "== [2/4] detect ESP32 port =="
arduino-cli board list || true
PORT="${PORT:-$(arduino-cli board list | grep -iE 'usbserial|SLAB|wchusb|cp210|/dev/cu\.usb|/dev/ttyUSB|/dev/ttyACM' | awk '{print $1}' | head -1)}"
echo "Using PORT=$PORT  FQBN=$FQBN"
[ -n "$PORT" ] || { echo "No port auto-detected; re-run with PORT=/dev/cu.xxx"; exit 1; }

echo "== [3/4] compile + upload =="
arduino-cli compile --fqbn "$FQBN" "$SKETCH"
arduino-cli upload -p "$PORT" --fqbn "$FQBN" "$SKETCH"

echo "== [4/4] capture serial (~30 s; opening the port resets the ESP32) =="
( timeout 30 arduino-cli monitor -p "$PORT" -c baudrate=115200 || true ) | tee "$LOG"

echo ""
echo "================ KEY LINES (${ARCH}, ESP32) ================"
grep -iE 'internal (heap|block)|arena (allocated|used)|Inference time|PSNR|AllocateTensors|didn.t find|Failed' "$LOG" || \
  echo "(no key lines captured — if empty, press the ESP32 reset button and re-run step 4)"
echo "Full log: $LOG"
