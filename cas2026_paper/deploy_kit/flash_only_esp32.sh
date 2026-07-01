#!/usr/bin/env bash
# Flash a PRE-EXPORTED INT8 model to the ESP32 (no export step) and capture the serial log.
# Use when *_int8_data.h was exported on another machine and copied into <BASE>/model/.
# Mirrors steps 2-4 of run_model_esp32.sh; portable serial timeout (no coreutils `timeout`).
#   PORT=/dev/cu.usbserial-0001 bash cas2026_paper/deploy_kit/flash_only_esp32.sh espcn_light
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"   # -> repo root

ARCH="${1:?usage: flash_only_esp32.sh <espcn_light|espcn|fsrcnn>}"
FQBN="${FQBN:-esp32:esp32:esp32}"
SECS="${SECS:-30}"
BASE="board_experiments/${ARCH}_thermal_32x24_to_64x48_x2"
SKETCH="${BASE}/esp32_${ARCH}"
HDR="${BASE}/model/${ARCH}_int8_data.h"
LOGDIR="board_results/esp32_dev_module/${ARCH}_32x24_to_64x48"
LOG="${LOGDIR}/board_log.txt"
mkdir -p "$LOGDIR"

# Bounded serial read without GNU coreutils `timeout` (perl fork + SIGALRM).
cap() { # cap <secs> <cmd...>
  perl -e '
    my $t = shift @ARGV;
    my $pid = fork();
    if ($pid == 0) { exec @ARGV; die "exec: $!"; }
    $SIG{ALRM} = sub { kill "INT", $pid; kill "TERM", $pid; };
    alarm $t;
    waitpid($pid, 0);
  ' "$@"
}

echo "== [1/4] stage INT8 header (${ARCH}) =="
[ -f "$HDR" ] || { echo "missing $HDR — export/copy it first"; exit 1; }
cp "$HDR" "${SKETCH}/"
ls -l "${SKETCH}/${ARCH}_int8_data.h" "${BASE}/model/${ARCH}_int8.tflite" 2>/dev/null || true

echo "== [2/4] detect ESP32 port =="
arduino-cli board list || true
PORT="${PORT:-$(arduino-cli board list | grep -iE 'usbserial|SLAB|wchusb|cp210|/dev/cu\.usb|/dev/ttyUSB|/dev/ttyACM' | awk '{print $1}' | head -1)}"
echo "Using PORT=$PORT  FQBN=$FQBN"
[ -n "$PORT" ] || { echo "No port auto-detected; re-run with PORT=/dev/cu.xxx"; exit 1; }

echo "== [3/4] compile + upload =="
arduino-cli compile --fqbn "$FQBN" "$SKETCH"
arduino-cli upload -p "$PORT" --fqbn "$FQBN" "$SKETCH"

echo "== [4/4] capture serial (~${SECS}s; opening the port resets the ESP32) =="
perl -e 'select(undef,undef,undef,3)'   # let the USB serial port re-enumerate after upload
( cap "$SECS" arduino-cli monitor -p "$PORT" -c baudrate=115200 || true ) | tee "$LOG"

echo ""
echo "================ KEY LINES (${ARCH}, ESP32) ================"
grep -iE 'internal (heap|block)|arena (allocated|used)|Inference time|PSNR|AllocateTensors|didn.?t find|Failed|opcode|Prelu|op for builtin' "$LOG" || \
  echo "(no key lines captured — press the ESP32 reset button and re-run step 4)"
echo "Full log: $LOG"
