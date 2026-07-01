#!/usr/bin/env bash
# Flash all three new models to the ESP32 back-to-back and write logs to board_results/.
# Run on the Mac (or hand to Codex):
#   cd ~/Desktop/sisr-embedded-systems && VAL=data/val/DIV2K_valid_HR bash cas2026_paper/deploy_kit/run_all_esp32.sh
set -uo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"   # -> repo root
export VAL="${VAL:-data/val/DIV2K_valid_HR}"
export PYTHON="${PYTHON:-.disertatie/bin/python}"
echo "repo: $(pwd)  |  VAL=$VAL  PYTHON=$PYTHON  FQBN=${FQBN:-esp32:esp32:esp32}"
for a in espcn_light espcn fsrcnn; do
  echo ""; echo "################## $a ##################"
  bash cas2026_paper/deploy_kit/run_model_esp32.sh "$a" || echo "[$a] FAILED — continuing to next"
done
echo ""
echo "DONE. Logs written to:"
ls -1 board_results/esp32_dev_module/*/board_log.txt 2>/dev/null || echo "  (none — check errors above)"
