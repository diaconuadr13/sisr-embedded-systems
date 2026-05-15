# ESPCN_Micro thermal 32x24 x2 ESP32 test

This is the fallback board experiment for ESP32 boards without PSRAM. The
EDSR_Tiny 32x24 test needs a tensor arena larger than the board's largest
contiguous internal heap block, so this folder uses the smaller ESPCN_Micro
thermal checkpoint for the same input/output size.

Source checkpoint:

```bash
runs/ESPCN_Micro_thermal_gray_x2/InfraredThermal32x24/exp_20260510_200842/best_model.pth
```

Generated model shape:

```text
input:  1x24x32x1 int8
output: 1x48x64x1 int8
```

This folder uses `export_full_integer_espcn_micro.py` to avoid the hybrid
Conv2D graph produced by the generic ONNX export path. The ESP32 TFLite Micro
runtime rejects hybrid Conv2D models.

Arduino sketch:

```text
esp32_espcn_micro_int8_32x24/esp32_espcn_micro_int8_32x24.ino
```

The sketch registers the ops found in the exported full-integer model:

```text
CONV_2D, DEPTH_TO_SPACE
```

`test_image.h` was generated from `/scratch/users/adiaconu/thermal_21.png`,
which is already a 32x24 LR image. The HR reference in the header is bicubic
64x48, so PSNR is only a sanity comparison against bicubic.

Regenerate:

```bash
cd /scratch/users/adiaconu/sisr-embedded-systems

.disertatie/bin/python board_experiments/espcn_micro_thermal_32x24_x2/export_full_integer_espcn_micro.py \
  runs/ESPCN_Micro_thermal_gray_x2/InfraredThermal32x24/exp_20260510_200842/best_model.pth \
  board_experiments/espcn_micro_thermal_32x24_x2/esp32_espcn_micro_int8_32x24 \
  --tile 24 32

.disertatie/bin/python board_experiments/edsr_tiny_thermal_32x24_x2/make_test_image_from_lr.py \
  /scratch/users/adiaconu/thermal_21.png \
  board_experiments/espcn_micro_thermal_32x24_x2/esp32_espcn_micro_int8_32x24/test_image.h \
  --size 24 32 --scale 2
```
