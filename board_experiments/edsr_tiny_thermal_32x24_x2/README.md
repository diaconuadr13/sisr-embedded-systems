# EDSR_Tiny thermal 32x24 x2 board test

This experiment tests a 32x24 grayscale thermal frame as LR input and produces a
64x48 SR output on the board.

Source checkpoint:

```bash
runs/EDSR_Tiny_thermal_gray_x2/InfraredThermal32x24/exp_20260510_194329/best_model.pth
```

Generated model shape:

```text
input:  1x24x32x1
output: 1x48x64x1
```

Folders:

```text
esp32_edsr_tiny_int8_32x24/
pico_edsr_tiny_int8_32x24/
```

Each folder contains a matching sketch, `edsr_tiny_int8.tflite`,
`edsr_tiny_int8_data.h`, float32 export artifacts, and `test_image.h`.

The ESP32 sketch allocates the 128 KB tensor arena at runtime instead of as a
global array. This avoids `.dram0.bss` linker overflow and uses PSRAM first when
the selected ESP32 board has it enabled.

`test_image.h` was generated from `/scratch/users/adiaconu/thermal_21.png`,
which is already a 32x24 LR image. The HR reference in the header is bicubic
64x48, so PSNR is only a sanity comparison against bicubic, not a real HR
ground truth score.

Regenerate:

```bash
cd /scratch/users/adiaconu/sisr-embedded-systems

.disertatie/bin/python tools/export_tflite.py edsr_tiny \
  runs/EDSR_Tiny_thermal_gray_x2/InfraredThermal32x24/exp_20260510_194329/best_model.pth \
  board_experiments/edsr_tiny_thermal_32x24_x2/esp32_edsr_tiny_int8_32x24 \
  --tile 24 32 --int8

.disertatie/bin/python board_experiments/edsr_tiny_thermal_32x24_x2/make_test_image_from_lr.py \
  /scratch/users/adiaconu/thermal_21.png \
  board_experiments/edsr_tiny_thermal_32x24_x2/esp32_edsr_tiny_int8_32x24/test_image.h \
  --size 24 32 --scale 2
```
