# ESPCN_Micro thermal 32x24 to 64x48 pseudo-HR experiment

This experiment trains a model for board input `32x24` and output `64x48`.

The source dataset contains native `32x24` thermal frames, so the `64x48` target
is pseudo-HR: each native frame is resized to `64x48`, and the LR input is made
by downsampling that target back to `32x24`.

Training config:

```text
train_espcn_micro_thermal_64x48_pseudo_x2.yaml
```

Completed run:

```text
runs/ESPCN_Micro_thermal_32x24_to_64x48_x2/InfraredThermal64x48Pseudo/exp_20260515_182835
```

Best validation point in `training_log.csv`:

```text
epoch 99: val_psnr=45.9205, val_ssim=0.9969
```

Expected training pair:

```text
LR: 32x24
HR: 64x48
```

The ESP32 export folder is:

```text
esp32_espcn_micro_int8_32x24_to_64x48/
```

It contains:

```text
esp32_espcn_micro_int8_32x24_to_64x48.ino
espcn_micro_int8.tflite
espcn_micro_int8_data.h
test_image.h
```

Export command:

```bash
cd /scratch/users/adiaconu/sisr-embedded-systems

.disertatie/bin/python board_experiments/espcn_micro_thermal_32x24_to_64x48_x2/export_full_integer_espcn_micro.py \
  runs/ESPCN_Micro_thermal_32x24_to_64x48_x2/InfraredThermal64x48Pseudo/exp_20260515_182835/best_model.pth \
  board_experiments/espcn_micro_thermal_32x24_to_64x48_x2/esp32_espcn_micro_int8_32x24_to_64x48 \
  --tile 24 32

.disertatie/bin/python board_experiments/espcn_micro_thermal_32x24_to_64x48_x2/make_test_image_from_lr.py \
  /scratch/users/adiaconu/thermal_21.png \
  board_experiments/espcn_micro_thermal_32x24_to_64x48_x2/esp32_espcn_micro_int8_32x24_to_64x48/test_image.h \
  --size 24 32 --scale 2
```
