# Raspberry Pi Pico firmware

This folder contains Pico/RP2040 Arduino sketches for models that already have
ESP32 firmware in this project:

- `edsr_tiny_int8/` - EDSR_Tiny INT8, float32 input/output
- `srcnn_inference_int8/` - SRCNN INT8, float32 input/output

Start with `srcnn_inference_int8` if you want the smallest first test. Use
`edsr_tiny_int8` when you specifically want to compare the EDSR_Tiny deployment.

## Raspberry Pi Pico setup

The original Raspberry Pi Pico board marked `C 2020` uses the RP2040
microcontroller. ArduTFLite does not list RP2040/Pico support, so these sketches
expect a Pico-compatible TensorFlow Lite Micro runtime, such as Raspberry Pi's
`pico-tflmicro` port, that exposes the normal TFLite Micro headers:

```cpp
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
```

If Arduino IDE reports `tensorflow/lite/micro/micro_interpreter.h: No such file
or directory`, the selected board/runtime does not provide TFLite Micro for
RP2040. Installing ArduTFLite will not fix that for an original Pico. Use a
Pico SDK / `pico-tflmicro` build, or install an Arduino RP2040 TFLite Micro port
that provides those exact headers.

## Generate headers

Each sketch expects the generated model header and `test_image.h` in its own
sketch folder. The `.tflite` file is not included by the sketch directly, but
it is kept beside the generated header so the Pico folder mirrors the ESP32
deployment artifact set. You can either regenerate these files or copy the
matching generated artifacts from `esp32_firmware/`.

EDSR_Tiny INT8:

```bash
python tools/export_tflite.py edsr_tiny \
  runs/EDSR_Tiny_gray_x2/Flickr2K/exp_<ts>/best_model.pth \
  pico_firmware/edsr_tiny_int8/ --tile 8 8 --int8
python tools/image_to_c_array.py data/val/DIV2K_valid_HR/0801.png \
  pico_firmware/edsr_tiny_int8/test_image.h --size 8 8 --arch edsr_tiny
```

SRCNN INT8:

```bash
python tools/export_tflite.py srcnn \
  runs/SRCNN_gray_x2/Flickr2K/exp_<ts>/best_model.pth \
  pico_firmware/srcnn_inference_int8/ --tile 8 8 --int8
python tools/image_to_c_array.py data/val/DIV2K_valid_HR/0801.png \
  pico_firmware/srcnn_inference_int8/test_image.h --size 8 8 --arch srcnn
```

If you just want to use the currently generated ESP32 artifacts:

```bash
cp esp32_firmware/edsr_tiny_int8/edsr_tiny_int8.tflite pico_firmware/edsr_tiny_int8/
cp esp32_firmware/edsr_tiny_int8/edsr_tiny_int8_data.h pico_firmware/edsr_tiny_int8/
cp esp32_firmware/edsr_tiny_int8/test_image.h pico_firmware/edsr_tiny_int8/
cp esp32_firmware/srcnn_inference_int8/srcnn_int8.tflite pico_firmware/srcnn_inference_int8/
cp esp32_firmware/srcnn_inference_int8/srcnn_int8_data.h pico_firmware/srcnn_inference_int8/
cp esp32_firmware/srcnn_inference_int8/test_image.h pico_firmware/srcnn_inference_int8/
```

## Board log

The sketches print a block that can be saved and passed to
`tools/collect_deployment_metrics.py`:

```text
target=pico
tile=8x8
scale=2
tensor_arena_bytes=49152
free_sram_before=180224
free_sram_after=130560
inference_ms=42.500
sample_ms=42.000
sample_ms=43.000
```

Use that file with the normal deployment metrics command:

```bash
python tools/collect_deployment_metrics.py \
  --checkpoint runs/.../best_model.pth \
  --tile 8 8 \
  --board-log pico_benchmark.log \
  --output runs/.../deployment_metrics_pico_8x8.json
```
