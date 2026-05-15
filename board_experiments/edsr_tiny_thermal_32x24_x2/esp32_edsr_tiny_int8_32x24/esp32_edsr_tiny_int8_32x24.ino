/*
 * SISR TFLite Micro inference test for ESP32 — EDSR_Tiny int8
 *
 * Model exported via onnx2tf with full INT8 PTQ. All weights and activations
 * are INT8; IO is float32 (QUANTIZE/DEQUANTIZE fused at boundaries).
 * No manual quantization needed — input/output are float32.
 *
 * Wire: just USB cable to PC. Open Serial Monitor at 115200 baud.
 *
 * Generate headers (run on server):
 *   python tools/export_tflite.py edsr_tiny \
 *       runs/EDSR_Tiny_gray_x2/Flickr2K/exp_<ts>/best_model.pth \
 *       esp32_firmware/edsr_tiny_int8/ --tile 8 8 --int8
 *   python tools/image_to_c_array.py data/val/DIV2K_valid_HR/0801.png \
 *       esp32_firmware/edsr_tiny_int8/test_image.h --size 8 8 --arch edsr_tiny
 *
 * Ops registered below match the onnx2tf full-integer graph for EDSR_Tiny.
 * ReLU is fused into Conv2D as an activation by the quantizer — no AddRelu().
 * If AllocateTensors fails with "operation not found", inspect the model in
 * Netron and add the missing op. Increase kTensorArenaSize if arena too small.
 */

#include <Arduino.h>
#include <esp_heap_caps.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "edsr_tiny_int8_data.h"
#include "test_image.h"

// Larger 32x24 LR tile needs more activation memory than the original 8x8 test.
constexpr int kMaxTensorArenaSize = 128 * 1024;
static uint8_t* tensor_arena = nullptr;
static size_t tensor_arena_size = 0;

static uint8_t* allocate_tensor_arena(size_t* allocated_size) {
    const size_t candidate_sizes[] = {
        128 * 1024, 120 * 1024, 112 * 1024, 104 * 1024,
        96 * 1024, 88 * 1024, 80 * 1024, 72 * 1024, 64 * 1024,
    };

    Serial.printf("Free internal heap: %u bytes\n",
                  heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    Serial.printf("Largest internal block: %u bytes\n",
                  heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    Serial.printf("Free PSRAM heap: %u bytes\n",
                  heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    Serial.printf("Largest PSRAM block: %u bytes\n",
                  heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

    for (size_t size : candidate_sizes) {
        uint8_t* arena = static_cast<uint8_t*>(
            heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        if (arena != nullptr) {
            *allocated_size = size;
            Serial.printf("Tensor arena allocated in PSRAM: %u bytes\n",
                          static_cast<unsigned>(size));
            return arena;
        }
    }

    for (size_t size : candidate_sizes) {
        uint8_t* arena = static_cast<uint8_t*>(
            heap_caps_malloc(size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
        if (arena != nullptr) {
            *allocated_size = size;
            Serial.printf("Tensor arena allocated in internal heap: %u bytes\n",
                          static_cast<unsigned>(size));
            return arena;
        }
    }

    *allocated_size = 0;
    return nullptr;
}

static void dump_pixels_serial(const char* tag, int h, int w, const float* data) {
    Serial.print("SISR:");
    Serial.print(tag);
    Serial.print(":");
    Serial.print(h);
    Serial.print(":");
    Serial.print(w);
    Serial.print(":");
    int n = h * w;
    for (int i = 0; i < n; i++) {
        Serial.print(data[i], 6);
        if (i < n - 1) Serial.print(",");
    }
    Serial.println();
}

float compute_psnr(const float* a, const float* b, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = (double)a[i] - (double)b[i];
        mse += diff * diff;
    }
    mse /= n;
    if (mse == 0.0) return 100.0f;
    return (float)(10.0 * log10(1.0 / mse));
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== SISR TFLite Micro Test (EDSR_Tiny int8) ===");

    tensor_arena = allocate_tensor_arena(&tensor_arena_size);
    if (tensor_arena == nullptr) {
        Serial.println("ERROR: Unable to allocate tensor arena");
        Serial.printf("Largest requested: %d bytes\n", kMaxTensorArenaSize);
        return;
    }

    const tflite::Model* model = tflite::GetModel(edsr_tiny_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Model schema version mismatch");
        return;
    }

    // Full-integer quantized graph:
    // QUANTIZE → [Conv2D(+relu fused) → Conv2D → Add] × 8 → Conv2D → Add
    //          → Conv2D → DEPTH_TO_SPACE → DEQUANTIZE
    tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddQuantize();
    resolver.AddConv2D();
    resolver.AddAdd();
    resolver.AddDepthToSpace();
    resolver.AddDequantize();

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, tensor_arena_size);

    TfLiteStatus alloc_status = interpreter.AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors() failed — tensor arena too small");
        Serial.printf("  Arena used: %d bytes\n", interpreter.arena_used_bytes());
        return;
    }
    Serial.printf("Arena used: %d / %d bytes\n",
                  interpreter.arena_used_bytes(),
                  static_cast<int>(tensor_arena_size));

    TfLiteTensor* input = interpreter.input(0);
    Serial.printf("Input shape: [%d, %d, %d, %d]  type: %d\n",
                  input->dims->data[0], input->dims->data[1],
                  input->dims->data[2], input->dims->data[3],
                  input->type);

    // Input is float32 (QUANTIZE op handles float→int8 internally)
    memcpy(input->data.f, test_input_data, TEST_INPUT_LEN * sizeof(float));

    unsigned long t0 = micros();
    TfLiteStatus invoke_status = interpreter.Invoke();
    unsigned long elapsed_us = micros() - t0;

    if (invoke_status != kTfLiteOk) {
        Serial.println("ERROR: Invoke() failed");
        return;
    }

    // Output is float32 (DEQUANTIZE op handles int8→float internally)
    TfLiteTensor* output = interpreter.output(0);
    int out_len = output->bytes / sizeof(float);
    float* out_f32 = output->data.f;

    float psnr = -1.0f;
    if (out_len == TEST_HR_LEN && out_f32) {
        psnr = compute_psnr(out_f32, test_hr_data, out_len);
    } else {
        Serial.printf("WARN: output len %d != HR ref len %d\n", out_len, TEST_HR_LEN);
    }

    Serial.println("\n--- Results ---");
    Serial.printf("Inference time : %lu us  (%.2f ms)\n", elapsed_us, elapsed_us / 1000.0f);
    Serial.printf("Output pixels  : %d\n", out_len);
    if (psnr >= 0) Serial.printf("PSNR vs HR     : %.2f dB\n", psnr);
    Serial.println("\n--- Image dump ---");
    Serial.println("SISR_IMG_START");
    dump_pixels_serial("LR", TEST_INPUT_H, TEST_INPUT_W, test_input_data);
    if (out_len == TEST_HR_LEN && out_f32)
        dump_pixels_serial("SR", TEST_HR_H, TEST_HR_W, out_f32);
    dump_pixels_serial("HR", TEST_HR_H, TEST_HR_W, test_hr_data);
    Serial.println("SISR_IMG_END");
    Serial.println("Done.");
}

void loop() {}
