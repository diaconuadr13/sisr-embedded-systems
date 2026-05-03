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
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "edsr_tiny_int8_data.h"
#include "test_image.h"

// INT8 activations are 4× smaller than float32; 48 KB is comfortable for
// 8-resblock EDSR_Tiny at 8×8 tile. Increase to 64 KB if needed.
constexpr int kTensorArenaSize = 48 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

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
        model, resolver, tensor_arena, kTensorArenaSize);

    TfLiteStatus alloc_status = interpreter.AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors() failed — tensor arena too small");
        Serial.printf("  Arena used: %d bytes\n", interpreter.arena_used_bytes());
        return;
    }
    Serial.printf("Arena used: %d / %d bytes\n",
                  interpreter.arena_used_bytes(), kTensorArenaSize);

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
    Serial.println("Done.");
}

void loop() {}
