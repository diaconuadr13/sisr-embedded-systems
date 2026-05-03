/*
 * SISR TFLite Micro inference test for ESP32 — ESPCN_Light float32
 *
 * Model exported via onnx2tf (PyTorch → ONNX → TFLite). Uses DEPTH_TO_SPACE
 * for pixel shuffle — compatible with espressif TFLite Micro.
 *
 * Wire: just USB cable to PC. Open Serial Monitor at 115200 baud.
 *
 * Generate headers (run on server):
 *   python tools/export_tflite.py espcn_light \
 *       runs/ESPCN_Light_gray_x2/Flickr2K/exp_20260426_175335/best_model.pth \
 *       esp32_firmware/sisr_inference/ --tile 8 8
 *   python tools/image_to_c_array.py data/val/DIV2K_valid_HR/0801.png \
 *       esp32_firmware/sisr_inference/test_image.h --size 8 8 --arch espcn_light
 */

#include <Arduino.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "espcn_light_float32_data.h"
#include "test_image.h"

// Tile size in the test image must match the model input size used during export.
// If you change --tile in export_tflite.py, regenerate both headers and update
// TEST_INPUT_H / TEST_HR / kTensorArenaSize below accordingly.
constexpr int kTensorArenaSize = 64 * 1024;
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
    Serial.println("\n=== SISR TFLite Micro Test (float32) ===");

    const tflite::Model* model = tflite::GetModel(espcn_light_float32_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Model schema version mismatch");
        return;
    }

    tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddConv2D();
    resolver.AddDepthToSpace();
    resolver.AddGather();
    resolver.AddTanh();

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

    memcpy(input->data.f, test_input_data, TEST_INPUT_LEN * sizeof(float));

    unsigned long t0 = micros();
    TfLiteStatus invoke_status = interpreter.Invoke();
    unsigned long elapsed_us = micros() - t0;

    if (invoke_status != kTfLiteOk) {
        Serial.println("ERROR: Invoke() failed");
        return;
    }

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
