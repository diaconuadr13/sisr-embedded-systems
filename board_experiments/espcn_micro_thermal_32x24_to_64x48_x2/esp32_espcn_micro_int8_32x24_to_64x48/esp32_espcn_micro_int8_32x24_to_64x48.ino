/*
 * SISR TFLite Micro inference test for ESP32 - ESPCN_Micro int8.
 *
 * Test shape:
 *   LR input:  32x24 grayscale
 *   SR output: 64x48 grayscale
 */

#include <Arduino.h>
#include <esp_heap_caps.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "espcn_micro_int8_data.h"
#include "test_image.h"

constexpr int kMaxTensorArenaSize = 64 * 1024;
static uint8_t* tensor_arena = nullptr;
static size_t tensor_arena_size = 0;

static uint8_t* allocate_tensor_arena(size_t* allocated_size) {
    const size_t candidate_sizes[] = {
        64 * 1024, 56 * 1024, 48 * 1024, 40 * 1024, 32 * 1024, 24 * 1024, 16 * 1024,
    };

    Serial.printf("Free internal heap: %u bytes\n",
                  heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    Serial.printf("Largest internal block: %u bytes\n",
                  heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));

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

static int8_t quantize_float(float value, float scale, int zero_point) {
    int q = static_cast<int>(roundf(value / scale)) + zero_point;
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    return static_cast<int8_t>(q);
}

static float dequantize_int8(int8_t value, float scale, int zero_point) {
    return (static_cast<int>(value) - zero_point) * scale;
}

static void fill_input(TfLiteTensor* input) {
    if (input->type == kTfLiteFloat32) {
        memcpy(input->data.f, test_input_data, TEST_INPUT_LEN * sizeof(float));
        return;
    }

    float scale = input->params.scale;
    int zero_point = input->params.zero_point;
    for (int i = 0; i < TEST_INPUT_LEN; i++) {
        input->data.int8[i] = quantize_float(test_input_data[i], scale, zero_point);
    }
}

static float output_value(const TfLiteTensor* output, int i) {
    if (output->type == kTfLiteFloat32) {
        return output->data.f[i];
    }
    return dequantize_int8(output->data.int8[i], output->params.scale,
                           output->params.zero_point);
}

static float compute_psnr(const TfLiteTensor* output, const float* ref, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = static_cast<double>(output_value(output, i)) - ref[i];
        mse += diff * diff;
    }
    mse /= n;
    if (mse == 0.0) return 100.0f;
    return static_cast<float>(10.0 * log10(1.0 / mse));
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

static void dump_output_serial(const char* tag, int h, int w, const TfLiteTensor* output) {
    Serial.print("SISR:");
    Serial.print(tag);
    Serial.print(":");
    Serial.print(h);
    Serial.print(":");
    Serial.print(w);
    Serial.print(":");
    int n = h * w;
    for (int i = 0; i < n; i++) {
        Serial.print(output_value(output, i), 6);
        if (i < n - 1) Serial.print(",");
    }
    Serial.println();
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== SISR TFLite Micro Test (ESPCN_Micro int8) ===");

    tensor_arena = allocate_tensor_arena(&tensor_arena_size);
    if (tensor_arena == nullptr) {
        Serial.println("ERROR: Unable to allocate tensor arena");
        Serial.printf("Largest requested: %d bytes\n", kMaxTensorArenaSize);
        return;
    }

    const tflite::Model* model = tflite::GetModel(espcn_micro_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Model schema version mismatch");
        return;
    }

    tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddConv2D();
    resolver.AddDepthToSpace();

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, tensor_arena_size);

    TfLiteStatus alloc_status = interpreter.AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors() failed - tensor arena too small");
        Serial.printf("Arena used: %d bytes\n", interpreter.arena_used_bytes());
        return;
    }
    Serial.printf("Arena used: %d / %d bytes\n",
                  interpreter.arena_used_bytes(),
                  static_cast<int>(tensor_arena_size));

    TfLiteTensor* input = interpreter.input(0);
    Serial.printf("Input shape: [%d, %d, %d, %d] type: %d scale: %.8f zp: %d\n",
                  input->dims->data[0], input->dims->data[1],
                  input->dims->data[2], input->dims->data[3],
                  input->type, input->params.scale, input->params.zero_point);

    fill_input(input);

    unsigned long t0 = micros();
    TfLiteStatus invoke_status = interpreter.Invoke();
    unsigned long elapsed_us = micros() - t0;

    if (invoke_status != kTfLiteOk) {
        Serial.println("ERROR: Invoke() failed");
        return;
    }

    TfLiteTensor* output = interpreter.output(0);
    int out_len = output->bytes;
    if (output->type == kTfLiteFloat32) {
        out_len /= sizeof(float);
    }

    Serial.printf("Output shape: [%d, %d, %d, %d] type: %d scale: %.8f zp: %d\n",
                  output->dims->data[0], output->dims->data[1],
                  output->dims->data[2], output->dims->data[3],
                  output->type, output->params.scale, output->params.zero_point);

    float psnr = -1.0f;
    if (out_len == TEST_HR_LEN) {
        psnr = compute_psnr(output, test_hr_data, out_len);
    } else {
        Serial.printf("WARN: output len %d != HR ref len %d\n", out_len, TEST_HR_LEN);
    }

    Serial.println("\n--- Results ---");
    Serial.printf("Inference time : %lu us (%.2f ms)\n", elapsed_us, elapsed_us / 1000.0f);
    Serial.printf("Output pixels  : %d\n", out_len);
    if (psnr >= 0.0f) Serial.printf("PSNR vs bicubic: %.2f dB\n", psnr);

    Serial.println();
    Serial.println("--- Image dump ---");
    Serial.println("Starting image dump in 5 seconds...");
    delay(5000);
    Serial.println("SISR_IMG_START");
    dump_pixels_serial("LR", TEST_INPUT_H, TEST_INPUT_W, test_input_data);
    if (out_len == TEST_HR_LEN) {
        dump_output_serial("SR", TEST_HR_H, TEST_HR_W, output);
    }
    dump_pixels_serial("HR", TEST_HR_H, TEST_HR_W, test_hr_data);
    Serial.println("SISR_IMG_END");
    Serial.println("Done.");
}

void loop() {}
