/*
 * SISR TFLite Micro inference test for Raspberry Pi Pico / RP2040 - ESPCN_Full int8.
 *
 * Test shape:
 *   LR input:  32x24 grayscale
 *   SR output: 64x48 grayscale
 */

#include <Arduino.h>
#include <TensorFlowLiteMicro.h>
#include <unistd.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "espcn_int8_data.h"
#include "test_image.h"

constexpr int kScale = 2;
constexpr int kBenchmarkSamples = 5;
constexpr int kTensorArenaSize = 112 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static int free_sram_bytes() {
#if defined(ARDUINO_ARCH_RP2040)
    char stack_marker = 0;
    void* heap_end = sbrk(0);
    if (heap_end == reinterpret_cast<void*>(-1)) {
        return -1;
    }
    return static_cast<int>(&stack_marker - static_cast<char*>(heap_end));
#else
    return -1;
#endif
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
    unsigned long wait_start = millis();
    while (!Serial && (millis() - wait_start) < 30000) {
        delay(10);
    }
    delay(500);

    Serial.println("\n=== SISR TFLite Micro Test (Pico, ESPCN_Full int8) ===");
    int free_before = free_sram_bytes();

    const tflite::Model* model = tflite::GetModel(espcn_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Model schema version mismatch");
        return;
    }

    tflite::MicroMutableOpResolver<3> resolver;
    resolver.AddConv2D();
    resolver.AddDepthToSpace();
    resolver.AddTanh();

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    TfLiteStatus alloc_status = interpreter.AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors() failed - tensor arena too small");
        Serial.printf("Arena used: %d bytes\n", interpreter.arena_used_bytes());
        return;
    }
    int free_after = free_sram_bytes();
    Serial.printf("Arena used: %d / %d bytes\n",
                  interpreter.arena_used_bytes(),
                  kTensorArenaSize);

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
    Serial.println("--- Board log ---");
    Serial.println("target=pico");
    Serial.printf("tile=%dx%d\n", TEST_INPUT_W, TEST_INPUT_H);
    Serial.printf("scale=%d\n", kScale);
    Serial.printf("tensor_arena_bytes=%d\n", kTensorArenaSize);
    Serial.printf("free_sram_before=%d\n", free_before);
    Serial.printf("free_sram_after=%d\n", free_after);
    Serial.printf("inference_ms=%.3f\n", elapsed_us / 1000.0f);

    Serial.println();
    Serial.println("--- Image dump ---");
    Serial.println("SISR_IMG_START");
    dump_pixels_serial("LR", TEST_INPUT_H, TEST_INPUT_W, test_input_data);
    if (out_len == TEST_HR_LEN) {
        dump_output_serial("SR", TEST_HR_H, TEST_HR_W, output);
    }
    dump_pixels_serial("HR", TEST_HR_H, TEST_HR_W, test_hr_data);
    Serial.println("SISR_IMG_END");

    for (int i = 0; i < kBenchmarkSamples; i++) {
        fill_input(input);
        unsigned long sample_t0 = micros();
        if (interpreter.Invoke() != kTfLiteOk) {
            Serial.println("ERROR: Benchmark Invoke() failed");
            return;
        }
        unsigned long sample_us = micros() - sample_t0;
        Serial.printf("sample_ms=%.3f\n", sample_us / 1000.0f);
    }

    Serial.println("Done.");
}

void loop() {}
