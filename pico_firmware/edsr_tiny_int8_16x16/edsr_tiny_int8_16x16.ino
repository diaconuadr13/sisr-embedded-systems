/*
 * SISR TFLite Micro inference test for Raspberry Pi Pico / RP2040.
 * Model: EDSR_Tiny int8 with float32 input/output.
 *
 * Arduino setup:
 *   - Board core: Raspberry Pi Pico/RP2040
 *   - Board: Raspberry Pi Pico
 *   - Runtime: Pico-compatible TFLite Micro. ArduTFLite does not list
 *     RP2040/Pico support and does not provide these low-level headers.
 *   - Serial monitor: 115200 baud
 *
 * Generate local headers from a checkpoint:
 *   python tools/export_tflite.py edsr_tiny \
 *       runs/EDSR_Tiny_gray_x2/Flickr2K/exp_<ts>/best_model.pth \
 *       pico_firmware/edsr_tiny_int8/ --tile 8 8 --int8
 *   python tools/image_to_c_array.py data/val/DIV2K_valid_HR/0801.png \
 *       pico_firmware/edsr_tiny_int8/test_image.h --size 8 8 --arch edsr_tiny
 *
 * Or copy the matching generated headers from esp32_firmware/edsr_tiny_int8/.
 */

# */
#include <Arduino.h>
#include <TensorFlowLiteMicro.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "edsr_tiny_int8_data.h"
#include "test_image.h"

constexpr int kScale = 2;
constexpr int kBenchmarkSamples = 5;
constexpr int kTensorArenaSize = 96 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static int free_sram_bytes() {
#if defined(ARDUINO_ARCH_RP2040)
    return static_cast<int>(rp2040.getFreeHeap());
#else
    return -1;
#endif
}

static float compute_psnr(const float* a, const float* b, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        mse += diff * diff;
    }
    mse /= n;
    if (mse == 0.0) {
        return 100.0f;
    }
    return static_cast<float>(10.0 * log10(1.0 / mse));
}

static TfLiteStatus invoke_once(
    tflite::MicroInterpreter& interpreter,
    unsigned long* elapsed_us
) {
    TfLiteTensor* input = interpreter.input(0);
    memcpy(input->data.f, test_input_data, TEST_INPUT_LEN * sizeof(float));

    unsigned long t0 = micros();
    TfLiteStatus status = interpreter.Invoke();
    *elapsed_us = micros() - t0;
    return status;
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

static void print_board_log(
    float inference_ms,
    int free_sram_before,
    int free_sram_after
) {
    Serial.println();
    Serial.println("--- Board log ---");
    Serial.println("target=pico");
    Serial.print("tile=");
    Serial.print(TEST_HR_H / kScale);
    Serial.print("x");
    Serial.println(TEST_HR_W / kScale);
    Serial.print("scale=");
    Serial.println(kScale);
    Serial.print("tensor_arena_bytes=");
    Serial.println(kTensorArenaSize);
    Serial.print("free_sram_before=");
    Serial.println(free_sram_before);
    Serial.print("free_sram_after=");
    Serial.println(free_sram_after);
    Serial.print("inference_ms=");
    Serial.println(inference_ms, 3);
}

void setup() {
    Serial.begin(115200);
    unsigned long wait_start = millis();
    while (!Serial && (millis() - wait_start) < 5000) {
        delay(10);
    }
    delay(500);

    Serial.println();
    Serial.println("=== SISR TFLite Micro Test (Pico, EDSR_Tiny int8) ===");

    int free_before = free_sram_bytes();

    const tflite::Model* model = tflite::GetModel(edsr_tiny_int8_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Model schema version mismatch");
        return;
    }

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
        Serial.println("ERROR: AllocateTensors() failed");
        Serial.print("Arena used: ");
        Serial.println(interpreter.arena_used_bytes());
        return;
    }

    int free_after = free_sram_bytes();

    Serial.print("Arena used: ");
    Serial.print(interpreter.arena_used_bytes());
    Serial.print(" / ");
    Serial.println(kTensorArenaSize);

    TfLiteTensor* input = interpreter.input(0);
    Serial.print("Input shape: [");
    Serial.print(input->dims->data[0]);
    Serial.print(", ");
    Serial.print(input->dims->data[1]);
    Serial.print(", ");
    Serial.print(input->dims->data[2]);
    Serial.print(", ");
    Serial.print(input->dims->data[3]);
    Serial.print("] type: ");
    Serial.println(input->type);

    unsigned long elapsed_us = 0;
    if (invoke_once(interpreter, &elapsed_us) != kTfLiteOk) {
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
        Serial.print("WARN: output len ");
        Serial.print(out_len);
        Serial.print(" != HR ref len ");
        Serial.println(TEST_HR_LEN);
    }

    float inference_ms = elapsed_us / 1000.0f;

    Serial.println();
    Serial.println("--- Results ---");
    Serial.print("Inference time: ");
    Serial.print(elapsed_us);
    Serial.print(" us (");
    Serial.print(inference_ms, 3);
    Serial.println(" ms)");
    Serial.print("Output pixels: ");
    Serial.println(out_len);
    if (psnr >= 0.0f) {
        Serial.print("PSNR vs HR: ");
        Serial.print(psnr, 2);
        Serial.println(" dB");
    }

    print_board_log(inference_ms, free_before, free_after);

    Serial.println();
    Serial.println("--- Image dump ---");
    Serial.println("SISR_IMG_START");
    dump_pixels_serial("LR", TEST_INPUT_H, TEST_INPUT_W, test_input_data);
    if (out_len == TEST_HR_LEN && out_f32)
        dump_pixels_serial("SR", TEST_HR_H, TEST_HR_W, out_f32);
    dump_pixels_serial("HR", TEST_HR_H, TEST_HR_W, test_hr_data);
    Serial.println("SISR_IMG_END");

    for (int i = 0; i < kBenchmarkSamples; i++) {
        unsigned long sample_us = 0;
        if (invoke_once(interpreter, &sample_us) != kTfLiteOk) {
            Serial.println("ERROR: Benchmark Invoke() failed");
            return;
        }
        Serial.print("sample_ms=");
        Serial.println(sample_us / 1000.0f, 3);
    }

    Serial.println("Done.");
}

void loop() {}
