/*
 * ESP32 entry point for SISR C inference library.
 *
 * Protocol (UART):
 *   Host sends:   [4 bytes: tile_w LE] [4 bytes: tile_h LE]
 *                 [tile_w*tile_h*4 bytes: input float32]
 *   ESP32 replies:[tile_w*2*tile_h*2*4 bytes: output float32]
 *
 * Workspace is a static array sized for 8x8 CARN_M tiles.
 * Adjust TILE_W / TILE_H to trade RAM vs throughput on each board.
 * For a board with more RAM, increase TILE_W/TILE_H for better throughput.
 *
 * To use ESPCNLight instead of CARN_M, swap:
 *   - workspace size: espcn_light_workspace_bytes(TILE_W, TILE_H)
 *   - inference call: espcn_light_run(...)
 *   - layer_buf size: 16 is sufficient
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "sisr_weights.h"
#include "carn_m.h"
#include "espcn_light.h"

#define TILE_W 8
#define TILE_H 8

/*
 * Pre-allocate workspace for CARN_M (9 * 64 * TILE_W * TILE_H floats).
 * ESPCNLight needs far less; this single buffer covers both models.
 */
static float workspace[9 * 64 * TILE_W * TILE_H];
static float input_buf[TILE_W * TILE_H];
static float output_buf[TILE_W * 2 * TILE_H * 2];

/* Layer descriptor array — CARN_M needs 21, add margin. */
static SISRLayer layer_buf[32];
static SISRWeights weights;

/*
 * Weights must be loaded from flash. Two options on ESP32:
 *
 * Option A — Embedded array (for small models, prototyping):
 *   1. Convert .sisr to C array: xxd -i model.sisr > esp32/weights_data.h
 *   2. Include it and call sisr_weights_load_mem() [to be added in Phase 2]
 *
 * Option B — SPI flash via SPIFFS/LittleFS (recommended for production):
 *   Use esp-idf SPIFFS to mount flash, then sisr_weights_load("/spiffs/model.sisr", ...)
 */

/*
 * board_init() — platform setup (UART, flash mount, etc.)
 * Replace with esp-idf equivalents: uart_driver_install(), esp_vfs_spiffs_register()
 */
static void board_init(void) {
    /* esp-idf example:
     *   uart_config_t cfg = { .baud_rate = 115200, ... };
     *   uart_driver_install(UART_NUM_0, 1024, 0, 0, NULL, 0);
     *   uart_param_config(UART_NUM_0, &cfg);
     */
}

/*
 * board_read() — blocking read from UART.
 * Replace with: uart_read_bytes(UART_NUM_0, buf, len, portMAX_DELAY)
 */
static void board_read(void *buf, size_t len) {
    /* Placeholder — use esp-idf UART API in production */
    (void)buf; (void)len;
}

/*
 * board_write() — blocking write to UART.
 * Replace with: uart_write_bytes(UART_NUM_0, buf, len)
 */
static void board_write(const void *buf, size_t len) {
    /* Placeholder — use esp-idf UART API in production */
    (void)buf; (void)len;
}

int app_main(void) {
    board_init();

    /*
     * Load model weights from flash.
     * Replace path with your SPIFFS mount point, e.g. "/spiffs/carn_m.sisr"
     * For prototyping without flash, use embedded array (see Option A above).
     */
    if (sisr_weights_load("/spiffs/carn_m.sisr", &weights, layer_buf, 32) != 0) {
        /* Weight load failed — halt or signal error LED */
        while (1) {}
    }

    /* Inference loop: receive tile, run model, send result */
    while (1) {
        /* Receive input tile from host over UART */
        board_read(input_buf, sizeof(input_buf));

        /* Run CARN_M inference */
        int ret = carn_m_run(input_buf, output_buf, &weights,
                             workspace, TILE_W, TILE_H);

        if (ret == 0) {
            /* Send super-resolved tile back to host */
            board_write(output_buf, sizeof(output_buf));
        }
    }

    return 0;
}
