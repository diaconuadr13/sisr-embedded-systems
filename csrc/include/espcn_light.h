#pragma once
#include "sisr_weights.h"
#include <stddef.h>

/* Bytes needed in the workspace buffer for given tile dimensions. */
size_t espcn_light_workspace_bytes(int tile_w, int tile_h);

/*
 * Run ESPCNLight inference on one grayscale tile.
 * input     : float32[tile_h * tile_w], values in [0.0, 1.0]
 * output    : float32[tile_h*2 * tile_w*2]  (scale factor = 2)
 * weights   : loaded with sisr_weights_load(); must have 4 layers
 * workspace : caller-allocated, size >= espcn_light_workspace_bytes()
 * Returns 0 on success, -1 on error.
 */
int espcn_light_run(
    const float *input, float *output,
    const SISRWeights *weights,
    float *workspace, int tile_w, int tile_h
);
