#pragma once
#include "sisr_weights.h"
#include <stddef.h>

size_t carn_m_workspace_bytes(int tile_w, int tile_h);

/*
 * Run CARN_M inference on one grayscale tile.
 * input     : float32[tile_h * tile_w], values in [0.0, 1.0]
 * output    : float32[tile_h*2 * tile_w*2]
 * weights   : loaded with sisr_weights_load(); must have 21 layers
 * workspace : caller-allocated, size >= carn_m_workspace_bytes()
 * Returns 0 on success, -1 on error.
 */
int carn_m_run(
    const float *input, float *output,
    const SISRWeights *weights,
    float *workspace, int tile_w, int tile_h
);
