#include "espcn_light.h"
#include "sisr_primitives.h"
#include <stddef.h>

size_t espcn_light_workspace_bytes(int tile_w, int tile_h) {
    return (size_t)2 * 32 * tile_w * tile_h * sizeof(float);
}

int espcn_light_run(
    const float *input, float *output,
    const SISRWeights *weights,
    float *workspace, int tile_w, int tile_h)
{
    float *buf_a = workspace;
    float *buf_b = workspace + 32 * tile_w * tile_h;
    int    hw    = tile_w * tile_h;

    const SISRLayer *L = weights->layers;

    /* Layer 0: Conv(1->32, 5x5, pad=2) + Tanh */
    sisr_conv2d(input, 1, tile_h, tile_w,
                L[0].weights, 32, 5, 5, 2, 1, L[0].bias, buf_a);
    sisr_tanh(buf_a, 32 * hw);

    /* Layer 1: Conv(32->16, 3x3, pad=1) + Tanh */
    sisr_conv2d(buf_a, 32, tile_h, tile_w,
                L[1].weights, 16, 3, 3, 1, 1, L[1].bias, buf_b);
    sisr_tanh(buf_b, 16 * hw);

    /* Layer 2: Conv(16->4, 3x3, pad=1) */
    sisr_conv2d(buf_b, 16, tile_h, tile_w,
                L[2].weights, 4, 3, 3, 1, 1, L[2].bias, buf_a);

    /* Layer 3: PixelShuffle(2) */
    sisr_pixel_shuffle(buf_a, 4, tile_h, tile_w, 2, output);

    return 0;
}
