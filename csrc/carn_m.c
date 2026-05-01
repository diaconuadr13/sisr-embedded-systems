#include "carn_m.h"
#include "sisr_primitives.h"
#include <string.h>

#define FEATS 64

size_t carn_m_workspace_bytes(int tile_w, int tile_h) {
    return (size_t)9 * FEATS * tile_w * tile_h * sizeof(float);
}

/*
 * 1x1 convolution over logically concatenated input tensors (no physical cat).
 * inputs    : array of num_inputs pointers, each [channels, hw]
 * weight    : [out_c, num_inputs * channels]  (PyTorch layout for 1x1 conv)
 * total_in  : num_inputs * channels
 */
static void conv1x1_multi(
    const float **inputs, int num_inputs, int channels, int hw,
    const float *weight, int out_c, const float *bias,
    float *output)
{
    int total_in = num_inputs * channels;
    for (int oc = 0; oc < out_c; oc++) {
        const float *w_row = weight + (size_t)oc * total_in;
        for (int p = 0; p < hw; p++) {
            float acc = bias ? bias[oc] : 0.0f;
            for (int n = 0; n < num_inputs; n++) {
                const float *in_n    = inputs[n];
                const float *w_chunk = w_row + n * channels;
                for (int ic = 0; ic < channels; ic++)
                    acc += w_chunk[ic] * in_n[(size_t)ic * hw + p];
            }
            output[(size_t)oc * hw + p] = acc;
        }
    }
}

/*
 * shared_block: output = input + conv2(leaky_relu(conv1(input)))
 * Groups=4 for both conv1 and conv2.
 * conv1 result -> tmp_a; conv2 result + residual -> tmp_b
 */
static void shared_block(
    const float *cur, float *tmp_a, float *tmp_b,
    const SISRLayer *conv1_L, const SISRLayer *conv2_L,
    int tile_h, int tile_w)
{
    int hw = tile_h * tile_w;
    sisr_conv2d(cur,   FEATS, tile_h, tile_w,
                conv1_L->weights, FEATS, 3, 3, 1, 4, conv1_L->bias, tmp_a);
    sisr_leaky_relu(tmp_a, FEATS * hw, 0.1f);
    sisr_conv2d(tmp_a, FEATS, tile_h, tile_w,
                conv2_L->weights, FEATS, 3, 3, 1, 4, conv2_L->bias, tmp_b);
    /* residual add */
    for (int i = 0; i < FEATS * hw; i++) tmp_b[i] += cur[i];
}

int carn_m_run(
    const float *input, float *output,
    const SISRWeights *weights,
    float *workspace, int tile_w, int tile_h)
{
    int hw = tile_h * tile_w;

    /* Workspace layout: 9 slots of FEATS*hw floats each
       slot 0: global_h[0] = head output (kept alive throughout)
       slot 1: global_h[1] = after unit_fuse[0]
       slot 2: global_h[2] = after unit_fuse[1]
       slot 3: global_h[3] = after unit_fuse[2] (final features)
       slot 4: unit_h[1]   = fuse[0] output within current unit
       slot 5: unit_h[2]   = fuse[1] output within current unit
       slot 6: unit_h[3]   = fuse[2] output = unit output
       slot 7: tmp_a        = shared_block conv1 output
       slot 8: tmp_b        = shared_block output (conv2 + residual) */
    float *gh[4], *unit_h1, *unit_h2, *unit_h3, *tmp_a, *tmp_b;
    for (int i = 0; i < 4; i++)
        gh[i] = workspace + (size_t)i * FEATS * hw;
    unit_h1 = workspace + (size_t)4 * FEATS * hw;
    unit_h2 = workspace + (size_t)5 * FEATS * hw;
    unit_h3 = workspace + (size_t)6 * FEATS * hw;
    tmp_a   = workspace + (size_t)7 * FEATS * hw;
    tmp_b   = workspace + (size_t)8 * FEATS * hw;

    const SISRLayer *L = weights->layers;

    /* head conv (1->64, 3x3, pad=1) */
    sisr_conv2d(input, 1, tile_h, tile_w,
                L[0].weights, FEATS, 3, 3, 1, 1, L[0].bias, gh[0]);

    /* 3 cascade units */
    for (int u = 0; u < 3; u++) {
        /* Layer indices for unit u:
           base+0 = shared_block.conv1
           base+1 = shared_block.conv2
           base+2 = fuse[0]  (128->64, 1x1)
           base+3 = fuse[1]  (192->64, 1x1)
           base+4 = fuse[2]  (256->64, 1x1) */
        int base    = 1 + u * 5;
        float *u_in = gh[u];   /* unit input = current global */

        /* Inner cascade iteration 0 */
        shared_block(u_in, tmp_a, tmp_b, &L[base], &L[base+1], tile_h, tile_w);
        { const float *ins[2] = {u_in, tmp_b};
          conv1x1_multi(ins, 2, FEATS, hw,
                        L[base+2].weights, FEATS, L[base+2].bias, unit_h1); }

        /* Inner cascade iteration 1 */
        shared_block(unit_h1, tmp_a, tmp_b, &L[base], &L[base+1], tile_h, tile_w);
        { const float *ins[3] = {u_in, unit_h1, tmp_b};
          conv1x1_multi(ins, 3, FEATS, hw,
                        L[base+3].weights, FEATS, L[base+3].bias, unit_h2); }

        /* Inner cascade iteration 2 */
        shared_block(unit_h2, tmp_a, tmp_b, &L[base], &L[base+1], tile_h, tile_w);
        { const float *ins[4] = {u_in, unit_h1, unit_h2, tmp_b};
          conv1x1_multi(ins, 4, FEATS, hw,
                        L[base+4].weights, FEATS, L[base+4].bias, unit_h3); }

        /* Global cascade fuse after unit u */
        int gf = 16 + u;   /* unit_fuse[u] index */
        if (u == 0) {
            const float *ins[2] = {gh[0], unit_h3};
            conv1x1_multi(ins, 2, FEATS, hw,
                          L[gf].weights, FEATS, L[gf].bias, gh[1]);
        } else if (u == 1) {
            const float *ins[3] = {gh[0], gh[1], unit_h3};
            conv1x1_multi(ins, 3, FEATS, hw,
                          L[gf].weights, FEATS, L[gf].bias, gh[2]);
        } else {
            const float *ins[4] = {gh[0], gh[1], gh[2], unit_h3};
            conv1x1_multi(ins, 4, FEATS, hw,
                          L[gf].weights, FEATS, L[gf].bias, gh[3]);
        }
    }

    /* upsample conv (64->4, 3x3, pad=1) into tmp_a */
    sisr_conv2d(gh[3], FEATS, tile_h, tile_w,
                L[19].weights, 4, 3, 3, 1, 1, L[19].bias, tmp_a);

    /* pixel_shuffle(2) -> output [1, tile_h*2, tile_w*2] */
    sisr_pixel_shuffle(tmp_a, 4, tile_h, tile_w, 2, output);

    return 0;
}
