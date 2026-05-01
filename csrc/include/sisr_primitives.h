#pragma once

/* All tensors use CHW layout: [channels, height, width], float32, row-major. */

/*
 * 2D convolution with optional grouping and bias.
 * input  : [in_c, in_h, in_w]
 * weight : [out_c, in_c/groups, kH, kW]  (PyTorch weight layout)
 * bias   : [out_c] or NULL
 * output : [out_c, in_h, in_w]  (assumes padding = (k-1)/2, same-size output)
 */
void sisr_conv2d(
    const float *input,  int in_c,  int in_h,  int in_w,
    const float *weight, int out_c, int kH,    int kW,
    int padding, int groups,
    const float *bias,
    float *output
);

/*
 * Sub-pixel shuffle (depth-to-space).
 * input  : [channels * scale * scale, h, w]
 * output : [channels, h * scale, w * scale]
 */
void sisr_pixel_shuffle(
    const float *input, int channels, int h, int w, int scale,
    float *output
);

/* In-place tanh activation over n elements. */
void sisr_tanh(float *data, int n);

/* In-place leaky ReLU over n elements. */
void sisr_leaky_relu(float *data, int n, float neg_slope);
