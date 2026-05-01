#include "sisr_primitives.h"
#include <math.h>

void sisr_conv2d(
    const float *input,  int in_c,  int in_h,  int in_w,
    const float *weight, int out_c, int kH,    int kW,
    int padding, int groups,
    const float *bias,
    float *output)
{
    int group_in  = in_c  / groups;
    int group_out = out_c / groups;
    int out_h = in_h;
    int out_w = in_w;

    for (int oc = 0; oc < out_c; oc++) {
        int g        = oc / group_out;
        int ic_start = g * group_in;
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float acc = bias ? bias[oc] : 0.0f;
                for (int ic = 0; ic < group_in; ic++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int ih = oh + kh - padding;
                            int iw = ow + kw - padding;
                            if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w)
                                continue;
                            int in_idx = (ic_start + ic) * in_h * in_w
                                         + ih * in_w + iw;
                            int w_idx  = oc * group_in * kH * kW
                                         + ic * kH * kW + kh * kW + kw;
                            acc += input[in_idx] * weight[w_idx];
                        }
                    }
                }
                output[oc * out_h * out_w + oh * out_w + ow] = acc;
            }
        }
    }
}

void sisr_pixel_shuffle(
    const float *input, int channels, int h, int w, int scale,
    float *output)
{
    int out_c = channels / (scale * scale);
    int out_h = h * scale;
    int out_w = w * scale;

    for (int c = 0; c < out_c; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int h_sub = oh % scale;
                int w_sub = ow % scale;
                int in_c_idx = c * (scale * scale) + h_sub * scale + w_sub;
                int in_idx   = in_c_idx * h * w + (oh / scale) * w + (ow / scale);
                int out_idx  = c * out_h * out_w + oh * out_w + ow;
                output[out_idx] = input[in_idx];
            }
        }
    }
}

void sisr_tanh(float *data, int n) {
    for (int i = 0; i < n; i++) data[i] = tanhf(data[i]);
}

void sisr_leaky_relu(float *data, int n, float neg_slope) {
    for (int i = 0; i < n; i++)
        data[i] = data[i] >= 0.0f ? data[i] : neg_slope * data[i];
}
