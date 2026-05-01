#pragma once
#include <stddef.h>

#define SISR_LAYER_CONV2D      0
#define SISR_LAYER_PIXSHUFFLE  2

typedef struct {
    int         layer_type;
    int         out_channels;
    int         in_channels;
    int         kernel_h, kernel_w;
    int         groups;
    int         has_bias;
    const float *weights;   /* points into file buffer; NULL for pixel_shuffle */
    const float *bias;      /* NULL if has_bias=0 or pixel_shuffle */
} SISRLayer;

typedef struct {
    int        num_layers;
    SISRLayer *layers;      /* caller-allocated array of SISRLayer */
    void      *_buf;        /* internal: raw file contents kept alive */
    size_t     _buf_size;
} SISRWeights;

/*
 * Load a .sisr file.
 * layers_buf: caller-allocated array; must hold at least max_layers entries.
 * Returns 0 on success, -1 on error (prints message to stderr).
 */
int  sisr_weights_load(const char *path, SISRWeights *out,
                       SISRLayer *layers_buf, int max_layers);

/* Free the internal file buffer (does NOT free layers_buf). */
void sisr_weights_free(SISRWeights *w);
