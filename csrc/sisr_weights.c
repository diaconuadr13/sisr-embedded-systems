#include "sisr_weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int sisr_weights_load(const char *path, SISRWeights *out,
                      SISRLayer *layers_buf, int max_layers)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }

    /* Read entire file into buffer */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);
    unsigned char *buf = malloc(fsize);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, fsize, f) != (size_t)fsize) {
        fclose(f); free(buf); return -1;
    }
    fclose(f);

    /* Parse header: magic(4) + version(1) + num_layers(2) = 7 bytes */
    if (fsize < 7 || memcmp(buf, "SISR", 4) != 0) {
        fprintf(stderr, "Bad magic in %s\n", path);
        free(buf); return -1;
    }
    /* buf[4] = version (unused for now) */
    int num_layers = (int)buf[5] | ((int)buf[6] << 8);
    if (num_layers > max_layers) {
        fprintf(stderr, "Too many layers %d (max %d)\n", num_layers, max_layers);
        free(buf); return -1;
    }

    const unsigned char *p = buf + 7;

    for (int i = 0; i < num_layers; i++) {
        SISRLayer *L = &layers_buf[i];
        L->layer_type   = p[0];
        L->out_channels = (int)p[1] | ((int)p[2] << 8);
        L->in_channels  = (int)p[3] | ((int)p[4] << 8);
        L->kernel_h     = p[5];
        L->kernel_w     = p[6];
        L->groups       = p[7];
        L->has_bias     = p[8];
        p += 9;

        if (L->layer_type == SISR_LAYER_PIXSHUFFLE) {
            /* out_channels holds scale factor; no weight data */
            L->weights = NULL;
            L->bias    = NULL;
        } else {
            int gin = L->in_channels / L->groups;
            size_t w_count = (size_t)L->out_channels * gin
                             * L->kernel_h * L->kernel_w;
            L->weights = (const float *)p;
            p += w_count * sizeof(float);
            if (L->has_bias) {
                L->bias = (const float *)p;
                p += (size_t)L->out_channels * sizeof(float);
            } else {
                L->bias = NULL;
            }
        }
    }

    out->num_layers = num_layers;
    out->layers     = layers_buf;
    out->_buf       = buf;
    out->_buf_size  = (size_t)fsize;
    return 0;
}

void sisr_weights_free(SISRWeights *w) {
    free(w->_buf);
    w->_buf = NULL;
}
