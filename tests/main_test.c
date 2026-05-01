#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sisr_primitives.h"
#include "sisr_weights.h"
#include "espcn_light.h"
#include "carn_m.h"

static float *read_bin(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    float *buf = malloc(n * sizeof(float));
    if (!buf) { fprintf(stderr, "malloc failed\n"); exit(1); }
    if (fread(buf, sizeof(float), n, f) != n) {
        fprintf(stderr, "Short read %s\n", path); fclose(f); free(buf); exit(1);
    }
    fclose(f);
    return buf;
}

static void write_bin(const char *path, const float *data, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    if (fwrite(data, sizeof(float), n, f) != n)
        fprintf(stderr, "Short write %s\n", path);
    fclose(f);
}

/* conv2d in_c in_h in_w out_c kH kW pad groups has_bias input.bin weight.bin [bias.bin] output.bin */
static int cmd_conv2d(int argc, char **argv) {
    if (argc < 13) { fprintf(stderr, "Usage: conv2d ...\n"); return 1; }
    int in_c=atoi(argv[1]), in_h=atoi(argv[2]), in_w=atoi(argv[3]);
    int out_c=atoi(argv[4]), kH=atoi(argv[5]), kW=atoi(argv[6]);
    int pad=atoi(argv[7]), groups=atoi(argv[8]), has_bias=atoi(argv[9]);
    if (has_bias && argc < 14) { fprintf(stderr, "Usage: conv2d ... (with bias needs 14 args)\n"); return 1; }
    const char *inp_path = argv[10];
    const char *wgt_path = argv[11];
    const char *bia_path = has_bias ? argv[12] : NULL;
    const char *out_path = has_bias ? argv[13] : argv[12];

    int gin = in_c / groups;
    float *input  = read_bin(inp_path, (size_t)in_c * in_h * in_w);
    float *weight = read_bin(wgt_path, (size_t)out_c * gin * kH * kW);
    float *bias   = has_bias ? read_bin(bia_path, out_c) : NULL;
    float *output = malloc((size_t)out_c * in_h * in_w * sizeof(float));
    if (!output) { fprintf(stderr, "malloc failed\n"); exit(1); }

    sisr_conv2d(input, in_c, in_h, in_w, weight, out_c, kH, kW,
                pad, groups, bias, output);
    write_bin(out_path, output, (size_t)out_c * in_h * in_w);

    free(input); free(weight); free(bias); free(output);
    return 0;
}

/* pixel_shuffle channels h w scale input.bin output.bin */
static int cmd_pixel_shuffle(int argc __attribute__((unused)), char **argv) {
    int channels=atoi(argv[1]), h=atoi(argv[2]);
    int w=atoi(argv[3]), scale=atoi(argv[4]);
    float *input  = read_bin(argv[5], (size_t)channels * h * w);
    if (!input) return 1;
    int out_c = channels / (scale * scale);
    float *output = malloc((size_t)out_c * h * scale * w * scale * sizeof(float));
    if (!output) { free(input); return 1; }
    sisr_pixel_shuffle(input, channels, h, w, scale, output);
    write_bin(argv[6], output, (size_t)out_c * h * scale * w * scale);
    free(input); free(output);
    return 0;
}

/* tanh n input.bin output.bin */
static int cmd_tanh(int argc __attribute__((unused)), char **argv) {
    int n = atoi(argv[1]);
    float *data = read_bin(argv[2], n);
    if (!data) return 1;
    sisr_tanh(data, n);
    write_bin(argv[3], data, n);
    free(data); return 0;
}

/* leaky_relu n neg_slope input.bin output.bin */
static int cmd_leaky_relu(int argc __attribute__((unused)), char **argv) {
    int n = atoi(argv[1]);
    float neg_slope = (float)atof(argv[2]);
    float *data = read_bin(argv[3], n);
    if (!data) return 1;
    sisr_leaky_relu(data, n, neg_slope);
    write_bin(argv[4], data, n);
    free(data); return 0;
}

/* espcn_light tile_w tile_h input.bin weights.sisr output.bin */
static int cmd_espcn_light(int argc __attribute__((unused)), char **argv) {
    int tile_w = atoi(argv[1]), tile_h = atoi(argv[2]);
    const char *inp_path  = argv[3];
    const char *sisr_path = argv[4];
    const char *out_path  = argv[5];

    SISRLayer   layer_buf[16];
    SISRWeights weights;
    if (sisr_weights_load(sisr_path, &weights, layer_buf, 16) != 0) return -1;

    float *input = read_bin(inp_path, (size_t)tile_w * tile_h);
    if (!input) { sisr_weights_free(&weights); return 1; }
    float *output = malloc((size_t)tile_w * 2 * tile_h * 2 * sizeof(float));
    if (!output) { free(input); sisr_weights_free(&weights); return 1; }
    size_t ws = espcn_light_workspace_bytes(tile_w, tile_h);
    float *workspace = malloc(ws);
    if (!workspace) { free(input); free(output); sisr_weights_free(&weights); return 1; }

    int ret = espcn_light_run(input, output, &weights, workspace, tile_w, tile_h);
    if (ret == 0)
        write_bin(out_path, output, (size_t)tile_w * 2 * tile_h * 2);

    free(input); free(output); free(workspace);
    sisr_weights_free(&weights);
    return ret;
}

/* carn_m tile_w tile_h input.bin weights.sisr output.bin */
static int cmd_carn_m(int argc __attribute__((unused)), char **argv) {
    int tile_w = atoi(argv[1]), tile_h = atoi(argv[2]);
    const char *inp_path  = argv[3];
    const char *sisr_path = argv[4];
    const char *out_path  = argv[5];

    SISRLayer   layer_buf[32];
    SISRWeights weights;
    if (sisr_weights_load(sisr_path, &weights, layer_buf, 32) != 0) return -1;

    float *input = read_bin(inp_path, (size_t)tile_w * tile_h);
    if (!input) { sisr_weights_free(&weights); return 1; }
    float *output = malloc((size_t)tile_w * 2 * tile_h * 2 * sizeof(float));
    if (!output) { free(input); sisr_weights_free(&weights); return 1; }
    size_t ws = carn_m_workspace_bytes(tile_w, tile_h);
    float *workspace = malloc(ws);
    if (!workspace) { free(input); free(output); sisr_weights_free(&weights); return 1; }

    int ret = carn_m_run(input, output, &weights, workspace, tile_w, tile_h);
    if (ret == 0)
        write_bin(out_path, output, (size_t)tile_w * 2 * tile_h * 2);

    free(input); free(output); free(workspace);
    sisr_weights_free(&weights);
    return ret;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: sisr_test <op> ...\n"); return 1; }
    if (strcmp(argv[1], "conv2d") == 0) return cmd_conv2d(argc - 1, argv + 1);
    if (strcmp(argv[1], "pixel_shuffle") == 0) return cmd_pixel_shuffle(argc - 1, argv + 1);
    if (strcmp(argv[1], "tanh")       == 0) return cmd_tanh(argc - 1, argv + 1);
    if (strcmp(argv[1], "leaky_relu") == 0) return cmd_leaky_relu(argc - 1, argv + 1);
    if (strcmp(argv[1], "espcn_light") == 0) return cmd_espcn_light(argc - 1, argv + 1);
    if (strcmp(argv[1], "carn_m") == 0) return cmd_carn_m(argc - 1, argv + 1);
    fprintf(stderr, "Unknown op: %s\n", argv[1]);
    return 1;
}
