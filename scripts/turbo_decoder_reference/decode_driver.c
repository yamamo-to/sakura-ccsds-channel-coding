/*
 * decode_driver.c — encode/decode CCSDS 131.0-B-2 Turbo codes with the
 * deepspace-turbo reference implementation.
 *
 * Usage:
 *   decode_driver encode <K> <rate> <in.payload> <out.bits>
 *   decode_driver decode <K> <rate> <iterations> <noise_variance> <in.symbols> <out.bits>
 *
 * Symbol convention (from the reference README):
 *   received[i] = 2*encoded[i] - 1   ->  bit 0 = -1, bit 1 = +1
 *
 * The RSC constituent definitions and the interleaver construction are copied
 * verbatim from main.c (case 2/3/4) of the deepspace-turbo repository so that
 * the code under test is byte-identical to the reference simulator.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "libturbocodes.h"

static double *read_doubles(const char *path, int *n) {
    FILE *f = fopen(path, "r");
    if (!f) { perror("fopen"); exit(1); }
    int cap = 1 << 16;
    double *buf = malloc(cap * sizeof(double));
    int cnt = 0;
    double v;
    while (fscanf(f, "%lf", &v) == 1) {
        if (cnt == cap) { cap *= 2; buf = realloc(buf, cap * sizeof(double)); }
        buf[cnt++] = v;
    }
    fclose(f);
    *n = cnt;
    return buf;
}

static int *read_ints(const char *path, int *n) {
    FILE *f = fopen(path, "r");
    if (!f) { perror("fopen"); exit(1); }
    int cap = 1 << 16;
    int *buf = malloc(cap * sizeof(int));
    int cnt = 0, v;
    while (fscanf(f, "%d", &v) == 1) {
        if (cnt == cap) { cap *= 2; buf = realloc(buf, cap * sizeof(int)); }
        buf[cnt++] = v;
    }
    fclose(f);
    *n = cnt;
    return buf;
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 8) {
        fprintf(stderr, "usage: %s encode <K> <rate> <in.payload> <out.bits>\n"
                        "       %s decode <K> <rate> <iterations> <noise_variance> <in.symbols> <out.bits>\n",
                argv[0], argv[0]);
        return 2;
    }

    int mode = 0; /* 0 = encode, 1 = decode */
    if (strcmp(argv[1], "encode") == 0) mode = 0;
    else if (strcmp(argv[1], "decode") == 0) mode = 1;
    else { fprintf(stderr, "unknown mode %s\n", argv[1]); return 2; }

    int K = atoi(argv[2]);
    const char *rate = argv[3];

    int base = 223;
    if (K % (base * 8) != 0) {
        fprintf(stderr, "K must be a multiple of 223*8=%d, got %d\n", base * 8, K);
        return 2;
    }
    int octets = K / (base * 8);

    /* ---- interleaver construction (verbatim from main.c) ---- */
    int p[8] = {31, 37, 43, 47, 53, 59, 61, 67};
    int k1 = 8;
    int k2 = base * octets;
    int *pi = malloc(K * sizeof(int));
    for (int s = 1; s <= K; ++s) {
        int m = (s - 1) % 2;
        int i = (int)floor((s - 1) / (2 * k2));
        int j = (int)floor((s - 1) / 2) - i * k2;
        int t = (19 * i + 1) % (k1 / 2);
        int q = t % 8 + 1;
        int c = (p[q - 1] * j + 21 * m) % k2;
        pi[s - 1] = 2 * (t + c * (k1 / 2) + 1) - m - 1;
    }

    /* ---- RSC constituent codes (verbatim from main.c cases 2/3/4) ---- */
    char *backward = "0011";
    t_convcode *code1, *code2;
    if (strcmp(rate, "1/3") == 0) {
        char *fu[2] = {"10011", "11011"};
        char *fl[1] = {"11011"};
        code1 = convcode_initialize(fu, backward, 2);
        code2 = convcode_initialize(fl, backward, 1);
    } else if (strcmp(rate, "1/4") == 0) {
        char *fu[3] = {"10011", "10101", "11111"};
        char *fl[1] = {"11011"};
        code1 = convcode_initialize(fu, backward, 3);
        code2 = convcode_initialize(fl, backward, 1);
    } else if (strcmp(rate, "1/6") == 0) {
        char *fu[4] = {"10011", "11011", "10101", "11111"};
        char *fl[2] = {"11011", "11111"};
        code1 = convcode_initialize(fu, backward, 4);
        code2 = convcode_initialize(fl, backward, 2);
    } else {
        fprintf(stderr, "unsupported rate %s (use 1/3, 1/4 or 1/6)\n", rate);
        return 2;
    }

    t_turbocode *turbo = turbo_initialize(code1, code2, pi, K);

    if (mode == 0) {
        /* ---- encode mode ---- */
        int n = 0;
        int *packet = read_ints(argv[4], &n);
        if (n != K) {
            fprintf(stderr, "expected %d payload bits, got %d\n", K, n);
            return 2;
        }
        int *encoded = turbo_encode(packet, turbo);
        FILE *out = fopen(argv[5], "w");
        if (!out) { perror("fopen out"); return 1; }
        for (int j = 0; j < turbo->encoded_length; ++j)
            fprintf(out, "%s%d", j ? " " : "", encoded[j]);
        fprintf(out, "\n");
        fclose(out);
        free(encoded);
        free(packet);
        turbocode_clear(turbo); /* frees pi (code->interleaver) */
        return 0;
    }

    /* ---- decode mode ---- */
    int iterations = atoi(argv[4]);
    double noise_variance = atof(argv[5]);

    int n = 0;
    double *received = read_doubles(argv[6], &n);
    if (n != turbo->encoded_length) {
        fprintf(stderr, "expected %d symbols, got %d (rate %s, K %d)\n",
                turbo->encoded_length, n, rate, K);
        return 2;
    }

    int *decoded = turbo_decode(received, iterations, noise_variance, turbo);

    FILE *out = fopen(argv[7], "w");
    if (!out) { perror("fopen out"); return 1; }
    for (int j = 0; j < K; ++j)
        fprintf(out, "%s%d", j ? " " : "", decoded[j]);
    fprintf(out, "\n");
    fclose(out);

    free(decoded);
    free(received);
    turbocode_clear(turbo); /* frees pi (code->interleaver) */
    return 0;
}
