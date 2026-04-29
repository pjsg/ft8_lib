/* test_refine.c — Generate a known FT8 signal, run decoder + refinement,
   and check TOA/freq accuracy. */

#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "common/wave.h"
#include "ft8/pack.h"
#include "ft8/encode.h"
#include "ft8/constants.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SAMPLE_RATE 12000
#define SLOT_TIME   15.0f
#define NUM_SAMPLES (SAMPLE_RATE * SLOT_TIME)

/* FT8 GFSK constants (match gen_ft8.c) */
#define FT8_SYMBOL_BT 2.0f
#define GFSK_CONST_K  5.336446f

static void gfsk_pulse(int n_spsym, float symbol_bt, float *pulse)
{
    for (int i = 0; i < 3 * n_spsym; ++i) {
        float t = (float)i / (float)n_spsym - 1.5f;
        float arg1 = GFSK_CONST_K * symbol_bt * (t + 0.5f);
        float arg2 = GFSK_CONST_K * symbol_bt * (t - 0.5f);
        pulse[i] = (erff(arg1) - erff(arg2)) / 2;
    }
}

static void synth_gfsk(const uint8_t *sym, int n_sym, float f0,
                       float symbol_bt, float symbol_period,
                       float *out)
{
    int n_spsym = (int)(0.5f + SAMPLE_RATE * symbol_period);
    int n_wave  = n_sym * n_spsym;
    float dphi_peak = 2 * M_PI / n_spsym;

    float *dphi = malloc((n_wave + 2 * n_spsym) * sizeof(float));
    for (int i = 0; i < n_wave + 2 * n_spsym; ++i)
        dphi[i] = 2 * M_PI * f0 / SAMPLE_RATE;

    float *pulse = malloc(3 * n_spsym * sizeof(float));
    gfsk_pulse(n_spsym, symbol_bt, pulse);

    for (int i = 0; i < n_sym; ++i)
        for (int j = 0; j < 3 * n_spsym; ++j)
            dphi[j + i * n_spsym] += dphi_peak * sym[i] * pulse[j];

    for (int j = 0; j < 2 * n_spsym; ++j) {
        dphi[j]               += dphi_peak * pulse[j + n_spsym] * sym[0];
        dphi[j + n_wave]      += dphi_peak * pulse[j] * sym[n_sym - 1];
    }

    float phi = 0;
    for (int k = 0; k < n_wave; ++k) {
        out[k] = sinf(phi);
        phi = fmodf(phi + dphi[k + n_spsym], 2 * M_PI);
    }

    int n_ramp = n_spsym / 8;
    for (int i = 0; i < n_ramp; ++i) {
        float env = (1 - cosf(2 * M_PI * i / (2 * n_ramp))) / 2;
        out[i] *= env;
        out[n_wave - 1 - i] *= env;
    }

    free(dphi);
    free(pulse);
}

static float randn(float mean, float std)
{
    static int has_spare = 0;
    static float spare;
    if (!has_spare) {
        float x, y, r2;
        do {
            x = 2.0f * (float)rand() / RAND_MAX - 1.0f;
            y = 2.0f * (float)rand() / RAND_MAX - 1.0f;
            r2 = x * x + y * y;
        } while (r2 >= 1.0f || r2 == 0.0f);
        float mag = sqrtf(-2.0f * logf(r2) / r2);
        spare = x * mag;
        has_spare = 1;
        return mean + std * y * mag;
    }
    has_spare = 0;
    return mean + std * spare;
}

int main(void)
{
    srand(42);

    const char *message = "CQ TEST";
    printf("=== Refinement Self-Test ===\n");
    printf("Message: %s\n", message);

    /* Pack and encode */
    uint8_t bits[10];
    if (pack77(message, bits) < 0) {
        fprintf(stderr, "pack77 failed!\n");
        return 1;
    }

    uint8_t tones[FT8_NN];
    ft8_encode(bits, tones);

    int n_spsym = (int)(0.5f + SAMPLE_RATE * FT8_SYMBOL_PERIOD);
    int sig_len = FT8_NN * n_spsym;

    /* ---- Test cases ---- */
    struct {
        float freq;
        int   offset;    /* sample offset in 15s buffer */
        float noise;     /* noise std dev */
    } tests[] = {
        { 1000.0f, 5000,  0.1f },   /* 416.67ms offset, clean */
        { 1500.5f, 10000, 0.05f },  /* 833.33ms offset, less noise */
        { 750.25f, 2000, 0.15f },   /* 166.67ms offset, noisy */
        { 2500.0f, 12000, 0.08f },  /* 1000ms offset */
    };
    int ntests = sizeof(tests) / sizeof(tests[0]);

    for (int t = 0; t < ntests; ++t) {
        float freq    = tests[t].freq;
        int   offset  = tests[t].offset;
        float noise_s = tests[t].noise;
        double expected_toa_ms = offset * 1000.0 / SAMPLE_RATE;

        /* Generate signal buffer: noise + FT8 at known offset */
        float *buf = calloc(NUM_SAMPLES, sizeof(float));
        for (int i = 0; i < NUM_SAMPLES; ++i)
            buf[i] = randn(0, noise_s);

        /* Add FT8 signal */
        float *sig = malloc(sig_len * sizeof(float));
        synth_gfsk(tones, FT8_NN, freq, FT8_SYMBOL_BT, FT8_SYMBOL_PERIOD, sig);
        for (int i = 0; i < sig_len; ++i)
            buf[offset + i] += sig[i] * 0.5f;  /* scale signal to 50% amplitude */

        /* Save as WAV */
        char wavpath[256];
        snprintf(wavpath, sizeof(wavpath), "/tmp/test_refine_%d.wav", t);
        save_wav(buf, NUM_SAMPLES, SAMPLE_RATE, wavpath);

        /* Run decoder */
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
                 "./decode_ft8 -8 -f 0 -n %s 2>/dev/null", wavpath);
        FILE *fp = popen(cmd, "r");
        if (!fp) {
            fprintf(stderr, "Failed to run decode_ft8 for test %d\n", t);
            free(buf);
            free(sig);
            continue;
        }

        char line[1024];
        while (fgets(line, sizeof(line), fp)) {
            if (strstr(line, message)) {
                /* Parse output */
                float toa_ms, fine_freq, snr;
                char *tp = strstr(line, "[TOA=");
                if (tp && sscanf(tp + 5, "%fms FINE=%fHz SNR=%f",
                           &toa_ms, &fine_freq, &snr) == 3) {
                    double toa_err_ms  = toa_ms - expected_toa_ms;
                    double freq_err_hz = fine_freq - freq;
                    printf("Test %d: freq=%.2f Hz, offset=%d samples (%.2fms)\n",
                           t, freq, offset, expected_toa_ms);
                    printf("  Refined TOA  = %.3f ms (err = %+6.3f ms, %+4.0f μs)\n",
                           toa_ms, toa_err_ms, toa_err_ms * 1000);
                    printf("  Refined Freq = %.2f Hz (err = %+6.2f Hz)\n",
                           fine_freq, freq_err_hz);
                    printf("  SNR = %.1f\n", snr);
                } else {
                    printf("Test %d: decoded but no refinement data\n", t);
                    printf("  Line: %s\n", line);
                }
            }
        }
        pclose(fp);

        /* Clean up */
        unlink(wavpath);
        free(buf);
        free(sig);
    }

    return 0;
}
