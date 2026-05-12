#include "refine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fftw3.h>

#include "ft8/encode.h"
#include "ft8/constants.h"

#include <stdbool.h>

#include "common/debug.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern bool Trace;

/* Shared template generator (Double Precision) */
static void make_tpl_shared(const uint8_t *tones, int n_sym, float f0, float symbol_bt, float symbol_period, int sample_rate, float *out_re, float *out_im) {
    int n_spsym = (int)(0.5f + sample_rate * symbol_period);
    int n_wave = n_sym * n_spsym;
    double dphi_peak = 2.0 * M_PI / n_spsym;

    double *dphi = malloc((n_wave + 2 * n_spsym) * sizeof(double));
    double f_base = 2.0 * M_PI * (double)f0 / (double)sample_rate;
    for (int i = 0; i < n_wave + 2 * n_spsym; ++i) dphi[i] = f_base;

    double *pulse = malloc(3 * n_spsym * sizeof(double));
    for (int i = 0; i < 3 * n_spsym; ++i) {
        double t = (double)i / (double)n_spsym - 1.5;
        pulse[i] = (erf(5.336446 * (double)symbol_bt * (t + 0.5)) - erf(5.336446 * (double)symbol_bt * (t - 0.5))) / 2.0;
    }

    for (int i = 0; i < n_sym; ++i)
        for (int j = 0; j < 3 * n_spsym; ++j)
            dphi[j + i * n_spsym] += dphi_peak * (double)tones[i] * pulse[j];

    for (int k = 0; k < 2 * n_spsym; ++k) {
        dphi[k]               += dphi_peak * pulse[k + n_spsym] * (double)tones[0];
        dphi[k + n_wave]      += dphi_peak * pulse[k]           * (double)tones[n_sym - 1];
    }

    double phi = 0;
    for (int i = 0; i < n_wave; ++i) {
        out_re[i] = (float)cos(phi);
        out_im[i] = (float)sin(phi);
        phi += dphi[i + n_spsym];
    }
    free(dphi); free(pulse);
}

int refine_signal_params(const float *signal, int signal_len, int sample_rate,
                         const uint8_t *payload, const char *text,
                         float coarse_freq_hz, float coarse_time_sec,
                         int n_sym, float symbol_period, float symbol_bt,
                         precision_report_t *report) {
    uint8_t tones[FT8_NN];
    ft8_encode(payload, tones);
    int n_spsym = (int)(0.5f + sample_rate * symbol_period);
    int tpl_len = n_sym * n_spsym;

    // FFT setup
    int padding = sample_rate; // 1s padding for robustness
    int fft_whole = 1;
    while (fft_whole < signal_len + padding + tpl_len) fft_whole <<= 1;
    
    float *a_w = fftwf_malloc(fft_whole * sizeof(float));
    fftwf_complex *A_w = fftwf_alloc_complex(fft_whole/2+1), *B_w = fftwf_alloc_complex(fft_whole), *C_w = fftwf_alloc_complex(fft_whole);
    fftwf_plan pw_a = fftwf_plan_dft_r2c_1d(fft_whole, a_w, A_w, FFTW_ESTIMATE);
    fftwf_plan pw_b = fftwf_plan_dft_1d(fft_whole, B_w, B_w, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_plan pw_i = fftwf_plan_dft_1d(fft_whole, B_w, C_w, FFTW_BACKWARD, FFTW_ESTIMATE);

    memset(a_w, 0, fft_whole * sizeof(float));
    for(int i=0; i<signal_len; i++) a_w[i+padding] = signal[i];
    fftwf_execute(pw_a);

    float *tpl_re = malloc(tpl_len * sizeof(float)), *tpl_im = malloc(tpl_len * sizeof(float));
    float *mag_total = calloc(fft_whole, sizeof(float));
    float *best_mag_grid = malloc(fft_whole * sizeof(float));
    double best_mag = -1.0; int best_lag = 0; float best_f = coarse_freq_hz;

    /* ---- Stage 1: Coherent FFT Sync ---- */
    for (float f = coarse_freq_hz - 1.0f; f <= coarse_freq_hz + 1.0f; f += 0.1f) {
        memset(mag_total, 0, fft_whole * sizeof(float));
        make_tpl_shared(tones, n_sym, f, symbol_bt, symbol_period, sample_rate, tpl_re, tpl_im);
        
        for (int b = 0; b < 3; b++) {
            int b_off = ((b == 0) ? 0 : (b == 1) ? 26 : 52) * n_spsym;
            int b_len = ((b == 0) ? 26 : (b == 1) ? 26 : 27) * n_spsym;
            double b_energy = 0;
            memset(B_w, 0, fft_whole * sizeof(fftwf_complex));
            for (int i = 0; i < b_len; i++) {
                B_w[i][0] = tpl_re[b_off + i]; B_w[i][1] = tpl_im[b_off + i];
                b_energy += (double)B_w[i][0]*B_w[i][0] + (double)B_w[i][1]*B_w[i][1];
            }
            fftwf_execute(pw_b);
            for (int k = 0; k <= fft_whole/2; k++) {
                float r = A_w[k][0]*B_w[k][0] + A_w[k][1]*B_w[k][1], im = A_w[k][1]*B_w[k][0] - A_w[k][0]*B_w[k][1];
                B_w[k][0] = r; B_w[k][1] = im;
            }
            memset(&B_w[fft_whole/2+1], 0, (fft_whole - (fft_whole/2+1)) * sizeof(fftwf_complex));
            fftwf_execute(pw_i);
            float norm = (b_energy > 0) ? 1.0f / sqrtf((float)b_energy) : 0;
            for (int i = 0; i < fft_whole; i++) {
                int lag = i - b_off;
                if (lag >= 0 && lag < fft_whole) mag_total[lag] += sqrtf(C_w[i][0]*C_w[i][0] + C_w[i][1]*C_w[i][1]) * norm;
            }
        }

        int cs = (int)(round(coarse_time_sec * sample_rate) + padding) - sample_rate/2;
        int ce = (int)(round(coarse_time_sec * sample_rate) + padding) + sample_rate/2;
        if (cs < 0) cs = 0; if (ce >= fft_whole) ce = fft_whole - 1;
        for (int i = cs; i <= ce; i++) {
            double mag = mag_total[i] / 3.0;
            if (mag > best_mag) {
                best_mag = mag; best_lag = i; best_f = f;
                memcpy(best_mag_grid, mag_total, fft_whole * sizeof(float));
            }
        }
    }

    /* ---- Stage 2: Metadata Calculation (Local to window) ---- */
    int cs_meta = (int)(round(coarse_time_sec * sample_rate) + padding) - sample_rate;
    int ce_meta = (int)(round(coarse_time_sec * sample_rate) + padding) + sample_rate;
    if (cs_meta < 0) cs_meta = 0; if (ce_meta >= fft_whole) ce_meta = fft_whole - 1;

    double sum_mag = 0, next_best = 0;
    int count = 0;
    for (int i = cs_meta; i <= ce_meta; i++) {
        float m = best_mag_grid[i] / 3.0f;
        sum_mag += m; count++;
        if (abs(i - best_lag) > n_spsym && m > next_best) next_best = m;
    }
    float avg_noise = (count > 0) ? (float)(sum_mag / count) : 1.0f;
    report->snr_refined = (avg_noise > 0) ? 10.0f * log10f((float)best_mag / avg_noise) : 0;
    // CONF = Primary / Secondary peak ratio in local window (Higher is better)
    report->sync_confidence = (next_best > 0) ? (float)best_mag / (float)next_best : 5.0f;

    report->freq_hz = best_f;
    // TOA is relative to the start of the signal buffer (removing padding)
    report->toa_ms = (float)(best_lag - padding) * 1000.0f / sample_rate;

    free(tpl_re); free(tpl_im); free(mag_total); free(best_mag_grid);
    fftwf_free(a_w); fftwf_free(A_w); fftwf_free(B_w); fftwf_free(C_w);
    fftwf_destroy_plan(pw_a); fftwf_destroy_plan(pw_b); fftwf_destroy_plan(pw_i);
    return 0;
}
