#include "refine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft8/encode.h"
#include "ft8/constants.h"

#include <stdbool.h>

#include "common/debug.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Standard GFSK Template Generator (Double Precision) */
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

/* Fast Correlation Scorer (Sync-Weighted) */
static double get_mag(const float *signal, int signal_len, int sample_rate, int n_start, float *tpl_re, float *tpl_im, int n_spsym) {
    double total_mag = 0;
    float weights[3] = {10.0f, 10.0f, 10.0f}; // Sync blocks
    float total_weight = 30.0f;

    for (int block = 0; block < 3; block++) {
        int b_start = ((block == 0) ? 0 : (block == 1) ? 36 : 72) * n_spsym;
        int b_len = 7 * n_spsym;
        
        double block_re = 0, block_im = 0, block_tpl_e = 0;
        for (int i = 0; i < b_len; i++) {
            int idx = n_start + b_start + i;
            if (idx < 0 || idx >= signal_len) continue;
            block_re += (double)signal[idx] * (double)tpl_re[b_start + i];
            block_im += (double)signal[idx] * (double)tpl_im[b_start + i];
            block_tpl_e += (double)tpl_re[b_start + i] * (double)tpl_re[b_start + i] + (double)tpl_im[b_start + i] * (double)tpl_im[b_start + i];
        }
        if (block_tpl_e > 0) {
            total_mag += weights[block] * sqrt((block_re * block_re + block_im * block_im) / block_tpl_e);
        }
    }
    return total_mag / total_weight;
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
    float *tpl_re = malloc(tpl_len * sizeof(float)), *tpl_im = malloc(tpl_len * sizeof(float));

    double best_mag = -1.0;
    float best_f = coarse_freq_hz;
    float best_t = coarse_time_sec;

    /* Hierarchical Search for Robustness and Speed */
    
    // 1. Frequency Scan (+/- 5Hz)
    for (float f = coarse_freq_hz - 5.0f; f <= coarse_freq_hz + 5.0f; f += 0.2f) {
        make_tpl_shared(tones, n_sym, f, symbol_bt, symbol_period, sample_rate, tpl_re, tpl_im);
        
        // Coarse Time Scan (+/- 0.5s in 5ms steps)
        for (float t = coarse_time_sec - 0.5f; t <= coarse_time_sec + 0.5f; t += 0.005f) {
            int n_start = (int)round(t * sample_rate);
            double mag = get_mag(signal, signal_len, sample_rate, n_start, tpl_re, tpl_im, n_spsym);
            if (mag > best_mag) {
                best_mag = mag; best_f = f; best_t = t;
            }
        }
    }

    // 2. Fine Polish (Time +/- 10ms in 0.1ms steps, Freq +/- 0.5Hz in 0.05Hz steps)
    float start_f = best_f, start_t = best_t;
    for (float f = start_f - 0.5f; f <= start_f + 0.5f; f += 0.05f) {
        make_tpl_shared(tones, n_sym, f, symbol_bt, symbol_period, sample_rate, tpl_re, tpl_im);
        for (float t = start_t - 0.010f; t <= start_t + 0.010f; t += 0.0001f) {
            int n_start = (int)round(t * sample_rate);
            double mag = get_mag(signal, signal_len, sample_rate, n_start, tpl_re, tpl_im, n_spsym);
            if (mag > best_mag) {
                best_mag = mag; best_f = f; best_t = t;
            }
        }
    }

    report->freq_hz = best_f;
    report->toa_ms = best_t * 1000.0f;
    report->snr_refined = (float)best_mag; // Simplified for stability
    report->sync_confidence = 1.0f;

    free(tpl_re); free(tpl_im);
    return 0;
}
