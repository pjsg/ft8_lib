#include "refine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fftw3.h>

#include "ft8/encode.h"
#include "ft8/constants.h"

#include "common/debug.h"

#define LOG_LEVEL LOG_INFO

/* ---------- GFSK constants (mirror gen_ft8.c) ---------- */
#define GFSK_CONST_K 5.336446f
#define PI_F         3.14159265358979323846f

/* ---------- GFSK pulse shaping ---------- */

static void gfsk_pulse_impl(int n_spsym, float symbol_bt, float *pulse)
{
    for (int i = 0; i < 3 * n_spsym; ++i) {
        float t = (float)i / (float)n_spsym - 1.5f;
        float arg1 = GFSK_CONST_K * symbol_bt * (t + 0.5f);
        float arg2 = GFSK_CONST_K * symbol_bt * (t - 0.5f);
        pulse[i] = (erff(arg1) - erff(arg2)) / 2;
    }
}

/* Build a coherent sine template (for final polish) */
static void make_sine_tpl(const uint8_t *tones,
                          int n_sym, float f0,
                          float symbol_bt, float symbol_period,
                          int sample_rate, float *out)
{
    int n_spsym = (int)(0.5f + sample_rate * symbol_period);
    int n_wave  = n_sym * n_spsym;
    float dphi_peak = 2 * PI_F / n_spsym;

    float *dphi = malloc((n_wave + 2 * n_spsym) * sizeof(float));
    for (int i = 0; i < n_wave + 2 * n_spsym; ++i) dphi[i] = 2 * PI_F * f0 / sample_rate;

    float *pulse = malloc(3 * n_spsym * sizeof(float));
    gfsk_pulse_impl(n_spsym, symbol_bt, pulse);

    for (int i = 0; i < n_sym; ++i)
        for (int j = 0; j < 3 * n_spsym; ++j)
            dphi[j + i * n_spsym] += dphi_peak * tones[i] * pulse[j];

    for (int j = 0; j < 2 * n_spsym; ++j) {
        dphi[j]               += dphi_peak * pulse[j + n_spsym] * tones[0];
        dphi[j + n_wave]      += dphi_peak * pulse[j]           * tones[n_sym - 1];
    }

    float phi = 0;
    for (int k = 0; k < n_wave; ++k) {
        out[k] = sinf(phi);
        phi = fmodf(phi + dphi[k + n_spsym], 2 * PI_F);
    }
    int n_ramp = n_spsym / 8;
    for (int i = 0; i < n_ramp; ++i) {
        float env = (1 - cosf(2 * PI_F * i / (2 * n_ramp))) / 2;
        out[i] *= env;
        out[n_wave - 1 - i] *= env;
    }
    free(dphi); free(pulse);
}

/* ---------- Public API ---------- */

float *refine_generate_template(const uint8_t bits[10],
                                int n_sym,
                                float symbol_period,
                                float symbol_bt,
                                float freq_hz,
                                int sample_rate)
{
    uint8_t tones[FT8_NN];
    ft8_encode(bits, tones);
    int n_spsym   = (int)(0.5f + sample_rate * symbol_period);
    int n_samples = n_sym * n_spsym;
    float *tpl = malloc(n_samples * sizeof(float));
    make_sine_tpl(tones, n_sym, freq_hz, symbol_bt, symbol_period, sample_rate, tpl);
    return tpl;
}

/* ---------- Main refinement function ---------- */

int refine_signal_params(const float *signal,
                         int signal_len,
                         int sample_rate,
                         const uint8_t bits[10],
                         const char *text,
                         float coarse_freq_hz,
                         float coarse_time_sec,
                         int n_sym,
                         float symbol_period,
                         float symbol_bt,
                         precision_report_t *report)
{
    int n_spsym = (int)(0.5f + sample_rate * symbol_period);
    int tpl_len = n_sym * n_spsym;
    uint8_t tones[FT8_NN];
    ft8_encode(bits, tones);

    /* ---- FAST Symbol-Synchronized Spectrogram Search ---- */
    int est_lag = (int)(coarse_time_sec * sample_rate);
    int padding = sample_rate;
    
    int fft_n = n_spsym; 
    fftwf_complex *in = fftwf_alloc_complex(fft_n);
    fftwf_complex *out = fftwf_alloc_complex(fft_n);
    fftwf_plan p = fftwf_plan_dft_1d(fft_n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    int step = sample_rate / 200; 
    int win_start = est_lag - (int)(0.300 * sample_rate);
    int win_end = est_lag + tpl_len + (int)(0.300 * sample_rate);
    if (win_start < 0) win_start = 0;
    if (win_end > signal_len) win_end = signal_len;
    
    int num_steps = (win_end - win_start) / step;
    float *spectro = calloc(num_steps * (fft_n / 2 + 1), sizeof(float));

    for (int i = 0; i < num_steps; i++) {
        int base = win_start + i * step;
        for (int j = 0; j < fft_n; j++) {
            if (base + j < signal_len) {
                in[j][0] = signal[base + j];
                in[j][1] = 0;
            } else {
                in[j][0] = 0; in[j][1] = 0;
            }
        }
        fftwf_execute(p);
        for (int j = 0; j <= fft_n / 2; j++) {
            spectro[i * (fft_n / 2 + 1) + j] = out[j][0]*out[j][0] + out[j][1]*out[j][1];
        }
    }

    float best_nc_f = coarse_freq_hz;
    int best_nc_lag = est_lag;
    double best_nc_val = -1e30;

    float df_bin = (float)sample_rate / fft_n;

    for (float f = coarse_freq_hz - 3.0f; f <= coarse_freq_hz + 3.0f; f += 0.25f) {
        for (int dt = -16; dt <= 16; dt++) { // +/- 80ms
            int t_start_abs = est_lag + dt * step;
            int t_start_idx = (t_start_abs - win_start) / step;
            double total_pwr = 0;
            for (int s = 0; s < n_sym; s++) {
                float f_sym = f + tones[s] * 6.25f;
                int bin = (int)roundf(f_sym / df_bin);
                if (bin < 0 || bin > fft_n / 2) continue;
                int s_idx = t_start_idx + (s * n_spsym / step);
                if (s_idx >= 0 && s_idx < num_steps) {
                    total_pwr += spectro[s_idx * (fft_n / 2 + 1) + bin];
                }
            }
            if (total_pwr > best_nc_val) {
                best_nc_val = total_pwr;
                best_nc_lag = t_start_abs;
                best_nc_f = f;
            }
        }
    }
    free(spectro); fftwf_free(in); fftwf_free(out); fftwf_destroy_plan(p);

    /* ---- Coherent Refinement (Polish) ---- */
    int osr = 16;
    int fft_whole = 1;
    while (fft_whole < signal_len + padding + tpl_len) fft_whole <<= 1;
    
    float *a_w = fftwf_malloc(fft_whole * sizeof(float));
    float *b_w = fftwf_malloc(fft_whole * sizeof(float));
    fftwf_complex *A_w = fftwf_alloc_complex(fft_whole / 2 + 1);
    fftwf_complex *B_w = fftwf_alloc_complex(fft_whole);
    fftwf_complex *C_w = fftwf_alloc_complex(fft_whole);
    fftwf_plan pw_a = fftwf_plan_dft_r2c_1d(fft_whole, a_w, A_w, FFTW_ESTIMATE);
    fftwf_plan pw_b = fftwf_plan_dft_r2c_1d(fft_whole, b_w, B_w, FFTW_ESTIMATE);
    fftwf_plan pw_i = fftwf_plan_dft_1d(fft_whole, B_w, C_w, FFTW_BACKWARD, FFTW_ESTIMATE);

    for(int i=0; i<padding; i++) a_w[i] = 0;
    for(int i=0; i<signal_len; i++) a_w[i+padding] = signal[i];
    for(int i=signal_len+padding; i<fft_whole; i++) a_w[i] = 0;
    fftwf_execute(pw_a);

    float *tpl_fine = malloc(tpl_len * sizeof(float));
    float best_f = best_nc_f; double best_f_val = -1e30;
    for (float f = best_nc_f - 0.5f; f <= best_nc_f + 0.5f; f += 0.05f) {
        make_sine_tpl(tones, n_sym, f, symbol_bt, symbol_period, sample_rate, tpl_fine);
        double e = 0; for (int i=0; i<tpl_len; i++) e += tpl_fine[i]*tpl_fine[i];
        double nrm = sqrt(e);
        for(int i=0; i<tpl_len; i++) b_w[i] = tpl_fine[i]/(float)nrm;
        for(int i=tpl_len; i<fft_whole; i++) b_w[i] = 0;
        fftwf_execute(pw_b);
        for (int k = 0; k <= fft_whole / 2; k++) {
            float re = A_w[k][0] * B_w[k][0] + A_w[k][1] * B_w[k][1];
            float im = A_w[k][1] * B_w[k][0] - A_w[k][0] * B_w[k][1];
            B_w[k][0] = re; B_w[k][1] = im;
        }
        for (int k = fft_whole / 2 + 1; k < fft_whole; k++) { B_w[k][0] = 0; B_w[k][1] = 0; }
        fftwf_execute(pw_i);
        int start = (best_nc_lag + padding) - 20, end = (best_nc_lag + padding) + 20;
        if(start<0) start=0; if(end>=fft_whole) end=fft_whole-1;
        for(int i=start; i<=end; i++) {
            double mag = sqrt(C_w[i][0]*C_w[i][0] + C_w[i][1]*C_w[i][1]);
            if(mag > best_f_val) { best_f_val = mag; best_f = f; }
        }
    }
    report->freq_hz = best_f;

    /* Final 16x Time Refinement */
    int fft_os = fft_whole * osr;
    fftwf_complex *B_os = fftwf_alloc_complex(fft_os);
    fftwf_complex *corr_os = fftwf_alloc_complex(fft_os);
    fftwf_plan pi_os = fftwf_plan_dft_1d(fft_os, B_os, corr_os, FFTW_BACKWARD, FFTW_ESTIMATE);

    make_sine_tpl(tones, n_sym, best_f, symbol_bt, symbol_period, sample_rate, tpl_fine);
    double e = 0; for (int i=0; i<tpl_len; i++) e += tpl_fine[i]*tpl_fine[i];
    double nrm = sqrt(e);
    for(int i=0; i<tpl_len; i++) b_w[i] = tpl_fine[i]/(float)nrm;
    for(int i=tpl_len; i<fft_whole; i++) b_w[i] = 0;
    fftwf_execute(pw_b);
    for (int k = 0; k <= fft_whole / 2; k++) {
        float re = A_w[k][0] * B_w[k][0] + A_w[k][1] * B_w[k][1];
        float im = A_w[k][1] * B_w[k][0] - A_w[k][0] * B_w[k][1];
        B_os[k][0] = re; B_os[k][1] = im;
    }
    for (int k = fft_whole / 2 + 1; k < fft_os; k++) { B_os[k][0] = 0; B_os[k][1] = 0; }
    fftwf_execute(pi_os);

    int best_lag_os = 0; double best_val_os = -1e30;
    int s_rs = (best_nc_lag + padding - 50)*osr, s_re = (best_nc_lag + padding + 50)*osr;
    if(s_rs<0) s_rs=0; if(s_re>=fft_os) s_re=fft_os-1;
    for(int i=s_rs; i<=s_re; i++) {
        double mag = sqrt(corr_os[i][0]*corr_os[i][0] + corr_os[i][1]*corr_os[i][1]);
        if(mag > best_val_os) { best_val_os = mag; best_lag_os = i; }
    }
    
    report->toa_ms = (best_lag_os / (float)osr - padding) * 1000.0 / sample_rate;

    /* Confidence Metrics: Noise floor estimation from non-peak region */
    double *sorted = malloc(fft_whole * sizeof(double));
    for (int i = 0; i < fft_whole; i++) sorted[i] = sqrt(C_w[i][0]*C_w[i][0] + C_w[i][1]*C_w[i][1]);
    for (int gap = fft_whole / 2; gap > 0; gap /= 2)
        for (int i = gap; i < fft_whole; ++i) {
            double t = sorted[i]; int j;
            for (j = i; j >= gap && sorted[j - gap] > t; j -= gap) sorted[j] = sorted[j - gap];
            sorted[j] = t;
        }
    double median = sorted[fft_whole / 2];
    
    /* CONF is the ratio of the peak to the second highest peak (in symbols) */
    double second_max = 0;
    for (int i = 0; i < fft_whole; i++) {
        if (abs(i - (best_nc_lag + padding)) < n_spsym) continue;
        double mag = sqrt(C_w[i][0]*C_w[i][0] + C_w[i][1]*C_w[i][1]);
        if (mag > second_max) second_max = mag;
    }
    
    report->sync_confidence = (second_max > 1e-12) ? (float)(best_f_val / second_max) : 10.0f;
    report->snr_refined = (median > 1e-12) ? (float)(best_f_val / median) : 0;

    free(sorted); free(tpl_fine); fftwf_free(a_w); fftwf_free(b_w); fftwf_free(A_w); fftwf_free(B_w); fftwf_free(C_w);
    fftwf_free(B_os); fftwf_free(corr_os);
    fftwf_destroy_plan(pw_a); fftwf_destroy_plan(pw_b); fftwf_destroy_plan(pw_i); fftwf_destroy_plan(pi_os);
    snprintf(report->message, sizeof(report->message), "%s", text);
    return 0;
}
