#include "refine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fftw3.h>

#include "ft8/encode.h"
#include "ft8/constants.h"

#include "common/debug.h"

#define LOG_LEVEL LOG_WARN

/* ---------- GFSK constants (mirror gen_ft8.c) ---------- */
#define GFSK_CONST_K 5.336446f
#define PI_F         3.14159265358979323846f
#define PI_D         3.14159265358979323846

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

/* Build a sine GFSK template at frequency f0. */
static void make_sine_tpl(const uint8_t *tones,
                          int n_sym, float f0,
                          float symbol_bt, float symbol_period,
                          int sample_rate, float *out)
{
    int n_spsym = (int)(0.5f + sample_rate * symbol_period);
    int n_wave  = n_sym * n_spsym;
    float dphi_peak = 2 * PI_F / n_spsym;

    float *dphi = malloc((n_wave + 2 * n_spsym) * sizeof(float));
    for (int i = 0; i < n_wave + 2 * n_spsym; ++i)
        dphi[i] = 2 * PI_F * f0 / sample_rate;

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

    free(dphi);
    free(pulse);
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
    make_sine_tpl(tones, n_sym, freq_hz, symbol_bt, symbol_period,
                  sample_rate, tpl);
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

    LOG(LOG_INFO, "refine: tpl_len=%d, signal_len=%d, coarse_freq=%.2f, coarse_time=%.3f\n",
        tpl_len, signal_len, coarse_freq_hz, coarse_time_sec);

    /* FFT size for linear correlation */
    int fft_n = 1;
    while (fft_n < signal_len + tpl_len - 1)
        fft_n <<= 1;

    /* ---- Generate template at coarse frequency ---- */
    uint8_t tones[FT8_NN];
    ft8_encode(bits, tones);

    float *tpl = malloc(tpl_len * sizeof(float));
    make_sine_tpl(tones, n_sym, coarse_freq_hz, symbol_bt, symbol_period,
                  sample_rate, tpl);

    /* Normalize template */
    double tpl_energy = 0;
    for (int i = 0; i < tpl_len; ++i)
        tpl_energy += tpl[i] * tpl[i];
    double tpl_norm = sqrt(tpl_energy);
    for (int i = 0; i < tpl_len; ++i)
        tpl[i] /= (float)tpl_norm;

    /* ---- FFT plans (real-to-complex) ---- */
    double *a = fftw_malloc(fft_n * sizeof(double));
    double *b = fftw_malloc(fft_n * sizeof(double));
    fftw_complex *A = fftw_alloc_complex(fft_n / 2 + 1);
    fftw_complex *B = fftw_alloc_complex(fft_n / 2 + 1);
    double *corr = fftw_malloc(fft_n * sizeof(double));

    fftw_plan pa = fftw_plan_dft_r2c_1d(fft_n, a, A, FFTW_ESTIMATE);
    fftw_plan pb = fftw_plan_dft_r2c_1d(fft_n, b, B, FFTW_ESTIMATE);
    fftw_plan pi = fftw_plan_dft_c2r_1d(fft_n, B, corr, FFTW_ESTIMATE);

    int nfreq = fft_n / 2 + 1;

    /* FFT of signal */
    for (int i = 0; i < signal_len; ++i)
        a[i] = signal[i];
    for (int i = signal_len; i < fft_n; ++i)
        a[i] = 0;
    fftw_execute(pa);

    /* FFT of template */
    for (int i = 0; i < tpl_len; ++i)
        b[i] = tpl[i];
    for (int i = tpl_len; i < fft_n; ++i)
        b[i] = 0;
    fftw_execute(pb);

    /* Cross-correlation: IFFT(A * conj(B)) */
    for (int k = 0; k < nfreq; ++k) {
        double re = A[k][0] * B[k][0] + A[k][1] * B[k][1];
        double im = A[k][1] * B[k][0] - A[k][0] * B[k][1];
        B[k][0] = re;
        B[k][1] = im;
    }
    fftw_execute(pi);

    /* Find peak near the coarse time estimate (within +/-250 ms) */
    int npts = signal_len - tpl_len + 1;
    int est_lag = (int)(coarse_time_sec * sample_rate);
    int search_window = (int)(0.25 * sample_rate);  /* 250 ms */
    int rs = est_lag - search_window;
    int re = est_lag + search_window;
    if (rs < 0) rs = 0;
    if (re >= npts) re = npts - 1;

    int best_lag = rs;
    double best_val = -1e30;

    for (int i = rs; i <= re; ++i) {
        if (corr[i] > best_val) {
            best_val = corr[i];
            best_lag = i;
        }
    }

    /* Find secondary peak for confidence metric (must be at least 1/2 symbol away) */
    int min_dist = n_spsym / 2;
    double second_val = 0;
    for (int i = rs; i <= re; ++i) {
        if (abs(i - best_lag) < min_dist) continue;
        if (corr[i] > second_val) {
            second_val = corr[i];
        }
    }
    report->sync_confidence = (second_val > 1e-6) ? (float)(best_val / second_val) : 10.0f;

    LOG(LOG_INFO, "refine: search [%d,%d], peak lag=%d (%.3f ms), val=%.1f, conf=%.2f\n",
        rs, re, best_lag, best_lag * 1000.0 / sample_rate, best_val, report->sync_confidence);

    /* ---- Frequency sweep around current best lag ---- */
    {
        const int nf = 101;
        const double f_step = 0.1;
        double *peak_vals = malloc(nf * sizeof(double));

        for (int fi = 0; fi < nf; ++fi) {
            double df = (fi - nf / 2) * f_step;
            float f = coarse_freq_hz + df;

            float *tpl_f = malloc(tpl_len * sizeof(float));
            make_sine_tpl(tones, n_sym, f, symbol_bt, symbol_period,
                          sample_rate, tpl_f);

            /* Normalize */
            double e = 0;
            for (int i = 0; i < tpl_len; ++i)
                e += tpl_f[i] * tpl_f[i];
            double nrm = sqrt(e);
            for (int i = 0; i < tpl_len; ++i)
                b[i] = tpl_f[i] / (float)nrm;
            for (int i = tpl_len; i < fft_n; ++i)
                b[i] = 0;

            fftw_execute(pb);

            for (int k = 0; k < nfreq; ++k) {
                double re = A[k][0] * B[k][0] + A[k][1] * B[k][1];
                double im = A[k][1] * B[k][0] - A[k][0] * B[k][1];
                B[k][0] = re;
                B[k][1] = im;
            }
            fftw_execute(pi);

            /* Check only near our current best_lag */
            double pk = -1e30;
            int sweep_rs = best_lag - 10;
            int sweep_re = best_lag + 10;
            if (sweep_rs < 0) sweep_rs = 0;
            if (sweep_re >= npts) sweep_re = npts - 1;
            for (int i = sweep_rs; i <= sweep_re; ++i)
                if (corr[i] > pk)
                    pk = corr[i];
            peak_vals[fi] = pk;
            free(tpl_f);
        }

        /* Find best frequency */
        int best_fi = 0;
        for (int fi = 1; fi < nf; ++fi)
            if (peak_vals[fi] > peak_vals[best_fi])
                best_fi = fi;

        /* Parabolic interpolation */
        double fine_freq = coarse_freq_hz + (best_fi - nf / 2) * f_step;
        if (best_fi > 0 && best_fi < nf - 1) {
            double ym = peak_vals[best_fi - 1];
            double y0 = peak_vals[best_fi];
            double yp = peak_vals[best_fi + 1];
            double denom = ym - 2 * y0 + yp;
            if (fabs(denom) > 1e-12) {
                double d = 0.5 * (ym - yp) / denom;
                fine_freq += d * f_step;
            }
        }
        report->freq_hz = fine_freq;

        /* Final TOA refinement at best frequency */
        float *tpl_fine = malloc(tpl_len * sizeof(float));
        make_sine_tpl(tones, n_sym, (float)fine_freq, symbol_bt, symbol_period,
                      sample_rate, tpl_fine);
        double e = 0;
        for (int i = 0; i < tpl_len; ++i)
            e += tpl_fine[i] * tpl_fine[i];
        double nrm = sqrt(e);
        for (int i = 0; i < tpl_len; ++i)
            b[i] = tpl_fine[i] / (float)nrm;
        for (int i = tpl_len; i < fft_n; ++i)
            b[i] = 0;
        fftw_execute(pb);
        for (int k = 0; k < nfreq; ++k) {
            double re = A[k][0] * B[k][0] + A[k][1] * B[k][1];
            double im = A[k][1] * B[k][0] - A[k][0] * B[k][1];
            B[k][0] = re;
            B[k][1] = im;
        }
        fftw_execute(pi);

        int final_lag = rs;
        double final_val = -1e30;
        for (int i = rs; i <= re; ++i) {
            if (corr[i] > final_val) {
                final_val = corr[i];
                final_lag = i;
            }
        }

        double final_second_val = 0;
        for (int i = rs; i <= re; ++i) {
            if (abs(i - final_lag) < min_dist) continue;
            if (corr[i] > final_second_val) {
                final_second_val = corr[i];
            }
        }
        report->sync_confidence = (final_second_val > 1e-6) ? (float)(final_val / final_second_val) : 10.0f;
        
        /* Update best_val so SNR calculation works correctly later */
        best_val = final_val;

        /* Parabolic interpolation for sub-sample TOA */
        double delta = 0;
        if (final_lag > 0 && final_lag < npts - 1) {
            double ya = corr[final_lag - 1];
            double yc = corr[final_lag];
            double yb = corr[final_lag + 1];
            double denom = ya - 2 * yc + yb;
            if (fabs(denom) > 1e-12)
                delta = 0.5 * (ya - yb) / denom;
        }
        report->toa_ms = (final_lag + delta) * 1000.0 / sample_rate;
        free(tpl_fine);
        free(peak_vals);
    }

    /* SNR: peak / median of |corr| (approximate median using shell sort) */
    {
        double *sorted = malloc(npts * sizeof(double));
        for (int i = 0; i < npts; ++i)
            sorted[i] = fabs(corr[i]);
        for (int gap = npts / 2; gap > 0; gap /= 2)
            for (int i = gap; i < npts; ++i) {
                double tmp = sorted[i];
                int j;
                for (j = i; j >= gap && sorted[j - gap] > tmp; j -= gap)
                    sorted[j] = sorted[j - gap];
                sorted[j] = tmp;
            }
        double median = sorted[npts / 2];
        report->snr_refined = (median > 0) ? (float)(fabs(best_val) / median) : 0;
        free(sorted);
    }

    /* Fill report */
    snprintf(report->message, sizeof(report->message), "%s", text);

    LOG(LOG_INFO, "refine -> TOA=%.3f ms, freq=%.2f Hz, SNR=%.1f, CONF=%.2f\n",
        report->toa_ms, report->freq_hz, report->snr_refined, report->sync_confidence);

    /* Cleanup */
    free(tpl);
    fftw_free(a);
    fftw_free(b);
    fftw_free(A);
    fftw_free(B);
    fftw_free(corr);
    fftw_destroy_plan(pa);
    fftw_destroy_plan(pb);
    fftw_destroy_plan(pi);

    return 0;
}
