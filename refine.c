#include "refine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft8/constants.h"
#include "ft8/encode.h"

#include <stdbool.h>

#include "common/debug.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Standard GFSK Template Generator (Double Precision) */
static void make_tpl_shared_double(const uint8_t *tones, int n_sym, float f0,
                                   float symbol_bt, double symbol_period,
                                   int sample_rate, double *out_re,
                                   double *out_im, double *out_tone_sum_sq) {
  int n_spsym = (int)(0.5f + sample_rate * symbol_period);
  int n_wave = n_sym * n_spsym;
  double dphi_peak = 2.0 * M_PI / n_spsym;

  double *dphi = malloc((n_wave + 2 * n_spsym) * sizeof(double));
  double f_base = 2.0 * M_PI * (double)f0 / (double)sample_rate;
  for (int i = 0; i < n_wave + 2 * n_spsym; ++i)
    dphi[i] = f_base;

  double *pulse = malloc(3 * n_spsym * sizeof(double));
  for (int i = 0; i < 3 * n_spsym; ++i) {
    double t = (double)i / (double)n_spsym - 1.5;
    pulse[i] = (erf(5.336446 * (double)symbol_bt * (t + 0.5)) -
                erf(5.336446 * (double)symbol_bt * (t - 0.5))) /
               2.0;
  }

  for (int i = 0; i < n_sym; ++i)
    for (int j = 0; j < 3 * n_spsym; ++j)
      dphi[j + i * n_spsym] += dphi_peak * (double)tones[i] * pulse[j];

  for (int k = 0; k < 2 * n_spsym; ++k) {
    dphi[k] += dphi_peak * pulse[k + n_spsym] * (double)tones[0];
    dphi[k + n_wave] += dphi_peak * pulse[k] * (double)tones[n_sym - 1];
  }

  double phi = 0;
  for (int i = 0; i < n_wave; ++i) {
    out_re[i] = cos(phi);
    out_im[i] = sin(phi);
    phi += dphi[i + n_spsym];
  }

  // Calculate the magnitude of each tone
  for (int i = 0; i < 79; i++) {
    out_tone_sum_sq[i] = 0;
    for (int j = 0; j < n_spsym; j++) {
      out_tone_sum_sq[i] += out_re[i * n_spsym + j] * out_re[i * n_spsym + j] +
                            out_im[i * n_spsym + j] * out_im[i * n_spsym + j];
    }
  }

  free(dphi);
  free(pulse);
}

/* Fast Correlation Scorer (Sync-Weighted) */
static double get_mag(const float *signal, int signal_len, int sample_rate,
               int n_start, const double *tpl_re, const double *tpl_im, const double *tone_sum_sq,
               int n_spsym) {
  double total_mag = 0;
  for (int block = 0; block < 3; block++) {
    int t_start = ((block == 0) ? 0 : (block == 1) ? 36 : 72);
    int b_start = t_start * n_spsym;
    int b_len = 7 * n_spsym;
    double block_re = 0, block_im = 0, block_tpl_e = 0;
    // See if we can use a fast path
    if (n_start + b_start < 0 || n_start + b_start + b_len >= signal_len) {
      for (int i = 0; i < b_len; i++) {
        int idx = n_start + b_start + i;
        if (idx < 0 || idx >= signal_len)
          continue;
        block_re += (double)signal[idx] * tpl_re[b_start + i];
        block_im += (double)signal[idx] * tpl_im[b_start + i];
        block_tpl_e += tpl_re[b_start + i] * tpl_re[b_start + i] +
                       tpl_im[b_start + i] * tpl_im[b_start + i];
      }
    } else {
      // Fast path
      for (int i = 0; i < b_len; i++) {
        int idx = n_start + b_start + i;
        block_re += (double)signal[idx] * tpl_re[b_start + i];
        block_im += (double)signal[idx] * tpl_im[b_start + i];
     }
      for (int i = 0; i < 7; i++) {
        block_tpl_e += tone_sum_sq[i + t_start];
      }
    }
    if (block_tpl_e > 0) {
      total_mag +=
          sqrt((block_re * block_re + block_im * block_im) / block_tpl_e);
    }
  }
  return total_mag / 3.0;
}

int refine_signal_params(const float *signal, int signal_len, int sample_rate,
                         const uint8_t *payload, const char *text,
                         float coarse_freq_hz, float coarse_time_sec, int n_sym,
                         float symbol_period, float symbol_bt,
                         precision_report_t *report) {
  uint8_t tones[FT8_NN];
  ft8_encode(payload, tones);
  int n_spsym = (int)(0.5f + sample_rate * symbol_period);
  int tpl_len = n_sym * n_spsym;
  double *tpl_re = malloc(tpl_len * sizeof(double)),
         *tpl_im = malloc(tpl_len * sizeof(double));
  double *tone_sum_sq = malloc(79 * sizeof(double));

  double best_mag = -1.0;
  float best_f = coarse_freq_hz;
  float best_t = coarse_time_sec;

  /* Hierarchical Search */

  // 1. Frequency Scan (+/- 4Hz)
  for (float f = coarse_freq_hz - 4.0f; f <= coarse_freq_hz + 4.0f; f += 0.2f) {
    make_tpl_shared_double(tones, n_sym, f, symbol_bt, symbol_period,
                           sample_rate, tpl_re, tpl_im, tone_sum_sq);
    for (float t = coarse_time_sec - 0.5f; t <= coarse_time_sec + 0.5f;
         t += 0.005f) {
      int n_start = (int)round(t * sample_rate);
      double mag = get_mag(signal, signal_len, sample_rate, n_start, tpl_re,
                           tpl_im, tone_sum_sq, n_spsym);
      if (mag > best_mag) {
        best_mag = mag;
        best_f = f;
        best_t = t;
      }
    }
  }

  // 2. Fine Polish
  float start_f = best_f, start_t = best_t;
  for (float f = start_f - 0.1f; f <= start_f + 0.1f; f += 0.025f) {
    make_tpl_shared_double(tones, n_sym, f, symbol_bt, symbol_period,
                           sample_rate, tpl_re, tpl_im, tone_sum_sq);
    for (float t = start_t - 0.010f; t <= start_t + 0.010f; t += 0.0001f) {
      int n_start = (int)round(t * sample_rate);
      double mag = get_mag(signal, signal_len, sample_rate, n_start, tpl_re,
                           tpl_im, tone_sum_sq, n_spsym);
      if (mag > best_mag) {
        best_mag = mag;
        best_f = f;
        best_t = t;
      }
    }
  }

  // 3. Metadata Calculation (Confidence and Local SNR)
  make_tpl_shared_double(tones, n_sym, best_f, symbol_bt, symbol_period,
                         sample_rate, tpl_re, tpl_im, tone_sum_sq);
  double next_best = 0, sum_mag = 0;
  int count = 0;
  for (float t = best_t - 0.5f; t <= best_t + 0.5f; t += 0.005f) {
    int n_start = (int)round(t * sample_rate);
    double mag = get_mag(signal, signal_len, sample_rate, n_start, tpl_re,
                         tpl_im, tone_sum_sq, n_spsym);
    sum_mag += mag;
    count++;
    // Secondary peak must be at least one symbol away to count as a "ghost"
    if (fabs(t - best_t) > symbol_period && mag > next_best)
      next_best = mag;
  }
  double avg_noise = (count > 0) ? (sum_mag / count) : 1.0;

  report->freq_hz = best_f;
  report->toa_ms = best_t * 1000.0f;
  report->snr_refined =
      (avg_noise > 0) ? 10.0f * log10((best_mag / avg_noise)) : -20.0f;
  report->sync_confidence =
      (next_best > 0) ? (float)(best_mag / next_best) : 5.0f;

  free(tpl_re);
  free(tpl_im);
  free(tone_sum_sq);
  return 0;
}
