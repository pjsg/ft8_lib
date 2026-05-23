#include "refine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft8/constants.h"
#include "ft8/encode.h"

#include <stdbool.h>

#include "common/debug.h"

#include "brent.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct brent_user_data {
  const float *signal;
  int signal_len;
  int sample_rate;
  const uint8_t *tones;
  int n_sym;
  float symbol_period;
  float symbol_bt;
  float start_t;
  float end_t;
  float step_t;
  double *tpl_re;
  double *tpl_im;
  double *tone_sum_sq;
  double best_f;
  double best_t;
  double best_mag;

} brent_user_data_t;

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// LUT Definitions: 4096 entries provides excellent speed and an 
// SFDR (Spurious-Free Dynamic Range) generally exceeding 70-80 dB.
#define LUT_BITS 12                  
#define LUT_SIZE (1 << LUT_BITS)
#define LUT_MASK (LUT_SIZE - 1)

// Interleaved lookup table: [cos0, sin0, cos1, sin1, ...]
// Total size: 4096 * 2 * 8 bytes = 64 KB (fits entirely in L1/L2 Cache)
static double g_sin_cos_lut[2 * LUT_SIZE];
static int g_lut_initialized = 0;

/**
 * Initializes the global Sine/Cosine Lookup Table.
 * Call this ONCE at program startup.
 */
void init_sin_cos_lut(void) {
    if (g_lut_initialized) return;
    for (int i = 0; i < LUT_SIZE; i++) {
        double angle = (2.0 * M_PI * i) / LUT_SIZE;
        g_sin_cos_lut[2 * i]     = cos(angle); // Cosine at even indices
        g_sin_cos_lut[2 * i + 1] = sin(angle); // Sine at odd indices
    }
    g_lut_initialized = 1;
}

/**
 * Optimized GMSK-like signal generator.
 * Eliminates out_tone_sum_sq calculation loops and swaps runtime 
 * trigonometric functions for a fast fixed-point DDS lookup table.
 */
void make_tpl_shared_double(const uint8_t *tones, int n_sym, double f0,
                            float symbol_bt, double symbol_period,
                            int sample_rate, double *out_re, double *out_im,
                            double *out_tone_sum_sq) {
  
  // Make sure the LUT is ready just in case it wasn't initialized at startup
  if (!g_lut_initialized) {
    init_sin_cos_lut();
  }

  int n_spsym = (int)(0.5f + sample_rate * symbol_period);
  int n_wave = n_sym * n_spsym;
  double dphi_peak = 2.0 * M_PI / n_spsym;

  // Allocate delta phase tracking array
  double *dphi = malloc((n_wave + 2 * n_spsym) * sizeof(double));
  double f_base = 2.0 * M_PI * (double)f0 / (double)sample_rate;
  for (int i = 0; i < n_wave + 2 * n_spsym; ++i) {
    dphi[i] = f_base;
  }

  // Precompute the Gaussian/Error Function pulse shape
  double *pulse = malloc(3 * n_spsym * sizeof(double));
  for (int i = 0; i < 3 * n_spsym; ++i) {
    double t = (double)i / (double)n_spsym - 1.5;
    pulse[i] = (erf(5.336446 * (double)symbol_bt * (t + 0.5)) -
                erf(5.336446 * (double)symbol_bt * (t - 0.5))) /
               2.0;
  }

  // Integrate data tones into phase transitions
  for (int i = 0; i < n_sym; ++i) {
    for (int j = 0; j < 3 * n_spsym; ++j) {
      dphi[j + i * n_spsym] += dphi_peak * (double)tones[i] * pulse[j];
    }
  }

  // Handle boundary ramp-up and ramp-down conditions
  for (int k = 0; k < 2 * n_spsym; ++k) {
    dphi[k] += dphi_peak * pulse[k + n_spsym] * (double)tones[0];
    dphi[k + n_wave] += dphi_peak * pulse[k] * (double)tones[n_sym - 1];
  }

  // Conversion factor from continuous radians to 32-bit unsigned fixed-point steps
  double rad_to_uint32 = 4294967296.0 / (2.0 * M_PI);
  const int shift_amt = 32 - LUT_BITS;

  // Modern CPUs loop much faster over clean integer steps.
  // Pre-convert our continuous phase jumps into precise 32-bit integer strides.
  uint32_t *dphi_steps = malloc(n_wave * sizeof(uint32_t));
  for (int i = 0; i < n_wave; ++i) {
    dphi_steps[i] = (uint32_t)(dphi[i + n_spsym] * rad_to_uint32);
  }

  // Initialize phase accumulator
  uint32_t phase = 0;

  // Primary Wave Synthesis Loop (Fully optimized table-lookups)
  for (int i = 0; i < n_wave; ++i) {
    // Drop lower precision bits to obtain table index
    uint32_t idx = phase >> shift_amt;

    // Direct memory read replacing heavy runtime sin/cos computations
    out_re[i] = g_sin_cos_lut[2 * idx];
    out_im[i] = g_sin_cos_lut[2 * idx + 1];

    // Accumulate the stride. Integer overflow automatically handles 2*pi wrapping.
    phase += dphi_steps[i];
  }

  // Identity Optimization: Since cos^2(x) + sin^2(x) = 1.0 everywhere,
  // the inner accumulation loop is mathematically hardcoded to equal n_spsym.
  for (int i = 0; i < 79; i++) {
    out_tone_sum_sq[i] = (double)n_spsym;
  }

  // Clean up all localized temporary buffers
  free(dphi_steps);
  free(dphi);
  free(pulse);
}

/* Standard GFSK Template Generator (Double Precision) */
void make_tpl_shared_double_old(const uint8_t *tones, int n_sym, double f0,
                                float symbol_bt, double symbol_period,
                                int sample_rate, double *out_re, double *out_im,
                                double *out_tone_sum_sq) {
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
    out_tone_sum_sq[i] = n_spsym;
  }

  free(dphi);
  free(pulse);
}

void make_tpl_shared_double_1(const uint8_t *tones, int n_sym, double f0,
                            float symbol_bt, double symbol_period,
                            int sample_rate, double *out_re, double *out_im,
                            double *out_tone_sum_sq) {
  int n_spsym = (int)(0.5f + sample_rate * symbol_period);
  int n_wave = n_sym * n_spsym;
  double dphi_peak = 2.0 * M_PI / n_spsym;

  double *dphi = malloc((n_wave + 2 * n_spsym) * sizeof(double));
  double f_base = 2.0 * M_PI * (double)f0 / (double)sample_rate;
  for (int i = 0; i < n_wave + 2 * n_spsym; ++i)
    dphi[i] = f_base;

  double *pulse = malloc(3 * n_spsym * sizeof(double));

  // Optimization: Keep an eye on compiler flags (-O3 -ffast-math)
  // to ensure erf() gets vectorized by the compiler here.
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

  // Optimization: Complex recurrence oscillator
  double cur_re = 1.0;
  double cur_im = 0.0;
  for (int i = 0; i < n_wave; ++i) {
    out_re[i] = cur_re;
    out_im[i] = cur_im;

    double step_re = cos(dphi[i + n_spsym]);
    double step_im = sin(dphi[i + n_spsym]);

    double next_re = cur_re * step_re - cur_im * step_im;
    double next_im = cur_re * step_im + cur_im * step_re;
    cur_re = next_re;
    cur_im = next_im;
  }

  // Optimization: Trigonometric Identity substitution (cos^2 + sin^2 = 1)
  for (int i = 0; i < 79; i++) {
    out_tone_sum_sq[i] = (double)n_spsym;
  }

  free(dphi);
  free(pulse);
}



/* Fast Correlation Scorer (Sync-Weighted) */
static double get_mag(const float *signal, int signal_len, int sample_rate,
                      int n_start, const double *tpl_re, const double *tpl_im,
                      const double *tone_sum_sq, int n_spsym,
                      bool costas_only) {
  double block_re = 0, block_im = 0, block_tpl_e = 0;
  if (costas_only) {
    for (int block = 0; block < 3; block++) {
      int t_start = ((block == 0) ? 0 : (block == 1) ? 36 : 72);
      int b_start = t_start * n_spsym;
      int b_len = 7 * n_spsym;

      // See if we can use a fast path
      if (n_start + b_start < 0 || n_start + b_start + b_len >= signal_len) {
        for (int i = 0; i < b_len; i++) {
          int idx = n_start + b_start + i;
          if (idx < 0 || idx >= signal_len)
            continue;
          block_re += signal[idx] * tpl_re[b_start + i];
          block_im += signal[idx] * tpl_im[b_start + i];
          block_tpl_e += tpl_re[b_start + i] * tpl_re[b_start + i] +
                         tpl_im[b_start + i] * tpl_im[b_start + i];
        }
      } else {
        // Fast path
        for (int i = 0; i < b_len; i++) {
          int idx = n_start + b_start + i;
          block_re += signal[idx] * tpl_re[b_start + i];
          block_im += signal[idx] * tpl_im[b_start + i];
        }
        for (int i = 0; i < 7; i++) {
          block_tpl_e += tone_sum_sq[i + t_start];
        }
      }
    }
  } else {
/*     int n_wave = 79 * n_spsym;
    // See if we can use a fast path
    if (n_start < 0 || n_start + n_wave >= signal_len) {
      // Slow path
      for (int i = 0; i < n_wave; i++) {
        int idx = n_start + i;
        if (idx < 0 || idx >= signal_len)
          continue;
        block_re += signal[idx] * tpl_re[i];
        block_im += signal[idx] * tpl_im[i];
        block_tpl_e += tpl_re[i] * tpl_re[i] + tpl_im[i] * tpl_im[i];
      }
    } else {
      // Fast path
      for (int i = 0; i < n_wave; i++) {
        int idx = n_start + i;
        block_re += signal[idx] * tpl_re[i];
        block_im += signal[idx] * tpl_im[i];
      }
      for (int i = 0; i < 79; i++) {
        block_tpl_e += tone_sum_sq[i];
      }
    } */
    int n_wave = 79 * n_spsym;
    // Fast path 
    // Create a direct pointer to the slice of the signal we care about.
    // This allows the compiler to see two linear, perfectly aligned array strides.
    const float *restrict sig_ptr = &signal[n_start];
    
    double local_re = 0.0;
    double local_im = 0.0;

    // Hinting to the compiler to unroll and vectorize this loop natively
    #pragma omp simd reduction(+:local_re, local_im)
    for (int i = 0; i < n_wave; i++) {
      double s = (double)sig_ptr[i];
      local_re += s * tpl_re[i];
      local_im += s * tpl_im[i];
    }
    
    block_re = local_re;
    block_im = local_im;
    
    // Mathematical Optimization: Deletes the 79-iteration loop entirely.
    // Summing tone_sum_sq 79 times is strictly equal to 79 * n_spsym (n_wave)
    block_tpl_e = (double)n_wave;
  }
    
  double total_mag = 0;
  if (block_tpl_e > 0) {
    total_mag = sqrt((block_re * block_re + block_im * block_im) / block_tpl_e);
  }
  return total_mag;
}

double get_mag_brent(double f, void *user_data) {
  brent_user_data_t *ud = (brent_user_data_t *)user_data;

  double best_mag = -1;
  double best_t = 0;
  int n_spsym = (int)(0.5f + ud->sample_rate * ud->symbol_period);

  make_tpl_shared_double(ud->tones, ud->n_sym, f, ud->symbol_bt,
                         ud->symbol_period, ud->sample_rate, ud->tpl_re,
                         ud->tpl_im, ud->tone_sum_sq);
  for (float t = ud->start_t; t <= ud->end_t; t += ud->step_t) {
    int n_start = (int)round(t * ud->sample_rate);
    double mag =
        get_mag(ud->signal, ud->signal_len, ud->sample_rate, n_start,
                ud->tpl_re, ud->tpl_im, ud->tone_sum_sq, n_spsym, false);
    if (mag > best_mag) {
      best_mag = mag;
      best_t = t;
    }
  }
#ifdef LOGIT
  printf("get_mag_brent: freq %f: mag %f\n", f, best_mag);
  fflush(stdout);
#endif
  if (best_mag > ud->best_mag) {
    ud->best_mag = best_mag;
    ud->best_t = best_t;
    ud->best_mag = best_mag;
  }
  return -best_mag;
}

int refine_signal_params(const float *signal, int signal_len, int sample_rate,
                         const uint8_t *payload, const char *text,
                         double coarse_freq_hz, double coarse_time_sec,
                         int n_sym, float symbol_period, float symbol_bt,
                         precision_report_t *report) {
  uint8_t tones[FT8_NN];
  ft8_encode(payload, tones);
  int n_spsym = (int)(0.5f + sample_rate * symbol_period);
  int tpl_len = n_sym * n_spsym;
  double *tpl_re = malloc(tpl_len * sizeof(double)),
         *tpl_im = malloc(tpl_len * sizeof(double));
  double *tone_sum_sq = malloc(79 * sizeof(double));

  double best_mag = -1.0;
  double best_f = coarse_freq_hz;
  double best_t = coarse_time_sec;

#ifdef LOGIT
  printf("Starting refine for '%s' at %f with time %f (sample rate: %d)\n",
         text, coarse_freq_hz, coarse_time_sec, sample_rate);
  fflush(stdout);
#endif

  /* Hierarchical Search */

  // 1. Frequency Scan (+/- 4Hz)
  for (double f = coarse_freq_hz - 3.0f; f <= coarse_freq_hz + 2.5f;
       f += 0.15f) {
    make_tpl_shared_double(tones, n_sym, f, symbol_bt, symbol_period,
                           sample_rate, tpl_re, tpl_im, tone_sum_sq);
    for (float t = coarse_time_sec - 0.05f; t <= coarse_time_sec + 0.03f;
         t += 0.005f) {
      int n_start = (int)round(t * sample_rate);
      double mag = get_mag(signal, signal_len, sample_rate, n_start, tpl_re,
                           tpl_im, tone_sum_sq, n_spsym, false);
      // printf("1: f: %f, t: %f, mag: %f\n", f, t, mag);
      if (mag > best_mag) {
        best_mag = mag;
        best_f = f;
        best_t = t;
      }
    }
  }

#ifdef LOGIT
  printf("After phase 1: best frequency %fHz, time %f, mag %f\n", best_f,
         best_t, best_mag);
  fflush(stdout);
#endif

  double best_brent_f = 0;
  brent_user_data_t ud = {0};
  ud.signal = signal;
  ud.signal_len = signal_len;
  ud.sample_rate = sample_rate;
  ud.tones = tones;
  ud.n_sym = n_sym;
  ud.symbol_period = symbol_period;
  ud.symbol_bt = symbol_bt;
  ud.start_t = best_t - 0.01f;
  ud.end_t = best_t + 0.01f;
  ud.step_t = 0.0001f;
  ud.tpl_re = tpl_re;
  ud.tpl_im = tpl_im;
  ud.tone_sum_sq = tone_sum_sq;
  ud.best_mag = best_mag;

  double best_brent_mag =
      -brent_minimize(best_f - 0.25, best_f, best_f + 0.25, get_mag_brent, &ud,
                      0.001 / best_f, &best_brent_f);

  if (best_brent_mag < best_mag) {
    printf("******* Brent search failed to improve on best_mag %f\n", best_mag);
    fflush(stdout);
  }

  best_mag = ud.best_mag;
  double best_brent_t = ud.best_t;

#ifdef LOGIT
  printf(
      "Brent found best_f: %f (%f - %f), best_t: %f (%f - %f), best_mag: %f\n",
      best_brent_f, best_f - 0.3, best_f + 0.3, best_brent_t, ud.start_t,
      ud.end_t, best_mag);
  fflush(stdout);
#endif
#ifdef NOBRENT
  best_mag = -1;

  // 2. Fine Polish
  float start_f = best_f, start_t = best_t;
  for (float f = start_f - 0.10f; f <= start_f + 0.10f; f += 0.004f) {
    make_tpl_shared_double(tones, n_sym, f, symbol_bt, symbol_period,
                           sample_rate, tpl_re, tpl_im, tone_sum_sq);
    for (float t = start_t - 0.010f; t <= start_t + 0.010f; t += 0.0002f) {
      int n_start = (int)round(t * sample_rate);
      double mag = get_mag(signal, signal_len, sample_rate, n_start, tpl_re,
                           tpl_im, tone_sum_sq, n_spsym, false);
      // printf("2: f: %f, t: %f, mag: %f\n", f, t, mag);
      if (mag > best_mag) {
        best_mag = mag;
        best_f = f;
        best_t = t;
      }
    }
  }
#else
  // Use brent
  best_f = best_brent_f;
  best_t = best_brent_t;
#endif

  // 3. Metadata Calculation (Confidence and Local SNR)
  make_tpl_shared_double(tones, n_sym, best_f, symbol_bt, symbol_period,
                         sample_rate, tpl_re, tpl_im, tone_sum_sq);
  double next_best = 0, sum_mag = 0;
  int count = 0;
  for (float t = best_t - 0.3f; t <= best_t + 0.3f; t += 0.005f) {
    int n_start = (int)round(t * sample_rate);
    double mag = get_mag(signal, signal_len, sample_rate, n_start, tpl_re,
                         tpl_im, tone_sum_sq, n_spsym, false);
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

#ifdef LOGIT
  printf("Refinement complete. Final frequency: %f Hz, time: %f ms, mag: %f\n",
         best_f, best_t * 1000.0f, best_mag);
  fflush(stdout);
#endif

  free(tpl_re);
  free(tpl_im);
  free(tone_sum_sq);
  return 0;
}
