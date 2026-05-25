#include "refine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft8/constants.h"
#include "ft8/encode.h"

#include <stdbool.h>

#include <sys/time.h>

#include "common/debug.h"

#include "brent.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct brent_user_data {
  const float *signal;
  int signal_len;
  double sample_rate;
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

static double *saved_pulse;
static int saved_n_spsym;
static double saved_symbol_bt;

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

#define GOLDEN_RATIO_RES 0.61803398874989484820

/**
 * Single-point frequency evaluator.
 * Generates a single target frequency template on the fly and correlates it.
 */
static double evaluate_single_freq(const float *restrict signal, int n_start, int n_wave,
                                   double sample_rate, double freq, 
                                   const float *restrict base_tpl_re,
                                   const float *restrict base_tpl_im) {
    
    double omega = 2.0 * M_PI * freq / (double)sample_rate;
    float block_re = 0.0f;
    float block_im = 0.0f;
    const float *restrict sig_ptr = &signal[n_start];

    // Fully vectorizable 3-stream loop
    #pragma omp simd reduction(+:block_re, block_im)
    for (int i = 0; i < n_wave; i++) {
        double phi = omega * (double)i;
        float c = (float)cos(phi);
        float s = (float)sin(phi);

        float mod_re = base_tpl_re[i] * c - base_tpl_im[i] * s;
        float mod_mod_im = base_tpl_re[i] * s + base_tpl_im[i] * c;

        block_re += sig_ptr[i] * mod_re;
        block_im += sig_ptr[i] * mod_mod_im;
    }

    return sqrt(((double)block_re * block_re + (double)block_im * block_im) / (double)n_wave);
}

static double evaluate_single_freq_with_time_track(const float *restrict signal, int n_start_base, 
                                                  int n_wave, double sample_rate, double freq, 
                                                  const float *restrict base_tpl_re,
                                                  const float *restrict base_tpl_im) {
    
    double omega = 2.0 * M_PI * freq / (double)sample_rate;
    double max_total_mag = 0.0;

    // Check a tight cluster of sample offsets (e.g., -1, 0, +1 samples)
    // to mimic Brent's ability to latch onto the exact peak of the time ridge.
    // If your time offset requires a wider look, expand this window (e.g., -4 to +4)
    for (int dt = -1; dt <= 1; dt++) {
        int n_start = n_start_base + dt;
        const float *restrict sig_ptr = &signal[n_start];

        float block_re = 0.0f;
        float block_im = 0.0f;

        #pragma omp simd reduction(+:block_re, block_im)
        for (int i = 0; i < n_wave; i++) {
            double phi = omega * (double)i;
            float c = (float)cos(phi);
            float s = (float)sin(phi);

            float mod_re = base_tpl_re[i] * c - base_tpl_im[i] * s;
            float mod_im = base_tpl_re[i] * s + base_tpl_im[i] * c;

            block_re += sig_ptr[i] * mod_re;
            block_im += sig_ptr[i] * mod_im;
        }

        double mag = sqrt(((double)block_re * block_re + (double)block_im * block_im) / (double)n_wave);
        if (mag > max_total_mag) {
            max_total_mag = mag;
        }
    }

    return max_total_mag;
}

/**
 * Squeezes the frequency down to millihertz precision using a Golden Section Search.
 * Replaces brent_minimize for the final frequency extraction.
 */
double refine_frequency_to_millihertz(const float *signal, int signal_len, double sample_rate,
                                      int n_wave, int n_start, double coarse_f, double grid_step,
                                      const float *base_tpl_re, const float *base_tpl_im, double *best_mag) {
    
    // Set up the bounding bracket based on your grid size (e.g., +/- 0.15 Hz)
    double a = coarse_f - grid_step;
    double b = coarse_f + grid_step;
    
    // Determine internal probe points based on the Golden Ratio split
    double k = GOLDEN_RATIO_RES * (b - a);
    double x1 = b - k;
    double x2 = a + k;
    
    double f1 = evaluate_single_freq_with_time_track(signal, n_start, n_wave, sample_rate, x1, base_tpl_re, base_tpl_im);
    double f2 = evaluate_single_freq_with_time_track(signal, n_start, n_wave, sample_rate, x2, base_tpl_re, base_tpl_im);
    
    // 11 iterations shrinks a 0.30 Hz window down to < 0.001 Hz (millihertz accuracy)
    for (int iter = 0; iter < 11; iter++) {
        if (f1 > f2) {
            b = x2;
            x2 = x1;
            f2 = f1;
            k = GOLDEN_RATIO_RES * (b - a);
            x1 = b - k;
            f1 = evaluate_single_freq_with_time_track(signal, n_start, n_wave, sample_rate, x1, base_tpl_re, base_tpl_im);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            k = GOLDEN_RATIO_RES * (b - a);
            x2 = a + k;
            f2 = evaluate_single_freq_with_time_track(signal, n_start, n_wave, sample_rate, x2, base_tpl_re, base_tpl_im);
        }
    }
    
    // Return the absolute center midpoint of the final sub-millihertz bracket
    if (f1 > f2) {
        *best_mag = f1;
    } else {
        *best_mag = f2;
    }
    return 0.5 * (a + b);
}

/**
 * Optimized GMSK-like signal generator.
 * Eliminates out_tone_sum_sq calculation loops and swaps runtime 
 * trigonometric functions for a fast fixed-point DDS lookup table.
 */
void make_tpl_shared_double(const uint8_t *tones, int n_sym, double f0,
                            float symbol_bt, double symbol_period,
                            double sample_rate, double *out_re, double *out_im,
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
  double *pulse = saved_pulse;

  if (!pulse || n_spsym != saved_n_spsym || symbol_bt != saved_symbol_bt) {
    if (pulse) {
      free(pulse);
    }
    pulse = malloc(3 * n_spsym * sizeof(double));
    for (int i = 0; i < 3 * n_spsym; ++i) {
      double t = (double)i / (double)n_spsym - 1.5;
      pulse[i] = (erf(5.336446 * (double)symbol_bt * (t + 0.5)) -
                  erf(5.336446 * (double)symbol_bt * (t - 0.5))) /
                2.0;
    }
    saved_pulse = pulse;
    saved_n_spsym = n_spsym;
    saved_symbol_bt = symbol_bt;
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
  //free(pulse);
}

/* Standard GFSK Template Generator (Double Precision) */
void make_tpl_shared_double_old(const uint8_t *tones, int n_sym, double f0,
                                float symbol_bt, double symbol_period,
                                double sample_rate, double *out_re, double *out_im,
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
                            double sample_rate, double *out_re, double *out_im,
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
double get_mag(const float *signal, int signal_len, double sample_rate,
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
    double local_re = 0.0;
    double local_im = 0.0;

    if (n_start < 0) {
      const float *restrict sig_ptr = signal;
      
      // Hinting to the compiler to unroll and vectorize this loop natively
      int i_limit = n_wave + n_start;
      #pragma omp simd reduction(+:local_re, local_im)
      for (int i = 0; i < i_limit; i++) {
        local_re += (double)signal[i] * tpl_re[i - n_start];
        local_im += (double)signal[i] * tpl_im[i - n_start];
      }
    } else {
      const float *restrict sig_ptr = &signal[n_start];
      
      // Hinting to the compiler to unroll and vectorize this loop natively
      int i_limit = n_wave;
      if (i_limit + n_start >= signal_len) {
        i_limit = signal_len - n_start;
      }
      #pragma omp simd reduction(+:local_re, local_im)
      for (int i = 0; i < i_limit; i++) {
        double s = (double)sig_ptr[i];
        local_re += s * tpl_re[i];
        local_im += s * tpl_im[i];
      }
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

#include <fftw3.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>

void optimize_search_grid(const float *restrict signal, int signal_len, double sample_rate,
                          int n_sym, float symbol_bt, double symbol_period, int n_spsym,
                          const uint8_t *tones, double coarse_freq_hz, float coarse_time_sec,
                          double *best_mag, double *best_f, double *best_t) {

  int n_wave = 79 * n_spsym;
  double block_tpl_e = (double)n_wave;

  // 1. Pre-calculate time steps to eliminate rounding overhead
  int t_steps = 0;
  for (float t = coarse_time_sec - 0.05f; t <= coarse_time_sec + 0.03f; t += 0.005f) t_steps++;

  int *n_start_arr = malloc(t_steps * sizeof(int));
  float *t_val_arr = malloc(t_steps * sizeof(float));
  int t_idx = 0;
  for (float t = coarse_time_sec - 0.05f; t <= coarse_time_sec + 0.03f; t += 0.005f) {
    t_val_arr[t_idx] = t;
    n_start_arr[t_idx++] = (int)round(t * sample_rate);
  }

  // 2. Pre-calculate frequency candidates
  int f_steps = 0;
  for (double f = coarse_freq_hz - 4.0f; f <= coarse_freq_hz + 3.5f; f += 0.15f) f_steps++;

  double *f_val_arr = malloc(f_steps * sizeof(double));
  int f_c = 0;
  for (double f = coarse_freq_hz - 4.0f; f <= coarse_freq_hz + 3.5f; f += 0.15f) {
    f_val_arr[f_c++] = f;
  }

  // 3. PRECOMPUTATION STRATEGY: 
  // Generate the actual full templates for every frequency upfront.
  // This takes ALL template creation math out of the search grid loops entirely.
  float **mod_tpl_re = malloc(f_steps * sizeof(float*));
  float **mod_tpl_im = malloc(f_steps * sizeof(float*));
  
  double *tpl_re_d = malloc(n_wave * sizeof(double));
  double *tpl_im_d = malloc(n_wave * sizeof(double));
  double *tone_sum_sq = malloc(79 * sizeof(double));

  for (int f_i = 0; f_i < f_steps; f_i++) {
    mod_tpl_re[f_i] = malloc(n_wave * sizeof(float));
    mod_tpl_im[f_i] = malloc(n_wave * sizeof(float));

    // Generate the raw template at the target frequency
    make_tpl_shared_double(tones, n_sym, f_val_arr[f_i], symbol_bt, symbol_period,
                           sample_rate, tpl_re_d, tpl_im_d, tone_sum_sq);

    // Cast straight to a sequential float buffer for maximum SIMD throughput
    #pragma omp simd
    for (int i = 0; i < n_wave; i++) {
      mod_tpl_re[f_i][i] = (float)tpl_re_d[i];
      mod_tpl_im[f_i][i] = (float)tpl_im_d[i];
    }
  }
  free(tpl_re_d);
  free(tpl_im_d);
  free(tone_sum_sq);

  // --- Initialize tracking variables before the loops ---
  int best_f_index = -1;
  int t_i_winner = -1;

  // 4. MAIN CROSS-CORRELATION GRID (Time -> Frequency -> Lean Dot Product)
  for (int t_i = 0; t_i < t_steps; t_i++) {
    int n_start = n_start_arr[t_i];
    float t = t_val_arr[t_i];

    if (n_start < 0 || n_start + n_wave >= signal_len) continue;

    const float *restrict sig_ptr = &signal[n_start];

    for (int f_i = 0; f_i < f_steps; f_i++) {
      double f = f_val_arr[f_i];
      
      // Contiguous float arrays map perfectly to modern CPU L1 caches
      const float *restrict t_re = mod_tpl_re[f_i];
      const float *restrict t_im = mod_tpl_im[f_i];

      float block_re = 0.0f;
      float block_im = 0.0f;

      // ULTRALIGHT HOT LOOP: 
      // Only 3 data streams are active. This is 100% vectorizable and cleanly 
      // fits into the CPU's high-speed data registers.
      #pragma omp simd reduction(+:block_re, block_im)
      for (int i = 0; i < n_wave; i++) {
        float s = sig_ptr[i];
        block_re += s * t_re[i];
        block_im += s * t_im[i];
      }

      double mag = sqrt(((double)block_re * block_re + (double)block_im * block_im) / block_tpl_e);

      if (mag > *best_mag) {
        *best_mag = mag;
        *best_f = f;
        *best_t = t;
        // Capture the index locations of the current winner
        best_f_index = f_i;
        t_i_winner = t_i;
      }
    }
  }

  // Ensure the winning frequency isn't on the absolute edge of your search limits
  if (best_f_index > 0 && best_f_index < f_steps - 1) {
#ifdef LOGIT
    printf("Interpolating frequency\n");
    fflush(stdout);
#endif    
    // 1. Run a lightweight evaluation for the left and right neighbors 
    // at the winning time step (t_i_winner) to get y1 and y3.
    const float *restrict sig_ptr = &signal[n_start_arr[t_i_winner]];
    
    // Left Neighbor
    float b_re1 = 0.0f, b_im1 = 0.0f;
    const float *restrict t_re1 = mod_tpl_re[best_f_index - 1];
    const float *restrict t_im1 = mod_tpl_im[best_f_index - 1];
    #pragma omp simd reduction(+:b_re1, b_im1)
    for (int i = 0; i < n_wave; i++) {
      b_re1 += sig_ptr[i] * t_re1[i];
      b_im1 += sig_ptr[i] * t_im1[i];
    }
    double y1 = sqrt(((double)b_re1*b_re1 + (double)b_im1*b_im1) / block_tpl_e);

    // Center Winner (We already know this value)
    double y2 = *best_mag; 

    // Right Neighbor
    float b_re3 = 0.0f, b_im3 = 0.0f;
    const float *restrict t_re3 = mod_tpl_re[best_f_index + 1];
    const float *restrict t_im3 = mod_tpl_im[best_f_index + 1];
    #pragma omp simd reduction(+:b_re3, b_im3)
    for (int i = 0; i < n_wave; i++) {
      b_re3 += sig_ptr[i] * t_re3[i];
      b_im3 += sig_ptr[i] * t_im3[i];
    }
    double y3 = sqrt(((double)b_re3*b_re3 + (double)b_im3*b_im3) / block_tpl_e);

    // 2. Apply Parabolic Interpolation Formula
    double denominator = y1 - (2.0 * y2) + y3;
#ifdef LOGIT
    printf("y1 = %f, y2 = %f, y3 = %f, denom = %f\n", y1, y2, y3, denominator);
    fflush(stdout);
#endif    
    if (fabs(denominator) > 1e-6) {
      double k = 0.5 * (y1 - y3) / denominator;
      
      // Adjust the starting frequency for Brent by the fractional step
      double freq_step_size = 0.15; 
      *best_f = f_val_arr[best_f_index] + (k * freq_step_size);
      
      // Update the magnitude estimate to reflect the predicted peak top
      *best_mag = y2 - 0.125 * (y1 - y3) * (y1 - y3) / denominator;
    }
  } else {
#ifdef LOGIT    
    printf("Frequency not interpolated: best_f_index = %d, f_steps = %d\n", best_f_index, f_steps);
    fflush(stdout);
#endif    
  }

  // 5. Memory Cleanup
  for (int f_i = 0; f_i < f_steps; f_i++) {
    free(mod_tpl_re[f_i]);
    free(mod_tpl_im[f_i]);
  }
  free(mod_tpl_re);
  free(mod_tpl_im);
  free(f_val_arr);
  free(n_start_arr);
  free(t_val_arr);
}

int refine_signal_params(const float *signal, int signal_len, double sample_rate,
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
  printf("Starting refine for '%s' at %f with time %f (sample rate: %f)\n",
         text, coarse_freq_hz, coarse_time_sec, sample_rate);
  fflush(stdout);
#endif

  /* Hierarchical Search */

  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);

#ifdef DO_OLD_COARSE_SCAN
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

  gettimeofday(&end_time, NULL);
  double coarse_scan_time = ((end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                      (end_time.tv_usec - start_time.tv_usec) / 1000.0) / 1000.0;
  printf("Coarse scan time: %f\n", coarse_scan_time);
  fflush(stdout);

#ifndef LOGIT
  printf("After coarse scan: best frequency %fHz, time %f, mag %f\n", best_f,
          best_t, best_mag);
  fflush(stdout);
#endif

  best_f = coarse_freq_hz;
  best_t = coarse_time_sec;
  best_mag = -1;

  gettimeofday(&start_time, NULL);
#endif

  optimize_search_grid(signal, signal_len, sample_rate, n_sym, symbol_bt,
                        symbol_period, n_spsym, tones, coarse_freq_hz,
                        coarse_time_sec, &best_mag, &best_f, &best_t);

  gettimeofday(&end_time, NULL);
  double optimize_time = ((end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                      (end_time.tv_usec - start_time.tv_usec) / 1000.0) / 1000.0;

#ifdef LOGIT
  printf("Optimize time: %f\n", optimize_time);
  fflush(stdout);
#endif

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
  ud.best_mag = 0;

  gettimeofday(&start_time, NULL);

  double best_brent_mag =
      -brent_minimize(best_f - 0.075, best_f, best_f + 0.075, get_mag_brent, &ud,
                      0.001 / best_f, &best_brent_f);

  gettimeofday(&end_time, NULL);
  double brent_time = ((end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                      (end_time.tv_usec - start_time.tv_usec) / 1000.0) / 1000.0;

#ifdef LOGIT
  printf("Brent time: %f\n", brent_time);
  fflush(stdout);

  printf("frequency updated from %fHz to %fHz by %fHz\n", best_f, best_brent_f, best_f - best_brent_f);
  fflush(stdout);
#endif

#ifdef USE_GOLDEN_SECTION
  // 2. Convert the winning time back into the absolute starting sample index
  int n_start = (int)round(best_t * sample_rate);
  int n_wave = 79 * n_spsym;

  // 3. Extract the clean 0-Hz base template to feed the solver
  float *base_tpl_re = malloc(n_wave * sizeof(float));
  float *base_tpl_im = malloc(n_wave * sizeof(float));

  gettimeofday(&start_time, NULL);

  make_tpl_shared_double(tones, n_sym, 0.0f, symbol_bt, symbol_period,
                         sample_rate, tpl_re, tpl_im, tone_sum_sq);
  for (int i = 0; i < n_wave; i++) {
    base_tpl_re[i] = tpl_re[i];
    base_tpl_im[i] = tpl_im[i];
  }
                         
  // ... (Populate base_tpl_re/im using make_tpl_shared_double at 0.0 Hz 
  // and downcast to float, exactly as we did in the setup step earlier) ...

  // 4. Pin down the frequency to the millihertz level instantly
  double grid_step_size = 0.15; // Matches the step size of your coarse grid

  double best_refine_mag = 0;
  double ultra_precise_freq = refine_frequency_to_millihertz(
      signal, signal_len, sample_rate, n_wave, n_start, 
      best_f, grid_step_size, base_tpl_re, base_tpl_im, &best_refine_mag
  );

  // Cleanup
  free(base_tpl_re);
  free(base_tpl_im);
 
  gettimeofday(&end_time, NULL);
  double refine_time = ((end_time.tv_sec - start_time.tv_sec) * 1000.0 +  
                       (end_time.tv_usec - start_time.tv_usec) / 1000.0) / 1000.0;
  printf("Refine frequency time: %f\n", refine_time);
  fflush(stdout);

  printf("Frequency updated from %fHz to %fHz by %fHz to mag %f from %f\n", best_f, ultra_precise_freq, best_f - ultra_precise_freq, best_refine_mag, best_mag);
  fflush(stdout);
#endif

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
