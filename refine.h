#ifndef _INCLUDE_REFINE_H_
#define _INCLUDE_REFINE_H_

#include <stdint.h>
#include <stdbool.h>
#include "ft8/constants.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Holds high-precision TOA and FOA for a single decoded message.
typedef struct {
    char   message[25];
    double freq_hz;    ///< Refined frequency (0.01 Hz precision)
    double toa_ms;     ///< Refined TOA in ms from start of 15s window (0.001 ms precision)
    float  snr_refined; ///< Correlation peak-to-median ratio
    float  sync_confidence; ///< Ratio of primary peak to next highest non-adjacent peak
} precision_report_t;

/// Generate a real-valued GFSK template for the decoded message.
/// Output contains n_sym * n_spsym samples (no silence padding).
/// Caller must free() the returned pointer.
float *refine_generate_template(const uint8_t bits[10],
                                int n_sym,
                                float symbol_period,
                                float symbol_bt,
                                float freq_hz,
                                double sample_rate);

/// Refine TOA and FOA for one decoded message using matched filtering.
/// Returns 0 on success, -1 on error.
int refine_signal_params(const float *signal,
                         int signal_len,
                         double sample_rate,
                         const uint8_t bits[10],
                         const char *text,
                         bool is_ft8,
                         double coarse_freq_hz,
                         double coarse_time_sec,
                         int n_sym,
                         float symbol_period,
                         float symbol_bt,
                         precision_report_t *report);

void make_tpl_shared_double(const uint8_t *tones, int n_sym, double f0,
                            float symbol_bt, double symbol_period,
                            double sample_rate, double *out_re,
                            double *out_im, double *out_tone_sum_sq);
#ifdef __cplusplus
}
#endif

#endif // _INCLUDE_REFINE_H_
