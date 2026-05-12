#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <sys/fcntl.h>
#include <unistd.h>
#include <stdbool.h>

#include "ft8/pack.h"
#include "ft8/encode.h"
#include "ft8/constants.h"
#include "common/wave.h"
#include "refine.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Global required by refine.c
bool Trace = true;

// FT8 constants are mostly in ft8/constants.h
// FT8_SYMBOL_PERIOD, FT8_NN are already defined there.
#define FT8_TONE_SPACING (1.0f / FT8_SYMBOL_PERIOD)

typedef struct {
    double min;
    double max;
    double step;
} range_t;

int parse_range(const char *str, range_t *range) {
    return sscanf(str, "%lf:%lf:%lf", &range->min, &range->max, &range->step) == 3;
}

static void gfsk_pulse_impl(int n_spsym, float symbol_bt, float *pulse) {
    for (int i = 0; i < 3 * n_spsym; ++i) {
        float t = (float)i / (float)n_spsym - 1.5f;
        float arg1 = 5.336446f * symbol_bt * (t + 0.5f);
        float arg2 = 5.336446f * symbol_bt * (t - 0.5f);
        pulse[i] = (erff(arg1) - erff(arg2)) / 2;
    }
}

static void make_gfsk_tpl(const uint8_t *tones, int n_sym, float f0, float symbol_bt, float symbol_period, int sample_rate, float *out_re, float *out_im) {
    int n_spsym = (int)(0.5f + sample_rate * symbol_period);
    int n_wave = n_sym * n_spsym;
    double dphi_peak = 2.0 * M_PI / n_spsym;

    double *dphi = malloc((n_wave + 2 * n_spsym) * sizeof(double));
    for (int i = 0; i < n_wave + 2 * n_spsym; ++i) dphi[i] = 2.0 * M_PI * (double)f0 / (double)sample_rate;

    double *pulse = malloc(3 * n_spsym * sizeof(double));
    for (int i = 0; i < 3 * n_spsym; ++i) {
        double t = (double)i / (double)n_spsym - 1.5;
        pulse[i] = (erf(5.336446 * (double)symbol_bt * (t + 0.5)) - erf(5.336446 * (double)symbol_bt * (t - 0.5))) / 2.0;
    }

    for (int i = 0; i < n_sym; ++i)
        for (int j = 0; j < 3 * n_spsym; ++j)
            dphi[j + i * n_spsym] += dphi_peak * (double)tones[i] * pulse[j];

    for (int j = 0; j < 2 * n_spsym; ++j) {
        dphi[j]               += dphi_peak * pulse[j + n_spsym] * (double)tones[0];
        dphi[j + n_wave]      += dphi_peak * pulse[j]           * (double)tones[n_sym - 1];
    }

    double phase = 0;
    for (int i = 0; i < n_wave; ++i) {
        out_re[i] = (float)cos(phase);
        out_im[i] = (float)sin(phase);
        phase += dphi[i + n_spsym];
    }
    free(dphi); free(pulse);
}

int main(int argc, char **argv) {
    char *message_text = NULL;
    range_t t_range = {0, 1.0, 0.01};
    range_t f_range = {1000.0, 1100.0, 1.0};
    float corr_len_s = 15.0f;
    float override_sample_rate = 0;
    char *wav_path = NULL;
    int use_refine = 0;

    int opt;
    while ((opt = getopt(argc, argv, "m:t:f:l:F:X")) != -1) {
        switch (opt) {
            case 'm': message_text = optarg; break;
            case 't': if (!parse_range(optarg, &t_range)) return 1; break;
            case 'f': if (!parse_range(optarg, &f_range)) return 1; break;
            case 'l': corr_len_s = atof(optarg); break;
            case 'F': override_sample_rate = atof(optarg); break;
            case 'X': use_refine = 1; break;
            default: fprintf(stderr, "Usage: %s -m <message> -t min:max:step -f min:max:step [-l len] [-F fs] [-X] <wav>\n", argv[0]); return 1;
        }
    }

    if (optind < argc) {
        wav_path = argv[optind];
    }

    if (!message_text || !wav_path) {
        fprintf(stderr, "Error: Message and WAV file are required.\n");
        return 1;
    }

    // Prepare output filename
    char *out_filename = malloc(strlen(message_text) + 32);
    strcpy(out_filename, message_text);
    for (int i = 0; out_filename[i]; i++) {
        if (out_filename[i] == ' ' || out_filename[i] == '/' || out_filename[i] == '<' || out_filename[i] == '>') out_filename[i] = '_';
    }
    strcat(out_filename, ".simple.jsonl");

    FILE *out_f = fopen(out_filename, "w");
    if (!out_f) {
        fprintf(stderr, "Error: Could not open output file %s\n", out_filename);
        return 1;
    }

    // 1. Load WAV file
    float *signal = NULL;
    int num_samples;
    int int_sample_rate;    
    double sample_rate;
    int wav_fd = open(wav_path, O_RDONLY);
    if (wav_fd < 0) {
        fprintf(stderr, "Error: Could not open WAV file %s\n", wav_path);
        return 1;
    }
    if (load_wav(&signal, &num_samples, &int_sample_rate, wav_path, wav_fd) != 0) {
        fprintf(stderr, "Error: Could not read WAV file %s\n", wav_path);
        close(wav_fd);
        return 1;
    }
    close(wav_fd);
    printf("Signal loaded: %d samples. First 5: %f %f %f %f %f\n", num_samples, signal[0], signal[1], signal[2], signal[3], signal[4]);
    sample_rate = (double)int_sample_rate;
    if (override_sample_rate > 0) sample_rate = (double)override_sample_rate;

    // 2. Generate FT8 tones
    uint8_t payload[10]; // 77 bits in 10 bytes
    if (pack77(message_text, payload) < 0) {
        fprintf(stderr, "Error: Could not pack message text '%s'\n", message_text);
        return 1;
    }

    printf("Payload hex: ");
    for(int i=0; i<10; i++) printf("%02x", payload[i]);
    printf("\n");

    uint8_t tones[FT8_NN];
    ft8_encode(payload, tones);
    
    printf("First 10 tones: ");
    for(int i=0; i<10; i++) printf("%d ", tones[i]);
    printf("\n");

    // 3. Brute force correlation
    int n_spsym = (int)(FT8_SYMBOL_PERIOD * sample_rate);
    int n_wave = FT8_NN * n_spsym;
    float *tpl_re = malloc(n_wave * sizeof(float));
    float *tpl_im = malloc(n_wave * sizeof(float));
    
    double best_mag = -1;
    double best_t = 0;
    double best_f = 0;
    bool self_tested = false;

    printf("Starting brute-force correlation (GFSK)...\n");
    for (double f_base = f_range.min; f_base <= f_range.max; f_base += f_range.step) {
        make_gfsk_tpl(tones, FT8_NN, (float)f_base, 2.0f, FT8_SYMBOL_PERIOD, (int)sample_rate, tpl_re, tpl_im);
        
        if (!self_tested) {
            double st_mag = 0;
            for (int block = 0; block < 3; block++) {
                int b_start = (block == 0) ? 0 : (block == 1) ? 26 : 52;
                int b_end   = (block == 0) ? 26 : (block == 1) ? 52 : 79;
                double br = 0, bi = 0;
                for (int i = b_start * n_spsym; i < b_end * n_spsym; i++) {
                    br += (double)tpl_re[i] * (double)tpl_re[i];
                    bi += (double)tpl_re[i] * (double)tpl_im[i];
                }
                st_mag += sqrt(br * br + bi * bi);
            }
            printf("Self-test Mag (should be ~0.5): %f\n", st_mag / n_wave);
            self_tested = true;
        }

        for (double t_offset = t_range.min; t_offset <= t_range.max; t_offset += t_range.step) {
            double total_mag = 0;
            int total_samples = 0;
            int n_start = (int)(round(t_offset * sample_rate));

            for (int block = 0; block < 3; block++) {
                int b_start = (block == 0) ? 0 : (block == 1) ? 26 : 52;
                int b_end   = (block == 0) ? 26 : (block == 1) ? 52 : 79;
                
                double block_re = 0;
                double block_im = 0;
                double block_tpl_e = 0;
                for (int i = b_start * n_spsym; i < b_end * n_spsym; i++) {
                    int idx = n_start + i;
                    if (idx < 0 || idx >= num_samples) continue;
                    
                    block_re += (double)signal[idx] * (double)tpl_re[i];
                    block_im += (double)signal[idx] * (double)tpl_im[i];
                    block_tpl_e += (double)tpl_re[i] * (double)tpl_re[i] + (double)tpl_im[i] * (double)tpl_im[i];
                    total_samples++;
                }
                if (block_tpl_e > 0) {
                    total_mag += sqrt((block_re * block_re + block_im * block_im) / block_tpl_e);
                }
            }

            if (total_samples > 0) {
                double mag = total_mag / 3.0; // Average magnitude across 3 blocks
                fprintf(out_f, "{\"time\": %.6f, \"frequency\": %.3f, \"value\": %.6e}\n", t_offset, f_base, mag);
                if (mag > best_mag) {
                    best_mag = mag;
                    best_t = t_offset;
                    best_f = f_base;
                }
            }
        }
    }
    free(tpl_re); free(tpl_im);    fclose(out_f);
    printf("Simple search complete. Output saved to %s\n", out_filename);
    printf("Simple Best Results: Time=%.6fs, Freq=%.3fHz, Mag=%.6e\n", best_t, best_f, best_mag);

    if (use_refine) {
        printf("\nInvoking refine_signal_params...\n");
        precision_report_t report = {0};
        int n_sym = FT8_NN;
        float sym_period = FT8_SYMBOL_PERIOD;
        float sym_bt = 2.0f;
        
        if (refine_signal_params(signal, num_samples, (int)sample_rate, payload, message_text, (float)best_f, (float)best_t, n_sym, sym_period, sym_bt, &report) == 0) {
            printf("\nFinal Comparison Results:\n");
            printf("Method      | Time (s)    | Freq (Hz) \n");
            printf("------------|-------------|-----------\n");
            printf("Simple      | %11.6f | %9.3f\n", best_t, best_f);
            printf("Refined     | %11.6f | %9.3f\n", report.toa_ms / 1000.0, report.freq_hz);
            printf("Difference  | %11.6f | %9.3f\n", (report.toa_ms / 1000.0) - best_t, report.freq_hz - best_f);

            // Audit: Check Refined coordinates with Simple math
            double r_t = report.toa_ms / 1000.0;
            double r_f = report.freq_hz;
            float *r_re = malloc(n_wave * sizeof(float));
            float *r_im = malloc(n_wave * sizeof(float));
            make_gfsk_tpl(tones, FT8_NN, (float)r_f, 2.0f, FT8_SYMBOL_PERIOD, (int)sample_rate, r_re, r_im);
            
            double audit_mag = 0;
            int audit_n = (int)(round(r_t * sample_rate));
            for (int block = 0; block < 3; block++) {
                int b_start = (block == 0) ? 0 : (block == 1) ? 26 : 52;
                int b_end   = (block == 0) ? 26 : (block == 1) ? 52 : 79;
                double br = 0, bi = 0, be = 0;
                for (int i = b_start * n_spsym; i < b_end * n_spsym; i++) {
                    int idx = audit_n + i;
                    if (idx >= 0 && idx < num_samples) {
                        br += (double)signal[idx] * (double)r_re[i];
                        bi += (double)signal[idx] * (double)r_im[i];
                        be += (double)r_re[i] * (double)r_re[i] + (double)r_im[i] * (double)r_im[i];
                    }
                }
                if (be > 0) audit_mag += sqrt((br*br + bi*bi)/be);
            }
            printf("\nAudit Results:\n");
            printf("Simple Mag at Simple Peak:  %e\n", best_mag);
            printf("Simple Mag at Refined Peak: %e\n", audit_mag / 3.0);
            free(r_re); free(r_im);
        } else {
            fprintf(stderr, "Error: refine_signal_params failed.\n");
        }
    }

    free(out_filename);
    free(signal);
    return 0;
}
