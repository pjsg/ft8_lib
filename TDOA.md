## LLM Prompt: High-Precision FT8 TOA/FOA Refinement

**Context:**
You will modify `ft8_lib` (C/C++) to perform high-precision Time of Arrival (TOA) and Frequency of Arrival (FOA) estimation on all decoded FT8 signals. The goal is TDOA geolocation. Use `FFTW3` for transforms.

**System Requirements:**
* **Input:** Access to the full 15-second audio buffer (standard 12kHz sample rate) and the metadata from a successful `ft8_lib` decode (message string, coarse frequency, coarse DT).

**Task 1: The Refinement Function**
Create a C function `refine_signal_params` that:
1.  **Re-encodes the Message:** Takes the decoded FT8 message and generates the ideal 79-symbol complex baseband sequence (the "template").
2.  **Implements a Matched Filter:** * Heterodynes a segment of the raw IQ buffer to DC using the coarse frequency.
    * Performs a point-wise complex multiplication of the signal and the conjugate of the template.
    * Executes a large `FFTW` (e.g., $N=32768$) on the result to find the fine frequency offset ($\Delta f$).
3.  **Sub-sample TOA Estimation:** * Uses cross-correlation magnitude peaks to find the timing.
    * Applies **Parabolic Interpolation** on the three peak magnitude samples to achieve sub-sample (microsecond) resolution.
    * Formula for offset $\delta$: $\delta = 0.5 \times (y_{k-1} - y_{k+1}) / (y_{k-1} - 2y_k + y_{k+1})$.

**Task 2: Integration Logic**
1.  **Gating:** Inside the main decode loop, wrap the refinement call in a timer check: `if ((utc_seconds % 1200) < 60)`. 
2.  **Threading:** Ensure the refinement pass happens post-decode so it doesn't block the search for other signals.
3.  **Data Structure:** Define a `precision_report_t` struct containing:
    * `char message[25]`
    * `double freq_hz` (with $0.01\text{ Hz}$ precision)
    * `double toa_ms` (milliseconds from start of window, with $0.001\text{ ms}$ precision)
    * `float snr_refined` (based on correlation peak-to-sidelobe ratio)

**Constraints:**
* **Memory:** Reuse `FFTW` plans where possible to avoid allocation overhead during the 60-second burst.
* **Accuracy:** The target resolution is $<100\mu\text{s}$ and $<0.1\text{ Hz}$.

---

### Implementation Notes for Your Review

#### 1. Why 12kHz isn't enough (Interpolation is Key)
At 12,000 samples per second, one sample is **$83.33\mu\text{s}$**. Since you want millisecond precision (and ideally better for TDOA), simple peak finding is sufficient, but the parabolic interpolation mentioned in the prompt will get you down to $\approx 5\text{--}10\mu\text{s}$ if the SNR is decent. 

#### 2. Handling Multiple Signals
Since FT8 is often a "pile-up," the LLM needs to know to treat each decode as a separate candidate for refinement. The re-encoding step is what makes this powerful; by correlating against the *actual decoded message*, you are effectively "nulling" the other signals in the passband.

