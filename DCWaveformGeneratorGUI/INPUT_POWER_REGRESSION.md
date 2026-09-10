# Input calibration: output power regression

In **Calibration → Input / ADC**, select **Sweep by → Output power (dBm,
gain regression)** and enter the start/end output power and point count.
These are DAC output connector powers. The existing external path loss is
subtracted when assigning power at the ADC input connector.

**Check output power / Update plot** reads the matching output calibration
from the selected QCoDeS database. The graph shows measured gain/power points,
the estimated power curve, and the selected calibration points. Choose a
frequency and a linear or logarithmic gain axis. Measured powers are adjusted
for the difference between the calibration and current output attenuators.
For an interpolated frequency, measurements from the two neighboring
frequencies are shown with their actual frequency labels.

The regression is `P_dBm = a(f)*20*log10(gain/32767) + b(f) - ATT_delta`.
It uses positive finite measured gains in the upper factor-of-eight range.
If that leaves fewer than three points, it uses up to eight highest gains.
This follows the existing high-gain selection to reduce noise-floor bias;
the new model fits the slope instead of assuming it is exactly one.
The graph reports slope, R² and RMS residual in dB. Coefficients are
interpolated within the calibration's measured frequency coverage.

Gain extrapolation displays a persistent warning and does **not** require a
confirmation or prevent calibration. A weak fit (R² below 0.98) or a two-point
fit also warns. An increasing finite regression requires at least two
distinct measured gains. Missing frequency coverage, incompatible RF paths,
unreachable hardware gain, and duplicate integer gain points remain errors.
An extrapolated estimate is not an independent power measurement.

The run recalculates gains on the actual DDS frequency grid and with the
commanded attenuator settings. Power mode uses the existing per-frequency
gain table in the tProcessor. FIR DDR and AVG acquisition both support it.
The actual integer DAC gain is saved in the existing `gain` QCoDeS column
for each frequency. `Calibration_Config.output_power_estimation` records
the model, fits, warnings, requested powers, applied gain matrix, and achieved
estimated powers after integer gain quantization. `meas_in_pwr` uses the
achieved estimate minus external path loss, not the unquantized target.

The original **DAC gain** sweep and its existing linear amplitude model
remain the default, including when loading old settings. In gain mode,
**Use measured gain/power regression** enables the new estimator without
changing the entered gains. Existing AWG and other power-calibration users
retain their original model.

Tests cover low-gain extrapolation, noise-floor exclusion, unavailable fits,
frequency interpolation, attenuation/path loss, power-to-gain rounding,
FIR/AVG program compilation and gain tables, QCoDeS round-trip compatibility,
plot data, warnings, and old/new settings. Hardware power accuracy still
requires measurements on the physical RF path.
