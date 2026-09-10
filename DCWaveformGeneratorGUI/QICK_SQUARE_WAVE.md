# QICK Square Wave

The additional **QICK Square Wave** tab continuously drives one
`axis_awg_tuning_v1` generator on a QICK tProcessor v1 board. It uses the QICK
connection and tProcessor clock configured in **Setup**, independently of the
selected QCS experiment backend. No readout, DDR capture, or RF tone is needed.

Enter frequency in Hz, peak amplitude in mV, waveform offset in mV, and high
level duty cycle. The requested levels are `offset + amplitude` and
`offset - amplitude`. The initial waveform is 40 kHz, +/-10 mV, 50% duty.

Select the measured **DC_Out** run from the existing QICK calibration database
(`gain_pwr_calb.db`). A blank DB path uses the database selected in the
**Calibration** tab. **Load / refresh** lists DC_Out runs and loads the chosen
run; **Auto** selects the newest DC_Out run covering the waveform frequency.
The loader is the same `CalibrationDatabase.output_calibration()` used by
the other QICK tabs, supporting both legacy notebook and GUI `gain/freq/pwr`
records. Calibration is reloaded at the actual clock-quantized frequency before
starting. Missing data, a wrong board type, or out-of-range frequency prevents
output; there is no fallback to a manual slope or frequency extrapolation.

The measured sine-tone response supplies the amplitude scale:

```
reference_peak_mV = 1000 * sqrt(2 * reference_ohm * 10**((response_dBm - 30)/10))
codes_per_mV = reference_gain / reference_peak_mV
DAC code = requested voltage in mV * codes_per_mv + zero_code
```

No slope is entered by the user. The existing calibration model estimates the
linear DAC gain response from the measured points, avoiding low-gain noise
floor artifacts. The dBm reference impedance defaults to 50 ohms and must
match the measurement convention; use data for the selected output path and
load. Sine-tone gain calibration at the waveform frequency does not determine
DC offset, DC transfer, harmonic response, or settled square-wave levels.
It is an amplitude-scale estimate, not complete square-wave predistortion.

The original **+1120-code offset compensation** is therefore retained as a
separate setting. Waveform offset in mV is also separate. QICK ADC DC-voltage
and QCS M5301A records are not applied to this QICK DAC. A 400..500 MHz run
cannot calibrate the default 40 kHz waveform; it needs data covering 0.04 MHz.

Each result is rounded once to the channel's DAC code quantum, using the
firmware's `dac_invalid_lsb`, and passed to QICK as a Python integer. Values
outside the DAC range and levels that quantize to the same code are rejected.
Frequency and duty are rounded to tProcessor cycles; the running status shows
the resulting frequency, duty and DAC codes. At 300 MHz, 40 kHz uses 7500
cycles per period and 3750 per level. The maximum supported frequency is
the tProcessor clock divided by 512; each level needs at least 16 DAC fabric
clocks. This leaves time for issuing the commands.

**Start continuous output** loads an infinite hardware branch. Both SETs use
explicit edge timestamps; no data-memory reload or cached Python pulse length
determines the next edge. A separate timed port paces instruction execution
without issuing a second DAC output. Waits are split into at most 256-clock
intervals so low frequencies do not leave the processor inside one long WAIT
while the v1 stop API temporarily overwrites program memory with END commands.
**Stop** calls `soc.stop_tproc()` without
`lazy=True`. This stops execution, but it does not command zero volts: queued
commands can finish and the DAC can retain its last level.

The connection remains on its worker thread during output. Other GUI hardware
tasks cannot start concurrently. A failed stop retains the worker and enables
retry; it does not report success. Closing the window requests Stop before
closing. Settings, including calibration DB/run and offset compensation, are
saved in the existing JSON settings file. Old manual-slope settings discard
that slope and require measured calibration on the next start.

Automated tests cover calibrated quantization, repeating command timestamps,
firmware/clock validation, start cancellation, stop/retry, and GUI settings
and close behavior. The event interpreter checks reference timestamps and
command words; it does not simulate analog DAC response or cycle-accurate RTL.
