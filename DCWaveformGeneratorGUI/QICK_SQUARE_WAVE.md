# QICK Square Wave

The additional **QICK Square Wave** tab continuously drives one
`axis_awg_tuning_v1` generator on a QICK tProcessor v1 board. It uses the QICK
connection and tProcessor clock configured in **Setup**. No readout, DDR
capture, calibration database, or RF tone is needed.

Enter frequency in Hz, peak amplitude in mV, waveform offset in mV, and high
level duty cycle. The requested levels are `offset + amplitude` and
`offset - amplitude`. The initial waveform is 40 kHz, +/-10 mV, 50% duty.

Set **Maximum output (+/- mV)** to the voltage corresponding to full DAC
scale, as in AWG Tuning. Its default is **800 mV**, meaning a nominal range
of -800 to +800 mV. This setting belongs to the square-wave tab and is saved
independently of the other experiment tabs. It does not depend on frequency.

```
requested voltage = waveform offset +/- peak amplitude
ideal DAC code = requested voltage / full_scale_mv * full_scale_code + zero_code
```

The voltage range is user configured; it is not a measured calibration.
The original **+1120-code DAC offset compensation** remains a separate
setting from the waveform offset in mV. The DAC scale comes from the loaded
firmware. For the usual signed 16-bit interface, `full_scale_code = 32768`.

The shared AWG Tuning `normalized_to_dac()` conversion rounds once to the
channel's DAC code quantum after adding offset compensation. Values are
passed to QICK as Python integers. Requested levels outside the voltage
range, compensated values outside the DAC range, and levels that quantize
to the same code are rejected. Positive full scale uses the largest legal
positive code, as in AWG Tuning. With +/-800 mV, +/-10 mV amplitude, +1120
offset codes, and two invalid low bits, High/Low are **1528 / 712**.
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
closing. Maximum output and offset compensation are saved in the existing
JSON settings file. Old square-wave calibration DB/run, dBm-reference and
manual-slope fields are ignored on load. Existing waveform and offset values
are preserved; files without `full_scale_mv` use the default +/-800 mV range.

Automated tests cover voltage-range conversion, repeating command timestamps,
firmware/clock validation, start cancellation, stop/retry, and GUI settings
and close behavior. The event interpreter checks reference timestamps and
command words; it does not simulate analog DAC response or cycle-accurate RTL.
