# SquarePulse hardware DDS

The QICK GUI recognizes `axis_square_pulse_v1` from the loaded firmware's
generator description. Use the matching QSTL_QICK Python library on both the
desktop and board/server. The new project is
`qstl_awg_tuning_fir_1msps_iq64_sq_pulse`: it replaces the fourth AWG in physical
DAC order, DAC7 / RFDC `s13_axis`, normally QICK generator 7. Identify the loaded
firmware instead of assuming a generator number on another design.

Use GUI branch `codex/square-pulse-dds-gui` together with the firmware/library
on `QSTL_QICK` branch `codex/1msps-square-pulse-dds`. The matching BIT and HWH
are in that firmware project directory and are extracted from the same XSA.

## Measurement setup

1. Identify QICK and open **AWG Tuning > SquarePulse**.
2. Enable SquarePulse and select its detected generator.
3. Enter frequency in MHz, peak amplitude in mV, and phase offset in degrees.
   Voltage conversion uses the existing Experiment output-full-scale setting.
   Amplitude is symmetric +/-peak, with a nominal 50% duty cycle.
4. Enable Hardware sweep for any of the three parameters and specify endpoints
   and point counts. Multiple axes form a Cartesian product with the existing
   AWG/RF sweeps. SquarePulse axes follow existing axes in frequency, amplitude,
   phase order; the last enabled axis varies fastest. Plot-axis selection does
   not change hardware loop nesting.
5. Configure the normal AWG sequence and readout, then run the Experiment.

The tProcessor loads exact frequency/phase/amplitude words from DMEM and advances
them in nested hardware loops. Increasing point count increases the data table,
not a list of unrolled pulse instructions or a host-side measurement loop.
The DDS continues during the AWG/readout sequence and between repetitions;
updates preserve its accumulated phase. It is muted at the end and also through
the board's AXI-Lite stop method on cancellation or an acquisition exception.
This does not replace the older standalone **QICK Square Wave** tab.

Frequency is quantized to a 32-bit increment at the actual scalar sample rate.
At 4.8 GSPS the step is approximately 1.1176 Hz. Phase is a 32-bit offset, not a
phase reset. Amplitude is quantized to the DAC's valid code grid (multiples of
four, maximum 32764). The GUI stores requested physical sweep coordinates and
the hardware executes the corresponding quantized words. A frequency of zero
holds the present phase-dependent DC sign. GUI sweep updates never clear phase.

The SquarePulse command is a short prelude before the first AWG timestamp in
each repetition. Its four-clock IP pipeline plus command transport is allowed
to settle before the AWG sequence begins. It does not reserve a physical AWG
output in the voltage matrix; assigning that same generator as an AWG or RF
output is rejected during compilation.

The program adds one initial 128-cycle command lookahead for each enabled new
feature (SquarePulse and external markers). The per-repetition SquarePulse
prelude advances the timeline by at least eight fabric cycles. Marker endpoints
advance the timeline without adding a blocking wait to every loop; the final
epilogue waits before reporting completion. These are scheduling allowances,
separate from FIR capture-delay correction.

## Output markers

**AWG Tuning > Experiment > Triggering** selects an external output pin, width,
and either the entire experiment or each repetition loop. Start, End and Start
and end are available. Width is rounded up to tProcessor clock cycles.

Start refers to the first AWG sequence timestamp, including any existing FIR
warm-up shift. It is a simultaneous digital event, not a pulse that must finish
before waveform generation. End follows readout, compensation and recovery.
The final acquisition counter is published only after the requested final
marker and SquarePulse mute have completed. A full-experiment marker applies
to one compiled hardware acquisition; separately compiled software-sweep
programs each have their own experiment boundary.

The current project exposes pin 0, `SPARE1_1V8` (tProcessor port 7, bit 6).
The existing GPIO IP is reused. DDR/readout bits sharing that port are merged
with start-marker events so overlapping pulses retain their individual widths,
including when a duration sweep changes the readout timestamp. Stop/cancellation
can interrupt a marker pulse; its programmed width applies to completed runs.

## Persistence and compatibility

SquarePulse settings, axis order and trigger settings are included in saved
JSON and generated Python. QCoDeS stores square frequency/amplitude/phase axes
in MHz/mV/degrees. Raw signed-int64 full-trace storage and mean-only storage
follow the existing measurement settings. Full-scale voltage conversion is not
applied a second time to SquarePulse coordinates when plotting/reloading.

Old firmware is detected without a SquarePulse IP and disables that feature.
Unchanged AWG, RF, ADC, CPMG and IQ-storage paths remain usable. Triggering also
works with old firmware that describes an external output pin. Enabled new
features are rejected with an explicit error if the loaded firmware or board
library does not support them.

Digital RTL and Python/GUI tests are documented with the firmware project's
validation results. Physical DAC voltage, analog edge shape, and board-level
marker alignment require an oscilloscope measurement on the loaded hardware.
