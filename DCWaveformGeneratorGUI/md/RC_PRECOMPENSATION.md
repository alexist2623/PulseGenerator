# DC and RC compensation

## AWG Tuning v3: measured digital area

Firmware `axis_awg_tuning_v3` reports `dc_area_version=1` and
`dc_area_bits=48`. With the matching QSTL_QICK Python library, the same DC
checkbox in AWG Tuning and Stability Diagram selects hardware compensation.
The GUI does not need a second DC control. Older v1/v2 firmware keeps the
software implementation described below.

The IP sums every effective DAC sample of the actual SET/RAMP output **before
the RC IIR**. Fixed-voltage mode emits the opposite voltage until the measured
area is exhausted, with at most one smaller final scalar sample. Fixed-time
mode uses two adjacent DAC codes over the requested number of fabric clocks
to cancel the integer area exactly. No FPGA divider is used: Python supplies
the fixed-time reciprocal. The engine sends the final zero edge through the
IIR before clearing both histories at the end of each shot.

This removes the legacy DC SET-to-SET minimum interval and the accumulation
of rounded compensation increments. It does **not** change the existing
constant integer voltage-sweep increment or eliminate its target-voltage
drift. It also does not sum the extra DAC ramp/offset area after the IIR.

Normal and bypass AWG output have 16 fabric clocks of latency (5 area-pipeline
clocks plus 11 IIR clocks). Compensation begins 17 clocks after its command at
the IIR input, and 28 clocks after that command at the DAC output. The compiler
uses these different paths for segment, RF, DDR, and compensation timing.
The FIR/DDR capture delay configured by the firmware remains unchanged.

The compiler checks overflow, compensation voltage and RC headroom, reserves
a conservative completion deadline, and waits for it before the next shot.
FIR capture may finish before or after DC according to the existing
`after_readout` / `overlap_readout` option. Both policies still wait for the
capture before advancing the shot. The timing preview includes the hardware
DC start/end. No per-point Cartesian voltage table is uploaded.

The following sections describe the legacy v1/v2 compensation path unless
explicitly stated otherwise. v3 still requires a validated matching BIT/HWH
and board-side Python update; recognizing the IP in the GUI alone does not
upgrade an installed bitstream.

AWG Tuning and Stability Diagram each have two independent checkboxes:

- **DC compensation** appends the existing opposite-area pulse, with the
  existing fixed-voltage or fixed-time behavior.
- **RC compensation** enables the stateful FPGA inverse high-pass and its
  tau setting (10 us through 1000 ms).

Both may be enabled simultaneously. Each hardware repetition ends the AWG
waveform at zero, completes any DC compensation, then clears the AWG IIR
history. This includes repetitions within one sweep point, sweep transitions,
and the final repetition. The existing recovery interval follows the reset.
There is no additional analog settling wait based on tau. Waveform previews show the requested post-RC voltage;
the actual compensated DAC voltage needs sufficient output headroom.

AWG Tuning and Stability now check that headroom before running. The shared
GUI builder and exported `build_sequence()` calculate the physical waveform
after cross-capacitance conversion, including SET, RAMP, DC compensation, and
the zero-input hold before reset. They evaluate `x + integral(x)/tau`, including
stationary points inside ramps. Out-of-range settings raise a visible error
with the output name, sweep corner, predicted voltage bounds, and allowed
range; acquisition does not start.

The compiler repeats the check using executed endpoint command words after
hardware sweep-step rounding, the quantized RC coefficient, and actual DC
compensation fields. It reserves conservative margins for DAC rounding,
RAMP step error, and timing quantization. Bounds near a rail may therefore be
rejected even when an ideal continuous model just fits. This is a preflight
estimate, not a substitute for RTL or analog circuit validation.

Voltage-range checks use only Cartesian corners: a 200 x 200 sweep checks
four points, not 40,000. They do not propagate through repetitions because
the AWG digital IIR is reset on every repetition. Existing compiler checks
for duration tables remain separate. SquarePulse retains its separate
continuous-history behavior; this AWG check does not model SquarePulse.

RF duration sweeps in `extend_by_rf_duration` mode also change the AWG hold
area. When combined with a voltage sweep, DC compensation now factors that
voltage-times-duration interaction into coefficient rows. The earlier of
the voltage/duration axes selects a row; the later axis uses its row-specific
increment. Both axis orders are supported without reordering the scan or
allocating a full point table. A 200 x 200 voltage/RF-duration example uses
600 coefficient words and still only four RC voltage-range checks. Fixed
AWG-length RF sweeps retain their previous compensation behavior.

This fixes the omitted cross term; finite DAC and register-increment
quantization still applies. In the 4 x 3 regression, fixed-time compensation
error falls from 88 to at most 4 DAC codes (one effective DAC LSB), and the
fixed-voltage duration error falls from 4614 to at most 2 Q8 tProcessor
duration units. These are test-case bounds, not universal sweep error limits.

Dual-AWG/RF duration sweeps can exhaust a register page. The coefficient
loader now also spills ordinary sweep state to DMEM to make room for its
pointer; it previously considered only coefficient-table state, even when
those fields were already in DMEM. Command values and scan order are unchanged.

Resetting the digital IIR does not discharge the physical RC capacitor, so
reset can introduce a decaying transient after the RC circuit. This removes
the accumulated digital baseline between shots; it does not make the actual
DAC waveform area zero or fix compensation-pulse quantization. The software
uses the existing firmware reset command; no new bitstream is required.

The previous software SET+RAMP flat-segment compensation implementation has
been removed. `FineTuneSequence.set_rc_compensation(tau_us)` configures the new
FPGA path; it does not modify the nominal SET/RAMP commands. Unsupported
firmware rejects enabled RC before output starts. DC and ordinary experiments
continue to work with old firmware.

Settings retain the existing `bias_t_compensation.enabled/type` fields for
file compatibility: `dc`, legacy `filter` (now FPGA RC), or `dc_rc` for both.
The sequence itself has separate `bias_t_compensation` and `rc_compensation`
objects. Old `filter` settings migrate to FPGA RC; the old algorithm is never
run. Saving, loading, code export, and Stability acquisition use the same path.

SquarePulse has its own RC checkbox/tau in both the AWG SquarePulse panel and
the standalone Square Wave tab. Amplitude hardware sweeps update the packed
amplitude and RC increment together using exact DMEM words. Updates preserve
accumulated phase and RC history. Dedicated SquarePulse firmware is required.

New firmware reports `rc_precomp_version=1`. Its output pipeline adds 11
fabric clocks whether RC is enabled or bypassed; physical AWG segment anchors,
RF timing and readout timing include that delay. Core command occupancy and
DC pulse widths remain unchanged. Square command settling uses the reported
15-clock latency. A firmware without the parameter uses legacy timing.

RC correction cannot remove DAC headroom limits or implement unlimited DC
through a capacitor. The inverse integrates a nonzero mean until clipping.
DAC clipping is saturating and observable in the firmware status. Physical
verification should use the actual RC time constant and a relaxed or otherwise
known initial capacitor state.
