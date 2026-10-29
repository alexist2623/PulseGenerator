# RF pulse duration and hardware sweeps

With **Fine duration (DAC sample)** disabled, RF pulses in AWG Tuning and the shared experiment compiler select their output
mode from the generator fabric-clock length:

- A fixed pulse of 3 through 65,535 clocks uses a one-shot command with that
  length in the generator mode register.
- A duration sweep uses one-shot commands if every executed point fits that
  range. A compact DMEM table supplies the complete mode/length word at each
  hardware-loop point. Repetitions retain the same length; Cartesian axis
  transitions reset the table pointer.
- If any duration coordinate exceeds 65,535 clocks, the entire duration axis
  uses the existing periodic start plus timed zero-gain stop. This includes
  descending sweeps. With a count of one, only the start coordinate executes.
- Fixed pulses longer than 65,535 clocks also keep the periodic implementation.

The cutoff uses each RF generator's `f_fabric`, not the tProcessor frequency.
At 300 MHz, 65,535 clocks are 218.45 us. Short one-shot lengths can vary by one
fabric clock; they do not have the three-clock periodic stop quantization.
The retained long-pulse implementation accepts stop commands at three-clock
block boundaries, so its output can remain active for up to two extra clocks.

This change applies to the shared RF pulse compiler. Firmware and the separate
continuous-tone S-parameter acquisition path are unchanged.

Verification scripts live in QSTL_QICK under
`qick/firmware/ip/rc_precomp_validation/run_rf_duration_validation.py`.
Software tests cover fixed lengths, threshold crossings, descending sweeps,
20 x 20 Cartesian loops with two repetitions, and existing GUI/RC integrations.
The RTL batch checks both short-duration AWG timing modes on the full 20 x 20
grid, plus separate 2 x 2 threshold fixtures with zero AWG waveforms.

## Fine duration on v5

Fine duration uses the existing SGv6 envelope memory and v5 output path. At a
300 MHz fabric clock and 16 samples per clock, one sample is 0.208333... ns.
The requested duration is converted to a positive integer sample count. Values
on the GUI's serialized sample grid are snapped to that sample; other values
round upward. The minimum active length is one sample. A short envelope still
occupies at least three fabric clocks, so active duration and generator busy
duration are different quantities.

Sixteen reusable ending patterns occupy 2,048 envelope samples per RF channel.
They are shared by pulses on that channel. Short pulses select a suffix of a
pattern; longer pulses use a DDS plateau followed by the selected ending. Very
long plateaus use periodic mode and a tail at a three-clock boundary. Padding
remains zero and does not extend the intended active pulse.

For a fine duration sweep, the tProcessor now calculates `q = N >> 4` and
`r = N & 15`, command mode/length, pattern address, and tail time in registers.
The six-word-per-point duration table is no longer stored in DMEM. Three state
registers track the sample count and a two-limb, 60-bit fractional remainder.
The compiler checks every requested coordinate against the existing sample
quantizer before emitting the fixed-size loop. This preserves noninteger sample
steps and descending sweeps without accumulating a rounded-increment error.

Repetitions of one point keep the same duration. Advancing its independent or
shared sweep axis updates the state, and an outer Cartesian-loop transition
restores the initial state. The instruction sequence is not unrolled by point
count. Other features, including frequency/calibration and RAMP coefficient
tables, can still consume DMEM; fixed duration memory is not a guarantee that
every combined sweep fits the device.

Fine duration requires the standard SGv6 16-sample interface, matching generator
and tProcessor clocks, zero idle output, and absolute gain at most 16,383. A
fine hardware duration sweep keeps the anchor segment length fixed; configure a
sufficient HOLD explicitly. `require_within_segment` checks the padded busy
endpoint when enabled. Turning it off does not automatically extend the HOLD.
Shared-port command-spacing checks remain active.

The new runtime and RTL validation target v5. Older settings JSON files remain
loadable; a missing fine-duration option retains its previous disabled default.
See [sweep grouping](SWEEP_GROUPING.md) for synchronized parameter sweeps and
[hardware validation](validation/README.md) for the actual GUI Run acquisition
path and its required fixture evidence.

## Ramsey composite preset

Select **Composite pulse → Template → Ramsey** for `pi/2 → tau → pi/2`.
Tau is the free-evolution gap from the end of the first pulse to the start of
the second pulse. There is no refocusing pi pulse or N sweep. Both endpoint
pulses are always included; their checkboxes are locked while Ramsey is selected.

The **pi/2 duration** selector displays the chosen pi reference divided by two
(for example, `d0/2`). Both pulses share Global frequency and the Global gain
parameter, or the Global power parameter in dBm when calibration is selected.
Initial phase and Final phase use separate named phase parameters. Fine duration
uses the same sample quantization and busy-time restrictions described above.

The existing software tau sweep builds one program per tau coordinate. A sweep
of the pi reference updates both half-width endpoints together. Switching to
Custom preserves the two endpoint pulses, their half-width parameter binding,
and the current gap, and makes the endpoint choices independently editable.
