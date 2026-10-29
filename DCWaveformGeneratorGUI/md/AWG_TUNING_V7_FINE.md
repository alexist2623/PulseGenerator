# AWG Tuning v7: sample-position RAMP editor

## Editing a pulse

Enter a short transition directly in **Ramp [ns]**, either in the segment
table, the Add segment input, or the Trace Plot ramp editor. Fractional-clock
durations automatically enable v7 sample-position RAMP; no context-menu setup
is required. For example, `0.833333333 ns` produces a four-sample transition.
Positive times round up to the next 0.208333333 ns sample (at 300 MHz), so
`1 ns` becomes `1.041666667 ns`. Ordinary whole-clock durations keep their
legacy RAMP encoding. Existing files are not converted merely by opening them.

The table displays the first-edge transition duration. Its tooltip separately
reports the whole-clock command reservation. A short transition holds its
target through the unused part of that reservation; the following Flat
duration is unchanged. Simple transition edits are shared across output rows
with the existing shared timing controls; advanced per-output edge positions
remain independent.
If reducing the shared command reservation would truncate another output's
advanced edge positions, the edit is rejected and the original waveform is kept.

For offset edges or an up/down pulse, use the advanced editor:

In the AWG Tuning segment table, right-click a row after the initial SET and
select **Edit RAMP sample positions (v7)…**. Enable sample-position RAMP, then
enter the first edge's start and end. Optionally enable a second edge and set
its start, end, and the first edge's target voltage. The selected row's SET
voltage remains the final voltage after the second edge. Either edge can rise
or fall; the field names do not restrict its voltage direction.

Both sample indices and time in ns are editable. At the current 300 MHz fabric
clock and 16 samples per word, the grid is **0.208333333 ns per sample** and
**3.333333333 ns per word**. Entered times snap to this grid. This is digital
scheduling resolution, not a claim about measured analog rise time or jitter.

Commands reserve whole fabric words. Increase the ordinary RAMP duration
before using the advanced editor for an edge that will extend beyond its
current reservation. Editing the main duration of an advanced pulse changes
its first-edge length while retaining the other positions; an overlap with
the second edge is rejected without changing the waveform. The first edge starts at
lane 0…15; subsequent endpoints and the second edge may cross word boundaries.
Endpoints are inclusive: sample 2 to sample 7 contains five transition
intervals. Samples after the final endpoint hold the final voltage until the
following SET. No extra SET is inserted at a fractional time.

For example, reserve a 10 ns RAMP and set:

| Field | Sample | Time at 300 MHz |
| --- | ---: | ---: |
| First edge start | 2 | 0.416666667 ns |
| First edge end | 5 | 1.041666667 ns |
| Second edge start | 14 | 2.916666667 ns |
| Second edge end | 20 | 4.166666667 ns |

Set the first-edge target to 100 mV and the following SET to 20 mV to produce
an up/down pulse ending at 20 mV. The second edge crosses the first word
boundary. Save/load, copy, insertion/deletion, ideal preview, Python export,
compile caching, and execution-coordinate export preserve these settings.

## Driver and timing

The compiler uses the canonical `qick.ramp_v7.ramp_packet` encoder. This is the
extended form of the existing RAMP opcode, with two signed 35-bit Q18 slopes,
an optional-second-edge sentinel, and five exact 32-bit command words. It does
not introduce a FAST_RAMP opcode. Old firmware is rejected before execution.

The matching three-AWG image has AWG/RF pairs on front-panel DAC4, DAC5 and DAC6,
and Square Pulse on DAC7. DAC8…11 are digital zero outputs. AWG and RF share
the declared final 31-cycle latency; the v7 two-cycle alignment delay and
two-cycle output-sum latency are each included once. The GUI uses the queued
tProcessor command path with one command accepted per fabric cycle.

DC-area accounting follows the exact endpoint-clamped and quantized scalar
samples, including the final sample and word padding. RC range checks use that
ledger. DC/RC on/off combinations preserve the extended command words.

## Current limits

- Fine edge positions and voltages are fixed within a hardware sweep. HOLD
  duration sweeps, RF sweeps, and repetition counts remain available.
- Voltage and RAMP-duration hardware sweeps with fine RAMP are rejected. Their
  legacy register-add plan cannot update the new packed fields safely.
- Fine RAMP requires an identity cross-capacitance matrix. Mixing differently
  positioned edges can require more than the two edges this command supports.
- QCS export rejects fine RAMP. Use QICK v7 Python export or normal GUI Run.
- The ideal GUI preview is not an analog simulation. The legacy QickSim analog
  waveform model explicitly does not support v7; CPU command tests do.

## Validation

The dedicated tests cover every first/second start-lane combination, word
boundary crossings, one-interval full-scale 35-bit slopes, adjacent SET
timestamps, exact DC area, compensation combinations, save/load, preview,
generated Python, cache invalidation, and the offscreen editor. They also
check old-firmware rejection and unsupported sweep rejection.

On 2026-10-27, **795 unique selected offline GUI/compiler/driver tests passed**.
The dedicated v7 GUI suite passed 271 cases using the actual driver latency
and control metadata. GUI tests ran with an offscreen Qt platform and did not
open a visible GUI, contact a board, or operate measurement equipment. Evidence is stored outside Git in
`F:/QICK_Private_Analysis/awg_tuning_v7/gui_driver_20261027_1800`.

An unrelated legacy `test_capture_preview_matches_actual_tproc_commands`
DDR/DC timing discrepancy reproduced with the pre-change compiler snapshot.
It is not counted as passing and was not changed as part of the v7 editor work.

The subsequent direct-duration fix is covered by 424 selected regression tests,
including every sub-clock duration in both voltage directions, typed Add/table/
Trace inputs, shared timing, command generation, save/load and old-firmware
rejection. Its evidence is in
`F:/QICK_Private_Analysis/awg_tuning_v7/direct_ramp_duration_20261027_204601Z`.
