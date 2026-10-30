# Automatic RF duration and sample HOLD sweep

RF duration controls use the DAC sample grid automatically. The Fine duration
checkbox has been removed. Legacy JSON values (including false) load as automatic
fine duration; their requested timing values remain available in the editor.
RF delays and trigger timestamps retain their tProcessor clock grid.

The existing Applied step and Applied end columns are reused. General SET HOLD
and hardware RF duration axes quantize their endpoints to samples, round the
increment to the nearest integer sample (half away from zero), and calculate
`actual_end = first + (count - 1) * step`. No alternating fractional increments
are used for these axes. A descending sweep whose rounded final duration is
nonpositive is rejected. Equal RF endpoints can remain constant in a grouped loop.

At 4.8 GSPS, one sample is 0.208333333 ns. For example, HOLD 1 ns to 300 ns with
300 points becomes first=5 samples, step=5 samples, end=1500 samples (312.5 ns).
The requested end remains editable; the existing result columns show the change.
RAMP coefficient sweeps retain their existing endpoint-preserving sample grid
and DMEM coefficients.

## HOLD compilation

The v7 compiler keeps absolute sample boundaries and HOLD increments in registers
(or a constant number of spilled state words). Each point advances the HOLD state
with integer ADD. Word timestamps and lane positions use `>> 4` and `& 15`.

An adjacent rising/holding/falling sequence uses one two-edge packet while the
falling start, measured from the first packet's word origin, fits 14 bits
(0 through 16383). A runtime branch emits separate single-edge packets for a
longer HOLD. Each single-edge packet holds its terminal target until the next
command. There is no point-by-point HOLD waveform, packet, or duration table and
no PMEM unrolling. Multiple repetitions and nested/grouped loops rewind the state.
RAMP tables stay independent of a separate HOLD axis, including when HOLD is the
outer loop. CPU validation data and exported coordinates are not FPGA DMEM.

RF anchors and DDR readout/trigger times follow swept segment boundaries, rounded
up to their clock grid. RF width remains sample granular; RF start time is still
clock granular. DC/RC processing uses the emitted branch's commands. The final
compensation SET-zero remains on the next clock boundary. Execution-coordinate
exports identify optional falling packets with an emitted column and null values
at points where the combined packet replaces them.

## Supported scope and validation

Sample HOLD requires v7, 16 samples per word, matching AWG/tProcessor clocks,
and ordinary aligned single-transition RAMPs between SETs. Advanced two-edge
RAMP components retain their separate existing compiler path and cannot be
combined with a general fractional SET HOLD in this path. Unencodable command
word overlaps, shared-port ordering changes, signed-register overflow, and memory
overflow are rejected before output. Acquisition uses the DDR readout path.

Existing fine RF requirements still apply: SGv6 with 16 lanes, absolute gain at
most 16383, zero steady output, and fixed anchor-segment length for hardware RF
duration sweeps. Gain is never silently lowered. Fine RF can extend outside its
anchor when require_within_segment is disabled; automatic segment extension by a
fine hardware RF duration axis remains unsupported. Host-driven T2 duration
sweeps retain their separate host coordinate policy.

`test/test_sample_hold_sweep.py` executes tProcessor instructions in the strict
behavioral model and compares packet words to the independent v7 encoder. It
covers ascending/descending long HOLD, both branch paths, repetitions, grouped
and nested voltage loops, RAMP tables in either loop order, RF anchors, DDR with
both DC scheduling modes, RC compensation, command export, GUI summaries,
JSON/export round trips, and compiled-program cache replay.

The initial compiler update used offline behavioral checks. The 2026-10-30
follow-up also measured RF and HOLD grouped together at one-sample increments:
RF 0.208333333..100 ns, HOLD RF+16.666666667 ns, 480 points, rep=1, DAC5 / CH1
at 50 ohms, DC/RC OFF. Every Run executed the full hardware loop. Both RF+AWG
and separate AWG-only reference Runs have complete sweep records and native
0.1 ns coverage of every coordinate. The actual commands preserve one-sample
increments and both 10 ns gaps. Within-word measured width progression confirms
that HOLD does not round back to 3.333333333 ns steps. Raw traces, program binaries,
DMEM, saved/reloaded GUI JSON, QCoDeS DB, metrics and figures are archived outside
Git in `C:/Jeonghyun/QICK_Private_Analysis/scope_checks/2026-10-30_fine_ramp_fix_rf_hold_480/`.
No firmware or RTL changes were required.
