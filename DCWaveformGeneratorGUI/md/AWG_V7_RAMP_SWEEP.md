# AWG v7 RAMP duration sweep

## Direct duration and JSON boundary fixes (2026-10-30)

Normal duration entries through one fabric clock use sample intervals, including
16 intervals. An existing fine RAMP keeps its fine encoding when edited to a
whole-clock duration. Thus 15, 16 and 17 intervals have distinct terminal samples;
16 no longer silently switches to the legacy 16-sample/15-interval command.
The terminal sample must fit in the reserved words, so a 16-interval edge reserves
two words. Previously coarse, longer whole-clock entries retain their legacy
semantics for compatibility; this fix does not rewrite stored coarse waveforms.

Both modern and legacy GUI JSON loading restore independent copies of
`fine_ramps` and `fine_fabric_mhz`. Loading an older waveform without these fields
clears previously active fine settings. A general HOLD sweep following an aligned
fine RAMP uses the actual edge endpoint, including the exact 16-interval boundary,
rather than adding the unused part of its command reservation to the plateau.

The 2026-10-30 follow-up used saved/reloaded GUI configurations without re-entering
fine fields. All 480 RF+HOLD coordinates and the matching AWG-only coordinates
were measured on DAC5 / CH1 at 50 ohms with native 0.1 ns fine records. RF ran
from 1 to 480 samples; HOLD ran from 81 to 560 samples in the same loop. Both
increments were exactly one sample in decoded emitted commands; RF front/rear
gaps stayed 48 samples. The measured AWG-width slope was 0.208520 ns/point with
RF and 0.208483 ns/point without RF, versus 0.208333 ns requested. Individual
adjacent analog differences remain noise limited. Ordinary 15/16/17-interval
RAMPs were additionally acquired as separate normal Runs. Firmware was unchanged.
All evidence is outside Git under
`C:/Jeonghyun/QICK_Private_Analysis/scope_checks/2026-10-30_fine_ramp_fix_rf_hold_480/`.

## Sweep controls

In an AWG output's segment table, right-click a row after the initial SET,
then choose **RAMP rate sweep > Configure sweep...**. Start/stop durations
are entered in microseconds. The dialog and Experiment sweep table display
the applied DAC sample increment. At 300 MHz fabric / 4.8 GSPS, one sample
interval is 0.208333333 ns. For 30 increasing intervals, use start
`0.000208333333 us`, stop `0.00625 us`, and 30 points.

Fine duration grids preserve both quantized endpoints. Each point is rounded
independently from the linear grid between those endpoints. When the range
is not divisible by `count - 1`, adjacent increments alternate between two
integer sample counts; the dialog displays both. This avoids accumulated
rounding error at the last point. Ranges that would repeat a sample length
are rejected. Whole-clock legacy grids retain their previous behavior.

Fractional-clock ranges select the v7 packet path automatically. A segment
already using the v7 sample-position editor also uses that path. Whole-clock
legacy sweeps retain their existing command encoding. Unsupported firmware
is rejected before output. The public sequence API can explicitly request
`add_ramp_duration_sweep(..., sample_grid=True)`.

The swept value is the duration of the **first transition**, in either voltage
direction. With a second transition enabled in **Edit RAMP sample positions
(v7)...**, its start/end move by the same sample offset as the first endpoint.
This preserves the internal hold and second-transition duration. For example,
`rise_start=0, rise_end=1, fall_start=481, fall_end=482` with a 400 mV turn
voltage describes a one-sample rise, a 480-sample (100 ns) plateau and a
one-sample fall. Sweeping the rise through 30 samples keeps both other
durations fixed.

For a two-transition packet, the same row's context menu also provides
**Falling duration sweep (v7)** and **Internal hold sweep (v7)**. These use
the same sample-grid duration dialog. The hold is the plateau *inside* the
packet, rather than the separate whole-clock SET hold. Each component keeps
its own start/stop range. Select the rising and falling parameter checkboxes
in Experiment and click **Group sweep** to advance both in a single loop.
Group members share their point count. Removing a component from either tab
or deleting its source segment removes its group membership; re-adding it
creates an independent parameter.

The runtime/export API identifies components with `ramp_0_to_1` (first edge),
`ramp_0_to_1:fall`, and `ramp_0_to_1:hold`. Use these names with
`add_ramp_duration_sweep`. Falling/hold sweeps require a two-edge fine packet
on every output. Multiple components in the same packet compose their
lengths before encoding: `rise_end = rise_start + rise`,
`fall_start = rise_end + hold`, and `fall_end = fall_start + fall`.

Fine edges occupy a fixed whole-clock command window large enough for the
longest point (or the existing reservation, if larger). This window is
distinct from the swept transition time. Trailing samples hold the terminal
voltage. For a single transition followed by a separate SET, that padding
extends the plateau before the SET; use the combined two-transition packet
when an exact plateau duration must be maintained. Subsequent command/RF/
readout times remain referenced to the reserved window.

The tProcessor runs a real sweep loop. For a single independent axis, varying
packed command words are stored in DMEM tables. Multiple independent fine
components use separate length and step tables, and the tProcessor combines
them into the packet at emission (`qick_fine_packet_sweep.py`). Table storage
therefore scales with the sum of component counts, rather than their Cartesian
product. Step division is performed before execution. Grouped components use
their shared index. There is no host loop per point or sequence expansion
into one segment per point.
Grouped axes share their existing loop index and tables rewind on outer
loop boundaries. Table capacity is checked before execution. Waveform
previews and execution metadata use each point's updated edge positions.

Fine RAMP still requires fixed voltage endpoints and an identity virtual-gate
matrix. Voltage sweep support is unchanged. The encoder validates positive
sample intervals, the second-edge start limit and the total sample count.

Regression coverage includes increasing/decreasing duration, sample-word
boundaries, both voltage directions, nested/grouped loop reset, GUI text
entry, save/load coordinates, old-firmware rejection, and exact command
words emitted by the instruction-level tProcessor model.

The 2026-10-29 DAC5 verification exercised both sweeps at 1 through 30 samples,
with and without DC+RC compensation. Grouped edges used a 100 ns plateau and
100 ns DC time. The hold sweep used one-sample edges and 3.333333 ns DC time
(one fabric clock, nearest to its 3.229167 ns mean hold); RC tau was 200 us.
All four complete 30-point acquisitions, intended waveforms, fits, GUI settings,
assembly and exact RTL output records are archived outside Git under
`C:/Jeonghyun/QICK_Private_Analysis/scope_checks/2026-10-29_dac5_fine_hold_30/`.
The final analog comparison uses one calibrated step response and one gain/delay
per acquisition. It does not establish the much smaller RC correction separately
from oscilloscope noise; the complete RC datapath and resets are checked bit-for-bit
in production RTL using the same uploaded program.

The subsequent combined verification used 100 grouped rise/fall points from
0.2 to 100 ns, 300 independent internal-hold points from 1 to 300 ns, and two
repetitions per coordinate: 60,000 pulses per run. Quantized starts were
0.208333333 ns and 1.041666667 ns; the endpoints remained 100 and 300 ns.
The factored program used 800 runtime DMEM words. All 60,000 emitted packets
in each OFF/ON program matched an independent encoder, including paired
repetitions and all 99 outer-loop boundaries. The requested DAC5 current was
31 mA (readback 30.975 mA), the GUI voltage was 400 mV, DC time was 300 ns,
and RC tau was 200 us.

Each condition has a complete 250 ms scope record at 125 ns/sample, plus four
separately triggered 100 us records at 0.2 ns/sample covering the beginning,
an outer-loop boundary, the middle and the end. Each acquisition reran the
full program with rep=2. The fine-resolution comparisons contain 139 OFF and
113 ON pulses, including 124 complete repetition pairs in total. They do not
establish analog correctness for every one of the 60,000 pulses. A shared
empirical step response, one gain/delay per capture, and fixed GUI durations
gave median pulse residuals of 2.08 mV OFF and 4.02 mV ON. The RC correction
alone remains below single-shot noise. CH1 was measured at 50 ohms and restored
to 1 Mohm; CH2 and CH4 remained at 1 Mohm.

A separate 12-pulse GUI fixture spanning the duration endpoints exercised the
new packet assembly path through production tProcessor, TMUX, AWG v7 and final
output-sum RTL. All 203,600 valid final scalar samples matched the independent
integer reference, including DC compensation and IIR resets. This is a
representative RTL fixture, not a simulation of the full 60,000-pulse program.
Firmware and production RTL were unchanged. Raw measurements, intended
waveforms, fits, settings, assembly, RTL records and provenance are archived
outside Git under
`C:/Jeonghyun/QICK_Private_Analysis/scope_checks/2026-10-29_dac5_combined_100x300_rep2/`.
