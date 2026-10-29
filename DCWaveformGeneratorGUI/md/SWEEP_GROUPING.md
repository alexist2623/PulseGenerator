# Sweep groups and execution order

In the Experiment sweep table, check at least two hardware parameters and click
**Group sweep**. Members share one loop index and one point count. Each member
keeps its own start, stop and units. Two grouped 100-point parameters execute
100 points; two independent 100-point groups execute 10,000 points. Repetitions
keep all coordinates unchanged until the next point.

When selected members have different counts, the first selected row in the
table supplies the common count. Editing any member's count updates its group.
Editing a range changes that member only. Group badges and light background
colors persist independently of checkbox selection. Wide target names and
requested steps remain readable through horizontal scrolling.

Use **Remove from group** in a member's context menu to make it independent.
It retains its range and last shared count. A remaining singleton becomes
independent too. Regrouping moves only the selected members. Deleting a source
parameter removes its membership; re-adding the same name or location creates a
new identity and does not restore the old linkage.

Hardware voltage, HOLD/RAMP duration, RF duration, RF frequency and RF power
parameters can be grouped. Host software-only and square-wave axes remain
independent and do not offer grouping checkboxes.

## Changing sweep order (v7 GUI)

Select a row and use **Move up** or **Move down**. The table shows the actual
nesting: the top independent axis/group is slowest (outer loop), and the bottom
is fastest (inner loop). Repetitions still finish at each coordinate before the
next coordinate. Moving a grouped member moves its entire group; members keep
one shared index. Header sorting is disabled so it cannot suggest a different
execution order. Grouping still takes the point count from the first checked
row in this displayed order. Order buttons are disabled while an experiment is
running and become available again when it finishes.

The same order reaches Run, waveform preview, Show QICK Program, Python export
and saved coordinates. Time axes can move across voltage/RF hardware axes.
When an outer voltage affects an inner RAMP coefficient, the compiler includes
that outer coordinate in its coefficient table. This can require more DMEM;
compilation rejects a table that cannot fit rather than silently changing the
order or dropping coordinates.

Software sweeps that recompile a program per point must remain outside all
hardware sweeps. They can move relative to other software axes, and hardware
axes can move relative to other hardware axes. The buttons disable moves
across that execution boundary. A hardware CPMG N axis remains a hardware
axis, subject to its existing standalone-axis restriction.

## Settings and execution

Settings schema 46 added `sweep_grouping.parameters` and `sweep_grouping.groups`.
Schema 47 adds `sweep_grouping.execution_order`, a list of stable parameter IDs
from slowest to fastest. Older settings retain the legacy duration-first hardware
order, now reflected in the table. Deleting a parameter also removes its order
entry; a newly added parameter follows existing explicitly ordered parameters
within its execution domain. Save/load and source-row remapping retain order.
Parameters and groups have stable IDs; group records contain their common count
and member IDs. Existing JSON files without these fields load with independent
axes. Missing members are removed and singleton groups normalized. Duplicate
identities, overlapping group membership and nonpositive counts are rejected.
Loading an older file does not edit the original file automatically.

The GUI registry uses keys `(kind, output, segment)`. Its compiler adapter emits
`(kind, segment, output)` keys to `FineTuneSequence.set_sweep_groups()`. Compiler
members retain their full coordinate definitions while each group contributes
one hardware loop. Preview and Run pass the same mapping, parameter IDs and
`FineTuneSequence.set_sweep_order()` keys.
Virtual-to-physical conversion, bounds, port spacing and DC/RC checks still
apply to the actual correlated coordinates. Shared duration/voltage groups
include interior points in DC completion and RC range checks because their area
can have an interior maximum even when both endpoints are small.

## Saved results and plots

QCoDeS results retain every member coordinate, the shared group index, stable
IDs, membership and the effective sweep shape. The virtual-gate matrix and
waveform recipe remain in metadata. A member coordinate uses its existing
requested/quantized axis convention; it is not a measured physical voltage.
For v5, Run snapshots `program_summary.execution_coordinates` before acquisition.
It records physical pre-IIR DAC target codes/voltages, quantized RAMP lengths,
RF active sample counts and command register values separately from the requested
member coordinates. Host software sweeps retain one snapshot per hardware
program with explicit combined point indices. Scalar columns broadcast; array
columns follow their stored point indices. Coarse periodic RF records its
scheduled stop interval and leaves the actual active duration unset.

Show QICK Program has a **Compiled coordinates** tab that expands only the
selected point, using the same exporter. FPGA-generated DC compensation and
RC-filtered analog output are not represented as measured parameter coordinates.
Physical waveform measurements require separate acquisition evidence.

A 2D map requires two independent groups or axes. Two parameters in one group
cannot serve as independent X/Y dimensions. Saved-map loading and reprojection
preserve these relationships; a single group uses a one-dimensional trace.

Sweep order controls target the v7 GUI while retaining older JSON input compatibility.
Execution-model, RTL and physical results are separate evidence levels. Personal
validation logs and waveforms belong outside Git, as described in the
[hardware validation guide](validation/README.md).
