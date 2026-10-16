# Sweep groups on v5

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

## Settings and execution

Settings schema 46 adds `sweep_grouping.parameters` and `sweep_grouping.groups`.
Parameters and groups have stable IDs; group records contain their common count
and member IDs. Existing JSON files without these fields load with independent
axes. Missing members are removed and singleton groups normalized. Duplicate
identities, overlapping group membership and nonpositive counts are rejected.
Loading an older file does not edit the original file automatically.

The GUI registry uses keys `(kind, output, segment)`. Its compiler adapter emits
`(kind, segment, output)` keys to `FineTuneSequence.set_sweep_groups()`. Compiler
members retain their full coordinate definitions while each group contributes
one hardware loop. Preview and Run pass the same mapping and parameter IDs.
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

New feature validation targets v5 while retaining older JSON input compatibility.
Execution-model, RTL and physical results are separate evidence levels. Personal
validation logs and waveforms belong outside Git, as described in the
[hardware validation guide](validation/README.md).
