# Incremental fine RAMP voltage sweeps

Fine RAMP no longer rejects an experiment merely because it contains a voltage
sweep. The existing GUI voltage and duration sweep controls work together,
including grouped loops, independent loops, repetitions, and JSON reload.
The change is in the desktop compiler; no driver, firmware or RTL update is
required. Restart the GUI to load the new compiler.

## Runtime arithmetic

- SET endpoint voltages retain the existing quantized DAC-code grid:
  `value = first_code + index * code_increment`. The tProcessor advances the
  value with ADD and rewinds it on an inner-loop boundary.
- Integer sample-duration increments use the same ADD/reset mechanism.
  Non-affine rounded RAMP duration grids retain duration coefficient tables.
- A duration row supplies each edge's starting Q18 slope and its voltage
  increment. Independent voltage loops ADD the increment. Grouped voltage and
  duration use their shared row index. No host acquisition loop or complete
  packet table is introduced.
- Signed 35-bit slopes use two additive radix-256 registers. Emission folds
  the low-register carry into the high register and packs the v7 low32/high3
  fields. Both rising and falling slopes support negative values and full-scale
  one-sample transitions.
- Fine SET-HOLD retains its sample counter and short/long-packet branch.
  Independent HOLD coordinates do not become DMEM table dimensions.

As for coarse RAMP, voltage increments and duration-conditioned slope increments
are rounded once to their integer grids. An independently rounded ideal Q18
slope at each voltage point can differ slightly from an incremented slope.
The executed words, exported coordinates, DC area and RC checks use the
incremented slopes. Voltage and time coordinates retain their existing actual
step/end displays. The packet still clamps to and holds its specified endpoint.

In the default duration-outer/voltage-inner order, increasing the number of
voltage points does not increase coefficient-table storage. Multiple independent
rise/hold/fall axes retain separate fields. If the user explicitly moves a
coupled voltage axis outside a duration axis, the existing coefficient loader
conditions that row on the outer coordinate; this can require more DMEM.
The existing capacity check rejects such a program before execution rather than
silently changing its loop order. Use duration-outer order for large grids.

Voltage sweep controls address SET endpoints. The explicit `turn` value in a
two-edge fine packet remains its separately configured intermediate voltage;
sweeping the terminal SET does not rename or silently repurpose that value.
The identity virtual-gate matrix requirement is unchanged.

## Offline validation

`test_fine_voltage_sweep.py` exercises both loop orders, grouped voltage/time,
rep=2, signed 35-bit carries, 14/15/16/17-sample edges, the long-HOLD branch,
DC/RC checks, unrelated-output voltage sweeps, memory scaling, and the GUI
group/save/load/remove flow. Instruction-level execution is compared with the
reconstructed emitted packet words and independent target/slope checks.
These are offline checks, not oscilloscope measurements.
