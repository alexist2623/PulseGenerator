# AWG v8 output limits and DDR cancellation

Both repositories use branch `awg_tuning_v8`. Use the matching v8 BIT/HWH and updated
QICK server library. Older firmware still runs with limits disabled; enabling limits
on a generator without the v8 capability is rejected before execution.

## AWG Tuning and Stability Diagram

Each panel has an **AWG output limits** table keyed by physical AWG generator index.
For each selected output, enable the limit and enter **Minimum (mV)** and **Maximum (mV)**.
Limits must include zero. Reordering channels retains each physical generator's limits.
AWG Tuning and Stability Diagram store their settings separately in JSON. Legacy JSON
without these fields loads with limits disabled. Exported Python preserves the settings.

The limits apply after DC/RC compensation, before RF addition. RF, external attenuators
and amplifiers are not included. The compiler converts mV with each DAC's current-based
full-scale value and rounds inward to the effective four-code quantum. For example,
with 800 mV full scale, -80/+160 mV become -3276/+6552 codes. Requested and effective
bounds and hardware readback are recorded in the program summary. Disabling a previously
enabled v8 limit explicitly restores disabled/full-range configuration on the next run.

Show Program/compilation and editing do not write registers. At execution, the worker
stops the tProcessor, applies and verifies the limits, then uses the existing program
configuration/start path. Fine RAMP commands, sweep order and output latency are unchanged.
Clipping can change DC balance; it does not change the compensation integrator.

## Cancel

After a successful FIR DDR ARM, cancellation or an acquisition/readback error follows:

1. Stop the tProcessor trigger producer in the experiment worker.
2. Request DDR abort and wait for hardware acknowledgement.
3. Require pending=false, busy=false and armed=false before reporting successful cancellation.

The v4 capture IP drains an outstanding AXI write and clears pending triggers/partial
FIFO data without waiting for the missing trigger count. An abort timeout or missing
driver capability is a cleanup failure, not successful cancellation. The GUI displays
the failure and the next ARM remains blocked by hardware busy until abort can complete.
Cancelled captures are not returned as successful experiment data.

Cancellation before ARM does not touch DDR. AWG Tuning and Stability Diagram use the
same acquisition path. This change is not an independent analog-zero guarantee and does
not reset unrelated generators or power supplies.

## Verification

Offscreen tests exercise settings round trips, legacy JSON, channel reordering, per-DAC
conversion, exported code, compile/cache behavior, readback, and cancel immediately after
ARM followed by another run. Tests also retain abort/stop failures rather than masking
them. Local package review recognizes v8 AWG + v7 sum + DDR sample v4 and checks the final
manifest/XSA pair before deployment. No GUI window or device connection is needed.

## Run forever (output only)

The AWG Tuning Experiment panel has a separate **Run forever** button and uses the
existing **Stop** button. Disable RF Readout first. No readout configuration or QCoDeS
run/database name is required. Enabled ADC/DDR acquisition is rejected before start;
host-controlled RF sweeps are rejected rather than silently reduced to one point.
RF output, AWG hardware sweeps and external markers remain available.

This compiles a hardware loop around the complete finite reps × sweep grid. Each grid
reinitializes axis counters, coefficient/table pointers and compensation setup. It does
not simulate forever with a large reps number or allocate an ever-growing measurement
buffer. The status shows elapsed time; it does not save acquisition rows. The execution
mode is explicit in generated Python, program metadata and the compile cache key.

Checking Stop waits for the current shot to finish. The next shot therefore has an
additional instruction-preparation gap; another gap is inserted at the grid boundary.
The compiler budgets this from the complete program size and records the value in
`output_forever_preparation_guard_cycles` / `output_forever_preparation_guard_us`.
The start log also reports it. Pulse-internal edges and compensation intervals are
unchanged, but the repetition period is longer than the finite acquisition mode.
This mode is not a gap-free waveform loop.

Stop requests use tProcessor DMEM words 2 (request) and 3 (acknowledgement), not new AXI
addresses. At the next completed shot boundary, the program finishes DC/RC work, clears
the AWG to zero, sends zero-gain RF commands, mutes enabled SquarePulse outputs even if
finite-run mute was disabled, finishes markers, waits for the output pipelines, and
acknowledges completion. It then stops the tProcessor and checks AWG zero/idle status.
It never restores a prior nonzero output automatically.

The updated board server owns a two-second renewable lease. If the PC disconnects,
the board watchdog requests the same cooperative stop. A missing acknowledgement after
30 seconds or an invalid zero/idle readback locks the session against restart, stops
the producer and attempts a zero-width AWG clamp; the GUI reports cleanup failure rather
than successful Stop. The clamp precedes RF addition and is not an independent RF mute.
This watchdog requires a functioning board Python server and FPGA; it is not a hardware
interlock or an oscilloscope verification of analog zero. Recovery from a locked session
requires diagnosis; there is no automatic FPGA reset or retry.

The loop has been checked offline and in the real tProcessor/output RTL, including
exact command timestamps for short fine pulses and sweep/grid transitions. Continuous
multi-day operation and the 48-bit global timestamp rollover are not hardware-qualified.
The existing finite Start experiment behavior is retained.
