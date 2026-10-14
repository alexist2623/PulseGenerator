# Quantum Circuit

The **Quantum Circuit** control tab edits a separate circuit document. Its right-hand dock shows the requested AWG, RF and acquisition timeline above the qubit circuit. The existing AWG Tuning waveform is not replaced.

The right-hand **Quantum Circuit** view shares the result tab group with Trace Plot, Stability Diagram, AWG Tuning, RF S-Parameter and Noise Analysis. All these views remain selectable from every control tab; selecting Quantum Circuit does not hide the waveform or other result views.

## Editing

1. Use **Add Qubit** on the left; double-click a qubit name to rename it.
2. Use **Add Gate**, choose the affected qubits, and set its start and duration in microseconds. Independent qubits may have concurrent gates.
3. Add one or more AWG/RF settings to the gate. Right-click a gate on the right and choose **Edit** or **Delete**. Double-click also opens Edit.
4. For AWG, **Absolute** sets the requested voltage in mV. **Relative** adds the entered change to the preceding held voltage on that AWG. A ramp starts at the gate start; afterwards the final voltage remains held. **Initial AWG voltages…** defines the starting values; unlisted channels start at zero.
5. RF rows specify channel, delay, duration, frequency, raw DAC gain and phase. Delay is relative to the gate start. A Readout gate can contain AWG/RF settings and an **Acquisition** window together.
6. Use Ctrl+wheel, the zoom buttons or **Fit** in the circuit view. The circuit and pulse lanes share one time axis.

Gate names such as X, Y and CNOT label the configured pulses. They do not load a calibrated quantum-gate pulse library. Configure pulse actions before Run; use **Wait** for an intentional idle interval. CNOT/CZ editing includes a control-qubit selector. Acquisition is shown as a single interval; no DDR write lane is displayed.

**Save Settings JSON…** saves the circuit, initial voltages and shot count together with the other GUI settings. Version 44 adds the `quantum_circuit` field; earlier settings load an empty circuit and retain their original tab selection.

## Loops and n sweep

Select gates in the left list (Ctrl/Shift for multiple selection) and click **Add Loop**. Include every gate in that time interval, including simultaneous gates on other qubits. Set **Repeat n**, or enable **Sweep n** and enter an inclusive integer Start/Stop/Step range. For example, `1 → 5, step 2` measures n = 1, 3, 5; descending ranges are also supported. n = 0 skips the block. The ordinary **Shots** value repeats the entire resulting circuit independently at every n.

Gate times in the editor describe one pass through each block. Later gates and Acquisition automatically move by `(n − 1) × block duration`. Idle gaps between selected gates belong to the block. AWG Relative changes continue from the previous iteration; Absolute sets the same requested value again. To return to an initial voltage on every iteration, put an Absolute setting in the block.

The right pane shows the expanded pulse sequence and the highlighted **Loop ×n** interval. **Preview n** displays a sweep point without changing the saved counts or the range Run will measure. Edit/Remove Loop is available on the left and by right-clicking the loop label/background. Removing a loop preserves its original gates once. Deleting its last gate also removes that loop.

Multiple separate blocks are supported; overlapping/nested loops are rejected. One loop can have n sweep enabled at a time. The editor bounds expansion to 4,096 gate instances / 16,384 pulse settings and the sweep to 1,000 points. The actual loaded QICK program-memory capacity can impose a smaller limit, which is checked before execution. Programs use expanded existing instructions, so no firmware or compiler-loop logic is changed.

Run preflights **every n** before the first acquisition, then measures points sequentially. Each n is saved as a separate ordinary QCoDeS Run with its sweep ID, point index and n in metadata. A `*.circuit-sweep-<id>.json` index beside the selected database connects all Run IDs to n and records completion, cancellation or failure. Completed Runs remain available if a later point stops. Circuit settings and sweeps are saved in GUI settings version **45**, circuit document version **2**; earlier circuits continue to load.

Keep the Acquisition gate after the repeated block when n > 1: the existing backend still supports one Acquisition per shot. Repeating an Acquisition inside the block is rejected before outputs are configured.

## QICK Run

Set **Shots**, then click **Run**. **Stop** uses the existing cooperative cancellation path. Another QICK operation cannot start concurrently. Connection, QCoDeS database and compensation options come from **AWG Tuning → Experiment**. Configure the readout/input calibration in **RF Readout** and RF attenuation/filter settings in **RF Outputs**. Apply analog front-panel changes with the existing **Front Panel → Update** action before Run; editing their values alone does not apply them to the board. The circuit supplies the actual AWG/RF pulses and acquisition timing; the AWG editor's sweeps and SquarePulse outputs are not included.

The worker reads live HWH metadata, lowers the circuit through the existing compiler, and validates timing before configuring outputs. AWG/RF channel labels use the existing front-panel maps. Unsupported or missing generator mappings are rejected. RF settings for a configured channel can be reused even when its ordinary RF Outputs checkbox is off; the circuit defines whether the pulse runs.

The existing execution backend currently supports **one acquisition window per shot**. Its duration must be a whole number of samples at the live FIR rate. Gate transitions must be representable on the loaded clock and satisfy the existing command spacing. Simultaneous AWG ramps with matching bounds are supported; independently overlapping ramps or SET transitions inside another ramp are rejected rather than changing their timing.

Acquisition input time is referenced through the FIR group delay. Continuous decimation can move a stored boundary within one sample period. A recorded common startup offset separates circuit time zero from the compiler's startup. DAC/RF quantization and the existing compensation behavior remain in effect. The editor draws requested waveforms; it does not claim calibrated quantum-state evolution or measured analog output.

RF timing uses a checked digital output profile. The current QSTL AWG v3 / `axis_signal_gen_v6` path at 300 MHz has a previously measured 42-clock RF command-to-output latency; the circuit bridge advances its RF commands by that amount only when the matching metadata is present. This is alignment at the tested digital interface, not RFDC, cable or analog path calibration. An unrecognized RF timing profile is rejected before execution instead of assuming the same latency.

## Checks

Focused checks live in `test_quantum_circuit_model.py`, `test_quantum_circuit_widgets.py`, `test_quantum_circuit_integration.py`, `test_quantum_circuit_execution.py` and `test_quantum_circuit_loops.py`. They cover voltage-state calculation, editing transactions, JSON compatibility, real Qt worker lifecycle and lowering through the production QICK compiler. Loop checks include n = 0/1/3/100, shifted RF and Acquisition timing, relative AWG accumulation, compensation options, integer n sweeps, preflight failure and partial/cancelled Run preservation. Execution tests use synthetic HWH and a fake hardware connection; they do not send a physical experiment.
