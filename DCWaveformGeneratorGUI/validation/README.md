# GUI hardware validation infrastructure

This package records a new attempt through real Qt mouse/key events, the visible
**Run QICK Experiment** button, and the existing production `QickExperimentWorker`.
It does not replace the compiler, upload, hardware acquisition, or QCoDeS storage
functions. Import and construction never connect to an instrument or run an
experiment. The configured runner requires an explicit `start(allow_hardware=True)`.

An earlier scope-only campaign is not evidence that this path ran. Each new
attempt records `execution_route.json`, including the observed worker class,
source hash, real button event, and `physical` versus `offline_test` context.
The unit tests use the actual production Run method and worker, replacing only
the instrument-execution boundary. Their results are software tests, not physical
measurements or validation of the compiler's numerical output.

## Integration

The caller provides a visible production `MainWindow`, its module, and
`window._experiment_panel.run_button`. Keep the runner alive until `finished`.
No production GUI file needs to be patched. A short observer factory attaches
signals to the original worker during the button click, before its thread starts;
the original class is restored immediately after the click.

1. Create `Fixture` with the queried scope identity and explicit CH1/CH2 wiring,
   generator, connector, cable, impedance, probe ratio, coupling, and attenuation.
   A separate third AWG loopback and marker/trigger wiring are optional fixture
   descriptions. Their coverage is **unavailable** until actually supplied; a
   split of A or B does not substitute for output C.
2. Create `FirmwareIdentity` with the loaded board configuration, verified
   BIT/HWH SHA256 relationship, GUI and driver source hashes, and capabilities.
   Branch names and filenames are not loaded-firmware evidence. A fresh
   `live_identity_check()` callback must return the same version, BIT/HWH hashes,
   loaded configuration and capabilities, plus `verified=True` and a `method`.
   It must query/verify the actual setup; do not synthesize these values from a
   local path. This callback is read-only and runs before scope mutation or Run.
3. Declare every `Point(index, coordinates, repetitions, groups)`. Add numerical
   `Reference` arrays for **requested**, **quantized**, and **physical** expectations
   on both channels before execution. Reference metadata specifies units, time
   origin, reference plane, model version and transformations. The physical model
   must include the actual RC/DC/measurement circuit when applicable. The package
   cannot infer an analog circuit from a waveform name.
4. `RunArchive.seal(gui_definition=...)` freezes the intended arrays and GUI
   definition. `QtAction` supports text plus Tab commit, click, tab, key, combo,
   and modal input. For modal input, supply a function receiving the active dialog
   and returning its actions, including an explicit accept/cancel button click.
   A read-only `gui_snapshot()` must match the archived definition after events.
5. Supply `scope_factory(directory)`, `result_identity`, `point_mapper`, `cleanup`
   and a `comparison` policy to `GuiHardwareRunner`. All scope operations, cleanup,
   native-data reconstruction and plotting run on one background executor. GUI
   events and normal worker signals remain on the Qt thread.

```python
from validation.qt_runner import GuiHardwareRunner, stored_run_identity

runner = GuiHardwareRunner(
    window=window,
    production_module=gui_module,
    run_button=window._experiment_panel.run_button,
    archive=archive,                 # Intended references already sealed.
    scope_factory=scope_factory,     # Explicit verified wiring/settings.
    live_identity_check=read_live_identity,
    result_identity=stored_run_identity,
    point_mapper=identify_points,
    cleanup=reset_rc_and_zero_outputs,
    comparison=comparison_policy,
    timeout_s=120,
)
runner.start(allow_hardware=True, actions=actions, gui_snapshot=read_gui_definition)
```

`point_mapper(capture_metadata, stored_result)` returns rows with `point`,
`repetition`, `channel`, native `start`/`stop` indices, and `identity_evidence`
containing `verified=True` and the explicit identification method. Marker or
independent sequence identity must justify this mapping. Sample count alone does
not identify a point. Duplicate, stale, out-of-order, overlapping, out-of-range,
and missing mappings cannot yield a complete result. A mean-only QCoDeS result
does not prove that all individual repetitions were captured.

`comparison(row, stored_result)` returns `reference_kind`, explicit
`time_offset_s`/`voltage_offset_v`, and `absolute_tolerance_v`. There is no automatic
gain fit or hidden alignment. Corrected arrays, interpolation, residuals and plots
are stored separately from native data. References must cover the entire assigned
ROI, including compensation/reset windows as required by the case. Invalid scope
codes, absent tolerances, insufficient model coverage, and missing comparisons
remain inconclusive. Sampling alone does not certify analog bandwidth or timing.

`cleanup()` must implement and verify the case's explicit RC off/reset and zero
output procedure using existing supported operations; a plain SET zero may retain
RC state. Return `passed=True` only after verification, and record its details.
The callback is not called if no DUT worker started. Scope restoration is always
attempted when a scope session exists. On timeout/cancel, the runner requests the
production worker's cooperative cancellation. It never resets outputs concurrently
with a worker that has not stopped; `attention_required` reports that condition.
The caller retains exclusive ownership of scope and DUT for the entire attempt.

## Scope session

`ScopeSession` accepts an unopened `resource_factory`, explicit expected identity,
configuration, maximum native sample interval, and permitted channel-origin skew.
`visa_resource_factory(address, backend=...)` is lazy and owns/closes its PyVISA
resource manager. The caller chooses the address; none is auto-discovered.

The preserved DSO-X6004A adapters use read-only capability queries, complete
binary SETUP backup, one fresh SINGLE acquisition, AER/TER/operation-register
confirmation, and continuous STOP during both native RAW/WORD reads. Channel
origins are preserved separately. A bounded origin difference is not a statement
of zero physical skew. Restoration verifies the original setup and RUN/STOP
state; the previous acquisition buffer itself cannot be restored.

Protocol code was copied unchanged from the reviewed adapters. Their SHA256:

- `scope_capture.py`: `f01b8e861eef7dc71b34219c861acc52ecaf2796d3c5ae6da94cfdc498ef39e8`
- `compact_capture.py`: `adaedb0c019fdf3a7882843a3614e18ad99dc8d13430e12ed56e08f12f29a37c`

The wrapper injects per-session atomic JSON/gzip writers without changing the
protocol. RAW responses and preambles are stored before decoding and validation,
including failed binary responses. See the
[Keysight programming guide](https://www.keysight.com/cf/en/assets/9018-07124/programming-guides/9018-07124.pdf)
for waveform transfer/scaling and source-specific XORigin (pp. 1524–1555).
Unsupported `RSTate?`, `OPERation:CONDition?`, and `ADER?` are not used.

## Private evidence and coverage

The default root is `F:/QICK_Private_Analysis/gui_hardware_validation`. Any Git
checkout/worktree or `analysis_results` output path is refused. These personal
measurements and reports must not be committed or pushed. Code and tests may live
in Git. Use a new attempt for every retry; old files are never replaced.

Each attempt retains original IEEE blocks as gzip, preambles, full native codes
and calibrated time/voltage arrays as lossless NPZ, requested/quantized/physical
references, live identity, GUI events, worker stages, cleanup, QCoDeS identity,
point/repetition/acquisition IDs, derived overlays/residuals, and a browsable
`index.html`. SHA256 and decompression/reload checks precede final completion.
Incremental immutable checkpoints preserve partial progress after a crash.
`verify_archive()` can independently reload every finalized file.

`SuiteManifest` declares expected cases before execution, retains failed attempts
and retries, verifies their archives, and writes versioned JSON/HTML indexes.
No category becomes passed merely because some other cases passed. Unplanned or
unevaluated categories and unavailable fixtures keep `full_suite_passed=False`.
`duration_case_plan('hold'/'ramp')` supplies ten 100 ns–100 us geometric values as
**individual GUI executions**, not a native logarithmic hardware sweep. Arithmetic
native loops are separately labelled `hardware_loop`; group membership and
coordinates belong in each point's metadata.

## Concrete A/B cases

`basic_awg_cases(outputs)` in `awg_cases.py` creates explicit three-stage A/B
specifications: separate logarithmic HOLD/RAMP runs, arithmetic duration loops,
opposite grouped voltage ranges, a nonidentity matrix, an independent duration
axis, count-one repetitions, and independent DC/RC combinations. A generated
plan is not a measured result. Supply outputs from the verified fixture and
review the voltage limits before executing it.

`MainWindowCaseBinding(window, fixture, case)` resolves the production controls
and supplies a lazy `actions()` iterable. It uses real table key input, sweep
dialogs, grouping checkboxes, the matrix dialog, and compensation controls.
It also provides group removal/deletion/re-addition and settings-dialog actions.
It refuses an unrelated waveform with more than two outputs or three stages and
requires the existing front-panel mapping and full-scale current to match the
case. It does not connect to devices or click Run. Use a dedicated visible window.

`AwgReference` constructs integer DAC targets and RAMP samples independently
from the case specification. It imports neither the production compiler nor its
preview. Requested virtual voltages, matrix-converted requested values, and
quantized physical samples are separate arrays. Group members have independent
ranges and a common index. The GUI's duration-before-voltage axis order is
normalized explicitly, with both the original and normalized plans archived.

```python
from validation.awg_cases import basic_awg_cases, prepare_awg_case

plan = basic_awg_cases(verified_outputs)
case = next(case for case in plan if case.get('adapter_status') != 'unavailable')
prepared = prepare_awg_case(
    window=window, fixture=fixture, identity=identity, case=case,
    output_models=fixed_analog_models, root=private_directory,
)
# Real Qt preparation events and committed readback are now archived; no Run yet.
runner = prepared.create_runner(
    production_module=gui_module, scope_factory=scope_factory,
    live_identity_check=read_live_identity, result_identity=stored_run_identity,
    point_mapper=identify_points, cleanup=reset_rc_and_zero_outputs,
    comparison=comparison_policy,
)
runner.start(allow_hardware=True, actions=(), gui_snapshot=prepared.binding.snapshot)
```

The fixed analog models must identify independently obtained calibration
evidence. Supported transfers are a calibrated flat DC path and a first-order
high-pass path with an explicit settled-zero initial state. No capture is used
to fit the expected gain, offset, delay, or time constant. The default preparation
supports compensation-off baseband cases. DC/RC or RF cases require their
independent composition and timing models; unsupported cases fail preparation
and retain an unsealed archive. They never silently inherit an identity model.

Verified point identification, the separate third-output loopback/marker fixture,
live firmware evidence, suitable analog bandwidth, and actual acquisitions are
still needed before claiming completion of the full physical suite. Offline Qt
and numerical tests establish code behavior, not physical validation.

The plan retains a SET-only case as explicitly unavailable: the current
production waveform editor requires a positive incoming RAMP between SET
stages. Supported HOLD cases declare their incoming RAMP durations explicitly;
the adapter never substitutes a short RAMP for a requested instantaneous SET.
Record the unsupported plan entry with `SuiteManifest.unavailable` rather than
attempting to execute it. A rejected Add segment stops the adapter immediately.

`CompensationReference` adds independently computed 48-bit area balancing and
the Q48 RC inverse-filter arithmetic. `CompensationSchedule` must explicitly
describe the main end, DC first-output time, barrier, initial state, clear/reset
policy and complete capture interval. It does not infer them from captured
edges. `declared_compensation_builder(schedules, normalized_case=...)` connects
this model to `prepare_awg_case`. Bind the complete `validate_case(case, fixture)`
specification: schedules are indexed in that normalized duration-first order.
Requested arrays remain the user waveform, padded with zero after its end;
automatic DC/RC output is preserved in separate pre-RC and final-DAC arrays.

`RfCaseBinding` in `rf_case_controls.py` supports Single pulse inputs through
the actual RF controls, after preparing the A/B case. Optional `duration_sweep`
(`start_us`, `stop_us`, `count`) and `frequency_sweep` (`start_mhz`, `stop_mhz`,
`count`) describe independent Cartesian axes; omission or `None` disables the
axis. The binding preserves existing AWG groups and does not automatically group
the RF axes. Manual integer `gain` and independently calibrated reference
`amplitude_v` are distinct required fields. Committed readback includes both
ranges, counts, enabled states and the explicit within-segment option. The
snapshot stores requested coordinates, not measured or hardware-quantized ones.
Composite, calibration-database and power-sweep bindings remain unavailable in
this validation adapter. This limitation is about the adapter, not the production
GUI's feature support. The A/B archive helper still rejects RF composition until
a full sum/analog reference is supplied. Qt-to-Run-argument tests do not establish
physical or RTL validation of those new adapter cases.

The AWG-only binding rejects active RF outputs in both snapshot and requested-case
verification. Enabling RF after A/B preparation cannot reuse an AWG-only sealed
reference. The separate RF control snapshot describes those controls; it does
not certify an AWG-plus-RF physical reference.

`GuiHardwareRunner` compares the supplied read-only snapshot with the sealed
definition before scope preparation and again immediately before clicking Run.
A changed definition or snapshot error prevents worker creation, restores the
opened scope, and records a failed run. It skips DUT output cleanup when no
worker was started. The callback must represent every execution-relevant setting
in the supported case: comparison covers its definition, not unrelated or omitted
GUI fields. This check does not lock the GUI after Run or replace missing
physical reference models.

`rf_reference.py` separately provides fine/coarse active-versus-busy timing,
DDS frequency quantization, explicit DDS phase origin, calibrated RF transfer,
boundary plans, and conservative frequency/interval acceptance. Declare the
latency and its reference plane explicitly: native generator input and an
observed tProcessor issue event are different origins. Aliasing, inadequate
bandwidth, too few cycles or excessive uncertainty must remain inconclusive.

The actual static QFileDialog save/reload test is unavailable in Windows
offscreen Qt because that environment terminates during dialog acceptance.
`settings_actions` refuses that environment before opening the dialog. JSON
data roundtrip and other real Qt controls are tested separately; they do not
turn the unavailable file-dialog check into a PASS.
