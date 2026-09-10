# S-parameter phase display and optional QCoDeS storage

The RF S-Parameter tab has a **Phase Display** selector and a **Save measurement
to QCoDeS DB** checkbox. Defaults remain unwrapped display and database saving.

- **Unwrapped** uses the existing phase unwrapping along frequency, independently
  for each power trace and contiguous interval of valid data.
- **Wrapped (-180 to 180 deg)** displays `degrees(atan2(mean_q, mean_i))` directly.
  The I/Q mean is the existing FIR trace mean or AVG-buffer result for each
  frequency. It is not a new averaging operation. Zero or nonfinite I/Q retains
  an undefined phase (`NaN`).
- Switching the display does not reacquire data or change the captured I/Q.
  It resets any display-only phase-line subtraction. The line-fit controls are
  enabled only for unwrapped display.

With saving enabled, both single sweeps and live power sweeps save both scalar
parameters, regardless of the chosen display mode:

| QCoDeS parameter | Meaning |
| --- | --- |
| `s_parameter_phase_unwrapped_deg` | Existing unwrapped phase, degrees |
| `s_parameter_phase_wrapped_deg` | Direct I/Q angle, degrees |

Both arrays are also included in result metadata as `phase_unwrapped_deg` and
`phase_wrapped_deg`. Existing I/Q storage, including exact signed-int64 raw
traces, remains in place. Older databases without the new phase parameter still
load: both phase representations are reconstructed from their saved I/Q.

With saving disabled, acquisition and power-by-power plot updates continue, but
no measurement database, staging database or QCoDeS run is created. Results
remain in memory. Loading an existing run and reading a selected calibration
database are independent of this checkbox. The status explicitly says that the
measurement was not saved.

The choices persist in GUI JSON settings through `SParameterSweepConfig` fields
`phase_display` (`"unwrapped"` or `"wrapped"`) and `save_to_qcodes` (boolean).
When `save_to_qcodes=False`, `run_sparameter_sweep()` accepts `run_config=None`
and returns the usual result container with `run_id=None`, `database_path=None`
and `dataset=None`.

Validation covers phase wrap crossings, zero/nonfinite points, both acquisition
sources, single and power sweeps with saving on/off, both saved phase columns,
old database loading, GUI settings and both plotting backends. The existing
440-test suite plus 13 new cases passed together; the additional Matplotlib
fallback case also passed. Tests use simulated input and mock hardware.
# ADC magnitude display

**S-Parameter → Plot Display → Magnitude** offers the existing response in dB,
ADC units on a linear axis, ADC units on a logarithmic axis, and
`20*log10(ADC magnitude)` in dB. ADC magnitude is `hypot(mean I, mean Q)` before
input/output power calibration. The existing IQ64 coefficient-gain conversion
and AVG normalization are retained. The original stored I/Q is unchanged.

Linear ADC display includes zeros. Log-axis and ADC-dB display omit zeros
instead of introducing an artificial noise floor. Markers report ADC units
on both ADC-unit axes. Power selection and phase-fit state survive changes
of magnitude display. The selection is saved with GUI settings; old settings
default to the original response-dB display. PyQtGraph and Matplotlib are
both supported.
