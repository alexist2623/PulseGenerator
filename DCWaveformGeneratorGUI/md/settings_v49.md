# Settings v49: QICK repetition means and dated HDF5 files

The Experiment tab uses one **I/Q and trace storage** selector with four choices:

| Choice | Network transfer | Saved data per sweep point |
| --- | --- | --- |
| Full traces | All repetitions and time samples | All repetition traces |
| Mean I/Q (PC) | All repetitions and time samples | One I/Q pair, averaged on the PC |
| Mean trace (QICK) | One repetition-averaged trace | Every time sample of the mean trace |
| Mean I/Q (QICK) | One I/Q pair | Mean over repetitions and time samples |

I and Q are averaged separately, coherently. Magnitude/power/phase are not
averaged independently. Each sweep point, including correlated/grouped sweep
coordinates and host RF software sweep coordinates, remains separate.

Settings keys are `qick.iq_storage_mode` (`full_traces`, `mean_iq`,
`qick_mean_trace`, `qick_mean_iq`), `experiment.storage_format` (`qcodes` or
`hdf5`), and `experiment.hdf5_directory`. Earlier settings retain their existing
storage choice and use QCoDeS when the new format field is absent. The existing
settings migration notification is shown when loading an older version.

For unswept Mean I/Q acquisitions, QCoDeS stores a display-only `point_index=0`
setpoint. Plottr excludes standalone scalar parameters, so omitting this axis
would hide valid I/Q values in the database viewer. It does not add a sweep or
another measurement. Both PC and QICK mean I/Q modes use this setpoint.

## Board update

Select the existing **v7** firmware in the QICK updater and the updated local
QSTL_QICK repository. The existing file deployment includes `qick.py` and the new
`trace_averaging.py`; the Python package version and installation metadata stay
unchanged. No new RTL or bitstream is needed for these averages.

After the file update, restart the QICK Python/Pyro server and the GUI using the
existing update workflow. The optional updater runtime verification checks
`get_fir_average_capabilities()` and the averaging API version, in addition to
firmware identity and process restart time. A v7 firmware label alone is not
proof that the running Python driver has the feature. Acquisition also checks
the capability before configuring the readout or starting a program. It never
silently falls back to raw transfer when a QICK mean was selected.

The board API `get_ddr4_fir_mean(n_samples, repetitions, n_points, start,
stride_bytes, reduce_samples=False)` reads completed point-major,
repetition-minor DDR captures. Physical offsets and per-trigger padding are
preserved. Integer samples are converted to float64 before summation and IQ64
is scaled by `2**-46`; this avoids integer-sum overflow. Temporary read/conversion
buffers are chunked. The sample dimension is retained unless `reduce_samples`
is true. The pure averaging helper also supports real-valued traces.

All repetitions are still captured in DDR. Network traffic and PC memory are
reduced; DDR capacity and acquisition time are unchanged. Trigger/sample
alignment and coherent RF phase must be consistent for meaningful averaging.
Cancellation is checked between RPC requests; an active board CPU request must
finish before cancellation is observed on the client.

## HDF5 files

Choose **Save format → HDF5 file per experiment** and select an **HDF5 root
directory**. PC-side `h5py` is included in the GUI requirements. The local
experiment start timestamp determines both directory and filename, even when
acquisition finishes after midnight:

```text
<root>/2026-11-02/20261102_235959_ExperimentName.h5
```

Invalid Windows filename characters are replaced; Unicode experiment names
are retained. Files are created exclusively. A collision adds `_001`, `_002`,
etc., and never replaces an existing experiment. HDF5 mode does not create a
QCoDeS database. Tools with their own database destination retain that
destination; the Experiment setting does not replace calibration databases.

The `qick_hdf5_v1` layout contains:

- `/data/iq`: exact original integers for Full traces, or float64 input-code
  means. Shape is `(point, repetition, sample, 2)` for raw traces,
  `(point, sample, 2)` for QICK mean traces, and `(point, 2)` for mean I/Q.
  Attributes describe axes, I/Q components, units, fixed-point scale, sample
  rate and actual source repetition/sample counts.
- `/axes/time_s`: sample times relative to the first captured sample, present
  when a time trace is saved. Trigger offsets remain in acquisition settings.
- `/sweep/coordinates`, `/sweep/display_coordinates`, `/sweep/shape`, and the
  cross-capacitance matrix, when available.
- `/metadata/json`: GUI configuration, waveform recipe/optional expanded
  vertices, RF/readout settings, calibration conversion metadata, sweep axes,
  groups/parameter IDs, program summary and averaging location/axes.
- Root attributes: schema, GUID, start/end timestamps with timezone, status,
  and completed point count.

Files are created after acquisition, during persistence. Interrupted writes
retain `cancelled` or `failed` status and the number of written points; a run
cancelled before persistence does not produce a file. `load_hdf5_result(path)`
rejects unfinished files. For raw IQ64, convert a copy using
`np.ldexp(iq.astype(np.float64), -iq_scale_log2)`; do not apply the source scale
again to a QICK mean. Calibration formulas are recorded separately from the
input-code dataset. The existing saved-run database selector remains a QCoDeS
loader; HDF5 files can be read with `load_hdf5_result` or `h5py`.

## Validation

Offline tests exercise real and I/Q traces, 1/3/300 repetitions, signed IQ64
overflow avoidance, nonzero DDR addresses, odd sample padding and explicit
strides, complete-point transfer chunks, invalid replies, cancellation, all
four storage choices in both formats, grouped coordinates, exact raw int64
round trips, timestamp/collision handling, old JSON defaults, capability checks
and the existing update/rollback transaction tests. A local Pyro connection
also exercises NumPy array serialization through the new driver API. This
validation does not deploy files to the physical QICK board or emit pulses.
