# DC Waveform Generator GUI installation

## Validated versions

The application was validated with this exact profile:

| Component | Version |
| --- | --- |
| Python | 3.11.15 |
| NumPy | 2.4.6 |
| PyQt5 | 5.15.11 |
| PyQtGraph | 0.14.0 |
| Matplotlib | 3.11.0 |
| QCoDeS | 0.58.0 |
| pytest (development only) | 9.0.3 |

PyQtGraph is the primary interactive plotting backend. Matplotlib is kept for
the compatibility fallback and for waveform preview plots. Install both so the
application does not silently switch to a slower fallback because PyQtGraph is
missing.

## Recommended Conda installation

From this directory:

```powershell
conda env create -f environment.yml
conda activate pulse_generator_gui
python DCWaveform_Generator.py
```

To install into the existing `qick_test` environment instead:

```powershell
conda activate qick_test
python -m pip install -r requirements.txt
python DCWaveform_Generator.py
```

## QICK integration

The basic GUI and Keysight QCS code generation do not require a local QICK
installation. Local QICK preview and execution helpers require the QICK library.
For the current workspace, install QICK 0.2.357 in editable mode:

```powershell
conda activate pulse_generator_gui
python -m pip install -e C:\JeonghyunPark\Workspace\QSTL_QICK\qick
```

Keysight QCS is a separate vendor package and is not installed by these public
requirements. Install it in the Keysight-supported environment when generated
QCS programs need to be executed.

## Direct QICK experiment runs

The **Experiment** tab connects to the configured QICK Pyro nameserver, runs
the AWG/RF/FIR-DDR sequence, and writes the returned 1 MSPS IQ traces to the
selected QCoDeS SQLite database. The QICK server must already be running and
reachable from this PC.

Each database run stores `I`, `Q`, magnitude, and phase against descriptive
voltage sweep axes, repetition, and sample index. Sweep axes use names such as
`awg_0_set_1_voltage_mv` and values are stored in mV, so Plottr exposes the
actual output/segment sweep controls instead of a flattened point index. The run
metadata includes the GUI settings, cross-capacitance matrix, RF output/readout
settings, QICK connection settings, and compiled program summary.

AWG waveforms are stored as compact ordered vertices for every Cartesian sweep
point as QCoDeS experiment parameters, not as JSON-only metadata. Query
`awg_virtual_vertex_mv` and `awg_physical_vertex_mv`; their setpoints are
the descriptive mV sweep coordinates, `awg_output_index`, `awg_vertex_index`,
and `awg_vertex_time_us`. Connecting adjacent vertices reconstructs each complete
SET/RAMP pulse; repeated times represent instantaneous SET changes. The full
per-clock AWG trace is not duplicated in the database.

`point_index` and IQ `time_us` are intentionally not registered as QCoDeS
parameters, so they do not appear as misleading x/y selectors. Trace time is
reconstructed as `sample_index * sample_period_us`; `sample_period_us` and the
sample rate are stored in `qick_experiment_json` metadata. At 1 MSPS one sample
index step is one microsecond.

Repetition is counted within each Cartesian sweep point. For example, two
independent two-point sweep axes produce four sweep points; two repetitions
produce eight acquisitions total, while `repetition_index` remains only 0 or 1.

The Experiment tab reports real progress. The hardware interval follows the
tProcessor completion counter, and the database interval follows the number of
IQ sample rows flushed to SQLite. IQ rows are inserted in bounded batches to
avoid the long post-pulse delay caused by one `add_result()` call per sample.

## Verification

```powershell
python -c "import numpy, PyQt5, pyqtgraph, matplotlib, qcodes; print('GUI dependencies OK')"
python DCWaveform_Generator.py
```

For development tests, install `requirements-dev.txt` and run pytest from the
source test directory.
