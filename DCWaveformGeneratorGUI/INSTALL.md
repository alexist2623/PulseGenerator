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

Each sweep-point/repetition acquisition is stored as two QCoDeS array results:
`i_trace[sample]` and `q_trace[sample]`. This keeps I and Q independently
selectable in Plottr while retaining one compact array per channel and trace.
Sweep axes use names such as
`awg_0_set_1_voltage_mv` and values are stored in mV, so Plottr exposes the
actual output/segment sweep controls instead of a flattened point index. The run
metadata includes the GUI settings, cross-capacitance matrix, RF output/readout
settings, QICK connection settings, and compiled program summary.

AWG waveforms are stored as compact ordered vertex arrays for every Cartesian
sweep point and for every channel. Query channel-specific parameters such as
`awg_0_virtual_vertices_mv`, `awg_0_physical_vertices_mv`,
`awg_1_virtual_vertices_mv`, and `awg_1_physical_vertices_mv`. Each dependent
uses its own channel-specific array time axis, for example
`awg_0_vertex_time_us`. The generic `awg_vertex_time_us` parameter is not used.
In Plottr's Dimension assignment panel, select the channel's vertex-time
parameter as X and the physical or virtual voltage as the dependent value.
Sweep-voltage axes remain attached, so one time trace is available for every
Cartesian sweep coordinate. Repeated times represent instantaneous SET changes.
The full per-clock AWG trace is not duplicated in the database.

`point_index`, `sample_index`, and IQ `time_us` are intentionally not registered
as QCoDeS parameters, so they do not appear as misleading x/y selectors. The
sample index is reconstructed from the `i_trace`/`q_trace` array length, and
trace time is `arange(sample_count) * sample_period_us`. `sample_period_us` and
the sample rate are stored in `qick_experiment_json` metadata. At 1 MSPS one
sample step is one microsecond. Magnitude and phase are also derived when reading
instead of being duplicated in SQLite. Use `load_qick_iq_arrays(dataset)` to
reconstruct I, Q, magnitude, phase, sample index, and time arrays. The loader
also accepts older packed `iq_trace[sample, component]` runs.

Repetition is counted within each Cartesian sweep point. For example, two
independent two-point sweep axes produce four sweep points; two repetitions
produce eight acquisitions total, while `repetition_index` remains only 0 or 1.

The Experiment tab reports real progress. The hardware interval follows the
tProcessor completion counter, and the database interval follows completed IQ
trace arrays. The database is written to a local SSD staging directory first.
After the run it performs a WAL checkpoint and copies the completed SQLite
database to the configured location, including Nextcloud paths. This avoids one
SQL row per sample and slow network-synchronized writes during acquisition.

## Verification

```powershell
python -c "import numpy, PyQt5, pyqtgraph, matplotlib, qcodes, plottr; print('GUI dependencies OK')"
python DCWaveform_Generator.py
```

For development tests, install `requirements-dev.txt` and run pytest from the
source test directory.
