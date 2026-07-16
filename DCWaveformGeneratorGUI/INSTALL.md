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

## RF S-parameter sweep

The **RF S-Parameter** tab is independent of the AWG-tuning waveform editor. It
uses a normal DDS signal-generator channel and a dynamic-readout channel, and
advances both frequency registers inside one tProcessor hardware loop. The tab
configures output ATT1/ATT2 and filter settings, input attenuation and filter
settings, frequency start/end/point count, output gain, and FIR acquisition time
per frequency. Output gain is hard-limited to 32766 and is also checked against
the selected generator's reported hardware limit.

The optional **Power Sweep (Software)** repeats that complete tProcessor
frequency sweep at a sequence of DAC gain codes. Linear spacing uses rounded
`linspace(start_gain, end_gain)`, while logarithmic spacing uses rounded
`geomspace(start_gain, end_gain)` and therefore requires positive endpoints.
The gain code controls RF amplitude; it is not a calibrated dBm value. Power is
not advanced by tProcessor arithmetic: Python compiles and runs one hardware
frequency sweep per gain point.

RF output duration is not encoded as one 16-bit generator pulse length. A
3-fabric-cycle periodic DDS command starts the output, and a separately timed
zero-gain one-shot command stops it after the FIR capture interval. This removes
the 65,535-fabric-cycle const-pulse limit; the stop takes effect at the next
3-cycle periodic boundary. The dynamic readout uses the same short periodic
word so each hardware-sweep frequency update is accepted promptly.

Each frequency point produces one post-FIR 1 MSPS DDR trace. The trace is saved
as separate `i_trace[sample]` and `q_trace[sample]` QCoDeS arrays, then reduced
to one complex response using the mean I and mean Q. The stored scalar response
uses `20*log10(hypot(mean_i, mean_q))` for magnitude and an unwrapped
`angle(mean_i + 1j*mean_q)` in degrees for phase. The GUI displays both curves
against the actual common-quantized RF frequency. **Load Saved Run** reconstructs
the response from the stored I/Q arrays; run ID 0 selects the latest run carrying
RF S-parameter metadata rather than the latest unrelated run in the database.
For a power sweep, scalar and trace data share one QCoDeS run with
`rf_power_gain` and `rf_frequency_mhz` setpoints. After every completed gain,
the new rows and result metadata are flushed and the local SQLite database is
published to the configured DB path. Plottr and the GUI can therefore refresh
the same run while later gain points are still being acquired. The GUI overlays
the completed magnitude and phase traces with one color per gain code.

## Verification

```powershell
python -c "import numpy, PyQt5, pyqtgraph, matplotlib, qcodes, plottr; print('GUI dependencies OK')"
python DCWaveform_Generator.py
```

For development tests, install `requirements-dev.txt` and run pytest from the
source test directory.
