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

## Keysight QCS 2.5.5 environment

QCS 2.5.5 requires NumPy below 2.0, while the validated QICK-only profile uses
NumPy 2.4.6. Keep the runtimes isolated and use `requirements-qcs.txt` for the
QCS application:

```powershell
conda create -y -n qcs python=3.12 pip
conda activate qcs
python -m pip install keysight-qcs==2.5.5 --extra-index-url $env:KEYSIGHT_QCS_INDEX
python -m pip install -r requirements-qcs.txt
python DCWaveform_Generator.py
```

Set `KEYSIGHT_QCS_INDEX` to the authenticated package-index URL supplied by
Keysight. Do not save an access token in this repository. QCS Common 0.10.20
still imports `pkg_resources`; `requirements-qcs.txt` therefore pins
Setuptools below version 81.

The QCS Experiment backend loads a serialized `ChannelMapper`. The GUI can
build this file: select **Keysight QCS** in the **Experiment** tab, click
**Configure M5000 front panel...**, then click an empty chassis slot to install
a module. Clicking an installed module opens the same module list plus a
**Remove** action. Open the front panel from a DC, RF, or acquisition control
and click the required physical SMA; the GUI updates the virtual binding and
generates/applies the native mapper automatically once the required channels
are complete. Optional inspection, validation, imported-mapper, and explicit
Save As controls remain in **Advanced Hardware Settings**, outside the normal
front-panel window. The builder writes and reloads the native `.qcs` file
through the QCS 2.5.5 API; it never edits the serialized mapper as ordinary
JSON. The same editor is available from **Setup > Keysight QCS M5000 Front
Panel...**.

An M5201A is an analog Down Converter with four RF-input/IF-output channel
pairs; it is not a separate digitizer. First install both an M5200A and an
M5201A in the Front panel. From the Acquisition tab, open the front panel and
click the M5201A module or one of its RF/IF connectors. In the route window,
choose the M5201A pair, the physically connected M5200A SMA, and the shared LO,
then click **Apply Route Automatically**. The GUI creates or moves the
acquisition binding, records the explicit cable, and applies the resulting
ChannelMapper without requiring the user to edit mapping tables. A digitizer
channel and a Down Converter pair can each appear in only one link. The LO must
be in the hardware-supported 1-18 GHz range, and every link targeting the same
physical M5201A module uses the same value because its four pairs share one
internal LO. The native mapper is written with
`ChannelMapper.add_downconverters()` and the shared LO is applied to every
linked pair on that module.

The Front panel defaults to QCS controller IP `192.168.2.105`; edit that field
when using another controller. **Identify Hardware Configuration** reads the
installed module model, host-controller number, chassis number, and slot from
the QCS Common system inventory service, then replaces the displayed topology.
Mappings and explicit Down Converter links that still target compatible
connectors are preserved. The GUI asks before removing incompatible items and
never guesses external cables or silently reroutes channels. In particular,
module inventory cannot reveal which M5201A IF output is physically cabled to
which M5200A input, so identification never creates an M5200A-to-M5201A link
from module presence alone. Identification uses an existing QCS access token
from the current user's `.qcs_token.json`; it does not store credentials or
attempt a default login. If the saved token has expired, refresh it once in an
interactive `qcs` shell and click Identify again:

```powershell
conda activate qcs
python -c "import keysight.qcs as q; m=q.ChannelMapper(ip_address='192.168.2.105'); q.HclBackend(m).login()"
```

**Load Mapper...** resolves an existing mapper for inspection and for choosing
the experiment's role bindings. Imported mappers are read-only for **Save
Mapper As...** even though their M5200A-to-M5201A pair associations and shared
M5201A LO values can be displayed by this editor. A native mapper can contain
other channel settings, constraints, and relationships that the focused GUI
does not model; allowing Save As could silently discard them. Use **Restore
diagram layout** to explicitly start a fresh, editable mapper recipe. Imported
phase, M5300 LO, M5201 link, and M5201 shared-LO values are shown and preserved
for inspection. Channels that do not match an existing DC, RF, or acquisition
name remain **Unassigned** until a role is selected. The original imported file
name is protected; restoring the diagram proposes a separate
`_front_panel.qcs` file. For a run-eligible imported mapping, the GUI reads the
native file to require an LO on every active M5300 RF channel and on an M5201
linked to the active acquisition channel. Adding or removing waveform outputs
promotes or demotes matching unassigned M5301 channels as role-only changes,
preserving the native mapper whenever its channel definitions do not need to
change.

For automation, this abbreviated equivalent maps two DC outputs, one RF
output, and one M5200A digitizer input through an M5201A Down Converter pair:

```python
import keysight.qcs as qcs

mapper = qcs.ChannelMapper()

dc_left = qcs.Channels(0, "dc_left")
dc_right = qcs.Channels(0, "dc_right")
rf_readout = qcs.Channels(0, "rf_readout", absolute_phase=True)
digitizer = qcs.Channels(0, "digitizer", absolute_phase=True)

mapper.add_channel_mapping(
    dc_left, [(1, 2, 1)], qcs.InstrumentEnum.M5301AWG
)
mapper.add_channel_mapping(
    dc_right, [(1, 2, 2)], qcs.InstrumentEnum.M5301AWG
)
rf_address = qcs.Address(1, 3, 1)
mapper.add_channel_mapping(
    rf_readout, rf_address, qcs.InstrumentEnum.M5300AWG
)
mapper.set_lo_frequencies(rf_address, 6.0e9)
digitizer_address = qcs.Address(1, 5, 1)
downconverter_address = qcs.Address(1, 6, 1)
mapper.add_channel_mapping(
    digitizer, digitizer_address, qcs.InstrumentEnum.M5200Digitizer
)
mapper.add_downconverters(digitizer_address, downconverter_address)
mapper.set_lo_frequencies(downconverter_address, 6.0e9)
qcs.save(mapper, "lab_channel_mapper.qcs")
```

The addresses above and the packaged front-panel diagram are starting points,
not a claim about the laboratory wiring. The mapper builder validates module
overlap, connector ranges, QCS-compatible channel names, one DC mapping per
waveform output, role/module compatibility, an explicit 0-18 GHz LO for every
M5300 channel, one-to-one M5200A/M5201A pair associations, and one shared
1-18 GHz LO for every linked M5201A module. **Apply to Experiment**
synchronizes the DC names in output order, RF generator-number bindings, and
the digitizer binding with the existing QCS controls. **Save Mapper As...**
also applies those bindings after a save/reload check. Applying a new or
physically modified layout records it as a draft and blocks **Run QCS
Experiment** until a matching native mapper is saved. Existing external
mappers remain authoritative until the front-panel builder is explicitly
applied. Saved and imported configurations also record the mapper file's
SHA-256 identity; if that file is replaced or edited later, Run is blocked
until it is reloaded or saved again. Set **DC full scale
(+/-)** to the physical voltage represented by QCS amplitude `+1.0`; the
M5301A default is 2.5 V. The backend uses this value to preserve the millivolt
levels shown in the waveform editor.

The QCS path currently covers the primary Experiment workflow: physical DC
waveforms (including cross-capacitance and Bias-T compensation), Cartesian
software sweeps (up to 10,000 Cartesian points), RF pulses, digitizer
acquisition, and QCoDeS persistence. The QCS hardware-demod timing-rate control
converts the existing samples-per-trigger value into an integration duration.
In raw mode, the requested sample count is converted using the physical
digitizer rate reported by the channel mapper (4.8 GS/s for M5200), and that
hardware rate is saved with the trace. Hardware-demodulation mode requests an
RF integration filter; raw mode requests a duration-based trace acquisition
and reads it with `get_trace()`. Experiment runs are blocking so every
software-sweep point is complete before its data is normalized and written to
QCoDeS.
QICK-specific DDR controls, tProcessor assembly preview, calibrated QICK RF
power sweeps, Stability Diagram, RF S-Parameter, and Noise Analysis remain
QICK-only and are not silently translated.

## Direct QICK experiment runs

The **Experiment** tab connects to the configured QICK Pyro nameserver, runs
the AWG/RF/FIR-DDR sequence, and writes IQ traces at the sample rate detected
from the loaded HWH (currently 1 MSPS or 50 kSPS) to the
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
the sample rate are stored in `qick_experiment_json` metadata. A 1 MSPS sample
step is 1 microsecond and a 50 kSPS sample step is 20 microseconds. Magnitude and phase are also derived when reading
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

The S-parameter tab has its own QCoDeS DB file selector. Sweep execution and
**Load Saved Run** use this dedicated path and do not use the DB path from the
Experiment tab. QICK connection, experiment name, sample name, and notes remain
shared with the Experiment tab.

The optional **Power Sweep (Software)** repeats that complete tProcessor
frequency sweep at a sequence of DAC gain codes. Linear spacing uses rounded
`linspace(start_gain, end_gain)`, while logarithmic spacing uses rounded
`geomspace(start_gain, end_gain)` and therefore requires positive endpoints.
The gain code controls RF amplitude; it is not a calibrated dBm value. Power is
not advanced by tProcessor arithmetic: Python compiles and runs one hardware
frequency sweep per gain point.

Enable **Frequency Response Compensation** to select a board-matched
`gain_pwr_calb.db` run covering the complete frequency range. The calibration
code does not invert the measured power-versus-gain points. It removes the
linear DAC-gain term from each measured point, extracts only the relative
frequency response, and normalizes that response to the weakest frequency in
the requested sweep. ATT1/ATT2 contribute only a frequency-independent level
offset.

For a single S-parameter sweep, Python creates one nominal gain for the target
power and loads one frequency-dependent gain table into tProcessor DMEM. The
hardware frequency loop reads the corresponding gain with `memr` at every
frequency. For a power sweep, the power axis remains a Python software loop:
each power point gets a new nominal gain and a newly loaded frequency-gain
table, followed by one hardware frequency sweep. At most 4000 gain words are
stored from DMEM address 16. Sweeps with more than 4000 frequency points map
adjacent frequency points to the nearest representative table entry.

RF output duration is not encoded as one 16-bit generator pulse length. A
3-fabric-cycle periodic DDS command starts the output, and a separately timed
zero-gain one-shot command stops it after the FIR capture interval. This removes
the 65,535-fabric-cycle const-pulse limit; the stop takes effect at the next
3-cycle periodic boundary. The dynamic readout uses the same short periodic
word so each hardware-sweep frequency update is accepted promptly.

Each frequency point produces one post-FIR DDR trace at the HWH-selected rate. The trace is saved
as separate `i_trace[sample]` and `q_trace[sample]` QCoDeS arrays, then reduced
to one complex response using the mean I and mean Q. Without input calibration,
the stored scalar response uses `20*log10(hypot(mean_i, mean_q))` for magnitude.
When the selected calibration DB contains both a matching output-board run and
a matching input-board run, the raw ADC magnitude is retained as
`adc_magnitude_db`, actual connector output/input powers are stored, and the
displayed S-parameter becomes `P_input_dBm - P_output_dBm`. Phase remains the
unwrapped `angle(mean_i + 1j*mean_q)` in degrees. The GUI displays both curves
against the actual common-quantized RF frequency. **Load Saved Run** reconstructs
the response from the stored I/Q arrays; run ID 0 selects the latest run carrying
RF S-parameter metadata rather than the latest unrelated run in the database.
For a power sweep, scalar and trace data share one QCoDeS run with
`rf_power_gain` and `rf_frequency_mhz` setpoints. After every completed gain,
the new rows and result metadata are flushed and the local SQLite database is
published to the configured DB path. Plottr and the GUI can therefore refresh
the same run while later gain points are still being acquired. The GUI overlays
the completed magnitude and phase traces with one color per gain code.

## Power calibration

The **Calibration** tab writes output and input runs to a dedicated
`gain_pwr_calb.db`. Output calibration starts a periodic QICK DDS tone at every
frequency/gain pair and reads a Keysight/Agilent oscilloscope FFT marker through
PyVISA. Configure the scope VISA resource, input channel, FFT math function,
span, settling time, and marker-average count. The resulting QCoDeS columns are
`gain`, `freq`, and `pwr`, matching the original calibration notebooks. The
scope must support the DSO-X 6000-series SCPI commands used by
`KeysightFftPowerMeter`; adapters for other instruments can implement the same
`measure_power_dbm()` interface.

Input calibration selects a covering output-board run, performs one FIR-DDR
hardware frequency sweep for every configured gain, and computes the known
input power as calibrated output power minus the configured external path loss.
At every frequency it fits
`input_power_dbm = slope * 20*log10(hypot(mean_i,mean_q)) + intercept`.
The QCoDeS columns `measured_value`, `meas_in_pwr`, `meas_slope`, and
`meas_intercept`, plus `Calibration_Result` and `Calibration_Config` metadata,
remain compatible with existing RF_In/DC_In runs. Input attenuation is recorded
separately and corrected when a later S-parameter run uses a different setting.
Both calibration workflows use local SQLite staging and publish the completed
run to the selected DB after a WAL checkpoint.

## Verification

```powershell
python -c "import numpy, PyQt5, pyqtgraph, matplotlib, qcodes, plottr, pyvisa; print('GUI dependencies OK')"
python DCWaveform_Generator.py
```

For development tests, install `requirements-dev.txt` and run pytest from the
source test directory.
