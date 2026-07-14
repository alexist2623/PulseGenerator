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

Each database run stores `I`, `Q`, magnitude, and phase against Cartesian sweep
point, sweep amplitudes, repetition, sample index, and time in microseconds.
The run metadata includes the complete GUI settings, virtual and physical
output waveforms, cross-capacitance matrix, RF output/readout settings, QICK
connection settings, and the compiled program summary.

## Verification

```powershell
python -c "import numpy, PyQt5, pyqtgraph, matplotlib, qcodes; print('GUI dependencies OK')"
python DCWaveform_Generator.py
```

For development tests, install `requirements-dev.txt` and run pytest from the
source test directory.
