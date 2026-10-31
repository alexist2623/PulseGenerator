# Settings v48: legacy import and v7 terminal hold

Loading a supported older settings JSON converts it in memory to the current
format and displays a non-modal **Settings updated** notice. The source file is
unchanged until the user saves. Pulse vertices, fine-ramp metadata, sweep groups,
hold times, RF delays, and compensation settings pass through the existing
version-aware decoder. Current v48 files do not show a migration notice.

The original `qick-awg-waveform-recipe-v1` SET/RAMP recipe can also be imported.
Clock cycles and normalized voltages are converted to GUI units, and voltage
sweeps and the virtual-gate matrix are retained. Recipes lack connection and RF
acquisition settings: existing AWG channel assignments are retained, while RF
and acquisition are disabled. Unsupported recipe sweep forms are rejected
explicitly instead of being discarded. Import never connects to hardware.

On AWG v7 a completed RAMP holds its terminal target indefinitely. The compiler
therefore omits a following SET to the same target. The logical hold segment
remains in the GUI and in timing/sweep calculations. Initial SETs, standalone
voltage changes, compensation controls, and zero/reset commands still execute.
Older firmware retains its existing command sequence.

Shared-port validation and executed-command metadata use the same omission rule.
Fine RF events are checked using their exact dependent timing rows; their
placeholder affine timestamps are not used to reject a preceding ramp sweep.

Validation for the received legacy corpus includes 12 source JSON files (the
previously excluded error.json is omitted), 14 separate RF routing combinations,
and 71,785 sweep coordinates in the strict tProcessor behavior model. All 14
combinations compile and execute in that model without timestamp conflicts.
Physical verification completed on 2026-10-30 using the production GUI Run
button, rep=1, DAC5 to oscilloscope CH1 at 50 ohms. Each of the 14 combinations
has 100 complete detailed waveform records: 1,400 records covering 1,043 unique
source coordinates. Single-point files use 100 separate rep=1 runs; the
40-point combination repeats selected blocks. Actual PMEM and runtime DMEM
matched the compiled references for every acquisition.

An additional 112 narrow RF acquisitions at a native 0.1 ns interval all match
the requested carrier within max(0.6 MHz, two FFT bins). These include 100
distinct coordinates across the 200-1,500 MHz sweep and one representative
coordinate from each other RF combination. This is not a calibrated RF power
test. Full-frame measurements use a wider voltage range and 0.4-4 ns sample
intervals; they do not resolve every 0.208333 ns edge or every RF carrier.

Stable-envelope RMSE medians range from 9.50 to 13.79 mV across the 14 cases;
the largest individual RMSE is 28.23 mV. Successful execution is not a claim
of exact analog agreement. The September 10 file was reacquired with a positive
DC-edge trigger after its initial trigger selected the second pulse. The
temporary-file case was reacquired at -75 mV after its low threshold produced
a noisy time origin. Superseded records are excluded from the final counts.

Expected and raw measured waveforms, settings, scope logs and the 701-page
complete atlas remain outside this repository. Original source JSON hashes
are unchanged. Final zero/reset completed, and independent readback confirmed
unchanged DAC currents, CH1/CH2/CH4 at 1 megohm, and no scope error.

Regression results: 731 passing compiler/migration tests and 10 passing GUI
settings tests. Sixteen v3 DC-preview timing tests fail identically on the
pre-change source; the baseline reproduction is retained with the measurement
evidence. These are not represented as passing tests. No firmware was changed.
