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
Initial physical checks of all 14 combinations used the production GUI Run
button, rep=1, DAC5 to oscilloscope CH1 at 50 ohms, with expected and raw measured
waveforms stored outside this repository. CH1 is restored to 1 megohm after
acquisition; CH4 remains at 1 megohm.

Regression results: 731 passing compiler/migration tests and 10 passing GUI
settings tests. Sixteen v3 DC-preview timing tests fail identically on the
pre-change source; the baseline reproduction is retained with the measurement
evidence. These are not represented as passing tests. No firmware was changed.
