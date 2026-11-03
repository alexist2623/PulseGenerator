# RF filter compatibility

Output-power calibration matching resolves the requested filter and the stored
calibration metadata to `(HPF bank, HPF state, LPF bank, LPF state)`. Different
frequency inputs are compatible if these four values are identical. Different
states are incompatible even when their requested frequencies are very close.
Nyquist-zone, PCB, generator-channel and frequency-coverage checks remain active.

`qick_filter_settings.py` mirrors the current `FilterADMV8818.set_filter` driver:
first matching band, strict endpoints and division by 16. The driver's HPF3
12.60 GHz extension is retained for compatibility, not treated as a measured
12.60 GHz physical cutoff. Changing this policy requires a coordinated driver
change, not an independent calibration-loader change.

Legacy calibration JSON/configuration fields are converted in memory. Supported
representations include cutoff/center plus bandwidth, explicitly named center,
and low/high cutoff pairs (GHz). `rf_settings_actual` fields take precedence over
`configuration` fields. Existing GUI JSON `filter_cutoff`/`filter_bandwidth`
values flow through the same comparison after normal GUI restoration. Original
JSON/DB data are not rewritten. Summaries retain original frequency fields and
add derived register settings, explicitly marked as not hardware readback.

Bypass resolves to both banks bypassed and both states zero. Its unused cutoff
and bandwidth values are ignored; LPF/HPF ignore bandwidth. Missing or invalid
metadata is not assumed to mean bypass. Calls with no filter constraint retain
their legacy browsing behavior. Existing explicitly selected manual calibration
overrides are unchanged; automatic matching uses the register identity.

The production Qt Front panel now embeds `QickFilterEditor` for RF output and
readout ports. Its mode selector supports Bypass/lowpass/highpass/bandpass. The
0–8 GHz plot allows cutoff dragging; bandpass edits its low/high edges and
converts them to the existing center/bandwidth API. A center line is shown for
bandpass. Faint dotted lines show nominal cutoff estimates from selected states
and datasheet endpoints, not measured per-state cutoffs. Register identities are
shown below the plot. Bypass hides cutoff controls/lines and bypasses both banks.
Invalid requests disable Apply. DC/baseband ports disable the analog editor.

Target frequency is explicitly for visualization. Its images use converter
`fs` from the selected generator/readout configuration (MHz converted to GHz),
not the DDS or decimated FIR rate. Dragging an image moves the linked target and
other images. Missing converter-rate metadata hides images instead of inventing
a sampling rate. These display controls do not alter the experiment frequency.

Dragging edits local values only. The existing Front panel Apply action passes
them to the existing measurement/hardware-update path. Legacy JSON retains
`filter_type`, `filter_cutoff` and `filter_bandwidth`; loading does not rewrite
requests to the estimated state cutoffs. The dialog scrolls to accommodate the
graph and port controls.

For the current QICK API, `set_filter(fc=2.5, bw=1, ftype="lowpass")` selects
HPF bypass and LPF1 state 4: WR0_SW=0xC1, WR0_FILTER=0x04. Bandwidth is ignored.
The driver computes `int((2.5 - 2.05) / ((3.85 - 2.05) / 16)) == 4`.
The graph's endpoint-interpolated estimate for state 4 is 2.530 GHz; it is not a
measurement or a guarantee of the physical -3 dB cutoff.

Validation is offline: calibration/GUI tests use temporary databases and the
Qt offscreen platform; real Qt mouse events exercise cutoff and image dragging.
A saved full experiment JSON is loaded through MainWindow, edited through the
Front panel, and checked at the execution-spec boundary with hardware dispatch
replaced by a collector. A fake SPI check compares resolver bytes against the
local QICK driver. No instrument connection, real SPI write, or visible GUI is
needed. `test_qick_filter_editor.py`, `test_qick_front_panel.py`,
`test_qick_filter_settings.py` and `test_power_calibration.py` passed 53 tests on
2026-11-03. Hidden widget renders were also visually checked; the offscreen
renderer explicitly loaded Segoe UI to avoid missing default-platform glyphs.
