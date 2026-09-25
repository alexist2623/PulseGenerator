# Shared DAC current and voltage scale

The selected output in the shared Front Panel displays the physical RFDC DAC's
full-scale current in mA. `Apply DAC current` writes through the QICK RPC server
and reads the actual quantized current back. Every experiment tab uses that same
per-converter state. Selecting a different output does not copy the previous
output's current. Loading a settings file does not write to hardware.

Eligibility follows generator IP connectivity, independently of the daughter
card: `axis_awg_tuning*` and `axis_square_pulse*` are DC waveform sources.
RF DDS generators are read-only. A physical DAC shared with any RF generator
is also read-only. The server enforces these restrictions as well as the GUI.

Voltage conversion preserves the existing reference of **20 mA = +/-800 mV**:

    full_scale_mv = 800 * actual_current_uA / 20000
    normalized_DAC_amplitude = requested_physical_mv / full_scale_mv

Thus 10 mA gives +/-400 mV, and 32 mA gives +/-1280 mV. These are estimates from
the existing reference under unchanged load/output circuitry, not measured
voltage calibration. Current changes do not improve the effective DAC bit depth.

The former editable full-scale controls in Experiment, Python export and
QICK Square Wave have been removed from the UI. Virtual voltage coordinates
remain in mV. Cross-capacitance is applied in voltage coordinates before each
physical channel's DAC normalization. SET, RAMP, voltage/duration sweeps,
fixed-voltage and fixed-time DC compensation, RC headroom checks, Stability,
SquarePulse and exported Python all use the physical channel scale. Metadata
retains the common virtual-coordinate reference and the individual physical
full scales. Sweep range checks inspect Cartesian endpoints, not the whole grid.

Actual current and routing are rechecked before loading/running a program.
A changed or unreadable current blocks execution of stale DAC codes. Current
updates are blocked while a GUI hardware task is active; generate/start the next
program after changing current. This is not a real-time current sweep mechanism.

## Board support

Update the board QSTL_QICK Python library and restart its Pyro server to expose
`get_dac_current_settings()` and `set_dac_current(converter_id, current_uA)`.
The setter returns fresh hardware readback, not the requested value. No FPGA RTL
change is needed for the software scaling.

Gen 3/DFE RFDC and PYNQ `SetDACVOP` are required. This DC control supports
6.425--32 mA. Legacy DAC compatibility mode is read-only: enabling VOP requires
DAC_AVTT=3.0 V. The software never changes the board supply or silently disables
compatibility mode. The checked-in HWH describes the initial firmware VOP
configuration, not the live DAC mode after software initialization. Existing
board code may already have enabled Gen 3 mode. Only the live `DACCompMode`
read determines whether the button is enabled; the HWH value does not disable
the control. The board library update exposes the new GUI RPC methods; it does
not introduce the underlying `SetDACVOP` API, which may already be available.
Older servers
without the new API keep the +/-800 mV legacy reference and show current control
as unavailable. If an updated server reports a DC current read failure, output
is blocked until identification succeeds.

RTL verification checks generated code, tProcessor/TMUX commands, DAC samples,
and the RC recurrence. It models analog current-to-voltage gain using the above
reference; it does not measure physical RFDC current or external voltage.
