"""Spin qubit rabi oscillation program"""
import keysight.qcs as qcs
import numpy as np

V_max_M5300 = 0.250

def rabi_continuous_read_out(
    program: qcs.Program,

    control_rf_awg: qcs.Channels,
    control_dc_awg: qcs.Channels,
    read_out_rf_awg: qcs.Channels,
    read_out_dc_awg: qcs.Channels,
    read_out_digitizer: qcs.Channels,

    read_out_rf_frequency: float,
    control_rf_frequency: float,

    t_burst: float,
    t_cb: float,
    t_m: float,
    t_c_ramp: float,
    t_r_ramp: float,
    t_wait: float,
    t_read_out: float,

    control_dc_voltage: float,
    read_out_dc_voltage: float,
    read_out_rf_voltage: float,

    p_rabi: float # dBm
) -> qcs.Program:
    """
    Rabi oscillation with continuous readout.

    Args:

        program
            The QCS program to which the sequence will be added.
        control_rf_awg
            The control RF AWG channel.
        control_dc_awg
            The control DC AWG channel.
        read_out_rf_awg
            The read-out RF AWG channel.
        read_out_dc_awg
            The read-out DC AWG channel.
        read_out_digitizer
            The read-out digitizer channel.
        read_out_rf_frequency
            The read-out RF frequency (Hz).
        control_rf_frequency
            The control RF frequency (Hz).

        t_burst
            The duration of rabi (s).
        t_cb
            The duration of DC voltage of control gate (s).
        t_m
            The duration of read-out (s).
        t_c_ramp
            The ramp time for control DC voltage (s).
        t_r_ramp
            The ramp time for read-out DC voltage (s).
        t_wait
            The waiting time between control and read-out (s).
        t_read_out
            The duration of integration filter for read-out (s).

        control_dc_voltage
            The amplitude of control DC voltage (percentage).
        read_out_dc_voltage
            The amplitude of read-out DC voltage (percentage).
        read_out_rf_voltage
            The amplitude of read-out RF voltage (percentage).

        p_rabi
            The power of Rabi (dBm).

    Returns:
        The updated QCS program with the Rabi sequence added.

    Ref :
        Crippa, A., Ezzouch, R., Aprá, A. et al. Gate-reflectometry 
        dispersive readout and coherent control of a spin qubit in silicon. 
        Nat Commun 10, 2776 (2019). https://doi.org/10.1038/s41467-019-10848-z
    """
    if t_burst > t_cb:
        raise ValueError(
            f"{t_burst} cannot be larger than {t_cb}"
        )
    v_rabi              = np.sqrt((10 ** (p_rabi/20)) * 1e-3 * 50) / V_max_M5300

    t_burst_var = qcs.Scalar(
        name            = "t_burst",
        value           = t_burst,
        dtype           = float
    )

    control_dc_waveform = qcs.DCWaveform(
        duration        = t_c_ramp * 2,
        envelope        = qcs.GaussianEnvelope(),
        amplitude       = control_dc_voltage
    )
    read_out_dc_waveform = qcs.DCWaveform(
        duration        = t_r_ramp * 2,
        envelope        = qcs.GaussianEnvelope(),
        amplitude       = read_out_dc_voltage
    )
    control_rf_waveform = qcs.RFWaveform(
        duration        = t_burst_var,
        envelope        = qcs.ConstantEnvelope(),
        amplitude       = v_rabi,
        rf_frequency    = control_rf_frequency
    )
    read_out_rf_waveform = qcs.RFWaveform(
        duration        = t_m + t_r_ramp * 2,
        envelope        = qcs.ConstantEnvelope(),
        amplitude       = read_out_rf_voltage,
        rf_frequency    =read_out_rf_frequency
    )
    read_out_int_filter = qcs.RFWaveform(
        duration        = t_read_out,
        envelope        = qcs.ConstantEnvelope(),
        amplitude       = 1.0,
        rf_frequency    = read_out_rf_frequency
    )

    program.add_waveform(
        read_out_dc_waveform.to_flattop(t_m),
        read_out_dc_awg,
        new_layer = True
    )
    program.add_waveform(
        control_dc_waveform.to_flattop(t_cb),
        control_dc_awg,
    )

    program.add_waveform(
        read_out_rf_waveform,
        read_out_rf_awg,
    )
    program.add_waveform(
        control_rf_waveform,
        control_rf_awg,
    )

    program.add_acquisition(
        read_out_int_filter,
        read_out_digitizer,
        pre_delay = (t_cb + t_c_ramp * 2 + t_wait)
    )

if __name__ == "__main__":
    MHz             = 1e6
    GHz             = 1e9
    us              = 1e-6
    ns              = 1e-9

    V_max_M5301AWG  = 5
    mV              = 1/(V_max_M5301AWG * 1000)

    control_rf_awg  = qcs.Channels(
        range(1),
        name        = "control_rf_awg",
        absolute_phase = False
    )
    control_dc_awg  = qcs.Channels(
        range(1),
        name        = "control_dc_awg",
    )
    read_out_rf_awg = qcs.Channels(
        range(1),
        name        = "read_out_rf_awg",
        absolute_phase = True
    )
    read_out_dc_awg = qcs.Channels(
        range(1),
        name        = "read_out_dc_awg"
    )
    read_out_digitizer = qcs.Channels(
        range(1),
        name        = "read_out_digitizer"
    )

    program = qcs.Program()

    rabi_continuous_read_out(
        program = program,
        control_rf_awg = control_rf_awg,
        control_dc_awg = control_dc_awg,
        read_out_rf_awg=read_out_rf_awg,
        read_out_dc_awg=read_out_dc_awg,
        read_out_digitizer=read_out_digitizer,

        read_out_rf_frequency = 300 * MHz,
        control_rf_frequency = 1.2 * GHz,

        t_burst = 100 * ns,
        t_cb = 300 * ns,
        t_m = 3 * us,
        t_c_ramp = 30 * ns,
        t_r_ramp = 30 * ns,
        t_wait = 1 * us,
        t_read_out = 100 * ns,

        control_dc_voltage = 1000 * mV,
        read_out_dc_voltage = 300 * mV,
        read_out_rf_voltage = 0.1,

        p_rabi = -15
    )

    html_str = program.render().to_html()
    with open("program_render.html", "w", encoding = "utf-8") as f:
        f.write(html_str)
        f.close()
