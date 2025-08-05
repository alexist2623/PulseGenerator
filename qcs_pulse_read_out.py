"""Spin qubit pulse read out program"""
import keysight.qcs as qcs

def read_out(
        program: qcs.Program,
        read_out_rf_awg: qcs.Channels,
        read_out_digitizer: qcs.Channels,
        read_out_rf_frequency: float,
        read_out_duration: float,
        read_out_delay: float,
) -> qcs.Program:
    """
    Function which makes read out sequence.
    """
    read_out_duration_var = qcs.Scalar(
        name = "read_out_duration_var",
        value = read_out_duration,
        dtype = float
    )
    read_out_rf_frequency_var = qcs.Scalar(
        name = "read_out_rf_frequency_var",
        value = read_out_rf_frequency,
        dtype = float
    )


    read_out_rf_pulse = qcs.RFWaveform(
        duration = read_out_duration_var,
        envelope = qcs.ConstantEnvelope(),
        rf_frequency = read_out_rf_frequency_var,
        amplitude = 1.0
    )
    program.add_waveform(
        pulse = read_out_rf_pulse,
        channels = read_out_rf_awg,
        new_layer = True
    )
    program.add_acquisition(
        pulse = read_out_rf_pulse,
        channels = read_out_digitizer,
        pre_delay = read_out_delay
    )

    return program
