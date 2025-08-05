"""Spin qubit pulse read out program"""
import keysight.qcs as qcs

def spin_echo(
        program: qcs.Program,
        control_rf_awg: qcs.Channels,
        rf_frequency: float,
        t: float,
        T: float
) -> qcs.Program:
    pass
