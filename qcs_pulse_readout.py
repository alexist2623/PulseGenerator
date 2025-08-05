"""Spin qubit pulse read out program"""
import keysight.qcs as qcs

def read_out(
        program: qcs.Program,
        readout_awg: qcs.Channels,
        readout_digitizer: qcs.Channels
) -> qcs.Program:
    pass