"""QCS 2D Scan for Charge stability"""
import keysight.qcs as qcs
import numpy as np
import time
from pprint import pprint

##############################################
# Parameter Definition
##############################################
V_m5301AWG  = 2.5

ns          = 1e-9
us          = 1e-6
GHz         = 1e9
MHz         = 1e6

mV          = 1/(V_m5301AWG * 1000)
V           = 1/(V_m5301AWG)

rf_amp      = 1.0
rf_duration = 1 * us
rf_freq     = 1.0 * GHz

dc_duration = 10 * us

##############################################
# Mapper, program instantiation
##############################################
mapper  = qcs.ChannelMapper()
program = qcs.Program()

##############################################
# Channel Definition
##############################################
# For control qubit
ctrl_awgs   = qcs.Channels(
    range(2),
    "ctrl_awgs"
)
# For RF reflectometry via sensor dot
meas_awgs   = qcs.Channels(
    range(2),
    "meas_awgs"
)
# To make dc voltage
dc_awgs     = qcs.Channels(
    range(4),
    "dc_awgs"
)
# For RF reflectometry
digs        = qcs.Channels(
    range(4),
    "digs"
)

##############################################
# Channel mapping
##############################################
mapper.add_channel_mapping(
    channels = ctrl_awgs,
    addresses = [
        (1,4,1), (1,4,2)
    ],
    instrument_types = qcs.InstrumentEnum.M5300AWG
)
mapper.add_channel_mapping(
    channels = meas_awgs,
    addresses = [
        (1,4,3), (1,4,4)
    ],
    instrument_types = qcs.InstrumentEnum.M5300AWG
)
mapper.add_channel_mapping(
    channels = dc_awgs,
    addresses = [
        (1,7,1), (1,7,2), (1,7,3), (1,7,4)
    ],
    instrument_types = qcs.InstrumentEnum.M5301AWG
)
mapper.add_channel_mapping(
    channels = digs,
    addresses = [
        (1,18,1), (1,18,2), (1,18,3), (1,18,4)
    ],
    instrument_types = qcs.InstrumentEnum.M5200Digitizer
)

mapper.set_lo_frequencies(
    addresses= [
        (1,4,1), (1,4,2), (1,4,3), (1,4,4), 
    ],
    lo_frequency = 1.0* GHz
)

##############################################
# Reflectometry waveform & integration filter
##############################################
rf_reflect      = qcs.RFWaveform(
    rf_duration,
    qcs.GaussianEnvelope(),
    1.0,
    rf_freq,
    0.0
)
##############################################
# DC Voltage (Charge stability voltage)
##############################################
plunge_voltage1 = qcs.Scalar(
    name        = "plunge_voltage1",
    value       = 100 * mV,
    dtype       = float
)
plunge_voltage2 = qcs.Scalar(
    name        = "plunge_voltage2",
    value       = 100 * mV,
    dtype       = float
)
plunge_voltage3 = qcs.Scalar(
    name        = "plunge_voltage3",
    value       = 100 * mV,
    dtype       = float
)

plunge_voltages1= qcs.Array(
    name        = "plunge_voltages1",
    value       = np.array(
        [(100 + x) * mV for x in range(100)]
    ),
    dtype       = float,
)
plunge_voltages2= qcs.Array(
    name        = "plunge_voltages2",
    value       = np.array(
        [(100 + x) * mV for x in range(100)]
    ),
    dtype       = float,
)

plunge_waveform1 = qcs.DCWaveform(
    duration    = dc_duration,
    name        = "plunge_waveform1",
    envelope    = qcs.ConstantEnvelope(),
    amplitude   = plunge_voltage1
)
plunge_waveform2 = qcs.DCWaveform(
    duration    = dc_duration,
    name        = "plunge_waveform2",
    envelope    = qcs.ConstantEnvelope(),
    amplitude   = plunge_voltage2
)
plunge_waveform3 = qcs.DCWaveform(
    duration    = dc_duration,
    name        = "plunge_waveform3",
    envelope    = qcs.ConstantEnvelope(),
    amplitude   = plunge_voltage3
)
##############################################
# Program
##############################################
program.add_waveform(
    plunge_waveform1,
    dc_awgs[0]
)
program.add_waveform(
    plunge_waveform2,
    dc_awgs[1]
)
program.add_waveform(
    plunge_waveform3,
    dc_awgs[2]
)

program.add_waveform(
    qcs.Delay(dc_duration-rf_duration),
    meas_awgs[1]
)
program.add_waveform(
    rf_reflect,
    meas_awgs[1]
)
program.add_acquisition(
    rf_reflect,
    digs[3],
    pre_delay = dc_duration-rf_duration
)

##############################################
# Sweep parameter
##############################################
program.sweep(plunge_voltages1, plunge_voltage1)
program.sweep(plunge_voltages2, plunge_voltage2)
program.n_shots(1)

backend = qcs.HclBackend(
    channel_mapper  = mapper,
    hw_demod        = True,
    init_time       = 60 * ns,
    keep_progress_bar= False,
    suppress_rounding_warnings= True
)

for i in range(30):
    start_time = time.time()
    # print(program.variables.plunge_voltages2)
    program.variables.plunge_voltages2.value = np.array(
        [(300 + x) * mV for x in range(100)]
    )
    # print(program.variables.plunge_voltages2)
    # raise Exception
    program.variables.plunge_voltage3.value = (i + 300) * mV
    result = qcs.Executor(backend).execute(program)
    iq_val = [np.abs(result.get_iq().to_numpy().reshape(100,100))]
    print(iq_val)
    break
    end_time = time.time()

    print("execute time : {:.2f}".format(end_time-start_time))
