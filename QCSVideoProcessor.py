# -*- coding: utf-8 -*-
"""Video processor for qcs based qtt videomode"""
import numpy as np
import warnings
import logging
import copy
import qcodes
import ast

import qtt
from keysight import qcs
from qtt.measurements.videomode import VideoMode
from qtt.measurements.videomode_processor import VideoModeProcessor
from qtt.measurements.scans import makeDataset_sweep, makeDataset_sweep_2D

##############################################
# Parameter Definition
##############################################
# Keysight M5301A AWG output voltage range
V_m5301AWG  = 2.5

ns          = 1e-9
us          = 1e-6
GHz         = 1e9
MHz         = 1e6

mV          = 1/(V_m5301AWG * 1000)
V           = 1/(V_m5301AWG)

class QCSVideomodeProcessor(VideoModeProcessor):
    """
    Videomode processor for QCS
    """

    def __init__(
            self,
            station: qcodes.station,
            model_name: str,
            verbose: int = 1,
        ):
        self.station = station
        self.verbose = verbose

        # Videomode setup
        self.minstrumenthandle = "qcs"
        self.sweepparams = []
        self.sweepranges = []
        self.resolution = []
        self.channels = []

        # Measurement configuration
        self.gate_mapping = station[model_name].gate_mapping
        self.rf_mapping = station[model_name].rf_mapping
        self.dig_mapping = station[model_name].dig_mapping
        self.meas_setup = station[model_name].meas_setup

        self.scan_parameters = {}
        self.period_1d = qcodes.ManualParameter(
            'period_1d',
            initial_value = 1e-3
        )

        # QCS program setup
        self.qcs_program = qcs.Program()
        self.qcs_mapper = qcs.ChannelMapper()
        self.qcs_gates = {}
        self.qcs_rf = {}
        self.qcs_dig = {}
        self.qcs_gate_v_scalar = {}
        self.qcs_program_generated = False
        self._generate_qcs_program()
        self.qcs_program_generated = True
        self.qcs_backend = qcs.HclBackend(
            channel_mapper  = self.qcs_mapper,
            hw_demod        = True,
            init_time       = 70e-9,
            suppress_rounding_warnings = True,
            keep_progress_bar = False,
        )

    def _generate_qcs_program(self) -> None:
        """Generate qcs program"""
        # Gate AWG Channels setup
        print("============<Gate Channel Setup>============")
        for gate_name, dac_addr in self.gate_mapping.items():
            self.qcs_gates[gate_name] = qcs.Channels(
                labels      = 0,
                name        = gate_name,
            )
            self.qcs_mapper.add_channel_mapping(
                channels    = self.qcs_gates[gate_name],
                addresses   = [dac_addr],
                instrument_types = qcs.InstrumentEnum.M5301AWG,
            )
            print(gate_name + " : " + str(dac_addr))

        print("============<RF Channel Setup>============")
        for rf_name, rf_addr in self.rf_mapping.items():
            self.qcs_rf[rf_name] = qcs.Channels(
                labels      = 0,
                name        = rf_name,
                absolute_phase = True,
            )
            self.qcs_mapper.add_channel_mapping(
                self.qcs_rf[rf_name],
                [rf_addr],
                qcs.InstrumentEnum.M5300AWG,
            )
            print(rf_name + " : " + str(rf_addr))

        print("============<Digitizer Channel Setup>============")
        for dig_name, dig_addr in self.dig_mapping.items():
            self.qcs_dig[dig_name] = qcs.Channels(
                labels      = 0,
                name        = dig_name,
                absolute_phase = True
            )
            self.qcs_mapper.add_channel_mapping(
                self.qcs_dig[dig_name],
                [dig_addr],
                qcs.InstrumentEnum.M5200Digitizer
            )
            print(dig_name + " : " + str(dig_addr))

        # qcs Scalar declaration for sweep
        for gate_name, _ in self.gate_mapping.items():
            self.qcs_gate_v_scalar[gate_name] = qcs.Scalar(
                name  = gate_name + "_amp",
                dtype = float,
                value = self.station.pgates.parameters[gate_name].get() * mV
            )
        print("====================================")

        meas = self.meas_setup
        # DC Waveform setup
        for gate_name, _ in self.gate_mapping.items():
            waveform = qcs.DCWaveform(
                name = gate_name + "_waveform",
                duration = meas["gate_time"] + meas["acquisition_time"],
                envelope = qcs.ConstantEnvelope(),
                amplitude= self.qcs_gate_v_scalar[gate_name]
            )
            self.qcs_program.add_waveform(
                waveform,
                self.qcs_gates[gate_name]
            )

        # RF Waveform
        for _, meas_param in meas["meas_params"].items():
            # Connect downconverter with digitizer if exist.
            if hasattr(meas_param, "downconverter"):
                self.qcs_mapper.add_downconverters(
                    self.dig_mapping[meas_param["dig"]],
                    ast.literal_eval(meas_param["downconverter"])
                )
                self.qcs_mapper.set_lo_frequencies(
                    [
                        ast.literal_eval(meas_param["downconverter"]),
                        self.rf_mapping[meas_param["rf"]]
                    ],
                    meas["frequency"] - 50e6 # Set lo frequency below 50 MHz from
                                             # target frequency since digitizer 
                                             # hardware demodulation does not support
                                             # DC input (minimum frequency : 20 MHz)
                )
            else:
                # If there is no downconverter, directly generate RF signal using
                # M5300A's DDS.
                self.qcs_mapper.set_lo_frequencies(
                    [
                        self.rf_mapping[meas_param["rf"]]
                    ],
                    0
                )
            waveform = qcs.RFWaveform(
                name        = meas_param["rf"] + "_waveform",
                duration    = meas["acquisition_time"],
                envelope    = qcs.GaussianEnvelope(),
                amplitude   = 1.0,
                rf_frequency= meas["frequency"]
            )
            self.qcs_program.add_waveform(
                pulse       = qcs.Delay(meas["gate_time"]),
                channels    = self.qcs_rf[meas_param["rf"]]
            )
            self.qcs_program.add_waveform(
                pulse       = waveform,
                channels    = self.qcs_rf[meas_param["rf"]]
            )
            self.qcs_program.add_acquisition(
                integration_filter = waveform,
                channels    = self.qcs_dig[meas_param["dig"]],
                pre_delay   = meas["gate_time"]
            )
            self.channels.append(meas_param["dig"])
        self.unique_channels = list(np.unique(self.channels))
        
        # Calculate Physical gate voltages
        for gate_name, sweep_value in meas["sweepparams"].items():
            # "V" + gate_name -> virtual gate name
            self.sweepparams.append("V" + gate_name)
            self.sweepranges.append(sweep_value["range"])
            self.resolution.append(sweep_value["resolution"])

        # Setup sweep array.
        # Sweep range : [-range/2 + current DC, +range/2 + current DC]
        qcs_pgates_sweep_arrays = []
        for pgate_name, _ in self.gate_mapping.items():
            a = (
                self.station.gates.get_crosscap_map()[self.sweepparams[0]][pgate_name] *
                (
                    np.arange(self.resolution[0]) * self.sweepranges[0] /
                    (self.resolution[0]-1) - self.sweepranges[0]/2
                )
            )
            b = (
                self.station.gates.get_crosscap_map()[self.sweepparams[1]][pgate_name] *
                (
                    np.arange(self.resolution[1]) * self.sweepranges[1] /
                    (self.resolution[1]-1) - self.sweepranges[1]/2
                )
            )
            qcs_pgates_sweep_arrays.append(
                qcs.Array(
                    name = pgate_name + "_array",
                    value = (
                        b[:, None] + a[None, :] +
                        self.station.pgates.parameters[pgate_name].get()
                    ).ravel() * mV
                )
            )
        # Sweep physical gate parallely
        self.qcs_program.sweep(
            qcs_pgates_sweep_arrays,
            list(self.qcs_gate_v_scalar.values())
        )
        # Repeat number of n_shots
        self.qcs_program.n_shots(meas["n_shots"])
        self.n_shots = meas["n_shots"]

    def plot_title(self, index: int) -> str:
        """Make plot title"""
        plot_title = str(self.minstrumenthandle) + ' ' + str(self.channels[index])
        return plot_title

    def create_dataset(self, processed_data, metadata):
        alldata = [None] * len(processed_data)
        if processed_data.ndim == 2:
            for jj, data_block in enumerate(processed_data):
                dataset, _ = makeDataset_sweep(
                    data_block,
                    self.sweepparams,
                    self.sweepranges[0],
                    gates=self.station.gates,
                    loc_record={'label': 'videomode_1d'}
                )
                dataset.metadata = copy.copy(metadata)
                alldata[jj] = dataset
        elif processed_data.ndim == 3:
            for jj, data_block in enumerate(processed_data):
                dataset, _ = makeDataset_sweep_2D(
                    data_block,
                    self.station.gates,
                    self.sweepparams,
                    self.sweepranges,
                    loc_record={'label': 'videomode_2d'}
                )
                dataset.metadata = copy.copy(metadata)
                alldata[jj] = dataset
        else:
            raise Exception('makeDataset: data.ndim %d' % processed_data.ndim)

        return alldata

    def update_position(self, position, verbose=1):
        if verbose:
            print('# %s: update position: %s' % (self.__class__.__name__, position,))

        station = self.station
        if self.scan_dimension() == 1:
            delta = position[0]
            if isinstance(self.scan_parameters['sweepparams'], str):
                param = getattr(station.gates, self.scan_parameters['sweepparams'])
                param.set(delta)
                if verbose > 2:
                    print('  set %s to %s' % (param, delta))
                return
        try:
            for ii, parameter in enumerate(self.scan_parameters['sweepparams']):
                delta = position[ii]
                if verbose:
                    print('param %s: delta %.3f' % (parameter, delta))

                if isinstance(parameter, str):
                    if parameter == 'gates_horz' or parameter == 'gates_vert':
                        d = self.scan_parameters['sweepparams'][parameter]
                        for gate, factor in d.items():
                            if verbose >= 2:
                                print('  %s: increment %s with %s' % (parameter, gate, factor * delta))
                            param = getattr(station.gates, gate)
                            param.increment(factor * delta)
                    else:
                        if verbose > 2:
                            print('  set %s to %s' % (parameter, delta))
                        param = getattr(station.gates, parameter)
                        param.set(delta)
                else:
                    raise Exception('_update_position with parameter type %s not supported' % (type(parameter),))
        except Exception as ex:
            logging.exception(ex)

    def ppt_notes(self):
        return str(self.scan_parameters)

    def initialize(self, videomode: VideoMode) -> None:
        """If qcs program is not setup, generate it"""
        if not self.qcs_program_generated:
            self._generate_qcs_program()

    def stop(self):
        """Stop the measurement."""
        pass

    def measure(self, videomode: VideoMode) -> np.array:
        """Measure the reflected signal sweeping the gate."""
        device_parameters = {}
        self._device_parameters = device_parameters

        # Calculate physical gate voltage values. We donnot make entire program again,
        # but update parameters of program.
        for pgate_name, _ in self.gate_mapping.items():
            a = (
                self.station.gates.get_crosscap_map()[self.sweepparams[0]][pgate_name] *
                (
                    np.arange(self.resolution[0]) * self.sweepranges[0] /
                    (self.resolution[0]-1) - self.sweepranges[0]/2
                )
            )
            b = (
                self.station.gates.get_crosscap_map()[self.sweepparams[1]][pgate_name] *
                (
                    np.arange(self.resolution[1]) * self.sweepranges[1] /
                    (self.resolution[1]-1) - self.sweepranges[1]/2
                )
            )
            arr = getattr(self.qcs_program.variables, pgate_name + "_array")
            arr.value = (
                b[:, None] + a[None, :] +
                self.station.pgates.parameters[pgate_name].get()
            ).ravel() * mV

        # Run QCS program and get the result
        result = qcs.Executor(self.qcs_backend).execute(self.qcs_program)
        data = []
        # Note that result.get_iq() will make dataframe, and it take quite large time
        # (10000 sweep -> ~ 5 secs). So, result.results.get_iq() should be used to avoid
        # making dataframe.
        result = result.results.get_iq()

        # Reshape the measured result
        for _, meas_params in self.meas_setup["meas_params"].items():
            data.append(
                np.abs(result[self.qcs_dig[meas_params["dig"]]]).reshape(
                    self.n_shots,
                    self.resolution[1],
                    self.resolution[0]
                ).mean(axis=0)
            )
        if np.all(data == 0):
            raise Exception('data returned contained only zeros, aborting')
        return data

    def process(self, measurement_data, videomode):
        measurement_data_selected = []
        for _, channel in enumerate(self.channels):
            unique_channel_idx = self.unique_channels.index(channel)
            data_processed = np.array(measurement_data[unique_channel_idx])
            measurement_data_selected.append(data_processed)

        dd = self.default_processing(measurement_data_selected, videomode)
        return dd

    def parse_instrument(self, measurement_instrument_handle, sample_rate):
        # TODO Calculate FPS
        return self.sampling_frequency

    def scan_dimension(self):
        if isinstance(self.sweepranges, (float, int)):
            return 1
        elif isinstance(self.sweepranges, list):
            if len(self.sweepranges) == 2:
                return 2
            elif len(self.sweepranges) == 1:
                return 1
            else:
                raise Exception('scan dimension not supported')
        else:
            return -1

    def extend_videomode_name(self, name):
        """ String to append to VideoMode name """
        if name is not None:
            if isinstance(self.sweepparams, (str, list)):
                name += ': %s' % str(self.sweepparams)
        return name
