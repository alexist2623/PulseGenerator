from qstl_program import Program
from qstl_channel import (
    ChannelMapper,
    Channels,
    SingleVirtualChannel,
    InstrumentEnum
)
from qstl_waveform import (
    Delay,
    ConstantEnvelope,
    GaussianEnvelope,
    DCWaveform,
    RFWaveform,
    Synchronize
)
from qick import *

class Executor:
    def __init__(
        self,
        soc: QickConfig,
        channel_mapper: ChannelMapper
    ):
        self._soc = soc
        self._channel_mapper = channel_mapper
        self._qick_program: QickProgram = None
        self._qstl_program: Program = None

    def execute(self, program: Program):
        r"""
        Generate QICK program and run
        """
        self._qick_program = self.make_program(program)

    def make_program(self, program: Program) -> QickProgram:
        r"""
        Convert QSTL program to QICK program
        """
        _prog = QickProgram(self._soc)
        for op in program.operations:
            if isinstance(op, Synchronize):
                _prog.sync_all()
                for ch in self._channel_mapper.out_channel_map.keys():
                    ch.last_time = 0
                for ch in self._channel_mapper.in_channel_map.keys():
                    ch.last_time = 0
            elif op[0] in self._channel_mapper.out_channel_map:
                channel, operation = op
                if isinstance(operation, Delay):
                    channel.last_time += self._soc.us2cycles(operation.duration * 1e6, 1, None)
                elif isinstance(operation, DCWaveform):
                    pass
                elif isinstance(operation, RFWaveform):
                    if self.get_instrument_type(channel) is InstrumentEnum.RF:
                        if isinstance(operation.envelope, ConstantEnvelope):
                            _prog.setup_and_pulse(
                                ch      = self.get_physical_channel(channel),
                                style   = "const",  # Output is gain * DDS output
                                freq    = self._soc.freq2reg(operation.rf_frequency * 1e6, 0, 0),
                                phase   = operation.instantaneous_phase,
                                gain    = int(operation.amplitude * (32767)),
                                length  = self._soc.us2cycles(operation.duration * 1e6, 1, None),
                                phrst   = 1 if channel.absolute_phase is True else 0,
                                t       = channel.last_time,
                            )
                        elif isinstance(operation.envelope, GaussianEnvelope):
                            _prog.setup_and_pulse(
                                ch      = self.get_physical_channel(channel),
                                style   = "arb",
                                freq    = self._soc.freq2reg(operation.rf_frequency * 1e6, 0, 0),
                                phase   = operation.instantaneous_phase,
                                gain    = int(operation.amplitude * (32767)),
                                phrst   = 1 if channel.absolute_phase is True else 0,
                                outsel  = "product",
                                waveform= operation.name,
                                t       = channel.last_time,
                            )
                        channel.last_time += self._soc.us2cycles(operation.duration * 1e6, 1, None)
                elif op[0] in self._channel_mapper.in_channel_map:
                    # TODO
                    pass
                else:
                    raise NotImplementedError(f"Operation {operation} not implemented")
        return _prog

    def get_physical_channel(self, channel: SingleVirtualChannel) -> int:
        r"""
        Return physical channel address of virtual channel
        """
        return self._channel_mapper.get_physical_channel(channel).addr

    def get_instrument_type(self, channel: SingleVirtualChannel) -> InstrumentEnum:
        r"""
        Return instrument type of virtual channel
        """
        return self._channel_mapper.get_physical_channel(channel).inst_type
