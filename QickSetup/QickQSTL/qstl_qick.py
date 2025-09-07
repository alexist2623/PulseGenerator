from qstl_program import Program
from qstl_channel import (
    ChannelMapper,
    Channels,
    SingleVirtualChannel,
    InstrumentEnum
)
from qstl_waveform import (
    Delay,
    HardwareOperation,
    ConstantEnvelope,
    GaussianEnvelope,
    DCWaveform,
    RFWaveform,
    Synchronize
)
from qstl_variable import (
    Scalar,
)
from qick import *

class Executor:
    r"""
    QICK Executor class to convert QSTL programs to QICK programs and run them
    """
    def __init__(
        self,
        soc: QickConfig,
        channel_mapper: ChannelMapper
    ):
        # QICK SoC object
        self._soc = soc
        # Virtual to physical channel mapper
        self._channel_mapper = channel_mapper
        # QICK program object
        self._qick_program: QickProgram = None
        # QSTL program object
        self._qstl_program: Program = None
        # Map which converts Scalar variables to QICK register values
        self._scalar_val_map: dict[Scalar, function] = {}
        self._scalar_reg_map: dict[Scalar, int] = {}
        self._time_reg_map: dict[SingleVirtualChannel, tuple[int, int]] = {}
        self._duration_reg_map: dict[SingleVirtualChannel, int] = {}
        for ch in self._channel_mapper.out_channel_map.keys():
            last_reg_num = 3
            for _ch in self._time_reg_map.keys():
                if self.get_physical_channel(_ch) == self.get_physical_channel(ch):
                    last_reg_num = last_reg_num + 1
            if last_reg_num > 12:
                raise ValueError(f"Channel {ch} has too many operations (>12) to fit in QICK timing registers.")
            self._time_reg_map[ch] = (self.get_physical_channel(ch), last_reg_num)
        
        for ch in self._channel_mapper.out_channel_map.keys():
            reg_nums = []
            for _ch in self._time_reg_map.keys():
                if self.get_physical_channel(_ch) == self.get_physical_channel(ch):
                    reg_nums.append(self._time_reg_map[_ch][1])
            last_reg_num = max(reg_nums) + 1
            for _ch in self._duration_reg_map.keys():
                if self.get_physical_channel(_ch) == self.get_physical_channel(ch):
                    last_reg_num = last_reg_num + 1
            if last_reg_num > 12:
                raise ValueError(f"Channel {ch} has too many operations (>12) to fit in QICK timing registers.")
            self._duration_reg_map[ch] = (self.get_physical_channel(ch), last_reg_num)

    def execute(self, program: Program):
        r"""
        Generate QICK program and run
        """
        self.make_program(program)

    def make_program(self, program: Program) -> None:
        r"""
        Convert QSTL program to QICK program
        """
        self._qick_program = QickProgram(self._soc)
        for op in program.operations:
            if isinstance(op, Synchronize):
                self._qick_program.sync_all()
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
                        (freq, phase, gain, length) = self.setup_pulse_regs(operation, out=True)
                        channel.last_time += self._soc.us2cycles(operation.duration * 1e6, 1, None)
                elif op[0] in self._channel_mapper.in_channel_map:
                    # TODO
                    pass
                else:
                    raise NotImplementedError(f"Operation {operation} not implemented")
        return

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
    
    def setup_pulse_regs(
        self,
        channel: SingleVirtualChannel,
        operation: HardwareOperation,
        out: bool,
    ) -> tuple[int, int, int, int]:
        r"""
        Map pulse parameters to QICK register values
        """
        freq    = operation.rf_frequency
        phase   = operation.instantaneous_phase
        gain    = operation.amplitude
        length  = operation.duration

        if out is True:
            # Map Scalar variables to QICK register value functions
            if isinstance(freq, Scalar) and freq not in self._scalar_val_map:
                self._scalar_val_map[freq] = lambda x: self._soc.freq2reg(x * 1e6, 0, 0)
                freq = freq.get_value()
            if isinstance(phase, Scalar) and phase not in self._scalar_val_map:
                self._scalar_val_map[phase] = lambda x: self._soc.deg2reg(x, 0, 0)
                phase = phase.get_value()
            if isinstance(gain, Scalar) and gain not in self._scalar_val_map:
                self._scalar_val_map[gain] = lambda x: int(x * (32767)) & 0xFFFF
                gain = gain.get_value()
            if isinstance(length, Scalar) and length not in self._scalar_val_map:
                self._scalar_val_map[length] = lambda x: self._soc.us2cycles(x * 1e6, 1, None)
                length = length.get_value()

            freq_reg    = self._soc.freq2reg(operation.rf_frequency * 1e6, 0, 0)
            phase_reg   = self._soc.deg2reg(phase, 0, 0)
            gain_reg    = int(gain * (32767)) & 0xFFFF
            length_reg  = self._soc.us2cycles(length * 1e6, 1, None) & 0xFFFF
            ch          = self.get_physical_channel(channel)
            
            if isinstance(operation.envelope, ConstantEnvelope):
                self._qick_program.set_pulse_registers(
                    ch      = ch,
                    style   = "const",
                    # Can be setup with qstl.Scalar
                    freq    = freq_reg,
                    phase   = phase_reg,
                    gain    = gain_reg,
                    length  = length_reg,
                    phrst   = 1 if channel.absolute_phase is True else 0,
                )
            elif isinstance(operation.envelope, GaussianEnvelope):
                self._qick_program.set_pulse_registers(
                    ch      = ch,
                    style   = "arb",
                    # Can be setup with qstl.Scalar
                    freq    = freq_reg,
                    phase   = phase_reg,
                    gain    = gain_reg,
                    phrst   = 1 if channel.absolute_phase is True else 0,
                    outsel  = "product",
                    waveform= operation.name,
                )
            rp, rt = self._qick_program._gen_regmap[(ch, "t")]
            next_pulse = self._gen_mgrs[ch].next_pulse
            for regs in next_pulse['regs']:
                self._qick_program.set(
                    self.soccfg['gens'][ch]['tproc_ch'],
                    rp,
                    *regs,
                    rt,
                    f"ch = {ch}, pulse @t = ${rt}"
                )
                rm = self.get_gen_reg(
                    gen_ch = self.get_physical_channel(channel),
                    name = "mode"
                )
                rl = self._qick_program.bitw(rp, rl, rm, "&", 0xFFFF)
                self._qick_program.math(rp, rt, rt, "+", rl)

            # Map Scalar variables to QICK register addresses
            if isinstance(freq, Scalar) and freq not in self._scalar_reg_map:
                self._scalar_reg_map[freq]  = self._qick_program._gen_regmap[(ch, "freq")]
            if isinstance(phase, Scalar) and phase not in self._scalar_reg_map:
                self._scalar_reg_map[phase]  = self._qick_program._gen_regmap[(ch, "phase")]
            if isinstance(gain, Scalar) and gain not in self._scalar_reg_map:
                self._scalar_reg_map[gain]  = self._qick_program._gen_regmap[(ch, "gain")]
            if isinstance(length, Scalar) and length not in self._scalar_reg_map:
                self._scalar_reg_map[length]  = self._qick_program._gen_regmap[(ch, "mode")]

        return freq_reg, phase_reg, gain_reg, length_reg
