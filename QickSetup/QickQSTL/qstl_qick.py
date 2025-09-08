"""QSTL Program to QICK Program Executor"""
from typing import (
    Optional,
    Any,
    Callable
)
import numpy as np

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

# Reg[0:7][3:11] is free.

ns = 1e-9
us = 1e-6
ms = 1e-3
s  = 1.0
GHz = 1e9
MHz = 1e6
kHz = 1e3
Hz  = 1.0

class DummyChannel:
    r"""
    Dummy channel class to allow Executor to run without generating a QICK program
    """
    def __init__(self, rp: int = 0):
        self.next_pulse = {"regs": []}
        self.rp = rp
        self.ch = rp
        self.regmap = {
            (x, "0"): (x, 0) for x in range(8)
        }
        self.regmap.update({
            (x, "freq"): (x, 21) for x in range(8)
        })
        self.regmap.update({
            (x, "phase"): (x, 22) for x in range(8)
        })
        self.regmap.update({
            (x, "gain"): (x, 23) for x in range(8)
        })
        self.regmap.update({
            (x, "mode"): (x, 24) for x in range(8)
        })
        self.regmap.update({
            (x, "addr"): (x, 25) for x in range(8)
        })
        self.regmap.update({
            (x, "addr2"): (x, 26) for x in range(8)
        })
        self.regmap.update({
            (x, "gain2"): (x, 27) for x in range(8)
        })
        self.regmap.update({
            (x, "mode2"): (x, 28) for x in range(8)
        })
        self.regmap.update({
            (x, "mode3"): (x, 29) for x in range(8)
        })

    def set_reg(self, reg, value, defaults=None) -> None:
        print(f"DummyChannel set_reg called with reg={reg}, value={value}, defaults={defaults}")

    def get_mode_code(self, length, mode=None, outsel=None, stdysel=None, phrst=None):
        """Creates mode code for the mode register in the set command, by setting flags and adding the pulse length.

        Parameters
        ----------
        length : int
            The number of DAC fabric cycles in the pulse
        mode : str
            Selects whether the output is "oneshot" or "periodic". The default is "oneshot".
        outsel : str
            Selects the output source. The output is complex. Tables define envelopes for I and Q.
            The default is "product".

            * If "product", the output is the product of table and DDS. 

            * If "dds", the output is the DDS only. 

            * If "input", the output is from the table for the real part, and zeros for the imaginary part. 
            
            * If "zero", the output is always zero.

        stdysel : str
            Selects what value is output continuously by the signal generator after the generation of a pulse.
            The default is "zero".

            * If "last", it is the last calculated sample of the pulse.

            * If "zero", it is a zero value.

        phrst : int
            If 1, it resets the phase coherent accumulator. The default is 0.

        Returns
        -------
        int
            Compiled mode code in binary

        """
        if mode is None: mode = "oneshot"
        if outsel is None: outsel = "product"
        if stdysel is None: stdysel = "zero"
        if phrst is None: phrst = 0
        if length >= 2**16 or length < 3:
            raise RuntimeError("Pulse length of %d is out of range (exceeds 16 bits, or less than 3) - use multiple pulses, or zero-pad the waveform" % (length))
        stdysel_reg = {"last": 0, "zero": 1}[stdysel]
        mode_reg = {"oneshot": 0, "periodic": 1}[mode]
        outsel_reg = {"product": 0, "dds": 1, "input": 2, "zero": 3}[outsel]
        mc = phrst*0b10000+stdysel_reg*0b01000+mode_reg*0b00100+outsel_reg
        return mc << 16 | int(np.uint16(length))

    def set_registers(self, params, defaults=None) -> None:
        print("DummyChannel set_registers called with", params)
    
        if 'waveform' in params:
            pinfo = self.envelopes[params['waveform']]
            wfm_length = pinfo['data'].shape[0] // self.gencfg['samps_per_clk']
            addr = pinfo['addr'] // self.gencfg['samps_per_clk']
            self.set_reg('addr', addr, defaults=defaults)

        style = params['style']
        # these mode bits could be defined, or left as None
        phrst, stdysel, mode, outsel = [params.get(x) for x in ['phrst', 'stdysel', 'mode', 'outsel']]

        self.next_pulse = {}
        self.next_pulse['rp'] = self.rp
        self.next_pulse['regs'] = []
        if style=='const':
            mc = self.get_mode_code(phrst=phrst, stdysel=stdysel, mode=mode, outsel="dds", length=params['length'])
            self.set_reg('mode', mc, f'phrst| stdysel | mode | | outsel = 0b{mc//2**16:>05b} | length = {mc % 2**16} ')
            self.next_pulse['regs'].append([self.regmap[(self.ch,x)][1] for x in ['freq', 'phase', '0', 'gain', 'mode']])
            self.next_pulse['length'] = params['length']
        elif style=='arb':
            mc = self.get_mode_code(phrst=phrst, stdysel=stdysel, mode=mode, outsel=outsel, length=wfm_length)
            self.set_reg('mode', mc, f'phrst| stdysel | mode | | outsel = 0b{mc//2**16:>05b} | length = {mc % 2**16} ')
            self.next_pulse['regs'].append([self.regmap[(self.ch,x)][1] for x in ['freq', 'phase', 'addr', 'gain', 'mode']])
            self.next_pulse['length'] = wfm_length
        elif style=='flat_top':
            # address for ramp-down
            self.set_reg('addr2', addr+(wfm_length+1)//2)
            # gain for flat segment
            self.set_reg('gain2', params['gain']//2)
            # mode for ramp up
            mc = self.get_mode_code(phrst=phrst, stdysel=stdysel, mode='oneshot', outsel='product', length=wfm_length//2)
            self.set_reg('mode2', mc, f'phrst| stdysel | mode | | outsel = 0b{mc//2**16:>05b} | length = {mc % 2**16} ')
            # mode for flat segment
            mc = self.get_mode_code(phrst=False, stdysel=stdysel, mode='oneshot', outsel='dds', length=params['length'])
            self.set_reg('mode', mc, f'phrst| stdysel | mode | | outsel = 0b{mc//2**16:>05b} | length = {mc % 2**16} ')
            # mode for ramp down
            mc = self.get_mode_code(phrst=False, stdysel=stdysel, mode='oneshot', outsel='product', length=wfm_length//2)
            self.set_reg('mode3', mc, f'phrst| stdysel | mode | | outsel = 0b{mc//2**16:>05b} | length = {mc % 2**16} ')

            self.next_pulse['regs'].append([self.regmap[(self.ch,x)][1] for x in ['freq', 'phase', 'addr', 'gain', 'mode2']])
            self.next_pulse['regs'].append([self.regmap[(self.ch,x)][1] for x in ['freq', 'phase', '0', 'gain2', 'mode']])
            self.next_pulse['regs'].append([self.regmap[(self.ch,x)][1] for x in ['freq', 'phase', 'addr2', 'gain', 'mode3']])
            self.next_pulse['length'] = (wfm_length//2)*2 + params['length']

class DummyQickProgram:
    r"""
    Dummy QICK program class to allow Executor to run without generating a QICK program
    """
    instructions = {'pushi': {'type': "I", 'bin': 0b00010000, 'fmt': ((0, 53), (1, 41), (2, 36), (3, 0)), 'repr': "{0}, ${1}, ${2}, {3}"},
                    'popi':  {'type': "I", 'bin': 0b00010001, 'fmt': ((0, 53), (1, 41)), 'repr': "{0}, ${1}"},
                    'mathi': {'type': "I", 'bin': 0b00010010, 'fmt': ((0, 53), (1, 41), (2, 36), (3, 46), (4, 0)), 'repr': "{0}, ${1}, ${2} {3} {4}"},
                    'seti':  {'type': "I", 'bin': 0b00010011, 'fmt': ((1, 53), (0, 50), (2, 36), (3, 0)), 'repr': "{0}, {1}, ${2}, {3}"},
                    'synci': {'type': "I", 'bin': 0b00010100, 'fmt': ((0, 0),), 'repr': "{0}"},
                    'waiti': {'type': "I", 'bin': 0b00010101, 'fmt': ((0, 50), (1, 0)), 'repr': "{0}, {1}"},
                    'bitwi': {'type': "I", 'bin': 0b00010110, 'fmt': ((0, 53), (3, 46), (1, 41), (2, 36), (4, 0)), 'repr': "{0}, ${1}, ${2} {3} {4}"},
                    'memri': {'type': "I", 'bin': 0b00010111, 'fmt': ((0, 53), (1, 41), (2, 0)), 'repr': "{0}, ${1}, {2}"},
                    'memwi': {'type': "I", 'bin': 0b00011000, 'fmt': ((0, 53), (1, 31), (2, 0)), 'repr': "{0}, ${1}, {2}"},
                    'regwi': {'type': "I", 'bin': 0b00011001, 'fmt': ((0, 53), (1, 41), (2, 0)), 'repr': "{0}, ${1}, {2}"},
                    'setbi': {'type': "I", 'bin': 0b00011010, 'fmt': ((0, 53), (1, 41), (2, 0)), 'repr': "{0}, ${1}, {2}"},

                    'loopnz': {'type': "J1", 'bin': 0b00110000, 'fmt': ((0, 53), (1, 41), (1, 36), (2, 0)), 'repr': "{0}, ${1}, @{2}"},
                    'end':    {'type': "J1", 'bin': 0b00111111, 'fmt': (), 'repr': ""},

                    'condj':  {'type': "J2", 'bin': 0b00110001, 'fmt': ((0, 53), (2, 46), (1, 36), (3, 31), (4, 0)), 'repr': "{0}, ${1}, {2}, ${3}, @{4}"},

                    'math':  {'type': "R", 'bin': 0b01010000, 'fmt': ((0, 53), (3, 46), (1, 41), (2, 36), (4, 31)), 'repr': "{0}, ${1}, ${2} {3} ${4}"},
                    'set':  {'type': "R", 'bin': 0b01010001, 'fmt': ((1, 53), (0, 50), (2, 36), (7, 31), (3, 26), (4, 21), (5, 16), (6, 11)), 'repr': "{0}, {1}, ${2}, ${3}, ${4}, ${5}, ${6}, ${7}"},
                    'sync': {'type': "R", 'bin': 0b01010010, 'fmt': ((0, 53), (1, 31)), 'repr': "{0}, ${1}"},
                    'read': {'type': "R", 'bin': 0b01010011, 'fmt': ((1, 53), (0, 50), (2, 46), (3, 41)), 'repr': "{0}, {1}, {2} ${3}"},
                    'wait': {'type': "R", 'bin': 0b01010100, 'fmt': ((1, 53), (0, 50), (2, 31)), 'repr': "{0}, {1}, ${2}"},
                    'bitw': {'type': "R", 'bin': 0b01010101, 'fmt': ((0, 53), (1, 41), (2, 36), (3, 46), (4, 31)), 'repr': "{0}, ${1}, ${2} {3} ${4}"},
                    'memr': {'type': "R", 'bin': 0b01010110, 'fmt': ((0, 53), (1, 41), (2, 36)), 'repr': "{0}, ${1}, ${2}"},
                    'memw': {'type': "R", 'bin': 0b01010111, 'fmt': ((0, 53), (2, 36), (1, 31)), 'repr': "{0}, ${1}, ${2}"},
                    'setb': {'type': "R", 'bin': 0b01011000, 'fmt': ((0, 53), (2, 36), (1, 31)), 'repr': "{0}, ${1}, ${2}"},
                    'comment': {'fmt': ()}
                    }

    # op codes for math and bitwise operations
    op_codes = {">": 0b0000, ">=": 0b0001, "<": 0b0010, "<=": 0b0011, "==": 0b0100, "!=": 0b0101,
                "+": 0b1000, "-": 0b1001, "*": 0b1010,
                "&": 0b0000, "|": 0b0001, "^": 0b0010, "~": 0b0011, "<<": 0b0100, ">>": 0b0101,
                "upper": 0b1010, "lower": 0b0101
                }
    def __init__(self):
        self._dac_sample_rate = 400 * MHz
        self._adc_sample_rate = 300 * MHz
        self._gen_mgrs = [DummyChannel(ch+2) for ch in range(8)]
        self._ro_mgrs = [DummyChannel(ch+2) for ch in range(8)]
        self._gen_regmap = {
            (ch, "0"): ((ch + 2) % 8, 0) for ch in range(8)
        }
        self._gen_regmap.update({
            (ch, "freq"): ((ch + 2) % 8, 21) for ch in range(8)
        })
        self._gen_regmap.update({
            (ch, "phase"): ((ch + 2) % 8, 22) for ch in range(8)
        })
        self._gen_regmap.update({
            (ch, "gain"): ((ch + 2) % 8, 23) for ch in range(8)
        })
        self._gen_regmap.update({
            (ch, "mode"): ((ch + 2) % 8, 24) for ch in range(8)
        })
        self._gen_regmap.update({
            (ch, "t"): ((ch + 2) % 8, 25) for ch in range(8)
        })
        self._label_next = None
        self.prog_list = []

    def append_instruction(self, name, *args):
        """Append instruction to the program list

        Parameters
        ----------
        name : str
            Instruction name
        *args : dict
            Instruction arguments
        """
        n_args = max([f[0] for f in self.instructions[name]['fmt']]+[-1])+1
        if len(args)==n_args:
            inst = {'name': name, 'args': args}
        elif len(args)==n_args+1:
            inst = {'name': name, 'args': args[:n_args], 'comment': args[n_args]}
        else:
            raise RuntimeError("wrong number of args:", name, args)
        if self._label_next is not None:
            # store the label with the instruction, for printing
            inst['label'] = self._label_next
            self._label_next = None
        self.prog_list.append(inst)
    
    def us2cycles(self, value: float, gen_ch: Optional[int] = None, ro_ch: Optional[int] = None) -> int:
        r"""
        Convert microseconds to QICK clock cycles
        """
        return int(value * 1e-6 * self._dac_sample_rate)

    def freq2reg(self, value: float, gen_ch: Optional[int] = None, ro_ch: Optional[int] = None) -> int:
        r"""
        Convert frequency in Hz to QICK frequency register value
        """
        return int((value / self._dac_sample_rate) * (2**32)) & 0xFFFFFFFF

    def deg2reg(self, value: float, gen_ch: Optional[int] = None, ro_ch: Optional[int] = None) -> int:
        r"""
        Convert phase in degrees to QICK phase register value
        """
        return int((value % 360) / 360 * (2**16)) & 0xFFFF

    def set_pulse_registers(
        self,
        ch: int,
        **kwargs
    ) -> None:
        r"""
        Set pulse registers for a given channel
        """
        self._gen_mgrs[ch].set_registers(kwargs)

    def _inst2asm(self, inst, max_label_len):
        if inst['name']=='comment':
            return "// "+inst['comment']
        template = inst['name'] + " " + self.__class__.instructions[inst['name']]['repr'] + ";"
        line = " "*(max_label_len+2) + template.format(*inst['args'])
        if 'comment' in inst:
            line += " "*(48-len(line)) + "//" + (inst['comment'] if inst['comment'] is not None else "")
        if 'label' in inst:
            label = inst['label']
            line = label + ": " + line[len(label)+2:]
        return line

    def asm(self):
        """Returns assembly representation of program as string, should be compatible with the parse_prog from the parser module.

        Returns
        -------
        str
            asm file
        """
        label_list = [inst['label'] for inst in self.prog_list if 'label' in inst]
        if label_list:
            max_label_len = max([len(label) for label in label_list])
        else:
            max_label_len = 0
        s = "\n// Program\n\n"
        lines = [self._inst2asm(inst, max_label_len) for inst in self.prog_list]
        return s+"\n".join(lines)

    def __getattr__(self, a):
        """
        Uses instructions dictionary to automatically generate methods for the standard instruction set.

        Also include all QickConfig methods as methods of the QickProgram.
        This allows e.g. this.freq2reg(f) instead of this.soccfg.freq2reg(f).

        :param a: Instruction name
        :type a: str
        :return: Instruction arguments
        :rtype: *args object
        """
        if a in self.__class__.instructions:
            return lambda *args: self.append_instruction(a, *args)
        else:
            return object.__getattribute__(self, a)

    def __str__(self):
        """
        Print as assembly by default.

        :return: The asm file associated with the class
        :rtype: str
        """
        return self.asm()

    def __repr__(self):
        """
        Print as assembly by default.

        :return: The asm file associated with the class
        :rtype: str
        """
        return self.asm()

    def __len__(self):
        """
        :return: number of instructions in the program
        :rtype: int
        """
        return len(self.prog_list)

    def __enter__(self):
        """
        Enter the runtime context related to this object.

        :return: self
        :rtype: self
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        Exit the runtime context related to this object.

        :param type: type of error
        :type type: type
        :param value: value of error
        :type value: int
        :param traceback: traceback of error
        :type traceback: str
        """
        pass

class Executor:
    r"""
    QICK Executor class to convert QSTL programs to QICK programs and run them
    """
    MAX_REGISTERS = 9  # max number of variable registers in QICK (3, 4, ..., 11)
    LAST_REG = 11
    START_REG = 3

    def __init__(
        self,
        channel_mapper: ChannelMapper,
        soc: Optional[QickConfig] = None,
    ):
        # QICK SoC object
        self._soc = soc
        # Virtual to physical channel mapper
        self._channel_mapper = channel_mapper
        # QICK program object
        self._qick_program: QickProgram = None
        # QSTL program object
        self._qstl_program: Program = None
        # Time register to control output generators
        self._time_reg_scalar: dict[SingleVirtualChannel, Scalar] = {}
        # Temporal register to control output generators
        self._temporal_reg_scalar: dict[SingleVirtualChannel, Scalar] = {}
        # Map which converts Scalar variables to QICK register values
        self._scalar_eval_map: dict[Scalar, function] = {}
        self._scalar_treg_map: dict[Scalar, tuple[int, int]] = {}
        self._scalar_vreg_map: dict[Scalar, tuple[int, int]] = {}

    def execute(self, program: Program) -> None | DummyQickProgram:
        r"""
        Generate QICK program and run
        """
        self._qick_program = QickProgram(self._soc) if self._soc is not None else DummyQickProgram()
        self.walk_program(program)
        self.make_program(program)

        if self._soc is None:
            return self._qick_program

    def add_vreg(self, ch:int, scalar: Scalar) -> None:
        r"""
        Add Scalar variable to QICK program as a variable register.
        Note that page should be equal to physical channel address.
        """
        values = self._scalar_vreg_map.values()
        (page, _) = self._qick_program._gen_regmap[(ch, "0")]
        count = len([1 for (p, _) in values if p == page]) + self.START_REG - 1
        if count >= self.LAST_REG:
            raise ValueError("Exceeded maximum number of variable registers in QICK.")
        self._scalar_vreg_map[scalar] = (page, count + 1)
    
    def get_reg_from_scalar(self, scalar: Scalar) -> tuple[int, int]:
        r"""
        Return QICK variable register address of Scalar variable
        """
        return self._scalar_vreg_map[scalar]

    def write_vreg2treg(self, scalar: Scalar) -> None:
        r"""
        Write Scalar variable values to QICK program variable registers
        """
        vpage, vreg = self._scalar_vreg_map[scalar]
        tpage, treg = self._scalar_treg_map[scalar]
        if vpage != tpage:
            raise ValueError("Variable and target register pages do not match.")
        # There is no direct way to move data between variable registers and target registers in QICK.
        # So, use add instruction with 0.
        self._qick_program.add(vpage, treg, vreg, 0)

    def walk_program(self, program: Program) -> None:
        r"""
        Walk through the QICK program to setup Scalar variables
        """
        # Setup time registers for output channels
        for channel in self._channel_mapper.out_channel_map:
            ch = self.get_physical_channel(channel)
            # Setup time registers for output channels
            time_reg = Scalar(
                name = f"time_reg_ch{ch}",
                value = 0,
                dtype = int,
            )

            # Setup temporal registers for output channels
            temporal_reg = Scalar(
                name = f"temporal_reg{ch}",
                value = 0,
                dtype = int,
            )

            # Set actual registers in QICK program
            self.add_vreg(ch, time_reg)
            self.add_vreg(ch, temporal_reg)

            # Map virtual channel to time and temporal registers
            self._time_reg_scalar[channel] = time_reg
            self._temporal_reg_scalar[channel] = temporal_reg

        # Walk through the program to setup Scalar variables
        for op in program.operations:
            if isinstance(op, Synchronize):
                pass
            elif op[0] in self._channel_mapper.out_channel_map:
                channel, operation = op
                if isinstance(operation, Delay):
                    if isinstance(operation.duration, Scalar):
                        self._scalar_eval_map[operation.duration] = lambda x: self._qick_program.us2cycles(x * 1e6, 0, 0)
                    else:
                        pass
                elif isinstance(operation, DCWaveform):
                    pass
                elif isinstance(operation, RFWaveform) and self.get_instrument_type(channel) is InstrumentEnum.RF:
                    freq    = operation.rf_frequency
                    phase   = operation.instantaneous_phase
                    gain    = operation.amplitude
                    length  = operation.duration
                    ch      = self.get_physical_channel(channel)
                    if isinstance(freq, Scalar) and freq not in self._scalar_eval_map:
                        self._scalar_eval_map[freq] = lambda x: self._qick_program.freq2reg(x * 1e6, 0, 0)
                        self._scalar_treg_map[freq] = self._qick_program._gen_regmap[(ch, "freq")]
                        self.add_vreg(ch, freq)
                    if isinstance(phase, Scalar) and phase not in self._scalar_eval_map:
                        self._scalar_eval_map[phase] = lambda x: self._qick_program.deg2reg(x, 0, 0)
                        self._scalar_treg_map[phase] = self._qick_program._gen_regmap[(ch, "phase")]
                        self.add_vreg(ch, phase)
                    if isinstance(gain, Scalar) and gain not in self._scalar_eval_map:
                        self._scalar_eval_map[gain] = lambda x: int(x * (32767)) & 0xFFFF
                        self._scalar_treg_map[gain] = self._qick_program._gen_regmap[(ch, "gain")]
                        self.add_vreg(ch, gain)
                    if isinstance(length, Scalar) and length not in self._scalar_eval_map:
                        self._scalar_eval_map[length] = lambda x: self._qick_program.us2cycles(x * 1e6, 0, 0)
                        self._scalar_treg_map[length] = self._qick_program._gen_regmap[(ch, "mode")]
                        self.add_vreg(ch, length)

                elif op[0] in self._channel_mapper.in_channel_map:
                    # TODO
                    pass
                else:
                    raise NotImplementedError(f"Operation {operation} not implemented")
        return

    def make_program(self, program: Program) -> None:
        r"""
        Convert QSTL program to QICK program
        """
        for op in program.operations:
            if isinstance(op, Synchronize):
                pass
            elif op[0] in self._channel_mapper.out_channel_map:
                channel, operation = op
                if isinstance(operation, Delay):
                    (_, rl) = self.get_reg_from_scalar(self._time_reg_scalar[channel])
                    rp = self.get_physical_channel(channel)
                    self._qick_program.math(rp, rl, rl, "+", rl)
                elif isinstance(operation, DCWaveform):
                    pass
                elif isinstance(operation, RFWaveform):
                    self.setup_pulse_regs(channel, operation, out=True)
                elif op[0] in self._channel_mapper.in_channel_map:
                    # TODO
                    pass
                else:
                    raise NotImplementedError(f"Operation {operation} not implemented")
        return

    def get_physical_channel(self, channel: SingleVirtualChannel) -> int:
        r"""
        Return physical channel address of virtual channel. Note that this 
        is different from output channel number.
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
    ) -> None:
        r"""
        Map pulse parameters to QICK register values
        """
        freq    = operation.rf_frequency
        phase   = operation.instantaneous_phase
        gain    = operation.amplitude
        length  = operation.duration

        if out is True:
            # Map Scalar variables to QICK register value functions
            if isinstance(freq, Scalar):
                freq = freq.get_value()
            if isinstance(phase, Scalar):
                phase = phase.get_value()
            if isinstance(gain, Scalar):
                gain = gain.get_value()
            if isinstance(length, Scalar):
                length = length.get_value()

            freq_reg    = self._qick_program.freq2reg(operation.rf_frequency * 1e6, 0, 0)
            phase_reg   = self._qick_program.deg2reg(phase, 0, 0)
            gain_reg    = int(gain * (32767)) & 0xFFFF
            length_reg  = self._qick_program.us2cycles(length * 1e6, 0, 0) & 0xFFFF
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
            next_pulse = self._qick_program._gen_mgrs[ch].next_pulse

            for regs in next_pulse["regs"]:
                # Set pulse registers
                self._qick_program.set(
                    ch,
                    rp,
                    *regs,
                    rt,
                    f"ch = {ch}, pulse @t = ${rt}"
                )
                # Update time register
                (rp1, rm)   = self._qick_program._gen_regmap[(ch, "mode")]
                (rp2, rl)   = self.get_reg_from_scalar(self._time_reg_scalar[channel])
                (rp3, rtemp)= self.get_reg_from_scalar(self._temporal_reg_scalar[channel])
                if rp != rp1 or rp != rp2 or rp != rp3:
                    raise ValueError(
                        "Register pages do not match."
                        f"rp={rp}, rp1={rp1}, rp2={rp2}, rp3={rp3}"
                )
                # Set output time to current time register
                self._qick_program.mathi(rp, rt, rl, "+", 0)
                # Get pulse length
                self._qick_program.bitw(rp, rtemp, rm, "&", 0xFFFF)
                # Update time register
                self._qick_program.math(rp, rl, rt, "+", rtemp)
