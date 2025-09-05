"""Python driver for QICK control for QSTL"""
from __future__ import annotations
from dataclasses import dataclass, field
import re
from pathlib import Path
import numpy as np
# import plotly.graph_objects as go
# from qick import QickSoc
from typing import (
    Iterable,
    Generator,
    List,
    Union,
    Any,
)

SliceLike = Union[int, slice, List[int]]

def slice_tuple(val: tuple, idx: SliceLike) -> any | Iterable:
    r"""
    Performs advanced slicing on a tuple.

    :param val: The tuple to slice.
    :param idx: The index.
    """
    if isinstance(idx, list):
        return [val[index] for index in idx]
    return val[idx]

class Channels:
    r"""
    A collection of channels to target.

    Channels are destinations for channel-level operations. Channels
    that are grouped together can be used as a target for a common operation. Individual
    channels are identified by an integer label and the name of the collection.

    :param labels: An iterable of channel labels, or an integer representing a single
        label.
    :param name: The name of this channel collection.
    :param absolute_phase: Whether :py:class:`~keysight.qcs.channels.RFWaveform`\s
        played on the channels in this collection are rendered with a relative
        or an absolute phase.
    :raises ValueError: If the name contains symbols other than letters, numbers and
        underscores.
    """
    def __init__(
        self,
        labels: int | Iterable[int],
        name: str | None = None,
        absolute_phase: bool = False,
    ) -> None:
        self.labels = labels
        self.name = "channels" if name is None else name
        self.absolute_phase = absolute_phase
        self._channels = list(labels) if isinstance(labels, Iterable) else [labels]
        self.last_time = 0

        if not re.match("^[A-Za-z0-9_]*$", name):
            raise ValueError(
                "Channel names can only contain letters, numbers and underscores."
            )

    def __getitem__(self, idx: int | slice) -> Channels:
        return self._channels[idx]

@dataclass
class InstrumentEnum:
    RF = "RF"
    DC = "DC"
    Digitizer = "Digitizer"

class PhysicalChannel:
    def __init__(self, addr: int, inst_type: InstrumentEnum):
        self.addr: int = addr
        self.inst_type: InstrumentEnum = inst_type

class ChannelMapper:
    r"""
    Virtual to physical channel mapping.
    """

    __slots__ = tuple()

    def __init__(self, ip_address: str | None = None):
        self.ip_address = ip_address
        self.channels: list[Channels] = []
        self.physical_channels: list[PhysicalChannel] = []
        self.out_channel_map: dict[int, PhysicalChannel] = {}
        self.in_channel_map: dict[int, PhysicalChannel] = {}

    def add_channel_mapping(
        self,
        channels: Channels,
        addresses: int | Iterable[int],
        instrument_types: InstrumentEnum | Iterable[InstrumentEnum],
    ) -> None:
        r"""
        Adds a channel configuration

        :param channels: The channels to map from.
        :param addresses: The physical channel addresses to map to.
        :param instrument_types: The type of instrument present at ``address``.
        :raises ValueError: If the number of physical channels does not match
            the number of labels.
        :raises ValueError: If the attributes of ``channels`` does not
            match those specified in the channel map.
        """
        addresses = addresses if isinstance(addresses, list) else [addresses]
        if isinstance(instrument_types, InstrumentEnum):
            instrument_types = [instrument_types] * len(addresses)
        if len(channels.labels) != len(instrument_types):
            raise ValueError(
                "The number of instrument_types must be one "
                "or must match the number of labels."
            )
        if instrument_types in (
            InstrumentEnum.RF,
            InstrumentEnum.DC,
        ):
            self.out_channel_map.update(
                {
                    channel: PhysicalChannel(addr, inst_type)
                    for channel, addr, inst_type in zip(channels, addresses, instrument_types)
                }
            )
        elif instrument_types in (
            InstrumentEnum.Digitizer,
        ):
            self.in_channel_map.update(
                {
                    channel: PhysicalChannel(addr, inst_type)
                    for channel, addr, inst_type in zip(channels, addresses, instrument_types)
                }
            )
    def add_downconverters(
        self,
        dig_addresses: int | Iterable[int],
        downcon_addresses: int,
    ) -> None:
        r"""
        No downconverter in QICK
        """
        raise NotImplementedError

    def get_downconverter(
        self, channel: PhysicalChannel
    ) -> PhysicalChannel | None:
        r"""
        No downconverter in QICK
        """
        raise NotImplementedError

    def get_physical_channel(
        self, address: int
    ) -> PhysicalChannel:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def get_physical_channels(self, channels: Channels) -> list[PhysicalChannel]:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def get_virtual_channels(
        self, address: int, out: bool = True
    ) -> Iterable[Channels]:
        r"""
        Returns the :py:class:`~keysight.qcs.channels.Channels`\s for the given address.

        :param address: The address to get the virtual channels of.
        """
        for channel in (
            self.out_channel_map if out else self.in_channel_map
        ).values():
            if channel.addr == address:
                return channel
        raise ValueError(f"No channel found with address {address}.")

    def constrain_lo_frequencies(
        self,
        addresses: int,
        min_freq: float,
        max_freq: float,
    ) -> float:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def set_delays(
        self,
        addresses: int
        | list[int],
        delays: float | list[float] | Scalar[float] | list[Scalar[float]],
    ) -> None:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def set_lo_frequencies(
        self,
        addresses: int
        | list[int],
        lo_frequency: float,
    ) -> None:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def _add_physical_channels(
        self,
        addresses: int | list[int],
        instrument_types: InstrumentEnum | list[InstrumentEnum],
    ) -> None:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

class BaseOperation:
    r"""
    A base class for operations acting on QICK.
    """

    __slots__ = tuple()

    def __init__(self):
        self.name = None

    @property
    def assert_variables_have_values(self):
        r"""
        Assert that all variables in this operation have values.
        """
        for val in self.yield_variables:
            if val.value is None:
                raise ValueError(f"Variable {val} does not have a value.")

    @property
    def slice_width(self) -> int | None:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    @property
    def yield_variables(self) -> Generator[Variable]:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def broadcast(self, prefix: str, length: int) -> BaseOperation:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def __getitem__(self, idxs: int):
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

class HardwareOperation(BaseOperation):
    r"""
    A base class for hardware operations.

    :param duration: The duration of the operation in seconds.
    """

    __slots__ = tuple()

    def __init__(self):
        super().__init__()
        self.duration = None

    def n_samples(self, sample_rate: float) -> int:
        r"""
        The number of samples in this operation when sampled at a specific rate.

        :param sample_rate: The sample rate in Hz.
        """
        raise NotImplementedError

    def sampled_duration(self, sample_rate: float) -> float:
        r"""
        The duration of the operation in seconds when sampled at a specified rate.

        :param sample_rate: The rate at which to sample the operation in Hz.
        """
        raise NotImplementedError

class Delay(HardwareOperation):
    r"""
    A :py:class:`~keysight.qcs.channels.HardwareOperation` representing a delay.

    When a delay is present in a series of waveforms, the next RF waveform is modulated
    by the phase accumulated during the delay, which depends on the frequency of the
    next RF waveform. This ensures that channels can track phase evolution.

    The value of the phase is ``exp(1j * sampled_delay * int_freq)`` where
    ``sampled_delay`` is the exact duration of the delay accounting for finite sampling
    effects (that is, the output of ``delay.sampled_duration(sample_rate)``) and
    ``int_freq`` is the output of ``waveform.intermediate_frequency(lo_frequency)``\.

    :param duration: The duration of the delay in seconds.
    :param name: An optional name for this.
    """

    __slots__ = tuple()

    def __init__(
        self,
        duration: float | Iterable[float] | Variable[float],
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.duration = duration
        self.name = name

    def render(
        self,
        sample_rate: float,
        lo_frequency: float = 0.0,
        start: float = 0.0
    ) -> np.ndarray:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

class Hold(HardwareOperation):
    r"""
    A :py:class:`~keysight.qcs.channels.HardwareOperation` representing a hold.

    When a hold is present in a series of waveforms, the value of the last sample will
    be held as a constant waveform and modulated based on the previous waveform.

    :param duration: The duration of the hold in seconds.
    :param name: An optional name for this.
    """

    __slots__ = tuple()

    def __init__(
        self,
        duration: float | Iterable[float] | Variable[float],
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.duration = duration
        self.name = name

    def make_waveform(
        self, amplitude: complex, frequency: float | None
    ) -> BaseWaveform:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

class Envelope:
    r"""
    An abstract base class for envelopes.
    """

    __slots__ = tuple()

    def render(
        self,
        n_samples: int = 64,
        sample_offset: float = 0.5
    ) -> np.ndarray:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

class BaseWaveform(HardwareOperation):
    r"""
    A base class for waveforms.

    :param duration: The duration of the waveform in seconds.
    """

    __slots__ = tuple()

    def __init__(self):
        super().__init__()
        self.amplitudes: list[Variable[float]] = None
        self.envelopes: dict[Envelope, Variable[float]] = None

    def render(
        self,
        sample_rate: float,
        lo_frequency: float = 0.0,
        sample_offset: float = 0.5
    ) -> np.ndarray | list[np.ndarray]:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def to_flattop(
        self,
        hold_duration: float | Iterable[float] | Variable[float],
        fraction: float = 0.5,
    ) -> list[HardwareOperation]:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def __add__(self, other):
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def __mul__(self, other):
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    __radd__ = __add__

    __rmul__ = __mul__


class DCWaveform(BaseWaveform):
    r"""
    A class for unmodulated waveforms.

    :param duration: The duration of the waveform.
    :param envelope: The shape of the waveform.
    :param amplitude: The amplitude of the waveform relative to the range of the signal
        generator.
    :param name: An optional name for this.
    """

    __slots__ = tuple()

    def __init__(
        self,
        duration: float | Iterable[float] | Variable[float],
        envelope: Envelope,
        amplitude: float | Variable[float],
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.amplitudes = amplitude
        self.duration = duration
        self.envelope = envelope
        self.name = "DCWaveform" if name is None else name

class RFWaveform(BaseWaveform):
    r"""
    Represents a waveform with target frequency or frequencies.

    The signal :math:`V(t)` at time :math:`t` after modulating an envelope
    :math:`E(t)` by a frequency :math:`f` and a phase :math:`\phi` is

    .. math::

        V(t) = E(t) \exp(2 \pi j f t + \phi).

    .. jupyter-execute::

        import keysight.qcs as qcs

        # initialize a 100ns base envelope
        base = qcs.ConstantEnvelope()

        # initialize an RFWaveform with an amplitude of 0.3 and a frequency of 5 GHz
        pulse1 = qcs.RFWaveform(100e-9, base, 0.3, 5e9)

        # initialize a sliceable RFWaveform with different frequencies
        rf = qcs.Array("rf", value=[5e9, 6e9])
        pulse2 = qcs.RFWaveform(100e-9, base, 0.3, rf)

    .. note::

        If ``rf_frequency`` and ``instantaneous_phase`` are given as ``float``\s, they
        are converted to a scalar. Otherwise, if they are given as scalar or array,
        they are stored as provided.

    :param duration: The duration of the waveform.
    :param envelope: The envelope of the waveform before modulation.
    :param amplitude: The amplitude of the waveform relative to the range of the signal
        generator.
    :param rf_frequency: The RF frequency of the output pulse in Hz.
    :param instantaneous_phase: The amount (in radians) by which the phase of this
        waveform is shifted, relative to the rotating frame set by ``rf_frequency``.
    :param post_phase: The amount (in radians) by which the phase of all subsequent
        RF waveforms are shifted relative to the rotating frame.
    :param name: An optional name for this.

    :raises ValueError: If ``rf_frequency`, ``instantaneous_phase``, and ``post_phase``
        have invalid (not one-dimensional) or inconsistent shapes.
    """

    __slots__ = tuple()

    def __init__(
        self,
        duration: float | Iterable[float] | Scalar[float],
        envelope: Envelope,
        amplitude: float | Variable[float],
        rf_frequency: float | Variable[float],
        instantaneous_phase: float | Variable[float] = 0.0,
        post_phase: float | Variable[float] = 0.0,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.amplitudes = amplitude
        self.envelope = envelope
        self.duration = duration
        self.name = name
        self.rf_frequency = rf_frequency
        self.instantaneous_phase = instantaneous_phase
        self.post_phase = post_phase
        self.name = name


    def drag(self, coeff: float | Iterable[float] | Variable[float]) -> RFWaveform:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def intermediate_frequency(self, lo_frequency: float) -> float:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def phase_update(self, sample_rate: float, lo_frequency: float = 0.0) -> complex:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def phase_per_fractional_sample(
        self,
        sample_rate: float,
        lo_frequency: float = 0.0,
        fraction: float = 1
    ) -> complex:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

@dataclass
class UnitsEnum:
    mV = "mV"
    V = "V"
    GHz = "GHz"
    MHz = "MHz"
    ns = "ns"
    us = "us"

    @classmethod
    def from_string(cls, unit: str) -> UnitsEnum:
        try:
            return cls[unit]
        except KeyError:
            raise ValueError(f"Invalid unit: {unit}")

class Variable:
    r"""
    An abstract base class for quantities whose values can be changed.
    """

    __slots__ = tuple()

    def __init__(self):
        if type(self) is Variable:
            raise TypeError(
                "Variable is an abstract base class and cannot be instantiated directly."
            )
        self.constant: bool = None
        self.dtype: type = None
        self.name: str = None
        self.parents: Generator[Variable] = None
        self.read_only: bool = None
        self.unit: UnitsEnum = None
        self.value: Any = None

    @classmethod
    def _from_value(
        cls,
        value: any,
        dtype: type | None = None
    ) -> Variable:
        r"""
        Create a Variable from a value or pass through a value that is already a
        variable.

        :param value: The value to create a variable from.
        :param dtype: The dtype of the variable to create.
        :raises ValueError: If ``value`` cannot be used to construct a subclass of this.
        """
        dtype = dtype or complex
        if isinstance(value, Variable):
            value._validate_type(dtype)
            return value
        for subclass in cls.__subclasses__():
            try:
                return subclass._from_value(value, dtype)
            except (TypeError, ValueError):
                pass
        raise ValueError(f"Could not construct a Variable from {value}.")

    def _validate_type(
        self,
        dtype: type
    ) -> None:
        r"""
        Validate that this variable has a specifed dtype.

        :raises ValueError: If ``dtype`` is not ``self.dtype``.
        """
        if self.dtype is not dtype:
            raise ValueError(f"The dtype of {self} is not {dtype}.")

    def __add__(self, other) -> Variable:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def __truediv__(self, other) -> Variable:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def __mul__(self, other) -> Variable:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def __rsub__(self, other) -> Variable:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def __rtruediv__(self, other) -> Variable:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    def __sub__(self, other) -> Variable:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

    __radd__ = __add__

    __rdiv__ = __rtruediv__

    __rmul__ = __mul__

    __div__ = __truediv__

class Scalar(Variable):
    r"""
    A class for scalar quantities whose values are unknown or can be changed.

    .. jupyter-execute::

        import keysight.qcs as qcs

        # initialize a scalar with no set value
        scalar = qcs.Scalar("my_scalar", dtype=float)

        # initialize a scalar with set value
        scalar = qcs.Scalar("my_scalar", value=0.1, dtype=float)

    :param name: The name of scalar.
    :param value: The value of scalar, or ``None`` for a scalar with no set
        value.
    :param dtype: The dtype of scalar, which must be one of
        :py:class:`~keysight.qcs.utils.DTYPE_COERCIONS`\. Defaults to ``complex`` as
        it is the broadest supported type.
    :param unit: The unit of the scalar, which must be one of
        :py:class:`~keysight.qcs.variables.UnitsEnum`\. Defaults to None.
    :param read_only: Whether the scalar is read-only.
    :raises ValueError: If ``dtype`` is not one of
        :py:class:`~keysight.qcs.utils.DTYPE_COERCIONS`\.
    :raises ValueError: If the ``unit`` is not one of
        :py:class:`~keysight.qcs.variables.UnitsEnum`\.
    """

    __slots__ = tuple()

    def __init__(
        self,
        name: str,
        value: Any = None,
        dtype: type | None = None,
        unit: str | UnitsEnum | None = None,
        read_only: bool = False,
    ) -> None:
        if value is None and read_only:
            raise ValueError("Tried to create a read-only variable with no value.")
        self.dtype = dtype or complex
        self.value = None if value is None else dtype(value)

        if unit is not None:
            try:
                self.unit = (
                    unit if isinstance(unit, UnitsEnum) else UnitsEnum.from_string(unit)
                )
            except:
                raise ValueError

    @classmethod
    def _from_value(cls, value: any, dtype: type | None = None) -> Variable:
        r"""
        No reason to use this in QICK
        """
        raise NotImplementedError

class Program:
    r"""
    A program described as a sequence of back-to-back layers, where each layer describes
    all the control instructions to be performed during a time interval.

    :param layers: The layers of this program.
    :param name: The name of this program.
    :param save_path: The path to save the program to.
    """

    __slots__ = tuple()

    def __init__(
        self,
        name: str | None = None,
        save_path: str | Path | None = None,
    ) -> None:
        self.name = "Program" if name is None else name
        self.save_path = save_path
        self.results = None
        self.results = None
        self.repetitions = None
        self.save_path = None
        self.variables = None

        self.operations = []

    def add_acquisition(
        self,
        integration_filter: (
            HardwareOperation
            | float
            | Scalar[float]
        ),
        channels: Channels,
        new_layer: bool | None = None,
        pre_delay: Variable[float] | float | None = None,
    ) -> None:
        r"""
        Adds an acquisition to perform on a digitizer channel.

        The channels are added to the results attribute to enable the results to be
        retrieved by channel.

        :param integration_filter: The integration filter to be used when integrating
            the acquired data, or a duration in seconds for a raw acquisition.
        :param channels: The channels to acquire results from.
        :param classifier: The classifiers used to classify the integrated acquisition.
        :param new_layer: Whether to insert the operation into a new layer. The default
            of ``None`` will insert in the last layer if possible otherwise it will
            insert into a new layer.
        :param pre_delay: An optional delay in seconds to insert before the operation.
        """
        channels[0].last_time += pre_delay if pre_delay is not None else 0
        self.operations[channels[0].last_time] = (integration_filter)


    def add_waveform(
        self,
        pulse: HardwareOperation,
        channels: Channels,
        new_layer: bool | None = None,
        pre_delay: Variable[float] | float | Iterable[float] | None = None,
    ) -> None:
        r"""
        Adds a waveform to play on an AWG channel.

        :param pulse: The waveform to play.
        :param channels: The channels on which to play the waveform.
        :param new_layer: Whether to insert the operation into a new layer. The default
            of ``None`` will insert in the last layer if possible otherwise it will
            insert into a new layer.
        :param pre_delay: The delay in seconds to insert before the operation.
        """
        waveforms = to_tuple(pulse, HardwareOperation)
        waveforms = list_to_cs(waveforms, qcsc.HardwareOperations.HardwareOperation)
        pre_delay = None if pre_delay is None else to_variable(pre_delay, float)
        wrap_call(self._impl.AddWaveform, waveforms, channels, new_layer, pre_delay)

    def declare(self, variable: Variable) -> Variable:
        r"""
        Declares a variable as part of this program.

        :param variable: The variable to be declared.
        """
        return from_cs(wrap_call(self._impl.Variables.Declare, variable))

    def duration(
        self,
        duration_map: dict[BaseOperation, float | Scalar[float]] | None = None,
    ) -> Scalar[float]:
        r"""
        The duration of this program in seconds.

        :param duration_map: A map from operation/target pairs to known durations.
        """
        raise NotImplementedError

    def extend(self, *layers: Layer) -> None:
        r"""
        Extend this program.

        :param layers: The layers to extend this program with.
        """
        raise NotImplementedError

    def n_shots(self, num_reps: int) -> Program:
        r"""
        Repeat this program a specified number of times.

        :param num_reps: The number of times to repeat this program.
        :raises Warning: If `n_shots` has already been called on this program.
        """
        if any(type(rep) is Repeat for rep in self.repetitions.hw_items):
            warn(
                "`n_shots` has already been called on this program. Repeated calls to "
                "`n_shots` will add software-time looping onto this program. "
                "This may be exceptionally slow."
            )
        self._impl.Repeat(num_reps)
        return self

    def sweep(
        self, sweep_values: Array | tuple[Array], targets: Variable | tuple[Variable]
    ) -> Program:
        r"""
        Creates a program that sweeps a target variable. Additionally, can sweep
        several targets provided their sweep value shapes are compatible.

        :param sweep_values: The values to sweep.
        :param targets: The variables of this program to sweep.
        :raises ValueError: If the number of targets does not match the number of sweep
            values.
        """
        raise NotImplementedError
