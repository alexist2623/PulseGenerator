"""Build multi-output AWG-tuning amplitude sweeps for QICK ASM v1.

Normalized amplitudes use the user-facing range [-1.0, 1.0].  They are
converted to the 16-bit, 14-effective-bit DAC codes used by
``axis_awg_tuning_v1``.  SET segments define levels for one to eight AWG
tuning outputs.  RAMP segments define only transition duration: each RAMP
automatically targets the following SET level.  ``None`` on a SET means that
an output holds its previous value.

Independent sweep axes are expanded as a Cartesian product, with the last
axis varying fastest.  Sweep points and repetitions execute as nested
tProcessor hardware loops.  SET target and dependent RAMP target/step values
are held in tProcessor registers and advanced directly with add instructions;
there is no point table and no sweep-point PMEM unrolling.

An optional cross-capacitance matrix maps all virtual SET/RAMP waveforms to
the physical AWG outputs with ``physical = matrix @ virtual``.  It therefore
applies to fixed pulses as well as swept coordinates.

Optional normal RF generator pulses and a FIR-decimated DDR readout can be
attached to named SET segments.  Each sweep point may be repeated N times;
the returned DDR array is grouped as ``(point, repetition, sample, I/Q)`` and
``iq_grid`` restores the original Cartesian sweep dimensions.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from math import ceil, isfinite
from numbers import Integral, Real
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from qick.averager_program import RAveragerProgram


MAX_OUTPUTS = 8
OP_SET = 0b01
OP_RAMP = 0b10
COMMAND_REGISTER_NAMES = (
    "target",
    "reserved_start",
    "duration",
    "step",
    "control",
)


def _require_int(value, name: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_amplitude(value, name: str = "amplitude") -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < -1.0 or value > 1.0:
        raise ValueError(f"{name}={value} is outside [-1.0, 1.0]")
    return value


def cycles_from_ns(duration_ns: Real, fabric_mhz: Real) -> int:
    """Round a nanosecond duration up to whole fabric-clock cycles."""
    duration_ns = float(duration_ns)
    fabric_mhz = float(fabric_mhz)
    if duration_ns <= 0 or fabric_mhz <= 0:
        raise ValueError("duration_ns and fabric_mhz must be positive")
    return max(1, int(ceil(duration_ns * fabric_mhz / 1000.0)))


def cycles_from_us(duration_us: Real, fabric_mhz: Real) -> int:
    """Round a microsecond duration up to whole fabric-clock cycles."""
    duration_us = float(duration_us)
    fabric_mhz = float(fabric_mhz)
    if duration_us <= 0 or fabric_mhz <= 0:
        raise ValueError("duration_us and fabric_mhz must be positive")
    return max(1, int(ceil(duration_us * fabric_mhz)))


def normalized_to_dac(
    amplitude: Real,
    *,
    min_code: int = -32768,
    max_code: int = 32764,
    invalid_lsb: int = 2,
) -> int:
    """Convert [-1, 1] to a signed, MSB-aligned DAC code.

    The conversion uses a symmetric 32768 scale, rounds to the nearest legal
    code, clamps positive full scale to 32764, and keeps the lower two bits at
    zero for the default 14-effective-bit RF-DAC input format.
    """
    amplitude = _require_amplitude(amplitude)
    min_code = int(min_code)
    max_code = int(max_code)
    invalid_lsb = _require_int(invalid_lsb, "invalid_lsb", 0)
    quantum = 1 << invalid_lsb
    full_scale = max(abs(min_code), max_code + quantum)
    code = int(round(amplitude * full_scale / quantum)) * quantum
    code = max(min_code, min(max_code, code))
    return code


def dac_to_normalized(code: Integral, *, full_scale: int = 32768) -> float:
    """Convert a signed DAC code back to the normalized display scale."""
    code = _require_int(code, "code")
    return float(code) / float(full_scale)


def _div_trunc_zero(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("division by zero")
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    return sign * (abs(numerator) // abs(denominator))


def _round_div_nearest(numerator: int, denominator: int) -> int:
    """Round an integer ratio to nearest, with ties away from zero."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(int(numerator)) + denominator // 2) // denominator)


@dataclass(frozen=True)
class PulseSegment:
    name: str
    kind: str
    amplitudes: Tuple[Optional[float], ...]
    duration_cycles: int


@dataclass(frozen=True)
class AmplitudeSweep:
    segment_name: str
    output_name: str
    start: float
    stop: float
    count: int

    @property
    def points(self) -> Tuple[float, ...]:
        if self.count == 1:
            return (self.start,)
        values = [
            self.start + (self.stop - self.start) * index / (self.count - 1)
            for index in range(self.count)
        ]
        values[0] = self.start
        values[-1] = self.stop
        return tuple(values)


@dataclass(frozen=True)
class ReadoutConfig:
    """Optional readout/trigger configuration for each sweep experiment."""

    ro_ch: int
    length: int
    freq: int = 0
    phrst: int = 0
    at_segment: Optional[str] = None
    measure_delay_tproc_cycles: int = 0
    trigger_width_tproc_cycles: int = 10
    wait: bool = True

    def __post_init__(self):
        _require_int(self.ro_ch, "ro_ch", 0)
        _require_int(self.length, "length", 1)
        _require_int(self.measure_delay_tproc_cycles, "measure_delay_tproc_cycles", 0)
        _require_int(self.trigger_width_tproc_cycles, "trigger_width_tproc_cycles", 1)


@dataclass(frozen=True)
class RfPulseConfig:
    """One-shot RF pulse emitted during a named SET/gate segment."""

    gen_ch: int
    at_segment: str
    length_cycles: int
    gain: int
    freq_mhz: float = 0.0
    phase_degrees: float = 0.0
    nqz: int = 1
    delay_tproc_cycles: int = 0
    phrst: int = 1
    stdysel: str = "zero"
    require_within_segment: bool = True

    def __post_init__(self):
        _require_int(self.gen_ch, "gen_ch", 0)
        _require_int(self.length_cycles, "length_cycles", 1)
        _require_int(self.gain, "gain")
        _require_int(self.nqz, "nqz", 1)
        _require_int(self.delay_tproc_cycles, "delay_tproc_cycles", 0)
        if not str(self.at_segment):
            raise ValueError("at_segment must not be empty")
        if not isfinite(float(self.freq_mhz)):
            raise ValueError("freq_mhz must be finite")
        if not isfinite(float(self.phase_degrees)):
            raise ValueError("phase_degrees must be finite")


@dataclass(frozen=True)
class DdrFirReadoutConfig:
    """FIR-decimated 1 MSPS DDR capture performed once per repetition."""

    ro_ch: int
    samples_per_trigger: int
    at_segment: str
    readout_freq_mhz: float = 0.0
    trigger_delay_tproc_cycles: int = 0
    trigger_width_tproc_cycles: int = 12
    readout_period_cycles: int = 65535
    margin_input_samples: int = 1024
    address: int = 0
    stride_bytes: Optional[int] = None
    force_overwrite: bool = False
    settle_seconds: float = 0.05

    def __post_init__(self):
        _require_int(self.ro_ch, "ro_ch", 0)
        _require_int(self.samples_per_trigger, "samples_per_trigger", 1)
        _require_int(self.trigger_delay_tproc_cycles, "trigger_delay_tproc_cycles", 0)
        _require_int(self.trigger_width_tproc_cycles, "trigger_width_tproc_cycles", 1)
        period = _require_int(self.readout_period_cycles, "readout_period_cycles", 1)
        if period > 65535:
            raise ValueError("readout_period_cycles must fit in 16 bits")
        _require_int(self.margin_input_samples, "margin_input_samples", 0)
        _require_int(self.address, "address", 0)
        if self.stride_bytes is not None:
            _require_int(self.stride_bytes, "stride_bytes", 1)
        if not str(self.at_segment):
            raise ValueError("at_segment must not be empty")
        if not isfinite(float(self.readout_freq_mhz)):
            raise ValueError("readout_freq_mhz must be finite")
        if float(self.settle_seconds) < 0:
            raise ValueError("settle_seconds must be nonnegative")


@dataclass(frozen=True)
class FineTuneDdrResult:
    """Post-FIR DDR samples grouped by Cartesian point and repetition."""

    sweep_points: np.ndarray
    iq: np.ndarray
    reserved_physical_words: Optional[int] = None
    sweep_axes: Tuple[AmplitudeSweep, ...] = ()
    sweep_shape: Tuple[int, ...] = (1,)
    cross_capacitance: Optional[np.ndarray] = None

    @property
    def mean_iq(self) -> np.ndarray:
        """Average the N repetitions while preserving sample and I/Q axes."""
        return self.iq.astype(np.float64).mean(axis=1)

    @property
    def i(self) -> np.ndarray:
        return self.iq[..., 0]

    @property
    def q(self) -> np.ndarray:
        return self.iq[..., 1]

    @property
    def iq_grid(self) -> np.ndarray:
        """IQ reshaped to ``(*sweep_shape, repetitions, samples, I/Q)``."""
        return self.iq.reshape(*self.sweep_shape, *self.iq.shape[1:])

    @property
    def mean_iq_grid(self) -> np.ndarray:
        """Repetition-averaged IQ retaining all Cartesian sweep axes."""
        return self.iq_grid.astype(np.float64).mean(axis=len(self.sweep_shape))


@dataclass(frozen=True)
class CompiledCommand:
    point_index: int
    segment_index: int
    segment_name: str
    output_index: int
    output_name: str
    gen_ch: int
    kind: str
    target_code: int
    duration_samples: int
    step: int
    words: Tuple[int, int, int, int, int]


@dataclass(frozen=True)
class CompiledPoint:
    sweep_value: Union[float, Tuple[float, ...]]
    segment_commands: Tuple[Tuple[CompiledCommand, ...], ...]


class FineTuneSequence:
    """A named multi-output sequence composed of SET and RAMP segments."""

    def __init__(self, outputs: Union[int, Sequence[str]]):
        if isinstance(outputs, Integral):
            count = _require_int(outputs, "outputs", 1)
            output_names = tuple(f"out{index}" for index in range(count))
        else:
            output_names = tuple(str(name) for name in outputs)
        if not output_names:
            raise ValueError("at least one output is required")
        if len(output_names) > MAX_OUTPUTS:
            raise ValueError(f"at most {MAX_OUTPUTS} outputs are supported")
        if len(set(output_names)) != len(output_names):
            raise ValueError("output names must be unique")
        if any(not name for name in output_names):
            raise ValueError("output names must not be empty")

        self.output_names = output_names
        self.segments = []
        self.sweeps = []
        self._sweep_coordinate_cache: Optional[np.ndarray] = None
        self.cross_capacitance = np.eye(self.n_outputs, dtype=float)

    @property
    def n_outputs(self) -> int:
        return len(self.output_names)

    def set_cross_capacitance(self, matrix):
        """Set the virtual-gate to physical-AWG linear transform.

        Every SET and its dependent RAMP use ``physical = matrix @ virtual``.
        ``matrix[dst, src]`` is therefore the contribution of virtual gate
        ``src`` to physical AWG output ``dst``.  Diagonal entries remain one.
        """
        values = np.asarray(matrix, dtype=float)
        expected_shape = (self.n_outputs, self.n_outputs)
        if values.shape != expected_shape:
            raise ValueError(
                f"cross-capacitance matrix shape must be {expected_shape}, "
                f"received {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("cross-capacitance coefficients must be finite")
        if not np.allclose(np.diag(values), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("cross-capacitance diagonal entries must equal 1")
        self.cross_capacitance = values.copy()
        return self

    @property
    def sweep(self) -> Optional[AmplitudeSweep]:
        """Legacy view of the first sweep axis, retained for single-sweep code."""
        return self.sweeps[0] if self.sweeps else None

    @sweep.setter
    def sweep(self, value: Optional[AmplitudeSweep]) -> None:
        self.sweeps = [] if value is None else [value]
        self._sweep_coordinate_cache = None

    def _coerce_amplitudes(
        self,
        amplitudes: Union[Real, Sequence[Optional[Real]], Mapping[str, Optional[Real]]],
    ) -> Tuple[Optional[float], ...]:
        if isinstance(amplitudes, Mapping):
            unknown = set(amplitudes) - set(self.output_names)
            if unknown:
                raise KeyError(f"unknown output name(s): {sorted(unknown)}")
            raw = [amplitudes.get(name) for name in self.output_names]
        elif isinstance(amplitudes, Real) and not isinstance(amplitudes, bool):
            if self.n_outputs != 1:
                raise ValueError("a scalar amplitude is valid only for a one-output sequence")
            raw = [amplitudes]
        else:
            raw = list(amplitudes)
            if len(raw) != self.n_outputs:
                raise ValueError(
                    f"expected {self.n_outputs} amplitudes, received {len(raw)}"
                )

        converted = []
        for index, value in enumerate(raw):
            if value is None:
                converted.append(None)
            else:
                converted.append(
                    _require_amplitude(value, f"amplitude[{self.output_names[index]}]")
                )
        if all(value is None for value in converted):
            raise ValueError("a segment must update at least one output")
        return tuple(converted)

    def _add_segment(self, kind: str, name: str, amplitudes, duration_cycles: int):
        name = str(name)
        if not name:
            raise ValueError("segment name must not be empty")
        if any(segment.name == name for segment in self.segments):
            raise ValueError(f"duplicate segment name {name!r}")
        duration_cycles = _require_int(duration_cycles, "duration_cycles", 1)
        if kind == "ramp":
            values = (None,) * self.n_outputs
        else:
            values = self._coerce_amplitudes(amplitudes)
        self.segments.append(PulseSegment(name, kind, values, duration_cycles))
        return self

    def add_set(self, name: str, amplitudes, duration_cycles: int):
        """Immediately set selected outputs, then hold for fabric cycles."""
        return self._add_segment("set", name, amplitudes, duration_cycles)

    def add_ramp(self, name: str, duration_cycles: int):
        """Transition to the following SET levels over fabric-clock cycles.

        A RAMP carries no independent amplitude.  Its target is resolved from
        the immediately following SET, so changing or sweeping that SET also
        updates the incoming RAMP target and the outgoing RAMP start value.
        """
        return self._add_segment("ramp", name, None, duration_cycles)

    def set_amplitude_sweep(
        self,
        segment: str,
        output: Union[str, int],
        start: Real,
        stop: Real,
        count: int,
    ):
        """Replace all sweep axes with one axis for backward compatibility."""
        previous_sweeps = self.sweeps
        self.sweeps = []
        self._sweep_coordinate_cache = None
        try:
            return self.add_amplitude_sweep(segment, output, start, stop, count)
        except Exception:
            self.sweeps = previous_sweeps
            self._sweep_coordinate_cache = None
            raise

    def add_amplitude_sweep(
        self,
        segment: str,
        output: Union[str, int],
        start: Real,
        stop: Real,
        count: int,
    ):
        """Add one independent Cartesian sweep axis.

        Re-adding the same ``(segment, output)`` target updates that axis while
        preserving the order of all other axes.  The last axis varies fastest.
        """
        segment_name = str(segment)
        if not segment_name:
            raise ValueError("swept SET segment name must not be empty")

        if isinstance(output, Integral):
            output_index = _require_int(output, "output", 0)
            if output_index >= self.n_outputs:
                raise IndexError("output index is out of range")
            output_name = self.output_names[output_index]
        else:
            output_name = str(output)
            if output_name not in self.output_names:
                raise KeyError(f"unknown output name {output_name!r}")
            output_index = self.output_names.index(output_name)

        by_name = {item.name: item for item in self.segments}
        if segment_name not in by_name:
            raise KeyError(f"unknown segment name {segment_name!r}")
        swept_segment = by_name[segment_name]
        if swept_segment.kind != "set":
            raise ValueError("amplitude sweep must select a SET segment")
        if swept_segment.amplitudes[output_index] is None:
            raise ValueError(
                f"SET segment {segment_name!r} does not update output {output_name!r}"
            )

        count = _require_int(count, "count", 1)
        start = _require_amplitude(start, "sweep start")
        stop = _require_amplitude(stop, "sweep stop")
        new_sweep = AmplitudeSweep(segment_name, output_name, start, stop, count)
        target = (segment_name, output_name)
        for index, current in enumerate(self.sweeps):
            if (current.segment_name, current.output_name) == target:
                self.sweeps[index] = new_sweep
                break
        else:
            self.sweeps.append(new_sweep)
        self._sweep_coordinate_cache = None
        return self

    def clear_amplitude_sweeps(self):
        self.sweeps = []
        self._sweep_coordinate_cache = None
        return self

    @property
    def sweep_axes(self) -> Tuple[AmplitudeSweep, ...]:
        return tuple(self.sweeps)

    @property
    def sweep_shape(self) -> Tuple[int, ...]:
        return tuple(item.count for item in self.sweeps) or (1,)

    @property
    def sweep_coordinates(self) -> np.ndarray:
        """Cartesian coordinates with shape ``(point_count, axis_count)``."""
        if self._sweep_coordinate_cache is None:
            if not self.sweeps:
                coordinates = np.empty((1, 0), dtype=float)
            else:
                axis_points = [item.points for item in self.sweeps]
                coordinates = np.asarray(tuple(product(*axis_points)), dtype=float)
                coordinates = coordinates.reshape(-1, len(self.sweeps))
            self._sweep_coordinate_cache = coordinates
        return self._sweep_coordinate_cache

    @property
    def sweep_point_count(self) -> int:
        return int(self.sweep_coordinates.shape[0])

    def sweep_coordinate(self, point_index: int) -> Tuple[float, ...]:
        point_index = _require_int(point_index, "point_index", 0)
        if point_index >= self.sweep_point_count:
            raise IndexError("point_index is out of range")
        return tuple(float(value) for value in self.sweep_coordinates[point_index])

    @property
    def sweep_points(self) -> np.ndarray:
        if not self.sweeps:
            return np.array([0.0], dtype=float)
        if len(self.sweeps) == 1:
            return self.sweep_coordinates[:, 0].copy()
        return self.sweep_coordinates.copy()

    def _validate(self):
        if not self.segments:
            raise ValueError("the sequence has no segments")
        first = self.segments[0]
        if first.kind != "set" or any(value is None for value in first.amplitudes):
            raise ValueError(
                "the first segment must be a SET with values for every output"
            )
        for index, segment in enumerate(self.segments):
            if segment.kind != "ramp":
                continue
            if index + 1 >= len(self.segments) or self.segments[index + 1].kind != "set":
                raise ValueError(
                    f"RAMP {segment.name!r} must be immediately followed by a SET; "
                    "that SET defines the RAMP target"
                )

    def _virtual_set_amplitudes_at(
        self, point_index: int, segment_index: int
    ) -> Tuple[Optional[float], ...]:
        segment = self.segments[segment_index]
        if segment.kind != "set":
            raise ValueError("_virtual_set_amplitudes_at requires a SET segment")
        values = list(segment.amplitudes)
        coordinate = self.sweep_coordinate(point_index)
        for axis_index, sweep in enumerate(self.sweeps):
            if segment.name == sweep.segment_name:
                source_index = self.output_names.index(sweep.output_name)
                if values[source_index] is None:
                    raise ValueError(
                        f"cross-capacitance source {sweep.output_name!r} has no "
                        f"nominal value in SET {segment.name!r}"
                    )
                values[source_index] = coordinate[axis_index]
        return tuple(values)

    def _set_amplitudes_at(
        self, point_index: int, segment_index: int
    ) -> Tuple[Optional[float], ...]:
        requested_values = self._virtual_set_amplitudes_at(point_index, segment_index)
        if np.allclose(
            self.cross_capacitance,
            np.eye(self.n_outputs),
            rtol=0.0,
            atol=1.0e-12,
        ):
            return requested_values
        virtual_values = self._effective_virtual_set_amplitudes_at(
            point_index, segment_index
        )
        physical_values = self.cross_capacitance @ np.asarray(
            virtual_values, dtype=float
        )
        return tuple(
            _require_amplitude(
                float(value),
                f"physical amplitude[{self.output_names[index]}]",
            )
            for index, value in enumerate(physical_values)
        )

    def _effective_virtual_set_amplitudes_at(
        self, point_index: int, segment_index: int
    ) -> Tuple[float, ...]:
        """Resolve held virtual outputs before applying cross-capacitance."""
        values = list(self._virtual_set_amplitudes_at(point_index, segment_index))
        if not any(value is None for value in values):
            return tuple(float(value) for value in values)
        previous_index = next(
            (
                index
                for index in range(segment_index - 1, -1, -1)
                if self.segments[index].kind == "set"
            ),
            None,
        )
        if previous_index is None:
            raise ValueError("the first SET must define every virtual output")
        previous = self._effective_virtual_set_amplitudes_at(
            point_index, previous_index
        )
        return tuple(
            previous[index] if value is None else float(value)
            for index, value in enumerate(values)
        )

    def virtual_amplitudes_at(
        self, point_index: int, segment_index: int
    ) -> Tuple[Optional[float], ...]:
        """Return pre-compensation virtual amplitudes for one segment."""
        point_index = _require_int(point_index, "point_index", 0)
        segment_index = _require_int(segment_index, "segment_index", 0)
        if point_index >= self.sweep_point_count:
            raise IndexError("point_index is out of range")
        if segment_index >= len(self.segments):
            raise IndexError("segment_index is out of range")
        segment = self.segments[segment_index]
        if segment.kind == "ramp":
            if segment_index + 1 >= len(self.segments):
                raise ValueError(f"RAMP {segment.name!r} has no following SET")
            return self._effective_virtual_set_amplitudes_at(
                point_index, segment_index + 1
            )
        return self._effective_virtual_set_amplitudes_at(point_index, segment_index)

    def amplitudes_at(self, point_index: int, segment_index: int) -> Tuple[Optional[float], ...]:
        point_index = _require_int(point_index, "point_index", 0)
        segment_index = _require_int(segment_index, "segment_index", 0)
        if point_index >= self.sweep_point_count:
            raise IndexError("point_index is out of range")
        if segment_index >= len(self.segments):
            raise IndexError("segment_index is out of range")
        segment = self.segments[segment_index]
        if segment.kind == "ramp":
            if segment_index + 1 >= len(self.segments):
                raise ValueError(f"RAMP {segment.name!r} has no following SET")
            next_segment = self.segments[segment_index + 1]
            if next_segment.kind != "set":
                raise ValueError(
                    f"RAMP {segment.name!r} must be immediately followed by a SET"
                )
            return self._set_amplitudes_at(point_index, segment_index + 1)
        return self._set_amplitudes_at(point_index, segment_index)

    def sample_waveforms(
        self,
        point_index: int = 0,
        points_per_ramp: int = 64,
        *,
        space: str = "physical",
    ):
        """Return an ideal preview in physical or virtual voltage space."""
        self._validate()
        points_per_ramp = _require_int(points_per_ramp, "points_per_ramp", 2)
        if space == "physical":
            amplitude_getter = self.amplitudes_at
        elif space == "virtual":
            amplitude_getter = self.virtual_amplitudes_at
        else:
            raise ValueError("space must be 'physical' or 'virtual'")
        current = list(amplitude_getter(point_index, 0))
        times = [0.0]
        values = {name: [float(current[index])] for index, name in enumerate(self.output_names)}
        boundaries = []
        time_now = 0.0

        for segment_index, segment in enumerate(self.segments):
            targets = amplitude_getter(point_index, segment_index)
            start_time = time_now
            if segment.kind == "set":
                if segment_index != 0:
                    times.append(time_now)
                    for index, name in enumerate(self.output_names):
                        values[name].append(float(current[index]))
                    for index, target in enumerate(targets):
                        if target is not None:
                            current[index] = target
                    times.append(time_now)
                    for index, name in enumerate(self.output_names):
                        values[name].append(float(current[index]))
                time_now += segment.duration_cycles
                times.append(time_now)
                for index, name in enumerate(self.output_names):
                    values[name].append(float(current[index]))
            else:
                starts = list(current)
                for sample_index in range(1, points_per_ramp + 1):
                    fraction = sample_index / points_per_ramp
                    times.append(time_now + segment.duration_cycles * fraction)
                    for index, name in enumerate(self.output_names):
                        target = targets[index]
                        if target is None:
                            value = starts[index]
                        else:
                            value = starts[index] + (target - starts[index]) * fraction
                        values[name].append(float(value))
                for index, target in enumerate(targets):
                    if target is not None:
                        current[index] = target
                time_now += segment.duration_cycles
            boundaries.append((segment.name, start_time, time_now))

        return (
            np.asarray(times, dtype=float),
            {name: np.asarray(samples, dtype=float) for name, samples in values.items()},
            tuple(boundaries),
        )

    def plot_preview(self, point_indices: Optional[Iterable[int]] = None, *, show=True):
        """Plot ideal normalized waveforms for selected sweep points."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required for plot_preview()") from exc

        if point_indices is None:
            count = self.sweep_point_count
            point_indices = sorted(set((0, count // 2, count - 1)))
        point_indices = tuple(int(index) for index in point_indices)
        fig, axes = plt.subplots(self.n_outputs, 1, sharex=True, squeeze=False)
        for point_index in point_indices:
            times, waveforms, boundaries = self.sample_waveforms(point_index)
            coordinate = self.sweep_coordinate(point_index)
            if not self.sweeps:
                label = "fixed"
            else:
                label = ", ".join(
                    f"{sweep.output_name}/{sweep.segment_name}={coordinate[index]:.4f}"
                    for index, sweep in enumerate(self.sweeps)
                )
            for output_index, output_name in enumerate(self.output_names):
                axes[output_index, 0].plot(times, waveforms[output_name], label=label)
        for output_index, output_name in enumerate(self.output_names):
            axis = axes[output_index, 0]
            axis.set_ylabel(output_name)
            axis.set_ylim(-1.05, 1.05)
            axis.grid(True, alpha=0.3)
            axis.legend(loc="best")
            for _, start, _ in boundaries:
                axis.axvline(start, color="0.75", linewidth=0.7, linestyle="--")
        axes[-1, 0].set_xlabel("AWG fabric cycles (ideal sequence time)")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes[:, 0]

    def make_program(
        self,
        soccfg,
        awg_channels: Union[Sequence[int], Mapping[str, int]],
        *,
        repetitions_per_sweep: int = 1,
        reps: Optional[int] = None,
        readout: Optional[ReadoutConfig] = None,
        rf_pulse: Optional[RfPulseConfig] = None,
        rf_pulses: Optional[Sequence[RfPulseConfig]] = None,
        ddr_readout: Optional[DdrFirReadoutConfig] = None,
        command_lead_tproc_cycles: int = 128,
        command_spacing_tproc_cycles: int = 1,
        recovery_tproc_cycles: int = 20,
    ):
        return FineTuneAmplitudeSweepProgram(
            soccfg,
            self,
            awg_channels,
            repetitions_per_sweep=repetitions_per_sweep,
            reps=reps,
            readout=readout,
            rf_pulse=rf_pulse,
            rf_pulses=rf_pulses,
            ddr_readout=ddr_readout,
            command_lead_tproc_cycles=command_lead_tproc_cycles,
            command_spacing_tproc_cycles=command_spacing_tproc_cycles,
            recovery_tproc_cycles=recovery_tproc_cycles,
        )


def _resolve_awg_channels(sequence: FineTuneSequence, awg_channels) -> Tuple[int, ...]:
    if isinstance(awg_channels, Mapping):
        missing = set(sequence.output_names) - set(awg_channels)
        extra = set(awg_channels) - set(sequence.output_names)
        if missing or extra:
            raise ValueError(
                f"AWG channel map mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        channels = tuple(int(awg_channels[name]) for name in sequence.output_names)
    else:
        channels = tuple(int(ch) for ch in awg_channels)
        if len(channels) != sequence.n_outputs:
            raise ValueError(
                f"expected {sequence.n_outputs} AWG channels, received {len(channels)}"
            )
    if len(set(channels)) != len(channels):
        raise ValueError("AWG channels must be unique")
    if any(ch < 0 for ch in channels):
        raise ValueError("AWG channel indices must be nonnegative")
    return channels


def _validate_gen_config(gen_ch: int, gen_cfg: Mapping) -> None:
    if gen_cfg.get("type") != "axis_awg_tuning_v1" and gen_cfg.get("gen_type") != "awg_tuning":
        raise ValueError(f"generator channel {gen_ch} is not axis_awg_tuning_v1")
    required = ("tproc_ch", "f_fabric", "n_pts", "frac")
    missing = [name for name in required if name not in gen_cfg]
    if missing:
        raise KeyError(f"generator channel {gen_ch} is missing config fields {missing}")


def _pack_command_words(gen_cfg: Mapping, target: int, duration: int, step: int, opcode: int):
    if target < -(1 << 31) or target > (1 << 31) - 1:
        raise ValueError("target does not fit signed 32 bits")
    if duration < 0 or duration > (1 << 23) - 1:
        raise ValueError("RAMP duration does not fit unsigned 23 bits")
    if step < -(1 << 23) or step > (1 << 23) - 1:
        raise ValueError("RAMP step does not fit signed 24 bits")
    control = (int(opcode) & 0x3) << 16
    tmux_ch = gen_cfg.get("tmux_ch")
    if tmux_ch is not None:
        control |= (int(tmux_ch) & 0xFF) << 24
    return (
        target & 0xFFFFFFFF,
        0,
        duration & 0x7FFFFF,
        step & 0xFFFFFF,
        control & 0xFFFFFFFF,
    )


def compile_sequence(
    sequence: FineTuneSequence,
    soccfg,
    awg_channels: Union[Sequence[int], Mapping[str, int]],
) -> Tuple[Tuple[int, ...], Tuple[CompiledPoint, ...]]:
    """Compile every sweep point into exact five-word AWG commands."""
    sequence._validate()
    channels = _resolve_awg_channels(sequence, awg_channels)
    gens = soccfg["gens"]
    gen_cfgs = []
    for gen_ch in channels:
        if gen_ch >= len(gens):
            raise IndexError(f"generator channel {gen_ch} is out of range")
        gen_cfg = gens[gen_ch]
        _validate_gen_config(gen_ch, gen_cfg)
        gen_cfgs.append(gen_cfg)

    compiled_points = []
    for point_index in range(sequence.sweep_point_count):
        sweep_coordinate = sequence.sweep_coordinate(point_index)
        current_codes = [None] * sequence.n_outputs
        compiled_segments = []
        for segment_index, segment in enumerate(sequence.segments):
            amplitudes = sequence.amplitudes_at(point_index, segment_index)
            commands = []
            for output_index, amplitude in enumerate(amplitudes):
                if amplitude is None:
                    continue
                gen_cfg = gen_cfgs[output_index]
                target = normalized_to_dac(
                    amplitude,
                    min_code=int(gen_cfg.get("minv", -32768)),
                    max_code=int(gen_cfg.get("maxv", 32764)),
                    invalid_lsb=int(gen_cfg.get("dac_invalid_lsb", 2)),
                )
                if segment.kind == "set":
                    duration_samples = 0
                    step = 0
                    opcode = OP_SET
                else:
                    if current_codes[output_index] is None:
                        raise ValueError(
                            f"RAMP {segment.name!r} has no known start value for "
                            f"output {sequence.output_names[output_index]!r}"
                        )
                    duration_samples = segment.duration_cycles * int(gen_cfg["n_pts"])
                    if duration_samples > (1 << 23) - 1:
                        raise ValueError(
                            f"RAMP {segment.name!r} duration expands to {duration_samples} "
                            "scalar samples, exceeding the 23-bit command field"
                        )
                    denominator = max(1, duration_samples - 1)
                    numerator = (target - current_codes[output_index]) << int(gen_cfg["frac"])
                    step = _div_trunc_zero(numerator, denominator)
                    if step < -(1 << 23) or step > (1 << 23) - 1:
                        raise ValueError(
                            f"RAMP {segment.name!r} step {step} exceeds signed 24 bits; "
                            "increase duration_cycles or reduce the amplitude span"
                        )
                    opcode = OP_RAMP
                words = _pack_command_words(gen_cfg, target, duration_samples, step, opcode)
                commands.append(
                    CompiledCommand(
                        point_index=point_index,
                        segment_index=segment_index,
                        segment_name=segment.name,
                        output_index=output_index,
                        output_name=sequence.output_names[output_index],
                        gen_ch=channels[output_index],
                        kind=segment.kind,
                        target_code=target,
                        duration_samples=duration_samples,
                        step=step,
                        words=words,
                    )
                )
                current_codes[output_index] = target
            compiled_segments.append(tuple(commands))
        if len(sweep_coordinate) == 1:
            sweep_value = sweep_coordinate[0]
        elif sweep_coordinate:
            sweep_value = sweep_coordinate
        else:
            sweep_value = 0.0
        compiled_points.append(CompiledPoint(sweep_value, tuple(compiled_segments)))
    return channels, tuple(compiled_points)


class FineTuneAmplitudeSweepProgram(RAveragerProgram):
    """ASM v1 program generated from :class:`FineTuneSequence`.

    Sweep points and repetitions are both hardware loops.  Variable SET
    targets and dependent RAMP target/step fields live in tProcessor
    registers.  Nested Cartesian loops advance and reset those fields with
    register adds; no per-point command or delta table is stored.
    Commands sharing one tProcessor AXIS port are separated by the configured
    number of tProcessor cycles, which supports TMUX fanout to as many as
    eight AWG tuning outputs without same-port/same-cycle collisions.
    """

    def __init__(
        self,
        soccfg,
        sequence: FineTuneSequence,
        awg_channels,
        *,
        repetitions_per_sweep: int = 1,
        reps: Optional[int] = None,
        readout: Optional[ReadoutConfig] = None,
        rf_pulse: Optional[RfPulseConfig] = None,
        rf_pulses: Optional[Sequence[RfPulseConfig]] = None,
        ddr_readout: Optional[DdrFirReadoutConfig] = None,
        command_lead_tproc_cycles: int = 128,
        command_spacing_tproc_cycles: int = 1,
        recovery_tproc_cycles: int = 20,
    ):
        sequence._validate()
        self.sequence = sequence
        self.awg_channel_spec = awg_channels
        self.readout_config = readout
        if rf_pulse is not None and rf_pulses is not None:
            raise ValueError("use either rf_pulse or rf_pulses, not both")
        self.rf_pulse_configs = tuple(
            rf_pulses
            if rf_pulses is not None
            else (() if rf_pulse is None else (rf_pulse,))
        )
        if any(not isinstance(item, RfPulseConfig) for item in self.rf_pulse_configs):
            raise TypeError("every RF pulse entry must be an RfPulseConfig")
        rf_channels = tuple(item.gen_ch for item in self.rf_pulse_configs)
        if len(set(rf_channels)) != len(rf_channels):
            raise ValueError("each RF pulse must use a unique generator channel")
        # Retain the singular attribute for older callers which inspect it.
        self.rf_pulse_config = (
            self.rf_pulse_configs[0] if len(self.rf_pulse_configs) == 1 else None
        )
        self.ddr_readout_config = ddr_readout
        if readout is not None and ddr_readout is not None:
            raise ValueError("readout and ddr_readout cannot be enabled together")
        self.command_lead_tproc_cycles = _require_int(
            command_lead_tproc_cycles, "command_lead_tproc_cycles", 0
        )
        self.command_spacing_tproc_cycles = _require_int(
            command_spacing_tproc_cycles, "command_spacing_tproc_cycles", 1
        )
        self.recovery_tproc_cycles = _require_int(
            recovery_tproc_cycles, "recovery_tproc_cycles", 0
        )
        repetitions_per_sweep = _require_int(
            repetitions_per_sweep, "repetitions_per_sweep", 1
        )
        if reps is not None:
            reps = _require_int(reps, "reps", 1)
            if repetitions_per_sweep != 1 and repetitions_per_sweep != reps:
                raise ValueError("reps and repetitions_per_sweep disagree")
            repetitions_per_sweep = reps
        points = sequence.sweep_points
        if points.ndim == 1:
            cfg_start = float(points[0])
            nominal_step = 0.0 if len(points) < 2 else float(points[1] - points[0])
        else:
            # RAveragerProgram requires scalar start/step metadata.  Cartesian
            # coordinates are returned by get_expt_pts() instead.
            cfg_start = 0.0
            nominal_step = 1.0
        cfg = {
            "reps": repetitions_per_sweep,
            "expts": sequence.sweep_point_count,
            "start": cfg_start,
            "step": nominal_step,
        }
        super().__init__(soccfg, cfg)

    def get_expt_pts(self):
        return self.sequence.sweep_points.copy()

    def initialize(self):
        self.awg_channels, self.compiled_points = compile_sequence(
            self.sequence, self.soccfg, self.awg_channel_spec
        )
        self._build_sweep_register_plan()
        self._channel_slots = self._build_channel_slots()
        self.timing = self._build_timing()
        self.aux_timing = {}

        for rf_config in self.rf_pulse_configs:
            self._configure_rf_pulse(rf_config)
        if self.ddr_readout_config is not None:
            self._configure_ddr_readout()
        if self.rf_pulse_configs or self.ddr_readout_config is not None:
            self._build_aux_timing()

        if self.readout_config is not None:
            ro = self.readout_config
            self.declare_readout(ch=ro.ro_ch, length=ro.length)
            self.set_readout_registers(
                ch=ro.ro_ch,
                freq=ro.freq,
                length=ro.length,
                phrst=ro.phrst,
            )

    @staticmethod
    def _command_key(command: CompiledCommand) -> Tuple[int, int]:
        return command.segment_index, command.output_index

    def _build_sweep_register_plan(self):
        """Map Cartesian sweep axes to tProcessor register increments."""
        command_maps = []
        command_order = []
        for point_index, point in enumerate(self.compiled_points):
            point_map = {}
            for commands in point.segment_commands:
                for command in commands:
                    key = self._command_key(command)
                    if key in point_map:
                        raise RuntimeError(f"duplicate compiled command key {key}")
                    point_map[key] = command
                    if point_index == 0:
                        command_order.append(key)
            command_maps.append(point_map)

        expected_keys = set(command_order)
        for point_index, point_map in enumerate(command_maps[1:], start=1):
            if set(point_map) != expected_keys:
                raise RuntimeError(
                    f"sweep point {point_index} changes the active AWG command set"
                )

        sweep_axes = self.sequence.sweep_axes
        sweep_shape = tuple(axis.count for axis in sweep_axes)
        models = {}
        for key in command_order:
            first_command = command_maps[0][key]
            for register_name, word_index in (("target", 0), ("step", 3)):
                if register_name == "target":
                    base = int(first_command.target_code)
                    quantum = 1 << int(
                        self.soccfg["gens"][first_command.gen_ch].get(
                            "dac_invalid_lsb", 2
                        )
                    )
                else:
                    base = int(first_command.step)
                    quantum = 1

                axis_deltas = []
                for axis_index, axis in enumerate(sweep_axes):
                    if axis.count <= 1:
                        axis_deltas.append(0)
                        continue
                    indices = [0] * len(sweep_axes)
                    indices[axis_index] = axis.count - 1
                    endpoint_index = int(
                        np.ravel_multi_index(tuple(indices), sweep_shape, order="C")
                    )
                    endpoint_command = command_maps[endpoint_index][key]
                    endpoint = int(
                        endpoint_command.target_code
                        if register_name == "target"
                        else endpoint_command.step
                    )
                    units = _round_div_nearest(
                        endpoint - base,
                        (axis.count - 1) * quantum,
                    )
                    axis_deltas.append(units * quantum)
                models[(key[0], key[1], register_name)] = {
                    "key": (key[0], key[1], register_name),
                    "command_key": key,
                    "register_name": register_name,
                    "word_index": word_index,
                    "base": base,
                    "axis_deltas": tuple(axis_deltas),
                }

        requested_points = self.compiled_points
        actual_points = []
        max_target_error = 0
        max_step_error = 0
        for point_index, requested_point in enumerate(requested_points):
            if sweep_axes:
                indices = np.unravel_index(point_index, sweep_shape, order="C")
            else:
                indices = ()
            actual_segments = []
            for commands in requested_point.segment_commands:
                actual_commands = []
                for command in commands:
                    target_model = models[
                        (command.segment_index, command.output_index, "target")
                    ]
                    step_model = models[
                        (command.segment_index, command.output_index, "step")
                    ]
                    target = int(target_model["base"]) + sum(
                        int(index) * int(delta)
                        for index, delta in zip(indices, target_model["axis_deltas"])
                    )
                    step = int(step_model["base"]) + sum(
                        int(index) * int(delta)
                        for index, delta in zip(indices, step_model["axis_deltas"])
                    )
                    gen_cfg = self.soccfg["gens"][command.gen_ch]
                    opcode = OP_SET if command.kind == "set" else OP_RAMP
                    words = _pack_command_words(
                        gen_cfg,
                        target,
                        command.duration_samples,
                        step,
                        opcode,
                    )
                    actual_commands.append(
                        replace(command, target_code=target, step=step, words=words)
                    )
                    max_target_error = max(
                        max_target_error, abs(target - int(command.target_code))
                    )
                    max_step_error = max(max_step_error, abs(step - int(command.step)))
                actual_segments.append(tuple(actual_commands))
            actual_points.append(
                replace(requested_point, segment_commands=tuple(actual_segments))
            )

        self.requested_compiled_points = requested_points
        self.compiled_points = tuple(actual_points)

        fields = []
        for model in models.values():
            if not any(model["axis_deltas"]):
                continue
            first_command = command_maps[0][model["command_key"]]
            page, command_register = self._gen_regmap[
                (first_command.gen_ch, model["register_name"])
            ]
            field = dict(model)
            field.update({
                "page": int(page),
                "command_register": int(command_register),
            })
            fields.append(field)

        occupied = {page: {0} for page in range(8)}
        for register_map in (self._gen_regmap, self._ro_regmap):
            for page, register in register_map.values():
                occupied[int(page)].add(int(register))
        # These are used below for the shot count, repetition loop, and trigger.
        occupied[0].update((13, 15, 16))

        for field in fields:
            page = field["page"]
            available = [
                register for register in range(1, 32) if register not in occupied[page]
            ]
            if not available:
                raise RuntimeError(
                    f"register page {page} has no room for direct sweep state "
                    f"{field['key']}"
                )
            field["state_register"] = available[0]
            occupied[page].add(available[0])

        axis_runtime = {}
        for axis_index, axis in enumerate(sweep_axes):
            if axis.count <= 1:
                continue
            allocated = None
            for page in range(8):
                available = [
                    register
                    for register in range(1, 32)
                    if register not in occupied[page]
                ]
                if available:
                    allocated = (page, available[0])
                    occupied[page].add(available[0])
                    break
            if allocated is None:
                raise RuntimeError(
                    f"no tProcessor register is available for sweep axis {axis_index}"
                )
            axis_runtime[axis_index] = {
                "counter_page": allocated[0],
                "counter_register": allocated[1],
                "count": int(axis.count),
            }

        self._sweep_models = models
        self._sweep_fields = tuple(fields)
        self._sweep_field_by_key = {field["key"]: field for field in fields}
        self._sweep_axis_runtime = axis_runtime
        self._sweep_max_target_error = int(max_target_error)
        self._sweep_max_step_error = int(max_step_error)

    def _segment_index(self, name: str, *, require_set: bool = False) -> int:
        names = [segment.name for segment in self.sequence.segments]
        if name not in names:
            raise KeyError(f"unknown sequence segment {name!r}")
        index = names.index(name)
        if require_set and self.sequence.segments[index].kind != "set":
            raise ValueError(f"segment {name!r} must be a SET segment")
        return index

    def _configure_rf_pulse(self, rf: RfPulseConfig):
        if rf.gen_ch in self.awg_channels:
            raise ValueError("RF generator channel must be separate from AWG tuning channels")
        if rf.gen_ch >= len(self.soccfg["gens"]):
            raise IndexError("RF generator channel is out of range")
        gen_cfg = self.soccfg["gens"][rf.gen_ch]
        if gen_cfg.get("type") == "axis_awg_tuning_v1" or gen_cfg.get("gen_type") == "awg_tuning":
            raise ValueError("rf_pulse requires a normal QICK RF signal generator")
        self._segment_index(rf.at_segment, require_set=True)

        self.declare_gen(ch=rf.gen_ch, nqz=rf.nqz)
        ro_ch = self.ddr_readout_config.ro_ch if self.ddr_readout_config is not None else None
        try:
            freq_word = self.freq2reg(rf.freq_mhz, gen_ch=rf.gen_ch, ro_ch=ro_ch)
        except KeyError as exc:
            if exc.args != ("refclk_freq",):
                raise
            b_dds = int(gen_cfg["b_dds"])
            freq_word = int(round(float(rf.freq_mhz) * (1 << b_dds) / float(gen_cfg["f_dds"])))
            freq_word %= 1 << b_dds
        phase_word = self.deg2reg(rf.phase_degrees, gen_ch=rf.gen_ch)
        self.set_pulse_registers(
            ch=rf.gen_ch,
            style="const",
            freq=freq_word,
            phase=phase_word,
            gain=rf.gain,
            length=rf.length_cycles,
            phrst=rf.phrst,
            stdysel=rf.stdysel,
            mode="oneshot",
        )

    def _configure_ddr_readout(self):
        ddr = self.ddr_readout_config
        self._segment_index(ddr.at_segment, require_set=True)
        try:
            ddr_cfg = self.soccfg["ddr4_buf"]
        except KeyError as exc:
            raise RuntimeError("soccfg does not contain a DDR sample buffer") from exc
        if not ddr_cfg.get("sample_capture", False):
            raise RuntimeError("DDR readout requires axis_buffer_ddr_sample_v1")
        if not ddr_cfg.get("fir_enabled", False):
            raise RuntimeError("DDR readout requires the FIR 300:1 data path")
        output_rate = float(ddr_cfg.get("fir_output_fs_mhz", 0.0))
        if not np.isclose(output_rate, 1.0, rtol=0.0, atol=1e-9):
            raise RuntimeError(
                f"expected a 1 MSPS FIR output, HWH reports {output_rate} MSPS"
            )
        if ddr.ro_ch >= len(self.soccfg["readouts"]):
            raise IndexError("DDR readout channel is out of range")

        ro_cfg = self.soccfg["readouts"][ddr.ro_ch]
        monitor_length = min(
            ddr.samples_per_trigger,
            int(ro_cfg.get("buf_maxlen", ddr.samples_per_trigger)),
        )
        self.declare_readout(ch=ddr.ro_ch, length=max(1, monitor_length))
        gen_ch = self.rf_pulse_configs[0].gen_ch if self.rf_pulse_configs else None
        try:
            freq_word = self.freq2reg_adc(
                ddr.readout_freq_mhz,
                ro_ch=ddr.ro_ch,
                gen_ch=gen_ch,
            )
        except KeyError as exc:
            if exc.args != ("refclk_freq",):
                raise
            b_dds = int(ro_cfg["b_dds"])
            freq_word = int(
                round(float(ddr.readout_freq_mhz) * (1 << b_dds) / float(ro_cfg["f_dds"]))
            )
            freq_word %= 1 << b_dds
        self.set_readout_registers(
            ch=ddr.ro_ch,
            freq=freq_word,
            length=ddr.readout_period_cycles,
            mode="periodic",
        )
        self._fir_cfg = {
            "decimation": int(ddr_cfg.get("fir_decimation", 300)),
            "group_delay_input_samples": int(
                ceil(float(ddr_cfg.get("fir_group_delay_input_samples", 8677)))
            ),
            "input_fs_mhz": float(ddr_cfg.get("fir_input_fs_mhz", 300.0)),
            "output_fs_mhz": output_rate,
        }

    def _shift_timing(self, delta: int):
        if delta <= 0:
            return
        self.timing["command_times"] = {
            key: value + delta for key, value in self.timing["command_times"].items()
        }
        self.timing["segment_starts"] = tuple(
            value + delta for value in self.timing["segment_starts"]
        )
        self.timing["segment_ends"] = tuple(
            value + delta for value in self.timing["segment_ends"]
        )
        self.timing["point_end"] += delta

    def _build_aux_timing(self):
        f_time = float(self.soccfg["tprocs"][0]["f_time"])
        if f_time <= 0:
            raise ValueError("tProcessor clock must be positive")

        if self.ddr_readout_config is not None:
            ddr = self.ddr_readout_config
            segment_index = self._segment_index(ddr.at_segment, require_set=True)
            trigger_time = (
                self.timing["segment_starts"][segment_index]
                + ddr.trigger_delay_tproc_cycles
            )
            input_fs = self._fir_cfg["input_fs_mhz"]
            group_delay = self._fir_cfg["group_delay_input_samples"]
            warmup_cycles = int(ceil(group_delay * f_time / input_fs))
            readout_start = trigger_time - warmup_cycles
            if readout_start < 0:
                shift = -readout_start
                self._shift_timing(shift)
                trigger_time += shift
                readout_start += shift

            feed_input_samples = (
                ddr.samples_per_trigger * self._fir_cfg["decimation"]
                + group_delay
                + ddr.margin_input_samples
            )
            feed_tproc_cycles = int(ceil(feed_input_samples * f_time / input_fs))
            capture_end = readout_start + feed_tproc_cycles
            self.aux_timing.update({
                "ddr_readout_start": int(readout_start),
                "ddr_trigger_time": int(trigger_time),
                "ddr_capture_end": int(capture_end),
                "fir_warmup_tproc_cycles": int(warmup_cycles),
                "fir_feed_input_samples": int(feed_input_samples),
            })
            self.timing["point_end"] = max(self.timing["point_end"], capture_end)

        rf_timings = []
        occupied_by_port = {}
        for (_awg_segment, output_index), command_time in self.timing["command_times"].items():
            awg_ch = self.awg_channels[output_index]
            awg_port = int(self.soccfg["gens"][awg_ch]["tproc_ch"])
            occupied_by_port.setdefault(awg_port, set()).add(int(command_time))
        if self.ddr_readout_config is not None:
            ro_cfg = self.soccfg["readouts"][self.ddr_readout_config.ro_ch]
            occupied_by_port.setdefault(int(ro_cfg["tproc_ctrl"]), set()).add(
                int(self.aux_timing["ddr_readout_start"])
            )

        for rf_index, rf in enumerate(self.rf_pulse_configs):
            segment_index = self._segment_index(rf.at_segment, require_set=True)
            requested_start = (
                self.timing["segment_starts"][segment_index] + rf.delay_tproc_cycles
            )
            rf_port = int(self.soccfg["gens"][rf.gen_ch]["tproc_ch"])
            occupied = occupied_by_port.setdefault(rf_port, set())
            rf_start = int(requested_start)
            while rf_start in occupied:
                rf_start += self.command_spacing_tproc_cycles
            occupied.add(rf_start)
            rf_end = rf_start + self._fabric_to_tproc(rf.gen_ch, rf.length_cycles)
            if rf.require_within_segment and rf_end > self.timing["segment_ends"][segment_index]:
                raise ValueError(
                    f"RF pulse ends at t={rf_end}, after SET segment {rf.at_segment!r} "
                    f"ends at t={self.timing['segment_ends'][segment_index]}"
                )
            timing = {
                "index": rf_index,
                "gen_ch": rf.gen_ch,
                "requested_start": int(requested_start),
                "start": int(rf_start),
                "end": int(rf_end),
                "command_skew_tproc_cycles": int(rf_start - requested_start),
            }
            rf_timings.append(timing)
            self.aux_timing.update({
                f"rf_{rf_index}_requested_start": timing["requested_start"],
                f"rf_{rf_index}_start": timing["start"],
                f"rf_{rf_index}_end": timing["end"],
                f"rf_{rf_index}_command_skew_tproc_cycles": (
                    timing["command_skew_tproc_cycles"]
                ),
            })
            self.timing["point_end"] = max(self.timing["point_end"], rf_end)
        self.aux_timing["rf_pulses"] = tuple(rf_timings)
        if len(rf_timings) == 1:
            timing = rf_timings[0]
            self.aux_timing.update({
                "rf_requested_start": timing["requested_start"],
                "rf_start": timing["start"],
                "rf_end": timing["end"],
                "rf_command_skew_tproc_cycles": timing["command_skew_tproc_cycles"],
            })

    def _build_channel_slots(self) -> Dict[int, int]:
        next_slot: Dict[int, int] = {}
        slots = {}
        for gen_ch in self.awg_channels:
            port = int(self.soccfg["gens"][gen_ch]["tproc_ch"])
            slot = next_slot.get(port, 0)
            slots[gen_ch] = slot * self.command_spacing_tproc_cycles
            next_slot[port] = slot + 1
        return slots

    def _fabric_to_tproc(self, gen_ch: int, fabric_cycles: int) -> int:
        f_time = float(self.soccfg["tprocs"][0]["f_time"])
        f_fabric = float(self.soccfg["gens"][gen_ch]["f_fabric"])
        if f_time <= 0 or f_fabric <= 0:
            raise ValueError("tProcessor and AWG fabric clocks must be positive")
        return max(1, int(ceil(fabric_cycles * f_time / f_fabric - 1e-12)))

    def _build_timing(self):
        first_point = self.compiled_points[0]
        command_times = {}
        segment_starts = []
        segment_ends = []
        # Command timestamps are point-relative. The startup lead is applied
        # once with synci before entering the hardware sweep loops, rather than
        # being inserted again before every SET at every point/repetition.
        time_now = 0
        for segment_index, segment in enumerate(self.sequence.segments):
            segment_starts.append(time_now)
            active = {
                command.output_index: command
                for command in first_point.segment_commands[segment_index]
            }
            output_end_times = []
            for output_index, gen_ch in enumerate(self.awg_channels):
                command = active.get(output_index)
                if command is None:
                    output_end_times.append(
                        time_now + self._fabric_to_tproc(gen_ch, segment.duration_cycles)
                    )
                    continue
                if segment.kind == "ramp":
                    gen_cfg = self.soccfg["gens"][gen_ch]
                    startup_cycles = int(
                        gen_cfg.get("ramp_startup_latency_cycles", 5)
                    )
                    startup_tproc = self._fabric_to_tproc(
                        gen_ch, startup_cycles
                    )
                    # Dispatch the RAMP before the logical segment boundary.
                    # Its pipeline then starts changing the DAC at the end of
                    # the preceding SET instead of extending that SET.
                    command_time = (
                        time_now
                        - startup_tproc
                        + self._channel_slots[gen_ch]
                    )
                    if command_time < 0:
                        raise ValueError(
                            "the first SET is too short to hide AWG RAMP startup latency"
                        )
                    occupancy = (
                        segment.duration_cycles
                        + startup_cycles
                        + int(gen_cfg.get("ramp_guard_cycles", 1))
                    )
                else:
                    command_time = time_now + self._channel_slots[gen_ch]
                    occupancy = segment.duration_cycles
                command_times[(segment_index, output_index)] = command_time
                output_end_times.append(
                    command_time + self._fabric_to_tproc(gen_ch, occupancy)
                )
            time_now = max(output_end_times)
            segment_ends.append(time_now)

        return {
            "command_times": command_times,
            "segment_starts": tuple(segment_starts),
            "segment_ends": tuple(segment_ends),
            "point_end": int(time_now),
        }

    def _emit_command(self, command: CompiledCommand, tproc_time: int):
        """Emit one command, copying swept fields from tProcessor state regs."""
        gen_ch = command.gen_ch
        page = self._gen_regmap[(gen_ch, "target")][0]
        regs = []
        for register_name, word in zip(COMMAND_REGISTER_NAMES, command.words):
            reg_page, reg = self._gen_regmap[(gen_ch, register_name)]
            if reg_page != page:
                raise RuntimeError("AWG command registers must share one register page")
            field = self._sweep_field_by_key.get(
                (command.segment_index, command.output_index, register_name)
            )
            comment = f"{command.output_name}:{command.segment_name}:{register_name}"
            if field is None:
                self.safe_regwi(page, reg, int(word), comment)
            elif register_name == "step":
                self.bitwi(
                    page,
                    reg,
                    int(field["state_register"]),
                    "&",
                    0xFFFFFF,
                    comment,
                )
            else:
                self.mathi(
                    page,
                    reg,
                    int(field["state_register"]),
                    "+",
                    0,
                    comment,
                )
            regs.append(reg)
        _, time_reg = self._gen_regmap[(gen_ch, "t")]
        self.safe_regwi(page, time_reg, int(tproc_time), f"{command.segment_name} t")
        self.set(
            int(self.soccfg["gens"][gen_ch]["tproc_ch"]),
            page,
            *regs,
            time_reg,
            f"{command.output_name}:{command.segment_name}",
        )

    def _emit_point(self):
        # The instruction body is point-independent. Swept SET/RAMP fields are
        # copied from state registers immediately before each command.
        point = self.compiled_points[0]
        for segment_index, commands in enumerate(point.segment_commands):
            for command in commands:
                command_time = self.timing["command_times"][(
                    segment_index,
                    command.output_index,
                )]
                self._emit_command(command, command_time)

        if self.ddr_readout_config is not None:
            ddr = self.ddr_readout_config
            self.readout(ch=ddr.ro_ch, t=self.aux_timing["ddr_readout_start"])
            self.trigger(
                ddr4=True,
                adc_trig_offset=0,
                t=self.aux_timing["ddr_trigger_time"],
                width=ddr.trigger_width_tproc_cycles,
            )

        for rf_index, rf_config in enumerate(self.rf_pulse_configs):
            self.pulse(
                ch=rf_config.gen_ch,
                t=self.aux_timing[f"rf_{rf_index}_start"],
            )

        point_end = int(self.timing["point_end"])
        for gen_ch in self.awg_channels:
            self.set_timestamp(point_end, gen_ch=gen_ch)

        if self.readout_config is not None:
            ro = self.readout_config
            if ro.at_segment is None:
                measure_time = point_end
            else:
                names = [segment.name for segment in self.sequence.segments]
                if ro.at_segment not in names:
                    raise KeyError(f"unknown readout segment {ro.at_segment!r}")
                measure_time = self.timing["segment_ends"][names.index(ro.at_segment)]
            measure_time += ro.measure_delay_tproc_cycles
            self.readout(ch=ro.ro_ch, t=int(measure_time))
            self.trigger(
                adcs=[ro.ro_ch],
                adc_trig_offset=int(measure_time),
                t=0,
                width=ro.trigger_width_tproc_cycles,
            )
            if ro.wait:
                self.wait_all()

        self.sync_all(self.recovery_tproc_cycles)

    def _emit_axis_adds(self, axis_index: int, *, reset: bool = False):
        """Advance or reset one Cartesian axis using register-immediate adds."""
        axis = self.sequence.sweep_axes[axis_index]
        multiplier = -(axis.count - 1) if reset else 1
        action = "reset" if reset else "advance"
        for field in self._sweep_fields:
            amount = int(field["axis_deltas"][axis_index]) * multiplier
            if amount == 0:
                continue
            page = int(field["page"])
            state_register = int(field["state_register"])
            self.mathi(
                page,
                state_register,
                state_register,
                "+",
                amount,
                f"{action} axis {axis_index} {field['key']}",
            )

    def make_program(self):
        """Emit nested point/repetition loops and register-add sweep updates."""
        rcount = 13
        rrep = 15
        self.initialize()
        self.regwi(0, rcount, 0)
        for field in self._sweep_fields:
            self.safe_regwi(
                int(field["page"]),
                int(field["state_register"]),
                int(field["base"]),
                f"initialize direct sweep state {field['key']}",
            )
        active_axes = tuple(sorted(self._sweep_axis_runtime))
        for axis_index in active_axes:
            runtime = self._sweep_axis_runtime[axis_index]
            self.safe_regwi(
                int(runtime["counter_page"]),
                int(runtime["counter_register"]),
                int(runtime["count"]) - 1,
                f"initialize sweep axis {axis_index} counter",
            )
        if self.command_lead_tproc_cycles:
            self.synci(self.command_lead_tproc_cycles)

        self.label("FINE_TUNE_POINT")
        self.regwi(0, rrep, self.cfg["reps"] - 1)
        self.label("FINE_TUNE_REP")
        self._emit_point()
        self.mathi(0, rcount, rcount, "+", 1)
        self.memwi(0, rcount, self.COUNTER_ADDR)
        self.loopnz(0, rrep, "FINE_TUNE_REP")

        # The last axis varies fastest. A finished inner loop is reset before
        # the next outer axis is advanced, matching itertools.product order.
        for position in range(len(active_axes) - 1, -1, -1):
            axis_index = active_axes[position]
            runtime = self._sweep_axis_runtime[axis_index]
            page = int(runtime["counter_page"])
            register = int(runtime["counter_register"])
            self.loopnz(page, register, f"FINE_TUNE_ADVANCE_{axis_index}")
            if position > 0:
                self.safe_regwi(
                    page,
                    register,
                    int(runtime["count"]) - 1,
                    f"reload sweep axis {axis_index} counter",
                )
                self._emit_axis_adds(axis_index, reset=True)
        self.end()

        for axis_index in active_axes:
            self.label(f"FINE_TUNE_ADVANCE_{axis_index}")
            self._emit_axis_adds(axis_index)
            self.condj(0, 0, "==", 0, "FINE_TUNE_POINT")

    def body(self):
        raise RuntimeError(
            "FineTuneAmplitudeSweepProgram uses nested hardware loops in make_program()"
        )

    def update(self):
        pass

    def acquire_fir_ddr(self, soc, *, progress: bool = True, **run_kwargs):
        """Arm DDR, execute all sweep repetitions, and return grouped 1 MSPS IQ.

        The nested hardware loops run Cartesian-point-major and
        repetition-minor, so the flat DDR stream is reshaped to
        ``(point_count, N, samples, I/Q)``.  ``result.iq_grid`` restores the
        independent sweep axes before the repetition dimension.
        """
        if self.ddr_readout_config is None:
            raise RuntimeError("acquire_fir_ddr requires ddr_readout configuration")
        import time

        ddr = self.ddr_readout_config
        n_points = self.sequence.sweep_point_count
        repetitions = int(self.cfg["reps"])
        n_triggers = n_points * repetitions
        reserved = soc.arm_ddr4_fir_samples(
            ch=ddr.ro_ch,
            n_samples=ddr.samples_per_trigger,
            n_triggers=n_triggers,
            address=ddr.address,
            stride_bytes=ddr.stride_bytes,
            force_overwrite=ddr.force_overwrite,
        )
        self.run_rounds(soc, progress=progress, **run_kwargs)
        if ddr.settle_seconds:
            time.sleep(float(ddr.settle_seconds))
        raw = np.asarray(soc.get_ddr4_fir_samples(
            n_samples=ddr.samples_per_trigger,
            n_triggers=n_triggers,
            start=ddr.address,
            stride_bytes=ddr.stride_bytes,
        ))
        expected_shape = (n_triggers * ddr.samples_per_trigger, 2)
        if raw.shape != expected_shape:
            raise RuntimeError(
                f"unexpected DDR IQ shape {raw.shape}; expected {expected_shape}"
            )
        iq = raw.reshape(
            n_points,
            repetitions,
            ddr.samples_per_trigger,
            2,
        )
        return FineTuneDdrResult(
            sweep_points=self.get_expt_pts(),
            iq=iq,
            reserved_physical_words=reserved,
            sweep_axes=self.sequence.sweep_axes,
            sweep_shape=self.sequence.sweep_shape,
            cross_capacitance=self.sequence.cross_capacitance.copy(),
        )

    def summary(self):
        return {
            "outputs": self.sequence.output_names,
            "awg_channels": self.awg_channels,
            "sweep_points": self.get_expt_pts(),
            "sweep_axes": self.sequence.sweep_axes,
            "sweep_shape": self.sequence.sweep_shape,
            "cartesian_point_count": self.sequence.sweep_point_count,
            "cross_capacitance": self.sequence.cross_capacitance.copy(),
            "segments": tuple(segment.name for segment in self.sequence.segments),
            "repetitions_per_sweep": self.cfg["reps"],
            "commands_per_point": sum(
                len(commands) for commands in self.compiled_points[0].segment_commands
            ),
            "sweep_execution": "tproc_loop_and_add",
            "sweep_dynamic_register_fields": len(self._sweep_fields),
            "sweep_uses_point_table": False,
            "sweep_max_target_quantization_error": self._sweep_max_target_error,
            "sweep_max_step_quantization_error": self._sweep_max_step_error,
            "startup_lead_tproc_cycles_once": self.command_lead_tproc_cycles,
            "point_end_tproc_cycles": self.timing["point_end"],
            "program_instructions": len(self.prog_list),
            "rf_pulse": bool(self.rf_pulse_configs),
            "rf_pulse_count": len(self.rf_pulse_configs),
            "fir_ddr_readout": self.ddr_readout_config is not None,
            "total_ddr_triggers": (
                self.sequence.sweep_point_count * self.cfg["reps"]
                if self.ddr_readout_config is not None
                else 0
            ),
            "aux_timing": dict(self.aux_timing),
        }


__all__ = [
    "AmplitudeSweep",
    "CompiledCommand",
    "CompiledPoint",
    "DdrFirReadoutConfig",
    "FineTuneAmplitudeSweepProgram",
    "FineTuneDdrResult",
    "FineTuneSequence",
    "MAX_OUTPUTS",
    "PulseSegment",
    "ReadoutConfig",
    "RfPulseConfig",
    "compile_sequence",
    "cycles_from_ns",
    "cycles_from_us",
    "dac_to_normalized",
    "normalized_to_dac",
]
