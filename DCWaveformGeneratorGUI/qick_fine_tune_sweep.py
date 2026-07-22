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

Optional Bias-T compensation has two distinct modes. DC compensation appends
the existing opposite-polarity physical-AWG SET after each shot and can keep
either voltage or duration fixed. Filter compensation treats the Bias-T as a
first-order high-pass and replaces each physical flat with a SET followed by a
linear ``target/tau`` slew. Both modes reuse tProcessor hardware loops without
expanding sweep points in PMEM.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
from math import ceil, isfinite
from numbers import Integral, Real
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from qick.averager_program import RAveragerProgram


MAX_OUTPUTS = 8
MAX_RF_ONESHOT_CYCLES = 65535
RF_PERIODIC_WORD_CYCLES = 3
RF_STOP_WORD_CYCLES = 3
OP_SET = 0b01
OP_RAMP = 0b10
COMMAND_REGISTER_NAMES = (
    "target",
    "reserved_start",
    "duration",
    "step",
    "control",
)
DEFAULT_BIAS_T_DURATION_FRAC_BITS = 8
BIAS_T_INSTRUCTION_LEAD_PER_OUTPUT = 32
BIAS_T_COMPENSATION_MODES = ("fixed_voltage", "fixed_time")
BIAS_T_COMPENSATION_TYPES = ("dc", "filter")


def _require_int(value, name: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_positive_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
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
class BiasTCompensationConfig:
    """Per-shot opposite-polarity area compensation settings.

    In ``fixed_voltage`` mode, ``amplitude`` is a positive normalized
    physical-AWG voltage and the compensation duration varies with pulse area.
    In ``fixed_time`` mode, ``fixed_duration_cycles`` is held constant and the
    opposite-polarity voltage varies with pulse area. Every independent output
    starts at one common time.
    """

    amplitude: float
    mode: str = "fixed_voltage"
    fixed_duration_cycles: Optional[int] = None
    duration_frac_bits: int = DEFAULT_BIAS_T_DURATION_FRAC_BITS
    # Legacy API name: this is now one common pre-compensation guard, not a
    # delay inserted between outputs.
    inter_output_gap_cycles: int = 1

    def __post_init__(self):
        amplitude = _require_amplitude(self.amplitude, "Bias-T compensation amplitude")
        if amplitude <= 0.0:
            raise ValueError("Bias-T compensation amplitude must be positive")
        if self.mode not in BIAS_T_COMPENSATION_MODES:
            raise ValueError(
                f"Bias-T compensation mode must be one of "
                f"{BIAS_T_COMPENSATION_MODES}"
            )
        if self.mode == "fixed_time":
            _require_int(
                self.fixed_duration_cycles,
                "Bias-T fixed duration cycles",
                1,
            )
        elif self.fixed_duration_cycles is not None:
            _require_int(
                self.fixed_duration_cycles,
                "Bias-T fixed duration cycles",
                1,
            )
        frac_bits = _require_int(
            self.duration_frac_bits,
            "Bias-T duration fractional bits",
            1,
        )
        if frac_bits > 24:
            raise ValueError("Bias-T duration fractional bits must be <= 24")
        _require_int(
            self.inter_output_gap_cycles,
            "Bias-T common guard cycles",
            0,
        )

    @property
    def compensation_type(self) -> str:
        return "dc"


@dataclass(frozen=True)
class BiasTFilterCompensationConfig:
    """Flat-segment inverse response for a first-order Bias-T high-pass.

    ``tau_cycles`` is the Bias-T time constant in AWG fabric-clock cycles.
    During a flat desired output ``target``, the AWG input is ramped with
    ``dV/dcycle = target / tau_cycles``. This is the exact inverse response
    for an isolated flat level with a correctly initialized filter state.
    """

    tau_cycles: float

    def __post_init__(self):
        _require_positive_real(self.tau_cycles, "Bias-T filter tau_cycles")

    @property
    def compensation_type(self) -> str:
        return "filter"


@dataclass(frozen=True)
class BiasTCompensationPreview:
    """Ideal physical-AWG area and its quantized compensation pulse."""

    output_index: int
    output_name: str
    pulse_area: float
    target_amplitude: float
    duration_cycles: int
    residual_area: float


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
        if _require_int(self.phrst, "readout phrst", 0) != 0:
            raise ValueError("readout phrst is fixed to 0")
        _require_int(self.measure_delay_tproc_cycles, "measure_delay_tproc_cycles", 0)
        _require_int(self.trigger_width_tproc_cycles, "trigger_width_tproc_cycles", 1)


@dataclass(frozen=True)
class RfPulseConfig:
    """RF pulse emitted during a named SET/gate segment.

    Pulses up to 65,535 generator-fabric cycles use the regular one-shot
    command. Longer pulses use a short periodic word followed by a timed
    zero-gain one-shot stop command, avoiding the 16-bit length-field limit.
    """

    gen_ch: int
    at_segment: str
    length_cycles: int
    gain: int
    freq_mhz: float = 0.0
    phase_degrees: float = 0.0
    nqz: int = 1
    delay_tproc_cycles: int = 0
    phrst: int = 0
    stdysel: str = "zero"
    require_within_segment: bool = True

    def __post_init__(self):
        _require_int(self.gen_ch, "gen_ch", 0)
        _require_int(self.length_cycles, "length_cycles", 1)
        _require_int(self.gain, "gain")
        _require_int(self.nqz, "nqz", 1)
        _require_int(self.delay_tproc_cycles, "delay_tproc_cycles", 0)
        if _require_int(self.phrst, "RF output phrst", 0) != 0:
            raise ValueError("RF output phrst is fixed to 0")
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
    command_slot: int = 0


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
        self.bias_t_compensation: Optional[
            Union[BiasTCompensationConfig, BiasTFilterCompensationConfig]
        ] = None

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

    def set_bias_t_compensation(
        self,
        amplitude: Real = 0.1,
        *,
        enabled: bool = True,
        mode: str = "fixed_voltage",
        fixed_duration_cycles: Optional[int] = None,
        duration_frac_bits: int = DEFAULT_BIAS_T_DURATION_FRAC_BITS,
        inter_output_gap_cycles: int = 1,
    ):
        """Enable opposite-polarity Bias-T area compensation per shot.

        The supplied positive amplitude is in normalized physical-AWG units.
        ``fixed_time`` ignores it for output generation and calculates the
        required voltage for ``fixed_duration_cycles``. Compensation is
        applied after the pulse/readout portion of each shot. Disabling the
        option restores the original sequence behavior exactly.
        """
        if not isinstance(enabled, (bool, np.bool_)):
            raise TypeError("Bias-T compensation enabled must be boolean")
        if not enabled:
            self.bias_t_compensation = None
            return self
        self.bias_t_compensation = BiasTCompensationConfig(
            amplitude=float(amplitude),
            mode=str(mode),
            fixed_duration_cycles=fixed_duration_cycles,
            duration_frac_bits=duration_frac_bits,
            inter_output_gap_cycles=inter_output_gap_cycles,
        )
        return self

    def set_bias_t_filter_compensation(
        self,
        tau_cycles: Real,
        *,
        enabled: bool = True,
    ):
        """Enable first-order flat-segment Bias-T filter compensation.

        Each physical flat level starts at its nominal target and ramps at
        ``target / tau_cycles`` for that flat's duration. DC area
        compensation is not appended in this mode.
        """
        if not isinstance(enabled, (bool, np.bool_)):
            raise TypeError("Bias-T filter compensation enabled must be boolean")
        if not enabled:
            self.bias_t_compensation = None
            return self
        self.bias_t_compensation = BiasTFilterCompensationConfig(
            tau_cycles=float(tau_cycles)
        )
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

    def _effective_physical_set_amplitudes_at(
        self,
        point_index: int,
        segment_index: int,
    ) -> Tuple[float, ...]:
        """Resolve held virtual levels and apply cross-capacitance."""
        virtual_values = self._effective_virtual_set_amplitudes_at(
            point_index,
            segment_index,
        )
        physical_values = self.cross_capacitance @ np.asarray(
            virtual_values,
            dtype=float,
        )
        return tuple(
            _require_amplitude(
                float(value),
                f"physical amplitude[{self.output_names[index]}]",
            )
            for index, value in enumerate(physical_values)
        )

    def filter_compensated_segment_levels(
        self,
        point_index: int = 0,
    ) -> Tuple[Tuple[Tuple[float, ...], Tuple[float, ...]], ...]:
        """Return compensated input start/end levels for every segment."""
        config = self.bias_t_compensation
        if not isinstance(config, BiasTFilterCompensationConfig):
            raise ValueError("filter compensation is not enabled")
        self._validate()
        levels = []
        current = None
        for segment_index, segment in enumerate(self.segments):
            if segment.kind == "set":
                start = np.asarray(
                    self._effective_physical_set_amplitudes_at(
                        point_index,
                        segment_index,
                    ),
                    dtype=float,
                )
                end = start + start * (
                    float(segment.duration_cycles) / float(config.tau_cycles)
                )
            else:
                if current is None:
                    raise RuntimeError("filter-compensated RAMP has no start")
                start = current.copy()
                end = np.asarray(
                    self._effective_physical_set_amplitudes_at(
                        point_index,
                        segment_index + 1,
                    ),
                    dtype=float,
                )
            for output_index, value in enumerate(end):
                _require_amplitude(
                    float(value),
                    "Bias-T filter compensated endpoint "
                    f"{self.output_names[output_index]}/{segment.name}",
                )
            levels.append((tuple(start), tuple(end)))
            current = end
        return tuple(levels)

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

    def waveform_vertices(
        self,
        point_index: int = 0,
        *,
        space: str = "physical",
    ):
        """Return the minimum ordered vertices needed to reconstruct a pulse.

        Adjacent vertices are joined by straight lines. Equal adjacent times
        with different values represent an instantaneous SET transition.
        RAMP interiors are intentionally omitted because their two endpoints
        fully define the linear segment.
        """
        self._validate()
        if space == "physical":
            amplitude_getter = self.amplitudes_at
        elif space == "virtual":
            amplitude_getter = self.virtual_amplitudes_at
        else:
            raise ValueError("space must be 'physical' or 'virtual'")

        current = np.asarray(amplitude_getter(point_index, 0), dtype=float)
        times = []
        vertices = []
        boundaries = []

        def append_vertex(time_value: float) -> None:
            if (
                times
                and time_value == times[-1]
                and np.array_equal(current, vertices[-1])
            ):
                return
            times.append(float(time_value))
            vertices.append(current.copy())

        time_now = 0.0
        append_vertex(time_now)
        for segment_index, segment in enumerate(self.segments):
            start_time = time_now
            targets = amplitude_getter(point_index, segment_index)
            if segment.kind == "set":
                if segment_index != 0:
                    for output_index, target in enumerate(targets):
                        if target is not None:
                            current[output_index] = float(target)
                    append_vertex(time_now)
                time_now += segment.duration_cycles
                append_vertex(time_now)
            else:
                for output_index, target in enumerate(targets):
                    if target is not None:
                        current[output_index] = float(target)
                time_now += segment.duration_cycles
                append_vertex(time_now)
            boundaries.append((segment.name, start_time, time_now))

        vertex_matrix = np.asarray(vertices, dtype=float).T
        return (
            np.asarray(times, dtype=float),
            {
                name: vertex_matrix[index]
                for index, name in enumerate(self.output_names)
            },
            tuple(boundaries),
        )

    def bias_t_compensation_preview(
        self,
        point_index: int = 0,
    ) -> Tuple[BiasTCompensationPreview, ...]:
        """Calculate ideal physical-output compensation pulses for one point.

        SET intervals contribute ``level * duration`` and RAMP intervals use
        the exact trapezoid area ``(start + target) * duration / 2``. The
        In fixed-voltage mode, duration is rounded to the nearest whole AWG
        fabric cycle. In fixed-time mode, voltage is rounded to the nearest
        legal 14-effective-bit DAC code for the selected duration.
        """
        self._validate()
        config = self.bias_t_compensation
        if config is None or isinstance(config, BiasTFilterCompensationConfig):
            return ()

        current = np.asarray(self.amplitudes_at(point_index, 0), dtype=float)
        areas = np.zeros(self.n_outputs, dtype=float)
        for segment_index, segment in enumerate(self.segments):
            targets = self.amplitudes_at(point_index, segment_index)
            if segment.kind == "set":
                for output_index, target in enumerate(targets):
                    if target is not None:
                        current[output_index] = float(target)
                areas += current * int(segment.duration_cycles)
            else:
                next_values = current.copy()
                for output_index, target in enumerate(targets):
                    if target is not None:
                        next_values[output_index] = float(target)
                areas += (
                    (current + next_values)
                    * int(segment.duration_cycles)
                    / 2.0
                )
                current = next_values

        previews = []
        for output_index, area in enumerate(areas):
            if np.isclose(area, 0.0, rtol=0.0, atol=1.0e-15):
                target = 0.0
                duration = 0
            elif config.mode == "fixed_time":
                duration = int(config.fixed_duration_cycles)
                ideal_target = -float(area) / duration
                if abs(ideal_target) > 1.0:
                    raise ValueError(
                        "Bias-T fixed compensation time is too short for "
                        f"{self.output_names[output_index]}; required voltage "
                        f"is {ideal_target:+.6g} full scale"
                    )
                target_code = normalized_to_dac(ideal_target)
                target = dac_to_normalized(target_code)
                if target_code == 0:
                    duration = 0
            else:
                target = -config.amplitude if area > 0.0 else config.amplitude
                duration = max(1, int(np.floor(abs(area) / config.amplitude + 0.5)))
            previews.append(BiasTCompensationPreview(
                output_index=output_index,
                output_name=self.output_names[output_index],
                pulse_area=float(area),
                target_amplitude=float(target),
                duration_cycles=duration,
                residual_area=float(area + target * duration),
            ))
        return tuple(previews)

    def filter_compensated_waveform_vertices(self, point_index: int = 0):
        """Return physical AWG input vertices for filter compensation."""
        levels = self.filter_compensated_segment_levels(point_index)
        time_values = []
        columns = []
        boundaries = []
        time_now = 0.0

        def append_vertex(time_value: float, values, *, force: bool = False):
            vector = np.asarray(values, dtype=float)
            if (
                not force
                and time_values
                and float(time_value) == time_values[-1]
                and np.array_equal(vector, columns[-1])
            ):
                return
            time_values.append(float(time_value))
            columns.append(vector.copy())

        for segment, (start, end) in zip(self.segments, levels):
            start_time = time_now
            append_vertex(start_time, start, force=bool(time_values))
            time_now += int(segment.duration_cycles)
            append_vertex(time_now, end)
            boundaries.append((segment.name, start_time, time_now))

        matrix = np.asarray(columns, dtype=float).T
        return (
            np.asarray(time_values, dtype=float),
            {
                name: matrix[index]
                for index, name in enumerate(self.output_names)
            },
            tuple(boundaries),
        )

    def compensated_waveform_vertices(self, point_index: int = 0):
        """Return physical pulse vertices with simultaneous Bias-T starts.

        Physical outputs first return to zero at the end of the user pulse.
        Every active output then receives its opposite-polarity SET at one
        common start time. Each output independently returns to zero after its
        own compensation duration.
        """
        if isinstance(
            self.bias_t_compensation,
            BiasTFilterCompensationConfig,
        ):
            return self.filter_compensated_waveform_vertices(point_index)
        times, waveforms, boundaries = self.waveform_vertices(
            point_index,
            space="physical",
        )
        previews = self.bias_t_compensation_preview(point_index)
        if not previews:
            return times, waveforms, boundaries

        time_values = list(np.asarray(times, dtype=float))
        value_matrix = np.vstack(
            [np.asarray(waveforms[name], dtype=float) for name in self.output_names]
        )
        columns = [value_matrix[:, index].copy() for index in range(value_matrix.shape[1])]
        current = columns[-1].copy()
        boundary_values = list(boundaries)
        time_now = float(time_values[-1])

        def append_vertex(time_value: float, *, force: bool = False) -> None:
            if (
                not force
                and time_values
                and time_value == time_values[-1]
                and np.array_equal(current, columns[-1])
            ):
                return
            time_values.append(float(time_value))
            columns.append(current.copy())

        current[:] = 0.0
        append_vertex(time_now)
        active = [preview for preview in previews if preview.duration_cycles > 0]
        gap = int(self.bias_t_compensation.inter_output_gap_cycles)
        lead = BIAS_T_INSTRUCTION_LEAD_PER_OUTPUT * self.n_outputs
        if active and (gap or lead):
            time_now += gap + lead
            append_vertex(time_now)

        start_time = time_now
        for preview in active:
            current[preview.output_index] = preview.target_amplitude
        if active:
            append_vertex(start_time, force=True)

        for duration in sorted({preview.duration_cycles for preview in active}):
            time_now = start_time + duration
            append_vertex(time_now)
            for preview in active:
                if preview.duration_cycles == duration:
                    current[preview.output_index] = 0.0
            append_vertex(time_now, force=True)

        for preview in active:
            boundary_values.append((
                f"bias_t_comp_{preview.output_name}",
                start_time,
                start_time + preview.duration_cycles,
            ))

        matrix = np.asarray(columns, dtype=float).T
        return (
            np.asarray(time_values, dtype=float),
            {
                name: matrix[index]
                for index, name in enumerate(self.output_names)
            },
            tuple(boundary_values),
        )

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
            if self.bias_t_compensation is None:
                times, waveforms, boundaries = self.sample_waveforms(point_index)
            else:
                times, waveforms, boundaries = self.compensated_waveform_vertices(
                    point_index
                )
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
        tproc_mhz: Optional[Real] = None,
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
            tproc_mhz=tproc_mhz,
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


def _target_code(gen_cfg: Mapping, amplitude: Real) -> int:
    return normalized_to_dac(
        amplitude,
        min_code=int(gen_cfg.get("minv", -32768)),
        max_code=int(gen_cfg.get("maxv", 32764)),
        invalid_lsb=int(gen_cfg.get("dac_invalid_lsb", 2)),
    )


def _ramp_step(
    gen_cfg: Mapping,
    start_code: int,
    target_code: int,
    duration_cycles: int,
    segment_name: str,
) -> Tuple[int, int]:
    duration_samples = int(duration_cycles) * int(gen_cfg["n_pts"])
    if duration_samples > (1 << 23) - 1:
        raise ValueError(
            f"RAMP {segment_name!r} duration expands to {duration_samples} "
            "scalar samples, exceeding the 23-bit command field"
        )
    denominator = max(1, duration_samples - 1)
    numerator = (int(target_code) - int(start_code)) << int(gen_cfg["frac"])
    step = _div_trunc_zero(numerator, denominator)
    if step < -(1 << 23) or step > (1 << 23) - 1:
        raise ValueError(
            f"RAMP {segment_name!r} step {step} exceeds signed 24 bits; "
            "increase duration_cycles or reduce the amplitude span"
        )
    return duration_samples, step


def _compile_filter_point(
    sequence: FineTuneSequence,
    point_index: int,
    channels: Sequence[int],
    gen_cfgs: Sequence[Mapping],
) -> Tuple[Tuple[CompiledCommand, ...], ...]:
    """Compile flat target/tau slew commands without point-table storage."""
    levels = sequence.filter_compensated_segment_levels(point_index)
    current_codes = [None] * sequence.n_outputs
    compiled_segments = []
    for segment_index, (segment, (starts, ends)) in enumerate(
        zip(sequence.segments, levels)
    ):
        commands = []
        for output_index, gen_cfg in enumerate(gen_cfgs):
            start_code = _target_code(gen_cfg, starts[output_index])
            target_code = _target_code(gen_cfg, ends[output_index])
            if segment.kind == "set":
                set_words = _pack_command_words(
                    gen_cfg,
                    start_code,
                    0,
                    0,
                    OP_SET,
                )
                commands.append(
                    CompiledCommand(
                        point_index=point_index,
                        segment_index=segment_index,
                        segment_name=segment.name,
                        output_index=output_index,
                        output_name=sequence.output_names[output_index],
                        gen_ch=channels[output_index],
                        kind="set",
                        target_code=start_code,
                        duration_samples=0,
                        step=0,
                        words=set_words,
                        command_slot=0,
                    )
                )
                current_codes[output_index] = start_code
                command_slot = 1
            else:
                if current_codes[output_index] is None:
                    raise RuntimeError("filter-compensated RAMP has no start code")
                command_slot = 0

            duration_samples, step = _ramp_step(
                gen_cfg,
                int(current_codes[output_index]),
                target_code,
                segment.duration_cycles,
                segment.name,
            )
            ramp_words = _pack_command_words(
                gen_cfg,
                target_code,
                duration_samples,
                step,
                OP_RAMP,
            )
            commands.append(
                CompiledCommand(
                    point_index=point_index,
                    segment_index=segment_index,
                    segment_name=segment.name,
                    output_index=output_index,
                    output_name=sequence.output_names[output_index],
                    gen_ch=channels[output_index],
                    kind="ramp",
                    target_code=target_code,
                    duration_samples=duration_samples,
                    step=step,
                    words=ramp_words,
                    command_slot=command_slot,
                )
            )
            current_codes[output_index] = target_code
        compiled_segments.append(tuple(commands))
    return tuple(compiled_segments)


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
        if isinstance(
            sequence.bias_t_compensation,
            BiasTFilterCompensationConfig,
        ):
            compiled_segments = _compile_filter_point(
                sequence,
                point_index,
                channels,
                gen_cfgs,
            )
            if len(sweep_coordinate) == 1:
                sweep_value = sweep_coordinate[0]
            elif sweep_coordinate:
                sweep_value = sweep_coordinate
            else:
                sweep_value = 0.0
            compiled_points.append(
                CompiledPoint(sweep_value, compiled_segments)
            )
            continue
        current_codes = [None] * sequence.n_outputs
        compiled_segments = []
        for segment_index, segment in enumerate(sequence.segments):
            amplitudes = sequence.amplitudes_at(point_index, segment_index)
            commands = []
            for output_index, amplitude in enumerate(amplitudes):
                if amplitude is None:
                    continue
                gen_cfg = gen_cfgs[output_index]
                target = _target_code(gen_cfg, amplitude)
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
                    duration_samples, step = _ramp_step(
                        gen_cfg,
                        int(current_codes[output_index]),
                        target,
                        segment.duration_cycles,
                        segment.name,
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
        tproc_mhz: Optional[Real] = None,
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
        self.hwh_tproc_mhz = _require_positive_real(
            soccfg["tprocs"][0]["f_time"], "HWH tProcessor clock"
        )
        self.tproc_mhz = (
            self.hwh_tproc_mhz
            if tproc_mhz is None
            else _require_positive_real(tproc_mhz, "tproc_mhz")
        )
        self.tproc_clock_is_manual = tproc_mhz is not None
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
        # QICK's pulse/readout helpers use self.tproccfg for timestamp math.
        # Keep the hardware description intact and override only this program's
        # local timing view when the caller supplied a manual clock.
        self.tproccfg = dict(self.tproccfg)
        self.tproccfg["f_time"] = self.tproc_mhz
        self.awg_channels, self.compiled_points = compile_sequence(
            self.sequence, self.soccfg, self.awg_channel_spec
        )
        self._validate_bias_t_tproc_ports()
        self.bias_t_simultaneous_start_lead_cycles = (
            BIAS_T_INSTRUCTION_LEAD_PER_OUTPUT * len(self.awg_channels)
            if isinstance(
                self.sequence.bias_t_compensation,
                BiasTCompensationConfig,
            )
            else 0
        )
        self._build_sweep_register_plan()
        self._channel_slots = self._build_channel_slots()
        self.timing = self._build_timing()
        self.aux_timing = {}

        self._rf_runtime = {}
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
                phrst=0,
            )

    @staticmethod
    def _command_key(command: CompiledCommand) -> Tuple[int, int, int]:
        return (
            command.segment_index,
            command.output_index,
            command.command_slot,
        )

    @staticmethod
    def _timing_key(command: CompiledCommand):
        """Keep legacy two-item keys for the primary command slot."""
        if command.command_slot == 0:
            return command.segment_index, command.output_index
        return (
            command.segment_index,
            command.output_index,
            command.command_slot,
        )

    def _validate_bias_t_tproc_ports(self):
        """Require independent tProcessor outputs for exact simultaneous SETs."""
        if not isinstance(
            self.sequence.bias_t_compensation,
            BiasTCompensationConfig,
        ):
            return
        channels_by_port = {}
        for output_name, gen_ch in zip(self.sequence.output_names, self.awg_channels):
            port = int(self.soccfg["gens"][gen_ch]["tproc_ch"])
            channels_by_port.setdefault(port, []).append((output_name, gen_ch))
        conflicts = {
            port: channels
            for port, channels in channels_by_port.items()
            if len(channels) > 1
        }
        if conflicts:
            details = "; ".join(
                f"tproc_ch {port}: "
                + ", ".join(f"{name} (gen {gen_ch})" for name, gen_ch in channels)
                for port, channels in sorted(conflicts.items())
            )
            raise ValueError(
                "simultaneous Bias-T compensation requires one independent "
                f"tProcessor output per AWG; shared TMUX ports found: {details}"
            )

    def _compiled_point_area2(self, point: CompiledPoint) -> Tuple[int, ...]:
        """Return twice the ideal pulse area in DAC-code fabric cycles."""
        current = [None] * self.sequence.n_outputs
        area2 = [0] * self.sequence.n_outputs
        for segment, commands in zip(
            self.sequence.segments,
            point.segment_commands,
        ):
            targets = {command.output_index: command.target_code for command in commands}
            duration = int(segment.duration_cycles)
            if segment.kind == "set":
                for output_index, target in targets.items():
                    current[output_index] = int(target)
                if any(value is None for value in current):
                    raise RuntimeError("the first compiled SET must initialize every output")
                for output_index, value in enumerate(current):
                    area2[output_index] += 2 * int(value) * duration
            else:
                for output_index, start in enumerate(current):
                    if start is None:
                        raise RuntimeError("compiled RAMP has no known start value")
                    target = int(targets.get(output_index, start))
                    area2[output_index] += (int(start) + target) * duration
                    current[output_index] = target
        return tuple(area2)

    def _linear_bias_t_models(
        self,
        requested_array,
        sweep_axes,
        sweep_shape,
        *,
        register_name,
        metadata,
    ):
        """Represent a linear per-output Bias-T state with sweep-axis adds."""
        models = []
        for output_index in range(self.sequence.n_outputs):
            base = int(requested_array[0, output_index])
            quantum = int(metadata[output_index].get("quantum", 1))
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
                endpoint = int(requested_array[endpoint_index, output_index])
                units = _round_div_nearest(
                    endpoint - base,
                    (axis.count - 1) * quantum,
                )
                axis_deltas.append(units * quantum)
            model = {
                "key": ("bias_t", output_index, register_name),
                "register_name": register_name,
                "base": base,
                "axis_deltas": tuple(axis_deltas),
                "output_index": output_index,
                "gen_ch": self.awg_channels[output_index],
            }
            model.update(metadata[output_index])
            models.append(model)

        actual = np.empty_like(requested_array)
        max_error = 0
        for point_index in range(len(self.compiled_points)):
            indices = (
                np.unravel_index(point_index, sweep_shape, order="C")
                if sweep_axes
                else ()
            )
            for output_index, model in enumerate(models):
                value = int(model["base"]) + sum(
                    int(index) * int(delta)
                    for index, delta in zip(indices, model["axis_deltas"])
                )
                actual[point_index, output_index] = value
                max_error = max(
                    max_error,
                    abs(value - int(requested_array[point_index, output_index])),
                )
        return tuple(models), actual, int(max_error)

    def _build_bias_t_models(self, sweep_axes, sweep_shape):
        """Build fixed-voltage duration or fixed-time target-code states."""
        config = self.sequence.bias_t_compensation
        self._bias_t_mode = (
            None
            if config is None
            else (
                "filter"
                if isinstance(config, BiasTFilterCompensationConfig)
                else config.mode
            )
        )
        self._bias_t_comp_codes = ()
        self._bias_t_duration_q_requested = np.empty((0, 0), dtype=np.int64)
        self._bias_t_duration_q_actual = np.empty((0, 0), dtype=np.int64)
        self._bias_t_target_code_requested = np.empty((0, 0), dtype=np.int64)
        self._bias_t_target_code_actual = np.empty((0, 0), dtype=np.int64)
        self._bias_t_max_duration_q_error = 0
        self._bias_t_max_target_code_error = 0
        if config is None or isinstance(config, BiasTFilterCompensationConfig):
            self._bias_t_fields = ()
            return ()

        if config.mode == "fixed_time":
            duration = int(config.fixed_duration_cycles)
            metadata = []
            limits = []
            for gen_ch in self.awg_channels:
                gen_cfg = self.soccfg["gens"][gen_ch]
                minimum = int(gen_cfg.get("minv", -32768))
                maximum = int(gen_cfg.get("maxv", 32764))
                quantum = 1 << int(gen_cfg.get("dac_invalid_lsb", 2))
                fabric_mhz = Fraction(str(gen_cfg["f_fabric"]))
                duration_ratio = (
                    Fraction(duration) * Fraction(str(self.tproc_mhz)) / fabric_mhz
                )
                duration_tproc = max(
                    1,
                    (duration_ratio.numerator + duration_ratio.denominator - 1)
                    // duration_ratio.denominator,
                )
                limits.append((minimum, maximum, quantum))
                metadata.append(
                    {
                        "fixed_duration_fabric_cycles": duration,
                        "fixed_duration_tproc_cycles": int(duration_tproc),
                        "quantum": quantum,
                    }
                )

            requested = []
            for point in self.compiled_points:
                point_values = []
                for output_index, signed_area2 in enumerate(
                    self._compiled_point_area2(point)
                ):
                    minimum, maximum, quantum = limits[output_index]
                    units = _round_div_nearest(
                        -int(signed_area2),
                        2 * duration * quantum,
                    )
                    target_code = units * quantum
                    if target_code < minimum or target_code > maximum:
                        raise ValueError(
                            "Bias-T fixed compensation time is too short for "
                            f"{self.sequence.output_names[output_index]}; required "
                            f"DAC code {target_code} is outside "
                            f"[{minimum}, {maximum}]"
                        )
                    point_values.append(target_code)
                requested.append(tuple(point_values))
            requested_array = np.asarray(requested, dtype=np.int64)
            models, actual, max_error = self._linear_bias_t_models(
                requested_array,
                sweep_axes,
                sweep_shape,
                register_name="bias_t_target_code",
                metadata=metadata,
            )
            self._bias_t_target_code_requested = requested_array
            self._bias_t_target_code_actual = actual
            self._bias_t_max_target_code_error = max_error
            return models

        scale = 1 << int(config.duration_frac_bits)
        rounding = scale >> 1
        comp_codes = []
        metadata = []
        for gen_ch in self.awg_channels:
            gen_cfg = self.soccfg["gens"][gen_ch]
            kwargs = {
                "min_code": int(gen_cfg.get("minv", -32768)),
                "max_code": int(gen_cfg.get("maxv", 32764)),
                "invalid_lsb": int(gen_cfg.get("dac_invalid_lsb", 2)),
            }
            positive = normalized_to_dac(config.amplitude, **kwargs)
            negative = normalized_to_dac(-config.amplitude, **kwargs)
            if positive <= 0 or negative >= 0:
                raise ValueError(
                    "Bias-T compensation voltage is below one legal DAC code"
                )
            comp_codes.append((positive, negative))
            metadata.append(
                {
                    "positive_code": int(positive),
                    "negative_code": int(negative),
                    "duration_frac_bits": int(config.duration_frac_bits),
                }
            )

        requested = []
        for point in self.compiled_points:
            area2 = self._compiled_point_area2(point)
            point_values = []
            for output_index, signed_area2 in enumerate(area2):
                if signed_area2 == 0:
                    duration_q = 0
                else:
                    positive, negative = comp_codes[output_index]
                    target_magnitude = abs(negative) if signed_area2 > 0 else positive
                    gen_ch = self.awg_channels[output_index]
                    fabric_mhz = Fraction(
                        str(self.soccfg["gens"][gen_ch]["f_fabric"])
                    )
                    ratio = Fraction(
                        int(signed_area2) * scale,
                        2 * int(target_magnitude),
                    ) * Fraction(str(self.tproc_mhz)) / fabric_mhz
                    duration_q = _round_div_nearest(
                        ratio.numerator,
                        ratio.denominator,
                    )
                if abs(duration_q) + rounding > (1 << 31) - 1:
                    raise ValueError(
                        "Bias-T compensation duration exceeds the signed 32-bit "
                        "tProcessor fixed-point range; increase compensation voltage"
                    )
                point_values.append(duration_q)
            requested.append(tuple(point_values))
        requested_array = np.asarray(requested, dtype=np.int64)
        models, actual, max_error = self._linear_bias_t_models(
            requested_array,
            sweep_axes,
            sweep_shape,
            register_name="bias_t_duration_q",
            metadata=metadata,
        )
        self._bias_t_comp_codes = tuple(comp_codes)
        self._bias_t_duration_q_requested = requested_array
        self._bias_t_duration_q_actual = actual
        self._bias_t_max_duration_q_error = max_error
        return models

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
                models[(*key, register_name)] = {
                    "key": (*key, register_name),
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
                    command_key = self._command_key(command)
                    target_model = models[(*command_key, "target")]
                    step_model = models[(*command_key, "step")]
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

        bias_t_models = self._build_bias_t_models(sweep_axes, sweep_shape)

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
        for model in bias_t_models:
            page, _target_register = self._gen_regmap[
                (int(model["gen_ch"]), "target")
            ]
            field = dict(model)
            field["page"] = int(page)
            fields.append(field)

        occupied = {page: {0} for page in range(8)}
        for register_map in (self._gen_regmap, self._ro_regmap):
            for page, register in register_map.values():
                occupied[int(page)].add(int(register))
        # These are used below for the shot count, repetition loop, and trigger.
        occupied[0].update((13, 15, 16))

        dmem_size = int(self.tproccfg.get("dmem_size", 0))
        next_dmem_addr = dmem_size - 1

        def assign_dmem(field, work_register):
            nonlocal next_dmem_addr
            # DMEM address 1 is owned by the acquisition counter. Allocate
            # sweep backing words downward from the end of memory.
            if next_dmem_addr <= 1:
                raise RuntimeError(
                    "tProcessor DMEM has no room for spilled sweep state"
                )
            field["storage"] = "dmem"
            field["dmem_addr"] = next_dmem_addr
            field["work_register"] = int(work_register)
            next_dmem_addr -= 1

        for field in fields:
            page = field["page"]
            if field.get("register_name") in {
                "bias_t_duration_q",
                "bias_t_target_code",
            }:
                work_page, work_register = self._gen_regmap[
                    (int(field["gen_ch"]), "duration")
                ]
                if int(work_page) != page:
                    raise RuntimeError(
                        "AWG duration and Bias-T state registers must share one page"
                    )
                # Compensation uses OP_SET, whose duration field is ignored by
                # the AWG RTL. It is therefore a safe calculation scratch reg.
                assign_dmem(field, work_register)
                field["work_register_source"] = "awg_duration_command"
                continue

            available = [
                register for register in range(1, 32) if register not in occupied[page]
            ]
            if not available:
                # Arithmetic cannot operate on DMEM directly, so use the
                # destination command register as a temporary load/add/store
                # register when this state is consumed or advanced.
                assign_dmem(field, int(field["command_register"]))
                field["work_register_source"] = "awg_command"
                continue
            field["storage"] = "register"
            field["state_register"] = available[0]
            occupied[page].add(available[0])

        if bias_t_models:
            if next_dmem_addr <= 1:
                raise RuntimeError(
                    "tProcessor DMEM has no room for the Bias-T maximum duration"
                )
            self._bias_t_max_duration_dmem_addr = next_dmem_addr
            next_dmem_addr -= 1
        else:
            self._bias_t_max_duration_dmem_addr = None

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
        self._bias_t_fields = tuple(
            field
            for field in fields
            if str(field.get("register_name", "")).startswith("bias_t_")
        )
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
        periodic = rf.length_cycles > MAX_RF_ONESHOT_CYCLES
        self._rf_runtime[rf.gen_ch] = {
            "freq": int(freq_word),
            "phase": int(phase_word),
            "periodic": bool(periodic),
        }
        self.set_pulse_registers(
            ch=rf.gen_ch,
            style="const",
            freq=freq_word,
            phase=phase_word,
            gain=rf.gain,
            length=(
                RF_PERIODIC_WORD_CYCLES if periodic else rf.length_cycles
            ),
            phrst=0,
            stdysel="last" if periodic else rf.stdysel,
            mode="periodic" if periodic else "oneshot",
        )

    def _emit_rf_start(self, rf: RfPulseConfig, tproc_time: int) -> None:
        runtime = self._rf_runtime[rf.gen_ch]
        if runtime["periodic"]:
            # A previous point leaves the generator configured with the stop
            # word, so restore the periodic tone before every new start.
            self.set_pulse_registers(
                ch=rf.gen_ch,
                style="const",
                freq=runtime["freq"],
                phase=runtime["phase"],
                gain=rf.gain,
                length=RF_PERIODIC_WORD_CYCLES,
                phrst=0,
                stdysel="last",
                mode="periodic",
            )
        self.pulse(ch=rf.gen_ch, t=tproc_time)

    def _emit_rf_stop(self, rf: RfPulseConfig, tproc_time: int) -> None:
        runtime = self._rf_runtime[rf.gen_ch]
        self.set_pulse_registers(
            ch=rf.gen_ch,
            style="const",
            freq=runtime["freq"],
            phase=runtime["phase"],
            gain=0,
            length=RF_STOP_WORD_CYCLES,
            phrst=0,
            stdysel="zero",
            mode="oneshot",
        )
        self.pulse(ch=rf.gen_ch, t=tproc_time)

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
            phrst=0,
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
        f_time = self.tproc_mhz

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
        for timing_key, command_time in self.timing["command_times"].items():
            output_index = int(timing_key[1])
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
            requested_end = rf_start + self._fabric_to_tproc(
                rf.gen_ch,
                rf.length_cycles,
            )
            periodic = bool(self._rf_runtime[rf.gen_ch]["periodic"])
            rf_end = int(requested_end)
            if periodic:
                while rf_end in occupied:
                    rf_end += self.command_spacing_tproc_cycles
                occupied.add(rf_end)
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
                "requested_end": int(requested_end),
                "end": int(rf_end),
                "mode": "periodic_timed_stop" if periodic else "oneshot",
                "command_skew_tproc_cycles": int(rf_start - requested_start),
                "stop_skew_tproc_cycles": int(rf_end - requested_end),
            }
            rf_timings.append(timing)
            self.aux_timing.update({
                f"rf_{rf_index}_requested_start": timing["requested_start"],
                f"rf_{rf_index}_start": timing["start"],
                f"rf_{rf_index}_end": timing["end"],
                f"rf_{rf_index}_mode": timing["mode"],
                f"rf_{rf_index}_command_skew_tproc_cycles": (
                    timing["command_skew_tproc_cycles"]
                ),
                f"rf_{rf_index}_stop_skew_tproc_cycles": (
                    timing["stop_skew_tproc_cycles"]
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
                "rf_mode": timing["mode"],
                "rf_command_skew_tproc_cycles": timing["command_skew_tproc_cycles"],
                "rf_stop_skew_tproc_cycles": timing["stop_skew_tproc_cycles"],
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
        f_time = self.tproc_mhz
        f_fabric = float(self.soccfg["gens"][gen_ch]["f_fabric"])
        if f_time <= 0 or f_fabric <= 0:
            raise ValueError("tProcessor and AWG fabric clocks must be positive")
        return max(1, int(ceil(fabric_cycles * f_time / f_fabric - 1e-12)))

    def _build_filter_timing(self):
        """Schedule SET+RAMP flat compensation without dropped commands.

        Filter-compensated segments are consecutive RAMP operations. The next
        command is therefore issued only after the previous RAMP and guard
        have completed. A flat begins after its SET command and the following
        RAMP startup pipeline; this latency is included in all anchor times.
        """
        first_point = self.compiled_points[0]
        command_times = {}
        segment_starts = []
        segment_ends = []
        time_now = 0
        for segment_index, segment in enumerate(self.sequence.segments):
            output_starts = []
            output_ends = []
            commands_by_output = {
                output_index: sorted(
                    (
                        command
                        for command in first_point.segment_commands[segment_index]
                        if command.output_index == output_index
                    ),
                    key=lambda command: command.command_slot,
                )
                for output_index in range(self.sequence.n_outputs)
            }
            for output_index, gen_ch in enumerate(self.awg_channels):
                commands = commands_by_output[output_index]
                ramp_command = commands[-1]
                if ramp_command.kind != "ramp":
                    raise RuntimeError("filter segment must end with a RAMP command")
                shared_slot = int(self._channel_slots[gen_ch])
                if segment.kind == "set":
                    set_command = commands[0]
                    set_time = time_now + 2 * shared_slot
                    ramp_time = set_time + self.command_spacing_tproc_cycles
                    command_times[self._timing_key(set_command)] = set_time
                else:
                    ramp_time = time_now + shared_slot
                command_times[self._timing_key(ramp_command)] = ramp_time

                gen_cfg = self.soccfg["gens"][gen_ch]
                startup_cycles = int(
                    gen_cfg.get("ramp_startup_latency_cycles", 5)
                )
                startup_tproc = self._fabric_to_tproc(
                    gen_ch,
                    startup_cycles,
                )
                output_starts.append(ramp_time + startup_tproc)
                occupancy = (
                    startup_cycles
                    + segment.duration_cycles
                    + int(gen_cfg.get("ramp_guard_cycles", 1))
                )
                output_ends.append(
                    ramp_time + self._fabric_to_tproc(gen_ch, occupancy)
                )
            segment_starts.append(max(output_starts))
            time_now = max(output_ends)
            segment_ends.append(time_now)

        return {
            "command_times": command_times,
            "segment_starts": tuple(segment_starts),
            "segment_ends": tuple(segment_ends),
            "point_end": int(time_now),
        }

    def _build_timing(self):
        if isinstance(
            self.sequence.bias_t_compensation,
            BiasTFilterCompensationConfig,
        ):
            return self._build_filter_timing()
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
                command_times[self._timing_key(command)] = command_time
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
                (*self._command_key(command), register_name)
            )
            comment = f"{command.output_name}:{command.segment_name}:{register_name}"
            if field is None:
                self.safe_regwi(page, reg, int(word), comment)
            elif field.get("storage") == "dmem":
                self.memri(
                    page,
                    reg,
                    int(field["dmem_addr"]),
                    f"load spilled {comment}",
                )
                if register_name == "step":
                    self.bitwi(
                        page,
                        reg,
                        reg,
                        "&",
                        0xFFFFFF,
                        comment,
                    )
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

    def _static_set_command(
        self,
        output_index: int,
        target_code: int,
        segment_name: str,
    ) -> CompiledCommand:
        """Build an unswept SET command used by the compensation epilogue."""
        gen_ch = self.awg_channels[output_index]
        words = _pack_command_words(
            self.soccfg["gens"][gen_ch],
            int(target_code),
            0,
            0,
            OP_SET,
        )
        return CompiledCommand(
            point_index=0,
            segment_index=-1,
            segment_name=segment_name,
            output_index=output_index,
            output_name=self.sequence.output_names[output_index],
            gen_ch=gen_ch,
            kind="set",
            target_code=int(target_code),
            duration_samples=0,
            step=0,
            words=words,
        )

    def _emit_bias_t_compensation(self):
        """Schedule simultaneous SET starts and per-channel SET-zero stops."""
        config = self.sequence.bias_t_compensation
        if config is None or isinstance(config, BiasTFilterCompensationConfig):
            return
        if config.mode == "fixed_time":
            self._emit_bias_t_fixed_time()
            return

        max_duration_addr = int(self._bias_t_max_duration_dmem_addr)
        first_field = self._bias_t_fields[0]
        first_page = int(first_field["page"])
        first_target_register = self._gen_regmap[
            (int(first_field["gen_ch"]), "target")
        ][1]
        self.safe_regwi(
            first_page,
            first_target_register,
            0,
            "clear maximum Bias-T duration",
        )
        self.memwi(
            first_page,
            first_target_register,
            max_duration_addr,
            "store cleared maximum Bias-T duration",
        )

        for field in self._bias_t_fields:
            output_index = int(field["output_index"])
            output_name = self.sequence.output_names[output_index]
            gen_ch = int(field["gen_ch"])
            page = int(field["page"])
            work_register = int(field["work_register"])
            target_register = self._gen_regmap[(gen_ch, "target")][1]
            register_names = COMMAND_REGISTER_NAMES[1:]
            command_words = _pack_command_words(
                self.soccfg["gens"][gen_ch],
                0,
                0,
                0,
                OP_SET,
            )
            for register_name, word in zip(register_names, command_words[1:]):
                reg_page, register = self._gen_regmap[(gen_ch, register_name)]
                if int(reg_page) != page:
                    raise RuntimeError("AWG command registers must share one page")
                self.safe_regwi(
                    page,
                    register,
                    int(word),
                    f"{output_name} Bias-T {register_name}",
                )
            _, time_register = self._gen_regmap[(gen_ch, "t")]
            self.safe_regwi(
                page,
                time_register,
                self.bias_t_simultaneous_start_lead_cycles,
                f"{output_name} common Bias-T start offset",
            )
            self.memri(
                page,
                work_register,
                int(field["dmem_addr"]),
                f"load signed Bias-T duration for {output_name}",
            )

            negative_label = f"BIAS_T_NEGATIVE_AREA_{output_index}"
            ready_label = f"BIAS_T_DURATION_READY_{output_index}"
            max_ready_label = f"BIAS_T_MAX_READY_{output_index}"
            done_label = f"BIAS_T_DONE_{output_index}"
            self.condj(
                page,
                work_register,
                "==",
                0,
                done_label,
                f"skip zero-area Bias-T output {output_name}",
            )
            self.condj(
                page,
                work_register,
                "<",
                0,
                negative_label,
                f"select Bias-T polarity for {output_name}",
            )
            # Positive pulse area requires a negative compensation voltage.
            self.safe_regwi(
                page,
                target_register,
                int(field["negative_code"]),
                f"{output_name} negative Bias-T target",
            )
            self.condj(page, 0, "==", 0, ready_label, "Bias-T polarity ready")

            self.label(negative_label)
            self.math(
                page,
                work_register,
                0,
                "-",
                work_register,
                f"absolute Bias-T duration for {output_name}",
            )
            self.safe_regwi(
                page,
                target_register,
                int(field["positive_code"]),
                f"{output_name} positive Bias-T target",
            )

            self.label(ready_label)
            frac_bits = int(field["duration_frac_bits"])
            self.mathi(
                page,
                work_register,
                work_register,
                "+",
                1 << (frac_bits - 1),
                f"round Bias-T duration for {output_name}",
            )
            self.bitwi(
                page,
                work_register,
                work_register,
                ">>",
                frac_bits,
                f"Bias-T duration in tProcessor cycles for {output_name}",
            )
            self.condj(
                page,
                work_register,
                "==",
                0,
                done_label,
                f"skip sub-cycle Bias-T output {output_name}",
            )
            command_registers = [
                self._gen_regmap[(gen_ch, name)][1]
                for name in COMMAND_REGISTER_NAMES
            ]
            self.set(
                int(self.soccfg["gens"][gen_ch]["tproc_ch"]),
                page,
                *command_registers,
                time_register,
                f"start Bias-T compensation on {output_name}",
            )
            self.safe_regwi(
                page,
                target_register,
                0,
                f"{output_name} return to zero after Bias-T compensation",
            )
            self.mathi(
                page,
                time_register,
                work_register,
                "+",
                self.bias_t_simultaneous_start_lead_cycles,
                f"{output_name} Bias-T stop offset",
            )
            self.set(
                int(self.soccfg["gens"][gen_ch]["tproc_ch"]),
                page,
                *command_registers,
                time_register,
                f"finish Bias-T compensation on {output_name}",
            )

            # The target word is no longer needed after both commands have
            # been queued, so reuse its register for the running maximum.
            self.memri(
                page,
                target_register,
                max_duration_addr,
                f"load maximum Bias-T duration for {output_name}",
            )
            self.condj(
                page,
                work_register,
                "<=",
                target_register,
                max_ready_label,
                f"keep maximum Bias-T duration after {output_name}",
            )
            self.memwi(
                page,
                work_register,
                max_duration_addr,
                f"update maximum Bias-T duration from {output_name}",
            )
            self.label(max_ready_label)
            self.mathi(
                page,
                work_register,
                work_register,
                "+",
                0,
                f"maximum Bias-T duration checked for {output_name}",
            )

            self.label(done_label)
            self.mathi(
                page,
                work_register,
                work_register,
                "+",
                0,
                f"Bias-T output {output_name} complete",
            )

        final_page = int(first_field["page"])
        final_register = int(first_field["work_register"])
        no_compensation_label = "BIAS_T_NO_ACTIVE_OUTPUT"
        self.memri(
            final_page,
            final_register,
            max_duration_addr,
            "load maximum Bias-T duration",
        )
        self.condj(
            final_page,
            final_register,
            "==",
            0,
            no_compensation_label,
            "skip Bias-T timing advance when all areas are zero",
        )
        self.mathi(
            final_page,
            final_register,
            final_register,
            "+",
            self.bias_t_simultaneous_start_lead_cycles,
            "include common Bias-T command lead",
        )
        # RTL sync is additive: advance once to the latest channel stop.
        self.sync(
            final_page,
            final_register,
            "advance to latest simultaneous Bias-T stop",
        )
        self.label(no_compensation_label)
        self.mathi(
            final_page,
            final_register,
            final_register,
            "+",
            0,
            "simultaneous Bias-T compensation complete",
        )

    def _emit_bias_t_fixed_time(self):
        """Emit fixed-time compensation with a sweep-dependent SET voltage."""
        active_addr = int(self._bias_t_max_duration_dmem_addr)
        first_field = self._bias_t_fields[0]
        first_page = int(first_field["page"])
        first_target_register = self._gen_regmap[
            (int(first_field["gen_ch"]), "target")
        ][1]
        self.safe_regwi(
            first_page,
            first_target_register,
            0,
            "clear fixed-time Bias-T active marker",
        )
        self.memwi(
            first_page,
            first_target_register,
            active_addr,
            "store cleared fixed-time Bias-T active marker",
        )

        for field in self._bias_t_fields:
            output_index = int(field["output_index"])
            output_name = self.sequence.output_names[output_index]
            gen_ch = int(field["gen_ch"])
            page = int(field["page"])
            work_register = int(field["work_register"])
            target_register = self._gen_regmap[(gen_ch, "target")][1]
            _, time_register = self._gen_regmap[(gen_ch, "t")]
            command_words = _pack_command_words(
                self.soccfg["gens"][gen_ch],
                0,
                0,
                0,
                OP_SET,
            )
            for register_name, word in zip(
                COMMAND_REGISTER_NAMES[1:],
                command_words[1:],
            ):
                reg_page, register = self._gen_regmap[(gen_ch, register_name)]
                if int(reg_page) != page:
                    raise RuntimeError("AWG command registers must share one page")
                self.safe_regwi(
                    page,
                    register,
                    int(word),
                    f"{output_name} fixed-time Bias-T {register_name}",
                )

            self.safe_regwi(
                page,
                time_register,
                self.bias_t_simultaneous_start_lead_cycles,
                f"{output_name} common fixed-time Bias-T start offset",
            )
            self.memri(
                page,
                work_register,
                int(field["dmem_addr"]),
                f"load fixed-time Bias-T target for {output_name}",
            )
            done_label = f"BIAS_T_FIXED_TIME_DONE_{output_index}"
            self.condj(
                page,
                work_register,
                "==",
                0,
                done_label,
                f"skip zero-area fixed-time Bias-T output {output_name}",
            )
            self.mathi(
                page,
                target_register,
                work_register,
                "+",
                0,
                f"apply fixed-time Bias-T target for {output_name}",
            )
            command_registers = [
                self._gen_regmap[(gen_ch, name)][1]
                for name in COMMAND_REGISTER_NAMES
            ]
            self.set(
                int(self.soccfg["gens"][gen_ch]["tproc_ch"]),
                page,
                *command_registers,
                time_register,
                f"start fixed-time Bias-T compensation on {output_name}",
            )
            self.safe_regwi(
                page,
                target_register,
                0,
                f"{output_name} return to zero after fixed-time Bias-T",
            )
            self.safe_regwi(
                page,
                time_register,
                self.bias_t_simultaneous_start_lead_cycles
                + int(field["fixed_duration_tproc_cycles"]),
                f"{output_name} fixed-time Bias-T stop offset",
            )
            self.set(
                int(self.soccfg["gens"][gen_ch]["tproc_ch"]),
                page,
                *command_registers,
                time_register,
                f"finish fixed-time Bias-T compensation on {output_name}",
            )
            self.memwi(
                page,
                work_register,
                active_addr,
                f"mark fixed-time Bias-T active for {output_name}",
            )
            self.label(done_label)
            self.mathi(
                page,
                work_register,
                work_register,
                "+",
                0,
                f"fixed-time Bias-T output {output_name} complete",
            )

        final_page = int(first_field["page"])
        final_register = int(first_field["work_register"])
        no_compensation_label = "BIAS_T_FIXED_TIME_NO_ACTIVE_OUTPUT"
        self.memri(
            final_page,
            final_register,
            active_addr,
            "load fixed-time Bias-T active marker",
        )
        self.condj(
            final_page,
            final_register,
            "==",
            0,
            no_compensation_label,
            "skip fixed-time Bias-T advance when all areas are zero",
        )
        final_offset = self.bias_t_simultaneous_start_lead_cycles + max(
            int(field["fixed_duration_tproc_cycles"])
            for field in self._bias_t_fields
        )
        self.safe_regwi(
            final_page,
            final_register,
            final_offset,
            "fixed-time Bias-T latest stop offset",
        )
        self.sync(
            final_page,
            final_register,
            "advance to fixed-time Bias-T stop",
        )
        self.label(no_compensation_label)
        self.mathi(
            final_page,
            final_register,
            final_register,
            "+",
            0,
            "fixed-time Bias-T compensation complete",
        )

    def _emit_point(self):
        # Timed-output queues are FIFO ordered. Generators behind the same TMUX
        # therefore must be enqueued in timestamp order, even when their
        # commands target different downstream IPs.
        scheduled_events = []
        insertion_order = 0

        def schedule(time, priority, kind, payload):
            nonlocal insertion_order
            scheduled_events.append((
                int(time),
                int(priority),
                insertion_order,
                kind,
                payload,
            ))
            insertion_order += 1

        # The instruction body is point-independent. Swept SET/RAMP fields are
        # copied from state registers immediately before each command.
        point = self.compiled_points[0]
        for segment_index, commands in enumerate(point.segment_commands):
            for command in commands:
                command_time = self.timing["command_times"][
                    self._timing_key(command)
                ]
                schedule(command_time, 10, "awg", command)

        if self.ddr_readout_config is not None:
            ddr = self.ddr_readout_config
            schedule(
                self.aux_timing["ddr_readout_start"],
                0,
                "readout",
                ddr.ro_ch,
            )
            schedule(
                self.aux_timing["ddr_trigger_time"],
                20,
                "ddr_trigger",
                ddr,
            )

        for rf_index, rf_config in enumerate(self.rf_pulse_configs):
            schedule(
                self.aux_timing[f"rf_{rf_index}_start"],
                10,
                "rf_start",
                rf_config,
            )
            if self._rf_runtime[rf_config.gen_ch]["periodic"]:
                schedule(
                    self.aux_timing[f"rf_{rf_index}_end"],
                    15,
                    "rf_stop",
                    rf_config,
                )

        point_end = int(self.timing["point_end"])
        bias_t_static_end = point_end
        if isinstance(
            self.sequence.bias_t_compensation,
            BiasTCompensationConfig,
        ):
            # End the user-defined AWG pulse before the compensation epilogue.
            # This also prevents a long FIR/readout window from adding an
            # unmodeled final-level hold to the area being compensated.
            pulse_end = int(self.timing["segment_ends"][-1])
            for output_index, gen_ch in enumerate(self.awg_channels):
                zero_time = pulse_end + int(self._channel_slots[gen_ch])
                schedule(
                    zero_time,
                    30,
                    "awg",
                    self._static_set_command(
                        output_index,
                        0,
                        "bias_t_pre_zero",
                    ),
                )
                bias_t_static_end = max(bias_t_static_end, zero_time)
            self.aux_timing["bias_t_user_pulse_end"] = pulse_end
            self.aux_timing["bias_t_static_end"] = bias_t_static_end

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
            schedule(measure_time, 0, "readout", ro.ro_ch)
            schedule(
                measure_time,
                20,
                "adc_trigger",
                ro,
            )

        for event_time, _priority, _order, kind, payload in sorted(
            scheduled_events,
            key=lambda item: item[:3],
        ):
            if kind == "awg":
                self._emit_command(payload, event_time)
            elif kind == "readout":
                self.readout(ch=payload, t=event_time)
            elif kind == "rf_start":
                self._emit_rf_start(payload, event_time)
            elif kind == "rf_stop":
                self._emit_rf_stop(payload, event_time)
            elif kind == "ddr_trigger":
                self.trigger(
                    ddr4=True,
                    adc_trig_offset=0,
                    t=event_time,
                    width=payload.trigger_width_tproc_cycles,
                )
            elif kind == "adc_trigger":
                self.trigger(
                    adcs=[payload.ro_ch],
                    adc_trig_offset=event_time,
                    t=0,
                    width=payload.trigger_width_tproc_cycles,
                )
            else:
                raise RuntimeError(f"unknown scheduled event kind {kind!r}")

        for gen_ch in self.awg_channels:
            self.set_timestamp(bias_t_static_end, gen_ch=gen_ch)

        if self.readout_config is not None:
            ro = self.readout_config
            if ro.wait:
                self.wait_all()

        if not isinstance(
            self.sequence.bias_t_compensation,
            BiasTCompensationConfig,
        ):
            self.sync_all(self.recovery_tproc_cycles)
        else:
            # Move the reference beyond all statically timed AWG/RF/readout
            # events. Dynamic compensation uses one common start timestamp,
            # channel-specific stop timestamps, and one final max-duration sync.
            self.sync_all(0)
            if self.sequence.bias_t_compensation.inter_output_gap_cycles:
                self.synci(
                    int(
                        self.sequence.bias_t_compensation.inter_output_gap_cycles
                    ),
                    "common Bias-T guard after pre-compensation zero",
                )
            self._emit_bias_t_compensation()
            if self.recovery_tproc_cycles:
                self.synci(
                    self.recovery_tproc_cycles,
                    "post Bias-T shot recovery",
                )

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
            if field.get("storage") == "dmem":
                work_register = int(field["work_register"])
                dmem_addr = int(field["dmem_addr"])
                self.memri(
                    page,
                    work_register,
                    dmem_addr,
                    f"load {action} axis {axis_index} {field['key']}",
                )
                self.mathi(
                    page,
                    work_register,
                    work_register,
                    "+",
                    amount,
                    f"{action} axis {axis_index} {field['key']}",
                )
                self.memwi(
                    page,
                    work_register,
                    dmem_addr,
                    f"store {action} axis {axis_index} {field['key']}",
                )
                continue
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
            page = int(field["page"])
            if field.get("storage") == "dmem":
                work_register = int(field["work_register"])
                self.safe_regwi(
                    page,
                    work_register,
                    int(field["base"]),
                    f"initialize DMEM sweep state {field['key']}",
                )
                self.memwi(
                    page,
                    work_register,
                    int(field["dmem_addr"]),
                    f"store initial DMEM sweep state {field['key']}",
                )
            else:
                self.safe_regwi(
                    page,
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

    def _run_rounds_with_counter_progress(
        self,
        soc,
        progress_callback,
        *,
        poll_interval_seconds: float = 0.02,
    ) -> None:
        """Run without readout streaming and report the real tProc loop count."""
        import time

        total_per_round = int(np.prod(self.loop_dims, dtype=np.int64))
        rounds = int(self.rounds)
        total = total_per_round * rounds
        self.config_all(soc, load_envelopes=True, load_mem=False)
        progress_callback(0, total)
        completed_rounds = 0
        try:
            for _round_index in range(rounds):
                soc.reload_mem()
                soc.clear_tproc_counter(addr=self.counter_addr)
                soc.start_src("internal")
                soc.start_tproc()
                count = 0
                while count < total_per_round:
                    count = min(
                        total_per_round,
                        int(soc.get_tproc_counter(addr=self.counter_addr)),
                    )
                    progress_callback(completed_rounds + count, total)
                    if count < total_per_round:
                        time.sleep(poll_interval_seconds)
                completed_rounds += total_per_round
        finally:
            soc.start_src("internal")

    def acquire_fir_ddr(
        self,
        soc,
        *,
        progress: bool = True,
        counter_progress=None,
        **run_kwargs,
    ):
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
        if counter_progress is None:
            self.run_rounds(soc, progress=progress, **run_kwargs)
        else:
            if run_kwargs:
                unexpected = ", ".join(sorted(run_kwargs))
                raise TypeError(
                    "counter-progress execution does not support extra run "
                    f"arguments: {unexpected}"
                )
            self._run_rounds_with_counter_progress(soc, counter_progress)
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
        dmem_addresses = {
            int(field["dmem_addr"])
            for field in self._sweep_fields
            if field.get("storage") == "dmem"
        }
        if self._bias_t_max_duration_dmem_addr is not None:
            dmem_addresses.add(int(self._bias_t_max_duration_dmem_addr))
        for instruction in self.prog_list:
            if instruction.get("name") in {"memri", "memwi"}:
                args = instruction.get("args", ())
                if len(args) >= 3:
                    dmem_addresses.add(int(args[2]))
        compensation = self.sequence.bias_t_compensation
        compensation_type = (
            None
            if compensation is None
            else compensation.compensation_type
        )
        pmem_words = len(self.prog_list)
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
            "repetitions_per_sweep_point": self.cfg["reps"],
            "total_acquisitions": (
                self.sequence.sweep_point_count * self.cfg["reps"]
            ),
            "commands_per_point": sum(
                len(commands) for commands in self.compiled_points[0].segment_commands
            ),
            "sweep_execution": "tproc_loop_and_add",
            "sweep_dynamic_register_fields": sum(
                field.get("storage") != "dmem" for field in self._sweep_fields
            ),
            "sweep_dynamic_dmem_fields": sum(
                field.get("storage") == "dmem" for field in self._sweep_fields
            ),
            "sweep_uses_point_table": False,
            "sweep_max_target_quantization_error": self._sweep_max_target_error,
            "sweep_max_step_quantization_error": self._sweep_max_step_error,
            "bias_t_compensation": self.sequence.bias_t_compensation is not None,
            "bias_t_compensation_type": compensation_type,
            "bias_t_compensation_config": self.sequence.bias_t_compensation,
            "bias_t_compensation_mode": self._bias_t_mode,
            "bias_t_duration_execution": (
                None
                if self.sequence.bias_t_compensation is None
                else (
                    "awg_flat_ramp_target_over_tau"
                    if self._bias_t_mode == "filter"
                    else (
                        "tproc_fixed_time_dynamic_voltage"
                        if self._bias_t_mode == "fixed_time"
                        else "tproc_simultaneous_set_and_max_sync"
                    )
                )
            ),
            "bias_t_dynamic_register_fields": 0,
            "bias_t_dynamic_dmem_fields": len(self._bias_t_fields),
            "bias_t_dmem_addresses": tuple(
                int(field["dmem_addr"]) for field in self._bias_t_fields
            ),
            "bias_t_max_duration_q_error": self._bias_t_max_duration_q_error,
            "bias_t_max_target_code_error": self._bias_t_max_target_code_error,
            "bias_t_simultaneous_start": (
                isinstance(
                    self.sequence.bias_t_compensation,
                    BiasTCompensationConfig,
                )
            ),
            "bias_t_simultaneous_start_lead_cycles": (
                self.bias_t_simultaneous_start_lead_cycles
            ),
            "bias_t_max_duration_dmem_address": (
                self._bias_t_max_duration_dmem_addr
            ),
            "startup_lead_tproc_cycles_once": self.command_lead_tproc_cycles,
            "tproc_mhz": self.tproc_mhz,
            "hwh_tproc_mhz": self.hwh_tproc_mhz,
            "tproc_clock_is_manual": self.tproc_clock_is_manual,
            "point_end_tproc_cycles": self.timing["point_end"],
            "program_instructions": len(self.prog_list),
            "tproc_pmem_words_used": pmem_words,
            "tproc_dmem_words_reserved": len(dmem_addresses),
            "tproc_dmem_words_required": (
                max(dmem_addresses) + 1 if dmem_addresses else 0
            ),
            "tproc_dmem_addresses": tuple(sorted(dmem_addresses)),
            "tproc_pmem_capacity": int(self.tproccfg.get("pmem_size", 0)),
            "tproc_dmem_capacity": int(self.tproccfg.get("dmem_size", 0)),
            "tproc_memory_within_4096": (
                pmem_words < 4096
                and all(0 <= address < 4096 for address in dmem_addresses)
            ),
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
    "BIAS_T_COMPENSATION_TYPES",
    "BiasTCompensationConfig",
    "BiasTCompensationPreview",
    "BiasTFilterCompensationConfig",
    "CompiledCommand",
    "CompiledPoint",
    "DdrFirReadoutConfig",
    "DEFAULT_BIAS_T_DURATION_FRAC_BITS",
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
