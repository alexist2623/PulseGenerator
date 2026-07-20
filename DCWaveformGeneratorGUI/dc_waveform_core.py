"""Core waveform model and QCS/QICK code generators.

The GUI stores time in nanoseconds and voltage in millivolts.  Keeping those
units inside this module prevents UI code from mixing SI seconds with display
values.  QICK export maps the configured millivolt full scale to the normalized
[-1.0, 1.0] range used by :mod:`qick_fine_tune_sweep`.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from numbers import Integral, Real
import keyword
import re
from typing import Optional, Sequence, Tuple

import numpy as np


DEFAULT_INITIAL_VOLTAGE_MV = 100.0
DEFAULT_INITIAL_DURATION_NS = 100.0
DEFAULT_INSERT_RAMP_NS = 50.0
DEFAULT_INSERT_FLAT_NS = 100.0
DEFAULT_MIN_DURATION_NS = 1.0e-6
DEFAULT_QCS_FULL_SCALE_V = 5.0
DEFAULT_QICK_FABRIC_MHZ = 300.0
DEFAULT_QICK_TPROC_MHZ = 300.0
DEFAULT_QICK_FULL_SCALE_MV = 2500.0
DEFAULT_DC_MEASURE_GAIN_V_PER_A = 1.0
DEFAULT_BIAS_T_COMPENSATION_FRACTION = 0.1
DEFAULT_BIAS_T_COMPENSATION_DURATION_US = 1.0
DEFAULT_BIAS_T_FILTER_TAU_US = 100.0
BIAS_T_COMPENSATION_MODES = ("fixed_voltage", "fixed_time")
BIAS_T_COMPENSATION_TYPES = ("dc", "filter")
MAX_QICK_OUTPUTS = 8
QICK_OUTPUT_BOARD_TYPES = ("RF_Out", "DC_Out")
QICK_INPUT_BOARD_TYPES = ("RF_In", "DC_In")


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_real(value: Real, name: str) -> float:
    value = _finite_real(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def adc_iq_to_voltage(iq) -> np.ndarray:
    """Convert ADC I/Q values to volts using the temporary identity model."""
    values = np.asarray(iq, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("ADC I/Q values must be finite")
    return values


def dc_iq_to_current(
    iq,
    gain_v_per_a: Real = DEFAULT_DC_MEASURE_GAIN_V_PER_A,
) -> np.ndarray:
    """Convert DC-input I/Q to amperes using ``voltage / gain``.

    ADC-to-voltage conversion is intentionally identity until a calibrated
    transfer function is introduced.
    """
    gain = _positive_real(gain_v_per_a, "DC measurement gain_v_per_a")
    return adc_iq_to_voltage(iq) / gain


def _nonnegative_real(value: Real, name: str) -> float:
    value = _finite_real(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_int(value: Integral, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _bounded_int(value: Integral, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _identifier(value: str, fallback: str) -> str:
    text = re.sub(r"\W+", "_", str(value).strip())
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"_{text}"
    if keyword.iskeyword(text):
        text = f"{text}_channel"
    return text


def _unique_names(names: Sequence[str], fallback_prefix: str) -> Tuple[str, ...]:
    result = []
    used = set()
    for index, raw_name in enumerate(names):
        base = _identifier(raw_name, f"{fallback_prefix}_{index + 1}")
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        result.append(name)
    return tuple(result)


@dataclass(frozen=True)
class WaveformInterval:
    """One interval in a piecewise-linear pulse."""

    index: int
    kind: str
    start_ns: float
    end_ns: float
    start_mv: float
    end_mv: float

    @property
    def duration_ns(self) -> float:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class QickSweepSpec:
    """Optional amplitude sweep applied to one generated QICK SET segment."""

    segment_name: str
    output_name: str
    start: float
    stop: float
    count: int

    def __post_init__(self) -> None:
        if not str(self.segment_name):
            raise ValueError("segment_name must not be empty")
        if not str(self.output_name):
            raise ValueError("output_name must not be empty")
        start = _finite_real(self.start, "sweep start")
        stop = _finite_real(self.stop, "sweep stop")
        if not -1.0 <= start <= 1.0 or not -1.0 <= stop <= 1.0:
            raise ValueError("sweep endpoints must be inside [-1.0, 1.0]")
        _positive_int(self.count, "sweep count")


def _coerce_sweep_specs(
    sweep: Optional[QickSweepSpec],
    sweeps: Optional[Sequence[QickSweepSpec]],
) -> Tuple[QickSweepSpec, ...]:
    """Normalize legacy singular and Cartesian multi-sweep arguments."""
    if sweep is not None and sweeps is not None:
        raise ValueError("use either sweep or sweeps, not both")
    if sweeps is None:
        normalized = () if sweep is None else (sweep,)
    else:
        normalized = tuple(sweeps)
    if any(not isinstance(item, QickSweepSpec) for item in normalized):
        raise TypeError("every sweep entry must be a QickSweepSpec")
    targets = [(item.segment_name, item.output_name) for item in normalized]
    if len(set(targets)) != len(targets):
        raise ValueError("each (segment, output) sweep target must be unique")
    return normalized


def _coerce_cross_capacitance(matrix, output_count: int) -> Tuple[Tuple[float, ...], ...]:
    if matrix is None:
        values = np.eye(output_count, dtype=float)
    else:
        values = np.asarray(matrix, dtype=float)
    expected_shape = (output_count, output_count)
    if values.shape != expected_shape:
        raise ValueError(
            f"cross-capacitance matrix shape must be {expected_shape}, "
            f"received {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("cross-capacitance coefficients must be finite")
    if not np.allclose(np.diag(values), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("cross-capacitance diagonal entries must equal 1")
    return tuple(tuple(float(value) for value in row) for row in values)


@dataclass(frozen=True)
class QickRfPulseSpec:
    """RF pulse and RF-board output attenuation/filter settings.

    ``delay_us`` is relative to the start of ``segment_name``.  Pulse duration
    is converted to the selected RF generator's fabric clock when the generated
    module receives ``soccfg``.
    """

    gen_ch: int
    segment_name: str
    delay_us: float
    duration_us: float
    frequency_mhz: float
    gain: int
    att1_db: float
    att2_db: float
    phase_degrees: float = 0.0
    nqz: int = 1
    require_within_segment: bool = True
    filter_type: str = "bypass"
    filter_cutoff: float = 2.5
    filter_bandwidth: float = 1.0
    output_board_type: str = "RF_Out"

    def __post_init__(self) -> None:
        _bounded_int(self.gen_ch, "RF generator channel", 0, 1_000_000)
        if not str(self.segment_name):
            raise ValueError("RF segment_name must not be empty")
        _nonnegative_real(self.delay_us, "RF delay_us")
        _positive_real(self.duration_us, "RF duration_us")
        _finite_real(self.frequency_mhz, "RF frequency_mhz")
        _bounded_int(self.gain, "RF gain", -32768, 32767)
        for name, value in (("RF att1_db", self.att1_db), ("RF att2_db", self.att2_db)):
            value = _finite_real(value, name)
            if value < 0.0 or value > 31.75:
                raise ValueError(f"{name} must be in [0, 31.75] dB")
        _finite_real(self.phase_degrees, "RF phase_degrees")
        _bounded_int(self.nqz, "RF nqz", 1, 3)
        if not isinstance(self.require_within_segment, bool):
            raise TypeError("require_within_segment must be bool")
        if self.filter_type not in {"bypass", "lowpass", "highpass", "bandpass"}:
            raise ValueError(
                "RF output filter_type must be bypass, lowpass, highpass, or bandpass"
            )
        _nonnegative_real(self.filter_cutoff, "RF output filter_cutoff")
        _positive_real(self.filter_bandwidth, "RF output filter_bandwidth")
        if self.output_board_type not in QICK_OUTPUT_BOARD_TYPES:
            raise ValueError(
                f"output_board_type must be one of {QICK_OUTPUT_BOARD_TYPES}"
            )

    @property
    def effective_att1_db(self) -> float:
        return float(self.att1_db) if self.output_board_type == "RF_Out" else 0.0

    @property
    def effective_att2_db(self) -> float:
        return float(self.att2_db) if self.output_board_type == "RF_Out" else 0.0


@dataclass(frozen=True)
class QickDdrReadoutSpec:
    """FIR-decimated 1 MSPS DRAM capture settings."""

    ro_ch: int
    segment_name: str
    delay_us: float
    samples_per_trigger: int
    readout_frequency_mhz: float = 0.0
    margin_input_samples: int = 1024
    address: int = 0
    force_overwrite: bool = False
    attenuation_db: float = 20.0
    filter_type: str = "bypass"
    filter_cutoff: float = 2.5
    filter_bandwidth: float = 1.0
    input_board_type: str = "RF_In"
    dc_gain_db: float = 0.0
    dc_measure_mode: bool = False
    dc_measure_gain_v_per_a: float = DEFAULT_DC_MEASURE_GAIN_V_PER_A

    def __post_init__(self) -> None:
        _bounded_int(self.ro_ch, "DDR readout channel", 0, 1_000_000)
        if not str(self.segment_name):
            raise ValueError("DDR segment_name must not be empty")
        _nonnegative_real(self.delay_us, "DDR delay_us")
        _positive_int(self.samples_per_trigger, "DDR samples_per_trigger")
        _finite_real(self.readout_frequency_mhz, "DDR readout_frequency_mhz")
        _bounded_int(self.margin_input_samples, "DDR margin_input_samples", 0, 1 << 30)
        _bounded_int(self.address, "DDR address", 0, (1 << 63) - 1)
        if not isinstance(self.force_overwrite, bool):
            raise TypeError("force_overwrite must be bool")
        attenuation = _finite_real(self.attenuation_db, "RF input attenuation_db")
        if attenuation < 0.0 or attenuation > 31.75:
            raise ValueError("RF input attenuation_db must be in [0, 31.75] dB")
        if self.filter_type not in {"bypass", "lowpass", "highpass", "bandpass"}:
            raise ValueError(
                "RF input filter_type must be bypass, lowpass, highpass, or bandpass"
            )
        _nonnegative_real(self.filter_cutoff, "RF input filter_cutoff")
        _positive_real(self.filter_bandwidth, "RF input filter_bandwidth")
        if self.input_board_type not in QICK_INPUT_BOARD_TYPES:
            raise ValueError(
                f"input_board_type must be one of {QICK_INPUT_BOARD_TYPES}"
            )
        dc_gain = _finite_real(self.dc_gain_db, "DC input dc_gain_db")
        if dc_gain < -6.0 or dc_gain > 26.0:
            raise ValueError("DC input dc_gain_db must be in [-6, 26] dB")
        if not isinstance(self.dc_measure_mode, bool):
            raise TypeError("DC measure mode must be bool")
        _positive_real(
            self.dc_measure_gain_v_per_a,
            "DC measurement gain_v_per_a",
        )
        if self.dc_measure_mode and self.input_board_type != "DC_In":
            raise ValueError("DC measure mode requires the DC_In input board")

    @property
    def effective_attenuation_db(self) -> float:
        return float(self.attenuation_db) if self.input_board_type == "RF_In" else 0.0

    @property
    def effective_dc_gain_db(self) -> float:
        return float(self.dc_gain_db) if self.input_board_type == "DC_In" else 0.0

    @property
    def measurement_unit(self) -> str:
        return "A" if self.dc_measure_mode else "ADC units"


@dataclass(frozen=True)
class QickSegmentSpec:
    """Dependency-free representation of a FineTuneSequence segment."""

    name: str
    kind: str
    amplitudes: Tuple[Optional[float], ...]
    duration_cycles: int


class PulseSequence:
    """Piecewise-linear voltage waveform using ns and mV internally.

    The point layout is ``SET, RAMP, SET, ...``.  Consequently, even-numbered
    intervals are flat SET intervals and odd-numbered intervals are ramps to
    the next SET level.  This is the same layout expected by FineTuneSequence.
    """

    def __init__(
        self,
        initial_voltage: Real = 0.0,
        initial_duration_ns: Real = DEFAULT_INITIAL_DURATION_NS,
    ) -> None:
        initial_voltage = _finite_real(initial_voltage, "initial_voltage")
        initial_duration_ns = _positive_real(initial_duration_ns, "initial_duration_ns")
        self.t = np.asarray([0.0, initial_duration_ns], dtype=float)
        self.v = np.asarray([initial_voltage, initial_voltage], dtype=float)
        self.v_bounds = (-2500.0, 2500.0)

    @property
    def duration_ns(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def set_count(self) -> int:
        return (len(self.t) + 1) // 2

    def copy(self) -> "PulseSequence":
        duplicate = PulseSequence(float(self.v[0]), float(self.t[1] - self.t[0]))
        duplicate.t = self.t.copy()
        duplicate.v = self.v.copy()
        duplicate.v_bounds = tuple(self.v_bounds)
        return duplicate

    def validate(self) -> None:
        if self.t.ndim != 1 or self.v.ndim != 1 or len(self.t) != len(self.v):
            raise ValueError("time and voltage arrays must be one-dimensional and equal length")
        if len(self.t) < 2 or len(self.t) % 2:
            raise ValueError("waveform must contain SET/RAMP/SET point pairs")
        if not np.all(np.isfinite(self.t)) or not np.all(np.isfinite(self.v)):
            raise ValueError("waveform points must be finite")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("waveform times must be strictly increasing")
        for index in range(0, len(self.t) - 1, 2):
            if not np.isclose(self.v[index], self.v[index + 1], rtol=0.0, atol=1.0e-12):
                raise ValueError(f"SET interval {index // 2} is not flat")

    def _clip_voltage(self, value: Real) -> float:
        value = _finite_real(value, "voltage")
        return float(np.clip(value, self.v_bounds[0], self.v_bounds[1]))

    def edit_ramp(self, flat_idx: int, new_ramp: Real) -> bool:
        """Change the incoming ramp duration for one SET interval."""
        if flat_idx <= 0 or flat_idx >= len(self.t) - 1 or flat_idx % 2:
            return False
        try:
            new_ramp = _positive_real(new_ramp, "ramp duration")
        except (TypeError, ValueError):
            return False
        old_ramp = self.t[flat_idx] - self.t[flat_idx - 1]
        self.t[flat_idx:] += new_ramp - old_ramp
        return True

    def edit_flat(self, flat_idx: int, new_flat: Real) -> bool:
        """Change one SET hold duration and shift all following intervals."""
        if flat_idx < 0 or flat_idx >= len(self.t) - 1 or flat_idx % 2:
            return False
        try:
            new_flat = _positive_real(new_flat, "flat duration")
        except (TypeError, ValueError):
            return False
        old_flat = self.t[flat_idx + 1] - self.t[flat_idx]
        self.t[flat_idx + 1:] += new_flat - old_flat
        return True

    def edit_voltage(self, flat_idx: int, new_v: Real) -> bool:
        """Change one SET level, preserving adjacent automatic ramps."""
        if flat_idx < 0 or flat_idx >= len(self.v) - 1 or flat_idx % 2:
            return False
        self.v[flat_idx:flat_idx + 2] = self._clip_voltage(new_v)
        return True

    def add_flat_ramp(self, ramp: Real, flat: Real, target_v: Real) -> None:
        """Append a ramp followed by a flat SET interval."""
        ramp = _positive_real(ramp, "ramp duration")
        flat = _positive_real(flat, "flat duration")
        target_v = self._clip_voltage(target_v)
        t0 = float(self.t[-1])
        self.t = np.append(self.t, [t0 + ramp, t0 + ramp + flat])
        self.v = np.append(self.v, [target_v, target_v])

    def flat_segments(self) -> Tuple[Tuple[int, int], ...]:
        return tuple((index, index + 1) for index in range(0, len(self.t) - 1, 2))

    def intervals(self) -> Tuple[WaveformInterval, ...]:
        self.validate()
        return tuple(
            WaveformInterval(
                index=index,
                kind="set" if index % 2 == 0 else "ramp",
                start_ns=float(self.t[index]),
                end_ns=float(self.t[index + 1]),
                start_mv=float(self.v[index]),
                end_mv=float(self.v[index + 1]),
            )
            for index in range(len(self.t) - 1)
        )

    def update_flat(self, rng: Tuple[int, int], new_v: Real) -> None:
        i0, _ = rng
        if not self.edit_voltage(i0, new_v):
            raise ValueError("invalid SET interval")

    def update_point(self, rng: Tuple[int, int], new_t: Real) -> None:
        """Move one boundary while retaining all following interval lengths."""
        i0, _ = rng
        if i0 == 0:
            return
        new_t = _finite_real(new_t, "time")
        new_t = max(new_t, float(self.t[i0 - 1]) + DEFAULT_MIN_DURATION_NS)
        delta = new_t - float(self.t[i0])
        self.t[i0:] += delta

    def delete_flat_ramp(self, flat_idx: int) -> bool:
        """Delete one noninitial SET and its incoming ramp."""
        if flat_idx <= 0 or flat_idx + 1 >= len(self.t) or flat_idx % 2:
            return False
        shrink = self.t[flat_idx + 1] - self.t[flat_idx - 1]
        self.t = np.delete(self.t, [flat_idx, flat_idx + 1])
        self.v = np.delete(self.v, [flat_idx, flat_idx + 1])
        if flat_idx < len(self.t):
            self.t[flat_idx:] -= shrink
        return True

    def insert_flat_ramp(
        self,
        flat_idx: int,
        ramp_ns: Real = DEFAULT_INSERT_RAMP_NS,
        flat_ns: Real = DEFAULT_INSERT_FLAT_NS,
    ) -> bool:
        """Insert an editable SET and incoming ramp before ``flat_idx``."""
        if flat_idx < 2 or flat_idx > len(self.t) or flat_idx % 2:
            return False
        ramp_ns = _positive_real(ramp_ns, "ramp_ns")
        flat_ns = _positive_real(flat_ns, "flat_ns")
        target_v = float(self.v[flat_idx - 1])
        t0 = float(self.t[flat_idx - 1])
        self.t[flat_idx:] += ramp_ns + flat_ns
        self.t = np.insert(self.t, flat_idx, [t0 + ramp_ns, t0 + ramp_ns + flat_ns])
        self.v = np.insert(self.v, flat_idx, [target_v, target_v])
        return True

    def to_dict(self) -> dict:
        return {
            "time_ns": self.t.tolist(),
            "voltage_mv": self.v.tolist(),
            "voltage_bounds_mv": list(self.v_bounds),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PulseSequence":
        pulse = cls()
        pulse.t = np.asarray(data["time_ns"], dtype=float)
        pulse.v = np.asarray(data["voltage_mv"], dtype=float)
        if "voltage_bounds_mv" in data:
            bounds = tuple(float(item) for item in data["voltage_bounds_mv"])
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError("voltage_bounds_mv must contain increasing min/max values")
            pulse.v_bounds = bounds
        pulse.validate()
        return pulse

    def to_qcs_dcwaveform(self, ch_name: str) -> str:
        """Return QCS statements for this channel (legacy-compatible API)."""
        return _qcs_channel_code(self, _identifier(ch_name, "dc_ch"))


def _normalize_mv(value_mv: Real, full_scale_mv: Real) -> float:
    value_mv = _finite_real(value_mv, "voltage_mv")
    full_scale_mv = _positive_real(full_scale_mv, "full_scale_mv")
    normalized = value_mv / full_scale_mv
    if normalized < -1.0 - 1.0e-12 or normalized > 1.0 + 1.0e-12:
        raise ValueError(
            f"voltage {value_mv:g} mV exceeds configured QICK full scale "
            f"+/-{full_scale_mv:g} mV"
        )
    return float(np.clip(normalized, -1.0, 1.0))


def _cycles_from_ns(duration_ns: Real, fabric_mhz: Real) -> int:
    duration_ns = _positive_real(duration_ns, "duration_ns")
    fabric_mhz = _positive_real(fabric_mhz, "fabric_mhz")
    return max(1, int(ceil(duration_ns * fabric_mhz / 1000.0)))


def _validate_aligned_pulses(pulses: Sequence[PulseSequence]) -> None:
    if not pulses:
        raise ValueError("at least one pulse is required")
    if len(pulses) > MAX_QICK_OUTPUTS:
        raise ValueError(f"QICK supports at most {MAX_QICK_OUTPUTS} AWG tuning outputs")
    reference = pulses[0].t
    for index, pulse in enumerate(pulses):
        pulse.validate()
        if pulse.t.shape != reference.shape or not np.allclose(
            pulse.t, reference, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(
                "QICK export requires all ports to use the same SET/RAMP timing; "
                f"port {index + 1} does not match port 1"
            )


def transform_virtual_waveforms(
    pulses: Sequence[PulseSequence],
    cross_capacitance=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned virtual and physical waveforms in millivolts.

    The configured GUI traces are virtual-gate voltages.  The values sent to
    the AWG outputs are physical voltages computed point-by-point as
    ``physical = cross_capacitance @ virtual``.  Rows identify physical AWG
    outputs and columns identify virtual gates.  A union time grid preserves
    transitions from ports whose editable timing grids are not yet aligned.
    """
    pulses = tuple(pulses)
    if not pulses:
        raise ValueError("at least one pulse is required")
    for pulse in pulses:
        pulse.validate()
    matrix = np.asarray(
        _coerce_cross_capacitance(cross_capacitance, len(pulses)),
        dtype=float,
    )
    time_ns = np.unique(np.concatenate([pulse.t for pulse in pulses])).astype(
        float, copy=False
    )
    virtual_mv = np.vstack(
        [np.interp(time_ns, pulse.t, pulse.v) for pulse in pulses]
    )
    physical_mv = matrix @ virtual_mv
    return time_ns, virtual_mv, physical_mv


def qick_set_segment_names(pulse: PulseSequence) -> Tuple[str, ...]:
    pulse.validate()
    return tuple(f"set_{index}" for index in range(pulse.set_count))


def make_qick_segment_specs(
    pulses: Sequence[PulseSequence],
    *,
    fabric_mhz: Real = DEFAULT_QICK_FABRIC_MHZ,
    full_scale_mv: Real = DEFAULT_QICK_FULL_SCALE_MV,
) -> Tuple[QickSegmentSpec, ...]:
    """Convert aligned GUI pulses into dependency-free QICK segment specs."""
    pulses = tuple(pulses)
    _validate_aligned_pulses(pulses)
    fabric_mhz = _positive_real(fabric_mhz, "fabric_mhz")
    full_scale_mv = _positive_real(full_scale_mv, "full_scale_mv")
    specs = []
    for interval_index in range(len(pulses[0].t) - 1):
        duration_ns = pulses[0].t[interval_index + 1] - pulses[0].t[interval_index]
        duration_cycles = _cycles_from_ns(duration_ns, fabric_mhz)
        if interval_index % 2 == 0:
            set_index = interval_index // 2
            amplitudes = tuple(
                _normalize_mv(pulse.v[interval_index], full_scale_mv) for pulse in pulses
            )
            specs.append(QickSegmentSpec(f"set_{set_index}", "set", amplitudes, duration_cycles))
        else:
            ramp_index = interval_index // 2
            specs.append(
                QickSegmentSpec(
                    f"ramp_{ramp_index}_to_{ramp_index + 1}",
                    "ramp",
                    (None,) * len(pulses),
                    duration_cycles,
                )
            )
    return tuple(specs)


def build_qick_sequence(
    pulses: Sequence[PulseSequence],
    *,
    output_names: Optional[Sequence[str]] = None,
    fabric_mhz: Real = DEFAULT_QICK_FABRIC_MHZ,
    full_scale_mv: Real = DEFAULT_QICK_FULL_SCALE_MV,
    sweep: Optional[QickSweepSpec] = None,
    sweeps: Optional[Sequence[QickSweepSpec]] = None,
    cross_capacitance=None,
    bias_t_compensation_enabled: bool = False,
    bias_t_compensation_type: str = "dc",
    bias_t_compensation_voltage_mv: Optional[Real] = None,
    bias_t_compensation_mode: str = "fixed_voltage",
    bias_t_compensation_duration_us: Real = DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
    bias_t_filter_tau_us: Real = DEFAULT_BIAS_T_FILTER_TAU_US,
):
    """Build a real qick_fine_tune_sweep.FineTuneSequence instance."""
    pulses = tuple(pulses)
    if output_names is None:
        output_names = tuple(f"awg_{index}" for index in range(len(pulses)))
    output_names = _unique_names(tuple(output_names), "awg")
    if len(output_names) != len(pulses):
        raise ValueError("output_names length must match the pulse count")
    full_scale_mv = _positive_real(full_scale_mv, "full_scale_mv")
    specs = make_qick_segment_specs(
        pulses, fabric_mhz=fabric_mhz, full_scale_mv=full_scale_mv
    )

    from qick_fine_tune_sweep import FineTuneSequence

    sequence = FineTuneSequence(output_names)
    sequence.set_cross_capacitance(
        _coerce_cross_capacitance(cross_capacitance, len(output_names))
    )
    if not isinstance(bias_t_compensation_enabled, (bool, np.bool_)):
        raise TypeError("bias_t_compensation_enabled must be boolean")
    if bias_t_compensation_type not in BIAS_T_COMPENSATION_TYPES:
        raise ValueError(
            "bias_t_compensation_type must be one of "
            f"{BIAS_T_COMPENSATION_TYPES}"
        )
    if bias_t_compensation_voltage_mv is None:
        bias_t_compensation_voltage_mv = (
            full_scale_mv * DEFAULT_BIAS_T_COMPENSATION_FRACTION
        )
    compensation_mv = _positive_real(
        bias_t_compensation_voltage_mv,
        "bias_t_compensation_voltage_mv",
    )
    if compensation_mv > full_scale_mv:
        raise ValueError("Bias-T compensation voltage exceeds QICK full scale")
    if bias_t_compensation_mode not in BIAS_T_COMPENSATION_MODES:
        raise ValueError(
            f"bias_t_compensation_mode must be one of {BIAS_T_COMPENSATION_MODES}"
        )
    compensation_duration_us = _positive_real(
        bias_t_compensation_duration_us,
        "bias_t_compensation_duration_us",
    )
    compensation_duration_cycles = _cycles_from_ns(
        compensation_duration_us * 1000.0,
        fabric_mhz,
    )
    filter_tau_us = _positive_real(
        bias_t_filter_tau_us,
        "bias_t_filter_tau_us",
    )
    if bias_t_compensation_type == "filter":
        sequence.set_bias_t_filter_compensation(
            filter_tau_us * fabric_mhz,
            enabled=bool(bias_t_compensation_enabled),
        )
    else:
        sequence.set_bias_t_compensation(
            compensation_mv / full_scale_mv,
            enabled=bool(bias_t_compensation_enabled),
            mode=bias_t_compensation_mode,
            fixed_duration_cycles=(
                compensation_duration_cycles
                if bias_t_compensation_mode == "fixed_time"
                else None
            ),
        )
    for spec in specs:
        if spec.kind == "set":
            sequence.add_set(spec.name, spec.amplitudes, spec.duration_cycles)
        else:
            sequence.add_ramp(spec.name, spec.duration_cycles)
    for sweep_spec in _coerce_sweep_specs(sweep, sweeps):
        sequence.add_amplitude_sweep(
            segment=sweep_spec.segment_name,
            output=sweep_spec.output_name,
            start=sweep_spec.start,
            stop=sweep_spec.stop,
            count=sweep_spec.count,
        )
    sequence._validate()
    return sequence


def _qcs_channel_code(pulse: PulseSequence, channel_name: str) -> str:
    pulse.validate()
    return _qcs_channel_samples_code(pulse.t, pulse.v, channel_name)


def _qcs_channel_samples_code(time_ns, voltage_mv, channel_name: str) -> str:
    """Generate QCS segments from an arbitrary piecewise-linear trace."""
    time_values = np.asarray(time_ns, dtype=float)
    voltage_values = np.asarray(voltage_mv, dtype=float)
    if (
        time_values.ndim != 1
        or voltage_values.shape != time_values.shape
        or time_values.size < 2
    ):
        raise ValueError("QCS time and voltage traces must be equal-length vectors")
    if not np.all(np.isfinite(time_values)) or not np.all(np.isfinite(voltage_values)):
        raise ValueError("QCS waveform points must be finite")
    if np.any(np.diff(time_values) <= 0.0):
        raise ValueError("QCS waveform times must be strictly increasing")

    lines = []
    for index in range(time_values.size - 1):
        duration_name = f"{channel_name}_dc_segment_duration_{index}"
        duration_ns = float(time_values[index + 1] - time_values[index])
        lines.extend(
            [
                f"    {duration_name} = qcs.Scalar(",
                f"        name={duration_name!r},",
                f"        value={duration_ns:.12g} * ns,",
                "        dtype=float,",
                "    )",
            ]
        )
    for index in range(time_values.size - 1):
        duration_name = f"{channel_name}_dc_segment_duration_{index}"
        waveform_name = f"{channel_name}_dc_segment_{index}"
        start_mv = float(voltage_values[index])
        end_mv = float(voltage_values[index + 1])
        if np.isclose(start_mv, end_mv, rtol=0.0, atol=1.0e-12):
            lines.extend(
                [
                    f"    {waveform_name} = qcs.DCWaveform(",
                    f"        duration={duration_name},",
                    "        envelope=qcs.ConstantEnvelope(),",
                    f"        amplitude={start_mv:.12g} * mV,",
                    "    )",
                    "",
                ]
            )
        else:
            peak_mv = max(abs(start_mv), abs(end_mv))
            if peak_mv == 0:
                start_shape = end_shape = 0.0
            else:
                start_shape = start_mv / peak_mv
                end_shape = end_mv / peak_mv
            lines.extend(
                [
                    f"    {waveform_name} = qcs.DCWaveform(",
                    f"        duration={duration_name},",
                    "        envelope=qcs.ArbitraryEnvelope(",
                    f"            [0.0, 1.0], [{start_shape:.12g}, {end_shape:.12g}],",
                    "        ),",
                    f"        amplitude={peak_mv:.12g} * mV,",
                    "    )",
                    "",
                ]
            )
    for index in range(time_values.size - 1):
        lines.append(
            f"    program.add_waveform({channel_name}_dc_segment_{index}, {channel_name})"
        )
    lines.append("")
    return "\n".join(lines)


def generate_qcs_program_code(
    pulses: Sequence[PulseSequence],
    *,
    channel_names: Optional[Sequence[str]] = None,
    full_scale_v: Real = DEFAULT_QCS_FULL_SCALE_V,
    cross_capacitance=None,
) -> str:
    """Generate Keysight QCS code for physical, cross-compensated outputs."""
    pulses = tuple(pulses)
    if not pulses:
        raise ValueError("at least one pulse is required")
    full_scale_v = _positive_real(full_scale_v, "full_scale_v")
    if channel_names is None:
        channel_names = tuple(f"dc_ch_{index + 1}" for index in range(len(pulses)))
    channel_names = _unique_names(tuple(channel_names), "dc_ch")
    if len(channel_names) != len(pulses):
        raise ValueError("channel_names length must match the pulse count")
    matrix = _coerce_cross_capacitance(cross_capacitance, len(pulses))
    if np.allclose(matrix, np.eye(len(pulses)), rtol=0.0, atol=1.0e-12):
        traces = tuple((pulse.t, pulse.v) for pulse in pulses)
    else:
        time_ns, _virtual_mv, physical_mv = transform_virtual_waveforms(
            pulses,
            matrix,
        )
        traces = tuple((time_ns, physical_mv[index]) for index in range(len(pulses)))

    lines = [
        '"""Generated Keysight QCS DC waveforms."""',
        "",
        "import keysight.qcs as qcs",
        "",
        "ns = 1e-9",
        f"QCS_FULL_SCALE_V = {full_scale_v:.12g}",
        f"CROSS_CAPACITANCE = {matrix!r}",
        "mV = 1.0 / (QCS_FULL_SCALE_V * 1000.0)",
        "",
        "",
        "def generate_dc_waveforms(",
        "    program: qcs.Program,",
    ]
    for name in channel_names:
        lines.append(f"    {name}: qcs.Channels,")
    lines.extend([") -> qcs.Program:", "    # RAMP envelopes are unit-normalized; amplitude carries physical scale."])
    for (time_ns, voltage_mv), name in zip(traces, channel_names):
        lines.append(
            _qcs_channel_samples_code(time_ns, voltage_mv, name).rstrip()
        )
    lines.extend(["    return program", ""])
    return "\n".join(lines)


def generate_qick_program_code(
    pulses: Sequence[PulseSequence],
    *,
    output_names: Optional[Sequence[str]] = None,
    awg_channels: Optional[Sequence[int]] = None,
    fabric_mhz: Real = DEFAULT_QICK_FABRIC_MHZ,
    tproc_mhz: Real = DEFAULT_QICK_TPROC_MHZ,
    full_scale_mv: Real = DEFAULT_QICK_FULL_SCALE_MV,
    repetitions_per_sweep: Integral = 1,
    sweep: Optional[QickSweepSpec] = None,
    sweeps: Optional[Sequence[QickSweepSpec]] = None,
    cross_capacitance=None,
    rf_pulse_spec: Optional[QickRfPulseSpec] = None,
    rf_pulse_specs: Optional[Sequence[QickRfPulseSpec]] = None,
    ddr_readout_spec: Optional[QickDdrReadoutSpec] = None,
    bias_t_compensation_enabled: bool = False,
    bias_t_compensation_type: str = "dc",
    bias_t_compensation_voltage_mv: Optional[Real] = None,
    bias_t_compensation_mode: str = "fixed_voltage",
    bias_t_compensation_duration_us: Real = DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
    bias_t_filter_tau_us: Real = DEFAULT_BIAS_T_FILTER_TAU_US,
) -> str:
    """Generate a QICK builder/execution module.

    The generated ``run_experiment()`` helper optionally configures RF-board
    attenuators, executes the pulse program, and returns FIR-decimated 1 MSPS
    DDR IQ data grouped by sweep point and repetition.
    """
    pulses = tuple(pulses)
    if output_names is None:
        output_names = tuple(f"awg_{index}" for index in range(len(pulses)))
    output_names = _unique_names(tuple(output_names), "awg")
    if len(output_names) != len(pulses):
        raise ValueError("output_names length must match the pulse count")
    if awg_channels is None:
        awg_channels = tuple(range(len(pulses)))
    awg_channels = tuple(int(channel) for channel in awg_channels)
    if len(awg_channels) != len(pulses):
        raise ValueError("awg_channels length must match the pulse count")
    if len(set(awg_channels)) != len(awg_channels) or any(channel < 0 for channel in awg_channels):
        raise ValueError("AWG channels must be unique nonnegative integers")
    tproc_mhz = _positive_real(tproc_mhz, "tproc_mhz")
    full_scale_mv = _positive_real(full_scale_mv, "full_scale_mv")
    if not isinstance(bias_t_compensation_enabled, (bool, np.bool_)):
        raise TypeError("bias_t_compensation_enabled must be boolean")
    if bias_t_compensation_type not in BIAS_T_COMPENSATION_TYPES:
        raise ValueError(
            "bias_t_compensation_type must be one of "
            f"{BIAS_T_COMPENSATION_TYPES}"
        )
    if bias_t_compensation_voltage_mv is None:
        bias_t_compensation_voltage_mv = (
            full_scale_mv * DEFAULT_BIAS_T_COMPENSATION_FRACTION
        )
    bias_t_compensation_voltage_mv = _positive_real(
        bias_t_compensation_voltage_mv,
        "bias_t_compensation_voltage_mv",
    )
    if bias_t_compensation_voltage_mv > full_scale_mv:
        raise ValueError("Bias-T compensation voltage exceeds QICK full scale")
    if bias_t_compensation_mode not in BIAS_T_COMPENSATION_MODES:
        raise ValueError(
            f"bias_t_compensation_mode must be one of {BIAS_T_COMPENSATION_MODES}"
        )
    bias_t_compensation_duration_us = _positive_real(
        bias_t_compensation_duration_us,
        "bias_t_compensation_duration_us",
    )
    bias_t_compensation_duration_cycles = _cycles_from_ns(
        bias_t_compensation_duration_us * 1000.0,
        fabric_mhz,
    )
    bias_t_filter_tau_us = _positive_real(
        bias_t_filter_tau_us,
        "bias_t_filter_tau_us",
    )
    repetitions_per_sweep = _positive_int(repetitions_per_sweep, "repetitions_per_sweep")
    sweep_specs = _coerce_sweep_specs(sweep, sweeps)
    if rf_pulse_spec is not None and rf_pulse_specs is not None:
        if tuple(rf_pulse_specs) != (rf_pulse_spec,):
            raise ValueError("use either rf_pulse_spec or rf_pulse_specs, not both")
        rf_pulse_spec = None
    normalized_rf_specs = tuple(
        rf_pulse_specs
        if rf_pulse_specs is not None
        else (() if rf_pulse_spec is None else (rf_pulse_spec,))
    )
    if any(not isinstance(spec, QickRfPulseSpec) for spec in normalized_rf_specs):
        raise TypeError("every RF pulse entry must be a QickRfPulseSpec")
    rf_channels = tuple(spec.gen_ch for spec in normalized_rf_specs)
    if len(set(rf_channels)) != len(rf_channels):
        raise ValueError("each RF pulse must use a unique generator channel")
    cross_capacitance = _coerce_cross_capacitance(
        cross_capacitance, len(output_names)
    )
    specs = make_qick_segment_specs(
        pulses, fabric_mhz=fabric_mhz, full_scale_mv=full_scale_mv
    )
    valid_set_names = {spec.name for spec in specs if spec.kind == "set"}
    for sweep_spec in sweep_specs:
        if sweep_spec.segment_name not in valid_set_names:
            raise ValueError(f"unknown QICK SET segment {sweep_spec.segment_name!r}")
        if sweep_spec.output_name not in output_names:
            raise ValueError(f"unknown QICK output {sweep_spec.output_name!r}")
    for rf_spec in normalized_rf_specs:
        if rf_spec.segment_name not in valid_set_names:
            raise ValueError(f"unknown RF SET segment {rf_spec.segment_name!r}")
        if rf_spec.gen_ch in awg_channels:
            raise ValueError("RF generator channels must be separate from AWG tuning channels")
    if ddr_readout_spec is not None:
        if ddr_readout_spec.segment_name not in valid_set_names:
            raise ValueError(f"unknown DDR SET segment {ddr_readout_spec.segment_name!r}")

    channel_map = dict(zip(output_names, awg_channels))
    sweep_config = tuple(
        (
            item.segment_name,
            item.output_name,
            float(item.start),
            float(item.stop),
            int(item.count),
        )
        for item in sweep_specs
    )
    rf_configs = tuple(
        {
            "gen_ch": int(spec.gen_ch),
            "segment_name": str(spec.segment_name),
            "delay_us": float(spec.delay_us),
            "duration_us": float(spec.duration_us),
            "frequency_mhz": float(spec.frequency_mhz),
            "gain": int(spec.gain),
            "att1_db": float(spec.att1_db),
            "att2_db": float(spec.att2_db),
            "phase_degrees": float(spec.phase_degrees),
            "nqz": int(spec.nqz),
            "require_within_segment": bool(spec.require_within_segment),
            "filter_type": str(spec.filter_type),
            "filter_cutoff": float(spec.filter_cutoff),
            "filter_bandwidth": float(spec.filter_bandwidth),
            "output_board_type": str(spec.output_board_type),
        }
        for spec in normalized_rf_specs
    )
    ddr_config = None if ddr_readout_spec is None else {
        "ro_ch": int(ddr_readout_spec.ro_ch),
        "segment_name": str(ddr_readout_spec.segment_name),
        "delay_us": float(ddr_readout_spec.delay_us),
        "samples_per_trigger": int(ddr_readout_spec.samples_per_trigger),
        "readout_frequency_mhz": float(ddr_readout_spec.readout_frequency_mhz),
        "margin_input_samples": int(ddr_readout_spec.margin_input_samples),
        "address": int(ddr_readout_spec.address),
        "force_overwrite": bool(ddr_readout_spec.force_overwrite),
        "attenuation_db": float(ddr_readout_spec.attenuation_db),
        "filter_type": str(ddr_readout_spec.filter_type),
        "filter_cutoff": float(ddr_readout_spec.filter_cutoff),
        "filter_bandwidth": float(ddr_readout_spec.filter_bandwidth),
        "input_board_type": str(ddr_readout_spec.input_board_type),
        "dc_gain_db": float(ddr_readout_spec.dc_gain_db),
        "dc_measure_mode": bool(ddr_readout_spec.dc_measure_mode),
        "dc_measure_gain_v_per_a": float(
            ddr_readout_spec.dc_measure_gain_v_per_a
        ),
    }
    lines = [
        '"""Generated QICK AWG-tuning, RF pulse, and 1 MSPS DDR program."""',
        "",
        "from qick_fine_tune_sweep import (",
        "    DdrFirReadoutConfig,",
        "    FineTuneSequence,",
        "    RfPulseConfig,",
        "    cycles_from_us,",
        ")",
        "",
        f"OUTPUT_NAMES = {output_names!r}",
        f"AWG_CHANNELS = {channel_map!r}",
        f"FABRIC_MHZ = {float(fabric_mhz)!r}",
        f"TPROC_MHZ = {tproc_mhz!r}",
        f"FULL_SCALE_MV = {float(full_scale_mv)!r}",
        f"BIAS_T_COMPENSATION_ENABLED = {bool(bias_t_compensation_enabled)!r}",
        f"BIAS_T_COMPENSATION_TYPE = {bias_t_compensation_type!r}",
        f"BIAS_T_COMPENSATION_VOLTAGE_MV = {float(bias_t_compensation_voltage_mv)!r}",
        f"BIAS_T_COMPENSATION_MODE = {bias_t_compensation_mode!r}",
        f"BIAS_T_COMPENSATION_DURATION_US = {float(bias_t_compensation_duration_us)!r}",
        f"BIAS_T_COMPENSATION_DURATION_CYCLES = {bias_t_compensation_duration_cycles}",
        f"BIAS_T_FILTER_TAU_US = {float(bias_t_filter_tau_us)!r}",
        f"BIAS_T_FILTER_TAU_CYCLES = {float(bias_t_filter_tau_us * fabric_mhz)!r}",
        f"REPETITIONS_PER_SWEEP = {repetitions_per_sweep}",
        f"SWEEP_SPECS = {sweep_config!r}",
        f"CROSS_CAPACITANCE = {cross_capacitance!r}",
        f"RF_CONFIGS = {rf_configs!r}",
        "RF_CONFIG = RF_CONFIGS[0] if len(RF_CONFIGS) == 1 else None",
        f"DDR_1MSPS_CONFIG = {ddr_config!r}",
        "",
        "",
        "def build_sequence() -> FineTuneSequence:",
        "    sequence = FineTuneSequence(OUTPUT_NAMES)",
        "    sequence.set_cross_capacitance(CROSS_CAPACITANCE)",
        "    if BIAS_T_COMPENSATION_TYPE == 'filter':",
        "        sequence.set_bias_t_filter_compensation(",
        "            BIAS_T_FILTER_TAU_CYCLES,",
        "            enabled=BIAS_T_COMPENSATION_ENABLED,",
        "        )",
        "    else:",
        "        sequence.set_bias_t_compensation(",
        "            BIAS_T_COMPENSATION_VOLTAGE_MV / FULL_SCALE_MV,",
        "            enabled=BIAS_T_COMPENSATION_ENABLED,",
        "            mode=BIAS_T_COMPENSATION_MODE,",
        "            fixed_duration_cycles=(",
        "                BIAS_T_COMPENSATION_DURATION_CYCLES",
        "                if BIAS_T_COMPENSATION_MODE == 'fixed_time'",
        "                else None",
        "            ),",
        "        )",
    ]
    for spec in specs:
        if spec.kind == "set":
            lines.append(
                f"    sequence.add_set({spec.name!r}, {spec.amplitudes!r}, "
                f"duration_cycles={spec.duration_cycles})"
            )
        else:
            lines.append(
                f"    sequence.add_ramp({spec.name!r}, duration_cycles={spec.duration_cycles})"
            )
    for sweep_spec in sweep_specs:
        lines.extend(
            [
                "    sequence.add_amplitude_sweep(",
                f"        segment={sweep_spec.segment_name!r},",
                f"        output={sweep_spec.output_name!r},",
                f"        start={float(sweep_spec.start)!r},",
                f"        stop={float(sweep_spec.stop)!r},",
                f"        count={int(sweep_spec.count)},",
                "    )",
            ]
        )
    lines.extend(
        [
            "    return sequence",
            "",
            "",
            "def _delay_cycles(duration_us, clock_mhz):",
            "    return 0 if duration_us <= 0 else cycles_from_us(duration_us, clock_mhz)",
            "",
            "",
            "def build_rf_pulses(soccfg):",
            "    pulses = []",
            "    for cfg in RF_CONFIGS:",
            "        gen_cfg = soccfg['gens'][cfg['gen_ch']]",
            "        pulses.append(RfPulseConfig(",
            "            gen_ch=cfg['gen_ch'],",
            "            at_segment=cfg['segment_name'],",
            "            length_cycles=cycles_from_us(",
            "                cfg['duration_us'], gen_cfg['f_fabric']",
            "            ),",
            "            gain=cfg['gain'],",
            "            freq_mhz=cfg['frequency_mhz'],",
            "            phase_degrees=cfg['phase_degrees'],",
            "            nqz=cfg['nqz'],",
            "            delay_tproc_cycles=_delay_cycles(",
            "                cfg['delay_us'], TPROC_MHZ",
            "            ),",
            "            require_within_segment=cfg['require_within_segment'],",
            "        ))",
            "    return tuple(pulses)",
            "",
            "",
            "def build_rf_pulse(soccfg):",
            "    pulses = build_rf_pulses(soccfg)",
            "    return pulses[0] if len(pulses) == 1 else None",
            "",
            "",
            "def build_ddr_readout(soccfg):",
            "    if DDR_1MSPS_CONFIG is None:",
            "        return None",
            "    cfg = DDR_1MSPS_CONFIG",
            "    return DdrFirReadoutConfig(",
            "        ro_ch=cfg['ro_ch'],",
            "        samples_per_trigger=cfg['samples_per_trigger'],",
            "        at_segment=cfg['segment_name'],",
            "        readout_freq_mhz=cfg['readout_frequency_mhz'],",
            "        trigger_delay_tproc_cycles=_delay_cycles(",
            "            cfg['delay_us'], TPROC_MHZ",
            "        ),",
            "        margin_input_samples=cfg['margin_input_samples'],",
            "        address=cfg['address'],",
            "        force_overwrite=cfg['force_overwrite'],",
            "    )",
            "",
            "",
            "def build_program(",
            "    soccfg,",
            "    *,",
            "    awg_channels=AWG_CHANNELS,",
            "    repetitions_per_sweep=REPETITIONS_PER_SWEEP,",
            "    readout=None,",
            "    rf_pulse=None,",
            "    rf_pulses=None,",
            "    ddr_readout=None,",
            "    use_generated_aux=True,",
            "):",
            "    if use_generated_aux:",
            "        if rf_pulse is None and rf_pulses is None:",
            "            rf_pulses = build_rf_pulses(soccfg)",
            "        if ddr_readout is None:",
            "            ddr_readout = build_ddr_readout(soccfg)",
            "    return build_sequence().make_program(",
            "        soccfg,",
            "        awg_channels=awg_channels,",
            "        tproc_mhz=TPROC_MHZ,",
            "        repetitions_per_sweep=repetitions_per_sweep,",
            "        readout=readout,",
            "        rf_pulse=rf_pulse,",
            "        rf_pulses=rf_pulses,",
            "        ddr_readout=ddr_readout,",
            "    )",
            "",
            "",
            "def configure_rf_chain(soc):",
            "    if not RF_CONFIGS:",
            "        return None",
            "    actual = []",
            "    for cfg in RF_CONFIGS:",
            "        if cfg['output_board_type'] == 'RF_Out':",
            "            configured = soc.rfb_set_gen_rf(",
            "                cfg['gen_ch'], cfg['att1_db'], cfg['att2_db']",
            "            )",
            "        else:",
            "            soc.rfb_set_gen_dc(cfg['gen_ch'])",
            "            configured = (0.0, 0.0)",
            "        actual.append(configured)",
            "        if cfg['output_board_type'] == 'RF_Out':",
            "            soc.rfb_set_gen_filter(",
            "                cfg['gen_ch'],",
            "                fc=cfg['filter_cutoff'],",
            "                bw=cfg['filter_bandwidth'],",
            "                ftype=cfg['filter_type'],",
            "            )",
            "    actual = tuple(actual)",
            "    return actual[0] if len(actual) == 1 else actual",
            "",
            "",
            "def configure_readout_chain(soc):",
            "    if DDR_1MSPS_CONFIG is None:",
            "        return None",
            "    cfg = DDR_1MSPS_CONFIG",
            "    if cfg['input_board_type'] == 'RF_In':",
            "        configured = soc.rfb_set_ro_rf(cfg['ro_ch'], cfg['attenuation_db'])",
            "    else:",
            "        configured = soc.rfb_set_ro_dc(cfg['ro_ch'], cfg['dc_gain_db'])",
            "    if cfg['input_board_type'] == 'RF_In':",
            "        soc.rfb_set_ro_filter(",
            "            cfg['ro_ch'],",
            "            fc=cfg['filter_cutoff'],",
            "            bw=cfg['filter_bandwidth'],",
            "            ftype=cfg['filter_type'],",
            "        )",
            "    return configured",
            "",
            "",
            "def run_experiment(soc, soccfg, *, progress=True, configure_rf=True, **run_kwargs):",
            "    actual_outputs = configure_rf_chain(soc) if configure_rf else None",
            "    actual_input = configure_readout_chain(soc) if configure_rf else None",
            "    program = build_program(soccfg)",
            "    if DDR_1MSPS_CONFIG is not None:",
            "        ddr_result = program.acquire_fir_ddr(",
            "            soc, progress=progress, **run_kwargs",
            "        )",
            "    else:",
            "        program.run_rounds(soc, progress=progress, **run_kwargs)",
            "        ddr_result = None",
            "    rf_settings = {'outputs': actual_outputs, 'readout': actual_input}",
            "    return program, ddr_result, rf_settings",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "BIAS_T_COMPENSATION_MODES",
    "BIAS_T_COMPENSATION_TYPES",
    "DEFAULT_BIAS_T_COMPENSATION_DURATION_US",
    "DEFAULT_BIAS_T_COMPENSATION_FRACTION",
    "DEFAULT_BIAS_T_FILTER_TAU_US",
    "DEFAULT_DC_MEASURE_GAIN_V_PER_A",
    "DEFAULT_INITIAL_DURATION_NS",
    "DEFAULT_INITIAL_VOLTAGE_MV",
    "DEFAULT_INSERT_FLAT_NS",
    "DEFAULT_INSERT_RAMP_NS",
    "DEFAULT_QCS_FULL_SCALE_V",
    "DEFAULT_QICK_FABRIC_MHZ",
    "DEFAULT_QICK_TPROC_MHZ",
    "DEFAULT_QICK_FULL_SCALE_MV",
    "PulseSequence",
    "QickDdrReadoutSpec",
    "QickRfPulseSpec",
    "QickSegmentSpec",
    "QickSweepSpec",
    "WaveformInterval",
    "adc_iq_to_voltage",
    "build_qick_sequence",
    "dc_iq_to_current",
    "generate_qcs_program_code",
    "generate_qick_program_code",
    "make_qick_segment_specs",
    "qick_set_segment_names",
    "transform_virtual_waveforms",
]
