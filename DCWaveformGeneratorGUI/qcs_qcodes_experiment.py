"""Keysight QCS execution backend for the fine-tune Experiment workflow.

The existing waveform editor and sweep model are vendor-neutral.  This module
translates general :class:`FineTuneSequence` Cartesian points into QCS
``Program`` objects. General Experiment sweeps retain software point
iteration so every existing sweep type is preserved. Stability Diagram has a
specialized two-axis compiler that lowers DC amplitudes into native QCS 2.5.5
hardware sweeps and executes one Program for the complete grid.

``keysight.qcs`` is imported lazily so the QICK application remains usable in
environments where the proprietary Keysight package is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from .dc_waveform_core import (
        DEFAULT_QCS_FULL_SCALE_V,
        DEFAULT_QICK_FULL_SCALE_MV,
    )
    from .qick_fine_tune_sweep import (
        FineTuneDdrResult,
        RfDurationSweep,
        RfFrequencySweep,
        RfPowerSweep,
    )
    from .qick_qcodes_experiment import (
        AWG_METADATA_MODE_EXPANDED,
        DEFAULT_AWG_METADATA_MODE,
        QcodesRunConfig,
        build_awg_vertex_metadata,
        build_awg_waveform_recipe,
        normalize_awg_metadata_mode,
        store_experiment_result,
    )
except ImportError:
    from dc_waveform_core import (
        DEFAULT_QCS_FULL_SCALE_V,
        DEFAULT_QICK_FULL_SCALE_MV,
    )
    from qick_fine_tune_sweep import (
        FineTuneDdrResult,
        RfDurationSweep,
        RfFrequencySweep,
        RfPowerSweep,
    )
    from qick_qcodes_experiment import (
        AWG_METADATA_MODE_EXPANDED,
        DEFAULT_AWG_METADATA_MODE,
        QcodesRunConfig,
        build_awg_vertex_metadata,
        build_awg_waveform_recipe,
        normalize_awg_metadata_mode,
        store_experiment_result,
    )


ProgressCallback = Callable[[int, str], None]
EventCallback = Callable[[str, str, str], None]
MAX_QCS_SOFTWARE_SWEEP_POINTS = 10_000
MAX_QCS_HARDWARE_SWEEP_ARRAY_VALUES = 24_576
MAX_QCS_STABILITY_GRID_POINTS = 1_000_000
MAX_QCS_STABILITY_RESULT_VALUES = 2_000_000
# Keysight QCS 2.5.5 ``SAMPLE_RATES`` defines the M5200 digitizer at
# 4.8 GSa/s. Integration-filter acquisitions are emitted in 16-sample blocks.
QCS_M5200_SAMPLE_RATE_HZ = 4_800_000_000.0
QCS_M5200_INTEGRATION_BLOCK_SAMPLES = 16


class QcsUnsupportedFeatureError(ValueError):
    """Raised when a QICK-only semantic cannot be translated safely."""


def _positive_finite(value: Any, label: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _nonnegative_finite(value: Any, label: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be nonnegative and finite")
    return result


def _channel_name(value: Any, label: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


@dataclass(frozen=True)
class QcsConnectionConfig:
    """Serialized mapper and HCL execution settings."""

    mapper_path: str
    dc_channel_names: Tuple[str, ...]
    mapper_sha256: Optional[str] = None
    dc_full_scale_v: float = DEFAULT_QCS_FULL_SCALE_V
    rf_channel_names: Mapping[int, str] = field(default_factory=dict)
    acquisition_channel_name: Optional[str] = None
    hw_demod: bool = True
    init_time_s: float = 100e-6
    blocking: bool = True

    def __post_init__(self) -> None:
        mapper_path = str(self.mapper_path).strip()
        if not mapper_path:
            raise ValueError("QCS ChannelMapper file path must not be empty")
        mapper_sha256 = self.mapper_sha256
        if mapper_sha256 is not None:
            mapper_sha256 = str(mapper_sha256).strip().lower()
            if (
                len(mapper_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in mapper_sha256
                )
            ):
                raise ValueError(
                    "QCS mapper_sha256 must contain 64 hexadecimal characters"
                )
        dc_names = tuple(
            _channel_name(value, "QCS DC channel name")
            for value in self.dc_channel_names
        )
        if not dc_names:
            raise ValueError("at least one QCS DC channel name is required")
        if len(set(dc_names)) != len(dc_names):
            raise ValueError("QCS DC channel names must be unique")
        dc_full_scale_v = _positive_finite(
            self.dc_full_scale_v, "QCS DC full scale"
        )
        rf_names = {}
        for raw_channel, raw_name in dict(self.rf_channel_names).items():
            if isinstance(raw_channel, bool):
                raise TypeError("QCS RF generator numbers must be integers")
            channel = int(raw_channel)
            if channel < 0 or channel != raw_channel:
                raise ValueError(
                    "QCS RF generator numbers must be nonnegative integers"
                )
            rf_names[channel] = _channel_name(
                raw_name, "QCS RF channel name"
            )
        acquisition_name = self.acquisition_channel_name
        if acquisition_name is not None:
            acquisition_name = _channel_name(
                acquisition_name, "QCS acquisition channel name"
            )
        if not isinstance(self.hw_demod, bool):
            raise TypeError("QCS hw_demod must be boolean")
        if not isinstance(self.blocking, bool):
            raise TypeError("QCS blocking must be boolean")
        init_time_s = _nonnegative_finite(
            self.init_time_s, "QCS initialization time"
        )
        object.__setattr__(self, "mapper_path", mapper_path)
        object.__setattr__(self, "mapper_sha256", mapper_sha256)
        object.__setattr__(self, "dc_channel_names", dc_names)
        object.__setattr__(self, "dc_full_scale_v", dc_full_scale_v)
        object.__setattr__(self, "rf_channel_names", rf_names)
        object.__setattr__(
            self, "acquisition_channel_name", acquisition_name
        )
        object.__setattr__(self, "init_time_s", init_time_s)


@dataclass(frozen=True)
class QcsRfPulseConfig:
    """One RF waveform placed relative to a named fine-tune segment."""

    gen_ch: int
    at_segment: str
    duration_s: float
    amplitude: float
    frequency_hz: float
    phase_rad: float = 0.0
    delay_s: float = 0.0
    envelope: str = "constant"
    require_within_segment: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.gen_ch, bool) or int(self.gen_ch) != self.gen_ch:
            raise TypeError("QCS RF gen_ch must be an integer")
        if int(self.gen_ch) < 0:
            raise ValueError("QCS RF gen_ch must be nonnegative")
        at_segment = _channel_name(self.at_segment, "QCS RF segment")
        duration_s = _positive_finite(
            self.duration_s, "QCS RF duration"
        )
        amplitude = float(self.amplitude)
        if not isfinite(amplitude) or not -1.0 <= amplitude <= 1.0:
            raise ValueError("QCS RF amplitude must be in [-1, 1]")
        frequency_hz = float(self.frequency_hz)
        phase_rad = float(self.phase_rad)
        if not isfinite(frequency_hz):
            raise ValueError("QCS RF frequency must be finite")
        if not isfinite(phase_rad):
            raise ValueError("QCS RF phase must be finite")
        delay_s = _nonnegative_finite(self.delay_s, "QCS RF delay")
        envelope = str(self.envelope).strip().lower()
        if envelope not in {"constant", "gaussian"}:
            raise ValueError(
                "QCS RF envelope must be 'constant' or 'gaussian'"
            )
        if not isinstance(self.require_within_segment, bool):
            raise TypeError("QCS RF require_within_segment must be boolean")
        object.__setattr__(self, "gen_ch", int(self.gen_ch))
        object.__setattr__(self, "at_segment", at_segment)
        object.__setattr__(self, "duration_s", duration_s)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "phase_rad", phase_rad)
        object.__setattr__(self, "delay_s", delay_s)
        object.__setattr__(self, "envelope", envelope)


@dataclass(frozen=True)
class QcsAcquisitionConfig:
    """Digitizer acquisition placed relative to a named segment."""

    at_segment: str
    duration_s: float
    pre_delay_s: float = 0.0
    sample_rate_hz: float = QCS_M5200_SAMPLE_RATE_HZ
    sample_count: Optional[int] = None
    integration_filter: Any = None
    frequency_hz: float = 0.0
    phase_rad: float = 0.0
    envelope: str = "constant"

    def __post_init__(self) -> None:
        at_segment = _channel_name(
            self.at_segment, "QCS acquisition segment"
        )
        duration_s = _positive_finite(
            self.duration_s, "QCS acquisition duration"
        )
        pre_delay_s = _nonnegative_finite(
            self.pre_delay_s, "QCS acquisition pre-delay"
        )
        sample_rate_hz = _positive_finite(
            self.sample_rate_hz, "QCS acquisition sample rate"
        )
        sample_count = self.sample_count
        if sample_count is not None:
            if isinstance(sample_count, bool):
                raise TypeError("QCS acquisition sample count must be an integer")
            integer_sample_count = int(sample_count)
            if integer_sample_count < 1 or integer_sample_count != sample_count:
                raise ValueError(
                    "QCS acquisition sample count must be a positive integer"
                )
            sample_count = integer_sample_count
        frequency_hz = float(self.frequency_hz)
        phase_rad = float(self.phase_rad)
        if not isfinite(frequency_hz):
            raise ValueError("QCS acquisition frequency must be finite")
        if not isfinite(phase_rad):
            raise ValueError("QCS acquisition phase must be finite")
        envelope = str(self.envelope).strip().lower()
        if envelope not in {"constant", "gaussian"}:
            raise ValueError(
                "QCS acquisition envelope must be 'constant' or 'gaussian'"
            )
        object.__setattr__(self, "at_segment", at_segment)
        object.__setattr__(self, "duration_s", duration_s)
        object.__setattr__(self, "pre_delay_s", pre_delay_s)
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "phase_rad", phase_rad)
        object.__setattr__(self, "envelope", envelope)


@dataclass(frozen=True)
class QcsCompiledPoint:
    """One QCS Program and the channels needed to read its result."""

    point_index: int
    program: Any
    acquisition_channels: Any
    duration_s: float
    acquisition_duration_s: Optional[float] = None
    acquisition_sample_rate_hz: Optional[float] = None


@dataclass(frozen=True)
class QcsCompiledHardwareSweep:
    """One QCS Program containing an instrument-side Cartesian sweep."""

    program: Any
    acquisition_channels: Any
    duration_s: float
    acquisition_duration_s: float
    acquisition_sample_rate_hz: float
    sweep_shape: Tuple[int, int]


@dataclass(frozen=True)
class QcsExecutionResult:
    """Persistence-free output from QCS compilation and execution."""

    ddr_result: FineTuneDdrResult
    programs: Tuple[Any, ...]
    raw_results: Tuple[Any, ...]
    program_summary: Mapping[str, Any]
    rf_settings: Mapping[str, Any]


@dataclass
class StoredQcsExperiment:
    """Stored result compatible with the existing Experiment result handlers."""

    run_id: int
    guid: str
    database_path: Path
    row_count: int
    dataset: Any
    program: Any
    ddr_result: FineTuneDdrResult
    rf_settings: Mapping[str, Any]
    programs: Tuple[Any, ...] = ()
    raw_results: Tuple[Any, ...] = ()
    program_summary: Mapping[str, Any] = field(default_factory=dict)


def _import_qcs():
    try:
        import keysight.qcs as qcs
    except ImportError as exc:
        raise ImportError(
            "Keysight QCS 2.5.5 is required for the QCS backend. "
            "Install it in the QCS Conda environment or select QICK."
        ) from exc
    return qcs


def load_qcs_channel_mapper(
    connection_config: QcsConnectionConfig,
    *,
    qcs_module=None,
):
    """Load and type-check the serialized QCS ChannelMapper."""
    qcs = _import_qcs() if qcs_module is None else qcs_module
    path = Path(connection_config.mapper_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"QCS ChannelMapper file not found: {path}")
    if connection_config.mapper_sha256 is not None:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != connection_config.mapper_sha256:
            raise ValueError(
                "The QCS ChannelMapper file does not match the hardware "
                "configuration selected in the GUI"
            )
    mapper = qcs.load(path)
    mapper_type = getattr(qcs, "ChannelMapper", None)
    if mapper_type is not None and not isinstance(mapper, mapper_type):
        raise TypeError(
            f"{path} contains {type(mapper).__name__}, not ChannelMapper"
        )
    return mapper


def _mapper_channels(mapper: Any) -> Tuple[Any, ...]:
    channels = getattr(mapper, "channels", None)
    if channels is None and isinstance(mapper, Mapping):
        channels = tuple(mapper.values())
    if channels is None:
        raise TypeError("QCS ChannelMapper does not expose virtual channels")
    return tuple(channels)


def _resolve_mapper_channel(mapper: Any, name: str) -> Any:
    matches = [
        channel
        for channel in _mapper_channels(mapper)
        if str(getattr(channel, "name", "")).strip() == name
    ]
    if not matches:
        available = sorted(
            {
                str(getattr(channel, "name", "")).strip()
                for channel in _mapper_channels(mapper)
                if str(getattr(channel, "name", "")).strip()
            }
        )
        suffix = f"; available names: {', '.join(available)}" if available else ""
        raise KeyError(f"QCS mapper has no virtual channel {name!r}{suffix}")
    if len(matches) > 1:
        raise ValueError(
            f"QCS mapper contains duplicate virtual channel name {name!r}"
        )
    channel = matches[0]
    labels = getattr(channel, "labels", None)
    if labels is not None and len(tuple(labels)) != 1:
        raise ValueError(
            f"QCS virtual channel {name!r} must contain exactly one label"
        )
    return channel


def _validate_mapped_hardware_role(
    mapper: Any,
    channel: Any,
    *,
    name: str,
    role: str,
    expected_instruments: Sequence[str],
    require_relative_phase: bool = False,
) -> None:
    """Preflight role semantics exposed by a native QCS ChannelMapper.

    Lightweight injected test adapters do not necessarily expose physical
    mappings, so this check is conditional on the real mapper API. HCL-backed
    operation always supplies that API.
    """
    get_physical_channels = getattr(mapper, "get_physical_channels", None)
    if not callable(get_physical_channels):
        return
    physical_channels = tuple(get_physical_channels(channel))
    if len(physical_channels) != 1:
        raise ValueError(
            f"QCS {role} virtual channel {name!r} must map to exactly one "
            f"physical connector; found {len(physical_channels)}"
        )
    physical = physical_channels[0]
    instrument = str(getattr(physical, "instrument", ""))
    allowed = tuple(str(value) for value in expected_instruments)
    if instrument not in allowed:
        raise ValueError(
            f"QCS {role} virtual channel {name!r} must map to "
            f"{' or '.join(allowed)}; found {instrument or 'unknown hardware'}"
        )
    if require_relative_phase and bool(
        getattr(channel, "absolute_phase", False)
    ):
        raise ValueError(
            f"QCS {role} virtual channel {name!r} must use "
            "absolute_phase=False for a native amplitude hardware sweep"
        )


def _mapped_channel_sample_rate(
    mapper: Any,
    channel: Any,
) -> Optional[float]:
    """Return the mapper's physical sample rate when the API exposes it."""
    get_physical_channels = getattr(mapper, "get_physical_channels", None)
    if not callable(get_physical_channels):
        return None
    physical_channels = tuple(get_physical_channels(channel))
    if len(physical_channels) != 1:
        raise ValueError(
            "QCS acquisition virtual channel must map to exactly one "
            "physical digitizer channel"
        )
    sample_rate = getattr(physical_channels[0], "sample_rate", None)
    if sample_rate is None:
        return None
    return _positive_finite(
        sample_rate, "mapped QCS digitizer sample rate"
    )


def _resolved_acquisition_timing(
    mapper: Any,
    channel: Any,
    acquisition: QcsAcquisitionConfig,
    *,
    hardware_demodulation: bool,
) -> tuple[float, float]:
    """Resolve M5200 timing and enforce QCS integration-block alignment."""

    sample_rate_hz = _mapped_channel_sample_rate(mapper, channel)
    if sample_rate_hz is None:
        sample_rate_hz = acquisition.sample_rate_hz
    duration_s = acquisition.duration_s
    if acquisition.sample_count is not None:
        sample_count = acquisition.sample_count
        if (
            hardware_demodulation
            and sample_count % QCS_M5200_INTEGRATION_BLOCK_SAMPLES != 0
        ):
            raise ValueError(
                "QCS M5200 hardware-demodulation integration length must "
                f"be a multiple of "
                f"{QCS_M5200_INTEGRATION_BLOCK_SAMPLES} samples; got "
                f"{sample_count}"
            )
        duration_s = sample_count / sample_rate_hz
    elif hardware_demodulation:
        rendered_samples = duration_s * sample_rate_hz
        sample_count = int(round(rendered_samples))
        if (
            not np.isclose(
                rendered_samples,
                sample_count,
                rtol=0.0,
                atol=1.0e-6,
            )
            or sample_count % QCS_M5200_INTEGRATION_BLOCK_SAMPLES != 0
        ):
            raise ValueError(
                "QCS M5200 hardware-demodulation duration must render to "
                f"a multiple of {QCS_M5200_INTEGRATION_BLOCK_SAMPLES} "
                f"samples at {sample_rate_hz:g} S/s; got "
                f"{rendered_samples:g} samples"
            )
    return duration_s, sample_rate_hz


def _executed_program_sample_rate(
    program: Any,
    channel: Any,
) -> Optional[float]:
    get_sample_rates = getattr(program, "get_sample_rates", None)
    if not callable(get_sample_rates):
        return None
    values = get_sample_rates(channel)
    value = _first_result_value(values, channel)
    return _positive_finite(
        value, "executed QCS digitizer sample rate"
    )


def _segment_boundaries_seconds(
    boundaries: Sequence[Tuple[str, float, float]],
    *,
    fabric_mhz: float,
) -> Mapping[str, Tuple[float, float]]:
    scale = 1.0 / (_positive_finite(fabric_mhz, "fabric clock") * 1e6)
    return {
        str(name): (float(start) * scale, float(stop) * scale)
        for name, start, stop in boundaries
    }


def _qcs_envelope(qcs: Any, name: str) -> Any:
    if name == "gaussian":
        return qcs.GaussianEnvelope()
    return qcs.ConstantEnvelope()


def _dc_waveform(
    qcs: Any,
    *,
    duration_s: float,
    times_s: np.ndarray,
    amplitudes: np.ndarray,
    name: str,
) -> Any:
    amplitudes = np.asarray(amplitudes, dtype=float)
    if amplitudes.ndim != 1 or len(amplitudes) != len(times_s):
        raise ValueError("QCS DC vertex times and amplitudes must be 1D peers")
    scale = float(np.max(np.abs(amplitudes), initial=0.0))
    if scale == 0.0:
        envelope = qcs.ConstantEnvelope()
        amplitude = 0.0
    else:
        # ArbitraryEnvelope normalizes its input to the unit disc.  Supplying
        # the physical scale separately preserves sub-full-scale waveforms.
        envelope = qcs.ArbitraryEnvelope(times_s, amplitudes)
        amplitude = scale
    return qcs.DCWaveform(
        duration=duration_s,
        envelope=envelope,
        amplitude=amplitude,
        name=name,
    )


def _point_rf_pulse(
    sequence: Any,
    pulse: QcsRfPulseConfig,
    point_index: int,
) -> QcsRfPulseConfig:
    duration_s = pulse.duration_s
    frequency_hz = pulse.frequency_hz
    coordinate = sequence.sweep_coordinate(point_index)
    for axis_index, axis in enumerate(sequence.sweep_axes):
        if (
            isinstance(axis, RfDurationSweep)
            and axis.gen_ch == pulse.gen_ch
            and axis.segment_name == pulse.at_segment
        ):
            duration_s = float(coordinate[axis_index]) * 1e-6
        elif (
            isinstance(axis, RfFrequencySweep)
            and axis.gen_ch == pulse.gen_ch
            and axis.segment_name == pulse.at_segment
        ):
            frequency_hz = float(coordinate[axis_index]) * 1e6
        elif (
            isinstance(axis, RfPowerSweep)
            and axis.gen_ch == pulse.gen_ch
            and axis.segment_name == pulse.at_segment
        ):
            raise QcsUnsupportedFeatureError(
                "calibrated RF connector-power sweeps are QICK-specific; "
                "use a QCS amplitude calibration before enabling this sweep"
            )
    return QcsRfPulseConfig(
        gen_ch=pulse.gen_ch,
        at_segment=pulse.at_segment,
        duration_s=duration_s,
        amplitude=pulse.amplitude,
        frequency_hz=frequency_hz,
        phase_rad=pulse.phase_rad,
        delay_s=pulse.delay_s,
        envelope=pulse.envelope,
        require_within_segment=pulse.require_within_segment,
    )


def validate_qcs_capabilities(
    *,
    connection_config: QcsConnectionConfig,
    sequence: Any,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
) -> None:
    """Reject configurations whose semantics cannot be preserved."""
    if len(connection_config.dc_channel_names) != int(sequence.n_outputs):
        raise ValueError(
            "QCS DC channel count must match the waveform output count "
            f"({len(connection_config.dc_channel_names)} configured, "
            f"{sequence.n_outputs} required)"
        )
    if int(sequence.sweep_point_count) > MAX_QCS_SOFTWARE_SWEEP_POINTS:
        raise QcsUnsupportedFeatureError(
            "QCS software sweeps are limited to "
            f"{MAX_QCS_SOFTWARE_SWEEP_POINTS:,} Cartesian points; reduce "
            "the sweep or add a hardware-sweep implementation"
        )
    if not connection_config.blocking:
        raise QcsUnsupportedFeatureError(
            "QCS Experiment execution requires blocking=True so acquisition "
            "data is complete before it is normalized and saved"
        )
    segment_names = {str(segment.name) for segment in sequence.segments}
    for axis in sequence.sweep_axes:
        if isinstance(axis, RfPowerSweep):
            raise QcsUnsupportedFeatureError(
                "calibrated RF connector-power sweeps are not yet available "
                "on the QCS backend"
            )
    for pulse in rf_pulses:
        if pulse.at_segment not in segment_names:
            raise KeyError(
                f"QCS RF pulse references unknown segment {pulse.at_segment!r}"
            )
        if pulse.gen_ch not in connection_config.rf_channel_names:
            raise KeyError(
                f"no QCS virtual RF channel is mapped for gen_ch {pulse.gen_ch}"
            )
    if acquisition is not None:
        if acquisition.at_segment not in segment_names:
            raise KeyError(
                "QCS acquisition references unknown segment "
                f"{acquisition.at_segment!r}"
            )
        if connection_config.acquisition_channel_name is None:
            raise ValueError(
                "a QCS acquisition virtual-channel name is required"
            )
        if (
            not connection_config.hw_demod
            and acquisition.integration_filter is not None
        ):
            raise QcsUnsupportedFeatureError(
                "a custom QCS integration filter requires hardware "
                "demodulation; raw-trace mode uses an acquisition duration"
            )


def compile_qcs_point(
    sequence: Any,
    point_index: int,
    *,
    connection_config: QcsConnectionConfig,
    mapper: Any,
    repetitions_per_sweep: int,
    fabric_mhz: float = 300.0,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
    qcs_module=None,
) -> QcsCompiledPoint:
    """Compile one Cartesian point into an executable QCS Program."""
    qcs = _import_qcs() if qcs_module is None else qcs_module
    if isinstance(repetitions_per_sweep, bool):
        raise TypeError("repetitions_per_sweep must be an integer")
    repetitions = int(repetitions_per_sweep)
    if repetitions < 1 or repetitions != repetitions_per_sweep:
        raise ValueError("repetitions_per_sweep must be a positive integer")
    point_index = int(point_index)
    if not 0 <= point_index < int(sequence.sweep_point_count):
        raise IndexError("QCS point index is out of range")
    validate_qcs_capabilities(
        connection_config=connection_config,
        sequence=sequence,
        rf_pulses=rf_pulses,
        acquisition=acquisition,
    )

    times_cycles, waveforms, boundaries = (
        sequence.compensated_waveform_vertices(point_index)
    )
    seconds_per_cycle = 1.0 / (
        _positive_finite(fabric_mhz, "fabric clock") * 1e6
    )
    relative_amplitude_scale = _positive_finite(
        source_full_scale_mv, "source waveform full scale"
    ) / (connection_config.dc_full_scale_v * 1000.0)
    times_s = np.asarray(times_cycles, dtype=float) * seconds_per_cycle
    if times_s.ndim != 1 or len(times_s) < 2 or times_s[-1] <= 0.0:
        raise ValueError("QCS sequence must have a positive duration")
    duration_s = float(times_s[-1])
    boundary_seconds = _segment_boundaries_seconds(
        boundaries, fabric_mhz=fabric_mhz
    )

    program = qcs.Program(
        name=f"PulseGenerator point {point_index + 1}"
    )
    for output_index, (output_name, channel_name) in enumerate(
        zip(sequence.output_names, connection_config.dc_channel_names)
    ):
        channel = _resolve_mapper_channel(mapper, channel_name)
        source_amplitudes = np.asarray(
            waveforms[output_name], dtype=float
        )
        peak_voltage_v = (
            float(np.max(np.abs(source_amplitudes), initial=0.0))
            * float(source_full_scale_mv)
            / 1000.0
        )
        if peak_voltage_v > connection_config.dc_full_scale_v + 1e-12:
            raise ValueError(
                f"QCS DC output {output_name!r} at software point "
                f"{point_index + 1} reaches {peak_voltage_v:.6g} V, "
                "exceeding the configured +/-"
                f"{connection_config.dc_full_scale_v:.6g} V full scale"
            )
        waveform = _dc_waveform(
            qcs,
            duration_s=duration_s,
            times_s=times_s,
            amplitudes=(
                source_amplitudes * relative_amplitude_scale
            ),
            name=f"{output_name}_point_{point_index}",
        )
        program.add_waveform(
            waveform,
            channel,
            new_layer=output_index == 0,
        )

    for pulse in rf_pulses:
        current = _point_rf_pulse(sequence, pulse, point_index)
        if current.at_segment not in boundary_seconds:
            raise KeyError(
                f"no timing boundary for RF segment {current.at_segment!r}"
            )
        segment_start_s, segment_stop_s = boundary_seconds[
            current.at_segment
        ]
        pre_delay_s = segment_start_s + current.delay_s
        if (
            current.require_within_segment
            and pre_delay_s + current.duration_s > segment_stop_s + 1e-15
        ):
            raise ValueError(
                f"QCS RF pulse on gen_ch {current.gen_ch} exceeds segment "
                f"{current.at_segment!r}"
            )
        channel_name = connection_config.rf_channel_names[current.gen_ch]
        channel = _resolve_mapper_channel(mapper, channel_name)
        waveform = qcs.RFWaveform(
            duration=current.duration_s,
            envelope=_qcs_envelope(qcs, current.envelope),
            amplitude=current.amplitude,
            rf_frequency=current.frequency_hz,
            instantaneous_phase=current.phase_rad,
            name=f"rf_{current.gen_ch}_point_{point_index}",
        )
        program.add_waveform(
            waveform,
            channel,
            new_layer=False,
            pre_delay=pre_delay_s,
        )

    acquisition_channels = None
    acquisition_duration_s = None
    acquisition_sample_rate_hz = None
    if acquisition is not None:
        segment_start_s, segment_stop_s = boundary_seconds[
            acquisition.at_segment
        ]
        pre_delay_s = segment_start_s + acquisition.pre_delay_s
        acquisition_channels = _resolve_mapper_channel(
            mapper, connection_config.acquisition_channel_name
        )
        (
            acquisition_duration_s,
            acquisition_sample_rate_hz,
        ) = _resolved_acquisition_timing(
            mapper,
            acquisition_channels,
            acquisition,
            hardware_demodulation=connection_config.hw_demod,
        )
        if pre_delay_s + acquisition_duration_s > segment_stop_s + 1e-15:
            raise ValueError(
                "QCS acquisition exceeds segment "
                f"{acquisition.at_segment!r}"
            )
        if connection_config.hw_demod:
            integration_filter = acquisition.integration_filter
            if integration_filter is None:
                integration_filter = qcs.RFWaveform(
                    duration=acquisition_duration_s,
                    envelope=_qcs_envelope(qcs, acquisition.envelope),
                    amplitude=1.0,
                    rf_frequency=acquisition.frequency_hz,
                    instantaneous_phase=acquisition.phase_rad,
                    name=f"acquisition_filter_point_{point_index}",
                )
        else:
            # QCS requests raw trace capture by supplying a duration instead
            # of an IntegrationFilter/RFWaveform.
            integration_filter = acquisition_duration_s
        program.add_acquisition(
            integration_filter=integration_filter,
            channels=acquisition_channels,
            new_layer=False,
            pre_delay=pre_delay_s,
        )

    program.n_shots(repetitions)
    return QcsCompiledPoint(
        point_index=point_index,
        program=program,
        acquisition_channels=acquisition_channels,
        duration_s=duration_s,
        acquisition_duration_s=acquisition_duration_s,
        acquisition_sample_rate_hz=acquisition_sample_rate_hz,
    )


def compile_qcs_sequence(
    sequence: Any,
    *,
    connection_config: QcsConnectionConfig,
    mapper: Any,
    repetitions_per_sweep: int,
    fabric_mhz: float = 300.0,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
    qcs_module=None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[QcsCompiledPoint, ...]:
    """Compile every Cartesian point in C order."""
    count = int(sequence.sweep_point_count)
    compiled = []
    for point_index in range(count):
        compiled.append(
            compile_qcs_point(
                sequence,
                point_index,
                connection_config=connection_config,
                mapper=mapper,
                repetitions_per_sweep=repetitions_per_sweep,
                fabric_mhz=fabric_mhz,
                source_full_scale_mv=source_full_scale_mv,
                rf_pulses=rf_pulses,
                acquisition=acquisition,
                qcs_module=qcs_module,
            )
        )
        if progress_callback is not None:
            percent = 10 + int(25 * (point_index + 1) / count)
            progress_callback(
                percent,
                f"Compiled QCS point {point_index + 1:,}/{count:,}",
            )
    return tuple(compiled)


def _executor_execute(executor: Any, program: Any) -> Any:
    if hasattr(executor, "execute"):
        return executor.execute(program)
    if callable(executor):
        return executor(program)
    raise TypeError("QCS executor must be callable or expose execute(program)")


def _first_result_value(value: Any, channels: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    try:
        if channels in value:
            return value[channels]
    except TypeError:
        pass
    if len(value) != 1:
        raise ValueError(
            "QCS acquisition returned multiple channel arrays unexpectedly"
        )
    return next(iter(value.values()))


def extract_qcs_acquisition(
    raw_result: Any,
    channels: Any,
    *,
    prefer_trace: bool = False,
) -> Any:
    """Extract an IQ/trace array from a real or injected QCS result."""
    results = getattr(raw_result, "results", None)
    if results is not None:
        get_trace = getattr(results, "get_trace", None)
        if prefer_trace and callable(get_trace):
            return _first_result_value(
                get_trace(channels, avg=False), channels
            )
        get_iq = getattr(results, "get_iq", None)
        if callable(get_iq):
            try:
                return _first_result_value(
                    get_iq(channels, avg=False), channels
                )
            except (KeyError, RuntimeError, TypeError, ValueError):
                if callable(get_trace):
                    return _first_result_value(
                        get_trace(channels, avg=False), channels
                    )
                raise
    try:
        return raw_result[channels]
    except (IndexError, KeyError, TypeError):
        return raw_result


def normalize_qcs_iq(
    values: Any,
    *,
    repetitions_per_sweep: int,
    real_is_i_trace: bool = False,
) -> np.ndarray:
    """Normalize one point to ``(repetition, sample, I/Q)``."""
    repetitions = int(repetitions_per_sweep)
    if repetitions < 1:
        raise ValueError("repetitions_per_sweep must be positive")

    if isinstance(values, (tuple, list)) and len(values) == 2:
        i_values = np.asarray(values[0])
        q_values = np.asarray(values[1])
        if i_values.shape != q_values.shape:
            raise ValueError("QCS I and Q arrays have different shapes")
        array = i_values.astype(float) + 1j * q_values.astype(float)
    else:
        array = np.asarray(values)

    if array.size == 0:
        raise ValueError("QCS acquisition returned no samples")
    if np.iscomplexobj(array):
        if array.size % repetitions:
            raise ValueError(
                "QCS IQ element count is not divisible by repetitions"
            )
        # QCS stores shots on the final result axis.  Accept an already
        # repetition-first injected array as a convenience for tests/adapters,
        # then fall back to the legacy flat representation when neither
        # boundary dimension identifies the shot count.
        if array.ndim > 0 and array.shape[0] == repetitions:
            complex_values = array.reshape(repetitions, -1)
        elif array.ndim > 1 and array.shape[-1] == repetitions:
            complex_values = np.moveaxis(array, -1, 0).reshape(
                repetitions, -1
            )
        else:
            complex_values = array.reshape(repetitions, -1)
        return np.stack(
            (complex_values.real, complex_values.imag), axis=-1
        ).astype(np.float64, copy=False)

    if (
        not real_is_i_trace
        and array.ndim >= 1
        and array.shape[-1] == 2
    ):
        if array.size % (repetitions * 2):
            raise ValueError(
                "QCS I/Q element count is not divisible by repetitions"
            )
        if array.shape[0] == repetitions:
            iq_values = array
        elif array.ndim > 2 and array.shape[-2] == repetitions:
            iq_values = np.moveaxis(array, -2, 0)
        else:
            iq_values = array.reshape(repetitions, -1, 2)
        return iq_values.reshape(repetitions, -1, 2).astype(
            np.float64, copy=False
        )
    if np.issubdtype(array.dtype, np.number):
        if array.size % repetitions:
            raise ValueError(
                "QCS trace element count is not divisible by repetitions"
            )
        if array.ndim > 0 and array.shape[0] == repetitions:
            i_values = array.reshape(repetitions, -1)
        elif array.ndim > 1 and array.shape[-1] == repetitions:
            i_values = np.moveaxis(array, -1, 0).reshape(
                repetitions, -1
            )
        else:
            i_values = array.reshape(repetitions, -1)
        i_values = i_values.astype(np.float64, copy=False)
        return np.stack((i_values, np.zeros_like(i_values)), axis=-1)
    raise TypeError(
        "QCS acquisition must be numeric, complex-valued, an (I, Q) pair, "
        "or have a final I/Q axis of length 2"
    )


def normalize_qcs_hardware_sweep_iq(
    values: Any,
    *,
    repetitions_per_point: int,
    sweep_shape: Sequence[int],
) -> np.ndarray:
    """Normalize QCS hardware-sweep IQ to ``(point, shot, 1, I/Q)``.

    Native QCS 2.5.5 results follow the Program repetition order with the
    shot axis first. A shot-last form is also accepted for injected adapters
    and older result loaders.
    """
    repetitions = int(repetitions_per_point)
    if repetitions < 1:
        raise ValueError("repetitions_per_point must be positive")
    shape = tuple(int(value) for value in sweep_shape)
    if len(shape) != 2 or any(value < 1 for value in shape):
        raise ValueError("QCS Stability sweep_shape must contain two axes")

    if isinstance(values, (tuple, list)) and len(values) == 2:
        i_values = np.asarray(values[0])
        q_values = np.asarray(values[1])
        if i_values.shape != q_values.shape:
            raise ValueError("QCS hardware-sweep I and Q shapes differ")
        array = i_values.astype(float) + 1j * q_values.astype(float)
    else:
        array = np.asarray(values)
        if (
            not np.iscomplexobj(array)
            and array.ndim >= 1
            and array.shape[-1] == 2
        ):
            array = (
                array[..., 0].astype(float)
                + 1j * array[..., 1].astype(float)
            )

    expected_count = repetitions * int(np.prod(shape))
    if array.size != expected_count:
        raise ValueError(
            "QCS hardware-sweep IQ contains "
            f"{array.size:,} values; expected {expected_count:,} for "
            f"{shape[0]} x {shape[1]} points x {repetitions} repetitions"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("QCS hardware-sweep IQ must be numeric")

    repetition_first = (repetitions, *shape)
    repetition_last = (*shape, repetitions)
    if array.shape == repetition_first:
        grid = array
    elif array.shape == repetition_last:
        grid = np.moveaxis(array, -1, 0)
    elif repetitions == 1 and array.shape == shape:
        grid = array[np.newaxis, ...]
    elif array.ndim == 1:
        # Native QCS repetition order is (shot, X, Y). A loader that strips
        # shape metadata still preserves that C-order in its flat buffer.
        grid = array.reshape(repetition_first)
    else:
        raise ValueError(
            "QCS hardware-sweep IQ shape must be "
            f"{repetition_first} or {repetition_last}; received {array.shape}"
        )

    point_shot = np.moveaxis(grid, 0, -1).reshape(-1, repetitions)
    return np.stack(
        (point_shot.real, point_shot.imag),
        axis=-1,
    ).astype(np.float64, copy=False)[:, :, np.newaxis, :]


def build_qcs_executor(
    connection_config: QcsConnectionConfig,
    mapper: Any,
    *,
    qcs_module=None,
) -> Any:
    """Create the blocking HCL executor shared by QCS scan iterations."""
    qcs = _import_qcs() if qcs_module is None else qcs_module
    backend = qcs.HclBackend(
        channel_mapper=mapper,
        hw_demod=connection_config.hw_demod,
        init_time=connection_config.init_time_s,
        blocking=connection_config.blocking,
        suppress_rounding_warnings=True,
        keep_progress_bar=False,
    )
    return qcs.Executor(backend)


def compile_qcs_stability_hardware_sweep(
    sequence: Any,
    *,
    connection_config: QcsConnectionConfig,
    mapper: Any,
    repetitions_per_point: int,
    fabric_mhz: float = 300.0,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
    qcs_module=None,
) -> QcsCompiledHardwareSweep:
    """Compile a two-axis Stability Diagram into one native QCS sweep.

    The Y sweep is added first, followed by X and then ``n_shots``. QCS
    prepends each repetition, yielding ``(shot, X, Y)`` with every loop in
    hardware and Y as the fastest Cartesian axis.
    """
    qcs = _import_qcs() if qcs_module is None else qcs_module
    if acquisition is None:
        raise ValueError(
            "QCS Stability hardware sweep requires an acquisition"
        )
    if not connection_config.hw_demod:
        raise QcsUnsupportedFeatureError(
            "QCS Stability hardware sweep requires hardware demodulation; "
            "QCS 2.5.5 cannot return raw traces from a hardware sweep"
        )
    if not connection_config.blocking:
        raise QcsUnsupportedFeatureError(
            "QCS Stability hardware sweep requires blocking=True"
        )
    if getattr(sequence, "bias_t_compensation", None) is not None:
        raise QcsUnsupportedFeatureError(
            "QCS Stability hardware sweep does not support Bias-T "
            "compensation; disable it before running with QCS"
        )

    if isinstance(repetitions_per_point, bool):
        raise TypeError("repetitions_per_point must be an integer")
    repetitions = int(repetitions_per_point)
    if repetitions < 1 or repetitions != repetitions_per_point:
        raise ValueError("repetitions_per_point must be a positive integer")

    axes = tuple(sequence.sweep_axes)
    if len(axes) != 2 or any(
        getattr(axis, "axis_kind", "amplitude") != "amplitude"
        for axis in axes
    ):
        raise QcsUnsupportedFeatureError(
            "QCS Stability requires exactly two DC-amplitude sweep axes"
        )
    x_axis, y_axis = axes
    x_values = np.asarray(x_axis.points, dtype=float)
    y_values = np.asarray(y_axis.points, dtype=float)
    sweep_shape = (x_values.size, y_values.size)
    if tuple(sequence.sweep_shape) != sweep_shape:
        raise ValueError(
            "QCS Stability sequence sweep shape does not match its axes"
        )
    point_count = int(x_values.size) * int(y_values.size)
    if point_count > MAX_QCS_STABILITY_GRID_POINTS:
        raise QcsUnsupportedFeatureError(
            "QCS Stability grid contains "
            f"{point_count:,} Cartesian points; the safe application limit "
            f"is {MAX_QCS_STABILITY_GRID_POINTS:,}"
        )
    result_value_count = point_count * repetitions
    if result_value_count > MAX_QCS_STABILITY_RESULT_VALUES:
        raise QcsUnsupportedFeatureError(
            "QCS Stability would return "
            f"{result_value_count:,} hardware-demodulated IQ values; the "
            f"safe application limit is "
            f"{MAX_QCS_STABILITY_RESULT_VALUES:,}. Reduce points or "
            "repetitions."
        )

    if len(connection_config.dc_channel_names) != int(sequence.n_outputs):
        raise ValueError(
            "QCS DC channel count must match the Stability output count"
        )
    if connection_config.acquisition_channel_name is None:
        raise ValueError(
            "a QCS acquisition virtual-channel name is required"
        )
    dc_channels = []
    for name in connection_config.dc_channel_names:
        channel = _resolve_mapper_channel(mapper, name)
        _validate_mapped_hardware_role(
            mapper,
            channel,
            name=name,
            role="DC",
            expected_instruments=("M5301AWG",),
            require_relative_phase=True,
        )
        dc_channels.append(channel)
    for pulse in rf_pulses:
        if pulse.gen_ch not in connection_config.rf_channel_names:
            raise KeyError(
                f"no QCS virtual RF channel is mapped for gen_ch "
                f"{pulse.gen_ch}"
            )
        rf_channel = _resolve_mapper_channel(
            mapper, connection_config.rf_channel_names[pulse.gen_ch]
        )
        _validate_mapped_hardware_role(
            mapper,
            rf_channel,
            name=connection_config.rf_channel_names[pulse.gen_ch],
            role="RF",
            expected_instruments=("M5300AWG", "M5301AWG"),
        )
    acquisition_channels = _resolve_mapper_channel(
        mapper, connection_config.acquisition_channel_name
    )
    _validate_mapped_hardware_role(
        mapper,
        acquisition_channels,
        name=connection_config.acquisition_channel_name,
        role="acquisition",
        expected_instruments=("M5200Digitizer",),
    )
    (
        acquisition_duration_s,
        acquisition_sample_rate_hz,
    ) = _resolved_acquisition_timing(
        mapper,
        acquisition_channels,
        acquisition,
        hardware_demodulation=True,
    )

    segments = tuple(sequence.segments)
    if (
        len(segments) != 1
        or str(segments[0].kind) != "set"
        or any(
            str(axis.segment_name) != str(segments[0].name)
            for axis in axes
        )
    ):
        raise QcsUnsupportedFeatureError(
            "QCS Stability hardware sweep requires its dedicated "
            "single SET-and-hold sequence"
        )
    segment = segments[0]
    fabric_hz = _positive_finite(fabric_mhz, "fabric clock") * 1e6
    duration_s = float(segment.duration_cycles) / fabric_hz
    if duration_s <= 0.0:
        raise ValueError("QCS Stability sequence duration must be positive")

    output_names = tuple(str(name) for name in sequence.output_names)
    try:
        x_output_index = output_names.index(str(x_axis.output_name))
        y_output_index = output_names.index(str(y_axis.output_name))
    except ValueError as exc:
        raise ValueError(
            "QCS Stability sweep axes do not match the output names"
        ) from exc
    if x_output_index == y_output_index:
        raise ValueError("QCS Stability X and Y outputs must differ")

    cross_capacitance = np.asarray(
        sequence.cross_capacitance,
        dtype=float,
    )
    expected_matrix_shape = (len(output_names), len(output_names))
    if cross_capacitance.shape != expected_matrix_shape:
        raise ValueError(
            "QCS Stability cross-capacitance matrix must have shape "
            f"{expected_matrix_shape}"
        )
    if not np.all(np.isfinite(cross_capacitance)):
        raise ValueError(
            "QCS Stability cross-capacitance coefficients must be finite"
        )

    raw_offsets = tuple(segment.amplitudes)
    if len(raw_offsets) != len(output_names):
        raise ValueError("QCS Stability SET amplitude count is invalid")
    virtual_offset = np.asarray(
        [0.0 if value is None else float(value) for value in raw_offsets],
        dtype=float,
    )
    # The selected coordinates replace, rather than add to, the SET values.
    virtual_offset[x_output_index] = 0.0
    virtual_offset[y_output_index] = 0.0
    qcs_scale = _positive_finite(
        source_full_scale_mv, "source waveform full scale"
    ) / (
        _positive_finite(
            connection_config.dc_full_scale_v,
            "QCS DC full scale",
        )
        * 1000.0
    )
    physical_offset = cross_capacitance @ virtual_offset * qcs_scale
    x_coefficients = (
        cross_capacitance[:, x_output_index] * qcs_scale
    )
    y_coefficients = (
        cross_capacitance[:, y_output_index] * qcs_scale
    )
    array_values_by_output = tuple(
        (
            (int(x_values.size) if x_coefficient != 0.0 else 0)
            + (int(y_values.size) if y_coefficient != 0.0 else 0)
        )
        for x_coefficient, y_coefficient in zip(
            x_coefficients,
            y_coefficients,
        )
    )
    if max(array_values_by_output, default=0) > (
        MAX_QCS_HARDWARE_SWEEP_ARRAY_VALUES
    ):
        output_index = int(np.argmax(array_values_by_output))
        raise QcsUnsupportedFeatureError(
            f"QCS DC output {output_names[output_index]!r} requires "
            f"{array_values_by_output[output_index]:,} FPGA sweep-array "
            "values; each M5301 channel supports at most "
            f"{MAX_QCS_HARDWARE_SWEEP_ARRAY_VALUES:,}"
        )
    corner_amplitudes = np.asarray([
        physical_offset + x_coefficients * x + y_coefficients * y
        for x in (x_values[0], x_values[-1])
        for y in (y_values[0], y_values[-1])
    ])
    peak_by_output = np.max(np.abs(corner_amplitudes), axis=0)
    if np.any(peak_by_output > 1.0 + 1e-12):
        output_index = int(np.argmax(peak_by_output))
        peak_voltage_v = (
            float(peak_by_output[output_index])
            * connection_config.dc_full_scale_v
        )
        raise ValueError(
            f"QCS DC output {output_names[output_index]!r} reaches "
            f"{peak_voltage_v:.6g} V after cross-capacitance correction, "
            "exceeding the configured +/-"
            f"{connection_config.dc_full_scale_v:.6g} V full scale"
        )

    x_variable = qcs.Scalar(
        "stability_x_voltage",
        value=float(x_values[0]),
        dtype=float,
    )
    y_variable = qcs.Scalar(
        "stability_y_voltage",
        value=float(y_values[0]),
        dtype=float,
    )
    program = qcs.Program(name="PulseGenerator QCS Stability hardware sweep")
    for output_index, (output_name, channel_name) in enumerate(
        zip(output_names, connection_config.dc_channel_names)
    ):
        amplitude = float(physical_offset[output_index])
        if x_coefficients[output_index] != 0.0:
            amplitude = (
                amplitude
                + x_variable * float(x_coefficients[output_index])
            )
        if y_coefficients[output_index] != 0.0:
            amplitude = (
                amplitude
                + y_variable * float(y_coefficients[output_index])
            )
        program.add_waveform(
            qcs.DCWaveform(
                duration=duration_s,
                envelope=qcs.ConstantEnvelope(),
                amplitude=amplitude,
                name=f"{output_name}_stability_hold",
            ),
            dc_channels[output_index],
            new_layer=output_index == 0,
        )

    segment_name = str(segment.name)
    for pulse in rf_pulses:
        if pulse.at_segment != segment_name:
            raise KeyError(
                f"QCS RF pulse references unknown Stability segment "
                f"{pulse.at_segment!r}"
            )
        if pulse.delay_s + pulse.duration_s > duration_s + 1e-15:
            raise ValueError(
                f"QCS RF pulse on gen_ch {pulse.gen_ch} exceeds the "
                "Stability hold"
            )
        program.add_waveform(
            qcs.RFWaveform(
                duration=pulse.duration_s,
                envelope=_qcs_envelope(qcs, pulse.envelope),
                amplitude=pulse.amplitude,
                rf_frequency=pulse.frequency_hz,
                instantaneous_phase=pulse.phase_rad,
                name=f"stability_rf_{pulse.gen_ch}",
            ),
            _resolve_mapper_channel(
                mapper, connection_config.rf_channel_names[pulse.gen_ch]
            ),
            new_layer=False,
            pre_delay=pulse.delay_s,
        )

    if acquisition.at_segment != segment_name:
        raise KeyError(
            "QCS acquisition references unknown Stability segment "
            f"{acquisition.at_segment!r}"
        )
    if acquisition.pre_delay_s + acquisition_duration_s > duration_s + 1e-15:
        raise ValueError("QCS acquisition exceeds the Stability hold")
    integration_filter = acquisition.integration_filter
    if integration_filter is None:
        integration_filter = qcs.RFWaveform(
            duration=acquisition_duration_s,
            envelope=_qcs_envelope(qcs, acquisition.envelope),
            amplitude=1.0,
            rf_frequency=acquisition.frequency_hz,
            instantaneous_phase=acquisition.phase_rad,
            name="stability_acquisition_filter",
        )
    program.add_acquisition(
        integration_filter=integration_filter,
        channels=acquisition_channels,
        new_layer=False,
        pre_delay=acquisition.pre_delay_s,
    )

    # Program repetition calls prepend their loop. Reverse axis call order so
    # the final native result is (shot, X, Y), with Y varying fastest.
    program.sweep(
        qcs.Array(
            "stability_y_values",
            value=y_values,
            dtype=float,
        ),
        y_variable,
    )
    program.sweep(
        qcs.Array(
            "stability_x_values",
            value=x_values,
            dtype=float,
        ),
        x_variable,
    )
    program.n_shots(repetitions)
    return QcsCompiledHardwareSweep(
        program=program,
        acquisition_channels=acquisition_channels,
        duration_s=duration_s,
        acquisition_duration_s=acquisition_duration_s,
        acquisition_sample_rate_hz=acquisition_sample_rate_hz,
        sweep_shape=sweep_shape,
    )


def execute_qcs_stability_hardware_sweep(
    *,
    connection_config: QcsConnectionConfig,
    sequence: Any,
    repetitions_per_point: int,
    fabric_mhz: float = 300.0,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
    progress_callback: Optional[ProgressCallback] = None,
    qcs_module=None,
    mapper=None,
    executor=None,
    compiled: Optional[QcsCompiledHardwareSweep] = None,
) -> QcsExecutionResult:
    """Execute one complete two-axis Stability scan with one HCL call."""
    qcs = _import_qcs() if qcs_module is None else qcs_module
    if progress_callback is not None:
        progress_callback(0, "Validating QCS hardware sweep")
    if not connection_config.hw_demod:
        raise QcsUnsupportedFeatureError(
            "QCS Stability hardware sweep requires hardware demodulation"
        )
    if mapper is None:
        mapper = load_qcs_channel_mapper(
            connection_config,
            qcs_module=qcs,
        )
    if compiled is None:
        if progress_callback is not None:
            progress_callback(10, "Compiling one native QCS X/Y sweep")
        compiled = compile_qcs_stability_hardware_sweep(
            sequence,
            connection_config=connection_config,
            mapper=mapper,
            repetitions_per_point=repetitions_per_point,
            fabric_mhz=fabric_mhz,
            source_full_scale_mv=source_full_scale_mv,
            rf_pulses=rf_pulses,
            acquisition=acquisition,
            qcs_module=qcs,
        )
    if executor is None:
        executor = build_qcs_executor(
            connection_config,
            mapper,
            qcs_module=qcs,
        )

    if progress_callback is not None:
        progress_callback(
            35,
            (
                "Running one QCS hardware program for "
                f"{compiled.sweep_shape[0]} x "
                f"{compiled.sweep_shape[1]} points"
            ),
        )
    raw_result = _executor_execute(executor, compiled.program)
    values = extract_qcs_acquisition(
        raw_result,
        compiled.acquisition_channels,
    )
    iq = normalize_qcs_hardware_sweep_iq(
        values,
        repetitions_per_point=repetitions_per_point,
        sweep_shape=compiled.sweep_shape,
    )
    ddr_result = FineTuneDdrResult(
        sweep_points=np.asarray(sequence.sweep_points),
        iq=iq,
        sweep_axes=tuple(sequence.sweep_axes),
        sweep_shape=tuple(sequence.sweep_shape),
        cross_capacitance=np.asarray(sequence.cross_capacitance).copy(),
        sample_rate_hz=compiled.acquisition_sample_rate_hz,
        fir_rate_profile="qcs_hardware_demod",
    )
    if progress_callback is not None:
        progress_callback(100, "QCS hardware sweep acquired")

    point_count = int(np.prod(compiled.sweep_shape))
    summary = {
        "backend": "qcs",
        "hardware_sweep": True,
        "hardware_sweep_dimensions": 2,
        "hardware_sweep_shape": list(compiled.sweep_shape),
        "hardware_sweep_points": point_count,
        "software_sweep_points": 0,
        "program_count": 1,
        "repetitions_per_point": int(repetitions_per_point),
        "fabric_mhz": float(fabric_mhz),
        "source_full_scale_mv": float(source_full_scale_mv),
        "qcs_dc_full_scale_v": connection_config.dc_full_scale_v,
        "dc_channel_names": list(connection_config.dc_channel_names),
        "rf_channel_names": {
            str(key): value
            for key, value in connection_config.rf_channel_names.items()
        },
        "acquisition_channel_name": (
            connection_config.acquisition_channel_name
        ),
        "hw_demod": True,
        "sample_rate_hz": compiled.acquisition_sample_rate_hz,
        "acquisition_duration_s": compiled.acquisition_duration_s,
        "requested_sample_count": (
            None if acquisition is None else acquisition.sample_count
        ),
        "iq_shape": list(iq.shape),
    }
    rf_settings = {
        "backend": "qcs",
        "output_details": tuple(
            {
                "gen_ch": pulse.gen_ch,
                "amplitude": pulse.amplitude,
                "frequency_hz": pulse.frequency_hz,
                "duration_s": pulse.duration_s,
            }
            for pulse in rf_pulses
        ),
        "readout_details": {
            "sample_rate_hz": compiled.acquisition_sample_rate_hz,
            "hw_demod": True,
            "frequency_hz": (
                0.0 if acquisition is None else acquisition.frequency_hz
            ),
            "duration_s": compiled.acquisition_duration_s,
            "requested_sample_count": (
                None if acquisition is None else acquisition.sample_count
            ),
            "measurement_representation": "adc",
        },
    }
    return QcsExecutionResult(
        ddr_result=ddr_result,
        programs=(compiled.program,),
        raw_results=(raw_result,),
        program_summary=summary,
        rf_settings=rf_settings,
    )


def execute_qcs_sequence(
    *,
    connection_config: QcsConnectionConfig,
    sequence: Any,
    repetitions_per_sweep: int,
    fabric_mhz: float = 300.0,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
    progress_callback: Optional[ProgressCallback] = None,
    event_callback: Optional[EventCallback] = None,
    qcs_module=None,
    mapper=None,
    executor=None,
) -> QcsExecutionResult:
    """Compile and execute a sequence without writing a database."""
    qcs = _import_qcs() if qcs_module is None else qcs_module
    if acquisition is None:
        raise ValueError("QCS execution requires an acquisition configuration")
    if progress_callback is not None:
        progress_callback(0, "Validating QCS experiment")
    if event_callback is not None:
        event_callback(
            "validation", "started", "Validating QCS capabilities"
        )
    validate_qcs_capabilities(
        connection_config=connection_config,
        sequence=sequence,
        rf_pulses=rf_pulses,
        acquisition=acquisition,
    )
    if event_callback is not None:
        event_callback(
            "validation", "completed", "QCS settings validated"
        )

    if event_callback is not None:
        event_callback(
            "connection", "started", "Loading QCS ChannelMapper"
        )
    if mapper is None:
        mapper = load_qcs_channel_mapper(
            connection_config, qcs_module=qcs
        )
    # Resolve all configured names before compiling any points.
    for name in connection_config.dc_channel_names:
        _resolve_mapper_channel(mapper, name)
    for name in connection_config.rf_channel_names.values():
        _resolve_mapper_channel(mapper, name)
    _resolve_mapper_channel(
        mapper, connection_config.acquisition_channel_name
    )
    if event_callback is not None:
        event_callback(
            "connection", "completed", "QCS ChannelMapper loaded"
        )
    if progress_callback is not None:
        progress_callback(8, "QCS ChannelMapper loaded")

    if event_callback is not None:
        event_callback(
            "program_build", "started", "Compiling QCS software sweep points"
        )
    compiled = compile_qcs_sequence(
        sequence,
        connection_config=connection_config,
        mapper=mapper,
        repetitions_per_sweep=repetitions_per_sweep,
        fabric_mhz=fabric_mhz,
        source_full_scale_mv=source_full_scale_mv,
        rf_pulses=rf_pulses,
        acquisition=acquisition,
        qcs_module=qcs,
        progress_callback=progress_callback,
    )
    if event_callback is not None:
        event_callback(
            "program_build",
            "completed",
            f"Compiled {len(compiled):,} QCS program(s)",
        )

    if executor is None:
        backend = qcs.HclBackend(
            channel_mapper=mapper,
            hw_demod=connection_config.hw_demod,
            init_time=connection_config.init_time_s,
            blocking=connection_config.blocking,
            suppress_rounding_warnings=True,
            keep_progress_bar=False,
        )
        executor = qcs.Executor(backend)

    if event_callback is not None:
        event_callback(
            "acquisition", "started", "Executing QCS programs"
        )
    raw_results = []
    point_iq = []
    executed_sample_rates = []
    point_count = len(compiled)
    effective_acquisition_duration_s = (
        compiled[0].acquisition_duration_s
        if compiled and compiled[0].acquisition_duration_s is not None
        else acquisition.duration_s
    )
    effective_sample_rate_hz = (
        compiled[0].acquisition_sample_rate_hz
        if compiled and compiled[0].acquisition_sample_rate_hz is not None
        else acquisition.sample_rate_hz
    )
    for point_index, item in enumerate(compiled):
        raw_result = _executor_execute(executor, item.program)
        raw_results.append(raw_result)
        if not connection_config.hw_demod:
            executed_sample_rate = _executed_program_sample_rate(
                raw_result, item.acquisition_channels
            )
            if executed_sample_rate is not None:
                executed_sample_rates.append(executed_sample_rate)
        values = extract_qcs_acquisition(
            raw_result,
            item.acquisition_channels,
            prefer_trace=not connection_config.hw_demod,
        )
        point_iq.append(
            normalize_qcs_iq(
                values,
                repetitions_per_sweep=repetitions_per_sweep,
                real_is_i_trace=not connection_config.hw_demod,
            )
        )
        if progress_callback is not None:
            percent = 35 + int(30 * (point_index + 1) / point_count)
            progress_callback(
                percent,
                f"Acquired QCS point {point_index + 1:,}/{point_count:,}",
            )
    if executed_sample_rates:
        reference_rate = executed_sample_rates[0]
        if not np.allclose(
            executed_sample_rates,
            reference_rate,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "QCS programs reported inconsistent digitizer sample rates"
            )
        effective_sample_rate_hz = reference_rate
    sample_shapes = {array.shape[1:] for array in point_iq}
    if len(sample_shapes) != 1:
        raise ValueError(
            "QCS points returned inconsistent acquisition sample shapes"
        )
    iq = np.stack(point_iq, axis=0)
    ddr_result = FineTuneDdrResult(
        sweep_points=np.asarray(sequence.sweep_points),
        iq=iq,
        sweep_axes=tuple(sequence.sweep_axes),
        sweep_shape=tuple(sequence.sweep_shape),
        cross_capacitance=np.asarray(sequence.cross_capacitance).copy(),
        sample_rate_hz=effective_sample_rate_hz,
        fir_rate_profile=(
            "qcs_hardware_demod"
            if connection_config.hw_demod
            else "qcs_trace"
        ),
    )
    if event_callback is not None:
        event_callback(
            "acquisition",
            "completed",
            f"Acquired {point_count:,} QCS point(s)",
        )

    summary = {
        "backend": "qcs",
        "software_sweep_points": point_count,
        "repetitions_per_sweep": int(repetitions_per_sweep),
        "program_count": point_count,
        "fabric_mhz": float(fabric_mhz),
        "source_full_scale_mv": float(source_full_scale_mv),
        "qcs_dc_full_scale_v": connection_config.dc_full_scale_v,
        "dc_channel_names": list(connection_config.dc_channel_names),
        "rf_channel_names": {
            str(key): value
            for key, value in connection_config.rf_channel_names.items()
        },
        "acquisition_channel_name": (
            connection_config.acquisition_channel_name
        ),
        "hw_demod": connection_config.hw_demod,
        "sample_rate_hz": effective_sample_rate_hz,
        "acquisition_duration_s": effective_acquisition_duration_s,
        "requested_sample_count": acquisition.sample_count,
        "iq_shape": list(iq.shape),
    }
    rf_settings = {
        "backend": "qcs",
        "output_details": (),
        "readout_details": {
            "sample_rate_hz": effective_sample_rate_hz,
            "hw_demod": connection_config.hw_demod,
            "frequency_hz": acquisition.frequency_hz,
            "duration_s": effective_acquisition_duration_s,
            "requested_sample_count": acquisition.sample_count,
        },
    }
    return QcsExecutionResult(
        ddr_result=ddr_result,
        programs=tuple(item.program for item in compiled),
        raw_results=tuple(raw_results),
        program_summary=summary,
        rf_settings=rf_settings,
    )


def run_qcs_qcodes_experiment(
    *,
    connection_config: QcsConnectionConfig,
    run_config: QcodesRunConfig,
    sequence: Any,
    repetitions_per_sweep: int,
    fabric_mhz: float = 300.0,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
    gui_settings: Optional[Mapping[str, Any]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    event_callback: Optional[EventCallback] = None,
    qcs_module=None,
    mapper=None,
    executor=None,
) -> StoredQcsExperiment:
    """Execute QCS programs and persist their normalized I/Q arrays."""
    if event_callback is not None:
        event_callback(
            "experiment", "started", "Starting Keysight QCS experiment"
        )
    try:
        execution = execute_qcs_sequence(
            connection_config=connection_config,
            sequence=sequence,
            repetitions_per_sweep=repetitions_per_sweep,
            fabric_mhz=fabric_mhz,
            source_full_scale_mv=source_full_scale_mv,
            rf_pulses=rf_pulses,
            acquisition=acquisition,
            progress_callback=progress_callback,
            event_callback=event_callback,
            qcs_module=qcs_module,
            mapper=mapper,
            executor=executor,
        )
        effective_run_config = replace(
            run_config,
            sample_rate_hz=(
                run_config.sample_rate_hz
                if acquisition is None
                else execution.ddr_result.sample_rate_hz
            ),
        )
        stored_gui_settings = dict(gui_settings or {})
        qick_settings = stored_gui_settings.get("qick", {})
        if not isinstance(qick_settings, Mapping):
            raise TypeError("gui_settings['qick'] must be a mapping")
        qick_settings = dict(qick_settings)
        full_scale_mv = _positive_finite(
            source_full_scale_mv, "source waveform full scale"
        )
        # The sequence and its sweep coordinates are normalized against this
        # scale.  Keep the persisted GUI snapshot authoritative even for
        # direct API callers that omit it or supply a stale value.
        qick_settings["full_scale_mv"] = full_scale_mv
        stored_gui_settings["qick"] = qick_settings
        metadata_mode = normalize_awg_metadata_mode(
            qick_settings.get(
                "awg_metadata_mode", DEFAULT_AWG_METADATA_MODE
            )
        )
        stored_gui_settings["awg_waveform_recipe"] = (
            build_awg_waveform_recipe(
                sequence,
                fabric_mhz=fabric_mhz,
                full_scale_mv=full_scale_mv,
            )
        )
        if metadata_mode == AWG_METADATA_MODE_EXPANDED:
            stored_gui_settings["awg_waveform_vertices"] = (
                build_awg_vertex_metadata(
                    sequence,
                    fabric_mhz=fabric_mhz,
                    full_scale_mv=full_scale_mv,
                )
            )
        if event_callback is not None:
            event_callback(
                "qcodes_save", "started", "Writing QCS result to QCoDeS"
            )
        dataset, row_count = store_experiment_result(
            execution.ddr_result,
            run_config=effective_run_config,
            connection_config=connection_config,
            program_summary=execution.program_summary,
            gui_settings=stored_gui_settings,
            rf_settings=execution.rf_settings,
            backend_name="qcs",
            progress_callback=progress_callback,
            progress_start=65,
            progress_end=99,
        )
        if event_callback is not None:
            event_callback(
                "qcodes_save",
                "completed",
                f"Saved QCoDeS Run {int(dataset.run_id)}",
            )
            event_callback(
                "experiment",
                "completed",
                f"Keysight QCS experiment saved as Run {int(dataset.run_id)}",
            )
        if progress_callback is not None:
            progress_callback(100, "Keysight QCS experiment complete")
        return StoredQcsExperiment(
            run_id=int(dataset.run_id),
            guid=str(dataset.guid),
            database_path=effective_run_config.resolved_database_path,
            row_count=int(row_count),
            dataset=dataset,
            program=execution.programs,
            ddr_result=execution.ddr_result,
            rf_settings=execution.rf_settings,
            programs=execution.programs,
            raw_results=execution.raw_results,
            program_summary=execution.program_summary,
        )
    except Exception as exc:
        if event_callback is not None:
            event_callback("experiment", "failed", str(exc))
        raise


__all__ = [
    "MAX_QCS_HARDWARE_SWEEP_ARRAY_VALUES",
    "MAX_QCS_STABILITY_GRID_POINTS",
    "MAX_QCS_STABILITY_RESULT_VALUES",
    "MAX_QCS_SOFTWARE_SWEEP_POINTS",
    "QCS_M5200_INTEGRATION_BLOCK_SAMPLES",
    "QCS_M5200_SAMPLE_RATE_HZ",
    "QcsAcquisitionConfig",
    "QcsCompiledHardwareSweep",
    "QcsCompiledPoint",
    "QcsConnectionConfig",
    "QcsExecutionResult",
    "QcsRfPulseConfig",
    "QcsUnsupportedFeatureError",
    "StoredQcsExperiment",
    "build_qcs_executor",
    "compile_qcs_point",
    "compile_qcs_sequence",
    "compile_qcs_stability_hardware_sweep",
    "execute_qcs_sequence",
    "execute_qcs_stability_hardware_sweep",
    "extract_qcs_acquisition",
    "load_qcs_channel_mapper",
    "normalize_qcs_hardware_sweep_iq",
    "normalize_qcs_iq",
    "run_qcs_qcodes_experiment",
    "validate_qcs_capabilities",
]
