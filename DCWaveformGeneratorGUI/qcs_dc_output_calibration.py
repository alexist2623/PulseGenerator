"""M5301A DC-output calibration against a 1-Mohm oscilloscope input.

The M5301A path is calibrated as a connector-voltage scale error, not as a
QICK DC loopback response.  For commanded connector voltage ``x`` and the
oscilloscope measurement ``y`` the only supported model is::

    y = A * x

The fit is deliberately constrained through the origin.  A DC offset seen by
the scope is retained as a diagnostic residual and is never hidden in the
compensation.  Requested connector voltage is therefore programmed as
``target / A`` and remains subject to the configured nominal M5301A full
scale.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


QCS_M5301_DC_OUTPUT_CALIBRATION_SCHEMA = (
    "qstl-qcs-m5301a-dc-output-calibration-v1"
)
QCS_M5301_INSTRUMENT_MODEL = "M5301A"
QCS_M5301_SCOPE_INPUT_IMPEDANCE_OHMS = 1_000_000.0
QCS_M5301_FABRIC_HZ = 300.0e6
QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES = 4
QCS_M5301_WAVEFORM_GRANULARITY_FABRIC_CYCLES = 2
DEFAULT_QCS_M5301_NOMINAL_FULL_SCALE_V = 5.0
MIN_QCS_M5301_DC_CALIBRATION_R_SQUARED = 0.99

ProgressCallback = Callable[[int, str], None]


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _emit_progress(
    callback: Optional[ProgressCallback], percent: int, message: str
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(percent))), str(message))


def _is_one_megohm(value: Any) -> bool:
    try:
        impedance = float(value)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(impedance)
        and np.isclose(
            impedance,
            QCS_M5301_SCOPE_INPUT_IMPEDANCE_OHMS,
            rtol=0.0,
            atol=1.0,
        )
    )


def _validate_bipolar_commands(values: Any) -> np.ndarray:
    commands = np.asarray(values, dtype=np.float64).reshape(-1)
    if commands.size < 3:
        raise ValueError("M5301A DC calibration requires at least three points")
    if not np.all(np.isfinite(commands)):
        raise ValueError("M5301A DC calibration commands must be finite")
    if np.unique(commands).size < 3:
        raise ValueError("M5301A DC calibration requires distinct command points")
    if not np.any(commands < 0.0) or not np.any(commands > 0.0):
        raise ValueError(
            "M5301A DC calibration points must be bipolar (negative and positive)"
        )
    if not np.any(commands == 0.0):
        raise ValueError(
            "M5301A DC calibration requires an explicit zero-voltage command point"
        )
    positive = np.sort(commands[commands > 0.0])
    negative_magnitudes = np.sort(-commands[commands < 0.0])
    scale = max(1.0, float(np.max(np.abs(commands))))
    if positive.size != negative_magnitudes.size or not np.allclose(
        positive,
        negative_magnitudes,
        rtol=1.0e-12,
        atol=1.0e-12 * scale,
    ):
        raise ValueError(
            "M5301A DC calibration command points must be symmetric about zero "
            "with matching negative and positive voltage magnitudes"
        )
    return commands


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _json_mapping(value: Any) -> Mapping[str, Any]:
    if value in (None, ""):
        return {}
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, Mapping) else {}


@dataclass(frozen=True)
class QcsDcOscilloscopeConfig:
    """DC-voltage measurement settings for a Keysight oscilloscope."""

    visa_resource: str = ""
    channel: int = 1
    input_impedance_ohm: float = QCS_M5301_SCOPE_INPUT_IMPEDANCE_OHMS
    average_count: int = 8
    settle_seconds: float = 0.05
    sample_interval_seconds: float = 0.01
    timeout_ms: int = 10_000

    def __post_init__(self) -> None:
        _integer(self.channel, "oscilloscope channel", 1)
        if not _is_one_megohm(self.input_impedance_ohm):
            raise ValueError(
                "M5301A DC calibration requires the oscilloscope input to be "
                "exactly 1 Mohm"
            )
        _integer(self.average_count, "oscilloscope average_count", 1)
        if _finite(self.settle_seconds, "oscilloscope settle_seconds") < 0.0:
            raise ValueError("oscilloscope settle_seconds must be nonnegative")
        if (
            _finite(
                self.sample_interval_seconds,
                "oscilloscope sample_interval_seconds",
            )
            < 0.0
        ):
            raise ValueError(
                "oscilloscope sample_interval_seconds must be nonnegative"
            )
        _integer(self.timeout_ms, "oscilloscope timeout_ms", 1)


@dataclass(frozen=True)
class QcsM5301DcCalibrationConfig:
    """One M5301A connector and its bipolar oscilloscope calibration sweep."""

    database_path: str
    chassis: int
    slot: int
    channel: int
    module_serial: str = ""
    virtual_channel_name: str = "m5301_dc_calibration"
    voltage_start_v: float = -4.0
    voltage_stop_v: float = 4.0
    voltage_points: int = 9
    nominal_full_scale_v: float = DEFAULT_QCS_M5301_NOMINAL_FULL_SCALE_V
    waveform_duration_s: float = 1.0e-6
    fabric_hz: float = QCS_M5301_FABRIC_HZ
    minimum_r_squared: float = MIN_QCS_M5301_DC_CALIBRATION_R_SQUARED
    experiment_name: str = "QCS M5301A DC output calibration"
    sample_name: str = ""
    notes: str = ""
    oscilloscope: QcsDcOscilloscopeConfig = QcsDcOscilloscopeConfig()

    def __post_init__(self) -> None:
        if not str(self.database_path).strip():
            raise ValueError("calibration database path must not be empty")
        _integer(self.chassis, "chassis", 1)
        _integer(self.slot, "slot", 1)
        _integer(self.channel, "channel", 1)
        if not str(self.virtual_channel_name).strip():
            raise ValueError("virtual_channel_name must not be empty")
        start = _finite(self.voltage_start_v, "voltage_start_v")
        stop = _finite(self.voltage_stop_v, "voltage_stop_v")
        points = _integer(self.voltage_points, "voltage_points", 3)
        full_scale = _finite(
            self.nominal_full_scale_v,
            "nominal_full_scale_v",
            positive=True,
        )
        commands = np.linspace(start, stop, points, dtype=float)
        _validate_bipolar_commands(commands)
        if np.max(np.abs(commands)) > full_scale + 1.0e-12:
            raise ValueError(
                "M5301A DC calibration commands must fit within the nominal "
                f"+/-{full_scale:g} V full scale"
            )
        duration = _finite(
            self.waveform_duration_s,
            "waveform_duration_s",
            positive=True,
        )
        fabric_hz = _finite(self.fabric_hz, "fabric_hz", positive=True)
        cycles_float = duration * fabric_hz
        cycles = int(round(cycles_float))
        if not np.isclose(cycles_float, cycles, rtol=0.0, atol=1.0e-9):
            raise ValueError(
                "M5301A calibration waveform duration must be an integer "
                "number of QCS fabric cycles"
            )
        if cycles < QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES:
            raise ValueError(
                "M5301A calibration waveform duration must be at least "
                f"{QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES} QCS fabric cycles"
            )
        if cycles % QCS_M5301_WAVEFORM_GRANULARITY_FABRIC_CYCLES:
            raise ValueError(
                "M5301A calibration waveform duration must be a multiple of "
                "2 QCS fabric cycles (16 waveform samples)"
            )
        minimum_r_squared = _finite(
            self.minimum_r_squared,
            "minimum_r_squared",
        )
        if not -1.0 <= minimum_r_squared <= 1.0:
            raise ValueError("minimum_r_squared must be in [-1, 1]")
        if not str(self.experiment_name).strip():
            raise ValueError("experiment_name must not be empty")
        if not isinstance(self.oscilloscope, QcsDcOscilloscopeConfig):
            object.__setattr__(
                self,
                "oscilloscope",
                QcsDcOscilloscopeConfig(**dict(self.oscilloscope)),
            )

    @property
    def commanded_voltages_v(self) -> np.ndarray:
        return np.linspace(
            float(self.voltage_start_v),
            float(self.voltage_stop_v),
            int(self.voltage_points),
            dtype=float,
        )

    @property
    def relative_amplitudes(self) -> np.ndarray:
        return self.commanded_voltages_v / float(self.nominal_full_scale_v)

    @property
    def physical_address(self) -> tuple[int, int, int]:
        return int(self.chassis), int(self.slot), int(self.channel)


@dataclass(frozen=True)
class QcsM5301DcOutputCalibration:
    """Origin-constrained M5301A connector-voltage scale calibration."""

    database_path: Path
    run_id: int
    chassis: int
    slot: int
    channel: int
    module_serial: str
    scope_resource: str
    scope_channel: int
    scope_input_impedance_ohm: float
    nominal_full_scale_v: float
    gain_a: float
    rmse_v: float
    max_abs_residual_v: float
    r_squared: float
    commanded_min_v: float
    commanded_max_v: float
    measured_min_v: float
    measured_max_v: float
    point_count: int
    measured_zero_v: Optional[float] = None

    def __post_init__(self) -> None:
        if not _is_one_megohm(self.scope_input_impedance_ohm):
            raise ValueError(
                "M5301A DC calibration is valid only for a 1 Mohm scope input"
            )
        gain = _finite(self.gain_a, "M5301A calibration gain A")
        if gain <= 0.0:
            raise ValueError(
                "M5301A calibration gain A must be positive; a negative "
                "transfer indicates the wrong path or reversed polarity"
            )
        _finite(
            self.nominal_full_scale_v,
            "nominal_full_scale_v",
            positive=True,
        )

    @classmethod
    def fit(
        cls,
        commanded_voltages_v: Any,
        measured_voltages_v: Any,
        *,
        database_path: Any = "",
        run_id: int = 0,
        chassis: int,
        slot: int,
        channel: int,
        module_serial: str = "",
        scope_resource: str = "",
        scope_channel: int = 1,
        scope_input_impedance_ohm: float = (
            QCS_M5301_SCOPE_INPUT_IMPEDANCE_OHMS
        ),
        nominal_full_scale_v: float = DEFAULT_QCS_M5301_NOMINAL_FULL_SCALE_V,
    ) -> "QcsM5301DcOutputCalibration":
        commands = _validate_bipolar_commands(commanded_voltages_v)
        measured = np.asarray(measured_voltages_v, dtype=np.float64).reshape(-1)
        if measured.shape != commands.shape:
            raise ValueError(
                "commanded and measured M5301A voltage arrays must have equal length"
            )
        if not np.all(np.isfinite(measured)):
            raise ValueError("measured M5301A voltages must be finite")
        denominator = float(np.dot(commands, commands))
        if denominator <= 0.0:
            raise ValueError("M5301A DC calibration requires nonzero commands")

        # Intentionally no intercept: y_measured = A * x_commanded.
        gain_a = float(np.dot(commands, measured) / denominator)
        if not np.isfinite(gain_a) or gain_a <= 0.0:
            raise ValueError(
                "M5301A DC calibration gain A must be positive; verify the "
                "selected output, scope path, and polarity"
            )
        predicted = gain_a * commands
        residuals = measured - predicted
        residual_sum = float(np.dot(residuals, residuals))
        centered = measured - float(np.mean(measured))
        total_sum = float(np.dot(centered, centered))
        r_squared = (
            1.0
            if total_sum == 0.0 and residual_sum == 0.0
            else (
                0.0
                if total_sum == 0.0
                else 1.0 - residual_sum / total_sum
            )
        )
        zero_indices = np.flatnonzero(np.isclose(commands, 0.0, atol=1.0e-15))
        measured_zero = (
            float(np.mean(measured[zero_indices]))
            if zero_indices.size
            else None
        )
        return cls(
            database_path=Path(database_path).expanduser(),
            run_id=int(run_id),
            chassis=_integer(chassis, "chassis", 1),
            slot=_integer(slot, "slot", 1),
            channel=_integer(channel, "channel", 1),
            module_serial=str(module_serial).strip(),
            scope_resource=str(scope_resource).strip(),
            scope_channel=_integer(scope_channel, "scope_channel", 1),
            scope_input_impedance_ohm=float(scope_input_impedance_ohm),
            nominal_full_scale_v=_finite(
                nominal_full_scale_v,
                "nominal_full_scale_v",
                positive=True,
            ),
            gain_a=gain_a,
            rmse_v=float(np.sqrt(np.mean(np.square(residuals)))),
            max_abs_residual_v=float(np.max(np.abs(residuals))),
            r_squared=float(r_squared),
            commanded_min_v=float(np.min(commands)),
            commanded_max_v=float(np.max(commands)),
            measured_min_v=float(np.min(measured)),
            measured_max_v=float(np.max(measured)),
            point_count=int(commands.size),
            measured_zero_v=measured_zero,
        )

    @property
    def corrected_maximum_abs_voltage_v(self) -> float:
        """Largest calibrated connector target reachable within full scale."""

        return abs(float(self.gain_a)) * float(self.nominal_full_scale_v)

    def command_voltage_for_target(self, target_voltage_v: Any) -> Any:
        """Return the uncompensated M5301A command required for ``target``."""

        targets = np.asarray(target_voltage_v, dtype=np.float64)
        if not np.all(np.isfinite(targets)):
            raise ValueError("target M5301A connector voltages must be finite")
        commands = targets / float(self.gain_a)
        limit = float(self.nominal_full_scale_v)
        if np.any(np.abs(commands) > limit + 1.0e-12):
            requested = float(np.max(np.abs(targets)))
            raise ValueError(
                f"requested M5301A target {requested:.9g} V exceeds the "
                f"calibrated reachable +/-{self.corrected_maximum_abs_voltage_v:.9g} "
                f"V; compensation would require more than +/-{limit:.9g} V command"
            )
        if targets.ndim == 0:
            return float(commands)
        return commands

    def relative_amplitude_for_target(self, target_voltage_v: Any) -> Any:
        """Map connector target voltage to the QCS relative DC amplitude."""

        commands = np.asarray(
            self.command_voltage_for_target(target_voltage_v),
            dtype=np.float64,
        )
        relative = commands / float(self.nominal_full_scale_v)
        if relative.ndim == 0:
            return float(relative)
        return relative

    def as_dict(self) -> Mapping[str, Any]:
        values = asdict(self)
        values["database_path"] = str(self.database_path)
        values["schema"] = QCS_M5301_DC_OUTPUT_CALIBRATION_SCHEMA
        values["instrument_model"] = QCS_M5301_INSTRUMENT_MODEL
        values["corrected_maximum_abs_voltage_v"] = (
            self.corrected_maximum_abs_voltage_v
        )
        values["formula"] = (
            "measured_voltage_v = gain_a * commanded_voltage_v; "
            "commanded_voltage_v = target_voltage_v / gain_a; no intercept"
        )
        return values


@dataclass
class StoredQcsM5301DcCalibrationRun:
    run_id: int
    guid: str
    database_path: Path
    row_count: int
    calibration: QcsM5301DcOutputCalibration
    dataset: Any = None
    result: Optional[Mapping[str, Any]] = None


class KeysightDcVoltageMeter:
    """Minimal PyVISA DC voltage reader with verified 1-Mohm termination."""

    def __init__(self, config: QcsDcOscilloscopeConfig):
        self.config = config
        self.resource_manager = None
        self.instrument = None
        self.idn = ""
        self.input_impedance_ohm: Optional[float] = None

    def __enter__(self) -> "KeysightDcVoltageMeter":
        if not self.config.visa_resource.strip():
            raise ValueError("oscilloscope VISA resource must not be empty")
        try:
            import pyvisa
        except ImportError as exc:
            raise RuntimeError(
                "PyVISA is required for M5301A DC calibration; install "
                "PyVISA==1.16.2"
            ) from exc
        self.resource_manager = pyvisa.ResourceManager()
        try:
            self.instrument = self.resource_manager.open_resource(
                self.config.visa_resource
            )
            self.instrument.timeout = int(self.config.timeout_ms)
            self.instrument.write_termination = "\n"
            self.instrument.read_termination = "\n"
            self.idn = str(self.instrument.query("*IDN?")).strip()
            self.ensure_one_megohm()
            self.instrument.write(
                f":CHANnel{int(self.config.channel)}:COUPling DC"
            )
            self.instrument.write(":RUN")
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *_exc: Any) -> None:
        if self.instrument is not None:
            self.instrument.close()
            self.instrument = None
        if self.resource_manager is not None:
            self.resource_manager.close()
            self.resource_manager = None

    @staticmethod
    def _parse_impedance(response: Any) -> float:
        text = str(response).strip().upper().replace(" ", "")
        if text in {"ONEM", "ONEMEG", "1MEG", "1M", "ONEMEGOHM"}:
            return QCS_M5301_SCOPE_INPUT_IMPEDANCE_OHMS
        try:
            return float(text)
        except ValueError as exc:
            raise RuntimeError(
                f"oscilloscope returned unknown input impedance {response!r}"
            ) from exc

    def ensure_one_megohm(self) -> float:
        if self.instrument is None:
            raise RuntimeError("oscilloscope is not open")
        channel = int(self.config.channel)
        self.instrument.write(f":CHANnel{channel}:IMPedance ONEMeg")
        response = self.instrument.query(f":CHANnel{channel}:IMPedance?")
        impedance = self._parse_impedance(response)
        if not _is_one_megohm(impedance):
            raise RuntimeError(
                "oscilloscope did not accept 1 Mohm input mode; measured "
                f"termination is {impedance:g} ohm. Calibration was stopped."
            )
        self.input_impedance_ohm = float(impedance)
        return self.input_impedance_ohm

    def measure_voltage(self) -> float:
        if self.instrument is None:
            raise RuntimeError("oscilloscope is not open")
        if not _is_one_megohm(self.input_impedance_ohm):
            self.ensure_one_megohm()
        if self.config.settle_seconds:
            time.sleep(float(self.config.settle_seconds))
        channel = int(self.config.channel)
        readings = []
        for _index in range(int(self.config.average_count)):
            response = self.instrument.query(
                f":MEASure:VAVerage? DISPlay,CHANnel{channel}"
            )
            value = float(str(response).strip())
            if not np.isfinite(value):
                raise RuntimeError(
                    f"oscilloscope returned invalid DC voltage {response!r}"
                )
            readings.append(value)
            if self.config.sample_interval_seconds:
                time.sleep(float(self.config.sample_interval_seconds))
        return float(np.mean(np.asarray(readings, dtype=float)))


def _load_qcs() -> Any:
    try:
        import keysight.qcs as qcs
    except ImportError as exc:
        raise RuntimeError(
            "keysight-qcs==2.5.5 is required for M5301A DC calibration"
        ) from exc
    return qcs


def _mapper_channels(mapper: Any) -> tuple[Any, ...]:
    channels = getattr(mapper, "channels", None)
    if channels is None:
        raise TypeError("QCS ChannelMapper does not expose virtual channels")
    return tuple(channels)


def _resolve_mapper_channel(mapper: Any, name: str) -> Any:
    matches = [
        channel
        for channel in _mapper_channels(mapper)
        if str(getattr(channel, "name", "")).strip() == str(name).strip()
    ]
    if len(matches) != 1:
        raise KeyError(
            f"QCS mapper must contain exactly one virtual channel {name!r}"
        )
    return matches[0]


def _reject_nonzero_mapped_offset(
    mapper: Any,
    channel: Any,
    *,
    name: str,
) -> None:
    """Reject a mapper offset that would invalidate the origin-only fit.

    Real QCS channel mappers expose their physical channels through
    ``get_physical_channels``.  Lightweight injected adapters may not, so the
    check is deliberately conditional on that native introspection API.
    """

    get_physical_channels = getattr(mapper, "get_physical_channels", None)
    if not callable(get_physical_channels):
        return
    physical_channels = tuple(get_physical_channels(channel))
    if len(physical_channels) != 1:
        raise ValueError(
            f"QCS DC calibration virtual channel {name!r} must map to exactly "
            f"one physical connector; found {len(physical_channels)}"
        )
    settings = getattr(physical_channels[0], "settings", None)
    offset = None if settings is None else getattr(settings, "offset", None)
    if offset is None:
        return
    offset_value = getattr(offset, "value", offset)
    if offset_value is None:
        return
    mapped_offset = _finite(offset_value, "mapped M5301A physical-channel offset")
    if not np.isclose(mapped_offset, 0.0, rtol=0.0, atol=1.0e-15):
        raise ValueError(
            "M5301A DC calibration requires the mapped physical-channel offset "
            f"to be zero; virtual channel {name!r} has offset "
            f"{mapped_offset:.9g}. Clear the ChannelMapper offset before "
            "calibrating y = A*x."
        )


def _create_mapper(qcs: Any, config: QcsM5301DcCalibrationConfig) -> tuple[Any, Any]:
    channel = qcs.Channels(range(1), str(config.virtual_channel_name))
    mapper = qcs.ChannelMapper()
    mapper.add_channel_mapping(
        channels=channel,
        addresses=[config.physical_address],
        instrument_types=qcs.InstrumentEnum.M5301AWG,
    )
    return mapper, channel


def _create_executor(qcs: Any, mapper: Any) -> Any:
    backend = qcs.HclBackend(
        channel_mapper=mapper,
        blocking=True,
        suppress_rounding_warnings=True,
        keep_progress_bar=False,
    )
    return qcs.Executor(backend)


def build_qcs_m5301_dc_level_program(
    qcs: Any,
    *,
    channel: Any,
    commanded_voltage_v: float,
    nominal_full_scale_v: float,
    duration_s: float,
    name: str = "M5301A DC calibration level",
) -> Any:
    """Build one short legal, constant M5301A connector-voltage program."""

    commanded = _finite(commanded_voltage_v, "commanded_voltage_v")
    full_scale = _finite(
        nominal_full_scale_v,
        "nominal_full_scale_v",
        positive=True,
    )
    relative_amplitude = commanded / full_scale
    if abs(relative_amplitude) > 1.0 + 1.0e-12:
        raise ValueError(
            "M5301A commanded voltage exceeds the configured nominal full scale"
        )
    program = qcs.Program(name=str(name))
    waveform = qcs.DCWaveform(
        duration=float(duration_s),
        envelope=qcs.ConstantEnvelope(),
        amplitude=float(relative_amplitude),
        name="m5301_dc_calibration_level",
    )
    program.add_waveform(waveform, channel, new_layer=True)
    program.n_shots(1)
    return program


def _execute(executor: Any, program: Any) -> Any:
    if hasattr(executor, "execute"):
        return executor.execute(program)
    if callable(executor):
        return executor(program)
    raise TypeError("QCS executor must be callable or expose execute(program)")


def _confirm_scope_one_megohm(scope: Any) -> float:
    confirm = getattr(scope, "ensure_one_megohm", None)
    if callable(confirm):
        impedance = confirm()
    else:
        impedance = getattr(scope, "input_impedance_ohm", None)
    if not _is_one_megohm(impedance):
        raise RuntimeError(
            "scope adapter must explicitly confirm a 1 Mohm input before "
            "M5301A DC calibration"
        )
    return float(impedance)


def _invoke_program_callback(
    callback: Optional[Callable[..., Any]],
    *,
    program: Any,
    commanded_voltage_v: float,
    relative_amplitude: float,
    is_reset: bool,
) -> Any:
    if callback is None:
        return program
    replacement = callback(
        program=program,
        commanded_voltage_v=float(commanded_voltage_v),
        relative_amplitude=float(relative_amplitude),
        is_reset=bool(is_reset),
    )
    return program if replacement is None else replacement


def _storage_helpers() -> tuple[Callable[..., Any], ...]:
    try:
        from .qick_qcodes_experiment import (
            _checkpoint_sqlite_database,
            _prepare_local_database,
            _publish_local_database,
        )
    except ImportError:
        from qick_qcodes_experiment import (
            _checkpoint_sqlite_database,
            _prepare_local_database,
            _publish_local_database,
        )
    return (
        _checkpoint_sqlite_database,
        _prepare_local_database,
        _publish_local_database,
    )


def store_qcs_m5301_dc_output_calibration(
    config: QcsM5301DcCalibrationConfig,
    commanded_voltages_v: Sequence[float],
    measured_voltages_v: Sequence[float],
    calibration: QcsM5301DcOutputCalibration,
    *,
    scope_identity: str = "",
) -> StoredQcsM5301DcCalibrationRun:
    """Store a distinct, address-tagged M5301A scope calibration in QCoDeS."""

    try:
        from qcodes import (
            Measurement,
            Parameter,
            Station,
            initialise_or_create_database_at,
            load_by_guid,
            load_or_create_experiment,
        )
    except ImportError as exc:
        raise RuntimeError(
            "QCoDeS==0.58.0 is required for M5301A DC calibration storage"
        ) from exc
    checkpoint_database, prepare_local_database, publish_local_database = (
        _storage_helpers()
    )
    commands = _validate_bipolar_commands(commanded_voltages_v)
    measured = np.asarray(measured_voltages_v, dtype=float).reshape(-1)
    if measured.shape != commands.shape or not np.all(np.isfinite(measured)):
        raise ValueError("stored M5301A commanded/measured arrays are invalid")
    predicted = float(calibration.gain_a) * commands
    residual = measured - predicted

    database_path = Path(config.database_path).expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    staging, local_path = prepare_local_database(database_path)
    try:
        initialise_or_create_database_at(str(local_path))
        sample_name = str(config.sample_name).strip() or (
            f"M5301A_{config.chassis}_{config.slot}_{config.channel}"
        )
        experiment = load_or_create_experiment(config.experiment_name, sample_name)
        measurement = Measurement(exp=experiment, station=Station())
        command_parameter = Parameter(
            "m5301_commanded_voltage_v",
            label="M5301A commanded connector voltage",
            unit="V",
        )
        measured_parameter = Parameter(
            "scope_measured_voltage_v",
            label="1 Mohm oscilloscope voltage",
            unit="V",
        )
        predicted_parameter = Parameter(
            "origin_fit_voltage_v",
            label="Origin-constrained fitted voltage",
            unit="V",
        )
        residual_parameter = Parameter(
            "origin_fit_residual_v",
            label="Origin-constrained fit residual",
            unit="V",
        )
        measurement.register_parameter(command_parameter)
        for parameter in (
            measured_parameter,
            predicted_parameter,
            residual_parameter,
        ):
            measurement.register_parameter(parameter, setpoints=(command_parameter,))

        config_metadata = {
            "schema": QCS_M5301_DC_OUTPUT_CALIBRATION_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "m5301a_dc_output_voltage_scale",
            "instrument_model": QCS_M5301_INSTRUMENT_MODEL,
            "physical_address": {
                "chassis": int(config.chassis),
                "slot": int(config.slot),
                "channel": int(config.channel),
            },
            "module_serial": str(config.module_serial).strip(),
            "scope": {
                "visa_resource": str(config.oscilloscope.visa_resource),
                "channel": int(config.oscilloscope.channel),
                "input_impedance_ohm": (
                    QCS_M5301_SCOPE_INPUT_IMPEDANCE_OHMS
                ),
                "identity": str(scope_identity),
            },
            "nominal_full_scale_v": float(config.nominal_full_scale_v),
            "configuration": asdict(config),
            "fit_model": "y_measured_volts = A * x_commanded_volts",
            "fit_intercept_fixed_v": 0.0,
        }
        result_metadata = dict(calibration.as_dict())
        result_metadata["database_path"] = str(database_path)
        with measurement.run(
            write_in_background=False,
            in_memory_cache=False,
        ) as datasaver:
            dataset = datasaver.dataset
            dataset.add_metadata(
                "QCS_M5301_DC_Output_Calibration_Config",
                json.dumps(config_metadata, sort_keys=True),
            )
            dataset.add_metadata(
                "QCS_M5301_DC_Output_Calibration_Result",
                json.dumps(result_metadata, sort_keys=True),
            )
            if config.notes:
                dataset.add_metadata("calibration_notes", str(config.notes))
            for index, commanded in enumerate(commands):
                datasaver.add_result(
                    (command_parameter, float(commanded)),
                    (measured_parameter, float(measured[index])),
                    (predicted_parameter, float(predicted[index])),
                    (residual_parameter, float(residual[index])),
                )
            datasaver.flush_data_to_database()
        guid = str(dataset.guid)
        run_id = int(dataset.run_id)
        dataset.conn.close()
        checkpoint_database(local_path)
        publish_local_database(local_path, database_path)
        initialise_or_create_database_at(str(database_path))
        final_dataset = load_by_guid(guid)
        final_calibration = replace(
            calibration,
            database_path=database_path,
            run_id=run_id,
        )
        return StoredQcsM5301DcCalibrationRun(
            run_id=run_id,
            guid=guid,
            database_path=database_path,
            row_count=int(commands.size),
            calibration=final_calibration,
            dataset=final_dataset,
            result={
                "commanded_voltages_v": commands.tolist(),
                "measured_voltages_v": measured.tolist(),
                "calibration": dict(final_calibration.as_dict()),
            },
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _metadata_field(row: sqlite3.Row, name: str) -> Mapping[str, Any]:
    fields = {str(key).lower(): str(key) for key in row.keys()}
    key = fields.get(str(name).lower())
    return _json_mapping(row[key]) if key else {}


def _reject_exact_incompatible_run(
    row: sqlite3.Row,
    *,
    selected_run_id: int,
    table_columns: set[str],
) -> None:
    qick_metadata = _metadata_field(row, "DC_Voltage_Calibration_Config")
    qick_schema = str(qick_metadata.get("schema", ""))
    if qick_schema.startswith("qstl-qick-dc-voltage-calibration") or {
        "dc_voltage_mv",
        "mean_adc",
    }.issubset(table_columns):
        raise ValueError(
            f"QCoDeS Run {selected_run_id} is a QICK DC_Out-to-DC_In "
            "loopback calibration, not an M5301A 1 Mohm oscilloscope "
            "calibration"
        )
    raise ValueError(
        f"QCoDeS Run {selected_run_id} is not a compatible "
        f"{QCS_M5301_DC_OUTPUT_CALIBRATION_SCHEMA} run"
    )


def load_qcs_m5301_dc_output_calibration(
    database_path: Any,
    *,
    chassis: int,
    slot: int,
    channel: int,
    run_id: Optional[int] = None,
    module_serial: Optional[str] = None,
) -> QcsM5301DcOutputCalibration:
    """Load an exact run or latest strict M5301A physical-address match."""

    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"calibration database does not exist: {path}")
    address = (
        _integer(chassis, "chassis", 1),
        _integer(slot, "slot", 1),
        _integer(channel, "channel", 1),
    )
    selected_run_id = (
        None if run_id in (None, 0) else _integer(run_id, "run_id", 1)
    )
    requested_serial = (
        None if module_serial is None else str(module_serial).strip()
    )

    matches: list[
        tuple[int, Mapping[str, Any], Mapping[str, Any], np.ndarray]
    ] = []
    exact_row_seen = False
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT r.*, e.sample_name FROM runs r "
            "JOIN experiments e ON e.exp_id = r.exp_id ORDER BY r.run_id DESC"
        ).fetchall()
        for row in rows:
            candidate_run_id = int(row["run_id"])
            if selected_run_id is not None and candidate_run_id != selected_run_id:
                continue
            if selected_run_id is not None:
                exact_row_seen = True
            table_name = str(row["result_table_name"])
            table_columns = {
                str(item[1])
                for item in connection.execute(
                    f"PRAGMA table_info({_quote_identifier(table_name)})"
                )
            }
            required = {
                "m5301_commanded_voltage_v",
                "scope_measured_voltage_v",
            }
            config_metadata = _metadata_field(
                row, "QCS_M5301_DC_Output_Calibration_Config"
            )
            result_metadata = _metadata_field(
                row, "QCS_M5301_DC_Output_Calibration_Result"
            )
            if (
                not required.issubset(table_columns)
                or config_metadata.get("schema")
                != QCS_M5301_DC_OUTPUT_CALIBRATION_SCHEMA
            ):
                if selected_run_id is not None:
                    _reject_exact_incompatible_run(
                        row,
                        selected_run_id=selected_run_id,
                        table_columns=table_columns,
                    )
                continue
            if (
                str(config_metadata.get("instrument_model", "")).upper()
                != QCS_M5301_INSTRUMENT_MODEL
            ):
                if selected_run_id is not None:
                    raise ValueError(
                        f"QCoDeS Run {selected_run_id} is not tagged M5301A"
                    )
                continue
            physical = config_metadata.get("physical_address", {})
            try:
                candidate_address = (
                    int(physical["chassis"]),
                    int(physical["slot"]),
                    int(physical["channel"]),
                )
            except (KeyError, TypeError, ValueError):
                candidate_address = (-1, -1, -1)
            if candidate_address != address:
                if selected_run_id is not None:
                    raise ValueError(
                        f"QCoDeS Run {selected_run_id} is for M5301A address "
                        f"{candidate_address}, not {address}"
                    )
                continue
            candidate_serial = str(config_metadata.get("module_serial", "")).strip()
            if requested_serial is not None and candidate_serial != requested_serial:
                if selected_run_id is not None:
                    raise ValueError(
                        f"QCoDeS Run {selected_run_id} module serial "
                        f"{candidate_serial!r} does not match {requested_serial!r}"
                    )
                continue
            scope_metadata = config_metadata.get("scope", {})
            impedance = scope_metadata.get("input_impedance_ohm")
            if not _is_one_megohm(impedance):
                if selected_run_id is not None:
                    raise ValueError(
                        f"QCoDeS Run {selected_run_id} was not measured with "
                        "a verified 1 Mohm scope input"
                    )
                continue
            if not result_metadata or result_metadata.get("schema") != (
                QCS_M5301_DC_OUTPUT_CALIBRATION_SCHEMA
            ):
                if selected_run_id is not None:
                    raise ValueError(
                        f"QCoDeS Run {selected_run_id} has no strict M5301A "
                        "calibration result metadata"
                    )
                continue
            records = connection.execute(
                f"SELECT m5301_commanded_voltage_v, scope_measured_voltage_v "
                f"FROM {_quote_identifier(table_name)} WHERE "
                "m5301_commanded_voltage_v IS NOT NULL AND "
                "scope_measured_voltage_v IS NOT NULL "
                "ORDER BY m5301_commanded_voltage_v"
            ).fetchall()
            if len(records) < 3:
                if selected_run_id is not None:
                    raise ValueError(
                        f"QCoDeS Run {selected_run_id} has too few DC points"
                    )
                continue
            values = np.asarray([tuple(record) for record in records], dtype=float)
            matches.append(
                (candidate_run_id, config_metadata, result_metadata, values)
            )

    if not matches:
        if selected_run_id is not None and not exact_row_seen:
            raise LookupError(f"QCoDeS Run {selected_run_id} does not exist in {path}")
        selector = (
            f"Run {selected_run_id}"
            if selected_run_id is not None
            else "a compatible run"
        )
        raise LookupError(
            f"could not find {selector} for M5301A address {address} in {path}"
        )
    candidate_run_id, metadata, stored_result, values = max(
        matches, key=lambda item: item[0]
    )
    scope_metadata = metadata["scope"]
    calibration = QcsM5301DcOutputCalibration.fit(
        values[:, 0],
        values[:, 1],
        database_path=path,
        run_id=candidate_run_id,
        chassis=address[0],
        slot=address[1],
        channel=address[2],
        module_serial=str(metadata.get("module_serial", "")),
        scope_resource=str(scope_metadata.get("visa_resource", "")),
        scope_channel=int(scope_metadata.get("channel", 1)),
        scope_input_impedance_ohm=float(
            scope_metadata["input_impedance_ohm"]
        ),
        nominal_full_scale_v=float(metadata["nominal_full_scale_v"]),
    )
    stored_gain = float(stored_result.get("gain_a", np.nan))
    if not np.isfinite(stored_gain) or not np.isclose(
        stored_gain,
        calibration.gain_a,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(
            f"QCoDeS Run {candidate_run_id} M5301A fit metadata does not "
            "match its stored voltage points"
        )
    return calibration


def run_qcs_m5301_dc_output_calibration(
    *,
    calibration_config: QcsM5301DcCalibrationConfig,
    mapper: Any = None,
    executor: Any = None,
    scope: Any = None,
    qcs_module: Any = None,
    scope_factory: Optional[Callable[[QcsDcOscilloscopeConfig], Any]] = None,
    program_callback: Optional[Callable[..., Any]] = None,
    storage_callback: Optional[Callable[..., Any]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> StoredQcsM5301DcCalibrationRun:
    """Drive each M5301A level, read 1-Mohm scope voltage, fit, and save.

    ``mapper``, ``executor``, ``scope`` and ``program_callback`` are injectable
    so the safety/reset sequence can be tested without real hardware.  The
    runner always executes a final zero-amplitude QCS program, including after
    scope or executor failure.
    """

    qcs = qcs_module or _load_qcs()
    if mapper is None:
        mapper, channel = _create_mapper(qcs, calibration_config)
    else:
        channel = _resolve_mapper_channel(
            mapper, calibration_config.virtual_channel_name
        )
    _reject_nonzero_mapped_offset(
        mapper,
        channel,
        name=calibration_config.virtual_channel_name,
    )
    if executor is None:
        executor = _create_executor(qcs, mapper)

    factory_created_scope = scope is None
    if scope is None:
        factory = scope_factory or KeysightDcVoltageMeter
        scope = factory(calibration_config.oscilloscope)
    context = scope if factory_created_scope and hasattr(scope, "__enter__") else nullcontext(scope)

    commands = calibration_config.commanded_voltages_v
    measured = np.full(commands.shape, np.nan, dtype=float)
    scope_identity = ""
    primary_error: Optional[BaseException] = None
    _emit_progress(progress_callback, 0, "Preparing QCS M5301A DC calibration")
    try:
        with context as active_scope:
            scope_identity = str(getattr(active_scope, "idn", ""))
            measure_voltage = getattr(active_scope, "measure_voltage", None)
            if not callable(measure_voltage):
                raise TypeError("scope adapter must expose measure_voltage()")
            for index, commanded in enumerate(commands):
                # Query the termination for every point.  A cached confirmation
                # is insufficient because front-panel or remote scope settings
                # can change while a calibration is running.
                _confirm_scope_one_megohm(active_scope)
                relative = float(commanded) / float(
                    calibration_config.nominal_full_scale_v
                )
                program = build_qcs_m5301_dc_level_program(
                    qcs,
                    channel=channel,
                    commanded_voltage_v=float(commanded),
                    nominal_full_scale_v=calibration_config.nominal_full_scale_v,
                    duration_s=calibration_config.waveform_duration_s,
                    name=f"M5301A DC calibration point {index + 1}",
                )
                program = _invoke_program_callback(
                    program_callback,
                    program=program,
                    commanded_voltage_v=float(commanded),
                    relative_amplitude=relative,
                    is_reset=False,
                )
                _execute(executor, program)
                measured[index] = _finite(
                    measure_voltage(),
                    "oscilloscope measured voltage",
                )
                _emit_progress(
                    progress_callback,
                    5 + round(75 * (index + 1) / commands.size),
                    (
                        f"M5301A DC point {index + 1}/{commands.size}: "
                        f"command {commanded:.9g} V, scope {measured[index]:.9g} V"
                    ),
                )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            reset_program = build_qcs_m5301_dc_level_program(
                qcs,
                channel=channel,
                commanded_voltage_v=0.0,
                nominal_full_scale_v=calibration_config.nominal_full_scale_v,
                duration_s=calibration_config.waveform_duration_s,
                name="M5301A DC calibration safe reset",
            )
            reset_program = _invoke_program_callback(
                program_callback,
                program=reset_program,
                commanded_voltage_v=0.0,
                relative_amplitude=0.0,
                is_reset=True,
            )
            _execute(executor, reset_program)
        except BaseException as reset_error:
            if primary_error is None:
                raise RuntimeError(
                    "M5301A calibration mandatory zero-output reset failed; "
                    "the output state is unsafe and must be checked manually. "
                    f"Reset error: {reset_error}"
                ) from reset_error
            unsafe_error = RuntimeError(
                "M5301A calibration failed and its mandatory zero-output "
                "reset also failed; the output state is unsafe and must be "
                "checked manually. "
                f"Calibration error: {primary_error}. "
                f"Reset error: {reset_error}"
            )
            # Retain both original exception objects for callers, logs, and
            # tests; the reset failure is also the explicit chained cause.
            unsafe_error.calibration_error = primary_error
            unsafe_error.reset_error = reset_error
            raise unsafe_error from reset_error

    calibration = QcsM5301DcOutputCalibration.fit(
        commands,
        measured,
        database_path=calibration_config.database_path,
        chassis=calibration_config.chassis,
        slot=calibration_config.slot,
        channel=calibration_config.channel,
        module_serial=calibration_config.module_serial,
        scope_resource=calibration_config.oscilloscope.visa_resource,
        scope_channel=calibration_config.oscilloscope.channel,
        scope_input_impedance_ohm=(QCS_M5301_SCOPE_INPUT_IMPEDANCE_OHMS),
        nominal_full_scale_v=calibration_config.nominal_full_scale_v,
    )
    if calibration.r_squared < float(calibration_config.minimum_r_squared):
        raise RuntimeError(
            "M5301A DC output did not produce the required linear origin fit: "
            f"A={calibration.gain_a:.9g}, R^2={calibration.r_squared:.9g}, "
            f"RMSE={calibration.rmse_v:.9g} V. Verify the selected M5301A "
            "SMA, oscilloscope channel, cable, and 1 Mohm termination. The "
            "invalid calibration was not saved."
        )
    _emit_progress(
        progress_callback,
        85,
        (
            f"Fitted M5301A A={calibration.gain_a:.9g}, "
            f"R^2={calibration.r_squared:.9g}"
        ),
    )
    store = storage_callback or store_qcs_m5301_dc_output_calibration
    stored = store(
        calibration_config,
        commands,
        measured,
        calibration,
        scope_identity=scope_identity,
    )
    _emit_progress(
        progress_callback,
        100,
        f"M5301A DC calibration Run {int(stored.run_id)} saved",
    )
    return stored


__all__ = [
    "DEFAULT_QCS_M5301_NOMINAL_FULL_SCALE_V",
    "KeysightDcVoltageMeter",
    "MIN_QCS_M5301_DC_CALIBRATION_R_SQUARED",
    "QCS_M5301_DC_OUTPUT_CALIBRATION_SCHEMA",
    "QCS_M5301_INSTRUMENT_MODEL",
    "QCS_M5301_SCOPE_INPUT_IMPEDANCE_OHMS",
    "QcsDcOscilloscopeConfig",
    "QcsM5301DcCalibrationConfig",
    "QcsM5301DcOutputCalibration",
    "StoredQcsM5301DcCalibrationRun",
    "build_qcs_m5301_dc_level_program",
    "load_qcs_m5301_dc_output_calibration",
    "run_qcs_m5301_dc_output_calibration",
    "store_qcs_m5301_dc_output_calibration",
]
