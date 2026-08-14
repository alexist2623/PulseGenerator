"""M5300A output-power calibration through an M5200A input.

This module deliberately uses an application-owned SQLite schema instead of
the legacy QICK gain-code calibration schema.  One calibration point records
the M5300A relative amplitude, the integrated M5200A I/Q value, and the power
at the M5300A 50-ohm connector.

QCS returns M5200A trace samples in volts.  For a matched coherent RF
IntegrationFilter, ``abs(I + 1j*Q)`` is the RF peak voltage at the M5200A
connector.  The production calibration therefore converts that measured
voltage directly to power into 50 ohms:

``qcs_m5200_voltage_50ohm``
    Compute ``Vpk**2 / (2*50 ohm)`` from the measured I/Q magnitude.  The
    resulting M5300A map is derived directly from the voltage measured by the
    connected M5200A; no external source or operator-supplied fit is used.

Two older reference modes remain readable for database compatibility, but
new GUI calibrations no longer create them:

``reference_calibrated``
    Apply a previously measured linear transfer from ``20*log10(|I+jQ|)`` to
    dBm at the M5200A input, then add the measured cable/path loss.

``nominal_m5200_50ohm``
    Legacy user-supplied volts-per-I/Q scaling.

The fitted response follows the QICK 50 kSPS calibration strategy: at each
frequency, upper-amplitude points estimate
``P_full_scale = P_measured - 20*log10(relative_amplitude)``.  The median is
interpolated in frequency (never extrapolated), and target dBm is converted
back to a relative M5300A amplitude.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path
import sqlite3
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np


QCS_RF_POWER_CALIBRATION_SCHEMA = (
    "pulse-generator-qcs-m5300-m5200-power-calibration-v1"
)
QCS_RF_CALIBRATION_RUN_TABLE = "qcs_rf_power_calibration_runs"
QCS_RF_CALIBRATION_POINT_TABLE = "qcs_rf_power_calibration_points"
REFERENCE_CALIBRATED = "reference_calibrated"
NOMINAL_M5200_50OHM = "nominal_m5200_50ohm"
QCS_M5200_VOLTAGE_50OHM = "qcs_m5200_voltage_50ohm"
CALIBRATION_QUALITIES = (
    QCS_M5200_VOLTAGE_50OHM,
    REFERENCE_CALIBRATED,
    NOMINAL_M5200_50OHM,
)

M5200_SAMPLE_RATE_HZ = 4_800_000_000.0
M5200_INTEGRATION_BLOCK_SAMPLES = 16
M5200_MAX_INTEGRATION_SAMPLES = 32_768
QCS_FABRIC_CLOCK_HZ = 300_000_000.0
# Keep each submitted acquisition program inside the segmented 100 us
# construction already exercised on the connected QCS 2.5.5 system.  A
# longer calibration average reuses the same mapper, backend, executor, and
# (for equal-sized passes) Program rather than creating thousands of layers in
# one HCL graph.
QCS_RF_CALIBRATION_INTER_SEGMENT_DELAY_S = 20.0e-9
QCS_RF_CALIBRATION_MAX_PASS_INTEGRATION_SAMPLES = 480_000
QCS_RF_CALIBRATION_MAX_PASS_INTEGRATION_DURATION_S = (
    QCS_RF_CALIBRATION_MAX_PASS_INTEGRATION_SAMPLES
    / M5200_SAMPLE_RATE_HZ
)
QCS_RF_CALIBRATION_MAX_TOTAL_INTEGRATION_SAMPLES = 480_000_000
QCS_RF_CALIBRATION_MAX_TOTAL_INTEGRATION_DURATION_S = (
    QCS_RF_CALIBRATION_MAX_TOTAL_INTEGRATION_SAMPLES
    / M5200_SAMPLE_RATE_HZ
)


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _model_name(value: Any) -> str:
    text = str(value).strip().upper()
    if "M5300" in text:
        return "M5300A"
    if "M5200" in text:
        return "M5200A"
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class QcsPhysicalChannelIdentity:
    """Exact physical connector identity persisted with a calibration."""

    host_controller: int
    chassis: int
    slot: int
    channel: int
    module_model: str

    def __post_init__(self) -> None:
        for name in ("host_controller", "chassis", "slot", "channel"):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, 1),
            )
        model = _model_name(self.module_model)
        if not model:
            raise ValueError("module_model must not be empty")
        object.__setattr__(self, "module_model", model)

    @property
    def address_tuple(self) -> Tuple[int, int, int, int]:
        return (
            self.host_controller,
            self.chassis,
            self.slot,
            self.channel,
        )

    def describe(self) -> str:
        h, chassis, slot, channel = self.address_tuple
        return (
            f"{self.module_model} host {h}, chassis {chassis}, "
            f"slot {slot}, channel {channel}"
        )


@dataclass(frozen=True)
class M5200PowerReference:
    """Conversion from an M5200A integrated-IQ magnitude to connector dBm.

    For a current QCS voltage reference, ``iq_magnitude`` is the coherent RF
    peak voltage in volts and is converted directly using a 50-ohm load.
    For a calibrated reference, ``power_dbm`` is
    ``slope * 20*log10(iq_magnitude) + intercept_dbm + path_loss_db``.
    The calibrated and nominal branches are retained only to load legacy
    calibration databases.
    """

    mode: str
    path_loss_db: float = 0.0
    slope: float = 1.0
    intercept_dbm: Optional[float] = None
    volts_per_iq_unit: float = 1.0
    acknowledge_nominal_scaling: bool = False
    source: str = ""
    uncertainty_db: Optional[float] = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in CALIBRATION_QUALITIES:
            raise ValueError(
                "M5200 reference mode must be "
                "'qcs_m5200_voltage_50ohm', 'reference_calibrated', or "
                "'nominal_m5200_50ohm'"
            )
        path_loss = _finite(self.path_loss_db, "path_loss_db")
        volts_per_unit = _finite(
            self.volts_per_iq_unit,
            "volts_per_iq_unit",
            positive=True,
        )
        slope = _finite(self.slope, "reference slope", positive=True)
        intercept = self.intercept_dbm
        if mode == REFERENCE_CALIBRATED:
            if intercept is None:
                raise ValueError(
                    "reference_calibrated mode requires intercept_dbm"
                )
            intercept = _finite(intercept, "reference intercept_dbm")
        elif mode == NOMINAL_M5200_50OHM and not bool(
            self.acknowledge_nominal_scaling
        ):
            raise ValueError(
                "nominal M5200 50-ohm conversion requires explicit "
                "acknowledge_nominal_scaling=True"
            )
        uncertainty = self.uncertainty_db
        if uncertainty is not None:
            uncertainty = _finite(
                uncertainty,
                "reference uncertainty_db",
            )
            if uncertainty < 0.0:
                raise ValueError("reference uncertainty_db must be nonnegative")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "path_loss_db", path_loss)
        object.__setattr__(self, "slope", slope)
        object.__setattr__(self, "intercept_dbm", intercept)
        object.__setattr__(self, "volts_per_iq_unit", volts_per_unit)
        object.__setattr__(self, "source", str(self.source).strip())
        object.__setattr__(self, "uncertainty_db", uncertainty)

    @classmethod
    def qcs_voltage_50ohm(
        cls,
    ) -> "M5200PowerReference":
        """Build the automatic QCS volts-to-50-ohm power reference."""

        return cls(
            mode=QCS_M5200_VOLTAGE_50OHM,
            source="Keysight QCS M5200 voltage-scaled I/Q",
        )

    @classmethod
    def nominal_50ohm(
        cls,
        *,
        volts_per_iq_unit: float,
        path_loss_db: float = 0.0,
        acknowledge_nominal_scaling: bool = False,
        source: str = "",
    ) -> "M5200PowerReference":
        return cls(
            mode=NOMINAL_M5200_50OHM,
            volts_per_iq_unit=volts_per_iq_unit,
            path_loss_db=path_loss_db,
            acknowledge_nominal_scaling=acknowledge_nominal_scaling,
            source=source,
        )

    @classmethod
    def calibrated(
        cls,
        *,
        slope: float,
        intercept_dbm: float,
        path_loss_db: float = 0.0,
        source: str = "",
        uncertainty_db: Optional[float] = None,
    ) -> "M5200PowerReference":
        return cls(
            mode=REFERENCE_CALIBRATED,
            slope=slope,
            intercept_dbm=intercept_dbm,
            path_loss_db=path_loss_db,
            source=source,
            uncertainty_db=uncertainty_db,
        )

    def output_power_dbm(self, iq_magnitude: Any) -> np.ndarray:
        magnitude = np.asarray(iq_magnitude, dtype=float)
        if (
            magnitude.size == 0
            or not np.all(np.isfinite(magnitude))
            or np.any(magnitude <= 0.0)
        ):
            raise ValueError(
                "M5200 IQ magnitude must contain finite positive values"
            )
        if self.mode == REFERENCE_CALIBRATED:
            result = (
                self.slope * 20.0 * np.log10(magnitude)
                + float(self.intercept_dbm)
            )
        else:
            # QCS M5200 I/Q is already expressed in volts.  Only the legacy
            # nominal mode applies an additional user-provided multiplier.
            # For a coherent sine, the matched IntegrationFilter magnitude is
            # Vpk, so P = Vpk^2 / (2R).
            peak_voltage = magnitude * self.volts_per_iq_unit
            if self.mode == QCS_M5200_VOLTAGE_50OHM:
                peak_voltage = magnitude
            watts = peak_voltage**2 / (2.0 * 50.0)
            result = 10.0 * np.log10(watts * 1000.0)
        return result + self.path_loss_db


@dataclass(frozen=True)
class M5300PowerCalibrationConfig:
    """Acquisition settings for one M5300A-to-M5200A calibration run."""

    database_path: str
    mapper_path: str
    rf_channel_name: str
    acquisition_channel_name: str
    frequencies_hz: Tuple[float, ...]
    relative_amplitudes: Tuple[float, ...]
    input_reference: M5200PowerReference
    integration_duration_s: float = 1.0e-6
    repetitions: int = 100
    init_time_s: float = 100.0e-6
    expected_lo_frequency_hz: Optional[float] = None
    notes: str = ""

    def __post_init__(self) -> None:
        database_path = str(self.database_path).strip()
        mapper_path = str(self.mapper_path).strip()
        if not database_path:
            raise ValueError("calibration database_path must not be empty")
        if not mapper_path:
            raise ValueError("QCS mapper_path must not be empty")
        rf_name = str(self.rf_channel_name).strip()
        input_name = str(self.acquisition_channel_name).strip()
        if not rf_name or not input_name:
            raise ValueError("QCS output and acquisition channel names are required")
        frequencies = tuple(float(value) for value in self.frequencies_hz)
        amplitudes = tuple(float(value) for value in self.relative_amplitudes)
        if len(frequencies) < 2 or len(set(frequencies)) != len(frequencies):
            raise ValueError(
                "M5300 calibration requires at least two unique frequencies"
            )
        if len(amplitudes) < 2 or len(set(amplitudes)) != len(amplitudes):
            raise ValueError(
                "M5300 calibration requires at least two unique amplitudes"
            )
        if (
            not all(isfinite(value) and value > 0.0 for value in frequencies)
            or tuple(sorted(frequencies)) != frequencies
        ):
            raise ValueError(
                "calibration frequencies must be finite, positive, and increasing"
            )
        if (
            not all(isfinite(value) and 0.0 < value <= 1.0 for value in amplitudes)
            or tuple(sorted(amplitudes)) != amplitudes
        ):
            raise ValueError(
                "relative amplitudes must be increasing values in (0, 1]"
            )
        integration = _finite(
            self.integration_duration_s,
            "integration_duration_s",
            positive=True,
        )
        if (
            integration
            > QCS_RF_CALIBRATION_MAX_TOTAL_INTEGRATION_DURATION_S
        ):
            raise ValueError(
                "M5300/M5200 calibration total integrated I/Q averaging "
                f"time must not exceed "
                f"{QCS_RF_CALIBRATION_MAX_TOTAL_INTEGRATION_DURATION_S * 1e3:g} "
                "ms"
            )
        repetitions = _integer(self.repetitions, "repetitions", 1)
        init_time = _finite(self.init_time_s, "init_time_s")
        if init_time < 0.0:
            raise ValueError("init_time_s must be nonnegative")
        lo_frequency = self.expected_lo_frequency_hz
        if lo_frequency is not None:
            lo_frequency = _finite(
                lo_frequency,
                "expected_lo_frequency_hz",
            )
            if lo_frequency < 0.0:
                raise ValueError(
                    "expected_lo_frequency_hz must be nonnegative"
                )
        object.__setattr__(self, "database_path", database_path)
        object.__setattr__(self, "mapper_path", mapper_path)
        object.__setattr__(self, "rf_channel_name", rf_name)
        object.__setattr__(self, "acquisition_channel_name", input_name)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "relative_amplitudes", amplitudes)
        object.__setattr__(self, "integration_duration_s", integration)
        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(self, "init_time_s", init_time)
        object.__setattr__(self, "expected_lo_frequency_hz", lo_frequency)
        object.__setattr__(self, "notes", str(self.notes).strip())


@dataclass(frozen=True)
class M5300PowerCalibration:
    """Loaded M5300A relative-amplitude to 50-ohm connector-power map."""

    database_path: Path
    run_id: int
    created_utc: str
    output_identity: QcsPhysicalChannelIdentity
    input_identity: QcsPhysicalChannelIdentity
    mapper_sha256: str
    lo_frequency_hz: float
    termination_ohm: float
    calibration_quality: str
    input_reference: M5200PowerReference
    integration_duration_s: float
    repetitions: int
    frequencies_hz: np.ndarray
    relative_amplitudes: np.ndarray
    mean_i: np.ndarray
    mean_q: np.ndarray
    iq_magnitude: np.ndarray
    power_dbm: np.ndarray
    notes: str = ""

    def __post_init__(self) -> None:
        if self.output_identity.module_model != "M5300A":
            raise ValueError("QCS RF calibration output must be M5300A")
        if self.input_identity.module_model != "M5200A":
            raise ValueError("QCS RF calibration input must be M5200A")
        lo_frequency = _finite(self.lo_frequency_hz, "lo_frequency_hz")
        if lo_frequency < 0.0:
            raise ValueError("lo_frequency_hz must be nonnegative")
        object.__setattr__(self, "lo_frequency_hz", lo_frequency)
        if not np.isclose(float(self.termination_ohm), 50.0):
            raise ValueError("M5300 RF calibration requires 50-ohm termination")
        if self.calibration_quality not in CALIBRATION_QUALITIES:
            raise ValueError("unknown QCS RF calibration quality")
        arrays = []
        for name in (
            "frequencies_hz",
            "relative_amplitudes",
            "mean_i",
            "mean_q",
            "iq_magnitude",
            "power_dbm",
        ):
            value = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if value.size == 0 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite calibration rows")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
            arrays.append(value)
        if len({array.size for array in arrays}) != 1:
            raise ValueError("QCS RF calibration columns have different lengths")
        if np.any(self.frequencies_hz <= 0.0):
            raise ValueError("calibration frequencies must be positive")
        if np.any(self.relative_amplitudes <= 0.0) or np.any(
            self.relative_amplitudes > 1.0
        ):
            raise ValueError("calibration amplitudes must be in (0, 1]")
        if np.any(self.iq_magnitude <= 0.0):
            raise ValueError("calibration IQ magnitudes must be positive")
        pairs = np.column_stack(
            (self.frequencies_hz, self.relative_amplitudes)
        )
        if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
            raise ValueError("QCS RF calibration contains duplicate grid rows")
        unique_frequencies = np.unique(self.frequencies_hz)
        if unique_frequencies.size < 2:
            raise ValueError("QCS RF calibration needs at least two frequencies")
        expected_amplitudes = None
        for frequency in unique_frequencies:
            values = np.sort(
                self.relative_amplitudes[self.frequencies_hz == frequency]
            )
            if values.size < 2:
                raise ValueError(
                    "each QCS RF calibration frequency needs two amplitudes"
                )
            if expected_amplitudes is None:
                expected_amplitudes = values
            elif not np.array_equal(values, expected_amplitudes):
                raise ValueError("QCS RF calibration grid is incomplete")

    @property
    def frequency_coverage_hz(self) -> Tuple[float, float]:
        return (
            float(np.min(self.frequencies_hz)),
            float(np.max(self.frequencies_hz)),
        )

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "schema": QCS_RF_POWER_CALIBRATION_SCHEMA,
            "run_id": int(self.run_id),
            "database_path": str(self.database_path),
            "output": asdict(self.output_identity),
            "input": asdict(self.input_identity),
            "mapper_sha256": self.mapper_sha256,
            "lo_frequency_hz": float(self.lo_frequency_hz),
            "termination_ohm": float(self.termination_ohm),
            "calibration_quality": self.calibration_quality,
            "frequency_coverage_hz": list(self.frequency_coverage_hz),
        }

    def _frequency_response_rows(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        unique_frequencies = np.unique(self.frequencies_hz)
        full_scale = []
        minimum_power = []
        maximum_power = []
        for frequency in unique_frequencies:
            mask = self.frequencies_hz == frequency
            amplitudes = self.relative_amplitudes[mask]
            powers = self.power_dbm[mask]
            order = np.argsort(amplitudes)
            amplitudes = amplitudes[order]
            powers = powers[order]
            # Use the upper half of measured amplitudes, with at least two
            # points.  A median makes the fit insensitive to one noisy point.
            upper_count = max(2, int(np.ceil(amplitudes.size / 2.0)))
            upper_amplitudes = amplitudes[-upper_count:]
            upper_powers = powers[-upper_count:]
            full_scale.append(
                float(
                    np.median(
                        upper_powers
                        - 20.0 * np.log10(upper_amplitudes)
                    )
                )
            )
            minimum_power.append(float(np.min(powers)))
            maximum_power.append(float(np.max(powers)))
        return (
            unique_frequencies,
            np.asarray(full_scale, dtype=float),
            np.asarray(minimum_power, dtype=float),
            np.asarray(maximum_power, dtype=float),
        )

    def _checked_frequencies(self, frequencies_hz: Any) -> np.ndarray:
        frequencies = np.asarray(frequencies_hz, dtype=float)
        if frequencies.size == 0 or not np.all(np.isfinite(frequencies)):
            raise ValueError("requested frequencies must be finite and nonempty")
        low, high = self.frequency_coverage_hz
        tolerance = max(1.0e-6, abs(high) * 1.0e-12)
        if np.any(frequencies < low - tolerance) or np.any(
            frequencies > high + tolerance
        ):
            raise ValueError(
                "requested frequency is outside calibration coverage "
                f"[{low:g}, {high:g}] Hz; frequency extrapolation is disabled"
            )
        return frequencies

    def full_scale_power_dbm(self, frequencies_hz: Any) -> np.ndarray:
        frequencies = self._checked_frequencies(frequencies_hz)
        fit_frequencies, full_scale, _minimum, _maximum = (
            self._frequency_response_rows()
        )
        return np.interp(frequencies, fit_frequencies, full_scale)

    def measured_power_bounds_dbm(
        self, frequencies_hz: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        frequencies = self._checked_frequencies(frequencies_hz)
        fit_frequencies, _full_scale, minimum, maximum = (
            self._frequency_response_rows()
        )
        return (
            np.interp(frequencies, fit_frequencies, minimum),
            np.interp(frequencies, fit_frequencies, maximum),
        )

    def relative_amplitudes_for_power(
        self,
        frequencies_hz: Any,
        target_power_dbm: Any,
        *,
        allow_power_extrapolation: bool = False,
    ) -> np.ndarray:
        """Map target connector dBm to M5300A relative amplitudes.

        Frequency extrapolation is always rejected.  Power extrapolation is
        rejected by default so a calibration cannot silently claim accuracy
        outside its measured dynamic range.
        """

        frequencies = self._checked_frequencies(frequencies_hz)
        target = np.asarray(target_power_dbm, dtype=float)
        try:
            target = np.broadcast_to(target, frequencies.shape)
        except ValueError as exc:
            raise ValueError(
                "target_power_dbm is not broadcastable to frequencies"
            ) from exc
        if not np.all(np.isfinite(target)):
            raise ValueError("target power must be finite")
        if not allow_power_extrapolation:
            lower, upper = self.measured_power_bounds_dbm(frequencies)
            tolerance = 1.0e-9
            if np.any(target < lower - tolerance) or np.any(
                target > upper + tolerance
            ):
                raise ValueError(
                    "target power is outside the measured calibration range; "
                    "power extrapolation is disabled"
                )
        full_scale = self.full_scale_power_dbm(frequencies)
        amplitudes = 10.0 ** ((target - full_scale) / 20.0)
        if np.any(~np.isfinite(amplitudes)) or np.any(amplitudes <= 0.0):
            raise ValueError("target power produced an invalid relative amplitude")
        if np.any(amplitudes > 1.0 + 1.0e-12):
            raise ValueError(
                "target power exceeds the calibrated M5300A full-scale output"
            )
        return np.minimum(amplitudes, 1.0)

    @classmethod
    def load(
        cls,
        database_path: str | Path,
        *,
        run_id: Optional[int] = None,
        expected_output: Optional[QcsPhysicalChannelIdentity] = None,
        expected_input: Optional[QcsPhysicalChannelIdentity] = None,
        expected_mapper_sha256: Optional[str] = None,
        expected_lo_frequency_hz: Optional[float] = None,
        required_frequencies_hz: Optional[Sequence[float]] = None,
        termination_ohm: float = 50.0,
    ) -> "M5300PowerCalibration":
        return load_m5300_power_calibration(
            database_path,
            run_id=run_id,
            expected_output=expected_output,
            expected_input=expected_input,
            expected_mapper_sha256=expected_mapper_sha256,
            expected_lo_frequency_hz=expected_lo_frequency_hz,
            required_frequencies_hz=required_frequencies_hz,
            termination_ohm=termination_ohm,
        )


@dataclass(frozen=True)
class StoredM5300PowerCalibration:
    database_path: Path
    run_id: int
    point_count: int
    calibration: M5300PowerCalibration


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {QCS_RF_CALIBRATION_RUN_TABLE} (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_tag TEXT NOT NULL,
            created_utc TEXT NOT NULL,
            output_model TEXT NOT NULL,
            output_host_controller INTEGER NOT NULL,
            output_chassis INTEGER NOT NULL,
            output_slot INTEGER NOT NULL,
            output_channel INTEGER NOT NULL,
            input_model TEXT NOT NULL,
            input_host_controller INTEGER NOT NULL,
            input_chassis INTEGER NOT NULL,
            input_slot INTEGER NOT NULL,
            input_channel INTEGER NOT NULL,
            mapper_sha256 TEXT NOT NULL,
            lo_frequency_hz REAL NOT NULL,
            termination_ohm REAL NOT NULL,
            calibration_quality TEXT NOT NULL,
            input_reference_json TEXT NOT NULL,
            integration_duration_s REAL NOT NULL,
            repetitions INTEGER NOT NULL,
            frequency_min_hz REAL NOT NULL,
            frequency_max_hz REAL NOT NULL,
            amplitude_min REAL NOT NULL,
            amplitude_max REAL NOT NULL,
            point_count INTEGER NOT NULL,
            notes TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {QCS_RF_CALIBRATION_POINT_TABLE} (
            run_id INTEGER NOT NULL,
            point_index INTEGER NOT NULL,
            frequency_hz REAL NOT NULL,
            relative_amplitude REAL NOT NULL,
            mean_i REAL NOT NULL,
            mean_q REAL NOT NULL,
            iq_magnitude REAL NOT NULL,
            power_dbm REAL NOT NULL,
            PRIMARY KEY (run_id, point_index),
            UNIQUE (run_id, frequency_hz, relative_amplitude),
            FOREIGN KEY (run_id) REFERENCES
                {QCS_RF_CALIBRATION_RUN_TABLE}(run_id) ON DELETE CASCADE
        )
        """
    )


def _grid_columns(
    frequencies_hz: Sequence[float],
    relative_amplitudes: Sequence[float],
    mean_i: Any,
    mean_q: Any,
    iq_magnitude: Optional[Any],
    power_dbm: Optional[Any],
    input_reference: M5200PowerReference,
) -> Tuple[np.ndarray, ...]:
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    amplitudes = np.asarray(relative_amplitudes, dtype=float).reshape(-1)
    if frequencies.size < 2 or amplitudes.size < 2:
        raise ValueError("calibration grid requires at least 2 x 2 points")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("calibration frequencies must be finite and positive")
    if (
        not np.all(np.isfinite(amplitudes))
        or np.any(amplitudes <= 0.0)
        or np.any(amplitudes > 1.0)
    ):
        raise ValueError("relative amplitudes must be finite values in (0, 1]")
    if np.unique(frequencies).size != frequencies.size:
        raise ValueError("calibration frequencies must be unique")
    if np.unique(amplitudes).size != amplitudes.size:
        raise ValueError("relative amplitudes must be unique")
    frequency_grid, amplitude_grid = np.meshgrid(
        frequencies,
        amplitudes,
        indexing="ij",
    )
    shape = frequency_grid.shape

    def column(values: Any, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        if result.shape == shape:
            result = result.reshape(-1)
        elif result.size == frequency_grid.size:
            result = result.reshape(-1)
        else:
            raise ValueError(
                f"{name} must have shape {shape} or "
                f"{frequency_grid.size} elements"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain finite values")
        return result

    i_values = column(mean_i, "mean_i")
    q_values = column(mean_q, "mean_q")
    if iq_magnitude is None:
        magnitude = np.hypot(i_values, q_values)
    else:
        magnitude = column(iq_magnitude, "iq_magnitude")
    if np.any(magnitude <= 0.0):
        raise ValueError("iq_magnitude must be positive")
    powers = (
        input_reference.output_power_dbm(magnitude)
        if power_dbm is None
        else column(power_dbm, "power_dbm")
    )
    return (
        frequency_grid.reshape(-1),
        amplitude_grid.reshape(-1),
        i_values,
        q_values,
        magnitude,
        np.asarray(powers, dtype=float).reshape(-1),
    )


def store_m5300_power_calibration(
    database_path: str | Path,
    *,
    frequencies_hz: Sequence[float],
    relative_amplitudes: Sequence[float],
    mean_i: Any,
    mean_q: Any,
    output_identity: QcsPhysicalChannelIdentity,
    input_identity: QcsPhysicalChannelIdentity,
    mapper_sha256: str,
    lo_frequency_hz: float,
    integration_duration_s: float,
    repetitions: int,
    input_reference: M5200PowerReference,
    iq_magnitude: Optional[Any] = None,
    power_dbm: Optional[Any] = None,
    termination_ohm: float = 50.0,
    notes: str = "",
) -> StoredM5300PowerCalibration:
    """Persist measured arrays without requiring QCS hardware in the caller."""

    database = Path(database_path).expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    if output_identity.module_model != "M5300A":
        raise ValueError("output_identity must identify an M5300A")
    if input_identity.module_model != "M5200A":
        raise ValueError("input_identity must identify an M5200A")
    mapper_digest = str(mapper_sha256).strip().lower()
    if len(mapper_digest) != 64 or any(
        value not in "0123456789abcdef" for value in mapper_digest
    ):
        raise ValueError("mapper_sha256 must contain 64 hexadecimal characters")
    lo_frequency = _finite(lo_frequency_hz, "lo_frequency_hz")
    if lo_frequency < 0.0:
        raise ValueError("lo_frequency_hz must be nonnegative")
    integration = _finite(
        integration_duration_s,
        "integration_duration_s",
        positive=True,
    )
    repetitions = _integer(repetitions, "repetitions", 1)
    termination = _finite(termination_ohm, "termination_ohm", positive=True)
    if not np.isclose(termination, 50.0):
        raise ValueError("M5300 output calibration requires 50-ohm termination")
    columns = _grid_columns(
        frequencies_hz,
        relative_amplitudes,
        mean_i,
        mean_q,
        iq_magnitude,
        power_dbm,
        input_reference,
    )
    frequency_column, amplitude_column = columns[:2]
    created = datetime.now(timezone.utc).isoformat()
    output = output_identity
    input_channel = input_identity
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _create_schema(connection)
        cursor = connection.execute(
            f"""
            INSERT INTO {QCS_RF_CALIBRATION_RUN_TABLE} (
                schema_tag, created_utc,
                output_model, output_host_controller, output_chassis,
                output_slot, output_channel,
                input_model, input_host_controller, input_chassis,
                input_slot, input_channel,
                mapper_sha256, lo_frequency_hz, termination_ohm,
                calibration_quality, input_reference_json,
                integration_duration_s, repetitions,
                frequency_min_hz, frequency_max_hz,
                amplitude_min, amplitude_max, point_count, notes
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            """,
            (
                QCS_RF_POWER_CALIBRATION_SCHEMA,
                created,
                output.module_model,
                *output.address_tuple,
                input_channel.module_model,
                *input_channel.address_tuple,
                mapper_digest,
                lo_frequency,
                termination,
                input_reference.mode,
                json.dumps(asdict(input_reference), sort_keys=True),
                integration,
                repetitions,
                float(np.min(frequency_column)),
                float(np.max(frequency_column)),
                float(np.min(amplitude_column)),
                float(np.max(amplitude_column)),
                int(frequency_column.size),
                str(notes).strip(),
            ),
        )
        run_id = int(cursor.lastrowid)
        connection.executemany(
            f"""
            INSERT INTO {QCS_RF_CALIBRATION_POINT_TABLE} (
                run_id, point_index, frequency_hz, relative_amplitude,
                mean_i, mean_q, iq_magnitude, power_dbm
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    index,
                    *(float(column[index]) for column in columns),
                )
                for index in range(frequency_column.size)
            ],
        )
    calibration = load_m5300_power_calibration(database, run_id=run_id)
    return StoredM5300PowerCalibration(
        database_path=database,
        run_id=run_id,
        point_count=int(frequency_column.size),
        calibration=calibration,
    )


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _identity_from_run(row: Mapping[str, Any], prefix: str) -> QcsPhysicalChannelIdentity:
    return QcsPhysicalChannelIdentity(
        host_controller=row[f"{prefix}_host_controller"],
        chassis=row[f"{prefix}_chassis"],
        slot=row[f"{prefix}_slot"],
        channel=row[f"{prefix}_channel"],
        module_model=row[f"{prefix}_model"],
    )


def _calibration_from_connection(
    connection: sqlite3.Connection,
    database: Path,
    run_id: int,
) -> M5300PowerCalibration:
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        f"SELECT * FROM {QCS_RF_CALIBRATION_RUN_TABLE} WHERE run_id=?",
        (int(run_id),),
    ).fetchone()
    if row is None:
        raise KeyError(f"QCS RF calibration run {run_id} was not found")
    record = dict(row)
    if record["schema_tag"] != QCS_RF_POWER_CALIBRATION_SCHEMA:
        raise ValueError(
            f"run {run_id} is not a supported QCS M5300 calibration schema"
        )
    reference = M5200PowerReference(**json.loads(record["input_reference_json"]))
    points = connection.execute(
        f"""
        SELECT frequency_hz, relative_amplitude, mean_i, mean_q,
               iq_magnitude, power_dbm
        FROM {QCS_RF_CALIBRATION_POINT_TABLE}
        WHERE run_id=?
        ORDER BY frequency_hz, relative_amplitude
        """,
        (int(run_id),),
    ).fetchall()
    if len(points) != int(record["point_count"]):
        raise ValueError(f"QCS RF calibration run {run_id} is incomplete")
    columns = list(zip(*(tuple(point) for point in points)))
    return M5300PowerCalibration(
        database_path=database,
        run_id=int(run_id),
        created_utc=str(record["created_utc"]),
        output_identity=_identity_from_run(record, "output"),
        input_identity=_identity_from_run(record, "input"),
        mapper_sha256=str(record["mapper_sha256"]),
        lo_frequency_hz=float(record["lo_frequency_hz"]),
        termination_ohm=float(record["termination_ohm"]),
        calibration_quality=str(record["calibration_quality"]),
        input_reference=reference,
        integration_duration_s=float(record["integration_duration_s"]),
        repetitions=int(record["repetitions"]),
        frequencies_hz=np.asarray(columns[0]),
        relative_amplitudes=np.asarray(columns[1]),
        mean_i=np.asarray(columns[2]),
        mean_q=np.asarray(columns[3]),
        iq_magnitude=np.asarray(columns[4]),
        power_dbm=np.asarray(columns[5]),
        notes=str(record["notes"]),
    )


def _validate_loaded_calibration(
    calibration: M5300PowerCalibration,
    *,
    expected_output: Optional[QcsPhysicalChannelIdentity],
    expected_input: Optional[QcsPhysicalChannelIdentity],
    expected_mapper_sha256: Optional[str],
    expected_lo_frequency_hz: Optional[float],
    required_frequencies_hz: Optional[Sequence[float]],
    termination_ohm: float,
) -> None:
    if expected_output is not None and calibration.output_identity != expected_output:
        raise ValueError(
            "calibration M5300 output does not match the selected connector: "
            f"stored {calibration.output_identity.describe()}, expected "
            f"{expected_output.describe()}"
        )
    if expected_input is not None and calibration.input_identity != expected_input:
        raise ValueError(
            "calibration M5200 input does not match the selected connector: "
            f"stored {calibration.input_identity.describe()}, expected "
            f"{expected_input.describe()}"
        )
    if expected_mapper_sha256 is not None:
        digest = str(expected_mapper_sha256).strip().lower()
        if calibration.mapper_sha256 != digest:
            raise ValueError("calibration mapper does not match the active QCS mapper")
    if expected_lo_frequency_hz is not None:
        expected_lo = _finite(
            expected_lo_frequency_hz,
            "expected_lo_frequency_hz",
        )
        if expected_lo < 0.0:
            raise ValueError("expected_lo_frequency_hz must be nonnegative")
        if not np.isclose(
            calibration.lo_frequency_hz,
            expected_lo,
            rtol=0.0,
            atol=1.0,
        ):
            raise ValueError("calibration M5300 LO frequency does not match")
    if not np.isclose(
        calibration.termination_ohm,
        float(termination_ohm),
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("calibration termination does not match 50 ohm")
    if required_frequencies_hz is not None:
        calibration._checked_frequencies(required_frequencies_hz)


def load_m5300_power_calibration(
    database_path: str | Path,
    *,
    run_id: Optional[int] = None,
    expected_output: Optional[QcsPhysicalChannelIdentity] = None,
    expected_input: Optional[QcsPhysicalChannelIdentity] = None,
    expected_mapper_sha256: Optional[str] = None,
    expected_lo_frequency_hz: Optional[float] = None,
    required_frequencies_hz: Optional[Sequence[float]] = None,
    termination_ohm: float = 50.0,
) -> M5300PowerCalibration:
    """Load the newest matching run, rejecting legacy QICK databases."""

    database = Path(database_path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"QCS RF calibration database not found: {database}")
    with sqlite3.connect(database) as connection:
        tables = _table_names(connection)
        required_tables = {
            QCS_RF_CALIBRATION_RUN_TABLE,
            QCS_RF_CALIBRATION_POINT_TABLE,
        }
        if not required_tables.issubset(tables):
            legacy_hint = (
                " This appears to be a legacy QICK/QCoDeS database."
                if "runs" in tables
                else ""
            )
            raise ValueError(
                "database does not contain the tagged QCS M5300/M5200 RF "
                f"calibration schema.{legacy_hint}"
            )
        if run_id is None:
            run_ids = [
                int(row[0])
                for row in connection.execute(
                    f"""
                    SELECT run_id FROM {QCS_RF_CALIBRATION_RUN_TABLE}
                    WHERE schema_tag=? ORDER BY run_id DESC
                    """,
                    (QCS_RF_POWER_CALIBRATION_SCHEMA,),
                )
            ]
            if not run_ids:
                raise ValueError("database contains no QCS M5300 calibration runs")
        else:
            run_ids = [_integer(run_id, "run_id", 1)]
        failures = []
        for candidate_id in run_ids:
            try:
                calibration = _calibration_from_connection(
                    connection,
                    database,
                    candidate_id,
                )
                _validate_loaded_calibration(
                    calibration,
                    expected_output=expected_output,
                    expected_input=expected_input,
                    expected_mapper_sha256=expected_mapper_sha256,
                    expected_lo_frequency_hz=expected_lo_frequency_hz,
                    required_frequencies_hz=required_frequencies_hz,
                    termination_ohm=termination_ohm,
                )
                return calibration
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(f"Run {candidate_id}: {exc}")
        if run_id is not None and failures and "was not found" in failures[0]:
            raise KeyError(failures[0])
        raise ValueError(
            "no QCS M5300 calibration run matches the requested hardware: "
            + "; ".join(failures[:5])
        )


def _resolve_mapper_channel(mapper: Any, name: str) -> Any:
    channels = tuple(getattr(mapper, "channels", ()))
    matches = [channel for channel in channels if getattr(channel, "name", None) == name]
    if len(matches) != 1:
        raise ValueError(
            f"QCS mapper must contain exactly one virtual channel named {name!r}"
        )
    return matches[0]


def _identity_from_mapper(
    mapper: Any,
    channel: Any,
    expected_model: str,
) -> Tuple[QcsPhysicalChannelIdentity, Any]:
    get_physical = getattr(mapper, "get_physical_channels", None)
    if not callable(get_physical):
        raise TypeError("QCS mapper does not expose physical channel identities")
    physical_channels = tuple(get_physical(channel))
    if len(physical_channels) != 1:
        raise ValueError("calibration channel must map to one physical connector")
    physical = physical_channels[0]
    address = getattr(physical, "address", None)
    if address is None:
        raise TypeError("QCS physical channel does not expose an address")
    identity = QcsPhysicalChannelIdentity(
        host_controller=getattr(address, "host_controller"),
        chassis=getattr(address, "chassis"),
        slot=getattr(address, "slot"),
        channel=getattr(address, "channel"),
        module_model=_model_name(getattr(physical, "instrument", "")),
    )
    if identity.module_model != expected_model:
        raise ValueError(
            f"calibration channel maps to {identity.module_model}, "
            f"expected {expected_model}"
        )
    return identity, physical


def _scalar_value(value: Any) -> float:
    return float(getattr(value, "value", value))


def resolve_m5300_m5200_identities(
    mapper: Any,
    output_channel_name: str,
    input_channel_name: str,
) -> Tuple[
    QcsPhysicalChannelIdentity,
    QcsPhysicalChannelIdentity,
    float,
]:
    """Resolve exact connector identities and current M5300A LO from a mapper."""

    output_channels = _resolve_mapper_channel(mapper, output_channel_name)
    input_channels = _resolve_mapper_channel(mapper, input_channel_name)
    output_identity, output_physical = _identity_from_mapper(
        mapper,
        output_channels,
        "M5300A",
    )
    input_identity, _input_physical = _identity_from_mapper(
        mapper,
        input_channels,
        "M5200A",
    )
    settings = getattr(output_physical, "settings", None)
    lo_scalar = getattr(settings, "lo_frequency", None)
    if lo_scalar is None:
        raise ValueError("mapped M5300A output does not expose LO frequency")
    lo_frequency_hz = _finite(
        _scalar_value(lo_scalar),
        "mapped M5300A LO frequency",
    )
    if lo_frequency_hz < 0.0:
        raise ValueError("mapped M5300A LO frequency must be nonnegative")
    return output_identity, input_identity, lo_frequency_hz


@dataclass(frozen=True)
class _M5200IntegrationPass:
    """One bounded QCS execution contributing to a longer I/Q average."""

    segment_sample_counts: Tuple[int, ...]

    @property
    def sample_count(self) -> int:
        return int(sum(self.segment_sample_counts))

    @property
    def duration_s(self) -> float:
        return self.sample_count / M5200_SAMPLE_RATE_HZ


def _quantized_integration_sample_count(requested_s: float) -> int:
    """Round total requested averaging time upward to one 16-sample block."""

    requested = _finite(requested_s, "integration_duration_s", positive=True)
    sample_count = max(
        M5200_INTEGRATION_BLOCK_SAMPLES,
        int(
            np.ceil(
                requested
                * M5200_SAMPLE_RATE_HZ
                / M5200_INTEGRATION_BLOCK_SAMPLES
                - 1.0e-12
            )
        )
        * M5200_INTEGRATION_BLOCK_SAMPLES,
    )
    if sample_count > QCS_RF_CALIBRATION_MAX_TOTAL_INTEGRATION_SAMPLES:
        raise ValueError(
            "M5200 calibration total integrated I/Q averaging time exceeds "
            f"{QCS_RF_CALIBRATION_MAX_TOTAL_INTEGRATION_DURATION_S * 1e3:g} "
            "ms"
        )
    return sample_count


def _integration_filter_segments(sample_count: int) -> Tuple[int, ...]:
    """Split one <=100 us pass into equal, reusable IntegrationFilters."""

    count = _integer(sample_count, "integration pass sample_count", 1)
    if count % M5200_INTEGRATION_BLOCK_SAMPLES:
        raise ValueError(
            "integration pass sample count must be a multiple of "
            f"{M5200_INTEGRATION_BLOCK_SAMPLES}"
        )
    if count > QCS_RF_CALIBRATION_MAX_PASS_INTEGRATION_SAMPLES:
        raise ValueError(
            "integration pass exceeds the bounded 100 us QCS pass"
        )
    if count <= M5200_MAX_INTEGRATION_SAMPLES:
        return (count,)
    segment_count = int(
        np.ceil(count / M5200_MAX_INTEGRATION_SAMPLES)
    )
    segment_samples = int(
        np.ceil(
            count
            / segment_count
            / M5200_INTEGRATION_BLOCK_SAMPLES
        )
        * M5200_INTEGRATION_BLOCK_SAMPLES
    )
    if segment_samples > M5200_MAX_INTEGRATION_SAMPLES:
        raise RuntimeError(
            "internal M5200 calibration segment exceeds the measured "
            f"{M5200_MAX_INTEGRATION_SAMPLES:,}-sample limit"
        )
    return (segment_samples,) * segment_count


def _integration_pass_plan(
    requested_s: float,
) -> Tuple[_M5200IntegrationPass, ...]:
    """Plan a <=100 ms average as reusable <=100 us QCS passes."""

    remaining = _quantized_integration_sample_count(requested_s)
    passes = []
    while remaining:
        requested_pass_samples = min(
            remaining,
            QCS_RF_CALIBRATION_MAX_PASS_INTEGRATION_SAMPLES,
        )
        segments = _integration_filter_segments(requested_pass_samples)
        passes.append(_M5200IntegrationPass(segments))
        # Equal-filter segmentation can round a partial final pass upward by
        # a few 16-sample blocks. It still fulfills (never shortens) the user
        # request, so consume the requested part rather than the rounded part.
        remaining -= requested_pass_samples
    return tuple(passes)


def _quantized_integration_duration(requested_s: float) -> Tuple[float, int]:
    """Return the actual total integrated duration of the bounded pass plan."""

    passes = _integration_pass_plan(requested_s)
    sample_count = int(sum(current.sample_count for current in passes))
    return sample_count / M5200_SAMPLE_RATE_HZ, sample_count


def _first_result_value(value: Any, channels: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    try:
        if channels in value:
            return value[channels]
    except TypeError:
        pass
    if len(value) != 1:
        raise ValueError("QCS calibration returned multiple result channels")
    return next(iter(value.values()))


def _point_repetition_iq_from_result(
    raw_result: Any,
    channels: Any,
    point_count: int,
    repetitions: int,
    *,
    acquisition_index: int = -1,
) -> np.ndarray:
    """Return one acquisition as complex ``(point, repetition)`` values."""

    results = getattr(raw_result, "results", None)
    if results is not None and callable(getattr(results, "get_iq", None)):
        values = _first_result_value(
            results.get_iq(
                channels,
                avg=False,
                acq_index=int(acquisition_index),
            ),
            channels,
        )
    else:
        try:
            values = raw_result[channels]
        except (KeyError, TypeError, IndexError):
            values = raw_result
    if isinstance(values, (tuple, list)) and len(values) == 2:
        values = np.asarray(values[0], dtype=float) + 1j * np.asarray(
            values[1], dtype=float
        )
    array = np.asarray(values)
    if not np.iscomplexobj(array):
        if array.ndim and array.shape[-1] == 2:
            array = array[..., 0] + 1j * array[..., 1]
        else:
            array = array.astype(float) + 0j
    if array.size != point_count * repetitions:
        raise ValueError(
            "QCS calibration IQ result size does not match grid x repetitions"
        )
    # This Program is constructed as an outer QCS-resolved sweep containing
    # an inner hardware Repeat, so production results are point-first. Keep a
    # repetition-first adapter branch for injected/older results, but resolve
    # a square point_count == repetitions result according to the real graph.
    if array.ndim >= 2 and array.shape[:2] == (
        point_count,
        repetitions,
    ):
        point_shot = array.reshape(point_count, repetitions)
    elif array.ndim >= 2 and array.shape[:2] == (
        repetitions,
        point_count,
    ):
        point_shot = array.reshape(repetitions, point_count).T
    elif array.ndim >= 2 and array.shape[-1] == repetitions:
        point_shot = array.reshape(point_count, repetitions)
    else:
        point_shot = array.reshape(point_count, repetitions)
    point_shot = np.asarray(point_shot, dtype=np.complex128)
    if not np.all(np.isfinite(point_shot)):
        raise ValueError("QCS calibration IQ result contains non-finite values")
    return point_shot


def _mean_iq_from_result(
    raw_result: Any,
    channels: Any,
    point_count: int,
    repetitions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    point_shot = _point_repetition_iq_from_result(
        raw_result,
        channels,
        point_count,
        repetitions,
    )
    means = np.mean(point_shot, axis=1)
    return means.real, means.imag


CalibrationMeasurementRunner = Callable[..., Tuple[Any, Any]]


def _build_m5300_calibration_pass_program(
    *,
    qcs_module: Any,
    output_channels: Any,
    input_channels: Any,
    frequencies_hz: np.ndarray,
    relative_amplitudes: np.ndarray,
    segment_sample_counts: Tuple[int, ...],
    repetitions: int,
) -> Any:
    """Build one reusable <=100 us segmented calibration Program."""

    segments = tuple(int(value) for value in segment_sample_counts)
    if not segments:
        raise ValueError("calibration integration pass has no segments")
    if any(
        value < M5200_INTEGRATION_BLOCK_SAMPLES
        or value % M5200_INTEGRATION_BLOCK_SAMPLES
        or value > M5200_MAX_INTEGRATION_SAMPLES
        for value in segments
    ):
        raise ValueError("calibration integration pass contains an invalid filter")
    if len(segments) > 1 and bool(
        getattr(output_channels, "absolute_phase", False)
    ) != bool(getattr(input_channels, "absolute_phase", False)):
        raise ValueError(
            "segmented M5300/M5200 calibration requires matching "
            "absolute_phase settings on output and acquisition channels"
        )

    frequency = qcs_module.Scalar(
        "m5300_calibration_frequency_hz",
        value=float(frequencies_hz[0]),
        dtype=float,
    )
    amplitude = qcs_module.Scalar(
        "m5300_calibration_relative_amplitude",
        value=float(relative_amplitudes[0]),
        dtype=float,
    )
    frequency_values = qcs_module.Array(
        "m5300_calibration_frequency_values_hz",
        value=frequencies_hz,
        dtype=float,
    )
    amplitude_values = qcs_module.Array(
        "m5300_calibration_amplitude_values",
        value=relative_amplitudes,
        dtype=float,
    )
    program = qcs_module.Program(
        name="M5300A RF power calibration integration pass"
    )
    integration_filter_cache = {}
    last_segment_index = len(segments) - 1
    for segment_index, segment_samples in enumerate(segments):
        segment_duration_s = segment_samples / M5200_SAMPLE_RATE_HZ
        output_duration_s = segment_duration_s
        if segment_index != last_segment_index:
            output_duration_s += QCS_RF_CALIBRATION_INTER_SEGMENT_DELAY_S
        output_waveform = qcs_module.RFWaveform(
            duration=output_duration_s,
            envelope=qcs_module.ConstantEnvelope(),
            amplitude=amplitude,
            rf_frequency=frequency,
            instantaneous_phase=0.0,
            name=f"m5300_power_calibration_output_{segment_index}",
        )
        integration_filter = integration_filter_cache.get(segment_samples)
        if integration_filter is None:
            filter_waveform = qcs_module.RFWaveform(
                duration=segment_duration_s,
                envelope=qcs_module.ConstantEnvelope(),
                amplitude=1.0,
                rf_frequency=frequency,
                instantaneous_phase=0.0,
                name=(
                    "m5200_power_calibration_filter_"
                    f"{segment_samples}_samples"
                ),
            )
            integration_filter = qcs_module.IntegrationFilter(
                filter_waveform
            )
            integration_filter_cache[segment_samples] = integration_filter
        program.add_waveform(
            output_waveform,
            output_channels,
            new_layer=segment_index == 0,
        )
        acquisition_options = {"new_layer": False}
        if segment_index > 0:
            acquisition_options["pre_delay"] = (
                QCS_RF_CALIBRATION_INTER_SEGMENT_DELAY_S
            )
        program.add_acquisition(
            integration_filter=integration_filter,
            channels=input_channels,
            **acquisition_options,
        )
    # Frequency appears inside the IntegrationFilter and therefore remains a
    # QCS-resolved software sweep. Repeat is inserted first so repetitions are
    # performed at each frequency/amplitude point.
    program.n_shots(repetitions)
    program.sweep(
        (frequency_values, amplitude_values),
        (frequency, amplitude),
    )
    return program


def _integrated_iq_from_pass_result(
    raw_result: Any,
    channels: Any,
    *,
    point_count: int,
    repetitions: int,
    segment_sample_counts: Tuple[int, ...],
) -> np.ndarray:
    """Sample-weight one pass into complex ``(point, repetition)`` I/Q."""

    segments = tuple(int(value) for value in segment_sample_counts)
    weighted = np.zeros((point_count, repetitions), dtype=np.complex128)
    for segment_index, segment_samples in enumerate(segments):
        values = _point_repetition_iq_from_result(
            raw_result,
            channels,
            point_count,
            repetitions,
            acquisition_index=segment_index,
        )
        weighted += values * float(segment_samples)
    return weighted / float(sum(segments))


def run_m5300_power_calibration(
    config: M5300PowerCalibrationConfig,
    *,
    mapper: Any = None,
    qcs_module: Any = None,
    measurement_runner: Optional[CalibrationMeasurementRunner] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> StoredM5300PowerCalibration:
    """Acquire and store one M5300A/M5200A grid.

    ``measurement_runner`` is an injectable hardware adapter.  It receives
    keyword arguments ``config``, ``mapper``, ``output_channels``,
    ``input_channels``, ``frequencies_hz`` (flattened Cartesian grid), and
    ``relative_amplitudes``.  It returns mean-I and mean-Q arrays with one
    value per grid point.  When omitted, total averaging time is divided into
    <=100 us QCS passes. Each pass contains <=32,768-sample flat
    IntegrationFilters, and completed pass/segment values are sample-weighted
    before repetitions are averaged. Mapper, backend, executor, and equal-pass
    Program objects are reused across the complete calibration.
    """

    def progress(percent: int, text: str) -> None:
        if progress_callback is not None:
            progress_callback(int(percent), str(text))

    mapper_path = Path(config.mapper_path).expanduser().resolve()
    if mapper is None:
        if not mapper_path.is_file():
            raise FileNotFoundError(f"QCS ChannelMapper not found: {mapper_path}")
        if qcs_module is None:
            import keysight.qcs as qcs_module
        mapper = qcs_module.load(mapper_path)
    if mapper_path.is_file():
        mapper_sha256 = _sha256_file(mapper_path)
    else:
        # Injected unit-test mappers need an explicit stable identity without
        # weakening production mapper-file provenance.
        mapper_sha256 = hashlib.sha256(
            str(mapper_path).encode("utf-8")
        ).hexdigest()
    output_channels = _resolve_mapper_channel(mapper, config.rf_channel_name)
    input_channels = _resolve_mapper_channel(
        mapper, config.acquisition_channel_name
    )
    output_identity, input_identity, lo_frequency_hz = (
        resolve_m5300_m5200_identities(
            mapper,
            config.rf_channel_name,
            config.acquisition_channel_name,
        )
    )
    if config.expected_lo_frequency_hz is not None and not np.isclose(
        lo_frequency_hz,
        config.expected_lo_frequency_hz,
        rtol=0.0,
        atol=1.0,
    ):
        raise ValueError("active M5300A LO does not match calibration settings")
    frequencies = np.asarray(config.frequencies_hz, dtype=float)
    amplitudes = np.asarray(config.relative_amplitudes, dtype=float)
    frequency_grid, amplitude_grid = np.meshgrid(
        frequencies,
        amplitudes,
        indexing="ij",
    )
    flat_frequencies = frequency_grid.reshape(-1)
    flat_amplitudes = amplitude_grid.reshape(-1)
    pass_plan = _integration_pass_plan(config.integration_duration_s)
    actual_sample_count = int(
        sum(current.sample_count for current in pass_plan)
    )
    actual_duration_s = actual_sample_count / M5200_SAMPLE_RATE_HZ
    progress(10, "Prepared M5300A/M5200A calibration grid")

    if measurement_runner is not None:
        mean_i, mean_q = measurement_runner(
            config=config,
            mapper=mapper,
            output_channels=output_channels,
            input_channels=input_channels,
            frequencies_hz=flat_frequencies.copy(),
            relative_amplitudes=flat_amplitudes.copy(),
            integration_duration_s=actual_duration_s,
        )
    else:
        if qcs_module is None:
            import keysight.qcs as qcs_module
        backend = qcs_module.HclBackend(
            channel_mapper=mapper,
            hw_demod=True,
            init_time=config.init_time_s,
            blocking=True,
            suppress_rounding_warnings=True,
            keep_progress_bar=False,
            reset_phase_every_shot=True,
        )
        executor = qcs_module.Executor(backend)
        program_cache = {}
        weighted_iq = np.zeros(
            (flat_frequencies.size, config.repetitions),
            dtype=np.complex128,
        )
        completed_samples = 0
        pass_count = len(pass_plan)
        progress(
            15,
            f"Executing {pass_count:,} bounded QCS calibration pass(es)",
        )
        last_progress_percent = 15
        for pass_index, current_pass in enumerate(pass_plan):
            program_key = current_pass.segment_sample_counts
            program = program_cache.get(program_key)
            if program is None:
                program = _build_m5300_calibration_pass_program(
                    qcs_module=qcs_module,
                    output_channels=output_channels,
                    input_channels=input_channels,
                    frequencies_hz=flat_frequencies,
                    relative_amplitudes=flat_amplitudes,
                    segment_sample_counts=program_key,
                    repetitions=config.repetitions,
                )
                program_cache[program_key] = program
            raw_result = executor.execute(program)
            pass_iq = _integrated_iq_from_pass_result(
                raw_result,
                input_channels,
                point_count=flat_frequencies.size,
                repetitions=config.repetitions,
                segment_sample_counts=program_key,
            )
            weighted_iq += pass_iq * float(current_pass.sample_count)
            completed_samples += current_pass.sample_count
            completed_percent = 15 + int(
                60 * (pass_index + 1) / pass_count
            )
            if (
                completed_percent != last_progress_percent
                or pass_index + 1 == pass_count
            ):
                progress(
                    completed_percent,
                    (
                        f"Acquired QCS M5300A calibration pass "
                        f"{pass_index + 1:,}/{pass_count:,}"
                    ),
                )
                last_progress_percent = completed_percent
        if completed_samples != actual_sample_count:
            raise RuntimeError(
                "internal QCS RF calibration integration accounting mismatch"
            )
        point_repetition_iq = weighted_iq / float(completed_samples)
        means = np.mean(point_repetition_iq, axis=1)
        mean_i, mean_q = means.real, means.imag
    mean_i = np.asarray(mean_i, dtype=float).reshape(frequency_grid.shape)
    mean_q = np.asarray(mean_q, dtype=float).reshape(frequency_grid.shape)
    progress(80, "Storing tagged QCS RF power calibration")
    stored = store_m5300_power_calibration(
        config.database_path,
        frequencies_hz=frequencies,
        relative_amplitudes=amplitudes,
        mean_i=mean_i,
        mean_q=mean_q,
        output_identity=output_identity,
        input_identity=input_identity,
        mapper_sha256=mapper_sha256,
        lo_frequency_hz=lo_frequency_hz,
        integration_duration_s=actual_duration_s,
        repetitions=config.repetitions,
        input_reference=config.input_reference,
        notes=config.notes,
    )
    progress(100, f"Stored QCS RF calibration Run {stored.run_id}")
    return stored


__all__ = [
    "CALIBRATION_QUALITIES",
    "M5200PowerReference",
    "M5300PowerCalibration",
    "M5300PowerCalibrationConfig",
    "NOMINAL_M5200_50OHM",
    "QCS_M5200_VOLTAGE_50OHM",
    "QCS_RF_CALIBRATION_MAX_PASS_INTEGRATION_DURATION_S",
    "QCS_RF_CALIBRATION_MAX_PASS_INTEGRATION_SAMPLES",
    "QCS_RF_CALIBRATION_MAX_TOTAL_INTEGRATION_DURATION_S",
    "QCS_RF_CALIBRATION_MAX_TOTAL_INTEGRATION_SAMPLES",
    "QCS_RF_POWER_CALIBRATION_SCHEMA",
    "QcsPhysicalChannelIdentity",
    "REFERENCE_CALIBRATED",
    "StoredM5300PowerCalibration",
    "load_m5300_power_calibration",
    "resolve_m5300_m5200_identities",
    "run_m5300_power_calibration",
    "store_m5300_power_calibration",
]
