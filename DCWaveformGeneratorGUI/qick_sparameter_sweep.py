"""Hardware RF frequency sweep and FIR-DDR S-parameter acquisition.

This module is intentionally independent of the AWG-tuning waveform path.
The tProcessor advances the normal RF generator DDS and dynamic-readout DDS
registers together, captures one FIR-decimated DDR trace per frequency, and
reduces each trace to one complex I/Q value.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from math import ceil, isfinite
from numbers import Integral, Real
from pathlib import Path
import json
import shutil
import sqlite3
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from .power_calibration import (
        CalibrationDatabase,
        CalibrationRunSummary,
        GAIN_DMEM_BASE_ADDRESS,
        GainSchedule,
        InputPowerCalibration,
        INPUT_BOARD_TYPES,
        MAX_DMEM_GAIN_ENTRIES,
        OUTPUT_BOARD_TYPES,
    )
except ImportError:
    from power_calibration import (
        CalibrationDatabase,
        CalibrationRunSummary,
        GAIN_DMEM_BASE_ADDRESS,
        GainSchedule,
        InputPowerCalibration,
        INPUT_BOARD_TYPES,
        MAX_DMEM_GAIN_ENTRIES,
        OUTPUT_BOARD_TYPES,
    )

try:
    from qick.averager_program import RAveragerProgram
except ImportError as exc:  # keep the waveform editor importable without QICK
    _QICK_IMPORT_ERROR = exc

    class RAveragerProgram:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("QICK is required for RF S-parameter sweeps") from (
                _QICK_IMPORT_ERROR
            )

else:
    _QICK_IMPORT_ERROR = None


MAX_RF_OUTPUT_GAIN = 32766
FILTER_TYPES = ("bypass", "lowpass", "highpass", "bandpass")
POWER_SCALES = ("linear", "log")
POWER_GAIN_PARAMETER = "rf_power_gain"
OUTPUT_POWER_PARAMETER = "rf_output_power_dbm"
CALIBRATED_GAIN_PARAMETER = "rf_calibrated_gain_code"
FREQUENCY_PARAMETER = "rf_frequency_mhz"
SAMPLE_INDEX_PARAMETER = "sample_index"
I_TRACE_PARAMETER = "i_trace"
Q_TRACE_PARAMETER = "q_trace"
MEAN_I_PARAMETER = "mean_i"
MEAN_Q_PARAMETER = "mean_q"
MAGNITUDE_DB_PARAMETER = "s_parameter_magnitude_db"
ADC_MAGNITUDE_DB_PARAMETER = "adc_magnitude_db"
ACTUAL_OUTPUT_POWER_PARAMETER = "actual_output_power_dbm"
ACTUAL_INPUT_POWER_PARAMETER = "actual_input_power_dbm"
PHASE_DEG_PARAMETER = "s_parameter_phase_unwrapped_deg"

ProgressCallback = Callable[[int, str], None]


def _require_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_attenuation(value: Any, name: str) -> float:
    value = _require_finite(value, name)
    if not 0.0 <= value <= 31.75:
        raise ValueError(f"{name} must be in [0, 31.75] dB")
    return value


def _require_filter(value: Any, name: str) -> str:
    value = str(value)
    if value not in FILTER_TYPES:
        raise ValueError(f"{name} must be one of {FILTER_TYPES}")
    return value


def _emit_progress(
    callback: Optional[ProgressCallback], percent: int, message: str
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(percent))), str(message))


@dataclass(frozen=True)
class SParameterSweepConfig:
    """Independent RF output/readout configuration for one hardware sweep."""

    output_ch: int = 0
    readout_ch: int = 0
    frequency_start_mhz: float = 10.0
    frequency_end_mhz: float = 100.0
    frequency_points: int = 101
    gain: int = 20000
    power_calibration_enabled: bool = False
    calibration_database_path: str = ""
    output_board_type: str = "RF_Out"
    input_board_type: str = "DC_In"
    output_power_dbm: float = -20.0
    power_sweep_enabled: bool = False
    power_start_gain: int = 1000
    power_end_gain: int = 20000
    power_start_dbm: float = -30.0
    power_end_dbm: float = -10.0
    power_points: int = 5
    power_scale: str = "linear"
    scan_time_us: float = 10.0
    output_att1_db: float = 10.0
    output_att2_db: float = 10.0
    output_filter_type: str = "bypass"
    output_filter_cutoff_ghz: float = 2.5
    output_filter_bandwidth_ghz: float = 1.0
    readout_attenuation_db: float = 20.0
    readout_dc_gain_db: float = 0.0
    readout_filter_type: str = "bypass"
    readout_filter_cutoff_ghz: float = 2.5
    readout_filter_bandwidth_ghz: float = 1.0
    loss1_db: float = 0.0
    loss2_db: float = 0.0
    amplifier_gain_db: float = 0.0
    nqz: int = 1
    margin_input_samples: int = 1024
    address: int = 0
    stride_bytes: Optional[int] = None
    force_overwrite: bool = False
    settle_seconds: float = 0.05
    trigger_width_tproc_cycles: int = 12
    recovery_tproc_cycles: int = 20
    calibrated_gain_table: Tuple[int, ...] = ()
    calibrated_frequency_point_count: int = 0
    calibrated_output_power_dbm: Optional[float] = None
    calibrated_nominal_gain_code: Optional[int] = None
    calibrated_reference_response_dbm: Optional[float] = None
    calibrated_correction_min_db: Optional[float] = None
    calibrated_correction_max_db: Optional[float] = None
    gain_dmem_base_address: int = GAIN_DMEM_BASE_ADDRESS
    calibration_output_run_id: int = 0
    calibration_input_run_id: int = 0
    calibration_output_sample_name: str = ""
    calibration_input_sample_name: str = ""

    def __post_init__(self) -> None:
        _require_int(self.output_ch, "output_ch")
        _require_int(self.readout_ch, "readout_ch")
        start = _require_finite(self.frequency_start_mhz, "frequency_start_mhz")
        stop = _require_finite(self.frequency_end_mhz, "frequency_end_mhz")
        points = _require_int(self.frequency_points, "frequency_points", 2)
        if start == stop and points > 1:
            raise ValueError("frequency start and end must differ")
        gain = _require_int(self.gain, "gain")
        if gain > MAX_RF_OUTPUT_GAIN:
            raise ValueError(
                f"gain must not exceed the RF output limit {MAX_RF_OUTPUT_GAIN}"
            )
        if not isinstance(self.power_calibration_enabled, bool):
            raise TypeError("power_calibration_enabled must be boolean")
        if self.output_board_type not in OUTPUT_BOARD_TYPES:
            raise ValueError(f"output_board_type must be one of {OUTPUT_BOARD_TYPES}")
        if self.input_board_type not in INPUT_BOARD_TYPES:
            raise ValueError(f"input_board_type must be one of {INPUT_BOARD_TYPES}")
        _require_finite(self.output_power_dbm, "output_power_dbm")
        _require_finite(self.power_start_dbm, "power_start_dbm")
        _require_finite(self.power_end_dbm, "power_end_dbm")
        if self.power_calibration_enabled:
            calibration_path = str(self.calibration_database_path).strip()
            if not calibration_path:
                raise ValueError(
                    "calibration_database_path is required when power calibration "
                    "is enabled"
                )
        if not isinstance(self.power_sweep_enabled, bool):
            raise TypeError("power_sweep_enabled must be boolean")
        power_start = _require_int(self.power_start_gain, "power_start_gain")
        power_end = _require_int(self.power_end_gain, "power_end_gain")
        power_points = _require_int(self.power_points, "power_points", 2)
        if power_start > MAX_RF_OUTPUT_GAIN or power_end > MAX_RF_OUTPUT_GAIN:
            raise ValueError(
                "power sweep gain must not exceed the RF output limit "
                f"{MAX_RF_OUTPUT_GAIN}"
            )
        if self.power_scale not in POWER_SCALES:
            raise ValueError(f"power_scale must be one of {POWER_SCALES}")
        if self.power_sweep_enabled:
            if self.power_calibration_enabled:
                if self.power_start_dbm == self.power_end_dbm:
                    raise ValueError("power sweep start and end dBm must differ")
            else:
                if power_start == power_end:
                    raise ValueError("power sweep start and end gains must differ")
                if self.power_scale == "log" and min(power_start, power_end) <= 0:
                    raise ValueError("log power sweep gains must be greater than zero")
                if np.unique(self.power_gains).size != power_points:
                    raise ValueError(
                        "power sweep points collapse to duplicate integer gain codes; "
                        "reduce the point count or widen the gain range"
                    )
        _require_finite(self.scan_time_us, "scan_time_us", positive=True)
        _require_attenuation(self.output_att1_db, "output_att1_db")
        _require_attenuation(self.output_att2_db, "output_att2_db")
        _require_attenuation(self.readout_attenuation_db, "readout_attenuation_db")
        readout_dc_gain = _require_finite(self.readout_dc_gain_db, "readout_dc_gain_db")
        if not -6.0 <= readout_dc_gain <= 26.0:
            raise ValueError("readout_dc_gain_db must be in [-6, 26] dB")
        for value, name in (
            (self.loss1_db, "loss1_db"),
            (self.loss2_db, "loss2_db"),
        ):
            loss = _require_finite(value, name)
            if not 0.0 <= loss <= 200.0:
                raise ValueError(f"{name} must be in [0, 200] dB")
        amplifier_gain = _require_finite(self.amplifier_gain_db, "amplifier_gain_db")
        if not -200.0 <= amplifier_gain <= 200.0:
            raise ValueError("amplifier_gain_db must be in [-200, 200] dB")
        _require_filter(self.output_filter_type, "output_filter_type")
        _require_filter(self.readout_filter_type, "readout_filter_type")
        _require_finite(
            self.output_filter_cutoff_ghz,
            "output_filter_cutoff_ghz",
            positive=True,
        )
        _require_finite(
            self.output_filter_bandwidth_ghz,
            "output_filter_bandwidth_ghz",
            positive=True,
        )
        _require_finite(
            self.readout_filter_cutoff_ghz,
            "readout_filter_cutoff_ghz",
            positive=True,
        )
        _require_finite(
            self.readout_filter_bandwidth_ghz,
            "readout_filter_bandwidth_ghz",
            positive=True,
        )
        _require_int(self.nqz, "nqz", 1)
        _require_int(self.margin_input_samples, "margin_input_samples")
        _require_int(self.address, "address")
        if self.stride_bytes is not None:
            _require_int(self.stride_bytes, "stride_bytes", 1)
        if not isinstance(self.force_overwrite, bool):
            raise TypeError("force_overwrite must be boolean")
        _require_finite(self.settle_seconds, "settle_seconds")
        if self.settle_seconds < 0.0:
            raise ValueError("settle_seconds must be nonnegative")
        _require_int(
            self.trigger_width_tproc_cycles,
            "trigger_width_tproc_cycles",
            1,
        )
        _require_int(self.recovery_tproc_cycles, "recovery_tproc_cycles")
        gain_table = tuple(int(value) for value in self.calibrated_gain_table)
        if len(gain_table) > MAX_DMEM_GAIN_ENTRIES:
            raise ValueError(
                f"calibrated gain table exceeds {MAX_DMEM_GAIN_ENTRIES} entries"
            )
        if any(value < 0 or value > MAX_RF_OUTPUT_GAIN for value in gain_table):
            raise ValueError("calibrated gain table contains an invalid gain")
        if gain_table:
            point_count = _require_int(
                self.calibrated_frequency_point_count,
                "calibrated_frequency_point_count",
                len(gain_table),
            )
            if point_count != points:
                raise ValueError(
                    "calibrated gain-table frequency count must match frequency_points"
                )
            if self.calibrated_output_power_dbm is None:
                raise ValueError("calibrated gain table requires output power metadata")
            _require_finite(
                self.calibrated_output_power_dbm,
                "calibrated_output_power_dbm",
            )
            if self.calibrated_nominal_gain_code is None:
                raise ValueError("calibrated gain table requires nominal gain metadata")
            nominal_gain = _require_int(
                self.calibrated_nominal_gain_code,
                "calibrated_nominal_gain_code",
                1,
            )
            if nominal_gain > MAX_RF_OUTPUT_GAIN:
                raise ValueError("calibrated nominal gain exceeds the generator limit")
            for value, name in (
                (
                    self.calibrated_reference_response_dbm,
                    "calibrated_reference_response_dbm",
                ),
                (self.calibrated_correction_min_db, "calibrated_correction_min_db"),
                (self.calibrated_correction_max_db, "calibrated_correction_max_db"),
            ):
                if value is None:
                    raise ValueError(f"calibrated gain table requires {name}")
                _require_finite(value, name)
        _require_int(self.gain_dmem_base_address, "gain_dmem_base_address", 2)
        _require_int(self.calibration_output_run_id, "calibration_output_run_id")
        _require_int(self.calibration_input_run_id, "calibration_input_run_id")

    @property
    def output_has_attenuators(self) -> bool:
        return self.output_board_type == "RF_Out"

    @property
    def input_has_attenuator(self) -> bool:
        return self.input_board_type == "RF_In"

    @property
    def effective_output_att1_db(self) -> float:
        return float(self.output_att1_db) if self.output_has_attenuators else 0.0

    @property
    def effective_output_att2_db(self) -> float:
        return float(self.output_att2_db) if self.output_has_attenuators else 0.0

    @property
    def effective_input_attenuation_db(self) -> float:
        return float(self.readout_attenuation_db) if self.input_has_attenuator else 0.0

    @property
    def effective_input_dc_gain_db(self) -> float:
        return float(self.readout_dc_gain_db) if not self.input_has_attenuator else 0.0

    @property
    def requested_frequencies_mhz(self) -> np.ndarray:
        return np.linspace(
            self.frequency_start_mhz,
            self.frequency_end_mhz,
            self.frequency_points,
            dtype=float,
        )

    @property
    def power_gains(self) -> np.ndarray:
        """Return software-swept QICK DAC gain codes in execution order."""
        if not self.power_sweep_enabled:
            return np.asarray([self.gain], dtype=np.int64)
        if self.power_scale == "linear":
            values = np.linspace(
                self.power_start_gain,
                self.power_end_gain,
                self.power_points,
                dtype=float,
            )
        else:
            values = np.geomspace(
                self.power_start_gain,
                self.power_end_gain,
                self.power_points,
                dtype=float,
            )
        gains = np.rint(values).astype(np.int64)
        gains[0] = self.power_start_gain
        gains[-1] = self.power_end_gain
        return gains

    @property
    def target_powers_dbm(self) -> np.ndarray:
        """Return calibrated output-power targets in execution order.

        ``linear`` is uniform in physical milliwatts. ``log`` is uniform on
        the logarithmic dBm axis.
        """
        if not self.power_calibration_enabled:
            raise RuntimeError("target_powers_dbm requires power calibration")
        if not self.power_sweep_enabled:
            return np.asarray([self.output_power_dbm], dtype=float)
        if self.power_scale == "log":
            return np.linspace(
                self.power_start_dbm,
                self.power_end_dbm,
                self.power_points,
                dtype=float,
            )
        start_mw = 10.0 ** (self.power_start_dbm / 10.0)
        end_mw = 10.0 ** (self.power_end_dbm / 10.0)
        values_mw = np.linspace(start_mw, end_mw, self.power_points, dtype=float)
        return 10.0 * np.log10(values_mw)

    def for_gain(self, gain: int) -> "SParameterSweepConfig":
        """Build one hardware-frequency-sweep config for a software power point."""
        return replace(
            self,
            gain=_require_int(gain, "gain"),
            power_sweep_enabled=False,
        )

    def for_gain_schedule(
        self,
        schedule: GainSchedule,
        output_run: CalibrationRunSummary,
        input_run: Optional[CalibrationRunSummary],
    ) -> "SParameterSweepConfig":
        """Build one fixed-power sweep with frequency gains loaded from DMEM."""
        if schedule.frequency_point_count != self.frequency_points:
            raise ValueError("gain schedule does not match the frequency sweep")
        return replace(
            self,
            gain=int(schedule.nominal_gain_code),
            power_sweep_enabled=False,
            calibrated_gain_table=tuple(int(value) for value in schedule.gain_codes),
            calibrated_frequency_point_count=schedule.frequency_point_count,
            calibrated_output_power_dbm=float(schedule.target_power_dbm),
            calibrated_nominal_gain_code=int(schedule.nominal_gain_code),
            calibrated_reference_response_dbm=float(schedule.reference_response_dbm),
            calibrated_correction_min_db=float(np.min(schedule.correction_db)),
            calibrated_correction_max_db=float(np.max(schedule.correction_db)),
            gain_dmem_base_address=int(schedule.dmem_base_address),
            calibration_output_run_id=int(output_run.run_id),
            calibration_input_run_id=(
                0 if input_run is None else int(input_run.run_id)
            ),
            calibration_output_sample_name=str(output_run.sample_name),
            calibration_input_sample_name=(
                "" if input_run is None else str(input_run.sample_name)
            ),
        )


@dataclass(frozen=True)
class SParameterSweepResult:
    """FIR DDR traces and derived one-complex-value response per frequency."""

    requested_frequencies_mhz: np.ndarray
    frequencies_mhz: np.ndarray
    iq_traces: np.ndarray
    mean_i: np.ndarray
    mean_q: np.ndarray
    adc_magnitude_db: np.ndarray
    magnitude_db: np.ndarray
    phase_unwrapped_deg: np.ndarray
    sample_rate_hz: float
    reserved_physical_words: Optional[int] = None
    output_power_dbm: Optional[float] = None
    nominal_gain_code: Optional[int] = None
    frequency_gain_codes: Optional[np.ndarray] = None
    actual_output_powers_dbm: Optional[np.ndarray] = None
    input_powers_dbm: Optional[np.ndarray] = None

    @classmethod
    def from_iq(
        cls,
        requested_frequencies_mhz: Any,
        frequencies_mhz: Any,
        iq_traces: Any,
        *,
        sample_rate_hz: float = 1_000_000.0,
        reserved_physical_words: Optional[int] = None,
        output_power_dbm: Optional[float] = None,
        nominal_gain_code: Optional[int] = None,
        frequency_gain_codes: Optional[Any] = None,
        actual_output_powers_dbm: Optional[Any] = None,
        input_powers_dbm: Optional[Any] = None,
    ) -> "SParameterSweepResult":
        requested = np.asarray(requested_frequencies_mhz, dtype=float).reshape(-1)
        frequencies = np.asarray(frequencies_mhz, dtype=float).reshape(-1)
        iq = np.asarray(iq_traces)
        if iq.ndim != 3 or iq.shape[-1] != 2:
            raise ValueError("S-parameter IQ must have shape (frequency, sample, 2)")
        if iq.shape[0] != frequencies.size or requested.size != frequencies.size:
            raise ValueError("frequency and IQ point counts do not match")
        if iq.shape[1] < 1:
            raise ValueError("every frequency must contain at least one IQ sample")
        mean = iq.astype(np.float64).mean(axis=1)
        mean_i = mean[:, 0]
        mean_q = mean[:, 1]
        magnitude = np.hypot(mean_i, mean_q)
        adc_magnitude_db = 20.0 * np.log10(
            np.maximum(magnitude, np.finfo(np.float64).tiny)
        )
        phase = np.degrees(np.unwrap(np.angle(mean_i + 1j * mean_q)))
        applied_gains = None
        if frequency_gain_codes is not None:
            applied_gains = np.asarray(frequency_gain_codes, dtype=np.int64).reshape(-1)
            if applied_gains.size != frequencies.size:
                raise ValueError(
                    "frequency_gain_codes must contain one gain per frequency"
                )
            if np.any(applied_gains < 0) or np.any(applied_gains > MAX_RF_OUTPUT_GAIN):
                raise ValueError("frequency_gain_codes contains an invalid gain")
        calibrated_power = None
        nominal_gain = None
        if output_power_dbm is not None:
            calibrated_power = _require_finite(output_power_dbm, "output_power_dbm")
            if applied_gains is None:
                raise ValueError(
                    "frequency-compensated output power requires frequency gain codes"
                )
            if nominal_gain_code is None:
                raise ValueError(
                    "frequency compensation requires nominal gain metadata"
                )
            nominal_gain = _require_int(nominal_gain_code, "nominal_gain_code", 1)
            if nominal_gain > MAX_RF_OUTPUT_GAIN:
                raise ValueError("nominal_gain_code exceeds the generator limit")
        actual_output = None
        actual_input = None
        if actual_output_powers_dbm is not None:
            actual_output = np.asarray(actual_output_powers_dbm, dtype=float).reshape(
                -1
            )
            if actual_output.size != frequencies.size or not np.all(
                np.isfinite(actual_output)
            ):
                raise ValueError(
                    "actual output powers must contain one finite value per frequency"
                )
        if input_powers_dbm is not None:
            actual_input = np.asarray(input_powers_dbm, dtype=float).reshape(-1)
            if actual_input.size != frequencies.size or not np.all(
                np.isfinite(actual_input)
            ):
                raise ValueError(
                    "input powers must contain one finite value per frequency"
                )
            if actual_output is None:
                raise ValueError("input powers require actual output powers")
        displayed_magnitude_db = (
            actual_input - actual_output
            if actual_input is not None
            else adc_magnitude_db
        )
        return cls(
            requested_frequencies_mhz=requested,
            frequencies_mhz=frequencies,
            iq_traces=np.ascontiguousarray(iq),
            mean_i=np.ascontiguousarray(mean_i),
            mean_q=np.ascontiguousarray(mean_q),
            adc_magnitude_db=np.ascontiguousarray(adc_magnitude_db),
            magnitude_db=np.ascontiguousarray(displayed_magnitude_db),
            phase_unwrapped_deg=np.ascontiguousarray(phase),
            sample_rate_hz=_require_finite(
                sample_rate_hz, "sample_rate_hz", positive=True
            ),
            reserved_physical_words=reserved_physical_words,
            output_power_dbm=calibrated_power,
            nominal_gain_code=nominal_gain,
            frequency_gain_codes=(
                None if applied_gains is None else np.ascontiguousarray(applied_gains)
            ),
            actual_output_powers_dbm=(
                None if actual_output is None else np.ascontiguousarray(actual_output)
            ),
            input_powers_dbm=(
                None if actual_input is None else np.ascontiguousarray(actual_input)
            ),
        )

    @property
    def sample_count(self) -> int:
        return int(self.iq_traces.shape[1])

    @property
    def calibrated(self) -> bool:
        return self.output_power_dbm is not None

    @property
    def physical_power_calibrated(self) -> bool:
        return (
            self.actual_output_powers_dbm is not None
            and self.input_powers_dbm is not None
        )

    @property
    def dut_input_powers_dbm(self) -> Optional[np.ndarray]:
        """Power delivered to the DUT input reference plane."""
        return self.actual_output_powers_dbm

    @property
    def dut_output_powers_dbm(self) -> Optional[np.ndarray]:
        """Power leaving the DUT output reference plane."""
        return self.input_powers_dbm


@dataclass(frozen=True)
class SParameterPowerSweepResult:
    """Completed gain points from a software power sweep."""

    power_gains: np.ndarray
    requested_frequencies_mhz: np.ndarray
    frequencies_mhz: np.ndarray
    iq_traces: np.ndarray
    mean_i: np.ndarray
    mean_q: np.ndarray
    adc_magnitude_db: np.ndarray
    magnitude_db: np.ndarray
    phase_unwrapped_deg: np.ndarray
    sample_rate_hz: float
    reserved_physical_words: Tuple[Optional[int], ...] = ()
    output_powers_dbm: Optional[np.ndarray] = None
    frequency_gain_codes: Optional[np.ndarray] = None
    actual_output_powers_dbm: Optional[np.ndarray] = None
    input_powers_dbm: Optional[np.ndarray] = None

    @classmethod
    def from_iq(
        cls,
        power_gains: Any,
        requested_frequencies_mhz: Any,
        frequencies_mhz: Any,
        iq_traces: Any,
        *,
        sample_rate_hz: float = 1_000_000.0,
        reserved_physical_words: Sequence[Optional[int]] = (),
        output_powers_dbm: Optional[Any] = None,
        frequency_gain_codes: Optional[Any] = None,
        actual_output_powers_dbm: Optional[Any] = None,
        input_powers_dbm: Optional[Any] = None,
    ) -> "SParameterPowerSweepResult":
        gains = np.asarray(power_gains, dtype=np.int64).reshape(-1)
        requested = np.asarray(requested_frequencies_mhz, dtype=float).reshape(-1)
        frequencies = np.asarray(frequencies_mhz, dtype=float).reshape(-1)
        iq = np.asarray(iq_traces)
        if iq.ndim != 4 or iq.shape[-1] != 2:
            raise ValueError(
                "power-sweep IQ must have shape (power, frequency, sample, 2)"
            )
        if iq.shape[:2] != (gains.size, frequencies.size):
            raise ValueError("power/frequency axes do not match the IQ data")
        if requested.size != frequencies.size:
            raise ValueError("requested and actual frequency counts do not match")
        if iq.shape[2] < 1:
            raise ValueError("every power/frequency point needs at least one sample")
        mean = iq.astype(np.float64).mean(axis=2)
        mean_i = mean[:, :, 0]
        mean_q = mean[:, :, 1]
        magnitude = np.hypot(mean_i, mean_q)
        phase = np.degrees(np.unwrap(np.angle(mean_i + 1j * mean_q), axis=1))
        reserved = tuple(reserved_physical_words)
        if reserved and len(reserved) != gains.size:
            raise ValueError("reserved DDR word counts must match power points")
        powers = None
        applied_gains = None
        if output_powers_dbm is not None:
            powers = np.asarray(output_powers_dbm, dtype=float).reshape(-1)
            if powers.size != gains.size or not np.all(np.isfinite(powers)):
                raise ValueError("output power axis must match power points")
            if frequency_gain_codes is None:
                raise ValueError(
                    "frequency-compensated powers require per-frequency gain codes"
                )
        if frequency_gain_codes is not None:
            applied_gains = np.asarray(frequency_gain_codes, dtype=np.int64)
            if applied_gains.shape != (gains.size, frequencies.size):
                raise ValueError(
                    "frequency gain codes must have shape (power, frequency)"
                )
        adc_magnitude_db = 20.0 * np.log10(
            np.maximum(magnitude, np.finfo(np.float64).tiny)
        )
        actual_output = None
        actual_input = None
        if actual_output_powers_dbm is not None:
            actual_output = np.asarray(actual_output_powers_dbm, dtype=float)
            if actual_output.shape != (gains.size, frequencies.size) or not np.all(
                np.isfinite(actual_output)
            ):
                raise ValueError(
                    "actual output powers must have shape (power, frequency)"
                )
        if input_powers_dbm is not None:
            actual_input = np.asarray(input_powers_dbm, dtype=float)
            if actual_input.shape != (gains.size, frequencies.size) or not np.all(
                np.isfinite(actual_input)
            ):
                raise ValueError("input powers must have shape (power, frequency)")
            if actual_output is None:
                raise ValueError("input powers require actual output powers")
        return cls(
            power_gains=np.ascontiguousarray(gains),
            requested_frequencies_mhz=np.ascontiguousarray(requested),
            frequencies_mhz=np.ascontiguousarray(frequencies),
            iq_traces=np.ascontiguousarray(iq),
            mean_i=np.ascontiguousarray(mean_i),
            mean_q=np.ascontiguousarray(mean_q),
            adc_magnitude_db=np.ascontiguousarray(adc_magnitude_db),
            magnitude_db=np.ascontiguousarray(
                actual_input - actual_output
                if actual_input is not None
                else adc_magnitude_db
            ),
            phase_unwrapped_deg=np.ascontiguousarray(phase),
            sample_rate_hz=_require_finite(
                sample_rate_hz, "sample_rate_hz", positive=True
            ),
            reserved_physical_words=reserved,
            output_powers_dbm=(
                None if powers is None else np.ascontiguousarray(powers)
            ),
            frequency_gain_codes=(
                None if applied_gains is None else np.ascontiguousarray(applied_gains)
            ),
            actual_output_powers_dbm=(
                None if actual_output is None else np.ascontiguousarray(actual_output)
            ),
            input_powers_dbm=(
                None if actual_input is None else np.ascontiguousarray(actual_input)
            ),
        )

    @classmethod
    def from_sweeps(
        cls,
        power_gains: Sequence[int],
        sweeps: Sequence[SParameterSweepResult],
    ) -> "SParameterPowerSweepResult":
        if not sweeps:
            raise ValueError("at least one completed power sweep is required")
        reference = sweeps[0]
        for result in sweeps[1:]:
            if not np.array_equal(
                result.requested_frequencies_mhz,
                reference.requested_frequencies_mhz,
            ) or not np.array_equal(
                result.frequencies_mhz,
                reference.frequencies_mhz,
            ):
                raise ValueError("all power points must share one frequency axis")
            if result.iq_traces.shape != reference.iq_traces.shape:
                raise ValueError("all power points must share one IQ trace shape")
            if result.sample_rate_hz != reference.sample_rate_hz:
                raise ValueError("all power points must share one sample rate")
        return cls.from_iq(
            power_gains,
            reference.requested_frequencies_mhz,
            reference.frequencies_mhz,
            np.stack([result.iq_traces for result in sweeps], axis=0),
            sample_rate_hz=reference.sample_rate_hz,
            reserved_physical_words=[
                result.reserved_physical_words for result in sweeps
            ],
            output_powers_dbm=(
                [float(result.output_power_dbm) for result in sweeps]
                if all(result.output_power_dbm is not None for result in sweeps)
                else None
            ),
            frequency_gain_codes=(
                np.stack([result.frequency_gain_codes for result in sweeps], axis=0)
                if all(result.frequency_gain_codes is not None for result in sweeps)
                else None
            ),
            actual_output_powers_dbm=(
                np.stack(
                    [result.actual_output_powers_dbm for result in sweeps],
                    axis=0,
                )
                if all(result.actual_output_powers_dbm is not None for result in sweeps)
                else None
            ),
            input_powers_dbm=(
                np.stack([result.input_powers_dbm for result in sweeps], axis=0)
                if all(result.input_powers_dbm is not None for result in sweeps)
                else None
            ),
        )

    @property
    def power_count(self) -> int:
        return int(self.power_gains.size)

    @property
    def sample_count(self) -> int:
        return int(self.iq_traces.shape[2])

    @property
    def calibrated(self) -> bool:
        return self.output_powers_dbm is not None

    @property
    def physical_power_calibrated(self) -> bool:
        return (
            self.actual_output_powers_dbm is not None
            and self.input_powers_dbm is not None
        )

    @property
    def dut_input_powers_dbm(self) -> Optional[np.ndarray]:
        return self.actual_output_powers_dbm

    @property
    def dut_output_powers_dbm(self) -> Optional[np.ndarray]:
        return self.input_powers_dbm


def apply_power_calibration(
    result: SParameterSweepResult,
    *,
    output_calibration: Any,
    input_calibration: Optional[InputPowerCalibration],
    output_att1_db: float,
    output_att2_db: float,
    input_attenuation_db: float,
    input_gain_db: float = 0.0,
    loss1_db: float = 0.0,
    loss2_db: float = 0.0,
    amplifier_gain_db: float = 0.0,
) -> SParameterSweepResult:
    """Attach DUT-plane powers and derive calibrated transmission.

    Calibration first recovers powers at the RF-board connectors.  The path
    terms then move those values to the DUT planes::

        P_DUT_IN  = P_RF_OUT - LOSS1
        P_DUT_OUT = P_RF_IN + LOSS2 - AMP_GAIN
        S21_DB    = P_DUT_OUT - P_DUT_IN

    ``actual_output_powers_dbm`` and ``input_powers_dbm`` retain their legacy
    field names but hold ``P_DUT_IN`` and ``P_DUT_OUT`` respectively.  Raw ADC
    magnitude remains available in ``adc_magnitude_db``.
    """
    if result.frequency_gain_codes is None:
        raise ValueError(
            "physical power calibration requires one applied gain per frequency"
        )
    connector_output = output_calibration.output_power_dbm(
        result.frequencies_mhz,
        result.frequency_gain_codes,
        output_att1_db=output_att1_db,
        output_att2_db=output_att2_db,
    )
    dut_input = np.asarray(connector_output, dtype=float) - float(loss1_db)
    dut_output = None
    if input_calibration is not None:
        connector_input = input_calibration.input_power_dbm(
            result.frequencies_mhz,
            result.adc_magnitude_db,
            input_attenuation_db=input_attenuation_db,
            input_gain_db=input_gain_db,
        )
        dut_output = (
            np.asarray(connector_input, dtype=float)
            + float(loss2_db)
            - float(amplifier_gain_db)
        )
    return SParameterSweepResult.from_iq(
        result.requested_frequencies_mhz,
        result.frequencies_mhz,
        result.iq_traces,
        sample_rate_hz=result.sample_rate_hz,
        reserved_physical_words=result.reserved_physical_words,
        output_power_dbm=result.output_power_dbm,
        nominal_gain_code=result.nominal_gain_code,
        frequency_gain_codes=result.frequency_gain_codes,
        actual_output_powers_dbm=dut_input,
        input_powers_dbm=dut_output,
    )


def _signed_modular_delta(first: int, second: int, width: int) -> int:
    modulus = 1 << int(width)
    delta = (int(second) - int(first)) % modulus
    if delta >= modulus // 2:
        delta -= modulus
    return delta


def _common_frequency_quantum_from_channels(*channels: Mapping[str, Any]) -> float:
    """Reproduce QICK's common DDS step when top-level refclk is absent."""
    if not channels:
        raise ValueError("at least one DDS channel is required")
    reference_clocks = [
        float(channel["f_dds"]) * int(channel["fdds_div"]) / int(channel["fs_mult"])
        for channel in channels
    ]
    if not np.allclose(
        reference_clocks,
        reference_clocks[0],
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "generator and readout DDS clocks do not share one reference"
        )
    max_div = int(np.lcm.reduce([int(channel["fdds_div"]) for channel in channels]))
    max_bits = max(int(channel["b_dds"]) for channel in channels)
    multipliers = [
        int(channel["fs_mult"])
        * (max_div // int(channel["fdds_div"]))
        * (1 << (max_bits - int(channel["b_dds"])))
        for channel in channels
    ]
    multiplier_lcm = int(np.lcm.reduce(multipliers))
    return reference_clocks[0] * multiplier_lcm / max_div / (1 << max_bits)


def _frequency_to_register(frequency_mhz: float, channel: Mapping[str, Any]) -> int:
    width = int(channel["b_dds"])
    register = round(float(frequency_mhz) * (1 << width) / float(channel["f_dds"]))
    return int(register) % (1 << width)


class SParameterSweepProgram(RAveragerProgram):
    """ASM v1 RF frequency sweep using one tProcessor register-add loop."""

    COUNTER_ADDR = 1
    PERIODIC_WORD_CYCLES = 3
    STOP_WORD_CYCLES = 3
    READOUT_PERIOD_CYCLES = 3

    def __init__(
        self,
        soccfg,
        sweep: SParameterSweepConfig,
        *,
        tproc_mhz: Optional[float] = None,
    ):
        if _QICK_IMPORT_ERROR is not None:
            raise RuntimeError("QICK is required for RF S-parameter sweeps") from (
                _QICK_IMPORT_ERROR
            )
        if not isinstance(sweep, SParameterSweepConfig):
            raise TypeError("sweep must be an SParameterSweepConfig")
        self.sweep = sweep
        hwh_tproc = float(soccfg["tprocs"][0]["f_time"])
        self.tproc_mhz = (
            hwh_tproc
            if tproc_mhz is None
            else _require_finite(tproc_mhz, "tproc_mhz", positive=True)
        )
        super().__init__(
            soccfg,
            {
                "reps": 1,
                "expts": sweep.frequency_points,
                "start": sweep.frequency_start_mhz,
                "step": (sweep.frequency_end_mhz - sweep.frequency_start_mhz)
                / (sweep.frequency_points - 1),
            },
        )

    def _validate_hardware(self) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        sweep = self.sweep
        if sweep.output_ch >= len(self.soccfg["gens"]):
            raise IndexError("RF output channel is out of range")
        if sweep.readout_ch >= len(self.soccfg["readouts"]):
            raise IndexError("RF readout channel is out of range")
        gen_cfg = self.soccfg["gens"][sweep.output_ch]
        ro_cfg = self.soccfg["readouts"][sweep.readout_ch]
        if (
            gen_cfg.get("type") == "axis_awg_tuning_v1"
            or gen_cfg.get("gen_type") == "awg_tuning"
        ):
            raise ValueError(
                "RF S-parameter output requires a normal DDS signal generator, "
                "not axis_awg_tuning_v1"
            )
        if not gen_cfg.get("has_dds", True):
            raise ValueError("selected RF output channel does not have a DDS")
        hardware_max = int(gen_cfg.get("maxv", MAX_RF_OUTPUT_GAIN))
        effective_max = min(MAX_RF_OUTPUT_GAIN, hardware_max)
        if self.sweep.gain > effective_max:
            raise ValueError(
                f"gain {self.sweep.gain} exceeds channel {self.sweep.output_ch} "
                f"limit {effective_max}"
            )
        self._gain_table = np.asarray(
            self.sweep.calibrated_gain_table, dtype=np.int32
        ).reshape(-1)
        self._uses_gain_table = bool(self._gain_table.size)
        if self._uses_gain_table:
            if np.any(self._gain_table > effective_max):
                raise ValueError(
                    "calibrated gain table exceeds the selected generator limit "
                    f"{effective_max}"
                )
            dmem_size = int(self.tproccfg.get("dmem_size", 0))
            table_end = int(self.sweep.gain_dmem_base_address) + self._gain_table.size
            if dmem_size < 1 or table_end > dmem_size:
                raise RuntimeError(
                    f"calibrated gain table uses DMEM addresses "
                    f"{self.sweep.gain_dmem_base_address}..{table_end - 1}, "
                    f"but tProcessor DMEM depth is {dmem_size}"
                )
        try:
            ddr_cfg = self.soccfg["ddr4_buf"]
        except KeyError as exc:
            raise RuntimeError("soccfg does not contain a DDR sample buffer") from exc
        if not ddr_cfg.get("sample_capture", False):
            raise RuntimeError("S-parameter sweep requires sample-triggered DDR")
        if not ddr_cfg.get("fir_enabled", False):
            raise RuntimeError("S-parameter sweep requires the FIR DDR path")
        output_rate = float(
            ddr_cfg.get(
                "fir_output_fs_mhz",
                ddr_cfg.get("fir_sample_rate_msps", 0.0),
            )
        )
        if not np.isclose(output_rate, 1.0, rtol=0.0, atol=1.0e-9):
            raise RuntimeError(
                f"S-parameter sweep requires 1 MSPS FIR output, got "
                f"{output_rate:.9g} MSPS"
            )
        self._ddr_cfg = ddr_cfg
        self.fir_output_rate_msps = output_rate
        return gen_cfg, ro_cfg

    def _frequency_grid(self, gen_cfg, ro_cfg) -> None:
        requested = self.sweep.requested_frequencies_mhz
        try:
            common_quantum = float(self.soccfg.calc_fstep([gen_cfg, ro_cfg]))
        except (AttributeError, KeyError):
            common_quantum = _common_frequency_quantum_from_channels(gen_cfg, ro_cfg)
        start = round(float(requested[0]) / common_quantum) * common_quantum
        requested_step = float(requested[1] - requested[0])
        step = round(requested_step / common_quantum) * common_quantum
        if step == 0.0:
            raise ValueError(
                "frequency increment is below the common generator/readout DDS "
                f"resolution ({common_quantum:.12g} MHz)"
            )
        frequencies = start + np.arange(self.sweep.frequency_points) * step
        self.requested_frequencies_mhz = requested
        self.frequencies_mhz = np.ascontiguousarray(frequencies, dtype=float)
        self.frequency_step_mhz = float(step)
        self.frequency_quantum_mhz = common_quantum

        try:
            gen_first = self.freq2reg(
                float(frequencies[0]),
                gen_ch=self.sweep.output_ch,
                ro_ch=self.sweep.readout_ch,
            )
            gen_second = self.freq2reg(
                float(frequencies[1]),
                gen_ch=self.sweep.output_ch,
                ro_ch=self.sweep.readout_ch,
            )
            ro_first = self.freq2reg_adc(
                float(frequencies[0]),
                ro_ch=self.sweep.readout_ch,
                gen_ch=self.sweep.output_ch,
            )
            ro_second = self.freq2reg_adc(
                float(frequencies[1]),
                ro_ch=self.sweep.readout_ch,
                gen_ch=self.sweep.output_ch,
            )
        except KeyError as exc:
            if exc.args != ("refclk_freq",):
                raise
            gen_first = _frequency_to_register(frequencies[0], gen_cfg)
            gen_second = _frequency_to_register(frequencies[1], gen_cfg)
            ro_first = _frequency_to_register(frequencies[0], ro_cfg)
            ro_second = _frequency_to_register(frequencies[1], ro_cfg)
        self._gen_frequency_first = int(gen_first)
        self._ro_frequency_first = int(ro_first)
        self._gen_frequency_delta = _signed_modular_delta(
            gen_first, gen_second, int(gen_cfg["b_dds"])
        )
        self._ro_frequency_delta = _signed_modular_delta(
            ro_first, ro_second, int(ro_cfg["b_dds"])
        )

    def _allocate_step_register(self, page: int) -> int:
        occupied = {0}
        for register_map in (self._gen_regmap, self._ro_regmap):
            for reg_page, register in register_map.values():
                if int(reg_page) == int(page):
                    occupied.add(int(register))
        occupied.update(self._step_registers_by_page.get(int(page), ()))
        if int(page) == 0:
            occupied.update((13, 14, 15, 16))
        for register in range(1, 32):
            if register not in occupied:
                self._step_registers_by_page.setdefault(int(page), set()).add(register)
                return register
        raise RuntimeError(f"no tProcessor work register is available on page {page}")

    def initialize(self) -> None:
        self.tproccfg = dict(self.tproccfg)
        self.tproccfg["f_time"] = self.tproc_mhz
        gen_cfg, ro_cfg = self._validate_hardware()
        self._frequency_grid(gen_cfg, ro_cfg)

        self.declare_gen(
            ch=self.sweep.output_ch,
            nqz=self.sweep.nqz,
        )
        self.scan_samples = max(
            1,
            int(ceil(self.sweep.scan_time_us * self.fir_output_rate_msps)),
        )
        monitor_length = min(
            self.scan_samples,
            int(ro_cfg.get("buf_maxlen", self.scan_samples)),
        )
        self.declare_readout(
            ch=self.sweep.readout_ch,
            length=max(1, monitor_length),
        )

        decimation = int(self._ddr_cfg.get("fir_decimation", 300))
        group_delay = int(
            ceil(float(self._ddr_cfg.get("fir_group_delay_input_samples", 8677)))
        )
        input_fs_mhz = float(self._ddr_cfg.get("fir_input_fs_mhz", 300.0))
        if decimation < 1 or input_fs_mhz <= 0.0:
            raise RuntimeError("invalid FIR decimation metadata")
        self.fir_decimation = decimation
        self.fir_group_delay_input_samples = group_delay
        self.fir_input_fs_mhz = input_fs_mhz
        self.fir_warmup_tproc_cycles = int(
            ceil(group_delay * self.tproc_mhz / input_fs_mhz)
        )
        self.fir_feed_input_samples = (
            self.scan_samples * decimation
            + group_delay
            + self.sweep.margin_input_samples
        )
        capture_cycles = int(
            ceil(self.fir_feed_input_samples * self.tproc_mhz / input_fs_mhz)
        )

        gen_port = int(gen_cfg["tproc_ch"])
        ro_port = int(ro_cfg["tproc_ctrl"])
        self.readout_command_time = 0
        self.output_command_time = 1 if gen_port == ro_port else 0
        self.ddr_trigger_time = self.output_command_time + self.fir_warmup_tproc_cycles
        self.capture_end_tproc_cycles = self.output_command_time + capture_cycles
        self.rf_stop_command_time = self.capture_end_tproc_cycles
        self.point_period_tproc_cycles = (
            self.rf_stop_command_time + self.sweep.recovery_tproc_cycles
        )

        self.default_pulse_registers(
            ch=self.sweep.output_ch,
            freq=self._gen_frequency_first,
            phase=0,
        )
        self.set_readout_registers(
            ch=self.sweep.readout_ch,
            freq=self._ro_frequency_first,
            length=self.READOUT_PERIOD_CYCLES,
            mode="periodic",
            phrst=0,
        )

        self._gen_frequency_page = self.ch_page(self.sweep.output_ch)
        self._gen_frequency_register = self.sreg(self.sweep.output_ch, "freq")
        self._ro_frequency_page = self.ch_page_ro(self.sweep.readout_ch)
        self._ro_frequency_register = self.sreg_ro(self.sweep.readout_ch, "freq")
        self._step_registers_by_page = {}
        self._gen_step_register = self._allocate_step_register(self._gen_frequency_page)
        self._ro_step_register = self._allocate_step_register(self._ro_frequency_page)
        self.safe_regwi(
            self._gen_frequency_page,
            self._gen_step_register,
            self._gen_frequency_delta,
            "RF generator DDS frequency step",
        )
        self.safe_regwi(
            self._ro_frequency_page,
            self._ro_step_register,
            self._ro_frequency_delta,
            "RF readout DDS frequency step",
        )
        self._gen_gain_register = self.sreg(self.sweep.output_ch, "gain")
        self._gain_dmem_address_register = None
        self._gain_dmem_error_register = None
        self._gain_dmem_threshold_register = None
        if self._uses_gain_table:
            self._gain_dmem_address_register = self._allocate_step_register(
                self._gen_frequency_page
            )
            self.safe_regwi(
                self._gen_frequency_page,
                self._gain_dmem_address_register,
                int(self.sweep.gain_dmem_base_address),
                "calibrated gain DMEM address",
            )
            if self._gain_table.size < self.sweep.frequency_points:
                self._gain_dmem_error_register = self._allocate_step_register(
                    self._gen_frequency_page
                )
                self._gain_dmem_threshold_register = self._allocate_step_register(
                    self._gen_frequency_page
                )
                self.safe_regwi(
                    self._gen_frequency_page,
                    self._gain_dmem_error_register,
                    0,
                    "calibrated gain nearest-point phase",
                )
                self.safe_regwi(
                    self._gen_frequency_page,
                    self._gain_dmem_threshold_register,
                    self.sweep.frequency_points,
                    "calibrated gain frequency-point count",
                )
        self.synci(128, "allow RF/readout register initialization")

    def body(self) -> None:
        self.readout(self.sweep.readout_ch, t=self.readout_command_time)

        self.set_pulse_registers(
            ch=self.sweep.output_ch,
            style="const",
            gain=self.sweep.gain,
            length=self.PERIODIC_WORD_CYCLES,
            phrst=0,
            stdysel="zero",
            mode="periodic",
        )
        if self._uses_gain_table:
            self.memr(
                self._gen_frequency_page,
                self._gen_gain_register,
                self._gain_dmem_address_register,
                "load frequency-calibrated gain from DMEM",
            )
        self.pulse(self.sweep.output_ch, t=self.output_command_time)
        self.trigger(
            ddr4=True,
            adc_trig_offset=self.ddr_trigger_time,
            t=0,
            width=self.sweep.trigger_width_tproc_cycles,
        )

        self.set_pulse_registers(
            ch=self.sweep.output_ch,
            style="const",
            gain=0,
            length=self.STOP_WORD_CYCLES,
            phrst=0,
            stdysel="zero",
            mode="oneshot",
        )
        self.pulse(self.sweep.output_ch, t=self.rf_stop_command_time)
        self.synci(
            self.point_period_tproc_cycles,
            "wait for FIR DDR capture before next frequency",
        )

    def update(self) -> None:
        self.math(
            self._gen_frequency_page,
            self._gen_frequency_register,
            self._gen_frequency_register,
            "+",
            self._gen_step_register,
            "advance RF generator frequency",
        )
        self.math(
            self._ro_frequency_page,
            self._ro_frequency_register,
            self._ro_frequency_register,
            "+",
            self._ro_step_register,
            "advance RF readout frequency",
        )
        if self._uses_gain_table:
            if self._gain_table.size == self.sweep.frequency_points:
                self.mathi(
                    self._gen_frequency_page,
                    self._gain_dmem_address_register,
                    self._gain_dmem_address_register,
                    "+",
                    1,
                    "advance calibrated gain DMEM address",
                )
            else:
                no_advance_label = "CAL_GAIN_KEEP_DMEM_ADDRESS"
                self.mathi(
                    self._gen_frequency_page,
                    self._gain_dmem_error_register,
                    self._gain_dmem_error_register,
                    "+",
                    int(self._gain_table.size),
                    "advance calibrated gain nearest-point phase",
                )
                self.condj(
                    self._gen_frequency_page,
                    self._gain_dmem_error_register,
                    "<",
                    self._gain_dmem_threshold_register,
                    no_advance_label,
                    "reuse adjacent calibrated gain when phase has not wrapped",
                )
                self.math(
                    self._gen_frequency_page,
                    self._gain_dmem_error_register,
                    self._gain_dmem_error_register,
                    "-",
                    self._gain_dmem_threshold_register,
                    "wrap calibrated gain nearest-point phase",
                )
                self.mathi(
                    self._gen_frequency_page,
                    self._gain_dmem_address_register,
                    self._gain_dmem_address_register,
                    "+",
                    1,
                    "advance compressed calibrated gain DMEM address",
                )
                self.label(no_advance_label)
                self.mathi(
                    self._gen_frequency_page,
                    self._gain_dmem_error_register,
                    self._gain_dmem_error_register,
                    "+",
                    0,
                    "calibrated gain mapping join",
                )

    def _load_runtime_dmem(self, soc) -> None:
        if hasattr(soc, "reload_mem"):
            soc.reload_mem()
        if self._uses_gain_table:
            soc.load_mem(
                np.ascontiguousarray(self._gain_table, dtype=np.int32),
                mem_sel="dmem",
                addr=int(self.sweep.gain_dmem_base_address),
            )

    def _run_with_counter_progress(
        self,
        soc,
        progress_callback: Callable[[int, int], None],
        *,
        poll_interval_seconds: float = 0.02,
    ) -> None:
        total = int(np.prod(self.loop_dims, dtype=np.int64))
        expected_seconds = (
            total * self.point_period_tproc_cycles / (self.tproc_mhz * 1_000_000.0)
        )
        timeout_seconds = max(30.0, 5.0 * expected_seconds + 10.0)
        deadline = time.monotonic() + timeout_seconds
        self.config_all(soc, load_envelopes=True, load_mem=False)
        progress_callback(0, total)
        self._load_runtime_dmem(soc)
        soc.clear_tproc_counter(addr=self.counter_addr)
        soc.start_src("internal")
        soc.start_tproc()
        try:
            completed = 0
            while completed < total:
                completed = min(
                    total,
                    int(soc.get_tproc_counter(addr=self.counter_addr)),
                )
                progress_callback(completed, total)
                if completed < total:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "RF S-parameter hardware sweep timed out after "
                            f"{timeout_seconds:.1f} s at {completed}/{total} points"
                        )
                    time.sleep(poll_interval_seconds)
        finally:
            soc.start_src("internal")

    def acquire_fir_ddr(
        self,
        soc,
        *,
        progress: bool = False,
        counter_progress: Optional[Callable[[int, int], None]] = None,
    ) -> SParameterSweepResult:
        n_triggers = self.sweep.frequency_points
        reserved = soc.arm_ddr4_fir_samples(
            ch=self.sweep.readout_ch,
            n_samples=self.scan_samples,
            n_triggers=n_triggers,
            address=self.sweep.address,
            stride_bytes=self.sweep.stride_bytes,
            force_overwrite=self.sweep.force_overwrite,
        )
        if counter_progress is None and not self._uses_gain_table:
            self.run_rounds(soc, progress=progress)
        else:
            self._run_with_counter_progress(
                soc,
                counter_progress or (lambda _completed, _total: None),
            )
        if self.sweep.settle_seconds:
            time.sleep(self.sweep.settle_seconds)
        raw = np.asarray(
            soc.get_ddr4_fir_samples(
                n_samples=self.scan_samples,
                n_triggers=n_triggers,
                start=self.sweep.address,
                stride_bytes=self.sweep.stride_bytes,
            )
        )
        expected = (n_triggers * self.scan_samples, 2)
        if raw.shape != expected:
            raise RuntimeError(
                f"unexpected FIR DDR IQ shape {raw.shape}; expected {expected}"
            )
        return SParameterSweepResult.from_iq(
            self.requested_frequencies_mhz,
            self.frequencies_mhz,
            raw.reshape(n_triggers, self.scan_samples, 2),
            sample_rate_hz=self.fir_output_rate_msps * 1_000_000.0,
            reserved_physical_words=reserved,
            output_power_dbm=self.sweep.calibrated_output_power_dbm,
            nominal_gain_code=self.sweep.calibrated_nominal_gain_code,
            frequency_gain_codes=(
                None if not self._uses_gain_table else self._expanded_gain_codes()
            ),
        )

    def _expanded_gain_codes(self) -> np.ndarray:
        if not self._uses_gain_table:
            return np.full(
                self.sweep.frequency_points,
                self.sweep.gain,
                dtype=np.int64,
            )
        points = np.arange(self.sweep.frequency_points, dtype=np.int64)
        indices = np.minimum(
            self._gain_table.size - 1,
            (points * self._gain_table.size) // self.sweep.frequency_points,
        )
        return np.ascontiguousarray(self._gain_table[indices], dtype=np.int64)

    def get_expt_pts(self) -> np.ndarray:
        return self.frequencies_mhz.copy()

    def summary(self) -> Mapping[str, Any]:
        return {
            "measurement": "rf_s_parameter",
            "awg_tuning_used": False,
            "sweep_execution": "tproc_hardware_register_add",
            "output_ch": self.sweep.output_ch,
            "readout_ch": self.sweep.readout_ch,
            "frequency_points": self.sweep.frequency_points,
            "requested_frequencies_mhz": self.requested_frequencies_mhz.copy(),
            "actual_frequencies_mhz": self.frequencies_mhz.copy(),
            "frequency_step_mhz": self.frequency_step_mhz,
            "common_frequency_quantum_mhz": self.frequency_quantum_mhz,
            "gain": self.sweep.gain,
            "gain_limit": MAX_RF_OUTPUT_GAIN,
            "power_calibration_enabled": self.sweep.power_calibration_enabled,
            "calibrated_output_power_dbm": (self.sweep.calibrated_output_power_dbm),
            "calibrated_nominal_gain_code": (self.sweep.calibrated_nominal_gain_code),
            "calibrated_reference_response_dbm": (
                self.sweep.calibrated_reference_response_dbm
            ),
            "calibrated_correction_min_db": (self.sweep.calibrated_correction_min_db),
            "calibrated_correction_max_db": (self.sweep.calibrated_correction_max_db),
            "frequency_compensation_model": (
                "nominal_gain * 10**((weakest_response_db - response_db(f))/20)"
                if self._uses_gain_table
                else None
            ),
            "calibration_output_run_id": self.sweep.calibration_output_run_id,
            "calibration_input_run_id": self.sweep.calibration_input_run_id,
            "calibration_database_path": (
                self.sweep.calibration_database_path
                if self.sweep.power_calibration_enabled
                else None
            ),
            "output_board_type": self.sweep.output_board_type,
            "input_board_type": self.sweep.input_board_type,
            "calibration_output_sample_name": (
                self.sweep.calibration_output_sample_name
            ),
            "calibration_input_sample_name": (self.sweep.calibration_input_sample_name),
            "gain_source": (
                "tproc_dmem_frequency_table"
                if self._uses_gain_table
                else "immediate_gain_register"
            ),
            "gain_dmem_base_address": (
                self.sweep.gain_dmem_base_address if self._uses_gain_table else None
            ),
            "gain_dmem_entry_count": int(self._gain_table.size),
            "gain_frequency_point_count": self.sweep.frequency_points,
            "gain_table_compressed": (
                bool(self._uses_gain_table)
                and self._gain_table.size < self.sweep.frequency_points
            ),
            "scan_time_requested_us": self.sweep.scan_time_us,
            "scan_samples": self.scan_samples,
            "scan_time_actual_us": (self.scan_samples / self.fir_output_rate_msps),
            "fir_output_rate_msps": self.fir_output_rate_msps,
            "fir_decimation": self.fir_decimation,
            "fir_group_delay_input_samples": self.fir_group_delay_input_samples,
            "fir_warmup_tproc_cycles": self.fir_warmup_tproc_cycles,
            "rf_output_mode": "periodic_start_timed_zero_stop",
            "rf_periodic_word_fabric_cycles": self.PERIODIC_WORD_CYCLES,
            "rf_stop_word_fabric_cycles": self.STOP_WORD_CYCLES,
            "rf_stop_command_tproc_cycle": self.rf_stop_command_time,
            "readout_period_fabric_cycles": self.READOUT_PERIOD_CYCLES,
            "point_period_tproc_cycles": self.point_period_tproc_cycles,
            "estimated_hardware_seconds": (
                self.sweep.frequency_points
                * self.point_period_tproc_cycles
                / (self.tproc_mhz * 1_000_000.0)
            ),
        }


def configure_sparameter_rf_board(
    soc, config: SParameterSweepConfig
) -> Mapping[str, Any]:
    """Apply settings through the API matching each selected board type."""
    if config.output_board_type == "RF_Out":
        output_att = soc.rfb_set_gen_rf(
            config.output_ch,
            config.output_att1_db,
            config.output_att2_db,
        )
        actual_att1, actual_att2 = map(float, output_att)
    else:
        soc.rfb_set_gen_dc(config.output_ch)
        actual_att1 = 0.0
        actual_att2 = 0.0
    soc.rfb_set_gen_filter(
        config.output_ch,
        fc=config.output_filter_cutoff_ghz,
        bw=config.output_filter_bandwidth_ghz,
        ftype=config.output_filter_type,
    )
    if config.input_board_type == "RF_In":
        input_att = float(
            soc.rfb_set_ro_rf(
                config.readout_ch,
                config.readout_attenuation_db,
            )
        )
        input_gain = 0.0
    else:
        input_att = 0.0
        input_gain = float(
            soc.rfb_set_ro_dc(config.readout_ch, config.readout_dc_gain_db)
        )
    soc.rfb_set_ro_filter(
        config.readout_ch,
        fc=config.readout_filter_cutoff_ghz,
        bw=config.readout_filter_bandwidth_ghz,
        ftype=config.readout_filter_type,
    )
    return {
        "output": {
            "channel": config.output_ch,
            "board_type": config.output_board_type,
            "requested_att1_db": config.output_att1_db,
            "requested_att2_db": config.output_att2_db,
            "commanded_att1_db": actual_att1,
            "commanded_att2_db": actual_att2,
            "attenuators_present": config.output_has_attenuators,
            "filter_type": config.output_filter_type,
            "filter_cutoff_ghz": config.output_filter_cutoff_ghz,
            "filter_bandwidth_ghz": config.output_filter_bandwidth_ghz,
        },
        "readout": {
            "channel": config.readout_ch,
            "board_type": config.input_board_type,
            "requested_attenuation_db": config.readout_attenuation_db,
            "commanded_attenuation_db": input_att,
            "requested_dc_gain_db": config.readout_dc_gain_db,
            "commanded_dc_gain_db": input_gain,
            "attenuator_present": config.input_has_attenuator,
            "filter_type": config.readout_filter_type,
            "filter_cutoff_ghz": config.readout_filter_cutoff_ghz,
            "filter_bandwidth_ghz": config.readout_filter_bandwidth_ghz,
        },
    }


def build_sparameter_program(
    soccfg,
    config: SParameterSweepConfig,
    *,
    tproc_mhz: Optional[float] = None,
) -> SParameterSweepProgram:
    return SParameterSweepProgram(soccfg, config, tproc_mhz=tproc_mhz)


def _calibration_probe_config(
    config: SParameterSweepConfig,
) -> SParameterSweepConfig:
    return replace(
        config,
        power_calibration_enabled=False,
        power_sweep_enabled=False,
        calibrated_gain_table=(),
        calibrated_frequency_point_count=0,
        calibrated_output_power_dbm=None,
        calibrated_nominal_gain_code=None,
        calibrated_reference_response_dbm=None,
        calibrated_correction_min_db=None,
        calibrated_correction_max_db=None,
        calibration_output_run_id=0,
        calibration_input_run_id=0,
        calibration_output_sample_name="",
        calibration_input_sample_name="",
    )


def _prepare_power_calibration(
    soccfg,
    config: SParameterSweepConfig,
    *,
    tproc_mhz: Optional[float],
) -> Tuple[
    SParameterSweepProgram,
    Any,
    CalibrationRunSummary,
    Optional[InputPowerCalibration],
]:
    """Quantize the frequency grid, then select matching board calibrations."""
    probe = build_sparameter_program(
        soccfg,
        _calibration_probe_config(config),
        tproc_mhz=tproc_mhz,
    )
    catalog = CalibrationDatabase(config.calibration_database_path)
    output_calibration = catalog.output_calibration(
        config.output_board_type,
        probe.frequencies_mhz,
    )
    try:
        input_calibration = catalog.input_calibration(
            config.input_board_type,
            probe.frequencies_mhz,
        )
    except LookupError:
        input_calibration = None
    return (
        probe,
        output_calibration,
        output_calibration.summary,
        input_calibration,
    )


def _build_calibrated_program(
    soccfg,
    config: SParameterSweepConfig,
    *,
    target_power_dbm: float,
    probe: SParameterSweepProgram,
    output_calibration: Any,
    output_run: CalibrationRunSummary,
    input_calibration: Optional[InputPowerCalibration],
    actual_output_att1_db: float,
    actual_output_att2_db: float,
    tproc_mhz: Optional[float],
) -> Tuple[SParameterSweepProgram, GainSchedule]:
    schedule = output_calibration.build_gain_schedule(
        probe.frequencies_mhz,
        target_power_dbm,
        output_att1_db=actual_output_att1_db,
        output_att2_db=actual_output_att2_db,
        max_entries=MAX_DMEM_GAIN_ENTRIES,
        dmem_base_address=GAIN_DMEM_BASE_ADDRESS,
    )
    point_config = config.for_gain_schedule(
        schedule,
        output_run,
        None if input_calibration is None else input_calibration.summary,
    )
    program = build_sparameter_program(
        soccfg,
        point_config,
        tproc_mhz=tproc_mhz,
    )
    if not np.array_equal(program.frequencies_mhz, probe.frequencies_mhz):
        raise RuntimeError("calibrated program frequency grid changed unexpectedly")
    return program, schedule


@dataclass
class StoredSParameterSweep:
    run_id: int
    guid: str
    database_path: Path
    row_count: int
    result: SParameterSweepResult
    dataset: Any = None
    program: Any = None
    rf_settings: Optional[Mapping[str, Any]] = None


def _qcodes_helpers():
    try:
        from .qick_qcodes_experiment import (
            _checkpoint_sqlite_database,
            _json_text,
            _prepare_local_database,
            _publish_local_database,
            connect_qick,
        )
    except ImportError:
        from qick_qcodes_experiment import (
            _checkpoint_sqlite_database,
            _json_text,
            _prepare_local_database,
            _publish_local_database,
            connect_qick,
        )
    return (
        connect_qick,
        _json_text,
        _prepare_local_database,
        _checkpoint_sqlite_database,
        _publish_local_database,
    )


def _config_for_metadata(config: SParameterSweepConfig) -> Mapping[str, Any]:
    """Serialize user settings without duplicating a potentially 4000-word table."""
    payload = asdict(config)
    gain_table = payload.pop("calibrated_gain_table", [])
    payload["calibrated_gain_table_entries"] = len(gain_table)
    return payload


def _power_result_payload(
    result: SParameterPowerSweepResult,
    *,
    planned_power_gains: Sequence[int],
    planned_output_powers_dbm: Sequence[float] = (),
    power_scale: str,
) -> Mapping[str, Any]:
    payload = {
        "schema": (
            "qick-rf-s-parameter-frequency-compensated-power-sweep-v3"
            if result.calibrated
            else "qick-rf-s-parameter-power-sweep-v1"
        ),
        "power_sweep_execution": "python_software_loop",
        "frequency_sweep_execution": "tproc_hardware_register_add",
        "power_scale": power_scale,
        "planned_power_gains": [int(value) for value in planned_power_gains],
        "power_gains": result.power_gains.tolist(),
        "completed_power_points": result.power_count,
        "requested_frequencies_mhz": (result.requested_frequencies_mhz.tolist()),
        "frequencies_mhz": result.frequencies_mhz.tolist(),
        "mean_i": result.mean_i.tolist(),
        "mean_q": result.mean_q.tolist(),
        "adc_magnitude_db": result.adc_magnitude_db.tolist(),
        "magnitude_db": result.magnitude_db.tolist(),
        "phase_unwrapped_deg": result.phase_unwrapped_deg.tolist(),
        "sample_rate_hz": result.sample_rate_hz,
        "iq_shape": list(result.iq_traces.shape),
        "physical_power_calibrated": result.physical_power_calibrated,
    }
    if result.calibrated:
        payload.update(
            {
                "planned_output_powers_dbm": [
                    float(value) for value in planned_output_powers_dbm
                ],
                "output_powers_dbm": result.output_powers_dbm.tolist(),
                "frequency_gain_codes": result.frequency_gain_codes.tolist(),
                "actual_output_powers_dbm": (
                    None
                    if result.actual_output_powers_dbm is None
                    else result.actual_output_powers_dbm.tolist()
                ),
                "input_powers_dbm": (
                    None
                    if result.input_powers_dbm is None
                    else result.input_powers_dbm.tolist()
                ),
                "gain_mapping": (
                    "one nominal gain per software power point multiplied by "
                    "relative frequency-response correction"
                ),
                "dmem_execution": (
                    "one frequency-gain table is loaded for each software power point"
                ),
            }
        )
    return payload


class _SParameterPowerRunWriter:
    """Append and publish one QCoDeS power point at a time."""

    def __init__(
        self,
        *,
        config: SParameterSweepConfig,
        connection_config: Any,
        run_config: Any,
        rf_settings: Mapping[str, Any],
    ):
        self.config = config
        self.connection_config = connection_config
        self.run_config = run_config
        self.rf_settings = rf_settings
        self.calibrated = bool(config.power_calibration_enabled)
        self.planned_power_gains = tuple(
            int(value) for value in ([] if self.calibrated else config.power_gains)
        )
        self.planned_output_powers_dbm = tuple(
            float(value)
            for value in (config.target_powers_dbm if self.calibrated else [])
        )
        self.database_path = (
            Path(
                getattr(
                    run_config,
                    "resolved_database_path",
                    run_config.database_path,
                )
            )
            .expanduser()
            .resolve()
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.staging_directory = None
        self.local_database_path = None
        self.datasaver_context = None
        self.datasaver = None
        self.dataset = None
        self.final_dataset = None
        self.results = []
        self.completed_power_gains = []
        self.completed_output_powers_dbm = []
        self.program_summaries = []
        self.row_count = 0
        self._helpers = None
        self._load_by_guid = None
        self._json_text = None
        self._parameters = None

    def open(self) -> None:
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
                "QCoDeS is required to save S-parameter power sweeps; "
                "install qcodes==0.58.0"
            ) from exc
        (
            _connect_qick,
            json_text,
            prepare_local_database,
            checkpoint_database,
            publish_database,
        ) = _qcodes_helpers()
        self._helpers = (
            initialise_or_create_database_at,
            checkpoint_database,
            publish_database,
        )
        self._load_by_guid = load_by_guid
        self._json_text = json_text
        self.staging_directory, self.local_database_path = prepare_local_database(
            self.database_path
        )
        initialise_or_create_database_at(str(self.local_database_path))
        experiment = load_or_create_experiment(
            self.run_config.experiment_name,
            self.run_config.sample_name,
        )
        measurement = Measurement(exp=experiment, station=Station())
        if self.calibrated:
            power_axis = Parameter(
                OUTPUT_POWER_PARAMETER,
                label="Calibrated RF output power",
                unit="dBm",
            )
        else:
            power_axis = Parameter(
                POWER_GAIN_PARAMETER,
                label="RF output gain code",
                unit="",
            )
        frequency = Parameter(
            FREQUENCY_PARAMETER,
            label="RF frequency",
            unit="MHz",
        )
        sample_index = Parameter(
            SAMPLE_INDEX_PARAMETER,
            label="FIR sample index",
            unit="",
        )
        mean_i = Parameter(MEAN_I_PARAMETER, label="Mean I", unit="ADC units")
        mean_q = Parameter(MEAN_Q_PARAMETER, label="Mean Q", unit="ADC units")
        magnitude_db = Parameter(
            MAGNITUDE_DB_PARAMETER,
            label="S-parameter magnitude",
            unit="dB",
        )
        adc_magnitude_db = Parameter(
            ADC_MAGNITUDE_DB_PARAMETER,
            label="Raw ADC magnitude",
            unit="dB ADC",
        )
        phase_deg = Parameter(
            PHASE_DEG_PARAMETER,
            label="S-parameter unwrapped phase",
            unit="deg",
        )
        i_trace = Parameter(I_TRACE_PARAMETER, label="I trace", unit="ADC units")
        q_trace = Parameter(Q_TRACE_PARAMETER, label="Q trace", unit="ADC units")
        measurement.register_parameter(power_axis)
        measurement.register_parameter(frequency)
        measurement.register_parameter(sample_index, paramtype="array")
        calibrated_gain = None
        actual_output_power = None
        actual_input_power = None
        if self.calibrated:
            calibrated_gain = Parameter(
                CALIBRATED_GAIN_PARAMETER,
                label="Applied frequency-compensated QICK gain",
                unit="",
            )
            measurement.register_parameter(
                calibrated_gain,
                setpoints=(power_axis, frequency),
            )
            actual_output_power = Parameter(
                ACTUAL_OUTPUT_POWER_PARAMETER,
                label="Power at DUT input plane",
                unit="dBm",
            )
            actual_input_power = Parameter(
                ACTUAL_INPUT_POWER_PARAMETER,
                label="Power at DUT output plane",
                unit="dBm",
            )
            measurement.register_parameter(
                actual_output_power,
                setpoints=(power_axis, frequency),
            )
            measurement.register_parameter(
                actual_input_power,
                setpoints=(power_axis, frequency),
            )
        for parameter in (
            mean_i,
            mean_q,
            adc_magnitude_db,
            magnitude_db,
            phase_deg,
        ):
            measurement.register_parameter(
                parameter,
                setpoints=(power_axis, frequency),
            )
        for parameter in (i_trace, q_trace):
            measurement.register_parameter(
                parameter,
                setpoints=(power_axis, frequency, sample_index),
                paramtype="array",
            )
        self._parameters = {
            "power_axis": power_axis,
            "calibrated_gain": calibrated_gain,
            "frequency": frequency,
            "sample_index": sample_index,
            "mean_i": mean_i,
            "mean_q": mean_q,
            "adc_magnitude_db": adc_magnitude_db,
            "magnitude_db": magnitude_db,
            "phase_deg": phase_deg,
            "actual_output_power": actual_output_power,
            "actual_input_power": actual_input_power,
            "i_trace": i_trace,
            "q_trace": q_trace,
        }
        self.datasaver_context = measurement.run(
            write_in_background=False,
            in_memory_cache=False,
        )
        self.datasaver = self.datasaver_context.__enter__()
        self.dataset = self.datasaver.dataset
        if self.run_config.notes:
            self.dataset.add_metadata("experiment_notes", self.run_config.notes)

    def _metadata(self, result: SParameterPowerSweepResult) -> Mapping[str, Any]:
        payload = _power_result_payload(
            result,
            planned_power_gains=self.planned_power_gains,
            planned_output_powers_dbm=self.planned_output_powers_dbm,
            power_scale=self.config.power_scale,
        )
        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "measurement": "rf_s_parameter_power_sweep",
            "awg_tuning_used": False,
            "connection": asdict(self.connection_config),
            "run": {
                **asdict(self.run_config),
                "database_path": str(self.database_path),
            },
            "config": _config_for_metadata(self.config),
            "rf_settings_actual": self.rf_settings,
            "program_summaries": list(self.program_summaries),
            "result": payload,
            "formulas": {
                "mean_i": "mean(i_trace)",
                "mean_q": "mean(q_trace)",
                "adc_magnitude_db": "20*log10(hypot(mean_i, mean_q))",
                "magnitude_db": (
                    "P_DUT_OUT - P_DUT_IN"
                    if result.physical_power_calibrated
                    else "adc_magnitude_db"
                ),
                "phase_unwrapped_deg": (
                    "degrees(unwrap(angle(mean_i + 1j*mean_q), frequency))"
                ),
                "frequency_gain": (
                    "nominal_gain * 10**((weakest_response_db - "
                    "response_db(frequency))/20)"
                ),
            },
            "storage": (
                "one split I/Q array row and scalar response per "
                "power-gain/frequency point"
            ),
            "live_update": "database snapshot published after every software power point",
        }

    def append(
        self,
        power_coordinate: float,
        result: SParameterSweepResult,
        program_summary: Mapping[str, Any],
    ) -> SParameterPowerSweepResult:
        if self.datasaver is None or self.dataset is None:
            raise RuntimeError("power-sweep database writer is not open")
        if self.calibrated:
            coordinate = float(power_coordinate)
            if not any(
                np.isclose(coordinate, planned, rtol=0.0, atol=1.0e-9)
                for planned in self.planned_output_powers_dbm
            ):
                raise ValueError(f"unexpected compensated power {coordinate} dBm")
            if not result.calibrated or result.frequency_gain_codes is None:
                raise ValueError("compensated power sweep result lacks gain mapping")
            if result.nominal_gain_code is None:
                raise ValueError("frequency-compensated result lacks nominal gain")
            nominal_gain = int(result.nominal_gain_code)
        else:
            nominal_gain = int(power_coordinate)
            coordinate = nominal_gain
            if nominal_gain not in self.planned_power_gains:
                raise ValueError(f"unexpected power gain {nominal_gain}")
        sample_values = np.arange(result.sample_count, dtype=np.int32)
        parameters = self._parameters
        for index, frequency_mhz in enumerate(result.frequencies_mhz):
            values = [
                (parameters["power_axis"], coordinate),
                (parameters["frequency"], float(frequency_mhz)),
                (parameters["mean_i"], float(result.mean_i[index])),
                (parameters["mean_q"], float(result.mean_q[index])),
                (
                    parameters["adc_magnitude_db"],
                    float(result.adc_magnitude_db[index]),
                ),
                (
                    parameters["magnitude_db"],
                    float(result.magnitude_db[index]),
                ),
                (
                    parameters["phase_deg"],
                    float(result.phase_unwrapped_deg[index]),
                ),
                (parameters["sample_index"], sample_values),
                (
                    parameters["i_trace"],
                    np.ascontiguousarray(result.iq_traces[index, :, 0]),
                ),
                (
                    parameters["q_trace"],
                    np.ascontiguousarray(result.iq_traces[index, :, 1]),
                ),
            ]
            if self.calibrated:
                values.append(
                    (
                        parameters["calibrated_gain"],
                        int(result.frequency_gain_codes[index]),
                    )
                )
                if result.actual_output_powers_dbm is not None:
                    values.append(
                        (
                            parameters["actual_output_power"],
                            float(result.actual_output_powers_dbm[index]),
                        )
                    )
                if result.input_powers_dbm is not None:
                    values.append(
                        (
                            parameters["actual_input_power"],
                            float(result.input_powers_dbm[index]),
                        )
                    )
            self.datasaver.add_result(*values)
        self.results.append(result)
        self.completed_power_gains.append(nominal_gain)
        if self.calibrated:
            self.completed_output_powers_dbm.append(coordinate)
        self.program_summaries.append(dict(program_summary))
        self.row_count += int(result.frequencies_mhz.size * result.sample_count)
        combined = SParameterPowerSweepResult.from_sweeps(
            self.completed_power_gains,
            self.results,
        )
        metadata = self._metadata(combined)
        self.dataset.add_metadata(
            "sparameter_experiment_json",
            self._json_text(metadata),
        )
        self.dataset.add_metadata(
            "sparameter_result_json",
            self._json_text(metadata["result"]),
        )
        self.datasaver.flush_data_to_database()
        self._helpers[2](self.local_database_path, self.database_path)
        return combined

    def close(self):
        if self.datasaver_context is None or self.dataset is None:
            raise RuntimeError("power-sweep database writer is not open")
        self.datasaver_context.__exit__(None, None, None)
        self.datasaver_context = None
        guid = str(self.dataset.guid)
        self.dataset.conn.close()
        self._helpers[1](self.local_database_path)
        self._helpers[2](self.local_database_path, self.database_path)
        self._helpers[0](str(self.database_path))
        self.final_dataset = self._load_by_guid(guid)
        shutil.rmtree(self.staging_directory)
        self.staging_directory = None
        return self.final_dataset

    def abort(self, exc: BaseException) -> None:
        if self.datasaver_context is not None:
            self.datasaver_context.__exit__(type(exc), exc, exc.__traceback__)
            self.datasaver_context = None
        if self.dataset is not None:
            try:
                self.dataset.conn.close()
            except Exception:
                pass
        if self.staging_directory is not None:
            shutil.rmtree(self.staging_directory, ignore_errors=True)
            self.staging_directory = None


def store_sparameter_result(
    result: SParameterSweepResult,
    *,
    config: SParameterSweepConfig,
    connection_config: Any,
    run_config: Any,
    program_summary: Mapping[str, Any],
    rf_settings: Mapping[str, Any],
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Any, int]:
    """Store scalar response and one split I/Q array row per frequency."""
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
            "QCoDeS is required to save S-parameter sweeps; install qcodes==0.58.0"
        ) from exc
    (
        _connect_qick,
        json_text,
        prepare_local_database,
        checkpoint_database,
        publish_database,
    ) = _qcodes_helpers()

    database_path = (
        Path(getattr(run_config, "resolved_database_path", run_config.database_path))
        .expanduser()
        .resolve()
    )
    database_path.parent.mkdir(parents=True, exist_ok=True)
    staging_directory, local_database_path = prepare_local_database(database_path)
    initialise_or_create_database_at(str(local_database_path))
    experiment = load_or_create_experiment(
        run_config.experiment_name,
        run_config.sample_name,
    )
    measurement = Measurement(exp=experiment, station=Station())
    frequency = Parameter(
        FREQUENCY_PARAMETER,
        label="RF frequency",
        unit="MHz",
    )
    sample_index = Parameter(
        SAMPLE_INDEX_PARAMETER,
        label="FIR sample index",
        unit="",
    )
    mean_i = Parameter(MEAN_I_PARAMETER, label="Mean I", unit="ADC units")
    mean_q = Parameter(MEAN_Q_PARAMETER, label="Mean Q", unit="ADC units")
    magnitude_db = Parameter(
        MAGNITUDE_DB_PARAMETER,
        label="S-parameter magnitude",
        unit="dB",
    )
    adc_magnitude_db = Parameter(
        ADC_MAGNITUDE_DB_PARAMETER,
        label="Raw ADC magnitude",
        unit="dB ADC",
    )
    phase_deg = Parameter(
        PHASE_DEG_PARAMETER,
        label="S-parameter unwrapped phase",
        unit="deg",
    )
    i_trace = Parameter(I_TRACE_PARAMETER, label="I trace", unit="ADC units")
    q_trace = Parameter(Q_TRACE_PARAMETER, label="Q trace", unit="ADC units")
    measurement.register_parameter(frequency)
    measurement.register_parameter(sample_index, paramtype="array")
    calibrated_gain = None
    actual_output_power = None
    actual_input_power = None
    if result.calibrated:
        calibrated_gain = Parameter(
            CALIBRATED_GAIN_PARAMETER,
            label="Applied frequency-compensated QICK gain",
            unit="",
        )
        measurement.register_parameter(calibrated_gain, setpoints=(frequency,))
        actual_output_power = Parameter(
            ACTUAL_OUTPUT_POWER_PARAMETER,
            label="Power at DUT input plane",
            unit="dBm",
        )
        actual_input_power = Parameter(
            ACTUAL_INPUT_POWER_PARAMETER,
            label="Power at DUT output plane",
            unit="dBm",
        )
        measurement.register_parameter(actual_output_power, setpoints=(frequency,))
        measurement.register_parameter(actual_input_power, setpoints=(frequency,))
    for parameter in (
        mean_i,
        mean_q,
        adc_magnitude_db,
        magnitude_db,
        phase_deg,
    ):
        measurement.register_parameter(parameter, setpoints=(frequency,))
    for parameter in (i_trace, q_trace):
        measurement.register_parameter(
            parameter,
            setpoints=(frequency, sample_index),
            paramtype="array",
        )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": "rf_s_parameter",
        "awg_tuning_used": False,
        "connection": asdict(connection_config),
        "run": {
            **asdict(run_config),
            "database_path": str(database_path),
        },
        "config": _config_for_metadata(config),
        "rf_settings_actual": rf_settings,
        "program_summary": program_summary,
        "result": {
            "requested_frequencies_mhz": result.requested_frequencies_mhz.tolist(),
            "frequencies_mhz": result.frequencies_mhz.tolist(),
            "mean_i": result.mean_i.tolist(),
            "mean_q": result.mean_q.tolist(),
            "adc_magnitude_db": result.adc_magnitude_db.tolist(),
            "magnitude_db": result.magnitude_db.tolist(),
            "phase_unwrapped_deg": result.phase_unwrapped_deg.tolist(),
            "sample_rate_hz": result.sample_rate_hz,
            "iq_shape": list(result.iq_traces.shape),
            "output_power_dbm": result.output_power_dbm,
            "nominal_gain_code": result.nominal_gain_code,
            "frequency_gain_codes": (
                None
                if result.frequency_gain_codes is None
                else result.frequency_gain_codes.tolist()
            ),
            "actual_output_powers_dbm": (
                None
                if result.actual_output_powers_dbm is None
                else result.actual_output_powers_dbm.tolist()
            ),
            "input_powers_dbm": (
                None
                if result.input_powers_dbm is None
                else result.input_powers_dbm.tolist()
            ),
            "physical_power_calibrated": result.physical_power_calibrated,
        },
        "formulas": {
            "mean_i": "mean(i_trace)",
            "mean_q": "mean(q_trace)",
            "adc_magnitude_db": "20*log10(hypot(mean_i, mean_q))",
            "magnitude_db": (
                "P_DUT_OUT - P_DUT_IN"
                if result.physical_power_calibrated
                else "adc_magnitude_db"
            ),
            "phase_unwrapped_deg": ("degrees(unwrap(angle(mean_i + 1j*mean_q)))"),
            "frequency_gain": (
                "nominal_gain * 10**((weakest_response_db - "
                "response_db(frequency))/20)"
            ),
        },
        "storage": "one split I/Q array row and scalar response per frequency",
    }

    _emit_progress(progress_callback, 65, "Preparing S-parameter QCoDeS run")
    sample_values = np.arange(result.sample_count, dtype=np.int32)
    with measurement.run(
        write_in_background=False,
        in_memory_cache=False,
    ) as datasaver:
        dataset = datasaver.dataset
        dataset.add_metadata("sparameter_experiment_json", json_text(metadata))
        dataset.add_metadata(
            "sparameter_result_json",
            json_text(metadata["result"]),
        )
        if run_config.notes:
            dataset.add_metadata("experiment_notes", run_config.notes)
        for index, frequency_mhz in enumerate(result.frequencies_mhz):
            values = [
                (frequency, float(frequency_mhz)),
                (mean_i, float(result.mean_i[index])),
                (mean_q, float(result.mean_q[index])),
                (adc_magnitude_db, float(result.adc_magnitude_db[index])),
                (magnitude_db, float(result.magnitude_db[index])),
                (phase_deg, float(result.phase_unwrapped_deg[index])),
                (sample_index, sample_values),
                (i_trace, np.ascontiguousarray(result.iq_traces[index, :, 0])),
                (q_trace, np.ascontiguousarray(result.iq_traces[index, :, 1])),
            ]
            if calibrated_gain is not None:
                values.append(
                    (
                        calibrated_gain,
                        int(result.frequency_gain_codes[index]),
                    )
                )
                if result.actual_output_powers_dbm is not None:
                    values.append(
                        (
                            actual_output_power,
                            float(result.actual_output_powers_dbm[index]),
                        )
                    )
                if result.input_powers_dbm is not None:
                    values.append(
                        (
                            actual_input_power,
                            float(result.input_powers_dbm[index]),
                        )
                    )
            datasaver.add_result(*values)
            percent = 65 + round(25 * (index + 1) / result.frequencies_mhz.size)
            _emit_progress(
                progress_callback,
                percent,
                f"Saving RF sweep point {index + 1}/{result.frequencies_mhz.size}",
            )
        datasaver.flush_data_to_database()

    guid = str(dataset.guid)
    dataset.conn.close()
    _emit_progress(progress_callback, 92, "Checkpointing S-parameter database")
    checkpoint_database(local_database_path)
    _emit_progress(progress_callback, 95, "Publishing S-parameter database")
    publish_database(local_database_path, database_path)
    initialise_or_create_database_at(str(database_path))
    dataset = load_by_guid(guid)
    import shutil

    shutil.rmtree(staging_directory)
    _emit_progress(progress_callback, 99, "S-parameter database saved")
    return dataset, int(result.frequencies_mhz.size * result.sample_count)


def run_sparameter_sweep(
    *,
    connection_config: Any,
    run_config: Any,
    sweep_config: SParameterSweepConfig,
    tproc_mhz: Optional[float] = None,
    connector: Optional[Callable[..., Tuple[Any, Any]]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    partial_callback: Optional[Callable[[StoredSParameterSweep], None]] = None,
) -> StoredSParameterSweep:
    """Run a frequency sweep, optionally inside a software gain sweep."""
    connect_qick, *_helpers = _qcodes_helpers()
    _emit_progress(progress_callback, 0, "Starting independent RF S-parameter sweep")
    _emit_progress(progress_callback, 2, "Connecting to QICK")
    soc, soccfg = connect_qick(connection_config, connector=connector)
    _emit_progress(progress_callback, 5, "Configuring RF output/readout chains")
    rf_settings = configure_sparameter_rf_board(soc, sweep_config)
    calibration_context = None
    if sweep_config.power_calibration_enabled:
        _emit_progress(
            progress_callback,
            6,
            "Selecting board-matched frequency-response calibration",
        )
        calibration_context = _prepare_power_calibration(
            soccfg,
            sweep_config,
            tproc_mhz=tproc_mhz,
        )
        probe, output_calibration, output_run, input_calibration = calibration_context
        input_run = None if input_calibration is None else input_calibration.summary
        rf_settings = {
            **dict(rf_settings),
            "board_selection": {
                "output_board_type": sweep_config.output_board_type,
                "input_board_type": sweep_config.input_board_type,
            },
            "frequency_response_compensation": {
                "database_path": str(
                    Path(sweep_config.calibration_database_path).expanduser().resolve()
                ),
                "output_run": output_run.as_dict(),
                "input_run": None if input_run is None else input_run.as_dict(),
                "input_mapping_applied": input_calibration is not None,
                "s_parameter_formula": (
                    "(P_RF_IN + LOSS2 - AMP_GAIN) - (P_RF_OUT - LOSS1)"
                    if input_calibration is not None
                    else "20*log10(hypot(mean_i, mean_q))"
                ),
                "gain_axis_usage": (
                    "used only to remove linear gain scaling before extracting "
                    "frequency dependence"
                ),
                "dmem_scope": "one gain table for the current software power point",
            },
        }
    output_settings = dict(rf_settings.get("output", {}))
    actual_output_att1_db = float(
        output_settings.get(
            "commanded_att1_db",
            sweep_config.effective_output_att1_db,
        )
    )
    actual_output_att2_db = float(
        output_settings.get(
            "commanded_att2_db",
            sweep_config.effective_output_att2_db,
        )
    )
    readout_settings = dict(rf_settings.get("readout", {}))
    actual_input_attenuation_db = float(
        readout_settings.get(
            "commanded_attenuation_db",
            sweep_config.effective_input_attenuation_db,
        )
    )
    actual_input_gain_db = float(
        readout_settings.get(
            "commanded_dc_gain_db",
            sweep_config.effective_input_dc_gain_db,
        )
    )
    if sweep_config.power_sweep_enabled:
        if sweep_config.power_calibration_enabled:
            power_coordinates = tuple(
                float(value) for value in sweep_config.target_powers_dbm
            )
        else:
            power_coordinates = tuple(int(value) for value in sweep_config.power_gains)
        writer = _SParameterPowerRunWriter(
            config=sweep_config,
            connection_config=connection_config,
            run_config=run_config,
            rf_settings=rf_settings,
        )
        programs = []
        combined_result = None
        _emit_progress(
            progress_callback,
            7,
            f"Preparing live DB run for {len(power_coordinates)} power points",
        )
        writer.open()
        try:
            for power_index, power_coordinate in enumerate(power_coordinates):
                if sweep_config.power_calibration_enabled:
                    probe, output_calibration, output_run, input_calibration = (
                        calibration_context
                    )
                    program, _schedule = _build_calibrated_program(
                        soccfg,
                        sweep_config,
                        target_power_dbm=float(power_coordinate),
                        probe=probe,
                        output_calibration=output_calibration,
                        output_run=output_run,
                        input_calibration=input_calibration,
                        actual_output_att1_db=actual_output_att1_db,
                        actual_output_att2_db=actual_output_att2_db,
                        tproc_mhz=tproc_mhz,
                    )
                    power_label = f"{float(power_coordinate):.6g} dBm"
                else:
                    point_config = sweep_config.for_gain(int(power_coordinate))
                    program = build_sparameter_program(
                        soccfg,
                        point_config,
                        tproc_mhz=tproc_mhz,
                    )
                    power_label = f"gain {int(power_coordinate)}"
                _emit_progress(
                    progress_callback,
                    8 + round(87 * power_index / len(power_coordinates)),
                    (
                        f"Compiling power {power_index + 1}/"
                        f"{len(power_coordinates)} at {power_label}"
                    ),
                )
                programs.append(program)

                def counter_progress(
                    completed: int,
                    total: int,
                    *,
                    _power_index: int = power_index,
                    _power_label: str = power_label,
                ) -> None:
                    del total
                    power_progress = (
                        _power_index + 0.85 * completed / sweep_config.frequency_points
                    )
                    fraction = power_progress / len(power_coordinates)
                    _emit_progress(
                        progress_callback,
                        8 + round(87 * max(0.0, min(1.0, fraction))),
                        (
                            f"Power {_power_index + 1}/{len(power_coordinates)} "
                            f"{_power_label}: frequency {completed}/"
                            f"{sweep_config.frequency_points}"
                        ),
                    )

                result = program.acquire_fir_ddr(
                    soc,
                    counter_progress=(
                        counter_progress if progress_callback is not None else None
                    ),
                )
                if sweep_config.power_calibration_enabled:
                    result = apply_power_calibration(
                        result,
                        output_calibration=output_calibration,
                        input_calibration=input_calibration,
                        output_att1_db=actual_output_att1_db,
                        output_att2_db=actual_output_att2_db,
                        input_attenuation_db=actual_input_attenuation_db,
                        input_gain_db=actual_input_gain_db,
                        loss1_db=sweep_config.loss1_db,
                        loss2_db=sweep_config.loss2_db,
                        amplifier_gain_db=sweep_config.amplifier_gain_db,
                    )
                _emit_progress(
                    progress_callback,
                    8 + round(87 * (power_index + 0.9) / len(power_coordinates)),
                    (
                        f"Saving power {power_index + 1}/"
                        f"{len(power_coordinates)} {power_label}"
                    ),
                )
                combined_result = writer.append(
                    power_coordinate,
                    result,
                    program.summary(),
                )
                partial = StoredSParameterSweep(
                    run_id=int(writer.dataset.run_id),
                    guid=str(writer.dataset.guid),
                    database_path=writer.database_path,
                    row_count=writer.row_count,
                    result=combined_result,
                    program=program,
                    rf_settings=rf_settings,
                )
                if partial_callback is not None:
                    partial_callback(partial)
                _emit_progress(
                    progress_callback,
                    8 + round(87 * (power_index + 1) / len(power_coordinates)),
                    (
                        f"Power {power_index + 1}/{len(power_coordinates)} saved; "
                        "live database updated"
                    ),
                )
            dataset = writer.close()
        except BaseException as exc:
            writer.abort(exc)
            raise
        _emit_progress(progress_callback, 100, "RF power sweep saved")
        return StoredSParameterSweep(
            run_id=int(dataset.run_id),
            guid=str(dataset.guid),
            database_path=writer.database_path,
            row_count=writer.row_count,
            result=combined_result,
            dataset=dataset,
            program=tuple(programs),
            rf_settings=rf_settings,
        )

    _emit_progress(progress_callback, 8, "Compiling hardware frequency sweep")
    if sweep_config.power_calibration_enabled:
        probe, output_calibration, output_run, input_calibration = calibration_context
        program, _schedule = _build_calibrated_program(
            soccfg,
            sweep_config,
            target_power_dbm=float(sweep_config.output_power_dbm),
            probe=probe,
            output_calibration=output_calibration,
            output_run=output_run,
            input_calibration=input_calibration,
            actual_output_att1_db=actual_output_att1_db,
            actual_output_att2_db=actual_output_att2_db,
            tproc_mhz=tproc_mhz,
        )
    else:
        program = build_sparameter_program(
            soccfg,
            sweep_config,
            tproc_mhz=tproc_mhz,
        )

    def counter_progress(completed: int, total: int) -> None:
        fraction = 1.0 if total <= 0 else completed / total
        _emit_progress(
            progress_callback,
            10 + round(50 * max(0.0, min(1.0, fraction))),
            f"RF hardware sweep {completed}/{total} frequency points",
        )

    result = program.acquire_fir_ddr(
        soc,
        counter_progress=(counter_progress if progress_callback is not None else None),
    )
    if sweep_config.power_calibration_enabled:
        result = apply_power_calibration(
            result,
            output_calibration=output_calibration,
            input_calibration=input_calibration,
            output_att1_db=actual_output_att1_db,
            output_att2_db=actual_output_att2_db,
            input_attenuation_db=actual_input_attenuation_db,
            input_gain_db=actual_input_gain_db,
            loss1_db=sweep_config.loss1_db,
            loss2_db=sweep_config.loss2_db,
            amplifier_gain_db=sweep_config.amplifier_gain_db,
        )
    _emit_progress(progress_callback, 62, "Averaging FIR IQ traces")
    dataset, row_count = store_sparameter_result(
        result,
        config=sweep_config,
        connection_config=connection_config,
        run_config=run_config,
        program_summary=program.summary(),
        rf_settings=rf_settings,
        progress_callback=progress_callback,
    )
    _emit_progress(progress_callback, 100, "RF S-parameter sweep saved")
    return StoredSParameterSweep(
        run_id=int(dataset.run_id),
        guid=str(dataset.guid),
        database_path=Path(
            getattr(run_config, "resolved_database_path", run_config.database_path)
        )
        .expanduser()
        .resolve(),
        row_count=row_count,
        result=result,
        dataset=dataset,
        program=program,
        rf_settings=rf_settings,
    )


def _coerce_trace_rows(values: Any, point_count: int) -> np.ndarray:
    array = np.asarray(values)
    array = np.squeeze(array)
    if array.dtype == object:
        array = np.stack([np.asarray(value) for value in array.reshape(-1)])
    if point_count == 1 and array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2 and array.size % point_count == 0:
        array = array.reshape(point_count, -1)
    if array.ndim != 2 or array.shape[0] != point_count:
        raise RuntimeError(f"cannot decode stored IQ trace shape {array.shape}")
    return array


def load_sparameter_run(
    database_path: Any,
    run_id: int = 0,
) -> StoredSParameterSweep:
    """Load a saved S-parameter run; ``run_id=0`` selects the latest run."""
    try:
        from qcodes import initialise_or_create_database_at, load_by_id
    except ImportError as exc:
        raise RuntimeError("QCoDeS is required to load S-parameter data") from exc
    path = Path(database_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    initialise_or_create_database_at(str(path))
    run_id = _require_int(run_id, "run_id")
    if run_id == 0:
        with sqlite3.connect(path) as connection:
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(runs)")
            }
            row = (
                connection.execute(
                    "SELECT run_id FROM runs "
                    "WHERE sparameter_result_json IS NOT NULL "
                    "AND sparameter_result_json != '' "
                    "ORDER BY run_id DESC LIMIT 1"
                ).fetchone()
                if "sparameter_result_json" in columns
                else None
            )
        if row is None:
            raise RuntimeError(f"database contains no RF S-parameter runs: {path}")
        run_id = int(row[0])
    dataset = load_by_id(run_id)
    try:
        payload_text = dataset.get_metadata("sparameter_result_json")
    except (AttributeError, KeyError):
        payload_text = dataset.metadata.get("sparameter_result_json")
    if not payload_text:
        raise ValueError(f"run {run_id} is not an RF S-parameter run")
    payload = json.loads(payload_text)
    frequencies = np.asarray(payload["frequencies_mhz"], dtype=float)
    requested = np.asarray(
        payload.get("requested_frequencies_mhz", frequencies), dtype=float
    )
    i_data = dataset.get_parameter_data(I_TRACE_PARAMETER)[I_TRACE_PARAMETER]
    q_data = dataset.get_parameter_data(Q_TRACE_PARAMETER)[Q_TRACE_PARAMETER]
    if "power_gains" in payload:
        power_gains = np.asarray(payload["power_gains"], dtype=np.int64)
        trace_count = int(power_gains.size * frequencies.size)
        i_trace = _coerce_trace_rows(i_data[I_TRACE_PARAMETER], trace_count)
        q_trace = _coerce_trace_rows(q_data[Q_TRACE_PARAMETER], trace_count)
        iq = np.stack((i_trace, q_trace), axis=-1).reshape(
            power_gains.size,
            frequencies.size,
            i_trace.shape[1],
            2,
        )
        result = SParameterPowerSweepResult.from_iq(
            power_gains,
            requested,
            frequencies,
            iq,
            sample_rate_hz=float(payload.get("sample_rate_hz", 1_000_000.0)),
            output_powers_dbm=payload.get("output_powers_dbm"),
            frequency_gain_codes=payload.get("frequency_gain_codes"),
            actual_output_powers_dbm=payload.get("actual_output_powers_dbm"),
            input_powers_dbm=payload.get("input_powers_dbm"),
        )
    else:
        i_trace = _coerce_trace_rows(i_data[I_TRACE_PARAMETER], frequencies.size)
        q_trace = _coerce_trace_rows(q_data[Q_TRACE_PARAMETER], frequencies.size)
        iq = np.stack((i_trace, q_trace), axis=-1)
        result = SParameterSweepResult.from_iq(
            requested,
            frequencies,
            iq,
            sample_rate_hz=float(payload.get("sample_rate_hz", 1_000_000.0)),
            output_power_dbm=payload.get("output_power_dbm"),
            nominal_gain_code=payload.get(
                "nominal_gain_code",
                (
                    None
                    if payload.get("frequency_gain_codes") is None
                    else max(payload["frequency_gain_codes"])
                ),
            ),
            frequency_gain_codes=payload.get("frequency_gain_codes"),
            actual_output_powers_dbm=payload.get("actual_output_powers_dbm"),
            input_powers_dbm=payload.get("input_powers_dbm"),
        )
    return StoredSParameterSweep(
        run_id=int(dataset.run_id),
        guid=str(dataset.guid),
        database_path=path,
        row_count=int(iq.shape[0] * iq.shape[1]),
        result=result,
        dataset=dataset,
    )


__all__ = [
    "ACTUAL_INPUT_POWER_PARAMETER",
    "ACTUAL_OUTPUT_POWER_PARAMETER",
    "ADC_MAGNITUDE_DB_PARAMETER",
    "FILTER_TYPES",
    "CALIBRATED_GAIN_PARAMETER",
    "FREQUENCY_PARAMETER",
    "I_TRACE_PARAMETER",
    "MAGNITUDE_DB_PARAMETER",
    "MAX_RF_OUTPUT_GAIN",
    "MEAN_I_PARAMETER",
    "MEAN_Q_PARAMETER",
    "OUTPUT_POWER_PARAMETER",
    "PHASE_DEG_PARAMETER",
    "POWER_GAIN_PARAMETER",
    "POWER_SCALES",
    "Q_TRACE_PARAMETER",
    "SAMPLE_INDEX_PARAMETER",
    "SParameterPowerSweepResult",
    "SParameterSweepConfig",
    "SParameterSweepProgram",
    "SParameterSweepResult",
    "StoredSParameterSweep",
    "apply_power_calibration",
    "build_sparameter_program",
    "configure_sparameter_rf_board",
    "load_sparameter_run",
    "run_sparameter_sweep",
    "store_sparameter_result",
]
