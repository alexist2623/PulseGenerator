"""QICK RF output and ADC input power calibration workflows.

Output calibration drives a periodic QICK tone and reads a Keysight/Agilent
oscilloscope FFT marker.  Input calibration uses the FIR-DDR readout and a
previous output calibration to fit, at each frequency,

``input_power_dbm = slope * 20*log10(hypot(I, Q)) + intercept``.

The QCoDeS column names intentionally match the original calibration
notebooks so existing ``gain_pwr_calb.db`` files and new GUI-generated runs
can be consumed by the same :mod:`power_calibration` loader.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np

try:
    from .power_calibration import (
        CalibrationDatabase,
        INPUT_BOARD_TYPES,
        MAX_QICK_GAIN,
        OUTPUT_BOARD_TYPES,
    )
    from .qick_qcodes_experiment import (
        QickConnectionConfig,
        _checkpoint_sqlite_database,
        _json_text,
        _prepare_local_database,
        _publish_local_database,
        connect_qick,
    )
    from .qick_sparameter_sweep import (
        FILTER_TYPES,
        SParameterSweepConfig,
        build_sparameter_program,
        configure_sparameter_rf_board,
    )
except ImportError:
    from power_calibration import (
        CalibrationDatabase,
        INPUT_BOARD_TYPES,
        MAX_QICK_GAIN,
        OUTPUT_BOARD_TYPES,
    )
    from qick_qcodes_experiment import (
        QickConnectionConfig,
        _checkpoint_sqlite_database,
        _json_text,
        _prepare_local_database,
        _publish_local_database,
        connect_qick,
    )
    from qick_sparameter_sweep import (
        FILTER_TYPES,
        SParameterSweepConfig,
        build_sparameter_program,
        configure_sparameter_rf_board,
    )


ProgressCallback = Callable[[int, str], None]
ToneRunner = Callable[[Any, Any, int, int, float, int, int], float]


def _emit_progress(
    callback: Optional[ProgressCallback], percent: int, message: str
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(percent))), str(message))


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


def _attenuation(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 31.75:
        raise ValueError(f"{name} must be in [0, 31.75] dB")
    return result


def _gain_axis(start: int, stop: int, points: int, scale: str) -> np.ndarray:
    start = _integer(start, "gain_start", 1)
    stop = _integer(stop, "gain_end", 1)
    points = _integer(points, "gain_points", 2)
    if start > MAX_QICK_GAIN or stop > MAX_QICK_GAIN:
        raise ValueError(f"gain must not exceed {MAX_QICK_GAIN}")
    if start == stop:
        raise ValueError("gain start and end must differ")
    if scale not in ("linear", "log"):
        raise ValueError("gain_scale must be 'linear' or 'log'")
    values = (
        np.linspace(start, stop, points, dtype=float)
        if scale == "linear"
        else np.geomspace(start, stop, points, dtype=float)
    )
    gains = np.rint(values).astype(np.int64)
    gains[0] = start
    gains[-1] = stop
    if np.unique(gains).size != points:
        raise ValueError("gain points collapse to duplicate integer codes")
    return gains


def _frequency_axis(start: float, stop: float, points: int) -> np.ndarray:
    start = _finite(start, "frequency_start_mhz")
    stop = _finite(stop, "frequency_end_mhz")
    points = _integer(points, "frequency_points", 2)
    if start == stop:
        raise ValueError("frequency start and end must differ")
    return np.linspace(start, stop, points, dtype=float)


def _sample_name(board_type: str, configured: str, frequencies: np.ndarray) -> str:
    configured = str(configured).strip()
    if configured:
        if not configured.lower().startswith(board_type.lower()):
            return f"{board_type}_{configured}"
        return configured
    start = f"{float(frequencies[0]):.9g}".replace("-", "m").replace(".", "p")
    stop = f"{float(frequencies[-1]):.9g}".replace("-", "m").replace(".", "p")
    return f"{board_type}_{start}_{stop}MHz"


@dataclass(frozen=True)
class OscilloscopeConfig:
    """Keysight/Agilent oscilloscope FFT marker settings."""

    visa_resource: str = ""
    channel: int = 2
    fft_function: int = 1
    span_mhz: float = 100.0
    average_count: int = 100
    settle_seconds: float = 2.0
    sample_interval_seconds: float = 0.01
    timeout_ms: int = 10_000

    def __post_init__(self) -> None:
        _integer(self.channel, "oscilloscope channel", 1)
        _integer(self.fft_function, "FFT function", 1)
        _finite(self.span_mhz, "FFT span", positive=True)
        _integer(self.average_count, "scope average_count", 1)
        if _finite(self.settle_seconds, "scope settle_seconds") < 0.0:
            raise ValueError("scope settle_seconds must be nonnegative")
        if _finite(
            self.sample_interval_seconds,
            "scope sample_interval_seconds",
        ) < 0.0:
            raise ValueError("scope sample_interval_seconds must be nonnegative")
        _integer(self.timeout_ms, "scope timeout_ms", 1)


@dataclass(frozen=True)
class OutputPowerCalibrationConfig:
    """Periodic-tone output calibration measured by an oscilloscope."""

    database_path: str
    output_board_type: str = "RF_Out"
    output_ch: int = 0
    frequency_start_mhz: float = 400.0
    frequency_end_mhz: float = 500.0
    frequency_points: int = 11
    gain_start: int = 1000
    gain_end: int = MAX_QICK_GAIN
    gain_points: int = 16
    gain_scale: str = "linear"
    output_att1_db: float = 0.0
    output_att2_db: float = 0.0
    output_filter_type: str = "bypass"
    output_filter_cutoff_ghz: float = 2.5
    output_filter_bandwidth_ghz: float = 1.0
    nqz: int = 1
    tone_length_fabric_cycles: int = 16
    experiment_name: str = "QICK output power calibration"
    sample_name: str = ""
    notes: str = ""
    oscilloscope: OscilloscopeConfig = OscilloscopeConfig()

    def __post_init__(self) -> None:
        if not str(self.database_path).strip():
            raise ValueError("calibration database path must not be empty")
        if self.output_board_type not in OUTPUT_BOARD_TYPES:
            raise ValueError(f"output board must be one of {OUTPUT_BOARD_TYPES}")
        _integer(self.output_ch, "output_ch")
        self.frequencies_mhz
        self.gains
        _attenuation(self.output_att1_db, "output_att1_db")
        _attenuation(self.output_att2_db, "output_att2_db")
        if self.output_filter_type not in FILTER_TYPES:
            raise ValueError(f"output filter must be one of {FILTER_TYPES}")
        _finite(self.output_filter_cutoff_ghz, "output filter cutoff", positive=True)
        _finite(
            self.output_filter_bandwidth_ghz,
            "output filter bandwidth",
            positive=True,
        )
        _integer(self.nqz, "nqz", 1)
        _integer(self.tone_length_fabric_cycles, "tone length", 3)
        if not str(self.experiment_name).strip():
            raise ValueError("experiment_name must not be empty")
        if not isinstance(self.oscilloscope, OscilloscopeConfig):
            object.__setattr__(
                self,
                "oscilloscope",
                OscilloscopeConfig(**dict(self.oscilloscope)),
            )

    @property
    def frequencies_mhz(self) -> np.ndarray:
        return _frequency_axis(
            self.frequency_start_mhz,
            self.frequency_end_mhz,
            self.frequency_points,
        )

    @property
    def gains(self) -> np.ndarray:
        return _gain_axis(
            self.gain_start,
            self.gain_end,
            self.gain_points,
            self.gain_scale,
        )


@dataclass(frozen=True)
class InputPowerCalibrationConfig:
    """ADC magnitude calibration using a previously calibrated output path."""

    database_path: str
    output_board_type: str = "RF_Out"
    input_board_type: str = "RF_In"
    output_ch: int = 0
    readout_ch: int = 0
    frequency_start_mhz: float = 400.0
    frequency_end_mhz: float = 500.0
    frequency_points: int = 11
    gain_start: int = 1000
    gain_end: int = MAX_QICK_GAIN
    gain_points: int = 16
    gain_scale: str = "linear"
    scan_time_us: float = 100.0
    output_att1_db: float = 0.0
    output_att2_db: float = 0.0
    input_attenuation_db: float = 0.0
    output_filter_type: str = "bypass"
    output_filter_cutoff_ghz: float = 2.5
    output_filter_bandwidth_ghz: float = 1.0
    readout_filter_type: str = "bypass"
    readout_filter_cutoff_ghz: float = 2.5
    readout_filter_bandwidth_ghz: float = 1.0
    path_loss_db: float = 0.0
    fit_trim_low: int = 0
    fit_trim_high: int = 0
    nqz: int = 1
    margin_input_samples: int = 1024
    settle_seconds: float = 0.05
    force_overwrite: bool = True
    experiment_name: str = "QICK ADC input power calibration"
    sample_name: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not str(self.database_path).strip():
            raise ValueError("calibration database path must not be empty")
        if self.output_board_type not in OUTPUT_BOARD_TYPES:
            raise ValueError(f"output board must be one of {OUTPUT_BOARD_TYPES}")
        if self.input_board_type not in INPUT_BOARD_TYPES:
            raise ValueError(f"input board must be one of {INPUT_BOARD_TYPES}")
        _integer(self.output_ch, "output_ch")
        _integer(self.readout_ch, "readout_ch")
        self.frequencies_mhz
        gains = self.gains
        _finite(self.scan_time_us, "scan_time_us", positive=True)
        _attenuation(self.output_att1_db, "output_att1_db")
        _attenuation(self.output_att2_db, "output_att2_db")
        _attenuation(self.input_attenuation_db, "input_attenuation_db")
        for value, name in (
            (self.output_filter_type, "output_filter_type"),
            (self.readout_filter_type, "readout_filter_type"),
        ):
            if value not in FILTER_TYPES:
                raise ValueError(f"{name} must be one of {FILTER_TYPES}")
        for value, name in (
            (self.output_filter_cutoff_ghz, "output filter cutoff"),
            (self.output_filter_bandwidth_ghz, "output filter bandwidth"),
            (self.readout_filter_cutoff_ghz, "readout filter cutoff"),
            (self.readout_filter_bandwidth_ghz, "readout filter bandwidth"),
        ):
            _finite(value, name, positive=True)
        _finite(self.path_loss_db, "path_loss_db")
        low = _integer(self.fit_trim_low, "fit_trim_low")
        high = _integer(self.fit_trim_high, "fit_trim_high")
        if low + high > gains.size - 2:
            raise ValueError("fit trimming must leave at least two gain points")
        _integer(self.nqz, "nqz", 1)
        _integer(self.margin_input_samples, "margin_input_samples")
        if _finite(self.settle_seconds, "settle_seconds") < 0.0:
            raise ValueError("settle_seconds must be nonnegative")
        if not isinstance(self.force_overwrite, bool):
            raise TypeError("force_overwrite must be boolean")
        if not str(self.experiment_name).strip():
            raise ValueError("experiment_name must not be empty")

    @property
    def frequencies_mhz(self) -> np.ndarray:
        return _frequency_axis(
            self.frequency_start_mhz,
            self.frequency_end_mhz,
            self.frequency_points,
        )

    @property
    def gains(self) -> np.ndarray:
        return _gain_axis(
            self.gain_start,
            self.gain_end,
            self.gain_points,
            self.gain_scale,
        )

    def sweep_config(self, gain: int) -> SParameterSweepConfig:
        return SParameterSweepConfig(
            output_ch=self.output_ch,
            readout_ch=self.readout_ch,
            frequency_start_mhz=self.frequency_start_mhz,
            frequency_end_mhz=self.frequency_end_mhz,
            frequency_points=self.frequency_points,
            gain=int(gain),
            scan_time_us=self.scan_time_us,
            output_att1_db=self.output_att1_db,
            output_att2_db=self.output_att2_db,
            output_filter_type=self.output_filter_type,
            output_filter_cutoff_ghz=self.output_filter_cutoff_ghz,
            output_filter_bandwidth_ghz=self.output_filter_bandwidth_ghz,
            readout_attenuation_db=self.input_attenuation_db,
            readout_filter_type=self.readout_filter_type,
            readout_filter_cutoff_ghz=self.readout_filter_cutoff_ghz,
            readout_filter_bandwidth_ghz=self.readout_filter_bandwidth_ghz,
            nqz=self.nqz,
            margin_input_samples=self.margin_input_samples,
            force_overwrite=self.force_overwrite,
            settle_seconds=self.settle_seconds,
        )


@dataclass
class StoredCalibrationRun:
    run_id: int
    guid: str
    database_path: Path
    row_count: int
    board_type: str
    dataset: Any = None
    result: Optional[Mapping[str, Any]] = None


class KeysightFftPowerMeter:
    """Minimal PyVISA adapter for the calibration notebook's FFT marker flow."""

    def __init__(self, config: OscilloscopeConfig):
        self.config = config
        self.resource_manager = None
        self.instrument = None
        self.idn = ""

    def __enter__(self) -> "KeysightFftPowerMeter":
        if not self.config.visa_resource.strip():
            raise ValueError("oscilloscope VISA resource must not be empty")
        try:
            import pyvisa
        except ImportError as exc:
            raise RuntimeError(
                "PyVISA is required for oscilloscope power calibration; "
                "install PyVISA==1.16.2"
            ) from exc
        self.resource_manager = pyvisa.ResourceManager()
        self.instrument = self.resource_manager.open_resource(
            self.config.visa_resource
        )
        self.instrument.timeout = int(self.config.timeout_ms)
        self.instrument.write_termination = "\n"
        self.instrument.read_termination = "\n"
        self.idn = str(self.instrument.query("*IDN?")).strip()
        self.instrument.write(":RUN")
        self.instrument.write(f":CHANnel{int(self.config.channel)}:IMPedance FIFTy")
        return self

    def __exit__(self, *_exc) -> None:
        if self.instrument is not None:
            self.instrument.close()
            self.instrument = None
        if self.resource_manager is not None:
            self.resource_manager.close()
            self.resource_manager = None

    def configure_frequency(self, frequency_mhz: float) -> None:
        if self.instrument is None:
            raise RuntimeError("oscilloscope is not open")
        function = int(self.config.fft_function)
        channel = int(self.config.channel)
        frequency_hz = float(frequency_mhz) * 1.0e6
        span_hz = float(self.config.span_mhz) * 1.0e6
        commands = (
            f":FUNCtion{function}:DISPlay ON",
            f":FUNCtion{function}:OPERation FFT",
            f":FUNCtion{function}:SOURce CHANnel{channel}",
            f":FUNCtion{function}:CENTer {frequency_hz:.12g}",
            f":FUNCtion{function}:SPAN {span_hz:.12g}",
            ":SYSTem:PRECision ON",
            ":MARKer:MODE WAVeform",
            f":MARKer:X1Y1Source MATH{function}",
            f":MARKer:X1Position {frequency_hz:.12g}",
        )
        for command in commands:
            self.instrument.write(command)

    def measure_power_dbm(self, frequency_mhz: float) -> float:
        self.configure_frequency(frequency_mhz)
        if self.config.settle_seconds:
            time.sleep(self.config.settle_seconds)
        readings = []
        for _index in range(int(self.config.average_count)):
            response = self.instrument.query(":MARKer:Y1Position?")
            readings.append(float(str(response).strip()))
            if self.config.sample_interval_seconds:
                time.sleep(self.config.sample_interval_seconds)
        values = np.asarray(readings, dtype=float)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("oscilloscope returned a non-finite FFT power")
        return float(np.mean(values))


def _run_tone_program(
    soc: Any,
    soccfg: Any,
    output_ch: int,
    nqz: int,
    frequency_mhz: float,
    gain: int,
    length_cycles: int,
) -> float:
    """Start or stop one generator using an ASM-v1 periodic/oneshot command."""
    try:
        from qick.asm_v1 import QickProgram
    except ImportError as exc:
        raise RuntimeError("QICK ASM v1 is required for output calibration") from exc
    program = QickProgram(soccfg)
    program.declare_gen(ch=int(output_ch), nqz=int(nqz))
    frequency_word = program.freq2reg(float(frequency_mhz), gen_ch=int(output_ch))
    periodic = int(gain) != 0
    program.set_pulse_registers(
        ch=int(output_ch),
        style="const",
        freq=frequency_word,
        phase=0,
        gain=int(gain),
        length=int(length_cycles if periodic else 3),
        phrst=0,
        stdysel="last" if periodic else "zero",
        mode="periodic" if periodic else "oneshot",
    )
    program.pulse(ch=int(output_ch), t=0)
    program.end()
    program.run(soc)
    try:
        return float(program.reg2freq(frequency_word, gen_ch=int(output_ch)))
    except (AttributeError, TypeError):
        return float(frequency_mhz)


def _configure_output_chain(soc: Any, config: OutputPowerCalibrationConfig) -> Mapping[str, Any]:
    actual_att1, actual_att2 = soc.rfb_set_gen_rf(
        config.output_ch,
        config.output_att1_db,
        config.output_att2_db,
    )
    soc.rfb_set_gen_filter(
        config.output_ch,
        fc=config.output_filter_cutoff_ghz,
        bw=config.output_filter_bandwidth_ghz,
        ftype=config.output_filter_type,
    )
    return {
        "output_ch": config.output_ch,
        "att1_db": float(actual_att1),
        "att2_db": float(actual_att2),
        "filter_type": config.output_filter_type,
        "filter_cutoff_ghz": config.output_filter_cutoff_ghz,
        "filter_bandwidth_ghz": config.output_filter_bandwidth_ghz,
    }


def _store_output_calibration(
    config: OutputPowerCalibrationConfig,
    frequencies_mhz: np.ndarray,
    gains: np.ndarray,
    powers_dbm: np.ndarray,
    *,
    rf_settings: Mapping[str, Any],
    scope_identity: str,
) -> StoredCalibrationRun:
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
        raise RuntimeError("QCoDeS==0.58.0 is required for calibration storage") from exc
    database_path = Path(config.database_path).expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    staging, local_path = _prepare_local_database(database_path)
    try:
        initialise_or_create_database_at(str(local_path))
        sample_name = _sample_name(
            config.output_board_type,
            config.sample_name,
            frequencies_mhz,
        )
        experiment = load_or_create_experiment(config.experiment_name, sample_name)
        measurement = Measurement(exp=experiment, station=Station())
        gain_parameter = Parameter("gain", label="QICK DAC gain", unit="")
        frequency_parameter = Parameter("freq", label="Frequency", unit="MHz")
        power_parameter = Parameter("pwr", label="Output power", unit="dBm")
        measurement.register_parameter(gain_parameter)
        measurement.register_parameter(frequency_parameter)
        measurement.register_parameter(
            power_parameter,
            setpoints=(gain_parameter, frequency_parameter),
        )
        metadata = {
            "schema": "qstl-qick-output-power-calibration-v2",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "board_type": config.output_board_type,
            "configuration": {
                **asdict(config),
                "oscilloscope": asdict(config.oscilloscope),
            },
            "rf_settings_actual": dict(rf_settings),
            "oscilloscope_identity": scope_identity,
            "formula": "P(f,g)=response(f)+20*log10(g/32766)-attenuation_delta",
        }
        with measurement.run(
            write_in_background=False,
            in_memory_cache=False,
        ) as datasaver:
            dataset = datasaver.dataset
            dataset.add_metadata(
                "Attenuation",
                _json_text({
                    "att1": float(rf_settings["att1_db"]),
                    "att2": float(rf_settings["att2_db"]),
                }),
            )
            dataset.add_metadata("Calibration_Config", _json_text(metadata))
            if config.notes:
                dataset.add_metadata("calibration_notes", config.notes)
            for gain_index, gain in enumerate(gains):
                for frequency_index, frequency in enumerate(frequencies_mhz):
                    datasaver.add_result(
                        (gain_parameter, int(gain)),
                        (frequency_parameter, float(frequency)),
                        (power_parameter, float(powers_dbm[gain_index, frequency_index])),
                    )
            datasaver.flush_data_to_database()
        guid = str(dataset.guid)
        run_id = int(dataset.run_id)
        dataset.conn.close()
        _checkpoint_sqlite_database(local_path)
        _publish_local_database(local_path, database_path)
        initialise_or_create_database_at(str(database_path))
        final_dataset = load_by_guid(guid)
        return StoredCalibrationRun(
            run_id=run_id,
            guid=guid,
            database_path=database_path,
            row_count=int(gains.size * frequencies_mhz.size),
            board_type=config.output_board_type,
            dataset=final_dataset,
            result={
                "frequencies_mhz": frequencies_mhz.tolist(),
                "gains": gains.tolist(),
                "powers_dbm": powers_dbm.tolist(),
            },
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run_output_power_calibration(
    *,
    connection_config: QickConnectionConfig,
    calibration_config: OutputPowerCalibrationConfig,
    connector: Optional[Callable[..., Tuple[Any, Any]]] = None,
    power_meter_factory: Optional[Callable[[OscilloscopeConfig], Any]] = None,
    tone_runner: Optional[ToneRunner] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> StoredCalibrationRun:
    """Measure QICK output power over gain and frequency and save one run."""
    if not calibration_config.oscilloscope.visa_resource.strip() and power_meter_factory is None:
        raise ValueError("oscilloscope VISA resource must not be empty")
    tone_runner = tone_runner or _run_tone_program
    power_meter_factory = power_meter_factory or KeysightFftPowerMeter
    frequencies = calibration_config.frequencies_mhz
    gains = calibration_config.gains
    total = int(frequencies.size * gains.size)
    _emit_progress(progress_callback, 0, "Connecting to QICK")
    soc, soccfg = connect_qick(connection_config, connector=connector)
    _emit_progress(progress_callback, 3, "Configuring calibrated RF output chain")
    rf_settings = _configure_output_chain(soc, calibration_config)
    measured = np.empty((gains.size, frequencies.size), dtype=float)
    actual_frequencies = np.empty(frequencies.size, dtype=float)
    completed = 0
    scope_identity = ""
    try:
        with power_meter_factory(calibration_config.oscilloscope) as meter:
            scope_identity = str(getattr(meter, "idn", ""))
            for frequency_index, requested_frequency in enumerate(frequencies):
                for gain_index, gain in enumerate(gains):
                    actual_frequency = tone_runner(
                        soc,
                        soccfg,
                        calibration_config.output_ch,
                        calibration_config.nqz,
                        float(requested_frequency),
                        int(gain),
                        calibration_config.tone_length_fabric_cycles,
                    )
                    if gain_index == 0:
                        actual_frequencies[frequency_index] = actual_frequency
                    measured[gain_index, frequency_index] = meter.measure_power_dbm(
                        actual_frequency
                    )
                    completed += 1
                    _emit_progress(
                        progress_callback,
                        5 + round(83 * completed / total),
                        (
                            f"Scope calibration {completed}/{total}: "
                            f"{actual_frequency:.9g} MHz, gain {int(gain)}"
                        ),
                    )
    finally:
        try:
            tone_runner(
                soc,
                soccfg,
                calibration_config.output_ch,
                calibration_config.nqz,
                float(frequencies[0]),
                0,
                3,
            )
        except Exception:
            pass
    _emit_progress(progress_callback, 90, "Saving output calibration to QCoDeS")
    stored = _store_output_calibration(
        calibration_config,
        actual_frequencies,
        gains,
        measured,
        rf_settings=rf_settings,
        scope_identity=scope_identity,
    )
    _emit_progress(progress_callback, 100, f"Output calibration Run {stored.run_id} saved")
    return stored


def _store_input_calibration(
    config: InputPowerCalibrationConfig,
    frequencies_mhz: np.ndarray,
    gains: np.ndarray,
    adc_magnitude_db: np.ndarray,
    input_power_dbm: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
    *,
    output_run_id: int,
    rf_settings: Mapping[str, Any],
) -> StoredCalibrationRun:
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
        raise RuntimeError("QCoDeS==0.58.0 is required for calibration storage") from exc
    database_path = Path(config.database_path).expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    staging, local_path = _prepare_local_database(database_path)
    try:
        initialise_or_create_database_at(str(local_path))
        sample_name = _sample_name(
            config.input_board_type,
            config.sample_name,
            frequencies_mhz,
        )
        experiment = load_or_create_experiment(config.experiment_name, sample_name)
        measurement = Measurement(exp=experiment, station=Station())
        gain_parameter = Parameter("gain", label="QICK DAC gain", unit="")
        frequency_parameter = Parameter("freq", label="Frequency", unit="MHz")
        measured_parameter = Parameter(
            "measured_value",
            label="ADC magnitude",
            unit="dB ADC",
        )
        input_power_parameter = Parameter(
            "meas_in_pwr",
            label="Known input power",
            unit="dBm",
        )
        slope_parameter = Parameter("meas_slope", label="ADC calibration slope")
        intercept_parameter = Parameter(
            "meas_intercept",
            label="ADC calibration intercept",
            unit="dBm",
        )
        measurement.register_parameter(gain_parameter)
        measurement.register_parameter(frequency_parameter)
        measurement.register_parameter(
            measured_parameter,
            setpoints=(gain_parameter, frequency_parameter),
        )
        measurement.register_parameter(
            input_power_parameter,
            setpoints=(gain_parameter, frequency_parameter),
        )
        measurement.register_parameter(slope_parameter, setpoints=(frequency_parameter,))
        measurement.register_parameter(
            intercept_parameter,
            setpoints=(frequency_parameter,),
        )
        result_metadata = {
            "freq": frequencies_mhz.tolist(),
            "att1": float(config.output_att1_db),
            "att2": float(config.output_att2_db),
            "pwr_id": int(output_run_id),
            "output_run_id": int(output_run_id),
            "out_ch": int(config.output_ch),
            "in_ch": int(config.readout_ch),
        }
        config_metadata = {
            "schema": "qstl-qick-adc-input-power-calibration-v2",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_board_type": config.output_board_type,
            "input_board_type": config.input_board_type,
            "output_run_id": int(output_run_id),
            "output_ch": int(config.output_ch),
            "readout_ch": int(config.readout_ch),
            "output_att1_db": float(config.output_att1_db),
            "output_att2_db": float(config.output_att2_db),
            "input_attenuation_db": float(config.input_attenuation_db),
            "path_loss_db": float(config.path_loss_db),
            "fit_trim_low": int(config.fit_trim_low),
            "fit_trim_high": int(config.fit_trim_high),
            "formula": (
                "input_power_dbm = slope(f) * "
                "20*log10(hypot(mean_i,mean_q)) + intercept(f)"
            ),
            "configuration": asdict(config),
            "rf_settings_actual": dict(rf_settings),
        }
        with measurement.run(
            write_in_background=False,
            in_memory_cache=False,
        ) as datasaver:
            dataset = datasaver.dataset
            dataset.add_metadata("Calibration_Result", _json_text(result_metadata))
            dataset.add_metadata("Calibration_Config", _json_text(config_metadata))
            dataset.add_metadata(
                "Attenuation",
                _json_text({
                    "att1": float(config.output_att1_db),
                    "att2": float(config.output_att2_db),
                }),
            )
            if config.notes:
                dataset.add_metadata("calibration_notes", config.notes)
            for gain_index, gain in enumerate(gains):
                for frequency_index, frequency in enumerate(frequencies_mhz):
                    datasaver.add_result(
                        (gain_parameter, int(gain)),
                        (frequency_parameter, float(frequency)),
                        (
                            measured_parameter,
                            float(adc_magnitude_db[gain_index, frequency_index]),
                        ),
                        (
                            input_power_parameter,
                            float(input_power_dbm[gain_index, frequency_index]),
                        ),
                    )
            for frequency_index, frequency in enumerate(frequencies_mhz):
                datasaver.add_result(
                    (frequency_parameter, float(frequency)),
                    (slope_parameter, float(slopes[frequency_index])),
                    (intercept_parameter, float(intercepts[frequency_index])),
                )
            datasaver.flush_data_to_database()
        guid = str(dataset.guid)
        run_id = int(dataset.run_id)
        dataset.conn.close()
        _checkpoint_sqlite_database(local_path)
        _publish_local_database(local_path, database_path)
        initialise_or_create_database_at(str(database_path))
        final_dataset = load_by_guid(guid)
        return StoredCalibrationRun(
            run_id=run_id,
            guid=guid,
            database_path=database_path,
            row_count=int(gains.size * frequencies_mhz.size + frequencies_mhz.size),
            board_type=config.input_board_type,
            dataset=final_dataset,
            result={
                "frequencies_mhz": frequencies_mhz.tolist(),
                "gains": gains.tolist(),
                "adc_magnitude_db": adc_magnitude_db.tolist(),
                "input_power_dbm": input_power_dbm.tolist(),
                "slopes": slopes.tolist(),
                "intercepts_dbm": intercepts.tolist(),
                "output_run_id": int(output_run_id),
            },
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run_input_power_calibration(
    *,
    connection_config: QickConnectionConfig,
    calibration_config: InputPowerCalibrationConfig,
    tproc_mhz: Optional[float] = None,
    connector: Optional[Callable[..., Tuple[Any, Any]]] = None,
    program_factory: Optional[Callable[..., Any]] = None,
    acquisition_callback: Optional[Callable[[Any, Any], Any]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> StoredCalibrationRun:
    """Fit ADC dB values to actual connector input power at each frequency."""
    _emit_progress(progress_callback, 0, "Connecting to QICK")
    soc, soccfg = connect_qick(connection_config, connector=connector)
    gains = calibration_config.gains
    program_factory = program_factory or build_sparameter_program
    programs = [
        program_factory(
            soccfg,
            calibration_config.sweep_config(int(gain)),
            tproc_mhz=tproc_mhz,
        )
        for gain in gains
    ]
    frequencies = np.asarray(programs[0].frequencies_mhz, dtype=float)
    for program in programs[1:]:
        if not np.array_equal(program.frequencies_mhz, frequencies):
            raise RuntimeError("input calibration frequency grids are inconsistent")
    _emit_progress(progress_callback, 3, "Selecting matching output calibration")
    output_calibration = CalibrationDatabase(
        calibration_config.database_path
    ).output_calibration(
        calibration_config.output_board_type,
        frequencies,
    )
    _emit_progress(progress_callback, 5, "Configuring RF output and input chains")
    rf_settings = configure_sparameter_rf_board(
        soc,
        calibration_config.sweep_config(int(gains[0])),
    )
    actual_output = dict(rf_settings.get("output", {}))
    actual_input = dict(rf_settings.get("readout", {}))
    output_att1 = float(
        actual_output.get("commanded_att1_db", calibration_config.output_att1_db)
    )
    output_att2 = float(
        actual_output.get("commanded_att2_db", calibration_config.output_att2_db)
    )
    input_attenuation = float(
        actual_input.get(
            "commanded_attenuation_db",
            calibration_config.input_attenuation_db,
        )
    )
    measured_db = np.empty((gains.size, frequencies.size), dtype=float)
    known_input_dbm = np.empty_like(measured_db)
    for gain_index, (gain, program) in enumerate(zip(gains, programs)):
        result = (
            acquisition_callback(soc, program)
            if acquisition_callback is not None
            else program.acquire_fir_ddr(soc)
        )
        if not np.array_equal(result.frequencies_mhz, frequencies):
            raise RuntimeError("acquired input-calibration frequency grid changed")
        measured_db[gain_index] = np.asarray(result.magnitude_db, dtype=float)
        known_input_dbm[gain_index] = (
            output_calibration.output_power_dbm(
                frequencies,
                int(gain),
                output_att1_db=output_att1,
                output_att2_db=output_att2,
            )
            - float(calibration_config.path_loss_db)
        )
        _emit_progress(
            progress_callback,
            7 + round(76 * (gain_index + 1) / gains.size),
            f"ADC calibration gain {gain_index + 1}/{gains.size}: {int(gain)}",
        )
    low = int(calibration_config.fit_trim_low)
    high = int(calibration_config.fit_trim_high)
    stop = gains.size - high if high else gains.size
    fit_slice = slice(low, stop)
    slopes = np.empty(frequencies.size, dtype=float)
    intercepts = np.empty(frequencies.size, dtype=float)
    for frequency_index in range(frequencies.size):
        x_values = measured_db[fit_slice, frequency_index]
        y_values = known_input_dbm[fit_slice, frequency_index]
        if np.unique(x_values).size < 2:
            raise RuntimeError(
                f"ADC values at {frequencies[frequency_index]:.9g} MHz "
                "do not contain two distinct fit points"
            )
        slopes[frequency_index], intercepts[frequency_index] = np.polyfit(
            x_values,
            y_values,
            1,
        )
    adjusted_config = replace(
        calibration_config,
        input_attenuation_db=input_attenuation,
        output_att1_db=output_att1,
        output_att2_db=output_att2,
    )
    _emit_progress(progress_callback, 86, "Saving ADC input calibration to QCoDeS")
    stored = _store_input_calibration(
        adjusted_config,
        frequencies,
        gains,
        measured_db,
        known_input_dbm,
        slopes,
        intercepts,
        output_run_id=output_calibration.summary.run_id,
        rf_settings=rf_settings,
    )
    _emit_progress(progress_callback, 100, f"Input calibration Run {stored.run_id} saved")
    return stored


__all__ = [
    "InputPowerCalibrationConfig",
    "KeysightFftPowerMeter",
    "OscilloscopeConfig",
    "OutputPowerCalibrationConfig",
    "StoredCalibrationRun",
    "run_input_power_calibration",
    "run_output_power_calibration",
]
