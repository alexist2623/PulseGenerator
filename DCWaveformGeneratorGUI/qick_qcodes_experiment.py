"""Execute a QICK fine-tune program and persist its IQ trace with QCoDeS.

QCoDeS and the QICK Pyro client are imported only when an experiment runs, so
the waveform editor remains usable on machines without laboratory packages.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np


ProgressCallback = Callable[[int, str], None]
DEFAULT_QCODES_BATCH_ROWS = 8192
QCODES_STAGING_ENV = "QSTL_QCODES_STAGING_DIR"
# Legacy packed-IQ parameter used by runs written before split trace storage.
IQ_TRACE_PARAMETER = "iq_trace"
I_TRACE_PARAMETER = "i_trace"
Q_TRACE_PARAMETER = "q_trace"
SAMPLE_INDEX_PARAMETER = "sample_index"

try:
    from .dc_waveform_core import (
        DEFAULT_QICK_TPROC_MHZ,
        QickDdrReadoutSpec,
        QickRfPulseSpec,
        dc_iq_to_current,
    )
except ImportError:
    from dc_waveform_core import (
        DEFAULT_QICK_TPROC_MHZ,
        QickDdrReadoutSpec,
        QickRfPulseSpec,
        dc_iq_to_current,
    )


def _runtime_types():
    try:
        from .qick_fine_tune_sweep import (
            DdrFirReadoutConfig,
            RfPulseConfig,
            cycles_from_us,
        )
    except ImportError:
        from qick_fine_tune_sweep import (
            DdrFirReadoutConfig,
            RfPulseConfig,
            cycles_from_us,
        )
    return DdrFirReadoutConfig, RfPulseConfig, cycles_from_us


def _resolve_tproc_mhz(soccfg, tproc_mhz: Optional[float]) -> float:
    value = (
        soccfg["tprocs"][0]["f_time"]
        if tproc_mhz is None
        else tproc_mhz
    )
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("tProcessor clock must be positive and finite")
    return value


@dataclass(frozen=True)
class QickConnectionConfig:
    """Pyro nameserver location used to obtain ``soc`` and ``soccfg``."""

    host: str = "192.168.2.99"
    ns_port: int = 8888
    proxy_name: str = "myqick"

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("QICK host/IP must not be empty")
        if isinstance(self.ns_port, bool) or not 1 <= int(self.ns_port) <= 65535:
            raise ValueError("QICK nameserver port must be in [1, 65535]")
        if not self.proxy_name.strip():
            raise ValueError("QICK proxy name must not be empty")


@dataclass(frozen=True)
class QcodesRunConfig:
    """QCoDeS database and experiment naming settings."""

    database_path: str
    experiment_name: str = "QICK pulse experiment"
    sample_name: str = "PulseGenerator"
    notes: str = ""
    sample_rate_hz: float = 1_000_000.0

    def __post_init__(self) -> None:
        if not str(self.database_path).strip():
            raise ValueError("QCoDeS database path must not be empty")
        if not self.experiment_name.strip():
            raise ValueError("QCoDeS experiment name must not be empty")
        if not self.sample_name.strip():
            raise ValueError("QCoDeS sample name must not be empty")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive and finite")

    @property
    def resolved_database_path(self) -> Path:
        path = Path(self.database_path).expanduser()
        if path.suffix.lower() != ".db":
            path = path.with_suffix(".db")
        return path.resolve()


@dataclass
class StoredQickExperiment:
    """Objects and identifiers produced by one hardware/database run."""

    run_id: int
    guid: str
    database_path: Path
    row_count: int
    dataset: Any
    program: Any
    ddr_result: Any
    rf_settings: Mapping[str, Any]


def _json_ready(value: Any) -> Any:
    """Convert NumPy, dataclass, and QICK objects to strict JSON values."""
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _json_ready(vars(value))
    return repr(value)


def _json_text(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _emit_progress(
    callback: Optional[ProgressCallback],
    percent: int,
    message: str,
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(percent))), str(message))


def build_awg_vertex_metadata(
    sequence: Any,
    *,
    fabric_mhz: float,
    full_scale_mv: float,
) -> Mapping[str, Any]:
    """Build compact virtual/physical AWG vertices for every sweep point."""
    fabric_mhz = float(fabric_mhz)
    full_scale_mv = float(full_scale_mv)
    if not np.isfinite(fabric_mhz) or fabric_mhz <= 0.0:
        raise ValueError("fabric_mhz must be positive and finite")
    if not np.isfinite(full_scale_mv) or full_scale_mv <= 0.0:
        raise ValueError("full_scale_mv must be positive and finite")

    output_names = tuple(sequence.output_names)
    point_count = int(sequence.sweep_point_count)
    common_times = None
    virtual_points = []
    physical_points = []
    for point_index in range(point_count):
        virtual_times, virtual_values, _ = sequence.waveform_vertices(
            point_index, space="virtual"
        )
        physical_times, physical_values, _ = sequence.waveform_vertices(
            point_index, space="physical"
        )
        if not np.array_equal(virtual_times, physical_times):
            raise RuntimeError("virtual and physical AWG vertex times differ")
        if common_times is None:
            common_times = virtual_times
        elif not np.array_equal(common_times, virtual_times):
            raise RuntimeError("AWG vertex times differ between sweep points")
        virtual_points.append(
            np.vstack([virtual_values[name] for name in output_names])
            * full_scale_mv
        )
        physical_points.append(
            np.vstack([physical_values[name] for name in output_names])
            * full_scale_mv
        )

    if common_times is None:
        raise ValueError("the AWG sequence has no waveform vertices")
    coordinates = np.asarray(sequence.sweep_coordinates, dtype=float)
    shared = {
        "schema": "qick-awg-waveform-vertices-v1",
        "output_names": list(output_names),
        "point_index": list(range(point_count)),
        "sweep_coordinates": coordinates.tolist(),
        "sweep_axes": _json_ready(sequence.sweep_axes),
        "time_cycles": common_times.tolist(),
        "time_us": (common_times / fabric_mhz).tolist(),
        "time_reference": "start of each pulse sequence repetition",
        "amplitude_unit": "mV",
        "full_scale_mv": full_scale_mv,
        "vertex_rule": (
            "Connect adjacent vertices in order; equal adjacent times encode "
            "an instantaneous SET transition."
        ),
        "value_shape": [point_count, len(output_names), len(common_times)],
    }
    return {
        "virtual": {
            **shared,
            "voltage_space": "virtual",
            "values_mv": np.asarray(virtual_points).tolist(),
        },
        "physical": {
            **shared,
            "voltage_space": "physical",
            "values_mv": np.asarray(physical_points).tolist(),
        },
    }


def _coerce_awg_vertex_data(
    awg_vertices: Mapping[str, Any],
    *,
    point_count: int,
):
    """Validate compact AWG vertices and return arrays for QCoDeS storage."""
    if not awg_vertices:
        return None
    try:
        virtual = awg_vertices["virtual"]
        physical = awg_vertices["physical"]
        output_names = tuple(str(name) for name in virtual["output_names"])
        time_us = np.asarray(virtual["time_us"], dtype=float)
        virtual_mv = np.asarray(virtual["values_mv"], dtype=float)
        physical_mv = np.asarray(physical["values_mv"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid AWG waveform vertex data") from exc

    if not output_names or time_us.ndim != 1 or time_us.size < 1:
        raise ValueError("AWG vertex output names and times must not be empty")
    expected_shape = (point_count, len(output_names), time_us.size)
    if virtual_mv.shape != expected_shape or physical_mv.shape != expected_shape:
        raise ValueError(
            "AWG vertex arrays must have shape "
            f"{expected_shape}, received {virtual_mv.shape} and {physical_mv.shape}"
        )
    if tuple(str(name) for name in physical.get("output_names", ())) != output_names:
        raise ValueError("virtual and physical AWG output names differ")
    physical_time_us = np.asarray(physical.get("time_us", ()), dtype=float)
    if not np.array_equal(time_us, physical_time_us):
        raise ValueError("virtual and physical AWG vertex times differ")
    if not all(np.all(np.isfinite(values)) for values in (
        time_us,
        virtual_mv,
        physical_mv,
    )):
        raise ValueError("AWG vertex times and voltages must be finite")
    return output_names, time_us, virtual_mv, physical_mv


def connect_qick(
    config: QickConnectionConfig,
    *,
    connector: Optional[Callable[..., Tuple[Any, Any]]] = None,
) -> Tuple[Any, Any]:
    """Connect to a QICK Pyro server, with injection support for tests."""
    if connector is None:
        try:
            from qick.pyro import make_proxy
        except ImportError as exc:
            raise RuntimeError(
                "QICK Python library is required for hardware execution"
            ) from exc
        connector = make_proxy
    return connector(
        ns_host=config.host,
        ns_port=config.ns_port,
        proxy_name=config.proxy_name,
    )


def build_runtime_rf_pulses(
    soccfg,
    specs: Sequence[QickRfPulseSpec],
    *,
    tproc_mhz: Optional[float] = None,
) -> Tuple[Any, ...]:
    """Convert GUI RF timing in microseconds to actual QICK clock cycles."""
    _ddr_type, rf_type, cycles_from_us = _runtime_types()
    tproc_mhz = _resolve_tproc_mhz(soccfg, tproc_mhz)
    pulses = []
    for spec in specs:
        gen_cfg = soccfg["gens"][spec.gen_ch]
        delay_cycles = (
            0
            if spec.delay_us <= 0.0
            else cycles_from_us(spec.delay_us, tproc_mhz)
        )
        pulses.append(rf_type(
            gen_ch=spec.gen_ch,
            at_segment=spec.segment_name,
            length_cycles=cycles_from_us(
                spec.duration_us, float(gen_cfg["f_fabric"])
            ),
            gain=spec.gain,
            freq_mhz=spec.frequency_mhz,
            phase_degrees=spec.phase_degrees,
            nqz=spec.nqz,
            delay_tproc_cycles=delay_cycles,
            require_within_segment=spec.require_within_segment,
        ))
    return tuple(pulses)


def build_runtime_ddr_readout(
    soccfg,
    spec: QickDdrReadoutSpec,
    *,
    tproc_mhz: Optional[float] = None,
) -> Any:
    """Convert the GUI's FIR DDR readout settings to program timing."""
    ddr_type, _rf_type, cycles_from_us = _runtime_types()
    tproc_mhz = _resolve_tproc_mhz(soccfg, tproc_mhz)
    delay_cycles = (
        0
        if spec.delay_us <= 0.0
        else cycles_from_us(spec.delay_us, tproc_mhz)
    )
    return ddr_type(
        ro_ch=spec.ro_ch,
        samples_per_trigger=spec.samples_per_trigger,
        at_segment=spec.segment_name,
        readout_freq_mhz=spec.readout_frequency_mhz,
        trigger_delay_tproc_cycles=delay_cycles,
        margin_input_samples=spec.margin_input_samples,
        address=spec.address,
        force_overwrite=spec.force_overwrite,
    )


def build_qick_program(
    soccfg,
    sequence,
    *,
    awg_channels: Sequence[int],
    repetitions_per_sweep: int,
    tproc_mhz: Optional[float] = None,
    rf_specs: Sequence[QickRfPulseSpec] = (),
    readout_spec: Optional[QickDdrReadoutSpec] = None,
):
    """Build the tProcessor program without configuring or running hardware."""
    effective_tproc_mhz = _resolve_tproc_mhz(soccfg, tproc_mhz)
    program_kwargs = {
        "awg_channels": tuple(int(channel) for channel in awg_channels),
        "tproc_mhz": effective_tproc_mhz,
        "repetitions_per_sweep": int(repetitions_per_sweep),
        "rf_pulses": build_runtime_rf_pulses(
            soccfg, rf_specs, tproc_mhz=effective_tproc_mhz
        ),
    }
    if readout_spec is not None:
        program_kwargs["ddr_readout"] = build_runtime_ddr_readout(
            soccfg, readout_spec, tproc_mhz=effective_tproc_mhz
        )
    return sequence.make_program(soccfg, **program_kwargs)


def configure_rf_board(
    soc,
    rf_specs: Sequence[QickRfPulseSpec],
    readout_spec: QickDdrReadoutSpec,
) -> Mapping[str, Any]:
    """Apply RF-board output/input attenuation and programmable filters."""
    outputs = []
    output_details = []
    for spec in rf_specs:
        if spec.output_board_type == "RF_Out":
            actual_attenuation = soc.rfb_set_gen_rf(
                spec.gen_ch, spec.att1_db, spec.att2_db
            )
            try:
                actual_att1, actual_att2 = (
                    float(actual_attenuation[0]),
                    float(actual_attenuation[1]),
                )
            except (IndexError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"RF generator {spec.gen_ch} returned an invalid attenuation "
                    f"result: {actual_attenuation!r}"
                ) from exc
        else:
            soc.rfb_set_gen_dc(spec.gen_ch)
            actual_att1 = 0.0
            actual_att2 = 0.0
        outputs.append((actual_att1, actual_att2))

        if spec.output_board_type == "RF_Out":
            soc.rfb_set_gen_filter(
                spec.gen_ch,
                fc=spec.filter_cutoff,
                bw=spec.filter_bandwidth,
                ftype=spec.filter_type,
            )
        output_details.append({
            "gen_ch": int(spec.gen_ch),
            "board_type": str(spec.output_board_type),
            "attenuators_present": spec.output_board_type == "RF_Out",
            "requested_att1_db": float(spec.att1_db),
            "requested_att2_db": float(spec.att2_db),
            "commanded_att1_db": actual_att1,
            "commanded_att2_db": actual_att2,
            "filter_type": str(spec.filter_type),
            "filter_cutoff_ghz": float(spec.filter_cutoff),
            "filter_bandwidth_ghz": float(spec.filter_bandwidth),
        })
    if readout_spec.input_board_type == "RF_In":
        readout_setting = float(
            soc.rfb_set_ro_rf(
                readout_spec.ro_ch,
                readout_spec.attenuation_db,
            )
        )
        readout_attenuation = readout_setting
        readout_dc_gain = 0.0
    else:
        readout_setting = float(
            soc.rfb_set_ro_dc(
                readout_spec.ro_ch,
                readout_spec.dc_gain_db,
            )
        )
        readout_attenuation = 0.0
        readout_dc_gain = readout_setting
    if readout_spec.input_board_type == "RF_In":
        soc.rfb_set_ro_filter(
            readout_spec.ro_ch,
            fc=readout_spec.filter_cutoff,
            bw=readout_spec.filter_bandwidth,
            ftype=readout_spec.filter_type,
        )
    return {
        "outputs": tuple(outputs),
        "output_details": tuple(output_details),
        "readout": readout_setting,
        "readout_details": {
            "ro_ch": int(readout_spec.ro_ch),
            "board_type": str(readout_spec.input_board_type),
            "attenuator_present": readout_spec.input_board_type == "RF_In",
            "requested_attenuation_db": float(readout_spec.attenuation_db),
            "commanded_attenuation_db": readout_attenuation,
            "requested_dc_gain_db": float(readout_spec.dc_gain_db),
            "commanded_dc_gain_db": readout_dc_gain,
            "dc_measure_mode": bool(readout_spec.dc_measure_mode),
            "dc_measure_gain_v_per_a": float(
                readout_spec.dc_measure_gain_v_per_a
            ),
            "adc_to_voltage_conversion": "identity",
            "measurement_unit": readout_spec.measurement_unit,
            "filter_type": str(readout_spec.filter_type),
            "filter_cutoff_ghz": float(readout_spec.filter_cutoff),
            "filter_bandwidth_ghz": float(readout_spec.filter_bandwidth),
        },
    }


def execute_qick_sequence(
    soc,
    soccfg,
    sequence,
    *,
    awg_channels: Sequence[int],
    repetitions_per_sweep: int,
    tproc_mhz: Optional[float] = None,
    rf_specs: Sequence[QickRfPulseSpec],
    readout_spec: QickDdrReadoutSpec,
    progress: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Any, Any, Mapping[str, Any]]:
    """Configure RF hardware, execute the tProcessor program, and read DDR."""
    _emit_progress(progress_callback, 5, "Configuring RF hardware")
    rf_settings = configure_rf_board(soc, rf_specs, readout_spec)
    _emit_progress(progress_callback, 8, "Compiling the tProcessor program")
    program = build_qick_program(
        soccfg,
        sequence,
        awg_channels=tuple(int(channel) for channel in awg_channels),
        tproc_mhz=tproc_mhz,
        repetitions_per_sweep=int(repetitions_per_sweep),
        rf_specs=rf_specs,
        readout_spec=readout_spec,
    )
    sweep_point_count = int(getattr(
        sequence,
        "sweep_point_count",
        getattr(program, "cfg", {}).get("expts", 1),
    ))
    _emit_progress(
        progress_callback,
        10,
        (
            "Running pulse sequence: "
            f"{sweep_point_count:,} sweep points x "
            f"{int(repetitions_per_sweep):,} repetitions"
        ),
    )

    def counter_progress(completed: int, total: int) -> None:
        fraction = 1.0 if total <= 0 else completed / total
        percent = 10 + round(max(0.0, min(1.0, fraction)) * 45)
        _emit_progress(
            progress_callback,
            percent,
            (
                f"Running acquisitions {completed:,}/{total:,} "
                f"({sweep_point_count:,} sweep points x "
                f"{int(repetitions_per_sweep):,} repetitions)"
            ),
        )

    ddr_result = program.acquire_fir_ddr(
        soc,
        progress=progress,
        counter_progress=counter_progress if progress_callback is not None else None,
    )
    _emit_progress(progress_callback, 60, "FIR DDR acquisition completed")
    return program, ddr_result, rf_settings


def _sweep_coordinates(ddr_result: Any) -> np.ndarray:
    axis_count = len(tuple(ddr_result.sweep_axes))
    point_count = int(np.asarray(ddr_result.iq).shape[0])
    if axis_count == 0:
        return np.empty((point_count, 0), dtype=float)
    coordinates = np.asarray(ddr_result.sweep_points, dtype=float)
    if axis_count == 1:
        coordinates = coordinates.reshape(-1, 1)
    if coordinates.shape != (point_count, axis_count):
        raise ValueError(
            "sweep coordinate shape does not match DDR point/axis counts"
        )
    return coordinates


def _qcodes_identifier(value: Any) -> str:
    """Convert a user-facing output/segment label to a QCoDeS-safe name."""
    identifier = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_").lower()
    if not identifier:
        identifier = "unnamed"
    if identifier[0].isdigit():
        identifier = f"p_{identifier}"
    return identifier


def _sweep_parameter_names(axes: Sequence[Any]) -> Tuple[str, ...]:
    """Name sweep axes by their physical output and segment target."""
    used = set()
    names = []
    for axis in axes:
        base = (
            f"{_qcodes_identifier(axis.output_name)}_"
            f"{_qcodes_identifier(axis.segment_name)}_voltage_mv"
        )
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        names.append(name)
    return tuple(names)


def _full_scale_mv(gui_settings: Mapping[str, Any]) -> float:
    qick_settings = gui_settings.get("qick", {})
    if not isinstance(qick_settings, Mapping):
        raise ValueError("GUI qick settings must be a mapping")
    value = float(qick_settings.get("full_scale_mv", 100.0))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("QICK full_scale_mv must be finite and positive")
    return value


def _qcodes_staging_root() -> Path:
    configured = os.environ.get(QCODES_STAGING_ENV)
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        local_root = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
        root = Path(local_root) / "QSTL_PulseGenerator" / "qcodes_staging"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _backup_sqlite_database(source: Path, destination: Path) -> None:
    """Copy a live SQLite database, including committed WAL content."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_uri = source.resolve().as_uri() + "?mode=ro"
    source_connection = sqlite3.connect(source_uri, uri=True, timeout=60.0)
    destination_connection = sqlite3.connect(str(destination), timeout=60.0)
    try:
        source_connection.backup(destination_connection, pages=4096, sleep=0.01)
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()


def _prepare_local_database(database_path: Path) -> Tuple[Path, Path]:
    staging_directory = Path(tempfile.mkdtemp(
        prefix="qick_qcodes_",
        dir=str(_qcodes_staging_root()),
    ))
    local_database_path = staging_directory / database_path.name
    if database_path.exists() and database_path.stat().st_size > 0:
        _backup_sqlite_database(database_path, local_database_path)
    return staging_directory, local_database_path


def _checkpoint_sqlite_database(database_path: Path) -> None:
    connection = sqlite3.connect(str(database_path), timeout=60.0)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.commit()
    finally:
        connection.close()


def _publish_local_database(local_path: Path, database_path: Path) -> None:
    """Publish the completed local database with SQLite's backup API."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    source_uri = local_path.resolve().as_uri() + "?mode=ro"
    source_connection = sqlite3.connect(source_uri, uri=True, timeout=60.0)
    destination_connection = sqlite3.connect(str(database_path), timeout=60.0)
    try:
        destination_connection.execute("PRAGMA busy_timeout=60000")
        source_connection.backup(destination_connection, pages=4096, sleep=0.01)
        destination_connection.commit()
        destination_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()


def _first_value_per_trace(values: Any, trace_count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[0] != trace_count:
        raise ValueError("QCoDeS trace setpoint count does not match IQ traces")
    return array.reshape(trace_count, -1)[:, 0]


def load_qick_iq_arrays(dataset: Any) -> Mapping[str, Any]:
    """Load split or legacy packed IQ trace arrays and derived quantities."""
    metadata = json.loads(dataset.get_metadata("qick_experiment_json"))
    layout = metadata["measurement_layout"]
    expected_shape = tuple(int(value) for value in layout["iq_shape"])
    if len(expected_shape) != 4 or expected_shape[-1] != 2:
        raise ValueError("stored IQ shape must be (point, repetition, sample, 2)")

    trace_count = expected_shape[0] * expected_shape[1]
    parameter_names = {parameter.name for parameter in dataset.get_parameters()}
    if {I_TRACE_PARAMETER, Q_TRACE_PARAMETER}.issubset(parameter_names):
        i_parameter_data = dataset.get_parameter_data(I_TRACE_PARAMETER)
        q_parameter_data = dataset.get_parameter_data(Q_TRACE_PARAMETER)
        trace_data = i_parameter_data[I_TRACE_PARAMETER]
        flat_i = np.asarray(trace_data[I_TRACE_PARAMETER])
        flat_q = np.asarray(
            q_parameter_data[Q_TRACE_PARAMETER][Q_TRACE_PARAMETER]
        )
        expected_flat_shape = (trace_count, expected_shape[2])
        if (
            flat_i.shape != expected_flat_shape
            or flat_q.shape != expected_flat_shape
        ):
            raise ValueError(
                "stored split IQ arrays have shapes "
                f"{flat_i.shape} and {flat_q.shape}, expected "
                f"{expected_flat_shape}"
            )
        iq = np.stack((flat_i, flat_q), axis=-1).reshape(expected_shape)
    elif IQ_TRACE_PARAMETER in parameter_names:
        parameter_data = dataset.get_parameter_data(IQ_TRACE_PARAMETER)
        trace_data = parameter_data[IQ_TRACE_PARAMETER]
        flat_iq = np.asarray(trace_data[IQ_TRACE_PARAMETER])
        expected_flat_shape = (trace_count, expected_shape[2], 2)
        if flat_iq.shape != expected_flat_shape:
            raise ValueError(
                f"stored IQ arrays have shape {flat_iq.shape}, expected "
                f"{expected_flat_shape}"
            )
        iq = flat_iq.reshape(expected_shape)
    else:
        raise ValueError("dataset does not contain supported IQ trace storage")

    i_values = iq[..., 0]
    q_values = iq[..., 1]
    sample_index_parameter = layout.get(
        "sample_index_parameter",
        SAMPLE_INDEX_PARAMETER,
    )
    if sample_index_parameter in trace_data:
        flat_sample_index = np.asarray(trace_data[sample_index_parameter])
        expected_index_shape = (trace_count, expected_shape[2])
        if flat_sample_index.shape != expected_index_shape:
            raise ValueError(
                f"stored sample-index arrays have shape {flat_sample_index.shape}, "
                f"expected {expected_index_shape}"
            )
        sample_index = flat_sample_index[0].astype(np.int64, copy=False)
        if not np.all(flat_sample_index == sample_index[None, :]):
            raise ValueError("stored sample-index axes are inconsistent between traces")
    else:
        # Compatibility with v1/v2 datasets, which omitted the array axis.
        sample_index = np.arange(expected_shape[2], dtype=np.int64)
    sample_period_us = float(layout["sample_period_us"])

    sweep_coordinates_mv = {}
    for axis in layout.get("sweep_axes", []):
        parameter_name = axis["parameter"]
        sweep_coordinates_mv[parameter_name] = _first_value_per_trace(
            trace_data[parameter_name],
            trace_count,
        ).reshape(expected_shape[0], expected_shape[1])
    repetitions = _first_value_per_trace(
        trace_data["repetition_index"],
        trace_count,
    ).astype(np.int64).reshape(expected_shape[0], expected_shape[1])

    return {
        "iq": iq,
        "i": i_values,
        "q": q_values,
        "magnitude": np.hypot(i_values.astype(float), q_values.astype(float)),
        "phase_deg": np.degrees(np.arctan2(q_values, i_values)),
        "sample_index": sample_index,
        "time_us": sample_index * sample_period_us,
        "repetition_index": repetitions,
        "sweep_coordinates_mv": sweep_coordinates_mv,
        "iq_unit": str(layout.get("iq_unit", "ADC units")),
        "measurement_mode": str(layout.get("measurement_mode", "raw_iq")),
        "metadata": metadata,
    }


def _measurement_iq_values(
    iq: Any,
    rf_settings: Mapping[str, Any],
) -> Tuple[np.ndarray, str, str, Mapping[str, Any]]:
    """Apply the selected readout-domain representation before storage."""
    raw_iq = np.asarray(iq)
    readout_details = rf_settings.get("readout_details", {})
    if not isinstance(readout_details, Mapping):
        readout_details = {}
    dc_measure_mode = bool(readout_details.get("dc_measure_mode", False))
    if not dc_measure_mode:
        return raw_iq, "ADC units", "raw_iq", {
            "adc_to_voltage": "not_applied",
            "voltage_to_current": "not_applied",
        }
    if readout_details.get("board_type") != "DC_In":
        raise ValueError("DC measure mode requires a DC_In readout")
    gain_v_per_a = float(readout_details.get("dc_measure_gain_v_per_a", 1.0))
    current_iq = dc_iq_to_current(raw_iq, gain_v_per_a)
    return current_iq, "A", "dc_current_iq", {
        "adc_to_voltage": "identity",
        "voltage_to_current": "current_a = voltage_v / gain_v_per_a",
        "gain_v_per_a": gain_v_per_a,
    }


def store_qick_result(
    ddr_result: Any,
    *,
    run_config: QcodesRunConfig,
    connection_config: QickConnectionConfig,
    program_summary: Mapping[str, Any],
    gui_settings: Mapping[str, Any],
    rf_settings: Mapping[str, Any],
    progress_callback: Optional[ProgressCallback] = None,
    progress_start: int = 65,
    progress_end: int = 99,
    batch_rows: int = DEFAULT_QCODES_BATCH_ROWS,
) -> Tuple[Any, int]:
    """Store one I array and one Q array per point/repetition acquisition."""
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
            "QCoDeS is required to save experiments; install qcodes==0.58.0"
        ) from exc

    raw_iq = np.asarray(ddr_result.iq)
    iq, iq_unit, measurement_mode, measurement_conversion = (
        _measurement_iq_values(raw_iq, rf_settings)
    )
    if iq.ndim != 4 or iq.shape[-1] != 2:
        raise ValueError("DDR IQ must have shape (point, repetition, sample, 2)")
    if isinstance(batch_rows, bool) or int(batch_rows) < 1:
        raise ValueError("batch_rows must be a positive integer")
    batch_rows = int(batch_rows)
    progress_start = int(progress_start)
    progress_end = int(progress_end)
    if not 0 <= progress_start <= progress_end <= 100:
        raise ValueError("progress range must satisfy 0 <= start <= end <= 100")
    point_count, repetition_count, sample_count, _ = iq.shape
    total_rows = point_count * repetition_count * sample_count
    if total_rows < 1:
        raise ValueError("DDR IQ result contains no samples")

    stored_gui_settings = dict(gui_settings)
    awg_vertices = stored_gui_settings.pop("awg_waveform_vertices", {})
    vertex_data = _coerce_awg_vertex_data(
        awg_vertices,
        point_count=point_count,
    )

    _emit_progress(progress_callback, progress_start, "Preparing QCoDeS database")
    sweep_axes = tuple(ddr_result.sweep_axes)
    coordinates = _sweep_coordinates(ddr_result)
    if not sweep_axes and point_count != 1:
        raise ValueError(
            "multiple DDR sweep points require at least one named sweep axis"
        )
    full_scale_mv = _full_scale_mv(stored_gui_settings)
    coordinates_mv = coordinates * full_scale_mv
    database_path = run_config.resolved_database_path
    database_path.parent.mkdir(parents=True, exist_ok=True)
    staging_directory, local_database_path = _prepare_local_database(database_path)
    initialise_or_create_database_at(str(local_database_path))
    experiment = load_or_create_experiment(
        run_config.experiment_name,
        run_config.sample_name,
    )
    measurement = Measurement(exp=experiment, station=Station())

    repetition_index = Parameter("repetition_index", label="Repetition", unit="")
    sample_index = Parameter(
        SAMPLE_INDEX_PARAMETER,
        label="Sample index",
        unit="",
    )
    setpoint_parameters = []
    sweep_parameters = []
    sweep_parameter_names = _sweep_parameter_names(sweep_axes)
    for axis, parameter_name in zip(sweep_axes, sweep_parameter_names):
        parameter = Parameter(
            parameter_name,
            label=f"{axis.output_name} / {axis.segment_name} voltage",
            unit="mV",
        )
        sweep_parameters.append(parameter)
        setpoint_parameters.append(parameter)
    setpoint_parameters.append(repetition_index)
    for parameter in setpoint_parameters:
        measurement.register_parameter(parameter)
    measurement.register_parameter(sample_index, paramtype="array")

    i_trace = Parameter(
        I_TRACE_PARAMETER,
        label="I current trace" if iq_unit == "A" else "I trace",
        unit=iq_unit,
    )
    q_trace = Parameter(
        Q_TRACE_PARAMETER,
        label="Q current trace" if iq_unit == "A" else "Q trace",
        unit=iq_unit,
    )
    for parameter in (i_trace, q_trace):
        measurement.register_parameter(
            parameter,
            # Plottr assigns the final dimension to the x-axis by default.
            setpoints=(*setpoint_parameters, sample_index),
            paramtype="array",
        )

    vertex_parameters = []
    if vertex_data is not None:
        output_names, vertex_time_us, _virtual_mv, _physical_mv = vertex_data
        used_prefixes = set()
        for output_index, output_name in enumerate(output_names):
            base_prefix = _qcodes_identifier(output_name)
            prefix = base_prefix
            suffix = 2
            while prefix in used_prefixes:
                prefix = f"{base_prefix}_{suffix}"
                suffix += 1
            used_prefixes.add(prefix)
            time_parameter = Parameter(
                f"{prefix}_vertex_time_us",
                label=f"{output_name} AWG vertex time",
                unit="us",
            )
            measurement.register_parameter(
                time_parameter,
                paramtype="array",
            )
            # Keep time last so Plottr opens a waveform against time.
            vertex_setpoints = (*sweep_parameters, time_parameter)
            virtual_parameter = Parameter(
                f"{prefix}_virtual_vertices_mv",
                label=f"{output_name} virtual AWG waveform vertices",
                unit="mV",
            )
            physical_parameter = Parameter(
                f"{prefix}_physical_vertices_mv",
                label=f"{output_name} physical AWG waveform vertices",
                unit="mV",
            )
            for parameter in (virtual_parameter, physical_parameter):
                measurement.register_parameter(
                    parameter,
                    setpoints=vertex_setpoints,
                    paramtype="array",
                )
            vertex_parameters.append({
                "output_index": output_index,
                "output_name": output_name,
                "time": time_parameter,
                "virtual": virtual_parameter,
                "physical": physical_parameter,
                "time_values": np.ascontiguousarray(
                    vertex_time_us,
                    dtype=float,
                ),
            })

    sweep_axis_metadata = [
        {
            "parameter": parameter.name,
            "output_name": axis.output_name,
            "segment_name": axis.segment_name,
            "unit": "mV",
            "normalized_start": float(axis.start),
            "normalized_stop": float(axis.stop),
            "voltage_start_mv": float(axis.start) * full_scale_mv,
            "voltage_stop_mv": float(axis.stop) * full_scale_mv,
            "count": int(axis.count),
        }
        for axis, parameter in zip(sweep_axes, sweep_parameters)
    ]
    setpoint_meanings = {
        parameter.name: (
            f"Voltage applied to {axis.output_name}/{axis.segment_name}; "
            "this is a directly selectable Cartesian sweep axis."
        )
        for axis, parameter in zip(sweep_axes, sweep_parameters)
    }
    setpoint_meanings.update({
        SAMPLE_INDEX_PARAMETER: (
            "Zero-based index of each sample within the captured I/Q trace."
        ),
        "repetition_index": "Zero-based repetition within one sweep coordinate.",
    })
    sample_period_us = 1_000_000.0 / run_config.sample_rate_hz
    sample_index_values = np.arange(sample_count, dtype=np.int32)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "qick_connection": asdict(connection_config),
        "qcodes_run": {
            **asdict(run_config),
            "database_path": str(database_path),
            "write_mode": "local_staging_then_sqlite_backup",
        },
        "gui_settings": stored_gui_settings,
        "program_summary": program_summary,
        "rf_settings_actual": rf_settings,
        "measurement_layout": {
            "iq_shape": list(iq.shape),
            "iq_trace_parameters": {
                "i": I_TRACE_PARAMETER,
                "q": Q_TRACE_PARAMETER,
            },
            "iq_trace_shape": [sample_count],
            "iq_trace_dtypes": {
                "i": str(iq[..., 0].dtype),
                "q": str(iq[..., 1].dtype),
            },
            "raw_iq_dtype": str(raw_iq.dtype),
            "iq_unit": iq_unit,
            "measurement_mode": measurement_mode,
            "measurement_conversion": dict(measurement_conversion),
            "storage_format": "qcodes_split_array_per_trace_v3",
            "trace_count": point_count * repetition_count,
            "sql_rows_per_trace": 2,
            "cartesian_point_count": point_count,
            "sample_rate_hz": run_config.sample_rate_hz,
            "sample_period_us": sample_period_us,
            "sample_index_parameter": SAMPLE_INDEX_PARAMETER,
            "row_order": (
                "sweep_axes,repetition"
                if sweep_axes
                else "repetition"
            ),
            "sweep_axes": sweep_axis_metadata,
            "setpoint_meanings": setpoint_meanings,
            "time_reconstruction": (
                f"{SAMPLE_INDEX_PARAMETER} is stored as the array setpoint for "
                "every I/Q trace; time_us = sample_index * sample_period_us."
            ),
            "derived_quantities": {
                "magnitude": "hypot(i_trace, q_trace)",
                "phase_deg": (
                    "degrees(arctan2(q_trace, i_trace))"
                ),
            },
        },
    }
    if vertex_data is not None:
        output_names, vertex_time_us, virtual_mv, _physical_mv = vertex_data
        parameter_by_output = {
            entry["output_name"]: entry for entry in vertex_parameters
        }
        metadata["measurement_layout"].update({
            "awg_vertex_shape": list(virtual_mv.shape),
            "awg_vertex_storage": "per_channel_timed_qcodes_array_v2",
            "awg_vertex_row_order": "one array per sweep point and channel",
            "awg_output_names": list(output_names),
            "awg_vertex_time_reference": "start of each pulse sequence repetition",
            "awg_vertex_channels": {
                output_name: {
                    "time_parameter": parameter_by_output[output_name][
                        "time"
                    ].name,
                    "virtual_parameter": parameter_by_output[output_name][
                        "virtual"
                    ].name,
                    "physical_parameter": parameter_by_output[output_name][
                        "physical"
                    ].name,
                    "time_us": vertex_time_us.tolist(),
                    "vertex_count": int(vertex_time_us.size),
                }
                for output_name in output_names
            },
        })

    row_count = 0
    with measurement.run(
        write_in_background=False,
        in_memory_cache=False,
    ) as datasaver:
        dataset = datasaver.dataset
        dataset.add_metadata("qick_experiment_json", _json_text(metadata))
        dataset.add_metadata(
            "output_waveforms_json",
            _json_text(
                stored_gui_settings.get("waveforms")
                or stored_gui_settings.get("awg", {}).get("outputs", [])
            ),
        )
        dataset.add_metadata(
            "cross_capacitance_json",
            _json_text(
                stored_gui_settings.get("awg", {}).get("cross_capacitance", [])
            ),
        )
        dataset.add_metadata(
            "rf_configuration_json",
            _json_text({
                "outputs": stored_gui_settings.get("rf_outputs", []),
                "readout": stored_gui_settings.get("rf_readout", {}),
            }),
        )
        if run_config.notes:
            dataset.add_metadata("experiment_notes", run_config.notes)

        if vertex_data is not None:
            output_names, _vertex_time_us, virtual_mv, physical_mv = vertex_data
            for point_index in range(point_count):
                coordinate_results = [
                    (
                        parameter,
                        float(coordinates_mv[point_index, axis_index]),
                    )
                    for axis_index, parameter in enumerate(sweep_parameters)
                ]
                channel_results = []
                for entry in vertex_parameters:
                    output_index = entry["output_index"]
                    channel_results.extend((
                        (
                            entry["time"],
                            entry["time_values"],
                        ),
                        (
                            entry["virtual"],
                            np.ascontiguousarray(
                                virtual_mv[point_index, output_index],
                                dtype=float,
                            ),
                        ),
                        (
                            entry["physical"],
                            np.ascontiguousarray(
                                physical_mv[point_index, output_index],
                                dtype=float,
                            ),
                        ),
                    ))
                datasaver.add_result(
                    *coordinate_results,
                    *channel_results,
                )
            datasaver.flush_data_to_database()
            _emit_progress(
                progress_callback,
                progress_start,
                f"Saved per-channel AWG vertex arrays for "
                f"{point_count:,} sweep points",
            )

        trace_count = point_count * repetition_count
        traces_per_flush = max(1, batch_rows // sample_count)
        traces_written = 0
        data_progress_end = max(progress_start, progress_end - 4)
        for point_index in range(point_count):
            coordinate_results = [
                (
                    parameter,
                    float(coordinates_mv[point_index, axis_index]),
                )
                for axis_index, parameter in enumerate(sweep_parameters)
            ]
            for repetition in range(repetition_count):
                datasaver.add_result(
                    *coordinate_results,
                    (repetition_index, repetition),
                    (sample_index, sample_index_values),
                    (
                        i_trace,
                        np.ascontiguousarray(
                            iq[point_index, repetition, :, 0]
                        ),
                    ),
                    (
                        q_trace,
                        np.ascontiguousarray(
                            iq[point_index, repetition, :, 1]
                        ),
                    ),
                )
                traces_written += 1
                row_count += sample_count
                if (
                    traces_written % traces_per_flush == 0
                    or traces_written == trace_count
                ):
                    datasaver.flush_data_to_database()
                    fraction = traces_written / trace_count
                    percent = progress_start + round(
                        fraction * (data_progress_end - progress_start)
                    )
                    _emit_progress(
                        progress_callback,
                        percent,
                        f"Saving split I/Q trace arrays {traces_written:,}/"
                        f"{trace_count:,} ({row_count:,}/{total_rows:,} samples)",
                    )

    local_guid = str(dataset.guid)
    dataset.conn.close()
    _emit_progress(
        progress_callback,
        max(progress_start, progress_end - 3),
        "Checkpointing local QCoDeS database",
    )
    _checkpoint_sqlite_database(local_database_path)
    _emit_progress(
        progress_callback,
        max(progress_start, progress_end - 2),
        "Copying completed QCoDeS database to the configured path",
    )
    _publish_local_database(local_database_path, database_path)
    initialise_or_create_database_at(str(database_path))
    dataset = load_by_guid(local_guid)
    shutil.rmtree(staging_directory)
    _emit_progress(
        progress_callback,
        progress_end,
        "QCoDeS database copied and WAL checkpoint completed",
    )
    return dataset, row_count


def run_qick_qcodes_experiment(
    *,
    connection_config: QickConnectionConfig,
    run_config: QcodesRunConfig,
    sequence: Any,
    awg_channels: Sequence[int],
    repetitions_per_sweep: int,
    rf_specs: Sequence[QickRfPulseSpec],
    readout_spec: QickDdrReadoutSpec,
    gui_settings: Mapping[str, Any],
    progress: bool = False,
    connector: Optional[Callable[..., Tuple[Any, Any]]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> StoredQickExperiment:
    """Connect, execute, acquire FIR DDR IQ, and commit one QCoDeS run."""
    _emit_progress(progress_callback, 0, "Starting QICK experiment")
    _emit_progress(progress_callback, 2, "Connecting to QICK Pyro server")
    soc, soccfg = connect_qick(connection_config, connector=connector)
    stored_gui_settings = dict(gui_settings)
    qick_settings = gui_settings.get("qick", {})
    if not isinstance(qick_settings, Mapping):
        raise TypeError("gui_settings['qick'] must be a mapping")
    tproc_mhz = float(
        qick_settings.get("tproc_mhz", DEFAULT_QICK_TPROC_MHZ)
    )
    if hasattr(sequence, "waveform_vertices"):
        _emit_progress(progress_callback, 4, "Building compact AWG vertices")
        stored_gui_settings["awg_waveform_vertices"] = build_awg_vertex_metadata(
            sequence,
            fabric_mhz=float(qick_settings.get("fabric_mhz", 300.0)),
            full_scale_mv=float(qick_settings.get("full_scale_mv", 100.0)),
        )
    program, ddr_result, rf_settings = execute_qick_sequence(
        soc,
        soccfg,
        sequence,
        awg_channels=awg_channels,
        repetitions_per_sweep=repetitions_per_sweep,
        tproc_mhz=tproc_mhz,
        rf_specs=rf_specs,
        readout_spec=readout_spec,
        progress=progress,
        progress_callback=progress_callback,
    )
    dataset, row_count = store_qick_result(
        ddr_result,
        run_config=run_config,
        connection_config=connection_config,
        program_summary=program.summary(),
        gui_settings=stored_gui_settings,
        rf_settings=rf_settings,
        progress_callback=progress_callback,
    )
    _emit_progress(progress_callback, 100, "Experiment saved")
    return StoredQickExperiment(
        run_id=int(dataset.run_id),
        guid=str(dataset.guid),
        database_path=run_config.resolved_database_path,
        row_count=row_count,
        dataset=dataset,
        program=program,
        ddr_result=ddr_result,
        rf_settings=rf_settings,
    )


__all__ = [
    "DEFAULT_QCODES_BATCH_ROWS",
    "I_TRACE_PARAMETER",
    "IQ_TRACE_PARAMETER",
    "Q_TRACE_PARAMETER",
    "QCODES_STAGING_ENV",
    "ProgressCallback",
    "QcodesRunConfig",
    "QickConnectionConfig",
    "StoredQickExperiment",
    "build_awg_vertex_metadata",
    "build_qick_program",
    "build_runtime_ddr_readout",
    "build_runtime_rf_pulses",
    "configure_rf_board",
    "connect_qick",
    "execute_qick_sequence",
    "load_qick_iq_arrays",
    "run_qick_qcodes_experiment",
    "store_qick_result",
]
