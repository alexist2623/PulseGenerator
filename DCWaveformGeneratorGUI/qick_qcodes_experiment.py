"""Execute a QICK fine-tune program and persist its IQ trace with QCoDeS.

QCoDeS and the QICK Pyro client are imported only when an experiment runs, so
the waveform editor remains usable on machines without laboratory packages.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np


ProgressCallback = Callable[[int, str], None]
DEFAULT_QCODES_BATCH_ROWS = 8192

try:
    from .dc_waveform_core import QickDdrReadoutSpec, QickRfPulseSpec
except ImportError:
    from dc_waveform_core import QickDdrReadoutSpec, QickRfPulseSpec


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
) -> Tuple[Any, ...]:
    """Convert GUI RF timing in microseconds to actual QICK clock cycles."""
    _ddr_type, rf_type, cycles_from_us = _runtime_types()
    tproc_mhz = float(soccfg["tprocs"][0]["f_time"])
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
) -> Any:
    """Convert the GUI's FIR DDR readout settings to program timing."""
    ddr_type, _rf_type, cycles_from_us = _runtime_types()
    tproc_mhz = float(soccfg["tprocs"][0]["f_time"])
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


def configure_rf_board(
    soc,
    rf_specs: Sequence[QickRfPulseSpec],
    readout_spec: QickDdrReadoutSpec,
) -> Mapping[str, Any]:
    """Apply RF-board output attenuation and input attenuation/filter."""
    outputs = tuple(
        soc.rfb_set_gen_rf(spec.gen_ch, spec.att1_db, spec.att2_db)
        for spec in rf_specs
    )
    readout_attenuation = soc.rfb_set_ro_rf(
        readout_spec.ro_ch, readout_spec.attenuation_db
    )
    soc.rfb_set_ro_filter(
        readout_spec.ro_ch,
        fc=readout_spec.filter_cutoff,
        bw=readout_spec.filter_bandwidth,
        ftype=readout_spec.filter_type,
    )
    return {
        "outputs": outputs,
        "readout": readout_attenuation,
    }


def execute_qick_sequence(
    soc,
    soccfg,
    sequence,
    *,
    awg_channels: Sequence[int],
    repetitions_per_sweep: int,
    rf_specs: Sequence[QickRfPulseSpec],
    readout_spec: QickDdrReadoutSpec,
    progress: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Any, Any, Mapping[str, Any]]:
    """Configure RF hardware, execute the tProcessor program, and read DDR."""
    _emit_progress(progress_callback, 5, "Configuring RF hardware")
    rf_settings = configure_rf_board(soc, rf_specs, readout_spec)
    _emit_progress(progress_callback, 8, "Compiling the tProcessor program")
    program = sequence.make_program(
        soccfg,
        awg_channels=tuple(int(channel) for channel in awg_channels),
        repetitions_per_sweep=int(repetitions_per_sweep),
        rf_pulses=build_runtime_rf_pulses(soccfg, rf_specs),
        ddr_readout=build_runtime_ddr_readout(soccfg, readout_spec),
    )
    _emit_progress(
        progress_callback,
        10,
        "Running pulse sequence and acquiring FIR DDR data",
    )

    def counter_progress(completed: int, total: int) -> None:
        fraction = 1.0 if total <= 0 else completed / total
        percent = 10 + round(max(0.0, min(1.0, fraction)) * 45)
        _emit_progress(
            progress_callback,
            percent,
            f"Running sweep/repetitions {completed:,}/{total:,}",
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
    """Write grouped DDR IQ and metadata to QCoDeS in bounded batches."""
    try:
        from qcodes import (
            Measurement,
            Parameter,
            Station,
            initialise_or_create_database_at,
            load_or_create_experiment,
        )
    except ImportError as exc:
        raise RuntimeError(
            "QCoDeS is required to save experiments; install qcodes==0.58.0"
        ) from exc

    iq = np.asarray(ddr_result.iq)
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

    _emit_progress(progress_callback, progress_start, "Preparing QCoDeS database")
    coordinates = _sweep_coordinates(ddr_result)
    database_path = run_config.resolved_database_path
    database_path.parent.mkdir(parents=True, exist_ok=True)
    initialise_or_create_database_at(str(database_path))
    experiment = load_or_create_experiment(
        run_config.experiment_name,
        run_config.sample_name,
    )
    measurement = Measurement(exp=experiment, station=Station())

    point_index = Parameter(
        "point_index",
        label="Flattened Cartesian sweep point (zero-based)",
        unit="",
    )
    repetition_index = Parameter("repetition_index", label="Repetition", unit="")
    sample_index = Parameter("sample_index", label="Trace sample", unit="")
    time_us = Parameter(
        "time_us",
        label="Time within each acquired IQ trace",
        unit="us",
    )
    setpoint_parameters = [point_index]
    sweep_parameters = []
    for axis_index, axis in enumerate(tuple(ddr_result.sweep_axes)):
        parameter = Parameter(
            f"sweep_{axis_index}",
            label=f"{axis.output_name} {axis.segment_name} amplitude",
            unit="normalized",
        )
        sweep_parameters.append(parameter)
        setpoint_parameters.append(parameter)
    setpoint_parameters.extend((repetition_index, sample_index, time_us))
    for parameter in setpoint_parameters:
        measurement.register_parameter(parameter)

    i_value = Parameter("i", label="I", unit="ADC units")
    q_value = Parameter("q", label="Q", unit="ADC units")
    magnitude = Parameter("magnitude", label="IQ magnitude", unit="ADC units")
    phase_deg = Parameter("phase_deg", label="IQ phase", unit="deg")
    dependencies = tuple(setpoint_parameters)
    for parameter in (i_value, q_value, magnitude, phase_deg):
        measurement.register_parameter(parameter, setpoints=dependencies)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "qick_connection": asdict(connection_config),
        "qcodes_run": {
            **asdict(run_config),
            "database_path": str(database_path),
        },
        "gui_settings": gui_settings,
        "program_summary": program_summary,
        "rf_settings_actual": rf_settings,
        "measurement_layout": {
            "iq_shape": list(iq.shape),
            "sample_rate_hz": run_config.sample_rate_hz,
            "sample_period_us": 1_000_000.0 / run_config.sample_rate_hz,
            "row_order": "point,repetition,sample",
            "setpoint_meanings": {
                "point_index": (
                    "Zero-based flattened Cartesian sweep index; the last "
                    "configured sweep axis varies fastest."
                ),
                "repetition_index": "Zero-based repetition within one sweep point.",
                "sample_index": "Zero-based sample within one triggered IQ trace.",
                "time_us": (
                    "Physical time within each IQ trace, reset to zero for every "
                    "trigger; time_us = sample_index / sample_rate_hz."
                ),
            },
        },
    }

    row_count = 0
    sample_period_us = 1_000_000.0 / run_config.sample_rate_hz
    awg_vertices = gui_settings.get("awg_waveform_vertices", {})
    with measurement.run(
        write_in_background=False,
        in_memory_cache=False,
    ) as datasaver:
        dataset = datasaver.dataset
        dataset.add_metadata("qick_experiment_json", _json_text(metadata))
        dataset.add_metadata(
            "output_waveforms_json",
            _json_text(awg_vertices or gui_settings.get("waveforms", {})),
        )
        dataset.add_metadata(
            "awg_virtual_vertices_json",
            _json_text(awg_vertices.get("virtual", {})),
        )
        dataset.add_metadata(
            "awg_physical_vertices_json",
            _json_text(awg_vertices.get("physical", {})),
        )
        dataset.add_metadata(
            "cross_capacitance_json",
            _json_text(gui_settings.get("awg", {}).get("cross_capacitance", [])),
        )
        dataset.add_metadata(
            "rf_configuration_json",
            _json_text({
                "outputs": gui_settings.get("rf_outputs", []),
                "readout": gui_settings.get("rf_readout", {}),
            }),
        )
        if run_config.notes:
            dataset.add_metadata("experiment_notes", run_config.notes)

        flat_iq = iq.reshape(total_rows, 2)
        rows_per_point = repetition_count * sample_count
        for offset in range(0, total_rows, batch_rows):
            end = min(offset + batch_rows, total_rows)
            linear_index = np.arange(offset, end, dtype=np.int64)
            sample_values = linear_index % sample_count
            repetition_values = (
                linear_index // sample_count
            ) % repetition_count
            point_values = linear_index // rows_per_point
            i_values = flat_iq[offset:end, 0].astype(float, copy=False)
            q_values = flat_iq[offset:end, 1].astype(float, copy=False)
            coordinate_results = [
                (parameter, coordinates[point_values, axis_index])
                for axis_index, parameter in enumerate(sweep_parameters)
            ]
            datasaver.add_result(
                (point_index, point_values),
                *coordinate_results,
                (repetition_index, repetition_values),
                (sample_index, sample_values),
                (time_us, sample_values * sample_period_us),
                (i_value, i_values),
                (q_value, q_values),
                (magnitude, np.hypot(i_values, q_values)),
                (phase_deg, np.degrees(np.arctan2(q_values, i_values))),
            )
            datasaver.flush_data_to_database()
            row_count = end
            fraction = row_count / total_rows
            percent = progress_start + round(
                fraction * (progress_end - progress_start)
            )
            _emit_progress(
                progress_callback,
                percent,
                f"Saving QCoDeS IQ rows {row_count:,}/{total_rows:,}",
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
    if hasattr(sequence, "waveform_vertices"):
        qick_settings = gui_settings.get("qick", {})
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
    "ProgressCallback",
    "QcodesRunConfig",
    "QickConnectionConfig",
    "StoredQickExperiment",
    "build_awg_vertex_metadata",
    "build_runtime_ddr_readout",
    "build_runtime_rf_pulses",
    "configure_rf_board",
    "connect_qick",
    "execute_qick_sequence",
    "run_qick_qcodes_experiment",
    "store_qick_result",
]
