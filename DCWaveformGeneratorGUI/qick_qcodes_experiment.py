"""Execute a QICK fine-tune program and persist its IQ trace with QCoDeS.

QCoDeS and the QICK Pyro client are imported only when an experiment runs, so
the waveform editor remains usable on machines without laboratory packages.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
from math import hypot
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

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
) -> Tuple[Any, Any, Mapping[str, Any]]:
    """Configure RF hardware, execute the tProcessor program, and read DDR."""
    rf_settings = configure_rf_board(soc, rf_specs, readout_spec)
    program = sequence.make_program(
        soccfg,
        awg_channels=tuple(int(channel) for channel in awg_channels),
        repetitions_per_sweep=int(repetitions_per_sweep),
        rf_pulses=build_runtime_rf_pulses(soccfg, rf_specs),
        ddr_readout=build_runtime_ddr_readout(soccfg, readout_spec),
    )
    ddr_result = program.acquire_fir_ddr(soc, progress=progress)
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
) -> Tuple[Any, int]:
    """Write a grouped DDR IQ result and complete configuration to QCoDeS."""
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
    coordinates = _sweep_coordinates(ddr_result)
    database_path = run_config.resolved_database_path
    database_path.parent.mkdir(parents=True, exist_ok=True)
    initialise_or_create_database_at(str(database_path))
    experiment = load_or_create_experiment(
        run_config.experiment_name,
        run_config.sample_name,
    )
    measurement = Measurement(exp=experiment, station=Station())

    point_index = Parameter("point_index", label="Cartesian sweep point", unit="")
    repetition_index = Parameter("repetition_index", label="Repetition", unit="")
    sample_index = Parameter("sample_index", label="Trace sample", unit="")
    time_us = Parameter("time_us", label="Trace time", unit="us")
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
        },
    }

    row_count = 0
    sample_period_us = 1_000_000.0 / run_config.sample_rate_hz
    with measurement.run() as datasaver:
        dataset = datasaver.dataset
        dataset.add_metadata("qick_experiment_json", _json_text(metadata))
        dataset.add_metadata(
            "output_waveforms_json",
            _json_text(gui_settings.get("waveforms", {})),
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

        for point in range(iq.shape[0]):
            coordinate_results = [
                (parameter, float(coordinates[point, axis_index]))
                for axis_index, parameter in enumerate(sweep_parameters)
            ]
            for repetition in range(iq.shape[1]):
                for sample in range(iq.shape[2]):
                    i_sample = float(iq[point, repetition, sample, 0])
                    q_sample = float(iq[point, repetition, sample, 1])
                    datasaver.add_result(
                        (point_index, point),
                        *coordinate_results,
                        (repetition_index, repetition),
                        (sample_index, sample),
                        (time_us, sample * sample_period_us),
                        (i_value, i_sample),
                        (q_value, q_sample),
                        (magnitude, hypot(i_sample, q_sample)),
                        (phase_deg, float(np.degrees(np.arctan2(q_sample, i_sample)))),
                    )
                    row_count += 1
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
) -> StoredQickExperiment:
    """Connect, execute, acquire FIR DDR IQ, and commit one QCoDeS run."""
    soc, soccfg = connect_qick(connection_config, connector=connector)
    program, ddr_result, rf_settings = execute_qick_sequence(
        soc,
        soccfg,
        sequence,
        awg_channels=awg_channels,
        repetitions_per_sweep=repetitions_per_sweep,
        rf_specs=rf_specs,
        readout_spec=readout_spec,
        progress=progress,
    )
    dataset, row_count = store_qick_result(
        ddr_result,
        run_config=run_config,
        connection_config=connection_config,
        program_summary=program.summary(),
        gui_settings=gui_settings,
        rf_settings=rf_settings,
    )
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
    "QcodesRunConfig",
    "QickConnectionConfig",
    "StoredQickExperiment",
    "build_runtime_ddr_readout",
    "build_runtime_rf_pulses",
    "configure_rf_board",
    "connect_qick",
    "execute_qick_sequence",
    "run_qick_qcodes_experiment",
    "store_qick_result",
]
