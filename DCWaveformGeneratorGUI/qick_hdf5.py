"""One HDF5 file per experiment, grouped by the local start date."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from uuid import uuid4

import numpy as np


@dataclass
class Hdf5StoredRecord:
    path: Path
    guid: str
    run_id: int = 0


def prepare_hdf5_directory(directory):
    """Check the optional dependency and output directory before acquisition."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("HDF5 storage requires h5py; install the GUI requirements") from exc
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Check write access without changing or replacing existing files.
    import tempfile
    with tempfile.TemporaryFile(dir=root):
        pass
    return root


def _open_unique_file(root, experiment_name, started_at):
    import h5py
    day = root / started_at.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(experiment_name)).strip(" .")[:100] or "experiment"
    stem = started_at.strftime("%Y%m%d_%H%M%S") + "_" + name
    for suffix in range(10000):
        path = day / (stem + (f"_{suffix:03d}" if suffix else "") + ".h5")
        try:
            return path, h5py.File(path, "x")
        except FileExistsError:
            continue
    raise FileExistsError("Too many HDF5 files with the same experiment timestamp")


def store_hdf5_result(result, *, run_config, connection_config, program_summary,
                       gui_settings, rf_settings, iq_storage_mode,
                       started_at=None, progress_callback=None,
                       cancel_check=None, **unused):
    import h5py
    try:
        from .qick_qcodes_experiment import (
            _json_text, _sweep_coordinates, _sweep_axis_display, _full_scale_mv,
            _json_ready, ExperimentCancelled, measurement_iq_values,
        )
        from .fir_ddr_profile import result_iq_in_input_units
    except ImportError:
        from qick_qcodes_experiment import (
            _json_text, _sweep_coordinates, _sweep_axis_display, _full_scale_mv,
            _json_ready, ExperimentCancelled, measurement_iq_values,
        )
        from fir_ddr_profile import result_iq_in_input_units
    from dataclasses import asdict

    root = prepare_hdf5_directory(run_config.hdf5_directory)
    started_at = started_at or datetime.now().astimezone()
    if started_at.tzinfo is None:
        raise ValueError("HDF5 experiment start time must include a timezone")
    iq = np.asarray(result.iq)
    if iq.ndim != 4 or iq.shape[-1] != 2 or any(n < 1 for n in iq.shape):
        raise ValueError("Acquisition IQ must have shape (point, repetition, sample, 2)")
    point_count, stored_reps, stored_samples, _ = iq.shape
    if np.prod(result.sweep_shape) != point_count:
        raise ValueError("Sweep shape does not match the acquired point count")
    kind = getattr(result, "data_kind", "raw")
    reps = int(getattr(result, "accumulation_repetitions", stored_reps)) if kind != "raw" else stored_reps
    source_samples = getattr(result, "source_samples_per_trace", None) or stored_samples
    scale = int(getattr(result, "iq_scale_log2", 0))
    if scale and iq.dtype != np.dtype("int64"):
        raise ValueError("Fixed-point IQ64 must retain exact int64 raw samples")
    scalar = iq_storage_mode in {"mean_iq", "qick_mean_iq"}
    if scalar:
        payload = result_iq_in_input_units(result).mean(axis=(1, 2), dtype=np.float64)
        axes = ("point", "iq")
        scale = 0
    elif kind == "repetition_mean":
        payload = iq[:, 0]
        axes = ("point", "sample", "iq")
    else:
        payload = iq
        axes = ("point", "repetition", "sample", "iq")
    coordinates = _sweep_coordinates(result)
    display_coordinates = coordinates.copy()
    axis_metadata = []
    for index, axis in enumerate(result.sweep_axes):
        quantity, unit, factor = _sweep_axis_display(axis, _full_scale_mv(gui_settings))
        display_coordinates[:, index] *= factor
        axis_metadata.append({**_json_ready(axis), "display_quantity": quantity, "display_unit": unit})
    guid = str(uuid4())
    # Snapshot any fitted calibration formula while its source DB is available.
    # The primary dataset remains raw/fixed-point or averaged input-code units.
    _, display_unit, display_mode, conversion = measurement_iq_values(np.zeros((1, 1, 1, 2)), rf_settings)
    metadata = dict(
        schema="qick_hdf5_v1", guid=guid,
        started_at=started_at.isoformat(),
        run_config=asdict(run_config), connection=asdict(connection_config),
        gui_settings=gui_settings, program_summary=program_summary,
        rf_settings_actual=rf_settings, sweep_axes=axis_metadata,
        measurement_representation=dict(unit=display_unit, mode=display_mode, conversion=conversion),
        sweep_shape=list(result.sweep_shape),
        sweep_groups=list(getattr(result, "sweep_groups", ())),
        sweep_parameter_ids=list(getattr(result, "sweep_parameter_ids", ())),
        acquisition=dict(
            source=getattr(result, "acquisition_source", "fir_ddr"),
            data_kind=kind, iq_storage_mode=iq_storage_mode,
            repetitions=reps, source_samples_per_trace=int(source_samples),
            received_shape=list(iq.shape), stored_shape=list(payload.shape),
            averaging_location="qick_cpu" if kind != "raw" else "pc" if scalar else "none",
            averaged_axes=["repetition", "sample"] if scalar else ["repetition"] if kind == "repetition_mean" else [],
            source_iq_scale_log2=getattr(result, "source_iq_scale_log2", None),
            source_iq_component_bits=getattr(result, "source_iq_component_bits", None),
        ),
    )
    if cancel_check:
        cancel_check()
    path, handle = _open_unique_file(root, run_config.experiment_name, started_at)
    with handle as h5:
        h5.attrs.update(schema="qick_hdf5_v1", status="writing", guid=guid,
                        started_at=started_at.isoformat(), completed_points=0)
        try:
            h5.create_dataset("metadata/json", data=_json_text(metadata), dtype=h5py.string_dtype("utf-8"))
            h5.create_dataset("sweep/coordinates", data=coordinates)
            h5.create_dataset("sweep/display_coordinates", data=display_coordinates)
            h5.create_dataset("sweep/shape", data=np.asarray(result.sweep_shape, dtype=np.int64))
            if result.cross_capacitance is not None:
                h5.create_dataset("sweep/cross_capacitance", data=result.cross_capacitance)
            if not scalar:
                time_axis = h5.create_dataset("axes/time_s", data=np.arange(stored_samples) / float(result.sample_rate_hz))
                time_axis.attrs.update(unit="s", reference="first captured sample")
            data = h5.create_dataset("data/iq", shape=payload.shape, dtype=payload.dtype,
                                     chunks=True, compression="lzf", shuffle=True)
            data.attrs.update(axes=json.dumps(axes), components="I,Q", iq_scale_log2=scale,
                              unit="fixed_point_input_code" if scale else "input_code",
                              sample_rate_hz=float(result.sample_rate_hz),
                              repetitions=reps, source_samples_per_trace=int(source_samples))
            for index, label in enumerate(axes):
                data.dims[index].label = label
            # Bound each write and preserve valid progress after cancellation.
            point_bytes = max(1, payload[0].nbytes)
            batch = max(1, (8*1024*1024) // point_bytes)
            for first in range(0, point_count, batch):
                if cancel_check:
                    cancel_check()
                end = min(point_count, first + batch)
                data[first:end] = payload[first:end]
                h5.attrs["completed_points"] = end
                h5.flush()
                if progress_callback:
                    progress_callback(65 + round(34 * end / point_count), f"Saving HDF5 points {end:,}/{point_count:,}")
            if cancel_check:
                cancel_check()
            h5.attrs["status"] = "completed"
        except BaseException as exc:
            h5.attrs["status"] = "cancelled" if isinstance(exc, ExperimentCancelled) else "failed"
            h5.attrs["error"] = str(exc)
            raise
        finally:
            h5.attrs["finished_at"] = datetime.now().astimezone().isoformat()
    return Hdf5StoredRecord(path, guid), int(payload.size // 2)


def load_hdf5_result(path):
    """Read the recorded arrays and metadata; reject unfinished experiments."""
    import h5py
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema") != "qick_hdf5_v1" or h5.attrs.get("status") != "completed":
            raise ValueError("Not a completed QICK HDF5 experiment")
        return dict(iq=h5["data/iq"][:], metadata=json.loads(h5["metadata/json"].asstr()[()]),
                    iq_scale_log2=int(h5["data/iq"].attrs["iq_scale_log2"]),
                    coordinates=h5["sweep/coordinates"][:],
                    time_s=h5["axes/time_s"][:] if "axes/time_s" in h5 else None)
