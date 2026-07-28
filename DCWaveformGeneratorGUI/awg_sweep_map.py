"""Two-dimensional maps reduced from Cartesian AWG tuning sweeps.

The hardware acquisition stores one I/Q trace for every Cartesian sweep point
and repetition. This module coherently averages I and Q over repetitions and
FIR samples, then arranges two user-selected sweep variables on X and Y.
Additional sweep variables are averaged at each selected X/Y coordinate.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import traceback
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

try:
    import pyqtgraph as pg
except ImportError:
    pg = None

try:
    from .dc_waveform_core import DEFAULT_QICK_FULL_SCALE_MV
    from .fir_ddr_profile import format_sample_rate_hz
    from .measurement_display import (
        ColorRangeControl,
        attach_color_bar,
        scale_iq_for_display,
    )
    from .qick_qcodes_experiment import load_qick_iq_arrays
except ImportError:
    from dc_waveform_core import DEFAULT_QICK_FULL_SCALE_MV
    from fir_ddr_profile import format_sample_rate_hz
    from measurement_display import (
        ColorRangeControl,
        attach_color_bar,
        scale_iq_for_display,
    )
    from qick_qcodes_experiment import load_qick_iq_arrays


SweepAxisKey = Tuple[str, str]
DEFAULT_AWG_SWEEP_DB_PATH = str(Path.home() / "qick_experiments.db")
DEFAULT_AWG_SWEEP_COLOR_RANGES = {
    "i": {"auto": True, "minimum": -1.0, "maximum": 1.0},
    "q": {"auto": True, "minimum": -1.0, "maximum": 1.0},
    "magnitude": {"auto": True, "minimum": 0.0, "maximum": 1.0},
    "angle": {"auto": False, "minimum": -180.0, "maximum": 180.0},
}
AWG_SWEEP_DATA_KEYS = ("i", "q", "magnitude", "angle")
DEFAULT_AWG_SWEEP_VISIBLE_DATA = AWG_SWEEP_DATA_KEYS


def normalize_awg_sweep_color_ranges(
    settings: Optional[Mapping[str, Any]],
) -> dict:
    """Validate persisted AWG map color ranges with backward-safe defaults."""
    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise TypeError("AWG sweep color_ranges must be a JSON object")
    normalized = {}
    for name, defaults in DEFAULT_AWG_SWEEP_COLOR_RANGES.items():
        raw = settings.get(name, defaults)
        if not isinstance(raw, Mapping):
            raise TypeError(f"AWG sweep {name} color range must be an object")
        auto = raw.get("auto", defaults["auto"])
        if not isinstance(auto, (bool, np.bool_)):
            raise TypeError(f"AWG sweep {name} auto range must be boolean")
        minimum = float(raw.get("minimum", defaults["minimum"]))
        maximum = float(raw.get("maximum", defaults["maximum"]))
        if (
            not np.isfinite(minimum)
            or not np.isfinite(maximum)
            or minimum >= maximum
        ):
            raise ValueError(
                f"AWG sweep {name} color minimum must be finite and below maximum"
            )
        normalized[name] = {
            "auto": bool(auto),
            "minimum": minimum,
            "maximum": maximum,
        }
    return normalized


def normalize_awg_sweep_visible_data(values: Any) -> Tuple[str, ...]:
    """Validate the ordered set of maps shown in the AWG 2-D result dock."""
    if values is None:
        return DEFAULT_AWG_SWEEP_VISIBLE_DATA
    if not isinstance(values, (list, tuple)):
        raise TypeError("AWG sweep visible_data must be a JSON array")
    normalized = []
    for value in values:
        key = str(value).strip().lower()
        if key not in AWG_SWEEP_DATA_KEYS:
            raise ValueError(f"unknown AWG sweep plot data {value!r}")
        if key not in normalized:
            normalized.append(key)
    if not normalized:
        raise ValueError("at least one AWG sweep plot must be visible")
    return tuple(normalized)


def sweep_axis_key(axis: Any) -> SweepAxisKey:
    """Return the stable ``(output_name, segment_name)`` key for one axis."""
    return str(axis.output_name), str(axis.segment_name)


def sweep_axis_label(axis: Any) -> str:
    """Return the compact user-facing name for one sweep variable."""
    output_name, segment_name = sweep_axis_key(axis)
    axis_kind = getattr(axis, "axis_kind", "amplitude")
    if axis_kind == "rf_duration":
        return f"{output_name} / {segment_name} RF duration"
    if axis_kind == "ramp_duration":
        return f"{segment_name} RAMP duration (rate derived)"
    return f"{output_name} / {segment_name}"


def _axis_display_values(
    coordinates: np.ndarray,
    axis: Any,
    full_scale_mv: float,
) -> Tuple[np.ndarray, str]:
    if getattr(axis, "axis_kind", "amplitude") in {
        "rf_duration",
        "ramp_duration",
    }:
        return np.asarray(coordinates, dtype=np.float64), "us"
    return np.asarray(coordinates, dtype=np.float64) * full_scale_mv, "mV"


@dataclass(frozen=True)
class AwgSweepMapResult:
    """I/Q and derived values on two selected AWG sweep axes."""

    x_values: np.ndarray
    y_values: np.ndarray
    i_mean: np.ndarray
    q_mean: np.ndarray
    magnitude: np.ndarray
    angle_deg: np.ndarray
    x_axis_key: SweepAxisKey
    y_axis_key: SweepAxisKey
    x_axis_label: str
    y_axis_label: str
    x_unit: str
    y_unit: str
    value_unit: str
    base_value_unit: str
    display_scale: float
    measurement_mode: str
    repetition_count: int
    samples_per_trace: int
    averaged_axis_labels: Tuple[str, ...]
    source_points_per_cell: int
    sample_rate_hz: float
    source_label: str = ""
    database_path: str = ""
    run_id: int = 0

    @property
    def x_values_mv(self) -> np.ndarray:
        """Backward-compatible alias for pre-duration-sweep callers."""
        return self.x_values

    @property
    def y_values_mv(self) -> np.ndarray:
        """Backward-compatible alias for pre-duration-sweep callers."""
        return self.y_values


@dataclass(frozen=True)
class AwgSweepRunSummary:
    """Metadata-only description of one saved AWG Cartesian sweep."""

    database_path: str
    run_id: int
    created_at_utc: str
    x_axis_label: str
    y_axis_label: str
    x_points: int
    y_points: int
    sweep_axis_count: int
    iq_unit: str
    sample_rate_hz: float

    @property
    def display_label(self) -> str:
        timestamp = self.created_at_utc.replace("T", " ")[:19]
        timestamp_text = f" | {timestamp}" if timestamp else ""
        averaged_count = max(0, self.sweep_axis_count - 2)
        averaged_text = (
            f" | +{averaged_count} averaged axis/axes"
            if averaged_count
            else ""
        )
        return (
            f"Run {self.run_id}{timestamp_text} | "
            f"{self.x_axis_label} x {self.y_axis_label} | "
            f"{self.x_points} x {self.y_points}{averaged_text} | "
            f"{format_sample_rate_hz(self.sample_rate_hz)} | {self.iq_unit}"
        )


def _stored_sweep_axes(
    metadata: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], ...]:
    layout = metadata.get("measurement_layout", {})
    if not isinstance(layout, Mapping):
        raise ValueError("stored QICK measurement layout is missing")
    raw_axes = layout.get("sweep_axes", ())
    if not isinstance(raw_axes, Sequence) or isinstance(raw_axes, (str, bytes)):
        raise ValueError("stored QICK sweep-axis metadata is invalid")
    axes = tuple(axis for axis in raw_axes if isinstance(axis, Mapping))
    if len(axes) < 2:
        raise ValueError("a saved AWG 2D map requires at least two sweep axes")
    return axes


def _stored_axis_key(axis: Mapping[str, Any]) -> SweepAxisKey:
    return str(axis.get("output_name", "")), str(axis.get("segment_name", ""))


def _stored_axis_label(axis: Mapping[str, Any]) -> str:
    output_name, segment_name = _stored_axis_key(axis)
    axis_kind = str(axis.get("axis_kind", "amplitude"))
    if axis_kind == "rf_duration":
        return f"{output_name} / {segment_name} RF duration"
    if axis_kind == "ramp_duration":
        return f"{segment_name} RAMP duration (rate derived)"
    return f"{output_name} / {segment_name}"


def _stored_selected_axes(
    metadata: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Resolve the X/Y axes selected when the QCoDeS run was saved."""
    axes = _stored_sweep_axes(metadata)
    by_key = {_stored_axis_key(axis): axis for axis in axes}
    gui_settings = metadata.get("gui_settings", {})
    experiment_settings = (
        gui_settings.get("experiment", {})
        if isinstance(gui_settings, Mapping)
        else {}
    )
    sweep_map_settings = (
        experiment_settings.get("sweep_map", {})
        if isinstance(experiment_settings, Mapping)
        else {}
    )
    if not isinstance(sweep_map_settings, Mapping):
        sweep_map_settings = {}

    requested = []
    for name in ("x_axis", "y_axis"):
        axis_settings = sweep_map_settings.get(name, {})
        if not isinstance(axis_settings, Mapping):
            requested.append(("", ""))
            continue
        requested.append(
            (
                str(axis_settings.get("output_name", "")),
                str(axis_settings.get("segment_name", "")),
            )
        )
    if (
        len(requested) == 2
        and requested[0] != requested[1]
        and all(key in by_key for key in requested)
    ):
        return by_key[requested[0]], by_key[requested[1]]
    return axes[0], axes[1]


def _is_awg_sweep_metadata(metadata: Mapping[str, Any]) -> bool:
    """Return true for regular AWG experiments with a plottable 2D sweep."""
    try:
        _stored_sweep_axes(metadata)
    except (TypeError, ValueError):
        return False
    gui_settings = metadata.get("gui_settings", {})
    qick_settings = (
        gui_settings.get("qick", {})
        if isinstance(gui_settings, Mapping)
        else {}
    )
    return not (
        isinstance(qick_settings, Mapping)
        and qick_settings.get("fir_stability_capture_mode")
        == "immediate_continuous_fir_output"
    )


def list_awg_sweep_runs(
    database_path: Any,
) -> Tuple[AwgSweepRunSummary, ...]:
    """List compatible saved AWG sweep runs without loading trace arrays."""
    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"QCoDeS database does not exist: {path}")

    connection = sqlite3.connect(str(path), timeout=30.0)
    try:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(runs)")
        }
        if "qick_experiment_json" not in columns:
            return ()
        rows = connection.execute(
            "SELECT run_id, qick_experiment_json FROM runs "
            "WHERE qick_experiment_json IS NOT NULL "
            "AND qick_experiment_json != '' "
            "ORDER BY run_id DESC"
        ).fetchall()
    finally:
        connection.close()

    summaries = []
    for run_id, payload_text in rows:
        try:
            metadata = json.loads(payload_text)
            if not isinstance(metadata, Mapping) or not _is_awg_sweep_metadata(
                metadata
            ):
                continue
            axes = _stored_sweep_axes(metadata)
            x_axis, y_axis = _stored_selected_axes(metadata)
            layout = metadata.get("measurement_layout", {})
            sample_rate_hz = float(
                layout.get(
                    "sample_rate_hz",
                    1.0e6 / float(layout.get("sample_period_us", 1.0)),
                )
            )
            summaries.append(
                AwgSweepRunSummary(
                    database_path=str(path),
                    run_id=int(run_id),
                    created_at_utc=str(metadata.get("created_at_utc", "")),
                    x_axis_label=_stored_axis_label(x_axis),
                    y_axis_label=_stored_axis_label(y_axis),
                    x_points=int(x_axis.get("count", 0)),
                    y_points=int(y_axis.get("count", 0)),
                    sweep_axis_count=len(axes),
                    iq_unit=str(layout.get("iq_unit", "ADC units")),
                    sample_rate_hz=sample_rate_hz,
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return tuple(summaries)


def _stored_full_scale_mv(metadata: Mapping[str, Any]) -> float:
    gui_settings = metadata.get("gui_settings", {})
    qick_settings = (
        gui_settings.get("qick", {})
        if isinstance(gui_settings, Mapping)
        else {}
    )
    full_scale_mv = float(
        qick_settings.get("full_scale_mv", DEFAULT_QICK_FULL_SCALE_MV)
    )
    if not np.isfinite(full_scale_mv) or full_scale_mv <= 0.0:
        raise ValueError("stored AWG full scale must be positive and finite")
    return full_scale_mv


def _stored_axis_native_values(
    values: Any,
    axis: Mapping[str, Any],
    *,
    full_scale_mv: float,
) -> np.ndarray:
    """Convert QCoDeS display units back to the in-memory sweep convention."""
    coordinates = np.asarray(values, dtype=np.float64)
    unit = str(axis.get("unit", "")).strip().lower()
    axis_kind = str(axis.get("axis_kind", "amplitude"))
    if axis_kind in {"rf_duration", "ramp_duration"}:
        duration_scales = {
            "": 1.0,
            "us": 1.0,
            "µs": 1.0,
            "ns": 1.0e-3,
            "ms": 1.0e3,
            "s": 1.0e6,
        }
        try:
            return coordinates * duration_scales[unit]
        except KeyError as exc:
            raise ValueError(
                f"unsupported stored AWG duration unit {axis.get('unit')!r}"
            ) from exc

    if unit == "mv":
        return coordinates / full_scale_mv
    if unit == "v":
        return coordinates * 1_000.0 / full_scale_mv
    if unit in {"", "normalized", "fraction"}:
        return coordinates
    raise ValueError(
        f"unsupported stored AWG voltage unit {axis.get('unit')!r}"
    )


def awg_sweep_result_from_stored_arrays(
    arrays: Mapping[str, Any],
    *,
    database_path: Any,
    run_id: int,
) -> AwgSweepMapResult:
    """Reconstruct an AWG I/Q map from one split-array QCoDeS run."""
    metadata = arrays.get("metadata", {})
    if not isinstance(metadata, Mapping) or not _is_awg_sweep_metadata(metadata):
        raise ValueError(f"QCoDeS Run {run_id} is not an AWG 2D sweep run")
    axes_metadata = _stored_sweep_axes(metadata)
    selected_x, selected_y = _stored_selected_axes(metadata)
    sweep_coordinates = arrays.get("sweep_coordinates", {})
    if not isinstance(sweep_coordinates, Mapping):
        raise ValueError("stored AWG sweep coordinates are missing")

    iq = np.asarray(arrays["iq"])
    if iq.ndim != 4 or iq.shape[-1] != 2:
        raise ValueError(
            "stored AWG sweep IQ must have "
            "(point, repetition, sample, 2) shape"
        )
    full_scale_mv = _stored_full_scale_mv(metadata)
    coordinate_columns = []
    axis_objects = []
    for axis in axes_metadata:
        parameter_name = str(axis.get("parameter", ""))
        if not parameter_name or parameter_name not in sweep_coordinates:
            raise ValueError(
                "stored AWG coordinate parameters do not match metadata"
            )
        per_repetition = np.asarray(
            sweep_coordinates[parameter_name],
            dtype=np.float64,
        )
        if per_repetition.shape != iq.shape[:2]:
            raise ValueError(
                "stored AWG coordinate shape does not match I/Q traces"
            )
        if not np.allclose(per_repetition, per_repetition[:, :1]):
            raise ValueError(
                "stored AWG sweep coordinates change between repetitions"
            )
        native_values = _stored_axis_native_values(
            per_repetition[:, 0],
            axis,
            full_scale_mv=full_scale_mv,
        )
        coordinate_columns.append(native_values)
        axis_objects.append(
            SimpleNamespace(
                output_name=str(axis.get("output_name", "")),
                segment_name=str(axis.get("segment_name", "")),
                axis_kind=str(axis.get("axis_kind", "amplitude")),
                start=float(np.min(native_values)),
                stop=float(np.max(native_values)),
                count=int(axis.get("count", np.unique(native_values).size)),
            )
        )

    layout = metadata.get("measurement_layout", {})
    sample_rate_hz = float(
        layout.get(
            "sample_rate_hz",
            1.0e6 / float(layout.get("sample_period_us", 1.0)),
        )
    )
    ddr_result = SimpleNamespace(
        sweep_axes=tuple(axis_objects),
        sweep_points=np.column_stack(coordinate_columns),
        iq=iq,
        sample_rate_hz=sample_rate_hz,
    )
    result = reduce_awg_sweep_map(
        ddr_result,
        x_axis_key=_stored_axis_key(selected_x),
        y_axis_key=_stored_axis_key(selected_y),
        full_scale_mv=full_scale_mv,
        iq_values=iq,
        value_unit=str(arrays.get("iq_unit", "ADC units")),
        measurement_mode=str(arrays.get("measurement_mode", "raw_iq")),
    )
    path = Path(database_path).expanduser().resolve()
    return replace(
        result,
        source_label=f"QCoDeS Run {int(run_id)}",
        database_path=str(path),
        run_id=int(run_id),
    )


def load_awg_sweep_run(
    database_path: Any,
    run_id: int,
) -> AwgSweepMapResult:
    """Load and reduce one saved AWG 2D sweep QCoDeS run."""
    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"QCoDeS database does not exist: {path}")
    if isinstance(run_id, bool):
        raise TypeError("AWG sweep Run ID must be an integer")
    run_id = int(run_id)
    if run_id < 1:
        raise ValueError("AWG sweep Run ID must be at least 1")
    try:
        from qcodes import initialise_or_create_database_at, load_by_id
    except ImportError as exc:
        raise RuntimeError(
            "QCoDeS==0.58.0 is required to load AWG sweep runs"
        ) from exc
    initialise_or_create_database_at(str(path))
    dataset = load_by_id(run_id)
    arrays = load_qick_iq_arrays(dataset)
    return awg_sweep_result_from_stored_arrays(
        arrays,
        database_path=path,
        run_id=run_id,
    )


class AwgSweepMapLoadWorker(QtCore.QObject):
    """Load one saved AWG sweep without blocking the GUI thread."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, database_path: str, run_id: int, parent=None):
        super().__init__(parent)
        self._database_path = str(database_path)
        self._run_id = int(run_id)

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            result = load_awg_sweep_run(
                self._database_path,
                self._run_id,
            )
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(result)


class AwgSweepRunSelector(QtWidgets.QGroupBox):
    """Select the database and saved run displayed in the AWG map dock."""

    load_requested = QtCore.pyqtSignal(str, int)
    latest_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        parent=None,
        *,
        default_database_path: str = DEFAULT_AWG_SWEEP_DB_PATH,
    ):
        super().__init__("Saved AWG Sweep", parent)
        self._summaries: Tuple[AwgSweepRunSummary, ...] = ()
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(4)

        database_row = QtWidgets.QHBoxLayout()
        database_row.addWidget(QtWidgets.QLabel("QCoDeS DB:", self))
        self.database_path = QtWidgets.QLineEdit(
            str(default_database_path),
            self,
        )
        self.database_path.setPlaceholderText(
            "Select a QCoDeS database containing AWG sweep runs"
        )
        database_row.addWidget(self.database_path, 1)
        self.browse_button = QtWidgets.QToolButton(self)
        self.browse_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        self.browse_button.setToolTip("Choose an AWG sweep QCoDeS database")
        self.browse_button.clicked.connect(self._browse_database)
        database_row.addWidget(self.browse_button)
        self.refresh_button = QtWidgets.QPushButton("Refresh Runs", self)
        self.refresh_button.clicked.connect(self.refresh_runs)
        database_row.addWidget(self.refresh_button)
        outer.addLayout(database_row)

        selection_row = QtWidgets.QHBoxLayout()
        selection_row.addWidget(QtWidgets.QLabel("Run:", self))
        self.run_combo = QtWidgets.QComboBox(self)
        self.run_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.run_combo.setMinimumContentsLength(38)
        selection_row.addWidget(self.run_combo, 1)
        self.load_button = QtWidgets.QPushButton("Load Saved Map", self)
        self.load_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        self.load_button.clicked.connect(self._emit_load_requested)
        selection_row.addWidget(self.load_button)
        self.latest_button = QtWidgets.QPushButton(
            "Use Latest Experiment",
            self,
        )
        self.latest_button.clicked.connect(self.latest_requested.emit)
        selection_row.addWidget(self.latest_button)
        outer.addLayout(selection_row)

        self.status = QtWidgets.QLabel(
            "The map follows the latest in-memory AWG experiment.",
            self,
        )
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        outer.addWidget(self.status)

    def _browse_database(self) -> None:
        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select AWG sweep QCoDeS database",
            self.database_path.text().strip(),
            "SQLite databases (*.db *.sqlite *.sqlite3);;All files (*)",
        )
        if not selected:
            return
        self.database_path.setText(selected)
        self.refresh_runs()

    def refresh_runs(self) -> None:
        previous_run_id = self.run_combo.currentData()
        try:
            summaries = list_awg_sweep_runs(
                self.database_path.text().strip()
            )
        except Exception as exc:
            self._summaries = ()
            self.run_combo.clear()
            self.status.setText(str(exc))
            return
        self._summaries = summaries
        self.run_combo.clear()
        for summary in summaries:
            self.run_combo.addItem(summary.display_label, summary.run_id)
        if previous_run_id is not None:
            previous_index = self.run_combo.findData(previous_run_id)
            if previous_index >= 0:
                self.run_combo.setCurrentIndex(previous_index)
        if summaries:
            self.status.setText(
                f"Found {len(summaries)} compatible AWG sweep run(s)."
            )
        else:
            self.status.setText(
                "This database contains no compatible AWG 2D sweep runs."
            )

    def _emit_load_requested(self) -> None:
        run_id = self.run_combo.currentData()
        if run_id is None:
            self.status.setText("Refresh the DB and select a saved run first.")
            return
        self.load_requested.emit(
            self.database_path.text().strip(),
            int(run_id),
        )

    def set_loading(self, loading: bool, *, run_id: int = 0) -> None:
        for widget in (
            self.database_path,
            self.browse_button,
            self.refresh_button,
            self.run_combo,
            self.load_button,
            self.latest_button,
        ):
            widget.setEnabled(not loading)
        if loading:
            self.status.setText(f"Loading QCoDeS Run {run_id}...")

    def show_loaded_result(self, result: AwgSweepMapResult) -> None:
        self.set_loading(False)
        self.status.setText(
            f"Displaying {result.source_label or 'saved AWG sweep'} from "
            f"{result.database_path}."
        )

    def show_latest_result(
        self,
        result: Optional[AwgSweepMapResult],
    ) -> None:
        self.set_loading(False)
        if result is None:
            self.status.setText(
                "Following latest experiment; no AWG 2D map is available yet."
            )
        else:
            source = result.source_label or "latest in-memory experiment"
            self.status.setText(f"Displaying {source}.")


def reduce_awg_sweep_map(
    ddr_result: Any,
    *,
    x_axis_key: SweepAxisKey,
    y_axis_key: SweepAxisKey,
    full_scale_mv: float,
    iq_values: Optional[Any] = None,
    value_unit: str = "ADC units",
    measurement_mode: str = "raw_iq",
) -> AwgSweepMapResult:
    """Reduce one Cartesian FIR acquisition to a selected two-axis map.

    I and Q are averaged coherently over repetitions, FIR samples, and any
    non-selected sweep axes. Magnitude and angle are calculated only after
    this complex averaging.
    """
    full_scale_mv = float(full_scale_mv)
    if not np.isfinite(full_scale_mv) or full_scale_mv <= 0.0:
        raise ValueError("AWG full scale must be positive and finite")

    axes = tuple(ddr_result.sweep_axes)
    if len(axes) < 2:
        raise ValueError("at least two AWG sweep variables are required")
    axis_keys = tuple(sweep_axis_key(axis) for axis in axes)
    x_axis_key = tuple(map(str, x_axis_key))
    y_axis_key = tuple(map(str, y_axis_key))
    if x_axis_key == y_axis_key:
        raise ValueError("X and Y must use different AWG sweep variables")
    try:
        x_column = axis_keys.index(x_axis_key)
        y_column = axis_keys.index(y_axis_key)
    except ValueError as exc:
        raise ValueError(
            "selected AWG map axes are not present in the acquired sweep"
        ) from exc

    iq = np.asarray(ddr_result.iq if iq_values is None else iq_values)
    if iq.ndim != 4 or iq.shape[-1] != 2:
        raise ValueError(
            "AWG sweep FIR IQ must have shape "
            "(point, repetition, sample, 2)"
        )
    coordinates = np.asarray(ddr_result.sweep_points, dtype=np.float64)
    expected_coordinate_shape = (iq.shape[0], len(axes))
    if coordinates.shape != expected_coordinate_shape:
        raise ValueError(
            "AWG sweep-coordinate shape does not match the acquired FIR IQ"
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("AWG sweep coordinates must be finite")

    point_iq = iq.astype(np.float64, copy=False).mean(axis=(1, 2))
    if not np.all(np.isfinite(point_iq)):
        raise ValueError("AWG sweep I/Q contains NaN or infinity")
    display_i, display_q, display_scale = scale_iq_for_display(
        point_iq[:, 0],
        point_iq[:, 1],
        value_unit,
    )
    point_iq = np.column_stack((display_i, display_q))

    x_coordinates, x_unit = _axis_display_values(
        coordinates[:, x_column],
        axes[x_column],
        full_scale_mv,
    )
    y_coordinates, y_unit = _axis_display_values(
        coordinates[:, y_column],
        axes[y_column],
        full_scale_mv,
    )
    x_values = np.unique(x_coordinates)
    y_values = np.unique(y_coordinates)
    i_sum = np.zeros((y_values.size, x_values.size), dtype=np.float64)
    q_sum = np.zeros_like(i_sum)
    counts = np.zeros_like(i_sum, dtype=np.int64)

    for point_index, (x_value, y_value) in enumerate(
        zip(x_coordinates, y_coordinates)
    ):
        x_index = int(np.argmin(np.abs(x_values - x_value)))
        y_index = int(np.argmin(np.abs(y_values - y_value)))
        i_sum[y_index, x_index] += point_iq[point_index, 0]
        q_sum[y_index, x_index] += point_iq[point_index, 1]
        counts[y_index, x_index] += 1

    if np.any(counts == 0):
        raise ValueError("AWG result does not cover the selected X/Y grid")
    if not np.all(counts == counts.flat[0]):
        raise ValueError(
            "AWG result has an uneven number of points per selected X/Y cell"
        )
    i_mean = i_sum / counts
    q_mean = q_sum / counts
    magnitude = np.hypot(i_mean, q_mean)
    angle_deg = np.degrees(np.arctan2(q_mean, i_mean))
    averaged_axis_labels = tuple(
        sweep_axis_label(axis)
        for index, axis in enumerate(axes)
        if index not in (x_column, y_column)
    )

    return AwgSweepMapResult(
        x_values=x_values,
        y_values=y_values,
        i_mean=i_mean,
        q_mean=q_mean,
        magnitude=magnitude,
        angle_deg=angle_deg,
        x_axis_key=x_axis_key,
        y_axis_key=y_axis_key,
        x_axis_label=sweep_axis_label(axes[x_column]),
        y_axis_label=sweep_axis_label(axes[y_column]),
        x_unit=x_unit,
        y_unit=y_unit,
        value_unit=display_scale.unit,
        base_value_unit=display_scale.base_unit,
        display_scale=display_scale.factor,
        measurement_mode=str(measurement_mode),
        repetition_count=int(iq.shape[1]),
        samples_per_trace=int(iq.shape[2]),
        averaged_axis_labels=averaged_axis_labels,
        source_points_per_cell=int(counts.flat[0]),
        sample_rate_hz=float(
            getattr(ddr_result, "sample_rate_hz", 1_000_000.0)
        ),
    )


if pg is not None:

    class AwgSweepMapPlotWidget(QtWidgets.QWidget):
        """Four synchronized image plots for I, Q, magnitude, and angle."""

        _PLOT_SPECS = (
            ("i", "I", "CET-D1"),
            ("q", "Q", "CET-D1"),
            ("magnitude", "Magnitude", "viridis"),
            ("angle", "Angle", "CET-C7"),
        )

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            selector_layout = QtWidgets.QHBoxLayout()
            selector_layout.setContentsMargins(4, 2, 4, 2)
            selector_layout.addWidget(QtWidgets.QLabel("Displayed data:", self))
            self.data_selectors = {}
            for key, title, _color_map in self._PLOT_SPECS:
                selector = QtWidgets.QCheckBox(title, self)
                selector.setChecked(True)
                selector.toggled.connect(
                    lambda checked, name=key: self._set_plot_visible(
                        name,
                        checked,
                    )
                )
                selector_layout.addWidget(selector)
                self.data_selectors[key] = selector
            selector_layout.addStretch(1)
            layout.addLayout(selector_layout)
            plot_grid = QtWidgets.QGridLayout()
            plot_grid.setContentsMargins(0, 0, 0, 0)
            plot_grid.setSpacing(4)
            layout.addLayout(plot_grid, 1)
            self.plot_grid = plot_grid

            self.plots = {}
            self.images = {}
            self.plot_cells = {}
            self.range_controls = {}
            self.color_bars = {}
            self.color_maps = {}
            self._mouse_connections = []
            for index, (key, title, color_map) in enumerate(self._PLOT_SPECS):
                cell = QtWidgets.QWidget(self)
                cell_layout = QtWidgets.QVBoxLayout(cell)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setSpacing(2)
                defaults = DEFAULT_AWG_SWEEP_COLOR_RANGES[key]
                unit = "deg" if key == "angle" else "ADC units"
                range_control = ColorRangeControl(
                    f"{title} color range",
                    auto=defaults["auto"],
                    minimum=defaults["minimum"],
                    maximum=defaults["maximum"],
                    unit=unit,
                    parent=cell,
                )
                plot = pg.PlotWidget(cell)
                image = pg.ImageItem(axisOrder="row-major")
                plot.addItem(image)
                plot.setTitle(title)
                plot.setLabel("bottom", "X sweep", units="mV")
                plot.setLabel("left", "Y sweep", units="mV")
                plot.showGrid(x=True, y=True, alpha=0.18)
                selected_map = self._color_map(color_map)
                image.setColorMap(selected_map)
                color_bar = attach_color_bar(
                    plot,
                    image,
                    selected_map,
                    unit=unit,
                    levels=(
                        defaults["minimum"],
                        defaults["maximum"],
                    ),
                    range_control=range_control,
                )
                cell_layout.addWidget(range_control)
                cell_layout.addWidget(plot, 1)
                plot_grid.addWidget(cell, index // 2, index % 2)
                self.plot_cells[key] = cell
                self.plots[key] = plot
                self.images[key] = image
                self.range_controls[key] = range_control
                self.color_bars[key] = color_bar
                self.color_maps[key] = selected_map
                range_control.levels_changed.connect(
                    lambda minimum, maximum, name=key: self._set_color_levels(
                        name,
                        minimum,
                        maximum,
                    )
                )
                slot = lambda event, source=plot: self._mouse_moved(
                    event, source
                )
                plot.scene().sigMouseMoved.connect(slot)
                self._mouse_connections.append(
                    (plot.scene().sigMouseMoved, slot)
                )

            self.hover_status = QtWidgets.QLabel(
                "Run an experiment with at least two AWG sweep variables",
                self,
            )
            self.hover_status.setWordWrap(True)
            self.hover_status.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse
            )
            layout.addWidget(self.hover_status)
            self._result: Optional[AwgSweepMapResult] = None

        def _set_plot_visible(self, name: str, checked: bool) -> None:
            if not checked and not any(
                selector.isChecked()
                for key, selector in self.data_selectors.items()
                if key != name
            ):
                with QtCore.QSignalBlocker(self.data_selectors[name]):
                    self.data_selectors[name].setChecked(True)
                checked = True
            self.plot_cells[name].setVisible(checked)
            self._reflow_visible_plots()
            if checked and self._result is not None:
                self.plots[name].enableAutoRange(x=True, y=True)

        def _reflow_visible_plots(self) -> None:
            visible = [
                key
                for key, _title, _color_map in self._PLOT_SPECS
                if self.data_selectors[key].isChecked()
            ]
            for cell in self.plot_cells.values():
                self.plot_grid.removeWidget(cell)
            columns = 2 if len(visible) > 1 else 1
            for index, key in enumerate(visible):
                self.plot_grid.addWidget(
                    self.plot_cells[key],
                    index // columns,
                    index % columns,
                )

        def visible_data(self) -> Tuple[str, ...]:
            return tuple(
                key
                for key, _title, _color_map in self._PLOT_SPECS
                if self.data_selectors[key].isChecked()
            )

        def load_visible_data(self, values: Any) -> None:
            visible = set(normalize_awg_sweep_visible_data(values))
            for key, selector in self.data_selectors.items():
                with QtCore.QSignalBlocker(selector):
                    selector.setChecked(key in visible)
                self.plot_cells[key].setVisible(key in visible)
            self._reflow_visible_plots()
            self.fit_view()

        @staticmethod
        def _color_map(name: str):
            try:
                return pg.colormap.get(name)
            except (FileNotFoundError, KeyError):
                return pg.colormap.get("viridis")

        def _set_color_levels(
            self,
            name: str,
            minimum: float,
            maximum: float,
        ) -> None:
            color_bar = self.color_bars[name]
            if color_bar is None:
                self.images[name].setLevels((minimum, maximum))
                return
            color_bar.setLevels((minimum, maximum))

        def color_range_settings(self) -> dict:
            return {
                name: control.settings_dict()
                for name, control in self.range_controls.items()
            }

        def load_color_range_settings(
            self,
            settings: Mapping[str, Any],
        ) -> None:
            normalized = normalize_awg_sweep_color_ranges(settings)
            for name, control in self.range_controls.items():
                control.load_settings(normalized[name])

        @staticmethod
        def _axis_edges(values: np.ndarray) -> Tuple[float, float]:
            if values.size == 1:
                return float(values[0] - 0.5), float(values[0] + 0.5)
            spacing = float(np.median(np.diff(values)))
            return (
                float(values[0] - spacing / 2.0),
                float(values[-1] + spacing / 2.0),
            )

        @staticmethod
        def _levels(values: np.ndarray) -> Tuple[float, float]:
            low = float(np.nanmin(values))
            high = float(np.nanmax(values))
            if np.isclose(low, high):
                delta = max(1.0, abs(low) * 0.01)
                low -= delta
                high += delta
            return low, high

        @staticmethod
        def _symmetric_levels(values: np.ndarray) -> Tuple[float, float]:
            limit = float(np.nanmax(np.abs(values)))
            if np.isclose(limit, 0.0):
                limit = 1.0
            return -limit, limit

        def set_result(self, result: AwgSweepMapResult) -> None:
            self._result = result
            for plot in self.plots.values():
                plot.setLabel(
                    "bottom",
                    result.x_axis_label,
                    units=result.x_unit,
                )
                plot.setLabel(
                    "left",
                    result.y_axis_label,
                    units=result.y_unit,
                )
            self.plots["i"].setTitle(f"I [{result.value_unit}]")
            self.plots["q"].setTitle(f"Q [{result.value_unit}]")
            self.plots["magnitude"].setTitle(
                f"Magnitude [{result.value_unit}]"
            )
            self.plots["angle"].setTitle("Angle [deg]")

            x_low, x_high = self._axis_edges(result.x_values)
            y_low, y_high = self._axis_edges(result.y_values)
            rect = QtCore.QRectF(
                x_low,
                y_low,
                x_high - x_low,
                y_high - y_low,
            )
            image_values = {
                "i": result.i_mean,
                "q": result.q_mean,
                "magnitude": result.magnitude,
                "angle": result.angle_deg,
            }
            levels = {
                "i": self._symmetric_levels(result.i_mean),
                "q": self._symmetric_levels(result.q_mean),
                "magnitude": self._levels(result.magnitude),
                "angle": (-180.0, 180.0),
            }
            for key, values in image_values.items():
                self.images[key].setImage(
                    values,
                    autoLevels=False,
                )
                unit = "deg" if key == "angle" else result.value_unit
                self.range_controls[key].set_unit(unit)
                self.range_controls[key].set_data_levels(*levels[key])
                color_bar = self.color_bars[key]
                if color_bar is not None:
                    color_bar.setLabel("right", text=unit)
                self.images[key].setRect(rect)
            self.fit_view()

            averaged = (
                "none"
                if not result.averaged_axis_labels
                else ", ".join(result.averaged_axis_labels)
            )
            self.hover_status.setText(
                f"{result.repetition_count} repetitions x "
                f"{result.samples_per_trace} FIR samples; "
                f"display {result.value_unit} "
                f"({result.display_scale:g} x {result.base_value_unit}); "
                f"other averaged sweep axes: {averaged}"
            )

        def fit_view(self) -> None:
            for plot in self.plots.values():
                plot.enableAutoRange(x=True, y=True)

        def closeEvent(self, event) -> None:
            for signal, slot in self._mouse_connections:
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
            self._mouse_connections.clear()
            super().closeEvent(event)

        def _mouse_moved(self, event, plot) -> None:
            if self._result is None:
                return
            position = event[0] if isinstance(event, tuple) else event
            if not plot.sceneBoundingRect().contains(position):
                return
            point = plot.plotItem.vb.mapSceneToView(position)
            x_index = int(
                np.argmin(np.abs(self._result.x_values - point.x()))
            )
            y_index = int(
                np.argmin(np.abs(self._result.y_values - point.y()))
            )
            self.hover_status.setText(
                f"{self._result.x_axis_label} "
                f"{self._result.x_values[x_index]:.6g} "
                f"{self._result.x_unit} | "
                f"{self._result.y_axis_label} "
                f"{self._result.y_values[y_index]:.6g} "
                f"{self._result.y_unit} | "
                f"I {self._result.i_mean[y_index, x_index]:.6g} "
                f"{self._result.value_unit} | "
                f"Q {self._result.q_mean[y_index, x_index]:.6g} "
                f"{self._result.value_unit} | "
                f"Mag {self._result.magnitude[y_index, x_index]:.6g} "
                f"{self._result.value_unit} | "
                f"Angle {self._result.angle_deg[y_index, x_index]:.6g} deg"
            )

else:

    class AwgSweepMapPlotWidget(QtWidgets.QLabel):
        """Dependency error displayed when pyqtgraph is unavailable."""

        def __init__(self, parent=None):
            super().__init__(
                "pyqtgraph is required for AWG two-dimensional sweep plots",
                parent,
            )
            self.setAlignment(QtCore.Qt.AlignCenter)
            self._color_ranges = normalize_awg_sweep_color_ranges(None)
            self._visible_data = DEFAULT_AWG_SWEEP_VISIBLE_DATA

        def set_result(self, _result: AwgSweepMapResult) -> None:
            return

        def fit_view(self) -> None:
            return

        def color_range_settings(self) -> dict:
            return {
                name: dict(values)
                for name, values in self._color_ranges.items()
            }

        def load_color_range_settings(
            self,
            settings: Mapping[str, Any],
        ) -> None:
            self._color_ranges = normalize_awg_sweep_color_ranges(settings)

        def visible_data(self) -> Tuple[str, ...]:
            return self._visible_data

        def load_visible_data(self, values: Any) -> None:
            self._visible_data = normalize_awg_sweep_visible_data(values)


__all__ = [
    "AwgSweepMapLoadWorker",
    "AwgSweepMapPlotWidget",
    "AwgSweepMapResult",
    "AwgSweepRunSelector",
    "AwgSweepRunSummary",
    "AWG_SWEEP_DATA_KEYS",
    "DEFAULT_AWG_SWEEP_DB_PATH",
    "DEFAULT_AWG_SWEEP_COLOR_RANGES",
    "DEFAULT_AWG_SWEEP_VISIBLE_DATA",
    "SweepAxisKey",
    "awg_sweep_result_from_stored_arrays",
    "list_awg_sweep_runs",
    "load_awg_sweep_run",
    "normalize_awg_sweep_color_ranges",
    "normalize_awg_sweep_visible_data",
    "reduce_awg_sweep_map",
    "sweep_axis_key",
    "sweep_axis_label",
]
