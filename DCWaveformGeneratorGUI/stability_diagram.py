"""Two-electrode QICK stability-diagram acquisition and display.

Continuous acquisition intentionally bypasses QCoDeS.  A saved single shot
uses the existing split-array FIR DDR storage path so the complete I/Q trace
remains available in addition to the displayed coherent mean.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
import traceback
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

try:
    import pyqtgraph as pg
except ImportError:
    pg = None

try:
    from .dc_waveform_core import dc_iq_to_current
    from .qick_qcodes_experiment import (
        StoredQickExperiment,
        build_awg_vertex_metadata,
        connect_qick,
        execute_qick_sequence,
        store_qick_result,
    )
    from .sparameter_gui import RfPathCorrectionWidget
except ImportError:
    from dc_waveform_core import dc_iq_to_current
    from qick_qcodes_experiment import (
        StoredQickExperiment,
        build_awg_vertex_metadata,
        connect_qick,
        execute_qick_sequence,
        store_qick_result,
    )
    from sparameter_gui import RfPathCorrectionWidget


DEFAULT_STABILITY_START_MV = -100.0
DEFAULT_STABILITY_STOP_MV = 100.0
DEFAULT_STABILITY_POINTS = 51
DEFAULT_STABILITY_REPETITIONS = 1
DEFAULT_STABILITY_DB_PATH = str(Path.home() / "qick_stability_diagrams.db")


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True)
class StabilitySweepAxis:
    """One virtual-electrode voltage axis of a stability diagram."""

    output_name: str
    segment_name: str
    start_mv: float
    stop_mv: float
    points: int

    def __post_init__(self) -> None:
        if not str(self.output_name):
            raise ValueError("stability output name must not be empty")
        if not str(self.segment_name):
            raise ValueError("stability SET segment name must not be empty")
        _finite_float(self.start_mv, "stability start voltage")
        _finite_float(self.stop_mv, "stability stop voltage")
        _integer(self.points, "stability point count", 2)

    @property
    def voltages_mv(self) -> np.ndarray:
        return np.linspace(
            float(self.start_mv),
            float(self.stop_mv),
            int(self.points),
            dtype=float,
        )


@dataclass(frozen=True)
class StabilityDiagramConfig:
    """Two-axis hardware sweep and coherent FIR reduction settings."""

    x_axis: StabilitySweepAxis
    y_axis: StabilitySweepAxis
    repetitions_per_point: int = DEFAULT_STABILITY_REPETITIONS

    def __post_init__(self) -> None:
        if self.x_axis.output_name == self.y_axis.output_name:
            raise ValueError("X and Y stability axes must use different AWG outputs")
        if (
            self.x_axis.output_name,
            self.x_axis.segment_name,
        ) == (
            self.y_axis.output_name,
            self.y_axis.segment_name,
        ):
            raise ValueError("X and Y stability axes must use different electrodes")
        _integer(
            self.repetitions_per_point,
            "stability repetitions per point",
            1,
        )

    @property
    def point_count(self) -> int:
        return int(self.x_axis.points) * int(self.y_axis.points)

    def validate_full_scale(self, full_scale_mv: float) -> None:
        full_scale_mv = _finite_float(full_scale_mv, "AWG full scale")
        if full_scale_mv <= 0.0:
            raise ValueError("AWG full scale must be positive")
        for label, axis in (("X", self.x_axis), ("Y", self.y_axis)):
            if max(abs(float(axis.start_mv)), abs(float(axis.stop_mv))) > full_scale_mv:
                raise ValueError(
                    f"{label} stability sweep exceeds +/-{full_scale_mv:g} mV "
                    "AWG full scale"
                )


@dataclass(frozen=True)
class StabilityDiagramResult:
    """One complete diagram reduced from FIR I/Q traces."""

    x_voltage_mv: np.ndarray
    y_voltage_mv: np.ndarray
    i_mean: np.ndarray
    q_mean: np.ndarray
    magnitude: np.ndarray
    phase_deg: np.ndarray
    x_axis_label: str
    y_axis_label: str
    value_unit: str
    measurement_mode: str
    iteration: int
    repetition_count: int
    samples_per_trace: int


@dataclass(frozen=True)
class StoredStabilityDiagram:
    """Displayed diagram paired with its full QCoDeS single-shot run."""

    diagram: StabilityDiagramResult
    experiment: StoredQickExperiment

    @property
    def run_id(self) -> int:
        return int(self.experiment.run_id)

    @property
    def database_path(self):
        return self.experiment.database_path


def default_stability_settings(
    output_names: Sequence[str] = ("awg_0", "awg_1"),
    segment_names: Sequence[str] = ("set_0",),
) -> dict:
    """Return settings that remain loadable even before two ports exist."""
    outputs = tuple(str(value) for value in output_names) or ("awg_0",)
    segments = tuple(str(value) for value in segment_names) or ("set_0",)
    return {
        "x_axis": {
            "output_name": outputs[0],
            "segment_name": segments[0],
            "start_mv": DEFAULT_STABILITY_START_MV,
            "stop_mv": DEFAULT_STABILITY_STOP_MV,
            "points": DEFAULT_STABILITY_POINTS,
        },
        "y_axis": {
            "output_name": outputs[1] if len(outputs) > 1 else outputs[0],
            "segment_name": segments[0],
            "start_mv": DEFAULT_STABILITY_START_MV,
            "stop_mv": DEFAULT_STABILITY_STOP_MV,
            "points": DEFAULT_STABILITY_POINTS,
        },
        "repetitions_per_point": DEFAULT_STABILITY_REPETITIONS,
        "database_path": DEFAULT_STABILITY_DB_PATH,
    }


def normalize_stability_settings(
    settings: Optional[Mapping[str, Any]],
    *,
    output_names: Sequence[str],
    segment_names: Sequence[str],
) -> dict:
    """Validate a JSON settings object without requiring runnable hardware."""
    outputs = tuple(str(value) for value in output_names)
    segments = tuple(str(value) for value in segment_names)
    defaults = default_stability_settings(outputs, segments)
    if settings is None:
        return defaults
    if not isinstance(settings, Mapping):
        raise TypeError("stability_diagram must be a JSON object")

    normalized = {}
    for key, label in (("x_axis", "X"), ("y_axis", "Y")):
        raw_axis = settings.get(key, defaults[key])
        if not isinstance(raw_axis, Mapping):
            raise TypeError(f"stability {label} axis must be a JSON object")
        output_name = str(raw_axis.get("output_name", defaults[key]["output_name"]))
        segment_name = str(
            raw_axis.get("segment_name", defaults[key]["segment_name"])
        )
        if output_name not in outputs:
            raise ValueError(
                f"stability {label} output {output_name!r} is not present"
            )
        if segment_name not in segments:
            raise ValueError(
                f"stability {label} SET segment {segment_name!r} is not present"
            )
        normalized[key] = {
            "output_name": output_name,
            "segment_name": segment_name,
            "start_mv": _finite_float(
                raw_axis.get("start_mv", defaults[key]["start_mv"]),
                f"stability {label} start voltage",
            ),
            "stop_mv": _finite_float(
                raw_axis.get("stop_mv", defaults[key]["stop_mv"]),
                f"stability {label} stop voltage",
            ),
            "points": _integer(
                raw_axis.get("points", defaults[key]["points"]),
                f"stability {label} point count",
                2,
            ),
        }
    normalized["repetitions_per_point"] = _integer(
        settings.get(
            "repetitions_per_point",
            defaults["repetitions_per_point"],
        ),
        "stability repetitions per point",
        1,
    )
    database_path = str(
        settings.get("database_path", defaults["database_path"])
    ).strip()
    if not database_path:
        raise ValueError("stability database path must not be empty")
    normalized["database_path"] = database_path
    return normalized


def reduce_fir_stability_result(
    ddr_result: Any,
    config: StabilityDiagramConfig,
    *,
    full_scale_mv: float,
    iteration: int = 1,
    readout_spec: Optional[Any] = None,
) -> StabilityDiagramResult:
    """Coherently average FIR I/Q and restore the two voltage axes.

    The arithmetic mean is taken independently on I and Q over all
    repetitions and all FIR-output samples at each Cartesian coordinate.
    Magnitude and phase are then derived from that complex mean.
    """
    config.validate_full_scale(full_scale_mv)
    iq = np.asarray(ddr_result.iq)
    if iq.ndim != 4 or iq.shape[-1] != 2:
        raise ValueError(
            "stability FIR IQ must have shape "
            "(point, repetition, sample, 2)"
        )
    if iq.shape[0] != config.point_count:
        raise ValueError(
            f"received {iq.shape[0]} Cartesian points; "
            f"expected {config.point_count}"
        )

    sweep_axes = tuple(ddr_result.sweep_axes)
    axis_keys = [
        (str(axis.output_name), str(axis.segment_name)) for axis in sweep_axes
    ]
    x_key = (config.x_axis.output_name, config.x_axis.segment_name)
    y_key = (config.y_axis.output_name, config.y_axis.segment_name)
    try:
        x_column = axis_keys.index(x_key)
        y_column = axis_keys.index(y_key)
    except ValueError as exc:
        raise ValueError(
            "FIR result sweep axes do not match the selected stability electrodes"
        ) from exc
    if x_column == y_column:
        raise ValueError("stability result requires two independent sweep axes")

    coordinates = np.asarray(ddr_result.sweep_points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape != (
        config.point_count,
        len(sweep_axes),
    ):
        raise ValueError("stability sweep-coordinate shape does not match FIR IQ")

    point_iq = iq.astype(np.float64, copy=False).mean(axis=(1, 2))
    dc_measure_mode = bool(
        getattr(readout_spec, "dc_measure_mode", False)
    )
    if dc_measure_mode:
        if getattr(readout_spec, "input_board_type", None) != "DC_In":
            raise ValueError("DC measure mode requires a DC_In readout")
        point_iq = dc_iq_to_current(
            point_iq,
            getattr(readout_spec, "dc_measure_gain_v_per_a", 1.0),
        )
        value_unit = "A"
        measurement_mode = "dc_current_iq"
    else:
        value_unit = "ADC units"
        measurement_mode = "raw_iq"
    requested_x = config.x_axis.voltages_mv
    requested_y = config.y_axis.voltages_mv
    x_voltage_mv = np.sort(requested_x)
    y_voltage_mv = np.sort(requested_y)
    i_mean = np.full((y_voltage_mv.size, x_voltage_mv.size), np.nan, dtype=float)
    q_mean = np.full_like(i_mean, np.nan)
    populated = np.zeros_like(i_mean, dtype=bool)

    x_coordinates_mv = coordinates[:, x_column] * float(full_scale_mv)
    y_coordinates_mv = coordinates[:, y_column] * float(full_scale_mv)
    x_tolerance = max(
        1.0e-8,
        abs(float(config.x_axis.stop_mv) - float(config.x_axis.start_mv))
        * 1.0e-8,
    )
    y_tolerance = max(
        1.0e-8,
        abs(float(config.y_axis.stop_mv) - float(config.y_axis.start_mv))
        * 1.0e-8,
    )
    for point_index, (x_mv, y_mv) in enumerate(
        zip(x_coordinates_mv, y_coordinates_mv)
    ):
        x_index = int(np.argmin(np.abs(x_voltage_mv - x_mv)))
        y_index = int(np.argmin(np.abs(y_voltage_mv - y_mv)))
        if abs(float(x_voltage_mv[x_index]) - float(x_mv)) > x_tolerance:
            raise ValueError(f"unexpected X sweep coordinate {x_mv:g} mV")
        if abs(float(y_voltage_mv[y_index]) - float(y_mv)) > y_tolerance:
            raise ValueError(f"unexpected Y sweep coordinate {y_mv:g} mV")
        if populated[y_index, x_index]:
            raise ValueError("duplicate Cartesian coordinate in FIR result")
        populated[y_index, x_index] = True
        i_mean[y_index, x_index] = point_iq[point_index, 0]
        q_mean[y_index, x_index] = point_iq[point_index, 1]
    if not np.all(populated):
        raise ValueError("FIR result does not cover the full stability grid")

    magnitude = np.hypot(i_mean, q_mean)
    phase_deg = np.degrees(np.arctan2(q_mean, i_mean))
    return StabilityDiagramResult(
        x_voltage_mv=x_voltage_mv,
        y_voltage_mv=y_voltage_mv,
        i_mean=i_mean,
        q_mean=q_mean,
        magnitude=magnitude,
        phase_deg=phase_deg,
        x_axis_label=(
            f"{config.x_axis.output_name} / {config.x_axis.segment_name}"
        ),
        y_axis_label=(
            f"{config.y_axis.output_name} / {config.y_axis.segment_name}"
        ),
        value_unit=value_unit,
        measurement_mode=measurement_mode,
        iteration=_integer(iteration, "stability iteration", 1),
        repetition_count=int(iq.shape[1]),
        samples_per_trace=int(iq.shape[2]),
    )


def _stored_gui_settings_with_vertices(
    gui_settings: Mapping[str, Any],
    sequence: Any,
) -> dict:
    stored = dict(gui_settings)
    if not hasattr(sequence, "waveform_vertices"):
        return stored
    qick_settings = stored.get("qick", {})
    if not isinstance(qick_settings, Mapping):
        return stored
    stored["awg_waveform_vertices"] = build_awg_vertex_metadata(
        sequence,
        fabric_mhz=float(qick_settings.get("fabric_mhz", 300.0)),
        full_scale_mv=float(qick_settings.get("full_scale_mv", 2500.0)),
    )
    return stored


class StabilityDiagramWorker(QtCore.QObject):
    """Run one saved scan or repeated non-persistent scans off the GUI thread."""

    scan_ready = QtCore.pyqtSignal(object)
    single_finished = QtCore.pyqtSignal(object)
    stopped = QtCore.pyqtSignal()
    failed = QtCore.pyqtSignal(str)
    progress_changed = QtCore.pyqtSignal(int, str)

    def __init__(
        self,
        kwargs: Mapping[str, Any],
        *,
        continuous: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._kwargs = dict(kwargs)
        self._continuous = bool(continuous)
        self._stop_event = Event()

    def request_stop(self) -> None:
        """Stop after the active full hardware scan reaches a safe boundary."""
        self._stop_event.set()

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            self._run()
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _run(self) -> None:
        kwargs = dict(self._kwargs)
        connector = kwargs.pop("connector", None)
        connection_config = kwargs.pop("connection_config")
        run_config = kwargs.pop("run_config", None)
        gui_settings = kwargs.pop("gui_settings", None)
        stability_config = kwargs.pop("stability_config")
        full_scale_mv = float(kwargs.pop("full_scale_mv"))
        sequence = kwargs["sequence"]
        readout_spec = kwargs["readout_spec"]

        self.progress_changed.emit(1, "Connecting to QICK")
        soc, soccfg = connect_qick(connection_config, connector=connector)
        if self._stop_event.is_set():
            self.stopped.emit()
            return

        iteration = 0
        while not self._stop_event.is_set():
            iteration += 1

            def scan_progress(percent: int, message: str) -> None:
                self.progress_changed.emit(
                    int(percent),
                    f"Scan {iteration}: {message}",
                )

            program, ddr_result, rf_settings = execute_qick_sequence(
                soc,
                soccfg,
                progress_callback=scan_progress,
                **kwargs,
            )
            diagram = reduce_fir_stability_result(
                ddr_result,
                stability_config,
                full_scale_mv=full_scale_mv,
                iteration=iteration,
                readout_spec=readout_spec,
            )
            self.scan_ready.emit(diagram)

            if self._continuous:
                self.progress_changed.emit(
                    100,
                    f"Scan {iteration} complete; starting next scan",
                )
                continue

            if run_config is None or gui_settings is None:
                raise RuntimeError(
                    "single-shot stability acquisition requires QCoDeS settings"
                )
            stored_settings = _stored_gui_settings_with_vertices(
                gui_settings,
                sequence,
            )
            dataset, row_count = store_qick_result(
                ddr_result,
                run_config=run_config,
                connection_config=connection_config,
                program_summary=program.summary(),
                gui_settings=stored_settings,
                rf_settings=rf_settings,
                progress_callback=self.progress_changed.emit,
            )
            experiment = StoredQickExperiment(
                run_id=int(dataset.run_id),
                guid=str(dataset.guid),
                database_path=run_config.resolved_database_path,
                row_count=int(row_count),
                dataset=dataset,
                program=program,
                ddr_result=ddr_result,
                rf_settings=rf_settings,
            )
            self.single_finished.emit(
                StoredStabilityDiagram(diagram=diagram, experiment=experiment)
            )
            return
        self.stopped.emit()


class _StabilityAxisEditor(QtWidgets.QGroupBox):
    """Compact editor for one voltage axis."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        form = QtWidgets.QFormLayout(self)
        self.output = QtWidgets.QComboBox(self)
        self.segment = QtWidgets.QComboBox(self)
        self.start_mv = self._voltage_spin(DEFAULT_STABILITY_START_MV)
        self.stop_mv = self._voltage_spin(DEFAULT_STABILITY_STOP_MV)
        self.points = QtWidgets.QSpinBox(self)
        self.points.setRange(2, 1_000_000)
        self.points.setValue(DEFAULT_STABILITY_POINTS)
        form.addRow("Virtual electrode:", self.output)
        form.addRow("SET segment:", self.segment)
        form.addRow("Start:", self.start_mv)
        form.addRow("Stop:", self.stop_mv)
        form.addRow("Points:", self.points)

    @staticmethod
    def _voltage_spin(value: float) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(-1.0e6, 1.0e6)
        widget.setDecimals(6)
        widget.setValue(float(value))
        widget.setSuffix(" mV")
        return widget

    def refresh_targets(
        self,
        outputs: Sequence[Tuple[str, int]],
        segments: Sequence[str],
        *,
        preferred_output_index: int,
    ) -> None:
        previous_output = self.output.currentData()
        previous_segment = self.segment.currentData()
        with QtCore.QSignalBlocker(self.output):
            self.output.clear()
            for output_name, gen_ch in outputs:
                self.output.addItem(f"{output_name} (gen {gen_ch})", output_name)
            output_index = self.output.findData(previous_output)
            if output_index < 0 and self.output.count():
                output_index = min(preferred_output_index, self.output.count() - 1)
            self.output.setCurrentIndex(output_index)
        with QtCore.QSignalBlocker(self.segment):
            self.segment.clear()
            for segment_name in segments:
                self.segment.addItem(segment_name, segment_name)
            segment_index = self.segment.findData(previous_segment)
            if segment_index < 0 and self.segment.count():
                segment_index = 0
            self.segment.setCurrentIndex(segment_index)

    def settings_dict(self) -> dict:
        return {
            "output_name": str(self.output.currentData() or ""),
            "segment_name": str(self.segment.currentData() or ""),
            "start_mv": self.start_mv.value(),
            "stop_mv": self.stop_mv.value(),
            "points": self.points.value(),
        }

    def load_settings(self, settings: Mapping[str, Any]) -> None:
        output_index = self.output.findData(str(settings["output_name"]))
        segment_index = self.segment.findData(str(settings["segment_name"]))
        if output_index < 0 or segment_index < 0:
            raise ValueError("saved stability electrode is not present")
        self.output.setCurrentIndex(output_index)
        self.segment.setCurrentIndex(segment_index)
        self.start_mv.setValue(float(settings["start_mv"]))
        self.stop_mv.setValue(float(settings["stop_mv"]))
        self.points.setValue(int(settings["points"]))

    def value(self) -> StabilitySweepAxis:
        return StabilitySweepAxis(
            output_name=str(self.output.currentData() or ""),
            segment_name=str(self.segment.currentData() or ""),
            start_mv=self.start_mv.value(),
            stop_mv=self.stop_mv.value(),
            points=self.points.value(),
        )


if pg is not None:

    class StabilityDiagramPlotWidget(QtWidgets.QWidget):
        """Side-by-side magnitude and wrapped-phase image plots."""

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
            self.magnitude_plot = pg.PlotWidget(splitter)
            self.phase_plot = pg.PlotWidget(splitter)
            self.magnitude_image = pg.ImageItem(axisOrder="row-major")
            self.phase_image = pg.ImageItem(axisOrder="row-major")
            self.magnitude_plot.addItem(self.magnitude_image)
            self.phase_plot.addItem(self.phase_image)
            self.magnitude_plot.setTitle("Magnitude")
            self.phase_plot.setTitle("Phase")
            for plot in (self.magnitude_plot, self.phase_plot):
                plot.setLabel("bottom", "X electrode", units="mV")
                plot.setLabel("left", "Y electrode", units="mV")
                plot.showGrid(x=True, y=True, alpha=0.18)
            self._set_colormap(self.magnitude_image, "viridis")
            self._set_colormap(self.phase_image, "CET-C7")
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)
            layout.addWidget(splitter, 1)
            self.hover_status = QtWidgets.QLabel("No stability scan acquired", self)
            self.hover_status.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse
            )
            layout.addWidget(self.hover_status)
            self._result: Optional[StabilityDiagramResult] = None
            self._magnitude_proxy = pg.SignalProxy(
                self.magnitude_plot.scene().sigMouseMoved,
                rateLimit=30,
                slot=lambda event: self._mouse_moved(event, self.magnitude_plot),
            )
            self._phase_proxy = pg.SignalProxy(
                self.phase_plot.scene().sigMouseMoved,
                rateLimit=30,
                slot=lambda event: self._mouse_moved(event, self.phase_plot),
            )

        @staticmethod
        def _set_colormap(image, name: str) -> None:
            try:
                color_map = pg.colormap.get(name)
            except (FileNotFoundError, KeyError):
                color_map = pg.colormap.get("viridis")
            image.setColorMap(color_map)

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
        def _axis_edges(values: np.ndarray) -> Tuple[float, float]:
            if values.size == 1:
                return float(values[0] - 0.5), float(values[0] + 0.5)
            step = float(np.median(np.diff(values)))
            return float(values[0] - step / 2.0), float(values[-1] + step / 2.0)

        def set_result(self, result: StabilityDiagramResult) -> None:
            self._result = result
            self.magnitude_plot.setTitle(
                f"Magnitude [{result.value_unit}]"
            )
            self.phase_plot.setTitle("Phase [deg]")
            for plot in (self.magnitude_plot, self.phase_plot):
                plot.setLabel("bottom", result.x_axis_label, units="mV")
                plot.setLabel("left", result.y_axis_label, units="mV")
            x_low, x_high = self._axis_edges(result.x_voltage_mv)
            y_low, y_high = self._axis_edges(result.y_voltage_mv)
            rect = QtCore.QRectF(
                x_low,
                y_low,
                x_high - x_low,
                y_high - y_low,
            )
            self.magnitude_image.setImage(
                result.magnitude,
                autoLevels=False,
                levels=self._levels(result.magnitude),
            )
            self.phase_image.setImage(
                result.phase_deg,
                autoLevels=False,
                levels=(-180.0, 180.0),
            )
            self.magnitude_image.setRect(rect)
            self.phase_image.setRect(rect)
            self.fit_view()
            self.hover_status.setText(
                f"Scan {result.iteration}: {result.repetition_count} repetitions, "
                f"{result.samples_per_trace} FIR samples per point"
            )

        def fit_view(self) -> None:
            for plot in (self.magnitude_plot, self.phase_plot):
                plot.enableAutoRange(x=True, y=True)

        def _mouse_moved(self, event, plot) -> None:
            if self._result is None:
                return
            position = event[0] if isinstance(event, tuple) else event
            if not plot.sceneBoundingRect().contains(position):
                return
            point = plot.plotItem.vb.mapSceneToView(position)
            x_index = int(
                np.argmin(np.abs(self._result.x_voltage_mv - point.x()))
            )
            y_index = int(
                np.argmin(np.abs(self._result.y_voltage_mv - point.y()))
            )
            self.hover_status.setText(
                f"{self._result.x_axis_label} "
                f"{self._result.x_voltage_mv[x_index]:.6g} mV | "
                f"{self._result.y_axis_label} "
                f"{self._result.y_voltage_mv[y_index]:.6g} mV | "
                f"I {self._result.i_mean[y_index, x_index]:.6g} "
                f"{self._result.value_unit} | "
                f"Q {self._result.q_mean[y_index, x_index]:.6g} "
                f"{self._result.value_unit} | "
                f"Mag {self._result.magnitude[y_index, x_index]:.6g} "
                f"{self._result.value_unit} | "
                f"Phase {self._result.phase_deg[y_index, x_index]:.6g} deg"
            )

else:

    class StabilityDiagramPlotWidget(QtWidgets.QLabel):
        """Dependency error shown only when the required plot package is absent."""

        def __init__(self, parent=None):
            super().__init__("pyqtgraph is required for stability-diagram plots", parent)
            self.setAlignment(QtCore.Qt.AlignCenter)

        def set_result(self, _result: StabilityDiagramResult) -> None:
            return

        def fit_view(self) -> None:
            return


class StabilityDiagramPanel(QtWidgets.QWidget):
    """Controls and live plots for a two-electrode hardware sweep."""

    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    single_shot_requested = QtCore.pyqtSignal()
    dc_measure_changed = QtCore.pyqtSignal(bool, float)
    path_settings_applied = QtCore.pyqtSignal(object)
    front_panel_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        self.controls_scroll = QtWidgets.QScrollArea(self)
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        controls_content = QtWidgets.QWidget(self.controls_scroll)
        controls = QtWidgets.QVBoxLayout(controls_content)
        controls.setContentsMargins(2, 2, 2, 2)
        controls.setSpacing(6)

        self.path_diagram = RfPathCorrectionWidget(controls_content, compact=True)
        self.path_diagram.settings_applied.connect(self.path_settings_applied.emit)
        self.path_diagram.front_panel_requested.connect(
            self.front_panel_requested.emit
        )
        controls.addWidget(self.path_diagram)

        self.x_axis = _StabilityAxisEditor("X Electrode", controls_content)
        self.y_axis = _StabilityAxisEditor("Y Electrode", controls_content)
        controls.addWidget(self.x_axis)
        controls.addWidget(self.y_axis)

        acquisition = QtWidgets.QGroupBox("Acquisition", controls_content)
        acquisition_form = QtWidgets.QFormLayout(acquisition)
        self.repetitions = QtWidgets.QSpinBox(acquisition)
        self.repetitions.setRange(1, 1_000_000)
        self.repetitions.setValue(DEFAULT_STABILITY_REPETITIONS)
        self.point_count = QtWidgets.QLabel("2,601", acquisition)
        self.point_count.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.dc_measure_mode = QtWidgets.QCheckBox("DC measure mode", acquisition)
        self.dc_measure_mode.setToolTip(
            "DC_In only: convert FIR I/Q to current using voltage / gain"
        )
        self.dc_measure_gain_v_per_a = QtWidgets.QDoubleSpinBox(acquisition)
        self.dc_measure_gain_v_per_a.setRange(1.0e-9, 1.0e15)
        self.dc_measure_gain_v_per_a.setDecimals(6)
        self.dc_measure_gain_v_per_a.setValue(1.0)
        self.dc_measure_gain_v_per_a.setSuffix(" V/A")
        acquisition_form.addRow("Repetitions / point:", self.repetitions)
        acquisition_form.addRow("Cartesian points:", self.point_count)
        acquisition_form.addRow(self.dc_measure_mode)
        acquisition_form.addRow(
            "DC measurement gain:",
            self.dc_measure_gain_v_per_a,
        )
        controls.addWidget(acquisition)

        database_group = QtWidgets.QGroupBox(
            "Single-Shot Database",
            controls_content,
        )
        database_form = QtWidgets.QFormLayout(database_group)
        self.database_path = QtWidgets.QLineEdit(
            DEFAULT_STABILITY_DB_PATH,
            database_group,
        )
        self.browse_database = QtWidgets.QToolButton(database_group)
        self.browse_database.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        )
        self.browse_database.setToolTip("Choose the stability-diagram QCoDeS DB")
        self.browse_database.clicked.connect(self._browse_database)
        database_row = QtWidgets.QHBoxLayout()
        database_row.addWidget(self.database_path, 1)
        database_row.addWidget(self.browse_database)
        database_form.addRow("QCoDeS DB file:", database_row)
        controls.addWidget(database_group)

        self.start_button = QtWidgets.QPushButton("Start", controls_content)
        self.start_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        )
        self.start_button.setToolTip(
            "Continuously repeat full hardware scans without writing QCoDeS"
        )
        self.stop_button = QtWidgets.QPushButton("Stop", controls_content)
        self.stop_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop)
        )
        self.stop_button.setToolTip(
            "Stop after the currently active full hardware scan completes"
        )
        self.stop_button.setEnabled(False)
        self.single_shot_button = QtWidgets.QPushButton(
            "Single Shot && Save",
            controls_content,
        )
        self.single_shot_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        )
        self.single_shot_button.setToolTip(
            "Acquire one full stability diagram and save its FIR I/Q traces "
            "to the database selected above"
        )
        self.fit_button = QtWidgets.QPushButton("Fit", controls_content)
        self.fit_button.setText("Fit")
        self.fit_button.setToolTip("Fit both stability plots to the full sweep")
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.single_shot_button)
        controls.addWidget(self.fit_button)

        self.progress = QtWidgets.QProgressBar(controls_content)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.hide()
        controls.addWidget(self.progress)
        self.status = QtWidgets.QLabel("Ready", controls_content)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        controls.addWidget(self.status)
        controls.addStretch(1)
        self.controls_scroll.setWidget(controls_content)
        outer.addWidget(self.controls_scroll, 1)

        self.plot = StabilityDiagramPlotWidget(self)
        outer.addWidget(self.plot, 1)

        self.start_button.clicked.connect(self.start_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.single_shot_button.clicked.connect(self.single_shot_requested.emit)
        self.fit_button.clicked.connect(self.plot.fit_view)
        self.x_axis.points.valueChanged.connect(self._update_point_count)
        self.y_axis.points.valueChanged.connect(self._update_point_count)
        self.dc_measure_mode.toggled.connect(self._emit_dc_measure_changed)
        self.dc_measure_gain_v_per_a.valueChanged.connect(
            self._emit_dc_measure_changed
        )
        self._targets_available = False
        self._dc_input_available = False
        self._running = False
        self._update_point_count()
        self._update_dc_measure_controls()

    def _update_dc_measure_controls(self) -> None:
        editable = self._dc_input_available and not self._running
        self.dc_measure_mode.setEnabled(editable)
        self.dc_measure_gain_v_per_a.setEnabled(
            editable and self.dc_measure_mode.isChecked()
        )

    def _emit_dc_measure_changed(self, *_args) -> None:
        self._update_dc_measure_controls()
        self.dc_measure_changed.emit(
            self.dc_measure_mode.isChecked(),
            self.dc_measure_gain_v_per_a.value(),
        )

    def set_dc_measure_context(
        self,
        input_board_type: str,
        enabled: bool,
        gain_v_per_a: float,
    ) -> None:
        """Mirror the shared FIR readout measurement settings."""
        self._dc_input_available = str(input_board_type) == "DC_In"
        with QtCore.QSignalBlocker(self.dc_measure_mode):
            self.dc_measure_mode.setChecked(
                bool(enabled) if self._dc_input_available else False
            )
        with QtCore.QSignalBlocker(self.dc_measure_gain_v_per_a):
            self.dc_measure_gain_v_per_a.setValue(float(gain_v_per_a))
        self._update_dc_measure_controls()

    def refresh_targets(
        self,
        output_names: Sequence[str],
        awg_channels: Sequence[int],
        segment_names: Sequence[str],
    ) -> None:
        outputs = tuple(zip(output_names, awg_channels))
        self.x_axis.refresh_targets(outputs, segment_names, preferred_output_index=0)
        self.y_axis.refresh_targets(outputs, segment_names, preferred_output_index=1)
        if (
            len(outputs) >= 2
            and self.x_axis.output.currentData() == self.y_axis.output.currentData()
        ):
            for index in range(self.y_axis.output.count()):
                if (
                    self.y_axis.output.itemData(index)
                    != self.x_axis.output.currentData()
                ):
                    self.y_axis.output.setCurrentIndex(index)
                    break
        self._targets_available = len(outputs) >= 2 and bool(segment_names)
        if not self._targets_available:
            self.status.setText("Add at least two AWG outputs to run a stability scan")
        self._set_idle_button_state()

    def _browse_database(self) -> None:
        path, _selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Choose stability-diagram database",
            self.database_path.text().strip() or DEFAULT_STABILITY_DB_PATH,
            "QCoDeS SQLite database (*.db)",
        )
        if path:
            selected = Path(path)
            if selected.suffix.lower() != ".db":
                selected = selected.with_suffix(".db")
            self.database_path.setText(str(selected))

    def database_path_value(self) -> str:
        value = self.database_path.text().strip()
        if not value:
            raise ValueError("stability database path must not be empty")
        path = Path(value).expanduser()
        if path.suffix.lower() != ".db":
            path = path.with_suffix(".db")
        return str(path)

    def set_front_panel_configuration(self, configuration) -> None:
        self.path_diagram.set_front_panel_configuration(configuration)

    def apply_path_settings(self, values: Mapping[str, Any]) -> None:
        self.path_diagram.apply_external_settings(values)

    def config(self, *, full_scale_mv: float) -> StabilityDiagramConfig:
        if not self._targets_available:
            raise ValueError("stability diagram requires at least two AWG outputs")
        config = StabilityDiagramConfig(
            x_axis=self.x_axis.value(),
            y_axis=self.y_axis.value(),
            repetitions_per_point=self.repetitions.value(),
        )
        config.validate_full_scale(full_scale_mv)
        return config

    def settings_dict(self) -> dict:
        return {
            "x_axis": self.x_axis.settings_dict(),
            "y_axis": self.y_axis.settings_dict(),
            "repetitions_per_point": self.repetitions.value(),
            "database_path": self.database_path_value(),
        }

    def load_settings(self, settings: Mapping[str, Any]) -> None:
        self.x_axis.load_settings(settings["x_axis"])
        self.y_axis.load_settings(settings["y_axis"])
        self.repetitions.setValue(int(settings["repetitions_per_point"]))
        self.database_path.setText(
            str(settings.get("database_path", DEFAULT_STABILITY_DB_PATH))
        )
        self._update_point_count()

    def set_running(self, running: bool, message: str) -> None:
        self._running = bool(running)
        self.start_button.setEnabled(not running and self._targets_available)
        self.single_shot_button.setEnabled(not running and self._targets_available)
        self.stop_button.setEnabled(running)
        for editor in (self.x_axis, self.y_axis):
            editor.setEnabled(not running)
        self.repetitions.setEnabled(not running)
        self.path_diagram.setEnabled(not running)
        self.database_path.setEnabled(not running)
        self.browse_database.setEnabled(not running)
        self._update_dc_measure_controls()
        self.progress.setVisible(running)
        if not running:
            self.progress.setValue(0)
        self.status.setText(message)

    def set_stopping(self) -> None:
        self.stop_button.setEnabled(False)
        self.status.setText("Stopping after the active full scan completes...")

    def update_progress(self, percent: int, message: str) -> None:
        self.progress.setValue(max(0, min(100, int(percent))))
        self.status.setText(f"{int(percent)}% - {message}")

    def show_result(self, result: StabilityDiagramResult) -> None:
        self.plot.set_result(result)
        self.status.setText(
            f"Scan {result.iteration} complete: "
            f"{result.magnitude.shape[1]} x {result.magnitude.shape[0]} points "
            f"({result.value_unit})"
        )

    def show_saved_result(self, stored: StoredStabilityDiagram) -> None:
        self.plot.set_result(stored.diagram)
        self.set_running(
            False,
            f"QCoDeS Run {stored.run_id} saved to {stored.database_path}",
        )

    def detach_plot(self) -> StabilityDiagramPlotWidget:
        """Remove the plot from this panel for use in a main-window dock."""
        self.layout().removeWidget(self.plot)
        self.plot.setParent(None)
        return self.plot

    def _set_idle_button_state(self) -> None:
        if self.stop_button.isEnabled():
            return
        self.start_button.setEnabled(self._targets_available)
        self.single_shot_button.setEnabled(self._targets_available)

    def _update_point_count(self, *_args) -> None:
        self.point_count.setText(
            f"{self.x_axis.points.value() * self.y_axis.points.value():,}"
        )


__all__ = [
    "DEFAULT_STABILITY_POINTS",
    "DEFAULT_STABILITY_REPETITIONS",
    "DEFAULT_STABILITY_START_MV",
    "DEFAULT_STABILITY_STOP_MV",
    "StabilityDiagramConfig",
    "StabilityDiagramPanel",
    "StabilityDiagramPlotWidget",
    "StabilityDiagramResult",
    "StabilityDiagramWorker",
    "StabilitySweepAxis",
    "StoredStabilityDiagram",
    "default_stability_settings",
    "normalize_stability_settings",
    "reduce_fir_stability_result",
]
