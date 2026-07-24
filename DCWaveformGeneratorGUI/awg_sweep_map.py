"""Two-dimensional maps reduced from Cartesian AWG tuning sweeps.

The hardware acquisition stores one I/Q trace for every Cartesian sweep point
and repetition. This module coherently averages I and Q over repetitions and
FIR samples, then arranges two user-selected sweep variables on X and Y.
Additional sweep variables are averaged at each selected X/Y coordinate.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

try:
    import pyqtgraph as pg
except ImportError:
    pg = None


SweepAxisKey = Tuple[str, str]


def sweep_axis_key(axis: Any) -> SweepAxisKey:
    """Return the stable ``(output_name, segment_name)`` key for one axis."""
    return str(axis.output_name), str(axis.segment_name)


def sweep_axis_label(axis: Any) -> str:
    """Return the compact user-facing name for one sweep variable."""
    output_name, segment_name = sweep_axis_key(axis)
    if getattr(axis, "axis_kind", "amplitude") == "rf_duration":
        return f"{output_name} / {segment_name} RF duration"
    return f"{output_name} / {segment_name}"


def _axis_display_values(
    coordinates: np.ndarray,
    axis: Any,
    full_scale_mv: float,
) -> Tuple[np.ndarray, str]:
    if getattr(axis, "axis_kind", "amplitude") == "rf_duration":
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
    measurement_mode: str
    repetition_count: int
    samples_per_trace: int
    averaged_axis_labels: Tuple[str, ...]
    source_points_per_cell: int
    sample_rate_hz: float

    @property
    def x_values_mv(self) -> np.ndarray:
        """Backward-compatible alias for pre-duration-sweep callers."""
        return self.x_values

    @property
    def y_values_mv(self) -> np.ndarray:
        """Backward-compatible alias for pre-duration-sweep callers."""
        return self.y_values


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
        value_unit=str(value_unit),
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
            plot_grid = QtWidgets.QGridLayout()
            plot_grid.setContentsMargins(0, 0, 0, 0)
            plot_grid.setSpacing(4)
            layout.addLayout(plot_grid, 1)

            self.plots = {}
            self.images = {}
            self._mouse_connections = []
            for index, (key, title, color_map) in enumerate(self._PLOT_SPECS):
                plot = pg.PlotWidget(self)
                image = pg.ImageItem(axisOrder="row-major")
                plot.addItem(image)
                plot.setTitle(title)
                plot.setLabel("bottom", "X sweep", units="mV")
                plot.setLabel("left", "Y sweep", units="mV")
                plot.showGrid(x=True, y=True, alpha=0.18)
                self._set_colormap(image, color_map)
                plot_grid.addWidget(plot, index // 2, index % 2)
                self.plots[key] = plot
                self.images[key] = image
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

        @staticmethod
        def _set_colormap(image, name: str) -> None:
            try:
                color_map = pg.colormap.get(name)
            except (FileNotFoundError, KeyError):
                color_map = pg.colormap.get("viridis")
            image.setColorMap(color_map)

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
                    levels=levels[key],
                )
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

        def set_result(self, _result: AwgSweepMapResult) -> None:
            return

        def fit_view(self) -> None:
            return


__all__ = [
    "AwgSweepMapPlotWidget",
    "AwgSweepMapResult",
    "SweepAxisKey",
    "reduce_awg_sweep_map",
    "sweep_axis_key",
    "sweep_axis_label",
]
