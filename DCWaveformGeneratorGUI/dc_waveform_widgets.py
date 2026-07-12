"""Fast PyQtGraph widgets for the DC waveform editor.

The waveform plot updates existing graphics items in place.  It does not clear
and rebuild axes while dragging, and the X/Y trace widget reuses one curve.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from math import hypot
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from .dc_waveform_core import PulseSequence
except ImportError:
    from dc_waveform_core import PulseSequence


pg.setConfigOptions(antialias=False)


def _plot_color(index: int) -> QtGui.QColor:
    return QtGui.QColor(pg.intColor(index, hues=8, values=1, minValue=110, maxValue=220))


class TracePlotWidget(pg.PlotWidget):
    """Voltage-voltage trace for two selected waveform outputs."""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setBackground("w")
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setLabel("bottom", "Pulse-X", units="mV")
        self.setLabel("left", "Pulse-Y", units="mV")
        self.getPlotItem().hideButtons()
        self._curve = self.plot(
            [],
            [],
            pen=pg.mkPen(_plot_color(0), width=1.8),
            symbol="o",
            symbolSize=6,
            symbolPen=pg.mkPen(_plot_color(0)),
            symbolBrush=pg.mkBrush(_plot_color(0)),
        )
        self.x_idx: Optional[int] = None
        self.y_idx: Optional[int] = None
        self._pulses: Sequence[PulseSequence] = ()

    @property
    def has_selection(self) -> bool:
        return (
            self.x_idx is not None
            and self.y_idx is not None
            and self.x_idx < len(self._pulses)
            and self.y_idx < len(self._pulses)
        )

    def uses_port(self, index: int) -> bool:
        return index == self.x_idx or index == self.y_idx

    def refresh_trace(self, pulses: Sequence[PulseSequence]) -> None:
        """Update the existing curve; interpolation runs only when X/Y exist."""
        self._pulses = pulses
        if not self.has_selection:
            self._curve.setData([], [])
            return
        pulse_x = pulses[self.x_idx]
        pulse_y = pulses[self.y_idx]
        time_union = np.union1d(pulse_x.t, pulse_y.t)
        voltage_x = np.interp(time_union, pulse_x.t, pulse_x.v)
        voltage_y = np.interp(time_union, pulse_y.t, pulse_y.v)
        self._curve.setData(voltage_x, voltage_y)
        self.setLabel("bottom", f"Pulse {self.x_idx + 1}", units="mV")
        self.setLabel("left", f"Pulse {self.y_idx + 1}", units="mV")

    def fit_view(self) -> None:
        if self.has_selection:
            self.getPlotItem().autoRange(padding=0.08)


class RfPulsePreviewWidget(pg.PlotWidget):
    """Preview a DDS RF pulse without rendering every RF sample.

    Short pulses use a bounded number of representative carrier points.  Long
    or high-frequency pulses use a six-point amplitude envelope, which keeps
    interactive edits independent of the underlying RF sample count.
    """

    MAX_CARRIER_CYCLES = 32.0
    MAX_CARRIER_POINTS = 768

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setBackground("w")
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setLabel("bottom", "sequence time", units="us")
        self.setLabel("left", "DAC code (pre-attenuation)")
        self.getPlotItem().hideButtons()
        self.getPlotItem().setMenuEnabled(False)
        color = _plot_color(1)
        self._carrier_curve = self.plot([], [], pen=pg.mkPen(color, width=1.4))
        # Keep the previous attribute as an alias for callers which inspect the
        # preview curve in tests or notebooks.
        self._curve = self._carrier_curve
        self._upper_curve = self.plot([], [], pen=pg.mkPen(color, width=1.0))
        self._lower_curve = self.plot([], [], pen=pg.mkPen(color, width=1.0))
        self._envelope_fill = pg.FillBetweenItem(
            self._upper_curve,
            self._lower_curve,
            brush=pg.mkBrush(color.red(), color.green(), color.blue(), 42),
        )
        self.addItem(self._envelope_fill)
        self._start_line = pg.InfiniteLine(angle=90, pen=pg.mkPen(color, style=QtCore.Qt.DashLine))
        self._end_line = pg.InfiniteLine(angle=90, pen=pg.mkPen(color, style=QtCore.Qt.DashLine))
        self.addItem(self._start_line)
        self.addItem(self._end_line)
        self._start_line.hide()
        self._end_line.hide()
        self.preview_mode = "empty"

    def clear_preview(self) -> None:
        self._carrier_curve.setData([], [])
        self._upper_curve.setData([], [])
        self._lower_curve.setData([], [])
        self._start_line.hide()
        self._end_line.hide()
        self.setTitle("")
        self.preview_mode = "empty"

    def set_pulse(
        self,
        *,
        start_us: float,
        duration_us: float,
        frequency_mhz: float,
        gain: int,
        phase_degrees: float,
        att1_db: float,
        att2_db: float,
    ) -> None:
        end_us = start_us + duration_us
        padding = max(0.02, duration_us * 0.08)
        cycle_count = abs(frequency_mhz) * duration_us
        if cycle_count <= self.MAX_CARRIER_CYCLES:
            points = min(
                self.MAX_CARRIER_POINTS,
                max(64, int(cycle_count * 16) + 2),
            )
            pulse_time = np.linspace(start_us, end_us, points, dtype=float)
            phase = np.deg2rad(phase_degrees)
            pulse_values = float(gain) * np.cos(
                2.0 * np.pi * frequency_mhz * (pulse_time - start_us) + phase
            )
            time_us = np.concatenate(
                ([start_us - padding, start_us], pulse_time, [end_us, end_us + padding])
            )
            values = np.concatenate(([0.0, 0.0], pulse_values, [0.0, 0.0]))
            self._carrier_curve.setData(time_us, values)
            self._upper_curve.setData([], [])
            self._lower_curve.setData([], [])
            self.preview_mode = "carrier"
            mode_text = f"representative carrier ({points} points)"
        else:
            amplitude = abs(float(gain))
            envelope_time = np.asarray(
                [
                    start_us - padding,
                    start_us,
                    start_us,
                    end_us,
                    end_us,
                    end_us + padding,
                ],
                dtype=float,
            )
            upper = np.asarray([0.0, 0.0, amplitude, amplitude, 0.0, 0.0])
            lower = -upper
            self._carrier_curve.setData([], [])
            self._upper_curve.setData(envelope_time, upper)
            self._lower_curve.setData(envelope_time, lower)
            self.preview_mode = "envelope"
            mode_text = "amplitude envelope"
        self._start_line.setPos(start_us)
        self._end_line.setPos(end_us)
        self._start_line.show()
        self._end_line.show()
        self.setTitle(
            f"{frequency_mhz:.6g} MHz, gain {gain}, "
            f"ATT1 {att1_db:.2f} dB, ATT2 {att2_db:.2f} dB - {mode_text}"
        )
        y_limit = max(1.0, abs(float(gain)) * 1.08)
        self.setXRange(start_us - padding, end_us + padding, padding=0.0)
        self.setYRange(-y_limit, y_limit, padding=0.0)


class RfPulseTimelineWidget(pg.PlotWidget):
    """RF output lane sharing the editable DC waveform's nanosecond time axis."""

    MAX_CARRIER_CYCLES = RfPulsePreviewWidget.MAX_CARRIER_CYCLES
    MAX_CARRIER_POINTS = RfPulsePreviewWidget.MAX_CARRIER_POINTS

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setBackground("w")
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setLabel("bottom", "sequence time", units="ns")
        self.setLabel("left", "RF DAC code")
        self.getPlotItem().hideButtons()
        self.getPlotItem().setMenuEnabled(False)
        self.setMouseEnabled(x=True, y=False)
        self.setMinimumHeight(120)
        self.setMaximumHeight(260)

        color = _plot_color(6)
        self._carrier_curve = self.plot([], [], pen=pg.mkPen(color, width=1.2))
        self._upper_curve = self.plot([], [], pen=pg.mkPen(color, width=1.0))
        self._lower_curve = self.plot([], [], pen=pg.mkPen(color, width=1.0))
        self._envelope_fill = pg.FillBetweenItem(
            self._upper_curve,
            self._lower_curve,
            brush=pg.mkBrush(color.red(), color.green(), color.blue(), 34),
        )
        self.addItem(self._envelope_fill)
        self._start_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(color, style=QtCore.Qt.DashLine),
        )
        self._end_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(color, style=QtCore.Qt.DashLine),
        )
        self.addItem(self._start_line)
        self.addItem(self._end_line)
        self.preview_mode = "empty"
        self.start_ns = 0.0
        self.end_ns = 0.0
        self.clear_pulse()

    def clear_pulse(self) -> None:
        self._carrier_curve.setData([], [])
        self._upper_curve.setData([], [])
        self._lower_curve.setData([], [])
        self._start_line.hide()
        self._end_line.hide()
        self.setTitle("")
        self.preview_mode = "empty"

    def set_pulse(
        self,
        *,
        gen_ch: int,
        start_ns: float,
        duration_ns: float,
        frequency_mhz: float,
        gain: int,
        phase_degrees: float,
        att1_db: float,
        att2_db: float,
    ) -> None:
        self.start_ns = float(start_ns)
        self.end_ns = float(start_ns + duration_ns)
        cycle_count = abs(frequency_mhz) * duration_ns / 1000.0
        if cycle_count <= self.MAX_CARRIER_CYCLES:
            points = min(
                self.MAX_CARRIER_POINTS,
                max(64, int(cycle_count * 16) + 2),
            )
            pulse_time = np.linspace(self.start_ns, self.end_ns, points, dtype=float)
            phase = np.deg2rad(phase_degrees)
            pulse_values = float(gain) * np.cos(
                2.0
                * np.pi
                * frequency_mhz
                * (pulse_time - self.start_ns)
                / 1000.0
                + phase
            )
            time_ns = np.concatenate(
                ([self.start_ns, self.start_ns], pulse_time, [self.end_ns, self.end_ns])
            )
            values = np.concatenate(([0.0, 0.0], pulse_values, [0.0, 0.0]))
            self._carrier_curve.setData(time_ns, values)
            self._upper_curve.setData([], [])
            self._lower_curve.setData([], [])
            self.preview_mode = "carrier"
            mode_text = f"representative carrier ({points} points)"
        else:
            amplitude = abs(float(gain))
            envelope_time = np.asarray(
                [
                    self.start_ns,
                    self.start_ns,
                    self.start_ns,
                    self.end_ns,
                    self.end_ns,
                    self.end_ns,
                ],
                dtype=float,
            )
            upper = np.asarray([0.0, 0.0, amplitude, amplitude, 0.0, 0.0])
            self._carrier_curve.setData([], [])
            self._upper_curve.setData(envelope_time, upper)
            self._lower_curve.setData(envelope_time, -upper)
            self.preview_mode = "envelope"
            mode_text = "amplitude envelope"

        self._start_line.setPos(self.start_ns)
        self._end_line.setPos(self.end_ns)
        self._start_line.show()
        self._end_line.show()
        y_limit = max(1.0, abs(float(gain)) * 1.08)
        self.setYRange(-y_limit, y_limit, padding=0.0)
        self.setTitle(
            f"RF gen {gen_ch}: {frequency_mhz:.6g} MHz, gain {gain}, "
            f"ATT1/ATT2 {att1_db:.2f}/{att2_db:.2f} dB - {mode_text}"
        )


class WaveformPlotWidget(pg.PlotWidget):
    """Interactive multi-output waveform editor backed by PyQtGraph."""

    flat_moved = QtCore.pyqtSignal(int, int, float)
    point_moved = QtCore.pyqtSignal(int, int, float)
    MAX_VISIBLE_GRID_LINES = 80

    def __init__(self, pulse: PulseSequence, parent=None):
        super().__init__(parent=parent)
        self.setBackground("w")
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setLabel("bottom", "time", units="ns")
        self.setLabel("left", "voltage", units="mV")
        self.getPlotItem().hideButtons()
        self.getPlotItem().setMenuEnabled(False)
        self.setMouseEnabled(x=True, y=True)

        self._pulses: List[PulseSequence] = [pulse]
        # Preserve the old attribute names used by MainWindow while switching
        # their contents from Matplotlib lines to PlotDataItem objects.
        self._pulse = self._pulses
        self._line: List[pg.PlotDataItem] = []
        self._physical_line: List[pg.PlotDataItem] = []
        self._orig_colors: List[QtGui.QColor] = []
        self._selected_port_idx = 0
        self._default_width = 1.5
        self._highlight_width = 2.8
        self._physical_width = 1.6
        self._voltage_view = "both"
        self._physical_time_ns = np.asarray([], dtype=float)
        self._physical_values_mv = np.empty((0, 0), dtype=float)
        self._drag_flat: Optional[Tuple[int, int]] = None
        self._drag_point: Optional[Tuple[int, int]] = None
        self._grid_time_ns = 10.0
        self._grid_voltage_mv = 10.0
        self._grid_snap_enabled = False
        self._grid_visible = True
        self._sweep_port_index: Optional[int] = None
        self._sweep_time_ns = np.asarray([], dtype=float)
        self._sweep_lower_mv = np.asarray([], dtype=float)
        self._sweep_upper_mv = np.asarray([], dtype=float)

        sweep_color = _plot_color(0)
        self._sweep_lower_curve = self.plot(
            [],
            [],
            pen=pg.mkPen(sweep_color, width=1.0, style=QtCore.Qt.DotLine),
        )
        self._sweep_upper_curve = self.plot(
            [],
            [],
            pen=pg.mkPen(sweep_color, width=1.0, style=QtCore.Qt.DotLine),
        )
        self._sweep_fill = pg.FillBetweenItem(
            self._sweep_upper_curve,
            self._sweep_lower_curve,
            brush=pg.mkBrush(
                sweep_color.red(),
                sweep_color.green(),
                sweep_color.blue(),
                24,
            ),
        )
        self.addItem(self._sweep_fill)
        self._sweep_fill.setZValue(0.1)
        self._sweep_lower_curve.setZValue(0.8)
        self._sweep_upper_curve.setZValue(0.8)
        self._sweep_graphics = {}

        self._annotation = pg.TextItem(
            text="",
            color=QtGui.QColor("black"),
            fill=pg.mkBrush(255, 255, 255, 225),
            border=pg.mkPen(120, 120, 120),
            anchor=(0.0, 1.0),
        )
        self.addItem(self._annotation, ignoreBounds=True)
        self._annotation.hide()

        self._append_curve(pulse)
        self.set_physical_waveforms(pulse.t, np.asarray([pulse.v]))
        # QAbstractScrollArea delivers pointer events through its viewport.
        # Filtering it makes drag editing deterministic across Qt/PyQtGraph
        # versions while unhandled events still reach the normal pan/zoom path.
        self.viewport().installEventFilter(self)
        self.getPlotItem().vb.sigRangeChanged.connect(self._refresh_grid_tick_spacing)
        self.set_grid(
            time_step_ns=self._grid_time_ns,
            voltage_step_mv=self._grid_voltage_mv,
            snap_enabled=self._grid_snap_enabled,
            visible=self._grid_visible,
        )
        self.fit_view()

    @staticmethod
    def _nearest_grid_value(value: float, step: float) -> float:
        """Round to a zero-anchored grid, with half steps away from zero."""
        scaled = value / step
        if scaled >= 0.0:
            return float(np.floor(scaled + 0.5) * step)
        return float(np.ceil(scaled - 0.5) * step)

    def set_grid(
        self,
        *,
        time_step_ns: float,
        voltage_step_mv: float,
        snap_enabled: bool,
        visible: bool,
    ) -> None:
        """Set fixed plot spacing and drag snapping for the editable waveform."""
        if not np.isfinite(time_step_ns) or time_step_ns <= 0.0:
            raise ValueError("time grid spacing must be a positive finite value")
        if not np.isfinite(voltage_step_mv) or voltage_step_mv <= 0.0:
            raise ValueError("voltage grid spacing must be a positive finite value")
        self._grid_time_ns = float(time_step_ns)
        self._grid_voltage_mv = float(voltage_step_mv)
        self._grid_snap_enabled = bool(snap_enabled)
        self._grid_visible = bool(visible)
        self._refresh_grid_tick_spacing()
        self.showGrid(x=self._grid_visible, y=self._grid_visible, alpha=0.25)

    @classmethod
    def _display_grid_step(cls, base_step: float, visible_span: float) -> float:
        """Return a grid-aligned display step capped to a practical line count."""
        required_multiple = max(
            1,
            int(np.ceil(visible_span / (base_step * cls.MAX_VISIBLE_GRID_LINES))),
        )
        if required_multiple <= 1:
            nice_multiple = 1
        else:
            magnitude = 10 ** int(np.floor(np.log10(required_multiple)))
            normalized = required_multiple / magnitude
            if normalized <= 2:
                nice_multiple = 2 * magnitude
            elif normalized <= 5:
                nice_multiple = 5 * magnitude
            else:
                nice_multiple = 10 * magnitude
        return float(base_step * nice_multiple)

    def _refresh_grid_tick_spacing(self, *_args) -> None:
        view_range = self.getPlotItem().vb.viewRange()
        x_span = max(0.0, float(view_range[0][1] - view_range[0][0]))
        y_span = max(0.0, float(view_range[1][1] - view_range[1][0]))
        time_display_step = self._display_grid_step(self._grid_time_ns, x_span)
        voltage_display_step = self._display_grid_step(self._grid_voltage_mv, y_span)
        self._display_time_grid_ns = time_display_step
        self._display_voltage_grid_mv = voltage_display_step
        self.getPlotItem().getAxis("bottom").setTickSpacing(
            levels=[(time_display_step, 0.0)],
        )
        self.getPlotItem().getAxis("left").setTickSpacing(
            levels=[(voltage_display_step, 0.0)],
        )

    @property
    def grid_settings(self) -> Tuple[float, float, bool, bool]:
        return (
            self._grid_time_ns,
            self._grid_voltage_mv,
            self._grid_snap_enabled,
            self._grid_visible,
        )

    def _snap_voltage(self, value: float, pulse: PulseSequence) -> float:
        value = float(np.clip(value, pulse.v_bounds[0], pulse.v_bounds[1]))
        if not self._grid_snap_enabled:
            return value
        lower = np.ceil(pulse.v_bounds[0] / self._grid_voltage_mv) * self._grid_voltage_mv
        upper = np.floor(pulse.v_bounds[1] / self._grid_voltage_mv) * self._grid_voltage_mv
        return float(
            np.clip(
                self._nearest_grid_value(value, self._grid_voltage_mv),
                lower,
                upper,
            )
        )

    def _snap_time(self, value: float, pulse: PulseSequence, point_index: int) -> float:
        if not self._grid_snap_enabled or point_index == 0:
            return value
        snapped = self._nearest_grid_value(value, self._grid_time_ns)
        minimum = np.nextafter(float(pulse.t[point_index - 1]), np.inf)
        if snapped < minimum:
            snapped = float(np.ceil(minimum / self._grid_time_ns) * self._grid_time_ns)
        return snapped

    def _append_curve(self, pulse: PulseSequence) -> None:
        index = len(self._line)
        color = _plot_color(index)
        physical_curve = self.plot(
            pulse.t,
            pulse.v,
            pen=pg.mkPen(
                color,
                width=self._physical_width,
                style=QtCore.Qt.DashLine,
            ),
        )
        physical_curve.setZValue(index + 0.5)
        curve = self.plot(
            pulse.t,
            pulse.v,
            pen=pg.mkPen(color, width=self._default_width),
            symbol="o",
            symbolSize=7,
            symbolPen=pg.mkPen(color),
            symbolBrush=pg.mkBrush(color),
        )
        curve.setZValue(index + 1)
        curve.setVisible(self._voltage_view in {"both", "virtual"})
        physical_curve.setVisible(self._voltage_view in {"both", "physical"})
        self._physical_line.append(physical_curve)
        self._line.append(curve)
        self._orig_colors.append(color)

    def line_color(self, index: int) -> QtGui.QColor:
        return QtGui.QColor(self._orig_colors[index])

    @property
    def voltage_view(self) -> str:
        return self._voltage_view

    def set_physical_waveforms(
        self,
        time_ns: Sequence[float],
        waveforms_mv,
    ) -> None:
        """Update dashed physical-AWG traces on a common time grid."""
        time_values = np.asarray(time_ns, dtype=float)
        waveform_values = np.asarray(waveforms_mv, dtype=float)
        expected_shape = (len(self._pulses), time_values.size)
        if time_values.ndim != 1 or waveform_values.shape != expected_shape:
            raise ValueError(
                "physical waveforms must have shape "
                f"{expected_shape}, received {waveform_values.shape}"
            )
        self._physical_time_ns = time_values.copy()
        self._physical_values_mv = waveform_values.copy()
        for index, curve in enumerate(self._physical_line):
            curve.setData(time_values, waveform_values[index])

    def set_voltage_view(self, mode: str) -> None:
        """Select virtual, physical, or simultaneous voltage rendering."""
        if mode not in {"both", "virtual", "physical"}:
            raise ValueError("voltage view must be 'both', 'virtual', or 'physical'")
        self._voltage_view = mode
        self._drag_flat = None
        self._drag_point = None
        for curve in self._line:
            curve.setVisible(mode in {"both", "virtual"})
        for curve in self._physical_line:
            curve.setVisible(mode in {"both", "physical"})
        for graphics in self._sweep_graphics.values():
            visible = mode in {"both", "physical"}
            graphics["lower_curve"].setVisible(visible)
            graphics["upper_curve"].setVisible(visible)
            graphics["fill"].setVisible(visible)
        self._update_highlight()
        self.fit_view()

    def set_sweep_envelope(
        self,
        port_index: int,
        time_ns: Sequence[float],
        endpoint_a_mv: Sequence[float],
        endpoint_b_mv: Sequence[float],
    ) -> None:
        """Backward-compatible single-envelope wrapper."""
        self.set_sweep_envelopes(
            (("legacy", port_index, time_ns, endpoint_a_mv, endpoint_b_mv),)
        )

    def _create_sweep_graphics(self):
        if not self._sweep_graphics:
            return {
                "lower_curve": self._sweep_lower_curve,
                "upper_curve": self._sweep_upper_curve,
                "fill": self._sweep_fill,
            }
        color = _plot_color(0)
        lower_curve = self.plot(
            [],
            [],
            pen=pg.mkPen(color, width=1.0, style=QtCore.Qt.DotLine),
        )
        upper_curve = self.plot(
            [],
            [],
            pen=pg.mkPen(color, width=1.0, style=QtCore.Qt.DotLine),
        )
        fill = pg.FillBetweenItem(
            upper_curve,
            lower_curve,
            brush=pg.mkBrush(color.red(), color.green(), color.blue(), 24),
        )
        self.addItem(fill)
        fill.setZValue(0.1)
        lower_curve.setZValue(0.8)
        upper_curve.setZValue(0.8)
        return {"lower_curve": lower_curve, "upper_curve": upper_curve, "fill": fill}

    def set_sweep_envelopes(self, envelopes) -> None:
        """Draw independent sweep endpoint/fill graphics for multiple targets."""
        active_keys = set()
        bounds_time = []
        bounds_lower = []
        bounds_upper = []
        first = None
        for key, port_index, time_ns, endpoint_a_mv, endpoint_b_mv in envelopes:
            if port_index < 0 or port_index >= len(self._pulses):
                raise IndexError("sweep port index is out of range")
            time_values = np.asarray(time_ns, dtype=float)
            endpoint_a = np.asarray(endpoint_a_mv, dtype=float)
            endpoint_b = np.asarray(endpoint_b_mv, dtype=float)
            if (
                time_values.ndim != 1
                or endpoint_a.shape != time_values.shape
                or endpoint_b.shape != time_values.shape
            ):
                raise ValueError("sweep endpoint traces must be equal-length vectors")
            lower = np.minimum(endpoint_a, endpoint_b)
            upper = np.maximum(endpoint_a, endpoint_b)
            graphics = self._sweep_graphics.get(key)
            if graphics is None:
                graphics = self._create_sweep_graphics()
                self._sweep_graphics[key] = graphics
            color = self.line_color(port_index)
            endpoint_pen = pg.mkPen(color, width=1.0, style=QtCore.Qt.DotLine)
            graphics["lower_curve"].setPen(endpoint_pen)
            graphics["upper_curve"].setPen(endpoint_pen)
            graphics["fill"].setBrush(
                pg.mkBrush(color.red(), color.green(), color.blue(), 24)
            )
            graphics["lower_curve"].setData(time_values, lower)
            graphics["upper_curve"].setData(time_values, upper)
            visible = self._voltage_view in {"both", "physical"}
            graphics["lower_curve"].setVisible(visible)
            graphics["upper_curve"].setVisible(visible)
            graphics["fill"].setVisible(visible)
            graphics.update(
                port_index=port_index,
                time_ns=time_values,
                lower_mv=lower,
                upper_mv=upper,
            )
            active_keys.add(key)
            bounds_time.append(time_values)
            bounds_lower.append(lower)
            bounds_upper.append(upper)
            if first is None:
                first = graphics

        for key, graphics in self._sweep_graphics.items():
            if key not in active_keys:
                graphics["lower_curve"].setData([], [])
                graphics["upper_curve"].setData([], [])

        if first is None:
            self._sweep_port_index = None
            self._sweep_time_ns = np.asarray([], dtype=float)
            self._sweep_lower_mv = np.asarray([], dtype=float)
            self._sweep_upper_mv = np.asarray([], dtype=float)
            return

        # Preserve the original single-sweep inspection attributes.
        self._sweep_lower_curve = first["lower_curve"]
        self._sweep_upper_curve = first["upper_curve"]
        self._sweep_fill = first["fill"]
        self._sweep_port_index = first["port_index"]
        self._sweep_time_ns = np.concatenate(bounds_time)
        self._sweep_lower_mv = np.concatenate(bounds_lower)
        self._sweep_upper_mv = np.concatenate(bounds_upper)

    def clear_sweep_envelope(self) -> None:
        self.set_sweep_envelopes(())

    def fit_view(self) -> None:
        if not self._pulses:
            return
        x_values = []
        y_values = []
        if self._voltage_view in {"both", "virtual"}:
            x_values.extend(pulse.t for pulse in self._pulses)
            y_values.extend(pulse.v for pulse in self._pulses)
        if (
            self._voltage_view in {"both", "physical"}
            and self._physical_time_ns.size
            and self._physical_values_mv.size
        ):
            x_values.append(self._physical_time_ns)
            y_values.extend(self._physical_values_mv)
        if not x_values or not y_values:
            return
        x_min = min(float(np.min(values)) for values in x_values)
        x_max = max(float(np.max(values)) for values in x_values)
        y_min = min(float(np.min(values)) for values in y_values)
        y_max = max(float(np.max(values)) for values in y_values)
        if self._sweep_time_ns.size:
            x_min = min(x_min, float(np.min(self._sweep_time_ns)))
            x_max = max(x_max, float(np.max(self._sweep_time_ns)))
            y_min = min(y_min, float(np.min(self._sweep_lower_mv)))
            y_max = max(y_max, float(np.max(self._sweep_upper_mv)))
        x_margin = max(1.0, 0.03 * max(1.0, x_max - x_min))
        y_margin = max(0.5, 0.10 * max(1.0, y_max - y_min))
        self.setXRange(x_min - x_margin, x_max + x_margin, padding=0.0)
        self.setYRange(y_min - y_margin, y_max + y_margin, padding=0.0)

    def refresh(self, index: Optional[int] = None) -> None:
        """Update existing curves without rebuilding axes or graphics items."""
        indices = range(len(self._pulses)) if index is None else (index,)
        for pulse_index in indices:
            pulse = self._pulses[pulse_index]
            self._line[pulse_index].setData(pulse.t, pulse.v)

    def add_pulse(self, pulse: PulseSequence) -> None:
        self._pulses.append(pulse)
        self._append_curve(pulse)
        self._selected_port_idx = len(self._pulses) - 1
        self._update_highlight()
        self.fit_view()

    def get_selected_port_idx(self) -> int:
        return self._selected_port_idx

    def set_selected_port_idx(self, index: int) -> None:
        if index < 0 or index >= len(self._pulses):
            raise IndexError("selected port index is out of range")
        self._selected_port_idx = index
        self._update_highlight()

    def remove_pulse(self, index: int) -> None:
        if index < 0 or index >= len(self._pulses):
            raise IndexError("pulse index is out of range")
        curve = self._line.pop(index)
        self.removeItem(curve)
        physical_curve = self._physical_line.pop(index)
        self.removeItem(physical_curve)
        self._pulses.pop(index)
        self._orig_colors.pop(index)
        self._physical_time_ns = np.asarray([], dtype=float)
        self._physical_values_mv = np.empty((0, 0), dtype=float)
        self._selected_port_idx = min(self._selected_port_idx, len(self._pulses) - 1)
        self._update_highlight()

    def _scene_position(self, event) -> QtCore.QPointF:
        return self.mapToScene(event.pos())

    def _view_position(self, event) -> QtCore.QPointF:
        return self.getPlotItem().vb.mapSceneToView(self._scene_position(event))

    def _locate_point(self, event, tolerance_px: float = 9.0) -> Optional[Tuple[int, int]]:
        scene_position = self._scene_position(event)
        pulse = self._pulses[self._selected_port_idx]
        view_box = self.getPlotItem().vb
        for index, (time_ns, voltage_mv) in enumerate(zip(pulse.t, pulse.v)):
            point = view_box.mapViewToScene(
                QtCore.QPointF(float(time_ns), float(voltage_mv))
            )
            if hypot(point.x() - scene_position.x(), point.y() - scene_position.y()) <= tolerance_px:
                return index, min(index + 1, len(pulse.t) - 1)
        return None

    def _locate_flat(self, event, tolerance_px: float = 9.0) -> Optional[Tuple[int, int]]:
        scene_position = self._scene_position(event)
        pulse = self._pulses[self._selected_port_idx]
        view_box = self.getPlotItem().vb
        for start_index, end_index in pulse.flat_segments():
            start = view_box.mapViewToScene(
                QtCore.QPointF(float(pulse.t[start_index]), float(pulse.v[start_index]))
            )
            end = view_box.mapViewToScene(
                QtCore.QPointF(float(pulse.t[end_index]), float(pulse.v[end_index]))
            )
            left, right = sorted((start.x(), end.x()))
            inner_left = left + 0.18 * (right - left)
            inner_right = right - 0.18 * (right - left)
            if (
                inner_left <= scene_position.x() <= inner_right
                and abs(scene_position.y() - start.y()) <= tolerance_px
            ):
                return start_index, end_index
        return None

    def eventFilter(self, watched, event):
        if self._voltage_view == "physical":
            return super().eventFilter(watched, event)
        if watched is self.viewport():
            if event.type() == QtCore.QEvent.MouseButtonPress:
                if event.button() == QtCore.Qt.LeftButton:
                    point = self._locate_point(event)
                    flat = None if point is not None else self._locate_flat(event)
                    if point is not None:
                        self._drag_point = point
                        self.setCursor(QtCore.Qt.SizeHorCursor)
                        return True
                    if flat is not None:
                        self._drag_flat = flat
                        self.setCursor(QtCore.Qt.SizeVerCursor)
                        return True
            elif event.type() == QtCore.QEvent.MouseMove:
                if self._drag_flat is not None or self._drag_point is not None:
                    self.mouseMoveEvent(event)
                    return True
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                if self._drag_flat is not None or self._drag_point is not None:
                    self.mouseReleaseEvent(event)
                    return True
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event) -> None:
        if self._voltage_view == "physical":
            super().mousePressEvent(event)
            return
        if event.button() == QtCore.Qt.LeftButton:
            point = self._locate_point(event)
            flat = None if point is not None else self._locate_flat(event)
            if point is not None:
                self._drag_point = point
                self.setCursor(QtCore.Qt.SizeHorCursor)
                event.accept()
                return
            if flat is not None:
                self._drag_flat = flat
                self.setCursor(QtCore.Qt.SizeVerCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_flat is not None:
            pulse = self._pulses[self._selected_port_idx]
            position = self._view_position(event)
            new_voltage = self._snap_voltage(float(position.y()), pulse)
            start_index, end_index = self._drag_flat
            pulse.update_flat(self._drag_flat, new_voltage)
            self.refresh(self._selected_port_idx)
            self.flat_moved.emit(start_index, end_index, new_voltage)
            self._show_annotation(
                float(pulse.t[start_index]),
                new_voltage,
                f"{pulse.t[start_index]:.6g} ns\n{new_voltage:.6g} mV",
            )
            event.accept()
            return
        if self._drag_point is not None:
            pulse = self._pulses[self._selected_port_idx]
            position = self._view_position(event)
            x_range = self.getPlotItem().vb.viewRange()[0]
            new_time = float(np.clip(position.x(), x_range[0], x_range[1]))
            start_index, end_index = self._drag_point
            new_time = self._snap_time(new_time, pulse, start_index)
            pulse.update_point(self._drag_point, new_time)
            actual_time = float(pulse.t[start_index])
            self.refresh(self._selected_port_idx)
            self.point_moved.emit(start_index, end_index, actual_time)
            self._show_annotation(
                actual_time,
                float(pulse.v[start_index]),
                f"{actual_time:.6g} ns\n{pulse.v[start_index]:.6g} mV",
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_flat is not None or self._drag_point is not None:
            self._drag_flat = None
            self._drag_point = None
            self.unsetCursor()
            self._annotation.hide()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        view_box = self.getPlotItem().vb
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers & QtCore.Qt.ControlModifier:
            view_box.setMouseEnabled(x=True, y=False)
        elif modifiers & QtCore.Qt.ShiftModifier:
            view_box.setMouseEnabled(x=False, y=True)
        else:
            view_box.setMouseEnabled(x=True, y=True)
        try:
            super().wheelEvent(event)
        finally:
            view_box.setMouseEnabled(x=True, y=True)

    def _show_annotation(self, x: float, y: float, text: str) -> None:
        self._annotation.setText(text)
        self._annotation.setPos(x, y)
        self._annotation.show()

    def _update_highlight(self) -> None:
        for index, curve in enumerate(self._line):
            base = QtGui.QColor(self._orig_colors[index])
            if index == self._selected_port_idx:
                color = base
                width = self._highlight_width
                curve.setZValue(100)
            else:
                color = base.lighter(125)
                color.setAlpha(165)
                width = self._default_width
                curve.setZValue(index + 1)
            curve.setPen(pg.mkPen(color, width=width))
            curve.setSymbolPen(pg.mkPen(color))
            curve.setSymbolBrush(pg.mkBrush(color))
            physical = self._physical_line[index]
            physical_color = QtGui.QColor(color)
            physical_color.setAlpha(220 if index == self._selected_port_idx else 145)
            physical.setPen(
                pg.mkPen(
                    physical_color,
                    width=(
                        self._physical_width + 0.5
                        if index == self._selected_port_idx
                        else self._physical_width
                    ),
                    style=QtCore.Qt.DashLine,
                )
            )
            physical.setZValue(99 if index == self._selected_port_idx else index + 0.5)

    def _restore_full_intensity(self) -> None:
        for index, curve in enumerate(self._line):
            color = self._orig_colors[index]
            curve.setPen(pg.mkPen(color, width=self._default_width))
            curve.setSymbolPen(pg.mkPen(color))
            curve.setSymbolBrush(pg.mkBrush(color))
            curve.setZValue(index + 1)
            physical = self._physical_line[index]
            physical.setPen(
                pg.mkPen(
                    color,
                    width=self._physical_width,
                    style=QtCore.Qt.DashLine,
                )
            )
            physical.setZValue(index + 0.5)


__all__ = [
    "RfPulsePreviewWidget",
    "RfPulseTimelineWidget",
    "TracePlotWidget",
    "WaveformPlotWidget",
]
