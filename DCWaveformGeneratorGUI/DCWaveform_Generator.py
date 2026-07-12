#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive DC waveform editor with Keysight QCS and QICK export.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

import json
from math import prod
from pathlib import Path
import sys
from typing import Tuple, Optional, List, Callable, Sequence
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph  # noqa: F401
except ImportError:
    # Matplotlib remains a compatibility fallback.  The normal optimized path
    # imports no Matplotlib modules during application startup.
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib import colors
    from matplotlib.ticker import MultipleLocator
    _USE_PYQTGRAPH = False
else:
    # Names below are referenced only by the legacy fallback class bodies.
    Canvas = QtWidgets.QWidget
    Figure = Axes = Line2D = object
    colors = None
    MultipleLocator = None
    _USE_PYQTGRAPH = True

try:
    from .dc_waveform_core import (
        DEFAULT_QICK_FABRIC_MHZ,
        DEFAULT_QICK_FULL_SCALE_MV,
        PulseSequence,
        QickDdrReadoutSpec,
        QickRfPulseSpec,
        QickSweepSpec,
        build_qick_sequence,
        generate_qcs_program_code,
        generate_qick_program_code,
        qick_set_segment_names,
        transform_virtual_waveforms,
    )
except ImportError:
    from dc_waveform_core import (
        DEFAULT_QICK_FABRIC_MHZ,
        DEFAULT_QICK_FULL_SCALE_MV,
        PulseSequence,
        QickDdrReadoutSpec,
        QickRfPulseSpec,
        QickSweepSpec,
        build_qick_sequence,
        generate_qcs_program_code,
        generate_qick_program_code,
        qick_set_segment_names,
        transform_virtual_waveforms,
    )


DEFAULT_QSTL_AWG_CHANNELS = (1, 3, 5, 7, 8, 9, 10, 11)


def rf_pulse_absolute_times_us(
    pulse: PulseSequence,
    spec: QickRfPulseSpec,
) -> Tuple[float, float, float]:
    """Resolve an RF pulse's SET-relative timing onto the GUI sequence axis."""
    try:
        set_index = int(spec.segment_name.rsplit("_", 1)[1])
        start_point, end_point = pulse.flat_segments()[set_index]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"unknown RF SET segment {spec.segment_name!r}") from exc
    set_start_us = float(pulse.t[start_point]) / 1000.0
    set_end_us = float(pulse.t[end_point]) / 1000.0
    pulse_start_us = set_start_us + spec.delay_us
    pulse_end_us = pulse_start_us + spec.duration_us
    if spec.require_within_segment and pulse_end_us > set_end_us + 1.0e-12:
        raise ValueError(
            f"RF pulse ends at {pulse_end_us:.6g} us, after {spec.segment_name} "
            f"ends at {set_end_us:.6g} us"
        )
    return pulse_start_us, pulse_end_us, set_end_us


class _MatplotlibTracePlotWidget(Canvas):
    """Scatter/line plot that traces Pulse-X vs Pulse-Y (voltage-voltage)."""
    def __init__(self, parent=None):
        fig = Figure(tight_layout=False)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.line: Line2D | None = None
        self.x_idx: int | None = None
        self.y_idx: int | None = None
        self.ax.set_xlabel("Pulse-X [mV]")
        self.ax.set_ylabel("Pulse-Y [mV]")
        self.ax.grid(True)
        self._pulse: List[PulseSequence] = []
        self._pan_origin: Optional[Tuple[float, float]] = None

        self.mpl_connect("motion_notify_event",  self._on_move)
        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("scroll_event",         self._on_scroll)

    def refresh_trace(self, pulses: list[PulseSequence]):
        """Refresh the trace plot with the selected pulses."""
        self._pulse = pulses
        if (
            self.x_idx is None or
            self.y_idx is None or
            self.x_idx >= len(self._pulse) or
            self.y_idx >= len(self._pulse)
        ):
            self.ax.cla()
            self.draw_idle()
            return
        px, py = pulses[self.x_idx], pulses[self.y_idx]

        # Build common time grid then interpolate
        t_union = np.unique(np.concatenate((px.t, py.t)))
        vx = np.interp(t_union, px.t, px.v)
        vy = np.interp(t_union, py.t, py.v)

        prev_x_lim = self.ax.get_xlim()
        prev_y_lim = self.ax.get_ylim()
        self.ax.cla()
        self.ax.set_xlabel(f"Pulse {self.x_idx+1} [mV]")
        self.ax.set_ylabel(f"Pulse {self.y_idx+1} [mV]")
        self.ax.grid(True)
        self.ax.plot(vx, vy, "-o")
        self.ax.set_xlim(prev_x_lim)
        self.ax.set_ylim(prev_y_lim)
        self.draw_idle()

    def fit_view(self):
        """Fit the view to the pulse data."""
        if (
            self.x_idx is None or
            self.y_idx is None or
            self.x_idx >= len(self._pulse) or
            self.y_idx >= len(self._pulse)
        ):
            self.ax.cla()
            self.draw_idle()
            return
        margin_x = 0.03 * (np.ptp(self._pulse[self.x_idx].v) + 1e-12) or 0.5
        margin_y = 0.10 * (np.ptp(self._pulse[self.y_idx].v) + 1e-12) or 0.5
        x_min = self._pulse[self.x_idx].v.min() - margin_x
        x_max = self._pulse[self.x_idx].v.max() + margin_x
        y_min = self._pulse[self.y_idx].v.min() - margin_y
        y_max = self._pulse[self.y_idx].v.max() + margin_y

        self.ax.set_xlim(
            x_min - margin_x,
            x_max + margin_x
        )
        self.ax.set_ylim(
            y_min - margin_y,
            y_max + margin_y
        )
        self.draw_idle()

    def _on_scroll(self, event):
        factor = -0.1 if event.step > 0 else +0.1
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        mods = (
            set(event.modifiers) if hasattr(event, "modifiers")
            else {event.key} if event.key else set()
        )

        if "ctrl" in mods:      # Ctrl + wheel : horizontal zoom
            new_w = (xmax - xmin) * factor
            self.ax.set_xlim(xmin - 0.5*new_w, xmax + 0.5*new_w)

        elif "shift" in mods:   # Shift + wheel : vertical zoom
            new_h = (ymax - ymin) * factor
            self.ax.set_ylim(ymin - 0.5*new_h, ymax + 0.5*new_h)

        else:                   # plain wheel : isotropic zoom
            new_w = (xmax - xmin) * factor
            new_h = (ymax - ymin) * factor
            self.ax.set_xlim(xmin - 0.5*new_w, xmax + 0.5*new_w)
            self.ax.set_ylim(ymin - 0.5*new_h, ymax + 0.5*new_h)

        self.draw_idle()

    def _on_press(self, event):
        if (event.button != 1 or not event.inaxes):
            return
        self._pan_origin = (event.xdata, event.ydata)

    def _on_move(self, event):
        if self._pan_origin and event.xdata is not None and event.ydata is not None:
            x0, y0 = self._pan_origin
            dx = x0 - event.xdata
            dy = y0 - event.ydata
            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            self.ax.set_xlim(xmin + dx, xmax + dx)
            self.ax.set_ylim(ymin + dy, ymax + dy)
            self.draw_idle()

    def _on_release(self, _event):
        self._pan_origin = None

class _MatplotlibWaveformPlotWidget(Canvas): # pylint: disable=too-many-instance-attributes
    """Matplotlib widget for displaying and editing a PulseSequence."""
    flat_moved = QtCore.pyqtSignal(int, int, float)   # i0, i1, new_v
    point_moved = QtCore.pyqtSignal(int, int, float)  # i0, i1, new_v

    def __init__(
            self,
            pulse: PulseSequence,
            parent=None
        ):
        fig                 = Figure(tight_layout=False)
        super().__init__(fig)
        # Pulse and line settings
        self._pulse: List[PulseSequence]    = [pulse]
        self._line: List[Axes]              = []
        self._selected_port_idx             = 0

        # Color highlight settings
        self._default_alpha   = 1.0
        self._dim_alpha       = 0.95
        self._default_lw      = 1.5
        self._highlight_lw    = 2.5
        self._orig_colors     = []

        self.ax             = fig.add_subplot(111)
        self._line,         = [self.ax.plot(self._pulse[0].t, self._pulse[0].v, "-o", picker=5)]
        self._orig_colors.append(self._line[0].get_color())
        physical_line, = self.ax.plot(
            self._pulse[0].t,
            self._pulse[0].v,
            "--",
            color=self._orig_colors[0],
            linewidth=1.4,
            zorder=0.5,
        )
        self._physical_line = [physical_line]
        self._physical_time_ns = self._pulse[0].t.copy()
        self._physical_values_mv = np.asarray([self._pulse[0].v.copy()])
        self._voltage_view = "both"

        # Graph settings
        self.ax.set_xlabel("time [ns]")
        self.ax.set_ylabel("voltage [mV]")
        self.ax.grid(True)
        self.ax.set_autoscale_on(False)

        # Dragging and panning settings
        self._drag_flat: Optional[Tuple[int, int]]      = None
        self._drag_point: Optional[Tuple[int, int]]     = None
        self._pan_origin: Optional[Tuple[float, float]] = None
        self._grid_time_ns = 10.0
        self._grid_voltage_mv = 10.0
        self._grid_snap_enabled = False
        self._grid_visible = True
        self._sweep_artists = []
        self._sweep_port_index = None
        self._sweep_time_ns = np.asarray([], dtype=float)
        self._sweep_lower_mv = np.asarray([], dtype=float)
        self._sweep_upper_mv = np.asarray([], dtype=float)
        self._annot         = self.ax.annotate(
            "",
            xy              = (0, 0),
            xytext          = (12, 12),
            textcoords      = "offset points",
            bbox            = dict(boxstyle="round", fc="w"),
            arrowprops      = dict(arrowstyle="->"),
            visible         = False
        )

        self.mpl_connect("motion_notify_event",  self._on_move)
        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("scroll_event",         self._on_scroll)

        self.set_grid(
            time_step_ns=self._grid_time_ns,
            voltage_step_mv=self._grid_voltage_mv,
            snap_enabled=self._grid_snap_enabled,
            visible=self._grid_visible,
        )
        self.fit_view()

    @staticmethod
    def _nearest_grid_value(value: float, step: float) -> float:
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
        if not np.isfinite(time_step_ns) or time_step_ns <= 0.0:
            raise ValueError("time grid spacing must be a positive finite value")
        if not np.isfinite(voltage_step_mv) or voltage_step_mv <= 0.0:
            raise ValueError("voltage grid spacing must be a positive finite value")
        self._grid_time_ns = float(time_step_ns)
        self._grid_voltage_mv = float(voltage_step_mv)
        self._grid_snap_enabled = bool(snap_enabled)
        self._grid_visible = bool(visible)
        self.ax.xaxis.set_major_locator(MultipleLocator(self._grid_time_ns))
        self.ax.yaxis.set_major_locator(MultipleLocator(self._grid_voltage_mv))
        self.ax.grid(self._grid_visible)
        self.draw_idle()

    @property
    def grid_settings(self) -> Tuple[float, float, bool, bool]:
        return (
            self._grid_time_ns,
            self._grid_voltage_mv,
            self._grid_snap_enabled,
            self._grid_visible,
        )

    @property
    def voltage_view(self) -> str:
        return self._voltage_view

    def set_physical_waveforms(self, time_ns, waveforms_mv) -> None:
        time_values = np.asarray(time_ns, dtype=float)
        waveform_values = np.asarray(waveforms_mv, dtype=float)
        expected_shape = (len(self._pulse), time_values.size)
        if time_values.ndim != 1 or waveform_values.shape != expected_shape:
            raise ValueError(
                "physical waveforms must have shape "
                f"{expected_shape}, received {waveform_values.shape}"
            )
        self._physical_time_ns = time_values.copy()
        self._physical_values_mv = waveform_values.copy()
        for index, line in enumerate(self._physical_line):
            line.set_data(time_values, waveform_values[index])
        self.draw_idle()

    def set_voltage_view(self, mode: str) -> None:
        if mode not in {"both", "virtual", "physical"}:
            raise ValueError("voltage view must be 'both', 'virtual', or 'physical'")
        self._voltage_view = mode
        self._drag_flat = None
        self._drag_point = None
        for line in self._line:
            line.set_visible(mode in {"both", "virtual"})
        for line in self._physical_line:
            line.set_visible(mode in {"both", "physical"})
        for artist in self._sweep_artists:
            artist.set_visible(mode in {"both", "physical"})
        self.fit_view()

    def set_sweep_envelope(
        self,
        port_index: int,
        time_ns,
        endpoint_a_mv,
        endpoint_b_mv,
    ) -> None:
        self.set_sweep_envelopes(
            (("legacy", port_index, time_ns, endpoint_a_mv, endpoint_b_mv),)
        )

    def set_sweep_envelopes(self, envelopes) -> None:
        self.clear_sweep_envelope()
        time_bounds = []
        lower_bounds = []
        upper_bounds = []
        for _, port_index, time_ns, endpoint_a_mv, endpoint_b_mv in envelopes:
            time_values = np.asarray(time_ns, dtype=float)
            endpoint_a = np.asarray(endpoint_a_mv, dtype=float)
            endpoint_b = np.asarray(endpoint_b_mv, dtype=float)
            lower = np.minimum(endpoint_a, endpoint_b)
            upper = np.maximum(endpoint_a, endpoint_b)
            color = self._orig_colors[port_index]
            lower_line, = self.ax.plot(
                time_values, lower, ":", color=color, linewidth=1.0
            )
            upper_line, = self.ax.plot(
                time_values, upper, ":", color=color, linewidth=1.0
            )
            fill = self.ax.fill_between(
                time_values, lower, upper, color=color, alpha=0.09
            )
            self._sweep_artists.extend([lower_line, upper_line, fill])
            for artist in (lower_line, upper_line, fill):
                artist.set_visible(self._voltage_view in {"both", "physical"})
            if self._sweep_port_index is None:
                self._sweep_port_index = port_index
            time_bounds.append(time_values)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        if time_bounds:
            self._sweep_time_ns = np.concatenate(time_bounds)
            self._sweep_lower_mv = np.concatenate(lower_bounds)
            self._sweep_upper_mv = np.concatenate(upper_bounds)
        self.draw_idle()

    def clear_sweep_envelope(self) -> None:
        for artist in getattr(self, "_sweep_artists", ()):
            artist.remove()
        self._sweep_artists = []
        self._sweep_port_index = None
        self._sweep_time_ns = np.asarray([], dtype=float)
        self._sweep_lower_mv = np.asarray([], dtype=float)
        self._sweep_upper_mv = np.asarray([], dtype=float)
        self.draw_idle()

    def fit_view(self) -> None:
        """Fit the view to the pulse data."""
        x_values = []
        y_values = []
        if self._voltage_view in {"both", "virtual"}:
            x_values.extend(pulse.t for pulse in self._pulse)
            y_values.extend(pulse.v for pulse in self._pulse)
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
        margin_x = max(1.0, 0.03 * max(1.0, x_max - x_min))
        margin_y = max(0.5, 0.10 * max(1.0, y_max - y_min))
        if self._sweep_time_ns.size:
            x_min = min(x_min, float(np.min(self._sweep_time_ns)) - margin_x)
            x_max = max(x_max, float(np.max(self._sweep_time_ns)) + margin_x)
            y_min = min(y_min, float(np.min(self._sweep_lower_mv)) - margin_y)
            y_max = max(y_max, float(np.max(self._sweep_upper_mv)) + margin_y)
        self.ax.set_xlim(
            x_min,
            x_max
        )
        self.ax.set_ylim(
            y_min,
            y_max
        )
        self.draw_idle()

    def refresh(self) -> None:
        """Refresh plot"""
        for i, pulse in enumerate(self._pulse):
            self._line[i].set_data(pulse.t, pulse.v)
        self.draw_idle()

    def add_pulse(self, pulse: PulseSequence) -> None:
        """Add a new pulse to the plot."""
        self._pulse.append(pulse)
        line,                   = self.ax.plot(pulse.t, pulse.v, "-o", picker=5)
        self._line.append(line)
        self._orig_colors.append(line.get_color())
        physical_line, = self.ax.plot(
            pulse.t,
            pulse.v,
            "--",
            color=line.get_color(),
            linewidth=1.4,
            zorder=0.5,
        )
        self._physical_line.append(physical_line)
        line.set_visible(self._voltage_view in {"both", "virtual"})
        physical_line.set_visible(self._voltage_view in {"both", "physical"})
        self._selected_port_idx = len(self._pulse) - 1
        self.refresh()
        self.fit_view()
        self._update_highlight()

    def get_selected_port_idx(self) -> int:
        """Get the index of the currently selected port."""
        return self._selected_port_idx

    def set_selected_port_idx(self, idx: int) -> None:
        """Set the index of the currently selected port."""
        self._selected_port_idx = idx
        self._update_highlight()

    def remove_pulse(self, idx: int) -> None:
        """Remove the pulse/line at position `idx`."""
        if idx >= len(self._pulse):
            raise ValueError("Index out of bounds for pulse removal.")
        ln = self._line.pop(idx)
        ln.remove()
        physical_line = self._physical_line.pop(idx)
        physical_line.remove()
        self._pulse.pop(idx)
        self._orig_colors.pop(idx)
        self._physical_time_ns = np.asarray([], dtype=float)
        self._physical_values_mv = np.empty((0, 0), dtype=float)
        if self._selected_port_idx >= len(self._pulse):
            self._selected_port_idx = max(0, len(self._pulse) - 1)
        self._update_highlight()
        self.draw_idle()

    def _locate_flat_segment(self, event) -> Optional[Tuple[int, int]]:
        """Return range (i0,i1) if click is on *flat* part, else None."""
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return None
        line_x, line_y = self._line[self._selected_port_idx].get_data()
        # Check Flat Click Events
        for i0 in range(len(self._pulse[self._selected_port_idx].t)):
            if i0 % 2 != 0:
                continue
            if i0 == len(self._pulse[self._selected_port_idx].t) - 1:
                break
            i1 = i0 + 1
            # mid-point of flat
            xm = 0.5 * (line_x[i0] + line_x[i1])
            ym = line_y[i0]
            dx = abs(x - xm)
            dy = abs(y - ym)
            if (
                dx < 0.25 * (line_x[i1] - line_x[i0])
                and dy < 0.05 * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
            ):
                return (i0, i1)
        return None

    def _locate_point_segment(self, event) -> Optional[Tuple[int, int]]:
        """Return range (i0,i1) if click is on *flat* part, else None."""
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return None
        line_x, line_y = self._line[self._selected_port_idx].get_data()
        # Check Point Click Events
        for i0 in range(len(self._pulse[self._selected_port_idx].t)):
            dx = abs(x - line_x[i0])
            dy = abs(y - line_y[i0])
            if (
                dx < 0.05 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
                and dy < 0.05 * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
            ):
                return (i0, i0+1)
        return None

    def _on_press(self, event) -> None:
        if (event.button != 1 or not event.inaxes):
            return
        if self._voltage_view == "physical":
            self._pan_origin = (event.xdata, event.ydata)
            return
        seg     = self._locate_flat_segment(event)
        point   = self._locate_point_segment(event)
        if seg:
            self._drag_flat = seg
        elif point:
            self._drag_point = point
        else:
            self._pan_origin = (event.xdata, event.ydata)

    def _on_move(self, event) -> None:
        # tooltip
        self._annot.set_visible(False)

        # drag flat
        if self._drag_flat and event.ydata is not None:
            new_v       = np.clip(
                event.ydata,
                self._pulse[self._selected_port_idx].v_bounds[0],
                self._pulse[self._selected_port_idx].v_bounds[1]
            )
            if self._grid_snap_enabled:
                new_v = self._nearest_grid_value(new_v, self._grid_voltage_mv)
                new_v = np.clip(
                    new_v,
                    np.ceil(self._pulse[self._selected_port_idx].v_bounds[0] / self._grid_voltage_mv)
                    * self._grid_voltage_mv,
                    np.floor(self._pulse[self._selected_port_idx].v_bounds[1] / self._grid_voltage_mv)
                    * self._grid_voltage_mv,
                )
            i0, i1      = self._drag_flat
            self._pulse[self._selected_port_idx].update_flat(self._drag_flat, new_v)
            self.flat_moved.emit(i0, i1, new_v)
            self.refresh()

            self._update_annot(
                self._pulse[self._selected_port_idx].t[i0],
                new_v,
                f"{self._pulse[self._selected_port_idx].t[i0]:0.3g} ns\n{new_v:0.3g} mV"
            )
            self._annot.set_visible(True)

        # drag point
        if self._drag_point and event.xdata is not None:
            xmin, xmax  = self.ax.get_xlim()
            new_t       = np.clip(event.xdata, xmin, xmax)
            i0, i1      = self._drag_point
            if self._grid_snap_enabled and i0 > 0:
                new_t = self._nearest_grid_value(new_t, self._grid_time_ns)
                minimum = np.nextafter(
                    float(self._pulse[self._selected_port_idx].t[i0 - 1]),
                    np.inf,
                )
                if new_t < minimum:
                    new_t = np.ceil(minimum / self._grid_time_ns) * self._grid_time_ns
            self._pulse[self._selected_port_idx].update_point(self._drag_point, new_t)
            actual_t = float(self._pulse[self._selected_port_idx].t[i0])
            self.point_moved.emit(i0, i1, actual_t)
            self.refresh()

            self._update_annot(
                actual_t,
                self._pulse[self._selected_port_idx].v[i0],
                f"{actual_t:0.3g} ns\n{self._pulse[self._selected_port_idx].v[i0]:0.3g} mV"
            )
            self._annot.set_visible(True)
        # pan
        if self._pan_origin and event.xdata is not None and event.ydata is not None:
            x0, y0      = self._pan_origin
            dx          = x0 - event.xdata
            dy          = y0 - event.ydata
            xmin, xmax  = self.ax.get_xlim()
            ymin, ymax  = self.ax.get_ylim()
            self.ax.set_xlim(xmin + dx, xmax + dx)
            self.ax.set_ylim(ymin + dy, ymax + dy)
            self.draw_idle()

    def _on_release(self, _event):
        self._drag_flat  = None
        self._drag_point = None
        self._pan_origin = None

    def _on_scroll(self, event):
        if not event.inaxes:
            return

        factor = -0.1 if event.step > 0 else +0.1
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        mods = (
            set(event.modifiers) if hasattr(event, "modifiers")
            else {event.key} if event.key else set()
        )

        # Ctrl + wheel : horizontal zoom
        if "ctrl" in mods:
            new_w = (xmax - xmin) * factor
            self.ax.set_xlim(xmin - 0.5*new_w, xmax + 0.5*new_w)

        # Shift + wheel : vertical zoom
        elif "shift" in mods:
            new_h = (ymax - ymin) * factor
            self.ax.set_ylim(ymin - 0.5*new_h, ymax + 0.5*new_h)

        # plain wheel : isotropic zoom
        else:
            new_w = (xmax - xmin) * factor
            new_h = (ymax - ymin) * factor
            self.ax.set_xlim(xmin - 0.5*new_w, xmax + 0.5*new_w)
            self.ax.set_ylim(ymin - 0.5*new_h, ymax + 0.5*new_h)

        self.draw_idle()

    def _update_highlight(self):
        for i, line in enumerate(self._line):
            if i == self._selected_port_idx:
                line.set_zorder(10)
                line.set_alpha(self._default_alpha)
                line.set_linewidth(self._highlight_lw)
                line.set_color(self._orig_colors[i])
            else:
                line.set_zorder(1)
                line.set_alpha(self._dim_alpha)
                line.set_linewidth(self._default_lw)
                line.set_color(self._lighten(self._orig_colors[i], 0.1))
            physical = self._physical_line[i]
            physical.set_color(line.get_color())
            physical.set_alpha(line.get_alpha())
            physical.set_linewidth(1.8 if i == self._selected_port_idx else 1.4)
            physical.set_zorder(9 if i == self._selected_port_idx else 0.5)
        self.draw_idle()

    @staticmethod
    def _lighten(c, amount: float = 0.5):
        r, g, b, a = colors.to_rgba(c)
        white      = 1.0
        return (
            r + (white - r)*amount,
            g + (white - g)*amount,
            b + (white - b)*amount,
            a
        )

    def _restore_full_intensity(self):
        for i, line in enumerate(self._line):
            line.set_alpha(self._default_alpha)
            line.set_linewidth(self._default_lw)
            line.set_color(self._orig_colors[i])
            line.set_zorder(1)
            physical = self._physical_line[i]
            physical.set_alpha(self._default_alpha)
            physical.set_linewidth(1.4)
            physical.set_color(self._orig_colors[i])
            physical.set_zorder(0.5)
        self.draw_idle()

    def _update_annot(self, x, y, text: str):
        self._annot.xy = (x, y)
        self._annot.set_text(text)

        xmid = sum(self.ax.get_xlim()) / 2
        ymid = sum(self.ax.get_ylim()) / 2

        dx, dy =  +12, +12
        ha,  va = "left", "bottom"

        if x > xmid:
            dx, ha = -dx, "right"
        if y > ymid:
            dy, va = -dy, "top"

        self._annot.set_position((dx, dy))
        self._annot.set_ha(ha)
        self._annot.set_va(va)
        self._annot.set_visible(True)
        self.ax.figure.canvas.draw_idle()


if _USE_PYQTGRAPH:
    try:
        from .dc_waveform_widgets import (
            RfPulsePreviewWidget,
            RfPulseTimelineWidget,
            TracePlotWidget,
            WaveformPlotWidget,
        )
    except ImportError:
        from dc_waveform_widgets import (
            RfPulsePreviewWidget,
            RfPulseTimelineWidget,
            TracePlotWidget,
            WaveformPlotWidget,
        )
    MatplotWidget = WaveformPlotWidget
else:
    TracePlotWidget = _MatplotlibTracePlotWidget
    MatplotWidget = _MatplotlibWaveformPlotWidget
    RfPulsePreviewWidget = None
    RfPulseTimelineWidget = None


class ControlPanel(QtWidgets.QWidget): # pylint: disable=too-few-public-methods
    """Input boxes plus a live table of all segments."""

    add_requested       = QtCore.pyqtSignal(float, float, float)
    update_plot         = QtCore.pyqtSignal()
    port_is_selected    = QtCore.pyqtSignal(int)
    sweep_requested     = QtCore.pyqtSignal(int, int)
    sweep_remove_requested = QtCore.pyqtSignal(int, int)
    segment_structure_changed = QtCore.pyqtSignal(int)
    port_idx: int       = 0

    def __init__(
            self,
            pulse: PulseSequence,
            parent=None
        ):
        super().__init__(parent)
        self._pulse         = pulse
        self.idx            = ControlPanel.port_idx
        ControlPanel.port_idx += 1
        self._sweep_row: Optional[int] = None
        self._sweep_rows = set()
        self._sweep_color: Optional[QtGui.QColor] = None

        v_splitter          = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        layout              = QtWidgets.QVBoxLayout(self)

        form                = QtWidgets.QFormLayout()
        self.edit_ramp      = QtWidgets.QLineEdit("50")
        self.edit_flat      = QtWidgets.QLineEdit("200")
        self.edit_v         = QtWidgets.QLineEdit("10")

        for w in (self.edit_ramp, self.edit_flat, self.edit_v):
            w.setValidator(QtGui.QDoubleValidator(decimals=9))

        form.addRow("Ramp [ns]:",  self.edit_ramp)
        form.addRow("Flat [ns]:",  self.edit_flat)
        form.addRow("Virtual V [mV]:", self.edit_v)

        btn_add             = QtWidgets.QPushButton("Add segment")
        btn_add.clicked.connect(self._on_add)
        form.addRow(btn_add)

        btn_select_port     = QtWidgets.QPushButton("Select Port")
        btn_select_port.clicked.connect(self._select_port)
        form.addRow(btn_select_port)

        form_widget         = QtWidgets.QWidget()
        form_widget.setLayout(form)

        self.table          = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["#", "Ramp [ns]", "Flat [ns]", "Virtual V [mV]"]
        )
        self.table.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding
        )
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_menu)

        header              = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
        )
        self.table.itemChanged.connect(self._on_item_changed)

        v_splitter.addWidget(form_widget)
        v_splitter.addWidget(self.table)
        layout.addWidget(v_splitter)
        self.refresh_table()

        layout.setStretch(0, 1)

    def _on_table_menu(self, pos: QtCore.QPoint) -> None:
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        self.table.selectRow(row)
        self.port_is_selected.emit(self.idx)
        menu = QtWidgets.QMenu(self)
        act_ins_above = menu.addAction("Insert segment above")
        act_ins_below = menu.addAction("Insert segment below")
        act_ins_above.setEnabled(row > 0)
        sweep_menu = menu.addMenu("Amplitude sweep")
        act_sweep = sweep_menu.addAction("Configure sweep...")
        act_remove_sweep = sweep_menu.addAction("Remove sweep")
        act_remove_sweep.setEnabled(row in self._sweep_rows)
        menu.addSeparator()
        act_del       = menu.addAction("Delete segment")

        chosen = menu.exec_(self.table.viewport().mapToGlobal(pos))
        if chosen == act_sweep:
            self.sweep_requested.emit(self.idx, row)
            return
        if chosen == act_remove_sweep:
            self.sweep_remove_requested.emit(self.idx, row)
            return
        if chosen == act_ins_above:
            ok = self._pulse.insert_flat_ramp(row * 2)
        elif chosen == act_ins_below:
            ok = self._pulse.insert_flat_ramp(row * 2 + 2)
        elif chosen == act_del:
            ok = self._pulse.delete_flat_ramp(row * 2)
        else:
            return

        if not ok:
            QtWidgets.QMessageBox.warning(
                self,
                "Edit failed",
                "The requested insertion or deletion is not valid for this segment.",
            )
            return

        self.refresh_table()
        self.segment_structure_changed.emit(self.idx)
        self.update_plot.emit()

    def set_sweep_row(
        self,
        row: Optional[int],
        color: Optional[QtGui.QColor] = None,
    ) -> None:
        self.set_sweep_rows(() if row is None else (row,), color)

    def set_sweep_rows(
        self,
        rows,
        color: Optional[QtGui.QColor] = None,
    ) -> None:
        self._sweep_rows = {int(row) for row in rows}
        self._sweep_row = min(self._sweep_rows) if self._sweep_rows else None
        self._sweep_color = QtGui.QColor(color) if color is not None else None
        self.refresh_table()

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        row, col = item.row(), item.column()
        try:
            val = float(item.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Bad edit", "Enter a number")
            return

        # Map table row
        flat_idx = row * 2              # because every table row is (i,i+1) with i even
        if col == 1:
            ok = self._pulse.edit_ramp(flat_idx, val)
        elif col == 2:
            ok = self._pulse.edit_flat(flat_idx, val)
        elif col == 3:
            ok = self._pulse.edit_voltage(flat_idx, val)
        else:
            return
        if not ok:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid edit",
                "Durations must be positive and the first SET has no incoming ramp.",
            )
            self.refresh_table()
            return
        self.refresh_table()
        self.update_plot.emit()

    def _on_add(self):
        self.port_is_selected.emit(self.idx)
        try:
            ramp = float(self.edit_ramp.text())
            flat = float(self.edit_flat.text())
            v    = float(self.edit_v.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Bad input",
                "Please enter valid numbers."
            )
            return
        self.add_requested.emit(ramp, flat, v)

    def refresh_table(self):
        with QtCore.QSignalBlocker(self.table):
            segs = self._pulse.flat_segments()
            self.table.setRowCount(len(segs))
            for row, (i0, i1) in enumerate(segs):
                ramp  = self._pulse.t[i0] - self._pulse.t[i0 - 1] if i0 > 0 else 0
                flat  = self._pulse.t[i1] - self._pulse.t[i0]
                v     = self._pulse.v[i0]
                for col, val in enumerate([row + 1, ramp, flat, v]):
                    item = QtWidgets.QTableWidgetItem(f"{val:.6g}")
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    if row in self._sweep_rows:
                        highlight = QtGui.QColor(
                            self._sweep_color or QtGui.QColor("#f2c14e")
                        )
                        highlight.setAlpha(42)
                        item.setBackground(QtGui.QBrush(highlight))
                        item.setToolTip("Amplitude sweep target")
                    if col == 0 or (col == 1 and row == 0):
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                    self.table.setItem(row, col, item)

    def _select_port(self):
        """Emit signal to select this port."""
        self.port_is_selected.emit(self.idx)

class MultiControlPanel(QtWidgets.QWidget): # pylint: disable=too-few-public-methods
    """Multi waveform control panel"""

    def __init__(
            self,
            pulse: PulseSequence,
            initial_color: str,
            add_port: Callable,
            parent=None
        ):
        super().__init__()
        self._ctrl_pannels: List[ControlPanel]  = [ControlPanel(pulse)]
        self._color_map: List[str]              = [initial_color]
        self.splitter                           = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)

        # Port Add Button
        btn_add_port                            = QtWidgets.QPushButton("Add Port")
        btn_add_port.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding
        )
        btn_widget                              = QtWidgets.QWidget()
        btn_layout                              = QtWidgets.QVBoxLayout(btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_widget.setFixedWidth(btn_add_port.sizeHint().width())
        btn_layout.addWidget(btn_add_port)
        btn_add_port.clicked.connect(add_port)

        self.splitter.addWidget(self._ctrl_pannels[0])
        self.splitter.addWidget(btn_widget)

        # Control Panels list
        self.panel_table                        = QtWidgets.QTableWidget(0, 4)
        self.panel_table.setHorizontalHeaderLabels(["#", "Color", "set_x", "set_y"])
        self.panel_table.verticalHeader().setVisible(False)
        self.panel_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.panel_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        panel_table_header = self.panel_table.horizontalHeader()
        panel_table_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        panel_table_header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        panel_table_header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        panel_table_header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)

        self.btn_reset = QtWidgets.QPushButton("Reset highlight")

        control_panel_v_splitter                = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        control_panel_v_splitter.addWidget(self.splitter)
        control_panel_v_splitter.addWidget(self.panel_table)
        control_panel_v_splitter.addWidget(self.btn_reset)

        layout                                  = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(control_panel_v_splitter)
        self.refresh_table()

    def refresh_table(self):
        """Refresh all control panels' tables."""
        for ctrl in self._ctrl_pannels:
            ctrl.refresh_table()


class GridSettingsDialog(QtWidgets.QDialog):
    """Configure fixed waveform-grid spacing and drag snapping."""

    def __init__(
        self,
        *,
        time_step_ns: float,
        voltage_step_mv: float,
        snap_enabled: bool,
        visible: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Waveform grid settings")
        form = QtWidgets.QFormLayout(self)

        self.time_step_ns = QtWidgets.QDoubleSpinBox()
        self.time_step_ns.setRange(1.0e-6, 1.0e12)
        self.time_step_ns.setDecimals(6)
        self.time_step_ns.setValue(time_step_ns)
        self.time_step_ns.setSuffix(" ns")

        self.voltage_step_mv = QtWidgets.QDoubleSpinBox()
        self.voltage_step_mv.setRange(1.0e-6, 1.0e9)
        self.voltage_step_mv.setDecimals(6)
        self.voltage_step_mv.setValue(voltage_step_mv)
        self.voltage_step_mv.setSuffix(" mV")

        self.snap_enabled = QtWidgets.QCheckBox("Snap dragged points to the grid")
        self.snap_enabled.setChecked(snap_enabled)
        self.visible = QtWidgets.QCheckBox("Show fixed grid lines")
        self.visible.setChecked(visible)
        origin_note = QtWidgets.QLabel(
            "Time and voltage grids are anchored at 0 ns and 0 mV."
        )
        origin_note.setWordWrap(True)

        form.addRow("Time spacing:", self.time_step_ns)
        form.addRow("Voltage spacing:", self.voltage_step_mv)
        form.addRow(self.snap_enabled)
        form.addRow(self.visible)
        form.addRow(origin_note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> Tuple[float, float, bool, bool]:
        return (
            self.time_step_ns.value(),
            self.voltage_step_mv.value(),
            self.snap_enabled.isChecked(),
            self.visible.isChecked(),
        )


class CrossCapacitanceDialog(QtWidgets.QDialog):
    """Edit the virtual-gate to physical-AWG voltage transform."""

    def __init__(self, output_count: int, matrix, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Virtual-to-physical cross-capacitance matrix")
        self.resize(max(620, 115 * output_count), max(330, 62 * output_count))
        values = np.asarray(matrix, dtype=float)
        expected_shape = (output_count, output_count)
        if values.shape != expected_shape:
            raise ValueError(
                f"cross-capacitance matrix shape must be {expected_shape}"
            )

        layout = QtWidgets.QVBoxLayout(self)
        equation = QtWidgets.QLabel(
            "Vphysical = C x Vvirtual (rows: physical AWGs, columns: virtual gates)",
            self,
        )
        equation.setWordWrap(True)
        layout.addWidget(equation)
        self.table = QtWidgets.QTableWidget(output_count, output_count, self)
        self.table.setHorizontalHeaderLabels(
            [f"virtual awg_{index}" for index in range(output_count)]
        )
        self.table.setVerticalHeaderLabels(
            [f"physical awg_{index}" for index in range(output_count)]
        )
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self._editors = []
        for row in range(output_count):
            editor_row = []
            for column in range(output_count):
                editor = QtWidgets.QDoubleSpinBox(self.table)
                editor.setRange(-10.0, 10.0)
                editor.setDecimals(6)
                editor.setSingleStep(0.01)
                editor.setValue(float(values[row, column]))
                if row == column:
                    editor.setValue(1.0)
                    editor.setReadOnly(True)
                    editor.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
                self.table.setCellWidget(row, column, editor)
                editor_row.append(editor)
            self._editors.append(editor_row)
        layout.addWidget(self.table)

        button_row = QtWidgets.QHBoxLayout()
        reset_button = QtWidgets.QPushButton("Reset identity")
        reset_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)
        )
        reset_button.clicked.connect(self._reset_identity)
        button_row.addWidget(reset_button)
        button_row.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        button_row.addWidget(buttons)
        layout.addLayout(button_row)

    def _reset_identity(self) -> None:
        for row, editors in enumerate(self._editors):
            for column, editor in enumerate(editors):
                editor.setValue(1.0 if row == column else 0.0)

    def matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [editor.value() for editor in editor_row]
                for editor_row in self._editors
            ],
            dtype=float,
        )


class SweepSettingsDialog(QtWidgets.QDialog):
    """Configure the amplitude sweep attached to one SET/output pair."""

    def __init__(
        self,
        *,
        output_name: str,
        segment_name: str,
        current_amplitude: float,
        full_scale_mv: float,
        initial: Optional[QickSweepSpec] = None,
        cartesian_base_count: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Amplitude sweep settings")
        self._output_name = output_name
        self._segment_name = segment_name
        self._full_scale_mv = float(full_scale_mv)
        self._cartesian_base_count = int(cartesian_base_count)
        form = QtWidgets.QFormLayout(self)

        form.addRow("Output:", QtWidgets.QLabel(output_name))
        form.addRow("SET segment:", QtWidgets.QLabel(segment_name))
        form.addRow(
            "Current level:",
            QtWidgets.QLabel(
                f"{current_amplitude:.6g} normalized "
                f"({current_amplitude * self._full_scale_mv:.6g} mV)"
            ),
        )

        default_start = max(-1.0, current_amplitude - 0.2)
        default_stop = min(1.0, current_amplitude + 0.2)
        if initial is not None:
            default_start = initial.start
            default_stop = initial.stop

        self.start = QtWidgets.QDoubleSpinBox()
        self.stop = QtWidgets.QDoubleSpinBox()
        for widget, value in ((self.start, default_start), (self.stop, default_stop)):
            widget.setRange(-1.0, 1.0)
            widget.setDecimals(6)
            widget.setSingleStep(0.05)
            widget.setValue(value)
        self.count = QtWidgets.QSpinBox()
        self.count.setRange(1, 1_000_000)
        self.count.setValue(initial.count if initial is not None else 9)
        self.endpoint_summary = QtWidgets.QLabel()
        self.cartesian_summary = QtWidgets.QLabel()

        form.addRow("Start [-1, 1]:", self.start)
        form.addRow("Stop [-1, 1]:", self.stop)
        form.addRow("Sweep point count:", self.count)
        form.addRow("Endpoint levels:", self.endpoint_summary)
        form.addRow("Cartesian total:", self.cartesian_summary)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.start.valueChanged.connect(self._refresh_summary)
        self.stop.valueChanged.connect(self._refresh_summary)
        self.count.valueChanged.connect(self._refresh_summary)
        self._refresh_summary()

    def _refresh_summary(self, *_args) -> None:
        self.endpoint_summary.setText(
            f"{self.start.value() * self._full_scale_mv:.6g} mV to "
            f"{self.stop.value() * self._full_scale_mv:.6g} mV"
        )
        self.cartesian_summary.setText(
            f"{self._cartesian_base_count} x {self.count.value()} = "
            f"{self._cartesian_base_count * self.count.value()} points"
        )

    def value(self) -> QickSweepSpec:
        return QickSweepSpec(
            segment_name=self._segment_name,
            output_name=self._output_name,
            start=self.start.value(),
            stop=self.stop.value(),
            count=self.count.value(),
        )


class RfPulseEditorPanel(QtWidgets.QWidget):
    """Main-window editor and waveform preview for one QICK RF pulse."""

    spec_applied = QtCore.pyqtSignal(object)
    spec_removed = QtCore.pyqtSignal()

    def __init__(self, pulse: PulseSequence, parent=None):
        super().__init__(parent)
        self._pulse = pulse
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        controls = QtWidgets.QWidget(splitter)
        form = QtWidgets.QFormLayout(controls)
        self.gen_ch = QtWidgets.QSpinBox()
        self.gen_ch.setRange(0, 255)
        self.segment = QtWidgets.QComboBox()
        self.delay_us = QtWidgets.QDoubleSpinBox()
        self.delay_us.setRange(0.0, 1.0e9)
        self.delay_us.setDecimals(6)
        self.delay_us.setSuffix(" us")
        self.duration_us = QtWidgets.QDoubleSpinBox()
        self.duration_us.setRange(0.001, 1.0e9)
        self.duration_us.setDecimals(6)
        self.duration_us.setValue(0.05)
        self.duration_us.setSuffix(" us")
        self.frequency_mhz = QtWidgets.QDoubleSpinBox()
        self.frequency_mhz.setRange(-10000.0, 10000.0)
        self.frequency_mhz.setDecimals(6)
        self.frequency_mhz.setValue(50.0)
        self.frequency_mhz.setSuffix(" MHz")
        self.gain = QtWidgets.QSpinBox()
        self.gain.setRange(-32768, 32767)
        self.gain.setValue(20000)
        self.att1_db = QtWidgets.QDoubleSpinBox()
        self.att2_db = QtWidgets.QDoubleSpinBox()
        for attenuator in (self.att1_db, self.att2_db):
            attenuator.setRange(0.0, 31.75)
            attenuator.setDecimals(2)
            attenuator.setSingleStep(0.25)
            attenuator.setSuffix(" dB")
        self.phase_degrees = QtWidgets.QDoubleSpinBox()
        self.phase_degrees.setRange(-360.0, 360.0)
        self.phase_degrees.setDecimals(6)
        self.phase_degrees.setSuffix(" deg")
        self.nqz = QtWidgets.QSpinBox()
        self.nqz.setRange(1, 3)
        self.nqz.setValue(1)
        self.require_within = QtWidgets.QCheckBox("Require pulse inside selected SET")
        self.require_within.setChecked(True)
        self.absolute_time = QtWidgets.QLabel()

        form.addRow("RF generator index:", self.gen_ch)
        form.addRow("Anchor SET segment:", self.segment)
        form.addRow("Start delay:", self.delay_us)
        form.addRow("Duration:", self.duration_us)
        form.addRow("Frequency:", self.frequency_mhz)
        form.addRow("Gain:", self.gain)
        form.addRow("ATT1:", self.att1_db)
        form.addRow("ATT2:", self.att2_db)
        form.addRow("Phase:", self.phase_degrees)
        form.addRow("Nyquist zone:", self.nqz)
        form.addRow(self.require_within)
        form.addRow("Absolute timing:", self.absolute_time)

        button_row = QtWidgets.QHBoxLayout()
        apply_button = QtWidgets.QPushButton("Apply RF pulse")
        apply_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton))
        clear_button = QtWidgets.QPushButton("Remove RF pulse")
        clear_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton))
        button_row.addWidget(apply_button)
        button_row.addWidget(clear_button)
        form.addRow(button_row)
        apply_button.clicked.connect(self._apply)
        clear_button.clicked.connect(self._remove)

        if RfPulsePreviewWidget is None:
            self.preview = QtWidgets.QLabel("Install pyqtgraph to preview RF pulses.")
            self.preview.setAlignment(QtCore.Qt.AlignCenter)
        else:
            self.preview = RfPulsePreviewWidget(splitter)
        splitter.addWidget(controls)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.refresh_segments(pulse)
        for widget in (
            self.gen_ch,
            self.segment,
            self.delay_us,
            self.duration_us,
            self.frequency_mhz,
            self.gain,
            self.att1_db,
            self.att2_db,
            self.phase_degrees,
            self.nqz,
            self.require_within,
        ):
            signal = getattr(widget, "valueChanged", None)
            if signal is None:
                signal = getattr(widget, "currentIndexChanged", None)
            if signal is None:
                signal = getattr(widget, "toggled", None)
            signal.connect(self._preview_current)
        self._preview_current()

    def refresh_segments(self, pulse: Optional[PulseSequence] = None) -> None:
        if pulse is not None:
            self._pulse = pulse
        previous = self.segment.currentData()
        self.segment.blockSignals(True)
        self.segment.clear()
        durations = []
        for index, (start, end) in enumerate(self._pulse.flat_segments()):
            name = f"set_{index}"
            start_us = float(self._pulse.t[start]) / 1000.0
            end_us = float(self._pulse.t[end]) / 1000.0
            durations.append(end_us - start_us)
            self.segment.addItem(f"{name}  [{start_us:.6g}, {end_us:.6g}] us", name)
        if previous is not None:
            match = self.segment.findData(previous)
            if match >= 0:
                self.segment.setCurrentIndex(match)
        elif durations:
            self.segment.setCurrentIndex(int(np.argmax(durations)))
        self.segment.blockSignals(False)
        self._preview_current()

    def current_spec(self) -> QickRfPulseSpec:
        return QickRfPulseSpec(
            gen_ch=self.gen_ch.value(),
            segment_name=str(self.segment.currentData()),
            delay_us=self.delay_us.value(),
            duration_us=self.duration_us.value(),
            frequency_mhz=self.frequency_mhz.value(),
            gain=self.gain.value(),
            att1_db=self.att1_db.value(),
            att2_db=self.att2_db.value(),
            phase_degrees=self.phase_degrees.value(),
            nqz=self.nqz.value(),
            require_within_segment=self.require_within.isChecked(),
        )

    def _absolute_times(self, spec: QickRfPulseSpec) -> Tuple[float, float, float]:
        return rf_pulse_absolute_times_us(self._pulse, spec)

    def _preview_current(self, *_args) -> None:
        try:
            spec = self.current_spec()
            start_us, end_us, set_end_us = self._absolute_times(spec)
        except (IndexError, TypeError, ValueError):
            self.absolute_time.setText("Invalid")
            if hasattr(self.preview, "clear_preview"):
                self.preview.clear_preview()
            return
        self.absolute_time.setText(
            f"{start_us:.6g} to {end_us:.6g} us (SET ends {set_end_us:.6g} us)"
        )
        if hasattr(self.preview, "set_pulse"):
            self.preview.set_pulse(
                start_us=start_us,
                duration_us=spec.duration_us,
                frequency_mhz=spec.frequency_mhz,
                gain=spec.gain,
                phase_degrees=spec.phase_degrees,
                att1_db=spec.att1_db,
                att2_db=spec.att2_db,
            )

    def _apply(self) -> None:
        try:
            spec = self.current_spec()
            self._absolute_times(spec)
        except (IndexError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid RF pulse", str(exc))
            return
        self.spec_applied.emit(spec)

    def _remove(self) -> None:
        if hasattr(self.preview, "clear_preview"):
            self.preview.clear_preview()
        self.absolute_time.setText("Disabled")
        self.spec_removed.emit()

    def load_spec(self, spec: Optional[QickRfPulseSpec]) -> None:
        if spec is None:
            self._remove()
            return
        self.gen_ch.setValue(spec.gen_ch)
        segment_index = self.segment.findData(spec.segment_name)
        if segment_index >= 0:
            self.segment.setCurrentIndex(segment_index)
        self.delay_us.setValue(spec.delay_us)
        self.duration_us.setValue(spec.duration_us)
        self.frequency_mhz.setValue(spec.frequency_mhz)
        self.gain.setValue(spec.gain)
        self.att1_db.setValue(spec.att1_db)
        self.att2_db.setValue(spec.att2_db)
        self.phase_degrees.setValue(spec.phase_degrees)
        self.nqz.setValue(spec.nqz)
        self.require_within.setChecked(spec.require_within_segment)
        self._preview_current()


class QickExportDialog(QtWidgets.QDialog):
    """Collect QICK timing, channel-map, repetition, and sweep settings."""

    def __init__(
        self,
        pulse_count: int,
        set_names: Tuple[str, ...],
        parent=None,
        default_set_index: int = 0,
        initial_rf_spec: Optional[QickRfPulseSpec] = None,
        initial_sweep: Optional[QickSweepSpec] = None,
        initial_sweeps: Optional[Sequence[QickSweepSpec]] = None,
        initial_cross_capacitance=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("QICK export settings")
        self.resize(660, 780)
        self._pulse_count = pulse_count
        if initial_sweep is not None and initial_sweeps is not None:
            raise ValueError("use either initial_sweep or initial_sweeps, not both")
        self._dialog_sweep_specs = list(
            initial_sweeps
            if initial_sweeps is not None
            else (() if initial_sweep is None else (initial_sweep,))
        )
        self._cross_capacitance = np.asarray(
            np.eye(pulse_count)
            if initial_cross_capacitance is None
            else initial_cross_capacitance,
            dtype=float,
        ).copy()
        if self._cross_capacitance.shape != (pulse_count, pulse_count):
            raise ValueError("cross-capacitance matrix size must match pulse count")

        outer_layout = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget(scroll)
        form = QtWidgets.QFormLayout(content)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        self.fabric_mhz = QtWidgets.QDoubleSpinBox()
        self.fabric_mhz.setRange(1.0, 5000.0)
        self.fabric_mhz.setDecimals(6)
        self.fabric_mhz.setValue(DEFAULT_QICK_FABRIC_MHZ)
        self.fabric_mhz.setSuffix(" MHz")

        self.full_scale_mv = QtWidgets.QDoubleSpinBox()
        self.full_scale_mv.setRange(1.0, 1.0e6)
        self.full_scale_mv.setDecimals(6)
        self.full_scale_mv.setValue(DEFAULT_QICK_FULL_SCALE_MV)
        self.full_scale_mv.setSuffix(" mV")

        self.awg_channels = QtWidgets.QLineEdit(
            ", ".join(str(index) for index in DEFAULT_QSTL_AWG_CHANNELS[:pulse_count])
        )
        self.repetitions = QtWidgets.QSpinBox()
        self.repetitions.setRange(1, 1_000_000)
        self.repetitions.setValue(1)

        form.addRow("AWG fabric clock:", self.fabric_mhz)
        form.addRow("QICK full scale (+/-):", self.full_scale_mv)
        form.addRow("AWG generator indices:", self.awg_channels)
        form.addRow("Repetitions per sweep:", self.repetitions)

        sweep_group = QtWidgets.QGroupBox("Independent Cartesian amplitude sweeps")
        sweep_group.setCheckable(True)
        sweep_group.setChecked(False)
        sweep_form = QtWidgets.QFormLayout(sweep_group)
        self.sweep_group = sweep_group
        self.sweep_output = QtWidgets.QComboBox()
        self.sweep_output.addItems([f"awg_{index}" for index in range(pulse_count)])
        self.sweep_segment = QtWidgets.QComboBox()
        self.sweep_segment.addItems(list(set_names))
        self.sweep_start = QtWidgets.QDoubleSpinBox()
        self.sweep_stop = QtWidgets.QDoubleSpinBox()
        for widget, value in ((self.sweep_start, -0.2), (self.sweep_stop, 0.8)):
            widget.setRange(-1.0, 1.0)
            widget.setDecimals(6)
            widget.setSingleStep(0.05)
            widget.setValue(value)
        self.sweep_count = QtWidgets.QSpinBox()
        self.sweep_count.setRange(1, 1_000_000)
        self.sweep_count.setValue(9)
        sweep_form.addRow("Output:", self.sweep_output)
        sweep_form.addRow("SET segment:", self.sweep_segment)
        sweep_form.addRow("Start [-1, 1]:", self.sweep_start)
        sweep_form.addRow("Stop [-1, 1]:", self.sweep_stop)
        sweep_form.addRow("Point count:", self.sweep_count)
        self.sweep_table = QtWidgets.QTableWidget(0, 5)
        self.sweep_table.setHorizontalHeaderLabels(
            ["Output", "SET", "Start", "Stop", "Count"]
        )
        self.sweep_table.verticalHeader().setVisible(False)
        self.sweep_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.sweep_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.sweep_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.sweep_table.setMaximumHeight(140)
        sweep_header = self.sweep_table.horizontalHeader()
        sweep_header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        sweep_form.addRow("Active axes:", self.sweep_table)
        sweep_buttons = QtWidgets.QHBoxLayout()
        add_sweep = QtWidgets.QPushButton("Add/update axis")
        remove_sweep = QtWidgets.QPushButton("Remove selected")
        add_sweep.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton))
        remove_sweep.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton))
        sweep_buttons.addWidget(add_sweep)
        sweep_buttons.addWidget(remove_sweep)
        sweep_form.addRow(sweep_buttons)
        self.sweep_total = QtWidgets.QLabel()
        sweep_form.addRow("Cartesian total:", self.sweep_total)
        form.addRow(sweep_group)
        add_sweep.clicked.connect(self._upsert_current_sweep)
        remove_sweep.clicked.connect(self._remove_selected_sweep)
        self.sweep_table.itemSelectionChanged.connect(self._load_selected_sweep)
        self.sweep_count.valueChanged.connect(self._refresh_sweep_total)
        self.sweep_group.toggled.connect(self._refresh_sweep_total)
        if self._dialog_sweep_specs:
            self.sweep_group.setChecked(True)
        self._refresh_sweep_table(select_row=0 if self._dialog_sweep_specs else None)

        cross_row = QtWidgets.QHBoxLayout()
        self.cross_capacitance_summary = QtWidgets.QLabel()
        edit_cross_capacitance = QtWidgets.QPushButton("Edit matrix...")
        edit_cross_capacitance.clicked.connect(self._edit_cross_capacitance)
        cross_row.addWidget(self.cross_capacitance_summary)
        cross_row.addStretch(1)
        cross_row.addWidget(edit_cross_capacitance)
        form.addRow("Virtual-to-physical matrix:", cross_row)
        self._refresh_cross_capacitance_summary()

        self.rf_group = QtWidgets.QGroupBox("RF pulse and output attenuation")
        self.rf_group.setCheckable(True)
        self.rf_group.setChecked(True)
        rf_form = QtWidgets.QFormLayout(self.rf_group)
        self.rf_gen_ch = QtWidgets.QSpinBox()
        self.rf_gen_ch.setRange(0, 255)
        self.rf_gen_ch.setValue(0)
        self.rf_segment = QtWidgets.QComboBox()
        self.rf_segment.addItems(list(set_names))
        self.rf_segment.setCurrentIndex(min(default_set_index, len(set_names) - 1))
        self.rf_delay_us = QtWidgets.QDoubleSpinBox()
        self.rf_delay_us.setRange(0.0, 1.0e9)
        self.rf_delay_us.setDecimals(6)
        self.rf_delay_us.setSuffix(" us")
        self.rf_duration_us = QtWidgets.QDoubleSpinBox()
        self.rf_duration_us.setRange(0.001, 1.0e9)
        self.rf_duration_us.setDecimals(6)
        self.rf_duration_us.setValue(0.05)
        self.rf_duration_us.setSuffix(" us")
        self.rf_frequency_mhz = QtWidgets.QDoubleSpinBox()
        self.rf_frequency_mhz.setRange(-10000.0, 10000.0)
        self.rf_frequency_mhz.setDecimals(6)
        self.rf_frequency_mhz.setValue(50.0)
        self.rf_frequency_mhz.setSuffix(" MHz")
        self.rf_gain = QtWidgets.QSpinBox()
        self.rf_gain.setRange(-32768, 32767)
        self.rf_gain.setValue(20000)
        self.rf_att1_db = QtWidgets.QDoubleSpinBox()
        self.rf_att2_db = QtWidgets.QDoubleSpinBox()
        for attenuator in (self.rf_att1_db, self.rf_att2_db):
            attenuator.setRange(0.0, 31.75)
            attenuator.setDecimals(2)
            attenuator.setSingleStep(0.25)
            attenuator.setSuffix(" dB")
        self.rf_phase_degrees = QtWidgets.QDoubleSpinBox()
        self.rf_phase_degrees.setRange(-360.0, 360.0)
        self.rf_phase_degrees.setDecimals(6)
        self.rf_phase_degrees.setSuffix(" deg")
        self.rf_nqz = QtWidgets.QSpinBox()
        self.rf_nqz.setRange(1, 3)
        self.rf_nqz.setValue(1)
        self.rf_require_within = QtWidgets.QCheckBox("Require pulse to finish inside SET")
        self.rf_require_within.setChecked(True)
        rf_form.addRow("RF generator index:", self.rf_gen_ch)
        rf_form.addRow("Anchor SET segment:", self.rf_segment)
        rf_form.addRow("Start delay from SET:", self.rf_delay_us)
        rf_form.addRow("RF duration:", self.rf_duration_us)
        rf_form.addRow("RF frequency:", self.rf_frequency_mhz)
        rf_form.addRow("RF gain:", self.rf_gain)
        rf_form.addRow("Output ATT1:", self.rf_att1_db)
        rf_form.addRow("Output ATT2:", self.rf_att2_db)
        rf_form.addRow("RF phase:", self.rf_phase_degrees)
        rf_form.addRow("Nyquist zone:", self.rf_nqz)
        rf_form.addRow(self.rf_require_within)
        form.addRow(self.rf_group)

        self.ddr_group = QtWidgets.QGroupBox("FIR DDR input capture (fixed 1 MSPS)")
        self.ddr_group.setCheckable(True)
        self.ddr_group.setChecked(True)
        ddr_form = QtWidgets.QFormLayout(self.ddr_group)
        self.ddr_ro_ch = QtWidgets.QSpinBox()
        self.ddr_ro_ch.setRange(0, 255)
        self.ddr_segment = QtWidgets.QComboBox()
        self.ddr_segment.addItems(list(set_names))
        self.ddr_segment.setCurrentIndex(min(default_set_index, len(set_names) - 1))
        self.ddr_delay_us = QtWidgets.QDoubleSpinBox()
        self.ddr_delay_us.setRange(0.0, 1.0e9)
        self.ddr_delay_us.setDecimals(6)
        self.ddr_delay_us.setSuffix(" us")
        self.ddr_samples = QtWidgets.QSpinBox()
        self.ddr_samples.setRange(1, 10_000_000)
        self.ddr_samples.setValue(64)
        self.ddr_samples.setSuffix(" samples")
        self.ddr_readout_frequency_mhz = QtWidgets.QDoubleSpinBox()
        self.ddr_readout_frequency_mhz.setRange(-10000.0, 10000.0)
        self.ddr_readout_frequency_mhz.setDecimals(6)
        self.ddr_readout_frequency_mhz.setValue(50.0)
        self.ddr_readout_frequency_mhz.setSuffix(" MHz")
        self.ddr_margin_samples = QtWidgets.QSpinBox()
        self.ddr_margin_samples.setRange(0, 10_000_000)
        self.ddr_margin_samples.setValue(1024)
        self.ddr_force_overwrite = QtWidgets.QCheckBox("Allow overwrite of reserved DDR range")
        ddr_form.addRow("Readout channel index:", self.ddr_ro_ch)
        ddr_form.addRow("Trigger SET segment:", self.ddr_segment)
        ddr_form.addRow("Trigger delay from SET:", self.ddr_delay_us)
        ddr_form.addRow("Stored 1 MSPS samples:", self.ddr_samples)
        ddr_form.addRow("Readout/DDC frequency:", self.ddr_readout_frequency_mhz)
        ddr_form.addRow("FIR input margin:", self.ddr_margin_samples)
        ddr_form.addRow(self.ddr_force_overwrite)
        ddr_note = QtWidgets.QLabel(
            "At 1 MSPS, N stored samples represent N microseconds. "
            "FIR warmup and 300:1 decimation are handled automatically."
        )
        ddr_note.setWordWrap(True)
        ddr_form.addRow(ddr_note)
        form.addRow(self.ddr_group)

        if initial_rf_spec is not None:
            self.rf_group.setChecked(True)
            self.rf_gen_ch.setValue(initial_rf_spec.gen_ch)
            segment_index = self.rf_segment.findText(initial_rf_spec.segment_name)
            if segment_index >= 0:
                self.rf_segment.setCurrentIndex(segment_index)
            self.rf_delay_us.setValue(initial_rf_spec.delay_us)
            self.rf_duration_us.setValue(initial_rf_spec.duration_us)
            self.rf_frequency_mhz.setValue(initial_rf_spec.frequency_mhz)
            self.rf_gain.setValue(initial_rf_spec.gain)
            self.rf_att1_db.setValue(initial_rf_spec.att1_db)
            self.rf_att2_db.setValue(initial_rf_spec.att2_db)
            self.rf_phase_degrees.setValue(initial_rf_spec.phase_degrees)
            self.rf_nqz.setValue(initial_rf_spec.nqz)
            self.rf_require_within.setChecked(initial_rf_spec.require_within_segment)

        note = QtWidgets.QLabel(
            "All exported ports must have identical SET/RAMP timing. "
            "Voltages are normalized by the configured full scale."
        )
        note.setWordWrap(True)
        form.addRow(note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _current_sweep_spec(self) -> QickSweepSpec:
        return QickSweepSpec(
            segment_name=self.sweep_segment.currentText(),
            output_name=self.sweep_output.currentText(),
            start=self.sweep_start.value(),
            stop=self.sweep_stop.value(),
            count=self.sweep_count.value(),
        )

    def _refresh_sweep_table(self, select_row: Optional[int] = None) -> None:
        with QtCore.QSignalBlocker(self.sweep_table):
            self.sweep_table.setRowCount(len(self._dialog_sweep_specs))
            for row, spec in enumerate(self._dialog_sweep_specs):
                values = (
                    spec.output_name,
                    spec.segment_name,
                    f"{spec.start:.6g}",
                    f"{spec.stop:.6g}",
                    str(spec.count),
                )
                for column, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.sweep_table.setItem(row, column, item)
            if select_row is not None and self._dialog_sweep_specs:
                self.sweep_table.selectRow(
                    min(max(0, select_row), len(self._dialog_sweep_specs) - 1)
                )
        if select_row is not None and self._dialog_sweep_specs:
            self._load_selected_sweep()
        self._refresh_sweep_total()

    def _load_selected_sweep(self) -> None:
        row = self.sweep_table.currentRow()
        if row < 0 or row >= len(self._dialog_sweep_specs):
            return
        spec = self._dialog_sweep_specs[row]
        output_index = self.sweep_output.findText(spec.output_name)
        segment_index = self.sweep_segment.findText(spec.segment_name)
        if output_index >= 0:
            self.sweep_output.setCurrentIndex(output_index)
        if segment_index >= 0:
            self.sweep_segment.setCurrentIndex(segment_index)
        self.sweep_start.setValue(spec.start)
        self.sweep_stop.setValue(spec.stop)
        self.sweep_count.setValue(spec.count)

    def _upsert_current_sweep(self) -> None:
        spec = self._current_sweep_spec()
        target = (spec.segment_name, spec.output_name)
        for row, current in enumerate(self._dialog_sweep_specs):
            if (current.segment_name, current.output_name) == target:
                self._dialog_sweep_specs[row] = spec
                self._refresh_sweep_table(select_row=row)
                return
        self._dialog_sweep_specs.append(spec)
        self._refresh_sweep_table(select_row=len(self._dialog_sweep_specs) - 1)

    def _remove_selected_sweep(self) -> None:
        row = self.sweep_table.currentRow()
        if row < 0 or row >= len(self._dialog_sweep_specs):
            return
        self._dialog_sweep_specs.pop(row)
        self._refresh_sweep_table(
            select_row=(
                min(row, len(self._dialog_sweep_specs) - 1)
                if self._dialog_sweep_specs
                else None
            )
        )

    def _effective_sweeps(self) -> Tuple[QickSweepSpec, ...]:
        if not self.sweep_group.isChecked():
            return ()
        if not self._dialog_sweep_specs:
            return (self._current_sweep_spec(),)
        specs = list(self._dialog_sweep_specs)
        row = self.sweep_table.currentRow()
        if 0 <= row < len(specs):
            specs[row] = self._current_sweep_spec()
        targets = [(spec.segment_name, spec.output_name) for spec in specs]
        if len(set(targets)) != len(targets):
            raise ValueError("each Cartesian sweep axis must target a unique output/SET")
        return tuple(specs)

    def _refresh_sweep_total(self, *_args) -> None:
        if not self.sweep_group.isChecked():
            self.sweep_total.setText("disabled")
            return
        counts = [spec.count for spec in self._dialog_sweep_specs]
        row = self.sweep_table.currentRow()
        if 0 <= row < len(counts):
            counts[row] = self.sweep_count.value()
        elif not counts:
            counts = [self.sweep_count.value()]
        self.sweep_total.setText(
            " x ".join(str(count) for count in counts)
            + f" = {prod(counts)} combinations"
        )

    def _edit_cross_capacitance(self) -> None:
        dialog = CrossCapacitanceDialog(
            self._pulse_count,
            self._cross_capacitance,
            self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        self._cross_capacitance = dialog.matrix()
        self._refresh_cross_capacitance_summary()

    def _refresh_cross_capacitance_summary(self) -> None:
        off_diagonal = self._cross_capacitance.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        nonzero = int(np.count_nonzero(np.abs(off_diagonal) > 1.0e-15))
        self.cross_capacitance_summary.setText(
            f"{self._pulse_count} x {self._pulse_count}, {nonzero} active terms"
        )

    def _parse_channels(self) -> Tuple[int, ...]:
        fields = [field.strip() for field in self.awg_channels.text().split(",")]
        if any(not field for field in fields):
            raise ValueError("AWG generator indices must be comma-separated integers")
        try:
            channels = tuple(int(field) for field in fields)
        except ValueError as exc:
            raise ValueError("AWG generator indices must be integers") from exc
        if len(channels) != self._pulse_count:
            raise ValueError(f"exactly {self._pulse_count} AWG generator indices are required")
        if len(set(channels)) != len(channels) or any(channel < 0 for channel in channels):
            raise ValueError("AWG generator indices must be unique and nonnegative")
        return channels

    def values(self) -> dict:
        awg_channels = self._parse_channels()
        sweep_specs = self._effective_sweeps()
        sweep = sweep_specs[0] if len(sweep_specs) == 1 else None
        sweeps = sweep_specs if len(sweep_specs) > 1 else None
        rf_pulse_spec = None
        if self.rf_group.isChecked():
            if self.rf_gen_ch.value() in awg_channels:
                raise ValueError("RF generator index must differ from all AWG tuning indices")
            rf_pulse_spec = QickRfPulseSpec(
                gen_ch=self.rf_gen_ch.value(),
                segment_name=self.rf_segment.currentText(),
                delay_us=self.rf_delay_us.value(),
                duration_us=self.rf_duration_us.value(),
                frequency_mhz=self.rf_frequency_mhz.value(),
                gain=self.rf_gain.value(),
                att1_db=self.rf_att1_db.value(),
                att2_db=self.rf_att2_db.value(),
                phase_degrees=self.rf_phase_degrees.value(),
                nqz=self.rf_nqz.value(),
                require_within_segment=self.rf_require_within.isChecked(),
            )
        ddr_readout_spec = None
        if self.ddr_group.isChecked():
            ddr_readout_spec = QickDdrReadoutSpec(
                ro_ch=self.ddr_ro_ch.value(),
                segment_name=self.ddr_segment.currentText(),
                delay_us=self.ddr_delay_us.value(),
                samples_per_trigger=self.ddr_samples.value(),
                readout_frequency_mhz=self.ddr_readout_frequency_mhz.value(),
                margin_input_samples=self.ddr_margin_samples.value(),
                force_overwrite=self.ddr_force_overwrite.isChecked(),
            )
        return {
            "fabric_mhz": self.fabric_mhz.value(),
            "full_scale_mv": self.full_scale_mv.value(),
            "awg_channels": awg_channels,
            "repetitions_per_sweep": self.repetitions.value(),
            "sweep": sweep,
            "sweeps": sweeps,
            "cross_capacitance": tuple(
                tuple(float(value) for value in row)
                for row in self._cross_capacitance
            ),
            "rf_pulse_spec": rf_pulse_spec,
            "ddr_readout_spec": ddr_readout_spec,
        }

    def accept(self) -> None:
        try:
            self.values()
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid QICK settings", str(exc))
            return
        super().accept()


class MainWindow(QtWidgets.QMainWindow): # pylint: disable=too-few-public-methods
    """Main window for the DCWaveform generator application."""

    @property
    def _sweep_spec(self) -> Optional[QickSweepSpec]:
        """Legacy single-sweep view used by older tests and callers."""
        return self._sweep_specs[0] if self._sweep_specs else None

    @_sweep_spec.setter
    def _sweep_spec(self, value: Optional[QickSweepSpec]) -> None:
        self._sweep_specs = [] if value is None else [value]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DC Waveform Generator - QCS and QICK")

        self._pulse: List[PulseSequence]    = [PulseSequence(0.0)]
        self._pulse[0].v_bounds             = (-2500, 2500)
        self._rf_pulse_spec: Optional[QickRfPulseSpec] = None
        self._rf_panel: Optional[RfPulseEditorPanel] = None
        self._dock_rf: Optional[QtWidgets.QDockWidget] = None
        self._sweep_specs: List[QickSweepSpec] = []
        self._cross_capacitance = np.eye(1, dtype=float)
        self._qick_full_scale_mv = float(DEFAULT_QICK_FULL_SCALE_MV)
        self._grid_time_ns = 10.0
        self._grid_voltage_mv = 10.0
        self._grid_snap_enabled = False
        self._grid_visible = True
        self._grid_configured = False
        self._plot                          = MatplotWidget(self._pulse[0])
        self._selected_port_idx             = 0
        initial_color = (
            self._plot.line_color(0)
            if hasattr(self._plot, "line_color")
            else self._plot._line[0].get_color()
        )
        ControlPanel.port_idx = 0
        self._multi_ctrl: MultiControlPanel = MultiControlPanel(
            self._pulse[0],
            initial_color,
            self._add_port
        )
        self.refresh_panel_table()
        self._wire_control_panel(self._multi_ctrl._ctrl_pannels[0])
        self._multi_ctrl.panel_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self._multi_ctrl.panel_table.customContextMenuRequested.connect(self._on_port_menu)

        # Create the trace plot only after both X and Y are requested.
        self._trace: Optional[TracePlotWidget] = None
        self._trace_placeholder = QtWidgets.QLabel(
            "Select set_x and set_y ports to create the X/Y trace plot."
        )
        self._trace_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._trace_placeholder.setWordWrap(True)

        self._rf_timeline = (
            RfPulseTimelineWidget(self)
            if RfPulseTimelineWidget is not None
            else None
        )
        self._waveform_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        self._waveform_splitter.addWidget(self._plot)
        self._waveform_splitter.setCollapsible(0, False)
        self._waveform_splitter.setStretchFactor(0, 4)
        if self._rf_timeline is not None:
            self._waveform_splitter.addWidget(self._rf_timeline)
            self._waveform_splitter.setCollapsible(1, True)
            self._waveform_splitter.setStretchFactor(1, 1)
            self._rf_timeline.getPlotItem().getViewBox().setXLink(
                self._plot.getPlotItem().getViewBox()
            )
            self._rf_timeline.hide()

        self._dock_ctrl  = QtWidgets.QDockWidget("Control Panel", self)
        self._dock_ctrl.setWidget(self._multi_ctrl)

        self._dock_plot = QtWidgets.QDockWidget(
            "Waveform Plot - Virtual (solid) / Physical (dashed)",
            self,
        )
        self._dock_plot.setWidget(self._waveform_splitter)

        self._dock_trace = QtWidgets.QDockWidget("Trace Plot", self)
        self._dock_trace.setWidget(self._trace_placeholder)

        for dock in (self._dock_ctrl, self._dock_plot, self._dock_trace):
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable |
                QtWidgets.QDockWidget.DockWidgetFloatable
            )
        self._multi_ctrl.btn_reset.clicked.connect(self._plot._restore_full_intensity)

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._dock_ctrl)
        self.splitDockWidget(self._dock_ctrl, self._dock_plot, QtCore.Qt.Horizontal)
        self.splitDockWidget(self._dock_plot, self._dock_trace, QtCore.Qt.Horizontal)
        self.resizeDocks(
            [self._dock_ctrl, self._dock_plot, self._dock_trace],
            [200, 400, 300],
            QtCore.Qt.Horizontal
        )
        self._dock_plot.raise_()

        # status bar
        self.statusBar().showMessage("Ready")

        self._plot.flat_moved.connect(self._flat_update)
        self._plot.point_moved.connect(self._point_update)

        # The curve moves immediately in PyQtGraph.  Table and X/Y trace
        # updates are coalesced to roughly 30 Hz while the mouse is moving.
        self._pending_table_refresh = False
        self._pending_trace_refresh = False
        self._pending_rf_refresh = False
        self._pending_sweep_refresh = False
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(33)
        self._refresh_timer.timeout.connect(self._flush_deferred_refresh)

        self._build_menu()
        self._build_toolbar()
        self._refresh_sweep_overlay()

    def _wire_control_panel(self, control: ControlPanel) -> None:
        control.add_requested.connect(self._add_segment)
        control.update_plot.connect(self._plot_refresh)
        control.port_is_selected.connect(self._port_select)
        control.sweep_requested.connect(self._configure_segment_sweep)
        control.sweep_remove_requested.connect(self._remove_segment_sweep)
        control.segment_structure_changed.connect(self._on_segment_structure_changed)

    def _on_port_menu(self, pos: QtCore.QPoint) -> None:
        row = self._multi_ctrl.panel_table.rowAt(pos.y())
        if row < 0:
            return
        menu = QtWidgets.QMenu(self)
        act_del = menu.addAction("Delete port")
        if len(self._pulse) <= 1:
            act_del.setEnabled(False)     # must keep at least one
        if menu.exec_(
            self._multi_ctrl.panel_table.viewport().mapToGlobal(pos)
        ) == act_del:
            self._delete_port(row)

    def _delete_port(self, idx: int) -> None:
        """Completely remove pulse, plot line and control panel at index idx."""
        sweep_entries = [
            (spec, self._sweep_target_indices(spec)) for spec in self._sweep_specs
        ]
        self._plot.remove_pulse(idx)
        self._pulse.pop(idx)
        self._cross_capacitance = np.delete(
            np.delete(self._cross_capacitance, idx, axis=0),
            idx,
            axis=1,
        )
        ctrl = self._multi_ctrl._ctrl_pannels.pop(idx)
        ctrl.deleteLater()
        self._multi_ctrl._color_map.pop(idx)
        self._multi_ctrl.splitter.widget(idx).deleteLater()
        ControlPanel.port_idx -= 1  # decrement port index counter
        for i, ctrl in enumerate(self._multi_ctrl._ctrl_pannels):
            ctrl.idx = i

        if self._selected_port_idx >= len(self._pulse):
            self._selected_port_idx = len(self._pulse) - 1
        self._plot.set_selected_port_idx(self._selected_port_idx)

        self._multi_ctrl.refresh_table()
        self.refresh_panel_table()
        updated_sweeps = []
        for spec, target in sweep_entries:
            if target is None or target[0] == idx:
                continue
            if target[0] > idx:
                spec = QickSweepSpec(
                    segment_name=spec.segment_name,
                    output_name=f"awg_{target[0] - 1}",
                    start=spec.start,
                    stop=spec.stop,
                    count=spec.count,
                )
            updated_sweeps.append(spec)
        self._sweep_specs = updated_sweeps
        self._refresh_sweep_overlay(sync_rows=True)
        self._refresh_rf_editor()
        if self._trace is not None:
            for attr in ("x_idx", "y_idx"):
                trace_index = getattr(self._trace, attr)
                if trace_index == idx:
                    setattr(self._trace, attr, None)
                elif trace_index is not None and trace_index > idx:
                    setattr(self._trace, attr, trace_index - 1)
            self._trace.refresh_trace(self._pulse)

    def _build_menu(self):
        mb          = self.menuBar()

        # File
        m_file      = mb.addMenu("&File")
        m_file.addAction("Load JSON…",  self._load_json)
        m_file.addSeparator()
        m_file.addAction("E&xit",       self.close)

        # Pulse
        m_pulse     = mb.addMenu("&Pulse")
        gen         = m_pulse.addAction("&Generate QCS string…")
        gen.setShortcut(QtGui.QKeySequence("Ctrl+G"))
        gen.triggered.connect(self._generate_pulse)
        m_pulse.addSeparator()
        edit_rf = m_pulse.addAction("Create/Edit RF pulse...")
        edit_rf.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        edit_rf.triggered.connect(self._show_rf_editor)
        gen_qick = m_pulse.addAction("Generate QICK program...")
        gen_qick.setShortcut(QtGui.QKeySequence("Ctrl+Shift+G"))
        gen_qick.triggered.connect(self._generate_qick_pulse)
        preview_qick = m_pulse.addAction("Preview QICK waveforms...")
        preview_qick.triggered.connect(self._preview_qick_pulse)
        sync_timing = m_pulse.addAction("Synchronize port timing from selected port")
        sync_timing.triggered.connect(self._synchronize_port_timing)
        m_pulse.addSeparator()
        cross_capacitance = m_pulse.addAction(
            "Virtual-to-physical cross-capacitance matrix..."
        )
        cross_capacitance.setShortcut(QtGui.QKeySequence("Ctrl+Alt+C"))
        cross_capacitance.triggered.connect(self._configure_cross_capacitance)

        # Sweep
        m_sweep = mb.addMenu("&Sweep")
        clear_sweeps = m_sweep.addAction("Clear all amplitude sweeps")
        clear_sweeps.triggered.connect(
            lambda: self._clear_sweep("All amplitude sweeps removed")
        )

        # Grid
        m_grid = mb.addMenu("&Grid")
        configure_grid = m_grid.addAction("Configure grid...")
        configure_grid.setShortcut(QtGui.QKeySequence("Ctrl+Alt+G"))
        configure_grid.triggered.connect(self._configure_grid)
        m_grid.addSeparator()
        self._grid_snap_action = m_grid.addAction("Snap dragged points")
        self._grid_snap_action.setCheckable(True)
        self._grid_snap_action.setChecked(self._grid_snap_enabled)
        self._grid_snap_action.toggled.connect(self._toggle_grid_snap)
        self._grid_visible_action = m_grid.addAction("Show fixed grid")
        self._grid_visible_action.setCheckable(True)
        self._grid_visible_action.setChecked(self._grid_visible)
        self._grid_visible_action.toggled.connect(self._toggle_grid_visible)

        # View
        m_view = mb.addMenu("&View")
        voltage_view = m_view.addMenu("Voltage traces")
        self._voltage_view_actions = {}
        voltage_view_group = QtWidgets.QActionGroup(self)
        voltage_view_group.setExclusive(True)
        for label, mode in (
            ("Virtual + Physical", "both"),
            ("Virtual only", "virtual"),
            ("Physical only", "physical"),
        ):
            action = voltage_view.addAction(label)
            action.setCheckable(True)
            action.setChecked(mode == "both")
            action.triggered.connect(
                lambda checked=False, selected=mode: self._set_voltage_view(selected)
            )
            voltage_view_group.addAction(action)
            self._voltage_view_actions[mode] = action

    def _set_grid_settings(
        self,
        *,
        time_step_ns: float,
        voltage_step_mv: float,
        snap_enabled: bool,
        visible: bool,
    ) -> None:
        self._grid_time_ns = float(time_step_ns)
        self._grid_voltage_mv = float(voltage_step_mv)
        self._grid_snap_enabled = bool(snap_enabled)
        self._grid_visible = bool(visible)
        self._plot.set_grid(
            time_step_ns=self._grid_time_ns,
            voltage_step_mv=self._grid_voltage_mv,
            snap_enabled=self._grid_snap_enabled,
            visible=self._grid_visible,
        )
        if hasattr(self, "_grid_snap_action"):
            with QtCore.QSignalBlocker(self._grid_snap_action):
                self._grid_snap_action.setChecked(self._grid_snap_enabled)
        if hasattr(self, "_grid_visible_action"):
            with QtCore.QSignalBlocker(self._grid_visible_action):
                self._grid_visible_action.setChecked(self._grid_visible)

    def _configure_grid(self) -> None:
        dialog = GridSettingsDialog(
            time_step_ns=self._grid_time_ns,
            voltage_step_mv=self._grid_voltage_mv,
            snap_enabled=(self._grid_snap_enabled if self._grid_configured else True),
            visible=self._grid_visible,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        time_step_ns, voltage_step_mv, snap_enabled, visible = dialog.values()
        self._grid_configured = True
        self._set_grid_settings(
            time_step_ns=time_step_ns,
            voltage_step_mv=voltage_step_mv,
            snap_enabled=snap_enabled,
            visible=visible,
        )
        self.statusBar().showMessage(
            f"Grid: {time_step_ns:.6g} ns x {voltage_step_mv:.6g} mV; "
            f"snap {'on' if snap_enabled else 'off'}"
        )

    def _toggle_grid_snap(self, enabled: bool) -> None:
        self._grid_configured = True
        self._set_grid_settings(
            time_step_ns=self._grid_time_ns,
            voltage_step_mv=self._grid_voltage_mv,
            snap_enabled=enabled,
            visible=self._grid_visible,
        )

    def _toggle_grid_visible(self, visible: bool) -> None:
        self._set_grid_settings(
            time_step_ns=self._grid_time_ns,
            voltage_step_mv=self._grid_voltage_mv,
            snap_enabled=self._grid_snap_enabled,
            visible=visible,
        )

    def _configure_cross_capacitance(self) -> None:
        dialog = CrossCapacitanceDialog(
            len(self._pulse),
            self._cross_capacitance,
            self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        self._cross_capacitance = dialog.matrix()
        self._refresh_sweep_overlay(fit_view=True)
        off_diagonal = self._cross_capacitance.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        nonzero = int(np.count_nonzero(np.abs(off_diagonal) > 1.0e-15))
        self.statusBar().showMessage(
            "Virtual-to-physical matrix applied: "
            f"{nonzero} nonzero cross-capacitance terms"
        )

    def _set_voltage_view(self, mode: str) -> None:
        self._plot.set_voltage_view(mode)
        self.statusBar().showMessage(
            {
                "both": "Showing virtual (solid) and physical (dashed) voltages",
                "virtual": "Showing editable virtual-gate voltages",
                "physical": "Showing physical AWG voltages",
            }[mode]
        )

    def _refresh_physical_waveforms(self, *, fit_view: bool = False) -> None:
        time_ns, _virtual_mv, physical_mv = transform_virtual_waveforms(
            self._pulse,
            self._cross_capacitance,
        )
        self._plot.set_physical_waveforms(time_ns, physical_mv)
        if fit_view:
            self._plot.fit_view()

    def _sweep_target_indices(
        self,
        spec: Optional[QickSweepSpec] = None,
    ) -> Optional[Tuple[int, int]]:
        if spec is None:
            spec = self._sweep_spec
        if spec is None:
            return None
        try:
            port_index = self._qick_output_names().index(spec.output_name)
            segment_index = int(spec.segment_name.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return None
        if port_index >= len(self._pulse):
            return None
        if segment_index >= len(self._pulse[port_index].flat_segments()):
            return None
        return port_index, segment_index

    def _sweep_cartesian_count(self) -> int:
        return prod(spec.count for spec in self._sweep_specs) if self._sweep_specs else 1

    def _sync_sweep_rows(self) -> None:
        rows_by_port = {index: set() for index in range(len(self._pulse))}
        for spec in self._sweep_specs:
            target = self._sweep_target_indices(spec)
            if target is not None:
                rows_by_port[target[0]].add(target[1])
        for port_index, control in enumerate(self._multi_ctrl._ctrl_pannels):
            rows = rows_by_port.get(port_index, set())
            color = None
            if rows:
                color = (
                    self._plot.line_color(port_index)
                    if hasattr(self._plot, "line_color")
                    else QtGui.QColor(self._multi_ctrl._color_map[port_index])
                )
            control.set_sweep_rows(rows, color)

    def _refresh_sweep_overlay(
        self,
        *,
        fit_view: bool = False,
        sync_rows: bool = False,
    ) -> None:
        self._refresh_physical_waveforms()
        envelopes = []
        sweeps_by_segment = {}
        for spec in self._sweep_specs:
            target = self._sweep_target_indices(spec)
            if target is None:
                continue
            source_port, segment_index = target
            sweeps_by_segment.setdefault(
                (spec.segment_name, segment_index), []
            ).append((spec, source_port))

        for (segment_name, segment_index), segment_sweeps in sweeps_by_segment.items():
            point_index = segment_index * 2
            for destination_port in range(len(self._pulse)):
                if not any(
                    abs(self._cross_capacitance[destination_port, source_port])
                    > 1.0e-15
                    for _spec, source_port in segment_sweeps
                ):
                    continue
                lower_virtual = [pulse.copy() for pulse in self._pulse]
                upper_virtual = [pulse.copy() for pulse in self._pulse]
                for spec, source_port in segment_sweeps:
                    coefficient = float(
                        self._cross_capacitance[destination_port, source_port]
                    )
                    candidates = (
                        (coefficient * spec.start, spec.start),
                        (coefficient * spec.stop, spec.stop),
                    )
                    lower_value = min(candidates, key=lambda item: item[0])[1]
                    upper_value = max(candidates, key=lambda item: item[0])[1]
                    lower_virtual[source_port].v[
                        point_index:point_index + 2
                    ] = lower_value * self._qick_full_scale_mv
                    upper_virtual[source_port].v[
                        point_index:point_index + 2
                    ] = upper_value * self._qick_full_scale_mv
                time_ns, _virtual_lower, physical_lower = transform_virtual_waveforms(
                    lower_virtual,
                    self._cross_capacitance,
                )
                upper_time_ns, _virtual_upper, physical_upper = (
                    transform_virtual_waveforms(
                        upper_virtual,
                        self._cross_capacitance,
                    )
                )
                if not np.array_equal(time_ns, upper_time_ns):
                    raise RuntimeError("sweep endpoint time grids do not match")
                envelopes.append(
                    (
                        (f"awg_{destination_port}", segment_name),
                        destination_port,
                        time_ns,
                        physical_lower[destination_port],
                        physical_upper[destination_port],
                    )
                )
        self._plot.set_sweep_envelopes(envelopes)
        if sync_rows:
            self._sync_sweep_rows()
        if fit_view:
            self._plot.fit_view()

    def _clear_sweep(self, message: Optional[str] = None) -> None:
        self._sweep_specs = []
        self._plot.clear_sweep_envelope()
        self._sync_sweep_rows()
        if message:
            self.statusBar().showMessage(message)

    def _configure_segment_sweep(self, port_index: int, segment_index: int) -> None:
        if port_index < 0 or port_index >= len(self._pulse):
            return
        flat_segments = self._pulse[port_index].flat_segments()
        if segment_index < 0 or segment_index >= len(flat_segments):
            return
        output_name = f"awg_{port_index}"
        segment_name = f"set_{segment_index}"
        point_index = segment_index * 2
        current_amplitude = float(
            np.clip(
                self._pulse[port_index].v[point_index] / self._qick_full_scale_mv,
                -1.0,
                1.0,
            )
        )
        initial = next(
            (
                spec
                for spec in self._sweep_specs
                if spec.output_name == output_name and spec.segment_name == segment_name
            ),
            None,
        )
        other_point_count = prod(
            spec.count for spec in self._sweep_specs if spec is not initial
        ) if self._sweep_specs else 1
        dialog = SweepSettingsDialog(
            output_name=output_name,
            segment_name=segment_name,
            current_amplitude=current_amplitude,
            full_scale_mv=self._qick_full_scale_mv,
            initial=initial,
            cartesian_base_count=other_point_count,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_spec = dialog.value()
        for index, spec in enumerate(self._sweep_specs):
            if spec.output_name == output_name and spec.segment_name == segment_name:
                self._sweep_specs[index] = new_spec
                break
        else:
            self._sweep_specs.append(new_spec)
        self._port_select(port_index)
        self._refresh_sweep_overlay(fit_view=True, sync_rows=True)
        self.statusBar().showMessage(
            f"Sweep applied to {output_name}/{segment_name}: "
            f"{new_spec.start:.6g} to {new_spec.stop:.6g}, "
            f"{new_spec.count} axis points; "
            f"{self._sweep_cartesian_count()} Cartesian combinations"
        )

    def _remove_segment_sweep(self, port_index: int, segment_index: int) -> None:
        remaining = [
            spec
            for spec in self._sweep_specs
            if self._sweep_target_indices(spec) != (port_index, segment_index)
        ]
        if len(remaining) != len(self._sweep_specs):
            self._sweep_specs = remaining
            self._refresh_sweep_overlay(fit_view=True, sync_rows=True)
            self.statusBar().showMessage(
                f"Amplitude sweep removed; {self._sweep_cartesian_count()} "
                "Cartesian combinations remain"
            )

    def _on_segment_structure_changed(self, port_index: int) -> None:
        remaining = [
            spec
            for spec in self._sweep_specs
            if (self._sweep_target_indices(spec) or (-1, -1))[0] != port_index
        ]
        if len(remaining) != len(self._sweep_specs):
            self._sweep_specs = remaining
            self._refresh_sweep_overlay(fit_view=True, sync_rows=True)
            self.statusBar().showMessage(
                "Sweeps for the modified port were removed after segment structure changed"
            )

    def _build_toolbar(self):
        tb          = self.addToolBar("Tools")
        btn_fit     = QtWidgets.QAction("Fit", self)
        btn_fit.setToolTip("Show the full pulse (keyboard: F)")
        btn_fit.triggered.connect(self._fit_view)
        tb.addAction(btn_fit)
        rf_action = QtWidgets.QAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay),
            "RF Pulse",
            self,
        )
        rf_action.setToolTip("Create or edit the QICK RF pulse")
        rf_action.triggered.connect(self._show_rf_editor)
        tb.addAction(rf_action)
        # keyboard shortcut
        QtWidgets.QShortcut(QtGui.QKeySequence("F"), self, self._fit_view)

    def _fit_view(self):
        """Fit the view to the pulse data."""
        self._plot.fit_view()
        if self._rf_pulse_spec is not None and self._rf_timeline is not None:
            try:
                start_us, end_us, _ = rf_pulse_absolute_times_us(
                    self._pulse[0],
                    self._rf_pulse_spec,
                )
            except ValueError:
                pass
            else:
                start_ns = start_us * 1000.0
                end_ns = end_us * 1000.0
                x_range = self._plot.getPlotItem().vb.viewRange()[0]
                x_min = min(float(x_range[0]), start_ns)
                x_max = max(float(x_range[1]), end_ns)
                margin = max(1.0, 0.03 * max(1.0, x_max - x_min))
                self._plot.setXRange(x_min - margin, x_max + margin, padding=0.0)
        if self._trace is not None:
            self._trace.fit_view()

    def _show_rf_editor(self) -> None:
        if self._rf_panel is None:
            self._rf_panel = RfPulseEditorPanel(self._pulse[0], self)
            self._rf_panel.spec_applied.connect(self._apply_rf_spec)
            self._rf_panel.spec_removed.connect(self._remove_rf_spec)
            self._dock_rf = QtWidgets.QDockWidget("RF Pulse", self)
            self._dock_rf.setWidget(self._rf_panel)
            self._dock_rf.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable
                | QtWidgets.QDockWidget.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetClosable
            )
            self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._dock_rf)
            if self._rf_pulse_spec is not None:
                self._rf_panel.load_spec(self._rf_pulse_spec)
        self._dock_rf.show()
        self._dock_rf.raise_()

    def _apply_rf_spec(self, spec: QickRfPulseSpec) -> None:
        self._rf_pulse_spec = spec
        self._refresh_rf_timeline(fit_view=True)
        self.statusBar().showMessage(
            f"RF pulse applied: {spec.frequency_mhz:.6g} MHz, gain {spec.gain}, "
            f"ATT1/ATT2 {spec.att1_db:.2f}/{spec.att2_db:.2f} dB"
        )

    def _remove_rf_spec(self) -> None:
        self._rf_pulse_spec = None
        self._refresh_rf_timeline()
        self.statusBar().showMessage("RF pulse disabled")

    def _refresh_rf_timeline(self, *, fit_view: bool = False) -> None:
        if self._rf_timeline is None:
            return
        if self._rf_pulse_spec is None:
            self._rf_timeline.clear_pulse()
            self._rf_timeline.hide()
            return
        try:
            start_us, end_us, _ = rf_pulse_absolute_times_us(
                self._pulse[0],
                self._rf_pulse_spec,
            )
        except ValueError:
            self._rf_timeline.clear_pulse()
            self._rf_timeline.hide()
            return
        self._rf_timeline.set_pulse(
            gen_ch=self._rf_pulse_spec.gen_ch,
            start_ns=start_us * 1000.0,
            duration_ns=(end_us - start_us) * 1000.0,
            frequency_mhz=self._rf_pulse_spec.frequency_mhz,
            gain=self._rf_pulse_spec.gain,
            phase_degrees=self._rf_pulse_spec.phase_degrees,
            att1_db=self._rf_pulse_spec.att1_db,
            att2_db=self._rf_pulse_spec.att2_db,
        )
        self._rf_timeline.show()
        total_height = max(360, self._waveform_splitter.height())
        self._waveform_splitter.setSizes([max(220, total_height - 150), 150])
        if fit_view:
            self._fit_view()

    def _refresh_rf_editor(self) -> None:
        self._refresh_rf_timeline()
        if self._rf_panel is None:
            return
        self._rf_panel.refresh_segments(self._pulse[0])
        if self._rf_pulse_spec is not None:
            self._rf_panel.load_spec(self._rf_pulse_spec)

    def _add_segment(self, ramp: float, flat: float, v: float) -> None:
        try:
            self._pulse[self._selected_port_idx].add_flat_ramp(
                ramp,
                flat,
                v
            )
            self._plot.set_selected_port_idx(self._selected_port_idx)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid input", str(exc))
            return
        self._plot.refresh()
        self._plot.fit_view()
        self._multi_ctrl.refresh_table()
        self._refresh_trace_if_needed(force=True)
        self._refresh_sweep_overlay()
        if self._selected_port_idx == 0:
            self._refresh_rf_editor()

    def _flat_update(self, i0: int, i1: int, new_v: float) -> None:
        self.statusBar().showMessage(f"Flat {i0}-{i1} moved to {new_v:.6g} mV")
        self._schedule_deferred_refresh()

    def _point_update(self, i0, i1, new_t):
        self.statusBar().showMessage(f"Point {i0}-{i1} moved to {new_t:.6g} ns")
        self._schedule_deferred_refresh(timing_changed=True)

    def _schedule_deferred_refresh(self, *, timing_changed: bool = False) -> None:
        self._pending_table_refresh = True
        if self._trace is not None and self._trace.uses_port(self._selected_port_idx):
            self._pending_trace_refresh = True
        self._pending_rf_refresh |= timing_changed
        sweep_targets = {
            target
            for target in (
                self._sweep_target_indices(spec) for spec in self._sweep_specs
            )
            if target is not None
        }
        self._pending_sweep_refresh |= (
            bool(self._sweep_specs)
            or any(target[0] == self._selected_port_idx for target in sweep_targets)
        )
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _flush_deferred_refresh(self) -> None:
        if self._pending_table_refresh:
            self._multi_ctrl._ctrl_pannels[self._selected_port_idx].refresh_table()
        if self._pending_trace_refresh and self._trace is not None:
            self._trace.refresh_trace(self._pulse)
        if self._pending_rf_refresh:
            self._refresh_rf_editor()
        if self._pending_sweep_refresh:
            self._refresh_sweep_overlay()
        elif self._pending_table_refresh:
            self._refresh_physical_waveforms()
        self._pending_table_refresh = False
        self._pending_trace_refresh = False
        self._pending_rf_refresh = False
        self._pending_sweep_refresh = False

    def _ensure_trace_widget(self) -> TracePlotWidget:
        if self._trace is None:
            self._trace = TracePlotWidget(self)
            self._dock_trace.setWidget(self._trace)
            self._trace_placeholder.deleteLater()
        return self._trace

    def _refresh_trace_if_needed(self, *, force: bool = False) -> None:
        if self._trace is None:
            return
        if force or self._trace.uses_port(self._selected_port_idx):
            self._trace.refresh_trace(self._pulse)

    def _load_json(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open settings JSON", "", "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data                    = json.load(fp)
            if "initial_voltage" in data:
                self._pulse[0].v[0:2]   = float(data["initial_voltage"])
            if "voltage_bounds" in data and len(data["voltage_bounds"]) == 2:
                self._pulse[0].v_bounds = tuple(map(float, data["voltage_bounds"]))
            if "time_ns" in data and "voltage_mv" in data:
                loaded = PulseSequence.from_dict(data)
                self._pulse[0].t = loaded.t
                self._pulse[0].v = loaded.v
                self._pulse[0].v_bounds = loaded.v_bounds
            self._plot.refresh()
            self._plot.fit_view()
            self._multi_ctrl.refresh_table()
            self._refresh_trace_if_needed(force=True)
            self._refresh_rf_editor()
            self._refresh_sweep_overlay(fit_view=True, sync_rows=True)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Load failed", f"Could not read JSON:\n{exc}"
            )

    def _generate_pulse(self):
        try:
            code_str = self._generate_qcs_code()
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "QCS export failed", str(exc))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save generated QCS module",
            str(Path(__file__).resolve().parent / "qcs_dc_waveforms.py"),
            "Python files (*.py)",
        )
        if not path:
            return
        Path(path).write_text(code_str, encoding="utf-8")
        self._show_generated_code("Generated QCS pulse", path, code_str)

    def _generate_qcs_code(self) -> str:
        """Generate QCS code for the current pulse sequence."""
        return generate_qcs_program_code(
            self._pulse,
            cross_capacitance=self._cross_capacitance,
        )

    def _qick_output_names(self) -> Tuple[str, ...]:
        return tuple(f"awg_{index}" for index in range(len(self._pulse)))

    def _qick_export_settings(self) -> Optional[dict]:
        if len(self._pulse) > 8:
            QtWidgets.QMessageBox.warning(
                self,
                "QICK export unavailable",
                "axis_awg_tuning_v1 supports at most eight outputs.",
            )
            return None
        try:
            set_names = qick_set_segment_names(self._pulse[0])
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid waveform", str(exc))
            return None
        flat_durations = [
            self._pulse[0].t[end] - self._pulse[0].t[start]
            for start, end in self._pulse[0].flat_segments()
        ]
        default_set_index = int(np.argmax(flat_durations))
        dialog = QickExportDialog(
            len(self._pulse),
            set_names,
            self,
            default_set_index=default_set_index,
            initial_rf_spec=self._rf_pulse_spec,
            initial_sweeps=tuple(self._sweep_specs),
            initial_cross_capacitance=self._cross_capacitance,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None
        settings = dialog.values()
        self._rf_pulse_spec = settings["rf_pulse_spec"]
        if settings["sweeps"] is not None:
            self._sweep_specs = list(settings["sweeps"])
        elif settings["sweep"] is not None:
            self._sweep_specs = [settings["sweep"]]
        else:
            self._sweep_specs = []
        self._qick_full_scale_mv = settings["full_scale_mv"]
        self._cross_capacitance = np.asarray(
            settings["cross_capacitance"], dtype=float
        )
        if self._rf_panel is not None:
            self._rf_panel.load_spec(self._rf_pulse_spec)
        self._refresh_rf_timeline(fit_view=True)
        self._refresh_sweep_overlay(fit_view=True, sync_rows=True)
        return settings

    def _generate_qick_pulse(self) -> None:
        settings = self._qick_export_settings()
        if settings is None:
            return
        try:
            code_str = generate_qick_program_code(
                self._pulse,
                output_names=self._qick_output_names(),
                **settings,
            )
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "QICK export failed", str(exc))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save generated QICK module",
            str(Path(__file__).resolve().parent / "qick_dc_waveforms.py"),
            "Python files (*.py)",
        )
        if not path:
            return
        Path(path).write_text(code_str, encoding="utf-8")
        self._show_generated_code("Generated QICK pulse", path, code_str)

    def _preview_qick_pulse(self) -> None:
        settings = self._qick_export_settings()
        if settings is None:
            return
        sequence_settings = {
            name: settings[name]
            for name in (
                "fabric_mhz",
                "full_scale_mv",
                "sweep",
                "sweeps",
                "cross_capacitance",
            )
        }
        try:
            sequence = build_qick_sequence(
                self._pulse,
                output_names=self._qick_output_names(),
                **sequence_settings,
            )
            # FineTuneSequence defaults to first/middle/last sweep points,
            # keeping preview cost bounded for large sweeps.
            sequence.plot_preview(point_indices=None, show=True)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "QICK preview failed", str(exc))

    def _show_generated_code(self, title: str, path: str, code_str: str) -> None:
        self.statusBar().showMessage(f"Saved {path}")
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setText(f"Generated module saved to:\n{path}")
        dlg.setDetailedText(code_str)
        dlg.exec_()

    def _synchronize_port_timing(self) -> None:
        """Copy the selected SET/RAMP timing grid to every other port."""
        source = self._pulse[self._selected_port_idx]
        for index, pulse in enumerate(self._pulse):
            if index == self._selected_port_idx:
                continue
            levels = [float(np.interp(source.t[set_idx], pulse.t, pulse.v))
                      for set_idx in range(0, len(source.t) - 1, 2)]
            pulse.t = source.t.copy()
            pulse.v = np.empty_like(source.v)
            for set_number, level in enumerate(levels):
                pulse.v[2 * set_number:2 * set_number + 2] = level
        self._plot.refresh()
        self._plot.fit_view()
        self._multi_ctrl.refresh_table()
        self._refresh_trace_if_needed(force=True)
        self._refresh_rf_editor()
        self._refresh_sweep_overlay(fit_view=True, sync_rows=True)
        self.statusBar().showMessage("Port timing synchronized for QICK export")

    def _add_port(self):
        """Add a new control panel for a new PulseSequence."""
        # Start with the existing timing grid so multi-output QICK export is
        # immediately valid; levels remain independently editable per port.
        new_pulse       = self._pulse[0].copy()
        new_pulse.v[:]  = 0.0
        self._plot.add_pulse(new_pulse)
        color = (
            self._plot.line_color(len(self._plot._line) - 1)
            if hasattr(self._plot, "line_color")
            else self._plot._line[-1].get_color()
        )
        self._multi_ctrl._color_map.append(color)
        self._pulse.append(new_pulse)
        old_count = self._cross_capacitance.shape[0]
        expanded_coupling = np.eye(old_count + 1, dtype=float)
        expanded_coupling[:old_count, :old_count] = self._cross_capacitance
        self._cross_capacitance = expanded_coupling
        new_ctrl        = ControlPanel(new_pulse)
        self._wire_control_panel(new_ctrl)
        self._multi_ctrl._ctrl_pannels.append(new_ctrl)
        self._multi_ctrl.splitter.insertWidget(len(self._multi_ctrl._ctrl_pannels) - 1, new_ctrl)
        self._multi_ctrl.refresh_table()
        self.refresh_panel_table()
        self._refresh_sweep_overlay(fit_view=True, sync_rows=True)

    def _port_select(self, idx: int) -> None:
        """Activate the selected control panel."""
        self._selected_port_idx         = idx
        self._plot._selected_port_idx   = idx
        self._plot.set_selected_port_idx(idx)
        self._plot.refresh(idx if _USE_PYQTGRAPH else None)

    def _plot_refresh(self):
        self._plot.refresh(self._selected_port_idx if _USE_PYQTGRAPH else None)
        self._refresh_trace_if_needed()
        self._refresh_sweep_overlay()
        if self._selected_port_idx == 0:
            self._refresh_rf_editor()

    def _make_row_slot(self, real_slot):
        def wrapper(checked=False):
            btn   = self.sender()
            row   = self._multi_ctrl.panel_table.indexAt(btn.pos()).row()
            real_slot(row)
        return wrapper
    def refresh_panel_table(self):
        """Refresh the control panel table with current pulse data."""
        self._multi_ctrl.panel_table.setRowCount(len(self._multi_ctrl._ctrl_pannels))
        for idx, _ in enumerate(self._multi_ctrl._ctrl_pannels):
            item_idx = QtWidgets.QTableWidgetItem(str(idx + 1))
            item_idx.setTextAlignment(QtCore.Qt.AlignCenter)
            self._multi_ctrl.panel_table.setItem(idx, 0, item_idx)

            color = self._multi_ctrl._color_map[idx]
            item_color = QtWidgets.QTableWidgetItem()
            if isinstance(color, QtGui.QColor):
                table_color = QtGui.QColor(color)
            elif isinstance(color, tuple):
                rgba = tuple(color) + (1.0,) * (4 - len(color))
                table_color = QtGui.QColor.fromRgbF(*rgba[:4])
            else:
                table_color = QtGui.QColor(str(color))
            item_color.setBackground(table_color)
            self._multi_ctrl.panel_table.setItem(idx, 1, item_color)

            btn_x = QtWidgets.QPushButton("set_x")
            btn_x.clicked.connect(self._make_row_slot(self._set_x))
            self._multi_ctrl.panel_table.setCellWidget(idx, 2, btn_x)

            btn_y = QtWidgets.QPushButton("set_y")
            btn_y.clicked.connect(self._make_row_slot(self._set_y))
            self._multi_ctrl.panel_table.setCellWidget(idx, 3, btn_y)

    def _set_x(self, idx):
        trace = self._ensure_trace_widget()
        trace.x_idx = idx
        trace.refresh_trace(self._pulse)
        trace.fit_view()

    def _set_y(self, idx):
        trace = self._ensure_trace_widget()
        trace.y_idx = idx
        trace.refresh_trace(self._pulse)
        trace.fit_view()

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1500, 650)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
