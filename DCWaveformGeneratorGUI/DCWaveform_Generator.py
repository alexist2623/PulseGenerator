#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive DC waveform editor with Keysight QCS and QICK export.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

import json
from dataclasses import asdict, replace
from math import prod
from pathlib import Path
import sys
import traceback
from typing import Tuple, Optional, List, Callable, Mapping, Sequence
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
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    _USE_PYQTGRAPH = False
else:
    # Names below are referenced only by the legacy fallback class bodies.
    Canvas = QtWidgets.QWidget
    Figure = Axes = Line2D = object
    colors = None
    FuncFormatter = MultipleLocator = None
    _USE_PYQTGRAPH = True

try:
    from .dc_waveform_core import (
        BIAS_T_COMPENSATION_MODES,
        BIAS_T_COMPENSATION_TYPES,
        DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
        DEFAULT_BIAS_T_COMPENSATION_FRACTION,
        DEFAULT_BIAS_T_FILTER_TAU_US,
        DEFAULT_DC_MEASURE_GAIN_V_PER_A,
        DEFAULT_INITIAL_VOLTAGE_MV,
        DEFAULT_QICK_FABRIC_MHZ,
        DEFAULT_QICK_TPROC_MHZ,
        DEFAULT_QICK_FULL_SCALE_MV,
        PulseSequence,
        QICK_INPUT_BOARD_TYPES,
        QICK_OUTPUT_BOARD_TYPES,
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
        BIAS_T_COMPENSATION_MODES,
        BIAS_T_COMPENSATION_TYPES,
        DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
        DEFAULT_BIAS_T_COMPENSATION_FRACTION,
        DEFAULT_BIAS_T_FILTER_TAU_US,
        DEFAULT_DC_MEASURE_GAIN_V_PER_A,
        DEFAULT_INITIAL_VOLTAGE_MV,
        DEFAULT_QICK_FABRIC_MHZ,
        DEFAULT_QICK_TPROC_MHZ,
        DEFAULT_QICK_FULL_SCALE_MV,
        PulseSequence,
        QICK_INPUT_BOARD_TYPES,
        QICK_OUTPUT_BOARD_TYPES,
        QickDdrReadoutSpec,
        QickRfPulseSpec,
        QickSweepSpec,
        build_qick_sequence,
        generate_qcs_program_code,
        generate_qick_program_code,
        qick_set_segment_names,
        transform_virtual_waveforms,
    )

try:
    from .qick_qcodes_experiment import (
        QcodesRunConfig,
        QickConnectionConfig,
        build_qick_program,
        connect_qick,
        run_qick_qcodes_experiment,
    )
except ImportError:
    from qick_qcodes_experiment import (
        QcodesRunConfig,
        QickConnectionConfig,
        build_qick_program,
        connect_qick,
        run_qick_qcodes_experiment,
    )

try:
    from .qick_sparameter_sweep import SParameterSweepConfig
    from .sparameter_gui import (
        DEFAULT_SPARAMETER_DB_PATH,
        SParameterLoadWorker,
        SParameterPlotWidget,
        SParameterSweepPanel,
        SParameterSweepWorker,
    )
except ImportError:
    from qick_sparameter_sweep import SParameterSweepConfig
    from sparameter_gui import (
        DEFAULT_SPARAMETER_DB_PATH,
        SParameterLoadWorker,
        SParameterPlotWidget,
        SParameterSweepPanel,
        SParameterSweepWorker,
    )

try:
    from .stability_diagram import (
        StabilityDiagramPanel,
        StabilityDiagramWorker,
        normalize_stability_settings,
    )
except ImportError:
    from stability_diagram import (
        StabilityDiagramPanel,
        StabilityDiagramWorker,
        normalize_stability_settings,
    )

try:
    from .noise_analysis import (
        NoiseAnalysisPanel,
        NoiseTraceLoadWorker,
        normalize_noise_analysis_settings,
    )
except ImportError:
    from noise_analysis import (
        NoiseAnalysisPanel,
        NoiseTraceLoadWorker,
        normalize_noise_analysis_settings,
    )

try:
    from .calibration_gui import (
        CalibrationPanel,
        CalibrationWorker,
        default_calibration_settings,
    )
    from .qick_power_calibration import (
        InputPowerCalibrationConfig,
        OscilloscopeConfig,
        OutputPowerCalibrationConfig,
    )
    from .dc_voltage_calibration import DcVoltageCalibrationConfig
except ImportError:
    from calibration_gui import (
        CalibrationPanel,
        CalibrationWorker,
        default_calibration_settings,
    )
    from qick_power_calibration import (
        InputPowerCalibrationConfig,
        OscilloscopeConfig,
        OutputPowerCalibrationConfig,
    )
    from dc_voltage_calibration import DcVoltageCalibrationConfig

try:
    from .qick_front_panel import (
        QickFrontPanelControl,
        QickFrontPanelPreview,
        identify_qick_front_panel,
    )
except ImportError:
    from qick_front_panel import (
        QickFrontPanelControl,
        QickFrontPanelPreview,
        identify_qick_front_panel,
    )


DEFAULT_QSTL_AWG_CHANNELS = (1, 3, 5, 7, 8, 9, 10, 11)
DEFAULT_QSTL_RF_CHANNELS = (0, 2, 4, 6, 12, 13, 14, 15)
TIME_UNIT_NS = {"ns": 1.0, "us": 1000.0, "ms": 1_000_000.0}
DEFAULT_TIME_UNIT = "us"
DEFAULT_GUI_DURATION_NS = 1000.0
DEFAULT_GUI_RAMP_NS = 1000.0
DEFAULT_GUI_FLAT_NS = 1000.0
SETTINGS_SCHEMA = "qstl-pulse-generator-gui"
SETTINGS_VERSION = 20
SUPPORTED_SETTINGS_VERSIONS = tuple(range(1, SETTINGS_VERSION + 1))
DEFAULT_QICK_HOST = "192.168.2.99"
DEFAULT_QICK_NS_PORT = 8888
DEFAULT_QICK_PROXY_NAME = "myqick"
DEFAULT_QCODES_DB_PATH = str(Path.home() / "qick_experiments.db")
DEFAULT_BIAS_T_COMPENSATION_MV = (
    DEFAULT_QICK_FULL_SCALE_MV * DEFAULT_BIAS_T_COMPENSATION_FRACTION
)

DEFAULT_RF_OUTPUT_SETTINGS = {
    "enabled": False,
    "gen_ch": DEFAULT_QSTL_RF_CHANNELS[0],
    "segment_name": "set_0",
    "delay_us": 0.0,
    "duration_us": 1.0,
    "frequency_mhz": 50.0,
    "gain": 20000,
    "output_board_type": "RF_Out",
    "att1_db": 0.0,
    "att2_db": 0.0,
    "filter_type": "bypass",
    "filter_cutoff": 2.5,
    "filter_bandwidth": 1.0,
    "phase_degrees": 0.0,
    "nqz": 1,
    "require_within_segment": True,
}

DEFAULT_RF_READOUT_SETTINGS = {
    "enabled": False,
    "ro_ch": 0,
    "segment_name": "set_0",
    "delay_us": 0.0,
    "samples_per_trigger": 64,
    "readout_frequency_mhz": 50.0,
    "margin_input_samples": 1024,
    "force_overwrite": False,
    "post_run_read_delay_seconds": 0.1,
    "input_board_type": "RF_In",
    "attenuation_db": 20.0,
    "dc_gain_db": 0.0,
    "dc_measure_mode": False,
    "dc_measure_gain_v_per_a": DEFAULT_DC_MEASURE_GAIN_V_PER_A,
    "dc_voltage_calibration_enabled": False,
    "dc_voltage_calibration_database_path": "",
    "dc_voltage_calibration_run_id": 0,
    "filter_type": "bypass",
    "filter_cutoff": 2.5,
    "filter_bandwidth": 1.0,
    "nqz": 1,
}

DEFAULT_SPARAMETER_SETTINGS = {
    "database_path": DEFAULT_SPARAMETER_DB_PATH,
    **asdict(SParameterSweepConfig()),
}
DEFAULT_CALIBRATION_SETTINGS = default_calibration_settings()


def _time_from_ns(value_ns: float, unit: str) -> float:
    return float(value_ns) / TIME_UNIT_NS[unit]


def _time_to_ns(value: float, unit: str) -> float:
    return float(value) * TIME_UNIT_NS[unit]


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
        margin_x = max(0.5, 0.03 * float(np.ptp(self._pulse[self.x_idx].v)))
        margin_y = max(0.5, 0.10 * float(np.ptp(self._pulse[self.y_idx].v)))
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
            time_unit: str = DEFAULT_TIME_UNIT,
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
        self._time_unit = time_unit

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

    def set_time_unit(self, unit: str) -> None:
        if unit not in TIME_UNIT_NS:
            raise ValueError(f"unsupported time unit {unit!r}")
        self._time_unit = unit
        scale = 1.0 / TIME_UNIT_NS[unit]
        self.ax.set_xlabel(f"time [{unit}]")
        self.ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{value * scale:.6g}")
        )
        self.draw_idle()

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
            x_min - margin_x,
            x_max + margin_x
        )
        self.ax.set_ylim(
            y_min - margin_y,
            y_max + margin_y
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
                f"{_time_from_ns(self._pulse[self._selected_port_idx].t[i0], self._time_unit):0.3g} "
                f"{self._time_unit}\n{new_v:0.3g} mV"
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
                f"{_time_from_ns(actual_t, self._time_unit):0.3g} {self._time_unit}\n"
                f"{self._pulse[self._selected_port_idx].v[i0]:0.3g} mV"
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
            time_unit: str = DEFAULT_TIME_UNIT,
            parent=None
        ):
        super().__init__(parent)
        self._pulse         = pulse
        self.idx            = ControlPanel.port_idx
        ControlPanel.port_idx += 1
        self._sweep_row: Optional[int] = None
        self._sweep_rows = set()
        self._sweep_color: Optional[QtGui.QColor] = None
        self._time_unit = time_unit

        v_splitter          = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        layout              = QtWidgets.QVBoxLayout(self)

        form                = QtWidgets.QFormLayout()
        self.edit_ramp = QtWidgets.QLineEdit(
            f"{_time_from_ns(DEFAULT_GUI_RAMP_NS, self._time_unit):.6g}"
        )
        self.edit_flat = QtWidgets.QLineEdit(
            f"{_time_from_ns(DEFAULT_GUI_FLAT_NS, self._time_unit):.6g}"
        )
        self.edit_v = QtWidgets.QLineEdit(f"{DEFAULT_INITIAL_VOLTAGE_MV:.6g}")

        for w in (self.edit_ramp, self.edit_flat, self.edit_v):
            w.setValidator(QtGui.QDoubleValidator(decimals=9))

        self._ramp_label = QtWidgets.QLabel()
        self._flat_label = QtWidgets.QLabel()
        form.addRow(self._ramp_label, self.edit_ramp)
        form.addRow(self._flat_label, self.edit_flat)
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
        self._refresh_time_labels()

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
        sweep_menu = menu.addMenu("Voltage sweep")
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
            ok = self._pulse.edit_ramp(flat_idx, _time_to_ns(val, self._time_unit))
        elif col == 2:
            ok = self._pulse.edit_flat(flat_idx, _time_to_ns(val, self._time_unit))
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
        self.add_requested.emit(
            _time_to_ns(ramp, self._time_unit),
            _time_to_ns(flat, self._time_unit),
            v,
        )

    def _refresh_time_labels(self) -> None:
        self._ramp_label.setText(f"Ramp [{self._time_unit}]:")
        self._flat_label.setText(f"Flat [{self._time_unit}]:")
        self.table.setHorizontalHeaderLabels(
            [
                "#",
                f"Ramp [{self._time_unit}]",
                f"Flat [{self._time_unit}]",
                "Virtual V [mV]",
            ]
        )

    def set_time_unit(self, unit: str) -> None:
        if unit not in TIME_UNIT_NS:
            raise ValueError(f"unsupported time unit {unit!r}")
        if unit == self._time_unit:
            return
        old_unit = self._time_unit
        for editor in (self.edit_ramp, self.edit_flat):
            try:
                value_ns = _time_to_ns(float(editor.text()), old_unit)
            except ValueError:
                continue
            editor.setText(f"{_time_from_ns(value_ns, unit):.6g}")
        self._time_unit = unit
        self._refresh_time_labels()
        self.refresh_table()

    def refresh_table(self):
        with QtCore.QSignalBlocker(self.table):
            segs = self._pulse.flat_segments()
            self.table.setRowCount(len(segs))
            for row, (i0, i1) in enumerate(segs):
                ramp  = self._pulse.t[i0] - self._pulse.t[i0 - 1] if i0 > 0 else 0
                flat  = self._pulse.t[i1] - self._pulse.t[i0]
                v     = self._pulse.v[i0]
                for col, val in enumerate(
                    [
                        row + 1,
                        _time_from_ns(ramp, self._time_unit),
                        _time_from_ns(flat, self._time_unit),
                        v,
                    ]
                ):
                    item = QtWidgets.QTableWidgetItem(f"{val:.6g}")
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    if row in self._sweep_rows:
                        highlight = QtGui.QColor(
                            self._sweep_color or QtGui.QColor("#f2c14e")
                        )
                        highlight.setAlpha(42)
                        item.setBackground(QtGui.QBrush(highlight))
                        item.setToolTip("Voltage sweep target")
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
            time_unit: str = DEFAULT_TIME_UNIT,
            parent=None
        ):
        super().__init__()
        self._time_unit = time_unit
        self._ctrl_pannels: List[ControlPanel] = [
            ControlPanel(pulse, time_unit=time_unit)
        ]
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

    def set_time_unit(self, unit: str) -> None:
        self._time_unit = unit
        for control in self._ctrl_pannels:
            control.set_time_unit(unit)


class GridSettingsDialog(QtWidgets.QDialog):
    """Configure fixed waveform-grid spacing and drag snapping."""

    def __init__(
        self,
        *,
        time_step_ns: float,
        voltage_step_mv: float,
        snap_enabled: bool,
        visible: bool,
        time_unit: str = DEFAULT_TIME_UNIT,
        parent=None,
    ):
        super().__init__(parent)
        self._time_unit = time_unit
        self.setWindowTitle("Waveform grid settings")
        form = QtWidgets.QFormLayout(self)

        self.time_step_ns = QtWidgets.QDoubleSpinBox()
        self.time_step_ns.setRange(1.0e-6, 1.0e12)
        self.time_step_ns.setDecimals(6)
        self.time_step_ns.setValue(_time_from_ns(time_step_ns, time_unit))
        self.time_step_ns.setSuffix(f" {time_unit}")

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
            f"Time and voltage grids are anchored at 0 {time_unit} and 0 mV."
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
            _time_to_ns(self.time_step_ns.value(), self._time_unit),
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
    """Configure the voltage sweep attached to one SET/output pair."""

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
        self.setWindowTitle("Voltage sweep settings")
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
                f"{current_amplitude * self._full_scale_mv:.6g} mV"
            ),
        )

        default_start = max(-1.0, current_amplitude - 0.2) * self._full_scale_mv
        default_stop = min(1.0, current_amplitude + 0.2) * self._full_scale_mv
        if initial is not None:
            default_start = initial.start * self._full_scale_mv
            default_stop = initial.stop * self._full_scale_mv

        self.start = QtWidgets.QDoubleSpinBox()
        self.stop = QtWidgets.QDoubleSpinBox()
        for widget, value in ((self.start, default_start), (self.stop, default_stop)):
            widget.setRange(-self._full_scale_mv, self._full_scale_mv)
            widget.setDecimals(6)
            widget.setSingleStep(max(0.001, self._full_scale_mv / 100.0))
            widget.setSuffix(" mV")
            widget.setValue(value)
        self.count = QtWidgets.QSpinBox()
        self.count.setRange(1, 1_000_000)
        self.count.setValue(initial.count if initial is not None else 9)
        self.endpoint_summary = QtWidgets.QLabel()
        self.cartesian_summary = QtWidgets.QLabel()

        form.addRow("Start voltage:", self.start)
        form.addRow("Stop voltage:", self.stop)
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
            f"{self.start.value():.6g} mV to {self.stop.value():.6g} mV"
        )
        self.cartesian_summary.setText(
            f"{self._cartesian_base_count} x {self.count.value()} = "
            f"{self._cartesian_base_count * self.count.value()} points"
        )

    def value(self) -> QickSweepSpec:
        return QickSweepSpec(
            segment_name=self._segment_name,
            output_name=self._output_name,
            start=self.start.value() / self._full_scale_mv,
            stop=self.stop.value() / self._full_scale_mv,
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
        self.filter_type = QtWidgets.QComboBox()
        self.filter_type.addItems(["bypass", "lowpass", "highpass", "bandpass"])
        self.filter_cutoff = QtWidgets.QDoubleSpinBox()
        self.filter_cutoff.setRange(0.0, 100.0)
        self.filter_cutoff.setDecimals(6)
        self.filter_cutoff.setValue(2.5)
        self.filter_cutoff.setSuffix(" GHz")
        self.filter_bandwidth = QtWidgets.QDoubleSpinBox()
        self.filter_bandwidth.setRange(0.001, 100.0)
        self.filter_bandwidth.setDecimals(6)
        self.filter_bandwidth.setValue(1.0)
        self.filter_bandwidth.setSuffix(" GHz")
        self.phase_degrees = QtWidgets.QDoubleSpinBox()
        self.phase_degrees.setRange(-360.0, 360.0)
        self.phase_degrees.setDecimals(6)
        self.phase_degrees.setSuffix(" deg")
        self.nqz = QtWidgets.QSpinBox()
        self.nqz.setRange(1, 2)
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
        form.addRow("Output filter:", self.filter_type)
        form.addRow("Filter cutoff/center:", self.filter_cutoff)
        form.addRow("Filter bandwidth:", self.filter_bandwidth)
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
            self.filter_type,
            self.filter_cutoff,
            self.filter_bandwidth,
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
            filter_type=self.filter_type.currentText(),
            filter_cutoff=self.filter_cutoff.value(),
            filter_bandwidth=self.filter_bandwidth.value(),
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
        self.filter_type.setCurrentText(spec.filter_type)
        self.filter_cutoff.setValue(spec.filter_cutoff)
        self.filter_bandwidth.setValue(spec.filter_bandwidth)
        self.phase_degrees.setValue(spec.phase_degrees)
        self.nqz.setValue(spec.nqz)
        self.require_within.setChecked(spec.require_within_segment)
        self._preview_current()


class RfPulsePortPanel(QtWidgets.QGroupBox):
    """Compact, always-available editor for one RF output generator."""

    changed = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal(object)
    front_panel_requested = QtCore.pyqtSignal(object)

    def __init__(
        self,
        pulse: PulseSequence,
        index: int,
        *,
        default_gen_ch: Optional[int] = None,
        time_unit: str = DEFAULT_TIME_UNIT,
        parent=None,
    ):
        super().__init__(parent)
        self._pulse = pulse
        self._index = index
        self._time_unit = time_unit
        self._front_panel_configuration = None
        self.setCheckable(True)
        self.setChecked(False)
        form = QtWidgets.QFormLayout(self)

        self.gen_ch = QtWidgets.QSpinBox()
        self.gen_ch.setRange(0, 255)
        self.gen_ch.setValue(index if default_gen_ch is None else default_gen_ch)
        self.output_board_type = QtWidgets.QComboBox()
        self.output_board_type.addItems(QICK_OUTPUT_BOARD_TYPES)
        self.segment = QtWidgets.QComboBox()
        self.delay = QtWidgets.QDoubleSpinBox()
        self.duration = QtWidgets.QDoubleSpinBox()
        for editor in (self.delay, self.duration):
            editor.setRange(0.0, 1.0e12)
            editor.setDecimals(9)
        self.duration.setMinimum(1.0e-9)
        self.duration.setValue(_time_from_ns(1000.0, time_unit))
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
        self.filter_type = QtWidgets.QComboBox()
        self.filter_type.addItems(["bypass", "lowpass", "highpass", "bandpass"])
        self.filter_cutoff = QtWidgets.QDoubleSpinBox()
        self.filter_cutoff.setRange(0.0, 100.0)
        self.filter_cutoff.setDecimals(6)
        self.filter_cutoff.setValue(2.5)
        self.filter_cutoff.setSuffix(" GHz")
        self.filter_bandwidth = QtWidgets.QDoubleSpinBox()
        self.filter_bandwidth.setRange(0.001, 100.0)
        self.filter_bandwidth.setDecimals(6)
        self.filter_bandwidth.setValue(1.0)
        self.filter_bandwidth.setSuffix(" GHz")
        self.phase_degrees = QtWidgets.QDoubleSpinBox()
        self.phase_degrees.setRange(-360.0, 360.0)
        self.phase_degrees.setDecimals(6)
        self.phase_degrees.setSuffix(" deg")
        self.nqz = QtWidgets.QSpinBox()
        self.nqz.setRange(1, 2)
        self.nqz.setValue(1)
        self.require_within = QtWidgets.QCheckBox("Keep pulse inside anchor SET")
        self.require_within.setChecked(True)

        self.front_panel_preview = QickFrontPanelPreview(self)
        self.front_panel_preview.activated.connect(
            lambda: self.front_panel_requested.emit(self)
        )
        form.addRow("Front panel:", self.front_panel_preview)
        form.addRow("Generator index:", self.gen_ch)
        form.addRow("Output board:", self.output_board_type)
        form.addRow("Anchor SET:", self.segment)
        self._delay_label = QtWidgets.QLabel()
        self._duration_label = QtWidgets.QLabel()
        form.addRow(self._delay_label, self.delay)
        form.addRow(self._duration_label, self.duration)
        form.addRow("Frequency:", self.frequency_mhz)
        form.addRow("Gain:", self.gain)
        shared_path_note = QtWidgets.QLabel(
            "ATT and filter settings are edited from the HWH-backed Front Panel."
        )
        shared_path_note.setWordWrap(True)
        shared_path_note.setStyleSheet("QLabel { color: #4f5b66; }")
        form.addRow(shared_path_note)
        form.addRow("Phase:", self.phase_degrees)
        form.addRow("Nyquist zone:", self.nqz)
        form.addRow(self.require_within)
        remove_button = QtWidgets.QPushButton("Remove RF Port")
        remove_button.clicked.connect(lambda: self.remove_requested.emit(self))
        form.addRow(remove_button)

        self.refresh_segments(pulse)
        self.set_time_unit(time_unit, force=True)
        for widget in (
            self.gen_ch,
            self.output_board_type,
            self.segment,
            self.delay,
            self.duration,
            self.frequency_mhz,
            self.gain,
            self.att1_db,
            self.att2_db,
            self.filter_type,
            self.filter_cutoff,
            self.filter_bandwidth,
            self.phase_degrees,
            self.nqz,
            self.require_within,
        ):
            signal = getattr(widget, "valueChanged", None)
            if signal is None:
                signal = getattr(widget, "currentIndexChanged", None)
            if signal is None:
                signal = getattr(widget, "toggled", None)
            signal.connect(self.changed.emit)
        self.toggled.connect(self.changed.emit)
        self.output_board_type.currentTextChanged.connect(
            self._update_board_controls
        )
        self.gen_ch.valueChanged.connect(self._sync_front_panel_selection)
        self._update_board_controls()
        self.set_index(index)

    def _update_board_controls(self, *_args) -> None:
        has_attenuators = self.output_board_type.currentText() == "RF_Out"
        self.att1_db.setEnabled(has_attenuators)
        self.att2_db.setEnabled(has_attenuators)
        self.filter_type.setEnabled(has_attenuators)
        self.filter_cutoff.setEnabled(has_attenuators)
        self.filter_bandwidth.setEnabled(has_attenuators)
        if has_attenuators:
            tooltip = "RF_Out onboard attenuator"
        else:
            tooltip = "DC_Out has no onboard ATT1/ATT2; these values are ignored"
        self.att1_db.setToolTip(tooltip)
        self.att2_db.setToolTip(tooltip)
        self.filter_type.setToolTip(tooltip)
        self.filter_cutoff.setToolTip(tooltip)
        self.filter_bandwidth.setToolTip(tooltip)

    def set_front_panel_configuration(self, configuration) -> None:
        self._front_panel_configuration = configuration
        self.front_panel_preview.set_configuration(configuration)
        self._sync_front_panel_selection()

    def _sync_front_panel_selection(self, *_args) -> None:
        self.front_panel_preview.set_channels(output_ch=self.gen_ch.value())
        if self._front_panel_configuration is None:
            return
        for port in self._front_panel_configuration.outputs:
            if self.gen_ch.value() in port.qick_channels:
                if port.board_type in QICK_OUTPUT_BOARD_TYPES:
                    self.output_board_type.setCurrentText(port.board_type)
                return

    def apply_front_panel_settings(self, values: Mapping[str, object]) -> None:
        self.gen_ch.setValue(int(values["output_ch"]))
        self.output_board_type.setCurrentText(str(values["output_board_type"]))
        self.att1_db.setValue(float(values["output_att1_db"]))
        self.att2_db.setValue(float(values["output_att2_db"]))
        self.filter_type.setCurrentText(str(values["output_filter_type"]))
        self.filter_cutoff.setValue(float(values["output_filter_cutoff_ghz"]))
        self.filter_bandwidth.setValue(
            float(values["output_filter_bandwidth_ghz"])
        )
        self.nqz.setValue(int(values.get("output_nqz", self.nqz.value())))
        self._update_board_controls()
        self._sync_front_panel_selection()
        self.changed.emit()

    def set_index(self, index: int) -> None:
        self._index = index
        self.setTitle(f"RF Output {index + 1}")

    def set_time_unit(self, unit: str, *, force: bool = False) -> None:
        if unit not in TIME_UNIT_NS:
            raise ValueError(f"unsupported time unit {unit!r}")
        if not force and unit == self._time_unit:
            return
        old_unit = self._time_unit
        delay_ns = _time_to_ns(self.delay.value(), old_unit)
        duration_ns = _time_to_ns(self.duration.value(), old_unit)
        self._time_unit = unit
        with QtCore.QSignalBlocker(self.delay), QtCore.QSignalBlocker(self.duration):
            self.delay.setValue(_time_from_ns(delay_ns, unit))
            self.duration.setValue(_time_from_ns(duration_ns, unit))
            self.delay.setSuffix(f" {unit}")
            self.duration.setSuffix(f" {unit}")
        self._delay_label.setText(f"Delay [{unit}]:")
        self._duration_label.setText(f"Duration [{unit}]:")

    def refresh_segments(self, pulse: Optional[PulseSequence] = None) -> None:
        if pulse is not None:
            self._pulse = pulse
        previous = self.segment.currentData()
        with QtCore.QSignalBlocker(self.segment):
            self.segment.clear()
            for index, (start, end) in enumerate(self._pulse.flat_segments()):
                name = f"set_{index}"
                self.segment.addItem(name, name)
            match = self.segment.findData(previous)
            if match >= 0:
                self.segment.setCurrentIndex(match)

    def configured_spec(self) -> QickRfPulseSpec:
        """Return the editor values even when this output is disabled."""
        return QickRfPulseSpec(
            gen_ch=self.gen_ch.value(),
            segment_name=str(self.segment.currentData()),
            delay_us=_time_to_ns(self.delay.value(), self._time_unit) / 1000.0,
            duration_us=_time_to_ns(self.duration.value(), self._time_unit) / 1000.0,
            frequency_mhz=self.frequency_mhz.value(),
            gain=self.gain.value(),
            att1_db=self.att1_db.value(),
            att2_db=self.att2_db.value(),
            phase_degrees=self.phase_degrees.value(),
            nqz=self.nqz.value(),
            require_within_segment=self.require_within.isChecked(),
            filter_type=self.filter_type.currentText(),
            filter_cutoff=self.filter_cutoff.value(),
            filter_bandwidth=self.filter_bandwidth.value(),
            output_board_type=self.output_board_type.currentText(),
        )

    def spec(self) -> Optional[QickRfPulseSpec]:
        if not self.isChecked():
            return None
        return self.configured_spec()

    def settings_dict(self) -> dict:
        spec = self.configured_spec()
        return {
            "enabled": self.isChecked(),
            "gen_ch": spec.gen_ch,
            "segment_name": spec.segment_name,
            "delay_us": spec.delay_us,
            "duration_us": spec.duration_us,
            "frequency_mhz": spec.frequency_mhz,
            "gain": spec.gain,
            "output_board_type": spec.output_board_type,
            "att1_db": spec.att1_db,
            "att2_db": spec.att2_db,
            "filter_type": spec.filter_type,
            "filter_cutoff": spec.filter_cutoff,
            "filter_bandwidth": spec.filter_bandwidth,
            "phase_degrees": spec.phase_degrees,
            "nqz": spec.nqz,
            "require_within_segment": spec.require_within_segment,
        }

    def load_settings(self, data: dict) -> None:
        if not isinstance(data, dict):
            raise TypeError("each RF output setting must be a JSON object")
        enabled = data.get("enabled", True)
        require_within = data.get("require_within_segment", True)
        if not isinstance(enabled, bool):
            raise TypeError("RF output enabled must be boolean")
        if not isinstance(require_within, bool):
            raise TypeError("RF require_within_segment must be boolean")
        spec = QickRfPulseSpec(
            gen_ch=int(data["gen_ch"]),
            segment_name=str(data["segment_name"]),
            delay_us=float(data["delay_us"]),
            duration_us=float(data["duration_us"]),
            frequency_mhz=float(data["frequency_mhz"]),
            gain=int(data["gain"]),
            att1_db=float(data["att1_db"]),
            att2_db=float(data["att2_db"]),
            phase_degrees=float(data.get("phase_degrees", 0.0)),
            nqz=int(data.get("nqz", 1)),
            require_within_segment=require_within,
            filter_type=str(data.get("filter_type", "bypass")),
            filter_cutoff=float(data.get("filter_cutoff", 2.5)),
            filter_bandwidth=float(data.get("filter_bandwidth", 1.0)),
            output_board_type=str(data.get("output_board_type", "RF_Out")),
        )
        segment = self.segment.findData(spec.segment_name)
        if segment < 0:
            raise ValueError(f"unknown RF output anchor {spec.segment_name!r}")
        self.gen_ch.setValue(spec.gen_ch)
        self.segment.setCurrentIndex(segment)
        self.delay.setValue(_time_from_ns(spec.delay_us * 1000.0, self._time_unit))
        self.duration.setValue(
            _time_from_ns(spec.duration_us * 1000.0, self._time_unit)
        )
        self.frequency_mhz.setValue(spec.frequency_mhz)
        self.gain.setValue(spec.gain)
        self.output_board_type.setCurrentText(spec.output_board_type)
        self.att1_db.setValue(spec.att1_db)
        self.att2_db.setValue(spec.att2_db)
        self.filter_type.setCurrentText(spec.filter_type)
        self.filter_cutoff.setValue(spec.filter_cutoff)
        self.filter_bandwidth.setValue(spec.filter_bandwidth)
        self.phase_degrees.setValue(spec.phase_degrees)
        self.nqz.setValue(spec.nqz)
        self.require_within.setChecked(spec.require_within_segment)
        self._update_board_controls()
        self.setChecked(enabled)


class RfPortsPanel(QtWidgets.QWidget):
    """Add/remove editor for up to eight normal QICK RF outputs."""

    specs_changed = QtCore.pyqtSignal(object)
    front_panel_requested = QtCore.pyqtSignal(object)
    MAX_PORTS = 8

    def __init__(self, pulse: PulseSequence, *, time_unit: str, parent=None):
        super().__init__(parent)
        self._pulse = pulse
        self._time_unit = time_unit
        self._panels: List[RfPulsePortPanel] = []
        self._front_panel_configuration = None
        layout = QtWidgets.QVBoxLayout(self)
        self._scroll = QtWidgets.QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._content = QtWidgets.QWidget(self._scroll)
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.addStretch(1)
        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll)
        self.add_button = QtWidgets.QPushButton("Add RF Port")
        self.add_button.clicked.connect(lambda: self.add_port())
        layout.addWidget(self.add_button)
        self.add_port()

    def add_port(
        self,
        spec: Optional[QickRfPulseSpec] = None,
        *,
        settings: Optional[dict] = None,
    ) -> None:
        if len(self._panels) >= self.MAX_PORTS:
            return
        if spec is not None and settings is not None:
            raise ValueError("use either RF spec or RF settings, not both")
        panel = RfPulsePortPanel(
            self._pulse,
            len(self._panels),
            default_gen_ch=DEFAULT_QSTL_RF_CHANNELS[len(self._panels)],
            time_unit=self._time_unit,
            parent=self._content,
        )
        panel.changed.connect(self._emit_specs)
        panel.remove_requested.connect(self.remove_port)
        panel.front_panel_requested.connect(self.front_panel_requested.emit)
        if self._front_panel_configuration is not None:
            panel.set_front_panel_configuration(self._front_panel_configuration)
        self._content_layout.insertWidget(self._content_layout.count() - 1, panel)
        self._panels.append(panel)
        if spec is not None:
            self._load_spec(panel, spec)
        elif settings is not None:
            panel.load_settings(settings)
        self.add_button.setEnabled(len(self._panels) < self.MAX_PORTS)
        self._emit_specs()

    def remove_port(self, panel: RfPulsePortPanel) -> None:
        if panel not in self._panels:
            return
        self._panels.remove(panel)
        panel.deleteLater()
        for index, current in enumerate(self._panels):
            current.set_index(index)
        self.add_button.setEnabled(True)
        self._emit_specs()

    def _load_spec(self, panel: RfPulsePortPanel, spec: QickRfPulseSpec) -> None:
        panel.setChecked(True)
        panel.gen_ch.setValue(spec.gen_ch)
        segment = panel.segment.findData(spec.segment_name)
        if segment >= 0:
            panel.segment.setCurrentIndex(segment)
        panel.delay.setValue(
            _time_from_ns(spec.delay_us * 1000.0, panel._time_unit)
        )
        panel.duration.setValue(
            _time_from_ns(spec.duration_us * 1000.0, panel._time_unit)
        )
        panel.frequency_mhz.setValue(spec.frequency_mhz)
        panel.gain.setValue(spec.gain)
        panel.output_board_type.setCurrentText(spec.output_board_type)
        panel.att1_db.setValue(spec.att1_db)
        panel.att2_db.setValue(spec.att2_db)
        panel.phase_degrees.setValue(spec.phase_degrees)
        panel.nqz.setValue(spec.nqz)
        panel.require_within.setChecked(spec.require_within_segment)

    def apply_path_settings(self, values: Mapping[str, object]) -> int:
        """Apply a committed RF path to the matching Experiment output editor."""
        output_ch = int(values["output_ch"])
        target = next(
            (panel for panel in self._panels if panel.gen_ch.value() == output_ch),
            self._panels[0] if self._panels else None,
        )
        if target is None:
            self.add_port()
            target = self._panels[0]
        target.gen_ch.setValue(output_ch)
        target.output_board_type.setCurrentText(str(values["output_board_type"]))
        target.att1_db.setValue(float(values["output_att1_db"]))
        target.att2_db.setValue(float(values["output_att2_db"]))
        target.filter_type.setCurrentText(str(values["output_filter_type"]))
        target.filter_cutoff.setValue(float(values["output_filter_cutoff_ghz"]))
        target.filter_bandwidth.setValue(
            float(values["output_filter_bandwidth_ghz"])
        )
        target.nqz.setValue(int(values.get("output_nqz", target.nqz.value())))
        target._update_board_controls()
        self._emit_specs()
        return self._panels.index(target)

    def set_front_panel_configuration(self, configuration) -> None:
        self._front_panel_configuration = configuration
        for panel in self._panels:
            panel.set_front_panel_configuration(configuration)

    def specs(self) -> Tuple[QickRfPulseSpec, ...]:
        specs = tuple(spec for panel in self._panels if (spec := panel.spec()) is not None)
        channels = tuple(spec.gen_ch for spec in specs)
        if len(set(channels)) != len(channels):
            raise ValueError("enabled RF output ports must use unique generator indices")
        return specs

    def configured_specs(self) -> Tuple[QickRfPulseSpec, ...]:
        """Return every editor value, including currently disabled outputs."""
        return tuple(panel.configured_spec() for panel in self._panels)

    def settings(self) -> Tuple[dict, ...]:
        return tuple(panel.settings_dict() for panel in self._panels)

    def load_settings(self, entries: Sequence[dict]) -> None:
        entries = tuple(entries)
        if len(entries) > self.MAX_PORTS:
            raise ValueError(f"at most {self.MAX_PORTS} RF output ports are supported")
        with QtCore.QSignalBlocker(self):
            for panel in self._panels:
                self._content_layout.removeWidget(panel)
                panel.setParent(None)
                panel.deleteLater()
            self._panels.clear()
            for entry in entries:
                self.add_port(settings=entry)
            self.add_button.setEnabled(len(self._panels) < self.MAX_PORTS)
        self._emit_specs()

    def _emit_specs(self, *_args) -> None:
        try:
            specs = self.specs()
        except (TypeError, ValueError):
            return
        self.specs_changed.emit(specs)

    def refresh_segments(self, pulse: Optional[PulseSequence] = None) -> None:
        if pulse is not None:
            self._pulse = pulse
        for panel in self._panels:
            panel.refresh_segments(self._pulse)
        self._emit_specs()

    def set_time_unit(self, unit: str) -> None:
        self._time_unit = unit
        for panel in self._panels:
            panel.set_time_unit(unit)


class RfReadoutPanel(QtWidgets.QGroupBox):
    """RF input and FIR-decimated DDR capture configuration."""

    spec_changed = QtCore.pyqtSignal(object)
    front_panel_requested = QtCore.pyqtSignal(object)

    def __init__(self, pulse: PulseSequence, *, time_unit: str, parent=None):
        super().__init__("RF Readout 1", parent)
        self._pulse = pulse
        self._time_unit = time_unit
        self._front_panel_configuration = None
        self.setCheckable(True)
        self.setChecked(False)
        form = QtWidgets.QFormLayout(self)
        self.ro_ch = QtWidgets.QSpinBox()
        self.ro_ch.setRange(0, 255)
        self.input_board_type = QtWidgets.QComboBox()
        self.input_board_type.addItems(QICK_INPUT_BOARD_TYPES)
        self.segment = QtWidgets.QComboBox()
        self.delay = QtWidgets.QDoubleSpinBox()
        self.delay.setRange(0.0, 1.0e12)
        self.delay.setDecimals(9)
        self.samples = QtWidgets.QSpinBox()
        self.samples.setRange(1, 10_000_000)
        self.samples.setValue(64)
        self.frequency_mhz = QtWidgets.QDoubleSpinBox()
        self.frequency_mhz.setRange(-10000.0, 10000.0)
        self.frequency_mhz.setDecimals(6)
        self.frequency_mhz.setValue(50.0)
        self.frequency_mhz.setSuffix(" MHz")
        self.attenuation_db = QtWidgets.QDoubleSpinBox()
        self.attenuation_db.setRange(0.0, 31.75)
        self.attenuation_db.setDecimals(2)
        self.attenuation_db.setSingleStep(0.25)
        self.attenuation_db.setValue(20.0)
        self.attenuation_db.setSuffix(" dB")
        self.dc_gain_db = QtWidgets.QDoubleSpinBox()
        self.dc_gain_db.setRange(-6.0, 26.0)
        self.dc_gain_db.setDecimals(2)
        self.dc_gain_db.setSingleStep(1.0)
        self.dc_gain_db.setValue(0.0)
        self.dc_gain_db.setSuffix(" dB")
        self.dc_measure_mode = QtWidgets.QCheckBox("Convert FIR I/Q to current")
        self.dc_measure_mode.setToolTip(
            "DC_In only: use identity ADC-to-voltage conversion, then divide "
            "I and Q by the measurement gain"
        )
        self.dc_measure_gain_v_per_a = QtWidgets.QDoubleSpinBox()
        self.dc_measure_gain_v_per_a.setRange(1.0e-9, 1.0e15)
        self.dc_measure_gain_v_per_a.setDecimals(6)
        self.dc_measure_gain_v_per_a.setValue(
            DEFAULT_DC_MEASURE_GAIN_V_PER_A
        )
        self.dc_measure_gain_v_per_a.setSuffix(" V/A")
        self.dc_measure_gain_v_per_a.setToolTip(
            "Current conversion: I_current = I_voltage / gain and "
            "Q_current = Q_voltage / gain"
        )
        self.input_condition_stack = QtWidgets.QStackedWidget()
        self.input_condition_stack.addWidget(self.attenuation_db)
        self.input_condition_stack.addWidget(self.dc_gain_db)
        self.filter_type = QtWidgets.QComboBox()
        self.filter_type.addItems(["bypass", "lowpass", "highpass", "bandpass"])
        self.filter_cutoff = QtWidgets.QDoubleSpinBox()
        self.filter_cutoff.setRange(0.0, 100.0)
        self.filter_cutoff.setDecimals(6)
        self.filter_cutoff.setValue(2.5)
        self.filter_cutoff.setSuffix(" GHz")
        self.filter_bandwidth = QtWidgets.QDoubleSpinBox()
        self.filter_bandwidth.setRange(0.001, 100.0)
        self.filter_bandwidth.setDecimals(6)
        self.filter_bandwidth.setValue(1.0)
        self.filter_bandwidth.setSuffix(" GHz")
        self.nqz = QtWidgets.QSpinBox()
        self.nqz.setRange(1, 2)
        self.nqz.setValue(1)
        self.margin_samples = QtWidgets.QSpinBox()
        self.margin_samples.setRange(0, 10_000_000)
        self.margin_samples.setValue(1024)
        self.force_overwrite = QtWidgets.QCheckBox("Allow overwrite of reserved DDR range")
        self.post_run_read_delay = QtWidgets.QDoubleSpinBox()
        self.post_run_read_delay.setRange(0.0, 60.0)
        self.post_run_read_delay.setDecimals(6)
        self.post_run_read_delay.setValue(0.1)
        self.post_run_read_delay.setSuffix(" s")
        self.post_run_read_delay.setToolTip(
            "Wait after the tProcessor program completes before reading DDR"
        )
        self.dc_voltage_calibration_enabled = QtWidgets.QCheckBox(
            "Apply DC input voltage calibration"
        )
        self.dc_voltage_calibration_path = QtWidgets.QLineEdit()
        self.dc_voltage_calibration_path.setPlaceholderText(
            "QCoDeS calibration DB containing a DC Voltage run"
        )
        self.dc_voltage_calibration_browse = QtWidgets.QToolButton()
        self.dc_voltage_calibration_browse.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)
        )
        self.dc_voltage_calibration_browse.setToolTip(
            "Choose DC voltage calibration database"
        )
        self.dc_voltage_calibration_run_id = QtWidgets.QSpinBox()
        self.dc_voltage_calibration_run_id.setRange(0, (1 << 31) - 1)
        self.dc_voltage_calibration_run_id.setSpecialValueText(
            "Latest matching channel/gain"
        )

        self.front_panel_preview = QickFrontPanelPreview(self)
        self.front_panel_preview.activated.connect(
            lambda: self.front_panel_requested.emit(self)
        )
        form.addRow("Front panel:", self.front_panel_preview)
        form.addRow("Readout index:", self.ro_ch)
        form.addRow("Input board:", self.input_board_type)
        form.addRow("Anchor SET:", self.segment)
        self._delay_label = QtWidgets.QLabel()
        form.addRow(self._delay_label, self.delay)
        form.addRow("Stored 1 MSPS samples:", self.samples)
        form.addRow("Readout/DDC frequency:", self.frequency_mhz)
        self.input_condition_label = QtWidgets.QLabel("Input attenuation:")
        form.addRow(self.input_condition_label, self.input_condition_stack)
        form.addRow(self.dc_measure_mode)
        form.addRow("DC measurement gain:", self.dc_measure_gain_v_per_a)
        form.addRow("Input filter:", self.filter_type)
        form.addRow("Filter cutoff/center:", self.filter_cutoff)
        form.addRow("Filter bandwidth:", self.filter_bandwidth)
        form.addRow("FIR input margin:", self.margin_samples)
        form.addRow("DDR read delay after run:", self.post_run_read_delay)
        form.addRow(self.force_overwrite)
        note = QtWidgets.QLabel(
            "The qstl_awg_tuning_fir path stores post-FIR samples at 1 MSPS."
        )
        note.setWordWrap(True)
        form.addRow(note)

        self.refresh_segments(pulse)
        self.set_time_unit(time_unit, force=True)
        for widget in (
            self.ro_ch,
            self.input_board_type,
            self.segment,
            self.delay,
            self.samples,
            self.frequency_mhz,
            self.attenuation_db,
            self.dc_gain_db,
            self.dc_measure_mode,
            self.dc_measure_gain_v_per_a,
            self.dc_voltage_calibration_enabled,
            self.dc_voltage_calibration_path,
            self.dc_voltage_calibration_run_id,
            self.filter_type,
            self.filter_cutoff,
            self.filter_bandwidth,
            self.nqz,
            self.margin_samples,
            self.post_run_read_delay,
            self.force_overwrite,
        ):
            signal = getattr(widget, "valueChanged", None)
            if signal is None:
                signal = getattr(widget, "currentIndexChanged", None)
            if signal is None:
                signal = getattr(widget, "toggled", None)
            if signal is None:
                signal = getattr(widget, "textChanged", None)
            signal.connect(self._emit_spec)
        self.toggled.connect(self._emit_spec)
        self.input_board_type.currentTextChanged.connect(
            self._update_board_controls
        )
        self.dc_measure_mode.toggled.connect(self._update_board_controls)
        self.dc_voltage_calibration_enabled.toggled.connect(
            self._update_board_controls
        )
        self.dc_voltage_calibration_browse.clicked.connect(
            self._browse_dc_voltage_calibration
        )
        self.ro_ch.valueChanged.connect(self._sync_front_panel_selection)
        self._update_board_controls()

    def _update_board_controls(self, *_args) -> None:
        rf_input = self.input_board_type.currentText() == "RF_In"
        if rf_input and self.dc_measure_mode.isChecked():
            with QtCore.QSignalBlocker(self.dc_measure_mode):
                self.dc_measure_mode.setChecked(False)
        if rf_input and self.dc_voltage_calibration_enabled.isChecked():
            with QtCore.QSignalBlocker(self.dc_voltage_calibration_enabled):
                self.dc_voltage_calibration_enabled.setChecked(False)
        self.input_condition_stack.setCurrentIndex(0 if rf_input else 1)
        self.input_condition_label.setText(
            "Input attenuation:" if rf_input else "DC input gain:"
        )
        self.filter_type.setEnabled(rf_input)
        self.filter_cutoff.setEnabled(rf_input)
        self.filter_bandwidth.setEnabled(rf_input)
        self.dc_measure_mode.setEnabled(not rf_input)
        self.dc_measure_gain_v_per_a.setEnabled(
            not rf_input and self.dc_measure_mode.isChecked()
        )
        calibration_enabled = (
            not rf_input and self.dc_voltage_calibration_enabled.isChecked()
        )
        self.dc_voltage_calibration_enabled.setEnabled(not rf_input)
        self.dc_voltage_calibration_path.setEnabled(calibration_enabled)
        self.dc_voltage_calibration_browse.setEnabled(calibration_enabled)
        self.dc_voltage_calibration_run_id.setEnabled(calibration_enabled)
        tooltip = (
            "RF_In onboard filter"
            if rf_input
            else "DC_In has no onboard RF filter; these values are ignored"
        )
        self.filter_type.setToolTip(tooltip)
        self.filter_cutoff.setToolTip(tooltip)
        self.filter_bandwidth.setToolTip(tooltip)

    def _browse_dc_voltage_calibration(self) -> None:
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose DC voltage calibration database",
            self.dc_voltage_calibration_path.text().strip(),
            "QCoDeS SQLite database (*.db)",
        )
        if path:
            self.dc_voltage_calibration_path.setText(path)

    def set_front_panel_configuration(self, configuration) -> None:
        self._front_panel_configuration = configuration
        self.front_panel_preview.set_configuration(configuration)
        self._sync_front_panel_selection()

    def _sync_front_panel_selection(self, *_args) -> None:
        self.front_panel_preview.set_channels(input_ch=self.ro_ch.value())
        if self._front_panel_configuration is None:
            return
        for port in self._front_panel_configuration.inputs:
            if self.ro_ch.value() in port.qick_channels:
                if port.board_type in QICK_INPUT_BOARD_TYPES:
                    self.input_board_type.setCurrentText(port.board_type)
                return

    def apply_front_panel_settings(self, values: Mapping[str, object]) -> None:
        self.ro_ch.setValue(int(values["readout_ch"]))
        self.input_board_type.setCurrentText(str(values["input_board_type"]))
        self.attenuation_db.setValue(float(values["readout_attenuation_db"]))
        self.dc_gain_db.setValue(float(values["readout_dc_gain_db"]))
        self.filter_type.setCurrentText(str(values["readout_filter_type"]))
        self.filter_cutoff.setValue(float(values["readout_filter_cutoff_ghz"]))
        self.filter_bandwidth.setValue(
            float(values["readout_filter_bandwidth_ghz"])
        )
        self.nqz.setValue(int(values.get("readout_nqz", self.nqz.value())))
        self._update_board_controls()
        self._sync_front_panel_selection()
        self._emit_spec()

    def set_time_unit(self, unit: str, *, force: bool = False) -> None:
        if unit not in TIME_UNIT_NS:
            raise ValueError(f"unsupported time unit {unit!r}")
        if not force and unit == self._time_unit:
            return
        delay_ns = _time_to_ns(self.delay.value(), self._time_unit)
        self._time_unit = unit
        with QtCore.QSignalBlocker(self.delay):
            self.delay.setValue(_time_from_ns(delay_ns, unit))
            self.delay.setSuffix(f" {unit}")
        self._delay_label.setText(f"Trigger delay [{unit}]:")

    def refresh_segments(self, pulse: Optional[PulseSequence] = None) -> None:
        if pulse is not None:
            self._pulse = pulse
        previous = self.segment.currentData()
        with QtCore.QSignalBlocker(self.segment):
            self.segment.clear()
            for index, _segment in enumerate(self._pulse.flat_segments()):
                name = f"set_{index}"
                self.segment.addItem(name, name)
            match = self.segment.findData(previous)
            if match >= 0:
                self.segment.setCurrentIndex(match)

    def configured_spec(self) -> QickDdrReadoutSpec:
        """Return the editor values even when capture is disabled."""
        return QickDdrReadoutSpec(
            ro_ch=self.ro_ch.value(),
            segment_name=str(self.segment.currentData()),
            delay_us=_time_to_ns(self.delay.value(), self._time_unit) / 1000.0,
            samples_per_trigger=self.samples.value(),
            readout_frequency_mhz=self.frequency_mhz.value(),
            margin_input_samples=self.margin_samples.value(),
            force_overwrite=self.force_overwrite.isChecked(),
            post_run_read_delay_seconds=self.post_run_read_delay.value(),
            attenuation_db=self.attenuation_db.value(),
            filter_type=self.filter_type.currentText(),
            filter_cutoff=self.filter_cutoff.value(),
            filter_bandwidth=self.filter_bandwidth.value(),
            input_board_type=self.input_board_type.currentText(),
            dc_gain_db=self.dc_gain_db.value(),
            dc_measure_mode=self.dc_measure_mode.isChecked(),
            dc_measure_gain_v_per_a=self.dc_measure_gain_v_per_a.value(),
            dc_voltage_calibration_enabled=(
                self.dc_voltage_calibration_enabled.isChecked()
            ),
            dc_voltage_calibration_database_path=(
                self.dc_voltage_calibration_path.text().strip()
            ),
            dc_voltage_calibration_run_id=(
                self.dc_voltage_calibration_run_id.value()
            ),
            nqz=self.nqz.value(),
        )

    def spec(self) -> Optional[QickDdrReadoutSpec]:
        if not self.isChecked():
            return None
        return self.configured_spec()

    def set_dc_measurement(self, enabled: bool, gain_v_per_a: float) -> None:
        """Update a readout's DC current conversion from linked controls."""
        if enabled and self.input_board_type.currentText() != "DC_In":
            raise ValueError("DC measure mode requires the DC_In input board")
        with QtCore.QSignalBlocker(self.dc_measure_mode):
            self.dc_measure_mode.setChecked(bool(enabled))
        with QtCore.QSignalBlocker(self.dc_measure_gain_v_per_a):
            self.dc_measure_gain_v_per_a.setValue(float(gain_v_per_a))
        self._update_board_controls()
        self._emit_spec()

    def set_dc_voltage_calibration(
        self,
        database_path: str,
        run_id: int,
    ) -> None:
        """Select a completed DC calibration for subsequent measurements."""
        if self.input_board_type.currentText() != "DC_In":
            self.input_board_type.setCurrentText("DC_In")
        self.set_dc_voltage_calibration_selection(
            True,
            database_path,
            run_id,
        )

    def set_dc_voltage_calibration_selection(
        self,
        enabled: bool,
        database_path: str,
        run_id: int,
    ) -> None:
        """Update a readout's DC calibration from linked controls."""
        if enabled and self.input_board_type.currentText() != "DC_In":
            raise ValueError("DC voltage calibration requires the DC_In input board")
        with QtCore.QSignalBlocker(self.dc_voltage_calibration_path):
            self.dc_voltage_calibration_path.setText(str(database_path))
        with QtCore.QSignalBlocker(self.dc_voltage_calibration_run_id):
            self.dc_voltage_calibration_run_id.setValue(int(run_id))
        with QtCore.QSignalBlocker(self.dc_voltage_calibration_enabled):
            self.dc_voltage_calibration_enabled.setChecked(bool(enabled))
        self._update_board_controls()
        self._emit_spec()

    def settings_dict(self) -> dict:
        spec = self.configured_spec()
        return {
            "enabled": self.isChecked(),
            "ro_ch": spec.ro_ch,
            "segment_name": spec.segment_name,
            "delay_us": spec.delay_us,
            "samples_per_trigger": spec.samples_per_trigger,
            "readout_frequency_mhz": spec.readout_frequency_mhz,
            "margin_input_samples": spec.margin_input_samples,
            "force_overwrite": spec.force_overwrite,
            "post_run_read_delay_seconds": spec.post_run_read_delay_seconds,
            "input_board_type": spec.input_board_type,
            "attenuation_db": spec.attenuation_db,
            "dc_gain_db": spec.dc_gain_db,
            "dc_measure_mode": spec.dc_measure_mode,
            "dc_measure_gain_v_per_a": spec.dc_measure_gain_v_per_a,
            "dc_voltage_calibration_enabled": (
                spec.dc_voltage_calibration_enabled
            ),
            "dc_voltage_calibration_database_path": (
                spec.dc_voltage_calibration_database_path
            ),
            "dc_voltage_calibration_run_id": (
                spec.dc_voltage_calibration_run_id
            ),
            "filter_type": spec.filter_type,
            "filter_cutoff": spec.filter_cutoff,
            "filter_bandwidth": spec.filter_bandwidth,
            "nqz": spec.nqz,
        }

    def load_settings(self, data: dict) -> None:
        if not isinstance(data, dict):
            raise TypeError("RF readout setting must be a JSON object")
        enabled = data.get("enabled", False)
        force_overwrite = data.get("force_overwrite", False)
        if not isinstance(enabled, bool):
            raise TypeError("RF readout enabled must be boolean")
        if not isinstance(force_overwrite, bool):
            raise TypeError("RF readout force_overwrite must be boolean")
        dc_measure_mode = data.get("dc_measure_mode", False)
        if not isinstance(dc_measure_mode, bool):
            raise TypeError("RF readout dc_measure_mode must be boolean")
        spec = QickDdrReadoutSpec(
            ro_ch=int(data["ro_ch"]),
            segment_name=str(data["segment_name"]),
            delay_us=float(data["delay_us"]),
            samples_per_trigger=int(data["samples_per_trigger"]),
            readout_frequency_mhz=float(data.get("readout_frequency_mhz", 0.0)),
            margin_input_samples=int(data.get("margin_input_samples", 1024)),
            force_overwrite=force_overwrite,
            post_run_read_delay_seconds=float(
                data.get("post_run_read_delay_seconds", 0.1)
            ),
            attenuation_db=float(data.get("attenuation_db", 20.0)),
            filter_type=str(data.get("filter_type", "bypass")),
            filter_cutoff=float(data.get("filter_cutoff", 2.5)),
            filter_bandwidth=float(data.get("filter_bandwidth", 1.0)),
            input_board_type=str(data.get("input_board_type", "RF_In")),
            dc_gain_db=float(data.get("dc_gain_db", 0.0)),
            dc_measure_mode=dc_measure_mode,
            dc_measure_gain_v_per_a=float(
                data.get(
                    "dc_measure_gain_v_per_a",
                    DEFAULT_DC_MEASURE_GAIN_V_PER_A,
                )
            ),
            dc_voltage_calibration_enabled=bool(
                data.get("dc_voltage_calibration_enabled", False)
            ),
            dc_voltage_calibration_database_path=str(
                data.get("dc_voltage_calibration_database_path", "")
            ),
            dc_voltage_calibration_run_id=int(
                data.get("dc_voltage_calibration_run_id", 0)
            ),
            nqz=int(data.get("nqz", 1)),
        )
        segment = self.segment.findData(spec.segment_name)
        if segment < 0:
            raise ValueError(f"unknown RF readout anchor {spec.segment_name!r}")
        filter_index = self.filter_type.findText(spec.filter_type)
        if filter_index < 0:
            raise ValueError(f"unknown RF input filter {spec.filter_type!r}")
        with QtCore.QSignalBlocker(self):
            self.ro_ch.setValue(spec.ro_ch)
            self.segment.setCurrentIndex(segment)
            self.delay.setValue(
                _time_from_ns(spec.delay_us * 1000.0, self._time_unit)
            )
            self.samples.setValue(spec.samples_per_trigger)
            self.frequency_mhz.setValue(spec.readout_frequency_mhz)
            self.input_board_type.setCurrentText(spec.input_board_type)
            self.margin_samples.setValue(spec.margin_input_samples)
            self.force_overwrite.setChecked(spec.force_overwrite)
            self.post_run_read_delay.setValue(
                spec.post_run_read_delay_seconds
            )
            self.attenuation_db.setValue(spec.attenuation_db)
            self.dc_gain_db.setValue(spec.dc_gain_db)
            self.dc_measure_mode.setChecked(spec.dc_measure_mode)
            self.dc_measure_gain_v_per_a.setValue(
                spec.dc_measure_gain_v_per_a
            )
            self.dc_voltage_calibration_enabled.setChecked(
                spec.dc_voltage_calibration_enabled
            )
            self.dc_voltage_calibration_path.setText(
                spec.dc_voltage_calibration_database_path
            )
            self.dc_voltage_calibration_run_id.setValue(
                spec.dc_voltage_calibration_run_id
            )
            self.filter_type.setCurrentIndex(filter_index)
            self.filter_cutoff.setValue(spec.filter_cutoff)
            self.filter_bandwidth.setValue(spec.filter_bandwidth)
            self.nqz.setValue(spec.nqz)
            self.setChecked(enabled)
        self._update_board_controls()
        self._emit_spec()

    def apply_path_settings(self, values: Mapping[str, object]) -> None:
        """Apply a committed RF path to this Experiment readout editor."""
        self.ro_ch.setValue(int(values["readout_ch"]))
        self.input_board_type.setCurrentText(str(values["input_board_type"]))
        self.attenuation_db.setValue(float(values["readout_attenuation_db"]))
        self.dc_gain_db.setValue(float(values["readout_dc_gain_db"]))
        self.filter_type.setCurrentText(str(values["readout_filter_type"]))
        self.filter_cutoff.setValue(float(values["readout_filter_cutoff_ghz"]))
        self.filter_bandwidth.setValue(
            float(values["readout_filter_bandwidth_ghz"])
        )
        self.nqz.setValue(int(values.get("readout_nqz", self.nqz.value())))
        self._update_board_controls()
        self._emit_spec()

    def _emit_spec(self, *_args) -> None:
        try:
            spec = self.spec()
        except (TypeError, ValueError):
            return
        self.spec_changed.emit(spec)


class ExperimentPanel(QtWidgets.QWidget):
    """QICK connection, QCoDeS database, and direct-run controls."""

    run_requested = QtCore.pyqtSignal()
    show_program_requested = QtCore.pyqtSignal()
    bias_t_changed = QtCore.pyqtSignal(bool, str, float, str, float, float)

    def __init__(
        self,
        *,
        fabric_mhz: float,
        tproc_mhz: float,
        full_scale_mv: float,
        awg_channels: Sequence[int],
        repetitions: int,
        bias_t_enabled: bool = False,
        bias_t_compensation_type: str = "dc",
        bias_t_compensation_mv: Optional[float] = None,
        bias_t_mode: str = "fixed_voltage",
        bias_t_duration_us: float = DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
        bias_t_filter_tau_us: float = DEFAULT_BIAS_T_FILTER_TAU_US,
        parent=None,
    ):
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget(scroll)
        form = QtWidgets.QFormLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        self.qick_host = QtWidgets.QLineEdit(DEFAULT_QICK_HOST)
        self.ns_port = QtWidgets.QSpinBox()
        self.ns_port.setRange(1, 65535)
        self.ns_port.setValue(DEFAULT_QICK_NS_PORT)
        self.proxy_name = QtWidgets.QLineEdit(DEFAULT_QICK_PROXY_NAME)

        self.database_path = QtWidgets.QLineEdit(DEFAULT_QCODES_DB_PATH)
        browse_database = QtWidgets.QToolButton()
        browse_database.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        )
        browse_database.setToolTip("Choose QCoDeS SQLite database")
        browse_database.clicked.connect(self._browse_database)
        database_row = QtWidgets.QHBoxLayout()
        database_row.addWidget(self.database_path, 1)
        database_row.addWidget(browse_database)

        self.experiment_name = QtWidgets.QLineEdit("QICK pulse experiment")
        self.sample_name = QtWidgets.QLineEdit("PulseGenerator")
        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setPlaceholderText("Optional experiment notes")
        self.notes.setMaximumHeight(90)

        self.fabric_mhz = QtWidgets.QDoubleSpinBox()
        self.fabric_mhz.setRange(1.0, 5000.0)
        self.fabric_mhz.setDecimals(6)
        self.fabric_mhz.setSuffix(" MHz")
        self.tproc_mhz = QtWidgets.QDoubleSpinBox()
        self.tproc_mhz.setRange(1.0, 5000.0)
        self.tproc_mhz.setDecimals(6)
        self.tproc_mhz.setSuffix(" MHz")
        self.full_scale_mv = QtWidgets.QDoubleSpinBox()
        self.full_scale_mv.setRange(1.0, 1.0e6)
        self.full_scale_mv.setDecimals(6)
        self.full_scale_mv.setSuffix(" mV")
        self.awg_channels = QtWidgets.QLineEdit()
        self.repetitions = QtWidgets.QSpinBox()
        self.repetitions.setRange(1, 1_000_000)
        self.bias_t_group = QtWidgets.QGroupBox("Bias-T compensation")
        self.bias_t_group.setCheckable(True)
        self.bias_t_group.setChecked(bool(bias_t_enabled))
        bias_t_form = QtWidgets.QFormLayout(self.bias_t_group)
        self.bias_t_type = QtWidgets.QComboBox()
        self.bias_t_type.addItem("DC compensation", "dc")
        self.bias_t_type.addItem("Filter compensation", "filter")
        type_index = self.bias_t_type.findData(str(bias_t_compensation_type))
        if type_index < 0:
            raise ValueError(
                "Bias-T compensation type must be one of "
                f"{BIAS_T_COMPENSATION_TYPES}"
            )
        self.bias_t_type.setCurrentIndex(type_index)
        self.bias_t_mode = QtWidgets.QComboBox()
        self.bias_t_mode.addItem("Fixed voltage (adjust time)", "fixed_voltage")
        self.bias_t_mode.addItem("Fixed time (adjust voltage)", "fixed_time")
        mode_index = self.bias_t_mode.findData(str(bias_t_mode))
        if mode_index < 0:
            raise ValueError(
                f"Bias-T compensation mode must be one of {BIAS_T_COMPENSATION_MODES}"
            )
        self.bias_t_mode.setCurrentIndex(mode_index)
        self.bias_t_compensation_mv = QtWidgets.QDoubleSpinBox()
        self.bias_t_compensation_mv.setRange(0.001, float(full_scale_mv))
        self.bias_t_compensation_mv.setDecimals(6)
        self.bias_t_compensation_mv.setSuffix(" mV")
        self.bias_t_compensation_mv.setValue(
            float(
                full_scale_mv * DEFAULT_BIAS_T_COMPENSATION_FRACTION
                if bias_t_compensation_mv is None
                else bias_t_compensation_mv
            )
        )
        self.bias_t_duration_us = QtWidgets.QDoubleSpinBox()
        self.bias_t_duration_us.setRange(1.0e-6, 1.0e9)
        self.bias_t_duration_us.setDecimals(6)
        self.bias_t_duration_us.setSuffix(" us")
        self.bias_t_duration_us.setValue(float(bias_t_duration_us))
        self.bias_t_filter_tau_us = QtWidgets.QDoubleSpinBox()
        self.bias_t_filter_tau_us.setRange(1.0e-6, 1.0e12)
        self.bias_t_filter_tau_us.setDecimals(6)
        self.bias_t_filter_tau_us.setSuffix(" us")
        self.bias_t_filter_tau_us.setValue(float(bias_t_filter_tau_us))
        bias_t_form.addRow("Compensation type:", self.bias_t_type)
        bias_t_form.addRow("DC control mode:", self.bias_t_mode)
        bias_t_form.addRow("DC voltage:", self.bias_t_compensation_mv)
        bias_t_form.addRow("DC time:", self.bias_t_duration_us)
        bias_t_form.addRow("Filter time constant (tau):", self.bias_t_filter_tau_us)
        self.set_qick_values(
            fabric_mhz=fabric_mhz,
            tproc_mhz=tproc_mhz,
            full_scale_mv=full_scale_mv,
            awg_channels=awg_channels,
            repetitions=repetitions,
        )

        form.addRow("QICK IP/host:", self.qick_host)
        form.addRow("Pyro nameserver port:", self.ns_port)
        form.addRow("Pyro proxy name:", self.proxy_name)
        form.addRow("QCoDeS DB file:", database_row)
        form.addRow("Experiment name:", self.experiment_name)
        form.addRow("Sample name:", self.sample_name)
        form.addRow("AWG fabric clock:", self.fabric_mhz)
        form.addRow("tProcessor clock:", self.tproc_mhz)
        form.addRow("AWG full scale (+/-):", self.full_scale_mv)
        form.addRow("AWG generator indices:", self.awg_channels)
        form.addRow("Repetitions per sweep point:", self.repetitions)
        form.addRow(self.bias_t_group)
        form.addRow("Notes:", self.notes)

        self.full_scale_mv.valueChanged.connect(self._update_bias_t_range)
        self.bias_t_group.toggled.connect(self._emit_bias_t_changed)
        self.bias_t_compensation_mv.valueChanged.connect(self._emit_bias_t_changed)
        self.bias_t_duration_us.valueChanged.connect(self._emit_bias_t_changed)
        self.bias_t_filter_tau_us.valueChanged.connect(self._emit_bias_t_changed)
        self.bias_t_type.currentIndexChanged.connect(self._on_bias_t_mode_changed)
        self.bias_t_mode.currentIndexChanged.connect(self._on_bias_t_mode_changed)
        self._update_bias_t_mode_controls()

        self.run_button = QtWidgets.QPushButton("Run QICK Experiment")
        self.run_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        )
        self.run_button.clicked.connect(self.run_requested.emit)
        self.show_program_button = QtWidgets.QPushButton("Show QICK Program")
        self.show_program_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView)
        )
        self.show_program_button.setToolTip(
            "Compile the current settings and show the tProcessor assembly"
        )
        self.show_program_button.clicked.connect(self.show_program_requested.emit)
        action_row = QtWidgets.QHBoxLayout()
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.show_program_button)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.progress.hide()
        self.run_status = QtWidgets.QLabel("Ready")
        self.run_status.setWordWrap(True)
        form.addRow(action_row)
        form.addRow(self.progress)
        form.addRow("Status:", self.run_status)

    def _update_bias_t_range(self, full_scale_mv: float) -> None:
        self.bias_t_compensation_mv.setMaximum(max(0.001, float(full_scale_mv)))

    def _emit_bias_t_changed(self, *_args) -> None:
        self.bias_t_changed.emit(
            self.bias_t_group.isChecked(),
            str(self.bias_t_type.currentData()),
            self.bias_t_compensation_mv.value(),
            str(self.bias_t_mode.currentData()),
            self.bias_t_duration_us.value(),
            self.bias_t_filter_tau_us.value(),
        )

    def _update_bias_t_mode_controls(self) -> None:
        filter_mode = self.bias_t_type.currentData() == "filter"
        fixed_time = self.bias_t_mode.currentData() == "fixed_time"
        self.bias_t_mode.setEnabled(not filter_mode)
        self.bias_t_compensation_mv.setEnabled(not filter_mode and not fixed_time)
        self.bias_t_duration_us.setEnabled(not filter_mode and fixed_time)
        self.bias_t_filter_tau_us.setEnabled(filter_mode)

    def _on_bias_t_mode_changed(self, *_args) -> None:
        self._update_bias_t_mode_controls()
        self._emit_bias_t_changed()

    def _browse_database(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Choose QCoDeS database",
            self.database_path.text().strip() or DEFAULT_QCODES_DB_PATH,
            "QCoDeS SQLite database (*.db)",
        )
        if path:
            if Path(path).suffix.lower() != ".db":
                path = str(Path(path).with_suffix(".db"))
            self.database_path.setText(path)

    def _parse_awg_channels(self, output_count: int) -> Tuple[int, ...]:
        fields = [field.strip() for field in self.awg_channels.text().split(",")]
        if any(not field for field in fields):
            raise ValueError("AWG generator indices must be comma-separated integers")
        try:
            channels = tuple(int(field) for field in fields)
        except ValueError as exc:
            raise ValueError("AWG generator indices must be integers") from exc
        if len(channels) != output_count:
            raise ValueError(
                f"exactly {output_count} AWG generator indices are required"
            )
        if any(channel < 0 for channel in channels) or len(set(channels)) != len(channels):
            raise ValueError("AWG generator indices must be unique and nonnegative")
        return channels

    def values(
        self,
        output_count: int,
        *,
        require_run_config: bool = True,
        database_path: Optional[str] = None,
    ) -> dict:
        connection, run = self.connection_values(
            require_run_config=require_run_config,
            database_path=database_path,
        )
        return {
            "connection": connection,
            "run": run,
            "fabric_mhz": self.fabric_mhz.value(),
            "tproc_mhz": self.tproc_mhz.value(),
            "full_scale_mv": self.full_scale_mv.value(),
            "awg_channels": self._parse_awg_channels(output_count),
            "repetitions_per_sweep": self.repetitions.value(),
            "bias_t_compensation_enabled": self.bias_t_group.isChecked(),
            "bias_t_compensation_type": str(self.bias_t_type.currentData()),
            "bias_t_compensation_voltage_mv": self.bias_t_compensation_mv.value(),
            "bias_t_compensation_mode": str(self.bias_t_mode.currentData()),
            "bias_t_compensation_duration_us": self.bias_t_duration_us.value(),
            "bias_t_filter_tau_us": self.bias_t_filter_tau_us.value(),
        }

    def connection_values(
        self,
        *,
        require_run_config: bool = True,
        database_path: Optional[str] = None,
    ) -> Tuple[QickConnectionConfig, Optional[QcodesRunConfig]]:
        """Return shared QICK/QCoDeS settings without validating AWG fields."""
        connection = QickConnectionConfig(
            host=self.qick_host.text().strip(),
            ns_port=self.ns_port.value(),
            proxy_name=self.proxy_name.text().strip(),
        )
        run = (
            QcodesRunConfig(
                database_path=(
                    self.database_path.text().strip()
                    if database_path is None
                    else str(database_path).strip()
                ),
                experiment_name=self.experiment_name.text().strip(),
                sample_name=self.sample_name.text().strip(),
                notes=self.notes.toPlainText(),
            )
            if require_run_config
            else None
        )
        return connection, run

    def settings_dict(self, output_count: int) -> dict:
        values = self.values(output_count)
        connection = values["connection"]
        run = values["run"]
        return {
            "qick_host": connection.host,
            "ns_port": connection.ns_port,
            "proxy_name": connection.proxy_name,
            "database_path": run.database_path,
            "experiment_name": run.experiment_name,
            "sample_name": run.sample_name,
            "notes": run.notes,
        }

    def set_qick_values(
        self,
        *,
        fabric_mhz: float,
        tproc_mhz: float,
        full_scale_mv: float,
        awg_channels: Sequence[int],
        repetitions: int,
    ) -> None:
        self.fabric_mhz.setValue(float(fabric_mhz))
        self.tproc_mhz.setValue(float(tproc_mhz))
        self.full_scale_mv.setValue(float(full_scale_mv))
        self.awg_channels.setText(", ".join(str(value) for value in awg_channels))
        self.repetitions.setValue(int(repetitions))

    def set_bias_t_values(
        self,
        *,
        enabled: bool,
        compensation_type: str = "dc",
        compensation_mv: float,
        mode: str = "fixed_voltage",
        duration_us: float = DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
        filter_tau_us: float = DEFAULT_BIAS_T_FILTER_TAU_US,
    ) -> None:
        type_index = self.bias_t_type.findData(str(compensation_type))
        if type_index < 0:
            raise ValueError(
                "Bias-T compensation type must be one of "
                f"{BIAS_T_COMPENSATION_TYPES}"
            )
        mode_index = self.bias_t_mode.findData(str(mode))
        if mode_index < 0:
            raise ValueError(
                f"Bias-T compensation mode must be one of {BIAS_T_COMPENSATION_MODES}"
            )
        with QtCore.QSignalBlocker(self.bias_t_group), QtCore.QSignalBlocker(
            self.bias_t_type
        ), QtCore.QSignalBlocker(self.bias_t_compensation_mv), QtCore.QSignalBlocker(
            self.bias_t_mode
        ), QtCore.QSignalBlocker(self.bias_t_duration_us), QtCore.QSignalBlocker(
            self.bias_t_filter_tau_us
        ):
            self.bias_t_group.setChecked(bool(enabled))
            self.bias_t_type.setCurrentIndex(type_index)
            self.bias_t_compensation_mv.setValue(float(compensation_mv))
            self.bias_t_mode.setCurrentIndex(mode_index)
            self.bias_t_duration_us.setValue(float(duration_us))
            self.bias_t_filter_tau_us.setValue(float(filter_tau_us))
        self._update_bias_t_mode_controls()
        self._emit_bias_t_changed()

    def load_settings(
        self,
        connection: QickConnectionConfig,
        run: QcodesRunConfig,
        *,
        fabric_mhz: float,
        tproc_mhz: float,
        full_scale_mv: float,
        awg_channels: Sequence[int],
        repetitions: int,
        bias_t_enabled: bool = False,
        bias_t_compensation_type: str = "dc",
        bias_t_compensation_mv: float = DEFAULT_BIAS_T_COMPENSATION_MV,
        bias_t_mode: str = "fixed_voltage",
        bias_t_duration_us: float = DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
        bias_t_filter_tau_us: float = DEFAULT_BIAS_T_FILTER_TAU_US,
    ) -> None:
        self.qick_host.setText(connection.host)
        self.ns_port.setValue(connection.ns_port)
        self.proxy_name.setText(connection.proxy_name)
        self.database_path.setText(run.database_path)
        self.experiment_name.setText(run.experiment_name)
        self.sample_name.setText(run.sample_name)
        self.notes.setPlainText(run.notes)
        self.set_qick_values(
            fabric_mhz=fabric_mhz,
            tproc_mhz=tproc_mhz,
            full_scale_mv=full_scale_mv,
            awg_channels=awg_channels,
            repetitions=repetitions,
        )
        self.set_bias_t_values(
            enabled=bias_t_enabled,
            compensation_type=bias_t_compensation_type,
            compensation_mv=bias_t_compensation_mv,
            mode=bias_t_mode,
            duration_us=bias_t_duration_us,
            filter_tau_us=bias_t_filter_tau_us,
        )

    def set_running(
        self, running: bool, message: str, *, show_progress: bool = True
    ) -> None:
        self.run_button.setEnabled(not running)
        self.show_program_button.setEnabled(not running)
        if running:
            self.progress.setValue(0)
        self.progress.setVisible(running and show_progress)
        self.run_status.setText(message)

    def update_progress(self, percent: int, message: str) -> None:
        percent = max(0, min(100, int(percent)))
        self.progress.setValue(percent)
        self.run_status.setText(f"{percent}% - {message}")

    def show_result(self, result) -> None:
        output_details = tuple(result.rf_settings.get("output_details", ()))
        rf_summary = ""
        if output_details:
            entries = [
                (
                    f"gen {item['gen_ch']}: ATT1/ATT2 "
                    f"{item['commanded_att1_db']:.2f}/{item['commanded_att2_db']:.2f} dB, "
                    f"{item['filter_type']} filter"
                )
                for item in output_details
            ]
            rf_summary = "\nRF applied: " + "; ".join(entries)
        self.set_running(
            False,
            f"Run {result.run_id}, {result.row_count} IQ samples\n"
            f"{result.database_path}{rf_summary}",
        )


class QickExperimentWorker(QtCore.QObject):
    """Run blocking Pyro/QICK/QCoDeS work outside the GUI thread."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress_changed = QtCore.pyqtSignal(int, str)

    def __init__(self, kwargs: dict, parent=None):
        super().__init__(parent)
        self._kwargs = kwargs

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            kwargs = dict(self._kwargs)
            kwargs["progress_callback"] = self.progress_changed.emit
            result = run_qick_qcodes_experiment(**kwargs)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(result)


class QickProgramWorker(QtCore.QObject):
    """Compile a QICK program from live HWH metadata without running it."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, connection_config, program_kwargs: dict, parent=None):
        super().__init__(parent)
        self._connection_config = connection_config
        self._program_kwargs = dict(program_kwargs)

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            _soc, soccfg = connect_qick(self._connection_config)
            program = build_qick_program(soccfg, **self._program_kwargs)
            program.compile()
            summary = program.summary() if hasattr(program, "summary") else {}
            result = {
                "assembly": program.asm(),
                "instruction_count": len(program.prog_list),
                "machine_word_count": len(program.binprog),
                "summary": summary,
                "program": program,
            }
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(result)


class QickConfigurationWorker(QtCore.QObject):
    """Fetch the live HWH-derived config and reconstruct physical SMA routing."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, connection_config, parent=None):
        super().__init__(parent)
        self._connection_config = connection_config

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            _soc, soccfg = connect_qick(self._connection_config)
            configuration = identify_qick_front_panel(soccfg)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(configuration)


class DetailedErrorMessageBox(QtWidgets.QMessageBox):
    """Error message with expandable traceback and one-click clipboard copy."""

    def __init__(self, title: str, summary: str, details: str, parent=None):
        super().__init__(parent)
        self._copy_text = f"{title}\n{summary}\n\n{details}".strip()
        self.setIcon(QtWidgets.QMessageBox.Critical)
        self.setWindowTitle(title)
        self.setText(summary)
        self.setDetailedText(details)
        self.setStandardButtons(QtWidgets.QMessageBox.Close)
        self.copy_button = self.addButton(
            "Copy Details", QtWidgets.QMessageBox.ActionRole
        )
        self.copy_button.setToolTip("Copy the complete error and traceback")
        self.copy_button.clicked.connect(self.copy_details)

    def copy_details(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self._copy_text)


class QickAssemblyDialog(QtWidgets.QDialog):
    """Read-only tProcessor assembly viewer with copy and save actions."""

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self._assembly = str(result["assembly"])
        self.setWindowTitle("QICK tProcessor Assembly")
        self.resize(1000, 760)

        layout = QtWidgets.QVBoxLayout(self)
        instruction_count = int(result.get("instruction_count", 0))
        machine_word_count = int(result.get("machine_word_count", 0))
        self.summary_label = QtWidgets.QLabel(
            f"{instruction_count:,} assembly instructions, "
            f"{machine_word_count:,} compiled machine words"
        )
        layout.addWidget(self.summary_label)

        self.assembly_text = QtWidgets.QPlainTextEdit(self)
        self.assembly_text.setReadOnly(True)
        self.assembly_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.assembly_text.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        )
        self.assembly_text.setPlainText(self._assembly)
        layout.addWidget(self.assembly_text, 1)

        self.status_label = QtWidgets.QLabel(
            "Compiled from the current GUI settings and connected QICK HWH."
        )
        layout.addWidget(self.status_label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.copy_button = buttons.addButton(
            "Copy", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.save_button = buttons.addButton(
            "Save As...", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.copy_button.clicked.connect(self._copy_assembly)
        self.save_button.clicked.connect(self._save_assembly)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _copy_assembly(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self._assembly)
        self.status_label.setText("Assembly copied to the clipboard.")

    def _save_assembly(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save QICK tProcessor assembly",
            "qick_program.asm",
            "QICK assembly (*.asm);;Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        output_path = Path(path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".asm")
        output_path.write_text(self._assembly, encoding="utf-8")
        self.status_label.setText(f"Saved to {output_path}")


class QickExportDialog(QtWidgets.QDialog):
    """Collect QICK timing, channel-map, repetition, and sweep settings."""

    def __init__(
        self,
        pulse_count: int,
        set_names: Tuple[str, ...],
        parent=None,
        default_set_index: int = 0,
        initial_rf_spec: Optional[QickRfPulseSpec] = None,
        initial_rf_specs: Optional[Sequence[QickRfPulseSpec]] = None,
        initial_ddr_readout_spec: Optional[QickDdrReadoutSpec] = None,
        initial_sweep: Optional[QickSweepSpec] = None,
        initial_sweeps: Optional[Sequence[QickSweepSpec]] = None,
        initial_cross_capacitance=None,
        initial_fabric_mhz: float = DEFAULT_QICK_FABRIC_MHZ,
        initial_tproc_mhz: float = DEFAULT_QICK_TPROC_MHZ,
        initial_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
        initial_awg_channels: Optional[Sequence[int]] = None,
        initial_repetitions: int = 1,
        initial_bias_t_enabled: bool = False,
        initial_bias_t_compensation_type: str = "dc",
        initial_bias_t_compensation_mv: Optional[float] = None,
        initial_bias_t_mode: str = "fixed_voltage",
        initial_bias_t_duration_us: float = DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
        initial_bias_t_filter_tau_us: float = DEFAULT_BIAS_T_FILTER_TAU_US,
    ):
        super().__init__(parent)
        self.setWindowTitle("QICK export settings")
        self.resize(660, 780)
        self._pulse_count = pulse_count
        if initial_rf_spec is not None and initial_rf_specs is not None:
            raise ValueError("use either initial_rf_spec or initial_rf_specs, not both")
        self._rf_pulse_specs = tuple(
            initial_rf_specs
            if initial_rf_specs is not None
            else (() if initial_rf_spec is None else (initial_rf_spec,))
        )
        self._ddr_readout_spec = initial_ddr_readout_spec
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
        self.fabric_mhz.setValue(float(initial_fabric_mhz))
        self.fabric_mhz.setSuffix(" MHz")

        self.tproc_mhz = QtWidgets.QDoubleSpinBox()
        self.tproc_mhz.setRange(1.0, 5000.0)
        self.tproc_mhz.setDecimals(6)
        self.tproc_mhz.setValue(float(initial_tproc_mhz))
        self.tproc_mhz.setSuffix(" MHz")

        self.full_scale_mv = QtWidgets.QDoubleSpinBox()
        self.full_scale_mv.setRange(1.0, 1.0e6)
        self.full_scale_mv.setDecimals(6)
        self.full_scale_mv.setValue(float(initial_full_scale_mv))
        self.full_scale_mv.setSuffix(" mV")
        self._sweep_display_scale_mv = float(initial_full_scale_mv)

        if initial_awg_channels is None:
            initial_awg_channels = DEFAULT_QSTL_AWG_CHANNELS[:pulse_count]
        self.awg_channels = QtWidgets.QLineEdit(
            ", ".join(str(index) for index in initial_awg_channels)
        )
        self.repetitions = QtWidgets.QSpinBox()
        self.repetitions.setRange(1, 1_000_000)
        self.repetitions.setValue(int(initial_repetitions))
        self.bias_t_group = QtWidgets.QGroupBox("Bias-T compensation")
        self.bias_t_group.setCheckable(True)
        self.bias_t_group.setChecked(bool(initial_bias_t_enabled))
        bias_t_form = QtWidgets.QFormLayout(self.bias_t_group)
        self.bias_t_type = QtWidgets.QComboBox()
        self.bias_t_type.addItem("DC compensation", "dc")
        self.bias_t_type.addItem("Filter compensation", "filter")
        type_index = self.bias_t_type.findData(
            str(initial_bias_t_compensation_type)
        )
        if type_index < 0:
            raise ValueError(
                "Bias-T compensation type must be one of "
                f"{BIAS_T_COMPENSATION_TYPES}"
            )
        self.bias_t_type.setCurrentIndex(type_index)
        self.bias_t_mode = QtWidgets.QComboBox()
        self.bias_t_mode.addItem("Fixed voltage (adjust time)", "fixed_voltage")
        self.bias_t_mode.addItem("Fixed time (adjust voltage)", "fixed_time")
        mode_index = self.bias_t_mode.findData(str(initial_bias_t_mode))
        if mode_index < 0:
            raise ValueError(
                f"Bias-T compensation mode must be one of {BIAS_T_COMPENSATION_MODES}"
            )
        self.bias_t_mode.setCurrentIndex(mode_index)
        self.bias_t_compensation_mv = QtWidgets.QDoubleSpinBox()
        self.bias_t_compensation_mv.setRange(0.001, float(initial_full_scale_mv))
        self.bias_t_compensation_mv.setDecimals(6)
        self.bias_t_compensation_mv.setSuffix(" mV")
        self.bias_t_compensation_mv.setValue(float(
            initial_full_scale_mv * DEFAULT_BIAS_T_COMPENSATION_FRACTION
            if initial_bias_t_compensation_mv is None
            else initial_bias_t_compensation_mv
        ))
        self.bias_t_duration_us = QtWidgets.QDoubleSpinBox()
        self.bias_t_duration_us.setRange(1.0e-6, 1.0e9)
        self.bias_t_duration_us.setDecimals(6)
        self.bias_t_duration_us.setSuffix(" us")
        self.bias_t_duration_us.setValue(float(initial_bias_t_duration_us))
        self.bias_t_filter_tau_us = QtWidgets.QDoubleSpinBox()
        self.bias_t_filter_tau_us.setRange(1.0e-6, 1.0e12)
        self.bias_t_filter_tau_us.setDecimals(6)
        self.bias_t_filter_tau_us.setSuffix(" us")
        self.bias_t_filter_tau_us.setValue(float(initial_bias_t_filter_tau_us))
        bias_t_form.addRow("Compensation type:", self.bias_t_type)
        bias_t_form.addRow("DC control mode:", self.bias_t_mode)
        bias_t_form.addRow("DC voltage:", self.bias_t_compensation_mv)
        bias_t_form.addRow("DC time:", self.bias_t_duration_us)
        bias_t_form.addRow("Filter time constant (tau):", self.bias_t_filter_tau_us)
        self.full_scale_mv.valueChanged.connect(
            lambda value: self.bias_t_compensation_mv.setMaximum(
                max(0.001, float(value))
            )
        )
        self.bias_t_mode.currentIndexChanged.connect(
            self._update_bias_t_mode_controls
        )
        self.bias_t_type.currentIndexChanged.connect(
            self._update_bias_t_mode_controls
        )
        self._update_bias_t_mode_controls()

        form.addRow("AWG fabric clock:", self.fabric_mhz)
        form.addRow("tProcessor clock:", self.tproc_mhz)
        form.addRow("QICK full scale (+/-):", self.full_scale_mv)
        form.addRow("AWG generator indices:", self.awg_channels)
        form.addRow("Repetitions per sweep point:", self.repetitions)
        form.addRow(self.bias_t_group)

        sweep_group = QtWidgets.QGroupBox("Independent Cartesian voltage sweeps")
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
        for widget, value in (
            (self.sweep_start, -0.2 * self._sweep_display_scale_mv),
            (self.sweep_stop, 0.8 * self._sweep_display_scale_mv),
        ):
            widget.setRange(
                -self._sweep_display_scale_mv,
                self._sweep_display_scale_mv,
            )
            widget.setDecimals(6)
            widget.setSingleStep(max(0.001, self._sweep_display_scale_mv / 100.0))
            widget.setSuffix(" mV")
            widget.setValue(value)
        self.sweep_count = QtWidgets.QSpinBox()
        self.sweep_count.setRange(1, 1_000_000)
        self.sweep_count.setValue(9)
        sweep_form.addRow("Output:", self.sweep_output)
        sweep_form.addRow("SET segment:", self.sweep_segment)
        sweep_form.addRow("Start voltage:", self.sweep_start)
        sweep_form.addRow("Stop voltage:", self.sweep_stop)
        sweep_form.addRow("Point count:", self.sweep_count)
        self.sweep_table = QtWidgets.QTableWidget(0, 5)
        self.sweep_table.setHorizontalHeaderLabels(
            ["Output", "SET", "Start (mV)", "Stop (mV)", "Count"]
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
        self.full_scale_mv.valueChanged.connect(
            self._rescale_sweep_voltage_controls
        )
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

        note = QtWidgets.QLabel(
            "All exported ports must have identical SET/RAMP timing. "
            "Sweep endpoints are entered and displayed in mV; command generation "
            "normalizes them using the configured full scale. RF output and "
            "readout settings come from the main-window RF tabs."
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
        full_scale_mv = self.full_scale_mv.value()
        return QickSweepSpec(
            segment_name=self.sweep_segment.currentText(),
            output_name=self.sweep_output.currentText(),
            start=self.sweep_start.value() / full_scale_mv,
            stop=self.sweep_stop.value() / full_scale_mv,
            count=self.sweep_count.value(),
        )

    def _update_bias_t_mode_controls(self, *_args) -> None:
        filter_mode = self.bias_t_type.currentData() == "filter"
        fixed_time = self.bias_t_mode.currentData() == "fixed_time"
        self.bias_t_mode.setEnabled(not filter_mode)
        self.bias_t_compensation_mv.setEnabled(not filter_mode and not fixed_time)
        self.bias_t_duration_us.setEnabled(not filter_mode and fixed_time)
        self.bias_t_filter_tau_us.setEnabled(filter_mode)

    def _rescale_sweep_voltage_controls(self, value: float) -> None:
        new_scale_mv = float(value)
        old_scale_mv = self._sweep_display_scale_mv
        if old_scale_mv <= 0.0 or new_scale_mv <= 0.0:
            return
        for widget in (self.sweep_start, self.sweep_stop):
            with QtCore.QSignalBlocker(widget):
                normalized_value = widget.value() / old_scale_mv
                widget.setRange(-new_scale_mv, new_scale_mv)
                widget.setSingleStep(max(0.001, new_scale_mv / 100.0))
                widget.setValue(normalized_value * new_scale_mv)
        self._sweep_display_scale_mv = new_scale_mv
        selected_row = self.sweep_table.currentRow()
        self._refresh_sweep_table()
        if 0 <= selected_row < self.sweep_table.rowCount():
            with QtCore.QSignalBlocker(self.sweep_table):
                self.sweep_table.selectRow(selected_row)

    def _refresh_sweep_table(self, select_row: Optional[int] = None) -> None:
        with QtCore.QSignalBlocker(self.sweep_table):
            self.sweep_table.setRowCount(len(self._dialog_sweep_specs))
            for row, spec in enumerate(self._dialog_sweep_specs):
                values = (
                    spec.output_name,
                    spec.segment_name,
                    f"{spec.start * self.full_scale_mv.value():.6g}",
                    f"{spec.stop * self.full_scale_mv.value():.6g}",
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
        self.sweep_start.setValue(spec.start * self.full_scale_mv.value())
        self.sweep_stop.setValue(spec.stop * self.full_scale_mv.value())
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
            raise ValueError("each Cartesian voltage sweep must target a unique output/SET")
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
        except (TypeError, ValueError) as exc:
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
        rf_channels = tuple(spec.gen_ch for spec in self._rf_pulse_specs)
        if any(channel in awg_channels for channel in rf_channels):
            raise ValueError("RF generator indices must differ from all AWG tuning indices")
        if len(set(rf_channels)) != len(rf_channels):
            raise ValueError("RF generator indices must be unique")
        return {
            "fabric_mhz": self.fabric_mhz.value(),
            "tproc_mhz": self.tproc_mhz.value(),
            "full_scale_mv": self.full_scale_mv.value(),
            "awg_channels": awg_channels,
            "repetitions_per_sweep": self.repetitions.value(),
            "bias_t_compensation_enabled": self.bias_t_group.isChecked(),
            "bias_t_compensation_type": str(self.bias_t_type.currentData()),
            "bias_t_compensation_voltage_mv": self.bias_t_compensation_mv.value(),
            "bias_t_compensation_mode": str(self.bias_t_mode.currentData()),
            "bias_t_compensation_duration_us": self.bias_t_duration_us.value(),
            "bias_t_filter_tau_us": self.bias_t_filter_tau_us.value(),
            "sweep": sweep,
            "sweeps": sweeps,
            "cross_capacitance": tuple(
                tuple(float(value) for value in row)
                for row in self._cross_capacitance
            ),
            "rf_pulse_spec": (
                self._rf_pulse_specs[0] if len(self._rf_pulse_specs) == 1 else None
            ),
            "rf_pulse_specs": self._rf_pulse_specs,
            "ddr_readout_spec": self._ddr_readout_spec,
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

        self._settings_path: Optional[Path] = None
        self._time_unit = DEFAULT_TIME_UNIT
        self._pulse: List[PulseSequence] = [
            PulseSequence(
                DEFAULT_INITIAL_VOLTAGE_MV,
                initial_duration_ns=DEFAULT_GUI_DURATION_NS,
            )
        ]
        self._pulse[0].v_bounds             = (-2500, 2500)
        self._rf_pulse_spec: Optional[QickRfPulseSpec] = None
        self._rf_pulse_specs: List[QickRfPulseSpec] = []
        self._ddr_readout_spec: Optional[QickDdrReadoutSpec] = None
        self._rf_panel: Optional[RfPulseEditorPanel] = None
        self._dock_rf: Optional[QtWidgets.QDockWidget] = None
        self._sweep_specs: List[QickSweepSpec] = []
        self._cross_capacitance = np.eye(1, dtype=float)
        self._qick_fabric_mhz = float(DEFAULT_QICK_FABRIC_MHZ)
        self._qick_tproc_mhz = float(DEFAULT_QICK_TPROC_MHZ)
        self._qick_full_scale_mv = float(DEFAULT_QICK_FULL_SCALE_MV)
        self._qick_awg_channels = (DEFAULT_QSTL_AWG_CHANNELS[0],)
        self._qick_repetitions_per_sweep = 1
        self._bias_t_compensation_enabled = False
        self._bias_t_compensation_type = "dc"
        self._bias_t_compensation_voltage_mv = float(
            DEFAULT_BIAS_T_COMPENSATION_MV
        )
        self._bias_t_compensation_mode = "fixed_voltage"
        self._bias_t_compensation_duration_us = float(
            DEFAULT_BIAS_T_COMPENSATION_DURATION_US
        )
        self._bias_t_filter_tau_us = float(DEFAULT_BIAS_T_FILTER_TAU_US)
        self._experiment_thread: Optional[QtCore.QThread] = None
        self._experiment_worker: Optional[QtCore.QObject] = None
        self._grid_time_ns = 1000.0
        self._grid_voltage_mv = 100.0
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
            self._add_port,
            time_unit=self._time_unit,
        )
        self._rf_ports_panel = RfPortsPanel(
            self._pulse[0], time_unit=self._time_unit, parent=self
        )
        self._rf_readout_panel = RfReadoutPanel(
            self._pulse[0], time_unit=self._time_unit, parent=self
        )
        self._experiment_panel = ExperimentPanel(
            fabric_mhz=self._qick_fabric_mhz,
            tproc_mhz=self._qick_tproc_mhz,
            full_scale_mv=self._qick_full_scale_mv,
            awg_channels=self._qick_awg_channels,
            repetitions=self._qick_repetitions_per_sweep,
            bias_t_enabled=self._bias_t_compensation_enabled,
            bias_t_compensation_type=self._bias_t_compensation_type,
            bias_t_compensation_mv=self._bias_t_compensation_voltage_mv,
            bias_t_mode=self._bias_t_compensation_mode,
            bias_t_duration_us=self._bias_t_compensation_duration_us,
            bias_t_filter_tau_us=self._bias_t_filter_tau_us,
            parent=self,
        )
        self._stability_panel = StabilityDiagramPanel(self)
        self._stability_rf_ports_panel = RfPortsPanel(
            self._pulse[0], time_unit=self._time_unit, parent=self._stability_panel
        )
        self._stability_rf_readout_panel = RfReadoutPanel(
            self._pulse[0], time_unit=self._time_unit, parent=self._stability_panel
        )
        self._stability_panel.set_rf_editors(
            self._stability_rf_ports_panel,
            self._stability_rf_readout_panel,
        )
        self._stability_panel.apply_path_settings(
            self._stability_panel.front_panel_values()
        )
        self._sparameter_panel = SParameterSweepPanel(self)
        self._calibration_panel = CalibrationPanel(self)
        self._noise_panel = NoiseAnalysisPanel(
            self,
            default_database_path=DEFAULT_QCODES_DB_PATH,
        )
        self._qick_configuration = None
        self._qick_front_panel_target = None
        self._qick_front_panel_dialog = QtWidgets.QDialog(self)
        self._qick_front_panel_dialog.setModal(False)
        self._qick_front_panel_dialog.setWindowTitle("QICK Front Panel")
        self._qick_front_panel_dialog.resize(1080, 680)
        front_panel_layout = QtWidgets.QVBoxLayout(self._qick_front_panel_dialog)
        self._qick_front_panel = QickFrontPanelControl(
            self._qick_front_panel_dialog
        )
        front_panel_layout.addWidget(self._qick_front_panel)
        front_panel_buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Close,
            parent=self._qick_front_panel_dialog,
        )
        front_panel_buttons.rejected.connect(self._qick_front_panel_dialog.close)
        front_panel_layout.addWidget(front_panel_buttons)
        self._qick_front_panel.set_path_values(
            self._sparameter_panel.front_panel_values()
        )
        self._rf_ports_panel.specs_changed.connect(self._on_rf_specs_changed)
        self._rf_readout_panel.spec_changed.connect(self._on_readout_spec_changed)
        self._stability_rf_readout_panel.spec_changed.connect(
            self._on_stability_readout_spec_changed
        )
        self._stability_rf_ports_panel.specs_changed.connect(
            self._sync_stability_path_from_editors
        )
        self._on_stability_readout_spec_changed(
            self._stability_rf_readout_panel.spec()
        )
        self._stability_panel.dc_measure_changed.connect(
            self._on_stability_dc_measure_changed
        )
        self._stability_panel.dc_calibration_changed.connect(
            self._on_stability_dc_calibration_changed
        )
        self._experiment_panel.run_requested.connect(self._run_qick_experiment)
        self._experiment_panel.show_program_requested.connect(
            self._show_qick_program
        )
        self._experiment_panel.bias_t_changed.connect(self._on_bias_t_changed)
        self._stability_panel.start_requested.connect(
            lambda: self._run_stability_diagram(continuous=True)
        )
        self._stability_panel.stop_requested.connect(
            self._stop_stability_diagram
        )
        self._stability_panel.single_shot_requested.connect(
            lambda: self._run_stability_diagram(continuous=False)
        )
        self._stability_panel.front_panel_requested.connect(
            lambda: self._show_qick_front_panel("path", self._stability_panel)
        )
        self._sparameter_panel.run_requested.connect(
            self._run_sparameter_sweep
        )
        self._sparameter_panel.load_requested.connect(
            self._load_sparameter_run
        )
        self._sparameter_panel.front_panel_requested.connect(
            lambda: self._show_qick_front_panel("path", self._sparameter_panel)
        )
        self._rf_ports_panel.front_panel_requested.connect(
            lambda panel: self._show_qick_front_panel("output", panel)
        )
        self._rf_readout_panel.front_panel_requested.connect(
            lambda panel: self._show_qick_front_panel("input", panel)
        )
        self._stability_rf_ports_panel.front_panel_requested.connect(
            lambda panel: self._show_qick_front_panel("output", panel)
        )
        self._stability_rf_readout_panel.front_panel_requested.connect(
            lambda panel: self._show_qick_front_panel("input", panel)
        )
        self._calibration_panel.output_requested.connect(
            lambda: self._run_power_calibration("output")
        )
        self._calibration_panel.input_requested.connect(
            lambda: self._run_power_calibration("input")
        )
        self._calibration_panel.dc_voltage_requested.connect(
            lambda: self._run_power_calibration("dc_voltage")
        )
        self._calibration_panel.front_panel_requested.connect(
            lambda: self._show_qick_front_panel("path", self._calibration_panel)
        )
        self._noise_panel.load_requested.connect(self._load_noise_trace)
        self._qick_front_panel.identify_requested.connect(
            self._identify_qick_configuration
        )
        self._qick_front_panel.settings_applied.connect(
            self._apply_front_panel_settings
        )

        self._time_unit_combo = QtWidgets.QComboBox()
        self._time_unit_combo.addItems(tuple(TIME_UNIT_NS))
        self._time_unit_combo.setCurrentText(self._time_unit)
        self._time_unit_combo.currentTextChanged.connect(self._set_time_unit)
        unit_row = QtWidgets.QHBoxLayout()
        unit_row.addWidget(QtWidgets.QLabel("Time unit:"))
        unit_row.addWidget(self._time_unit_combo)
        unit_row.addStretch(1)

        self._awg_tuning_page = QtWidgets.QWidget(self)
        awg_tuning_layout = QtWidgets.QVBoxLayout(self._awg_tuning_page)
        awg_tuning_layout.setContentsMargins(0, 0, 0, 0)
        self._awg_tuning_tabs = QtWidgets.QTabWidget(self._awg_tuning_page)
        self._awg_tuning_tabs.addTab(self._multi_ctrl, "AWG Outputs")
        self._awg_tuning_tabs.addTab(self._rf_ports_panel, "RF Outputs")
        self._awg_tuning_tabs.addTab(self._rf_readout_panel, "RF Readout")
        self._awg_tuning_tabs.addTab(self._experiment_panel, "Experiment")
        self._awg_tuning_tabs.setCurrentWidget(self._multi_ctrl)
        awg_tuning_layout.addWidget(self._awg_tuning_tabs)

        self._control_tabs = QtWidgets.QTabWidget()
        self._control_tabs.addTab(self._awg_tuning_page, "AWG Tuning")
        self._control_tabs.addTab(
            self._stability_panel,
            "Stability Diagram",
        )
        self._control_tabs.addTab(self._sparameter_panel, "RF S-Parameter")
        self._control_tabs.addTab(self._calibration_panel, "Calibration")
        self._control_tabs.addTab(self._noise_panel, "Noise Analysis")
        self._control_tabs.setCurrentWidget(self._awg_tuning_page)
        self._control_tabs.currentChanged.connect(self._on_control_tab_changed)
        control_container = QtWidgets.QWidget(self)
        control_layout = QtWidgets.QVBoxLayout(control_container)
        control_layout.setContentsMargins(4, 4, 4, 4)
        control_layout.addLayout(unit_row)
        control_layout.addWidget(self._control_tabs)
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

        self._rf_timelines = []
        self._rf_timeline = None
        self._rf_timeline_container = QtWidgets.QWidget(self)
        self._rf_timeline_layout = QtWidgets.QVBoxLayout(self._rf_timeline_container)
        self._rf_timeline_layout.setContentsMargins(0, 0, 0, 0)
        self._rf_timeline_layout.setSpacing(2)
        self._waveform_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        self._waveform_splitter.addWidget(self._plot)
        self._waveform_splitter.setCollapsible(0, False)
        self._waveform_splitter.setStretchFactor(0, 4)
        if RfPulseTimelineWidget is not None:
            self._waveform_splitter.addWidget(self._rf_timeline_container)
            self._waveform_splitter.setCollapsible(1, True)
            self._waveform_splitter.setStretchFactor(1, 1)
            self._rf_timeline_container.hide()

        self._dock_ctrl  = QtWidgets.QDockWidget("Control Panel", self)
        self._dock_ctrl.setWidget(control_container)

        self._dock_plot = QtWidgets.QDockWidget(
            "Waveform Plot - Virtual (solid) / Physical (dashed)",
            self,
        )
        self._dock_plot.setWidget(self._waveform_splitter)

        self._dock_trace = QtWidgets.QDockWidget("Trace Plot", self)
        self._dock_trace.setWidget(self._trace_placeholder)

        self._stability_plot = self._stability_panel.detach_plot()
        self._dock_stability = QtWidgets.QDockWidget(
            "Stability Diagram - FIR Magnitude / Phase",
            self,
        )
        self._dock_stability.setWidget(self._stability_plot)

        self._sparameter_plot = SParameterPlotWidget(self)
        self._dock_sparameter = QtWidgets.QDockWidget(
            "RF S-Parameter", self
        )
        self._dock_sparameter.setWidget(self._sparameter_plot)

        self._noise_plot = self._noise_panel.detach_plot()
        self._dock_noise = QtWidgets.QDockWidget(
            "Noise Analysis - Current / ASD", self
        )
        self._dock_noise.setWidget(self._noise_plot)

        for dock in (
            self._dock_ctrl,
            self._dock_plot,
            self._dock_trace,
            self._dock_stability,
            self._dock_sparameter,
            self._dock_noise,
        ):
            dock.setFeatures(
                QtWidgets.QDockWidget.DockWidgetMovable |
                QtWidgets.QDockWidget.DockWidgetFloatable
            )
        self._multi_ctrl.btn_reset.clicked.connect(self._plot._restore_full_intensity)

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._dock_ctrl)
        self.splitDockWidget(self._dock_ctrl, self._dock_plot, QtCore.Qt.Horizontal)
        self.splitDockWidget(self._dock_plot, self._dock_trace, QtCore.Qt.Horizontal)
        self.tabifyDockWidget(self._dock_trace, self._dock_stability)
        self.tabifyDockWidget(self._dock_trace, self._dock_sparameter)
        self.tabifyDockWidget(self._dock_trace, self._dock_noise)
        self.resizeDocks(
            [self._dock_ctrl, self._dock_plot, self._dock_trace],
            [200, 400, 300],
            QtCore.Qt.Horizontal
        )
        self._dock_plot.raise_()
        self._dock_trace.raise_()

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
        self._set_grid_settings(
            time_step_ns=self._grid_time_ns,
            voltage_step_mv=self._grid_voltage_mv,
            snap_enabled=self._grid_snap_enabled,
            visible=self._grid_visible,
        )
        self._set_time_unit(self._time_unit)
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
        self._qick_awg_channels = tuple(
            channel
            for channel_index, channel in enumerate(self._qick_awg_channels)
            if channel_index != idx
        )
        if hasattr(self, "_experiment_panel"):
            self._experiment_panel.set_qick_values(
                fabric_mhz=self._qick_fabric_mhz,
                tproc_mhz=self._qick_tproc_mhz,
                full_scale_mv=self._qick_full_scale_mv,
                awg_channels=self._qick_awg_channels,
                repetitions=self._qick_repetitions_per_sweep,
            )
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
        save_settings = m_file.addAction("Save Settings JSON...")
        save_settings.setShortcut(QtGui.QKeySequence.Save)
        save_settings.triggered.connect(self._save_json)
        load_settings = m_file.addAction("Load Settings JSON...")
        load_settings.setShortcut(QtGui.QKeySequence.Open)
        load_settings.triggered.connect(self._load_json)
        m_file.addSeparator()
        m_file.addAction("E&xit",       self.close)

        # Pulse
        m_pulse     = mb.addMenu("&Pulse")
        gen         = m_pulse.addAction("&Generate QCS string…")
        gen.setShortcut(QtGui.QKeySequence("Ctrl+G"))
        gen.triggered.connect(self._generate_pulse)
        m_pulse.addSeparator()
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

    def _set_time_unit(self, unit: str) -> None:
        if unit not in TIME_UNIT_NS:
            raise ValueError(f"unsupported time unit {unit!r}")
        self._time_unit = unit
        self._multi_ctrl.set_time_unit(unit)
        self._rf_ports_panel.set_time_unit(unit)
        self._rf_readout_panel.set_time_unit(unit)
        self._stability_rf_ports_panel.set_time_unit(unit)
        self._stability_rf_readout_panel.set_time_unit(unit)
        if hasattr(self._plot, "set_time_unit"):
            self._plot.set_time_unit(unit)
        for timeline in self._rf_timelines:
            if hasattr(timeline, "set_time_unit"):
                timeline.set_time_unit(unit)
        self.statusBar().showMessage(f"Time display unit: {unit}")

    def _configure_grid(self) -> None:
        dialog = GridSettingsDialog(
            time_step_ns=self._grid_time_ns,
            voltage_step_mv=self._grid_voltage_mv,
            snap_enabled=(self._grid_snap_enabled if self._grid_configured else True),
            visible=self._grid_visible,
            time_unit=self._time_unit,
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
            f"Grid: {_time_from_ns(time_step_ns, self._time_unit):.6g} "
            f"{self._time_unit} x {voltage_step_mv:.6g} mV; "
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
        if self._bias_t_compensation_enabled:
            sequence = build_qick_sequence(
                self._pulse,
                output_names=self._qick_output_names(),
                fabric_mhz=self._qick_fabric_mhz,
                full_scale_mv=self._qick_full_scale_mv,
                sweep=(self._sweep_specs[0] if len(self._sweep_specs) == 1 else None),
                sweeps=(tuple(self._sweep_specs) if len(self._sweep_specs) > 1 else None),
                cross_capacitance=self._cross_capacitance,
                bias_t_compensation_enabled=True,
                bias_t_compensation_type=self._bias_t_compensation_type,
                bias_t_compensation_voltage_mv=self._bias_t_compensation_voltage_mv,
                bias_t_compensation_mode=self._bias_t_compensation_mode,
                bias_t_compensation_duration_us=(
                    self._bias_t_compensation_duration_us
                ),
                bias_t_filter_tau_us=self._bias_t_filter_tau_us,
            )
            cycles, waveforms, _boundaries = (
                sequence.compensated_waveform_vertices(0)
            )
            time_ns = np.asarray(cycles, dtype=float) * 1000.0 / self._qick_fabric_mhz
            physical_mv = np.vstack([
                np.asarray(waveforms[name], dtype=float)
                * self._qick_full_scale_mv
                for name in self._qick_output_names()
            ])
        else:
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
            f"{new_spec.start * self._qick_full_scale_mv:.6g} mV to "
            f"{new_spec.stop * self._qick_full_scale_mv:.6g} mV, "
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
                f"Voltage sweep removed; {self._sweep_cartesian_count()} "
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
        self._refresh_stability_targets()

    def _build_toolbar(self):
        tb          = self.addToolBar("Tools")
        btn_fit     = QtWidgets.QAction("Fit", self)
        btn_fit.setToolTip("Show the full pulse (keyboard: F)")
        btn_fit.triggered.connect(self._fit_view)
        tb.addAction(btn_fit)
        # keyboard shortcut
        QtWidgets.QShortcut(QtGui.QKeySequence("F"), self, self._fit_view)

    def _fit_view(self):
        """Fit the view to the pulse data."""
        self._plot.fit_view()
        rf_ranges = []
        for spec in self._rf_pulse_specs:
            try:
                start_us, end_us, _ = rf_pulse_absolute_times_us(self._pulse[0], spec)
            except ValueError:
                continue
            rf_ranges.append((start_us * 1000.0, end_us * 1000.0))
        if rf_ranges:
            x_range = self._plot.getPlotItem().vb.viewRange()[0]
            x_min = min(float(x_range[0]), *(start for start, _end in rf_ranges))
            x_max = max(float(x_range[1]), *(end for _start, end in rf_ranges))
            margin = max(1.0, 0.03 * max(1.0, x_max - x_min))
            self._plot.setXRange(x_min - margin, x_max + margin, padding=0.0)
        if self._trace is not None:
            self._trace.fit_view()
        self._stability_panel.plot.fit_view()
        self._sparameter_plot.fit_view()
        self._noise_plot.fit_view()

    def _show_rf_editor(self) -> None:
        """Compatibility helper: reveal the always-present RF Outputs tab."""
        self._control_tabs.setCurrentWidget(self._awg_tuning_page)
        self._awg_tuning_tabs.setCurrentWidget(self._rf_ports_panel)
        self._dock_ctrl.show()
        self._dock_ctrl.raise_()

    def _on_control_tab_changed(self, _index: int) -> None:
        current = self._control_tabs.currentWidget()
        if current is self._stability_panel:
            self._dock_stability.show()
            self._dock_stability.raise_()
        elif current is self._sparameter_panel:
            self._dock_sparameter.show()
            self._dock_sparameter.raise_()
        elif current is self._noise_panel:
            self._dock_noise.show()
            self._dock_noise.raise_()
        else:
            self._dock_plot.show()
            self._dock_plot.raise_()

    def _apply_rf_spec(self, spec: QickRfPulseSpec) -> None:
        self._rf_pulse_specs = [spec]
        self._rf_pulse_spec = spec
        self._refresh_rf_timeline(fit_view=True)
        self.statusBar().showMessage(
            f"RF pulse applied: {spec.frequency_mhz:.6g} MHz, gain {spec.gain}, "
            f"ATT1/ATT2 {spec.att1_db:.2f}/{spec.att2_db:.2f} dB, "
            f"{spec.filter_type} filter"
        )

    def _remove_rf_spec(self) -> None:
        self._rf_pulse_specs = []
        self._rf_pulse_spec = None
        self._refresh_rf_timeline()
        self.statusBar().showMessage("RF pulse disabled")

    def _on_rf_specs_changed(self, specs) -> None:
        self._rf_pulse_specs = list(specs)
        self._rf_pulse_spec = self._rf_pulse_specs[0] if self._rf_pulse_specs else None
        self._refresh_rf_timeline()

    def _on_readout_spec_changed(self, spec) -> None:
        self._ddr_readout_spec = spec
        if spec is None:
            self.statusBar().showMessage("RF readout disabled")
        elif spec.dc_measure_mode:
            self.statusBar().showMessage(
                f"DC current readout {spec.ro_ch}: "
                f"{spec.samples_per_trigger} samples at 1 MSPS, "
                f"gain {spec.dc_measure_gain_v_per_a:g} V/A"
            )
        elif spec.dc_voltage_calibration_enabled:
            run_label = (
                "latest matching"
                if spec.dc_voltage_calibration_run_id == 0
                else f"Run {spec.dc_voltage_calibration_run_id}"
            )
            self.statusBar().showMessage(
                f"Calibrated DC voltage readout {spec.ro_ch}: "
                f"{spec.samples_per_trigger} samples at 1 MSPS, {run_label}"
            )
        else:
            self.statusBar().showMessage(
                f"RF readout {spec.ro_ch}: {spec.samples_per_trigger} samples at 1 MSPS"
            )

    def _on_stability_readout_spec_changed(self, _spec) -> None:
        configured = self._stability_rf_readout_panel.configured_spec()
        self._stability_panel.set_dc_measure_context(
            configured.input_board_type,
            configured.dc_measure_mode,
            configured.dc_measure_gain_v_per_a,
            configured.dc_voltage_calibration_enabled,
            configured.dc_voltage_calibration_database_path,
            configured.dc_voltage_calibration_run_id,
        )
        self._sync_stability_path_from_editors()

    def _sync_stability_path_from_editors(self, *_args) -> None:
        self._stability_panel.path_diagram.apply_external_settings(
            self._stability_panel.front_panel_values()
        )

    def _on_stability_dc_measure_changed(
        self,
        enabled: bool,
        gain_v_per_a: float,
    ) -> None:
        self._stability_rf_readout_panel.set_dc_measurement(
            enabled,
            gain_v_per_a,
        )

    def _on_stability_dc_calibration_changed(
        self,
        enabled: bool,
        database_path: str,
        run_id: int,
    ) -> None:
        self._stability_rf_readout_panel.set_dc_voltage_calibration_selection(
            enabled,
            database_path,
            run_id,
        )

    def _on_bias_t_changed(
        self,
        enabled: bool,
        compensation_type: str,
        compensation_mv: float,
        mode: str,
        duration_us: float,
        filter_tau_us: float,
    ) -> None:
        self._bias_t_compensation_enabled = bool(enabled)
        self._bias_t_compensation_type = str(compensation_type)
        self._bias_t_compensation_voltage_mv = float(compensation_mv)
        self._bias_t_compensation_mode = str(mode)
        self._bias_t_compensation_duration_us = float(duration_us)
        self._bias_t_filter_tau_us = float(filter_tau_us)
        try:
            self._refresh_physical_waveforms(fit_view=False)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            self.statusBar().showMessage(f"Bias-T preview unavailable: {exc}")
            return
        if enabled:
            if compensation_type == "filter":
                self.statusBar().showMessage(
                    "Bias-T filter compensation enabled; flat-segment slew is "
                    f"target/tau with tau={filter_tau_us:.6g} us"
                )
            elif mode == "fixed_time":
                self.statusBar().showMessage(
                    "Bias-T compensation enabled for "
                    f"{duration_us:.6g} us; voltage follows pulse area"
                )
            else:
                self.statusBar().showMessage(
                    f"Bias-T compensation enabled at {compensation_mv:.6g} mV"
                )
        else:
            self.statusBar().showMessage("Bias-T compensation disabled")

    def _experiment_run_arguments(
        self,
        *,
        require_readout: bool = True,
        require_run_config: bool = True,
    ) -> dict:
        values = self._experiment_panel.values(
            len(self._pulse), require_run_config=require_run_config
        )
        self._qick_fabric_mhz = values["fabric_mhz"]
        self._qick_tproc_mhz = values["tproc_mhz"]
        self._qick_full_scale_mv = values["full_scale_mv"]
        self._qick_awg_channels = values["awg_channels"]
        self._qick_repetitions_per_sweep = values["repetitions_per_sweep"]
        self._bias_t_compensation_enabled = values[
            "bias_t_compensation_enabled"
        ]
        self._bias_t_compensation_type = values["bias_t_compensation_type"]
        self._bias_t_compensation_voltage_mv = values[
            "bias_t_compensation_voltage_mv"
        ]
        self._bias_t_compensation_mode = values["bias_t_compensation_mode"]
        self._bias_t_compensation_duration_us = values[
            "bias_t_compensation_duration_us"
        ]
        self._bias_t_filter_tau_us = values["bias_t_filter_tau_us"]
        rf_specs = self._rf_ports_panel.specs()
        readout_spec = self._rf_readout_panel.spec()
        if require_readout and readout_spec is None:
            raise ValueError(
                "enable RF Readout before running so the 1 MSPS IQ trace can be saved"
            )
        rf_channels = {spec.gen_ch for spec in rf_specs}
        overlap = rf_channels.intersection(self._qick_awg_channels)
        if overlap:
            raise ValueError(
                "RF and AWG generator indices overlap: "
                + ", ".join(str(channel) for channel in sorted(overlap))
            )

        sweep = self._sweep_specs[0] if len(self._sweep_specs) == 1 else None
        sweeps = tuple(self._sweep_specs) if len(self._sweep_specs) > 1 else None
        sequence = build_qick_sequence(
            tuple(pulse.copy() for pulse in self._pulse),
            output_names=self._qick_output_names(),
            fabric_mhz=self._qick_fabric_mhz,
            full_scale_mv=self._qick_full_scale_mv,
            sweep=sweep,
            sweeps=sweeps,
            cross_capacitance=self._cross_capacitance.copy(),
            bias_t_compensation_enabled=self._bias_t_compensation_enabled,
            bias_t_compensation_type=self._bias_t_compensation_type,
            bias_t_compensation_voltage_mv=self._bias_t_compensation_voltage_mv,
            bias_t_compensation_mode=self._bias_t_compensation_mode,
            bias_t_compensation_duration_us=self._bias_t_compensation_duration_us,
            bias_t_filter_tau_us=self._bias_t_filter_tau_us,
        )
        gui_settings = self._settings_to_dict() if require_run_config else None
        return {
            "connection_config": values["connection"],
            "run_config": values["run"],
            "sequence": sequence,
            "awg_channels": self._qick_awg_channels,
            "repetitions_per_sweep": self._qick_repetitions_per_sweep,
            "rf_specs": rf_specs,
            "readout_spec": readout_spec,
            "gui_settings": gui_settings,
            "progress": False,
        }

    def _stability_run_arguments(self, *, save: bool) -> dict:
        values = self._experiment_panel.values(
            len(self._pulse),
            require_run_config=save,
            database_path=(
                self._stability_panel.database_path_value() if save else None
            ),
        )
        self._qick_fabric_mhz = values["fabric_mhz"]
        self._qick_tproc_mhz = values["tproc_mhz"]
        self._qick_full_scale_mv = values["full_scale_mv"]
        self._qick_awg_channels = values["awg_channels"]
        self._bias_t_compensation_enabled = values[
            "bias_t_compensation_enabled"
        ]
        self._bias_t_compensation_type = values["bias_t_compensation_type"]
        self._bias_t_compensation_voltage_mv = values[
            "bias_t_compensation_voltage_mv"
        ]
        self._bias_t_compensation_mode = values["bias_t_compensation_mode"]
        self._bias_t_compensation_duration_us = values[
            "bias_t_compensation_duration_us"
        ]
        self._bias_t_filter_tau_us = values["bias_t_filter_tau_us"]
        self._refresh_stability_targets()
        stability_config = self._stability_panel.config(
            full_scale_mv=self._qick_full_scale_mv
        )
        rf_specs = self._stability_rf_ports_panel.specs()
        readout_spec = self._stability_rf_readout_panel.spec()
        if readout_spec is None:
            raise ValueError(
                "enable RF Readout before acquiring a stability diagram"
            )
        readout_spec = replace(
            readout_spec,
            samples_per_trigger=stability_config.trace_samples_per_point,
        )
        overlap = {spec.gen_ch for spec in rf_specs}.intersection(
            self._qick_awg_channels
        )
        if overlap:
            raise ValueError(
                "RF and AWG generator indices overlap: "
                + ", ".join(str(channel) for channel in sorted(overlap))
            )

        sweeps = tuple(
            QickSweepSpec(
                segment_name=axis.segment_name,
                output_name=axis.output_name,
                start=axis.start_mv / self._qick_full_scale_mv,
                stop=axis.stop_mv / self._qick_full_scale_mv,
                count=axis.points,
            )
            for axis in (stability_config.x_axis, stability_config.y_axis)
        )
        sequence = build_qick_sequence(
            tuple(pulse.copy() for pulse in self._pulse),
            output_names=self._qick_output_names(),
            fabric_mhz=self._qick_fabric_mhz,
            full_scale_mv=self._qick_full_scale_mv,
            sweeps=sweeps,
            cross_capacitance=self._cross_capacitance.copy(),
            bias_t_compensation_enabled=self._bias_t_compensation_enabled,
            bias_t_compensation_type=self._bias_t_compensation_type,
            bias_t_compensation_voltage_mv=self._bias_t_compensation_voltage_mv,
            bias_t_compensation_mode=self._bias_t_compensation_mode,
            bias_t_compensation_duration_us=self._bias_t_compensation_duration_us,
            bias_t_filter_tau_us=self._bias_t_filter_tau_us,
        )
        return {
            "connection_config": values["connection"],
            "run_config": values["run"],
            "gui_settings": self._settings_to_dict() if save else None,
            "stability_config": stability_config,
            "full_scale_mv": self._qick_full_scale_mv,
            "sequence": sequence,
            "awg_channels": self._qick_awg_channels,
            "repetitions_per_sweep": stability_config.repetitions_per_point,
            "tproc_mhz": self._qick_tproc_mhz,
            "rf_specs": rf_specs,
            "readout_spec": readout_spec,
            "progress": False,
        }

    def _sparameter_run_arguments(self) -> dict:
        connection, run = self._experiment_panel.connection_values(
            database_path=self._sparameter_panel.database_path_value()
        )
        return {
            "connection_config": connection,
            "run_config": run,
            "sweep_config": self._sparameter_panel.config(),
            "tproc_mhz": self._experiment_panel.tproc_mhz.value(),
        }

    def _show_qick_front_panel(self, scope: str, target=None) -> None:
        """Open the live front-panel selector for the requesting editor."""
        self._qick_front_panel_target = target
        self._qick_front_panel.set_scope(scope)
        titles = {
            "path": "QICK Front Panel - RF Measurement Path",
            "output": "QICK Front Panel - RF Output",
            "input": "QICK Front Panel - RF Readout",
        }
        self._qick_front_panel_dialog.setWindowTitle(titles[scope])

        if scope == "output":
            spec = target.configured_spec()
            values = {
                "output_ch": spec.gen_ch,
                "output_board_type": spec.output_board_type,
                "output_nqz": spec.nqz,
                "output_att1_db": spec.att1_db,
                "output_att2_db": spec.att2_db,
                "output_filter_type": spec.filter_type,
                "output_filter_cutoff_ghz": spec.filter_cutoff,
                "output_filter_bandwidth_ghz": spec.filter_bandwidth,
            }
        elif scope == "input":
            spec = target.configured_spec()
            values = {
                "readout_ch": spec.ro_ch,
                "input_board_type": spec.input_board_type,
                "readout_nqz": spec.nqz,
                "readout_attenuation_db": spec.attenuation_db,
                "readout_dc_gain_db": spec.dc_gain_db,
                "readout_filter_type": spec.filter_type,
                "readout_filter_cutoff_ghz": spec.filter_cutoff,
                "readout_filter_bandwidth_ghz": spec.filter_bandwidth,
            }
        else:
            if target is None or not hasattr(target, "front_panel_values"):
                raise RuntimeError("RF path front-panel target is not available")
            values = dict(target.front_panel_values())
        self._qick_front_panel.set_path_values(values)
        if self._qick_configuration is not None:
            self._qick_front_panel.set_configuration(self._qick_configuration)
        self._qick_front_panel_dialog.show()
        self._qick_front_panel_dialog.raise_()
        self._qick_front_panel_dialog.activateWindow()
        if self._qick_configuration is None:
            QtCore.QTimer.singleShot(0, self._identify_qick_configuration)

    def _identify_qick_configuration(self) -> None:
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "QICK task running",
                "Wait for the current QICK task to finish.",
            )
            return
        try:
            connection, _run = self._experiment_panel.connection_values(
                require_run_config=False
            )
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot identify QICK",
                str(exc),
            )
            return

        self._qick_front_panel.set_identifying(
            True,
            "Connecting and reading HWH configuration...",
        )
        self.statusBar().showMessage("Identifying QICK front-panel configuration")
        thread = QtCore.QThread(self)
        worker = QickConfigurationWorker(connection)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_qick_configuration_identified)
        worker.failed.connect(self._on_qick_configuration_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _on_qick_configuration_identified(self, configuration) -> None:
        self._qick_configuration = configuration
        self._qick_front_panel.set_configuration(configuration)
        self._sparameter_panel.set_front_panel_configuration(configuration)
        self._stability_panel.set_front_panel_configuration(configuration)
        self._rf_ports_panel.set_front_panel_configuration(configuration)
        self._rf_readout_panel.set_front_panel_configuration(configuration)
        self._calibration_panel.set_front_panel_configuration(configuration)
        self._qick_front_panel.set_identifying(
            False,
            f"{configuration.mapped_output_count} DAC / "
            f"{configuration.mapped_input_count} ADC ports mapped",
        )
        self.statusBar().showMessage(
            f"QICK identified: {configuration.mapped_output_count} DAC and "
            f"{configuration.mapped_input_count} ADC front-panel mappings"
        )

    def _on_qick_configuration_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown QICK identification error"
        self._qick_front_panel.set_identifying(False, f"Failed: {summary}")
        self.statusBar().showMessage("QICK front-panel identification failed")
        dialog = DetailedErrorMessageBox(
            "QICK configuration identification failed",
            summary,
            details,
            self,
        )
        dialog.exec_()

    def _apply_front_panel_settings(self, values: Mapping[str, object]) -> None:
        """Commit graphical SMA selections through the existing RF path model."""
        scope = str(values.get("selection_scope", "path"))
        if scope == "output":
            target = self._qick_front_panel_target
            if not isinstance(target, RfPulsePortPanel):
                raise RuntimeError("RF output front-panel target is no longer available")
            target.apply_front_panel_settings(values)
            self._qick_front_panel_dialog.close()
            self.statusBar().showMessage(
                f"RF output {target._index + 1} mapped to "
                f"DAC{values['output_panel_port']} / generator {values['output_ch']}"
            )
            return
        if scope == "input":
            target = self._qick_front_panel_target
            if not isinstance(target, RfReadoutPanel):
                raise RuntimeError("RF readout front-panel target is no longer available")
            target.apply_front_panel_settings(values)
            self._qick_front_panel_dialog.close()
            self.statusBar().showMessage(
                f"RF readout mapped to ADC{values['input_panel_port']} / "
                f"readout {values['readout_ch']}"
            )
            return

        target = self._qick_front_panel_target
        if target is None or not hasattr(target, "apply_front_panel_settings"):
            raise RuntimeError("RF path front-panel target is no longer available")
        target.apply_front_panel_settings(values)
        self._qick_front_panel_dialog.close()
        self.statusBar().showMessage(
            f"Front-panel RF path applied only to {target.__class__.__name__}"
        )

    def _run_sparameter_sweep(self) -> None:
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "QICK task running",
                "Wait for the current QICK task to finish.",
            )
            return
        try:
            arguments = self._sparameter_run_arguments()
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self, "Cannot run RF sweep", str(exc)
            )
            return

        config = arguments["sweep_config"]
        sample_count = max(1, int(np.ceil(config.scan_time_us)))
        power_count = int(config.power_gains.size)
        self._sparameter_panel.set_running(
            True,
            (
                f"0% - Preparing {power_count:,} power point(s) x "
                f"{config.frequency_points:,} frequency points, "
                f"about {sample_count:,} FIR samples per point"
            ),
        )
        self.statusBar().showMessage("RF S-parameter sweep running")
        thread = QtCore.QThread(self)
        worker = SParameterSweepWorker(arguments)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_sparameter_finished)
        worker.failed.connect(self._on_sparameter_failed)
        worker.progress_changed.connect(self._on_sparameter_progress)
        worker.partial_result.connect(self._on_sparameter_partial)
        worker.warning_raised.connect(self._on_sparameter_warning)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _load_sparameter_run(self, run_id: int) -> None:
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "QICK task running",
                "Wait for the current QICK task to finish.",
            )
            return
        try:
            database_path = self._sparameter_panel.database_path_value()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Cannot load RF sweep", str(exc)
            )
            return
        self._sparameter_panel.set_running(
            True,
            "Loading saved RF S-parameter run...",
        )
        thread = QtCore.QThread(self)
        worker = SParameterLoadWorker(database_path, run_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_sparameter_finished)
        worker.failed.connect(self._on_sparameter_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _on_sparameter_progress(self, percent: int, message: str) -> None:
        self._sparameter_panel.update_progress(percent, message)
        self.statusBar().showMessage(f"RF sweep {percent}%: {message}")

    def _on_sparameter_warning(self, message: str) -> None:
        self._sparameter_panel.status.setText(f"Warning: {message}")
        self.statusBar().showMessage(f"RF sweep warning: {message}")
        QtWidgets.QMessageBox.warning(
            self,
            "Input calibration warning",
            message,
        )

    def _on_sparameter_partial(self, stored) -> None:
        self._sparameter_panel.show_partial_result(stored)
        self._sparameter_plot.set_result(stored.result)
        self._dock_sparameter.show()
        self._dock_sparameter.raise_()
        power_count = int(getattr(stored.result, "power_count", 1))
        self.statusBar().showMessage(
            f"RF sweep run {stored.run_id}: {power_count} power point(s) saved"
        )

    def _on_sparameter_finished(self, stored) -> None:
        self._sparameter_panel.show_result(stored)
        self._sparameter_plot.set_result(stored.result)
        self._dock_sparameter.show()
        self._dock_sparameter.raise_()
        self.statusBar().showMessage(
            f"RF S-parameter run {stored.run_id} loaded from {stored.database_path}"
        )

    def _on_sparameter_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown RF S-parameter error"
        self._sparameter_panel.set_running(False, f"Failed: {summary}")
        self.statusBar().showMessage("RF S-parameter sweep failed")
        dialog = DetailedErrorMessageBox(
            "RF S-parameter sweep failed", summary, details, self
        )
        dialog.exec_()

    def _load_noise_trace(self, database_path: str, run_id: int) -> None:
        """Load one saved experiment's I traces outside the GUI thread."""
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "QICK task running",
                "Wait for the current QICK task to finish.",
            )
            return
        self._noise_panel.set_loading(
            True,
            "Loading saved QCoDeS I traces...",
        )
        self.statusBar().showMessage("Loading I traces for noise analysis")
        thread = QtCore.QThread(self)
        worker = NoiseTraceLoadWorker(database_path, run_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_noise_trace_loaded)
        worker.failed.connect(self._on_noise_trace_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _on_noise_trace_loaded(self, collection) -> None:
        self._noise_panel.set_collection(collection)
        self._dock_noise.show()
        self._dock_noise.raise_()
        self.statusBar().showMessage(
            f"Noise analysis loaded {collection.source}: "
            f"{collection.sample_count:,} I samples per trace"
        )

    def _on_noise_trace_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown noise-analysis load error"
        self._noise_panel.show_load_error(summary)
        self.statusBar().showMessage("Noise-analysis trace load failed")
        dialog = DetailedErrorMessageBox(
            "Cannot load noise-analysis trace",
            summary,
            details,
            self,
        )
        dialog.exec_()

    def _run_power_calibration(self, mode: str) -> None:
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "QICK task running",
                "Wait for the current QICK task to finish.",
            )
            return
        try:
            connection, _run = self._experiment_panel.connection_values(
                database_path=self._calibration_panel.database_path_value()
            )
            calibration_config = (
                self._calibration_panel.output_config()
                if mode == "output"
                else (
                    self._calibration_panel.input_config()
                    if mode == "input"
                    else self._calibration_panel.dc_voltage_config()
                )
            )
            if mode == "output" and not (
                calibration_config.oscilloscope.visa_resource.strip()
            ):
                raise ValueError("Oscilloscope VISA resource must not be empty")
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot run calibration",
                str(exc),
            )
            return
        arguments = {
            "connection_config": connection,
            "calibration_config": calibration_config,
        }
        if mode in ("input", "dc_voltage"):
            arguments["tproc_mhz"] = self._experiment_panel.tproc_mhz.value()
        label = {
            "output": "oscilloscope output",
            "input": "FIR-DDR input",
            "dc_voltage": "0 MHz DC voltage",
        }[mode]
        self._calibration_panel.set_running(
            True,
            f"0% - Preparing {label} calibration",
        )
        self.statusBar().showMessage(f"QICK {label} calibration running")
        thread = QtCore.QThread(self)
        worker = CalibrationWorker(mode, arguments)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_calibration_finished)
        worker.failed.connect(self._on_calibration_failed)
        worker.progress_changed.connect(self._on_calibration_progress)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _on_calibration_progress(self, percent: int, message: str) -> None:
        self._calibration_panel.update_progress(percent, message)
        self.statusBar().showMessage(f"Calibration {percent}%: {message}")

    def _on_calibration_finished(self, stored) -> None:
        self._calibration_panel.show_result(stored)
        self.statusBar().showMessage(
            f"Calibration Run {stored.run_id} saved to {stored.database_path}"
        )

    def _on_calibration_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown calibration error"
        self._calibration_panel.set_running(False, f"Failed: {summary}")
        self.statusBar().showMessage("QICK calibration failed")
        dialog = DetailedErrorMessageBox(
            "QICK calibration failed", summary, details, self
        )
        dialog.exec_()

    def _run_stability_diagram(self, *, continuous: bool) -> None:
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "QICK task running",
                "Wait for the current QICK task to finish.",
            )
            return
        try:
            arguments = self._stability_run_arguments(save=not continuous)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot run stability diagram",
                str(exc),
            )
            return

        config = arguments["stability_config"]
        mode = "continuous" if continuous else "single shot"
        self._stability_panel.set_running(
            True,
            (
                f"Preparing {mode} scan: {config.x_axis.points} x "
                f"{config.y_axis.points} points, "
                f"{config.repetitions_per_point} repetitions / point, "
                f"{config.trace_samples_per_point:,} FIR samples / trace"
            ),
        )
        self.statusBar().showMessage(f"QICK stability diagram {mode} running")
        thread = QtCore.QThread(self)
        worker = StabilityDiagramWorker(arguments, continuous=continuous)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.scan_ready.connect(self._on_stability_scan_ready)
        worker.progress_changed.connect(self._on_stability_progress)
        worker.single_finished.connect(self._on_stability_single_finished)
        worker.stopped.connect(self._on_stability_stopped)
        worker.failed.connect(self._on_stability_failed)
        worker.single_finished.connect(thread.quit)
        worker.stopped.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.single_finished.connect(worker.deleteLater)
        worker.stopped.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _stop_stability_diagram(self) -> None:
        worker = self._experiment_worker
        if not isinstance(worker, StabilityDiagramWorker):
            return
        worker.request_stop()
        self._stability_panel.set_stopping()
        self.statusBar().showMessage(
            "Stopping after the active stability scan completes"
        )

    def _on_stability_progress(self, percent: int, message: str) -> None:
        self._stability_panel.update_progress(percent, message)
        self.statusBar().showMessage(
            f"Stability diagram {percent}%: {message}"
        )

    def _on_stability_scan_ready(self, result) -> None:
        self._stability_panel.show_result(result)
        self.statusBar().showMessage(
            f"Stability diagram scan {result.iteration} complete"
        )

    def _on_stability_single_finished(self, stored) -> None:
        self._stability_panel.show_saved_result(stored)
        self.statusBar().showMessage(
            f"Stability diagram QCoDeS Run {stored.run_id} saved to "
            f"{stored.database_path}"
        )

    def _on_stability_stopped(self) -> None:
        self._stability_panel.set_running(False, "Continuous acquisition stopped")
        self.statusBar().showMessage("Stability diagram acquisition stopped")

    def _on_stability_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown stability-diagram error"
        self._stability_panel.set_running(False, f"Failed: {summary}")
        self.statusBar().showMessage("QICK stability diagram failed")
        dialog = DetailedErrorMessageBox(
            "QICK stability diagram failed",
            summary,
            details,
            self,
        )
        dialog.exec_()

    def _run_qick_experiment(self) -> None:
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Experiment running",
                "Wait for the current QICK experiment to finish.",
            )
            return
        try:
            arguments = self._experiment_run_arguments()
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot run experiment", str(exc))
            return

        expected_rows = (
            arguments["sequence"].sweep_point_count
            * arguments["repetitions_per_sweep"]
            * arguments["readout_spec"].samples_per_trigger
        )
        sweep_points = arguments["sequence"].sweep_point_count
        repetitions = arguments["repetitions_per_sweep"]
        self._experiment_panel.set_running(
            True,
            (
                f"0% - Preparing {sweep_points:,} sweep points x "
                f"{repetitions:,} repetitions "
                f"({sweep_points * repetitions:,} acquisitions, "
                f"{expected_rows:,} IQ sample rows)"
            ),
        )
        self.statusBar().showMessage("QICK experiment running")
        thread = QtCore.QThread(self)
        worker = QickExperimentWorker(arguments)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_experiment_finished)
        worker.failed.connect(self._on_experiment_failed)
        worker.progress_changed.connect(self._on_experiment_progress)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _show_qick_program(self) -> None:
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "QICK task running",
                "Wait for the current QICK task to finish.",
            )
            return
        try:
            arguments = self._experiment_run_arguments(
                require_readout=False, require_run_config=False
            )
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot show QICK program", str(exc))
            return

        program_kwargs = {
            "sequence": arguments["sequence"],
            "awg_channels": arguments["awg_channels"],
            "repetitions_per_sweep": arguments["repetitions_per_sweep"],
            "tproc_mhz": self._qick_tproc_mhz,
            "rf_specs": arguments["rf_specs"],
            "readout_spec": arguments["readout_spec"],
        }
        self._experiment_panel.set_running(
            True,
            "Connecting to QICK and compiling tProcessor assembly...",
            show_progress=False,
        )
        self.statusBar().showMessage("Compiling QICK tProcessor assembly")
        thread = QtCore.QThread(self)
        worker = QickProgramWorker(
            arguments["connection_config"], program_kwargs
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_qick_program_ready)
        worker.failed.connect(self._on_qick_program_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_experiment_thread)
        self._experiment_thread = thread
        self._experiment_worker = worker
        thread.start()

    def _on_qick_program_ready(self, result: dict) -> None:
        instruction_count = int(result.get("instruction_count", 0))
        self._experiment_panel.set_running(
            False,
            f"QICK program compiled: {instruction_count:,} instructions",
        )
        self.statusBar().showMessage(
            f"QICK tProcessor program compiled ({instruction_count:,} instructions)"
        )
        dialog = QickAssemblyDialog(result, self)
        dialog.exec_()

    def _on_qick_program_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown QICK program compilation error"
        self._experiment_panel.set_running(False, f"Failed: {summary}")
        self.statusBar().showMessage("QICK program compilation failed")
        dialog = DetailedErrorMessageBox(
            "Cannot show QICK program", summary, details, self
        )
        dialog.exec_()

    def _on_experiment_progress(self, percent: int, message: str) -> None:
        self._experiment_panel.update_progress(percent, message)
        self.statusBar().showMessage(f"QICK experiment {percent}%: {message}")

    def _on_experiment_finished(self, result) -> None:
        self._experiment_panel.show_result(result)
        try:
            self._noise_panel.set_experiment_result(result)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            self._noise_panel.show_load_error(str(exc))
        self.statusBar().showMessage(
            f"QCoDeS run {result.run_id} saved to {result.database_path}"
        )

    def _on_experiment_failed(self, details: str) -> None:
        lines = [line for line in details.rstrip().splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown QICK experiment error"
        self._experiment_panel.set_running(False, f"Failed: {summary}")
        self.statusBar().showMessage("QICK experiment failed")
        dialog = DetailedErrorMessageBox(
            "QICK experiment failed", summary, details, self
        )
        dialog.exec_()

    def _clear_experiment_thread(self) -> None:
        self._experiment_thread = None
        self._experiment_worker = None

    def _refresh_rf_timeline(self, *, fit_view: bool = False) -> None:
        if RfPulseTimelineWidget is None:
            return
        valid_specs = []
        valid_ranges = []
        for spec in self._rf_pulse_specs:
            try:
                start_us, end_us, _ = rf_pulse_absolute_times_us(self._pulse[0], spec)
            except ValueError:
                continue
            valid_specs.append(spec)
            valid_ranges.append((start_us, end_us))
        while len(self._rf_timelines) < len(valid_specs):
            timeline = RfPulseTimelineWidget(self._rf_timeline_container)
            if hasattr(timeline, "set_time_unit"):
                timeline.set_time_unit(self._time_unit)
            timeline.getPlotItem().getViewBox().setXLink(
                self._plot.getPlotItem().getViewBox()
            )
            self._rf_timeline_layout.addWidget(timeline)
            self._rf_timelines.append(timeline)
        while len(self._rf_timelines) > len(valid_specs):
            timeline = self._rf_timelines.pop()
            self._rf_timeline_layout.removeWidget(timeline)
            timeline.deleteLater()
        self._rf_timeline = self._rf_timelines[0] if self._rf_timelines else None
        if not valid_specs:
            self._rf_timeline_container.hide()
            return
        for timeline, spec, (start_us, end_us) in zip(
            self._rf_timelines, valid_specs, valid_ranges
        ):
            timeline.set_pulse(
                gen_ch=spec.gen_ch,
                start_ns=start_us * 1000.0,
                duration_ns=(end_us - start_us) * 1000.0,
                frequency_mhz=spec.frequency_mhz,
                gain=spec.gain,
                phase_degrees=spec.phase_degrees,
                att1_db=spec.att1_db,
                att2_db=spec.att2_db,
            )
            timeline.show()
        self._rf_timeline_container.show()
        total_height = max(360, self._waveform_splitter.height())
        rf_height = min(520, max(130, 130 * len(valid_specs)))
        self._waveform_splitter.setSizes([max(220, total_height - rf_height), rf_height])
        if fit_view:
            self._fit_view()

    def _refresh_rf_editor(self) -> None:
        self._refresh_rf_timeline()
        self._rf_ports_panel.refresh_segments(self._pulse[0])
        self._rf_readout_panel.refresh_segments(self._pulse[0])
        self._stability_rf_ports_panel.refresh_segments(self._pulse[0])
        self._stability_rf_readout_panel.refresh_segments(self._pulse[0])
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
        self._refresh_stability_targets()
        self._refresh_trace_if_needed(force=True)
        self._refresh_sweep_overlay()
        if self._selected_port_idx == 0:
            self._refresh_rf_editor()

    def _flat_update(self, i0: int, i1: int, new_v: float) -> None:
        self.statusBar().showMessage(f"Flat {i0}-{i1} moved to {new_v:.6g} mV")
        self._schedule_deferred_refresh()

    def _point_update(self, i0, i1, new_t):
        self.statusBar().showMessage(
            f"Point {i0}-{i1} moved to "
            f"{_time_from_ns(new_t, self._time_unit):.6g} {self._time_unit}"
        )
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

    def _settings_to_dict(self) -> dict:
        """Return every user-editable experiment setting in canonical units."""
        experiment_values = self._experiment_panel.values(len(self._pulse))
        connection = experiment_values["connection"]
        run = experiment_values["run"]
        self._qick_fabric_mhz = experiment_values["fabric_mhz"]
        self._qick_tproc_mhz = experiment_values["tproc_mhz"]
        self._qick_full_scale_mv = experiment_values["full_scale_mv"]
        self._qick_awg_channels = experiment_values["awg_channels"]
        self._qick_repetitions_per_sweep = experiment_values[
            "repetitions_per_sweep"
        ]
        self._bias_t_compensation_enabled = experiment_values[
            "bias_t_compensation_enabled"
        ]
        self._bias_t_compensation_type = experiment_values[
            "bias_t_compensation_type"
        ]
        self._bias_t_compensation_voltage_mv = experiment_values[
            "bias_t_compensation_voltage_mv"
        ]
        self._bias_t_compensation_mode = experiment_values[
            "bias_t_compensation_mode"
        ]
        self._bias_t_compensation_duration_us = experiment_values[
            "bias_t_compensation_duration_us"
        ]
        self._bias_t_filter_tau_us = experiment_values["bias_t_filter_tau_us"]
        stability_settings = self._stability_panel.settings_dict()
        stability_settings["rf_outputs"] = list(
            self._stability_rf_ports_panel.settings()
        )
        stability_settings["rf_readout"] = (
            self._stability_rf_readout_panel.settings_dict()
        )
        return {
            "schema": SETTINGS_SCHEMA,
            "version": SETTINGS_VERSION,
            "display": {
                "time_unit": self._time_unit,
                "voltage_view": self._plot.voltage_view,
                "selected_awg_output": self._selected_port_idx,
                "selected_control_tab": self._control_tabs.currentIndex(),
                "selected_awg_tuning_tab": self._awg_tuning_tabs.currentIndex(),
            },
            "grid": {
                "time_step_ns": self._grid_time_ns,
                "voltage_step_mv": self._grid_voltage_mv,
                "snap_enabled": self._grid_snap_enabled,
                "visible": self._grid_visible,
                "configured": self._grid_configured,
            },
            "awg": {
                "outputs": [pulse.to_dict() for pulse in self._pulse],
                "cross_capacitance": self._cross_capacitance.tolist(),
                "sweeps": [
                    {
                        "segment_name": spec.segment_name,
                        "output_name": spec.output_name,
                        "start": spec.start,
                        "stop": spec.stop,
                        "count": spec.count,
                    }
                    for spec in self._sweep_specs
                ],
            },
            "qick": {
                "fabric_mhz": self._qick_fabric_mhz,
                "tproc_mhz": self._qick_tproc_mhz,
                "full_scale_mv": self._qick_full_scale_mv,
                "awg_channels": list(self._qick_awg_channels),
                "repetitions_per_sweep": self._qick_repetitions_per_sweep,
                "bias_t_compensation": {
                    "enabled": self._bias_t_compensation_enabled,
                    "type": self._bias_t_compensation_type,
                    "mode": self._bias_t_compensation_mode,
                    "voltage_mv": self._bias_t_compensation_voltage_mv,
                    "duration_us": self._bias_t_compensation_duration_us,
                    "filter_tau_us": self._bias_t_filter_tau_us,
                },
            },
            "experiment": {
                "qick_host": connection.host,
                "ns_port": connection.ns_port,
                "proxy_name": connection.proxy_name,
                "database_path": run.database_path,
                "experiment_name": run.experiment_name,
                "sample_name": run.sample_name,
                "notes": run.notes,
            },
            "rf_outputs": list(self._rf_ports_panel.settings()),
            "rf_readout": self._rf_readout_panel.settings_dict(),
            "stability_diagram": stability_settings,
            "s_parameter": dict(self._sparameter_panel.settings_dict()),
            "calibration": dict(self._calibration_panel.settings_dict()),
            "noise_analysis": dict(self._noise_panel.settings_dict()),
        }

    @staticmethod
    def _json_bool(value, name: str) -> bool:
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
        return value

    @staticmethod
    def _json_finite_float(value, name: str, *, positive: bool = False) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite")
        if positive and result <= 0.0:
            raise ValueError(f"{name} must be positive")
        return result

    @staticmethod
    def _json_int(value, name: str, *, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        result = int(value)
        if result < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return result

    def _decode_rf_output_entries(
        self,
        raw_entries,
        *,
        set_names: set,
        label: str,
    ) -> Tuple[dict, ...]:
        """Validate one tab's independently stored RF output editors."""
        if not isinstance(raw_entries, list) or len(raw_entries) > 8:
            raise ValueError(f"{label} must contain at most eight entries")
        decoded = []
        for index, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, dict):
                raise TypeError(f"each {label} entry must be a JSON object")
            defaults = dict(DEFAULT_RF_OUTPUT_SETTINGS)
            defaults["enabled"] = True
            defaults["gen_ch"] = DEFAULT_QSTL_RF_CHANNELS[index]
            entry = {**defaults, **raw_entry}
            enabled = self._json_bool(entry["enabled"], f"{label} enabled")
            require_within = self._json_bool(
                entry["require_within_segment"],
                f"{label} require_within_segment",
            )
            spec = QickRfPulseSpec(
                gen_ch=self._json_int(entry["gen_ch"], f"{label} generator channel"),
                segment_name=str(entry["segment_name"]),
                delay_us=float(entry["delay_us"]),
                duration_us=float(entry["duration_us"]),
                frequency_mhz=float(entry["frequency_mhz"]),
                gain=int(entry["gain"]),
                att1_db=float(entry["att1_db"]),
                att2_db=float(entry["att2_db"]),
                phase_degrees=float(entry["phase_degrees"]),
                nqz=int(entry["nqz"]),
                require_within_segment=require_within,
                filter_type=str(entry["filter_type"]),
                filter_cutoff=float(entry["filter_cutoff"]),
                filter_bandwidth=float(entry["filter_bandwidth"]),
                output_board_type=str(entry["output_board_type"]),
            )
            if spec.segment_name not in set_names:
                raise ValueError(f"unknown {label} anchor {spec.segment_name!r}")
            decoded.append(
                {
                    "enabled": enabled,
                    "gen_ch": spec.gen_ch,
                    "segment_name": spec.segment_name,
                    "delay_us": spec.delay_us,
                    "duration_us": spec.duration_us,
                    "frequency_mhz": spec.frequency_mhz,
                    "gain": spec.gain,
                    "output_board_type": spec.output_board_type,
                    "att1_db": spec.att1_db,
                    "att2_db": spec.att2_db,
                    "filter_type": spec.filter_type,
                    "filter_cutoff": spec.filter_cutoff,
                    "filter_bandwidth": spec.filter_bandwidth,
                    "phase_degrees": spec.phase_degrees,
                    "nqz": spec.nqz,
                    "require_within_segment": spec.require_within_segment,
                }
            )
        return tuple(decoded)

    def _decode_rf_readout_entry(
        self,
        raw_entry,
        *,
        set_names: set,
        label: str,
    ) -> dict:
        """Validate one tab's independently stored FIR DDR readout editor."""
        if raw_entry is None:
            raw_entry = {}
        if not isinstance(raw_entry, dict):
            raise TypeError(f"{label} must be a JSON object")
        entry = {**DEFAULT_RF_READOUT_SETTINGS, **raw_entry}
        enabled = self._json_bool(entry["enabled"], f"{label} enabled")
        spec = QickDdrReadoutSpec(
            ro_ch=self._json_int(entry["ro_ch"], f"{label} channel"),
            segment_name=str(entry["segment_name"]),
            delay_us=float(entry["delay_us"]),
            samples_per_trigger=self._json_int(
                entry["samples_per_trigger"],
                f"{label} samples_per_trigger",
                minimum=1,
            ),
            readout_frequency_mhz=float(entry["readout_frequency_mhz"]),
            margin_input_samples=self._json_int(
                entry["margin_input_samples"],
                f"{label} margin_input_samples",
            ),
            force_overwrite=self._json_bool(
                entry["force_overwrite"],
                f"{label} force_overwrite",
            ),
            post_run_read_delay_seconds=self._json_finite_float(
                entry.get("post_run_read_delay_seconds", 0.1),
                f"{label} post_run_read_delay_seconds",
            ),
            attenuation_db=float(entry["attenuation_db"]),
            filter_type=str(entry["filter_type"]),
            filter_cutoff=float(entry["filter_cutoff"]),
            filter_bandwidth=float(entry["filter_bandwidth"]),
            input_board_type=str(entry["input_board_type"]),
            dc_gain_db=float(entry["dc_gain_db"]),
            dc_measure_mode=self._json_bool(
                entry["dc_measure_mode"],
                f"{label} dc_measure_mode",
            ),
            dc_measure_gain_v_per_a=self._json_finite_float(
                entry["dc_measure_gain_v_per_a"],
                f"{label} dc_measure_gain_v_per_a",
                positive=True,
            ),
            dc_voltage_calibration_enabled=self._json_bool(
                entry.get("dc_voltage_calibration_enabled", False),
                f"{label} dc_voltage_calibration_enabled",
            ),
            dc_voltage_calibration_database_path=str(
                entry.get("dc_voltage_calibration_database_path", "")
            ),
            dc_voltage_calibration_run_id=self._json_int(
                entry.get("dc_voltage_calibration_run_id", 0),
                f"{label} dc_voltage_calibration_run_id",
            ),
            nqz=self._json_int(entry["nqz"], f"{label} nqz", minimum=1),
        )
        if spec.segment_name not in set_names:
            raise ValueError(f"unknown {label} anchor {spec.segment_name!r}")
        return {
            "enabled": enabled,
            "ro_ch": spec.ro_ch,
            "segment_name": spec.segment_name,
            "delay_us": spec.delay_us,
            "samples_per_trigger": spec.samples_per_trigger,
            "readout_frequency_mhz": spec.readout_frequency_mhz,
            "margin_input_samples": spec.margin_input_samples,
            "force_overwrite": spec.force_overwrite,
            "post_run_read_delay_seconds": spec.post_run_read_delay_seconds,
            "input_board_type": spec.input_board_type,
            "attenuation_db": spec.attenuation_db,
            "dc_gain_db": spec.dc_gain_db,
            "dc_measure_mode": spec.dc_measure_mode,
            "dc_measure_gain_v_per_a": spec.dc_measure_gain_v_per_a,
            "dc_voltage_calibration_enabled": (
                spec.dc_voltage_calibration_enabled
            ),
            "dc_voltage_calibration_database_path": (
                spec.dc_voltage_calibration_database_path
            ),
            "dc_voltage_calibration_run_id": spec.dc_voltage_calibration_run_id,
            "filter_type": spec.filter_type,
            "filter_cutoff": spec.filter_cutoff,
            "filter_bandwidth": spec.filter_bandwidth,
            "nqz": spec.nqz,
        }

    def _decode_settings(self, data: dict) -> dict:
        """Validate a versioned settings document without changing the GUI."""
        if not isinstance(data, dict):
            raise TypeError("settings JSON root must be an object")
        if data.get("schema") != SETTINGS_SCHEMA:
            raise ValueError(f"unsupported settings schema {data.get('schema')!r}")
        settings_version = data.get("version", 1)
        if settings_version not in SUPPORTED_SETTINGS_VERSIONS:
            supported = ", ".join(map(str, SUPPORTED_SETTINGS_VERSIONS))
            raise ValueError(
                f"unsupported settings version {settings_version!r}; "
                f"expected one of {supported}"
            )

        display = data.get("display", {})
        grid = data.get("grid", {})
        awg = data.get("awg")
        qick = data.get("qick", {})
        if not all(isinstance(item, dict) for item in (display, grid, awg, qick)):
            raise TypeError("display, grid, awg, and qick must be JSON objects")

        time_unit = str(display.get("time_unit", DEFAULT_TIME_UNIT))
        if time_unit not in TIME_UNIT_NS:
            raise ValueError(f"unsupported time unit {time_unit!r}")
        voltage_view = str(display.get("voltage_view", "both"))
        if voltage_view not in {"both", "virtual", "physical"}:
            raise ValueError(f"unsupported voltage view {voltage_view!r}")

        raw_outputs = awg.get("outputs")
        if not isinstance(raw_outputs, list) or not 1 <= len(raw_outputs) <= 8:
            raise ValueError("AWG outputs must contain between one and eight waveforms")
        pulses = tuple(PulseSequence.from_dict(entry) for entry in raw_outputs)
        selected_output = self._json_int(
            display.get("selected_awg_output", 0),
            "selected AWG output",
        )
        if selected_output >= len(pulses):
            raise ValueError("selected AWG output is out of range")
        stored_selected_tab = self._json_int(
            display.get("selected_control_tab", 0),
            "selected control tab",
        )
        if settings_version < 12:
            if stored_selected_tab <= 3:
                selected_tab = 0
                selected_awg_tuning_tab = stored_selected_tab
            elif stored_selected_tab in {4, 6}:
                # Version 10 briefly stored the former full-size Front Panel
                # tab at index 6; restore that view to RF S-Parameter.
                selected_tab = self._control_tabs.indexOf(self._sparameter_panel)
                selected_awg_tuning_tab = 0
            elif stored_selected_tab == 5:
                selected_tab = self._control_tabs.indexOf(self._calibration_panel)
                selected_awg_tuning_tab = 0
            else:
                raise ValueError("selected control tab is out of range")
        elif settings_version == 12:
            selected_awg_tuning_tab = self._json_int(
                display.get("selected_awg_tuning_tab", 0),
                "selected AWG Tuning tab",
            )
            if stored_selected_tab == 0:
                selected_tab = self._control_tabs.indexOf(self._awg_tuning_page)
            elif stored_selected_tab == 1:
                selected_tab = self._control_tabs.indexOf(self._sparameter_panel)
            elif stored_selected_tab == 2:
                selected_tab = self._control_tabs.indexOf(self._calibration_panel)
            else:
                raise ValueError("selected control tab is out of range")
        else:
            selected_tab = stored_selected_tab
            selected_awg_tuning_tab = self._json_int(
                display.get("selected_awg_tuning_tab", 0),
                "selected AWG Tuning tab",
            )
        if selected_tab >= self._control_tabs.count():
            raise ValueError("selected control tab is out of range")
        if selected_awg_tuning_tab >= self._awg_tuning_tabs.count():
            raise ValueError("selected AWG Tuning tab is out of range")

        expected_shape = (len(pulses), len(pulses))
        if "cross_capacitance" in awg:
            matrix = np.asarray(awg["cross_capacitance"], dtype=float)
        else:
            matrix = np.eye(len(pulses), dtype=float)
        if matrix.shape != expected_shape or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"cross-capacitance matrix must be a finite {expected_shape} array"
            )
        if not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("cross-capacitance diagonal entries must equal 1")

        raw_sweeps = awg.get("sweeps", [])
        if not isinstance(raw_sweeps, list):
            raise TypeError("AWG sweeps must be a JSON array")
        sweeps = tuple(
            QickSweepSpec(
                segment_name=str(entry["segment_name"]),
                output_name=str(entry["output_name"]),
                start=float(entry["start"]),
                stop=float(entry["stop"]),
                count=self._json_int(entry["count"], "sweep count", minimum=1),
            )
            for entry in raw_sweeps
        )
        sweep_targets = set()
        for spec in sweeps:
            target = (spec.output_name, spec.segment_name)
            if target in sweep_targets:
                raise ValueError("each AWG output/SET may have only one sweep")
            sweep_targets.add(target)
            try:
                output_index = int(spec.output_name.rsplit("_", 1)[1])
                segment_index = int(spec.segment_name.rsplit("_", 1)[1])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid sweep target {target!r}") from exc
            if spec.output_name != f"awg_{output_index}":
                raise ValueError(f"invalid sweep output {spec.output_name!r}")
            if not 0 <= output_index < len(pulses):
                raise ValueError(f"unknown sweep output {spec.output_name!r}")
            if not 0 <= segment_index < pulses[output_index].set_count:
                raise ValueError(f"unknown sweep SET {spec.segment_name!r}")

        fabric_mhz = self._json_finite_float(
            qick.get("fabric_mhz", DEFAULT_QICK_FABRIC_MHZ),
            "QICK fabric_mhz",
            positive=True,
        )
        tproc_mhz = self._json_finite_float(
            qick.get("tproc_mhz", DEFAULT_QICK_TPROC_MHZ),
            "QICK tproc_mhz",
            positive=True,
        )
        full_scale_mv = self._json_finite_float(
            qick.get("full_scale_mv", DEFAULT_QICK_FULL_SCALE_MV),
            "QICK full_scale_mv",
            positive=True,
        )
        raw_stability_settings = data.get("stability_diagram")
        stability_settings = normalize_stability_settings(
            raw_stability_settings,
            output_names=tuple(f"awg_{index}" for index in range(len(pulses))),
            segment_names=qick_set_segment_names(pulses[0]),
        )
        noise_analysis_settings = normalize_noise_analysis_settings(
            data.get("noise_analysis")
        )
        for axis_name in ("x_axis", "y_axis"):
            axis = stability_settings[axis_name]
            if max(abs(axis["start_mv"]), abs(axis["stop_mv"])) > full_scale_mv:
                raise ValueError(
                    f"{axis_name} stability sweep exceeds +/-{full_scale_mv:g} mV "
                    "AWG full scale"
                )
        raw_bias_t = qick.get("bias_t_compensation", {})
        if raw_bias_t is None:
            raw_bias_t = {}
        if not isinstance(raw_bias_t, dict):
            raise TypeError("QICK bias_t_compensation must be a JSON object")
        bias_t_enabled = self._json_bool(
            raw_bias_t.get("enabled", False),
            "QICK Bias-T compensation enabled",
        )
        bias_t_type = str(raw_bias_t.get("type", "dc"))
        if bias_t_type not in BIAS_T_COMPENSATION_TYPES:
            raise ValueError(
                "QICK Bias-T compensation type must be one of "
                f"{BIAS_T_COMPENSATION_TYPES}"
            )
        bias_t_mode = str(raw_bias_t.get("mode", "fixed_voltage"))
        if bias_t_mode not in BIAS_T_COMPENSATION_MODES:
            raise ValueError(
                f"QICK Bias-T compensation mode must be one of "
                f"{BIAS_T_COMPENSATION_MODES}"
            )
        bias_t_compensation_mv = self._json_finite_float(
            raw_bias_t.get(
                "voltage_mv",
                full_scale_mv * DEFAULT_BIAS_T_COMPENSATION_FRACTION,
            ),
            "QICK Bias-T compensation voltage_mv",
            positive=True,
        )
        if bias_t_compensation_mv > full_scale_mv:
            raise ValueError(
                "QICK Bias-T compensation voltage exceeds full_scale_mv"
            )
        bias_t_duration_us = self._json_finite_float(
            raw_bias_t.get(
                "duration_us",
                DEFAULT_BIAS_T_COMPENSATION_DURATION_US,
            ),
            "QICK Bias-T compensation duration_us",
            positive=True,
        )
        bias_t_filter_tau_us = self._json_finite_float(
            raw_bias_t.get("filter_tau_us", DEFAULT_BIAS_T_FILTER_TAU_US),
            "QICK Bias-T filter_tau_us",
            positive=True,
        )
        repetitions = self._json_int(
            qick.get("repetitions_per_sweep", 1),
            "QICK repetitions_per_sweep",
            minimum=1,
        )
        raw_channels = qick.get(
            "awg_channels", list(DEFAULT_QSTL_AWG_CHANNELS[:len(pulses)])
        )
        if not isinstance(raw_channels, list) or len(raw_channels) != len(pulses):
            raise ValueError("QICK awg_channels must match the AWG output count")
        awg_channels = tuple(
            self._json_int(channel, "QICK AWG channel") for channel in raw_channels
        )
        if len(set(awg_channels)) != len(awg_channels):
            raise ValueError("QICK AWG channels must be unique")

        experiment = data.get("experiment", {})
        if not isinstance(experiment, dict):
            raise TypeError("experiment must be a JSON object")
        connection_config = QickConnectionConfig(
            host=str(experiment.get("qick_host", DEFAULT_QICK_HOST)),
            ns_port=self._json_int(
                experiment.get("ns_port", DEFAULT_QICK_NS_PORT),
                "QICK nameserver port",
                minimum=1,
            ),
            proxy_name=str(
                experiment.get("proxy_name", DEFAULT_QICK_PROXY_NAME)
            ),
        )
        run_config = QcodesRunConfig(
            database_path=str(
                experiment.get("database_path", DEFAULT_QCODES_DB_PATH)
            ),
            experiment_name=str(
                experiment.get("experiment_name", "QICK pulse experiment")
            ),
            sample_name=str(experiment.get("sample_name", "PulseGenerator")),
            notes=str(experiment.get("notes", "")),
        )

        raw_sparameter = data.get("s_parameter", {})
        if raw_sparameter is None:
            raw_sparameter = {}
        if not isinstance(raw_sparameter, dict):
            raise TypeError("s_parameter must be a JSON object")
        sparameter_settings = {
            **DEFAULT_SPARAMETER_SETTINGS,
            **raw_sparameter,
        }
        sparameter_database_path = str(
            sparameter_settings.pop("database_path")
        ).strip()
        if not sparameter_database_path:
            raise ValueError("RF S-parameter database path must not be empty")
        sparameter_config = SParameterSweepConfig(**sparameter_settings)

        raw_calibration = data.get("calibration", {})
        if raw_calibration is None:
            raw_calibration = {}
        if not isinstance(raw_calibration, dict):
            raise TypeError("calibration must be a JSON object")
        calibration_defaults = default_calibration_settings()
        calibration_database_path = str(
            raw_calibration.get(
                "database_path",
                calibration_defaults["database_path"],
            )
        ).strip()
        if not calibration_database_path:
            raise ValueError("calibration database path must not be empty")
        raw_output_calibration = raw_calibration.get("output", {})
        raw_input_calibration = raw_calibration.get("input", {})
        raw_dc_voltage_calibration = raw_calibration.get("dc_voltage", {})
        dc_application_explicit = "dc_voltage_application" in raw_calibration
        raw_dc_application = raw_calibration.get(
            "dc_voltage_application",
            calibration_defaults["dc_voltage_application"],
        )
        if not isinstance(raw_output_calibration, dict) or not isinstance(
            raw_input_calibration,
            dict,
        ):
            raise TypeError("calibration output and input must be JSON objects")
        if not isinstance(raw_dc_voltage_calibration, dict):
            raise TypeError("calibration dc_voltage must be a JSON object")
        if not isinstance(raw_dc_application, dict):
            raise TypeError(
                "calibration dc_voltage_application must be a JSON object"
            )
        output_calibration_values = {
            **calibration_defaults["output"],
            **raw_output_calibration,
        }
        scope_values = {
            **calibration_defaults["output"]["oscilloscope"],
            **dict(output_calibration_values.pop("oscilloscope", {})),
        }
        output_calibration = OutputPowerCalibrationConfig(
            database_path=calibration_database_path,
            oscilloscope=OscilloscopeConfig(**scope_values),
            **output_calibration_values,
        )
        input_calibration = InputPowerCalibrationConfig(
            database_path=calibration_database_path,
            **{
                **calibration_defaults["input"],
                **raw_input_calibration,
            },
        )
        dc_voltage_calibration = DcVoltageCalibrationConfig(
            database_path=calibration_database_path,
            **{
                **calibration_defaults["dc_voltage"],
                **raw_dc_voltage_calibration,
            },
        )
        calibration_selected_tab = self._json_int(
            raw_calibration.get(
                "selected_tab",
                calibration_defaults["selected_tab"],
            ),
            "calibration selected tab",
        )
        if calibration_selected_tab > 2:
            raise ValueError("calibration selected tab must be 0, 1, or 2")
        raw_input_calibration_plot = raw_calibration.get("input_plot", {})
        if not isinstance(raw_input_calibration_plot, dict):
            raise TypeError("calibration input_plot must be a JSON object")
        input_calibration_plot = {
            **calibration_defaults["input_plot"],
            **raw_input_calibration_plot,
        }
        for axis_name in ("x_scale", "y_scale"):
            axis_scale = str(input_calibration_plot[axis_name]).lower()
            if axis_scale not in ("linear", "log"):
                raise ValueError(
                    f"calibration input plot {axis_name} must be linear or log"
                )
            input_calibration_plot[axis_name] = axis_scale
        dc_application_enabled = self._json_bool(
            raw_dc_application.get("enabled", False),
            "DC voltage calibration application enabled",
        )
        dc_application_database_path = str(
            raw_dc_application.get(
                "database_path",
                calibration_database_path,
            )
        ).strip()
        if dc_application_enabled and not dc_application_database_path:
            raise ValueError(
                "DC voltage calibration application database path must not be empty"
            )
        dc_application_run_id = self._json_int(
            raw_dc_application.get("run_id", 0),
            "DC voltage calibration application Run ID",
        )
        output_calibration_settings = asdict(output_calibration)
        input_calibration_settings = asdict(input_calibration)
        dc_voltage_calibration_settings = asdict(dc_voltage_calibration)
        output_calibration_settings.pop("database_path")
        input_calibration_settings.pop("database_path")
        dc_voltage_calibration_settings.pop("database_path")
        calibration_settings = {
            "database_path": calibration_database_path,
            "selected_tab": calibration_selected_tab,
            "output": output_calibration_settings,
            "input": input_calibration_settings,
            "dc_voltage": dc_voltage_calibration_settings,
            "dc_voltage_application": {
                "enabled": dc_application_enabled,
                "database_path": dc_application_database_path,
                "run_id": dc_application_run_id,
            },
            "input_plot": input_calibration_plot,
        }

        set_names = {f"set_{index}" for index in range(pulses[0].set_count)}
        if "rf_outputs" in data:
            raw_rf_outputs = data["rf_outputs"]
        else:
            raw_rf_outputs = [dict(DEFAULT_RF_OUTPUT_SETTINGS)]
        if not isinstance(raw_rf_outputs, list) or len(raw_rf_outputs) > 8:
            raise ValueError("rf_outputs must contain at most eight entries")
        rf_outputs = []
        for index, entry in enumerate(raw_rf_outputs):
            if not isinstance(entry, dict):
                raise TypeError("each RF output setting must be a JSON object")
            defaults = dict(DEFAULT_RF_OUTPUT_SETTINGS)
            defaults["enabled"] = True
            defaults["gen_ch"] = DEFAULT_QSTL_RF_CHANNELS[index]
            entry = {**defaults, **entry}
            enabled = self._json_bool(entry["enabled"], "RF output enabled")
            require_within = self._json_bool(
                entry["require_within_segment"],
                "RF require_within_segment",
            )
            spec = QickRfPulseSpec(
                gen_ch=self._json_int(entry["gen_ch"], "RF generator channel"),
                segment_name=str(entry["segment_name"]),
                delay_us=float(entry["delay_us"]),
                duration_us=float(entry["duration_us"]),
                frequency_mhz=float(entry["frequency_mhz"]),
                gain=int(entry["gain"]),
                att1_db=float(entry["att1_db"]),
                att2_db=float(entry["att2_db"]),
                phase_degrees=float(entry["phase_degrees"]),
                nqz=int(entry["nqz"]),
                require_within_segment=require_within,
                filter_type=str(entry["filter_type"]),
                filter_cutoff=float(entry["filter_cutoff"]),
                filter_bandwidth=float(entry["filter_bandwidth"]),
                output_board_type=str(entry["output_board_type"]),
            )
            if spec.segment_name not in set_names:
                raise ValueError(f"unknown RF output anchor {spec.segment_name!r}")
            rf_outputs.append({"enabled": enabled, **{
                "gen_ch": spec.gen_ch,
                "segment_name": spec.segment_name,
                "delay_us": spec.delay_us,
                "duration_us": spec.duration_us,
                "frequency_mhz": spec.frequency_mhz,
                "gain": spec.gain,
                "output_board_type": spec.output_board_type,
                "att1_db": spec.att1_db,
                "att2_db": spec.att2_db,
                "filter_type": spec.filter_type,
                "filter_cutoff": spec.filter_cutoff,
                "filter_bandwidth": spec.filter_bandwidth,
                "phase_degrees": spec.phase_degrees,
                "nqz": spec.nqz,
                "require_within_segment": spec.require_within_segment,
            }})

        raw_readout = data.get("rf_readout", {})
        if raw_readout is None:
            raw_readout = {}
        if not isinstance(raw_readout, dict):
            raise TypeError("rf_readout must be a JSON object")
        raw_readout = {**DEFAULT_RF_READOUT_SETTINGS, **raw_readout}
        readout_enabled = self._json_bool(
            raw_readout["enabled"], "RF readout enabled"
        )
        force_overwrite = self._json_bool(
            raw_readout["force_overwrite"],
            "RF readout force_overwrite",
        )
        readout_spec = QickDdrReadoutSpec(
            ro_ch=self._json_int(raw_readout["ro_ch"], "RF readout channel"),
            segment_name=str(raw_readout["segment_name"]),
            delay_us=float(raw_readout["delay_us"]),
            samples_per_trigger=self._json_int(
                raw_readout["samples_per_trigger"],
                "RF readout samples_per_trigger",
                minimum=1,
            ),
            readout_frequency_mhz=float(raw_readout["readout_frequency_mhz"]),
            margin_input_samples=self._json_int(
                raw_readout["margin_input_samples"],
                "RF readout margin_input_samples",
            ),
            force_overwrite=force_overwrite,
            post_run_read_delay_seconds=self._json_finite_float(
                raw_readout.get("post_run_read_delay_seconds", 0.1),
                "RF readout post_run_read_delay_seconds",
            ),
            attenuation_db=float(raw_readout["attenuation_db"]),
            filter_type=str(raw_readout["filter_type"]),
            filter_cutoff=float(raw_readout["filter_cutoff"]),
            filter_bandwidth=float(raw_readout["filter_bandwidth"]),
            input_board_type=str(raw_readout["input_board_type"]),
            dc_gain_db=float(raw_readout["dc_gain_db"]),
            dc_measure_mode=self._json_bool(
                raw_readout["dc_measure_mode"],
                "RF readout dc_measure_mode",
            ),
            dc_measure_gain_v_per_a=self._json_finite_float(
                raw_readout["dc_measure_gain_v_per_a"],
                "RF readout dc_measure_gain_v_per_a",
                positive=True,
            ),
            dc_voltage_calibration_enabled=self._json_bool(
                raw_readout.get("dc_voltage_calibration_enabled", False),
                "RF readout dc_voltage_calibration_enabled",
            ),
            dc_voltage_calibration_database_path=str(
                raw_readout.get("dc_voltage_calibration_database_path", "")
            ),
            dc_voltage_calibration_run_id=self._json_int(
                raw_readout.get("dc_voltage_calibration_run_id", 0),
                "RF readout dc_voltage_calibration_run_id",
            ),
            nqz=self._json_int(raw_readout["nqz"], "RF readout nqz", minimum=1),
        )
        if readout_spec.segment_name not in set_names:
            raise ValueError(
                f"unknown RF readout anchor {readout_spec.segment_name!r}"
            )
        rf_readout = {
            "enabled": readout_enabled,
            "ro_ch": readout_spec.ro_ch,
            "segment_name": readout_spec.segment_name,
            "delay_us": readout_spec.delay_us,
            "samples_per_trigger": readout_spec.samples_per_trigger,
            "readout_frequency_mhz": readout_spec.readout_frequency_mhz,
            "margin_input_samples": readout_spec.margin_input_samples,
            "force_overwrite": readout_spec.force_overwrite,
            "post_run_read_delay_seconds": (
                readout_spec.post_run_read_delay_seconds
            ),
            "input_board_type": readout_spec.input_board_type,
            "attenuation_db": readout_spec.attenuation_db,
            "dc_gain_db": readout_spec.dc_gain_db,
            "dc_measure_mode": readout_spec.dc_measure_mode,
            "dc_measure_gain_v_per_a": readout_spec.dc_measure_gain_v_per_a,
            "dc_voltage_calibration_enabled": (
                readout_spec.dc_voltage_calibration_enabled
            ),
            "dc_voltage_calibration_database_path": (
                readout_spec.dc_voltage_calibration_database_path
            ),
            "dc_voltage_calibration_run_id": (
                readout_spec.dc_voltage_calibration_run_id
            ),
            "filter_type": readout_spec.filter_type,
            "filter_cutoff": readout_spec.filter_cutoff,
            "filter_bandwidth": readout_spec.filter_bandwidth,
            "nqz": readout_spec.nqz,
        }
        if isinstance(raw_stability_settings, Mapping) and (
            "rf_outputs" in raw_stability_settings
        ):
            stability_rf_outputs = self._decode_rf_output_entries(
                raw_stability_settings["rf_outputs"],
                set_names=set_names,
                label="stability RF outputs",
            )
        else:
            stability_rf_outputs = tuple(dict(entry) for entry in rf_outputs)
        if isinstance(raw_stability_settings, Mapping) and (
            "rf_readout" in raw_stability_settings
        ):
            stability_rf_readout = self._decode_rf_readout_entry(
                raw_stability_settings["rf_readout"],
                set_names=set_names,
                label="stability RF readout",
            )
        else:
            stability_rf_readout = dict(rf_readout)
        stability_settings["rf_outputs"] = stability_rf_outputs
        stability_settings["rf_readout"] = stability_rf_readout
        if not dc_application_explicit:
            calibration_settings["dc_voltage_application"] = {
                "enabled": readout_spec.dc_voltage_calibration_enabled,
                "database_path": (
                    readout_spec.dc_voltage_calibration_database_path
                    or calibration_database_path
                ),
                "run_id": readout_spec.dc_voltage_calibration_run_id,
            }

        return {
            "pulses": pulses,
            "cross_capacitance": matrix,
            "sweeps": sweeps,
            "time_unit": time_unit,
            "voltage_view": voltage_view,
            "selected_output": selected_output,
            "selected_tab": selected_tab,
            "selected_awg_tuning_tab": selected_awg_tuning_tab,
            "grid_time_ns": self._json_finite_float(
                grid.get("time_step_ns", 1000.0),
                "grid time_step_ns",
                positive=True,
            ),
            "grid_voltage_mv": self._json_finite_float(
                grid.get("voltage_step_mv", 100.0),
                "grid voltage_step_mv",
                positive=True,
            ),
            "grid_snap": self._json_bool(
                grid.get("snap_enabled", False), "grid snap_enabled"
            ),
            "grid_visible": self._json_bool(
                grid.get("visible", True), "grid visible"
            ),
            "grid_configured": self._json_bool(
                grid.get("configured", False), "grid configured"
            ),
            "fabric_mhz": fabric_mhz,
            "tproc_mhz": tproc_mhz,
            "full_scale_mv": full_scale_mv,
            "awg_channels": awg_channels,
            "repetitions": repetitions,
            "bias_t_enabled": bias_t_enabled,
            "bias_t_type": bias_t_type,
            "bias_t_mode": bias_t_mode,
            "bias_t_compensation_mv": bias_t_compensation_mv,
            "bias_t_duration_us": bias_t_duration_us,
            "bias_t_filter_tau_us": bias_t_filter_tau_us,
            "connection_config": connection_config,
            "run_config": run_config,
            "rf_outputs": tuple(rf_outputs),
            "rf_readout": rf_readout,
            "stability_diagram": stability_settings,
            "s_parameter": {
                "database_path": sparameter_database_path,
                **asdict(sparameter_config),
            },
            "calibration": calibration_settings,
            "noise_analysis": noise_analysis_settings,
        }

    def _apply_decoded_settings(self, settings: dict) -> None:
        """Apply a fully validated settings object to all GUI panels."""
        pulses = settings["pulses"]
        self._sweep_specs = []
        while len(self._pulse) > len(pulses):
            self._delete_port(len(self._pulse) - 1)
        while len(self._pulse) < len(pulses):
            self._add_port()
        for target, source in zip(self._pulse, pulses):
            target.t = source.t.copy()
            target.v = source.v.copy()
            target.v_bounds = tuple(source.v_bounds)

        self._cross_capacitance = settings["cross_capacitance"].copy()
        self._sweep_specs = list(settings["sweeps"])
        self._qick_fabric_mhz = settings["fabric_mhz"]
        self._qick_tproc_mhz = settings["tproc_mhz"]
        self._qick_full_scale_mv = settings["full_scale_mv"]
        self._qick_awg_channels = tuple(settings["awg_channels"])
        self._qick_repetitions_per_sweep = settings["repetitions"]
        self._bias_t_compensation_enabled = settings["bias_t_enabled"]
        self._bias_t_compensation_type = settings["bias_t_type"]
        self._bias_t_compensation_voltage_mv = settings[
            "bias_t_compensation_mv"
        ]
        self._bias_t_compensation_mode = settings["bias_t_mode"]
        self._bias_t_compensation_duration_us = settings["bias_t_duration_us"]
        self._bias_t_filter_tau_us = settings["bias_t_filter_tau_us"]
        self._experiment_panel.load_settings(
            settings["connection_config"],
            settings["run_config"],
            fabric_mhz=self._qick_fabric_mhz,
            tproc_mhz=self._qick_tproc_mhz,
            full_scale_mv=self._qick_full_scale_mv,
            awg_channels=self._qick_awg_channels,
            repetitions=self._qick_repetitions_per_sweep,
            bias_t_enabled=self._bias_t_compensation_enabled,
            bias_t_compensation_type=self._bias_t_compensation_type,
            bias_t_compensation_mv=self._bias_t_compensation_voltage_mv,
            bias_t_mode=self._bias_t_compensation_mode,
            bias_t_duration_us=self._bias_t_compensation_duration_us,
            bias_t_filter_tau_us=self._bias_t_filter_tau_us,
        )
        self._refresh_stability_targets()
        self._stability_panel.load_settings(settings["stability_diagram"])

        with QtCore.QSignalBlocker(self._time_unit_combo):
            self._time_unit_combo.setCurrentText(settings["time_unit"])
        self._set_time_unit(settings["time_unit"])
        self._grid_configured = settings["grid_configured"]
        self._set_grid_settings(
            time_step_ns=settings["grid_time_ns"],
            voltage_step_mv=settings["grid_voltage_mv"],
            snap_enabled=settings["grid_snap"],
            visible=settings["grid_visible"],
        )

        self._rf_ports_panel.refresh_segments(self._pulse[0])
        self._rf_ports_panel.load_settings(settings["rf_outputs"])
        self._rf_readout_panel.refresh_segments(self._pulse[0])
        self._rf_readout_panel.load_settings(settings["rf_readout"])
        self._stability_rf_ports_panel.refresh_segments(self._pulse[0])
        self._stability_rf_ports_panel.load_settings(
            settings["stability_diagram"]["rf_outputs"]
        )
        self._stability_rf_readout_panel.refresh_segments(self._pulse[0])
        self._stability_rf_readout_panel.load_settings(
            settings["stability_diagram"]["rf_readout"]
        )
        self._on_stability_readout_spec_changed(
            self._stability_rf_readout_panel.spec()
        )
        self._rf_pulse_specs = list(self._rf_ports_panel.specs())
        self._rf_pulse_spec = self._rf_pulse_specs[0] if self._rf_pulse_specs else None
        self._ddr_readout_spec = self._rf_readout_panel.spec()
        self._sparameter_panel.load_settings(settings["s_parameter"])
        self._calibration_panel.load_settings(settings["calibration"])
        self._noise_panel.load_settings(settings["noise_analysis"])
        self._qick_front_panel.set_path_values(
            self._sparameter_panel.front_panel_values()
        )

        self._selected_port_idx = settings["selected_output"]
        self._plot.set_selected_port_idx(self._selected_port_idx)
        self._control_tabs.setCurrentIndex(settings["selected_tab"])
        self._awg_tuning_tabs.setCurrentIndex(
            settings["selected_awg_tuning_tab"]
        )
        with QtCore.QSignalBlocker(self._voltage_view_actions[settings["voltage_view"]]):
            self._voltage_view_actions[settings["voltage_view"]].setChecked(True)
        self._set_voltage_view(settings["voltage_view"])

        self._plot.refresh()
        self._multi_ctrl.refresh_table()
        self.refresh_panel_table()
        self._refresh_trace_if_needed(force=True)
        self._refresh_rf_timeline(fit_view=True)
        self._refresh_sweep_overlay(fit_view=True, sync_rows=True)

    def _apply_legacy_settings(self, data: dict) -> None:
        """Read the original single-waveform JSON format."""
        loaded = self._pulse[0].copy()
        if "initial_voltage" in data:
            loaded.v[0:2] = float(data["initial_voltage"])
        if "voltage_bounds" in data and len(data["voltage_bounds"]) == 2:
            loaded.v_bounds = tuple(map(float, data["voltage_bounds"]))
        if "time_ns" in data and "voltage_mv" in data:
            loaded = PulseSequence.from_dict(data)
        loaded.validate()
        self._pulse[0].t = loaded.t.copy()
        self._pulse[0].v = loaded.v.copy()
        self._pulse[0].v_bounds = tuple(loaded.v_bounds)
        self._plot.refresh()
        self._multi_ctrl.refresh_table()
        self._refresh_trace_if_needed(force=True)
        self._refresh_rf_editor()
        self._refresh_sweep_overlay(fit_view=True, sync_rows=True)

    def _save_settings_json(self, path) -> Path:
        output_path = Path(path)
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")
        text = json.dumps(
            self._settings_to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        output_path.write_text(text + "\n", encoding="utf-8")
        self._settings_path = output_path
        return output_path

    def _load_settings_json(self, path) -> Path:
        input_path = Path(path)
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema") == SETTINGS_SCHEMA:
            self._apply_decoded_settings(self._decode_settings(data))
        else:
            if not isinstance(data, dict):
                raise TypeError("legacy settings JSON root must be an object")
            self._apply_legacy_settings(data)
        self._settings_path = input_path
        return input_path

    def _save_json(self) -> None:
        default_path = self._settings_path or (
            Path(__file__).resolve().parent / "pulse_generator_settings.json"
        )
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save settings JSON",
            str(default_path),
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            saved_path = self._save_settings_json(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Save failed", f"Could not write JSON:\n{exc}"
            )
            return
        self.statusBar().showMessage(f"Saved settings to {saved_path}")

    def _load_json(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open settings JSON",
            str(self._settings_path.parent if self._settings_path else ""),
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            loaded_path = self._load_settings_json(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Load failed", f"Could not read JSON:\n{exc}"
            )
            return
        self.statusBar().showMessage(f"Loaded settings from {loaded_path}")

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

    def _next_qick_awg_channel(self) -> int:
        used = set(self._qick_awg_channels)
        for channel in DEFAULT_QSTL_AWG_CHANNELS:
            if channel not in used:
                return channel
        channel = 0
        while channel in used:
            channel += 1
        return channel

    def _qick_export_settings(self) -> Optional[dict]:
        if len(self._pulse) > 8:
            QtWidgets.QMessageBox.warning(
                self,
                "QICK export unavailable",
                "axis_awg_tuning_v1 supports at most eight outputs.",
            )
            return None
        try:
            experiment_values = self._experiment_panel.values(len(self._pulse))
            self._qick_fabric_mhz = experiment_values["fabric_mhz"]
            self._qick_tproc_mhz = experiment_values["tproc_mhz"]
            self._qick_full_scale_mv = experiment_values["full_scale_mv"]
            self._qick_awg_channels = experiment_values["awg_channels"]
            self._qick_repetitions_per_sweep = experiment_values[
                "repetitions_per_sweep"
            ]
            self._bias_t_compensation_enabled = experiment_values[
                "bias_t_compensation_enabled"
            ]
            self._bias_t_compensation_type = experiment_values[
                "bias_t_compensation_type"
            ]
            self._bias_t_compensation_voltage_mv = experiment_values[
                "bias_t_compensation_voltage_mv"
            ]
            self._bias_t_compensation_mode = experiment_values[
                "bias_t_compensation_mode"
            ]
            self._bias_t_compensation_duration_us = experiment_values[
                "bias_t_compensation_duration_us"
            ]
            self._bias_t_filter_tau_us = experiment_values[
                "bias_t_filter_tau_us"
            ]
            set_names = qick_set_segment_names(self._pulse[0])
            self._rf_pulse_specs = list(self._rf_ports_panel.specs())
            self._rf_pulse_spec = (
                self._rf_pulse_specs[0] if self._rf_pulse_specs else None
            )
            self._ddr_readout_spec = self._rf_readout_panel.spec()
        except (TypeError, ValueError) as exc:
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
            initial_rf_specs=tuple(self._rf_pulse_specs),
            initial_ddr_readout_spec=self._ddr_readout_spec,
            initial_sweeps=tuple(self._sweep_specs),
            initial_cross_capacitance=self._cross_capacitance,
            initial_fabric_mhz=self._qick_fabric_mhz,
            initial_tproc_mhz=self._qick_tproc_mhz,
            initial_full_scale_mv=self._qick_full_scale_mv,
            initial_awg_channels=self._qick_awg_channels,
            initial_repetitions=self._qick_repetitions_per_sweep,
            initial_bias_t_enabled=self._bias_t_compensation_enabled,
            initial_bias_t_compensation_type=self._bias_t_compensation_type,
            initial_bias_t_compensation_mv=self._bias_t_compensation_voltage_mv,
            initial_bias_t_mode=self._bias_t_compensation_mode,
            initial_bias_t_duration_us=self._bias_t_compensation_duration_us,
            initial_bias_t_filter_tau_us=self._bias_t_filter_tau_us,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None
        settings = dialog.values()
        self._rf_pulse_specs = list(settings["rf_pulse_specs"])
        self._rf_pulse_spec = self._rf_pulse_specs[0] if self._rf_pulse_specs else None
        self._ddr_readout_spec = settings["ddr_readout_spec"]
        if settings["sweeps"] is not None:
            self._sweep_specs = list(settings["sweeps"])
        elif settings["sweep"] is not None:
            self._sweep_specs = [settings["sweep"]]
        else:
            self._sweep_specs = []
        self._qick_fabric_mhz = settings["fabric_mhz"]
        self._qick_tproc_mhz = settings["tproc_mhz"]
        self._qick_full_scale_mv = settings["full_scale_mv"]
        self._qick_awg_channels = tuple(settings["awg_channels"])
        self._qick_repetitions_per_sweep = settings["repetitions_per_sweep"]
        self._bias_t_compensation_enabled = settings[
            "bias_t_compensation_enabled"
        ]
        self._bias_t_compensation_type = settings["bias_t_compensation_type"]
        self._bias_t_compensation_voltage_mv = settings[
            "bias_t_compensation_voltage_mv"
        ]
        self._bias_t_compensation_mode = settings["bias_t_compensation_mode"]
        self._bias_t_compensation_duration_us = settings[
            "bias_t_compensation_duration_us"
        ]
        self._bias_t_filter_tau_us = settings["bias_t_filter_tau_us"]
        self._experiment_panel.set_qick_values(
            fabric_mhz=self._qick_fabric_mhz,
            tproc_mhz=self._qick_tproc_mhz,
            full_scale_mv=self._qick_full_scale_mv,
            awg_channels=self._qick_awg_channels,
            repetitions=self._qick_repetitions_per_sweep,
        )
        self._experiment_panel.set_bias_t_values(
            enabled=self._bias_t_compensation_enabled,
            compensation_type=self._bias_t_compensation_type,
            compensation_mv=self._bias_t_compensation_voltage_mv,
            mode=self._bias_t_compensation_mode,
            duration_us=self._bias_t_compensation_duration_us,
            filter_tau_us=self._bias_t_filter_tau_us,
        )
        self._cross_capacitance = np.asarray(
            settings["cross_capacitance"], dtype=float
        )
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
                "bias_t_compensation_enabled",
                "bias_t_compensation_type",
                "bias_t_compensation_voltage_mv",
                "bias_t_compensation_mode",
                "bias_t_compensation_duration_us",
                "bias_t_filter_tau_us",
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
        new_pulse.v[:]  = DEFAULT_INITIAL_VOLTAGE_MV
        self._plot.add_pulse(new_pulse)
        color = (
            self._plot.line_color(len(self._plot._line) - 1)
            if hasattr(self._plot, "line_color")
            else self._plot._line[-1].get_color()
        )
        self._multi_ctrl._color_map.append(color)
        self._pulse.append(new_pulse)
        self._qick_awg_channels = (
            *self._qick_awg_channels,
            self._next_qick_awg_channel(),
        )
        if hasattr(self, "_experiment_panel"):
            self._experiment_panel.set_qick_values(
                fabric_mhz=self._qick_fabric_mhz,
                tproc_mhz=self._qick_tproc_mhz,
                full_scale_mv=self._qick_full_scale_mv,
                awg_channels=self._qick_awg_channels,
                repetitions=self._qick_repetitions_per_sweep,
            )
        old_count = self._cross_capacitance.shape[0]
        expanded_coupling = np.eye(old_count + 1, dtype=float)
        expanded_coupling[:old_count, :old_count] = self._cross_capacitance
        self._cross_capacitance = expanded_coupling
        new_ctrl = ControlPanel(new_pulse, time_unit=self._time_unit)
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
        self._refresh_waveform_plot(idx)

    def _refresh_waveform_plot(self, index: Optional[int] = None) -> None:
        """Refresh either plot backend without mixing their call signatures."""
        if _USE_PYQTGRAPH:
            self._plot.refresh(index)
        else:
            self._plot.refresh()

    def _plot_refresh(self):
        self._refresh_waveform_plot(self._selected_port_idx)
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

        self._refresh_stability_targets()

    def _refresh_stability_targets(self) -> None:
        if not hasattr(self, "_stability_panel"):
            return
        try:
            segment_names = qick_set_segment_names(self._pulse[0])
        except ValueError:
            segment_names = ()
        self._stability_panel.refresh_targets(
            self._qick_output_names(),
            self._qick_awg_channels,
            segment_names,
        )
        if hasattr(self, "_stability_rf_ports_panel"):
            self._stability_rf_ports_panel.refresh_segments(self._pulse[0])
            self._stability_rf_readout_panel.refresh_segments(self._pulse[0])

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

    def closeEvent(self, event) -> None:
        if self._experiment_thread is not None and self._experiment_thread.isRunning():
            QtWidgets.QMessageBox.warning(
                self,
                "QICK task running",
                "The QICK task is still running. Wait for it to finish before closing.",
            )
            event.ignore()
            return
        super().closeEvent(event)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1500, 650)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
