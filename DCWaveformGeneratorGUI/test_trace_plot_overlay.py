"""Tests for AWG X/Y trace timing labels and Stability underlay.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import replace
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtWidgets

import DCWaveform_Generator as gui
from dc_waveform_core import PulseSequence
from dc_waveform_widgets import TracePlotWidget
from stability_diagram import StabilityDiagramResult


_APP = None


def _application():
    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _APP


def _pulse(levels_mv, holds_ns, ramps_ns) -> PulseSequence:
    pulse = PulseSequence(
        initial_voltage=levels_mv[0],
        initial_duration_ns=holds_ns[0],
    )
    for level_mv, hold_ns, ramp_ns in zip(
        levels_mv[1:],
        holds_ns[1:],
        ramps_ns,
    ):
        pulse.add_flat_ramp(ramp_ns, hold_ns, level_mv)
    return pulse


def _result(
    *,
    x_axis_label: str = "awg_0",
    y_axis_label: str = "awg_1",
) -> StabilityDiagramResult:
    magnitude = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=float,
    )
    return StabilityDiagramResult(
        x_voltage_mv=np.asarray([-100.0, 0.0, 100.0]),
        y_voltage_mv=np.asarray([-50.0, 50.0]),
        i_mean=magnitude.copy(),
        q_mean=np.zeros_like(magnitude),
        magnitude=magnitude,
        phase_deg=np.zeros_like(magnitude),
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        value_unit="nA",
        base_value_unit="A",
        display_scale=1.0e9,
        measurement_mode="current",
        iteration=7,
        repetition_count=2,
        samples_per_trace=64,
        sample_rate_hz=50_000.0,
        fir_rate_profile="50_ksps",
    )


def _trace_widget() -> TracePlotWidget:
    pulse_x = _pulse(
        [0.0, 100.0, -100.0],
        [1_000.0, 2_000.0, 3_000.0],
        [500.0, 1_500.0],
    )
    pulse_y = _pulse(
        [10.0, -50.0, 75.0],
        [1_000.0, 2_500.0, 3_000.0],
        [750.0, 1_500.0],
    )
    widget = TracePlotWidget()
    widget.x_idx = 0
    widget.y_idx = 1
    widget.refresh_trace((pulse_x, pulse_y))
    return widget


def test_trace_points_include_hold_and_incoming_ramp_timing():
    app = _application()
    widget = _trace_widget()

    assert len(widget._point_records) == 3
    assert len(widget._point_labels) == 3
    assert len(widget._point_scatter.points()) == 3
    point_one = widget._point_records[1]
    assert point_one["point_index"] == 1
    assert point_one["ramp_x_ns"] == 500.0
    assert point_one["ramp_y_ns"] == 750.0
    assert point_one["hold_x_ns"] == 2_000.0
    assert point_one["hold_y_ns"] == 2_500.0
    label = widget._point_labels[1].textItem.toPlainText()
    assert "P1" in label
    assert "Ramp X 0.5 / Y 0.75 us" in label
    assert "Hold X 2 / Y 2.5 us" in label
    tooltip = widget._point_tooltip(0.0, 0.0, point_one)
    assert "P1" in tooltip
    assert "Ramp X 0.5 / Y 0.75 us" in tooltip
    assert "Hold X 2 / Y 2.5 us" in tooltip

    widget.set_time_unit("ns")
    tooltip_ns = widget._point_tooltip(
        0.0,
        0.0,
        widget._point_records[1],
    )
    assert "Ramp X 500 / Y 750 ns" in tooltip_ns
    assert "Hold X 2000 / Y 2500 ns" in tooltip_ns
    app.processEvents()
    widget.close()


def test_last_stability_magnitude_is_drawn_below_matching_trace():
    app = _application()
    widget = _trace_widget()
    result = _result()

    widget.set_stability_overlay(result)

    assert widget._stability_overlay_active is True
    assert widget._stability_image.isVisible() is True
    assert widget._stability_color_bar is None
    np.testing.assert_allclose(widget._stability_image.image, result.magnitude)
    assert "Stability scan 7" in widget._default_title
    app.processEvents()
    widget.close()


def test_stability_overlay_transposes_reversed_axes_and_hides_mismatch():
    app = _application()
    widget = _trace_widget()
    widget.x_idx = 1
    widget.y_idx = 0
    widget.refresh_trace(widget._pulses)
    result = _result()

    widget.set_stability_overlay(result)

    assert widget._stability_overlay_active is True
    np.testing.assert_allclose(
        widget._stability_image.image,
        result.magnitude.T,
    )

    widget.set_stability_overlay(
        replace(
            result,
            x_axis_label="awg_2",
            y_axis_label="awg_3",
        )
    )
    assert widget._stability_overlay_active is False
    assert widget._stability_image.isVisible() is False
    assert "do not match" in widget._default_title
    app.processEvents()
    widget.close()


def test_main_window_retains_and_forwards_latest_stability_result():
    class _Panel:
        def __init__(self):
            self.result = None

        def show_result(self, value):
            self.result = value

    class _StatusBar:
        def showMessage(self, _message):
            return

    class _Trace:
        def __init__(self):
            self.result = None
            self.fit_count = 0

        def set_stability_overlay(self, value):
            self.result = value

        def fit_view(self):
            self.fit_count += 1

    class _Window:
        def __init__(self):
            self._last_stability_result = None
            self._stability_panel = _Panel()
            self._trace = None
            self._status_bar = _StatusBar()

        def statusBar(self):
            return self._status_bar

    window = _Window()
    result = _result()

    gui.MainWindow._on_stability_scan_ready(window, result)
    assert window._trace is None
    assert window._last_stability_result is result
    assert window._stability_panel.result is result

    window._trace = _Trace()
    gui.MainWindow._on_stability_scan_ready(window, result)
    assert window._trace.result is result
    assert window._trace.fit_count == 1
