"""Tests for AWG X/Y trace timing labels and Stability underlay.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import replace
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtWidgets

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


def test_trace_points_show_hold_and_ramps_above_their_segments():
    app = _application()
    widget = _trace_widget()

    assert len(widget._point_records) == 3
    assert len(widget._point_labels) == 3
    assert len(widget._ramp_labels) == 2
    assert len(widget._point_scatter.points()) == 3
    point_one = widget._point_records[1]
    assert point_one["point_index"] == 1
    assert point_one["ramp_x_ns"] == 500.0
    assert point_one["ramp_y_ns"] == 750.0
    assert point_one["hold_x_ns"] == 2_000.0
    assert point_one["hold_y_ns"] == 2_500.0
    label = widget._point_labels[1].textItem.toPlainText()
    assert "set_1" in label
    assert "Hold X 2 / Y 2.5 us" in label
    assert "Ramp" not in label
    ramp_label = widget._ramp_labels[0]
    assert (
        ramp_label.textItem.toPlainText()
        == "Ramp X 0.5 / Y 0.75 us"
    )
    np.testing.assert_allclose(
        [ramp_label.pos().x(), ramp_label.pos().y()],
        [50.0, -20.0],
    )
    tooltip = widget._point_tooltip(0.0, 0.0, point_one)
    assert "set_1" in tooltip
    assert "X 100 mV" in tooltip
    assert "Y -50 mV" in tooltip
    assert "Hold X 2 / Y 2.5 us" in tooltip
    assert "Ramp" not in tooltip

    hovered_point = widget._point_scatter.points()[1]
    widget._points_hovered(widget._point_scatter, [hovered_point], None)
    hover_text = widget._hover_label.textItem.toPlainText()
    assert widget._hover_label.isVisible()
    assert "X 100 mV" in hover_text
    assert "Y -50 mV" in hover_text
    widget._points_hovered(
        widget._point_scatter,
        np.asarray([], dtype=object),
        None,
    )
    assert not widget._hover_label.isVisible()

    widget.set_time_unit("ns")
    tooltip_ns = widget._point_tooltip(
        0.0,
        0.0,
        widget._point_records[1],
    )
    assert "Hold X 2000 / Y 2500 ns" in tooltip_ns
    assert (
        widget._ramp_labels[0].textItem.toPlainText()
        == "Ramp X 500 / Y 750 ns"
    )
    app.processEvents()
    widget.close()


def test_trace_points_use_custom_segment_names():
    app = _application()
    pulse_x = _pulse(
        [0.0, 100.0],
        [1_000.0, 2_000.0],
        [500.0],
    )
    pulse_y = _pulse(
        [10.0, -50.0],
        [1_000.0, 2_000.0],
        [500.0],
    )
    for pulse in (pulse_x, pulse_y):
        pulse.rename_segment(0, "Reset")
        pulse.rename_segment(1, "Measure")
    widget = TracePlotWidget()
    widget.x_idx = 0
    widget.y_idx = 1
    widget.refresh_trace((pulse_x, pulse_y))

    assert "Reset" in widget._point_labels[0].textItem.toPlainText()
    assert "Measure" in widget._point_labels[1].textItem.toPlainText()
    assert "P0" not in widget._point_labels[0].textItem.toPlainText()
    assert "Measure" in widget._point_tooltip(
        100.0,
        -50.0,
        widget._point_records[1],
    )

    pulse_y.rename_segment(1, "Sense")
    widget.refresh_trace((pulse_x, pulse_y))
    assert (
        "X Measure / Y Sense"
        in widget._point_labels[1].textItem.toPlainText()
    )
    app.processEvents()
    widget.close()


def test_trace_hold_and_ramp_labels_emit_edit_requests():
    _application()
    widget = _trace_widget()
    hold_requests = []
    ramp_requests = []
    widget.hold_edit_requested.connect(hold_requests.append)
    widget.ramp_edit_requested.connect(ramp_requests.append)

    widget._points_clicked(
        widget._point_scatter,
        np.asarray([], dtype=object),
        None,
    )
    assert hold_requests == []

    class _ClickEvent:
        accepted = False

        @staticmethod
        def button():
            return QtCore.Qt.LeftButton

        def accept(self):
            self.accepted = True

        @staticmethod
        def ignore():
            return

    hold_event = _ClickEvent()
    widget._point_labels[1].mouseClickEvent(hold_event)
    ramp_event = _ClickEvent()
    widget._ramp_labels[0].mouseClickEvent(ramp_event)

    assert hold_event.accepted is True
    assert ramp_event.accepted is True
    assert hold_requests == [1]
    assert ramp_requests == [1]
    assert "Click to edit X, Y" in widget._point_labels[1].toolTip()
    assert "Click to edit ramp" in widget._ramp_labels[0].toolTip()
    widget.close()


def test_trace_edits_update_xy_values_and_share_segment_timing():
    _application()
    window = gui.MainWindow()
    window._add_segment(500.0, 2_000.0, 100.0)
    window._add_port()
    window._add_port()
    third_output_voltage = window._pulse[2].v.copy()
    window._set_x(0)
    window._set_y(1)

    window._apply_trace_hold_edit(
        segment_index=1,
        x_mv=225.0,
        y_mv=-175.0,
        hold_ns=3_250.0,
        segment_name="Readout",
    )

    assert window._pulse[0].v[2] == 225.0
    assert window._pulse[1].v[2] == -175.0
    np.testing.assert_array_equal(window._pulse[2].v, third_output_voltage)
    for pulse in window._pulse:
        assert pulse.t[3] - pulse.t[2] == 3_250.0
        assert pulse.segment_name(1) == "Readout"
    assert "Readout" in window._trace._point_labels[1].textItem.toPlainText()

    window._apply_trace_ramp_edit(
        segment_index=1,
        ramp_ns=875.0,
    )
    for pulse in window._pulse:
        assert pulse.t[2] - pulse.t[1] == 875.0
    window.close()


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


def test_trace_fit_uses_exact_stability_overlay_cell_edges():
    app = _application()
    widget = _trace_widget()
    widget.set_stability_overlay(_result())

    widget.fit_view()
    app.processEvents()

    view_range = widget.getPlotItem().vb.viewRange()
    np.testing.assert_allclose(view_range[0], [-150.0, 150.0])
    np.testing.assert_allclose(view_range[1], [-100.0, 100.0])
    widget.close()


def test_stability_overlay_can_select_i_q_magnitude_or_phase():
    app = _application()
    widget = _trace_widget()
    result = _result()
    q_values = result.magnitude * -2.0
    phase_values = np.asarray(
        [[-180.0, -90.0, 0.0], [45.0, 90.0, 180.0]],
        dtype=float,
    )
    result = replace(
        result,
        q_mean=q_values,
        phase_deg=phase_values,
        source_label="QCoDeS Run 42",
        database_path="stability.db",
        run_id=42,
    )

    widget.set_stability_overlay(result, "q")
    np.testing.assert_allclose(widget._stability_image.image, q_values)
    assert "QCoDeS Run 42 Q" in widget._default_title
    assert "[nA]" in widget._default_title

    widget.set_stability_overlay(result, "phase")
    np.testing.assert_allclose(widget._stability_image.image, phase_values)
    assert "Phase [deg]" in widget._default_title
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
            self.quantity = None
            self.fit_count = 0

        def set_stability_overlay(self, value, quantity="magnitude"):
            self.result = value
            self.quantity = quantity

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
    assert window._trace.quantity == "magnitude"
    assert window._trace.fit_count == 1


def test_pinned_saved_overlay_is_not_replaced_by_new_scan():
    class _Panel:
        def __init__(self):
            self.result = None

        def show_result(self, value):
            self.result = value

    class _Trace:
        def __init__(self, result):
            self.result = result
            self.fit_count = 0

        def set_stability_overlay(self, value, quantity="magnitude"):
            self.result = value

        def fit_view(self):
            self.fit_count += 1

    class _Window:
        def __init__(self, pinned):
            self._last_stability_result = None
            self._trace_overlay_result = pinned
            self._trace_overlay_pinned = True
            self._trace_overlay_quantity = "magnitude"
            self._stability_panel = _Panel()
            self._trace = _Trace(pinned)

        def statusBar(self):
            return type("_Status", (), {"showMessage": lambda *_args: None})()

    pinned = replace(_result(), source_label="QCoDeS Run 8", run_id=8)
    latest = replace(_result(), iteration=9)
    window = _Window(pinned)

    gui.MainWindow._on_stability_scan_ready(window, latest)

    assert window._last_stability_result is latest
    assert window._stability_panel.result is latest
    assert window._trace.result is pinned
    assert window._trace.fit_count == 0


def test_saved_stability_run_refreshes_trace_overlay_selector(tmp_path):
    class _Panel:
        def __init__(self):
            self.stored = None

        def show_saved_result(self, stored):
            self.stored = stored

    class _DatabasePath:
        def __init__(self):
            self.value = ""

        def setText(self, value):
            self.value = value

    class _Selector:
        def __init__(self):
            self.database_path = _DatabasePath()
            self.refresh_count = 0
            self.latest = None

        def refresh_runs(self):
            self.refresh_count += 1

        def show_latest_result(self, result):
            self.latest = result

    class _Stored:
        def __init__(self, diagram, database_path):
            self.diagram = diagram
            self.database_path = database_path
            self.run_id = 57

    class _Window:
        def __init__(self):
            self._last_stability_result = None
            self._trace_overlay_pinned = False
            self._stability_panel = _Panel()
            self._trace_overlay_selector = _Selector()
            self._trace = None

        def statusBar(self):
            return type("_Status", (), {"showMessage": lambda *_args: None})()

    database_path = tmp_path / "new_stability.db"
    stored = _Stored(_result(), database_path)
    window = _Window()

    gui.MainWindow._on_stability_single_finished(window, stored)

    assert window._stability_panel.stored is stored
    assert window._trace_overlay_selector.database_path.value == str(database_path)
    assert window._trace_overlay_selector.refresh_count == 1
    assert window._trace_overlay_selector.latest is stored.diagram
