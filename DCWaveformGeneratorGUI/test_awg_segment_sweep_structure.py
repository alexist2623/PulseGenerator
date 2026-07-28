"""Regression tests for AWG segment edits with configured sweep axes.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtWidgets

import DCWaveform_Generator as gui
from dc_waveform_core import QickRampRateSweepSpec, QickSweepSpec


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _sweep_targets(window):
    return tuple(
        (spec.output_name, spec.segment_name)
        for spec in window._sweep_specs
    )


def _add_segments(window, count):
    for index in range(count):
        window._add_segment(
            100.0 + index,
            200.0 + index,
            10.0 * (index + 1),
        )


def _sweep_parameter_row(panel, axis_kind):
    for row in range(panel.sweep_parameter_table.rowCount()):
        key = panel.sweep_parameter_table.item(row, 0).data(
            gui.QtCore.Qt.UserRole
        )
        if key[0] == axis_kind:
            return row
    raise AssertionError(f"missing {axis_kind!r} sweep parameter row")


def _segment_timing_ns(pulse, row):
    start, end = pulse.flat_segments()[row]
    ramp = pulse.t[start] - pulse.t[start - 1] if start else 0.0
    flat = pulse.t[end] - pulse.t[start]
    return float(ramp), float(flat)


def test_matching_awg_segments_share_ramp_and_hold_table_edits():
    app = _application()
    window = gui.MainWindow()
    _add_segments(window, 2)
    window._add_port()
    control_0, control_1 = window._multi_ctrl._ctrl_pannels
    voltage_0 = window._pulse[0].v.copy()
    window._pulse[1].v[:] = (
        -300.0,
        -300.0,
        450.0,
        450.0,
        -125.0,
        -125.0,
    )
    voltage_1 = window._pulse[1].v.copy()
    control_1.refresh_table()

    control_0.table.item(1, 1).setText("0.75")
    app.processEvents()
    assert _segment_timing_ns(window._pulse[0], 1)[0] == 750.0
    assert _segment_timing_ns(window._pulse[1], 1)[0] == 750.0

    control_1.table.item(2, 2).setText("1.25")
    app.processEvents()
    assert _segment_timing_ns(window._pulse[0], 2)[1] == 1250.0
    assert _segment_timing_ns(window._pulse[1], 2)[1] == 1250.0
    assert np.array_equal(window._pulse[0].v, voltage_0)
    assert np.array_equal(window._pulse[1].v, voltage_1)
    window.close()


def test_waveform_time_drag_shares_matching_segment_timing():
    _application()
    window = gui.MainWindow()
    _add_segments(window, 2)
    window._add_port()
    window._port_select(1)

    pulse = window._pulse[1]
    pulse.update_point((2, 3), float(pulse.t[2]) + 375.0)
    window._point_update(2, 3, float(pulse.t[2]))
    window._flush_deferred_refresh()

    assert _segment_timing_ns(window._pulse[0], 1) == (
        _segment_timing_ns(window._pulse[1], 1)
    )
    window.close()


def test_awg_sweep_parameter_table_edits_all_supported_sweep_types():
    app = _application()
    window = gui.MainWindow()
    _add_segments(window, 2)
    window._sweep_specs = [
        QickSweepSpec("set_1", "awg_0", -0.25, 0.5, 5),
        QickRampRateSweepSpec("ramp_0_to_1", 0.1, 0.3, 4),
    ]
    window._experiment_panel.full_scale_mv.setValue(800.0)

    rf_panel = window._rf_ports_panel._panels[0]
    with gui.QtCore.QSignalBlocker(rf_panel):
        rf_panel.setChecked(True)
        rf_panel.segment.setCurrentIndex(
            rf_panel.segment.findData("set_2")
        )
        rf_panel.duration_sweep_enabled.setChecked(True)
        rf_panel.duration_sweep_start.setValue(
            gui._time_from_ns(1500.0, rf_panel._time_unit)
        )
        rf_panel.duration_sweep_stop.setValue(
            gui._time_from_ns(3500.0, rf_panel._time_unit)
        )
        rf_panel.duration_sweep_count.setValue(8)
    window._rf_ports_panel._emit_specs()
    window._refresh_sweep_overlay(sync_rows=True)
    app.processEvents()

    experiment = window._experiment_panel
    assert experiment.sweep_parameter_table.rowCount() == 3
    assert {
        experiment.sweep_parameter_table.item(row, 0).text()
        for row in range(3)
    } == {"Voltage", "RAMP duration", "RF duration"}

    voltage_row = _sweep_parameter_row(experiment, "amplitude")
    experiment.sweep_parameter_table.selectRow(voltage_row)
    app.processEvents()
    assert experiment.sweep_parameter_target.text() == "awg_0 / set_1"
    assert experiment.sweep_parameter_start.suffix() == " mV"
    assert experiment.sweep_parameter_start.value() == -200.0
    assert experiment.sweep_parameter_stop.value() == 400.0
    experiment.sweep_parameter_start.setValue(-100.0)
    experiment.sweep_parameter_stop.setValue(240.0)
    experiment.sweep_parameter_count.setValue(7)
    experiment.sweep_parameter_apply.click()
    app.processEvents()

    voltage_spec = next(
        spec
        for spec in window._sweep_specs
        if isinstance(spec, QickSweepSpec)
    )
    assert voltage_spec.start == -0.125
    assert voltage_spec.stop == 0.3
    assert voltage_spec.count == 7

    ramp_row = _sweep_parameter_row(experiment, "ramp_duration")
    experiment.sweep_parameter_table.selectRow(ramp_row)
    app.processEvents()
    assert experiment.sweep_parameter_start.suffix() == " us"
    experiment.sweep_parameter_start.setValue(0.2)
    experiment.sweep_parameter_stop.setValue(0.8)
    experiment.sweep_parameter_count.setValue(6)
    experiment.sweep_parameter_apply.click()
    app.processEvents()

    ramp_spec = next(
        spec
        for spec in window._sweep_specs
        if isinstance(spec, QickRampRateSweepSpec)
    )
    assert ramp_spec.start == 0.2
    assert ramp_spec.stop == 0.8
    assert ramp_spec.count == 6

    rf_row = _sweep_parameter_row(experiment, "rf_duration")
    experiment.sweep_parameter_table.selectRow(rf_row)
    app.processEvents()
    assert experiment.sweep_parameter_target.text() == "RF gen 0 / set_2"
    experiment.sweep_parameter_start.setValue(2.0)
    experiment.sweep_parameter_stop.setValue(4.0)
    experiment.sweep_parameter_count.setValue(9)
    experiment.sweep_parameter_apply.click()
    app.processEvents()

    rf_spec = window._rf_pulse_specs[0]
    assert rf_spec.duration_sweep_start_us == 2.0
    assert rf_spec.duration_sweep_stop_us == 4.0
    assert rf_spec.duration_sweep_count == 9
    assert rf_panel.duration_sweep_enabled.isChecked()

    rf_row = _sweep_parameter_row(experiment, "rf_duration")
    experiment.sweep_parameter_table.selectRow(rf_row)
    experiment.sweep_parameter_remove.click()
    app.processEvents()
    assert rf_panel.isChecked()
    assert not rf_panel.duration_sweep_enabled.isChecked()
    assert all(
        getattr(spec, "axis_kind", "") != "rf_duration"
        for spec in window._active_map_sweep_specs()
    )
    assert experiment.sweep_parameter_table.rowCount() == 2

    ramp_row = _sweep_parameter_row(experiment, "ramp_duration")
    experiment.sweep_parameter_table.selectRow(ramp_row)
    experiment.sweep_parameter_remove.click()
    app.processEvents()
    assert not any(
        isinstance(spec, QickRampRateSweepSpec)
        for spec in window._sweep_specs
    )

    voltage_row = _sweep_parameter_row(experiment, "amplitude")
    experiment.sweep_parameter_table.selectRow(voltage_row)
    experiment.sweep_parameter_remove.click()
    app.processEvents()
    assert window._sweep_specs == []
    assert experiment.sweep_parameter_table.rowCount() == 0
    assert experiment.sweep_parameter_table.isHidden()
    assert not experiment.sweep_parameter_editor.isEnabled()
    window.close()


def test_ramp_rate_sweep_is_not_drawn_on_waveform_plot():
    app = _application()
    window = gui.MainWindow()
    _add_segments(window, 1)
    window._sweep_specs = [
        QickRampRateSweepSpec("ramp_0_to_1", 0.1, 0.3, 4),
    ]

    window._refresh_sweep_overlay(sync_rows=True)
    app.processEvents()

    assert window._sweep_specs == [
        QickRampRateSweepSpec("ramp_0_to_1", 0.1, 0.3, 4),
    ]
    assert window._multi_ctrl._ctrl_pannels[0]._ramp_sweep_rows == {1}
    assert window._plot._sweep_time_ns.size == 0
    assert all(
        graphics["lower_curve"].getData()[0].size == 0
        for graphics in window._plot._sweep_graphics.values()
    )

    window._sweep_specs.append(
        QickSweepSpec("set_1", "awg_0", -0.25, 0.5, 5)
    )
    window._refresh_sweep_overlay(sync_rows=True)
    app.processEvents()

    assert len(window._plot._sweep_graphics) == 1
    assert window._plot._sweep_time_ns.size > 0
    window.close()


def test_appending_voltage_segments_preserves_existing_sweeps():
    app = _application()
    window = gui.MainWindow()
    window._sweep_specs = [
        QickSweepSpec("set_0", "awg_0", -0.2, 0.2, 3),
    ]
    window._refresh_sweep_overlay(sync_rows=True)

    _add_segments(window, 4)
    app.processEvents()

    assert window._pulse[0].set_count == 5
    assert _sweep_targets(window) == (("awg_0", "set_0"),)
    assert window._multi_ctrl._ctrl_pannels[0]._sweep_rows == {0}
    window.close()


def test_insert_delete_sequence_remaps_only_affected_voltage_sweeps():
    app = _application()
    window = gui.MainWindow()
    _add_segments(window, 3)
    window._add_port()
    window._sweep_specs = [
        QickSweepSpec("set_0", "awg_0", -0.1, 0.1, 2),
        QickSweepSpec("set_1", "awg_0", -0.2, 0.2, 3),
        QickSweepSpec("set_3", "awg_0", -0.3, 0.3, 4),
        QickSweepSpec("set_2", "awg_1", -0.4, 0.4, 5),
        QickRampRateSweepSpec("ramp_0_to_1", 0.1, 0.2, 2),
        QickRampRateSweepSpec("ramp_2_to_3", 0.2, 0.4, 3),
    ]
    window._refresh_sweep_overlay(sync_rows=True)
    panel = window._experiment_panel
    panel.set_sweep_specs(
        window._active_map_sweep_specs(),
        selected_keys=(("awg_0", "set_3"), ("awg_1", "set_2")),
    )

    control_0 = window._multi_ctrl._ctrl_pannels[0]
    assert control_0._edit_segment_structure("insert_above", 2)
    app.processEvents()

    assert _sweep_targets(window) == (
        ("awg_0", "set_0"),
        ("awg_0", "set_1"),
        ("awg_0", "set_4"),
        ("awg_1", "set_2"),
        ("all_awg_outputs", "ramp_0_to_1"),
    )
    assert panel.selected_sweep_axis_keys() == (
        ("awg_0", "set_4"),
        ("awg_1", "set_2"),
    )
    assert control_0._sweep_rows == {0, 1, 4}
    assert window._multi_ctrl._ctrl_pannels[1]._sweep_rows == {2}

    assert control_0._edit_segment_structure("delete", 1)
    app.processEvents()
    assert _sweep_targets(window) == (
        ("awg_0", "set_0"),
        ("awg_0", "set_3"),
        ("awg_1", "set_2"),
    )
    assert panel.selected_sweep_axis_keys() == (
        ("awg_0", "set_3"),
        ("awg_1", "set_2"),
    )
    assert control_0._sweep_rows == {0, 3}
    assert window._multi_ctrl._ctrl_pannels[1]._sweep_rows == {2}

    control_1 = window._multi_ctrl._ctrl_pannels[1]
    assert control_1._edit_segment_structure("insert_below", 0)
    app.processEvents()
    assert _sweep_targets(window) == (
        ("awg_0", "set_0"),
        ("awg_0", "set_3"),
        ("awg_1", "set_3"),
    )
    assert panel.selected_sweep_axis_keys() == (
        ("awg_0", "set_3"),
        ("awg_1", "set_3"),
    )

    assert control_1._edit_segment_structure("delete", 3)
    app.processEvents()
    assert _sweep_targets(window) == (
        ("awg_0", "set_0"),
        ("awg_0", "set_3"),
    )
    assert control_0._sweep_rows == {0, 3}
    assert control_1._sweep_rows == set()
    window.close()


def test_deleting_one_target_keeps_other_sweeps_on_same_port():
    app = _application()
    window = gui.MainWindow()
    _add_segments(window, 4)
    window._sweep_specs = [
        QickSweepSpec(f"set_{index}", "awg_0", -0.1, 0.1, index + 2)
        for index in range(5)
    ]
    window._refresh_sweep_overlay(sync_rows=True)

    control = window._multi_ctrl._ctrl_pannels[0]
    assert control._edit_segment_structure("delete", 2)
    app.processEvents()

    assert _sweep_targets(window) == (
        ("awg_0", "set_0"),
        ("awg_0", "set_1"),
        ("awg_0", "set_2"),
        ("awg_0", "set_3"),
    )
    assert tuple(spec.count for spec in window._sweep_specs) == (2, 3, 5, 6)
    assert control._sweep_rows == {0, 1, 2, 3}
    window.close()


def test_deleting_preceding_segment_keeps_sweep_visible_and_in_experiment():
    app = _application()
    window = gui.MainWindow()
    _add_segments(window, 3)
    window._sweep_specs = [
        QickSweepSpec("set_3", "awg_0", -0.2, 0.2, 5),
    ]
    window._refresh_sweep_overlay(sync_rows=True)

    control = window._multi_ctrl._ctrl_pannels[0]
    control.table.selectRow(2)
    assert control.table.currentRow() == 2
    assert control._edit_segment_structure("delete", 2)
    app.processEvents()

    assert _sweep_targets(window) == (("awg_0", "set_2"),)
    assert control._sweep_rows == {2}
    assert control.table.selectedItems() == []
    marker_item = control.table.item(2, 0)
    assert not marker_item.icon().isNull()
    assert marker_item.toolTip() == "Voltage sweep target"

    experiment = window._experiment_panel
    assert experiment.sweep_parameter_table.rowCount() == 1
    experiment.sweep_parameter_table.selectRow(0)
    app.processEvents()
    assert experiment.sweep_parameter_target.text() == "awg_0 / set_2"
    window.close()


def test_insert_keeps_voltage_sweep_on_original_physical_segment():
    app = _application()
    window = gui.MainWindow()
    _add_segments(window, 4)
    window._sweep_specs = [
        QickSweepSpec("set_3", "awg_0", -0.2, 0.2, 5),
    ]
    window._refresh_sweep_overlay(sync_rows=True)
    original_voltage = float(window._pulse[0].v[6])

    control = window._multi_ctrl._ctrl_pannels[0]
    assert control._edit_segment_structure("insert_above", 2)
    app.processEvents()

    assert _sweep_targets(window) == (("awg_0", "set_4"),)
    assert window._sweep_target_indices(window._sweep_specs[0]) == (0, 4)
    assert float(window._pulse[0].v[8]) == original_voltage
    assert control._sweep_rows == {4}
    window.close()


def test_insert_remaps_rf_duration_sweep_anchor_and_selected_map_axis():
    app = _application()
    window = gui.MainWindow()
    _add_segments(window, 3)
    window._sweep_specs = [
        QickSweepSpec("set_3", "awg_0", -0.2, 0.2, 5),
    ]

    rf_panel = window._rf_ports_panel._panels[0]
    with gui.QtCore.QSignalBlocker(rf_panel):
        rf_panel.setChecked(True)
        rf_panel.segment.setCurrentIndex(
            rf_panel.segment.findData("set_2")
        )
        rf_panel.duration_sweep_enabled.setChecked(True)
        rf_panel.duration_sweep_start.setValue(1.0)
        rf_panel.duration_sweep_stop.setValue(2.0)
        rf_panel.duration_sweep_count.setValue(7)
    window._rf_ports_panel._emit_specs()
    window._experiment_panel.set_sweep_specs(
        window._active_map_sweep_specs(),
        selected_keys=(
            ("awg_0", "set_3"),
            ("rf_gen_0", "set_2"),
        ),
    )

    control = window._multi_ctrl._ctrl_pannels[0]
    assert control._edit_segment_structure("insert_above", 1)
    app.processEvents()

    assert len(window._rf_pulse_specs) == 1
    rf_spec = window._rf_pulse_specs[0]
    assert rf_spec.segment_name == "set_3"
    assert rf_spec.duration_sweep_enabled is True
    assert rf_spec.duration_sweep_start_us == 1.0
    assert rf_spec.duration_sweep_stop_us == 2.0
    assert rf_spec.duration_sweep_count == 7
    assert rf_panel.segment.currentData() == "set_3"
    assert window._experiment_panel.selected_sweep_axis_keys() == (
        ("awg_0", "set_4"),
        ("rf_gen_0", "set_3"),
    )
    window.close()


def test_experiment_sweep_edit_atomically_refreshes_waveform_and_markers():
    app = _application()
    window = gui.MainWindow()
    _add_segments(window, 2)
    window._sweep_specs = [
        QickSweepSpec("set_2", "awg_0", -0.25, 0.5, 5),
    ]
    window._refresh_sweep_overlay(sync_rows=True)
    sweep_events = []
    window.sweep_state_changed.connect(sweep_events.append)

    experiment = window._experiment_panel
    experiment.sweep_parameter_table.selectRow(0)
    app.processEvents()
    experiment.sweep_parameter_start.setValue(-160.0)
    experiment.sweep_parameter_stop.setValue(320.0)
    experiment.sweep_parameter_count.setValue(9)
    experiment.sweep_parameter_apply.click()
    app.processEvents()

    full_scale_mv = experiment.full_scale_mv.value()
    assert len(sweep_events) == 1
    assert window._sweep_specs == [
        QickSweepSpec(
            "set_2",
            "awg_0",
            -160.0 / full_scale_mv,
            320.0 / full_scale_mv,
            9,
        ),
    ]
    graphics = window._plot._sweep_graphics[("awg_0", "set_2")]
    assert np.allclose(graphics["lower_mv"][4:6], -160.0)
    assert np.allclose(graphics["upper_mv"][4:6], 320.0)
    assert graphics["fill"].isVisible()
    control = window._multi_ctrl._ctrl_pannels[0]
    assert control._sweep_rows == {2}
    assert experiment.sweep_parameter_target.text() == "awg_0 / set_2"
    window.close()
    window.deleteLater()
    app.processEvents()


def test_repeated_insert_delete_hides_stale_envelopes_and_relinks_target():
    app = _application()
    window = gui.MainWindow()
    _add_segments(window, 3)
    window._sweep_specs = [
        QickSweepSpec("set_2", "awg_0", -0.2, 0.4, 7),
    ]
    window._refresh_sweep_overlay(sync_rows=True)
    control = window._multi_ctrl._ctrl_pannels[0]
    original = window._plot._sweep_graphics[("awg_0", "set_2")]

    assert control._edit_segment_structure("insert_above", 2)
    app.processEvents()

    assert _sweep_targets(window) == (("awg_0", "set_3"),)
    inserted = window._plot._sweep_graphics[("awg_0", "set_3")]
    assert inserted["fill"].isVisible()
    assert not original["fill"].isVisible()
    original_x = original["lower_curve"].getData()[0]
    assert original_x is None or original_x.size == 0
    assert control._sweep_rows == {3}

    assert control._edit_segment_structure("delete", 2)
    app.processEvents()

    assert _sweep_targets(window) == (("awg_0", "set_2"),)
    assert original["fill"].isVisible()
    assert not inserted["fill"].isVisible()
    inserted_x = inserted["lower_curve"].getData()[0]
    assert inserted_x is None or inserted_x.size == 0
    assert control._sweep_rows == {2}
    assert (
        window._experiment_panel.sweep_parameter_target.text()
        == "awg_0 / set_2"
    )
    window.close()
    window.deleteLater()
    app.processEvents()


def test_sweep_state_event_prunes_orphaned_segment_references():
    app = _application()
    window = gui.MainWindow()
    _add_segments(window, 1)
    window._sweep_specs = [
        QickSweepSpec("set_0", "awg_0", -0.1, 0.1, 3),
        QickSweepSpec("set_99", "awg_0", -0.2, 0.2, 5),
        QickRampRateSweepSpec("ramp_8_to_9", 0.1, 0.3, 4),
    ]

    window._notify_sweep_state_changed()
    app.processEvents()

    assert window._sweep_specs == [
        QickSweepSpec("set_0", "awg_0", -0.1, 0.1, 3),
    ]
    assert window._multi_ctrl._ctrl_pannels[0]._sweep_rows == {0}
    assert window._experiment_panel.sweep_parameter_table.rowCount() == 1
    assert set(window._plot._sweep_graphics) == {("awg_0", "set_0")}
    window.close()
    window.deleteLater()
    app.processEvents()
