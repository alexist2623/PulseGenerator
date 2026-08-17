"""Headless tests for GUI time units and RF port/readout integration.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pytest

import DCWaveform_Generator as gui
import qcs_front_panel
from dc_waveform_core import (
    DEFAULT_QICK_FULL_SCALE_MV,
    PulseSequence,
    QickDdrReadoutSpec,
    QickHoldDurationSweepSpec,
    QickRampRateSweepSpec,
    QickRfPulseSpec,
    QickSweepSpec,
    adc_iq_to_voltage,
    dc_iq_to_current,
    generate_qick_program_code,
)
from stability_diagram import DEFAULT_STABILITY_POINT_GUARD_US
from qick_fine_tune_sweep import FineTuneSequence


_APP = None


def _application():
    global _APP
    if _APP is None:
        _APP = (
            QtWidgets.QApplication.instance()
            or QtWidgets.QApplication([])
        )
    return _APP


def _send_wheel(widget, delta=120):
    local_position = QtCore.QPointF(widget.rect().center())
    global_position = QtCore.QPointF(
        widget.mapToGlobal(widget.rect().center())
    )
    event = QtGui.QWheelEvent(
        local_position,
        global_position,
        QtCore.QPoint(),
        QtCore.QPoint(0, delta),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
        QtCore.Qt.ScrollUpdate,
        False,
    )
    QtWidgets.QApplication.sendEvent(widget, event)


def test_mouse_wheel_does_not_change_value_inputs():
    app = _application()
    window = gui.MainWindow()

    spin_box = window._experiment_panel.fabric_mhz
    spin_box.setValue(300.0)
    spin_box.setFocus()
    _send_wheel(spin_box, 120)
    _send_wheel(spin_box, -120)
    assert spin_box.value() == 300.0

    combo_box = window._time_unit_combo
    combo_box.setCurrentText("us")
    combo_box.setFocus()
    _send_wheel(combo_box, 120)
    _send_wheel(combo_box, -120)
    assert combo_box.currentText() == "us"

    scroll_area = QtWidgets.QScrollArea()
    scroll_content = QtWidgets.QWidget()
    scroll_content.resize(180, 1200)
    nested_spin_box = QtWidgets.QSpinBox(scroll_content)
    nested_spin_box.move(10, 10)
    nested_spin_box.setValue(5)
    scroll_area.setWidget(scroll_content)
    scroll_area.resize(200, 200)
    scroll_area.show()
    app.processEvents()

    _send_wheel(nested_spin_box, -120)
    app.processEvents()
    assert nested_spin_box.value() == 5
    assert scroll_area.verticalScrollBar().value() > 0

    scroll_area.close()
    app.processEvents()
    window.close()


def test_gui_defaults_and_time_unit_round_trip():
    app = _application()
    window = gui.MainWindow()
    control = window._multi_ctrl._ctrl_pannels[0]

    assert window._time_unit == "us"
    assert window._experiment_panel.fabric_mhz.value() == 300.0
    assert window._experiment_panel.tproc_mhz.value() == 300.0
    assert DEFAULT_QICK_FULL_SCALE_MV == 800.0
    assert window._experiment_panel.full_scale_mv.value() == 800.0
    assert window._qick_full_scale_mv == 800.0
    assert window._pulse[0].v_bounds == (-2500.0, 2500.0)
    assert window._experiment_panel.bias_t_group.isChecked() is False
    assert window._experiment_panel.bias_t_type.currentData() == "dc"
    assert window._experiment_panel.bias_t_mode.currentData() == "fixed_voltage"
    assert window._experiment_panel.bias_t_compensation_mv.value() == 80.0
    assert window._experiment_panel.bias_t_duration_us.value() == 1.0
    assert window._experiment_panel.bias_t_filter_tau_us.value() == 100.0
    assert window._experiment_panel.bias_t_compensation_mv.isEnabledTo(
        window._experiment_panel.bias_t_group
    ) is True
    assert window._experiment_panel.bias_t_duration_us.isEnabledTo(
        window._experiment_panel.bias_t_group
    ) is False
    assert window._experiment_panel.bias_t_filter_tau_us.isEnabledTo(
        window._experiment_panel.bias_t_group
    ) is False
    assert window._pulse[0].t.tolist() == [0.0, 1000.0]
    assert window._pulse[0].v.tolist() == [100.0, 100.0]
    assert control.edit_ramp.text() == "1"
    assert control.edit_flat.text() == "1"
    assert control.edit_v.text() == "100"
    assert window._plot.grid_settings == (1000.0, 100.0, False, True)
    file_actions = [
        action.text()
        for action in window.menuBar().actions()[0].menu().actions()
        if not action.isSeparator()
    ]
    assert file_actions[:2] == [
        "Save Settings JSON...",
        "Load Settings JSON...",
    ]

    window._time_unit_combo.setCurrentText("ns")
    app.processEvents()
    assert control.edit_ramp.text() == "1000"
    assert control.table.horizontalHeaderItem(1).text() == "Name"
    assert control.table.horizontalHeaderItem(3).text() == "Flat [ns]"

    window._time_unit_combo.setCurrentText("ms")
    app.processEvents()
    assert control.edit_ramp.text() == "0.001"

    window._time_unit_combo.setCurrentText("us")
    app.processEvents()
    assert control.edit_ramp.text() == "1"
    assert window._plot.getPlotItem().getAxis("bottom").labelText == "time [us]"
    window.close()


def test_shared_qick_setup_replaces_duplicate_tab_controls():
    app = _application()
    window = gui.MainWindow()

    menu_names = [action.text().replace("&", "") for action in window.menuBar().actions()]
    assert "Setup" in menu_names
    setup_menu = next(
        action.menu()
        for action in window.menuBar().actions()
        if action.text().replace("&", "") == "Setup"
    )
    assert "QICK Connection and Clocks..." in [
        action.text() for action in setup_menu.actions()
    ]

    experiment_labels = {
        label.text() for label in window._experiment_panel.findChildren(QtWidgets.QLabel)
    }
    assert "QICK IP/host:" not in experiment_labels
    assert "Pyro nameserver port:" not in experiment_labels
    assert "Pyro proxy name:" not in experiment_labels
    assert "AWG fabric clock:" not in experiment_labels
    assert "tProcessor clock:" not in experiment_labels
    assert "AWG generator indices:" not in experiment_labels
    noise_labels = {
        label.text() for label in window._noise_panel.findChildren(QtWidgets.QLabel)
    }
    assert "QICK connection:" not in noise_labels
    assert window._noise_panel.acquisition_host.isHidden()
    assert window._noise_panel.acquisition_port.isHidden()
    assert window._noise_panel.acquisition_proxy.isHidden()

    connection = gui.QickConnectionConfig(
        host="192.0.2.88",
        ns_port=9777,
        proxy_name="shared-qick",
    )
    window._apply_shared_qick_setup(connection, 312.5, 287.5)
    assert window._experiment_panel.qick_host.text() == "192.0.2.88"
    assert window._experiment_panel.ns_port.value() == 9777
    assert window._experiment_panel.proxy_name.text() == "shared-qick"
    assert window._experiment_panel.fabric_mhz.value() == 312.5
    assert window._experiment_panel.tproc_mhz.value() == 287.5
    assert window._noise_panel.acquisition_host.text() == "192.0.2.88"
    assert window._noise_panel.acquisition_port.value() == 9777
    assert window._noise_panel.acquisition_proxy.text() == "shared-qick"
    app.processEvents()
    window.close()


def test_awg_front_panel_mapping_and_horizontal_port_scroll():
    app = _application()
    window = gui.MainWindow()
    window._experiment_panel.set_execution_backend(
        gui.EXECUTION_BACKEND_QICK
    )
    window.resize(860, 720)
    window.show()
    for _ in range(3):
        window._add_port()
    app.processEvents()

    multi = window._multi_ctrl
    assert multi.front_panel_preview._scope == "output"
    assert multi.panel_table.columnCount() == 5
    assert multi.splitter.minimumWidth() >= 4 * 310
    assert multi.panel_scroll.horizontalScrollBar().maximum() > 0
    assert window._selected_port_idx == 3
    assert "awg_3" in multi.mapping_summary.text()
    assert "generator 7" in multi.mapping_summary.text()

    multi.set_selected_port(0)
    multi.apply_front_panel_settings({"output_ch": 3})
    assert window._qick_awg_channels == (3, 1, 5, 7)
    assert multi.panel_table.item(0, 2).text() == "gen 3"
    assert multi.panel_table.item(1, 2).text() == "gen 1"
    assert window._experiment_panel.awg_channels.text() == "3, 1, 5, 7"
    window._port_select(3)
    window._delete_port(3)
    assert window._selected_port_idx == 2
    assert window._qick_awg_channels == (3, 1, 5)
    assert len(multi._ctrl_pannels) == 3
    app.processEvents()
    window.close()


def test_rf_readout_input_condition_stays_single_row_height():
    app = _application()
    pulse = PulseSequence(100.0, initial_duration_ns=1000.0)
    panel = gui.RfReadoutPanel(pulse, time_unit="us")
    panel.setChecked(True)
    panel.resize(640, 900)
    panel.show()
    app.processEvents()

    expected_height = max(
        panel.attenuation_db.sizeHint().height(),
        panel.dc_gain_db.sizeHint().height(),
    )
    assert panel.input_condition_stack.height() == expected_height
    assert (
        panel.input_condition_stack.sizePolicy().verticalPolicy()
        == QtWidgets.QSizePolicy.Fixed
    )
    panel.input_board_type.setCurrentText("DC_In")
    app.processEvents()
    assert panel.input_condition_stack.height() == expected_height
    panel.close()


def test_segment_sweep_dialog_uses_voltage_values_but_returns_normalized_spec():
    app = _application()
    dialog = gui.SweepSettingsDialog(
        output_name="awg_2",
        segment_name="gate",
        current_amplitude=0.25,
        full_scale_mv=200.0,
        initial=QickSweepSpec("gate", "awg_2", -0.5, 0.75, 7),
    )

    assert dialog.start.suffix() == " mV"
    assert dialog.start.value() == -100.0
    assert dialog.stop.value() == 150.0
    dialog.start.setValue(-40.0)
    dialog.stop.setValue(80.0)
    spec = dialog.value()
    assert spec.start == -0.2
    assert spec.stop == 0.4
    assert spec.count == 7
    app.processEvents()
    dialog.close()


def test_qcs_awg_voltage_and_segment_sweep_use_qcs_full_scale(monkeypatch):
    app = _application()
    window = gui.MainWindow()
    experiment = window._experiment_panel
    control = window._multi_ctrl._ctrl_pannels[0]
    experiment.qcs_dc_full_scale_v.setValue(2.5)
    app.processEvents()

    assert experiment.execution_backend() == gui.EXECUTION_BACKEND_QCS
    assert experiment.full_scale_mv.value() == pytest.approx(800.0)
    assert window._pulse[0].v_bounds == (-2500.0, 2500.0)

    # The AWG table stores physical millivolts.  QCS must not silently clip
    # this legal M5301 value to the dormant QICK +/-800 mV limit.
    control.table.item(0, 4).setText("1500")
    app.processEvents()
    np.testing.assert_allclose(window._pulse[0].v[:2], [1500.0, 1500.0])
    assert float(control.table.item(0, 4).text()) == pytest.approx(1500.0)

    captured = {}

    class AcceptedSweepDialog:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        @staticmethod
        def exec_():
            return QtWidgets.QDialog.Accepted

        @staticmethod
        def value():
            return QickSweepSpec("set_0", "awg_0", -0.6, 0.8, 5)

    monkeypatch.setattr(gui, "SweepSettingsDialog", AcceptedSweepDialog)
    window._configure_segment_sweep(0, 0)
    app.processEvents()

    assert captured["full_scale_mv"] == pytest.approx(2500.0)
    assert captured["current_amplitude"] == pytest.approx(0.6)
    assert window._sweep_specs == [
        QickSweepSpec("set_0", "awg_0", -0.6, 0.8, 5)
    ]
    assert experiment.sweep_parameter_table.rowCount() == 1
    experiment.sweep_parameter_table.selectRow(0)
    app.processEvents()
    assert experiment.sweep_parameter_start.minimum() == pytest.approx(-2500.0)
    assert experiment.sweep_parameter_start.maximum() == pytest.approx(2500.0)
    assert experiment.sweep_parameter_start.value() == pytest.approx(-1500.0)
    assert experiment.sweep_parameter_stop.value() == pytest.approx(2000.0)

    graphics = window._plot._sweep_graphics[("awg_0", "set_0")]
    np.testing.assert_allclose(graphics["lower_mv"][:2], [-1500.0, -1500.0])
    np.testing.assert_allclose(graphics["upper_mv"][:2], [2000.0, 2000.0])
    window.close()
    window.deleteLater()
    app.processEvents()


def test_awg_scale_changes_preserve_physical_voltage_sweep_coordinates():
    app = _application()
    window = gui.MainWindow()
    experiment = window._experiment_panel
    window._sweep_specs = [
        QickSweepSpec("set_0", "awg_0", -0.2, 0.24, 5)
    ]
    window._refresh_sweep_overlay(sync_rows=True)

    # Expand QCS from +/-2.5 V to +/-5 V.  The normalized coordinates change,
    # while their physical -500 mV and +600 mV endpoints do not.
    experiment.qcs_dc_full_scale_v.setValue(5.0)
    app.processEvents()
    assert window._sweep_specs[0].start == pytest.approx(-0.1)
    assert window._sweep_specs[0].stop == pytest.approx(0.12)
    assert window._pulse[0].v_bounds == (-5000.0, 5000.0)

    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    app.processEvents()
    assert window._sweep_specs[0].start == pytest.approx(-0.625)
    assert window._sweep_specs[0].stop == pytest.approx(0.75)
    assert window._pulse[0].v_bounds == (-800.0, 800.0)

    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    app.processEvents()
    assert window._sweep_specs[0].start == pytest.approx(-0.1)
    assert window._sweep_specs[0].stop == pytest.approx(0.12)
    assert window._pulse[0].v_bounds == (-5000.0, 5000.0)
    document = window._settings_to_dict()
    assert document["awg"]["voltage_coordinate_full_scale_mv"] == 5000.0

    window.close()
    window.deleteLater()
    app.processEvents()


def test_ramp_rate_sweep_dialog_uses_duration_and_reports_derived_rate():
    app = _application()
    initial = QickRampRateSweepSpec("ramp_0_to_1", 0.08, 0.12, 5)
    dialog = gui.RampRateSweepSettingsDialog(
        segment_name="ramp_0_to_1",
        current_duration_us=0.1,
        voltage_delta_mv=200.0,
        initial=initial,
        cartesian_base_count=7,
    )

    assert dialog.start.value() == 0.08
    assert dialog.stop.value() == 0.12
    assert dialog.count.value() == 5
    assert "2500" in dialog.rate_summary.text()
    assert "1666.666" in dialog.rate_summary.text()
    assert dialog.cartesian_summary.text() == "7 x 5 = 35 points"
    assert dialog.value() == initial
    app.processEvents()
    dialog.close()


def test_hold_duration_sweep_dialog_uses_shared_set_timing():
    app = _application()
    initial = QickHoldDurationSweepSpec("set_1", 1.0, 5.0, 5)
    dialog = gui.HoldDurationSweepSettingsDialog(
        segment_name="set_1",
        current_duration_us=2.0,
        initial=initial,
        cartesian_base_count=7,
    )

    assert dialog.start.value() == 1.0
    assert dialog.stop.value() == 5.0
    assert dialog.count.value() == 5
    assert dialog.cartesian_summary.text() == "7 x 5 = 35 points"
    assert dialog.value() == initial
    app.processEvents()
    dialog.close()


def test_export_sweep_editor_displays_mv_and_tracks_full_scale():
    app = _application()
    dialog = gui.QickExportDialog(
        pulse_count=1,
        set_names=("set_0", "set_1"),
        initial_full_scale_mv=200.0,
        initial_sweeps=(QickSweepSpec("set_1", "awg_0", -0.5, 0.5, 5),),
    )
    app.processEvents()

    assert dialog.sweep_table.horizontalHeaderItem(2).text() == "Start (mV)"
    assert dialog.sweep_start.value() == -100.0
    assert dialog.sweep_stop.value() == 100.0
    dialog.sweep_start.setValue(-40.0)
    dialog.sweep_stop.setValue(60.0)
    assert dialog._current_sweep_spec().start == -0.2
    assert dialog._current_sweep_spec().stop == 0.3

    dialog.full_scale_mv.setValue(400.0)
    app.processEvents()
    assert dialog.sweep_start.value() == -80.0
    assert dialog.sweep_stop.value() == 120.0
    assert dialog._current_sweep_spec().start == -0.2
    assert dialog._current_sweep_spec().stop == 0.3
    dialog.close()


def test_export_dialog_preserves_multiple_ramp_rate_axes():
    app = _application()
    ramp_a = QickRampRateSweepSpec("ramp_0_to_1", 0.08, 0.12, 3)
    ramp_b = QickRampRateSweepSpec("ramp_1_to_2", 0.10, 0.16, 4)
    hold = QickHoldDurationSweepSpec("set_1", 0.5, 1.0, 2)
    voltage = QickSweepSpec("set_2", "awg_0", -0.5, 0.5, 5)
    dialog = gui.QickExportDialog(
        pulse_count=1,
        set_names=("set_0", "set_1", "set_2"),
        initial_full_scale_mv=200.0,
        initial_sweeps=(ramp_a, ramp_b, hold, voltage),
    )
    app.processEvents()

    assert dialog._effective_sweeps() == (ramp_a, ramp_b, hold, voltage)
    assert dialog.sweep_total.text() == "3 x 4 x 2 x 5 = 120 combinations"
    dialog.sweep_group.setChecked(False)
    app.processEvents()
    assert dialog._effective_sweeps() == (ramp_a, ramp_b, hold)
    assert dialog.sweep_total.text() == "3 x 4 x 2 = 24 combinations"
    dialog.close()


def test_awg_tuning_tab_groups_awg_rf_and_experiment_controls():
    app = _application()
    window = gui.MainWindow()
    assert [
        window._control_tabs.tabText(i)
        for i in range(window._control_tabs.count())
    ] == [
        "AWG Tuning",
        "Stability Diagram",
        "RF S-Parameter",
        "Calibration",
        "Noise Analysis",
    ]
    assert [
        window._awg_tuning_tabs.tabText(i)
        for i in range(window._awg_tuning_tabs.count())
    ] == [
        "AWG Outputs",
        "RF Outputs",
        "QCS Acquisition",
        "Experiment",
    ]
    window._show_rf_editor()
    assert window._control_tabs.currentWidget() is window._awg_tuning_page
    assert window._awg_tuning_tabs.currentWidget() is window._rf_ports_panel
    toolbar_labels = [action.text() for bar in window.findChildren(QtWidgets.QToolBar)
                      for action in bar.actions()]
    assert "RF Pulse" not in toolbar_labels

    first = window._rf_ports_panel._panels[0]
    first.setChecked(True)
    first.filter_type.setCurrentText("lowpass")
    first.filter_cutoff.setValue(2.25)
    first.filter_bandwidth.setValue(0.75)
    window._rf_ports_panel.add_port()
    second = window._rf_ports_panel._panels[1]
    second.setChecked(True)
    app.processEvents()
    specs = window._rf_ports_panel.specs()
    assert [spec.gen_ch for spec in specs] == [0, 2]
    assert all(spec.duration_us == 1.0 for spec in specs)
    assert specs[0].filter_type == "lowpass"
    assert specs[0].filter_cutoff == 2.25
    assert specs[0].filter_bandwidth == 0.75
    assert len(window._rf_timelines) == 2

    window._rf_ports_panel.remove_port(second)
    app.processEvents()
    assert len(window._rf_ports_panel.specs()) == 1
    assert len(window._rf_timelines) == 1
    window.close()


def test_qcs_acquisition_hides_qick_fir_ddr_rows_and_uses_qcs_terms():
    app = _application()
    window = gui.MainWindow()
    window.show()
    window._control_tabs.setCurrentWidget(window._awg_tuning_page)
    window._awg_tuning_tabs.setCurrentWidget(window._rf_readout_panel)
    panel = window._rf_readout_panel
    experiment = window._experiment_panel
    app.processEvents()

    qick_only_fields = (
        panel.input_board_type,
        panel.input_condition_stack,
        panel.measurement_unit,
        panel.dc_measure_gain_v_per_a,
        panel.dc_voltage_calibration_enabled,
        panel.calibration_path_widget,
        panel.dc_voltage_calibration_run_id,
        panel.filter_type,
        panel.filter_cutoff,
        panel.filter_bandwidth,
        panel.margin_samples,
        panel.fpga_delay_widget,
        panel.post_run_read_delay,
        panel.force_overwrite,
        panel.fir_profile_note,
    )
    assert experiment.execution_backend() == gui.EXECUTION_BACKEND_QCS
    assert panel.title() == "QCS Acquisition 1"
    assert (
        window._awg_tuning_tabs.tabText(
            window._awg_tuning_tabs.indexOf(panel)
        )
        == "QCS Acquisition"
    )
    assert all(field.isHidden() for field in qick_only_fields)
    assert panel.ro_ch.isHidden() is True
    assert panel.segment.isHidden() is False
    assert panel.delay.isHidden() is False
    assert panel.samples.isHidden() is True
    assert panel.qcs_acquisition_duration.isHidden() is False
    assert panel.frequency_mhz.isHidden() is False
    assert panel.qcs_acquisition_mode_widget.isHidden() is False
    assert panel.qcs_single_iq_radio.isChecked() is True
    assert panel.qcs_trace_radio.isChecked() is False
    assert experiment.qcs_hw_demod.isHidden() is True
    assert panel.segment_label.text() == "Acquisition segment:"
    assert panel._delay_label.text() == "Acquisition pre-delay [us]:"
    assert panel.qcs_acquisition_duration_label.text() == (
        "Total I/Q averaging time [us]:"
    )
    assert (
        panel.frequency_label.text()
        == "Integration-filter RF frequency:"
    )
    acquisition_note = panel.qcs_acquisition_note.text()
    assert "IntegrationFilter" in acquisition_note
    assert "64 M5200 samples at 4.8 GSPS" in acquisition_note
    assert "programmed time 0.0133333333333 us" in acquisition_note
    assert experiment.qcs_sample_rate_hz.text() == "4.8 GSPS"
    assert experiment.qcs_sample_rate_hz.isReadOnly() is True
    assert experiment.qcs_sample_rate_hz.isHidden() is True
    panel.qcs_acquisition_duration.setValue(1.0)
    app.processEvents()
    assert panel.configured_spec().samples_per_trigger == 4_800
    assert "4,800 M5200 samples" in panel.qcs_acquisition_note.text()
    panel.setChecked(True)
    app.processEvents()
    assert "one I/Q value/shot from 4,800 total averaging samples" in (
        window.statusBar().currentMessage()
    )
    visible_text = "\n".join(
        label.text()
        for label in panel.findChildren(QtWidgets.QLabel)
        if label.isVisible()
    ).lower()
    assert all(
        legacy_word not in visible_text
        for legacy_word in ("fir", "ddr", "ddc", "hwh")
    )

    panel.margin_samples.setValue(4321)
    panel.override_fpga_trigger_delay.setChecked(True)
    panel.fpga_trigger_delay_us.setValue(17.5)
    panel.input_board_type.setCurrentText("DC_In")
    panel.measurement_unit.setCurrentIndex(
        panel.measurement_unit.findData("current")
    )
    panel.qcs_trace_radio.click()
    app.processEvents()

    assert panel.qcs_trace_radio.isChecked() is True
    assert panel.qcs_single_iq_radio.isChecked() is False
    assert experiment.qcs_hw_demod.isChecked() is False
    assert panel.frequency_mhz.isHidden() is True
    assert panel.frequency_label.isHidden() is True
    assert "Raw acquisition" in panel.qcs_acquisition_note.text()
    assert "4,800 trace samples/shot" in window.statusBar().currentMessage()

    panel.qcs_single_iq_radio.click()
    app.processEvents()
    assert panel.qcs_single_iq_radio.isChecked() is True
    assert panel.qcs_trace_radio.isChecked() is False
    assert experiment.qcs_hw_demod.isChecked() is True
    assert panel.frequency_mhz.isHidden() is False
    assert "one I/Q value/shot from 4,800 total averaging samples" in (
        window.statusBar().currentMessage()
    )

    experiment.set_running(True, "Test run active")
    app.processEvents()
    assert panel.qcs_acquisition_mode_widget.isEnabled() is False
    assert panel.qcs_acquisition_duration.isEnabled() is False
    panel.qcs_trace_radio.click()
    assert panel.qcs_single_iq_radio.isChecked() is True
    experiment.set_running(False, "Test run complete")
    app.processEvents()
    assert panel.qcs_acquisition_mode_widget.isEnabled() is True
    assert panel.qcs_acquisition_duration.isEnabled() is True

    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    app.processEvents()

    assert panel.title() == "RF Readout 1"
    assert (
        window._awg_tuning_tabs.tabText(
            window._awg_tuning_tabs.indexOf(panel)
        )
        == "RF Readout"
    )
    assert all(not field.isHidden() for field in qick_only_fields)
    assert panel.ro_ch.isHidden() is False
    assert panel.qcs_acquisition_note.isHidden() is True
    assert panel.qcs_acquisition_mode_widget.isHidden() is True
    assert panel.qcs_acquisition_duration.isHidden() is True
    assert panel.samples.isHidden() is False
    assert panel.segment_label.text() == "Anchor SET:"
    assert panel._delay_label.text() == "Trigger delay [us]:"
    assert panel.samples_label.text() == "Stored FIR samples:"
    assert panel.samples.value() == 64
    assert panel.frequency_label.text() == "Readout/DDC frequency:"
    assert panel.margin_samples.value() == 4321
    assert panel.override_fpga_trigger_delay.isChecked() is True
    assert panel.fpga_trigger_delay_us.value() == pytest.approx(17.5)
    assert panel.measurement_unit.currentData() == "current"

    window.close()
    app.processEvents()


def test_qcs_acquisition_time_rounds_up_and_preserves_qick_samples():
    app = _application()
    window = gui.MainWindow()
    panel = window._rf_readout_panel
    experiment = window._experiment_panel
    panel.samples.setValue(321)

    panel.qcs_acquisition_duration.setValue(0.0101)  # 10.1 ns
    app.processEvents()
    assert panel.qcs_acquisition_duration.value() == pytest.approx(0.0101)
    assert panel.configured_spec().samples_per_trigger == 64
    assert "64 M5200 samples" in panel.qcs_acquisition_note.text()
    assert "programmed time 0.0133333333333 us" in (
        panel.qcs_acquisition_note.text()
    )

    panel.qcs_trace_radio.click()
    app.processEvents()
    assert panel.qcs_acquisition_duration.value() == pytest.approx(0.0101)
    assert panel.configured_spec().samples_per_trigger == 49
    assert "49 M5200 samples" in panel.qcs_acquisition_note.text()
    assert "programmed time 0.0102083333333 us" in (
        panel.qcs_acquisition_note.text()
    )

    panel.qcs_acquisition_duration.setValue(0.001)  # 1 ns
    assert panel.configured_spec().samples_per_trigger == 5
    panel.qcs_single_iq_radio.click()
    app.processEvents()
    assert panel.qcs_acquisition_duration.value() == pytest.approx(0.001)
    assert panel.configured_spec().samples_per_trigger == 16

    panel.qcs_trace_radio.click()
    panel.qcs_acquisition_duration.setValue(0.0101)

    panel.set_time_unit("ns")
    assert panel.qcs_acquisition_duration.value() == pytest.approx(10.1)
    assert panel.qcs_acquisition_duration_label.text() == (
        "Raw trace duration [ns]:"
    )
    assert panel.configured_spec().samples_per_trigger == 49

    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    app.processEvents()
    assert panel.samples.value() == 321
    assert panel.samples.isHidden() is False
    assert panel.qcs_acquisition_duration.isHidden() is True

    window.close()
    app.processEvents()


def test_qcs_total_iq_duration_has_a_separate_100_ms_ceiling_from_trace():
    app = _application()
    window = gui.MainWindow()
    panel = window._rf_readout_panel

    assert panel.qcs_single_iq_radio.isChecked() is True
    assert panel.qcs_acquisition_duration.maximum() == pytest.approx(100_000.0)
    panel.qcs_acquisition_duration.setValue(100_000.0)
    app.processEvents()

    assert panel.configured_spec().samples_per_trigger == 480_000_000
    assert "1,000 bounded QCS execution passes" in (
        panel.qcs_acquisition_note.text()
    )
    assert "Total I/Q averaging time" in (
        panel.qcs_acquisition_duration_label.text()
    )

    panel.qcs_trace_radio.click()
    app.processEvents()

    assert panel.qcs_acquisition_duration.maximum() == pytest.approx(
        10_000_000 / 4.8e9 * 1.0e6
    )
    assert panel.configured_spec().samples_per_trigger == 10_000_000
    assert "Raw trace duration" in panel.qcs_acquisition_duration_label.text()
    assert "10,000,000-sample" in panel.qcs_acquisition_duration.toolTip()

    panel.qcs_single_iq_radio.click()
    app.processEvents()
    assert panel.qcs_acquisition_duration.maximum() == pytest.approx(100_000.0)

    window.close()
    app.processEvents()


def test_single_iq_repetition_save_policy_tracks_acquisition_mode():
    app = _application()
    window = gui.MainWindow()
    experiment = window._experiment_panel
    readout = window._rf_readout_panel

    assert experiment.iq_repetition_policy_value() == (
        gui.IQ_REPETITION_POLICY_PRESERVE
    )
    assert experiment.iq_repetition_policy.isEnabled() is False
    assert "Enable acquisition" in experiment.iq_repetition_policy_note.text()

    readout.setChecked(True)
    experiment.repetitions.setValue(4)
    app.processEvents()
    assert readout.qcs_single_iq_radio.isChecked() is True
    assert experiment.iq_repetition_policy.isEnabled() is True
    experiment.set_iq_repetition_policy(
        gui.IQ_REPETITION_POLICY_COHERENT_AVERAGE
    )
    assert experiment.iq_repetition_policy_value(effective=True) == (
        gui.IQ_REPETITION_POLICY_COHERENT_AVERAGE
    )
    assert "mean I" in experiment.iq_repetition_policy_note.text()

    readout.qcs_trace_radio.click()
    app.processEvents()
    assert experiment.iq_repetition_policy.isEnabled() is False
    assert experiment.iq_repetition_policy_value() == (
        gui.IQ_REPETITION_POLICY_COHERENT_AVERAGE
    )
    assert experiment.iq_repetition_policy_value(effective=True) == (
        gui.IQ_REPETITION_POLICY_PRESERVE
    )
    assert "Trace acquisition always preserves" in (
        experiment.iq_repetition_policy_note.text()
    )

    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    readout.samples.setValue(1)
    app.processEvents()
    assert experiment.iq_repetition_policy.isEnabled() is True
    assert experiment.iq_repetition_policy_value(effective=True) == (
        gui.IQ_REPETITION_POLICY_COHERENT_AVERAGE
    )
    readout.samples.setValue(2)
    app.processEvents()
    assert experiment.iq_repetition_policy.isEnabled() is False
    assert experiment.iq_repetition_policy_value(effective=True) == (
        gui.IQ_REPETITION_POLICY_PRESERVE
    )

    window.close()
    app.processEvents()


def test_qick_run_arguments_forward_only_effective_repetition_policy(tmp_path):
    app = _application()
    window = gui.MainWindow()
    experiment = window._experiment_panel
    readout = window._rf_readout_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    experiment.database_path.setText(str(tmp_path / "qick_single_iq.db"))
    experiment.repetitions.setValue(5)
    experiment.set_iq_repetition_policy(
        gui.IQ_REPETITION_POLICY_COHERENT_AVERAGE
    )
    readout.setChecked(True)
    readout.samples.setValue(1)
    app.processEvents()

    arguments = window._experiment_run_arguments(
        validate_qick_hardware=False,
    )
    assert arguments["repetitions_per_sweep"] == 5
    assert arguments["iq_repetition_policy"] == (
        gui.IQ_REPETITION_POLICY_COHERENT_AVERAGE
    )

    readout.samples.setValue(2)
    app.processEvents()
    trace_arguments = window._experiment_run_arguments(
        validate_qick_hardware=False,
    )
    assert trace_arguments["iq_repetition_policy"] == (
        gui.IQ_REPETITION_POLICY_PRESERVE
    )

    window.close()
    app.processEvents()


def test_qcs_rf_output_uses_qcs_waveform_labels_and_module_choices():
    app = _application()
    window = gui.MainWindow()
    experiment = window._experiment_panel
    panel = window._rf_ports_panel._panels[0]

    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    panel.output_board_type.setCurrentText("DC_Out")
    panel.gain.setValue(-12345)
    panel.nqz.setValue(2)
    panel.att1_db.setValue(7.25)
    panel.att2_db.setValue(12.5)
    panel.filter_type.setCurrentText("bandpass")
    panel.filter_cutoff.setValue(3.25)
    panel.filter_bandwidth.setValue(0.75)
    panel.power_calibration_database_path.setText("legacy_power.db")
    panel.target_output_power_dbm.setValue(-31.5)

    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    window._control_tabs.setCurrentWidget(window._awg_tuning_page)
    window._awg_tuning_tabs.setCurrentWidget(window._rf_ports_panel)
    window.show()
    app.processEvents()

    module_items = [
        (
            panel.qcs_module_model.itemText(index),
            panel.qcs_module_model.itemData(index),
        )
        for index in range(panel.qcs_module_model.count())
    ]
    assert module_items == [
        ("M5300A RF AWG", "M5300A"),
        ("M5301A Precision AWG", "M5301A"),
    ]
    assert panel.qcs_module_model.isVisible() is True
    assert panel.qcs_module_model.isEnabled() is False
    assert panel.qcs_amplitude.isVisible() is True
    assert window._rf_ports_panel.add_button.text() == "Add RF Output"
    assert panel.remove_button.text() == "Remove RF Output"
    assert panel.qcs_amplitude.minimum() == pytest.approx(-1.0)
    assert panel.qcs_amplitude.maximum() == pytest.approx(1.0)
    assert panel.qcs_amplitude.value() == pytest.approx(
        -12345 / 32768,
        abs=0.5e-6,
    )
    configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("gate",),
        {panel.gen_ch.value(): "rf_drive"},
        "digitizer",
    )
    panel.set_qcs_front_panel_configuration(configuration)
    assert panel.qcs_module_model.currentData() == "M5300A"
    assert panel.qcs_power_calibration_group.isHidden() is False
    assert panel.qcs_power_calibration_group.isEnabled() is True
    moved_mappings = []
    for mapping in configuration["channel_mappings"]:
        mapping = dict(mapping)
        if mapping["role"] == "rf":
            mapping.update(
                slot=2,
                channel=2,
                absolute_phase=False,
                lo_frequency_hz=None,
            )
        moved_mappings.append(mapping)
    configuration = qcs_front_panel.normalize_qcs_hardware_configuration(
        {
            **configuration,
            "channel_mappings": moved_mappings,
        }
    )
    panel.set_qcs_front_panel_configuration(configuration)
    assert panel.qcs_module_model.currentData() == "M5301A"
    assert panel.qcs_power_calibration_group.isVisible() is True
    assert panel.qcs_power_calibration_group.isEnabled() is False
    assert "slot 2, SMA CH 2" in panel.qcs_module_model.toolTip()

    form = panel._form_layout
    assert form.labelForField(panel.qcs_module_model).text() == "Module:"
    assert form.labelForField(panel.segment).text() == "Waveform segment:"
    assert form.labelForField(panel.delay).text() == "Pre-delay [us]:"
    assert form.labelForField(panel.duration).text() == "Waveform duration [us]:"
    assert form.labelForField(panel.frequency_mhz).text() == "RF frequency:"
    assert form.labelForField(panel.qcs_amplitude).text() == "Relative amplitude:"
    assert form.labelForField(panel.phase_degrees).text() == "Instantaneous phase:"

    qick_only_fields = (
        panel.output_board_type,
        panel.power_calibration_group,
        panel.gain,
        panel.nqz,
    )
    for field in qick_only_fields:
        assert field.isHidden() is True
        label = form.labelForField(field)
        if label is not None:
            assert label.isHidden() is True
    visible_text = " ".join(
        label.text()
        for label in panel.findChildren(QtWidgets.QLabel)
        if label.isVisible()
    )
    assert "HWH" not in visible_text
    assert "Nyquist" not in visible_text
    assert "RF_Out" not in visible_text
    assert "DC_Out" not in visible_text

    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    app.processEvents()

    assert panel.qcs_module_model.isHidden() is True
    assert panel.qcs_amplitude.isHidden() is True
    assert panel.qcs_power_calibration_group.isHidden() is True
    assert window._rf_ports_panel.add_button.text() == "Add RF Port"
    assert panel.remove_button.text() == "Remove RF Port"
    for field in qick_only_fields:
        assert field.isHidden() is False
    assert form.labelForField(panel.output_board_type).text() == "Output board:"
    assert form.labelForField(panel.segment).text() == "Anchor SET:"
    assert form.labelForField(panel.delay).text() == "Delay [us]:"
    assert form.labelForField(panel.duration).text() == "Duration [us]:"
    assert form.labelForField(panel.frequency_mhz).text() == "Frequency:"
    assert form.labelForField(panel.gain).text() == "Gain:"
    assert form.labelForField(panel.phase_degrees).text() == "Phase:"
    assert form.labelForField(panel.nqz).text() == "Nyquist zone:"
    assert panel.output_board_type.currentText() == "DC_Out"
    assert panel.gain.value() == -12345
    assert panel.nqz.value() == 2
    assert panel.att1_db.value() == pytest.approx(7.25)
    assert panel.att2_db.value() == pytest.approx(12.5)
    assert panel.filter_type.currentText() == "bandpass"
    assert panel.filter_cutoff.value() == pytest.approx(3.25)
    assert panel.filter_bandwidth.value() == pytest.approx(0.75)
    assert panel.power_calibration_database_path.text() == "legacy_power.db"
    assert panel.target_output_power_dbm.value() == pytest.approx(-31.5)
    visible_text = " ".join(
        label.text()
        for label in panel.findChildren(QtWidgets.QLabel)
        if label.isVisible()
    )
    assert "HWH-backed Front Panel" in visible_text

    window.close()
    app.processEvents()


def test_awg_tuning_qcs_fixed_output_power_builds_calibration_request(
    tmp_path,
):
    app = _application()
    window = gui.MainWindow()
    experiment = window._experiment_panel
    panel = window._rf_ports_panel._panels[0]
    logical_index = panel.gen_ch.value()
    mapper_path = tmp_path / "qcs_mapper.qcs"
    mapper_path.write_text("{}", encoding="utf-8")
    calibration_path = tmp_path / "m5300_power.db"
    calibration_path.write_bytes(b"calibration")

    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    experiment.qcs_mapper_path.setText(str(mapper_path))
    experiment.qcs_dc_channel_names.setText("gate")
    experiment.qcs_rf_channel_names.setText(
        f"{logical_index}=rf_drive"
    )
    experiment.qcs_acquisition_channel_name.setText("digitizer")
    configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("gate",),
        {logical_index: "rf_drive"},
        "digitizer",
    )
    panel.set_qcs_front_panel_configuration(configuration)
    panel.setChecked(True)
    panel.frequency_mhz.setValue(1250.0)
    panel.qcs_amplitude.setValue(0.007)
    panel.qcs_power_calibration_database_path.setText(
        str(calibration_path)
    )
    panel.qcs_power_calibration_run_id.setValue(23)
    panel.qcs_target_output_power_dbm.setValue(-27.5)
    panel.qcs_power_calibration_group.setChecked(True)
    window._rf_readout_panel.setChecked(True)
    app.processEvents()

    assert panel.qcs_power_calibration_group.isHidden() is False
    assert panel.qcs_power_calibration_group.isEnabled() is True
    assert panel.qcs_amplitude.isEnabled() is False
    arguments = window._qcs_experiment_run_arguments()
    pulse = arguments["rf_pulses"][0]

    assert pulse.amplitude == pytest.approx(panel.qcs_amplitude.value())
    assert pulse.power_calibration.database_path == str(calibration_path)
    assert pulse.power_calibration.run_id == 23
    assert pulse.power_calibration.target_power_dbm == pytest.approx(-27.5)
    settings = panel.settings_dict()
    assert settings["qcs_power_calibration_enabled"] is True
    assert settings["qcs_power_calibration_database_path"] == str(
        calibration_path
    )
    assert settings["qcs_power_calibration_run_id"] == 23
    assert settings["qcs_target_output_power_dbm"] == pytest.approx(-27.5)

    panel.frequency_sweep_enabled.setChecked(True)
    with pytest.raises(ValueError, match="fixed RF frequency"):
        window._qcs_experiment_run_arguments()

    window.close()
    app.processEvents()


def test_qcs_rf_output_shows_and_requests_m5300_lo_without_front_panel():
    app = _application()
    window = gui.MainWindow()
    panel = window._rf_ports_panel._panels[0]
    logical_index = panel.gen_ch.value()
    configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("gate",),
        {logical_index: "rf_drive"},
        "digitizer",
    )
    for mapping in configuration["channel_mappings"]:
        if mapping["role"] == "rf":
            mapping["lo_frequency_hz"] = 6.25e9
    configuration = qcs_front_panel.normalize_qcs_hardware_configuration(
        configuration
    )
    window._rf_ports_panel.set_hardware_backend(
        gui.EXECUTION_BACKEND_QCS
    )
    window._rf_ports_panel.set_qcs_front_panel_configuration(
        configuration,
        {logical_index: "rf_drive"},
    )
    window._control_tabs.setCurrentWidget(window._awg_tuning_page)
    window._awg_tuning_tabs.setCurrentWidget(window._rf_ports_panel)
    window.show()
    app.processEvents()

    assert panel.qcs_lo_frequency_row.isVisible() is True
    assert panel.qcs_lo_frequency_ghz.value() == pytest.approx(6.25)
    assert "6.25 GHz" in panel.qcs_lo_frequency_status.text()
    requests = []
    window._rf_ports_panel.lo_frequency_change_requested.disconnect()
    window._rf_ports_panel.lo_frequency_change_requested.connect(
        lambda channel, frequency: requests.append((channel, frequency))
    )
    panel.qcs_lo_frequency_ghz.setValue(6.5)
    panel.apply_qcs_lo_frequency.click()
    app.processEvents()

    assert requests == [(logical_index, pytest.approx(6.5e9))]
    window.close()
    app.processEvents()


def test_qcs_rf_output_enables_lo_after_m5301_to_m5300_remap():
    app = _application()
    window = gui.MainWindow()
    panel = window._rf_ports_panel._panels[0]
    logical_index = panel.gen_ch.value()
    m5300_configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("gate",),
        {logical_index: "rf_drive"},
        "digitizer",
    )
    m5301_configuration = {
        **m5300_configuration,
        "modules": [
            dict(module)
            for module in m5300_configuration["modules"]
            if module["model"] != "M5300A"
        ],
        "channel_mappings": [
            dict(mapping)
            for mapping in m5300_configuration["channel_mappings"]
        ],
    }
    for mapping in m5301_configuration["channel_mappings"]:
        if mapping["role"] == "rf":
            mapping.update(slot=2, channel=2, lo_frequency_hz=None)
    m5301_configuration = (
        qcs_front_panel.normalize_qcs_hardware_configuration(
            m5301_configuration
        )
    )
    for mapping in m5300_configuration["channel_mappings"]:
        if mapping["role"] == "rf":
            mapping["lo_frequency_hz"] = 6.25e9
    m5300_configuration = (
        qcs_front_panel.normalize_qcs_hardware_configuration(
            m5300_configuration
        )
    )

    window._rf_ports_panel.set_hardware_backend(
        gui.EXECUTION_BACKEND_QCS
    )
    window._rf_ports_panel.set_qcs_front_panel_configuration(
        m5301_configuration,
        {logical_index: "rf_drive"},
    )
    panel._keep_front_panel_preview_enabled()
    assert panel.isChecked() is False
    assert panel.qcs_lo_frequency_row.isEnabled() is False

    window._rf_ports_panel.set_qcs_front_panel_configuration(
        m5300_configuration,
        {logical_index: "rf_drive"},
    )
    window._control_tabs.setCurrentWidget(window._awg_tuning_page)
    window._awg_tuning_tabs.setCurrentWidget(window._rf_ports_panel)
    window.show()
    app.processEvents()

    assert panel.qcs_lo_frequency_row.isVisible() is True
    assert panel.qcs_lo_frequency_row.isEnabled() is True
    assert panel.qcs_lo_frequency_ghz.isEnabled() is True
    assert panel.apply_qcs_lo_frequency.isEnabled() is True
    window.close()
    app.processEvents()


def test_main_window_routes_m5300_lo_change_through_mapper_commit(monkeypatch):
    app = _application()
    window = gui.MainWindow()
    routed = []
    monkeypatch.setattr(
        window,
        "_on_qcs_front_panel_connector_selected",
        lambda *args: routed.append(args),
    )

    window._on_qcs_m5300_lo_frequency_changed(
        "rf", 2, 4, 1, 1.2e9
    )

    assert window._qcs_front_panel_auto_apply_selection == ("rf", 2)
    assert routed == [("rf", 2, 4, 1, True)]
    window.close()
    app.processEvents()


def test_direct_lo_hydration_preserves_live_front_panel_draft():
    app = _application()
    window = gui.MainWindow()
    window._qcs_front_panel_editor_initialized = True
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    window._qcs_front_panel_dialog.show()
    app.processEvents()

    # Model a value typed into the modeless editor immediately before Apply
    # LO is clicked in the RF Outputs tab.  Hydration must not replace the
    # live widget tree before editingFinished has staged this value.
    window._qcs_front_panel.ip_address.setText("10.20.30.40")
    window._hydrate_qcs_front_panel_editor()

    assert window._qcs_front_panel.ip_address.text() == "10.20.30.40"
    window.close()
    app.processEvents()


def test_rf_duration_sweep_controls_build_sequence_axis_and_round_trip():
    app = _application()
    window = gui.MainWindow()
    panel = window._rf_ports_panel._panels[0]
    panel.setChecked(True)
    panel.gen_ch.setValue(0)
    panel.segment.setCurrentIndex(panel.segment.findData("set_0"))
    panel.duration_sweep_enabled.setChecked(True)
    panel.duration_sweep_start.setValue(0.25)
    panel.duration_sweep_stop.setValue(1.25)
    panel.duration_sweep_count.setValue(5)
    panel.segment_length_mode.setCurrentIndex(
        panel.segment_length_mode.findData("extend_by_rf_duration")
    )
    app.processEvents()

    arguments = window._experiment_run_arguments(
        require_readout=False,
        require_run_config=False,
    )
    axes = arguments["sequence"].sweep_axes
    assert len(axes) == 1
    assert axes[0].axis_kind == "rf_duration"
    assert axes[0].gen_ch == 0
    assert axes[0].start == 0.25
    assert axes[0].stop == 1.25
    assert axes[0].count == 5
    assert axes[0].segment_length_mode == "extend_by_rf_duration"
    assert window._experiment_panel.sweep_map_x.itemText(0).startswith(
        "RF gen 0"
    )

    settings = panel.settings_dict()
    restored = gui.RfPulsePortPanel(
        window._pulse[0],
        0,
        time_unit="us",
    )
    restored.load_settings(settings)
    restored_spec = restored.spec()
    assert restored_spec is not None
    assert restored_spec.duration_sweep_enabled is True
    assert restored_spec.duration_sweep_start_us == 0.25
    assert restored_spec.duration_sweep_stop_us == 1.25
    assert restored_spec.duration_sweep_count == 5
    assert restored_spec.segment_length_mode == "extend_by_rf_duration"
    window.close()


def test_rf_frequency_and_power_sweep_controls_round_trip():
    app = _application()
    window = gui.MainWindow()
    panel = window._rf_ports_panel._panels[0]
    panel.setChecked(True)
    panel.gen_ch.setValue(0)
    panel.segment.setCurrentIndex(panel.segment.findData("set_0"))
    panel.frequency_sweep_enabled.setChecked(True)
    panel.frequency_sweep_start_mhz.setValue(100.0)
    panel.frequency_sweep_stop_mhz.setValue(200.0)
    panel.frequency_sweep_count.setValue(11)
    panel.power_calibration_group.setChecked(True)
    panel.power_calibration_database_path.setText("calibration.db")
    panel.power_sweep_enabled.setChecked(True)
    panel.power_sweep_start_dbm.setValue(-40.0)
    panel.power_sweep_stop_dbm.setValue(-20.0)
    panel.power_sweep_count.setValue(5)
    app.processEvents()

    spec = panel.configured_spec()
    assert [
        (axis.axis_kind, axis.start, axis.stop, axis.count)
        for axis in spec.sweep_axes
    ] == [
        ("rf_frequency", 100.0, 200.0, 11),
        ("rf_power", -40.0, -20.0, 5),
    ]
    settings = panel.settings_dict()
    restored = gui.RfPulsePortPanel(
        window._pulse[0],
        0,
        time_unit="us",
    )
    restored.load_settings(settings)
    restored_spec = restored.configured_spec()
    assert restored_spec.frequency_sweep_enabled is True
    assert restored_spec.frequency_sweep_start_mhz == 100.0
    assert restored_spec.frequency_sweep_stop_mhz == 200.0
    assert restored_spec.frequency_sweep_count == 11
    assert restored_spec.power_sweep_enabled is True
    assert restored_spec.power_sweep_start_dbm == -40.0
    assert restored_spec.power_sweep_stop_dbm == -20.0
    assert restored_spec.power_sweep_count == 5
    assert restored_spec.power_calibration_database_path == "calibration.db"
    restored.close()
    window.close()


def test_settings_restore_rf_frequency_slice_and_ignore_removed_slice(tmp_path):
    app = _application()
    window = gui.MainWindow()
    window._add_port()
    window._sweep_specs = [
        QickSweepSpec("set_0", "awg_0", -0.1, 0.1, 3),
        QickSweepSpec("set_0", "awg_1", -0.2, 0.2, 5),
    ]

    rf_panel = window._rf_ports_panel._panels[0]
    rf_panel.setChecked(True)
    rf_panel.gen_ch.setValue(0)
    rf_panel.segment.setCurrentIndex(rf_panel.segment.findData("set_0"))
    rf_panel.frequency_sweep_enabled.setChecked(True)
    rf_panel.frequency_sweep_start_mhz.setValue(100.0)
    rf_panel.frequency_sweep_stop_mhz.setValue(200.0)
    rf_panel.frequency_sweep_count.setValue(11)
    app.processEvents()
    window._refresh_sweep_overlay()

    axis_settings = {
        "x_axis": {
            "output_name": "awg_0",
            "segment_name": "set_0",
        },
        "y_axis": {
            "output_name": "awg_1",
            "segment_name": "set_0",
        },
        "slice_axes": [
            {
                "output_name": "rf_gen_0_frequency",
                "segment_name": "set_0",
                "mode": "value",
                "value": 150.0,
            },
        ],
    }
    window._awg_sweep_plot.load_axis_selection_settings(axis_settings)
    saved_path = window._save_settings_json(tmp_path / "rf_frequency_slice")

    restored = gui.MainWindow()
    restored._load_settings_json(saved_path)
    app.processEvents()
    assert restored._awg_sweep_plot.axis_selection_settings() == axis_settings

    document = json.loads(saved_path.read_text(encoding="utf-8"))
    document["rf_outputs"][0]["frequency_sweep_enabled"] = False
    stale_path = tmp_path / "removed_rf_frequency_slice.json"
    stale_path.write_text(json.dumps(document), encoding="utf-8")
    restored._load_settings_json(stale_path)
    app.processEvents()
    assert restored._awg_sweep_plot.axis_selection_settings() == {
        "x_axis": {
            "output_name": "awg_0",
            "segment_name": "set_0",
        },
        "y_axis": {
            "output_name": "awg_1",
            "segment_name": "set_0",
        },
        "slice_axes": [],
    }

    restored.close()
    window.close()
    restored.deleteLater()
    window.deleteLater()
    app.processEvents()


def test_settings_clamp_stale_indexed_rf_anchors(tmp_path):
    app = _application()
    source = gui.MainWindow()
    pulse = PulseSequence(0.0, initial_duration_ns=1000.0)
    pulse.add_flat_ramp(100.0, 100.0, 1000.0)
    pulse.add_flat_ramp(200.0, 100.0, 1000.0)
    document = source._settings_to_dict()
    document["awg"]["outputs"] = [pulse.to_dict()]
    document["rf_outputs"][0]["enabled"] = True
    document["rf_outputs"][0]["segment_name"] = "set_4"
    document["rf_readout"]["enabled"] = True
    document["rf_readout"]["segment_name"] = "set_3"
    path = tmp_path / "stale_rf_anchors.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    restored = gui.MainWindow()
    restored._load_settings_json(path)
    app.processEvents()

    assert restored._rf_ports_panel._panels[0].segment.currentData() == "set_2"
    assert restored._rf_readout_panel.segment.currentData() == "set_2"
    upgraded = restored._settings_to_dict()
    assert upgraded["rf_outputs"][0]["segment_name"] == "set_2"
    assert upgraded["rf_readout"]["segment_name"] == "set_2"

    malformed = json.loads(json.dumps(document))
    malformed["rf_readout"]["segment_name"] = "measurement"
    with pytest.raises(
        ValueError,
        match="unknown RF readout anchor 'measurement'",
    ):
        source._decode_settings(malformed)

    restored.close()
    source.close()
    restored.deleteLater()
    source.deleteLater()
    app.processEvents()


def test_rf_output_power_calibration_applies_matching_gain_and_round_trips(
    monkeypatch,
):
    app = _application()
    window = gui.MainWindow()
    panel = window._rf_ports_panel._panels[0]
    panel.setChecked(True)
    panel.output_board_type.setCurrentText("RF_Out")
    panel.frequency_mhz.setValue(450.0)
    panel.att1_db.setValue(10.0)
    panel.att2_db.setValue(5.0)
    panel.nqz.setValue(2)
    panel.filter_type.setCurrentText("highpass")
    panel.filter_cutoff.setValue(1.5)
    panel.filter_bandwidth.setValue(0.4)
    panel.power_calibration_database_path.setText("calibration.db")
    panel.power_calibration_run_id.setValue(0)
    panel.target_output_power_dbm.setValue(-25.0)
    panel.power_calibration_group.setChecked(True)
    app.processEvents()

    calls = {}

    class FakeCalibration:
        summary = SimpleNamespace(run_id=42)

        def frequency_response_dbm(self, frequencies):
            calls["response_frequencies"] = list(frequencies)
            return np.asarray([-5.0])

        def nominal_gain_for_power(self, target, **kwargs):
            calls["nominal"] = (target, kwargs)
            return 1234

        def output_power_dbm(self, frequencies, gains, **kwargs):
            calls["predicted"] = (list(frequencies), list(gains), kwargs)
            return np.asarray([-25.0])

    class FakeCalibrationDatabase:
        def __init__(self, path):
            calls["database_path"] = path

        def output_calibration_candidates(self, *_args, **_kwargs):
            return (
                SimpleNamespace(
                    exact_match=True,
                    display_label=(
                        "Run 42 | PCB RF_Out | NQZ 2 | highpass "
                        "(fc 1.5 GHz, BW 0.4 GHz) | exact match"
                    ),
                    detail_text=(
                        "Exact PCB, frequency coverage, Nyquist zone, "
                        "and filter match."
                    ),
                    summary=SimpleNamespace(
                        run_id=42,
                        board_type="RF_Out",
                    ),
                ),
                SimpleNamespace(
                    exact_match=False,
                    display_label=(
                        "Run 41 | PCB DC_Out | NQZ 1 | lowpass "
                        "(fc 2.5 GHz, BW 1 GHz) | candidate"
                    ),
                    detail_text="PCB DC_Out != RF_Out",
                    summary=SimpleNamespace(
                        run_id=41,
                        board_type="DC_Out",
                    ),
                ),
            )

        def output_calibration(self, board_type, frequencies, **kwargs):
            calls["lookup"] = (board_type, list(frequencies), kwargs)
            return FakeCalibration()

    monkeypatch.setattr(gui, "CalibrationDatabase", FakeCalibrationDatabase)

    assert panel._apply_calibrated_output_power() == 1234
    assert panel.gain.value() == 1234
    assert panel.power_calibration_run.currentData() == 0
    assert "Run 42" in panel.power_calibration_run.currentText()
    assert calls["database_path"] == "calibration.db"
    board_type, frequencies, lookup = calls["lookup"]
    assert board_type == "RF_Out"
    assert frequencies == [450.0]
    assert lookup == {
        "run_id": 42,
        "nqz": 2,
        "output_filter_type": "highpass",
        "output_filter_cutoff_ghz": 1.5,
        "output_filter_bandwidth_ghz": 0.4,
    }
    assert calls["nominal"] == (
        -25.0,
        {
            "reference_response_dbm": -5.0,
            "output_att1_db": 10.0,
            "output_att2_db": 5.0,
        },
    )
    assert calls["predicted"] == (
        [450.0],
        [1234],
        {
            "output_att1_db": 10.0,
            "output_att2_db": 5.0,
        },
    )
    assert "Run 42" in panel.power_calibration_status.text()

    settings = panel.settings_dict()
    restored = gui.RfPulsePortPanel(
        window._pulse[0],
        0,
        time_unit="us",
    )
    restored.load_settings(settings)
    assert restored.power_calibration_group.isChecked() is True
    assert restored.power_calibration_database_path.text() == "calibration.db"
    assert restored.power_calibration_run_id.value() == 0
    assert restored.target_output_power_dbm.value() == -25.0

    panel.validate_power_calibration()
    panel.gain.setValue(1235)
    with pytest.raises(ValueError, match="click Apply calibrated gain"):
        panel.validate_power_calibration()
    restored.close()
    window.close()


def test_rf_output_power_calibration_allows_explicit_mismatch_override(
    monkeypatch,
):
    app = _application()
    window = gui.MainWindow()
    panel = window._rf_ports_panel._panels[0]
    panel.setChecked(True)
    panel.output_board_type.setCurrentText("RF_Out")
    panel.frequency_mhz.setValue(450.0)
    panel.nqz.setValue(2)
    panel.filter_type.setCurrentText("highpass")
    panel.filter_cutoff.setValue(1.5)
    panel.filter_bandwidth.setValue(0.4)
    panel.power_calibration_database_path.setText("calibration.db")
    panel.power_calibration_group.setChecked(True)
    app.processEvents()

    candidate = SimpleNamespace(
        exact_match=False,
        display_label=(
            "Run 73 | PCB DC_Out | NQZ 1 | lowpass "
            "(fc 2.5 GHz, BW 1 GHz) | candidate"
        ),
        detail_text=(
            "PCB DC_Out != RF_Out; Nyquist zone 1 != 2; "
            "filter lowpass != highpass"
        ),
        summary=SimpleNamespace(run_id=73, board_type="DC_Out"),
    )
    calls = {}

    class FakeCalibration:
        summary = SimpleNamespace(run_id=73)

        def frequency_response_dbm(self, _frequencies):
            return np.asarray([-10.0])

        def nominal_gain_for_power(self, _target, **_kwargs):
            return 4321

        def output_power_dbm(self, _frequencies, _gains, **_kwargs):
            return np.asarray([-20.0])

    class FakeCalibrationDatabase:
        def __init__(self, _path):
            pass

        def output_calibration_candidates(self, *_args, **_kwargs):
            return (candidate,)

        def output_calibration(self, board_type, frequencies, **kwargs):
            calls["lookup"] = (board_type, list(frequencies), kwargs)
            return FakeCalibration()

    monkeypatch.setattr(gui, "CalibrationDatabase", FakeCalibrationDatabase)
    panel._refresh_power_calibration_runs()
    run_index = panel.power_calibration_run.findData(73)
    assert run_index > 0
    panel.power_calibration_run.setCurrentIndex(run_index)

    assert panel._apply_calibrated_output_power() == 4321
    assert calls["lookup"] == (
        "DC_Out",
        [450.0],
        {"run_id": 73},
    )
    assert "manually overridden" in panel.power_calibration_status.text()
    assert "PCB DC_Out != RF_Out" in panel.power_calibration_status.text()
    window.close()


def test_rf_output_power_calibration_mismatch_preserves_gain(monkeypatch):
    app = _application()
    window = gui.MainWindow()
    panel = window._rf_ports_panel._panels[0]
    panel.output_board_type.setCurrentText("RF_Out")
    panel.gain.setValue(20_000)
    panel.power_calibration_database_path.setText("calibration.db")
    panel.power_calibration_group.setChecked(True)
    app.processEvents()

    class FakeCalibrationDatabase:
        def __init__(self, _path):
            pass

        def output_calibration(self, *_args, **_kwargs):
            raise LookupError(
                "no compatible calibration: Nyquist zone or filter mismatch"
            )

    warnings = []
    monkeypatch.setattr(gui, "CalibrationDatabase", FakeCalibrationDatabase)
    monkeypatch.setattr(
        gui.QtWidgets.QMessageBox,
        "warning",
        lambda *args: warnings.append(args),
    )

    assert panel._apply_calibrated_output_power() is None
    assert panel.gain.value() == 20_000
    assert "filter mismatch" in panel.power_calibration_status.text()
    assert len(warnings) == 1
    window.close()


def test_generated_qick_module_preserves_rf_duration_sweep_mode():
    pulse = PulseSequence(0.0, initial_duration_ns=10_000.0)
    rf_spec = QickRfPulseSpec(
        0,
        "set_0",
        0.5,
        1.0,
        50.0,
        12_000,
        10.0,
        12.0,
        duration_sweep_enabled=True,
        duration_sweep_start_us=1.0,
        duration_sweep_stop_us=4.0,
        duration_sweep_count=4,
        segment_length_mode="extend_by_rf_duration",
    )
    code = generate_qick_program_code(
        (pulse,),
        output_names=("awg_0",),
        awg_channels=(1,),
        tproc_mhz=300.0,
        rf_pulse_specs=(rf_spec,),
    )
    ast.parse(code)
    namespace = {}
    exec(compile(code, "<rf-duration-generated>", "exec"), namespace)

    sequence = namespace["build_sequence"]()
    assert len(sequence.sweep_axes) == 1
    axis = sequence.sweep_axes[0]
    assert axis.axis_kind == "rf_duration"
    assert axis.points == (1.0, 2.0, 3.0, 4.0)
    assert axis.segment_length_mode == "extend_by_rf_duration"
    runtime_rf = namespace["build_rf_pulses"]({
        "gens": [{"f_fabric": 300.0}],
    })
    assert len(runtime_rf) == 1
    assert runtime_rf[0].length_cycles == 300


def test_generated_qick_module_preserves_multiple_ramp_rate_sweeps():
    pulse = PulseSequence(0.0, initial_duration_ns=100.0)
    pulse.add_flat_ramp(100.0, 200.0, 200.0)
    pulse.add_flat_ramp(120.0, 150.0, -100.0)
    ramp_sweep_a = QickRampRateSweepSpec(
        "ramp_0_to_1",
        0.08,
        0.12,
        3,
    )
    ramp_sweep_b = QickRampRateSweepSpec(
        "ramp_1_to_2",
        0.10,
        0.16,
        4,
    )
    code = generate_qick_program_code(
        (pulse,),
        output_names=("awg_0",),
        awg_channels=(1,),
        fabric_mhz=300.0,
        tproc_mhz=300.0,
        sweeps=(ramp_sweep_a, ramp_sweep_b),
    )
    ast.parse(code)
    namespace = {}
    exec(compile(code, "<ramp-rate-generated>", "exec"), namespace)

    sequence = namespace["build_sequence"]()
    assert len(sequence.sweep_axes) == 2
    assert tuple(axis.segment_name for axis in sequence.sweep_axes) == (
        "ramp_0_to_1",
        "ramp_1_to_2",
    )
    assert sequence.sweep_axes[0].points == (0.08, 0.1, 0.12)
    assert sequence.sweep_axes[1].points == (0.1, 0.12, 0.14, 0.16)


def test_generated_qick_module_preserves_hold_duration_sweep():
    pulse = PulseSequence(0.0, initial_duration_ns=1000.0)
    pulse.add_flat_ramp(100.0, 2000.0, 200.0)
    hold_sweep = QickHoldDurationSweepSpec("set_1", 1.0, 5.0, 5)
    code = generate_qick_program_code(
        (pulse,),
        output_names=("awg_0",),
        awg_channels=(1,),
        fabric_mhz=300.0,
        tproc_mhz=300.0,
        sweeps=(hold_sweep,),
    )
    ast.parse(code)
    namespace = {}
    exec(compile(code, "<hold-duration-generated>", "exec"), namespace)

    sequence = namespace["build_sequence"]()
    assert len(sequence.sweep_axes) == 1
    axis = sequence.sweep_axes[0]
    assert axis.axis_kind == "hold_duration"
    assert axis.segment_name == "set_1"
    assert axis.points == (1.0, 2.0, 3.0, 4.0, 5.0)


def test_rf_readout_panel_builds_analog_input_and_ddr_settings():
    app = _application()
    window = gui.MainWindow()
    window._experiment_panel.set_execution_backend(
        gui.EXECUTION_BACKEND_QICK
    )
    panel = window._rf_readout_panel
    panel.setChecked(True)
    panel.ro_ch.setValue(2)
    panel.delay.setValue(1.5)
    panel.samples.setValue(128)
    panel.frequency_mhz.setValue(75.0)
    panel.attenuation_db.setValue(21.0)
    panel.filter_type.setCurrentText("lowpass")
    panel.filter_cutoff.setValue(2.5)
    panel.filter_bandwidth.setValue(0.75)
    app.processEvents()

    calibration_row, _role = panel.layout().getWidgetPosition(
        panel.dc_voltage_calibration_enabled
    )
    assert calibration_row >= 0
    assert panel.measurement_unit.currentData() == "adc"

    spec = panel.spec()
    assert spec == QickDdrReadoutSpec(
        ro_ch=2,
        segment_name="set_0",
        delay_us=1.5,
        samples_per_trigger=128,
        readout_frequency_mhz=75.0,
        margin_input_samples=1024,
        attenuation_db=21.0,
        filter_type="lowpass",
        filter_cutoff=2.5,
        filter_bandwidth=0.75,
        measurement_representation="adc",
    )
    window.close()


def test_dc_measure_mode_converts_iq_and_is_available_only_for_dc_input():
    raw_iq = np.asarray([[2.0, -4.0], [6.0, 8.0]])
    np.testing.assert_array_equal(adc_iq_to_voltage(raw_iq), raw_iq)
    np.testing.assert_allclose(
        dc_iq_to_current(raw_iq, gain_v_per_a=2.0),
        [[1.0, -2.0], [3.0, 4.0]],
    )
    with pytest.raises(ValueError, match="must be positive"):
        dc_iq_to_current(raw_iq, gain_v_per_a=0.0)
    with pytest.raises(ValueError, match="requires the DC_In"):
        QickDdrReadoutSpec(
            0,
            "set_0",
            0.0,
            8,
            input_board_type="RF_In",
            dc_measure_mode=True,
        )

    app = _application()
    window = gui.MainWindow()
    panel = window._rf_readout_panel
    panel.setChecked(True)
    panel.input_board_type.setCurrentText("DC_In")
    panel.measurement_unit.setCurrentIndex(
        panel.measurement_unit.findData("current")
    )
    panel.dc_measure_gain_v_per_a.setValue(2.0)
    app.processEvents()

    spec = panel.spec()
    assert spec is not None
    assert spec.input_board_type == "DC_In"
    assert spec.dc_measure_mode is True
    assert spec.measurement_representation == "current"
    assert spec.dc_measure_gain_v_per_a == 2.0
    assert spec.measurement_unit == "A"
    assert panel.dc_measure_gain_v_per_a.isEnabled() is True

    panel.input_board_type.setCurrentText("RF_In")
    app.processEvents()
    assert panel.dc_measure_mode.isChecked() is False
    assert panel.dc_measure_mode.isEnabled() is False
    assert panel.measurement_unit.currentData() == "adc"
    assert panel.measurement_unit.isEnabled() is False
    assert panel.dc_measure_gain_v_per_a.isEnabled() is False
    window.close()


def test_awg_tuning_rf_readout_exposes_calibration_and_unit_selection(tmp_path):
    app = _application()
    window = gui.MainWindow()
    window._experiment_panel.set_execution_backend(
        gui.EXECUTION_BACKEND_QICK
    )
    panel = window._rf_readout_panel
    panel.setChecked(True)
    panel.input_board_type.setCurrentText("DC_In")
    panel.measurement_unit.setCurrentIndex(
        panel.measurement_unit.findData("voltage")
    )
    panel.dc_voltage_calibration_enabled.setChecked(True)
    panel.dc_voltage_calibration_path.setText(str(tmp_path / "calibration.db"))
    panel.dc_voltage_calibration_run_id.setValue(17)
    app.processEvents()

    spec = panel.spec()
    assert spec is not None
    assert spec.effective_measurement_representation == "voltage"
    assert spec.measurement_unit == "V"
    assert spec.dc_voltage_calibration_enabled is True
    assert spec.dc_voltage_calibration_run_id == 17
    assert panel.dc_voltage_calibration_path.isHidden() is False
    settings = panel.settings_dict()
    assert settings["measurement_representation"] == "voltage"

    restored = gui.RfReadoutPanel(window._pulse[0], time_unit="us")
    restored.load_settings(settings)
    assert restored.measurement_unit.currentData() == "voltage"
    assert restored.dc_voltage_calibration_enabled.isChecked() is True
    assert restored.dc_voltage_calibration_run_id.value() == 17
    restored.close()
    window.close()


def test_generated_module_supports_multiple_rf_outputs_and_readout_chain():
    pulse = PulseSequence(100.0, initial_duration_ns=1000.0)
    pulse.add_flat_ramp(1000.0, 5000.0, 200.0)
    rf_specs = (
        QickRfPulseSpec(
            0, "set_1", 0.0, 1.0, 50.0, 12000, 10.0, 12.0,
            filter_type="lowpass", filter_cutoff=2.25, filter_bandwidth=0.75,
        ),
        QickRfPulseSpec(2, "set_1", 1.0, 1.0, 75.0, 8000, 8.0, 9.0),
    )
    readout = QickDdrReadoutSpec(
        0,
        "set_1",
        0.0,
        8,
        50.0,
        attenuation_db=21.0,
        filter_type="lowpass",
        filter_cutoff=2.5,
    )
    code = generate_qick_program_code(
        (pulse,),
        output_names=("awg_0",),
        awg_channels=(1,),
        tproc_mhz=300.0,
        rf_pulse_specs=rf_specs,
        ddr_readout_spec=readout,
        bias_t_compensation_enabled=True,
        bias_t_compensation_voltage_mv=125.0,
        bias_t_compensation_mode="fixed_time",
        bias_t_compensation_duration_us=2.5,
    )
    ast.parse(code)
    namespace = {}
    exec(compile(code, "<multi-rf-generated>", "exec"), namespace)
    assert namespace["TPROC_MHZ"] == 300.0
    assert namespace["BIAS_T_COMPENSATION_ENABLED"] is True
    assert namespace["BIAS_T_COMPENSATION_VOLTAGE_MV"] == 125.0
    assert namespace["BIAS_T_COMPENSATION_MODE"] == "fixed_time"
    assert namespace["BIAS_T_COMPENSATION_DURATION_CYCLES"] == 750
    bias_t_config = namespace["build_sequence"]().bias_t_compensation
    assert bias_t_config.amplitude == 0.15625
    assert bias_t_config.mode == "fixed_time"
    assert bias_t_config.fixed_duration_cycles == 750
    stale_hwh_soccfg = {
        "tprocs": [{"f_time": 400.0}],
        "gens": [
            {"f_fabric": 300.0},
            {"f_fabric": 300.0},
            {"f_fabric": 300.0},
        ],
    }
    generated_rf = namespace["build_rf_pulses"](stale_hwh_soccfg)
    assert generated_rf[1].delay_tproc_cycles == 300

    class FakeSoc:
        def __init__(self):
            self.calls = []

        def rfb_set_gen_rf(self, gen_ch, att1, att2):
            self.calls.append(("output", gen_ch, att1, att2))
            return att1, att2

        def rfb_set_gen_filter(self, gen_ch, **kwargs):
            self.calls.append(("output_filter", gen_ch, kwargs))

        def rfb_set_ro_rf(self, ro_ch, attenuation):
            self.calls.append(("readout", ro_ch, attenuation))
            return attenuation

        def rfb_set_ro_filter(self, ro_ch, **kwargs):
            self.calls.append(("filter", ro_ch, kwargs))

    soc = FakeSoc()
    assert namespace["configure_rf_chain"](soc) == (
        (10.0, 12.0),
        (8.0, 9.0),
    )
    assert ("output_filter", 0, {
        "fc": 2.25, "bw": 0.75, "ftype": "lowpass"
    }) in soc.calls
    assert ("output_filter", 2, {
        "fc": 2.5, "bw": 1.0, "ftype": "bypass"
    }) in soc.calls
    assert namespace["configure_readout_chain"](soc) == 21.0
    assert soc.calls[-1] == (
        "filter",
        0,
        {"fc": 2.5, "bw": 1.0, "ftype": "lowpass"},
    )


def test_settings_json_round_trip_restores_complete_gui_state(tmp_path):
    app = _application()
    window = gui.MainWindow()
    window._add_segment(500.0, 2000.0, -250.0)
    window._add_segment(750.0, 1500.0, 125.0)
    window._add_port()
    window._pulse[1].v[:] = [300.0, 300.0, 450.0, 450.0, -125.0, -125.0]
    window._cross_capacitance = np.asarray(
        ((1.0, 0.2), (-0.15, 1.0)), dtype=float
    )
    window._sweep_specs = [
        QickRampRateSweepSpec("ramp_0_to_1", 0.08, 0.12, 3),
        QickRampRateSweepSpec("ramp_1_to_2", 0.10, 0.16, 4),
        QickHoldDurationSweepSpec("set_1", 1.0, 5.0, 5),
        QickSweepSpec("set_2", "awg_0", -0.4, 0.6, 7),
    ]
    window._qick_fabric_mhz = 300.0
    window._qick_full_scale_mv = 2000.0
    window._qick_awg_channels = (1, 3)
    window._qick_repetitions_per_sweep = 11
    window._grid_configured = True
    window._set_grid_settings(
        time_step_ns=250.0,
        voltage_step_mv=25.0,
        snap_enabled=True,
        visible=False,
    )
    window._time_unit_combo.setCurrentText("ms")
    window._set_voltage_view("physical")
    window._voltage_view_actions["physical"].setChecked(True)
    window._port_select(1)
    window._control_tabs.setCurrentWidget(window._awg_tuning_page)
    window._awg_tuning_tabs.setCurrentWidget(window._rf_readout_panel)

    rf_panel = window._rf_ports_panel._panels[0]
    rf_panel.setChecked(True)
    rf_panel.gen_ch.setValue(0)
    rf_panel.segment.setCurrentIndex(rf_panel.segment.findData("set_1"))
    rf_panel.delay.setValue(0.00025)
    rf_panel.duration.setValue(0.0015)
    rf_panel.frequency_mhz.setValue(123.5)
    rf_panel.gain.setValue(12345)
    rf_panel.att1_db.setValue(7.25)
    rf_panel.att2_db.setValue(8.5)
    rf_panel.filter_type.setCurrentText("highpass")
    rf_panel.filter_cutoff.setValue(1.75)
    rf_panel.filter_bandwidth.setValue(0.625)
    rf_panel.qcs_power_calibration_group.setChecked(True)
    rf_panel.qcs_power_calibration_database_path.setText(
        str(tmp_path / "qcs_m5300_power.db")
    )
    rf_panel.qcs_power_calibration_run_id.setValue(29)
    rf_panel.qcs_target_output_power_dbm.setValue(-18.25)
    window._rf_ports_panel.add_port()
    disabled_rf_panel = window._rf_ports_panel._panels[1]
    disabled_rf_panel.gen_ch.setValue(4)
    disabled_rf_panel.frequency_mhz.setValue(80.0)

    readout = window._rf_readout_panel
    readout.setChecked(True)
    readout.ro_ch.setValue(2)
    readout.segment.setCurrentIndex(readout.segment.findData("set_1"))
    readout.delay.setValue(0.0005)
    readout.samples.setValue(256)
    readout.frequency_mhz.setValue(42.0)
    readout.attenuation_db.setValue(21.25)
    readout.filter_type.setCurrentText("bandpass")
    readout.filter_cutoff.setValue(2.25)
    readout.filter_bandwidth.setValue(0.5)
    readout.margin_samples.setValue(2048)
    readout.force_overwrite.setChecked(True)
    readout.input_board_type.setCurrentText("DC_In")
    readout.dc_gain_db.setValue(12.0)
    readout.dc_measure_mode.setChecked(True)
    readout.dc_measure_gain_v_per_a.setValue(1.0e6)
    window._stability_panel.apply_path_settings(
        {
            "output_ch": 2,
            "output_att1_db": 12.25,
            "readout_ch": 1,
            "input_board_type": "RF_In",
            "readout_attenuation_db": 9.5,
        }
    )
    window._stability_panel.modulation_frequency_mhz.setValue(211.0)
    window._stability_panel.trace_samples.setValue(321)
    window._stability_panel.qcs_power_calibration_group.setChecked(True)
    window._stability_panel.qcs_power_calibration_database_path.setText(
        str(tmp_path / "stability_m5300_power.db")
    )
    window._stability_panel.qcs_power_calibration_run_id.setValue(31)
    window._stability_panel.qcs_target_output_power_dbm.setValue(-22.75)
    window._stability_panel.bias_t_group.setChecked(True)
    window._stability_panel.bias_t_type.setCurrentIndex(
        window._stability_panel.bias_t_type.findData("filter")
    )
    window._stability_panel.bias_t_filter_tau_us.setValue(42.0)
    experiment = window._experiment_panel
    experiment.qick_host.setText("192.0.2.44")
    experiment.ns_port.setValue(9999)
    experiment.proxy_name.setText("labqick")
    experiment.database_path.setText(str(tmp_path / "experiment.db"))
    experiment.experiment_name.setText("Fine tune sweep")
    experiment.sample_name.setText("device A")
    experiment.notes.setPlainText("JSON round-trip notes")
    experiment.tproc_mhz.setValue(275.0)
    experiment.bias_t_group.setChecked(True)
    experiment.bias_t_compensation_mv.setValue(125.0)
    experiment.bias_t_mode.setCurrentIndex(
        experiment.bias_t_mode.findData("fixed_time")
    )
    experiment.bias_t_duration_us.setValue(2.5)
    experiment.set_compile_validation_mode("full")
    app.processEvents()

    expected = window._settings_to_dict()
    saved_path = window._save_settings_json(tmp_path / "complete_experiment")
    assert saved_path.suffix == ".json"
    document = json.loads(saved_path.read_text(encoding="utf-8"))
    assert document["schema"] == gui.SETTINGS_SCHEMA
    assert document["version"] == gui.SETTINGS_VERSION
    assert document["qick"]["tproc_mhz"] == 275.0
    assert document["qick"]["compile_validation_mode"] == "full"
    assert document["qick"]["bias_t_compensation"] == {
        "enabled": True,
        "type": "dc",
        "mode": "fixed_time",
        "voltage_mv": 125.0,
        "duration_us": 2.5,
        "filter_tau_us": 100.0,
    }
    assert len(document["awg"]["outputs"]) == 2
    assert document["awg"]["sweeps"][0] == {
        "axis_kind": "ramp_duration",
        "segment_name": "ramp_0_to_1",
        "output_name": "all_awg_outputs",
        "start": 0.08,
        "stop": 0.12,
        "count": 3,
    }
    assert document["awg"]["sweeps"][1] == {
        "axis_kind": "ramp_duration",
        "segment_name": "ramp_1_to_2",
        "output_name": "all_awg_outputs",
        "start": 0.10,
        "stop": 0.16,
        "count": 4,
    }
    assert document["awg"]["sweeps"][2] == {
        "axis_kind": "hold_duration",
        "segment_name": "set_1",
        "output_name": "all_awg_outputs",
        "start": 1.0,
        "stop": 5.0,
        "count": 5,
    }
    assert len(document["rf_outputs"]) == 2
    assert document["rf_outputs"][0]["filter_type"] == "highpass"
    assert document["rf_outputs"][0]["filter_cutoff"] == 1.75
    assert document["rf_outputs"][0]["filter_bandwidth"] == 0.625
    assert document["rf_outputs"][0][
        "qcs_power_calibration_enabled"
    ] is True
    assert document["rf_outputs"][0][
        "qcs_power_calibration_run_id"
    ] == 29
    assert document["rf_outputs"][0][
        "qcs_target_output_power_dbm"
    ] == pytest.approx(-18.25)
    assert document["rf_readout"]["input_board_type"] == "DC_In"
    assert document["rf_readout"]["dc_measure_mode"] is True
    assert document["rf_readout"]["dc_measure_gain_v_per_a"] == 1.0e6
    assert document["stability_diagram"]["rf_path"]["output_ch"] == 2
    assert document["stability_diagram"]["rf_path"]["output_att1_db"] == 12.25
    assert document["stability_diagram"]["rf_path"]["readout_ch"] == 1
    assert document["stability_diagram"]["rf_path"]["readout_attenuation_db"] == 9.5
    assert document["stability_diagram"]["modulation_frequency_mhz"] == 211.0
    assert "rf_outputs" not in document["stability_diagram"]
    assert "rf_readout" not in document["stability_diagram"]
    assert document["stability_diagram"]["trace_samples_per_point"] == 321
    assert document["stability_diagram"][
        "qcs_power_calibration_enabled"
    ] is True
    assert document["stability_diagram"][
        "qcs_power_calibration_run_id"
    ] == 31
    assert document["stability_diagram"][
        "qcs_target_output_power_dbm"
    ] == pytest.approx(-22.75)
    assert document["stability_diagram"]["bias_t_compensation"] == {
        "enabled": True,
        "type": "filter",
        "mode": "fixed_voltage",
        "voltage_mv": 80.0,
        "duration_us": 1.0,
        "filter_tau_us": 42.0,
    }

    restored = gui.MainWindow()
    restored._load_settings_json(saved_path)
    app.processEvents()
    assert restored._settings_to_dict() == expected
    assert restored._ddr_readout_spec == readout.spec()
    assert len(restored._rf_pulse_specs) == 1
    restored_stability_path = restored._stability_panel.front_panel_values()
    assert restored_stability_path["output_ch"] == 2
    assert restored_stability_path["output_att1_db"] == 12.25
    assert restored_stability_path["readout_ch"] == 1
    assert restored_stability_path["readout_attenuation_db"] == 9.5
    assert restored._stability_panel.modulation_frequency_mhz.value() == 211.0
    assert restored._stability_panel.trace_samples.value() == 321
    assert (
        restored._stability_panel.qcs_power_calibration_group.isChecked()
        is True
    )
    assert (
        restored._stability_panel.qcs_power_calibration_run_id.value()
        == 31
    )
    assert (
        restored._stability_panel.qcs_target_output_power_dbm.value()
        == pytest.approx(-22.75)
    )
    assert restored._stability_panel.bias_t_group.isChecked() is True
    assert restored._stability_panel.bias_t_type.currentData() == "filter"
    assert restored._stability_panel.bias_t_filter_tau_us.value() == 42.0
    assert restored._experiment_panel.qick_host.text() == "192.0.2.44"
    assert restored._experiment_panel.tproc_mhz.value() == 275.0
    assert restored._experiment_panel.bias_t_group.isChecked() is True
    assert restored._experiment_panel.bias_t_type.currentData() == "dc"
    assert restored._experiment_panel.bias_t_mode.currentData() == "fixed_time"
    assert restored._experiment_panel.bias_t_compensation_mv.value() == 125.0
    assert restored._experiment_panel.bias_t_duration_us.value() == 2.5
    assert (
        restored._experiment_panel.compile_validation_mode.currentData()
        == "full"
    )
    assert restored._control_tabs.currentWidget() is restored._awg_tuning_page
    assert restored._awg_tuning_tabs.currentWidget() is restored._rf_readout_panel
    assert restored._experiment_panel.database_path.text().endswith("experiment.db")
    window.close()
    restored.close()


def test_experiment_panel_builds_hardware_run_snapshot(tmp_path):
    app = _application()
    window = gui.MainWindow()
    window._experiment_panel.set_execution_backend(
        gui.EXECUTION_BACKEND_QICK
    )
    window._experiment_panel.database_path.setText(str(tmp_path / "run.db"))
    window._experiment_panel.experiment_name.setText("GUI run")
    window._experiment_panel.sample_name.setText("sample 1")
    window._rf_readout_panel.setChecked(True)
    window._rf_readout_panel.samples.setValue(32)
    window._experiment_panel.bias_t_group.setChecked(True)
    window._experiment_panel.bias_t_compensation_mv.setValue(200.0)

    arguments = window._experiment_run_arguments()
    assert arguments["connection_config"].host == gui.DEFAULT_QICK_HOST
    assert arguments["run_config"].resolved_database_path == (tmp_path / "run.db")
    assert arguments["awg_channels"] == (1,)
    assert arguments["compile_validation_mode"] == "boundary"
    assert arguments["readout_spec"].samples_per_trigger == 32
    assert arguments["gui_settings"]["qick"]["tproc_mhz"] == 300.0
    assert arguments["sequence"].bias_t_compensation.amplitude == 0.25
    assert "waveforms" not in arguments["gui_settings"]
    assert arguments["gui_settings"]["awg"]["outputs"][0]["time_ns"] == [
        0.0,
        1000.0,
    ]
    window._experiment_panel.set_running(True, "Starting")
    window._experiment_panel.update_progress(47, "Saving QCoDeS IQ rows")
    assert window._experiment_panel.progress.minimum() == 0
    assert window._experiment_panel.progress.maximum() == 100
    assert window._experiment_panel.progress.value() == 47
    assert "47%" in window._experiment_panel.run_status.text()
    app.processEvents()
    window.close()


def test_stability_tab_builds_two_axis_hardware_sweep_without_database():
    app = _application()
    window = gui.MainWindow()
    window._experiment_panel.set_execution_backend(
        gui.EXECUTION_BACKEND_QICK
    )
    window._add_port()
    window._rf_readout_panel.setChecked(True)
    window._rf_readout_panel.samples.setValue(16)
    window._rf_readout_panel.input_board_type.setCurrentText("RF_In")
    window._rf_readout_panel.attenuation_db.setValue(5.0)
    window._stability_panel.apply_path_settings(
        {
            "input_board_type": "DC_In",
            "readout_ch": 0,
            "output_board_type": "DC_Out",
            "output_ch": 0,
        }
    )
    window._stability_panel.dc_measure_mode.setChecked(True)
    window._stability_panel.dc_measure_gain_v_per_a.setValue(2.0e6)
    window._stability_panel.dc_calibration_group.setChecked(True)
    window._stability_panel.dc_calibration_path.setText("dc_calibration.db")
    window._stability_panel.dc_calibration_run_id.setValue(7)
    window._stability_panel.modulation_frequency_mhz.setValue(0.0)
    window._stability_panel.trace_samples.setValue(96)
    window._stability_panel.x_axis.start_mv.setValue(-200.0)
    window._stability_panel.x_axis.stop_mv.setValue(100.0)
    window._stability_panel.x_axis.points.setValue(5)
    window._stability_panel.y_axis.start_mv.setValue(-50.0)
    window._stability_panel.y_axis.stop_mv.setValue(75.0)
    window._stability_panel.y_axis.points.setValue(3)
    window._stability_panel.repetitions.setValue(4)
    window._experiment_panel.bias_t_group.setChecked(True)
    window._experiment_panel.bias_t_type.setCurrentIndex(
        window._experiment_panel.bias_t_type.findData("filter")
    )
    window._experiment_panel.bias_t_filter_tau_us.setValue(999.0)
    window._stability_panel.bias_t_group.setChecked(True)
    window._stability_panel.bias_t_mode.setCurrentIndex(
        window._stability_panel.bias_t_mode.findData("fixed_time")
    )
    window._stability_panel.bias_t_duration_us.setValue(2.5)
    app.processEvents()

    assert window._stability_panel.dc_measure_mode.isChecked() is True
    assert window._stability_panel.dc_measure_mode.isEnabled() is True
    assert window._stability_panel.dc_measure_gain_v_per_a.value() == 2.0e6
    assert window._stability_panel.dc_calibration_group.isChecked() is True
    assert (
        window._stability_panel.dc_calibration_path.text()
        == "dc_calibration.db"
    )
    assert window._stability_panel.dc_calibration_run_id.value() == 7
    window._stability_panel.dc_measure_gain_v_per_a.setValue(3.0e6)
    window._stability_panel.dc_calibration_run_id.setValue(8)
    app.processEvents()
    assert window._rf_readout_panel.input_board_type.currentText() == "RF_In"
    assert window._rf_readout_panel.attenuation_db.value() == 5.0
    assert window._rf_readout_panel.dc_measure_mode.isChecked() is False

    arguments = window._stability_run_arguments(save=False)

    assert arguments["run_config"] is None
    assert arguments["gui_settings"] is None
    assert arguments["repetitions_per_sweep"] == 4
    assert arguments["sequence"].sweep_shape == (5, 3)
    assert [axis.output_name for axis in arguments["sequence"].sweep_axes] == [
        "awg_0",
        "awg_1",
    ]
    assert arguments["sequence"].sweep_point_count == 15
    assert arguments["sequence"].bias_t_compensation.compensation_type == "dc"
    assert arguments["sequence"].bias_t_compensation.mode == "fixed_time"
    assert arguments["sequence"].bias_t_compensation.fixed_duration_cycles == 750
    assert arguments["readout_spec"].samples_per_trigger == 96
    assert window._rf_readout_panel.samples.value() == 16
    assert arguments["stability_config"].trace_samples_per_point == 96
    assert arguments["readout_spec"].dc_measure_mode is True
    assert arguments["readout_spec"].dc_measure_gain_v_per_a == 3.0e6
    assert arguments["readout_spec"].dc_voltage_calibration_enabled is True
    assert arguments["readout_spec"].dc_voltage_calibration_run_id == 8
    assert arguments["readout_spec"].readout_frequency_mhz == 0.0
    assert arguments["rf_specs"][0].frequency_mhz == 0.0
    assert arguments["rf_specs"][0].output_board_type == "DC_Out"

    window._stability_panel.bias_t_type.setCurrentIndex(
        window._stability_panel.bias_t_type.findData("filter")
    )
    window._stability_panel.bias_t_filter_tau_us.setValue(25.0)
    filter_arguments = window._stability_run_arguments(save=False)
    filter_compensation = filter_arguments["sequence"].bias_t_compensation
    assert filter_compensation.compensation_type == "filter"
    assert filter_compensation.tau_cycles == 7_500.0
    window.close()


def test_stability_run_arguments_use_identified_50ksps_timing():
    app = _application()
    window = gui.MainWindow()
    window._experiment_panel.set_execution_backend(
        gui.EXECUTION_BACKEND_QICK
    )
    window._add_port()
    window._qick_configuration = SimpleNamespace(
        fir_sample_rate_hz=50_000.0,
        fir_trigger_delay_us=1000.0,
    )
    window._stability_panel.trace_samples.setValue(100)
    window._stability_panel.settle_time_us.setValue(25.0)

    arguments = window._stability_run_arguments(save=False)

    expected_hold_us = (
        25.0
        + 100 * 20.0
        + DEFAULT_STABILITY_POINT_GUARD_US
    )
    assert arguments["sequence"].segments[0].duration_cycles == int(
        np.ceil(expected_hold_us * 300.0)
    )
    assert arguments["rf_specs"][0].duration_us == 2000.0
    assert arguments["readout_spec"].fpga_trigger_delay_samples is None
    assert arguments["readout_spec"].fpga_trigger_delay_us is None
    assert arguments["stability_fabric_mhz"] == 300.0
    app.processEvents()
    window.close()


def test_qcs_stability_arguments_build_native_hardware_sweep(
    tmp_path,
    monkeypatch,
):
    app = _application()
    window = gui.MainWindow()
    window._add_port()
    mapper_path = tmp_path / "stability_mapper.qcs"
    mapper_path.write_bytes(b"offline mapper placeholder")
    connection = gui.QcsConnectionConfig(
        mapper_path=str(mapper_path),
        dc_channel_names=("dc_x", "dc_y"),
        dc_full_scale_v=2.5,
        rf_channel_names={7: "rf_drive"},
        acquisition_channel_name="digitizer",
        hw_demod=False,
        blocking=True,
    )
    configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("dc_x", "dc_y"),
        {7: "rf_drive"},
        "digitizer",
    )
    window._stability_panel.set_qcs_front_panel_configuration(
        configuration
    )
    window._stability_panel.set_hardware_backend(
        gui.EXECUTION_BACKEND_QCS
    )
    window._stability_panel.repetitions.setValue(4)
    window._stability_panel.x_axis.points.setValue(3)
    window._stability_panel.y_axis.points.setValue(2)
    window._stability_panel.x_axis.start_mv.setValue(-1000.0)
    window._stability_panel.x_axis.stop_mv.setValue(1000.0)
    window._stability_panel.qcs_integration_duration_us.setValue(0.020)
    window._stability_panel.settle_time_us.setValue(25.0)
    window._stability_panel.modulation_frequency_mhz.setValue(125.0)
    window._stability_panel.qcs_modulation_amplitude.setValue(0.25)
    calibration_path = tmp_path / "m5300_power_calibration.db"
    calibration_path.write_bytes(b"offline calibration placeholder")
    window._stability_panel.qcs_power_calibration_database_path.setText(
        str(calibration_path)
    )
    window._stability_panel.qcs_power_calibration_run_id.setValue(29)
    window._stability_panel.qcs_target_output_power_dbm.setValue(-18.25)
    window._stability_panel.qcs_power_calibration_group.setChecked(True)
    window._stability_panel.bias_t_group.setChecked(True)
    window._experiment_panel.qcs_sample_rate_hz.setValue(2.0e6)
    assert window._experiment_panel.qcs_sample_rate_hz.value() == pytest.approx(
        4.8e9
    )
    window._experiment_panel.qick_host.clear()
    window._experiment_panel.proxy_name.clear()
    window._experiment_panel.awg_channels.setText("invalid dormant value")
    window._experiment_panel.qcs_dc_channel_names.setText(
        "duplicate, duplicate"
    )
    window._experiment_panel.qcs_rf_channel_names.setText("invalid mapping")
    monkeypatch.setattr(
        gui,
        "qcs_workflow_mapper_output_path",
        lambda _configuration, _workflow: mapper_path,
    )
    window._rf_readout_panel.qcs_trace_radio.click()
    app.processEvents()
    assert window._rf_readout_panel.qcs_trace_radio.isChecked() is True
    assert window._experiment_panel.qcs_hw_demod.isChecked() is False
    arguments = window._stability_run_arguments(save=False)

    assert connection.hw_demod is False
    assert arguments["connection_config"].hw_demod is True
    assert arguments["connection_config"].mapper_path == connection.mapper_path
    assert arguments["connection_config"].dc_channel_names == (
        "dc_x",
        "dc_y",
    )
    assert arguments["connection_config"].rf_channel_names == {
        7: "rf_drive"
    }
    assert arguments["mapper_configuration"] == (
        window._stability_panel.qcs_mapper_configuration()
    )
    assert window._rf_readout_panel.qcs_trace_radio.isChecked() is True
    assert window._experiment_panel.qcs_hw_demod.isChecked() is False
    assert arguments["repetitions_per_point"] == 4
    assert arguments["full_scale_mv"] == pytest.approx(2500.0)
    assert arguments["sequence"].sweep_shape == (3, 2)
    np.testing.assert_allclose(
        arguments["sequence"].sweep_axes[0].points,
        [-0.4, 0.0, 0.4],
    )
    compensation = arguments["sequence"].bias_t_compensation
    assert compensation is not None
    assert compensation.compensation_type == "dc"
    assert compensation.mode == "fixed_time"
    assert compensation.fixed_duration_cycles == 300
    assert window._stability_panel.bias_t_group.isChecked() is True
    assert arguments["rf_pulses"][0].gen_ch == 7
    assert arguments["rf_pulses"][0].amplitude == pytest.approx(
        window._stability_panel.qcs_modulation_amplitude.value()
    )
    assert arguments["rf_pulses"][0].power_calibration == (
        gui.QcsRfPowerCalibrationConfig(
            database_path=str(calibration_path),
            run_id=29,
            target_power_dbm=-18.25,
        )
    )
    assert window._stability_panel.qcs_modulation_amplitude.isEnabled() is False
    assert arguments["rf_pulses"][0].frequency_hz == pytest.approx(125e6)
    assert arguments["acquisition"].duration_s == pytest.approx(20e-9)
    assert arguments["acquisition"].sample_rate_hz == pytest.approx(4.8e9)
    expected_readout_delay_s = 25e-6 + gui.QCS_STABILITY_DC_RAMP_S
    assert arguments["acquisition"].pre_delay_s == pytest.approx(
        expected_readout_delay_s
    )
    assert arguments["rf_pulses"][0].delay_s == pytest.approx(
        expected_readout_delay_s
    )
    expected_target_us = (
        25.0
        + 0.020
        + DEFAULT_STABILITY_POINT_GUARD_US
        + gui.QCS_STABILITY_DC_EDGE_PADDING_S * 1.0e6
    )
    assert arguments["sequence"].segments[0].duration_cycles == int(
        np.ceil(expected_target_us * 300.0)
    )
    assert arguments["acquisition"].sample_count == 96
    assert arguments["readout_spec"] is None

    window._stability_panel.qcs_integration_duration_us.setValue(100.0)
    app.processEvents()
    long_arguments = window._stability_run_arguments(save=False)
    assert long_arguments["acquisition"].sample_count == 480_000
    assert long_arguments["acquisition"].duration_s == pytest.approx(100e-6)
    assert long_arguments["rf_pulses"][0].duration_s == pytest.approx(100e-6)
    expected_long_target_us = (
        25.0
        + 100.0
        + 0.280
        + DEFAULT_STABILITY_POINT_GUARD_US
        + gui.QCS_STABILITY_DC_EDGE_PADDING_S * 1.0e6
    )
    assert long_arguments["sequence"].segments[0].duration_cycles == int(
        np.ceil(expected_long_target_us * 300.0)
    )
    integration_note = window._stability_panel.qcs_integration_note.text()
    assert "15 IntegrationFilter segment(s)" in integration_note
    assert "0.28 us total dead time" in integration_note
    assert "100.28 us" in integration_note

    window._stability_panel.qcs_integration_duration_us.setValue(100_000.0)
    app.processEvents()
    aggregate_arguments = window._stability_run_arguments(save=False)
    assert aggregate_arguments["acquisition"].sample_count == 480_000_000
    assert aggregate_arguments["acquisition"].duration_s == pytest.approx(0.1)
    # Only one bounded pass is scheduled at a time; the backend reuses this
    # same 100 us RF/DC program and sample-weights all 1,000 pass results.
    assert aggregate_arguments["rf_pulses"][0].duration_s == pytest.approx(
        100e-6
    )
    assert aggregate_arguments["sequence"].segments[0].duration_cycles == int(
        np.ceil(expected_long_target_us * 300.0)
    )
    aggregate_note = window._stability_panel.qcs_integration_note.text()
    assert "1,000 bounded QCS passes" in aggregate_note
    assert "100000 us total I/Q averaging time" in aggregate_note

    saved_arguments = window._stability_run_arguments(save=True)
    assert saved_arguments["gui_settings"]["qcs"][
        "dc_channel_names"
    ] == ["dc_x", "dc_y"]
    assert saved_arguments["gui_settings"]["qcs"][
        "rf_channel_names"
    ] == {"7": "rf_drive"}
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_qcs_sparameter_uses_single_iq_without_changing_awg_tuning_mode(
    tmp_path,
    monkeypatch,
):
    app = _application()
    window = gui.MainWindow()
    mapper_path = tmp_path / "sparameter_mapper.qcs"
    mapper_path.write_bytes(b"offline mapper placeholder")
    connection = gui.QcsConnectionConfig(
        mapper_path=str(mapper_path),
        dc_channel_names=("dc_gate",),
        dc_full_scale_v=2.5,
        rf_channel_names={7: "rf_drive"},
        acquisition_channel_name="digitizer",
        hw_demod=False,
        blocking=True,
    )
    configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("dc_gate",),
        {7: "rf_drive"},
        "digitizer",
    )
    window._sparameter_panel.set_qcs_front_panel_configuration(configuration)
    window._sparameter_panel.set_hardware_backend(
        gui.EXECUTION_BACKEND_QCS
    )
    monkeypatch.setattr(
        window._experiment_panel,
        "qcs_connection_values",
        lambda _output_count: connection,
    )
    monkeypatch.setattr(
        window._experiment_panel,
        "run_config_values",
        lambda **_kwargs: None,
    )
    window._rf_readout_panel.qcs_trace_radio.click()
    app.processEvents()

    arguments = window._sparameter_run_arguments()

    assert connection.hw_demod is False
    assert arguments["connection_config"].hw_demod is True
    assert arguments["connection_config"].mapper_path == connection.mapper_path
    assert window._rf_readout_panel.qcs_trace_radio.isChecked() is True
    assert window._experiment_panel.qcs_hw_demod.isChecked() is False
    app.processEvents()
    window.close()


def test_experiment_panel_exposes_show_program_action():
    app = _application()
    panel = gui.ExperimentPanel(
        fabric_mhz=300.0,
        tproc_mhz=300.0,
        full_scale_mv=800.0,
        awg_channels=(1,),
        repetitions=1,
    )
    panel.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    emitted = []
    panel.show_program_requested.connect(lambda: emitted.append(True))

    assert panel.show_program_button.text() == "Show QICK Program"
    panel.show_program_button.click()
    app.processEvents()
    assert emitted == [True]

    panel.set_running(True, "Compiling", show_progress=False)
    assert panel.run_button.isEnabled() is False
    assert panel.show_program_button.isEnabled() is False
    assert panel.progress.isVisible() is False
    panel.close()


def test_experiment_panel_qcs_stop_button_lifecycle():
    app = _application()
    panel = gui.ExperimentPanel(
        fabric_mhz=300.0,
        tproc_mhz=300.0,
        full_scale_mv=800.0,
        awg_channels=(1,),
        repetitions=1,
    )
    stopped = []
    panel.stop_requested.connect(lambda: stopped.append(True))

    assert panel.execution_backend() == gui.EXECUTION_BACKEND_QCS
    assert panel.stop_button.isHidden() is False
    assert panel.stop_button.isEnabled() is False

    panel.set_running(True, "Running QCS", allow_stop=True)
    assert panel.run_button.isEnabled() is False
    assert panel.stop_button.isEnabled() is True
    panel.stop_button.click()
    app.processEvents()
    assert stopped == [True]

    panel.set_stopping()
    assert panel.stop_button.isEnabled() is False
    assert "Stopping" in panel.run_status.text()
    panel.set_running(False, "Stopped")
    assert panel.run_button.isEnabled() is True
    assert panel.stop_button.isEnabled() is False

    panel.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    assert panel.stop_button.isHidden() is True
    panel.close()


def test_qcs_stop_button_disables_when_hardware_acquisition_finishes():
    app = _application()
    window = gui.MainWindow()
    panel = window._experiment_panel
    panel.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    panel.set_running(True, "Running QCS", allow_stop=True)
    assert panel.stop_button.isEnabled() is True

    window._on_experiment_event(
        "acquisition",
        "completed",
        "QCS hardware execution completed",
    )
    app.processEvents()

    assert panel.stop_button.isEnabled() is False
    window.close()
    app.processEvents()


def test_experiment_panel_selects_qcs_and_preserves_qick_connection(tmp_path):
    app = _application()
    panel = gui.ExperimentPanel(
        fabric_mhz=300.0,
        tproc_mhz=300.0,
        full_scale_mv=800.0,
        awg_channels=(1,),
        repetitions=1,
    )

    assert panel.execution_backend() == gui.EXECUTION_BACKEND_QCS
    assert panel.run_button.text() == "Run QCS Experiment"
    assert panel.backend_selector.isHidden() is True
    assert panel.execution_system_label.text() == "Keysight QCS / M5000"
    assert panel.show_program_button.isHidden() is True
    assert panel.ddr_usage_group.isHidden() is True
    assert panel.qcs_waveform_usage_group.isHidden() is False
    assert panel.qcs_waveform_usage_group.isEnabled() is True
    assert panel.full_scale_mv_label.isHidden() is True
    assert panel.full_scale_mv.isHidden() is True
    assert panel.qcs_sample_rate_hz.isHidden() is True
    assert "M5200 ADC sample rate:" not in {
        label.text()
        for label in panel.qcs_connection_group.findChildren(QtWidgets.QLabel)
    }
    assert panel.compile_validation_mode.isHidden() is True
    assert panel.qcs_mapper_path.isReadOnly() is True
    assert panel.qcs_dc_channel_names.isReadOnly() is True
    assert panel.qcs_rf_channel_names.isReadOnly() is True
    assert panel.qcs_acquisition_channel_name.isReadOnly() is True
    assert panel.qcs_init_time_us.value() == pytest.approx(0.07)
    assert panel.qcs_init_time_us.singleStep() == pytest.approx(0.01)
    assert "Inter-point / repetition delay:" in {
        label.text()
        for label in panel.qcs_connection_group.findChildren(QtWidgets.QLabel)
    }

    panel.qcs_mapper_path.setText(str(tmp_path / "mapper.json"))
    panel.qcs_dc_channel_names.setText("gate_a")
    panel.qcs_dc_full_scale_v.setValue(2.5)
    panel.qcs_rf_channel_names.setText("0=rf_drive, 2=rf_probe")
    panel.qcs_acquisition_channel_name.setText("digitizer")
    panel.qcs_hw_demod.setChecked(False)
    panel.qcs_init_time_us.setValue(0.125)
    app.processEvents()

    assert panel.run_button.text() == "Run QCS Experiment"
    assert panel.qcs_connection_group.isHidden() is False
    assert panel.show_program_button.isEnabled() is False
    assert panel.ddr_usage_group.isEnabled() is False
    assert panel.compile_validation_mode.isEnabled() is False
    values = panel.values(1)
    assert values["execution_backend"] == gui.EXECUTION_BACKEND_QCS
    assert isinstance(values["connection"], gui.QickConnectionConfig)
    assert values["connection"].host == gui.DEFAULT_QICK_HOST
    qcs_connection = values["qcs_connection"]
    assert qcs_connection.mapper_path == str(tmp_path / "mapper.json")
    assert tuple(qcs_connection.dc_channel_names) == ("gate_a",)
    assert qcs_connection.dc_full_scale_v == pytest.approx(2.5)
    assert dict(qcs_connection.rf_channel_names) == {
        0: "rf_drive",
        2: "rf_probe",
    }
    assert qcs_connection.acquisition_channel_name == "digitizer"
    assert qcs_connection.hw_demod is False
    # HCL stores whole nanoseconds, and 300 MHz timing is exactly representable
    # every 10 ns. The inter-iteration delay rounds upward, never shorter.
    assert qcs_connection.init_time_s == pytest.approx(0.130e-6)
    assert panel.qcs_init_time_us.value() == pytest.approx(0.130)

    panel.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    assert panel.full_scale_mv_label.isHidden() is False
    assert panel.full_scale_mv.isHidden() is False
    assert panel.qcs_sample_rate_hz.isHidden() is True
    panel.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    assert panel.full_scale_mv_label.isHidden() is True
    assert panel.full_scale_mv.isHidden() is True
    panel.close()


def test_qcs_waveform_capacity_bar_blocks_over_limit_awg_setup():
    app = _application()
    window = gui.MainWindow()
    panel = window._experiment_panel
    app.processEvents()

    assert panel.execution_backend() == gui.EXECUTION_BACKEND_QCS
    assert panel.qcs_waveform_usage_group.isHidden() is False
    assert panel.qcs_waveform_usage_progress.maximum() == 98_304
    # A globally constant output is now carried by the physical M5301
    # offset, so use a true ramp to exercise the rendered-waveform ceiling.
    def set_ramp(duration_ns):
        pulse = PulseSequence(0.0, initial_duration_ns=1_000.0)
        pulse.add_flat_ramp(duration_ns, 400.0, 100.0)
        window._pulse[0] = pulse
        window._plot._pulses[0] = pulse

    set_ramp(1_000.0)
    window._refresh_qcs_waveform_capacity()
    assert panel.qcs_waveform_usage_progress.value() == 2_400
    assert "2,400 / 98,304 samples" in (
        panel.qcs_waveform_usage_progress.format()
    )

    set_ramp(1_001.0)
    window._refresh_qcs_waveform_capacity()
    assert panel.qcs_waveform_usage_progress.format() == "Invalid waveform setup"
    assert "16-sample waveform granularity" in (
        panel.qcs_waveform_usage_detail.text()
    )

    set_ramp(40_960.0)
    window._refresh_qcs_waveform_capacity()
    assert panel.qcs_waveform_usage_progress.value() == 98_304
    assert "100.00%" in panel.qcs_waveform_usage_progress.format()
    assert "#c58a1c" in panel.qcs_waveform_usage_progress.styleSheet()

    set_ramp(40_966.0)
    window._refresh_qcs_waveform_capacity()
    assert panel.qcs_waveform_usage_progress.value() == 98_304
    assert "98,320 / 98,304 samples (100.02%)" == (
        panel.qcs_waveform_usage_progress.format()
    )
    assert "#b33a3a" in panel.qcs_waveform_usage_progress.styleSheet()
    assert "run is blocked" in panel.qcs_waveform_usage_detail.text()
    with pytest.raises(ValueError, match=r"98,320 / 98,304 samples"):
        window._qcs_experiment_run_arguments()

    panel.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    assert panel.qcs_waveform_usage_group.isHidden() is True
    panel.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    assert panel.qcs_waveform_usage_group.isHidden() is False

    window.close()
    app.processEvents()


def test_qcs_no_sweep_capacity_bar_counts_each_outputs_ramps_only():
    app = _application()
    window = gui.MainWindow()
    pulses = []
    for control_mv in (150.0, -200.0):
        pulse = PulseSequence(0.0, initial_duration_ns=10_000.0)
        pulse.add_flat_ramp(15_000.0, 400.0, control_mv)
        pulse.add_flat_ramp(15_000.0, 100_000.0, 100.0)
        pulses.append(pulse)
    window._pulse[0] = pulses[0]
    window._plot._pulses[0] = pulses[0]
    window._add_port()
    window._pulse[1].t = pulses[1].t.copy()
    window._pulse[1].v = pulses[1].v.copy()
    window._pulse[1].segment_names = list(pulses[1].segment_names)
    window._cross_capacitance = np.eye(2)
    window._sweep_specs = []

    window._refresh_sweep_overlay()
    app.processEvents()

    panel = window._experiment_panel
    assert panel.qcs_waveform_usage_progress.value() == 72_000
    assert panel.qcs_waveform_usage_progress.format() == (
        "72,000 / 98,304 samples (73.24%)"
    )
    assert "30.000000 / 40.960000 us" in (
        panel.qcs_waveform_usage_summary.text()
    )
    assert "awg_0 -> dc_ch_1: 72,000 samples (30.000000 us)" in (
        panel.qcs_waveform_usage_detail.text()
    )
    assert "awg_1 -> dc_ch_2: 72,000 samples (30.000000 us)" in (
        panel.qcs_waveform_usage_detail.text()
    )
    assert "outputs are not added together" in (
        panel.qcs_waveform_usage_detail.text()
    )

    window.close()
    app.processEvents()


def test_qcs_constant_second_output_uses_offset_not_waveform_memory():
    app = _application()
    window = gui.MainWindow()

    changing = PulseSequence(0.0, initial_duration_ns=1_000.0)
    changing.add_flat_ramp(10_000.0, 400.0, 100.0)
    changing.add_flat_ramp(10_000.0, 100_000.0, 0.0)
    constant = PulseSequence(100.0, initial_duration_ns=1_000.0)
    constant.add_flat_ramp(10_000.0, 400.0, 100.0)
    constant.add_flat_ramp(10_000.0, 100_000.0, 100.0)

    window._pulse[0] = changing
    window._plot._pulses[0] = changing
    window._add_port()
    window._pulse[1].t = constant.t.copy()
    window._pulse[1].v = constant.v.copy()
    window._pulse[1].segment_names = list(constant.segment_names)
    window._cross_capacitance = np.eye(2)
    window._sweep_specs = []
    window._refresh_sweep_overlay()
    app.processEvents()

    panel = window._experiment_panel
    assert panel.qcs_waveform_usage_progress.value() == 48_000
    assert panel.qcs_waveform_usage_progress.format() == (
        "48,000 / 98,304 samples (48.83%)"
    )
    detail = panel.qcs_waveform_usage_detail.text()
    assert "awg_0 -> dc_ch_1: 48,000 samples (20.000000 us)" in detail
    assert "awg_1 -> dc_ch_2: 0 samples (0.000000 us)" in detail
    assert "fixed physical offset +100 mV" in detail
    assert "zero residual waveform" in detail
    assert panel.qcs_sweep_execution_mode_label.text().endswith("No sweep")
    assert "100 mV" in panel.qcs_sweep_execution_reason_label.text()

    window.close()
    app.processEvents()


def test_qcs_sweep_execution_indicator_shows_mode_and_fallback_reason():
    app = _application()
    panel = gui.ExperimentPanel(
        fabric_mhz=300.0,
        tproc_mhz=300.0,
        full_scale_mv=800.0,
        awg_channels=(1,),
        repetitions=1,
    )

    panel.set_qcs_sweep_execution_status(
        gui.QcsSweepExecutionPreview(
            mode="hardware",
            reasons=("The mapped offset is checked during compilation.",),
            exact=False,
        )
    )
    assert "Hardware sweep (planned)" in (
        panel.qcs_sweep_execution_mode_label.text()
    )
    assert "mapped offset" in panel.qcs_sweep_execution_reason_label.text()
    assert "#1b5e20" in panel.qcs_sweep_execution_indicator.styleSheet()

    panel.set_qcs_sweep_execution_status(
        gui.QcsSweepExecutionPreview(
            mode="software",
            reasons=("Raw trace acquisition requires software resolution.",),
        )
    )
    assert panel.qcs_sweep_execution_mode_label.text().endswith(
        "Software sweep"
    )
    assert "Raw trace" in panel.qcs_sweep_execution_indicator.toolTip()
    assert "#8a4b00" in panel.qcs_sweep_execution_indicator.styleSheet()

    panel.set_qcs_sweep_execution_status(
        gui.QcsSweepExecutionPreview(
            mode="hybrid",
            reasons=(
                "Voltage is swept in hardware inside an RF-duration "
                "software loop.",
            ),
        )
    )
    assert panel.qcs_sweep_execution_mode_label.text().endswith(
        "Hardware sweep inside software loop"
    )
    assert "RF-duration software loop" in (
        panel.qcs_sweep_execution_reason_label.text()
    )
    assert "RF-duration software loop" in (
        panel.qcs_sweep_execution_indicator.toolTip()
    )
    assert "#8a4b00" not in (
        panel.qcs_sweep_execution_indicator.styleSheet()
    )

    panel.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    assert panel.qcs_sweep_execution_indicator.isHidden() is True
    panel.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    assert panel.qcs_sweep_execution_indicator.isHidden() is False
    panel.close()
    app.processEvents()


def test_qcs_sweep_execution_indicator_tracks_live_voltage_sweep_and_trace_mode():
    app = _application()
    window = gui.MainWindow()
    pulse = PulseSequence(-100.0, initial_duration_ns=10_000.0)
    pulse.add_flat_ramp(10_000.0, 400.0, 100.0)
    pulse.add_flat_ramp(10_000.0, 100_000.0, -100.0)
    window._pulse = [pulse]
    window._cross_capacitance = np.eye(1)
    window._sweep_specs = [
        QickSweepSpec("set_2", "awg_0", -0.125, -0.3125, 20)
    ]

    window._refresh_sweep_overlay()
    app.processEvents()
    panel = window._experiment_panel
    assert panel.qcs_sweep_execution_mode_label.text().endswith(
        "Hardware sweep (planned)"
    )
    assert "M5301 offset" in panel.qcs_sweep_execution_reason_label.text()
    assert panel.qcs_waveform_usage_progress.value() == 48_032

    panel.qcs_hw_demod.setChecked(False)
    app.processEvents()
    assert panel.qcs_sweep_execution_mode_label.text().endswith(
        "Software sweep"
    )
    assert "Raw trace" in panel.qcs_sweep_execution_reason_label.text()

    window.close()
    app.processEvents()


def test_qcs_two_output_fixed_voltage_bias_t_preview_is_m5301_safe(
    monkeypatch,
    tmp_path,
):
    app = _application()
    window = gui.MainWindow()
    pulses = []
    for first_mv in (150.0, -200.0):
        pulse = PulseSequence(0.0, initial_duration_ns=10_000.0)
        pulse.add_flat_ramp(15_000.0, 400.0, first_mv)
        pulse.add_flat_ramp(15_000.0, 100_000.0, 100.0)
        pulses.append(pulse)
    window._pulse[0] = pulses[0]
    window._plot._pulses[0] = pulses[0]
    window._add_port()
    window._pulse[1].t = pulses[1].t.copy()
    window._pulse[1].v = pulses[1].v.copy()
    window._pulse[1].segment_names = list(pulses[1].segment_names)
    window._cross_capacitance = np.eye(2)
    window._sweep_specs = [
        QickSweepSpec("set_2", "awg_0", -0.375, 0.375, 3),
        QickSweepSpec("set_2", "awg_1", -0.625, 0.625, 3),
    ]
    panel = window._experiment_panel
    panel.bias_t_group.setChecked(True)
    panel.bias_t_type.setCurrentIndex(
        panel.bias_t_type.findData("dc")
    )
    panel.bias_t_mode.setCurrentIndex(
        panel.bias_t_mode.findData("fixed_voltage")
    )
    panel.bias_t_compensation_mv.setValue(80.0)

    capacity_calls = []
    real_capacity_report = gui.qcs_m5301_waveform_capacity_report

    def recording_capacity_report(sequence, **kwargs):
        capacity_calls.append(kwargs)
        return real_capacity_report(sequence, **kwargs)

    monkeypatch.setattr(
        gui,
        "qcs_m5301_waveform_capacity_report",
        recording_capacity_report,
    )

    window._refresh_sweep_overlay()
    app.processEvents()

    assert panel.qcs_sweep_execution_mode_label.text().endswith(
        "Hardware sweep inside software loop"
    )
    assert "Per-slice capacity preflight at Run" in (
        panel.qcs_waveform_usage_progress.format()
    )
    assert "Hybrid sweep" in panel.qcs_waveform_usage_summary.text()
    assert capacity_calls == []

    mapper_path = tmp_path / "bias_t_mapper.qcs"
    mapper_path.write_text("{}", encoding="utf-8")
    panel.qcs_mapper_path.setText(str(mapper_path))
    panel.qcs_dc_channel_names.setText("dc_0, dc_1")
    panel.qcs_acquisition_channel_name.setText("digitizer")
    window._rf_readout_panel.setChecked(True)

    validation_calls = []
    real_validate = gui.validate_qcs_m5301_waveform_capacity

    def recording_validate(sequence, **kwargs):
        validation_calls.append(kwargs)
        return real_validate(sequence, **kwargs)

    monkeypatch.setattr(
        gui,
        "validate_qcs_m5301_waveform_capacity",
        recording_validate,
    )
    arguments = window._qcs_experiment_run_arguments()
    assert validation_calls == []
    assert "Per-slice capacity preflight at Run" in (
        panel.qcs_waveform_usage_progress.format()
    )
    assert arguments["sequence"].bias_t_compensation.mode == "fixed_voltage"
    assert arguments["sequence"].sweep_shape == (3, 3)

    window.close()
    app.processEvents()


def test_qcs_sweep_indicator_accepts_two_output_101_by_101_hardware_grid():
    app = _application()
    window = gui.MainWindow()
    pulses = []
    for initial_mv, final_mv in ((-100.0, -100.0), (-100.0, -75.0)):
        pulse = PulseSequence(initial_mv, initial_duration_ns=10_000.0)
        pulse.add_flat_ramp(10_000.0, 400.0, 100.0)
        pulse.add_flat_ramp(10_000.0, 100_000.0, final_mv)
        pulses.append(pulse)
    window._pulse[0] = pulses[0]
    window._plot._pulses[0] = pulses[0]
    window._add_port()
    window._pulse[1].v = pulses[1].v.copy()
    window._pulse[1].segment_names = list(pulses[1].segment_names)
    window._cross_capacitance = np.eye(2)
    window._sweep_specs = [
        QickSweepSpec("set_2", "awg_0", -0.125, -0.3125, 101),
        QickSweepSpec("set_2", "awg_1", -0.09375, -0.375, 101),
    ]

    window._refresh_sweep_overlay()
    app.processEvents()

    panel = window._experiment_panel
    assert panel.qcs_sweep_execution_mode_label.text().endswith(
        "Hardware sweep (planned)"
    )
    reason = panel.qcs_sweep_execution_reason_label.text()
    assert "0.07 us inter-iteration delay" in reason
    assert panel.qcs_waveform_usage_progress.value() == 48_032

    window._active_experiment_backend = gui.EXECUTION_BACKEND_QCS
    window._on_experiment_event(
        "program_build",
        "started",
        "Compiling fixed numeric QCS DC-ramp programs",
    )
    assert panel.qcs_sweep_execution_mode_label.text().endswith(
        "Software sweep"
    )

    window.close()
    app.processEvents()


def test_qcs_run_preflight_samples_large_sweep_capacity(
    monkeypatch,
    tmp_path,
):
    app = _application()
    window = gui.MainWindow()
    pulses = []
    for initial_mv, final_mv in ((-100.0, -100.0), (-100.0, -75.0)):
        pulse = PulseSequence(initial_mv, initial_duration_ns=10_000.0)
        pulse.add_flat_ramp(10_000.0, 400.0, 100.0)
        pulse.add_flat_ramp(10_000.0, 100_000.0, final_mv)
        pulses.append(pulse)
    window._pulse[0] = pulses[0]
    window._plot._pulses[0] = pulses[0]
    window._add_port()
    window._pulse[1].v = pulses[1].v.copy()
    window._pulse[1].segment_names = list(pulses[1].segment_names)
    window._cross_capacitance = np.eye(2)
    window._sweep_specs = [
        QickSweepSpec("set_2", "awg_0", -0.125, -0.3125, 101),
        QickSweepSpec("set_2", "awg_1", -0.09375, -0.375, 101),
    ]

    panel = window._experiment_panel
    mapper_path = tmp_path / "large_sweep_mapper.qcs"
    mapper_path.write_text("{}", encoding="utf-8")
    panel.qcs_mapper_path.setText(str(mapper_path))
    panel.qcs_dc_channel_names.setText("dc_0, dc_1")
    panel.qcs_acquisition_channel_name.setText("digitizer")
    window._rf_readout_panel.setChecked(True)

    calls = []
    real_validate = gui.validate_qcs_m5301_waveform_capacity

    def recording_validate(sequence, **kwargs):
        calls.append(kwargs.get("point_indices"))
        return real_validate(sequence, **kwargs)

    monkeypatch.setattr(
        gui,
        "validate_qcs_m5301_waveform_capacity",
        recording_validate,
    )

    arguments = window._qcs_experiment_run_arguments()

    assert arguments["sequence"].sweep_point_count == 101 * 101
    assert calls == [
        gui.qcs_m5301_capacity_preview_point_indices(arguments["sequence"])
    ]
    assert 0 < len(calls[0]) < arguments["sequence"].sweep_point_count

    window.close()
    app.processEvents()


def test_qcs_experiment_arguments_convert_rf_and_acquisition(tmp_path):
    app = _application()
    window = gui.MainWindow()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    mapper_path = tmp_path / "mapper.json"
    mapper_path.write_text("{}", encoding="utf-8")
    experiment.qcs_mapper_path.setText(str(mapper_path))
    experiment.qcs_dc_channel_names.setText("gate_a")
    experiment.qcs_rf_channel_names.setText("1=rf_drive")
    experiment.qcs_acquisition_channel_name.setText("digitizer")
    experiment.database_path.setText(str(tmp_path / "qcs_run.db"))
    experiment.repetitions.setValue(3)
    experiment.set_iq_repetition_policy(
        gui.IQ_REPETITION_POLICY_COHERENT_AVERAGE
    )

    rf_panel = window._rf_ports_panel._panels[0]
    rf_panel.setChecked(True)
    rf_panel.gen_ch.setValue(1)
    rf_panel.delay.setValue(0.25)
    rf_panel.duration.setValue(2.5)
    rf_panel.frequency_mhz.setValue(75.0)
    rf_panel.qcs_amplitude.setValue(-0.25)
    assert rf_panel.gain.value() == -8192
    rf_panel.phase_degrees.setValue(90.0)

    readout = window._rf_readout_panel
    readout.setChecked(True)
    readout.delay.setValue(0.5)
    readout.qcs_acquisition_duration.setValue(32 / 4.8e9 * 1.0e6)
    readout.frequency_mhz.setValue(42.0)
    readout.margin_samples.setValue(9876)
    readout.override_fpga_trigger_delay.setChecked(True)
    readout.fpga_trigger_delay_us.setValue(13.25)
    readout.input_board_type.setCurrentText("DC_In")
    readout.measurement_unit.setCurrentIndex(
        readout.measurement_unit.findData("current")
    )
    app.processEvents()

    arguments = window._qcs_experiment_run_arguments()
    assert isinstance(arguments["connection_config"], gui.QcsConnectionConfig)
    assert arguments["connection_config"].acquisition_channel_name == "digitizer"
    assert arguments["repetitions_per_sweep"] == 3
    assert arguments["iq_repetition_policy"] == (
        gui.IQ_REPETITION_POLICY_COHERENT_AVERAGE
    )
    assert arguments["fabric_mhz"] == 300.0
    assert arguments["source_full_scale_mv"] == pytest.approx(2500.0)
    assert len(arguments["rf_pulses"]) == 1
    qcs_rf = arguments["rf_pulses"][0]
    assert qcs_rf.gen_ch == 1
    assert qcs_rf.at_segment == "set_0"
    assert qcs_rf.delay_s == pytest.approx(0.25e-6)
    assert qcs_rf.duration_s == pytest.approx(2.5e-6)
    assert qcs_rf.amplitude == pytest.approx(-0.25)
    assert qcs_rf.frequency_hz == pytest.approx(75.0e6)
    assert qcs_rf.phase_rad == pytest.approx(np.pi / 2)
    assert qcs_rf.require_within_segment is True
    acquisition = arguments["acquisition"]
    assert readout.qcs_single_iq_radio.isChecked() is True
    assert arguments["connection_config"].hw_demod is True
    assert acquisition.at_segment == "set_0"
    assert acquisition.pre_delay_s == pytest.approx(0.5e-6)
    assert acquisition.sample_rate_hz == pytest.approx(4.8e9)
    assert acquisition.sample_count == 32
    assert acquisition.duration_s == pytest.approx(32 / 4.8e9)
    assert acquisition.frequency_hz == pytest.approx(42.0e6)
    assert arguments["gui_settings"]["experiment"]["execution_backend"] == "qcs"

    readout.qcs_trace_radio.click()
    app.processEvents()
    raw_arguments = window._qcs_experiment_run_arguments()
    assert readout.qcs_trace_radio.isChecked() is True
    assert experiment.qcs_hw_demod.isChecked() is False
    assert raw_arguments["connection_config"].hw_demod is False
    assert raw_arguments["iq_repetition_policy"] == (
        gui.IQ_REPETITION_POLICY_PRESERVE
    )
    assert raw_arguments["acquisition"].sample_count == 32
    assert raw_arguments["acquisition"].frequency_hz == 0.0
    assert readout.frequency_mhz.isHidden() is True
    assert readout.margin_samples.value() == 9876
    assert readout.fpga_trigger_delay_us.value() == pytest.approx(13.25)
    assert readout.measurement_unit.currentData() == "current"
    app.processEvents()
    window.close()


def test_qcs_run_arguments_preserve_physical_awg_voltage_and_sweep(
    tmp_path,
):
    app = _application()
    window = gui.MainWindow()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    experiment.qcs_dc_full_scale_v.setValue(2.5)
    mapper_path = tmp_path / "physical_voltage_mapper.qcs"
    mapper_path.write_text("{}", encoding="utf-8")
    experiment.qcs_mapper_path.setText(str(mapper_path))
    experiment.qcs_dc_channel_names.setText("gate_a")
    experiment.qcs_acquisition_channel_name.setText("digitizer")
    window._rf_readout_panel.setChecked(True)
    app.processEvents()

    assert window._pulse[0].edit_voltage(0, 1500.0)
    window._sweep_specs = [
        QickSweepSpec("set_0", "awg_0", -0.6, 0.8, 5)
    ]
    window._notify_sweep_state_changed(fit_view=True)
    app.processEvents()

    arguments = window._qcs_experiment_run_arguments()

    source_scale_mv = arguments["source_full_scale_mv"]
    assert arguments["connection_config"].dc_full_scale_v == pytest.approx(2.5)
    assert source_scale_mv == pytest.approx(2500.0)
    set_0 = next(
        segment
        for segment in arguments["sequence"].segments
        if segment.name == "set_0"
    )
    assert set_0.amplitudes[0] * source_scale_mv == pytest.approx(1500.0)
    sweep_axis = arguments["sequence"].sweep_axes[0]
    assert sweep_axis.points[0] * source_scale_mv == pytest.approx(-1500.0)
    assert sweep_axis.points[-1] * source_scale_mv == pytest.approx(2000.0)

    window.close()
    window.deleteLater()
    app.processEvents()


def test_generated_qcs_code_uses_configured_dc_full_scale():
    app = _application()
    window = gui.MainWindow()
    window._experiment_panel.qcs_dc_channel_names.setText("gate_plunger")
    window._experiment_panel.qcs_dc_full_scale_v.setValue(2.5)

    code = window._generate_qcs_code()

    assert "QCS_FULL_SCALE_V = 2.5" in code
    assert "gate_plunger: qcs.Channels" in code
    assert "dc_ch_1: qcs.Channels" not in code
    window._experiment_panel.qcs_dc_full_scale_v.setValue(0.05)
    with pytest.raises(ValueError, match=r"exceeding.*\+/-50 mV"):
        window._generate_qcs_code()
    window.close()
    app.processEvents()


def test_qcs_ignores_dormant_qick_rf_power_calibration_without_mutation(
    tmp_path,
):
    app = _application()
    window = gui.MainWindow()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    experiment.qcs_mapper_path.setText(str(tmp_path / "mapper.json"))
    (tmp_path / "mapper.json").write_text("{}", encoding="utf-8")
    experiment.qcs_dc_channel_names.setText("gate_a")
    experiment.qcs_rf_channel_names.setText("0=rf_drive")
    experiment.qcs_acquisition_channel_name.setText("digitizer")
    rf_panel = window._rf_ports_panel._panels[0]
    rf_panel.setChecked(True)
    rf_panel.gen_ch.setValue(0)
    rf_panel.gain.setValue(8192)
    rf_panel.duration_sweep_enabled.setChecked(True)
    rf_panel.duration_sweep_start.setValue(1.0)
    rf_panel.duration_sweep_stop.setValue(2.0)
    rf_panel.duration_sweep_count.setValue(3)
    rf_panel.frequency_sweep_enabled.setChecked(True)
    rf_panel.frequency_sweep_start_mhz.setValue(100.0)
    rf_panel.frequency_sweep_stop_mhz.setValue(200.0)
    rf_panel.frequency_sweep_count.setValue(5)
    rf_panel.power_calibration_group.setChecked(True)
    rf_panel.power_calibration_database_path.setText(str(tmp_path / "power.db"))
    rf_panel.power_calibration_run_id.setValue(17)
    rf_panel.target_output_power_dbm.setValue(-27.5)
    rf_panel.power_sweep_enabled.setChecked(True)
    rf_panel.power_sweep_start_dbm.setValue(-40.0)
    rf_panel.power_sweep_stop_dbm.setValue(-20.0)
    rf_panel.power_sweep_count.setValue(7)
    window._rf_readout_panel.setChecked(True)
    legacy_settings = rf_panel.settings_dict()

    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    arguments = window._qcs_experiment_run_arguments()

    assert len(arguments["rf_pulses"]) == 1
    assert arguments["rf_pulses"][0].amplitude == pytest.approx(8192 / 32767)
    assert [
        axis.axis_kind for axis in arguments["sequence"].sweep_axes
    ] == ["rf_duration", "rf_frequency"]
    assert rf_panel.settings_dict() == legacy_settings
    assert rf_panel.power_calibration_group.isChecked() is True
    assert rf_panel.power_sweep_enabled.isChecked() is True
    assert arguments["gui_settings"]["rf_outputs"][0] == legacy_settings

    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    assert rf_panel.settings_dict() == legacy_settings
    app.processEvents()
    window.close()


def test_experiment_panel_records_run_elapsed_time_and_stage_events():
    app = _application()
    panel = gui.ExperimentPanel(
        fabric_mhz=300.0,
        tproc_mhz=300.0,
        full_scale_mv=800.0,
        awg_channels=(1,),
        repetitions=1,
    )

    panel.start_run_timeline("Run requested")
    panel.record_run_event("validation", "started", "Checking settings")
    panel.record_run_event("validation", "completed", "Settings validated")
    panel.record_run_event("compile", "started", "Compiling")
    panel.record_run_event("compile", "completed", "Compiled")
    panel.finish_run_timeline("Run completed", success=True)
    app.processEvents()

    log = panel.run_event_log.toPlainText()
    assert "Experiment STARTED" in log
    assert "Run validation COMPLETED" in log
    assert "tProcessor compile STARTED" in log
    assert "tProcessor compile COMPLETED" in log
    assert "stage 00:00:" in log
    assert "Experiment COMPLETED" in log
    assert "Finished | Started " in panel.run_elapsed_label.text()
    assert panel.clear_run_log_button.isEnabled() is True
    panel.close()


def test_experiment_panel_defaults_to_parametric_awg_metadata():
    app = _application()
    panel = gui.ExperimentPanel(
        fabric_mhz=300.0,
        tproc_mhz=300.0,
        full_scale_mv=800.0,
        awg_channels=(1,),
        repetitions=1,
    )
    emitted = []
    panel.awg_metadata_requested.connect(lambda: emitted.append(True))

    assert panel.awg_metadata_mode.currentData() == "parametric"
    assert panel.values(1)["awg_metadata_mode"] == "parametric"
    assert panel.compile_validation_mode.currentData() == "boundary"
    assert panel.values(1)["compile_validation_mode"] == "boundary"
    panel.awg_metadata_button.click()
    app.processEvents()
    assert emitted == [True]

    panel.set_awg_metadata_mode("expanded")
    assert panel.values(1)["awg_metadata_mode"] == "expanded"
    panel.set_compile_validation_mode("full")
    assert panel.values(1)["compile_validation_mode"] == "full"
    panel.close()


def test_experiment_panel_visualizes_sweep_repetition_ddr_usage():
    app = _application()
    panel = gui.ExperimentPanel(
        fabric_mhz=300.0,
        tproc_mhz=300.0,
        full_scale_mv=800.0,
        awg_channels=(1,),
        repetitions=3,
    )
    panel.set_sweep_specs(
        (
            SimpleNamespace(
                axis_kind="amplitude",
                output_name="awg_0",
                segment_name="set_0",
                start=-0.1,
                stop=0.1,
                count=5,
            ),
            SimpleNamespace(
                axis_kind="rf_frequency",
                output_name="rf_gen_0_frequency",
                segment_name="set_0",
                gen_ch=0,
                start=100.0,
                stop=200.0,
                count=7,
            ),
        )
    )
    panel.set_ddr_readout_spec(
        QickDdrReadoutSpec(
            ro_ch=0,
            segment_name="set_0",
            delay_us=0.0,
            samples_per_trigger=10,
        )
    )
    panel.set_ddr_memory_configuration(
        SimpleNamespace(
            ddr_capacity_words_32b=4096,
            ddr_samples_per_axi_word=8,
        )
    )
    app.processEvents()

    assert "35 Cartesian point(s)" in panel.ddr_usage_summary.text()
    assert "105 DDR trigger(s)" in panel.ddr_usage_summary.text()
    assert "16 padded 32-bit word(s)/trigger" in panel.ddr_usage_detail.text()
    assert "6.562 KiB" in panel.ddr_usage_detail.text()
    assert panel.ddr_usage_progress.value() == 4102
    assert "41.016%" in panel.ddr_usage_progress.format()

    panel.repetitions.setValue(4)
    app.processEvents()
    assert "140 DDR trigger(s)" in panel.ddr_usage_summary.text()
    assert "54.688%" in panel.ddr_usage_progress.format()
    panel.close()


def test_awg_metadata_dialog_expands_only_the_selected_point():
    app = _application()
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("gate", (0.0,), 30)
    sequence.set_amplitude_sweep("gate", "awg_0", -0.5, 0.5, 3)
    dialog = gui.AwgMetadataDialog(
        sequence,
        fabric_mhz=300.0,
        full_scale_mv=800.0,
    )

    assert dialog.recipe["point_count"] == 3
    assert dialog.recipe["schema"] == "qick-awg-waveform-recipe-v1"
    assert dialog.point_record["point_index"] == 0
    dialog.point_index.setValue(2)
    app.processEvents()
    assert dialog.point_record["point_index"] == 2
    assert dialog.point_record["sweep_coordinate"] == [0.5]
    assert dialog.point_record["virtual_values_mv"]["awg_0"] == [400.0, 400.0]
    assert dialog.vertex_table.rowCount() == 2
    dialog.close()


def test_show_program_snapshot_allows_disabled_readout():
    app = _application()
    window = gui.MainWindow()
    window._experiment_panel.database_path.clear()
    window._experiment_panel.experiment_name.clear()
    window._experiment_panel.sample_name.clear()

    assert window._rf_readout_panel.spec() is None
    arguments = window._experiment_run_arguments(
        require_readout=False, require_run_config=False
    )

    assert arguments["readout_spec"] is None
    assert arguments["run_config"] is None
    assert arguments["sequence"].n_outputs == 1
    app.processEvents()
    window.close()


def test_qick_program_worker_compiles_and_returns_assembly(monkeypatch):
    app = _application()
    calls = []

    class FakeProgram:
        prog_list = [{"name": "regwi"}, {"name": "end"}]

        def compile(self):
            calls.append("compile")
            self.binprog = [1, 2]

        def asm(self):
            return "// Program\nregwi 0, 1, 2;\nend;"

        def summary(self):
            return {"sweep_points": 3}

    fake_program = FakeProgram()
    monkeypatch.setattr(
        gui, "connect_qick", lambda config: (object(), {"gens": []})
    )
    monkeypatch.setattr(
        gui,
        "build_qick_program",
        lambda soccfg, **kwargs: fake_program,
    )
    results = []
    failures = []
    worker = gui.QickProgramWorker(
        object(),
        {
            "sequence": object(),
            "awg_channels": (1,),
            "repetitions_per_sweep": 1,
        },
    )
    worker.finished.connect(results.append)
    worker.failed.connect(failures.append)
    worker.run()
    app.processEvents()

    assert failures == []
    assert calls == ["compile"]
    assert results[0]["instruction_count"] == 2
    assert results[0]["machine_word_count"] == 2
    assert "regwi" in results[0]["assembly"]


def test_qcs_experiment_worker_forwards_progress_and_events(monkeypatch):
    app = _application()
    sentinel = object()
    received = {}

    def fake_run_qcs_qcodes_experiment(**kwargs):
        received.update(kwargs)
        kwargs["progress_callback"](42, "Executing QCS")
        kwargs["event_callback"](
            "execution",
            "started",
            "Keysight executor started",
        )
        return sentinel

    monkeypatch.setattr(
        gui,
        "run_qcs_qcodes_experiment",
        fake_run_qcs_qcodes_experiment,
    )
    results = []
    failures = []
    progress = []
    events = []
    worker = gui.QcsExperimentWorker({"sequence": object()})
    worker.finished.connect(results.append)
    worker.failed.connect(failures.append)
    worker.progress_changed.connect(lambda *args: progress.append(args))
    worker.event_changed.connect(lambda *args: events.append(args))
    worker.run()
    app.processEvents()

    assert failures == []
    assert results == [sentinel]
    assert received["sequence"] is not None
    assert isinstance(
        received["cancellation"],
        gui.QcsCancellationController,
    )
    assert progress == [(42, "Executing QCS")]
    assert events == [
        ("execution", "started", "Keysight executor started")
    ]


def test_experiment_panel_shows_calibrated_qcs_rf_result_without_qick_keys():
    app = _application()
    panel = gui.ExperimentPanel(
        fabric_mhz=300.0,
        tproc_mhz=300.0,
        full_scale_mv=800.0,
        awg_channels=(1,),
        repetitions=1,
    )
    result = SimpleNamespace(
        rf_settings={
            "backend": gui.EXECUTION_BACKEND_QCS,
            "output_details": (
                {
                    "gen_ch": 2,
                    "amplitude": 0.125,
                    "frequency_hz": 1.25e9,
                    "duration_s": 1.0e-6,
                    "power_calibration": {
                        "run_id": 17,
                        "target_power_dbm": -26.0,
                    },
                },
            ),
            "readout_details": {},
        },
        program_summary={},
        ddr_result=None,
        run_id=5,
        row_count=1,
        database_path="qcs_measurement.db",
    )

    panel.show_result(result)

    assert "-26 dBm" in panel.run_status.text()
    assert "calibration Run 17" in panel.run_status.text()
    assert "relative amplitude 0.125" in panel.run_status.text()
    panel.close()
    panel.deleteLater()
    app.processEvents()


def test_qcs_experiment_worker_forwards_partial_results(monkeypatch):
    app = _application()
    partials_sent = [object(), object()]
    final_result = object()
    received = {}

    def fake_run_qcs_qcodes_experiment(**kwargs):
        received.update(kwargs)
        for partial in partials_sent:
            kwargs["partial_callback"](partial)
        return final_result

    monkeypatch.setattr(
        gui,
        "run_qcs_qcodes_experiment",
        fake_run_qcs_qcodes_experiment,
    )
    partials_received = []
    results = []
    failures = []
    worker = gui.QcsExperimentWorker({"sequence": object()})
    worker.partial_result.connect(partials_received.append)
    worker.finished.connect(results.append)
    worker.failed.connect(failures.append)

    worker.run()
    app.processEvents()

    assert failures == []
    assert partials_received == partials_sent
    assert results == [final_result]
    assert callable(received["partial_callback"])


def test_qcs_experiment_worker_reports_user_stop_without_failure(monkeypatch):
    app = _application()

    def fake_run_qcs_qcodes_experiment(**kwargs):
        assert kwargs["cancellation"].is_stop_requested() is True
        raise gui.QcsExperimentCancelled("stopped by user")

    monkeypatch.setattr(
        gui,
        "run_qcs_qcodes_experiment",
        fake_run_qcs_qcodes_experiment,
    )
    stopped = []
    failures = []
    worker = gui.QcsExperimentWorker({"sequence": object()})
    worker.stopped.connect(stopped.append)
    worker.failed.connect(failures.append)

    assert worker.request_stop() is True
    assert worker.request_stop() is False
    worker.run()
    app.processEvents()

    assert failures == []
    assert len(stopped) == 1
    assert "Stopped by user" in stopped[0]


def test_qick_assembly_dialog_is_read_only_and_copyable():
    app = _application()
    assembly = "// Program\nregwi 0, 1, 2;\nend;"
    dialog = gui.QickAssemblyDialog(
        {
            "assembly": assembly,
            "instruction_count": 2,
            "machine_word_count": 2,
        }
    )

    assert dialog.assembly_text.isReadOnly() is True
    assert dialog.assembly_text.toPlainText() == assembly
    assert "2 assembly instructions" in dialog.summary_label.text()
    dialog.copy_button.click()
    app.processEvents()
    assert QtWidgets.QApplication.clipboard().text() == assembly


def test_detailed_error_dialog_copies_summary_and_traceback():
    app = _application()
    details = "Traceback (most recent call last):\nModuleNotFoundError: numpy._core"
    dialog = gui.DetailedErrorMessageBox(
        "RF S-parameter sweep failed",
        "ModuleNotFoundError: numpy._core",
        details,
    )

    assert dialog.detailedText() == details
    assert dialog.copy_button.text() == "Copy Details"
    dialog.copy_button.click()
    app.processEvents()
    copied = QtWidgets.QApplication.clipboard().text()
    assert "RF S-parameter sweep failed" in copied
    assert details in copied
    dialog.close()


def test_main_fit_action_also_fits_sparameter_plot():
    app = _application()
    window = gui.MainWindow()
    calls = []
    window._sparameter_plot.fit_view = lambda: calls.append("sparameter")

    window._fit_view()
    app.processEvents()

    assert calls == ["sparameter"]
    window.close()


def test_legacy_single_waveform_json_remains_loadable(tmp_path):
    app = _application()
    pulse = PulseSequence(-125.0, initial_duration_ns=750.0)
    pulse.add_flat_ramp(125.0, 500.0, 225.0)
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(pulse.to_dict()), encoding="utf-8")

    window = gui.MainWindow()
    window._load_settings_json(path)
    app.processEvents()
    assert window._pulse[0].to_dict() == pulse.to_dict()
    window.close()


def test_segment_names_survive_insert_delete_copy_and_json():
    pulse = PulseSequence(-125.0, initial_duration_ns=750.0)
    pulse.rename_segment(0, "Reset")
    pulse.add_flat_ramp(125.0, 500.0, 225.0)
    pulse.rename_segment(1, "Readout")

    assert pulse.insert_flat_ramp(2, 50.0, 100.0)
    inserted_name = pulse.segment_name(1)
    assert inserted_name not in {"Reset", "Readout"}
    assert pulse.segment_names == ["Reset", inserted_name, "Readout"]

    assert pulse.delete_flat_ramp(2)
    assert pulse.segment_names == ["Reset", "Readout"]
    assert pulse.copy().segment_names == ["Reset", "Readout"]
    restored = PulseSequence.from_dict(pulse.to_dict())
    assert restored.segment_names == ["Reset", "Readout"]

    legacy = pulse.to_dict()
    legacy.pop("segment_names")
    restored_legacy = PulseSequence.from_dict(legacy)
    assert restored_legacy.segment_names == ["set_0", "set_1"]


def test_settings_without_tproc_clock_use_300_mhz_default():
    app = _application()
    window = gui.MainWindow()
    document = window._settings_to_dict()
    document["qick"].pop("tproc_mhz")

    decoded = window._decode_settings(document)
    assert decoded["tproc_mhz"] == 300.0
    window._apply_decoded_settings(decoded)
    assert window._experiment_panel.tproc_mhz.value() == 300.0
    app.processEvents()
    window.close()


def test_qcs_backend_settings_round_trip_and_old_files_default_to_qick(tmp_path):
    app = _application()
    source = gui.MainWindow()
    panel = source._experiment_panel
    panel.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    panel.qcs_mapper_path.setText(str(tmp_path / "mapper.json"))
    panel.qcs_dc_channel_names.setText("gate_a")
    panel.qcs_dc_full_scale_v.setValue(2.5)
    panel.qcs_rf_channel_names.setText("0=rf_drive")
    panel.qcs_acquisition_channel_name.setText("digitizer")
    source._rf_readout_panel.samples.setValue(777)
    source._rf_readout_panel.qcs_acquisition_duration.setValue(0.0101)
    panel.set_iq_repetition_policy(
        gui.IQ_REPETITION_POLICY_COHERENT_AVERAGE
    )
    source._rf_readout_panel.qcs_trace_radio.click()
    assert panel.qcs_hw_demod.isChecked() is False
    panel.qcs_sample_rate_hz.setValue(2.4e9)
    assert panel.qcs_sample_rate_hz.value() == pytest.approx(4.8e9)
    panel.qcs_init_time_us.setValue(0.25)

    document = source._settings_to_dict()
    assert document["version"] == 42
    assert document["experiment"]["execution_backend"] == "qcs"
    assert document["experiment"]["iq_repetition_policy"] == (
        gui.IQ_REPETITION_POLICY_COHERENT_AVERAGE
    )
    assert document["qcs"] == {
        "mapper_path": str(tmp_path / "mapper.json"),
        "dc_channel_names": ["gate_a"],
        "dc_full_scale_v": 2.5,
        "rf_channel_names": {"0": "rf_drive"},
        "acquisition_channel_name": "digitizer",
        "hw_demod": False,
        "sample_rate_hz": 4.8e9,
        "init_time_s": pytest.approx(0.25e-6),
        "blocking": True,
        "hardware_configuration": None,
        "hardware_configuration_state": "external",
        "hardware_mapper_sha256": None,
    }
    assert document["rf_readout"]["samples_per_trigger"] == 777
    assert document["rf_readout"][
        "qcs_acquisition_duration_s"
    ] == pytest.approx(10.1e-9)

    restored = gui.MainWindow()
    restored._apply_decoded_settings(restored._decode_settings(document))
    restored_panel = restored._experiment_panel
    assert restored_panel.execution_backend() == gui.EXECUTION_BACKEND_QCS
    assert restored_panel.qcs_mapper_path.text() == str(tmp_path / "mapper.json")
    assert restored_panel.qcs_dc_channel_names.text() == "gate_a"
    assert restored_panel.qcs_dc_full_scale_v.value() == pytest.approx(2.5)
    assert restored_panel.qcs_rf_channel_names.text() == "0=rf_drive"
    assert restored_panel.qcs_acquisition_channel_name.text() == "digitizer"
    assert restored_panel.qcs_hw_demod.isChecked() is False
    assert restored_panel.iq_repetition_policy_value() == (
        gui.IQ_REPETITION_POLICY_COHERENT_AVERAGE
    )
    assert restored_panel.iq_repetition_policy_value(effective=True) == (
        gui.IQ_REPETITION_POLICY_PRESERVE
    )
    assert restored_panel.iq_repetition_policy.isEnabled() is False
    assert restored._rf_readout_panel.qcs_trace_radio.isChecked() is True
    assert restored._rf_readout_panel.qcs_single_iq_radio.isChecked() is False
    assert restored._rf_readout_panel.samples.value() == 777
    assert restored._rf_readout_panel.qcs_acquisition_duration.value() == (
        pytest.approx(0.0101)
    )
    assert restored_panel.qcs_sample_rate_hz.value() == pytest.approx(4.8e9)
    assert restored_panel.qcs_init_time_us.value() == pytest.approx(0.25)

    legacy_rate_document = json.loads(json.dumps(document))
    legacy_rate_document["qcs"]["sample_rate_hz"] = 1.0e6
    decoded_legacy_rate = source._decode_settings(legacy_rate_document)
    assert decoded_legacy_rate["qcs_settings"]["sample_rate_hz"] == (
        pytest.approx(4.8e9)
    )

    version_38_default_delay = json.loads(json.dumps(document))
    version_38_default_delay["version"] = 38
    version_38_default_delay["qcs"]["init_time_s"] = 100.0e-6
    decoded_version_38_default_delay = source._decode_settings(
        version_38_default_delay
    )
    assert decoded_version_38_default_delay["qcs_settings"][
        "init_time_s"
    ] == pytest.approx(gui.DEFAULT_QCS_INIT_TIME_US * 1.0e-6)

    current_explicit_delay = json.loads(json.dumps(document))
    current_explicit_delay["qcs"]["init_time_s"] = 100.0e-6
    assert source._decode_settings(current_explicit_delay)["qcs_settings"][
        "init_time_s"
    ] == pytest.approx(100.0e-6)

    document_without_mode = json.loads(json.dumps(document))
    document_without_mode["qcs"].pop("hw_demod")
    restored_without_mode = gui.MainWindow()
    restored_without_mode._apply_decoded_settings(
        restored_without_mode._decode_settings(document_without_mode)
    )
    assert (
        restored_without_mode._rf_readout_panel
        .qcs_single_iq_radio.isChecked()
        is True
    )
    assert (
        restored_without_mode._rf_readout_panel.qcs_trace_radio.isChecked()
        is False
    )

    version_37 = json.loads(json.dumps(document))
    version_37["version"] = 37
    version_37["experiment"].pop("iq_repetition_policy")
    restored_version_37 = gui.MainWindow()
    restored_version_37._apply_decoded_settings(
        restored_version_37._decode_settings(version_37)
    )
    assert restored_version_37._experiment_panel.iq_repetition_policy_value() == (
        gui.IQ_REPETITION_POLICY_PRESERVE
    )

    version_36 = json.loads(json.dumps(document))
    version_36["version"] = 36
    version_36["rf_readout"].pop("qcs_acquisition_duration_s")
    restored_version_36 = gui.MainWindow()
    restored_version_36._apply_decoded_settings(
        restored_version_36._decode_settings(version_36)
    )
    assert restored_version_36._rf_readout_panel.samples.value() == 777
    assert (
        restored_version_36._rf_readout_panel
        .qcs_acquisition_duration.value()
        == pytest.approx(777 / 4.8e9 * 1.0e6)
    )

    version_35 = json.loads(json.dumps(document))
    version_35["version"] = 35
    version_35["qcs"].pop("hardware_configuration")
    version_35["qcs"].pop("hardware_configuration_state")
    version_35["qcs"].pop("hardware_mapper_sha256")
    decoded_version_35 = source._decode_settings(version_35)
    assert (
        decoded_version_35["qcs_settings"]["hardware_configuration"] is None
    )
    restored_version_35 = gui.MainWindow()
    restored_version_35._apply_decoded_settings(decoded_version_35)
    upgraded_qcs = restored_version_35._settings_to_dict()["qcs"]
    assert upgraded_qcs["hardware_configuration"] is None
    assert upgraded_qcs["hardware_configuration_state"] == "external"
    assert upgraded_qcs["hardware_mapper_sha256"] is None

    legacy = json.loads(json.dumps(document))
    legacy["version"] = 34
    legacy.pop("qcs")
    legacy["experiment"].pop("execution_backend")
    decoded_legacy = source._decode_settings(legacy)
    assert decoded_legacy["execution_backend"] == gui.EXECUTION_BACKEND_QICK
    assert (
        decoded_legacy["qcs_settings"]["mapper_path"]
        == gui.DEFAULT_QCS_MAPPER_PATH
    )
    assert decoded_legacy["qcs_settings"]["dc_channel_names"] == ["dc_ch_1"]
    assert decoded_legacy["qcs_settings"]["dc_full_scale_v"] == pytest.approx(
        gui.DEFAULT_QCS_FULL_SCALE_V
    )
    assert decoded_legacy["qcs_settings"]["init_time_s"] == pytest.approx(
        gui.DEFAULT_QCS_INIT_TIME_US * 1.0e-6
    )
    for window in (
        restored_version_35,
        restored_version_36,
        restored_version_37,
        restored_without_mode,
        restored,
        source,
    ):
        window.close()
        window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_legacy_stability_settings_adopt_existing_qcs_rf_calibration(
    tmp_path,
):
    app = _application()
    database_path = tmp_path / "m5300_calibration.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE qcs_rf_power_calibration_runs (id INTEGER)"
        )
        connection.execute(
            "INSERT INTO qcs_rf_power_calibration_runs VALUES (1)"
        )

    window = gui.MainWindow()
    document = window._settings_to_dict()
    document["version"] = 41
    for key in (
        "qcs_power_calibration_enabled",
        "qcs_power_calibration_database_path",
        "qcs_power_calibration_run_id",
        "qcs_target_output_power_dbm",
    ):
        document["stability_diagram"].pop(key, None)
    document["calibration"]["database_path"] = str(database_path)

    decoded = window._decode_settings(document)
    migrated = decoded["stability_diagram"]

    assert migrated["qcs_power_calibration_enabled"] is True
    assert migrated["qcs_power_calibration_database_path"] == str(
        database_path
    )
    assert migrated["qcs_power_calibration_run_id"] == 0
    assert migrated["qcs_target_output_power_dbm"] == pytest.approx(-20.0)
    window._apply_decoded_settings(decoded)
    assert window._stability_panel.qcs_power_calibration_group.isChecked()
    window.close()
    window.deleteLater()
    app.processEvents()


def test_qcs_unsaved_front_panel_configuration_blocks_run(tmp_path):
    app = _application()
    window = gui.MainWindow()
    panel = window._experiment_panel
    configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("dc_ch_1",),
        {},
        None,
    )
    settings = panel.qcs_settings_dict()
    mapper_path = tmp_path / "mapper.qcs"
    mapper_path.write_bytes(b"saved mapper contents")
    settings.update(
        {
            "mapper_path": str(mapper_path),
            "hardware_configuration": configuration,
            "hardware_configuration_state": "draft",
            "hardware_mapper_sha256": None,
        }
    )
    panel.set_qcs_settings(settings, 1)
    panel.set_execution_backend(gui.EXECUTION_BACKEND_QCS)

    with pytest.raises(ValueError, match="unsaved physical changes"):
        panel.qcs_connection_values(1)

    settings["hardware_configuration_state"] = "saved"
    settings["hardware_mapper_sha256"] = (
        qcs_front_panel.qcs_mapper_file_sha256(mapper_path)
    )
    panel.set_qcs_settings(settings, 1)
    assert panel.qcs_connection_values(1).mapper_path == str(mapper_path)
    mapper_path.write_bytes(b"replacement mapper contents")
    with pytest.raises(ValueError, match="file changed"):
        panel.qcs_connection_values(1)
    mapper_path.write_bytes(b"saved mapper contents")
    panel.qcs_mapper_path.setText(str(tmp_path / "different_mapper.qcs"))
    with pytest.raises(ValueError, match="unsaved physical changes"):
        panel.qcs_connection_values(1)
    window.close()
    app.processEvents()


def test_qcs_dc_calibration_reuses_existing_full_scale_control(monkeypatch):
    app = _application()
    window = gui.MainWindow()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    experiment.qcs_dc_full_scale_v.setValue(2.5)

    configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("dc_ch_1",),
        {},
        None,
    )
    calibration_panel = window._calibration_panel
    calibration_panel.set_qcs_front_panel_configuration(configuration)
    calibration_panel.qcs_dc_scope_resource.setText("USB::MOCK_SCOPE")
    config = calibration_panel.qcs_dc_output_config(
        nominal_full_scale_v=experiment.qcs_dc_full_scale_v.value(),
    )
    assert config.nominal_full_scale_v == pytest.approx(2.5)

    monkeypatch.setattr(calibration_panel, "show_result", lambda _stored: None)
    monkeypatch.setattr(window, "_refresh_qcs_waveform_capacity", lambda: None)
    stored = SimpleNamespace(
        run_id=31,
        database_path="calibration.db",
        calibration=SimpleNamespace(
            gain_a=0.98,
            corrected_maximum_abs_voltage_v=2.45,
        ),
    )
    window._on_calibration_finished(stored)

    assert experiment.qcs_dc_full_scale_v.value() == pytest.approx(2.45)
    assert experiment.qcs_settings_dict()["dc_full_scale_v"] == pytest.approx(
        2.45
    )
    assert not hasattr(calibration_panel, "qcs_dc_full_scale_v")
    window.close()
    app.processEvents()


def test_qcs_saved_dc_name_reorder_changes_gui_bindings_not_native_names(
    tmp_path,
):
    app = _application()
    window = gui.MainWindow()
    window._add_port()
    panel = window._experiment_panel
    configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("left", "right"),
        {},
        None,
    )
    mapper_path = tmp_path / "two_dc.qcs"
    mapper_path.write_bytes(b"two channel mapper")
    settings = panel.qcs_settings_dict()
    settings.update(
        {
            "mapper_path": str(mapper_path),
            "dc_channel_names": ["left", "right"],
            "hardware_configuration": configuration,
            "hardware_configuration_state": "saved",
            "hardware_mapper_sha256": (
                qcs_front_panel.qcs_mapper_file_sha256(mapper_path)
            ),
        }
    )
    panel.set_qcs_settings(settings, 2)
    panel.qcs_dc_channel_names.setText("right, left")

    reordered = panel.qcs_settings_dict()
    assert reordered["hardware_configuration_state"] == "saved"
    dc_mappings = [
        mapping
        for mapping in reordered["hardware_configuration"][
            "channel_mappings"
        ]
        if mapping["role"] == "dc"
    ]
    assert [
        (
            mapping["logical_index"],
            mapping["virtual_name"],
            mapping["channel"],
        )
        for mapping in dc_mappings
    ] == [(0, "right", 2), (1, "left", 1)]
    window._add_port()
    assert panel.qcs_dc_channel_names.text() == "right, left, dc_ch_1"
    window._delete_port(0)
    assert panel.qcs_dc_channel_names.text() == "left, dc_ch_1"
    window.close()
    app.processEvents()


def test_qcs_anticipatory_added_name_preserves_reordered_physical_bindings(
    tmp_path,
):
    app = _application()
    window = gui.MainWindow()
    window._add_port()
    panel = window._experiment_panel
    configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("left", "right"),
        {},
        None,
    )
    mapper_path = tmp_path / "two_dc.qcs"
    mapper_path.write_bytes(b"two channel mapper")
    settings = panel.qcs_settings_dict()
    settings.update(
        {
            "mapper_path": str(mapper_path),
            "dc_channel_names": ["left", "right"],
            "hardware_configuration": configuration,
            "hardware_configuration_state": "saved",
            "hardware_mapper_sha256": (
                qcs_front_panel.qcs_mapper_file_sha256(mapper_path)
            ),
        }
    )
    panel.set_qcs_settings(settings, 2)

    panel.qcs_dc_channel_names.setText("right, left, third")
    window._add_port()

    assert panel.qcs_dc_channel_names.text() == "right, left, third"
    resized = panel.qcs_settings_dict()["hardware_configuration"]
    dc_mappings = [
        mapping
        for mapping in resized["channel_mappings"]
        if mapping["role"] == "dc"
    ]
    assert [
        (
            mapping["logical_index"],
            mapping["virtual_name"],
            mapping["channel"],
        )
        for mapping in dc_mappings
    ] == [(0, "right", 2), (1, "left", 1), (2, "third", 3)]
    window.close()
    app.processEvents()


def test_qcs_anticipatory_inserted_name_uses_new_physical_binding(tmp_path):
    app = _application()
    window = gui.MainWindow()
    window._add_port()
    panel = window._experiment_panel
    configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("left", "right"),
        {},
        None,
    )
    mapper_path = tmp_path / "two_dc.qcs"
    mapper_path.write_bytes(b"two channel mapper")
    settings = panel.qcs_settings_dict()
    settings.update(
        {
            "mapper_path": str(mapper_path),
            "dc_channel_names": ["left", "right"],
            "hardware_configuration": configuration,
            "hardware_configuration_state": "saved",
            "hardware_mapper_sha256": (
                qcs_front_panel.qcs_mapper_file_sha256(mapper_path)
            ),
        }
    )
    panel.set_qcs_settings(settings, 2)

    panel.qcs_dc_channel_names.setText("third, left, right")
    window._add_port()

    resized = panel.qcs_settings_dict()["hardware_configuration"]
    dc_mappings = [
        mapping
        for mapping in resized["channel_mappings"]
        if mapping["role"] == "dc"
    ]
    assert [
        (
            mapping["logical_index"],
            mapping["virtual_name"],
            mapping["channel"],
        )
        for mapping in dc_mappings
    ] == [(0, "third", 3), (1, "left", 1), (2, "right", 2)]
    window.close()
    app.processEvents()


def test_qcs_output_add_remove_reuses_imported_unassigned_m5301(tmp_path):
    app = _application()
    window = gui.MainWindow()
    panel = window._experiment_panel
    configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("left",),
        {},
        None,
    )
    configuration["channel_mappings"].append(
        {
            "role": "unassigned",
            "logical_index": 0,
            "virtual_name": "spare",
            "label": 0,
            "absolute_phase": False,
            "lo_frequency_hz": None,
            "slot": 2,
            "channel": 2,
        }
    )
    configuration = qcs_front_panel.normalize_qcs_hardware_configuration(
        configuration
    )
    mapper_path = tmp_path / "imported_spare.qcs"
    mapper_path.write_bytes(b"imported mapper with spare")
    digest = qcs_front_panel.qcs_mapper_file_sha256(mapper_path)
    settings = panel.qcs_settings_dict()
    settings.update(
        {
            "mapper_path": str(mapper_path),
            "dc_channel_names": ["left"],
            "hardware_configuration": configuration,
            "hardware_configuration_state": "imported",
            "hardware_mapper_sha256": digest,
        }
    )
    panel.set_qcs_settings(settings, 1)

    window._add_port()

    expanded = panel.qcs_settings_dict()
    assert panel.qcs_dc_channel_names.text() == "left, spare"
    assert expanded["hardware_configuration_state"] == "imported"
    assert expanded["hardware_mapper_sha256"] == digest
    assert qcs_front_panel.qcs_role_bindings(
        expanded["hardware_configuration"],
        required_dc_count=2,
    )[0] == ["left", "spare"]

    window._delete_port(1)

    contracted = panel.qcs_settings_dict()
    assert panel.qcs_dc_channel_names.text() == "left"
    assert contracted["hardware_configuration_state"] == "imported"
    assert contracted["hardware_mapper_sha256"] == digest
    assert any(
        mapping["role"] == "unassigned"
        and mapping["virtual_name"] == "spare"
        for mapping in contracted["hardware_configuration"][
            "channel_mappings"
        ]
    )
    window.close()
    app.processEvents()


def test_qcs_delete_does_not_mask_pending_native_name_change(tmp_path):
    app = _application()
    window = gui.MainWindow()
    window._add_port()
    panel = window._experiment_panel
    configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("left", "right"),
        {},
        None,
    )
    mapper_path = tmp_path / "saved_two_dc.qcs"
    mapper_path.write_bytes(b"saved mapper")
    settings = panel.qcs_settings_dict()
    settings.update(
        {
            "mapper_path": str(mapper_path),
            "dc_channel_names": ["left", "right"],
            "hardware_configuration": configuration,
            "hardware_configuration_state": "saved",
            "hardware_mapper_sha256": (
                qcs_front_panel.qcs_mapper_file_sha256(mapper_path)
            ),
        }
    )
    panel.set_qcs_settings(settings, 2)
    panel.qcs_dc_channel_names.setText("right, new")

    window._delete_port(0)

    current = panel.qcs_settings_dict()
    assert current["dc_channel_names"] == ["new"]
    assert current["hardware_configuration_state"] == "draft"
    assert current["hardware_mapper_sha256"] is None
    with pytest.raises(ValueError, match="unsaved physical changes"):
        panel.qcs_connection_values(1)
    window.close()
    app.processEvents()


def test_adding_output_uses_known_qcs_bindings_not_dormant_invalid_text(
    tmp_path,
):
    app = _application()
    window = gui.MainWindow()
    panel = window._experiment_panel
    configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("dc_ch_1",),
        {},
        None,
    )
    mapper_path = tmp_path / "known_mapper.qcs"
    mapper_path.write_bytes(b"known mapper")
    settings = panel.qcs_settings_dict()
    settings.update(
        {
            "mapper_path": str(mapper_path),
            "hardware_configuration": configuration,
            "hardware_configuration_state": "saved",
            "hardware_mapper_sha256": (
                qcs_front_panel.qcs_mapper_file_sha256(mapper_path)
            ),
        }
    )
    panel.set_qcs_settings(settings, 1)
    panel.qcs_dc_channel_names.setText("dup, dup")

    window._add_port()
    updated = panel.qcs_settings_dict()
    assert len(window._pulse) == 2
    assert updated["dc_channel_names"] == ["dc_ch_1", "dc_ch_2"]
    assert "dc_channel_names_text" not in updated
    assert updated["hardware_configuration_state"] == "draft"
    window.close()
    app.processEvents()


def test_dormant_invalid_qcs_text_survives_qick_settings_round_trip():
    app = _application()
    source = gui.MainWindow()
    panel = source._experiment_panel
    panel.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    panel.qcs_dc_channel_names.setText("gate_a, gate_a")
    panel.qcs_rf_channel_names.setText("not-a-generator-mapping")

    document = source._settings_to_dict()
    assert document["experiment"]["execution_backend"] == "qick"
    assert document["qcs"]["dc_channel_names"] == []
    assert document["qcs"]["rf_channel_names"] == {}
    assert document["qcs"]["dc_channel_names_text"] == "gate_a, gate_a"
    assert (
        document["qcs"]["rf_channel_names_text"]
        == "not-a-generator-mapping"
    )
    assert document["qcs"]["hardware_configuration"] is None

    restored = gui.MainWindow()
    restored._apply_decoded_settings(restored._decode_settings(document))
    assert (
        restored._experiment_panel.qcs_dc_channel_names.text()
        == "gate_a, gate_a"
    )
    assert (
        restored._experiment_panel.qcs_rf_channel_names.text()
        == "not-a-generator-mapping"
    )
    restored.close()
    source.close()
    app.processEvents()


def test_dormant_valid_qcs_role_shape_edit_preserves_known_recipe(tmp_path):
    app = _application()
    window = gui.MainWindow()
    panel = window._experiment_panel
    configuration = qcs_front_panel.default_qcs_hardware_configuration(
        ("dc_ch_1",),
        {},
        None,
    )
    mapper_path = tmp_path / "known_mapper.qcs"
    mapper_path.write_bytes(b"known mapper")
    settings = panel.qcs_settings_dict()
    settings.update(
        {
            "mapper_path": str(mapper_path),
            "hardware_configuration": configuration,
            "hardware_configuration_state": "saved",
            "hardware_mapper_sha256": (
                qcs_front_panel.qcs_mapper_file_sha256(mapper_path)
            ),
        }
    )
    panel.set_qcs_settings(settings, 1)
    panel.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    panel.qcs_rf_channel_names.setText("0=rf_drive")

    document = window._settings_to_dict()
    assert document["qcs"]["rf_channel_names"] == {}
    assert document["qcs"]["rf_channel_names_text"] == "0=rf_drive"
    assert document["qcs"]["hardware_configuration"] == configuration
    assert document["qcs"]["hardware_configuration_state"] == "saved"

    panel.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    with pytest.raises(ValueError, match="roles differ"):
        panel.qcs_settings_dict()
    panel.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    window._show_qcs_front_panel()
    window._qcs_front_panel.apply_settings()
    refreshed = panel.qcs_settings_dict()
    assert panel.qcs_rf_channel_names.text() == ""
    assert "rf_channel_names_text" not in refreshed
    window.close()
    app.processEvents()


def test_qcs_front_panel_closes_when_waveform_output_count_changes():
    app = _application()
    window = gui.MainWindow()
    window._show_qcs_front_panel()
    app.processEvents()
    assert window._qcs_front_panel_dialog.isVisible() is True

    window._add_port()
    app.processEvents()
    assert window._qcs_front_panel_dialog.isVisible() is False
    window.close()
    app.processEvents()


def test_settings_decoder_rejects_divergent_qcs_role_bindings():
    app = _application()
    window = gui.MainWindow()
    document = window._settings_to_dict()
    document["qcs"]["hardware_configuration"] = (
        qcs_front_panel.default_qcs_hardware_configuration(
            ("different_dc_name",),
            {},
            None,
        )
    )
    document["qcs"]["hardware_configuration_state"] = "draft"

    with pytest.raises(ValueError, match="role bindings must exactly match"):
        window._decode_settings(document)
    window.close()
    app.processEvents()


@pytest.mark.parametrize(
    ("old_index", "panel_name"),
    ((1, "_sparameter_panel"), (2, "_calibration_panel")),
)
def test_version_12_top_level_tabs_migrate_after_stability_tab_insertion(
    old_index,
    panel_name,
):
    app = _application()
    window = gui.MainWindow()
    document = window._settings_to_dict()
    document["version"] = 12
    document["display"]["selected_control_tab"] = old_index
    document.pop("stability_diagram")

    decoded = window._decode_settings(document)
    window._apply_decoded_settings(decoded)
    app.processEvents()

    assert window._control_tabs.currentWidget() is getattr(window, panel_name)
    window.close()


def test_older_settings_apply_defaults_and_resave_as_current(tmp_path):
    app = _application()
    window = gui.MainWindow()
    document = window._settings_to_dict()
    document["version"] = 2
    document["display"]["selected_control_tab"] = 2
    document["display"].pop("selected_awg_tuning_tab")
    document["display"].pop("voltage_view")
    document["grid"].pop("snap_enabled")
    document["awg"].pop("cross_capacitance")
    document["awg"].pop("sweeps")
    document["qick"].pop("tproc_mhz")
    document["qick"].pop("repetitions_per_sweep")
    document["qick"].pop("bias_t_compensation")
    document["experiment"].pop("notes")
    document.pop("rf_outputs")
    document["rf_readout"] = {"enabled": False}
    document["s_parameter"].pop("database_path")

    old_path = tmp_path / "settings_v2.json"
    old_path.write_text(json.dumps(document), encoding="utf-8")
    window._load_settings_json(old_path)
    app.processEvents()

    assert np.array_equal(window._cross_capacitance, np.eye(1))
    assert window._plot.voltage_view == "both"
    assert window._grid_snap_enabled is False
    assert window._experiment_panel.tproc_mhz.value() == 300.0
    assert window._experiment_panel.repetitions.value() == 1
    assert window._experiment_panel.bias_t_group.isChecked() is False
    assert window._experiment_panel.bias_t_compensation_mv.value() == 80.0
    assert window._control_tabs.currentWidget() is window._awg_tuning_page
    assert window._awg_tuning_tabs.currentWidget() is window._rf_readout_panel
    assert len(window._rf_ports_panel._panels) == 1
    assert window._rf_ports_panel.settings()[0] == gui.DEFAULT_RF_OUTPUT_SETTINGS
    assert (
        window._rf_readout_panel.settings_dict()
        == gui.DEFAULT_RF_READOUT_SETTINGS
    )
    assert (
        window._sparameter_panel.settings_dict()
        == gui.DEFAULT_SPARAMETER_SETTINGS
    )

    upgraded_path = window._save_settings_json(tmp_path / "settings_upgraded")
    upgraded = json.loads(upgraded_path.read_text(encoding="utf-8"))
    assert upgraded["version"] == gui.SETTINGS_VERSION == 42
    assert upgraded["qick"]["awg_metadata_mode"] == "parametric"
    assert upgraded["qick"]["compile_validation_mode"] == "boundary"
    assert upgraded["display"]["selected_control_tab"] == 0
    assert upgraded["display"]["selected_awg_tuning_tab"] == 2
    assert upgraded["display"]["voltage_view"] == "both"
    assert upgraded["grid"]["snap_enabled"] is False
    assert upgraded["awg"]["cross_capacitance"] == [[1.0]]
    assert upgraded["awg"]["sweeps"] == []
    assert upgraded["stability_diagram"]["x_axis"]["output_name"] == "awg_0"
    assert upgraded["stability_diagram"]["y_axis"]["output_name"] == "awg_0"
    assert upgraded["stability_diagram"]["trace_samples_per_point"] == 64
    assert upgraded["stability_diagram"]["bias_t_compensation"] == {
        "enabled": False,
        "type": "dc",
        "mode": "fixed_voltage",
        "voltage_mv": 80.0,
        "duration_us": 1.0,
        "filter_tau_us": 100.0,
    }
    assert "rf_readout" not in upgraded["stability_diagram"]
    assert "rf_outputs" not in upgraded["stability_diagram"]
    assert upgraded["qick"]["tproc_mhz"] == 300.0
    assert upgraded["qick"]["repetitions_per_sweep"] == 1
    assert upgraded["qick"]["bias_t_compensation"] == {
        "enabled": False,
        "type": "dc",
        "mode": "fixed_voltage",
        "voltage_mv": 80.0,
        "duration_us": 1.0,
        "filter_tau_us": 100.0,
    }
    assert upgraded["experiment"]["notes"] == ""
    assert upgraded["experiment"]["iq_repetition_policy"] == (
        gui.IQ_REPETITION_POLICY_PRESERVE
    )
    assert upgraded["rf_outputs"] == [gui.DEFAULT_RF_OUTPUT_SETTINGS]
    assert upgraded["rf_readout"] == gui.DEFAULT_RF_READOUT_SETTINGS
    assert upgraded["rf_readout"]["dc_measure_mode"] is False
    assert upgraded["rf_readout"]["dc_measure_gain_v_per_a"] == 1.0
    assert (
        upgraded["s_parameter"]["database_path"]
        == gui.DEFAULT_SPARAMETER_SETTINGS["database_path"]
    )
    window.close()


def test_bias_t_gui_extends_and_plots_the_physical_awg_trace():
    app = _application()
    window = gui.MainWindow()
    original_end_ns = float(window._plot._physical_time_ns[-1])
    window._experiment_panel.bias_t_compensation_mv.setValue(100.0)
    window._experiment_panel.bias_t_group.setChecked(True)
    app.processEvents()

    assert window._bias_t_compensation_enabled is True
    assert window._plot._physical_time_ns[-1] > original_end_ns
    assert np.min(window._plot._physical_values_mv[0]) == -100.0
    assert window._plot._physical_values_mv[0, -1] == 0.0
    window.close()


def test_bias_t_gui_fixed_time_disables_voltage_and_adjusts_preview_level():
    app = _application()
    window = gui.MainWindow()
    panel = window._experiment_panel
    panel.bias_t_mode.setCurrentIndex(panel.bias_t_mode.findData("fixed_time"))
    panel.bias_t_duration_us.setValue(2.0)
    panel.bias_t_group.setChecked(True)
    app.processEvents()

    assert panel.bias_t_compensation_mv.isEnabled() is False
    assert panel.bias_t_duration_us.isEnabled() is True
    assert window._bias_t_compensation_mode == "fixed_time"
    assert window._bias_t_compensation_duration_us == 2.0
    assert np.min(window._plot._physical_values_mv[0]) == pytest.approx(
        -50.0,
        abs=0.2,
    )
    assert window._plot._physical_values_mv[0, -1] == 0.0
    window.close()


def test_bias_t_gui_filter_mode_enables_tau_and_slopes_flat_segment(tmp_path):
    app = _application()
    window = gui.MainWindow()
    panel = window._experiment_panel
    panel.bias_t_type.setCurrentIndex(panel.bias_t_type.findData("filter"))
    panel.bias_t_filter_tau_us.setValue(50.0)
    panel.bias_t_group.setChecked(True)
    app.processEvents()

    assert panel.bias_t_mode.isEnabled() is False
    assert panel.bias_t_compensation_mv.isEnabled() is False
    assert panel.bias_t_duration_us.isEnabled() is False
    assert panel.bias_t_filter_tau_us.isEnabled() is True
    assert window._bias_t_compensation_type == "filter"
    assert window._bias_t_filter_tau_us == 50.0
    assert window._plot._physical_values_mv[0, 0] == pytest.approx(100.0)
    assert window._plot._physical_values_mv[0, -1] == pytest.approx(102.0)

    sequence = window._experiment_run_arguments(
        require_readout=False,
        require_run_config=False,
    )["sequence"]
    assert sequence.bias_t_compensation.compensation_type == "filter"
    assert sequence.bias_t_compensation.tau_cycles == 15_000.0

    settings_path = window._save_settings_json(tmp_path / "filter_compensation")
    restored = gui.MainWindow()
    restored._load_settings_json(settings_path)
    app.processEvents()
    assert restored._experiment_panel.bias_t_type.currentData() == "filter"
    assert restored._experiment_panel.bias_t_filter_tau_us.value() == 50.0
    assert restored._bias_t_compensation_type == "filter"
    assert restored._bias_t_filter_tau_us == 50.0
    restored.close()
    window.close()


def test_generated_qick_filter_compensation_preserves_tau_configuration():
    pulse = PulseSequence(
        initial_voltage=100.0,
        initial_duration_ns=1_000.0,
    )
    code = generate_qick_program_code(
        (pulse,),
        output_names=("awg_0",),
        awg_channels=(1,),
        fabric_mhz=300.0,
        bias_t_compensation_enabled=True,
        bias_t_compensation_type="filter",
        bias_t_filter_tau_us=50.0,
    )
    ast.parse(code)
    namespace = {}
    exec(compile(code, "<filter-comp-generated>", "exec"), namespace)

    assert namespace["BIAS_T_COMPENSATION_TYPE"] == "filter"
    assert namespace["BIAS_T_FILTER_TAU_US"] == 50.0
    assert namespace["BIAS_T_FILTER_TAU_CYCLES"] == 15_000.0
    config = namespace["build_sequence"]().bias_t_compensation
    assert config.compensation_type == "filter"
    assert config.tau_cycles == 15_000.0


def test_partial_legacy_rf_settings_fill_nested_defaults():
    app = _application()
    window = gui.MainWindow()
    document = window._settings_to_dict()
    document["version"] = 1
    document["rf_outputs"] = [{"gen_ch": 4, "segment_name": "set_0"}]
    document["rf_readout"] = {"ro_ch": 2, "segment_name": "set_0"}

    decoded = window._decode_settings(document)
    rf_output = decoded["rf_outputs"][0]
    assert rf_output["enabled"] is True
    assert rf_output["gen_ch"] == 4
    assert rf_output["duration_us"] == 1.0
    assert rf_output["frequency_mhz"] == 50.0
    assert rf_output["gain"] == 20000
    assert rf_output["filter_type"] == "bypass"
    assert rf_output["filter_cutoff"] == 2.5
    assert rf_output["filter_bandwidth"] == 1.0
    assert decoded["rf_readout"] == {
        **gui.DEFAULT_RF_READOUT_SETTINGS,
        "ro_ch": 2,
    }
    app.processEvents()
    window.close()
