"""Headless tests for GUI time units and RF port/readout integration.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import ast
import json
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pytest

import DCWaveform_Generator as gui
from dc_waveform_core import (
    PulseSequence,
    QickDdrReadoutSpec,
    QickRfPulseSpec,
    QickSweepSpec,
    adc_iq_to_voltage,
    dc_iq_to_current,
    generate_qick_program_code,
)
from stability_diagram import DEFAULT_STABILITY_POINT_GUARD_US


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


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
    assert window._experiment_panel.bias_t_group.isChecked() is False
    assert window._experiment_panel.bias_t_type.currentData() == "dc"
    assert window._experiment_panel.bias_t_mode.currentData() == "fixed_voltage"
    assert window._experiment_panel.bias_t_compensation_mv.value() == 250.0
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
    assert control.table.horizontalHeaderItem(2).text() == "Flat [ns]"

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
        "RF Readout",
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


def test_rf_readout_panel_builds_analog_input_and_ddr_settings():
    app = _application()
    window = gui.MainWindow()
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
    assert calibration_row == -1

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
    panel.dc_measure_mode.setChecked(True)
    panel.dc_measure_gain_v_per_a.setValue(2.0)
    app.processEvents()

    spec = panel.spec()
    assert spec is not None
    assert spec.input_board_type == "DC_In"
    assert spec.dc_measure_mode is True
    assert spec.dc_measure_gain_v_per_a == 2.0
    assert spec.measurement_unit == "A"
    assert panel.dc_measure_gain_v_per_a.isEnabled() is True

    panel.input_board_type.setCurrentText("RF_In")
    app.processEvents()
    assert panel.dc_measure_mode.isChecked() is False
    assert panel.dc_measure_mode.isEnabled() is False
    assert panel.dc_measure_gain_v_per_a.isEnabled() is False
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
    assert bias_t_config.amplitude == 0.05
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
    window._add_port()
    window._pulse[1].v[:] = [300.0, 300.0, 450.0, 450.0]
    window._cross_capacitance = np.asarray(
        ((1.0, 0.2), (-0.15, 1.0)), dtype=float
    )
    window._sweep_specs = [
        QickSweepSpec("set_1", "awg_0", -0.4, 0.6, 7)
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
    app.processEvents()

    expected = window._settings_to_dict()
    saved_path = window._save_settings_json(tmp_path / "complete_experiment")
    assert saved_path.suffix == ".json"
    document = json.loads(saved_path.read_text(encoding="utf-8"))
    assert document["schema"] == gui.SETTINGS_SCHEMA
    assert document["version"] == gui.SETTINGS_VERSION
    assert document["qick"]["tproc_mhz"] == 275.0
    assert document["qick"]["bias_t_compensation"] == {
        "enabled": True,
        "type": "dc",
        "mode": "fixed_time",
        "voltage_mv": 125.0,
        "duration_us": 2.5,
        "filter_tau_us": 100.0,
    }
    assert len(document["awg"]["outputs"]) == 2
    assert len(document["rf_outputs"]) == 2
    assert document["rf_outputs"][0]["filter_type"] == "highpass"
    assert document["rf_outputs"][0]["filter_cutoff"] == 1.75
    assert document["rf_outputs"][0]["filter_bandwidth"] == 0.625
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
    assert document["stability_diagram"]["bias_t_compensation"] == {
        "enabled": True,
        "type": "filter",
        "mode": "fixed_voltage",
        "voltage_mv": 250.0,
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
    assert restored._control_tabs.currentWidget() is restored._awg_tuning_page
    assert restored._awg_tuning_tabs.currentWidget() is restored._rf_readout_panel
    assert restored._experiment_panel.database_path.text().endswith("experiment.db")
    window.close()
    restored.close()


def test_experiment_panel_builds_hardware_run_snapshot(tmp_path):
    app = _application()
    window = gui.MainWindow()
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
    assert arguments["readout_spec"].samples_per_trigger == 32
    assert arguments["gui_settings"]["qick"]["tproc_mhz"] == 300.0
    assert arguments["sequence"].bias_t_compensation.amplitude == 0.08
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
    assert arguments["readout_spec"].fpga_trigger_delay_samples == 0
    assert arguments["stability_fabric_mhz"] == 300.0
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
    assert window._experiment_panel.bias_t_compensation_mv.value() == 250.0
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
    assert upgraded["version"] == gui.SETTINGS_VERSION == 27
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
        "voltage_mv": 250.0,
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
        "voltage_mv": 250.0,
        "duration_us": 1.0,
        "filter_tau_us": 100.0,
    }
    assert upgraded["experiment"]["notes"] == ""
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
