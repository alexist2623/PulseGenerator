"""Tests for live HWH-to-front-panel channel discovery and GUI synchronization.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets
import pytest

import DCWaveform_Generator as gui
from qick_front_panel import (
    QickFrontPanelControl,
    QickFrontPanelPreview,
    identify_qick_front_panel,
)


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _live_config():
    return {
        "board": "ZCU216",
        "fw_timestamp": "2026-07-19 12:34:56",
        "extra_description": [
            "\nQICK box daughter cards detected:",
            "\tDAC slot 0: RF Out card has ports [0, 1, 2, 3]",
            "\tDAC slot 1: DC Out card has ports [4, 5, 6, 7]",
            "\tDAC slot 2: No card detected",
            "\tDAC slot 3: RF Out card has ports [12, 13, 14, 15]",
            "\tADC slot 0: DC In card has ports [0, 1]",
            "\tADC slot 1: RF In card has ports [2, 3]",
            "\tADC slot 2: No card detected",
            "\tADC slot 3: RF In card has ports [6, 7]",
        ],
        "gens": [
            {"dac": "00", "fullpath": "axis_signal_gen_v6_0"},
            {"dac": "10", "fullpath": "axis_awg_tuning_v1_0"},
            {"dac": "33", "fullpath": "axis_signal_gen_v6_15"},
        ],
        "readouts": [
            {"adc": "10", "avgbuf_fullpath": "axis_avg_buffer_0"},
            {"adc": "12", "avgbuf_fullpath": "axis_avg_buffer_2"},
            {"adc": "23", "avgbuf_fullpath": "axis_avg_buffer_7"},
        ],
    }


def test_hwh_and_card_detection_build_zcu216_physical_port_map():
    configuration = identify_qick_front_panel(_live_config())

    assert configuration.board == "ZCU216"
    assert configuration.firmware_timestamp == "2026-07-19 12:34:56"
    assert configuration.port("output", 0).qick_channels == (0,)
    assert configuration.port("output", 0).converter_id == "00"
    assert configuration.port("output", 0).board_type == "RF_Out"
    assert configuration.port("output", 4).qick_channels == (1,)
    assert configuration.port("output", 4).converter_id == "10"
    assert configuration.port("output", 4).board_type == "DC_Out"
    assert configuration.port("output", 15).qick_channels == (2,)
    assert configuration.port("output", 8).board_type is None

    assert configuration.port("input", 0).qick_channels == (0,)
    assert configuration.port("input", 0).converter_id == "10"
    assert configuration.port("input", 0).board_type == "DC_In"
    assert configuration.port("input", 2).qick_channels == (1,)
    assert configuration.port("input", 2).converter_id == "12"
    assert configuration.port("input", 2).board_type == "RF_In"
    assert configuration.port("input", 7).qick_channels == (2,)


def test_front_panel_control_selects_smas_and_reports_channel_attenuation():
    app = _application()
    panel = QickFrontPanelControl()
    panel.set_path_values(
        {
            "output_ch": 1,
            "readout_ch": 1,
            "output_att1_db": 3.25,
            "output_att2_db": 4.5,
            "output_nqz": 2,
            "readout_nqz": 2,
            "readout_attenuation_db": 7.25,
            "readout_dc_gain_db": 12.0,
        }
    )
    panel.set_configuration(identify_qick_front_panel(_live_config()))
    app.processEvents()

    assert panel.output_sma.text() == "DAC4"
    assert panel.output_channel.currentData() == 1
    assert panel.output_board.text() == "DC Out"
    assert panel.output_att1_db.isEnabled() is False
    assert panel.input_sma.text() == "ADC2"
    assert panel.input_channel.currentData() == 1
    assert panel.input_board.text() == "RF In"
    assert panel.input_condition_stack.currentWidget() is panel.input_attenuation_db
    assert panel.output_nqz.value() == 2
    assert panel.input_nqz.value() == 2
    assert "QICK gen 1" in panel.summary.text()
    assert "ATT 7.25 dB" in panel.summary.text()

    panel.canvas.select_port("output", 0)
    app.processEvents()
    assert panel.output_sma.text() == "DAC0"
    assert panel.output_channel.currentData() == 0
    assert panel.output_att1_db.isEnabled() is True

    applied = []
    panel.settings_applied.connect(applied.append)
    panel.apply_button.click()
    app.processEvents()
    assert applied[0]["output_ch"] == 0
    assert applied[0]["readout_ch"] == 1
    assert applied[0]["output_board_type"] == "RF_Out"
    assert applied[0]["input_board_type"] == "RF_In"
    assert applied[0]["readout_attenuation_db"] == 7.25
    assert applied[0]["output_nqz"] == 2
    assert applied[0]["readout_nqz"] == 2
    panel.close()


def test_graphical_path_updates_sparameter_experiment_and_calibration():
    app = _application()
    window = gui.MainWindow()
    panel = window._qick_front_panel
    assert window._control_tabs.indexOf(panel) == -1
    assert window._control_tabs.currentWidget() is window._awg_tuning_page
    assert window._awg_tuning_tabs.currentWidget() is window._multi_ctrl

    panel.set_configuration(identify_qick_front_panel(_live_config()))
    panel.canvas.select_port("output", 0)
    panel.canvas.select_port("input", 2)
    panel.output_att1_db.setValue(6.25)
    panel.output_att2_db.setValue(8.5)
    panel.input_attenuation_db.setValue(11.75)
    panel.output_nqz.setValue(2)
    panel.input_nqz.setValue(2)
    panel.apply_button.click()
    app.processEvents()

    sparameter = window._sparameter_panel.config()
    assert sparameter.output_ch == 0
    assert sparameter.readout_ch == 1
    assert sparameter.output_board_type == "RF_Out"
    assert sparameter.input_board_type == "RF_In"
    assert sparameter.output_att1_db == 6.25
    assert sparameter.output_att2_db == 8.5
    assert sparameter.readout_attenuation_db == 11.75
    assert sparameter.nqz == 2
    assert sparameter.readout_nqz == 2

    experiment_output = window._rf_ports_panel._panels[0]
    assert experiment_output.gen_ch.value() == 0
    assert experiment_output.att1_db.value() == 6.25
    assert window._rf_readout_panel.ro_ch.value() == 1
    assert window._rf_readout_panel.attenuation_db.value() == 11.75
    assert window._rf_readout_panel.nqz.value() == 2

    calibration = window._calibration_panel
    assert calibration.output_ch.value() == 0
    assert calibration.output_att1.value() == 6.25
    assert calibration.input_output_ch.value() == 0
    assert calibration.input_readout_ch.value() == 1
    assert calibration.input_attenuation.value() == 11.75
    window.close()


def test_front_panel_visual_order_is_physical_descending_order():
    app = _application()
    preview = QickFrontPanelPreview()
    preview.set_configuration(identify_qick_front_panel(_live_config()))
    preview.show()
    app.processEvents()

    assert preview._port_centers[("output", 15)].x() < preview._port_centers[("output", 0)].x()
    assert preview._port_centers[("input", 7)].x() < preview._port_centers[("input", 0)].x()
    activated = []
    preview.activated.connect(lambda: activated.append(True))
    QtTest.QTest.mouseClick(preview, QtCore.Qt.LeftButton)
    assert activated == [True]
    preview.close()


def test_output_preview_opens_scoped_dialog_and_applies_hwh_board_settings():
    app = _application()
    window = gui.MainWindow()
    configuration = identify_qick_front_panel(_live_config())
    window._on_qick_configuration_identified(configuration)
    output = window._rf_ports_panel._panels[0]
    output.gen_ch.setValue(1)
    app.processEvents()

    assert output.output_board_type.currentText() == "DC_Out"
    assert output.filter_type.isEnabled() is False

    output.gen_ch.setValue(0)
    window._show_qick_front_panel("output", output)
    app.processEvents()
    panel = window._qick_front_panel
    assert panel.scope == "output"
    assert panel.output_group.isVisible()
    assert panel.input_group.isVisible() is False
    panel.canvas.select_port("output", 0)
    panel.output_att1_db.setValue(5.25)
    panel.output_att2_db.setValue(7.5)
    panel.output_filter_type.setCurrentText("lowpass")
    panel.apply_button.click()
    app.processEvents()

    assert output.gen_ch.value() == 0
    assert output.output_board_type.currentText() == "RF_Out"
    assert output.att1_db.value() == 5.25
    assert output.att2_db.value() == 7.5
    assert output.filter_type.currentText() == "lowpass"
    assert window._qick_front_panel_dialog.isVisible() is False
    window.close()


def test_sparameter_path_uses_compact_preview_and_update_below_dut():
    _application()
    window = gui.MainWindow()
    path = window._sparameter_panel.path_diagram
    layout = path.layout()
    update_position = layout.getItemPosition(layout.indexOf(path.update_button))
    preview_position = layout.getItemPosition(layout.indexOf(path.front_panel_preview))

    assert not hasattr(path, "summary")
    assert layout.horizontalSpacing() == 10
    assert preview_position == (0, 1, 1, 1)
    assert update_position == (5, 0, 1, 3)
    window.close()


def test_configuration_worker_uses_live_qick_config(monkeypatch):
    app = _application()
    monkeypatch.setattr(gui, "connect_qick", lambda _connection: (object(), _live_config()))
    worker = gui.QickConfigurationWorker(object())
    finished = []
    failed = []
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)
    worker.run()
    app.processEvents()

    assert failed == []
    assert finished[0].mapped_output_count == 3
    assert finished[0].mapped_input_count == 3


def test_front_panel_rejects_non_zcu216_mapping():
    with pytest.raises(ValueError, match="supports ZCU216"):
        identify_qick_front_panel({"board": "ZCU111"})
