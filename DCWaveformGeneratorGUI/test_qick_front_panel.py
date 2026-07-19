"""Tests for live HWH-to-front-panel channel discovery and GUI synchronization.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets
import pytest

import DCWaveform_Generator as gui
from qick_front_panel import QickFrontPanelControl, identify_qick_front_panel


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
    panel.close()


def test_graphical_path_updates_sparameter_experiment_and_calibration():
    app = _application()
    window = gui.MainWindow()
    panel = window._qick_front_panel
    assert window._control_tabs.currentWidget() is panel

    panel.set_configuration(identify_qick_front_panel(_live_config()))
    panel.canvas.select_port("output", 0)
    panel.canvas.select_port("input", 2)
    panel.output_att1_db.setValue(6.25)
    panel.output_att2_db.setValue(8.5)
    panel.input_attenuation_db.setValue(11.75)
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

    experiment_output = window._rf_ports_panel._panels[0]
    assert experiment_output.gen_ch.value() == 0
    assert experiment_output.att1_db.value() == 6.25
    assert window._rf_readout_panel.ro_ch.value() == 1
    assert window._rf_readout_panel.attenuation_db.value() == 11.75

    calibration = window._calibration_panel
    assert calibration.output_ch.value() == 0
    assert calibration.output_att1.value() == 6.25
    assert calibration.input_output_ch.value() == 0
    assert calibration.input_readout_ch.value() == 1
    assert calibration.input_attenuation.value() == 11.75
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
