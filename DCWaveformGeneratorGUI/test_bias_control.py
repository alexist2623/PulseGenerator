"""Tests for DAC11001 bias controls and front-panel selection.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest, QtWidgets
import pytest

import bias_control
import DCWaveform_Generator as gui
from bias_control import BiasControlPanel, BiasHardwareWorker
from qick_front_panel import identify_qick_front_panel
from qick_qcodes_experiment import QickConnectionConfig


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _live_config():
    return {
        "board": "ZCU216",
        "fw_timestamp": "2026-07-30 12:00:00",
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
        ],
        "readouts": [
            {"adc": "10", "avgbuf_fullpath": "axis_avg_buffer_0"},
            {"adc": "12", "avgbuf_fullpath": "axis_avg_buffer_2"},
        ],
    }


def test_bias_front_panel_click_selects_and_highlights_editor():
    app = _application()
    panel = BiasControlPanel()
    panel.set_configuration(identify_qick_front_panel(_live_config()))
    panel.resize(1200, 700)
    panel.show()
    app.processEvents()
    scale, offset_x, offset_y, origin_x, origin_y = (
        panel.front_panel._display_transform()
    )
    logical = panel.front_panel._port_centers[("bias", 5)]
    click_point = QtCore.QPoint(
        round((logical.x() - origin_x) * scale + offset_x),
        round((logical.y() - origin_y) * scale + offset_y),
    )
    QtTest.QTest.mouseClick(
        panel.front_panel,
        QtCore.Qt.LeftButton,
        pos=click_point,
    )
    app.processEvents()

    assert panel.selected_channel == 5
    assert panel.front_panel._selected_bias == 5
    assert bool(panel.editors[5].property("selected")) is True
    assert bool(panel.editors[4].property("selected")) is False
    assert "4px solid" in panel.editors[5].styleSheet()
    assert "BIAS5" in panel.status.text()
    panel.close()


def test_bias_settings_round_trip_without_touching_hardware():
    app = _application()
    panel = BiasControlPanel()
    settings = {
        "selected_channel": 6,
        "voltage_limit_v": 4.0,
        "channel_names": ["P", "BL", "BR", "S0", "AccL", "AccR", "", ""],
        "setpoints_v": [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
    }
    panel.load_settings(settings)
    app.processEvents()

    persisted = panel.settings_dict()
    for key, value in settings.items():
        assert persisted[key] == value
    assert set(persisted["measurements"]) == {
        "two_point",
        "gate",
        "wall_wall",
        "nested",
    }
    assert panel.editors[6].voltage.value() == pytest.approx(3.0)
    assert panel.editors[1].channel_name == "BL"
    panel.close()


def test_bias_channel_name_is_editable_and_used_in_status():
    app = _application()
    panel = BiasControlPanel()
    panel.select_channel(2, focus=False)
    panel.editors[2].name_edit.setText("BR")
    panel.editors[2]._name_edited("BR")
    app.processEvents()

    assert panel.channel_description(2) == "BIAS2 (BR)"
    assert "BIAS2 (BR)" in panel.status.text()
    assert panel.settings_dict()["channel_names"][2] == "BR"
    panel.close()


def test_bias_voltage_limit_updates_every_editor_and_rejects_loaded_overage():
    app = _application()
    panel = BiasControlPanel()
    panel.set_voltage_limit(1.25)

    for editor in panel.editors:
        assert editor.voltage.minimum() == pytest.approx(-1.25)
        assert editor.voltage.maximum() == pytest.approx(1.25)
    panel.editors[2].voltage.setValue(2.0)
    assert panel.editors[2].voltage.value() == pytest.approx(1.25)
    assert panel.settings_dict()["voltage_limit_v"] == pytest.approx(1.25)

    with pytest.raises(ValueError, match="absolute values"):
        panel.load_settings(
            {
                "voltage_limit_v": 1.0,
                "setpoints_v": [0.0, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
    app.processEvents()
    panel.close()


def test_bias_busy_state_does_not_move_selection_to_last_editor():
    app = _application()
    panel = BiasControlPanel()
    panel.resize(1200, 700)
    panel.show()
    panel.select_channel(3)
    panel.editors[3].voltage.setFocus()
    app.processEvents()

    panel.set_busy(True, "Applying BIAS3...")
    app.processEvents()
    assert panel.selected_channel == 3
    assert bool(panel.editors[3].property("selected")) is True
    assert bool(panel.editors[7].property("selected")) is False

    panel.set_busy(False, "BIAS3 applied")
    app.processEvents()
    assert panel.selected_channel == 3
    panel.close()


def test_bias_voltage_limit_is_locked_for_measurement_duration():
    app = _application()
    panel = BiasControlPanel()
    panel.set_measurement_running("nested", True, "Running nested sweep")
    app.processEvents()
    assert panel.voltage_limit.isEnabled() is False

    panel.set_measurement_running("nested", False, "Stopped")
    app.processEvents()
    assert panel.voltage_limit.isEnabled() is True
    panel.close()


def test_bias_worker_reads_and_sets_dac11001(monkeypatch):
    class FakeSoc:
        def __init__(self):
            self.values = {channel: channel / 10.0 for channel in range(8)}

        def rfb_get_bias(self, channel):
            return self.values[channel]

        def rfb_set_bias(self, channel, voltage):
            self.values[channel] = float(voltage)
            return float(voltage)

    soc = FakeSoc()
    monkeypatch.setattr(
        bias_control,
        "connect_qick",
        lambda _connection: (soc, {}),
    )
    connection = QickConnectionConfig(
        host="127.0.0.1",
        ns_port=8888,
        proxy_name="myqick",
    )

    read_results = []
    reader = BiasHardwareWorker(connection, "read")
    reader.finished.connect(read_results.append)
    reader.run()
    assert read_results[0][7] == pytest.approx(0.7)

    set_results = []
    writer = BiasHardwareWorker(connection, "set", {2: -1.25, 6: 3.5})
    writer.finished.connect(set_results.append)
    writer.run()
    assert set_results[0] == {2: pytest.approx(-1.25), 6: pytest.approx(3.5)}
    assert soc.values[2] == pytest.approx(-1.25)
    assert soc.values[6] == pytest.approx(3.5)

    with pytest.raises(ValueError, match="exceeds"):
        BiasHardwareWorker(
            connection,
            "set",
            {1: 1.01},
            voltage_limit_v=1.0,
        )


def test_main_window_contains_bias_tab_and_persists_setpoints():
    app = _application()
    window = gui.MainWindow()
    bias_index = window._control_tabs.indexOf(window._bias_panel)
    assert bias_index >= 0
    assert window._control_tabs.tabText(bias_index) == "Bias"

    window._bias_panel.load_settings(
        {
            "selected_channel": 3,
            "channel_names": ["P", "BL", "BR", "S0", "AccL", "AccR", "", ""],
            "setpoints_v": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        }
    )
    document = window._settings_to_dict()
    assert document["version"] == 35
    assert document["bias"]["selected_channel"] == 3
    assert document["bias"]["voltage_limit_v"] == pytest.approx(10.0)
    assert document["bias"]["channel_names"][3] == "S0"
    assert document["bias"]["setpoints_v"][7] == pytest.approx(0.7)

    decoded = window._decode_settings(document)
    assert decoded["bias"]["voltage_limit_v"] == pytest.approx(10.0)
    assert decoded["bias"]["channel_names"][4] == "AccL"
    assert decoded["bias"]["setpoints_v"][3] == pytest.approx(0.3)

    legacy_document = dict(document)
    legacy_document["bias"] = dict(document["bias"])
    legacy_document["bias"].pop("voltage_limit_v")
    legacy_document["bias"].pop("channel_names")
    legacy_decoded = window._decode_settings(legacy_document)
    assert legacy_decoded["bias"]["voltage_limit_v"] == pytest.approx(10.0)
    assert legacy_decoded["bias"]["channel_names"] == ("",) * 8
    window.close()
    app.processEvents()
