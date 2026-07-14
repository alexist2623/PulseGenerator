"""Headless tests for GUI time units and RF port/readout integration.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import ast
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets
import numpy as np

import DCWaveform_Generator as gui
from dc_waveform_core import (
    PulseSequence,
    QickDdrReadoutSpec,
    QickRfPulseSpec,
    QickSweepSpec,
    generate_qick_program_code,
)


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_gui_defaults_and_time_unit_round_trip():
    app = _application()
    window = gui.MainWindow()
    control = window._multi_ctrl._ctrl_pannels[0]

    assert window._time_unit == "us"
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


def test_rf_controls_are_left_tabs_and_support_multiple_ports():
    app = _application()
    window = gui.MainWindow()
    assert [window._control_tabs.tabText(i) for i in range(3)] == [
        "AWG Outputs",
        "RF Outputs",
        "RF Readout",
    ]
    toolbar_labels = [action.text() for bar in window.findChildren(QtWidgets.QToolBar)
                      for action in bar.actions()]
    assert "RF Pulse" not in toolbar_labels

    first = window._rf_ports_panel._panels[0]
    first.setChecked(True)
    window._rf_ports_panel.add_port()
    second = window._rf_ports_panel._panels[1]
    second.setChecked(True)
    app.processEvents()
    specs = window._rf_ports_panel.specs()
    assert [spec.gen_ch for spec in specs] == [0, 2]
    assert all(spec.duration_us == 1.0 for spec in specs)
    assert len(window._rf_timelines) == 2

    window._rf_ports_panel.remove_port(second)
    app.processEvents()
    assert len(window._rf_ports_panel.specs()) == 1
    assert len(window._rf_timelines) == 1
    window.close()


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


def test_generated_module_supports_multiple_rf_outputs_and_readout_chain():
    pulse = PulseSequence(100.0, initial_duration_ns=1000.0)
    pulse.add_flat_ramp(1000.0, 5000.0, 200.0)
    rf_specs = (
        QickRfPulseSpec(0, "set_1", 0.0, 1.0, 50.0, 12000, 10.0, 12.0),
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
        rf_pulse_specs=rf_specs,
        ddr_readout_spec=readout,
    )
    ast.parse(code)
    namespace = {}
    exec(compile(code, "<multi-rf-generated>", "exec"), namespace)

    class FakeSoc:
        def __init__(self):
            self.calls = []

        def rfb_set_gen_rf(self, gen_ch, att1, att2):
            self.calls.append(("output", gen_ch, att1, att2))
            return att1, att2

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
    window._control_tabs.setCurrentWidget(window._rf_readout_panel)

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
    app.processEvents()

    expected = window._settings_to_dict()
    saved_path = window._save_settings_json(tmp_path / "complete_experiment")
    assert saved_path.suffix == ".json"
    document = json.loads(saved_path.read_text(encoding="utf-8"))
    assert document["schema"] == gui.SETTINGS_SCHEMA
    assert document["version"] == gui.SETTINGS_VERSION
    assert len(document["awg"]["outputs"]) == 2
    assert len(document["rf_outputs"]) == 2

    restored = gui.MainWindow()
    restored._load_settings_json(saved_path)
    app.processEvents()
    assert restored._settings_to_dict() == expected
    assert restored._ddr_readout_spec == readout.spec()
    assert len(restored._rf_pulse_specs) == 1
    window.close()
    restored.close()


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
