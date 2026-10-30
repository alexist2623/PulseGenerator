"""Offline Ramsey editor, persistence and sweep-preview contracts."""
from copy import deepcopy
from dataclasses import replace
import json
import socket

import pytest
from PyQt5 import QtWidgets

from dc_waveform_core import (
    QickRfAmplitudeParameterSpec,
    QickRfDurationParameterSpec,
    QickRfFrequencyParameterSpec,
    QickRfPhaseParameterSpec,
    QickRfPulseSpec,
)
from test_rf_t2_delay_editor import editor, select


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Ramsey editor checks must remain offline")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def ramsey(editor):
    select(editor.segment_length_mode, "extend_by_rf_duration")
    select(editor.predefined_template, "ramsey")
    editor.predefined_tau.setValue(3.0)
    return editor.configured_spec()


def test_ramsey_selection_creates_two_half_pulses_and_hides_refocusing_controls(editor):
    editor.predefined_n.setValue(8)
    editor.predefined_n_sweep_enabled.setChecked(True)
    spec = ramsey(editor)
    assert [item.kind for item in spec.effective_composite_items] == ["pulse", "delay", "pulse"]
    assert [event.duration_us for event in spec.pulse_events] == [1.0, 1.0]
    assert spec.pulse_events[1].delay_us - spec.pulse_events[0].delay_us == 4.0
    assert editor.t2_duration_parameter.currentText() == "d0/2"
    for checkbox in (editor.t2_initial_pi2, editor.t2_final_pi2):
        assert checkbox.isChecked() and not checkbox.isEnabled()
    assert not spec.predefined_n_sweep_enabled
    for control in (editor.predefined_n, editor.predefined_n_sweep_enabled,
                    editor.predefined_duration_parameter, editor.predefined_phase_parameter):
        assert control.isHidden()
    assert "gap between" in editor.predefined_description.text()
    select(editor.predefined_template, "cpmg")
    assert not editor.predefined_n.isHidden() and editor.predefined_n.isEnabled()
    assert not editor.predefined_duration_parameter.isHidden()
    assert not editor.predefined_phase_parameter.isHidden()
    assert editor.t2_initial_pi2.isEnabled() and editor.t2_final_pi2.isEnabled()


def test_ramsey_uses_selected_pi_reference_initial_final_phase_and_global_parameters(editor):
    ramsey(editor)
    editor._add_duration_parameter(QickRfDurationParameterSpec("pi", 0.4))
    select(editor.t2_duration_parameter, "pi")
    editor._add_frequency_parameter(QickRfFrequencyParameterSpec("shared_f", 123.0))
    editor.global_frequency_parameter.setCurrentText("shared_f")
    editor._add_numeric_parameter("amplitude", QickRfAmplitudeParameterSpec("shared_a", 4321))
    select(editor.global_amplitude_parameter, "shared_a")
    for name, phase, selector in (
        ("initial", 30.0, editor.t2_initial_phase_parameter),
        ("final", 120.0, editor.t2_final_phase_parameter),
    ):
        editor._add_numeric_parameter("phase", QickRfPhaseParameterSpec(name, phase))
        select(selector, name)
    events = editor.configured_spec().pulse_events
    assert [event.duration_us for event in events] == [0.2, 0.2]
    assert [event.phase_degrees for event in events] == [30.0, 120.0]
    assert all(event.frequency_mhz == 123.0 and event.gain == 4321 for event in events)
    assert events[1].delay_us - events[0].delay_us == pytest.approx(3.2)


@pytest.mark.parametrize("fine", [False, True])
def test_ramsey_json_roundtrip_and_both_mainwindow_decoders(editor, fine):
    ramsey(editor)
    editor.load_settings({**editor.settings_dict(), 'fine_duration': fine})
    assert editor.configured_spec().fine_duration
    expected = editor.configured_spec()
    saved = json.loads(json.dumps(editor.settings_dict()))
    editor.load_settings(deepcopy(saved))
    assert editor.settings_dict() == saved
    assert editor.configured_spec() == expected
    window = editor.window()
    for decoded in (
        window._decode_settings(deepcopy(window._settings_to_dict()))["rf_outputs"][0],
        window._decode_rf_output_entries([saved], set_names={"set_0"}, label="Ramsey")[0],
    ):
        assert decoded.pop("enabled")
        decoded.pop("amplitude_parameter_banks", None)
        assert QickRfPulseSpec(**decoded) == expected


def test_ramsey_tau_sweep_moves_only_final_pulse_in_mainwindow_preview(editor):
    ramsey(editor)
    editor.predefined_tau_sweep_start.setValue(2.0)
    editor.predefined_tau_sweep_stop.setValue(6.0)
    editor.predefined_tau_sweep_count.setValue(3)
    editor.predefined_tau_sweep_enabled.setChecked(True)
    QtWidgets.QApplication.processEvents()
    spec = editor.configured_spec()
    assert [axis.axis_kind for axis in spec.software_sweep_axes] == ["rf_template_tau"]
    assert spec.software_sweep_axes[0].points.tolist() == [2.0, 4.0, 6.0]
    window = editor.window()
    assert window._waveform_point_index.maximum() == 2
    first = window._waveform_preview_rf_specs[0].pulse_events
    window._waveform_point_index.setValue(2)
    QtWidgets.QApplication.processEvents()
    last = window._waveform_preview_rf_specs[0].pulse_events
    assert len(first) == len(last) == 2
    assert last[0] == first[0]
    assert last[1].delay_us - first[1].delay_us == 4.0
    assert [event.duration_us for event in last] == [1.0, 1.0]
    assert window._rf_timelines[0].pulse_names == ("T2_initial_pi2", "T2_final_pi2")


@pytest.mark.parametrize("calibrated", [False, True])
def test_ramsey_custom_conversion_keeps_waveform_and_amplitude_units(editor, calibrated):
    ramsey(editor)
    # Settings-only calibration check; no database is opened or hardware contacted.
    editor.global_calibration_database_path.setText("ramsey-offline-not-opened.db")
    editor.global_calibration_run_id.setValue(12)
    editor.amplitude_calibration_enabled.setChecked(calibrated)
    name = editor.global_amplitude_parameter.currentData()
    row = editor._numeric_parameter_names("amplitude").index(name)
    value = -32.5 if calibrated else 4321
    editor.amplitude_parameter_table.cellWidget(row, 1).setValue(value)
    before = editor.configured_spec().pulse_events
    assert all(event.power_calibration_enabled == calibrated for event in before)
    assert all((event.target_output_power_dbm if calibrated else event.gain) == value
               for event in before)
    select(editor.predefined_template, "custom")
    custom = editor.configured_spec()
    assert custom.pulse_events == before
    assert custom.global_amplitude_parameter == ""
    assert editor.t2_duration_parameter.currentText() == "d0/2"
    assert editor.t2_initial_pi2.isEnabled() and editor.t2_final_pi2.isEnabled()
    assert editor.composite_item_table.isEnabled()
    assert editor.composite_item_table.isColumnHidden(6)
    saved = json.loads(json.dumps(editor.settings_dict()))
    editor.load_settings(saved)
    assert editor.configured_spec().pulse_events == before


@pytest.mark.parametrize("named", [False, True])
def test_loading_ramsey_without_global_amplitude_keeps_endpoint_gain(editor, named):
    saved = editor.settings_dict()
    saved.update(predefined_template="ramsey", t2_gain=4321,
                 global_amplitude_mode="manual" if named else "legacy",
                 global_amplitude_parameter="", t2_amplitude_parameter="ends" if named else "",
                 amplitude_parameters=[dict(name="ends", value=4321)] if named else [],
                 predefined_amplitude_parameter="", amplitude_parameter_banks={})
    editor.load_settings(saved)
    assert [event.gain for event in editor.configured_spec().pulse_events] == [4321, 4321]
    editor.load_settings(json.loads(json.dumps(editor.settings_dict())))
    assert [event.gain for event in editor.configured_spec().pulse_events] == [4321, 4321]


@pytest.mark.parametrize("decode", ["panel", "main", "tab"])
def test_ramsey_survives_single_mode_save_reload_without_changing_active_single_pulse(editor, decode):
    ramsey(editor)
    editor._add_numeric_parameter("amplitude", QickRfAmplitudeParameterSpec("shared_a", 4321))
    select(editor.global_amplitude_parameter, "shared_a")
    expected_ramsey = editor.configured_spec()
    editor.single_mode.setChecked(True)
    editor.duration_sweep_start.setValue(0.1)
    editor.duration_sweep_stop.setValue(0.2)
    editor.duration_sweep_count.setValue(3)
    editor.duration_sweep_enabled.setChecked(True)
    expected_single = editor.configured_spec()
    assert expected_single.pulse_mode == "single"
    assert len(expected_single.pulse_events) == 1
    assert expected_single.software_sweep_axes == ()
    saved = json.loads(json.dumps(editor.settings_dict()))
    if decode == "main":
        saved = editor.window()._decode_settings(
            deepcopy(editor.window()._settings_to_dict()))["rf_outputs"][0]
    elif decode == "tab":
        saved = editor.window()._decode_rf_output_entries(
            [saved], set_names={"set_0"}, label="Ramsey inactive")[0]
    editor.load_settings(saved)
    assert editor.configured_spec() == expected_single
    assert editor.single_mode.isChecked()
    select(editor.predefined_template, "ramsey")
    editor.composite_mode.setChecked(True)
    assert editor.configured_spec() == replace(
        expected_ramsey, duration_sweep_start_us=0.1,
        duration_sweep_stop_us=0.2, duration_sweep_count=3)
