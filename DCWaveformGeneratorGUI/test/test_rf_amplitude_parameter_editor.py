"""Offline amplitude editor contracts: presets, Custom and legacy migration."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import socket

import pytest
from PyQt5 import QtCore

import DCWaveform_Generator as gui
from dc_waveform_core import (
    QickRfAmplitudeParameterSpec,
    QickRfCompositeItemSpec,
    QickRfDurationParameterSpec,
)
from test_rf_t2_delay_editor import editor, parameter_row, select


PRESETS = ("cpmg", "udd", "hahn_echo")
OFFLINE_CALIBRATION = "offline-amplitude-editor-not-opened.db"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Amplitude editor tests must not connect to a board")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def pulse_rows(panel):
    table = panel.composite_item_table
    return [row for row in range(table.rowCount())
            if table.cellWidget(row, 0).currentData() == "pulse"]


def amplitude_selector(panel):
    if panel.predefined_template.currentData() == "custom":
        return panel.composite_item_table.cellWidget(pulse_rows(panel)[0], 5)
    return panel.global_amplitude_parameter


def set_amplitude(panel, name, *, value, start, stop, count, sweep=True):
    row = parameter_row(panel, "amplitude", name)
    table = panel.amplitude_parameter_table
    for column, number in ((1, value), (3, start), (4, stop), (5, count)):
        table.cellWidget(row, column).setValue(number)
    table.cellWidget(row, 2).setChecked(sweep)


def amplitude(panel, name):
    return next(parameter for parameter in panel.configured_spec().amplitude_parameters
                if parameter.name == name)


def roundtrip(panel):
    """Exercise JSON serialization without touching user settings files."""
    saved = panel.settings_dict()
    panel.load_settings(json.loads(json.dumps(saved)))
    assert panel.settings_dict() == saved
    return saved


def assert_waveform_matches_editor_precision(panel, actual, expected):
    """UDD's irrational delay positions round to each delay spinbox's grid."""
    spec = panel.configured_spec()
    delay_count = sum(item.kind == "delay" for item in spec.effective_composite_items)
    table = panel.delay_parameter_table
    resolution_us = max((
        gui._time_to_ns(10 ** -table.cellWidget(row, 1).decimals(), panel._time_unit) / 1000
        for row in range(table.rowCount())
    ), default=0.0)
    # At most half a display step per preceding delay, plus floating-point noise.
    tolerance_us = delay_count * resolution_us / 2 + 1e-12
    assert len(actual) == len(expected)
    for actual_event, expected_event in zip(actual, expected):
        observed, reference = asdict(actual_event), asdict(expected_event)
        assert observed.pop("delay_us") == pytest.approx(
            reference.pop("delay_us"), abs=tolerance_us, rel=0)
        assert observed == reference


def legacy_settings(*, calibrated=False):
    settings = deepcopy(gui.DEFAULT_RF_OUTPUT_SETTINGS)
    settings.update(
        enabled=True, pulse_mode="composite", predefined_template="custom",
        global_frequency_parameter="", global_gain=None,
        global_amplitude_mode="legacy", global_amplitude_parameter="",
        global_calibration_database_path="", global_calibration_run_id=0,
        amplitude_parameters=[], predefined_amplitude_parameter="",
        t2_amplitude_parameter="", t2_initial_pi2=False, t2_final_pi2=False,
        duration_parameters=[asdict(QickRfDurationParameterSpec("d0", 2.0))],
        composite_items=[
            asdict(QickRfCompositeItemSpec(
                "pulse", "first", 2.0, "f0", gain=1234, phase_degrees=15,
                duration_parameter="d0", power_calibration_enabled=calibrated,
                power_calibration_database_path=OFFLINE_CALIBRATION,
                power_calibration_run_id=12, target_output_power_dbm=-34.5)),
            asdict(QickRfCompositeItemSpec("delay", "wait", 3.0)),
            asdict(QickRfCompositeItemSpec(
                "pulse", "second", 2.0, "f0", gain=-5678, phase_degrees=-45,
                duration_parameter="d0", power_calibration_enabled=calibrated,
                power_calibration_database_path=OFFLINE_CALIBRATION,
                power_calibration_run_id=12, target_output_power_dbm=-21.25)),
        ],
    )
    return settings


@pytest.mark.parametrize("template", PRESETS)
def test_presets_use_one_named_amplitude_for_body_and_both_t2_endpoints(editor, template):
    select(editor.predefined_template, template)
    editor.t2_initial_pi2.setChecked(True)
    editor.t2_final_pi2.setChecked(True)
    editor._add_numeric_parameter("amplitude", QickRfAmplitudeParameterSpec("shared", -12345))
    select(editor.global_amplitude_parameter, "shared")

    assert not editor.amplitude_mode_group.isCheckable()
    assert not editor.amplitude_mode_group.isHidden()
    assert not editor.amplitude_calibration_enabled.isHidden()
    assert editor.global_amplitude_parameter.isEnabled()
    assert not editor.global_amplitude_parameter.isHidden()
    assert editor.global_gain.isHidden()
    assert editor.common_amplitude_enabled.isHidden()
    assert editor.composite_item_table.isColumnHidden(6)
    spec = editor.configured_spec()
    assert spec.global_amplitude_mode == "manual"
    assert spec.global_amplitude_parameter == "shared"
    assert len(spec.pulse_events) >= 3
    assert all(event.gain == -12345 and not event.power_calibration_enabled
               for event in spec.pulse_events)
    assert all(item.amplitude_parameter == "shared"
               for item in spec.effective_composite_items if item.kind == "pulse")
    roundtrip(editor)
    assert editor.configured_spec() == spec


@pytest.mark.parametrize("template", PRESETS)
def test_preset_calibration_checkbox_uses_shared_dbm_parameter_for_all_pulses(editor, template):
    select(editor.predefined_template, template)
    editor.t2_initial_pi2.setChecked(True)
    editor.t2_final_pi2.setChecked(True)
    editor.global_calibration_database_path.setText(OFFLINE_CALIBRATION)
    editor.global_calibration_run_id.setValue(12)
    editor.amplitude_calibration_enabled.setChecked(True)
    selector = editor.global_amplitude_parameter
    set_amplitude(editor, selector.currentData(), value=-32.5, start=-40, stop=-25, count=4, sweep=False)

    spec = editor.configured_spec()
    assert spec.global_amplitude_mode == "calibration"
    assert editor.global_gain.isHidden()
    assert editor.composite_item_table.isColumnHidden(6)
    assert all(event.power_calibration_enabled
               and event.target_output_power_dbm == -32.5
               and event.power_calibration_database_path == OFFLINE_CALIBRATION
               and event.power_calibration_run_id == 12 for event in spec.pulse_events)
    assert "dBm" in editor.amplitude_parameter_table.horizontalHeaderItem(1).text()
    roundtrip(editor)
    assert editor.configured_spec() == spec


def test_custom_has_independent_named_rows_and_no_common_amplitude_override(editor):
    select(editor.predefined_template, "custom")
    editor._add_composite_item("pulse")
    editor._add_numeric_parameter("amplitude", QickRfAmplitudeParameterSpec("first_amp", 1234))
    editor._add_numeric_parameter("amplitude", QickRfAmplitudeParameterSpec("second_amp", -5678))
    rows = pulse_rows(editor)
    for row, name in zip(rows, ("first_amp", "second_amp")):
        selector = editor.composite_item_table.cellWidget(row, 5)
        assert selector.isEnabled()
        select(selector, name)
    assert not editor.amplitude_mode_group.isCheckable()
    assert not editor.amplitude_mode_group.isHidden()
    assert not editor.amplitude_calibration_enabled.isHidden()
    assert editor.common_amplitude_enabled.isHidden()
    assert editor.global_gain.isHidden()
    assert editor.composite_item_table.isColumnHidden(6)
    spec = editor.configured_spec()
    assert spec.global_amplitude_mode == "manual"
    assert spec.global_amplitude_parameter == ""
    assert [event.gain for event in spec.pulse_events] == [1234, -5678]
    assert [item.amplitude_parameter for item in spec.composite_items if item.kind == "pulse"] == ["first_amp", "second_amp"]
    assert roundtrip(editor)["global_amplitude_parameter"] == ""


@pytest.mark.parametrize("template", ("custom", "hahn_echo"))
def test_gain_and_dbm_value_sweep_banks_survive_toggles_and_json_reload(editor, template):
    select(editor.predefined_template, template)
    selector = amplitude_selector(editor)
    name = selector.currentData()
    assert name
    set_amplitude(editor, name, value=1234, start=-1000, stop=1000, count=3)
    manual = amplitude(editor, name)
    manual_axis = next(axis for axis in editor.configured_spec().software_sweep_axes
                       if axis.axis_kind == "rf_amplitude" and axis.parameter_name == name)
    assert tuple(manual_axis.points) == (-1000, 0, 1000)
    assert manual_axis.coordinate_unit == "gain"

    editor.global_calibration_database_path.setText(OFFLINE_CALIBRATION)
    editor.global_calibration_run_id.setValue(12)
    editor.amplitude_calibration_enabled.setChecked(True)
    assert amplitude_selector(editor).currentData() == name
    set_amplitude(editor, name, value=-31.25, start=-42, stop=-22, count=5)
    calibrated = amplitude(editor, name)
    cal_axis = next(axis for axis in editor.configured_spec().software_sweep_axes
                   if axis.axis_kind == "rf_amplitude" and axis.parameter_name == name)
    assert tuple(cal_axis.points) == (-42, -37, -32, -27, -22)
    assert cal_axis.coordinate_unit == "dBm"
    saved = roundtrip(editor)
    assert saved["global_calibration_database_path"] == OFFLINE_CALIBRATION
    assert saved["global_calibration_run_id"] == 12

    editor.amplitude_calibration_enabled.setChecked(False)
    assert amplitude(editor, name) == manual
    assert "Gain" in editor.amplitude_parameter_table.horizontalHeaderItem(1).text()
    roundtrip(editor)
    editor.amplitude_calibration_enabled.setChecked(True)
    assert amplitude(editor, name) == calibrated
    assert "dBm" in editor.amplitude_parameter_table.horizontalHeaderItem(1).text()


@pytest.mark.parametrize("template", PRESETS)
@pytest.mark.parametrize("calibration", (False, True))
def test_preset_to_custom_copies_shared_reference_and_preserves_resolved_waveform(editor, template, calibration):
    select(editor.predefined_template, template)
    editor.t2_initial_pi2.setChecked(True)
    editor.t2_final_pi2.setChecked(True)
    editor.global_calibration_database_path.setText(OFFLINE_CALIBRATION)
    editor.global_calibration_run_id.setValue(12)
    editor.amplitude_calibration_enabled.setChecked(calibration)
    name = editor.global_amplitude_parameter.currentData()
    set_amplitude(editor, name, value=-32.5 if calibration else -12345,
                  start=-32.5 if calibration else -12345,
                  stop=-32.5 if calibration else -12345, count=1, sweep=False)
    before = editor.configured_spec().pulse_events
    select(editor.predefined_template, "custom")
    spec = editor.configured_spec()
    assert spec.global_amplitude_parameter == ""
    assert spec.t2_amplitude_parameter == name
    assert all(item.amplitude_parameter == name for item in spec.composite_items if item.kind == "pulse")
    assert_waveform_matches_editor_precision(editor, spec.pulse_events, before)
    assert all(editor.composite_item_table.cellWidget(row, 5).isEnabled() for row in pulse_rows(editor))
    assert editor.composite_item_table.isColumnHidden(6)
    roundtrip(editor)
    assert_waveform_matches_editor_precision(editor, editor.configured_spec().pulse_events, before)


def test_uniform_legacy_manual_mode_migrates_each_gain_without_changing_values(editor):
    settings = legacy_settings()
    settings.update(t2_initial_pi2=True, t2_final_pi2=True, t2_gain=8765)
    editor.load_settings(settings)
    spec = editor.configured_spec()
    assert spec.global_amplitude_mode == "manual"
    assert not editor.amplitude_calibration_enabled.isChecked()
    assert [event.gain for event in spec.pulse_events] == [8765, 1234, -5678, 8765]
    assert [event.phase_degrees for event in spec.pulse_events] == [0, 15, -45, 0]
    assert all(item.amplitude_parameter for item in spec.effective_composite_items if item.kind == "pulse")
    assert not editor.amplitude_legacy_notice.isVisibleTo(editor)
    roundtrip(editor)
    assert editor.configured_spec() == spec


def test_uniform_legacy_calibration_migrates_db_run_and_individual_dbm_values(editor):
    editor.load_settings(legacy_settings(calibrated=True))
    spec = editor.configured_spec()
    assert spec.global_amplitude_mode == "calibration"
    assert editor.amplitude_calibration_enabled.checkState() == QtCore.Qt.Checked
    assert spec.global_calibration_database_path == OFFLINE_CALIBRATION
    assert spec.global_calibration_run_id == 12
    assert [event.target_output_power_dbm for event in spec.pulse_events] == [-34.5, -21.25]
    assert all(event.power_calibration_enabled
               and event.power_calibration_database_path == OFFLINE_CALIBRATION
               and event.power_calibration_run_id == 12 for event in spec.pulse_events)
    assert all(item.amplitude_parameter for item in spec.composite_items if item.kind == "pulse")
    roundtrip(editor)
    assert editor.configured_spec() == spec


@pytest.mark.parametrize("conflict", ("mode", "database", "run"))
def test_mixed_legacy_is_preserved_until_user_explicitly_selects_a_common_mode(editor, conflict):
    settings = legacy_settings(calibrated=True)
    last = settings["composite_items"][2]
    if conflict == "mode":
        last["power_calibration_enabled"] = False
    elif conflict == "database":
        last["power_calibration_database_path"] = "another-offline-not-opened.db"
    else:
        last["power_calibration_run_id"] = 99
    editor.load_settings(settings)
    spec = editor.configured_spec()
    before = spec.pulse_events
    assert spec.global_amplitude_mode == "legacy"
    assert editor.amplitude_calibration_enabled.checkState() == QtCore.Qt.PartiallyChecked
    assert not editor.amplitude_legacy_notice.isHidden()
    assert editor.amplitude_legacy_notice.text().strip()
    assert editor.composite_item_table.isColumnHidden(6)
    assert [event.gain for event in before] == [1234, -5678]
    assert [event.target_output_power_dbm for event in before] == [-34.5, -21.25]
    assert before[1].power_calibration_enabled == last["power_calibration_enabled"]
    assert before[1].power_calibration_database_path == last["power_calibration_database_path"]
    assert before[1].power_calibration_run_id == last["power_calibration_run_id"]
    roundtrip(editor)
    assert editor.configured_spec().pulse_events == before
    assert editor.amplitude_calibration_enabled.checkState() == QtCore.Qt.PartiallyChecked

    editor.amplitude_calibration_enabled.setCheckState(QtCore.Qt.Unchecked)
    changed = editor.configured_spec()
    assert changed.global_amplitude_mode == "manual"
    assert all(not event.power_calibration_enabled for event in changed.pulse_events)
    assert [event.gain for event in changed.pulse_events] == [1234, -5678]
    assert all(item.amplitude_parameter for item in changed.composite_items if item.kind == "pulse")
    assert all(editor.composite_item_table.cellWidget(row, 5).isEnabled() for row in pulse_rows(editor))


def populate_both_banks(panel):
    name = amplitude_selector(panel).currentData()
    set_amplitude(panel, name, value=1234, start=-1000, stop=1000, count=3)
    manual = amplitude(panel, name)
    panel.global_calibration_database_path.setText(OFFLINE_CALIBRATION)
    panel.global_calibration_run_id.setValue(12)
    panel.amplitude_calibration_enabled.setChecked(True)
    set_amplitude(panel, name, value=-31.25, start=-42, stop=-22, count=5)
    calibrated = amplitude(panel, name)
    return name, manual, calibrated


@pytest.mark.parametrize("active_calibration", (False, True))
def test_mainwindow_and_independent_tab_json_decoders_preserve_both_amplitude_banks(editor, active_calibration):
    name, manual, calibrated = populate_both_banks(editor)
    editor.amplitude_calibration_enabled.setChecked(active_calibration)
    window = editor.window()
    assert isinstance(window, gui.MainWindow)
    document = json.loads(json.dumps(window._settings_to_dict()))
    saved_rf = document["rf_outputs"][0]
    decoded = window._decode_settings(document)
    assert decoded["rf_outputs"][0]["amplitude_parameter_banks"] == saved_rf["amplitude_parameter_banks"]
    tab_decoded = window._decode_rf_output_entries(
        document["rf_outputs"], set_names={saved_rf["segment_name"]}, label="offline RF editors")
    assert tab_decoded[0]["amplitude_parameter_banks"] == saved_rf["amplitude_parameter_banks"]
    window._apply_decoded_settings(decoded)
    restored = window._rf_ports_panel._panels[0]
    restored.amplitude_calibration_enabled.setChecked(False)
    assert amplitude(restored, name) == manual
    restored.amplitude_calibration_enabled.setChecked(True)
    assert amplitude(restored, name) == calibrated


@pytest.mark.parametrize("active_calibration", (False, True))
def test_amplitude_rename_preserves_active_and_inactive_bank_values_and_sweeps(editor, active_calibration):
    old_name, manual, calibrated = populate_both_banks(editor)
    editor.amplitude_calibration_enabled.setChecked(active_calibration)
    row = parameter_row(editor, "amplitude", old_name)
    name_editor = editor.amplitude_parameter_table.cellWidget(row, 0)
    before = editor.configured_spec()
    name_editor.setText("renamed_amplitude")
    assert editor.configured_spec() == before
    name_editor.editingFinished.emit()
    assert amplitude_selector(editor).currentData() == "renamed_amplitude"
    assert old_name not in editor._numeric_parameter_names("amplitude")
    roundtrip(editor)
    editor.amplitude_calibration_enabled.setChecked(False)
    assert amplitude(editor, "renamed_amplitude") == replace(manual, name="renamed_amplitude")
    editor.amplitude_calibration_enabled.setChecked(True)
    assert amplitude(editor, "renamed_amplitude") == replace(calibrated, name="renamed_amplitude")


def test_legacy_cpmg_keeps_independent_t2_and_body_until_global_parameter_is_selected(editor):
    settings = legacy_settings()
    settings.update(predefined_template="cpmg", predefined_n=2,
                    predefined_tau_us=12.0, predefined_gain=1200,
                    predefined_n_sweep_enabled=False,
                    predefined_tau_sweep_enabled=True,
                    predefined_tau_sweep_start_us=10.0,
                    predefined_tau_sweep_stop_us=14.0,
                    predefined_tau_sweep_count=3,
                    t2_initial_pi2=True, t2_final_pi2=True, t2_gain=4300)
    editor.load_settings(settings)
    before = editor.configured_spec()
    assert before.global_amplitude_mode == "manual"
    assert before.global_amplitude_parameter == ""
    assert not editor.global_amplitude_parameter.currentData()
    assert [event.gain for event in before.pulse_events] == [4300, 1200, 1200, 4300]
    preserved = (before.predefined_n, before.predefined_tau_us,
                 before.predefined_n_sweep_enabled, before.predefined_tau_sweep_enabled,
                 before.predefined_tau_sweep_start_us, before.predefined_tau_sweep_stop_us,
                 before.predefined_tau_sweep_count)
    assert preserved == (2, 12.0, False, True, 10.0, 14.0, 3)
    roundtrip(editor)
    assert editor.configured_spec().pulse_events == before.pulse_events

    editor._add_numeric_parameter("amplitude", QickRfAmplitudeParameterSpec("explicit_global", 3210))
    assert editor.global_amplitude_parameter.isEnabled()
    select(editor.global_amplitude_parameter, "explicit_global")
    shared = editor.configured_spec()
    assert shared.global_amplitude_parameter == "explicit_global"
    assert [event.gain for event in shared.pulse_events] == [3210] * 4
    assert (shared.predefined_n, shared.predefined_tau_us,
            shared.predefined_n_sweep_enabled, shared.predefined_tau_sweep_enabled,
            shared.predefined_tau_sweep_start_us, shared.predefined_tau_sweep_stop_us,
            shared.predefined_tau_sweep_count) == preserved
    assert [event.delay_us for event in shared.pulse_events] == [event.delay_us for event in before.pulse_events]


def test_deleting_and_reusing_amplitude_name_does_not_restore_stale_other_mode_bank(editor):
    editor._add_numeric_parameter("amplitude", QickRfAmplitudeParameterSpec("reused", 1111))
    select(amplitude_selector(editor), "reused")
    editor.global_calibration_database_path.setText(OFFLINE_CALIBRATION)
    editor.global_calibration_run_id.setValue(12)
    editor.amplitude_calibration_enabled.setChecked(True)
    set_amplitude(editor, "reused", value=-31, start=-41, stop=-21, count=3)
    editor.amplitude_calibration_enabled.setChecked(False)
    editor.amplitude_parameter_table.selectRow(parameter_row(editor, "amplitude", "reused"))
    editor._remove_numeric_parameter("amplitude")
    assert "reused" not in editor._numeric_parameter_names("amplitude")
    assert all(parameter["name"] != "reused"
               for bank in editor.settings_dict()["amplitude_parameter_banks"].values()
               for parameter in bank)

    editor._add_numeric_parameter("amplitude", QickRfAmplitudeParameterSpec("reused", 2345, True, 2000, 3000, 3))
    select(amplitude_selector(editor), "reused")
    new_manual = amplitude(editor, "reused")
    editor.amplitude_calibration_enabled.setChecked(True)
    assert amplitude_selector(editor).currentData() == "reused"
    assert amplitude(editor, "reused").value == -20.0
    assert not amplitude(editor, "reused").sweep_enabled
    name_editor = editor.amplitude_parameter_table.cellWidget(parameter_row(editor, "amplitude", "reused"), 0)
    name_editor.setText("reused_renamed")
    name_editor.editingFinished.emit()
    assert amplitude_selector(editor).currentData() == "reused_renamed"
    saved = roundtrip(editor)
    for bank in saved["amplitude_parameter_banks"].values():
        names = [parameter["name"] for parameter in bank]
        assert len(names) == len(set(names))
        assert "reused" not in names
    editor.amplitude_calibration_enabled.setChecked(False)
    assert amplitude(editor, "reused_renamed") == replace(new_manual, name="reused_renamed")


def test_legacy_single_with_empty_dormant_composite_loads_values_and_sweep(editor):
    sweep = "duration"
    settings = legacy_settings()
    settings.update(
        pulse_mode="single", composite_items=[], frequency_parameters=[],
        duration_parameters=[], delay_parameters=[], amplitude_parameters=[],
        phase_parameters=[], amplitude_parameter_banks={},
        t2_initial_pi2=False, t2_final_pi2=False,
        delay_us=0.125, duration_us=2.5, frequency_mhz=87.5, gain=4321,
        phase_degrees=35.0, att1_db=1.5, att2_db=2.0,
        duration_sweep_enabled=sweep == "duration",
        duration_sweep_start_us=1.25, duration_sweep_stop_us=3.75,
        duration_sweep_count=3,
        frequency_sweep_enabled=sweep == "frequency",
        frequency_sweep_start_mhz=80.0, frequency_sweep_stop_mhz=95.0,
        frequency_sweep_count=4,
        power_calibration_enabled=sweep == "power",
        power_calibration_database_path=OFFLINE_CALIBRATION,
        power_calibration_run_id=12, target_output_power_dbm=-31.25,
        power_sweep_enabled=sweep == "power",
        power_sweep_start_dbm=-40.0, power_sweep_stop_dbm=-25.0,
        power_sweep_count=4,
    )
    editor.load_settings(settings)
    assert editor._loading_rf_settings is False
    actual = editor.configured_spec()
    assert actual.pulse_mode == "single"
    assert not actual.composite_items
    assert not actual.t2_initial_pi2 and not actual.t2_final_pi2
    for key in (
        "delay_us", "duration_us", "frequency_mhz", "gain", "phase_degrees",
        "att1_db", "att2_db", "duration_sweep_enabled", "duration_sweep_start_us",
        "duration_sweep_stop_us", "duration_sweep_count", "frequency_sweep_enabled",
        "frequency_sweep_start_mhz", "frequency_sweep_stop_mhz", "frequency_sweep_count",
        "power_calibration_enabled", "power_calibration_database_path",
        "power_calibration_run_id", "target_output_power_dbm", "power_sweep_enabled",
        "power_sweep_start_dbm", "power_sweep_stop_dbm", "power_sweep_count",
    ):
        assert getattr(actual, key) == settings[key], key
    assert len(actual.pulse_events) == 1
    event = actual.pulse_events[0]
    assert (event.delay_us, event.duration_us, event.frequency_mhz,
            event.gain, event.phase_degrees) == (0.125, 2.5, 87.5, 4321, 35.0)
    roundtrip(editor)
    assert editor._loading_rf_settings is False
    assert editor.configured_spec() == actual
