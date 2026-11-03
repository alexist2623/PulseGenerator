"""Register-level RF calibration matching without GUI or instrument access."""

import hashlib
import json
import sqlite3

import pytest

from power_calibration import CalibrationDatabase
from qick_filter_settings import (
    filter_settings_from_mapping,
    resolve_filter_settings,
)
from test_power_calibration import _calibration_database


@pytest.mark.parametrize("mode,first,second,bw1,bw2", [
    ("lowpass", 2.51, 2.55, 1.0, 8.0),
    ("highpass", 3.20, 3.205, 1.0, None),
    ("bandpass", 3.80, 3.805, 1.20, 1.20),
    ("bypass", 0, 18.0, -1.0, 12.0),
])
def test_different_requests_with_identical_registers_match(
    tmp_path, mode, first, second, bw1, bw2,
):
    path = tmp_path / "legacy.db"
    _calibration_database(path)
    metadata = {"configuration": {
        "nqz": 1, "output_filter_type": mode,
        "output_filter_cutoff_ghz": first,
        "output_filter_bandwidth_ghz": bw1,
    }}
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE runs ADD COLUMN Calibration_Config TEXT")
        connection.execute("UPDATE runs SET Calibration_Config=? WHERE run_id=2",
                           (json.dumps(metadata),))
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    # These are the same fields restored from existing GUI experiment JSONs.
    request = json.loads(json.dumps({
        "filter_type": mode, "filter_cutoff": second, "filter_bandwidth": bw2,
    }))
    expected = filter_settings_from_mapping(request)
    catalog = CalibrationDatabase(path)
    kwargs = dict(nqz=1, output_filter_type=request["filter_type"],
                  output_filter_cutoff_ghz=request["filter_cutoff"],
                  output_filter_bandwidth_ghz=request["filter_bandwidth"])
    candidate = catalog.output_calibration_candidates("RF_Out", [450], **kwargs)[0]
    loaded = catalog.output_calibration("RF_Out", [450], run_id=2, **kwargs)
    assert candidate.exact_match
    assert loaded.summary.output_filter_settings == expected
    assert loaded.summary.as_dict()["output_filter_registers"] == expected.as_dict()
    assert expected.label in candidate.display_label
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash


@pytest.mark.parametrize("recorded,requested", [
    (("lowpass", 2.49, 1), ("lowpass", 2.51, 1)),
    (("bandpass", 3.8, 1.2), ("bandpass", 3.9, 1.2)),
    (("lowpass", 2.51, 1), ("bypass", 2.51, 1)),
    (("", None, None), ("bypass", None, None)),
    (("lowpass", 2.05, 1), ("lowpass", 2.051, 1)),
])
def test_different_or_unknown_settings_are_not_accepted(recorded, requested):
    assert CalibrationDatabase._output_setting_mismatches(
        (1, *recorded), nqz=1, output_filter_type=requested[0],
        output_filter_cutoff_ghz=requested[1],
        output_filter_bandwidth_ghz=requested[2],
    )


@pytest.mark.parametrize("legacy", [
    {"output_filter_type": "bandpass", "output_filter_center_ghz": 3.8,
     "output_filter_bandwidth_ghz": 1.2},
    {"filter_type": "bandpass", "filter_cutoff": 3.8, "filter_bandwidth": 1.2},
    {"filter_type": "bandpass", "low_cutoff_ghz": 3.2, "high_cutoff_ghz": 4.4},
])
def test_legacy_json_and_calibration_frequency_representations(legacy):
    expected = resolve_filter_settings("bandpass", 3.8, 1.2)
    assert filter_settings_from_mapping(legacy) == expected
    assert filter_settings_from_mapping({"configuration": legacy}) == expected


def test_actual_metadata_precedes_requested_configuration():
    assert filter_settings_from_mapping({
        "configuration": {"output_filter_type": "bandpass",
                          "output_filter_center_ghz": 5.0,
                          "output_filter_bandwidth_ghz": 2.0},
        "rf_settings_actual": {"filter_type": "bandpass",
                               "filter_cutoff_ghz": 3.8,
                               "filter_bandwidth_ghz": 1.2},
    }) == resolve_filter_settings("bandpass", 3.8, 1.2)


def test_overlap_and_extended_hpf3_keep_existing_driver_mapping():
    settings = resolve_filter_settings("bandpass", 5.3, 3.6)
    assert (settings.hpf_bank, settings.lpf_bank) == (1, 2)
    settings = resolve_filter_settings("highpass", 12.3)
    assert (settings.hpf_bank, settings.hpf_state) == (3, 15)
    assert settings.as_dict()["WR0_SW"] == 0xD8
    assert settings.as_dict()["WR0_FILTER"] == 0xF0


def test_bypass_is_explicit_and_never_assumed_for_unknown_metadata():
    assert resolve_filter_settings("bypass", float("nan"), None).as_dict()["WR0_SW"] == 0xC0
    with pytest.raises(ValueError, match="unknown filter type"):
        filter_settings_from_mapping({})


def test_dc_output_filter_exemption_and_nyquist_check_are_preserved():
    assert not CalibrationDatabase._output_setting_mismatches(
        (1, "unknown", None, None), board_type="DC_Out", nqz=1,
        output_filter_type="bandpass", output_filter_cutoff_ghz=2,
        output_filter_bandwidth_ghz=1,
    )
    assert CalibrationDatabase._output_setting_mismatches(
        (1, "bypass", None, None), nqz=2, output_filter_type="bypass",
        output_filter_cutoff_ghz=None, output_filter_bandwidth_ghz=None,
    ) == ("Nyquist zone 1 != 2",)


def test_gui_json_restore_uses_register_match_without_opening_window(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtWidgets
    import DCWaveform_Generator as gui
    from dc_waveform_core import PulseSequence

    path = tmp_path / "legacy.db"
    _calibration_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE runs ADD COLUMN Calibration_Config TEXT")
        connection.execute("UPDATE runs SET Calibration_Config=? WHERE run_id=2", (
            json.dumps({"configuration": {
                "nqz": 1, "output_filter_type": "lowpass",
                "output_filter_cutoff_ghz": 2.51, "output_filter_bandwidth_ghz": 1,
            }}),
        ))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = gui.RfPulsePortPanel(PulseSequence(0.0, initial_duration_ns=10000), 0)
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args))
    try:
        legacy_json = panel.settings_dict()
        legacy_json.update({
            "enabled": True, "output_board_type": "RF_Out", "nqz": 1,
            "filter_type": "lowpass", "filter_cutoff": 2.55, "filter_bandwidth": 8,
            "frequency_mhz": 450, "power_calibration_enabled": True,
            "power_calibration_database_path": str(path),
            "power_calibration_run_id": 0, "target_output_power_dbm": -20,
        })
        panel.load_settings(json.loads(json.dumps(legacy_json)))
        app.processEvents()
        assert not panel.isVisible()
        assert panel._apply_calibrated_output_power() is not None
        assert "matched" in panel.power_calibration_status.text()
        assert not warnings
        panel.validate_power_calibration()
        panel.filter_cutoff.setValue(2.65)
        assert panel._apply_calibrated_output_power() is None
        assert warnings
    finally:
        panel.close()
        panel.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
