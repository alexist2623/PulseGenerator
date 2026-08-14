"""Regression tests for calibration-owned QCS channel mappings."""

from __future__ import annotations

import copy
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtTest, QtWidgets

import calibration_gui
from calibration_gui import _QcsCalibrationPreview
import qcs_front_panel as front_panel
from qcs_rf_power_calibration import (
    M5200PowerReference,
    M5300PowerCalibrationConfig,
)


_APP = None


def _application():
    global _APP
    if _APP is None:
        _APP = (
            QtWidgets.QApplication.instance()
            or QtWidgets.QApplication([])
        )
    return _APP


def _shared_inventory() -> dict:
    """Return one inventory with unrelated Experiment-owned routes."""

    configuration = front_panel.default_qcs_hardware_configuration(
        ("experiment_dc_0", "experiment_dc_1"),
        {0: "experiment_rf"},
        "experiment_digitizer",
    )
    configuration["modules"].append({"slot": 6, "model": "M5201A"})
    configuration["downconverter_links"] = [
        {
            "digitizer_slot": 5,
            "digitizer_channel": 1,
            "downconverter_slot": 6,
            "downconverter_channel": 1,
            "lo_frequency_hz": 7.25e9,
        }
    ]
    return front_panel.normalize_qcs_hardware_configuration(configuration)


def _local_mapping(
    role: str,
    *,
    slot: int,
    channel: int,
    lo_frequency_hz=None,
) -> dict:
    return {
        "role": role,
        "logical_index": 0,
        "virtual_name": {
            "rf": "calibration_rf_output",
            "acquisition": "calibration_acquisition",
        }[role],
        "label": 0,
        "absolute_phase": True,
        "lo_frequency_hz": lo_frequency_hz,
        "slot": slot,
        "channel": channel,
    }


def test_calibration_scoped_recipe_excludes_experiment_dc_and_mixer_routes():
    shared = _shared_inventory()
    output = _local_mapping(
        "rf",
        slot=3,
        channel=2,
        lo_frequency_hz=0.0,
    )
    acquisition = _local_mapping(
        "acquisition",
        slot=5,
        channel=2,
    )

    scoped = front_panel.scoped_qcs_hardware_configuration(
        shared,
        (output, acquisition),
    )

    assert scoped["modules"] == shared["modules"]
    assert scoped["downconverter_links"] == []
    assert [
        (mapping["role"], mapping["virtual_name"])
        for mapping in scoped["channel_mappings"]
    ] == [
        ("rf", "calibration_rf_output"),
        ("acquisition", "calibration_acquisition"),
    ]
    assert {
        (mapping["slot"], mapping["channel"])
        for mapping in scoped["channel_mappings"]
    } == {(3, 2), (5, 2)}
    assert all(
        mapping["role"] not in {"dc", "unassigned"}
        for mapping in scoped["channel_mappings"]
    )


def test_calibration_preview_keeps_local_connector_when_shared_mappings_change():
    _application()
    shared = _shared_inventory()
    preview = _QcsCalibrationPreview("rf", "RF calibration output")
    preview.set_qcs_front_panel_configuration(shared)
    preview.accept_qcs_front_panel_connector(
        _local_mapping(
            "rf",
            slot=3,
            channel=2,
            lo_frequency_hz=6.125e9,
        )
    )

    updated_shared = copy.deepcopy(shared)
    dc_mapping = next(
        mapping
        for mapping in updated_shared["channel_mappings"]
        if mapping["role"] == "dc" and mapping["logical_index"] == 1
    )
    dc_mapping["virtual_name"] = "experiment_dc_renamed"
    dc_mapping["channel"] = 3
    updated_shared = front_panel.normalize_qcs_hardware_configuration(
        updated_shared
    )
    preview.set_qcs_front_panel_configuration(updated_shared)

    selected = preview.selected_mapping()
    assert selected["virtual_name"] == "calibration_rf_output"
    assert (selected["slot"], selected["channel"]) == (3, 2)
    assert selected["lo_frequency_hz"] == 6.125e9
    assert preview.mapping.currentData() == (3, 2)
    preview.close()


def test_null_endpoint_clears_previous_calibration_selection():
    _application()
    shared = _shared_inventory()
    preview = _QcsCalibrationPreview("rf", "RF calibration output")
    preview.set_qcs_front_panel_configuration(shared)
    preview.accept_qcs_front_panel_connector(
        _local_mapping(
            "rf",
            slot=3,
            channel=2,
            lo_frequency_hz=0.0,
        )
    )

    preview.set_selection_settings(None)

    with pytest.raises(ValueError, match="Select the QCS calibration rf SMA"):
        preview.selected_mapping()
    preview.close()


def test_local_selection_is_cleared_for_a_different_qcs_controller():
    _application()
    shared = _shared_inventory()
    preview = _QcsCalibrationPreview("rf", "RF calibration output")
    preview.set_qcs_front_panel_configuration(shared)
    preview.accept_qcs_front_panel_connector(
        _local_mapping(
            "rf",
            slot=3,
            channel=2,
            lo_frequency_hz=0.0,
        )
    )
    other_controller = copy.deepcopy(shared)
    other_controller["ip_address"] = "192.168.2.200"
    other_controller["chassis"] = 2
    other_controller["host_controller"] = 2
    other_controller = front_panel.normalize_qcs_hardware_configuration(
        other_controller
    )

    preview.set_qcs_front_panel_configuration(other_controller)

    with pytest.raises(ValueError, match="Select the QCS calibration rf SMA"):
        preview.selected_mapping()
    preview.close()


def test_pick_only_connector_emits_selection_without_mutating_shared_mapper(
    tmp_path,
):
    _application()
    shared = _shared_inventory()
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "experiment_mapper.qcs"),
            "dc_channel_names": ["experiment_dc_0", "experiment_dc_1"],
            "rf_channel_names": {"0": "experiment_rf"},
            "acquisition_channel_name": "experiment_digitizer",
            "hardware_configuration": shared,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
        },
        output_count=2,
    )
    assert control.focus_mapping("rf", 0)
    before = control.working_configuration()
    picked = QtTest.QSignalSpy(control.connector_picked)
    selected_events = QtTest.QSignalSpy(control.connector_selected)
    control.set_connector_pick_only(
        True,
        _local_mapping(
            "rf",
            slot=3,
            channel=2,
            lo_frequency_hz=6.5e9,
        ),
    )
    assert control._focused_mapping_addresses(before) == ((3, 2),)

    assert control.select_connector(3, 2) is True

    assert len(picked) == 1
    assert tuple(picked[0][:4]) == ("rf", 0, 3, 2)
    payload = picked[0][4]
    assert payload["virtual_name"] == "calibration_rf_output"
    assert payload["lo_frequency_hz"] == 6.5e9
    assert len(selected_events) == 0
    assert control.working_configuration() == before
    control.close()


@pytest.mark.parametrize(
    "values",
    (
        "not-an-object",
        {"slot": 3, "channel": 1},
        {
            "ip_address": "192.168.2.105",
            "chassis": 1,
            "host_controller": 0,
            "slot": 3,
            "channel": 0,
            "lo_frequency_hz": 0.0,
        },
    ),
)
def test_invalid_persisted_endpoint_is_rejected(values):
    with pytest.raises((TypeError, ValueError)):
        calibration_gui.normalize_qcs_calibration_endpoint_settings(
            values,
            label="test endpoint",
            allow_lo_frequency=True,
        )


def test_rf_calibration_worker_builds_its_scoped_mapper_without_global_connection(
    monkeypatch,
    tmp_path,
):
    shared = _shared_inventory()
    output = _local_mapping(
        "rf",
        slot=3,
        channel=2,
        lo_frequency_hz=0.0,
    )
    acquisition = _local_mapping(
        "acquisition",
        slot=5,
        channel=2,
    )
    scoped = front_panel.scoped_qcs_hardware_configuration(
        shared,
        (output, acquisition),
    )
    mapper_path = tmp_path / "calibration_only.qcs"
    config = M5300PowerCalibrationConfig(
        database_path=str(tmp_path / "calibration.db"),
        mapper_path=str(mapper_path),
        rf_channel_name="calibration_rf_output",
        acquisition_channel_name="calibration_acquisition",
        frequencies_hz=(1.0e9, 2.0e9),
        relative_amplitudes=(0.1, 0.2),
        input_reference=M5200PowerReference.qcs_voltage_50ohm(),
        expected_lo_frequency_hz=0.0,
    )
    captured = {}

    def save_mapper(configuration, path):
        captured["saved_configuration"] = configuration
        captured["saved_path"] = str(path)
        mapper_path.write_bytes(b"scoped mapper")
        return mapper_path

    stored = object()

    def run_calibration(*, config, progress_callback):
        captured["run_config"] = config
        captured["progress_callback"] = progress_callback
        return stored

    monkeypatch.setattr(calibration_gui, "save_qcs_channel_mapper", save_mapper)
    monkeypatch.setattr(
        calibration_gui,
        "run_m5300_power_calibration",
        run_calibration,
    )
    worker = calibration_gui.CalibrationWorker(
        "qcs_rf_output",
        {
            "calibration_config": config,
            "mapper_configuration": scoped,
        },
    )
    finished = QtTest.QSignalSpy(worker.finished)
    failed = QtTest.QSignalSpy(worker.failed)

    worker.run()

    assert len(failed) == 0
    assert len(finished) == 1
    assert finished[0][0] is stored
    assert captured["saved_configuration"] == scoped
    assert captured["saved_path"] == str(mapper_path)
    assert captured["run_config"].mapper_path == str(mapper_path.resolve())
    assert "connection_config" not in worker.kwargs


def test_calibration_physical_endpoints_survive_settings_round_trip():
    _application()
    shared = _shared_inventory()
    original = calibration_gui.CalibrationPanel()
    original.set_qcs_front_panel_configuration(shared)
    original.qcs_rf_output_preview.accept_qcs_front_panel_connector(
        _local_mapping(
            "rf",
            slot=3,
            channel=2,
            lo_frequency_hz=0.0,
        )
    )
    original.qcs_rf_input_preview.accept_qcs_front_panel_connector(
        _local_mapping("acquisition", slot=5, channel=2)
    )
    settings = original.settings_dict()

    restored = calibration_gui.CalibrationPanel()
    restored.load_settings(settings)
    restored.set_qcs_front_panel_configuration(shared)

    output = restored.qcs_rf_output_preview.selected_mapping()
    acquisition = restored.qcs_rf_input_preview.selected_mapping()
    assert (output["slot"], output["channel"]) == (3, 2)
    assert output["lo_frequency_hz"] == 0.0
    assert (acquisition["slot"], acquisition["channel"]) == (5, 2)
    original.close()
    restored.close()


def test_native_scoped_mapper_contains_no_unused_experiment_channels(tmp_path):
    qcs = pytest.importorskip("keysight.qcs")
    shared = _shared_inventory()
    scoped = front_panel.scoped_qcs_hardware_configuration(
        shared,
        (
            _local_mapping(
                "rf",
                slot=3,
                channel=2,
                lo_frequency_hz=0.0,
            ),
            _local_mapping("acquisition", slot=5, channel=2),
        ),
    )

    path = front_panel.save_qcs_channel_mapper(
        scoped,
        tmp_path / "calibration_only.qcs",
        qcs_module=qcs,
    )
    mapper = qcs.load(path)

    assert {str(channel.name) for channel in mapper.channels} == {
        "calibration_rf_output",
        "calibration_acquisition",
    }


def test_dc_calibration_worker_uses_its_own_mapper(monkeypatch):
    calibration_config = object()
    stored = object()
    captured = {}

    def run_dc_calibration(*, calibration_config, progress_callback):
        captured["calibration_config"] = calibration_config
        captured["progress_callback"] = progress_callback
        return stored

    monkeypatch.setattr(
        calibration_gui,
        "run_qcs_m5301_dc_output_calibration",
        run_dc_calibration,
    )
    worker = calibration_gui.CalibrationWorker(
        "qcs_dc_output",
        {"calibration_config": calibration_config},
    )
    finished = QtTest.QSignalSpy(worker.finished)
    failed = QtTest.QSignalSpy(worker.failed)

    worker.run()

    assert len(failed) == 0
    assert len(finished) == 1
    assert finished[0][0] is stored
    assert captured["calibration_config"] is calibration_config
    assert "connection_config" not in worker.kwargs
