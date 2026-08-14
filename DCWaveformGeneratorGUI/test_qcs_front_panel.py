"""Headless tests for the editable Keysight QCS front panel."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets
import pytest

import DCWaveform_Generator as gui
import qcs_chassis_renderer as chassis_renderer
import qcs_front_panel as front_panel
from noise_analysis import NoiseAcquisitionRequest, NoiseTraceCollection


_APP = None


def _application():
    global _APP
    if _APP is None:
        _APP = (
            QtWidgets.QApplication.instance()
            or QtWidgets.QApplication([])
        )
    return _APP


def _wait_for_qcs_mapper_commit(window, *, timeout_ms: int = 5000) -> None:
    """Drain Qt events until the latest background mapper request is idle."""

    app = _application()
    elapsed = QtCore.QElapsedTimer()
    elapsed.start()
    while elapsed.elapsed() < timeout_ms:
        # The connector signal first queues request capture with singleShot(0),
        # then the worker's result returns through queued cross-thread signals.
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        thread = window._qcs_mapper_commit_thread
        if (
            thread is None
            and window._qcs_mapper_commit_active is None
            and window._qcs_mapper_commit_queued is None
        ):
            # Give a pending singleShot(0) one additional opportunity to start.
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            if (
                window._qcs_mapper_commit_thread is None
                and window._qcs_mapper_commit_active is None
                and window._qcs_mapper_commit_queued is None
            ):
                return
        QtTest.QTest.qWait(5)

    thread = window._qcs_mapper_commit_thread
    pytest.fail(
        "QCS mapper commit did not finish within "
        f"{timeout_ms} ms (thread_running="
        f"{thread is not None and thread.isRunning()}, "
        f"active={window._qcs_mapper_commit_active is not None}, "
        f"queued={window._qcs_mapper_commit_queued is not None})"
    )


def _open_saved_qcs_output_window(tmp_path, *, acquisition_name=None):
    """Open one run-ready QCS output picker for mapper race tests."""

    app = _application()
    original_mapper = tmp_path / "race_original.qcs"
    original_mapper.write_bytes(b"original mapper")
    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_only",),
        {},
        acquisition_name,
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "mapper_path": str(original_mapper),
            "dc_channel_names": ["dc_only"],
            "rf_channel_names": {},
            "acquisition_channel_name": acquisition_name,
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_SAVED
            ),
            "hardware_mapper_sha256": (
                front_panel.qcs_mapper_file_sha256(original_mapper)
            ),
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    assert window._apply_qcs_front_panel_settings(payload) is True
    window._show_active_front_panel("output", window._multi_ctrl)
    app.processEvents()
    assert window._qcs_front_panel._focused_mapping == ("dc", 0)
    return app, window, experiment, original_mapper


def _wait_for_event(event, *, timeout_ms: int = 2000) -> None:
    """Wait for a worker-side threading.Event while keeping Qt responsive."""

    app = _application()
    elapsed = QtCore.QElapsedTimer()
    elapsed.start()
    while not event.is_set() and elapsed.elapsed() < timeout_ms:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        QtTest.QTest.qWait(5)
    assert event.is_set() is True


def _close_qcs_race_window(window, *release_events) -> None:
    """Release test workers, drain them, and safely destroy their window."""

    app = _application()
    for event in release_events:
        event.set()
    _wait_for_qcs_mapper_commit(window)
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def _example_configuration(name: str) -> dict:
    path = (
        Path(__file__).resolve().parent
        / "examples"
        / "qcs_hardware"
        / f"{name}.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _pixmap_png(pixmap) -> bytes:
    output = QtCore.QByteArray()
    buffer = QtCore.QBuffer(output)
    assert buffer.open(QtCore.QIODevice.WriteOnly)
    assert pixmap.save(buffer, "PNG")
    buffer.close()
    return bytes(output)


def _m5201_configuration(
    *,
    link_count: int = 1,
    lo_frequency_hz=7.25e9,
) -> dict:
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_left",),
        {},
        "digitizer",
    )
    configuration["modules"].append({"slot": 6, "model": "M5201A"})
    if link_count == 2:
        configuration["channel_mappings"].append(
            {
                "role": "unassigned",
                "logical_index": 0,
                "virtual_name": "digitizer_spare",
                "label": 0,
                "absolute_phase": True,
                "lo_frequency_hz": None,
                "slot": 5,
                "channel": 2,
            }
        )
    elif link_count != 1:
        raise ValueError("link_count must be 1 or 2")
    configuration["downconverter_links"] = [
        {
            "digitizer_slot": 5,
            "digitizer_channel": channel,
            "downconverter_slot": 6,
            "downconverter_channel": channel,
            "lo_frequency_hz": lo_frequency_hz,
        }
        for channel in range(1, link_count + 1)
    ]
    return configuration


def test_diagram_layout_resolves_runtime_role_bindings():
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_left", "dc_right"),
        {0: "rf_drive", 2: "rf_probe"},
        "digitizer",
    )

    assert configuration["chassis_model"] == "M9046A"
    assert (
        configuration["ip_address"]
        == front_panel.DEFAULT_QCS_IP_ADDRESS
    )
    assert configuration["modules"][:4] == [
        {"slot": 1, "model": "M9032A"},
        {"slot": 2, "model": "M5301A"},
        {"slot": 3, "model": "M5300A"},
        {"slot": 5, "model": "M5200A"},
    ]
    dc_names, rf_names, acquisition_name = front_panel.qcs_role_bindings(
        configuration,
        required_dc_count=2,
    )
    assert dc_names == ["dc_left", "dc_right"]
    assert rf_names == {"0": "rf_drive", "2": "rf_probe"}
    assert acquisition_name == "digitizer"


def _fake_qcs_module(
    model: str,
    slot: int,
    *,
    chassis: int = 1,
    host_controller: int = 1,
):
    return SimpleNamespace(
        HostControllerId=host_controller,
        HostIpAddress="192.168.2.105",
        ModuleInfo=SimpleNamespace(
            ModelNumber=model,
            Chassis=chassis,
            Slot=slot,
            SerialNumber=f"SN-{model}-{slot}",
            FirmwareVersion="1.2.3",
        ),
        BoardInfos=[],
    )


def test_live_qcs_inventory_uses_saved_token_and_disposes_client(tmp_path):
    token_path = tmp_path / ".qcs_token.json"
    token_path.write_text(
        json.dumps({"192.168.2.105": "saved-token"}),
        encoding="utf-8",
    )

    class FakeClient:
        def __init__(self, address):
            self.address = address
            self.token = None
            self.disposed = False
            self.GrpcChannel = SimpleNamespace(Dispose=self._dispose)

        def _dispose(self):
            self.disposed = True

        def SetAccessToken(self, token):
            self.token = token

        def GetSystemHardwareInfo(self):
            return SimpleNamespace(
                Response=SimpleNamespace(ResponseCase="Success"),
                ModuleHardwareInfos=[
                    _fake_qcs_module("M5300A", 4),
                    _fake_qcs_module("M5301A", 7),
                    _fake_qcs_module("M5201", 17),
                    _fake_qcs_module("M5200A", 18),
                ],
            )

    clients = []

    def factory(address):
        client = FakeClient(address)
        clients.append(client)
        return client

    result = front_panel.identify_qcs_hardware_configuration(
        "192.168.2.105",
        client_factory=factory,
        token_path=token_path,
    )

    assert clients[0].address == "192.168.2.105"
    assert clients[0].token == "saved-token"
    assert clients[0].disposed is True
    assert result["ip_address"] == "192.168.2.105"
    assert result["inventories"] == [
        {
            "chassis_model": "M9046A",
            "chassis": 1,
            "host_controller": 1,
            "modules": [
                {
                    "slot": 4,
                    "model": "M5300A",
                    "reported_model": "M5300A",
                    "serial_number": "SN-M5300A-4",
                    "firmware_version": "1.2.3",
                    "host_ip_address": "192.168.2.105",
                    "boards": [],
                },
                {
                    "slot": 7,
                    "model": "M5301A",
                    "reported_model": "M5301A",
                    "serial_number": "SN-M5301A-7",
                    "firmware_version": "1.2.3",
                    "host_ip_address": "192.168.2.105",
                    "boards": [],
                },
                {
                    "slot": 17,
                    "model": "M5201A",
                    "reported_model": "M5201",
                    "serial_number": "SN-M5201-17",
                    "firmware_version": "1.2.3",
                    "host_ip_address": "192.168.2.105",
                    "boards": [],
                },
                {
                    "slot": 18,
                    "model": "M5200A",
                    "reported_model": "M5200A",
                    "serial_number": "SN-M5200A-18",
                    "firmware_version": "1.2.3",
                    "host_ip_address": "192.168.2.105",
                    "boards": [],
                },
            ],
        }
    ]


def test_live_qcs_inventory_reports_authentication_failure(tmp_path):
    class FakeClient:
        def __init__(self, _address):
            self.disposed = False
            self.GrpcChannel = SimpleNamespace(
                Dispose=lambda: setattr(self, "disposed", True)
            )

        def GetSystemHardwareInfo(self):
            raise RuntimeError("Unauthenticated: invalid or expired token")

    clients = []

    def factory(address):
        client = FakeClient(address)
        clients.append(client)
        return client

    with pytest.raises(RuntimeError, match="authentication"):
        front_panel.identify_qcs_hardware_configuration(
            "192.168.2.105",
            client_factory=factory,
            token_path=tmp_path / "missing-token.json",
        )
    assert clients[0].disposed is True


def test_discovered_topology_preserves_only_compatible_mappings():
    current = _example_configuration("lab_baseline")
    inventory = {
        "chassis_model": "M9046A",
        "chassis": 1,
        "host_controller": 1,
        "modules": [
            {"slot": 4, "model": "M5300A"},
            {"slot": 7, "model": "M5301A"},
            {"slot": 17, "model": "M5200A"},
        ],
    }

    merged, removed = (
        front_panel.merge_qcs_discovered_hardware_configuration(
            current,
            inventory,
            ip_address="192.168.2.105",
        )
    )

    assert merged["chassis"] == 1
    assert merged["host_controller"] == 1
    assert merged["modules"] == inventory["modules"]
    assert [mapping["virtual_name"] for mapping in removed] == [
        "digitizer"
    ]
    assert {
        mapping["virtual_name"] for mapping in merged["channel_mappings"]
    } == {
        "gate_left",
        "gate_right",
        "qubit_drive",
        "readout_drive",
    }


def test_discovery_never_preserves_mappings_across_physical_namespaces():
    current = _example_configuration("lab_baseline")
    inventory = {
        "chassis_model": "M9046A",
        "chassis": 2,
        "host_controller": 3,
        "modules": [
            {"slot": 4, "model": "M5300A"},
            {"slot": 7, "model": "M5301A"},
            {"slot": 18, "model": "M5200A"},
        ],
    }

    merged, removed = (
        front_panel.merge_qcs_discovered_hardware_configuration(
            current,
            inventory,
            ip_address="192.168.2.105",
        )
    )

    assert merged["channel_mappings"] == []
    assert {
        mapping["virtual_name"] for mapping in removed
    } == {
        "gate_left",
        "gate_right",
        "qubit_drive",
        "readout_drive",
        "digitizer",
    }


def test_discovery_preserves_or_drops_explicit_m5201_links_with_modules():
    current = front_panel.normalize_qcs_hardware_configuration(
        _m5201_configuration()
    )
    inventory = {
        "chassis_model": "M9046A",
        "chassis": 1,
        "host_controller": 1,
        "modules": list(current["modules"]),
    }

    preserved, removed = (
        front_panel.merge_qcs_discovered_hardware_configuration(
            current,
            inventory,
            ip_address="192.168.2.105",
        )
    )
    assert removed == []
    assert preserved["downconverter_links"] == current[
        "downconverter_links"
    ]

    inventory["modules"] = [
        module
        for module in inventory["modules"]
        if module["model"] != "M5201A"
    ]
    without_m5201, removed = (
        front_panel.merge_qcs_discovered_hardware_configuration(
            current,
            inventory,
            ip_address="192.168.2.105",
        )
    )
    assert removed == []
    assert without_m5201["downconverter_links"] == []
    assert front_panel.qcs_role_bindings(
        without_m5201,
        required_dc_count=1,
    )[2] == "digitizer"


def test_compact_preview_renders_the_applied_qcs_chassis_configuration():
    _application()
    preview = front_panel.QcsFrontPanelPreview()
    preview.set_selection("rf", 0)
    preview.set_configuration(_example_configuration("lab_baseline"))
    baseline_png = _pixmap_png(preview._pixmap)

    assert preview._pixmap.size() == QtCore.QSize(1914, 652)
    assert "M5300A slot 4 ch1" in preview.binding_label.text()
    assert "LO 6 GHz" in preview.binding_label.text()

    preview.set_configuration(_example_configuration("dc_expanded"))
    assert preview._pixmap.width() == 1914
    assert preview._pixmap.height() >= 652
    assert _pixmap_png(preview._pixmap) != baseline_png
    assert "M5300A slot 8 ch1" in preview.binding_label.text()
    assert "LO 5.8 GHz" in preview.binding_label.text()
    preview.close()


def test_compact_preview_highlights_selected_physical_sma():
    _application()
    configuration = _example_configuration("lab_baseline")
    preview = front_panel.QcsFrontPanelPreview()
    preview.set_configuration(configuration)
    preview.set_selection("dc", 0)
    first_png = _pixmap_png(preview._pixmap)

    assert preview._selected_address() == (7, 1)
    preview.set_selection("dc", 1)
    second_png = _pixmap_png(preview._pixmap)

    assert preview._selected_address() == (7, 2)
    assert second_png != first_png
    assert "M5301A slot 7 ch2" in preview.binding_label.text()
    preview.close()


def test_compact_preview_skips_unchanged_configuration_and_selection(
    monkeypatch,
):
    _application()
    configuration = _example_configuration("lab_baseline")
    preview = front_panel.QcsFrontPanelPreview()
    preview.set_configuration(configuration)
    refreshes = []
    monkeypatch.setattr(
        preview,
        "_refresh_pixmap",
        lambda: refreshes.append("pixmap"),
    )

    preview.set_configuration(json.loads(json.dumps(configuration)))
    preview.set_selection("dc", 0)

    assert refreshes == []
    preview.set_selection("dc", 1)
    assert refreshes == ["pixmap"]
    preview.close()


def test_hidden_parented_preview_defers_render_until_shown(monkeypatch):
    app = _application()
    parent = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(parent)
    preview = front_panel.QcsFrontPanelPreview(parent)
    layout.addWidget(preview)
    configuration = _example_configuration("lab_baseline")
    original_pixmap = front_panel._qcs_front_panel_pixmap
    rendered_addresses = []

    def tracked_pixmap(
        candidate,
        highlighted_address=None,
        **kwargs,
    ):
        rendered_addresses.append(highlighted_address)
        return original_pixmap(
            candidate,
            highlighted_address,
            **kwargs,
        )

    monkeypatch.setattr(
        front_panel,
        "_qcs_front_panel_pixmap",
        tracked_pixmap,
    )
    preview.set_configuration(configuration)
    preview.set_selection("dc", 1)

    assert rendered_addresses == []
    assert preview._configuration is not None
    assert preview._selected_address() == (7, 2)
    assert "M5301A slot 7 ch2" in preview.binding_label.text()
    assert preview._pixmap_refresh_pending is True

    parent.show()
    app.processEvents()

    assert rendered_addresses == [(7, 2)]
    assert preview._pixmap_refresh_pending is False
    assert preview._pixmap.size() == QtCore.QSize(1914, 652)
    parent.close()


def test_compact_preview_reuses_decoded_chassis_image():
    _application()
    configuration = _example_configuration("lab_baseline")
    front_panel._cached_qcs_front_panel_png.cache_clear()
    front_panel._cached_qcs_front_panel_image.cache_clear()

    first = front_panel._qcs_front_panel_pixmap(configuration, (7, 1))
    second = front_panel._qcs_front_panel_pixmap(configuration, (7, 1))

    assert first.isNull() is False
    assert _pixmap_png(second) == _pixmap_png(first)
    image_cache = front_panel._cached_qcs_front_panel_image.cache_info()
    assert image_cache.misses == 1
    assert image_cache.hits == 1


def test_hardware_preview_skips_unchanged_backend(monkeypatch):
    _application()
    preview = gui.HardwareFrontPanelPreview()
    assert preview.currentWidget() is preview.qcs_preview
    synchronizations = []
    monkeypatch.setattr(
        preview,
        "_sync_current_size_constraints",
        lambda: synchronizations.append("size"),
    )

    preview.set_backend(gui.EXECUTION_BACKEND_QCS)
    assert synchronizations == []

    preview.set_backend(gui.EXECUTION_BACKEND_QICK)
    assert synchronizations == ["size"]
    assert preview.currentWidget() is preview.qick_preview
    preview.close()


def test_compact_preview_shows_m5201_binding_for_acquisition():
    _application()
    preview = front_panel.QcsFrontPanelPreview()
    preview.set_configuration(_m5201_configuration())
    preview.set_selection("acquisition", 0)

    assert "M5200A slot 5 ch1" in preview.binding_label.text()
    assert "via M5201A slot 6 pair 1" in preview.binding_label.text()
    assert "LO 7.25 GHz" in preview.binding_label.text()
    preview.close()


def test_editor_propagates_shared_m5201_lo_to_every_pair(tmp_path):
    app = _application()
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_left"],
            "rf_channel_names": {},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": _m5201_configuration(link_count=2),
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
        },
        output_count=1,
    )

    assert control.downconverter_table.rowCount() == 2
    first_lo = control.downconverter_table.cellWidget(0, 4)
    second_lo = control.downconverter_table.cellWidget(1, 4)
    first_lo.setText("8.125")
    app.processEvents()

    assert second_lo.text() == "8.125"
    assert [
        link["lo_frequency_hz"]
        for link in control.settings_dict()["hardware_configuration"][
            "downconverter_links"
        ]
    ] == pytest.approx([8.125e9, 8.125e9])
    control.close()


def test_editor_front_panel_updates_live_and_keeps_last_valid_layout(
    tmp_path,
):
    app = _application()
    configuration = _example_configuration("lab_baseline")
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {
                "0": "qubit_drive",
                "1": "readout_drive",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
        },
        output_count=2,
    )
    baseline_png = _pixmap_png(control._image_pixmap)
    assert control._image_pixmap.size() == QtCore.QSize(1914, 652)
    assert control.tabs.currentWidget() is control.front_panel_tab
    assert not hasattr(control, "module_table")

    assert control._set_module_at_slot(
        1,
        "M5301A",
        confirm_mapping_removal=False,
    )
    QtTest.QTest.qWait(control._preview_refresh_timer.interval() + 20)
    app.processEvents()
    updated_png = _pixmap_png(control._image_pixmap)
    assert updated_png != baseline_png

    assert (
        control._set_module_at_slot(
            17,
            "M5300A",
            confirm_mapping_removal=False,
        )
        is False
    )
    QtTest.QTest.qWait(control._preview_refresh_timer.interval() + 20)
    app.processEvents()
    assert _pixmap_png(control._image_pixmap) == updated_png
    control.close()


def test_front_panel_identify_button_uses_editable_default_ip():
    app = _application()
    control = front_panel.QcsFrontPanelControl()
    requested = []
    control.identify_requested.connect(requested.append)

    assert control.ip_address.text() == "192.168.2.105"
    assert (
        control.identify_hardware_button.parentWidget()
        is control.front_panel_tab
    )
    control.ip_address.setText("192.168.2.44")
    control.identify_hardware_button.click()
    app.processEvents()

    assert requested == ["192.168.2.44"]
    control.set_identifying(True, "Identifying...")
    assert control.tabs.isEnabled() is False
    assert control.identify_hardware_button.text() == "Identifying..."
    control.set_identifying(False, "Ready")
    assert control.tabs.isEnabled() is True
    assert (
        control.identify_hardware_button.text()
        == "Identify Hardware Configuration"
    )
    control.close()


def test_discovered_hardware_becomes_current_configuration(tmp_path):
    app = _application()
    configuration = _example_configuration("lab_baseline")
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {
                "0": "qubit_drive",
                "1": "readout_drive",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
        },
        output_count=2,
    )
    applied = []
    control.settings_applied.connect(applied.append)

    assert control.apply_discovered_hardware_configuration(
        {
            "ip_address": "192.168.2.105",
            "inventories": [
                {
                    "chassis_model": "M9046A",
                    "chassis": 1,
                    "host_controller": 1,
                    "modules": [
                        {"slot": 4, "model": "M5300A"},
                        {"slot": 7, "model": "M5301A"},
                        {"slot": 18, "model": "M5200A"},
                    ],
                }
            ],
        }
    )
    app.processEvents()

    assert len(applied) == 1
    assert applied[0]["hardware_configuration"] == (
        front_panel.normalize_qcs_hardware_configuration(configuration)
    )
    assert "Identified and applied" in control.status.text()
    control.close()


def test_discovered_hardware_never_silently_reroutes_channels(
    monkeypatch,
    tmp_path,
):
    _application()
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {
                "0": "qubit_drive",
                "1": "readout_drive",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": _example_configuration(
                "lab_baseline"
            ),
        },
        output_count=2,
    )
    applied = []
    control.settings_applied.connect(applied.append)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args: QtWidgets.QMessageBox.Yes,
    )

    assert control.apply_discovered_hardware_configuration(
        {
            "ip_address": "192.168.2.105",
            "inventories": [
                {
                    "chassis_model": "M9046A",
                    "chassis": 1,
                    "host_controller": 1,
                    "modules": [
                        {"slot": 4, "model": "M5300A"},
                        {"slot": 7, "model": "M5301A"},
                        {"slot": 17, "model": "M5200A"},
                    ],
                }
            ],
        }
    )

    assert applied == []
    assert control._module_models_by_slot[17] == "M5200A"
    assert 18 not in control._module_models_by_slot
    assert {
        mapping["virtual_name"]
        for mapping in control._mappings_from_widgets()
    } == {
        "gate_left",
        "gate_right",
        "qubit_drive",
        "readout_drive",
    }
    assert "No channels will be automatically rerouted" not in (
        control.status.text()
    )
    assert "removed 1 incompatible mapping" in control.status.text()
    control.close()


def test_focused_sma_click_creates_missing_dc_mapping(tmp_path):
    _application()
    configuration = _example_configuration("lab_baseline")
    configuration["channel_mappings"] = [
        mapping
        for mapping in configuration["channel_mappings"]
        if mapping["role"] != "dc"
    ]
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_ch_1"],
            "rf_channel_names": {
                "0": "qubit_drive",
                "1": "readout_drive",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
        },
        output_count=1,
    )

    assert control.focus_mapping("dc", 0) is True
    assert control._focused_mapping_row() is None
    assert control.select_connector(7, 3) is True

    mapping = next(
        mapping
        for mapping in control._mappings_from_widgets()
        if mapping["role"] == "dc"
    )
    assert mapping["logical_index"] == 0
    assert mapping["virtual_name"] == "dc_ch_1"
    assert (mapping["slot"], mapping["channel"]) == (7, 3)
    assert "Created and selected" in control.status.text()
    control.close()


def test_focused_sma_click_creates_missing_rf_mapping(tmp_path):
    _application()
    configuration = _example_configuration("lab_baseline")
    configuration["channel_mappings"] = [
        mapping
        for mapping in configuration["channel_mappings"]
        if not (
            mapping["role"] == "rf" and mapping["logical_index"] == 1
        )
    ]
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {
                "0": "qubit_drive",
                "1": "readout_drive",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
        },
        output_count=2,
    )

    assert control.focus_mapping("rf", 1) is True
    assert control._focused_mapping_row() is None
    assert control.select_connector(7, 3) is True

    mapping = next(
        mapping
        for mapping in control._mappings_from_widgets()
        if mapping["role"] == "rf" and mapping["logical_index"] == 1
    )
    assert mapping["virtual_name"] == "readout_drive"
    assert mapping["absolute_phase"] is True
    assert mapping["lo_frequency_hz"] is None
    assert (mapping["slot"], mapping["channel"]) == (7, 3)
    assert control.working_source_bindings()[1] == {
        0: "qubit_drive",
        1: "readout_drive",
    }
    assert "Created and selected" in control.status.text()
    control.close()


def test_focused_sma_click_generates_first_rf_binding(tmp_path):
    _application()
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_ch_1",),
        {},
        None,
    )
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_ch_1"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
            "hardware_configuration": configuration,
        },
        output_count=1,
    )

    assert control.focus_mapping("rf", 0) is True
    assert control.select_connector(2, 2) is True

    mapping = next(
        mapping
        for mapping in control._mappings_from_widgets()
        if mapping["role"] == "rf"
    )
    assert mapping["logical_index"] == 0
    assert mapping["virtual_name"] == "rf_drive"
    assert (mapping["slot"], mapping["channel"]) == (2, 2)
    assert control.working_source_bindings()[1] == {0: "rf_drive"}
    control.close()


def test_focused_m5300_sma_requests_required_lo(monkeypatch, tmp_path):
    _application()
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_ch_1",),
        {},
        None,
    )
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_ch_1"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
            "hardware_configuration": configuration,
        },
        output_count=1,
    )
    prompts = []

    def fake_get_double(*args):
        prompts.append(args)
        return 6.25, True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getDouble", fake_get_double)
    assert control.focus_mapping("rf", 0) is True
    assert control.select_connector(3, 1) is True

    mapping = next(
        mapping
        for mapping in control._mappings_from_widgets()
        if mapping["role"] == "rf"
    )
    assert len(prompts) == 1
    assert mapping["virtual_name"] == "rf_drive"
    assert mapping["lo_frequency_hz"] == pytest.approx(6.25e9)
    assert (mapping["slot"], mapping["channel"]) == (3, 1)
    control.close()


def test_m5300_lo_setter_emits_one_canonical_mapper_change(tmp_path):
    _application()
    mapper_path = tmp_path / "saved_mapper.qcs"
    mapper_path.write_bytes(b"saved M5300 mapper")
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(mapper_path),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {
                "0": "qubit_drive",
                "1": "readout_drive",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": _example_configuration(
                "lab_baseline"
            ),
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_SAVED
            ),
            "hardware_mapper_sha256": (
                front_panel.qcs_mapper_file_sha256(mapper_path)
            ),
        },
        output_count=2,
    )
    changes = []
    control.m5300_lo_frequency_changed.connect(
        lambda *args: changes.append(args)
    )

    assert control.set_m5300_lo_frequency(4, 1, 6.5e9) is True
    assert changes == [("rf", 0, 4, 1, 6.5e9)]
    mapping = next(
        mapping
        for mapping in control.working_configuration()["channel_mappings"]
        if mapping["role"] == "rf" and mapping["logical_index"] == 0
    )
    assert mapping["lo_frequency_hz"] == pytest.approx(6.5e9)
    assert (
        control._configuration_state
        == front_panel.QCS_HARDWARE_STATE_DRAFT
    )
    assert control._mapper_file_sha256 is None

    # Reapplying the canonical value is a successful no-op and must not
    # schedule a second native mapper write.
    assert control.set_m5300_lo_frequency(4, 1, 6.5e9) is True
    assert changes == [("rf", 0, 4, 1, 6.5e9)]
    control.close()


def test_right_click_m5300_sma_prompts_with_current_lo(
    monkeypatch,
    tmp_path,
):
    app = _application()
    configuration = _example_configuration("lab_baseline")
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {
                "0": "qubit_drive",
                "1": "readout_drive",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
        },
        output_count=2,
    )
    control.resize(1200, 720)
    control.show()
    control.show_front_panel()
    app.processEvents()
    prompts = []
    changes = []

    def fake_get_double(*args):
        prompts.append(args)
        return 6.75, True

    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getDouble",
        fake_get_double,
    )
    control.m5300_lo_frequency_changed.connect(
        lambda *args: changes.append(args)
    )
    source_x = (
        chassis_renderer.DEFAULT_CHASSIS_LEFT_MARGIN
        + 3 * chassis_renderer.DEFAULT_SLOT_WIDTH
        + 185 * 2 * chassis_renderer.DEFAULT_SLOT_WIDTH / 600
    )
    source_y = (
        chassis_renderer.DEFAULT_CHASSIS_HEADER_HEIGHT
        + chassis_renderer.DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT
        + 285 * chassis_renderer.DEFAULT_PANEL_HEIGHT / 1300
    )
    connector = chassis_renderer.qcs_chassis_connector_at_point(
        configuration,
        source_x,
        source_y,
        role="rf",
    )
    assert (connector["slot"], connector["channel"]) == (4, 1)
    displayed = control.reference_label.pixmap()
    click_position = QtCore.QPoint(
        round(source_x * displayed.width() / control._image_pixmap.width()),
        round(source_y * displayed.height() / control._image_pixmap.height()),
    )

    QtTest.QTest.mouseClick(
        control.reference_label,
        QtCore.Qt.RightButton,
        pos=click_position,
    )
    app.processEvents()

    assert len(prompts) == 1
    assert prompts[0][3] == pytest.approx(6.0)
    assert changes == [("rf", 0, 4, 1, 6.75e9)]
    assert "6.75 GHz" in control.status.text()
    control.close()
    app.processEvents()


def test_imported_mapper_rejects_m5300_lo_change_without_prompt(
    monkeypatch,
    tmp_path,
):
    _application()
    mapper_path = tmp_path / "imported.qcs"
    mapper_path.write_bytes(b"third-party mapper")
    configuration = _example_configuration("lab_baseline")
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(mapper_path),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {
                "0": "qubit_drive",
                "1": "readout_drive",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_IMPORTED
            ),
            "hardware_mapper_sha256": (
                front_panel.qcs_mapper_file_sha256(mapper_path)
            ),
        },
        output_count=2,
    )
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getDouble",
        lambda *_args: pytest.fail(
            "read-only imported mapper must not open an LO editor"
        ),
    )
    changes = []
    control.m5300_lo_frequency_changed.connect(
        lambda *args: changes.append(args)
    )

    assert control._prompt_m5300_lo_frequency(4, 1) is False
    assert control.set_m5300_lo_frequency(4, 1, 6.5e9) is False
    assert changes == []
    assert "imported" in control.status.text().lower()
    assert control.working_configuration() == (
        front_panel.normalize_qcs_hardware_configuration(configuration)
    )
    control.close()


def test_rf_mapping_can_move_from_m5300_to_m5301(tmp_path):
    _application()
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {
                "0": "qubit_drive",
                "1": "readout_drive",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": _example_configuration(
                "lab_baseline"
            ),
        },
        output_count=2,
    )

    assert control.focus_mapping("rf", 0) is True
    assert control.select_connector(7, 3) is True
    mapping = next(
        mapping
        for mapping in control._mappings_from_widgets()
        if mapping["role"] == "rf" and mapping["logical_index"] == 0
    )
    assert (mapping["slot"], mapping["channel"]) == (7, 3)
    assert mapping["lo_frequency_hz"] is None
    control.close()


def test_focused_sma_click_creates_missing_acquisition_mapping(tmp_path):
    _application()
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_ch_1",),
        {},
        None,
    )
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_ch_1"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
            "hardware_configuration": configuration,
        },
        output_count=1,
    )

    assert control.focus_mapping("acquisition", 0) is True
    assert control._focused_mapping_row() is None
    assert control.select_connector(5, 2) is True

    mapping = next(
        mapping
        for mapping in control._mappings_from_widgets()
        if mapping["role"] == "acquisition"
    )
    assert mapping["logical_index"] == 0
    assert mapping["virtual_name"] == "digitizer"
    assert mapping["absolute_phase"] is True
    assert (mapping["slot"], mapping["channel"]) == (5, 2)
    assert control.working_source_bindings()[2] == "digitizer"
    assert "Created and selected" in control.status.text()
    control.close()


def test_stability_path_focus_routes_sma_click_by_module(tmp_path):
    _application()
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_ch_1",),
        {0: "rf_drive"},
        "digitizer",
    )
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_ch_1"],
            "rf_channel_names": {"0": "rf_drive"},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
        },
        output_count=1,
    )
    selected = []
    control.connector_selected.connect(
        lambda role, logical_index, slot, channel, changed: selected.append(
            (role, logical_index, slot, channel, changed)
        )
    )

    assert control.focus_rf_acquisition_path(0, 0) is True
    initial_configuration = control._preview_configuration_from_widgets()
    initial_acquisition = next(
        (
            int(mapping["slot"]),
            int(mapping["channel"]),
        )
        for mapping in initial_configuration["channel_mappings"]
        if mapping["role"] == "acquisition"
    )
    assert control.select_connector(2, 2) is True
    assert control._focused_mapping == ("rf", 0)
    assert control._focused_mapping_addresses(
        control._preview_configuration_from_widgets()
    ) == ((2, 2), initial_acquisition)
    rf_mapping = next(
        mapping
        for mapping in control._mappings_from_widgets()
        if mapping["role"] == "rf"
    )
    assert (rf_mapping["slot"], rf_mapping["channel"]) == (2, 2)

    assert control.select_connector(5, 2) is True
    assert control._focused_mapping == ("acquisition", 0)
    acquisition_mapping = next(
        mapping
        for mapping in control._mappings_from_widgets()
        if mapping["role"] == "acquisition"
    )
    assert (
        acquisition_mapping["slot"],
        acquisition_mapping["channel"],
    ) == (5, 2)
    assert control._focused_mapping_addresses(
        control._preview_configuration_from_widgets()
    ) == ((2, 2), (5, 2))

    # Selecting the RF endpoint again must retain the acquisition highlight.
    assert control.select_connector(2, 3) is True
    assert control._focused_mapping == ("rf", 0)
    assert control._focused_mapping_addresses(
        control._preview_configuration_from_widgets()
    ) == ((2, 3), (5, 2))
    assert [item[:4] for item in selected] == [
        ("rf", 0, 2, 2),
        ("acquisition", 0, 5, 2),
        ("rf", 0, 2, 3),
    ]

    # Leaving the Stability path restores the strict single-role picker.
    assert control.focus_mapping("dc", 0) is True
    assert control._rf_acquisition_path_focus is None
    assert control.select_connector(5, 3) is False
    control.close()


def test_qcs_front_panel_exposes_only_graphical_chassis():
    app = _application()
    control = front_panel.QcsFrontPanelControl()
    control.show()
    app.processEvents()

    assert control.tabs.count() == 1
    assert control.tabs.tabText(0) == "Front panel"
    assert control.tabs.indexOf(control.mapping_tab) == -1
    assert control.mapping_tab.isVisible() is False
    assert control.mapping_table is not None
    assert control.validate_button.isVisible() is False
    assert control.write_mapper_button.isVisible() is False
    assert control.apply_button.isVisible() is False
    assert control.reference_label.isVisible() is True

    QtTest.QTest.mouseClick(
        control.advanced_hardware_button,
        QtCore.Qt.LeftButton,
    )
    app.processEvents()
    assert control.tabs.count() == 1
    assert control.mapping_tab.isVisible() is True
    assert control.validate_button.isVisible() is True
    assert control.write_mapper_button.isVisible() is True
    assert control.apply_button.isVisible() is True
    control.set_editing_enabled(False)
    app.processEvents()
    assert control._advanced_hardware_dialog.isVisible() is False
    assert control.mapping_tab.isEnabled() is False
    assert control.advanced_hardware_button.isEnabled() is False
    control.set_editing_enabled(True)

    control.close()
    app.processEvents()


def test_acquisition_sma_selection_never_reroutes_explicit_m5201_link(
    tmp_path,
):
    _application()
    configuration = _m5201_configuration()
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_left"],
            "rf_channel_names": {},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
        },
        output_count=1,
    )
    original_links = control._downconverter_links_from_widgets()

    assert control.focus_mapping("acquisition", 0) is True
    assert control.select_connector(5, 2) is False

    mapping = next(
        mapping
        for mapping in control._mappings_from_widgets()
        if mapping["role"] == "acquisition"
    )
    assert (mapping["slot"], mapping["channel"]) == (5, 1)
    assert control._downconverter_links_from_widgets() == original_links
    assert "physical cable cannot be rerouted automatically" in (
        control.status.text()
    )
    control.close()


def test_acquisition_m5201_picture_click_opens_route_dialog(
    monkeypatch,
    tmp_path,
):
    app = _application()
    configuration = _m5201_configuration()
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_left"],
            "rf_channel_names": {},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
        },
        output_count=1,
    )
    control.resize(1200, 720)
    control.show()
    assert control.focus_mapping("acquisition", 0)
    app.processEvents()
    monkeypatch.setattr(
        control,
        "_show_module_menu_for_slot",
        lambda *_args: pytest.fail(
            "M5201 acquisition click must not open the module menu"
        ),
    )

    source_x = (
        chassis_renderer.DEFAULT_CHASSIS_LEFT_MARGIN
        + 5 * chassis_renderer.DEFAULT_SLOT_WIDTH
        + 92 * chassis_renderer.DEFAULT_SLOT_WIDTH / 300
    )
    source_y = (
        chassis_renderer.DEFAULT_CHASSIS_HEADER_HEIGHT
        + chassis_renderer.DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT
        + 750 * chassis_renderer.DEFAULT_PANEL_HEIGHT / 1300
    )
    connector = chassis_renderer.qcs_chassis_connector_at_point(
        configuration,
        source_x,
        source_y,
        role="downconverter",
    )
    assert (connector["slot"], connector["channel"]) == (6, 3)
    displayed = control.reference_label.pixmap()
    click_position = QtCore.QPoint(
        round(source_x * displayed.width() / control._image_pixmap.width()),
        round(source_y * displayed.height() / control._image_pixmap.height()),
    )
    QtTest.QTest.mouseClick(
        control.reference_label,
        QtCore.Qt.LeftButton,
        pos=click_position,
    )
    app.processEvents()

    dialog = control._m5201_route_dialog
    assert dialog.isVisible() is True
    assert dialog.module_label.text() == "M5201A slot 6"
    assert dialog.pair_combo.currentData() == 3
    assert tuple(dialog.digitizer_combo.currentData()) == (5, 1)
    assert dialog.lo_frequency_ghz.value() == pytest.approx(7.25)
    dialog.close()
    app.processEvents()

    # A neutral faceplate click opens the same route dialog and selects the
    # currently linked pair rather than the generic replace/remove menu.
    source_x = (
        chassis_renderer.DEFAULT_CHASSIS_LEFT_MARGIN
        + 5.75 * chassis_renderer.DEFAULT_SLOT_WIDTH
    )
    source_y = (
        chassis_renderer.DEFAULT_CHASSIS_HEADER_HEIGHT
        + chassis_renderer.DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT
        + 20
    )
    displayed = control.reference_label.pixmap()
    click_position = QtCore.QPoint(
        round(source_x * displayed.width() / control._image_pixmap.width()),
        round(source_y * displayed.height() / control._image_pixmap.height()),
    )
    QtTest.QTest.mouseClick(
        control.reference_label,
        QtCore.Qt.LeftButton,
        pos=click_position,
    )
    app.processEvents()
    assert dialog.isVisible() is True
    assert dialog.pair_combo.currentData() == 1
    dialog.close()
    control.close()
    app.processEvents()


def test_m5201_dialog_creates_acquisition_mapping_and_explicit_link(
    tmp_path,
):
    app = _application()
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_left",),
        {},
        None,
    )
    configuration["modules"].append({"slot": 6, "model": "M5201A"})
    configuration = front_panel.normalize_qcs_hardware_configuration(
        configuration
    )
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_left"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
            "hardware_configuration": configuration,
        },
        output_count=1,
    )
    assert control.focus_mapping("acquisition", 0)
    selected = []
    control.connector_selected.connect(
        lambda *args: selected.append(args)
    )
    assert control.show_m5201_route_dialog(6, 2)
    dialog = control._m5201_route_dialog
    address_index = dialog._combo_index_for_address(
        dialog.digitizer_combo,
        (5, 3),
    )
    assert address_index >= 0
    dialog.digitizer_combo.setCurrentIndex(address_index)
    dialog.lo_frequency_ghz.setValue(8.125)
    QtTest.QTest.mouseClick(dialog.save_button, QtCore.Qt.LeftButton)
    app.processEvents()

    # This isolated control has no experiment commit callback. The route is
    # staged, but the dialog stays open until automatic mapper application is
    # actually acknowledged.
    assert dialog.isVisible() is True
    working = control.working_configuration()
    acquisition = next(
        mapping
        for mapping in working["channel_mappings"]
        if mapping["role"] == "acquisition"
    )
    assert acquisition["virtual_name"] == "digitizer"
    assert acquisition["absolute_phase"] is True
    assert (acquisition["slot"], acquisition["channel"]) == (5, 3)
    assert working["downconverter_links"] == [
        {
            "digitizer_slot": 5,
            "digitizer_channel": 3,
            "downconverter_slot": 6,
            "downconverter_channel": 2,
            "lo_frequency_hz": 8.125e9,
        }
    ]
    assert control.working_source_bindings()[2] == "digitizer"
    assert selected == [("acquisition", 0, 5, 3, True)]
    dialog.close()
    control.close()


def test_m5201_route_edit_replaces_link_and_propagates_shared_lo(tmp_path):
    _application()
    configuration = _m5201_configuration(link_count=2)
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_left"],
            "rf_channel_names": {},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
        },
        output_count=1,
    )
    assert control.focus_mapping("acquisition", 0)
    assert control.configure_m5201_route(
        downconverter_slot=6,
        downconverter_pair=1,
        digitizer_slot=5,
        digitizer_channel=1,
        lo_frequency_hz=8.5e9,
    )

    links = control.working_configuration()["downconverter_links"]
    assert len(links) == 2
    assert {
        (link["downconverter_channel"], link["digitizer_channel"])
        for link in links
    } == {(1, 1), (2, 2)}
    assert {link["lo_frequency_hz"] for link in links} == {8.5e9}
    control.close()


def test_m5201_existing_pair_swaps_spare_mapping_without_losing_links(
    tmp_path,
):
    app = _application()
    configuration = _m5201_configuration(link_count=2)
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_left"],
            "rf_channel_names": {},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
        },
        output_count=1,
    )
    assert control.focus_mapping("acquisition", 0)
    assert control.show_m5201_route_dialog(6, 2)
    dialog = control._m5201_route_dialog

    # Pair 2's existing cable must be selected even though its M5200 endpoint
    # currently has a spare virtual mapping.
    assert tuple(dialog.digitizer_combo.currentData()) == (5, 2)
    assert "Current cable: pair 2 -> M5200A slot 5 CH2" in (
        dialog.route_note.text()
    )
    assert control.configure_m5201_route(
        downconverter_slot=6,
        downconverter_pair=2,
        digitizer_slot=5,
        digitizer_channel=2,
        lo_frequency_hz=8.25e9,
    )

    working = control.working_configuration()
    mappings = {
        mapping["virtual_name"]: (
            mapping["role"],
            mapping["slot"],
            mapping["channel"],
        )
        for mapping in working["channel_mappings"]
    }
    assert mappings["digitizer"] == ("acquisition", 5, 2)
    assert mappings["digitizer_spare"] == ("unassigned", 5, 1)
    assert {
        (
            link["downconverter_channel"],
            link["digitizer_channel"],
        )
        for link in working["downconverter_links"]
    } == {(1, 1), (2, 2)}
    assert {
        link["lo_frequency_hz"]
        for link in working["downconverter_links"]
    } == {8.25e9}
    dialog.close()
    control.close()
    app.processEvents()


def test_m5201_route_dialog_closes_when_selection_context_changes(tmp_path):
    app = _application()
    settings = {
        "mapper_path": str(tmp_path / "mapper.qcs"),
        "dc_channel_names": ["dc_left"],
        "rf_channel_names": {},
        "acquisition_channel_name": "digitizer",
        "hardware_configuration": _m5201_configuration(),
    }
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(settings, output_count=1)

    assert control.focus_mapping("acquisition", 0)
    assert control.show_m5201_route_dialog(6, 1)
    assert control._m5201_route_dialog.isVisible() is True
    assert control.focus_mapping("dc", 0)
    assert control._m5201_route_dialog.isVisible() is False

    assert control.focus_mapping("acquisition", 0)
    assert control.show_m5201_route_dialog(6, 1)
    control.clear_mapping_focus()
    assert control._m5201_route_dialog.isVisible() is False

    assert control.focus_mapping("acquisition", 0)
    assert control.show_m5201_route_dialog(6, 1)
    control.set_settings(settings, output_count=1)
    assert control._m5201_route_dialog.isVisible() is False
    control.close()
    app.processEvents()


def test_imported_mapper_blocks_m5201_route_changes_atomically(tmp_path):
    _application()
    imported_mapper = tmp_path / "imported_m5201.qcs"
    original_bytes = b"opaque imported mapper with private settings"
    imported_mapper.write_bytes(original_bytes)
    configuration = front_panel.normalize_qcs_hardware_configuration(
        _m5201_configuration(link_count=2)
    )
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(imported_mapper),
            "dc_channel_names": ["dc_left"],
            "rf_channel_names": {},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_IMPORTED
            ),
            "hardware_mapper_sha256": (
                front_panel.qcs_mapper_file_sha256(imported_mapper)
            ),
        },
        output_count=1,
    )
    assert control.focus_mapping("acquisition", 0)
    before = control.working_configuration()

    assert control.show_m5201_route_dialog(6, 2) is False
    assert control.configure_m5201_route(
        downconverter_slot=6,
        downconverter_pair=2,
        digitizer_slot=5,
        digitizer_channel=2,
        lo_frequency_hz=8.25e9,
    ) is False
    assert control.working_configuration() == before
    assert imported_mapper.read_bytes() == original_bytes
    assert control._m5201_route_dialog.isVisible() is False
    control.close()


def test_focused_dc_mapping_selects_and_swaps_m5301_smas(tmp_path):
    _application()
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {
                "0": "qubit_drive",
                "1": "readout_drive",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": _example_configuration(
                "lab_baseline"
            ),
        },
        output_count=2,
    )

    assert control.focus_mapping("dc", 0)
    assert control._focused_mapping == ("dc", 0)
    assert control._focused_mapping_address(
        control._preview_configuration_from_widgets()
    ) == (7, 1)

    # CH2 is occupied by dc logical output 1, so the two DC mappings swap.
    assert control.select_connector(7, 2)
    mappings = {
        (mapping["role"], mapping["logical_index"]): mapping
        for mapping in control._mappings_from_widgets()
    }
    assert (
        mappings[("dc", 0)]["slot"],
        mappings[("dc", 0)]["channel"],
    ) == (7, 2)
    assert (
        mappings[("dc", 1)]["slot"],
        mappings[("dc", 1)]["channel"],
    ) == (7, 1)
    assert "Swapped" in control.status.text()

    # DC output selection cannot choose an M5300 RF SMA.
    assert control.select_connector(4, 3) is False
    mappings_after_rejection = {
        (mapping["role"], mapping["logical_index"]): mapping
        for mapping in control._mappings_from_widgets()
    }
    assert mappings_after_rejection == mappings
    assert "requires a channel SMA on M5301A" in control.status.text()
    control.close()


def test_full_front_panel_click_rebinds_focused_awg_output_sma(tmp_path):
    app = _application()
    configuration = _example_configuration("lab_baseline")
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {
                "0": "qubit_drive",
                "1": "readout_drive",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
        },
        output_count=2,
    )
    control.resize(1200, 720)
    control.show()
    assert control.focus_mapping("dc", 0)
    app.processEvents()

    connector = chassis_renderer.qcs_chassis_connector_at_point(
        configuration,
        (
            chassis_renderer.DEFAULT_CHASSIS_LEFT_MARGIN
            + 6 * chassis_renderer.DEFAULT_SLOT_WIDTH
            + 95 * chassis_renderer.DEFAULT_SLOT_WIDTH / 300
        ),
        (
            chassis_renderer.DEFAULT_CHASSIS_HEADER_HEIGHT
            + chassis_renderer.DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT
            + 1080 * chassis_renderer.DEFAULT_PANEL_HEIGHT / 1300
        ),
        role="dc",
    )
    assert connector["slot"] == 7
    assert connector["channel"] == 4
    source_x, source_y = connector["center"]
    displayed = control.reference_label.pixmap()
    click_position = QtCore.QPoint(
        round(
            source_x
            * displayed.width()
            / control._image_pixmap.width()
        ),
        round(
            source_y
            * displayed.height()
            / control._image_pixmap.height()
        ),
    )
    QtTest.QTest.mouseClick(
        control.reference_label,
        QtCore.Qt.LeftButton,
        pos=click_position,
    )
    app.processEvents()

    mapping = next(
        mapping
        for mapping in control._mappings_from_widgets()
        if mapping["role"] == "dc" and mapping["logical_index"] == 0
    )
    assert (mapping["slot"], mapping["channel"]) == (7, 4)
    assert control._focused_mapping_address(
        control._preview_configuration_from_widgets()
    ) == (7, 4)
    assert "updated automatically" in control.status.text()
    control.close()
    app.processEvents()


def test_chassis_board_menu_installs_replaces_and_removes_modules(tmp_path):
    app = _application()
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {
                "0": "qubit_drive",
                "1": "readout_drive",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": _example_configuration(
                "lab_baseline"
            ),
        },
        output_count=2,
    )

    empty_menu = control._create_module_menu(1)
    empty_actions = {
        action.data(): action
        for action in empty_menu.actions()
        if action.data() is not None
    }
    assert set(empty_actions) == set(front_panel.QCS_MODULE_MODELS)
    assert "__remove__" not in empty_actions
    control.set_editing_enabled(False)
    empty_actions["M5301A"].trigger()
    assert control._module_covering_slot(1) == (1, None)
    control.set_editing_enabled(True)
    empty_actions["M5301A"].trigger()
    app.processEvents()
    assert control._module_covering_slot(1) == (1, "M5301A")

    installed_menu = control._create_module_menu(1)
    installed_actions = {
        action.data(): action
        for action in installed_menu.actions()
        if action.data() is not None
    }
    assert set(front_panel.QCS_MODULE_MODELS).issubset(installed_actions)
    assert installed_actions["M5301A"].isChecked() is True
    assert installed_actions["__remove__"].text() == "Remove M5301A"
    installed_actions["__remove__"].trigger()
    app.processEvents()
    assert control._module_covering_slot(1) == (1, None)

    # Clicking either half of a two-slot M5300 operates on its start slot.
    assert control._module_covering_slot(5) == (4, "M5300A")
    continuation_menu = control._create_module_menu(5)
    continuation_actions = {
        action.data(): action
        for action in continuation_menu.actions()
        if action.data() is not None
    }
    assert continuation_actions["M5300A"].isChecked() is True
    assert continuation_actions["__remove__"].text() == "Remove M5300A"
    assert control._remove_module_at_slot(
        5,
        confirm_mapping_removal=False,
    )
    assert control._module_covering_slot(4) == (4, None)
    assert control._module_covering_slot(5) == (5, None)
    assert all(
        int(mapping["slot"]) != 4
        for mapping in control._mappings_from_widgets()
    )

    # A two-slot module cannot overlap the M5200 already in slot 18.
    slot_seventeen_menu = control._create_module_menu(17)
    slot_seventeen_actions = {
        action.data(): action
        for action in slot_seventeen_menu.actions()
        if action.data() is not None
    }
    assert slot_seventeen_actions["M5300A"].isEnabled() is False
    control.close()
    app.processEvents()


def test_empty_slot_install_confirms_before_removing_incompatible_mapping(
    monkeypatch,
    tmp_path,
):
    _application()
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {"0": "rf_drive"},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": _example_configuration(
                "lab_baseline"
            ),
        },
        output_count=2,
    )
    control._insert_mapping(
        {
            "role": "unassigned",
            "logical_index": 0,
            "virtual_name": "future_slot_one",
            "slot": 1,
            "channel": 1,
        }
    )
    mapping_count = control.mapping_table.rowCount()
    confirmations = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: (
            confirmations.append(args)
            or QtWidgets.QMessageBox.No
        ),
    )
    assert control._set_module_at_slot(1, "M9032A") is False
    assert len(confirmations) == 1
    assert control.mapping_table.rowCount() == mapping_count
    assert control._module_covering_slot(1) == (1, None)

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args: QtWidgets.QMessageBox.Yes,
    )
    assert control._set_module_at_slot(1, "M9032A") is True
    assert control.mapping_table.rowCount() == mapping_count - 1
    assert control._module_covering_slot(1) == (1, "M9032A")
    control.close()


def test_front_panel_image_click_resolves_slot_and_respects_editor_lock(
    monkeypatch,
    tmp_path,
):
    app = _application()
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {"0": "rf_drive"},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": _example_configuration(
                "lab_baseline"
            ),
        },
        output_count=2,
    )
    control.resize(1200, 720)
    control.show()
    control.show_front_panel()
    app.processEvents()

    requested_slots = []
    monkeypatch.setattr(
        control,
        "_show_module_menu_for_slot",
        lambda slot, _position: requested_slots.append(slot),
    )
    displayed = control.reference_label.pixmap()
    assert displayed is not None and not displayed.isNull()
    source_x = (
        chassis_renderer.DEFAULT_CHASSIS_LEFT_MARGIN
        + 4.5 * chassis_renderer.DEFAULT_SLOT_WIDTH
    )
    source_y = (
        chassis_renderer.DEFAULT_CHASSIS_HEADER_HEIGHT
        + chassis_renderer.DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT
        + 20
    )
    click_position = QtCore.QPoint(
        int(
            source_x
            * displayed.width()
            / control._image_pixmap.width()
        ),
        int(
            source_y
            * displayed.height()
            / control._image_pixmap.height()
        ),
    )
    QtTest.QTest.mouseClick(
        control.reference_label,
        QtCore.Qt.LeftButton,
        pos=click_position,
    )
    assert requested_slots == [5]

    QtTest.QTest.mouseClick(
        control.reference_label,
        QtCore.Qt.LeftButton,
        pos=QtCore.QPoint(1, 1),
    )
    assert requested_slots == [5]

    control.set_editing_enabled(False)
    QtTest.QTest.mouseClick(
        control.reference_label,
        QtCore.Qt.LeftButton,
        pos=click_position,
    )
    assert requested_slots == [5]
    control.close()
    app.processEvents()


def test_missing_renderer_reports_unavailable_instead_of_legacy_image(
    monkeypatch,
    tmp_path,
):
    _application()
    configuration = _example_configuration("lab_baseline")
    monkeypatch.setattr(front_panel, "_render_qcs_chassis_png", None)
    front_panel._cached_qcs_front_panel_png.cache_clear()

    preview = front_panel.QcsFrontPanelPreview()
    preview.set_configuration(configuration)
    assert preview._pixmap.isNull()
    assert "preview unavailable" in preview.image_label.text()

    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {
                "0": "qubit_drive",
                "1": "readout_drive",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
        },
        output_count=2,
    )
    assert control._image_pixmap.isNull()
    assert "preview is unavailable" in control.reference_label.text()
    control.close()
    preview.close()
    front_panel._cached_qcs_front_panel_png.cache_clear()


def test_front_panel_control_applies_edited_bindings(monkeypatch, tmp_path):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_left", "dc_right"],
            "rf_channel_names": {"0": "rf_drive"},
            "acquisition_channel_name": "digitizer",
        },
        output_count=2,
    )
    assert control._image_pixmap.isNull() is False
    assert control.mapping_table.rowCount() == 4

    control.mapping_table.cellWidget(0, 2).setText("gate_left")
    applied = []
    control.settings_applied.connect(applied.append)
    control.apply_button.click()
    app.processEvents()

    assert len(applied) == 1
    assert applied[0]["dc_channel_names"] == ["gate_left", "dc_right"]
    assert applied[0]["rf_channel_names"] == {"0": "rf_drive"}
    assert applied[0]["acquisition_channel_name"] == "digitizer"
    assert (
        applied[0]["hardware_configuration_state"]
        == front_panel.QCS_HARDWARE_STATE_DRAFT
    )
    assert "unsaved hardware draft" in control.status.text()
    control.diagram_defaults_button.click()
    assert control.settings_dict()["dc_channel_names"][0] == "gate_left"
    control.close()


def test_front_panel_requires_one_dc_mapping_per_waveform(monkeypatch, tmp_path):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_left", "dc_right"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
        },
        output_count=2,
    )
    control.mapping_table.removeRow(1)

    assert control.validate_settings() is None
    assert "one DC mapping for every waveform output" in control.status.text()
    control.close()
    app.processEvents()


def test_imported_mapper_is_read_only_for_save_as(monkeypatch, tmp_path):
    app = _application()
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_left",),
        {},
        None,
    )
    monkeypatch.setattr(
        front_panel,
        "configuration_from_qcs_mapper",
        lambda path, **_kwargs: configuration,
    )
    monkeypatch.setattr(
        front_panel,
        "validate_imported_qcs_role_configuration",
        lambda configuration, path: configuration,
    )
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_left"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
        },
        output_count=1,
    )

    imported_path = tmp_path / "existing.qcs"
    imported_path.write_bytes(b"opaque native mapper")
    control.load_mapper(imported_path)
    assert control.write_mapper_button.isEnabled() is False
    assert "Save As is disabled" in control.status.text()

    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    applied = []
    control.settings_applied.connect(applied.append)
    control.apply_button.click()
    app.processEvents()
    assert (
        applied[-1]["hardware_configuration_state"]
        == front_panel.QCS_HARDWARE_STATE_IMPORTED
    )
    control.set_settings(applied[-1], output_count=1)
    assert control.write_mapper_button.isEnabled() is False
    control.mapping_table.cellWidget(0, 7).setValue(2)
    assert (
        control.settings_dict()["hardware_configuration_state"]
        == front_panel.QCS_HARDWARE_STATE_IMPORTED_DIRTY
    )
    assert control.write_mapper_button.isEnabled() is False

    control.diagram_defaults_button.click()
    assert control.write_mapper_button.isEnabled() is True
    assert control.mapper_path.text() != str(imported_path)
    control.close()
    app.processEvents()


def test_imported_acquisition_picker_reuses_virtual_channel_without_save(
    monkeypatch,
    tmp_path,
):
    _application()
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_left",),
        {},
        None,
    )
    configuration["channel_mappings"].append(
        {
            "role": "unassigned",
            "logical_index": 4,
            "virtual_name": "digitizer_spare",
            "label": 17,
            "absolute_phase": True,
            "lo_frequency_hz": None,
            "slot": 5,
            "channel": 2,
        }
    )
    configuration = front_panel.normalize_qcs_hardware_configuration(
        configuration
    )
    monkeypatch.setattr(
        front_panel,
        "configuration_from_qcs_mapper",
        lambda path, **_kwargs: configuration,
    )
    monkeypatch.setattr(
        front_panel,
        "validate_imported_qcs_role_configuration",
        lambda configuration, path: configuration,
    )
    monkeypatch.setattr(
        front_panel,
        "save_qcs_channel_mapper",
        lambda *_args, **_kwargs: pytest.fail(
            "role-only imported selection must not rewrite the mapper"
        ),
    )
    imported_path = tmp_path / "existing.qcs"
    original_bytes = b"opaque third-party mapper"
    imported_path.write_bytes(original_bytes)
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(imported_path),
            "dc_channel_names": ["dc_left"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
        },
        output_count=1,
    )
    control.load_mapper(imported_path)
    original_fingerprint = front_panel.qcs_hardware_mapper_fingerprint(
        control.working_configuration()
    )
    original_digest = control._mapper_file_sha256

    assert control.focus_mapping("acquisition", 0) is True
    assert control.select_connector(5, 2) is True
    applied = []
    assert control.apply_connector_selection(
        lambda settings: applied.append(settings) or True
    ) is True

    assert len(applied) == 1
    assert applied[0]["acquisition_channel_name"] == "digitizer_spare"
    assert (
        applied[0]["hardware_configuration_state"]
        == front_panel.QCS_HARDWARE_STATE_IMPORTED
    )
    assert Path(applied[0]["mapper_path"]) == imported_path
    assert applied[0]["hardware_mapper_sha256"] == original_digest
    assert imported_path.read_bytes() == original_bytes
    assert front_panel.qcs_hardware_mapper_fingerprint(
        applied[0]["hardware_configuration"]
    ) == original_fingerprint
    selected = next(
        mapping
        for mapping in applied[0]["hardware_configuration"][
            "channel_mappings"
        ]
        if mapping["role"] == "acquisition"
    )
    assert selected["virtual_name"] == "digitizer_spare"
    assert selected["label"] == 17
    assert (selected["slot"], selected["channel"]) == (5, 2)
    assert control._mapper_write_allowed is False
    control.close()


def test_main_window_qcs_front_panel_updates_experiment(monkeypatch, tmp_path):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    window = gui.MainWindow()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    experiment.qcs_mapper_path.setText(str(tmp_path / "mapper.qcs"))

    experiment.qcs_hardware_configuration_button.click()
    app.processEvents()
    assert window._qcs_front_panel_dialog.isVisible() is True
    control = window._qcs_front_panel
    assert control.tabs.currentWidget() is control.front_panel_tab
    control.mapping_table.cellWidget(0, 2).setText("not_applied")
    window._qcs_front_panel_dialog.close()
    app.processEvents()
    assert experiment.qcs_dc_channel_names.text() == "dc_ch_1"

    experiment.qcs_hardware_configuration_button.click()
    app.processEvents()
    control.mapping_table.cellWidget(0, 2).setText("gate_plunger")
    control.apply_button.click()
    app.processEvents()

    assert experiment.qcs_dc_channel_names.text() == "gate_plunger"
    settings = experiment.qcs_settings_dict()
    assert settings["hardware_configuration"]["channel_mappings"][0][
        "virtual_name"
    ] == "gate_plunger"
    experiment.set_running(True, "Running", show_progress=False)
    assert control.apply_button.isEnabled() is False
    assert control.tabs.isEnabled() is False
    assert "locked" in control.status.text()
    experiment.set_running(False, "Ready", show_progress=False)
    assert control.apply_button.isEnabled() is True
    assert "Hardware configuration is locked" not in control.status.text()
    assert "#167c3a" in control.status.styleSheet()
    window.close()


def test_main_window_identifies_qcs_hardware_off_gui_thread(monkeypatch):
    app = _application()
    window = gui.MainWindow()
    received = []
    fake_inventory = {
        "ip_address": "192.168.2.105",
        "inventories": [
            {
                "chassis_model": "M9046A",
                "chassis": 1,
                "host_controller": 1,
                "modules": [
                    {"slot": 4, "model": "M5300A"},
                    {"slot": 7, "model": "M5301A"},
                    {"slot": 18, "model": "M5200A"},
                ],
            }
        ],
    }
    monkeypatch.setattr(
        gui,
        "identify_qcs_hardware_configuration",
        lambda ip_address: (
            fake_inventory
            if ip_address == "192.168.2.105"
            else None
        ),
    )
    monkeypatch.setattr(
        window._qcs_front_panel,
        "apply_discovered_hardware_configuration",
        lambda inventory, **_kwargs: received.append(inventory) or True,
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )

    window._qcs_front_panel.ip_address.setText("192.168.2.105")
    window._qcs_front_panel.identify_hardware_button.click()
    assert window._qcs_front_panel.tabs.isEnabled() is False
    for _index in range(100):
        app.processEvents()
        if window._experiment_thread is None:
            break
        QtTest.QTest.qWait(10)

    assert window._experiment_thread is None
    assert received == [fake_inventory]
    assert window._qcs_front_panel.tabs.isEnabled() is True
    assert (
        window._qcs_front_panel.identify_hardware_button.text()
        == "Identify Hardware Configuration"
    )
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_main_qcs_mapping_drives_previews_rf_roles_and_settings_reload(
    monkeypatch,
):
    app = _application()
    window = gui.MainWindow()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)

    first_rf = window._rf_ports_panel._panels[0]
    first_rf.setChecked(True)
    first_rf.gen_ch.setValue(99)
    window._rf_ports_panel.add_port()
    second_rf = window._rf_ports_panel._panels[1]
    second_rf.setChecked(True)
    second_rf.gen_ch.setValue(98)
    window._rf_readout_panel.ro_ch.setValue(23)
    dormant_awg_channels = tuple(window._qick_awg_channels)

    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_gate",),
        {3: "rf_drive", 7: "rf_probe"},
        "digitizer",
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "dc_channel_names": ["dc_gate"],
            "rf_channel_names": {
                "3": "rf_drive",
                "7": "rf_probe",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
            "hardware_mapper_sha256": None,
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    emitted_specs = []
    window._rf_ports_panel.specs_changed.connect(emitted_specs.append)

    window._apply_qcs_front_panel_settings(payload)
    app.processEvents()

    assert experiment.execution_backend() == gui.EXECUTION_BACKEND_QCS
    assert len(emitted_specs) == 1
    assert [
        panel.gen_ch.value()
        for panel in window._rf_ports_panel._panels
        if panel.isChecked()
    ] == [3, 7]
    assert tuple(window._qick_awg_channels) == dormant_awg_channels
    assert window._rf_readout_panel.ro_ch.value() == 23
    assert isinstance(
        window._multi_ctrl.front_panel_preview.currentWidget(),
        front_panel.QcsFrontPanelPreview,
    )
    assert isinstance(
        window._rf_ports_panel._panels[
            0
        ].front_panel_preview.currentWidget(),
        front_panel.QcsFrontPanelPreview,
    )
    assert isinstance(
        window._rf_readout_panel.front_panel_preview.currentWidget(),
        front_panel.QcsFrontPanelPreview,
    )
    assert "dc_gate" in window._multi_ctrl.mapping_summary.text()
    assert (
        "rf_drive"
        in window._rf_ports_panel._panels[
            0
        ].front_panel_preview.qcs_preview.binding_label.text()
    )
    assert (
        "rf_probe"
        in window._rf_ports_panel._panels[
            1
        ].front_panel_preview.qcs_preview.binding_label.text()
    )
    assert (
        "digitizer"
        in window._rf_readout_panel.front_panel_preview.qcs_preview.binding_label.text()
    )
    assert "slot 2 ch1" in window._multi_ctrl.panel_table.item(0, 2).text()

    document = window._settings_to_dict()
    document["rf_outputs"][0]["gen_ch"] = 51
    document["rf_outputs"][1]["gen_ch"] = 52
    window._apply_decoded_settings(window._decode_settings(document))
    assert [
        panel.gen_ch.value()
        for panel in window._rf_ports_panel._panels
        if panel.isChecked()
    ] == [3, 7]

    window._multi_ctrl.front_panel_preview.activated.emit()
    app.processEvents()
    assert window._qcs_front_panel_dialog.isVisible() is True
    assert window._qick_front_panel_dialog.isVisible() is False
    selected_rows = (
        window._qcs_front_panel.mapping_table.selectionModel().selectedRows()
    )
    assert len(selected_rows) == 1
    selected_row = selected_rows[0].row()
    assert (
        window._qcs_front_panel.mapping_table.cellWidget(
            selected_row,
            0,
        ).currentData()
        == "dc"
    )
    window._qcs_front_panel_dialog.close()

    qcs_calls = []
    qick_calls = []
    monkeypatch.setattr(
        window,
        "_show_qcs_front_panel",
        lambda role=None, logical_index=0: qcs_calls.append(
            (role, logical_index)
        ),
    )
    monkeypatch.setattr(
        window,
        "_show_qick_front_panel",
        lambda scope, target=None: qick_calls.append((scope, target)),
    )
    window._rf_ports_panel.front_panel_requested.emit(
        window._rf_ports_panel._panels[1]
    )
    window._rf_readout_panel.front_panel_requested.emit(
        window._rf_readout_panel
    )
    assert qcs_calls == [("rf", 7), ("acquisition", 0)]
    assert qick_calls == []

    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    assert isinstance(
        window._multi_ctrl.front_panel_preview.currentWidget(),
        gui.QickFrontPanelPreview,
    )
    window._multi_ctrl.front_panel_requested.emit(window._multi_ctrl)
    assert qick_calls == [("output", window._multi_ctrl)]
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_awg_edit_opens_focused_qcs_sma_picker_and_applies_selection(
    monkeypatch,
    tmp_path,
):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    automatic_mapper = tmp_path / "automatic_awg_mapper.qcs"

    def fake_save(_configuration, path):
        output_path = Path(path)
        output_path.write_bytes(b"automatic AWG mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", fake_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )
    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    window._add_port()
    app.processEvents()

    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_left", "dc_right"),
        {},
        None,
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "dc_channel_names": ["dc_left", "dc_right"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
            "hardware_mapper_sha256": None,
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    window._apply_qcs_front_panel_settings(payload)
    window._port_select(0)
    app.processEvents()

    compact_preview = window._multi_ctrl.front_panel_preview.qcs_preview
    first_output_png = _pixmap_png(compact_preview._pixmap)
    second_control = window._multi_ctrl._ctrl_pannels[1]
    edit_button = next(
        button
        for button in second_control.findChildren(QtWidgets.QPushButton)
        if button.text() == "Edit this AWG output"
    )
    QtTest.QTest.mouseClick(edit_button, QtCore.Qt.LeftButton)
    app.processEvents()

    assert window._selected_port_idx == 1
    assert window._multi_ctrl._selected_index == 1
    assert compact_preview._selected_address() == (2, 2)
    assert _pixmap_png(compact_preview._pixmap) != first_output_png
    assert window._qcs_front_panel_dialog.isVisible() is True
    assert window._qcs_front_panel._focused_mapping == ("dc", 1)
    assert (
        window._qcs_front_panel._focused_mapping_address(
            window._qcs_front_panel._preview_configuration_from_widgets()
        )
        == (2, 2)
    )

    assert window._qcs_front_panel.select_connector(2, 3) is True
    _wait_for_qcs_mapper_commit(window)

    applied_configuration = experiment.qcs_settings_dict()[
        "hardware_configuration"
    ]
    applied_mapping = next(
        mapping
        for mapping in applied_configuration["channel_mappings"]
        if (
            mapping["role"] == "dc"
            and int(mapping["logical_index"]) == 1
        )
    )
    assert (
        int(applied_mapping["slot"]),
        int(applied_mapping["channel"]),
    ) == (2, 3)
    assert compact_preview._selected_address() == (2, 3)
    assert "slot 2 ch3" in window._multi_ctrl.panel_table.item(1, 2).text()
    assert window._qcs_front_panel_dialog.isVisible() is False
    assert automatic_mapper.is_file()
    assert (
        experiment.qcs_settings_dict()["hardware_configuration_state"]
        == front_panel.QCS_HARDWARE_STATE_SAVED
    )

    window._qcs_front_panel_dialog.close()
    app.processEvents()
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_qcs_sma_selection_stays_responsive_and_blocks_run_until_mapper_saved(
    monkeypatch,
    tmp_path,
):
    app = _application()
    original_mapper = tmp_path / "original_mapper.qcs"
    original_mapper.write_bytes(b"original mapper")
    automatic_mapper = tmp_path / "background_mapper.qcs"
    save_started = threading.Event()
    release_save = threading.Event()
    worker_thread_ids = []
    warnings = []

    def blocking_save(_configuration, path):
        worker_thread_ids.append(int(QtCore.QThread.currentThreadId()))
        save_started.set()
        if not release_save.wait(5.0):
            raise TimeoutError("test did not release background mapper save")
        output_path = Path(path)
        output_path.write_bytes(b"background mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", blocking_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, message, *_args, **_kwargs: (
            warnings.append((str(title), str(message)))
            or QtWidgets.QMessageBox.Ok
        ),
    )

    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_only",),
        {},
        None,
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "mapper_path": str(original_mapper),
            "dc_channel_names": ["dc_only"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_SAVED
            ),
            "hardware_mapper_sha256": (
                front_panel.qcs_mapper_file_sha256(original_mapper)
            ),
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    assert window._apply_qcs_front_panel_settings(payload) is True
    window._show_active_front_panel("output", window._multi_ctrl)
    app.processEvents()

    gui_thread_id = int(QtCore.QThread.currentThreadId())
    watchdog = threading.Timer(3.0, release_save.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        assert window._qcs_front_panel.select_connector(2, 2) is True
        # The save is still unreleased, so returning here proves that the SMA
        # click did not execute native mapper work synchronously.
        assert release_save.is_set() is False
        assert experiment._qcs_front_panel_draft_pending is True
        assert window._qcs_front_panel._focused_mapping_address(
            window._qcs_front_panel.working_configuration()
        ) == (2, 2)

        elapsed = QtCore.QElapsedTimer()
        elapsed.start()
        while not save_started.is_set() and elapsed.elapsed() < 2000:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            QtTest.QTest.qWait(5)
        assert save_started.is_set() is True
        assert release_save.is_set() is False
        assert worker_thread_ids and worker_thread_ids[0] != gui_thread_id

        heartbeat = []
        QtCore.QTimer.singleShot(0, lambda: heartbeat.append(True))
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        assert heartbeat == [True]
        assert release_save.is_set() is False

        assert experiment._qcs_front_panel_draft_pending is True
        with pytest.raises(ValueError, match="assignment is incomplete"):
            experiment.qcs_connection_values(1)

        # Exercise the actual Run entrypoint while avoiding unrelated waveform
        # and acquisition validation. It must stop at the pending-mapper guard.
        monkeypatch.setattr(
            window,
            "_qcs_experiment_run_arguments",
            lambda: experiment.qcs_connection_values(1),
        )
        window._run_experiment()
        assert window._experiment_thread is None
        assert warnings
        assert "assignment is incomplete" in warnings[-1][1]

        canonical_before_save = experiment.qcs_settings_dict()[
            "hardware_configuration"
        ]
        assert canonical_before_save["channel_mappings"][0]["channel"] == 1

        release_save.set()
        _wait_for_qcs_mapper_commit(window)
        saved = experiment.qcs_settings_dict()
        assert saved["hardware_configuration_state"] == (
            front_panel.QCS_HARDWARE_STATE_SAVED
        )
        assert saved["hardware_configuration"]["channel_mappings"][0][
            "channel"
        ] == 2
        assert experiment._qcs_front_panel_draft_pending is False
        assert automatic_mapper.is_file()
    finally:
        release_save.set()
        watchdog.cancel()
        _wait_for_qcs_mapper_commit(window)
        window.close()
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(
            None,
            QtCore.QEvent.DeferredDelete,
        )
        app.processEvents()


def test_rapid_qcs_sma_selection_ignores_stale_success_and_latest_wins(
    monkeypatch,
    tmp_path,
):
    app = _application()
    original_mapper = tmp_path / "rapid_original.qcs"
    original_mapper.write_bytes(b"original mapper")
    automatic_mapper = tmp_path / "rapid_background.qcs"
    first_started = threading.Event()
    release_first = threading.Event()
    latest_started = threading.Event()
    release_latest = threading.Event()
    saved_channels = []

    def controlled_save(configuration, path):
        dc_mapping = next(
            mapping
            for mapping in configuration["channel_mappings"]
            if mapping["role"] == "dc" and mapping["logical_index"] == 0
        )
        channel = int(dc_mapping["channel"])
        saved_channels.append(channel)
        if channel == 2:
            first_started.set()
            if not release_first.wait(5.0):
                raise TimeoutError("test did not release stale mapper save")
        elif channel == 3:
            latest_started.set()
            if not release_latest.wait(5.0):
                raise TimeoutError("test did not release latest mapper save")
        else:
            raise AssertionError(f"unexpected saved channel {channel}")
        output_path = Path(path)
        output_path.write_bytes(f"mapper channel {channel}".encode("ascii"))
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", controlled_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )

    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_only",),
        {},
        None,
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "mapper_path": str(original_mapper),
            "dc_channel_names": ["dc_only"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_SAVED
            ),
            "hardware_mapper_sha256": (
                front_panel.qcs_mapper_file_sha256(original_mapper)
            ),
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    assert window._apply_qcs_front_panel_settings(payload) is True
    window._show_active_front_panel("output", window._multi_ctrl)
    app.processEvents()

    watchdog = threading.Timer(
        5.0,
        lambda: (release_first.set(), release_latest.set()),
    )
    watchdog.daemon = True
    watchdog.start()
    try:
        assert window._qcs_front_panel.select_connector(2, 2) is True
        elapsed = QtCore.QElapsedTimer()
        elapsed.start()
        while not first_started.is_set() and elapsed.elapsed() < 2000:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            QtTest.QTest.qWait(5)
        assert first_started.is_set() is True
        assert release_first.is_set() is False

        # This later click updates the visible draft immediately while the CH2
        # native mapper is still being written by the serialized worker.
        assert window._qcs_front_panel.select_connector(2, 3) is True
        assert window._qcs_front_panel._focused_mapping_address(
            window._qcs_front_panel.working_configuration()
        ) == (2, 3)
        elapsed.restart()
        while (
            window._qcs_mapper_commit_queued is None
            and elapsed.elapsed() < 2000
        ):
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            QtTest.QTest.qWait(5)
        assert window._qcs_mapper_commit_queued is not None
        assert window._qcs_mapper_commit_queued["address"] == (2, 3)
        assert experiment._qcs_front_panel_draft_pending is True

        release_first.set()
        elapsed.restart()
        while not latest_started.is_set() and elapsed.elapsed() < 3000:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            QtTest.QTest.qWait(5)
        assert latest_started.is_set() is True
        assert release_latest.is_set() is False

        # CH2 completed successfully, including writing a native file, but its
        # stale revision must not reach canonical Experiment settings.
        canonical = experiment.qcs_settings_dict()["hardware_configuration"]
        assert canonical["channel_mappings"][0]["channel"] == 1
        assert experiment._qcs_front_panel_draft_pending is True
        assert window._qcs_front_panel_dialog.isVisible() is True
        assert window._qcs_front_panel._focused_mapping_address(
            window._qcs_front_panel.working_configuration()
        ) == (2, 3)

        release_latest.set()
        _wait_for_qcs_mapper_commit(window)
        final_settings = experiment.qcs_settings_dict()
        assert saved_channels == [2, 3]
        assert final_settings["hardware_configuration"]["channel_mappings"][0][
            "channel"
        ] == 3
        assert final_settings["hardware_configuration_state"] == (
            front_panel.QCS_HARDWARE_STATE_SAVED
        )
        assert experiment._qcs_front_panel_draft_pending is False
        assert window._qcs_front_panel_dialog.isVisible() is False
    finally:
        release_first.set()
        release_latest.set()
        watchdog.cancel()
        _wait_for_qcs_mapper_commit(window)
        window.close()
        window.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(
            None,
            QtCore.QEvent.DeferredDelete,
        )
        app.processEvents()


def test_qcs_mapper_completion_is_stale_after_switching_to_qick(
    monkeypatch,
    tmp_path,
):
    automatic_mapper = tmp_path / "backend_switch_mapper.qcs"
    save_started = threading.Event()
    release_save = threading.Event()

    def blocking_save(_configuration, path):
        save_started.set()
        if not release_save.wait(5.0):
            raise TimeoutError("test did not release backend-switch save")
        output_path = Path(path)
        output_path.write_bytes(b"backend-switch mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", blocking_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )
    _app, window, experiment, original_mapper = (
        _open_saved_qcs_output_window(tmp_path)
    )
    watchdog = threading.Timer(4.0, release_save.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        assert window._qcs_front_panel.select_connector(2, 2) is True
        _wait_for_event(save_started)
        assert release_save.is_set() is False
        assert experiment._qcs_front_panel_draft_pending is True

        experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
        assert experiment.execution_backend() == gui.EXECUTION_BACKEND_QICK
        release_save.set()
        _wait_for_qcs_mapper_commit(window)

        # The stale worker may finish its atomic local file, but it cannot
        # restore QCS or replace the last run-ready canonical mapper.
        assert experiment.execution_backend() == gui.EXECUTION_BACKEND_QICK
        settings = experiment.qcs_settings_dict()
        assert Path(settings["mapper_path"]) == original_mapper
        assert settings["hardware_configuration_state"] == (
            front_panel.QCS_HARDWARE_STATE_SAVED
        )
        assert settings["hardware_configuration"]["channel_mappings"][0][
            "channel"
        ] == 1
        pending = window._pending_qcs_front_panel_state()
        assert pending is not None
        assert pending[0]["channel_mappings"][0]["channel"] == 2
        assert experiment._qcs_front_panel_draft_pending is True
        with pytest.raises(ValueError, match="background mapper update"):
            experiment.qcs_connection_values(1)
    finally:
        watchdog.cancel()
        _close_qcs_race_window(window, release_save)


def test_qcs_mapper_completion_is_stale_after_source_binding_edit(
    monkeypatch,
    tmp_path,
):
    automatic_mapper = tmp_path / "binding_edit_mapper.qcs"
    save_started = threading.Event()
    release_save = threading.Event()

    def blocking_save(_configuration, path):
        save_started.set()
        if not release_save.wait(5.0):
            raise TimeoutError("test did not release binding-edit save")
        output_path = Path(path)
        output_path.write_bytes(b"binding-edit mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", blocking_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )
    _app, window, experiment, original_mapper = (
        _open_saved_qcs_output_window(tmp_path)
    )
    watchdog = threading.Timer(4.0, release_save.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        assert window._qcs_front_panel.select_connector(2, 2) is True
        _wait_for_event(save_started)
        assert release_save.is_set() is False

        # This is a source-binding-only edit: the physical configuration and
        # mapper fingerprint stay unchanged, but the in-editor role name no
        # longer belongs to the request being validated.
        window._qcs_front_panel.update_source_dc_channels(
            ("renamed_dc",),
            output_count=1,
        )
        assert window._stage_qcs_front_panel_pending_configuration(
            window._qcs_front_panel.working_configuration(),
            propagate=False,
        ) is True
        assert window._qcs_front_panel.working_source_bindings()[0] == [
            "renamed_dc"
        ]

        release_save.set()
        _wait_for_qcs_mapper_commit(window)

        settings = experiment.qcs_settings_dict()
        assert Path(settings["mapper_path"]) == original_mapper
        assert settings["hardware_configuration"]["channel_mappings"][0][
            "channel"
        ] == 1
        pending = window._pending_qcs_front_panel_state()
        assert pending is not None
        assert pending[1] == ("renamed_dc",)
        assert pending[0]["channel_mappings"][0]["channel"] == 2
        assert experiment._qcs_front_panel_draft_pending is True
        with pytest.raises(ValueError, match="background mapper update"):
            experiment.qcs_connection_values(1)
    finally:
        watchdog.cancel()
        _close_qcs_race_window(window, release_save)


def test_settings_loads_are_rejected_while_qcs_mapper_save_is_active(
    monkeypatch,
    tmp_path,
):
    automatic_mapper = tmp_path / "settings_load_guard_mapper.qcs"
    save_started = threading.Event()
    release_save = threading.Event()

    def blocking_save(_configuration, path):
        save_started.set()
        if not release_save.wait(5.0):
            raise TimeoutError("test did not release settings-load save")
        output_path = Path(path)
        output_path.write_bytes(b"settings-load mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", blocking_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )
    _app, window, experiment, original_mapper = (
        _open_saved_qcs_output_window(tmp_path)
    )
    decoded_settings = window._decode_settings(window._settings_to_dict())
    legacy_settings = {
        "initial_voltage": 123.0,
        "voltage_bounds": [-800.0, 800.0],
    }
    watchdog = threading.Timer(4.0, release_save.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        assert window._qcs_front_panel.select_connector(2, 2) is True
        _wait_for_event(save_started)
        assert release_save.is_set() is False
        assert experiment._qcs_front_panel_draft_pending is True

        with pytest.raises(RuntimeError, match="background QCS mapper update"):
            window._apply_decoded_settings(decoded_settings)
        with pytest.raises(RuntimeError, match="background QCS mapper update"):
            window._apply_legacy_settings(legacy_settings)

        # Neither rejected load may discard the selected physical draft or
        # expose the previous run-ready mapper to hardware execution.
        assert experiment._qcs_front_panel_draft_pending is True
        pending = window._pending_qcs_front_panel_state()
        assert pending is not None
        assert pending[0]["channel_mappings"][0]["channel"] == 2
        canonical = experiment.qcs_settings_dict()
        assert Path(canonical["mapper_path"]) == original_mapper
        assert canonical["hardware_configuration"]["channel_mappings"][0][
            "channel"
        ] == 1
        with pytest.raises(ValueError, match="background mapper update"):
            experiment.qcs_connection_values(1)

        release_save.set()
        _wait_for_qcs_mapper_commit(window)

        # Once the same request finishes normally, its selected mapper becomes
        # canonical and the standard QCS connection path is run-ready again.
        settings = experiment.qcs_settings_dict()
        assert Path(settings["mapper_path"]) == automatic_mapper
        assert settings["hardware_configuration"]["channel_mappings"][0][
            "channel"
        ] == 2
        assert settings["hardware_configuration_state"] == (
            front_panel.QCS_HARDWARE_STATE_SAVED
        )
        assert experiment._qcs_front_panel_draft_pending is False
        connection = experiment.qcs_connection_values(1)
        assert Path(connection.mapper_path) == automatic_mapper
        assert connection.dc_channel_names == ("dc_only",)
    finally:
        watchdog.cancel()
        _close_qcs_race_window(window, release_save)


def test_old_qcs_mapper_completion_does_not_close_reopened_input_context(
    monkeypatch,
    tmp_path,
):
    automatic_mapper = tmp_path / "reopened_context_mapper.qcs"
    save_started = threading.Event()
    release_save = threading.Event()

    def blocking_save(_configuration, path):
        save_started.set()
        if not release_save.wait(5.0):
            raise TimeoutError("test did not release reopened-context save")
        output_path = Path(path)
        output_path.write_bytes(b"reopened-context mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", blocking_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )
    app, window, experiment, _original_mapper = (
        _open_saved_qcs_output_window(
            tmp_path,
            acquisition_name="digitizer",
        )
    )
    watchdog = threading.Timer(4.0, release_save.set)
    watchdog.daemon = True
    watchdog.start()
    try:
        assert window._qcs_front_panel.select_connector(2, 2) is True
        _wait_for_event(save_started)
        original_session = window._qcs_front_panel_session_revision

        window._qcs_front_panel_dialog.close()
        app.processEvents()
        window._show_active_front_panel("input", window._rf_readout_panel)
        app.processEvents()
        assert window._qcs_front_panel_dialog.isVisible() is True
        assert window._qcs_front_panel_session_revision > original_session
        assert window._qcs_front_panel._focused_mapping == ("acquisition", 0)
        assert window._qcs_front_panel_auto_apply_selection == (
            "acquisition",
            0,
        )

        release_save.set()
        _wait_for_qcs_mapper_commit(window)

        # The output request remains valid and may commit, but it no longer
        # owns the dialog session that is now selecting an acquisition input.
        settings = experiment.qcs_settings_dict()
        dc_mapping = next(
            mapping
            for mapping in settings["hardware_configuration"][
                "channel_mappings"
            ]
            if mapping["role"] == "dc"
        )
        assert (dc_mapping["slot"], dc_mapping["channel"]) == (2, 2)
        assert experiment._qcs_front_panel_draft_pending is False
        assert window._qcs_front_panel_dialog.isVisible() is True
        assert window._qcs_front_panel._focused_mapping == ("acquisition", 0)
        assert window._qcs_front_panel_auto_apply_selection == (
            "acquisition",
            0,
        )
    finally:
        watchdog.cancel()
        _close_qcs_race_window(window, release_save)


def test_unchecked_rf_output_preview_remains_clickable():
    app = _application()
    panel = gui.RfPulsePortPanel(
        gui.PulseSequence(),
        0,
        time_unit="us",
    )
    panel.set_hardware_backend(gui.EXECUTION_BACKEND_QCS)
    panel.resize(720, 900)
    panel.show()
    app.processEvents()

    requested = QtTest.QSignalSpy(panel.front_panel_requested)
    assert panel.isChecked() is False
    assert panel.front_panel_preview.isEnabled() is True
    assert panel.front_panel_preview.currentWidget().isEnabled() is True
    QtTest.QTest.mouseClick(
        panel.front_panel_preview.currentWidget(),
        QtCore.Qt.LeftButton,
    )
    assert len(requested) == 1

    panel.setChecked(True)
    panel.setChecked(False)
    app.processEvents()
    assert panel.front_panel_preview.isEnabled() is True
    assert panel.front_panel_preview.currentWidget().isEnabled() is True
    QtTest.QTest.mouseClick(
        panel.front_panel_preview.currentWidget(),
        QtCore.Qt.LeftButton,
    )
    assert len(requested) == 2
    panel.close()


def test_rf_output_front_panel_sma_selection_auto_applies(
    monkeypatch,
    tmp_path,
):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    automatic_mapper = tmp_path / "automatic_rf_mapper.qcs"

    def fake_save(_configuration, path):
        output_path = Path(path)
        output_path.write_bytes(b"automatic RF mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", fake_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )
    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)

    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_only",),
        {},
        None,
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "dc_channel_names": ["dc_only"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
            "hardware_mapper_sha256": None,
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    assert window._apply_qcs_front_panel_settings(payload) is True
    rf_output = window._rf_ports_panel._panels[0]
    window._awg_tuning_tabs.setCurrentWidget(window._rf_ports_panel)
    app.processEvents()

    QtTest.QTest.mouseClick(
        rf_output.front_panel_preview.currentWidget(),
        QtCore.Qt.LeftButton,
    )
    app.processEvents()
    assert window._qcs_front_panel._focused_mapping == ("rf", 0)
    assert window._qcs_front_panel_auto_apply_selection == ("rf", 0)

    assert window._qcs_front_panel.select_connector(2, 2) is True
    _wait_for_qcs_mapper_commit(window)
    applied_configuration = experiment.qcs_settings_dict()[
        "hardware_configuration"
    ]
    applied_mapping = next(
        mapping
        for mapping in applied_configuration["channel_mappings"]
        if mapping["role"] == "rf" and mapping["logical_index"] == 0
    )
    assert applied_mapping["virtual_name"] == "rf_drive"
    assert (applied_mapping["slot"], applied_mapping["channel"]) == (2, 2)
    assert experiment.qcs_settings_dict()["rf_channel_names"] == {
        "0": "rf_drive"
    }
    assert (
        rf_output.front_panel_preview.qcs_preview._selected_address()
        == (2, 2)
    )
    assert automatic_mapper.is_file()
    assert window._qcs_front_panel_dialog.isVisible() is False
    assert "Mapped QCS RF output 0" in window.statusBar().currentMessage()
    window.close()


def test_rf_output_lo_editor_persists_mapper_and_updates_shared_preview(
    monkeypatch,
    tmp_path,
):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    automatic_mapper = tmp_path / "automatic_rf_lo_mapper.qcs"

    def fake_save(_configuration, path):
        output_path = Path(path)
        output_path.write_bytes(b"automatic RF LO mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", fake_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )
    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_only",),
        {0: "rf_drive"},
        None,
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "dc_channel_names": ["dc_only"],
            "rf_channel_names": {"0": "rf_drive"},
            "acquisition_channel_name": None,
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
            "hardware_mapper_sha256": None,
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    assert window._apply_qcs_front_panel_settings(payload) is True
    rf_output = window._rf_ports_panel._panels[0]
    window._control_tabs.setCurrentWidget(window._awg_tuning_page)
    window._awg_tuning_tabs.setCurrentWidget(window._rf_ports_panel)
    app.processEvents()

    rf_output.qcs_lo_frequency_ghz.setValue(6.5)
    rf_output.apply_qcs_lo_frequency.click()
    _wait_for_qcs_mapper_commit(window)

    saved = experiment.qcs_settings_dict()
    rf_mapping = next(
        mapping
        for mapping in saved["hardware_configuration"]["channel_mappings"]
        if mapping["role"] == "rf" and mapping["logical_index"] == 0
    )
    assert rf_mapping["lo_frequency_hz"] == pytest.approx(6.5e9)
    assert saved["hardware_configuration_state"] == (
        front_panel.QCS_HARDWARE_STATE_SAVED
    )
    assert saved["hardware_mapper_sha256"] == (
        front_panel.qcs_mapper_file_sha256(automatic_mapper)
    )
    assert automatic_mapper.is_file()
    assert rf_output.qcs_lo_frequency_ghz.value() == pytest.approx(6.5)
    assert "6.5 GHz" in rf_output.qcs_lo_frequency_status.text()
    window.close()


def test_stability_same_sma_draft_retries_automatic_mapper_save(
    monkeypatch,
    tmp_path,
):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    automatic_mapper = tmp_path / "automatic_same_sma_mapper.qcs"
    save_attempts = []

    def fake_save(_configuration, path):
        save_attempts.append(Path(path))
        if len(save_attempts) == 1:
            raise OSError("temporary mapper write failure")
        output_path = Path(path)
        output_path.write_bytes(b"automatic same-SMA mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", fake_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )
    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)

    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_only",),
        {},
        None,
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "dc_channel_names": ["dc_only"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
            "hardware_mapper_sha256": None,
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    assert window._apply_qcs_front_panel_settings(payload) is True

    window._show_active_front_panel("output", window._multi_ctrl)
    app.processEvents()
    assert window._qcs_front_panel._focused_mapping == ("dc", 0)
    assert (
        window._qcs_front_panel._focused_mapping_address(
            window._qcs_front_panel._preview_configuration_from_widgets()
        )
        == (2, 1)
    )

    # The SMA is unchanged in the editor, but the draft still needs to become
    # an executable native mapper without exposing the Channel mappings tab.
    assert window._qcs_front_panel.select_connector(2, 1) is True
    _wait_for_qcs_mapper_commit(window)
    assert save_attempts == [automatic_mapper]
    assert automatic_mapper.is_file() is False
    assert window._qcs_front_panel_dialog.isVisible() is True
    with pytest.raises(ValueError, match="assignment is incomplete"):
        experiment.qcs_connection_values(1)

    # Retrying the same SMA also reports changed=False. It must retry the
    # failed commit instead of treating the editor-only address as applied.
    assert window._qcs_front_panel.select_connector(2, 1) is True
    _wait_for_qcs_mapper_commit(window)

    assert save_attempts == [automatic_mapper, automatic_mapper]
    assert automatic_mapper.is_file()
    assert window._qcs_front_panel_dialog.isVisible() is False
    assert (
        experiment.qcs_settings_dict()["hardware_configuration_state"]
        == front_panel.QCS_HARDWARE_STATE_SAVED
    )
    connection = experiment.qcs_connection_values(1)
    assert connection.dc_channel_names == ("dc_only",)

    window._show_active_front_panel("output", window._multi_ctrl)
    app.processEvents()
    assert window._qcs_front_panel.select_connector(2, 1) is True
    app.processEvents()
    assert save_attempts == [automatic_mapper, automatic_mapper]
    assert window._qcs_front_panel_dialog.isVisible() is False

    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_acquisition_preview_picture_click_creates_and_applies_mapping(
    monkeypatch,
    tmp_path,
):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    automatic_mapper = tmp_path / "automatic_acquisition_mapper.qcs"

    def fake_save(_configuration, path):
        output_path = Path(path)
        output_path.write_bytes(b"automatic acquisition mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", fake_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )
    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)

    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_only",),
        {},
        None,
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "dc_channel_names": ["dc_only"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
            "hardware_mapper_sha256": None,
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    assert window._apply_qcs_front_panel_settings(payload) is True
    window._awg_tuning_tabs.setCurrentWidget(window._rf_readout_panel)
    app.processEvents()

    QtTest.QTest.mouseClick(
        window._rf_readout_panel.front_panel_preview.qcs_preview,
        QtCore.Qt.LeftButton,
    )
    app.processEvents()

    assert window._qcs_front_panel_dialog.isVisible() is True
    assert window._qcs_front_panel._focused_mapping == ("acquisition", 0)
    assert window._qcs_front_panel._focused_mapping_row() is None
    assert window._qcs_front_panel.tabs.count() == 1

    working = window._qcs_front_panel._preview_configuration_from_widgets()
    source_x = (
        chassis_renderer.DEFAULT_CHASSIS_LEFT_MARGIN
        + 4 * chassis_renderer.DEFAULT_SLOT_WIDTH
        + 95 * chassis_renderer.DEFAULT_SLOT_WIDTH / 300
    )
    source_y = (
        chassis_renderer.DEFAULT_CHASSIS_HEADER_HEIGHT
        + chassis_renderer.DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT
        + 560 * chassis_renderer.DEFAULT_PANEL_HEIGHT / 1300
    )
    connector = chassis_renderer.qcs_chassis_connector_at_point(
        working,
        source_x,
        source_y,
        role="acquisition",
    )
    assert (connector["slot"], connector["channel"]) == (5, 2)
    displayed = window._qcs_front_panel.reference_label.pixmap()
    click_position = QtCore.QPoint(
        round(
            source_x
            * displayed.width()
            / window._qcs_front_panel._image_pixmap.width()
        ),
        round(
            source_y
            * displayed.height()
            / window._qcs_front_panel._image_pixmap.height()
        ),
    )
    QtTest.QTest.mouseClick(
        window._qcs_front_panel.reference_label,
        QtCore.Qt.LeftButton,
        pos=click_position,
    )
    _wait_for_qcs_mapper_commit(window)

    settings = experiment.qcs_settings_dict()
    mapping = next(
        mapping
        for mapping in settings["hardware_configuration"][
            "channel_mappings"
        ]
        if mapping["role"] == "acquisition"
    )
    assert mapping["virtual_name"] == "digitizer"
    assert (mapping["slot"], mapping["channel"]) == (5, 2)
    assert settings["acquisition_channel_name"] == "digitizer"
    assert (
        settings["hardware_configuration_state"]
        == front_panel.QCS_HARDWARE_STATE_SAVED
    )
    assert automatic_mapper.is_file()
    assert window._qcs_front_panel_dialog.isVisible() is False
    assert (
        window._rf_readout_panel.front_panel_preview.qcs_preview._selected_address()
        == (5, 2)
    )
    assert "digitizer" in (
        window._rf_readout_panel.front_panel_preview.qcs_preview.binding_label.text()
    )
    assert experiment.qcs_connection_values(1).acquisition_channel_name == (
        "digitizer"
    )

    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_same_acquisition_sma_commits_pending_graphical_topology(
    monkeypatch,
    tmp_path,
):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    updated_mapper = tmp_path / "updated_topology.qcs"
    save_calls = []

    def fake_save(_configuration, path):
        save_calls.append(Path(path))
        output_path = Path(path)
        output_path.write_bytes(b"updated topology mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", fake_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: updated_mapper,
    )
    original_mapper = tmp_path / "original.qcs"
    original_mapper.write_bytes(b"original mapper")
    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_only",),
        {},
        "digitizer",
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "mapper_path": str(original_mapper),
            "dc_channel_names": ["dc_only"],
            "rf_channel_names": {},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_SAVED
            ),
            "hardware_mapper_sha256": (
                front_panel.qcs_mapper_file_sha256(original_mapper)
            ),
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    assert window._apply_qcs_front_panel_settings(payload) is True
    window._awg_tuning_tabs.setCurrentWidget(window._rf_readout_panel)
    app.processEvents()
    window._rf_readout_panel.front_panel_preview.activated.emit()
    app.processEvents()

    assert window._qcs_front_panel._set_module_at_slot(6, "M5201A")
    app.processEvents()
    assert experiment._qcs_front_panel_draft_pending is True
    assert save_calls == []

    # The selected acquisition address itself is unchanged. The click must
    # still commit the pending module/IP topology instead of taking the
    # canonical-address no-op shortcut.
    assert window._qcs_front_panel.select_connector(5, 1) is True
    _wait_for_qcs_mapper_commit(window)

    settings = experiment.qcs_settings_dict()
    assert save_calls == [updated_mapper]
    assert {module["slot"] for module in settings["hardware_configuration"]["modules"]} >= {
        6
    }
    assert (
        settings["hardware_configuration_state"]
        == front_panel.QCS_HARDWARE_STATE_SAVED
    )
    assert experiment._qcs_front_panel_draft_pending is False
    assert window._qcs_front_panel_dialog.isVisible() is False

    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_m5201_dialog_automatically_saves_applies_and_persists(
    monkeypatch,
    tmp_path,
):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    automatic_mapper = tmp_path / "automatic_m5201_mapper.qcs"
    saved_configurations = []

    def fake_save(configuration, path):
        saved_configurations.append(
            front_panel.normalize_qcs_hardware_configuration(configuration)
        )
        if len(saved_configurations) == 1:
            raise OSError("simulated mapper write failure")
        output_path = Path(path)
        output_path.write_bytes(b"automatic M5201 mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", fake_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )
    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_only",),
        {},
        None,
    )
    configuration["modules"].append({"slot": 6, "model": "M5201A"})
    configuration = front_panel.normalize_qcs_hardware_configuration(
        configuration
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "dc_channel_names": ["dc_only"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
            "hardware_mapper_sha256": None,
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    assert window._apply_qcs_front_panel_settings(payload) is True
    window._awg_tuning_tabs.setCurrentWidget(window._rf_readout_panel)
    app.processEvents()
    QtTest.QTest.mouseClick(
        window._rf_readout_panel.front_panel_preview.qcs_preview,
        QtCore.Qt.LeftButton,
    )
    app.processEvents()
    assert window._qcs_front_panel._focused_mapping == ("acquisition", 0)

    source_x = (
        chassis_renderer.DEFAULT_CHASSIS_LEFT_MARGIN
        + 5 * chassis_renderer.DEFAULT_SLOT_WIDTH
        + 92 * chassis_renderer.DEFAULT_SLOT_WIDTH / 300
    )
    source_y = (
        chassis_renderer.DEFAULT_CHASSIS_HEADER_HEIGHT
        + chassis_renderer.DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT
        + 540 * chassis_renderer.DEFAULT_PANEL_HEIGHT / 1300
    )
    displayed = window._qcs_front_panel.reference_label.pixmap()
    click_position = QtCore.QPoint(
        round(
            source_x
            * displayed.width()
            / window._qcs_front_panel._image_pixmap.width()
        ),
        round(
            source_y
            * displayed.height()
            / window._qcs_front_panel._image_pixmap.height()
        ),
    )
    QtTest.QTest.mouseClick(
        window._qcs_front_panel.reference_label,
        QtCore.Qt.LeftButton,
        pos=click_position,
    )
    app.processEvents()
    dialog = window._qcs_front_panel._m5201_route_dialog
    assert dialog.isVisible() is True
    assert dialog.pair_combo.currentData() == 2
    address_index = dialog._combo_index_for_address(
        dialog.digitizer_combo,
        (5, 2),
    )
    dialog.digitizer_combo.setCurrentIndex(address_index)
    dialog.lo_frequency_ghz.setValue(8.125)
    QtTest.QTest.mouseClick(dialog.save_button, QtCore.Qt.LeftButton)
    _wait_for_qcs_mapper_commit(window)

    # A failed mapper write leaves the route editor open so the user can fix
    # the cause and retry without reselecting the module, pair, SMA, or LO.
    assert len(saved_configurations) == 1
    assert dialog.isVisible() is True
    assert window._qcs_front_panel_dialog.isVisible() is True
    assert automatic_mapper.is_file() is False
    assert "could not be generated" in window._qcs_front_panel.status.text()

    QtTest.QTest.mouseClick(dialog.save_button, QtCore.Qt.LeftButton)
    _wait_for_qcs_mapper_commit(window)

    assert len(saved_configurations) == 2
    settings = experiment.qcs_settings_dict()
    assert settings["acquisition_channel_name"] == "digitizer"
    assert (
        settings["hardware_configuration_state"]
        == front_panel.QCS_HARDWARE_STATE_SAVED
    )
    assert Path(settings["mapper_path"]) == automatic_mapper
    assert automatic_mapper.is_file()
    assert settings["hardware_configuration"]["downconverter_links"] == [
        {
            "digitizer_slot": 5,
            "digitizer_channel": 2,
            "downconverter_slot": 6,
            "downconverter_channel": 2,
            "lo_frequency_hz": 8.125e9,
        }
    ]
    assert window._qcs_front_panel_dialog.isVisible() is False
    assert dialog.isVisible() is False
    compact = window._rf_readout_panel.front_panel_preview.qcs_preview
    assert compact._selected_address() == (5, 2)
    assert "via M5201A slot 6 pair 2" in compact.binding_label.text()
    assert "LO 8.125 GHz" in compact.binding_label.text()
    assert experiment.qcs_connection_values(1).acquisition_channel_name == (
        "digitizer"
    )

    # Reopening reloads the canonical saved route instead of resetting it.
    window._rf_readout_panel.front_panel_preview.activated.emit()
    app.processEvents()
    assert window._qcs_front_panel.working_configuration()[
        "downconverter_links"
    ] == settings["hardware_configuration"]["downconverter_links"]
    window._qcs_front_panel_dialog.close()
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_awg_front_panel_click_maps_m5201_to_m5200_and_shares_route(
    monkeypatch,
    tmp_path,
):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    automatic_mapper = tmp_path / "awg_m5201_route.qcs"
    save_calls = []

    def fake_save(configuration, path):
        save_calls.append(
            front_panel.normalize_qcs_hardware_configuration(configuration)
        )
        output_path = Path(path)
        output_path.write_bytes(b"shared M5201 route mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", fake_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )
    original_mapper = tmp_path / "original_m5201_route.qcs"
    original_mapper.write_bytes(b"original route mapper")
    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    configuration = front_panel.normalize_qcs_hardware_configuration(
        _m5201_configuration()
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "mapper_path": str(original_mapper),
            "dc_channel_names": ["dc_left"],
            "rf_channel_names": {},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_SAVED
            ),
            "hardware_mapper_sha256": (
                front_panel.qcs_mapper_file_sha256(original_mapper)
            ),
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    assert window._apply_qcs_front_panel_settings(payload) is True

    window._edit_awg_output_hardware(0)
    app.processEvents()
    control = window._qcs_front_panel
    assert control._focused_mapping == ("dc", 0)
    original_dc_address = control._focused_mapping_address(
        control.working_configuration()
    )

    def click_chassis(x, y):
        displayed = control.reference_label.pixmap()
        position = QtCore.QPoint(
            round(x * displayed.width() / control._image_pixmap.width()),
            round(y * displayed.height() / control._image_pixmap.height()),
        )
        QtTest.QTest.mouseClick(
            control.reference_label,
            QtCore.Qt.LeftButton,
            pos=position,
        )
        app.processEvents()

    bay_top = (
        chassis_renderer.DEFAULT_CHASSIS_HEADER_HEIGHT
        + chassis_renderer.DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT
    )
    click_chassis(
        chassis_renderer.DEFAULT_CHASSIS_LEFT_MARGIN
        + 5 * chassis_renderer.DEFAULT_SLOT_WIDTH
        + 92 * chassis_renderer.DEFAULT_SLOT_WIDTH / 300,
        bay_top
        + 540 * chassis_renderer.DEFAULT_PANEL_HEIGHT / 1300,
    )
    assert control.m5201_route_selection_active() is True
    assert control._pending_m5201_route == (6, 2)
    assert control._focused_mapping == ("dc", 0)

    click_chassis(
        chassis_renderer.DEFAULT_CHASSIS_LEFT_MARGIN
        + 4 * chassis_renderer.DEFAULT_SLOT_WIDTH
        + 95 * chassis_renderer.DEFAULT_SLOT_WIDTH / 300,
        bay_top
        + 820 * chassis_renderer.DEFAULT_PANEL_HEIGHT / 1300,
    )
    _wait_for_qcs_mapper_commit(window)

    settings = experiment.qcs_settings_dict()
    canonical = settings["hardware_configuration"]
    acquisition = next(
        mapping
        for mapping in canonical["channel_mappings"]
        if mapping["role"] == "acquisition"
    )
    assert (acquisition["slot"], acquisition["channel"]) == (5, 3)
    assert canonical["downconverter_links"] == [
        {
            "digitizer_slot": 5,
            "digitizer_channel": 3,
            "downconverter_slot": 6,
            "downconverter_channel": 2,
            "lo_frequency_hz": 7.25e9,
        }
    ]
    assert control.m5201_route_selection_active() is False
    assert control._m5201_route_dialog.isVisible() is False
    assert control._focused_mapping == ("dc", 0)
    assert control._focused_mapping_address(control.working_configuration()) == (
        original_dc_address
    )
    assert window._qcs_front_panel_dialog.isVisible() is True
    assert window._qcs_front_panel_auto_apply_selection == ("dc", 0)
    assert len(save_calls) == 1

    # Repeating the already-applied cable gesture is a true no-op: it closes
    # only the route dialog, retains the AWG picker/focus, and does not write
    # another native mapper.
    click_chassis(
        chassis_renderer.DEFAULT_CHASSIS_LEFT_MARGIN
        + 5 * chassis_renderer.DEFAULT_SLOT_WIDTH
        + 92 * chassis_renderer.DEFAULT_SLOT_WIDTH / 300,
        bay_top
        + 540 * chassis_renderer.DEFAULT_PANEL_HEIGHT / 1300,
    )
    click_chassis(
        chassis_renderer.DEFAULT_CHASSIS_LEFT_MARGIN
        + 4 * chassis_renderer.DEFAULT_SLOT_WIDTH
        + 95 * chassis_renderer.DEFAULT_SLOT_WIDTH / 300,
        bay_top
        + 820 * chassis_renderer.DEFAULT_PANEL_HEIGHT / 1300,
    )
    app.processEvents()
    assert len(save_calls) == 1
    assert control.m5201_route_selection_active() is False
    assert control._m5201_route_dialog.isVisible() is False
    assert control._focused_mapping == ("dc", 0)
    assert window._qcs_front_panel_dialog.isVisible() is True
    assert window._qcs_front_panel_auto_apply_selection == ("dc", 0)

    previews = [
        window._multi_ctrl.front_panel_preview,
        window._rf_readout_panel.front_panel_preview,
        window._stability_panel.front_panel_preview,
        window._stability_panel.x_axis.front_panel_preview,
        window._stability_panel.y_axis.front_panel_preview,
        window._sparameter_panel.path_diagram.front_panel_preview,
        window._noise_panel.front_panel_preview,
    ]
    previews.extend(
        panel.front_panel_preview
        for panel in window._rf_ports_panel._panels
    )
    previews.extend(
        window._calibration_panel.path_diagram_for(mode).front_panel_preview
        for mode in ("output", "input", "dc_voltage")
    )
    assert all(
        preview.qcs_preview._configuration == canonical
        for preview in previews
    )

    window._qcs_front_panel_dialog.close()
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_qcs_front_panel_is_shared_by_auxiliary_measurement_tabs(monkeypatch):
    app = _application()
    window = gui.MainWindow()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_gate",),
        {0: "rf_drive"},
        "digitizer",
    )
    legacy_path_values = {
        "stability": dict(window._stability_panel.front_panel_values()),
        "sparameter": dict(window._sparameter_panel.front_panel_values()),
        "calibration": {
            mode: dict(
                window._calibration_panel.path_diagram_for(
                    mode
                ).front_panel_values()
            )
            for mode in ("output", "input", "dc_voltage")
        },
        "noise": (
            window._noise_panel.readout_channel.value(),
            window._noise_panel.input_board.currentText(),
            window._noise_panel.input_nqz.value(),
        ),
    }

    window._propagate_qcs_hardware_configuration(
        configuration,
        ("dc_gate",),
        {0: "rf_drive"},
        "digitizer",
    )
    app.processEvents()

    calibration_paths = tuple(
        window._calibration_panel.path_diagram_for(mode)
        for mode in ("output", "input", "dc_voltage")
    )
    previews = (
        window._stability_panel.front_panel_preview,
        window._sparameter_panel.path_diagram.front_panel_preview,
        window._noise_panel.front_panel_preview,
        *(path.front_panel_preview for path in calibration_paths),
    )
    assert all(
        isinstance(preview.currentWidget(), front_panel.QcsFrontPanelPreview)
        for preview in previews
    )
    assert (
        "rf_drive"
        in window._stability_panel.front_panel_preview.qcs_preview.binding_label.text()
    )
    assert (
        "rf_drive"
        in window._sparameter_panel.path_diagram.front_panel_preview.qcs_preview.binding_label.text()
    )
    assert (
        "digitizer"
        in window._noise_panel.front_panel_preview.qcs_preview.binding_label.text()
    )
    assert "rf_drive" in calibration_paths[0].front_panel_preview.qcs_preview.binding_label.text()
    assert "digitizer" in calibration_paths[1].front_panel_preview.qcs_preview.binding_label.text()
    assert "dc_gate" in calibration_paths[2].front_panel_preview.qcs_preview.binding_label.text()
    assert "dc_gate" in window._stability_panel.x_axis.front_panel_status.text()
    assert dict(window._stability_panel.front_panel_values()) == legacy_path_values[
        "stability"
    ]
    assert dict(window._sparameter_panel.front_panel_values()) == legacy_path_values[
        "sparameter"
    ]
    assert {
        mode: dict(
            window._calibration_panel.path_diagram_for(
                mode
            ).front_panel_values()
        )
        for mode in ("output", "input", "dc_voltage")
    } == legacy_path_values["calibration"]
    assert (
        window._noise_panel.readout_channel.value(),
        window._noise_panel.input_board.currentText(),
        window._noise_panel.input_nqz.value(),
    ) == legacy_path_values["noise"]

    qcs_calls = []
    qick_calls = []
    monkeypatch.setattr(
        window,
        "_show_qcs_front_panel",
        lambda role=None, logical_index=0: qcs_calls.append(
            (role, logical_index)
        ),
    )
    monkeypatch.setattr(
        window,
        "_show_qick_front_panel",
        lambda scope, target=None: qick_calls.append((scope, target)),
    )
    window._stability_panel.front_panel_requested.emit()
    window._stability_panel.electrode_front_panel_requested.emit(
        window._stability_panel.x_axis
    )
    window._sparameter_panel.front_panel_requested.emit()
    for path in calibration_paths:
        window._calibration_panel.front_panel_requested.emit(path)
    window._noise_panel.front_panel_requested.emit(window._noise_panel)

    assert qcs_calls == [
        ("rf", 0),
        ("dc", 0),
        ("rf", 0),
        ("rf", 0),
        ("acquisition", 0),
        ("dc", 0),
        ("acquisition", 0),
    ]
    assert qick_calls == []

    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    assert all(
        isinstance(preview.currentWidget(), gui.QickFrontPanelPreview)
        for preview in previews
    )
    assert (
        window._stability_panel.x_axis.front_panel_button.text()
        == "Select DAC SMA on Front Panel"
    )
    window._noise_panel.front_panel_requested.emit(window._noise_panel)
    assert qick_calls == [("input", window._noise_panel)]
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_stability_front_panel_enables_rf_and_acquisition_path_focus(
    monkeypatch,
):
    app = _application()
    window = gui.MainWindow()
    window._experiment_panel.set_execution_backend(
        gui.EXECUTION_BACKEND_QCS
    )
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_gate",),
        {3: "rf_drive"},
        "digitizer",
    )
    window._propagate_qcs_hardware_configuration(
        configuration,
        ("dc_gate",),
        {3: "rf_drive"},
        "digitizer",
    )
    path = window._stability_panel.path_diagram
    path.qcs_output_mapping_selector.setCurrentIndex(
        path.qcs_output_mapping_selector.findData(3)
    )
    focused = []
    monkeypatch.setattr(
        window,
        "_show_qcs_front_panel",
        lambda role=None, logical_index=0: True,
    )
    monkeypatch.setattr(
        window._qcs_front_panel,
        "focus_rf_acquisition_path",
        lambda rf_index, acquisition_index=0: focused.append(
            (rf_index, acquisition_index)
        ),
    )

    window._show_active_front_panel("path", window._stability_panel)

    assert focused == [(3, 0)]
    assert window._qcs_front_panel_keep_open_after_selection is True
    assert window._qcs_front_panel_auto_apply_selection == frozenset(
        {("rf", 3), ("acquisition", 0)}
    )
    window.close()
    window.deleteLater()
    app.processEvents()


def test_sparameter_front_panel_enables_rf_and_acquisition_path_focus(
    monkeypatch,
):
    app = _application()
    window = gui.MainWindow()
    window._experiment_panel.set_execution_backend(
        gui.EXECUTION_BACKEND_QCS
    )
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_gate",),
        {7: "rf_drive"},
        "digitizer",
    )
    window._propagate_qcs_hardware_configuration(
        configuration,
        ("dc_gate",),
        {7: "rf_drive"},
        "digitizer",
    )
    path = window._sparameter_panel.path_diagram
    path.qcs_mapping_selector.setCurrentIndex(
        path.qcs_mapping_selector.findData(7)
    )
    assert (
        window._sparameter_panel.qcs_rf_acquisition_front_panel_selections()
        == (
            ("rf", 7),
            ("acquisition", 0),
        )
    )
    focused = []
    monkeypatch.setattr(
        window,
        "_show_qcs_front_panel",
        lambda role=None, logical_index=0: True,
    )
    monkeypatch.setattr(
        window._qcs_front_panel,
        "focus_rf_acquisition_path",
        lambda rf_index, acquisition_index=0: focused.append(
            (rf_index, acquisition_index)
        ),
    )

    window._show_active_front_panel("path", window._sparameter_panel)

    assert focused == [(7, 0)]
    assert window._qcs_front_panel_keep_open_after_selection is True
    assert window._qcs_front_panel_auto_apply_selection == frozenset(
        {("rf", 7), ("acquisition", 0)}
    )
    window.close()
    window.deleteLater()
    app.processEvents()


def test_stability_path_sma_clicks_auto_apply_both_endpoints(
    monkeypatch,
    tmp_path,
):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    automatic_mapper = tmp_path / "automatic_stability_path_mapper.qcs"
    saved_configurations = []

    def fake_save(configuration, path):
        saved_configurations.append(
            front_panel.normalize_qcs_hardware_configuration(configuration)
        )
        output_path = Path(path)
        output_path.write_bytes(b"automatic Stability RF path mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", fake_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )
    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_gate",),
        {0: "rf_drive"},
        "digitizer",
    )
    for mapping in configuration["channel_mappings"]:
        if mapping["role"] == "rf":
            mapping["lo_frequency_hz"] = 1.2e9
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "dc_channel_names": ["dc_gate"],
            "rf_channel_names": {"0": "rf_drive"},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
            "hardware_mapper_sha256": None,
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    assert window._apply_qcs_front_panel_settings(payload) is True

    window._show_active_front_panel("path", window._stability_panel)
    app.processEvents()
    assert window._qcs_front_panel_auto_apply_selection == frozenset(
        {("rf", 0), ("acquisition", 0)}
    )
    previous_panel_png = _pixmap_png(window._qcs_front_panel._image_pixmap)
    assert window._qcs_front_panel.select_connector(3, 2) is True
    # Selection feedback is ready before the expensive native mapper commit
    # is dispatched on the next event-loop turn.
    assert window._qcs_front_panel._focused_mapping == ("rf", 0)
    assert window._qcs_front_panel._focused_mapping_address(
        window._qcs_front_panel._preview_configuration_from_widgets()
    ) == (3, 2)
    expected_rf_highlight = front_panel._qcs_front_panel_pixmap(
        window._qcs_front_panel.working_configuration(),
        highlighted_addresses=((3, 2), (5, 1)),
    )
    assert _pixmap_png(window._qcs_front_panel._image_pixmap) == (
        _pixmap_png(expected_rf_highlight)
    )
    assert _pixmap_png(window._qcs_front_panel._image_pixmap) != (
        previous_panel_png
    )
    assert experiment.qcs_settings_dict()["hardware_configuration"] != (
        window._qcs_front_panel.working_configuration()
    )
    _wait_for_qcs_mapper_commit(window)
    first_configuration = experiment.qcs_settings_dict()[
        "hardware_configuration"
    ]
    rf_mapping = next(
        mapping
        for mapping in first_configuration["channel_mappings"]
        if mapping["role"] == "rf"
    )
    assert (rf_mapping["slot"], rf_mapping["channel"]) == (3, 2)
    assert len(saved_configurations) == 1
    assert window._qcs_front_panel_dialog.isVisible() is True
    assert window._qcs_front_panel_auto_apply_selection == frozenset(
        {("rf", 0), ("acquisition", 0)}
    )

    assert window._qcs_front_panel.select_connector(5, 2) is True
    assert window._qcs_front_panel._focused_mapping == ("acquisition", 0)
    assert window._qcs_front_panel._focused_mapping_address(
        window._qcs_front_panel._preview_configuration_from_widgets()
    ) == (5, 2)
    assert len(saved_configurations) == 1
    expected_acquisition_highlight = front_panel._qcs_front_panel_pixmap(
        window._qcs_front_panel.working_configuration(),
        highlighted_addresses=((3, 2), (5, 2)),
    )
    assert _pixmap_png(window._qcs_front_panel._image_pixmap) == (
        _pixmap_png(expected_acquisition_highlight)
    )
    _wait_for_qcs_mapper_commit(window)
    second_configuration = experiment.qcs_settings_dict()[
        "hardware_configuration"
    ]
    acquisition_mapping = next(
        mapping
        for mapping in second_configuration["channel_mappings"]
        if mapping["role"] == "acquisition"
    )
    assert (
        acquisition_mapping["slot"],
        acquisition_mapping["channel"],
    ) == (5, 2)
    assert len(saved_configurations) == 2
    assert automatic_mapper.is_file()
    assert window._qcs_front_panel_dialog.isVisible() is True
    assert window._qcs_front_panel_auto_apply_selection == frozenset(
        {("rf", 0), ("acquisition", 0)}
    )
    assert "Mapped QCS acquisition input" in (
        window.statusBar().currentMessage()
    )
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_edit_awg_output_reopens_current_identified_qcs_front_panel(
    monkeypatch,
):
    app = _application()
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args: QtWidgets.QMessageBox.Yes,
    )
    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)

    assert window._show_qcs_front_panel() is True
    assert window._qcs_front_panel.apply_discovered_hardware_configuration(
        {
            "ip_address": "192.168.2.105",
            "inventories": [
                {
                    "chassis_model": "M9046A",
                    "chassis": 1,
                    "host_controller": 1,
                    "modules": [
                        {"slot": 4, "model": "M5300A"},
                        {"slot": 7, "model": "M5301A"},
                        {"slot": 17, "model": "M5201A"},
                        {"slot": 18, "model": "M5200A"},
                    ],
                }
            ],
        }
    )
    identified = window._qcs_front_panel.working_configuration()
    assert window._qcs_front_panel_pending_configuration == identified
    window._qcs_front_panel_dialog.close()
    app.processEvents()

    # This was the real stale-state failure: changing the AWG output count
    # used to discard the identified draft, so Edit rebuilt slots 1/2/3/5.
    window._add_port()
    app.processEvents()
    assert len(window._pulse) == 2
    assert window._qcs_front_panel_pending_configuration == identified

    control = window._multi_ctrl._ctrl_pannels[1]
    edit_button = next(
        button
        for button in control.findChildren(QtWidgets.QPushButton)
        if button.text() == "Edit this AWG output"
    )
    QtTest.QTest.mouseClick(edit_button, QtCore.Qt.LeftButton)
    app.processEvents()

    assert window._qcs_front_panel_dialog.isVisible() is True
    assert window._qcs_front_panel._focused_mapping == ("dc", 1)
    assert window._qcs_front_panel.working_configuration() == identified
    assert {
        (int(module["slot"]), str(module["model"]))
        for module in identified["modules"]
    } == {
        (4, "M5300A"),
        (7, "M5301A"),
        (17, "M5201A"),
        (18, "M5200A"),
    }
    assert window._qcs_front_panel_pending_configuration == identified
    pending = window._pending_qcs_front_panel_state()
    assert pending is not None
    assert pending[1] == ("dc_ch_1", "dc_ch_2")

    window._qcs_front_panel_dialog.close()
    window._delete_port(1)
    app.processEvents()
    assert len(window._pulse) == 1
    assert window._qcs_front_panel_pending_configuration == identified
    first_edit_button = next(
        button
        for button in window._multi_ctrl._ctrl_pannels[0].findChildren(
            QtWidgets.QPushButton
        )
        if button.text() == "Edit this AWG output"
    )
    QtTest.QTest.mouseClick(first_edit_button, QtCore.Qt.LeftButton)
    app.processEvents()
    assert window._qcs_front_panel._focused_mapping == ("dc", 0)
    assert window._qcs_front_panel.working_configuration() == identified

    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_compatible_identification_commits_and_edit_rehydrates_canonical_state():
    app = _application()
    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)

    initial = _example_configuration("lab_baseline")
    initial["channel_mappings"] = [
        mapping
        for mapping in initial["channel_mappings"]
        if mapping["role"] == "dc" and mapping["logical_index"] == 0
    ]
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "dc_channel_names": ["gate_left"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
            "hardware_configuration": initial,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
            "hardware_mapper_sha256": None,
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    assert window._apply_qcs_front_panel_settings(payload) is True
    assert window._show_qcs_front_panel() is True

    assert window._qcs_front_panel.apply_discovered_hardware_configuration(
        {
            "ip_address": "192.168.2.105",
            "inventories": [
                {
                    "chassis_model": "M9046A",
                    "chassis": 1,
                    "host_controller": 1,
                    "modules": [
                        {"slot": 4, "model": "M5300A"},
                        {"slot": 7, "model": "M5301A"},
                        {"slot": 17, "model": "M5201A"},
                        {"slot": 18, "model": "M5200A"},
                    ],
                }
            ],
        },
        commit_callback=window._apply_qcs_front_panel_settings,
    )
    identified = window._qcs_front_panel.working_configuration()
    assert experiment.qcs_settings_dict()["hardware_configuration"] == identified
    assert window._qcs_front_panel_pending_configuration is None
    assert {module["slot"] for module in identified["modules"]} == {
        4,
        7,
        17,
        18,
    }

    # Simulate an initialized but stale widget tree.  The closed editor must
    # treat the canonical Experiment configuration as authoritative on Edit.
    window._qcs_front_panel_dialog.close()
    window._qcs_front_panel._set_configuration_widgets(initial)
    assert 17 not in {
        module["slot"]
        for module in window._qcs_front_panel.working_configuration()["modules"]
    }
    edit_button = next(
        button
        for button in window._multi_ctrl._ctrl_pannels[0].findChildren(
            QtWidgets.QPushButton
        )
        if button.text() == "Edit this AWG output"
    )
    QtTest.QTest.mouseClick(edit_button, QtCore.Qt.LeftButton)
    app.processEvents()

    assert window._qcs_front_panel_dialog.isVisible() is True
    assert window._qcs_front_panel.working_configuration() == identified
    assert window._qcs_front_panel._focused_mapping == ("dc", 0)
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_rejected_compatible_identification_is_retained_as_pending_draft(
    tmp_path,
):
    _application()
    control = front_panel.QcsFrontPanelControl()
    configuration = _example_configuration("lab_baseline")
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["gate_left", "gate_right"],
            "rf_channel_names": {
                "0": "qubit_drive",
                "1": "readout_drive",
            },
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
        },
        output_count=2,
    )
    staged = []
    control.draft_staged.connect(staged.append)

    assert control.apply_discovered_hardware_configuration(
        {
            "ip_address": "192.168.2.105",
            "inventories": [
                {
                    "chassis_model": "M9046A",
                    "chassis": 1,
                    "host_controller": 1,
                    "modules": [
                        {"slot": 4, "model": "M5300A"},
                        {"slot": 7, "model": "M5301A"},
                        {"slot": 17, "model": "M5201A"},
                        {"slot": 18, "model": "M5200A"},
                    ],
                }
            ],
        },
        commit_callback=lambda _settings: False,
    )
    assert len(staged) == 1
    assert staged[0] == control.working_configuration()
    assert control._configuration_state == front_panel.QCS_HARDWARE_STATE_DRAFT
    assert "retained as a pending front-panel draft" in control.status.text()
    control.close()


def test_identified_draft_survives_reopen_and_stability_sma_auto_applies(
    monkeypatch,
    tmp_path,
):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    automatic_mapper = tmp_path / "automatic_stability_mapper.qcs"

    def fake_save(_configuration, path):
        output_path = Path(path)
        output_path.write_bytes(b"automatic Stability mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", fake_save)
    monkeypatch.setattr(
        front_panel.QcsFrontPanelControl,
        "_automatic_mapper_output_path",
        lambda _self, _configuration: automatic_mapper,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args: QtWidgets.QMessageBox.Yes,
    )
    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    app.processEvents()
    assert len(window._pulse) == 1
    assert window._stability_panel.y_axis.output.currentData() is None

    assert window._show_qcs_front_panel() is True
    assert window._qcs_front_panel.apply_discovered_hardware_configuration(
        {
            "ip_address": "192.168.2.105",
            "inventories": [
                {
                    "chassis_model": "M9046A",
                    "chassis": 1,
                    "host_controller": 1,
                    "modules": [
                        {"slot": 4, "model": "M5300A"},
                        {"slot": 7, "model": "M5301A"},
                        {"slot": 18, "model": "M5200A"},
                    ],
                }
            ],
        }
    )
    assert window._qcs_front_panel._mappings_from_widgets() == []
    assert experiment.qcs_settings_dict()["hardware_configuration"] is None
    assert window._qcs_front_panel_pending_configuration is not None
    with pytest.raises(ValueError, match="assignment is incomplete"):
        experiment.qcs_connection_values(2)

    window._qcs_front_panel_dialog.close()
    app.processEvents()
    assert window._show_qcs_front_panel() is True
    assert window._qcs_front_panel._module_models_by_slot == {
        4: "M5300A",
        7: "M5301A",
        18: "M5200A",
    }
    assert window._qcs_front_panel._mappings_from_widgets() == []

    window._qcs_front_panel_dialog.close()
    QtTest.QTest.mouseClick(
        window._stability_panel.x_axis.front_panel_preview.currentWidget(),
        QtCore.Qt.LeftButton,
    )
    app.processEvents()
    assert len(window._pulse) == 2
    assert window._stability_panel.x_axis.current_gen_ch() == 0
    assert window._stability_panel.y_axis.current_gen_ch() == 1
    expected_slots = {4, 7, 18}
    calibration_previews = tuple(
        window._calibration_panel.path_diagram_for(mode).front_panel_preview
        for mode in ("output", "input", "dc_voltage")
    )
    shared_previews = (
        window._multi_ctrl.front_panel_preview,
        window._rf_ports_panel._panels[0].front_panel_preview,
        window._rf_readout_panel.front_panel_preview,
        window._stability_panel.front_panel_preview,
        window._stability_panel.x_axis.front_panel_preview,
        window._stability_panel.y_axis.front_panel_preview,
        window._sparameter_panel.path_diagram.front_panel_preview,
        window._noise_panel.front_panel_preview,
        *calibration_previews,
    )
    for tab_index in range(window._control_tabs.count()):
        window._control_tabs.setCurrentIndex(tab_index)
        app.processEvents()
        for preview in shared_previews:
            configuration = preview.qcs_preview._configuration
            assert configuration is not None
            assert {
                int(module["slot"])
                for module in configuration["modules"]
            } == expected_slots
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    app.processEvents()
    assert window._qcs_front_panel_pending_configuration is not None
    for preview in shared_previews:
        assert {
            int(module["slot"])
            for module in preview.qcs_preview._configuration["modules"]
        } == expected_slots
    assert window._qcs_front_panel._focused_mapping == ("dc", 0)
    assert window._qcs_front_panel.select_connector(7, 3) is True
    app.processEvents()

    assert experiment.qcs_settings_dict()["hardware_configuration"] is None
    assert "M5301A slot 7 ch3" in (
        window._stability_panel.x_axis.front_panel_status.text()
    )
    assert window._stability_panel.x_axis.current_gen_ch() == 0
    assert window._qcs_front_panel_dialog.isVisible() is True
    assert window._qcs_front_panel_auto_apply_selection == ("dc", 0)
    assert window._qcs_front_panel._focused_mapping_address(
        window._qcs_front_panel._preview_configuration_from_widgets()
    ) == (7, 3)
    assert automatic_mapper.is_file() is False
    with pytest.raises(ValueError, match="assignment is incomplete"):
        experiment.qcs_connection_values(2)

    QtTest.QTest.mouseClick(
        window._stability_panel.y_axis.front_panel_preview.currentWidget(),
        QtCore.Qt.LeftButton,
    )
    app.processEvents()
    assert window._qcs_front_panel._focused_mapping == ("dc", 1)
    assert window._qcs_front_panel.select_connector(7, 3) is False
    assert {
        (mapping["slot"], mapping["channel"])
        for mapping in window._qcs_front_panel._mappings_from_widgets()
        if mapping["role"] == "dc"
    } == {(7, 3)}
    assert window._qcs_front_panel.select_connector(7, 4) is True
    _wait_for_qcs_mapper_commit(window)

    configuration = experiment.qcs_settings_dict()["hardware_configuration"]
    dc_mappings = {
        int(mapping["logical_index"]): mapping
        for mapping in configuration["channel_mappings"]
        if mapping["role"] == "dc"
    }
    assert {
        index: (mapping["slot"], mapping["channel"])
        for index, mapping in dc_mappings.items()
    } == {0: (7, 3), 1: (7, 4)}
    assert "M5301A slot 7 ch3" in (
        window._stability_panel.x_axis.front_panel_status.text()
    )
    assert "M5301A slot 7 ch4" in (
        window._stability_panel.y_axis.front_panel_status.text()
    )
    assert window._stability_panel.y_axis.current_gen_ch() == 1
    assert window._qcs_front_panel_dialog.isVisible() is True
    assert window._qcs_front_panel_auto_apply_selection == ("dc", 1)
    assert window._qcs_front_panel._focused_mapping_address(
        window._qcs_front_panel._preview_configuration_from_widgets()
    ) == (7, 4)
    assert automatic_mapper.is_file()
    assert (
        experiment.qcs_settings_dict()["hardware_configuration_state"]
        == front_panel.QCS_HARDWARE_STATE_SAVED
    )
    assert window._qcs_front_panel_pending_configuration is None
    assert len(experiment.qcs_connection_values(2).dc_channel_names) == 2

    QtTest.QTest.mouseClick(
        window._stability_panel.y_axis.front_panel_button,
        QtCore.Qt.LeftButton,
    )
    app.processEvents()
    assert (
        window._qcs_front_panel._focused_mapping_address(
            window._qcs_front_panel._preview_configuration_from_widgets()
        )
        == (7, 4)
    )
    window._qcs_front_panel_dialog.close()
    assert window._show_qcs_front_panel() is True
    assert window._qcs_front_panel._focused_mapping is None
    assert (
        window._qcs_front_panel._focused_mapping_address(
            window._qcs_front_panel._preview_configuration_from_widgets()
        )
        is None
    )
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_external_qcs_source_change_discards_pending_identified_draft(
    monkeypatch,
    tmp_path,
):
    app = _application()
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args: QtWidgets.QMessageBox.Yes,
    )
    window = gui.MainWindow()
    window.show()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)

    canonical = front_panel.default_qcs_hardware_configuration(
        ("dc_only",),
        {},
        None,
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "dc_channel_names": ["dc_only"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
            "hardware_configuration": canonical,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
            "hardware_mapper_sha256": None,
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    assert window._apply_qcs_front_panel_settings(payload) is True
    assert window._show_qcs_front_panel() is True
    assert window._qcs_front_panel.apply_discovered_hardware_configuration(
        {
            "ip_address": "192.168.2.105",
            "inventories": [
                {
                    "chassis_model": "M9046A",
                    "chassis": 1,
                    "host_controller": 1,
                    "modules": [
                        {"slot": 4, "model": "M5300A"},
                        {"slot": 7, "model": "M5301A"},
                        {"slot": 18, "model": "M5200A"},
                    ],
                }
            ],
        }
    )
    assert window._qcs_front_panel._mappings_from_widgets() == []
    assert experiment._qcs_front_panel_draft_pending is True
    draft_preview = (
        window._stability_panel.front_panel_preview.qcs_preview._configuration
    )
    assert {module["slot"] for module in draft_preview["modules"]} == {
        4,
        7,
        18,
    }

    window._qcs_front_panel_dialog.close()
    experiment.qcs_mapper_path.setText(str(tmp_path / "external.qcs"))
    assert (
        window._qcs_front_panel_source_snapshot
        != window._current_qcs_front_panel_source_snapshot()
    )
    assert window._show_qcs_front_panel() is True

    canonical = front_panel.normalize_qcs_hardware_configuration(
        canonical,
        required_dc_count=1,
    )
    assert (
        window._qcs_front_panel._preview_configuration_from_widgets()
        == canonical
    )
    assert experiment._qcs_front_panel_draft_pending is False
    assert (
        window._stability_panel.front_panel_preview.qcs_preview._configuration
        == canonical
    )
    assert experiment.qcs_settings_dict()["hardware_configuration"] == canonical
    with pytest.raises(ValueError, match="unsaved physical changes"):
        experiment.qcs_connection_values(1)

    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_auxiliary_qcs_panels_follow_settings_and_enable_qcs_sparameter(
    monkeypatch,
):
    app = _application()
    window = gui.MainWindow()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_gate",),
        {7: "rf_drive"},
        "digitizer",
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "dc_channel_names": ["dc_gate"],
            "rf_channel_names": {"7": "rf_drive"},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
            "hardware_mapper_sha256": None,
        }
    )
    experiment.set_qcs_settings(payload, len(window._pulse))

    legacy_output_channel = (
        window._sparameter_panel.path_diagram.output_ch.value()
    )
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    app.processEvents()

    path = window._sparameter_panel.path_diagram
    assert path.output_ch.value() == legacy_output_channel
    assert path.qcs_mapping_selector.currentData() == 7
    assert path.qcs_front_panel_selection() == ("rf", 7)
    assert (
        window._sparameter_panel.qcs_rf_acquisition_front_panel_selections()
        == (
            ("rf", 7),
            ("acquisition", 0),
        )
    )
    assert "rf_drive" in (
        path.front_panel_preview.qcs_preview.binding_label.text()
    )

    window._sparameter_panel.front_panel_requested.emit()
    app.processEvents()
    assert window._qcs_front_panel_dialog.isVisible() is True
    assert window._qcs_front_panel._rf_acquisition_path_focus == (
        ("rf", 7),
        ("acquisition", 0),
    )
    assert window._qcs_front_panel_keep_open_after_selection is True
    assert window._qcs_front_panel_auto_apply_selection == frozenset(
        {("rf", 7), ("acquisition", 0)}
    )
    selected_rows = (
        window._qcs_front_panel.mapping_table.selectionModel().selectedRows()
    )
    assert len(selected_rows) == 1
    selected_row = selected_rows[0].row()
    assert (
        window._qcs_front_panel.mapping_table.cellWidget(
            selected_row,
            0,
        ).currentData()
        == "rf"
    )
    assert (
        window._qcs_front_panel.mapping_table.cellWidget(
            selected_row,
            1,
        ).value()
        == 7
    )
    window._qcs_front_panel_dialog.close()

    assert not hasattr(window._stability_panel, "backend_warning")
    assert not hasattr(window._sparameter_panel, "backend_warning")
    assert window._sparameter_panel.fir_ddr_capture_group.isHidden() is True
    assert not hasattr(window._calibration_panel, "backend_warning")
    assert not hasattr(window._noise_panel, "backend_warning")
    # This fixture has only one AWG output, so Stability remains unavailable
    # for the backend-independent two-electrode requirement.
    assert window._stability_panel.start_button.isEnabled() is False
    assert window._stability_panel.single_shot_button.isEnabled() is False
    assert window._sparameter_panel.run_button.isEnabled() is True
    assert window._sparameter_panel.qcs_amplitude.isHidden() is False
    assert window._calibration_panel.run_output_button.isEnabled() is False
    assert window._calibration_panel.run_input_button.isEnabled() is False
    assert (
        window._calibration_panel.run_dc_voltage_button.isEnabled() is False
    )
    assert window._calibration_panel.run_qcs_rf_button.isEnabled() is True
    assert window._calibration_panel.run_qcs_dc_button.isEnabled() is True
    assert window._noise_panel.acquire_button.isEnabled() is True
    assert window._noise_panel.fir_samples.isHidden() is True
    assert window._noise_panel.input_filter.isHidden() is True
    assert window._noise_panel.input_margin.isHidden() is True

    notices = []
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda _parent, title, message: notices.append((title, message)),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    window._run_sparameter_sweep()
    noise_request = window._noise_panel.acquisition_request()
    assert isinstance(noise_request, NoiseAcquisitionRequest)
    assert noise_request.backend == "qcs"
    window._run_noise_acquisition(noise_request)
    window._run_power_calibration("output")
    assert notices == []
    assert len(warnings) >= 2
    assert warnings[0][0] == "Cannot run RF sweep"
    assert "incomplete" in warnings[0][1]
    assert warnings[1][0] == "Cannot acquire QCS noise trace"
    assert "incomplete" in warnings[1][1]
    assert window._experiment_thread is None

    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_main_window_qcs_noise_request_forces_raw_trace_worker(
    monkeypatch,
    tmp_path,
):
    app = _application()
    window = gui.MainWindow()
    panel = window._noise_panel
    panel.set_hardware_backend(gui.EXECUTION_BACKEND_QCS)
    panel.qcs_duration_us.setValue(10.0001)
    experiment = window._experiment_panel
    connection = gui.QcsConnectionConfig(
        mapper_path=str(tmp_path / "noise_mapper.qcs"),
        dc_channel_names=("dc_gate",),
        acquisition_channel_name="digitizer",
        hw_demod=True,
        blocking=False,
    )
    monkeypatch.setattr(
        experiment,
        "execution_backend",
        lambda: gui.EXECUTION_BACKEND_QCS,
    )
    monkeypatch.setattr(
        experiment,
        "qcs_connection_values",
        lambda output_count: connection,
    )
    captured = []

    class FakeQcsNoiseWorker(QtCore.QObject):
        finished = QtCore.pyqtSignal(object)
        failed = QtCore.pyqtSignal(str)
        progress_changed = QtCore.pyqtSignal(int, str)

        def __init__(self, config):
            super().__init__()
            captured.append(config)

        @QtCore.pyqtSlot()
        def run(self):
            self.progress_changed.emit(50, "Reading raw M5200 trace")
            self.finished.emit(NoiseTraceCollection(
                i_traces=np.arange(32, dtype=float).reshape(1, 1, 32),
                sample_rate_hz=gui.QCS_M5200_SAMPLE_RATE_HZ,
                unit="V",
                source="Direct QCS raw trace",
            ))

    monkeypatch.setattr(gui, "QcsNoiseAcquisitionWorker", FakeQcsNoiseWorker)
    request = panel.acquisition_request()
    window._run_noise_acquisition(request)

    timer = QtCore.QElapsedTimer()
    timer.start()
    while window._experiment_thread is not None and timer.elapsed() < 5000:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)
        QtTest.QTest.qWait(5)

    assert window._experiment_thread is None
    assert len(captured) == 1
    config = captured[0]
    assert config.duration_s == pytest.approx(10.0001e-6)
    assert config.connection_config.hw_demod is False
    assert config.connection_config.blocking is True
    assert panel._collection is not None
    assert panel._collection.sample_count == 32
    assert panel._collection.source == "Direct QCS raw trace"

    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_calibration_qcs_dc_preview_can_select_a_nonzero_logical_output():
    app = _application()
    panel = gui.CalibrationPanel()
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_left", "dc_right"),
        {7: "rf_drive"},
        "digitizer",
    )
    panel.set_qcs_front_panel_configuration(configuration)
    panel.set_hardware_backend(gui.EXECUTION_BACKEND_QCS)

    dc_path = panel.path_diagram_for("dc_voltage")
    selector_index = dc_path.qcs_mapping_selector.findData(1)
    assert selector_index >= 0
    dc_path.qcs_mapping_selector.setCurrentIndex(selector_index)
    app.processEvents()

    assert dc_path.qcs_front_panel_selection() == ("dc", 1)
    assert "dc_right" in (
        dc_path.front_panel_preview.qcs_preview.binding_label.text()
    )
    panel.close()
    panel.deleteLater()
    app.processEvents()


def test_calibration_qcs_arrows_use_visible_native_endpoints():
    app = _application()
    panel = gui.CalibrationPanel()
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_left", "dc_right"),
        {7: "rf_drive"},
        "digitizer",
    )
    panel.set_qcs_front_panel_configuration(configuration)
    panel.set_hardware_backend(gui.EXECUTION_BACKEND_QCS)
    panel.resize(1100, 900)
    panel.show()
    app.processEvents()

    for tab_index, mode in enumerate(("output", "input", "dc_voltage")):
        panel.tabs.setCurrentIndex(tab_index)
        app.processEvents()
        path = panel.path_diagram_for(mode)
        output_nodes, input_nodes = path._active_arrow_nodes()
        assert output_nodes == (
            path.qcs_output_endpoint,
            path.loss1_component,
        )
        assert input_nodes == (
            path.loss2_component,
            path.amplifier_component,
            path.qcs_input_endpoint,
        )
        assert path.output_endpoint not in output_nodes
        assert path.input_endpoint not in input_nodes
        assert path.input_condition not in input_nodes
        assert (
            path._top_center(path.qcs_output_endpoint).y()
            > path.front_panel_preview.geometry().bottom()
        )
        assert (
            path._top_center(path.qcs_input_endpoint).y()
            > path.front_panel_preview.geometry().bottom()
        )
        assert (
            path.qcs_output_endpoint.geometry().right()
            <= path.rect().right()
        )
        assert (
            path.qcs_input_endpoint.geometry().right()
            <= path.rect().right()
        )

    panel.close()
    panel.deleteLater()
    app.processEvents()


def test_loaded_path_settings_reset_transient_qcs_mapping_selection():
    app = _application()
    panel = gui.SParameterSweepPanel()
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_gate",),
        {0: "rf_primary", 7: "rf_secondary"},
        "digitizer",
    )
    panel.set_qcs_front_panel_configuration(configuration)
    panel.set_hardware_backend(gui.EXECUTION_BACKEND_QCS)
    path = panel.path_diagram

    path.qcs_mapping_selector.setCurrentIndex(
        path.qcs_mapping_selector.findData(7)
    )
    assert path.qcs_front_panel_selection() == ("rf", 7)
    path.apply_external_settings({"output_ch": 0})

    assert path.qcs_front_panel_selection() == ("rf", 0)
    assert path.qcs_mapping_selector.currentData() == 0
    panel.close()
    panel.deleteLater()
    app.processEvents()


def test_sparameter_qcs_hides_qick_only_controls_and_restores_them():
    app = _application()
    panel = gui.SParameterSweepPanel()
    panel.gain.setValue(12_345)
    panel.output_power_dbm.setValue(-12.5)
    panel.margin_input_samples.setValue(2_048)
    panel.address.setValue(37)
    panel.stride_bytes.setValue(256)
    panel.force_overwrite.setChecked(True)
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_gate",),
        {7: "rf_drive"},
        "digitizer",
    )
    panel.set_qcs_front_panel_configuration(configuration)
    panel.set_hardware_backend(gui.EXECUTION_BACKEND_QCS)
    panel.resize(1100, 1000)
    panel.show()
    app.processEvents()

    path = panel.path_diagram
    output_nodes, input_nodes = path._active_arrow_nodes()
    assert output_nodes[0] is path.qcs_output_endpoint
    assert input_nodes[-1] is path.qcs_input_endpoint
    assert path.output_endpoint not in output_nodes
    assert path.input_endpoint not in input_nodes
    assert path.title() == "QCS RF Path and DUT De-embedding"
    assert path.qcs_output_endpoint.isVisible() is True
    assert path.qcs_input_endpoint.isVisible() is True
    assert path.output_endpoint.isVisible() is False
    assert path.input_endpoint.isVisible() is False
    assert path.output_att1_component.isVisible() is False
    assert path.output_att2_component.isVisible() is False
    assert path.input_condition.isVisible() is False
    assert path.qcs_mapping_widget.isVisible() is False
    assert panel.fir_ddr_capture_group.isVisible() is False
    assert panel.power_calibration_enabled.isVisible() is True
    assert panel.power_sweep_enabled.isVisible() is False
    assert panel.gain.isVisible() is False
    assert panel.gain_label.isVisible() is False
    assert panel.output_power_dbm.isVisible() is True
    assert panel.output_power_label.isVisible() is True
    assert panel.output_power_label.text() == "Target M5300 connector power:"
    assert panel.qcs_amplitude.isVisible() is True
    assert panel.override_fpga_trigger_delay.isEnabled() is False
    assert panel.run_button.isEnabled() is True
    assert panel.scan_time_label.text() == "Total I/Q averaging time:"
    assert all(
        term not in panel.path_hint.text()
        for term in ("HWH", "Nyquist", "board selection")
    )
    qcs_settings = panel.settings_dict()
    assert qcs_settings["qcs_amplitude"] == pytest.approx(0.005)
    assert qcs_settings["gain"] == 12_345
    assert qcs_settings["output_power_dbm"] == pytest.approx(-12.5)
    assert qcs_settings["margin_input_samples"] == 2_048
    assert qcs_settings["address"] == 37
    assert qcs_settings["stride_bytes"] == 256
    assert qcs_settings["force_overwrite"] is True

    panel.set_hardware_backend(gui.EXECUTION_BACKEND_QICK)
    app.processEvents()

    assert path.output_endpoint.isVisible() is True
    assert path.input_endpoint.isVisible() is True
    assert path.qcs_output_endpoint.isVisible() is False
    assert path.qcs_input_endpoint.isVisible() is False
    assert path.output_att1_component.isVisible() is True
    assert path.output_att2_component.isVisible() is True
    assert path.input_condition.isVisible() is True
    assert panel.fir_ddr_capture_group.isVisible() is True
    assert panel.power_calibration_enabled.isVisible() is True
    assert panel.power_sweep_enabled.isVisible() is True
    assert panel.gain.isVisible() is True
    assert panel.gain_label.isVisible() is True
    assert panel.output_power_dbm.isVisible() is True
    assert panel.output_power_label.isVisible() is True
    assert panel.qcs_amplitude.isVisible() is False
    assert panel.override_fpga_trigger_delay.isEnabled() is True
    assert panel.scan_time_label.text() == "Scan time per point:"
    assert panel.gain.value() == 12_345
    assert panel.output_power_dbm.value() == pytest.approx(-12.5)
    assert panel.margin_input_samples.value() == 2_048
    assert panel.address.value() == 37
    assert panel.stride_bytes.value() == 256
    assert panel.force_overwrite.isChecked() is True
    panel.close()
    panel.deleteLater()
    app.processEvents()


def test_qcs_front_panel_preflight_rejects_auxiliary_hardware_activity():
    app = _application()
    window = gui.MainWindow()
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )

    class _RunningThread:
        @staticmethod
        def isRunning():
            return True

    window._experiment_thread = _RunningThread()
    with pytest.raises(ValueError, match="hardware task is running"):
        window._validate_qcs_front_panel_source_snapshot()
    window._experiment_thread = None
    window.close()
    window.deleteLater()
    app.processEvents()


def test_stability_uses_scoped_mapper_when_shared_experiment_is_draft(
    monkeypatch,
    tmp_path,
):
    """A complete Stability path must not validate AWG Tuning mapper state."""

    app = _application()
    window = gui.MainWindow()
    window._add_port()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QCS)
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_x", "dc_y"),
        {3: "rf_probe"},
        "digitizer",
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "dc_channel_names": ["dc_x", "dc_y"],
            "rf_channel_names": {"3": "rf_probe"},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
            "hardware_mapper_sha256": None,
        }
    )
    experiment.set_qcs_settings(payload, 2)
    window._propagate_qcs_hardware_configuration(
        configuration,
        ("dc_x", "dc_y"),
        {3: "rf_probe"},
        "digitizer",
    )
    scoped_path = tmp_path / "stability_scoped.qcs"
    monkeypatch.setattr(
        gui,
        "qcs_workflow_mapper_output_path",
        lambda _configuration, workflow: (
            scoped_path if workflow == "stability" else None
        ),
    )

    with pytest.raises(ValueError, match="unsaved physical changes"):
        experiment.qcs_connection_values(2)

    connection, scoped = window._qcs_stability_connection_values()

    assert connection.mapper_path == str(scoped_path)
    assert connection.mapper_sha256 is None
    assert connection.dc_channel_names == ("dc_x", "dc_y")
    assert connection.rf_channel_names == {3: "rf_probe"}
    assert connection.acquisition_channel_name == "digitizer"
    assert {
        mapping["role"] for mapping in scoped["channel_mappings"]
    } == {"dc", "rf", "acquisition"}
    assert experiment.qcs_settings_dict()[
        "hardware_configuration_state"
    ] == front_panel.QCS_HARDWARE_STATE_DRAFT
    window.close()
    window.deleteLater()
    app.processEvents()


def test_invalid_qcs_settings_clear_all_auxiliary_previews(monkeypatch):
    app = _application()
    window = gui.MainWindow()
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_gate",),
        {7: "rf_drive"},
        "digitizer",
    )
    window._propagate_qcs_hardware_configuration(
        configuration,
        ("dc_gate",),
        {7: "rf_drive"},
        "digitizer",
    )
    previews = (
        window._stability_panel.front_panel_preview.qcs_preview,
        window._sparameter_panel.path_diagram.front_panel_preview.qcs_preview,
        window._noise_panel.front_panel_preview.qcs_preview,
        *(
            window._calibration_panel.path_diagram_for(
                mode
            ).front_panel_preview.qcs_preview
            for mode in ("output", "input", "dc_voltage")
        ),
    )
    assert all(preview._configuration is not None for preview in previews)

    def _invalid_settings():
        raise ValueError("invalid mapper")

    monkeypatch.setattr(
        window._experiment_panel,
        "qcs_settings_dict",
        _invalid_settings,
    )
    window._on_execution_backend_changed(gui.EXECUTION_BACKEND_QCS)
    assert all(preview._configuration is None for preview in previews)

    window.close()
    window.deleteLater()
    app.processEvents()


def test_qcs_apply_rejects_too_few_rf_roles_without_partial_mutation(
    monkeypatch,
):
    app = _application()
    window = gui.MainWindow()
    experiment = window._experiment_panel
    experiment.set_execution_backend(gui.EXECUTION_BACKEND_QICK)
    first_rf = window._rf_ports_panel._panels[0]
    first_rf.setChecked(True)
    first_rf.gen_ch.setValue(11)
    window._rf_ports_panel.add_port()
    second_rf = window._rf_ports_panel._panels[1]
    second_rf.setChecked(True)
    second_rf.gen_ch.setValue(12)

    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_gate",),
        {3: "rf_drive"},
        "digitizer",
    )
    payload = experiment.qcs_settings_dict()
    payload.update(
        {
            "dc_channel_names": ["dc_gate"],
            "rf_channel_names": {"3": "rf_drive"},
            "acquisition_channel_name": "digitizer",
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
            "hardware_mapper_sha256": None,
        }
    )
    window._qcs_front_panel_source_snapshot = (
        window._current_qcs_front_panel_source_snapshot()
    )
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append(
            (title, message)
        ),
    )
    emitted_specs = []
    window._rf_ports_panel.specs_changed.connect(emitted_specs.append)

    window._apply_qcs_front_panel_settings(payload)

    assert warnings
    assert "only 1 RF role" in warnings[0][1]
    assert experiment.execution_backend() == gui.EXECUTION_BACKEND_QICK
    assert [first_rf.gen_ch.value(), second_rf.gen_ch.value()] == [11, 12]
    assert experiment.qcs_settings_dict()["hardware_configuration"] is None
    assert emitted_specs == []
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(
        None,
        QtCore.QEvent.DeferredDelete,
    )
    app.processEvents()


def test_main_window_rejects_stale_modeless_front_panel(
    monkeypatch,
    tmp_path,
):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    window = gui.MainWindow()
    experiment = window._experiment_panel
    experiment.qcs_mapper_path.setText(str(tmp_path / "before.qcs"))
    window._show_qcs_front_panel()
    app.processEvents()

    experiment.qcs_mapper_path.setText(str(tmp_path / "after.qcs"))
    experiment.qcs_dc_channel_names.setText("main_edit")
    window._qcs_front_panel.apply_button.click()
    app.processEvents()

    assert (
        "changed while the front panel was open"
        in window._qcs_front_panel.status.text()
    )
    assert experiment.qcs_mapper_path.text() == str(tmp_path / "after.qcs")
    assert experiment.qcs_dc_channel_names.text() == "main_edit"

    selected_paths = []
    mapper_path = tmp_path / "must_not_be_written.qcs"
    monkeypatch.setattr(
        window._qcs_front_panel,
        "_choose_mapper_output",
        lambda: selected_paths.append(mapper_path) or str(mapper_path),
    )
    window._qcs_front_panel.write_mapper_button.click()
    app.processEvents()
    assert selected_paths == []
    assert not mapper_path.exists()
    window.close()
    app.processEvents()


def test_saved_mapper_replacement_blocks_front_panel_apply(
    monkeypatch,
    tmp_path,
):
    app = _application()
    monkeypatch.setattr(
        front_panel,
        "build_qcs_channel_mapper",
        lambda configuration: object(),
    )
    mapper_path = tmp_path / "saved_mapper.qcs"
    mapper_path.write_bytes(b"original mapper")
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_ch_1",),
        {},
        None,
    )
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(mapper_path),
            "dc_channel_names": ["dc_ch_1"],
            "rf_channel_names": {},
            "acquisition_channel_name": None,
            "hardware_configuration": configuration,
            "hardware_configuration_state": "saved",
            "hardware_mapper_sha256": (
                front_panel.qcs_mapper_file_sha256(mapper_path)
            ),
        },
        output_count=1,
    )
    mapper_path.write_bytes(b"replacement mapper")
    applied = []
    control.settings_applied.connect(applied.append)

    control.apply_button.click()
    app.processEvents()

    assert applied == []
    assert "file changed" in control.status.text()

    repaired_path = tmp_path / "repaired_mapper.qcs"
    selected_paths = []
    monkeypatch.setattr(
        control,
        "_choose_mapper_output",
        lambda: selected_paths.append(repaired_path) or str(repaired_path),
    )

    def fake_save(configuration, path):
        output_path = Path(path)
        output_path.write_bytes(b"repaired mapper")
        return output_path.resolve()

    monkeypatch.setattr(front_panel, "save_qcs_channel_mapper", fake_save)
    control.write_mapper_button.click()
    app.processEvents()
    assert selected_paths == [repaired_path]
    assert repaired_path.read_bytes() == b"repaired mapper"
    assert applied[-1]["hardware_configuration_state"] == "saved"
    control.close()


def test_output_delete_closes_qcs_panel_with_empty_dormant_names():
    app = _application()
    window = gui.MainWindow()
    window._add_port()
    experiment = window._experiment_panel
    experiment.qcs_dc_channel_names.setText("")
    window._show_qcs_front_panel()
    app.processEvents()
    assert window._qcs_front_panel_dialog.isVisible()

    window._delete_port(0)
    app.processEvents()

    assert len(window._pulse) == 1
    assert not window._qcs_front_panel_dialog.isVisible()
    assert window._qcs_front_panel_source_snapshot is None
    window.close()
    app.processEvents()


def test_qcs_dc_mapping_tracks_added_and_removed_outputs():
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_left",),
        {},
        None,
    )
    expanded = front_panel.resize_qcs_dc_mappings(
        configuration,
        ("dc_left", "dc_right"),
    )
    dc_names, _rf_names, _acquisition = front_panel.qcs_role_bindings(
        expanded,
        required_dc_count=2,
    )
    assert dc_names == ["dc_left", "dc_right"]
    assert [
        (mapping["slot"], mapping["channel"])
        for mapping in expanded["channel_mappings"]
        if mapping["role"] == "dc"
    ] == [(2, 1), (2, 2)]

    contracted = front_panel.resize_qcs_dc_mappings(
        expanded,
        ("dc_right",),
        removed_index=0,
    )
    dc_names, _rf_names, _acquisition = front_panel.qcs_role_bindings(
        contracted,
        required_dc_count=1,
    )
    assert dc_names == ["dc_right"]
    assert contracted["channel_mappings"][0]["logical_index"] == 0
    assert contracted["channel_mappings"][0]["channel"] == 2

    incomplete_expanded = front_panel.resize_incomplete_qcs_dc_mappings(
        configuration,
        ("dc_left", "dc_right"),
    )
    assert [
        (mapping["logical_index"], mapping["slot"], mapping["channel"])
        for mapping in incomplete_expanded["channel_mappings"]
        if mapping["role"] == "dc"
    ] == [(0, 2, 1)]

    incomplete_contracted = front_panel.resize_incomplete_qcs_dc_mappings(
        expanded,
        ("dc_right",),
        removed_index=0,
    )
    assert [
        (mapping["logical_index"], mapping["slot"], mapping["channel"])
        for mapping in incomplete_contracted["channel_mappings"]
        if mapping["role"] == "dc"
    ] == [(0, 2, 2)]
    assert any(
        mapping["role"] == "unassigned"
        and mapping["slot"] == 2
        and mapping["channel"] == 1
        for mapping in incomplete_contracted["channel_mappings"]
    )


def test_qcs_dc_resize_matches_existing_names_before_allocating_new_output():
    configuration = front_panel.default_qcs_hardware_configuration(
        ("left", "right"),
        {},
        None,
    )
    expanded = front_panel.resize_qcs_dc_mappings(
        configuration,
        ("third", "left", "right"),
    )
    dc_mappings = [
        mapping
        for mapping in expanded["channel_mappings"]
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


def test_qcs_role_name_sync_preserves_surviving_dc_and_rf_bindings():
    configuration = front_panel.default_qcs_hardware_configuration(
        ("left", "right"),
        {0: "drive", 1: "probe"},
        None,
    )
    synchronized = front_panel.synchronize_qcs_hardware_role_names(
        configuration,
        ("right", "new_dc"),
        {0: "probe", 1: "new_rf"},
        None,
    )
    mappings = synchronized["channel_mappings"]
    assert [
        (
            mapping["logical_index"],
            mapping["virtual_name"],
            mapping["channel"],
        )
        for mapping in mappings
        if mapping["role"] == "dc"
    ] == [(0, "right", 2), (1, "new_dc", 1)]
    assert [
        (
            mapping["logical_index"],
            mapping["virtual_name"],
            mapping["channel"],
        )
        for mapping in mappings
        if mapping["role"] == "rf"
    ] == [(0, "probe", 2), (1, "new_rf", 1)]


def test_qcs_dc_resize_reuses_unmatched_binding_before_new_connector():
    configuration = front_panel.default_qcs_hardware_configuration(
        ("left", "right"),
        {},
        None,
    )
    expanded = front_panel.resize_qcs_dc_mappings(
        configuration,
        ("right", "new", "third"),
    )
    assert [
        (
            mapping["logical_index"],
            mapping["virtual_name"],
            mapping["channel"],
        )
        for mapping in expanded["channel_mappings"]
        if mapping["role"] == "dc"
    ] == [(0, "right", 2), (1, "new", 1), (2, "third", 3)]


def test_qcs_dc_resize_promotes_and_demotes_spare_native_m5301_mapping():
    configuration = front_panel.default_qcs_hardware_configuration(
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
    configuration = front_panel.normalize_qcs_hardware_configuration(
        configuration
    )
    native_fingerprint = front_panel.qcs_hardware_mapper_fingerprint(
        configuration
    )

    expanded = front_panel.resize_qcs_dc_mappings(
        configuration,
        ("left", "spare"),
    )
    assert front_panel.qcs_role_bindings(
        expanded,
        required_dc_count=2,
    )[0] == ["left", "spare"]
    assert (
        front_panel.qcs_hardware_mapper_fingerprint(expanded)
        == native_fingerprint
    )

    contracted = front_panel.resize_qcs_dc_mappings(
        expanded,
        ("left",),
        removed_index=1,
    )
    assert any(
        mapping["role"] == "unassigned"
        and mapping["virtual_name"] == "spare"
        and mapping["channel"] == 2
        for mapping in contracted["channel_mappings"]
    )
    assert (
        front_panel.qcs_hardware_mapper_fingerprint(contracted)
        == native_fingerprint
    )


def test_m5201_downconverter_links_normalize_to_explicit_channel_pairs():
    configuration = _m5201_configuration(
        link_count=2,
        lo_frequency_hz="7250000000",
    )

    normalized = front_panel.normalize_qcs_hardware_configuration(
        configuration
    )

    assert normalized["modules"][-1] == {"slot": 6, "model": "M5201A"}
    assert normalized["downconverter_links"] == [
        {
            "digitizer_slot": 5,
            "digitizer_channel": 1,
            "downconverter_slot": 6,
            "downconverter_channel": 1,
            "lo_frequency_hz": 7.25e9,
        },
        {
            "digitizer_slot": 5,
            "digitizer_channel": 2,
            "downconverter_slot": 6,
            "downconverter_channel": 2,
            "lo_frequency_hz": 7.25e9,
        },
    ]
    assert front_panel.QCS_MODULE_MODELS["M5201A"]["instrument"] == (
        "M5201Downconverter"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "digitizer_channel",
            5,
            "M5200A exposes digitizer channels 1-4",
        ),
        (
            "downconverter_channel",
            5,
            "M5201A exposes downconverter channels 1-4",
        ),
        (
            "lo_frequency_hz",
            999_999_999.0,
            r"M5201 LO frequency must be in \[1, 18\] GHz",
        ),
        (
            "lo_frequency_hz",
            18_000_000_001.0,
            r"M5201 LO frequency must be in \[1, 18\] GHz",
        ),
    ],
)
def test_m5201_downconverter_link_rejects_invalid_channels_and_lo(
    field,
    value,
    message,
):
    configuration = _m5201_configuration()
    configuration["downconverter_links"][0][field] = value

    with pytest.raises(ValueError, match=message):
        front_panel.normalize_qcs_hardware_configuration(configuration)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda links: links[1].update(digitizer_channel=1),
            "has more than one downconverter",
        ),
        (
            lambda links: links[1].update(downconverter_channel=1),
            "is linked more than once",
        ),
        (
            lambda links: links[1].update(lo_frequency_hz=7.5e9),
            "same shared LO frequency",
        ),
    ],
)
def test_m5201_downconverter_link_rejects_conflicts(mutation, message):
    configuration = _m5201_configuration(link_count=2)
    mutation(configuration["downconverter_links"])

    with pytest.raises(ValueError, match=message):
        front_panel.normalize_qcs_hardware_configuration(configuration)


def test_m5201_cannot_be_used_as_a_direct_virtual_channel_mapping():
    configuration = _m5201_configuration()
    acquisition = next(
        mapping
        for mapping in configuration["channel_mappings"]
        if mapping["role"] == "acquisition"
    )
    acquisition["slot"] = 6

    with pytest.raises(
        ValueError,
        match="requires M5200Digitizer, not M5201Downconverter",
    ):
        front_panel.normalize_qcs_hardware_configuration(configuration)


def test_m5201_mapper_requires_explicit_lo_frequency():
    pytest.importorskip("keysight.qcs")
    configuration = _m5201_configuration(lo_frequency_hz=None)

    with pytest.raises(ValueError, match="requires an LO frequency"):
        front_panel.build_qcs_channel_mapper(configuration)


def test_builder_rejects_duplicate_connectors_and_invalid_qcs_names():
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_left", "dc_right"),
        {},
        None,
    )
    configuration["channel_mappings"][1]["channel"] = 1
    with pytest.raises(ValueError, match="cannot share slot 2 channel 1"):
        front_panel.normalize_qcs_hardware_configuration(configuration)

    configuration["channel_mappings"][1]["channel"] = 2
    configuration["channel_mappings"][1]["virtual_name"] = "right gate"
    with pytest.raises(ValueError, match="letters, numbers, and underscores"):
        front_panel.normalize_qcs_hardware_configuration(configuration)


def test_front_panel_preserves_absolute_phase_and_m5300_lo(tmp_path):
    _application()
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_left",),
        {0: "rf_drive"},
        None,
    )
    rf_mapping = next(
        mapping
        for mapping in configuration["channel_mappings"]
        if mapping["role"] == "rf"
    )
    rf_mapping["absolute_phase"] = False
    rf_mapping["lo_frequency_hz"] = 6.125e9
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(tmp_path / "mapper.qcs"),
            "dc_channel_names": ["dc_left"],
            "rf_channel_names": {"0": "rf_drive"},
            "acquisition_channel_name": None,
            "hardware_configuration": configuration,
            "hardware_configuration_state": (
                front_panel.QCS_HARDWARE_STATE_DRAFT
            ),
        },
        output_count=1,
    )

    settings = control.settings_dict()
    restored_rf = next(
        mapping
        for mapping in settings["hardware_configuration"]["channel_mappings"]
        if mapping["role"] == "rf"
    )
    assert restored_rf["absolute_phase"] is False
    assert restored_rf["lo_frequency_hz"] == pytest.approx(6.125e9)

    control.close()


def test_m5300_mapper_requires_explicit_lo_frequency():
    pytest.importorskip("keysight.qcs")
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_left",),
        {0: "rf_drive"},
        None,
    )
    with pytest.raises(ValueError, match="requires an M5300 LO frequency"):
        front_panel.build_qcs_channel_mapper(configuration)


def test_native_qcs_mapper_save_reload_and_builder_import(tmp_path):
    pytest.importorskip("keysight.qcs")
    configuration = front_panel.default_qcs_hardware_configuration(
        ("dc_left", "dc_right"),
        {0: "rf_drive"},
        "digitizer",
    )
    configuration["ip_address"] = "127.0.0.1"
    configuration["channel_mappings"][0]["label"] = 7
    rf_mapping = next(
        mapping
        for mapping in configuration["channel_mappings"]
        if mapping["role"] == "rf"
    )
    rf_mapping["absolute_phase"] = False
    rf_mapping["lo_frequency_hz"] = 6.125e9

    saved_path = front_panel.save_qcs_channel_mapper(
        configuration,
        tmp_path / "lab_mapper",
    )
    assert saved_path.suffix == ".qcs"
    assert saved_path.is_file()
    assert len(front_panel.qcs_mapper_file_sha256(saved_path)) == 64

    restored = front_panel.configuration_from_qcs_mapper(
        saved_path,
        dc_channel_names=("dc_left", "dc_right"),
        rf_channel_names={0: "rf_drive"},
        acquisition_channel_name="digitizer",
    )
    assert restored["ip_address"] == "127.0.0.1"
    assert restored["channel_mappings"][0]["label"] == 7
    dc_names, rf_names, acquisition_name = front_panel.qcs_role_bindings(
        restored,
        required_dc_count=2,
    )
    assert dc_names == ["dc_left", "dc_right"]
    assert rf_names == {"0": "rf_drive"}
    assert acquisition_name == "digitizer"
    restored_rf = next(
        mapping
        for mapping in restored["channel_mappings"]
        if mapping["role"] == "rf"
    )
    assert restored_rf["absolute_phase"] is False
    assert restored_rf["lo_frequency_hz"] == pytest.approx(6.125e9)

    unbound = front_panel.configuration_from_qcs_mapper(saved_path)
    assert {
        mapping["role"] for mapping in unbound["channel_mappings"]
    } == {"unassigned"}


def test_native_m5201_save_load_import_round_trip_preserves_links_and_shared_lo(
    tmp_path,
):
    qcs = pytest.importorskip("keysight.qcs")
    configuration = _m5201_configuration(link_count=2)
    digitizer_addresses = [
        qcs.Address(1, 5, channel) for channel in (1, 2)
    ]
    built_mapper = front_panel.build_qcs_channel_mapper(configuration)
    built_los = [
        built_mapper.get_downconverter(address).settings.lo_frequency
        for address in digitizer_addresses
    ]
    assert built_los[0].reference_equals(built_los[1])

    saved_path = front_panel.save_qcs_channel_mapper(
        configuration,
        tmp_path / "m5201_mapper",
    )

    mapper = qcs.load(saved_path)
    expected_downconverter_addresses = [
        (1, 1, 6, channel) for channel in (1, 2)
    ]
    downconverters = [
        mapper.get_downconverter(address)
        for address in digitizer_addresses
    ]
    assert [
        str(downconverter.instrument) for downconverter in downconverters
    ] == ["M5201Downconverter", "M5201Downconverter"]
    assert [
        (
            int(downconverter.address.host_controller),
            int(downconverter.address.chassis),
            int(downconverter.address.slot),
            int(downconverter.address.channel),
        )
        for downconverter in downconverters
    ] == expected_downconverter_addresses
    lo_frequencies = [
        downconverter.settings.lo_frequency
        for downconverter in downconverters
    ]
    assert [lo.value for lo in lo_frequencies] == pytest.approx(
        [7.25e9, 7.25e9]
    )
    assert {
        str(channels.name) for channels in mapper.channels
    } == {"dc_left", "digitizer", "digitizer_spare"}

    restored = front_panel.configuration_from_qcs_mapper(
        saved_path,
        dc_channel_names=("dc_left",),
        acquisition_channel_name="digitizer",
    )
    assert {module["model"] for module in restored["modules"]} >= {
        "M5200A",
        "M5201A",
    }
    assert restored["downconverter_links"] == [
        {
            "digitizer_slot": 5,
            "digitizer_channel": 1,
            "downconverter_slot": 6,
            "downconverter_channel": 1,
            "lo_frequency_hz": pytest.approx(7.25e9),
        },
        {
            "digitizer_slot": 5,
            "digitizer_channel": 2,
            "downconverter_slot": 6,
            "downconverter_channel": 2,
            "lo_frequency_hz": pytest.approx(7.25e9),
        },
    ]
    assert front_panel.qcs_role_bindings(
        restored,
        required_dc_count=1,
    )[2] == "digitizer"
    assert any(
        mapping["virtual_name"] == "digitizer_spare"
        and mapping["role"] == "unassigned"
        for mapping in restored["channel_mappings"]
    )
    front_panel.validate_imported_qcs_role_configuration(
        restored,
        saved_path,
    )


def test_import_with_downconverter_and_unset_unused_m5300_is_read_only(tmp_path):
    qcs = pytest.importorskip("keysight.qcs")
    mapper = qcs.ChannelMapper()
    dc_address = qcs.Address(1, 2, 1)
    rf_address = qcs.Address(1, 3, 1)
    digitizer_address = qcs.Address(1, 5, 1)
    downconverter_address = qcs.Address(1, 6, 1)
    mapper.add_channel_mapping(
        qcs.Channels(0, "dc_left"),
        dc_address,
        qcs.InstrumentEnum.M5301AWG,
    )
    mapper.add_channel_mapping(
        qcs.Channels(0, "unused_rf"),
        rf_address,
        qcs.InstrumentEnum.M5300AWG,
    )
    mapper.add_channel_mapping(
        qcs.Channels(0, "digitizer"),
        digitizer_address,
        qcs.InstrumentEnum.M5200Digitizer,
    )
    mapper.add_downconverters(digitizer_address, downconverter_address)
    mapper.set_lo_frequencies(downconverter_address, 7.25e9)
    mapper_path = tmp_path / "imported_with_downconverter.qcs"
    qcs.save(mapper, mapper_path)

    configuration = front_panel.configuration_from_qcs_mapper(
        mapper_path,
        dc_channel_names=("dc_left",),
        acquisition_channel_name="digitizer",
    )
    assert any(
        mapping["virtual_name"] == "unused_rf"
        and mapping["role"] == "unassigned"
        for mapping in configuration["channel_mappings"]
    )

    app = _application()
    control = front_panel.QcsFrontPanelControl()
    control.set_settings(
        {
            "mapper_path": str(mapper_path),
            "dc_channel_names": ["dc_left"],
            "rf_channel_names": {},
            "acquisition_channel_name": "digitizer",
        },
        output_count=1,
    )
    control.load_mapper(mapper_path)
    applied = []
    control.settings_applied.connect(applied.append)
    control.apply_button.click()
    app.processEvents()
    assert applied[-1]["hardware_configuration_state"] == "imported"
    assert (
        applied[-1]["hardware_mapper_sha256"]
        == front_panel.qcs_mapper_file_sha256(mapper_path)
    )
    control.close()


def test_imported_active_m5300_requires_native_lo_but_unassigned_does_not(
    tmp_path,
):
    qcs = pytest.importorskip("keysight.qcs")
    mapper = qcs.ChannelMapper()
    rf_address = qcs.Address(1, 3, 1)
    mapper.add_channel_mapping(
        qcs.Channels(0, "rf_drive"),
        rf_address,
        qcs.InstrumentEnum.M5300AWG,
    )
    mapper_path = tmp_path / "imported_unset_m5300.qcs"
    qcs.save(mapper, mapper_path)
    original_digest = front_panel.qcs_mapper_file_sha256(mapper_path)

    unassigned = front_panel.configuration_from_qcs_mapper(mapper_path)
    front_panel.validate_imported_qcs_role_configuration(
        unassigned,
        mapper_path,
    )

    active = front_panel.configuration_from_qcs_mapper(
        mapper_path,
        rf_channel_names={0: "rf_drive"},
    )
    with pytest.raises(ValueError, match=r"M5300.*LO frequency is unset"):
        front_panel.validate_imported_qcs_role_configuration(
            active,
            mapper_path,
        )
    assert front_panel.qcs_mapper_file_sha256(mapper_path) == original_digest


def test_imported_active_m5201_downconverter_requires_native_lo(tmp_path):
    qcs = pytest.importorskip("keysight.qcs")
    mapper = qcs.ChannelMapper()
    digitizer_address = qcs.Address(1, 5, 1)
    downconverter_address = qcs.Address(1, 6, 1)
    mapper.add_channel_mapping(
        qcs.Channels(0, "digitizer"),
        digitizer_address,
        qcs.InstrumentEnum.M5200Digitizer,
    )
    mapper.add_downconverters(digitizer_address, downconverter_address)
    mapper_path = tmp_path / "imported_unset_m5201.qcs"
    qcs.save(mapper, mapper_path)
    original_digest = front_panel.qcs_mapper_file_sha256(mapper_path)
    configuration = front_panel.configuration_from_qcs_mapper(
        mapper_path,
        acquisition_channel_name="digitizer",
    )

    with pytest.raises(ValueError, match=r"M5201.*LO frequency is unset"):
        front_panel.validate_imported_qcs_role_configuration(
            configuration,
            mapper_path,
        )
    assert front_panel.qcs_mapper_file_sha256(mapper_path) == original_digest
