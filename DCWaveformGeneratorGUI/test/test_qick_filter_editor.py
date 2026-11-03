"""Real Qt filter interactions under the offscreen platform; no hardware."""

import ast
import hashlib
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
import pytest

from qick_filter_editor import QickFilterEditor
from qick_filter_settings import resolve_filter_settings
from qick_front_panel import QickFrontPanelControl, identify_qick_front_panel
from test_qick_front_panel import _live_config


@pytest.fixture
def app():
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert application.platformName() == "offscreen"
    yield application
    application.processEvents()
    # Retain Python owners through C++ deferred deletion. Do not independently
    # destroy pyqtgraph's orphaned scene helpers or combo transient windows.
    roots = [widget for widget in application.topLevelWidgets()
             if isinstance(widget, (QickFrontPanelControl, QickFilterEditor))
             or type(widget).__name__ == "MainWindow"]
    for widget in roots:
        widget.close()
        widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def drag(plot, kind, to_frequency, image=None):
    start = next(point for name, point, marker in plot.marker_positions()
                 if name == kind and (image is None or image == marker))
    QtTest.QTest.mousePress(plot, QtCore.Qt.LeftButton, pos=start.toPoint())
    finish = QtCore.QPointF(plot.frequency_x(to_frequency), start.y())
    QtWidgets.QApplication.sendEvent(plot, QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove, finish, QtCore.Qt.NoButton,
        QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
    QtTest.QTest.mouseRelease(plot, QtCore.Qt.LeftButton, pos=finish.toPoint())


def panel_for_test():
    config = _live_config()
    config["gens"][0].update(fs=4800, f_dds=1200, interpolation=4)
    config["gens"][2].update(fs=6400, f_dds=1600)
    config["readouts"][1].update(fs=2400, f_output=1, f_dds=1200)
    panel = QickFrontPanelControl()
    panel.set_configuration(identify_qick_front_panel(config))
    panel.set_path_values({"output_ch": 0, "readout_ch": 1})
    panel.output_filter_editor.plot.resize(460, 235)
    return panel


@pytest.mark.parametrize("mode,kind,cutoff", [("lowpass", "high", 3.0), ("highpass", "low", 3.2)])
def test_cutoff_drag_updates_real_front_panel_apply_values(app, mode, kind, cutoff):
    panel = panel_for_test()
    editor = panel.output_filter_editor
    editor.mode_combo.setCurrentText(mode)
    updates = []
    panel.settings_applied.connect(updates.append)
    drag(editor.plot, kind, cutoff)
    assert panel.output_filter_cutoff_ghz.value() == pytest.approx(cutoff)
    assert editor.resolved_settings == resolve_filter_settings(mode, cutoff)
    assert not updates  # Editing alone never applies settings.
    panel.apply_button.click()
    assert updates[-1]["output_filter_type"] == mode
    assert updates[-1]["output_filter_cutoff_ghz"] == pytest.approx(cutoff)
    assert not panel.isVisible()


def test_bandpass_drag_preserves_opposite_edge_and_updates_center_bw(app):
    panel = panel_for_test()
    panel.set_path_values({"output_filter_type": "bandpass",
                           "output_filter_cutoff_ghz": 3.6,
                           "output_filter_bandwidth_ghz": 1.2})
    editor = panel.output_filter_editor
    drag(editor.plot, "low", 3.2)
    assert editor.edges() == pytest.approx({"low": 3.2, "high": 4.2})
    assert editor.cutoff.value() == pytest.approx(3.7)
    assert editor.bandwidth.value() == pytest.approx(1)
    drag(editor.plot, "high", 4.8)
    assert editor.edges() == pytest.approx({"low": 3.2, "high": 4.8})
    settings = panel.selected_settings()
    assert settings["output_filter_cutoff_ghz"] == pytest.approx(4.)
    assert settings["output_filter_bandwidth_ghz"] == pytest.approx(1.6)
    # Crossing the other edge is constrained; no negative bandwidth from dragging.
    drag(editor.plot, "low", 5.5)
    assert editor.edges()["low"] < editor.edges()["high"]


def test_bypass_ignores_unused_values_hides_edges_and_keeps_target(app):
    panel = panel_for_test()
    editor = panel.output_filter_editor
    editor.mode_combo.setCurrentText("bandpass")
    editor.cutoff.setValue(100.)
    assert not panel.apply_button.isEnabled()
    editor.mode_combo.setCurrentText("bypass")
    assert panel.apply_button.isEnabled()
    assert editor.low.isHidden() and editor.high.isHidden()
    assert not editor.edges()
    assert editor.resolved_settings.as_dict()["WR0_SW"] == 0xC0
    assert not editor.resolved_settings.nominal_cutoffs_ghz()
    assert all(kind == "tone" for kind, *_ in editor.plot.marker_positions())
    assert editor.plot.response(0) == editor.plot.response(8) == 1
    assert panel.selected_settings()["output_filter_type"] == "bypass"


def test_live_converter_fs_and_linked_image_drag_do_not_change_filter(app):
    panel = panel_for_test()
    editor = panel.output_filter_editor
    assert editor.sample_rate_ghz == 4.8
    assert panel.input_filter_editor.sample_rate_ghz == 2.4
    assert [f for f, *_ in editor.image_frequencies()] == pytest.approx([1.2, 3.6, 6.])
    before = panel.selected_settings()
    drag(editor.plot, "tone", 3.3, image=(1, -1))
    assert editor.target.value() == pytest.approx(1.5)
    assert [f for f, *_ in editor.image_frequencies()] == pytest.approx([1.5, 3.3, 6.3])
    assert panel.selected_settings() == before
    panel.canvas.select_port("output", 15)
    assert editor.sample_rate_ghz == 6.4
    editor.set_sample_rate(None)
    assert len(editor.image_frequencies()) == 1
    assert "unavailable" in editor.fs_label.text()


def test_legacy_json_restores_requested_values_without_requantizing(app, tmp_path):
    legacy = {
        "output_ch": 0, "readout_ch": 1,
        "output_filter_type": "bandpass", "output_filter_cutoff_ghz": 3.805,
        "output_filter_bandwidth_ghz": 1.2,
        "readout_filter_type": "lowpass", "readout_filter_cutoff_ghz": 2.5,
        "readout_filter_bandwidth_ghz": 1.,
    }
    path = tmp_path / "legacy_path.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    panel = panel_for_test()
    panel.set_path_values(json.loads(path.read_text(encoding="utf-8")))
    assert panel.output_filter_editor.edges() == pytest.approx({"low": 3.205, "high": 4.405})
    for key, value in legacy.items():
        assert panel.selected_settings()[key] == value
    assert panel.input_filter_editor.resolved_settings == resolve_filter_settings("lowpass", 2.5, 1.)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
    assert not panel.isVisible()


def test_dc_port_disables_graph_and_forces_bypass(app):
    panel = panel_for_test()
    panel.output_filter_type.setCurrentText("lowpass")
    panel.canvas.select_port("output", 4)
    assert not panel.output_filter_editor.isEnabled()
    assert panel.selected_settings()["output_filter_type"] == "bypass"


def test_api_lowpass_2p5_ignores_bandwidth_and_matches_real_driver(app):
    source = Path(__file__).resolve().parents[3] / "QSTL_QICK/qick/qick_lib/qick/rfboard.py"
    if not source.exists():
        pytest.skip("local QICK reference checkout is unavailable")
    cls = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
               if isinstance(node, ast.ClassDef) and node.name == "FilterADMV8818")
    namespace = {"logger": logging.getLogger("filter-test")}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)

    class FakeSpi:
        def __init__(self):
            self.writes = []

        def send_receive_m(self, message, *_args):
            self.writes.append(bytes(message))
            return bytes(3)

    spi = FakeSpi()
    driver = namespace["FilterADMV8818"](spi, 0)
    driver.set_filter(fc=2.5, bw=1, ftype="lowpass")
    assert spi.writes == [bytes.fromhex("0020c1"), bytes.fromhex("002104")]
    assert resolve_filter_settings("lowpass", 2.5, 1) == resolve_filter_settings("lowpass", 2.5, 9)


def test_saved_project_json_to_front_panel_to_execution_spec(app, monkeypatch):
    import DCWaveform_Generator as gui
    path = Path(__file__).with_name("test.json")
    if not path.exists():
        pytest.skip("local saved experiment JSON is unavailable")
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    window = gui.MainWindow()
    monkeypatch.setattr(window, "_show_settings_migration", lambda _report: None)
    window._load_settings_json(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    target = window._rf_ports_panel._panels[0]
    for key in ("filter_type", "filter_cutoff", "filter_bandwidth"):
        assert target.settings_dict()[key] == saved["rf_outputs"][0][key]
    config = _live_config()
    config["gens"][0]["fs"] = 4800
    window._on_qick_configuration_identified(identify_qick_front_panel(config))
    for method in ("show", "raise_", "activateWindow"):
        monkeypatch.setattr(window._qick_front_panel_dialog, method, lambda: None)
    updates = []
    monkeypatch.setattr(window, "_start_rf_output_hardware_update",
                        lambda connection, spec: updates.append(spec))
    window._show_qick_front_panel("output", target)
    panel = window._qick_front_panel
    editor = panel.output_filter_editor
    editor.plot.resize(460, 235)
    editor.mode_combo.setCurrentText("bandpass")
    editor.cutoff.setValue(3.6)
    editor.bandwidth.setValue(1.2)
    drag(editor.plot, "high", 4.6)
    assert not updates
    panel.apply_button.click()
    assert len(updates) == 1
    assert updates[0].filter_type == "bandpass"
    assert updates[0].filter_cutoff == pytest.approx(3.8)
    assert updates[0].filter_bandwidth == pytest.approx(1.6)
    assert target.settings_dict()["filter_cutoff"] == pytest.approx(3.8)
    assert not window.isVisible() and not window._qick_front_panel_dialog.isVisible()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
