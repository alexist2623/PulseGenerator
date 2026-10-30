"""Direct RAMP duration entry must reach the v7 command, not just its label."""
import json
import socket

import numpy as np
import pytest
from PyQt5 import QtCore, QtTest, QtWidgets

from dc_waveform_core import PulseSequence, build_qick_sequence
from qick_fine_tune_sweep import compile_sequence
from qick_fine_ramp import is_extended
from test_awg_v7_fine import configuration
from test_awg_v6_timing import configuration as old_configuration


DT = 1000 / 4800


def waveform():
    p = PulseSequence(0, 10)
    p.add_flat_ramp(10, 20, 100)
    p.add_flat_ramp(10, 30, -100)
    return p


@pytest.mark.parametrize('samples', range(1, 18))
@pytest.mark.parametrize('row', (1, 2))
def test_every_sub_clock_duration_compiles_and_previews_both_directions(samples, row):
    p = waveform()
    p.set_ramp_duration_ns(row, samples * DT)
    assert p.ramp_duration_ns(row) == pytest.approx(samples * DT)
    assert p.t[2 * row] - p.t[2 * row - 1] == pytest.approx(((samples + 16) // 16) * 10 / 3)
    assert p.t[2 * row + 1] - p.t[2 * row] == pytest.approx(20 if row == 1 else 30)
    t, v = p.waveform_arrays()
    edge = p.t[2 * row - 1] + samples * DT
    assert np.interp(edge, t, v) == p.v[2 * row]
    restored = PulseSequence.from_dict(json.loads(json.dumps(p.to_dict())))
    _, points = compile_sequence(build_qick_sequence((restored,)), configuration(), (1,))
    command = points[0].segment_commands[2 * row - 1][0]
    assert is_extended(command)
    assert command.duration_samples == samples + 1
    assert p.fine_ramps[row] == dict(rise_start=0, rise_end=samples)


@pytest.mark.parametrize('samples', (16, 17, 31, 32, 33, 48))
def test_fine_word_boundaries_do_not_return_to_legacy(samples):
    p = waveform()
    p.set_ramp_duration_ns(1, DT)
    p.set_ramp_duration_ns(1, samples * DT)
    assert p.ramp_duration_ns(1) == pytest.approx(samples * DT)
    assert p.fine_ramps[1] == dict(rise_start=0, rise_end=samples)
    words = (samples + 16) // 16
    assert p.t[2] - p.t[1] == pytest.approx(words * 10 / 3)
    assert p.t[3] - p.t[2] == pytest.approx(20)


def test_fresh_one_word_entry_has_16_intervals_and_a_distinct_endpoint():
    from qick_fine_ramp import scalar_runs
    traces = []
    for samples in (15, 16, 17):
        p = waveform()
        p.set_ramp_duration_ns(1, samples * DT)
        _, points = compile_sequence(build_qick_sequence((p,)), configuration(), (1,))
        command = points[0].segment_commands[1][0]
        assert is_extended(command)
        trace = np.concatenate([
            np.clip((np.arange(n, dtype=np.int64) * a + b) // (1 << 20), lo, hi) * 4
            for n, a, b, lo, hi in scalar_runs(command, 0)
        ])
        assert trace[samples] == command.target_code
        assert trace[samples - 1] < command.target_code
        assert np.all(trace[samples:] == command.target_code)
        traces.append(trace)
    assert not np.array_equal(traces[0][:16], traces[1][:16])


def test_untouched_long_coarse_entries_keep_legacy_compatibility():
    p = waveform()
    p.set_ramp_duration_ns(1, 48 * DT)
    assert not p.fine_ramps
    compile_sequence(build_qick_sequence((p,)), old_configuration(), (1,))


def test_advanced_edge_positions_preserved_and_overlap_is_atomic():
    p = waveform()
    p.set_fine_ramp(1, dict(rise_start=2, rise_end=5, fall_start=14, fall_end=20, turn=150))
    p.set_ramp_duration_ns(1, 4 * DT)
    assert p.fine_ramps[1] == dict(rise_start=2, rise_end=6, fall_start=14, fall_end=20, turn=150)
    before = p.to_dict()
    with pytest.raises(ValueError, match='overlap'):
        p.set_ramp_duration_ns(1, 15 * DT)
    assert p.to_dict() == before


@pytest.mark.parametrize('duration', (0, -1, float('nan'), float('inf')))
def test_invalid_duration_leaves_original_unchanged(duration):
    p = waveform()
    before = p.to_dict()
    with pytest.raises(ValueError):
        p.set_ramp_duration_ns(1, duration)
    assert p.to_dict() == before


def test_old_firmware_rejected_and_actual_clock_used():
    p = waveform()
    p.set_ramp_duration_ns(1, .25, 250)
    assert p.fine_ramps[1]['rise_end'] == 1
    assert p.ramp_duration_ns(1) == .25
    p = waveform()
    p.set_ramp_duration_ns(1, 1.)
    assert p.ramp_duration_ns(1) == pytest.approx(5 * DT)
    with pytest.raises(ValueError, match='v7'):
        compile_sequence(build_qick_sequence((p,)), old_configuration(), (1,))


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_actual_table_add_controls_units_and_trace_dialog(app, monkeypatch):
    import DCWaveform_Generator as gui
    def unexpected(*args, **kwargs):
        pytest.fail('Unexpected UI error or device connection: ' + str(args))
    monkeypatch.setattr(socket.socket, 'connect', unexpected)
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', unexpected)
    p = waveform()
    panel = gui.ControlPanel(p, time_unit='ns')
    panel.table.item(1, 2).setText('0.833333333')
    assert float(panel.table.item(1, 2).text()) == pytest.approx(4 * DT)
    assert 'Command reservation: 3.33333333 ns' in panel.table.item(1, 2).toolTip()
    for unit in ('us', 'ns') * 3:
        panel.set_time_unit(unit)
    assert p.ramp_duration_ns(1) == pytest.approx(4 * DT)
    panel.table.item(1, 2).setText('10')
    assert p.fine_ramps[1] == dict(rise_start=0, rise_end=48)
    values = []
    panel.add_requested.connect(lambda *args: values.append(args))
    panel.edit_ramp.setText('0.208333333')
    panel.edit_flat.setText('100')
    panel.edit_v.setText('200')
    panel._on_add()
    assert values[0][0] == pytest.approx(DT)
    dialog = gui.TraceRampEditDialog(segment_index=1, ramp_ns=10, time_unit='ns')
    dialog.ramp_duration.lineEdit().selectAll()
    QtTest.QTest.keyClicks(dialog.ramp_duration.lineEdit(), '0.625')
    QtTest.QTest.keyClick(dialog.ramp_duration, QtCore.Qt.Key_Return)
    assert dialog.value_ns() == pytest.approx(3 * DT)
    dialog.deleteLater()
    panel.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_main_window_add_save_compile_and_shared_edit(app, monkeypatch, tmp_path):
    import DCWaveform_Generator as gui
    def unexpected(*args, **kwargs):
        pytest.fail('Unexpected UI error or device connection: ' + str(args))
    monkeypatch.setattr(socket.socket, 'connect', unexpected)
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', unexpected)
    window = gui.MainWindow()
    try:
        window._add_port()
        window._add_segment(DT, 100, 100)
        panel = window._multi_ctrl._ctrl_pannels[0]
        panel.set_time_unit('ns')
        assert panel._pulse.ramp_duration_ns(1) == pytest.approx(DT)
        panel.table.item(1, 2).setText('0.833333333')
        for p in window._pulse:
            assert p.ramp_duration_ns(1) == pytest.approx(4 * DT)
        # Advanced-editor padding must survive a later shared hold edit.
        panel._pulse.edit_ramp(2, 10)
        panel.table.item(1, 3).setText('200')
        for p in window._pulse:
            assert p.ramp_duration_ns(1) == pytest.approx(4 * DT)
            assert p.t[2] - p.t[1] == pytest.approx(10)
        runtime = window._experiment_run_arguments(require_readout=False, require_run_config=False)['sequence']
        assert runtime.fine_ramps['ramp_0_to_1'][0]['rise_end'] == 4
        # A direct edit must not silently truncate another port's advanced edge.
        window._pulse[1].set_fine_ramp(1, dict(rise_start=2, rise_end=20))
        before = [p.to_dict() for p in window._pulse]
        warnings = []
        monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', lambda *args: warnings.append(args[2]))
        panel.table.item(1, 2).setText('0.625')
        assert warnings and 'truncate output 2' in warnings[0]
        assert [p.to_dict() for p in window._pulse] == before
        assert not window.isVisible()
        saved = tmp_path / 'settings.json'
        window._save_settings_json(saved)
        assert 'fine_ramps' in saved.read_text()
    finally:
        window.close()
        window.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()


@pytest.mark.parametrize('legacy', (False, True))
def test_real_json_load_restores_fine_metadata_and_clears_stale_edges(app, monkeypatch, tmp_path, legacy):
    import DCWaveform_Generator as gui
    def unexpected(*args, **kwargs):
        pytest.fail('Unexpected device connection or warning: ' + str(args))
    monkeypatch.setattr(socket.socket, 'connect', unexpected)
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', unexpected)
    source, restored = gui.MainWindow(), gui.MainWindow()
    try:
        source._add_segment(10, 100, 100)
        source._add_segment(10, 1000, 0)
        source._pulse[0].set_fine_ramp(1, dict(rise_start=2, rise_end=17), 250)
        source._pulse[0].set_fine_ramp(2, dict(rise_start=0, rise_end=16), 250)
        path = tmp_path/'fine.json'
        if legacy:
            path.write_text(json.dumps(source._pulse[0].to_dict()), encoding='utf-8')
        else:
            source._save_settings_json(path)
        restored._load_settings_json(path)
        assert restored._pulse[0].fine_ramps == source._pulse[0].fine_ramps
        assert restored._pulse[0].fine_fabric_mhz == 250
        assert restored._pulse[0].fine_ramps is not source._pulse[0].fine_ramps
        for row in (1, 2):
            assert restored._pulse[0].ramp_duration_ns(row) == source._pulse[0].ramp_duration_ns(row)
        # An older document without fine fields must not inherit the current edges.
        data = json.loads(path.read_text(encoding='utf-8'))
        if legacy:
            data.pop('fine_ramps');data.pop('fine_fabric_mhz')
        else:
            for output in data['awg']['outputs']:
                output.pop('fine_ramps', None);output.pop('fine_fabric_mhz', None)
        path.write_text(json.dumps(data), encoding='utf-8')
        restored._load_settings_json(path)
        assert not restored._pulse[0].fine_ramps
    finally:
        for w in (source, restored):
            w.close();w.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()
