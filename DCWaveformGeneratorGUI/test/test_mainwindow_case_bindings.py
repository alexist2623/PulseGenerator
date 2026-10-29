"""Offline real Qt controls; no worker, board, socket or scope execution."""
from copy import deepcopy
import socket
import sys

import numpy as np
import pytest
from PyQt5 import QtCore, QtGui, QtWidgets
import qick.sim  # noqa: F401 - desktop driver bootstrap

import DCWaveform_Generator as gui
from qick_front_panel import identify_qick_front_panel
from test_qick_front_panel import _live_config
from validation.coverage import Fixture
from validation.awg_reference import ReferenceUnavailable
from validation.mainwindow_cases import MainWindowCaseBinding, validate_case
from validation.qt_runner import apply_actions


@pytest.fixture
def window(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline case binding must not connect to instruments')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle('Fusion')
    app.setAttribute(QtCore.Qt.AA_DontUseNativeDialogs, True)
    cfg = _live_config()
    cfg['gens'] = []
    for index in range(2):
        for kind, role in (('axis_signal_gen_v6', 'rf'), ('axis_awg_tuning_v3', 'awg')):
            cfg['gens'].append(dict(dac=f'1{index}', type=kind, fullpath=f'{kind}_{index}',
                output_sum_version=2, output_sum_role=role, output_sum_bypass=False))
    window = gui.MainWindow()
    # Test-only read-only firmware discovery; waveform/control values use Qt.
    window._on_qick_configuration_identified(identify_qick_front_panel(cfg))
    window.resize(1600, 1000)
    window.show()
    app.processEvents()
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', lambda *a: errors.append(a[1:]) or QtWidgets.QMessageBox.Ok)
    yield window
    assert not errors, errors
    window.close()
    window.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


def fixture():
    return Fixture('TEST::OFFLINE', 'TEST,DSO-X6004A,OFFLINE,1', {
        str(i): dict(generator=2*i-1, physical_connector=f'DAC{i+3}', cable=f'offline-{i}',
                     termination_ohm=1e6, probe_ratio=1, attenuation_db=0, coupling='DC')
        for i in (1, 2)})


def case():
    return dict(name='actual_qt_ab', category='grouped_voltage', execution='hardware_loop',
        outputs=[dict(scope_channel=str(i), generator=2*i-1, physical_connector=f'DAC{i+3}', full_scale_mv=800.)
                 for i in (1, 2)],
        stages=[dict(name='baseline', ramp_us=0., hold_us=10., voltage_mv=[0., 0.]),
                dict(name='pulse', ramp_us=1., hold_us=10., voltage_mv=[100., -150.]),
                dict(name='reset', ramp_us=1., hold_us=10., voltage_mv=[0., 0.])],
        axes=[dict(kind='amplitude', output=0, segment=1, start=-100., stop=200., count=3, group='voltage'),
              dict(kind='amplitude', output=1, segment=1, start=150., stop=-50., count=3, group='voltage')],
        matrix=[[1., .1], [-.2, 1.]], repetitions=2)


def test_actual_controls_build_waveform_matrix_and_grouped_voltage(window):
    binding = MainWindowCaseBinding(window, fixture(), case())
    events = apply_actions(binding.actions())
    snapshot = binding.snapshot()
    assert snapshot['point_count'] == 3
    assert snapshot['repetitions'] == 2
    assert snapshot['matrix'] == [[1., .1], [-.2, 1.]]
    assert [s['name'] for s in snapshot['stages']] == ['baseline', 'pulse', 'reset']
    np.testing.assert_allclose([s['hold_us'] for s in snapshot['stages']], [10., 10., 10.])
    assert snapshot['stages'][1]['voltage_mv'] == [100., -150.]
    assert [a['group'] for a in snapshot['axes']] == ['group_0', 'group_0']
    assert any(e['operation'] == 'context_menu' for e in events)
    assert any(e['operation'] == 'table_text' for e in events)
    args = window._experiment_run_arguments(require_readout=False, require_run_config=False)
    assert args['sequence'].sweep_shape == (3,)
    np.testing.assert_allclose(args['sequence'].sweep_coordinates*window._qick_full_scale_mv,
                               [[-100., 150.], [50., 50.], [200., -50.]])


@pytest.mark.parametrize('kind', ['hold_duration', 'ramp_duration'])
@pytest.mark.parametrize('descending', [False, True])
def test_duration_axes_real_modals_and_independent_loop(window, kind, descending):
    spec = case()
    endpoints = (1., .1) if descending else (.1, 1.)
    spec['axes'].append(dict(kind=kind, output=0, segment=1, start=endpoints[0], stop=endpoints[1], count=2))
    binding = MainWindowCaseBinding(window, fixture(), spec)
    apply_actions(binding.actions())
    snap = binding.snapshot()
    assert snap['point_count'] == 6
    axis = next(a for a in snap['axes'] if a['kind'] == kind)
    assert (axis['start'], axis['stop']) == endpoints
    seq = window._experiment_run_arguments(require_readout=False, require_run_config=False)['sequence']
    assert seq.sweep_shape == (2, 3)
    assert seq.sweep_coordinates.shape == (6, 3)


@pytest.mark.parametrize('dc,rc,mode', [(False,False,'fixed_voltage'), (True,False,'fixed_voltage'),
    (False,True,'fixed_voltage'), (True,True,'fixed_time')])
def test_real_independent_compensation_checkboxes(window, dc, rc, mode):
    spec = case()
    spec['compensation'] = dict(dc_enabled=dc, rc_enabled=rc, dc_mode=mode,
        dc_voltage_mv=150., dc_duration_us=20., rc_tau_us=40.)
    binding = MainWindowCaseBinding(window, fixture(), spec)
    apply_actions(binding.actions())
    comp = binding.snapshot()['compensation']
    assert (comp['dc_enabled'], comp['rc_enabled']) == (dc, rc)
    if dc:
        assert comp['dc_mode'] == mode
        assert comp['dc_duration_us' if mode=='fixed_time' else 'dc_voltage_mv'] == (20. if mode=='fixed_time' else 150.)
    if rc:
        assert comp['rc_tau_us'] == 40.


def test_real_group_lifecycle_removal_deletion_readdition(window):
    spec = case()
    binding = MainWindowCaseBinding(window, fixture(), spec)
    apply_actions(binding.actions())
    a,b = spec['axes']
    for member,count in ((a,4),(b,5)):
        apply_actions(binding.edit_axis_actions(member, count=count))
        assert [s['count'] for s in binding.snapshot()['axes']] == [count,count]
    apply_actions(binding.remove_group_actions(a))
    apply_actions(binding.edit_axis_actions(a, count=2))
    apply_actions(binding.edit_axis_actions(b, count=3))
    assert binding.snapshot()['point_count'] == 6
    assert all(s['group'] is None for s in binding.snapshot()['axes'])
    for member,source in ((a,'experiment'),(b,'awg_output')):
        old = dict(window._sweep_grouping.parameters)
        apply_actions(binding.delete_axis_actions(member, source=source))
        apply_actions(binding.add_axis_actions(member))
        assert set(window._sweep_grouping.parameters.values()) != set(old.values())
        assert all(s['group'] is None for s in binding.snapshot()['axes'])
    apply_actions(binding.group_actions((a,b)))
    apply_actions(binding.edit_axis_actions(b, count=1))
    assert binding.snapshot()['point_count'] == 1
    assert [s['count'] for s in binding.snapshot()['axes']] == [1,1]


def test_invalid_fixture_and_duplicate_shared_duration_are_rejected_before_events(window):
    bad = case()
    bad['outputs'][1]['physical_connector'] = 'DAC6'
    with pytest.raises(ValueError, match='fixture wiring'):
        MainWindowCaseBinding(window, fixture(), bad)
    bad = case()
    bad['axes'] = [dict(kind='ramp_duration', output=i, segment=1, start=.1, stop=1., count=2) for i in (0,1)]
    with pytest.raises(ValueError, match='Duplicate parameter'):
        validate_case(bad, fixture())
    assert len(window._pulse) == 1


def test_binding_import_and_construction_do_not_touch_gui_or_run(window):
    before = window._settings_to_dict()
    MainWindowCaseBinding(window, fixture(), case())
    assert before == window._settings_to_dict()
    assert window._experiment_thread is None


def test_settings_roundtrip_uses_real_file_dialog_and_preserves_groups(window, tmp_path):
    if sys.platform == 'win32' and QtWidgets.QApplication.platformName() == 'offscreen':
        pytest.skip('Static QFileDialog reload access-violates on Windows offscreen; original failures preserved')
    binding = MainWindowCaseBinding(window, fixture(), case())
    apply_actions(binding.actions())
    before = binding.snapshot()
    groups_before = window._sweep_grouping.to_dict()
    path = tmp_path/'case_settings.json'
    apply_actions(binding.settings_actions(path))
    assert path.exists()
    apply_actions(binding.remove_group_actions(case()['axes'][0]))
    apply_actions(binding.edit_axis_actions(case()['axes'][0], count=7))
    assert binding.snapshot()['point_count'] == 21
    apply_actions(binding.settings_actions(path, load=True))
    assert binding.snapshot() == before
    assert window._sweep_grouping.to_dict() == groups_before


def test_offscreen_settings_dialog_limitation_is_explicit_before_open(window, tmp_path):
    if sys.platform != 'win32' or QtWidgets.QApplication.platformName() != 'offscreen':
        pytest.skip('Windows offscreen limitation only')
    binding = MainWindowCaseBinding(window, fixture(), case())
    with pytest.raises(RuntimeError, match='unavailable on Windows offscreen'):
        apply_actions(binding.settings_actions(tmp_path/'not_opened.json', load=True))
    assert QtWidgets.QApplication.activeModalWidget() is None


def test_verification_rejects_enabled_compensation_mismatch(window):
    from validation.qt_runner import QtAction
    binding = MainWindowCaseBinding(window, fixture(), case())
    apply_actions(binding.actions())
    panel = window._experiment_panel
    apply_actions([QtAction(panel.bias_t_type.dc_checkbox, 'check', True)])
    with pytest.raises(ValueError, match='compensation flag dc_enabled'):
        binding.verify_requested_case()


def test_logarithmic_grid_is_individual_actual_gui_input_not_native_log_sweep(window):
    from validation.coverage import logarithmic_durations_us
    from validation.qt_runner import QtAction
    binding = MainWindowCaseBinding(window, fixture(), case())
    apply_actions(binding.actions())
    table = window._multi_ctrl._ctrl_pannels[0].table
    for requested in logarithmic_durations_us():
        def actions():
            yield from binding._tab(window._multi_ctrl)
            yield QtAction(table, 'table_text', dict(row=1, column=3, text=format(requested, '.12g'), expected=requested))
        apply_actions(actions())
        assert binding.snapshot()['stages'][1]['hold_us'] == pytest.approx(requested, abs=1e-9)
        assert all(a['kind'] == 'amplitude' for a in binding.snapshot()['axes'])


def test_scale_mismatch_is_not_silently_used_for_reference(window):
    spec = case()
    spec['outputs'][0]['full_scale_mv'] = 801.48
    binding = MainWindowCaseBinding(window, fixture(), spec)
    with pytest.raises(ValueError, match='full scale'):
        apply_actions(binding.actions())
    assert window._experiment_thread is None


@pytest.mark.parametrize('stage_index', [1, 2])
@pytest.mark.parametrize('ramp_us', [0., -.1])
def test_nonpositive_noninitial_ramp_is_unavailable_before_any_gui_action(window, stage_index, ramp_us):
    spec = case()
    spec['stages'][stage_index]['ramp_us'] = ramp_us
    before = window._settings_to_dict()
    with pytest.raises(ReferenceUnavailable, match='requires positive incoming RAMP'):
        MainWindowCaseBinding(window, fixture(), spec)
    assert window._settings_to_dict() == before
    assert window._experiment_thread is None


def test_failed_add_segment_is_not_repeated(window, monkeypatch):
    attempts, warnings = [], []
    def reject_add(*args):
        attempts.append(args)
        raise ValueError('Injected offline add rejection')
    # Exercise a real Add button and the production error handler. This is an
    # intentional failure injection, never part of the physical case adapter.
    monkeypatch.setattr(window._pulse[0], 'add_flat_ramp', reject_add)
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning',
        lambda *args: warnings.append(args[1:]) or QtWidgets.QMessageBox.Ok)
    binding = MainWindowCaseBinding(window, fixture(), case())
    with pytest.raises(ValueError, match='did not append exactly one stage'):
        apply_actions(binding.actions())
    assert len(attempts) == 1
    assert warnings == [('Invalid input', 'Injected offline add rejection')]
    assert len(window._pulse[0].flat_segments()) == 1
    assert window._experiment_thread is None
