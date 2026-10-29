"""Offline infrastructure regression: no board, scope, network or VISA access.

The Qt integration test uses the actual production Run handler and worker with
only their instrument execution boundary replaced by an in-memory result.
"""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from validation import Fixture, FirmwareIdentity, Point, Reference, RunArchive, verify_archive
from validation.archive import atomic, digest, json_file, private_evidence_root
from validation.coverage import logarithmic_durations_us
from validation._scope import compact_capture as compact


@pytest.fixture
def identities(tmp_path):
    driver, gui = tmp_path/'driver.py', tmp_path/'gui.py'
    driver.write_text('driver = 1\n')
    gui.write_text('gui = 1\n')
    fixture = Fixture('TEST::NO_DEVICE', 'TEST,DSO-X6004A,OFFLINE,1', {
        str(ch): dict(generator=2*ch-1, physical_connector=f'DAC{ch+3}',
                     cable=f'cable-{ch}', termination_ohm=1e6, probe_ratio=1,
                     attenuation_db=0, coupling='DC') for ch in (1, 2)})
    identity = FirmwareIdentity('v5', {'gens': [1, 3], 'sum': 'v2'}, 'a'*64, 'b'*64,
        dict(verified=True, method='offline test double'),
        {key: dict(path=str(path), sha256=digest(path.read_bytes()))
         for key, path in [('qick_driver', driver), ('gui', gui)]}, ('sum_v2',))
    return fixture, identity


def archive(tmp_path, identities, points=1, repetitions=1):
    fixture, identity = identities
    return RunArchive(tmp_path/'private', case=dict(name='offline', required_capabilities=['sum_v2']),
                      fixture=fixture, identity=identity,
                      points=[Point(i, {'voltage_v': .01*i}, repetitions) for i in range(points)])


def references(target, gui_definition=None):
    for point in sorted({r['point_index'] for r in target.expected.values()}):
        for ch in (1, 2):
            for kind in ('requested', 'quantized', 'physical'):
                target.add_reference(point, ch, kind, Reference(
                    np.array([-1e-6, 1e-3]), np.array([.01, .01]),
                    dict(units='V', time_origin='program', reference_plane='scope input',
                         model_version='test', transformations=[], kind=kind)))
    target.seal(gui_definition=gui_definition or {'voltage': 10.})


def capture(directory, capture_id='fresh', *, codes=None, origins=(0., .1e-9)):
    """Construct valid original IEEE WORD files, never a fabricated physical run."""
    directory.mkdir(parents=True, exist_ok=False)
    codes = np.asarray(codes if codes is not None else [32778]*16, dtype='<u2')
    result = dict(capture_id=capture_id, same_acquisition=True, validation_passed=True,
                  same_acquisition_basis='offline synthetic fixture only', channels={})
    for ch in (1, 2):
        text = f'1,0,{len(codes)},1,1e-9,{origins[ch-1]},0,.001,0,32768'
        pre = directory/f'ch{ch}.preamble.json'
        raw_path = directory/f'ch{ch}.ieee_block.bin.gz'
        json_file(pre, dict(capture_id=capture_id, channel=ch, preamble_text=text,
                            queried_points=len(codes)))
        raw = compact.base.make_definite_block(codes.tobytes()) + b'\n'
        packed = gzip.compress(raw, mtime=0)
        atomic(raw_path, packed)
        decoded = compact.decode_codes(raw, text, len(codes))
        result['channels'][str(ch)] = dict(payload_sha256=decoded['payload_sha256'],
            archive=dict(path=str(raw_path), preamble_path=str(pre), gzip_sha256=digest(packed),
                         preamble_sha256=digest(pre.read_bytes()), raw_sha256=digest(raw)))
    return result


def fully_record(target):
    directory = target.root/'scope'/'raw'
    data = capture(directory)
    acq = target.record_capture(data, directory)
    for ch in (1, 2):
        target.link_measurement(0, 0, ch, acq, 0, 16,
            identity_evidence=dict(verified=True, method='offline known sequence'))
        target.compare(0, 0, ch, absolute_tolerance_v=.00001)
    return data


def test_private_output_rejects_checkout_and_worktree(tmp_path, identities):
    checkout = tmp_path/'checkout'
    checkout.mkdir()
    (checkout/'.git').mkdir()
    with pytest.raises(ValueError, match='Git checkout'):
        private_evidence_root(checkout/'new'/'deep')
    worktree = tmp_path/'worktree'
    worktree.mkdir()
    (worktree/'.git').write_text('gitdir: elsewhere')
    with pytest.raises(ValueError, match='Git checkout'):
        private_evidence_root(worktree/'measurements')
    with pytest.raises(ValueError, match='analysis_results'):
        private_evidence_root(tmp_path/'analysis_results'/'new')
    assert 'QICK_Private_Analysis' in str(private_evidence_root())


def test_atomic_never_replaces_existing_evidence(tmp_path):
    file = tmp_path/'original'
    atomic(file, b'first')
    with pytest.raises(FileExistsError):
        atomic(file, b'retry')
    assert file.read_bytes() == b'first'


def test_fixture_gaps_and_log_execution_are_explicit(identities):
    fixture, _ = identities
    assert fixture.coverage()['third_output_loopback']['status'] == 'unavailable'
    assert fixture.coverage()['marker_acquisition_timing']['status'] == 'unavailable'
    values = logarithmic_durations_us()
    assert len(values) == 10 and values[0] == .1 and values[-1] == 100
    assert not np.allclose(np.diff(values), np.diff(values)[0])


def test_missing_reference_and_unavailable_fixture_never_pass(tmp_path, identities):
    target = archive(tmp_path, identities)
    with pytest.raises(ValueError, match='references missing'):
        target.seal(gui_definition={})
    with pytest.raises(ValueError, match='Missing physical fixture'):
        target.record_coverage('third_output_loopback', 'passed', reason='split A', evidence=['A'])
    result = target.finalize(worker_status='not_started', cleanup={'passed': True})
    assert not result['complete'] and not result['full_suite_complete']


@pytest.mark.parametrize('field', ['firmware_version', 'hwh_sha256', 'bitstream_sha256', 'loaded_board_config', 'capabilities'])
def test_live_identity_mismatch_is_rejected(identities, field):
    identity = identities[1]
    observed = dict(identity.verify(), verified=True, method='offline live probe')
    observed[field] = None
    with pytest.raises((ValueError, TypeError)):
        identity.verify_observed(observed)


def test_full_native_reload_and_derived_comparison(tmp_path, identities):
    target = archive(tmp_path, identities)
    references(target)
    fully_record(target)
    acq = target.acquisitions['fresh']['channels']
    with np.load(target.root/acq['1']['path']) as a, np.load(target.root/acq['2']['path']) as b:
        assert np.array_equal(a['raw_codes'], np.full(16, 32778))
        assert np.allclose(a['voltage_v'], .01)
        assert b['time_s'][0] - a['time_s'][0] == .1e-9
    result = target.finalize(worker_status='finished', cleanup={'passed': True},
                             storage_identity=dict(run_id=1, guid='test', database_path='test.db'))
    assert result['complete'] and result['verdict'] == 'passed' and result['archive_verified']
    assert not result['full_suite_complete']
    assert verify_archive(target.root)['passed']
    assert (target.root/'index.html').is_file()
    with pytest.raises(RuntimeError, match='immutable'):
        target.event('late')
    raw_path = target.root/acq['1']['raw_path']
    raw_path.write_bytes(b'corrupted test data')
    assert not verify_archive(target.root)['passed']


@pytest.mark.parametrize('invalid', [0, 0x0100, 0xff00])
def test_scope_special_codes_cannot_become_pass(tmp_path, identities, invalid):
    target = archive(tmp_path, identities)
    references(target)
    directory = target.root/'scope'/'raw'
    metadata = capture(directory, codes=[32778, invalid])
    acq = target.record_capture(metadata, directory)
    target.link_measurement(0, 0, 1, acq, 0, 2, identity_evidence=dict(verified=True, method='test'))
    report = target.compare(0, 0, 1, absolute_tolerance_v=1.)
    assert report['status'] == 'inconclusive' and report['invalid_samples'] == 1


@pytest.mark.parametrize('damage', ['gzip', 'capture_id', 'channel'])
def test_corrupt_or_stale_originals_are_rejected_and_preserved(tmp_path, identities, damage):
    target = archive(tmp_path, identities)
    references(target)
    directory = target.root/'scope'/'raw'
    metadata = capture(directory)
    info = metadata['channels']['1']['archive']
    if damage == 'gzip':
        Path(info['path']).write_bytes(b'truncated')
    else:
        pre = Path(info['preamble_path'])
        data = json.loads(pre.read_text())
        data[damage] = 'other'
        pre.write_text(json.dumps(data))
        info['preamble_sha256'] = digest(pre.read_bytes())
    with pytest.raises(ValueError):
        target.record_capture(metadata, directory)
    assert len(list(directory.iterdir())) == 4
    assert not target.acquisitions


def test_duplicate_missing_out_of_order_and_overlap_detection(tmp_path, identities):
    target = archive(tmp_path, identities, points=2, repetitions=2)
    references(target)
    directory = target.root/'scope'/'raw'
    metadata = capture(directory)
    acq = target.record_capture(metadata, directory)
    with pytest.raises(ValueError, match='Duplicate or stale'):
        target.record_capture(metadata, directory)
    ident = dict(verified=True, method='test')
    target.link_measurement(0, 0, 1, acq, 0, 4, identity_evidence=ident)
    with pytest.raises(ValueError, match='Duplicate point'):
        target.link_measurement(0, 0, 1, acq, 4, 8, identity_evidence=ident)
    with pytest.raises(ValueError, match='Overlapping'):
        target.link_measurement(0, 1, 1, acq, 3, 6, identity_evidence=ident)
    target.link_measurement(1, 0, 1, acq, 8, 12, identity_evidence=ident)
    with pytest.raises(ValueError, match='Out-of-order'):
        target.link_measurement(0, 1, 1, acq, 4, 8, identity_evidence=ident)
    result = target.finalize(worker_status='finished', cleanup={'passed': True}, storage_identity={'run_id':1})
    assert len(result['missing_measurements']) == 6 and not result['complete']


@pytest.mark.parametrize('failure', ['cleanup', 'source', 'error'])
def test_terminal_failures_cannot_be_complete(tmp_path, identities, failure):
    target = archive(tmp_path, identities)
    references(target)
    fully_record(target)
    if failure == 'source':
        Path(identities[1].source_files['gui']['path']).write_text('changed')
    result = target.finalize(worker_status='finished', cleanup={'passed': failure != 'cleanup'},
        storage_identity={'run_id':1}, error='capture validation error' if failure == 'error' else None)
    assert not result['complete'] and result['verdict'] == 'inconclusive'


def test_residual_failure_and_original_values_remain_separate(tmp_path, identities):
    target = archive(tmp_path, identities)
    references(target)
    directory = target.root/'scope'/'raw'
    acq = target.record_capture(capture(directory), directory)
    for ch in (1,2):
        target.link_measurement(0, 0, ch, acq, 0, 16, identity_evidence=dict(verified=True, method='test'))
        assert target.compare(0, 0, ch, voltage_offset_v=.02,
                              absolute_tolerance_v=.015)['status'] == 'failed'
    result = target.finalize(worker_status='finished', cleanup={'passed':True},
                            storage_identity={'run_id':1,'guid':'offline','database_path':'offline.db'})
    assert result['complete'] and result['verdict'] == 'failed'
    with np.load(target.root/'comparisons'/'p00000000'/'r000000'/'ch1.npz') as data:
        assert np.allclose(data['original_voltage_v'], .01)
        assert np.allclose(data['corrected_voltage_v'], -.01)


def test_import_and_session_construction_never_connect(tmp_path):
    from validation.scope import ScopeSession, visa_resource_factory
    calls = []
    session = ScopeSession(resource_factory=lambda: calls.append('opened'),
        archive_directory=tmp_path/'new', expected_identity='offline', configuration={},
        max_sample_interval_s=1e-9)
    visa_resource_factory('TEST::NEVER_OPENED')
    assert not calls and not session.directory.exists()


def test_normal_detached_qcodes_identity_is_recorded():
    from validation.qt_runner import stored_run_identity
    result = SimpleNamespace(run_id=7, guid='offline-guid', database_path=Path('offline.db'),
                             row_count=200, iq_storage_mode='raw')
    assert stored_run_identity(result) == dict(run_id=7, guid='offline-guid',
        database_path='offline.db', row_count=200, iq_storage_mode='raw')


def test_atomic_scope_storage_preserves_malformed_binary(tmp_path):
    from validation.scope import _atomic_adapter_class
    class Resource:
        timeout = 1000
        read_termination = '\n'
        def write(self, command): pass
        def read_raw(self): return b'#216truncated'
    adapter = _atomic_adapter_class()(Resource(), expected_serial=None)
    path = tmp_path/'bad.ieee.gz'
    with pytest.raises(Exception, match='(block|length|payload|Payload|Truncated)'):
        adapter._binary_query(':WAVeform:DATA?', archive_path=path)
    assert gzip.decompress(path.read_bytes()) == b'#216truncated'
    assert str(path) in adapter.partial_archives


@pytest.fixture(scope='module')
def qt_app():
    from PyQt5 import QtWidgets, QtCore
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


def test_real_qt_events_commit_values_and_change_tabs(qt_app):
    from PyQt5 import QtWidgets
    from validation.qt_runner import QtAction, apply_actions
    window = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(window)
    tabs, spin = QtWidgets.QTabWidget(), QtWidgets.QDoubleSpinBox()
    spin.setRange(-400, 400)
    tabs.addTab(QtWidgets.QWidget(), 'First')
    tabs.addTab(QtWidgets.QWidget(), 'Second')
    layout.addWidget(spin)
    layout.addWidget(tabs)
    window.show()
    qt_app.processEvents()
    result = apply_actions([QtAction(spin, 'text', '-123.25', -123.25), QtAction(tabs, 'tab', 1, 1)])
    assert result[0]['actual'] == -123.25 and result[1]['actual'] == 1
    window.close()
    window.deleteLater()


def test_real_modal_and_combo_input(qt_app):
    from PyQt5 import QtWidgets
    from validation.qt_runner import QtAction, apply_actions
    button = QtWidgets.QPushButton('Open')
    retained = []
    def open_dialog():
        dialog = QtWidgets.QDialog(button)
        layout = QtWidgets.QVBoxLayout(dialog)
        dialog.combo = QtWidgets.QComboBox()
        dialog.combo.addItems(['first', 'second', 'third'])
        dialog.accept_button = QtWidgets.QPushButton('OK')
        dialog.accept_button.clicked.connect(dialog.accept)
        layout.addWidget(dialog.combo)
        layout.addWidget(dialog.accept_button)
        dialog.exec_()
        retained.append(dialog.combo.currentIndex())
    button.clicked.connect(open_dialog)
    button.show()
    qt_app.processEvents()
    report = apply_actions([QtAction(button, 'modal', lambda d:[
        QtAction(d.combo, 'combo', 2, 2), QtAction(d.accept_button, 'click')])])
    assert retained == [2] and len(report[0]['nested']) == 2
    button.close()
    button.deleteLater()


@pytest.mark.parametrize('failure', [None, 'worker', 'capture', 'identity', 'restore', 'cancel', 'gui_changed'])
def test_actual_production_run_worker_and_async_scope(tmp_path, identities, qt_app, monkeypatch, failure):
    from PyQt5 import QtCore, QtWidgets, QtTest
    from qick.sim import QickSim  # noqa: F401 - desktop bootstrap, not a device
    import DCWaveform_Generator as production
    from validation.qt_runner import GuiHardwareRunner, QtAction
    target = archive(tmp_path, identities)
    references(target, {'voltage':10.})
    calls = []
    from threading import Event
    arm_started, gui_changed = Event(), Event()
    original = production.QickExperimentWorker

    class Result:
        def detach_dataset(self): calls.append('dataset_detached')
    def execute(**kwargs):
        calls.append('normal_execution_boundary')
        kwargs['progress_callback'](10, 'offline test')
        kwargs['event_callback']('test', 'started', 'offline')
        for _ in range(4):
            time.sleep(.02)
            kwargs['cancel_check']()
        if failure == 'worker':
            raise RuntimeError('injected offline execution failure')
        return Result()
    monkeypatch.setattr(production, 'run_qick_qcodes_experiment', execute)

    class Panel:
        def __getattr__(self, _): return lambda *a, **k: None
    class Window(QtWidgets.QMainWindow):
        _run_qick_experiment = production.MainWindow._run_qick_experiment
        def __init__(self):
            super().__init__()
            self._experiment_thread = None
            self._experiment_panel = Panel()
            central = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(central)
            self.spin, self.button = QtWidgets.QDoubleSpinBox(), QtWidgets.QPushButton('Run')
            layout.addWidget(self.spin)
            layout.addWidget(self.button)
            self.setCentralWidget(central)
            self.button.clicked.connect(self._run_qick_experiment)
        def _experiment_run_arguments(self):
            calls.append('normal_run_arguments')
            return dict(rf_specs=(), sequence=SimpleNamespace(sweep_point_count=1),
                        repetitions_per_sweep=1, readout_spec=SimpleNamespace(samples_per_trigger=1),
                        iq_storage_mode=production.IQ_STORAGE_MEAN_IQ)
        def _on_experiment_finished(self, result): calls.append('normal_gui_finished')
        def _on_experiment_failed(self, error): calls.append('normal_gui_failed')
        def _on_experiment_cancelled(self, error): calls.append('normal_gui_cancelled')
        def _on_experiment_progress(self, *args): pass
        def _on_experiment_event(self, *args): pass
        def _clear_experiment_thread(self): self._experiment_thread = None

    class Scope:
        def __init__(self, directory): self.directory = directory
        def arm(self):
            calls.append('arm')
            arm_started.set()
            if failure == 'gui_changed' and not gui_changed.wait(2.):
                raise RuntimeError('The offline GUI edit did not occur while scope arm was pending')
            time.sleep(.05)
            return {'armed':True}
        def capture(self):
            time.sleep(.02)
            result = capture(self.directory/'raw')
            if failure == 'capture': raise RuntimeError('injected capture failure after raw storage')
            return result
        def close(self):
            calls.append('restore')
            if failure == 'restore': raise RuntimeError('injected restore failure')
            return {'passed':True}

    def live_identity():
        result = dict(identities[1].verify(), verified=True, method='offline test probe')
        if failure == 'identity': result['bitstream_sha256'] = 'c'*64
        return result
    def cleanup():
        calls.append('zero_cleanup')
        return {'passed':True, 'rc_explicit_off_reset':True, 'output_zero_verified':True}
    def mapping(*_):
        return [dict(point=0, repetition=0, channel=ch, start=0, stop=16,
                     identity_evidence=dict(verified=True, method='known offline fixture')) for ch in (1,2)]

    window = Window()
    window.show()
    qt_app.processEvents()
    runner = GuiHardwareRunner(window=window, production_module=production, run_button=window.button,
        archive=target, scope_factory=Scope, live_identity_check=live_identity, result_identity=lambda _:
        dict(run_id=1, guid='offline', database_path='offline.db'), point_mapper=mapping,
        cleanup=cleanup, comparison=lambda *_:dict(absolute_tolerance_v=.001), timeout_s=5.,
        execution_context='offline_test')
    done, ticks = [], []
    runner.finished.connect(done.append)
    heartbeat = QtCore.QTimer()
    def tick():
        ticks.append(time.monotonic())
        if failure == 'gui_changed' and arm_started.is_set() and not gui_changed.is_set():
            from validation.qt_runner import apply_actions
            apply_actions([QtAction(window.spin, 'text', '11', 11.)])
            calls.append('gui_changed_during_arm')
            gui_changed.set()
    heartbeat.timeout.connect(tick)
    heartbeat.start(5)
    runner.start(allow_hardware=True, actions=[QtAction(window.spin, 'text', '10', 10.)],
                 gui_snapshot=lambda:{'voltage':window.spin.value()})
    if failure == 'cancel': QtCore.QTimer.singleShot(75, runner.cancel)
    deadline = time.monotonic() + 15
    while not done and time.monotonic() < deadline:
        QtTest.QTest.qWait(10)
    heartbeat.stop()
    assert done, 'Runner did not complete its offline fixture'
    assert production.QickExperimentWorker is original
    assert len(ticks) >= 5  # Arm/capture/plots did not block the Qt event loop.
    if failure == 'gui_changed':
        assert 'gui_changed_during_arm' in calls
        assert 'normal_run_arguments' not in calls
        assert 'normal_execution_boundary' not in calls
        assert 'zero_cleanup' not in calls  # No DUT worker started, so no unsolicited DUT write.
        assert 'restore' in calls
        assert runner.worker is None and runner.cleanup_result['output_cleanup']['skipped']
        assert runner.cleanup_result['scope_restore']['passed']
        assert not (target.root/'execution_route.json').exists()
    elif failure != 'identity':
        assert 'zero_cleanup' in calls
        assert calls.index('arm') < calls.index('normal_run_arguments') < calls.index('normal_execution_boundary')
        assert 'restore' in calls
    else:
        assert 'normal_execution_boundary' not in calls
        assert 'zero_cleanup' not in calls  # Read-only identity failure cannot cause a DUT write.
    if failure is None:
        assert done[0]['complete'] and done[0]['verdict'] == 'passed'
        assert 'dataset_detached' in calls and 'normal_gui_finished' in calls
        assert done[0]['execution_route']['scope_only_bypass'] is False
        assert done[0]['execution_route']['execution_context'] == 'offline_test'
    else:
        assert not done[0]['complete']
    assert (target.root/'manifest.json').is_file()
    window.close()
    window.deleteLater()
    qt_app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    qt_app.processEvents()


def test_suite_preserves_failed_attempts_and_missing_coverage(tmp_path, identities):
    from validation.suite import SuiteManifest, duration_case_plan
    planned = dict(name='offline', category='voltage', required_capabilities=['sum_v2'])
    suite = SuiteManifest(tmp_path/'suite-private', name='test suite', fixture=identities[0],
                          planned_cases=[planned])
    run = RunArchive(tmp_path/'run-private', case=planned, fixture=identities[0], identity=identities[1],
                     points=[Point(0, {'voltage_v':.01})])
    references(run)
    run.finalize(worker_status='failed', cleanup={'passed':True}, error='offline test failure')
    suite.record(run.root)
    status = suite.checkpoint()
    assert not status['planned_cases_complete'] and not status['full_suite_passed']
    assert len(next(iter(status['cases'].values()))['attempts']) == 1
    assert status['coverage']['third_output_loopback']['status'] == 'unavailable'
    with pytest.raises(ValueError, match='Duplicate run'):
        suite.record(run.root)
    grid = duration_case_plan('ramp')
    assert len(grid) == 10 and all(c['execution'] == 'individual_gui_log_grid' for c in grid)
