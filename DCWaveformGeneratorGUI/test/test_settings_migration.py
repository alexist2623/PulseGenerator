"""Legacy files migrate through the real loader without hardware I/O."""
import copy
import json
import os
import socket

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt5 import QtCore, QtWidgets
import DCWaveform_Generator as gui


@pytest.fixture
def window(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    def unexpected(*args, **kwargs):
        pytest.fail('Import attempted network access or raised a warning: ' + str(args))
    monkeypatch.setattr(socket.socket, 'connect', unexpected)
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', unexpected)
    w = gui.MainWindow()
    yield w
    w.close()
    w.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


def test_old_native_preserves_pulse_and_notifies_then_current_reload_is_quiet(window, tmp_path):
    w = window
    w._add_segment(10, 100, 100)
    w._add_segment(10, 1000, 0)
    w._pulse[0].set_fine_ramp(1, dict(rise_start=2, rise_end=17), 300)
    source = w._settings_to_dict()
    source['version'] = 43
    path = tmp_path/'old.json'
    path.write_text(json.dumps(source), encoding='utf-8-sig')
    original = path.read_bytes()
    w._load_settings_json(path)
    restored = w._settings_to_dict()
    assert restored['version'] == gui.SETTINGS_VERSION
    for key in ('awg', 'sweep_grouping', 'rf_outputs', 'rf_readout', 'qick'):
        assert restored[key] == source[key]
    assert w._last_settings_migration['source_version'] == 43
    assert w._settings_migration_notice.isVisible()
    assert not w._settings_migration_notice.isModal()
    assert 'hold time and RF delay are retained' in w._settings_migration_notice.findChild(QtWidgets.QLabel, 'migrationDetails').text()
    assert path.read_bytes() == original
    saved = w._save_settings_json(tmp_path/'new.json')
    w._load_settings_json(saved)
    assert w._last_settings_migration is None
    assert w._settings_migration_notice is None


@pytest.mark.parametrize('bad', [{'schema':'unknown'}, {'unexpected':1},
    {'schema':gui.SETTINGS_SCHEMA,'version':gui.SETTINGS_VERSION+1}])
def test_invalid_file_leaves_current_pulse_unchanged(window, tmp_path, bad):
    before = window._settings_to_dict()
    path = tmp_path/'invalid.json'
    path.write_text(json.dumps(bad))
    with pytest.raises((ValueError, TypeError)):
        window._load_settings_json(path)
    assert window._settings_to_dict() == before


def test_legacy_single_waveform_notifies(window, tmp_path):
    path = tmp_path/'legacy.json'
    path.write_text(json.dumps(window._pulse[0].to_dict()))
    window._load_settings_json(path)
    assert window._last_settings_migration['source_version'] is None
    assert window._settings_migration_notice.isVisible()


def test_recipe_import_preserves_units_matrix_axes_and_compensation(window, tmp_path):
    recipe = dict(schema='qick-awg-waveform-recipe-v1', fabric_mhz=300., full_scale_mv=800.,
        output_names=['left','right'], cross_capacitance=[[1.,-.7],[.7,1.]],
        segments=[dict(kind='set',name='base',amplitudes=[0.,0.],duration_cycles=300),
                  dict(kind='ramp',name='rise',duration_cycles=300),
                  dict(kind='set',name='hold',amplitudes=[.2,-.5],duration_cycles=3000)],
        sweep_axes=[dict(axis_kind='amplitude',coordinate_unit='normalized',segment_name='hold',
                         output_name='right',start=-.5,stop=.5,count=101)],
        bias_t_compensation=dict(kind='dc',mode='fixed_time',amplitude=.1,
                                 fixed_duration_cycles=24000))
    path = tmp_path/'recipe.json'
    path.write_text(json.dumps(recipe))
    window._load_settings_json(path)
    doc = window._settings_to_dict()
    assert doc['awg']['outputs'][0]['time_ns'] == [0.,1000.,2000.,12000.]
    assert doc['awg']['outputs'][1]['voltage_mv'] == [0.,0.,-400.,-400.]
    assert doc['awg']['cross_capacitance'] == recipe['cross_capacitance']
    assert doc['awg']['sweeps'][0] == dict(axis_kind='amplitude',segment_name='set_1',
        output_name='awg_1',start=-.5,stop=.5,count=101)
    assert doc['qick']['bias_t_compensation']['duration_us'] == 80.
    assert doc['qick']['bias_t_compensation']['voltage_mv'] == 80.
    assert not doc['rf_readout']['enabled']
    assert not any(r['enabled'] for r in doc['rf_outputs'])
    assert window._settings_migration_notice.isVisible()
    before = copy.deepcopy(doc)
    recipe['sweep_axes'][0]['axis_kind'] = 'duration'
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='amplitude sweeps'):
        window._load_settings_json(path)
    assert window._settings_to_dict() == before
