"""Editable virtual-gate scaling through GUI, compilation and persisted metadata.

These are offline tests.  The strict tProcessor interpreter executes the
assembly instruction objects after PMEM compilation; it is not an RTL model.
"""
import json
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PyQt5 import QtCore, QtTest, QtWidgets
from qick.sim import QickSim  # noqa: F401 - desktop driver bootstrap
from qick.awg_tuning import TProcV1BehaviorModel
from qick.output_sum import apply_output_sum_config

import DCWaveform_Generator as gui
from dc_waveform_core import (
    PulseSequence, QickRampRateSweepSpec, QickSweepSpec,
    build_qick_sequence, transform_virtual_waveforms,
)
from qick_fine_tune_sweep import FineTuneDdrResult, FineTuneSequence
from qick_qcodes_experiment import (
    QCODES_STAGING_ENV, QcodesRunConfig, QickConnectionConfig,
    build_awg_vertex_record, build_awg_waveform_recipe, store_qick_result,
)
from test_qick_dc_area import config as base_config
from test_qick_fine_tune_sweep import _expected_words


@pytest.fixture(scope='module')
def app():
    instance = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield instance
    instance.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    instance.processEvents()


@pytest.fixture
def window(app):
    win = gui.MainWindow()
    yield win
    app.processEvents()  # Drain zero-delay editor scroll callbacks before deletion.
    win._qick_front_panel_dialog.close()
    win.close()
    win.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


def v5_config():
    cfg = base_config(2)
    for gen in cfg['gens']:
        apply_output_sum_config(gen, dict(
            output_sum_version=2, output_sum_role='awg',
            post_output_latency_cycles=2, output_sum_awg_delay_cycles=13))
    return cfg


@pytest.mark.parametrize('diagonal', [0.5, 2.0, -1.0, 0.0])
def test_diagonal_editor_accepts_keyboard_input_and_identity_reset(app, diagonal):
    dialog = gui.CrossCapacitanceDialog(2, np.eye(2))
    try:
        editor = dialog._editors[0][0]
        assert not editor.isReadOnly()
        editor.setFocus()
        editor.selectAll()
        QtTest.QTest.keyClicks(editor, str(diagonal))
        editor.interpretText()
        assert dialog.matrix()[0, 0] == diagonal
        dialog._editors[1][1].setValue(-0.5)
        dialog._editors[0][1].setValue(0.25)
        reset = next(b for b in dialog.findChildren(QtWidgets.QPushButton)
                     if b.text() == 'Reset identity')
        QtTest.QTest.mouseClick(reset, QtCore.Qt.LeftButton)
        np.testing.assert_array_equal(dialog.matrix(), np.eye(2))
    finally:
        dialog.close()
        dialog.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_opening_imported_coefficients_does_not_clamp_to_editor_range(app):
    matrix = np.array([[12.5, -12.25], [15.0, -20.0]])
    dialog = gui.CrossCapacitanceDialog(2, matrix)
    try:
        np.testing.assert_array_equal(dialog.matrix(), matrix)
    finally:
        dialog.close()
        dialog.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


@pytest.mark.parametrize('matrix', [
    [[0.5, 0.2], [-0.1, 2.0]],
    [[-1.0, 0.2], [-0.1, 0.0]],
])
def test_settings_json_roundtrip_retains_nonunit_diagonal(window, tmp_path, matrix):
    window._add_port()
    document = window._settings_to_dict()
    document['awg']['cross_capacitance'] = matrix
    source = tmp_path / 'nonunit.json'
    source.write_text(json.dumps(document), encoding='utf-8')
    window._load_settings_json(source)
    np.testing.assert_array_equal(window._cross_capacitance, matrix)
    saved = tmp_path / 'saved.json'
    window._save_settings_json(saved)
    assert json.loads(saved.read_text(encoding='utf-8'))['awg']['cross_capacitance'] == matrix


def test_legacy_json_without_matrix_still_loads_identity(window, tmp_path):
    window._add_port()
    document = window._settings_to_dict()
    document['awg'].pop('cross_capacitance')
    source = tmp_path / 'legacy.json'
    source.write_text(json.dumps(document), encoding='utf-8')
    window._load_settings_json(source)
    np.testing.assert_array_equal(window._cross_capacitance, np.eye(2))


@pytest.mark.parametrize('matrix', [
    [[0.5, 0.2], [-0.1, 2.0]],
    [[-1.0, 0.2], [-0.1, 0.0]],
    [[0.0, 0.0], [0.0, 0.0]],
])
def test_public_transform_and_sequence_keep_virtual_and_scale_physical(matrix):
    pulses = (PulseSequence(100., 1000.), PulseSequence(-80., 1000.))
    time, virtual, physical = transform_virtual_waveforms(pulses, matrix)
    np.testing.assert_array_equal(time, [0., 1000.])
    np.testing.assert_array_equal(virtual, [[100., 100.], [-80., -80.]])
    np.testing.assert_allclose(physical, np.asarray(matrix) @ virtual)
    seq = build_qick_sequence(pulses, output_names=('a', 'b'),
                              full_scale_mv=800., cross_capacitance=matrix)
    np.testing.assert_allclose(seq.virtual_amplitudes_at(0, 0), [100/800, -80/800])
    np.testing.assert_allclose(seq.amplitudes_at(0, 0),
                              np.asarray(matrix) @ np.array([100/800, -80/800]))


@pytest.mark.parametrize('matrix', [
    [[0.5, 0.25], [-0.25, 2.0]],
    [[-1.0, 0.25], [-0.25, 0.0]],
    [[0.0, 0.0], [0.0, 0.0]],
])
@pytest.mark.parametrize('grouped', [False, True])
def test_gui_run_compiles_nonunit_matrix_voltage_ramp_sweeps(window, matrix, grouped):
    window._add_segment(1000., 10000., 100.)
    window._add_segment(1000., 10000., 0.)
    window._add_port()
    window._cross_capacitance = np.asarray(matrix, dtype=float)
    window._experiment_panel.awg_channels.setText('0,1')
    window._experiment_panel.full_scale_mv.setValue(800.)
    window._sweep_specs = [
        QickSweepSpec('set_1', 'awg_0', -0.125, 0.125, 3),
        QickSweepSpec('set_1', 'awg_1', 0.0625, -0.0625, 3),
        QickRampRateSweepSpec('ramp_0_to_1', 0.1, 0.3, 3),
    ]
    window._notify_sweep_state_changed()
    if grouped:
        specs = window._sweep_specs[:2]
        window._sweep_grouping.create_group(
            [window._sweep_grouping.member_id(spec) for spec in specs],
            window._sweep_specs)
        window._notify_sweep_state_changed()
    args = window._experiment_run_arguments(require_readout=False, require_run_config=False)
    seq = args['sequence']
    assert seq.sweep_point_count == (9 if grouped else 27)
    np.testing.assert_array_equal(seq.cross_capacitance, matrix)
    program = seq.make_program(v5_config(), awg_channels=args['awg_channels'],
                               repetitions_per_sweep=2, compile_validation_mode='full')
    program.compile()
    assert len(program.binprog) > 0
    model = TProcV1BehaviorModel(strict=True)
    program.load_runtime_dmem_into_model(model)
    model.run(program, max_steps=1_000_000)
    assert model.ended and not model.timing_conflicts
    # V5 initialization precedes the sweep: RC-disable/reset, zero drain, and
    # DC-area disable, each once per output. Keep these explicitly checked.
    setup = model.output_events[:6]
    assert all((event.word >> 149) & 1 for event in setup[:2])
    assert all((event.word >> 144) & 3 == 1 and event.word & 0xffffffff == 0
               for event in setup[2:4])
    assert all((event.word >> 150) & 1 for event in setup[4:6])
    waveform_events = model.output_events[6:]
    assert [event.word for event in waveform_events] == _expected_words(program)
    # The independent oracle uses requested virtual coordinates, not compiled
    # target models. Binary fractions make all expected targets exact DAC codes.
    events_per_point = len(waveform_events) // seq.sweep_point_count
    voltage_indices = [next(i for i, axis in enumerate(seq.sweep_axes)
                            if axis.axis_kind == 'amplitude' and axis.output_name == name)
                       for name in ('awg_0', 'awg_1')]
    duration_index = next(i for i, axis in enumerate(seq.sweep_axes)
                          if axis.axis_kind == 'ramp_duration')
    for point in range(seq.sweep_point_count):
        coordinate = seq.sweep_coordinate(point)
        physical = np.asarray(matrix) @ np.array([coordinate[i] for i in voltage_indices])
        np.testing.assert_allclose(seq.amplitudes_at(point, 2), physical)
        events = waveform_events[point * events_per_point:(point + 1) * events_per_point]
        for repetition in range(2):
            shot = events[repetition * 10:(repetition + 1) * 10]
            for output in range(2):
                # start SET x2, rise RAMP x2, main SET x2, fall RAMP x2, zero SET x2.
                word = shot[4 + output].word
                target = int(word & 0xffffffff)
                if target >= 1 << 31:
                    target -= 1 << 32
                assert target == round(float(physical[output]) * 32768)
                assert (shot[2 + output].word >> 64) & 0xffffffff == round(coordinate[duration_index] * 300) * 16


@pytest.mark.parametrize('matrix', [
    [[float('nan'), 0.], [0., 1.]],
    [[float('inf'), 0.], [0., 1.]],
    [[1., 0., 0.], [0., 1., 0.]],
])
def test_invalid_matrix_shape_and_nonfinite_remain_rejected(window, matrix):
    window._add_port()
    document = window._settings_to_dict()
    document['awg']['cross_capacitance'] = matrix
    with pytest.raises(ValueError, match='finite|shape'):
        window._decode_settings(document)
    with pytest.raises(ValueError, match='finite|shape'):
        FineTuneSequence(('a', 'b')).set_cross_capacitance(matrix)
    with pytest.raises(ValueError, match='finite|shape'):
        transform_virtual_waveforms((PulseSequence(), PulseSequence()), matrix)


def test_nonunit_diagonal_does_not_bypass_physical_dac_limit():
    seq = FineTuneSequence(('a', 'b')).set_cross_capacitance([[2., 0.], [0., 0.5]])
    seq.add_set('hold', (0.6, 0.), 300)
    with pytest.raises(ValueError, match='physical amplitude'):
        seq.make_program(v5_config(), awg_channels=(0, 1))


def test_qcodes_metadata_stores_nonunit_matrix_and_physical_vertices(tmp_path, monkeypatch):
    monkeypatch.setenv(QCODES_STAGING_ENV, str(tmp_path / 'staging'))
    matrix = [[0.5, 0.25], [-0.25, -1.]]
    seq = FineTuneSequence(('a', 'b')).set_cross_capacitance(matrix)
    seq.add_set('hold', (0.125, -0.0625), 300)
    seq.add_amplitude_sweep('hold', 'a', -0.125, 0.125, 3)
    recipe = build_awg_waveform_recipe(seq, fabric_mhz=300., full_scale_mv=800.)
    result = FineTuneDdrResult(sweep_points=seq.sweep_coordinates,
        iq=np.zeros((3, 1, 4, 2), dtype=np.int32),
        sweep_axes=tuple(seq.sweep_axes), sweep_shape=seq.sweep_shape,
        cross_capacitance=seq.cross_capacitance)
    settings = {'qick': {'full_scale_mv': 800., 'fabric_mhz': 300.},
                'awg': {'cross_capacitance': matrix}, 'awg_waveform_recipe': recipe}
    dataset, _ = store_qick_result(result,
        run_config=QcodesRunConfig(str(tmp_path / 'diagonal.db'),
                                  experiment_name='diagonal', sample_name='offline'),
        connection_config=QickConnectionConfig('192.0.2.10'),
        program_summary={}, gui_settings=settings, rf_settings={})
    assert json.loads(dataset.get_metadata('cross_capacitance_json')) == matrix
    assert json.loads(dataset.get_metadata('awg_waveform_recipe_json'))['cross_capacitance'] == matrix
    for point in range(3):
        record = build_awg_vertex_record(seq, point, fabric_mhz=300., full_scale_mv=800.)
        virtual = np.array([record['virtual_values_mv'][name] for name in ('a', 'b')])
        physical = np.array([record['physical_values_mv'][name] for name in ('a', 'b')])
        np.testing.assert_allclose(physical, np.asarray(matrix) @ virtual)
