"""v5 Run metadata and Show-program coordinate inspection share one export."""
import json

import numpy as np
import pytest
import qick.sim  # noqa: F401 - desktop import bootstrap

from qick_execution_coordinates import build_execution_coordinates
from qick_fine_tune_sweep import FineTuneDdrResult
import qick_qcodes_experiment as experiment
from dc_waveform_core import QickDdrReadoutSpec, QickRfSoftwareSweepAxisSpec
from test_execution_coordinates import awg_program, compiled_bytes


def test_run_snapshots_before_acquire_and_cancellation_prevents_acquire(monkeypatch):
    program = awg_program()
    before = compiled_bytes(program)
    monkeypatch.setattr(experiment, 'configure_rf_readout', lambda *a: {})
    monkeypatch.setattr(experiment, 'build_qick_program', lambda *a, **k: program)
    calls = []

    def acquire(*args, **kwargs):
        assert program._execution_coordinate_snapshot['point_count'] == 9
        calls.append('acquire')
        return object()

    monkeypatch.setattr(program, 'acquire_fir_ddr', acquire)
    kwargs = dict(awg_channels=(0, 1), repetitions_per_sweep=2,
                  rf_specs=(), readout_spec=QickDdrReadoutSpec(0, 'hold', 0., 1))
    experiment._execute_qick_sequence_once(object(), program.soccfg, program.sequence, **kwargs)
    assert calls == ['acquire']
    assert compiled_bytes(program) == before

    del program._execution_coordinate_snapshot
    checks = []

    def cancel():
        checks.append(None)
        if len(checks) == 8:
            raise experiment.ExperimentCancelled('cancel during coordinate snapshot')

    with pytest.raises(experiment.ExperimentCancelled, match='coordinate snapshot'):
        experiment._execute_qick_sequence_once(
            object(), program.soccfg, program.sequence, cancel_check=cancel, **kwargs)
    assert calls == ['acquire']


def test_coordinate_export_error_prevents_acquisition(monkeypatch):
    import qick_execution_coordinates as exporter
    program = awg_program()
    monkeypatch.setattr(experiment, 'configure_rf_readout', lambda *a: {})
    monkeypatch.setattr(experiment, 'build_qick_program', lambda *a, **k: program)
    calls = []
    monkeypatch.setattr(program, 'acquire_fir_ddr', lambda *a, **k: calls.append('acquire'))

    def invalid(*args, **kwargs):
        raise ValueError('invalid execution coordinate model')

    monkeypatch.setattr(exporter, 'build_execution_coordinates', invalid)
    with pytest.raises(ValueError, match='execution coordinate model'):
        experiment._execute_qick_sequence_once(object(), program.soccfg, program.sequence,
            awg_channels=(0, 1), repetitions_per_sweep=2, rf_specs=(),
            readout_spec=QickDdrReadoutSpec(0, 'hold', 0., 1))
    assert calls == []


@pytest.mark.parametrize('bundle', [False, True])
def test_compiled_coordinates_survive_qcodes_metadata_roundtrip(tmp_path, monkeypatch, bundle):
    monkeypatch.setenv(experiment.QCODES_STAGING_ENV, str(tmp_path / 'staging'))
    first = awg_program()
    experiment._cache_execution_coordinates(first)
    seq = first.sequence
    data = FineTuneDdrResult(sweep_points=seq.sweep_coordinates,
        iq=np.arange(9 * 2 * 2 * 2, dtype=np.int32).reshape(9, 2, 2, 2),
        sweep_axes=tuple(seq.sweep_axes), sweep_shape=seq.sweep_shape,
        sweep_groups=seq.sweep_groups, sweep_parameter_ids=seq.sweep_member_ids,
        cross_capacitance=seq.cross_capacitance)
    program = first
    if bundle:
        second = awg_program()
        experiment._cache_execution_coordinates(second)
        axis = QickRfSoftwareSweepAxisSpec('hold', 'rf_0', 0, 0., 90., 2,
                                         'rf_phase', 'deg', 'phase')
        coordinates = np.array([[0.], [90.]])
        program = experiment.SoftwareSweepProgramBundle((first, second), (axis,), coordinates)
        data = experiment._combine_software_sweep_results(
            [data, data], ((0, axis),), coordinates)
    expected = build_execution_coordinates(program)
    summary = experiment._program_summary_with_coordinates(program)
    dataset, _ = experiment.store_qick_result(data,
        run_config=experiment.QcodesRunConfig(str(tmp_path / 'coordinates.db')),
        connection_config=experiment.QickConnectionConfig('192.0.2.10'),
        program_summary=summary, gui_settings={}, rf_settings={})
    loaded = experiment.load_qick_iq_arrays(dataset)
    actual = loaded['metadata']['program_summary']['execution_coordinates']
    assert actual == json.loads(json.dumps(expected))
    if bundle:
        assert len(actual['software_points']) == 2
        assert actual['software_points'][1]['combined_point_offset'] == 9
        hardware = actual['software_points'][0]['hardware']
    else:
        hardware = actual
    assert hardware['point_count'] == 9
    hold = next(c for c in hardware['awg_commands'] if c['segment_name'] == 'hold')
    assert 'target_code_14' in hold['columns']
    assert 'target_mv' in hold['columns']
    assert 'not measured analog' in hardware['coordinate_domain']


def test_show_program_coordinates_are_sparse_and_preserve_assembly(monkeypatch):
    from PyQt5 import QtWidgets
    import DCWaveform_Generator as gui
    import qick_execution_coordinates as exporter

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    program = awg_program()
    before = compiled_bytes(program)
    assembly = program.asm()
    calls = []
    original = exporter.build_execution_coordinates

    def sparse(*args, **kwargs):
        calls.append(kwargs.get('point_indices'))
        return original(*args, **kwargs)

    monkeypatch.setattr(exporter, 'build_execution_coordinates', sparse)
    dialog = gui.QickAssemblyDialog({'assembly': assembly, 'program': program})
    try:
        dialog.coordinate_point.setValue(8)
        app.processEvents()
        assert calls == [(0,), (8,)]
        assert 'Sweep point: 8' in dialog.coordinate_text.toPlainText()
        assert 'DAC code' in dialog.coordinate_text.toPlainText()
        assert 'mV' in dialog.coordinate_text.toPlainText()
        assert 'amplitude/hold/a' in dialog.coordinate_text.toPlainText()
        assert 'Group indices: 0: 2, 1: 2' in dialog.coordinate_text.toPlainText()
        assert dialog.coordinate_text.isReadOnly()
        dialog.copy_button.click()
        assert QtWidgets.QApplication.clipboard().text() == assembly
        assert compiled_bytes(program) == before
    finally:
        dialog.close()
        dialog.deleteLater()
        app.sendPostedEvents(None, 0)
        app.processEvents()
