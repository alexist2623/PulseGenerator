"""Four storage choices, board mean protocol, and dated HDF5 persistence."""

from dataclasses import replace
from datetime import datetime, timezone, timedelta
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from qick_fine_tune_sweep import FineTuneDdrResult, FineTuneSequence, DdrFirReadoutConfig
from qick_hdf5 import load_hdf5_result
from qick_trace_average import require_board_trace_average, read_board_mean
from qick_qcodes_experiment import (
    IQ_STORAGE_MODES, QcodesRunConfig, QickConnectionConfig, store_qick_result,
    load_qick_iq_arrays, ExperimentCancelled, QCODES_STAGING_ENV,
)
from test_grouped_sweep_compiler import grouped_sequence
from test_qick_fine_tune_sweep import _fir_soccfg


def make_result(mode):
    sequence = grouped_sequence(3)
    raw = np.arange(9 * 3 * 7 * 2, dtype=np.int64).reshape(9, 3, 7, 2) * 2**40 + 19
    result = FineTuneDdrResult(iq=raw, sweep_points=sequence.sweep_coordinates,
        sweep_axes=sequence.sweep_axes, sweep_shape=sequence.sweep_shape,
        sweep_groups=sequence.sweep_groups, sweep_parameter_ids=sequence.sweep_member_ids,
        cross_capacitance=sequence.cross_capacitance, iq_scale_log2=46,
        iq_component_bits=64, accumulation_repetitions=3)
    if mode.startswith('qick_'):
        iq = np.ldexp(raw.astype(float), -46).mean(axis=1, keepdims=True)
        if mode == 'qick_mean_iq':
            iq = iq.mean(axis=2, keepdims=True)
        result = replace(result, iq=iq, iq_scale_log2=0,
            data_kind='point_mean' if mode == 'qick_mean_iq' else 'repetition_mean',
            source_iq_scale_log2=46, source_iq_component_bits=64, source_samples_per_trace=7)
    return result


@pytest.mark.parametrize('mode', IQ_STORAGE_MODES)
@pytest.mark.parametrize('backend', ['qcodes', 'hdf5'])
def test_four_modes_roundtrip_with_grouped_coordinates_and_iq64(tmp_path, monkeypatch, mode, backend):
    monkeypatch.setenv(QCODES_STAGING_ENV, str(tmp_path / 'staging'))
    result = make_result(mode)
    dataset, count = store_qick_result(result,
        run_config=QcodesRunConfig(str(tmp_path / 'test.db'), storage_format=backend,
            hdf5_directory=str(tmp_path / 'measurements'), experiment_name='실험 / ramp: test'),
        connection_config=QickConnectionConfig(), program_summary={'executed': True},
        gui_settings={'qick': {'iq_storage_mode': mode}, 'awg_waveform_recipe': {'schema': 'test'}},
        rf_settings={}, iq_storage_mode=mode,
        started_at=datetime(2026, 11, 2, 23, 59, 59, tzinfo=timezone(timedelta(hours=-8))))
    scalar = mode in {'mean_iq', 'qick_mean_iq'}
    if backend == 'hdf5':
        assert dataset.path.parent.name == '2026-11-02'
        assert dataset.path.name == '20261102_235959_실험 _ ramp_ test.h5'
        loaded = load_hdf5_result(dataset.path)
        metadata = loaded['metadata']['acquisition']
        assert metadata['repetitions'] == 3 and metadata['source_samples_per_trace'] == 7
        assert loaded['metadata']['sweep_groups'] == json.loads(json.dumps(result.sweep_groups))
        np.testing.assert_array_equal(loaded['coordinates'], result.sweep_points)
        expected = result.analysis_iq.mean(axis=(1, 2)) if scalar else result.iq[:, 0] if mode == 'qick_mean_trace' else result.iq
        np.testing.assert_array_equal(loaded['iq'], expected)
        if scalar:
            assert loaded['time_s'] is None
        else:
            np.testing.assert_array_equal(loaded['time_s'], np.arange(7)/1e6)
        if mode == 'full_traces':
            assert loaded['iq'].dtype == np.int64 and loaded['iq_scale_log2'] == 46
        with h5py.File(dataset.path) as h5:
            assert h5.attrs['status'] == 'completed' and h5.attrs['completed_points'] == 9
    else:
        loaded = load_qick_iq_arrays(dataset)
        metadata = loaded['metadata']['measurement_layout']
        assert metadata['source_repetition_count'] == 3 and metadata['source_sample_count'] == 7
        assert metadata['data_kind'] == result.data_kind
        expected = result.analysis_iq.mean(axis=(1, 2), keepdims=True) if scalar else result.analysis_iq
        np.testing.assert_array_equal(loaded['iq'], expected)
        dataset.conn.close()
    assert count == (9 if scalar else result.iq.size // 2)


def test_hdf5_name_collision_never_overwrites_and_cancel_is_explicit(tmp_path):
    kwargs = dict(run_config=QcodesRunConfig('', storage_format='hdf5', hdf5_directory=str(tmp_path)),
        connection_config=QickConnectionConfig(), program_summary={}, gui_settings={}, rf_settings={},
        iq_storage_mode='qick_mean_trace', started_at=datetime(2026, 11, 2, tzinfo=timezone.utc))
    result = make_result('qick_mean_trace')
    first, _ = store_qick_result(result, **kwargs)
    original = first.path.read_bytes()
    second, _ = store_qick_result(result, **kwargs)
    assert first.path != second.path and second.path.stem.endswith('_001')
    assert first.path.read_bytes() == original
    def cancel(*unused):
        raise ExperimentCancelled('test stop during write')
    with pytest.raises(ExperimentCancelled):
        store_qick_result(result, **kwargs, progress_callback=cancel)
    partial = next(first.path.parent.glob('*_002.h5'))
    with h5py.File(partial) as h5:
        assert h5.attrs['status'] == 'cancelled'
        assert 'test stop' in h5.attrs['error']
    with pytest.raises(ValueError, match='completed'):
        load_hdf5_result(partial)


@pytest.mark.parametrize('mode', ['qick_mean_trace', 'qick_mean_iq'])
@pytest.mark.parametrize('backend', ['qcodes', 'hdf5'])
def test_malformed_mean_cannot_silently_discard_repetitions(tmp_path, mode, backend):
    result = make_result(mode)
    result = replace(result, iq=np.repeat(result.iq, 2, axis=1))
    with pytest.raises(ValueError, match='Invalid QICK averaged'):
        store_qick_result(result,
            run_config=QcodesRunConfig(str(tmp_path/'test.db'), storage_format=backend, hdf5_directory=str(tmp_path)),
            connection_config=QickConnectionConfig(), program_summary={}, gui_settings={}, rf_settings={},
            iq_storage_mode=mode)
    assert not list(tmp_path.iterdir())


class MeanSoc:
    def __init__(self):
        self.calls = []
    def get_fir_average_capabilities(self):
        return dict(api_version=1, modes=['repetition_mean', 'point_mean'], units='input_code', dtype='float64')
    def arm_ddr4_fir_samples(self, **kwargs):
        self.calls.append(('arm', kwargs))
        return kwargs['n_triggers'] * 16
    def get_ddr4_fir_mean(self, **kwargs):
        self.calls.append(('mean', kwargs))
        count = kwargs['n_points']
        first = (kwargs['start'] - 32) // (16 * kwargs['repetitions'])
        samples = 1 if kwargs['reduce_samples'] else kwargs['n_samples']
        iq = np.empty((count, samples, 2), dtype=float)
        for i in range(count):
            iq[i, :, 0] = (first + i) * 100 + np.arange(samples)
            iq[i, :, 1] = -iq[i, :, 0]
        return dict(iq=iq, data_kind='point_mean' if kwargs['reduce_samples'] else 'repetition_mean',
                    accumulation_repetitions=kwargs['repetitions'], iq_scale_log2=0,
                    source_iq_scale_log2=0, source_iq_component_bits=16,
                    source_samples_per_trace=kwargs['n_samples'])


@pytest.mark.parametrize('mode', ['qick_mean', 'qick_mean_iq'])
def test_program_requests_whole_points_and_preserves_source_counts(monkeypatch, mode):
    sequence = FineTuneSequence(('awg_0',))
    sequence.add_set('capture', (0.,), 300)
    sequence.set_amplitude_sweep('capture', 'awg_0', -.2, .2, 3)
    program = sequence.make_program(_fir_soccfg(fir_rate_profile='50_ksps'), awg_channels=(0,),
        repetitions_per_sweep=3, ddr_readout=DdrFirReadoutConfig(ro_ch=0, samples_per_trigger=10,
            at_segment='capture', margin_input_samples=0, address=128, settle_seconds=0))
    monkeypatch.setattr(program, 'run_rounds', lambda *a, **kw: None)
    soc = MeanSoc()
    progress = []
    result = program.acquire_fir_ddr(soc, repetition_mode=mode, readback_chunk_triggers=4,
                                    readback_progress=lambda *p: progress.append(p))
    assert [k['start'] for name, k in soc.calls if name == 'mean'] == [32, 80, 128]
    assert result.iq.shape == (3, 1, 1 if mode == 'qick_mean_iq' else 10, 2)
    assert result.accumulation_repetitions == 3 and result.source_samples_per_trace == 10
    assert result.iq_scale_log2 == 0 and result.source_iq_component_bits == 16
    assert progress == [(0, 9), (3, 9), (6, 9), (9, 9)]
    # An old board must fail before arming or running.
    old = SimpleNamespace(arm_ddr4_fir_samples=lambda **kw: pytest.fail('armed old board'))
    with pytest.raises(RuntimeError, match='updated board'):
        program.acquire_fir_ddr(old, repetition_mode=mode)


def test_stale_or_invalid_mean_api_is_rejected():
    soc = MeanSoc()
    soc.get_fir_average_capabilities = lambda: dict(api_version=0)
    with pytest.raises(RuntimeError, match='restart'):
        require_board_trace_average(soc)


@pytest.mark.parametrize('mode', IQ_STORAGE_MODES)
def test_run_dispatches_storage_selection_to_acquisition_and_hdf5(tmp_path, monkeypatch, mode):
    import qick_qcodes_experiment as experiment
    from dc_waveform_core import QickDdrReadoutSpec
    calls = []
    def execute(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(summary=lambda: {}), make_result(mode), {}
    monkeypatch.setattr(experiment, 'execute_qick_sequence', execute)
    stored = experiment.run_qick_qcodes_experiment(
        connection_config=QickConnectionConfig(),
        run_config=QcodesRunConfig('', storage_format='hdf5', hdf5_directory=str(tmp_path)),
        sequence=grouped_sequence(3), awg_channels=(0, 1), repetitions_per_sweep=3,
        rf_specs=(), readout_spec=QickDdrReadoutSpec(0, 'capture', 0., 7),
        gui_settings={}, iq_storage_mode=mode,
        connector=lambda **kw: (MeanSoc(), _fir_soccfg(fir_rate_profile='1_msps')))
    assert calls[0]['repetition_mode'] == {'qick_mean_trace': 'qick_mean', 'qick_mean_iq': 'qick_mean_iq'}.get(mode, 'raw')
    assert stored.storage_format == 'hdf5' and stored.database_path.suffix == '.h5'
    assert stored.source_label == stored.database_path.name
    assert stored.detach_dataset().dataset is None
    assert load_hdf5_result(stored.database_path)['metadata']['acquisition']['iq_storage_mode'] == mode


def test_missing_board_api_stops_before_readout_hardware_is_configured(monkeypatch):
    import qick_qcodes_experiment as experiment
    monkeypatch.setattr(experiment, 'configure_rf_readout', lambda *a: pytest.fail('Hardware was configured'))
    with pytest.raises(RuntimeError, match='updated board'):
        experiment.execute_qick_sequence(object(), {}, FineTuneSequence(('awg_0',)),
            awg_channels=(0,), repetitions_per_sweep=2, rf_specs=(), readout_spec=None,
            repetition_mode='qick_mean')


def test_partial_reply_and_cancellation_are_not_treated_as_complete():
    soc = MeanSoc()
    args = dict(n_points=3, repetitions=3, n_samples=10, base_word=32,
        stride_words=16, stride_bytes=None, source_scale_log2=0, source_component_bits=16,
        chunk_triggers=4, cancel_check=lambda: None)
    get = soc.get_ddr4_fir_mean
    soc.get_ddr4_fir_mean = lambda **kw: dict(get(**kw), accumulation_repetitions=2)
    with pytest.raises(RuntimeError, match='metadata'):
        read_board_mean(soc, **args)
    soc.get_ddr4_fir_mean = get
    def cancel():
        if len(soc.calls) >= 2:
            raise ExperimentCancelled('stop')
    with pytest.raises(ExperimentCancelled):
        read_board_mean(soc, **dict(args, cancel_check=cancel))


def test_gui_four_choices_persist_and_old_json_keeps_defaults(tmp_path, monkeypatch):
    import DCWaveform_Generator as gui
    app = gui.QtWidgets.QApplication.instance() or gui.QtWidgets.QApplication([])
    window = gui.MainWindow()
    try:
        panel = window._experiment_panel
        assert [panel.iq_storage_mode.itemData(i) for i in range(panel.iq_storage_mode.count())] == list(IQ_STORAGE_MODES)
        for mode in IQ_STORAGE_MODES:
            panel.set_iq_storage_mode(mode)
            panel.storage_format.setCurrentIndex(panel.storage_format.findData('hdf5'))
            panel.hdf5_directory.setText(str(tmp_path))
            document = window._settings_to_dict()
            assert document['experiment']['storage_format'] == 'hdf5'
            assert document['experiment']['hdf5_directory'] == str(tmp_path)
            decoded = window._decode_settings(document)
            assert decoded['iq_storage_mode'] == mode
            assert decoded['run_config'].storage_format == 'hdf5'
            window._apply_decoded_settings(decoded)
            assert panel.iq_storage_mode.currentData() == mode
            assert panel.storage_format.currentData() == 'hdf5'
            assert panel.hdf5_directory.isEnabled() and not panel.database_path.isEnabled()
        document['version'] = 48
        document['experiment'].pop('storage_format')
        document['experiment'].pop('hdf5_directory')
        document['qick'].pop('iq_storage_mode')
        decoded = window._decode_settings(document)
        assert decoded['iq_storage_mode'] == 'full_traces'
        assert decoded['run_config'].storage_format == 'qcodes'
        window._apply_decoded_settings(decoded)
        assert panel.iq_storage_mode.currentData() == 'full_traces'
        assert panel.database_path.isEnabled() and not panel.hdf5_directory.isEnabled()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
