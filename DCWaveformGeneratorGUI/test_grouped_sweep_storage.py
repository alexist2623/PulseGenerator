"""Persist correlated coordinates, group identities and exact shared indices."""
import json

import numpy as np
import pytest
from qick.sim import QickSim  # noqa: F401
from qick_fine_tune_sweep import FineTuneDdrResult
from qick_qcodes_experiment import (
    QcodesRunConfig, QickConnectionConfig, QCODES_STAGING_ENV,
    store_qick_result, load_qick_iq_arrays, build_awg_vertex_metadata,
    build_awg_waveform_recipe,
)
from test_grouped_sweep_compiler import grouped_sequence


@pytest.mark.parametrize('storage_mode', ['full_traces', 'mean_iq'])
def test_group_coordinates_survive_qcodes_roundtrip(tmp_path, monkeypatch, storage_mode):
    monkeypatch.setenv(QCODES_STAGING_ENV, str(tmp_path / 'staging'))
    seq = grouped_sequence(3)
    seq.set_cross_capacitance(((1., .2), (-.1, 1.)))
    seq.sweep_parameter_ids = {seq.sweep_axis_key(a): f'parameter-{i}'
                               for i, a in enumerate(seq.sweep_axes)}
    iq = np.arange(9 * 2 * 4 * 2, dtype=np.int32).reshape(9, 2, 4, 2)
    result = FineTuneDdrResult(sweep_points=seq.sweep_coordinates, iq=iq,
        sweep_axes=tuple(seq.sweep_axes), sweep_shape=seq.sweep_shape,
        sweep_groups=seq.sweep_groups, sweep_parameter_ids=seq.sweep_member_ids,
        cross_capacitance=seq.cross_capacitance)
    settings = {'qick': {'full_scale_mv': 800., 'fabric_mhz': 300.,
                        'iq_storage_mode': storage_mode},
                'awg': {'cross_capacitance': seq.cross_capacitance.tolist()},
                'awg_vertex_data': build_awg_vertex_metadata(seq, fabric_mhz=300., full_scale_mv=800.),
                'awg_waveform_recipe': build_awg_waveform_recipe(seq, fabric_mhz=300., full_scale_mv=800.)}
    dataset, _ = store_qick_result(result,
        run_config=QcodesRunConfig(str(tmp_path / 'grouped.db'), experiment_name='grouped', sample_name='offline'),
        connection_config=QickConnectionConfig('192.0.2.10'),
        program_summary={}, gui_settings=settings, rf_settings={}, iq_storage_mode=storage_mode)
    loaded = load_qick_iq_arrays(dataset)
    assert loaded['sweep_shape'] == (3, 3)
    assert len(loaded['sweep_coordinates']) == 4
    repetitions = loaded['iq'].shape[1]
    for index, expected in enumerate((np.repeat(np.arange(3), 3), np.tile(np.arange(3), 3))):
        np.testing.assert_array_equal(loaded['sweep_group_indices'][index],
                                     np.repeat(expected[:, None], repetitions, axis=1))
    layout = loaded['metadata']['measurement_layout']
    assert layout['sweep_parameter_ids'] == [f'parameter-{i}' for i in range(4)]
    assert [a['group_id'] for a in layout['sweep_axes']] == ['time', 'time', 'voltage', 'voltage']
    assert [a['loop_axis_index'] for a in layout['sweep_axes']] == [0, 0, 1, 1]
    actual = np.column_stack([loaded['sweep_coordinates'][a['parameter']][:, 0]
                              for a in layout['sweep_axes']])
    np.testing.assert_allclose(actual, seq.sweep_coordinates * [1., 1., 800., 800.])
    np.testing.assert_array_equal(loaded['iq'], iq if storage_mode == 'full_traces'
                                 else iq.mean(axis=(1, 2), keepdims=True))
    assert json.loads(dataset.get_metadata('cross_capacitance_json')) == [[1., .2], [-.1, 1.]]
    recipe = settings['awg_waveform_recipe']
    assert recipe['sweep_shape'] == [3, 3]
    assert len(recipe['sweep_groups']) == 2
    from awg_sweep_map import awg_sweep_result_from_stored_arrays
    chart = awg_sweep_result_from_stored_arrays(loaded, database_path=tmp_path / 'grouped.db', run_id=1)
    assert chart.i_mean.shape == (3, 3)
    assert chart.averaged_axis_labels == ()


def test_grouped_map_uses_two_independent_indices_and_preserves_reload():
    from awg_sweep_map import reduce_awg_sweep_map, reduce_awg_sweep_source, sweep_axis_key
    seq = grouped_sequence(3)
    iq = np.arange(18, dtype=np.int32).reshape(9, 1, 1, 2)
    result = FineTuneDdrResult(sweep_points=seq.sweep_coordinates, iq=iq,
        sweep_axes=tuple(seq.sweep_axes), sweep_shape=seq.sweep_shape,
        sweep_groups=seq.sweep_groups)
    keys = [sweep_axis_key(axis) for axis in seq.sweep_axes]
    with pytest.raises(ValueError, match='independent sweep groups'):
        reduce_awg_sweep_map(result, x_axis_key=keys[0], y_axis_key=keys[1], full_scale_mv=800.)
    chart = reduce_awg_sweep_map(result, x_axis_key=keys[0], y_axis_key=keys[2], full_scale_mv=800.)
    assert chart.i_mean.shape == (3, 3)
    assert chart.averaged_axis_labels == ()
    assert chart.source_points_per_cell == 1
    np.testing.assert_array_equal(chart.i_mean, iq[:, 0, 0, 0].reshape(3, 3).T)
    restored = reduce_awg_sweep_source(chart.source, x_axis_key=keys[1], y_axis_key=keys[3])
    np.testing.assert_array_equal(restored.i_mean, chart.i_mean[::-1])
    with pytest.raises(ValueError, match='cannot also be fixed'):
        reduce_awg_sweep_source(chart.source, x_axis_key=keys[0], y_axis_key=keys[2],
                               fixed_axis_values={keys[1]: .01})


@pytest.mark.parametrize('software_axis_count', [1, 2])
def test_software_prefix_keeps_recipe_and_saved_group_indices_identical(
        tmp_path, monkeypatch, software_axis_count):
    """Only the instrument boundary is replaced; recipe and QCoDeS are real."""
    from types import SimpleNamespace
    import qick_qcodes_experiment as experiment
    from dc_waveform_core import QickDdrReadoutSpec, QickRfSoftwareSweepAxisSpec
    from test_output_sum_v5_gui import configuration

    monkeypatch.setenv(QCODES_STAGING_ENV, str(tmp_path / 'staging'))
    seq = grouped_sequence(3)
    seq.sweep_parameter_ids = {seq.sweep_axis_key(a): f'member-{i}'
                               for i, a in enumerate(seq.sweep_axes)}
    hardware = FineTuneDdrResult(sweep_points=seq.sweep_coordinates,
        iq=np.arange(9 * 1 * 2 * 2, dtype=np.int32).reshape(9, 1, 2, 2),
        sweep_axes=tuple(seq.sweep_axes), sweep_shape=seq.sweep_shape,
        sweep_groups=seq.sweep_groups, sweep_parameter_ids=seq.sweep_member_ids)
    axes = (
        QickRfSoftwareSweepAxisSpec('gate', 'rf_0', 0, 0., 90., 2,
                                    'rf_phase', 'deg', 'phase'),
        QickRfSoftwareSweepAxisSpec('gate', 'rf_0', 0, 1000., 2000., 2,
                                    'rf_amplitude', 'gain', 'amplitude'),
    )[:software_axis_count]
    bindings = tuple((0, axis) for axis in axes)
    coordinates = experiment._software_sweep_coordinates(bindings)
    combined = experiment._combine_software_sweep_results(
        [hardware] * len(coordinates), bindings, coordinates)
    cfg = configuration()
    cfg['ddr4_buf'] = dict(sample_capture=True, fir_enabled=True,
        fir_rate_profile='1_msps', stored_sample_rate_hz=1_000_000.,
        fir_decimation=300, fir_input_fs_mhz=300.,
        fir_group_delay_input_samples=8677.)
    monkeypatch.setattr(experiment, 'connect_qick', lambda *a, **k: (object(), cfg))
    monkeypatch.setattr(experiment, 'execute_qick_sequence',
        lambda *a, **k: (SimpleNamespace(summary=lambda: {}), combined, {}))
    stored = experiment.run_qick_qcodes_experiment(
        connection_config=QickConnectionConfig('192.0.2.10'),
        run_config=QcodesRunConfig(str(tmp_path / 'software_groups.db')),
        sequence=seq, awg_channels=(1,), repetitions_per_sweep=1,
        rf_specs=(SimpleNamespace(software_sweep_axes=axes),),
        readout_spec=QickDdrReadoutSpec(0, 'gate', 0., 2),
        gui_settings={'qick': {'fabric_mhz': 300., 'full_scale_mv': 800.}},
        iq_storage_mode='full_traces')
    loaded = load_qick_iq_arrays(stored.dataset)
    recipe = loaded['metadata']['gui_settings']['awg_waveform_recipe']
    layout = loaded['metadata']['measurement_layout']
    assert recipe['sweep_shape'] == [2] * software_axis_count + [3, 3]
    assert recipe['point_count'] == 9 * 2 ** software_axis_count
    assert recipe['sweep_groups'] == layout['sweep_groups']
    assert recipe['sweep_parameter_ids'] == layout['sweep_parameter_ids']
    assert recipe['sweep_groups'][0]['member_indices'] == [software_axis_count,
                                                         software_axis_count + 1]
    assert recipe['sweep_groups'][0]['loop_axis_index'] == software_axis_count
    assert recipe['sweep_parameter_ids'] == [''] * software_axis_count + [
        f'member-{i}' for i in range(4)]
    stored.detach_dataset()
