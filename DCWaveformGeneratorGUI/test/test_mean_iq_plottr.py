"""Unswept means must appear as dependent data in QCoDeS/Plottr viewers."""

from dataclasses import replace
import json

import numpy as np
import pytest
from plottr.data.qcodes_dataset import ds_to_datadicts

from qick_fine_tune_sweep import FineTuneDdrResult, FineTuneSequence
from qick_qcodes_experiment import (
    QcodesRunConfig, QickConnectionConfig, QCODES_STAGING_ENV,
    store_qick_result, load_qick_iq_arrays, build_awg_vertex_metadata,
)


@pytest.mark.parametrize('mode', ['mean_iq', 'qick_mean_iq'])
@pytest.mark.parametrize('expanded_vertices', [False, True])
def test_unswept_mean_is_visible_to_plottr(tmp_path, monkeypatch, mode, expanded_vertices):
    monkeypatch.setenv(QCODES_STAGING_ENV, str(tmp_path/'staging'))
    iq = np.arange(3*64*2, dtype=np.float64).reshape(1, 3, 64, 2)
    expected = iq.mean(axis=(1, 2))
    result = FineTuneDdrResult(iq=iq, sweep_points=np.array([0.]))
    if mode == 'qick_mean_iq':
        result = replace(result, iq=expected.reshape(1, 1, 1, 2),
            data_kind='point_mean', accumulation_repetitions=3, source_samples_per_trace=64,
            source_iq_scale_log2=46, source_iq_component_bits=64)
    settings = {}
    if expanded_vertices:
        sequence = FineTuneSequence(('awg_0',))
        sequence.add_set('capture', (.1,), 300)
        settings['awg_waveform_vertices'] = build_awg_vertex_metadata(sequence, fabric_mhz=300., full_scale_mv=800.)
    dataset, count = store_qick_result(result,
        run_config=QcodesRunConfig(str(tmp_path/'mean.db')),
        connection_config=QickConnectionConfig(), program_summary={},
        gui_settings=settings, rf_settings={}, iq_storage_mode=mode)
    try:
        assert count == 1
        # Previously this returned {}, although the two scalar values existed.
        plots = ds_to_datadicts(dataset)
        for lane, name in enumerate(('i_mean', 'q_mean')):
            assert plots[name].axes(name) == ['point_index']
            assert plots[name].dependents() == [name]
            np.testing.assert_array_equal(plots[name].data_vals('point_index'), [0.])
            np.testing.assert_array_equal(plots[name].data_vals(name), expected[:, lane])
        if expanded_vertices:
            assert plots['awg_0_physical_vertices_mv'].axes('awg_0_physical_vertices_mv') == ['awg_0_vertex_time_us']
        loaded = load_qick_iq_arrays(dataset)
        np.testing.assert_array_equal(loaded['iq'].reshape(1, 2), expected)
        assert loaded['sweep_coordinates'] == {}  # The display index is not a new sweep.
        layout = json.loads(dataset.get_metadata('qick_experiment_json'))['measurement_layout']
        assert layout['point_index_parameter'] == 'point_index'
    finally:
        dataset.conn.close()
