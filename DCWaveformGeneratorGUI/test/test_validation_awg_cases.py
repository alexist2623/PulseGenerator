"""Concrete GUI case preparation and references; no physical execution."""
from copy import deepcopy
import json

import numpy as np
import pytest

from test_mainwindow_case_bindings import window, case, fixture  # noqa: F401
from test_hardware_validation import identities  # noqa: F401
from validation.archive import canonical, digest
from validation.awg_cases import basic_awg_cases, prepare_awg_case, declared_compensation_builder
from validation.awg_reference import AwgReference, ReferenceUnavailable


def paths():
    return [dict(kind='flat_dc', gain=1., offset_v=0., delay_s=0.,
                 calibration_evidence={'method': 'offline test fixture, not measured'}) for _ in range(2)]


def test_planned_cases_have_individual_logs_and_separate_arithmetic_loops():
    plan = basic_awg_cases(case()['outputs'], point_count=3)
    assert len({entry['name'] for entry in plan}) == len(plan) == 33
    assert plan[0]['adapter_status'] == 'unavailable'
    for kind in ('hold', 'ramp'):
        logs = [entry for entry in plan if entry['name'].startswith(kind+'_log_')]
        assert len(logs) == 10
        assert [entry['nominal_duration_us'] for entry in logs][::9] == [.1, 100.]
        assert all(entry['execution'] == 'individual_gui_log_grid' and not entry['axes'] for entry in logs)
        loops = [entry for entry in plan if entry['name'].startswith(kind+'_arithmetic_')]
        assert len(loops) == 2
        assert all(AwgReference(entry).point_count == 3 and entry['repetitions'] == 2 for entry in loops)
    grouped = next(entry for entry in plan if entry['name'] == 'matrix_grouped_voltage')
    assert AwgReference(grouped).point_count == 3
    combined = next(entry for entry in plan if entry['category'] == 'combined_axes')
    assert AwgReference(combined).point_count == 9
    one = next(entry for entry in plan if entry['category'] == 'repetition_reset')
    assert AwgReference(one).point_count == 1 and one['repetitions'] == 3
    with pytest.raises(ValueError, match='DAC scale'):
        basic_awg_cases(case()['outputs'], sweep_limit_mv=1000.)


def test_actual_gui_preparation_seals_all_reference_planes_without_running(window, identities, tmp_path):
    _, identity = identities
    spec = case()
    spec['repetitions'] = 1
    for stage in spec['stages']:
        stage['hold_us'] = .1
        if stage['ramp_us']:
            stage['ramp_us'] = .01
    spec['axes'].append(dict(kind='hold_duration', output=0, segment=1, start=.1, stop=.2, count=2))
    original = deepcopy(spec)
    prepared = prepare_awg_case(window=window, fixture=fixture(), identity=identity,
        case=spec, output_models=paths(), root=tmp_path/'private')
    target = prepared.archive
    assert spec == original
    assert prepared.reference.shape == (2, 3)  # GUI duration-first normalized order.
    assert target.case_id == digest(canonical(spec))[:20]
    assert target.sealed and not target.finalized
    assert len(target.expected) == 6*2
    assert all(set(row['references']) == {'requested', 'quantized', 'physical'} for row in target.expected.values())
    assert len(json.loads((target.root/'qt_preparation_events.json').read_text())) > 20
    assert json.loads((target.root/'gui_requested.json').read_text()) == prepared.binding.snapshot()
    assert getattr(window, '_experiment_thread', None) is None
    with np.load(target.root/'independent'/'p00000005.npz', allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays['requested_physical_v'],
            arrays['requested_virtual_v'] @ np.asarray(spec['matrix']).T)
        assert arrays['quantized_codes'].shape[1] == 2
    assert not any(event['name'] == 'worker_started' for event in target.events)


def test_missing_analog_model_is_preserved_as_unsealed_preparation_failure(window, identities, tmp_path):
    spec = case()
    for stage in spec['stages']:
        stage['hold_us'] = .1
        if stage['ramp_us']:
            stage['ramp_us'] = .01
    with pytest.raises(ReferenceUnavailable, match='calibration evidence'):
        prepare_awg_case(window=window, fixture=fixture(), identity=identities[1],
            case=spec, output_models=[{}, {}], root=tmp_path/'private')
    events = list((tmp_path/'private').rglob('events/*.json'))
    assert len(events) == 1
    assert json.loads(events[0].read_text())['name'] == 'preparation_failed'
    assert not list((tmp_path/'private').rglob('gui_requested.json'))
    assert getattr(window, '_experiment_thread', None) is None


def test_declared_dc_rc_case_retains_area_reset_and_all_reference_planes(window, identities, tmp_path):
    from validation.compensation_reference import CompensationSchedule
    from validation.mainwindow_cases import validate_case
    spec = case()
    spec['axes'] = []
    spec['repetitions'] = 1
    for stage in spec['stages']:
        stage['hold_us'] = .1
        if stage['ramp_us']:
            stage['ramp_us'] = .01
    spec['compensation'] = dict(dc_enabled=True, dc_mode='fixed_time', dc_duration_us=.3,
                                rc_enabled=True, rc_tau_us=30.)
    schedule = CompensationSchedule(main_end_cycle=98, dc_first_cycle=110,
        capture_end_cycle=202, initial_state='zero_area_zero_iir', barrier_policy='explicit',
        readout_policy='none', reset_policy='dc_zero_edge_then_clear')
    normalized = validate_case(spec, fixture())
    prepared = prepare_awg_case(window=window, fixture=fixture(), identity=identities[1],
        case=spec, output_models=paths(), root=tmp_path/'private',
        waveform_builder=declared_compensation_builder([schedule], normalized_case=normalized))
    assert prepared.archive.sealed
    with np.load(prepared.archive.root/'independent'/'p00000000.npz') as arrays:
        np.testing.assert_array_equal(arrays['pre_rc_codes'].astype(np.int64).sum(axis=0), [0, 0])
        np.testing.assert_array_equal(arrays['quantized_codes'][-16:], np.zeros((16, 2)))
        assert len(arrays['time_s']) == 202*16
    details = json.loads((prepared.archive.root/'independent'/'p00000000.json').read_text())
    assert details['schedule']['dc_first_cycle'] == 110
    assert details['requested_reference_note']
    assert getattr(window, '_experiment_thread', None) is None
    with pytest.raises(ValueError, match='count differs'):
        declared_compensation_builder([schedule, schedule], normalized_case=normalized)(prepared.reference, 0, 10000)
    altered = deepcopy(normalized)
    altered['stages'][0]['hold_us'] = 1.
    with pytest.raises(ValueError, match='different normalized case'):
        declared_compensation_builder([schedule], normalized_case=altered)(prepared.reference, 0, 10000)
