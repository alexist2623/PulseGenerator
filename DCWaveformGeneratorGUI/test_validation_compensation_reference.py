"""Independent arithmetic and explicit scheduling for the v5 DC/RC reference."""
from dataclasses import replace
from copy import deepcopy

import numpy as np
import pytest

from validation.awg_reference import AwgReference, ReferenceUnavailable
from validation.compensation_reference import (
    CompensationReference, CompensationSchedule, area_balance, rc_inverse,
)


def fixture(mode='fixed_time', rc=True):
    spec = dict(outputs=[dict(full_scale_mv=800), dict(full_scale_mv=800)],
                stages=[dict(name='zero', ramp_us=0, hold_us=.01, voltage_mv=[0, 0]),
                        dict(name='main', ramp_us=.01, hold_us=.02, voltage_mv=[100, -100]),
                        dict(name='end', ramp_us=.01, hold_us=.01, voltage_mv=[0, 0])],
                compensation=dict(dc_enabled=True, dc_mode=mode, dc_duration_us=.1,
                                  dc_voltage_mv=200, rc_enabled=rc, rc_tau_us=100))
    reference = AwgReference(spec)
    end = sum(c['hold_cycles'] + c['ramp_samples'] // 16 + bool(c['ramp_samples'])
              for c in reference.commands(0))
    schedule = CompensationSchedule(main_end_cycle=end, capture_end_cycle=end + 100,
        initial_state='zero_area_zero_iir', barrier_policy='explicit', readout_policy='none',
        reset_policy='dc_zero_edge_then_clear', dc_first_cycle=end + 20)
    return reference, schedule


@pytest.mark.parametrize('area', [-131, -1, 0, 1, 131])
def test_area_integer_remainder_sign_and_padding(area):
    timed = area_balance(area, mode='fixed_time', words=2)
    assert len(timed) == 32 and int(timed.sum()) == -area
    q, r = divmod(abs(area), 32)
    sign = -1 if area > 0 else 1
    np.testing.assert_array_equal(timed, [sign*(q+1)]*r + [sign*q]*(32-r))
    voltage = area_balance(area, mode='fixed_voltage', amplitude_code=7)
    assert len(voltage) % 16 == 0 and len(voltage) >= 16
    assert int(voltage.sum()) == -area
    full, remainder = divmod(abs(area), 7)
    np.testing.assert_array_equal(voltage[:full], np.full(full, sign*7))
    if remainder:
        assert voltage[full] == sign*remainder
    assert not np.any(voltage[full+bool(remainder):])


def test_area_bounds_and_no_large_implicit_allocation():
    with pytest.raises(ValueError, match='range'):
        area_balance(-8192 * 16, mode='fixed_time', words=1)
    assert np.all(area_balance(8192 * 16, mode='fixed_time', words=1) == -8192)
    with pytest.raises(ValueError, match='48-bit'):
        area_balance(1 << 47, mode='fixed_voltage', amplitude_code=1)
    with pytest.raises(ValueError, match='allocation'):
        area_balance(1000000, mode='fixed_voltage', amplitude_code=1, max_samples=100)


def test_rc_exact_scalar_trapezoid_and_clear_after_zero_edge():
    # Closed-form accumulated transport areas, including the nonzero->zero edge.
    raw = np.array([100, 100, 0, 0, 0, -100, 0], dtype=np.int16)
    coefficient = 1 << 31
    got, info = rc_inverse(raw, coefficient, reset_samples=(4,))
    weighted_area = [400, 1200, 1600, 1600, 0, -400, -800]
    expected = [((int(v)*4 << 48) + coefficient*a + (1 << 49)) >> 50
                for v, a in zip(raw, weighted_area)]
    np.testing.assert_array_equal(got, expected)
    assert info['reset_states'] == [dict(sample=4, integral_before=1600*coefficient,
                                        previous_transport_before=0)]
    assert info['integral_final'] == -800*coefficient
    assert info['clipped_samples'] == []


def test_rc_zero_input_does_not_erase_history_and_clipping_is_reported():
    raw = np.r_[np.full(256, 8191, dtype=np.int16), np.zeros(32, dtype=np.int16)]
    got, info = rc_inverse(raw, (1 << 32)-1, reset_samples=(272,))
    assert got[256] > 0 and got[271] > 0
    assert not np.any(got[272:]) and info['integral_final'] == 0
    assert info['clipped_samples']


def test_rc_positive_and_negative_half_code_ties_round_toward_positive_infinity():
    # Eight +4096 effective samples followed by zero produce exactly +1/2
    # effective code of trapezoidal integral with coefficient2^31.
    positive = np.r_[np.full(8, 4096, dtype=np.int16), np.zeros(1, dtype=np.int16)]
    plus, info_plus = rc_inverse(positive, 1 << 31)
    minus, info_minus = rc_inverse(-positive, 1 << 31)
    assert info_plus['integral_final'] == 1 << 49
    assert info_minus['integral_final'] == -(1 << 49)
    assert plus[-1] == 1 and minus[-1] == 0


@pytest.mark.parametrize('mode', ['fixed_time', 'fixed_voltage'])
@pytest.mark.parametrize('rc', [False, True])
def test_compensated_waveform_completes_with_exact_area_and_reset(mode, rc):
    reference, schedule = fixture(mode, rc)
    original = deepcopy(reference.spec)
    output = CompensationReference(reference).waveforms(0, schedule)
    assert output['full_reference_coverage'] and output['physical_reference_required']
    assert reference.spec == original
    assert output['pre_rc_codes'].shape == (schedule.capture_end_cycle * 16, 2)
    assert output['requested_virtual_v'].shape == output['pre_rc_codes'].shape
    assert not np.any(output['requested_virtual_v'][schedule.main_end_cycle * 16:])
    assert not np.any(output['requested_physical_v'][schedule.main_end_cycle * 16:])
    np.testing.assert_array_equal(output['pre_rc_codes'].astype(np.int64).sum(axis=0), [0, 0])
    for ch, detail in enumerate(output['channels']):
        assert detail['residual_area'] == 0
        assert detail['reset_cycle'] == detail['dc_end_cycle'] + 1
        assert not np.any(output['quantized_codes'][detail['reset_cycle']*16:, ch])
        if rc:
            assert detail['rc']['coefficient'] == 293203101
            assert detail['rc']['integral_final'] == 0


def test_schedule_unknown_and_incomplete_are_unavailable_not_truncated_pass():
    reference, schedule = fixture()
    model = CompensationReference(reference)
    with pytest.raises(ReferenceUnavailable, match='explicit'):
        model.waveforms(0, None)
    for field, value in [('initial_state', 'unknown'), ('barrier_policy', 'fit_edge'),
                         ('readout_policy', 'overlap'), ('reset_policy', 'SET_zero'),
                         ('dc_first_cycle', None)]:
        with pytest.raises(ReferenceUnavailable):
            model.waveforms(0, replace(schedule, **{field: value}))
    with pytest.raises(ValueError, match='complete main'):
        model.waveforms(0, replace(schedule, main_end_cycle=schedule.main_end_cycle-1))
    full = model.waveforms(0, schedule)
    with pytest.raises(ReferenceUnavailable, match='cleared word'):
        model.waveforms(0, replace(schedule, capture_end_cycle=full['channels'][0]['reset_cycle']))


def test_explicit_rc_only_clear_and_rf_not_silently_ignored():
    reference, schedule = fixture()
    reference.spec['compensation']['dc_enabled'] = False
    rc_schedule = replace(schedule, dc_first_cycle=None, reset_policy='explicit_rc_clear',
                          reset_cycles=schedule.main_end_cycle + 10)
    output = CompensationReference(reference).waveforms(0, rc_schedule)
    assert output['channels'][0]['main_area'] != 0
    assert not np.any(output['quantized_codes'][(schedule.main_end_cycle+10)*16:])
    with pytest.raises(ReferenceUnavailable, match='both outputs'):
        CompensationReference(reference).waveforms(0, replace(rc_schedule, reset_cycles=None))
    reference.spec['rf_pulses'] = [{'frequency_mhz': 100}]
    with pytest.raises(ReferenceUnavailable, match='RF composition'):
        CompensationReference(reference)


def test_common_basic_case_schema_is_used_without_silent_defaults():
    from validation.awg_cases import basic_awg_cases
    outputs = [dict(generator=i, full_scale_mv=800, scope_channel=str(i+1),
                    physical_connector=f'DAC{i+4}') for i in range(2)]
    case = next(c for c in basic_awg_cases(outputs) if c['name']=='matrix_dc1_rc1_fixed_voltage')
    reference = AwgReference(case)
    point = 0
    commands = reference.commands(point)
    end = sum(c['hold_cycles'] + c['ramp_samples']//16 + bool(c['ramp_samples']) for c in commands)
    schedule = CompensationSchedule(main_end_cycle=end, capture_end_cycle=end+20000,
        initial_state='zero_area_zero_iir', barrier_policy='explicit', readout_policy='none',
        reset_policy='dc_zero_edge_then_clear', dc_first_cycle=end+100)
    result = CompensationReference(reference).waveforms(point, schedule)
    assert result['channels'][0]['amplitude_code'] == 1024
    assert result['channels'][0]['rc']['coefficient'] == round((1 << 48)/(2*30*4800))
