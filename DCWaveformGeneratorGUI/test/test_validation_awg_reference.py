"""Independent case specification versus decoded v5 command-register models."""
from copy import deepcopy
import numpy as np
import pytest
import qick.sim  # noqa: F401
from qick_dc_area import executed_commands
from test_execution_coordinates import awg_program
from validation.awg_reference import AwgReference, ReferenceUnavailable, physical_references


def specification():
    return dict(name='matrix_grouped', outputs=[dict(scope_channel='1', generator=0,
        physical_connector='DAC4', full_scale_mv=800.), dict(scope_channel='2', generator=1,
        physical_connector='DAC5', full_scale_mv=900.)],
        matrix=[[1., .2], [-.1, 1.]], stages=[
            dict(name='zero', ramp_us=0., hold_us=.1, voltage_mv=[0., 0.]),
            dict(name='hold', ramp_us=.01, hold_us=1., voltage_mv=[80., 160.]),
            dict(name='reset', ramp_us=.01, hold_us=.1, voltage_mv=[0., 0.])],
        axes=[dict(kind='ramp_duration', output=0, segment=1, start=.01, stop=1., count=3, group='time'),
              dict(kind='ramp_duration', output=0, segment=2, start=.02, stop=.5, count=3, group='time'),
              dict(kind='amplitude', output=0, segment=1, start=-160., stop=240., count=3, group='voltage'),
              dict(kind='amplitude', output=1, segment=1, start=80., stop=-80., count=3, group='voltage')])


def test_independent_integer_model_matches_v5_matrix_groups():
    reference = AwgReference(specification())
    program = awg_program()
    assert reference.shape == (3, 3)
    for point in range(9):
        commands = list(executed_commands(program, point))
        expected = reference.commands(point)
        for stage, actual_name in enumerate(('zero', 'hold', 'reset')):
            for output in range(2):
                actual = next(c for c in commands if c.segment_name == actual_name and c.output_index == output)
                assert actual.target_code >> 2 == expected[stage]['target_codes'][output]
        for stage, actual_name in ((1, 'rise'), (2, 'fall')):
            for output in range(2):
                actual = next(c for c in commands if c.segment_name == actual_name and c.output_index == output)
                assert actual.step == expected[stage]['steps'][output]
                assert actual.duration_samples == expected[stage]['ramp_samples']
        waveform = reference.waveforms(point)
        assert waveform['quantized_codes'].shape[1] == 2
        assert np.all(np.isfinite(waveform['requested_virtual_v']))
        np.testing.assert_allclose(waveform['requested_physical_v'],
            waveform['requested_virtual_v'] @ np.array(specification()['matrix']).T)
        assert waveform['physical_reference_required']


def test_exact_ramp_last_sample_and_guard_then_hold():
    spec = specification()
    spec['axes'] = []
    reference = AwgReference(spec)
    expected = reference.commands(0)
    waveform = reference.waveforms(0)
    start = expected[0]['hold_cycles'] * 16
    count = expected[1]['ramp_samples']
    target = expected[1]['target_codes']
    np.testing.assert_array_equal(waveform['quantized_codes'][start], [0, 0])
    np.testing.assert_array_equal(waveform['quantized_codes'][start + count - 1], target)
    np.testing.assert_array_equal(waveform['quantized_codes'][start + count:start + count + 16],
                                  np.tile(target, (16, 1)))
    assert waveform['time_s'][1] == pytest.approx(1 / 4.8e9)


@pytest.mark.parametrize('change', [dict(dc_enabled=True), dict(rc_enabled=True)])
def test_compensation_never_silently_gets_identity_reference(change):
    spec = specification()
    spec['compensation'] = change
    with pytest.raises(ReferenceUnavailable, match='compensation/circuit'):
        AwgReference(spec).waveforms(0)


@pytest.mark.parametrize('key', ['rf', 'rf_pulses'])
def test_rf_composition_cannot_silently_use_baseband_reference(key):
    spec = specification()
    spec[key] = [dict(name='carrier')]
    with pytest.raises(ReferenceUnavailable, match='RF requires'):
        AwgReference(spec).waveforms(0)


def test_counts_member_ranges_quantization_and_limits():
    spec = specification()
    spec['axes'][0].update(start=1., stop=.01, count=100)
    spec['axes'][1].update(start=.5, stop=.02, count=100)
    reference = AwgReference(spec)
    assert reference.point_count == 300
    assert reference.coordinates(299) == (99, 2)
    before = deepcopy(spec)
    reference.commands(299)
    assert before == spec
    with pytest.raises(ValueError, match='allocation'):
        reference.waveforms(0, max_samples=1)
    with pytest.raises(ValueError, match='outside'):
        reference.commands(-1)
    spec['axes'][1]['count'] = 2
    with pytest.raises(ValueError, match='share'):
        AwgReference(spec)


def test_fixed_calibration_and_highpass_step_are_independent_of_capture():
    time = np.arange(1000) * 1e-8
    waveform = dict(time_s=time, quantized_physical_v=np.ones((1000, 2)) * .3,
                    time_origin='declared SET edge', model='test-step-v1')
    flat = dict(kind='flat_dc', gain=1.1, offset_v=.02, delay_s=2e-9,
                calibration_evidence={'id': 'fixed-fixture-measurement'})
    highpass = dict(kind='first_order_highpass', gain=1., offset_v=0., delay_s=0.,
                    tau_s=1e-5, initial_state='settled_zero',
                    calibration_evidence={'id': 'fixed-fixture-measurement'})
    first, second = physical_references(waveform, [flat, highpass])
    np.testing.assert_allclose(first.voltage_v, .35)
    np.testing.assert_allclose(first.time_s, time + 2e-9)
    np.testing.assert_allclose(second.voltage_v, .3 * np.exp(-time / 1e-5), rtol=1e-12)
    first.arrays()
    second.arrays()
    with pytest.raises(ReferenceUnavailable, match='calibration evidence'):
        physical_references(waveform, [{**flat, 'calibration_evidence': None}, highpass])
