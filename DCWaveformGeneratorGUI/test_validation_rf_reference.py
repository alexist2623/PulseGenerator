"""Independent RF expectations; no instruments, GUI Run, or physical claims."""
import math

import numpy as np
import pytest

from validation.rf_reference import (RfGrid, boundary_case_plan, duration_layout,
    frequency_case_plan, frequency_measurement, interval_acceptance,
    physical_reference, rf_point, rf_waveform)


def point(samples=97, **kwargs):
    args = dict(duration_s=samples/4.8e9, frequency_hz=10e6,
                output_start_cycle=331, anchor_end_cycle=1000,
                final_user_end_cycle=1300, require_within_segment=True,
                output_latency_cycles=31, command_time_plane='native_generator_input')
    args.update(kwargs)
    return rf_point(**args)


@pytest.mark.parametrize('samples,mode,busy,tail', [
    (1, 'short', 3, 0), (16, 'short', 3, 0), (17, 'short', 3, 0),
    (47, 'short', 3, 0), (48, 'short', 3, 0), (49, 'short', 4, 0),
    (95, 'short', 6, 0), (96, 'plateau_tail', 6, 48),
    (97, 'plateau_tail', 7, 48),
    (16*65538+15, 'plateau_tail', 65539, 65535*16),
    (16*65539, 'periodic_tail', 65539, 65535*16),
    (16*65541+15, 'periodic_tail', 65542, 65538*16),
])
def test_known_sample_boundaries_separate_activity_from_occupancy(samples, mode, busy, tail):
    result = duration_layout(samples)
    assert result == dict(samples=samples, remainder=samples % 16,
                         busy_cycles=busy, mode=mode, tail_start_samples=tail)


def test_serialized_sample_lengths_and_fractional_requests_have_distinct_policy():
    grid = RfGrid()
    for samples in (1, 15, 16, 17, 95, 96, 97, 4800):
        duration = round(samples/4800., 9)*1e-6
        assert grid.samples(duration) == samples
    assert grid.samples(17.2/4.8e9) == 18
    assert grid.samples(17.0001/4.8e9) == 18
    assert RfGrid(duration_input_quantum_s=0).samples(17.0001/4.8e9) == 18
    assert grid.samples(11e-9, fine=False) == 64
    assert grid.samples((3+1e-13)/300e6, fine=False) == 48
    assert grid.samples((3+1e-10)/300e6, fine=False) == 64
    with pytest.raises(ValueError, match='three fabric'):
        duration_layout(32, fine=False)


def test_frequency_grid_is_signed_half_even_and_declares_common_readout_step():
    grid = RfGrid(dds_hz=128., dds_bits=8, frequency_step_words=3)
    assert grid.frequency(2.25)['signed_word'] == 6
    assert grid.frequency(.75)['signed_word'] == 0
    assert grid.frequency(-2.25)['signed_word'] == -6
    assert grid.frequency(-2.25)['word'] == 250
    assert grid.frequency(-2.25)['quantized_hz'] == -3.
    with pytest.raises(ValueError, match='Nyquist'):
        grid.frequency(64.)


def test_declared_frequency_case_keeps_descending_and_n1_requested_coordinates():
    descending = frequency_case_plan(20e6, 10e6, 3, execution='individual_gui')
    assert [p['requested_hz'] for p in descending['points']] == [20e6, 15e6, 10e6]
    single = frequency_case_plan(20e6, 10e6, 1, execution='hardware_loop')
    assert [p['requested_hz'] for p in single['points']] == [20e6]
    assert single['physical_execution'] is False
    with pytest.raises(ValueError, match='execution'):
        frequency_case_plan(10e6, 20e6, 2, execution='software_or_hardware_unknown')


def test_one_active_sample_can_be_inside_while_busy_interval_is_rejected():
    result = point(1, output_start_cycle=998)
    assert result['active_end_s'] < 1000/300e6
    assert result['busy_end_cycle'] == 1001
    assert result['boundary_status'] == 'rejected'
    assert result['command_start_cycle'] == 967
    assert result['active_duration_s'] == 1/4.8e9
    assert result['busy_duration_s'] == 10e-9


def test_command_plane_latency_is_explicit_and_does_not_shift_physical_reference():
    # 44 is an illustrative supplied path value, not a claimed RF latency.
    native = point()
    observed = point(output_latency_cycles=44, command_time_plane='tprocessor_model_event')
    assert native['command_start_cycle']-observed['command_start_cycle'] == 13
    assert native['output_start_s'] == observed['output_start_s']
    assert native['active_end_s'] == observed['active_end_s']
    assert native['command_time_plane'] == 'native_generator_input'
    with pytest.raises(TypeError, match='output_latency_cycles'):
        rf_point(duration_s=1e-9, frequency_hz=0, output_start_cycle=31,
                 anchor_end_cycle=50, final_user_end_cycle=70, require_within_segment=True)


def test_later_point_violation_and_false_policy_do_not_extend_hold():
    rows = {row['name']: row for row in boundary_case_plan()}
    assert rows['later_point_violation_true']['expected_boundary_statuses'] == ['eligible', 'eligible', 'rejected']
    assert rows['later_point_violation_false']['expected_boundary_statuses'] == ['eligible']*3
    false = rows['beyond_final_false']['points'][-1]
    assert false['boundary_relation'] == 'beyond_final'
    assert false['required_point_end_cycle'] > false['final_user_end_cycle']
    assert false['anchor_end_cycle'] == 91
    assert 'shared_port_spacing' in false['additional_validation_required']


def test_reference_preserves_carrier_phase_and_zeros_after_active_not_busy_end():
    result = point(97, frequency_hz=300e6)
    start = result['output_start_s']
    times = start + np.arange(-1, 113)/4.8e9
    ref = rf_waveform(result, times, amplitude_v=.2, phase_degrees=90., phase_epoch_s=0.)
    mask = (times >= start) & (times < start+97/4.8e9)
    expected = np.where(mask, .2*np.cos(2*np.pi*300e6*times+np.pi/2), 0.)
    np.testing.assert_allclose(ref.voltage_v, expected, atol=1e-13)
    assert np.all(ref.voltage_v[times >= start+97/4.8e9] == 0)
    assert result['busy_end_s'] > result['active_end_s']
    ref.arrays()


def test_requested_reference_does_not_replace_requested_duration_with_padding():
    result = point(1, duration_s=1.2/4.8e9, frequency_hz=0.)
    start = result['output_start_s']
    times = start + np.arange(30)/4.8e10
    requested = rf_waveform(result, times, amplitude_v=.3, phase_degrees=0,
                            phase_epoch_s=0, requested_start_s=start, quantized=False)
    quantized = rf_waveform(result, times, amplitude_v=.3, phase_degrees=0,
                            phase_epoch_s=0)
    assert requested.metadata['active_end_s'] < quantized.metadata['active_end_s']
    assert np.count_nonzero(requested.voltage_v) < np.count_nonzero(quantized.voltage_v)


def test_known_lowpass_step_matches_analytic_exponential_without_data_fit():
    result = point(48000, frequency_hz=0, anchor_end_cycle=10000, final_user_end_cycle=10000)
    times = result['output_start_s'] + np.arange(1000)*1e-9
    source = rf_waveform(result, times, amplitude_v=.2, phase_degrees=0, phase_epoch_s=0.)
    model = dict(kind='first_order_lowpass_zoh', gain=2., offset_v=.01, delay_s=3e-9,
        cutoff_hz=5e6, initial_output_v=0., calibration_evidence='fixture-test-only',
        reference_plane='scope input', valid_band_hz=[0., 20e6],
        transient_model_evidence='analytic synthetic step fixture, not physical calibration',
        voltage_model_uncertainty_v=0., timing_model_uncertainty_s=0.)
    ref = physical_reference(source, model)
    np.testing.assert_allclose(ref.voltage_v, .4*(1-np.exp(-2*np.pi*5e6*(times-times[0])))+.01,
                               atol=2e-15)
    np.testing.assert_allclose(ref.time_s, times+3e-9, atol=0)
    assert ref.metadata['expectation'] == 'physical'
    assert ref.metadata['input_active_start_s_at_reference_plane'] == source.metadata['active_start_s']+3e-9
    assert ref.metadata['input_active_end_s_at_reference_plane'] == source.metadata['active_end_s']+3e-9
    assert 'active_end_s' not in ref.metadata
    assert ref.metadata['output_support_end_s'] is None
    assert ref.metadata['source_metadata'] == source.metadata
    with pytest.raises(ValueError, match='calibration'):
        physical_reference(source, dict(model, calibration_evidence=''))


def test_flat_model_cannot_silently_predict_outside_its_calibrated_band():
    result = point(960, frequency_hz=10e6)
    times = result['output_start_s'] + np.arange(100)*1e-9
    source = rf_waveform(result, times, amplitude_v=.1, phase_degrees=0, phase_epoch_s=0.)
    model = dict(kind='flat', gain=.5, offset_v=.02, delay_s=1e-9,
        calibration_evidence='fixture-test-only', reference_plane='scope input', valid_band_hz=[0, 5e6],
        transient_model_evidence='analytic synthetic fixture, not physical calibration',
        voltage_model_uncertainty_v=0., timing_model_uncertainty_s=0.)
    with pytest.raises(ValueError, match='calibrated'):
        physical_reference(source, model)
    out = physical_reference(source, dict(model, valid_band_hz=[0, 20e6]))
    np.testing.assert_array_equal(out.voltage_v, source.voltage_v*.5+.02)
    assert out.metadata['active_start_s'] == source.metadata['active_start_s']+1e-9
    assert out.metadata['active_end_s'] == source.metadata['active_end_s']+1e-9
    assert out.metadata['busy_end_s'] == source.metadata['busy_end_s']+1e-9
    assert out.metadata['phase_epoch_s'] == source.metadata['phase_epoch_s']+1e-9
    with pytest.raises(ValueError, match='transient model'):
        physical_reference(source, dict(model, transient_model_evidence=''))


@pytest.mark.parametrize('measurement,uncertainty,status', [
    (1., .01, 'passed'), (1.07, .01, 'failed'),
    (1.04, .02, 'inconclusive'), (1.05, .0, 'failed'),
])
def test_uncertainty_interval_must_fully_pass_the_predeclared_limit(measurement, uncertainty, status):
    assert interval_acceptance(measurement, 1., uncertainty=uncertainty,
                               tolerance=.05)['status'] == status


def test_unresolved_one_sample_width_is_not_passed_or_replaced_by_busy_time():
    expected = point(1)
    coarse_measurement = interval_acceptance(expected['active_duration_s'],
        expected['active_duration_s'], uncertainty=.5e-9, tolerance=.1e-9)
    assert coarse_measurement['status'] == 'inconclusive'
    wrong_quantity = interval_acceptance(expected['busy_duration_s'],
        expected['active_duration_s'], uncertainty=.05e-9, tolerance=.1e-9)
    assert wrong_quantity['status'] == 'failed'


def measured_tone(frequency=10e6, *, duration=20e-6, dt=1e-9, **kwargs):
    time = np.arange(0., duration, dt)
    values = .023+.2*np.cos(2*np.pi*frequency*time+.17)
    policy = dict(expected_hz=10e6, tolerance_hz=5000., baseline_v=.023,
        bandwidth_hz=100e6, allowed_band_hz=(8e6, 12e6), voltage_uncertainty_v=1e-4,
        edge_uncertainty_s=.1e-9, timebase_uncertainty_ppm=1.)
    policy.update(kwargs)
    return frequency_measurement(time, values, **policy)


def test_frequency_estimate_comes_from_waveform_not_expected_bin():
    result = measured_tone()
    assert result['status'] == 'passed'
    assert result['complete_cycles'] >= 190
    assert abs(result['measured_hz']-10e6) < 1.
    wrong = measured_tone(frequency=10.2e6)
    assert wrong['status'] == 'failed'
    assert abs(wrong['measured_hz']-10.2e6) < 100.


@pytest.mark.parametrize('kwargs,reason', [
    ({'duration': .5e-6}, 'Too few'),
    ({'dt': 20e-9}, 'sampling'),
    ({'bandwidth_hz': 9e6}, 'bandwidth'),
    ({'clipped': True}, 'Clipped'),
    ({'tolerance_hz': 1.}, None),
    ({'allowed_band_hz': (20e6, 30e6)}, 'outside'),
    ({'voltage_uncertainty_v': .3}, 'baseline'),
    ({'edge_uncertainty_s': 30e-9}, 'adjacent'),
])
def test_insufficient_measurement_is_inconclusive_not_pass(kwargs, reason):
    result = measured_tone(**kwargs)
    assert result['status'] == 'inconclusive'
    if reason:
        assert reason in result['reason']


def test_alias_plausible_waveform_cannot_pass_with_unbounded_fixture_band():
    # Native samples of 90 MHz at 100 MSPS resemble 10 MHz. The fixture admits
    # 90 MHz, so a 10 MHz result cannot be certified from this capture.
    result = measured_tone(frequency=90e6, dt=10e-9,
                           allowed_band_hz=(8e6, 95e6), bandwidth_hz=100e6)
    assert result['status'] == 'inconclusive'


def test_low_snr_wrong_carrier_cannot_pass_when_allowed_error_supplies_the_tone():
    time = np.arange(0., 100e-6, 1e-9)
    uncertainty = .01
    true_rf = .10*uncertainty*np.sin(2*np.pi*9.8e6*time)
    permitted_error = .99*uncertainty*np.sin(2*np.pi*10e6*time)
    assert np.max(np.abs(permitted_error)) <= uncertainty
    result = frequency_measurement(time, true_rf+permitted_error,
        expected_hz=10e6, tolerance_hz=5000., baseline_v=0.,
        bandwidth_hz=100e6, allowed_band_hz=(8e6, 12e6),
        voltage_uncertainty_v=uncertainty, edge_uncertainty_s=0.,
        timebase_uncertainty_ppm=0.)
    assert result['status'] == 'inconclusive'


def test_bounded_endpoint_voltage_errors_cannot_fake_a_precise_frequency_match():
    dt, duration, frequency, noise = 1e-9, 20e-6, 10.006e6, .01
    time = np.arange(0., duration, dt)
    first_crossing = 25e-9
    signal = 2*noise*np.sin(2*np.pi*frequency*(time-first_crossing))
    observed = signal.copy()
    first = int((first_crossing-6e-9)//dt)
    last_crossing = first_crossing+199/frequency
    last = int((last_crossing+6e-9)//dt)
    observed[first] -= noise
    observed[first+1] += noise
    observed[first+2:int(math.ceil(first_crossing/dt))+1] = .02*noise
    observed[int(math.ceil(last_crossing/dt)):last] = -.02*noise
    observed[last] -= noise
    observed[last+1] += noise
    assert np.max(np.abs(observed-signal)) <= noise+1e-14
    result = frequency_measurement(time, observed, expected_hz=10e6,
        tolerance_hz=5000., baseline_v=0., bandwidth_hz=100e6,
        allowed_band_hz=(8e6, 12e6), voltage_uncertainty_v=noise,
        edge_uncertainty_s=0., timebase_uncertainty_ppm=0.)
    assert result['status'] == 'inconclusive'


def test_adjacent_periods_alone_do_not_prove_one_carrier_phase_for_entire_record():
    from validation.rf_reference import _constant_period_exists
    lower = np.arange(201, dtype=float)*100e-9
    upper = lower+1e-9
    assert _constant_period_exists(lower, upper, 99e-9, 101e-9)
    lower[1:4] += np.array([1., 2., 1.])*1e-9
    upper = lower+1e-9
    # Adjacent intervals touch near100ns, yet first/third/last phases cannot
    # share a single linear sequence. The record must not pass as one tone.
    assert not _constant_period_exists(lower, upper, 99e-9, 101e-9)


def test_independent_reference_matches_real_v5_register_loop_and_repetition_reset():
    from test_rf_fine_duration_registers import compile_sweep, execute, decode_bursts
    program = compile_sweep(113.2, 1.2, 29, outer=2, reps=2)
    _, events = execute(program)
    bursts = decode_bursts(program, events)
    durations = np.linspace(113.2/4800., 1.2/4800., 29)*1e-6
    expected = [RfGrid().samples(d) for _ in range(2) for d in durations for _ in range(2)]
    assert [row['samples'] for row in bursts] == expected
    assert all(row['commands'] == (1 if duration_layout(n)['mode'] == 'short' else 2)
               for row, n in zip(bursts, expected))


def test_independent_dds_words_match_executed_v5_frequency_axis():
    from collections import Counter
    import qick.sim  # noqa: F401
    from qick_fine_tune_sweep import FineTuneSequence, RfPulseConfig
    from test_rf_fine_duration_registers import _v5_soccfg, execute, decode_bursts
    cfg = _v5_soccfg()
    # The reusable width-only fixture changes f_dds but keeps old ratio fields.
    # This new frequency case explicitly declares a consistent 4.8 GHz DDS.
    cfg['gens'][0]['fs_mult'] = 16
    grid = RfGrid.from_metadata(cfg['gens'][0], reference_clock_mhz=cfg['refclk_freq'],
                                frequency_step_words=1)
    seq = FineTuneSequence(('awg_0',)).add_set('gate', .01, 4000)
    seq.add_rf_duration_sweep('gate', 0, 1/4800., 3/4800., 3,
                             parameter_name='d0', sequence_fabric_mhz=300.)
    seq.add_rf_frequency_sweep('gate', 0, 10., 20., 2)
    pulse = RfPulseConfig(0, 'gate', 3, 1000, freq_mhz=10., delay_tproc_cycles=30,
        fine_duration=True, duration_samples=1, duration_parameter='d0')
    program = seq.make_program(cfg, awg_channels=(1,), rf_pulses=(pulse,), repetitions_per_sweep=2)
    program.compile()
    _, events = execute(program)
    assert len(decode_bursts(program, events)) == 12
    words = Counter(event.word & 0xffffffff for event in events)
    expected = Counter({grid.frequency(f)['word']: 6 for f in (10e6, 20e6)})
    assert words == expected


def test_inconsistent_fdds_and_ratio_metadata_are_rejected_before_reference_generation():
    generator = dict(f_fabric=300., f_dds=4800., fs_mult=1, fdds_div=1, b_dds=32)
    with pytest.raises(ValueError, match='metadata disagree'):
        RfGrid.from_metadata(generator, reference_clock_mhz=300., frequency_step_words=1)
    generator['fs_mult'] = 16
    grid = RfGrid.from_metadata(generator, reference_clock_mhz=300., frequency_step_words=1)
    assert grid.sample_hz == grid.dds_hz == 4.8e9


@pytest.mark.parametrize('policy', [True, False])
def test_real_v5_compiler_rejects_only_later_padded_boundary_or_preserves_extension(policy):
    import qick.sim  # noqa: F401
    from qick_fine_tune_sweep import FineTuneSequence, RfPulseConfig
    from test_rf_fine_duration_registers import _v5_soccfg, execute, decode_bursts
    # A 200 ns anchor is 60 clocks. Start at clock 55; N=1 and N=80 fit
    # while N=81 occupies six clocks. Keep a later SET far from RF commands.
    seq = FineTuneSequence(('awg_0',)).add_set('gate', .01, 60).add_set('last', 0., 300)
    seq.add_rf_duration_sweep('gate', 0, 1/4800., 81/4800., 81,
                             parameter_name='d0', sequence_fabric_mhz=300.)
    pulse = RfPulseConfig(0, 'gate', 3, 1000, freq_mhz=0., delay_tproc_cycles=55,
        fine_duration=True, duration_samples=1, duration_parameter='d0',
        require_within_segment=policy)
    if policy:
        with pytest.raises(ValueError, match='padded waveform exceeds'):
            program = seq.make_program(_v5_soccfg(), awg_channels=(1,), rf_pulses=(pulse,))
            program.compile()
        # Preserve the first eighty points exactly; removing only the violating
        # last point must make the real compiler accept the same HOLD boundary.
        seq.add_rf_duration_sweep('gate', 0, 1/4800., 80/4800., 80,
                                 parameter_name='d0', sequence_fabric_mhz=300.)
        program = seq.make_program(_v5_soccfg(), awg_channels=(1,), rf_pulses=(pulse,))
        program.compile()
        _, events = execute(program)
        assert [x['samples'] for x in decode_bursts(program, events)] == list(range(1, 81))
    else:
        program = seq.make_program(_v5_soccfg(), awg_channels=(1,), rf_pulses=(pulse,))
        program.compile()
        _, events = execute(program)
        assert [x['samples'] for x in decode_bursts(program, events)] == list(range(1, 82))


def test_real_v5_false_policy_preserves_final_point_busy_end_without_truncation():
    import qick.sim  # noqa: F401
    from qick_fine_tune_sweep import FineTuneSequence, RfPulseConfig
    from test_rf_fine_duration_registers import _v5_soccfg, execute, decode_bursts
    seq = FineTuneSequence(('awg_0',)).add_set('gate', .01, 60)
    seq.add_rf_duration_sweep('gate', 0, 1/4800., 81/4800., 81,
                             parameter_name='d0', sequence_fabric_mhz=300.)
    pulse = RfPulseConfig(0, 'gate', 3, 1000, freq_mhz=0., delay_tproc_cycles=55,
        fine_duration=True, duration_samples=1, duration_parameter='d0',
        require_within_segment=False)
    program = seq.make_program(_v5_soccfg(), awg_channels=(1,), rf_pulses=(pulse,))
    program.compile()
    _, events = execute(program)
    assert [x['samples'] for x in decode_bursts(program, events)] == list(range(1, 82))
    # Intended native output epoch includes 31 cycles, so the final RF busy end
    # is 31+55+6=92, one clock past the unchanged user segment end 31+60=91.
    assert program.timing['segment_ends'] == (91,)
    assert program._dynamic_point_end >= 92
