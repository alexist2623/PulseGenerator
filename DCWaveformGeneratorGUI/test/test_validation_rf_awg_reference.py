"""Offline AWG+RF composition, explicit scope-readiness and engineering limits."""
import numpy as np
import pytest

from validation.archive import Reference
from validation.rf_awg_reference import (engineering_screen, prepare_sum_references,
    sum_transport_codes, validate_v5_pair)
from validation.rf_reference import frequency_measurement, rf_point, rf_waveform


def configuration():
    common = dict(output_sum_version=2, output_sum_block='sum_4', output_sum_bypass=False,
        output_sum_awg='awg_4', output_sum_rf='rf_0', output_sum_awg_delay_cycles=13,
        post_output_latency_cycles=2, output_latency_cycles=31, samps_per_clk=16,
        dac='10', f_fabric=300., fs=4800., interpolation=1)
    rf = dict(common, type='axis_signal_gen_v6', fullpath='rf_0', output_sum_role='rf',
        fs_mult=32, fdds_div=2, f_dds=4800., b_dds=32)
    awg = dict(common, type='axis_awg_tuning_v3', fullpath='awg_4', output_sum_role='awg',
        dac_effective_bits=14, dac_invalid_lsb=2, dac_word_bits=16)
    return dict(refclk_freq=300., gens=[rf, awg])


def pair(config=None, **kwargs):
    inputs = dict(awg_generator=1, rf_generator=0, expected_dac='10',
                  nyquist_zone=1, mixer_hz=0., frequency_step_words=1)
    inputs.update(kwargs)
    return validate_v5_pair(config or configuration(), **inputs)


def inputs():
    time = np.arange(0, 1600)/4.8e9
    metadata = dict(units='V', time_origin='declared point epoch',
        reference_plane='declared final-DAC sample time', model_version='test-independent-v1',
        transformations=[], signal_plane='aligned_final_DAC_nominal_voltage',
        analog_transfer_applied=False)
    awg = Reference(time, np.full(len(time), .1), dict(metadata, expectation='requested'))
    quantized_awg = Reference(time, np.full(len(time), .1), dict(metadata, expectation='quantized'))
    point = rf_point(duration_s=97/4.8e9, frequency_hz=151e6,
        output_start_cycle=50, anchor_end_cycle=80, final_user_end_cycle=90,
        require_within_segment=True, output_latency_cycles=44,
        command_time_plane='tprocessor_model_event')
    quantized = rf_waveform(point, time, amplitude_v=.2, phase_degrees=30.,
                            phase_epoch_s=-1e-6)
    requested = rf_waveform(point, time, amplitude_v=.2, phase_degrees=30.,
        phase_epoch_s=-1e-6, quantized=False, requested_start_s=point['output_start_s'])
    for ref in (quantized, requested):
        ref.metadata.update(amplitude_scale_kind='nominal_dac_gain1',
                            signal_plane='aligned_final_DAC_nominal_voltage', analog_transfer_applied=False,
                            amplitude_scale_evidence='explicit fixture nominal voltage, not GUI gain')
    return dict(pair=pair(), requested_awg=awg, requested_rf=requested,
        quantized_awg=quantized_awg, quantized_rf=quantized, full_scale_v=.8,
        full_scale_evidence='fixture current/full-scale declaration',
        alignment_evidence='explicit common final-DAC epoch, 44 event clocks already included',
        history_evidence='AWG=.1 V throughout this supplied interval; RF explicitly off outside pulse',
        digital_model_evidence='ideal cosine nominal model; no bit-exact DDS claim',
        digital_uncertainty_v=.0002)


def test_pair_validates_actual_ratio_contract_without_authenticating_hardware():
    result = pair()
    assert result['grid']['dds_hz'] == result['grid']['fabric_hz']*16 == 4.8e9
    assert result['native_output_latency_cycles'] == 31
    assert result['dac'] == '10'
    assert result['hardware_identity_verified'] is False


@pytest.mark.parametrize('generator,key,value', [
    (0, 'output_sum_version', 1), (1, 'output_sum_awg_delay_cycles', 0),
    (0, 'output_sum_bypass', True), (0, 'output_sum_role', 'awg'),
    (0, 'dac', '11'), (1, 'output_sum_rf', 'other_rf'),
    (1, 'output_sum_block', 'other_sum'), (1, 'dac_invalid_lsb', 0),
    (0, 'fs_mult', 1), (1, 'f_fabric', 200.), (1, 'fs', 2400.),
    (0, 'interpolation', 2),
])
def test_inconsistent_metadata_cannot_silently_choose_a_pair(generator, key, value):
    config = configuration()
    config['gens'][generator][key] = value
    with pytest.raises(ValueError):
        pair(config)


@pytest.mark.parametrize('kwargs', [dict(nyquist_zone=2), dict(mixer_hz=1.),
                                  dict(awg_generator=-1)])
def test_unknown_or_unsupported_output_mapping_is_rejected(kwargs):
    with pytest.raises(ValueError):
        pair(**kwargs)


def test_every_signed17_reachable_total_matches_independent_scalar_sum_rule():
    totals = np.arange(-65536, 65535, dtype=np.int64)
    awg = np.clip(totals, -32768, 32767)
    rf = totals-awg
    actual = sum_transport_codes(awg, rf)
    expected = np.array([max(-8192, min(8191, (int(value)+2)//4)) for value in totals])
    np.testing.assert_array_equal(actual['effective14'], expected)
    np.testing.assert_array_equal(actual['transport16'], expected*4)
    np.testing.assert_array_equal(actual['clipped'], (totals < -32768) | (totals > 32764))
    assert actual['exact_given_supplied_inputs']


def test_half_ties_and_clipflag_are_not_inferred_from_rounded_output():
    result = sum_transport_codes(np.array([-6, -2, 2, 6, 32764]), np.array([0, 0, 0, 0, 1]))
    np.testing.assert_array_equal(result['effective14'], [-1, 0, 1, 2, 8191])
    np.testing.assert_array_equal(result['clipped'], [False, False, False, False, True])


@pytest.mark.parametrize('left,right', [([1.5], [0]), ([32768], [0]),
    ([-32769], [0]), ([1, 2], [0]), ([], []), ([True], [False])])
def test_invalid_integer_samples_are_not_cast_or_padded(left, right):
    with pytest.raises(ValueError):
        sum_transport_codes(left, right)


def test_no_calibration_keeps_frequency_and_envelope_readiness_separate():
    args = inputs()
    result = prepare_sum_references(**args)
    assert result['physical'] is None
    readiness = result['measurement_readiness']
    assert readiness['calibrated_amplitude'] == 'inconclusive'
    assert readiness['carrier_frequency'].startswith('requires_measurement')
    assert readiness['envelope'].startswith('requires_measurement')
    nominal = result['engineering']
    np.testing.assert_array_equal(nominal.time_s, args['requested_awg'].time_s)
    np.testing.assert_allclose(nominal.voltage_v, .1+args['quantized_rf'].voltage_v)
    assert nominal.metadata['rf_timing']['active_start_s'] == 50/300e6
    assert nominal.metadata['rf_timing']['busy_end_s'] == 57/300e6
    assert nominal.metadata['exact_integer_dds'] is False
    assert result['physical_execution'] is False


def test_composition_preserves_requested_frequency_and_duration_values():
    args = inputs()
    requested = args['requested_rf']
    quantized = args['quantized_rf']
    # Actual frequency-grid values differ; neither is silently replaced.
    assert requested.metadata['frequency_hz'] != quantized.metadata['frequency_hz']
    result = prepare_sum_references(**args)
    np.testing.assert_array_equal(result['requested'].voltage_v,
                                  args['requested_awg'].voltage_v+requested.voltage_v)
    np.testing.assert_array_equal(result['engineering'].voltage_v,
                                  args['quantized_awg'].voltage_v+quantized.voltage_v)


def test_nominal_clipping_is_reported_with_the_declared_digital_uncertainty():
    args = inputs()
    ref = args['quantized_awg']
    args['quantized_awg'] = Reference(ref.time_s, np.full(len(ref.time_s), .7999), ref.metadata)
    result = prepare_sum_references(**args)
    assert np.any(result['nominal_clip_mask'])
    assert np.all(result['possible_clip_mask'][result['nominal_clip_mask']])
    assert np.max(result['engineering'].voltage_v) == .8*8191/8192
    assert np.max(result['requested'].voltage_v) < .31


@pytest.mark.parametrize('missing', ['alignment_evidence', 'history_evidence',
                                    'digital_model_evidence', 'full_scale_evidence'])
def test_no_guessed_time_history_or_nominal_voltage_scale(missing):
    args = inputs()
    args[missing] = ''
    with pytest.raises(ValueError, match='Explicit'):
        prepare_sum_references(**args)


def test_missing_busy_padding_is_rejected_even_when_active_samples_fit():
    args = inputs()
    limit = 900  # RF active ends at 897, but busy ends at 912 samples.
    for key in ('requested_awg', 'requested_rf', 'quantized_awg', 'quantized_rf'):
        ref = args[key]
        args[key] = Reference(ref.time_s[:limit], ref.voltage_v[:limit], ref.metadata)
    with pytest.raises(ValueError, match='busy intervals'):
        prepare_sum_references(**args)


def test_missing_samples_or_rf_scale_cannot_be_filled_in():
    args = inputs()
    ref = args['quantized_rf']
    args['quantized_rf'] = Reference(ref.time_s[:-1], ref.voltage_v[:-1], ref.metadata)
    with pytest.raises(ValueError, match='identical full time grid'):
        prepare_sum_references(**args)
    args = inputs()
    del args['quantized_rf'].metadata['amplitude_scale_kind']
    with pytest.raises(ValueError, match='amplitude_scale_kind'):
        prepare_sum_references(**args)


def test_out_of_window_nonzero_rf_is_not_treated_as_zero_padding():
    args = inputs()
    args['quantized_rf'].voltage_v[0] = .01
    with pytest.raises(ValueError, match='outside'):
        prepare_sum_references(**args)


def calibration():
    return dict(kind='flat', calibration_evidence='synthetic traceable response',
        reference_plane='synthetic scope plane', valid_band_hz=[0, 200e6],
        transient_model_evidence='synthetic step/carrier response',
        composite_transient_evidence='complete AWG+RF composite response coverage',
        voltage_model_uncertainty_v=.001, timing_model_uncertainty_s=1e-10,
        gain=.9, offset_v=-.01, delay_s=2e-9)


def test_explicit_calibrated_composite_transfer_is_applied_once():
    args = inputs()
    args['calibrated_transfer'] = calibration()
    result = prepare_sum_references(**args)
    np.testing.assert_allclose(result['physical'].time_s,
                               result['engineering'].time_s+2e-9, rtol=0, atol=0)
    np.testing.assert_allclose(result['physical'].voltage_v,
                               .9*result['engineering'].voltage_v-.01)
    assert result['measurement_readiness']['calibrated_amplitude'] == 'prediction_available'
    assert result['physical'].metadata['digital_model_uncertainty_v'] == .9*(.0002+.8/16384)


@pytest.mark.parametrize('key,value', [('valid_band_hz', [1e6, 200e6]),
                                     ('composite_transient_evidence', '')])
def test_carrier_only_calibration_does_not_validate_composite_dc_or_transients(key, value):
    args = inputs()
    model = calibration()
    model[key] = value
    args['calibrated_transfer'] = model
    with pytest.raises(ValueError):
        prepare_sum_references(**args)


def test_fixed_offset_engineering_limit_is_not_calibration_pass():
    result = engineering_screen(measured_statistic_v=.214, expected_statistic_v=.2,
        fixed_offset_v=-.0005, offset_evidence='one fixed pre-frame baseline',
        statistic_definition='predeclared central plateau mean', uncertainty_v=.0002)
    assert result['status'] == 'passed'
    assert result['calibrated_amplitude_accuracy'] == 'inconclusive'
    result = engineering_screen(measured_statistic_v=.216, expected_statistic_v=.2,
        fixed_offset_v=0., offset_evidence='independent fixed baseline',
        statistic_definition='predeclared envelope peak', uncertainty_v=.0002)
    assert result['status'] == 'failed'
    result = engineering_screen(measured_statistic_v=.215, expected_statistic_v=.2,
        fixed_offset_v=0., offset_evidence='independent fixed baseline',
        statistic_definition='predeclared envelope peak', uncertainty_v=.001)
    assert result['status'] == 'inconclusive'


@pytest.mark.parametrize('key,value', [('units', 'mV'), ('expectation', 'physical')])
def test_non_dac_or_wrong_unit_component_cannot_be_relabelled_as_dac_voltage(key, value):
    args = inputs()
    args['quantized_rf'].metadata[key] = value
    with pytest.raises(ValueError):
        prepare_sum_references(**args)


def test_scope_plane_digital_uncertainty_scales_with_transfer_gain():
    args = inputs()
    args['calibrated_transfer'] = dict(calibration(), gain=2.)
    prepared = prepare_sum_references(**args)
    assert prepared['physical'].metadata['digital_model_uncertainty_v'] == 2*prepared['engineering'].metadata['digital_model_uncertainty_v']


def test_requested_metadata_keeps_its_own_frequency_and_timing():
    args = inputs()
    prepared = prepare_sum_references(**args)
    assert prepared['requested'].metadata['frequency_hz'] == args['requested_rf'].metadata['frequency_hz']
    assert prepared['requested'].metadata['rf_timing'] == args['requested_rf'].metadata


def test_uncertainty_overlapping_rail_is_nominal_not_definite_clipping():
    args = inputs()
    ref = args['quantized_awg']
    rail = args['full_scale_v']*8191/8192
    args['quantized_awg'] = Reference(ref.time_s, np.full(len(ref.time_s), rail+.0001), ref.metadata)
    prepared = prepare_sum_references(**args)
    meta = prepared['engineering'].metadata
    assert meta['nominal_clipped_samples'] > meta['definitely_clipped_samples']


@pytest.mark.parametrize('duration,expected_status', [(2e-6, 'inconclusive'), (10e-6, 'passed')])
def test_nominal_sum_supports_independent_carrier_measurement_without_amplitude_calibration(duration, expected_status):
    args = inputs()
    time = np.arange(round((duration+1e-6)*4.8e9)+1)/4.8e9
    point = rf_point(duration_s=duration, frequency_hz=10e6, output_start_cycle=50,
        anchor_end_cycle=round(duration*300e6)+100, final_user_end_cycle=round(duration*300e6)+200, require_within_segment=True,
        output_latency_cycles=44, command_time_plane='tprocessor_model_event')
    old = args['requested_awg']
    awg = Reference(time, np.full(len(time), .1), old.metadata)
    quantized_awg = Reference(time, awg.voltage_v.copy(), dict(old.metadata, expectation='quantized'))
    args.update(requested_awg=awg, quantized_awg=quantized_awg)
    for key, quantized in [('requested_rf', False), ('quantized_rf', True)]:
        reference = rf_waveform(point, time, amplitude_v=.2, phase_degrees=30.,
            phase_epoch_s=0., requested_start_s=point['output_start_s'], quantized=quantized)
        reference.metadata.update(amplitude_scale_kind='nominal_dac_gain1',
                                  signal_plane='aligned_final_DAC_nominal_voltage', analog_transfer_applied=False,
                                  amplitude_scale_evidence='explicit synthetic nominal scale')
        args[key] = reference
    prepared = prepare_sum_references(**args)
    roi = (time >= point['output_start_s']) & (time < point['active_end_s'])
    # The AWG baseline and an independently declared fixed offset are removed
    # for frequency measurement only; neither changes the RF amplitude status.
    measurement = frequency_measurement(time[roi], prepared['engineering'].voltage_v[roi]-.02,
        expected_hz=point['quantized_hz'], tolerance_hz=1000., baseline_v=.08,
        bandwidth_hz=100e6, allowed_band_hz=(9e6, 11e6), voltage_uncertainty_v=.0001,
        edge_uncertainty_s=1e-12, timebase_uncertainty_ppm=1., clipped=False)
    assert measurement['status'] == expected_status
    assert prepared['measurement_readiness']['calibrated_amplitude'] == 'inconclusive'
