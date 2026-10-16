"""Explicit v5 AWG+RF reference composition, without devices or compiler calls.

Every input sample is already aligned to a declared final-DAC time axis. The
native 31-clock generator latency is metadata, not an extra shift applied here.
Callers must supply the entire history they want compared; no missing interval,
RF phase, zero padding, analog calibration, or measured-data fit is inferred.
"""
from dataclasses import asdict
import math

import numpy as np

from .archive import Reference
from .rf_reference import RfGrid, interval_acceptance, physical_reference


def _finite(value, name, *, positive=False, nonnegative=False):
    value = float(value)
    if (not math.isfinite(value) or (positive and value <= 0)
            or (nonnegative and value < 0)):
        raise ValueError(f'{name} must be finite and within its declared range')
    return value


def _evidence(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Explicit {name} is required')
    return value


def validate_v5_pair(config, *, awg_generator, rf_generator, expected_dac,
                     nyquist_zone, mixer_hz, frequency_step_words):
    """Check queried metadata consistency, not loaded-hardware authenticity.

    Nyquist zone and mixer are required live/settings inputs: no absent setting
    is interpreted as the default. This helper supports baseband, unbypassed
    v5 pairs only. A connector-to-scope-channel map is still the caller's job.
    """
    for index in (awg_generator, rf_generator):
        if isinstance(index, bool) or int(index) != index or index < 0:
            raise ValueError('Generator indices must be nonnegative integers')
    awg, rf = (config['gens'][int(index)] for index in (awg_generator, rf_generator))
    if awg['type'] != 'axis_awg_tuning_v3' or rf['type'] != 'axis_signal_gen_v6':
        raise ValueError('Expected an AWG v3 / SGv6 pair')
    if nyquist_zone != 1 or _finite(mixer_hz, 'mixer_hz') != 0:
        raise ValueError('Only Nyquist zone 1 and zero mixer are modeled')
    for generator, role in ((awg, 'awg'), (rf, 'rf')):
        required = dict(output_sum_version=2, output_sum_role=role,
            output_sum_bypass=False, output_sum_awg_delay_cycles=13,
            post_output_latency_cycles=2, output_latency_cycles=31,
            samps_per_clk=16, dac=str(expected_dac))
        if any(generator.get(key) != value for key, value in required.items()):
            raise ValueError(f'{role} does not match the declared v5 pair metadata')
        if generator.get('output_sum_awg') != awg['fullpath'] or generator.get('output_sum_rf') != rf['fullpath']:
            raise ValueError('Sum block paths do not identify the declared AWG/RF generators')
    if not awg.get('output_sum_block') or awg['output_sum_block'] != rf['output_sum_block']:
        raise ValueError('AWG and RF must feed the same sum block')
    if (awg.get('dac_effective_bits'), awg.get('dac_invalid_lsb'), awg.get('dac_word_bits')) != (14, 2, 16):
        raise ValueError('Expected signed 14-bit DAC samples in 16-bit transport words')
    grid = RfGrid.from_metadata(rf, reference_clock_mhz=config['refclk_freq'],
                               frequency_step_words=frequency_step_words)
    for generator in (awg, rf):
        if not math.isclose(float(generator['f_fabric'])*1e6, grid.fabric_hz, rel_tol=1e-12):
            raise ValueError('Pair fabric clocks disagree')
        if not math.isclose(float(generator['fs'])*1e6, grid.sample_hz, rel_tol=1e-12):
            raise ValueError('Pair DAC sample rate does not match the 16-lane fabric rate')
        if generator.get('interpolation') != 1:
            raise ValueError('Interpolating RFDC behavior needs a separate reference')
    return dict(schema='v5-pair-consistency-v1', awg_generator=int(awg_generator),
        rf_generator=int(rf_generator), dac=str(expected_dac),
        awg_path=awg['fullpath'], rf_path=rf['fullpath'],
        sum_block=awg['output_sum_block'], grid=asdict(grid),
        native_output_latency_cycles=31,
        time_plane='Inputs must already be aligned to final DAC sample time',
        hardware_identity_verified=False, nyquist_zone=1, mixer_hz=0.)


def _transport(values, name):
    values = np.asarray(values)
    if (values.ndim != 1 or not len(values) or values.dtype.kind not in 'iu'
            or np.any(values < -32768) or np.any(values > 32767)):
        raise ValueError(f'{name} must be explicit signed-16-range integer samples')
    return values.astype(np.int64)


def sum_transport_codes(awg_transport, rf_transport):
    """Exact sum-v2 arithmetic conditional on supplied, aligned input samples.

    Invalid input cycles must have been explicitly resolved by the caller.
    This does not reconstruct a DDS, its pipeline, or AXIS valid history.
    The clip flag uses pre-rounding limits; 32765 clips even though rounding
    alone would still produce the largest positive 14-bit code.
    """
    awg, rf = _transport(awg_transport, 'AWG'), _transport(rf_transport, 'RF')
    if awg.shape != rf.shape:
        raise ValueError('AWG and RF need identical complete sample grids')
    total = awg+rf
    code = np.clip((total+2)//4, -8192, 8191).astype(np.int16)
    return dict(sum_signed17=total, effective14=code,
        transport16=(code.astype(np.int32)*4).astype(np.int16),
        clipped=(total > 32764) | (total < -32768),
        exact_given_supplied_inputs=True)


def _same_grid(reference, time, name):
    t, v = reference.arrays()
    if not np.array_equal(t, time):
        raise ValueError(f'{name} needs the identical full time grid; no padding or interpolation')
    return np.asarray(v, dtype=float)


def _component_plane(reference, *, expectation, time_origin):
    meta = reference.metadata
    if (meta.get('units') != 'V' or meta.get('expectation') != expectation
            or meta.get('signal_plane') != 'aligned_final_DAC_nominal_voltage'
            or meta.get('analog_transfer_applied') is not False):
        raise ValueError('Components must declare volts, requested/quantized expectation, '
                         'and aligned final-DAC voltage before analog transfer')
    if meta.get('time_origin') != time_origin:
        raise ValueError('Components must explicitly declare the same time origin')


def _rf_metadata(reference, time, name):
    meta = reference.metadata
    for key in ('active_start_s', 'active_end_s', 'busy_end_s', 'phase_epoch_s',
                'phase_degrees', 'frequency_hz'):
        _finite(meta[key], f'{name}.{key}')
    start, end, busy = (float(meta[key]) for key in ('active_start_s', 'active_end_s', 'busy_end_s'))
    if not time[0] <= start < end <= busy <= time[-1]:
        raise ValueError('Full RF active and busy intervals must be explicitly covered')
    # Outside a declared pulse is checked, not silently filled with zeros.
    _, values = reference.arrays()
    outside = (time < start) | (time >= end)
    if np.any(values[outside] != 0):
        raise ValueError('RF samples outside the explicitly declared active interval must be zero')


def prepare_sum_references(*, pair, requested_awg, requested_rf, quantized_awg,
                           quantized_rf, full_scale_v, full_scale_evidence,
                           alignment_evidence, history_evidence,
                           digital_model_evidence, digital_uncertainty_v,
                           calibrated_transfer=None):
    """Compose explicitly supplied voltage expectations for one paired DAC.

    RF amplitude in input References must be explicitly nominal or calibrated;
    a GUI gain is never silently treated as volts. `digital_uncertainty_v` must
    cover input DDS/envelope amplitude approximations. The helper adds half a
    DAC LSB for final sum rounding. Neither covers analog transfer errors. The
    nominal engineering prediction assumes
    analog gain 1 at the declared DAC full scale, and includes DAC clipping.
    It is *not* a calibrated amplitude prediction or a bit-exact DDS model.

    With no calibrated transfer, frequency and envelope measurement remain
    available, subject to their own bandwidth/SNR/timebase tests. Calibrated
    amplitude accuracy remains inconclusive. A separate 15 mV engineering
    screen can be reported against `engineering`; it is not calibration PASS.

    No latency is added, and no samples are extrapolated. Each component must
    be specified on the same native, final-DAC grid, including actual pre/post
    history. A missing or inactive RF source must be modeled explicitly by its
    caller rather than passed through this single-active-pulse helper.

    Component metadata must declare units='V', expectation='requested' or
    'quantized', signal_plane='aligned_final_DAC_nominal_voltage', and
    analog_transfer_applied=False. Physical/scope references cannot be fed back
    as digital components. The source time-origin declaration must also match.
    """
    if pair.get('schema') != 'v5-pair-consistency-v1':
        raise ValueError('Validate the v5 pair metadata before composing references')
    scale = _finite(full_scale_v, 'full_scale_v', positive=True)
    input_uncertainty = _finite(digital_uncertainty_v, 'digital_uncertainty_v', nonnegative=True)
    rounding_uncertainty = scale/16384
    uncertainty = input_uncertainty+rounding_uncertainty
    evidence = {name: _evidence(value, name) for name, value in (
        ('full_scale_evidence', full_scale_evidence), ('alignment_evidence', alignment_evidence),
        ('history_evidence', history_evidence), ('digital_model_evidence', digital_model_evidence))}
    time, requested_a = requested_awg.arrays()
    time = np.asarray(time, dtype=float)
    for reference, expectation in ((requested_awg, 'requested'), (requested_rf, 'requested'),
                                   (quantized_awg, 'quantized'), (quantized_rf, 'quantized')):
        _component_plane(reference, expectation=expectation,
                         time_origin=requested_awg.metadata['time_origin'])
    dt = np.diff(time)
    expected_dt = 1/(pair['grid']['fabric_hz']*pair['grid']['samples_per_clock'])
    if len(time) < 2 or not np.allclose(dt, expected_dt, rtol=1e-6, atol=0):
        raise ValueError('References need the explicit native final-DAC sample grid')
    requested_r = _same_grid(requested_rf, time, 'requested RF')
    quantized_a = _same_grid(quantized_awg, time, 'quantized AWG')
    quantized_r = _same_grid(quantized_rf, time, 'quantized RF')
    for ref, name in ((requested_rf, 'requested RF'), (quantized_rf, 'quantized RF')):
        _rf_metadata(ref, time, name)
        if ref.metadata.get('amplitude_scale_kind') not in ('nominal_dac_gain1', 'calibrated'):
            raise ValueError('Declare RF amplitude_scale_kind: nominal_dac_gain1 or calibrated')
        _evidence(ref.metadata.get('amplitude_scale_evidence'), 'RF amplitude_scale_evidence')
    lower, upper = -scale, scale*8191/8192
    combined = quantized_a+quantized_r
    # Continuous voltage expectations are not silently rounded into purported
    # exact DDS words. The caller declares their digital approximation bound.
    engineering = np.clip(combined, lower, upper)
    clipped = (combined < lower) | (combined > upper)
    definite_clip = (combined+uncertainty < lower) | (combined-uncertainty > upper)
    possible_clip = (combined-uncertainty < lower) | (combined+uncertainty > upper)
    metadata = dict(units='V', time_origin=requested_awg.metadata['time_origin'],
        reference_plane='final DAC on the explicitly supplied time grid',
        model_version='v5-awg-rf-composition-v1', transformations=[],
        pair=dict(pair), full_scale_v=scale, nominal_analog_gain=1.,
        digital_model_uncertainty_v=uncertainty,
        input_digital_model_uncertainty_v=input_uncertainty,
        final_rounding_uncertainty_v=rounding_uncertainty, evidence=evidence,
        exact_integer_dds=False, frequency_hz=quantized_rf.metadata['frequency_hz'],
        rf_timing=dict(quantized_rf.metadata),
        rf_timing_plane='final DAC before the declared analog transfer',
        comparison_class='nominal_DAC_engineering_screen',
        nominal_clipped_samples=int(clipped.sum()),
        definitely_clipped_samples=int(definite_clip.sum()),
        possibly_clipped_samples=int(possible_clip.sum()))
    requested = Reference(time.copy(), np.asarray(requested_a)+requested_r,
                          dict(metadata, expectation='requested_unclipped',
                               frequency_hz=requested_rf.metadata['frequency_hz'],
                               rf_timing=dict(requested_rf.metadata),
                               clipping_metadata_applies_to='quantized nominal prediction only'))
    nominal = Reference(time.copy(), engineering, dict(metadata, expectation='quantized_nominal'))
    calibrated = None
    if calibrated_transfer is not None:
        if not float(calibrated_transfer['valid_band_hz'][0]) == 0.:
            raise ValueError('AWG+RF calibrated transfer must include DC')
        # Transient evidence must cover the complete AWG+RF record, including
        # clipping if present; a carrier-only amplitude calibration is insufficient.
        _evidence(calibrated_transfer.get('composite_transient_evidence'), 'composite_transient_evidence')
        calibrated = physical_reference(nominal, calibrated_transfer)
        calibrated.metadata['comparison_class'] = 'calibrated_composite_prediction'
        calibrated.metadata['analog_transfer_applied'] = True
        calibrated.metadata['input_DAC_digital_model_uncertainty_v'] = uncertainty
        # Both available transfer types have induced sup-norm gain <= |gain|.
        # Transfer/initial-state uncertainty is separately declared by its model.
        propagated = abs(float(calibrated_transfer['gain']))*uncertainty
        calibrated.metadata['digital_model_uncertainty_v'] = propagated
        calibrated.metadata['combined_voltage_model_uncertainty_v'] = (
            propagated+float(calibrated_transfer['voltage_model_uncertainty_v']))
    return dict(requested=requested, engineering=nominal, physical=calibrated,
        nominal_clip_mask=clipped, definite_clip_mask=definite_clip,
        possible_clip_mask=possible_clip,
        measurement_readiness=dict(
            carrier_frequency='requires_measurement_and_own_uncertainty_checks',
            envelope='requires_measurement_and_own_bandwidth_timing_checks',
            engineering_amplitude='nominal_DAC_gain1_prediction_available',
            calibrated_amplitude='prediction_available' if calibrated is not None else 'inconclusive'),
        physical_execution=False)


def engineering_screen(*, measured_statistic_v, expected_statistic_v,
                       fixed_offset_v, offset_evidence, statistic_definition,
                       uncertainty_v, tolerance_v=.015):
    """Explicit scalar screen, not amplitude calibration or waveform fitting.

    Caller defines the statistic and ROI before measurement. The one supplied
    offset must be independently established, never optimized per point/edge.
    Equality at 15 mV passes the user's 'failure above 15 mV' engineering rule;
    measurement uncertainty overlapping the boundary stays inconclusive.
    """
    _evidence(offset_evidence, 'offset_evidence')
    _evidence(statistic_definition, 'statistic_definition')
    result = interval_acceptance(
        _finite(measured_statistic_v, 'measured_statistic_v')-_finite(fixed_offset_v, 'fixed_offset_v'),
        expected_statistic_v, uncertainty=uncertainty_v, tolerance=tolerance_v, strict=False)
    return dict(result, comparison_class='nominal_DAC_engineering_screen',
        calibrated_amplitude_accuracy='inconclusive', fixed_offset_v=float(fixed_offset_v),
        offset_evidence=offset_evidence, statistic_definition=statistic_definition)
