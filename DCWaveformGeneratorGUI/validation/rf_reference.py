"""Independent RF expectations and conservative measurement acceptance.

No compiler, GUI, device, or measured-data fit is used to build expectations.
These helpers describe baseband SGv6 RF alone, not an AWG+RF sum or an unknown
Nyquist-zone/analog transfer function. Missing physical models remain explicit.
All times and frequencies in the public numerical API are seconds and Hz.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .archive import Reference


def _finite(value, name, *, positive=False, nonnegative=False):
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f'{name} must be finite and within its declared range')
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


@dataclass(frozen=True)
class RfGrid:
    """Explicit queried clocks; frequency_step_words includes readout matching."""
    fabric_hz: float = 300e6
    samples_per_clock: int = 16
    dds_hz: float = 4800e6
    dds_bits: int = 32
    frequency_step_words: int = 1
    duration_input_quantum_s: float = 1e-15  # GUI nine-decimal microseconds.

    def __post_init__(self):
        for name in ('fabric_hz', 'dds_hz'):
            object.__setattr__(self, name, _finite(getattr(self, name), name, positive=True))
        for name in ('samples_per_clock', 'dds_bits', 'frequency_step_words'):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        if self.samples_per_clock != 16 or self.dds_bits > 63:
            raise ValueError('This reference supports 16-lane SGv6 with <=63 DDS bits')
        if self.frequency_step_words >= (1 << (self.dds_bits-1)):
            raise ValueError('The common frequency grid must fit inside baseband Nyquist')
        object.__setattr__(self, 'duration_input_quantum_s',
                           _finite(self.duration_input_quantum_s, 'duration_input_quantum_s', nonnegative=True))

    @property
    def sample_hz(self):
        return self.fabric_hz * self.samples_per_clock

    @classmethod
    def from_metadata(cls, generator, *, reference_clock_mhz, frequency_step_words):
        """Validate both clock descriptions used by QICK's frequency conversion.

        `f_dds` alone is insufficient when refclk/fs_mult/fdds_div are present:
        QICK derives its register grid from the ratio fields. The caller also
        declares the common-grid multiplier if matching to a readout channel.
        This checks consistency, not authenticity of loaded-board metadata.
        """
        refclk = _finite(reference_clock_mhz, 'reference_clock_mhz', positive=True)
        multiplier = _integer(generator['fs_mult'], 'fs_mult', 1)
        divider = _integer(generator['fdds_div'], 'fdds_div', 1)
        dds = _finite(generator['f_dds'], 'f_dds', positive=True)
        derived = refclk*multiplier/divider
        if not math.isclose(dds, derived, rel_tol=1e-12, abs_tol=0.):
            raise ValueError('DDS metadata disagree: f_dds != refclk * fs_mult / fdds_div')
        return cls(fabric_hz=float(generator['f_fabric'])*1e6,
                   dds_hz=derived*1e6, dds_bits=generator['b_dds'],
                   frequency_step_words=frequency_step_words)

    def samples(self, duration_s, *, fine=True):
        duration = _finite(duration_s, 'duration_s', positive=True)
        rate = self.sample_hz if fine else self.fabric_hz
        value = duration * rate
        # A serialized requested integer sample can lie half a display unit
        # either side of its exact value. This is an input-resolution rule, not
        # a measurement tolerance. Values outside that interval round upward.
        tolerance = (.5 * self.duration_input_quantum_s * rate if fine else
                     max(1e-12, abs(value)*1e-12))
        tolerance += 8 * math.ulp(value)
        nearest = round(value)
        units = nearest if abs(value-nearest) <= tolerance else math.ceil(value)
        return max(1, int(units)) * (1 if fine else self.samples_per_clock)

    def frequency(self, requested_hz):
        frequency = _finite(requested_hz, 'requested_hz')
        if not -self.dds_hz/2 < frequency < self.dds_hz/2:
            raise ValueError('Declare a baseband frequency strictly inside DDS Nyquist')
        step = self.frequency_step_words
        signed_word = round(frequency * (1 << self.dds_bits) / self.dds_hz / step) * step
        if abs(signed_word) >= (1 << (self.dds_bits-1)):
            raise ValueError('Quantized frequency reaches the ambiguous Nyquist boundary')
        return dict(requested_hz=frequency, signed_word=int(signed_word),
                    word=int(signed_word % (1 << self.dds_bits)),
                    quantized_hz=signed_word * self.dds_hz / (1 << self.dds_bits),
                    grid_hz=step * self.dds_hz / (1 << self.dds_bits))


def duration_layout(samples, *, fine=True):
    """Active support versus SGv6 occupancy, independent of emitted instructions.

    A short fine pulse consumes at least three words, even at one active sample.
    Above five whole words a DDS plateau precedes a product tail. A plateau
    longer than a 16-bit one-shot uses three-word periodic boundaries.
    """
    samples = _integer(samples, 'samples', 1)
    words, residual = divmod(samples, 16)
    occupied = (samples+15)//16
    if not fine:
        if residual:
            raise ValueError('Coarse RF must contain whole fabric words')
        if words < 3:
            raise ValueError('SGv6 coarse RF needs at least three fabric clocks')
        return dict(samples=samples, remainder=0, busy_cycles=words,
                    mode='coarse', tail_start_samples=None)
    if words < 6:
        return dict(samples=samples, remainder=residual, busy_cycles=max(3, occupied),
                    mode='short', tail_start_samples=0)
    plateau = words-3
    periodic = plateau > 65535
    if periodic:
        plateau -= plateau % 3
    return dict(samples=samples, remainder=residual, busy_cycles=occupied,
                mode='periodic_tail' if periodic else 'plateau_tail',
                tail_start_samples=plateau*16)


def rf_point(*, duration_s, frequency_hz, output_start_cycle, anchor_end_cycle,
             final_user_end_cycle, require_within_segment, fine=True,
             output_latency_cycles, command_time_plane='native_generator_input', grid=None):
    """Describe one independent intended point; this does not accept a program.

    Anchor and start cycles must come from an independent requested AWG schedule,
    not be inferred from emitted RF instructions. Output latency is subtracted
    only for command time. It is never added a second time to physical occupancy.
    There is no default latency: v5's 31 cycles describes accepted native
    generator command to output, not an observed tProcessor issue event. A
    tProcessor-event comparison must explicitly supply its independently known
    path latency and set command_time_plane='tprocessor_model_event'. Trigger
    delay, transport/TMUX stages and analog cable delay are not inferred here.
    `eligible` means the RF boundary condition alone permits compilation; shared
    port spacing, capture, compensation and other compiler guards still apply.
    """
    grid = grid or RfGrid()
    if not isinstance(fine, bool):
        raise ValueError('fine must explicitly select fine or coarse duration')
    start = _integer(output_start_cycle, 'output_start_cycle')
    end = _integer(anchor_end_cycle, 'anchor_end_cycle')
    final = _integer(final_user_end_cycle, 'final_user_end_cycle')
    latency = _integer(output_latency_cycles, 'output_latency_cycles')
    if command_time_plane not in ('native_generator_input', 'tprocessor_model_event'):
        raise ValueError('Declare the command-time reference plane explicitly')
    if start < latency or end > final or not isinstance(require_within_segment, bool):
        raise ValueError('Invalid physical timeline, latency, or boundary policy')
    layout = duration_layout(grid.samples(duration_s, fine=fine), fine=fine)
    if fine and layout['samples'] > 2**31-1:
        raise ValueError('Fine duration exceeds the signed 32-bit sample register')
    active_end = start*16 + layout['samples']
    busy_end = start + layout['busy_cycles']
    violates = bool(require_within_segment and busy_end > end)
    relation = ('inside' if busy_end <= end else
                'beyond_final' if busy_end > final else 'crossing_anchor')
    return dict(**layout, **grid.frequency(frequency_hz), fine=bool(fine),
        requested_duration_s=float(duration_s), active_duration_s=layout['samples']/grid.sample_hz,
        busy_duration_s=layout['busy_cycles']/grid.fabric_hz,
        output_start_cycle=start, command_start_cycle=start-latency,
        output_latency_cycles=latency, command_time_plane=command_time_plane,
        output_start_s=start/grid.fabric_hz, active_end_sample=active_end,
        active_end_s=active_end/grid.sample_hz, busy_end_cycle=busy_end,
        busy_end_s=busy_end/grid.fabric_hz, anchor_end_cycle=end,
        final_user_end_cycle=final, required_point_end_cycle=max(final, busy_end),
        require_within_segment=require_within_segment, boundary_relation=relation,
        boundary_status='rejected' if violates else 'eligible',
        boundary_reason='Padded/busy RF extends past the anchor HOLD' if violates else '',
        additional_validation_required=['shared_port_spacing', 'acquisition', 'compensation_reset'],
        grid=asdict(grid))


def rf_waveform(point, time_s, *, amplitude_v, phase_degrees, phase_epoch_s,
                requested_start_s=None, quantized=True):
    """Ideal real baseband RF, with an explicit continuous DDS phase origin.

    amplitude_v is a declared physical voltage scale, never an uncalibrated GUI
    gain code. This is an envelope/carrier reference, not bit-exact DDS rounding
    or an analog measurement prediction. Carrier zero crossings are not pulse
    boundaries. Occupied zero padding does not enlarge the active envelope.
    """
    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or not len(time) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError('Time must be finite, one-dimensional and strictly increasing')
    amplitude = _finite(amplitude_v, 'amplitude_v', nonnegative=True)
    phase = math.radians(_finite(phase_degrees, 'phase_degrees'))
    epoch = _finite(phase_epoch_s, 'phase_epoch_s')
    if quantized:
        start, duration, frequency = point['output_start_s'], point['active_duration_s'], point['quantized_hz']
    else:
        start = _finite(requested_start_s, 'requested_start_s')
        duration, frequency = point['requested_duration_s'], point['requested_hz']
    envelope = ((time >= start) & (time < start+duration)).astype(float)
    values = amplitude * envelope * np.cos(2*np.pi*frequency*(time-epoch)+phase)
    return Reference(time, values, dict(units='V', time_origin='declared point epoch',
        reference_plane='ideal baseband RF before declared analog transfer',
        model_version='independent-rf-v1', transformations=[],
        expectation='quantized' if quantized else 'requested', amplitude_v=amplitude,
        phase_epoch_s=epoch, phase_degrees=float(phase_degrees), frequency_hz=frequency,
        active_start_s=start, active_end_s=start+duration,
        busy_end_s=point['busy_end_s'], excludes=['AWG sum', 'DAC clipping',
        'DDS integer amplitude rounding', 'Nyquist images', 'unknown analog transfer']))


def physical_reference(reference, model):
    """Apply only an explicitly calibrated flat or sampled one-pole low-pass.

    The low-pass is exact for a zero-order-held input on the declared reference
    grid. It is not an inferred oscilloscope bandwidth model. Initial output is
    explicit, and the entire capture/pre-pulse history must be supplied. A
    carrier-only calibration does not validate a short pulse's transients. The
    supplied transient evidence and uncertainty bounds must cover this pulse;
    acceptance must include these model bounds as well as instrument errors.
    """
    if not model.get('calibration_evidence') or not model.get('reference_plane'):
        raise ValueError('Physical RF prediction requires explicit calibration and reference plane')
    if not model.get('transient_model_evidence'):
        raise ValueError('Carrier calibration alone does not establish a pulse transient model')
    _finite(model['voltage_model_uncertainty_v'], 'voltage_model_uncertainty_v', nonnegative=True)
    _finite(model['timing_model_uncertainty_s'], 'timing_model_uncertainty_s', nonnegative=True)
    time, source = reference.arrays()
    low, high = map(float, model['valid_band_hz'])
    carrier = abs(float(reference.metadata['frequency_hz']))
    if not 0 <= low <= carrier <= high or not math.isfinite(high):
        raise ValueError('Carrier is outside the calibrated transfer-function band')
    kind = model.get('kind')
    gain = _finite(model['gain'], 'gain', positive=True)
    offset = _finite(model['offset_v'], 'offset_v')
    delay = _finite(model['delay_s'], 'delay_s')
    if kind == 'flat':
        values = source.copy()
    elif kind == 'first_order_lowpass_zoh':
        cutoff = _finite(model['cutoff_hz'], 'cutoff_hz', positive=True)
        initial = _finite(model['initial_output_v'], 'initial_output_v')
        dt = np.diff(time)
        if not len(dt) or not np.allclose(dt, dt[0], rtol=1e-8, atol=0.):
            raise ValueError('Low-pass reference requires a uniform native reference grid')
        values = np.empty_like(source, dtype=float)
        values[0] = initial
        alpha = math.exp(-2*math.pi*cutoff*dt[0])
        for index in range(1, len(source)):
            values[index] = alpha*values[index-1] + (1-alpha)*source[index-1]
    else:
        raise ValueError('Unknown physical transfer function; no automatic fit is available')
    metadata = dict(reference.metadata, reference_plane=model['reference_plane'],
                    expectation='physical', transformations=[dict(model)],
                    source_metadata=dict(reference.metadata),
                    voltage_model_uncertainty_v=model['voltage_model_uncertainty_v'],
                    timing_model_uncertainty_s=model['timing_model_uncertainty_s'])
    for name in ('active_start_s', 'active_end_s', 'busy_end_s', 'phase_epoch_s'):
        value = metadata.pop(name, None)
        if value is not None:
            metadata['input_' + name + '_at_reference_plane'] = value+delay
            if kind == 'flat':
                metadata[name] = value+delay
    if kind == 'first_order_lowpass_zoh':
        metadata['output_support_end_s'] = None
        metadata['support_note'] = 'Low-pass response has a decaying tail; shifted input boundaries are not output support limits'
    return Reference(time+delay, gain*values+offset, metadata)


def interval_acceptance(measured, expected, *, uncertainty, tolerance, strict=True):
    """Bounded-uncertainty decision; overlapping the limit is inconclusive."""
    measured = _finite(measured, 'measured')
    expected = _finite(expected, 'expected')
    uncertainty = _finite(uncertainty, 'uncertainty', nonnegative=True)
    tolerance = _finite(tolerance, 'tolerance', nonnegative=True)
    error = abs(measured-expected)
    lower, upper = max(0., error-uncertainty), error+uncertainty
    passed = upper < tolerance if strict else upper <= tolerance
    failed = lower >= tolerance if strict else lower > tolerance
    return dict(status='passed' if passed else 'failed' if failed else 'inconclusive',
        measured=measured, expected=expected, absolute_error=error,
        uncertainty=uncertainty, tolerance=tolerance, strict=bool(strict),
        error_interval=[lower, upper])


def _constant_period_exists(lower_times, upper_times, period_low, period_high):
    """Conservative feasibility of one phase+k*period inside all brackets.

    This is only an ambiguity gate, not a fit used to move or correct measured
    data. max(L-k*T)-min(U-k*T) is convex in T; its active slopes support a
    bounded binary search. Failure to exhibit a feasible point is inconclusive.
    """
    if period_low > period_high:
        return False
    origin = float(lower_times[0])
    lower, upper = np.asarray(lower_times)-origin, np.asarray(upper_times)-origin
    indices = np.arange(len(lower), dtype=float)
    candidates = [float(period_low), float(period_high)]
    for iteration in range(66):
        period = candidates[iteration] if iteration < 2 else (period_low+period_high)/2
        lower_phases, upper_phases = lower-indices*period, upper-indices*period
        lower_index, upper_index = int(np.argmax(lower_phases)), int(np.argmin(upper_phases))
        if lower_phases[lower_index] <= upper_phases[upper_index]:
            return True
        if iteration < 2:
            continue
        derivative = upper_index-lower_index
        if derivative == 0:
            return False
        if derivative > 0:
            period_high = period
        else:
            period_low = period
    return False


def frequency_measurement(time_s, voltage_v, *, expected_hz, tolerance_hz,
                          baseline_v, bandwidth_hz, allowed_band_hz,
                          voltage_uncertainty_v, edge_uncertainty_s,
                          timebase_uncertainty_ppm, minimum_cycles=10,
                          minimum_samples_per_cycle=8, clipped=False):
    """Native-waveform rising-crossing estimate inside an independently set ROI.

    The supplied allowed band must exclude aliases using the fixture/filter,
    not the expected tone alone. The caller must isolate settled active RF and
    declare voltage/timing/timebase bounds. No FFT-bin/phase/amplitude fit or
    expected-frequency search is used. This measures frequency magnitude only.
    """
    time, values = np.asarray(time_s, dtype=float), np.asarray(voltage_v, dtype=float)
    expected = abs(_finite(expected_hz, 'expected_hz'))
    tolerance = _finite(tolerance_hz, 'tolerance_hz', nonnegative=True)
    baseline = _finite(baseline_v, 'baseline_v')
    bandwidth = _finite(bandwidth_hz, 'bandwidth_hz', positive=True)
    noise = _finite(voltage_uncertainty_v, 'voltage_uncertainty_v', nonnegative=True)
    edge_bound = _finite(edge_uncertainty_s, 'edge_uncertainty_s', nonnegative=True)
    ppm = _finite(timebase_uncertainty_ppm, 'timebase_uncertainty_ppm', nonnegative=True)
    low, high = map(float, allowed_band_hz)
    minimum_cycles = _integer(minimum_cycles, 'minimum_cycles', 2)
    minimum_samples_per_cycle = _integer(minimum_samples_per_cycle, 'minimum_samples_per_cycle', 3)
    result = dict(status='inconclusive', expected_hz=expected, tolerance_hz=tolerance,
                  method='native rising crossings with bounded timing uncertainty')
    def unavailable(reason):
        return dict(result, reason=reason)
    if time.ndim != 1 or values.shape != time.shape or len(time) < 3 or not np.isfinite(time).all() or not np.isfinite(values).all():
        return unavailable('Missing or nonfinite native waveform')
    dt = np.diff(time)
    if np.any(dt <= 0) or not np.allclose(dt, dt[0], rtol=1e-6, atol=0.):
        return unavailable('Nonuniform or unordered native waveform')
    if clipped:
        return unavailable('Clipped or invalid scope codes')
    if not math.isfinite(low) or not math.isfinite(high) or low <= 0 or high < low or not low <= expected <= high:
        return unavailable('Expected frequency is outside the declared unambiguous positive band')
    if high > bandwidth or high*dt[0]*minimum_samples_per_cycle > 1:
        return unavailable('Insufficient analog bandwidth or native sampling for the whole declared band')
    values = values-baseline
    if np.min(values) >= -noise or np.max(values) <= noise:
        return unavailable('Carrier does not cross the fixed baseline beyond voltage uncertainty')
    indices = np.flatnonzero((values[:-1] <= 0) & (values[1:] > 0))
    if len(indices) < minimum_cycles+1:
        return unavailable('Too few complete carrier cycles')
    # Each observed cycle must prove both signs outside the bounded voltage
    # error. A global peak is insufficient: noise can supply an expected tone
    # while the actual weak carrier has another frequency. Reject extra or
    # missing crossing candidates whose existence is uncertain.
    cycle_values = values[:indices[-1]+1]
    minima = np.minimum.reduceat(cycle_values, indices[:-1])
    maxima = np.maximum.reduceat(cycle_values, indices[:-1])
    if np.any(minima >= -noise) or np.any(maxima <= noise):
        return unavailable('Individual carrier cycles are unresolved within voltage uncertainty')
    slopes = (values[indices+1]-values[indices])/dt[0]
    crossings = time[indices] - values[indices]/slopes  # Estimate only, never its own error bound.
    negative = np.flatnonzero(values < -noise)
    positive = np.flatnonzero(values > noise)
    before = np.searchsorted(negative, indices, side='right')-1
    after = np.searchsorted(positive, indices+1, side='left')
    if np.any(before < 0) or np.any(after == len(positive)):
        return unavailable('Endpoint crossings lack certain negative/positive voltage brackets')
    # Guaranteed signs bracket a true crossing without estimating the unknown
    # signal slope from noisy samples. An adversarial bounded voltage error can
    # steepen a measured slope; noise/slope is therefore not a valid time bound.
    lower_times = time[negative[before]]-edge_bound
    upper_times = time[positive[after]]+edge_bound
    if np.max(upper_times-lower_times) >= .5/high or np.any(upper_times[:-1] >= lower_times[1:]):
        return unavailable('Crossing uncertainty cannot distinguish adjacent carrier cycles')
    period_lower = lower_times[1:]-upper_times[:-1]
    period_upper = upper_times[1:]-lower_times[:-1]
    if np.any(period_upper < 1/high) or np.any(period_lower > 1/low):
        return unavailable('Crossings are ambiguous or inconsistent with the declared frequency band')
    if np.max(period_lower) > np.min(period_upper):
        return unavailable('No constant carrier period is consistent with all crossing brackets')
    if not _constant_period_exists(lower_times, upper_times,
            max(float(np.max(period_lower)), 1/high),
            min(float(np.min(period_upper)), 1/low)):
        return unavailable('No common carrier period and phase is consistent with the entire crossing record')
    span = float(crossings[-1]-crossings[0])
    span_lower = float(lower_times[-1]-upper_times[0])*(1-ppm*1e-6)
    span_upper = float(upper_times[-1]-lower_times[0])*(1+ppm*1e-6)
    if span_lower <= 0:
        return unavailable('Frequency uncertainty is unbounded at this observation length')
    cycles = len(crossings)-1
    measured = cycles/span
    bounds = (cycles/span_upper, cycles/span_lower)
    uncertainty = max(measured-bounds[0], bounds[1]-measured)
    result.update(interval_acceptance(measured, expected, uncertainty=uncertainty,
                                     tolerance=tolerance), measured_hz=measured,
                  frequency_interval_hz=list(bounds), complete_cycles=cycles,
                  native_sample_interval_s=float(dt[0]), crossing_times_s=crossings.tolist(),
                  crossing_brackets_s=np.column_stack((lower_times, upper_times)).tolist(),
                  baseline_v=baseline, allowed_band_hz=[low, high])
    return result


def boundary_case_plan(grid=None):
    """Concrete deterministic cases for GUI/compiler adapters, never executions.

    Frequencies and voltage scales must be added from the actual fixture. The
    final two rows prove that only later points may violate a fixed HOLD. False
    policy permits extension; it does not extend the anchor itself.
    """
    grid = grid or RfGrid()
    rows = []
    for fine, counts in ((True, (1, 15, 16, 17, 31, 32, 95, 96, 97,
                                16*65538-1, 16*65538, 16*65539)),
                         (False, (48, 64, 80))):
        rows.append(dict(name='fine_implementation_boundaries' if fine else 'coarse_boundaries',
            duration_mode='fine' if fine else 'coarse', samples=list(counts),
            duration_s=[n/grid.sample_hz for n in counts], execution='individual_gui',
            note='Uneven implementation grid; not a native arithmetic sweep'))
    for policy in (True, False):
        for name, start, durations in (('inside', 30, [1, 31]),
                                       ('crossing_anchor', 58, [1, 49]),
                                       ('beyond_final', 88, [1, 49]),
                                       ('later_point_violation', 55, [1, 80, 81])):
            points = [rf_point(duration_s=n/grid.sample_hz, frequency_hz=0,
                output_start_cycle=start+31, anchor_end_cycle=60+31,
                final_user_end_cycle=90+31, require_within_segment=policy,
                output_latency_cycles=31, command_time_plane='native_generator_input',
                grid=grid) for n in durations]
            rows.append(dict(name=f'{name}_{str(policy).lower()}', duration_mode='fine',
                samples=durations, require_within_segment=policy, points=points,
                expected_boundary_statuses=[p['boundary_status'] for p in points],
                execution='individual_gui' if len(durations) == 3 else 'hardware_loop',
                note='Boundary-only decision; production compiler/GUI/physical result must be recorded separately'))
    return rows


def frequency_case_plan(start_hz, stop_hz, count, *, execution, grid=None):
    """A declared frequency-axis specification, including descending and N=1.

    The caller must separately provide the RF pulse, calibrated voltage scale,
    sufficiently long active interval and the GUI binding. Merely generating
    this plan proves neither that a hardware loop exists nor that it ran.
    """
    if execution not in ('hardware_loop', 'individual_gui'):
        raise ValueError('Declare actual intended frequency sweep execution semantics')
    count = _integer(count, 'count', 1)
    if count > 1_000_000:
        raise ValueError('Frequency case plan exceeds one million declared points')
    start = _finite(start_hz, 'start_hz')
    stop = _finite(stop_hz, 'stop_hz')
    grid = grid or RfGrid()
    coordinates = np.linspace(start, stop, count)
    return dict(name='rf_frequency', execution=execution, start_hz=start,
        stop_hz=stop, count=count, points=[dict(index=i, **grid.frequency(f))
                                       for i, f in enumerate(coordinates)],
        units='Hz', physical_execution=False,
        prerequisites=['explicit GUI pulse and frequency controls',
                       'calibrated RF amplitude versus frequency',
                       'unambiguous carrier band and adequate native sampling',
                       'enough complete cycles at every point'])
