"""Independent v5 area balancing and RC inverse-filter expectations.

This module consumes declared cases and declared scheduling, never production
command words or measured edges. The parent baseband reference supplies the
independently quantized main waveform. Both main and compensation are expressed
at the area engine output / IIR input; common downstream latency is removed.
Thus ``dc_first_cycle`` is the first compensation word, not its command time.
For this v5 path it is 12 cycles later than an ordinary SET at the same command
timestamp (17-cycle DC startup minus the 5-cycle ordinary area pipeline).

Readout/marker/RF barriers must be resolved by the case specification before
calling this model. They are deliberately not inferred from observed waveforms.
The result is a digital DAC expectation; analog transfer needs a separately
declared fixed calibration, as in awg_reference.physical_references.
"""
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from .awg_reference import AwgReference, ReferenceUnavailable


@dataclass(frozen=True)
class CompensationSchedule:
    """Fabric-cycle schedule relative to the first main SET at the IIR input.

    All durations are exclusive-end intervals. ``main_end_cycle`` must cover
    the entire independently generated main waveform, including its zero HOLD.
    ``dc_first_cycle`` may be one integer or two per-output integers. The gap
    after main_end is zero input, but RC history persists during that gap.
    DC completion emits one zero word, then clears area and IIR at the following
    word. RC-only cases instead require explicit per-output ``reset_cycles``.
    Capture must include at least one complete cleared word after both resets.
    These declarations are requirements, not evidence that hardware met them.
    """
    main_end_cycle: int
    capture_end_cycle: int
    initial_state: str
    barrier_policy: str
    readout_policy: str
    reset_policy: str
    dc_first_cycle: object = None
    reset_cycles: object = None


def _cycle(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + ' must be a nonnegative integer fabric cycle')
    return value


def _pair(value, name):
    if type(value) is int:
        value = (value, value)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ReferenceUnavailable(name + ' must explicitly specify both outputs')
    return tuple(_cycle(v, name) for v in value)


def _positive(value, name):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(name + ' must be finite and positive')
    return Fraction(str(value))


def area_balance(area, *, mode, words=None, amplitude_code=None, max_samples=4_800_000):
    """Effective 14-bit scalar DC codes with exact integer zero total area.

    Fixed time emits N=16*words samples: the first remainder samples use q+1,
    followed by q. Fixed voltage emits full-amplitude samples, one possible
    remainder sample, then zeros to a complete word. Zero area still emits a
    complete word (or the requested fixed-time duration).
    """
    if type(area) is not int or not -(1 << 47) <= area < (1 << 47):
        raise ValueError('Area must fit the signed 48-bit ledger')
    sign = -1 if area > 0 else 1
    magnitude = abs(area)
    if mode == 'fixed_time':
        if type(words) is not int or not 1 <= words < (1 << 32):
            raise ValueError('Fixed time requires a positive 32-bit word count')
        count = 16 * words
        quotient, remainder = divmod(magnitude, count)
        largest = sign * (quotient + bool(remainder))
        if not -8192 <= largest <= 8191:
            raise ValueError('Compensation exceeds the effective DAC range')
        if count > max_samples:
            raise ValueError('Compensation exceeds the sample allocation limit')
        codes = np.full(count, sign * quotient, dtype=np.int16)
        codes[:remainder] = sign * (quotient + 1)
    elif mode == 'fixed_voltage':
        if type(amplitude_code) is not int or not 1 <= amplitude_code <= 8191:
            raise ValueError('Fixed voltage requires a positive effective amplitude')
        full, remainder = divmod(magnitude, amplitude_code)
        count = max(16, ((full + bool(remainder) + 15) // 16) * 16)
        if count > max_samples:
            raise ValueError('Compensation exceeds the sample allocation limit')
        codes = np.zeros(count, dtype=np.int16)
        codes[:full] = sign * amplitude_code
        if remainder:
            codes[full] = sign * remainder
    else:
        raise ReferenceUnavailable('Unsupported DC mode')
    if int(codes.astype(np.int64).sum()) + area != 0:
        raise AssertionError('Independent integer area balancing failed')
    return codes


def rc_inverse(codes, coefficient, *, reset_samples=()):
    """Q48 trapezoidal integration with the v5 effective-DAC rounding rule.

    Input and result are effective 14-bit codes. A clear resets both the integral
    and previous scalar sample before that sample is processed. Rounding ties
    go toward positive infinity. Clipping is reported, never hidden. Accumulator
    overflow is unavailable here rather than approximating the RTL's word-prefix
    saturation with a different scalar saturation algorithm.
    """
    values = np.asarray(codes)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError('RC input must be a one-dimensional integer code array')
    if np.any(values < -8192) or np.any(values > 8191):
        raise ValueError('RC input is outside the effective DAC range')
    if type(coefficient) is not int or not 1 <= coefficient < (1 << 32):
        raise ValueError('RC coefficient must fit a positive unsigned 32-bit word')
    resets = set(reset_samples)
    if any(type(i) is not int or not 0 <= i < len(values) for i in resets):
        raise ValueError('RC clear is outside the declared waveform')
    result = np.empty(len(values), dtype=np.int16)
    integral = previous = 0
    clipped = []
    reset_states = []
    for index, value in enumerate(values):
        if index in resets:
            reset_states.append(dict(sample=index, integral_before=integral,
                                     previous_transport_before=previous))
            integral = previous = 0
        transport = int(value) * 4
        integral += coefficient * (transport + previous)
        corrected = (transport << 48) + integral
        if not -(1 << 71) <= integral < (1 << 71) or not -(1 << 71) <= corrected < (1 << 71):
            raise ReferenceUnavailable('RC accumulator saturation needs a word-prefix reference')
        rounded = (corrected + (1 << 49)) >> 50
        if corrected > (32764 << 48) or corrected < (-32768 << 48):
            clipped.append(index)
        result[index] = max(-8192, min(8191, rounded))
        previous = transport
    return result, dict(coefficient=coefficient, clipped_samples=clipped,
                        integral_final=integral, previous_transport_final=previous,
                        reset_states=reset_states)


class CompensationReference:
    """Explicitly scheduled DC/RC extension of the independent baseband case.

    Compensation settings use the common case schema: dc_enabled, dc_mode,
    dc_duration_us or dc_voltage_mv, rc_enabled and rc_tau_us.
    No automatic output, compiler call, instrument access, or fitting occurs.
    """
    def __init__(self, baseband):
        if not isinstance(baseband, AwgReference):
            raise TypeError('An independent AwgReference is required')
        self.baseband = baseband
        self.settings = deepcopy(baseband.spec.get('compensation', {}))
        if not (self.settings.get('dc_enabled') or self.settings.get('rc_enabled')):
            raise ValueError('Use AwgReference directly when compensation is off')
        if baseband.spec.get('rf') or baseband.spec.get('rf_pulses'):
            raise ReferenceUnavailable('RF composition requires its own reference')

    def waveforms(self, point, schedule, *, max_samples=4_800_000):
        if not isinstance(schedule, CompensationSchedule):
            raise ReferenceUnavailable('An explicit compensation schedule is required')
        if schedule.initial_state != 'zero_area_zero_iir':
            raise ReferenceUnavailable('Independent initial zero area and IIR state are required')
        if schedule.barrier_policy not in ('explicit', 'sweep_maximum'):
            raise ReferenceUnavailable('The case must declare its point-end barrier policy')
        if schedule.readout_policy not in ('none', 'after_declared_barrier'):
            raise ReferenceUnavailable('Readout overlap or an unknown barrier is unsupported')
        main_end = _cycle(schedule.main_end_cycle, 'main_end_cycle')
        capture_end = _cycle(schedule.capture_end_cycle, 'capture_end_cycle')
        if capture_end * 16 > max_samples:
            raise ValueError('Reference exceeds the sample allocation limit')
        plain = deepcopy(self.baseband.spec)
        plain['compensation'] = {}
        main = AwgReference(plain).waveforms(point, max_samples=max_samples)
        if main_end * 16 != len(main['quantized_codes']):
            raise ValueError('main_end_cycle disagrees with the declared complete main waveform')
        if np.any(main['quantized_codes'][-1]):
            raise ReferenceUnavailable('Compensation requires an explicitly zero final main stage')
        if capture_end < main_end:
            raise ReferenceUnavailable('Capture does not cover the complete main waveform')
        raw = np.zeros((capture_end * 16, 2), dtype=np.int16)
        raw[:main_end * 16] = main['quantized_codes']
        area = np.sum(raw.astype(np.int64), axis=0)
        prefixes = np.cumsum(raw.astype(np.int64), axis=0)
        if np.any(prefixes < -(1 << 47)) or np.any(prefixes >= (1 << 47)):
            raise ValueError('Main waveform exceeds the 48-bit area prefix range')
        del prefixes
        settings = self.settings
        details = []
        if settings.get('dc_enabled'):
            if schedule.reset_policy != 'dc_zero_edge_then_clear' or schedule.reset_cycles is not None:
                raise ReferenceUnavailable('DC must use its internal zero-edge then clear policy')
            starts = _pair(schedule.dc_first_cycle, 'dc_first_cycle')
            resets = []
            for output in range(2):
                if starts[output] < main_end + 12:
                    raise ValueError('DC startup must follow drained zero main output by at least 12 cycles')
                mode = settings.get('dc_mode')
                words = amplitude = None
                if mode == 'fixed_time':
                    words = round(_positive(settings['dc_duration_us'], 'dc_duration_us') * 300)
                elif mode == 'fixed_voltage':
                    amplitude = round(_positive(settings['dc_voltage_mv'], 'dc_voltage_mv')
                                      * 8192 / Fraction(str(self.baseband.scales[output])))
                pulse = area_balance(int(area[output]), mode=mode, words=words,
                                     amplitude_code=amplitude, max_samples=max_samples)
                end = starts[output] + len(pulse) // 16
                reset = end + 1  # One final zero word is integrated before clear.
                if capture_end < reset + 1:
                    raise ReferenceUnavailable('Capture does not cover DC completion and a cleared word')
                raw[starts[output] * 16:end * 16, output] = pulse
                resets.append(reset)
                details.append(dict(output=output, main_area=int(area[output]), dc_first_cycle=starts[output],
                                    dc_words=len(pulse) // 16, dc_end_cycle=end, reset_cycle=reset,
                                    residual_area=int(raw[:, output].astype(np.int64).sum()),
                                    amplitude_code=amplitude, fixed_time_words=words))
        else:
            if schedule.dc_first_cycle is not None or schedule.reset_policy != 'explicit_rc_clear':
                raise ReferenceUnavailable('RC-only cases require an explicitly scheduled clear')
            resets = _pair(schedule.reset_cycles, 'reset_cycles')
            if any(reset < main_end or capture_end < reset + 1 for reset in resets):
                raise ReferenceUnavailable('Capture must cover the zero main end and explicit RC clear')
            details = [dict(output=i, main_area=int(area[i]), reset_cycle=resets[i]) for i in range(2)]
        corrected = raw.copy()
        if settings.get('rc_enabled'):
            tau = _positive(settings['rc_tau_us'], 'rc_tau_us')
            coefficient = round(Fraction(1 << 48) / (2 * tau * 4800))
            for output in range(2):
                corrected[:, output], info = rc_inverse(raw[:, output], coefficient,
                                                       reset_samples=(resets[output] * 16,))
                details[output]['rc'] = info
        time = np.arange(len(raw), dtype=float) / 4.8e9
        requested_virtual = np.zeros((len(raw), 2), dtype=float)
        requested_physical = np.zeros((len(raw), 2), dtype=float)
        requested_virtual[:main_end * 16] = main['requested_virtual_v']
        requested_physical[:main_end * 16] = main['requested_physical_v']
        return dict(model='v5-declared-dc-rc-reference-v1', point=point,
                    group_indices=self.baseband.coordinates(point), time_s=time,
                    requested_main=main, requested_virtual_v=requested_virtual,
                    requested_physical_v=requested_physical,
                    requested_reference_note='Requested main waveform, zero after main; automatic compensation is separately represented by the quantized/physical reference.',
                    pre_rc_codes=raw, quantized_codes=corrected,
                    quantized_physical_v=corrected * self.baseband.scales / 8192000.,
                    channels=details, schedule=deepcopy(schedule.__dict__),
                    full_reference_coverage=True,
                    time_origin='First main SET at the IIR input; common downstream latency removed.',
                    reference_plane='Digital AWG at final DAC, RF absent', physical_reference_required=True)
