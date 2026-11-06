"""Per-generator AWG limits in mV; no hardware access while editing/compiling."""
from copy import deepcopy
from fractions import Fraction
import math


def normalize_output_limits(settings):
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise ValueError("AWG output limits must be a mapping by generator index")
    result = {}
    for key, value in settings.items():
        try:
            channel = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid AWG limit generator index") from exc
        if isinstance(key, bool) or str(channel) != str(key) or channel < 0:
            raise ValueError("Invalid AWG limit generator index")
        if not isinstance(value, dict) or not isinstance(value.get('enabled'), bool):
            raise ValueError("AWG limit enabled must be boolean")
        minimum, maximum = float(value['min_mv']), float(value['max_mv'])
        if not (math.isfinite(minimum) and math.isfinite(maximum)
                and minimum <= 0 <= maximum):
            raise ValueError("AWG limits must be finite and include 0 mV for stop/idle")
        result[str(channel)] = dict(enabled=value['enabled'], min_mv=minimum, max_mv=maximum)
    return result


def compile_output_limits(soccfg, sequence, channels):
    """Round bounds inward to effective DAC codes using each DAC's current scale."""
    settings = normalize_output_limits(getattr(sequence, 'output_clip_limits', {}))
    scales = getattr(sequence, 'output_full_scales_mv', None)
    if scales is None:
        scales = [getattr(sequence, 'output_full_scale_mv', 800.0)] * len(channels)
    if len(scales) != len(channels):
        raise ValueError("AWG output-limit voltage scales do not match selected outputs")
    result = []
    for index, channel in enumerate(channels):
        gen = soccfg['gens'][channel]
        value = settings.get(str(channel), dict(enabled=False, min_mv=0.0, max_mv=0.0))
        capable = int(gen.get('output_clip_version', 0)) == 1
        if value['enabled'] and not capable:
            raise ValueError(f"AWG gen {channel}: output limits require v8 BIT/HWH and driver")
        if not capable:
            continue
        scale = float(scales[index])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid per-DAC voltage scale for AWG output limits")
        minimum, maximum = -32768, 32764
        if value['enabled']:
            if value['min_mv'] < -scale or value['max_mv'] > scale:
                raise ValueError(f"AWG gen {channel}: limits exceed the DAC voltage range +/-{scale:g} mV")
            factor = Fraction(32768, 4) / Fraction(str(scale))
            minimum = max(-32768, 4 * math.ceil(Fraction(str(value['min_mv'])) * factor))
            maximum = min(32764, 4 * math.floor(Fraction(str(value['max_mv'])) * factor))
        result.append(dict(ch=int(channel), enabled=value['enabled'], min_code=minimum,
                           max_code=maximum, requested=deepcopy(value), full_scale_mv=scale,
                           effective_min_mv=minimum * scale / 32768,
                           effective_max_mv=maximum * scale / 32768))
    return result


def apply_output_limits(soc, compiled):
    if not compiled:
        return []
    setter = getattr(soc, 'set_awg_output_limits', None)
    if not callable(setter):
        raise RuntimeError("QICK server driver does not support v8 AWG output limits")
    results = []
    for value in compiled:
        actual = dict(setter(ch=value['ch'], min_code=value['min_code'],
                             max_code=value['max_code'], enabled=value['enabled']))
        if (actual.get('pending') or actual.get('configuration_error')
                or any(actual.get(k) != value[k] for k in ('enabled', 'min_code', 'max_code'))):
            raise RuntimeError(f"AWG gen {value['ch']}: output-limit readback mismatch")
        results.append(dict(value, readback=actual))
    return results
