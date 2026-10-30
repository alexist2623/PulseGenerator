"""Clock-grid conversion without relative tolerances that grow with duration."""
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP, localcontext
from math import ceil, isfinite, ulp


def integer_sample_sweep(start_us, stop_us, count, sample_mhz, *, allow_constant=False):
    """Constant integer increment; report the applied endpoint, never drift it silently."""
    count = int(count)
    if count < 1:
        raise ValueError('Sweep count must be positive')
    first = clock_cycles(start_us, sample_mhz)
    last = clock_cycles(stop_us, sample_mhz)
    delta, divisor = last - first, max(1, count - 1)
    step = (1 if delta >= 0 else -1) * ((2 * abs(delta) + divisor) // (2 * divisor)) if count > 1 else 0
    applied = first + (count - 1) * step
    if max(first, applied) >= 1 << 31:
        raise ValueError('Sweep sample duration must fit a positive signed 32-bit register')
    if applied < 1 or (count > 1 and step == 0 and not allow_constant):
        raise ValueError('Sweep requires distinct positive DAC sample durations; adjust the range or point count')
    return first, step, applied


def clock_cycles(value, rate_mhz, *, units_per_us=1.0, display_quantum=None):
    """Ceil a positive time, recovering clock points rounded by GUI decimals.

    The default serialization precision is nine decimal places in us. The
    tolerance covers half that last digit plus float representation error,
    never a relative fraction of the whole duration. Large values use decimal
    arithmetic so multiplying by a clock does not discard a fractional cycle.
    """
    value, rate, units = float(value), float(rate_mhz), float(units_per_us)
    if not all(isfinite(x) and x > 0 for x in (value, rate, units)):
        raise ValueError('duration, clock rate and unit scale must be finite and positive')
    quantum = 1e-9 * units if display_quantum is None else float(display_quantum)
    if not isfinite(quantum) or quantum < 0:
        raise ValueError('display quantum must be finite and nonnegative')
    exact = value * (rate / units)
    if not isfinite(exact):
        raise ValueError('duration exceeds the supported clock range')
    tolerance = min(.125, max(quantum * rate / units / 2,
                             ulp(value) * rate / units / 2 + ulp(exact)))
    if exact < 1e9:
        nearest = round(exact)
        return max(1, int(nearest if abs(exact-nearest) <= tolerance else ceil(exact)))
    with localcontext() as context:
        context.prec = 60
        ticks = Decimal(str(value)) * Decimal(str(rate)) / Decimal(str(units))
        nearest = ticks.to_integral_value(rounding=ROUND_HALF_UP)
        # Decimal multiplication introduces no float-product ULP error.
        tolerance = min(.125, max(quantum*rate/units/2, ulp(value)*rate/units/2))
        result = nearest if abs(ticks-nearest) <= Decimal(str(tolerance)) else ticks.to_integral_value(rounding=ROUND_CEILING)
        return max(1, int(result))


def clock_time(value, rate_mhz, *, units_per_us=1.0, display_quantum=None):
    """Canonical float for an integer number of clocks; preserve zero delays."""
    if float(value) == 0:
        return 0.0
    ticks = clock_cycles(value, rate_mhz, units_per_us=units_per_us,
                         display_quantum=display_quantum)
    return ticks / float(rate_mhz) * float(units_per_us)


def clock_time_text(value, rate_mhz, *, units_per_us=1.0, display_quantum=None):
    """Display 300 MHz ticks as 10n, 10n+3.3 or 10n+6.6 ns.

    Only the label is shortened. Values and serialized times retain the exact
    clock fraction. Imported times off this clock grid keep their precision.
    """
    value = float(value)
    if value == 0:
        return "0"
    if value > 0 and float(rate_mhz) == 300.:
        ticks = clock_cycles(value, rate_mhz, units_per_us=units_per_us,
                             display_quantum=display_quantum)
        canonical = ticks / 300. * float(units_per_us)
        quantum = 1e-9 * float(units_per_us) if display_quantum is None else float(display_quantum)
        if abs(value - canonical) <= max(quantum / 2, 2 * ulp(value)):
            blocks, remainder = divmod(ticks, 3)
            tenths_ns = blocks * 100 + (0, 33, 66)[remainder]
            with localcontext() as context:
                context.prec = 60
                shown = Decimal(tenths_ns) * Decimal(str(units_per_us)) / Decimal(10000)
                text = format(shown, 'f')
            return text.rstrip('0').rstrip('.') if '.' in text else text
    return str(int(value)) if value.is_integer() else repr(value)
