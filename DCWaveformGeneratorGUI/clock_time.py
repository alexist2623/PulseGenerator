"""Clock-grid conversion without relative tolerances that grow with duration."""
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP, localcontext
from math import ceil, isfinite, ulp


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
