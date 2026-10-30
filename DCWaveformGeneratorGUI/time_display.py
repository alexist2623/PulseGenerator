"""Display time without changing stored coordinates or pulse timing."""

TIME_SECONDS = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6, 'µs': 1e-6,
                'μs': 1e-6, 'ns': 1e-9, 'ps': 1e-12}


def set_plot_axis_units(axis, text, unit):
    """Let PyQtGraph prefix seconds, never an already-prefixed unit like us.

    Curve/image coordinates stay in their original unit. Axis.setScale only
    converts the tick labels. Reset it when the same plot switches to voltage.
    """
    axis.setScale(TIME_SECONDS.get(unit, 1.0))
    axis.setLabel(text=text, units='s' if unit in TIME_SECONDS else unit)
    axis.enableAutoSIPrefix(True)


def format_time_us(value):
    value = float(value)
    magnitude = abs(value)
    scale, unit = ((1000.0, 'ns') if magnitude < 1 else
                   (1.0, 'us') if magnitude < 1000 else
                   (0.001, 'ms') if magnitude < 1e6 else (1e-6, 's'))
    return f'{value * scale:.9g} {unit}'


def time_sweep_display(start_us, stop_us, count, *, fabric_mhz=None, samples_per_clock=1, integer_step=False, allow_constant=False):
    """Describe requested points, and exact AWG clock steps when available."""
    start, stop, count = float(start_us), float(stop_us), int(count)
    if count < 1:
        raise ValueError('Sweep count must be positive')
    if fabric_mhz is None:
        step = (stop - start) / (count - 1) if count > 1 else 0.0
        end = stop if count > 1 else start
        return dict(summary=(f'Start {format_time_us(start)}; end {format_time_us(end)}; '
                             f'requested step {format_time_us(step)}; {count} point(s)'),
                    step=format_time_us(step) + ' (requested)', end='Not quantized here')
    # Use the compiler's integer rounding rule, without constructing N points.
    try:
        from .qick_fine_tune_sweep import cycles_from_us, _round_div_nearest
    except ImportError:
        from qick_fine_tune_sweep import cycles_from_us, _round_div_nearest
    rate = fabric_mhz * samples_per_clock
    first = cycles_from_us(start, rate)
    last = cycles_from_us(stop, rate)
    step = _round_div_nearest(last - first, count - 1) if count > 1 else 0
    applied_last = last if samples_per_clock == 16 and count > 1 and not integer_step else first + (count - 1) * step
    if first < 1 or applied_last < 1 or (count > 1 and step == 0 and not allow_constant):
        return dict(summary='Invalid sweep: distinct positive clock steps required',
                    step='Invalid time sweep', end='Invalid time sweep')
    end = (first + (count - 1) * step) / rate
    grid = 'DAC sample(s)' if samples_per_clock == 16 else 'clock(s)'
    step_text = f'{format_time_us(step / rate)} ({step} {grid})'
    if samples_per_clock == 16 and count > 1 and not integer_step:
        import math
        average = (last - first) / (count - 1)
        if abs(last - first) < count - 1:
            return dict(summary='Invalid sweep: distinct positive DAC samples required',
                        step='Invalid time sweep', end='Invalid time sweep')
        end = last / rate
        lower, upper = math.floor(average), math.ceil(average)
        if lower != upper:
            step_text = (f'{format_time_us(lower/rate)} / {format_time_us(upper/rate)} '
                         f'({lower}/{upper} DAC sample(s), rounded linear grid)')
    return dict(summary=(f'Start {format_time_us(first / rate)}; '
                         f'requested end {format_time_us(stop)}; applied end {format_time_us(end)}; '
                         f'step {step_text}; {count} point(s)'),
                step=step_text, end=format_time_us(end))


def time_sweep_summary(*args, **kwargs):
    return time_sweep_display(*args, **kwargs)['summary']
