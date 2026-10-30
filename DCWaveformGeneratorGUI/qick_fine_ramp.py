"""Sample-position RAMP editing and v7 packet integration (no device I/O).

The ordinary SET/RAMP timeline reserves whole fabric words. Fine edges live
inside that reservation; the following SET is the terminal voltage. This
keeps adjacent commands and RF/readout timestamps on the same time base.
"""
from numbers import Integral
import math
import numpy as np


def needs_sample_grid(start_us, stop_us, count, fabric_mhz):
    """Keep whole-clock legacy sweeps; use v7 for fractional-clock edges."""
    first = float(start_us) * float(fabric_mhz)
    last = float(stop_us) * float(fabric_mhz)
    step = (last - first) / (int(count) - 1) if int(count) > 1 else 0.
    return any(abs(value - round(value)) > 1e-6 for value in (first, last, step))


def sweep_target(segment_name):
    """Separate a fine-packet component from its stable RAMP segment name."""
    base, separator, component = str(segment_name).partition(':')
    if separator and component not in ('fall', 'hold'):
        raise ValueError(f'Unknown fine RAMP sweep component {component!r}')
    return base, component if separator else 'rise'


def component_axes(sequence, segment_name):
    return tuple((i, axis, sweep_target(axis.segment_name)[1])
                 for i, axis in enumerate(sequence.sweeps)
                 if axis.axis_kind == 'ramp_duration'
                 and sweep_target(axis.segment_name)[0] == segment_name
                 and axis.samples_per_clock == 16)


def _with_lengths(spec, lengths):
    spec = dict(spec or dict(rise_start=0, rise_end=lengths['rise']))
    rise = lengths.get('rise', spec['rise_end'] - spec['rise_start'])
    if 'fall_start' in spec:
        hold = lengths.get('hold', spec['fall_start'] - spec['rise_end'])
        fall = lengths.get('fall', spec['fall_end'] - spec['fall_start'])
        spec['fall_start'] = spec['rise_start'] + rise + hold
        spec['fall_end'] = spec['fall_start'] + fall
    elif set(lengths) - {'rise'}:
        raise ValueError('Internal hold/falling sweeps require a two-edge fine RAMP on every output')
    spec['rise_end'] = spec['rise_start'] + rise
    return spec


def sweep_spec_at(sequence, segment_name, output, point_index):
    """Apply independent or grouped component lengths to one fine packet."""
    spec = sequence.fine_ramps.get(segment_name, (None,) * sequence.n_outputs)[output]
    lengths = {component: int(axis.duration_sample_points[sequence.point_indices(point_index)[index]])
               for index, axis, component in component_axes(sequence, segment_name)}
    return _with_lengths(spec, lengths) if lengths else spec


def sweep_reservation_cycles(sequence, segment, axis):
    """One fixed command window contains every edge in this hardware sweep.

    The window is separate from edge duration. The packet's plateau and second
    edge move with the first edge, so their durations do not acquire padding.
    """
    lengths = {component: max(a.duration_sample_points[0], a.duration_sample_points[-1])
               for _, a, component in component_axes(sequence, segment.name)}
    last = 0
    for spec in sequence.fine_ramps.get(segment.name, ()):
        moved = _with_lengths(spec, lengths)
        last = max(last, moved.get('fall_end', moved['rise_end']))
    return max(segment.duration_cycles, (int(last) + 16) // 16)


def validate_edges(spec, capacity):
    allowed = {'rise_start', 'rise_end', 'fall_start', 'fall_end', 'turn'}
    if set(spec) - allowed:
        raise ValueError('Unknown fine RAMP fields')
    def index(name, low, high):
        value = spec.get(name)
        if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer sample index in [{low}, {high}]')
        return int(value)
    r0 = index('rise_start', 0, 15)
    r1 = index('rise_end', r0, min(int(capacity), (1 << 23) - 1) - 1)
    result = dict(rise_start=r0, rise_end=r1)
    if spec.get('fall_start') is not None:
        f0 = index('fall_start', r1, min(16383, int(capacity) - 1))
        f1 = index('fall_end', f0, min(int(capacity), (1 << 23) - 1) - 1)
        turn = float(spec['turn'])
        if not math.isfinite(turn):
            raise ValueError('Fine RAMP turn voltage must be finite')
        result.update(fall_start=f0, fall_end=f1, turn=turn)
    elif spec.get('fall_end') is not None or spec.get('turn') is not None:
        raise ValueError('Disable the second edge by omitting its end and turn voltage')
    return result


def vertices(spec, start, terminal, capacity):
    """Ideal vertices in scalar-sample units; no per-point time rounding."""
    s = validate_edges(spec, capacity)
    turn = s.get('turn', terminal)
    if s['rise_start'] == s['rise_end'] and start != turn:
        raise ValueError('A nonzero first edge needs at least one sample interval')
    t = [0, s['rise_start'], s['rise_end']]
    v = [start, start, turn]
    if 'fall_start' in s:
        if s['fall_start'] == s['fall_end'] and turn != terminal:
            raise ValueError('A nonzero second edge needs at least one sample interval')
        t.extend((s['fall_start'], s['fall_end']))
        v.extend((turn, terminal))
    t.append(capacity)
    v.append(terminal)
    return np.asarray(t, dtype=float), np.asarray(v, dtype=float)


def is_extended(command):
    return command.kind == 'ramp' and bool(int(command.words[4]) & (1 << 23))


def packet(sequence, segment_index, output, cfg, start, terminal, point_index=0):
    from qick.ramp_v7 import ramp_packet
    from qick_fine_tune_sweep import _target_code
    segment = sequence.segments[segment_index]
    spec = sweep_spec_at(sequence, segment.name, output, point_index)
    if spec is None:
        return None
    if cfg.get('type') != 'axis_awg_tuning_v7' or cfg.get('extended_ramp_version') != 1:
        raise ValueError('Sample-position RAMP requires AWG v7 firmware and its matching driver')
    if int(cfg['n_pts']) != 16 or int(cfg['frac']) != 18:
        raise ValueError('Fine RAMP requires the v7 16-lane Q18 ABI')
    s = validate_edges(spec, sequence.segment_duration_cycles_at(point_index, segment_index) * 16)
    args = dict(start=start, target=terminal, rise_start=s['rise_start'], rise_end=s['rise_end'],
                tmux=cfg.get('tmux_ch') or 0)
    if 'fall_start' in s:
        args.update(target=_target_code(cfg, s['turn'] * sequence.voltage_scale_factors[output]),
                    fall_start=s['fall_start'], fall_end=s['fall_end'], end_target=terminal)
    return ramp_packet(**args)


def validate_sequence(sequence):
    if not sequence.fine_ramps:
        return
    from qick_fine_tune_sweep import AmplitudeSweep, RampDurationSweep
    # Legacy register-add sweeps address different bit fields. Never reinterpret
    # them as v7 packet words. HOLD/RF sweeps and repetitions remain supported.
    if any(isinstance(axis, AmplitudeSweep) for axis in sequence.sweeps):
        raise ValueError('Fine RAMP currently supports fixed voltages; remove voltage sweeps or disable fine RAMP')
    if not np.array_equal(sequence.cross_capacitance, np.eye(sequence.n_outputs)):
        raise ValueError('Fine RAMP requires an identity cross-capacitance matrix; mixing different edge positions needs more than two edges')
    by_name = {s.name: s for s in sequence.segments}
    for name, specs in sequence.fine_ramps.items():
        if name not in by_name or by_name[name].kind != 'ramp' or len(specs) != sequence.n_outputs:
            raise ValueError('Fine RAMP settings must refer to an existing RAMP and every output')
        for spec in specs:
            if spec is not None:
                validate_edges(spec, by_name[name].duration_cycles * 16)
        axes = component_axes(sequence, name)
        if axes:
            capacity = sweep_reservation_cycles(sequence, by_name[name], axes[0][1]) * 16
            # All edge indices increase with positive component lengths. The
            # upper corner validates independent as well as grouped sweeps.
            lengths = {c: max(a.duration_sample_points[0], a.duration_sample_points[-1])
                       for _, a, c in axes}
            for spec in specs:
                validate_edges(_with_lengths(spec, lengths), capacity)


def sequence_vertices(sequence, point, space):
    getter = sequence.amplitudes_at if space == 'physical' else sequence.virtual_amplitudes_at
    if space not in ('physical', 'virtual'):
        raise ValueError("space must be 'physical' or 'virtual'")
    current = np.asarray(getter(point, 0), dtype=float)
    times, values, boundaries = [], [], []
    now = 0
    for index, segment in enumerate(sequence.segments):
        duration = sequence.segment_duration_cycles_at(point, index)
        target = np.asarray([current[i] if v is None else v for i, v in enumerate(getter(point, index))])
        if segment.kind == 'set':
            times.extend((now, now, now + duration))
            values.extend((current.copy(), target.copy(), target.copy()))
        else:
            specs = tuple(sweep_spec_at(sequence, segment.name, output, point)
                          for output in range(sequence.n_outputs))
            traces = []
            for output, spec in enumerate(specs):
                if spec is None:
                    traces.append((np.asarray([0., duration]), np.asarray([current[output], target[output]])))
                else:
                    s = dict(spec)
                    if 'turn' in s and space == 'physical':
                        s['turn'] *= sequence.voltage_scale_factors[output]
                    t, v = vertices(s, current[output], target[output], duration * 16)
                    traces.append((t / 16., v))
            grid = np.unique(np.concatenate([t for t, _ in traces]))
            times.extend(now + grid)
            values.extend(np.asarray([np.interp(grid, t, v) for t, v in traces]).T)
        boundaries.append((segment.name, now, now + duration))
        now += duration
        current = target
    array = np.asarray(values).T
    return np.asarray(times), dict(zip(sequence.output_names, array)), tuple(boundaries)


def scalar_runs(command, start_code):
    """Exact clamped Q18 lines: (count, slope, intercept, low, high).

    Quantization is floor(Q18 / 2**20) into effective 14-bit DAC codes.
    Runs include the terminal sample and word padding; no large arrays.
    """
    w = sum(int(v) << (32 * i) for i, v in enumerate(command.words))
    def signed(v, bits):
        return v - (1 << bits) if v & (1 << (bits - 1)) else v
    turn = signed(w & 65535, 16)
    end = signed((w >> 16) & 65535, 16)
    up = signed((w >> 32) & ((1 << 35) - 1), 35)
    down = signed((w >> 67) & ((1 << 35) - 1), 35)
    r0, lane = (w >> 102) & 15, (w >> 106) & 31
    count = (w >> 111) & ((1 << 23) - 1)
    f0 = (((w >> 134) & 1023) * 16 + lane) if lane != 31 else count
    def line(n, step, base, target):
        return (n, step, base << 18, min(base, target) // 4, max(base, target) // 4)
    runs = []
    if r0:
        runs.append(line(r0, 0, start_code, start_code))
    runs.append(line(min(f0, count) - r0, up, start_code, turn))
    if lane != 31:
        # The final scalar sample is forced to the terminal target in RTL.
        runs.append(line(max(0, count - f0 - 1), down, turn, end))
        runs.append(line(1, 0, end, end))
    # For one edge the last sample is also forced to the target.
    else:
        n, a, b, low, high = runs[-1]
        runs[-1] = (n - 1, a, b, low, high)
        runs.append(line(1, 0, end, end))
    if (-count) % 16:
        runs.append(line((-count) % 16, 0, end, end))
    split = []
    for n, a, b, low, high in runs:
        if not n:
            continue
        if a:
            bound = (high if a > 0 else low) << 20
            first_clamped = max(0, (abs(bound - b) + abs(a) - 1) // abs(a))
            linear = min(n, first_clamped)
            if linear:
                split.append((linear, a, b, low, high))
            if n > linear:
                value = high if a > 0 else low
                split.append((n-linear, 0, value << 20, value, value))
        else:
            split.append((n, a, b, low, high))
    return tuple(split)
