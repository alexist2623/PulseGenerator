"""Incremental v7 voltage/edge sweeps, using the existing coefficient loader.

Voltage coordinates remain DAC-code register adds. Duration rows contain a
base slope and voltage increments, never complete packets. A signed 35-bit
Q18 slope is held in two additive 32-bit registers in radix 256. The low
register is deliberately unnormalised: carrying at emission preserves both
positive and negative steps without a 32-bit overflow at sample boundaries.
"""
from dataclasses import replace
import numpy as np


def has_voltage(sequence):
    return any(a.axis_kind == 'amplitude' for a in sequence.sweep_axes)


def point_at(sequence, coordinates):
    return int(np.ravel_multi_index(tuple(coordinates), sequence.model_shape)) if coordinates else 0


def affine_model(program, key, getter, *, quantum=1):
    from qick_fine_tune_sweep import _round_div_nearest
    axes = program.sequence.compiler_sweep_axes
    base = int(getter(0))
    deltas = []
    for a, axis in enumerate(axes):
        coordinate = [0] * len(axes)
        coordinate[a] = axis.count - 1
        delta = (0 if axis.count == 1 else quantum * _round_div_nearest(
            int(getter(point_at(program.sequence, coordinate))) - base, (axis.count - 1) * quantum))
        deltas.append(delta)
    return dict(key=key, base=base, axis_deltas=tuple(deltas))


def target_model(program, segment, output, key):
    """Use exactly the same DAC grid as the SET register-add compiler."""
    from qick_fine_tune_sweep import _target_code
    cfg = program.soccfg['gens'][program.awg_channels[output]]
    return affine_model(program, key, lambda p: _target_code(cfg,
        program.sequence._effective_physical_set_amplitudes_at(p, segment)[output]),
        quantum=1 << int(cfg.get('dac_invalid_lsb', 2)))


def affects_command(program, command):
    return any(any(target_model(program, i, command.output_index, ())['axis_deltas'])
               for i in (command.segment_index - 1, command.segment_index + 1))


def length_model(program, key, values, axis):
    axes = program.sequence.compiler_sweep_axes
    deltas = [0] * len(axes)
    result = dict(key=key, base=int(values[0]), axis_deltas=tuple(deltas))
    if axis is None or len(set(values)) == 1:
        return result
    delta = int(values[1]) - int(values[0])
    if all(int(v) == int(values[0]) + i * delta for i, v in enumerate(values)):
        deltas[axis] = delta
        result['axis_deltas'] = tuple(deltas)
    else:
        result.update(duration_table_bases=tuple(map(int, values)), duration_table_axis_deltas={},
            duration_axis_index=axis, duration_axis_indices=(axis,), duration_table_shape=(len(values),))
    return result


def step_models(program, key, getter, duration_axis):
    """Fit the same duration-conditioned integer increments as coarse RAMP.

    Quantization is explicit: independently rounded ideal slopes need not be
    affine in voltage. Metadata/area/RC checks use this executed integer model.
    Grouped voltage/duration coordinates select one exact coefficient row.
    """
    from qick_fine_tune_sweep import _round_div_nearest
    axes = program.sequence.compiler_sweep_axes
    voltage_axes = sorted({program.sequence.axis_representative(i)
        for i, a in enumerate(program.sequence.sweep_axes) if a.axis_kind == 'amplitude'})
    count = 1 if duration_axis is None else axes[duration_axis].count
    bases, deltas = [], {a: [] for a in voltage_axes if a != duration_axis and axes[a].count > 1}
    for j in range(count):
        program._check_cancel()
        c = [0] * len(axes)
        if duration_axis is not None:
            c[duration_axis] = j
        base = int(getter(point_at(program.sequence, c)))
        bases.append(base)
        for a in deltas:
            c[a] = axes[a].count - 1
            end = int(getter(point_at(program.sequence, c)))
            c[a] = 0
            deltas[a].append(_round_div_nearest(end - base, axes[a].count - 1))
    model = dict(key=key, base=bases[0], axis_deltas=tuple(deltas.get(a, (0,))[0] for a in range(len(axes))))
    if count > 1 and (len(set(bases)) > 1 or any(len(set(v)) > 1 for v in deltas.values())):
        model.update(axis_deltas=(0,) * len(axes), duration_table_bases=tuple(bases),
            duration_table_axis_deltas={a: tuple(v) for a, v in deltas.items()},
            duration_axis_index=duration_axis, duration_axis_indices=(duration_axis,),
            duration_table_shape=(count,))
    # Validate the unsplit signed-35-bit model at all affine row corners.
    for row, base in enumerate(bases):
        low = base + sum(min(0, v[row] * (axes[a].count - 1)) for a, v in deltas.items())
        high = base + sum(max(0, v[row] * (axes[a].count - 1)) for a, v in deltas.items())
        if low < -(1 << 34) or high >= 1 << 34:
            raise ValueError('Incremental fine RAMP slope exceeds signed 35 bits; reduce the voltage increment')
    result = []
    for part, transform in (('lo', lambda v: int(v) & 255), ('hi', lambda v: int(v) >> 8)):
        m = dict(model, key=(*key, part), base=transform(model['base']),
                 axis_deltas=tuple(transform(d) for d in model['axis_deltas']))
        if 'duration_table_bases' in model:
            m.update(duration_table_bases=tuple(map(transform, model['duration_table_bases'])),
                duration_table_axis_deltas={a: tuple(map(transform, v))
                                           for a, v in model['duration_table_axis_deltas'].items()})
        rows = m.get('duration_table_bases', (m['base'],))
        for row, base in enumerate(rows):
            ds = ({a: v[row] for a, v in m['duration_table_axis_deltas'].items()}
                  if 'duration_table_bases' in m else dict(enumerate(m['axis_deltas'])))
            low = base + sum(min(0, d * (axes[a].count - 1)) for a, d in ds.items())
            high = base + sum(max(0, d * (axes[a].count - 1)) for a, d in ds.items())
            if low < -(1 << 31) or high >= 1 << 31:
                raise ValueError('Fine RAMP additive slope state exceeds 32 bits; reduce the sweep point count')
        result.append(m)
    return result


def emit_step(program, load, low, high, scratch, page):
    """Convert two additive radix-256 states to the wire's low32/high3."""
    load('lo', low)
    load('hi', high)
    program.bitwi(page, scratch, low, '>>', 8, 'fine slope carry')
    program.math(page, high, high, '+', scratch, 'normalise additive fine slope')
    program.bitwi(page, low, low, '&', 255)
    program.bitwi(page, scratch, high, '<<', 8)
    program.bitw(page, low, low, '|', scratch)
    program.bitwi(page, high, high, '>>', 24)
    program.bitwi(page, high, high, '&', 7, 'signed 35-bit fine slope')


def build_models(program, key, first):
    from qick.ramp_v7 import edge_step
    from qick_fine_ramp import component_axes, sweep_spec_at
    from qick_fine_tune_sweep import _target_code
    seq = program.sequence
    axes = seq.compiler_sweep_axes
    spec = sweep_spec_at(seq, first.segment_name, first.output_index, 0)
    two = 'fall_start' in spec
    cfg = program.soccfg['gens'][first.gen_ch]
    prefix = (*key, 'fine_voltage')
    start = target_model(program, first.segment_index - 1, first.output_index, (*prefix, 'start'))
    end = target_model(program, first.segment_index + 1, first.output_index, (*prefix, 'end'))
    turn = (dict(key=(*prefix, 'turn'), base=_target_code(cfg, spec['turn'] * seq.voltage_scale_factors[first.output_index]),
                 axis_deltas=(0,) * len(axes)) if two else dict(end, key=(*prefix, 'turn')))
    models = {m['key']: m for m in (turn, end)}
    dependencies = {component: seq.axis_representative(i)
                    for i, _, component in component_axes(seq, first.segment_name)}
    def lengths(point):
        s = sweep_spec_at(seq, first.segment_name, first.output_index, point)
        return dict(rise=s['rise_end']-s['rise_start'],
                    hold=s['fall_start']-s['rise_end'] if two else 0,
                    fall=s['fall_end']-s['fall_start'] if two else 0)
    for name in ('rise', 'hold', 'fall'):
        axis = dependencies.get(name)
        values = []
        for j in range(1 if axis is None else axes[axis].count):
            c = [0] * len(axes)
            if axis is not None:
                c[axis] = j
            values.append(lengths(point_at(seq, c))[name])
        m = length_model(program, (*prefix, name), values, axis)
        models[m['key']] = m
    for name, a, b, component in (('up', start, turn, 'rise'), ('down', turn, end, 'fall')):
        def getter(point, a=a, b=b, component=component):
            return edge_step(program._sweep_model_value(a, point), program._sweep_model_value(b, point), lengths(point)[component])
        fields = step_models(program, (*prefix, name), getter if two or name == 'up' else lambda p: 0,
                             dependencies.get(component))
        models.update({m['key']: m for m in fields})
    for m in models.values():
        m.update(command_key=key, register_name='fine_voltage', command_register_name='reserved_start')
    return models, dict(voltage=True, rise_start=spec['rise_start'], two_edges=two,
                        control_word=int(first.words[4]) & 0xffff0000)


def command_at(program, command, point, models=None, recipe=None):
    """Reconstruct the executed packet for plotting, DC/RC checks and exports."""
    key = program._command_key(command)
    models = program._sweep_models if models is None else models
    recipe = program._fine_packet_recipes[key] if recipe is None else recipe
    prefix = (*key, 'fine_voltage')
    get = lambda name: program._sweep_model_value(models[(*prefix, name)], point)
    step = lambda name: (program._sweep_model_value(models[(*prefix, name, 'hi')], point) << 8) + program._sweep_model_value(models[(*prefix, name, 'lo')], point)
    turn, end = get('turn'), get('end')
    cfg = program.soccfg['gens'][command.gen_ch]
    if any(not int(cfg.get('minv', -32768)) <= v <= int(cfg.get('maxv', 32764)) for v in (turn, end)):
        raise ValueError(f'Incremental fine RAMP target exceeds DAC range at point {point}')
    r0 = recipe['rise_start']
    f0 = r0 + get('rise') + get('hold')
    count = (f0 + get('fall') if recipe['two_edges'] else r0 + get('rise')) + 1
    lane, coarse = (f0 % 16, f0 // 16) if recipe['two_edges'] else (31, 0)
    up, down = step('up'), step('down')
    word = ((turn & 65535) | ((end & 65535) << 16) | ((up & ((1 << 35)-1)) << 32)
            | ((down & ((1 << 35)-1)) << 67) | (r0 << 102) | (lane << 106)
            | (count << 111) | (coarse << 134) | (recipe['control_word'] << 128))
    return replace(command, point_index=point, target_code=end, step=up, duration_samples=count,
                   words=tuple((word >> (32*i)) & 0xffffffff for i in range(5)))


def emit_packet(program, command, registers):
    key = program._command_key(command)
    recipe = program._fine_packet_recipes[key]
    page, scratch = program._gen_regmap[(command.gen_ch, 't')]
    w0, w1, w2, w3, w4 = registers
    prefix = (*key, 'fine_voltage')
    def load(name, destination):
        k = (*prefix, *name) if isinstance(name, tuple) else (*prefix, name)
        program._write_swept_or_static_register(k, page, destination, program._sweep_models[k]['base'], 'fine '+str(name))
    load('rise', w3)
    program.mathi(page, w3, w3, '+', recipe['rise_start'])
    if recipe['two_edges']:
        load('hold', w4)
        program.math(page, w3, w3, '+', w4)
        load('fall', w4)
        program.math(page, w4, w4, '+', w3)
        program.bitwi(page, w0, w3, '>>', 4)
        program.bitwi(page, w0, w0, '<<', 6)
        program.bitwi(page, w3, w3, '&', 15)
        program.bitwi(page, w3, w3, '<<', 10)
    else:
        program.mathi(page, w4, w3, '+', 0)
        program.safe_regwi(page, w0, 0)
        program.safe_regwi(page, w3, 31 << 10)
    program.mathi(page, w4, w4, '+', 1)
    program.bitwi(page, scratch, w4, '>>', 17)
    program.bitw(page, w0, w0, '|', scratch)
    program.bitwi(page, w0, w0, '|', recipe['control_word'])
    program.bitwi(page, w4, w4, '<<', 15)
    program.bitw(page, w3, w3, '|', w4)
    program.bitwi(page, w3, w3, '|', recipe['rise_start'] << 6)
    program.mathi(page, w4, w0, '+', 0)
    emit_step(program, lambda p,r: load(('down',p),r), w0, w1, scratch, page)
    program.bitwi(page, w1, w1, '<<', 3)
    program.bitwi(page, scratch, w0, '>>', 29)
    program.bitw(page, w1, w1, '|', scratch)
    program.bitw(page, w3, w3, '|', w1)
    program.bitwi(page, w0, w0, '<<', 3)
    program.memwi(page, w0, 0, 'transient falling slope bits')
    emit_step(program, lambda p,r: load(('up',p),r), w1, w2, scratch, page)
    program.memri(page, scratch, 0)
    program.bitw(page, w2, w2, '|', scratch)
    load('turn', w0)
    program.bitwi(page, w0, w0, '&', 65535)
    load('end', scratch)
    program.bitwi(page, scratch, scratch, '<<', 16)
    program.bitw(page, w0, w0, '|', scratch)
