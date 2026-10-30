"""Factor independent v7 packet components into O(sum(axis counts)) DMEM.

The CPU combines lengths and packs cross-word bitfields at emission. No
Cartesian packet table and no host loop are needed. Step division remains
offline, once per edge-duration point; FPGA firmware is unchanged.
"""
import numpy as np
from qick_fine_ramp import component_axes, sweep_spec_at


def build_component_models(program, key, first, command_maps):
    sequence = program.sequence
    axes = sequence.compiler_sweep_axes
    dependencies = {component: sequence.axis_representative(i)
                    for i, _, component in component_axes(sequence, first.segment_name)}

    def values(point):
        cmd = command_maps[point][key]
        word = sum(int(w) << (32*i) for i, w in enumerate(cmd.words))
        spec = sweep_spec_at(sequence, first.segment_name, first.output_index, point)
        if 'fall_start' not in spec:
            raise ValueError('Independent fine components require a two-edge packet')
        return dict(rise=spec['rise_end']-spec['rise_start'],
                    hold=spec['fall_start']-spec['rise_end'],
                    fall=spec['fall_end']-spec['fall_start'],
                    up_low=(word >> 32) & 0xffffffff, up_high=(word >> 64) & 7,
                    down_low=(word >> 67) & ((1 << 29)-1), down_high=(word >> 96) & 63)

    base = values(0)
    models = {}
    for name, component in dict(rise='rise', hold='hold', fall='fall',
            up_low='rise', up_high='rise', down_low='fall', down_high='fall').items():
        model_key = (*key, 'fine_'+name)
        model = dict(key=model_key, command_key=key, register_name='fine_'+name,
                     command_register_name='reserved_start', base=int(base[name]),
                     axis_deltas=tuple(0 for _ in axes))
        axis = dependencies.get(component)
        if axis is not None:
            table = []
            for index in range(axes[axis].count):
                coordinate = [0] * len(axes); coordinate[axis] = index
                point = int(np.ravel_multi_index(tuple(coordinate), sequence.model_shape))
                table.append(int(values(point)[name]))
            if len(set(table)) > 1:
                model.update(duration_table_bases=tuple(table), duration_table_axis_deltas={},
                    duration_axis_index=axis, duration_axis_indices=(axis,),
                    duration_table_shape=(axes[axis].count,))
        models[model_key] = model
    spec = sweep_spec_at(sequence, first.segment_name, first.output_index, 0)
    return models, dict(rise_start=spec['rise_start'], target_word=int(first.words[0]),
                        control_word=int(first.words[4]) & 0xffff0000)


def emit_component_packet(program, command, registers):
    key = program._command_key(command)
    recipe = program._fine_packet_recipes[key]
    page, scratch = program._gen_regmap[(command.gen_ch, 't')]
    w0, w1, w2, w3, w4 = registers

    def load(name, destination):
        field_key = (*key, 'fine_'+name)
        program._write_swept_or_static_register(field_key, page, destination,
            int(program._sweep_models[field_key]['base']), 'fine packet '+name)

    load('rise', w3); load('hold', w4)
    program.math(page, w3, w3, '+', w4, 'fine falling start: rise + hold')
    program.mathi(page, w3, w3, '+', recipe['rise_start'], 'fine first edge offset')
    load('fall', w4)
    program.math(page, w4, w4, '+', w3, 'fine final sample index')
    program.mathi(page, w4, w4, '+', 1, 'fine sample count includes terminal')
    program.bitwi(page, w0, w3, '>>', 4, 'fine falling word')
    program.bitwi(page, w0, w0, '<<', 6, 'pack falling word')
    program.bitwi(page, scratch, w4, '>>', 17, 'fine count high bits')
    program.bitw(page, w0, w0, '|', scratch, 'pack count high bits')
    program.bitwi(page, w0, w0, '|', recipe['control_word'], 'pack v7 control')
    program.bitwi(page, w4, w4, '<<', 15, 'pack count low bits')
    program.bitwi(page, w3, w3, '&', 15, 'fine falling lane')
    program.bitwi(page, w3, w3, '<<', 10, 'pack falling lane')
    program.bitw(page, w3, w3, '|', w4, 'pack fine duration word')
    load('down_high', scratch)
    program.bitw(page, w3, w3, '|', scratch, 'pack falling step high bits')
    program.bitwi(page, w3, w3, '|', recipe['rise_start'] << 6, 'pack rising lane')
    program.mathi(page, w4, w0, '+', 0, 'store packed control word')
    program.safe_regwi(page, w0, recipe['target_word'], 'fine fixed targets')
    load('up_low', w1); load('up_high', w2); load('down_low', scratch)
    program.bitwi(page, scratch, scratch, '<<', 3, 'pack falling step low bits')
    program.bitw(page, w2, w2, '|', scratch, 'pack shared step word')
