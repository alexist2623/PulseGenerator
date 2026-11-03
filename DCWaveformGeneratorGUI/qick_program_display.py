"""Sparse, read-only descriptions of the AWG commands a program emits.

Decode extended RAMP words rather than treating their terminal voltage and
packet length as the first edge's voltage and duration. No waveform arrays,
compilation, hardware access, or changes to the acquisition metadata schema.
"""
from qick_dc_area import executed_commands
from qick_fine_ramp import is_extended


def _signed(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def _dac_code(value):
    return max(-32768, min(32764, value)) & -4


def _edge(start, limit, initial, target, step, fractional_bits, *, bounded):
    """First target DAC sample, with the RTL boundary/terminal override.

    An edge's endpoint is not independently encoded in a two-edge packet.
    Find it from the integer slope and effective DAC quantization, in O(log N).
    The next edge loads the turn target; the final sample forces the terminal.
    """
    initial, target = _dac_code(initial), _dac_code(target)

    def value_at(index):
        raw = (initial << fractional_bits) + step * (index - start)
        if bounded:
            low, high = sorted((initial << fractional_bits, target << fractional_bits))
            raw = max(low, min(high, raw))
        return _dac_code(raw >> fractional_bits)

    low, high = start, limit
    while low < high:
        middle = (low + high) // 2
        value = value_at(middle)
        reached = value >= target if target >= initial else value <= target
        if reached:
            high = middle
        else:
            low = middle + 1
    # A legacy slope can skip the target code. Only its forced terminal
    # sample is then an exact target, rather than the first crossing past it.
    if low < limit and value_at(low) != target:
        low = limit
    return dict(direction='Rising' if target > initial else 'Falling' if target < initial else 'Flat',
                start_sample=start, target_sample=low, boundary_sample=limit,
                start_code_16=initial, target_code_16=target, step=step,
                target_forced_at_boundary=(low == limit and value_at(limit) != target))


def compiled_awg_commands(program, point):
    """Evaluate one point's emitted command register values, in sequence order."""
    from qick_execution_coordinates import _event_time, _selected_points

    _selected_points(program.sequence.sweep_point_count, (point,))
    models = {model['key']: model for model in program._event_timing_models}
    scales = getattr(program.sequence, 'output_full_scales_mv', None)
    current = {}
    records = []
    for command in executed_commands(program, point):
        cfg = program.soccfg['gens'][command.gen_ch]
        start = current.get(command.output_index, 0)
        cycle = _event_time(program, models, ('event_time', 'awg', *program._command_key(command)),
                            program.timing['command_times'][program._timing_key(command)], point)
        record = dict(output_name=command.output_name, segment_name=command.segment_name,
                      segment_index=command.segment_index, gen_ch=command.gen_ch,
                      kind=command.kind, command_cycle=cycle,
                      command_time_ns=cycle * 1000 / float(program.tproc_mhz),
                      sample_ns=1000 / (float(cfg['f_fabric']) * int(cfg['n_pts'])),
                      full_scale_mv=None if scales is None else float(scales[command.output_index]),
                      start_code_16=start, terminal_code_16=_dac_code(int(command.target_code)),
                      edges=[])
        if is_extended(command):
            word = sum((int(value) & 0xffffffff) << (32 * i) for i, value in enumerate(command.words))
            turn, terminal = _signed(word, 16), _signed(word >> 16, 16)
            up, down = _signed(word >> 32, 35), _signed(word >> 67, 35)
            first, lane = (word >> 102) & 15, (word >> 106) & 31
            count = (word >> 111) & ((1 << 23) - 1)
            second = None if lane == 31 else ((word >> 134) & 1023) * 16 + lane
            limit = count - 1 if second is None else second
            record.update(encoding='extended', packet_hex=f'{word:040x}',
                          duration_samples=count, second_edge_enabled=second is not None,
                          terminal_code_16=_dac_code(terminal))
            record['edges'].append(_edge(first, limit, start, turn, up, 18, bounded=True))
            if second is not None:
                record['edges'].append(_edge(second, count - 1, turn, terminal, down, 18, bounded=True))
        elif command.kind == 'ramp':
            count = int(command.duration_samples)
            record.update(encoding='legacy', duration_samples=count)
            record['edges'].append(_edge(0, count - 1, start, command.target_code,
                                         int(command.step), int(cfg['frac']), bounded=False))
        else:
            record['encoding'] = 'set'
        current[command.output_index] = record['terminal_code_16']
        records.append(record)
    return records


def _voltage(code, full_scale):
    text = f'DAC code {code // 4}'
    if full_scale is not None:
        text += f' ({code / 32768 * full_scale:.9g} mV)'
    return text


def format_awg_commands(program, point):
    lines = ['Emitted AWG commands (pre-IIR):',
             'Command time: tProcessor timestamp relative to the repetition reference.',
             'Edge times: sample offsets from this RAMP output start; pipeline latency excluded.',
             'Target time: first target DAC code after integer slope/quantization; not an analog rise time.']
    for record in compiled_awg_commands(program, point):
        dt, scale = record['sample_ns'], record['full_scale_mv']
        lines.append(f"  {record['output_name']} / {record['segment_name']} ({record['encoding']}): "
                     f"command cycle {record['command_cycle']}, {record['command_time_ns']:.9g} ns")
        if record['kind'] == 'set':
            lines.append(f"    SET: {_voltage(record['terminal_code_16'], scale)}")
            continue
        if record['encoding'] == 'extended':
            lines.append(f"    Emitted packet: 0x{record['packet_hex']}")
        for index, edge in enumerate(record['edges'], 1):
            first, last = edge['start_sample'], edge['target_sample']
            lines.append(f"    Edge {index} ({edge['direction']}): "
                         f"{_voltage(edge['start_code_16'], scale)} -> "
                         f"{_voltage(edge['target_code_16'], scale)}")
            lines.append(f"      start {first * dt:.9g} ns [sample {first}], "
                         f"target reached {last * dt:.9g} ns [sample {last}], "
                         f"duration {(last-first) * dt:.9g} ns ({last-first} sample intervals)")
            if edge['target_forced_at_boundary']:
                lines.append('      Target is forced by the packet boundary/terminal sample.')
        if len(record['edges']) == 2:
            first = record['edges'][0]['target_sample']
            last = record['edges'][1]['start_sample']
            lines.append(f'    HOLD between edges: {first * dt:.9g} -> {last * dt:.9g} ns; '
                         f'duration {(last-first) * dt:.9g} ns')
        elif record['encoding'] == 'extended':
            lines.append('    Second edge: disabled (11111).')
        count = record['duration_samples']
        lines.append(f"    Terminal: {_voltage(record['terminal_code_16'], scale)} at "
                     f"{(count-1) * dt:.9g} ns [sample {count-1}]; "
                     f"{count} packet samples including endpoint, {(-count) % 16} padding samples.")
    return lines
