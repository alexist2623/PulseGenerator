"""Compiled-coordinate export agrees with decoded v5 tProcessor commands."""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import qick.sim  # noqa: F401 - desktop driver bootstrap
from qick.awg_tuning import TProcV1BehaviorModel
from qick.output_sum import apply_output_sum_config
from qick_execution_coordinates import build_execution_coordinates
from qick_fine_tune_sweep import FineTuneSequence, RfPulseConfig
from test_qick_dc_area import config as awg_config
from test_rf_fine_duration_registers import _v5_soccfg


def value(column, point):
    return column[point] if isinstance(column, list) else column


def signed(number, bits=32):
    number &= (1 << bits) - 1
    return number - (1 << bits) if number & (1 << (bits - 1)) else number


def execute(program):
    model = TProcV1BehaviorModel(strict=True)
    program.load_runtime_dmem_into_model(model)
    model.run(program, max_steps=2_000_000)
    assert model.ended and not model.timing_conflicts
    return model


def decode_bursts(program, events):
    """Decode memory windows and command edges, without compiler duration plans."""
    memory = np.zeros(16384, dtype=np.int16)
    for envelope in program.envelopes[0]['envs'].values():
        start = envelope['addr']
        memory[start:start + len(envelope['data'])] = envelope['data'][:, 0]
    result = []
    cursor = 0
    while cursor < len(events):
        event = events[cursor]
        mode = (event.word >> 128) & 0xffffffff
        length, selection = mode & 0xffff, (mode >> 16) & 3
        if selection == 0:
            address = ((event.word >> 64) & 0xffffffff) * 16
            samples = int(np.count_nonzero(memory[address:address + length * 16]))
            commands = 1
        else:
            assert selection == 1
            tail = events[cursor + 1]
            tail_mode = (tail.word >> 128) & 0xffffffff
            assert (tail_mode >> 16) & 3 == 0
            address = ((tail.word >> 64) & 0xffffffff) * 16
            samples = (tail.cycle - event.cycle) * 16 + int(np.count_nonzero(
                memory[address:address + (tail_mode & 0xffff) * 16]))
            assert signed(tail.word >> 96, 16) == 2 * signed(event.word >> 96, 16)
            commands = 2
        result.append(dict(samples=samples, commands=commands))
        cursor += commands
    return result


def compiled_bytes(program):
    program.compile()
    return (tuple(program.binprog), tuple(program._runtime_dmem_words),
            {channel: {name: (entry['addr'], entry['data'].tobytes())
                       for name, entry in item['envs'].items()}
             for channel, item in enumerate(program.envelopes)})


def awg_program(time_count=3, voltage_count=3):
    cfg = awg_config(2)
    for gen in cfg['gens']:
        apply_output_sum_config(gen, dict(output_sum_version=2, output_sum_role='awg',
            post_output_latency_cycles=2, output_sum_awg_delay_cycles=13))
    seq = FineTuneSequence(('a', 'b'))
    seq.set_cross_capacitance(((1., .2), (-.1, 1.)))
    seq.set_voltage_scales(800., (800., 900.))
    seq.add_set('zero', (0., 0.), 30).add_ramp('rise', 3)
    seq.add_set('hold', (.1, .2), 300).add_ramp('fall', 3)
    seq.add_set('reset', (0., 0.), 30)
    seq.add_ramp_duration_sweep('rise', .01, 1., time_count)
    seq.add_ramp_duration_sweep('fall', .02, .5, time_count)
    seq.add_amplitude_sweep('hold', 'a', -.2, .3, voltage_count)
    seq.add_amplitude_sweep('hold', 'b', .1, -.1, voltage_count)
    seq.set_sweep_groups({seq.sweep_axis_key(axis): 'time' if 'duration' in axis.axis_kind else 'voltage'
                          for axis in seq.sweep_axes})
    seq.sweep_parameter_ids = {seq.sweep_axis_key(axis): f'member-{i}' for i, axis in enumerate(seq.sweep_axes)}
    return seq.make_program(cfg, awg_channels=(0, 1), repetitions_per_sweep=2,
                            compile_validation_mode='boundary')


@pytest.mark.parametrize('counts', [(3, 3), (1, 3), (3, 1), (1, 1)])
def test_matrix_grouped_commands_count_one_reset_and_immutable_bytes(counts):
    program = awg_program(*counts)
    before = compiled_bytes(program)
    before_models = copy.deepcopy(program._sweep_models)
    snapshot = build_execution_coordinates(program)
    json.dumps(snapshot, allow_nan=False)
    assert compiled_bytes(program) == before
    assert program._sweep_models == before_models
    model = execute(program)
    # v5 initialization has RC/DC control commands, then the repeated waveform.
    initial_control_end = max(event.cycle for event in model.output_events
                              if ((event.word >> 144) & 3) == 3)
    events = [event for event in model.output_events if event.cycle > initial_control_end]
    commands = snapshot['awg_commands']
    points = snapshot['point_count']
    assert len(events) == points * 2 * len(commands)
    for point in range(points):
        ti, vi = np.unravel_index(point, counts)
        assert [value(item['coordinates'], point) for item in snapshot['group_indices']] == [ti, vi]
        assert [value(item['coordinates'], point) for item in snapshot['requested_members']] == list(
            program.sequence.sweep_coordinate(point))
        for repetition in range(2):
            shot = events[(point * 2 + repetition) * len(commands):(point * 2 + repetition + 1) * len(commands)]
            origins = []
            for command, event in zip(commands, shot):
                columns = command['columns']
                target = signed(event.word)
                assert value(columns['target_code_16'], point) == target
                assert value(columns['target_code_14'], point) == target >> 2
                assert value(columns['step'], point) == signed(event.word >> 96)
                assert value(columns['duration_samples'], point) == (event.word >> 64) & 0xffffffff
                assert value(columns['target_mv'], point) == (target >> 2) / 8192 * command['full_scale_mv']
                origins.append(event.cycle - value(columns['command_cycle'], point))
            assert len(set(origins)) == 1
    assert snapshot['coordinate_domain'].startswith('Compiled digital commands')
    assert [item['parameter_id'] for item in snapshot['requested_members']] == [f'member-{i}' for i in range(4)]


def rf_program(fine, first=1, last=113, count=4, *, gain=-1000, phase=45.):
    cfg = _v5_soccfg()
    seq = FineTuneSequence(('a',)).add_set('gate', .01, 400_000).add_set('zero', 0., 30)
    seq.add_amplitude_sweep('gate', 'a', .01, .02, count)
    divisor = 4800. if fine else 300.
    seq.add_rf_duration_sweep('gate', 0, first/divisor, last/divisor, count,
                             sequence_fabric_mhz=300.)
    seq.add_rf_frequency_sweep('gate', 0, 10., 21., count)
    seq.set_sweep_groups({seq.sweep_axis_key(axis): 'shared' for axis in seq.sweep_axes})
    rf = RfPulseConfig(0, 'gate', max(3, (first + 15)//16) if fine else first,
        gain, freq_mhz=10., phase_degrees=phase, delay_tproc_cycles=30,
        fine_duration=fine, duration_samples=first if fine else None)
    return seq.make_program(cfg, awg_channels=(1,), rf_pulse=rf, repetitions_per_sweep=2,
                            compile_validation_mode='boundary')


@pytest.mark.parametrize('fine,first,last,count', [
    (False, 30, 100, 4), (False, 66000, 66006, 3),
    (True, 1, 113, 4), (True, 113, 1, 4), (True, 31, 31, 1),
    (True, 1048601, 1048678, 3),
])
def test_rf_words_masked_samples_gain_and_grouped_index_match_execution(fine, first, last, count):
    program = rf_program(fine, first, last, count)
    before = compiled_bytes(program)
    snapshot = build_execution_coordinates(program)
    assert compiled_bytes(program) == before
    events = [event for event in execute(program).output_events if (event.word >> 152) & 255 == 0]
    columns = snapshot['rf_events'][0]['columns']
    bursts = decode_bursts(program, events) if fine else None
    cursor = 0
    for point in range(count):
        for repetition in range(2):
            event = events[cursor]
            assert event.word & 0xffffffff == value(columns['frequency_word'], point)
            assert (event.word >> 32) & 0xffffffff == value(columns['phase_word'], point)
            assert signed(event.word >> 96, 16) == value(columns['emitted_start_gain'], point)
            if fine:
                burst = bursts[point * 2 + repetition]
                assert burst['samples'] == value(columns['active_samples'], point)
                if burst['commands'] == 2:
                    tail = events[cursor + 1]
                    assert signed(tail.word >> 96, 16) == value(columns['emitted_tail_gain'], point)
                    assert tail.cycle - event.cycle == (value(columns['tail_command_cycle'], point)
                                                       - value(columns['command_start_cycle'], point))
                cursor += burst['commands']
            elif value(columns['periodic_plateau'], point):
                stop = events[cursor + 1]
                assert signed(stop.word >> 96, 16) == 0
                assert (stop.cycle - event.cycle) * 16 == value(columns['scheduled_stop_duration_samples'], point)
                assert value(columns['active_samples'], point) is None
                assert value(columns['active_duration_us'], point) is None
                cursor += 2
            else:
                assert ((event.word >> 128) & 0xffff) * 16 == value(columns['active_samples'], point)
                cursor += 1
            assert value(columns['logical_gain'], point) == -1000
    assert cursor == len(events)
    json.dumps(snapshot, allow_nan=False)


def test_sparse_selection_constant_columns_and_cancellation():
    program = awg_program()
    snapshot = build_execution_coordinates(program, point_indices=(8, 0))
    assert snapshot['point_count'] == 2 and snapshot['total_point_count'] == 9
    assert snapshot['point_index'] == [8, 0]
    assert snapshot['awg_commands'][0]['columns']['target_code_16'] == 0
    one = build_execution_coordinates(program, point_indices=(4,))
    assert one['point_index'] == 4
    assert all(not isinstance(item['coordinates'], list) for item in one['requested_members'])
    for indices in ((), (9,), (True,), (0, 0)):
        with pytest.raises(ValueError):
            build_execution_coordinates(program, point_indices=indices)
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        build_execution_coordinates(program, cancel_check=cancel)


def test_software_bundle_keeps_each_program_gain_phase_and_global_point_indices():
    from dc_waveform_core import QickRfSoftwareSweepAxisSpec
    from qick_qcodes_experiment import SoftwareSweepProgramBundle
    first = rf_program(True, count=3, gain=1000, phase=0.)
    second = rf_program(True, count=3, gain=2000, phase=90.)
    axis = QickRfSoftwareSweepAxisSpec('gate', 'rf_0', 0, 0., 90., 2, 'rf_phase', 'deg', 'phase')
    bundle = SoftwareSweepProgramBundle((first, second), (axis,), np.array([[0.], [90.]]))
    snapshot = build_execution_coordinates(bundle, point_indices=(5, 0, 3))
    assert snapshot['point_count'] == 3 and snapshot['total_point_count'] == 6
    assert snapshot['point_index'] == [5, 0, 3]
    assert [point['software_coordinate'] for point in snapshot['software_points']] == [[0.], [90.]]
    assert [point['combined_point_indices'] for point in snapshot['software_points']] == [0, [5, 3]]
    for point, gain, phase in zip(snapshot['software_points'], (1000, 2000), (0, 1 << 30)):
        columns = point['hardware']['rf_events'][0]['columns']
        assert columns['logical_gain'] == gain
        assert columns['phase_word'] == phase
    json.dumps(snapshot, allow_nan=False)


@pytest.mark.parametrize('fine', [False, True])
def test_correlated_frequency_power_table_resolves_actual_gain(fine):
    program = rf_program(fine, 30, 113, 3, gain=1000)
    sequence = program.sequence
    sequence.add_rf_power_sweep('gate', 0, -20., -10., 3)
    sequence.set_sweep_groups({sequence.sweep_axis_key(axis): 'all' for axis in sequence.sweep_axes})
    gains = tuple(1000 + 100 * index for index in range(9))
    rf = replace(program.rf_pulse_configs[0], sweep_gain_codes=gains, sweep_gain_shape=(3, 3))
    program = sequence.make_program(program.soccfg, awg_channels=(1,), rf_pulse=rf,
        repetitions_per_sweep=2, compile_validation_mode='boundary')
    snapshot = build_execution_coordinates(program)
    columns = snapshot['rf_events'][0]['columns']
    assert columns['logical_gain'] == [gains[index] for index in (0, 4, 8)]
    events = [event for event in execute(program).output_events if (event.word >> 152) & 255 == 0]
    cursor = 0
    for point in range(3):
        for repetition in range(2):
            assert signed(events[cursor].word >> 96, 16) == value(columns['emitted_start_gain'], point)
            cursor += 1 + int(value(columns['emitted_tail_gain'], point) is not None)
    assert cursor == len(events)


def test_full_snapshot_cache_is_used_but_sparse_preview_is_evaluated(monkeypatch):
    program = awg_program()
    snapshot = build_execution_coordinates(program)
    program._execution_coordinate_snapshot = snapshot
    import qick_execution_coordinates as exporter
    original = exporter.executed_commands
    calls = []
    def counted(*args):
        calls.append(args[1])
        return original(*args)
    monkeypatch.setattr(exporter, 'executed_commands', counted)
    assert build_execution_coordinates(program) is snapshot
    assert calls == []
    selected = build_execution_coordinates(program, point_indices=(8,))
    assert selected['point_index'] == 8
    assert calls == [0, 8]
