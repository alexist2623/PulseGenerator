"""Execute compact fine-duration arithmetic, including quantization and loops."""
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import qick.sim  # noqa: F401
from qick.awg_tuning import TProcV1BehaviorModel
from qick.output_sum import apply_output_sum_config
from qick_fine_tune_sweep import FineTuneSequence, RfPulseConfig
from qick_rf_fine_duration import sample_accumulator, samples_from_us, plan_duration
from test_qick_fine_tune_sweep import _shared_tmux_soccfg


def _v5_soccfg():
    cfg = _shared_tmux_soccfg()
    cfg['gens'][0]['f_dds'] = 4800.
    cfg['gens'][1].update(type='axis_awg_tuning_v3', frac=18, step_width=32,
        rc_precomp_version=1, dc_area_version=1, dc_area_bits=48,
        dc_start_cycles=17, output_latency_cycles=16)
    for channel, role in ((0, 'rf'), (1, 'awg')):
        apply_output_sum_config(cfg['gens'][channel], dict(output_sum_version=2,
            output_sum_role=role, post_output_latency_cycles=2, output_sum_awg_delay_cycles=13))
        cfg['gens'][channel]['dac'] = '10'
        assert cfg['gens'][channel]['output_latency_cycles'] == 31
    return cfg


def compile_sweep(first, last, count, *, outer=1, reps=1, durations_us=False,
                  frequency_count=0, preceding=False, grouped=False):
    cfg = _v5_soccfg()
    seq = FineTuneSequence(('awg_0',)).add_set('gate', .01, 400_000)
    if outer > 1:
        seq.add_amplitude_sweep('gate', 'awg_0', .01, .02, outer)
    start, stop = (first, last) if durations_us else (first / 4800., last / 4800.)
    seq.add_rf_duration_sweep('gate', 0, start, stop, count,
                             parameter_name='d0', sequence_fabric_mhz=300.)
    if frequency_count:
        seq.add_rf_frequency_sweep('gate', 0, 10., 20., frequency_count)
    if grouped:
        # Uses the public group contract; only explicit equal-count fixtures.
        assert outer == count
        seq.set_sweep_groups({(a.axis_kind, a.segment_name, a.output_name): 'voltage_duration'
                             for a in seq.sweep_axes[:2]})
    first_samples = samples_from_us(start, 300.)
    pulse = RfPulseConfig(0, 'gate', max(3, (first_samples + 15) // 16), 1000,
        freq_mhz=10., phase_degrees=45., delay_tproc_cycles=30,
        fine_duration=True, duration_samples=first_samples, duration_parameter='d0')
    pulses = [pulse]
    if preceding:
        pulses.append(replace(pulse, duration_parameter='', duration_samples=113,
            length_cycles=8, delay_tproc_cycles=30 + (first_samples + 15) // 16 + 60,
            preceding_duration_parameters=('d0',), event_id='following'))
    program = seq.make_program(cfg, awg_channels=(1,), rf_pulses=tuple(pulses),
        repetitions_per_sweep=reps, compile_validation_mode='boundary')
    program.compile()
    return program


def execute(program):
    model = TProcV1BehaviorModel(strict=True)
    program.load_runtime_dmem_into_model(model)
    model.run(program, max_steps=10_000_000)
    assert model.ended and not model.timing_conflicts
    return model, [event for event in model.output_events if (event.word >> 152) & 255 == 0]


def decode_bursts(program, events):
    memory = np.zeros(16384, dtype=np.int16)
    for envelope in program.envelopes[0]['envs'].values():
        start = envelope['addr']
        memory[start:start + len(envelope['data'])] = envelope['data'][:, 0]
    bursts = []
    cursor = 0
    while cursor < len(events):
        first = events[cursor]
        mode = (first.word >> 128) & 0xffffffff
        length, selection = mode & 65535, (mode >> 16) & 3
        assert not (mode >> 20) & 1  # Never reset DDS phase at a continuation.
        if selection == 0:
            assert not (mode >> 18) & 1
            address = ((first.word >> 64) & 0xffffffff) * 16
            active = int(np.count_nonzero(memory[address:address + length * 16]))
            assert length >= 3
            burst = dict(start=first.cycle, samples=active, commands=1, periodic=False)
            cursor += 1
        else:
            assert selection == 1 and cursor + 1 < len(events)
            tail = events[cursor + 1]
            tail_mode = (tail.word >> 128) & 0xffffffff
            assert (tail_mode >> 16) & 3 == 0 and not (tail_mode >> 18) & 1
            assert not (tail_mode >> 20) & 1
            span = tail.cycle - first.cycle
            periodic = bool((mode >> 18) & 1)
            assert span >= 3 and (span % 3 == 0 if periodic else span == length)
            if periodic:
                assert length == 3
            address = ((tail.word >> 64) & 0xffffffff) * 16
            tail_samples = int(np.count_nonzero(memory[address:address + (tail_mode & 65535) * 16]))
            assert (first.word & ((1 << 64) - 1)) == (tail.word & ((1 << 64) - 1))
            assert ((tail.word >> 96) & 65535) == 2 * ((first.word >> 96) & 65535)
            burst = dict(start=first.cycle, samples=span * 16 + tail_samples,
                         commands=2, periodic=periodic, plateau_cycles=span)
            cursor += 2
        bursts.append(burst)
    return bursts


@pytest.mark.parametrize('first,last,count', [
    (1, 4800, 4800), (1, 10000, 10000), (10000, 1, 10000),
    (1.2, 33.8, 91), (33.8, 1.2, 91), (95.2, 113.7, 83),
    (1048601.2, 1048678.7, 91), (1048678.7, 1048601.2, 91),
])
def test_fractional_accumulator_matches_every_existing_quantizer_point(first, last, count):
    values = np.linspace(first / 4800., last / 4800., count)
    accumulator = sample_accumulator(values, 300.)
    assert [accumulator.samples_at(i) for i in range(count)] == [samples_from_us(v, 300.) for v in values]


@pytest.mark.parametrize('first,last,count,outer,reps', [
    (1, 4800, 4800, 1, 1), (1, 10000, 10000, 1, 1),
    (10000, 1, 10000, 1, 1), (1.2, 113.8, 137, 2, 3),
    (113.8, 20.2, 80, 2, 2), (1048576, 1048660, 85, 2, 2),
    (1048660, 1048576, 85, 2, 2),
])
def test_actual_hardware_loop_exact_sample_counts(first, last, count, outer, reps):
    program = compile_sweep(first, last, count, outer=outer, reps=reps)
    model, events = execute(program)
    bursts = decode_bursts(program, events)
    from clock_time import integer_sample_sweep
    initial, step, _ = integer_sample_sweep(first/4800., last/4800., count, 4800.)
    one_axis = [initial+i*step for i in range(count)]
    expected = [n for _ in range(outer) for n in one_axis for _ in range(reps)]
    assert [b['samples'] for b in bursts] == expected
    assert [b['periodic'] for b in bursts] == [plan_duration(n).periodic_plateau for n in expected]
    assert model.dmem[program.COUNTER_ADDR] == len(expected)
    assert all(not table.get('fine_row') for table in program._rf_point_tables)
    assert len(program._runtime_dmem_words) == 0


def test_duration_memory_is_constant_in_point_count_with_other_feature_reported_separately():
    programs = [compile_sweep(1, 10000, count, frequency_count=2) for count in (2, 100, 4800, 10000)]
    assert len({len(p.binprog) for p in programs}) == 1
    assert {len(p._runtime_dmem_words) for p in programs} == {2}
    assert all(len(p._fine_axis_field_keys) == 1 for p in programs)
    assert all(p.summary()['rf_fine_duration'][0]['duration_table_words'] == 0 for p in programs)


@pytest.mark.parametrize('first,last', [(17, 113), (113, 17)])
def test_following_pulse_uses_same_exact_preceding_duration_state(first, last):
    program = compile_sweep(first, last, 97, outer=2, reps=2, preceding=True)
    _, events = execute(program)
    bursts = decode_bursts(program, events)
    expected = [samples_from_us(v, 300.) for v in np.linspace(first / 4800., last / 4800., 97)]
    for (first_burst, following), samples in zip(zip(bursts[::2], bursts[1::2]),
                                               [n for _ in range(2) for n in expected for _ in range(2)]):
        assert first_burst['samples'] == samples
        assert following['samples'] == 113
        assert following['start'] - first_burst['start'] == (samples + 15) // 16 + 60
    assert len(bursts) == 2 * 97 * 2 * 2


def test_single_point_and_repeated_equal_endpoints():
    for count in (1, 17):
        program = compile_sweep(97.2, 97.2, count, reps=3)
        _, events = execute(program)
        assert [b['samples'] for b in decode_bursts(program, events)] == [98] * count * 3


def test_non_affine_quantizer_input_is_rejected_instead_of_silently_rounded():
    with pytest.raises(ValueError, match='affine sample staircase'):
        sample_accumulator(np.array([1., 100., 2.]) / 4800., 300.)


def test_register_sample_range_is_explicit():
    with pytest.raises(ValueError, match='signed 32-bit'):
        sample_accumulator([2**31 / 4800.], 300.)


@pytest.mark.parametrize('count', [1, 2])
def test_fixed_or_swept_sample_range_cannot_wrap_hardware_register(count):
    with pytest.raises(ValueError, match='signed 32-bit'):
        compile_sweep(2**32, 2**32, count)


def test_unswept_sample_range_cannot_wrap_when_another_axis_varies():
    cfg = _v5_soccfg()
    seq = FineTuneSequence(('awg_0',)).add_set('gate', .01, 400_000_000)
    seq.add_amplitude_sweep('gate', 'awg_0', .01, .02, 2)
    pulse = RfPulseConfig(0, 'gate', 2**28, 1000, delay_tproc_cycles=30,
                         fine_duration=True, duration_samples=2**32)
    with pytest.raises(ValueError, match='signed 32-bit'):
        seq.make_program(cfg, awg_channels=(1,), rf_pulses=(pulse,),
                         compile_validation_mode='boundary')


@pytest.mark.parametrize('first,last', [(1, 113), (113, 1)])
def test_grouped_voltage_and_duration_share_one_loop(first, last):
    program = compile_sweep(first, last, 113, outer=113, reps=2, grouped=True)
    model, events = execute(program)
    bursts = decode_bursts(program, events)
    expected = [samples_from_us(v, 300.) for v in np.linspace(first / 4800., last / 4800., 113)]
    assert program.sequence.sweep_shape == (113,)
    assert program.sequence.point_indices(112) == (112, 112)
    assert [b['samples'] for b in bursts] == [n for n in expected for _ in range(2)]
    assert model.dmem[program.COUNTER_ADDR] == 226
    assert len(program._runtime_dmem_words) == 0


def test_grouped_composite_durations_preserve_spilled_state_and_outer_reset():
    cfg = _v5_soccfg()
    seq = FineTuneSequence(('awg_0',)).add_set('gate', .01, 4000)
    seq.add_amplitude_sweep('gate', 'awg_0', .01, .02, 2)
    ranges = ((1., 49.), (113., 81.), (1.2, 31.8), (97., 97.),
              (49., 1.), (81., 113.), (33.8, 1.2), (112., 96.))*4
    pulses = []
    for index, (first, last) in enumerate(ranges):
        parameter = f'd{index}'
        seq.add_rf_duration_sweep('gate', 0, first/4800., last/4800., 17,
                                 parameter_name=parameter, sequence_fabric_mhz=300.)
        samples = samples_from_us(first/4800., 300.)
        pulses.append(RfPulseConfig(0, 'gate', max(3, (samples+15)//16), 1000,
            freq_mhz=10., phase_degrees=45., delay_tproc_cycles=30+index*100,
            fine_duration=True, duration_samples=samples, duration_parameter=parameter,
            event_id=f'item_{index}'))
    seq.set_sweep_groups({(a.axis_kind,a.segment_name,a.output_name):'duration_group'
                         for a in seq.sweep_axes[1:]})
    program = seq.make_program(cfg, awg_channels=(1,), rf_pulses=pulses,
        repetitions_per_sweep=2, compile_validation_mode='boundary')
    program.compile()
    model, events = execute(program)
    fields = [field for field in program._sweep_fields if field['key'][0]=='fine_duration_state']
    assert any(field.get('storage')=='dmem' for field in fields)
    expected = []
    values = [axis.points for axis in seq.sweep_axes[1:]]
    for _ in range(2):
        for point in range(17):
            for _ in range(2):
                expected.extend(samples_from_us(axis[point], 300.) for axis in values)
    assert [burst['samples'] for burst in decode_bursts(program, events)] == expected
    assert seq.sweep_shape == (2, 17)
    assert model.dmem[program.COUNTER_ADDR] == 68
    assert len(program._runtime_dmem_words) == 0  # State spills are initialized by instructions.
