"""Shared-index execution, correlated timing and Cartesian reset coverage."""
import numpy as np
import pytest
from qick.sim import QickSim  # noqa: F401
from qick.awg_tuning import TProcV1BehaviorModel
from qick_fine_tune_sweep import FineTuneSequence, RfPulseConfig
from test_qick_fine_tune_sweep import _independent_awg_soccfg, _expected_words, _shared_tmux_soccfg


def grouped_sequence(count=3):
    sequence = FineTuneSequence(('a', 'b'))
    sequence.add_set('zero', (0., 0.), 300)
    sequence.add_ramp('rise', 3)
    sequence.add_set('hold', (-.2, .1), 3000)
    sequence.add_ramp('fall', 3)
    sequence.add_set('reset', (0., 0.), 300)
    sequence.add_ramp_duration_sweep('rise', .01, 1., count)
    sequence.add_ramp_duration_sweep('fall', .01, 1., count)
    sequence.add_amplitude_sweep('hold', 'a', -.2, .2, count)
    sequence.add_amplitude_sweep('hold', 'b', .1, -.1, count)
    sequence.set_sweep_groups({
        ('ramp_duration', 'rise', 'all_awg_outputs'): 'time',
        ('ramp_duration', 'fall', 'all_awg_outputs'): 'time',
        ('amplitude', 'hold', 'a'): 'voltage',
        ('amplitude', 'hold', 'b'): 'voltage',
    })
    return sequence


def config():
    cfg = _independent_awg_soccfg(2)
    for gen in cfg['gens']:
        gen.update(type='axis_awg_tuning_v2', frac=18, step_width=32)
    return cfg


def test_grouped_coordinates_retain_members_and_reduce_dimensions():
    seq = grouped_sequence(100)
    assert seq.sweep_shape == (100, 100)
    assert seq.model_shape == (100, 1, 100, 1)
    assert seq.sweep_point_count == 10000
    assert seq.sweep_coordinates.shape == (10000, 4)
    for point in (0, 99, 100, 5050, 9999):
        ti, vi = divmod(point, 100)
        assert seq.point_indices(point) == (ti, ti, vi, vi)
        coordinate = seq.sweep_coordinate(point)
        assert coordinate[0] == coordinate[1]
        assert coordinate[2] == pytest.approx(-2 * coordinate[3])


@pytest.mark.parametrize('validation', ['full', 'boundary'])
@pytest.mark.parametrize('same_group', [False, True])
def test_actual_loop_commands_match_grouped_coordinates_and_repeats(validation, same_group):
    seq = grouped_sequence()
    if same_group:
        seq.set_sweep_groups({seq.sweep_axis_key(a): 'all' for a in seq.sweep_axes})
    program = seq.make_program(config(), awg_channels=(0, 1),
                               repetitions_per_sweep=2, compile_validation_mode=validation)
    program.compile()
    model = TProcV1BehaviorModel(strict=True)
    program.load_runtime_dmem_into_model(model)
    model.run(program, max_steps=1_000_000)
    assert not model.timing_conflicts
    if validation == 'full':
        assert [event.word for event in model.output_events] == _expected_words(program)
    else:
        full = seq.make_program(config(), awg_channels=(0, 1), repetitions_per_sweep=2)
        assert [event.word for event in model.output_events] == _expected_words(full)
    assert len(program._sweep_axis_runtime) == (1 if same_group else 2)
    assert len(model.output_events) == seq.sweep_point_count * 2 * 10
    # Every RAMP command carries its group's indexed duration, including resets.
    for point in range(seq.sweep_point_count):
        events = model.output_events[point*20:(point+1)*20]
        duration = seq.segment_duration_cycles_at(point, 1) * 16
        for shot in (events[:10], events[10:]):
            for event in (shot[2], shot[3], shot[6], shot[7]):
                assert (event.word >> 64) & 0xffffffff == duration


def test_mismatched_counts_and_missing_member_fail_before_compilation():
    seq = grouped_sequence()
    with pytest.raises(ValueError, match='missing parameters'):
        seq.set_sweep_groups({('amplitude', 'missing', 'a'): 'g'})
    seq.add_amplitude_sweep('hold', 'b', .1, -.1, 4)
    with pytest.raises(ValueError, match='share one point count'):
        seq.set_sweep_groups({('amplitude','hold','a'):'g', ('amplitude','hold','b'):'g'})


def test_grouped_rf_duration_and_frequency_table_tracks_one_index():
    cfg = _shared_tmux_soccfg()
    seq = FineTuneSequence(('a',)).add_set('hold', .01, 3000)
    seq.add_rf_duration_sweep('hold', 0, .1, .3, 3, sequence_fabric_mhz=300.)
    seq.add_rf_frequency_sweep('hold', 0, 10., 30., 3)
    seq.set_sweep_groups({seq.sweep_axis_key(a): 'rf' for a in seq.sweep_axes})
    rf = RfPulseConfig(0, 'hold', 30, 1000, freq_mhz=10., delay_tproc_cycles=30)
    program = seq.make_program(cfg, awg_channels=(1,), rf_pulses=(rf,), repetitions_per_sweep=2)
    model = TProcV1BehaviorModel(strict=True)
    program.load_runtime_dmem_into_model(model)
    model.run(program)
    events = [e for e in model.output_events if (e.word >> 152) & 255 == 0]
    assert len(events) == 6
    assert [(e.word >> 128) & 65535 for e in events] == [30,30,60,60,90,90]
    assert [e.word & 0xffffffff for e in events] == [cfg.freq2reg(f, gen_ch=0) for f in (10,10,20,20,30,30)]
    assert not model.timing_conflicts


@pytest.mark.parametrize('mode', ['fixed_voltage', 'fixed_time'])
@pytest.mark.parametrize('rc_enabled', [False, True])
@pytest.mark.parametrize('same_group', [False, True])
def test_grouped_matrix_dc_rc_actual_commands(mode, rc_enabled, same_group):
    from test_rc_precompensation import upgraded
    from qick_rc_validation import validate_sequence_rc_range
    seq = grouped_sequence(4)
    seq.set_cross_capacitance(((1., .2), (-.15, 1.)))
    if same_group:
        seq.set_sweep_groups({seq.sweep_axis_key(a): 'all' for a in seq.sweep_axes})
    seq.set_bias_t_compensation(.1, mode=mode,
        fixed_duration_cycles=3000 if mode == 'fixed_time' else None)
    cfg = config()
    if rc_enabled:
        cfg = upgraded(cfg)
        seq.set_rc_compensation(30.)
        assert validate_sequence_rc_range(seq, 300., 800.)
    traces = []
    for validation in ('full', 'boundary'):
        program = seq.make_program(cfg, awg_channels=(0, 1),
            repetitions_per_sweep=2, compile_validation_mode=validation)
        model = TProcV1BehaviorModel(strict=True)
        program.load_runtime_dmem_into_model(model)
        model.run(program, max_steps=3_000_000)
        assert not model.timing_conflicts
        traces.append([(e.cycle, e.tproc_ch, e.word) for e in model.output_events])
    assert traces[0] == traces[1]


def test_correlated_duration_area_checks_interior_not_just_endpoints():
    from qick_rc_validation import sweep_corners, validate_sequence_rc_range
    seq = FineTuneSequence(('x',)).add_set('gate', 0., 300)
    seq.add_hold_duration_sweep('gate', 1., 100., 3)
    seq.add_amplitude_sweep('gate', 'x', .5, 0., 3)
    seq.set_sweep_groups({seq.sweep_axis_key(a): 'correlated' for a in seq.sweep_axes})
    seq.set_rc_compensation(10.)
    assert sweep_corners(seq) == (0, 1, 2)
    with pytest.raises(ValueError, match='sweep corner 1'):
        validate_sequence_rc_range(seq, 300., 800.)


@pytest.mark.parametrize('mode', ['fixed_voltage', 'fixed_time'])
@pytest.mark.parametrize('duration_first', [False, True])
def test_grouped_rf_extension_and_voltage_dc_area(mode, duration_first):
    cfg = _shared_tmux_soccfg()
    seq = FineTuneSequence(('a',)).add_set('gate', .01, 3000)
    axes = [lambda: seq.add_amplitude_sweep('gate', 'a', .01, .04, 4),
            lambda: seq.add_rf_duration_sweep('gate', 0, .1, 10., 4,
                sequence_fabric_mhz=300., segment_length_mode='extend_by_rf_duration')]
    for add in axes[::(-1 if duration_first else 1)]:
        add()
    seq.set_sweep_groups({seq.sweep_axis_key(a): 'g' for a in seq.sweep_axes})
    seq.set_bias_t_compensation(.1, mode=mode,
        fixed_duration_cycles=3000 if mode == 'fixed_time' else None)
    rf = RfPulseConfig(0, 'gate', 30, 1000, freq_mhz=10., delay_tproc_cycles=30)
    traces = []
    for validation in ('full', 'boundary'):
        program = seq.make_program(cfg, awg_channels=(1,), rf_pulses=(rf,),
            repetitions_per_sweep=2, compile_validation_mode=validation)
        if mode == 'fixed_voltage':
            assert program._bias_t_max_duration_q_error <= 2
        else:
            assert program._bias_t_max_target_code_error <= 8
        model = TProcV1BehaviorModel(strict=True)
        program.load_runtime_dmem_into_model(model)
        model.run(program, max_steps=1_000_000)
        assert not model.timing_conflicts
        traces.append([(e.cycle, e.tproc_ch, e.word) for e in model.output_events])
    assert traces[0] == traces[1]


@pytest.mark.parametrize('mode', ['fixed_voltage', 'fixed_time'])
@pytest.mark.parametrize('rc_enabled', [False, True])
@pytest.mark.parametrize('same_group', [False, True])
def test_v5_grouped_matrix_uses_fpga_dc_area(mode, rc_enabled, same_group):
    from test_qick_dc_area import config as dc_config
    from qick.output_sum import apply_output_sum_config
    from qick_dc_area import point_plan
    cfg = dc_config(2)
    for gen in cfg['gens']:
        apply_output_sum_config(gen, dict(output_sum_version=2,
            output_sum_role='awg', post_output_latency_cycles=2,
            output_sum_awg_delay_cycles=13))
        assert gen['output_latency_cycles'] == 31
    seq = grouped_sequence(4)
    seq.set_cross_capacitance(((1., .2), (-.15, 1.)))
    if same_group:
        seq.set_sweep_groups({seq.sweep_axis_key(a): 'all' for a in seq.sweep_axes})
    seq.set_bias_t_compensation(.3, mode=mode,
        fixed_duration_cycles=3000 if mode == 'fixed_time' else None)
    seq.set_rc_compensation(30., enabled=rc_enabled)
    program = seq.make_program(cfg, awg_channels=(0, 1), repetitions_per_sweep=2,
                               compile_validation_mode='boundary')
    assert program.summary()['hardware_dc_area']
    assert program.summary()['bias_t_duration_execution'] == 'hardware_pre_iir_area'
    assert program._bias_t_fields == ()
    for point in range(seq.sweep_point_count):
        for output in range(2):
            plan = point_plan(program, point, output)
            assert plan['residual'] == 0
            lo, hi = program._hardware_dc_net_bounds[output]
            assert lo <= plan['area'] <= hi
    model = TProcV1BehaviorModel(strict=True)
    program.load_runtime_dmem_into_model(model)
    model.run(program, max_steps=3_000_000)
    assert not model.timing_conflicts
    commands = [e.word for e in model.output_events if e.word & (1 << 150)]
    assert sum(((w >> 146) & 3) == (1 if mode == 'fixed_voltage' else 2)
               for w in commands) == 2 * seq.sweep_point_count * 2


@pytest.mark.parametrize('duration_kind', ['hold', 'rf_extension'])
def test_v5_correlated_duration_voltage_bounds_include_interior(duration_kind):
    from test_output_sum_v5_gui import configuration
    from qick_dc_area import point_plan, validation_points
    cfg = configuration()
    seq = FineTuneSequence(('a',)).add_set('gate', .5, 300)
    if duration_kind == 'hold':
        seq.add_hold_duration_sweep('gate', 1., 100., 3)
        rfs = ()
    else:
        seq.add_rf_duration_sweep('gate', 0, 1., 100., 3,
            segment_length_mode='extend_by_rf_duration')
        rfs = (RfPulseConfig(0, 'gate', 300, 1000, freq_mhz=10., delay_tproc_cycles=30),)
    seq.add_amplitude_sweep('gate', 'a', .5, 0., 3)
    seq.set_sweep_groups({seq.sweep_axis_key(a): 'correlated' for a in seq.sweep_axes})
    seq.set_bias_t_compensation(.1)
    prog = seq.make_program(cfg, awg_channels=(1,), rf_pulses=rfs,
                            compile_validation_mode='boundary')
    assert prog._hardware_dc_area
    assert validation_points(prog) == (0, 1, 2)
    middle = point_plan(prog, 1, 0)
    lo, hi = prog._hardware_dc_net_bounds[0]
    assert lo <= middle['area'] <= hi
    assert prog._hardware_dc_deadline > middle['words'] + 17
    seq.set_bias_t_compensation(.1, mode='fixed_time', fixed_duration_cycles=3000)
    with pytest.raises(ValueError):
        seq.make_program(cfg, awg_channels=(1,), rf_pulses=rfs,
                         compile_validation_mode='boundary')
