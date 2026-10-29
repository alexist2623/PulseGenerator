"""Common-latency firmware must preserve requested times, including one clock."""
import pytest
from qick.sim import QickSim  # noqa: F401 - desktop driver bootstrap
from qick.awg_tuning import AxisAwgTuningV4
from qick.output_sum import apply_output_sum_config
from qick_fine_tune_sweep import FineTuneSequence, RfPulseConfig
from test_output_sum_v4 import configuration as old_configuration


def configuration():
    cfg = old_configuration(False)
    cfg['tprocs'][0]['timed_output_spacing_cycles'] = 1
    cfg['gens'][1].update(type='axis_awg_tuning_v4', output_latency_cycles=23,
                         ramp_startup_latency_cycles=0, ramp_guard_cycles=0)
    for ch, role in ((0, 'rf'), (1, 'awg')):
        apply_output_sum_config(cfg['gens'][ch], dict(output_sum_version=3,
            output_sum_role=role, post_output_latency_cycles=2,
            output_sum_awg_delay_cycles=6))
    return cfg


@pytest.mark.parametrize('rise,hold,fall', [(1,1,1), (1,2,1), (2,1,2)])
def test_short_segments_have_no_advance_or_added_hold(rise, hold, fall):
    seq = (FineTuneSequence(('x',)).add_set('zero', (0.,), 1)
           .add_ramp('rise', rise).add_set('hold', (.02,), hold)
           .add_ramp('fall', fall).add_set('end', (0.,), 1))
    p = seq.make_program(configuration(), awg_channels=(1,))
    p.compile()
    starts = (0, 1, 1+rise, 1+rise+hold, 1+rise+hold+fall)
    assert p.timing['segment_starts'] == tuple(t+31 for t in starts)
    assert p.timing['segment_ends'] == tuple(t+31 for t in (*starts[1:], starts[-1]+1))
    assert tuple(p.timing['command_times'].values()) == starts


@pytest.mark.parametrize('delay', [1,2,3])
def test_queued_rf_command_preserves_one_cycle_offset(delay):
    seq = FineTuneSequence(('x',)).add_set('hold', (.02,), 300)
    p = seq.make_program(configuration(), awg_channels=(1,),
        rf_pulse=RfPulseConfig(0, 'hold', 30, 2000, delay_tproc_cycles=delay, freq_mhz=192.))
    p.compile()
    assert p.aux_timing['rf_0_start'] == delay
    assert p.awg_output_latency_tproc == 31


def test_identical_shared_port_timestamp_is_rejected():
    seq = FineTuneSequence(('x',)).add_set('hold', (.02,), 300)
    with pytest.raises(ValueError, match='v6 RF command conflicts'):
        seq.make_program(configuration(), awg_channels=(1,),
            rf_pulse=RfPulseConfig(0, 'hold', 30, 2000, freq_mhz=192.))


def test_old_dispatcher_cannot_be_used_for_v6():
    cfg = configuration()
    cfg['tprocs'][0]['timed_output_spacing_cycles'] = 3
    with pytest.raises(ValueError, match='version 2 timed dispatcher'):
        FineTuneSequence(('x',)).add_set('hold', (.02,), 1).make_program(cfg, awg_channels=(1,))


def test_driver_and_output_combiner_latency_are_separate():
    driver = AxisAwgTuningV4(dict(type='QICK:QICK:axis_awg_tuning_v4:1.0',
        fullpath='awg', parameters=dict(EXTRA_Y_PIPE_STAGES='3', RC_PRECOMP_VERSION='1')))
    assert driver.cfg['output_latency_cycles'] == 23
    assert driver.cfg['ramp_startup_latency_cycles'] == driver.cfg['ramp_guard_cycles'] == 0
    route = dict(output_sum_version=3, output_sum_role='awg',
                 post_output_latency_cycles=2, output_sum_awg_delay_cycles=6)
    apply_output_sum_config(driver.cfg, route)
    apply_output_sum_config(driver.cfg, route)
    assert driver.cfg['output_latency_cycles'] == 31


@pytest.mark.parametrize('rise,hold,fall', [(1,1,1), (1,2,1), (2,1,2)])
def test_gui_timestamp_subtraction_keeps_exact_clock_lengths(rise, hold, fall):
    from dc_waveform_core import _cycles_from_ns
    times = [100., 100.+rise*10/3, 100.+(rise+hold)*10/3,
             100.+(rise+hold+fall)*10/3]
    assert [_cycles_from_ns(b-a, 300.) for a,b in zip(times,times[1:])] == [rise,hold,fall]
    # A real fractional excess still rounds up; this is only FP tolerance.
    assert _cycles_from_ns(10/3 + 1e-6, 300.) == 2


@pytest.mark.parametrize('first,last', [(1,32), (95,112), (4785,4800)])
def test_v6_fine_rf_hardware_loop_lengths_and_outer_reset(monkeypatch, first, last):
    import test_rf_fine_duration_compiler as reference
    from qick_rf_fine_duration import plan_duration
    monkeypatch.setattr(reference, '_shared_tmux_soccfg', configuration)
    program, events = reference.execute(first, last, last-first+1, outer=2)
    decoded = reference.active_samples(program, events)
    cursor = 0
    for samples in list(range(first,last+1))*2:
        for repeat in range(2):
            count = 1 + plan_duration(samples).long
            assert sum(v[1] for v in decoded[cursor:cursor+count]) == samples
            cursor += count
    assert cursor == len(decoded)
    assert program._dispatch_spacing == 1
    assert len(program.envelopes[0]['envs']) == 16
    assert sum(len(v['data']) for v in program.envelopes[0]['envs'].values()) == 2048


@pytest.mark.parametrize('policy', ['after_readout', 'overlap_readout'])
@pytest.mark.parametrize('mode', ['fixed_time', 'fixed_voltage'])
def test_v6_readout_dc_rc_timing_keeps_dc_separate(policy, mode):
    from test_dc_readout_timing import fixture
    from qick.awg_tuning import TProcV1BehaviorModel
    cfg, seq, readout, rf = fixture(policy, mode=mode, sweep='hold')
    cfg['tprocs'][0]['timed_output_spacing_cycles'] = 1
    for gen in cfg['gens'][:2]:
        gen.update(type='axis_awg_tuning_v4', output_latency_cycles=31,
                   native_output_latency_cycles=23, output_sum_version=3,
                   output_sum_role='awg', output_sum_awg_delay_cycles=6,
                   ramp_startup_latency_cycles=0, ramp_guard_cycles=0,
                   dc_area_version=1, dc_area_bits=48, dc_start_cycles=17,
                   dc_area_pipeline_cycles=5)
    program = seq.make_program(cfg, awg_channels=(0,1), ddr_readout=readout,
                               repetitions_per_sweep=2)
    program.compile()
    model = TProcV1BehaviorModel(strict=True)
    program.load_runtime_dmem_into_model(model);model.run(program)
    assert not model.timing_conflicts
    assert program.awg_output_latency_tproc == 31
    assert program._hardware_dc_area
    for index in range(seq.sweep_point_count):
        for dc in program.fir_dc_timing_preview(index)['dc_outputs']:
            assert (dc['dac_start_us']-dc['command_start_us'])*300 == pytest.approx(43)


@pytest.mark.parametrize('shape', [(1,1,1), (1,2,1), (2,1,2), None])
def test_circuit_v6_uses_common_latency_and_one_clock_holds(shape):
    from dataclasses import replace
    import test_quantum_circuit_execution as reference
    cfg = reference.config()
    cfg['tprocs'][0]['timed_output_spacing_cycles'] = 1
    for ch,gen in enumerate(cfg['gens']):
        gen.update(output_sum_version=3, output_latency_cycles=31,
                   output_sum_role='awg' if ch%2 else 'rf')
        if ch%2:
            gen.update(type='axis_awg_tuning_v4', ramp_startup_latency_cycles=0,
                       ramp_guard_cycles=0)
    doc = reference.document()
    if shape:
        rise,hold,fall = shape
        doc['awg_baselines'] = [dict(channel=1, voltage_mv=0)]
        doc['gates'] = [
            reference.gate('rise',0,rise/300,awg=[reference.awg(value=20,ramp=rise/300)]),
            reference.gate('fall',(rise+hold)/300,fall/300,awg=[reference.awg(value=0,ramp=fall/300)]),
            reference.gate('read',1,20,acquisition=dict(delay_us=0,duration_us=20))]
    templates = tuple(replace(t,att1_db=0.,att2_db=0.,filter_type='bypass')
                      for t in reference.arguments()['rf_templates'])
    result = reference.compile_doc(doc,cfg,rf_templates=templates)
    if shape:
        segments = result.sequence.segments
        ri = next(i for i,s in enumerate(segments) if s.kind=='ramp')
        assert [s.duration_cycles for s in segments[ri:ri+3]] == list(shape)
    else:
        assert set(result.metadata['rf_output_latency_tproc_cycles'].values()) == {31}
        assert set(result.metadata['rf_output_latency_profiles'].values()) == {'qstl_v6_common_native_latency'}
