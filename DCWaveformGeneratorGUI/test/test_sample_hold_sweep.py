"""Execute sample HOLD loops, branch boundaries, and RF timing without hardware."""
import numpy as np
import pytest
import qick.sim
from qick.awg_tuning import TProcV1BehaviorModel
from qick.ramp_v7 import ramp_packet
from dc_waveform_core import PulseSequence, QickHoldDurationSweepSpec, build_qick_sequence
from qick_fine_tune_sweep import RfPulseConfig
from qick_execution_coordinates import build_execution_coordinates
from test_awg_v7_fine import configuration

DT = 1/4800


def compile_hold(first=1, last=20001, count=1001, *, compensate=False, rf=False, outer=False, grouped=False):
    pulse = PulseSequence(0, 1000)
    pulse.add_flat_ramp(10, 10, 100)
    pulse.add_flat_ramp(10, 1000, 0)
    sequence = build_qick_sequence((pulse,), sweeps=(QickHoldDurationSweepSpec('set_1', first*DT, last*DT, count),))
    if outer or grouped:
        sequence.add_amplitude_sweep('set_1', 'awg_0', .1, .2, count if grouped else 2)
        sequence.sweeps.reverse()
        if grouped:
            sequence.set_sweep_groups({sequence.sweep_axis_key(a): 'paired' for a in sequence.sweeps})
    if compensate:
        sequence.set_bias_t_compensation(.2, mode='fixed_time', fixed_duration_cycles=9000)
        sequence.set_rc_compensation(200.)
    pulses = (RfPulseConfig(0, 'set_2', 7, 1000, fine_duration=True,
                           duration_samples=97, delay_tproc_cycles=30),) if rf else ()
    p = sequence.make_program(configuration(), awg_channels=(1,), rf_pulses=pulses,
                              repetitions_per_sweep=2, compile_validation_mode='boundary')
    p.compile()
    return p


def execute(p):
    model = TProcV1BehaviorModel(strict=True)
    p.load_runtime_dmem_into_model(model)
    model.run(p, max_steps=10_000_000)
    assert model.ended and not model.timing_conflicts
    return model


@pytest.mark.parametrize('reverse', (False, True))
@pytest.mark.parametrize('compensate', (False, True))
def test_long_hold_crosses_branch_without_point_memory(reverse, compensate):
    first, last = (20001, 1) if reverse else (1, 20001)
    p = compile_hold(first, last, compensate=compensate)
    model = execute(p)
    packets = [e for e in model.output_events if e.word >> 151 & 1]
    expected = []
    cursor = 0
    for hold in range(first, last + (-1 if reverse else 1), -20 if reverse else 20):
        fall_start = 48 + hold
        if fall_start <= 16383:
            values = [ramp_packet(start=0, target=4096, rise_start=0, rise_end=48,
                       fall_start=fall_start, fall_end=fall_start+48, end_target=0, tmux=1)]
        else:
            values = [ramp_packet(start=0, target=4096, rise_start=0, rise_end=48, tmux=1),
                      ramp_packet(start=4096, target=0, rise_start=fall_start%16,
                                  rise_end=fall_start%16+48, tmux=1)]
        for _ in range(2):
            for value in values:
                expected.append(sum(int(w) << (32*i) for i,w in enumerate(value['words'])))
            if len(values) == 2:
                assert packets[cursor+1].cycle-packets[cursor].cycle == fall_start//16
            cursor += len(values)
    assert [e.word for e in packets] == expected
    assert not p._runtime_dmem_words
    assert len(p.binprog) < 220
    assert p.summary()['sample_hold_sweep']['duration_point_table_words'] == 0


@pytest.mark.parametrize('grouped', (False, True))
def test_nested_grouped_voltage_reset_and_rf_anchor(grouped):
    p = compile_hold(1, 33, 33, outer=not grouped, grouped=grouped, rf=True)
    model = execute(p)
    packets = [e.word for e in model.output_events if e.word >> 151 & 1]
    expected = []
    for point in range(p.sequence.sweep_point_count):
        expected.extend([sum(int(w) << (32*i) for i,w in enumerate(c.words))
                         for c in p.sample_hold_commands(point) if c.kind == 'ramp']*2)
    assert packets == expected
    from test_rf_fine_duration_registers import decode_bursts
    bursts = decode_bursts(p, [e for e in model.output_events if e.word >> 152 & 255 == 0])
    ramps = [e for e in model.output_events if e.word >> 151 & 1]
    for index, burst in enumerate(bursts):
        point = index//2
        hold = p.sequence.sweep_axes[-1].duration_sample_points[p.sequence.point_indices(point)[-1]]
        assert burst['start']-ramps[index].cycle == (96+hold+15)//16+30
        assert burst['samples'] == 97


def test_export_preserves_omitted_falling_packet_and_fractional_cycles():
    p = compile_hold(rf=True)
    x = build_execution_coordinates(p, point_indices=[0, 819, 1000])
    fall = next(c for c in x['awg_commands'] if c['segment_name'] == 'ramp_1_to_2')
    assert fall['columns']['emitted'] == [False, True, True]
    assert fall['columns']['command_words_u32'][0] is None
    hold = next(c for c in x['segment_duration_fabric_cycles'] if c['segment_name'] == 'set_1')
    assert hold['coordinates'] == [1/16, 16381/16, 20001/16]


def test_memory_does_not_scale_with_point_count():
    programs = [compile_hold(1, 1+(n-1)*20, n) for n in (2, 100, 1001)]
    assert len({len(p.binprog) for p in programs}) == 1
    assert all(not p._runtime_dmem_words for p in programs)


def test_constant_integer_step_matches_existing_display():
    from time_display import time_sweep_display
    from qick_fine_tune_sweep import RfDurationSweep
    from clock_time import integer_sample_sweep
    assert integer_sample_sweep(.001, .3, 300, 4800) == (5, 5, 1500)
    assert time_sweep_display(.001, .3, 300, fabric_mhz=300, samples_per_clock=16,
                              integer_step=True)['end'] == '312.5 ns'
    p = compile_hold(5, 1440, 300)
    assert p.sequence.sweep_axes[0].duration_sample_points[-1] == 1500



@pytest.mark.parametrize('policy', ('after_readout', 'overlap_readout'))
@pytest.mark.parametrize('ramp_sweep', (False, True))
def test_ddr_trigger_moves_on_sample_carries_with_compensation(policy, ramp_sweep):
    from qick_fine_tune_sweep import DdrFirReadoutConfig
    from test_qick_fine_tune_sweep import _fir_soccfg
    cfg = _fir_soccfg(fir_rate_profile='50_ksps', ddr_trigger_port=7)
    cfg['gens'] = configuration()['gens']
    cfg['tprocs'][0]['timed_output_spacing_cycles'] = 1
    cfg['readouts'][0]['tproc_ctrl'] = 2
    seq = compile_hold(1,33,33).sequence
    if ramp_sweep:
        seq.add_ramp_duration_sweep('ramp_0_to_1', DT, 19*DT, 5, sample_grid=True)
        seq.sweeps.reverse()
    seq.set_bias_t_compensation(.2, mode='fixed_time', fixed_duration_cycles=300)
    seq.set_rc_compensation(200.)
    ddr = DdrFirReadoutConfig(0, 2, 'set_2', margin_input_samples=0, dc_compensation_timing=policy)
    rf = RfPulseConfig(0, 'set_2', 7, 1000, fine_duration=True, duration_samples=97, delay_tproc_cycles=30)
    p = seq.make_program(cfg, awg_channels=(1,), ddr_readout=ddr, rf_pulses=(rf,),
                         repetitions_per_sweep=2, compile_validation_mode='boundary')
    model = execute(p)
    ramps = [e for e in model.output_events if e.word >> 151 & 1]
    triggers = [e for e in model.output_events if e.tproc_ch == 7 and e.word]
    from test_rf_fine_duration_registers import decode_bursts
    bursts = decode_bursts(p, [e for e in model.output_events if e.tproc_ch == 0 and e.word >> 152 & 255 == 0])
    assert all(b['samples'] == 97 for b in bursts)
    assert [b['start'] for b in bursts] == [e.cycle-1 for e in triggers]
    assert len(triggers) == len(ramps) == 66*(5 if ramp_sweep else 1)
    for index, (trigger, ramp) in enumerate(zip(triggers, ramps)):
        point = index//2
        hold = point//5+1 if ramp_sweep else point+1
        rise = (1,6,10,15,19)[point%5] if ramp_sweep else 48
        assert trigger.cycle-ramp.cycle == (rise+48+hold+15)//16+31


@pytest.mark.parametrize('count', (5, 101))
@pytest.mark.parametrize('hold_outer', (False, True))
def test_ramp_tables_do_not_include_independent_hold_points(count, hold_outer):
    seq = compile_hold(1, 1+(count-1)*20, count).sequence
    seq.add_ramp_duration_sweep('ramp_0_to_1', DT, 19*DT, 5, sample_grid=True)
    if hold_outer:
        seq.sweeps.reverse()
    p = seq.make_program(configuration(), awg_channels=(1,), repetitions_per_sweep=2,
                         compile_validation_mode='boundary')
    model = execute(p)
    expected = []
    for point in range(seq.sweep_point_count):
        expected.extend([sum(int(w)<<(32*i) for i,w in enumerate(c.words))
                         for c in p.sample_hold_commands(point) if c.kind == 'ramp']*2)
    assert [e.word for e in model.output_events if e.word >> 151 & 1] == expected
    assert len(p._runtime_dmem_words) < 120
    hold_axis = next(i for i,a in enumerate(seq.sweep_axes) if a.axis_kind == 'hold_duration')
    assert all(hold_axis not in field.get('duration_axis_indices', ()) for field in p._sweep_fields)


def test_clock_aligned_hold_after_one_sample_ramp_keeps_exact_plateau():
    pulse = PulseSequence(0, 1000)
    pulse.add_flat_ramp(10, 100, 100)
    pulse.add_flat_ramp(10, 1000, 0)
    pulse.set_ramp_duration_ns(1, 1000*DT)
    pulse.set_ramp_duration_ns(2, 1000*DT)
    seq = build_qick_sequence((pulse,), sweeps=(QickHoldDurationSweepSpec('set_1', .1, .2, 2),))
    p = seq.make_program(configuration(), awg_channels=(1,))
    assert p._sample_hold
    words = [e.word for e in execute(p).output_events if e.word >> 151 & 1]
    for word, hold in zip(words, (480, 960)):
        falling = ((word >> 134) & 1023)*16+((word >> 106) & 31)
        assert falling == 1+hold


def test_clock_aligned_hold_after_16_interval_ramp_uses_endpoint_not_reservation():
    pulse = PulseSequence(0, 1000)
    pulse.add_flat_ramp(10, 100, 100)
    pulse.add_flat_ramp(10, 1000, 0)
    for row in (1, 2):pulse.set_ramp_duration_ns(row, 16/4.8)
    seq = build_qick_sequence((pulse,), sweeps=(QickHoldDurationSweepSpec('set_1', .1, .2, 2),))
    p = seq.make_program(configuration(), awg_channels=(1,))
    assert p._sample_hold
    words = [e.word for e in execute(p).output_events if e.word >> 151 & 1]
    for word, hold in zip(words, (480, 960)):
        falling = ((word >> 134) & 1023)*16+((word >> 106) & 31)
        assert falling == 16+hold


@pytest.mark.parametrize('reverse', (False, True))
@pytest.mark.parametrize('compensate', (False, True))
def test_grouped_rf_and_hold_duration_advance_one_sample_through_100ns(reverse, compensate):
    from test_rf_fine_duration_registers import decode_bursts
    pulse = PulseSequence(0, 1000)
    pulse.add_flat_ramp(10, 100, 100)
    pulse.add_flat_ramp(10, 1000, 0)
    for row in (1, 2):pulse.set_ramp_duration_ns(row, 16/4.8)
    first,last=(480,1) if reverse else (1,480)
    seq = build_qick_sequence((pulse,), sweeps=(QickHoldDurationSweepSpec('set_1', (first+80)*DT, (last+80)*DT, 480),))
    seq.add_rf_duration_sweep('set_1',0,first*DT,last*DT,480,parameter_name='rf_duration',sequence_fabric_mhz=300.)
    seq.set_sweep_groups({seq.sweep_axis_key(a):'rf_hold' for a in seq.sweeps})
    if compensate:
        seq.set_bias_t_compensation(.1,mode='fixed_time',fixed_duration_cycles=6000)
        seq.set_rc_compensation(20.)
    rf=RfPulseConfig(0,'set_1',max(3,(first+15)//16),3000,freq_mhz=198.,delay_tproc_cycles=2,
                     fine_duration=True,duration_samples=first,duration_parameter='rf_duration')
    p=seq.make_program(configuration(),awg_channels=(1,),rf_pulses=(rf,),repetitions_per_sweep=2,compile_validation_mode='boundary')
    model=execute(p)
    ramps=[e for e in model.output_events if e.word>>151&1]
    bursts=decode_bursts(p,[e for e in model.output_events if (e.word>>152)&255==0])
    assert len(ramps)==len(bursts)==960
    values=range(first,last+(-1 if reverse else 1),-1 if reverse else 1)
    for (r,b),n in zip(zip(ramps,bursts),(n for n in values for _ in range(2))):
        falling=((r.word>>134)&1023)*16+((r.word>>106)&31)
        end=(r.word>>111)&((1<<23)-1)
        assert b['samples']==n
        assert (b['start']-r.cycle)*16==48
        assert falling==n+96
        assert falling-48-b['samples']==48
        assert end==n+113
    assert p.summary()['sample_hold_sweep']['duration_point_table_words']==0
    assert all(x['duration_table_words']==0 for x in p.summary()['rf_fine_duration'])


def test_cached_sample_program_replays_same_branch_state(tmp_path, monkeypatch):
    import qick_program_cache as cache
    import qick_qcodes_experiment as runtime
    monkeypatch.setenv(cache.CACHE_DIR_ENV, str(tmp_path/'program_cache'))
    cache.clear_memory_cache()
    seq = compile_hold(16321,16401,5).sequence
    kwargs = dict(awg_channels=(1,), repetitions_per_sweep=2, compile_validation_mode='boundary')
    first = runtime.build_qick_program(configuration(), seq, **kwargs)
    cache.clear_memory_cache()
    second = runtime.build_qick_program(configuration(), seq, **kwargs)
    assert second.compile_cache_info['source'] == 'disk'
    assert [(e.cycle,e.word) for e in execute(first).output_events] == [(e.cycle,e.word) for e in execute(second).output_events]


def test_gui_hold_existing_summary_json_and_export(monkeypatch, tmp_path):
    from PyQt5 import QtCore, QtWidgets
    import DCWaveform_Generator as gui
    from dc_waveform_core import generate_qick_program_code
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = gui.MainWindow()
    try:
        window._add_segment(10, 10, 100)
        window._add_segment(10, 1000, 0)
        def accept(dialog):
            dialog.start.setValue(.001)
            dialog.stop.setValue(.3)
            dialog.count.setValue(300)
            assert 'applied end 312.5 ns' in dialog.time_summary.text()
            dialog.accept()
            return dialog.result()
        monkeypatch.setattr(gui.HoldDurationSweepSettingsDialog, 'exec_', accept)
        window._configure_hold_duration_sweep(0, 1)
        table = window._experiment_panel.sweep_parameter_table
        assert table.item(0,6).text() == '1.04166667 ns (5 DAC sample(s))'
        assert table.item(0,7).text() == '312.5 ns'
        path = tmp_path/'sample_settings.json'
        window._save_settings_json(path)
        window._load_settings_json(path)
        seq = window._experiment_run_arguments(require_readout=False, require_run_config=False)['sequence']
        assert seq.sweep_axes[0].duration_sample_points[-1] == 1500
        source = generate_qick_program_code(window._pulse, sweeps=window._sweep_specs, awg_channels=(1,))
        namespace = {'__name__':'export_validation'}
        exec(source, namespace)
        exported = namespace['build_sequence']()
        assert exported.sweep_axes[0].duration_sample_points == seq.sweep_axes[0].duration_sample_points
        assert exported.make_program(configuration(), awg_channels=(1,))._sample_hold
    finally:
        window.close()
        window.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()


def test_two_physical_outputs_and_virtual_matrix_replay():
    from copy import deepcopy
    cfg = configuration()
    cfg['gens'].append(deepcopy(cfg['gens'][1]))
    cfg['gens'][2]['tproc_ch'] = 1
    pulses = []
    for voltage in (80, -120):
        pulse = PulseSequence(0, 1000)
        pulse.add_flat_ramp(10, 10, voltage)
        pulse.add_flat_ramp(10, 1000, 0)
        pulses.append(pulse)
    seq = build_qick_sequence(pulses, cross_capacitance=((1., .2),(-.3,1.)),
        sweeps=(QickHoldDurationSweepSpec('set_1', DT, 17001*DT, 851),))
    seq.add_amplitude_sweep('set_1', 'awg_0', .1, .2, 2)
    p = seq.make_program(cfg, awg_channels=(1,2), repetitions_per_sweep=2, compile_validation_mode='boundary')
    model = execute(p)
    for gen, port in ((1,0),(2,1)):
        expected = []
        for point in range(seq.sweep_point_count):
            words = [sum(int(w)<<(32*i) for i,w in enumerate(c.words)) for c in p.sample_hold_commands(point)
                     if c.kind == 'ramp' and c.gen_ch == gen]
            expected.extend(words*2)
        assert [e.word for e in model.output_events if e.tproc_ch == port and e.word >> 151 & 1] == expected


def test_zero_initial_voltage_does_not_hide_ramp_duration_dependence():
    seq = compile_hold(1, 5, 3).sequence
    seq.add_ramp_duration_sweep('ramp_0_to_1', 16*DT, 80*DT, 5)
    seq.add_amplitude_sweep('set_1', 'awg_0', 0., .5, 3)
    p = seq.make_program(configuration(), awg_channels=(1,), repetitions_per_sweep=2,
                         compile_validation_mode='boundary')
    expected = []
    for point in range(seq.sweep_point_count):
        expected.extend([sum(int(w)<<(32*i) for i,w in enumerate(c.words))
                         for c in p.sample_hold_commands(point) if c.kind == 'ramp']*2)
    assert [e.word for e in execute(p).output_events if e.word >> 151 & 1] == expected
    coordinates = build_execution_coordinates(p)
    duration = next(c for c in coordinates['segment_duration_fabric_cycles'] if c['segment_name'] == 'ramp_0_to_1')
    assert set(duration['coordinates']) == {n/16 for n in (16,32,48,64,80)}
    for command in coordinates['awg_commands']:
        assert len(command['columns']['command_words_u32']) == seq.sweep_point_count
