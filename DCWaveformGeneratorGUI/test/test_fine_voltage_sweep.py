"""Voltage register adds and v7 slope packing, without device access."""
from itertools import product
import numpy as np
import pytest
import qick.sim
from qick.awg_tuning import TProcV1BehaviorModel
from qick.ramp_v7 import edge_step, ramp_packet
from qick_fine_tune_sweep import FineTuneSequence
from qick_fine_ramp import is_extended, sweep_spec_at
from qick_dc_area import executed_commands, waveform_ledger
from test_awg_v7_fine import configuration

DT = 1 / 4800


@pytest.fixture(autouse=True)
def no_hardware(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        pytest.fail('Offline test attempted network access')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)


def sequence(two=False, reverse=False, start_sweep=False):
    seq = (FineTuneSequence(('x',)).add_set('start', (-.5,), 300)
           .add_ramp('edge', 40).add_set('end', (.5,), 300))
    spec = dict(rise_start=2, rise_end=16)
    if two:
        spec.update(fall_start=18, fall_end=35, turn=.25)
    seq.set_fine_ramp('edge', (spec,))
    seq.add_ramp_duration_sweep('edge', 14*DT, 17*DT, 4)
    seq.add_amplitude_sweep('end', 'x', .75 if reverse else -.75, -.75 if reverse else .75, 7)
    if start_sweep:
        seq.add_amplitude_sweep('start', 'x', .5, -.5, 3)
    return seq


def execute(seq, mode='boundary', reps=2):
    p = seq.make_program(configuration(), awg_channels=(1,), repetitions_per_sweep=reps,
                         compile_validation_mode=mode)
    p.compile()
    model = TProcV1BehaviorModel(strict=True)
    p.load_runtime_dmem_into_model(model)
    model.run(p, max_steps=10_000_000)
    assert model.ended and not model.timing_conflicts
    actual = [e.word for e in model.output_events if e.word >> 151 & 1]
    expected = []
    for point in range(seq.sweep_point_count):
        commands = [c for c in executed_commands(p, point) if is_extended(c)]
        words = [sum(int(w) << (32*i) for i, w in enumerate(c.words)) for c in commands]
        expected.extend(words * reps)
    assert actual == expected
    return p, actual


def signed(value, bits):
    return value - (1 << bits) if value & (1 << (bits-1)) else value


@pytest.mark.parametrize('mode', ('full', 'boundary'))
@pytest.mark.parametrize('two,reverse,start_sweep', product((False, True), repeat=3))
@pytest.mark.parametrize('voltage_outer', (False, True))
def test_independent_voltage_and_duration_packets(mode, two, reverse, start_sweep, voltage_outer):
    seq = sequence(two, reverse, start_sweep)
    if voltage_outer:
        seq.set_sweep_order([seq.sweep_axis_key(a) for a in seq.sweep_axes[1:] + seq.sweep_axes[:1]])
    p, words = execute(seq, mode)
    for point, word in enumerate(words[::2]):
        indices = seq.point_indices(point)
        by_name = {a.segment_name: i for a,i in zip(seq.sweep_axes, indices)}
        start = 16384 - by_name['start']*16384 if start_sweep else -16384
        end = (24576 - by_name['end']*8192) if reverse else (-24576 + by_name['end']*8192)
        spec = sweep_spec_at(seq, 'edge', 0, point)
        turn = 8192 if two else end
        assert signed(word & 65535, 16) == turn
        assert signed((word >> 16) & 65535, 16) == end
        # The integer coefficient approximation differs by at most a few Q18
        # units; the actual packet and area reference must agree exactly.
        up = signed((word >> 32) & ((1 << 35)-1), 35)
        down = signed((word >> 67) & ((1 << 35)-1), 35)
        assert abs(up - edge_step(start, turn, spec['rise_end']-spec['rise_start'])) <= 8
        if two:
            assert abs(down - edge_step(turn, end, 17)) <= 8
        assert waveform_ledger(p, point, 0)['scalar_samples'] > 0
    target_fields = [f for f in p._sweep_fields if f['key'][-1] in ('turn', 'end')]
    assert target_fields and all('duration_table_bases' not in f for f in target_fields)


@pytest.mark.parametrize('mode', ('full', 'boundary'))
def test_grouped_voltage_duration_is_exact_at_every_row(mode):
    seq = sequence(two=True)
    seq.add_amplitude_sweep('end', 'x', -.75, .75, 4)
    seq.add_ramp_duration_sweep('edge:fall', 17*DT, 14*DT, 4)
    seq.set_sweep_groups({seq.sweep_axis_key(a): 'all' for a in seq.sweep_axes})
    p, words = execute(seq, mode)
    for i, word in enumerate(words[::2]):
        spec = sweep_spec_at(seq, 'edge', 0, i)
        expected = ramp_packet(start=-16384, target=8192, end_target=-24576+i*16384,
            rise_start=2, rise_end=spec['rise_end'], fall_start=spec['fall_start'], fall_end=spec['fall_end'], tmux=1)
        assert word == expected['word']
    assert p.sequence.sweep_point_count == 4


@pytest.mark.parametrize('direction', (1, -1))
def test_signed_35_bit_carry_and_rewind(direction):
    seq = FineTuneSequence(('x',)).add_set('start', (-.999*direction,), 30).add_ramp('edge', 3).add_set('end', (.99,), 30)
    seq.set_fine_ramp('edge', (dict(rise_start=15, rise_end=16),))
    seq.add_amplitude_sweep('end', 'x', -.99*direction, .99*direction, 31)
    seq.add_hold_duration_sweep('start', 1., 2., 2)
    seq.set_sweep_order([seq.sweep_axis_key(a) for a in sorted(seq.sweep_axes, key=lambda a: a.axis_kind == 'amplitude')])
    p, words = execute(seq)
    assert words[:62] == words[62:]
    assert any(abs(signed((w >> 32) & ((1<<35)-1),35)) > (1<<31)-1 for w in words)


@pytest.mark.parametrize('compensated', (False, True))
def test_compensation(compensated):
    seq = sequence(two=True)
    if compensated:
        seq.set_bias_t_compensation(.1, mode='fixed_time', fixed_duration_cycles=1000)
        seq.set_rc_compensation(200.)
    p, words = execute(seq)
    assert len(words) == 56
    if compensated:
        assert p.hardware_dc_validation and p.rc_output_range_validation


def test_voltage_count_does_not_make_packet_table():
    counts = []
    for count in (10, 1000):
        seq = sequence(two=True)
        seq.add_amplitude_sweep('end', 'x', -.25, .25, count)
        p = seq.make_program(configuration(), awg_channels=(1,), compile_validation_mode='boundary')
        counts.append((len(p._runtime_dmem_words), len(p.prog_list)))
        assert all(not m.get('packed_word') for m in p._sweep_models.values())
    assert counts[0][0] == counts[1][0]
    assert counts[1][1] < 1000


def test_execution_export_and_disk_cache_keep_actual_falling_slope(tmp_path,monkeypatch):
    import qick_program_cache as cache
    from qick_qcodes_experiment import build_qick_program
    from qick_execution_coordinates import build_execution_coordinates
    monkeypatch.setenv(cache.CACHE_DIR_ENV,str(tmp_path/'cache'))
    cache.clear_memory_cache()
    try:
        kwargs=dict(awg_channels=(1,),compile_validation_mode='boundary',repetitions_per_sweep=2)
        original=build_qick_program(configuration(),sequence(two=True),**kwargs)
        cache.clear_memory_cache()
        restored=build_qick_program(configuration(),sequence(two=True),**kwargs)
        assert restored.compile_cache_info['source']=='disk'
        assert original.binprog==restored.binprog
        coordinates=build_execution_coordinates(restored,point_indices=[1,8,19,27])
        edge=next(c for c in coordinates['awg_commands'] if c['kind']=='ramp')
        values=edge['columns']['command_words_u32']
        assert len(values)==4
        for point,words in zip((1,8,19,27),values):
            cmd=next(c for c in executed_commands(original,point) if is_extended(c))
            assert words==list(cmd.words)
        assert len({tuple(v) for v in values})==4
    finally:
        cache.clear_memory_cache()


@pytest.mark.parametrize('order', ('voltage_inner', 'voltage_outer', 'grouped'))
@pytest.mark.parametrize('compensated', (False, True))
def test_fine_edges_voltage_and_long_hold_branch(order, compensated):
    seq = (FineTuneSequence(('x',)).add_set('zero',(0.,),300)
           .add_ramp('rise',4).add_set('hold',(.25,),30)
           .add_ramp('fall',4).add_set('end',(0.,),300))
    for name in ('rise','fall'):
        seq.set_fine_ramp(name,(dict(rise_start=0,rise_end=1),))
        seq.add_ramp_duration_sweep(name,14*DT,17*DT,4)
    seq.add_hold_duration_sweep('hold',16366*DT,16426*DT,4,sample_grid=True)
    seq.add_amplitude_sweep('hold','x',-.375,.375,4)
    keys = [seq.sweep_axis_key(a) for a in seq.sweep_axes]
    if order == 'grouped':
        seq.set_sweep_groups(dict.fromkeys(keys,'all'))
    elif order == 'voltage_outer':
        seq.set_sweep_order(keys[-1:] + keys[:-1])
    if compensated:
        seq.set_bias_t_compensation(.1, mode='fixed_time', fixed_duration_cycles=2000)
        seq.set_rc_compensation(200.)
    p, words = execute(seq)
    assert p._sample_hold
    assert any((w >> 106) & 31 == 31 for w in words)
    assert any((w >> 106) & 31 != 31 for w in words)
    if order != 'grouped':
        hold_axis = next(i for i,a in enumerate(seq.sweep_axes) if a.axis_kind == 'hold_duration')
        assert all(hold_axis not in f.get('duration_axis_indices',()) for f in p._sweep_fields)
    assert len(p._runtime_dmem_words) < 400


def test_two_outputs_with_unrelated_voltage():
    from copy import deepcopy
    cfg = configuration()
    cfg['gens'].append(deepcopy(cfg['gens'][1]))
    cfg['gens'][2].update(tproc_ch=1,tmux_ch=2)
    seq = (FineTuneSequence(('x','y')).add_set('zero',(0.,0.),300)
           .add_ramp('rise',40).add_set('hold',(.25,.25),300))
    seq.set_fine_ramp('rise',(dict(rise_start=15,rise_end=16),None))
    seq.add_amplitude_sweep('hold','y',-.5,.5,9)
    p = seq.make_program(cfg,awg_channels=(1,2),repetitions_per_sweep=2)
    m=TProcV1BehaviorModel(strict=True);p.load_runtime_dmem_into_model(m);m.run(p)
    fine=[e.word for e in m.output_events if e.word>>151&1]
    assert len(fine)==18 and len(set(fine))==1
    assert not m.timing_conflicts


@pytest.mark.parametrize('sign',(1,-1))
def test_both_full_scale_one_sample_slopes_at_falling_lane_15(sign):
    seq=(FineTuneSequence(('x',)).add_set('zero',(0.,),300)
         .add_ramp('edge',3).add_set('end',(0.,),300))
    seq.set_fine_ramp('edge',(dict(rise_start=0,rise_end=1,fall_start=15,fall_end=16,turn=.75*sign),))
    seq.add_amplitude_sweep('end','x',-.75,.75,7)
    _, words=execute(seq)
    for i, word in enumerate(words[::2]):
        expected=ramp_packet(start=0,target=24576*sign,end_target=-24576+i*8192,
                             rise_start=0,rise_end=1,fall_start=15,fall_end=16,tmux=1)
        assert word==expected['word']


def test_gui_grouped_voltage_fine_duration_roundtrip(monkeypatch,tmp_path):
    from PyQt5 import QtCore, QtWidgets
    import DCWaveform_Generator as gui
    from dc_waveform_core import QickSweepSpec, QickRampRateSweepSpec
    from test_sweep_grouping_gui import check_and_group, edit_cell
    from sweep_grouping import parameter_key
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w=gui.MainWindow()
    try:
        w._add_segment(10.,1000.,100.)
        w._pulse[0].set_ramp_duration_ns(1,1000*DT)
        w._sweep_specs=[QickRampRateSweepSpec('ramp_0_to_1',14*DT,17*DT,4),
                        QickSweepSpec('set_1','awg_0',-.375,.375,4)]
        w._notify_sweep_state_changed()
        check_and_group(w,w._sweep_specs)
        path=tmp_path/'fine_voltage.json'
        w._save_settings_json(path);w._load_settings_json(path)
        run=w._experiment_run_arguments(require_readout=False,require_run_config=False)
        assert run['sequence'].sweep_point_count==4
        p, words=execute(run['sequence'])
        assert len(set(words))==4
        edit_cell(w,parameter_key(w._sweep_specs[0]),4,7)
        assert [s.count for s in w._sweep_specs]==[7,7]
        w._remove_ramp_rate_sweep(0,1)
        seq=w._experiment_run_arguments(require_readout=False,require_run_config=False)['sequence']
        assert seq.fine_ramps and seq.sweep_point_count==7
        execute(seq)
    finally:
        w.close();w.deleteLater();app.sendPostedEvents(None,QtCore.QEvent.DeferredDelete);app.processEvents()
