"""GUI model -> v7 wire packet -> queued execution, without a board or GUI window."""
import json
from dataclasses import replace
import numpy as np
import pytest
from qick.sim import QickSim  # desktop driver bootstrap
from qick.awg_tuning import AxisAwgTuningV7, TProcV1BehaviorModel
from qick.output_sum import apply_output_sum_config
from qick.ramp_v7 import ramp_packet
from dc_waveform_core import PulseSequence, build_qick_sequence, transform_virtual_waveforms
from qick_fine_tune_sweep import compile_sequence, RfPulseConfig
from qick_fine_ramp import scalar_runs, is_extended
from qick_dc_area import waveform_ledger
from test_awg_v6_timing import configuration as v6_configuration


def configuration():
    cfg = v6_configuration()
    driver = AxisAwgTuningV7(dict(type='QICK:QICK:axis_awg_tuning_v7:1.0', fullpath='awg',
        parameters=dict(N_PTS='16', EXTRA_Y_PIPE_STAGES='3', DC_AREA_BITS='48',
                        DC_AREA_VERSION='1', RC_PRECOMP_VERSION='1')))
    # Replace the old IP before discovering its new output-sum route.
    apply_output_sum_config(cfg['gens'][1], None)
    # Connection fields are populated by hardware discovery, not __init__.
    cfg['gens'][1].update({k: v for k, v in driver.cfg.items() if v is not None})
    for ch, role in ((0, 'rf'), (1, 'awg')):
        apply_output_sum_config(cfg['gens'][ch], dict(output_sum_version=7,
            output_sum_role=role, post_output_latency_cycles=2, output_sum_awg_delay_cycles=2))
    assert cfg['gens'][1]['output_sum_native_latency_cycles'] == 27
    assert cfg['gens'][1]['dc_control_latency_cycles'] == 11
    assert cfg['gens'][1]['output_latency_cycles'] == cfg['gens'][0]['output_latency_cycles'] == 31
    return cfg


def pulse(spec=None, duration=10, end=20):
    p = PulseSequence(0, 10)
    p.add_flat_ramp(duration, 10, end)
    p.set_fine_ramp(1, spec or dict(rise_start=2, rise_end=5, fall_start=14, fall_end=20, turn=100))
    return p


@pytest.mark.parametrize('lane', range(16))
@pytest.mark.parametrize('fall_lane', range(16))
def test_all_lanes_and_cross_word_packet(lane, fall_lane):
    spec = dict(rise_start=lane, rise_end=lane+2, fall_start=32+fall_lane,
                fall_end=51+fall_lane, turn=-80)
    seq = build_qick_sequence((pulse(spec, 20),))
    cfg = configuration()
    _, points = compile_sequence(seq, cfg, (1,))
    cmd = points[0].segment_commands[1][0]
    expected = ramp_packet(start=0, target=-3276, rise_start=lane, rise_end=lane+2,
                           fall_start=32+fall_lane, fall_end=51+fall_lane,
                           end_target=820, tmux=1)
    assert tuple(expected['words']) == cmd.words
    assert cmd.duration_samples == 52+fall_lane


def test_save_load_copy_insert_delete_and_preview():
    p = pulse()
    q = PulseSequence.from_dict(json.loads(json.dumps(p.to_dict())))
    assert q.fine_ramps == p.fine_ramps
    assert q.copy().fine_ramps == p.fine_ramps
    q.insert_flat_ramp(2)
    assert 2 in q.fine_ramps and 1 not in q.fine_ramps
    q.delete_flat_ramp(2)
    assert q.fine_ramps == p.fine_ramps
    t, v = q.waveform_arrays()
    assert v.max() == 100 and v[-1] == 20
    assert t[np.where(v == 100)[0][0]] == pytest.approx(10+5/4.8)
    times, virtual, physical = transform_virtual_waveforms((q,))
    assert physical.max() == virtual.max() == 100
    seq = build_qick_sequence((q,))
    st, curves, _ = seq.waveform_vertices()
    assert np.interp(times, st/0.3, curves['awg_0']*800) == pytest.approx(virtual[0])
    assert not q.edit_ramp(2, 1/0.3)  # cannot truncate the second edge


@pytest.mark.parametrize('dc,rc', [(False, False), (True, False), (False, True), (True, True)])
def test_compiler_preserves_packet_and_repetitions(dc, rc):
    seq = build_qick_sequence((pulse(),))
    seq.set_bias_t_compensation(.1, enabled=dc, mode='fixed_time', fixed_duration_cycles=300)
    seq.set_rc_compensation(200, enabled=rc)
    p = seq.make_program(configuration(), awg_channels=(1,), repetitions_per_sweep=3)
    p.compile()
    cmd = p.compiled_points[0].segment_commands[1][0]
    assert is_extended(cmd)
    assert p.timing['command_times'][(2, 0)] - p.timing['command_times'][(1, 0)] == 3
    model = TProcV1BehaviorModel(strict=True)
    p.load_runtime_dmem_into_model(model); model.run(p)
    word = sum(int(w) << (32*i) for i, w in enumerate(cmd.words))
    assert sum(e.word == word for e in model.output_events) == 3
    assert not model.timing_conflicts


@pytest.mark.parametrize('spec', [dict(rise_start=15, rise_end=16),
    dict(rise_start=2, rise_end=5, fall_start=14, fall_end=20, turn=100),
    dict(rise_start=0, rise_end=15, fall_start=15, fall_end=31, turn=-100)])
def test_exact_area_matches_independent_scalar_reference(spec):
    seq = build_qick_sequence((pulse(spec),))
    p = seq.make_program(configuration(), awg_channels=(1,))
    cmd = p.compiled_points[0].segment_commands[1][0]
    # Independently evaluate the endpoint-clamped Q18 scalar construction.
    start = 0; turn = round(spec.get('turn', 20)/800*8192)*4
    end = cmd.target_code
    turn = max(-32768, min(32764, turn))
    r0, r1 = spec['rise_start'], spec['rise_end']
    def edge(a, b, k, length):
        step = (1 if b >= a else -1)*((abs((b-a)<<18)+length-1)//length) if length else 0
        fixed = (a<<18)+step*k
        fixed = max(b<<18, fixed) if b<a else min(b<<18, fixed)
        return fixed//(1<<20)
    values = []
    for n in range(((cmd.duration_samples+15)//16)*16):
        if n >= cmd.duration_samples-1:
            value = end//4
        elif 'fall_start' in spec and n >= spec['fall_start']:
            value = edge(turn,end,n-spec['fall_start'],spec['fall_end']-spec['fall_start'])
        elif n < r0:
            value = 0
        else:
            value = edge(start,turn,n-r0,r1-r0)
        values.append(value)
    from qick.dc_area import clipped_floor_sum
    actual = sum(clipped_floor_sum(n,1<<20,a,b,lo,hi) for n,a,b,lo,hi in scalar_runs(cmd,0))
    assert actual == sum(values)
    ledger = waveform_ledger(p,0,0)
    assert ledger['area'] == sum(values) + end//4*(48-len(values)+48)


def test_capability_and_unsupported_sweeps_fail_before_execution():
    seq = build_qick_sequence((pulse(),))
    with pytest.raises(ValueError, match='v7'):
        compile_sequence(seq, v6_configuration(), (1,))
    seq.add_amplitude_sweep('set_1', 'awg_0', .01, .02, 2)
    with pytest.raises(ValueError, match='fixed edge'):
        compile_sequence(seq, configuration(), (1,))


def test_hold_sweep_and_rf_keep_common_latency():
    seq = build_qick_sequence((pulse(),))
    seq.add_hold_duration_sweep('set_1', 1., 2., 3)
    p = seq.make_program(configuration(), awg_channels=(1,),
                        rf_pulse=RfPulseConfig(0,'set_1',30,1000,delay_tproc_cycles=1,freq_mhz=198))
    p.compile()
    assert p.awg_output_latency_tproc == 31
    assert all(is_extended(point.segment_commands[1][0]) for point in p.compiled_points)


def test_dialog_sample_ns_roundtrip_and_cancel():
    from PyQt5 import QtWidgets
    from qick_fine_ramp_widgets import FineRampDialog
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    p = pulse(); before = p.to_dict()
    dialog = FineRampDialog(p, 1, 300)
    dialog.times['fall_end'].setValue(5.0)
    assert dialog.samples['fall_end'].value() == 24
    dialog.times['fall_end'].setValue(5.001)
    assert dialog.times['fall_end'].value() == pytest.approx(5.)
    dialog.samples['rise_start'].setValue(3)
    assert dialog.times['rise_start'].value() == pytest.approx(.625)
    dialog.reject()
    assert p.to_dict() == before
    dialog.deleteLater(); app.processEvents()


def test_generated_python_preserves_fine_positions():
    from dc_waveform_core import generate_qick_program_code
    source = generate_qick_program_code((pulse(),), awg_channels=(1,))
    namespace = {'__name__': 'offline_export_test'}
    exec(compile(source, 'generated.py', 'exec'), namespace)
    original = build_qick_sequence((pulse(),))
    restored = namespace['build_sequence']()
    assert restored.fine_ramps == original.fine_ramps
    assert compile_sequence(restored, configuration(), (1,)) == compile_sequence(original, configuration(), (1,))


def test_shared_show_run_cache_preserves_packet_and_invalidates_edits(tmp_path, monkeypatch):
    import qick_program_cache as cache
    from qick_qcodes_experiment import build_qick_program
    from qick_execution_coordinates import build_execution_coordinates
    monkeypatch.setenv(cache.CACHE_DIR_ENV, str(tmp_path/'cache'))
    monkeypatch.setenv(cache.CACHE_ENABLED_ENV, '1')
    cache.clear_memory_cache()
    def build(wave):
        return build_qick_program(configuration(), build_qick_sequence((wave,)), awg_channels=(1,), repetitions_per_sweep=1)
    a, b = build(pulse()), build(pulse())
    assert a.compile_cache_info['status'] == 'miss'
    assert b.compile_cache_info['status'] == 'hit'
    assert a.asm() == b.asm()
    assert is_extended(b.compiled_points[0].segment_commands[1][0])
    export = build_execution_coordinates(b)
    assert any(row.get('extended_ramp_version') == 1 for row in export['awg_commands'])
    changed = pulse(); changed.fine_ramps[1]['fall_end'] = 21
    c = build(changed)
    assert c.compile_cache_info['status'] == 'miss'
    assert c.asm() != b.asm()
    cache.clear_memory_cache()


def test_full_scale_single_interval_35_bit_step_reaches_wire():
    p = PulseSequence(-800, 10)
    p.add_flat_ramp(10/3, 10, 799)
    p.set_fine_ramp(1, dict(rise_start=14, rise_end=15))
    seq = build_qick_sequence((p,))
    program = seq.make_program(configuration(), awg_channels=(1,))
    cmd = program.compiled_points[0].segment_commands[1][0]
    assert cmd.step > (1<<31)-1 and is_extended(cmd)
    model = TProcV1BehaviorModel(strict=True)
    program.load_runtime_dmem_into_model(model); model.run(program)
    word = sum(int(w) << (32*i) for i, w in enumerate(cmd.words))
    assert any(e.word == word for e in model.output_events)


def test_actual_control_panel_accepts_editor_and_updates_run_settings(tmp_path, monkeypatch):
    import socket
    from PyQt5 import QtCore, QtWidgets
    import DCWaveform_Generator as gui
    from qick_fine_ramp_widgets import FineRampDialog
    def forbidden(*args, **kwargs):
        pytest.fail('No network access is allowed in the editor test')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = gui.MainWindow()
    try:
        window._add_segment(10., 100., 20.)
        panel = window.findChildren(gui.ControlPanel)[0]
        def accept(dialog):
            dialog.enabled.setChecked(True)
            dialog.second.setChecked(True)
            for key, value in dict(rise_start=2, rise_end=5, fall_start=14, fall_end=20).items():
                dialog.samples[key].setValue(value)
            dialog.turn.setValue(200.)
            dialog.accept()
            return dialog.result()
        monkeypatch.setattr(FineRampDialog, 'exec_', accept)
        panel._edit_fine_ramp(1)
        assert panel._pulse.fine_ramps[1]['turn'] == 200.
        runtime = window._experiment_run_arguments(require_readout=False, require_run_config=False)['sequence']
        assert runtime.fine_ramps['ramp_0_to_1'][0]['fall_end'] == 20
        path = tmp_path/'saved.json'
        window._save_settings_json(path)
        saved = json.loads(path.read_text())
        assert 'fine_ramps' in json.dumps(saved)
        assert not window.isVisible()
    finally:
        window.close()
        window.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()
