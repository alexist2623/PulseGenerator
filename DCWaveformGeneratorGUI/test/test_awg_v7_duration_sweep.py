"""Actual duration-loop payloads, GUI inputs, preview and endpoint constraints."""
import json
import socket
import numpy as np
import pytest
from qick.sim import QickSim
from qick.awg_tuning import TProcV1BehaviorModel
from dc_waveform_core import PulseSequence, QickRampRateSweepSpec, build_qick_sequence
from qick_fine_ramp import is_extended, sweep_spec_at
from qick_dc_area import executed_commands, waveform_ledger
from test_awg_v7_fine import configuration
from test_awg_v6_timing import configuration as old_configuration

DT = 1 / 4800  # microseconds


@pytest.mark.parametrize('reverse', (False, True))
def test_fine_linear_grid_preserves_endpoints(reverse):
    from qick_fine_tune_sweep import RampDurationSweep
    from time_display import time_sweep_display
    for first, last, count, endpoints in ((.0002,.1,100,(1,480)),(.001,.3,300,(5,1440))):
        if reverse: first,last=last,first;endpoints=endpoints[::-1]
        axis=RampDurationSweep('ramp',first,last,count,300,16)
        points=axis.duration_sample_points
        assert (points[0],points[-1])==endpoints
        assert len(set(points))==count
        ideal=np.linspace(endpoints[0],endpoints[1],count)
        assert np.max(abs(np.asarray(points)-ideal))<=.5+1e-12
        assert 'Invalid' not in time_sweep_display(first,last,count,fabric_mhz=300,samples_per_clock=16)['summary']


def test_factored_packet_tables_pack_carries_and_reset_nested_axes():
    seq=sequence()
    seq.add_ramp_duration_sweep('ramp_0_to_1',DT,19*DT,3)
    seq.add_ramp_duration_sweep('ramp_0_to_1:hold',31*DT,2*DT,5)
    # Cross the 17-bit duration field boundary in word 3, without moving
    # the second-edge start beyond its independently limited range.
    seq.add_ramp_duration_sweep('ramp_0_to_1:fall',131070*DT,131100*DT,4)
    p=seq.make_program(configuration(),awg_channels=(1,),repetitions_per_sweep=2,
                       compile_validation_mode='boundary')
    assert len(p._runtime_dmem_words)<100
    model=TProcV1BehaviorModel(strict=True);p.load_runtime_dmem_into_model(model);model.run(p)
    expected=[]
    for point in range(60):
        c=next(c for c in executed_commands(p,point) if is_extended(c))
        word=sum(int(w)<<(32*i) for i,w in enumerate(c.words))
        expected.extend([word]*2)
    assert [e.word for e in model.output_events if e.word>>151&1]==expected
    assert not model.timing_conflicts


@pytest.mark.parametrize('component', ('grouped', 'hold', 'independent'))
@pytest.mark.parametrize('compensated', (False, True))
@pytest.mark.parametrize('mode', ('full', 'boundary'))
def test_packet_component_loops_and_compensation(component, compensated, mode):
    seq = sequence()
    if component == 'hold':
        seq.clear_ramp_duration_sweep()
        seq.add_ramp_duration_sweep('ramp_0_to_1:hold', DT, 30*DT, 30)
    else:
        seq.add_ramp_duration_sweep('ramp_0_to_1:fall', DT, (30 if component == 'grouped' else 2)*DT,
                                    30 if component == 'grouped' else 2)
        if component == 'grouped':
            seq.set_sweep_groups({seq.sweep_axis_key(a): 'edges' for a in seq.sweeps})
    if compensated:
        seq.set_bias_t_compensation(.1, mode='fixed_time', fixed_duration_cycles=1 if component == 'hold' else 30)
        seq.set_rc_compensation(200.)
    p = seq.make_program(configuration(), awg_channels=(1,), repetitions_per_sweep=2,
                         compile_validation_mode=mode)
    model = TProcV1BehaviorModel(strict=True)
    p.load_runtime_dmem_into_model(model); model.run(p)
    emitted = [e.word for e in model.output_events if (e.word >> 151) & 1]
    expected = []
    for point in range(seq.sweep_point_count):
        s = sweep_spec_at(seq, 'ramp_0_to_1', 0, point)
        indices = seq.point_indices(point)
        rise = 1 if component == 'hold' else indices[0]+1
        hold = indices[0]+1 if component == 'hold' else 480
        fall = 1 if component == 'hold' else indices[1]+1
        assert (s['rise_end'], s['fall_start'], s['fall_end']) == (rise, rise+hold, rise+hold+fall)
        c = next(c for c in executed_commands(p, point) if is_extended(c))
        word = sum(int(w) << (32*i) for i, w in enumerate(c.words))
        assert (word >> 111) & ((1<<23)-1) == rise+hold+fall+1
        expected.extend([word]*2)
    assert emitted == expected
    assert not model.timing_conflicts


def test_component_gui_group_edit_reload_remove(monkeypatch, tmp_path):
    from PyQt5 import QtWidgets
    import DCWaveform_Generator as gui
    from test_sweep_grouping_gui import check_and_group, edit_cell
    from sweep_grouping import parameter_key
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = gui.MainWindow()
    try:
        w._add_segment(120, 1000, 0)
        w._pulse[0].set_fine_ramp(1, dict(rise_start=0, rise_end=1, fall_start=481, fall_end=482, turn=400))
        def accept(d):
            d.start.setValue(DT); d.stop.setValue(30*DT); d.count.setValue(30)
            d.accept(); return d.result()
        monkeypatch.setattr(gui.RampRateSweepSettingsDialog, 'exec_', accept)
        for component in ('rise', 'fall'):
            w._configure_ramp_rate_sweep(0, 1, component)
        check_and_group(w, w._sweep_specs)
        run = w._experiment_run_arguments(require_readout=False, require_run_config=False)
        assert run['sequence'].sweep_point_count == 30
        run['sequence'].make_program(configuration(), awg_channels=(1,))
        path = tmp_path/'grouped.json'; w._save_settings_json(path); w._load_settings_json(path)
        assert w._experiment_run_arguments(require_readout=False, require_run_config=False)['sequence'].sweep_point_count == 30
        edit_cell(w, parameter_key(w._sweep_specs[1]), 4, 20)
        assert [s.count for s in w._sweep_specs] == [20, 20]
        old_id = w._sweep_grouping.member_id(w._sweep_specs[1])
        # Every component keeps the destination SET identity across insert/delete.
        assert gui._remap_ramp_segment_name('ramp_0_to_1:fall', 'insert', 1) == 'ramp_1_to_2:fall'
        w._remove_ramp_rate_sweep(0, 1, 'fall')
        assert len(w._sweep_specs) == 1
        w._configure_ramp_rate_sweep(0, 1, 'fall')
        assert w._sweep_grouping.member_id(w._sweep_specs[1]) != old_id
        edit_cell(w, parameter_key(w._sweep_specs[1]), 4, 15)
        assert [s.count for s in w._sweep_specs] == [20, 15]
        w._multi_ctrl._ctrl_pannels[0].fine_component_sweep_remove_requested.emit(0, 1, 'fall')
        w._remove_ramp_rate_sweep(0, 1)
        w._configure_ramp_rate_sweep(0, 1, 'hold')
        run = w._experiment_run_arguments(require_readout=False, require_run_config=False)
        assert run['sequence'].sweep_point_count == 30
        assert sweep_spec_at(run['sequence'], 'ramp_0_to_1', 0, 29)['fall_start'] == 31
        run['sequence'].make_program(configuration(), awg_channels=(1,))
        w._save_settings_json(path); w._load_settings_json(path)
        assert w._sweep_specs[0].segment_name.endswith(':hold')
        assert w._multi_ctrl._ctrl_pannels[0]._edit_segment_structure('delete', 1)
        assert not w._sweep_specs
    finally:
        w.close(); w.deleteLater(); app.processEvents()


def sequence(start=1, stop=30, count=30):
    pulse = PulseSequence(0, 1000)
    pulse.add_flat_ramp(110, 1000, 0)
    pulse.set_fine_ramp(1, dict(rise_start=0, rise_end=1, fall_start=481, fall_end=482, turn=400))
    return build_qick_sequence((pulse,), sweeps=(QickRampRateSweepSpec('ramp_0_to_1', start*DT, stop*DT, count),))


@pytest.mark.parametrize('reverse', (False, True))
@pytest.mark.parametrize('mode', ('full', 'boundary'))
def test_hardware_loop_changes_all_packet_words_preserves_hold_and_rewinds(reverse, mode):
    seq = sequence(30, 1) if reverse else sequence()
    # Two outer iterations prove the duration table rewinds at a loop boundary.
    seq.add_hold_duration_sweep('set_0', 1., 2., 2)
    seq.sweeps.reverse()
    p = seq.make_program(configuration(), awg_channels=(1,), repetitions_per_sweep=2,
                         compile_validation_mode=mode)
    p.compile()
    model = TProcV1BehaviorModel(strict=True)
    p.load_runtime_dmem_into_model(model)
    model.run(p)
    emitted = [e.word for e in model.output_events if (e.word >> 151) & 1]
    expected = []
    lengths = []
    for point in range(60):
        spec = sweep_spec_at(seq, 'ramp_0_to_1', 0, point)
        assert spec['fall_start'] - spec['rise_end'] == 480
        assert spec['fall_end'] - spec['fall_start'] == 1
        length = spec['rise_end'] - spec['rise_start']
        lengths.append(length)
        command = next(c for c in executed_commands(p, point) if is_extended(c))
        word = sum(int(w) << (32*i) for i, w in enumerate(command.words))
        assert (word >> 111) & ((1<<23)-1) == length + 482
        assert command.step == ((16384 << 18) + length-1)//length
        expected.extend([word]*2)
        t, traces, _ = seq.waveform_vertices(point)
        assert max(traces['awg_0']) == .5
        assert waveform_ledger(p, point, 0)['area'] > 0
    assert emitted == expected
    assert lengths[:30] == list(range(30,0,-1) if reverse else range(1,31))
    assert lengths[30:] == lengths[:30]
    assert not model.timing_conflicts
    assert len(p.prog_list) < 1000


def test_simple_fractional_sweep_and_old_firmware_rejection():
    pulse = PulseSequence(0, 1000)
    pulse.add_flat_ramp(10, 100, 100)
    spec = QickRampRateSweepSpec('ramp_0_to_1', DT, 33*DT, 33)
    seq = build_qick_sequence((pulse,), sweeps=(spec,))
    p = seq.make_program(configuration(), awg_channels=(1,))
    assert seq.sweeps[0].samples_per_clock == 16
    for point in range(33):
        cmd = next(c for c in executed_commands(p, point) if is_extended(c))
        assert cmd.duration_samples == point+2
    with pytest.raises(ValueError, match='v7'):
        seq.make_program(old_configuration(), awg_channels=(1,))
    seq.clear_ramp_duration_sweep()
    assert not seq.fine_ramps
    seq.add_ramp_duration_sweep('ramp_0_to_1', .01, .02, 4)
    assert seq.sweeps[0].samples_per_clock == 1
    seq.make_program(old_configuration(), awg_channels=(1,))


def test_limits_checked_before_execution_and_nondefault_clock():
    seq = sequence()
    seq.add_ramp_duration_sweep('ramp_0_to_1', 1/4000, 30/4000, 30, sequence_fabric_mhz=250)
    assert seq.sweeps[0].points[-1] == 30/4000
    with pytest.raises(ValueError, match='time unit'):
        seq.add_ramp_duration_sweep('ramp_0_to_1', DT, 2*DT, 30)
    with pytest.raises(ValueError, match='fall_start'):
        sequence(16000, 16001, 2).make_program(configuration(), awg_channels=(1,))


def test_grouped_two_edges_negative_voltage_and_runtime_metadata():
    from qick_execution_coordinates import build_execution_coordinates
    pulse = PulseSequence(0, 1000)
    pulse.add_flat_ramp(10, 100, -400)
    pulse.set_ramp_duration_ns(1, 1000*DT)
    pulse.add_flat_ramp(10, 1000, 0)
    pulse.set_ramp_duration_ns(2, 1000*DT)
    specs = (QickRampRateSweepSpec('ramp_0_to_1', DT, 33*DT, 33),
             QickRampRateSweepSpec('ramp_1_to_2', 33*DT, DT, 33))
    seq = build_qick_sequence((pulse,), sweeps=specs)
    seq.set_sweep_groups({seq.sweep_axis_key(a): 'edges' for a in seq.sweeps})
    p = seq.make_program(configuration(), awg_channels=(1,), compile_validation_mode='boundary')
    model = TProcV1BehaviorModel(strict=True); p.load_runtime_dmem_into_model(model); model.run(p)
    expected = []
    for point in range(33):
        for cmd in executed_commands(p, point):
            if is_extended(cmd):
                expected.append(sum(int(w)<<(32*i) for i,w in enumerate(cmd.words)))
    assert [e.word for e in model.output_events if (e.word>>151)&1] == expected
    assert len(expected) == 66
    assert not model.timing_conflicts
    metadata = build_execution_coordinates(p)
    assert 'edge_positions_by_point' in json.dumps(metadata)


def test_gui_sweep_dialog_and_save_reload_use_sample_grid(monkeypatch, tmp_path):
    from PyQt5 import QtCore, QtTest, QtWidgets
    import DCWaveform_Generator as gui
    def forbidden(*args, **kwargs):
        pytest.fail('Unexpected device access or GUI warning: '+str(args))
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', forbidden)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = gui.MainWindow()
    try:
        w._add_segment(110, 1000, 0)
        w._pulse[0].set_fine_ramp(1, dict(rise_start=0, rise_end=1, fall_start=481, fall_end=482, turn=400))
        def accept(d):
            for widget, text in ((d.start, '0.000208333333'), (d.stop, '0.00625')):
                widget.lineEdit().selectAll()
                QtTest.QTest.keyClicks(widget.lineEdit(), text)
                widget.interpretText()
                widget.commit_clock_value()
            d.count.setValue(30)
            assert d.start.value() == pytest.approx(DT)
            assert '1 DAC sample(s)' in d.time_summary.text()
            d.accept()
            return d.result()
        monkeypatch.setattr(gui.RampRateSweepSettingsDialog, 'exec_', accept)
        w._configure_ramp_rate_sweep(0, 1)
        args = w._experiment_run_arguments(require_readout=False, require_run_config=False)
        assert args['sequence'].sweep_point_count == 30
        assert args['sequence'].sweeps[0].points == pytest.approx(np.arange(1,31)*DT)
        p = args['sequence'].make_program(configuration(), awg_channels=(1,))
        assert len(p.compiled_points) == 30
        path = tmp_path/'settings.json'
        w._save_settings_json(path)
        assert 'ramp_duration' in path.read_text()
        # Saved values use us and survive round trip without decimal-grid drift.
        spec = w._sweep_specs[0]
        restored = QickRampRateSweepSpec(**json.loads(json.dumps(dict(
            segment_name=spec.segment_name, start=spec.start, stop=spec.stop, count=spec.count))))
        assert restored == spec
        w._load_settings_json(path)
        restored_run = w._experiment_run_arguments(require_readout=False, require_run_config=False)
        assert restored_run['sequence'].sweeps[0].points == pytest.approx(np.arange(1,31)*DT)
        w._remove_ramp_rate_sweep(0, 1)
        assert not w._sweep_specs
        assert w._pulse[0].fine_ramps[1]['rise_end'] == 1
    finally:
        w.close(); w.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete); app.processEvents()
