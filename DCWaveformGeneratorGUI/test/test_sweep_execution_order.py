"""Execution order is tested against emitted tProcessor events, not table labels."""
from itertools import permutations, product
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5 import QtCore, QtTest
from qick.awg_tuning import TProcV1BehaviorModel

from dc_waveform_core import QickSweepSpec, QickRfSoftwareSweepAxisSpec
from qick_fine_tune_sweep import FineTuneSequence
from sweep_grouping import SweepGroupingModel, parameter_key
from test_grouped_sweep_compiler import config, grouped_sequence
from test_qick_fine_tune_sweep import _expected_words
from test_square_awg_exclusion import window
from test_sweep_grouping_gui import row_for


def run_model(program):
    model = TProcV1BehaviorModel(strict=True)
    program.load_runtime_dmem_into_model(model)
    model.run(program, max_steps=3_000_000)
    assert not model.timing_conflicts
    return model


@pytest.mark.parametrize('order', list(permutations(range(3))))
def test_mixed_time_voltage_order_changes_actual_hardware_loops(order):
    seq = FineTuneSequence(('a', 'b'))
    seq.add_set('zero', (0., 0.), 300).add_ramp('rise', 3)
    seq.add_set('hold', (.1, .05), 3000).add_ramp('fall', 3)
    seq.add_set('reset', (0., 0.), 300)
    seq.add_ramp_duration_sweep('rise', .01, .02, 2)
    seq.add_hold_duration_sweep('hold', 1., 3., 3)
    seq.add_amplitude_sweep('hold', 'a', .1, .2, 2)
    original = seq.sweep_axes
    keys = [seq.sweep_axis_key(original[i]) for i in order]
    seq.set_sweep_order(keys)
    assert [seq.sweep_axis_key(a) for a in seq.sweep_axes] == keys
    expected = list(product(*(range(original[i].count) for i in order)))
    assert [seq.point_indices(i) for i in range(seq.sweep_point_count)] == expected
    traces = []
    for mode in ('full', 'boundary'):
        prog = seq.make_program(config(), awg_channels=(0, 1),
                                repetitions_per_sweep=2, compile_validation_mode=mode)
        model = run_model(prog)
        traces.append([(event.cycle, event.tproc_ch, event.word) for event in model.output_events])
        if mode == 'full':
            assert [event.word for event in model.output_events] == _expected_words(prog)
        # The innermost requested coordinate advances after both repetitions.
        for point, indices in enumerate(expected):
            ramp_index = indices[order.index(0)]
            for shot in range(2):
                rise = model.output_events[(point * 2 + shot) * 10 + 2]
                assert (rise.word >> 64) & 0xffffffff == (3 + 3 * ramp_index) * 16
    assert traces[0] == traces[1]


@pytest.mark.parametrize('dc_mode', ['fixed_voltage', 'fixed_time'])
@pytest.mark.parametrize('rc', [False, True])
def test_grouped_voltage_outer_duration_inner_with_compensation(dc_mode, rc):
    seq = grouped_sequence(3)
    seq.set_cross_capacitance(((1., .2), (-.15, 1.)))
    seq.set_bias_t_compensation(.1, mode=dc_mode,
        fixed_duration_cycles=3000 if dc_mode == 'fixed_time' else None)
    keys = [seq.sweep_axis_key(axis) for axis in seq.sweep_axes]
    seq.set_sweep_order(keys[2:] + keys[:2])
    assert seq.point_indices(1) == (0, 0, 1, 1)
    cfg = config()
    if rc:
        from test_rc_precompensation import upgraded
        cfg = upgraded(cfg)
        seq.set_rc_compensation(30.)
    traces = []
    for mode in ('full', 'boundary'):
        prog = seq.make_program(cfg, awg_channels=(0, 1),
            repetitions_per_sweep=2, compile_validation_mode=mode)
        traces.append([(e.cycle, e.tproc_ch, e.word) for e in run_model(prog).output_events])
    assert traces[0] == traces[1]


def voltage_fixture(window):
    window._add_segment(1000., 10000., 100.)
    window._add_segment(1000., 10000., 0.)
    window._add_port()
    specs = [QickSweepSpec('set_1', 'awg_0', .1, .2, 2),
             QickSweepSpec('set_1', 'awg_1', .2, .4, 3)]
    window._sweep_specs = specs
    window._notify_sweep_state_changed()
    return specs


def test_buttons_change_run_order_and_survive_settings_roundtrip(window, tmp_path):
    specs = voltage_fixture(window)
    panel = window._experiment_panel
    table = panel.sweep_parameter_table
    table.selectRow(row_for(panel, parameter_key(specs[1])))
    assert panel.sweep_move_up_button.isEnabled()
    assert not panel.sweep_move_down_button.isEnabled()
    QtTest.QTest.mouseClick(panel.sweep_move_up_button, QtCore.Qt.LeftButton)
    assert panel._map_sweep_specs == tuple(reversed(specs))
    assert not table.isSortingEnabled()
    assert not panel.sweep_move_up_button.isEnabled()
    assert panel.sweep_move_down_button.isEnabled()
    runtime = window._experiment_run_arguments(require_readout=False, require_run_config=False)
    seq = runtime['sequence']
    assert [axis.output_name for axis in seq.sweep_axes] == ['awg_1', 'awg_0']
    assert seq.sweep_shape == (3, 2)
    assert seq.point_indices(1) == (0, 1)
    saved = tmp_path / 'reordered.json'
    window._save_settings_json(saved)
    QtTest.QTest.mouseClick(panel.sweep_move_down_button, QtCore.Qt.LeftButton)
    window._load_settings_json(saved)
    restored = window._experiment_run_arguments(require_readout=False, require_run_config=False)
    assert [axis.output_name for axis in restored['sequence'].sweep_axes] == ['awg_1', 'awg_0']
    assert restored['sequence'].sweep_shape == (3, 2)
    panel.set_running(True, 'Offline test')
    panel.sweep_parameter_table.selectRow(1)
    assert not panel.sweep_move_up_button.isEnabled()
    assert not panel.sweep_move_down_button.isEnabled()
    order = window._compiler_sweep_order()
    panel._move_selected_sweep_parameter(-1)
    assert window._compiler_sweep_order() == order
    panel.set_running(False, 'Finished')
    assert panel.sweep_move_up_button.isEnabled()


def test_group_moves_as_a_unit_and_deleted_identity_is_not_reused():
    specs = [QickSweepSpec('hold', f'a{i}', 0., .1, 3) for i in range(3)]
    model = SweepGroupingModel()
    model.synchronize(specs)
    ids = [model.member_id(s) for s in specs]
    model.create_group(ids[:2], specs)
    assert model.move(ids[1], 1, specs)
    assert model.ordered_specs(specs) == (specs[2], specs[0], specs[1])
    restored = SweepGroupingModel.from_dict(model.to_dict())
    assert restored.ordered_specs(specs) == model.ordered_specs(specs)
    model.synchronize(specs[1:])
    model.synchronize(specs)
    assert model.member_id(specs[0]) != ids[0]
    assert ids[0] not in model.execution_order


def test_group_move_button_moves_both_members_and_keeps_selection(window):
    from dc_waveform_core import QickRampRateSweepSpec
    from test_sweep_grouping_gui import check_and_group
    specs = tuple(voltage_fixture(window))
    ramp = QickRampRateSweepSpec('ramp_0_to_1', .1, 1., 2)
    window._sweep_specs.append(ramp)
    window._notify_sweep_state_changed()
    check_and_group(window, specs)
    panel = window._experiment_panel
    panel.sweep_parameter_table.selectRow(row_for(panel, parameter_key(specs[1])))
    assert panel.sweep_move_up_button.isEnabled()
    QtTest.QTest.mouseClick(panel.sweep_move_up_button, QtCore.Qt.LeftButton)
    assert [parameter_key(s) for s in panel._map_sweep_specs] == [
        parameter_key(s) for s in (*specs, ramp)]
    assert parameter_key(panel._selected_sweep_parameter()) == parameter_key(specs[1])
    seq = window._experiment_run_arguments(require_readout=False, require_run_config=False)['sequence']
    assert seq.point_indices(1) == (0, 0, 1)


def test_host_order_changes_coordinates_but_cannot_cross_hardware_boundary():
    from qick_qcodes_experiment import _rf_software_sweep_bindings, _software_sweep_coordinates
    host = [QickRfSoftwareSweepAxisSpec('hold', 'rf0', 0, 1., 2., 2,
                kind, 'us', kind) for kind in ('rf_delay', 'rf_template_tau')]
    voltage = QickSweepSpec('hold', 'a', 0., .1, 3)
    specs = [voltage, *host]
    model = SweepGroupingModel(); model.synchronize(specs)
    assert not model.can_move(model.member_id(voltage), -1, specs)
    assert model.move(model.member_id(host[1]), -1, specs)
    bindings = _rf_software_sweep_bindings(
        [SimpleNamespace(software_sweep_axes=host)], model.compiler_order(specs))
    assert [axis for _, axis in bindings] == list(reversed(host))
    np.testing.assert_array_equal(_software_sweep_coordinates(bindings), [[1,1],[1,2],[2,1],[2,2]])


def test_reordered_rf_axes_emit_duration_frequency_and_voltage_in_requested_order():
    from qick_fine_tune_sweep import RfPulseConfig
    from test_qick_fine_tune_sweep import _shared_tmux_soccfg
    cfg = _shared_tmux_soccfg()
    seq = FineTuneSequence(('a',)).add_set('hold', .01, 3000)
    seq.add_amplitude_sweep('hold', 'a', .01, .02, 2)
    seq.add_rf_duration_sweep('hold', 0, .1, .3, 3, sequence_fabric_mhz=300.)
    seq.add_rf_frequency_sweep('hold', 0, 10., 20., 2)
    voltage, duration, frequency = seq.sweep_axes
    seq.set_sweep_order([seq.sweep_axis_key(a) for a in (frequency, voltage, duration)])
    rf = RfPulseConfig(0, 'hold', 30, 1000, freq_mhz=10., delay_tproc_cycles=30)
    prog = seq.make_program(cfg, awg_channels=(1,), rf_pulses=(rf,), repetitions_per_sweep=2)
    events = [e for e in run_model(prog).output_events if (e.word >> 152) & 255 == 0]
    expected = [(f, d) for f, _v, d in product(frequency.points, voltage.points, duration.points)
                for _rep in range(2)]
    assert [e.word & 0xffffffff for e in events] == [cfg.freq2reg(f, gen_ch=0) for f, _ in expected]
    assert [(e.word >> 128) & 65535 for e in events] == [round(d * 300) for _, d in expected]


def test_late_square_attachment_preserves_explicit_execution_order():
    from qick_square_dds import SquarePulseConfig, SquarePulseSweep, attach_square_settings
    from test_qick_square_dds import configuration
    cfg = configuration()
    seq = FineTuneSequence(('gate',)).add_set('measure', .1, 3000)
    seq.add_amplitude_sweep('measure', 'gate', .1, .2, 2)
    voltage = seq.sweep_axes[0]
    axis = SquarePulseSweep('frequency', 10., 30., 3, 1)
    order = [seq.sweep_axis_key(a) for a in (axis, voltage)]
    seq.set_sweep_order(order)
    settings = SquarePulseConfig(1, mute_on_finish=False)
    attach_square_settings(seq, settings, (axis,))
    assert seq.sweep_shape == (3, 2)
    prog = seq.make_program(cfg, awg_channels=(0,), repetitions_per_sweep=2)
    events = [e for e in run_model(prog).output_events if (e.word >> 152) & 255 == 2]
    assert [e.word & 0xffffffff for e in events] == [
        settings.word('frequency', f, cfg['gens'][1]) for f in axis.points for _ in range(4)]


def test_gui_order_matches_exported_python_sequence(window):
    from dc_waveform_core import generate_qick_program_code
    specs = voltage_fixture(window)
    window._move_sweep_parameter(window._sweep_grouping.member_id(specs[1]), -1)
    code = generate_qick_program_code(window._pulse,
        output_names=window._qick_output_names(), awg_channels=(0, 1),
        sweeps=window._sweep_specs, sweep_order=window._compiler_sweep_order())
    namespace = {}
    exec(code, namespace)
    exported = namespace['build_sequence']()
    runtime = window._experiment_run_arguments(require_readout=False, require_run_config=False)['sequence']
    assert exported.sweep_axes == runtime.sweep_axes
    np.testing.assert_array_equal(exported.sweep_coordinates, runtime.sweep_coordinates)


def test_host_actual_dispatch_and_metadata_follow_reordered_axes(monkeypatch):
    from dataclasses import replace
    import qick_qcodes_experiment as runtime
    from qick_fine_tune_sweep import FineTuneDdrResult
    from dc_waveform_core import QickDdrReadoutSpec, QickRfDurationParameterSpec
    from test_rf_delay_software_sweep import custom_spec
    base = custom_spec()
    spec = custom_spec(delay_parameters=(
        QickRfDurationParameterSpec('tau', 1., True, 1., 3., 3),
        QickRfDurationParameterSpec('idle', 4., True, 4., 5., 2)),
        composite_items=tuple(replace(item, delay_parameter='idle') if item.name == 'wait_b' else item
                              for item in base.composite_items))
    seq = FineTuneSequence(('awg_0',)).add_set('hold', .01, 6000)
    seq.add_amplitude_sweep('hold', 'awg_0', .01, .02, 2)
    seq.set_sweep_order([(a.axis_kind, a.segment_name, a.output_name)
                        for a in reversed(spec.software_sweep_axes)])
    calls = []
    def execute(_soc, _cfg, concrete_sequence, **kwargs):
        values = {p.name: p.duration_us for p in kwargs['rf_specs'][0].delay_parameters}
        calls.append((values['idle'], values['tau']))
        result = FineTuneDdrResult(sweep_points=np.array([.01, .02]),
            iq=np.zeros((2, 1, 1, 2), dtype=np.int32), reserved_physical_words=128,
            sweep_axes=concrete_sequence.sweep_axes, sweep_shape=(2,))
        return SimpleNamespace(summary=lambda: {}), result, {'outputs': []}
    monkeypatch.setattr(runtime, '_execute_qick_sequence_once', execute)
    _, result, metadata = runtime.execute_qick_sequence(
        object(), object(), seq, awg_channels=(1,), repetitions_per_sweep=1,
        rf_specs=(spec,), readout_spec=QickDdrReadoutSpec(0, 'hold', 0., 1))
    assert calls == list(product((4., 5.), (1., 2., 3.)))
    assert result.sweep_shape == (2, 3, 2)
    np.testing.assert_allclose(result.sweep_points, list(product((4., 5.), (1., 2., 3.), (.01, .02))))
    assert [a['parameter_name'] for a in metadata['software_sweep']['axes']] == ['idle', 'tau']


def test_reordered_table_capacity_is_checked_before_expansion():
    cfg = config()
    cfg['tprocs'][0]['dmem_size'] = 128
    seq = FineTuneSequence(('a', 'b'))
    seq.add_set('zero', (0., 0.), 300).add_ramp('rise', 3).add_set('hold', (.1, .05), 3000)
    seq.add_ramp_duration_sweep('rise', .01, 1., 100)
    seq.add_amplitude_sweep('hold', 'a', .1, .2, 100)
    seq.set_sweep_order([seq.sweep_axis_key(a) for a in reversed(seq.sweep_axes)])
    with pytest.raises(RuntimeError, match='reordered sweep coefficient table does not fit'):
        seq.make_program(cfg, awg_channels=(0, 1), compile_validation_mode='boundary')
