"""Offline Ramsey editor -> production compiler/export -> tProcessor checks."""
from dataclasses import asdict

import numpy as np
import pytest
from PyQt5 import QtCore, QtWidgets

import qick.sim  # noqa: F401 - install desktop QICK driver stubs first
from qick.awg_tuning import TProcV1BehaviorModel
from qick.output_sum import apply_output_sum_config

import DCWaveform_Generator as gui
from dc_waveform_core import (
    PulseSequence, QickRfPulseSpec, build_qick_sequence,
    generate_qick_program_code,
)
from qick_qcodes_experiment import (
    _rf_software_sweep_bindings, _rf_specs_at_software_coordinate,
    _software_sweep_coordinates, build_runtime_rf_pulses,
)
from qick_rf_fine_duration import plan_duration
from test_qick_fine_tune_sweep import _shared_tmux_soccfg


_APP = None


@pytest.fixture
def window():
    global _APP
    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = gui.MainWindow()
    yield widget
    widget.close()
    widget.deleteLater()
    _APP.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    _APP.processEvents()


def _configuration():
    cfg = _shared_tmux_soccfg()
    cfg['gens'][0]['f_dds'] = 4800.
    cfg['gens'][1].update(type='axis_awg_tuning_v3', frac=18,
                         step_width=32, rc_precomp_version=1,
                         dc_area_version=1, dc_area_bits=48,
                         dc_start_cycles=17, output_latency_cycles=16)
    for channel, role in ((0, 'rf'), (1, 'awg')):
        apply_output_sum_config(cfg['gens'][channel], dict(
            output_sum_version=2, output_sum_role=role,
            post_output_latency_cycles=2, output_sum_awg_delay_cycles=13))
        cfg['gens'][channel]['dac'] = '10'
    return cfg


def _editor_spec(window, *, fine, pi_sweep=False):
    # For fine mode each pi/2 is 97 DAC samples, requiring a DDS plateau
    # followed by a masked envelope tail. The displayed pi duration is rounded
    # to the same nine decimal places used by the editor.
    pi_duration = round(194 / 4800., 9) if fine else .4
    pi_stop = round(226 / 4800., 9) if fine else .8
    settings = dict(gui.DEFAULT_RF_OUTPUT_SETTINGS, enabled=True,
        gen_ch=0, segment_name='set_0', delay_us=.1,
        pulse_mode='composite', segment_length_mode='fixed',
        predefined_template='ramsey', predefined_n=4,
        predefined_tau_us=1., predefined_tau_sweep_enabled=True,
        predefined_tau_sweep_start_us=1., predefined_tau_sweep_stop_us=3.,
        predefined_tau_sweep_count=3, fine_duration=fine,
        duration_parameters=[dict(name='d0', duration_us=pi_duration,
            sweep_enabled=pi_sweep, sweep_start_us=pi_duration,
            sweep_stop_us=pi_stop, sweep_count=2)],
        frequency_parameters=[dict(name='f0', frequency_mhz=37.5)],
        amplitude_parameters=[dict(name='a0', value=2345)],
        phase_parameters=[dict(name='p0', phase_degrees=45.),
                          dict(name='p1', phase_degrees=135.)],
        global_frequency_parameter='f0', global_amplitude_mode='manual',
        global_amplitude_parameter='a0',
        predefined_duration_parameter='d0',
        predefined_frequency_parameter='f0',
        predefined_amplitude_parameter='a0', predefined_phase_parameter='p0',
        t2_duration_parameter='d0', t2_frequency_parameter='f0',
        t2_amplitude_parameter='a0', t2_initial_phase_parameter='p0',
        t2_final_phase_parameter='p1', att1_db=0., att2_db=0.,
        filter_type='bypass')
    panel = window._rf_ports_panel._panels[0]
    panel.load_settings(settings)
    _APP.processEvents()
    spec = panel.configured_spec()
    assert spec.predefined_template == 'ramsey'
    assert spec.t2_initial_pi2 and spec.t2_final_pi2
    assert [event.gain for event in spec.pulse_events] == [2345, 2345]
    assert [event.phase_degrees for event in spec.pulse_events] == [45., 135.]
    return spec


def _compiled_point(spec):
    cfg = _configuration()
    awg = (PulseSequence(0., initial_duration_ns=10_000.),)
    runtime = build_runtime_rf_pulses(cfg, (spec,), tproc_mhz=300.)
    sequence = build_qick_sequence(awg, output_names=('awg_0',),
                                  fabric_mhz=300., rf_pulse_specs=(spec,))
    live = sequence.make_program(cfg, awg_channels=(1,), tproc_mhz=300.,
                                 rf_pulses=runtime, repetitions_per_sweep=1)
    source = generate_qick_program_code(awg, output_names=('awg_0',),
        awg_channels=(1,), fabric_mhz=300., tproc_mhz=300.,
        rf_pulse_specs=(spec,), repetitions_per_sweep=1)
    namespace = {}
    exec(compile(source, '<ramsey-export>', 'exec'), namespace)
    assert QickRfPulseSpec(**namespace['RF_CONFIGS'][0]) == spec
    assert asdict(QickRfPulseSpec(**namespace['RF_CONFIGS'][0])) == asdict(spec)
    exported = namespace['build_program'](cfg)
    live.compile()
    exported.compile()
    assert live.asm() == exported.asm()
    np.testing.assert_array_equal(live.binprog, exported.binprog)
    np.testing.assert_array_equal(live._runtime_dmem_words,
                                  exported._runtime_dmem_words)
    model = TProcV1BehaviorModel(strict=True)
    live.load_runtime_dmem_into_model(model)
    model.run(live, max_steps=100_000)
    assert not model.timing_conflicts
    events = [event for event in model.output_events
              if event.tproc_ch == 0 and (event.word >> 152) & 255 == 0
              and (event.word >> 96) & 65535]
    return live, runtime, events


def _active_samples(program, event):
    mode = (event.word >> 128) & 0xffffffff
    length = mode & 65535
    selection = (mode >> 16) & 3
    if selection == 1:
        return length * 16
    memory = np.zeros(16384, dtype=np.int16)
    for envelope in program.envelopes[0]['envs'].values():
        start = envelope['addr']
        memory[start:start + len(envelope['data'])] = envelope['data'][:, 0]
    address = ((event.word >> 64) & 0xffffffff) * 16
    return int(np.count_nonzero(memory[address:address + length * 16]))


def _assert_ramsey_words(program, runtime, events, *, samples):
    assert len(runtime) == 2
    command_count = 1 + plan_duration(samples).long if runtime[0].fine_duration else 1
    assert len(events) == 2 * command_count
    initial, final = events[:command_count], events[command_count:]
    assert final[0].cycle - initial[0].cycle == (
        runtime[1].delay_tproc_cycles - runtime[0].delay_tproc_cycles)
    for pulse, commands in zip(runtime, (initial, final)):
        assert sum(_active_samples(program, event) for event in commands) == samples
        expected_phase = int(program.deg2reg(pulse.phase_degrees, gen_ch=0))
        assert all((event.word >> 32) & 0xffffffff == expected_phase
                   for event in commands)
        expected_frequency = program._rf_frequency_word(pulse, 37.5)
        assert all(event.word & 0xffffffff == expected_frequency for event in commands)
    if runtime[0].fine_duration:
        assert len(program.envelopes[0]['envs']) == 16
        assert sum(len(envelope['data'])
                   for envelope in program.envelopes[0]['envs'].values()) == 2048


@pytest.mark.parametrize('fine', [False, True])
@pytest.mark.parametrize('tau_index', range(3))
def test_ramsey_gui_tau_points_export_and_live_program_have_identical_words(
        window, fine, tau_index):
    spec = _editor_spec(window, fine=fine)
    bindings = _rf_software_sweep_bindings((spec,))
    assert [axis.axis_kind for _, axis in bindings] == ['rf_template_tau']
    coordinates = _software_sweep_coordinates(bindings)
    np.testing.assert_array_equal(coordinates[:, 0], [1., 2., 3.])
    point, = _rf_specs_at_software_coordinate((spec,), bindings, coordinates[tau_index])
    assert not point.software_sweep_axes
    first, last = point.pulse_events
    assert last.delay_us - first.delay_us - first.duration_us == pytest.approx(tau_index + 1.)
    program, runtime, events = _compiled_point(point)
    _assert_ramsey_words(program, runtime, events, samples=97 if fine else 960)


@pytest.mark.parametrize('fine', [False, True])
def test_ramsey_gui_pi_duration_sweep_halves_both_compiled_endpoints(window, fine):
    spec = _editor_spec(window, fine=fine, pi_sweep=True)
    bindings = _rf_software_sweep_bindings((spec,))
    assert [axis.axis_kind for _, axis in bindings] == ['rf_template_tau', 'rf_t2_duration']
    coordinates = _software_sweep_coordinates(bindings)
    assert coordinates.shape == (6, 2)
    point, = _rf_specs_at_software_coordinate((spec,), bindings, coordinates[-1])
    assert not point.software_sweep_axes
    assert [event.duration_us for event in point.pulse_events] == pytest.approx(
        [coordinates[-1, 1] / 2] * 2)
    program, runtime, events = _compiled_point(point)
    _assert_ramsey_words(program, runtime, events, samples=113 if fine else 1920)
