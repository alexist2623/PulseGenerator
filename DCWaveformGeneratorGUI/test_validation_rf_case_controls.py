"""Single RF actual Qt inputs and sweep grids; the fixture denies all sockets."""
import numpy as np
import pytest

from test_mainwindow_case_bindings import window, fixture, case
from validation.awg_reference import ReferenceUnavailable
from validation.mainwindow_cases import MainWindowCaseBinding
from validation.qt_runner import apply_actions
from validation.rf_case_controls import RfCaseBinding


def rf_pulse(**changes):
    value = dict(name='RF A', generator=0, anchor_stage=1, delay_us=.1, duration_us=.025,
        duration_mode='fine', frequency_mhz=75., gain=12345, amplitude_v=.123,
        phase_degrees=90., require_within_segment=True)
    value.update(changes)
    return value


@pytest.mark.parametrize('mode,within', [('fine', True), ('coarse', False)])
def test_fixed_rf_inputs_are_actual_controls_and_separate_gain_from_reference(window, mode, within):
    base = MainWindowCaseBinding(window, fixture(), case())
    apply_actions(base.actions())
    awg_before_rf = base.verify_requested_case()
    spec = dict(rf_pulses=[rf_pulse(duration_mode=mode, require_within_segment=within)])
    binding = RfCaseBinding(window, fixture(), spec)
    events = apply_actions(binding.actions())
    result = binding.verify_requested_case()
    pulse = result['rf_pulses'][0]
    assert pulse['scope_channel'] == '1' and pulse['physical_connector'] == 'DAC4'
    assert pulse['gain'] == 12345 and pulse['reference_amplitude_v'] == .123
    assert pulse['duration_mode'] == mode and pulse['require_within_segment'] is within
    assert any(e['operation']=='text' for e in events)
    assert not result['physical_amplitude_verified'] and not result['run_performed_by_binding']
    assert window._experiment_thread is None
    # RF controls use their own snapshot. An AWG-only reference is no longer
    # valid once RF is enabled, even though the matrix itself is unchanged.
    np.testing.assert_array_equal(window._cross_capacitance, awg_before_rf['matrix'])
    with pytest.raises(ReferenceUnavailable, match='AWG-only.*active RF'):
        base.snapshot()


def test_two_pulses_use_their_explicit_generator_and_connector(window):
    apply_actions(MainWindowCaseBinding(window, fixture(), case()).actions())
    binding = RfCaseBinding(window, fixture(), dict(rf_pulses=[rf_pulse(),
        rf_pulse(name='RF B', generator=2, frequency_mhz=150., gain=-5000)]))
    apply_actions(binding.actions())
    pulses = binding.snapshot()['rf_pulses']
    assert [(p['generator'],p['physical_connector'],p['scope_channel']) for p in pulses] == [
        (0,'DAC4','1'), (2,'DAC5','2')]
    assert pulses[1]['gain'] == -5000


@pytest.mark.parametrize('unsupported', [dict(pulse_mode='composite'),
    dict(amplitude_mode='calibration'), dict(power_calibration_enabled=True), dict(power_sweep_enabled=True)])
def test_unsupported_rf_controls_are_explicit_unavailable_before_gui_events(window, unsupported):
    before = window._settings_to_dict()
    with pytest.raises(ReferenceUnavailable, match='Single/manual'):
        RfCaseBinding(window, fixture(), dict(rf_pulses=[rf_pulse(**unsupported)]))
    assert before == window._settings_to_dict()


def test_physical_reference_amplitude_cannot_replace_gui_gain(window):
    pulse = rf_pulse()
    del pulse['gain']
    with pytest.raises(ValueError, match='explicit gain'):
        RfCaseBinding(window, fixture(), dict(rf_pulses=[pulse]))


@pytest.mark.parametrize('changes', [
    dict(duration_sweep=dict(start_us=1/4800, stop_us=31/4800, count=31)),
    dict(duration_mode='coarse', require_within_segment=False,
         duration_sweep=dict(start_us=.2, stop_us=.02, count=4),
         frequency_sweep=dict(start_mhz=80., stop_mhz=20., count=3)),
    dict(duration_sweep=dict(start_us=.025, stop_us=.025, count=1),
         frequency_sweep=dict(start_mhz=50., stop_mhz=50., count=1)),
    dict(duration_us=12., require_within_segment=False,
         frequency_sweep=dict(start_mhz=-50., stop_mhz=50., count=3)),
])
def test_actual_rf_sweep_controls_reach_public_run_argument_grid(window, changes):
    base = MainWindowCaseBinding(window, fixture(), case())
    apply_actions(base.actions())
    binding = RfCaseBinding(window, fixture(), dict(rf_pulses=[rf_pulse(**changes)]))
    events = apply_actions(binding.actions())
    snap = binding.verify_requested_case()['rf_pulses'][0]
    arguments = window._experiment_run_arguments(require_readout=False, require_run_config=False)
    seq, spec = arguments['sequence'], arguments['rf_specs'][0]
    expected_shape = [3]  # The two AWG voltage members share one group.
    for name, kind, unit in (('duration', 'rf_duration', 'us'), ('frequency', 'rf_frequency', 'mhz')):
        requested = binding.case['rf_pulses'][0][name + '_sweep']
        rf_axes = [a for a in seq.sweep_axes if a.axis_kind == kind]
        assert getattr(spec, name+'_sweep_enabled') is (requested is not None)
        if requested is None:
            assert not rf_axes and snap[name+'_sweep'] is None
            continue
        expected_shape.append(requested['count'])
        assert len(rf_axes) == 1 and rf_axes[0].count == requested['count']
        committed = snap[name+'_sweep']
        np.testing.assert_allclose(rf_axes[0].points,
            np.linspace(committed[f'start_{unit}'], committed[f'stop_{unit}'], requested['count']),
            rtol=0, atol=1e-12)
        assert sum(e['operation']=='text' for e in events) >= 10
    assert seq.sweep_shape == tuple(expected_shape)
    assert seq.sweep_point_count == int(np.prod(expected_shape))
    assert seq.sweep_coordinates.shape[0] == seq.sweep_point_count
    assert spec.require_within_segment == binding.case['rf_pulses'][0]['require_within_segment']
    assert spec.fine_duration == (binding.case['rf_pulses'][0]['duration_mode']=='fine')
    assert window._experiment_thread is None  # Prepared Run arguments, not a physical Run.


@pytest.mark.parametrize('changes', [
    dict(duration_sweep=dict(start_us=0., stop_us=.1, count=3)),
    dict(duration_sweep=dict(start_us=.1, stop_us=.2, count=0)),
    dict(frequency_sweep=dict(start_mhz=float('nan'), stop_mhz=100., count=3)),
    dict(frequency_sweep_enabled=True),
])
def test_invalid_sweep_definition_is_rejected_before_events(window, changes):
    before = window._settings_to_dict()
    with pytest.raises(ValueError):
        RfCaseBinding(window, fixture(), dict(rf_pulses=[rf_pulse(**changes)]))
    assert before == window._settings_to_dict()


@pytest.mark.parametrize('readback', ['snapshot', 'verify_requested_case'])
def test_awg_only_readback_rejects_rf_enabled_after_awg_preparation(window, readback):
    base = MainWindowCaseBinding(window, fixture(), case())
    apply_actions(base.actions())
    before = base.verify_requested_case()
    rf = RfCaseBinding(window, fixture(), dict(rf_pulses=[rf_pulse()]))
    apply_actions(rf.actions())
    assert rf.verify_requested_case()['rf_pulses'][0]['enabled']
    np.testing.assert_array_equal(window._cross_capacitance, before['matrix'])
    with pytest.raises(ReferenceUnavailable, match='AWG-only.*active RF'):
        getattr(base, readback)()
    assert window._experiment_thread is None
