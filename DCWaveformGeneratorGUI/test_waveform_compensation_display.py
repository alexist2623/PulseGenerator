"""Live channel editing limits and independent compensation display controls."""
from dataclasses import replace

import numpy as np
import pytest

from qick_waveform_preview import rc_precompensated_vertices
from test_square_awg_exclusion import window, firmware
from test_dac_current import records


def test_rc_integrates_steps_zero_holds_and_balanced_dc_pulse():
    # 100 mV for 100 us; wait at zero for 50 us; -50 mV for 200 us.
    t = np.array([0, 100, 100, 150, 150, 350, 350]) * 1000.
    values = np.array([[100, 100, 0, 0, -50, -50, 0.]])
    times, actual = rc_precompensated_vertices(t, values, 300)
    np.testing.assert_array_equal(times, t)
    np.testing.assert_allclose(actual[0], [100, 100+100/3, 100/3, 100/3,
                                         -50+100/3, -50, 0], atol=1e-12)


def test_rc_ramp_is_quadratic_and_includes_interior_extrema():
    times, result = rc_precompensated_vertices([0, 20000], [[100, -100]], 10)
    us = times / 1000
    np.testing.assert_allclose(result[0], 100 - 10*us + (100*us-5*us**2)/10)
    assert len(times) <= 34
    # A second slope puts an extremum inside the ramp at 5 us.
    times, result = rc_precompensated_vertices([0, 20000], [[150, -50]], 10, 4)
    assert 5000 in times
    assert max(result[0]) == pytest.approx(162.5)


def test_preview_bounded_memory_for_long_hold_and_independent_channels():
    times, result = rc_precompensated_vertices([0, 1e12], [[1, 1], [-2, -2]], 1e6)
    assert len(times) == 2
    np.testing.assert_allclose(result[:, -1], [1001, -2002])


@pytest.mark.parametrize('tau', [0, -1, float('nan'), float('inf')])
def test_invalid_tau_rejected(tau):
    with pytest.raises(ValueError):
        rc_precompensated_vertices([0, 10], [[1, 1]], tau)


def test_editor_limits_follow_current_channel_remapping_and_added_ports(window):
    window._on_qick_configuration_identified(replace(
        firmware(), dac_current_settings=records((32000, 10000, 20000))))
    assert window._pulse[0].v_bounds == (-1280, 1280)
    # Use the real editor model, not direct array assignment (the old bug).
    window._pulse[0].edit_voltage(0, 1000)
    assert window._pulse[0].v[0] == 1000
    seq = window._experiment_run_arguments(require_readout=False, require_run_config=False)['sequence']
    assert seq.amplitudes_at(0, 0)[0] == pytest.approx(1000/1280)
    window._add_port()
    assert window._pulse[1].v_bounds == (-400, 400)
    window._set_awg_output_channel(0, 3)
    assert window._pulse[0].v_bounds == (-400, 400)
    assert window._pulse[1].v_bounds == (-1280, 1280)
    assert window._pulse[0].v[0] == 1000  # Never silently rewrite existing volts.
    window._pulse[0].edit_voltage(0, 300)
    window._pulse[1].edit_voltage(0, -1000)
    assert window._pulse[1].v[0] == -1000
    assert 'gen 3: +/-400 mV' in window._waveform_scale_label.text()
    assert 'gen 1: +/-1280 mV' in window._waveform_scale_label.text()


def configure_compensated_pulse(window):
    window._pulse[0].t = np.array([0., 100000.])
    window._pulse[0].v = np.array([100., 100.])
    panel = window._experiment_panel
    panel.bias_t_compensation_mv.setValue(100)
    panel.bias_t_filter_tau_us.setValue(300)
    panel.bias_t_type.dc_checkbox.setChecked(True)
    panel.bias_t_type.rc_checkbox.setChecked(True)
    window._refresh_physical_waveforms(fit_view=True)


def test_display_switches_do_not_change_rc_history_or_experiment(window):
    configure_compensated_pulse(window)
    plot = window._plot
    assert plot._physical_time_ns[-1] > 200000
    assert np.max(plot._rc_values_mv) == pytest.approx(100+100/3)
    assert plot._rc_values_mv[0, -1] == pytest.approx(0, abs=1e-10)
    original = plot._rc_values_mv.copy()
    args_before = window._experiment_run_arguments(require_readout=False, require_run_config=False)
    window._show_dc_compensation.setChecked(False)
    assert plot._physical_time_ns[-1] == 100000
    np.testing.assert_array_equal(plot._rc_values_mv, original)
    window._show_rc_compensation.setChecked(False)
    assert not plot._rc_time_ns.size
    assert all(not curve.isVisible() for curve in plot._rc_line)
    args_after = window._experiment_run_arguments(require_readout=False, require_run_config=False)
    assert args_after['sequence'].bias_t_compensation == args_before['sequence'].bias_t_compensation
    assert args_after['sequence'].rc_compensation == args_before['sequence'].rc_compensation
    window._show_rc_compensation.setChecked(True)
    np.testing.assert_array_equal(plot._rc_values_mv, original)
    window._experiment_panel.bias_t_type.rc_checkbox.setChecked(False)
    assert not plot._rc_time_ns.size


def test_out_of_range_rc_still_plotted_but_execution_is_rejected(window):
    configure_compensated_pulse(window)
    window._experiment_panel.bias_t_filter_tau_us.setValue(10)
    assert np.max(window._plot._rc_values_mv) > 800
    assert 'exceeds range' in window._waveform_preview_note.text()
    with pytest.raises(ValueError, match='RC-compensated DAC output exceeds range'):
        window._experiment_run_arguments(require_readout=False, require_run_config=False)


def test_display_settings_roundtrip_and_stale_file_bounds(window):
    configure_compensated_pulse(window)
    window._on_qick_configuration_identified(replace(
        firmware(), dac_current_settings=records((32000, 10000, 20000))))
    window._show_dc_compensation.setChecked(False)
    window._show_rc_compensation.setChecked(False)
    data = window._settings_to_dict()
    data['awg']['outputs'][0]['voltage_bounds_mv'] = [-800, 800]
    window._apply_decoded_settings(window._decode_settings(data))
    assert window._pulse[0].v_bounds == (-1280, 1280)
    assert not window._show_dc_compensation.isChecked()
    assert not window._show_rc_compensation.isChecked()
    assert window._experiment_panel.bias_t_type.dc_checkbox.isChecked()
    assert window._experiment_panel.bias_t_type.rc_checkbox.isChecked()
    del data['display']['show_dc_compensation']
    del data['display']['show_rc_compensation']
    legacy = window._decode_settings(data)
    assert legacy['show_dc_compensation'] and legacy['show_rc_compensation']


def test_two_channel_rc_preview_preserves_voltage_when_current_changes(window):
    window._add_port()
    for pulse, level in zip(window._pulse, (100., -50.)):
        pulse.t = np.array([0., 100000.])
        pulse.v = np.array([level, level])
    window._cross_capacitance = np.array([[1., .2], [-.1, 1.]])
    panel = window._experiment_panel
    panel.bias_t_compensation_mv.setValue(100)
    panel.bias_t_filter_tau_us.setValue(300)
    panel.bias_t_type.dc_checkbox.setChecked(True)
    panel.bias_t_type.rc_checkbox.setChecked(True)
    window._refresh_physical_waveforms()
    before_t = window._plot._rc_time_ns.copy()
    before_v = window._plot._rc_values_mv.copy()
    np.testing.assert_allclose(before_v[:, 0], [90, -60])
    window._on_qick_configuration_identified(replace(
        firmware(), dac_current_settings=records((32000, 10000, 20000))))
    np.testing.assert_array_equal(window._plot._rc_time_ns, before_t)
    np.testing.assert_allclose(window._plot._rc_values_mv, before_v, atol=1e-12)
    window._set_voltage_view('virtual')
    assert all(not curve.isVisible() for curve in window._plot._rc_line)
    window._set_voltage_view('physical')
    assert all(curve.isVisible() for curve in window._plot._rc_line)


def test_rc_curve_auto_fit_and_port_deletion(window):
    configure_compensated_pulse(window)
    window._add_port()
    assert len(window._plot._rc_line) == 2
    window._delete_port(1)
    assert len(window._plot._rc_line) == 1
    window._plot.fit_view()
    assert window._plot.getViewBox().viewRange()[1][1] > np.max(window._plot._rc_values_mv)
