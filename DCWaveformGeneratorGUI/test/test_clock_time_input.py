"""Clock-aligned GUI input, serialization, large durations and FIR isolation."""
import json
from decimal import Decimal, ROUND_CEILING

import pytest
from PyQt5 import QtCore, QtTest, QtWidgets

from clock_time import clock_cycles, clock_time
from clock_time_widgets import ClockTimeSpinBox


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize("rate", [250., 300., 4800.])
def test_serialized_clock_points_do_not_gain_a_clock(rate):
    for ticks in range(1, 4801):
        rounded = float(f"{ticks / rate:.9f}")
        assert clock_cycles(rounded, rate) == ticks


@pytest.mark.parametrize("duration", [1000., 1e6, 1e9, 1e12])
@pytest.mark.parametrize("rate", [300., 4800.])
def test_large_times_preserve_integers_and_real_fractional_excess(duration, rate):
    assert clock_time(duration, rate) == duration
    for fraction in (.25, .5, .75):
        value = duration + fraction / rate
        expected = int((Decimal(str(value)) * Decimal(str(rate))).to_integral_value(rounding=ROUND_CEILING))
        assert clock_cycles(value, rate) == expected


def test_true_fractional_clock_still_rounds_up():
    from dc_waveform_core import _cycles_from_ns
    from qick_fine_tune_sweep import cycles_from_us
    from qick_rf_fine_duration import samples_from_us
    assert cycles_from_us(.006666667, 300) == 2
    assert _cycles_from_ns(10/3 + 1e-6, 300) == 2
    assert _cycles_from_ns(15, 300) == 5
    assert samples_from_us(1.01/4800, 300) == 2
    assert clock_time(0, 300) == 0
    for invalid in (float('nan'), float('inf'), -1):
        with pytest.raises(ValueError):
            clock_time(invalid, 300)


def enter(spin, text):
    spin.lineEdit().selectAll()
    QtTest.QTest.keyClicks(spin.lineEdit(), text)
    QtTest.QTest.keyClick(spin, QtCore.Qt.Key_Return)


def test_typed_input_and_steps_keep_exact_clock(app):
    spin = ClockTimeSpinBox(rate_mhz=lambda: 300)
    spin.setDecimals(9)
    spin.setRange(0, 1e12)
    enter(spin, '0.006666667')
    assert spin.value() == 2/300
    spin.stepBy(1)
    assert spin.value() == 3/300
    spin.stepBy(-2)
    assert spin.value() == 1/300
    for value in (1000., 1e6, 1e9, 1e12):
        enter(spin, str(value))
        assert spin.value() == value
    spin.deleteLater()


@pytest.fixture
def window(app):
    import DCWaveform_Generator as gui
    widget = gui.MainWindow()
    yield widget
    widget.close()
    widget.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


def test_gui_rf_delay_roundtrip_compiles_two_clocks(window):
    import DCWaveform_Generator as gui
    from qick_qcodes_experiment import build_runtime_rf_pulses
    from test_awg_v6_timing import configuration
    panel = window._rf_ports_panel._panels[0]
    panel.load_settings({**gui.DEFAULT_RF_OUTPUT_SETTINGS, 'enabled': True,
                         'gen_ch': 0, 'delay_us': 2/300, 'duration_us': .1})
    for unit in ('ns', 'us') * 5:
        panel.set_time_unit(unit)
    saved = json.loads(json.dumps(panel.settings_dict()))
    assert saved['delay_us'] == 2/300
    panel.load_settings(saved)
    runtime = build_runtime_rf_pulses(configuration(), [panel.configured_spec()])
    assert runtime[0].delay_tproc_cycles == 2
    enter(panel.delay, '0.006666667')
    assert panel.configured_spec().delay_us == 2/300


@pytest.mark.parametrize('value', [1000., 1e9, 1e12])
def test_large_gui_values_survive_unit_changes_without_range_clamping(window, value):
    panel = window._rf_ports_panel._panels[0]
    panel.set_time_unit('us')
    panel.delay.setValue(value)
    for unit in ('ns', 'us') * 3:
        panel.set_time_unit(unit)
    assert panel.delay.value() == value
    assert panel.settings_dict()['delay_us'] == value


def test_value_changed_observers_receive_full_imported_precision(app):
    spin = ClockTimeSpinBox(rate_mhz=lambda: 300)
    observed = []
    spin.valueChanged.connect(lambda value: observed.append((value, spin.value())))
    spin.setValue(2/300)
    assert observed == [(2/300, 2/300)]
    spin.deleteLater()


def test_fine_duration_uses_dac_samples_but_delay_stays_on_tproc(window):
    import DCWaveform_Generator as gui
    panel = window._rf_ports_panel._panels[0]
    panel.load_settings({**gui.DEFAULT_RF_OUTPUT_SETTINGS, 'enabled': True, 'gen_ch': 0})
    panel.set_time_unit('us')
    panel.fine_duration.setChecked(True)
    enter(panel.duration, '0.000208333')
    enter(panel.delay, '0.000208333')
    assert panel.duration.value() == 1/4800
    assert panel.delay.value() == 1/300
    panel.set_generator_timings([dict(f_fabric=250, samps_per_clk=16)])
    panel.gen_ch.setValue(0)
    enter(panel.duration, '0.00025')
    assert panel.duration.value() == 1/4000


def test_awg_table_and_duration_sweeps_are_clock_aligned(app):
    import DCWaveform_Generator as gui
    from dc_waveform_core import PulseSequence
    pulse = PulseSequence(100., initial_duration_ns=100.)
    panel = gui.ControlPanel(pulse, time_unit='ns')
    panel.table.item(0, 3).setText('6.66')
    assert clock_cycles(pulse.t[1], 300, units_per_us=1000) == 2
    text = panel.table.item(0, 3).text()
    panel.table.item(0, 3).setText(text)
    assert clock_cycles(pulse.t[1], 300, units_per_us=1000) == 2
    for cls, kwargs in ((gui.RampRateSweepSettingsDialog, dict(voltage_delta_mv=100)),
                        (gui.HoldDurationSweepSettingsDialog, {})):
        dialog = cls(segment_name='set_0', current_duration_us=.1, **kwargs)
        enter(dialog.start, '0.006666667')
        assert dialog.value().start == 2/300
        dialog.deleteLater()
    voltage = gui.SweepSettingsDialog(output_name='awg_0', segment_name='set_0', current_amplitude=.1,
                                     full_scale_mv=1000.)
    assert not isinstance(voltage.start, ClockTimeSpinBox)
    voltage.start.setValue(-.123456)
    assert voltage.start.value() == -.123456
    voltage.deleteLater()
    panel.deleteLater()


def test_fir_integer_samples_and_microseconds_unchanged(window):
    import DCWaveform_Generator as gui
    from dc_waveform_core import PulseSequence
    from fir_ddr_profile import resolve_fir_ddr_profile
    from test_fir_ddr_profile import _soccfg
    panel = gui.RfReadoutPanel(PulseSequence(100., initial_duration_ns=100.), time_unit='us')
    panel.setChecked(True)
    panel._fir_sample_rate_hz = 1e6
    profile = resolve_fir_ddr_profile(_soccfg('1_msps'))
    assert profile.sample_period_us == 1
    for count in (1, 10, 1000, 1_000_000, 10_000_000):
        panel.samples.setValue(count)
        panel.delay.setValue(float(count))
        for unit in ('ns', 'us'):
            panel.set_time_unit(unit)
        saved = json.loads(json.dumps(panel.settings_dict()))
        assert saved['samples_per_trigger'] == count
        assert saved['delay_us'] == float(count)
        assert count * profile.sample_period_us == count
    panel.deleteLater()


def test_dc_duration_follows_clock_without_quantizing_rc_tau(window):
    panel = window._experiment_panel
    panel.bias_t_group.setChecked(True)
    panel.bias_t_type.setCurrentIndex(panel.bias_t_type.findData('dc_rc'))
    panel.bias_t_mode.setCurrentIndex(panel.bias_t_mode.findData('fixed_time'))
    enter(panel.bias_t_duration_us, '0.006666667')
    assert panel.bias_t_duration_us.value() == 2/300
    panel.fabric_mhz.setValue(250.)
    enter(panel.bias_t_duration_us, '0.006666667')
    assert panel.bias_t_duration_us.value() == 2/250
    assert not isinstance(panel.bias_t_filter_tau_us, ClockTimeSpinBox)
    panel.bias_t_filter_tau_us.setValue(20.123456)
    assert panel.bias_t_filter_tau_us.value() == 20.123456
