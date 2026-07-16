"""Output-scope and input-ADC calibration tests.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtWidgets

from calibration_gui import CalibrationPanel, input_calibration_plot_data
from power_calibration import CalibrationDatabase, MAX_QICK_GAIN
from qick_power_calibration import (
    InputPowerCalibrationConfig,
    KeysightFftPowerMeter,
    OscilloscopeConfig,
    OutputPowerCalibrationConfig,
    run_input_power_calibration,
    run_output_power_calibration,
)
from qick_qcodes_experiment import QCODES_STAGING_ENV, QickConnectionConfig
from qick_sparameter_sweep import SParameterSweepResult, apply_power_calibration


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeSoc:
    def rfb_set_gen_rf(self, _channel, att1, att2):
        return float(att1), float(att2)

    def rfb_set_gen_filter(self, *_args, **_kwargs):
        return None

    def rfb_set_ro_rf(self, _channel, attenuation):
        return float(attenuation)

    def rfb_set_ro_filter(self, *_args, **_kwargs):
        return None


def test_keysight_fft_adapter_uses_original_notebook_scpi(monkeypatch):
    class Instrument:
        def __init__(self):
            self.commands = []
            self.responses = iter(("-20", "-22"))
            self.timeout = None
            self.write_termination = None
            self.read_termination = None

        def write(self, command):
            self.commands.append(command)

        def query(self, command):
            self.commands.append(command)
            if command == "*IDN?":
                return "AGILENT TECHNOLOGIES,DSO-X 6004A,TEST,1.0"
            return next(self.responses)

        def close(self):
            self.commands.append("CLOSE")

    instrument = Instrument()

    class ResourceManager:
        def open_resource(self, resource):
            assert resource == "USB::TEST"
            return instrument

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "pyvisa",
        SimpleNamespace(ResourceManager=ResourceManager),
    )
    config = OscilloscopeConfig(
        visa_resource="USB::TEST",
        average_count=2,
        settle_seconds=0.0,
        sample_interval_seconds=0.0,
    )
    with KeysightFftPowerMeter(config) as meter:
        assert meter.measure_power_dbm(450.0) == -21.0

    assert ":FUNCtion1:OPERation FFT" in instrument.commands
    assert ":FUNCtion1:SOURce CHANnel2" in instrument.commands
    assert ":MARKer:X1Position 450000000" in instrument.commands
    assert instrument.commands.count(":MARKer:Y1Position?") == 2


def test_input_calibration_plot_data_restores_positive_linear_quantities():
    frequencies, gains, input_mw, adc_magnitude = input_calibration_plot_data(
        {
            "frequencies_mhz": [400.0, 500.0],
            "gains": [1000, 2000],
            "input_power_dbm": [[-30.0, -20.0], [-20.0, -10.0]],
            "adc_magnitude_db": [
                [0.0, 20.0],
                [20.0 * np.log10(2.0), 40.0],
            ],
        }
    )

    np.testing.assert_array_equal(frequencies, [400.0, 500.0])
    np.testing.assert_array_equal(gains, [1000.0, 2000.0])
    np.testing.assert_allclose(input_mw, [[0.001, 0.01], [0.01, 0.1]])
    np.testing.assert_allclose(adc_magnitude, [[1.0, 10.0], [2.0, 100.0]])


def test_input_calibration_result_plot_has_independent_axis_scales(tmp_path):
    app = _application()
    panel = CalibrationPanel()
    stored = SimpleNamespace(
        run_id=17,
        row_count=8,
        board_type="RF_In",
        database_path=tmp_path / "gain_pwr_calb.db",
        result={
            "frequencies_mhz": [400.0, 500.0],
            "gains": [1000, 2000, 3000],
            "input_power_dbm": [
                [-40.0, -39.0],
                [-30.0, -29.0],
                [-20.0, -19.0],
            ],
            "adc_magnitude_db": [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
            ],
        },
    )

    panel.show_result(stored)
    app.processEvents()

    plot = panel.input_response_plot
    assert panel.tabs.currentIndex() == 1
    assert plot.x_scale_mode == "log"
    assert plot.y_scale_mode == "log"
    np.testing.assert_array_equal(plot.displayed_frequency_indices, [0, 1])
    plot.set_axis_scales("linear", "log")
    assert plot.x_scale_mode == "linear"
    assert plot.y_scale_mode == "log"
    assert panel.settings_dict()["input_plot"] == {
        "x_scale": "linear",
        "y_scale": "log",
    }
    panel.close()


def _create_output_calibration(tmp_path, monkeypatch):
    database_path = tmp_path / "gain_pwr_calb.db"
    monkeypatch.setenv(QCODES_STAGING_ENV, str(tmp_path / "staging"))
    state = {"gain": 0}
    tone_calls = []

    def tone_runner(
        _soc,
        _soccfg,
        output_ch,
        nqz,
        frequency_mhz,
        gain,
        length_cycles,
    ):
        state["gain"] = int(gain)
        tone_calls.append((output_ch, nqz, frequency_mhz, gain, length_cycles))
        return float(frequency_mhz)

    class Meter:
        idn = "AGILENT TECHNOLOGIES,DSO-X 6004A,TEST,1.0"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def measure_power_dbm(self, frequency_mhz):
            response = -10.0 - 0.02 * (float(frequency_mhz) - 400.0)
            return response + 20.0 * np.log10(state["gain"] / MAX_QICK_GAIN)

    config = OutputPowerCalibrationConfig(
        database_path=str(database_path),
        output_board_type="RF_Out",
        frequency_start_mhz=400.0,
        frequency_end_mhz=420.0,
        frequency_points=3,
        gain_start=1000,
        gain_end=10000,
        gain_points=3,
        oscilloscope=OscilloscopeConfig(visa_resource="MOCK"),
    )
    stored = run_output_power_calibration(
        connection_config=QickConnectionConfig(host="127.0.0.1"),
        calibration_config=config,
        connector=lambda **_kwargs: (_FakeSoc(), object()),
        power_meter_factory=lambda _config: Meter(),
        tone_runner=tone_runner,
    )
    assert stored.run_id > 0
    assert stored.row_count == 9
    assert tone_calls[-1][3] == 0
    return database_path, stored


def test_output_and_input_calibration_db_round_trip(tmp_path, monkeypatch):
    database_path, output_stored = _create_output_calibration(tmp_path, monkeypatch)
    frequencies = np.asarray([400.0, 410.0, 420.0])
    catalog = CalibrationDatabase(database_path)
    output_calibration = catalog.output_calibration("RF_Out", frequencies)
    np.testing.assert_allclose(
        output_calibration.output_power_dbm(frequencies, MAX_QICK_GAIN),
        [-10.0, -10.2, -10.4],
        atol=1.0e-9,
    )

    slopes_expected = np.asarray([1.0, 1.1, 1.2])
    intercepts_expected = np.asarray([-60.0, -61.0, -62.0])

    class Program:
        def __init__(self, config):
            self.sweep = config
            self.frequencies_mhz = frequencies.copy()

    def program_factory(_soccfg, config, **_kwargs):
        return Program(config)

    def acquire(_soc, program):
        actual_output = output_calibration.output_power_dbm(
            frequencies,
            program.sweep.gain,
        )
        known_input = actual_output - 3.0
        adc_db = (known_input - intercepts_expected) / slopes_expected
        amplitude = np.power(10.0, adc_db / 20.0)
        iq = np.zeros((frequencies.size, 4, 2), dtype=float)
        iq[:, :, 0] = amplitude[:, np.newaxis]
        return SParameterSweepResult.from_iq(
            frequencies,
            frequencies,
            iq,
        )

    input_config = InputPowerCalibrationConfig(
        database_path=str(database_path),
        output_board_type="RF_Out",
        input_board_type="RF_In",
        frequency_start_mhz=400.0,
        frequency_end_mhz=420.0,
        frequency_points=3,
        gain_start=1000,
        gain_end=10000,
        gain_points=3,
        path_loss_db=3.0,
        settle_seconds=0.0,
    )
    input_stored = run_input_power_calibration(
        connection_config=QickConnectionConfig(host="127.0.0.1"),
        calibration_config=input_config,
        connector=lambda **_kwargs: (_FakeSoc(), object()),
        program_factory=program_factory,
        acquisition_callback=acquire,
    )
    assert input_stored.run_id > output_stored.run_id
    input_calibration = CalibrationDatabase(database_path).input_calibration(
        "RF_In",
        frequencies,
    )
    slopes, intercepts = input_calibration.coefficients(frequencies)
    np.testing.assert_allclose(slopes, slopes_expected, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        intercepts,
        intercepts_expected,
        rtol=0.0,
        atol=1.0e-9,
    )
    assert input_calibration.summary.source_output_run_id == output_stored.run_id
    assert input_calibration.summary.path_loss_db == 3.0

    gains = np.asarray([5000, 6000, 7000])
    desired_adc_db = np.asarray([20.0, 21.0, 22.0])
    iq = np.zeros((3, 2, 2), dtype=float)
    iq[:, :, 0] = np.power(10.0, desired_adc_db / 20.0)[:, np.newaxis]
    raw_result = SParameterSweepResult.from_iq(
        frequencies,
        frequencies,
        iq,
        output_power_dbm=-20.0,
        nominal_gain_code=7000,
        frequency_gain_codes=gains,
    )
    calibrated = apply_power_calibration(
        raw_result,
        output_calibration=output_calibration,
        input_calibration=input_calibration,
        output_att1_db=0.0,
        output_att2_db=0.0,
        input_attenuation_db=0.0,
    )
    expected_input = slopes_expected * desired_adc_db + intercepts_expected
    expected_output = output_calibration.output_power_dbm(frequencies, gains)
    assert calibrated.physical_power_calibrated
    np.testing.assert_allclose(calibrated.adc_magnitude_db, desired_adc_db)
    np.testing.assert_allclose(calibrated.input_powers_dbm, expected_input)
    np.testing.assert_allclose(calibrated.actual_output_powers_dbm, expected_output)
    np.testing.assert_allclose(
        calibrated.magnitude_db,
        expected_input - expected_output,
    )

    deembedded = apply_power_calibration(
        raw_result,
        output_calibration=output_calibration,
        input_calibration=input_calibration,
        output_att1_db=0.0,
        output_att2_db=0.0,
        input_attenuation_db=0.0,
        loss1_db=2.0,
        loss2_db=3.0,
        amplifier_gain_db=10.0,
    )
    np.testing.assert_allclose(
        deembedded.dut_input_powers_dbm,
        expected_output - 2.0,
    )
    np.testing.assert_allclose(
        deembedded.dut_output_powers_dbm,
        expected_input + 3.0 - 10.0,
    )
    np.testing.assert_allclose(
        deembedded.magnitude_db,
        expected_input - expected_output - 5.0,
    )
