"""Noise-analysis numerical and headless GUI tests.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtWidgets

from noise_analysis import (
    DEFAULT_NOISE_ANALYSIS_SETTINGS,
    NoiseAnalysisConfig,
    NoiseAnalysisPanel,
    NoiseTraceCollection,
    analyze_i_trace,
    normalize_noise_analysis_settings,
)


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_voltage_trace_is_divided_by_transimpedance_gain():
    sample_rate = 100_000.0
    sample_count = 10_000
    frequency = 10_000.0
    time = np.arange(sample_count) / sample_rate
    expected_current = 2.5e-9 * np.sin(2.0 * np.pi * frequency * time)
    voltage_trace = expected_current * 1.0e8

    result = analyze_i_trace(
        voltage_trace,
        NoiseAnalysisConfig(
            input_mode="voltage",
            transimpedance_gain_v_per_a=1.0e8,
            sample_rate_hz=sample_rate,
            window="flattop",
        ),
    )

    np.testing.assert_allclose(result.current_a, expected_current, atol=1.0e-20)
    peak_index = int(np.argmax(result.asd_a_per_sqrt_hz[1:]) + 1)
    assert abs(result.frequency_hz[peak_index] - frequency) <= (
        sample_rate / sample_count
    )
    assert result.time_s[-1] == (sample_count - 1) / sample_rate


def test_adc_trace_uses_volts_per_unit_before_gain_conversion():
    adc_trace = np.asarray([0.0, 1000.0, -500.0, 250.0])
    result = analyze_i_trace(
        adc_trace,
        NoiseAnalysisConfig(
            input_mode="adc",
            input_scale_v_per_unit=2.0e-6,
            transimpedance_gain_v_per_a=1.0e6,
            sample_rate_hz=1.0e6,
            window="boxcar",
            detrend="none",
        ),
    )
    np.testing.assert_allclose(
        result.current_a,
        adc_trace * 2.0e-12,
    )


def test_noise_settings_fill_defaults_and_validate_modes():
    settings = normalize_noise_analysis_settings({"run_id": 7})
    assert settings["run_id"] == 7
    assert settings["window"] == "flattop"
    assert settings["transimpedance_gain_v_per_a"] == 1.0e8
    assert set(DEFAULT_NOISE_ANALYSIS_SETTINGS).issubset(settings)

    try:
        normalize_noise_analysis_settings({"input_mode": "magnitude"})
    except ValueError as exc:
        assert "input_mode" in str(exc)
    else:
        raise AssertionError("invalid noise-analysis input mode was accepted")


def test_noise_panel_selects_point_and_repetition_and_round_trips_settings():
    app = _application()
    panel = NoiseAnalysisPanel(default_database_path="noise.db")
    traces = np.arange(2 * 3 * 64, dtype=float).reshape(2, 3, 64)
    panel.set_collection(NoiseTraceCollection(
        i_traces=traces,
        sample_rate_hz=1.0e6,
        unit="V",
        source="test source",
        run_id=4,
        database_path="noise.db",
    ))
    panel.point_index.setValue(1)
    panel.repetition_index.setValue(2)
    panel.transimpedance_gain.setValue(1.0e7)
    panel.window.setCurrentText("hann")
    result = panel.analyze_selected_trace()

    assert result is not None
    np.testing.assert_allclose(
        result.input_trace,
        traces[1, 2],
    )
    settings = panel.settings_dict()
    restored = NoiseAnalysisPanel(default_database_path="other.db")
    restored.load_settings(settings)
    assert restored.settings_dict() == settings
    app.processEvents()
    panel.close()
    restored.close()
