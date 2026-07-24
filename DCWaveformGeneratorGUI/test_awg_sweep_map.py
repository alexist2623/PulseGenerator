"""Tests for the AWG Cartesian two-dimensional result map.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from itertools import product
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5 import QtWidgets
import pytest

import awg_sweep_map as awg_map
import DCWaveform_Generator as gui
from dc_waveform_core import QickSweepSpec


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _axis(output_name, segment_name, points):
    return SimpleNamespace(
        output_name=output_name,
        segment_name=segment_name,
        start=float(points[0]),
        stop=float(points[-1]),
        count=len(points),
    )


def _two_axis_result():
    x_points = (-0.5, 0.5)
    y_points = (-0.25, 0.0, 0.25)
    coordinates = np.asarray(tuple(product(x_points, y_points)), dtype=float)
    iq = np.empty((coordinates.shape[0], 2, 2, 2), dtype=np.int32)
    for point_index, (x_value, y_value) in enumerate(coordinates):
        base_i = int(100 * x_value + 20 * y_value)
        base_q = int(-40 * x_value + 80 * y_value)
        for repetition in range(2):
            for sample in range(2):
                iq[point_index, repetition, sample, 0] = (
                    base_i + repetition + sample
                )
                iq[point_index, repetition, sample, 1] = (
                    base_q - repetition - sample
                )
    return SimpleNamespace(
        sweep_axes=(
            _axis("awg_0", "set_1", x_points),
            _axis("awg_1", "set_2", y_points),
        ),
        sweep_points=coordinates,
        iq=iq,
        sample_rate_hz=50_000.0,
    )


def test_reduce_two_axis_awg_map_and_axis_swap():
    ddr_result = _two_axis_result()
    result = awg_map.reduce_awg_sweep_map(
        ddr_result,
        x_axis_key=("awg_0", "set_1"),
        y_axis_key=("awg_1", "set_2"),
        full_scale_mv=800.0,
    )

    np.testing.assert_allclose(result.x_values_mv, (-400.0, 400.0))
    np.testing.assert_allclose(result.y_values_mv, (-200.0, 0.0, 200.0))
    assert result.i_mean.shape == (3, 2)
    assert result.q_mean.shape == (3, 2)
    assert result.repetition_count == 2
    assert result.samples_per_trace == 2
    assert result.sample_rate_hz == 50_000.0
    assert result.source_points_per_cell == 1
    assert result.averaged_axis_labels == ()

    expected_i = np.asarray(
        (
            (-54.0, 46.0),
            (-49.0, 51.0),
            (-44.0, 56.0),
        )
    )
    expected_q = np.asarray(
        (
            (-1.0, -41.0),
            (19.0, -21.0),
            (39.0, -1.0),
        )
    )
    np.testing.assert_allclose(result.i_mean, expected_i)
    np.testing.assert_allclose(result.q_mean, expected_q)
    np.testing.assert_allclose(
        result.magnitude,
        np.hypot(expected_i, expected_q),
    )
    np.testing.assert_allclose(
        result.angle_deg,
        np.degrees(np.arctan2(expected_q, expected_i)),
    )

    swapped = awg_map.reduce_awg_sweep_map(
        ddr_result,
        x_axis_key=("awg_1", "set_2"),
        y_axis_key=("awg_0", "set_1"),
        full_scale_mv=800.0,
    )
    np.testing.assert_allclose(swapped.i_mean, expected_i.T)
    np.testing.assert_allclose(swapped.q_mean, expected_q.T)


def test_reduce_map_preserves_rf_duration_axis_in_microseconds():
    voltage_points = (-0.5, 0.5)
    duration_points = (1.0, 2.0, 3.0)
    coordinates = np.asarray(
        tuple(product(voltage_points, duration_points)),
        dtype=float,
    )
    iq = np.zeros((coordinates.shape[0], 1, 2, 2), dtype=np.int16)
    for point_index, (voltage, duration_us) in enumerate(coordinates):
        iq[point_index, ..., 0] = int(100 * voltage + duration_us)
        iq[point_index, ..., 1] = int(-100 * voltage + duration_us)
    result = awg_map.reduce_awg_sweep_map(
        SimpleNamespace(
            sweep_axes=(
                _axis("awg_0", "gate", voltage_points),
                SimpleNamespace(
                    output_name="rf_gen_0",
                    segment_name="gate",
                    start=duration_points[0],
                    stop=duration_points[-1],
                    count=len(duration_points),
                    axis_kind="rf_duration",
                    segment_length_mode="fixed",
                ),
            ),
            sweep_points=coordinates,
            iq=iq,
            sample_rate_hz=50_000.0,
        ),
        x_axis_key=("rf_gen_0", "gate"),
        y_axis_key=("awg_0", "gate"),
        full_scale_mv=800.0,
    )

    np.testing.assert_allclose(result.x_values, duration_points)
    np.testing.assert_allclose(result.y_values, (-400.0, 400.0))
    assert result.x_unit == "us"
    assert result.y_unit == "mV"
    assert result.x_axis_label == "rf_gen_0 / gate RF duration"
    assert result.y_axis_label == "awg_0 / gate"


def test_reduce_three_axes_averages_unselected_axis():
    x_points = (-1.0, 1.0)
    y_points = (-0.5, 0.5)
    z_points = (-0.25, 0.25)
    coordinates = np.asarray(
        tuple(product(x_points, y_points, z_points)),
        dtype=float,
    )
    iq = np.empty((coordinates.shape[0], 1, 1, 2), dtype=np.int32)
    for point_index, (x_value, y_value, z_value) in enumerate(coordinates):
        iq[point_index, 0, 0, 0] = int(10 * x_value + 4 * z_value)
        iq[point_index, 0, 0, 1] = int(20 * y_value - 4 * z_value)
    ddr_result = SimpleNamespace(
        sweep_axes=(
            _axis("awg_0", "set_0", x_points),
            _axis("awg_1", "set_0", y_points),
            _axis("awg_2", "set_3", z_points),
        ),
        sweep_points=coordinates,
        iq=iq,
    )

    result = awg_map.reduce_awg_sweep_map(
        ddr_result,
        x_axis_key=("awg_0", "set_0"),
        y_axis_key=("awg_1", "set_0"),
        full_scale_mv=100.0,
    )

    np.testing.assert_allclose(result.i_mean, ((-10.0, 10.0),) * 2)
    np.testing.assert_allclose(
        result.q_mean,
        ((-10.0, -10.0), (10.0, 10.0)),
    )
    assert result.source_points_per_cell == 2
    assert result.averaged_axis_labels == ("awg_2 / set_3",)


def test_reduce_awg_map_rejects_duplicate_or_unknown_axes():
    ddr_result = _two_axis_result()
    with pytest.raises(ValueError, match="different"):
        awg_map.reduce_awg_sweep_map(
            ddr_result,
            x_axis_key=("awg_0", "set_1"),
            y_axis_key=("awg_0", "set_1"),
            full_scale_mv=800.0,
        )
    with pytest.raises(ValueError, match="not present"):
        awg_map.reduce_awg_sweep_map(
            ddr_result,
            x_axis_key=("awg_0", "set_1"),
            y_axis_key=("awg_7", "set_7"),
            full_scale_mv=800.0,
        )


def test_experiment_panel_axis_selection_and_result_plot():
    app = _application()
    window = gui.MainWindow()
    window._sweep_specs = [
        QickSweepSpec("set_1", "awg_0", -0.5, 0.5, 2),
        QickSweepSpec("set_2", "awg_1", -0.25, 0.25, 3),
    ]
    window._refresh_sweep_overlay()
    panel = window._experiment_panel

    assert panel.sweep_map_x.count() == 2
    assert panel.sweep_map_y.count() == 2
    assert panel.selected_sweep_axis_keys() == (
        ("awg_0", "set_1"),
        ("awg_1", "set_2"),
    )
    panel.sweep_map_x.setCurrentIndex(1)
    app.processEvents()
    assert panel.selected_sweep_axis_keys() == (
        ("awg_1", "set_2"),
        ("awg_0", "set_1"),
    )

    ddr_result = _two_axis_result()
    stored = SimpleNamespace(
        run_id=123,
        row_count=12,
        database_path="test.db",
        ddr_result=ddr_result,
        rf_settings={},
    )
    window._on_experiment_finished(stored)
    app.processEvents()
    if hasattr(window._awg_sweep_plot, "_result"):
        plotted = window._awg_sweep_plot._result
        assert plotted is not None
        assert plotted.i_mean.shape == (2, 3)
        assert set(window._awg_sweep_plot.images) == {
            "i",
            "q",
            "magnitude",
            "angle",
        }
    window.close()


def test_awg_map_axis_selection_round_trips_in_settings(tmp_path):
    app = _application()
    window = gui.MainWindow()
    window._add_port()
    window._sweep_specs = [
        QickSweepSpec("set_0", "awg_0", -0.5, 0.5, 2),
        QickSweepSpec("set_0", "awg_1", -0.25, 0.25, 3),
    ]
    window._refresh_sweep_overlay()
    window._experiment_panel.sweep_map_x.setCurrentIndex(1)
    app.processEvents()
    assert window._experiment_panel.selected_sweep_axis_keys() == (
        ("awg_1", "set_0"),
        ("awg_0", "set_0"),
    )

    path = window._save_settings_json(tmp_path / "awg_map_axes.json")
    restored = gui.MainWindow()
    restored._load_settings_json(path)
    app.processEvents()
    assert restored._experiment_panel.selected_sweep_axis_keys() == (
        ("awg_1", "set_0"),
        ("awg_0", "set_0"),
    )

    restored.close()
    window.close()
    restored.deleteLater()
    window.deleteLater()
    app.processEvents()
