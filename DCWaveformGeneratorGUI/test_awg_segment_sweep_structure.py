"""Regression tests for AWG segment edits with configured sweep axes.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

import DCWaveform_Generator as gui
from dc_waveform_core import QickRampRateSweepSpec, QickSweepSpec


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _sweep_targets(window):
    return tuple(
        (spec.output_name, spec.segment_name)
        for spec in window._sweep_specs
    )


def _add_segments(window, count):
    for index in range(count):
        window._add_segment(
            100.0 + index,
            200.0 + index,
            10.0 * (index + 1),
        )


def test_appending_voltage_segments_preserves_existing_sweeps():
    app = _application()
    window = gui.MainWindow()
    window._sweep_specs = [
        QickSweepSpec("set_0", "awg_0", -0.2, 0.2, 3),
    ]
    window._refresh_sweep_overlay(sync_rows=True)

    _add_segments(window, 4)
    app.processEvents()

    assert window._pulse[0].set_count == 5
    assert _sweep_targets(window) == (("awg_0", "set_0"),)
    assert window._multi_ctrl._ctrl_pannels[0]._sweep_rows == {0}
    window.close()


def test_insert_delete_sequence_remaps_only_affected_voltage_sweeps():
    app = _application()
    window = gui.MainWindow()
    _add_segments(window, 3)
    window._add_port()
    window._sweep_specs = [
        QickSweepSpec("set_0", "awg_0", -0.1, 0.1, 2),
        QickSweepSpec("set_1", "awg_0", -0.2, 0.2, 3),
        QickSweepSpec("set_3", "awg_0", -0.3, 0.3, 4),
        QickSweepSpec("set_2", "awg_1", -0.4, 0.4, 5),
        QickRampRateSweepSpec("ramp_0_to_1", 0.1, 0.2, 2),
        QickRampRateSweepSpec("ramp_2_to_3", 0.2, 0.4, 3),
    ]
    window._refresh_sweep_overlay(sync_rows=True)
    panel = window._experiment_panel
    panel.set_sweep_specs(
        window._active_map_sweep_specs(),
        selected_keys=(("awg_0", "set_3"), ("awg_1", "set_2")),
    )

    control_0 = window._multi_ctrl._ctrl_pannels[0]
    assert control_0._edit_segment_structure("insert_above", 2)
    app.processEvents()

    assert _sweep_targets(window) == (
        ("awg_0", "set_0"),
        ("awg_0", "set_1"),
        ("awg_0", "set_4"),
        ("awg_1", "set_2"),
        ("all_awg_outputs", "ramp_0_to_1"),
    )
    assert panel.selected_sweep_axis_keys() == (
        ("awg_0", "set_4"),
        ("awg_1", "set_2"),
    )
    assert control_0._sweep_rows == {0, 1, 4}
    assert window._multi_ctrl._ctrl_pannels[1]._sweep_rows == {2}

    assert control_0._edit_segment_structure("delete", 1)
    app.processEvents()
    assert _sweep_targets(window) == (
        ("awg_0", "set_0"),
        ("awg_0", "set_3"),
        ("awg_1", "set_2"),
    )
    assert panel.selected_sweep_axis_keys() == (
        ("awg_0", "set_3"),
        ("awg_1", "set_2"),
    )
    assert control_0._sweep_rows == {0, 3}
    assert window._multi_ctrl._ctrl_pannels[1]._sweep_rows == {2}

    control_1 = window._multi_ctrl._ctrl_pannels[1]
    assert control_1._edit_segment_structure("insert_below", 0)
    app.processEvents()
    assert _sweep_targets(window) == (
        ("awg_0", "set_0"),
        ("awg_0", "set_3"),
        ("awg_1", "set_3"),
    )
    assert panel.selected_sweep_axis_keys() == (
        ("awg_0", "set_3"),
        ("awg_1", "set_3"),
    )

    assert control_1._edit_segment_structure("delete", 3)
    app.processEvents()
    assert _sweep_targets(window) == (
        ("awg_0", "set_0"),
        ("awg_0", "set_3"),
    )
    assert control_0._sweep_rows == {0, 3}
    assert control_1._sweep_rows == set()
    window.close()


def test_deleting_one_target_keeps_other_sweeps_on_same_port():
    app = _application()
    window = gui.MainWindow()
    _add_segments(window, 4)
    window._sweep_specs = [
        QickSweepSpec(f"set_{index}", "awg_0", -0.1, 0.1, index + 2)
        for index in range(5)
    ]
    window._refresh_sweep_overlay(sync_rows=True)

    control = window._multi_ctrl._ctrl_pannels[0]
    assert control._edit_segment_structure("delete", 2)
    app.processEvents()

    assert _sweep_targets(window) == (
        ("awg_0", "set_0"),
        ("awg_0", "set_1"),
        ("awg_0", "set_2"),
        ("awg_0", "set_3"),
    )
    assert tuple(spec.count for spec in window._sweep_specs) == (2, 3, 5, 6)
    assert control._sweep_rows == {0, 1, 2, 3}
    window.close()
