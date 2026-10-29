"""User-facing segment names must not replace persistent sweep target IDs."""
from copy import deepcopy

from PyQt5 import QtCore, QtTest, QtWidgets

from dc_waveform_core import (
    QickHoldDurationSweepSpec, QickRampRateSweepSpec, QickRfSweepAxisSpec,
    QickSweepSpec,
)
from sweep_grouping import parameter_key
from test_square_awg_exclusion import window
from test_sweep_execution_order import voltage_fixture
from test_sweep_grouping_gui import check_and_group, row_for


def rename(window, row, name, port=0):
    window._multi_ctrl._ctrl_pannels[port].table.item(row, 1).setText(name)
    QtWidgets.QApplication.processEvents()


def target(panel, spec):
    return panel.sweep_parameter_table.item(row_for(panel, parameter_key(spec)), 1).text()


def test_rename_updates_table_selection_and_axes_without_changing_targets(window):
    voltages = tuple(voltage_fixture(window))
    ramp = QickRampRateSweepSpec('ramp_0_to_1', .1, .2, 2)
    hold = QickHoldDurationSweepSpec('set_1', 1., 2., 2)
    window._sweep_specs.extend((ramp, hold))
    window._notify_sweep_state_changed()
    panel = window._experiment_panel
    panel.sweep_parameter_table.selectRow(row_for(panel, parameter_key(voltages[1])))
    original_specs = tuple(window._sweep_specs)
    original_registry = deepcopy(window._sweep_grouping.to_dict())
    original_axes = panel.selected_sweep_axis_keys()
    axes_changed = QtTest.QSignalSpy(panel.sweep_axes_changed)

    rename(window, 0, 'Reset')
    rename(window, 1, 'Readout gate', port=1)

    assert target(panel, voltages[0]) == 'awg_0 / Readout gate'
    assert target(panel, voltages[1]) == 'awg_1 / Readout gate'
    assert target(panel, hold) == 'All AWG outputs / Readout gate'
    assert target(panel, ramp) == 'All AWG outputs / Reset → Readout gate'
    assert panel.sweep_parameter_target.text() == 'awg_1 / Readout gate'
    for combo in (panel.sweep_map_x, panel.sweep_map_y):
        assert all('Readout gate' in combo.itemText(i) for i in range(combo.count()))
    assert panel.selected_sweep_axis_keys() == original_axes
    assert len(axes_changed) == 0
    assert tuple(window._sweep_specs) == original_specs
    assert window._sweep_grouping.to_dict() == original_registry
    assert parameter_key(panel._selected_sweep_parameter()) == parameter_key(voltages[1])


def test_group_order_and_runtime_survive_rename_and_settings_roundtrip(window, tmp_path):
    voltages = tuple(voltage_fixture(window))
    ramp = QickRampRateSweepSpec('ramp_0_to_1', .1, .2, 2)
    window._sweep_specs.append(ramp)
    window._notify_sweep_state_changed()
    check_and_group(window, voltages)
    panel = window._experiment_panel
    panel.sweep_parameter_table.selectRow(row_for(panel, parameter_key(voltages[1])))
    QtTest.QTest.mouseClick(panel.sweep_move_up_button, QtCore.Qt.LeftButton)
    registry = deepcopy(window._sweep_grouping.to_dict())
    runtime = lambda: window._experiment_run_arguments(
        require_readout=False, require_run_config=False)['sequence']
    before = runtime()
    rename(window, 1, '측정 HOLD')
    saved = tmp_path / 'segment_names.json'
    window._save_settings_json(saved)
    rename(window, 1, 'temporary')
    window._load_settings_json(saved)
    after = runtime()
    assert window._sweep_grouping.to_dict() == registry
    assert after.sweep_axes == before.sweep_axes
    assert after.sweep_shape == before.sweep_shape
    assert [after.point_indices(i) for i in range(after.sweep_point_count)] == [
        before.point_indices(i) for i in range(before.sweep_point_count)]
    assert [parameter_key(s) for s in panel._map_sweep_specs] == [
        parameter_key(s) for s in (*voltages, ramp)]
    assert target(panel, voltages[0]) == 'awg_0 / 측정 HOLD'
    assert target(panel, ramp).endswith(' → 측정 HOLD')


def test_names_before_sweep_creation_follow_insert_and_delete(window):
    window._add_segment(1000., 10000., 100.)
    window._add_segment(1000., 10000., 0.)
    rename(window, 1, 'Readout')
    window._sweep_specs = [QickSweepSpec('set_1', 'awg_0', .1, .2, 2)]
    window._notify_sweep_state_changed()
    panel = window._experiment_panel
    original_id = window._sweep_grouping.member_id(window._sweep_specs[0])
    assert target(panel, window._sweep_specs[0]) == 'awg_0 / Readout'
    control = window._multi_ctrl._ctrl_pannels[0]
    assert control._edit_segment_structure('insert_above', 1)
    assert window._sweep_specs[0].segment_name == 'set_2'
    assert target(panel, window._sweep_specs[0]) == 'awg_0 / Readout'
    assert control._edit_segment_structure('delete', 1)
    assert window._sweep_specs[0].segment_name == 'set_1'
    assert target(panel, window._sweep_specs[0]) == 'awg_0 / Readout'
    assert window._sweep_grouping.member_id(window._sweep_specs[0]) == original_id


def test_duplicate_names_do_not_merge_axes_or_redirect_edits(window):
    window._add_segment(1000., 10000., 100.)
    window._add_segment(1000., 10000., 0.)
    window._sweep_specs = [QickSweepSpec(f'set_{i}', 'awg_0', .1, .2, 2) for i in (1, 2)]
    window._notify_sweep_state_changed()
    # A display name that resembles another internal ID must remain literal.
    rename(window, 1, 'set_0')
    rename(window, 2, 'set_0')
    panel = window._experiment_panel
    assert [target(panel, s) for s in window._sweep_specs] == ['awg_0 / set_0'] * 2
    assert len(set(window._sweep_grouping.parameters.values())) == 2
    second = window._sweep_specs[1]
    panel.sweep_parameter_table.item(row_for(panel, parameter_key(second)), 4).setText('3')
    assert [(s.segment_name, s.count) for s in window._sweep_specs] == [('set_1', 2), ('set_2', 3)]


def test_rf_anchor_labels_and_unknown_target_fallback(window):
    window._add_segment(1000., 10000., 100.)
    rename(window, 1, 'RF window')
    panel = window._experiment_panel
    specs = [QickRfSweepAxisSpec('set_1', 'rf0_' + kind, 0, 1., 2., 2, kind, unit)
             for kind, unit in [('rf_duration', 'us'), ('rf_frequency', 'MHz'),
                                ('rf_power', 'dBm'), ('rf_template_n', 'count'),
                                ('rf_template_tau', 'us')]]
    panel.set_sweep_specs(specs)
    for spec in specs:
        assert target(panel, spec) == 'RF gen 0 / RF window'
        assert 'RF window' in panel._sweep_axis_label(spec)
    for name in ('custom_target', 'set_999', 'ramp_0_to_999'):
        assert window._sweep_segment_display_name(name) == name
    panel.sweep_segment_label = None
    assert panel._sweep_parameter_values(specs[0])[1] == 'RF gen 0 / set_1'
