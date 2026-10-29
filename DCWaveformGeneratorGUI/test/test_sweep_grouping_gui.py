"""Sweep membership, editable rows, and source lifecycle regressions."""
from copy import deepcopy
from dataclasses import replace

import pytest
from PyQt5 import QtCore, QtTest, QtWidgets

import DCWaveform_Generator as gui
from dc_waveform_core import QickRampRateSweepSpec, QickSweepSpec
from sweep_grouping import SweepGroupingModel, parameter_key
from test_square_awg_exclusion import window


def fixture_sweeps(window, counts=(3, 5, 7)):
    window._add_segment(1000., 10000., 100.)
    window._add_segment(1000., 10000., 0.)
    window._add_port()
    window._sweep_specs = [
        QickSweepSpec('set_1', 'awg_0', -.1, .2, counts[0]),
        QickSweepSpec('set_1', 'awg_1', .2, -.3, counts[1]),
        QickRampRateSweepSpec('ramp_0_to_1', .1, 1., counts[2]),
    ]
    window._notify_sweep_state_changed()
    return tuple(window._sweep_specs)


def row_for(panel, key):
    return next(row for row in range(panel.sweep_parameter_table.rowCount())
                if tuple(panel.sweep_parameter_table.item(row, 0).data(QtCore.Qt.UserRole)) == tuple(key))


def check_and_group(window, specs):
    panel = window._experiment_panel
    for spec in specs:
        item = panel.sweep_parameter_table.item(row_for(panel, parameter_key(spec)), 0)
        item.setCheckState(QtCore.Qt.Checked)
    assert panel.sweep_group_button.isEnabled()
    QtTest.QTest.mouseClick(panel.sweep_group_button, QtCore.Qt.LeftButton)
    QtWidgets.QApplication.processEvents()


def edit_cell(window, key, column, value):
    table = window._experiment_panel.sweep_parameter_table
    table.item(row_for(window._experiment_panel, key), column).setText(str(value))
    QtWidgets.QApplication.processEvents()


def test_registry_removes_singletons_and_never_reuses_deleted_identity():
    specs = [QickSweepSpec('set_1', f'awg_{i}', -.2, .3, 3) for i in range(3)]
    model = SweepGroupingModel()
    model.synchronize(specs)
    ids = [model.member_id(spec) for spec in specs]
    group = model.create_group(ids, specs)
    model.synchronize(specs[1:])
    assert model.groups[group]['members'] == ids[1:]
    model.synchronize(specs)
    assert model.member_id(specs[0]) != ids[0]
    assert model.group_for(specs[0])[0] is None
    model.remove_member(ids[1])
    assert not model.groups


def test_registry_regroup_remap_and_persistence_preserve_unique_members():
    specs = [QickSweepSpec('set_1', f'awg_{i}', -.2, .3, 3) for i in range(4)]
    model = SweepGroupingModel()
    model.synchronize(specs)
    ids = [model.member_id(spec) for spec in specs]
    model.create_group(ids[:2], specs)
    model.create_group(ids[2:], specs)
    new_group = model.create_group([ids[1], ids[2]], specs)
    assert list(model.groups) == [new_group]
    assert model.group_for(specs[0])[0] is None
    model.remap({parameter_key(specs[1]): ('amplitude', 'awg_1', 'set_2')})
    renamed = replace(specs[1], segment_name='set_2')
    assert model.member_id(renamed) == ids[1]
    loaded = SweepGroupingModel.from_dict(model.to_dict())
    assert loaded.to_dict() == model.to_dict()


def test_group_checkboxes_all_row_editing_and_count_one(window):
    specs = fixture_sweeps(window)
    check_and_group(window, specs[:2])
    panel = window._experiment_panel
    assert [spec.count for spec in window._sweep_specs] == [3, 3, 7]
    assert window._sweep_cartesian_count() == 21
    first = panel.sweep_parameter_table.item(row_for(panel, parameter_key(specs[0])), 5)
    second = panel.sweep_parameter_table.item(row_for(panel, parameter_key(specs[1])), 5)
    assert first.text() == second.text() == 'Group 1'
    assert first.background().color() == second.background().color()
    for index, original in enumerate(specs[:2]):
        edit_cell(window, parameter_key(original), 2, -50. - index * 25)
        edit_cell(window, parameter_key(original), 4, 4 + index)
        assert [spec.count for spec in window._sweep_specs[:2]] == [4 + index] * 2
    scale = window._experiment_panel.full_scale_mv.value()
    assert [spec.start * scale for spec in window._sweep_specs[:2]] == [-50., -75.]
    edit_cell(window, parameter_key(specs[1]), 4, 1)
    assert window._sweep_cartesian_count() == 7
    assert all(panel.sweep_parameter_table.item(row_for(panel, parameter_key(spec)), 6).text().startswith('0 ')
               for spec in specs[:2])


def test_context_ungroup_and_both_sides_count_edits_are_independent(window, monkeypatch):
    specs = fixture_sweeps(window)
    check_and_group(window, specs)
    panel = window._experiment_panel
    table = panel.sweep_parameter_table
    row = row_for(panel, parameter_key(specs[0]))
    item = table.item(row, 5)
    monkeypatch.setattr(QtWidgets.QMenu, 'exec_', lambda menu, *_: next(
        action for action in menu.actions() if action.text() == 'Remove from group'))
    panel._sweep_group_context_menu(table.visualItemRect(item).center())
    edit_cell(window, parameter_key(specs[0]), 4, 9)
    # RAMP is now displayed first to match execution, so its count (7)
    # supplies the group count under the existing first-selected-row rule.
    assert [spec.count for spec in window._sweep_specs] == [9, 7, 7]
    edit_cell(window, parameter_key(specs[1]), 4, 5)
    assert [spec.count for spec in window._sweep_specs] == [9, 5, 5]
    assert window._sweep_cartesian_count() == 45


@pytest.mark.parametrize('deleted', [0, 1, 2])
def test_source_deletion_readd_and_regroup_do_not_restore_old_links(window, deleted):
    specs = fixture_sweeps(window)
    check_and_group(window, specs)
    old_id = window._sweep_grouping.member_id(specs[deleted])
    if deleted < 2:
        window._multi_ctrl._ctrl_pannels[deleted].sweep_remove_requested.emit(deleted, 1)
    else:
        window._experiment_panel.sweep_remove_requested.emit(window._sweep_specs[deleted])
    assert old_id not in window._sweep_grouping.parameters.values()
    replacement = replace(specs[deleted], count=11)
    window._sweep_specs.append(replacement)
    window._notify_sweep_state_changed()
    assert window._sweep_grouping.member_id(replacement) != old_id
    assert window._sweep_grouping.group_for(replacement)[0] is None
    edit_cell(window, parameter_key(replacement), 4, 13)
    assert sorted(spec.count for spec in window._sweep_specs) == [7, 7, 13]
    check_and_group(window, tuple(window._sweep_specs))
    assert len(window._sweep_grouping.groups) == 1


def test_sort_refresh_tabs_save_load_and_deleted_reference_cleanup(window):
    specs = fixture_sweeps(window)
    check_and_group(window, specs[:2])
    original = window._sweep_grouping.to_dict()
    panel = window._experiment_panel
    panel.sweep_parameter_table.sortItems(1, QtCore.Qt.DescendingOrder)
    window._awg_tuning_tabs.setCurrentIndex(0)
    window._notify_sweep_state_changed()
    edit_cell(window, parameter_key(specs[0]), 3, 75.)
    assert window._sweep_grouping.to_dict() == original
    saved = window._settings_to_dict()
    window._apply_decoded_settings(window._decode_settings(saved))
    assert window._sweep_grouping.to_dict() == original
    assert window._sweep_cartesian_count() == 21
    window._remove_sweep_parameter(window._sweep_specs[0])
    window._remove_sweep_parameter(window._sweep_specs[0])
    assert not window._sweep_grouping.groups
    assert panel.sweep_parameter_table.rowCount() == 1
    assert not panel._selected_group_members


def test_group_ids_follow_segment_insert_delete_and_port_remap(window):
    specs = fixture_sweeps(window)
    check_and_group(window, specs[:2])
    ids = [window._sweep_grouping.member_id(spec) for spec in specs[:2]]
    control = window._multi_ctrl._ctrl_pannels[0]
    assert control._edit_segment_structure('insert_above', 1)
    assert [window._sweep_grouping.member_id(spec) for spec in window._sweep_specs[:2]] == ids
    assert all(spec.segment_name == 'set_2' for spec in window._sweep_specs[:2])
    assert control._edit_segment_structure('delete', 1)
    assert [window._sweep_grouping.member_id(spec) for spec in window._sweep_specs[:2]] == ids
    window._delete_port(0)
    remaining = next(spec for spec in window._sweep_specs if isinstance(spec, QickSweepSpec))
    assert remaining.output_name == 'awg_0'
    assert window._sweep_grouping.member_id(remaining) == ids[1]
    assert not window._sweep_grouping.groups


def test_grouped_two_ramps_and_two_voltage_parameters_reach_run_and_preview(window):
    fixture_sweeps(window, (3, 3, 3))
    window._sweep_specs.append(QickRampRateSweepSpec('ramp_1_to_2', .2, 1.4, 3))
    window._notify_sweep_state_changed()
    check_and_group(window, window._sweep_specs[:2])
    check_and_group(window, window._sweep_specs[2:])
    args = window._experiment_run_arguments(require_readout=False, require_run_config=False)
    assert args['sequence'].sweep_shape == (3, 3)
    assert args['sequence'].sweep_point_count == 9
    assert len(args['sequence'].sweep_axes) == 4
    assert window._waveform_preview_sequence.sweep_shape == (3, 3)
    assert window._waveform_point_index.maximum() == 8


def test_rf_duration_frequency_source_count_changes_and_reenable_are_safe(window):
    fixture_sweeps(window)
    rf = window._rf_ports_panel._panels[0]
    with QtCore.QSignalBlocker(rf):
        rf.setChecked(True)
        rf.segment.setCurrentIndex(rf.segment.findData('set_1'))
        rf.duration_sweep_enabled.setChecked(True)
        rf.duration_sweep_start.setValue(gui._time_from_ns(1000., rf._time_unit))
        rf.duration_sweep_stop.setValue(gui._time_from_ns(2000., rf._time_unit))
        rf.duration_sweep_count.setValue(3)
        rf.frequency_sweep_enabled.setChecked(True)
        rf.frequency_sweep_start_mhz.setValue(40.)
        rf.frequency_sweep_stop_mhz.setValue(70.)
        rf.frequency_sweep_count.setValue(5)
    window._rf_ports_panel._emit_specs()
    rf_specs = tuple(window._rf_pulse_specs[0].sweep_axes)
    check_and_group(window, rf_specs)
    assert rf.duration_sweep_count.value() == rf.frequency_sweep_count.value() == 3
    edit_cell(window, parameter_key(rf_specs[1]), 2, 45.)
    assert rf.frequency_sweep_start_mhz.value() == 45.
    assert window._rf_pulse_specs[0].duration_sweep_start_us == 1.
    rf.duration_sweep_count.setValue(7)
    assert rf.frequency_sweep_count.value() == 7
    old_id = window._sweep_grouping.member_id(rf_specs[0])
    rf.duration_sweep_enabled.setChecked(False)
    assert not window._sweep_grouping.groups
    rf.duration_sweep_enabled.setChecked(True)
    new_spec = window._rf_pulse_specs[0].sweep_axes[0]
    assert window._sweep_grouping.member_id(new_spec) != old_id
    rf.duration_sweep_count.setValue(9)
    assert rf.frequency_sweep_count.value() == 7


def test_group_count_edit_from_awg_output_dialog_updates_other_members(window, monkeypatch):
    specs = fixture_sweeps(window)
    check_and_group(window, specs[:2])
    monkeypatch.setattr(gui.SweepSettingsDialog, 'exec_', lambda _dialog: QtWidgets.QDialog.Accepted)
    monkeypatch.setattr(gui.SweepSettingsDialog, 'value', lambda _dialog: replace(specs[0], count=9))
    window._configure_segment_sweep(0, 1)
    assert [spec.count for spec in window._sweep_specs[:2]] == [9, 9]


def test_repeated_group_lifecycle_preserves_one_update_and_no_ghost_selection(window):
    specs = fixture_sweeps(window)
    for count in (4, 6, 8):
        check_and_group(window, window._sweep_specs[:2])
        edit_cell(window, parameter_key(specs[1]), 4, count)
        assert [spec.count for spec in window._sweep_specs[:2]] == [count, count]
        window._ungroup_sweep_parameter(window._sweep_grouping.member_id(specs[0]))
        assert not window._sweep_grouping.groups
        assert not window._experiment_panel._selected_group_members
        edit_cell(window, parameter_key(specs[0]), 4, count + 1)
        assert [spec.count for spec in window._sweep_specs[:2]] == [count + 1, count]


def test_invalid_count_keeps_source_and_group_unchanged(window, monkeypatch):
    specs = fixture_sweeps(window)
    check_and_group(window, specs[:2])
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', lambda *_args: warnings.append(_args[-1]))
    edit_cell(window, parameter_key(specs[1]), 4, '2.5')
    assert warnings and [spec.count for spec in window._sweep_specs[:2]] == [3, 3]
    assert len(window._sweep_grouping.groups) == 1


def test_one_group_is_one_trace_dimension_and_disables_two_dimensional_map(window):
    specs = fixture_sweeps(window, (3, 3, 3))
    check_and_group(window, specs)
    panel = window._experiment_panel
    assert panel.sweep_map_x.count() == panel.sweep_map_y.count() == 1
    assert not panel.sweep_map_x.isEnabled() and not panel.sweep_map_y.isEnabled()
    assert panel.selected_sweep_axis_keys() is None
    args = window._experiment_run_arguments(require_readout=False, require_run_config=False)
    assert args['sequence'].sweep_shape == (3,)
    assert args['sequence'].sweep_point_count == 3
    assert window._waveform_preview_sequence.sweep_shape == (3,)
    assert window._waveform_point_index.maximum() == 2


def test_legacy_versionless_settings_without_groups_keep_independent_axes(window):
    fixture_sweeps(window, (3, 4, 5))
    original_counts = {parameter_key(spec): spec.count for spec in window._sweep_specs}
    legacy = window._settings_to_dict()
    legacy.pop('version')
    legacy.pop('sweep_grouping')
    window._apply_decoded_settings(window._decode_settings(legacy))
    assert {parameter_key(spec): spec.count for spec in window._sweep_specs} == original_counts
    assert not window._sweep_grouping.groups
    assert window._sweep_cartesian_count() == 60
    assert len(set(window._sweep_grouping.parameters.values())) == 3


def test_loading_stale_group_members_prunes_deleted_sources_and_singletons(window):
    specs = fixture_sweeps(window, (3, 3, 3))
    check_and_group(window, specs)
    saved = window._settings_to_dict()
    remaining = deepcopy(saved['awg']['sweeps'][0])
    saved['awg']['sweeps'] = [remaining]
    saved['sweep_grouping']['groups'][0]['members'].append('already-deleted-member')
    window._apply_decoded_settings(window._decode_settings(saved))
    assert len(window._sweep_specs) == len(window._sweep_grouping.parameters) == 1
    assert not window._sweep_grouping.groups
    assert window._sweep_cartesian_count() == 3


@pytest.mark.parametrize('invalid', ['zero_count', 'duplicate_member_identity', 'overlapping_groups'])
def test_ambiguous_group_json_is_rejected_without_changing_current_state(window, invalid):
    specs = fixture_sweeps(window, (3, 3, 3))
    check_and_group(window, specs)
    before = window._settings_to_dict()
    bad = deepcopy(before)
    registry = bad['sweep_grouping']
    if invalid == 'zero_count':
        registry['groups'][0]['count'] = 0
    elif invalid == 'duplicate_member_identity':
        registry['parameters'][1]['parameter_id'] = registry['parameters'][0]['parameter_id']
    else:
        duplicate = deepcopy(registry['groups'][0])
        duplicate['group_id'] = 'another-group'
        registry['groups'].append(duplicate)
    with pytest.raises(ValueError):
        window._decode_settings(bad)
    assert window._settings_to_dict() == before


def test_sweep_columns_keep_readable_width_and_duration_step_in_narrow_panel(window):
    specs = fixture_sweeps(window, (100, 100, 100))
    table = window._experiment_panel.sweep_parameter_table
    table.resize(468, 180)
    table.show()
    QtWidgets.QApplication.processEvents()
    table.resizeColumnsToContents()
    assert table.columnWidth(1) >= 180
    assert table.columnWidth(1) >= max(table.fontMetrics().horizontalAdvance(table.item(row, 1).text())
                                      for row in range(table.rowCount()))
    assert table.columnWidth(2) >= 95 and table.columnWidth(3) >= 95
    assert table.columnWidth(6) >= 120
    assert table.horizontalHeader().sectionResizeMode(1) == QtWidgets.QHeaderView.ResizeToContents
    ramp_row = row_for(window._experiment_panel, parameter_key(specs[2]))
    assert table.item(ramp_row, 6).text() == '10 ns (3 clock(s))'
    assert table.item(ramp_row, 7).text() == '1.09 us'
    assert 'requested end 1 us' in table.item(ramp_row, 6).toolTip()
