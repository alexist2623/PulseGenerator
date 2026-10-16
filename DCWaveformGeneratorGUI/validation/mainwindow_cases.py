"""Actual MainWindow controls for A/B waveform and grouped-sweep cases.

The caller owns a dedicated visible MainWindow and an explicit verified fixture.
This module supplies lazy QtAction sequences and read-only snapshots. It never
clicks Run, connects to hardware, changes firmware, or writes a GUI model field.
The normal GuiHardwareRunner owns the production Run and worker lifecycle.
"""
from copy import deepcopy
import math
import sys

from PyQt5 import QtCore, QtWidgets

from .awg_reference import ReferenceUnavailable
from .qt_case_controls import reveal
from .qt_runner import QtAction


_KINDS = {'amplitude': 'Voltage sweep', 'hold_duration': 'Hold time sweep',
          'ramp_duration': 'RAMP rate sweep'}


def _button(parent, text):
    matches = [w for w in parent.findChildren(QtWidgets.QAbstractButton) if w.text() == text]
    if len(matches) != 1:
        raise ValueError(f'Expected one production button {text!r}, found {len(matches)}')
    reveal(matches[0])
    return matches[0]


def _text(widget, value):
    reveal(widget)
    expected = None
    if isinstance(widget, QtWidgets.QAbstractSpinBox):
        if not widget.minimum() <= value <= widget.maximum():
            raise ValueError('Requested input exceeds the visible control range')
        expected = round(value, widget.decimals()) if isinstance(widget, QtWidgets.QDoubleSpinBox) else value
    return QtAction(widget, 'text', format(value, '.12g') if isinstance(value, (int, float)) else str(value), expected)


def _axis_key(axis):
    kind, segment = axis['kind'], int(axis['segment'])
    return (kind, f'awg_{int(axis.get("output", 0))}' if kind == 'amplitude' else 'all_awg_outputs',
            f'ramp_{segment-1}_to_{segment}' if kind == 'ramp_duration' else f'set_{segment}')


def validate_case(case, fixture):
    """Validate declarative requested values, without touching the GUI."""
    fixture.validate()
    value = deepcopy(case)
    if len(value.get('outputs', ())) != 2:
        raise ValueError('Declare exactly two measured outputs A and B')
    for index, output in enumerate(value['outputs']):
        channel = str(output['scope_channel'])
        fixture_output = fixture.channels.get(channel, fixture.channels.get(int(channel)))
        if fixture_output is None or any(output[key] != fixture_output[key]
                for key in ('generator', 'physical_connector')):
            raise ValueError(f'Output {index} differs from the explicit fixture wiring')
        if not math.isfinite(output['full_scale_mv']) or output['full_scale_mv'] <= 0:
            raise ValueError('Declare a positive measured output full scale')
    if {str(o['scope_channel']) for o in value['outputs']} != {'1', '2'}:
        raise ValueError('Use distinct scope channels 1 and 2')
    stages = value.get('stages', ())
    if len(stages) != 3:
        raise ValueError('The basic A/B adapter uses exactly three SET stages with incoming RAMPs')
    for index, stage in enumerate(stages):
        if len(stage['voltage_mv']) != 2 or not all(math.isfinite(v) for v in stage['voltage_mv']):
            raise ValueError('Every stage needs two finite virtual voltages in mV')
        if (not all(math.isfinite(stage[key]) for key in ('hold_us', 'ramp_us')) or
                stage['hold_us'] <= 0 or (index == 0 and stage['ramp_us'] != 0)):
            raise ValueError('Invalid stage durations')
        if index > 0 and stage['ramp_us'] <= 0:
            raise ReferenceUnavailable(
                'The current GUI requires positive incoming RAMP durations; '
                'zero/negative RAMP SET-only stages are unavailable')
        stage.setdefault('name', f'set_{index}')
    matrix = value.get('matrix', ((1., 0.), (0., 1.)))
    if len(matrix) != 2 or any(len(row) != 2 or not all(math.isfinite(v) for v in row) for row in matrix):
        raise ValueError('Declare a finite 2 by 2 virtual-to-physical matrix')
    value['matrix'] = [list(row) for row in matrix]
    value.setdefault('axes', [])
    keys, counts = set(), {}
    for axis in value['axes']:
        if axis['kind'] not in _KINDS or int(axis.get('output', 0)) not in (0, 1):
            raise ValueError('Unsupported basic A/B sweep member')
        if not 0 <= int(axis['segment']) < 3 or (axis['kind'] == 'ramp_duration' and not axis['segment']):
            raise ValueError('Sweep references a missing incoming RAMP or SET')
        if type(axis['count']) is not int or not 1 <= axis['count'] <= 1_000_000:
            raise ValueError('Point count must be an integer from 1 through 1000000')
        if not all(math.isfinite(axis[v]) for v in ('start', 'stop')):
            raise ValueError('Sweep endpoints must be finite')
        if axis['kind'] != 'amplitude' and min(axis['start'], axis['stop']) <= 0:
            raise ValueError('Duration endpoints must be positive')
        key = _axis_key(axis)
        if key in keys:
            raise ValueError('Duplicate parameter; duration members already affect all AWG outputs')
        keys.add(key)
        if axis.get('group'):
            group = axis['group']
            if group in counts and counts[group] != axis['count']:
                raise ValueError('Grouped case members must declare the same count')
            counts[group] = axis['count']
    for group in counts:
        if sum(a.get('group') == group for a in value['axes']) < 2:
            raise ValueError('A group requires at least two members')
    # The production GUI places duration members outside voltage axes. Retain
    # order within each type and give the independent reference this same spec.
    value['axes'] = sorted(value['axes'], key=lambda a: ('ramp_duration', 'hold_duration', 'amplitude').index(a['kind']))
    value.setdefault('repetitions', 1)
    if type(value['repetitions']) is not int or value['repetitions'] < 1:
        raise ValueError('Repetitions must be a positive integer')
    value.setdefault('compensation', {})
    compensation = value['compensation']
    for key, default in dict(dc_enabled=False, rc_enabled=False, dc_mode='fixed_voltage',
                             dc_voltage_mv=100., dc_duration_us=10., rc_tau_us=30.).items():
        compensation.setdefault(key, default)
    if compensation['dc_mode'] not in ('fixed_voltage', 'fixed_time'):
        raise ValueError('Unknown DC compensation mode')
    return value


class MainWindowCaseBinding:
    """Bindings to production controls, resolved after every table rebuild.

    ``actions()`` is intentionally lazy: feed the iterable directly to
    ``apply_actions`` or ``GuiHardwareRunner.start``; do not materialize it.
    The dedicated window may initially contain one output/one stage, which is
    expanded using Add Port/Add segment. More than two outputs or three stages
    are refused so an unrelated user's waveform cannot be silently deleted.
    The generator-to-connector configuration must already have been identified;
    reading it is not proof of a physical cable, which remains fixture evidence.
    """
    def __init__(self, window, fixture, case):
        self.window, self.fixture = window, fixture
        self.case = validate_case(case, fixture)

    def _idle(self):
        if getattr(self.window, '_experiment_thread', None) is not None:
            raise ValueError('A production experiment is active')
        if not self.window.isVisible():
            raise ValueError('Use a dedicated visible MainWindow')

    def _require_rf_off(self):
        if any(panel.isChecked() for panel in self.window._rf_ports_panel._panels):
            raise ReferenceUnavailable('AWG-only references cannot certify active RF outputs')

    def _mapping(self):
        configured = self.window._qick_configuration
        if configured is None:
            raise ValueError('Identify the loaded configuration before preparing a physical case')
        for output in self.case['outputs']:
            matches = [p for p in configured.outputs if output['generator'] in p.qick_channels]
            if len(matches) != 1 or matches[0].label != output['physical_connector']:
                raise ValueError('Generator/connector mapping differs from the identified configuration')
            actual_scale = self.window._dac_current_state.scale(output['generator'])
            if not math.isclose(actual_scale, output['full_scale_mv'], rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError('Declared full scale differs from the identified GUI DAC current')
        actual = tuple(self.window._qick_awg_channels)
        expected = tuple(o['generator'] for o in self.case['outputs'])
        if actual != expected:
            raise ValueError(f'Select explicit A/B AWG outputs in the Front Panel first: {actual} != {expected}')

    def _tab(self, target):
        yield QtAction(self.window._control_tabs, 'tab', self.window._control_tabs.indexOf(self.window._awg_tuning_page))
        yield QtAction(self.window._awg_tuning_tabs, 'tab', self.window._awg_tuning_tabs.indexOf(target))

    def _row(self, axis):
        table = self.window._experiment_panel.sweep_parameter_table
        wanted = _axis_key(axis)
        for row in range(table.rowCount()):
            if tuple(table.item(row, 0).data(QtCore.Qt.UserRole)) == wanted:
                return row
        raise ValueError(f'GUI sweep row missing: {wanted}')

    def add_axis_actions(self, axis):
        yield from self._tab(self.window._multi_ctrl)
        table = self.window._multi_ctrl._ctrl_pannels[int(axis.get('output', 0))].table
        def dialog_actions(dialog):
            yield _text(dialog.start, axis['start'])
            yield _text(dialog.stop, axis['stop'])
            yield _text(dialog.count, max(dialog.count.minimum(), axis['count']))
            yield QtAction(dialog.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.Ok), 'click')
        yield QtAction(table, 'context_menu', dict(row=int(axis['segment']), column=1,
            path=(_KINDS[axis['kind']], 'Configure sweep...'), dialog_actions=dialog_actions))
        if axis['count'] == 1:
            yield from self.edit_axis_actions(axis, count=1)

    def edit_axis_actions(self, axis, *, start=None, stop=None, count=None):
        yield from self._tab(self.window._experiment_panel)
        table = self.window._experiment_panel.sweep_parameter_table
        for column, value in ((2, start), (3, stop), (4, count)):
            if value is not None:
                yield QtAction(table, 'table_text', dict(row=self._row(axis), column=column,
                    text=format(value, '.12g'), expected=value))

    def group_actions(self, members):
        yield from self._tab(self.window._experiment_panel)
        panel = self.window._experiment_panel
        table = panel.sweep_parameter_table
        # Selection is separate from membership; first clear any old selection.
        for row in range(table.rowCount()):
            if table.item(row, 0).checkState() == QtCore.Qt.Checked:
                yield QtAction(table, 'table_check', dict(row=row, checked=False))
        for member in members:
            yield QtAction(table, 'table_check', dict(row=self._row(member), checked=True))
        yield QtAction(panel.sweep_group_button, 'click')

    def remove_group_actions(self, member):
        yield from self._tab(self.window._experiment_panel)
        yield QtAction(self.window._experiment_panel.sweep_parameter_table, 'context_menu',
            dict(row=self._row(member), column=5, path=('Remove from group',)))

    def delete_axis_actions(self, member, *, source='experiment'):
        if source == 'experiment':
            yield from self._tab(self.window._experiment_panel)
            panel = self.window._experiment_panel
            yield QtAction(panel.sweep_parameter_table, 'table_select', dict(row=self._row(member)))
            reveal(panel.sweep_parameter_remove)
            yield QtAction(panel.sweep_parameter_remove, 'click')
        elif source == 'awg_output':
            yield from self._tab(self.window._multi_ctrl)
            table = self.window._multi_ctrl._ctrl_pannels[int(member.get('output', 0))].table
            yield QtAction(table, 'context_menu', dict(row=int(member['segment']), column=1,
                path=(_KINDS[member['kind']], 'Remove sweep')))
        else:
            raise ValueError('Select experiment or awg_output deletion')

    def matrix_actions(self, matrix):
        def dialog_actions(dialog):
            for row in range(2):
                for column in range(2):
                    yield _text(dialog._editors[row][column], matrix[row][column])
            yield QtAction(dialog.findChild(QtWidgets.QDialogButtonBox).button(QtWidgets.QDialogButtonBox.Ok), 'click')
        yield QtAction(self.window.menuBar(), 'menu_action', dict(
            path=('Pulse', 'Virtual-to-physical cross-capacitance matrix...'), dialog_actions=dialog_actions))

    def settings_actions(self, path, *, load=False):
        """Actual file shortcut/dialog inputs; Windows offscreen is unavailable.

        Static QFileDialog reload reproducibly access-violated in that backend.
        Refuse it before opening a dialog; JSON persistence tests are separate
        and must not be reported as successful real-dialog coverage.
        """
        if sys.platform == 'win32' and QtWidgets.QApplication.platformName() == 'offscreen':
            raise RuntimeError('Real QFileDialog settings roundtrip is unavailable on Windows offscreen')
        from pathlib import Path
        from .archive import private_evidence_root
        selected = private_evidence_root(Path(path).parent)/Path(path).name
        if not load and selected.exists():
            raise ValueError('Do not overwrite settings evidence; select a new path')
        def dialog_actions(dialog):
            filename = dialog.findChild(QtWidgets.QLineEdit, 'fileNameEdit')
            if filename is None:
                raise ValueError('Native file dialog is not a Qt control; use Qt file dialogs for this runner')
            yield _text(filename, str(selected))
            yield QtAction(dialog, 'dialog_button')
        yield QtAction(self.window, 'shortcut_modal', dict(key=QtCore.Qt.Key_O if load else QtCore.Qt.Key_S,
            modifiers=QtCore.Qt.ControlModifier, dialog_actions=dialog_actions))

    def actions(self):
        self._idle()
        w, case = self.window, self.case
        if len(w._pulse) > 2 or any(len(p.flat_segments()) > 3 for p in w._pulse):
            raise ValueError('Use a dedicated empty/basic two-output window; unrelated waveforms will not be deleted')
        self._require_rf_off()
        yield from self._tab(w._multi_ctrl)
        yield QtAction(w._time_unit_combo, 'combo', w._time_unit_combo.findText('us'))
        if len(w._pulse) == 1:
            yield QtAction(_button(w._multi_ctrl, 'Add Port'), 'click')
        self._mapping()
        # Clear only sweeps through the visible menu; base waveform is retained.
        yield QtAction(w.menuBar(), 'menu_action', dict(path=('Sweep', 'Clear all sweeps')))
        control = w._multi_ctrl._ctrl_pannels[0]
        while control.table.rowCount() < 3:
            previous_count = control.table.rowCount()
            stage = case['stages'][previous_count]
            yield _text(control.edit_ramp, stage['ramp_us'])
            yield _text(control.edit_flat, stage['hold_us'])
            yield _text(control.edit_v, stage['voltage_mv'][0])
            yield QtAction(_button(control, 'Add segment'), 'click')
            if control.table.rowCount() != previous_count + 1:
                raise ValueError('Add segment did not append exactly one stage; preparation stopped')
        for row, stage in enumerate(case['stages']):
            for column, value in ((1, stage['name']), (2, stage['ramp_us']), (3, stage['hold_us'])):
                if row == 0 and column == 2:
                    continue
                yield QtAction(control.table, 'table_text', dict(row=row, column=column,
                    text=str(value), expected=value))
            for output in range(2):
                table = w._multi_ctrl._ctrl_pannels[output].table
                yield QtAction(table, 'table_text', dict(row=row, column=4,
                    text=str(stage['voltage_mv'][output]), expected=stage['voltage_mv'][output]))
        yield from self.matrix_actions(case['matrix'])
        for axis in case['axes']:
            yield from self.add_axis_actions(axis)
        for group in dict.fromkeys(a['group'] for a in case['axes'] if a.get('group')):
            yield from self.group_actions([a for a in case['axes'] if a.get('group') == group])
        yield from self._tab(w._experiment_panel)
        panel, comp = w._experiment_panel, case['compensation']
        yield QtAction(panel.measurement_tabs, 'tab', 0)
        yield _text(panel.repetitions, case['repetitions'])
        yield QtAction(panel.iq_storage_mode, 'combo', panel.iq_storage_mode.findData('full_traces'))
        for checkbox, enabled in ((panel.bias_t_type.dc_checkbox, comp['dc_enabled']),
                                  (panel.bias_t_type.rc_checkbox, comp['rc_enabled'])):
            reveal(checkbox)
            yield QtAction(checkbox, 'check', enabled)
        if comp['dc_enabled']:
            reveal(panel.bias_t_mode)
            yield QtAction(panel.bias_t_mode, 'combo', panel.bias_t_mode.findData(comp['dc_mode']))
            if comp['dc_mode'] == 'fixed_voltage':
                yield _text(panel.bias_t_compensation_mv, comp['dc_voltage_mv'])
            else:
                yield _text(panel.bias_t_duration_us, comp['dc_duration_us'])
        if comp['rc_enabled']:
            yield _text(panel.bias_t_filter_tau_us, comp['rc_tau_us'])
        self._mapping()
        self.verify_requested_case()

    def verify_requested_case(self):
        """Check committed values, including precision hidden by table formatting."""
        actual = self.snapshot()
        if len(actual['stages']) != len(self.case['stages']):
            raise ValueError('GUI stage count differs from the declared case')
        def equal(found, wanted, label):
            if not math.isclose(float(found), float(wanted), rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError(f'GUI committed {label} differs: {found!r} != {wanted!r}')
        for index, (found, wanted) in enumerate(zip(actual['stages'], self.case['stages'])):
            if found['name'] != wanted['name']:
                raise ValueError('GUI stage name did not commit')
            for key in ('ramp_us', 'hold_us'):
                equal(found[key], wanted[key], f'stage {index} {key}')
            for output in range(2):
                equal(found['voltage_mv'][output], wanted['voltage_mv'][output], 'virtual voltage')
        for row in range(2):
            for column in range(2):
                equal(actual['matrix'][row][column], self.case['matrix'][row][column], 'matrix')
        wanted_axes = {_axis_key(a): a for a in self.case['axes']}
        found_axes = {_axis_key(a): a for a in actual['axes']}
        if set(found_axes) != set(wanted_axes):
            raise ValueError('GUI sweep members differ from the declared case')
        for key, wanted in wanted_axes.items():
            found = found_axes[key]
            for field in ('start', 'stop', 'count'):
                equal(found[field], wanted[field], field)
            for other, other_wanted in wanted_axes.items():
                expected_same = bool(wanted.get('group')) and wanted.get('group') == other_wanted.get('group')
                actual_same = bool(found.get('group')) and found.get('group') == found_axes[other].get('group')
                if expected_same != actual_same:
                    raise ValueError('GUI group membership differs from the declared case')
        if actual['repetitions'] != self.case['repetitions']:
            raise ValueError('Repetition count did not commit')
        found_comp, wanted_comp = actual['compensation'], self.case['compensation']
        for flag in ('dc_enabled', 'rc_enabled'):
            if found_comp[flag] != wanted_comp[flag]:
                raise ValueError(f'GUI compensation flag {flag} did not commit')
        if wanted_comp['dc_enabled']:
            if found_comp['dc_mode'] != wanted_comp['dc_mode']:
                raise ValueError('GUI DC compensation mode did not commit')
            key = 'dc_voltage_mv' if wanted_comp['dc_mode'] == 'fixed_voltage' else 'dc_duration_us'
            equal(found_comp[key], wanted_comp[key], key)
        if wanted_comp['rc_enabled']:
            equal(found_comp['rc_tau_us'], wanted_comp['rc_tau_us'], 'RC tau')
        return actual

    def snapshot(self):
        """Read AWG-only values; active RF needs a separate combined reference."""
        self._idle()
        self._require_rf_off()
        self._mapping()
        w = self.window
        stages = []
        for index, (start, end) in enumerate(w._pulse[0].flat_segments()):
            pulse = w._pulse[0]
            stages.append(dict(name=pulse.segment_name(index),
                ramp_us=0. if index == 0 else float(pulse.t[start]-pulse.t[start-1])/1000.,
                hold_us=float(pulse.t[end]-pulse.t[start])/1000.,
                voltage_mv=[float(p.v[start]) for p in w._pulse]))
        panel = w._experiment_panel
        members = []
        group_labels = {}
        for spec in w._sweep_specs:
            kind = getattr(spec, 'axis_kind', 'amplitude')
            if kind not in _KINDS:
                raise ValueError('Unexpected non-AWG sweep in the basic case')
            group = w._sweep_grouping.group_for(spec)[0]
            if group is not None:
                group_labels.setdefault(group, f'group_{len(group_labels)}')
            segment = int(spec.segment_name.split('_')[-1])
            scale = w._qick_full_scale_mv if kind == 'amplitude' else 1.
            members.append(dict(kind=kind, output=int(spec.output_name.split('_')[-1]) if kind == 'amplitude' else 0,
                segment=segment, start=float(spec.start)*scale, stop=float(spec.stop)*scale,
                count=int(spec.count), group=group_labels.get(group)))
        return dict(schema='qick-basic-ab-case-v1', stages=stages, axes=members,
            matrix=w._cross_capacitance.tolist(), outputs=deepcopy(self.case['outputs']),
            repetitions=panel.repetitions.value(), point_count=w._sweep_cartesian_count(),
            compensation=dict(dc_enabled=panel.bias_t_type.dc_checkbox.isChecked(),
                rc_enabled=panel.bias_t_type.rc_checkbox.isChecked(), dc_mode=panel.bias_t_mode.currentData(),
                dc_voltage_mv=panel.bias_t_compensation_mv.value(), dc_duration_us=panel.bias_t_duration_us.value(),
                rc_tau_us=panel.bias_t_filter_tau_us.value()),
            value_domain='GUI requested virtual mV and us; no physical/quantized measurement claim')
