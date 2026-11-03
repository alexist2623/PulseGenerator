"""Select preview coordinates by parameter values, without compiling or I/O."""
from dataclasses import dataclass
from math import isfinite, prod

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class ParameterChoice:
    key: tuple
    label: str
    unit: str
    values: tuple
    dimension: int
    group: str = ''
    group_id: str = ''


def format_value(value, unit=''):
    return f'{float(value):.12g}' + (f' {unit}' if unit else '')


def nearest_index(values, requested, current=0):
    """Preserve the selected coordinate when equal/duplicate values tie."""
    if not isfinite(requested):
        raise ValueError('Enter a finite numeric value')
    values = np.asarray(values, dtype=float)
    distances = np.abs(values - requested)
    tolerance = np.finfo(float).eps * max(abs(requested), float(np.max(np.abs(values))), np.finfo(float).tiny) * 4
    candidates = np.flatnonzero(distances <= float(np.min(distances)) + tolerance)
    return int(min(candidates, key=lambda i: (abs(int(i)-current), int(i))))


def coordinates_at(point, shape):
    coordinates = []
    for count in reversed(shape):
        point, index = divmod(point, count)
        coordinates.append(index)
    return tuple(reversed(coordinates))


def point_at(coordinates, shape):
    point = 0
    for index, count in zip(coordinates, shape):
        point = point * count + index
    return point


class SweepValuesModel(QtCore.QAbstractListModel):
    """Format popup rows on demand; never populate the Cartesian product."""
    def __init__(self, values, unit, parent=None):
        super().__init__(parent)
        self.values, self.unit = values, unit

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.values)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid() and 0 <= index.row() < len(self.values):
            if role == QtCore.Qt.DisplayRole:
                return f'{index.row()}: {format_value(self.values[index.row()], self.unit)}'
            if role == QtCore.Qt.UserRole:
                return self.values[index.row()]
        return None


class SweepValueSpinBox(QtWidgets.QAbstractSpinBox):
    """Numeric entry snaps to a real point; arrows visit adjacent values."""
    indexChanged = QtCore.pyqtSignal(int)

    def __init__(self, values, unit='', parent=None):
        super().__init__(parent)
        self.values, self.unit = values, unit
        self.index = 0
        self.direction = -1 if values[-1] < values[0] else 1
        self.number_validator = QtGui.QDoubleValidator(self)
        self.number_validator.setLocale(QtCore.QLocale.c())
        self.number_validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        self.setKeyboardTracking(False)
        self.setMinimumWidth(155)
        self.setToolTip('Enter a value and press Enter to select the nearest sweep value. '
                        'Up/down selects the next higher/lower value.')
        self.editingFinished.connect(self.commit_value)
        self.set_index(0)

    def _number_text(self, text):
        text = text.strip()
        if self.unit and text.endswith(self.unit):
            text = text[:-len(self.unit)].strip()
        return text

    def validate(self, text, position):
        # Qt may call validate while the base class is still being created.
        if not hasattr(self, 'number_validator'):
            return QtGui.QValidator.Intermediate, text, position
        raw = self._number_text(text)
        state, _, _ = self.number_validator.validate(raw, min(position, len(raw)))
        return state, text, position

    def fixup(self, text):
        try:
            index = nearest_index(self.values, float(self._number_text(text)), self.index)
        except ValueError:
            index = self.index
        return format_value(self.values[index], self.unit)

    def set_index(self, index, *, notify=False):
        index = max(0, min(int(index), len(self.values)-1))
        changed = index != self.index
        self.index = index
        self.lineEdit().setText(format_value(self.values[index], self.unit))
        self.update()
        if changed and notify:
            self.indexChanged.emit(index)

    def commit_value(self):
        try:
            index = nearest_index(self.values, float(self._number_text(self.lineEdit().text())), self.index)
        except ValueError:
            index = self.index
        self.set_index(index, notify=True)

    def stepBy(self, steps):
        self.commit_value()
        self.set_index(self.index + int(steps) * self.direction, notify=True)

    def stepEnabled(self):
        if not self.isEnabled():
            return QtWidgets.QAbstractSpinBox.StepNone
        result = QtWidgets.QAbstractSpinBox.StepNone
        for step, flag in ((1, self.StepUpEnabled), (-1, self.StepDownEnabled)):
            if 0 <= self.index + step * self.direction < len(self.values):
                result |= flag
        return result


class SweepPointDialog(QtWidgets.QDialog):
    pointSelected = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select index — sweep parameters')
        self.resize(880, 360)
        self.setModal(False)
        self.shape, self.choices, self.rows = (), (), {}
        self.point = 0
        self.maximum_point = 0
        self._signature = None
        self._generation = 0
        layout = QtWidgets.QVBoxLayout(self)
        help_text = QtWidgets.QLabel('Use the arrows or the list to choose a sweep value. '
                                    'Typed values snap to the nearest point. Grouped parameters move together.')
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(('Parameter', 'Value', 'Sweep values (index: value)', 'Group'))
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        layout.addWidget(self.table)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.button(QtWidgets.QDialogButtonBox.Close).setAutoDefault(False)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    def set_context(self, sequence, bindings, full_scale_mv, describe, group_names, point, maximum_point):
        """Use the same host-outer and grouped hardware dimensions as Run/Plot."""
        representatives = sequence.loop_axis_indices
        dimensions = {rep: len(bindings)+i for i, rep in enumerate(representatives)}
        hardware_groups = {int(member): group for group in sequence.sweep_groups
                           for member in group['member_indices'] if len(group['member_indices']) > 1}
        specs = []
        for i, (owner, axis) in enumerate(bindings):
            specs.append((('host', owner, axis.axis_kind, axis.output_name, axis.segment_name), axis, i, '', ''))
        for i, axis in enumerate(sequence.sweep_axes):
            group = hardware_groups.get(i, {})
            gid = group.get('group_id') or ''
            specs.append((('hardware', axis.axis_kind, axis.output_name, axis.segment_name), axis,
                          dimensions[sequence.axis_representative(i)], group_names.get(gid, 'Group') if gid else '', gid))
        descriptions = [describe(axis) for _, axis, *_ in specs]
        shape = tuple(axis.count for _, axis in bindings) + tuple(sequence.sweep_shape)
        signature = (float(full_scale_mv), shape, tuple((key, repr(axis), dimension, name, gid, label)
                     for (key, axis, dimension, name, gid), label in zip(specs, descriptions)))
        if signature != self._signature:
            choices = []
            for (key, axis, dimension, group, gid), (kind, target, _start, _stop, unit) in zip(specs, descriptions):
                scale = full_scale_mv if axis.axis_kind == 'amplitude' else 1.
                values = tuple(float(v) * scale for v in axis.points)
                # A common unit per row keeps typed values unambiguous.
                if unit == 'us' and max(map(abs, values)) < 1.:
                    values, unit = tuple(v * 1000 for v in values), 'ns'
                label = f'{target} / {kind}'
                parameter = str(getattr(axis, 'parameter_name', '')).strip()
                if parameter and parameter not in label:
                    label += f' / {parameter}'
                choices.append(ParameterChoice(key, label, unit, values, dimension, group, gid))
            self.set_choices(choices, shape, maximum_point)
            self._signature = signature
        self.maximum_point = int(maximum_point)
        self.set_point(point)

    def set_choices(self, choices, shape, maximum_point=None):
        self._generation += 1
        generation = self._generation
        self._signature = None
        self.choices, self.shape = tuple(choices), tuple(shape)
        self.maximum_point = min(prod(self.shape)-1, 2_147_483_647) if maximum_point is None else int(maximum_point)
        self.rows = {}
        self.table.clearContents()
        self.table.setRowCount(len(self.choices))
        palette = ('#eaf2fc', '#edf6ed', '#fff3df', '#f3edfa', '#e7f5f4')
        colors = {}
        for row, choice in enumerate(self.choices):
            color = colors.setdefault(choice.group_id, palette[len(colors) % len(palette)]) if choice.group_id else None
            for column, text in ((0, choice.label), (3, choice.group or 'Independent')):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(choice.label + (f' — {choice.group}: all members share one index.' if choice.group else ''))
                if color:
                    item.setBackground(QtGui.QColor(color))
                self.table.setItem(row, column, item)
            spin = SweepValueSpinBox(choice.values, choice.unit)
            combo = QtWidgets.QComboBox()
            combo.setModel(SweepValuesModel(choice.values, choice.unit, combo))
            combo.setMaxVisibleItems(16)
            combo.setMinimumWidth(190)
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(20)
            combo.setToolTip(f'{len(choice.values):,} sweep values; indices are 0-based.')
            if color:
                spin.setStyleSheet(f'QAbstractSpinBox {{ background-color: {color}; color: #182333; }}')
                combo.setStyleSheet(f'QComboBox {{ background-color: {color}; color: #182333; }}')
            self.table.setCellWidget(row, 1, spin)
            self.table.setCellWidget(row, 2, combo)
            self.table.setRowHeight(row, max(36, spin.sizeHint().height()+8))
            spin.indexChanged.connect(lambda index, key=choice.key, gen=generation: self._select(key, index, gen))
            combo.currentIndexChanged.connect(lambda index, key=choice.key, gen=generation: self._select(key, index, gen))
            self.rows[choice.key] = (choice, spin, combo)
        self.table.setEnabled(bool(self.choices))
        self.set_point(min(self.point, self.maximum_point))

    def set_point(self, point):
        self.point = max(0, min(int(point), self.maximum_point))
        coordinates = coordinates_at(self.point, self.shape)
        for choice, spin, combo in self.rows.values():
            index = coordinates[choice.dimension]
            with QtCore.QSignalBlocker(spin), QtCore.QSignalBlocker(combo):
                spin.set_index(index)
                combo.setCurrentIndex(index)
        self.status.setText(f'Point {self.point:,} / {prod(self.shape)-1:,} (0-based) | axis indices {coordinates}'
                            if self.choices else 'No sweep parameters are configured.')

    def _select(self, key, index, generation):
        if generation != self._generation or key not in self.rows or index < 0:
            return
        choice = self.rows[key][0]
        coordinates = list(coordinates_at(self.point, self.shape))
        coordinates[choice.dimension] = int(index)
        point = point_at(coordinates, self.shape)
        previous = self.point
        if point > self.maximum_point:
            self.set_point(previous)
            self.status.setText(f'This selection exceeds the plot index limit ({self.maximum_point:,}).')
            return
        self.set_point(point)
        if point != previous:
            self.pointSelected.emit(point)

    def clear_context(self):
        self.set_choices((), ())
