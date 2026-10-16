"""Concrete Qt event operations for the production waveform case adapter.

These operations never invoke model setters, QAction.trigger(), signals, Run,
or instruments. Menu selections and table edits use the visible Qt controls.
"""
import math

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets, sip


OPERATIONS = frozenset(('table_text', 'table_select', 'table_check', 'check',
                        'context_menu', 'menu_action', 'dialog_button', 'shortcut_modal'))


def _caption(text):
    return text.replace('&', '').replace('\u2026', '...').strip()


def reveal(widget):
    """Scroll only the viewport; never change a parameter value."""
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QtWidgets.QScrollArea):
            parent.ensureWidgetVisible(widget)
        parent = parent.parentWidget()
    QtWidgets.QApplication.processEvents()


def _cell(table, value):
    row, column = int(value['row']), int(value.get('column', 0))
    item = table.item(row, column)
    if item is None:
        raise ValueError(f'Missing table cell {row}, {column}')
    table.scrollToItem(item)
    reveal(table)
    return item, table.visualItemRect(item).center()


def _same(actual, expected):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        # Production waveform tables render six significant digits; the binding
        # separately verifies the committed model at its original precision.
        return math.isclose(float(actual), expected, abs_tol=1e-9, rel_tol=5e-6)
    return actual == expected


def _choose_menu(menu, path, dialog_actions, evidence):
    """Select nested menus by actual mouse events, with bounded modal cleanup."""
    from .qt_runner import apply_actions
    errors = []
    def edit_dialog():
        dialog = QtWidgets.QApplication.activeModalWidget()
        try:
            if dialog is None:
                raise ValueError('The selected menu did not open its dialog')
            evidence.extend(apply_actions(dialog_actions(dialog)))
            posted_accept = bool(evidence and evidence[-1].get('operation') == 'dialog_button')
            if not posted_accept and not sip.isdeleted(dialog) and dialog.isVisible():
                raise ValueError('Dialog actions must explicitly accept or cancel')
        except Exception as exc:
            errors.append(exc)
            if dialog is not None and not sip.isdeleted(dialog):
                QtTest.QTest.keyClick(dialog, QtCore.Qt.Key_Escape)
    for position, name in enumerate(path):
        action = next((a for a in menu.actions() if _caption(a.text()) == _caption(name)), None)
        if action is None or not action.isEnabled():
            raise ValueError(f'Unavailable menu item {name!r}')
        center = menu.actionGeometry(action).center()
        if position < len(path)-1:
            if action.menu() is None:
                raise ValueError(f'Not a submenu: {name!r}')
            QtTest.QTest.mouseMove(menu, center)
            QtTest.QTest.mouseClick(menu, QtCore.Qt.LeftButton, pos=center)
            QtTest.QTest.qWait(250)
            if not action.menu().isVisible():
                QtTest.QTest.keyClick(menu, QtCore.Qt.Key_Right)
                QtWidgets.QApplication.processEvents()
            menu = action.menu()
            if not menu.isVisible():
                raise ValueError(f'Submenu did not open: {name!r}')
        else:
            if dialog_actions is not None:
                QtCore.QTimer.singleShot(0, edit_dialog)
            QtTest.QTest.mouseClick(menu, QtCore.Qt.LeftButton, pos=center)
    if errors:
        raise errors[0]


def apply_control_action(action):
    """Return JSON-ready input evidence for one extended ``QtAction``."""
    widget, operation, value = action.target, action.operation, action.value
    if operation not in OPERATIONS:
        raise ValueError(f'Unsupported case-control Qt operation: {operation}')
    if not widget.isVisible() or not widget.isEnabled():
        raise ValueError('Qt action target must be visible and enabled')
    reveal(widget)
    evidence = dict(widget_type=type(widget).__name__, operation=operation,
                    object_name=widget.objectName())
    if operation == 'shortcut_modal':
        from .qt_runner import apply_actions
        nested, errors = [], []
        def edit_shortcut():
            dialog = QtWidgets.QApplication.activeModalWidget()
            try:
                if dialog is None:
                    raise ValueError('Shortcut did not open its dialog')
                nested.extend(apply_actions(value['dialog_actions'](dialog)))
            except Exception as exc:
                errors.append(exc)
                if dialog is not None and not sip.isdeleted(dialog):
                    QtTest.QTest.keyClick(dialog, QtCore.Qt.Key_Escape)
        widget.activateWindow()
        widget.setFocus()
        QtWidgets.QApplication.processEvents()
        deadline = QtCore.QTimer()
        deadline.setSingleShot(True)
        def timeout_shortcut():
            errors.append(TimeoutError('Qt shortcut dialog exceeded 10 seconds'))
            dialog = QtWidgets.QApplication.activeModalWidget()
            if dialog is not None:
                QtTest.QTest.keyClick(dialog, QtCore.Qt.Key_Escape)
        deadline.timeout.connect(timeout_shortcut)
        deadline.start(10_000)
        QtCore.QTimer.singleShot(0, edit_shortcut)
        QtTest.QTest.keyClick(widget, value['key'], value['modifiers'])
        deadline.stop()
        if errors:
            raise errors[0]
        if not nested:
            raise ValueError('No shortcut dialog input was observed')
        evidence.update(input=dict(key=int(value['key']), modifiers=int(value['modifiers'])), nested=nested)
    elif operation == 'dialog_button':
        # Queue keyboard acceptance so the nested input callback unwinds before
        # a static QFileDialog destroys its widgets on Windows.
        QtTest.QTest.qWait(150)
        for kind in (QtCore.QEvent.KeyPress, QtCore.QEvent.KeyRelease):
            QtCore.QCoreApplication.postEvent(widget, QtGui.QKeyEvent(kind,
                QtCore.Qt.Key_Return, QtCore.Qt.NoModifier))
        evidence.update(input='posted Return press/release', actual='dialog acceptance requested')
        # Let the outer QFileDialog event loop deliver these after the input
        # callback unwinds. Reentrant processEvents during its destruction can
        # invalidate the static Windows file-dialog internals.
        return evidence
    elif operation.startswith('table_'):
        item, position = _cell(widget, value)
        expected = value.get('expected', value.get('text'))
        QtTest.QTest.mouseClick(widget.viewport(), QtCore.Qt.LeftButton, pos=position)
        if operation == 'table_check':
            wanted = bool(value.get('checked', True))
            if bool(item.checkState() == QtCore.Qt.Checked) != wanted:
                QtTest.QTest.keyClick(widget, QtCore.Qt.Key_Space)
            actual = widget.item(value['row'], value.get('column', 0)).checkState() == QtCore.Qt.Checked
            if actual != wanted:
                raise ValueError('Table checkbox did not commit')
        elif operation == 'table_text':
            if not item.flags() & QtCore.Qt.ItemIsEditable:
                raise ValueError('Table cell is not editable')
            widget.setFocus()
            QtTest.QTest.mouseDClick(widget.viewport(), QtCore.Qt.LeftButton, pos=position)
            QtWidgets.QApplication.processEvents()
            editor = QtWidgets.QApplication.focusWidget()
            if not isinstance(editor, QtWidgets.QLineEdit):
                editors = [e for e in widget.findChildren(QtWidgets.QLineEdit) if e.isVisible()]
                editor = editors[0] if len(editors) == 1 else editor
            if not isinstance(editor, QtWidgets.QLineEdit):
                QtTest.QTest.keyClick(widget, QtCore.Qt.Key_F2)
                editor = QtWidgets.QApplication.focusWidget()
            if not isinstance(editor, QtWidgets.QLineEdit):
                raise ValueError(f'Table did not open a text editor: focus={type(editor).__name__}, cell={position}, viewport={widget.viewport().size()}')
            QtTest.QTest.keyClick(editor, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
            QtTest.QTest.keyClicks(editor, str(value['text']))
            QtTest.QTest.keyClick(editor, QtCore.Qt.Key_Tab)
            QtWidgets.QApplication.processEvents()
            actual = widget.item(value['row'], value.get('column', 0)).text()
            if not _same(actual, expected):
                raise ValueError(f'Table input readback {actual!r} differs from {expected!r}')
        else:
            actual = widget.currentRow()
        evidence.update(input=dict(value), actual=actual)
    elif operation == 'check':
        wanted = bool(value)
        if widget.isChecked() != wanted:
            widget.setFocus()
            QtTest.QTest.keyClick(widget, QtCore.Qt.Key_Space)
        if widget.isChecked() != wanted:
            raise ValueError('Checkbox did not commit')
        evidence.update(input=wanted, actual=widget.isChecked())
    else:
        nested, errors, opened = [], [], []
        path = tuple(value['path'])
        deadline = QtCore.QTimer()
        deadline.setSingleShot(True)
        def timeout():
            errors.append(TimeoutError('Qt case menu/dialog exceeded 10 seconds'))
            modal = QtWidgets.QApplication.activeModalWidget()
            if modal is not None:
                QtTest.QTest.keyClick(modal, QtCore.Qt.Key_Escape)
            for _ in range(10):
                popup = QtWidgets.QApplication.activePopupWidget()
                if popup is None:
                    break
                QtTest.QTest.keyClick(popup, QtCore.Qt.Key_Escape)
        deadline.timeout.connect(timeout)
        deadline.start(10_000)
        def choose():
            menu = QtWidgets.QApplication.activePopupWidget()
            try:
                if not isinstance(menu, QtWidgets.QMenu):
                    raise ValueError('Context/menu event did not open a QMenu')
                opened.append(True)
                _choose_menu(menu, path, value.get('dialog_actions'), nested)
            except Exception as exc:
                errors.append(exc)
                for _ in range(10):
                    popup = QtWidgets.QApplication.activePopupWidget()
                    if popup is None:
                        break
                    QtTest.QTest.keyClick(popup, QtCore.Qt.Key_Escape)
        if operation == 'context_menu':
            _item, center = _cell(widget, value)
            QtCore.QTimer.singleShot(0, choose)
            # QContextMenuEvent is the native event following a right mouse click.
            # QtTest mouse events alone do not synthesize it on every platform.
            QtTest.QTest.mouseClick(widget.viewport(), QtCore.Qt.RightButton, pos=center)
            if not opened and not errors:
                event = QtGui.QContextMenuEvent(QtGui.QContextMenuEvent.Mouse, center,
                                                widget.viewport().mapToGlobal(center))
                QtWidgets.QApplication.sendEvent(widget.viewport(), event)
        else:
            top = next((a for a in widget.actions() if _caption(a.text()) == _caption(path[0])), None)
            if top is None or top.menu() is None:
                raise ValueError(f'Unavailable menu {path[0]!r}')
            path = path[1:]
            QtCore.QTimer.singleShot(0, choose)
            QtTest.QTest.mouseClick(widget, QtCore.Qt.LeftButton,
                                   pos=widget.actionGeometry(top).center())
            QtWidgets.QApplication.processEvents()
        deadline.stop()
        if errors:
            raise errors[0]
        if not opened:
            raise ValueError('No real Qt menu selection was observed')
        evidence.update(path=list(value['path']), nested=nested)
    QtWidgets.QApplication.processEvents()
    return evidence
