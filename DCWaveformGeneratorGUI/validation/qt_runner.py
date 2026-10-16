"""Real Qt input, normal production Run worker, asynchronous scope coordination.

No hardware or GUI action occurs before start(). An observer factory briefly
wraps QickExperimentWorker construction during the actual button click, attaches
signals before its thread starts, and returns the original production object.
It does not replace worker.run, compiler, upload, acquisition, or QCoDeS storage.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import traceback

from PyQt5 import QtCore, QtTest, QtWidgets

from .archive import canonical, digest, json_file


def stored_run_identity(result):
    """Identifiers from the normal detached StoredQickExperiment result."""
    return dict(run_id=result.run_id, guid=result.guid, database_path=str(result.database_path),
                row_count=result.row_count, iq_storage_mode=result.iq_storage_mode)


@dataclass(frozen=True)
class QtAction:
    target: object
    operation: str
    value: object = None
    expected: object = None


def apply_actions(actions):
    """Mouse/key events including commit. No model or widget value setters."""
    if QtCore.QThread.currentThread() != QtWidgets.QApplication.instance().thread():
        raise RuntimeError('GUI actions must run on the Qt application thread')
    evidence = []
    for action in actions:
        widget = action.target
        if not widget.isVisible() or not widget.isEnabled():
            raise ValueError('Qt action target must be visible and enabled')
        if action.operation == 'click':
            QtTest.QTest.mouseClick(widget, QtCore.Qt.LeftButton)
        elif action.operation == 'text':
            edit = widget.lineEdit() if isinstance(widget, QtWidgets.QAbstractSpinBox) else widget
            QtTest.QTest.mouseClick(edit, QtCore.Qt.LeftButton)
            QtTest.QTest.keyClick(edit, QtCore.Qt.Key_A, QtCore.Qt.ControlModifier)
            QtTest.QTest.keyClicks(edit, str(action.value))
            QtTest.QTest.keyClick(edit, QtCore.Qt.Key_Tab)
        elif action.operation == 'tab':
            bar = widget.tabBar() if isinstance(widget, QtWidgets.QTabWidget) else widget
            index = int(action.value)
            if not 0 <= index < bar.count():
                raise ValueError('Tab index out of range')
            QtTest.QTest.mouseClick(bar, QtCore.Qt.LeftButton, pos=bar.tabRect(index).center())
        elif action.operation == 'key':
            QtTest.QTest.keyClick(widget, int(action.value))
        elif action.operation == 'combo':
            if not isinstance(widget, QtWidgets.QComboBox) or not 0 <= int(action.value) < widget.count():
                raise ValueError('Combo action requires a valid item index')
            widget.setFocus()  # Focus only; values are selected through key events.
            QtTest.QTest.keyClick(widget, QtCore.Qt.Key_Home)
            for _ in range(int(action.value)):
                QtTest.QTest.keyClick(widget, QtCore.Qt.Key_Down)
            QtTest.QTest.keyClick(widget, QtCore.Qt.Key_Tab)
        elif action.operation == 'modal':
            nested, errors = [], []
            def edit_modal():
                dialog = QtWidgets.QApplication.activeModalWidget()
                try:
                    if dialog is None:
                        raise ValueError('The click did not open a modal dialog')
                    nested.extend(apply_actions(action.value(dialog)))
                    if dialog.isVisible():
                        raise ValueError('Modal action list must explicitly click its accept/cancel button')
                except Exception as exc:
                    errors.append(exc)
                    if dialog is not None:
                        dialog.reject()
            QtCore.QTimer.singleShot(0, edit_modal)
            QtTest.QTest.mouseClick(widget, QtCore.Qt.LeftButton)
            if errors:
                raise errors[0]
            if not nested:
                raise ValueError('No modal Qt inputs were observed')
            evidence.append(dict(object_name=widget.objectName(), operation='modal', nested=nested))
            continue
        else:
            from .qt_case_controls import OPERATIONS, apply_control_action
            if action.operation not in OPERATIONS:
                raise ValueError(f'Unsupported Qt operation: {action.operation}')
            evidence.append(apply_control_action(action))
            continue
        actual = (widget.value() if isinstance(widget, QtWidgets.QAbstractSpinBox)
                  else widget.isChecked() if isinstance(widget, QtWidgets.QAbstractButton) and widget.isCheckable()
                  else widget.currentIndex() if isinstance(widget, (QtWidgets.QTabWidget, QtWidgets.QTabBar, QtWidgets.QComboBox))
                  else widget.text() if hasattr(widget, 'text') else None)
        if action.expected is not None and actual != action.expected:
            raise ValueError(f'GUI commit readback mismatch: {actual!r} != {action.expected!r}')
        evidence.append(dict(object_name=widget.objectName(), widget_type=type(widget).__name__,
                             operation=action.operation, input=action.value, actual=actual))
    return evidence


class GuiHardwareRunner(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    stage_changed = QtCore.pyqtSignal(str)
    attention_required = QtCore.pyqtSignal(str)
    _task_done = QtCore.pyqtSignal(str, object, object)

    def __init__(self, *, window, production_module, run_button, archive,
                 scope_factory, live_identity_check, result_identity, point_mapper,
                 cleanup, comparison=None, timeout_s=120., execution_context='physical', parent=None):
        super().__init__(parent)
        self.window, self.module, self.button = window, production_module, run_button
        self.archive, self.scope_factory = archive, scope_factory
        self.live_identity_check = live_identity_check
        self.result_identity, self.point_mapper = result_identity, point_mapper
        self.cleanup, self.comparison = cleanup, comparison
        self.timeout_s = timeout_s
        if execution_context not in ('physical', 'offline_test'):
            raise ValueError('Declare physical execution or an offline test double')
        self.execution_context = execution_context
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='qick-validation-scope')
        self.scope = self.worker = self.record = self.result = None
        self.started = self.ending = False
        self.worker_terminal = None
        self.failure = None
        self.cleanup_result = None
        self._gui_snapshot = self._sealed_gui_definition = None
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._timeout)
        self._task_done.connect(self._receive)

    def start(self, *, allow_hardware=False, actions=(), gui_snapshot=None):
        if not allow_hardware:
            raise ValueError('Explicit allow_hardware=True is required for this configured run')
        if self.started:
            raise RuntimeError('Runner is one attempt; use a new archive/runner for retries')
        if not self.archive.sealed:
            raise ValueError('Seal complete intended numerical references before Run')
        if not self.window.isVisible() or not self.button.isVisible() or not self.button.isEnabled():
            raise ValueError('Use the visible, enabled production GUI Run button')
        if getattr(self.window, '_experiment_thread', None) is not None:
            raise ValueError('An experiment is already active')
        if not callable(gui_snapshot):
            raise ValueError('Provide a read-only GUI snapshot callback')
        self.archive.identity.verify()
        input_events = apply_actions(actions)
        snapshot = gui_snapshot()
        expected = __import__('json').loads((self.archive.root/'gui_requested.json').read_text(encoding='utf8'))
        if canonical(snapshot) != canonical(expected):
            raise ValueError('Visible GUI differs from the archived requested definition')
        self._gui_snapshot = gui_snapshot
        self._sealed_gui_definition = canonical(expected)
        json_file(self.archive.root/'qt_input_events.json', input_events)
        self.started = True
        self.timer.start(int(self.timeout_s * 1000))
        self._submit('arm', self._arm)

    def _submit(self, name, fn):
        self.stage_changed.emit(name)
        future = self.pool.submit(fn)
        def complete(done):
            try:
                value = done.result()
                self._task_done.emit(name, value, None)
            except BaseException:
                self._task_done.emit(name, None, traceback.format_exc())
        future.add_done_callback(complete)

    def _arm(self):
        observed = self.live_identity_check()
        json_file(self.archive.root/'live_identity_before.json', observed)
        self.archive.identity.verify_observed(observed)
        self.scope = self.scope_factory(self.archive.root/'scope')
        return self.scope.arm()

    @QtCore.pyqtSlot(str, object, object)
    def _receive(self, name, value, error):
        if name == 'archive':
            self.pool.shutdown(wait=False)
            if error:
                self.attention_required.emit(error)
                self.finished.emit(dict(complete=False, verdict='inconclusive', error=error,
                                        archive_directory=str(self.archive.root)))
            else:
                self.finished.emit(value)
            return
        self.archive.event(name, success=error is None, result=value if name != 'capture' else None, error=error)
        if error:
            self.failure = error
            if name == 'cleanup':
                self.timer.stop()
                self.cleanup_result = dict(passed=False, error=error)
                self._submit('archive', self._archive_result)
                return
            self._request_cancel()
            self._finish_if_ready()
            return
        if name == 'arm':
            if self.failure:
                self._finish_if_ready()
                return
            self._click_run()
        elif name == 'capture':
            self.record = value
            self._finish_if_ready()
        elif name == 'cleanup':
            self.timer.stop()
            self.cleanup_result = value
            self._submit('archive', self._archive_result)

    def _click_run(self):
        # Scope arm runs asynchronously. Recheck on the GUI thread immediately
        # before the click so edits during arm cannot execute against old refs.
        try:
            if not self.window.isVisible() or not self.button.isVisible() or not self.button.isEnabled():
                raise ValueError('The production Run button is no longer visible and enabled')
            if getattr(self.window, '_experiment_thread', None) is not None:
                raise ValueError('Another experiment became active while scope arm was pending')
            if self._gui_snapshot is None:
                raise ValueError('Missing retained GUI snapshot callback')
            snapshot = self._gui_snapshot()
            if canonical(snapshot) != self._sealed_gui_definition:
                raise ValueError('Visible GUI changed after scope arm; archived references no longer match')
            self.archive.event('gui_pre_run_check', success=True,
                definition_sha256=digest(self._sealed_gui_definition))
        except Exception:
            self.failure = traceback.format_exc()
            self.worker_terminal = 'rejected'
            self.archive.event('gui_pre_run_check', success=False, error=self.failure)
            self._finish_if_ready()
            return
        original = self.module.QickExperimentWorker
        created = []
        def observe(*args, **kwargs):
            worker = original(*args, **kwargs)
            created.append(worker)
            self.worker = worker
            worker.finished.connect(self._worker_finished)
            worker.failed.connect(self._worker_failed)
            worker.cancelled.connect(self._worker_cancelled)
            worker.progress_changed.connect(self._progress)
            worker.event_changed.connect(self._worker_event)
            return worker
        self.module.QickExperimentWorker = observe
        try:
            self.archive.event('run_click', input='QtTest.mouseClick', worker_class=original.__name__)
            QtTest.QTest.mouseClick(self.button, QtCore.Qt.LeftButton)
        except Exception:
            self.failure = traceback.format_exc()
        finally:
            self.module.QickExperimentWorker = original
        if len(created) != 1:
            self.failure = self.failure or 'Run did not create exactly one normal QickExperimentWorker (GUI rejection or wrong button)'
            self.worker_terminal = 'rejected'
            self._finish_if_ready()
            return
        module_path = Path(self.module.__file__)
        json_file(self.archive.root/'execution_route.json', dict(
            path='actual_qt_run_button_to_production_worker', execution_context=self.execution_context,
            gui_input='QtTest mouse/key events', normal_run_button_clicked=True,
            worker_class=f'{original.__module__}.{original.__qualname__}',
            worker_run=f'{original.run.__module__}.{original.run.__qualname__}',
            production_module=str(module_path), production_module_sha256=digest(module_path.read_bytes()),
            compiler_upload_and_qcodes_functions_replaced_by_runner=False,
            scope_only_bypass=False, historical_acquisitions_reclassified=False))
        # Capture waits off the GUI thread, concurrently with the normal worker.
        self._submit('capture', self.scope.capture)

    @QtCore.pyqtSlot(object)
    def _worker_finished(self, result):
        self.result, self.worker_terminal = result, 'finished'
        self.archive.event('worker_finished')
        self._finish_if_ready()

    @QtCore.pyqtSlot(str)
    def _worker_failed(self, error):
        self.worker_terminal, self.failure = 'failed', error
        self.archive.event('worker_failed', error=error)
        self._finish_if_ready()

    @QtCore.pyqtSlot(str)
    def _worker_cancelled(self, message):
        self.worker_terminal = 'cancelled'
        self.failure = self.failure or message
        self.archive.event('worker_cancelled', message=message)
        self._finish_if_ready()

    @QtCore.pyqtSlot(int, str)
    def _progress(self, percent, message):
        if not self.ending:
            self.archive.event('worker_progress', percent=percent, message=message)

    @QtCore.pyqtSlot(str, str, str)
    def _worker_event(self, phase, status, message):
        if not self.ending:
            self.archive.event('worker_event', phase=phase, status=status, message=message)

    def _request_cancel(self):
        if self.worker is not None and self.worker_terminal is None:
            self.worker.request_cancel()  # Existing cooperative worker cancellation.

    def cancel(self):
        if not self.started or self.ending:
            return
        self.failure = 'Validation cancellation requested'
        self.archive.event('cancel_requested')
        self._request_cancel()
        if self.worker is not None and self.worker_terminal is None:
            self.attention_required.emit('Worker has not stopped; outputs cannot safely be reset concurrently. Await cancellation or inspect the worker.')
        self._finish_if_ready()

    def _timeout(self):
        self.failure = 'Validation timeout; cooperative worker cancellation requested'
        self.archive.event('timeout')
        self._request_cancel()
        if self.worker is not None and self.worker_terminal is None:
            self.attention_required.emit('Worker has not acknowledged timeout cancellation; inspect it before attempting any output reset.')
        self._finish_if_ready()

    def _finish_if_ready(self):
        if self.ending:
            return
        if self.worker is not None and self.worker_terminal is None:
            return  # Never reset outputs while a live worker may still issue commands.
        if not self.failure and (self.worker_terminal != 'finished' or self.record is None):
            return
        self.ending = True
        def finish():
            result = dict(passed=True, output_cleanup=None, scope_restore=None)
            try:
                result['output_cleanup'] = (self.cleanup() if self.worker is not None else
                    dict(passed=True, skipped=True, reason='No DUT worker was started'))
                result['passed'] = result['output_cleanup'].get('passed') is True
            except Exception:
                result.update(passed=False, output_cleanup_error=traceback.format_exc())
            finally:
                if self.scope is not None:
                    result['scope_restore'] = self.scope.close()
                    result['passed'] = result['passed'] and result['scope_restore'].get('passed') is True
            return result
        self._submit('cleanup', finish)

    def _archive_result(self):
        storage = None
        try:
            if self.record is not None:
                acquisition = self.archive.record_capture(self.record, self.scope.directory/'raw')
                if self.worker_terminal == 'finished':
                    storage = self.result_identity(self.result)
                    if not storage.get('run_id') or not storage.get('guid') or not storage.get('database_path'):
                        raise ValueError('Missing saved production QCoDeS run identity')
                    mapping = self.point_mapper(self.record, self.result)
                    for row in mapping:
                        self.archive.link_measurement(row['point'], row['repetition'], row['channel'],
                            acquisition, row['start'], row['stop'], identity_evidence=row['identity_evidence'])
                        if self.comparison is not None:
                            options = self.comparison(row, self.result)
                            self.archive.compare(row['point'], row['repetition'], row['channel'], **options)
        except Exception:
            self.failure = self.failure or traceback.format_exc()
        return self.archive.finalize(worker_status=self.worker_terminal or 'not_started',
            cleanup=self.cleanup_result, storage_identity=storage, error=self.failure)
