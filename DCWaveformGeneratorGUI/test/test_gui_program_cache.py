"""Clear cache uses the real local cache, without a board or acquisition."""
from pathlib import Path
from types import SimpleNamespace

from PyQt5 import QtCore, QtTest, QtWidgets

import qick_program_cache as cache
from test_qick_program_cache import build, isolated_cache  # noqa: F401
from test_square_awg_exclusion import window  # noqa: F401


def test_clear_button_deletes_artifacts_and_next_compile_misses(window):
    first = build()
    assert build().compile_cache_info['status'] == 'hit'
    panel = window._experiment_panel
    assert panel.clear_cache_button.text() == 'Clear cache'
    QtTest.QTest.mouseClick(panel.clear_cache_button, QtCore.Qt.LeftButton)
    assert not cache._memory
    assert not list(cache.cache_directory().glob('*.json.gz'))
    assert 'removed 1 files and 1 memory entries' in panel.run_status.text()
    rebuilt = build()
    assert rebuilt.compile_cache_info['status'] == 'miss'
    assert rebuilt.asm() == first.asm()


def test_clear_is_disabled_during_compile_or_run_and_handler_guards_worker(window):
    build()
    panel = window._experiment_panel
    panel.set_running(True, 'Compiling', show_progress=False)
    assert not panel.clear_cache_button.isEnabled()
    QtTest.QTest.mouseClick(panel.clear_cache_button, QtCore.Qt.LeftButton)
    assert list(cache.cache_directory().glob('*.json.gz'))
    panel.set_running(False, 'Ready')
    assert panel.clear_cache_button.isEnabled()
    # The worker may still be exiting after its completion signal restores UI.
    window._experiment_thread = SimpleNamespace(isRunning=lambda: True)
    try:
        window._clear_qick_program_cache()
        assert list(cache.cache_directory().glob('*.json.gz'))
        assert 'Wait for the current QICK task' in window.statusBar().currentMessage()
    finally:
        window._experiment_thread = None


def test_clear_button_reports_partial_failure(window, monkeypatch):
    build()
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning',
                        lambda parent, title, text: warnings.append((title, text)))

    def locked(_path, *args, **kwargs):
        raise PermissionError('file locked')

    monkeypatch.setattr(Path, 'unlink', locked)
    window._experiment_panel.clear_cache_button.click()
    assert not cache._memory
    assert list(cache.cache_directory().glob('*.json.gz'))
    assert warnings[0][0] == 'Cache partially cleared'
    assert 'file locked' in warnings[0][1]
    assert 'next run will compile again' not in window._experiment_panel.run_status.text()
