"""Worker-owned DDR cancellation; never touch instruments from the UI thread."""
from contextlib import contextmanager
from types import SimpleNamespace


@contextmanager
def ddr_capture_guard(soc, phase_callback=None):
    capture = SimpleNamespace(armed=False)
    try:
        yield capture
    except BaseException as original:
        if capture.armed:
            errors = []
            # Stop the producer before asking the capture IP to discard its queue.
            try:
                soc.stop_tproc()
            except Exception as exc:
                errors.append(f'tProcessor stop failed: {exc}')
            try:
                abort = getattr(soc, 'abort_ddr4_capture', None)
                if not callable(abort):
                    raise RuntimeError('safe DDR abort requires v8 firmware and updated server driver')
                status = dict(abort(timeout=2.0))
                if (not status.get('acknowledged') or status.get('pending')
                        or status.get('busy') or status.get('armed')):
                    raise RuntimeError(f'DDR abort was not acknowledged: {status}')
            except Exception as exc:
                errors.append(f'DDR capture abort failed: {exc}')
            if errors:
                raise RuntimeError('Cancellation cleanup incomplete; do not re-arm. '
                                   + '; '.join(errors)) from original
            if phase_callback is not None:
                phase_callback('ddr_abort', 'completed', 'DDR capture cancelled; AXI writes drained and capture queue cleared')
        raise
