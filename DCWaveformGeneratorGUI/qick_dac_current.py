"""Shared DAC current state, independent of the selected experiment tab."""
from copy import deepcopy
from datetime import datetime, timezone
import math
from PyQt5 import QtCore

REFERENCE_CURRENT_UA = 20000
REFERENCE_FULL_SCALE_MV = 800.0


def full_scale_mv(current_ua):
    current = float(current_ua)
    if not math.isfinite(current) or current <= 0:
        raise ValueError('DAC current must be positive and finite')
    return REFERENCE_FULL_SCALE_MV * current / REFERENCE_CURRENT_UA


def read_current_settings(soc):
    reader = getattr(soc, 'get_dac_current_settings', None)
    if not callable(reader):
        return {}
    return dict(reader())


def verify_current_settings(soc, settings):
    """Never execute codes compiled for a different hardware current."""
    if not settings:
        return
    actual = read_current_settings(soc)
    for dac, expected in settings.items():
        record = actual.get(str(dac), {})
        if record.get('current_ua') != expected['current_ua']:
            measured = record.get('current_ua')
            readback = 'unavailable' if measured is None else f'{measured} uA'
            raise RuntimeError(f'DAC {dac} current changed or cannot be read '
                               f'(generators {expected.get("channels", [])}; '
                               f'compiled {expected["current_ua"]} uA, readback {readback}). '
                               'Refresh the front panel and compile again.')
        if bool(record.get('dc_output')) != bool(expected.get('dc_output')):
            raise RuntimeError(f'DAC {dac} output-board type changed')
        if sorted(record.get('channels', ())) != sorted(expected.get('channels', ())):
            raise RuntimeError(f'DAC {dac} generator routing changed; identify QICK and compile again')


class DacCurrentState(QtCore.QObject):
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.records = {}
        self.source = 'unavailable'
        self.read_at = None
        self.connection = None

    def update(self, records, *, source='hardware', connection=None):
        self.records = deepcopy(dict(records))
        self.source = source
        self.connection = connection
        self.read_at = (datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                        if source == 'hardware' else None)
        self.changed.emit()

    def load_snapshot(self, records, *, connection=None):
        """A settings file must not replace readback from the same endpoint."""
        if self.source == 'hardware' and self.connection == connection:
            # Notify after the other settings have loaded, restoring per-DAC scales.
            self.changed.emit()
            return
        self.update(records, source='saved', connection=connection)

    def invalidate_readback(self):
        """Retain routing, but block output compilation for an unreadable DAC."""
        records = deepcopy(self.records)
        for record in records.values():
            record['current_ua'] = None
        self.update(records, source='unavailable', connection=self.connection)

    def for_channel(self, channel):
        return next((v for v in self.records.values()
                     if int(channel) in v.get('channels', ())), None)

    def scale(self, channel):
        record = self.for_channel(channel)
        if record and record.get('dc_output') and record.get('current_ua') is not None:
            return full_scale_mv(record['current_ua'])
        return REFERENCE_FULL_SCALE_MV

    def snapshot(self, channels, *, include_rf=False):
        wanted = set(map(int, channels))
        for record in self.records.values():
            if wanted.intersection(record.get('channels', ())) and (record.get('dc_output') or include_rf) and record.get('current_ua') is None:
                raise ValueError('DAC current could not be read; identify QICK again before output')
        return {dac: deepcopy(record) for dac, record in self.records.items()
                if wanted.intersection(record.get('channels', ()))
                and (record.get('dc_output') or include_rf) and record.get('current_ua') is not None}
