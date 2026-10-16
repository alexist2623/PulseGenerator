"""Explicit, opt-in GUI/hardware validation. Importing never opens a device.

Qt and PyVISA are deliberately not imported here. Software tests exercise this
infrastructure with injected sessions; they are not physical validation.
"""
from .archive import RunArchive, Point, Reference, verify_archive, private_evidence_root
from .coverage import Fixture, FirmwareIdentity, logarithmic_durations_us

__all__ = ['RunArchive', 'Point', 'Reference', 'verify_archive', 'Fixture',
           'FirmwareIdentity', 'logarithmic_durations_us', 'private_evidence_root']
