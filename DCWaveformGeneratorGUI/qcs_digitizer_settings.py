"""Shared Keysight M5200 digitizer settings.

The GUI stores the requested full-scale input range in volts.  QCS exposes
that setting on each mapped physical digitizer connector as
``physical.settings.range.value``.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Tuple


QCS_M5200_MIN_INPUT_RANGE_V = 0.045
QCS_M5200_MAX_INPUT_RANGE_V = 1.8
DEFAULT_QCS_M5200_INPUT_RANGE_V = 0.9


def normalize_qcs_m5200_input_range_v(value: Any) -> float:
    """Validate and return one M5200 input range in volts."""

    if isinstance(value, bool):
        raise TypeError("QCS M5200 input range must be a real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError("QCS M5200 input range must be finite")
    if not (
        QCS_M5200_MIN_INPUT_RANGE_V
        <= result
        <= QCS_M5200_MAX_INPUT_RANGE_V
    ):
        raise ValueError(
            "QCS M5200 input range must be between "
            f"{QCS_M5200_MIN_INPUT_RANGE_V:g} V and "
            f"{QCS_M5200_MAX_INPUT_RANGE_V:g} V"
        )
    return result


def apply_qcs_m5200_input_range(
    mapper: Any,
    acquisition_channels: Any,
    input_range_v: Any,
) -> Tuple[Any, ...]:
    """Apply an input range to every mapped physical M5200 connector.

    Lightweight injected mapper adapters used outside hardware execution may
    omit either ``get_physical_channels`` or the physical settings object.
    QCS 2.5.5 M5200 connectors expose both APIs and are updated directly.
    """

    value = normalize_qcs_m5200_input_range_v(input_range_v)
    get_physical_channels = getattr(mapper, "get_physical_channels", None)
    if not callable(get_physical_channels):
        return ()
    physical_channels = tuple(get_physical_channels(acquisition_channels))
    if not physical_channels:
        raise ValueError(
            "QCS acquisition channel does not map to a physical M5200 input"
        )
    range_settings = []
    for physical_channel in physical_channels:
        settings = getattr(physical_channel, "settings", None)
        range_setting = getattr(settings, "range", None)
        if range_setting is None or not hasattr(range_setting, "value"):
            continue
        range_setting.value = value
        range_settings.append(range_setting)
    return tuple(range_settings)


__all__ = [
    "DEFAULT_QCS_M5200_INPUT_RANGE_V",
    "QCS_M5200_MAX_INPUT_RANGE_V",
    "QCS_M5200_MIN_INPUT_RANGE_V",
    "apply_qcs_m5200_input_range",
    "normalize_qcs_m5200_input_range_v",
]
