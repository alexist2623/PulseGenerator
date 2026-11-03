"""Resolve RF filter requests to the ADMV8818 settings written by QICK.

This mirrors FilterADMV8818.set_filter, including first-band overlap selection,
strict endpoints and /16 quantization. These are register settings, not measured
cutoff frequencies. No hardware access or changes to legacy files are required.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


MAPPING_VERSION = "qick-admv8818-first-band-div16-v1"
FILTER_BANDS = {
    "HPF": ((1.75, 3.55), (3.40, 7.25), (6.60, 12.60), (12.50, 19.90)),
    "LPF": ((2.05, 3.85), (3.35, 7.25), (7.00, 13.00), (12.55, 18.85)),
}
# HPF3's 12.60 GHz limit intentionally follows the existing QICK driver.
# The datasheet's 12.00 GHz endpoint must not silently replace that mapping.


@dataclass(frozen=True)
class FilterSettings:
    hpf_bank: int = 0
    hpf_state: int = 0
    lpf_bank: int = 0
    lpf_state: int = 0

    def nominal_cutoffs_ghz(self):
        """Endpoint-interpolated estimates, never measured per-state cutoffs."""
        values = {}
        for section, bank, state in (("HPF", self.hpf_bank, self.hpf_state),
                                     ("LPF", self.lpf_bank, self.lpf_state)):
            if bank:
                lower, upper = FILTER_BANDS[section][bank - 1]
                if section == "HPF" and bank == 3:
                    upper = 12.0
                values[section] = lower + state * (upper - lower) / 15
        return values

    @property
    def label(self):
        def section(name, bank, state):
            return f"{name}{bank} state {state}" if bank else f"{name} bypass"

        return (section("HPF", self.hpf_bank, self.hpf_state) + "; "
                + section("LPF", self.lpf_bank, self.lpf_state))

    def as_dict(self):
        return {
            "mapping_version": MAPPING_VERSION,
            "source": "derived_from_request_not_hardware_readback",
            "hpf_bank": self.hpf_bank,
            "hpf_state": self.hpf_state,
            "lpf_bank": self.lpf_bank,
            "lpf_state": self.lpf_state,
            "WR0_SW": 0xC0 + (self.hpf_bank << 3) + self.lpf_bank,
            "WR0_FILTER": (self.hpf_state << 4) + self.lpf_state,
        }


def _frequency(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} is missing or invalid") from None
    if not isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _section(frequency, section):
    for bank, (lower, upper) in enumerate(FILTER_BANDS[section], 1):
        if lower < frequency < upper:
            return bank, int((frequency - lower) / ((upper - lower) / 16))
    raise ValueError(f"{section} cutoff {frequency:g} GHz is not supported by QICK")


def resolve_filter_settings(filter_type, cutoff_ghz=None, bandwidth_ghz=None):
    """Return exact bank/state identity; ignore settings unused by the mode."""
    mode = str(filter_type or "").strip().lower()
    mode = {"lpf": "lowpass", "hpf": "highpass"}.get(mode, mode)
    if mode == "bypass":
        return FilterSettings()
    if mode not in ("lowpass", "highpass", "bandpass"):
        raise ValueError(f"unknown filter type {filter_type!r}")
    cutoff = _frequency(cutoff_ghz, "filter cutoff/center")
    if mode == "lowpass":
        return FilterSettings(0, 0, *_section(cutoff, "LPF"))
    if mode == "highpass":
        return FilterSettings(*_section(cutoff, "HPF"), 0, 0)
    bandwidth = _frequency(bandwidth_ghz, "filter bandwidth")
    return FilterSettings(
        *_section(cutoff - bandwidth / 2, "HPF"),
        *_section(cutoff + bandwidth / 2, "LPF"),
    )


def filter_request_from_mappings(*mappings):
    """Read legacy JSON/config fields in precedence order (frequencies in GHz).

    The GUI's filter_cutoff field holds center frequency for bandpass. Explicit
    center/bandwidth and low/high cutoff representations are also supported.
    Missing or invalid fields stay unknown, never defaulting to bypass.
    """
    sources = [item for item in mappings if isinstance(item, Mapping)]

    def first(names):
        for source in sources:
            for name in names:
                if source.get(name) is not None:
                    return source[name]
        return None

    def fields(kind):
        return (f"filter_{kind}_ghz", f"output_filter_{kind}_ghz",
                f"filter_{kind}", f"{kind}_ghz")

    mode = str(first(("filter_type", "output_filter_type")) or "").lower()
    center_fields = fields("center") + fields("center_frequency")
    cutoff_fields = fields("cutoff") + fields("cutoff_frequency")
    cutoff = first(center_fields + cutoff_fields if mode == "bandpass"
                   else cutoff_fields)
    bandwidth = first(fields("bandwidth"))
    if mode == "bandpass" and (cutoff is None or bandwidth is None):
        low = first(fields("low_cutoff"))
        high = first(fields("high_cutoff"))
        if low is not None and high is not None:
            try:
                low, high = float(low), float(high)
                if cutoff is None:
                    cutoff = (low + high) / 2
                if bandwidth is None:
                    bandwidth = high - low
            except (TypeError, ValueError):
                pass
    return mode, cutoff, bandwidth


def filter_settings_from_mapping(values):
    """Normalize a legacy calibration envelope or one GUI RF-path JSON object."""
    if not isinstance(values, Mapping):
        raise ValueError("filter settings must be a mapping")
    return resolve_filter_settings(*filter_request_from_mappings(
        values.get("rf_settings_actual"), values.get("configuration"), values))
