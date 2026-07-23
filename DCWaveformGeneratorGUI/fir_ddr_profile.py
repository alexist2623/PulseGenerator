"""HWH-derived FIR-DDR capture-rate and trigger-delay metadata.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


_KNOWN_RATES_HZ = {
    "1_msps": 1_000_000.0,
    "50_ksps": 50_000.0,
}


@dataclass(frozen=True)
class FirDdrProfile:
    """Normalized FIR-DDR metadata reported by the loaded HWH/QICK driver."""

    name: str
    sample_rate_hz: float
    sample_rate_msps: float
    sample_period_us: float
    decimation: int
    input_rate_mhz: float
    group_delay_input_samples: float
    uses_fpga_trigger_delay: bool
    trigger_delay_samples: int
    software_warmup_compensation: bool
    config: Mapping[str, Any]

    @property
    def trigger_delay_us(self) -> float:
        return self.trigger_delay_samples * self.sample_period_us

    @property
    def rate_label(self) -> str:
        return format_sample_rate_hz(self.sample_rate_hz)

    @property
    def timing_label(self) -> str:
        delay = (
            f", FPGA delay {self.trigger_delay_samples} samples "
            f"({self.trigger_delay_us:g} us)"
            if self.uses_fpga_trigger_delay
            else ", tProcessor FIR warm-up compensation"
        )
        return (
            f"{self.rate_label} ({self.sample_period_us:g} us/sample{delay})"
        )


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise RuntimeError(f"HWH {name} must be positive and finite")
    return result


def format_sample_rate_hz(sample_rate_hz: Any) -> str:
    """Format supported and custom sample rates for compact GUI status text."""

    rate_hz = _positive_float(sample_rate_hz, "sample_rate_hz")
    if abs(rate_hz - 1_000_000.0) <= 0.5:
        return "1 MSPS"
    if abs(rate_hz - 50_000.0) <= 0.5:
        return "50 kSPS"
    if rate_hz >= 1_000_000.0:
        return f"{rate_hz / 1_000_000.0:g} MSPS"
    if rate_hz >= 1_000.0:
        return f"{rate_hz / 1_000.0:g} kSPS"
    return f"{rate_hz:g} S/s"


def _sample_rate_hz(ddr_cfg: Mapping[str, Any]) -> float:
    for key in (
        "stored_sample_rate_hz",
        "fir_output_sample_rate_hz",
        "fir_sample_rate_hz",
    ):
        value = ddr_cfg.get(key)
        if value is not None:
            return _positive_float(value, key)

    for key in (
        "stored_sample_rate_msps",
        "fir_output_fs_mhz",
        "fir_sample_rate_msps",
    ):
        value = ddr_cfg.get(key)
        if value is not None:
            return _positive_float(value, key) * 1_000_000.0

    raise RuntimeError("HWH does not report the FIR-DDR stored sample rate")


def _rate_profile(rate_hz: float, reported_name: Any) -> str:
    normalized_name = str(reported_name or "").strip().lower()
    for name, expected_hz in _KNOWN_RATES_HZ.items():
        if normalized_name == name:
            if abs(rate_hz - expected_hz) > max(0.5, expected_hz * 1.0e-9):
                raise RuntimeError(
                    f"HWH FIR rate profile {name!r} conflicts with "
                    f"reported rate {rate_hz:g} Hz"
                )
            return name

    for name, expected_hz in _KNOWN_RATES_HZ.items():
        if abs(rate_hz - expected_hz) <= max(0.5, expected_hz * 1.0e-9):
            return name

    supported = ", ".join(f"{value:g} Hz" for value in _KNOWN_RATES_HZ.values())
    raise RuntimeError(
        f"unsupported FIR-DDR stored sample rate {rate_hz:g} Hz; "
        f"supported HWH profiles are {supported}"
    )


def resolve_fir_ddr_profile(soccfg: Any, *, context: str = "FIR DDR") -> FirDdrProfile:
    """Resolve one supported capture profile from ``soccfg['ddr4_buf']``.

    The 50 kSPS firmware must expose the V2 programmable trigger-delay
    register. Its delay is applied in FPGA valid-sample units, so software must
    not move the tProcessor trigger by the FIR group delay.
    """

    try:
        ddr_cfg = dict(soccfg["ddr4_buf"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"{context} requires a DDR sample buffer in the HWH") from exc

    if not ddr_cfg.get("sample_capture", False):
        raise RuntimeError(f"{context} requires sample-triggered DDR firmware")
    if not ddr_cfg.get("fir_enabled", False):
        raise RuntimeError(f"{context} requires an HWH-detected FIR DDR path")

    sample_rate_hz = _sample_rate_hz(ddr_cfg)
    profile_name = _rate_profile(sample_rate_hz, ddr_cfg.get("fir_rate_profile"))
    decimation = int(ddr_cfg.get("fir_decimation", 0))
    if decimation <= 0:
        raise RuntimeError("HWH fir_decimation must be positive")
    input_rate_mhz = _positive_float(
        ddr_cfg.get("fir_input_fs_mhz", 300.0),
        "fir_input_fs_mhz",
    )
    group_delay = float(ddr_cfg.get("fir_group_delay_input_samples", 0.0))
    if not isfinite(group_delay) or group_delay < 0.0:
        raise RuntimeError("HWH fir_group_delay_input_samples must be nonnegative")

    uses_fpga_trigger_delay = profile_name == "50_ksps"
    trigger_delay_samples = 0
    if uses_fpga_trigger_delay:
        if not ddr_cfg.get("supports_trigger_delay", False):
            raise RuntimeError(
                "50 kSPS FIR-DDR HWH requires axis_buffer_ddr_sample_v2 "
                "programmable trigger delay"
            )
        if ddr_cfg.get("trigger_delay_units") not in (None, "valid_input_samples"):
            raise RuntimeError(
                "50 kSPS DDR trigger delay must use valid_input_samples units"
            )
        trigger_delay_samples = int(
            ddr_cfg.get("trigger_delay_default_samples", 0)
        )
        if trigger_delay_samples < 0:
            raise RuntimeError("HWH trigger_delay_default_samples must be nonnegative")

    return FirDdrProfile(
        name=profile_name,
        sample_rate_hz=sample_rate_hz,
        sample_rate_msps=sample_rate_hz / 1_000_000.0,
        sample_period_us=1_000_000.0 / sample_rate_hz,
        decimation=decimation,
        input_rate_mhz=input_rate_mhz,
        group_delay_input_samples=group_delay,
        uses_fpga_trigger_delay=uses_fpga_trigger_delay,
        trigger_delay_samples=trigger_delay_samples,
        software_warmup_compensation=not uses_fpga_trigger_delay,
        config=ddr_cfg,
    )


__all__ = [
    "FirDdrProfile",
    "format_sample_rate_hz",
    "resolve_fir_ddr_profile",
]
