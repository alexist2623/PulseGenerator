"""Tests for HWH-derived FIR DDR capture profiles.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import pytest

from fir_ddr_profile import resolve_fir_ddr_profile


def _soccfg(rate_profile: str):
    if rate_profile == "1_msps":
        ddr = {
            "sample_capture": True,
            "fir_enabled": True,
            "fir_rate_profile": "1_msps",
            "stored_sample_rate_hz": 1_000_000.0,
            "fir_decimation": 300,
            "fir_input_fs_mhz": 300.0,
            "fir_group_delay_input_samples": 8677.0,
        }
    else:
        ddr = {
            "sample_capture": True,
            "fir_enabled": True,
            "fir_rate_profile": "50_ksps",
            "stored_sample_rate_hz": 50_000.0,
            "fir_decimation": 6000,
            "fir_input_fs_mhz": 300.0,
            "fir_group_delay_input_samples": 296_677.0,
            "supports_trigger_delay": True,
            "trigger_delay_units": "valid_input_samples",
            "trigger_delay_default_samples": 50,
            "filter_state_continuous": True,
            "decimation_phase_continuous": True,
        }
    return {"ddr4_buf": ddr}


def test_1_msps_profile_keeps_legacy_software_warmup_compensation():
    profile = resolve_fir_ddr_profile(_soccfg("1_msps"))

    assert profile.name == "1_msps"
    assert profile.sample_rate_hz == 1_000_000.0
    assert profile.sample_period_us == 1.0
    assert profile.decimation == 300
    assert profile.software_warmup_compensation is True
    assert profile.uses_fpga_trigger_delay is False
    assert profile.trigger_delay_samples == 0


def test_50_ksps_profile_uses_hwh_v2_delay_without_software_compensation():
    profile = resolve_fir_ddr_profile(_soccfg("50_ksps"))

    assert profile.name == "50_ksps"
    assert profile.sample_rate_hz == 50_000.0
    assert profile.sample_period_us == 20.0
    assert profile.decimation == 6000
    assert profile.software_warmup_compensation is False
    assert profile.uses_fpga_trigger_delay is True
    assert profile.trigger_delay_samples == 50
    assert profile.trigger_delay_us == 1000.0


def test_50_ksps_profile_rejects_hwh_without_v2_trigger_delay():
    soccfg = _soccfg("50_ksps")
    del soccfg["ddr4_buf"]["supports_trigger_delay"]

    with pytest.raises(RuntimeError, match="programmable trigger delay"):
        resolve_fir_ddr_profile(soccfg)


def test_unsupported_hwh_rate_is_rejected_explicitly():
    soccfg = _soccfg("1_msps")
    soccfg["ddr4_buf"]["fir_rate_profile"] = "custom"
    soccfg["ddr4_buf"]["stored_sample_rate_hz"] = 100_000.0

    with pytest.raises(RuntimeError, match="unsupported FIR-DDR"):
        resolve_fir_ddr_profile(soccfg)
