"""Hardware RF frequency sweep and FIR-DDR S-parameter acquisition.

This module is intentionally independent of the AWG-tuning waveform path.
The tProcessor advances the normal RF generator DDS and dynamic-readout DDS
registers together, captures one FIR-decimated DDR trace per frequency, and
reduces each trace to one complex I/Q value.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import ceil, isfinite
from numbers import Integral, Real
from pathlib import Path
import json
import sqlite3
import time
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np

try:
    from qick.averager_program import RAveragerProgram
except ImportError as exc:  # keep the waveform editor importable without QICK
    _QICK_IMPORT_ERROR = exc

    class RAveragerProgram:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("QICK is required for RF S-parameter sweeps") from (
                _QICK_IMPORT_ERROR
            )
else:
    _QICK_IMPORT_ERROR = None


MAX_RF_OUTPUT_GAIN = 32766
FILTER_TYPES = ("bypass", "lowpass", "highpass", "bandpass")
FREQUENCY_PARAMETER = "rf_frequency_mhz"
SAMPLE_INDEX_PARAMETER = "sample_index"
I_TRACE_PARAMETER = "i_trace"
Q_TRACE_PARAMETER = "q_trace"
MEAN_I_PARAMETER = "mean_i"
MEAN_Q_PARAMETER = "mean_q"
MAGNITUDE_DB_PARAMETER = "s_parameter_magnitude_db"
PHASE_DEG_PARAMETER = "s_parameter_phase_unwrapped_deg"

ProgressCallback = Callable[[int, str], None]


def _require_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_attenuation(value: Any, name: str) -> float:
    value = _require_finite(value, name)
    if not 0.0 <= value <= 31.75:
        raise ValueError(f"{name} must be in [0, 31.75] dB")
    return value


def _require_filter(value: Any, name: str) -> str:
    value = str(value)
    if value not in FILTER_TYPES:
        raise ValueError(f"{name} must be one of {FILTER_TYPES}")
    return value


def _emit_progress(
    callback: Optional[ProgressCallback], percent: int, message: str
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(percent))), str(message))


@dataclass(frozen=True)
class SParameterSweepConfig:
    """Independent RF output/readout configuration for one hardware sweep."""

    output_ch: int = 0
    readout_ch: int = 0
    frequency_start_mhz: float = 10.0
    frequency_end_mhz: float = 100.0
    frequency_points: int = 101
    gain: int = 20000
    scan_time_us: float = 10.0
    output_att1_db: float = 10.0
    output_att2_db: float = 10.0
    output_filter_type: str = "bypass"
    output_filter_cutoff_ghz: float = 2.5
    output_filter_bandwidth_ghz: float = 1.0
    readout_attenuation_db: float = 20.0
    readout_filter_type: str = "bypass"
    readout_filter_cutoff_ghz: float = 2.5
    readout_filter_bandwidth_ghz: float = 1.0
    nqz: int = 1
    margin_input_samples: int = 1024
    address: int = 0
    stride_bytes: Optional[int] = None
    force_overwrite: bool = False
    settle_seconds: float = 0.05
    trigger_width_tproc_cycles: int = 12
    recovery_tproc_cycles: int = 20

    def __post_init__(self) -> None:
        _require_int(self.output_ch, "output_ch")
        _require_int(self.readout_ch, "readout_ch")
        start = _require_finite(self.frequency_start_mhz, "frequency_start_mhz")
        stop = _require_finite(self.frequency_end_mhz, "frequency_end_mhz")
        points = _require_int(self.frequency_points, "frequency_points", 2)
        if start == stop and points > 1:
            raise ValueError("frequency start and end must differ")
        gain = _require_int(self.gain, "gain")
        if gain > MAX_RF_OUTPUT_GAIN:
            raise ValueError(
                f"gain must not exceed the RF output limit {MAX_RF_OUTPUT_GAIN}"
            )
        _require_finite(self.scan_time_us, "scan_time_us", positive=True)
        _require_attenuation(self.output_att1_db, "output_att1_db")
        _require_attenuation(self.output_att2_db, "output_att2_db")
        _require_attenuation(
            self.readout_attenuation_db, "readout_attenuation_db"
        )
        _require_filter(self.output_filter_type, "output_filter_type")
        _require_filter(self.readout_filter_type, "readout_filter_type")
        _require_finite(
            self.output_filter_cutoff_ghz,
            "output_filter_cutoff_ghz",
            positive=True,
        )
        _require_finite(
            self.output_filter_bandwidth_ghz,
            "output_filter_bandwidth_ghz",
            positive=True,
        )
        _require_finite(
            self.readout_filter_cutoff_ghz,
            "readout_filter_cutoff_ghz",
            positive=True,
        )
        _require_finite(
            self.readout_filter_bandwidth_ghz,
            "readout_filter_bandwidth_ghz",
            positive=True,
        )
        _require_int(self.nqz, "nqz", 1)
        _require_int(self.margin_input_samples, "margin_input_samples")
        _require_int(self.address, "address")
        if self.stride_bytes is not None:
            _require_int(self.stride_bytes, "stride_bytes", 1)
        if not isinstance(self.force_overwrite, bool):
            raise TypeError("force_overwrite must be boolean")
        _require_finite(self.settle_seconds, "settle_seconds")
        if self.settle_seconds < 0.0:
            raise ValueError("settle_seconds must be nonnegative")
        _require_int(
            self.trigger_width_tproc_cycles,
            "trigger_width_tproc_cycles",
            1,
        )
        _require_int(self.recovery_tproc_cycles, "recovery_tproc_cycles")

    @property
    def requested_frequencies_mhz(self) -> np.ndarray:
        return np.linspace(
            self.frequency_start_mhz,
            self.frequency_end_mhz,
            self.frequency_points,
            dtype=float,
        )


@dataclass(frozen=True)
class SParameterSweepResult:
    """FIR DDR traces and derived one-complex-value response per frequency."""

    requested_frequencies_mhz: np.ndarray
    frequencies_mhz: np.ndarray
    iq_traces: np.ndarray
    mean_i: np.ndarray
    mean_q: np.ndarray
    magnitude_db: np.ndarray
    phase_unwrapped_deg: np.ndarray
    sample_rate_hz: float
    reserved_physical_words: Optional[int] = None

    @classmethod
    def from_iq(
        cls,
        requested_frequencies_mhz: Any,
        frequencies_mhz: Any,
        iq_traces: Any,
        *,
        sample_rate_hz: float = 1_000_000.0,
        reserved_physical_words: Optional[int] = None,
    ) -> "SParameterSweepResult":
        requested = np.asarray(requested_frequencies_mhz, dtype=float).reshape(-1)
        frequencies = np.asarray(frequencies_mhz, dtype=float).reshape(-1)
        iq = np.asarray(iq_traces)
        if iq.ndim != 3 or iq.shape[-1] != 2:
            raise ValueError("S-parameter IQ must have shape (frequency, sample, 2)")
        if iq.shape[0] != frequencies.size or requested.size != frequencies.size:
            raise ValueError("frequency and IQ point counts do not match")
        if iq.shape[1] < 1:
            raise ValueError("every frequency must contain at least one IQ sample")
        mean = iq.astype(np.float64).mean(axis=1)
        mean_i = mean[:, 0]
        mean_q = mean[:, 1]
        magnitude = np.hypot(mean_i, mean_q)
        magnitude_db = 20.0 * np.log10(
            np.maximum(magnitude, np.finfo(np.float64).tiny)
        )
        phase = np.degrees(np.unwrap(np.angle(mean_i + 1j * mean_q)))
        return cls(
            requested_frequencies_mhz=requested,
            frequencies_mhz=frequencies,
            iq_traces=np.ascontiguousarray(iq),
            mean_i=np.ascontiguousarray(mean_i),
            mean_q=np.ascontiguousarray(mean_q),
            magnitude_db=np.ascontiguousarray(magnitude_db),
            phase_unwrapped_deg=np.ascontiguousarray(phase),
            sample_rate_hz=_require_finite(
                sample_rate_hz, "sample_rate_hz", positive=True
            ),
            reserved_physical_words=reserved_physical_words,
        )

    @property
    def sample_count(self) -> int:
        return int(self.iq_traces.shape[1])


def _signed_modular_delta(first: int, second: int, width: int) -> int:
    modulus = 1 << int(width)
    delta = (int(second) - int(first)) % modulus
    if delta >= modulus // 2:
        delta -= modulus
    return delta


def _common_frequency_quantum_from_channels(*channels: Mapping[str, Any]) -> float:
    """Reproduce QICK's common DDS step when top-level refclk is absent."""
    if not channels:
        raise ValueError("at least one DDS channel is required")
    reference_clocks = [
        float(channel["f_dds"])
        * int(channel["fdds_div"])
        / int(channel["fs_mult"])
        for channel in channels
    ]
    if not np.allclose(
        reference_clocks,
        reference_clocks[0],
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "generator and readout DDS clocks do not share one reference"
        )
    max_div = int(np.lcm.reduce([
        int(channel["fdds_div"]) for channel in channels
    ]))
    max_bits = max(int(channel["b_dds"]) for channel in channels)
    multipliers = [
        int(channel["fs_mult"])
        * (max_div // int(channel["fdds_div"]))
        * (1 << (max_bits - int(channel["b_dds"])))
        for channel in channels
    ]
    multiplier_lcm = int(np.lcm.reduce(multipliers))
    return (
        reference_clocks[0]
        * multiplier_lcm
        / max_div
        / (1 << max_bits)
    )


def _frequency_to_register(frequency_mhz: float, channel: Mapping[str, Any]) -> int:
    width = int(channel["b_dds"])
    register = round(float(frequency_mhz) * (1 << width) / float(channel["f_dds"]))
    return int(register) % (1 << width)


class SParameterSweepProgram(RAveragerProgram):
    """ASM v1 RF frequency sweep using one tProcessor register-add loop."""

    COUNTER_ADDR = 1
    PERIODIC_WORD_CYCLES = 3
    STOP_WORD_CYCLES = 3
    READOUT_PERIOD_CYCLES = 3

    def __init__(
        self,
        soccfg,
        sweep: SParameterSweepConfig,
        *,
        tproc_mhz: Optional[float] = None,
    ):
        if _QICK_IMPORT_ERROR is not None:
            raise RuntimeError("QICK is required for RF S-parameter sweeps") from (
                _QICK_IMPORT_ERROR
            )
        if not isinstance(sweep, SParameterSweepConfig):
            raise TypeError("sweep must be an SParameterSweepConfig")
        self.sweep = sweep
        hwh_tproc = float(soccfg["tprocs"][0]["f_time"])
        self.tproc_mhz = (
            hwh_tproc
            if tproc_mhz is None
            else _require_finite(tproc_mhz, "tproc_mhz", positive=True)
        )
        super().__init__(soccfg, {
            "reps": 1,
            "expts": sweep.frequency_points,
            "start": sweep.frequency_start_mhz,
            "step": (
                sweep.frequency_end_mhz - sweep.frequency_start_mhz
            ) / (sweep.frequency_points - 1),
        })

    def _validate_hardware(self) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        sweep = self.sweep
        if sweep.output_ch >= len(self.soccfg["gens"]):
            raise IndexError("RF output channel is out of range")
        if sweep.readout_ch >= len(self.soccfg["readouts"]):
            raise IndexError("RF readout channel is out of range")
        gen_cfg = self.soccfg["gens"][sweep.output_ch]
        ro_cfg = self.soccfg["readouts"][sweep.readout_ch]
        if (
            gen_cfg.get("type") == "axis_awg_tuning_v1"
            or gen_cfg.get("gen_type") == "awg_tuning"
        ):
            raise ValueError(
                "RF S-parameter output requires a normal DDS signal generator, "
                "not axis_awg_tuning_v1"
            )
        if not gen_cfg.get("has_dds", True):
            raise ValueError("selected RF output channel does not have a DDS")
        hardware_max = int(gen_cfg.get("maxv", MAX_RF_OUTPUT_GAIN))
        effective_max = min(MAX_RF_OUTPUT_GAIN, hardware_max)
        if self.sweep.gain > effective_max:
            raise ValueError(
                f"gain {self.sweep.gain} exceeds channel {self.sweep.output_ch} "
                f"limit {effective_max}"
            )
        try:
            ddr_cfg = self.soccfg["ddr4_buf"]
        except KeyError as exc:
            raise RuntimeError("soccfg does not contain a DDR sample buffer") from exc
        if not ddr_cfg.get("sample_capture", False):
            raise RuntimeError("S-parameter sweep requires sample-triggered DDR")
        if not ddr_cfg.get("fir_enabled", False):
            raise RuntimeError("S-parameter sweep requires the FIR DDR path")
        output_rate = float(
            ddr_cfg.get(
                "fir_output_fs_mhz",
                ddr_cfg.get("fir_sample_rate_msps", 0.0),
            )
        )
        if not np.isclose(output_rate, 1.0, rtol=0.0, atol=1.0e-9):
            raise RuntimeError(
                f"S-parameter sweep requires 1 MSPS FIR output, got "
                f"{output_rate:.9g} MSPS"
            )
        self._ddr_cfg = ddr_cfg
        self.fir_output_rate_msps = output_rate
        return gen_cfg, ro_cfg

    def _frequency_grid(self, gen_cfg, ro_cfg) -> None:
        requested = self.sweep.requested_frequencies_mhz
        try:
            common_quantum = float(self.soccfg.calc_fstep([gen_cfg, ro_cfg]))
        except (AttributeError, KeyError):
            common_quantum = _common_frequency_quantum_from_channels(
                gen_cfg, ro_cfg
            )
        start = round(float(requested[0]) / common_quantum) * common_quantum
        requested_step = float(requested[1] - requested[0])
        step = round(requested_step / common_quantum) * common_quantum
        if step == 0.0:
            raise ValueError(
                "frequency increment is below the common generator/readout DDS "
                f"resolution ({common_quantum:.12g} MHz)"
            )
        frequencies = start + np.arange(self.sweep.frequency_points) * step
        self.requested_frequencies_mhz = requested
        self.frequencies_mhz = np.ascontiguousarray(frequencies, dtype=float)
        self.frequency_step_mhz = float(step)
        self.frequency_quantum_mhz = common_quantum

        try:
            gen_first = self.freq2reg(
                float(frequencies[0]),
                gen_ch=self.sweep.output_ch,
                ro_ch=self.sweep.readout_ch,
            )
            gen_second = self.freq2reg(
                float(frequencies[1]),
                gen_ch=self.sweep.output_ch,
                ro_ch=self.sweep.readout_ch,
            )
            ro_first = self.freq2reg_adc(
                float(frequencies[0]),
                ro_ch=self.sweep.readout_ch,
                gen_ch=self.sweep.output_ch,
            )
            ro_second = self.freq2reg_adc(
                float(frequencies[1]),
                ro_ch=self.sweep.readout_ch,
                gen_ch=self.sweep.output_ch,
            )
        except KeyError as exc:
            if exc.args != ("refclk_freq",):
                raise
            gen_first = _frequency_to_register(frequencies[0], gen_cfg)
            gen_second = _frequency_to_register(frequencies[1], gen_cfg)
            ro_first = _frequency_to_register(frequencies[0], ro_cfg)
            ro_second = _frequency_to_register(frequencies[1], ro_cfg)
        self._gen_frequency_first = int(gen_first)
        self._ro_frequency_first = int(ro_first)
        self._gen_frequency_delta = _signed_modular_delta(
            gen_first, gen_second, int(gen_cfg["b_dds"])
        )
        self._ro_frequency_delta = _signed_modular_delta(
            ro_first, ro_second, int(ro_cfg["b_dds"])
        )

    def _allocate_step_register(self, page: int) -> int:
        occupied = {0}
        for register_map in (self._gen_regmap, self._ro_regmap):
            for reg_page, register in register_map.values():
                if int(reg_page) == int(page):
                    occupied.add(int(register))
        occupied.update(self._step_registers_by_page.get(int(page), ()))
        if int(page) == 0:
            occupied.update((13, 14, 15, 16))
        for register in range(1, 32):
            if register not in occupied:
                self._step_registers_by_page.setdefault(int(page), set()).add(register)
                return register
        raise RuntimeError(f"no tProcessor work register is available on page {page}")

    def initialize(self) -> None:
        self.tproccfg = dict(self.tproccfg)
        self.tproccfg["f_time"] = self.tproc_mhz
        gen_cfg, ro_cfg = self._validate_hardware()
        self._frequency_grid(gen_cfg, ro_cfg)

        self.declare_gen(
            ch=self.sweep.output_ch,
            nqz=self.sweep.nqz,
        )
        self.scan_samples = max(
            1,
            int(ceil(self.sweep.scan_time_us * self.fir_output_rate_msps)),
        )
        monitor_length = min(
            self.scan_samples,
            int(ro_cfg.get("buf_maxlen", self.scan_samples)),
        )
        self.declare_readout(
            ch=self.sweep.readout_ch,
            length=max(1, monitor_length),
        )

        decimation = int(self._ddr_cfg.get("fir_decimation", 300))
        group_delay = int(
            ceil(float(self._ddr_cfg.get("fir_group_delay_input_samples", 8677)))
        )
        input_fs_mhz = float(self._ddr_cfg.get("fir_input_fs_mhz", 300.0))
        if decimation < 1 or input_fs_mhz <= 0.0:
            raise RuntimeError("invalid FIR decimation metadata")
        self.fir_decimation = decimation
        self.fir_group_delay_input_samples = group_delay
        self.fir_input_fs_mhz = input_fs_mhz
        self.fir_warmup_tproc_cycles = int(
            ceil(group_delay * self.tproc_mhz / input_fs_mhz)
        )
        self.fir_feed_input_samples = (
            self.scan_samples * decimation
            + group_delay
            + self.sweep.margin_input_samples
        )
        capture_cycles = int(
            ceil(self.fir_feed_input_samples * self.tproc_mhz / input_fs_mhz)
        )

        gen_port = int(gen_cfg["tproc_ch"])
        ro_port = int(ro_cfg["tproc_ctrl"])
        self.readout_command_time = 0
        self.output_command_time = 1 if gen_port == ro_port else 0
        self.ddr_trigger_time = (
            self.output_command_time + self.fir_warmup_tproc_cycles
        )
        self.capture_end_tproc_cycles = self.output_command_time + capture_cycles
        self.rf_stop_command_time = self.capture_end_tproc_cycles
        self.point_period_tproc_cycles = (
            self.rf_stop_command_time + self.sweep.recovery_tproc_cycles
        )

        self.default_pulse_registers(
            ch=self.sweep.output_ch,
            freq=self._gen_frequency_first,
            phase=0,
        )
        self.set_readout_registers(
            ch=self.sweep.readout_ch,
            freq=self._ro_frequency_first,
            length=self.READOUT_PERIOD_CYCLES,
            mode="periodic",
            phrst=0,
        )

        self._gen_frequency_page = self.ch_page(self.sweep.output_ch)
        self._gen_frequency_register = self.sreg(self.sweep.output_ch, "freq")
        self._ro_frequency_page = self.ch_page_ro(self.sweep.readout_ch)
        self._ro_frequency_register = self.sreg_ro(self.sweep.readout_ch, "freq")
        self._step_registers_by_page = {}
        self._gen_step_register = self._allocate_step_register(
            self._gen_frequency_page
        )
        self._ro_step_register = self._allocate_step_register(
            self._ro_frequency_page
        )
        self.safe_regwi(
            self._gen_frequency_page,
            self._gen_step_register,
            self._gen_frequency_delta,
            "RF generator DDS frequency step",
        )
        self.safe_regwi(
            self._ro_frequency_page,
            self._ro_step_register,
            self._ro_frequency_delta,
            "RF readout DDS frequency step",
        )
        self.synci(128, "allow RF/readout register initialization")

    def body(self) -> None:
        self.readout(self.sweep.readout_ch, t=self.readout_command_time)

        self.set_pulse_registers(
            ch=self.sweep.output_ch,
            style="const",
            gain=self.sweep.gain,
            length=self.PERIODIC_WORD_CYCLES,
            phrst=0,
            stdysel="zero",
            mode="periodic",
        )
        self.pulse(self.sweep.output_ch, t=self.output_command_time)
        self.trigger(
            ddr4=True,
            adc_trig_offset=self.ddr_trigger_time,
            t=0,
            width=self.sweep.trigger_width_tproc_cycles,
        )

        self.set_pulse_registers(
            ch=self.sweep.output_ch,
            style="const",
            gain=0,
            length=self.STOP_WORD_CYCLES,
            phrst=0,
            stdysel="zero",
            mode="oneshot",
        )
        self.pulse(self.sweep.output_ch, t=self.rf_stop_command_time)
        self.synci(
            self.point_period_tproc_cycles,
            "wait for FIR DDR capture before next frequency",
        )

    def update(self) -> None:
        self.math(
            self._gen_frequency_page,
            self._gen_frequency_register,
            self._gen_frequency_register,
            "+",
            self._gen_step_register,
            "advance RF generator frequency",
        )
        self.math(
            self._ro_frequency_page,
            self._ro_frequency_register,
            self._ro_frequency_register,
            "+",
            self._ro_step_register,
            "advance RF readout frequency",
        )

    def _run_with_counter_progress(
        self,
        soc,
        progress_callback: Callable[[int, int], None],
        *,
        poll_interval_seconds: float = 0.02,
    ) -> None:
        total = int(np.prod(self.loop_dims, dtype=np.int64))
        expected_seconds = (
            total
            * self.point_period_tproc_cycles
            / (self.tproc_mhz * 1_000_000.0)
        )
        timeout_seconds = max(30.0, 5.0 * expected_seconds + 10.0)
        deadline = time.monotonic() + timeout_seconds
        self.config_all(soc, load_envelopes=True, load_mem=False)
        progress_callback(0, total)
        soc.reload_mem()
        soc.clear_tproc_counter(addr=self.counter_addr)
        soc.start_src("internal")
        soc.start_tproc()
        try:
            completed = 0
            while completed < total:
                completed = min(
                    total,
                    int(soc.get_tproc_counter(addr=self.counter_addr)),
                )
                progress_callback(completed, total)
                if completed < total:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "RF S-parameter hardware sweep timed out after "
                            f"{timeout_seconds:.1f} s at {completed}/{total} points"
                        )
                    time.sleep(poll_interval_seconds)
        finally:
            soc.start_src("internal")

    def acquire_fir_ddr(
        self,
        soc,
        *,
        progress: bool = False,
        counter_progress: Optional[Callable[[int, int], None]] = None,
    ) -> SParameterSweepResult:
        n_triggers = self.sweep.frequency_points
        reserved = soc.arm_ddr4_fir_samples(
            ch=self.sweep.readout_ch,
            n_samples=self.scan_samples,
            n_triggers=n_triggers,
            address=self.sweep.address,
            stride_bytes=self.sweep.stride_bytes,
            force_overwrite=self.sweep.force_overwrite,
        )
        if counter_progress is None:
            self.run_rounds(soc, progress=progress)
        else:
            self._run_with_counter_progress(soc, counter_progress)
        if self.sweep.settle_seconds:
            time.sleep(self.sweep.settle_seconds)
        raw = np.asarray(soc.get_ddr4_fir_samples(
            n_samples=self.scan_samples,
            n_triggers=n_triggers,
            start=self.sweep.address,
            stride_bytes=self.sweep.stride_bytes,
        ))
        expected = (n_triggers * self.scan_samples, 2)
        if raw.shape != expected:
            raise RuntimeError(
                f"unexpected FIR DDR IQ shape {raw.shape}; expected {expected}"
            )
        return SParameterSweepResult.from_iq(
            self.requested_frequencies_mhz,
            self.frequencies_mhz,
            raw.reshape(n_triggers, self.scan_samples, 2),
            sample_rate_hz=self.fir_output_rate_msps * 1_000_000.0,
            reserved_physical_words=reserved,
        )

    def get_expt_pts(self) -> np.ndarray:
        return self.frequencies_mhz.copy()

    def summary(self) -> Mapping[str, Any]:
        return {
            "measurement": "rf_s_parameter",
            "awg_tuning_used": False,
            "sweep_execution": "tproc_hardware_register_add",
            "output_ch": self.sweep.output_ch,
            "readout_ch": self.sweep.readout_ch,
            "frequency_points": self.sweep.frequency_points,
            "requested_frequencies_mhz": self.requested_frequencies_mhz.copy(),
            "actual_frequencies_mhz": self.frequencies_mhz.copy(),
            "frequency_step_mhz": self.frequency_step_mhz,
            "common_frequency_quantum_mhz": self.frequency_quantum_mhz,
            "gain": self.sweep.gain,
            "gain_limit": MAX_RF_OUTPUT_GAIN,
            "scan_time_requested_us": self.sweep.scan_time_us,
            "scan_samples": self.scan_samples,
            "scan_time_actual_us": (
                self.scan_samples / self.fir_output_rate_msps
            ),
            "fir_output_rate_msps": self.fir_output_rate_msps,
            "fir_decimation": self.fir_decimation,
            "fir_group_delay_input_samples": self.fir_group_delay_input_samples,
            "fir_warmup_tproc_cycles": self.fir_warmup_tproc_cycles,
            "rf_output_mode": "periodic_start_timed_zero_stop",
            "rf_periodic_word_fabric_cycles": self.PERIODIC_WORD_CYCLES,
            "rf_stop_word_fabric_cycles": self.STOP_WORD_CYCLES,
            "rf_stop_command_tproc_cycle": self.rf_stop_command_time,
            "readout_period_fabric_cycles": self.READOUT_PERIOD_CYCLES,
            "point_period_tproc_cycles": self.point_period_tproc_cycles,
            "estimated_hardware_seconds": (
                self.sweep.frequency_points
                * self.point_period_tproc_cycles
                / (self.tproc_mhz * 1_000_000.0)
            ),
        }


def configure_sparameter_rf_board(
    soc, config: SParameterSweepConfig
) -> Mapping[str, Any]:
    """Apply independent RF output and readout analog-chain settings."""
    output_att = soc.rfb_set_gen_rf(
        config.output_ch,
        config.output_att1_db,
        config.output_att2_db,
    )
    actual_att1, actual_att2 = map(float, output_att)
    soc.rfb_set_gen_filter(
        config.output_ch,
        fc=config.output_filter_cutoff_ghz,
        bw=config.output_filter_bandwidth_ghz,
        ftype=config.output_filter_type,
    )
    input_att = float(
        soc.rfb_set_ro_rf(config.readout_ch, config.readout_attenuation_db)
    )
    soc.rfb_set_ro_filter(
        config.readout_ch,
        fc=config.readout_filter_cutoff_ghz,
        bw=config.readout_filter_bandwidth_ghz,
        ftype=config.readout_filter_type,
    )
    return {
        "output": {
            "channel": config.output_ch,
            "requested_att1_db": config.output_att1_db,
            "requested_att2_db": config.output_att2_db,
            "commanded_att1_db": actual_att1,
            "commanded_att2_db": actual_att2,
            "filter_type": config.output_filter_type,
            "filter_cutoff_ghz": config.output_filter_cutoff_ghz,
            "filter_bandwidth_ghz": config.output_filter_bandwidth_ghz,
        },
        "readout": {
            "channel": config.readout_ch,
            "requested_attenuation_db": config.readout_attenuation_db,
            "commanded_attenuation_db": input_att,
            "filter_type": config.readout_filter_type,
            "filter_cutoff_ghz": config.readout_filter_cutoff_ghz,
            "filter_bandwidth_ghz": config.readout_filter_bandwidth_ghz,
        },
    }


def build_sparameter_program(
    soccfg,
    config: SParameterSweepConfig,
    *,
    tproc_mhz: Optional[float] = None,
) -> SParameterSweepProgram:
    return SParameterSweepProgram(soccfg, config, tproc_mhz=tproc_mhz)


@dataclass
class StoredSParameterSweep:
    run_id: int
    guid: str
    database_path: Path
    row_count: int
    result: SParameterSweepResult
    dataset: Any = None
    program: Any = None
    rf_settings: Optional[Mapping[str, Any]] = None


def _qcodes_helpers():
    try:
        from .qick_qcodes_experiment import (
            _checkpoint_sqlite_database,
            _json_text,
            _prepare_local_database,
            _publish_local_database,
            connect_qick,
        )
    except ImportError:
        from qick_qcodes_experiment import (
            _checkpoint_sqlite_database,
            _json_text,
            _prepare_local_database,
            _publish_local_database,
            connect_qick,
        )
    return (
        connect_qick,
        _json_text,
        _prepare_local_database,
        _checkpoint_sqlite_database,
        _publish_local_database,
    )


def store_sparameter_result(
    result: SParameterSweepResult,
    *,
    config: SParameterSweepConfig,
    connection_config: Any,
    run_config: Any,
    program_summary: Mapping[str, Any],
    rf_settings: Mapping[str, Any],
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Any, int]:
    """Store scalar response and one split I/Q array row per frequency."""
    try:
        from qcodes import (
            Measurement,
            Parameter,
            Station,
            initialise_or_create_database_at,
            load_by_guid,
            load_or_create_experiment,
        )
    except ImportError as exc:
        raise RuntimeError(
            "QCoDeS is required to save S-parameter sweeps; install qcodes==0.58.0"
        ) from exc
    (
        _connect_qick,
        json_text,
        prepare_local_database,
        checkpoint_database,
        publish_database,
    ) = _qcodes_helpers()

    database_path = Path(
        getattr(run_config, "resolved_database_path", run_config.database_path)
    ).expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    staging_directory, local_database_path = prepare_local_database(database_path)
    initialise_or_create_database_at(str(local_database_path))
    experiment = load_or_create_experiment(
        run_config.experiment_name,
        run_config.sample_name,
    )
    measurement = Measurement(exp=experiment, station=Station())
    frequency = Parameter(
        FREQUENCY_PARAMETER,
        label="RF frequency",
        unit="MHz",
    )
    sample_index = Parameter(
        SAMPLE_INDEX_PARAMETER,
        label="FIR sample index",
        unit="",
    )
    mean_i = Parameter(MEAN_I_PARAMETER, label="Mean I", unit="ADC units")
    mean_q = Parameter(MEAN_Q_PARAMETER, label="Mean Q", unit="ADC units")
    magnitude_db = Parameter(
        MAGNITUDE_DB_PARAMETER,
        label="S-parameter magnitude",
        unit="dB",
    )
    phase_deg = Parameter(
        PHASE_DEG_PARAMETER,
        label="S-parameter unwrapped phase",
        unit="deg",
    )
    i_trace = Parameter(I_TRACE_PARAMETER, label="I trace", unit="ADC units")
    q_trace = Parameter(Q_TRACE_PARAMETER, label="Q trace", unit="ADC units")
    measurement.register_parameter(frequency)
    measurement.register_parameter(sample_index, paramtype="array")
    for parameter in (mean_i, mean_q, magnitude_db, phase_deg):
        measurement.register_parameter(parameter, setpoints=(frequency,))
    for parameter in (i_trace, q_trace):
        measurement.register_parameter(
            parameter,
            setpoints=(frequency, sample_index),
            paramtype="array",
        )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": "rf_s_parameter",
        "awg_tuning_used": False,
        "connection": asdict(connection_config),
        "run": {
            **asdict(run_config),
            "database_path": str(database_path),
        },
        "config": asdict(config),
        "rf_settings_actual": rf_settings,
        "program_summary": program_summary,
        "result": {
            "requested_frequencies_mhz": result.requested_frequencies_mhz.tolist(),
            "frequencies_mhz": result.frequencies_mhz.tolist(),
            "mean_i": result.mean_i.tolist(),
            "mean_q": result.mean_q.tolist(),
            "magnitude_db": result.magnitude_db.tolist(),
            "phase_unwrapped_deg": result.phase_unwrapped_deg.tolist(),
            "sample_rate_hz": result.sample_rate_hz,
            "iq_shape": list(result.iq_traces.shape),
        },
        "formulas": {
            "mean_i": "mean(i_trace)",
            "mean_q": "mean(q_trace)",
            "magnitude_db": "20*log10(hypot(mean_i, mean_q))",
            "phase_unwrapped_deg": (
                "degrees(unwrap(angle(mean_i + 1j*mean_q)))"
            ),
        },
        "storage": "one split I/Q array row and scalar response per frequency",
    }

    _emit_progress(progress_callback, 65, "Preparing S-parameter QCoDeS run")
    sample_values = np.arange(result.sample_count, dtype=np.int32)
    with measurement.run(
        write_in_background=False,
        in_memory_cache=False,
    ) as datasaver:
        dataset = datasaver.dataset
        dataset.add_metadata("sparameter_experiment_json", json_text(metadata))
        dataset.add_metadata(
            "sparameter_result_json",
            json_text(metadata["result"]),
        )
        if run_config.notes:
            dataset.add_metadata("experiment_notes", run_config.notes)
        for index, frequency_mhz in enumerate(result.frequencies_mhz):
            datasaver.add_result(
                (frequency, float(frequency_mhz)),
                (mean_i, float(result.mean_i[index])),
                (mean_q, float(result.mean_q[index])),
                (magnitude_db, float(result.magnitude_db[index])),
                (phase_deg, float(result.phase_unwrapped_deg[index])),
                (sample_index, sample_values),
                (i_trace, np.ascontiguousarray(result.iq_traces[index, :, 0])),
                (q_trace, np.ascontiguousarray(result.iq_traces[index, :, 1])),
            )
            percent = 65 + round(25 * (index + 1) / result.frequencies_mhz.size)
            _emit_progress(
                progress_callback,
                percent,
                f"Saving RF sweep point {index + 1}/{result.frequencies_mhz.size}",
            )
        datasaver.flush_data_to_database()

    guid = str(dataset.guid)
    dataset.conn.close()
    _emit_progress(progress_callback, 92, "Checkpointing S-parameter database")
    checkpoint_database(local_database_path)
    _emit_progress(progress_callback, 95, "Publishing S-parameter database")
    publish_database(local_database_path, database_path)
    initialise_or_create_database_at(str(database_path))
    dataset = load_by_guid(guid)
    import shutil

    shutil.rmtree(staging_directory)
    _emit_progress(progress_callback, 99, "S-parameter database saved")
    return dataset, int(result.frequencies_mhz.size * result.sample_count)


def run_sparameter_sweep(
    *,
    connection_config: Any,
    run_config: Any,
    sweep_config: SParameterSweepConfig,
    tproc_mhz: Optional[float] = None,
    connector: Optional[Callable[..., Tuple[Any, Any]]] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> StoredSParameterSweep:
    """Connect, run one RF-only hardware sweep, derive and persist response."""
    connect_qick, *_helpers = _qcodes_helpers()
    _emit_progress(progress_callback, 0, "Starting independent RF S-parameter sweep")
    _emit_progress(progress_callback, 2, "Connecting to QICK")
    soc, soccfg = connect_qick(connection_config, connector=connector)
    _emit_progress(progress_callback, 5, "Configuring RF output/readout chains")
    rf_settings = configure_sparameter_rf_board(soc, sweep_config)
    _emit_progress(progress_callback, 8, "Compiling hardware frequency sweep")
    program = build_sparameter_program(
        soccfg,
        sweep_config,
        tproc_mhz=tproc_mhz,
    )

    def counter_progress(completed: int, total: int) -> None:
        fraction = 1.0 if total <= 0 else completed / total
        _emit_progress(
            progress_callback,
            10 + round(50 * max(0.0, min(1.0, fraction))),
            f"RF hardware sweep {completed}/{total} frequency points",
        )

    result = program.acquire_fir_ddr(
        soc,
        counter_progress=(
            counter_progress if progress_callback is not None else None
        ),
    )
    _emit_progress(progress_callback, 62, "Averaging FIR IQ traces")
    dataset, row_count = store_sparameter_result(
        result,
        config=sweep_config,
        connection_config=connection_config,
        run_config=run_config,
        program_summary=program.summary(),
        rf_settings=rf_settings,
        progress_callback=progress_callback,
    )
    _emit_progress(progress_callback, 100, "RF S-parameter sweep saved")
    return StoredSParameterSweep(
        run_id=int(dataset.run_id),
        guid=str(dataset.guid),
        database_path=Path(
            getattr(run_config, "resolved_database_path", run_config.database_path)
        ).expanduser().resolve(),
        row_count=row_count,
        result=result,
        dataset=dataset,
        program=program,
        rf_settings=rf_settings,
    )


def _coerce_trace_rows(values: Any, point_count: int) -> np.ndarray:
    array = np.asarray(values)
    array = np.squeeze(array)
    if array.dtype == object:
        array = np.stack([np.asarray(value) for value in array.reshape(-1)])
    if point_count == 1 and array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim != 2 and array.size % point_count == 0:
        array = array.reshape(point_count, -1)
    if array.ndim != 2 or array.shape[0] != point_count:
        raise RuntimeError(f"cannot decode stored IQ trace shape {array.shape}")
    return array


def load_sparameter_run(
    database_path: Any,
    run_id: int = 0,
) -> StoredSParameterSweep:
    """Load a saved S-parameter run; ``run_id=0`` selects the latest run."""
    try:
        from qcodes import initialise_or_create_database_at, load_by_id
    except ImportError as exc:
        raise RuntimeError("QCoDeS is required to load S-parameter data") from exc
    path = Path(database_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    initialise_or_create_database_at(str(path))
    run_id = _require_int(run_id, "run_id")
    if run_id == 0:
        with sqlite3.connect(path) as connection:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(runs)")
            }
            row = (
                connection.execute(
                    "SELECT run_id FROM runs "
                    "WHERE sparameter_result_json IS NOT NULL "
                    "AND sparameter_result_json != '' "
                    "ORDER BY run_id DESC LIMIT 1"
                ).fetchone()
                if "sparameter_result_json" in columns
                else None
            )
        if row is None:
            raise RuntimeError(f"database contains no RF S-parameter runs: {path}")
        run_id = int(row[0])
    dataset = load_by_id(run_id)
    try:
        payload_text = dataset.get_metadata("sparameter_result_json")
    except (AttributeError, KeyError):
        payload_text = dataset.metadata.get("sparameter_result_json")
    if not payload_text:
        raise ValueError(f"run {run_id} is not an RF S-parameter run")
    payload = json.loads(payload_text)
    frequencies = np.asarray(payload["frequencies_mhz"], dtype=float)
    requested = np.asarray(
        payload.get("requested_frequencies_mhz", frequencies), dtype=float
    )
    i_data = dataset.get_parameter_data(I_TRACE_PARAMETER)[I_TRACE_PARAMETER]
    q_data = dataset.get_parameter_data(Q_TRACE_PARAMETER)[Q_TRACE_PARAMETER]
    i_trace = _coerce_trace_rows(i_data[I_TRACE_PARAMETER], frequencies.size)
    q_trace = _coerce_trace_rows(q_data[Q_TRACE_PARAMETER], frequencies.size)
    iq = np.stack((i_trace, q_trace), axis=-1)
    result = SParameterSweepResult.from_iq(
        requested,
        frequencies,
        iq,
        sample_rate_hz=float(payload.get("sample_rate_hz", 1_000_000.0)),
    )
    return StoredSParameterSweep(
        run_id=int(dataset.run_id),
        guid=str(dataset.guid),
        database_path=path,
        row_count=int(iq.shape[0] * iq.shape[1]),
        result=result,
        dataset=dataset,
    )


__all__ = [
    "FILTER_TYPES",
    "FREQUENCY_PARAMETER",
    "I_TRACE_PARAMETER",
    "MAGNITUDE_DB_PARAMETER",
    "MAX_RF_OUTPUT_GAIN",
    "MEAN_I_PARAMETER",
    "MEAN_Q_PARAMETER",
    "PHASE_DEG_PARAMETER",
    "Q_TRACE_PARAMETER",
    "SAMPLE_INDEX_PARAMETER",
    "SParameterSweepConfig",
    "SParameterSweepProgram",
    "SParameterSweepResult",
    "StoredSParameterSweep",
    "build_sparameter_program",
    "configure_sparameter_rf_board",
    "load_sparameter_run",
    "run_sparameter_sweep",
    "store_sparameter_result",
]
