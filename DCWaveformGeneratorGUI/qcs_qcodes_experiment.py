"""Keysight QCS execution backend for the fine-tune Experiment workflow.

The existing waveform editor and sweep model are vendor-neutral.  This module
translates a complete :class:`FineTuneSequence` Cartesian sweep into QCS
``Program`` objects. Physical waveform parameters are precomputed in C order.
Ordinary sweeps are paired in one ``program.sweep(...)`` call, following the
QSTL QCS examples. QCS-supported parameters run in an instrument-side hardware
sweep. When only part of a Cartesian grid is hardware-compatible, those axes
remain in a native inner sweep while Python iterates the unsupported outer
coordinates, exposing every completed block for live plotting and Stop-safe
retention. Other unsupported configurations use a QCS-managed software sweep.
M5301 channel offsets are never used to establish a sweep voltage. Compatible
ramps sweep DCWaveform amplitudes around zero; incompatible ramps use fixed
numeric per-point programs, avoiding a measured QCS 2.5.5 waveform-addition
failure.

Stability Diagram retains its specialized two-axis compiler because it also
implements dedicated Bias-T compensation and scan-budget validation.

``keysight.qcs`` is imported lazily so the QICK application remains usable in
environments where the proprietary Keysight package is not installed.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field, replace
import hashlib
from itertools import product
from math import isfinite
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic, sleep
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

try:
    from .dc_waveform_core import (
        DEFAULT_QCS_FULL_SCALE_V,
        DEFAULT_QICK_FULL_SCALE_MV,
    )
    from .qick_fine_tune_sweep import (
        BIAS_T_INSTRUCTION_LEAD_PER_OUTPUT,
        BiasTCompensationConfig,
        FineTuneDdrResult,
        RfDurationSweep,
        RfFrequencySweep,
        RfPowerSweep,
    )
    from .qick_qcodes_experiment import (
        AWG_METADATA_MODE_EXPANDED,
        DEFAULT_AWG_METADATA_MODE,
        IQ_REPETITION_POLICY_COHERENT_AVERAGE,
        IQ_REPETITION_POLICY_PRESERVE,
        QcodesRunConfig,
        build_awg_vertex_metadata,
        build_awg_waveform_recipe,
        normalize_awg_metadata_mode,
        normalize_iq_repetition_policy,
        store_experiment_result,
    )
except ImportError:
    from dc_waveform_core import (
        DEFAULT_QCS_FULL_SCALE_V,
        DEFAULT_QICK_FULL_SCALE_MV,
    )
    from qick_fine_tune_sweep import (
        BIAS_T_INSTRUCTION_LEAD_PER_OUTPUT,
        BiasTCompensationConfig,
        FineTuneDdrResult,
        RfDurationSweep,
        RfFrequencySweep,
        RfPowerSweep,
    )
    from qick_qcodes_experiment import (
        AWG_METADATA_MODE_EXPANDED,
        DEFAULT_AWG_METADATA_MODE,
        IQ_REPETITION_POLICY_COHERENT_AVERAGE,
        IQ_REPETITION_POLICY_PRESERVE,
        QcodesRunConfig,
        build_awg_vertex_metadata,
        build_awg_waveform_recipe,
        normalize_awg_metadata_mode,
        normalize_iq_repetition_policy,
        store_experiment_result,
    )


ProgressCallback = Callable[[int, str], None]
EventCallback = Callable[[str, str, str], None]
PartialResultCallback = Callable[[Any], None]
MAX_QCS_SOFTWARE_SWEEP_POINTS = 10_000
MAX_QCS_HARDWARE_SWEEP_ARRAY_VALUES = 24_576
MAX_QCS_HARDWARE_SWEEP_ARRAYS_PER_CHANNEL = 8
# QCS 2.5.5 requires the per-channel sum to be strictly less than 24,576.
MAX_QCS_STABILITY_GRID_POINTS = MAX_QCS_HARDWARE_SWEEP_ARRAY_VALUES - 1
MAX_QCS_STABILITY_RESULT_VALUES = 2_000_000
# The M5000-series synchronization fabric runs at 300 MHz. HCL represents
# ``init_time`` as an integer number of nanoseconds, so only every third
# fabric cycle (10 ns) can be represented without truncation.
QCS_FABRIC_CLOCK_HZ = 300_000_000.0
QCS_INIT_TIME_QUANTUM_NS = int(
    round(3 * 1e9 / QCS_FABRIC_CLOCK_HZ)
)
# QCS itself defaults HclBackend.init_time to 100 us.  That conservative
# library default dominates a video-mode hardware sweep, however.  The QSTL
# QCSVideoProcessor reference uses 70 ns, which is also exactly seven of the
# 10 ns HCL/fabric-safe quanta above.
DEFAULT_QCS_INIT_TIME_S = 70.0e-9
# Keysight QCS 2.5.5 ``SAMPLE_RATES`` defines the M5200 digitizer at
# 4.8 GSa/s. Integration-filter acquisitions are emitted in 16-sample blocks.
QCS_M5200_SAMPLE_RATE_HZ = 4_800_000_000.0
QCS_M5200_INTEGRATION_BLOCK_SAMPLES = 16
# Stability drives an RF waveform for the same interval as its M5200
# IntegrationFilter.  The synchronized pair therefore uses the stricter
# two-fabric-cycle waveform quantum rather than the digitizer-only one-cycle
# integration quantum: 6.666666... ns = 32 M5200 samples.
QCS_STABILITY_INTEGRATION_FABRIC_CYCLES = 2
QCS_STABILITY_INTEGRATION_BLOCK_SAMPLES = int(
    round(
        QCS_STABILITY_INTEGRATION_FABRIC_CYCLES
        * QCS_M5200_SAMPLE_RATE_HZ
        / QCS_FABRIC_CLOCK_HZ
    )
)
QCS_STABILITY_INTEGRATION_QUANTUM_S = (
    QCS_STABILITY_INTEGRATION_FABRIC_CYCLES / QCS_FABRIC_CLOCK_HZ
)
# The connected QCS 2.5.5 M5200 sandbox accepts one 32,768-sample
# IntegrationFilter (6.826666... us) and rejects the next legal 16-sample
# block with ``AllocateEngines``. Longer averages therefore use repeated equal
# filters at or below that ceiling. Although QCS documents a 10 ns minimum,
# inferred timing at that exact boundary serialized below the HCL threshold
# on the connected system; a hardware-verified 20 ns guard is used between
# acquisitions. The resulting hardware-demodulated complex values are
# combined with sample-count weights. An exact 100 us request uses 15 repeats
# of one unchanged 32,000-sample filter object.
QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES = 32_768
QCS_M5200_MAX_SINGLE_INTEGRATION_DURATION_S = (
    QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES / QCS_M5200_SAMPLE_RATE_HZ
)
# A 100 us aggregate average made from repeated identical flat filters was
# verified on the connected QCS 2.5.5 M5200A. Keep the production S-parameter
# editor within that measured aggregate envelope rather than presenting an
# arbitrary limit.
QCS_SPARAMETER_INTER_SEGMENT_DELAY_S = 20.0e-9
QCS_SPARAMETER_MAX_INTEGRATION_SAMPLES = 480_000
QCS_SPARAMETER_MAX_INTEGRATION_DURATION_S = (
    QCS_SPARAMETER_MAX_INTEGRATION_SAMPLES / QCS_M5200_SAMPLE_RATE_HZ
)
# A long Single-I/Q request is evaluated as repeated, independently safe
# hardware passes.  One pass retains the 100 us aggregate integration that
# was verified on the connected M5200A; the application-level total is capped
# at 100 ms so an accidental unit error cannot submit an unbounded number of
# programs.
QCS_MAX_TOTAL_IQ_AVERAGING_SAMPLES = 480_000_000
QCS_MAX_TOTAL_IQ_AVERAGING_DURATION_S = (
    QCS_MAX_TOTAL_IQ_AVERAGING_SAMPLES / QCS_M5200_SAMPLE_RATE_HZ
)
# Stability uses the same connected-M5200 segmented-integration contract as
# RF S-parameter acquisition.  Keep separate public names so each GUI can
# describe its own measurement without coupling user-facing terminology.
QCS_STABILITY_INTER_SEGMENT_DELAY_S = QCS_SPARAMETER_INTER_SEGMENT_DELAY_S
QCS_STABILITY_MAX_INTEGRATION_SAMPLES = (
    QCS_SPARAMETER_MAX_INTEGRATION_SAMPLES
)
QCS_STABILITY_MAX_INTEGRATION_DURATION_S = (
    QCS_STABILITY_MAX_INTEGRATION_SAMPLES / QCS_M5200_SAMPLE_RATE_HZ
)
# Raw M5200 captures have no documented QCS 2.5.5 hardware ceiling.  Keep a
# separate, explicitly application-owned payload limit so an accidental long
# Noise Analysis duration cannot allocate or transfer an unbounded trace.
# At the standard 4.8 GSa/s M5200 rate this is 2.083333... ms per repetition.
QCS_NOISE_MAX_RAW_TRACE_SAMPLES = 10_000_000
# A fixed nonzero plateau is established with a 1 us M5301 ``DCWaveform`` and
# retains that sample with ``Hold`` for the remainder.  Although the client-side
# renderer accepts a four-fabric-cycle seed, the connected HCL execution path
# can lower that minimum seed to a negative internal Delay.  A 300-cycle seed
# is still small compared with the 40.960 us rendered-waveform budget and is
# the hardware-safe construction used for long AWG Tuning plateaus.
QCS_M5301_HOLD_SEED_FABRIC_CYCLES = 300
QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES = 4
QCS_M5301_WAVEFORM_GRANULARITY_FABRIC_CYCLES = 2
# The connected HCL sandbox accepts an aggregate 98,304 rendered M5301
# samples per channel (40.960 us at 2.4 GSa/s) and rejects the next valid
# 16-sample increment. Multiple DCWaveforms share this same buffer; splitting
# a ramp therefore does not increase the executable ramp duration.
QCS_M5301_MAX_RENDERED_FABRIC_CYCLES = 12_288
QCS_M5301_SAMPLES_PER_FABRIC_CYCLE = 8
QCS_M5301_MAX_RENDERED_SAMPLES = (
    QCS_M5301_MAX_RENDERED_FABRIC_CYCLES
    * QCS_M5301_SAMPLES_PER_FABRIC_CYCLE
)
# A Stability target begins with one minimum M5301 ramp, then uses Hold, and
# ends with one minimum ramp plus one minimum explicit-zero interval.  The
# complete target therefore needs three minimum intervals in addition to the
# user-visible full-level settle/acquire/guard window.  Keeping these values
# public lets the GUI show the exact schedule that the compiler emits.
QCS_STABILITY_DC_RAMP_FABRIC_CYCLES = (
    QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES
)
QCS_STABILITY_DC_EDGE_PADDING_FABRIC_CYCLES = (
    3 * QCS_STABILITY_DC_RAMP_FABRIC_CYCLES
)
QCS_STABILITY_DC_RAMP_S = (
    QCS_STABILITY_DC_RAMP_FABRIC_CYCLES / QCS_FABRIC_CLOCK_HZ
)
QCS_STABILITY_DC_EDGE_PADDING_S = (
    QCS_STABILITY_DC_EDGE_PADDING_FABRIC_CYCLES / QCS_FABRIC_CLOCK_HZ
)
# QCS 2.5.5 documents ``BasebandAWGChannelSettings.offset`` as a fraction of
# full scale, but the connected M5301A loopback measured approximately one
# output volt per Scalar unit. Keep this conversion separate from the 2.5 V
# DCWaveform amplitude scale. The Scalar's documented limits remain +/-1.5.
QCS_M5301_OFFSET_VOLTS_PER_SCALAR = 1.0
QCS_M5301_MAX_ABS_OFFSET_SCALAR = 1.5


class QcsUnsupportedFeatureError(ValueError):
    """Raised when a QICK-only semantic cannot be translated safely."""


@dataclass(frozen=True)
class QcsIqAveragingPlan:
    """Quantized bounded-pass plan for one requested integrated-I/Q value."""

    requested_total_duration_s: float
    quantized_total_duration_s: float
    quantized_total_sample_count: int
    pass_sample_counts: Tuple[int, ...]
    sample_rate_hz: float
    block_samples: int
    max_pass_samples: int

    @property
    def pass_count(self) -> int:
        return len(self.pass_sample_counts)

    @property
    def max_pass_duration_s(self) -> float:
        return self.max_pass_samples / self.sample_rate_hz


def plan_qcs_total_iq_averaging(
    duration_s: Any,
    *,
    sample_rate_hz: float = QCS_M5200_SAMPLE_RATE_HZ,
    block_samples: int = QCS_M5200_INTEGRATION_BLOCK_SAMPLES,
    max_pass_samples: int = QCS_SPARAMETER_MAX_INTEGRATION_SAMPLES,
) -> QcsIqAveragingPlan:
    """Plan equal, reusable hardware passes for up to 100 ms of averaging.

    Equal pass sizes let one compiled QCS Program be reused for every pass.
    The requested total is rounded upward to the M5200 block grid and, when
    multiple passes are needed, by at most one block per pass so no pass is
    shorter than the requested aggregate integration.
    """

    requested_s = _positive_finite(
        duration_s, "QCS total I/Q averaging duration"
    )
    rate_hz = _positive_finite(
        sample_rate_hz, "QCS total I/Q averaging sample rate"
    )
    if isinstance(block_samples, bool):
        raise TypeError("QCS I/Q averaging block size must be an integer")
    block = int(block_samples)
    if block < 1 or block != block_samples:
        raise ValueError("QCS I/Q averaging block size must be positive")
    if isinstance(max_pass_samples, bool):
        raise TypeError("QCS I/Q averaging pass limit must be an integer")
    pass_cap = int(max_pass_samples)
    if pass_cap < block or pass_cap != max_pass_samples:
        raise ValueError(
            "QCS I/Q averaging pass limit must be a positive integer"
        )
    if pass_cap % block:
        raise ValueError(
            "QCS I/Q averaging pass limit must be a multiple of the "
            f"{block}-sample block"
        )
    if requested_s > QCS_MAX_TOTAL_IQ_AVERAGING_DURATION_S + 1.0e-15:
        raise QcsUnsupportedFeatureError(
            "QCS total Single-I/Q averaging is limited to "
            f"{QCS_MAX_TOTAL_IQ_AVERAGING_DURATION_S * 1.0e3:.9g} ms "
            f"({QCS_MAX_TOTAL_IQ_AVERAGING_SAMPLES:,} M5200 samples); "
            f"requested {requested_s * 1.0e3:.9g} ms"
        )
    rendered_samples = requested_s * rate_hz
    nearest_samples = int(round(rendered_samples))
    tolerance = max(1.0e-7, 8.0 * abs(float(np.spacing(rendered_samples))))
    if np.isclose(
        rendered_samples,
        nearest_samples,
        rtol=0.0,
        atol=tolerance,
    ):
        requested_samples = max(1, nearest_samples)
    else:
        requested_samples = max(1, int(np.ceil(rendered_samples)))
    block_count = max(1, int(np.ceil(requested_samples / block)))
    quantized_samples = block_count * block
    if quantized_samples > QCS_MAX_TOTAL_IQ_AVERAGING_SAMPLES:
        raise QcsUnsupportedFeatureError(
            "QCS total Single-I/Q averaging quantizes above the 100 ms "
            f"application limit ({quantized_samples:,} samples)"
        )
    pass_count = max(1, int(np.ceil(quantized_samples / pass_cap)))
    samples_per_pass = int(
        np.ceil(quantized_samples / pass_count / block) * block
    )
    if samples_per_pass > QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES:
        filter_count = int(
            np.ceil(
                samples_per_pass
                / QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES
            )
        )
        samples_per_filter = int(
            np.ceil(samples_per_pass / filter_count / block) * block
        )
        samples_per_pass = samples_per_filter * filter_count
    if samples_per_pass > pass_cap:
        raise RuntimeError("internal QCS I/Q averaging pass exceeds its cap")
    pass_samples = (samples_per_pass,) * pass_count
    actual_total_samples = int(sum(pass_samples))
    if actual_total_samples > QCS_MAX_TOTAL_IQ_AVERAGING_SAMPLES:
        raise QcsUnsupportedFeatureError(
            "QCS total Single-I/Q averaging equal-pass quantization exceeds "
            "the 100 ms application limit"
        )
    return QcsIqAveragingPlan(
        requested_total_duration_s=requested_s,
        quantized_total_duration_s=actual_total_samples / rate_hz,
        quantized_total_sample_count=actual_total_samples,
        pass_sample_counts=pass_samples,
        sample_rate_hz=rate_hz,
        block_samples=block,
        max_pass_samples=pass_cap,
    )


def _validate_qcs_software_sweep_point_count(point_count: int) -> None:
    """Keep the host-expanded fallback bounded without capping hardware sweeps."""

    count = int(point_count)
    if count > MAX_QCS_SOFTWARE_SWEEP_POINTS:
        raise QcsUnsupportedFeatureError(
            f"This QCS sweep has {count:,} Cartesian points and requires "
            "software execution. QCS software sweeps are limited to "
            f"{MAX_QCS_SOFTWARE_SWEEP_POINTS:,} points; change the waveform "
            "until the Hardware sweep indicator is shown, or reduce the grid."
        )


class QcsExperimentCancelled(RuntimeError):
    """Raised after a user-requested QCS stop reaches a safe boundary.

    ``partial_result`` contains every fully acquired mixed-sweep block.  An
    active inner program is atomic: when it is aborted, only that incomplete
    block is discarded. ``stored_result`` is populated by the QCoDeS wrapper
    when at least one completed block can be persisted before returning Stop
    to the GUI.
    """

    def __init__(
        self,
        message: str,
        *,
        partial_result: Optional[Any] = None,
        stored_result: Optional[Any] = None,
    ) -> None:
        super().__init__(str(message))
        self.partial_result = partial_result
        self.stored_result = stored_result


class QcsCancellationController:
    """Coordinate a GUI stop request with one blocking QCS executor.

    QCS exposes cancellation on :class:`HclBackend`, while ``Executor.execute``
    blocks the worker thread.  This controller therefore owns a thread-safe
    cooperative flag and, once the active mapper is known, uses a short-lived
    control backend on a daemon thread to abort only programs carrying this
    run's unique name tag.
    """

    _PENDING_STATES = {"queued", "compiling", "running", "paused"}

    def __init__(
        self,
        *,
        status_callback: Optional[Callable[[str], None]] = None,
        program_name_tag: Optional[str] = None,
    ) -> None:
        self._stop_event = Event()
        self._finished_event = Event()
        self._abort_complete = Event()
        self._lock = Lock()
        self._qcs_module = None
        self._mapper = None
        self._abort_thread: Optional[Thread] = None
        self._status_callback = status_callback
        self._accepting_stop_requests = True
        token = str(program_name_tag or uuid4().hex[:12]).strip()
        if not token:
            raise ValueError("QCS cancellation program tag must not be empty")
        self.program_name_tag = f"qcs-gui-{token}"

    def is_stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self) -> bool:
        """Request cooperative cancellation and start an HCL abort watcher."""
        with self._lock:
            if not self._accepting_stop_requests:
                accepted = False
                first_request = False
            else:
                accepted = True
                first_request = not self._stop_event.is_set()
                self._stop_event.set()
        if not accepted:
            self._notify(
                "QCS hardware execution has already completed; finalizing "
                "the acquired result"
            )
            return False
        if first_request:
            self._notify("QCS stop requested; locating the active program")
        self._start_abort_thread_if_ready()
        return first_request

    def close_stop_window(self) -> bool:
        """Atomically close cancellation after hardware result processing.

        Returns ``False`` when a stop request won the race. The executor then
        performs its emergency DC reset and raises cancellation instead of
        returning a successful result.
        """
        with self._lock:
            stop_requested = self._stop_event.is_set()
            self._accepting_stop_requests = False
        return not stop_requested

    def bind(self, qcs_module: Any, mapper: Any) -> None:
        """Expose the active mapper to the independent abort client."""
        with self._lock:
            self._qcs_module = qcs_module
            self._mapper = mapper
        if self.is_stop_requested():
            self._start_abort_thread_if_ready()

    def mark_finished(self) -> None:
        self._finished_event.set()

    def wait_for_abort(self, timeout: Optional[float] = None) -> bool:
        """Wait for the asynchronous abort lookup; primarily useful in tests."""
        return self._abort_complete.wait(timeout)

    def tag_program(self, program: Any) -> None:
        """Add this run's unique tag to the QCS Program description."""
        marker = f"[{self.program_name_tag}]"
        current = str(getattr(program, "name", "") or "PulseGenerator")
        if marker not in current:
            program.name = f"{current} {marker}"

    def raise_if_requested(self, stage: str) -> None:
        if self.is_stop_requested():
            raise QcsExperimentCancelled(
                f"QCS experiment stopped by user during {str(stage).strip()}"
            )

    def _notify(self, message: str) -> None:
        callback = self._status_callback
        if callback is not None:
            try:
                callback(str(message))
            except Exception:
                # Status reporting must never prevent hardware cancellation.
                pass

    def _start_abort_thread_if_ready(self) -> None:
        with self._lock:
            if (
                self._qcs_module is None
                or self._mapper is None
                or (
                    self._abort_thread is not None
                    and self._abort_thread.is_alive()
                )
                or self._abort_complete.is_set()
            ):
                return
            thread = Thread(
                target=self._abort_matching_pending_programs,
                name=f"{self.program_name_tag}-abort",
                daemon=True,
            )
            self._abort_thread = thread
        thread.start()

    @staticmethod
    def _pending_field(item: Mapping[str, Any], *names: str) -> Any:
        for name in names:
            if name in item:
                return item[name]
        return None

    def _abort_matching_pending_programs(self) -> None:
        try:
            with self._lock:
                qcs = self._qcs_module
                mapper = self._mapper
            backend = qcs.HclBackend(channel_mapper=mapper)
            aborted_ids = []
            rejected_ids = set()
            # A stop can arrive after the final cooperative check but before
            # ExecuteProgram has registered the accession with QCS. Poll for
            # a meaningful interval and also make at least five queries when
            # an authenticated inventory call itself is slow.
            deadline = monotonic() + 15.0
            attempt = 0
            while not self._finished_event.is_set():
                attempt += 1
                if self._finished_event.is_set():
                    break
                pending = backend.get_pending_programs_info()
                matches = []
                for raw_item in pending or ():
                    if not isinstance(raw_item, Mapping):
                        continue
                    description = str(
                        self._pending_field(
                            raw_item,
                            "Description",
                            "description",
                        )
                        or ""
                    )
                    state = str(
                        self._pending_field(raw_item, "State", "state") or ""
                    ).strip().lower()
                    accession_id = self._pending_field(
                        raw_item,
                        "AccessionId",
                        "accession_id",
                    )
                    if (
                        self.program_name_tag in description
                        and state in self._PENDING_STATES
                        and accession_id is not None
                    ):
                        matches.append((int(accession_id), state))
                if matches:
                    for accession_id, state in matches:
                        accepted = bool(backend.abort_program(accession_id))
                        if accepted:
                            aborted_ids.append(accession_id)
                            self._notify(
                                f"QCS accepted the abort for program "
                                f"#{accession_id} ({state})"
                            )
                        else:
                            rejected_ids.add(accession_id)
                    if aborted_ids:
                        break
                if attempt >= 5 and monotonic() >= deadline:
                    break
                sleep(0.1)
            if not aborted_ids and not self._finished_event.is_set():
                if rejected_ids:
                    self._notify(
                        "QCS did not accept the abort request; waiting for "
                        "the active execution to reach a safe boundary"
                    )
                else:
                    self._notify(
                        "No matching active QCS program was found; stopping "
                        "at the next safe execution boundary"
                    )
        except Exception as exc:
            if not self._finished_event.is_set():
                self._notify(
                    "QCS abort request could not be sent immediately; "
                    "stopping at the next safe boundary "
                    f"({type(exc).__name__}: {exc})"
                )
        finally:
            self._abort_complete.set()


def _positive_finite(value: Any, label: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _nonnegative_finite(value: Any, label: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be nonnegative and finite")
    return result


def _canonical_qcs_init_time(value: Any) -> float:
    """Return an HCL-safe inter-iteration delay on the 300 MHz fabric grid.

    QCS 2.5.5 converts this float with ``int(init_time * 1e9)``. Values
    produced by GUI unit conversion can otherwise lose one nanosecond (for
    example, 100 us can become 99,999 ns). A 10 ns quantum is both exactly
    representable by HCL's integer-nanosecond field and aligned to three
    300 MHz fabric cycles. ``nextafter`` protects the subsequent truncation.
    """
    seconds = _nonnegative_finite(value, "QCS inter-iteration delay")
    requested_quanta = seconds * 1e9 / QCS_INIT_TIME_QUANTUM_NS
    nearest_quanta = int(round(requested_quanta))
    if np.isclose(
        requested_quanta,
        nearest_quanta,
        rtol=0.0,
        atol=1e-9,
    ):
        quantum_count = nearest_quanta
    else:
        # This is an inter-iteration hold, so alignment must not
        # make it shorter than the value requested by the user.
        quantum_count = int(np.ceil(requested_quanta))
    nanoseconds = quantum_count * QCS_INIT_TIME_QUANTUM_NS
    if nanoseconds == 0:
        return 0.0
    return float(np.nextafter(nanoseconds / 1e9, np.inf))


def quantize_qcs_inter_iteration_delay(value_s: Any) -> float:
    """Return the effective HCL inter-iteration delay in seconds."""

    return _canonical_qcs_init_time(value_s)


def quantize_qcs_raw_trace_duration(
    duration_s: Any,
    sample_rate_hz: Any = QCS_M5200_SAMPLE_RATE_HZ,
) -> tuple[float, int]:
    """Round a raw M5200 acquisition upward to one whole sample.

    Raw acquisitions do not use an ``IntegrationFilter`` and therefore do
    not inherit its 16-sample block constraint.  The separate
    :data:`QCS_NOISE_MAX_RAW_TRACE_SAMPLES` ceiling is an application payload
    limit, not a claimed M5200 hardware limit.
    """

    requested_s = _positive_finite(
        duration_s, "QCS raw-trace measurement duration"
    )
    rate_hz = _positive_finite(
        sample_rate_hz, "QCS raw-trace M5200 sample rate"
    )
    requested_samples = requested_s * rate_hz
    nearest_samples = int(round(requested_samples))
    tolerance = max(
        1.0e-9,
        8.0 * abs(float(np.spacing(requested_samples))),
    )
    if np.isclose(
        requested_samples,
        nearest_samples,
        rtol=0.0,
        atol=tolerance,
    ):
        sample_count = max(1, nearest_samples)
    else:
        sample_count = max(1, int(np.ceil(requested_samples)))
    if sample_count > QCS_NOISE_MAX_RAW_TRACE_SAMPLES:
        maximum_s = QCS_NOISE_MAX_RAW_TRACE_SAMPLES / rate_hz
        raise ValueError(
            "QCS Noise Analysis raw trace exceeds the application limit of "
            f"{QCS_NOISE_MAX_RAW_TRACE_SAMPLES:,} samples "
            f"({maximum_s:.12g} s at {rate_hz:g} S/s)"
        )
    return sample_count / rate_hz, sample_count


def _fabric_aligned_seconds(
    value: Any,
    *,
    fabric_hz: float,
    label: str,
    positive: bool = False,
) -> float:
    """Round a QCS program time to the nearest synchronization cycle."""
    seconds = (
        _positive_finite(value, label)
        if positive
        else _nonnegative_finite(value, label)
    )
    cycles = int(round(seconds * fabric_hz))
    if positive and cycles < 1:
        raise ValueError(
            f"{label} must be at least one QCS fabric-clock period"
        )
    return cycles / fabric_hz


def quantize_qcs_stability_integration_duration(
    duration_s: Any,
) -> tuple[float, int]:
    """Round Stability integration upward to its synchronized time quantum.

    The small near-integer tolerance accepts a decimal display of the
    repeating 6.666666... ns period without accidentally adding a second
    block.  Values materially between two blocks are always rounded upward,
    so the programmed acquisition is never shorter than requested.
    """

    requested_s = _positive_finite(
        duration_s,
        "QCS Stability integration duration",
    )
    requested_blocks = requested_s / QCS_STABILITY_INTEGRATION_QUANTUM_S
    nearest_blocks = int(round(requested_blocks))
    if np.isclose(
        requested_blocks,
        nearest_blocks,
        rtol=0.0,
        atol=1.0e-7,
    ):
        block_count = max(1, nearest_blocks)
    else:
        block_count = max(1, int(np.ceil(requested_blocks)))
    requested_sample_count = (
        block_count * QCS_STABILITY_INTEGRATION_BLOCK_SAMPLES
    )
    plan = plan_qcs_total_iq_averaging(
        requested_sample_count / QCS_M5200_SAMPLE_RATE_HZ,
        sample_rate_hz=QCS_M5200_SAMPLE_RATE_HZ,
        block_samples=QCS_STABILITY_INTEGRATION_BLOCK_SAMPLES,
        max_pass_samples=QCS_STABILITY_MAX_INTEGRATION_SAMPLES,
    )
    return (
        plan.quantized_total_duration_s,
        plan.quantized_total_sample_count,
    )


def qcs_stability_integration_segment_sample_counts(
    sample_count: Any,
) -> tuple[int, ...]:
    """Split one Stability average into equal legal M5200 filters.

    Every segment uses the same sample count whenever more than one filter is
    required.  Reusing one IntegrationFilter object is the connected-QCS
    hardware-tested construction and avoids an additional filter allocation
    for a small final remainder.
    """

    if isinstance(sample_count, bool):
        raise TypeError("QCS Stability integration sample count must be integer")
    count = int(sample_count)
    if count < 1 or count != sample_count:
        raise ValueError(
            "QCS Stability integration sample count must be a positive integer"
        )
    block = QCS_STABILITY_INTEGRATION_BLOCK_SAMPLES
    if count % block:
        raise ValueError(
            "QCS Stability integration sample count must be a multiple of "
            f"{block}; got {count}"
        )
    if count > QCS_STABILITY_MAX_INTEGRATION_SAMPLES:
        raise QcsUnsupportedFeatureError(
            "QCS Stability integration exceeds the verified aggregate limit "
            f"of {QCS_STABILITY_MAX_INTEGRATION_SAMPLES:,} samples"
        )
    if count <= QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES:
        return (count,)
    segment_count = int(
        np.ceil(count / QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES)
    )
    segment_sample_count = int(
        np.ceil(count / segment_count / block) * block
    )
    if segment_sample_count > QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES:
        raise RuntimeError(
            "internal QCS Stability segment exceeds the connected M5200 "
            "single-IntegrationFilter ceiling"
        )
    return (segment_sample_count,) * segment_count


def _qcs_iq_integration_segment_sample_counts(
    sample_count: Any,
    *,
    block_samples: int = QCS_M5200_INTEGRATION_BLOCK_SAMPLES,
) -> tuple[int, ...]:
    """Split one verified pass into reusable legal IntegrationFilters."""

    if isinstance(sample_count, bool):
        raise TypeError("QCS integration sample count must be an integer")
    count = int(sample_count)
    block = int(block_samples)
    if count < 1 or count != sample_count:
        raise ValueError("QCS integration sample count must be positive")
    if block < 1 or count % block:
        raise ValueError(
            "QCS integration sample count must be a multiple of "
            f"{block}; got {count}"
        )
    if count > QCS_SPARAMETER_MAX_INTEGRATION_SAMPLES:
        raise QcsUnsupportedFeatureError(
            "one QCS integrated-I/Q pass exceeds the verified aggregate "
            f"limit of {QCS_SPARAMETER_MAX_INTEGRATION_SAMPLES:,} samples"
        )
    if count <= QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES:
        return (count,)
    segment_count = int(
        np.ceil(count / QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES)
    )
    segment_samples = int(np.ceil(count / segment_count / block) * block)
    if segment_samples > QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES:
        raise RuntimeError(
            "internal QCS segment exceeds the connected M5200 "
            "single-IntegrationFilter ceiling"
        )
    return (segment_samples,) * segment_count


def _channel_name(value: Any, label: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


@dataclass(frozen=True)
class QcsConnectionConfig:
    """Serialized mapper and HCL execution settings."""

    mapper_path: str
    dc_channel_names: Tuple[str, ...]
    mapper_sha256: Optional[str] = None
    dc_full_scale_v: float = DEFAULT_QCS_FULL_SCALE_V
    rf_channel_names: Mapping[int, str] = field(default_factory=dict)
    acquisition_channel_name: Optional[str] = None
    hw_demod: bool = True
    init_time_s: float = DEFAULT_QCS_INIT_TIME_S
    blocking: bool = True

    def __post_init__(self) -> None:
        mapper_path = str(self.mapper_path).strip()
        if not mapper_path:
            raise ValueError("QCS ChannelMapper file path must not be empty")
        mapper_sha256 = self.mapper_sha256
        if mapper_sha256 is not None:
            mapper_sha256 = str(mapper_sha256).strip().lower()
            if (
                len(mapper_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in mapper_sha256
                )
            ):
                raise ValueError(
                    "QCS mapper_sha256 must contain 64 hexadecimal characters"
                )
        dc_names = tuple(
            _channel_name(value, "QCS DC channel name")
            for value in self.dc_channel_names
        )
        if not dc_names:
            raise ValueError("at least one QCS DC channel name is required")
        if len(set(dc_names)) != len(dc_names):
            raise ValueError("QCS DC channel names must be unique")
        dc_full_scale_v = _positive_finite(
            self.dc_full_scale_v, "QCS DC full scale"
        )
        rf_names = {}
        for raw_channel, raw_name in dict(self.rf_channel_names).items():
            if isinstance(raw_channel, bool):
                raise TypeError("QCS RF generator numbers must be integers")
            channel = int(raw_channel)
            if channel < 0 or channel != raw_channel:
                raise ValueError(
                    "QCS RF generator numbers must be nonnegative integers"
                )
            rf_names[channel] = _channel_name(
                raw_name, "QCS RF channel name"
            )
        acquisition_name = self.acquisition_channel_name
        if acquisition_name is not None:
            acquisition_name = _channel_name(
                acquisition_name, "QCS acquisition channel name"
            )
        if not isinstance(self.hw_demod, bool):
            raise TypeError("QCS hw_demod must be boolean")
        if not isinstance(self.blocking, bool):
            raise TypeError("QCS blocking must be boolean")
        init_time_s = _canonical_qcs_init_time(self.init_time_s)
        object.__setattr__(self, "mapper_path", mapper_path)
        object.__setattr__(self, "mapper_sha256", mapper_sha256)
        object.__setattr__(self, "dc_channel_names", dc_names)
        object.__setattr__(self, "dc_full_scale_v", dc_full_scale_v)
        object.__setattr__(self, "rf_channel_names", rf_names)
        object.__setattr__(
            self, "acquisition_channel_name", acquisition_name
        )
        object.__setattr__(self, "init_time_s", init_time_s)


@dataclass(frozen=True)
class QcsRfPowerCalibrationConfig:
    """One fixed M5300A connector-power request for a QCS RF pulse."""

    database_path: str
    target_power_dbm: float
    run_id: int = 0

    def __post_init__(self) -> None:
        database_path = str(self.database_path).strip()
        if not database_path:
            raise ValueError(
                "QCS RF power-calibration database path must not be empty"
            )
        target_power_dbm = float(self.target_power_dbm)
        if not isfinite(target_power_dbm):
            raise ValueError("QCS target RF output power must be finite")
        if isinstance(self.run_id, bool) or int(self.run_id) != self.run_id:
            raise TypeError(
                "QCS RF power-calibration record ID must be an integer"
            )
        run_id = int(self.run_id)
        if run_id < 0:
            raise ValueError(
                "QCS RF power-calibration record ID must be nonnegative"
            )
        object.__setattr__(self, "database_path", database_path)
        object.__setattr__(self, "target_power_dbm", target_power_dbm)
        object.__setattr__(self, "run_id", run_id)


@dataclass(frozen=True)
class QcsRfPulseConfig:
    """One RF waveform placed relative to a named fine-tune segment."""

    gen_ch: int
    at_segment: str
    duration_s: float
    amplitude: float
    frequency_hz: float
    phase_rad: float = 0.0
    delay_s: float = 0.0
    envelope: str = "constant"
    require_within_segment: bool = True
    power_calibration: Optional[QcsRfPowerCalibrationConfig] = None
    power_calibration_provenance: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if isinstance(self.gen_ch, bool) or int(self.gen_ch) != self.gen_ch:
            raise TypeError("QCS RF gen_ch must be an integer")
        if int(self.gen_ch) < 0:
            raise ValueError("QCS RF gen_ch must be nonnegative")
        at_segment = _channel_name(self.at_segment, "QCS RF segment")
        duration_s = _positive_finite(
            self.duration_s, "QCS RF duration"
        )
        amplitude = float(self.amplitude)
        if not isfinite(amplitude) or not -1.0 <= amplitude <= 1.0:
            raise ValueError("QCS RF amplitude must be in [-1, 1]")
        frequency_hz = float(self.frequency_hz)
        phase_rad = float(self.phase_rad)
        if not isfinite(frequency_hz):
            raise ValueError("QCS RF frequency must be finite")
        if not isfinite(phase_rad):
            raise ValueError("QCS RF phase must be finite")
        delay_s = _nonnegative_finite(self.delay_s, "QCS RF delay")
        envelope = str(self.envelope).strip().lower()
        if envelope not in {"constant", "gaussian"}:
            raise ValueError(
                "QCS RF envelope must be 'constant' or 'gaussian'"
            )
        if not isinstance(self.require_within_segment, bool):
            raise TypeError("QCS RF require_within_segment must be boolean")
        if (
            self.power_calibration is not None
            and not isinstance(
                self.power_calibration,
                QcsRfPowerCalibrationConfig,
            )
        ):
            raise TypeError(
                "QCS RF power_calibration must be a "
                "QcsRfPowerCalibrationConfig"
            )
        provenance = self.power_calibration_provenance
        if provenance is not None:
            if not isinstance(provenance, Mapping):
                raise TypeError(
                    "QCS RF power-calibration provenance must be a mapping"
                )
            provenance = dict(provenance)
        object.__setattr__(self, "gen_ch", int(self.gen_ch))
        object.__setattr__(self, "at_segment", at_segment)
        object.__setattr__(self, "duration_s", duration_s)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "phase_rad", phase_rad)
        object.__setattr__(self, "delay_s", delay_s)
        object.__setattr__(self, "envelope", envelope)
        object.__setattr__(
            self,
            "power_calibration_provenance",
            provenance,
        )


@dataclass(frozen=True)
class QcsAcquisitionConfig:
    """Digitizer acquisition placed relative to a named segment."""

    at_segment: str
    duration_s: float
    pre_delay_s: float = 0.0
    sample_rate_hz: float = QCS_M5200_SAMPLE_RATE_HZ
    sample_count: Optional[int] = None
    integration_filter: Any = None
    frequency_hz: float = 0.0
    phase_rad: float = 0.0
    envelope: str = "constant"

    def __post_init__(self) -> None:
        at_segment = _channel_name(
            self.at_segment, "QCS acquisition segment"
        )
        duration_s = _positive_finite(
            self.duration_s, "QCS acquisition duration"
        )
        pre_delay_s = _nonnegative_finite(
            self.pre_delay_s, "QCS acquisition pre-delay"
        )
        sample_rate_hz = _positive_finite(
            self.sample_rate_hz, "QCS acquisition sample rate"
        )
        sample_count = self.sample_count
        if sample_count is not None:
            if isinstance(sample_count, bool):
                raise TypeError("QCS acquisition sample count must be an integer")
            integer_sample_count = int(sample_count)
            if integer_sample_count < 1 or integer_sample_count != sample_count:
                raise ValueError(
                    "QCS acquisition sample count must be a positive integer"
                )
            sample_count = integer_sample_count
        frequency_hz = float(self.frequency_hz)
        phase_rad = float(self.phase_rad)
        if not isfinite(frequency_hz):
            raise ValueError("QCS acquisition frequency must be finite")
        if not isfinite(phase_rad):
            raise ValueError("QCS acquisition phase must be finite")
        envelope = str(self.envelope).strip().lower()
        if envelope not in {"constant", "gaussian"}:
            raise ValueError(
                "QCS acquisition envelope must be 'constant' or 'gaussian'"
            )
        object.__setattr__(self, "at_segment", at_segment)
        object.__setattr__(self, "duration_s", duration_s)
        object.__setattr__(self, "pre_delay_s", pre_delay_s)
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "phase_rad", phase_rad)
        object.__setattr__(self, "envelope", envelope)


@dataclass(frozen=True)
class QcsNoiseTraceConfig:
    """One duration-defined raw M5200 trace for Noise Analysis.

    Noise Analysis always requires the time-domain trace, so hardware
    demodulation is disabled locally even when the shared Experiment setting
    requests integrated I/Q for other tabs.
    """

    connection_config: QcsConnectionConfig
    duration_s: float
    repetitions: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.connection_config, QcsConnectionConfig):
            raise TypeError(
                "QCS noise connection_config must be QcsConnectionConfig"
            )
        connection = replace(
            self.connection_config,
            hw_demod=False,
            # Results are read immediately after execute(); a nonblocking
            # backend would return before get_trace() is available.
            blocking=True,
        )
        if connection.acquisition_channel_name is None:
            raise ValueError(
                "QCS Noise Analysis requires a mapped M5200 acquisition "
                "channel"
            )
        duration_s = _positive_finite(
            self.duration_s, "QCS noise measurement duration"
        )
        if isinstance(self.repetitions, bool):
            raise TypeError("QCS noise repetitions must be an integer")
        repetitions = int(self.repetitions)
        if repetitions < 1 or repetitions != self.repetitions:
            raise ValueError("QCS noise repetitions must be a positive integer")
        object.__setattr__(self, "connection_config", connection)
        object.__setattr__(self, "duration_s", duration_s)
        object.__setattr__(self, "repetitions", repetitions)


@dataclass(frozen=True)
class QcsM5301ChannelCapacity:
    """Worst rendered-waveform usage found for one physical DC output."""

    output_name: str
    rendered_fabric_cycles: int
    point_index: int
    sweep_coordinate: Tuple[float, ...] = ()

    @property
    def rendered_samples(self) -> int:
        return (
            int(self.rendered_fabric_cycles)
            * QCS_M5301_SAMPLES_PER_FABRIC_CYCLE
        )

    @property
    def rendered_duration_us(self) -> float:
        return int(self.rendered_fabric_cycles) / (QCS_FABRIC_CLOCK_HZ / 1e6)


@dataclass(frozen=True)
class QcsM5301CapacityReport:
    """Per-channel M5301 waveform-buffer usage for a QCS sequence."""

    channels: Tuple[QcsM5301ChannelCapacity, ...]
    inspected_point_count: int
    sweep_point_count: int
    dc_channel_offsets_v: Tuple[float, ...] = ()

    @property
    def worst_channel(self) -> QcsM5301ChannelCapacity:
        if not self.channels:
            raise ValueError("QCS M5301 capacity report has no DC outputs")
        return max(
            self.channels,
            key=lambda channel: channel.rendered_fabric_cycles,
        )

    @property
    def maximum_samples(self) -> int:
        return QCS_M5301_MAX_RENDERED_SAMPLES

    @property
    def usage_fraction(self) -> float:
        return self.worst_channel.rendered_samples / self.maximum_samples

    @property
    def exceeds_capacity(self) -> bool:
        return self.worst_channel.rendered_samples > self.maximum_samples

    @property
    def exhaustive(self) -> bool:
        return int(self.inspected_point_count) == int(self.sweep_point_count)


@dataclass(frozen=True)
class QcsSweepExecutionPreview:
    """Host-side prediction shown by the GUI before QCS compilation.

    ``mode`` is ``none``, ``hardware``, ``hybrid``, ``software``, or
    ``invalid``. Mapper properties such as ``absolute_phase`` and
    physical-offset availability are confirmed when the real Program is
    compiled, so live GUI predictions use ``exact=False`` and the completed
    compiler/result may refine the state.
    """

    mode: str
    reasons: Tuple[str, ...] = ()
    exact: bool = True
    dc_channel_offsets_v: Tuple[float, ...] = ()


@dataclass(frozen=True)
class QcsCompiledPoint:
    """One QCS Program and the channels needed to read its result."""

    point_index: int
    program: Any
    acquisition_channels: Any
    duration_s: float
    acquisition_duration_s: Optional[float] = None
    acquisition_sample_rate_hz: Optional[float] = None
    acquisition_sample_count: Optional[int] = None
    integration_segment_sample_counts: Tuple[int, ...] = ()
    inter_segment_delay_s: float = 0.0
    acquisition_elapsed_duration_s: Optional[float] = None


@dataclass(frozen=True)
class QcsCompiledHardwareSweep:
    """One QCS Program containing a synchronized flattened sweep."""

    program: Any
    acquisition_channels: Any
    duration_s: float
    acquisition_duration_s: float
    acquisition_sample_rate_hz: float
    sweep_shape: Tuple[int, ...]
    hardware_sweep: bool = True
    sweep_variable_count: int = 0
    sweep_array_value_count: int = 0
    software_sweep_reasons: Tuple[str, ...] = ()
    bias_t_compensation_applied: bool = False
    bias_t_compensation_duration_s: Optional[float] = None
    bias_t_compensation_peak_amplitudes: Tuple[float, ...] = ()
    dc_channel_offsets_v: Tuple[float, ...] = ()
    dc_offset_init_compensation_v: Tuple[float, ...] = ()
    programmed_rf_pulses: Tuple[Mapping[str, Any], ...] = ()
    acquisition_pre_delay_s: Optional[float] = None
    acquisition_sample_count: Optional[int] = None
    integration_segment_sample_counts: Tuple[int, ...] = ()
    inter_segment_delay_s: float = 0.0
    acquisition_elapsed_duration_s: Optional[float] = None


@dataclass(frozen=True)
class QcsMixedSweepPlan:
    """A Python outer loop whose every item is one native hardware sweep."""

    hardware_axis_indices: Tuple[int, ...]
    software_axis_indices: Tuple[int, ...]
    hardware_shape: Tuple[int, ...]
    software_shape: Tuple[int, ...]
    first_compiled: QcsCompiledHardwareSweep
    planning_reasons: Tuple[str, ...] = ()
    planning_wall_s: float = 0.0

    @property
    def hardware_points_per_iteration(self) -> int:
        return int(np.prod(self.hardware_shape, dtype=np.int64))

    @property
    def software_iteration_count(self) -> int:
        return int(np.prod(self.software_shape, dtype=np.int64))


@dataclass(frozen=True)
class QcsCompiledSParameterSweep:
    """One RF-only M5300/M5200 QCS-resolved frequency-sweep program."""

    program: Any
    rf_channels: Any
    acquisition_channels: Any
    frequency_variable: Any
    amplitude_variable: Any
    frequencies_hz: np.ndarray
    rf_amplitudes: np.ndarray
    requested_integration_duration_s: float
    quantized_requested_integration_duration_s: float
    quantized_requested_integration_sample_count: int
    integration_duration_s: float
    integration_sample_count: int
    integration_segment_sample_counts: Tuple[int, ...]
    inter_segment_delay_s: float
    acquisition_sample_rate_hz: float
    repetitions_per_point: int


@dataclass(frozen=True)
class QcsExecutionResult:
    """Persistence-free output from QCS compilation and execution."""

    ddr_result: FineTuneDdrResult
    programs: Tuple[Any, ...]
    raw_results: Tuple[Any, ...]
    program_summary: Mapping[str, Any]
    rf_settings: Mapping[str, Any]


@dataclass(frozen=True)
class QcsSParameterExecutionResult:
    """Persistence-free result from one QCS-resolved RF frequency sweep."""

    frequencies_hz: np.ndarray
    iq: np.ndarray
    program: Any
    raw_result: Any
    program_summary: Mapping[str, Any]
    rf_settings: Mapping[str, Any]
    dut_input_powers_dbm: Optional[np.ndarray] = None
    dut_output_powers_dbm: Optional[np.ndarray] = None


@dataclass(frozen=True)
class QcsNoiseTraceResult:
    """One persistence-free raw M5200 Noise Analysis acquisition."""

    i_traces: np.ndarray
    requested_duration_s: float
    duration_s: float
    sample_rate_hz: float
    sample_count: int
    program: Any
    raw_result: Any

    def __post_init__(self) -> None:
        traces = np.asarray(self.i_traces, dtype=np.float64)
        if traces.ndim != 2:
            raise ValueError(
                "QCS noise I traces must have shape (repetition, sample)"
            )
        if not np.all(np.isfinite(traces)):
            raise ValueError("QCS noise I traces must contain finite values")
        sample_count = int(self.sample_count)
        if sample_count < 1 or sample_count != self.sample_count:
            raise ValueError("QCS noise sample_count must be a positive integer")
        if traces.shape[1] != sample_count:
            raise ValueError(
                "QCS noise I-trace sample axis does not match sample_count"
            )
        requested_duration_s = _positive_finite(
            self.requested_duration_s,
            "QCS noise requested measurement duration",
        )
        duration_s = _positive_finite(
            self.duration_s, "QCS noise effective measurement duration"
        )
        sample_rate_hz = _positive_finite(
            self.sample_rate_hz, "QCS noise M5200 sample rate"
        )
        if not np.isclose(
            duration_s * sample_rate_hz,
            sample_count,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(
                "QCS noise duration, sample rate, and sample count disagree"
            )
        object.__setattr__(self, "i_traces", np.ascontiguousarray(traces))
        object.__setattr__(
            self, "requested_duration_s", requested_duration_s
        )
        object.__setattr__(self, "duration_s", duration_s)
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(self, "sample_count", sample_count)


@dataclass(frozen=True)
class _QcsSweepTarget:
    """One direct QCS Scalar and its synchronized point array."""

    variable: Any
    array: Any
    values: np.ndarray
    hardware_supported: bool
    channel_name: str
    description: str


@dataclass(frozen=True)
class _QcsFixedDcOffsetPlan:
    """Fixed M5301 offsets used to residualize one synchronized sweep."""

    source_offsets: Tuple[float, ...]
    offset_volts: Tuple[float, ...]
    optional_offset_outputs: Tuple[bool, ...] = ()


def _qcs_without_optional_dc_offsets(
    plan: _QcsFixedDcOffsetPlan,
) -> _QcsFixedDcOffsetPlan:
    """Drop capacity-only constant-lane offsets from one compiler plan."""

    optional = tuple(bool(value) for value in plan.optional_offset_outputs)
    if len(optional) != len(plan.offset_volts):
        return plan
    return _QcsFixedDcOffsetPlan(
        source_offsets=tuple(
            0.0 if is_optional else float(value)
            for value, is_optional in zip(plan.source_offsets, optional)
        ),
        offset_volts=tuple(
            0.0 if is_optional else float(value)
            for value, is_optional in zip(plan.offset_volts, optional)
        ),
        optional_offset_outputs=optional,
    )


@dataclass
class StoredQcsExperiment:
    """Stored result compatible with the existing Experiment result handlers."""

    run_id: int
    guid: str
    database_path: Path
    row_count: int
    dataset: Any
    program: Any
    ddr_result: FineTuneDdrResult
    rf_settings: Mapping[str, Any]
    programs: Tuple[Any, ...] = ()
    raw_results: Tuple[Any, ...] = ()
    program_summary: Mapping[str, Any] = field(default_factory=dict)


def _import_qcs():
    try:
        import keysight.qcs as qcs
    except ImportError as exc:
        raise ImportError(
            "Keysight QCS 2.5.5 is required for the QCS backend. "
            "Install it in the QCS Conda environment or select QICK."
        ) from exc
    return qcs


def load_qcs_channel_mapper(
    connection_config: QcsConnectionConfig,
    *,
    qcs_module=None,
):
    """Load and type-check the serialized QCS ChannelMapper."""
    qcs = _import_qcs() if qcs_module is None else qcs_module
    path = Path(connection_config.mapper_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"QCS ChannelMapper file not found: {path}")
    if connection_config.mapper_sha256 is not None:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != connection_config.mapper_sha256:
            raise ValueError(
                "The QCS ChannelMapper file does not match the hardware "
                "configuration selected in the GUI"
            )
    mapper = qcs.load(path)
    mapper_type = getattr(qcs, "ChannelMapper", None)
    if mapper_type is not None and not isinstance(mapper, mapper_type):
        raise TypeError(
            f"{path} contains {type(mapper).__name__}, not ChannelMapper"
        )
    return mapper


def _mapper_channels(mapper: Any) -> Tuple[Any, ...]:
    channels = getattr(mapper, "channels", None)
    if channels is None and isinstance(mapper, Mapping):
        channels = tuple(mapper.values())
    if channels is None:
        raise TypeError("QCS ChannelMapper does not expose virtual channels")
    return tuple(channels)


def _resolve_mapper_channel(mapper: Any, name: str) -> Any:
    matches = [
        channel
        for channel in _mapper_channels(mapper)
        if str(getattr(channel, "name", "")).strip() == name
    ]
    if not matches:
        available = sorted(
            {
                str(getattr(channel, "name", "")).strip()
                for channel in _mapper_channels(mapper)
                if str(getattr(channel, "name", "")).strip()
            }
        )
        suffix = f"; available names: {', '.join(available)}" if available else ""
        raise KeyError(f"QCS mapper has no virtual channel {name!r}{suffix}")
    if len(matches) > 1:
        raise ValueError(
            f"QCS mapper contains duplicate virtual channel name {name!r}"
        )
    channel = matches[0]
    labels = getattr(channel, "labels", None)
    if labels is not None and len(tuple(labels)) != 1:
        raise ValueError(
            f"QCS virtual channel {name!r} must contain exactly one label"
        )
    return channel


def _validate_mapped_hardware_role(
    mapper: Any,
    channel: Any,
    *,
    name: str,
    role: str,
    expected_instruments: Sequence[str],
    require_relative_phase: bool = False,
) -> None:
    """Preflight role semantics exposed by a native QCS ChannelMapper.

    Lightweight injected test adapters do not necessarily expose physical
    mappings, so this check is conditional on the real mapper API. HCL-backed
    operation always supplies that API.
    """
    get_physical_channels = getattr(mapper, "get_physical_channels", None)
    if not callable(get_physical_channels):
        return
    physical_channels = tuple(get_physical_channels(channel))
    if len(physical_channels) != 1:
        raise ValueError(
            f"QCS {role} virtual channel {name!r} must map to exactly one "
            f"physical connector; found {len(physical_channels)}"
        )
    physical = physical_channels[0]
    instrument = str(getattr(physical, "instrument", ""))
    allowed = tuple(str(value) for value in expected_instruments)
    if instrument not in allowed:
        raise ValueError(
            f"QCS {role} virtual channel {name!r} must map to "
            f"{' or '.join(allowed)}; found {instrument or 'unknown hardware'}"
        )
    if require_relative_phase and bool(
        getattr(channel, "absolute_phase", False)
    ):
        raise ValueError(
            f"QCS {role} virtual channel {name!r} must use "
            "absolute_phase=False for a native amplitude hardware sweep"
        )


def _mapped_m5301_offset_scalar(
    mapper: Any,
    channel: Any,
    *,
    name: str,
) -> Optional[Any]:
    """Return one mapped M5301 physical-offset Scalar when exposed.

    Lightweight adapters used by callers and tests may expose only virtual
    channels. A nonzero automatic offset is disabled for those adapters, while
    a native QCS ChannelMapper always supplies ``get_physical_channels``.
    """

    get_physical_channels = getattr(mapper, "get_physical_channels", None)
    if not callable(get_physical_channels):
        return None
    physical_channels = tuple(get_physical_channels(channel))
    if len(physical_channels) != 1:
        raise ValueError(
            f"QCS DC virtual channel {name!r} must map to exactly one "
            f"physical connector; found {len(physical_channels)}"
        )
    physical = physical_channels[0]
    settings = getattr(physical, "settings", None)
    offset = None if settings is None else getattr(settings, "offset", None)
    if offset is None or not hasattr(offset, "value"):
        return None
    return offset


def _set_qcs_dc_channel_offsets(
    mapper: Any,
    *,
    channel_names: Sequence[str],
    offset_volts: Sequence[float],
    require_nonzero_support: bool,
) -> Tuple[Any, ...]:
    """Keep mapped physical offsets at zero.

    Channel offsets are not a PulseGenerator voltage-control or hardware-sweep
    mechanism.  The nonzero guard makes that contract explicit even if an old
    caller or saved execution plan still supplies offset metadata.
    """

    names = tuple(str(value) for value in channel_names)
    values = tuple(float(value) for value in offset_volts)
    if len(names) != len(values):
        raise ValueError("QCS DC offset count must match the DC channel count")
    if any(
        not np.isclose(value, 0.0, rtol=0.0, atol=1e-15)
        for value in values
    ):
        raise QcsUnsupportedFeatureError(
            "QCS physical channel offsets are disabled; voltage sweeps must "
            "use DCWaveform amplitudes"
        )
    scalars = []
    for name, offset_v in zip(names, values):
        channel = _resolve_mapper_channel(mapper, name)
        scalar = _mapped_m5301_offset_scalar(
            mapper,
            channel,
            name=name,
        )
        if scalar is None:
            if require_nonzero_support and not np.isclose(
                offset_v,
                0.0,
                rtol=0.0,
                atol=1e-15,
            ):
                raise QcsUnsupportedFeatureError(
                    f"QCS DC channel {name!r} does not expose the mapped "
                    "M5301 physical offset required for this hardware sweep"
                )
            continue
        scalar_value = offset_v / QCS_M5301_OFFSET_VOLTS_PER_SCALAR
        if abs(scalar_value) > QCS_M5301_MAX_ABS_OFFSET_SCALAR + 1e-12:
            raise QcsUnsupportedFeatureError(
                f"QCS DC channel {name!r} requires {offset_v:.6g} V fixed "
                "offset, exceeding the M5301 offset limit"
            )
        scalar.value = float(scalar_value)
        scalars.append(scalar)
    return tuple(scalars)


def _mapper_supports_qcs_dc_offsets(
    mapper: Any,
    *,
    channel_names: Sequence[str],
    offset_volts: Sequence[float],
) -> bool:
    """Return whether every requested nonzero offset has a physical Scalar."""

    for name, offset_v in zip(channel_names, offset_volts):
        if np.isclose(offset_v, 0.0, rtol=0.0, atol=1e-15):
            continue
        channel = _resolve_mapper_channel(mapper, str(name))
        if _mapped_m5301_offset_scalar(
            mapper,
            channel,
            name=str(name),
        ) is None:
            return False
    return True


def _mapped_channel_sample_rate(
    mapper: Any,
    channel: Any,
) -> Optional[float]:
    """Return the mapper's physical sample rate when the API exposes it."""
    get_physical_channels = getattr(mapper, "get_physical_channels", None)
    if not callable(get_physical_channels):
        return None
    physical_channels = tuple(get_physical_channels(channel))
    if len(physical_channels) != 1:
        raise ValueError(
            "QCS acquisition virtual channel must map to exactly one "
            "physical digitizer channel"
        )
    sample_rate = getattr(physical_channels[0], "sample_rate", None)
    if sample_rate is None:
        return None
    return _positive_finite(
        sample_rate, "mapped QCS digitizer sample rate"
    )


def _resolved_acquisition_timing(
    mapper: Any,
    channel: Any,
    acquisition: QcsAcquisitionConfig,
    *,
    hardware_demodulation: bool,
) -> tuple[float, float]:
    """Resolve M5200 timing and enforce QCS integration-block alignment."""

    sample_rate_hz = _mapped_channel_sample_rate(mapper, channel)
    if sample_rate_hz is None:
        sample_rate_hz = acquisition.sample_rate_hz
    duration_s = acquisition.duration_s
    if acquisition.sample_count is not None:
        sample_count = acquisition.sample_count
        if (
            hardware_demodulation
            and sample_count % QCS_M5200_INTEGRATION_BLOCK_SAMPLES != 0
        ):
            raise ValueError(
                "QCS M5200 hardware-demodulation integration length must "
                f"be a multiple of "
                f"{QCS_M5200_INTEGRATION_BLOCK_SAMPLES} samples; got "
                f"{sample_count}"
            )
        duration_s = sample_count / sample_rate_hz
    elif hardware_demodulation:
        rendered_samples = duration_s * sample_rate_hz
        sample_count = int(round(rendered_samples))
        if (
            not np.isclose(
                rendered_samples,
                sample_count,
                rtol=0.0,
                atol=1.0e-6,
            )
            or sample_count % QCS_M5200_INTEGRATION_BLOCK_SAMPLES != 0
        ):
            raise ValueError(
                "QCS M5200 hardware-demodulation duration must render to "
                f"a multiple of {QCS_M5200_INTEGRATION_BLOCK_SAMPLES} "
                f"samples at {sample_rate_hz:g} S/s; got "
                f"{rendered_samples:g} samples"
            )
        # Rebuild the duration from the validated integral sample count. This
        # removes GUI floating-point residue before the value reaches HCL.
        duration_s = sample_count / sample_rate_hz
    return duration_s, sample_rate_hz


def _executed_program_sample_rate(
    program: Any,
    channel: Any,
) -> Optional[float]:
    get_sample_rates = getattr(program, "get_sample_rates", None)
    if not callable(get_sample_rates):
        return None
    values = get_sample_rates(channel)
    value = _first_result_value(values, channel)
    return _positive_finite(
        value, "executed QCS digitizer sample rate"
    )


def _segment_boundaries_seconds(
    boundaries: Sequence[Tuple[str, float, float]],
    *,
    fabric_mhz: float,
) -> Mapping[str, Tuple[float, float]]:
    scale = 1.0 / (_positive_finite(fabric_mhz, "fabric clock") * 1e6)
    return {
        str(name): (float(start) * scale, float(stop) * scale)
        for name, start, stop in boundaries
    }


def _qcs_segment_layer_index(
    interval_start_times_s: Sequence[float],
    segment_start_s: float,
    *,
    segment_name: str,
) -> int:
    """Map a logical segment start to its synchronized QCS interval layer."""

    starts = np.asarray(interval_start_times_s, dtype=float)
    matches = np.flatnonzero(
        np.isclose(starts, float(segment_start_s), rtol=0.0, atol=1.0e-15)
    )
    if matches.size != 1:
        raise QcsUnsupportedFeatureError(
            f"QCS segment {segment_name!r} does not map to exactly one "
            "synchronized DC interval layer"
        )
    return int(matches[0])


def _qcs_swept_segment_layer_index(
    duration_table_s: np.ndarray,
    boundary_rows: Sequence[Mapping[str, Tuple[float, float]]],
    segment_name: str,
) -> int:
    """Return the fixed layer index for a segment across every sweep point."""

    durations = np.asarray(duration_table_s, dtype=float)
    layer_index = None
    for point_index, boundaries in enumerate(boundary_rows):
        if segment_name not in boundaries:
            raise KeyError(
                f"no timing boundary for QCS segment {segment_name!r} at "
                f"sweep point {point_index + 1}"
            )
        starts = np.r_[0.0, np.cumsum(durations[point_index, :-1])]
        current = _qcs_segment_layer_index(
            starts,
            boundaries[segment_name][0],
            segment_name=segment_name,
        )
        if layer_index is None:
            layer_index = current
        elif current != layer_index:
            raise QcsUnsupportedFeatureError(
                f"QCS segment {segment_name!r} changes synchronized layer "
                f"index at sweep point {point_index + 1}"
            )
    if layer_index is None:
        raise ValueError("QCS synchronized sweep has no timing rows")
    return int(layer_index)


def _qcs_segment_grouped_layer_layout(
    sequence: Any,
    *,
    duration_table_s: np.ndarray,
    boundary_rows: Sequence[Mapping[str, Tuple[float, float]]],
) -> tuple[
    tuple[tuple[int, ...], ...],
    Mapping[str, int],
    Mapping[str, np.ndarray],
]:
    """Group each incoming RAMP with its destination SET plateau.

    QCS operations on one virtual channel are sequential within a layer, while
    different virtual channels run in parallel. Keeping an incoming ramp and
    the following SET plateau in one layer therefore lets RF and acquisition
    start after the ramp, but still share the exact layer that establishes the
    swept SET voltage.
    """

    durations = np.asarray(duration_table_s, dtype=float)
    if durations.ndim != 2 or durations.shape[0] != len(boundary_rows):
        raise ValueError(
            "QCS grouped layer layout requires one duration row per point"
        )
    if durations.shape[1] < 1 or not np.all(np.isfinite(durations)):
        raise ValueError("QCS grouped layer durations must be finite")
    if np.any(durations <= 0.0):
        raise ValueError("QCS grouped layer durations must be positive")

    segments = tuple(sequence.segments)
    segment_interval_indices = {
        str(segment.name): _qcs_swept_segment_layer_index(
            durations,
            boundary_rows,
            str(segment.name),
        )
        for segment in segments
    }

    claimed: set[int] = set()
    groups: list[tuple[int, ...]] = []
    for segment_index, segment in enumerate(segments):
        if str(segment.kind) != "set":
            continue
        plateau_index = segment_interval_indices[str(segment.name)]
        group = (plateau_index,)
        if (
            segment_index > 0
            and str(segments[segment_index - 1].kind) == "ramp"
        ):
            ramp = segments[segment_index - 1]
            ramp_index = segment_interval_indices[str(ramp.name)]
            if plateau_index != ramp_index + 1:
                raise QcsUnsupportedFeatureError(
                    f"QCS RAMP {ramp.name!r} and destination SET "
                    f"{segment.name!r} are not consecutive DC intervals"
                )
            group = (ramp_index, plateau_index)
        if any(interval_index in claimed for interval_index in group):
            raise QcsUnsupportedFeatureError(
                "QCS logical segments overlap while constructing grouped "
                "DC layers"
            )
        groups.append(group)
        claimed.update(group)

    for interval_index in range(durations.shape[1]):
        if interval_index not in claimed:
            groups.append((interval_index,))
    groups.sort(key=lambda value: value[0])
    flattened = tuple(index for group in groups for index in group)
    if flattened != tuple(range(durations.shape[1])):
        raise QcsUnsupportedFeatureError(
            "QCS grouped DC layers do not form an ordered interval partition"
        )

    interval_to_layer = {
        interval_index: layer_index
        for layer_index, group in enumerate(groups)
        for interval_index in group
    }
    layer_start_times = np.c_[
        np.zeros(durations.shape[0], dtype=float),
        np.cumsum(durations[:, :-1], axis=1),
    ]
    segment_layer_indices: dict[str, int] = {}
    segment_layer_offsets: dict[str, np.ndarray] = {}
    for segment in segments:
        name = str(segment.name)
        interval_index = segment_interval_indices[name]
        layer_index = interval_to_layer[interval_index]
        group_start_index = groups[layer_index][0]
        offsets = np.asarray(
            [
                float(boundaries[name][0])
                - float(layer_start_times[point_index, group_start_index])
                for point_index, boundaries in enumerate(boundary_rows)
            ],
            dtype=float,
        )
        if np.any(offsets < -1e-15) or not np.all(np.isfinite(offsets)):
            raise QcsUnsupportedFeatureError(
                f"QCS segment {name!r} has an invalid grouped-layer offset"
            )
        offsets[np.isclose(offsets, 0.0, rtol=0.0, atol=1e-15)] = 0.0
        segment_layer_indices[name] = layer_index
        segment_layer_offsets[name] = offsets

    return (
        tuple(groups),
        segment_layer_indices,
        segment_layer_offsets,
    )


def _qcs_join_grouped_dc_intervals(
    interval_operations: Sequence[Any],
    interval_group: Sequence[int],
) -> Any:
    """Return the sequential QCS operations for one grouped DC layer."""

    operations = []
    for interval_index in interval_group:
        current = interval_operations[int(interval_index)]
        if isinstance(current, (list, tuple)):
            operations.extend(current)
        else:
            operations.append(current)
    if not operations:
        raise ValueError("QCS grouped DC layer has no operations")
    return operations[0] if len(operations) == 1 else operations


def _qcs_envelope(qcs: Any, name: str) -> Any:
    if name == "gaussian":
        return qcs.GaussianEnvelope()
    return qcs.ConstantEnvelope()


def _qcs_sweep_parameter(
    qcs: Any,
    *,
    name: str,
    values: Sequence[float],
    targets: list[_QcsSweepTarget],
    hardware_supported: bool,
    channel_name: str,
    description: str,
    force: bool = False,
) -> Any:
    """Return a constant or a direct Scalar backed by one point array.

    Physical values are calculated completely on the host.  In particular,
    the returned Scalar is never combined arithmetically with another Scalar;
    this avoids the ScalarAdder/ScalarMultiplier sandbox limitation observed
    on the connected QCS 2.5.5 system.
    """

    point_values = np.asarray(values, dtype=float)
    if point_values.ndim != 1 or point_values.size < 1:
        raise ValueError(f"QCS {description} values must be a nonempty 1D array")
    if not np.all(np.isfinite(point_values)):
        raise ValueError(f"QCS {description} values must be finite")
    if not force and np.allclose(
        point_values,
        point_values[0],
        rtol=0.0,
        atol=1e-15,
    ):
        return float(point_values[0])

    variable = qcs.Scalar(
        name,
        value=float(point_values[0]),
        dtype=float,
    )
    sweep_array = qcs.Array(
        f"{name}_values",
        value=point_values,
        dtype=float,
    )
    targets.append(
        _QcsSweepTarget(
            variable=variable,
            array=sweep_array,
            values=point_values.copy(),
            hardware_supported=bool(hardware_supported),
            channel_name=str(channel_name),
            description=str(description),
        )
    )
    return variable


def _qcs_parameterized_dc_interval(
    qcs: Any,
    *,
    duration_values_s: Sequence[float],
    start_values: Sequence[float],
    end_values: Sequence[float],
    targets: list[_QcsSweepTarget],
    channel_name: str,
    output_index: int,
    interval_index: int,
    fabric_hz: float,
) -> Any:
    """Build one fixed-topology M5301 interval for every sweep point.

    A changing interval is represented as the sum of fixed falling and rising
    envelopes.  Their direct start/end Scalars are swept with host-computed
    arrays, so arbitrary cross-capacitance-corrected ramps need no Scalar
    arithmetic in HCL.
    """

    durations = np.asarray(duration_values_s, dtype=float)
    starts = np.asarray(start_values, dtype=float)
    ends = np.asarray(end_values, dtype=float)
    if not (durations.shape == starts.shape == ends.shape):
        raise ValueError("QCS synchronized DC interval arrays must have equal shape")
    if durations.ndim != 1 or durations.size < 1:
        raise ValueError(
            "QCS synchronized DC interval arrays must be nonempty 1D arrays"
        )
    if not (
        np.all(np.isfinite(durations))
        and np.all(np.isfinite(starts))
        and np.all(np.isfinite(ends))
        and np.all(durations > 0.0)
    ):
        raise ValueError(
            "QCS synchronized DC interval durations must be positive and all "
            "interval values must be finite"
        )
    prefix = f"awg_dc_{output_index}_interval_{interval_index}"
    pointwise_constant = np.allclose(
        starts,
        ends,
        rtol=0.0,
        atol=1e-15,
    )
    zero_interval = bool(
        np.allclose(starts, 0.0, rtol=0.0, atol=1e-15)
        and np.allclose(ends, 0.0, rtol=0.0, atol=1e-15)
    )
    duration_cycles = np.rint(durations * float(fabric_hz)).astype(np.int64)
    duration_meets_hardware_minimum = bool(
        np.all(duration_cycles >= QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES)
    )
    if not zero_interval and not duration_meets_hardware_minimum:
        minimum_s = QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES / float(fabric_hz)
        first_short = int(
            np.flatnonzero(
                duration_cycles < QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES
            )[0]
        )
        raise QcsUnsupportedFeatureError(
            f"QCS synchronized {channel_name} interval {interval_index} is an "
            f"M5301 waveform but sweep point {first_short + 1} lasts only "
            f"{duration_cycles[first_short]} fabric cycles; M5301 waveforms "
            f"require at least {QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES} cycles "
            f"({minimum_s * 1e9:.9g} ns)"
        )
    fixed_duration = bool(np.all(duration_cycles == duration_cycles[0]))
    if not zero_interval and not (pointwise_constant and fixed_duration):
        invalid_granularity = np.flatnonzero(
            duration_cycles
            % QCS_M5301_WAVEFORM_GRANULARITY_FABRIC_CYCLES
        )
        if invalid_granularity.size:
            first_invalid = int(invalid_granularity[0])
            raise QcsUnsupportedFeatureError(
                f"QCS synchronized {channel_name} interval {interval_index} "
                f"lasts {duration_cycles[first_invalid]} fabric cycles at "
                f"sweep point {first_invalid + 1}; M5301 waveforms require "
                "a multiple of 2 fabric cycles (16 samples, 6.666667 ns)."
            )
    duration = _qcs_sweep_parameter(
        qcs,
        name=f"{prefix}_duration",
        values=durations,
        targets=targets,
        # QCS can hardware-sweep a Delay duration provided every value meets
        # its four-fabric-cycle sandbox minimum. DCWaveform duration remains
        # a software-resolved setting even when its amplitudes are hardware
        # sweepable.
        hardware_supported=zero_interval and duration_meets_hardware_minimum,
        channel_name=channel_name,
        description=f"{channel_name} interval {interval_index} duration",
    )

    if pointwise_constant:
        if zero_interval:
            return qcs.Delay(duration=duration, name=f"{prefix}_zero")
        amplitude = _qcs_sweep_parameter(
            qcs,
            name=f"{prefix}_amplitude",
            values=starts,
            targets=targets,
            hardware_supported=True,
            channel_name=channel_name,
            description=f"{channel_name} interval {interval_index} amplitude",
        )
        if fixed_duration:
            return _qcs_constant_dc_interval(
                qcs,
                duration_s=float(durations[0]),
                amplitude=amplitude,
                name=prefix,
                fabric_hz=fabric_hz,
            )
        # A swept duration cannot currently express ``duration - seed`` as a
        # direct HCL target without changing the operation topology at the
        # minimum duration. Preserve the existing complete-waveform lowering
        # for that case; fixed numeric points still use seed plus Hold.
        return qcs.DCWaveform(
            duration=duration,
            envelope=qcs.ConstantEnvelope(),
            amplitude=amplitude,
            name=prefix,
        )

    fixed_start = _qcs_constant_sweep_value(starts)
    fixed_end = _qcs_constant_sweep_value(ends)
    if fixed_start is not None and fixed_end is not None:
        scale = max(abs(fixed_start), abs(fixed_end))
        if scale == 0.0:
            return qcs.Delay(duration=duration, name=f"{prefix}_zero")
        return qcs.DCWaveform(
            duration=duration,
            envelope=qcs.ArbitraryEnvelope(
                [0.0, 1.0],
                [fixed_start, fixed_end],
            ),
            amplitude=scale,
            name=prefix,
        )

    composite_required = bool(
        not np.allclose(starts, 0.0, rtol=0.0, atol=1e-15)
        and not np.allclose(ends, 0.0, rtol=0.0, atol=1e-15)
    )
    if composite_required:
        raise QcsUnsupportedFeatureError(
            f"QCS synchronized {channel_name} interval {interval_index} "
            "would require M5301 waveform addition, which is disabled after "
            "hardware testing showed that one ramp component can be omitted; "
            "run it through execute_qcs_sequence so fixed numeric point "
            "programs are used"
        )

    components = []
    if not np.allclose(starts, 0.0, rtol=0.0, atol=1e-15):
        start_amplitude = _qcs_sweep_parameter(
            qcs,
            name=f"{prefix}_start_amplitude",
            values=starts,
            targets=targets,
            hardware_supported=True,
            channel_name=channel_name,
            description=f"{channel_name} interval {interval_index} start amplitude",
        )
        components.append(
            qcs.DCWaveform(
                duration=duration,
                envelope=qcs.ArbitraryEnvelope([0.0, 1.0], [1.0, 0.0]),
                amplitude=start_amplitude,
                name=f"{prefix}_falling",
            )
        )
    if not np.allclose(ends, 0.0, rtol=0.0, atol=1e-15):
        end_amplitude = _qcs_sweep_parameter(
            qcs,
            name=f"{prefix}_end_amplitude",
            values=ends,
            targets=targets,
            hardware_supported=True,
            channel_name=channel_name,
            description=f"{channel_name} interval {interval_index} end amplitude",
        )
        components.append(
            qcs.DCWaveform(
                duration=duration,
                envelope=qcs.ArbitraryEnvelope([0.0, 1.0], [0.0, 1.0]),
                amplitude=end_amplitude,
                name=f"{prefix}_rising",
            )
        )
    if not components:
        return qcs.Delay(duration=duration, name=f"{prefix}_zero")
    if len(components) != 1:
        raise AssertionError(
            "safe synchronized DC interval must contain exactly one waveform"
        )
    return components[0]


def _qcs_parameterized_dc_hold_mask(
    *,
    duration_table_s: np.ndarray,
    start_table: np.ndarray,
    end_table: np.ndarray,
    fabric_hz: float,
) -> np.ndarray:
    """Mark plateaus that can retain an already established DC endpoint.

    The first independent nonzero plateau emits a minimum legal constant seed
    followed by ``Hold``.  A later fixed plateau at the same endpoint can use
    ``Hold`` directly after either that seed or a changing waveform.
    """

    durations = np.asarray(duration_table_s, dtype=float)
    starts = np.asarray(start_table, dtype=float)
    ends = np.asarray(end_table, dtype=float)
    if durations.ndim != 2 or starts.shape != durations.shape or ends.shape != (
        durations.shape
    ):
        raise ValueError(
            "QCS synchronized DC hold analysis requires equal 2D tables"
        )
    hold_mask = np.zeros(durations.shape[1], dtype=bool)
    retained_output = False
    previous_ends = None
    for interval_index in range(durations.shape[1]):
        current_durations = durations[:, interval_index]
        current_starts = starts[:, interval_index]
        current_ends = ends[:, interval_index]
        pointwise_constant = bool(
            np.allclose(
                current_starts,
                current_ends,
                rtol=0.0,
                atol=1e-15,
            )
        )
        zero_interval = bool(
            np.allclose(current_starts, 0.0, rtol=0.0, atol=1e-15)
            and np.allclose(current_ends, 0.0, rtol=0.0, atol=1e-15)
        )
        duration_cycles = np.rint(current_durations * float(fabric_hz)).astype(
            np.int64
        )
        fixed_duration = bool(
            np.all(duration_cycles == duration_cycles[0])
            and np.allclose(
                current_durations * float(fabric_hz),
                duration_cycles,
                rtol=0.0,
                atol=1e-6,
            )
            and duration_cycles[0] >= QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES
        )
        continuous = bool(
            previous_ends is not None
            and np.allclose(
                current_starts,
                previous_ends,
                rtol=0.0,
                atol=1e-15,
            )
        )
        hold_mask[interval_index] = bool(
            retained_output
            and pointwise_constant
            and not zero_interval
            and fixed_duration
            and continuous
        )

        if zero_interval:
            retained_output = False
        elif hold_mask[interval_index]:
            retained_output = True
        elif pointwise_constant:
            # The lowering below emits a constant seed (or a complete short
            # waveform) whose endpoint can be retained by a following Hold.
            retained_output = True
        else:
            retained_output = True
        previous_ends = current_ends
    return hold_mask


def _qcs_parameterized_dc_operations(
    qcs: Any,
    *,
    duration_table_s: np.ndarray,
    start_table: np.ndarray,
    end_table: np.ndarray,
    targets: list[_QcsSweepTarget],
    channel_name: str,
    output_index: int,
    fabric_hz: float,
    grouped: bool = False,
) -> list[Any]:
    """Lower one output with minimum constant seeds and retained Holds."""

    durations = np.asarray(duration_table_s, dtype=float)
    starts = np.asarray(start_table, dtype=float)
    ends = np.asarray(end_table, dtype=float)
    hold_mask = _qcs_parameterized_dc_hold_mask(
        duration_table_s=durations,
        start_table=starts,
        end_table=ends,
        fabric_hz=fabric_hz,
    )
    interval_operations = []
    for interval_index in range(durations.shape[1]):
        prefix = f"awg_dc_{output_index}_interval_{interval_index}"
        if hold_mask[interval_index]:
            duration_cycles = int(
                round(float(durations[0, interval_index]) * fabric_hz)
            )
            interval_operations.append(
                qcs.Hold(
                    duration=duration_cycles / fabric_hz,
                    name=f"{prefix}_hold",
                )
            )
            continue
        interval = _qcs_parameterized_dc_interval(
            qcs,
            duration_values_s=durations[:, interval_index],
            start_values=starts[:, interval_index],
            end_values=ends[:, interval_index],
            targets=targets,
            channel_name=channel_name,
            output_index=output_index,
            interval_index=interval_index,
            fabric_hz=fabric_hz,
        )
        interval_operations.append(interval)
    if grouped:
        return interval_operations
    operations = []
    for interval in interval_operations:
        if isinstance(interval, (list, tuple)):
            operations.extend(interval)
        else:
            operations.append(interval)
    return operations


def _qcs_constant_dc_interval(
    qcs: Any,
    *,
    duration_s: float,
    amplitude: Any,
    name: str,
    fabric_hz: float,
) -> Any:
    """Build a constant M5301 interval without a long rendered waveform.

    ``qcs.Hold`` retains the final sample of the preceding waveform. A short
    constant seed therefore has exactly the same voltage and total duration
    as one long ``DCWaveform``, while remaining compatible with amplitude
    hardware sweeps and the finite waveform buffer in the HCL sandbox.
    """
    duration = _fabric_aligned_seconds(
        duration_s,
        fabric_hz=fabric_hz,
        label=f"{name} duration",
        positive=True,
    )
    duration_cycles = int(round(duration * fabric_hz))
    if duration_cycles < QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES:
        minimum_s = QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES / fabric_hz
        raise ValueError(
            f"{name} duration must be at least "
            f"{QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES} QCS fabric cycles "
            f"({minimum_s * 1e9:.9g} ns) for an M5301 waveform"
        )
    seed_cycles = min(duration_cycles, QCS_M5301_HOLD_SEED_FABRIC_CYCLES)
    # M5301 waveform data has a 16-sample granularity. At 2.4 GSa/s on
    # the 300 MHz fabric this is two fabric cycles. Leave an odd final
    # cycle to Hold instead of asking the driver to pad the voltage seed.
    seed_cycles -= (
        seed_cycles % QCS_M5301_WAVEFORM_GRANULARITY_FABRIC_CYCLES
    )
    seed_duration_s = seed_cycles / fabric_hz
    hold_cycles = duration_cycles - seed_cycles
    seed = qcs.DCWaveform(
        duration=seed_duration_s,
        envelope=qcs.ConstantEnvelope(),
        amplitude=amplitude,
        name=name if hold_cycles == 0 else f"{name}_set",
    )
    if hold_cycles == 0:
        return seed
    return [
        seed,
        qcs.Hold(
            duration=hold_cycles / fabric_hz,
            name=f"{name}_hold",
        ),
    ]


def _qcs_stability_measurement_interval(
    qcs: Any,
    *,
    duration_s: float,
    amplitude: Any,
    name: str,
    fabric_hz: float,
) -> Any:
    """Build the zero-terminated, hardware-safe swept target interval.

    This uses the same connected-M5301-tested direct ramp-to-Hold chain as
    fixed-time compensation.  Returning to zero before the layer ends also
    prevents HCL's inter-iteration delay from silently extending a swept DC
    level.
    """

    return _qcs_stability_compensation_interval(
        qcs,
        duration_s=duration_s,
        amplitude=amplitude,
        name=name,
        fabric_hz=fabric_hz,
    )


def _qcs_stability_compensation_interval(
    qcs: Any,
    *,
    duration_s: float,
    amplitude: Any,
    name: str,
    fabric_hz: float,
) -> Any:
    """Build a swept fixed-time compensation tail safe for real M5301 Hold.

    A minimum legal rising ramp establishes the swept level, a direct
    ``Hold`` retains it, a minimum falling ramp removes it, and an explicit
    zero delay terminates the interval before HCL's inter-iteration delay.
    The last item is required because M5301 midpoint sampling leaves the last
    falling-ramp sample slightly nonzero.  The caller scales ``amplitude`` so
    the two half-ramp areas plus the plateau have the requested area exactly.
    """

    duration = _fabric_aligned_seconds(
        duration_s,
        fabric_hz=fabric_hz,
        label=f"{name} duration",
        positive=True,
    )
    duration_cycles = int(round(duration * fabric_hz))
    ramp_cycles = QCS_STABILITY_DC_RAMP_FABRIC_CYCLES
    terminal_zero_cycles = QCS_STABILITY_DC_RAMP_FABRIC_CYCLES
    minimum_cycles = 2 * ramp_cycles + terminal_zero_cycles
    if duration_cycles < minimum_cycles:
        raise ValueError(
            f"{name} duration must be at least {minimum_cycles} QCS "
            "fabric cycles for a ramp-to-hold-to-explicit-zero M5301 "
            "interval"
        )
    hold_cycles = duration_cycles - 2 * ramp_cycles - terminal_zero_cycles
    ramp_duration_s = ramp_cycles / fabric_hz
    ramp_samples = ramp_cycles * QCS_M5301_SAMPLES_PER_FABRIC_CYCLE
    endpoint_sample_offset = 0.5 / ramp_samples
    # QCS samples envelopes at bin midpoints. Extend each ramp's unit endpoint
    # across its final/first half-bin so Hold receives exactly the requested
    # Scalar rather than (1 - 1/(2*N)) times that value.
    rising_envelope = qcs.ArbitraryEnvelope(
        [0.0, 1.0 - endpoint_sample_offset, 1.0],
        [0.0, 1.0, 1.0],
    )
    falling_envelope = qcs.ArbitraryEnvelope(
        [0.0, endpoint_sample_offset, 1.0],
        [1.0, 1.0, 0.0],
    )
    operations = [
        qcs.DCWaveform(
            duration=ramp_duration_s,
            envelope=rising_envelope,
            amplitude=amplitude,
            name=f"{name}_ramp_up",
        )
    ]
    if hold_cycles:
        operations.append(
            qcs.Hold(
                duration=hold_cycles / fabric_hz,
                name=f"{name}_hold",
            )
        )
    operations.append(
        qcs.DCWaveform(
            duration=ramp_duration_s,
            envelope=falling_envelope,
            amplitude=amplitude,
            name=f"{name}_ramp_down",
        )
    )
    operations.append(
        qcs.Delay(
            duration=terminal_zero_cycles / fabric_hz,
            name=f"{name}_terminal_zero",
        )
    )
    return operations


def _dc_waveform(
    qcs: Any,
    *,
    duration_s: float,
    times_s: np.ndarray,
    amplitudes: np.ndarray,
    name: str,
) -> Any:
    amplitudes = np.asarray(amplitudes, dtype=float)
    if amplitudes.ndim != 1 or len(amplitudes) != len(times_s):
        raise ValueError("QCS DC vertex times and amplitudes must be 1D peers")
    scale = float(np.max(np.abs(amplitudes), initial=0.0))
    if scale == 0.0:
        envelope = qcs.ConstantEnvelope()
        amplitude = 0.0
    else:
        # ArbitraryEnvelope normalizes its input to the unit disc.  Supplying
        # the physical scale separately preserves sub-full-scale waveforms.
        envelope = qcs.ArbitraryEnvelope(times_s, amplitudes)
        amplitude = scale
    return qcs.DCWaveform(
        duration=duration_s,
        envelope=envelope,
        amplitude=amplitude,
        name=name,
    )


def _qcs_ramp_dc_interval(
    qcs: Any,
    *,
    duration_cycles: int,
    start_amplitude: float,
    end_amplitude: float,
    name: str,
    fabric_hz: float,
) -> Any:
    """Build one exact linear M5301 DC waveform interval.

    Strictly bipolar ramps are deliberately split into two sequential numeric
    waveforms, ``start -> 0`` and ``0 -> end``.  Hardware loopback testing on
    QCS 2.5.5 showed that adding falling and rising waveforms could omit the
    second (negative) contribution.  The proportional split keeps the two
    slopes equal apart from the M5301 16-sample duration quantization, while
    preserving the original total interval duration exactly.
    """
    duration_cycles = int(duration_cycles)
    if duration_cycles < QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES:
        minimum_s = QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES / fabric_hz
        raise ValueError(
            f"{name} duration must be at least "
            f"{QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES} QCS fabric cycles "
            f"({minimum_s * 1e9:.9g} ns) for an M5301 ramp waveform"
        )
    if (
        duration_cycles
        % QCS_M5301_WAVEFORM_GRANULARITY_FABRIC_CYCLES
    ):
        raise ValueError(
            f"{name} duration must be a multiple of "
            f"{QCS_M5301_WAVEFORM_GRANULARITY_FABRIC_CYCLES} QCS "
            "fabric cycles for the M5301 16-sample waveform granularity"
        )

    start_amplitude = float(start_amplitude)
    end_amplitude = float(end_amplitude)
    bipolar = bool(
        (start_amplitude > 0.0 and end_amplitude < 0.0)
        or (start_amplitude < 0.0 and end_amplitude > 0.0)
    )
    if bipolar:
        minimum_split_cycles = 2 * QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES
        if duration_cycles < minimum_split_cycles:
            minimum_s = minimum_split_cycles / fabric_hz
            raise QcsUnsupportedFeatureError(
                f"{name} crosses 0 V and requires at least "
                f"{minimum_split_cycles} QCS fabric cycles "
                f"({minimum_s * 1e9:.9g} ns) for two safe M5301 ramp "
                "waveforms"
            )
        excursion = abs(start_amplitude) + abs(end_amplitude)
        raw_first_cycles = duration_cycles * abs(start_amplitude) / excursion
        granularity = QCS_M5301_WAVEFORM_GRANULARITY_FABRIC_CYCLES
        first_cycles = granularity * int(
            np.floor(raw_first_cycles / granularity + 0.5)
        )
        first_cycles = int(
            np.clip(
                first_cycles,
                QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES,
                duration_cycles - QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES,
            )
        )
        second_cycles = duration_cycles - first_cycles
        return [
            _qcs_ramp_dc_interval(
                qcs,
                duration_cycles=first_cycles,
                start_amplitude=start_amplitude,
                end_amplitude=0.0,
                name=f"{name}_to_zero",
                fabric_hz=fabric_hz,
            ),
            _qcs_ramp_dc_interval(
                qcs,
                duration_cycles=second_cycles,
                start_amplitude=0.0,
                end_amplitude=end_amplitude,
                name=f"{name}_from_zero",
                fabric_hz=fabric_hz,
            ),
        ]

    duration_s = duration_cycles / fabric_hz
    return _dc_waveform(
        qcs,
        duration_s=duration_s,
        times_s=np.asarray([0.0, duration_s], dtype=float),
        amplitudes=np.asarray(
            [float(start_amplitude), float(end_amplitude)],
            dtype=float,
        ),
        name=name,
    )


def _qcs_m5301_rendered_fabric_cycles(
    *,
    times_cycles: np.ndarray,
    amplitudes: np.ndarray,
    name: str,
    append_terminal_value: bool = False,
    allow_continuous_ramp_holds: bool = False,
) -> int:
    """Return rendered M5301 cycles using the selected lowering rules."""
    raw_times = np.asarray(times_cycles, dtype=float)
    values = np.asarray(amplitudes, dtype=float)
    if (
        raw_times.ndim != 1
        or values.ndim != 1
        or raw_times.shape != values.shape
        or raw_times.size < 2
    ):
        raise ValueError("QCS DC vertex times and amplitudes must be 1D peers")
    if not np.all(np.isfinite(raw_times)) or not np.all(np.isfinite(values)):
        raise ValueError("QCS DC vertex times and amplitudes must be finite")
    integer_times = np.rint(raw_times).astype(np.int64)
    if not np.allclose(raw_times, integer_times, rtol=0.0, atol=1e-9):
        raise ValueError("QCS DC vertex times must be whole fabric cycles")
    if integer_times[0] != 0 or np.any(np.diff(integer_times) < 0):
        raise ValueError(
            "QCS DC vertex times must start at zero and be nondecreasing"
        )

    group_starts = np.flatnonzero(np.r_[True, np.diff(integer_times) != 0])
    group_stops = np.r_[group_starts[1:], integer_times.size]
    unique_times = integer_times[group_starts]
    if unique_times.size < 2 or unique_times[-1] <= 0:
        raise ValueError("QCS DC vertices must have a positive duration")

    rendered_cycles = 0
    retained_output = False
    previous_end_value = None
    for interval_index in range(unique_times.size - 1):
        duration_cycles = int(
            unique_times[interval_index + 1]
            - unique_times[interval_index]
        )
        start_value = float(values[group_stops[interval_index] - 1])
        end_value = float(values[group_starts[interval_index + 1]])
        pointwise_constant = bool(
            np.isclose(start_value, end_value, rtol=0.0, atol=1e-15)
        )
        is_zero_delay = bool(
            pointwise_constant
            and np.isclose(start_value, 0.0, rtol=0.0, atol=1e-15)
        )
        is_retained_hold = bool(
            allow_continuous_ramp_holds
            and retained_output
            and pointwise_constant
            and not is_zero_delay
            and previous_end_value is not None
            and np.isclose(
                start_value,
                previous_end_value,
                rtol=0.0,
                atol=1e-15,
            )
            and duration_cycles >= QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES
        )
        if is_zero_delay:
            retained_output = False
            previous_end_value = end_value
            continue
        if is_retained_hold:
            retained_output = True
            previous_end_value = end_value
            continue
        interval_name = f"{name}_interval_{interval_index}"
        if duration_cycles < QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES:
            raise ValueError(
                f"{interval_name} duration must be at least "
                f"{QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES} QCS fabric cycles "
                "(13.333333 ns) for an M5301 waveform"
            )
        if pointwise_constant:
            seed_cycles = min(
                duration_cycles,
                QCS_M5301_HOLD_SEED_FABRIC_CYCLES,
            )
            seed_cycles -= (
                seed_cycles
                % QCS_M5301_WAVEFORM_GRANULARITY_FABRIC_CYCLES
            )
            rendered_cycles += seed_cycles
            retained_output = True
            previous_end_value = end_value
            continue
        if duration_cycles % QCS_M5301_WAVEFORM_GRANULARITY_FABRIC_CYCLES:
            raise ValueError(
                f"{interval_name} duration must be a multiple of "
                f"{QCS_M5301_WAVEFORM_GRANULARITY_FABRIC_CYCLES} QCS "
                "fabric cycles (6.666667 ns) for the M5301 16-sample "
                "waveform granularity"
            )
        rendered_cycles += duration_cycles
        retained_output = True
        previous_end_value = end_value

    if append_terminal_value:
        terminal_value = float(values[group_stops[-1] - 1])
        if not np.isclose(terminal_value, 0.0, rtol=0.0, atol=1e-15):
            rendered_cycles += QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES
    return int(rendered_cycles)


def _raise_qcs_m5301_capacity_error(
    *,
    output_name: str,
    rendered_fabric_cycles: int,
    point_index: Optional[int] = None,
) -> None:
    rendered_samples = (
        int(rendered_fabric_cycles) * QCS_M5301_SAMPLES_PER_FABRIC_CYCLE
    )
    rendered_duration_us = int(rendered_fabric_cycles) / (
        QCS_FABRIC_CLOCK_HZ / 1e6
    )
    point_text = (
        "" if point_index is None else f" at sweep point {int(point_index) + 1}"
    )
    raise QcsUnsupportedFeatureError(
        f"QCS M5301 waveform capacity exceeded for output {output_name!r}"
        f"{point_text}: {rendered_samples:,} / "
        f"{QCS_M5301_MAX_RENDERED_SAMPLES:,} samples "
        f"({rendered_duration_us:.6f} / 40.960000 us). The 40.960 us "
        "capacity includes every ramp and the minimum DCWaveform seed used to "
        "establish each independent nonzero plateau on that physical output. "
        "The rest of a fixed plateau uses QCS Hold, and zero-voltage delays "
        "use no waveform samples. Shorten the remaining rendered intervals."
    )


def qcs_m5301_capacity_preview_point_indices(sequence: Any) -> Tuple[int, ...]:
    """Return a small, deterministic set of sweep boundary/midpoint indices."""
    point_count = int(sequence.sweep_point_count)
    if point_count < 1:
        raise ValueError("QCS sequence must contain at least one sweep point")
    axes = tuple(sequence.sweep_axes)
    if not axes:
        return (0,)
    shape = tuple(int(axis.count) for axis in axes)
    centers = tuple((count - 1) // 2 for count in shape)
    selected = {0, point_count - 1}
    selected.add(int(np.ravel_multi_index(centers, shape, order="C")))

    for axis_index, count in enumerate(shape):
        for value in sorted({0, (count - 1) // 2, count - 1}):
            indices = list(centers)
            indices[axis_index] = int(value)
            selected.add(
                int(np.ravel_multi_index(tuple(indices), shape, order="C"))
            )

    # Interactions between a practical number of axes are checked at every
    # corner. High-dimensional sweeps retain the per-axis probes above so the
    # live GUI preview remains responsive; exhaustive validation still occurs
    # as each QCS point is compiled before hardware execution.
    if len(shape) <= 6:
        for indices in product(
            *((0,) if count <= 1 else (0, count - 1) for count in shape)
        ):
            selected.add(
                int(np.ravel_multi_index(indices, shape, order="C"))
            )
    return tuple(sorted(selected))


def qcs_m5301_waveform_capacity_report(
    sequence: Any,
    *,
    point_indices: Optional[Sequence[int]] = None,
    fabric_mhz: float = 300.0,
    amplitude_scale: float = 1.0,
    auto_fixed_dc_offsets: bool = False,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    dc_full_scale_v: float = DEFAULT_QCS_FULL_SCALE_V,
) -> QcsM5301CapacityReport:
    """Measure per-output M5301 usage without physical-channel offsets.

    ``auto_fixed_dc_offsets`` is retained only as a source-compatible legacy
    keyword. It no longer changes the waveform or capacity calculation.
    """
    fabric_hz = _positive_finite(fabric_mhz, "fabric clock") * 1.0e6
    if not np.isclose(
        fabric_hz,
        QCS_FABRIC_CLOCK_HZ,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "QCS M5000-series execution requires the fixed 300 MHz "
            "synchronization fabric clock"
        )
    amplitude_scale = _positive_finite(
        amplitude_scale,
        "QCS DC waveform amplitude scale",
    )
    point_count = int(sequence.sweep_point_count)
    if point_count < 1:
        raise ValueError("QCS sequence must contain at least one sweep point")
    if point_indices is None:
        inspected = tuple(range(point_count))
    else:
        inspected = tuple(dict.fromkeys(int(index) for index in point_indices))
        if not inspected:
            raise ValueError("QCS capacity point_indices must not be empty")
        if any(index < 0 or index >= point_count for index in inspected):
            raise IndexError("QCS capacity point index is out of range")

    output_names = tuple(str(name) for name in sequence.output_names)
    if not output_names:
        raise ValueError("QCS sequence must contain at least one DC output")
    # Legacy arguments remain accepted for callers that persist preview
    # settings, but ChannelMapper offsets are intentionally not part of the
    # waveform model.
    del auto_fixed_dc_offsets, source_full_scale_mv, dc_full_scale_v
    worst_cycles = [0] * len(output_names)
    worst_points = [inspected[0]] * len(output_names)
    worst_coordinates = [tuple(sequence.sweep_coordinate(inspected[0]))] * len(
        output_names
    )

    for point_index in inspected:
        (
            times_cycles,
            waveforms,
            _boundaries,
            force_terminal_layout,
        ) = _qcs_synchronized_waveform_vertices(sequence, point_index)
        times_cycles = np.asarray(times_cycles, dtype=float)
        source_waveforms = {
            output_name: (
                np.asarray(waveforms[output_name], dtype=float)
                * amplitude_scale
            )
            for output_name in output_names
        }
        if any(
            values.ndim != 1 or values.shape != times_cycles.shape
            for values in source_waveforms.values()
        ):
            raise ValueError(
                "QCS compensated waveform vertices must match the time vector"
            )
        terminal_indices = np.flatnonzero(times_cycles == times_cycles[-1])
        append_terminal_value = bool(
            force_terminal_layout
            or (
                terminal_indices.size > 1
                and any(
                    not np.isclose(
                        values[terminal_indices[0]],
                        values[terminal_indices[-1]],
                        rtol=0.0,
                        atol=1e-15,
                    )
                    for values in source_waveforms.values()
                )
            )
        )
        coordinate = tuple(sequence.sweep_coordinate(point_index))
        for output_index, output_name in enumerate(output_names):
            rendered_cycles = _qcs_m5301_rendered_fabric_cycles(
                times_cycles=times_cycles,
                amplitudes=source_waveforms[output_name],
                name=f"{output_name}_point_{point_index}",
                append_terminal_value=append_terminal_value,
                # A plateau that directly continues the preceding ramp can
                # retain that endpoint with QCS Hold in every execution mode.
                # This is independent of Bias-T compensation. Independently
                # established constants are charged for their seed waveform.
                allow_continuous_ramp_holds=True,
            )
            if rendered_cycles > worst_cycles[output_index]:
                worst_cycles[output_index] = rendered_cycles
                worst_points[output_index] = int(point_index)
                worst_coordinates[output_index] = coordinate

    channels = tuple(
        QcsM5301ChannelCapacity(
            output_name=output_name,
            rendered_fabric_cycles=worst_cycles[output_index],
            point_index=worst_points[output_index],
            sweep_coordinate=worst_coordinates[output_index],
        )
        for output_index, output_name in enumerate(output_names)
    )
    return QcsM5301CapacityReport(
        channels=channels,
        inspected_point_count=len(inspected),
        sweep_point_count=point_count,
        dc_channel_offsets_v=(0.0,) * len(output_names),
    )


def validate_qcs_m5301_waveform_capacity(
    sequence: Any,
    *,
    point_indices: Optional[Sequence[int]] = None,
    fabric_mhz: float = 300.0,
    amplitude_scale: float = 1.0,
    auto_fixed_dc_offsets: bool = False,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    dc_full_scale_v: float = DEFAULT_QCS_FULL_SCALE_V,
) -> QcsM5301CapacityReport:
    """Raise a user-facing error when any inspected output exceeds capacity."""
    report = qcs_m5301_waveform_capacity_report(
        sequence,
        point_indices=point_indices,
        fabric_mhz=fabric_mhz,
        amplitude_scale=amplitude_scale,
        auto_fixed_dc_offsets=auto_fixed_dc_offsets,
        source_full_scale_mv=source_full_scale_mv,
        dc_full_scale_v=dc_full_scale_v,
    )
    channel = report.worst_channel
    if channel.rendered_fabric_cycles > QCS_M5301_MAX_RENDERED_FABRIC_CYCLES:
        _raise_qcs_m5301_capacity_error(
            output_name=channel.output_name,
            rendered_fabric_cycles=channel.rendered_fabric_cycles,
            point_index=channel.point_index,
        )
    return report


def _qcs_dc_waveform_operations(
    qcs: Any,
    *,
    times_cycles: np.ndarray,
    amplitudes: np.ndarray,
    name: str,
    fabric_hz: float,
    append_terminal_value: bool = False,
    allow_continuous_ramp_holds: bool = False,
    grouped: bool = False,
) -> Any:
    """Lower piecewise-linear vertices to sequential M5301 operations.

    Repeated vertex times encode instantaneous SET transitions. For an
    interval, the value after the SET at its start is joined to the value
    before the SET at its end. Nonzero constant intervals and changing
    intervals use real DCWaveforms; zero intervals use Delay.

    A fixed nonzero plateau starts with the shortest legal constant
    ``DCWaveform`` and uses ``Hold`` for its remaining duration. If a preceding
    waveform has already established the same endpoint, the complete plateau
    can be represented by ``Hold`` without another seed.
    """
    rendered_cycles = _qcs_m5301_rendered_fabric_cycles(
        times_cycles=times_cycles,
        amplitudes=amplitudes,
        name=name,
        append_terminal_value=append_terminal_value,
        allow_continuous_ramp_holds=allow_continuous_ramp_holds,
    )
    if rendered_cycles > QCS_M5301_MAX_RENDERED_FABRIC_CYCLES:
        _raise_qcs_m5301_capacity_error(
            output_name=name,
            rendered_fabric_cycles=rendered_cycles,
        )

    raw_times = np.asarray(times_cycles, dtype=float)
    values = np.asarray(amplitudes, dtype=float)
    if (
        raw_times.ndim != 1
        or values.ndim != 1
        or raw_times.shape != values.shape
        or raw_times.size < 2
    ):
        raise ValueError("QCS DC vertex times and amplitudes must be 1D peers")
    if not np.all(np.isfinite(raw_times)) or not np.all(np.isfinite(values)):
        raise ValueError("QCS DC vertex times and amplitudes must be finite")
    integer_times = np.rint(raw_times).astype(np.int64)
    if not np.allclose(raw_times, integer_times, rtol=0.0, atol=1e-9):
        raise ValueError("QCS DC vertex times must be whole fabric cycles")
    if integer_times[0] != 0 or np.any(np.diff(integer_times) < 0):
        raise ValueError(
            "QCS DC vertex times must start at zero and be nondecreasing"
        )

    group_starts = np.flatnonzero(np.r_[True, np.diff(integer_times) != 0])
    group_stops = np.r_[group_starts[1:], integer_times.size]
    unique_times = integer_times[group_starts]
    if unique_times.size < 2 or unique_times[-1] <= 0:
        raise ValueError("QCS DC vertices must have a positive duration")

    interval_durations_s = (
        np.diff(unique_times).astype(float) / float(fabric_hz)
    )
    interval_starts = values[group_stops[:-1] - 1]
    interval_ends = values[group_starts[1:]]
    if allow_continuous_ramp_holds:
        hold_mask = _qcs_parameterized_dc_hold_mask(
            duration_table_s=interval_durations_s[np.newaxis, :],
            start_table=interval_starts[np.newaxis, :],
            end_table=interval_ends[np.newaxis, :],
            fabric_hz=fabric_hz,
        )
    else:
        hold_mask = np.zeros(interval_durations_s.shape, dtype=bool)

    interval_operations = []
    for interval_index in range(unique_times.size - 1):
        duration_cycles = int(
            unique_times[interval_index + 1]
            - unique_times[interval_index]
        )
        start_value = float(values[group_stops[interval_index] - 1])
        end_value = float(values[group_starts[interval_index + 1]])
        interval_name = f"{name}_interval_{interval_index}"
        if hold_mask[interval_index]:
            interval_operations.append(
                qcs.Hold(
                    duration=duration_cycles / fabric_hz,
                    name=f"{interval_name}_hold",
                )
            )
            continue
        if np.isclose(start_value, end_value, rtol=0.0, atol=1e-15):
            duration_s = duration_cycles / fabric_hz
            if np.isclose(start_value, 0.0, rtol=0.0, atol=1e-15):
                interval = qcs.Delay(
                    duration=duration_s,
                    name=interval_name,
                )
            else:
                interval = _qcs_constant_dc_interval(
                    qcs,
                    duration_s=duration_s,
                    amplitude=start_value,
                    name=interval_name,
                    fabric_hz=fabric_hz,
                )
        else:
            interval = _qcs_ramp_dc_interval(
                qcs,
                duration_cycles=duration_cycles,
                start_amplitude=start_value,
                end_amplitude=end_value,
                name=interval_name,
                fabric_hz=fabric_hz,
            )
        interval_operations.append(interval)

    if append_terminal_value:
        terminal_value = float(values[group_stops[-1] - 1])
        terminal_duration_s = (
            QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES / fabric_hz
        )
        if np.isclose(terminal_value, 0.0, rtol=0.0, atol=1e-15):
            terminal = qcs.Delay(
                duration=terminal_duration_s,
                name=f"{name}_terminal_zero",
            )
        else:
            terminal = qcs.DCWaveform(
                duration=terminal_duration_s,
                envelope=qcs.ConstantEnvelope(),
                amplitude=terminal_value,
                name=f"{name}_terminal_set",
            )
        interval_operations.append(terminal)

    if grouped:
        return interval_operations
    operations = []
    for interval in interval_operations:
        if isinstance(interval, (list, tuple)):
            operations.extend(interval)
        else:
            operations.append(interval)
    return operations[0] if len(operations) == 1 else operations


def _point_rf_pulse(
    sequence: Any,
    pulse: QcsRfPulseConfig,
    point_index: int,
) -> QcsRfPulseConfig:
    duration_s = pulse.duration_s
    frequency_hz = pulse.frequency_hz
    coordinate = sequence.sweep_coordinate(point_index)
    for axis_index, axis in enumerate(sequence.sweep_axes):
        if (
            isinstance(axis, RfDurationSweep)
            and axis.gen_ch == pulse.gen_ch
            and axis.segment_name == pulse.at_segment
        ):
            duration_s = float(coordinate[axis_index]) * 1e-6
        elif (
            isinstance(axis, RfFrequencySweep)
            and axis.gen_ch == pulse.gen_ch
            and axis.segment_name == pulse.at_segment
        ):
            frequency_hz = float(coordinate[axis_index]) * 1e6
        elif (
            isinstance(axis, RfPowerSweep)
            and axis.gen_ch == pulse.gen_ch
            and axis.segment_name == pulse.at_segment
        ):
            raise QcsUnsupportedFeatureError(
                "calibrated RF connector-power sweeps are QICK-specific; "
                "use a QCS amplitude calibration before enabling this sweep"
            )
    return QcsRfPulseConfig(
        gen_ch=pulse.gen_ch,
        at_segment=pulse.at_segment,
        duration_s=duration_s,
        amplitude=pulse.amplitude,
        frequency_hz=frequency_hz,
        phase_rad=pulse.phase_rad,
        delay_s=pulse.delay_s,
        envelope=pulse.envelope,
        require_within_segment=pulse.require_within_segment,
        power_calibration=pulse.power_calibration,
        power_calibration_provenance=(
            pulse.power_calibration_provenance
        ),
    )


def resolve_qcs_rf_power_calibrations(
    *,
    connection_config: QcsConnectionConfig,
    mapper: Any,
    rf_pulses: Sequence[QcsRfPulseConfig],
) -> Tuple[QcsRfPulseConfig, ...]:
    """Resolve fixed connector dBm requests to mapped M5300 amplitudes.

    Resolution intentionally happens after loading the active ChannelMapper.
    That lets the calibration loader verify the exact M5300 output SMA and
    active M5300 LO before a program is compiled or submitted to hardware.
    The M5200 used to create the calibration and its scoped mapper digest are
    retained as provenance; they are not properties of a later output path.
    """

    pulses = tuple(rf_pulses)
    if not any(pulse.power_calibration is not None for pulse in pulses):
        return pulses
    acquisition_name = connection_config.acquisition_channel_name
    if acquisition_name is None:
        raise ValueError(
            "QCS calibrated RF output power requires a mapped M5200 "
            "acquisition input"
        )
    try:
        from .qcs_rf_power_calibration import (
            load_m5300_power_calibration,
            resolve_m5300_m5200_identities,
        )
    except ImportError:
        from qcs_rf_power_calibration import (
            load_m5300_power_calibration,
            resolve_m5300_m5200_identities,
        )

    resolved = []
    for pulse in pulses:
        request = pulse.power_calibration
        if request is None:
            resolved.append(pulse)
            continue
        try:
            output_name = connection_config.rf_channel_names[pulse.gen_ch]
        except KeyError as exc:
            raise KeyError(
                "QCS calibrated RF output power has no mapped virtual "
                f"channel for gen_ch {pulse.gen_ch}"
            ) from exc
        output_identity, _input_identity, lo_frequency_hz = (
            resolve_m5300_m5200_identities(
                mapper,
                output_name,
                acquisition_name,
            )
        )
        calibration = load_m5300_power_calibration(
            request.database_path,
            run_id=(request.run_id or None),
            expected_output=output_identity,
            expected_input=None,
            expected_mapper_sha256=None,
            expected_lo_frequency_hz=lo_frequency_hz,
            required_frequencies_hz=[pulse.frequency_hz],
            termination_ohm=50.0,
        )
        relative_amplitude = float(
            calibration.relative_amplitudes_for_power(
                [pulse.frequency_hz],
                request.target_power_dbm,
                allow_power_extrapolation=False,
            )[0]
        )
        provenance = dict(calibration.provenance)
        provenance.update(
            {
                "target_power_dbm": float(request.target_power_dbm),
                "frequency_hz": float(pulse.frequency_hz),
                "full_scale_power_dbm": float(
                    calibration.full_scale_power_dbm(
                        [pulse.frequency_hz]
                    )[0]
                ),
                "relative_amplitude": relative_amplitude,
                "power_extrapolation": False,
            }
        )
        resolved.append(
            replace(
                pulse,
                amplitude=relative_amplitude,
                power_calibration_provenance=provenance,
            )
        )
    return tuple(resolved)


def _qcs_rf_output_details(
    rf_pulses: Sequence[QcsRfPulseConfig],
) -> Tuple[Mapping[str, Any], ...]:
    details = []
    for pulse in rf_pulses:
        current = {
            "gen_ch": pulse.gen_ch,
            "amplitude": pulse.amplitude,
            "frequency_hz": pulse.frequency_hz,
            "duration_s": pulse.duration_s,
            "delay_s": pulse.delay_s,
        }
        if pulse.power_calibration_provenance is not None:
            current["power_calibration"] = dict(
                pulse.power_calibration_provenance
            )
        details.append(current)
    return tuple(details)


def validate_qcs_capabilities(
    *,
    connection_config: QcsConnectionConfig,
    sequence: Any,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
) -> None:
    """Reject configurations whose semantics cannot be preserved."""
    if len(connection_config.dc_channel_names) != int(sequence.n_outputs):
        raise ValueError(
            "QCS DC channel count must match the waveform output count "
            f"({len(connection_config.dc_channel_names)} configured, "
            f"{sequence.n_outputs} required)"
        )
    if not connection_config.blocking:
        raise QcsUnsupportedFeatureError(
            "QCS Experiment execution requires blocking=True so acquisition "
            "data is complete before it is normalized and saved"
        )
    segment_names = {str(segment.name) for segment in sequence.segments}
    for pulse in rf_pulses:
        if pulse.power_calibration is None:
            continue
        if any(
            isinstance(axis, RfFrequencySweep)
            and axis.gen_ch == pulse.gen_ch
            and axis.segment_name == pulse.at_segment
            for axis in sequence.sweep_axes
        ):
            raise QcsUnsupportedFeatureError(
                "QCS calibrated connector power in AWG Tuning currently "
                "requires one fixed RF frequency; disable the RF frequency "
                "sweep or use direct relative amplitude"
            )
    for axis in sequence.sweep_axes:
        if isinstance(axis, RfPowerSweep):
            raise QcsUnsupportedFeatureError(
                "calibrated RF connector-power sweeps are not yet available "
                "on the QCS backend"
            )
    for pulse in rf_pulses:
        if pulse.at_segment not in segment_names:
            raise KeyError(
                f"QCS RF pulse references unknown segment {pulse.at_segment!r}"
            )
        if pulse.gen_ch not in connection_config.rf_channel_names:
            raise KeyError(
                f"no QCS virtual RF channel is mapped for gen_ch {pulse.gen_ch}"
            )
    if acquisition is not None:
        if acquisition.at_segment not in segment_names:
            raise KeyError(
                "QCS acquisition references unknown segment "
                f"{acquisition.at_segment!r}"
            )
        if connection_config.acquisition_channel_name is None:
            raise ValueError(
                "a QCS acquisition virtual-channel name is required"
            )
        if (
            not connection_config.hw_demod
            and acquisition.integration_filter is not None
        ):
            raise QcsUnsupportedFeatureError(
                "a custom QCS integration filter requires hardware "
                "demodulation; raw-trace mode uses an acquisition duration"
            )


def compile_qcs_point(
    sequence: Any,
    point_index: int,
    *,
    connection_config: QcsConnectionConfig,
    mapper: Any,
    repetitions_per_sweep: int,
    fabric_mhz: float = 300.0,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
    qcs_module=None,
) -> QcsCompiledPoint:
    """Compile one Cartesian point into an executable QCS Program."""
    qcs = _import_qcs() if qcs_module is None else qcs_module
    if isinstance(repetitions_per_sweep, bool):
        raise TypeError("repetitions_per_sweep must be an integer")
    repetitions = int(repetitions_per_sweep)
    if repetitions < 1 or repetitions != repetitions_per_sweep:
        raise ValueError("repetitions_per_sweep must be a positive integer")
    point_index = int(point_index)
    if not 0 <= point_index < int(sequence.sweep_point_count):
        raise IndexError("QCS point index is out of range")
    validate_qcs_capabilities(
        connection_config=connection_config,
        sequence=sequence,
        rf_pulses=rf_pulses,
        acquisition=acquisition,
    )

    (
        times_cycles,
        waveforms,
        boundaries,
        force_terminal_layout,
    ) = _qcs_synchronized_waveform_vertices(sequence, point_index)
    fabric_hz = _positive_finite(fabric_mhz, "fabric clock") * 1e6
    if not np.isclose(
        fabric_hz,
        QCS_FABRIC_CLOCK_HZ,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "QCS M5000-series execution requires the fixed 300 MHz "
            "synchronization fabric clock"
        )
    seconds_per_cycle = 1.0 / fabric_hz
    relative_amplitude_scale = _positive_finite(
        source_full_scale_mv, "source waveform full scale"
    ) / (connection_config.dc_full_scale_v * 1000.0)
    times_cycles = np.asarray(times_cycles, dtype=float)
    if (
        times_cycles.ndim != 1
        or len(times_cycles) < 2
        or times_cycles[-1] <= 0.0
    ):
        raise ValueError("QCS sequence must have a positive duration")
    integer_times = np.rint(times_cycles).astype(np.int64)
    if not np.allclose(times_cycles, integer_times, rtol=0.0, atol=1e-9):
        raise ValueError("QCS DC vertex times must be whole fabric cycles")
    unique_times = integer_times[
        np.flatnonzero(np.r_[True, np.diff(integer_times) != 0])
    ]
    interval_start_times_s = (
        unique_times[:-1].astype(float) * seconds_per_cycle
    )
    source_waveforms = {
        output_name: np.asarray(waveforms[output_name], dtype=float)
        for output_name in sequence.output_names
    }
    if any(
        values.ndim != 1 or values.shape != times_cycles.shape
        for values in source_waveforms.values()
    ):
        raise ValueError(
            "QCS compensated waveform vertices must match the time vector"
        )
    terminal_indices = np.flatnonzero(times_cycles == times_cycles[-1])
    append_terminal_value = bool(
        force_terminal_layout
        or (
            terminal_indices.size > 1
            and any(
                not np.isclose(
                    values[terminal_indices[0]],
                    values[terminal_indices[-1]],
                    rtol=0.0,
                    atol=1e-15,
                )
                for values in source_waveforms.values()
            )
        )
    )
    duration_s = float(times_cycles[-1]) * seconds_per_cycle
    if append_terminal_value:
        # A zero-time SET at the end has no following interval in which to
        # establish its new value. Emit a minimum-length terminal operation
        # on every DC lane so the state is explicit and layers stay aligned.
        duration_s += (
            QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES * seconds_per_cycle
        )
        interval_start_times_s = np.r_[
            interval_start_times_s,
            float(unique_times[-1]) * seconds_per_cycle,
        ]
    boundary_seconds = _segment_boundaries_seconds(
        boundaries, fabric_mhz=fabric_mhz
    )

    program = qcs.Program(name=f"PulseGenerator point {point_index + 1}")
    dc_channels = []
    dc_interval_operations = []
    for output_index, (output_name, channel_name) in enumerate(
        zip(sequence.output_names, connection_config.dc_channel_names)
    ):
        channel = _resolve_mapper_channel(mapper, channel_name)
        dc_channels.append(channel)
        source_amplitudes = source_waveforms[output_name]
        peak_voltage_v = (
            float(np.max(np.abs(source_amplitudes), initial=0.0))
            * float(source_full_scale_mv)
            / 1000.0
        )
        if peak_voltage_v > connection_config.dc_full_scale_v + 1e-12:
            raise ValueError(
                f"QCS DC output {output_name!r} at software point "
                f"{point_index + 1} reaches {peak_voltage_v:.6g} V, "
                "exceeding the configured +/-"
                f"{connection_config.dc_full_scale_v:.6g} V full scale"
            )
        waveform_intervals = _qcs_dc_waveform_operations(
            qcs,
            times_cycles=times_cycles,
            amplitudes=(
                source_amplitudes * relative_amplitude_scale
            ),
            name=f"{output_name}_point_{point_index}",
            fabric_hz=fabric_hz,
            append_terminal_value=append_terminal_value,
            # Match the synchronized/hardware-sweep lowering: once a ramp
            # establishes a voltage, a directly continuous plateau can use
            # Hold without consuming additional rendered-waveform samples.
            allow_continuous_ramp_holds=True,
            grouped=True,
        )
        dc_interval_operations.append(waveform_intervals)

    interval_count = len(interval_start_times_s)
    if any(
        len(operations) != interval_count
        for operations in dc_interval_operations
    ):
        raise ValueError("QCS DC outputs produced different interval layers")
    interval_duration_values_s = (
        np.diff(unique_times).astype(float) * seconds_per_cycle
    )
    if append_terminal_value:
        interval_duration_values_s = np.r_[
            interval_duration_values_s,
            QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES * seconds_per_cycle,
        ]
    (
        interval_groups,
        segment_layer_indices,
        segment_layer_offsets,
    ) = _qcs_segment_grouped_layer_layout(
        sequence,
        duration_table_s=interval_duration_values_s[np.newaxis, :],
        boundary_rows=(boundary_seconds,),
    )
    layer_count = len(interval_groups)

    rf_operations_by_layer: dict[int, list[dict[str, Any]]] = {}

    for pulse in rf_pulses:
        current = _point_rf_pulse(sequence, pulse, point_index)
        if current.at_segment not in boundary_seconds:
            raise KeyError(
                f"no timing boundary for RF segment {current.at_segment!r}"
            )
        segment_start_s, segment_stop_s = boundary_seconds[
            current.at_segment
        ]
        layer_index = segment_layer_indices[current.at_segment]
        local_pre_delay_s = current.delay_s
        pre_delay_s = (
            float(segment_layer_offsets[current.at_segment][0])
            + local_pre_delay_s
        )
        if (
            current.require_within_segment
            and segment_start_s + local_pre_delay_s + current.duration_s
            > segment_stop_s + 1e-15
        ):
            raise ValueError(
                f"QCS RF pulse on gen_ch {current.gen_ch} exceeds segment "
                f"{current.at_segment!r}"
            )
        channel_name = connection_config.rf_channel_names[current.gen_ch]
        channel = _resolve_mapper_channel(mapper, channel_name)
        waveform = qcs.RFWaveform(
            duration=current.duration_s,
            envelope=_qcs_envelope(qcs, current.envelope),
            amplitude=current.amplitude,
            rf_frequency=current.frequency_hz,
            instantaneous_phase=current.phase_rad,
            name=f"rf_{current.gen_ch}_point_{point_index}",
        )
        if (
            segment_start_s + local_pre_delay_s + current.duration_s
            > segment_stop_s + 1e-15
            and layer_index < layer_count - 1
        ):
            raise ValueError(
                f"QCS RF pulse on gen_ch {current.gen_ch} cannot cross "
                f"segment {current.at_segment!r} while later DC segment "
                "layers remain"
            )
        rf_operations_by_layer.setdefault(layer_index, []).append(
            {
                "channel": channel,
                "channel_name": channel_name,
                "waveform": waveform,
                "start_s": pre_delay_s,
                "duration_s": current.duration_s,
                "gen_ch": current.gen_ch,
            }
        )

    acquisition_channels = None
    acquisition_duration_s = None
    acquisition_sample_rate_hz = None
    acquisition_sample_count = None
    integration_segment_sample_counts: tuple[int, ...] = ()
    acquisition_elapsed_duration_s = None
    acquisition_layer_index = None
    acquisition_operations: list[dict[str, Any]] = []
    if acquisition is not None:
        segment_start_s, segment_stop_s = boundary_seconds[
            acquisition.at_segment
        ]
        acquisition_layer_index = segment_layer_indices[
            acquisition.at_segment
        ]
        local_acquisition_pre_delay_s = acquisition.pre_delay_s
        pre_delay_s = (
            float(segment_layer_offsets[acquisition.at_segment][0])
            + local_acquisition_pre_delay_s
        )
        acquisition_channels = _resolve_mapper_channel(
            mapper, connection_config.acquisition_channel_name
        )
        (
            acquisition_duration_s,
            acquisition_sample_rate_hz,
        ) = _resolved_acquisition_timing(
            mapper,
            acquisition_channels,
            acquisition,
            hardware_demodulation=connection_config.hw_demod,
        )
        acquisition_sample_count = int(
            round(acquisition_duration_s * acquisition_sample_rate_hz)
        )
        if connection_config.hw_demod:
            integration_segment_sample_counts = (
                _qcs_iq_integration_segment_sample_counts(
                    acquisition_sample_count
                )
            )
            acquisition_sample_count = int(
                sum(integration_segment_sample_counts)
            )
            acquisition_duration_s = (
                acquisition_sample_count / acquisition_sample_rate_hz
            )
            acquisition_elapsed_duration_s = (
                acquisition_duration_s
                + max(0, len(integration_segment_sample_counts) - 1)
                * QCS_SPARAMETER_INTER_SEGMENT_DELAY_S
            )
        else:
            integration_segment_sample_counts = (acquisition_sample_count,)
            acquisition_elapsed_duration_s = acquisition_duration_s
        acquisition_stop_s = (
            segment_start_s
            + local_acquisition_pre_delay_s
            + acquisition_elapsed_duration_s
        )
        if (
            connection_config.hw_demod
            and acquisition_stop_s > segment_stop_s + 1e-15
        ):
            raise ValueError(
                "QCS acquisition exceeds segment "
                f"{acquisition.at_segment!r}"
            )
        if (
            not connection_config.hw_demod
            and acquisition_stop_s > segment_stop_s + 1e-15
            and acquisition_layer_index < layer_count - 1
        ):
            raise ValueError(
                "raw QCS acquisition cannot cross segment "
                f"{acquisition.at_segment!r} while later DC segment layers "
                "remain"
            )
        if connection_config.hw_demod:
            if (
                len(integration_segment_sample_counts) > 1
                and acquisition.integration_filter is not None
            ):
                raise QcsUnsupportedFeatureError(
                    "segmented QCS I/Q integration requires the built-in "
                    "flat filter; a custom filter cannot be divided safely"
                )
            integration_filter_cache: dict[int, Any] = {}
            for segment_index, segment_samples in enumerate(
                integration_segment_sample_counts
            ):
                integration_filter = acquisition.integration_filter
                if integration_filter is None:
                    integration_filter = integration_filter_cache.get(
                        segment_samples
                    )
                    if integration_filter is None:
                        filter_waveform = qcs.RFWaveform(
                            duration=(
                                segment_samples / acquisition_sample_rate_hz
                            ),
                            envelope=_qcs_envelope(qcs, acquisition.envelope),
                            amplitude=1.0,
                            rf_frequency=acquisition.frequency_hz,
                            instantaneous_phase=acquisition.phase_rad,
                            name=(
                                f"acquisition_filter_point_{point_index}_"
                                f"{segment_samples}_samples"
                            ),
                        )
                        integration_filter = (
                            qcs.IntegrationFilter(filter_waveform)
                            if len(integration_segment_sample_counts) > 1
                            else filter_waveform
                        )
                        integration_filter_cache[segment_samples] = (
                            integration_filter
                        )
                acquisition_operations.append(
                    {
                        "integration_filter": integration_filter,
                        "channels": acquisition_channels,
                        "new_layer": False,
                        "pre_delay": (
                            pre_delay_s
                            if segment_index == 0
                            else QCS_SPARAMETER_INTER_SEGMENT_DELAY_S
                        ),
                    }
                )
        else:
            # QCS requests raw trace capture by supplying a duration instead
            # of an IntegrationFilter/RFWaveform.
            integration_filter = acquisition_duration_s
            acquisition_operations.append(
                {
                    "integration_filter": integration_filter,
                    "channels": acquisition_channels,
                    "new_layer": False,
                    "pre_delay": pre_delay_s,
                }
            )
        # A raw M5200 trace may intentionally continue across later segments
        # or beyond the DC waveform. QCS keeps it in this layer and pads the
        # shorter lanes with Delay; no M5301 waveform memory is consumed.
        if not connection_config.hw_demod:
            duration_s = max(duration_s, acquisition_stop_s)

    # Schedule every RF channel independently inside its selected DC layer.
    # ``pre_delay`` is relative to that channel's previous operation in the
    # current layer, so multiple pulses on one channel use inter-pulse gaps.
    scheduled_rf_by_layer: dict[int, list[dict[str, Any]]] = {}
    for layer_index, layer_operations in rf_operations_by_layer.items():
        by_channel: dict[str, list[dict[str, Any]]] = {}
        for operation in layer_operations:
            by_channel.setdefault(operation["channel_name"], []).append(
                operation
            )
        for channel_operations in by_channel.values():
            channel_operations.sort(key=lambda item: item["start_s"])
            previous_stop_s = 0.0
            for operation in channel_operations:
                pre_delay_s = operation["start_s"] - previous_stop_s
                if pre_delay_s < -1e-15:
                    raise ValueError(
                        "QCS RF pulses overlap on virtual channel "
                        f"{operation['channel_name']!r} in DC layer "
                        f"{layer_index}"
                    )
                operation["pre_delay_s"] = max(0.0, pre_delay_s)
                previous_stop_s = (
                    operation["start_s"] + operation["duration_s"]
                )
                scheduled_rf_by_layer.setdefault(layer_index, []).append(
                    operation
                )

    # Make the selected SET layer current, add all parallel DC lanes, then
    # insert RF/acquisition before opening the next sequential DC layer.
    for layer_index, interval_group in enumerate(interval_groups):
        for output_index, (channel, intervals) in enumerate(
            zip(dc_channels, dc_interval_operations)
        ):
            program.add_waveform(
                _qcs_join_grouped_dc_intervals(intervals, interval_group),
                channel,
                new_layer=output_index == 0,
            )
        for operation in scheduled_rf_by_layer.get(layer_index, ()):
            program.add_waveform(
                operation["waveform"],
                operation["channel"],
                new_layer=False,
                pre_delay=operation["pre_delay_s"],
            )
        if acquisition_layer_index == layer_index:
            for operation in acquisition_operations:
                program.add_acquisition(**operation)

    program.n_shots(repetitions)
    return QcsCompiledPoint(
        point_index=point_index,
        program=program,
        acquisition_channels=acquisition_channels,
        duration_s=duration_s,
        acquisition_duration_s=acquisition_duration_s,
        acquisition_sample_rate_hz=acquisition_sample_rate_hz,
        acquisition_sample_count=acquisition_sample_count,
        integration_segment_sample_counts=integration_segment_sample_counts,
        inter_segment_delay_s=(
            QCS_SPARAMETER_INTER_SEGMENT_DELAY_S
            if connection_config.hw_demod
            else 0.0
        ),
        acquisition_elapsed_duration_s=acquisition_elapsed_duration_s,
    )


def compile_qcs_sequence(
    sequence: Any,
    *,
    connection_config: QcsConnectionConfig,
    mapper: Any,
    repetitions_per_sweep: int,
    fabric_mhz: float = 300.0,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
    qcs_module=None,
    progress_callback: Optional[ProgressCallback] = None,
    cancellation: Optional[QcsCancellationController] = None,
    _capacity_prevalidated: bool = False,
) -> Tuple[QcsCompiledPoint, ...]:
    """Compile every Cartesian point in C order."""
    if cancellation is not None:
        cancellation.raise_if_requested("QCS point-program compilation")
    count = int(sequence.sweep_point_count)
    _validate_qcs_software_sweep_point_count(count)
    if not _capacity_prevalidated:
        validate_qcs_m5301_waveform_capacity(
            sequence,
            fabric_mhz=fabric_mhz,
            amplitude_scale=(
                float(source_full_scale_mv)
                / (float(connection_config.dc_full_scale_v) * 1000.0)
            ),
        )
    compiled = []
    for point_index in range(count):
        if cancellation is not None:
            cancellation.raise_if_requested("QCS point-program compilation")
        compiled.append(
            compile_qcs_point(
                sequence,
                point_index,
                connection_config=connection_config,
                mapper=mapper,
                repetitions_per_sweep=repetitions_per_sweep,
                fabric_mhz=fabric_mhz,
                source_full_scale_mv=source_full_scale_mv,
                rf_pulses=rf_pulses,
                acquisition=acquisition,
                qcs_module=qcs_module,
            )
        )
        if progress_callback is not None:
            percent = 10 + int(25 * (point_index + 1) / count)
            progress_callback(
                percent,
                f"Compiled QCS point {point_index + 1:,}/{count:,}",
            )
    return tuple(compiled)


def _qcs_constant_sweep_value(values: Sequence[float]) -> Optional[float]:
    """Return a value that is constant across all synchronized points."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 1 or not np.all(np.isfinite(array)):
        return None
    if not np.allclose(array, array[0], rtol=0.0, atol=1e-12):
        return None
    return float(array[0])


def _qcs_intersect_offset_candidates(
    current: Optional[Sequence[float]],
    interval: Sequence[float],
) -> list[float]:
    """Intersect approximate endpoint values while preserving determinism."""

    unique_interval: list[float] = []
    for value in interval:
        candidate = float(value)
        if not any(
            np.isclose(candidate, existing, rtol=0.0, atol=1e-12)
            for existing in unique_interval
        ):
            unique_interval.append(candidate)
    if current is None:
        return unique_interval
    return [
        float(candidate)
        for candidate in current
        if any(
            np.isclose(candidate, other, rtol=0.0, atol=1e-12)
            for other in unique_interval
        )
    ]


def _qcs_fixed_dc_offset_plan(
    sequence: Any,
    *,
    source_full_scale_mv: float,
    dc_full_scale_v: float,
    cancellation: Optional[QcsCancellationController] = None,
    analysis_point_indices: Optional[Sequence[int]] = None,
) -> Optional[_QcsFixedDcOffsetPlan]:
    """Validate a synchronized sweep using zero physical channel offsets.

    Every *swept* changing interval must have at least one endpoint that is
    zero across the complete Cartesian sweep. A fully numeric ramp is emitted
    as one fixed ArbitraryEnvelope and does not constrain this check. Nonzero
    common baselines deliberately return ``None`` so the caller uses fixed
    numeric point programs instead of changing a ChannelMapper offset.
    """

    point_count = int(sequence.sweep_point_count)
    if analysis_point_indices is None:
        inspected_points = tuple(range(point_count))
    else:
        inspected_points = tuple(
            dict.fromkeys(int(index) for index in analysis_point_indices)
        )
        if not inspected_points or any(
            index < 0 or index >= point_count for index in inspected_points
        ):
            raise IndexError("QCS fixed-offset analysis point is out of range")
    output_names = tuple(str(name) for name in sequence.output_names)
    start_rows: list[np.ndarray] = []
    end_rows: list[np.ndarray] = []
    interval_count = None
    terminal_layout = None
    for point_index in inspected_points:
        if cancellation is not None:
            cancellation.raise_if_requested("QCS fixed-offset analysis")
        (
            times_cycles,
            waveforms,
            _boundaries,
            force_terminal_layout,
        ) = _qcs_synchronized_waveform_vertices(sequence, point_index)
        raw_times = np.asarray(times_cycles, dtype=float)
        if raw_times.ndim != 1 or raw_times.size < 2:
            raise ValueError("QCS DC vertex times must be a nonempty 1D array")
        integer_times = np.rint(raw_times).astype(np.int64)
        if not np.allclose(raw_times, integer_times, rtol=0.0, atol=1e-9):
            raise ValueError("QCS DC vertex times must be whole fabric cycles")
        if integer_times[0] != 0 or np.any(np.diff(integer_times) < 0):
            raise ValueError(
                "QCS DC vertex times must start at zero and be nondecreasing"
            )
        group_starts = np.flatnonzero(
            np.r_[True, np.diff(integer_times) != 0]
        )
        group_stops = np.r_[group_starts[1:], integer_times.size]
        point_starts = []
        point_ends = []
        append_terminal = bool(force_terminal_layout)
        for output_name in output_names:
            values = np.asarray(waveforms[output_name], dtype=float)
            if values.shape != raw_times.shape or not np.all(np.isfinite(values)):
                raise ValueError(
                    "QCS compensated waveform vertices must match the time vector"
                )
            point_starts.append(values[group_stops[:-1] - 1])
            point_ends.append(values[group_starts[1:]])
            append_terminal = bool(
                append_terminal
                or (
                    np.count_nonzero(integer_times == integer_times[-1]) > 1
                    and not np.isclose(
                        values[group_starts[-1]],
                        values[group_stops[-1] - 1],
                        rtol=0.0,
                        atol=1e-15,
                    )
                )
            )
        if terminal_layout is None:
            terminal_layout = append_terminal
        elif terminal_layout != append_terminal:
            # A varying graph cannot use synchronized compilation, but every
            # fixed numeric point remains well defined.
            return None
        starts = np.asarray(point_starts, dtype=float).T
        ends = np.asarray(point_ends, dtype=float).T
        if append_terminal:
            terminal_values = np.asarray(
                [
                    np.asarray(waveforms[name], dtype=float)[group_stops[-1] - 1]
                    for name in output_names
                ],
                dtype=float,
            )
            starts = np.vstack((starts, terminal_values))
            ends = np.vstack((ends, terminal_values))
        if interval_count is None:
            interval_count = int(starts.shape[0])
        elif interval_count != int(starts.shape[0]):
            return None
        start_rows.append(starts)
        end_rows.append(ends)

    start_table = np.asarray(start_rows, dtype=float)
    end_table = np.asarray(end_rows, dtype=float)
    source_scale_v = _positive_finite(
        source_full_scale_mv,
        "source waveform full scale",
    ) / 1000.0
    waveform_scale_v = _positive_finite(
        dc_full_scale_v,
        "QCS DC full scale",
    )
    source_offsets = []
    offset_volts = []
    optional_offset_outputs = []
    for output_index in range(len(output_names)):
        candidates: Optional[list[float]] = None
        has_changing_interval = False
        for interval_index in range(int(interval_count or 0)):
            starts = start_table[:, interval_index, output_index]
            ends = end_table[:, interval_index, output_index]
            if np.allclose(starts, ends, rtol=0.0, atol=1e-12):
                continue
            interval_candidates = []
            constant_start = _qcs_constant_sweep_value(starts)
            constant_end = _qcs_constant_sweep_value(ends)
            # A completely numeric ramp needs no swept Scalar and therefore
            # does not constrain the fixed offset selected for other ramps.
            if constant_start is not None and constant_end is not None:
                continue
            has_changing_interval = True
            if constant_start is not None:
                interval_candidates.append(constant_start)
            if constant_end is not None:
                interval_candidates.append(constant_end)
            if not interval_candidates:
                return None
            candidates = _qcs_intersect_offset_candidates(
                candidates,
                interval_candidates,
            )
            if not candidates:
                return None

        if not has_changing_interval:
            # Fixed numeric levels remain DCWaveform operations. They must not
            # be moved into the mapped physical-channel offset.
            source_offsets.append(0.0)
            offset_volts.append(0.0)
            optional_offset_outputs.append(True)
            continue

        valid_candidates = []
        for raw_candidate in candidates or ():
            if not np.isclose(
                raw_candidate,
                0.0,
                rtol=0.0,
                atol=1e-12,
            ):
                continue
            candidate = 0.0
            residual_peak = float(
                max(
                    np.max(
                        np.abs(
                            start_table[:, :, output_index] - candidate
                        )
                    ),
                    np.max(
                        np.abs(end_table[:, :, output_index] - candidate)
                    ),
                )
                * source_scale_v
                / waveform_scale_v
            )
            if residual_peak > 1.0 + 1e-12:
                continue
            valid_candidates.append(candidate)
        if not valid_candidates:
            return None
        source_offsets.append(0.0)
        offset_volts.append(0.0)
        optional_offset_outputs.append(False)

    return _QcsFixedDcOffsetPlan(
        source_offsets=tuple(source_offsets),
        offset_volts=tuple(offset_volts),
        optional_offset_outputs=tuple(optional_offset_outputs),
    )


def _qcs_sequence_requires_fixed_numeric_dc_ramp(
    sequence: Any,
    *,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    dc_full_scale_v: float = DEFAULT_QCS_FULL_SCALE_V,
) -> bool:
    """Return whether no zero-offset synchronized representation exists."""

    return _qcs_fixed_dc_offset_plan(
        sequence,
        source_full_scale_mv=source_full_scale_mv,
        dc_full_scale_v=dc_full_scale_v,
    ) is None


def _qcs_preview_dc_sweep_reasons(
    sequence: Any,
    *,
    dc_offset_plan: _QcsFixedDcOffsetPlan,
    fabric_mhz: float,
    analysis_point_indices: Optional[Sequence[int]] = None,
) -> Tuple[str, ...]:
    """Reproduce the synchronized DC target budget without importing QCS."""

    point_count = int(sequence.sweep_point_count)
    if analysis_point_indices is None:
        inspected_points = tuple(range(point_count))
    else:
        inspected_points = tuple(
            dict.fromkeys(int(index) for index in analysis_point_indices)
        )
        if not inspected_points or any(
            index < 0 or index >= point_count for index in inspected_points
        ):
            raise IndexError("QCS sweep preview point is out of range")
    output_names = tuple(str(name) for name in sequence.output_names)
    if len(dc_offset_plan.source_offsets) != len(output_names):
        raise ValueError("QCS fixed DC offset count does not match the outputs")
    fabric_hz = _positive_finite(fabric_mhz, "fabric clock") * 1e6
    duration_rows = []
    start_rows = []
    end_rows = []
    interval_count = None
    terminal_layout = None
    for point_index in inspected_points:
        times, waveforms, _boundaries, force_terminal = (
            _qcs_synchronized_waveform_vertices(sequence, point_index)
        )
        raw_times = np.asarray(times, dtype=float)
        integer_times = np.rint(raw_times).astype(np.int64)
        if (
            raw_times.ndim != 1
            or raw_times.size < 2
            or not np.allclose(raw_times, integer_times, rtol=0.0, atol=1e-9)
            or integer_times[0] != 0
            or np.any(np.diff(integer_times) < 0)
        ):
            raise ValueError("QCS DC vertices have an invalid fabric-clock grid")
        group_starts = np.flatnonzero(
            np.r_[True, np.diff(integer_times) != 0]
        )
        group_stops = np.r_[group_starts[1:], integer_times.size]
        unique_times = integer_times[group_starts]
        point_durations = np.diff(unique_times).astype(np.int64)
        point_starts = []
        point_ends = []
        append_terminal = bool(force_terminal)
        for output_index, output_name in enumerate(output_names):
            values = np.asarray(waveforms[output_name], dtype=float)
            if values.shape != integer_times.shape:
                raise ValueError(
                    "QCS compensated waveform vertices must match the time vector"
                )
            residual = values - float(dc_offset_plan.source_offsets[output_index])
            point_starts.append(residual[group_stops[:-1] - 1])
            point_ends.append(residual[group_starts[1:]])
            append_terminal = bool(
                append_terminal
                or (
                    np.count_nonzero(integer_times == integer_times[-1]) > 1
                    and not np.isclose(
                        residual[group_starts[-1]],
                        residual[group_stops[-1] - 1],
                        rtol=0.0,
                        atol=1e-15,
                    )
                )
            )
        if terminal_layout is None:
            terminal_layout = append_terminal
        elif terminal_layout != append_terminal:
            return (
                "the terminal DC operation changes across sweep points",
            )
        starts = np.asarray(point_starts, dtype=float).T
        ends = np.asarray(point_ends, dtype=float).T
        if append_terminal:
            terminal_values = np.asarray(
                [
                    np.asarray(waveforms[name], dtype=float)[group_stops[-1] - 1]
                    - float(dc_offset_plan.source_offsets[index])
                    for index, name in enumerate(output_names)
                ],
                dtype=float,
            )
            point_durations = np.r_[
                point_durations,
                QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES,
            ]
            starts = np.vstack((starts, terminal_values))
            ends = np.vstack((ends, terminal_values))
        if interval_count is None:
            interval_count = int(point_durations.size)
        elif interval_count != int(point_durations.size):
            return ("the DC waveform topology changes across sweep points",)
        duration_rows.append(point_durations)
        start_rows.append(starts)
        end_rows.append(ends)

    duration_cycles = np.asarray(duration_rows, dtype=np.int64)
    duration_table_s = duration_cycles.astype(float) / fabric_hz
    start_table = np.asarray(start_rows, dtype=float)
    end_table = np.asarray(end_rows, dtype=float)
    reasons = []
    for output_index, output_name in enumerate(output_names):
        starts = start_table[:, :, output_index]
        ends = end_table[:, :, output_index]
        hold_mask = _qcs_parameterized_dc_hold_mask(
            duration_table_s=duration_table_s,
            start_table=starts,
            end_table=ends,
            fabric_hz=fabric_hz,
        )
        array_count = 0
        for interval_index in range(int(interval_count or 0)):
            current_starts = starts[:, interval_index]
            current_ends = ends[:, interval_index]
            current_durations = duration_cycles[:, interval_index]
            zero_interval = bool(
                np.allclose(current_starts, 0.0, rtol=0.0, atol=1e-15)
                and np.allclose(current_ends, 0.0, rtol=0.0, atol=1e-15)
            )
            if not np.all(current_durations == current_durations[0]):
                array_count += 1
                if (
                    not zero_interval
                    or np.any(
                        current_durations
                        < QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES
                    )
                ):
                    reasons.append(
                        f"{output_name} interval {interval_index} duration "
                        "requires QCS software resolution"
                    )
            if hold_mask[interval_index] or zero_interval:
                continue
            pointwise_constant = np.allclose(
                current_starts,
                current_ends,
                rtol=0.0,
                atol=1e-15,
            )
            if pointwise_constant:
                if not np.allclose(
                    current_starts,
                    current_starts[0],
                    rtol=0.0,
                    atol=1e-15,
                ):
                    array_count += 1
                continue
            if (
                _qcs_constant_sweep_value(current_starts) is not None
                and _qcs_constant_sweep_value(current_ends) is not None
            ):
                # One fixed ArbitraryEnvelope represents this ramp; waveform
                # addition and hardware sweep arrays are unnecessary.
                continue
            nonzero_sides = 0
            for side in (current_starts, current_ends):
                if np.allclose(side, 0.0, rtol=0.0, atol=1e-15):
                    continue
                nonzero_sides += 1
                if not np.allclose(
                    side,
                    side[0],
                    rtol=0.0,
                    atol=1e-15,
                ):
                    array_count += 1
            if nonzero_sides > 1:
                reasons.append(
                    f"{output_name} interval {interval_index} would require "
                    "unsupported M5301 waveform addition"
                )
        stored_values = array_count * point_count
        if array_count > MAX_QCS_HARDWARE_SWEEP_ARRAYS_PER_CHANNEL:
            reasons.append(
                f"{output_name} uses {array_count} swept arrays; QCS hardware "
                f"supports at most {MAX_QCS_HARDWARE_SWEEP_ARRAYS_PER_CHANNEL}"
            )
        if stored_values >= MAX_QCS_HARDWARE_SWEEP_ARRAY_VALUES:
            reasons.append(
                f"{output_name} stores {stored_values:,} sweep values; QCS "
                f"requires fewer than {MAX_QCS_HARDWARE_SWEEP_ARRAY_VALUES:,}"
            )
    return tuple(dict.fromkeys(reasons))


def _qcs_fixed_voltage_bias_t_varies_with_sweep(sequence: Any) -> bool:
    """Return whether a sweep can change a fixed-voltage compensation tail."""

    compensation = getattr(sequence, "bias_t_compensation", None)
    if not (
        isinstance(compensation, BiasTCompensationConfig)
        and compensation.mode == "fixed_voltage"
    ):
        return False
    for axis in sequence.sweep_axes:
        # Mixed execution fixes every Python-loop axis at one authoritative
        # coordinate while preserving it in the sequence. Such a singleton
        # axis cannot vary the compensation tail inside its native inner
        # hardware sweep.
        if int(getattr(axis, "count", 1)) <= 1:
            continue
        axis_kind = str(getattr(axis, "axis_kind", ""))
        if axis_kind in {"amplitude", "ramp_duration", "hold_duration"}:
            return True
        if (
            isinstance(axis, RfDurationSweep)
            and axis.segment_length_mode == "extend_by_rf_duration"
        ):
            return True
    return False


def _qcs_sequence_slice(
    sequence: Any,
    fixed_axis_coordinates: Mapping[int, int],
) -> Any:
    """Return a structural sequence clone with selected axes fixed.

    Axes stay in their original positions with ``count=1``. FineTuneSequence
    waveform, duration, RF, Bias-T, and cross-capacitance calculations all
    depend on those positional coordinates, so deleting fixed axes would
    silently change pulse semantics. A shallow clone avoids duplicating a
    potentially large Cartesian-coordinate cache.
    """

    axes = list(sequence.sweep_axes)
    fixed = {int(key): int(value) for key, value in fixed_axis_coordinates.items()}
    for axis_index, coordinate_index in fixed.items():
        if axis_index < 0 or axis_index >= len(axes):
            raise IndexError("QCS mixed-sweep axis index is out of range")
        axis = axes[axis_index]
        points = tuple(axis.points)
        if coordinate_index < 0 or coordinate_index >= len(points):
            raise IndexError("QCS mixed-sweep coordinate index is out of range")
        if not all(hasattr(axis, name) for name in ("start", "stop", "count")):
            raise TypeError(
                "QCS mixed sweep cannot fix unsupported axis type "
                f"{type(axis).__name__}"
            )
        value = float(points[coordinate_index])
        axes[axis_index] = replace(
            axis,
            start=value,
            stop=value,
            count=1,
        )
    sliced = copy(sequence)
    sliced.sweeps = axes
    sliced.segments = list(sequence.segments)
    sliced.cross_capacitance = np.asarray(
        sequence.cross_capacitance,
        dtype=float,
    ).copy()
    sliced._sweep_coordinate_cache = None
    return sliced


def _qcs_slice_global_indices(
    original_shape: Sequence[int],
    fixed_axis_coordinates: Mapping[int, int],
) -> np.ndarray:
    """Map one sequence slice's local C-order points to global C order."""

    shape = tuple(int(value) for value in original_shape)
    if not shape or any(value < 1 for value in shape):
        raise ValueError("QCS mixed-sweep shape must contain positive axes")
    fixed = {int(key): int(value) for key, value in fixed_axis_coordinates.items()}
    for axis_index, coordinate_index in fixed.items():
        if axis_index < 0 or axis_index >= len(shape):
            raise IndexError("QCS mixed-sweep axis index is out of range")
        if coordinate_index < 0 or coordinate_index >= shape[axis_index]:
            raise IndexError("QCS mixed-sweep coordinate index is out of range")
    slice_shape = tuple(
        1 if axis_index in fixed else axis_count
        for axis_index, axis_count in enumerate(shape)
    )
    result = np.empty(int(np.prod(slice_shape, dtype=np.int64)), dtype=np.int64)
    for local_index in range(result.size):
        coordinate = list(np.unravel_index(local_index, slice_shape, order="C"))
        for axis_index, coordinate_index in fixed.items():
            coordinate[axis_index] = coordinate_index
        result[local_index] = np.ravel_multi_index(
            tuple(coordinate),
            shape,
            order="C",
        )
    return result


def _qcs_mixed_candidate_hardware_axes(sequence: Any) -> Tuple[int, ...]:
    """Return axes that can plausibly remain in native QCS hardware time."""

    candidates = []
    for axis_index, axis in enumerate(sequence.sweep_axes):
        if int(getattr(axis, "count", 1)) <= 1:
            continue
        axis_kind = str(getattr(axis, "axis_kind", ""))
        # Amplitude and RF frequency are direct hardware Scalars. A SET hold
        # duration can also be native when it renders as a legal zero Delay;
        # the real compiler below is authoritative for that conditional case.
        if axis_kind in {"amplitude", "rf_frequency", "hold_duration"}:
            candidates.append(axis_index)
    return tuple(candidates)


def _qcs_software_coordinates(
    shape: Sequence[int],
    software_axis_indices: Sequence[int],
):
    software_axes = tuple(int(value) for value in software_axis_indices)
    software_shape = tuple(int(shape[index]) for index in software_axes)
    for coordinate in product(*(range(count) for count in software_shape)):
        yield dict(zip(software_axes, coordinate))


def _qcs_point_rf_pulses(
    sequence: Any,
    rf_pulses: Sequence[QcsRfPulseConfig],
    point_index: int,
) -> Tuple[QcsRfPulseConfig, ...]:
    """Bake all fixed outer RF coordinates into one mixed-sweep block."""

    return tuple(
        _point_rf_pulse(sequence, pulse, point_index)
        for pulse in rf_pulses
    )


def qcs_sweep_execution_preview(
    sequence: Any,
    *,
    hardware_demodulation: bool,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    dc_full_scale_v: float = DEFAULT_QCS_FULL_SCALE_V,
    fabric_mhz: float = 300.0,
    init_time_s: float = 0.0,
) -> QcsSweepExecutionPreview:
    """Predict the AWG-tuning QCS sweep mode for the live GUI indicator."""

    point_count = int(sequence.sweep_point_count)
    if point_count <= 1:
        return QcsSweepExecutionPreview(
            mode="none",
            reasons=(
                "No voltage or RF sweep is configured. Fixed levels use "
                "DCWaveform amplitudes; physical channel offsets are disabled.",
            ),
            dc_channel_offsets_v=(0.0,) * len(sequence.output_names),
        )
    axis_kinds = tuple(
        str(getattr(axis, "axis_kind", "amplitude"))
        for axis in sequence.sweep_axes
    )
    if "rf_power" in axis_kinds:
        return QcsSweepExecutionPreview(
            mode="invalid",
            reasons=(
                "Calibrated RF connector-power sweeps are not supported by "
                "the QCS backend.",
            ),
        )

    candidate_hardware_axes = _qcs_mixed_candidate_hardware_axes(sequence)
    candidate_hardware_points = int(
        np.prod(
            [sequence.sweep_shape[index] for index in candidate_hardware_axes],
            dtype=np.int64,
        )
    ) if candidate_hardware_axes else 1
    candidate_software_points = (
        point_count // candidate_hardware_points
        if candidate_hardware_points > 1
        else point_count
    )

    def early_software_or_hybrid(reason: str) -> QcsSweepExecutionPreview:
        if (
            bool(hardware_demodulation)
            and len(tuple(sequence.sweep_axes)) >= 2
            and candidate_hardware_points > 1
            and candidate_software_points <= MAX_QCS_SOFTWARE_SWEEP_POINTS
        ):
            return QcsSweepExecutionPreview(
                mode="hybrid",
                reasons=(
                    reason,
                    "Compatible axes are planned as a native QCS hardware "
                    "sweep inside a Python software loop; mapped compilation "
                    "confirms the exact partition at Run.",
                ),
                exact=False,
            )
        mode = (
            "invalid"
            if point_count > MAX_QCS_SOFTWARE_SWEEP_POINTS
            else "software"
        )
        limit_reason = (
            f" The {point_count:,}-point grid exceeds the "
            f"{MAX_QCS_SOFTWARE_SWEEP_POINTS:,}-point software-sweep limit."
            if mode == "invalid"
            else ""
        )
        return QcsSweepExecutionPreview(
            mode=mode,
            reasons=(reason + limit_reason,),
        )

    if _qcs_fixed_voltage_bias_t_varies_with_sweep(sequence):
        return early_software_or_hybrid(
            (
                "Fixed-voltage Bias-T compensation changes its duration "
                "with pulse area, so QCS uses fixed numeric point programs "
                "and a direct ramp-to-Hold compensation tail."
            )
        )
    inspected = (
        tuple(range(point_count))
        if point_count <= 256
        else qcs_m5301_capacity_preview_point_indices(sequence)
    )
    offset_plan = _qcs_fixed_dc_offset_plan(
        sequence,
        source_full_scale_mv=source_full_scale_mv,
        dc_full_scale_v=dc_full_scale_v,
        analysis_point_indices=inspected,
    )
    if offset_plan is None:
        return early_software_or_hybrid(
            (
                "Voltage ramps do not share a zero endpoint. Physical channel "
                "offsets are disabled, so QCS must execute fixed numeric "
                "point programs."
            )
        )

    reasons = list(
        _qcs_preview_dc_sweep_reasons(
            sequence,
            dc_offset_plan=offset_plan,
            fabric_mhz=fabric_mhz,
            analysis_point_indices=inspected,
        )
    )
    if not bool(hardware_demodulation):
        reasons.append("Raw trace acquisition requires QCS software resolution.")
    if "rf_duration" in axis_kinds:
        reasons.append("RF waveform duration requires QCS software resolution.")
    reasons = list(dict.fromkeys(reasons))
    if reasons:
        if (
            bool(hardware_demodulation)
            and len(tuple(sequence.sweep_axes)) >= 2
            and candidate_hardware_points > 1
            and candidate_software_points <= MAX_QCS_SOFTWARE_SWEEP_POINTS
        ):
            return QcsSweepExecutionPreview(
                mode="hybrid",
                reasons=(
                    *tuple(reasons),
                    "Compatible axes will run as a native QCS hardware "
                    "sweep inside a stoppable Python software loop; the "
                    "mapped compiler confirms the exact partition at Run.",
                ),
                exact=False,
                dc_channel_offsets_v=tuple(offset_plan.offset_volts),
            )
        if point_count > MAX_QCS_SOFTWARE_SWEEP_POINTS:
            reasons.append(
                f"The required software sweep exceeds the "
                f"{MAX_QCS_SOFTWARE_SWEEP_POINTS:,}-point limit."
            )
        return QcsSweepExecutionPreview(
            mode=(
                "invalid"
                if point_count > MAX_QCS_SOFTWARE_SWEEP_POINTS
                else "software"
            ),
            reasons=tuple(reasons),
            dc_channel_offsets_v=tuple(offset_plan.offset_volts),
        )
    del init_time_s
    confirmation = (
        "The waveform is hardware-sweepable using DCWaveform amplitudes; "
        "physical channel offsets are disabled."
    )
    return QcsSweepExecutionPreview(
        mode="hardware",
        reasons=(confirmation,),
        exact=False,
        dc_channel_offsets_v=tuple(offset_plan.offset_volts),
    )


def _qcs_aligned_generated_m5301_cycles(
    cycles: int,
    *,
    minimum: int = 0,
) -> int:
    """Round a program-generated duration up to the M5301 waveform grid."""

    value = max(int(cycles), int(minimum))
    granularity = QCS_M5301_WAVEFORM_GRANULARITY_FABRIC_CYCLES
    remainder = value % granularity
    return value if remainder == 0 else value + granularity - remainder


def _qcs_synchronized_waveform_vertices(
    sequence: Any,
    point_index: int,
    *,
    dc_offset_init_compensation_source: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, Mapping[str, np.ndarray], tuple, bool]:
    """Return QCS-safe point vertices, including an aligned Bias-T tail.

    Bias-T guard and compensation durations are generated by the program, not
    entered as visible waveform rows.  QICK may express them in any whole
    fabric-cycle duration, while every nonzero M5301 ``DCWaveform`` requires
    an even duration of at least four cycles.  Align only these generated QCS
    intervals and reduce their voltage by the reciprocal duration ratio so
    the existing programmed compensation area is unchanged.

    Fixed-voltage tails use a minimum legal ramp followed by ``Hold``.  This
    both establishes the requested level with the hardware-verified form and
    avoids consuming the finite M5301 waveform buffer for a long compensation
    plateau.

    Fixed-time compensation also retains zero-amplitude padding at zero-area
    points.  Its operation graph must remain identical across a synchronized
    sweep.  The final Boolean requests the matching terminal reset interval.
    """

    compensation = getattr(sequence, "bias_t_compensation", None)
    if not isinstance(compensation, BiasTCompensationConfig):
        if dc_offset_init_compensation_source is not None:
            raise ValueError(
                "QCS fixed-offset initialization compensation requires "
                "fixed-time DC Bias-T compensation"
            )
        times, waveforms, boundaries = (
            sequence.compensated_waveform_vertices(point_index)
        )
        return (
            np.asarray(times, dtype=float),
            waveforms,
            tuple(boundaries),
            False,
        )

    fixed_time = compensation.mode == "fixed_time"
    if dc_offset_init_compensation_source is not None and not fixed_time:
        raise ValueError(
            "QCS fixed-offset initialization compensation requires "
            "fixed-time DC Bias-T compensation"
        )

    times, waveforms, boundaries = sequence.waveform_vertices(
        point_index,
        space="physical",
    )
    time_values = list(np.asarray(times, dtype=float))
    output_names = tuple(str(name) for name in sequence.output_names)
    value_matrix = np.vstack(
        [np.asarray(waveforms[name], dtype=float) for name in output_names]
    )
    columns = [
        value_matrix[:, index].copy()
        for index in range(value_matrix.shape[1])
    ]
    current = columns[-1].copy()
    boundary_values = list(boundaries)
    time_now = float(time_values[-1])

    def append_vertex(time_value: float, *, force: bool = False) -> None:
        if (
            not force
            and time_values
            and float(time_value) == time_values[-1]
            and np.array_equal(current, columns[-1])
        ):
            return
        time_values.append(float(time_value))
        columns.append(current.copy())

    # Preserve the existing compensation semantics: return all outputs to
    # zero, wait the common guard/driver lead, then start every active
    # opposite-polarity pulse simultaneously.
    current[:] = 0.0
    append_vertex(time_now)
    previews = tuple(sequence.bias_t_compensation_preview(point_index))
    if fixed_time:
        raw_fixed_duration = int(compensation.fixed_duration_cycles)
        active_previews = tuple(
            (preview, raw_fixed_duration)
            for preview in previews
        )
    else:
        active_previews = tuple(
            (preview, int(preview.duration_cycles))
            for preview in previews
            if int(preview.duration_cycles) > 0
        )

    raw_guard_cycles = int(compensation.inter_output_gap_cycles) + (
        BIAS_T_INSTRUCTION_LEAD_PER_OUTPUT * len(output_names)
    )
    guard_cycles = _qcs_aligned_generated_m5301_cycles(raw_guard_cycles)
    if active_previews and guard_cycles:
        time_now += guard_cycles
        append_vertex(time_now)

    if dc_offset_init_compensation_source is None:
        init_adjustments = np.zeros(len(output_names), dtype=float)
    else:
        init_adjustments = np.asarray(
            dc_offset_init_compensation_source,
            dtype=float,
        )
        if init_adjustments.shape != (len(output_names),) or not np.all(
            np.isfinite(init_adjustments)
        ):
            raise ValueError(
                "QCS fixed-offset initialization compensation must contain "
                "one finite value per DC output"
            )
    start_time = time_now
    aligned_previews = []
    compensation_seed_cycles = (
        0 if fixed_time else QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES
    )
    for preview, raw_duration in active_previews:
        output_index = int(preview.output_index)
        if fixed_time:
            aligned_duration = _qcs_aligned_generated_m5301_cycles(
                raw_duration,
                minimum=QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES,
            )
            effective_area_cycles = float(aligned_duration)
        else:
            # A standalone constant seed followed by Hold returned to the
            # channel baseline on the connected QCS 2.5.5 system.  Establish
            # the compensation voltage with the hardware-validated direct
            # ramp-to-Hold form instead.  The ramp contributes half of its
            # duration to voltage-time area, so lengthen the complete tail
            # before scaling its target.
            aligned_duration = _qcs_aligned_generated_m5301_cycles(
                raw_duration + compensation_seed_cycles // 2,
                minimum=compensation_seed_cycles,
            )
            plateau_cycles = aligned_duration - compensation_seed_cycles
            if 0 < plateau_cycles < QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES:
                aligned_duration = (
                    compensation_seed_cycles
                    + QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES
                )
            effective_area_cycles = (
                float(aligned_duration)
                - float(compensation_seed_cycles) / 2.0
            )
        target = (
            float(preview.target_amplitude)
            * float(raw_duration)
            / effective_area_cycles
            + float(init_adjustments[output_index])
        )
        if abs(target) > 1.0 + 1e-12:
            raise QcsUnsupportedFeatureError(
                "QCS Bias-T compensation, including any M5301 offset during "
                "the inter-iteration delay, exceeds "
                f"the source full scale on {preview.output_name!r} "
                f"({target:+.6g}); reduce the inter-iteration delay or "
                "increase the compensation duration"
            )
        aligned_previews.append((preview, aligned_duration, target))

    if fixed_time:
        for preview, _duration, target in aligned_previews:
            current[int(preview.output_index)] = target
        if aligned_previews:
            append_vertex(start_time, force=True)
    elif aligned_previews:
        # Start at physical zero and ramp every active output to its target in
        # one minimum legal waveform.  The following constant interval is a
        # direct continuation and can therefore be lowered to QCS Hold.
        time_now = start_time + compensation_seed_cycles
        for preview, _duration, target in aligned_previews:
            current[int(preview.output_index)] = target
        append_vertex(time_now)

    for duration_cycles in sorted(
        {duration for _preview, duration, _target in aligned_previews}
    ):
        time_now = start_time + duration_cycles
        append_vertex(time_now)
        for preview, current_duration, _target in aligned_previews:
            if current_duration == duration_cycles:
                current[int(preview.output_index)] = 0.0
        append_vertex(time_now, force=True)
    for preview, duration_cycles, _target in aligned_previews:
        boundary_values.append(
            (
                f"bias_t_comp_{preview.output_name}",
                start_time,
                start_time + duration_cycles,
            )
        )

    matrix = np.asarray(columns, dtype=float).T
    return (
        np.asarray(time_values, dtype=float),
        {
            name: matrix[index]
            for index, name in enumerate(output_names)
        },
        tuple(boundary_values),
        fixed_time,
    )


def compile_qcs_synchronized_sweep(
    sequence: Any,
    *,
    connection_config: QcsConnectionConfig,
    mapper: Any,
    repetitions_per_sweep: int,
    fabric_mhz: float = 300.0,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
    qcs_module=None,
    cancellation: Optional[QcsCancellationController] = None,
    _capacity_prevalidated: bool = False,
    _dc_offset_plan: Optional[_QcsFixedDcOffsetPlan] = None,
) -> QcsCompiledHardwareSweep:
    """Compile every AWG-tuning point into one synchronized QCS Program.

    All sweepable physical values are calculated on the host in the same
    flattened C order as :class:`FineTuneSequence`.  A single paired QCS sweep
    then updates those direct Scalars together.  When every varying parameter
    is supported by the HCL hardware-sweep sandbox, ``sweep`` is inserted
    before ``n_shots``.  Otherwise their order is reversed so QCS resolves the
    unsupported settings in software while retaining one Program and one
    Executor call.
    """

    qcs = _import_qcs() if qcs_module is None else qcs_module
    if cancellation is not None:
        cancellation.raise_if_requested("synchronized QCS program compilation")
    if acquisition is None:
        raise ValueError("QCS execution requires an acquisition configuration")
    if isinstance(repetitions_per_sweep, bool):
        raise TypeError("repetitions_per_sweep must be an integer")
    repetitions = int(repetitions_per_sweep)
    if repetitions < 1 or repetitions != repetitions_per_sweep:
        raise ValueError("repetitions_per_sweep must be a positive integer")
    validate_qcs_capabilities(
        connection_config=connection_config,
        sequence=sequence,
        rf_pulses=rf_pulses,
        acquisition=acquisition,
    )
    relative_amplitude_scale = _positive_finite(
        source_full_scale_mv,
        "source waveform full scale",
    ) / (
        _positive_finite(connection_config.dc_full_scale_v, "QCS DC full scale")
        * 1000.0
    )
    dc_offset_plan = _dc_offset_plan
    if dc_offset_plan is None:
        dc_offset_plan = _qcs_fixed_dc_offset_plan(
            sequence,
            source_full_scale_mv=source_full_scale_mv,
            dc_full_scale_v=connection_config.dc_full_scale_v,
            cancellation=cancellation,
        )
    if dc_offset_plan is None:
        raise QcsUnsupportedFeatureError(
            "QCS synchronized compilation requires a zero-baseline M5301 "
            "amplitude sweep; physical channel offsets are disabled. Use "
            "execute_qcs_sequence for hardware-safe fixed numeric point "
            "programs."
        )
    # The exact fixed operation graph is validated below after all Cartesian
    # duration/amplitude tables have been assembled. Avoid another full sweep
    # expansion here; it materially increases preparation time for 101 x 101
    # grids and cannot be more accurate than the graph-level check.

    fabric_hz = _positive_finite(fabric_mhz, "fabric clock") * 1e6
    if not np.isclose(
        fabric_hz,
        QCS_FABRIC_CLOCK_HZ,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "QCS M5000-series execution requires the fixed 300 MHz "
            "synchronization fabric clock"
        )
    point_count = int(sequence.sweep_point_count)
    sweep_shape = tuple(int(value) for value in sequence.sweep_shape)
    if point_count < 1 or int(np.prod(sweep_shape)) != point_count:
        raise ValueError("QCS sequence has an invalid Cartesian sweep shape")

    output_names = tuple(str(name) for name in sequence.output_names)
    if len(output_names) != len(connection_config.dc_channel_names):
        raise ValueError(
            "QCS DC channel count must match the waveform output count"
        )
    if len(dc_offset_plan.source_offsets) != len(output_names):
        raise ValueError(
            "QCS fixed DC offset count must match the waveform output count"
        )
    if not _mapper_supports_qcs_dc_offsets(
        mapper,
        channel_names=connection_config.dc_channel_names,
        offset_volts=dc_offset_plan.offset_volts,
    ):
        # A globally constant lane uses offset only as a waveform-memory
        # optimization.  Older/injected mappers without physical settings can
        # render that lane exactly as before.  Offsets selected as the common
        # endpoint of a varying ramp remain mandatory for synchronized sweep
        # lowering and are deliberately not removed here.
        fallback_offset_plan = _qcs_without_optional_dc_offsets(
            dc_offset_plan
        )
        if _mapper_supports_qcs_dc_offsets(
            mapper,
            channel_names=connection_config.dc_channel_names,
            offset_volts=fallback_offset_plan.offset_volts,
        ):
            dc_offset_plan = fallback_offset_plan
    compensation_config = getattr(sequence, "bias_t_compensation", None)
    offset_init_compensation_source = None
    if (
        isinstance(compensation_config, BiasTCompensationConfig)
        and compensation_config.mode == "fixed_time"
    ):
        offset_init_compensation_source = np.zeros(
            len(output_names),
            dtype=float,
        )
    if (
        offset_init_compensation_source is not None
        and any(
            not np.isclose(value, 0.0, rtol=0.0, atol=1e-15)
            for value in dc_offset_plan.offset_volts
        )
    ):
        total_iterations = point_count * repetitions
        init_gap_fraction = (
            0.0
            if total_iterations <= 1
            else (total_iterations - 1) / total_iterations
        )
        compensation_duration_s = (
            _qcs_aligned_generated_m5301_cycles(
                int(compensation_config.fixed_duration_cycles),
                minimum=QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES,
            )
            / fabric_hz
        )
        source_scale_v = (
            _positive_finite(
                source_full_scale_mv,
                "source waveform full scale",
            )
            / 1000.0
        )
        offset_init_compensation_source = -(
            np.asarray(dc_offset_plan.offset_volts, dtype=float)
            * float(connection_config.init_time_s)
            * init_gap_fraction
            / compensation_duration_s
            / source_scale_v
        )
    dc_channels = []
    for channel_name in connection_config.dc_channel_names:
        channel = _resolve_mapper_channel(mapper, channel_name)
        _validate_mapped_hardware_role(
            mapper,
            channel,
            name=channel_name,
            role="DC",
            expected_instruments=("M5301AWG",),
        )
        dc_channels.append(channel)
    if not _mapper_supports_qcs_dc_offsets(
        mapper,
        channel_names=connection_config.dc_channel_names,
        offset_volts=dc_offset_plan.offset_volts,
    ):
        raise QcsUnsupportedFeatureError(
            "The mapped M5301 physical channel does not expose the fixed "
            "offset required for this synchronized amplitude sweep"
        )

    if connection_config.acquisition_channel_name is None:
        raise ValueError("a QCS acquisition virtual-channel name is required")
    acquisition_channels = _resolve_mapper_channel(
        mapper,
        connection_config.acquisition_channel_name,
    )
    _validate_mapped_hardware_role(
        mapper,
        acquisition_channels,
        name=connection_config.acquisition_channel_name,
        role="acquisition",
        expected_instruments=("M5200Digitizer",),
    )
    (
        acquisition_duration_s,
        acquisition_sample_rate_hz,
    ) = _resolved_acquisition_timing(
        mapper,
        acquisition_channels,
        acquisition,
        hardware_demodulation=connection_config.hw_demod,
    )
    acquisition_sample_count = int(
        round(acquisition_duration_s * acquisition_sample_rate_hz)
    )
    if connection_config.hw_demod:
        integration_segment_sample_counts = (
            _qcs_iq_integration_segment_sample_counts(
                acquisition_sample_count
            )
        )
        acquisition_sample_count = int(
            sum(integration_segment_sample_counts)
        )
        acquisition_duration_s = (
            acquisition_sample_count / acquisition_sample_rate_hz
        )
        acquisition_elapsed_duration_s = (
            acquisition_duration_s
            + max(0, len(integration_segment_sample_counts) - 1)
            * QCS_SPARAMETER_INTER_SEGMENT_DELAY_S
        )
        if (
            len(integration_segment_sample_counts) > 1
            and acquisition.integration_filter is not None
        ):
            raise QcsUnsupportedFeatureError(
                "segmented QCS I/Q integration requires the built-in flat "
                "filter; a custom filter cannot be divided safely"
            )
    else:
        integration_segment_sample_counts = (acquisition_sample_count,)
        acquisition_elapsed_duration_s = acquisition_duration_s
    normalized_offset_values = np.asarray(
        dc_offset_plan.offset_volts,
        dtype=float,
    ) / float(connection_config.dc_full_scale_v)

    duration_rows = []
    start_rows = []
    end_rows = []
    boundary_rows = []
    total_duration_values = []
    interval_count = None
    terminal_layout = None
    for point_index in range(point_count):
        if cancellation is not None:
            cancellation.raise_if_requested(
                "synchronized QCS program compilation"
            )
        (
            times_cycles,
            waveforms,
            boundaries,
            force_terminal_layout,
        ) = _qcs_synchronized_waveform_vertices(
            sequence,
            point_index,
            dc_offset_init_compensation_source=(
                offset_init_compensation_source
            ),
        )
        raw_times = np.asarray(times_cycles, dtype=float)
        if raw_times.ndim != 1 or raw_times.size < 2:
            raise ValueError("QCS DC vertex times must be a nonempty 1D array")
        integer_times = np.rint(raw_times).astype(np.int64)
        if not np.allclose(raw_times, integer_times, rtol=0.0, atol=1e-9):
            raise ValueError("QCS DC vertex times must be whole fabric cycles")
        if integer_times[0] != 0 or np.any(np.diff(integer_times) < 0):
            raise ValueError(
                "QCS DC vertex times must start at zero and be nondecreasing"
            )
        group_starts = np.flatnonzero(
            np.r_[True, np.diff(integer_times) != 0]
        )
        group_stops = np.r_[group_starts[1:], integer_times.size]
        unique_times = integer_times[group_starts]
        if unique_times.size < 2 or unique_times[-1] <= 0:
            raise ValueError("QCS sequence must have a positive duration")
        point_durations = np.diff(unique_times).astype(np.int64)

        point_waveforms = np.empty(
            (len(output_names), integer_times.size),
            dtype=float,
        )
        for output_index, output_name in enumerate(output_names):
            values = np.asarray(waveforms[output_name], dtype=float)
            if values.shape != integer_times.shape or not np.all(np.isfinite(values)):
                raise ValueError(
                    "QCS compensated waveform vertices must match the time vector"
                )
            physical_values = values * relative_amplitude_scale
            physical_peak = float(np.max(np.abs(physical_values)))
            if physical_peak > 1.0 + 1e-12:
                raise ValueError(
                    f"QCS DC output {output_name!r} at sweep point "
                    f"{point_index + 1} reaches "
                    f"{physical_peak * connection_config.dc_full_scale_v:.6g} "
                    "V, exceeding the configured +/-"
                    f"{connection_config.dc_full_scale_v:.6g} V full scale"
                )
            point_waveforms[output_index] = (
                physical_values - normalized_offset_values[output_index]
            )
        point_starts = point_waveforms[:, group_stops[:-1] - 1].T
        point_ends = point_waveforms[:, group_starts[1:]].T
        append_terminal = bool(
            force_terminal_layout
            or (
                np.count_nonzero(integer_times == integer_times[-1]) > 1
                and any(
                    not np.isclose(
                        point_waveforms[output_index, group_starts[-1]],
                        point_waveforms[output_index, group_stops[-1] - 1],
                        rtol=0.0,
                        atol=1e-15,
                    )
                    for output_index in range(len(output_names))
                )
            )
        )
        if terminal_layout is None:
            terminal_layout = append_terminal
        elif terminal_layout != append_terminal:
            raise QcsUnsupportedFeatureError(
                "QCS synchronized sweep requires the terminal SET layout to "
                "remain constant across all points"
            )
        if append_terminal:
            terminal_values = point_waveforms[:, group_stops[-1] - 1]
            point_durations = np.r_[
                point_durations,
                QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES,
            ]
            point_starts = np.vstack((point_starts, terminal_values))
            point_ends = np.vstack((point_ends, terminal_values))
        if interval_count is None:
            interval_count = int(point_durations.size)
        elif interval_count != int(point_durations.size):
            raise QcsUnsupportedFeatureError(
                "QCS synchronized sweep requires one fixed waveform topology; "
                "a swept duration changed the number of DC intervals"
            )
        if np.any(np.abs(point_starts) > 1.0 + 1e-12) or np.any(
            np.abs(point_ends) > 1.0 + 1e-12
        ):
            flat_index = int(
                np.argmax(
                    np.maximum(np.abs(point_starts), np.abs(point_ends))
                )
            )
            output_index = flat_index % len(output_names)
            peak = float(
                np.max(
                    np.maximum(
                        np.abs(point_starts[:, output_index]),
                        np.abs(point_ends[:, output_index]),
                    )
                )
            )
            raise ValueError(
                f"QCS fixed-offset residual for output "
                f"{output_names[output_index]!r} at sweep point "
                f"{point_index + 1} reaches "
                f"{peak * connection_config.dc_full_scale_v:.6g} V, "
                "exceeding the M5301 DCWaveform amplitude range"
            )
        duration_rows.append(point_durations)
        start_rows.append(point_starts)
        end_rows.append(point_ends)
        boundary_rows.append(
            _segment_boundaries_seconds(boundaries, fabric_mhz=fabric_mhz)
        )
        total_duration_values.append(float(np.sum(point_durations)) / fabric_hz)

    duration_table = np.asarray(duration_rows, dtype=float) / fabric_hz
    start_table = np.asarray(start_rows, dtype=float)
    end_table = np.asarray(end_rows, dtype=float)
    if duration_table.shape != (point_count, interval_count):
        raise ValueError("QCS synchronized duration table has an invalid shape")
    expected_amplitude_shape = (
        point_count,
        interval_count,
        len(output_names),
    )
    if start_table.shape != expected_amplitude_shape or end_table.shape != (
        expected_amplitude_shape
    ):
        raise ValueError("QCS synchronized amplitude tables have invalid shapes")

    # A synchronized Program has one fixed operation graph.  If an interval is
    # active at any point, HCL reserves its maximum rendered duration even at
    # points where that interval's swept amplitude happens to be zero.  This is
    # stricter than the legacy point-by-point capacity calculation when active
    # intervals are mutually exclusive across sweep points.
    duration_cycles_table = np.rint(duration_table * fabric_hz).astype(np.int64)
    for output_index, output_name in enumerate(output_names):
        hold_mask = _qcs_parameterized_dc_hold_mask(
            duration_table_s=duration_table,
            start_table=start_table[:, :, output_index],
            end_table=end_table[:, :, output_index],
            fabric_hz=fabric_hz,
        )
        fixed_graph_cycles = 0
        for current_interval in range(interval_count):
            active = not (
                np.allclose(
                    start_table[:, current_interval, output_index],
                    0.0,
                    rtol=0.0,
                    atol=1e-15,
                )
                and np.allclose(
                    end_table[:, current_interval, output_index],
                    0.0,
                    rtol=0.0,
                    atol=1e-15,
                )
            )
            if active and not hold_mask[current_interval]:
                interval_starts = start_table[
                    :, current_interval, output_index
                ]
                interval_ends = end_table[:, current_interval, output_index]
                interval_durations = duration_cycles_table[:, current_interval]
                fixed_constant = bool(
                    np.allclose(
                        interval_starts,
                        interval_ends,
                        rtol=0.0,
                        atol=1e-15,
                    )
                    and np.all(interval_durations == interval_durations[0])
                )
                if fixed_constant:
                    fixed_graph_cycles += min(
                        int(interval_durations[0]),
                        QCS_M5301_HOLD_SEED_FABRIC_CYCLES,
                    )
                else:
                    fixed_graph_cycles += int(np.max(interval_durations))
        if fixed_graph_cycles > QCS_M5301_MAX_RENDERED_FABRIC_CYCLES:
            _raise_qcs_m5301_capacity_error(
                output_name=output_name,
                rendered_fabric_cycles=fixed_graph_cycles,
            )

    program = qcs.Program(name="PulseGenerator synchronized AWG tuning sweep")
    targets: list[_QcsSweepTarget] = []
    dc_interval_operations = []
    for output_index, (output_name, channel_name, channel) in enumerate(
        zip(output_names, connection_config.dc_channel_names, dc_channels)
    ):
        operations = _qcs_parameterized_dc_operations(
            qcs,
            duration_table_s=duration_table,
            start_table=start_table[:, :, output_index],
            end_table=end_table[:, :, output_index],
            targets=targets,
            channel_name=channel_name,
            output_index=output_index,
            fabric_hz=fabric_hz,
            grouped=True,
        )
        dc_interval_operations.append(operations)
    if any(
        len(operations) != interval_count
        for operations in dc_interval_operations
    ):
        raise ValueError("QCS DC outputs produced different interval layers")
    (
        interval_groups,
        segment_layer_indices,
        segment_layer_offsets,
    ) = _qcs_segment_grouped_layer_layout(
        sequence,
        duration_table_s=duration_table,
        boundary_rows=boundary_rows,
    )
    layer_count = len(interval_groups)

    prepared_rf_pulses = []
    for pulse_index, pulse in enumerate(rf_pulses):
        layer_index = segment_layer_indices[pulse.at_segment]
        channel_name = connection_config.rf_channel_names[pulse.gen_ch]
        channel = _resolve_mapper_channel(mapper, channel_name)
        _validate_mapped_hardware_role(
            mapper,
            channel,
            name=channel_name,
            role="RF",
            expected_instruments=("M5300AWG", "M5301AWG"),
        )
        point_pulses = [
            _point_rf_pulse(sequence, pulse, point_index)
            for point_index in range(point_count)
        ]
        start_cycle_values = []
        duration_cycle_values = []
        frequency_values = []
        for point_index, current in enumerate(point_pulses):
            if current.at_segment not in boundary_rows[point_index]:
                raise KeyError(
                    f"no timing boundary for RF segment {current.at_segment!r}"
                )
            segment_start_s, segment_stop_s = boundary_rows[point_index][
                current.at_segment
            ]
            delay_s = _fabric_aligned_seconds(
                current.delay_s,
                fabric_hz=fabric_hz,
                label=f"QCS RF gen_ch {current.gen_ch} delay",
            )
            duration_s = _fabric_aligned_seconds(
                current.duration_s,
                fabric_hz=fabric_hz,
                label=f"QCS RF gen_ch {current.gen_ch} duration",
                positive=True,
            )
            segment_start_cycles = int(round(segment_start_s * fabric_hz))
            segment_stop_cycles = int(round(segment_stop_s * fabric_hz))
            delay_cycles = int(round(delay_s * fabric_hz))
            duration_cycles = int(round(duration_s * fabric_hz))
            absolute_start_cycles = segment_start_cycles + delay_cycles
            if (
                current.require_within_segment
                and absolute_start_cycles + duration_cycles
                > segment_stop_cycles
            ):
                raise ValueError(
                    f"QCS RF pulse on gen_ch {current.gen_ch} exceeds segment "
                    f"{current.at_segment!r} at sweep point {point_index + 1}"
                )
            if (
                absolute_start_cycles + duration_cycles
                > segment_stop_cycles
                and layer_index < layer_count - 1
            ):
                raise ValueError(
                    f"QCS RF pulse on gen_ch {current.gen_ch} cannot cross "
                    f"segment {current.at_segment!r} while later DC segment "
                    f"layers remain at sweep point {point_index + 1}"
                )
            layer_offset_cycles = int(
                round(
                    float(
                        segment_layer_offsets[current.at_segment][point_index]
                    )
                    * fabric_hz
                )
            )
            start_cycle_values.append(layer_offset_cycles + delay_cycles)
            duration_cycle_values.append(duration_cycles)
            frequency_values.append(current.frequency_hz)
        prepared_rf_pulses.append(
            {
                "pulse_index": int(pulse_index),
                "pulse": pulse,
                "layer_index": layer_index,
                "channel_name": channel_name,
                "channel": channel,
                "start_cycles": np.asarray(
                    start_cycle_values,
                    dtype=np.int64,
                ),
                "duration_cycles": np.asarray(
                    duration_cycle_values,
                    dtype=np.int64,
                ),
                "frequency_values": np.asarray(
                    frequency_values,
                    dtype=float,
                ),
            }
        )

    # ``pre_delay`` is relative to the current cursor of its virtual channel
    # inside the selected DC layer. Establish a fixed order per layer/channel
    # and turn every segment-local start into the inter-pulse gap.
    rf_pulses_by_channel = {}
    for prepared in prepared_rf_pulses:
        rf_pulses_by_channel.setdefault(
            (prepared["layer_index"], prepared["channel_name"]),
            [],
        ).append(prepared)
    scheduled_rf_pulses = []
    for (_layer_index, channel_name), channel_pulses in (
        rf_pulses_by_channel.items()
    ):
        channel_pulses.sort(
            key=lambda item: (
                int(item["start_cycles"][0]),
                int(item["pulse_index"]),
            )
        )
        previous = None
        for prepared in channel_pulses:
            start_cycles = prepared["start_cycles"]
            if previous is None:
                pre_delay_cycles = start_cycles.copy()
            else:
                previous_starts = previous["start_cycles"]
                previous_ends = (
                    previous_starts + previous["duration_cycles"]
                )
                reordered = np.flatnonzero(start_cycles < previous_starts)
                if reordered.size:
                    point_index = int(reordered[0])
                    raise QcsUnsupportedFeatureError(
                        f"QCS RF pulse order on {channel_name!r} changes at "
                        f"sweep point {point_index + 1}; use noncrossing RF "
                        "pulse start times"
                    )
                overlapping = np.flatnonzero(start_cycles < previous_ends)
                if overlapping.size:
                    point_index = int(overlapping[0])
                    raise ValueError(
                        f"QCS RF pulses overlap on {channel_name!r} at "
                        f"sweep point {point_index + 1}"
                    )
                pre_delay_cycles = start_cycles - previous_ends
            prepared["pre_delay_cycles"] = pre_delay_cycles
            scheduled_rf_pulses.append(prepared)
            previous = prepared

    rf_program_operations_by_layer: dict[int, list[dict[str, Any]]] = {}
    for prepared in scheduled_rf_pulses:
        pulse_index = prepared["pulse_index"]
        pulse = prepared["pulse"]
        channel_name = prepared["channel_name"]
        channel = prepared["channel"]
        duration_values = prepared["duration_cycles"].astype(float) / fabric_hz
        frequency_values = prepared["frequency_values"]
        pre_delay_cycles = prepared["pre_delay_cycles"]
        pre_delay_values = pre_delay_cycles.astype(float) / fabric_hz
        duration = _qcs_sweep_parameter(
            qcs,
            name=f"awg_rf_{pulse_index}_duration",
            values=duration_values,
            targets=targets,
            hardware_supported=False,
            channel_name=channel_name,
            description=f"RF gen_ch {pulse.gen_ch} duration",
        )
        frequency = _qcs_sweep_parameter(
            qcs,
            name=f"awg_rf_{pulse_index}_frequency",
            values=frequency_values,
            targets=targets,
            hardware_supported=True,
            channel_name=channel_name,
            description=f"RF gen_ch {pulse.gen_ch} frequency",
        )
        pre_delay = _qcs_sweep_parameter(
            qcs,
            name=f"awg_rf_{pulse_index}_pre_delay",
            values=pre_delay_values,
            targets=targets,
            hardware_supported=bool(
                np.all(
                    pre_delay_cycles
                    >= QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES
                )
            ),
            channel_name=channel_name,
            description=f"RF gen_ch {pulse.gen_ch} pre-delay",
        )
        rf_program_operations_by_layer.setdefault(
            prepared["layer_index"], []
        ).append(
            {
                "waveform": qcs.RFWaveform(
                    duration=duration,
                    envelope=_qcs_envelope(qcs, pulse.envelope),
                    amplitude=pulse.amplitude,
                    rf_frequency=frequency,
                    instantaneous_phase=pulse.phase_rad,
                    name=f"awg_rf_{pulse_index}",
                ),
                "channel": channel,
                "pre_delay": pre_delay,
            }
        )

    acquisition_layer_index = segment_layer_indices[acquisition.at_segment]
    acquisition_pre_delay_values = []
    acquisition_stop_values = []
    for point_index, boundaries in enumerate(boundary_rows):
        if acquisition.at_segment not in boundaries:
            raise KeyError(
                "QCS acquisition references unknown segment "
                f"{acquisition.at_segment!r}"
            )
        segment_start_s, segment_stop_s = boundaries[acquisition.at_segment]
        acquisition_delay_s = _fabric_aligned_seconds(
            acquisition.pre_delay_s,
            fabric_hz=fabric_hz,
            label="QCS acquisition pre-delay",
        )
        pre_delay_s = (
            float(
                segment_layer_offsets[acquisition.at_segment][point_index]
            )
            + acquisition_delay_s
        )
        acquisition_stop_s = (
            segment_start_s
            + acquisition_delay_s
            + acquisition_elapsed_duration_s
        )
        if (
            connection_config.hw_demod
            and acquisition_stop_s > segment_stop_s + 1e-15
        ):
            raise ValueError(
                "QCS acquisition exceeds segment "
                f"{acquisition.at_segment!r} at sweep point {point_index + 1}"
            )
        if (
            not connection_config.hw_demod
            and acquisition_stop_s > segment_stop_s + 1e-15
            and acquisition_layer_index < layer_count - 1
        ):
            raise ValueError(
                "raw QCS acquisition cannot cross segment "
                f"{acquisition.at_segment!r} while later DC segment layers "
                f"remain at sweep point {point_index + 1}"
            )
        acquisition_pre_delay_values.append(pre_delay_s)
        acquisition_stop_values.append(acquisition_stop_s)
    acquisition_pre_delay = _qcs_sweep_parameter(
        qcs,
        name="awg_acquisition_pre_delay",
        values=acquisition_pre_delay_values,
        targets=targets,
        hardware_supported=False,
        channel_name=connection_config.acquisition_channel_name,
        description="acquisition pre-delay",
        force=point_count > 1 and not targets,
    )
    acquisition_program_operations: list[dict[str, Any]] = []
    if connection_config.hw_demod:
        integration_filter_cache: dict[int, Any] = {}
        for segment_index, segment_samples in enumerate(
            integration_segment_sample_counts
        ):
            integration_filter = acquisition.integration_filter
            if integration_filter is None:
                integration_filter = integration_filter_cache.get(
                    segment_samples
                )
                if integration_filter is None:
                    filter_waveform = qcs.RFWaveform(
                        duration=(
                            segment_samples / acquisition_sample_rate_hz
                        ),
                        envelope=_qcs_envelope(qcs, acquisition.envelope),
                        amplitude=1.0,
                        rf_frequency=acquisition.frequency_hz,
                        instantaneous_phase=acquisition.phase_rad,
                        name=(
                            "awg_acquisition_filter_"
                            f"{segment_samples}_samples"
                        ),
                    )
                    integration_filter = (
                        qcs.IntegrationFilter(filter_waveform)
                        if len(integration_segment_sample_counts) > 1
                        else filter_waveform
                    )
                    integration_filter_cache[segment_samples] = (
                        integration_filter
                    )
            acquisition_program_operations.append(
                {
                    "integration_filter": integration_filter,
                    "channels": acquisition_channels,
                    "new_layer": False,
                    "pre_delay": (
                        acquisition_pre_delay
                        if segment_index == 0
                        else QCS_SPARAMETER_INTER_SEGMENT_DELAY_S
                    ),
                }
            )
    else:
        integration_filter = acquisition_duration_s
        acquisition_program_operations.append(
            {
                "integration_filter": integration_filter,
                "channels": acquisition_channels,
                "new_layer": False,
                "pre_delay": acquisition_pre_delay,
            }
        )

    # Keep every incoming ramp and its destination SET plateau in one layer.
    # RF and acquisition are delayed past the ramp on their own channels, so
    # all three operations share the layer that contains the swept voltage.
    for layer_index, interval_group in enumerate(interval_groups):
        for output_index, (channel, intervals) in enumerate(
            zip(dc_channels, dc_interval_operations)
        ):
            program.add_waveform(
                _qcs_join_grouped_dc_intervals(intervals, interval_group),
                channel,
                new_layer=output_index == 0,
            )
        for operation in rf_program_operations_by_layer.get(layer_index, ()):
            program.add_waveform(
                operation["waveform"],
                operation["channel"],
                new_layer=False,
                pre_delay=operation["pre_delay"],
            )
        if acquisition_layer_index == layer_index:
            for operation in acquisition_program_operations:
                program.add_acquisition(**operation)

    reasons = []
    if not connection_config.hw_demod and point_count > 1:
        reasons.append("raw trace acquisition requires QCS software resolution")
    reasons.extend(
        target.description
        for target in targets
        if not target.hardware_supported
    )
    per_channel_targets = {}
    for target in targets:
        if not target.hardware_supported:
            continue
        per_channel_targets.setdefault(target.channel_name, []).append(target)
    for channel_name, channel_targets in per_channel_targets.items():
        array_count = len(channel_targets)
        stored_values = sum(target.values.size for target in channel_targets)
        if array_count > MAX_QCS_HARDWARE_SWEEP_ARRAYS_PER_CHANNEL:
            reasons.append(
                f"{channel_name} uses {array_count} swept arrays; QCS hardware "
                f"supports at most {MAX_QCS_HARDWARE_SWEEP_ARRAYS_PER_CHANNEL}"
            )
        if stored_values >= MAX_QCS_HARDWARE_SWEEP_ARRAY_VALUES:
            reasons.append(
                f"{channel_name} stores {stored_values:,} sweep values; QCS "
                f"requires fewer than {MAX_QCS_HARDWARE_SWEEP_ARRAY_VALUES:,}"
            )
        channel = _resolve_mapper_channel(mapper, channel_name)
        if bool(getattr(channel, "absolute_phase", False)):
            reasons.append(
                f"{channel_name} uses absolute_phase=True, which is not "
                "hardware-sweepable"
            )
    reasons = list(dict.fromkeys(reasons))
    hardware_sweep = bool(point_count > 1 and not reasons)
    if point_count > 1 and not hardware_sweep:
        _validate_qcs_software_sweep_point_count(point_count)
    if point_count > 1:
        arrays = [target.array for target in targets]
        variables = [target.variable for target in targets]
        if hardware_sweep:
            # HCL hardware-sweeps only the innermost repetition.  Sweep must
            # therefore be inserted before Repeat/n_shots.
            program.sweep(arrays, variables)
            program.n_shots(repetitions)
        else:
            # QCS resolves unsupported settings outside the hardware boundary,
            # but this remains one Program and one Executor.execute call.
            program.n_shots(repetitions)
            program.sweep(arrays, variables)
    else:
        program.n_shots(repetitions)

    # Physical-channel settings are part of the mapper consumed by HCL. Set
    # them only after the complete Program has been built successfully, and
    # never include these Scalars in ``program.sweep`` (QCS 2.5.5 rejects
    # physical settings in hardware time).
    _set_qcs_dc_channel_offsets(
        mapper,
        channel_names=connection_config.dc_channel_names,
        offset_volts=dc_offset_plan.offset_volts,
        require_nonzero_support=True,
    )

    return QcsCompiledHardwareSweep(
        program=program,
        acquisition_channels=acquisition_channels,
        duration_s=float(
            max(max(total_duration_values), max(acquisition_stop_values))
        ),
        acquisition_duration_s=acquisition_duration_s,
        acquisition_sample_rate_hz=acquisition_sample_rate_hz,
        sweep_shape=sweep_shape,
        hardware_sweep=hardware_sweep,
        sweep_variable_count=len(targets),
        sweep_array_value_count=sum(target.values.size for target in targets),
        software_sweep_reasons=tuple(reasons),
        dc_channel_offsets_v=tuple(dc_offset_plan.offset_volts),
        dc_offset_init_compensation_v=tuple(
            (
                np.zeros(len(output_names), dtype=float)
                if offset_init_compensation_source is None
                else offset_init_compensation_source
            )
            * (float(source_full_scale_mv) / 1000.0)
        ),
        acquisition_sample_count=acquisition_sample_count,
        integration_segment_sample_counts=tuple(
            integration_segment_sample_counts
        ),
        inter_segment_delay_s=(
            QCS_SPARAMETER_INTER_SEGMENT_DELAY_S
            if connection_config.hw_demod
            else 0.0
        ),
        acquisition_elapsed_duration_s=acquisition_elapsed_duration_s,
    )


def _plan_qcs_mixed_sweep(
    sequence: Any,
    *,
    connection_config: QcsConnectionConfig,
    mapper: Any,
    repetitions_per_sweep: int,
    fabric_mhz: float,
    source_full_scale_mv: float,
    rf_pulses: Sequence[QcsRfPulseConfig],
    acquisition: QcsAcquisitionConfig,
    qcs_module: Any,
    cancellation: Optional[QcsCancellationController],
    include_full_candidate: bool = False,
    full_dc_offset_plan: Optional[_QcsFixedDcOffsetPlan] = None,
) -> Tuple[Optional[QcsMixedSweepPlan], Optional[QcsCompiledHardwareSweep]]:
    """Find the largest native inner sweep for a manual software loop.

    QCS can encode nested software/hardware repetitions, but one blocking
    ``Executor.execute`` call exposes no intermediate data. The GUI therefore
    executes every outer coordinate separately. Candidate subsets are tried
    from largest hardware point count to smallest; the real synchronized
    compiler and mapped-channel settings make the final decision.
    """

    axes = tuple(sequence.sweep_axes)
    if (
        len(axes) < 2
        or not connection_config.hw_demod
        or any(isinstance(axis, RfPowerSweep) for axis in axes)
    ):
        return None, None
    candidate_indices = _qcs_mixed_candidate_hardware_axes(sequence)
    if not candidate_indices:
        return None, None
    original_shape = tuple(int(value) for value in sequence.sweep_shape)
    all_axis_indices = tuple(range(len(axes)))
    planning_failures = []
    successful_compilations = {}
    failed_compilations = set()
    planning_started_s = monotonic()

    def compile_candidate(hardware_axes: Sequence[int]):
        hardware_axes = tuple(sorted(int(value) for value in hardware_axes))
        if hardware_axes in successful_compilations:
            return successful_compilations[hardware_axes]
        if hardware_axes in failed_compilations:
            return None
        software_axes = tuple(
            index for index in all_axis_indices if index not in hardware_axes
        )
        fixed = {
            axis_index: 0
            for axis_index in software_axes
        }
        sliced = _qcs_sequence_slice(sequence, fixed)
        representative_global_index = int(
            _qcs_slice_global_indices(original_shape, fixed)[0]
        )
        sliced_rf_pulses = _qcs_point_rf_pulses(
            sequence,
            rf_pulses,
            representative_global_index,
        )
        try:
            compiled = compile_qcs_synchronized_sweep(
                sliced,
                connection_config=connection_config,
                mapper=mapper,
                repetitions_per_sweep=repetitions_per_sweep,
                fabric_mhz=fabric_mhz,
                source_full_scale_mv=source_full_scale_mv,
                rf_pulses=sliced_rf_pulses,
                acquisition=acquisition,
                qcs_module=qcs_module,
                cancellation=cancellation,
                _capacity_prevalidated=False,
                _dc_offset_plan=(
                    full_dc_offset_plan
                    if hardware_axes == all_axis_indices
                    else None
                ),
            )
        except (QcsUnsupportedFeatureError, ValueError) as exc:
            failed_compilations.add(hardware_axes)
            planning_failures.append(
                f"axes {hardware_axes}: {type(exc).__name__}: {exc}"
            )
            return None
        finally:
            # Compilation can set mapped physical offsets. Planning must
            # never leave a voltage applied while Python evaluates candidates.
            _set_qcs_dc_channel_offsets(
                mapper,
                channel_names=connection_config.dc_channel_names,
                offset_volts=(0.0,) * len(connection_config.dc_channel_names),
                require_nonzero_support=False,
            )
        if not compiled.hardware_sweep:
            failed_compilations.add(hardware_axes)
            planning_failures.extend(compiled.software_sweep_reasons)
            return None
        successful_compilations[hardware_axes] = compiled
        return compiled

    def build_plan(hardware_axes: Sequence[int]):
        hardware_axes = tuple(sorted(int(value) for value in hardware_axes))
        software_axes = tuple(
            index for index in all_axis_indices if index not in hardware_axes
        )
        if not software_axes:
            return None
        compiled = successful_compilations.get(hardware_axes)
        if compiled is None:
            return None
        return QcsMixedSweepPlan(
            hardware_axis_indices=hardware_axes,
            software_axis_indices=software_axes,
            hardware_shape=tuple(original_shape[index] for index in hardware_axes),
            software_shape=tuple(original_shape[index] for index in software_axes),
            first_compiled=compiled,
            planning_reasons=tuple(dict.fromkeys(planning_failures)),
            planning_wall_s=monotonic() - planning_started_s,
        )

    # Compile the exact full sweep once when requested. This preserves every
    # already-native multidimensional sweep as one Program/one executor call,
    # while also detecting mapper-only reasons (for example absolute_phase)
    # that make a true mixed partition useful.
    if include_full_candidate:
        full_compiled = compile_candidate(all_axis_indices)
        if full_compiled is not None:
            return None, full_compiled

    # Keep planning bounded. Materializing the full power set makes a GUI run
    # exponential in the number of segment sweeps (20 axes would exceed one
    # million compilation attempts). First try the maximum set, then every
    # one-axis removal. That resolves the common QCS eight-array budget and a
    # single incompatible timing axis in at most N+1 attempts.
    candidate_indices = tuple(candidate_indices)
    compiled = compile_candidate(candidate_indices)
    if compiled is not None:
        if candidate_indices == all_axis_indices:
            return None, compiled
        plan = build_plan(candidate_indices)
        if plan is not None:
            return plan, None
    ranked_removals = tuple(
        sorted(
            candidate_indices,
            key=lambda index: (
                int(original_shape[index]),
                str(getattr(axes[index], "axis_kind", ""))
                in {"amplitude", "rf_frequency"},
                index,
            ),
        )
    )
    for removed_axis in ranked_removals:
        subset = tuple(
            index for index in candidate_indices if index != removed_axis
        )
        if not subset:
            continue
        compiled = compile_candidate(subset)
        if compiled is not None:
            plan = build_plan(subset)
            if plan is not None:
                return plan, None

    # Multiple incompatible axes require refinement. Establish which axes can
    # run natively on their own, then greedily combine them in descending
    # point-count / hardware-confidence order. This is linear in axis count
    # after the bounded one-axis-removal pass, never exponential.
    individually_supported = []
    for axis_index in sorted(
        candidate_indices,
        key=lambda index: (
            str(getattr(axes[index], "axis_kind", ""))
            in {"amplitude", "rf_frequency"},
            int(original_shape[index]),
            -index,
        ),
        reverse=True,
    ):
        if compile_candidate((axis_index,)) is not None:
            individually_supported.append(axis_index)
    selected = []
    for axis_index in individually_supported:
        trial = tuple(sorted((*selected, axis_index)))
        if compile_candidate(trial) is not None:
            selected.append(axis_index)
    if selected:
        return build_plan(tuple(selected)), None
    return None, None


def _qcs_partial_execution_result(
    *,
    sequence: Any,
    full_iq: np.ndarray,
    completed_mask: np.ndarray,
    sample_rate_hz: float,
    programs: Sequence[Any],
    raw_results: Sequence[Any],
    summary: Mapping[str, Any],
    rf_settings: Mapping[str, Any],
) -> QcsExecutionResult:
    """Build a finite, completed-only live snapshot for GUI consumers."""

    completed_indices = np.flatnonzero(np.asarray(completed_mask, dtype=bool))
    points = np.asarray(sequence.sweep_points)
    partial_points = points[completed_indices]
    partial_iq = np.asarray(full_iq)[completed_indices].copy()
    ddr_result = FineTuneDdrResult(
        sweep_points=partial_points.copy(),
        iq=partial_iq,
        sweep_axes=tuple(sequence.sweep_axes),
        # A partial Cartesian grid need not itself be rectangular. Consumers
        # use the explicit point coordinates; only the final result exposes
        # the semantic Cartesian shape through iq_grid.
        sweep_shape=(int(completed_indices.size),),
        cross_capacitance=np.asarray(sequence.cross_capacitance).copy(),
        sample_rate_hz=float(sample_rate_hz),
        fir_rate_profile="qcs_hardware_demod",
    )
    return QcsExecutionResult(
        ddr_result=ddr_result,
        programs=tuple(programs),
        raw_results=tuple(raw_results),
        program_summary=dict(summary),
        rf_settings=dict(rf_settings),
    )


def _execute_qcs_mixed_sweep(
    sequence: Any,
    *,
    plan: QcsMixedSweepPlan,
    connection_config: QcsConnectionConfig,
    mapper: Any,
    executor: Any,
    repetitions_per_sweep: int,
    fabric_mhz: float,
    source_full_scale_mv: float,
    rf_pulses: Sequence[QcsRfPulseConfig],
    acquisition: QcsAcquisitionConfig,
    iq_averaging_plan: Optional[QcsIqAveragingPlan],
    requested_acquisition: QcsAcquisitionConfig,
    qcs_module: Any,
    progress_callback: Optional[ProgressCallback],
    event_callback: Optional[EventCallback],
    partial_callback: Optional[PartialResultCallback],
    cancellation: Optional[QcsCancellationController],
) -> QcsExecutionResult:
    """Execute native inner sweeps one outer Python coordinate at a time."""

    qcs = qcs_module
    original_shape = tuple(int(value) for value in sequence.sweep_shape)
    point_count = int(sequence.sweep_point_count)
    outer_count = int(plan.software_iteration_count)
    inner_count = int(plan.hardware_points_per_iteration)
    _validate_qcs_software_sweep_point_count(outer_count)
    if outer_count * inner_count != point_count:
        raise ValueError("QCS mixed-sweep partition does not cover every point")
    averaging_plan = iq_averaging_plan
    if averaging_plan is None:
        averaging_plan = plan_qcs_total_iq_averaging(
            acquisition.duration_s,
            sample_rate_hz=acquisition.sample_rate_hz,
        )

    hardware_axes = tuple(plan.hardware_axis_indices)
    software_axes = tuple(plan.software_axis_indices)
    programs = []
    raw_results = []
    completed_mask = np.zeros(point_count, dtype=bool)
    full_iq = None
    latest_partial = None
    effective_sample_rate_hz = None
    effective_acquisition_duration_s = None
    sweep_variable_count = 0
    sweep_array_value_count = 0
    safety_reset_executor_call_count = 0
    mixed_program_build_wall_s = float(plan.planning_wall_s)
    mixed_executor_wall_s = 0.0
    measurement_executor_call_count = 0
    last_partial_publish_s = float("-inf")
    block_offset_values = []
    software_reasons = tuple(
        dict.fromkeys(
            (
                "Unsupported sweep axes are executed by a Python outer loop; "
                "each outer coordinate contains one native QCS hardware sweep.",
                *plan.planning_reasons,
            )
        )
    )

    rf_settings = {
        "backend": "qcs",
        "output_details": _qcs_rf_output_details(rf_pulses),
        "readout_details": {
            "sample_rate_hz": None,
            "hw_demod": True,
            "reset_phase_every_shot": True,
            "frequency_hz": acquisition.frequency_hz,
            "duration_s": None,
            "requested_sample_count": requested_acquisition.sample_count,
        },
    }

    def current_summary(*, partial: bool) -> Mapping[str, Any]:
        completed_points = int(np.count_nonzero(completed_mask))
        completed_iterations = len(raw_results)
        return {
            "backend": "qcs",
            "hardware_sweep": True,
            "partial": bool(partial),
            "completed_points": completed_points,
            "planned_points": point_count,
            "hardware_sweep_dimensions": len(hardware_axes),
            "hardware_sweep_axis_indices": list(hardware_axes),
            "hardware_sweep_shape": list(plan.hardware_shape),
            "hardware_sweep_points": point_count,
            "hardware_points_per_iteration": inner_count,
            "software_sweep_dimensions": len(software_axes),
            "software_sweep_axis_indices": list(software_axes),
            "software_sweep_shape": list(plan.software_shape),
            "software_sweep_points": outer_count,
            "software_iteration_count": outer_count,
            "completed_software_iterations": completed_iterations,
            "semantic_sweep_shape": list(original_shape),
            "sweep_execution_mode": "hybrid_hardware_software",
            "software_sweep_reasons": list(software_reasons),
            "sweep_variable_count": sweep_variable_count,
            "sweep_array_value_count": sweep_array_value_count,
            "repetitions_per_sweep": int(repetitions_per_sweep),
            "program_count": len(programs),
            "executor_call_count": measurement_executor_call_count,
            "safety_reset_executor_call_count": (
                safety_reset_executor_call_count
            ),
            "total_executor_call_count": (
                measurement_executor_call_count
                + safety_reset_executor_call_count
            ),
            "fabric_mhz": float(fabric_mhz),
            "source_full_scale_mv": float(source_full_scale_mv),
            "qcs_dc_full_scale_v": connection_config.dc_full_scale_v,
            "dc_channel_names": list(connection_config.dc_channel_names),
            "dc_channel_offsets_v": [],
            "dc_channel_offsets_v_by_iteration": [
                list(values) for values in block_offset_values
            ],
            "fixed_dc_offset_residualization": any(
                any(
                    not np.isclose(value, 0.0, rtol=0.0, atol=1e-15)
                    for value in values
                )
                for values in block_offset_values
            ),
            "rf_channel_names": {
                str(key): value
                for key, value in connection_config.rf_channel_names.items()
            },
            "acquisition_channel_name": (
                connection_config.acquisition_channel_name
            ),
            "hw_demod": True,
            "reset_phase_every_shot": True,
            "sample_rate_hz": effective_sample_rate_hz,
            "acquisition_duration_s": (
                averaging_plan.quantized_total_duration_s
            ),
            "requested_sample_count": requested_acquisition.sample_count,
            "requested_total_integration_duration_s": (
                averaging_plan.requested_total_duration_s
            ),
            "quantized_total_integration_duration_s": (
                averaging_plan.quantized_total_duration_s
            ),
            "iq_averaging_pass_count": averaging_plan.pass_count,
            "completed_iq_averaging_passes": (
                completed_iterations * averaging_plan.pass_count
            ),
            "per_pass_integration_duration_s": (
                effective_acquisition_duration_s
            ),
            "iq_shape": (
                [] if full_iq is None else list(full_iq.shape)
            ),
            "mixed_program_build_wall_s": mixed_program_build_wall_s,
            "mixed_partition_planning_wall_s": float(plan.planning_wall_s),
            "mixed_executor_wall_s": mixed_executor_wall_s,
        }

    if event_callback is not None:
        event_callback(
            "program_build",
            "started",
            (
                "Preparing a native QCS hardware sweep inside a Python "
                f"software loop ({outer_count:,} iteration(s))"
            ),
        )
    if progress_callback is not None:
        progress_callback(
            20,
            (
                f"Preflighting {outer_count:,} mixed QCS software-loop "
                "coordinate(s)"
            ),
        )

    # Validate the waveform capacity of every planned outer coordinate before
    # submitting the first program. The planner's representative compilation
    # proves the native inner partition, but an unsupported duration axis can
    # make a later slice longer than the M5301 buffer even when slice zero is
    # legal. This pass deliberately evaluates all inner points for every
    # outer coordinate: an invalid configuration must not partially touch
    # hardware before the GUI reports it.
    capacity_scale = (
        float(source_full_scale_mv)
        / (float(connection_config.dc_full_scale_v) * 1000.0)
    )
    for preflight_index, fixed in enumerate(
        _qcs_software_coordinates(original_shape, software_axes)
    ):
        if cancellation is not None:
            cancellation.raise_if_requested(
                "mixed QCS waveform-capacity preflight"
            )
        validate_qcs_m5301_waveform_capacity(
            _qcs_sequence_slice(sequence, fixed),
            fabric_mhz=fabric_mhz,
            amplitude_scale=capacity_scale,
            source_full_scale_mv=source_full_scale_mv,
            dc_full_scale_v=connection_config.dc_full_scale_v,
        )
        if progress_callback is not None and (
            outer_count <= 100
            or preflight_index == 0
            or preflight_index + 1 == outer_count
            or (preflight_index + 1) % max(1, outer_count // 100) == 0
        ):
            progress_callback(
                20 + int(15 * (preflight_index + 1) / outer_count),
                (
                    "Preflighted mixed QCS coordinate "
                    f"{preflight_index + 1:,}/{outer_count:,}"
                ),
            )

    if event_callback is not None:
        event_callback(
            "program_build",
            "completed",
            (
                "Preflighted and configured hardware sweep inside software "
                f"loop: {outer_count:,} outer iteration(s) x {inner_count:,} "
                "native hardware point(s)"
            ),
        )
        event_callback(
            "acquisition",
            "started",
            "Executing native QCS hardware sweeps inside the software loop",
        )
    if progress_callback is not None:
        progress_callback(
            35,
            (
                f"Running mixed QCS sweep: {outer_count:,} software "
                f"iteration(s) x {inner_count:,} hardware point(s)"
            ),
        )

    try:
        for outer_index, fixed in enumerate(
            _qcs_software_coordinates(original_shape, software_axes)
        ):
            if cancellation is not None:
                cancellation.raise_if_requested(
                    "mixed QCS software-loop compilation"
                )
            sliced = _qcs_sequence_slice(sequence, fixed)
            global_indices = _qcs_slice_global_indices(original_shape, fixed)
            representative_global_index = int(global_indices[0])
            sliced_rf_pulses = _qcs_point_rf_pulses(
                sequence,
                rf_pulses,
                representative_global_index,
            )
            if outer_index == 0:
                # The planner already compiled this exact all-zero outer
                # coordinate. Reuse it: a large native inner grid can take
                # seconds to build. Planning kept mapper offsets at zero;
                # enforce that invariant again immediately before submission.
                compiled = plan.first_compiled
                _set_qcs_dc_channel_offsets(
                    mapper,
                    channel_names=connection_config.dc_channel_names,
                    offset_volts=compiled.dc_channel_offsets_v,
                    require_nonzero_support=True,
                )
            else:
                build_started_s = monotonic()
                compiled = compile_qcs_synchronized_sweep(
                    sliced,
                    connection_config=connection_config,
                    mapper=mapper,
                    repetitions_per_sweep=repetitions_per_sweep,
                    fabric_mhz=fabric_mhz,
                    source_full_scale_mv=source_full_scale_mv,
                    rf_pulses=sliced_rf_pulses,
                    acquisition=acquisition,
                    qcs_module=qcs,
                    cancellation=cancellation,
                    _capacity_prevalidated=False,
                )
                mixed_program_build_wall_s += monotonic() - build_started_s
            if not compiled.hardware_sweep:
                raise QcsUnsupportedFeatureError(
                    "QCS mixed-sweep inner block unexpectedly downgraded to "
                    "software execution: "
                    + "; ".join(compiled.software_sweep_reasons)
                )
            if int(np.prod(compiled.sweep_shape, dtype=np.int64)) != inner_count:
                raise ValueError(
                    "QCS mixed-sweep inner block changed its point count"
                )
            block_offsets = tuple(compiled.dc_channel_offsets_v)
            block_offset_values.append(block_offsets)
            offset_reset_required = any(
                not np.isclose(value, 0.0, rtol=0.0, atol=1e-15)
                for value in block_offsets
            )
            if cancellation is not None:
                cancellation.tag_program(compiled.program)
                cancellation.raise_if_requested(
                    "mixed QCS hardware-sweep execution"
                )
            block_weighted_sum = None
            raw_result = None
            for pass_index, pass_samples in enumerate(
                averaging_plan.pass_sample_counts
            ):
                if cancellation is not None:
                    cancellation.raise_if_requested(
                        "mixed QCS integrated-I/Q averaging pass"
                    )
                executor_started_s = monotonic()
                raw_result = _executor_execute(executor, compiled.program)
                mixed_executor_wall_s += monotonic() - executor_started_s
                measurement_executor_call_count += 1
                pass_iq = _extract_qcs_segmented_hardware_sweep_iq(
                    raw_result,
                    acquisition_channels=compiled.acquisition_channels,
                    segment_sample_counts=(
                        compiled.integration_segment_sample_counts
                        or (int(compiled.acquisition_sample_count or 1),)
                    ),
                    repetitions_per_point=repetitions_per_sweep,
                    sweep_shape=compiled.sweep_shape,
                    hardware_sweep=True,
                )
                weighted = np.asarray(pass_iq) * float(pass_samples)
                if block_weighted_sum is None:
                    block_weighted_sum = weighted
                else:
                    block_weighted_sum += weighted
                if (
                    cancellation is not None
                    and pass_index + 1 < averaging_plan.pass_count
                ):
                    cancellation.raise_if_requested(
                        "mixed QCS integrated-I/Q averaging boundary"
                    )
            if block_weighted_sum is None or raw_result is None:
                raise RuntimeError("mixed QCS averaging completed no passes")
            block_iq = block_weighted_sum / float(
                sum(averaging_plan.pass_sample_counts)
            )
            if block_iq.shape[0] != global_indices.size:
                raise ValueError(
                    "QCS mixed-sweep block returned an unexpected point count"
                )
            if full_iq is None:
                full_iq = np.empty(
                    (point_count, *block_iq.shape[1:]),
                    dtype=block_iq.dtype,
                )
            elif block_iq.shape[1:] != full_iq.shape[1:]:
                raise ValueError(
                    "QCS mixed-sweep blocks returned inconsistent IQ shapes"
                )
            full_iq[global_indices] = block_iq
            completed_mask[global_indices] = True
            programs.append(compiled.program)
            raw_results.append(raw_result)
            sweep_variable_count = max(
                sweep_variable_count,
                int(compiled.sweep_variable_count),
            )
            sweep_array_value_count = max(
                sweep_array_value_count,
                int(compiled.sweep_array_value_count),
            )
            if effective_sample_rate_hz is None:
                effective_sample_rate_hz = float(
                    compiled.acquisition_sample_rate_hz
                )
                effective_acquisition_duration_s = float(
                    compiled.acquisition_duration_s
                )
            elif not np.isclose(
                effective_sample_rate_hz,
                compiled.acquisition_sample_rate_hz,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "QCS mixed-sweep blocks used inconsistent sample rates"
                )

            # Physical offsets are mapper settings, not waveform operations.
            # Remove them immediately after this complete block so Python/QCS
            # preparation time cannot contribute uncontrolled DC area.
            if offset_reset_required:
                _reset_qcs_dc_outputs_to_zero(
                    qcs,
                    executor=executor,
                    mapper=mapper,
                    channel_names=connection_config.dc_channel_names,
                    fabric_hz=float(fabric_mhz) * 1e6,
                )
                safety_reset_executor_call_count += 1
            else:
                _set_qcs_dc_channel_offsets(
                    mapper,
                    channel_names=connection_config.dc_channel_names,
                    offset_volts=(0.0,)
                    * len(connection_config.dc_channel_names),
                    require_nonzero_support=False,
                )

            rf_settings["readout_details"]["sample_rate_hz"] = (
                effective_sample_rate_hz
            )
            rf_settings["readout_details"]["duration_s"] = (
                effective_acquisition_duration_s
            )
            publish_now_s = monotonic()
            publish_partial = bool(
                partial_callback is not None
                and (
                    outer_count <= 100
                    or outer_index == 0
                    or outer_index + 1 == outer_count
                    or publish_now_s - last_partial_publish_s >= 0.2
                )
            )
            if publish_partial:
                latest_partial = _qcs_partial_execution_result(
                    sequence=sequence,
                    full_iq=full_iq,
                    completed_mask=completed_mask,
                    sample_rate_hz=effective_sample_rate_hz,
                    programs=programs,
                    raw_results=raw_results,
                    summary=current_summary(partial=True),
                    rf_settings=rf_settings,
                )
                partial_callback(latest_partial)
                last_partial_publish_s = publish_now_s
            if event_callback is not None:
                event_callback(
                    "sweep_iteration",
                    "completed",
                    (
                        f"Mixed QCS iteration {outer_index + 1:,}/"
                        f"{outer_count:,}; retained "
                        f"{int(np.count_nonzero(completed_mask)):,}/"
                        f"{point_count:,} point(s)"
                    ),
                )
            if progress_callback is not None:
                progress_callback(
                    35 + int(35 * (outer_index + 1) / outer_count),
                    (
                        f"Acquired mixed QCS iteration {outer_index + 1:,}/"
                        f"{outer_count:,} ({int(np.count_nonzero(completed_mask)):,}/"
                        f"{point_count:,} points available)"
                    ),
                )
            if cancellation is not None:
                cancellation.raise_if_requested(
                    "mixed QCS software-loop boundary"
                )
    except BaseException as execution_error:
        cancellation_requested = bool(
            cancellation is not None and cancellation.is_stop_requested()
        )
        reset_error_caught = None
        try:
            _reset_qcs_dc_outputs_to_zero(
                qcs,
                executor=executor,
                mapper=mapper,
                channel_names=connection_config.dc_channel_names,
                fabric_hz=float(fabric_mhz) * 1e6,
            )
            safety_reset_executor_call_count += 1
        except Exception as reset_error:
            reset_error_caught = reset_error
            if hasattr(execution_error, "add_note"):
                execution_error.add_note(
                    "Automatic QCS DC reset also failed: "
                    f"{type(reset_error).__name__}: {reset_error}"
                )
        if cancellation_requested:
            if reset_error_caught is not None:
                raise RuntimeError(
                    "QCS stop was requested, but the emergency DC reset "
                    "failed; the physical output state is unknown"
                ) from reset_error_caught
            completed_point_count = int(np.count_nonzero(completed_mask))
            latest_partial_point_count = (
                0
                if latest_partial is None
                else int(
                    latest_partial.program_summary.get(
                        "completed_points",
                        0,
                    )
                )
            )
            if (
                full_iq is not None
                and completed_point_count > 0
                and latest_partial_point_count != completed_point_count
            ):
                latest_partial = _qcs_partial_execution_result(
                    sequence=sequence,
                    full_iq=full_iq,
                    completed_mask=completed_mask,
                    sample_rate_hz=effective_sample_rate_hz,
                    programs=programs,
                    raw_results=raw_results,
                    summary=current_summary(partial=True),
                    rf_settings=rf_settings,
                )
            raise QcsExperimentCancelled(
                "QCS mixed sweep stopped by user; completed blocks remain "
                "available and DC outputs were reset to zero",
                partial_result=latest_partial,
            ) from execution_error
        raise

    if full_iq is None or not np.all(completed_mask):
        raise RuntimeError("QCS mixed sweep ended before every block completed")
    if cancellation is not None and not cancellation.close_stop_window():
        try:
            _reset_qcs_dc_outputs_to_zero(
                qcs,
                executor=executor,
                mapper=mapper,
                channel_names=connection_config.dc_channel_names,
                fabric_hz=float(fabric_mhz) * 1e6,
            )
        except Exception as reset_error:
            raise RuntimeError(
                "QCS stop was requested, but the emergency DC reset failed; "
                "the physical output state is unknown"
            ) from reset_error
        if latest_partial is None:
            latest_partial = _qcs_partial_execution_result(
                sequence=sequence,
                full_iq=full_iq,
                completed_mask=completed_mask,
                sample_rate_hz=effective_sample_rate_hz,
                programs=programs,
                raw_results=raw_results,
                summary=current_summary(partial=True),
                rf_settings=rf_settings,
            )
        raise QcsExperimentCancelled(
            "QCS mixed sweep stopped by user after its final hardware block; "
            "all acquired points remain available",
            partial_result=latest_partial,
        )

    final_ddr = FineTuneDdrResult(
        sweep_points=np.asarray(sequence.sweep_points).copy(),
        iq=full_iq,
        sweep_axes=tuple(sequence.sweep_axes),
        sweep_shape=tuple(sequence.sweep_shape),
        cross_capacitance=np.asarray(sequence.cross_capacitance).copy(),
        sample_rate_hz=float(effective_sample_rate_hz),
        fir_rate_profile="qcs_hardware_demod",
    )
    final_summary = current_summary(partial=False)
    final_summary["completed_points"] = point_count
    if event_callback is not None:
        event_callback(
            "acquisition",
            "completed",
            (
                f"Acquired {point_count:,} QCS point(s) with "
                f"{outer_count:,} native hardware-sweep execution(s) "
                "through one persistent executor"
            ),
        )
    return QcsExecutionResult(
        ddr_result=final_ddr,
        programs=tuple(programs),
        raw_results=tuple(raw_results),
        program_summary=final_summary,
        rf_settings=rf_settings,
    )


def _executor_execute(executor: Any, program: Any) -> Any:
    if hasattr(executor, "execute"):
        return executor.execute(program)
    if callable(executor):
        return executor(program)
    raise TypeError("QCS executor must be callable or expose execute(program)")


def _reset_qcs_dc_outputs_to_zero(
    qcs: Any,
    *,
    executor: Any,
    mapper: Any,
    channel_names: Sequence[str],
    fabric_hz: float,
) -> None:
    """Clear waveform amplitude and any persistent physical-channel offset."""

    channel_names = tuple(str(value) for value in channel_names)
    _set_qcs_dc_channel_offsets(
        mapper,
        channel_names=channel_names,
        offset_volts=(0.0,) * len(channel_names),
        require_nonzero_support=False,
    )
    program = qcs.Program(name="PulseGenerator emergency DC reset")
    duration_s = QCS_M5301_MIN_WAVEFORM_FABRIC_CYCLES / float(fabric_hz)
    for output_index, channel_name in enumerate(channel_names):
        program.add_waveform(
            qcs.DCWaveform(
                duration=duration_s,
                envelope=qcs.ConstantEnvelope(),
                amplitude=0.0,
                name=f"emergency_dc_zero_{output_index}",
            ),
            _resolve_mapper_channel(mapper, channel_name),
            new_layer=output_index == 0,
        )
    program.n_shots(1)
    _executor_execute(executor, program)


def _first_result_value(value: Any, channels: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    try:
        if channels in value:
            return value[channels]
    except TypeError:
        pass
    if len(value) != 1:
        raise ValueError(
            "QCS acquisition returned multiple channel arrays unexpectedly"
        )
    return next(iter(value.values()))


def extract_qcs_acquisition(
    raw_result: Any,
    channels: Any,
    *,
    prefer_trace: bool = False,
) -> Any:
    """Extract an IQ/trace array from a real or injected QCS result."""
    results = getattr(raw_result, "results", None)
    if results is not None:
        get_trace = getattr(results, "get_trace", None)
        if prefer_trace and callable(get_trace):
            return _first_result_value(
                get_trace(channels, avg=False), channels
            )
        get_iq = getattr(results, "get_iq", None)
        if callable(get_iq):
            try:
                return _first_result_value(
                    get_iq(channels, avg=False), channels
                )
            except (KeyError, RuntimeError, TypeError, ValueError):
                if callable(get_trace):
                    return _first_result_value(
                        get_trace(channels, avg=False), channels
                    )
                raise
    try:
        return raw_result[channels]
    except (IndexError, KeyError, TypeError):
        return raw_result


def normalize_qcs_iq(
    values: Any,
    *,
    repetitions_per_sweep: int,
    real_is_i_trace: bool = False,
) -> np.ndarray:
    """Normalize one point to ``(repetition, sample, I/Q)``."""
    repetitions = int(repetitions_per_sweep)
    if repetitions < 1:
        raise ValueError("repetitions_per_sweep must be positive")

    if isinstance(values, (tuple, list)) and len(values) == 2:
        i_values = np.asarray(values[0])
        q_values = np.asarray(values[1])
        if i_values.shape != q_values.shape:
            raise ValueError("QCS I and Q arrays have different shapes")
        array = i_values.astype(float) + 1j * q_values.astype(float)
    else:
        array = np.asarray(values)

    if array.size == 0:
        raise ValueError("QCS acquisition returned no samples")
    if np.iscomplexobj(array):
        if array.size % repetitions:
            raise ValueError(
                "QCS IQ element count is not divisible by repetitions"
            )
        # QCS stores shots on the final result axis.  Accept an already
        # repetition-first injected array as a convenience for tests/adapters,
        # then fall back to the legacy flat representation when neither
        # boundary dimension identifies the shot count.
        if array.ndim > 0 and array.shape[0] == repetitions:
            complex_values = array.reshape(repetitions, -1)
        elif array.ndim > 1 and array.shape[-1] == repetitions:
            complex_values = np.moveaxis(array, -1, 0).reshape(
                repetitions, -1
            )
        else:
            complex_values = array.reshape(repetitions, -1)
        return np.stack(
            (complex_values.real, complex_values.imag), axis=-1
        ).astype(np.float64, copy=False)

    if (
        not real_is_i_trace
        and array.ndim >= 1
        and array.shape[-1] == 2
    ):
        if array.size % (repetitions * 2):
            raise ValueError(
                "QCS I/Q element count is not divisible by repetitions"
            )
        if array.shape[0] == repetitions:
            iq_values = array
        elif array.ndim > 2 and array.shape[-2] == repetitions:
            iq_values = np.moveaxis(array, -2, 0)
        else:
            iq_values = array.reshape(repetitions, -1, 2)
        return iq_values.reshape(repetitions, -1, 2).astype(
            np.float64, copy=False
        )
    if np.issubdtype(array.dtype, np.number):
        if array.size % repetitions:
            raise ValueError(
                "QCS trace element count is not divisible by repetitions"
            )
        if array.ndim > 0 and array.shape[0] == repetitions:
            i_values = array.reshape(repetitions, -1)
        elif array.ndim > 1 and array.shape[-1] == repetitions:
            i_values = np.moveaxis(array, -1, 0).reshape(
                repetitions, -1
            )
        else:
            i_values = array.reshape(repetitions, -1)
        i_values = i_values.astype(np.float64, copy=False)
        return np.stack((i_values, np.zeros_like(i_values)), axis=-1)
    raise TypeError(
        "QCS acquisition must be numeric, complex-valued, an (I, Q) pair, "
        "or have a final I/Q axis of length 2"
    )


def normalize_qcs_hardware_sweep_iq(
    values: Any,
    *,
    repetitions_per_point: int,
    sweep_shape: Sequence[int],
    hardware_sweep: bool = True,
) -> np.ndarray:
    """Normalize synchronized QCS sweep IQ to ``(point, shot, 1, I/Q)``.

    Native QCS 2.5.5 results follow the Program repetition order with the
    shot axis first.  A single synchronized sweep has one flattened Cartesian
    point axis, while injected or older saved results may retain the semantic
    sweep axes.  Shot-last (software-resolved) forms are also accepted.
    """
    repetitions = int(repetitions_per_point)
    if repetitions < 1:
        raise ValueError("repetitions_per_point must be positive")
    shape = tuple(int(value) for value in sweep_shape)
    if not shape or any(value < 1 for value in shape):
        raise ValueError("QCS sweep_shape must contain positive axes")

    if isinstance(values, (tuple, list)) and len(values) == 2:
        i_values = np.asarray(values[0])
        q_values = np.asarray(values[1])
        if i_values.shape != q_values.shape:
            raise ValueError("QCS hardware-sweep I and Q shapes differ")
        array = i_values.astype(float) + 1j * q_values.astype(float)
    else:
        array = np.asarray(values)
        if (
            not np.iscomplexobj(array)
            and array.ndim >= 1
            and array.shape[-1] == 2
        ):
            array = (
                array[..., 0].astype(float)
                + 1j * array[..., 1].astype(float)
            )

    expected_count = repetitions * int(np.prod(shape))
    if array.size != expected_count:
        raise ValueError(
            "QCS synchronized-sweep IQ contains "
            f"{array.size:,} values; expected {expected_count:,} for "
            f"{int(np.prod(shape)):,} points x {repetitions} repetitions"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("QCS hardware-sweep IQ must be numeric")

    repetition_first = (repetitions, *shape)
    repetition_last = (*shape, repetitions)
    point_count = int(np.prod(shape))
    flattened_repetition_first = (repetitions, point_count)
    flattened_repetition_last = (point_count, repetitions)
    if array.ndim == 0 and expected_count == 1:
        # The real QCS Results API removes every singleton axis for a
        # one-point, one-shot acquisition and returns a scalar.  Restore the
        # Program repetition shape before applying the common point/shot
        # normalization.
        grid = array.reshape(repetition_first)
    elif array.shape == repetition_first and (
        hardware_sweep or repetition_first != repetition_last
    ):
        grid = array
    elif array.shape == repetition_last:
        grid = np.moveaxis(array, -1, 0)
    elif array.shape == repetition_first:
        grid = array
    elif array.shape == flattened_repetition_first:
        if hardware_sweep or (
            flattened_repetition_first != flattened_repetition_last
        ):
            grid = array.reshape(repetition_first)
        else:
            grid = np.moveaxis(array, -1, 0).reshape(repetition_first)
    elif array.shape == flattened_repetition_last:
        grid = np.moveaxis(array, -1, 0).reshape(repetition_first)
    elif repetitions == 1 and array.shape == shape:
        grid = array[np.newaxis, ...]
    elif repetitions == 1 and array.shape == (point_count,):
        grid = array.reshape(repetition_first)
    elif array.ndim == 1:
        # A shape-stripping loader preserves the active Program repetition
        # order. Hardware sweeps are shot-major; QCS-managed software sweeps
        # are point-major because ``n_shots`` is nested inside ``sweep``.
        if hardware_sweep:
            grid = array.reshape(repetition_first)
        else:
            grid = np.moveaxis(
                array.reshape(repetition_last),
                -1,
                0,
            )
    else:
        raise ValueError(
            "QCS synchronized-sweep IQ shape must be "
            f"{repetition_first}, {repetition_last}, "
            f"{flattened_repetition_first}, or "
            f"{flattened_repetition_last}; received {array.shape}"
        )

    point_shot = np.moveaxis(grid, 0, -1).reshape(-1, repetitions)
    return np.stack(
        (point_shot.real, point_shot.imag),
        axis=-1,
    ).astype(np.float64, copy=False)[:, :, np.newaxis, :]


def normalize_qcs_synchronized_trace(
    values: Any,
    *,
    repetitions_per_point: int,
    sweep_shape: Sequence[int],
) -> np.ndarray:
    """Normalize a QCS software-resolved trace sweep to point-first I/Q."""

    repetitions = int(repetitions_per_point)
    if repetitions < 1:
        raise ValueError("repetitions_per_point must be positive")
    shape = tuple(int(value) for value in sweep_shape)
    if not shape or any(value < 1 for value in shape):
        raise ValueError("QCS sweep_shape must contain positive axes")
    point_count = int(np.prod(shape))
    if isinstance(values, (tuple, list)) and len(values) == 2:
        i_values = np.asarray(values[0])
        q_values = np.asarray(values[1])
        if i_values.shape != q_values.shape:
            raise ValueError("QCS synchronized trace I and Q shapes differ")
        array = i_values.astype(float) + 1j * q_values.astype(float)
    else:
        array = np.asarray(values)
    if array.size == 0 or array.size % (point_count * repetitions):
        raise ValueError(
            "QCS synchronized trace element count must be divisible by "
            f"{point_count:,} points x {repetitions} repetitions"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("QCS synchronized trace must be numeric")

    semantic_software_prefix = (*shape, repetitions)
    semantic_hardware_prefix = (repetitions, *shape)
    flat_software_prefix = (point_count, repetitions)
    flat_hardware_prefix = (repetitions, point_count)
    if array.shape[: len(semantic_software_prefix)] == semantic_software_prefix:
        point_shot = array.reshape(point_count, repetitions, -1)
    elif array.shape[: len(semantic_hardware_prefix)] == semantic_hardware_prefix:
        sample_shape = array.shape[len(semantic_hardware_prefix) :]
        grid = array.reshape(repetitions, point_count, *sample_shape)
        point_shot = np.moveaxis(grid, 0, 1).reshape(
            point_count,
            repetitions,
            -1,
        )
    elif array.shape[:2] == flat_software_prefix:
        point_shot = array.reshape(point_count, repetitions, -1)
    elif array.shape[:2] == flat_hardware_prefix:
        point_shot = np.moveaxis(array, 0, 1).reshape(
            point_count,
            repetitions,
            -1,
        )
    elif array.ndim == 1:
        point_shot = array.reshape(point_count, repetitions, -1)
    else:
        raise ValueError(
            "QCS synchronized trace shape must start with "
            f"{flat_software_prefix}, {flat_hardware_prefix}, "
            f"{semantic_software_prefix}, or {semantic_hardware_prefix}; "
            f"received {array.shape}"
        )
    if np.iscomplexobj(point_shot):
        return np.stack((point_shot.real, point_shot.imag), axis=-1).astype(
            np.float64,
            copy=False,
        )
    i_values = point_shot.astype(np.float64, copy=False)
    return np.stack((i_values, np.zeros_like(i_values)), axis=-1)


def build_qcs_executor(
    connection_config: QcsConnectionConfig,
    mapper: Any,
    *,
    qcs_module=None,
    reset_phase_every_shot: bool = True,
) -> Any:
    """Create one phase-coherent HCL executor for a complete QCS run.

    The QSTL hardware-IQ examples reset phase before every shot so repeated
    integration-filter results share the same phase reference. Raw voltage
    traces opt out because they have no demodulation phase to reset. The same
    backend instance executes the complete synchronized sweep.
    """
    qcs = _import_qcs() if qcs_module is None else qcs_module
    backend = qcs.HclBackend(
        channel_mapper=mapper,
        hw_demod=connection_config.hw_demod,
        init_time=connection_config.init_time_s,
        blocking=connection_config.blocking,
        suppress_rounding_warnings=True,
        keep_progress_bar=False,
        reset_phase_every_shot=bool(reset_phase_every_shot),
    )
    return qcs.Executor(backend)


def acquire_qcs_noise_trace(
    config: QcsNoiseTraceConfig,
    *,
    qcs_module=None,
    mapper: Any = None,
    executor: Any = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> QcsNoiseTraceResult:
    """Acquire one duration-defined raw M5200 trace for Noise Analysis.

    Exactly one ``Program`` and one ``Executor.execute`` call are used.  A
    float duration is passed directly to ``Program.add_acquisition`` so QCS
    returns the time-domain trace rather than an IntegrationFilter result.
    """

    if not isinstance(config, QcsNoiseTraceConfig):
        raise TypeError("config must be QcsNoiseTraceConfig")

    def progress(percent: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(int(percent), str(message))

    qcs = _import_qcs() if qcs_module is None else qcs_module
    connection = config.connection_config
    if connection.hw_demod:
        # The dataclass already canonicalizes this setting.  Keep the
        # execution boundary defensive for callers deserializing older data.
        connection = replace(connection, hw_demod=False)

    progress(5, "Loading QCS ChannelMapper")
    if mapper is None:
        mapper = load_qcs_channel_mapper(connection, qcs_module=qcs)
    acquisition_name = connection.acquisition_channel_name
    if acquisition_name is None:
        raise ValueError(
            "QCS Noise Analysis requires a mapped M5200 acquisition channel"
        )
    acquisition_channels = _resolve_mapper_channel(
        mapper, acquisition_name
    )
    _validate_mapped_hardware_role(
        mapper,
        acquisition_channels,
        name=acquisition_name,
        role="noise acquisition",
        expected_instruments=("M5200Digitizer",),
    )
    sample_rate_hz = _mapped_channel_sample_rate(
        mapper, acquisition_channels
    )
    if sample_rate_hz is None:
        sample_rate_hz = QCS_M5200_SAMPLE_RATE_HZ
    duration_s, sample_count = quantize_qcs_raw_trace_duration(
        config.duration_s,
        sample_rate_hz,
    )

    progress(
        20,
        f"Building {sample_count:,}-sample raw M5200 acquisition",
    )
    program = qcs.Program(name="PulseGenerator QCS Noise Analysis raw trace")
    program.add_acquisition(
        integration_filter=duration_s,
        channels=acquisition_channels,
        new_layer=True,
    )
    program.n_shots(config.repetitions)
    if executor is None:
        executor = build_qcs_executor(
            connection,
            mapper,
            qcs_module=qcs,
            # A raw digitizer voltage trace has no demodulation phase to
            # reset. Avoid the extra phase-reset delay between captures.
            reset_phase_every_shot=False,
        )

    progress(35, "Executing one raw-trace QCS program")
    raw_result = _executor_execute(executor, program)
    progress(85, "Reading raw M5200 trace")
    reported_rate_hz = _executed_program_sample_rate(
        raw_result,
        acquisition_channels,
    )
    if reported_rate_hz is not None and not np.isclose(
        reported_rate_hz,
        sample_rate_hz,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise RuntimeError(
            "QCS raw trace reported a digitizer sample rate inconsistent "
            f"with its mapper ({reported_rate_hz:g} versus "
            f"{sample_rate_hz:g} S/s)"
        )
    values = extract_qcs_acquisition(
        raw_result,
        acquisition_channels,
        prefer_trace=True,
    )
    iq = normalize_qcs_iq(
        values,
        repetitions_per_sweep=config.repetitions,
        real_is_i_trace=True,
    )
    expected_shape = (config.repetitions, sample_count, 2)
    if iq.shape != expected_shape:
        raise RuntimeError(
            f"QCS raw noise trace has shape {iq.shape}; expected "
            f"{expected_shape} for {config.repetitions} repetition(s)"
        )
    result = QcsNoiseTraceResult(
        i_traces=iq[..., 0],
        requested_duration_s=config.duration_s,
        duration_s=duration_s,
        sample_rate_hz=sample_rate_hz,
        sample_count=sample_count,
        program=program,
        raw_result=raw_result,
    )
    progress(100, "QCS raw noise trace acquired")
    return result


def compile_qcs_sparameter_sweep(
    *,
    connection_config: QcsConnectionConfig,
    mapper: Any,
    frequencies_hz: Sequence[float],
    rf_gen_ch: int,
    rf_amplitude: float | Sequence[float],
    integration_duration_s: float,
    repetitions_per_point: int = 1,
    phase_rad: float = 0.0,
    calibrated_output: bool = False,
    qcs_module=None,
) -> QcsCompiledSParameterSweep:
    """Build one M5300/M5200 frequency sweep in a single QCS Program.

    A single direct QCS ``Scalar`` is referenced by both the output
    ``RFWaveform`` and the M5200 ``IntegrationFilter`` waveform.  Advancing
    that scalar therefore keeps generation and demodulation coherent without
    rebuilding a backend or making one Executor call per frequency. QCS 2.5.5
    cannot change an M5200 IntegrationFilter frequency inside hardware time,
    so this sweep is intentionally placed outside ``n_shots`` and resolved by
    QCS software in one submitted Program.
    """

    qcs = _import_qcs() if qcs_module is None else qcs_module
    if not connection_config.hw_demod:
        raise QcsUnsupportedFeatureError(
            "QCS RF S-parameter sweep requires hardware demodulation"
        )
    if not connection_config.blocking:
        raise QcsUnsupportedFeatureError(
            "QCS RF S-parameter sweep requires blocking=True"
        )
    if isinstance(rf_gen_ch, bool) or int(rf_gen_ch) != rf_gen_ch:
        raise TypeError("QCS S-parameter RF generator number must be an integer")
    rf_gen_ch = int(rf_gen_ch)
    if rf_gen_ch < 0:
        raise ValueError("QCS S-parameter RF generator number must be nonnegative")
    if rf_gen_ch not in connection_config.rf_channel_names:
        raise KeyError(
            f"QCS RF generator {rf_gen_ch} is not present in the active "
            "front-panel mapping"
        )
    if connection_config.acquisition_channel_name is None:
        raise ValueError(
            "QCS RF S-parameter sweep requires a mapped M5200 acquisition "
            "channel"
        )

    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or frequencies.size < 2:
        raise ValueError(
            "QCS RF S-parameter sweep requires at least two frequency points"
        )
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("QCS S-parameter frequencies must be finite and positive")
    if np.unique(frequencies).size != frequencies.size:
        raise ValueError("QCS S-parameter frequencies must be unique")
    if frequencies.size >= MAX_QCS_HARDWARE_SWEEP_ARRAY_VALUES:
        raise QcsUnsupportedFeatureError(
            "QCS RF S-parameter frequency sweep stores "
            f"{frequencies.size:,} values; QCS 2.5.5 requires fewer than "
            f"{MAX_QCS_HARDWARE_SWEEP_ARRAY_VALUES:,} values per channel"
        )

    amplitude_input = np.asarray(rf_amplitude, dtype=float)
    if amplitude_input.ndim == 0:
        amplitudes = np.full(
            frequencies.shape,
            float(amplitude_input),
            dtype=float,
        )
    elif amplitude_input.ndim == 1 and amplitude_input.shape == frequencies.shape:
        amplitudes = amplitude_input.astype(float, copy=True)
    else:
        raise ValueError(
            "QCS S-parameter RF amplitudes must be one scalar or one value "
            "for every frequency"
        )
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("QCS S-parameter RF amplitudes must be finite")
    if calibrated_output:
        if np.any(amplitudes <= 0.0) or np.any(amplitudes > 1.0):
            raise ValueError(
                "Calibrated QCS M5300 relative amplitudes must be in (0, 1]"
            )
    elif np.any(amplitudes < -1.0) or np.any(amplitudes > 1.0):
        raise ValueError("QCS S-parameter RF amplitudes must be in [-1, 1]")
    amplitude_varies = bool(np.any(amplitudes != amplitudes[0]))
    phase = float(phase_rad)
    if not isfinite(phase):
        raise ValueError("QCS S-parameter RF phase must be finite")
    requested_duration_s = _positive_finite(
        integration_duration_s,
        "QCS S-parameter integration duration",
    )
    if (
        isinstance(repetitions_per_point, bool)
        or int(repetitions_per_point) != repetitions_per_point
        or int(repetitions_per_point) < 1
    ):
        raise ValueError(
            "QCS S-parameter repetitions per point must be a positive integer"
        )
    repetitions = int(repetitions_per_point)

    rf_name = connection_config.rf_channel_names[rf_gen_ch]
    rf_channels = _resolve_mapper_channel(mapper, rf_name)
    acquisition_name = connection_config.acquisition_channel_name
    acquisition_channels = _resolve_mapper_channel(mapper, acquisition_name)
    _validate_mapped_hardware_role(
        mapper,
        rf_channels,
        name=rf_name,
        role="RF S-parameter output",
        expected_instruments=("M5300AWG",),
    )
    _validate_mapped_hardware_role(
        mapper,
        acquisition_channels,
        name=acquisition_name,
        role="RF S-parameter acquisition",
        expected_instruments=("M5200Digitizer",),
    )

    sample_rate_hz = _mapped_channel_sample_rate(
        mapper,
        acquisition_channels,
    )
    if sample_rate_hz is None:
        sample_rate_hz = QCS_M5200_SAMPLE_RATE_HZ
    if not np.isclose(
        sample_rate_hz,
        QCS_M5200_SAMPLE_RATE_HZ,
        rtol=0.0,
        atol=1.0,
    ):
        raise QcsUnsupportedFeatureError(
            "QCS RF S-parameter acquisition requires the M5200 "
            f"{QCS_M5200_SAMPLE_RATE_HZ:g} S/s rate; mapper reports "
            f"{sample_rate_hz:g} S/s"
        )
    rendered_samples = requested_duration_s * sample_rate_hz
    sample_count = max(
        QCS_M5200_INTEGRATION_BLOCK_SAMPLES,
        int(
            np.ceil(
                rendered_samples
                / QCS_M5200_INTEGRATION_BLOCK_SAMPLES
                - 1.0e-12
            )
        )
        * QCS_M5200_INTEGRATION_BLOCK_SAMPLES,
    )
    if sample_count > QCS_SPARAMETER_MAX_INTEGRATION_SAMPLES:
        raise QcsUnsupportedFeatureError(
            "QCS RF S-parameter segmented hardware demodulation has been "
            "verified on the connected QCS 2.5.5 hardware through "
            f"{QCS_SPARAMETER_MAX_INTEGRATION_SAMPLES:,} integrated samples "
            f"({QCS_SPARAMETER_MAX_INTEGRATION_DURATION_S * 1.0e6:.9g} us); "
            f"the requested duration quantizes to {sample_count:,} samples."
        )
    quantized_requested_sample_count = sample_count
    quantized_requested_duration_s = sample_count / sample_rate_hz
    if sample_count <= QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES:
        segment_sample_counts = (sample_count,)
    else:
        segment_count = int(
            np.ceil(
                sample_count
                / QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES
            )
        )
        segment_sample_count = int(
            np.ceil(
                sample_count
                / segment_count
                / QCS_M5200_INTEGRATION_BLOCK_SAMPLES
            )
            * QCS_M5200_INTEGRATION_BLOCK_SAMPLES
        )
        # Every segment references the exact same IntegrationFilter object.
        # This avoids allocating a distinct tiny remainder filter and keeps
        # every paired RF/filter layer comfortably within the measured
        # single-filter ceiling. At most (segment_count - 1) 16-sample blocks
        # are added by this equal-segment quantization.
        segment_sample_counts = (segment_sample_count,) * segment_count
    if any(
        count < QCS_M5200_INTEGRATION_BLOCK_SAMPLES
        or count % QCS_M5200_INTEGRATION_BLOCK_SAMPLES
        for count in segment_sample_counts
    ):
        raise RuntimeError(
            "internal QCS S-parameter segmentation did not preserve the "
            "M5200 16-sample IntegrationFilter granularity"
        )
    if any(
        count > QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES
        for count in segment_sample_counts
    ):
        raise RuntimeError(
            "internal QCS S-parameter segment exceeds the connected M5200 "
            "single-IntegrationFilter ceiling"
        )
    if len(segment_sample_counts) > 1:
        rf_absolute_phase = bool(
            getattr(rf_channels, "absolute_phase", False)
        )
        acquisition_absolute_phase = bool(
            getattr(acquisition_channels, "absolute_phase", False)
        )
        if rf_absolute_phase != acquisition_absolute_phase:
            raise QcsUnsupportedFeatureError(
                "Segmented QCS RF S-parameter integration requires the "
                "mapped M5300 RF output and M5200 acquisition virtual "
                "channels to use matching absolute_phase settings; got "
                f"RF={rf_absolute_phase} and acquisition="
                f"{acquisition_absolute_phase}"
            )
    sample_count = int(sum(segment_sample_counts))
    actual_duration_s = sample_count / sample_rate_hz
    rendered_fabric_cycles = actual_duration_s * QCS_FABRIC_CLOCK_HZ
    integer_fabric_cycles = int(round(rendered_fabric_cycles))
    if not np.isclose(
        rendered_fabric_cycles,
        integer_fabric_cycles,
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise QcsUnsupportedFeatureError(
            "QCS S-parameter integration duration cannot be aligned to both "
            "the M5200 16-sample block and the 300 MHz fabric clock"
        )

    frequency = qcs.Scalar(
        "sparameter_frequency_hz",
        value=float(frequencies[0]),
        dtype=float,
    )
    frequency_values = qcs.Array(
        "sparameter_frequency_values_hz",
        value=frequencies.copy(),
        dtype=float,
    )
    amplitude_variable = None
    waveform_amplitude: Any = float(amplitudes[0])
    amplitude_values = None
    if amplitude_varies:
        amplitude_variable = qcs.Scalar(
            "sparameter_relative_amplitude",
            value=float(amplitudes[0]),
            dtype=float,
        )
        amplitude_values = qcs.Array(
            "sparameter_relative_amplitude_values",
            value=amplitudes.copy(),
            dtype=float,
        )
        waveform_amplitude = amplitude_variable
    program = qcs.Program(name="PulseGenerator QCS RF S-parameter sweep")
    integration_filter_cache: dict[int, Any] = {}
    last_segment_index = len(segment_sample_counts) - 1
    for segment_index, segment_samples in enumerate(segment_sample_counts):
        segment_duration_s = segment_samples / sample_rate_hz
        # M5200 needs 10 ns between acquisitions on one virtual channel. Put
        # that gap at the tail of the RF waveform in the same layer, so the
        # next layer starts its M5300 output and IntegrationFilter together.
        output_duration_s = segment_duration_s
        if segment_index != last_segment_index:
            output_duration_s += QCS_SPARAMETER_INTER_SEGMENT_DELAY_S
        output_waveform = qcs.RFWaveform(
            duration=output_duration_s,
            envelope=qcs.ConstantEnvelope(),
            amplitude=waveform_amplitude,
            rf_frequency=frequency,
            instantaneous_phase=phase,
            name=f"sparameter_output_segment_{segment_index}",
        )
        integration_filter = integration_filter_cache.get(segment_samples)
        if integration_filter is None:
            filter_waveform = qcs.RFWaveform(
                duration=segment_duration_s,
                envelope=qcs.ConstantEnvelope(),
                amplitude=1.0,
                rf_frequency=frequency,
                instantaneous_phase=phase,
                name=(
                    "sparameter_integration_filter_"
                    f"{segment_samples}_samples"
                ),
            )
            integration_filter = qcs.IntegrationFilter(filter_waveform)
            integration_filter_cache[segment_samples] = integration_filter
        program.add_waveform(
            output_waveform,
            rf_channels,
            new_layer=segment_index == 0,
        )
        acquisition_options = {}
        if segment_index > 0:
            acquisition_options["pre_delay"] = (
                QCS_SPARAMETER_INTER_SEGMENT_DELAY_S
            )
        program.add_acquisition(
            integration_filter=integration_filter,
            channels=acquisition_channels,
            new_layer=False,
            **acquisition_options,
        )
    program.n_shots(repetitions)
    if amplitude_varies:
        # The frequency and calibrated connector amplitude are a zipped
        # one-dimensional sweep, not a Cartesian product.  This preserves one
        # requested output-power point for every requested frequency.
        program.sweep(
            [frequency_values, amplitude_values],
            [frequency, amplitude_variable],
        )
    else:
        program.sweep(frequency_values, frequency)
    return QcsCompiledSParameterSweep(
        program=program,
        rf_channels=rf_channels,
        acquisition_channels=acquisition_channels,
        frequency_variable=frequency,
        amplitude_variable=amplitude_variable,
        frequencies_hz=frequencies.copy(),
        rf_amplitudes=amplitudes.copy(),
        requested_integration_duration_s=requested_duration_s,
        quantized_requested_integration_duration_s=(
            quantized_requested_duration_s
        ),
        quantized_requested_integration_sample_count=(
            quantized_requested_sample_count
        ),
        integration_duration_s=actual_duration_s,
        integration_sample_count=sample_count,
        integration_segment_sample_counts=tuple(segment_sample_counts),
        inter_segment_delay_s=QCS_SPARAMETER_INTER_SEGMENT_DELAY_S,
        acquisition_sample_rate_hz=sample_rate_hz,
        repetitions_per_point=repetitions,
    )


def _qcs_sparameter_complex_values(values: Any) -> np.ndarray:
    """Return QCS I/Q values as one finite complex ndarray."""

    if isinstance(values, (tuple, list)) and len(values) == 2:
        i_values = np.asarray(values[0])
        q_values = np.asarray(values[1])
        if i_values.shape != q_values.shape:
            raise ValueError("QCS S-parameter I and Q arrays have different shapes")
        array = i_values.astype(float) + 1j * q_values.astype(float)
    else:
        array = np.asarray(values)
        if (
            not np.iscomplexobj(array)
            and array.ndim >= 1
            and array.shape[-1] == 2
        ):
            array = (
                array[..., 0].astype(float)
                + 1j * array[..., 1].astype(float)
            )
    if array.size == 0:
        raise ValueError("QCS S-parameter acquisition returned no I/Q values")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("QCS S-parameter I/Q values must be numeric")
    array = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError("QCS S-parameter I/Q values must be finite")
    return array


def _extract_qcs_sparameter_iq(
    raw_result: Any,
    compiled: QcsCompiledSParameterSweep,
) -> np.ndarray:
    """Decode and sample-weight segmented hardware-demodulated I/Q."""

    segment_samples = tuple(compiled.integration_segment_sample_counts)
    if len(segment_samples) == 1:
        values = extract_qcs_acquisition(
            raw_result,
            compiled.acquisition_channels,
        )
        return normalize_qcs_hardware_sweep_iq(
            values,
            repetitions_per_point=compiled.repetitions_per_point,
            sweep_shape=(int(compiled.frequencies_hz.size),),
            hardware_sweep=False,
        )

    results = getattr(raw_result, "results", None)
    get_iq = None if results is None else getattr(results, "get_iq", None)
    if callable(get_iq):
        normalized_segments = []
        for segment_index in range(len(segment_samples)):
            values = _first_result_value(
                get_iq(
                    compiled.acquisition_channels,
                    avg=False,
                    acq_index=segment_index,
                ),
                compiled.acquisition_channels,
            )
            normalized_segments.append(
                normalize_qcs_hardware_sweep_iq(
                    values,
                    repetitions_per_point=compiled.repetitions_per_point,
                    sweep_shape=(int(compiled.frequencies_hz.size),),
                    hardware_sweep=False,
                )
            )
        segmented_iq = np.stack(normalized_segments, axis=0)
        weights = np.asarray(segment_samples, dtype=np.float64)
        weights /= float(np.sum(weights))
        return np.sum(
            segmented_iq * weights[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
            axis=0,
        )

    # Injected executors without the real QCS Results API may provide all
    # acquisitions as one frequency/shot/segment array. Keep that test seam
    # explicit; production results are decoded one acquisition index at a
    # time above, so no undocumented QCS acquisition-axis order is assumed.
    try:
        values = raw_result[compiled.acquisition_channels]
    except (IndexError, KeyError, TypeError):
        values = raw_result
    array = _qcs_sparameter_complex_values(values)
    frequency_count = int(compiled.frequencies_hz.size)
    repetitions = int(compiled.repetitions_per_point)
    segment_count = len(segment_samples)
    expected_shape = (frequency_count, repetitions, segment_count)
    if array.shape != expected_shape:
        raise ValueError(
            "Injected QCS segmented S-parameter I/Q must have shape "
            f"{expected_shape}; received {array.shape}"
        )
    weights = np.asarray(segment_samples, dtype=np.float64)
    weights /= float(np.sum(weights))
    averaged = np.sum(array * weights[np.newaxis, np.newaxis, :], axis=-1)
    return normalize_qcs_hardware_sweep_iq(
        averaged,
        repetitions_per_point=repetitions,
        sweep_shape=(frequency_count,),
        hardware_sweep=False,
    )


def _extract_qcs_segmented_hardware_sweep_iq(
    raw_result: Any,
    *,
    acquisition_channels: Any,
    segment_sample_counts: Sequence[int],
    repetitions_per_point: int,
    sweep_shape: Sequence[int],
    hardware_sweep: bool = True,
) -> np.ndarray:
    """Decode repeated IntegrationFilters and return one weighted I/Q value.

    The real QCS Results API exposes each acquisition by ``acq_index``.  The
    injected-array branch is deliberately explicit so tests do not imply an
    undocumented production acquisition-axis order.
    """

    segment_samples = tuple(int(value) for value in segment_sample_counts)
    if not segment_samples:
        segment_samples = (1,)
    if len(segment_samples) == 1:
        values = extract_qcs_acquisition(raw_result, acquisition_channels)
        return normalize_qcs_hardware_sweep_iq(
            values,
            repetitions_per_point=repetitions_per_point,
            sweep_shape=sweep_shape,
            hardware_sweep=hardware_sweep,
        )

    results = getattr(raw_result, "results", None)
    get_iq = None if results is None else getattr(results, "get_iq", None)
    if callable(get_iq):
        normalized_segments = []
        for segment_index in range(len(segment_samples)):
            values = _first_result_value(
                get_iq(
                    acquisition_channels,
                    avg=False,
                    acq_index=segment_index,
                ),
                acquisition_channels,
            )
            normalized_segments.append(
                normalize_qcs_hardware_sweep_iq(
                    values,
                    repetitions_per_point=repetitions_per_point,
                    sweep_shape=sweep_shape,
                    hardware_sweep=hardware_sweep,
                )
            )
        segmented_iq = np.stack(normalized_segments, axis=0)
        weights = np.asarray(segment_samples, dtype=np.float64)
        weights /= float(np.sum(weights))
        return np.sum(
            segmented_iq
            * weights[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
            axis=0,
        )

    try:
        values = raw_result[acquisition_channels]
    except (IndexError, KeyError, TypeError):
        values = raw_result
    array = _qcs_sparameter_complex_values(values)
    repetitions = int(repetitions_per_point)
    shape = tuple(int(value) for value in sweep_shape)
    point_count = int(np.prod(shape))
    segment_count = len(segment_samples)
    weights = np.asarray(segment_samples, dtype=np.float64)
    weights /= float(np.sum(weights))
    point_major_shape = (point_count, repetitions, segment_count)
    repetition_major_shape = (repetitions, *shape, segment_count)
    flattened_repetition_major_shape = (
        repetitions,
        point_count,
        segment_count,
    )
    if array.shape == point_major_shape:
        averaged = np.sum(
            array * weights[np.newaxis, np.newaxis, :],
            axis=-1,
        )
        return normalize_qcs_hardware_sweep_iq(
            averaged,
            repetitions_per_point=repetitions,
            sweep_shape=shape,
            hardware_sweep=hardware_sweep,
        )
    if array.shape in (
        repetition_major_shape,
        flattened_repetition_major_shape,
    ):
        averaged = np.sum(
            array * weights.reshape((1,) * (array.ndim - 1) + (-1,)),
            axis=-1,
        )
        return normalize_qcs_hardware_sweep_iq(
            averaged,
            repetitions_per_point=repetitions,
            sweep_shape=shape,
            hardware_sweep=hardware_sweep,
        )
    raise ValueError(
        "Injected QCS segmented Stability I/Q must have shape "
        f"{point_major_shape}, {repetition_major_shape}, or "
        f"{flattened_repetition_major_shape}; received {array.shape}"
    )


def _extract_qcs_compiled_point_iq(
    raw_result: Any,
    compiled: QcsCompiledPoint,
    *,
    repetitions_per_sweep: int,
    hardware_demodulation: bool,
) -> np.ndarray:
    """Decode one fixed point, including repeated integration filters."""

    if not hardware_demodulation:
        values = extract_qcs_acquisition(
            raw_result,
            compiled.acquisition_channels,
            prefer_trace=True,
        )
        return normalize_qcs_iq(
            values,
            repetitions_per_sweep=repetitions_per_sweep,
            real_is_i_trace=True,
        )
    segmented = _extract_qcs_segmented_hardware_sweep_iq(
        raw_result,
        acquisition_channels=compiled.acquisition_channels,
        segment_sample_counts=(
            compiled.integration_segment_sample_counts
            or (int(compiled.acquisition_sample_count or 1),)
        ),
        repetitions_per_point=repetitions_per_sweep,
        sweep_shape=(1,),
        hardware_sweep=False,
    )
    return segmented[0]


def execute_qcs_sparameter_sweep(
    *,
    connection_config: QcsConnectionConfig,
    frequencies_hz: Sequence[float],
    rf_gen_ch: int,
    rf_amplitude: float | Sequence[float],
    integration_duration_s: float,
    repetitions_per_point: int = 1,
    phase_rad: float = 0.0,
    calibrated_output: bool = False,
    output_power_calibration: Optional[Mapping[str, Any]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    partial_callback: Optional[PartialResultCallback] = None,
    qcs_module=None,
    mapper=None,
    executor=None,
    compiled: Optional[QcsCompiledSParameterSweep] = None,
    cancellation: Optional[QcsCancellationController] = None,
) -> QcsSParameterExecutionResult:
    """Execute one coherent QCS S-parameter sweep without persistence."""

    qcs = _import_qcs() if qcs_module is None else qcs_module
    averaging_plan = plan_qcs_total_iq_averaging(integration_duration_s)
    pass_sample_count = averaging_plan.pass_sample_counts[0]
    pass_duration_s = pass_sample_count / averaging_plan.sample_rate_hz
    if progress_callback is not None:
        progress_callback(0, "Validating QCS RF S-parameter sweep")
    if mapper is None:
        mapper = load_qcs_channel_mapper(connection_config, qcs_module=qcs)
    if cancellation is not None:
        cancellation.bind(qcs, mapper)
        cancellation.raise_if_requested("QCS RF S-parameter setup")
    if compiled is None:
        if progress_callback is not None:
            progress_callback(10, "Compiling one QCS frequency sweep Program")
        compiled = compile_qcs_sparameter_sweep(
            connection_config=connection_config,
            mapper=mapper,
            frequencies_hz=frequencies_hz,
            rf_gen_ch=rf_gen_ch,
            rf_amplitude=rf_amplitude,
            integration_duration_s=pass_duration_s,
            repetitions_per_point=repetitions_per_point,
            phase_rad=phase_rad,
            calibrated_output=calibrated_output,
            qcs_module=qcs,
        )
    elif int(compiled.integration_sample_count) != pass_sample_count:
        raise ValueError(
            "precompiled QCS S-parameter Program integration does not match "
            "the planned per-pass duration"
        )
    if executor is None:
        executor = build_qcs_executor(
            connection_config,
            mapper,
            qcs_module=qcs,
        )
    if progress_callback is not None:
        progress_callback(
            35,
            "Executing one QCS-resolved frequency sweep for "
            f"{compiled.frequencies_hz.size:,} frequencies",
        )
    if cancellation is not None:
        cancellation.tag_program(compiled.program)
    iq_weighted_sum = None
    completed_passes = 0
    raw_result = None
    latest_partial = None
    try:
        for pass_index, pass_samples in enumerate(
            averaging_plan.pass_sample_counts
        ):
            if cancellation is not None:
                cancellation.raise_if_requested(
                    "QCS RF S-parameter averaging pass"
                )
            raw_result = _executor_execute(executor, compiled.program)
            pass_iq = _extract_qcs_sparameter_iq(raw_result, compiled)
            weighted = np.asarray(pass_iq) * float(
                pass_samples
            )
            if iq_weighted_sum is None:
                iq_weighted_sum = weighted
            else:
                iq_weighted_sum += weighted
            completed_passes += 1
            completed_samples = int(
                sum(averaging_plan.pass_sample_counts[:completed_passes])
            )
            current_iq = iq_weighted_sum / float(completed_samples)
            latest_partial = QcsSParameterExecutionResult(
                frequencies_hz=compiled.frequencies_hz.copy(),
                iq=current_iq.copy(),
                program=compiled.program,
                raw_result=raw_result,
                program_summary={
                    "backend": "qcs",
                    "measurement": "rf_s_parameter",
                    "partial": completed_passes < averaging_plan.pass_count,
                    "iq_averaging_pass_count": averaging_plan.pass_count,
                    "completed_iq_averaging_passes": completed_passes,
                    "requested_total_integration_duration_s": (
                        averaging_plan.requested_total_duration_s
                    ),
                    "effective_integration_duration_s": (
                        completed_samples / averaging_plan.sample_rate_hz
                    ),
                    "iq_shape": list(current_iq.shape),
                },
                rf_settings={"backend": "qcs"},
            )
            if partial_callback is not None:
                partial_callback(latest_partial)
            if progress_callback is not None:
                progress_callback(
                    35 + int(35 * completed_passes / averaging_plan.pass_count),
                    "QCS integrated-I/Q pass "
                    f"{completed_passes:,}/{averaging_plan.pass_count:,} acquired",
                )
            if cancellation is not None:
                cancellation.raise_if_requested(
                    "QCS RF S-parameter averaging boundary"
                )
    except BaseException as execution_error:
        if cancellation is not None and cancellation.is_stop_requested():
            raise QcsExperimentCancelled(
                "QCS RF S-parameter sweep stopped by user; completed "
                "averaging passes remain available",
                partial_result=latest_partial,
            ) from execution_error
        raise
    if iq_weighted_sum is None or raw_result is None:
        raise RuntimeError("QCS S-parameter averaging completed no passes")
    iq = iq_weighted_sum / float(
        sum(averaging_plan.pass_sample_counts[:completed_passes])
    )
    if cancellation is not None and not cancellation.close_stop_window():
        raise QcsExperimentCancelled(
            "QCS RF S-parameter sweep stopped after a completed averaging pass",
            partial_result=latest_partial,
        )
    if progress_callback is not None:
        progress_callback(70, "QCS integrated I/Q frequency sweep acquired")
    summary = {
        "backend": "qcs",
        "measurement": "rf_s_parameter",
        "hardware_sweep": False,
        "qcs_software_resolved_sweep": True,
        "hardware_sweep_dimensions": 0,
        "hardware_sweep_shape": [int(compiled.frequencies_hz.size)],
        "hardware_sweep_points": int(compiled.frequencies_hz.size),
        "program_count": 1,
        "executor_call_count": completed_passes,
        "repetitions_per_point": int(compiled.repetitions_per_point),
        "frequency_scalar_shared_with_integration_filter": True,
        "rf_amplitude_frequency_dependent": bool(
            compiled.amplitude_variable is not None
        ),
        "rf_relative_amplitudes": compiled.rf_amplitudes.tolist(),
        "software_sweep_reason": (
            "QCS 2.5.5 cannot modify an M5200 IntegrationFilter frequency "
            "inside hardware time"
        ),
        "requested_integration_duration_s": (
            averaging_plan.requested_total_duration_s
        ),
        "quantized_requested_integration_duration_s": (
            averaging_plan.quantized_total_duration_s
        ),
        "quantized_requested_integration_sample_count": (
            averaging_plan.quantized_total_sample_count
        ),
        "integration_duration_s": averaging_plan.quantized_total_duration_s,
        "integration_sample_count": averaging_plan.quantized_total_sample_count,
        "iq_averaging_pass_count": averaging_plan.pass_count,
        "completed_iq_averaging_passes": completed_passes,
        "per_pass_integration_duration_s": compiled.integration_duration_s,
        "per_pass_integration_sample_count": compiled.integration_sample_count,
        "effective_integration_duration_s": (
            sum(averaging_plan.pass_sample_counts[:completed_passes])
            / averaging_plan.sample_rate_hz
        ),
        "integration_segment_sample_counts": list(
            compiled.integration_segment_sample_counts
        ),
        "integration_segment_count": len(
            compiled.integration_segment_sample_counts
        ),
        "inter_segment_delay_s": compiled.inter_segment_delay_s,
        "total_inter_segment_dead_time_s": (
            max(0, len(compiled.integration_segment_sample_counts) - 1)
            * compiled.inter_segment_delay_s
        ),
        "hardware_demodulated_iq_reduction": (
            "sample_count_weighted_complex_mean"
            if len(compiled.integration_segment_sample_counts) > 1
            else "single_integration_filter"
        ),
        "sample_rate_hz": compiled.acquisition_sample_rate_hz,
        "acquisition_result_type": "integrated_iq",
        "reset_phase_every_shot": True,
        "rf_generator_number": int(rf_gen_ch),
        "rf_channel_name": connection_config.rf_channel_names[int(rf_gen_ch)],
        "acquisition_channel_name": (
            connection_config.acquisition_channel_name
        ),
        "iq_shape": list(iq.shape),
    }
    if output_power_calibration is not None:
        summary["output_power_calibration"] = dict(output_power_calibration)
    rf_settings = {
        "backend": "qcs",
        "output": {
            "gen_ch": int(rf_gen_ch),
            "virtual_channel": connection_config.rf_channel_names[int(rf_gen_ch)],
            "amplitude": (
                float(compiled.rf_amplitudes[0])
                if compiled.amplitude_variable is None
                else None
            ),
            "relative_amplitudes": compiled.rf_amplitudes.tolist(),
            "amplitude_frequency_dependent": bool(
                compiled.amplitude_variable is not None
            ),
            "phase_rad": float(phase_rad),
        },
        "readout": {
            "virtual_channel": connection_config.acquisition_channel_name,
            "hw_demod": True,
            "requested_integration_duration_s": (
                averaging_plan.requested_total_duration_s
            ),
            "quantized_requested_integration_duration_s": (
                averaging_plan.quantized_total_duration_s
            ),
            "quantized_requested_integration_sample_count": (
                averaging_plan.quantized_total_sample_count
            ),
            "integration_duration_s": averaging_plan.quantized_total_duration_s,
            "integration_sample_count": averaging_plan.quantized_total_sample_count,
            "iq_averaging_pass_count": averaging_plan.pass_count,
            "completed_iq_averaging_passes": completed_passes,
            "per_pass_integration_duration_s": compiled.integration_duration_s,
            "per_pass_integration_sample_count": compiled.integration_sample_count,
            "integration_segment_sample_counts": list(
                compiled.integration_segment_sample_counts
            ),
            "integration_segment_count": len(
                compiled.integration_segment_sample_counts
            ),
            "inter_segment_delay_s": compiled.inter_segment_delay_s,
            "total_inter_segment_dead_time_s": (
                max(0, len(compiled.integration_segment_sample_counts) - 1)
                * compiled.inter_segment_delay_s
            ),
            "sample_rate_hz": compiled.acquisition_sample_rate_hz,
            "frequency_tracks_output": True,
        },
    }
    if output_power_calibration is not None:
        rf_settings["output"]["power_calibration"] = dict(
            output_power_calibration
        )
    return QcsSParameterExecutionResult(
        frequencies_hz=compiled.frequencies_hz.copy(),
        iq=iq,
        program=compiled.program,
        raw_result=raw_result,
        program_summary=summary,
        rf_settings=rf_settings,
    )


def _apply_qcs_sparameter_power_reference(
    execution: QcsSParameterExecutionResult,
    *,
    calibration: Any,
    rf_amplitudes: float | Sequence[float],
    sweep_config: Any,
) -> QcsSParameterExecutionResult:
    """Convert QCS receiver voltage into calibrated, thru-normalized S21.

    The QCS M5300A power calibration is acquired through an M5200A loopback.
    It therefore supplies both the programmed source-power reference and the
    M5200 voltage-to-power conversion.  For each frequency this derives::

        P_DUT_IN  = P_M5300_REFERENCE - LOSS1
        P_DUT_OUT = P_M5200 + LOSS2 - AMPLIFIER_GAIN
        |S21|_dB  = P_DUT_OUT - P_DUT_IN

    Raw I/Q remains untouched so the uncalibrated receiver magnitude is still
    available as ``adc_magnitude_db`` in the shared result model.
    """

    frequencies_hz = np.asarray(
        execution.frequencies_hz,
        dtype=float,
    ).reshape(-1)
    iq = np.asarray(execution.iq)
    if iq.ndim != 4 or iq.shape[0] != frequencies_hz.size or iq.shape[-2:] != (
        1,
        2,
    ):
        raise ValueError(
            "QCS S-parameter power conversion requires I/Q with shape "
            "(frequency, repetition, 1, 2)"
        )
    mean_iq = np.mean(
        iq[:, :, 0, 0].astype(np.float64)
        + 1j * iq[:, :, 0, 1].astype(np.float64),
        axis=1,
    )
    receiver_magnitude = np.abs(mean_iq)
    if np.any(~np.isfinite(receiver_magnitude)):
        raise ValueError(
            "QCS S-parameter receiver magnitude must be finite"
        )
    receiver_magnitude = np.maximum(
        receiver_magnitude,
        np.finfo(np.float64).tiny,
    )
    receiver_connector_power_dbm = np.asarray(
        calibration.input_reference.output_power_dbm(receiver_magnitude),
        dtype=float,
    ).reshape(-1)

    amplitudes = np.asarray(rf_amplitudes, dtype=float).reshape(-1)
    if amplitudes.size == 1:
        amplitudes = np.full(frequencies_hz.shape, float(amplitudes[0]))
    if amplitudes.size != frequencies_hz.size:
        raise ValueError(
            "QCS calibrated RF amplitudes must contain one value per frequency"
        )
    amplitudes = np.abs(amplitudes)
    if np.any(~np.isfinite(amplitudes)) or np.any(amplitudes <= 0.0):
        raise ValueError(
            "QCS calibrated RF amplitudes must be finite and nonzero"
        )
    source_connector_power_dbm = np.asarray(
        calibration.full_scale_power_dbm(frequencies_hz),
        dtype=float,
    ).reshape(-1) + 20.0 * np.log10(amplitudes)

    def correction(name: str) -> float:
        value = float(getattr(sweep_config, name, 0.0))
        if not isfinite(value):
            raise ValueError(f"QCS S-parameter {name} must be finite")
        return value

    loss1_db = correction("loss1_db")
    loss2_db = correction("loss2_db")
    amplifier_gain_db = correction("amplifier_gain_db")
    dut_input_powers_dbm = source_connector_power_dbm - loss1_db
    dut_output_powers_dbm = (
        receiver_connector_power_dbm + loss2_db - amplifier_gain_db
    )
    calculation = {
        "quantity": "loopback_normalized_s21",
        "formula_db": "P_DUT_OUT_dBm - P_DUT_IN_dBm",
        "source_reference": "M5300A/M5200A RF power calibration thru",
        "receiver_conversion": (
            "calibration.input_reference.output_power_dbm(abs(mean(I+jQ)))"
        ),
        "loss1_db": loss1_db,
        "loss2_db": loss2_db,
        "amplifier_gain_db": amplifier_gain_db,
    }
    summary = dict(execution.program_summary)
    summary["s21_calculation"] = calculation
    rf_settings = dict(execution.rf_settings)
    rf_settings["s21"] = calculation
    return replace(
        execution,
        program_summary=summary,
        rf_settings=rf_settings,
        dut_input_powers_dbm=np.ascontiguousarray(dut_input_powers_dbm),
        dut_output_powers_dbm=np.ascontiguousarray(dut_output_powers_dbm),
    )


def run_qcs_sparameter_sweep(
    *,
    connection_config: QcsConnectionConfig,
    run_config: QcodesRunConfig,
    sweep_config: Any,
    rf_gen_ch: int,
    rf_amplitude: float | Sequence[float],
    repetitions_per_point: int = 1,
    progress_callback: Optional[ProgressCallback] = None,
    partial_callback: Optional[PartialResultCallback] = None,
    qcs_module=None,
    mapper=None,
    executor=None,
    cancellation: Optional[QcsCancellationController] = None,
) -> Any:
    """Execute and store a QCS sweep using the existing S-parameter schema."""

    if bool(getattr(sweep_config, "power_sweep_enabled", False)):
        raise QcsUnsupportedFeatureError(
            "QCS RF S-parameter currently supports one fixed RF amplitude; "
            "disable the QICK software power sweep"
        )
    requested_mhz = np.asarray(
        getattr(sweep_config, "requested_frequencies_mhz"),
        dtype=float,
    )
    requested_hz = requested_mhz * 1.0e6
    calibration_enabled = bool(
        getattr(sweep_config, "power_calibration_enabled", False)
    )
    calibration = None
    output_power_calibration = None
    calibrated_amplitudes: float | Sequence[float] = rf_amplitude
    if calibration_enabled:
        try:
            from .qcs_rf_power_calibration import (
                load_m5300_power_calibration,
                resolve_m5300_m5200_identities,
            )
        except ImportError:
            from qcs_rf_power_calibration import (
                load_m5300_power_calibration,
                resolve_m5300_m5200_identities,
            )

        qcs = _import_qcs() if qcs_module is None else qcs_module
        if mapper is None:
            mapper = load_qcs_channel_mapper(
                connection_config,
                qcs_module=qcs,
            )
        rf_name = connection_config.rf_channel_names[int(rf_gen_ch)]
        acquisition_name = connection_config.acquisition_channel_name
        if acquisition_name is None:
            raise ValueError(
                "QCS calibrated RF power requires a mapped M5200 input"
            )
        (
            output_identity,
            input_identity,
            expected_lo_frequency_hz,
        ) = resolve_m5300_m5200_identities(
            mapper,
            rf_name,
            acquisition_name,
        )
        requested_run_id = getattr(
            sweep_config,
            "qcs_power_calibration_run_id",
            None,
        )
        if requested_run_id is None:
            requested_run_id = getattr(
                sweep_config,
                "qcs_output_power_calibration_run_id",
                None,
            )
        if requested_run_id is None:
            # The shared S-parameter GUI stores this selector under its
            # legacy QICK field name. In the QCS backend it selects the
            # dedicated M5300A/M5200A calibration run instead.
            requested_run_id = getattr(
                sweep_config,
                "requested_input_calibration_run_id",
                None,
            )
        if requested_run_id is not None:
            requested_run_id = int(requested_run_id)
            if requested_run_id <= 0:
                requested_run_id = None
        calibration = load_m5300_power_calibration(
            getattr(sweep_config, "calibration_database_path"),
            run_id=requested_run_id,
            expected_output=output_identity,
            expected_input=input_identity,
            expected_mapper_sha256=None,
            expected_lo_frequency_hz=expected_lo_frequency_hz,
            required_frequencies_hz=requested_hz,
            termination_ohm=50.0,
        )
        target_power_dbm = float(
            getattr(sweep_config, "output_power_dbm")
        )
        calibrated_amplitudes = calibration.relative_amplitudes_for_power(
            requested_hz,
            target_power_dbm,
            allow_power_extrapolation=False,
        )
        output_power_calibration = dict(calibration.provenance)
        output_power_calibration.update(
            {
                "target_power_dbm": target_power_dbm,
                "full_scale_power_dbm": calibration.full_scale_power_dbm(
                    requested_hz
                ).tolist(),
                "relative_amplitudes": np.asarray(
                    calibrated_amplitudes,
                    dtype=float,
                ).tolist(),
                "power_extrapolation": False,
                "s21_reference": "M5300A/M5200A calibration thru",
            }
        )

    def with_power_reference(
        current: QcsSParameterExecutionResult,
    ) -> QcsSParameterExecutionResult:
        if calibration is None:
            return current
        return _apply_qcs_sparameter_power_reference(
            current,
            calibration=calibration,
            rf_amplitudes=calibrated_amplitudes,
            sweep_config=sweep_config,
        )

    def publish_partial(current: QcsSParameterExecutionResult) -> None:
        if partial_callback is not None:
            partial_callback(with_power_reference(current))

    try:
        execution = execute_qcs_sparameter_sweep(
            connection_config=connection_config,
            frequencies_hz=requested_hz,
            rf_gen_ch=rf_gen_ch,
            rf_amplitude=calibrated_amplitudes,
            integration_duration_s=(
                float(getattr(sweep_config, "scan_time_us")) * 1.0e-6
            ),
            repetitions_per_point=repetitions_per_point,
            calibrated_output=calibration_enabled,
            output_power_calibration=output_power_calibration,
            progress_callback=progress_callback,
            partial_callback=(
                publish_partial if partial_callback is not None else None
            ),
            qcs_module=(qcs if calibration_enabled else qcs_module),
            mapper=mapper,
            executor=executor,
            cancellation=cancellation,
        )
    except QcsExperimentCancelled as exc:
        partial = getattr(exc, "partial_result", None)
        if partial is not None:
            partial = with_power_reference(partial)
        raise QcsExperimentCancelled(
            str(exc),
            partial_result=partial,
            stored_result=getattr(exc, "stored_result", None),
        ) from exc
    execution = with_power_reference(execution)
    try:
        from .qick_sparameter_sweep import (
            SParameterSweepResult,
            StoredSParameterSweep,
            store_sparameter_result,
        )
    except ImportError:
        from qick_sparameter_sweep import (
            SParameterSweepResult,
            StoredSParameterSweep,
            store_sparameter_result,
        )

    # Each hardware-demodulated shot is one independent complex sample for
    # the existing SParameterSweepResult averaging/plotting contract.
    iq_traces = execution.iq[:, :, 0, :]
    result = SParameterSweepResult.from_iq(
        requested_mhz,
        execution.frequencies_hz / 1.0e6,
        iq_traces,
        sample_rate_hz=float(
            execution.program_summary["sample_rate_hz"]
        ),
        actual_output_powers_dbm=execution.dut_input_powers_dbm,
        input_powers_dbm=execution.dut_output_powers_dbm,
        s21_reference=(
            "qcs_m5300_m5200_calibration_thru"
            if execution.dut_input_powers_dbm is not None
            else None
        ),
    )
    if progress_callback is not None:
        progress_callback(75, "Writing QCS RF S-parameter result to QCoDeS")
    dataset, row_count = store_sparameter_result(
        result,
        config=sweep_config,
        connection_config=connection_config,
        run_config=run_config,
        program_summary=execution.program_summary,
        rf_settings=execution.rf_settings,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(100, "QCS RF S-parameter sweep saved")
    database_path = Path(
        getattr(run_config, "resolved_database_path", run_config.database_path)
    ).expanduser().resolve()
    return StoredSParameterSweep(
        run_id=int(dataset.run_id),
        guid=str(dataset.guid),
        database_path=database_path,
        row_count=row_count,
        result=result,
        dataset=dataset,
        program=execution.program,
        rf_settings=execution.rf_settings,
    )


def compile_qcs_stability_hardware_sweep(
    sequence: Any,
    *,
    connection_config: QcsConnectionConfig,
    mapper: Any,
    repetitions_per_point: int,
    fabric_mhz: float = 300.0,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
    qcs_module=None,
) -> QcsCompiledHardwareSweep:
    """Compile a two-axis Stability Diagram into one native QCS sweep.

    Following ``QTTVideoMode/QCSVideoProcessor.py``, Python computes each
    physical M5301 channel's full Cartesian voltage array. One direct Scalar
    per physical output is swept simultaneously, then ``n_shots`` prepends
    the shot axis. Y remains the fastest semantic Cartesian axis.
    """
    qcs = _import_qcs() if qcs_module is None else qcs_module
    if acquisition is None:
        raise ValueError(
            "QCS Stability hardware sweep requires an acquisition"
        )
    if not connection_config.hw_demod:
        raise QcsUnsupportedFeatureError(
            "QCS Stability hardware sweep requires hardware demodulation; "
            "QCS 2.5.5 cannot return raw traces from a hardware sweep"
        )
    if not connection_config.blocking:
        raise QcsUnsupportedFeatureError(
            "QCS Stability hardware sweep requires blocking=True"
        )
    compensation_config = getattr(sequence, "bias_t_compensation", None)
    if compensation_config is not None:
        if not isinstance(compensation_config, BiasTCompensationConfig):
            raise QcsUnsupportedFeatureError(
                "QCS Stability hardware sweep supports only DC Bias-T "
                "compensation; filter compensation is not supported"
            )
        if compensation_config.mode != "fixed_time":
            raise QcsUnsupportedFeatureError(
                "QCS Stability hardware sweep supports only fixed-time DC "
                "Bias-T compensation; select Fixed time (adjust voltage)"
            )

    if isinstance(repetitions_per_point, bool):
        raise TypeError("repetitions_per_point must be an integer")
    repetitions = int(repetitions_per_point)
    if repetitions < 1 or repetitions != repetitions_per_point:
        raise ValueError("repetitions_per_point must be a positive integer")

    axes = tuple(sequence.sweep_axes)
    if len(axes) != 2 or any(
        getattr(axis, "axis_kind", "amplitude") != "amplitude"
        for axis in axes
    ):
        raise QcsUnsupportedFeatureError(
            "QCS Stability requires exactly two DC-amplitude sweep axes"
        )
    x_axis, y_axis = axes
    x_values = np.asarray(x_axis.points, dtype=float)
    y_values = np.asarray(y_axis.points, dtype=float)
    sweep_shape = (x_values.size, y_values.size)
    if tuple(sequence.sweep_shape) != sweep_shape:
        raise ValueError(
            "QCS Stability sequence sweep shape does not match its axes"
        )
    point_count = int(x_values.size) * int(y_values.size)
    arrays_per_dc_output = 2 if compensation_config is not None else 1
    max_grid_points = (
        MAX_QCS_HARDWARE_SWEEP_ARRAY_VALUES - 1
    ) // arrays_per_dc_output
    if point_count > max_grid_points:
        raise QcsUnsupportedFeatureError(
            "QCS Stability grid contains "
            f"{point_count:,} Cartesian points; each physical M5301 "
            f"output uses {arrays_per_dc_output} hardware-sweep amplitude "
            f"array{'s' if arrays_per_dc_output != 1 else ''}, allowing at "
            f"most {max_grid_points:,} points"
        )
    if len(connection_config.dc_channel_names) != int(sequence.n_outputs):
        raise ValueError(
            "QCS DC channel count must match the Stability output count"
        )
    if connection_config.acquisition_channel_name is None:
        raise ValueError(
            "a QCS acquisition virtual-channel name is required"
        )
    rf_pulses = resolve_qcs_rf_power_calibrations(
        connection_config=connection_config,
        mapper=mapper,
        rf_pulses=rf_pulses,
    )
    dc_channels = []
    for name in connection_config.dc_channel_names:
        channel = _resolve_mapper_channel(mapper, name)
        _validate_mapped_hardware_role(
            mapper,
            channel,
            name=name,
            role="DC",
            expected_instruments=("M5301AWG",),
            require_relative_phase=True,
        )
        dc_channels.append(channel)
    rf_channels_by_gen_ch = {}
    for pulse in rf_pulses:
        if pulse.gen_ch not in connection_config.rf_channel_names:
            raise KeyError(
                f"no QCS virtual RF channel is mapped for gen_ch "
                f"{pulse.gen_ch}"
            )
        rf_channel = _resolve_mapper_channel(
            mapper, connection_config.rf_channel_names[pulse.gen_ch]
        )
        _validate_mapped_hardware_role(
            mapper,
            rf_channel,
            name=connection_config.rf_channel_names[pulse.gen_ch],
            role="RF",
            expected_instruments=("M5300AWG", "M5301AWG"),
        )
        rf_channels_by_gen_ch[int(pulse.gen_ch)] = rf_channel
    acquisition_channels = _resolve_mapper_channel(
        mapper, connection_config.acquisition_channel_name
    )
    _validate_mapped_hardware_role(
        mapper,
        acquisition_channels,
        name=connection_config.acquisition_channel_name,
        role="acquisition",
        expected_instruments=("M5200Digitizer",),
    )
    (
        acquisition_duration_s,
        acquisition_sample_rate_hz,
    ) = _resolved_acquisition_timing(
        mapper,
        acquisition_channels,
        acquisition,
        hardware_demodulation=True,
    )
    if not np.isclose(
        acquisition_sample_rate_hz,
        QCS_M5200_SAMPLE_RATE_HZ,
        rtol=0.0,
        atol=1.0,
    ):
        raise QcsUnsupportedFeatureError(
            "QCS Stability acquisition requires the M5200 "
            f"{QCS_M5200_SAMPLE_RATE_HZ:g} S/s rate; mapper reports "
            f"{acquisition_sample_rate_hz:g} S/s"
        )
    resolved_acquisition_sample_count = int(
        round(acquisition_duration_s * acquisition_sample_rate_hz)
    )
    if (
        not np.isclose(
            acquisition_duration_s,
            resolved_acquisition_sample_count / acquisition_sample_rate_hz,
            rtol=0.0,
            atol=1.0e-15,
        )
        or resolved_acquisition_sample_count
        % QCS_STABILITY_INTEGRATION_BLOCK_SAMPLES
    ):
        raise ValueError(
            "QCS Stability integration time must be a multiple of "
            f"{QCS_STABILITY_INTEGRATION_QUANTUM_S * 1.0e9:.9f} ns "
            f"({QCS_STABILITY_INTEGRATION_BLOCK_SAMPLES} M5200 samples at "
            f"{QCS_M5200_SAMPLE_RATE_HZ / 1.0e9:g} GSa/s); got "
            f"{acquisition_duration_s * 1.0e9:.12g} ns "
            f"({resolved_acquisition_sample_count} samples)"
        )
    integration_segment_sample_counts = (
        qcs_stability_integration_segment_sample_counts(
            resolved_acquisition_sample_count
        )
    )
    acquisition_sample_count = int(sum(integration_segment_sample_counts))
    acquisition_duration_s = (
        acquisition_sample_count / acquisition_sample_rate_hz
    )
    segment_count = len(integration_segment_sample_counts)
    total_inter_segment_dead_time_s = (
        max(0, segment_count - 1)
        * QCS_STABILITY_INTER_SEGMENT_DELAY_S
    )
    acquisition_elapsed_duration_s = (
        acquisition_duration_s + total_inter_segment_dead_time_s
    )
    result_value_count = point_count * repetitions * segment_count
    if result_value_count > MAX_QCS_STABILITY_RESULT_VALUES:
        raise QcsUnsupportedFeatureError(
            "QCS Stability would return "
            f"{result_value_count:,} segmented hardware-demodulated IQ "
            f"values ({point_count:,} points x {repetitions} repetitions x "
            f"{segment_count} integrations); the safe application limit is "
            f"{MAX_QCS_STABILITY_RESULT_VALUES:,}. Reduce points, "
            "repetitions, or integration time."
        )
    if segment_count > 1:
        if acquisition.integration_filter is not None:
            raise QcsUnsupportedFeatureError(
                "Segmented QCS Stability integration requires the built-in "
                "flat IntegrationFilter; a custom integration filter cannot "
                "be divided safely"
            )
        if acquisition.envelope != "constant":
            raise QcsUnsupportedFeatureError(
                "Segmented QCS Stability integration requires a flat "
                "constant acquisition envelope"
            )

    segments = tuple(sequence.segments)
    if (
        len(segments) != 1
        or str(segments[0].kind) != "set"
        or any(
            str(axis.segment_name) != str(segments[0].name)
            for axis in axes
        )
    ):
        raise QcsUnsupportedFeatureError(
            "QCS Stability hardware sweep requires its dedicated "
            "single SET-and-hold sequence"
        )
    segment = segments[0]
    fabric_hz = _positive_finite(fabric_mhz, "fabric clock") * 1e6
    if not np.isclose(
        fabric_hz,
        QCS_FABRIC_CLOCK_HZ,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "QCS Stability hardware sweep requires the M5000-series "
            "300 MHz synchronization fabric"
        )
    duration_s = _fabric_aligned_seconds(
        float(segment.duration_cycles) / fabric_hz,
        fabric_hz=fabric_hz,
        label="QCS Stability sequence duration",
        positive=True,
    )
    acquisition_duration_s = _fabric_aligned_seconds(
        acquisition_duration_s,
        fabric_hz=fabric_hz,
        label="QCS Stability acquisition duration",
        positive=True,
    )
    acquisition_elapsed_duration_s = (
        acquisition_duration_s + total_inter_segment_dead_time_s
    )
    acquisition_pre_delay_s = _fabric_aligned_seconds(
        acquisition.pre_delay_s,
        fabric_hz=fabric_hz,
        label="QCS Stability acquisition pre-delay",
    )
    programmed_rf_pulses = []

    output_names = tuple(str(name) for name in sequence.output_names)
    try:
        x_output_index = output_names.index(str(x_axis.output_name))
        y_output_index = output_names.index(str(y_axis.output_name))
    except ValueError as exc:
        raise ValueError(
            "QCS Stability sweep axes do not match the output names"
        ) from exc
    if x_output_index == y_output_index:
        raise ValueError("QCS Stability X and Y outputs must differ")

    cross_capacitance = np.asarray(
        sequence.cross_capacitance,
        dtype=float,
    )
    expected_matrix_shape = (len(output_names), len(output_names))
    if cross_capacitance.shape != expected_matrix_shape:
        raise ValueError(
            "QCS Stability cross-capacitance matrix must have shape "
            f"{expected_matrix_shape}"
        )
    if not np.all(np.isfinite(cross_capacitance)):
        raise ValueError(
            "QCS Stability cross-capacitance coefficients must be finite"
        )

    raw_offsets = tuple(segment.amplitudes)
    if len(raw_offsets) != len(output_names):
        raise ValueError("QCS Stability SET amplitude count is invalid")
    virtual_offset = np.asarray(
        [0.0 if value is None else float(value) for value in raw_offsets],
        dtype=float,
    )
    # The selected coordinates replace, rather than add to, the SET values.
    virtual_offset[x_output_index] = 0.0
    virtual_offset[y_output_index] = 0.0
    qcs_scale = _positive_finite(
        source_full_scale_mv, "source waveform full scale"
    ) / (
        _positive_finite(
            connection_config.dc_full_scale_v,
            "QCS DC full scale",
        )
        * 1000.0
    )
    physical_offset = cross_capacitance @ virtual_offset * qcs_scale
    x_coefficients = (
        cross_capacitance[:, x_output_index] * qcs_scale
    )
    y_coefficients = (
        cross_capacitance[:, y_output_index] * qcs_scale
    )
    # FineTuneSequence's authoritative coordinate table is C-order: X is the
    # outer axis and Y varies fastest. Compute the cross-capacitance transform
    # on the host so no waveform amplitude depends on a compound two-Scalar
    # expression.
    sweep_coordinates = np.asarray(sequence.sweep_coordinates, dtype=float)
    if sweep_coordinates.shape != (point_count, 2):
        raise ValueError(
            "QCS Stability coordinate table must have shape "
            f"({point_count}, 2); got {sweep_coordinates.shape}"
        )
    flattened_x = sweep_coordinates[:, 0]
    flattened_y = sweep_coordinates[:, 1]
    physical_amplitudes = (
        physical_offset[np.newaxis, :]
        + flattened_x[:, np.newaxis] * x_coefficients[np.newaxis, :]
        + flattened_y[:, np.newaxis] * y_coefficients[np.newaxis, :]
    )
    peak_by_output = np.max(np.abs(physical_amplitudes), axis=0)
    if np.any(peak_by_output > 1.0 + 1e-12):
        output_index = int(np.argmax(peak_by_output))
        peak_voltage_v = (
            float(peak_by_output[output_index])
            * connection_config.dc_full_scale_v
        )
        raise ValueError(
            f"QCS DC output {output_names[output_index]!r} reaches "
            f"{peak_voltage_v:.6g} V after cross-capacitance correction, "
            "exceeding the configured +/-"
            f"{connection_config.dc_full_scale_v:.6g} V full scale"
        )

    compensation_duration_s = None
    compensation_amplitudes = None
    compensation_peak_by_output = np.asarray([], dtype=float)
    if compensation_config is not None:
        compensation_duration_s = _fabric_aligned_seconds(
            float(compensation_config.fixed_duration_cycles) / fabric_hz,
            fabric_hz=fabric_hz,
            label="QCS Stability DC compensation duration",
            positive=True,
        )
        # Both the target and compensation use a minimum ramp-up, direct
        # Hold, ramp-down, and one minimum terminal zero interval. The two
        # half-ramp areas together consume one ramp duration and the terminal
        # zero consumes another, so balance actual voltage-time areas rather
        # than ideal rectangles.
        stability_ramp_s = QCS_STABILITY_DC_RAMP_FABRIC_CYCLES / fabric_hz
        stability_ramp_samples = (
            QCS_STABILITY_DC_RAMP_FABRIC_CYCLES
            * QCS_M5301_SAMPLES_PER_FABRIC_CYCLE
        )
        # With the endpoint-flattened envelopes above, each N-sample ramp has
        # mean N/(2*N-1). The interval also contains one ramp-duration of
        # explicit terminal zero. These are exact rendered voltage-time areas
        # per unit plateau amplitude.
        stability_ramp_mean = stability_ramp_samples / (
            2 * stability_ramp_samples - 1
        )
        stability_edge_area_s = (
            (3.0 - 2.0 * stability_ramp_mean) * stability_ramp_s
        )
        measurement_effective_area_s = (
            duration_s - stability_edge_area_s
        )
        compensation_effective_area_s = (
            compensation_duration_s - stability_edge_area_s
        )
        if (
            measurement_effective_area_s <= 0.0
            or compensation_effective_area_s <= 0.0
        ):
            raise ValueError(
                "QCS Stability target or DC compensation duration is too "
                "short for the M5301 ramp-to-hold-to-zero output sequence"
            )
        # Precompute the opposite-area level on the host so HCL only sees
        # direct amplitude arrays and does not need dependent Scalar math.
        compensation_amplitudes = (
            -physical_amplitudes
            * measurement_effective_area_s
            / compensation_effective_area_s
        )
        if not np.all(np.isfinite(compensation_amplitudes)):
            raise ValueError(
                "QCS Stability DC compensation amplitudes must be finite"
            )
        compensation_peak_by_output = np.max(
            np.abs(compensation_amplitudes), axis=0
        )
        if np.any(compensation_peak_by_output > 1.0 + 1e-12):
            output_index = int(np.argmax(compensation_peak_by_output))
            peak_voltage_v = (
                float(compensation_peak_by_output[output_index])
                * connection_config.dc_full_scale_v
            )
            required_area_cycles = (
                float(peak_by_output[output_index])
                * measurement_effective_area_s
                * fabric_hz
            )
            minimum_duration_cycles = int(
                np.ceil(
                    np.nextafter(
                        required_area_cycles
                        + stability_edge_area_s * fabric_hz,
                        -np.inf,
                    )
                )
            )
            minimum_duration_us = (
                minimum_duration_cycles / fabric_hz * 1e6
            )
            raise ValueError(
                "QCS Bias-T fixed compensation time is too short for "
                f"{output_names[output_index]!r}; the compensation reaches "
                f"{peak_voltage_v:.6g} V, exceeding the configured +/-"
                f"{connection_config.dc_full_scale_v:.6g} V full scale. "
                "Increase DC compensation time to at least "
                f"{minimum_duration_us:.9g} us"
            )

    program = qcs.Program(name="PulseGenerator QCS Stability hardware sweep")
    physical_variables = []
    physical_arrays = []
    for output_index, output_name in enumerate(output_names):
        values = physical_amplitudes[:, output_index]
        amplitude = qcs.Scalar(
            f"stability_dc_{output_index}_amplitude",
            value=float(values[0]),
            dtype=float,
        )
        physical_variables.append(amplitude)
        physical_arrays.append(
            qcs.Array(
                f"stability_dc_{output_index}_values",
                value=values,
                dtype=float,
            )
        )
        program.add_waveform(
            _qcs_stability_measurement_interval(
                qcs,
                duration_s=duration_s,
                amplitude=amplitude,
                name=f"{output_name}_stability_dc",
                fabric_hz=fabric_hz,
            ),
            dc_channels[output_index],
            new_layer=output_index == 0,
        )

    segment_name = str(segment.name)
    for pulse in rf_pulses:
        if pulse.at_segment != segment_name:
            raise KeyError(
                f"QCS RF pulse references unknown Stability segment "
                f"{pulse.at_segment!r}"
            )
        pulse_delay_s = _fabric_aligned_seconds(
            pulse.delay_s,
            fabric_hz=fabric_hz,
            label=f"QCS RF gen_ch {pulse.gen_ch} delay",
        )
        pulse_duration_s = _fabric_aligned_seconds(
            pulse.duration_s,
            fabric_hz=fabric_hz,
            label=f"QCS RF gen_ch {pulse.gen_ch} duration",
            positive=True,
        )
        if segment_count > 1:
            if pulse.envelope != "constant":
                raise QcsUnsupportedFeatureError(
                    "Segmented QCS Stability integration requires a flat "
                    f"constant RF envelope on gen_ch {pulse.gen_ch}"
                )
            if not np.isclose(
                pulse_delay_s,
                acquisition_pre_delay_s,
                rtol=0.0,
                atol=1.0e-15,
            ) or not np.isclose(
                pulse_duration_s,
                acquisition_duration_s,
                rtol=0.0,
                atol=1.0e-15,
            ):
                raise QcsUnsupportedFeatureError(
                    "Segmented QCS Stability requires each RF output to "
                    "start with the acquisition and cover its complete "
                    "integrated duration"
                )
            rf_channel = rf_channels_by_gen_ch[int(pulse.gen_ch)]
            rf_absolute_phase = bool(
                getattr(rf_channel, "absolute_phase", False)
            )
            acquisition_absolute_phase = bool(
                getattr(acquisition_channels, "absolute_phase", False)
            )
            if rf_absolute_phase != acquisition_absolute_phase:
                raise QcsUnsupportedFeatureError(
                    "Segmented QCS Stability integration requires the "
                    "mapped M5300 RF output and M5200 acquisition virtual "
                    "channels to use matching absolute_phase settings; got "
                    f"RF={rf_absolute_phase} and acquisition="
                    f"{acquisition_absolute_phase}"
                )
        programmed_rf_pulse = {
            "gen_ch": pulse.gen_ch,
            "amplitude": pulse.amplitude,
            "frequency_hz": pulse.frequency_hz,
            "duration_s": pulse_duration_s,
            "delay_s": pulse_delay_s,
            "elapsed_duration_s": (
                acquisition_elapsed_duration_s
                if segment_count > 1
                else pulse_duration_s
            ),
            "segment_count": segment_count,
        }
        if pulse.power_calibration_provenance is not None:
            programmed_rf_pulse["power_calibration"] = dict(
                pulse.power_calibration_provenance
            )
        programmed_rf_pulses.append(programmed_rf_pulse)
        pulse_elapsed_duration_s = (
            acquisition_elapsed_duration_s
            if segment_count > 1
            else pulse_duration_s
        )
        if pulse_delay_s + pulse_elapsed_duration_s > duration_s + 1e-15:
            raise ValueError(
                f"QCS RF pulse on gen_ch {pulse.gen_ch} exceeds the "
                "Stability hold"
            )
        if segment_count == 1:
            program.add_waveform(
                qcs.RFWaveform(
                    duration=pulse_duration_s,
                    envelope=_qcs_envelope(qcs, pulse.envelope),
                    amplitude=pulse.amplitude,
                    rf_frequency=pulse.frequency_hz,
                    instantaneous_phase=pulse.phase_rad,
                    name=f"stability_rf_{pulse.gen_ch}",
                ),
                rf_channels_by_gen_ch[int(pulse.gen_ch)],
                new_layer=False,
                pre_delay=pulse_delay_s,
            )
        else:
            last_segment_index = segment_count - 1
            for segment_index, segment_samples in enumerate(
                integration_segment_sample_counts
            ):
                output_duration_s = (
                    segment_samples / acquisition_sample_rate_hz
                )
                if segment_index != last_segment_index:
                    output_duration_s += QCS_STABILITY_INTER_SEGMENT_DELAY_S
                waveform_options = {"new_layer": False}
                if segment_index == 0:
                    waveform_options["pre_delay"] = pulse_delay_s
                program.add_waveform(
                    qcs.RFWaveform(
                        duration=output_duration_s,
                        envelope=qcs.ConstantEnvelope(),
                        amplitude=pulse.amplitude,
                        rf_frequency=pulse.frequency_hz,
                        instantaneous_phase=pulse.phase_rad,
                        name=(
                            f"stability_rf_{pulse.gen_ch}_segment_"
                            f"{segment_index}"
                        ),
                    ),
                    rf_channels_by_gen_ch[int(pulse.gen_ch)],
                    **waveform_options,
                )

    if acquisition.at_segment != segment_name:
        raise KeyError(
            "QCS acquisition references unknown Stability segment "
            f"{acquisition.at_segment!r}"
        )
    if (
        acquisition_pre_delay_s + acquisition_elapsed_duration_s
        > duration_s + 1e-15
    ):
        raise ValueError("QCS acquisition exceeds the Stability hold")
    if segment_count == 1:
        integration_filter = acquisition.integration_filter
        if integration_filter is None:
            integration_filter = qcs.RFWaveform(
                duration=acquisition_duration_s,
                envelope=_qcs_envelope(qcs, acquisition.envelope),
                amplitude=1.0,
                rf_frequency=acquisition.frequency_hz,
                instantaneous_phase=acquisition.phase_rad,
                name="stability_acquisition_filter",
            )
        program.add_acquisition(
            integration_filter=integration_filter,
            channels=acquisition_channels,
            new_layer=False,
            pre_delay=acquisition_pre_delay_s,
        )
    else:
        integration_filter_cache = {}
        for segment_index, segment_samples in enumerate(
            integration_segment_sample_counts
        ):
            integration_filter = integration_filter_cache.get(segment_samples)
            if integration_filter is None:
                segment_duration_s = (
                    segment_samples / acquisition_sample_rate_hz
                )
                filter_waveform = qcs.RFWaveform(
                    duration=segment_duration_s,
                    envelope=qcs.ConstantEnvelope(),
                    amplitude=1.0,
                    rf_frequency=acquisition.frequency_hz,
                    instantaneous_phase=acquisition.phase_rad,
                    name=(
                        "stability_acquisition_filter_"
                        f"{segment_samples}_samples"
                    ),
                )
                integration_filter = qcs.IntegrationFilter(filter_waveform)
                integration_filter_cache[segment_samples] = integration_filter
            acquisition_options = {"new_layer": False}
            acquisition_options["pre_delay"] = (
                acquisition_pre_delay_s
                if segment_index == 0
                else QCS_STABILITY_INTER_SEGMENT_DELAY_S
            )
            program.add_acquisition(
                integration_filter=integration_filter,
                channels=acquisition_channels,
                **acquisition_options,
            )

    if compensation_amplitudes is not None:
        for output_index, output_name in enumerate(output_names):
            values = compensation_amplitudes[:, output_index]
            amplitude = qcs.Scalar(
                f"stability_dc_{output_index}_compensation_amplitude",
                value=float(values[0]),
                dtype=float,
            )
            physical_variables.append(amplitude)
            physical_arrays.append(
                qcs.Array(
                    f"stability_dc_{output_index}_compensation_values",
                    value=values,
                    dtype=float,
                )
            )
            program.add_waveform(
                _qcs_stability_compensation_interval(
                    qcs,
                    duration_s=compensation_duration_s,
                    amplitude=amplitude,
                    name=f"{output_name}_stability_dc_compensation",
                    fabric_hz=fabric_hz,
                ),
                dc_channels[output_index],
                new_layer=output_index == 0,
            )

    # All physical outputs advance together through one flattened Cartesian
    # hardware sweep, matching the QCSVideoProcessor reference design.
    program.sweep(
        physical_arrays,
        physical_variables,
    )
    program.n_shots(repetitions)
    point_program_duration_s = duration_s + (
        0.0
        if compensation_duration_s is None
        else compensation_duration_s
    )
    return QcsCompiledHardwareSweep(
        program=program,
        acquisition_channels=acquisition_channels,
        # ``duration_s`` is the complete per-point Program duration.  Keep
        # the compensation duration separately as well so callers can recover
        # the measurement hold as ``duration_s - compensation_duration_s``.
        duration_s=point_program_duration_s,
        acquisition_duration_s=acquisition_duration_s,
        acquisition_sample_rate_hz=acquisition_sample_rate_hz,
        sweep_shape=sweep_shape,
        bias_t_compensation_applied=compensation_config is not None,
        bias_t_compensation_duration_s=compensation_duration_s,
        bias_t_compensation_peak_amplitudes=tuple(
            float(value) for value in compensation_peak_by_output
        ),
        programmed_rf_pulses=tuple(programmed_rf_pulses),
        acquisition_pre_delay_s=acquisition_pre_delay_s,
        acquisition_sample_count=acquisition_sample_count,
        integration_segment_sample_counts=tuple(
            integration_segment_sample_counts
        ),
        inter_segment_delay_s=QCS_STABILITY_INTER_SEGMENT_DELAY_S,
        acquisition_elapsed_duration_s=acquisition_elapsed_duration_s,
    )


def execute_qcs_stability_hardware_sweep(
    *,
    connection_config: QcsConnectionConfig,
    sequence: Any,
    repetitions_per_point: int,
    fabric_mhz: float = 300.0,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
    progress_callback: Optional[ProgressCallback] = None,
    partial_callback: Optional[PartialResultCallback] = None,
    qcs_module=None,
    mapper=None,
    executor=None,
    compiled: Optional[QcsCompiledHardwareSweep] = None,
    cancellation: Optional[QcsCancellationController] = None,
) -> QcsExecutionResult:
    """Execute a Stability scan, averaging repeated bounded HCL passes."""
    qcs = _import_qcs() if qcs_module is None else qcs_module
    if acquisition is None:
        raise ValueError("QCS Stability hardware sweep requires an acquisition")
    requested_total_duration_s = acquisition.duration_s
    if acquisition.sample_count is not None:
        requested_total_duration_s = (
            acquisition.sample_count / QCS_M5200_SAMPLE_RATE_HZ
        )
    averaging_plan = plan_qcs_total_iq_averaging(
        requested_total_duration_s,
        sample_rate_hz=QCS_M5200_SAMPLE_RATE_HZ,
        block_samples=QCS_STABILITY_INTEGRATION_BLOCK_SAMPLES,
        max_pass_samples=QCS_STABILITY_MAX_INTEGRATION_SAMPLES,
    )
    pass_sample_count = averaging_plan.pass_sample_counts[0]
    pass_acquisition = replace(
        acquisition,
        duration_s=pass_sample_count / averaging_plan.sample_rate_hz,
        sample_count=pass_sample_count,
        sample_rate_hz=averaging_plan.sample_rate_hz,
    )
    if progress_callback is not None:
        progress_callback(0, "Validating QCS hardware sweep")
    if not connection_config.hw_demod:
        raise QcsUnsupportedFeatureError(
            "QCS Stability hardware sweep requires hardware demodulation"
        )
    if mapper is None:
        mapper = load_qcs_channel_mapper(
            connection_config,
            qcs_module=qcs,
        )
    if cancellation is not None:
        cancellation.bind(qcs, mapper)
        cancellation.raise_if_requested("QCS Stability setup")
    if compiled is None:
        if progress_callback is not None:
            progress_callback(
                10,
                "Compiling one native QCS physical-channel sweep",
            )
        compiled = compile_qcs_stability_hardware_sweep(
            sequence,
            connection_config=connection_config,
            mapper=mapper,
            repetitions_per_point=repetitions_per_point,
            fabric_mhz=fabric_mhz,
            source_full_scale_mv=source_full_scale_mv,
            rf_pulses=rf_pulses,
            acquisition=pass_acquisition,
            qcs_module=qcs,
        )
    elif int(compiled.acquisition_sample_count or 0) != pass_sample_count:
        raise ValueError(
            "precompiled QCS Stability Program integration does not match "
            "the planned per-pass duration"
        )
    if executor is None:
        executor = build_qcs_executor(
            connection_config,
            mapper,
            qcs_module=qcs,
        )

    if progress_callback is not None:
        progress_callback(
            35,
            (
                "Running one QCS hardware program for "
                f"{compiled.sweep_shape[0]} x "
                f"{compiled.sweep_shape[1]} points"
            ),
    )
    if cancellation is not None:
        cancellation.tag_program(compiled.program)
    iq_weighted_sum = None
    completed_passes = 0
    raw_result = None
    latest_partial = None

    def stability_partial(current_iq: np.ndarray) -> QcsExecutionResult:
        completed_samples = int(
            sum(averaging_plan.pass_sample_counts[:completed_passes])
        )
        partial_ddr = FineTuneDdrResult(
            sweep_points=np.asarray(sequence.sweep_points),
            iq=np.asarray(current_iq).copy(),
            sweep_axes=tuple(sequence.sweep_axes),
            sweep_shape=tuple(sequence.sweep_shape),
            cross_capacitance=np.asarray(sequence.cross_capacitance).copy(),
            sample_rate_hz=compiled.acquisition_sample_rate_hz,
            fir_rate_profile="qcs_hardware_demod",
        )
        partial_summary = {
            "backend": "qcs",
            "measurement": "stability_diagram",
            "hardware_sweep": True,
            "partial": completed_passes < averaging_plan.pass_count,
            "completed_points": int(np.prod(compiled.sweep_shape)),
            "planned_points": int(np.prod(compiled.sweep_shape)),
            "iq_averaging_pass_count": averaging_plan.pass_count,
            "completed_iq_averaging_passes": completed_passes,
            "requested_total_integration_duration_s": (
                averaging_plan.requested_total_duration_s
            ),
            "quantized_total_integration_duration_s": (
                averaging_plan.quantized_total_duration_s
            ),
            "per_pass_integration_duration_s": (
                compiled.acquisition_duration_s
            ),
            "effective_integration_duration_s": (
                completed_samples / compiled.acquisition_sample_rate_hz
            ),
            "iq_shape": list(current_iq.shape),
        }
        return QcsExecutionResult(
            ddr_result=partial_ddr,
            programs=(compiled.program,),
            raw_results=(() if raw_result is None else (raw_result,)),
            program_summary=partial_summary,
            rf_settings={
                "backend": "qcs",
                "readout_details": {
                    "hw_demod": True,
                    "effective_integration_duration_s": (
                        completed_samples / compiled.acquisition_sample_rate_hz
                    ),
                },
            },
        )

    try:
        for pass_samples in averaging_plan.pass_sample_counts:
            if cancellation is not None:
                cancellation.raise_if_requested(
                    "QCS Stability averaging pass"
                )
            raw_result = _executor_execute(executor, compiled.program)
            pass_iq = _extract_qcs_segmented_hardware_sweep_iq(
                raw_result,
                acquisition_channels=compiled.acquisition_channels,
                segment_sample_counts=(
                    compiled.integration_segment_sample_counts
                    or (int(compiled.acquisition_sample_count or 1),)
                ),
                repetitions_per_point=repetitions_per_point,
                sweep_shape=compiled.sweep_shape,
            )
            weighted = np.asarray(pass_iq) * float(
                pass_samples
            )
            if iq_weighted_sum is None:
                iq_weighted_sum = weighted
            else:
                iq_weighted_sum += weighted
            completed_passes += 1
            completed_samples = int(
                sum(averaging_plan.pass_sample_counts[:completed_passes])
            )
            iq = iq_weighted_sum / float(completed_samples)
            latest_partial = stability_partial(iq)
            if partial_callback is not None:
                partial_callback(latest_partial)
            if progress_callback is not None:
                progress_callback(
                    35 + int(65 * completed_passes / averaging_plan.pass_count),
                    "QCS Stability averaging pass "
                    f"{completed_passes:,}/{averaging_plan.pass_count:,} acquired",
                )
            if cancellation is not None:
                cancellation.raise_if_requested(
                    "QCS Stability averaging boundary"
                )
    except BaseException as execution_error:
        if (
            cancellation is not None
            and cancellation.is_stop_requested()
        ):
            raise QcsExperimentCancelled(
                "QCS Stability scan stopped by user; completed averaging "
                "passes remain available",
                partial_result=latest_partial,
            ) from execution_error
        raise
    if iq_weighted_sum is None or raw_result is None:
        raise RuntimeError("QCS Stability averaging completed no passes")
    iq = iq_weighted_sum / float(
        sum(averaging_plan.pass_sample_counts[:completed_passes])
    )
    if cancellation is not None and not cancellation.close_stop_window():
        raise QcsExperimentCancelled(
            "QCS Stability scan stopped after its final averaging pass",
            partial_result=latest_partial,
        )
    ddr_result = FineTuneDdrResult(
        sweep_points=np.asarray(sequence.sweep_points),
        iq=iq,
        sweep_axes=tuple(sequence.sweep_axes),
        sweep_shape=tuple(sequence.sweep_shape),
        cross_capacitance=np.asarray(sequence.cross_capacitance).copy(),
        sample_rate_hz=compiled.acquisition_sample_rate_hz,
        fir_rate_profile="qcs_hardware_demod",
    )
    if progress_callback is not None:
        progress_callback(100, "QCS hardware sweep acquired")

    point_count = int(np.prod(compiled.sweep_shape))
    compensation_duration_s = float(
        compiled.bias_t_compensation_duration_s or 0.0
    )
    measurement_hold_duration_s = (
        float(compiled.duration_s) - compensation_duration_s
    )
    inter_iteration_delay_s = float(connection_config.init_time_s)
    summary = {
        "backend": "qcs",
        "hardware_sweep": True,
        "hardware_sweep_dimensions": 1,
        "hardware_sweep_shape": [point_count],
        "stability_grid_dimensions": 2,
        "stability_grid_shape": list(compiled.sweep_shape),
        "hardware_sweep_points": point_count,
        "software_sweep_points": 0,
        "program_count": 1,
        "executor_call_count": completed_passes,
        "compiled_program_reused": completed_passes > 1,
        "repetitions_per_point": int(repetitions_per_point),
        "fabric_mhz": float(fabric_mhz),
        "source_full_scale_mv": float(source_full_scale_mv),
        "qcs_dc_full_scale_v": connection_config.dc_full_scale_v,
        "dc_channel_names": list(connection_config.dc_channel_names),
        "rf_channel_names": {
            str(key): value
            for key, value in connection_config.rf_channel_names.items()
        },
        "acquisition_channel_name": (
            connection_config.acquisition_channel_name
        ),
        "hw_demod": True,
        "reset_phase_every_shot": True,
        "acquisition_result_type": "integrated_iq",
        "sample_rate_hz": compiled.acquisition_sample_rate_hz,
        "acquisition_duration_s": averaging_plan.quantized_total_duration_s,
        "integration_duration_s": averaging_plan.quantized_total_duration_s,
        "integration_sample_count": averaging_plan.quantized_total_sample_count,
        "requested_total_integration_duration_s": (
            averaging_plan.requested_total_duration_s
        ),
        "quantized_total_integration_duration_s": (
            averaging_plan.quantized_total_duration_s
        ),
        "quantized_total_integration_sample_count": (
            averaging_plan.quantized_total_sample_count
        ),
        "iq_averaging_pass_count": averaging_plan.pass_count,
        "completed_iq_averaging_passes": completed_passes,
        "per_pass_integration_duration_s": compiled.acquisition_duration_s,
        "per_pass_integration_sample_count": compiled.acquisition_sample_count,
        "effective_integration_duration_s": (
            sum(averaging_plan.pass_sample_counts[:completed_passes])
            / compiled.acquisition_sample_rate_hz
        ),
        "integration_segment_sample_counts": list(
            compiled.integration_segment_sample_counts
        ),
        "integration_segment_count": len(
            compiled.integration_segment_sample_counts
        ) or 1,
        "inter_segment_delay_s": compiled.inter_segment_delay_s,
        "total_inter_segment_dead_time_s": max(
            0,
            len(compiled.integration_segment_sample_counts) - 1,
        ) * compiled.inter_segment_delay_s,
        "acquisition_elapsed_duration_s": (
            compiled.acquisition_elapsed_duration_s
            if compiled.acquisition_elapsed_duration_s is not None
            else compiled.acquisition_duration_s
        ),
        "hardware_demodulated_iq_reduction": (
            "sample_count_weighted_complex_mean"
            if len(compiled.integration_segment_sample_counts) > 1
            else "single_integration_filter"
        ),
        "acquisition_pre_delay_s": compiled.acquisition_pre_delay_s,
        "measurement_hold_duration_s": measurement_hold_duration_s,
        "point_program_duration_s": float(compiled.duration_s),
        "inter_iteration_delay_s": inter_iteration_delay_s,
        "nominal_point_period_s": (
            float(compiled.duration_s) + inter_iteration_delay_s
        ),
        "requested_sample_count": acquisition.sample_count,
        "bias_t_compensation_applied": (
            compiled.bias_t_compensation_applied
        ),
        "bias_t_compensation_type": (
            "dc" if compiled.bias_t_compensation_applied else None
        ),
        "bias_t_compensation_mode": (
            "fixed_time" if compiled.bias_t_compensation_applied else None
        ),
        "bias_t_compensation_duration_s": (
            compiled.bias_t_compensation_duration_s
        ),
        "bias_t_compensation_peak_amplitudes": list(
            compiled.bias_t_compensation_peak_amplitudes
        ),
        "iq_shape": list(iq.shape),
    }
    rf_settings = {
        "backend": "qcs",
        "output_details": tuple(
            compiled.programmed_rf_pulses
            or tuple(
                {
                    "gen_ch": pulse.gen_ch,
                    "amplitude": pulse.amplitude,
                    "frequency_hz": pulse.frequency_hz,
                    "duration_s": pulse.duration_s,
                    "delay_s": pulse.delay_s,
                }
                for pulse in rf_pulses
            )
        ),
        "readout_details": {
            "sample_rate_hz": compiled.acquisition_sample_rate_hz,
            "hw_demod": True,
            "reset_phase_every_shot": True,
            "acquisition_result_type": "integrated_iq",
            "frequency_hz": (
                0.0 if acquisition is None else acquisition.frequency_hz
            ),
            "duration_s": averaging_plan.quantized_total_duration_s,
            "integration_duration_s": averaging_plan.quantized_total_duration_s,
            "integration_sample_count": averaging_plan.quantized_total_sample_count,
            "requested_total_integration_duration_s": (
                averaging_plan.requested_total_duration_s
            ),
            "quantized_total_integration_duration_s": (
                averaging_plan.quantized_total_duration_s
            ),
            "iq_averaging_pass_count": averaging_plan.pass_count,
            "completed_iq_averaging_passes": completed_passes,
            "per_pass_integration_duration_s": compiled.acquisition_duration_s,
            "per_pass_integration_sample_count": compiled.acquisition_sample_count,
            "integration_segment_sample_counts": list(
                compiled.integration_segment_sample_counts
            ),
            "integration_segment_count": len(
                compiled.integration_segment_sample_counts
            ) or 1,
            "inter_segment_delay_s": compiled.inter_segment_delay_s,
            "total_inter_segment_dead_time_s": max(
                0,
                len(compiled.integration_segment_sample_counts) - 1,
            ) * compiled.inter_segment_delay_s,
            "elapsed_duration_s": (
                compiled.acquisition_elapsed_duration_s
                if compiled.acquisition_elapsed_duration_s is not None
                else compiled.acquisition_duration_s
            ),
            "hardware_demodulated_iq_reduction": (
                "sample_count_weighted_complex_mean"
                if len(compiled.integration_segment_sample_counts) > 1
                else "single_integration_filter"
            ),
            "pre_delay_s": compiled.acquisition_pre_delay_s,
            "measurement_hold_duration_s": measurement_hold_duration_s,
            "point_program_duration_s": float(compiled.duration_s),
            "inter_iteration_delay_s": inter_iteration_delay_s,
            "requested_sample_count": acquisition.sample_count,
            # "adc" denotes the GUI's uncalibrated numeric representation;
            # the QCS payload itself is integrated I/Q, not a raw ADC trace.
            "measurement_representation": "adc",
        },
    }
    return QcsExecutionResult(
        ddr_result=ddr_result,
        programs=(compiled.program,),
        raw_results=(raw_result,),
        program_summary=summary,
        rf_settings=rf_settings,
    )


def execute_qcs_sequence(
    *,
    connection_config: QcsConnectionConfig,
    sequence: Any,
    repetitions_per_sweep: int,
    fabric_mhz: float = 300.0,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
    progress_callback: Optional[ProgressCallback] = None,
    event_callback: Optional[EventCallback] = None,
    partial_callback: Optional[PartialResultCallback] = None,
    qcs_module=None,
    mapper=None,
    executor=None,
    cancellation: Optional[QcsCancellationController] = None,
) -> QcsExecutionResult:
    """Compile and execute a sequence without writing a database."""
    qcs = _import_qcs() if qcs_module is None else qcs_module
    if cancellation is not None:
        cancellation.raise_if_requested("QCS validation")
    if acquisition is None:
        raise ValueError("QCS execution requires an acquisition configuration")
    if progress_callback is not None:
        progress_callback(0, "Validating QCS experiment")
    if event_callback is not None:
        event_callback(
            "validation", "started", "Validating QCS capabilities"
        )
    validate_qcs_capabilities(
        connection_config=connection_config,
        sequence=sequence,
        rf_pulses=rf_pulses,
        acquisition=acquisition,
    )
    if cancellation is not None:
        cancellation.raise_if_requested("QCS validation")
    fixed_voltage_bias_t_sweep = (
        _qcs_fixed_voltage_bias_t_varies_with_sweep(sequence)
    )
    point_count = int(sequence.sweep_point_count)
    dc_offset_plan = (
        None
        if fixed_voltage_bias_t_sweep
        else _qcs_fixed_dc_offset_plan(
            sequence,
            source_full_scale_mv=source_full_scale_mv,
            dc_full_scale_v=connection_config.dc_full_scale_v,
            cancellation=cancellation,
        )
    )
    nonzero_dc_offset = bool(
        dc_offset_plan is not None
        and any(
            not np.isclose(value, 0.0, rtol=0.0, atol=1e-15)
            for value in dc_offset_plan.offset_volts
        )
    )
    host_preview = qcs_sweep_execution_preview(
        sequence,
        hardware_demodulation=connection_config.hw_demod,
        source_full_scale_mv=source_full_scale_mv,
        dc_full_scale_v=connection_config.dc_full_scale_v,
        fabric_mhz=fabric_mhz,
        init_time_s=connection_config.init_time_s,
    )
    use_fixed_offset_plan = bool(
        dc_offset_plan is not None
        and not fixed_voltage_bias_t_sweep
        and (
            not nonzero_dc_offset
            or point_count <= 1
            or host_preview.mode == "hardware"
        )
    )
    defer_capacity_to_mixed_planner = bool(
        host_preview.mode == "hybrid"
        and connection_config.hw_demod
        and len(tuple(sequence.sweep_axes)) >= 2
        and _qcs_mixed_candidate_hardware_axes(sequence)
    )
    if defer_capacity_to_mixed_planner:
        # An outer timing/RF coordinate can turn a long plateau into a valid
        # short M5301 seed plus Hold. The exact synchronized compiler checks
        # every selected inner slice; validating the unsliced grid with the
        # fixed-numeric model here would reject that valid mixed graph before
        # partitioning. If planning fails, the fallback validates itself.
        synchronized_capacity_prevalidated = False
        fixed_numeric_capacity_prevalidated = False
    elif not use_fixed_offset_plan:
        validate_qcs_m5301_waveform_capacity(
            sequence,
            fabric_mhz=fabric_mhz,
            amplitude_scale=(
                float(source_full_scale_mv)
                / (float(connection_config.dc_full_scale_v) * 1000.0)
            ),
        )
        synchronized_capacity_prevalidated = False
        fixed_numeric_capacity_prevalidated = True
    else:
        point_indices = (
            None
            if int(sequence.sweep_point_count) <= 256
            else qcs_m5301_capacity_preview_point_indices(sequence)
        )
        validate_qcs_m5301_waveform_capacity(
            sequence,
            point_indices=point_indices,
            fabric_mhz=fabric_mhz,
            amplitude_scale=(
                float(source_full_scale_mv)
                / (float(connection_config.dc_full_scale_v) * 1000.0)
            ),
            source_full_scale_mv=source_full_scale_mv,
            dc_full_scale_v=connection_config.dc_full_scale_v,
        )
        synchronized_capacity_prevalidated = True
        fixed_numeric_capacity_prevalidated = False
    if cancellation is not None:
        cancellation.raise_if_requested("QCS waveform-capacity validation")
    if event_callback is not None:
        event_callback(
            "validation", "completed", "QCS settings validated"
        )

    if event_callback is not None:
        event_callback(
            "connection", "started", "Loading QCS ChannelMapper"
        )
    if mapper is None:
        mapper = load_qcs_channel_mapper(
            connection_config, qcs_module=qcs
        )
    if cancellation is not None:
        cancellation.bind(qcs, mapper)
        cancellation.raise_if_requested("QCS hardware connection")
    # Resolve all configured names before compiling any points.
    for name in connection_config.dc_channel_names:
        _resolve_mapper_channel(mapper, name)
    for name in connection_config.rf_channel_names.values():
        _resolve_mapper_channel(mapper, name)
    _resolve_mapper_channel(
        mapper, connection_config.acquisition_channel_name
    )
    requested_acquisition = acquisition
    iq_averaging_plan: Optional[QcsIqAveragingPlan] = None
    if connection_config.hw_demod:
        acquisition_channel = _resolve_mapper_channel(
            mapper, connection_config.acquisition_channel_name
        )
        acquisition_rate_hz = _mapped_channel_sample_rate(
            mapper, acquisition_channel
        )
        if acquisition_rate_hz is None:
            acquisition_rate_hz = acquisition.sample_rate_hz
        requested_total_duration_s = acquisition.duration_s
        if acquisition.sample_count is not None:
            requested_total_duration_s = (
                acquisition.sample_count / acquisition_rate_hz
            )
        iq_averaging_plan = plan_qcs_total_iq_averaging(
            requested_total_duration_s,
            sample_rate_hz=acquisition_rate_hz,
        )
        if (
            iq_averaging_plan.pass_count > 1
            and acquisition.integration_filter is not None
        ):
            raise QcsUnsupportedFeatureError(
                "multi-pass QCS I/Q averaging requires the built-in flat "
                "filter; a custom filter cannot be repeated safely"
            )
        pass_sample_count = iq_averaging_plan.pass_sample_counts[0]
        acquisition = replace(
            acquisition,
            duration_s=pass_sample_count / acquisition_rate_hz,
            sample_count=pass_sample_count,
            sample_rate_hz=acquisition_rate_hz,
        )
    rf_pulses = resolve_qcs_rf_power_calibrations(
        connection_config=connection_config,
        mapper=mapper,
        rf_pulses=rf_pulses,
    )
    if dc_offset_plan is not None and not _mapper_supports_qcs_dc_offsets(
        mapper,
        channel_names=connection_config.dc_channel_names,
        offset_volts=dc_offset_plan.offset_volts,
    ):
        fallback_offset_plan = _qcs_without_optional_dc_offsets(
            dc_offset_plan
        )
        if _mapper_supports_qcs_dc_offsets(
            mapper,
            channel_names=connection_config.dc_channel_names,
            offset_volts=fallback_offset_plan.offset_volts,
        ):
            if fallback_offset_plan.offset_volts != dc_offset_plan.offset_volts:
                synchronized_capacity_prevalidated = False
            dc_offset_plan = fallback_offset_plan
            nonzero_dc_offset = any(
                not np.isclose(value, 0.0, rtol=0.0, atol=1e-15)
                for value in dc_offset_plan.offset_volts
            )
    if event_callback is not None:
        event_callback(
            "connection", "completed", "QCS ChannelMapper loaded"
        )
    if progress_callback is not None:
        progress_callback(8, "QCS ChannelMapper loaded")

    full_sweep_preview = host_preview
    has_multiple_sweep_axes = len(tuple(sequence.sweep_axes)) >= 2
    has_native_candidate_axis = bool(
        _qcs_mixed_candidate_hardware_axes(sequence)
    )
    attempt_mixed_partition = bool(
        has_multiple_sweep_axes
        and connection_config.hw_demod
        and has_native_candidate_axis
        and full_sweep_preview.mode != "invalid"
    )
    # An oversized full grid can still be legal when only the Python outer
    # coordinate count is <=10k. Allow the exact planner to determine that
    # partition instead of applying the all-software total-point limit.
    if (
        has_multiple_sweep_axes
        and connection_config.hw_demod
        and has_native_candidate_axis
        and full_sweep_preview.mode == "invalid"
        and not any(isinstance(axis, RfPowerSweep) for axis in sequence.sweep_axes)
    ):
        attempt_mixed_partition = True
    if attempt_mixed_partition:
        if event_callback is not None:
            event_callback(
                "sweep_partition",
                "started",
                "Checking for a native hardware sweep inside a software loop",
            )
        if progress_callback is not None:
            progress_callback(10, "Analyzing QCS mixed-sweep partition")
    mixed_plan, precompiled_full_sweep = (
        _plan_qcs_mixed_sweep(
            sequence,
            connection_config=connection_config,
            mapper=mapper,
            repetitions_per_sweep=repetitions_per_sweep,
            fabric_mhz=fabric_mhz,
            source_full_scale_mv=source_full_scale_mv,
            rf_pulses=rf_pulses,
            acquisition=acquisition,
            qcs_module=qcs,
            cancellation=cancellation,
            include_full_candidate=(
                full_sweep_preview.mode == "hardware"
            ),
            full_dc_offset_plan=dc_offset_plan,
        )
        if attempt_mixed_partition
        else (None, None)
    )
    if attempt_mixed_partition and event_callback is not None:
        if mixed_plan is not None:
            partition_message = "Mixed hardware/software partition selected"
        elif precompiled_full_sweep is not None:
            partition_message = "Full sweep confirmed native; no software loop needed"
        else:
            partition_message = "No native inner hardware-sweep partition is available"
        event_callback(
            "sweep_partition",
            "completed",
            partition_message,
        )
    if mixed_plan is not None:
        if executor is None:
            executor = build_qcs_executor(
                connection_config,
                mapper,
                qcs_module=qcs,
            )
        if cancellation is not None:
            cancellation.raise_if_requested("QCS mixed-sweep executor setup")
        return _execute_qcs_mixed_sweep(
            sequence,
            plan=mixed_plan,
            connection_config=connection_config,
            mapper=mapper,
            executor=executor,
            repetitions_per_sweep=repetitions_per_sweep,
            fabric_mhz=fabric_mhz,
            source_full_scale_mv=source_full_scale_mv,
            rf_pulses=rf_pulses,
            acquisition=acquisition,
            iq_averaging_plan=iq_averaging_plan,
            requested_acquisition=requested_acquisition,
            qcs_module=qcs,
            progress_callback=progress_callback,
            event_callback=event_callback,
            partial_callback=partial_callback,
            cancellation=cancellation,
        )
    fixed_numeric_dc_ramp = bool(
        dc_offset_plan is None or fixed_voltage_bias_t_sweep
    )
    fixed_numeric_dc_reason = (
        "Voltage ramps do not share a zero endpoint. Physical channel offsets "
        "are disabled, so QCS uses fixed numeric point programs and splits "
        "bipolar ramps at 0 V."
    )
    if fixed_voltage_bias_t_sweep:
        fixed_numeric_dc_reason = (
            "Fixed-voltage Bias-T compensation changes duration across the "
            "voltage sweep, so QCS uses fixed numeric point programs with a "
            "direct ramp-to-Hold compensation tail."
        )
    if nonzero_dc_offset and point_count > 1:
        assert host_preview is not None
        if host_preview.mode != "hardware":
            fixed_numeric_dc_ramp = True
            fixed_numeric_dc_reason = (
                "This sweep requires software resolution, so the fixed M5301 "
                "offset cannot remain active across software-controlled gaps; "
                "QCS uses zero-offset fixed numeric point programs. "
                + " ".join(host_preview.reasons)
            )
    if dc_offset_plan is not None and not _mapper_supports_qcs_dc_offsets(
        mapper,
        channel_names=connection_config.dc_channel_names,
        offset_volts=dc_offset_plan.offset_volts,
    ):
        fixed_numeric_dc_ramp = True
        fixed_numeric_dc_reason = (
            "The mapped M5301 channel does not expose the fixed physical "
            "offset required by this voltage sweep, so QCS uses fixed numeric "
            "point programs."
        )
    if (
        nonzero_dc_offset
        and point_count > 1
        and not fixed_numeric_dc_ramp
        and any(
            bool(
                getattr(
                    _resolve_mapper_channel(mapper, channel_name),
                    "absolute_phase",
                    False,
                )
            )
            for channel_name in connection_config.dc_channel_names
        )
    ):
        fixed_numeric_dc_ramp = True
        fixed_numeric_dc_reason = (
            "A mapped M5301 DC channel uses absolute_phase=True, which is not "
            "hardware-sweepable. QCS uses zero-offset fixed numeric point "
            "programs so the physical offset cannot persist across software "
            "gaps."
        )
    if fixed_numeric_dc_ramp:
        _validate_qcs_software_sweep_point_count(point_count)
    safety_reset_executor_call_count = 0
    if fixed_numeric_dc_ramp and not fixed_numeric_capacity_prevalidated:
        validate_qcs_m5301_waveform_capacity(
            sequence,
            fabric_mhz=fabric_mhz,
            amplitude_scale=(
                float(source_full_scale_mv)
                / (float(connection_config.dc_full_scale_v) * 1000.0)
            ),
        )
        fixed_numeric_capacity_prevalidated = True
    if event_callback is not None:
        event_callback(
            "program_build",
            "started",
            (
                "Compiling fixed numeric QCS DC-ramp programs"
                if fixed_numeric_dc_ramp
                else "Compiling one synchronized QCS sweep"
            ),
        )

    if fixed_numeric_dc_ramp:
        compiled_points = compile_qcs_sequence(
            sequence,
            connection_config=connection_config,
            mapper=mapper,
            repetitions_per_sweep=repetitions_per_sweep,
            fabric_mhz=fabric_mhz,
            source_full_scale_mv=source_full_scale_mv,
            rf_pulses=rf_pulses,
            acquisition=acquisition,
            qcs_module=qcs,
            progress_callback=progress_callback,
            cancellation=cancellation,
            _capacity_prevalidated=True,
        )
        if event_callback is not None:
            event_callback(
                "program_build",
                "completed",
                (
                    f"Compiled {point_count:,} fixed numeric QCS DC-ramp "
                    "program(s)"
                ),
            )
    else:
        compiled = precompiled_full_sweep
        if compiled is not None:
            _set_qcs_dc_channel_offsets(
                mapper,
                channel_names=connection_config.dc_channel_names,
                offset_volts=compiled.dc_channel_offsets_v,
                require_nonzero_support=True,
            )
        else:
            compiled = compile_qcs_synchronized_sweep(
                sequence,
                connection_config=connection_config,
                mapper=mapper,
                repetitions_per_sweep=repetitions_per_sweep,
                fabric_mhz=fabric_mhz,
                source_full_scale_mv=source_full_scale_mv,
                rf_pulses=rf_pulses,
                acquisition=acquisition,
                qcs_module=qcs,
                cancellation=cancellation,
                _capacity_prevalidated=synchronized_capacity_prevalidated,
                _dc_offset_plan=dc_offset_plan,
            )
        if (
            nonzero_dc_offset
            and point_count > 1
            and not compiled.hardware_sweep
        ):
            _set_qcs_dc_channel_offsets(
                mapper,
                channel_names=connection_config.dc_channel_names,
                offset_volts=(0.0,) * len(connection_config.dc_channel_names),
                require_nonzero_support=False,
            )
            raise QcsUnsupportedFeatureError(
                "QCS compilation unexpectedly downgraded a nonzero-offset "
                "voltage sweep to software execution. The offset was reset "
                "to zero; adjust the mapper/settings until the GUI confirms "
                "Hardware sweep, or use a fixed-numeric waveform."
            )
        if not compiled.hardware_sweep:
            _validate_qcs_software_sweep_point_count(point_count)
        if event_callback is not None:
            if compiled.hardware_sweep:
                build_message = (
                    "Compiled one native QCS hardware sweep using "
                    "DCWaveform amplitudes; physical offsets are disabled"
                )
            else:
                build_message = "Compiled one QCS-managed software sweep"
            event_callback(
                "program_build",
                "completed",
                build_message,
            )

    if executor is None:
        try:
            executor = build_qcs_executor(
                connection_config,
                mapper,
                qcs_module=qcs,
            )
        except BaseException:
            if not fixed_numeric_dc_ramp:
                _set_qcs_dc_channel_offsets(
                    mapper,
                    channel_names=connection_config.dc_channel_names,
                    offset_volts=(0.0,)
                    * len(connection_config.dc_channel_names),
                    require_nonzero_support=False,
                )
            raise
    if cancellation is not None:
        cancellation.raise_if_requested("QCS executor setup")

    def close_cancellation_window(
        *,
        dc_already_reset: bool = False,
        partial_result: Optional[QcsExecutionResult] = None,
    ) -> None:
        """Resolve the final Stop/result-processing race atomically."""
        if cancellation is None or cancellation.close_stop_window():
            return
        if not dc_already_reset:
            try:
                _reset_qcs_dc_outputs_to_zero(
                    qcs,
                    executor=executor,
                    mapper=mapper,
                    channel_names=connection_config.dc_channel_names,
                    fabric_hz=float(fabric_mhz) * 1e6,
                )
            except Exception as reset_error:
                raise RuntimeError(
                    "QCS stop was requested, but the emergency DC reset "
                    "failed; the physical output state is unknown"
                ) from reset_error
        raise QcsExperimentCancelled(
            "QCS experiment stopped by user after acquisition; DC outputs "
            "were reset to zero",
            partial_result=partial_result,
        )

    averaging_pass_counts = (
        (1,)
        if iq_averaging_plan is None
        else iq_averaging_plan.pass_sample_counts
    )
    measurement_executor_call_count = 0

    if fixed_numeric_dc_ramp:
        if event_callback is not None:
            event_callback(
                "acquisition",
                "started",
                "Executing fixed numeric QCS DC-ramp programs",
            )
        if progress_callback is not None:
            progress_callback(
                35,
                (
                    f"Running {point_count:,} hardware-safe fixed numeric "
                    "QCS point(s)"
                ),
            )
        point_weighted_iq = [None] * point_count
        raw_results_list = [None] * point_count
        executed_sample_rates = []
        latest_partial_execution = None
        try:
            total_calls = point_count * len(averaging_pass_counts)
            completed_calls = 0
            for pass_index, pass_samples in enumerate(averaging_pass_counts):
                for point_index, current in enumerate(compiled_points):
                    if cancellation is not None:
                        cancellation.raise_if_requested(
                            "fixed-numeric QCS point execution"
                        )
                        cancellation.tag_program(current.program)
                        cancellation.raise_if_requested(
                            "fixed-numeric QCS point execution"
                        )
                    raw_result = _executor_execute(executor, current.program)
                    measurement_executor_call_count += 1
                    if cancellation is not None:
                        cancellation.raise_if_requested(
                            "fixed-numeric QCS point execution"
                        )
                    raw_results_list[point_index] = raw_result
                    if not connection_config.hw_demod:
                        executed_sample_rate = _executed_program_sample_rate(
                            raw_result,
                            current.acquisition_channels,
                        )
                        if executed_sample_rate is not None:
                            executed_sample_rates.append(executed_sample_rate)
                    pass_iq = _extract_qcs_compiled_point_iq(
                        raw_result,
                        current,
                        repetitions_per_sweep=repetitions_per_sweep,
                        hardware_demodulation=connection_config.hw_demod,
                    )
                    weighted = np.asarray(pass_iq) * float(
                        pass_samples
                    )
                    if point_weighted_iq[point_index] is None:
                        point_weighted_iq[point_index] = weighted
                    else:
                        point_weighted_iq[point_index] += weighted
                    completed_calls += 1
                    if progress_callback is not None:
                        progress_callback(
                            35 + int(35 * completed_calls / total_calls),
                            "Acquired fixed numeric QCS execution "
                            f"{completed_calls:,}/{total_calls:,}",
                        )
                if connection_config.hw_demod:
                    completed_pass_count = pass_index + 1
                    completed_samples = int(
                        sum(
                            averaging_pass_counts[:completed_pass_count]
                        )
                    )
                    partial_iq = np.stack(
                        [
                            value / float(completed_samples)
                            for value in point_weighted_iq
                        ],
                        axis=0,
                    )
                    partial_ddr = FineTuneDdrResult(
                        sweep_points=np.asarray(sequence.sweep_points),
                        iq=partial_iq,
                        sweep_axes=tuple(sequence.sweep_axes),
                        sweep_shape=tuple(sequence.sweep_shape),
                        cross_capacitance=np.asarray(
                            sequence.cross_capacitance
                        ).copy(),
                        sample_rate_hz=float(
                            compiled_points[0].acquisition_sample_rate_hz
                        ),
                        fir_rate_profile="qcs_hardware_demod",
                    )
                    latest_partial_execution = QcsExecutionResult(
                        ddr_result=partial_ddr,
                        programs=tuple(
                            current.program for current in compiled_points
                        ),
                        raw_results=tuple(
                            value
                            for value in raw_results_list
                            if value is not None
                        ),
                        program_summary={
                            "backend": "qcs",
                            "partial": (
                                completed_pass_count
                                < len(averaging_pass_counts)
                            ),
                            "completed_points": point_count,
                            "planned_points": point_count,
                            "iq_averaging_pass_count": len(
                                averaging_pass_counts
                            ),
                            "completed_iq_averaging_passes": (
                                completed_pass_count
                            ),
                            "effective_integration_duration_s": (
                                completed_samples
                                / float(
                                    compiled_points[
                                        0
                                    ].acquisition_sample_rate_hz
                                )
                            ),
                            "iq_shape": list(partial_iq.shape),
                        },
                        rf_settings={"backend": "qcs"},
                    )
                    if partial_callback is not None:
                        partial_callback(latest_partial_execution)
            try:
                iq = np.stack(
                    [
                        value / float(sum(averaging_pass_counts))
                        for value in point_weighted_iq
                    ],
                    axis=0,
                )
            except ValueError as exc:
                raise ValueError(
                    "fixed numeric QCS DC-ramp points returned inconsistent "
                    "acquisition shapes"
                ) from exc
        except BaseException as execution_error:
            cancellation_requested = bool(
                cancellation is not None
                and cancellation.is_stop_requested()
            )
            reset_error_caught = None
            try:
                _reset_qcs_dc_outputs_to_zero(
                    qcs,
                    executor=executor,
                    mapper=mapper,
                    channel_names=connection_config.dc_channel_names,
                    fabric_hz=float(fabric_mhz) * 1e6,
                )
            except Exception as reset_error:
                reset_error_caught = reset_error
                if hasattr(execution_error, "add_note"):
                    execution_error.add_note(
                        "Automatic QCS DC reset also failed: "
                        f"{type(reset_error).__name__}: {reset_error}"
                    )
            if cancellation_requested and reset_error_caught is not None:
                raise RuntimeError(
                    "QCS stop was requested, but the emergency DC reset "
                    "failed; the physical output state is unknown"
                ) from reset_error_caught
            if cancellation_requested:
                raise QcsExperimentCancelled(
                    "QCS experiment stopped by user; completed averaging "
                    "passes remain available and DC outputs were reset to zero",
                    partial_result=latest_partial_execution,
                ) from execution_error
            raise
        close_cancellation_window(partial_result=latest_partial_execution)
        programs = tuple(current.program for current in compiled_points)
        raw_results = tuple(
            value for value in raw_results_list if value is not None
        )
        effective_acquisition_duration_s = max(
            float(current.acquisition_duration_s) for current in compiled_points
        )
        effective_sample_rate_hz = float(
            compiled_points[0].acquisition_sample_rate_hz
        )
        if executed_sample_rates:
            if not np.allclose(
                executed_sample_rates,
                executed_sample_rates[0],
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "fixed numeric QCS points used inconsistent digitizer "
                    "sample rates"
                )
            effective_sample_rate_hz = float(executed_sample_rates[0])
        hardware_sweep = False
        compiled_sweep_shape = tuple(sequence.sweep_shape)
        software_sweep_reasons = (fixed_numeric_dc_reason,)
        sweep_variable_count = 0
        sweep_array_value_count = 0
        applied_dc_offsets_v = (0.0,) * len(
            connection_config.dc_channel_names
        )
        dc_offset_init_compensation_v = (0.0,) * len(
            connection_config.dc_channel_names
        )
    else:
        if event_callback is not None:
            event_callback(
                "acquisition", "started", "Executing one synchronized QCS program"
            )
        if progress_callback is not None:
            progress_callback(
                35,
                (
                    "Running one native QCS hardware sweep"
                    if compiled.hardware_sweep
                    else "Running one QCS-managed software sweep"
                ),
            )
        applied_dc_offsets_v = tuple(compiled.dc_channel_offsets_v)
        dc_offset_init_compensation_v = tuple(
            compiled.dc_offset_init_compensation_v
        )
        offset_reset_required = any(
            not np.isclose(value, 0.0, rtol=0.0, atol=1e-15)
            for value in applied_dc_offsets_v
        )
        if cancellation is not None:
            cancellation.tag_program(compiled.program)
        latest_partial_execution = None
        try:
            effective_acquisition_duration_s = compiled.acquisition_duration_s
            effective_sample_rate_hz = compiled.acquisition_sample_rate_hz
            iq_weighted_sum = None
            raw_result = None
            for pass_index, pass_samples in enumerate(
                averaging_pass_counts
            ):
                if cancellation is not None:
                    cancellation.raise_if_requested(
                        "synchronized QCS program execution"
                    )
                raw_result = _executor_execute(executor, compiled.program)
                measurement_executor_call_count += 1
                if cancellation is not None:
                    cancellation.raise_if_requested(
                        "synchronized QCS program execution"
                    )
                if not connection_config.hw_demod:
                    executed_sample_rate = _executed_program_sample_rate(
                        raw_result,
                        compiled.acquisition_channels,
                    )
                    if executed_sample_rate is not None:
                        effective_sample_rate_hz = executed_sample_rate
                    values = extract_qcs_acquisition(
                        raw_result,
                        compiled.acquisition_channels,
                        prefer_trace=True,
                    )
                    if point_count == 1:
                        pass_iq = normalize_qcs_iq(
                            values,
                            repetitions_per_sweep=repetitions_per_sweep,
                            real_is_i_trace=True,
                        )[np.newaxis, ...]
                    else:
                        pass_iq = normalize_qcs_synchronized_trace(
                            values,
                            repetitions_per_point=repetitions_per_sweep,
                            sweep_shape=compiled.sweep_shape,
                        )
                else:
                    pass_iq = _extract_qcs_segmented_hardware_sweep_iq(
                        raw_result,
                        acquisition_channels=compiled.acquisition_channels,
                        segment_sample_counts=(
                            compiled.integration_segment_sample_counts
                            or (int(compiled.acquisition_sample_count or 1),)
                        ),
                        repetitions_per_point=repetitions_per_sweep,
                        sweep_shape=compiled.sweep_shape,
                        hardware_sweep=compiled.hardware_sweep,
                    )
                weighted = np.asarray(pass_iq) * float(
                    pass_samples
                )
                if iq_weighted_sum is None:
                    iq_weighted_sum = weighted
                else:
                    iq_weighted_sum += weighted
                completed_pass_count = pass_index + 1
                completed_sample_count = int(
                    sum(averaging_pass_counts[:completed_pass_count])
                )
                iq = iq_weighted_sum / float(completed_sample_count)
                if connection_config.hw_demod:
                    partial_ddr = FineTuneDdrResult(
                        sweep_points=np.asarray(sequence.sweep_points),
                        iq=iq.copy(),
                        sweep_axes=tuple(sequence.sweep_axes),
                        sweep_shape=tuple(sequence.sweep_shape),
                        cross_capacitance=np.asarray(
                            sequence.cross_capacitance
                        ).copy(),
                        sample_rate_hz=effective_sample_rate_hz,
                        fir_rate_profile="qcs_hardware_demod",
                    )
                    latest_partial_execution = QcsExecutionResult(
                        ddr_result=partial_ddr,
                        programs=(compiled.program,),
                        raw_results=(raw_result,),
                        program_summary={
                            "backend": "qcs",
                            "partial": (
                                completed_pass_count
                                < len(averaging_pass_counts)
                            ),
                            "completed_points": point_count,
                            "planned_points": point_count,
                            "iq_averaging_pass_count": len(
                                averaging_pass_counts
                            ),
                            "completed_iq_averaging_passes": (
                                completed_pass_count
                            ),
                            "effective_integration_duration_s": (
                                completed_sample_count
                                / effective_sample_rate_hz
                            ),
                            "iq_shape": list(iq.shape),
                        },
                        rf_settings={"backend": "qcs"},
                    )
                    if partial_callback is not None:
                        partial_callback(latest_partial_execution)
                if progress_callback is not None and len(
                    averaging_pass_counts
                ) > 1:
                    progress_callback(
                        35
                        + int(
                            35
                            * completed_pass_count
                            / len(averaging_pass_counts)
                        ),
                        "QCS integrated-I/Q pass "
                        f"{completed_pass_count:,}/"
                        f"{len(averaging_pass_counts):,} acquired",
                    )
            if iq_weighted_sum is None or raw_result is None:
                raise RuntimeError("QCS averaging completed no passes")
        except BaseException as execution_error:
            cancellation_requested = bool(
                cancellation is not None
                and cancellation.is_stop_requested()
            )
            reset_error_caught = None
            if offset_reset_required or cancellation_requested:
                try:
                    _reset_qcs_dc_outputs_to_zero(
                        qcs,
                        executor=executor,
                        mapper=mapper,
                        channel_names=connection_config.dc_channel_names,
                        fabric_hz=float(fabric_mhz) * 1e6,
                    )
                except Exception as reset_error:
                    reset_error_caught = reset_error
                    if hasattr(execution_error, "add_note"):
                        execution_error.add_note(
                            "Automatic QCS cancellation/reset also failed: "
                            f"{type(reset_error).__name__}: {reset_error}"
                        )
            if cancellation_requested and reset_error_caught is not None:
                raise RuntimeError(
                    "QCS stop was requested, but the emergency DC reset "
                    "failed; the physical output state is unknown"
                ) from reset_error_caught
            if cancellation_requested:
                raise QcsExperimentCancelled(
                    "QCS experiment stopped by user; completed averaging "
                    "passes remain available and DC outputs were reset to zero",
                    partial_result=latest_partial_execution,
                ) from execution_error
            raise
        if offset_reset_required:
            try:
                _reset_qcs_dc_outputs_to_zero(
                    qcs,
                    executor=executor,
                    mapper=mapper,
                    channel_names=connection_config.dc_channel_names,
                    fabric_hz=float(fabric_mhz) * 1e6,
                )
                safety_reset_executor_call_count = 1
            except Exception as reset_error:
                raise RuntimeError(
                    "QCS acquisition completed, but the automatic M5301 "
                    "fixed-offset reset failed"
                ) from reset_error
        close_cancellation_window(
            dc_already_reset=offset_reset_required,
            partial_result=latest_partial_execution,
        )
        programs = (compiled.program,)
        raw_results = (raw_result,)
        hardware_sweep = compiled.hardware_sweep
        compiled_sweep_shape = compiled.sweep_shape
        software_sweep_reasons = compiled.software_sweep_reasons
        sweep_variable_count = compiled.sweep_variable_count
        sweep_array_value_count = compiled.sweep_array_value_count

    if progress_callback is not None:
        progress_callback(70, f"Acquired {point_count:,} QCS point(s)")
    if iq.shape[0] != point_count or iq.shape[1] != int(
        repetitions_per_sweep
    ):
        raise ValueError(
            "QCS execution returned an unexpected point/repetition "
            f"shape {iq.shape[:2]}; expected "
            f"({point_count}, {int(repetitions_per_sweep)})"
        )
    ddr_result = FineTuneDdrResult(
        sweep_points=np.asarray(sequence.sweep_points),
        iq=iq,
        sweep_axes=tuple(sequence.sweep_axes),
        sweep_shape=tuple(sequence.sweep_shape),
        cross_capacitance=np.asarray(sequence.cross_capacitance).copy(),
        sample_rate_hz=effective_sample_rate_hz,
        fir_rate_profile=(
            "qcs_hardware_demod"
            if connection_config.hw_demod
            else "qcs_trace"
        ),
    )
    if event_callback is not None:
        reset_text = (
            " plus one fixed-offset safety reset"
            if safety_reset_executor_call_count
            else ""
        )
        event_callback(
            "acquisition",
            "completed",
            (
                f"Acquired {point_count:,} QCS point(s) with "
                f"{measurement_executor_call_count:,} measurement "
                "execution(s) through "
                f"one executor{reset_text}"
            ),
        )

    if fixed_numeric_dc_ramp:
        execution_mode = "software_fixed_numeric_dc_ramp"
    elif point_count <= 1:
        execution_mode = "single_point"
    elif hardware_sweep:
        execution_mode = "hardware_flattened"
    else:
        execution_mode = "software_single_program"
    summary = {
        "backend": "qcs",
        "hardware_sweep": hardware_sweep,
        "hardware_sweep_dimensions": 1 if hardware_sweep else 0,
        "hardware_sweep_shape": (
            [point_count] if hardware_sweep else []
        ),
        "semantic_sweep_shape": list(compiled_sweep_shape),
        "hardware_sweep_points": (
            point_count if hardware_sweep else 0
        ),
        "software_sweep_points": (
            point_count
            if point_count > 1 and not hardware_sweep
            else 0
        ),
        "sweep_execution_mode": execution_mode,
        "software_sweep_reasons": list(software_sweep_reasons),
        "sweep_variable_count": sweep_variable_count,
        "sweep_array_value_count": sweep_array_value_count,
        "repetitions_per_sweep": int(repetitions_per_sweep),
        "program_count": len(programs),
        "executor_call_count": measurement_executor_call_count,
        "retained_raw_result_count": len(raw_results),
        "safety_reset_executor_call_count": (
            safety_reset_executor_call_count
        ),
        "total_executor_call_count": (
            measurement_executor_call_count
            + safety_reset_executor_call_count
        ),
        "fabric_mhz": float(fabric_mhz),
        "source_full_scale_mv": float(source_full_scale_mv),
        "qcs_dc_full_scale_v": connection_config.dc_full_scale_v,
        "dc_channel_names": list(connection_config.dc_channel_names),
        "dc_channel_offsets_v": list(applied_dc_offsets_v),
        "dc_offset_init_compensation_v": list(
            dc_offset_init_compensation_v
        ),
        "fixed_dc_offset_residualization": any(
            not np.isclose(value, 0.0, rtol=0.0, atol=1e-15)
            for value in applied_dc_offsets_v
        ),
        "rf_channel_names": {
            str(key): value
            for key, value in connection_config.rf_channel_names.items()
        },
        "acquisition_channel_name": (
            connection_config.acquisition_channel_name
        ),
        "hw_demod": connection_config.hw_demod,
        "reset_phase_every_shot": True,
        "sample_rate_hz": effective_sample_rate_hz,
        "acquisition_duration_s": (
            effective_acquisition_duration_s
            if iq_averaging_plan is None
            else iq_averaging_plan.quantized_total_duration_s
        ),
        "requested_sample_count": requested_acquisition.sample_count,
        "requested_total_integration_duration_s": (
            None
            if iq_averaging_plan is None
            else iq_averaging_plan.requested_total_duration_s
        ),
        "quantized_total_integration_duration_s": (
            None
            if iq_averaging_plan is None
            else iq_averaging_plan.quantized_total_duration_s
        ),
        "quantized_total_integration_sample_count": (
            None
            if iq_averaging_plan is None
            else iq_averaging_plan.quantized_total_sample_count
        ),
        "iq_averaging_pass_count": len(averaging_pass_counts),
        "completed_iq_averaging_passes": len(averaging_pass_counts),
        "per_pass_integration_duration_s": effective_acquisition_duration_s,
        "per_pass_integration_sample_count": (
            None
            if iq_averaging_plan is None
            else iq_averaging_plan.pass_sample_counts[0]
        ),
        "effective_integration_duration_s": (
            None
            if iq_averaging_plan is None
            else sum(averaging_pass_counts) / effective_sample_rate_hz
        ),
        "iq_shape": list(iq.shape),
    }
    rf_settings = {
        "backend": "qcs",
        "output_details": _qcs_rf_output_details(rf_pulses),
        "readout_details": {
            "sample_rate_hz": effective_sample_rate_hz,
            "hw_demod": connection_config.hw_demod,
            "reset_phase_every_shot": True,
            "frequency_hz": acquisition.frequency_hz,
            "duration_s": (
                effective_acquisition_duration_s
                if iq_averaging_plan is None
                else iq_averaging_plan.quantized_total_duration_s
            ),
            "requested_sample_count": requested_acquisition.sample_count,
            "iq_averaging_pass_count": len(averaging_pass_counts),
            "per_pass_integration_duration_s": (
                effective_acquisition_duration_s
            ),
            "effective_integration_duration_s": (
                None
                if iq_averaging_plan is None
                else sum(averaging_pass_counts) / effective_sample_rate_hz
            ),
        },
    }
    return QcsExecutionResult(
        ddr_result=ddr_result,
        programs=programs,
        raw_results=raw_results,
        program_summary=summary,
        rf_settings=rf_settings,
    )


def run_qcs_qcodes_experiment(
    *,
    connection_config: QcsConnectionConfig,
    run_config: QcodesRunConfig,
    sequence: Any,
    repetitions_per_sweep: int,
    iq_repetition_policy: str = IQ_REPETITION_POLICY_PRESERVE,
    fabric_mhz: float = 300.0,
    source_full_scale_mv: float = DEFAULT_QICK_FULL_SCALE_MV,
    rf_pulses: Sequence[QcsRfPulseConfig] = (),
    acquisition: Optional[QcsAcquisitionConfig] = None,
    gui_settings: Optional[Mapping[str, Any]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    event_callback: Optional[EventCallback] = None,
    partial_callback: Optional[PartialResultCallback] = None,
    qcs_module=None,
    mapper=None,
    executor=None,
    cancellation: Optional[QcsCancellationController] = None,
) -> StoredQcsExperiment:
    """Execute QCS programs and persist their normalized I/Q arrays."""
    iq_repetition_policy = normalize_iq_repetition_policy(
        iq_repetition_policy
    )
    if (
        iq_repetition_policy == IQ_REPETITION_POLICY_COHERENT_AVERAGE
        and not connection_config.hw_demod
    ):
        raise ValueError(
            "coherent-average IQ repetition storage requires QCS Single "
            "I/Q acquisition (hardware demodulation)"
        )
    if event_callback is not None:
        event_callback(
            "experiment", "started", "Starting Keysight QCS experiment"
        )
    latest_partial_execution = None

    def publish_partial_execution(partial: QcsExecutionResult) -> None:
        nonlocal latest_partial_execution
        latest_partial_execution = partial
        if partial_callback is not None:
            partial_callback(partial)

    try:
        execution = execute_qcs_sequence(
            connection_config=connection_config,
            sequence=sequence,
            repetitions_per_sweep=repetitions_per_sweep,
            fabric_mhz=fabric_mhz,
            source_full_scale_mv=source_full_scale_mv,
            rf_pulses=rf_pulses,
            acquisition=acquisition,
            progress_callback=progress_callback,
            event_callback=event_callback,
            partial_callback=publish_partial_execution,
            qcs_module=qcs_module,
            mapper=mapper,
            executor=executor,
            cancellation=cancellation,
        )
        effective_run_config = replace(
            run_config,
            sample_rate_hz=(
                run_config.sample_rate_hz
                if acquisition is None
                else execution.ddr_result.sample_rate_hz
            ),
        )
        stored_gui_settings = dict(gui_settings or {})
        qick_settings = stored_gui_settings.get("qick", {})
        if not isinstance(qick_settings, Mapping):
            raise TypeError("gui_settings['qick'] must be a mapping")
        qick_settings = dict(qick_settings)
        full_scale_mv = _positive_finite(
            source_full_scale_mv, "source waveform full scale"
        )
        # The sequence and its sweep coordinates are normalized against this
        # scale.  Keep the persisted GUI snapshot authoritative even for
        # direct API callers that omit it or supply a stale value.
        qick_settings["full_scale_mv"] = full_scale_mv
        stored_gui_settings["qick"] = qick_settings
        metadata_mode = normalize_awg_metadata_mode(
            qick_settings.get(
                "awg_metadata_mode", DEFAULT_AWG_METADATA_MODE
            )
        )
        stored_gui_settings["awg_waveform_recipe"] = (
            build_awg_waveform_recipe(
                sequence,
                fabric_mhz=fabric_mhz,
                full_scale_mv=full_scale_mv,
            )
        )
        if metadata_mode == AWG_METADATA_MODE_EXPANDED:
            stored_gui_settings["awg_waveform_vertices"] = (
                build_awg_vertex_metadata(
                    sequence,
                    fabric_mhz=fabric_mhz,
                    full_scale_mv=full_scale_mv,
                )
            )
        if event_callback is not None:
            event_callback(
                "qcodes_save", "started", "Writing QCS result to QCoDeS"
            )
        dataset, row_count = store_experiment_result(
            execution.ddr_result,
            run_config=effective_run_config,
            connection_config=connection_config,
            program_summary=execution.program_summary,
            gui_settings=stored_gui_settings,
            rf_settings=execution.rf_settings,
            backend_name="qcs",
            iq_repetition_policy=iq_repetition_policy,
            progress_callback=progress_callback,
            progress_start=65,
            progress_end=99,
        )
        if event_callback is not None:
            event_callback(
                "qcodes_save",
                "completed",
                f"Saved QCoDeS Run {int(dataset.run_id)}",
            )
            event_callback(
                "experiment",
                "completed",
                f"Keysight QCS experiment saved as Run {int(dataset.run_id)}",
            )
        if progress_callback is not None:
            progress_callback(100, "Keysight QCS experiment complete")
        return StoredQcsExperiment(
            run_id=int(dataset.run_id),
            guid=str(dataset.guid),
            database_path=effective_run_config.resolved_database_path,
            row_count=int(row_count),
            dataset=dataset,
            program=execution.programs,
            ddr_result=execution.ddr_result,
            rf_settings=execution.rf_settings,
            programs=execution.programs,
            raw_results=execution.raw_results,
            program_summary=execution.program_summary,
        )
    except QcsExperimentCancelled as exc:
        # Persist every fully completed mixed block after Stop. This is not a
        # live per-iteration database transaction (the GUI receives those
        # blocks in memory immediately), but it guarantees that cancellation
        # does not discard measured data or leave it available only through a
        # Python object.
        partial = exc.partial_result or latest_partial_execution
        if partial is not None and int(
            partial.program_summary.get("completed_points", 0)
        ) > 0:
            try:
                effective_run_config = replace(
                    run_config,
                    sample_rate_hz=float(partial.ddr_result.sample_rate_hz),
                )
                stored_gui_settings = dict(gui_settings or {})
                qick_settings = stored_gui_settings.get("qick", {})
                if not isinstance(qick_settings, Mapping):
                    qick_settings = {}
                qick_settings = dict(qick_settings)
                full_scale_mv = _positive_finite(
                    source_full_scale_mv,
                    "source waveform full scale",
                )
                qick_settings["full_scale_mv"] = full_scale_mv
                stored_gui_settings["qick"] = qick_settings
                stored_gui_settings["awg_waveform_recipe"] = (
                    build_awg_waveform_recipe(
                        sequence,
                        fabric_mhz=fabric_mhz,
                        full_scale_mv=full_scale_mv,
                    )
                )
                stopped_summary = dict(partial.program_summary)
                stopped_summary.update({
                    "partial": True,
                    "execution_status": "stopped",
                    "stop_message": str(exc),
                })
                if event_callback is not None:
                    event_callback(
                        "qcodes_save",
                        "started",
                        "Saving completed mixed-sweep blocks after Stop",
                    )
                dataset, row_count = store_experiment_result(
                    partial.ddr_result,
                    run_config=effective_run_config,
                    connection_config=connection_config,
                    program_summary=stopped_summary,
                    gui_settings=stored_gui_settings,
                    rf_settings=partial.rf_settings,
                    backend_name="qcs",
                    iq_repetition_policy=iq_repetition_policy,
                    progress_callback=progress_callback,
                    progress_start=70,
                    progress_end=99,
                )
                stored_partial = StoredQcsExperiment(
                    run_id=int(dataset.run_id),
                    guid=str(dataset.guid),
                    database_path=effective_run_config.resolved_database_path,
                    row_count=int(row_count),
                    dataset=dataset,
                    program=partial.programs,
                    ddr_result=partial.ddr_result,
                    rf_settings=partial.rf_settings,
                    programs=partial.programs,
                    raw_results=partial.raw_results,
                    program_summary=stopped_summary,
                )
                exc.stored_result = stored_partial
                if partial_callback is not None:
                    partial_callback(stored_partial)
                if event_callback is not None:
                    event_callback(
                        "qcodes_save",
                        "completed",
                        f"Saved stopped QCoDeS Run {int(dataset.run_id)}",
                    )
            except Exception as storage_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "Completed QCS data remains available in memory, but "
                        "the stopped-run database save failed: "
                        f"{type(storage_error).__name__}: {storage_error}"
                    )
                if event_callback is not None:
                    event_callback(
                        "qcodes_save",
                        "failed",
                        (
                            "Could not save completed blocks after Stop; "
                            "in-memory data is retained: "
                            f"{storage_error}"
                        ),
                    )
        if event_callback is not None:
            event_callback("experiment", "stopped", str(exc))
        raise
    except Exception as exc:
        if event_callback is not None:
            event_callback("experiment", "failed", str(exc))
        raise


__all__ = [
    "DEFAULT_QCS_INIT_TIME_S",
    "MAX_QCS_HARDWARE_SWEEP_ARRAYS_PER_CHANNEL",
    "MAX_QCS_HARDWARE_SWEEP_ARRAY_VALUES",
    "MAX_QCS_STABILITY_GRID_POINTS",
    "MAX_QCS_STABILITY_RESULT_VALUES",
    "MAX_QCS_SOFTWARE_SWEEP_POINTS",
    "QCS_FABRIC_CLOCK_HZ",
    "QCS_M5200_INTEGRATION_BLOCK_SAMPLES",
    "QCS_M5200_MAX_SINGLE_INTEGRATION_DURATION_S",
    "QCS_M5200_MAX_SINGLE_INTEGRATION_SAMPLES",
    "QCS_M5200_SAMPLE_RATE_HZ",
    "QCS_MAX_TOTAL_IQ_AVERAGING_DURATION_S",
    "QCS_MAX_TOTAL_IQ_AVERAGING_SAMPLES",
    "QCS_NOISE_MAX_RAW_TRACE_SAMPLES",
    "QCS_SPARAMETER_INTER_SEGMENT_DELAY_S",
    "QCS_SPARAMETER_MAX_INTEGRATION_DURATION_S",
    "QCS_SPARAMETER_MAX_INTEGRATION_SAMPLES",
    "QCS_STABILITY_INTEGRATION_BLOCK_SAMPLES",
    "QCS_STABILITY_INTEGRATION_FABRIC_CYCLES",
    "QCS_STABILITY_INTEGRATION_QUANTUM_S",
    "QCS_STABILITY_INTER_SEGMENT_DELAY_S",
    "QCS_STABILITY_MAX_INTEGRATION_DURATION_S",
    "QCS_STABILITY_MAX_INTEGRATION_SAMPLES",
    "QCS_STABILITY_DC_EDGE_PADDING_FABRIC_CYCLES",
    "QCS_STABILITY_DC_EDGE_PADDING_S",
    "QCS_STABILITY_DC_RAMP_FABRIC_CYCLES",
    "QCS_STABILITY_DC_RAMP_S",
    "quantize_qcs_inter_iteration_delay",
    "QCS_M5301_MAX_RENDERED_FABRIC_CYCLES",
    "QCS_M5301_MAX_RENDERED_SAMPLES",
    "QCS_M5301_SAMPLES_PER_FABRIC_CYCLE",
    "QcsAcquisitionConfig",
    "QcsCancellationController",
    "QcsCompiledHardwareSweep",
    "QcsCompiledPoint",
    "QcsCompiledSParameterSweep",
    "QcsConnectionConfig",
    "QcsExperimentCancelled",
    "QcsExecutionResult",
    "QcsIqAveragingPlan",
    "QcsM5301CapacityReport",
    "QcsM5301ChannelCapacity",
    "QcsNoiseTraceConfig",
    "QcsNoiseTraceResult",
    "QcsRfPowerCalibrationConfig",
    "QcsRfPulseConfig",
    "QcsSParameterExecutionResult",
    "QcsSweepExecutionPreview",
    "QcsUnsupportedFeatureError",
    "StoredQcsExperiment",
    "acquire_qcs_noise_trace",
    "build_qcs_executor",
    "compile_qcs_point",
    "compile_qcs_sequence",
    "compile_qcs_sparameter_sweep",
    "compile_qcs_synchronized_sweep",
    "compile_qcs_stability_hardware_sweep",
    "execute_qcs_sequence",
    "execute_qcs_sparameter_sweep",
    "execute_qcs_stability_hardware_sweep",
    "extract_qcs_acquisition",
    "load_qcs_channel_mapper",
    "normalize_qcs_hardware_sweep_iq",
    "normalize_qcs_iq",
    "normalize_qcs_synchronized_trace",
    "plan_qcs_total_iq_averaging",
    "quantize_qcs_raw_trace_duration",
    "quantize_qcs_stability_integration_duration",
    "qcs_stability_integration_segment_sample_counts",
    "qcs_m5301_capacity_preview_point_indices",
    "qcs_m5301_waveform_capacity_report",
    "qcs_sweep_execution_preview",
    "run_qcs_qcodes_experiment",
    "run_qcs_sparameter_sweep",
    "resolve_qcs_rf_power_calibrations",
    "validate_qcs_capabilities",
    "validate_qcs_m5301_waveform_capacity",
]
