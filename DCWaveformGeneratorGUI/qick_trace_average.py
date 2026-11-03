"""Client contract for board-side repetition-averaged FIR traces."""

import numpy as np


REPETITION_MODES = ("raw", "qick_mean", "qick_mean_iq")


def normalize_repetition_mode(value):
    mode = str(value or "raw").strip().lower()
    if mode not in REPETITION_MODES:
        raise ValueError("repetition_mode must be raw, qick_mean or qick_mean_iq")
    return mode


def require_board_trace_average(soc):
    capabilities = getattr(soc, "get_fir_average_capabilities", None)
    if not callable(getattr(soc, "get_ddr4_fir_mean", None)) or not callable(capabilities):
        raise RuntimeError(
            "QICK trace averaging requires an updated board QSTL_QICK Python "
            "library and a restarted Pyro server (get_ddr4_fir_mean is missing)."
        )
    info = capabilities()
    if (not isinstance(info, dict) or info.get("api_version") != 1
            or not {"repetition_mean", "point_mean"}.issubset(info.get("modes", ()))
            or info.get("units") != "input_code" or info.get("dtype") != "float64"):
        raise RuntimeError("Incompatible QICK averaging API. Update QSTL_QICK and restart the Pyro server and GUI.")


def read_board_mean(soc, *, n_points, repetitions, n_samples, base_word,
                    stride_words, stride_bytes, source_scale_log2,
                    source_component_bits, chunk_triggers, cancel_check,
                    progress=None, reduce_samples=False):
    """Request complete points, with a bounded reply, never split rep groups."""
    points_per_chunk = max(1, min(
        chunk_triggers // repetitions,
        (8 * 1024 * 1024) // (n_samples * 2 * 8),
    ))
    stored_samples = 1 if reduce_samples else n_samples
    result = np.empty((n_points, 1, stored_samples, 2), dtype=np.float64)
    if progress is not None:
        progress(0, n_points * repetitions)
    for first in range(0, n_points, points_per_chunk):
        cancel_check()
        count = min(points_per_chunk, n_points - first)
        reply = soc.get_ddr4_fir_mean(
            n_samples=n_samples, repetitions=repetitions, n_points=count,
            start=base_word + first * repetitions * stride_words,
            stride_bytes=stride_bytes,
            reduce_samples=reduce_samples,
        )
        cancel_check()
        expected_metadata = dict(
            data_kind="point_mean" if reduce_samples else "repetition_mean",
            accumulation_repetitions=repetitions,
            source_samples_per_trace=n_samples,
            iq_scale_log2=0, source_iq_scale_log2=source_scale_log2,
            source_iq_component_bits=source_component_bits,
        )
        if not isinstance(reply, dict) or any(
            reply.get(key) != value for key, value in expected_metadata.items()
        ):
            raise RuntimeError("QICK returned incompatible mean-trace metadata")
        iq = np.asarray(reply.get("iq"))
        if iq.shape != (count, stored_samples, 2) or iq.dtype != np.dtype("float64"):
            raise RuntimeError("QICK returned an invalid mean-trace shape or dtype")
        if not np.all(np.isfinite(iq)):
            raise RuntimeError("QICK returned non-finite mean-trace samples")
        result[first:first + count, 0] = iq
        if progress is not None:
            progress((first + count) * repetitions, n_points * repetitions)
    return result
