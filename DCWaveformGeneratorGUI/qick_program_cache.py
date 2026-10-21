"""Local, bounded cache of compiled programs, never of acquired measurements.

Artifacts use a restricted JSON object graph rather than pickle. Each hit is
decoded into a new program, including its manager back-references. Hardware
configuration, memory uploads, acquisition and database storage still run.

Enabled by default for the shared Show/Run compiler. Artifacts survive app
restarts under LOCALAPPDATA/PulseGenerator/compiled_program_cache/v1, outside
the repository. QSTL_QICK_COMPILE_CACHE_DIR overrides that location and
QSTL_QICK_COMPILE_CACHE=0 disables reuse. Older entries are evicted at 64
files or 512 MiB; incomplete, corrupt and unsupported entries recompile.
Inputs include firmware metadata, sequences, calibration-derived RF gains,
sweep grouping, readout settings, repetitions and compiler source hashes.
Generated sweep identifiers are metadata, not compiler inputs: group membership
is keyed by axis, and cached metadata is rebound to the requesting GUI's IDs.
"""
from __future__ import annotations

import base64
from collections import OrderedDict
from fractions import Fraction
from functools import lru_cache
import gzip
import hashlib
import importlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import threading
import zlib

import numpy as np

SCHEMA = 1
CACHE_DIR_ENV = "QSTL_QICK_COMPILE_CACHE_DIR"
CACHE_ENABLED_ENV = "QSTL_QICK_COMPILE_CACHE"
MAX_ENTRY_BYTES = 128 * 1024 * 1024
MAX_DISK_BYTES = 512 * 1024 * 1024
MAX_ENTRIES = 64
MAX_MEMORY_BYTES = 64 * 1024 * 1024
_memory = OrderedDict()
_lock = threading.RLock()
_log = logging.getLogger(__name__)


class Uncacheable(ValueError):
    pass


def _local(name):
    return importlib.import_module(f"{__package__}.{name}" if __package__ else name)


@lru_cache(maxsize=1)
def _classes():
    """Only these data/program classes may be reconstructed; no dynamic imports."""
    modules = {
        _local("qick_fine_tune_sweep"): (
            "FineTuneSequence FineTuneAmplitudeSweepProgram PulseSegment "
            "AmplitudeSweep RfDurationSweep RfSegmentLengthExtension "
            "RfFrequencySweep RfPowerSweep RampDurationSweep HoldDurationSweep CpmgNSweep "
            "BiasTCompensationConfig RCCompensationConfig ReadoutConfig "
            "RfPulseConfig DdrFirReadoutConfig CompiledCommand CompiledPoint"
        ),
        _local("qick_square_dds"): "SquarePulseSweep SquarePulseConfig OutputTriggerConfig",
        _local("qick_rf_fine_duration"): "FineDurationPlan SampleAccumulator",
        importlib.import_module("qick.qick_asm"): "QickConfig",
        importlib.import_module("qick.asm_v1"): (
            "AwgTuningGenManager SquarePulseGenManager FullSpeedGenManager "
            "InterpolatedGenManager MultiplexedGenManager ReadoutManager QickRegister"
        ),
    }
    return {f"{cls.__module__}.{cls.__name__}": cls
            for module, names in modules.items()
            for cls in (getattr(module, name) for name in names.split())}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _state(value, *, key=False):
    state = dict(vars(value))
    for name in list(state):
        if name in ("cancel_check", "compile_cache_info") or name.startswith("_compile_cache_"):
            state.pop(name, None)
    if key:
        state.pop("_sweep_coordinate_cache", None)
        if type(value) is _local("qick_fine_tune_sweep").FineTuneSequence:
            state.pop("sweep_parameter_ids", None)
            # Preserve which axes share a loop, without depending on randomly
            # generated group names or dictionary insertion order.
            groups = state.get("_sweep_group_ids", {})
            labels = {}
            state["_sweep_group_ids"] = {
                axis: labels.setdefault(groups[axis], len(labels))
                for axis in sorted(groups, key=lambda k: _json(_semantic(k)))
            }
    return state


def _semantic(value):
    """Value-based key: dict insertion order and previously evaluated previews do not matter."""
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, np.generic):
        return _semantic(value.item())
    if type(value) is float:
        return ["float", value.hex()]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise Uncacheable("object array")
        return ["array", value.dtype.str, list(value.shape),
                base64.b64encode(value.tobytes(order="C")).decode("ascii")]
    if isinstance(value, dict):
        pairs = [(_semantic(k), _semantic(v)) for k, v in value.items()]
        return ["dict", sorted(pairs, key=lambda pair: _json(pair[0]))]
    if type(value) in (list, tuple):
        return [type(value).__name__, [_semantic(v) for v in value]]
    if type(value) in (set, frozenset):
        return [type(value).__name__, sorted((_semantic(v) for v in value), key=_json)]
    if type(value) is Fraction:
        return ["fraction", value.numerator, value.denominator]
    if type(value) is range:
        return ["range", value.start, value.stop, value.step]
    name = f"{type(value).__module__}.{type(value).__name__}"
    if _classes().get(name) is type(value):
        return ["object", name, _semantic(_state(value, key=True))]
    raise Uncacheable(f"unsupported key value: {type(value).__name__}")


def _source_fingerprint():
    # Include every top-level GUI/compiler Python module and the full QICK
    # package. Hash bytes, not mtimes, so code edits invalidate even with restored timestamps.
    roots = {Path(__file__).resolve().parent,
             Path(_local("dc_waveform_core").__file__).resolve().parent}
    qick_spec = importlib.util.find_spec("qick")
    if qick_spec is None or qick_spec.origin is None:
        raise Uncacheable("QICK package source is unavailable")
    qick_root = Path(qick_spec.origin).resolve().parent
    files = {p for root in roots for p in root.glob("*.py")}
    files.update(qick_root.rglob("*.py"))
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(str(path).encode("utf8")); h.update(b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


# Do not publish artifacts from an old running GUI under newer on-disk code.
try:
    _loaded_source_fingerprint = _source_fingerprint()
except (OSError, ImportError, ValueError):
    _loaded_source_fingerprint = None


def _key(soccfg, sequence, kwargs):
    cfg = soccfg.get_cfg() if hasattr(soccfg, "get_cfg") else soccfg
    arguments = {k: v for k, v in kwargs.items() if k != "cancel_check"}
    sources = _source_fingerprint()
    if sources != _loaded_source_fingerprint or sources is None:
        raise Uncacheable("compiler sources changed since import; restart to cache")
    value = [SCHEMA, platform.python_implementation(), list(sys.version_info[:2]),
             np.__version__, sources,
             _semantic(cfg), _semantic(sequence), _semantic(arguments)]
    return hashlib.sha256(_json(value)).hexdigest()


def _encode(root):
    nodes, seen = [], {}

    def encode(value):
        if value is None or type(value) in (bool, int, str):
            return value
        if isinstance(value, np.generic):
            return encode(value.item())
        if type(value) is float:
            return {"float": value.hex()}
        identity = id(value)
        if identity in seen:
            return {"ref": seen[identity]}
        index = len(nodes); seen[identity] = index; nodes.append(None)
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise Uncacheable("object array")
            node = ["array", value.dtype.str, list(value.shape),
                    base64.b64encode(value.tobytes(order="C")).decode("ascii")]
        elif type(value) in (dict, OrderedDict):
            node = [type(value).__name__, [[encode(k), encode(v)] for k, v in value.items()]]
        elif type(value) in (list, tuple, set, frozenset):
            node = [type(value).__name__, [encode(v) for v in value]]
        elif type(value) is range:
            node = ["range", value.start, value.stop, value.step]
        elif type(value) is Fraction:
            node = ["fraction", value.numerator, value.denominator]
        else:
            name = f"{type(value).__module__}.{type(value).__name__}"
            if _classes().get(name) is not type(value):
                raise Uncacheable(f"unsupported program value: {type(value).__name__}")
            node = ["object", name, {k: encode(v) for k, v in _state(value).items()}]
        nodes[index] = node
        return {"ref": index}

    return {"root": encode(root), "nodes": nodes}


def _decode(graph):
    nodes, decoded = graph["nodes"], {}

    def decode(value):
        if not isinstance(value, dict):
            if value is None or type(value) in (bool, int, str):
                return value
            raise Uncacheable("invalid scalar")
        if set(value) == {"float"}:
            return float.fromhex(value["float"])
        if set(value) != {"ref"} or type(value["ref"]) is not int:
            raise Uncacheable("invalid reference")
        index = value["ref"]
        if index in decoded:
            return decoded[index]
        if not 0 <= index < len(nodes):
            raise Uncacheable("reference out of bounds")
        node = nodes[index]; kind = node[0]
        if kind == "object":
            cls = _classes().get(node[1])
            if cls is None:
                raise Uncacheable("unknown class")
            result = object.__new__(cls); decoded[index] = result
            result.__dict__.update({k: decode(v) for k, v in node[2].items()})
        elif kind in ("dict", "OrderedDict"):
            result = {} if kind == "dict" else OrderedDict(); decoded[index] = result
            result.update((decode(k), decode(v)) for k, v in node[1])
        elif kind == "list":
            result = []; decoded[index] = result
            result.extend(decode(v) for v in node[1])
        elif kind in ("tuple", "set", "frozenset"):
            ctor = {"tuple": tuple, "set": set, "frozenset": frozenset}[kind]
            result = ctor(decode(v) for v in node[1])
        elif kind == "range":
            result = range(*node[1:])
        elif kind == "fraction":
            result = Fraction(*node[1:])
        elif kind == "array":
            dtype = np.dtype(node[1]); shape = tuple(node[2])
            if dtype.hasobject or any(type(n) is not int or n < 0 for n in shape):
                raise Uncacheable("invalid array")
            data = base64.b64decode(node[3], validate=True)
            result = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        else:
            raise Uncacheable("unknown node type")
        decoded[index] = result
        return result

    return decode(graph["root"])


def cache_directory():
    if os.environ.get(CACHE_DIR_ENV):
        return Path(os.environ[CACHE_DIR_ENV]).expanduser().resolve()
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
    return base / "PulseGenerator" / "compiled_program_cache" / f"v{SCHEMA}"


def clear_memory_cache():
    with _lock:
        _memory.clear()


def clear_program_cache():
    """Remove compiled artifacts and this process's memory copies.

    Call while the GUI is idle. Only cache-key files in the configured schema
    directory are removed; other files and subdirectories are left alone.
    Report individual filesystem failures so the GUI cannot claim a full clear
    when an entry is locked. Other running processes retain their own memory.
    """
    with _lock:
        memory_entries = len(_memory)
        _memory.clear()
        directory = cache_directory()
        result = {"directory": str(directory), "files_removed": 0,
                  "memory_entries_removed": memory_entries, "errors": []}
        try:
            paths = list(directory.iterdir())
        except FileNotFoundError:
            return result
        except OSError as exc:
            result["errors"].append(f"{directory}: {exc}")
            return result
        for path in paths:
            if re.fullmatch(r"[0-9a-f]{64}\.json\.gz", path.name) is None:
                continue
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
                    result["files_removed"] += 1
            except FileNotFoundError:
                pass
            except OSError as exc:
                result["errors"].append(f"{path}: {exc}")
        return result


def _remember(token, data):
    with _lock:
        _memory[token] = data; _memory.move_to_end(token)
        while _memory and sum(map(len, _memory.values())) > MAX_MEMORY_BYTES:
            _memory.popitem(last=False)


def _read(key):
    path = cache_directory() / f"{key}.json.gz"; token = str(path)
    with _lock:
        data = _memory.get(token)
        if data is not None:
            _memory.move_to_end(token)
    source = "memory" if data is not None else "disk"
    if data is None:
        if path.stat().st_size > MAX_ENTRY_BYTES:
            raise Uncacheable("oversize entry")
        data = path.read_bytes()
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
        raw = f.read(MAX_ENTRY_BYTES + 1)
    if len(raw) > MAX_ENTRY_BYTES:
        raise Uncacheable("oversize expanded entry")
    envelope = json.loads(raw)
    if envelope["schema"] != SCHEMA or envelope["key"] != key:
        raise Uncacheable("entry key mismatch")
    payload = envelope["program"]
    if hashlib.sha256(_json(payload)).hexdigest() != envelope["sha256"]:
        raise Uncacheable("entry checksum mismatch")
    program = _decode(payload)
    if type(program) is not _local("qick_fine_tune_sweep").FineTuneAmplitudeSweepProgram:
        raise Uncacheable("invalid program type")
    if program.binprog is None:
        raise Uncacheable("uncompiled artifact")
    _remember(token, data)
    try:
        path.touch()
    except OSError:
        pass
    return program, source


def _write(key, program):
    if getattr(program, "acquire_params", None) is not None:
        raise Uncacheable("acquisition has already started")
    payload = _encode(program)
    raw = _json({"schema": SCHEMA, "key": key, "program": payload,
                 "sha256": hashlib.sha256(_json(payload)).hexdigest()})
    if len(raw) > MAX_ENTRY_BYTES:
        raise Uncacheable("program exceeds cache size limit")
    data = gzip.compress(raw, compresslevel=1, mtime=0)
    directory = cache_directory(); directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.json.gz"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as f:
            tmp = Path(f.name); f.write(data)
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
    _remember(str(path), data)
    files = sorted(directory.glob("*.json.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
    size = 0
    for i, old in enumerate(files):
        size += old.stat().st_size
        if i >= MAX_ENTRIES or size > MAX_DISK_BYTES:
            if old.resolve().parent != directory.resolve():
                continue
            old.unlink()
            with _lock:
                _memory.pop(str(old), None)


_CACHE_ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError,
                 OverflowError, EOFError, RecursionError, AttributeError, zlib.error)


def _bind_sweep_metadata(program, requested):
    """Keep this hit's GUI/storage identities without changing compiled values.

    _read returns an independent object graph. Modify that graph only, retaining
    its already computed coordinate arrays and compiler/manager back-references.
    """
    sequence = program.sequence
    sequence.sweep_parameter_ids = dict(getattr(requested, "sweep_parameter_ids", {}))
    sequence._sweep_group_ids = dict(requested._sweep_group_ids)
    snapshot = getattr(program, "_execution_coordinate_snapshot", None)
    if snapshot is not None:
        for member in snapshot.get("requested_members", ()):
            index = member["member_index"]
            if tuple(member["key"]) != sequence.sweep_axis_key(sequence.sweeps[index]):
                raise Uncacheable("cached sweep coordinate axis mismatch")
            member["parameter_id"] = sequence.sweep_member_ids[index]
        group_ids = {g["representative_member_index"]: g["group_id"]
                     for g in sequence.sweep_groups}
        for group in snapshot.get("sweep_groups", ()):
            group["group_id"] = group_ids[group["representative_member_index"]]


def build_cached_program(soccfg, sequence, kwargs):
    """Cache only standard FineTuneSequence programs; custom builders still work."""
    cancel = kwargs.get("cancel_check")
    if cancel is not None:
        cancel()
    standard = type(sequence) is _local("qick_fine_tune_sweep").FineTuneSequence
    enabled = os.environ.get(CACHE_ENABLED_ENV, "1").lower() not in ("0", "false", "off")
    if not enabled or not standard:
        return sequence.make_program(soccfg, **kwargs)
    key = None; reason = "not_found"
    try:
        key = _key(soccfg, sequence, kwargs)
        program, source = _read(key)
        _bind_sweep_metadata(program, sequence)
    except FileNotFoundError:
        pass
    except _CACHE_ERRORS as exc:
        reason = type(exc).__name__
        _log.debug("Compiled program cache miss: %s", exc)
    else:
        program.cancel_check = cancel
        if cancel is not None:
            cancel()
        program._compile_cache_key = key
        program._compile_cache_coordinates_saved = getattr(
            program, "_execution_coordinate_snapshot", None) is not None
        program.compile_cache_info = {"status": "hit", "source": source, "key": key}
        return program
    # Compilation/validation/cancellation exceptions must propagate and never
    # become reusable cache entries.
    program = sequence.make_program(soccfg, **kwargs)
    if cancel is not None:
        cancel()
    if key is not None:
        try:
            if program.binprog is None:
                program.compile()
            _write(key, program)
        except _CACHE_ERRORS as exc:
            reason = type(exc).__name__
            _log.debug("Could not save compiled program: %s", exc)
        else:
            program._compile_cache_key = key
            program.compile_cache_info = {"status": "miss", "saved": True, "key": key}
            return program
    program.compile_cache_info = {"status": "bypass", "reason": reason}
    return program


def save_execution_snapshot(program):
    """Add pre-acquisition command coordinates, without retaining any run data."""
    key = getattr(program, "_compile_cache_key", None)
    if (key is None or getattr(program, "_execution_coordinate_snapshot", None) is None
            or getattr(program, "_compile_cache_coordinates_saved", False)):
        return
    try:
        _write(key, program)
        program._compile_cache_coordinates_saved = True
    except _CACHE_ERRORS as exc:
        _log.debug("Could not cache execution coordinates: %s", exc)
