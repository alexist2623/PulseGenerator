"""Editable Keysight QCS M5000 front-panel and ChannelMapper builder.

The serialized ``.qcs`` mapper is intentionally treated as opaque.  This
module keeps a small, JSON-compatible builder recipe for the GUI and always
uses the installed Keysight QCS API to load or save the real mapper.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
from functools import lru_cache
from math import isfinite
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from .qcs_chassis_renderer import (
        qcs_chassis_connector_at_point,
        qcs_chassis_slot_at_point,
        render_qcs_chassis_png as _render_qcs_chassis_png,
    )
except ImportError:
    try:
        from qcs_chassis_renderer import (
            qcs_chassis_connector_at_point,
            qcs_chassis_slot_at_point,
            render_qcs_chassis_png as _render_qcs_chassis_png,
        )
    except ImportError:
        _render_qcs_chassis_png = None

        def qcs_chassis_slot_at_point(_x, _y):
            return None

        def qcs_chassis_connector_at_point(
            _configuration,
            _x,
            _y,
            *,
            role=None,
        ):
            del role
            return None


QCS_HARDWARE_CONFIGURATION_VERSION = 1
QCS_HARDWARE_STATE_EXTERNAL = "external"
QCS_HARDWARE_STATE_DRAFT = "draft"
QCS_HARDWARE_STATE_SAVED = "saved"
QCS_HARDWARE_STATE_IMPORTED = "imported"
QCS_HARDWARE_STATE_IMPORTED_DIRTY = "imported_dirty"
QCS_HARDWARE_STATES = (
    QCS_HARDWARE_STATE_EXTERNAL,
    QCS_HARDWARE_STATE_DRAFT,
    QCS_HARDWARE_STATE_SAVED,
    QCS_HARDWARE_STATE_IMPORTED,
    QCS_HARDWARE_STATE_IMPORTED_DIRTY,
)
QCS_CHASSIS_MODEL = "M9046A"
QCS_CHASSIS_SLOT_COUNT = 18
DEFAULT_QCS_IP_ADDRESS = "192.168.2.105"
QCS_HARDWARE_DISCOVERY_TIMEOUT_S = 8.0
QCS_FRONT_PANEL_IMAGE_PATH = (
    Path(__file__).resolve().parent / "assets" / "qcs_front_panel_mockup.png"
)

QCS_MODULE_MODELS = {
    "M9032A": {
        "label": "M9032A System Sync",
        "instrument": None,
        "span": 1,
        "channels": 0,
        "purpose": "System timing and synchronization",
    },
    "M5301A": {
        "label": "M5301A Baseband AWG",
        "instrument": "M5301AWG",
        "span": 1,
        "channels": 4,
        "purpose": "DC/baseband waveform outputs",
    },
    "M5300A": {
        "label": "M5300A RF AWG",
        "instrument": "M5300AWG",
        "span": 2,
        "channels": 4,
        "purpose": "RF waveform outputs (two-slot module)",
    },
    "M5200A": {
        "label": "M5200A Digitizer",
        "instrument": "M5200Digitizer",
        "span": 1,
        "channels": 4,
        "purpose": "Digitizer acquisition inputs",
    },
    "M5201A": {
        "label": "M5201A Down Converter",
        "instrument": "M5201Downconverter",
        "span": 1,
        "channels": 4,
        "purpose": "Four RF-input/IF-output downconverter channel pairs",
    },
}

QCS_ROLE_LABELS = {
    "dc": "DC output",
    "rf": "RF generator",
    "acquisition": "Acquisition",
    "unassigned": "Unassigned",
}
QCS_ROLE_INSTRUMENTS = {
    "dc": {"M5301AWG"},
    "rf": {"M5300AWG", "M5301AWG"},
    "acquisition": {"M5200Digitizer"},
    "unassigned": {"M5300AWG", "M5301AWG", "M5200Digitizer"},
}


def _qcs_mapping_display_name(role: str, logical_index: int) -> str:
    """Return a user-facing name without exposing zero-based AWG numbering."""

    if role == "dc":
        return f"AWG output {int(logical_index) + 1}"
    if role == "rf":
        return f"RF generator {int(logical_index)}"
    if role == "acquisition":
        return "acquisition input"
    return f"{QCS_ROLE_LABELS.get(role, role)} {int(logical_index)}"


_ROLE_SORT_ORDER = {"dc": 0, "rf": 1, "acquisition": 2, "unassigned": 3}
_INSTRUMENT_TO_MODEL = {
    str(spec["instrument"]): model
    for model, spec in QCS_MODULE_MODELS.items()
    if spec["instrument"] is not None
}
_VIRTUAL_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DEFAULT_MODULES = (
    {"slot": 1, "model": "M9032A"},
    {"slot": 2, "model": "M5301A"},
    {"slot": 3, "model": "M5300A"},
    {"slot": 5, "model": "M5200A"},
)
_QCS_MODEL_NUMBER_RE = re.compile(r"\bM\d{4}[A-Z]?\b", re.IGNORECASE)
_QCS_MODEL_ALIASES = {
    "M9032": "M9032A",
    "M5200": "M5200A",
    "M5201": "M5201A",
    "M5300": "M5300A",
    "M5301": "M5301A",
}


def normalize_qcs_server_ip(value: Any) -> str:
    """Return a canonical IP address suitable for QCS remote discovery."""

    text = str(value).strip()
    if not text:
        raise ValueError("QCS server IP address must not be empty")
    try:
        return str(ipaddress.ip_address(text))
    except ValueError as exc:
        raise ValueError(
            f"invalid QCS server IP address {text!r}"
        ) from exc


def _saved_qcs_access_token(
    ip_address: str,
    *,
    token_path: Optional[Path] = None,
) -> Optional[str]:
    """Read QCS's existing per-host token without logging or modifying it."""

    path = (
        Path.home() / ".qcs_token.json"
        if token_path is None
        else Path(token_path).expanduser()
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    token = payload.get(ip_address)
    if not isinstance(token, str) or not token:
        return None
    return token


def _qcs_response_case_name(response: Any) -> str:
    case = getattr(response, "ResponseCase", None)
    if case is None:
        return ""
    to_string = getattr(case, "ToString", None)
    if callable(to_string):
        try:
            return str(to_string())
        except Exception:
            pass
    return str(case)


def _validate_qcs_inventory_response(system_hardware_info: Any) -> None:
    response = getattr(system_hardware_info, "Response", None)
    if response is None:
        return
    case_name = _qcs_response_case_name(response).strip()
    if case_name.casefold() == "error" or case_name == "2":
        error = getattr(response, "Error", None)
        code = str(getattr(error, "Code", "Unknown"))
        message = str(
            getattr(error, "Message", "QCS inventory request failed")
        )
        if "token" in code.casefold() or "auth" in message.casefold():
            raise RuntimeError(
                "QCS authentication is missing or expired. Log in to the "
                "QCS server once from the qcs conda environment, then retry "
                "hardware identification."
            )
        raise RuntimeError(
            f"QCS hardware inventory failed ({code}): {message}"
        )
    if case_name.casefold() in {"none", "none_", "0"}:
        raise RuntimeError(
            "QCS hardware inventory returned no success response"
        )


def _qcs_inventory_rpc(client: Any, timeout_s: float) -> Any:
    """Call the generated inventory RPC with a deadline when available."""

    system_client = getattr(client, "SystemClient", None)
    if system_client is None:
        method = getattr(client, "GetSystemHardwareInfo", None)
        if not callable(method):
            raise RuntimeError(
                "the installed Keysight QCS client has no hardware inventory "
                "service"
            )
        return method()

    try:
        from Google.Protobuf.WellKnownTypes import Empty
        from System import DateTime
        from System.Threading import CancellationTokenSource
    except ImportError as exc:
        raise RuntimeError(
            "Keysight QCS hardware discovery requires keysight-qcs 2.5.5 "
            "and keysight-qcs-common"
        ) from exc

    cancellation = CancellationTokenSource()
    try:
        deadline = DateTime.UtcNow.AddSeconds(float(timeout_s))
        return system_client.GetSystemHardwareInfo(
            Empty(),
            None,
            deadline,
            cancellation.Token,
        )
    finally:
        cancellation.Dispose()


def _dispose_qcs_inventory_client(client: Any) -> None:
    if client is None:
        return
    channel = getattr(client, "GrpcChannel", None)
    dispose = getattr(channel, "Dispose", None)
    if callable(dispose):
        try:
            dispose()
        except Exception:
            pass


def _qcs_discovery_error(ip_address: str, exc: Exception) -> RuntimeError:
    status = getattr(exc, "Status", None)
    status_code = str(getattr(status, "StatusCode", ""))
    detail = str(getattr(status, "Detail", "")).strip()
    combined = " ".join(
        part for part in (status_code, detail, str(exc)) if part
    ).casefold()
    if any(
        marker in combined
        for marker in (
            "unauthenticated",
            "permissiondenied",
            "permission denied",
            "http status code: 401",
            "http status code: 403",
            "invalid or expired token",
            "invalidtoken",
        )
    ):
        return RuntimeError(
            f"QCS authentication for {ip_address} is missing, expired, or "
            "does not permit hardware inventory. Log in once from the qcs "
            "conda environment, then click Identify Hardware Configuration "
            "again."
        )
    if "deadlineexceeded" in combined or "deadline exceeded" in combined:
        return RuntimeError(
            f"QCS hardware identification at {ip_address} timed out"
        )
    if "unavailable" in combined or "connection refused" in combined:
        return RuntimeError(
            f"QCS hardware service at {ip_address} is unavailable"
        )
    return RuntimeError(
        f"unable to identify QCS hardware at {ip_address}: "
        f"{detail or str(exc)}"
    )


def _canonical_qcs_model_number(value: Any) -> str:
    text = str(value).strip().upper()
    match = _QCS_MODEL_NUMBER_RE.search(text)
    if match is None:
        return text
    model = match.group(0).upper()
    return _QCS_MODEL_ALIASES.get(model, model)


def identify_qcs_hardware_configuration(
    ip_address: str,
    *,
    timeout_s: float = QCS_HARDWARE_DISCOVERY_TIMEOUT_S,
    client_factory=None,
    token_path: Optional[Path] = None,
) -> dict:
    """Read installed QCS modules from the controller's system service.

    The inventory endpoint is provided by ``keysight-qcs-common`` rather than
    the documented ChannelMapper API, so it is feature-detected and isolated
    here.  This function never logs in, prompts for credentials, or changes
    hardware state.
    """

    address = normalize_qcs_server_ip(ip_address)
    timeout_s = float(timeout_s)
    if not isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("QCS hardware discovery timeout must be positive")

    client = None
    try:
        if client_factory is None:
            try:
                import keysight.qcs_common as qcs_common
            except ImportError as exc:
                raise RuntimeError(
                    "QCS hardware identification requires keysight-qcs "
                    "2.5.5 in the active Python environment"
                ) from exc
            client_factory = qcs_common.HclApiClient
        client = client_factory(address)
        token = _saved_qcs_access_token(
            address,
            token_path=token_path,
        )
        if token is not None:
            set_access_token = getattr(client, "SetAccessToken", None)
            if callable(set_access_token):
                set_access_token(token)
        system_hardware_info = _qcs_inventory_rpc(client, timeout_s)
    except (ImportError, OSError, TypeError, ValueError):
        raise
    except Exception as exc:
        raise _qcs_discovery_error(address, exc) from exc
    finally:
        _dispose_qcs_inventory_client(client)

    _validate_qcs_inventory_response(system_hardware_info)
    module_hardware_infos = getattr(
        system_hardware_info,
        "ModuleHardwareInfos",
        None,
    )
    if module_hardware_infos is None:
        raise RuntimeError(
            "QCS hardware inventory did not include module information"
        )

    inventories: dict[tuple[int, int], dict] = {}
    for hardware_info in module_hardware_infos:
        module_info = getattr(hardware_info, "ModuleInfo", None)
        if module_info is None:
            continue
        raw_model = str(getattr(module_info, "ModelNumber", "")).strip()
        board_infos = list(getattr(hardware_info, "BoardInfos", ()) or ())
        if not raw_model and board_infos:
            raw_model = str(
                getattr(board_infos[0], "BoardName", "")
            ).strip()
        model = _canonical_qcs_model_number(raw_model)
        if not model:
            model = "Unknown"

        host_controller = int(
            getattr(hardware_info, "HostControllerId", 0)
        )
        chassis = int(getattr(module_info, "Chassis", 0))
        slot = int(getattr(module_info, "Slot", 0))
        if host_controller <= 0 or chassis <= 0:
            raise RuntimeError(
                f"QCS returned an invalid address for {model}: host "
                f"controller {host_controller}, chassis {chassis}"
            )
        if model == QCS_CHASSIS_MODEL and slot <= 0:
            continue
        if slot <= 0:
            raise RuntimeError(
                f"QCS returned an invalid slot {slot} for {model}"
            )

        key = (host_controller, chassis)
        inventory = inventories.setdefault(
            key,
            {
                "chassis_model": QCS_CHASSIS_MODEL,
                "chassis": chassis,
                "host_controller": host_controller,
                "modules": [],
            },
        )
        inventory["modules"].append(
            {
                "slot": slot,
                "model": model,
                "reported_model": raw_model or model,
                "serial_number": str(
                    getattr(module_info, "SerialNumber", "")
                ),
                "firmware_version": str(
                    getattr(module_info, "FirmwareVersion", "")
                ),
                "host_ip_address": str(
                    getattr(hardware_info, "HostIpAddress", "")
                ),
                "boards": [
                    {
                        "name": str(getattr(board, "BoardName", "")),
                        "version": str(
                            getattr(board, "BoardVersion", "")
                        ),
                        "serial_number": str(
                            getattr(board, "SerialNumber", "")
                        ),
                    }
                    for board in board_infos
                ],
            }
        )

    if not inventories:
        raise RuntimeError(
            f"QCS at {address} reported no installed PXI modules"
        )
    result_inventories = []
    for key in sorted(inventories):
        inventory = inventories[key]
        inventory["modules"].sort(
            key=lambda module: (
                int(module["slot"]),
                str(module["model"]),
            )
        )
        result_inventories.append(inventory)
    return {
        "ip_address": address,
        "inventories": result_inventories,
    }


def merge_qcs_discovered_hardware_configuration(
    current_configuration: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    ip_address: str,
) -> tuple[dict, list[dict]]:
    """Merge live topology while retaining only compatible channel mappings."""

    current = normalize_qcs_hardware_configuration(current_configuration)
    modules = [
        {
            "slot": int(module["slot"]),
            "model": _canonical_qcs_model_number(module["model"]),
        }
        for module in inventory.get("modules", ())
    ]
    unsupported = [
        module
        for module in modules
        if module["model"] not in QCS_MODULE_MODELS
    ]
    if unsupported:
        details = ", ".join(
            f"{module['model']} in slot {module['slot']}"
            for module in unsupported
        )
        raise ValueError(
            "the front-panel renderer does not support discovered module(s): "
            f"{details}"
        )

    modules_by_slot = {
        int(module["slot"]): str(module["model"]) for module in modules
    }
    normalized_ip_address = normalize_qcs_server_ip(ip_address)
    same_physical_namespace = (
        str(current.get("ip_address") or "").strip()
        == normalized_ip_address
        and int(current["host_controller"])
        == int(inventory["host_controller"])
        and int(current["chassis"]) == int(inventory["chassis"])
    )
    preserved_mappings = []
    removed_mappings = []
    for mapping in current["channel_mappings"]:
        model = modules_by_slot.get(int(mapping["slot"]))
        spec = None if model is None else QCS_MODULE_MODELS[model]
        role = str(mapping["role"])
        compatible = (
            same_physical_namespace
            and spec is not None
            and spec["instrument"] is not None
            and int(mapping["channel"]) <= int(spec["channels"])
            and spec["instrument"] in QCS_ROLE_INSTRUMENTS[role]
        )
        if compatible:
            preserved_mappings.append(dict(mapping))
        else:
            removed_mappings.append(dict(mapping))

    preserved_addresses = {
        (int(mapping["slot"]), int(mapping["channel"]))
        for mapping in preserved_mappings
    }
    preserved_links = []
    for link in current["downconverter_links"]:
        digitizer_model = modules_by_slot.get(
            int(link["digitizer_slot"])
        )
        downconverter_model = modules_by_slot.get(
            int(link["downconverter_slot"])
        )
        if (
            same_physical_namespace
            and digitizer_model == "M5200A"
            and downconverter_model == "M5201A"
            and (
                int(link["digitizer_slot"]),
                int(link["digitizer_channel"]),
            )
            in preserved_addresses
        ):
            preserved_links.append(dict(link))

    merged = normalize_qcs_hardware_configuration(
        {
            "version": QCS_HARDWARE_CONFIGURATION_VERSION,
            "chassis_model": str(
                inventory.get("chassis_model", QCS_CHASSIS_MODEL)
            ),
            "chassis": inventory["chassis"],
            "host_controller": inventory["host_controller"],
            "ip_address": normalized_ip_address,
            "modules": modules,
            "channel_mappings": preserved_mappings,
            "downconverter_links": preserved_links,
        }
    )
    return merged, removed_mappings


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 1 or result != value:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 0 or result != value:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _virtual_name(value: Any) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError("QCS virtual-channel names must not be empty")
    if _VIRTUAL_NAME_RE.fullmatch(name) is None:
        raise ValueError(
            f"QCS virtual-channel name {name!r} may contain only letters, "
            "numbers, and underscores"
        )
    return name


def normalize_qcs_mapper_sha256(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    digest = str(value).strip().lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(
            "QCS hardware_mapper_sha256 must contain 64 hexadecimal characters"
        )
    return digest


def _module_occupied_slots(module: Mapping[str, Any]) -> tuple[int, ...]:
    slot = int(module["slot"])
    span = int(QCS_MODULE_MODELS[str(module["model"])]["span"])
    return tuple(range(slot, slot + span))


def _normalize_modules(raw_modules: Any) -> list[dict]:
    if raw_modules is None:
        raw_modules = _DEFAULT_MODULES
    if not isinstance(raw_modules, (list, tuple)):
        raise TypeError("QCS hardware modules must be an array")

    result = []
    occupied = {}
    for raw_module in raw_modules:
        if not isinstance(raw_module, Mapping):
            raise TypeError("each QCS hardware module must be an object")
        slot = _positive_integer(raw_module.get("slot"), "QCS module slot")
        model = str(raw_module.get("model", "")).strip()
        if model not in QCS_MODULE_MODELS:
            supported = ", ".join(QCS_MODULE_MODELS)
            raise ValueError(
                f"unsupported QCS module model {model!r}; expected {supported}"
            )
        span = int(QCS_MODULE_MODELS[model]["span"])
        if slot + span - 1 > QCS_CHASSIS_SLOT_COUNT:
            raise ValueError(
                f"{model} in slot {slot} extends past the "
                f"{QCS_CHASSIS_SLOT_COUNT}-slot chassis"
            )
        module = {"slot": slot, "model": model}
        for occupied_slot in _module_occupied_slots(module):
            if occupied_slot in occupied:
                previous = occupied[occupied_slot]
                raise ValueError(
                    f"QCS slot {occupied_slot} is occupied by both "
                    f"{previous} and {model}"
                )
            occupied[occupied_slot] = model
        result.append(module)

    slots = [module["slot"] for module in result]
    if len(set(slots)) != len(slots):
        raise ValueError("QCS module starting slots must be unique")
    return sorted(result, key=lambda module: module["slot"])


def _normalize_channel_mappings(
    raw_mappings: Any,
    modules: Sequence[Mapping[str, Any]],
) -> list[dict]:
    if raw_mappings is None:
        raw_mappings = []
    if not isinstance(raw_mappings, (list, tuple)):
        raise TypeError("QCS channel mappings must be an array")
    modules_by_slot = {int(module["slot"]): module for module in modules}

    result = []
    used_roles = set()
    used_names = set()
    independent_addresses = {}
    for raw_mapping in raw_mappings:
        if not isinstance(raw_mapping, Mapping):
            raise TypeError("each QCS channel mapping must be an object")
        role = str(raw_mapping.get("role", "")).strip().lower()
        if role not in QCS_ROLE_LABELS:
            supported = ", ".join(QCS_ROLE_LABELS)
            raise ValueError(
                f"unsupported QCS channel role {role!r}; expected {supported}"
            )
        logical_index = _nonnegative_integer(
            raw_mapping.get("logical_index", 0),
            f"QCS {role} logical index",
        )
        if role == "acquisition" and logical_index != 0:
            raise ValueError("QCS acquisition logical index must be 0")
        role_key = (role, logical_index)
        if role_key in used_roles:
            raise ValueError(
                f"QCS {role} logical index {logical_index} is mapped more than once"
            )
        used_roles.add(role_key)

        virtual_name = _virtual_name(raw_mapping.get("virtual_name", ""))
        if virtual_name in used_names:
            raise ValueError(
                f"QCS virtual-channel name {virtual_name!r} is used more than once"
            )
        used_names.add(virtual_name)
        label = _nonnegative_integer(
            raw_mapping.get("label", 0),
            f"QCS virtual-channel label for {virtual_name!r}",
        )
        absolute_phase = raw_mapping.get(
            "absolute_phase",
            role != "dc",
        )
        if not isinstance(absolute_phase, bool):
            raise TypeError(
                f"QCS absolute_phase for {virtual_name!r} must be boolean"
            )

        slot = _positive_integer(raw_mapping.get("slot"), "QCS mapping slot")
        channel = _positive_integer(
            raw_mapping.get("channel"),
            "QCS physical channel",
        )
        module = modules_by_slot.get(slot)
        if module is None:
            raise ValueError(
                f"QCS mapping {virtual_name!r} targets slot {slot}, but no "
                "module starts in that slot"
            )
        model = str(module["model"])
        module_spec = QCS_MODULE_MODELS[model]
        instrument = module_spec["instrument"]
        if instrument is None:
            raise ValueError(
                f"QCS mapping {virtual_name!r} cannot target {model}"
            )
        if instrument not in QCS_ROLE_INSTRUMENTS[role]:
            expected = " or ".join(sorted(QCS_ROLE_INSTRUMENTS[role]))
            raise ValueError(
                f"QCS {role} mapping {virtual_name!r} requires {expected}, "
                f"not {instrument}"
            )
        channel_count = int(module_spec["channels"])
        if channel > channel_count:
            raise ValueError(
                f"{model} exposes channels 1-{channel_count}; "
                f"{virtual_name!r} requests channel {channel}"
            )
        raw_lo_frequency_hz = raw_mapping.get("lo_frequency_hz")
        if raw_lo_frequency_hz in (None, ""):
            lo_frequency_hz = None
        else:
            lo_frequency_hz = float(raw_lo_frequency_hz)
            if (
                not isfinite(lo_frequency_hz)
                or lo_frequency_hz < 0.0
                or lo_frequency_hz > 18.0e9
            ):
                raise ValueError(
                    f"QCS LO frequency for {virtual_name!r} must be in "
                    "[0, 18] GHz"
                )
        if instrument != "M5300AWG" and lo_frequency_hz is not None:
            raise ValueError(
                f"{instrument} mapping {virtual_name!r} does not expose an "
                "editable LO frequency"
            )

        address_key = (slot, channel)
        previous = independent_addresses.get(address_key)
        if previous is not None:
            raise ValueError(
                f"QCS virtual channels {previous['virtual_name']!r} and "
                f"{virtual_name!r} cannot share slot {slot} channel "
                f"{channel} in this builder"
            )
        independent_addresses[address_key] = {
            "role": role,
            "virtual_name": virtual_name,
        }
        result.append(
            {
                "role": role,
                "logical_index": logical_index,
                "virtual_name": virtual_name,
                "label": label,
                "absolute_phase": absolute_phase,
                "lo_frequency_hz": lo_frequency_hz,
                "slot": slot,
                "channel": channel,
            }
        )

    result.sort(
        key=lambda mapping: (
            _ROLE_SORT_ORDER[mapping["role"]],
            mapping["logical_index"],
        )
    )
    return result


def _normalize_downconverter_links(
    raw_links: Any,
    modules: Sequence[Mapping[str, Any]],
    mappings: Sequence[Mapping[str, Any]],
) -> list[dict]:
    """Validate editable M5200-to-M5201 channel-pair relationships."""

    if raw_links is None:
        raw_links = []
    if not isinstance(raw_links, (list, tuple)):
        raise TypeError("QCS downconverter links must be an array")

    modules_by_slot = {int(module["slot"]): module for module in modules}
    mapped_addresses = {
        (int(mapping["slot"]), int(mapping["channel"]))
        for mapping in mappings
    }
    used_digitizers = set()
    used_downconverters = set()
    shared_lo_by_slot: dict[int, Optional[float]] = {}
    result = []
    for raw_link in raw_links:
        if not isinstance(raw_link, Mapping):
            raise TypeError("each QCS downconverter link must be an object")
        digitizer_slot = _positive_integer(
            raw_link.get("digitizer_slot"),
            "QCS M5200 digitizer slot",
        )
        digitizer_channel = _positive_integer(
            raw_link.get("digitizer_channel"),
            "QCS M5200 digitizer channel",
        )
        downconverter_slot = _positive_integer(
            raw_link.get("downconverter_slot"),
            "QCS M5201 downconverter slot",
        )
        downconverter_channel = _positive_integer(
            raw_link.get("downconverter_channel"),
            "QCS M5201 downconverter channel",
        )

        digitizer_module = modules_by_slot.get(digitizer_slot)
        digitizer_model = (
            None
            if digitizer_module is None
            else str(digitizer_module["model"])
        )
        if digitizer_model != "M5200A":
            raise ValueError(
                f"QCS downconverter link requires an M5200A in slot "
                f"{digitizer_slot}, not {digitizer_model or 'an empty slot'}"
            )
        downconverter_module = modules_by_slot.get(downconverter_slot)
        downconverter_model = (
            None
            if downconverter_module is None
            else str(downconverter_module["model"])
        )
        if downconverter_model != "M5201A":
            raise ValueError(
                f"QCS downconverter link requires an M5201A in slot "
                f"{downconverter_slot}, not "
                f"{downconverter_model or 'an empty slot'}"
            )
        if digitizer_channel > int(
            QCS_MODULE_MODELS["M5200A"]["channels"]
        ):
            raise ValueError("M5200A exposes digitizer channels 1-4")
        if downconverter_channel > int(
            QCS_MODULE_MODELS["M5201A"]["channels"]
        ):
            raise ValueError("M5201A exposes downconverter channels 1-4")

        digitizer_address = (digitizer_slot, digitizer_channel)
        if digitizer_address not in mapped_addresses:
            raise ValueError(
                f"QCS M5200A slot {digitizer_slot} channel "
                f"{digitizer_channel} must have a virtual-channel mapping "
                "before it can use an M5201A downconverter"
            )
        if digitizer_address in used_digitizers:
            raise ValueError(
                f"QCS M5200A slot {digitizer_slot} channel "
                f"{digitizer_channel} has more than one downconverter"
            )
        used_digitizers.add(digitizer_address)

        downconverter_address = (
            downconverter_slot,
            downconverter_channel,
        )
        if downconverter_address in used_downconverters:
            raise ValueError(
                f"QCS M5201A slot {downconverter_slot} channel "
                f"{downconverter_channel} is linked more than once"
            )
        used_downconverters.add(downconverter_address)

        raw_lo_frequency_hz = raw_link.get("lo_frequency_hz")
        if raw_lo_frequency_hz in (None, ""):
            lo_frequency_hz = None
        else:
            lo_frequency_hz = float(raw_lo_frequency_hz)
            if (
                not isfinite(lo_frequency_hz)
                or lo_frequency_hz < 1.0e9
                or lo_frequency_hz > 18.0e9
            ):
                raise ValueError(
                    "QCS M5201 LO frequency must be in [1, 18] GHz"
                )
        previous_lo = shared_lo_by_slot.get(downconverter_slot, ...)
        if previous_lo is ...:
            shared_lo_by_slot[downconverter_slot] = lo_frequency_hz
        elif previous_lo != lo_frequency_hz:
            raise ValueError(
                f"all channels on M5201A slot {downconverter_slot} must "
                "use the same shared LO frequency"
            )

        result.append(
            {
                "digitizer_slot": digitizer_slot,
                "digitizer_channel": digitizer_channel,
                "downconverter_slot": downconverter_slot,
                "downconverter_channel": downconverter_channel,
                "lo_frequency_hz": lo_frequency_hz,
            }
        )

    result.sort(
        key=lambda link: (
            link["digitizer_slot"],
            link["digitizer_channel"],
            link["downconverter_slot"],
            link["downconverter_channel"],
        )
    )
    return result


def normalize_qcs_hardware_configuration(
    configuration: Optional[Mapping[str, Any]],
    *,
    required_dc_count: Optional[int] = None,
) -> dict:
    """Return a validated, JSON-compatible QCS mapper-builder recipe."""
    if configuration is None:
        configuration = {}
    if not isinstance(configuration, Mapping):
        raise TypeError("QCS hardware_configuration must be an object")

    version = _positive_integer(
        configuration.get("version", QCS_HARDWARE_CONFIGURATION_VERSION),
        "QCS hardware configuration version",
    )
    if version != QCS_HARDWARE_CONFIGURATION_VERSION:
        raise ValueError(
            f"unsupported QCS hardware configuration version {version}"
        )
    chassis_model = str(
        configuration.get("chassis_model", QCS_CHASSIS_MODEL)
    ).strip()
    if chassis_model != QCS_CHASSIS_MODEL:
        raise ValueError(
            f"unsupported QCS chassis {chassis_model!r}; expected "
            f"{QCS_CHASSIS_MODEL}"
        )
    chassis = _positive_integer(
        configuration.get("chassis", 1),
        "QCS chassis number",
    )
    host_controller = _positive_integer(
        configuration.get("host_controller", 1),
        "QCS host-controller number",
    )
    raw_ip_address = configuration.get("ip_address")
    ip_address = (
        None
        if raw_ip_address is None
        else str(raw_ip_address).strip() or None
    )
    modules = _normalize_modules(configuration.get("modules"))
    mappings = _normalize_channel_mappings(
        configuration.get("channel_mappings"),
        modules,
    )
    downconverter_links = _normalize_downconverter_links(
        configuration.get("downconverter_links"),
        modules,
        mappings,
    )

    if required_dc_count is not None:
        required_dc_count = _positive_integer(
            required_dc_count,
            "QCS required DC output count",
        )
        dc_indices = [
            mapping["logical_index"]
            for mapping in mappings
            if mapping["role"] == "dc"
        ]
        expected = list(range(required_dc_count))
        if dc_indices != expected:
            raise ValueError(
                "QCS front panel requires one DC mapping for every waveform "
                f"output; expected GUI indices {expected}, got {dc_indices}"
            )
    acquisition_count = sum(
        mapping["role"] == "acquisition" for mapping in mappings
    )
    if acquisition_count > 1:
        raise ValueError(
            "PulseGenerator supports one QCS acquisition virtual channel"
        )

    return {
        "version": version,
        "chassis_model": chassis_model,
        "chassis": chassis,
        "host_controller": host_controller,
        "ip_address": ip_address,
        "modules": modules,
        "channel_mappings": mappings,
        "downconverter_links": downconverter_links,
    }


def qcs_role_bindings(
    configuration: Mapping[str, Any],
    *,
    required_dc_count: int,
) -> tuple[list[str], dict[str, str], Optional[str]]:
    """Extract PulseGenerator role bindings from a validated builder recipe."""
    normalized = normalize_qcs_hardware_configuration(
        configuration,
        required_dc_count=required_dc_count,
    )
    dc_names = [
        mapping["virtual_name"]
        for mapping in normalized["channel_mappings"]
        if mapping["role"] == "dc"
    ]
    rf_names = {
        str(mapping["logical_index"]): mapping["virtual_name"]
        for mapping in normalized["channel_mappings"]
        if mapping["role"] == "rf"
    }
    acquisition_names = [
        mapping["virtual_name"]
        for mapping in normalized["channel_mappings"]
        if mapping["role"] == "acquisition"
    ]
    return dc_names, rf_names, (
        acquisition_names[0] if acquisition_names else None
    )


def qcs_hardware_mapper_fingerprint(
    configuration: Mapping[str, Any],
) -> str:
    """Hash only the recipe fields that affect the serialized mapper."""
    normalized = normalize_qcs_hardware_configuration(configuration)
    modules_by_slot = {
        module["slot"]: module for module in normalized["modules"]
    }
    projection = {
        "chassis": normalized["chassis"],
        "host_controller": normalized["host_controller"],
        "ip_address": normalized["ip_address"],
        "modules": normalized["modules"],
        "channel_mappings": sorted(
            (
                {
                    "virtual_name": mapping["virtual_name"],
                    "label": mapping["label"],
                    "absolute_phase": mapping["absolute_phase"],
                    "lo_frequency_hz": mapping["lo_frequency_hz"],
                    "slot": mapping["slot"],
                    "channel": mapping["channel"],
                    "instrument": QCS_MODULE_MODELS[
                        modules_by_slot[mapping["slot"]]["model"]
                    ]["instrument"],
                }
                for mapping in normalized["channel_mappings"]
            ),
            key=lambda mapping: (
                mapping["virtual_name"],
                mapping["label"],
            ),
        ),
        "downconverter_links": normalized["downconverter_links"],
    }
    payload = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def synchronize_qcs_hardware_role_names(
    configuration: Mapping[str, Any],
    dc_channel_names: Sequence[str],
    rf_channel_names: Mapping[int, str],
    acquisition_channel_name: Optional[str],
) -> dict:
    """Update names only when the builder already has the same role shape."""
    normalized = normalize_qcs_hardware_configuration(configuration)
    target_dc = [_virtual_name(name) for name in dc_channel_names]
    target_rf = {
        int(index): _virtual_name(name)
        for index, name in dict(rf_channel_names).items()
    }
    target_acquisition = (
        None
        if not acquisition_channel_name
        else _virtual_name(acquisition_channel_name)
    )
    current_dc, current_rf, current_acquisition = qcs_role_bindings(
        normalized,
        required_dc_count=len(target_dc),
    )
    if len(set(target_dc)) != len(target_dc):
        raise ValueError("QCS DC virtual-channel names must be unique")
    if len(set(target_rf.values())) != len(target_rf):
        raise ValueError("QCS RF virtual-channel names must be unique")
    if len(current_rf) != len(target_rf):
        raise ValueError(
            "QCS RF role bindings differ from the hardware configuration"
        )
    if (current_acquisition is None) != (target_acquisition is None):
        raise ValueError(
            "QCS acquisition role differs from the hardware configuration"
        )

    dc_index_by_name = {
        name: index for index, name in enumerate(target_dc)
    }
    rf_index_by_name = {
        name: index for index, name in target_rf.items()
    }
    replacement_dc_items = iter(
        (index, name)
        for index, name in enumerate(target_dc)
        if name not in set(current_dc)
    )
    replacement_rf_items = iter(
        (index, name)
        for index, name in sorted(target_rf.items())
        if name not in set(current_rf.values())
    )

    updated = dict(normalized)
    updated_mappings = []
    for mapping in normalized["channel_mappings"]:
        mapping = dict(mapping)
        if mapping["role"] == "dc":
            if mapping["virtual_name"] in dc_index_by_name:
                mapping["logical_index"] = dc_index_by_name[
                    mapping["virtual_name"]
                ]
            else:
                logical_index, virtual_name = next(replacement_dc_items)
                mapping["logical_index"] = logical_index
                mapping["virtual_name"] = virtual_name
        elif mapping["role"] == "rf":
            if mapping["virtual_name"] in rf_index_by_name:
                mapping["logical_index"] = rf_index_by_name[
                    mapping["virtual_name"]
                ]
            else:
                logical_index, virtual_name = next(replacement_rf_items)
                mapping["logical_index"] = logical_index
                mapping["virtual_name"] = virtual_name
        elif mapping["role"] == "acquisition":
            mapping["virtual_name"] = target_acquisition
        updated_mappings.append(mapping)
    updated["channel_mappings"] = updated_mappings
    return normalize_qcs_hardware_configuration(
        updated,
        required_dc_count=len(target_dc),
    )


def _next_free_slot(
    modules: Sequence[Mapping[str, Any]],
    *,
    span: int,
) -> int:
    occupied = {
        slot
        for module in modules
        for slot in _module_occupied_slots(module)
    }
    for slot in range(1, QCS_CHASSIS_SLOT_COUNT - span + 2):
        if all(candidate not in occupied for candidate in range(slot, slot + span)):
            return slot
    raise ValueError("the QCS chassis has no free slots for another module")


def default_qcs_hardware_configuration(
    dc_channel_names: Sequence[str],
    rf_channel_names: Mapping[int, str],
    acquisition_channel_name: Optional[str],
) -> dict:
    """Build the diagram's M9046A/M5000 layout for current GUI bindings."""
    modules = [dict(module) for module in _DEFAULT_MODULES]
    mappings = []

    dc_names = list(dc_channel_names)
    dc_modules = [2]
    while len(dc_modules) * 4 < len(dc_names):
        slot = _next_free_slot(modules, span=1)
        modules.append({"slot": slot, "model": "M5301A"})
        dc_modules.append(slot)
    for logical_index, name in enumerate(dc_names):
        mappings.append(
            {
                "role": "dc",
                "logical_index": logical_index,
                "virtual_name": str(name),
                "label": 0,
                "absolute_phase": False,
                "lo_frequency_hz": None,
                "slot": dc_modules[logical_index // 4],
                "channel": logical_index % 4 + 1,
            }
        )

    rf_items = sorted(
        (int(index), str(name))
        for index, name in dict(rf_channel_names).items()
    )
    rf_modules = [3]
    while len(rf_modules) * 4 < len(rf_items):
        slot = _next_free_slot(modules, span=2)
        modules.append({"slot": slot, "model": "M5300A"})
        rf_modules.append(slot)
    for mapping_index, (logical_index, name) in enumerate(rf_items):
        mappings.append(
            {
                "role": "rf",
                "logical_index": logical_index,
                "virtual_name": name,
                "label": 0,
                "absolute_phase": True,
                "lo_frequency_hz": None,
                "slot": rf_modules[mapping_index // 4],
                "channel": mapping_index % 4 + 1,
            }
        )

    if acquisition_channel_name:
        mappings.append(
            {
                "role": "acquisition",
                "logical_index": 0,
                "virtual_name": str(acquisition_channel_name),
                "label": 0,
                "absolute_phase": True,
                "lo_frequency_hz": None,
                "slot": 5,
                "channel": 1,
            }
        )
    return normalize_qcs_hardware_configuration(
        {
            "version": QCS_HARDWARE_CONFIGURATION_VERSION,
            "chassis_model": QCS_CHASSIS_MODEL,
            "chassis": 1,
            "host_controller": 1,
            "ip_address": DEFAULT_QCS_IP_ADDRESS,
            "modules": modules,
            "channel_mappings": mappings,
        }
    )


def resize_qcs_dc_mappings(
    configuration: Mapping[str, Any],
    dc_channel_names: Sequence[str],
    *,
    removed_index: Optional[int] = None,
) -> dict:
    """Keep physical DC bindings aligned when GUI waveform outputs change."""
    normalized = normalize_qcs_hardware_configuration(configuration)
    modules = [dict(module) for module in normalized["modules"]]
    modules_by_slot = {
        module["slot"]: module for module in modules
    }
    target_names = [_virtual_name(name) for name in dc_channel_names]
    target_name_set = set(target_names)
    non_dc_mappings = [
        dict(mapping)
        for mapping in normalized["channel_mappings"]
        if mapping["role"] != "dc"
    ]
    old_dc_mappings = [
        dict(mapping)
        for mapping in normalized["channel_mappings"]
        if mapping["role"] == "dc"
    ]
    promoted_addresses = set()
    for mapping in non_dc_mappings:
        module = modules_by_slot[mapping["slot"]]
        if (
            mapping["role"] == "unassigned"
            and mapping["virtual_name"] in target_name_set
            and QCS_MODULE_MODELS[module["model"]]["instrument"]
            == "M5301AWG"
        ):
            promoted = dict(mapping)
            promoted["role"] = "dc"
            old_dc_mappings.append(promoted)
            promoted_addresses.add((mapping["slot"], mapping["channel"]))
    non_dc_mappings = [
        mapping
        for mapping in non_dc_mappings
        if (mapping["slot"], mapping["channel"]) not in promoted_addresses
    ]
    if removed_index is not None:
        removed_index = _nonnegative_integer(
            removed_index,
            "removed QCS DC output index",
        )
        used_unassigned_indices = {
            mapping["logical_index"]
            for mapping in non_dc_mappings
            if mapping["role"] == "unassigned"
        }
        next_unassigned_index = 0
        while next_unassigned_index in used_unassigned_indices:
            next_unassigned_index += 1
        adjusted = []
        for mapping in old_dc_mappings:
            logical_index = mapping["logical_index"]
            if logical_index == removed_index:
                demoted = dict(mapping)
                demoted["role"] = "unassigned"
                demoted["logical_index"] = next_unassigned_index
                non_dc_mappings.append(demoted)
                used_unassigned_indices.add(next_unassigned_index)
                while next_unassigned_index in used_unassigned_indices:
                    next_unassigned_index += 1
                continue
            if logical_index > removed_index:
                mapping["logical_index"] = logical_index - 1
            adjusted.append(mapping)
        old_dc_mappings = adjusted

    by_name = {
        mapping["virtual_name"]: mapping for mapping in old_dc_mappings
    }
    unmatched_old_mappings = iter(
        mapping
        for mapping in sorted(
            old_dc_mappings,
            key=lambda item: item["logical_index"],
        )
        if mapping["virtual_name"] not in target_name_set
    )
    used_addresses = {
        (mapping["slot"], mapping["channel"])
        for mapping in non_dc_mappings + old_dc_mappings
    }
    dc_modules = [
        module
        for module in modules
        if QCS_MODULE_MODELS[module["model"]]["instrument"] == "M5301AWG"
    ]
    new_dc_mappings = []
    for logical_index, name in enumerate(target_names):
        mapping = by_name.get(name)
        if mapping is None:
            mapping = next(unmatched_old_mappings, None)
        if mapping is None:
            address = next(
                (
                    (module["slot"], channel)
                    for module in dc_modules
                    for channel in range(
                        1,
                        int(QCS_MODULE_MODELS[module["model"]]["channels"]) + 1,
                    )
                    if (module["slot"], channel) not in used_addresses
                ),
                None,
            )
            if address is None:
                slot = _next_free_slot(modules, span=1)
                module = {"slot": slot, "model": "M5301A"}
                modules.append(module)
                dc_modules.append(module)
                address = (slot, 1)
            mapping = {
                "role": "dc",
                "logical_index": logical_index,
                "virtual_name": name,
                "label": 0,
                "absolute_phase": False,
                "lo_frequency_hz": None,
                "slot": address[0],
                "channel": address[1],
            }
        else:
            mapping = dict(mapping)
            mapping["logical_index"] = logical_index
            mapping["virtual_name"] = name
        used_addresses.add((mapping["slot"], mapping["channel"]))
        new_dc_mappings.append(mapping)

    updated = dict(normalized)
    updated["modules"] = modules
    updated["channel_mappings"] = non_dc_mappings + new_dc_mappings
    return normalize_qcs_hardware_configuration(updated)


def resize_incomplete_qcs_dc_mappings(
    configuration: Mapping[str, Any],
    dc_channel_names: Sequence[str],
    *,
    removed_index: Optional[int] = None,
) -> dict:
    """Resize GUI DC roles without inventing physical SMA assignments.

    Hardware identification can intentionally leave a topology incomplete
    after obsolete mappings are removed.  Changing the number of waveform
    outputs must preserve that discovered chassis: a newly added output stays
    unmapped until the user selects an SMA, while a deleted output is demoted
    to an unassigned virtual channel instead of disappearing from the mapper.
    """

    normalized = normalize_qcs_hardware_configuration(configuration)
    target_names = [_virtual_name(name) for name in dc_channel_names]
    if len(set(target_names)) != len(target_names):
        raise ValueError("QCS DC virtual-channel names must be unique")

    if removed_index is not None:
        removed_index = _nonnegative_integer(
            removed_index,
            "removed QCS DC output index",
        )

    used_unassigned_indices = {
        int(mapping["logical_index"])
        for mapping in normalized["channel_mappings"]
        if mapping["role"] == "unassigned"
    }
    next_unassigned_index = 0

    def allocate_unassigned_index() -> int:
        nonlocal next_unassigned_index
        while next_unassigned_index in used_unassigned_indices:
            next_unassigned_index += 1
        result = next_unassigned_index
        used_unassigned_indices.add(result)
        next_unassigned_index += 1
        return result

    mappings = []
    for original in normalized["channel_mappings"]:
        mapping = dict(original)
        if mapping["role"] != "dc":
            mappings.append(mapping)
            continue

        logical_index = int(mapping["logical_index"])
        if removed_index is not None and logical_index == removed_index:
            mapping["role"] = "unassigned"
            mapping["logical_index"] = allocate_unassigned_index()
            mappings.append(mapping)
            continue
        if removed_index is not None and logical_index > removed_index:
            logical_index -= 1
            mapping["logical_index"] = logical_index
        if logical_index >= len(target_names):
            mapping["role"] = "unassigned"
            mapping["logical_index"] = allocate_unassigned_index()
        else:
            mapping["virtual_name"] = target_names[logical_index]
        mappings.append(mapping)

    updated = dict(normalized)
    updated["channel_mappings"] = mappings
    return normalize_qcs_hardware_configuration(updated)


def _import_qcs():
    try:
        import keysight.qcs as qcs  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "Keysight QCS is required to load or save ChannelMapper files. "
            "Run this GUI from the 'qcs' Conda environment."
        ) from exc
    return qcs


def build_qcs_channel_mapper(
    configuration: Mapping[str, Any],
    *,
    qcs_module=None,
):
    """Build a fresh native QCS ChannelMapper from a validated GUI recipe."""
    qcs = qcs_module or _import_qcs()
    normalized = normalize_qcs_hardware_configuration(configuration)
    modules_by_slot = {
        module["slot"]: module for module in normalized["modules"]
    }
    mapper = qcs.ChannelMapper(ip_address=normalized["ip_address"])
    for mapping in normalized["channel_mappings"]:
        module = modules_by_slot[mapping["slot"]]
        instrument_name = QCS_MODULE_MODELS[module["model"]]["instrument"]
        instrument = getattr(qcs.InstrumentEnum, str(instrument_name))
        channels = qcs.Channels(
            mapping["label"],
            mapping["virtual_name"],
            absolute_phase=mapping["absolute_phase"],
        )
        address = qcs.Address(
            normalized["chassis"],
            mapping["slot"],
            mapping["channel"],
            host_controller=normalized["host_controller"],
        )
        mapper.add_channel_mapping(channels, address, instrument)
        if instrument_name == "M5300AWG":
            lo_frequency_hz = mapping["lo_frequency_hz"]
            if lo_frequency_hz is None:
                raise ValueError(
                    f"QCS M5300 mapping {mapping['virtual_name']!r} requires "
                    "an M5300 LO frequency before the mapper can be saved"
                )
            mapper.set_lo_frequencies(address, lo_frequency_hz)

    digitizer_addresses = []
    downconverter_addresses = []
    downconverter_groups: dict[int, dict[str, Any]] = {}
    for link in normalized["downconverter_links"]:
        lo_frequency_hz = link["lo_frequency_hz"]
        if lo_frequency_hz is None:
            raise ValueError(
                f"QCS M5201A slot {link['downconverter_slot']} requires "
                "an LO frequency before the mapper can be saved"
            )
        digitizer_address = qcs.Address(
            normalized["chassis"],
            link["digitizer_slot"],
            link["digitizer_channel"],
            host_controller=normalized["host_controller"],
        )
        downconverter_address = qcs.Address(
            normalized["chassis"],
            link["downconverter_slot"],
            link["downconverter_channel"],
            host_controller=normalized["host_controller"],
        )
        digitizer_addresses.append(digitizer_address)
        downconverter_addresses.append(downconverter_address)
        group = downconverter_groups.setdefault(
            int(link["downconverter_slot"]),
            {
                "lo_frequency_hz": float(lo_frequency_hz),
                "addresses": [],
            },
        )
        group["addresses"].append(downconverter_address)
    if digitizer_addresses:
        mapper.add_downconverters(
            digitizer_addresses,
            downconverter_addresses,
        )
        for group in downconverter_groups.values():
            mapper.set_lo_frequencies(
                group["addresses"],
                group["lo_frequency_hz"],
            )
    return mapper


def save_qcs_channel_mapper(
    configuration: Mapping[str, Any],
    path,
    *,
    qcs_module=None,
) -> Path:
    """Save and reload a native mapper so the selected file is run-ready."""
    qcs = qcs_module or _import_qcs()
    output_path = Path(path).expanduser()
    if output_path.suffix.lower() != ".qcs":
        output_path = output_path.with_suffix(".qcs")
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"QCS mapper directory does not exist: {output_path.parent}"
        )
    mapper = build_qcs_channel_mapper(configuration, qcs_module=qcs)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid4().hex}.tmp.qcs"
    )
    try:
        qcs.save(mapper, temporary_path)
        loaded = qcs.load(temporary_path)
        mapper_type = getattr(qcs, "ChannelMapper", None)
        if mapper_type is not None and not isinstance(loaded, mapper_type):
            raise TypeError(
                f"{temporary_path} reloaded as {type(loaded).__name__}, "
                "not ChannelMapper"
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path.resolve()


def qcs_mapper_file_sha256(path) -> str:
    """Return the native mapper's content digest for run-time identity checks."""
    mapper_path = Path(path).expanduser()
    if not mapper_path.is_file():
        raise FileNotFoundError(f"QCS ChannelMapper file not found: {mapper_path}")
    digest = hashlib.sha256()
    with mapper_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_imported_qcs_role_configuration(
    configuration: Mapping[str, Any],
    path,
    *,
    qcs_module=None,
) -> dict:
    """Validate active GUI roles and required LOs against a native mapper."""
    qcs = qcs_module or _import_qcs()
    normalized = normalize_qcs_hardware_configuration(configuration)
    mapper_path = Path(path).expanduser()
    if not mapper_path.is_file():
        raise FileNotFoundError(f"QCS ChannelMapper file not found: {mapper_path}")
    mapper = qcs.load(mapper_path)
    mapper_type = getattr(qcs, "ChannelMapper", None)
    if mapper_type is not None and not isinstance(mapper, mapper_type):
        raise TypeError(
            f"{mapper_path} contains {type(mapper).__name__}, not ChannelMapper"
        )

    modules_by_slot = {
        module["slot"]: module for module in normalized["modules"]
    }

    def address_key(address) -> tuple[int, int, int, int]:
        return (
            int(address.host_controller),
            int(address.chassis),
            int(address.slot),
            int(address.channel),
        )

    def require_lo(
        physical,
        label: str,
        *,
        minimum_hz: float = 0.0,
    ) -> float:
        settings = getattr(physical, "settings", None)
        if "lo_frequency" not in tuple(
            getattr(settings, "setting_names", ())
        ):
            raise ValueError(f"{label} does not expose an LO frequency setting")
        raw_value = getattr(
            getattr(settings, "lo_frequency", None),
            "value",
            None,
        )
        if raw_value is None:
            raise ValueError(f"{label} LO frequency is unset")
        value = float(raw_value)
        if (
            not isfinite(value)
            or not float(minimum_hz) <= value <= 18.0e9
        ):
            minimum_ghz = float(minimum_hz) / 1.0e9
            raise ValueError(
                f"{label} LO frequency must be in "
                f"[{minimum_ghz:g}, 18] GHz"
            )
        return value

    for mapping in normalized["channel_mappings"]:
        role = mapping["role"]
        if role == "unassigned":
            continue
        module = modules_by_slot[mapping["slot"]]
        expected_instrument = str(
            QCS_MODULE_MODELS[module["model"]]["instrument"]
        )
        address = qcs.Address(
            normalized["chassis"],
            mapping["slot"],
            mapping["channel"],
            host_controller=normalized["host_controller"],
        )
        try:
            physical = mapper.get_physical_channel(address)
        except ValueError as exc:
            raise ValueError(
                f"QCS {role} channel {mapping['virtual_name']!r} is not "
                f"assigned at chassis {normalized['chassis']}, slot "
                f"{mapping['slot']}, connector {mapping['channel']} in the "
                "imported mapper"
            ) from exc
        if str(physical.instrument) != expected_instrument:
            raise ValueError(
                f"QCS {role} channel {mapping['virtual_name']!r} expects "
                f"{expected_instrument}, but the imported mapper has "
                f"{physical.instrument}"
            )

        matching_channels = [
            channels
            for channels in tuple(getattr(mapper, "channels", ()))
            if str(getattr(channels, "name", ""))
            == mapping["virtual_name"]
            and tuple(getattr(channels, "labels", ()))
            == (mapping["label"],)
        ]
        if len(matching_channels) != 1:
            raise ValueError(
                f"QCS {role} channel {mapping['virtual_name']!r} with label "
                f"{mapping['label']} is not uniquely present in the imported "
                "mapper"
            )
        channels = matching_channels[0]
        if bool(channels.absolute_phase) != mapping["absolute_phase"]:
            raise ValueError(
                f"QCS channel {mapping['virtual_name']!r} absolute-phase "
                "setting differs from the imported mapper"
            )
        physicals = tuple(mapper.get_physical_channels(channels))
        if (
            len(physicals) != 1
            or address_key(physicals[0].address) != address_key(address)
        ):
            raise ValueError(
                f"QCS channel {mapping['virtual_name']!r} is bound to a "
                "different physical connector in the imported mapper"
            )

        if role == "rf" and expected_instrument == "M5300AWG":
            require_lo(
                physical,
                f"QCS RF channel {mapping['virtual_name']!r} M5300",
            )
        elif role == "acquisition":
            downconverter = mapper.get_downconverter(physical)
            if downconverter is not None:
                if str(downconverter.instrument) != "M5201Downconverter":
                    raise ValueError(
                        f"QCS acquisition channel "
                        f"{mapping['virtual_name']!r} uses unsupported "
                        f"downconverter {downconverter.instrument}"
                    )
                require_lo(
                    downconverter,
                    f"QCS acquisition channel "
                    f"{mapping['virtual_name']!r} M5201",
                    minimum_hz=1.0e9,
                )

    active_acquisition_addresses = {
        (int(mapping["slot"]), int(mapping["channel"]))
        for mapping in normalized["channel_mappings"]
        if mapping["role"] == "acquisition"
    }
    for link in normalized["downconverter_links"]:
        digitizer_address = qcs.Address(
            normalized["chassis"],
            link["digitizer_slot"],
            link["digitizer_channel"],
            host_controller=normalized["host_controller"],
        )
        expected_downconverter_address = qcs.Address(
            normalized["chassis"],
            link["downconverter_slot"],
            link["downconverter_channel"],
            host_controller=normalized["host_controller"],
        )
        try:
            digitizer = mapper.get_physical_channel(digitizer_address)
        except ValueError as exc:
            raise ValueError(
                f"QCS M5200A slot {link['digitizer_slot']} channel "
                f"{link['digitizer_channel']} is missing from the imported "
                "mapper"
            ) from exc
        downconverter = mapper.get_downconverter(digitizer)
        if downconverter is None:
            raise ValueError(
                f"QCS M5200A slot {link['digitizer_slot']} channel "
                f"{link['digitizer_channel']} has no M5201A downconverter "
                "in the imported mapper"
            )
        if (
            str(downconverter.instrument) != "M5201Downconverter"
            or address_key(downconverter.address)
            != address_key(expected_downconverter_address)
        ):
            raise ValueError(
                f"QCS M5200A slot {link['digitizer_slot']} channel "
                f"{link['digitizer_channel']} is linked to a different "
                "downconverter in the imported mapper"
            )
        configured_lo = link["lo_frequency_hz"]
        if (
            configured_lo is not None
            or (
                int(link["digitizer_slot"]),
                int(link["digitizer_channel"]),
            )
            in active_acquisition_addresses
        ):
            actual_lo = require_lo(
                downconverter,
                f"QCS M5201A slot {link['downconverter_slot']} channel "
                f"{link['downconverter_channel']}",
                minimum_hz=1.0e9,
            )
            if (
                configured_lo is not None
                and actual_lo != float(configured_lo)
            ):
                raise ValueError(
                    f"QCS M5201A slot {link['downconverter_slot']} LO "
                    "frequency differs from the imported mapper"
                )
    return normalized


def configuration_from_qcs_mapper(
    path,
    *,
    dc_channel_names: Sequence[str] = (),
    rf_channel_names: Optional[Mapping[int, str]] = None,
    acquisition_channel_name: Optional[str] = None,
    qcs_module=None,
) -> dict:
    """Resolve an existing mapper while preserving explicit GUI role bindings."""
    qcs = qcs_module or _import_qcs()
    mapper_path = Path(path).expanduser()
    if not mapper_path.is_file():
        raise FileNotFoundError(f"QCS ChannelMapper file not found: {mapper_path}")
    mapper = qcs.load(mapper_path)
    mapper_type = getattr(qcs, "ChannelMapper", None)
    if mapper_type is not None and not isinstance(mapper, mapper_type):
        raise TypeError(
            f"{mapper_path} contains {type(mapper).__name__}, not ChannelMapper"
        )

    physical_channels = tuple(getattr(mapper, "physical_channels", ()))
    modules_by_slot = {}
    chassis_values = set()
    host_values = set()
    for physical in physical_channels:
        address = physical.address
        chassis_values.add(int(address.chassis))
        host_values.add(int(address.host_controller))
        instrument_name = str(physical.instrument)
        model = _INSTRUMENT_TO_MODEL.get(instrument_name)
        if model is None:
            raise ValueError(
                f"QCS mapper uses unsupported instrument {instrument_name!r}"
            )
        slot = int(address.slot)
        previous = modules_by_slot.get(slot)
        if previous is not None and previous != model:
            raise ValueError(
                f"QCS slot {slot} contains both {previous} and {model}"
            )
        modules_by_slot[slot] = model
    if len(chassis_values) > 1 or len(host_values) > 1:
        raise ValueError(
            "the QCS front-panel editor supports one chassis and one host "
            "controller per mapper"
        )

    modules = [
        {"slot": slot, "model": model}
        for slot, model in sorted(modules_by_slot.items())
    ]
    occupied = {
        occupied_slot
        for module in modules
        for occupied_slot in _module_occupied_slots(module)
    }
    if 1 not in occupied:
        modules.insert(0, {"slot": 1, "model": "M9032A"})

    bindings_by_name = {}
    for logical_index, name in enumerate(dc_channel_names):
        name = _virtual_name(name)
        if name in bindings_by_name:
            raise ValueError(
                f"QCS virtual channel {name!r} is bound to more than one GUI role"
            )
        bindings_by_name[name] = ("dc", logical_index)
    for logical_index, name in dict(rf_channel_names or {}).items():
        logical_index = _nonnegative_integer(
            logical_index,
            "QCS RF generator identifier",
        )
        name = _virtual_name(name)
        if name in bindings_by_name:
            raise ValueError(
                f"QCS virtual channel {name!r} is bound to more than one GUI role"
            )
        bindings_by_name[name] = ("rf", logical_index)
    if acquisition_channel_name:
        name = _virtual_name(acquisition_channel_name)
        if name in bindings_by_name:
            raise ValueError(
                f"QCS virtual channel {name!r} is bound to more than one GUI role"
            )
        bindings_by_name[name] = ("acquisition", 0)

    unassigned_index = 0
    mappings = []
    for channels in tuple(getattr(mapper, "channels", ())):
        labels = tuple(getattr(channels, "labels", ()))
        if len(labels) != 1:
            raise ValueError(
                f"QCS collection {getattr(channels, 'name', '')!r} has "
                f"{len(labels)} labels; PulseGenerator requires one label "
                "per named virtual channel"
            )
        physicals = tuple(mapper.get_physical_channels(channels))
        if len(physicals) != 1:
            raise ValueError(
                f"QCS collection {getattr(channels, 'name', '')!r} maps to "
                f"{len(physicals)} physical channels; PulseGenerator "
                "requires exactly one"
            )
        physical = physicals[0]
        instrument_name = str(physical.instrument)
        if instrument_name not in _INSTRUMENT_TO_MODEL:
            continue
        virtual_name = str(channels.name)
        role_binding = bindings_by_name.get(virtual_name)
        if role_binding is None:
            role = "unassigned"
            logical_index = unassigned_index
            unassigned_index += 1
        else:
            role, logical_index = role_binding
        if instrument_name not in QCS_ROLE_INSTRUMENTS[role]:
            raise ValueError(
                f"QCS virtual channel {virtual_name!r} is bound as {role}, "
                f"but its physical instrument is {instrument_name}"
            )
        address = physical.address
        lo_frequency_hz = None
        if instrument_name == "M5300AWG":
            settings = getattr(physical, "settings", None)
            setting_names = tuple(getattr(settings, "setting_names", ()))
            if "lo_frequency" in setting_names:
                lo_setting = getattr(settings, "lo_frequency", None)
                raw_lo_frequency = getattr(lo_setting, "value", None)
                if raw_lo_frequency is not None:
                    lo_frequency_hz = float(raw_lo_frequency)
        mappings.append(
            {
                "role": role,
                "logical_index": logical_index,
                "virtual_name": virtual_name,
                "label": int(labels[0]),
                "absolute_phase": bool(channels.absolute_phase),
                "lo_frequency_hz": lo_frequency_hz,
                "slot": int(address.slot),
                "channel": int(address.channel),
            }
        )

    downconverter_links = []
    for physical in physical_channels:
        if str(physical.instrument) != "M5200Digitizer":
            continue
        downconverter = mapper.get_downconverter(physical)
        if downconverter is None:
            continue
        if str(downconverter.instrument) != "M5201Downconverter":
            raise ValueError(
                f"QCS digitizer at {physical.address} uses unsupported "
                f"downconverter {downconverter.instrument}"
            )
        settings = getattr(downconverter, "settings", None)
        lo_setting = getattr(settings, "lo_frequency", None)
        raw_lo_frequency = getattr(lo_setting, "value", None)
        digitizer_address = physical.address
        downconverter_address = downconverter.address
        downconverter_links.append(
            {
                "digitizer_slot": int(digitizer_address.slot),
                "digitizer_channel": int(digitizer_address.channel),
                "downconverter_slot": int(downconverter_address.slot),
                "downconverter_channel": int(
                    downconverter_address.channel
                ),
                "lo_frequency_hz": (
                    None
                    if raw_lo_frequency is None
                    else float(raw_lo_frequency)
                ),
            }
        )

    return normalize_qcs_hardware_configuration(
        {
            "version": QCS_HARDWARE_CONFIGURATION_VERSION,
            "chassis_model": QCS_CHASSIS_MODEL,
            "chassis": next(iter(chassis_values), 1),
            "host_controller": next(iter(host_values), 1),
            "ip_address": getattr(mapper, "ip_address", None),
            "modules": modules,
            "channel_mappings": mappings,
            "downconverter_links": downconverter_links,
        }
    )


@lru_cache(maxsize=64)
def _cached_qcs_front_panel_png(
    serialized_configuration: str,
    highlighted_addresses: tuple[tuple[int, int], ...] = (),
) -> bytes:
    if _render_qcs_chassis_png is None:
        return b""
    return _render_qcs_chassis_png(
        json.loads(serialized_configuration),
        highlighted_addresses=highlighted_addresses,
    )


@lru_cache(maxsize=8)
def _cached_qcs_front_panel_image(png_bytes: bytes) -> QtGui.QImage:
    """Decode one rendered chassis PNG once for all compact previews."""

    image = QtGui.QImage()
    if png_bytes:
        image.loadFromData(png_bytes, "PNG")
    return image


def _qcs_front_panel_pixmap(
    configuration: Mapping[str, Any],
    highlighted_address: Optional[tuple[int, int]] = None,
    *,
    highlighted_addresses: Optional[Sequence[tuple[int, int]]] = None,
) -> QtGui.QPixmap:
    """Render a normalized configuration into a GUI-thread QPixmap."""

    serialized = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    )
    requested_highlights = []
    if highlighted_address is not None:
        requested_highlights.append(tuple(highlighted_address))
    if highlighted_addresses is not None:
        requested_highlights.extend(
            tuple(address) for address in highlighted_addresses
        )
    normalized_highlights = tuple(dict.fromkeys(requested_highlights))
    png_bytes = _cached_qcs_front_panel_png(serialized, normalized_highlights)
    image = _cached_qcs_front_panel_image(png_bytes)
    if not image.isNull():
        return QtGui.QPixmap.fromImage(image)
    return QtGui.QPixmap()


class QcsFrontPanelPreview(QtWidgets.QFrame):
    """Compact clickable M5000 preview for the main experiment editors."""

    activated = QtCore.pyqtSignal()

    def __init__(self, parent=None, *, image_path=None):
        super().__init__(parent)
        self._configuration = None
        self._role = "dc"
        self._logical_index: Optional[int] = 0
        self._render_error = None
        self._pixmap_refresh_pending = False
        self._fallback_pixmap = QtGui.QPixmap(
            str(
                Path(image_path)
                if image_path is not None
                else QCS_FRONT_PANEL_IMAGE_PATH
            )
        )
        self._pixmap = QtGui.QPixmap(self._fallback_pixmap)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip(
            "Open the Keysight QCS M5000 hardware configuration"
        )
        self.setMinimumSize(210, 108)
        self.setMaximumHeight(410)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred,
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumHeight(74)
        self.image_label.setAttribute(
            QtCore.Qt.WA_TransparentForMouseEvents,
            True,
        )
        self.image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.binding_label = QtWidgets.QLabel(self)
        self.binding_label.setAlignment(QtCore.Qt.AlignCenter)
        self.binding_label.setWordWrap(True)
        self.binding_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.binding_label.setAttribute(
            QtCore.Qt.WA_TransparentForMouseEvents,
            True,
        )
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.binding_label)
        self._refresh_image()
        self._refresh_binding()

    def sizeHint(self) -> QtCore.QSize:
        width = 540
        return QtCore.QSize(width, self.heightForWidth(width))

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(210, 108)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        pixmap = self._pixmap
        image_height = 74
        if not pixmap.isNull() and pixmap.width() > 0:
            image_width = max(1, int(width) - 12)
            image_height = round(
                image_width * pixmap.height() / pixmap.width()
            )
        binding_height = max(24, self.binding_label.sizeHint().height())
        return max(108, min(410, image_height + binding_height + 16))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_image()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._pixmap_refresh_pending:
            self._pixmap_refresh_pending = False
            self._refresh_pixmap()
            self.updateGeometry()

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.activated.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def set_configuration(
        self,
        configuration: Optional[Mapping[str, Any]],
    ) -> None:
        normalized = (
            None
            if configuration is None
            else normalize_qcs_hardware_configuration(configuration)
        )
        if normalized == self._configuration:
            return
        self._configuration = normalized
        self._request_pixmap_refresh()
        self._refresh_binding()
        self.updateGeometry()

    def _selected_address(self) -> Optional[tuple[int, int]]:
        if self._configuration is None or self._logical_index is None:
            return None
        mapping = next(
            (
                candidate
                for candidate in self._configuration["channel_mappings"]
                if candidate["role"] == self._role
                and int(candidate["logical_index"])
                == self._logical_index
            ),
            None,
        )
        if mapping is None:
            return None
        return int(mapping["slot"]), int(mapping["channel"])

    def _refresh_pixmap(self) -> None:
        self._render_error = None
        if self._configuration is None:
            pixmap = QtGui.QPixmap(self._fallback_pixmap)
        else:
            try:
                pixmap = _qcs_front_panel_pixmap(
                    self._configuration,
                    self._selected_address(),
                )
            except (OSError, TypeError, ValueError):
                pixmap = QtGui.QPixmap()
            if pixmap.isNull():
                self._render_error = (
                    "QCS front-panel preview unavailable.\n"
                    "Verify Pillow and the M5000 panel PNG assets."
                )
        self._pixmap = pixmap
        self._refresh_image()

    def _request_pixmap_refresh(self) -> None:
        """Render now unless this preview is hidden inside another widget."""

        if self.parentWidget() is not None and not self.isVisible():
            self._pixmap_refresh_pending = True
            return
        self._pixmap_refresh_pending = False
        self._refresh_pixmap()

    def set_selection(
        self,
        role: str,
        logical_index: Optional[int] = 0,
    ) -> None:
        role = str(role).strip().lower()
        if role not in QCS_ROLE_LABELS or role == "unassigned":
            raise ValueError(f"unsupported QCS preview role {role!r}")
        if logical_index is not None:
            logical_index = _nonnegative_integer(
                logical_index,
                "QCS preview logical index",
            )
        if role == self._role and logical_index == self._logical_index:
            return
        self._role = role
        self._logical_index = logical_index
        self._request_pixmap_refresh()
        self._refresh_binding()
        self.updateGeometry()

    def _refresh_image(self) -> None:
        if self._pixmap.isNull():
            self.image_label.clear()
            self.image_label.setText(
                self._render_error or "Keysight QCS M5000"
            )
            return
        available = self.image_label.size()
        if available.width() <= 1 or available.height() <= 1:
            available = QtCore.QSize(max(210, self.width() - 12), 90)
        self.image_label.setPixmap(
            self._pixmap.scaled(
                available,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def _refresh_binding(self) -> None:
        role_label = QCS_ROLE_LABELS[self._role]
        if self._logical_index is None:
            self.binding_label.setText(
                f"{role_label}: no logical channel selected; "
                "click to configure"
            )
            return
        if self._configuration is None:
            self.binding_label.setText(
                f"{role_label} {self._logical_index}: "
                "click to configure M5000 hardware"
            )
            return
        mapping = next(
            (
                candidate
                for candidate in self._configuration["channel_mappings"]
                if (
                    candidate["role"] == self._role
                    and int(candidate["logical_index"])
                    == self._logical_index
                )
            ),
            None,
        )
        if mapping is None:
            self.binding_label.setText(
                f"{role_label} {self._logical_index}: not mapped"
            )
            return
        modules_by_slot = {
            int(module["slot"]): module
            for module in self._configuration["modules"]
        }
        module = modules_by_slot.get(int(mapping["slot"]))
        model = "M5000 module" if module is None else str(module["model"])
        binding_text = (
            f"{mapping['virtual_name']}  |  {model} slot "
            f"{int(mapping['slot'])} ch{int(mapping['channel'])}"
        )
        if self._role == "rf" and model == "M5300A":
            lo_frequency_hz = mapping.get("lo_frequency_hz")
            lo_text = (
                "LO unset"
                if lo_frequency_hz is None
                else f"LO {float(lo_frequency_hz) / 1.0e9:.6g} GHz"
            )
            binding_text += f"  |  {lo_text}"
        if self._role == "acquisition":
            link = next(
                (
                    candidate
                    for candidate in self._configuration[
                        "downconverter_links"
                    ]
                    if int(candidate["digitizer_slot"])
                    == int(mapping["slot"])
                    and int(candidate["digitizer_channel"])
                    == int(mapping["channel"])
                ),
                None,
            )
            if link is not None:
                lo_frequency_hz = link["lo_frequency_hz"]
                lo_text = (
                    "LO unset"
                    if lo_frequency_hz is None
                    else f"LO {float(lo_frequency_hz) / 1.0e9:.6g} GHz"
                )
                binding_text += (
                    f"  |  via M5201A slot "
                    f"{int(link['downconverter_slot'])} pair "
                    f"{int(link['downconverter_channel'])}, {lo_text}"
                )
        self.binding_label.setText(binding_text)


class QcsM5201RouteDialog(QtWidgets.QDialog):
    """Graphical editor for one explicit M5201-to-M5200 acquisition route."""

    save_requested = QtCore.pyqtSignal(int, int, int, int, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Configure M5201A Acquisition Route")
        self._downconverter_slot = 0
        self._links_by_pair: dict[int, dict[str, Any]] = {}
        self._default_digitizer_address: Optional[tuple[int, int]] = None
        self._module_lo_ghz = 0.0

        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Declare the physical cable from an M5201A RF/IF pair to an "
            "M5200A digitizer SMA. This explicit route and the shared M5201A "
            "LO are written to the QCS mapper automatically."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QtWidgets.QFormLayout()
        self.module_label = QtWidgets.QLabel()
        self.pair_combo = QtWidgets.QComboBox()
        for pair in range(1, 5):
            self.pair_combo.addItem(f"RF/IF pair {pair}", pair)
        self.digitizer_combo = QtWidgets.QComboBox()
        self.lo_frequency_ghz = QtWidgets.QDoubleSpinBox()
        self.lo_frequency_ghz.setRange(0.0, 18.0)
        self.lo_frequency_ghz.setDecimals(9)
        self.lo_frequency_ghz.setSingleStep(0.1)
        self.lo_frequency_ghz.setSuffix(" GHz")
        self.lo_frequency_ghz.setSpecialValueText("Set LO frequency")
        self.lo_frequency_ghz.setToolTip(
            "M5201A uses one shared 1-18 GHz LO across all configured pairs"
        )
        form.addRow("Down-converter module:", self.module_label)
        form.addRow("M5201A input/pair:", self.pair_combo)
        form.addRow("Connected M5200A SMA:", self.digitizer_combo)
        form.addRow("Shared M5201A LO:", self.lo_frequency_ghz)
        layout.addLayout(form)

        self.route_note = QtWidgets.QLabel()
        self.route_note.setWordWrap(True)
        layout.addWidget(self.route_note)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save
            | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        self.save_button = buttons.button(QtWidgets.QDialogButtonBox.Save)
        self.save_button.setText("Apply Route Automatically")
        self.save_button.clicked.connect(self._submit)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.pair_combo.currentIndexChanged.connect(
            self._load_selected_pair
        )

    @staticmethod
    def _combo_index_for_address(
        combo: QtWidgets.QComboBox,
        address: Optional[tuple[int, int]],
    ) -> int:
        if address is None:
            return -1
        expected = (int(address[0]), int(address[1]))
        for index in range(combo.count()):
            raw = combo.itemData(index)
            if raw is not None and tuple(raw) == expected:
                return index
        return -1

    def set_route_options(
        self,
        *,
        downconverter_slot: int,
        initial_pair: int,
        digitizer_addresses: Sequence[tuple[int, int]],
        current_acquisition_address: Optional[tuple[int, int]],
        links: Sequence[Mapping[str, Any]],
    ) -> None:
        """Populate the dialog from the current graphical hardware draft."""

        self._downconverter_slot = int(downconverter_slot)
        self.module_label.setText(
            f"M5201A slot {self._downconverter_slot}"
        )
        self._links_by_pair = {
            int(link["downconverter_channel"]): dict(link)
            for link in links
            if int(link["downconverter_slot"])
            == self._downconverter_slot
        }
        self._default_digitizer_address = (
            None
            if current_acquisition_address is None
            else (
                int(current_acquisition_address[0]),
                int(current_acquisition_address[1]),
            )
        )
        module_los = [
            float(link["lo_frequency_hz"]) / 1.0e9
            for link in self._links_by_pair.values()
            if link.get("lo_frequency_hz") is not None
        ]
        self._module_lo_ghz = module_los[0] if module_los else 0.0

        self.digitizer_combo.clear()
        for slot, channel in sorted(
            (int(slot), int(channel))
            for slot, channel in digitizer_addresses
        ):
            self.digitizer_combo.addItem(
                f"M5200A slot {slot} CH{channel}",
                (slot, channel),
            )
        pair_index = self.pair_combo.findData(int(initial_pair))
        with QtCore.QSignalBlocker(self.pair_combo):
            self.pair_combo.setCurrentIndex(max(0, pair_index))
        self._load_selected_pair()

    def _load_selected_pair(self, *_args) -> None:
        pair = int(self.pair_combo.currentData() or 1)
        link = self._links_by_pair.get(pair)
        address = self._default_digitizer_address
        if link is not None:
            address = (
                int(link["digitizer_slot"]),
                int(link["digitizer_channel"]),
            )
        address_index = self._combo_index_for_address(
            self.digitizer_combo,
            address,
        )
        if address_index < 0 and self.digitizer_combo.count():
            address_index = 0
        self.digitizer_combo.setCurrentIndex(address_index)
        self.lo_frequency_ghz.setValue(self._module_lo_ghz)
        if link is None:
            self.route_note.setText(
                f"M5201A pair {pair} has no declared physical cable."
            )
        else:
            self.route_note.setText(
                f"Current cable: pair {pair} -> M5200A slot "
                f"{int(link['digitizer_slot'])} CH"
                f"{int(link['digitizer_channel'])}."
            )

    def _submit(self) -> None:
        raw_address = self.digitizer_combo.currentData()
        if raw_address is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No M5200A input",
                "Install an M5200A module and choose one of its SMA inputs.",
            )
            return
        lo_ghz = float(self.lo_frequency_ghz.value())
        if not 1.0 <= lo_ghz <= 18.0:
            QtWidgets.QMessageBox.warning(
                self,
                "M5201A LO required",
                "Set the shared M5201A LO frequency between 1 and 18 GHz.",
            )
            self.lo_frequency_ghz.setFocus(QtCore.Qt.OtherFocusReason)
            return
        digitizer_slot, digitizer_channel = tuple(raw_address)
        self.save_requested.emit(
            self._downconverter_slot,
            int(self.pair_combo.currentData()),
            int(digitizer_slot),
            int(digitizer_channel),
            lo_ghz * 1.0e9,
        )


class QcsFrontPanelControl(QtWidgets.QWidget):
    """Mapper builder plus a zoomable live M5000 front-panel preview."""

    identify_requested = QtCore.pyqtSignal(str)
    connector_selected = QtCore.pyqtSignal(str, int, int, int, bool)
    m5300_lo_frequency_changed = QtCore.pyqtSignal(
        str,
        int,
        int,
        int,
        float,
    )
    draft_staged = QtCore.pyqtSignal(object)
    settings_applied = QtCore.pyqtSignal(object)
    mapper_saved = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, *, image_path=None):
        super().__init__(parent)
        self._loading = False
        self._output_count = 1
        self._source_settings = {}
        self._mapper_write_allowed = True
        self._configuration_state = QCS_HARDWARE_STATE_EXTERNAL
        self._authoritative_fingerprint = None
        self._authoritative_mapper_path = None
        self._mapper_file_sha256 = None
        self._protected_mapper_paths = set()
        self._status_before_lock = None
        self._apply_preflight = None
        self._editing_enabled = True
        self._identifying_hardware = False
        self._syncing_downconverter_lo = False
        self._focused_mapping: Optional[tuple[str, int]] = None
        # RF-path previews such as Stability and S-Parameter select either
        # endpoint from one full chassis view. Other callers retain a strict
        # single-role focus.
        self._rf_acquisition_path_focus: Optional[
            tuple[tuple[str, int], tuple[str, int]]
        ] = None
        self._module_models_by_slot: dict[int, str] = {}
        self._image_scale = 0.45
        self._image_pixmap = QtGui.QPixmap()
        self._fallback_image_pixmap = QtGui.QPixmap()
        self._preview_refresh_timer = QtCore.QTimer(self)
        self._preview_refresh_timer.setSingleShot(True)
        self._preview_refresh_timer.setInterval(90)
        self._preview_refresh_timer.timeout.connect(
            self._refresh_reference_preview
        )
        self.ip_address = QtWidgets.QLineEdit(
            DEFAULT_QCS_IP_ADDRESS,
            self,
        )
        self.ip_address.setPlaceholderText("QCS controller IP address")
        self.ip_address.setMaximumWidth(180)
        self.ip_address.textChanged.connect(
            self._schedule_reference_refresh
        )
        self.ip_address.editingFinished.connect(
            self._stage_graphical_configuration
        )

        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(self.tabs, 1)
        self._build_reference_tab(
            Path(image_path) if image_path is not None
            else QCS_FRONT_PANEL_IMAGE_PATH
        )
        self._build_hardware_tab()
        # The tables on ``mapping_tab`` remain the internal, lossless recipe
        # used to build the native ChannelMapper.  Mapping is performed from
        # the graphical chassis, so the manual table editor is intentionally
        # not exposed as a front-panel tab.
        mapping_tab_index = self.tabs.indexOf(self.mapping_tab)
        if mapping_tab_index >= 0:
            self.tabs.removeTab(mapping_tab_index)
        self.mapping_tab.hide()
        self.tabs.tabBar().hide()
        self.tabs.currentChanged.connect(self._on_editor_tab_changed)
        self.tabs.setCurrentWidget(self.front_panel_tab)

        button_row = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel(
            "Click a highlighted channel SMA to assign it. The native QCS "
            "ChannelMapper is updated automatically."
        )
        self.status.setWordWrap(True)
        button_row.addWidget(self.status, 1)
        self.validate_button = QtWidgets.QPushButton("Validate")
        self.write_mapper_button = QtWidgets.QPushButton(
            "Save Mapper As..."
        )
        self.apply_button = QtWidgets.QPushButton("Apply to Experiment")
        self.apply_button.setDefault(True)
        button_row.addWidget(self.validate_button)
        button_row.addWidget(self.write_mapper_button)
        button_row.addWidget(self.apply_button)
        layout.addLayout(button_row)

        # Keep the normal front-panel window chassis-only. Less common RF LO,
        # M5201 link, imported-mapper, and explicit save controls live in a
        # separate advanced window instead of a second front-panel tab.
        for button in (
            self.validate_button,
            self.write_mapper_button,
            self.apply_button,
        ):
            button_row.removeWidget(button)
        self._build_advanced_hardware_dialog()
        self._m5201_route_dialog = QcsM5201RouteDialog(self)
        self._m5201_route_dialog.save_requested.connect(
            self._commit_m5201_route_dialog
        )

        self.validate_button.clicked.connect(self.validate_settings)
        self.write_mapper_button.clicked.connect(self.write_mapper)
        self.apply_button.clicked.connect(self.apply_settings)

    def _build_advanced_hardware_dialog(self) -> None:
        self._advanced_hardware_dialog = QtWidgets.QDialog(self)
        self._advanced_hardware_dialog.setModal(False)
        self._advanced_hardware_dialog.setWindowTitle(
            "Advanced QCS Hardware Settings"
        )
        self._advanced_hardware_dialog.resize(1180, 820)
        layout = QtWidgets.QVBoxLayout(self._advanced_hardware_dialog)
        self.mapping_tab.setParent(self._advanced_hardware_dialog)
        layout.addWidget(self.mapping_tab, 1)
        action_row = QtWidgets.QHBoxLayout()
        action_row.addStretch(1)
        action_row.addWidget(self.validate_button)
        action_row.addWidget(self.write_mapper_button)
        action_row.addWidget(self.apply_button)
        layout.addLayout(action_row)
        close_buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Close,
            parent=self._advanced_hardware_dialog,
        )
        close_buttons.rejected.connect(
            self._advanced_hardware_dialog.close
        )
        layout.addWidget(close_buttons)
        self.advanced_hardware_button.clicked.connect(
            self.show_advanced_hardware_settings
        )

    def show_advanced_hardware_settings(self) -> None:
        """Open optional RF/down-converter and imported-mapper controls."""

        if not self.tabs.isEnabled():
            return
        self.mapping_tab.show()
        self.validate_button.show()
        self.write_mapper_button.show()
        self.apply_button.show()
        self._advanced_hardware_dialog.show()
        self._advanced_hardware_dialog.raise_()
        self._advanced_hardware_dialog.activateWindow()

    def _build_hardware_tab(self) -> None:
        tab = QtWidgets.QWidget(self)
        tab_layout = QtWidgets.QVBoxLayout(tab)
        explanation = QtWidgets.QLabel(
            "Install, replace, or remove modules directly on the Front panel "
            "tab. Use this page to bind PulseGenerator DC, RF, and acquisition "
            "roles to the installed connectors. Saving creates a fresh native "
            "Keysight QCS ChannelMapper."
        )
        explanation.setWordWrap(True)
        explanation.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Maximum,
        )
        tab_layout.addWidget(explanation)

        mapper_group = QtWidgets.QGroupBox("ChannelMapper file")
        mapper_layout = QtWidgets.QHBoxLayout(mapper_group)
        self.mapper_path = QtWidgets.QLineEdit()
        self.mapper_path.setPlaceholderText("Path to a native .qcs ChannelMapper")
        self.load_mapper_button = QtWidgets.QPushButton("Load Mapper...")
        self.choose_mapper_button = QtWidgets.QToolButton()
        self.choose_mapper_button.setText("...")
        self.choose_mapper_button.setToolTip("Choose a mapper output path")
        mapper_layout.addWidget(self.mapper_path, 1)
        mapper_layout.addWidget(self.choose_mapper_button)
        mapper_layout.addWidget(self.load_mapper_button)
        tab_layout.addWidget(mapper_group)
        self.choose_mapper_button.clicked.connect(self._choose_mapper_output)
        self.load_mapper_button.clicked.connect(self.load_mapper)

        chassis_group = QtWidgets.QGroupBox("Chassis")
        chassis_form = QtWidgets.QFormLayout(chassis_group)
        self.chassis_model = QtWidgets.QLineEdit(QCS_CHASSIS_MODEL)
        self.chassis_model.setReadOnly(True)
        self.chassis_number = QtWidgets.QSpinBox()
        self.chassis_number.setRange(1, 9999)
        self.chassis_number.setValue(1)
        self.host_controller = QtWidgets.QSpinBox()
        self.host_controller.setRange(1, 9999)
        self.host_controller.setValue(1)
        self.chassis_number.valueChanged.connect(
            self._schedule_reference_refresh
        )
        self.chassis_number.valueChanged.connect(
            self._stage_graphical_configuration
        )
        self.host_controller.valueChanged.connect(
            self._schedule_reference_refresh
        )
        self.host_controller.valueChanged.connect(
            self._stage_graphical_configuration
        )
        chassis_form.addRow("Chassis model:", self.chassis_model)
        chassis_form.addRow("Chassis number:", self.chassis_number)
        chassis_form.addRow("Host controller:", self.host_controller)
        tab_layout.addWidget(chassis_group)

        mapping_group = QtWidgets.QGroupBox(
            "PulseGenerator virtual-to-physical channel bindings"
        )
        mapping_layout = QtWidgets.QVBoxLayout(mapping_group)
        self.mapping_table = QtWidgets.QTableWidget(0, 9, mapping_group)
        self.mapping_table.setHorizontalHeaderLabels(
            (
                "Role",
                "GUI channel",
                "Virtual name",
                "QCS label",
                "Absolute phase",
                "LO (GHz)",
                "Slot",
                "Connector",
                "Resolved instrument",
            )
        )
        self.mapping_table.verticalHeader().setVisible(False)
        self.mapping_table.setAlternatingRowColors(True)
        self.mapping_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.mapping_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        self.mapping_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeToContents
        )
        self.mapping_table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.Stretch
        )
        self.mapping_table.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeToContents
        )
        self.mapping_table.horizontalHeader().setSectionResizeMode(
            4, QtWidgets.QHeaderView.ResizeToContents
        )
        self.mapping_table.horizontalHeader().setSectionResizeMode(
            5, QtWidgets.QHeaderView.ResizeToContents
        )
        for column in range(3, 9):
            self.mapping_table.horizontalHeader().setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeToContents
            )
        mapping_layout.addWidget(self.mapping_table)
        mapping_buttons = QtWidgets.QHBoxLayout()
        self.add_mapping_button = QtWidgets.QPushButton("Add mapping")
        self.remove_mapping_button = QtWidgets.QPushButton(
            "Remove selected"
        )
        self.diagram_defaults_button = QtWidgets.QPushButton(
            "Restore diagram layout"
        )
        mapping_buttons.addWidget(self.add_mapping_button)
        mapping_buttons.addWidget(self.remove_mapping_button)
        mapping_buttons.addStretch(1)
        mapping_buttons.addWidget(self.diagram_defaults_button)
        mapping_layout.addLayout(mapping_buttons)
        self.add_mapping_button.clicked.connect(self._add_empty_mapping)
        self.remove_mapping_button.clicked.connect(
            self._remove_selected_mappings
        )
        self.diagram_defaults_button.clicked.connect(
            self._restore_diagram_layout
        )
        tab_layout.addWidget(mapping_group, 1)

        downconverter_group = QtWidgets.QGroupBox(
            "M5201A Down Converter links"
        )
        downconverter_layout = QtWidgets.QVBoxLayout(downconverter_group)
        downconverter_note = QtWidgets.QLabel(
            "Explicitly pair each mapped M5200A digitizer channel with one "
            "M5201A RF/IF channel pair. Links are never inferred from the "
            "installed-module inventory."
        )
        downconverter_note.setWordWrap(True)
        downconverter_layout.addWidget(downconverter_note)
        self.downconverter_table = QtWidgets.QTableWidget(
            0,
            6,
            downconverter_group,
        )
        self.downconverter_table.setHorizontalHeaderLabels(
            (
                "M5200 slot",
                "Digitizer CH",
                "M5201 slot",
                "RF/IF pair",
                "Shared LO (GHz)",
                "Resolved link",
            )
        )
        self.downconverter_table.verticalHeader().setVisible(False)
        self.downconverter_table.setAlternatingRowColors(True)
        self.downconverter_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        for column in range(5):
            self.downconverter_table.horizontalHeader().setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.ResizeToContents,
            )
        self.downconverter_table.horizontalHeader().setSectionResizeMode(
            5,
            QtWidgets.QHeaderView.Stretch,
        )
        self.downconverter_table.setMaximumHeight(185)
        downconverter_layout.addWidget(self.downconverter_table)
        downconverter_buttons = QtWidgets.QHBoxLayout()
        self.add_downconverter_button = QtWidgets.QPushButton(
            "Add M5201 link"
        )
        self.remove_downconverter_button = QtWidgets.QPushButton(
            "Remove selected link"
        )
        downconverter_buttons.addWidget(self.add_downconverter_button)
        downconverter_buttons.addWidget(self.remove_downconverter_button)
        downconverter_buttons.addStretch(1)
        downconverter_layout.addLayout(downconverter_buttons)
        self.add_downconverter_button.clicked.connect(
            self._add_downconverter_link
        )
        self.remove_downconverter_button.clicked.connect(
            self._remove_selected_downconverter_links
        )
        tab_layout.addWidget(downconverter_group)
        self.downconverter_group = downconverter_group
        self.mapping_tab = tab
        self.tabs.addTab(tab, "Channel mappings")

    def _build_reference_tab(self, image_path: Path) -> None:
        tab = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(tab)
        controls = QtWidgets.QHBoxLayout()
        self.qcs_server_ip_label = QtWidgets.QLabel("QCS server IP:")
        self.identify_hardware_button = QtWidgets.QPushButton(
            "Identify Hardware Configuration"
        )
        self.identify_hardware_button.setToolTip(
            "Read installed module models and their host-controller, chassis, "
            "and slot addresses from this QCS controller. Compatible channel "
            "mappings are preserved; external cables are not inferred."
        )
        self.advanced_hardware_button = QtWidgets.QPushButton(
            "Advanced Hardware Settings..."
        )
        self.advanced_hardware_button.setToolTip(
            "Configure RF local oscillators, explicit M5201 links, or an "
            "imported third-party mapper"
        )
        fit_button = QtWidgets.QPushButton("Fit")
        actual_button = QtWidgets.QPushButton("100%")
        zoom_out_button = QtWidgets.QPushButton("-")
        zoom_in_button = QtWidgets.QPushButton("+")
        controls.addWidget(self.qcs_server_ip_label)
        controls.addWidget(self.ip_address)
        controls.addWidget(self.identify_hardware_button)
        controls.addWidget(self.advanced_hardware_button)
        controls.addStretch(1)
        controls.addWidget(fit_button)
        controls.addWidget(actual_button)
        controls.addWidget(zoom_out_button)
        controls.addWidget(zoom_in_button)
        self.reference_size_label = QtWidgets.QLabel()
        controls.addWidget(self.reference_size_label)
        layout.addLayout(controls)
        note = QtWidgets.QLabel(
            "Click an empty slot to install a module. Click an installed module "
            "to replace or remove it. Channel addresses are shown without "
            "inferred cables."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.reference_scroll = QtWidgets.QScrollArea(tab)
        self.reference_scroll.setWidgetResizable(False)
        self.reference_scroll.setAlignment(QtCore.Qt.AlignCenter)
        self.reference_label = QtWidgets.QLabel()
        self.reference_label.setAlignment(QtCore.Qt.AlignCenter)
        self.reference_label.setCursor(QtCore.Qt.PointingHandCursor)
        self.reference_label.setToolTip(
            "Click an empty slot to install a module, or click a module to "
            "replace or remove it."
        )
        self.reference_label.installEventFilter(self)
        self.reference_scroll.setWidget(self.reference_label)
        layout.addWidget(self.reference_scroll, 1)
        self.front_panel_tab = tab
        self.tabs.addTab(tab, "Front panel")

        self._fallback_image_pixmap = QtGui.QPixmap(str(image_path))
        self._image_pixmap = QtGui.QPixmap(self._fallback_image_pixmap)
        if self._image_pixmap.isNull():
            self.reference_label.setText(
                f"QCS front-panel image is unavailable:\n{image_path}"
            )
            self.reference_label.adjustSize()
        else:
            self._set_image_scale(self._image_scale)
        fit_button.clicked.connect(self._fit_reference_image)
        actual_button.clicked.connect(lambda: self._set_image_scale(1.0))
        zoom_out_button.clicked.connect(
            lambda: self._set_image_scale(self._image_scale / 1.25)
        )
        zoom_in_button.clicked.connect(
            lambda: self._set_image_scale(self._image_scale * 1.25)
        )
        self.identify_hardware_button.clicked.connect(
            self._request_hardware_identification
        )

    def _set_reference_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        if pixmap.isNull():
            return
        self._image_pixmap = QtGui.QPixmap(pixmap)
        self.reference_label.clear()
        self._set_image_scale(self._image_scale)

    def _show_reference_preview_unavailable(self) -> None:
        self._image_pixmap = QtGui.QPixmap()
        self.reference_label.clear()
        self.reference_label.setText(
            "Live QCS front-panel preview is unavailable.\n"
            "Verify Pillow and the M5000 panel PNG assets."
        )
        self.reference_label.adjustSize()
        self.reference_size_label.setText("Unavailable")

    def _set_image_scale(self, scale: float) -> None:
        if self._image_pixmap.isNull():
            return
        self._image_scale = min(2.0, max(0.1, float(scale)))
        size = self._image_pixmap.size() * self._image_scale
        scaled = self._image_pixmap.scaled(
            size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.reference_label.setPixmap(scaled)
        self.reference_label.resize(scaled.size())
        self.reference_size_label.setText(
            f"{self._image_scale * 100:.0f}%"
        )

    def _fit_reference_image(self) -> None:
        if self._image_pixmap.isNull():
            return
        available = self.reference_scroll.viewport().size()
        if available.width() <= 1 or available.height() <= 1:
            return
        scale = min(
            available.width() / self._image_pixmap.width(),
            available.height() / self._image_pixmap.height(),
        )
        self._set_image_scale(scale)

    def show_front_panel(self) -> None:
        """Show and fit the chassis-first editor surface."""

        self.tabs.setCurrentWidget(self.front_panel_tab)
        QtCore.QTimer.singleShot(0, self._fit_reference_image)

    def _close_m5201_route_dialog(self) -> None:
        route_dialog = getattr(self, "_m5201_route_dialog", None)
        if route_dialog is not None and route_dialog.isVisible():
            route_dialog.reject()

    def hideEvent(self, event) -> None:
        advanced_dialog = getattr(self, "_advanced_hardware_dialog", None)
        if advanced_dialog is not None and advanced_dialog.isVisible():
            advanced_dialog.close()
        self._close_m5201_route_dialog()
        super().hideEvent(event)

    def _on_editor_tab_changed(self, _index: int) -> None:
        if self.tabs.currentWidget() is self.front_panel_tab:
            QtCore.QTimer.singleShot(0, self._fit_reference_image)

    def _request_hardware_identification(self) -> None:
        try:
            ip_address = normalize_qcs_server_ip(self.ip_address.text())
        except ValueError as exc:
            self._set_status(str(exc), error=True)
            self.ip_address.setFocus(QtCore.Qt.OtherFocusReason)
            self.ip_address.selectAll()
            return
        self.ip_address.setText(ip_address)
        self.identify_requested.emit(ip_address)

    def set_identifying(
        self,
        identifying: bool,
        message: str,
        *,
        error: bool = False,
    ) -> None:
        """Set the busy state used while the controller inventory RPC runs."""

        self._identifying_hardware = bool(identifying)
        self.identify_hardware_button.setText(
            "Identifying..."
            if self._identifying_hardware
            else "Identify Hardware Configuration"
        )
        self._refresh_editing_enabled_state()
        self._set_status(message, error=error)

    def _select_discovered_inventory(
        self,
        discovery: Mapping[str, Any],
    ) -> Optional[Mapping[str, Any]]:
        raw_inventories = discovery.get("inventories")
        if not isinstance(raw_inventories, (list, tuple)):
            raise TypeError(
                "QCS hardware identification returned no chassis inventory"
            )
        inventories = [
            inventory
            for inventory in raw_inventories
            if isinstance(inventory, Mapping)
        ]
        if not inventories:
            raise ValueError(
                "QCS hardware identification returned no chassis inventory"
            )
        if len(inventories) == 1:
            return inventories[0]

        current_key = (
            self.host_controller.value(),
            self.chassis_number.value(),
        )
        matches = [
            inventory
            for inventory in inventories
            if (
                int(inventory.get("host_controller", 0)),
                int(inventory.get("chassis", 0)),
            )
            == current_key
        ]
        if len(matches) == 1:
            return matches[0]

        labels = [
            (
                f"Host controller {int(inventory['host_controller'])}, "
                f"chassis {int(inventory['chassis'])} "
                f"({len(inventory.get('modules', ()))} modules)"
            )
            for inventory in inventories
        ]
        selected_label, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Select discovered QCS chassis",
            "The QCS controller reported more than one chassis:",
            labels,
            0,
            False,
        )
        if not accepted:
            return None
        return inventories[labels.index(selected_label)]

    def apply_discovered_hardware_configuration(
        self,
        discovery: Mapping[str, Any],
        *,
        commit_callback=None,
    ) -> bool:
        """Make one discovered chassis the active front-panel configuration."""

        try:
            if not isinstance(discovery, Mapping):
                raise TypeError(
                    "QCS hardware identification result must be an object"
                )
            ip_address = normalize_qcs_server_ip(
                discovery.get("ip_address", self.ip_address.text())
            )
            inventory = self._select_discovered_inventory(discovery)
            if inventory is None:
                self._set_status(
                    "Hardware identification was not applied.",
                    error=True,
                )
                return False
            current = normalize_qcs_hardware_configuration(
                {
                    "version": QCS_HARDWARE_CONFIGURATION_VERSION,
                    "chassis_model": self.chassis_model.text().strip(),
                    "chassis": self.chassis_number.value(),
                    "host_controller": self.host_controller.value(),
                    "ip_address": self.ip_address.text().strip() or None,
                    "modules": self._modules_from_widgets(),
                    "channel_mappings": self._mappings_from_widgets(),
                    "downconverter_links": (
                        self._downconverter_links_from_widgets()
                    ),
                }
            )
            merged, removed_mappings = (
                merge_qcs_discovered_hardware_configuration(
                    current,
                    inventory,
                    ip_address=ip_address,
                )
            )
            removed_links = [
                dict(link)
                for link in current["downconverter_links"]
                if link not in merged["downconverter_links"]
            ]
        except (KeyError, OSError, TypeError, ValueError) as exc:
            self._set_status(
                f"Discovered hardware was not applied: {exc}",
                error=True,
            )
            return False

        if removed_mappings or removed_links:
            names = [
                str(mapping["virtual_name"])
                for mapping in removed_mappings
            ]
            link_names = [
                (
                    f"M5200 S{link['digitizer_slot']} "
                    f"CH{link['digitizer_channel']} <- M5201 "
                    f"S{link['downconverter_slot']} "
                    f"CH{link['downconverter_channel']}"
                )
                for link in removed_links
            ]
            removal_details = []
            if names:
                removal_details.append(
                    f"{len(names)} channel mapping(s): {', '.join(names)}"
                )
            if link_names:
                removal_details.append(
                    f"{len(link_names)} M5201 link(s): "
                    f"{', '.join(link_names)}"
                )
            answer = QtWidgets.QMessageBox.question(
                self,
                "Update discovered QCS hardware?",
                "The installed topology is incompatible with:\n"
                + "\n".join(removal_details)
                + "\n\nUpdate the front panel and remove only these "
                "items? No channels or downconverter links will be "
                "automatically rerouted.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                self._set_status(
                    "Discovered hardware was not applied; the current "
                    "configuration is unchanged.",
                    error=True,
                )
                return False

        if not self._apply_preflight_passes():
            return False
        settings = None
        if not removed_mappings and not removed_links:
            try:
                settings = self._settings_for_configuration(merged)
            except (OSError, TypeError, ValueError) as exc:
                self._set_status(
                    "Discovered hardware was not applied; the candidate "
                    f"configuration is incomplete or invalid: {exc}",
                    error=True,
                )
                return False

        self._set_configuration_widgets(merged)
        effective_state = (
            self._effective_configuration_state(
                merged,
                self.mapper_path.text().strip(),
            )
            if settings is None
            else settings["hardware_configuration_state"]
        )
        self._configuration_state = effective_state
        if effective_state not in {
            QCS_HARDWARE_STATE_SAVED,
            QCS_HARDWARE_STATE_IMPORTED,
        }:
            self._mapper_file_sha256 = None

        module_summary = ", ".join(
            f"{module['model']} slot {int(module['slot'])}"
            for module in merged["modules"]
        )
        address_summary = (
            f"host controller {merged['host_controller']}, "
            f"chassis {merged['chassis']}"
        )
        if removed_mappings or removed_links:
            removed_summary = []
            if removed_mappings:
                removed_summary.append(
                    f"{len(removed_mappings)} incompatible mapping(s)"
                )
            if removed_links:
                removed_summary.append(
                    f"{len(removed_links)} incompatible M5201 link(s)"
                )
            self._set_status(
                f"Identified {address_summary}: {module_summary}. Updated the "
                f"front panel and removed {' and '.join(removed_summary)}; "
                "reselect each affected SMA from its experiment panel. "
                "Explicit M5201A cable links are never rerouted "
                "automatically.",
                error=True,
            )
            self.draft_staged.emit(merged)
            return True

        self._configuration_state = settings[
            "hardware_configuration_state"
        ]
        self._mapper_file_sha256 = settings["hardware_mapper_sha256"]
        self._source_settings = dict(settings)
        if commit_callback is not None:
            try:
                committed = commit_callback(settings)
            except (OSError, TypeError, ValueError) as exc:
                committed = False
                rejection_detail = f": {exc}"
            else:
                rejection_detail = ""
            if committed is False:
                # Signal delivery cannot return the receiver's result.  Keep
                # the newly identified topology as the shared pending draft
                # when the owning experiment rejects a synchronous commit.
                self._configuration_state = QCS_HARDWARE_STATE_DRAFT
                self._mapper_file_sha256 = None
                pending_settings = dict(settings)
                pending_settings["hardware_configuration_state"] = (
                    QCS_HARDWARE_STATE_DRAFT
                )
                pending_settings["hardware_mapper_sha256"] = None
                self._source_settings = pending_settings
                self.draft_staged.emit(merged)
                self._set_status(
                    f"Identified {address_summary}: {module_summary}, but the "
                    "experiment could not apply it"
                    f"{rejection_detail}. The identified chassis is retained "
                    "as a pending front-panel draft.",
                    error=True,
                )
                return True
        else:
            self.settings_applied.emit(settings)
        state_suffix = (
            " The mapper is now an unsaved draft."
            if self._configuration_state == QCS_HARDWARE_STATE_DRAFT
            else (
                " The imported mapper topology is now modified."
                if self._configuration_state
                == QCS_HARDWARE_STATE_IMPORTED_DIRTY
                else ""
            )
        )
        self._set_status(
            f"Identified and applied {address_summary}: {module_summary}."
            f"{state_suffix}",
            error=False,
        )
        return True

    def show_m5201_route_dialog(
        self,
        downconverter_slot: int,
        downconverter_pair: Optional[int] = None,
    ) -> bool:
        """Open the graphical M5201 cable/LO editor for acquisition."""

        if (
            not self.tabs.isEnabled()
            or self._focused_mapping != ("acquisition", 0)
        ):
            return False
        downconverter_slot = _positive_integer(
            downconverter_slot,
            "QCS M5201 slot",
        )
        if downconverter_pair is not None:
            downconverter_pair = _positive_integer(
                downconverter_pair,
                "QCS M5201 pair",
            )
            if downconverter_pair > 4:
                raise ValueError("M5201A exposes pairs 1-4")
        if (
            self._module_models_by_slot.get(downconverter_slot)
            != "M5201A"
        ):
            self._set_status(
                f"Slot {downconverter_slot} does not contain an M5201A.",
                error=True,
            )
            return False
        if not self._mapper_write_allowed:
            self._set_status(
                "M5201 cable or LO changes cannot rewrite an imported "
                "third-party mapper. Use Advanced Hardware Settings to "
                "restore an app-owned mapper first.",
                error=True,
            )
            return False
        configuration = self._preview_configuration_from_widgets()
        if configuration is None:
            self._set_status(
                "The current hardware draft is invalid and cannot configure "
                "an M5201A route.",
                error=True,
            )
            return False
        current_address = self._focused_mapping_address(configuration)
        mappings_by_address = {
            (int(mapping["slot"]), int(mapping["channel"])): mapping
            for mapping in configuration["channel_mappings"]
        }
        linked_addresses = {
            (
                int(link["digitizer_slot"]),
                int(link["digitizer_channel"]),
            )
            for link in configuration["downconverter_links"]
            if int(link["downconverter_slot"]) == downconverter_slot
        }
        digitizer_addresses = []
        for module in configuration["modules"]:
            if str(module["model"]) != "M5200A":
                continue
            slot = int(module["slot"])
            for channel in range(1, 5):
                address = (slot, channel)
                occupant = mappings_by_address.get(address)
                if (
                    occupant is None
                    or address == current_address
                    or address in linked_addresses
                ):
                    digitizer_addresses.append(address)
        if not digitizer_addresses:
            self._set_status(
                "No unassigned M5200A SMA is available for acquisition.",
                error=True,
            )
            return False
        links = [dict(link) for link in configuration["downconverter_links"]]
        if downconverter_pair is None:
            current_link = next(
                (
                    link
                    for link in links
                    if int(link["downconverter_slot"])
                    == downconverter_slot
                    and current_address is not None
                    and (
                        int(link["digitizer_slot"]),
                        int(link["digitizer_channel"]),
                    )
                    == current_address
                ),
                None,
            )
            if current_link is None:
                current_link = next(
                    (
                        link
                        for link in links
                        if int(link["downconverter_slot"])
                        == downconverter_slot
                    ),
                    None,
                )
            downconverter_pair = (
                1
                if current_link is None
                else int(current_link["downconverter_channel"])
            )
        self._m5201_route_dialog.set_route_options(
            downconverter_slot=downconverter_slot,
            initial_pair=downconverter_pair,
            digitizer_addresses=digitizer_addresses,
            current_acquisition_address=current_address,
            links=links,
        )
        self._m5201_route_dialog.show()
        self._m5201_route_dialog.raise_()
        self._m5201_route_dialog.activateWindow()
        return True

    def _commit_m5201_route_dialog(
        self,
        downconverter_slot: int,
        downconverter_pair: int,
        digitizer_slot: int,
        digitizer_channel: int,
        lo_frequency_hz: float,
    ) -> None:
        configured = self.configure_m5201_route(
            downconverter_slot=downconverter_slot,
            downconverter_pair=downconverter_pair,
            digitizer_slot=digitizer_slot,
            digitizer_channel=digitizer_channel,
            lo_frequency_hz=lo_frequency_hz,
        )
        if configured and self._m5201_route_is_applied():
            self._m5201_route_dialog.accept()

    def _m5201_route_is_applied(self) -> bool:
        """Return whether the synchronous auto-apply committed this draft."""

        if self._configuration_state not in {
            QCS_HARDWARE_STATE_SAVED,
            QCS_HARDWARE_STATE_IMPORTED,
        }:
            return False
        source_configuration = self._source_settings.get(
            "hardware_configuration"
        )
        if not isinstance(source_configuration, Mapping):
            return False
        try:
            return normalize_qcs_hardware_configuration(
                source_configuration
            ) == self.working_configuration()
        except (TypeError, ValueError):
            return False

    def _mapping_row_for_address(
        self,
        slot: int,
        channel: int,
    ) -> Optional[int]:
        """Return the unique virtual mapping bound to one physical address."""

        expected = int(slot), int(channel)
        for row in range(self.mapping_table.rowCount()):
            address = (
                int(self.mapping_table.cellWidget(row, 6).value()),
                int(self.mapping_table.cellWidget(row, 7).value()),
            )
            if address == expected:
                return row
        return None

    def set_m5300_lo_frequency(
        self,
        slot: int,
        channel: int,
        lo_frequency_hz: float,
    ) -> bool:
        """Set one mapped M5300 SMA LO and request native mapper persistence.

        m5300_lo_frequency_changed is emitted exactly once after a real value
        change. Its arguments are role, logical index, slot, channel, and
        frequency in Hz, allowing the owning window to reuse the same
        asynchronous mapper-save path as a connector selection.
        """

        if not self.tabs.isEnabled():
            return False
        slot = _positive_integer(slot, "QCS M5300 slot")
        channel = _positive_integer(channel, "QCS M5300 channel")
        lo_frequency_hz = float(lo_frequency_hz)
        if (
            not isfinite(lo_frequency_hz)
            or not 0.0 <= lo_frequency_hz <= 18.0e9
        ):
            raise ValueError("M5300 LO frequency must be in [0, 18] GHz")
        if self._module_models_by_slot.get(slot) != "M5300A":
            self._set_status(
                f"Slot {slot} does not contain an M5300A RF AWG.",
                error=True,
            )
            return False
        row = self._mapping_row_for_address(slot, channel)
        if row is None:
            self._set_status(
                f"M5300A slot {slot} CH{channel} has no virtual-channel "
                "mapping. Map this SMA to an RF output before setting its "
                "LO frequency.",
                error=True,
            )
            return False
        if not self._mapper_write_allowed:
            self._set_status(
                "The M5300 LO cannot be changed in an imported third-party "
                "mapper because the editor cannot safely preserve all of "
                "its native settings. Restore the diagram layout and save "
                "an app-owned mapper first.",
                error=True,
            )
            return False

        lo_widget = self.mapping_table.cellWidget(row, 5)
        old_text = lo_widget.text().strip()
        old_frequency_hz = (
            None if not old_text else float(old_text) * 1.0e9
        )
        if (
            old_frequency_hz is not None
            and abs(old_frequency_hz - lo_frequency_hz) <= 0.5
        ):
            self._set_status(
                f"M5300A slot {slot} CH{channel} already uses "
                f"{lo_frequency_hz / 1.0e9:.12g} GHz LO.",
                error=False,
            )
            self._refresh_reference_preview()
            return True

        with QtCore.QSignalBlocker(lo_widget):
            lo_widget.setText(f"{lo_frequency_hz / 1.0e9:.12g}")
        configuration = self.working_configuration()
        self._configuration_state = self._effective_configuration_state(
            configuration,
            self.mapper_path.text().strip(),
        )
        if self._configuration_state not in {
            QCS_HARDWARE_STATE_SAVED,
            QCS_HARDWARE_STATE_IMPORTED,
        }:
            self._mapper_file_sha256 = None
        self.mapping_table.selectRow(row)
        self._preview_refresh_timer.stop()
        self._refresh_reference_preview()
        role = str(self.mapping_table.cellWidget(row, 0).currentData())
        logical_index = int(
            self.mapping_table.cellWidget(row, 1).value()
        )
        self._set_status(
            f"Set M5300A slot {slot} CH{channel} LO to "
            f"{lo_frequency_hz / 1.0e9:.12g} GHz. Regenerating the native "
            "QCS mapper automatically...",
            error=False,
        )
        self.m5300_lo_frequency_changed.emit(
            role,
            logical_index,
            slot,
            channel,
            lo_frequency_hz,
        )
        return True

    def _prompt_m5300_lo_frequency(
        self,
        slot: int,
        channel: int,
    ) -> bool:
        """Prompt once for the LO of a mapped M5300 front-panel SMA."""

        if self._module_models_by_slot.get(int(slot)) != "M5300A":
            return False
        row = self._mapping_row_for_address(slot, channel)
        if row is None:
            self._set_status(
                f"M5300A slot {int(slot)} CH{int(channel)} has no "
                "virtual-channel mapping. Map this SMA to an RF output "
                "before setting its LO frequency.",
                error=True,
            )
            return False
        if not self._mapper_write_allowed:
            self._set_status(
                "The M5300 LO cannot be changed in an imported third-party "
                "mapper. Restore the diagram layout and save an app-owned "
                "mapper first.",
                error=True,
            )
            return False
        lo_text = self.mapping_table.cellWidget(row, 5).text().strip()
        current_ghz = 5.0 if not lo_text else float(lo_text)
        lo_frequency_ghz, accepted = QtWidgets.QInputDialog.getDouble(
            self,
            "M5300 RF Output LO",
            (
                f"M5300A slot {int(slot)} CH{int(channel)} local-oscillator "
                "frequency [GHz]:"
            ),
            current_ghz,
            0.0,
            18.0,
            9,
        )
        if not accepted:
            self._set_status(
                f"M5300A slot {int(slot)} CH{int(channel)} LO change was "
                "cancelled.",
                error=False,
            )
            return False
        return self.set_m5300_lo_frequency(
            slot,
            channel,
            float(lo_frequency_ghz) * 1.0e9,
        )

    def configure_m5201_route(
        self,
        *,
        downconverter_slot: int,
        downconverter_pair: int,
        digitizer_slot: int,
        digitizer_channel: int,
        lo_frequency_hz: float,
    ) -> bool:
        """Atomically declare a physical M5201-to-M5200 acquisition route."""

        if (
            not self.tabs.isEnabled()
            or self._focused_mapping != ("acquisition", 0)
        ):
            return False
        downconverter_slot = _positive_integer(
            downconverter_slot,
            "QCS M5201 slot",
        )
        downconverter_pair = _positive_integer(
            downconverter_pair,
            "QCS M5201 pair",
        )
        digitizer_slot = _positive_integer(
            digitizer_slot,
            "QCS M5200 slot",
        )
        digitizer_channel = _positive_integer(
            digitizer_channel,
            "QCS M5200 channel",
        )
        lo_frequency_hz = float(lo_frequency_hz)
        if downconverter_pair > 4:
            self._set_status("M5201A exposes pairs 1-4.", error=True)
            return False
        if digitizer_channel > 4:
            self._set_status("M5200A exposes channels 1-4.", error=True)
            return False
        if (
            not isfinite(lo_frequency_hz)
            or not 1.0e9 <= lo_frequency_hz <= 18.0e9
        ):
            self._set_status(
                "M5201A LO frequency must be between 1 and 18 GHz.",
                error=True,
            )
            return False
        if not self._mapper_write_allowed:
            self._set_status(
                "An imported third-party mapper cannot be rewritten with a "
                "new M5201 cable or LO setting.",
                error=True,
            )
            return False
        try:
            current = self.working_configuration()
        except (TypeError, ValueError) as exc:
            self._set_status(str(exc), error=True)
            return False
        modules_by_slot = {
            int(module["slot"]): str(module["model"])
            for module in current["modules"]
        }
        if modules_by_slot.get(downconverter_slot) != "M5201A":
            self._set_status(
                f"Slot {downconverter_slot} does not contain an M5201A.",
                error=True,
            )
            return False
        if modules_by_slot.get(digitizer_slot) != "M5200A":
            self._set_status(
                f"Slot {digitizer_slot} does not contain an M5200A.",
                error=True,
            )
            return False

        mappings = [dict(mapping) for mapping in current["channel_mappings"]]
        acquisition_index = next(
            (
                index
                for index, mapping in enumerate(mappings)
                if str(mapping["role"]) == "acquisition"
                and int(mapping["logical_index"]) == 0
            ),
            None,
        )
        selected_address = (digitizer_slot, digitizer_channel)
        occupied_index = next(
            (
                index
                for index, mapping in enumerate(mappings)
                if index != acquisition_index
                and (
                    int(mapping["slot"]),
                    int(mapping["channel"]),
                )
                == selected_address
            ),
            None,
        )
        occupied = (
            None
            if occupied_index is None
            else mappings[occupied_index]
        )
        if (
            occupied is not None
            and str(occupied["role"]) != "unassigned"
        ):
            self._set_status(
                f"M5200A slot {digitizer_slot} CH{digitizer_channel} is "
                f"already assigned to {occupied['virtual_name']!r}. Choose "
                "an unused SMA or an M5200A spare channel.",
                error=True,
            )
            return False

        generated_acquisition_name = None
        previous_address = None
        if acquisition_index is None:
            if occupied_index is not None:
                promoted = mappings[occupied_index]
                promoted["role"] = "acquisition"
                promoted["logical_index"] = 0
                promoted["absolute_phase"] = True
                promoted["lo_frequency_hz"] = None
                generated_acquisition_name = str(
                    promoted["virtual_name"]
                )
                acquisition_index = occupied_index
            else:
                _dc_names, _rf_names, acquisition_name = (
                    self._current_source_bindings()
                )
                used_names = {
                    str(mapping["virtual_name"]) for mapping in mappings
                }
                virtual_name = acquisition_name
                if virtual_name is None or virtual_name in used_names:
                    virtual_name = "digitizer"
                    suffix = 2
                    while virtual_name in used_names:
                        virtual_name = f"digitizer_{suffix}"
                        suffix += 1
                generated_acquisition_name = virtual_name
                mappings.append(
                    {
                        "role": "acquisition",
                        "logical_index": 0,
                        "virtual_name": virtual_name,
                        "label": 0,
                        "absolute_phase": True,
                        "lo_frequency_hz": None,
                        "slot": digitizer_slot,
                        "channel": digitizer_channel,
                    }
                )
        else:
            previous_address = (
                int(mappings[acquisition_index]["slot"]),
                int(mappings[acquisition_index]["channel"]),
            )
            if occupied_index is not None:
                mappings[occupied_index]["slot"] = previous_address[0]
                mappings[occupied_index]["channel"] = previous_address[1]
            mappings[acquisition_index]["slot"] = digitizer_slot
            mappings[acquisition_index]["channel"] = digitizer_channel

        mapped_addresses = {
            (int(mapping["slot"]), int(mapping["channel"]))
            for mapping in mappings
        }
        links = []
        for raw_link in current["downconverter_links"]:
            link = dict(raw_link)
            digitizer_address = (
                int(link["digitizer_slot"]),
                int(link["digitizer_channel"]),
            )
            downconverter_address = (
                int(link["downconverter_slot"]),
                int(link["downconverter_channel"]),
            )
            if digitizer_address == selected_address:
                continue
            if downconverter_address == (
                downconverter_slot,
                downconverter_pair,
            ):
                continue
            if (
                previous_address is not None
                and digitizer_address == previous_address
                and previous_address not in mapped_addresses
            ):
                continue
            if int(link["downconverter_slot"]) == downconverter_slot:
                link["lo_frequency_hz"] = lo_frequency_hz
            links.append(link)
        links.append(
            {
                "digitizer_slot": digitizer_slot,
                "digitizer_channel": digitizer_channel,
                "downconverter_slot": downconverter_slot,
                "downconverter_channel": downconverter_pair,
                "lo_frequency_hz": lo_frequency_hz,
            }
        )
        candidate = dict(current)
        candidate["channel_mappings"] = mappings
        candidate["downconverter_links"] = links
        try:
            normalized = normalize_qcs_hardware_configuration(candidate)
        except (TypeError, ValueError) as exc:
            self._set_status(
                f"Cannot apply the M5201A acquisition route: {exc}",
                error=True,
            )
            return False

        changed = normalized != current
        if generated_acquisition_name is not None:
            self._source_settings["acquisition_channel_name"] = (
                generated_acquisition_name
            )
        self._set_configuration_widgets(normalized)
        self._configuration_state = self._effective_configuration_state(
            normalized,
            self.mapper_path.text().strip(),
        )
        if self._configuration_state not in {
            QCS_HARDWARE_STATE_SAVED,
            QCS_HARDWARE_STATE_IMPORTED,
        }:
            self._mapper_file_sha256 = None
        self._set_status(
            f"Declared M5201A slot {downconverter_slot} pair "
            f"{downconverter_pair} -> M5200A slot {digitizer_slot} CH"
            f"{digitizer_channel} with {lo_frequency_hz / 1.0e9:.9g} GHz "
            "LO. Applying the native mapper automatically...",
            error=False,
        )
        self.connector_selected.emit(
            "acquisition",
            0,
            digitizer_slot,
            digitizer_channel,
            changed,
        )
        return True

    def eventFilter(self, watched, event) -> bool:
        reference_label = getattr(self, "reference_label", None)
        if (
            reference_label is not None
            and watched is reference_label
            and event.type() == QtCore.QEvent.MouseButtonRelease
            and event.button() in (
                QtCore.Qt.LeftButton,
                QtCore.Qt.RightButton,
            )
            and self.tabs.isEnabled()
            and not self._image_pixmap.isNull()
        ):
            displayed = reference_label.pixmap()
            if (
                displayed is not None
                and not displayed.isNull()
                and displayed.width() > 0
                and displayed.height() > 0
            ):
                source_x = (
                    float(event.pos().x())
                    * self._image_pixmap.width()
                    / displayed.width()
                )
                source_y = (
                    float(event.pos().y())
                    * self._image_pixmap.height()
                    / displayed.height()
                )
                if event.button() == QtCore.Qt.RightButton:
                    configuration = (
                        self._preview_configuration_from_widgets()
                    )
                    connector = (
                        None
                        if configuration is None
                        else qcs_chassis_connector_at_point(
                            configuration,
                            source_x,
                            source_y,
                            role="rf",
                        )
                    )
                    if (
                        connector is not None
                        and connector.get("model") == "M5300A"
                    ):
                        self._prompt_m5300_lo_frequency(
                            int(connector["slot"]),
                            int(connector["channel"]),
                        )
                        return True
                    return super().eventFilter(watched, event)
                if self._focused_mapping is not None:
                    configuration = (
                        self._preview_configuration_from_widgets()
                    )
                    if configuration is not None:
                        role, _logical_index = self._focused_mapping
                        if (
                            role == "acquisition"
                            or self._rf_acquisition_path_focus is not None
                        ):
                            downconverter = (
                                qcs_chassis_connector_at_point(
                                    configuration,
                                    source_x,
                                    source_y,
                                    role="downconverter",
                                )
                            )
                            if downconverter is not None:
                                if self._rf_acquisition_path_focus is not None:
                                    self._focused_mapping = (
                                        "acquisition",
                                        dict(
                                            self._rf_acquisition_path_focus
                                        )["acquisition"],
                                    )
                                self.show_m5201_route_dialog(
                                    int(downconverter["slot"]),
                                    int(downconverter["channel"]),
                                )
                                return True
                        connector = qcs_chassis_connector_at_point(
                            configuration,
                            source_x,
                            source_y,
                        )
                        if connector is not None:
                            compatible_connector = (
                                qcs_chassis_connector_at_point(
                                    configuration,
                                    source_x,
                                    source_y,
                                    role=role,
                                )
                            )
                            if compatible_connector is None:
                                self.select_connector(
                                    int(connector["slot"]),
                                    int(connector["channel"]),
                                )
                            else:
                                self.select_connector(
                                    int(
                                        compatible_connector["slot"]
                                    ),
                                    int(
                                        compatible_connector["channel"]
                                    ),
                                )
                            return True
                slot = qcs_chassis_slot_at_point(source_x, source_y)
                if slot is not None:
                    if (
                        self._focused_mapping == ("acquisition", 0)
                        or self._rf_acquisition_path_focus is not None
                    ):
                        module_slot, module_model = (
                            self._module_covering_slot(slot)
                        )
                        if module_model == "M5201A":
                            if self._rf_acquisition_path_focus is not None:
                                self._focused_mapping = (
                                    "acquisition",
                                    dict(
                                        self._rf_acquisition_path_focus
                                    )["acquisition"],
                                )
                            self.show_m5201_route_dialog(module_slot)
                            return True
                    self._show_module_menu_for_slot(
                        slot,
                        event.globalPos(),
                    )
                    return True
        return super().eventFilter(watched, event)

    def _schedule_reference_refresh(self, *_args) -> None:
        if not self._loading:
            self._preview_refresh_timer.start()

    def _preview_configuration_from_widgets(self) -> Optional[dict]:
        candidate = {
            "version": QCS_HARDWARE_CONFIGURATION_VERSION,
            "chassis_model": self.chassis_model.text().strip(),
            "chassis": self.chassis_number.value(),
            "host_controller": self.host_controller.value(),
            "ip_address": self.ip_address.text().strip() or None,
            "modules": self._modules_from_widgets(),
            "channel_mappings": [],
            "downconverter_links": [],
        }
        try:
            candidate["channel_mappings"] = self._mappings_from_widgets()
            candidate["downconverter_links"] = (
                self._downconverter_links_from_widgets()
            )
            return normalize_qcs_hardware_configuration(candidate)
        except (TypeError, ValueError):
            candidate["channel_mappings"] = []
            candidate["downconverter_links"] = []
            try:
                return normalize_qcs_hardware_configuration(candidate)
            except (TypeError, ValueError):
                return None

    def _set_reference_configuration(
        self,
        configuration: Mapping[str, Any],
    ) -> bool:
        try:
            pixmap = _qcs_front_panel_pixmap(
                configuration,
                highlighted_addresses=(
                    self._focused_mapping_addresses(configuration)
                ),
            )
        except (OSError, TypeError, ValueError):
            self._show_reference_preview_unavailable()
            return False
        if pixmap.isNull():
            self._show_reference_preview_unavailable()
            return False
        self._set_reference_pixmap(pixmap)
        return True

    def _refresh_reference_preview(self) -> None:
        configuration = self._preview_configuration_from_widgets()
        if configuration is not None:
            self._set_reference_configuration(configuration)

    def _focused_mapping_address(
        self,
        configuration: Mapping[str, Any],
    ) -> Optional[tuple[int, int]]:
        if self._focused_mapping is None:
            return None
        role, logical_index = self._focused_mapping
        mapping = next(
            (
                candidate
                for candidate in configuration.get(
                    "channel_mappings",
                    (),
                )
                if str(candidate.get("role", "")).strip().lower() == role
                and int(candidate.get("logical_index", -1))
                == logical_index
            ),
            None,
        )
        if mapping is None:
            return None
        return int(mapping["slot"]), int(mapping["channel"])

    def _focused_mapping_addresses(
        self,
        configuration: Mapping[str, Any],
    ) -> tuple[tuple[int, int], ...]:
        """Return every endpoint highlighted by the active selection mode."""

        if self._rf_acquisition_path_focus is None:
            address = self._focused_mapping_address(configuration)
            return () if address is None else (address,)

        addresses = []
        mappings = configuration.get("channel_mappings", ())
        for role, logical_index in self._rf_acquisition_path_focus:
            mapping = next(
                (
                    candidate
                    for candidate in mappings
                    if str(candidate.get("role", "")).strip().lower() == role
                    and int(candidate.get("logical_index", -1))
                    == logical_index
                ),
                None,
            )
            if mapping is None:
                continue
            address = int(mapping["slot"]), int(mapping["channel"])
            if address not in addresses:
                addresses.append(address)
        return tuple(addresses)

    def _focused_mapping_row(self) -> Optional[int]:
        if self._focused_mapping is None:
            return None
        role, logical_index = self._focused_mapping
        for row in range(self.mapping_table.rowCount()):
            role_widget = self.mapping_table.cellWidget(row, 0)
            index_widget = self.mapping_table.cellWidget(row, 1)
            if (
                role_widget is not None
                and index_widget is not None
                and str(role_widget.currentData()) == role
                and int(index_widget.value()) == logical_index
            ):
                return row
        return None

    def _modules_from_widgets(self) -> list[dict]:
        return [
            {"slot": slot, "model": model}
            for slot, model in sorted(self._module_models_by_slot.items())
        ]

    def _set_modules(self, modules: Sequence[Mapping[str, Any]]) -> None:
        self._module_models_by_slot = {
            int(module["slot"]): str(module["model"]) for module in modules
        }
        self._refresh_mapping_instruments()
        self._schedule_reference_refresh()

    def _module_covering_slot(
        self,
        physical_slot: int,
    ) -> tuple[int, Optional[str]]:
        physical_slot = int(physical_slot)
        for start_slot, model in sorted(
            self._module_models_by_slot.items()
        ):
            span = int(QCS_MODULE_MODELS[model]["span"])
            if start_slot <= physical_slot < start_slot + span:
                return start_slot, model
        return physical_slot, None

    def _can_place_module(
        self,
        start_slot: int,
        model: str,
        *,
        replacing_start: Optional[int] = None,
    ) -> bool:
        if model not in QCS_MODULE_MODELS:
            return False
        start_slot = int(start_slot)
        span = int(QCS_MODULE_MODELS[model]["span"])
        if (
            start_slot < 1
            or start_slot + span - 1 > QCS_CHASSIS_SLOT_COUNT
        ):
            return False
        occupied = set()
        for other_start, other_model in self._module_models_by_slot.items():
            if other_start == replacing_start:
                continue
            other_span = int(QCS_MODULE_MODELS[other_model]["span"])
            occupied.update(range(other_start, other_start + other_span))
        return not occupied.intersection(
            range(start_slot, start_slot + span)
        )

    def _invalid_mapping_rows_for_module_change(
        self,
        start_slot: int,
        old_model: Optional[str],
        new_model: Optional[str],
    ) -> list[int]:
        old_span = (
            0
            if old_model is None
            else int(QCS_MODULE_MODELS[old_model]["span"])
        )
        new_span = (
            0
            if new_model is None
            else int(QCS_MODULE_MODELS[new_model]["span"])
        )
        affected_end = start_slot + max(old_span, new_span, 1)
        new_spec = (
            None if new_model is None else QCS_MODULE_MODELS[new_model]
        )
        invalid_rows = []
        for row in range(self.mapping_table.rowCount()):
            mapping_slot = int(
                self.mapping_table.cellWidget(row, 6).value()
            )
            if not start_slot <= mapping_slot < affected_end:
                continue
            valid = False
            if new_spec is not None and mapping_slot == start_slot:
                role = str(
                    self.mapping_table.cellWidget(row, 0).currentData()
                )
                channel = int(
                    self.mapping_table.cellWidget(row, 7).value()
                )
                instrument = new_spec["instrument"]
                valid = (
                    instrument is not None
                    and channel <= int(new_spec["channels"])
                    and instrument in QCS_ROLE_INSTRUMENTS[role]
                )
            if not valid:
                invalid_rows.append(row)
        return invalid_rows

    def _invalid_downconverter_rows_for_module_change(
        self,
        start_slot: int,
        old_model: Optional[str],
        new_model: Optional[str],
    ) -> list[int]:
        old_span = (
            0
            if old_model is None
            else int(QCS_MODULE_MODELS[old_model]["span"])
        )
        new_span = (
            0
            if new_model is None
            else int(QCS_MODULE_MODELS[new_model]["span"])
        )
        affected_end = start_slot + max(old_span, new_span, 1)
        invalid_rows = []
        for row, link in enumerate(self._downconverter_links_from_widgets()):
            endpoints = (
                (int(link["digitizer_slot"]), "M5200A"),
                (int(link["downconverter_slot"]), "M5201A"),
            )
            for endpoint_slot, required_model in endpoints:
                if not start_slot <= endpoint_slot < affected_end:
                    continue
                if endpoint_slot != start_slot or new_model != required_model:
                    invalid_rows.append(row)
                    break
        return invalid_rows

    def _confirm_mapping_removal(
        self,
        *,
        old_model: Optional[str],
        new_model: Optional[str],
        start_slot: int,
        rows: Sequence[int],
        link_rows: Sequence[int] = (),
    ) -> bool:
        if not rows and not link_rows:
            return True
        if old_model is None:
            operation = f"install {new_model}"
        elif new_model is None:
            operation = f"remove {old_model}"
        else:
            operation = f"replace {old_model} with {new_model}"
        names = [
            self.mapping_table.cellWidget(row, 2).text().strip()
            for row in rows
        ]
        link_names = []
        for row in link_rows:
            link = self._downconverter_links_from_widgets()[row]
            link_names.append(
                f"M5200 S{link['digitizer_slot']} "
                f"CH{link['digitizer_channel']} <- M5201 "
                f"S{link['downconverter_slot']} "
                f"CH{link['downconverter_channel']}"
            )
        details = []
        if names:
            details.append(
                f"{len(rows)} channel mapping(s): {', '.join(names)}"
            )
        if link_names:
            details.append(
                f"{len(link_rows)} M5201 link(s): {', '.join(link_names)}"
            )
        result = QtWidgets.QMessageBox.question(
            self,
            "Remove dependent configuration?",
            f"To {operation} in slot {start_slot}, "
            "the following incompatible configuration must also be "
            f"removed:\n" + "\n".join(details) + "\n\nContinue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return result == QtWidgets.QMessageBox.Yes

    def _change_module_at_slot(
        self,
        physical_slot: int,
        new_model: Optional[str],
        *,
        confirm_mapping_removal: bool = True,
    ) -> bool:
        if not self.tabs.isEnabled():
            return False
        physical_slot = int(physical_slot)
        if not 1 <= physical_slot <= QCS_CHASSIS_SLOT_COUNT:
            raise ValueError("QCS physical slot is out of range")
        if new_model is not None:
            new_model = str(new_model)
            if new_model not in QCS_MODULE_MODELS:
                raise ValueError(
                    f"unsupported QCS module model {new_model!r}"
                )
        start_slot, old_model = self._module_covering_slot(physical_slot)
        if old_model is None and new_model is None:
            return False
        if old_model == new_model:
            return True
        if new_model is not None and not self._can_place_module(
            start_slot,
            new_model,
            replacing_start=(
                start_slot if old_model is not None else None
            ),
        ):
            self._set_status(
                f"{new_model} does not fit in slot {start_slot}; its slots "
                "would extend past the chassis or overlap another module.",
                error=True,
            )
            return False
        invalid_rows = self._invalid_mapping_rows_for_module_change(
            start_slot,
            old_model,
            new_model,
        )
        invalid_link_rows = (
            self._invalid_downconverter_rows_for_module_change(
                start_slot,
                old_model,
                new_model,
            )
        )
        if (
            confirm_mapping_removal
            and (invalid_rows or invalid_link_rows)
            and not self._confirm_mapping_removal(
                old_model=old_model,
                new_model=new_model,
                start_slot=start_slot,
                rows=invalid_rows,
                link_rows=invalid_link_rows,
            )
        ):
            return False

        if old_model is not None:
            self._module_models_by_slot.pop(start_slot, None)
        if new_model is not None:
            self._module_models_by_slot[start_slot] = new_model
        for row in reversed(invalid_rows):
            self.mapping_table.removeRow(row)
        for row in reversed(invalid_link_rows):
            self.downconverter_table.removeRow(row)
        self._refresh_mapping_instruments()
        self._schedule_reference_refresh()

        if old_model is None:
            message = f"Installed {new_model} in slot {start_slot}."
        elif new_model is None:
            message = f"Removed {old_model} from slot {start_slot}."
        else:
            message = (
                f"Replaced {old_model} with {new_model} in slot "
                f"{start_slot}."
            )
        if invalid_rows:
            message += (
                f" Removed {len(invalid_rows)} incompatible channel "
                "mapping(s)."
            )
        if invalid_link_rows:
            message += (
                f" Removed {len(invalid_link_rows)} incompatible M5201 "
                "link(s)."
            )
        self._set_status(message, error=False)
        self._stage_graphical_configuration()
        return True

    def _set_module_at_slot(
        self,
        physical_slot: int,
        model: str,
        *,
        confirm_mapping_removal: bool = True,
    ) -> bool:
        return self._change_module_at_slot(
            physical_slot,
            model,
            confirm_mapping_removal=confirm_mapping_removal,
        )

    def _remove_module_at_slot(
        self,
        physical_slot: int,
        *,
        confirm_mapping_removal: bool = True,
    ) -> bool:
        return self._change_module_at_slot(
            physical_slot,
            None,
            confirm_mapping_removal=confirm_mapping_removal,
        )

    def _create_module_menu(self, physical_slot: int) -> QtWidgets.QMenu:
        start_slot, installed_model = self._module_covering_slot(
            physical_slot
        )
        menu = QtWidgets.QMenu(self)
        if installed_model is None:
            menu.addSection(f"Empty slot {start_slot}")
        else:
            menu.addSection(
                f"Slot {start_slot}: {installed_model}"
            )
        for model, spec in QCS_MODULE_MODELS.items():
            action = menu.addAction(str(spec["label"]))
            action.setData(model)
            action.setCheckable(installed_model == model)
            action.setChecked(installed_model == model)
            action.setEnabled(
                self._can_place_module(
                    start_slot,
                    model,
                    replacing_start=(
                        start_slot
                        if installed_model is not None
                        else None
                    ),
                )
            )
            action.setStatusTip(str(spec["purpose"]))
            action.triggered.connect(
                lambda _checked=False, selected=model, slot=start_slot: (
                    self._set_module_at_slot(slot, selected)
                )
            )
        if installed_model is not None:
            menu.addSeparator()
            remove_action = menu.addAction(
                f"Remove {installed_model}"
            )
            remove_action.setData("__remove__")
            remove_action.triggered.connect(
                lambda _checked=False, slot=start_slot: (
                    self._remove_module_at_slot(slot)
                )
            )
        return menu

    def _show_module_menu_for_slot(
        self,
        physical_slot: int,
        global_position: QtCore.QPoint,
    ) -> None:
        if not self.tabs.isEnabled():
            return
        menu = self._create_module_menu(physical_slot)
        menu.exec_(global_position)
        menu.deleteLater()

    def _insert_mapping(self, mapping: Mapping[str, Any]) -> None:
        row = self.mapping_table.rowCount()
        self.mapping_table.insertRow(row)
        role_combo = QtWidgets.QComboBox()
        for role, label in QCS_ROLE_LABELS.items():
            role_combo.addItem(label, role)
        role_index = role_combo.findData(str(mapping.get("role", "dc")))
        role_combo.setCurrentIndex(max(0, role_index))
        logical_index = QtWidgets.QSpinBox()
        logical_index.setRange(0, 9999)
        logical_index.setValue(int(mapping.get("logical_index", 0)))
        virtual_name = QtWidgets.QLineEdit(
            str(mapping.get("virtual_name", "channel"))
        )
        label = QtWidgets.QSpinBox()
        label.setRange(0, 999999)
        label.setValue(int(mapping.get("label", 0)))
        absolute_phase = QtWidgets.QCheckBox()
        absolute_phase.setChecked(
            bool(mapping.get("absolute_phase", role_combo.currentData() != "dc"))
        )
        absolute_phase.setToolTip(
            "Use a phase reference shared across separate QCS program executions"
        )
        lo_frequency = QtWidgets.QLineEdit()
        lo_frequency.setPlaceholderText("required for M5300")
        lo_frequency.setMaximumWidth(125)
        lo_frequency.setValidator(
            QtGui.QDoubleValidator(0.0, 18.0, 9, lo_frequency)
        )
        raw_lo_frequency_hz = mapping.get("lo_frequency_hz")
        if raw_lo_frequency_hz not in (None, ""):
            lo_frequency.setText(
                f"{float(raw_lo_frequency_hz) / 1.0e9:.12g}"
            )
        lo_frequency.setToolTip(
            "M5300 local-oscillator frequency in GHz (required before saving)"
        )
        slot = QtWidgets.QSpinBox()
        slot.setRange(1, QCS_CHASSIS_SLOT_COUNT)
        slot.setValue(int(mapping.get("slot", 2)))
        channel = QtWidgets.QSpinBox()
        channel.setRange(1, 64)
        channel.setValue(int(mapping.get("channel", 1)))
        instrument = QtWidgets.QTableWidgetItem()
        instrument.setFlags(instrument.flags() & ~QtCore.Qt.ItemIsEditable)
        self.mapping_table.setCellWidget(row, 0, role_combo)
        self.mapping_table.setCellWidget(row, 1, logical_index)
        self.mapping_table.setCellWidget(row, 2, virtual_name)
        self.mapping_table.setCellWidget(row, 3, label)
        self.mapping_table.setCellWidget(row, 4, absolute_phase)
        self.mapping_table.setCellWidget(row, 5, lo_frequency)
        self.mapping_table.setCellWidget(row, 6, slot)
        self.mapping_table.setCellWidget(row, 7, channel)
        self.mapping_table.setItem(row, 8, instrument)
        role_combo.currentIndexChanged.connect(
            lambda _index: self._refresh_mapping_role_states()
        )
        slot.valueChanged.connect(
            lambda _value: self._refresh_mapping_instruments()
        )
        slot.valueChanged.connect(self._refresh_downconverter_link_statuses)
        channel.valueChanged.connect(
            self._refresh_downconverter_link_statuses
        )
        role_combo.currentIndexChanged.connect(
            self._schedule_reference_refresh
        )
        logical_index.valueChanged.connect(
            self._schedule_reference_refresh
        )
        virtual_name.textChanged.connect(
            self._schedule_reference_refresh
        )
        label.valueChanged.connect(self._schedule_reference_refresh)
        absolute_phase.toggled.connect(
            self._schedule_reference_refresh
        )
        lo_frequency.textChanged.connect(
            self._schedule_reference_refresh
        )
        slot.valueChanged.connect(self._schedule_reference_refresh)
        channel.valueChanged.connect(self._schedule_reference_refresh)
        self._on_mapping_role_changed(row)
        self._update_mapping_instrument(row)
        self._schedule_reference_refresh()

    def _on_mapping_role_changed(self, row: int) -> None:
        if row >= self.mapping_table.rowCount():
            return
        role_combo = self.mapping_table.cellWidget(row, 0)
        logical_index = self.mapping_table.cellWidget(row, 1)
        if role_combo is None or logical_index is None:
            return
        is_acquisition = role_combo.currentData() == "acquisition"
        if is_acquisition:
            logical_index.setValue(0)
        logical_index.setEnabled(not is_acquisition)

    def _refresh_mapping_role_states(self) -> None:
        for row in range(self.mapping_table.rowCount()):
            self._on_mapping_role_changed(row)

    def _update_mapping_instrument(self, row: int) -> None:
        if row >= self.mapping_table.rowCount():
            return
        slot_widget = self.mapping_table.cellWidget(row, 6)
        lo_frequency = self.mapping_table.cellWidget(row, 5)
        item = self.mapping_table.item(row, 8)
        if slot_widget is None or item is None:
            return
        model = self._module_models_by_slot.get(slot_widget.value())
        if model is None:
            text = "No module"
        else:
            instrument = QCS_MODULE_MODELS[str(model)]["instrument"]
            text = str(instrument or model)
        is_rf_awg = text == "M5300AWG"
        if lo_frequency is not None:
            lo_frequency.setEnabled(is_rf_awg)
            if not is_rf_awg:
                lo_frequency.clear()
        item.setText(text)

    def _refresh_mapping_instruments(self) -> None:
        for row in range(self.mapping_table.rowCount()):
            self._update_mapping_instrument(row)
        if hasattr(self, "downconverter_table"):
            self._refresh_downconverter_link_statuses()

    def _mappings_from_widgets(self) -> list[dict]:
        mappings = []
        for row in range(self.mapping_table.rowCount()):
            lo_frequency_text = (
                self.mapping_table.cellWidget(row, 5).text().strip()
            )
            mappings.append(
                {
                    "role": str(
                        self.mapping_table.cellWidget(row, 0).currentData()
                    ),
                    "logical_index": (
                        self.mapping_table.cellWidget(row, 1).value()
                    ),
                    "virtual_name": (
                        self.mapping_table.cellWidget(row, 2).text().strip()
                    ),
                    "label": self.mapping_table.cellWidget(row, 3).value(),
                    "absolute_phase": (
                        self.mapping_table.cellWidget(row, 4).isChecked()
                    ),
                    "lo_frequency_hz": (
                        None
                        if not lo_frequency_text
                        else float(lo_frequency_text) * 1.0e9
                    ),
                    "slot": self.mapping_table.cellWidget(row, 6).value(),
                    "channel": self.mapping_table.cellWidget(row, 7).value(),
                }
            )
        return mappings

    def _set_mappings(self, mappings: Sequence[Mapping[str, Any]]) -> None:
        self.mapping_table.setRowCount(0)
        for mapping in mappings:
            self._insert_mapping(mapping)

    def _insert_downconverter_link(self, link: Mapping[str, Any]) -> None:
        row = self.downconverter_table.rowCount()
        self.downconverter_table.insertRow(row)
        digitizer_slot = QtWidgets.QSpinBox()
        digitizer_slot.setRange(1, QCS_CHASSIS_SLOT_COUNT)
        digitizer_slot.setValue(int(link.get("digitizer_slot", 1)))
        digitizer_channel = QtWidgets.QSpinBox()
        digitizer_channel.setRange(1, 4)
        digitizer_channel.setValue(int(link.get("digitizer_channel", 1)))
        downconverter_slot = QtWidgets.QSpinBox()
        downconverter_slot.setRange(1, QCS_CHASSIS_SLOT_COUNT)
        downconverter_slot.setValue(int(link.get("downconverter_slot", 1)))
        downconverter_channel = QtWidgets.QSpinBox()
        downconverter_channel.setRange(1, 4)
        downconverter_channel.setValue(
            int(link.get("downconverter_channel", 1))
        )
        lo_frequency = QtWidgets.QLineEdit()
        lo_frequency.setPlaceholderText("required: 1-18")
        lo_frequency.setMaximumWidth(125)
        lo_frequency.setValidator(
            QtGui.QDoubleValidator(1.0, 18.0, 9, lo_frequency)
        )
        raw_lo_frequency_hz = link.get("lo_frequency_hz")
        if raw_lo_frequency_hz not in (None, ""):
            lo_frequency.setText(
                f"{float(raw_lo_frequency_hz) / 1.0e9:.12g}"
            )
        lo_frequency.setToolTip(
            "M5201A shared local-oscillator frequency in GHz. All links on "
            "one M5201A module must use the same value."
        )
        resolved = QtWidgets.QTableWidgetItem()
        resolved.setFlags(resolved.flags() & ~QtCore.Qt.ItemIsEditable)
        for column, widget in enumerate(
            (
                digitizer_slot,
                digitizer_channel,
                downconverter_slot,
                downconverter_channel,
                lo_frequency,
            )
        ):
            self.downconverter_table.setCellWidget(row, column, widget)
        self.downconverter_table.setItem(row, 5, resolved)
        for widget in (
            digitizer_slot,
            digitizer_channel,
            downconverter_slot,
            downconverter_channel,
        ):
            widget.valueChanged.connect(
                self._refresh_downconverter_link_statuses
            )
            widget.valueChanged.connect(self._schedule_reference_refresh)
        lo_frequency.textChanged.connect(
            self._refresh_downconverter_link_statuses
        )
        lo_frequency.textChanged.connect(
            lambda text, source=lo_frequency: (
                self._propagate_downconverter_lo(source, text)
            )
        )
        lo_frequency.textChanged.connect(self._schedule_reference_refresh)
        downconverter_slot.valueChanged.connect(
            lambda _value, source=downconverter_slot: (
                self._adopt_downconverter_lo(source)
            )
        )
        self._update_downconverter_link_status(row)
        self._schedule_reference_refresh()

    def _downconverter_row_for_widget(self, widget) -> Optional[int]:
        for row in range(self.downconverter_table.rowCount()):
            for column in range(5):
                if self.downconverter_table.cellWidget(row, column) is widget:
                    return row
        return None

    def _set_downconverter_lo_for_slot(
        self,
        downconverter_slot: int,
        text: str,
    ) -> None:
        if self._syncing_downconverter_lo:
            return
        self._syncing_downconverter_lo = True
        try:
            for row in range(self.downconverter_table.rowCount()):
                slot_widget = self.downconverter_table.cellWidget(row, 2)
                lo_widget = self.downconverter_table.cellWidget(row, 4)
                if (
                    slot_widget is not None
                    and lo_widget is not None
                    and int(slot_widget.value()) == int(downconverter_slot)
                    and lo_widget.text() != text
                ):
                    lo_widget.setText(text)
        finally:
            self._syncing_downconverter_lo = False
        self._refresh_downconverter_link_statuses()
        self._schedule_reference_refresh()

    def _propagate_downconverter_lo(self, source_widget, text: str) -> None:
        """Apply an edited LO to every channel pair on the same M5201A."""

        if self._syncing_downconverter_lo:
            return
        row = self._downconverter_row_for_widget(source_widget)
        if row is None:
            return
        slot = self.downconverter_table.cellWidget(row, 2).value()
        self._set_downconverter_lo_for_slot(slot, str(text))

    def _adopt_downconverter_lo(self, source_widget) -> None:
        """Adopt the shared LO when a row moves to another M5201A slot."""

        if self._syncing_downconverter_lo:
            return
        row = self._downconverter_row_for_widget(source_widget)
        if row is None:
            return
        slot = int(self.downconverter_table.cellWidget(row, 2).value())
        source_lo = self.downconverter_table.cellWidget(row, 4)
        target_text = source_lo.text()
        for candidate_row in range(self.downconverter_table.rowCount()):
            if candidate_row == row:
                continue
            candidate_slot = self.downconverter_table.cellWidget(
                candidate_row,
                2,
            )
            if int(candidate_slot.value()) != slot:
                continue
            target_text = self.downconverter_table.cellWidget(
                candidate_row,
                4,
            ).text()
            break
        self._set_downconverter_lo_for_slot(slot, target_text)

    def _update_downconverter_link_status(self, row: int) -> None:
        if row >= self.downconverter_table.rowCount():
            return
        digitizer_slot = int(
            self.downconverter_table.cellWidget(row, 0).value()
        )
        digitizer_channel = int(
            self.downconverter_table.cellWidget(row, 1).value()
        )
        downconverter_slot = int(
            self.downconverter_table.cellWidget(row, 2).value()
        )
        downconverter_channel = int(
            self.downconverter_table.cellWidget(row, 3).value()
        )
        digitizer_model = self._module_models_by_slot.get(digitizer_slot)
        downconverter_model = self._module_models_by_slot.get(
            downconverter_slot
        )
        mapped_addresses = {
            (int(mapping["slot"]), int(mapping["channel"]))
            for mapping in self._mappings_from_widgets()
        }
        if digitizer_model != "M5200A":
            text = f"Slot {digitizer_slot}: install M5200A"
        elif downconverter_model != "M5201A":
            text = f"Slot {downconverter_slot}: install M5201A"
        elif (digitizer_slot, digitizer_channel) not in mapped_addresses:
            text = "M5200 channel needs a virtual mapping"
        else:
            text = (
                f"M5200A S{digitizer_slot} CH{digitizer_channel} <- "
                f"M5201A S{downconverter_slot} "
                f"pair {downconverter_channel}"
            )
        item = self.downconverter_table.item(row, 5)
        if item is not None:
            item.setText(text)

    def _refresh_downconverter_link_statuses(self, *_args) -> None:
        for row in range(self.downconverter_table.rowCount()):
            self._update_downconverter_link_status(row)

    def _downconverter_links_from_widgets(self) -> list[dict]:
        links = []
        for row in range(self.downconverter_table.rowCount()):
            lo_text = (
                self.downconverter_table.cellWidget(row, 4).text().strip()
            )
            links.append(
                {
                    "digitizer_slot": self.downconverter_table.cellWidget(
                        row,
                        0,
                    ).value(),
                    "digitizer_channel": self.downconverter_table.cellWidget(
                        row,
                        1,
                    ).value(),
                    "downconverter_slot": (
                        self.downconverter_table.cellWidget(row, 2).value()
                    ),
                    "downconverter_channel": (
                        self.downconverter_table.cellWidget(row, 3).value()
                    ),
                    "lo_frequency_hz": (
                        None if not lo_text else float(lo_text) * 1.0e9
                    ),
                }
            )
        return links

    def _set_downconverter_links(
        self,
        links: Sequence[Mapping[str, Any]],
    ) -> None:
        self.downconverter_table.setRowCount(0)
        for link in links:
            self._insert_downconverter_link(link)

    def _add_downconverter_link(self) -> None:
        mappings = self._mappings_from_widgets()
        digitizer_addresses = [
            (int(mapping["slot"]), int(mapping["channel"]))
            for mapping in mappings
            if self._module_models_by_slot.get(int(mapping["slot"]))
            == "M5200A"
        ]
        downconverter_slots = sorted(
            slot
            for slot, model in self._module_models_by_slot.items()
            if model == "M5201A"
        )
        if not digitizer_addresses:
            self._set_status(
                "Create an M5200A virtual-channel mapping before adding an "
                "M5201A link.",
                error=True,
            )
            return
        if not downconverter_slots:
            self._set_status(
                "Install an M5201A module from the Front panel before "
                "adding a downconverter link.",
                error=True,
            )
            return
        existing = self._downconverter_links_from_widgets()
        used_digitizers = {
            (int(link["digitizer_slot"]), int(link["digitizer_channel"]))
            for link in existing
        }
        used_downconverters = {
            (
                int(link["downconverter_slot"]),
                int(link["downconverter_channel"]),
            )
            for link in existing
        }
        digitizer_address = next(
            (
                address
                for address in digitizer_addresses
                if address not in used_digitizers
            ),
            None,
        )
        if digitizer_address is None:
            self._set_status(
                "Every mapped M5200A channel already has an M5201A link.",
                error=True,
            )
            return
        downconverter_address = None
        preferred_channel = digitizer_address[1]
        for slot in downconverter_slots:
            for channel in (
                preferred_channel,
                *(candidate for candidate in range(1, 5) if candidate != preferred_channel),
            ):
                if (slot, channel) not in used_downconverters:
                    downconverter_address = (slot, channel)
                    break
            if downconverter_address is not None:
                break
        if downconverter_address is None:
            self._set_status(
                "Every installed M5201A RF/IF pair is already linked.",
                error=True,
            )
            return
        existing_module_lo = next(
            (
                link["lo_frequency_hz"]
                for link in existing
                if int(link["downconverter_slot"])
                == downconverter_address[0]
            ),
            None,
        )
        self._insert_downconverter_link(
            {
                "digitizer_slot": digitizer_address[0],
                "digitizer_channel": digitizer_address[1],
                "downconverter_slot": downconverter_address[0],
                "downconverter_channel": downconverter_address[1],
                "lo_frequency_hz": existing_module_lo,
            }
        )
        row = self.downconverter_table.rowCount() - 1
        self.downconverter_table.selectRow(row)
        self.downconverter_table.setCurrentCell(row, 4)
        self._set_status(
            "Added an M5200A-to-M5201A link. Enter the shared M5201A LO "
            "before saving the mapper.",
            error=False,
        )

    def _remove_selected_downconverter_links(self) -> None:
        rows = sorted(
            {
                index.row()
                for index in self.downconverter_table.selectionModel().selectedRows()
            },
            reverse=True,
        )
        if not rows and self.downconverter_table.rowCount():
            rows = [self.downconverter_table.rowCount() - 1]
        for row in rows:
            self.downconverter_table.removeRow(row)
        self._schedule_reference_refresh()

    def select_connector(self, slot: int, channel: int) -> bool:
        """Bind the focused logical mapping to one physical channel SMA.

        Selecting an SMA already used by another mapping of the same role
        swaps the two addresses, matching the QICK AWG-output selector.  A
        connector owned by a different role is never silently reassigned.
        A missing focused DC, RF, or acquisition mapping is created directly
        from the SMA click. A newly selected M5300 output still requires its
        LO frequency before the native mapper can be saved.
        """

        if not self.tabs.isEnabled() or self._focused_mapping is None:
            return False
        slot = _positive_integer(slot, "QCS selected connector slot")
        channel = _positive_integer(
            channel,
            "QCS selected connector channel",
        )
        model = self._module_models_by_slot.get(slot)
        spec = None if model is None else QCS_MODULE_MODELS[model]
        if self._rf_acquisition_path_focus is not None:
            path_mappings = dict(self._rf_acquisition_path_focus)
            instrument = None if spec is None else spec["instrument"]
            if instrument in QCS_ROLE_INSTRUMENTS["acquisition"]:
                self._focused_mapping = (
                    "acquisition",
                    path_mappings["acquisition"],
                )
            elif instrument in QCS_ROLE_INSTRUMENTS["rf"]:
                self._focused_mapping = ("rf", path_mappings["rf"])
            else:
                self._set_status(
                    "RF-path selection requires an RF output SMA "
                    "on M5300A/M5301A or an acquisition input SMA on "
                    "M5200A.",
                    error=True,
                )
                return False
        role, logical_index = self._focused_mapping
        compatible = (
            spec is not None
            and spec["instrument"] is not None
            and channel <= int(spec["channels"])
            and spec["instrument"] in QCS_ROLE_INSTRUMENTS[role]
        )
        if not compatible:
            compatible_models = [
                candidate_model
                for candidate_model, candidate_spec
                in QCS_MODULE_MODELS.items()
                if candidate_spec["instrument"]
                in QCS_ROLE_INSTRUMENTS[role]
            ]
            self._set_status(
                f"{QCS_ROLE_LABELS[role]} selection requires a channel SMA "
                f"on {' or '.join(compatible_models)}.",
                error=True,
            )
            return False

        clean_imported_configuration = None
        if not self._mapper_write_allowed:
            try:
                candidate = self.working_configuration()
                if self._effective_configuration_state(
                    candidate,
                    self.mapper_path.text().strip(),
                ) == QCS_HARDWARE_STATE_IMPORTED:
                    clean_imported_configuration = candidate
            except (TypeError, ValueError):
                pass

        selected_address = (slot, channel)
        target_row = self._focused_mapping_row()
        creating_mapping = target_row is None
        if creating_mapping and role not in {"dc", "rf", "acquisition"}:
            self._set_status(
                f"No mapping exists for "
                f"{_qcs_mapping_display_name(role, logical_index)}.",
                error=True,
            )
            return False

        target_slot_widget = None
        target_channel_widget = None
        previous_address = None
        requested_m5300_lo_frequency_hz = None
        if target_row is not None:
            target_slot_widget = self.mapping_table.cellWidget(target_row, 6)
            target_channel_widget = self.mapping_table.cellWidget(target_row, 7)
            previous_address = (
                int(target_slot_widget.value()),
                int(target_channel_widget.value()),
            )
        if role == "rf" and model == "M5300A":
            existing_lo_frequency_hz = (
                None
                if target_row is None
                else self._mappings_from_widgets()[target_row][
                    "lo_frequency_hz"
                ]
            )
            if existing_lo_frequency_hz is None:
                lo_frequency_ghz, accepted = QtWidgets.QInputDialog.getDouble(
                    self,
                    "M5300 RF Output LO",
                    "M5300 local-oscillator frequency [GHz]:",
                    5.0,
                    0.0,
                    18.0,
                    9,
                )
                if not accepted:
                    self._set_status(
                        f"M5300 LO entry was cancelled; "
                        f"{_qcs_mapping_display_name(role, logical_index)} "
                        "was not changed.",
                        error=False,
                    )
                    return False
                requested_m5300_lo_frequency_hz = (
                    float(lo_frequency_ghz) * 1.0e9
                )
        if (
            previous_address == selected_address
            and requested_m5300_lo_frequency_hz is None
        ):
            self._set_status(
                f"{_qcs_mapping_display_name(role, logical_index)} already "
                "uses "
                f"{model} slot {slot} CH{channel}.",
                error=False,
            )
            self._refresh_reference_preview()
            self.connector_selected.emit(
                role,
                logical_index,
                slot,
                channel,
                False,
            )
            return True

        occupied_row = None
        for row in range(self.mapping_table.rowCount()):
            if row == target_row:
                continue
            row_address = (
                int(self.mapping_table.cellWidget(row, 6).value()),
                int(self.mapping_table.cellWidget(row, 7).value()),
            )
            if row_address == selected_address:
                occupied_row = row
                break
        if (
            clean_imported_configuration is not None
            and role == "acquisition"
            and previous_address != selected_address
        ):
            if occupied_row is None:
                self._set_status(
                    f"The imported mapper has no virtual channel at {model} "
                    f"slot {slot} CH{channel}. Choose an SMA already assigned "
                    "in that mapper, or use Advanced Hardware Settings to "
                    "start an app-owned mapper.",
                    error=True,
                )
                return False
            occupied_role = str(
                self.mapping_table.cellWidget(
                    occupied_row,
                    0,
                ).currentData()
            )
            if occupied_role == "unassigned":
                proposed_mappings = self._mappings_from_widgets()
                freed_unassigned_index = int(
                    proposed_mappings[occupied_row]["logical_index"]
                )
                selected_virtual_name = str(
                    proposed_mappings[occupied_row]["virtual_name"]
                )
                proposed_mappings[occupied_row]["role"] = "acquisition"
                proposed_mappings[occupied_row]["logical_index"] = 0
                if target_row is not None:
                    proposed_mappings[target_row]["role"] = "unassigned"
                    proposed_mappings[target_row]["logical_index"] = (
                        freed_unassigned_index
                    )
                proposed = dict(clean_imported_configuration)
                proposed["channel_mappings"] = proposed_mappings
                try:
                    normalized_proposed = (
                        normalize_qcs_hardware_configuration(proposed)
                    )
                except (TypeError, ValueError) as exc:
                    self._set_status(
                        f"The imported acquisition role cannot be changed: "
                        f"{exc}",
                        error=True,
                    )
                    return False
                if qcs_hardware_mapper_fingerprint(
                    normalized_proposed
                ) != qcs_hardware_mapper_fingerprint(
                    clean_imported_configuration
                ):
                    self._set_status(
                        "The requested imported-mapper change would alter "
                        "native channel data, so it was not applied.",
                        error=True,
                    )
                    return False

                widgets = [
                    self.mapping_table.cellWidget(occupied_row, 0),
                    self.mapping_table.cellWidget(occupied_row, 1),
                ]
                if target_row is not None:
                    widgets.extend(
                        (
                            self.mapping_table.cellWidget(target_row, 0),
                            self.mapping_table.cellWidget(target_row, 1),
                        )
                    )
                blockers = [
                    QtCore.QSignalBlocker(widget) for widget in widgets
                ]
                previous_loading = self._loading
                self._loading = True
                try:
                    occupied_role_widget = self.mapping_table.cellWidget(
                        occupied_row,
                        0,
                    )
                    occupied_role_widget.setCurrentIndex(
                        occupied_role_widget.findData("acquisition")
                    )
                    self.mapping_table.cellWidget(
                        occupied_row,
                        1,
                    ).setValue(0)
                    if target_row is not None:
                        target_role_widget = self.mapping_table.cellWidget(
                            target_row,
                            0,
                        )
                        target_role_widget.setCurrentIndex(
                            target_role_widget.findData("unassigned")
                        )
                        self.mapping_table.cellWidget(
                            target_row,
                            1,
                        ).setValue(freed_unassigned_index)
                finally:
                    self._loading = previous_loading
                    del blockers
                self._source_settings["acquisition_channel_name"] = (
                    selected_virtual_name
                )
                self._refresh_mapping_role_states()
                self._refresh_mapping_instruments()
                self.mapping_table.selectRow(occupied_row)
                self._preview_refresh_timer.stop()
                self._refresh_reference_preview()
                self._set_status(
                    f"Selected imported virtual channel "
                    f"{selected_virtual_name!r} at {model} slot {slot} "
                    "CH"
                    f"{channel} for acquisition. The native mapper remains "
                    "unchanged.",
                    error=False,
                )
                self.connector_selected.emit(
                    role,
                    logical_index,
                    slot,
                    channel,
                    True,
                )
                return True
        if occupied_row is not None:
            occupied_name = (
                self.mapping_table.cellWidget(
                    occupied_row,
                    2,
                ).text().strip()
            )
            if creating_mapping:
                self._set_status(
                    f"{model} slot {slot} CH{channel} is already mapped to "
                    f"{occupied_name}. Choose an unused SMA for "
                    f"{_qcs_mapping_display_name(role, logical_index)}.",
                    error=True,
                )
                return False
            occupied_role = str(
                self.mapping_table.cellWidget(
                    occupied_row,
                    0,
                ).currentData()
            )
            if occupied_role != role:
                self._set_status(
                    f"{model} slot {slot} CH{channel} is already mapped to "
                    f"{occupied_name}; mappings with different roles are not "
                    "automatically rerouted.",
                    error=True,
                )
                return False

        proposed_mappings = self._mappings_from_widgets()
        proposed_links = self._downconverter_links_from_widgets()
        created_mapping = None
        generated_rf_name = None
        generated_acquisition_name = None
        if creating_mapping:
            dc_names, rf_names, acquisition_name = (
                self._current_source_bindings()
            )
            if role == "dc":
                if logical_index >= len(dc_names):
                    self._set_status(
                        f"DC output {logical_index} is no longer present in "
                        "the current experiment.",
                        error=True,
                    )
                    return False
                virtual_name = dc_names[logical_index]
            elif role == "rf":
                virtual_name = rf_names.get(logical_index)
                if virtual_name is None:
                    used_names = {
                        str(mapping["virtual_name"])
                        for mapping in proposed_mappings
                    }
                    used_names.update(dc_names)
                    used_names.update(rf_names.values())
                    if acquisition_name is not None:
                        used_names.add(acquisition_name)
                    virtual_name = (
                        "rf_drive"
                        if logical_index == 0
                        else f"rf_drive_{logical_index}"
                    )
                    suffix = 2
                    base_name = virtual_name
                    while virtual_name in used_names:
                        virtual_name = f"{base_name}_{suffix}"
                        suffix += 1
                    generated_rf_name = virtual_name
            else:
                if logical_index != 0:
                    self._set_status(
                        "PulseGenerator supports acquisition input 0 only.",
                        error=True,
                    )
                    return False
                virtual_name = acquisition_name
                if virtual_name is None:
                    used_names = {
                        str(mapping["virtual_name"])
                        for mapping in proposed_mappings
                    }
                    virtual_name = "digitizer"
                    suffix = 2
                    while virtual_name in used_names:
                        virtual_name = f"digitizer_{suffix}"
                        suffix += 1
                    generated_acquisition_name = virtual_name
            created_mapping = {
                "role": role,
                "logical_index": logical_index,
                "virtual_name": virtual_name,
                "label": 0,
                "absolute_phase": role != "dc",
                "lo_frequency_hz": requested_m5300_lo_frequency_hz,
                "slot": slot,
                "channel": channel,
            }
            proposed_mappings.append(created_mapping)
        else:
            proposed_mappings[target_row]["slot"] = slot
            proposed_mappings[target_row]["channel"] = channel
            if role == "rf":
                if model == "M5300A":
                    if requested_m5300_lo_frequency_hz is not None:
                        proposed_mappings[target_row]["lo_frequency_hz"] = (
                            requested_m5300_lo_frequency_hz
                        )
                else:
                    # M5301 mappings do not expose a mapper-level LO setting.
                    proposed_mappings[target_row]["lo_frequency_hz"] = None
        linked_rows = []
        if role == "acquisition" and previous_address is not None:
            for link_row, link in enumerate(proposed_links):
                if (
                    int(link["digitizer_slot"]),
                    int(link["digitizer_channel"]),
                ) == previous_address:
                    linked_rows.append(link_row)
        if linked_rows and previous_address != selected_address:
            self._set_status(
                "The current acquisition input has an explicit M5201A "
                "down-converter link. Its physical cable cannot be rerouted "
                "automatically. Select the linked M5200A SMA, or update the "
                "explicit link in Advanced Hardware Settings.",
                error=True,
            )
            return False
        if occupied_row is not None:
            proposed_mappings[occupied_row]["slot"] = previous_address[0]
            proposed_mappings[occupied_row]["channel"] = previous_address[1]
        try:
            normalize_qcs_hardware_configuration(
                {
                    "version": QCS_HARDWARE_CONFIGURATION_VERSION,
                    "chassis_model": self.chassis_model.text().strip(),
                    "chassis": self.chassis_number.value(),
                    "host_controller": self.host_controller.value(),
                    "ip_address": self.ip_address.text().strip() or None,
                    "modules": self._modules_from_widgets(),
                    "channel_mappings": proposed_mappings,
                    "downconverter_links": proposed_links,
                }
            )
        except (TypeError, ValueError) as exc:
            self._set_status(
                f"Cannot select {model} slot {slot} CH{channel}: {exc}",
                error=True,
            )
            return False

        if creating_mapping:
            if generated_rf_name is not None:
                source_rf_names = dict(
                    self._source_settings.get("rf_channel_names", {})
                )
                source_rf_names[int(logical_index)] = generated_rf_name
                self._source_settings["rf_channel_names"] = source_rf_names
            if generated_acquisition_name is not None:
                self._source_settings["acquisition_channel_name"] = (
                    generated_acquisition_name
                )
            self._insert_mapping(created_mapping)
            target_row = self._focused_mapping_row()
            if target_row is None:
                self._set_status(
                    "The selected channel mapping could not be created.",
                    error=True,
                )
                return False
            target_slot_widget = self.mapping_table.cellWidget(target_row, 6)
            target_channel_widget = self.mapping_table.cellWidget(target_row, 7)

        widgets = [target_slot_widget, target_channel_widget]
        target_lo_widget = None
        target_lo_frequency_hz = None
        if role == "rf":
            target_lo_widget = self.mapping_table.cellWidget(target_row, 5)
            target_lo_frequency_hz = proposed_mappings[target_row][
                "lo_frequency_hz"
            ]
            widgets.append(target_lo_widget)
        if occupied_row is not None:
            widgets.extend(
                (
                    self.mapping_table.cellWidget(occupied_row, 6),
                    self.mapping_table.cellWidget(occupied_row, 7),
                )
            )
        blockers = [QtCore.QSignalBlocker(widget) for widget in widgets]
        previous_loading = self._loading
        self._loading = True
        try:
            target_slot_widget.setValue(slot)
            target_channel_widget.setValue(channel)
            if target_lo_widget is not None:
                target_lo_widget.setText(
                    ""
                    if target_lo_frequency_hz is None
                    else f"{float(target_lo_frequency_hz) / 1.0e9:.12g}"
                )
            if occupied_row is not None:
                self.mapping_table.cellWidget(
                    occupied_row,
                    6,
                ).setValue(previous_address[0])
                self.mapping_table.cellWidget(
                    occupied_row,
                    7,
                ).setValue(previous_address[1])
        finally:
            self._loading = previous_loading
            del blockers
        self._refresh_mapping_instruments()
        self.mapping_table.selectRow(target_row)
        self._preview_refresh_timer.stop()
        self._refresh_reference_preview()
        swap_message = (
            " Swapped the previous connector onto the other "
            f"{QCS_ROLE_LABELS[role].lower()} mapping."
            if occupied_row is not None
            else ""
        )
        self._set_status(
            f"{'Created and selected' if creating_mapping else 'Selected'} "
            f"{model} slot {slot} CH{channel} for "
            f"{_qcs_mapping_display_name(role, logical_index)}."
            f"{swap_message} The native mapper will be updated "
            "automatically.",
            error=False,
        )
        self.connector_selected.emit(
            role,
            logical_index,
            slot,
            channel,
            True,
        )
        return True

    def _add_empty_mapping(self) -> None:
        role_counts = {role: 0 for role in QCS_ROLE_LABELS}
        for mapping in self._mappings_from_widgets():
            role_counts[mapping["role"]] += 1
        role = "dc"
        logical_index = role_counts[role]
        self._insert_mapping(
            {
                "role": role,
                "logical_index": logical_index,
                "virtual_name": f"dc_ch_{logical_index + 1}",
                "label": 0,
                "absolute_phase": False,
                "lo_frequency_hz": None,
                "slot": 2,
                "channel": logical_index % 4 + 1,
            }
        )

    def _remove_selected_mappings(self) -> None:
        rows = sorted(
            {
                index.row()
                for index in self.mapping_table.selectionModel().selectedRows()
            },
            reverse=True,
        )
        if not rows and self.mapping_table.rowCount():
            rows = [self.mapping_table.rowCount() - 1]
        removed_addresses = {
            (
                int(self.mapping_table.cellWidget(row, 6).value()),
                int(self.mapping_table.cellWidget(row, 7).value()),
            )
            for row in rows
        }
        linked_rows = [
            row
            for row, link in enumerate(
                self._downconverter_links_from_widgets()
            )
            if (
                int(link["digitizer_slot"]),
                int(link["digitizer_channel"]),
            )
            in removed_addresses
        ]
        for row in reversed(linked_rows):
            self.downconverter_table.removeRow(row)
        for row in rows:
            self.mapping_table.removeRow(row)
        self._refresh_mapping_instruments()
        self._refresh_downconverter_link_statuses()
        self._schedule_reference_refresh()

    def _current_source_bindings(
        self,
    ) -> tuple[list[str], dict[int, str], Optional[str]]:
        dc_names = [
            str(name)
            for name in self._source_settings.get("dc_channel_names", ())
        ]
        dc_names = dc_names[: self._output_count]
        used_names = set(dc_names)
        next_index = 1
        while len(dc_names) < self._output_count:
            candidate = f"dc_ch_{next_index}"
            next_index += 1
            if candidate in used_names:
                continue
            dc_names.append(candidate)
            used_names.add(candidate)
        rf_names = {
            int(index): str(name)
            for index, name in dict(
                self._source_settings.get("rf_channel_names", {})
            ).items()
        }
        acquisition_name = self._source_settings.get(
            "acquisition_channel_name"
        )
        return dc_names, rf_names, (
            None
            if acquisition_name is None
            else str(acquisition_name).strip() or None
        )

    @staticmethod
    def _normalized_mapper_path(path: str) -> str:
        return str(Path(path).expanduser().resolve()).casefold()

    def _effective_configuration_state(
        self,
        configuration: Mapping[str, Any],
        mapper_path: str,
    ) -> str:
        state = self._configuration_state
        if state == QCS_HARDWARE_STATE_EXTERNAL:
            return QCS_HARDWARE_STATE_DRAFT
        if state == QCS_HARDWARE_STATE_DRAFT:
            return state
        if state == QCS_HARDWARE_STATE_IMPORTED_DIRTY:
            return state

        fingerprint_matches = (
            self._authoritative_fingerprint is not None
            and qcs_hardware_mapper_fingerprint(configuration)
            == self._authoritative_fingerprint
        )
        path_matches = (
            self._authoritative_mapper_path is not None
            and self._normalized_mapper_path(mapper_path)
            == self._authoritative_mapper_path
        )
        if fingerprint_matches and path_matches:
            return state
        if state == QCS_HARDWARE_STATE_IMPORTED:
            return QCS_HARDWARE_STATE_IMPORTED_DIRTY
        return QCS_HARDWARE_STATE_DRAFT

    def _restore_diagram_layout(self) -> None:
        dc_names, rf_names, acquisition_name = self._current_source_bindings()
        configuration = default_qcs_hardware_configuration(
            dc_names,
            rf_names,
            acquisition_name,
        )
        configuration["ip_address"] = (
            self.ip_address.text().strip() or DEFAULT_QCS_IP_ADDRESS
        )
        self._set_configuration_widgets(configuration)
        self._configuration_state = QCS_HARDWARE_STATE_DRAFT
        self._authoritative_fingerprint = None
        self._authoritative_mapper_path = None
        current_mapper_path = self.mapper_path.text().strip()
        if (
            current_mapper_path
            and (
                self._normalized_mapper_path(current_mapper_path)
                in self._protected_mapper_paths
                or Path(current_mapper_path).expanduser().is_file()
            )
        ):
            mapper_path = Path(current_mapper_path).expanduser()
            suffix = mapper_path.suffix or ".qcs"
            self.mapper_path.setText(
                str(
                    mapper_path.with_name(
                        f"{mapper_path.stem}_front_panel{suffix}"
                    )
                )
            )
        self._set_mapper_write_allowed(True)
        self._set_status(
            "Restored the M9046A diagram layout as a new draft. Enter an LO "
            "for every M5300 mapping, then save the mapper before running.",
            error=False,
        )

    def _choose_mapper_output(self) -> Optional[str]:
        initial_path = self.mapper_path.text().strip() or "qcs_channel_mapper.qcs"
        if (
            self._normalized_mapper_path(initial_path)
            in self._protected_mapper_paths
        ):
            path = Path(initial_path).expanduser()
            suffix = path.suffix or ".qcs"
            initial_path = str(
                path.with_name(f"{path.stem}_front_panel{suffix}")
            )
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Choose QCS ChannelMapper output",
            initial_path,
            "Keysight QCS mapper (*.qcs);;All files (*)",
        )
        if path:
            output_path = Path(path)
            if output_path.suffix.lower() != ".qcs":
                output_path = output_path.with_suffix(".qcs")
            if (
                self._normalized_mapper_path(str(output_path))
                in self._protected_mapper_paths
            ):
                self._set_status(
                    "Choose a different file name. The selected native mapper "
                    "is protected because this editor cannot preserve all of "
                    "its imported settings and links.",
                    error=True,
                )
                return None
            self.mapper_path.setText(str(output_path))
            return str(output_path)
        return None

    def _choose_mapper_input(self) -> Optional[str]:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load QCS ChannelMapper",
            self.mapper_path.text().strip(),
            "Keysight QCS mapper (*.qcs);;All files (*)",
        )
        return path or None

    def _set_status(self, message: str, *, error: bool) -> None:
        self.status.setText(str(message))
        self.status.setStyleSheet(
            "color: #b00020;" if error else "color: #167c3a;"
        )

    def _stage_graphical_configuration(self, *_args) -> None:
        """Publish picture/IP topology edits without making them executable."""

        if self._loading:
            return
        try:
            configuration = self.working_configuration()
            state = self._effective_configuration_state(
                configuration,
                self.mapper_path.text().strip(),
            )
        except (TypeError, ValueError) as exc:
            self._set_status(
                f"The graphical hardware draft is invalid: {exc}",
                error=True,
            )
            return
        self._configuration_state = state
        # QLineEdit emits editingFinished when the dialog closes even when
        # the IP text did not change. Do not turn an authoritative saved or
        # imported configuration back into a pending draft in that case.
        if state in {
            QCS_HARDWARE_STATE_SAVED,
            QCS_HARDWARE_STATE_IMPORTED,
        }:
            return
        self._mapper_file_sha256 = None
        self.draft_staged.emit(configuration)
        self._set_status(
            "Hardware draft updated. Select the affected channel SMA to "
            "save it automatically, or use Advanced Hardware Settings.",
            error=False,
        )

    def _set_mapper_write_allowed(self, allowed: bool) -> None:
        self._mapper_write_allowed = bool(allowed)
        self.write_mapper_button.setEnabled(
            self._mapper_write_allowed and self.tabs.isEnabled()
        )
        self.write_mapper_button.setToolTip(
            "Create and reload a fresh native QCS ChannelMapper"
            if self._mapper_write_allowed
            else (
                "Save As is disabled for imported mappers because channel "
                "settings, constraints, and timing details are not all "
                "represented by this editor"
            )
        )

    def set_apply_preflight(self, callback) -> None:
        """Install a synchronous guard run before Apply or native file writes."""
        self._apply_preflight = callback

    def _apply_preflight_passes(self) -> bool:
        if self._apply_preflight is None:
            return True
        try:
            self._apply_preflight()
        except (OSError, TypeError, ValueError) as exc:
            self._set_status(str(exc), error=True)
            return False
        return True

    def _configuration_from_widgets(self) -> dict:
        return normalize_qcs_hardware_configuration(
            {
                "version": QCS_HARDWARE_CONFIGURATION_VERSION,
                "chassis_model": self.chassis_model.text().strip(),
                "chassis": self.chassis_number.value(),
                "host_controller": self.host_controller.value(),
                "ip_address": self.ip_address.text().strip() or None,
                "modules": self._modules_from_widgets(),
                "channel_mappings": self._mappings_from_widgets(),
                "downconverter_links": (
                    self._downconverter_links_from_widgets()
                ),
            },
            required_dc_count=self._output_count,
        )

    def working_configuration(self) -> dict:
        """Return the valid editor topology even while role rows are incomplete."""

        return normalize_qcs_hardware_configuration(
            {
                "version": QCS_HARDWARE_CONFIGURATION_VERSION,
                "chassis_model": self.chassis_model.text().strip(),
                "chassis": self.chassis_number.value(),
                "host_controller": self.host_controller.value(),
                "ip_address": self.ip_address.text().strip() or None,
                "modules": self._modules_from_widgets(),
                "channel_mappings": self._mappings_from_widgets(),
                "downconverter_links": (
                    self._downconverter_links_from_widgets()
                ),
            }
        )

    def working_source_bindings(
        self,
    ) -> tuple[list[str], dict[int, str], Optional[str]]:
        """Return the experiment role names represented by this editor draft."""

        return self._current_source_bindings()

    def update_source_dc_channels(
        self,
        dc_channel_names: Sequence[str],
        *,
        output_count: int,
    ) -> None:
        """Expand draft source bindings without replacing its hardware layout."""

        output_count = _positive_integer(
            output_count,
            "QCS output count",
        )
        names = tuple(_virtual_name(name) for name in dc_channel_names)
        if len(names) != output_count:
            raise ValueError(
                "QCS DC source bindings must contain one name per output"
            )
        if len(set(names)) != len(names):
            raise ValueError("QCS DC source binding names must be unique")
        self._output_count = output_count
        self._source_settings = dict(self._source_settings)
        self._source_settings["dc_channel_names"] = names

    def _set_configuration_widgets(
        self,
        configuration: Mapping[str, Any],
    ) -> None:
        normalized = normalize_qcs_hardware_configuration(configuration)
        previous_loading = self._loading
        self._loading = True
        try:
            self.chassis_model.setText(normalized["chassis_model"])
            self.chassis_number.setValue(normalized["chassis"])
            self.host_controller.setValue(normalized["host_controller"])
            self.ip_address.setText(normalized["ip_address"] or "")
            self._set_modules(normalized["modules"])
            self._set_mappings(normalized["channel_mappings"])
            self._set_downconverter_links(
                normalized["downconverter_links"]
            )
        finally:
            self._loading = previous_loading
        self._preview_refresh_timer.stop()
        self._set_reference_configuration(normalized)

    def set_settings(
        self,
        settings: Mapping[str, Any],
        *,
        output_count: int,
    ) -> None:
        """Populate the builder without changing live experiment settings."""
        self._close_m5201_route_dialog()
        self._rf_acquisition_path_focus = None
        self._focused_mapping = None
        self.reference_label.setToolTip(
            "Click an empty slot to install a module, or click a module to "
            "replace or remove it."
        )
        self._output_count = _positive_integer(
            output_count, "QCS output count"
        )
        self._source_settings = dict(settings)
        mapper_path = str(
            settings.get("mapper_path", "qcs_channel_mapper.qcs")
        )
        self.mapper_path.setText(mapper_path)
        configuration = settings.get("hardware_configuration")
        raw_state = str(
            settings.get(
                "hardware_configuration_state",
                (
                    QCS_HARDWARE_STATE_EXTERNAL
                    if configuration is None
                    else QCS_HARDWARE_STATE_DRAFT
                ),
            )
        )
        if raw_state not in QCS_HARDWARE_STATES:
            raise ValueError(
                f"unsupported QCS hardware configuration state {raw_state!r}"
            )
        mapper_file_sha256 = normalize_qcs_mapper_sha256(
            settings.get("hardware_mapper_sha256")
        )
        if configuration is None:
            raw_state = QCS_HARDWARE_STATE_EXTERNAL
            mapper_file_sha256 = None
            dc_names, rf_names, acquisition_name = (
                self._current_source_bindings()
            )
            configuration = default_qcs_hardware_configuration(
                dc_names,
                rf_names,
                acquisition_name,
            )
        elif raw_state == QCS_HARDWARE_STATE_EXTERNAL:
            raw_state = QCS_HARDWARE_STATE_DRAFT
        if (
            raw_state == QCS_HARDWARE_STATE_SAVED
            and mapper_file_sha256 is None
        ):
            raw_state = QCS_HARDWARE_STATE_DRAFT
        elif (
            raw_state == QCS_HARDWARE_STATE_IMPORTED
            and mapper_file_sha256 is None
        ):
            raw_state = QCS_HARDWARE_STATE_IMPORTED_DIRTY
        configuration = normalize_qcs_hardware_configuration(configuration)
        self._configuration_state = raw_state
        self._mapper_file_sha256 = mapper_file_sha256
        self._protected_mapper_paths = set()
        mapper_exists = Path(mapper_path).expanduser().is_file()
        if raw_state in {
            QCS_HARDWARE_STATE_IMPORTED,
            QCS_HARDWARE_STATE_IMPORTED_DIRTY,
        } or (
            raw_state == QCS_HARDWARE_STATE_EXTERNAL and mapper_exists
        ):
            self._protected_mapper_paths.add(
                self._normalized_mapper_path(mapper_path)
            )
        if raw_state in {
            QCS_HARDWARE_STATE_SAVED,
            QCS_HARDWARE_STATE_IMPORTED,
        }:
            self._authoritative_fingerprint = (
                qcs_hardware_mapper_fingerprint(configuration)
            )
            self._authoritative_mapper_path = self._normalized_mapper_path(
                mapper_path
            )
        else:
            self._authoritative_fingerprint = None
            self._authoritative_mapper_path = None
        self._set_configuration_widgets(configuration)
        imported = raw_state in {
            QCS_HARDWARE_STATE_IMPORTED,
            QCS_HARDWARE_STATE_IMPORTED_DIRTY,
        }
        self._set_mapper_write_allowed(not imported)
        if raw_state == QCS_HARDWARE_STATE_EXTERNAL:
            message = (
                "The selected mapper is external; this diagram is only a new "
                "draft until it is saved and applied."
            )
        elif raw_state == QCS_HARDWARE_STATE_DRAFT:
            message = (
                "Unsaved hardware draft loaded. It cannot run until a native "
                "mapper is saved."
            )
        elif raw_state == QCS_HARDWARE_STATE_IMPORTED_DIRTY:
            message = (
                "The imported topology has unsaved edits. Reload the original "
                "mapper to discard them, or Restore diagram layout and save a "
                "new mapper file."
            )
        elif raw_state == QCS_HARDWARE_STATE_IMPORTED:
            message = (
                "Imported mapper loaded read-only. Role assignments may be "
                "applied without rebuilding the native mapper."
            )
        else:
            message = "Saved hardware configuration loaded."
        self._set_status(message, error=False)

    def _settings_for_configuration(
        self,
        configuration: Mapping[str, Any],
    ) -> dict:
        mapper_path = self.mapper_path.text().strip()
        if not mapper_path:
            raise ValueError("QCS ChannelMapper file path must not be empty")
        configuration = normalize_qcs_hardware_configuration(
            configuration,
            required_dc_count=self._output_count,
        )
        dc_names, rf_names, acquisition_name = qcs_role_bindings(
            configuration,
            required_dc_count=self._output_count,
        )
        state = self._effective_configuration_state(
            configuration,
            mapper_path,
        )
        mapper_file_sha256 = (
            self._mapper_file_sha256
            if state
            in {
                QCS_HARDWARE_STATE_SAVED,
                QCS_HARDWARE_STATE_IMPORTED,
            }
            else None
        )
        return {
            "mapper_path": mapper_path,
            "dc_channel_names": dc_names,
            "rf_channel_names": rf_names,
            "acquisition_channel_name": acquisition_name,
            "hardware_configuration": configuration,
            "hardware_configuration_state": state,
            "hardware_mapper_sha256": mapper_file_sha256,
        }

    def settings_dict(self) -> dict:
        return self._settings_for_configuration(
            self._configuration_from_widgets()
        )

    def validate_settings(
        self,
        *,
        verify_mapper_identity: bool = True,
    ) -> Optional[dict]:
        try:
            settings = self.settings_dict()
            state = settings["hardware_configuration_state"]
            if verify_mapper_identity and state in {
                QCS_HARDWARE_STATE_SAVED,
                QCS_HARDWARE_STATE_IMPORTED,
                QCS_HARDWARE_STATE_IMPORTED_DIRTY,
            }:
                mapper_path = Path(settings["mapper_path"]).expanduser()
                if not mapper_path.is_file():
                    raise FileNotFoundError(
                        f"QCS ChannelMapper file not found: {mapper_path}"
                    )
                if (
                    self._mapper_file_sha256 is not None
                    and qcs_mapper_file_sha256(mapper_path)
                    != self._mapper_file_sha256
                ):
                    raise ValueError(
                        "The QCS ChannelMapper file changed after it was "
                        "loaded or saved. Reload or save it again before "
                        "applying the hardware configuration."
                    )
                if state == QCS_HARDWARE_STATE_IMPORTED:
                    validate_imported_qcs_role_configuration(
                        settings["hardware_configuration"],
                        mapper_path,
                    )
                elif state == QCS_HARDWARE_STATE_SAVED:
                    build_qcs_channel_mapper(
                        settings["hardware_configuration"]
                    )
            elif state not in {
                QCS_HARDWARE_STATE_IMPORTED,
                QCS_HARDWARE_STATE_IMPORTED_DIRTY,
            }:
                build_qcs_channel_mapper(
                    settings["hardware_configuration"]
                )
        except (ImportError, OSError, TypeError, ValueError) as exc:
            self._set_status(str(exc), error=True)
            return None
        mapping_count = len(
            settings["hardware_configuration"]["channel_mappings"]
        )
        if settings["hardware_configuration_state"] in {
            QCS_HARDWARE_STATE_IMPORTED,
            QCS_HARDWARE_STATE_IMPORTED_DIRTY,
        }:
            message = (
                f"Imported role bindings valid: {mapping_count} virtual "
                "channel mapping(s); the native mapper remains unchanged."
            )
        else:
            message = (
                f"Topology valid: {mapping_count} virtual channel mapping(s)."
            )
        self._set_status(message, error=False)
        return settings

    def apply_settings(self) -> bool:
        if not self._apply_preflight_passes():
            return False
        settings = self.validate_settings()
        if settings is None:
            return False
        self._configuration_state = settings[
            "hardware_configuration_state"
        ]
        self._source_settings = dict(settings)
        self.settings_applied.emit(settings)
        if self._configuration_state == QCS_HARDWARE_STATE_DRAFT:
            message = (
                "Applied as an unsaved hardware draft. QCS Run is blocked "
                "until a native mapper is saved."
            )
        elif (
            self._configuration_state
            == QCS_HARDWARE_STATE_IMPORTED_DIRTY
        ):
            message = (
                "Applied imported-topology edits, but QCS Run remains blocked. "
                "Reload the original mapper to discard them, or Restore "
                "diagram layout and save a new mapper file."
            )
        else:
            message = "Applied QCS channel roles to the current experiment."
        self._set_status(message, error=False)
        return True

    def _automatic_mapper_output_path(
        self,
        configuration: Mapping[str, Any],
    ) -> Path:
        """Return a stable app-owned path for one generated native mapper."""

        app_data = QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.AppLocalDataLocation
        )
        root = Path(app_data) if app_data else Path.home() / ".pulsegenerator"
        mapper_directory = root / "qcs_mappers"
        mapper_directory.mkdir(parents=True, exist_ok=True)
        fingerprint = qcs_hardware_mapper_fingerprint(configuration)
        return mapper_directory / f"front_panel_{fingerprint[:24]}.qcs"

    def prepare_connector_selection(self) -> Optional[dict]:
        """Capture a validated connector edit without importing QCS.

        Native ``ChannelMapper`` construction, save, and reload can take
        several seconds on the first QCS import.  The main window uses this
        lightweight preparation step on the GUI thread, then performs the
        native work in a dedicated worker thread.  ``apply_connector_selection``
        remains as the synchronous API for standalone users of this widget.
        """

        if not self._apply_preflight_passes():
            return None
        try:
            settings = self.settings_dict()
            if self._mapper_write_allowed:
                mapper_path = self._automatic_mapper_output_path(
                    settings["hardware_configuration"]
                )
                write_mapper = True
            else:
                if (
                    settings["hardware_configuration_state"]
                    != QCS_HARDWARE_STATE_IMPORTED
                ):
                    raise ValueError(
                        "This selection changed an imported mapper. "
                        "Automatic saving is disabled because the editor "
                        "cannot preserve every third-party mapper setting; "
                        "open Advanced Hardware Settings to restore the "
                        "diagram layout or load an app-owned mapper."
                    )
                mapper_path = Path(settings["mapper_path"]).expanduser()
                write_mapper = False
        except (OSError, TypeError, ValueError) as exc:
            self._set_status(str(exc), error=True)
            return None

        self._set_status(
            (
                "SMA selected. Generating and validating the native QCS "
                "mapper in the background..."
                if write_mapper
                else (
                    "SMA selected. Validating the imported QCS mapper in "
                    "the background..."
                )
            ),
            error=False,
        )
        return {
            "settings": settings,
            "mapper_path": str(mapper_path),
            "write_mapper": write_mapper,
        }

    def complete_connector_selection(
        self,
        settings: Mapping[str, Any],
        *,
        commit_callback=None,
    ) -> bool:
        """Finalize a connector edit after native mapper work succeeds."""

        settings = dict(settings)
        configuration = normalize_qcs_hardware_configuration(
            settings.get("hardware_configuration"),
            required_dc_count=self._output_count,
        )
        try:
            current = self._configuration_from_widgets()
        except (TypeError, ValueError) as exc:
            self._set_status(
                f"The native mapper finished, but the current front panel "
                f"is invalid: {exc}",
                error=True,
            )
            return False
        if current != configuration:
            self._set_status(
                "A newer front-panel selection replaced this mapper result; "
                "the current selection is still being processed.",
                error=False,
            )
            return False

        settings["hardware_configuration"] = configuration
        state = str(settings.get("hardware_configuration_state", ""))
        if state not in {
            QCS_HARDWARE_STATE_SAVED,
            QCS_HARDWARE_STATE_IMPORTED,
        }:
            self._set_status(
                "The completed QCS mapper does not have a runnable state.",
                error=True,
            )
            return False
        mapper_path = str(settings["mapper_path"])
        mapper_sha256 = normalize_qcs_mapper_sha256(
            settings.get("hardware_mapper_sha256")
        )
        if mapper_sha256 is None:
            self._set_status(
                "The completed QCS mapper has no file identity digest.",
                error=True,
            )
            return False
        if commit_callback is not None:
            try:
                committed = commit_callback(settings)
            except (OSError, TypeError, ValueError) as exc:
                self._set_status(
                    "The native mapper was generated, but the experiment "
                    f"rejected the assignment: {exc}",
                    error=True,
                )
                return False
            if committed is False:
                self._set_status(
                    "The native mapper was generated, but the experiment "
                    "rejected the assignment.",
                    error=True,
                )
                return False
        else:
            self.settings_applied.emit(settings)

        self.mapper_path.setText(mapper_path)
        self._configuration_state = state
        self._mapper_file_sha256 = mapper_sha256
        self._authoritative_fingerprint = qcs_hardware_mapper_fingerprint(
            configuration
        )
        self._authoritative_mapper_path = self._normalized_mapper_path(
            mapper_path
        )
        self._source_settings = dict(settings)
        self._set_mapper_write_allowed(
            state != QCS_HARDWARE_STATE_IMPORTED
        )
        if state == QCS_HARDWARE_STATE_SAVED:
            self.mapper_saved.emit(mapper_path)
            message = (
                "Mapped the selected SMA and generated the native QCS "
                "mapper automatically."
            )
        else:
            message = (
                "Applied the selected role to the existing imported virtual "
                "channel. The native QCS mapper file was not rewritten."
            )
        self._set_status(message, error=False)
        return True

    def fail_connector_selection(self, details: str) -> None:
        """Keep a failed background selection visible and editable."""

        lines = [line for line in str(details).splitlines() if line.strip()]
        summary = lines[-1] if lines else "Unknown QCS mapper error"
        self._set_status(
            "The SMA remains selected, but its automatic QCS mapper could "
            f"not be generated: {summary}",
            error=True,
        )

    def apply_connector_selection(self, commit_callback=None) -> bool:
        """Generate and apply a run-ready mapper after a contextual SMA click."""

        prepared = self.prepare_connector_selection()
        if prepared is None:
            return False
        settings = dict(prepared["settings"])
        try:
            if prepared["write_mapper"]:
                saved_path = save_qcs_channel_mapper(
                    settings["hardware_configuration"],
                    prepared["mapper_path"],
                )
                state = QCS_HARDWARE_STATE_SAVED
            else:
                validate_imported_qcs_role_configuration(
                    settings["hardware_configuration"],
                    prepared["mapper_path"],
                )
                saved_path = Path(prepared["mapper_path"]).resolve()
                state = QCS_HARDWARE_STATE_IMPORTED
            mapper_sha256 = qcs_mapper_file_sha256(saved_path)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self.fail_connector_selection(str(exc))
            return False

        settings["mapper_path"] = str(saved_path)
        settings["hardware_configuration_state"] = state
        settings["hardware_mapper_sha256"] = mapper_sha256
        return self.complete_connector_selection(
            settings,
            commit_callback=commit_callback,
        )

    def write_mapper(self) -> None:
        if not self._apply_preflight_passes():
            return
        if not self._mapper_write_allowed:
            self._set_status(
                "This imported mapper is read-only for Save As so existing "
                "channel settings, constraints, and timing details cannot be "
                "lost. "
                "Use Restore diagram layout to start a fresh mapper.",
                error=True,
            )
            return
        settings = self.validate_settings(verify_mapper_identity=False)
        if settings is None:
            return
        path = self._choose_mapper_output()
        if not path:
            return
        try:
            settings = self.settings_dict()
            build_qcs_channel_mapper(settings["hardware_configuration"])
        except (ImportError, OSError, TypeError, ValueError) as exc:
            self._set_status(str(exc), error=True)
            return
        try:
            saved_path = save_qcs_channel_mapper(
                settings["hardware_configuration"],
                path,
            )
        except (ImportError, OSError, TypeError, ValueError) as exc:
            self._set_status(str(exc), error=True)
            return
        self.mapper_path.setText(str(saved_path))
        settings["mapper_path"] = str(saved_path)
        settings["hardware_configuration_state"] = (
            QCS_HARDWARE_STATE_SAVED
        )
        settings["hardware_mapper_sha256"] = qcs_mapper_file_sha256(
            saved_path
        )
        self._configuration_state = QCS_HARDWARE_STATE_SAVED
        self._mapper_file_sha256 = settings["hardware_mapper_sha256"]
        self._authoritative_fingerprint = qcs_hardware_mapper_fingerprint(
            settings["hardware_configuration"]
        )
        self._authoritative_mapper_path = self._normalized_mapper_path(
            str(saved_path)
        )
        self._source_settings = dict(settings)
        self._set_mapper_write_allowed(True)
        self.settings_applied.emit(settings)
        self.mapper_saved.emit(str(saved_path))
        self._set_status(
            f"Saved, reloaded, and applied QCS mapper: {saved_path}",
            error=False,
        )

    def load_mapper(self, path=None) -> None:
        if not self._apply_preflight_passes():
            return
        if isinstance(path, bool) or path is None:
            path = self._choose_mapper_input()
        if not path:
            return
        try:
            dc_names, rf_names, acquisition_name = (
                self._current_source_bindings()
            )
            configuration = configuration_from_qcs_mapper(
                path,
                dc_channel_names=dc_names,
                rf_channel_names=rf_names,
                acquisition_channel_name=acquisition_name,
            )
        except (ImportError, OSError, TypeError, ValueError) as exc:
            self._set_status(str(exc), error=True)
            return
        resolved_path = str(Path(path).resolve())
        self.mapper_path.setText(resolved_path)
        self._set_configuration_widgets(configuration)
        self._configuration_state = QCS_HARDWARE_STATE_IMPORTED
        self._authoritative_fingerprint = qcs_hardware_mapper_fingerprint(
            configuration
        )
        self._authoritative_mapper_path = self._normalized_mapper_path(
            resolved_path
        )
        self._mapper_file_sha256 = qcs_mapper_file_sha256(resolved_path)
        self._protected_mapper_paths = {
            self._authoritative_mapper_path
        }
        self._set_mapper_write_allowed(False)
        self._set_status(
            "Loaded mapper for inspection and role binding. Review roles and "
            "GUI channel numbers, then Apply. Save As is disabled so imported "
            "channel settings, constraints, and timing details cannot be lost.",
            error=False,
        )

    def clear_mapping_focus(self) -> None:
        """Leave connector-selection mode without discarding editor changes."""

        self._close_m5201_route_dialog()
        self._rf_acquisition_path_focus = None
        self._focused_mapping = None
        self.mapping_table.clearSelection()
        self.reference_label.setToolTip(
            "Click an empty slot to install a module, or click a module to "
            "replace or remove it."
        )
        self._preview_refresh_timer.stop()
        self._refresh_reference_preview()

    def focus_mapping(self, role: str, logical_index: int = 0) -> bool:
        """Highlight and enable SMA selection for one logical mapping."""
        self._close_m5201_route_dialog()
        self._rf_acquisition_path_focus = None
        role = str(role).strip().lower()
        if role not in QCS_ROLE_LABELS or role == "unassigned":
            raise ValueError(f"unsupported QCS mapping role {role!r}")
        logical_index = _nonnegative_integer(
            logical_index,
            "QCS mapping logical index",
        )
        self._focused_mapping = (role, logical_index)
        self.show_front_panel()
        row = self._focused_mapping_row()
        compatible_models = [
            model
            for model, spec in QCS_MODULE_MODELS.items()
            if spec["instrument"] in QCS_ROLE_INSTRUMENTS[role]
        ]
        model_text = " or ".join(compatible_models)
        self.reference_label.setToolTip(
            f"Click a {model_text} channel SMA to select it for "
            f"{_qcs_mapping_display_name(role, logical_index)}. Click a module "
            "away from its channel SMAs to replace or remove the module."
        )
        self._preview_refresh_timer.stop()
        self._refresh_reference_preview()
        if row is None:
            self.mapping_table.clearSelection()
            if role in {"dc", "rf", "acquisition"}:
                self._set_status(
                    f"{_qcs_mapping_display_name(role, logical_index)} is not "
                    f"mapped. Click a {model_text} channel SMA to create and "
                    "assign it.",
                    error=False,
                )
                return True
            self._set_status(
                f"{_qcs_mapping_display_name(role, logical_index)} is not "
                "mapped. This channel type needs additional hardware "
                "settings; configure it in Advanced Hardware Settings.",
                error=True,
            )
            return False
        self.mapping_table.selectRow(row)
        self.mapping_table.setCurrentCell(row, 2)
        item = self.mapping_table.item(row, 8)
        if item is not None:
            self.mapping_table.scrollToItem(
                item,
                QtWidgets.QAbstractItemView.PositionAtCenter,
            )
        slot = int(self.mapping_table.cellWidget(row, 6).value())
        channel = int(self.mapping_table.cellWidget(row, 7).value())
        model = self._module_models_by_slot.get(slot, "QCS module")
        self._set_status(
            f"Highlighted {model} slot {slot} CH{channel} for "
            f"{_qcs_mapping_display_name(role, logical_index)}. Click another "
            f"{model_text} channel SMA to change it; the native mapper will "
            "be updated automatically.",
            error=False,
        )
        return True

    def focus_rf_acquisition_path(
        self,
        rf_logical_index: int = 0,
        acquisition_logical_index: int = 0,
    ) -> bool:
        """Select an RF output or acquisition input by the clicked module.

        This scoped mode is used by RF-path previews such as Stability and
        S-Parameter. M5300A or M5301A SMA clicks target the selected RF virtual
        channel, while M5200A SMA clicks target the acquisition virtual
        channel. M5201A clicks keep using the existing graphical
        downconverter-route editor.
        """

        rf_logical_index = _nonnegative_integer(
            rf_logical_index,
            "QCS RF path logical index",
        )
        acquisition_logical_index = _nonnegative_integer(
            acquisition_logical_index,
            "QCS acquisition path logical index",
        )
        # Reuse the established RF focus for its current highlight and table
        # selection, then enable module-driven endpoint routing.
        focused = self.focus_mapping("rf", rf_logical_index)
        self._rf_acquisition_path_focus = (
            ("rf", rf_logical_index),
            ("acquisition", acquisition_logical_index),
        )
        self._preview_refresh_timer.stop()
        self._refresh_reference_preview()
        self.reference_label.setToolTip(
            "Click an M5300A/M5301A SMA to select the RF-path output, "
            "or an M5200A SMA to select the RF-path acquisition input. "
            "Click an M5201A pair to configure its acquisition route."
        )
        self._set_status(
            "Select an RF-path endpoint: click an RF output SMA on "
            "M5300A/M5301A, or an acquisition input SMA on M5200A.",
            error=False,
        )
        return focused

    def _refresh_editing_enabled_state(self) -> None:
        enabled = self._editing_enabled and not self._identifying_hardware
        self.tabs.setEnabled(enabled)
        self.mapping_tab.setEnabled(enabled)
        self.advanced_hardware_button.setEnabled(enabled)
        self.validate_button.setEnabled(enabled)
        self.write_mapper_button.setEnabled(
            enabled and self._mapper_write_allowed
        )
        self.apply_button.setEnabled(enabled)
        if not enabled and self._advanced_hardware_dialog.isVisible():
            self._advanced_hardware_dialog.close()
        if not enabled:
            self._close_m5201_route_dialog()

    def set_editing_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        was_enabled = self._editing_enabled
        if not enabled and was_enabled:
            self._status_before_lock = (
                self.status.text(),
                self.status.styleSheet(),
            )
        self._editing_enabled = enabled
        self._refresh_editing_enabled_state()
        if not enabled:
            self._set_status(
                "Hardware configuration is locked while an experiment runs.",
                error=True,
            )
        elif not was_enabled and self._status_before_lock is not None:
            message, style_sheet = self._status_before_lock
            self.status.setText(message)
            self.status.setStyleSheet(style_sheet)
            self._status_before_lock = None


__all__ = [
    "DEFAULT_QCS_IP_ADDRESS",
    "QCS_CHASSIS_MODEL",
    "QCS_CHASSIS_SLOT_COUNT",
    "QCS_FRONT_PANEL_IMAGE_PATH",
    "QCS_HARDWARE_DISCOVERY_TIMEOUT_S",
    "QCS_HARDWARE_CONFIGURATION_VERSION",
    "QCS_HARDWARE_STATE_DRAFT",
    "QCS_HARDWARE_STATE_EXTERNAL",
    "QCS_HARDWARE_STATE_IMPORTED",
    "QCS_HARDWARE_STATE_IMPORTED_DIRTY",
    "QCS_HARDWARE_STATE_SAVED",
    "QCS_HARDWARE_STATES",
    "QCS_MODULE_MODELS",
    "QcsFrontPanelControl",
    "QcsFrontPanelPreview",
    "QcsM5201RouteDialog",
    "build_qcs_channel_mapper",
    "configuration_from_qcs_mapper",
    "default_qcs_hardware_configuration",
    "identify_qcs_hardware_configuration",
    "merge_qcs_discovered_hardware_configuration",
    "normalize_qcs_hardware_configuration",
    "normalize_qcs_mapper_sha256",
    "normalize_qcs_server_ip",
    "qcs_hardware_mapper_fingerprint",
    "qcs_mapper_file_sha256",
    "qcs_role_bindings",
    "resize_qcs_dc_mappings",
    "save_qcs_channel_mapper",
    "synchronize_qcs_hardware_role_names",
    "validate_imported_qcs_role_configuration",
]
