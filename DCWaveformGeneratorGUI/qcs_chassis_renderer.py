"""Deterministic Pillow compositor for an M9046A QCS chassis.

This module has no Qt dependency.  It accepts the normalized QCS
``hardware_configuration`` dictionary used by PulseGenerator, loads the
finalized Python front-panel PNGs from ``assets/``, and places them into an
18-slot M9046A chassis.  M5300A occupies two slots; M5301A, M5200A, and
M5201A each occupy one.  Every other unoccupied slot receives a generated
filler panel.

Channel mappings are shown as an address legend below the chassis.  The
renderer deliberately draws no cables or signal-routing lines.

Typical use::

    image = render_qcs_chassis(configuration)
    save_qcs_chassis_png(configuration, "qcs_chassis.png")

The command-line entry point accepts a normalized configuration JSON file::

    python qcs_chassis_renderer.py configuration.json chassis.png
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image, ImageDraw, ImageFont


PathLike = Union[str, Path]

CHASSIS_MODEL = "M9046A"
CHASSIS_SLOT_COUNT = 18
ASSET_DIRECTORY = Path(__file__).resolve().parent / "assets"

MODULE_SPECS = {
    "M9032A": {
        "span": 1,
        "channels": 0,
        "asset_name": None,
    },
    "M5301A": {
        "span": 1,
        "channels": 4,
        "asset_name": "M5301A_python_front_panel.png",
    },
    "M5300A": {
        "span": 2,
        "channels": 4,
        "asset_name": "M5300A_python_front_panel.png",
    },
    "M5200A": {
        "span": 1,
        "channels": 4,
        "asset_name": "M5200A_python_front_panel.png",
    },
    "M5201A": {
        "span": 1,
        "channels": 4,
        "asset_name": "M5201A_python_front_panel.png",
    },
}

PANEL_ASSET_MODELS = ("M5300A", "M5301A", "M5200A", "M5201A")

ROLE_COLORS = {
    "dc": "#22AAB7",
    "rf": "#FF9F1C",
    "acquisition": "#4DBB73",
    "downconverter": "#8B5CF6",
    "unassigned": "#8E9AA4",
}

# Connector centers in the finalized Python-generated source assets.  Only
# the large front-panel SMA connectors represent QCS physical channels; the
# adjacent SMP connectors are internal/inter-module paths and are deliberately
# excluded from selection.
CHANNEL_CONNECTOR_LAYOUT = {
    "M5301A": {
        "source_size": (300, 1300),
        "channels": {
            1: (95, 300),
            2: (95, 560),
            3: (95, 820),
            4: (95, 1080),
        },
    },
    "M5300A": {
        "source_size": (600, 1300),
        "channels": {
            1: (185, 285),
            2: (415, 285),
            3: (185, 805),
            4: (415, 805),
        },
    },
    "M5200A": {
        "source_size": (300, 1300),
        "channels": {
            1: (95, 300),
            2: (95, 560),
            3: (95, 820),
            4: (95, 1080),
        },
    },
    "M5201A": {
        "source_size": (300, 1300),
        # QCS addresses an RF/IF pair as one downconverter channel.  The
        # selectable front-panel endpoint is its RF input connector.
        "channels": {
            1: (92, 330),
            2: (92, 540),
            3: (92, 750),
            4: (92, 960),
        },
    },
}

ROLE_SELECTABLE_MODELS = {
    "dc": {"M5301A"},
    "rf": {"M5300A", "M5301A"},
    "acquisition": {"M5200A"},
    "downconverter": {"M5201A"},
    "unassigned": {"M5300A", "M5301A", "M5200A"},
}

_BACKGROUND = "#E8EEF4"
_CHASSIS_OUTER = "#1E252A"
_CHASSIS_INNER = "#343C42"
_BAY_BACKGROUND = "#11171B"
_FILLER_BACKGROUND = "#242B30"
_FILLER_BORDER = "#657079"
_FILLER_VENT = "#11171B"
_TEXT = "#F3F6F8"
_DARK_TEXT = "#17212B"
_MUTED_TEXT = "#607386"

DEFAULT_SLOT_WIDTH = 101
DEFAULT_PANEL_HEIGHT = 438
DEFAULT_SCALE = 1
DEFAULT_CHASSIS_LEFT_MARGIN = 48
DEFAULT_CHASSIS_RIGHT_MARGIN = 48
DEFAULT_CHASSIS_HEADER_HEIGHT = 88
DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT = 34


def _strict_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 1 or result != value:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _validate_configuration(configuration: Mapping[str, Any]) -> dict:
    """Build a checked render plan from a normalized QCS configuration."""

    if not isinstance(configuration, Mapping):
        raise TypeError("QCS hardware configuration must be an object")
    chassis_model = str(configuration.get("chassis_model", "")).strip()
    if chassis_model != CHASSIS_MODEL:
        raise ValueError(
            f"unsupported QCS chassis {chassis_model!r}; expected {CHASSIS_MODEL}"
        )

    raw_modules = configuration.get("modules")
    if not isinstance(raw_modules, Sequence) or isinstance(
        raw_modules, (str, bytes, bytearray)
    ):
        raise TypeError("QCS hardware modules must be an array")

    modules = []
    occupied_slots: dict[int, str] = {}
    for raw_module in raw_modules:
        if not isinstance(raw_module, Mapping):
            raise TypeError("each QCS hardware module must be an object")
        slot = _strict_positive_int(raw_module.get("slot"), "QCS module slot")
        model = str(raw_module.get("model", "")).strip()
        if model not in MODULE_SPECS:
            supported = ", ".join(MODULE_SPECS)
            raise ValueError(
                f"unsupported QCS module model {model!r}; expected {supported}"
            )
        span = int(MODULE_SPECS[model]["span"])
        if slot + span - 1 > CHASSIS_SLOT_COUNT:
            raise ValueError(
                f"{model} in slot {slot} extends past the "
                f"{CHASSIS_SLOT_COUNT}-slot chassis"
            )
        for occupied_slot in range(slot, slot + span):
            previous = occupied_slots.get(occupied_slot)
            if previous is not None:
                raise ValueError(
                    f"QCS slot {occupied_slot} is occupied by both "
                    f"{previous} and {model}"
                )
            occupied_slots[occupied_slot] = model
        modules.append({"slot": slot, "model": model, "span": span})
    modules.sort(key=lambda item: item["slot"])

    modules_by_slot = {module["slot"]: module for module in modules}
    raw_mappings = configuration.get("channel_mappings")
    if not isinstance(raw_mappings, Sequence) or isinstance(
        raw_mappings, (str, bytes, bytearray)
    ):
        raise TypeError("QCS channel mappings must be an array")

    mappings = []
    used_addresses = set()
    for raw_mapping in raw_mappings:
        if not isinstance(raw_mapping, Mapping):
            raise TypeError("each QCS channel mapping must be an object")
        slot = _strict_positive_int(raw_mapping.get("slot"), "QCS mapping slot")
        channel = _strict_positive_int(
            raw_mapping.get("channel"), "QCS physical channel"
        )
        module = modules_by_slot.get(slot)
        if module is None:
            raise ValueError(
                f"QCS mapping targets slot {slot}, but no module starts there"
            )
        channel_count = int(MODULE_SPECS[module["model"]]["channels"])
        if channel > channel_count:
            raise ValueError(
                f"{module['model']} exposes channels 1-{channel_count}; "
                f"mapping requests channel {channel}"
            )
        address = (slot, channel)
        if address in used_addresses:
            raise ValueError(
                f"QCS slot {slot} channel {channel} is mapped more than once"
            )
        used_addresses.add(address)

        role = str(raw_mapping.get("role", "unassigned")).strip().lower()
        if role not in ROLE_COLORS:
            raise ValueError(f"unsupported QCS channel role {role!r}")
        virtual_name = str(raw_mapping.get("virtual_name", "")).strip()
        if not virtual_name:
            raise ValueError("QCS virtual-channel names must not be empty")
        mappings.append(
            {
                "slot": slot,
                "channel": channel,
                "role": role,
                "virtual_name": virtual_name,
            }
        )
    mappings.sort(key=lambda item: (item["slot"], item["channel"]))

    raw_links = configuration.get("downconverter_links", ())
    if not isinstance(raw_links, Sequence) or isinstance(
        raw_links, (str, bytes, bytearray)
    ):
        raise TypeError("QCS downconverter links must be an array")
    downconverter_links = []
    for raw_link in raw_links:
        if not isinstance(raw_link, Mapping):
            raise TypeError("each QCS downconverter link must be an object")
        digitizer_slot = _strict_positive_int(
            raw_link.get("digitizer_slot"),
            "QCS M5200 digitizer slot",
        )
        digitizer_channel = _strict_positive_int(
            raw_link.get("digitizer_channel"),
            "QCS M5200 digitizer channel",
        )
        downconverter_slot = _strict_positive_int(
            raw_link.get("downconverter_slot"),
            "QCS M5201 downconverter slot",
        )
        downconverter_channel = _strict_positive_int(
            raw_link.get("downconverter_channel"),
            "QCS M5201 downconverter channel",
        )
        digitizer_module = modules_by_slot.get(digitizer_slot)
        downconverter_module = modules_by_slot.get(downconverter_slot)
        if (
            digitizer_module is None
            or digitizer_module["model"] != "M5200A"
        ):
            raise ValueError(
                f"downconverter link requires M5200A in slot "
                f"{digitizer_slot}"
            )
        if (
            downconverter_module is None
            or downconverter_module["model"] != "M5201A"
        ):
            raise ValueError(
                f"downconverter link requires M5201A in slot "
                f"{downconverter_slot}"
            )
        if digitizer_channel > 4 or downconverter_channel > 4:
            raise ValueError("M5200A and M5201A expose channels 1-4")
        raw_lo_frequency_hz = raw_link.get("lo_frequency_hz")
        lo_frequency_hz = (
            None
            if raw_lo_frequency_hz in (None, "")
            else float(raw_lo_frequency_hz)
        )
        downconverter_links.append(
            {
                "digitizer_slot": digitizer_slot,
                "digitizer_channel": digitizer_channel,
                "downconverter_slot": downconverter_slot,
                "downconverter_channel": downconverter_channel,
                "lo_frequency_hz": lo_frequency_hz,
            }
        )
    downconverter_links.sort(
        key=lambda link: (
            link["digitizer_slot"],
            link["digitizer_channel"],
        )
    )

    return {
        "chassis_model": chassis_model,
        "chassis": _strict_positive_int(
            configuration.get("chassis", 1), "QCS chassis number"
        ),
        "host_controller": _strict_positive_int(
            configuration.get("host_controller", 1),
            "QCS host-controller number",
        ),
        "modules": modules,
        "mappings": mappings,
        "downconverter_links": downconverter_links,
        "occupied_slots": occupied_slots,
    }


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.ImageFont:
    """Use Pillow's bundled face so rendering is host-font independent."""

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow versions before scalable load_default().
        return ImageFont.load_default()


def _scaled(value: int, scale: int) -> int:
    return int(value) * int(scale)


def qcs_chassis_slot_at_point(
    x: float,
    y: float,
    *,
    slot_width: int = DEFAULT_SLOT_WIDTH,
    panel_height: int = DEFAULT_PANEL_HEIGHT,
    scale: int = DEFAULT_SCALE,
) -> Optional[int]:
    """Return the 1-based chassis slot at a rendered source-image point.

    The clickable region includes each slot's number label and physical
    module bay.  Header, footer, chassis margins, and right/bottom boundary
    coordinates are outside the slot region.  Coordinates use the same
    source-image coordinate system and render parameters as
    :func:`render_qcs_chassis`.
    """

    slot_width = _strict_positive_int(slot_width, "QCS render slot width")
    panel_height = _strict_positive_int(
        panel_height, "QCS render panel height"
    )
    scale = _strict_positive_int(scale, "QCS render scale")

    slot_pitch = _scaled(slot_width, scale)
    left = _scaled(DEFAULT_CHASSIS_LEFT_MARGIN, scale)
    top = _scaled(DEFAULT_CHASSIS_HEADER_HEIGHT, scale)
    right = left + CHASSIS_SLOT_COUNT * slot_pitch
    bottom = (
        top
        + _scaled(DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT, scale)
        + _scaled(panel_height, scale)
    )
    if x < left or x >= right or y < top or y >= bottom:
        return None
    return int((x - left) // slot_pitch) + 1


def _module_channel_geometry(
    module: Mapping[str, Any],
    channel: int,
    *,
    slot_width: int,
    panel_height: int,
    scale: int,
) -> Optional[tuple[float, float, float]]:
    model = str(module["model"])
    layout = CHANNEL_CONNECTOR_LAYOUT.get(model)
    if layout is None:
        return None
    source_point = layout["channels"].get(int(channel))
    if source_point is None:
        return None
    source_width, source_height = layout["source_size"]
    target_width = slot_width * int(module["span"]) * scale
    target_height = panel_height * scale
    bay_left = DEFAULT_CHASSIS_LEFT_MARGIN * scale
    bay_top = (
        DEFAULT_CHASSIS_HEADER_HEIGHT
        + DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT
    ) * scale
    module_left = (
        bay_left + (int(module["slot"]) - 1) * slot_width * scale
    )
    center_x = (
        module_left + float(source_point[0]) * target_width / source_width
    )
    center_y = (
        bay_top + float(source_point[1]) * target_height / source_height
    )
    source_radius = 52.0
    hit_radius = source_radius * min(
        target_width / source_width,
        target_height / source_height,
    )
    return center_x, center_y, hit_radius


def qcs_chassis_connector_at_point(
    configuration: Mapping[str, Any],
    x: float,
    y: float,
    *,
    role: Optional[str] = None,
    slot_width: int = DEFAULT_SLOT_WIDTH,
    panel_height: int = DEFAULT_PANEL_HEIGHT,
    scale: int = DEFAULT_SCALE,
) -> Optional[dict[str, Any]]:
    """Return the selectable physical SMA at a rendered image point.

    ``role`` limits selection to compatible module types.  For example, DC
    output selection accepts only M5301A channel SMAs, while acquisition
    selection accepts only M5200A channel SMAs.  SMP connectors are never
    returned.
    """

    slot_width = _strict_positive_int(slot_width, "QCS render slot width")
    panel_height = _strict_positive_int(
        panel_height, "QCS render panel height"
    )
    scale = _strict_positive_int(scale, "QCS render scale")
    plan = _validate_configuration(configuration)
    if role is None:
        selectable_models = set(CHANNEL_CONNECTOR_LAYOUT)
    else:
        role = str(role).strip().lower()
        if role not in ROLE_SELECTABLE_MODELS:
            raise ValueError(f"unsupported QCS connector role {role!r}")
        selectable_models = ROLE_SELECTABLE_MODELS[role]

    nearest = None
    nearest_distance = None
    for module in plan["modules"]:
        model = str(module["model"])
        if model not in selectable_models:
            continue
        channel_count = int(MODULE_SPECS[model]["channels"])
        for channel in range(1, channel_count + 1):
            geometry = _module_channel_geometry(
                module,
                channel,
                slot_width=slot_width,
                panel_height=panel_height,
                scale=scale,
            )
            if geometry is None:
                continue
            center_x, center_y, hit_radius = geometry
            distance = (float(x) - center_x) ** 2 + (
                float(y) - center_y
            ) ** 2
            if distance > hit_radius**2:
                continue
            if nearest_distance is not None and distance >= nearest_distance:
                continue
            nearest_distance = distance
            nearest = {
                "slot": int(module["slot"]),
                "channel": channel,
                "model": model,
                "center": (center_x, center_y),
            }
    return nearest


def _text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def _centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    x = left + (right - left - width) // 2
    y = top + (bottom - top - height) // 2 - text_box[1]
    draw.text((x, y), text, font=font, fill=fill)


def _encode_png(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(
        stream,
        format="PNG",
        compress_level=9,
        optimize=False,
    )
    return stream.getvalue()


def _asset_directory(path: Optional[PathLike]) -> Path:
    return ASSET_DIRECTORY if path is None else Path(path).expanduser()


def load_qcs_panel_png_assets(
    asset_directory: Optional[PathLike] = None,
) -> dict[str, bytes]:
    """Load the four finalized physical-panel PNGs as exact bytes.

    The keys are ``M5300A``, ``M5301A``, ``M5200A``, and ``M5201A``. Files are
    verified with Pillow before being returned, so missing or corrupt
    deployment assets fail before chassis composition begins.
    """

    directory = _asset_directory(asset_directory)
    assets = {}
    for model in PANEL_ASSET_MODELS:
        asset_name = str(MODULE_SPECS[model]["asset_name"])
        asset_path = directory / asset_name
        png_bytes = asset_path.read_bytes()
        with Image.open(BytesIO(png_bytes)) as image:
            if image.format != "PNG":
                raise ValueError(f"{asset_path} is not a PNG image")
            image.verify()
        assets[model] = png_bytes
    return assets


def _crop_transparent_margin(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha_box = rgba.getchannel("A").getbbox()
    if alpha_box is None:
        raise ValueError("QCS panel asset is completely transparent")
    return rgba.crop(alpha_box)


def _panel_canvas_from_asset(
    source_png: bytes,
    *,
    span: int,
    slot_width: int,
    panel_height: int,
    scale: int,
) -> Image.Image:
    canvas_width = _scaled(slot_width * span, scale)
    canvas_height = _scaled(panel_height, scale)
    with Image.open(BytesIO(source_png)) as source:
        panel = _crop_transparent_margin(source)
    return panel.resize(
        (canvas_width, canvas_height),
        resample=Image.Resampling.LANCZOS,
    )


def _system_sync_panel(
    *,
    slot_width: int,
    panel_height: int,
    scale: int,
) -> Image.Image:
    """Draw a neutral M9032A placeholder without inferred connectors."""

    width = _scaled(slot_width, scale)
    height = _scaled(panel_height, scale)
    margin = _scaled(7, scale)
    image = Image.new("RGBA", (width, height), "#4E555A")
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (0, 0, width - 1, height - 1),
        fill="#4E555A",
        outline="#AEB8BE",
        width=max(1, _scaled(3, scale)),
    )
    draw.rectangle(
        (
            _scaled(6, scale),
            _scaled(8, scale),
            width - _scaled(6, scale),
            _scaled(22, scale),
        ),
        fill="#3D8BDB",
    )
    title_font = _font(_scaled(17, scale))
    subtitle_font = _font(_scaled(10, scale))
    _centered_text(
        draw,
        (
            margin,
            _scaled(54, scale),
            width - margin,
            _scaled(90, scale),
        ),
        "M9032A",
        font=title_font,
        fill=_TEXT,
    )
    _centered_text(
        draw,
        (
            margin,
            _scaled(92, scale),
            width - margin,
            _scaled(122, scale),
        ),
        "SYSTEM SYNC",
        font=subtitle_font,
        fill=_TEXT,
    )
    for y in range(
        _scaled(164, scale),
        height - _scaled(42, scale),
        _scaled(40, scale),
    ):
        draw.rounded_rectangle(
            (
                margin + _scaled(26, scale),
                y,
                width - margin - _scaled(26, scale),
                y + _scaled(7, scale),
            ),
            radius=_scaled(3, scale),
            fill="#33393D",
        )
    return image


@lru_cache(maxsize=16)
def _filler_panel(
    slot_width: int,
    panel_height: int,
    scale: int,
) -> Image.Image:
    width = _scaled(slot_width, scale)
    height = _scaled(panel_height, scale)
    margin = _scaled(7, scale)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (margin, margin, width - margin, height - margin),
        fill=_FILLER_BACKGROUND,
        outline=_FILLER_BORDER,
        width=max(1, _scaled(2, scale)),
    )
    for y in range(
        _scaled(62, scale),
        height - _scaled(36, scale),
        _scaled(39, scale),
    ):
        draw.rounded_rectangle(
            (
                margin + _scaled(25, scale),
                y,
                width - margin - _scaled(25, scale),
                y + _scaled(7, scale),
            ),
            radius=_scaled(3, scale),
            fill=_FILLER_VENT,
        )
    return image


def generate_qcs_panel_png_assets(
    configuration: Mapping[str, Any],
    *,
    asset_directory: Optional[PathLike] = None,
    output_directory: Optional[PathLike] = None,
    slot_width: int = DEFAULT_SLOT_WIDTH,
    panel_height: int = DEFAULT_PANEL_HEIGHT,
    scale: int = DEFAULT_SCALE,
) -> dict[int, bytes]:
    """Build slot-sized PNG composites from the finalized physical assets.

    The result is keyed by module starting slot.  M5300A result canvases are
    exactly two slots wide; M5301A, M5200A, M5201A, and the neutral M9032A
    placeholder are one slot wide. Supplying ``output_directory`` additionally
    writes ``slot_XX_MODEL.png`` files for inspection or reuse.
    """

    slot_width = _strict_positive_int(slot_width, "QCS render slot width")
    panel_height = _strict_positive_int(
        panel_height, "QCS render panel height"
    )
    scale = _strict_positive_int(scale, "QCS render scale")
    plan = _validate_configuration(configuration)
    source_assets = load_qcs_panel_png_assets(asset_directory)

    assets = {}
    for module in plan["modules"]:
        slot = int(module["slot"])
        model = str(module["model"])
        span = int(module["span"])
        if model == "M9032A":
            panel = _system_sync_panel(
                slot_width=slot_width,
                panel_height=panel_height,
                scale=scale,
            )
        else:
            panel = _panel_canvas_from_asset(
                source_assets[model],
                span=span,
                slot_width=slot_width,
                panel_height=panel_height,
                scale=scale,
            )
        assets[slot] = _encode_png(panel)

    if output_directory is not None:
        directory = Path(output_directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        modules_by_slot = {
            int(module["slot"]): module for module in plan["modules"]
        }
        for slot, png_bytes in assets.items():
            model = modules_by_slot[slot]["model"]
            (directory / f"slot_{slot:02d}_{model}.png").write_bytes(
                png_bytes
            )
    return assets


def _mapping_rows(
    draw: ImageDraw.ImageDraw,
    mappings: Sequence[Mapping[str, Any]],
    downconverter_links: Sequence[Mapping[str, Any]],
    *,
    content_width: int,
    font: ImageFont.ImageFont,
    scale: int,
) -> list[list[tuple[str, str, int]]]:
    if not mappings and not downconverter_links:
        return [[("No virtual-channel mappings", "unassigned", content_width)]]

    horizontal_padding = _scaled(20, scale)
    gap = _scaled(10, scale)
    rows: list[list[tuple[str, str, int]]] = [[]]
    used_width = 0
    for mapping in mappings:
        label = (
            f"S{int(mapping['slot']):02d} CH{int(mapping['channel'])} · "
            f"{mapping['virtual_name']} · {str(mapping['role']).upper()}"
        )
        chip_width = min(
            content_width,
            _text_width(draw, label, font) + horizontal_padding,
        )
        additional = chip_width if not rows[-1] else gap + chip_width
        if rows[-1] and used_width + additional > content_width:
            rows.append([])
            used_width = 0
            additional = chip_width
        rows[-1].append((label, str(mapping["role"]), chip_width))
        used_width += additional
    for link in downconverter_links:
        lo_frequency_hz = link.get("lo_frequency_hz")
        lo_text = (
            "LO unset"
            if lo_frequency_hz is None
            else f"LO {float(lo_frequency_hz) / 1.0e9:.6g} GHz"
        )
        label = (
            f"M5200 S{int(link['digitizer_slot']):02d} "
            f"CH{int(link['digitizer_channel'])} <- M5201 "
            f"S{int(link['downconverter_slot']):02d} "
            f"CH{int(link['downconverter_channel'])} - {lo_text}"
        )
        chip_width = min(
            content_width,
            _text_width(draw, label, font) + horizontal_padding,
        )
        additional = chip_width if not rows[-1] else gap + chip_width
        if rows[-1] and used_width + additional > content_width:
            rows.append([])
            used_width = 0
            additional = chip_width
        rows[-1].append((label, "downconverter", chip_width))
        used_width += additional
    return rows


def render_qcs_chassis(
    configuration: Mapping[str, Any],
    *,
    output_path: Optional[PathLike] = None,
    asset_directory: Optional[PathLike] = None,
    highlighted_address: Optional[tuple[int, int]] = None,
    slot_width: int = DEFAULT_SLOT_WIDTH,
    panel_height: int = DEFAULT_PANEL_HEIGHT,
    scale: int = DEFAULT_SCALE,
) -> Image.Image:
    """Render and optionally save an RGB M9046A chassis image."""

    slot_width = _strict_positive_int(slot_width, "QCS render slot width")
    panel_height = _strict_positive_int(
        panel_height, "QCS render panel height"
    )
    scale = _strict_positive_int(scale, "QCS render scale")
    plan = _validate_configuration(configuration)
    module_assets = generate_qcs_panel_png_assets(
        configuration,
        asset_directory=asset_directory,
        slot_width=slot_width,
        panel_height=panel_height,
        scale=scale,
    )

    left_margin = _scaled(DEFAULT_CHASSIS_LEFT_MARGIN, scale)
    right_margin = _scaled(DEFAULT_CHASSIS_RIGHT_MARGIN, scale)
    header_height = _scaled(DEFAULT_CHASSIS_HEADER_HEIGHT, scale)
    slot_label_height = _scaled(
        DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT, scale
    )
    bay_padding = 0
    slot_pitch = _scaled(slot_width, scale)
    panel_height_px = _scaled(panel_height, scale)
    image_width = left_margin + right_margin + CHASSIS_SLOT_COUNT * slot_pitch
    content_width = image_width - left_margin - right_margin

    mapping_font = _font(_scaled(11, scale))
    sizing_image = Image.new("RGB", (1, 1))
    sizing_draw = ImageDraw.Draw(sizing_image)
    mapping_rows = _mapping_rows(
        sizing_draw,
        plan["mappings"],
        plan["downconverter_links"],
        content_width=content_width,
        font=mapping_font,
        scale=scale,
    )
    mapping_title_height = _scaled(36, scale)
    mapping_row_height = _scaled(30, scale)
    mapping_footer_height = (
        mapping_title_height
        + len(mapping_rows) * mapping_row_height
        + _scaled(26, scale)
    )

    bay_top = header_height + slot_label_height
    bay_height = panel_height_px + 2 * bay_padding
    chassis_bottom = bay_top + bay_height
    image_height = chassis_bottom + mapping_footer_height
    image = Image.new("RGB", (image_width, image_height), _BACKGROUND)
    draw = ImageDraw.Draw(image)

    title_font = _font(_scaled(26, scale))
    subtitle_font = _font(_scaled(13, scale))
    slot_font = _font(_scaled(11, scale))
    mapping_title_font = _font(_scaled(13, scale))
    draw.text(
        (left_margin, _scaled(20, scale)),
        "KEYSIGHT QCS FRONT PANEL",
        font=title_font,
        fill=_DARK_TEXT,
    )
    draw.text(
        (left_margin, _scaled(56, scale)),
        (
            f"{CHASSIS_MODEL} · Chassis {plan['chassis']} · "
            f"Host {plan['host_controller']} · "
            f"{len(plan['modules'])} modules"
        ),
        font=subtitle_font,
        fill=_MUTED_TEXT,
    )

    chassis_left = left_margin - _scaled(20, scale)
    chassis_right = image_width - right_margin + _scaled(20, scale)
    chassis_top = header_height
    draw.rounded_rectangle(
        (chassis_left, chassis_top, chassis_right, chassis_bottom),
        radius=_scaled(18, scale),
        fill=_CHASSIS_OUTER,
        outline="#0D1114",
        width=max(1, _scaled(3, scale)),
    )
    draw.rounded_rectangle(
        (
            chassis_left + _scaled(9, scale),
            chassis_top + _scaled(9, scale),
            chassis_right - _scaled(9, scale),
            chassis_bottom - _scaled(9, scale),
        ),
        radius=_scaled(10, scale),
        fill=_CHASSIS_INNER,
        outline="#7A858D",
        width=max(1, _scaled(1, scale)),
    )

    bay_left = left_margin
    bay_right = bay_left + CHASSIS_SLOT_COUNT * slot_pitch
    draw.rectangle(
        (bay_left, bay_top, bay_right, chassis_bottom),
        fill=_BAY_BACKGROUND,
        outline="#7D8991",
        width=max(1, _scaled(2, scale)),
    )
    for slot in range(1, CHASSIS_SLOT_COUNT + 1):
        slot_x = bay_left + (slot - 1) * slot_pitch
        _centered_text(
            draw,
            (
                slot_x,
                header_height,
                slot_x + slot_pitch,
                bay_top,
            ),
            str(slot),
            font=slot_font,
            fill=_TEXT,
        )
        draw.line(
            (
                slot_x,
                bay_top,
                slot_x,
                chassis_bottom,
            ),
            fill="#455058",
            width=max(1, _scaled(1, scale)),
        )

    paste_y = bay_top + bay_padding
    occupied_slots = set(plan["occupied_slots"])
    filler = _filler_panel(slot_width, panel_height, scale)
    for slot in range(1, CHASSIS_SLOT_COUNT + 1):
        if slot in occupied_slots:
            continue
        paste_x = bay_left + (slot - 1) * slot_pitch
        image.paste(filler, (paste_x, paste_y), filler)

    for module in plan["modules"]:
        slot = int(module["slot"])
        paste_x = bay_left + (slot - 1) * slot_pitch
        with Image.open(BytesIO(module_assets[slot])) as panel:
            rgba_panel = panel.convert("RGBA")
            image.paste(rgba_panel, (paste_x, paste_y), rgba_panel)

    if highlighted_address is not None:
        if (
            not isinstance(highlighted_address, Sequence)
            or isinstance(
                highlighted_address,
                (str, bytes, bytearray),
            )
            or len(highlighted_address) != 2
        ):
            raise TypeError(
                "QCS highlighted address must contain slot and channel"
            )
        highlight_slot = _strict_positive_int(
            highlighted_address[0],
            "QCS highlighted slot",
        )
        highlight_channel = _strict_positive_int(
            highlighted_address[1],
            "QCS highlighted channel",
        )
        highlighted_module = next(
            (
                module
                for module in plan["modules"]
                if int(module["slot"]) == highlight_slot
            ),
            None,
        )
        geometry = (
            None
            if highlighted_module is None
            else _module_channel_geometry(
                highlighted_module,
                highlight_channel,
                slot_width=slot_width,
                panel_height=panel_height,
                scale=scale,
            )
        )
        if geometry is None:
            raise ValueError(
                "QCS highlighted address does not identify a channel SMA"
            )
        center_x, center_y, connector_radius = geometry
        role = next(
            (
                str(mapping["role"])
                for mapping in plan["mappings"]
                if int(mapping["slot"]) == highlight_slot
                and int(mapping["channel"]) == highlight_channel
            ),
            "unassigned",
        )
        ring_radius = max(_scaled(9, scale), round(connector_radius * 1.35))
        outer_width = max(2, _scaled(7, scale))
        inner_width = max(2, _scaled(4, scale))
        ring_box = (
            round(center_x - ring_radius),
            round(center_y - ring_radius),
            round(center_x + ring_radius),
            round(center_y + ring_radius),
        )
        draw.ellipse(
            ring_box,
            outline="#FFFFFF",
            width=outer_width,
        )
        inner_inset = max(1, _scaled(3, scale))
        draw.ellipse(
            (
                ring_box[0] + inner_inset,
                ring_box[1] + inner_inset,
                ring_box[2] - inner_inset,
                ring_box[3] - inner_inset,
            ),
            outline=ROLE_COLORS.get(role, ROLE_COLORS["unassigned"]),
            width=inner_width,
        )

    mapping_top = chassis_bottom + _scaled(18, scale)
    draw.text(
        (left_margin, mapping_top),
        "CHANNEL MAP + DOWNCONVERTERS - ADDRESS LABELS ONLY",
        font=mapping_title_font,
        fill=_DARK_TEXT,
    )
    y = mapping_top + mapping_title_height
    gap = _scaled(10, scale)
    chip_height = _scaled(22, scale)
    for row in mapping_rows:
        x = left_margin
        for label, role, chip_width in row:
            color = ROLE_COLORS[role]
            draw.rounded_rectangle(
                (x, y, x + chip_width, y + chip_height),
                radius=_scaled(7, scale),
                fill="#F7FAFC",
                outline=color,
                width=max(1, _scaled(2, scale)),
            )
            draw.text(
                (x + _scaled(9, scale), y + _scaled(3, scale)),
                label,
                font=mapping_font,
                fill=_DARK_TEXT,
            )
            x += chip_width + gap
        y += mapping_row_height

    if output_path is not None:
        destination = Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_encode_png(image))
    return image


def render_qcs_chassis_png(
    configuration: Mapping[str, Any],
    *,
    asset_directory: Optional[PathLike] = None,
    highlighted_address: Optional[tuple[int, int]] = None,
    slot_width: int = DEFAULT_SLOT_WIDTH,
    panel_height: int = DEFAULT_PANEL_HEIGHT,
    scale: int = DEFAULT_SCALE,
) -> bytes:
    """Return the rendered chassis as deterministic PNG bytes."""

    image = render_qcs_chassis(
        configuration,
        asset_directory=asset_directory,
        highlighted_address=highlighted_address,
        slot_width=slot_width,
        panel_height=panel_height,
        scale=scale,
    )
    return _encode_png(image)


def save_qcs_chassis_png(
    configuration: Mapping[str, Any],
    output_path: PathLike,
    *,
    asset_directory: Optional[PathLike] = None,
    slot_width: int = DEFAULT_SLOT_WIDTH,
    panel_height: int = DEFAULT_PANEL_HEIGHT,
    scale: int = DEFAULT_SCALE,
) -> Path:
    """Save a rendered chassis PNG and return its resolved destination."""

    destination = Path(output_path).expanduser()
    render_qcs_chassis(
        configuration,
        output_path=destination,
        asset_directory=asset_directory,
        slot_width=slot_width,
        panel_height=panel_height,
        scale=scale,
    )
    return destination.resolve()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a normalized QCS M9046A configuration as PNG."
    )
    parser.add_argument("configuration", type=Path, help="configuration JSON")
    parser.add_argument("output", type=Path, help="destination chassis PNG")
    parser.add_argument(
        "--asset-dir",
        type=Path,
        help="directory containing the four finalized panel PNGs",
    )
    parser.add_argument(
        "--panel-output-dir",
        type=Path,
        help="also save the slot-sized module composites",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE,
        help="integer render scale (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    configuration = json.loads(
        args.configuration.read_text(encoding="utf-8")
    )
    save_qcs_chassis_png(
        configuration,
        args.output,
        asset_directory=args.asset_dir,
        scale=args.scale,
    )
    if args.panel_output_dir is not None:
        generate_qcs_panel_png_assets(
            configuration,
            asset_directory=args.asset_dir,
            output_directory=args.panel_output_dir,
            scale=args.scale,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ASSET_DIRECTORY",
    "CHANNEL_CONNECTOR_LAYOUT",
    "CHASSIS_MODEL",
    "CHASSIS_SLOT_COUNT",
    "DEFAULT_CHASSIS_HEADER_HEIGHT",
    "DEFAULT_CHASSIS_LEFT_MARGIN",
    "DEFAULT_CHASSIS_RIGHT_MARGIN",
    "DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT",
    "DEFAULT_PANEL_HEIGHT",
    "DEFAULT_SCALE",
    "DEFAULT_SLOT_WIDTH",
    "MODULE_SPECS",
    "PANEL_ASSET_MODELS",
    "ROLE_SELECTABLE_MODELS",
    "generate_qcs_panel_png_assets",
    "load_qcs_panel_png_assets",
    "qcs_chassis_connector_at_point",
    "qcs_chassis_slot_at_point",
    "render_qcs_chassis",
    "render_qcs_chassis_png",
    "save_qcs_chassis_png",
]
