"""Render a compact QCS M5000 front-panel diagram using Pillow only.

The output matches the measured 1195 x 135 DC preview area (8.85:1), avoiding
the large letterbox margins of the previous 2400 x 1450 reference artwork.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


WIDTH = 2390
HEIGHT = 270

BACKGROUND = "#e8eef3"
WHITE = "#ffffff"
INK = "#162231"
MUTED = "#6e7e8d"
CHASSIS = "#363c41"
CHASSIS_EDGE = "#20262a"
SLOT_FACE = "#30363a"
SLOT_EDGE = "#aeb5ba"
EMPTY = "#41474c"
RED = "#e61e2a"
SYNC = "#4b8bd8"
RF = "#ff9d20"
BASEBAND_A = "#20a9b5"
BASEBAND_B = "#8d69d4"
DIGITIZER = "#4bb778"
GOLD = "#d6a83e"

FONT_REGULAR = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\segoeuib.ttf")


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    draw.text(
        (
            left + (right - left - text_width) / 2,
            top + (bottom - top - text_height) / 2 - bounds[1],
        ),
        text,
        font=text_font,
        fill=fill,
    )


def connector(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    label: str,
    accent: str,
    *,
    radius: int = 8,
) -> tuple[int, int]:
    draw.ellipse(
        (x - radius - 4, y - radius - 4, x + radius + 4, y + radius + 4),
        fill="#15191c",
        outline="#c7cdd1",
        width=1,
    )
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=GOLD,
        outline="#f7d77e",
        width=2,
    )
    draw.ellipse(
        (x - 5, y - 5, x + 5, y + 5),
        fill="#22272a",
        outline="#101417",
        width=2,
    )
    label_font = font(11, bold=True)
    draw.text(
        (x + radius + 4, y - 7),
        label,
        font=label_font,
        fill="#f0f2f3",
    )
    return round(x), round(y)


def small_sma(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    *,
    radius: int = 5,
) -> None:
    draw.ellipse(
        (x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2),
        fill="#1b2023",
        outline="#c6ccd0",
        width=1,
    )
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=GOLD,
        outline="#f0cf75",
        width=1,
    )
    draw.ellipse(
        (x - 2, y - 2, x + 2, y + 2),
        fill="#31363a",
    )


def slot_geometry(slot: int, span: int = 1) -> tuple[float, float, float, float]:
    rack_left = 42
    rack_right = WIDTH - 42
    pitch = (rack_right - rack_left) / 18
    gap = 6
    left = rack_left + (slot - 1) * pitch + gap / 2
    right = rack_left + (slot - 1 + span) * pitch - gap / 2
    return left, 91, right, 211


def draw_empty_slot(draw: ImageDraw.ImageDraw, slot: int) -> None:
    left, top, right, bottom = slot_geometry(slot)
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=5,
        fill=EMPTY,
        outline="#a3aaaf",
        width=2,
    )
    draw.line(
        (right - 6, top + 7, right - 6, bottom - 7),
        fill="#363c40",
        width=2,
    )
    for y in (top + 11, bottom - 11):
        draw.ellipse(
            (left + 8, y - 4, left + 16, y + 4),
            fill="#c7cccf",
            outline="#30363a",
            width=1,
        )
    draw.rectangle(
        (left + 18, top + 12, right - 14, top + 17),
        fill="#666d72",
    )
    draw.rectangle(
        (left + 18, bottom - 18, right - 14, bottom - 13),
        fill="#454b4f",
    )


def draw_module_face(
    draw: ImageDraw.ImageDraw,
    *,
    slot: int,
    span: int,
    model: str,
    subtitle: str,
    accent: str,
) -> tuple[float, float, float, float]:
    left, top, right, bottom = slot_geometry(slot, span)
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=7,
        fill=SLOT_FACE,
        outline="#728089",
        width=2,
    )
    draw.rectangle((right - 7, top, right, bottom), fill=RED)
    draw.rectangle((left, top, right - 7, top + 5), fill="#c3c8cb")
    draw.text(
        (left + 13, top + 7),
        "KEYSIGHT",
        font=font(9, bold=True),
        fill="#f0f2f3",
    )
    draw.text(
        (left + 13, top + 20),
        model,
        font=font(17, bold=True),
        fill="#f3f6f8",
    )
    draw.text(
        (left + 13, top + 39),
        subtitle,
        font=font(10),
        fill="#aeb9c0",
    )
    handle_width = min(74, max(48, (right - left) * 0.45))
    handle_left = left + (right - left - handle_width) / 2
    draw.rounded_rectangle(
        (handle_left, bottom - 13, handle_left + handle_width, bottom + 9),
        radius=5,
        fill="#15191c",
        outline="#050708",
        width=2,
    )
    draw.text(
        (handle_left + 8, bottom - 11),
        "PXI",
        font=font(12, bold=True),
        fill="#d9dde0",
    )
    return left, top, right, bottom


def draw_sync(draw: ImageDraw.ImageDraw, slot: int) -> None:
    left, top, right, bottom = draw_module_face(
        draw,
        slot=slot,
        span=1,
        model="M9032A",
        subtitle="S1  ·  SYSTEM SYNC",
        accent=SYNC,
    )
    cx = (left + right) / 2
    connector(draw, cx - 25, top + 83, "IN", SYNC, radius=8)
    connector(draw, cx + 25, top + 83, "OUT", SYNC, radius=8)


def draw_four_channel_module(
    draw: ImageDraw.ImageDraw,
    *,
    slot: int,
    span: int,
    model: str,
    subtitle: str,
    accent: str,
) -> dict[int, tuple[int, int]]:
    left, top, right, bottom = draw_module_face(
        draw,
        slot=slot,
        span=span,
        model=model,
        subtitle=subtitle,
        accent=accent,
    )
    result: dict[int, tuple[int, int]] = {}
    if span == 1:
        x_positions = [left + 36] * 4
        y_positions = [top + 54, top + 69, top + 84, top + 99]
        radius = 6
    else:
        x_positions = [
            left + 49,
            left + 101,
            left + 49,
            left + 101,
        ]
        y_positions = [
            top + 66,
            top + 66,
            top + 95,
            top + 95,
        ]
        radius = 8
    for channel, (x, y) in enumerate(zip(x_positions, y_positions), start=1):
        result[channel] = connector(
            draw,
            x,
            y,
            str(channel),
            accent,
            radius=radius,
        )
    trigger_columns = 1 if span == 1 else 2
    trigger_left = left + (right - left) * (0.73 if span == 1 else 0.69)
    trigger_step = 23 if span == 2 else 0
    for column in range(trigger_columns):
        for row in range(4):
            small_sma(
                draw,
                trigger_left + column * trigger_step,
                top + 55 + row * 15,
                radius=4,
            )
    port_left = left + (right - left) * (0.64 if span == 1 else 0.80)
    draw.rounded_rectangle(
        (port_left, top + 15, port_left + 13, top + 42),
        radius=2,
        fill="#15191c",
        outline="#aab0b4",
        width=1,
    )
    return result


def draw_legend_item(
    draw: ImageDraw.ImageDraw,
    x: int,
    color: str,
    text: str,
) -> int:
    draw.rounded_rectangle((x, 240, x + 17, 257), radius=5, fill=color)
    legend_font = font(15, bold=True)
    draw.text((x + 25, 237), text, font=legend_font, fill=INK)
    bounds = draw.textbbox((0, 0), text, font=legend_font)
    return x + 25 + (bounds[2] - bounds[0]) + 42


def render(output_path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (18, 8, WIDTH - 18, 47),
        radius=11,
        fill=WHITE,
        outline="#c5d0d9",
        width=2,
    )
    draw.rectangle((31, 14, 39, 41), fill=RED)
    draw.text(
        (53, 10),
        "Keysight QCS  ·  M9046A hardware view",
        font=font(25, bold=True),
        fill=INK,
    )
    badge = (WIDTH - 394, 14, WIDTH - 31, 41)
    draw.rounded_rectangle(badge, radius=15, fill="#35424c")
    centered_text(
        draw,
        badge,
        "PYTHON-DRAWN FRONT VIEW  |  NO CABLES",
        font(14, bold=True),
        WHITE,
    )

    draw.rounded_rectangle(
        (18, 54, WIDTH - 18, 231),
        radius=15,
        fill=CHASSIS,
        outline=CHASSIS_EDGE,
        width=4,
    )
    draw.rounded_rectangle(
        (34, 62, WIDTH - 34, 88),
        radius=6,
        fill="#30363a",
        outline="#a9b0b5",
        width=2,
    )
    for x in range(48, WIDTH - 48, 24):
        draw.ellipse((x, 67, x + 5, 72), fill="#161b1f")
        draw.ellipse((x + 11, 77, x + 16, 82), fill="#161b1f")
    draw.text(
        (47, 65),
        "KEYSIGHT   PXI",
        font=font(12, bold=True),
        fill="#dbe2e7",
        stroke_width=2,
        stroke_fill="#30363a",
    )
    draw.text(
        (186, 66),
        "M9046A HIGH-POWER PXIe CHASSIS",
        font=font(11, bold=True),
        fill="#eef1f3",
        stroke_width=2,
        stroke_fill="#30363a",
    )
    draw.rectangle((35, 86, WIDTH - 35, 90), fill=RED)
    draw.rectangle((35, 211, WIDTH - 35, 215), fill=RED)

    occupied = {4, 5, 7, 18}
    for slot in range(1, 19):
        if slot not in occupied:
            draw_empty_slot(draw, slot)

    for slot in range(1, 19):
        left, _, right, _ = slot_geometry(slot)
        centered_text(
            draw,
            (left, 72, right, 89),
            str(slot),
            font(10, bold=True),
            "#f2f4f5",
        )

    draw_four_channel_module(
        draw,
        slot=4,
        span=2,
        model="M5300A",
        subtitle="RF AWG  ·  S4–5",
        accent=RED,
    )
    draw_four_channel_module(
        draw,
        slot=7,
        span=1,
        model="M5301A",
        subtitle="BASEBAND  ·  S7",
        accent=RED,
    )
    draw_four_channel_module(
        draw,
        slot=18,
        span=1,
        model="M5200A",
        subtitle="DIGITIZER  ·  S18",
        accent=RED,
    )

    # The real chassis has a lower status/control rail below the module handles.
    draw.rounded_rectangle(
        (38, 216, WIDTH - 38, 227),
        radius=3,
        fill="#3d4348",
        outline="#959da2",
        width=1,
    )
    draw.rounded_rectangle((51, 217, 77, 226), radius=3, fill="#eef1f2")
    for x, color in ((93, DIGITIZER), (111, RF), (129, "#e4d14a")):
        draw.ellipse((x, 219, x + 6, 225), fill=color)
    draw.text((142, 217), "TEMP   FAN   POWER", font=font(8), fill="#e0e4e6")
    for x in range(731, 810, 19):
        small_sma(draw, x, 222, radius=3)
    draw.text(
        (820, 217),
        "10 MHz REF / TRIGGER DISTRIBUTION",
        font=font(8, bold=True),
        fill="#e4e7e9",
    )

    x = 34
    x = draw_legend_item(draw, x, RED, "S4–5  M5300A RF AWG")
    x = draw_legend_item(draw, x, "#879198", "S7  M5301A BASEBAND AWG")
    x = draw_legend_item(draw, x, "#879198", "S18  M5200A DIGITIZER")
    draw_legend_item(draw, x, "#4d555b", "HARDWARE ONLY  ·  NO ROUTING IMPLIED")
    draw.text(
        (WIDTH - 191, 254),
        "PYTHON / PILLOW RENDER",
        font=font(10, bold=True),
        fill=MUTED,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a compact QCS M5000 front-panel PNG."
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("output/qcs_m5000_alternate_python.png"),
        help="PNG output path",
    )
    return parser.parse_args(arguments)


if __name__ == "__main__":
    args = parse_args()
    render(args.output)
