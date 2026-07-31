"""Draw a standalone Keysight-style M5300A front panel with Pillow."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# M5300A occupies two chassis slots, exactly twice the 300 px single-slot
# M5301A/M5200A faceplate width.
WIDTH = 600
HEIGHT = 1300

FONT_REGULAR = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\segoeuib.ttf")

CHARCOAL = "#5a5c5f"
CHARCOAL_LIGHT = "#74777a"
SILVER = "#bec5ca"
WHITE = "#f4f6f7"
MUTED = "#b7c0c6"
RED = "#e51d2a"
GOLD = "#d0a13a"
GOLD_LIGHT = "#f2d47a"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        str(FONT_BOLD if bold else FONT_REGULAR),
        size=size,
    )


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (
            left + (right - left - width) / 2,
            top + (bottom - top - height) / 2 - bounds[1],
        ),
        text,
        font=text_font,
        fill=fill,
    )


def sma_connector(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    *,
    radius: int,
) -> None:
    draw.regular_polygon(
        (x, y, radius + 13),
        n_sides=6,
        rotation=30,
        fill="#b1842f",
        outline=GOLD_LIGHT,
    )
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=GOLD,
        outline=GOLD_LIGHT,
        width=4,
    )
    draw.ellipse(
        (x - radius + 9, y - radius + 9, x + radius - 9, y + radius - 9),
        fill="#252a2d",
        outline="#0d1012",
        width=3,
    )
    draw.ellipse(
        (x - 5, y - 5, x + 5, y + 5),
        fill="#d9b759",
    )


def channel_connector(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    channel: int,
) -> None:
    sma_connector(draw, x, y, radius=38)
    centered_text(
        draw,
        (x - 60, y + 52, x + 60, y + 88),
        f"CH{channel}",
        font(25, bold=True),
        WHITE,
    )


def smp_connector(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    number: int,
) -> None:
    radius = 24
    draw.ellipse(
        (x - radius - 6, y - radius - 6, x + radius + 6, y + radius + 6),
        fill="#23282b",
        outline="#d6dadd",
        width=2,
    )
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=GOLD,
        outline=GOLD_LIGHT,
        width=4,
    )
    draw.ellipse(
        (x - 13, y - 13, x + 13, y + 13),
        fill="#262b2e",
        outline="#111416",
        width=3,
    )
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=GOLD_LIGHT)
    centered_text(
        draw,
        (x - 68, y + 36, x + 68, y + 70),
        f"SMP {number}",
        font(22, bold=True),
        WHITE,
    )


def render(output_path: Path) -> None:
    image = Image.new("RGBA", (WIDTH, HEIGHT), CHARCOAL)
    draw = ImageDraw.Draw(image)

    # A square-corner faceplate fills the complete asset.  This lets the
    # chassis compositor place it edge-to-edge without exposing the black bay.
    draw.rectangle(
        (0, 0, WIDTH - 1, HEIGHT - 1),
        fill=CHARCOAL,
        outline=SILVER,
        width=6,
    )
    draw.rectangle((16, 18, WIDTH - 16, 34), fill=RED)
    draw.rectangle((16, HEIGHT - 34, WIDTH - 16, HEIGHT - 18), fill=RED)
    draw.rectangle((WIDTH - 16, 18, WIDTH - 1, HEIGHT - 18), fill=RED)
    draw.line((36, 48, 36, HEIGHT - 48), fill=CHARCOAL_LIGHT, width=3)
    draw.line(
        (WIDTH - 36, 48, WIDTH - 36, HEIGHT - 48),
        fill="#2d3134",
        width=4,
    )

    centered_text(
        draw,
        (77, 86, WIDTH - 77, 153),
        "M5300A",
        font(48, bold=True),
        WHITE,
    )
    draw.line((92, 174, WIDTH - 92, 174), fill="#92999e", width=3)

    # Exact connector order requested by the user:
    # SMA CH1/CH2, SMP 1/3, SMP 2/4, SMA CH3/CH4,
    # SMP 5/7, SMP 6/8.
    left_x = 185
    right_x = WIDTH - left_x
    channel_connector(draw, left_x, 285, 1)
    channel_connector(draw, right_x, 285, 2)

    smp_connector(draw, left_x, 465, 1)
    smp_connector(draw, right_x, 465, 3)

    smp_connector(draw, left_x, 610, 2)
    smp_connector(draw, right_x, 610, 4)

    channel_connector(draw, left_x, 805, 3)
    channel_connector(draw, right_x, 805, 4)

    smp_connector(draw, left_x, 985, 5)
    smp_connector(draw, right_x, 985, 7)

    smp_connector(draw, left_x, 1130, 6)
    smp_connector(draw, right_x, 1130, 8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a standalone Python-drawn M5300A PNG."
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("output/M5300A_python_front_panel.png"),
    )
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
