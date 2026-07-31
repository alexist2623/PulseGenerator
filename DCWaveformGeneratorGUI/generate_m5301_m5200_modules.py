"""Generate simplified M5301A, M5200A, and M5201A front panels."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from generate_m5300_module import (
    CHARCOAL,
    CHARCOAL_LIGHT,
    GOLD_LIGHT,
    RED,
    SILVER,
    WHITE,
    centered_text,
    channel_connector,
    font,
    sma_connector,
    smp_connector,
)


PANEL_WIDTH = 300
PANEL_HEIGHT = 1300


def _panel_faceplate(model: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new(
        "RGBA",
        (PANEL_WIDTH, PANEL_HEIGHT),
        CHARCOAL,
    )
    draw = ImageDraw.Draw(image)

    draw.rectangle(
        (0, 0, PANEL_WIDTH - 1, PANEL_HEIGHT - 1),
        fill=CHARCOAL,
        outline=SILVER,
        width=6,
    )
    draw.rectangle((16, 18, PANEL_WIDTH - 16, 34), fill=RED)
    draw.rectangle(
        (16, PANEL_HEIGHT - 34, PANEL_WIDTH - 16, PANEL_HEIGHT - 18),
        fill=RED,
    )
    draw.rectangle(
        (PANEL_WIDTH - 16, 18, PANEL_WIDTH - 1, PANEL_HEIGHT - 18),
        fill=RED,
    )
    draw.line(
        (36, 48, 36, PANEL_HEIGHT - 48),
        fill=CHARCOAL_LIGHT,
        width=3,
    )
    draw.line(
        (
            PANEL_WIDTH - 36,
            48,
            PANEL_WIDTH - 36,
            PANEL_HEIGHT - 48,
        ),
        fill="#2d3134",
        width=4,
    )

    centered_text(
        draw,
        (43, 86, PANEL_WIDTH - 43, 153),
        model,
        font(42, bold=True),
        WHITE,
    )
    draw.line(
        (58, 174, PANEL_WIDTH - 58, 174),
        fill="#92999e",
        width=3,
    )

    return image, draw


def render_panel(model: str) -> Image.Image:
    image, draw = _panel_faceplate(model)

    channel_x = 95
    smp_x = PANEL_WIDTH - channel_x
    row_positions = (300, 560, 820, 1080)
    for number, y in enumerate(row_positions, start=1):
        channel_connector(draw, channel_x, y, number)
        smp_connector(draw, smp_x, y, number)

    return image


def _named_sma_connector(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    *,
    radius: int = 35,
) -> None:
    sma_connector(draw, x, y, radius=radius)
    centered_text(
        draw,
        (x - 57, y + radius + 10, x + 57, y + radius + 42),
        label,
        font(20, bold=True),
        WHITE,
    )


def _clock_connector(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    """Draw the smaller round 2.4 GHz snap-on clock connector."""

    radius = 25
    draw.ellipse(
        (x - radius - 5, y - radius - 5, x + radius + 5, y + radius + 5),
        fill="#252a2d",
        outline="#d6dadd",
        width=2,
    )
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill="#b1842f",
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


def render_m5201_panel() -> Image.Image:
    """Draw the four staggered RF-input/IF-output pairs of an M5201A."""

    image, draw = _panel_faceplate("M5201A")
    rf_x = 92
    if_x = PANEL_WIDTH - 92
    rf_positions = (330, 540, 750, 960)
    if_positions = (245, 455, 665, 875)
    for number, (rf_y, if_y) in enumerate(
        zip(rf_positions, if_positions),
        start=1,
    ):
        _named_sma_connector(draw, rf_x, rf_y, f"RF {number}")
        _named_sma_connector(draw, if_x, if_y, f"IF {number}")

    for y, direction in ((1090, "IN"), (1195, "OUT")):
        _clock_connector(draw, if_x, y)
        centered_text(
            draw,
            (42, y - 25, 155, y + 25),
            f"2.4 GHz {direction}",
            font(17, bold=True),
            GOLD_LIGHT,
        )
    return image


def save_outputs(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    m5301 = render_panel("M5301A")
    m5200 = render_panel("M5200A")
    m5201 = render_m5201_panel()

    m5301_path = output_dir / "M5301A_python_front_panel.png"
    m5200_path = output_dir / "M5200A_python_front_panel.png"
    m5201_path = output_dir / "M5201A_python_front_panel.png"
    combined_path = output_dir / "M5301A_M5200A_python_front_panels.png"

    m5301.save(m5301_path, format="PNG", optimize=True)
    m5200.save(m5200_path, format="PNG", optimize=True)
    m5201.save(m5201_path, format="PNG", optimize=True)

    combined = Image.new(
        "RGBA",
        (PANEL_WIDTH * 2 + 60, PANEL_HEIGHT),
        (0, 0, 0, 0),
    )
    combined.alpha_composite(m5301, (0, 0))
    combined.alpha_composite(m5200, (PANEL_WIDTH + 60, 0))
    combined.save(combined_path, format="PNG", optimize=True)

    return m5301_path, m5200_path, m5201_path, combined_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate simplified M5301A, M5200A, and M5201A PNGs."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("output"),
    )
    args = parser.parse_args()
    save_outputs(args.output_dir)


if __name__ == "__main__":
    main()
