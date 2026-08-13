"""Tests for the standalone Pillow QCS chassis renderer."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw, ImageFont
import pytest

import qcs_chassis_renderer as renderer
from qcs_chassis_renderer import (
    ASSET_DIRECTORY,
    DEFAULT_CHASSIS_HEADER_HEIGHT,
    DEFAULT_CHASSIS_LEFT_MARGIN,
    DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT,
    DEFAULT_PANEL_HEIGHT,
    DEFAULT_SLOT_WIDTH,
    ROLE_COLORS,
    generate_qcs_panel_png_assets,
    load_qcs_panel_png_assets,
    qcs_chassis_connector_at_point,
    qcs_chassis_slot_at_point,
    render_qcs_chassis,
    render_qcs_chassis_png,
    save_qcs_chassis_png,
)


def _configuration() -> dict:
    return {
        "version": 1,
        "chassis_model": "M9046A",
        "chassis": 1,
        "host_controller": 1,
        "ip_address": None,
        "modules": [
            {"slot": 1, "model": "M9032A"},
            {"slot": 2, "model": "M5301A"},
            {"slot": 3, "model": "M5300A"},
            {"slot": 5, "model": "M5200A"},
            {"slot": 6, "model": "M5201A"},
        ],
        "channel_mappings": [
            {
                "role": "dc",
                "logical_index": 0,
                "virtual_name": "dc_left",
                "label": 0,
                "absolute_phase": False,
                "lo_frequency_hz": None,
                "slot": 2,
                "channel": 1,
            },
            {
                "role": "rf",
                "logical_index": 0,
                "virtual_name": "rf_drive",
                "label": 0,
                "absolute_phase": True,
                "lo_frequency_hz": 6.25e9,
                "slot": 3,
                "channel": 1,
            },
            {
                "role": "acquisition",
                "logical_index": 0,
                "virtual_name": "digitizer",
                "label": 0,
                "absolute_phase": True,
                "lo_frequency_hz": None,
                "slot": 5,
                "channel": 1,
            },
        ],
        "downconverter_links": [
            {
                "digitizer_slot": 5,
                "digitizer_channel": 1,
                "downconverter_slot": 6,
                "downconverter_channel": 2,
                "lo_frequency_hz": 7.25e9,
            }
        ],
    }


def test_mapping_footer_reports_m5300_lo_frequency():
    configuration = _configuration()
    image = Image.new("RGB", (1, 1))
    rows = renderer._mapping_rows(
        ImageDraw.Draw(image),
        configuration["modules"],
        configuration["channel_mappings"],
        configuration["downconverter_links"],
        content_width=10000,
        font=ImageFont.load_default(),
        scale=1,
    )
    labels = [label for row in rows for label, _role, _width in row]

    assert any(
        "rf_drive" in label and "LO 6.25 GHz" in label
        for label in labels
    )
    assert all(
        "LO" not in label
        for label in labels
        if "dc_left" in label
    )


def test_finalized_physical_panel_assets_are_loaded_as_exact_png_bytes():
    assets = load_qcs_panel_png_assets()

    assert set(assets) == {
        "M5300A",
        "M5301A",
        "M5200A",
        "M5201A",
    }
    for model, expected_size in {
        "M5300A": (600, 1300),
        "M5301A": (300, 1300),
        "M5200A": (300, 1300),
        "M5201A": (300, 1300),
    }.items():
        asset_path = ASSET_DIRECTORY / f"{model}_python_front_panel.png"
        assert assets[model] == asset_path.read_bytes()
        with Image.open(BytesIO(assets[model])) as panel:
            assert panel.size == expected_size
            assert panel.mode == "RGBA"
            assert all(
                panel.getpixel(point)[3] == 255
                for point in (
                    (0, 0),
                    (panel.width - 1, 0),
                    (0, panel.height - 1),
                    (panel.width - 1, panel.height - 1),
                )
            )


def test_generated_slot_composites_preserve_module_spans(tmp_path):
    assets = generate_qcs_panel_png_assets(
        _configuration(),
        output_directory=tmp_path,
    )

    assert set(assets) == {1, 2, 3, 5, 6}
    assert all(asset.startswith(b"\x89PNG\r\n\x1a\n") for asset in assets.values())
    with Image.open(BytesIO(assets[2])) as single_slot:
        assert single_slot.size == (101, 438)
        assert single_slot.getpixel((0, 0))[3] == 255
    with Image.open(BytesIO(assets[3])) as double_slot:
        assert double_slot.size == (202, 438)
        assert double_slot.getpixel((0, 0))[3] == 255
    with Image.open(BytesIO(assets[6])) as m5201_single_slot:
        assert m5201_single_slot.size == (101, 438)
        assert m5201_single_slot.getpixel((0, 0))[3] == 255
    assert (tmp_path / "slot_02_M5301A.png").is_file()
    assert (tmp_path / "slot_03_M5300A.png").is_file()
    assert (tmp_path / "slot_05_M5200A.png").is_file()
    assert (tmp_path / "slot_06_M5201A.png").is_file()


def test_module_composite_cache_tracks_asset_bytes_and_still_writes_files(
    tmp_path,
):
    source_assets = load_qcs_panel_png_assets()
    asset_directory = tmp_path / "assets"
    asset_directory.mkdir()
    for model, png_bytes in source_assets.items():
        (asset_directory / f"{model}_python_front_panel.png").write_bytes(
            png_bytes
        )

    renderer._cached_qcs_module_panel_png.cache_clear()
    first = generate_qcs_panel_png_assets(
        _configuration(),
        asset_directory=asset_directory,
    )
    output_directory = tmp_path / "cached-output"
    second = generate_qcs_panel_png_assets(
        _configuration(),
        asset_directory=asset_directory,
        output_directory=output_directory,
    )

    assert second == first
    assert renderer._cached_qcs_module_panel_png.cache_info().misses == 5
    assert renderer._cached_qcs_module_panel_png.cache_info().hits == 5
    assert (output_directory / "slot_02_M5301A.png").read_bytes() == first[2]

    # A changed source PNG must invalidate only that module composite even
    # when its path and the rest of the chassis topology remain unchanged.
    (asset_directory / "M5301A_python_front_panel.png").write_bytes(
        source_assets["M5200A"]
    )
    changed = generate_qcs_panel_png_assets(
        _configuration(),
        asset_directory=asset_directory,
    )

    assert changed[2] != first[2]
    assert all(changed[slot] == first[slot] for slot in (1, 3, 5, 6))
    cache_info = renderer._cached_qcs_module_panel_png.cache_info()
    assert cache_info.misses == 6
    assert cache_info.hits == 9
    renderer._cached_qcs_module_panel_png.cache_clear()


def test_chassis_render_is_deterministic_and_savable(tmp_path):
    first = render_qcs_chassis_png(_configuration())
    second = render_qcs_chassis_png(_configuration())
    assert first == second

    with Image.open(BytesIO(first)) as rendered:
        assert rendered.mode == "RGB"
        assert rendered.size == (1914, 652)
        rendered.load()

    output = tmp_path / "front-panel.png"
    returned = save_qcs_chassis_png(_configuration(), output)
    assert returned == output.resolve()
    assert output.read_bytes() == first

    direct_output = tmp_path / "direct.png"
    image = render_qcs_chassis(
        _configuration(),
        output_path=direct_output,
    )
    assert image.mode == "RGB"
    assert direct_output.read_bytes() == first


def test_channel_sma_hit_test_resolves_role_compatible_connectors():
    configuration = _configuration()
    bay_top = (
        DEFAULT_CHASSIS_HEADER_HEIGHT
        + DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT
    )

    m5301_left = DEFAULT_CHASSIS_LEFT_MARGIN + DEFAULT_SLOT_WIDTH
    dc_ch1 = (
        m5301_left + 95 * DEFAULT_SLOT_WIDTH / 300,
        bay_top + 300 * DEFAULT_PANEL_HEIGHT / 1300,
    )
    assert qcs_chassis_connector_at_point(
        configuration,
        *dc_ch1,
        role="dc",
    ) == {
        "slot": 2,
        "channel": 1,
        "model": "M5301A",
        "center": pytest.approx(dc_ch1),
    }
    assert (
        qcs_chassis_connector_at_point(
            configuration,
            *dc_ch1,
            role="acquisition",
        )
        is None
    )

    m5300_left = (
        DEFAULT_CHASSIS_LEFT_MARGIN + 2 * DEFAULT_SLOT_WIDTH
    )
    rf_ch4 = (
        m5300_left + 415 * (2 * DEFAULT_SLOT_WIDTH) / 600,
        bay_top + 805 * DEFAULT_PANEL_HEIGHT / 1300,
    )
    rf_hit = qcs_chassis_connector_at_point(
        configuration,
        *rf_ch4,
        role="rf",
    )
    assert (rf_hit["slot"], rf_hit["channel"], rf_hit["model"]) == (
        3,
        4,
        "M5300A",
    )

    m5200_left = (
        DEFAULT_CHASSIS_LEFT_MARGIN + 4 * DEFAULT_SLOT_WIDTH
    )
    acquisition_ch3 = (
        m5200_left + 95 * DEFAULT_SLOT_WIDTH / 300,
        bay_top + 820 * DEFAULT_PANEL_HEIGHT / 1300,
    )
    acquisition_hit = qcs_chassis_connector_at_point(
        configuration,
        *acquisition_ch3,
        role="acquisition",
    )
    assert (
        acquisition_hit["slot"],
        acquisition_hit["channel"],
        acquisition_hit["model"],
    ) == (5, 3, "M5200A")

    m5201_left = (
        DEFAULT_CHASSIS_LEFT_MARGIN + 5 * DEFAULT_SLOT_WIDTH
    )
    downconverter_ch3 = (
        m5201_left + 92 * DEFAULT_SLOT_WIDTH / 300,
        bay_top + 750 * DEFAULT_PANEL_HEIGHT / 1300,
    )
    downconverter_hit = qcs_chassis_connector_at_point(
        configuration,
        *downconverter_ch3,
        role="downconverter",
    )
    assert downconverter_hit == {
        "slot": 6,
        "channel": 3,
        "model": "M5201A",
        "center": pytest.approx(downconverter_ch3),
    }
    assert (
        qcs_chassis_connector_at_point(
            configuration,
            *downconverter_ch3,
            role="acquisition",
        )
        is None
    )
    assert (
        qcs_chassis_connector_at_point(
            configuration,
            *acquisition_ch3,
            role="downconverter",
        )
        is None
    )

    # M5301 SMP1 shares CH1's y coordinate but is not a physical channel.
    m5301_smp1 = (
        m5301_left + 205 * DEFAULT_SLOT_WIDTH / 300,
        dc_ch1[1],
    )
    assert (
        qcs_chassis_connector_at_point(
            configuration,
            *m5301_smp1,
        )
        is None
    )


def test_highlighted_channel_sma_changes_only_transient_rendering():
    configuration = _configuration()
    plain = render_qcs_chassis_png(configuration)
    highlighted = render_qcs_chassis_png(
        configuration,
        highlighted_address=(2, 1),
    )

    assert highlighted != plain
    assert render_qcs_chassis_png(configuration) == plain
    with Image.open(BytesIO(highlighted)) as rendered:
        rendered.load()
        assert rendered.size == (1914, 652)


def test_multiple_channel_smas_are_highlighted_in_one_render():
    configuration = _configuration()
    plain = render_qcs_chassis_png(configuration)
    rf_only = render_qcs_chassis_png(
        configuration,
        highlighted_address=(3, 1),
    )
    acquisition_only = render_qcs_chassis_png(
        configuration,
        highlighted_address=(5, 1),
    )
    both = render_qcs_chassis_png(
        configuration,
        highlighted_addresses=((3, 1), (5, 1)),
    )

    assert both not in {plain, rf_only, acquisition_only}
    assert render_qcs_chassis_png(
        configuration,
        highlighted_addresses=((3, 1), (5, 1), (3, 1)),
    ) == both


def test_downconverter_link_is_rendered_in_the_address_footer():
    configuration = _configuration()
    linked_png = render_qcs_chassis_png(configuration)

    without_link = dict(configuration)
    without_link["downconverter_links"] = []
    unlinked_png = render_qcs_chassis_png(without_link)

    assert linked_png != unlinked_png
    footer_top = (
        DEFAULT_CHASSIS_HEADER_HEIGHT
        + DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT
        + DEFAULT_PANEL_HEIGHT
    )
    link_color = tuple(
        int(ROLE_COLORS["downconverter"][offset : offset + 2], 16)
        for offset in (1, 3, 5)
    )
    with Image.open(BytesIO(linked_png)) as linked:
        footer = linked.crop((0, footer_top, linked.width, linked.height))
        footer_colors = {
            color
            for _count, color in footer.getcolors(
                maxcolors=footer.width * footer.height
            )
        }
        assert link_color in footer_colors
    with Image.open(BytesIO(unlinked_png)) as unlinked:
        footer = unlinked.crop(
            (0, footer_top, unlinked.width, unlinked.height)
        )
        footer_colors = {
            color
            for _count, color in footer.getcolors(
                maxcolors=footer.width * footer.height
            )
        }
        assert link_color not in footer_colors


def test_chassis_render_rejects_overlapping_two_slot_module():
    configuration = _configuration()
    configuration["modules"].append({"slot": 4, "model": "M5301A"})

    with pytest.raises(ValueError, match="slot 4 is occupied"):
        render_qcs_chassis(configuration)


def test_chassis_slot_hit_test_resolves_all_default_slots():
    label_y = DEFAULT_CHASSIS_HEADER_HEIGHT
    module_y = (
        DEFAULT_CHASSIS_HEADER_HEIGHT
        + DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT
    )

    for slot in range(1, 19):
        left = (
            DEFAULT_CHASSIS_LEFT_MARGIN
            + (slot - 1) * DEFAULT_SLOT_WIDTH
        )
        assert qcs_chassis_slot_at_point(left, label_y) == slot
        assert (
            qcs_chassis_slot_at_point(
                left + DEFAULT_SLOT_WIDTH - 1,
                module_y + DEFAULT_PANEL_HEIGHT - 1,
            )
            == slot
        )


@pytest.mark.parametrize(
    ("x", "y"),
    (
        (DEFAULT_CHASSIS_LEFT_MARGIN, DEFAULT_CHASSIS_HEADER_HEIGHT - 1),
        (DEFAULT_CHASSIS_LEFT_MARGIN - 1, DEFAULT_CHASSIS_HEADER_HEIGHT),
        (
            DEFAULT_CHASSIS_LEFT_MARGIN
            + 18 * DEFAULT_SLOT_WIDTH,
            DEFAULT_CHASSIS_HEADER_HEIGHT,
        ),
        (
            DEFAULT_CHASSIS_LEFT_MARGIN,
            DEFAULT_CHASSIS_HEADER_HEIGHT
            + DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT
            + DEFAULT_PANEL_HEIGHT,
        ),
        (0, 0),
    ),
)
def test_chassis_slot_hit_test_rejects_header_footer_and_outside(x, y):
    assert qcs_chassis_slot_at_point(x, y) is None


def test_chassis_slot_hit_test_uses_custom_render_geometry():
    slot_width = 17
    panel_height = 29
    scale = 3
    left = DEFAULT_CHASSIS_LEFT_MARGIN * scale
    top = DEFAULT_CHASSIS_HEADER_HEIGHT * scale
    pitch = slot_width * scale
    bottom = (
        DEFAULT_CHASSIS_HEADER_HEIGHT
        + DEFAULT_CHASSIS_SLOT_LABEL_HEIGHT
        + panel_height
    ) * scale

    assert qcs_chassis_slot_at_point(
        left + 6 * pitch,
        top,
        slot_width=slot_width,
        panel_height=panel_height,
        scale=scale,
    ) == 7
    assert qcs_chassis_slot_at_point(
        left + 7 * pitch - 1,
        bottom - 1,
        slot_width=slot_width,
        panel_height=panel_height,
        scale=scale,
    ) == 7
    assert qcs_chassis_slot_at_point(
        left + 7 * pitch,
        top,
        slot_width=slot_width,
        panel_height=panel_height,
        scale=scale,
    ) == 8
    assert qcs_chassis_slot_at_point(
        left,
        top - 1,
        slot_width=slot_width,
        panel_height=panel_height,
        scale=scale,
    ) is None
    assert qcs_chassis_slot_at_point(
        left,
        bottom,
        slot_width=slot_width,
        panel_height=panel_height,
        scale=scale,
    ) is None
