"""HWH-backed ZCU216 front-panel discovery and interactive channel selection.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets


OUTPUT_BOARD_TYPES = ("RF_Out", "DC_Out")
INPUT_BOARD_TYPES = ("RF_In", "DC_In")


@dataclass(frozen=True)
class QickFrontPanelPort:
    """One physical front-panel SMA and its HWH/runtime channel mapping."""

    direction: str
    panel_index: int
    converter_id: str
    board_slot: int
    board_type: Optional[str]
    qick_channels: Tuple[int, ...]
    block_paths: Tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{'DAC' if self.direction == 'output' else 'ADC'}{self.panel_index}"

    @property
    def channel_label(self) -> str:
        noun = "generator" if self.direction == "output" else "readout"
        if not self.qick_channels:
            return f"No QICK {noun}"
        return ", ".join(f"{noun} {channel}" for channel in self.qick_channels)

    @property
    def board_label(self) -> str:
        if self.board_type is None:
            return "No daughter card detected"
        return self.board_type.replace("_", " ")


@dataclass(frozen=True)
class QickFrontPanelConfiguration:
    """Physical QICK-box port map reconstructed from a live QICK config."""

    board: str
    firmware_timestamp: str
    outputs: Tuple[QickFrontPanelPort, ...]
    inputs: Tuple[QickFrontPanelPort, ...]

    def port(self, direction: str, panel_index: int) -> QickFrontPanelPort:
        ports = self.outputs if direction == "output" else self.inputs
        if not 0 <= int(panel_index) < len(ports):
            raise IndexError(f"unknown {direction} front-panel port {panel_index}")
        return ports[int(panel_index)]

    @property
    def mapped_output_count(self) -> int:
        return sum(bool(port.qick_channels) for port in self.outputs)

    @property
    def mapped_input_count(self) -> int:
        return sum(bool(port.qick_channels) for port in self.inputs)


_CARD_RE = re.compile(
    r"\b(DAC|ADC)\s+slot\s+(\d+)\s*:\s*"
    r"(No\s+card\s+detected|(?:RF|DC)\s+(?:Out|In))"
    r"(?:\s+card\s+has\s+ports\s+\[([^\]]*)\])?",
    re.IGNORECASE,
)


def _config_mapping(soccfg: Any) -> Mapping[str, Any]:
    if hasattr(soccfg, "get_cfg"):
        config = soccfg.get_cfg()
    elif isinstance(soccfg, Mapping):
        config = soccfg
    elif hasattr(soccfg, "_cfg"):
        config = soccfg._cfg
    else:
        raise TypeError("soccfg must be a QickConfig or mapping")
    if not isinstance(config, Mapping):
        raise TypeError("QICK get_cfg() did not return a mapping")
    return config


def _converter_pair(value: Any, label: str) -> Tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"{label} converter identifier must have two fields")
        return int(value[0]), int(value[1])
    text = str(value).strip()
    if len(text) == 2 and text.isdigit():
        return int(text[0]), int(text[1])
    fields = re.findall(r"\d+", text)
    if len(fields) == 2:
        return int(fields[0]), int(fields[1])
    raise ValueError(f"cannot decode {label} converter identifier {value!r}")


def _extra_description_text(config: Mapping[str, Any]) -> str:
    description = config.get("extra_description", ())
    if isinstance(description, str):
        return description
    if isinstance(description, Sequence):
        return "\n".join(str(item) for item in description)
    return str(description)


def _detected_cards(config: Mapping[str, Any]) -> Dict[str, Dict[int, Optional[str]]]:
    cards: Dict[str, Dict[int, Optional[str]]] = {"DAC": {}, "ADC": {}}
    for match in _CARD_RE.finditer(_extra_description_text(config)):
        direction = match.group(1).upper()
        slot = int(match.group(2))
        raw_type = " ".join(match.group(3).split())
        board_type = (
            None
            if raw_type.lower() == "no card detected"
            else raw_type.replace(" ", "_")
        )
        cards[direction][slot] = board_type
    return cards


def identify_qick_front_panel(soccfg: Any) -> QickFrontPanelConfiguration:
    """Combine HWH-derived converter routing with detected ZCU216 card types.

    QICK's ZCU216 mapping is DAC port ``4*tile + block``. The exposed ADC
    ports use RFDC tiles 1 and 2, with panel port ``4*(tile-1) + block``.
    Daughter-card names come from ``RFQickSoc216V1`` startup detection stored
    in ``extra_description``.
    """

    config = _config_mapping(soccfg)
    board = str(config.get("board", "")).strip()
    if board.upper() != "ZCU216":
        raise ValueError(
            f"interactive front-panel mapping currently supports ZCU216, got {board!r}"
        )

    cards = _detected_cards(config)
    output_channels: Dict[int, list] = {index: [] for index in range(16)}
    output_paths: Dict[int, list] = {index: [] for index in range(16)}
    for channel, gen in enumerate(config.get("gens", ())):
        if not isinstance(gen, Mapping) or "dac" not in gen:
            continue
        tile, block = _converter_pair(gen["dac"], "DAC")
        panel_index = 4 * tile + block
        if 0 <= panel_index < 16:
            output_channels[panel_index].append(channel)
            output_paths[panel_index].append(str(gen.get("fullpath", gen.get("type", ""))))

    input_channels: Dict[int, list] = {index: [] for index in range(8)}
    input_paths: Dict[int, list] = {index: [] for index in range(8)}
    for channel, readout in enumerate(config.get("readouts", ())):
        if not isinstance(readout, Mapping) or "adc" not in readout:
            continue
        tile, block = _converter_pair(readout["adc"], "ADC")
        panel_index = 4 * (tile - 1) + block
        if 0 <= panel_index < 8:
            input_channels[panel_index].append(channel)
            input_paths[panel_index].append(
                str(
                    readout.get(
                        "avgbuf_fullpath",
                        readout.get("fullpath", readout.get("type", "")),
                    )
                )
            )

    outputs = tuple(
        QickFrontPanelPort(
            direction="output",
            panel_index=index,
            converter_id=f"{index // 4}{index % 4}",
            board_slot=index // 4,
            board_type=cards["DAC"].get(index // 4),
            qick_channels=tuple(output_channels[index]),
            block_paths=tuple(output_paths[index]),
        )
        for index in range(16)
    )
    inputs = tuple(
        QickFrontPanelPort(
            direction="input",
            panel_index=index,
            converter_id=f"{1 + index // 4}{index % 4}",
            board_slot=index // 2,
            board_type=cards["ADC"].get(index // 2),
            qick_channels=tuple(input_channels[index]),
            block_paths=tuple(input_paths[index]),
        )
        for index in range(8)
    )
    return QickFrontPanelConfiguration(
        board=board,
        firmware_timestamp=str(config.get("fw_timestamp", "unknown")),
        outputs=outputs,
        inputs=inputs,
    )


class QickFrontPanelCanvas(QtWidgets.QWidget):
    """Deterministic, scalable QICK-box front panel with clickable SMAs."""

    port_clicked = QtCore.pyqtSignal(str, int)

    LOGICAL_WIDTH = 1200.0
    LOGICAL_HEIGHT = 410.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(1080, 370)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self._configuration: Optional[QickFrontPanelConfiguration] = None
        self._selected_output: Optional[int] = None
        self._selected_input: Optional[int] = None
        self._hover: Optional[Tuple[str, int]] = None
        self._port_centers = self._build_port_centers()

    @staticmethod
    def _build_port_centers() -> Dict[Tuple[str, int], QtCore.QPointF]:
        centers: Dict[Tuple[str, int], QtCore.QPointF] = {}
        for slot in range(4):
            group_x = 30.0 + 190.0 * slot
            for local in range(4):
                centers[("output", 4 * slot + local)] = QtCore.QPointF(
                    group_x + 28.0 + 43.0 * local,
                    155.0,
                )
        for slot in range(4):
            group_x = 815.0 + 91.0 * slot
            for local in range(2):
                centers[("input", 2 * slot + local)] = QtCore.QPointF(
                    group_x + 25.0 + 40.0 * local,
                    155.0,
                )
        return centers

    def set_configuration(self, configuration: QickFrontPanelConfiguration) -> None:
        self._configuration = configuration
        self.update()

    def set_selected(self, direction: str, panel_index: Optional[int]) -> None:
        if direction == "output":
            self._selected_output = panel_index
        else:
            self._selected_input = panel_index
        self.update()

    def select_port(self, direction: str, panel_index: int) -> None:
        """Select a port programmatically; useful for keyboard flows and tests."""
        if direction not in ("output", "input"):
            raise ValueError("direction must be output or input")
        self.port_clicked.emit(direction, int(panel_index))

    def _display_transform(self) -> Tuple[float, float, float]:
        scale = min(
            self.width() / self.LOGICAL_WIDTH,
            self.height() / self.LOGICAL_HEIGHT,
        )
        return (
            scale,
            (self.width() - self.LOGICAL_WIDTH * scale) / 2.0,
            (self.height() - self.LOGICAL_HEIGHT * scale) / 2.0,
        )

    def _logical_point(self, point: QtCore.QPoint) -> QtCore.QPointF:
        scale, offset_x, offset_y = self._display_transform()
        return QtCore.QPointF(
            (point.x() - offset_x) / max(scale, 1.0e-9),
            (point.y() - offset_y) / max(scale, 1.0e-9),
        )

    def _hit_test(self, point: QtCore.QPoint) -> Optional[Tuple[str, int]]:
        logical = self._logical_point(point)
        for key, center in self._port_centers.items():
            delta = logical - center
            if delta.x() * delta.x() + delta.y() * delta.y() <= 22.0 * 22.0:
                return key
        return None

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            hit = self._hit_test(event.pos())
            if hit is not None:
                self.port_clicked.emit(*hit)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        hover = self._hit_test(event.pos())
        if hover != self._hover:
            self._hover = hover
            if hover is None or self._configuration is None:
                self.setToolTip("")
            else:
                port = self._configuration.port(*hover)
                self.setToolTip(
                    f"{port.label} | {port.board_label} | {port.channel_label} | "
                    f"RFDC {port.converter_id}"
                )
            self.update()
        super().mouseMoveEvent(event)

    @staticmethod
    def _indicator_color(active: bool, present: bool) -> QtGui.QColor:
        if active:
            return QtGui.QColor("#36b66a")
        if present:
            return QtGui.QColor("#5c4a36")
        return QtGui.QColor("#555c63")

    @staticmethod
    def _draw_indicator(
        painter: QtGui.QPainter,
        center: QtCore.QPointF,
        color: QtGui.QColor,
    ) -> None:
        painter.setPen(QtGui.QPen(QtGui.QColor("#2a1518"), 1.5))
        painter.setBrush(QtGui.QBrush(color))
        painter.drawEllipse(center, 9.0, 9.0)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, 85))
        painter.drawEllipse(center + QtCore.QPointF(-2.5, -2.5), 2.7, 2.7)

    def _draw_sma(
        self,
        painter: QtGui.QPainter,
        key: Tuple[str, int],
        center: QtCore.QPointF,
    ) -> None:
        direction, index = key
        port = (
            self._configuration.port(direction, index)
            if self._configuration is not None and direction in ("output", "input")
            else None
        )
        mapped = direction == "aux" or bool(port and port.qick_channels)
        selected = (
            index == self._selected_output
            if direction == "output"
            else index == self._selected_input
            if direction == "input"
            else False
        )
        hovered = key == self._hover
        outline = QtGui.QColor(
            "#32b6d8" if direction == "output" else "#43c581"
        )
        if not (selected or hovered):
            outline = QtGui.QColor("#49362b" if mapped else "#756c67")
        painter.setPen(QtGui.QPen(outline, 4.0 if selected else 2.0))
        painter.setBrush(QtGui.QColor("#d7a83d" if mapped else "#9d8a66"))
        painter.drawEllipse(center, 16.0, 16.0)
        painter.setPen(QtGui.QPen(QtGui.QColor("#6d511e"), 1.5))
        painter.setBrush(QtGui.QColor("#f3df9b"))
        painter.drawEllipse(center, 10.0, 10.0)
        painter.setBrush(QtGui.QColor("#8b6d2c"))
        painter.drawEllipse(center, 4.0, 4.0)

    def _draw_card_group(
        self,
        painter: QtGui.QPainter,
        direction: str,
        slot: int,
        rect: QtCore.QRectF,
    ) -> None:
        ports = (
            self._configuration.outputs
            if self._configuration is not None and direction == "output"
            else self._configuration.inputs
            if self._configuration is not None
            else ()
        )
        first_index = 4 * slot if direction == "output" else 2 * slot
        board_type = (
            ports[first_index].board_type if len(ports) > first_index else None
        )
        present = board_type is not None
        rf_active = board_type in ("RF_Out", "RF_In")
        dc_active = board_type in ("DC_Out", "DC_In")
        painter.setPen(QtGui.QPen(QtGui.QColor("#d49b49"), 1.5))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRoundedRect(rect, 4.0, 4.0)
        painter.setPen(QtGui.QColor("#f2d7a1"))
        painter.drawText(
            QtCore.QRectF(rect.left(), rect.top() + 5.0, rect.width(), 18.0),
            QtCore.Qt.AlignCenter,
            "DAUGHTER CARD",
        )
        center_x = rect.center().x()
        self._draw_indicator(
            painter,
            QtCore.QPointF(center_x - 26.0, rect.top() + 42.0),
            self._indicator_color(rf_active, present),
        )
        self._draw_indicator(
            painter,
            QtCore.QPointF(center_x + 26.0, rect.top() + 42.0),
            self._indicator_color(dc_active, present),
        )
        painter.setPen(QtGui.QColor("#f4dfbb"))
        painter.drawText(
            QtCore.QRectF(center_x - 43.0, rect.top() + 52.0, 34.0, 16.0),
            QtCore.Qt.AlignCenter,
            "RF",
        )
        painter.drawText(
            QtCore.QRectF(center_x + 9.0, rect.top() + 52.0, 34.0, 16.0),
            QtCore.Qt.AlignCenter,
            "DC",
        )

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        scale, offset_x, offset_y = self._display_transform()
        painter.translate(offset_x, offset_y)
        painter.scale(scale, scale)

        panel = QtCore.QRectF(4.0, 4.0, self.LOGICAL_WIDTH - 8.0, self.LOGICAL_HEIGHT - 8.0)
        painter.setPen(QtGui.QPen(QtGui.QColor("#481018"), 3.0))
        painter.setBrush(QtGui.QColor("#8e1d2b"))
        painter.drawRoundedRect(panel, 8.0, 8.0)
        painter.setPen(QtGui.QColor("#f2d7a1"))
        title_font = painter.font()
        title_font.setBold(True)
        title_font.setPointSizeF(11.0)
        painter.setFont(title_font)
        painter.drawText(QtCore.QRectF(20.0, 12.0, 760.0, 26.0), QtCore.Qt.AlignCenter, "DAC OUTPUTS")
        painter.drawText(QtCore.QRectF(805.0, 12.0, 375.0, 26.0), QtCore.Qt.AlignCenter, "ADC INPUTS")

        body_font = painter.font()
        body_font.setBold(False)
        body_font.setPointSizeF(8.0)
        painter.setFont(body_font)
        for slot in range(4):
            self._draw_card_group(
                painter,
                "output",
                slot,
                QtCore.QRectF(30.0 + 190.0 * slot, 42.0, 178.0, 176.0),
            )
            self._draw_card_group(
                painter,
                "input",
                slot,
                QtCore.QRectF(815.0 + 91.0 * slot, 42.0, 84.0, 176.0),
            )

        for key, center in self._port_centers.items():
            self._draw_sma(painter, key, center)
            direction, index = key
            painter.setPen(QtGui.QColor("#f5e4c4"))
            painter.drawText(
                QtCore.QRectF(center.x() - 23.0, center.y() + 20.0, 46.0, 18.0),
                QtCore.Qt.AlignCenter,
                f"{'DAC' if direction == 'output' else 'ADC'}{index}",
            )

        painter.setPen(QtGui.QPen(QtGui.QColor("#d49b49"), 1.2))
        painter.drawLine(QtCore.QPointF(20.0, 240.0), QtCore.QPointF(1180.0, 240.0))
        painter.setPen(QtGui.QColor("#f2d7a1"))
        painter.drawText(QtCore.QRectF(35.0, 250.0, 330.0, 20.0), QtCore.Qt.AlignCenter, "BIAS OUTPUTS")
        painter.drawText(QtCore.QRectF(495.0, 250.0, 310.0, 20.0), QtCore.Qt.AlignCenter, "STATUS LEDS")
        painter.drawText(QtCore.QRectF(825.0, 250.0, 320.0, 20.0), QtCore.Qt.AlignCenter, "DIGITAL I/O")

        for index in range(8):
            center = QtCore.QPointF(55.0 + 40.0 * index, 310.0)
            self._draw_sma(painter, ("aux", index), center)
            painter.setPen(QtGui.QColor("#f5e4c4"))
            painter.drawText(QtCore.QRectF(center.x() - 22.0, 332.0, 44.0, 18.0), QtCore.Qt.AlignCenter, f"BIAS{index}")

        painter.setPen(QtGui.QPen(QtGui.QColor("#25282c"), 2.0))
        painter.setBrush(QtGui.QColor("#a9adb1"))
        painter.drawRoundedRect(QtCore.QRectF(390.0, 282.0, 74.0, 55.0), 3.0, 3.0)
        for row in range(2):
            for column in range(8):
                painter.setBrush(QtGui.QColor("#3d4247"))
                painter.drawEllipse(QtCore.QPointF(399.0 + 8.0 * column, 295.0 + 18.0 * row), 2.2, 2.2)

        for index in range(8):
            self._draw_indicator(
                painter,
                QtCore.QPointF(515.0 + 38.0 * index, 310.0),
                QtGui.QColor("#536168"),
            )
            painter.setPen(QtGui.QColor("#f5e4c4"))
            painter.drawText(QtCore.QRectF(495.0 + 38.0 * index, 332.0, 40.0, 18.0), QtCore.Qt.AlignCenter, f"LED{index}")

        for index in range(8):
            center = QtCore.QPointF(845.0 + 39.0 * index, 310.0)
            self._draw_sma(painter, ("aux", 8 + index), center)
            painter.setPen(QtGui.QColor("#f5e4c4"))
            painter.drawText(QtCore.QRectF(center.x() - 20.0, 332.0, 40.0, 18.0), QtCore.Qt.AlignCenter, f"IO{index}")

        self._draw_indicator(painter, QtCore.QPointF(1166.0, 310.0), QtGui.QColor("#36b66a"))
        painter.setPen(QtGui.QColor("#f5e4c4"))
        painter.drawText(QtCore.QRectF(1138.0, 332.0, 56.0, 18.0), QtCore.Qt.AlignCenter, "POWER")
        painter.end()


class QickFrontPanelControl(QtWidgets.QWidget):
    """Shared graphical QICK path selector; manual per-tab fields remain valid."""

    identify_requested = QtCore.pyqtSignal()
    settings_applied = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._configuration: Optional[QickFrontPanelConfiguration] = None
        self._selected_output: Optional[int] = None
        self._selected_input: Optional[int] = None
        self._preferred_output_ch = 0
        self._preferred_input_ch = 0

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        self.identify_button = QtWidgets.QPushButton("Identify QICK Configuration")
        self.identify_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
        )
        self.identify_button.clicked.connect(self.identify_requested.emit)
        self.device_label = QtWidgets.QLabel("QICK not identified")
        self.device_label.setStyleSheet("QLabel { font-weight: 600; }")
        self.identify_status = QtWidgets.QLabel("Ready")
        self.identify_status.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        header.addWidget(self.identify_button)
        header.addWidget(self.device_label)
        header.addStretch(1)
        header.addWidget(self.identify_status)
        layout.addLayout(header)

        self.canvas = QickFrontPanelCanvas(self)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(390)
        scroll.setWidget(self.canvas)
        layout.addWidget(scroll)
        self.canvas.port_clicked.connect(self._select_port)

        controls = QtWidgets.QHBoxLayout()
        self.output_group = QtWidgets.QGroupBox("Selected Output SMA")
        output_form = QtWidgets.QFormLayout(self.output_group)
        self.output_sma = QtWidgets.QLabel("-")
        self.output_channel = QtWidgets.QComboBox()
        self.output_board = QtWidgets.QLabel("-")
        self.output_att1_db = self._attenuation_spin(10.0)
        self.output_att2_db = self._attenuation_spin(10.0)
        output_form.addRow("Front-panel connector:", self.output_sma)
        output_form.addRow("QICK generator channel:", self.output_channel)
        output_form.addRow("Detected board:", self.output_board)
        output_form.addRow("ATT1:", self.output_att1_db)
        output_form.addRow("ATT2:", self.output_att2_db)
        controls.addWidget(self.output_group)

        self.input_group = QtWidgets.QGroupBox("Selected Input SMA")
        input_form = QtWidgets.QFormLayout(self.input_group)
        self.input_sma = QtWidgets.QLabel("-")
        self.input_channel = QtWidgets.QComboBox()
        self.input_board = QtWidgets.QLabel("-")
        self.input_condition_stack = QtWidgets.QStackedWidget()
        self.input_attenuation_db = self._attenuation_spin(20.0)
        self.input_dc_gain_db = QtWidgets.QDoubleSpinBox()
        self.input_dc_gain_db.setRange(-6.0, 26.0)
        self.input_dc_gain_db.setDecimals(1)
        self.input_dc_gain_db.setSingleStep(1.0)
        self.input_dc_gain_db.setSuffix(" dB")
        self.input_unknown = QtWidgets.QLabel("-")
        self.input_condition_stack.addWidget(self.input_attenuation_db)
        self.input_condition_stack.addWidget(self.input_dc_gain_db)
        self.input_condition_stack.addWidget(self.input_unknown)
        self.input_condition_label = QtWidgets.QLabel("Input ATT:")
        input_form.addRow("Front-panel connector:", self.input_sma)
        input_form.addRow("QICK readout channel:", self.input_channel)
        input_form.addRow("Detected board:", self.input_board)
        input_form.addRow(self.input_condition_label, self.input_condition_stack)
        controls.addWidget(self.input_group)
        layout.addLayout(controls)

        self.summary = QtWidgets.QLabel("No live QICK configuration loaded")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.summary.setStyleSheet(
            "QLabel { color: #20252b; background: #f3f6f8; "
            "border: 1px solid #aeb7c2; padding: 8px; }"
        )
        layout.addWidget(self.summary)
        action_row = QtWidgets.QHBoxLayout()
        action_row.addStretch(1)
        self.apply_button = QtWidgets.QPushButton("Update Measurement Channels")
        self.apply_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        )
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_settings)
        action_row.addWidget(self.apply_button)
        layout.addLayout(action_row)

        self.output_channel.currentIndexChanged.connect(self._update_summary)
        self.input_channel.currentIndexChanged.connect(self._update_summary)
        self.output_att1_db.valueChanged.connect(self._update_summary)
        self.output_att2_db.valueChanged.connect(self._update_summary)
        self.input_attenuation_db.valueChanged.connect(self._update_summary)
        self.input_dc_gain_db.valueChanged.connect(self._update_summary)

    @staticmethod
    def _attenuation_spin(value: float) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(0.0, 31.75)
        widget.setDecimals(2)
        widget.setSingleStep(0.25)
        widget.setValue(value)
        widget.setSuffix(" dB")
        return widget

    def set_identifying(self, identifying: bool, message: str) -> None:
        self.identify_button.setEnabled(not identifying)
        self.identify_status.setText(message)
        self.apply_button.setEnabled(not identifying and self._selection_is_valid())

    def set_configuration(self, configuration: QickFrontPanelConfiguration) -> None:
        self._configuration = configuration
        self.canvas.set_configuration(configuration)
        self.device_label.setText(
            f"{configuration.board} | HWH {configuration.firmware_timestamp}"
        )
        self.identify_status.setText(
            f"{configuration.mapped_output_count} DAC / "
            f"{configuration.mapped_input_count} ADC ports mapped"
        )
        output_index = self._find_port_for_channel(
            configuration.outputs,
            self._preferred_output_ch,
        )
        input_index = self._find_port_for_channel(
            configuration.inputs,
            self._preferred_input_ch,
        )
        self._select_port("output", output_index)
        self._select_port("input", input_index)

    @staticmethod
    def _find_port_for_channel(
        ports: Sequence[QickFrontPanelPort],
        channel: int,
    ) -> int:
        for port in ports:
            if channel in port.qick_channels:
                return port.panel_index
        for port in ports:
            if port.qick_channels and port.board_type is not None:
                return port.panel_index
        for port in ports:
            if port.qick_channels:
                return port.panel_index
        return 0

    def _select_port(self, direction: str, panel_index: int) -> None:
        if self._configuration is None:
            return
        port = self._configuration.port(direction, panel_index)
        channel_combo = self.output_channel if direction == "output" else self.input_channel
        preferred = (
            self._preferred_output_ch if direction == "output" else self._preferred_input_ch
        )
        with QtCore.QSignalBlocker(channel_combo):
            channel_combo.clear()
            for channel, path in zip(port.qick_channels, port.block_paths):
                channel_combo.addItem(f"{channel}  |  {path}", channel)
            match = channel_combo.findData(preferred)
            if match >= 0:
                channel_combo.setCurrentIndex(match)
        if direction == "output":
            self._selected_output = int(panel_index)
            self.output_sma.setText(port.label)
            self.output_board.setText(port.board_label)
            rf_output = port.board_type == "RF_Out"
            self.output_att1_db.setEnabled(rf_output)
            self.output_att2_db.setEnabled(rf_output)
        else:
            self._selected_input = int(panel_index)
            self.input_sma.setText(port.label)
            self.input_board.setText(port.board_label)
            if port.board_type == "RF_In":
                self.input_condition_stack.setCurrentIndex(0)
                self.input_condition_label.setText("Input ATT:")
            elif port.board_type == "DC_In":
                self.input_condition_stack.setCurrentIndex(1)
                self.input_condition_label.setText("DC input gain:")
            else:
                self.input_condition_stack.setCurrentIndex(2)
                self.input_condition_label.setText("Input setting:")
        self.canvas.set_selected(direction, int(panel_index))
        self.apply_button.setEnabled(self._selection_is_valid())
        self._update_summary()

    def _selection_is_valid(self) -> bool:
        if self._configuration is None:
            return False
        if self._selected_output is None or self._selected_input is None:
            return False
        output = self._configuration.port("output", self._selected_output)
        input_port = self._configuration.port("input", self._selected_input)
        return bool(
            output.board_type in OUTPUT_BOARD_TYPES
            and input_port.board_type in INPUT_BOARD_TYPES
            and self.output_channel.currentData() is not None
            and self.input_channel.currentData() is not None
        )

    def set_path_values(self, values: Mapping[str, Any]) -> None:
        """Mirror manually edited path values without replacing live card identity."""
        self._preferred_output_ch = int(values.get("output_ch", self._preferred_output_ch))
        self._preferred_input_ch = int(values.get("readout_ch", self._preferred_input_ch))
        self.output_att1_db.setValue(float(values.get("output_att1_db", self.output_att1_db.value())))
        self.output_att2_db.setValue(float(values.get("output_att2_db", self.output_att2_db.value())))
        self.input_attenuation_db.setValue(
            float(values.get("readout_attenuation_db", self.input_attenuation_db.value()))
        )
        self.input_dc_gain_db.setValue(
            float(values.get("readout_dc_gain_db", self.input_dc_gain_db.value()))
        )
        if self._configuration is not None:
            self._select_port(
                "output",
                self._find_port_for_channel(
                    self._configuration.outputs,
                    self._preferred_output_ch,
                ),
            )
            self._select_port(
                "input",
                self._find_port_for_channel(
                    self._configuration.inputs,
                    self._preferred_input_ch,
                ),
            )

    def selected_settings(self) -> Dict[str, Any]:
        if not self._selection_is_valid() or self._configuration is None:
            raise ValueError("select mapped DAC and ADC SMAs before applying")
        output = self._configuration.port("output", self._selected_output)
        input_port = self._configuration.port("input", self._selected_input)
        return {
            "output_ch": int(self.output_channel.currentData()),
            "readout_ch": int(self.input_channel.currentData()),
            "output_board_type": str(output.board_type),
            "input_board_type": str(input_port.board_type),
            "output_att1_db": self.output_att1_db.value(),
            "output_att2_db": self.output_att2_db.value(),
            "readout_attenuation_db": self.input_attenuation_db.value(),
            "readout_dc_gain_db": self.input_dc_gain_db.value(),
            "output_panel_port": output.panel_index,
            "input_panel_port": input_port.panel_index,
            "output_converter_id": output.converter_id,
            "input_converter_id": input_port.converter_id,
        }

    def apply_settings(self) -> None:
        values = self.selected_settings()
        self._preferred_output_ch = values["output_ch"]
        self._preferred_input_ch = values["readout_ch"]
        self.settings_applied.emit(values)
        self.identify_status.setText(
            f"Applied DAC{values['output_panel_port']} / ADC{values['input_panel_port']}"
        )

    def _update_summary(self, *_args) -> None:
        if self._configuration is None:
            self.summary.setText("No live QICK configuration loaded")
            return
        output = self._configuration.port("output", self._selected_output or 0)
        input_port = self._configuration.port("input", self._selected_input or 0)
        output_ch = self.output_channel.currentData()
        input_ch = self.input_channel.currentData()
        output_path = self.output_channel.currentText().partition("|")[2].strip() or "unmapped"
        input_path = self.input_channel.currentText().partition("|")[2].strip() or "unmapped"
        output_condition = (
            f"ATT1 {self.output_att1_db.value():.2f} dB, "
            f"ATT2 {self.output_att2_db.value():.2f} dB"
            if output.board_type == "RF_Out"
            else "no output attenuator"
        )
        input_condition = (
            f"ATT {self.input_attenuation_db.value():.2f} dB"
            if input_port.board_type == "RF_In"
            else f"gain {self.input_dc_gain_db.value():.1f} dB"
            if input_port.board_type == "DC_In"
            else "no detected input board"
        )
        self.summary.setText(
            f"OUTPUT  {output.label} | QICK gen {output_ch if output_ch is not None else '-'} | "
            f"RFDC DAC {output.converter_id} | slot {output.board_slot} {output.board_label} | "
            f"{output_condition} | {output_path}\n"
            f"INPUT   {input_port.label} | QICK readout {input_ch if input_ch is not None else '-'} | "
            f"RFDC ADC {input_port.converter_id} | slot {input_port.board_slot} {input_port.board_label} | "
            f"{input_condition} | {input_path}"
        )


__all__ = [
    "QickFrontPanelConfiguration",
    "QickFrontPanelControl",
    "QickFrontPanelPort",
    "identify_qick_front_panel",
]
