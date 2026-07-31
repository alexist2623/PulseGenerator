"""Independent DAC11001 bias-output controls for the QICK RF board.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import traceback
from typing import Mapping, Optional, Sequence

from PyQt5 import QtCore, QtWidgets

try:
    from .qick_front_panel import (
        QickFrontPanelCanvas,
        QickFrontPanelConfiguration,
    )
    from .qick_qcodes_experiment import (
        QickConnectionConfig,
        connect_qick,
    )
except ImportError:
    from qick_front_panel import (
        QickFrontPanelCanvas,
        QickFrontPanelConfiguration,
    )
    from qick_qcodes_experiment import (
        QickConnectionConfig,
        connect_qick,
    )


BIAS_CHANNEL_COUNT = 8
BIAS_MIN_V = -10.0
BIAS_MAX_V = 10.0
BIAS_DEFAULT_V = 0.0


class BiasChannelEditor(QtWidgets.QFrame):
    """One DAC11001 channel setpoint editor."""

    selected = QtCore.pyqtSignal(int)
    apply_requested = QtCore.pyqtSignal(int, float)

    def __init__(self, channel: int, parent=None):
        super().__init__(parent)
        self.channel = int(channel)
        self.setObjectName("biasChannelEditor")
        self.setProperty("selected", False)
        self.setMinimumWidth(178)
        self.setMaximumWidth(220)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setCursor(QtCore.Qt.PointingHandCursor)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(11, 9, 11, 9)
        layout.setSpacing(6)

        title_row = QtWidgets.QHBoxLayout()
        self.channel_label = QtWidgets.QLabel(f"BIAS{self.channel}")
        title_font = self.channel_label.font()
        title_font.setBold(True)
        self.channel_label.setFont(title_font)
        self.device_label = QtWidgets.QLabel("DAC11001")
        self.device_label.setStyleSheet("QLabel { color: #56616d; }")
        title_row.addWidget(self.channel_label)
        title_row.addStretch(1)
        title_row.addWidget(self.device_label)
        layout.addLayout(title_row)

        self.voltage = QtWidgets.QDoubleSpinBox(self)
        self.voltage.setRange(BIAS_MIN_V, BIAS_MAX_V)
        self.voltage.setDecimals(6)
        self.voltage.setSingleStep(0.001)
        self.voltage.setSuffix(" V")
        self.voltage.setValue(BIAS_DEFAULT_V)
        self.voltage.setKeyboardTracking(False)
        layout.addWidget(self.voltage)

        self.actual_label = QtWidgets.QLabel("Setpoint: not read")
        self.actual_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        layout.addWidget(self.actual_label)

        self.apply_button = QtWidgets.QPushButton("Apply", self)
        self.apply_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        )
        layout.addWidget(self.apply_button)

        self.apply_button.clicked.connect(self._apply)
        self.voltage.valueChanged.connect(
            lambda _value: self.selected.emit(self.channel)
        )
        self.voltage.installEventFilter(self)
        self.apply_button.installEventFilter(self)
        self._refresh_style()

    def eventFilter(self, watched, event) -> bool:
        if (
            watched in (self.voltage, self.apply_button)
            and event.type()
            in (QtCore.QEvent.FocusIn, QtCore.QEvent.MouseButtonPress)
        ):
            self.selected.emit(self.channel)
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.selected.emit(self.channel)
        super().mousePressEvent(event)

    def _apply(self) -> None:
        self.selected.emit(self.channel)
        self.apply_requested.emit(self.channel, self.voltage.value())

    def set_selected(self, selected: bool) -> None:
        selected = bool(selected)
        if bool(self.property("selected")) == selected:
            return
        self.setProperty("selected", selected)
        self._refresh_style()

    def _refresh_style(self) -> None:
        selected = bool(self.property("selected"))
        border_width = 4 if selected else 1
        border_color = "#1696b6" if selected else "#aeb7c2"
        background = "#eefbff" if selected else "#f7f9fb"
        self.setStyleSheet(
            "QFrame#biasChannelEditor {"
            f"border: {border_width}px solid {border_color};"
            "border-radius: 6px;"
            f"background: {background};"
            "}"
            "QFrame#biasChannelEditor QLabel, "
            "QFrame#biasChannelEditor QDoubleSpinBox, "
            "QFrame#biasChannelEditor QPushButton {"
            "border: none;"
            "background: transparent;"
            "}"
            "QFrame#biasChannelEditor QDoubleSpinBox {"
            "background: white;"
            "border: 1px solid #9ea8b3;"
            "padding: 3px;"
            "}"
            "QFrame#biasChannelEditor QPushButton {"
            "background: #e8edf2;"
            "border: 1px solid #9ea8b3;"
            "padding: 4px;"
            "}"
        )

    def set_actual_voltage(self, voltage: float) -> None:
        voltage = float(voltage)
        with QtCore.QSignalBlocker(self.voltage):
            self.voltage.setValue(voltage)
        self.actual_label.setText(f"Setpoint: {voltage:+.6f} V")

    def set_busy(self, busy: bool) -> None:
        self.voltage.setEnabled(not busy)
        self.apply_button.setEnabled(not busy)


class BiasControlPanel(QtWidgets.QWidget):
    """Front-panel-centric editor for all eight QICK bias outputs."""

    read_requested = QtCore.pyqtSignal()
    set_requested = QtCore.pyqtSignal(int, float)
    set_all_requested = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        self._configuration: Optional[QickFrontPanelConfiguration] = None
        self._selected_channel = 0
        self._busy = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("DAC11001 Bias Outputs")
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)
        self.device_label = QtWidgets.QLabel(
            "8 channels | -10 V to +10 V | 20-bit"
        )
        self.read_button = QtWidgets.QPushButton("Read All Setpoints")
        self.read_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
        )
        self.apply_all_button = QtWidgets.QPushButton("Apply All")
        self.apply_all_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        )
        header.addWidget(title)
        header.addWidget(self.device_label)
        header.addStretch(1)
        header.addWidget(self.read_button)
        header.addWidget(self.apply_all_button)
        layout.addLayout(header)

        self.front_panel = QickFrontPanelCanvas(self)
        self.front_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.front_panel.setMinimumHeight(220)
        self.front_panel.setMaximumHeight(360)
        self.front_panel.port_clicked.connect(self._front_panel_clicked)
        layout.addWidget(self.front_panel)

        self.editor_scroll = QtWidgets.QScrollArea(self)
        self.editor_scroll.setWidgetResizable(True)
        self.editor_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded
        )
        self.editor_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.editor_scroll.setMinimumHeight(150)
        editor_content = QtWidgets.QWidget(self.editor_scroll)
        editor_layout = QtWidgets.QHBoxLayout(editor_content)
        editor_layout.setContentsMargins(2, 2, 2, 2)
        editor_layout.setSpacing(7)
        self.editors = []
        for channel in range(BIAS_CHANNEL_COUNT):
            editor = BiasChannelEditor(channel, editor_content)
            editor.selected.connect(self.select_channel)
            editor.apply_requested.connect(self.set_requested.emit)
            self.editors.append(editor)
            editor_layout.addWidget(editor)
        editor_layout.addStretch(1)
        self.editor_scroll.setWidget(editor_content)
        layout.addWidget(self.editor_scroll)

        self.status = QtWidgets.QLabel(
            "Ready | selected BIAS0 | hardware setpoints not read"
        )
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.status.setStyleSheet(
            "QLabel { color: #20252b; background: #f3f6f8; "
            "border: 1px solid #aeb7c2; padding: 6px; }"
        )
        layout.addWidget(self.status)

        self.read_button.clicked.connect(self.read_requested.emit)
        self.apply_all_button.clicked.connect(self._apply_all)
        self.select_channel(0, focus=False)

    @property
    def selected_channel(self) -> int:
        return self._selected_channel

    def set_configuration(
        self,
        configuration: QickFrontPanelConfiguration,
    ) -> None:
        self._configuration = configuration
        self.front_panel.set_configuration(configuration)
        self.device_label.setText(
            f"{configuration.board} | 8 DAC11001 channels | -10 V to +10 V"
        )

    def _front_panel_clicked(self, direction: str, channel: int) -> None:
        if direction == "bias":
            self.select_channel(channel)

    @QtCore.pyqtSlot(int)
    def select_channel(self, channel: int, *, focus: bool = True) -> None:
        channel = int(channel)
        if not 0 <= channel < BIAS_CHANNEL_COUNT:
            raise IndexError("bias channel must be between 0 and 7")
        self._selected_channel = channel
        self.front_panel.set_selected("bias", channel)
        for index, editor in enumerate(self.editors):
            editor.set_selected(index == channel)
        selected = self.editors[channel]
        if channel == 0:
            self.editor_scroll.horizontalScrollBar().setValue(0)
        else:
            self.editor_scroll.ensureWidgetVisible(selected, 20, 10)
        if focus:
            selected.voltage.setFocus(QtCore.Qt.MouseFocusReason)
        self.status.setText(
            f"Ready | selected BIAS{channel} | "
            f"requested {selected.voltage.value():+.6f} V"
        )

    def _apply_all(self) -> None:
        values = tuple(editor.voltage.value() for editor in self.editors)
        self.set_all_requested.emit(values)

    def set_busy(self, busy: bool, message: str) -> None:
        self._busy = bool(busy)
        self.read_button.setEnabled(not busy)
        self.apply_all_button.setEnabled(not busy)
        for editor in self.editors:
            editor.set_busy(busy)
        self.status.setText(str(message))

    def apply_hardware_values(self, values: Mapping[int, float]) -> None:
        for channel, voltage in values.items():
            channel = int(channel)
            if 0 <= channel < len(self.editors):
                self.editors[channel].set_actual_voltage(float(voltage))
        selected_voltage = self.editors[
            self._selected_channel
        ].voltage.value()
        self.status.setText(
            f"Updated {len(values)} channel(s) | selected "
            f"BIAS{self._selected_channel} = {selected_voltage:+.6f} V"
        )

    def settings_dict(self) -> dict:
        return {
            "selected_channel": self._selected_channel,
            "setpoints_v": [
                editor.voltage.value() for editor in self.editors
            ],
        }

    def load_settings(self, settings: Mapping[str, object]) -> None:
        values = settings.get(
            "setpoints_v",
            [BIAS_DEFAULT_V] * BIAS_CHANNEL_COUNT,
        )
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != BIAS_CHANNEL_COUNT
        ):
            raise ValueError("bias setpoints_v must contain eight values")
        for editor, value in zip(self.editors, values):
            value = float(value)
            if not BIAS_MIN_V <= value <= BIAS_MAX_V:
                raise ValueError(
                    "bias setpoints must be between -10 V and +10 V"
                )
            with QtCore.QSignalBlocker(editor.voltage):
                editor.voltage.setValue(value)
        self.select_channel(
            int(settings.get("selected_channel", 0)),
            focus=False,
        )


class BiasHardwareWorker(QtCore.QObject):
    """Read or update DAC11001 setpoints without blocking the Qt GUI."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        connection_config: QickConnectionConfig,
        operation: str,
        values: Optional[Mapping[int, float]] = None,
        parent=None,
    ):
        super().__init__(parent)
        if operation not in {"read", "set"}:
            raise ValueError("bias operation must be read or set")
        self._connection_config = connection_config
        self._operation = operation
        self._values = {
            int(channel): float(voltage)
            for channel, voltage in (values or {}).items()
        }

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            soc, _soccfg = connect_qick(self._connection_config)
            if self._operation == "read":
                values = {
                    channel: float(soc.rfb_get_bias(channel))
                    for channel in range(BIAS_CHANNEL_COUNT)
                }
            else:
                values = {
                    channel: float(soc.rfb_set_bias(channel, voltage))
                    for channel, voltage in sorted(self._values.items())
                }
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(values)


__all__ = [
    "BIAS_CHANNEL_COUNT",
    "BIAS_DEFAULT_V",
    "BIAS_MAX_V",
    "BIAS_MIN_V",
    "BiasChannelEditor",
    "BiasControlPanel",
    "BiasHardwareWorker",
]
