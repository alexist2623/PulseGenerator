"""Backend-aware front-panel preview shared by experiment editors."""

from __future__ import annotations

from typing import Mapping, Optional

from PyQt5 import QtCore, QtWidgets

try:
    from .qcs_front_panel import QcsFrontPanelPreview
    from .qick_front_panel import QickFrontPanelPreview
except ImportError:
    from qcs_front_panel import QcsFrontPanelPreview
    from qick_front_panel import QickFrontPanelPreview


EXECUTION_BACKEND_QICK = "qick"
EXECUTION_BACKEND_QCS = "qcs"
EXECUTION_BACKENDS = (EXECUTION_BACKEND_QICK, EXECUTION_BACKEND_QCS)


class HardwareFrontPanelPreview(QtWidgets.QStackedWidget):
    """Switch between the QICK and QCS previews without changing panel APIs."""

    activated = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scope = "path"
        self._backend = EXECUTION_BACKEND_QCS
        self._width_reference: Optional[QtWidgets.QWidget] = None
        self._visible_width_margin = 36
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.qick_preview = QickFrontPanelPreview(self)
        self.qcs_preview = QcsFrontPanelPreview(self)
        self.qick_preview.installEventFilter(self)
        self.qcs_preview.installEventFilter(self)
        self.addWidget(self.qick_preview)
        self.addWidget(self.qcs_preview)
        self.qick_preview.activated.connect(self.activated.emit)
        self.qcs_preview.activated.connect(self.activated.emit)
        self.set_backend(self._backend)

    def sizeHint(self) -> QtCore.QSize:
        current = self.currentWidget()
        return current.sizeHint().expandedTo(current.minimumSize())

    def minimumSizeHint(self) -> QtCore.QSize:
        current = self.currentWidget()
        return current.minimumSizeHint().expandedTo(current.minimumSize())

    def hasHeightForWidth(self) -> bool:
        return self.currentWidget().hasHeightForWidth()

    def heightForWidth(self, width: int) -> int:
        current = self.currentWidget()
        if current.hasHeightForWidth():
            return current.heightForWidth(width)
        return current.sizeHint().height()

    def set_backend(self, backend: str) -> None:
        backend = str(backend).strip().lower()
        if backend not in EXECUTION_BACKENDS:
            raise ValueError(f"unsupported execution backend {backend!r}")
        self._backend = backend
        self.setCurrentWidget(
            self.qcs_preview
            if backend == EXECUTION_BACKEND_QCS
            else self.qick_preview
        )
        self._sync_current_size_constraints()
        QtCore.QTimer.singleShot(0, self._sync_current_size_constraints)
        QtCore.QTimer.singleShot(0, self._fit_visible_width)

    def eventFilter(self, watched, event) -> bool:
        width_reference = getattr(self, "_width_reference", None)
        if (
            width_reference is not None
            and watched is width_reference
            and event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show)
        ):
            QtCore.QTimer.singleShot(0, self._fit_visible_width)
        if (
            watched is self.currentWidget()
            and event.type()
            in (
                QtCore.QEvent.LayoutRequest,
                QtCore.QEvent.Resize,
                QtCore.QEvent.Show,
            )
        ):
            QtCore.QTimer.singleShot(0, self._sync_current_size_constraints)
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._bind_visible_width_reference()
        self._fit_visible_width()
        QtCore.QTimer.singleShot(0, self._fit_visible_width)

    def _bind_visible_width_reference(self) -> None:
        reference = self._nearest_scroll_viewport()
        if reference is None:
            reference = self.parentWidget()
        if reference is self._width_reference:
            return
        if self._width_reference is not None:
            self._width_reference.removeEventFilter(self)
        self._width_reference = reference
        if reference is not None:
            reference.installEventFilter(self)

    def _nearest_scroll_viewport(self) -> Optional[QtWidgets.QWidget]:
        widget = self.parentWidget()
        while widget is not None:
            parent = widget.parentWidget()
            if (
                isinstance(parent, QtWidgets.QAbstractScrollArea)
                and parent.viewport() is widget
            ):
                return widget
            widget = parent
        return None

    def _fit_visible_width(self) -> None:
        self._bind_visible_width_reference()
        if self._width_reference is None:
            return
        available_width = self._width_reference.width()
        ancestor = self.parentWidget()
        while ancestor is not None and ancestor is not self._width_reference:
            if ancestor.isVisible() and ancestor.width() > 0:
                available_width = min(available_width, ancestor.width())
            ancestor = ancestor.parentWidget()
        target_width = max(210, available_width - self._visible_width_margin)
        if self.maximumWidth() != target_width:
            self.setMaximumWidth(target_width)
        self.updateGeometry()

    def _sync_current_size_constraints(self) -> None:
        current = self.currentWidget()
        minimum_height = max(
            current.minimumSizeHint().height(),
            current.minimumHeight(),
        )
        maximum_height = max(minimum_height, current.maximumHeight())
        if self.minimumHeight() != minimum_height:
            self.setMinimumHeight(minimum_height)
        if self.maximumHeight() != maximum_height:
            self.setMaximumHeight(maximum_height)
        self.updateGeometry()

    def set_scope(self, scope: str) -> None:
        self._scope = str(scope)
        self.qick_preview.set_scope(scope)

    def set_channels(
        self,
        *,
        output_ch: Optional[int] = None,
        input_ch: Optional[int] = None,
    ) -> None:
        self.qick_preview.set_channels(
            output_ch=output_ch,
            input_ch=input_ch,
        )

    def set_configuration(self, configuration) -> None:
        """Set the live QICK HWH configuration."""

        self.qick_preview.set_configuration(configuration)

    def set_qcs_configuration(
        self,
        configuration: Optional[Mapping[str, object]],
    ) -> None:
        """Set the editable QCS hardware recipe rendered by the preview."""

        self.qcs_preview.set_configuration(configuration)

    def set_qcs_selection(self, role: str, logical_index: int = 0) -> None:
        self.qcs_preview.set_selection(role, logical_index)


__all__ = [
    "EXECUTION_BACKEND_QCS",
    "EXECUTION_BACKEND_QICK",
    "EXECUTION_BACKENDS",
    "HardwareFrontPanelPreview",
]
