"""Local-only RF filter editor; hardware changes use the existing Apply path."""

from math import ceil, isfinite

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from .qick_filter_settings import resolve_filter_settings
except ImportError:
    from qick_filter_settings import resolve_filter_settings


class FilterResponsePlot(QtWidgets.QWidget):
    """Schematic response and movable frequency markers on a fixed 0..8 GHz axis."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self._drag = None
        self.setMinimumSize(280, 235)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setAccessibleName("RF filter response and draggable frequency markers")

    def sizeHint(self):
        return QtCore.QSize(460, 235)

    def plot_rect(self):
        return QtCore.QRectF(28, 35, max(1, self.width() - 56), self.height() - 82)

    def frequency_x(self, frequency):
        rect = self.plot_rect()
        return rect.left() + frequency * rect.width() / 8

    def marker_positions(self):
        """Real event hit positions, also used by offscreen interaction tests."""
        editor, rect = self.editor, self.plot_rect()
        markers = []
        if editor.available and editor.mode != "bypass":
            for key, value in editor.edges().items():
                if 0 <= value <= 8:
                    markers.append((key, QtCore.QPointF(self.frequency_x(value), rect.top()), None))
        for frequency, k, sign in editor.image_frequencies():
            markers.append(("tone", QtCore.QPointF(self.frequency_x(frequency), rect.top() + 58), (k, sign)))
        return markers

    def response(self, frequency):
        editor = self.editor
        if not editor.available or editor.mode == "bypass":
            return 1.0
        edges = editor.edges()
        width = min(.2, max(.001, (edges.get("high", 8) - edges.get("low", 0)) * .12))

        def rise(cutoff):
            u = min(1., max(0., (frequency - cutoff + width) / (2 * width)))
            return u * u * (3 - 2 * u)

        return ((rise(edges["low"]) if "low" in edges else 1.)
                * (1 - rise(edges["high"]) if "high" in edges else 1.))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        palette, rect, editor = self.palette(), self.plot_rect(), self.editor
        foreground = palette.color(QtGui.QPalette.Text)
        muted = palette.color(QtGui.QPalette.Mid)
        blue = palette.color(QtGui.QPalette.Highlight)
        orange, green = QtGui.QColor(193, 108, 30), QtGui.QColor(64, 154, 109)
        painter.fillRect(self.rect(), palette.brush(QtGui.QPalette.Base))
        painter.setPen(foreground)
        title = (f"Center {editor.cutoff.value():.3f} GHz" if editor.mode == "bandpass"
                 and editor.available else "Filter response · schematic")
        painter.drawText(QtCore.QRectF(0, 0, self.width(), 25), QtCore.Qt.AlignCenter, title)
        painter.save()
        painter.setClipRect(rect)
        if editor.sample_rate_ghz:
            # Do not substitute DDS or decimated FIR rate for converter Fs.
            step = editor.sample_rate_ghz / 2
            for index in range(min(ceil(8 / step), 64)):
                if index % 2:
                    painter.fillRect(QtCore.QRectF(self.frequency_x(index * step), rect.top(),
                                                  step * rect.width() / 8, rect.height()),
                                     palette.brush(QtGui.QPalette.AlternateBase))
        path = QtGui.QPainterPath()
        for index in range(401):
            frequency = index * 8 / 400
            point = QtCore.QPointF(self.frequency_x(frequency),
                                  rect.bottom() - self.response(frequency) * (rect.height() - 24))
            path.moveTo(point) if index == 0 else path.lineTo(point)
        fill = QtGui.QPainterPath(path)
        fill.lineTo(rect.bottomRight())
        fill.lineTo(rect.bottomLeft())
        fill.closeSubpath()
        shade = QtGui.QColor(green)
        shade.setAlpha(32)
        painter.fillPath(fill, shade)
        painter.setPen(QtGui.QPen(green, 2))
        painter.drawPath(path)
        if editor.available and editor.mode == "bandpass":
            painter.setPen(QtGui.QPen(green, 1, QtCore.Qt.DashDotLine))
            x = self.frequency_x(editor.cutoff.value())
            painter.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))
        if editor.available and editor.resolved_settings is not None:
            faint = QtGui.QColor(foreground)
            faint.setAlpha(85)
            painter.setPen(QtGui.QPen(faint, 1.3, QtCore.Qt.DotLine))
            for value in editor.resolved_settings.nominal_cutoffs_ghz().values():
                x = self.frequency_x(value)
                painter.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))
        for kind, point, image in self.marker_positions():
            if kind == "tone":
                primary = image == (0, 1)
                color = blue if primary else QtGui.QColor(155, 104, 190)
                painter.setPen(QtGui.QPen(color, 2))
                painter.drawLine(point, QtCore.QPointF(point.x(), rect.bottom()))
                painter.setBrush(color)
                painter.drawEllipse(point, 4, 4)
            else:
                painter.setPen(QtGui.QPen(orange, 1.7, QtCore.Qt.DashLine))
                painter.drawLine(point, QtCore.QPointF(point.x(), rect.bottom()))
                painter.setBrush(orange)
                painter.drawPolygon(QtGui.QPolygonF([
                    point + QtCore.QPointF(-6, 0), point + QtCore.QPointF(6, 0),
                    point + QtCore.QPointF(0, 11)]))
        painter.restore()
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(muted)
        painter.drawRect(rect)
        painter.setPen(foreground)
        for frequency in ((0, 4, 8) if self.width() < 400 else (0, 2, 4, 6, 8)):
            x = self.frequency_x(frequency)
            painter.drawText(QtCore.QRectF(x - 15, rect.bottom() + 4, 30, 20),
                             QtCore.Qt.AlignCenter, str(frequency))
        painter.drawText(QtCore.QRectF(0, self.height() - 22, self.width(), 20),
                         QtCore.Qt.AlignCenter, "Frequency (GHz)")

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isEnabled():
            hits = [(abs(point.x() - event.localPos().x()) + abs(point.y() - event.localPos().y()),
                     kind, image) for kind, point, image in self.marker_positions()
                    if abs(point.x() - event.localPos().x()) <= 14
                    and abs(point.y() - event.localPos().y()) <= 18]
            if hits:
                _, kind, image = min(hits, key=lambda hit: hit[0])
                self._drag = (kind, image)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag and self.isEnabled():
            rect = self.plot_rect()
            frequency = min(8., max(0., (event.localPos().x() - rect.left()) * 8 / rect.width()))
            kind, image = self._drag
            if kind == "tone":
                k, sign = image
                self.editor.target.setValue(sign * (frequency - k * (self.editor.sample_rate_ghz or 0)))
            else:
                self.editor.drag_edge(kind, frequency)
            event.accept()
            return
        hit = any(abs(point.x() - event.localPos().x()) < 14
                  and abs(point.y() - event.localPos().y()) < 18
                  for _, point, _ in self.marker_positions())
        self.setCursor(QtCore.Qt.SizeHorCursor if hit else QtCore.Qt.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._drag = None
        super().mouseReleaseEvent(event)


class QickFilterEditor(QtWidgets.QWidget):
    """Keep legacy fc/BW controls as the API; edit cutoff edges in the view."""

    settings_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.available = True
        self.sample_rate_ghz = None
        self.resolved_settings = None
        self.error = ""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(("bypass", "lowpass", "highpass", "bandpass"))
        layout.addWidget(self.mode_combo)
        # These retain the old attribute/signal API and exact loaded values.
        self.cutoff = self._spin(2.5, self)
        self.bandwidth = self._spin(1., self)
        self.cutoff.hide()
        self.bandwidth.hide()
        self.low, self.high = self._spin(2.), self._spin(3.)
        self.low_label, self.high_label = QtWidgets.QLabel(), QtWidgets.QLabel()
        form = QtWidgets.QGridLayout()
        for column, (label, widget) in enumerate(((self.low_label, self.low), (self.high_label, self.high))):
            form.addWidget(label, 0, column)
            form.addWidget(widget, 1, column)
        layout.addLayout(form)
        target_row = QtWidgets.QHBoxLayout()
        target_row.addWidget(QtWidgets.QLabel("Target frequency:"))
        self.target = self._spin(1.2)
        self.target.setRange(0, 100)
        target_row.addWidget(self.target)
        layout.addLayout(target_row)
        purpose = QtWidgets.QLabel("Frequency for visualization")
        layout.addWidget(purpose)
        self.fs_label = QtWidgets.QLabel()
        layout.addWidget(self.fs_label)
        self.plot = FilterResponsePlot(self)
        layout.addWidget(self.plot)
        legend = QtWidgets.QLabel(
            "Orange dashed: requested cutoff · Faint dotted: state cutoff estimate\n"
            "Blue: target · Purple: images · Green: schematic response")
        legend.setWordWrap(True)
        layout.addWidget(legend)
        self.state_label, self.status = QtWidgets.QLabel(), QtWidgets.QLabel()
        self.state_label.setWordWrap(True)
        self.status.setWordWrap(True)
        layout.addWidget(self.state_label)
        layout.addWidget(self.status)
        for widget in (self.cutoff, self.bandwidth):
            widget.valueChanged.connect(self._from_api)
        self.mode_combo.currentIndexChanged.connect(self._from_api)
        self.low.valueChanged.connect(self._from_edges)
        self.high.valueChanged.connect(self._from_edges)
        self.target.valueChanged.connect(lambda *_args: self.plot.update())
        self._from_api()
        self.set_sample_rate(None)

    @staticmethod
    def _spin(value, parent=None):
        widget = QtWidgets.QDoubleSpinBox(parent)
        widget.setRange(-200, 200)
        widget.setDecimals(6)
        widget.setSingleStep(.01)
        widget.setSuffix(" GHz")
        widget.setKeyboardTracking(False)
        widget.setValue(value)
        return widget

    @property
    def mode(self):
        return self.mode_combo.currentText()

    def set_available(self, available):
        self.available = bool(available)
        self.setEnabled(self.available)
        self._from_api()

    def set_sample_rate(self, rate_ghz):
        self.sample_rate_ghz = (float(rate_ghz) if rate_ghz is not None
                                and isfinite(rate_ghz) and rate_ghz > 0 else None)
        self.fs_label.setText(
            f"Converter Fs: {self.sample_rate_ghz:g} GS/s"
            if self.sample_rate_ghz else "Converter Fs unavailable · images hidden")
        self.plot.update()

    def edges(self):
        fc, bw = self.cutoff.value(), self.bandwidth.value()
        if self.mode == "lowpass":
            return {"high": fc}
        if self.mode == "highpass":
            return {"low": fc}
        if self.mode == "bandpass":
            return {"low": fc - bw / 2, "high": fc + bw / 2}
        return {}

    def image_frequencies(self):
        f, fs = self.target.value(), self.sample_rate_ghz
        found = [(f, 0, 1)] if 0 <= f <= 8 else []
        if fs:
            # Frequencies stay in 0..8 even when the target is above that range.
            for sign in (1, -1):
                first, last = ceil(-sign * f / fs), int((8 - sign * f) // fs)
                for k in range(first, min(last + 1, first + 128)):
                    value = k * fs + sign * f
                    if not any(abs(value - item[0]) < 1e-9 for item in found):
                        found.append((value, k, sign))
        return sorted(found)

    def drag_edge(self, kind, frequency):
        limits = (1.750001, 19.899999) if kind == "low" else (2.050001, 18.849999)
        value = min(limits[1], max(limits[0], frequency))
        if self.mode == "bandpass":
            value = min(value, self.high.value() - .001) if kind == "low" else max(value, self.low.value() + .001)
        (self.low if kind == "low" else self.high).setValue(value)

    def _from_edges(self, *_args):
        with QtCore.QSignalBlocker(self.cutoff), QtCore.QSignalBlocker(self.bandwidth):
            if self.mode == "bandpass":
                self.cutoff.setValue((self.low.value() + self.high.value()) / 2)
                self.bandwidth.setValue(self.high.value() - self.low.value())
            elif self.mode == "lowpass":
                self.cutoff.setValue(self.high.value())
            elif self.mode == "highpass":
                self.cutoff.setValue(self.low.value())
        self._from_api()

    def _from_api(self, *_args):
        edges = self.edges()
        for key, widget, label in (("low", self.low, self.low_label), ("high", self.high, self.high_label)):
            widget.setVisible(key in edges)
            label.setVisible(key in edges)
            if key in edges:
                with QtCore.QSignalBlocker(widget):
                    widget.setValue(edges[key])
            label.setText(("Low cutoff · HPF" if key == "low" else "High cutoff · LPF")
                          if self.mode == "bandpass" else "Cutoff frequency")
        try:
            self.resolved_settings = resolve_filter_settings(self.mode, self.cutoff.value(), self.bandwidth.value())
            self.error = ""
        except ValueError as error:
            self.resolved_settings, self.error = None, str(error)
        if not self.available:
            self.state_label.setText("Analog filter unavailable on this port")
            self.status.setText("")
        elif self.resolved_settings:
            estimates = self.resolved_settings.nominal_cutoffs_ghz()
            self.state_label.setText(self.resolved_settings.label + ("\n" if estimates else "")
                                     + "; ".join(f"{name} cutoff ≈ {value:.3f} GHz" for name, value in estimates.items()))
            self.status.setText("Cutoff estimates are not measured values. Dragging does not apply hardware settings.")
        else:
            self.state_label.setText("Filter settings invalid")
            self.status.setText(self.error)
        self.plot.update()
        self.settings_changed.emit()
