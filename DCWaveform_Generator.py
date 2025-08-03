#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DC-Waveform pulse-sequence generator (flat + rising segments)"""

import json
import sys
from typing import Tuple, Optional, List, Any, Callable

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes

class PulseSequence:
    """Piece-wise-linear voltage waveform."""

    def __init__(self, initial_voltage: float = 0.0):
        self.t = np.array([0.0, 100])
        self.v = np.array([initial_voltage, initial_voltage])
        self.v_bounds = (-2500, 2500)  # default voltage bounds in mV
        self.flat = []

    def edit_ramp(self, flat_idx: int, new_ramp: float) -> bool:
        """Edit the ramp segment at index flat_idx to have a new duration new_ramp."""
        if new_ramp < 0:
            return False
        if flat_idx == 0:
            return False
        i_ramp   = flat_idx - 1
        i_flat   = flat_idx + 1
        if i_ramp is None:                      # first flat → ramp length is implicit
            return False
        t_left   = self.t[i_ramp]
        t_right  = self.t[i_flat]               # start of flat
        delta    = new_ramp - (self.t[flat_idx] - self.t[i_ramp])
        if t_left + new_ramp >= t_right:        # would cross neighbour
            return False
        self.t[flat_idx:] += delta      # move ramp end
        return True

    def edit_flat(self, flat_idx: int, new_flat: float) -> bool:
        """Edit the flat segment at index flat_idx to have a new duration new_flat."""
        if new_flat < 0:
            return False
        i_flat_end = flat_idx + 1
        t_start    = self.t[flat_idx]
        delta      = new_flat - (self.t[i_flat_end] - t_start)
        if delta == 0:
            return True
        # shift all subsequent points
        self.t[i_flat_end:] += delta
        return True

    def edit_voltage(self, flat_idx: int, new_v: float) -> bool:
        """Edit the voltage of the flat segment at index flat_idx."""
        new_v = np.clip(new_v, self.v_bounds[0], self.v_bounds[1])
        self.v[flat_idx:flat_idx+2] = new_v
        return True

    def add_flat_ramp(self, ramp: float, flat: float, target_v: float) -> None:
        """Add a new segment with a ramp and a flat part."""
        if ramp < 0 or flat < 0:
            raise ValueError("Times must be non-negative.")
        t0, v0 = self.t[-1], self.v[-1]

        if ramp > 0:
            self.t = np.append(self.t, t0 + ramp)
            self.v = np.append(self.v, target_v)
        else:
            self.v[-1] = target_v

        if flat > 0:
            self.t = np.append(self.t, self.t[-1] + flat)
            self.v = np.append(self.v, target_v)

    def flat_segments(self) -> List[Tuple[int, int]]:
        """Return a list of index ranges [i0, i1] that are perfectly flat."""
        flat = []
        i = 0
        while i < len(self.t) - 1:
            if i % 2 == 0:
                flat.append((i, i + 1))
            i += 1
        return flat

    def update_flat(self, rng: Tuple[int, int], new_v: float) -> None:
        """Update the voltage of a flat segment."""
        i0, i1 = rng
        self.v[i0 : i1 + 1] = new_v

    def update_point(self, rng: Tuple[int, int], new_t: float) -> None:
        """Update the voltage of a flat segment."""
        i0, _ = rng

        if i0 == 0:
            return
        elif self.t[i0-1] >= new_t + 10e-9:
            self.t[i0] = self.t[i0-1] + 10e-9
            return
        for i in range(i0+1, len(self.t)):
            self.t[i] = self.t[i] + (new_t - self.t[i0])
        self.t[i0] = new_t

    def to_qcs_dcwaveform(self) -> str:
        """Return QCS DCWaveform Generation Code"""
        chan = "ch_0"
        code_str = (
            "import keysights.qcs as qcs\n"
            "\n"
            "ns              = 1e-9\n"
            "V_max_M5301AWG  = 5\n"
            "mV              = 1/(V_max_M5301AWG * 1000)\n"
        )
        for i in range(len(self.t)-1):
            if i % 2 == 0:
                code_str += (
                    f"dc_segment_{i} = qcs.DCWaveform(\n"
                    f"    duration    = {self.t[i+1] - self.t[i]} * ns,\n"
                    f"    envelope    = qcs.ConstantEnvelope(),\n"
                    f"    amplitude   = {self.v[i]} * mV\n"
                    f")\n\n"
                    f"program.add_waveform(dc_segment_{i}, {chan})\n"
                )
            else:
                code_str += (
                    f"dc_segment_{i}_p = qcs.DCWaveform(\n"
                    f"    duration    = {self.t[i+1] - self.t[i]} * ns,\n"
                    f"    envelope    = qcs.ArbitraryEnvelope(\n"
                    f"        [0,1], [0,1]\n"
                    f"    ),\n"
                    f"    amplitude   = {self.v[i+1]} * mV\n"
                    f")\n\n"
                )
                code_str += (
                    f"dc_segment_{i}_n = qcs.DCWaveform(\n"
                    f"    duration    = {self.t[i+1] - self.t[i]} * ns,\n"
                    f"    envelope    = qcs.ArbitraryEnvelope(\n"
                    f"        [0,1], [1,0]\n"
                    f"    ),\n"
                    f"    amplitude   = {self.v[i]} * mV\n"
                    f")\n\n"
                    f"dc_segment_{i} = dc_segment_{i}_n + dc_segment_{i}_p\n"
                    f"program.add_waveform(dc_segment_{i}, {chan})\n"
                )

        return code_str

class MatplotWidget(Canvas): # pylint: disable=too-many-instance-attributes
    """Matplotlib widget for displaying and editing a PulseSequence."""
    flat_moved = QtCore.pyqtSignal(int, int, float)   # i0, i1, new_v
    point_moved = QtCore.pyqtSignal(int, int, float)  # i0, i1, new_v

    def __init__(
            self,
            pulse: PulseSequence,
            parent=None
        ):
        fig                 = Figure(tight_layout=False)
        super().__init__(fig)
        self._pulse: List[PulseSequence] = [pulse]
        self._line: List[Axes] = []
        self._selected_port_idx = 0

        self.ax             = fig.add_subplot(111)
        self._line,         = [self.ax.plot(self._pulse[0].t, self._pulse[0].v, "-o", picker=5)]
        self.ax.set_xlabel("time [ns]")
        self.ax.set_ylabel("voltage [mV]")
        self.ax.grid(True)
        self.ax.set_autoscale_on(False)

        self._drag_flat: Optional[Tuple[int, int]] = None
        self._drag_point: Optional[Tuple[int, int]] = None
        self._pan_origin: Optional[Tuple[float, float]] = None
        self._annot         = self.ax.annotate(
            "",
            xy              = (0, 0),
            xytext          = (12, 12),
            textcoords      = "offset points",
            bbox            = dict(boxstyle="round", fc="w"),
            arrowprops      = dict(arrowstyle="->"),
            visible         = False
        )

        self.mpl_connect("motion_notify_event",  self._on_move)
        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("scroll_event",         self._on_scroll)

        self.fit_view()

    def fit_view(self):
        """Fit the view to the pulse data."""
        margin_x = 0.03 * (self._pulse[0].t.max() - self._pulse[0].t.min() + 1e-12)
        margin_y = 0.10 * (self._pulse[0].v.ptp() + 1e-12) or 0.5
        x_min = self._pulse[0].t.min() - margin_x
        x_max = self._pulse[0].t.max() + margin_x
        y_min = self._pulse[0].v.min() - margin_y
        y_max = self._pulse[0].v.max() + margin_y

        for pulse in self._pulse:
            if margin_x < 0.03 * (pulse.t.max() - pulse.t.min() + 1e-12):
                margin_x = 0.03 * (pulse.t.max() - pulse.t.min() + 1e-12)
            if margin_y < 0.10 * (pulse.v.ptp() + 1e-12):
                margin_y = 0.10 * (pulse.v.ptp() + 1e-12)
            if x_min > pulse.t.min() - margin_x:
                x_min = pulse.t.min() - margin_x
            if x_max < pulse.t.max() + margin_x:
                x_max = pulse.t.max() + margin_x
            if y_min > pulse.v.min() - margin_y:
                y_min = pulse.v.min() - margin_y
            if y_max < pulse.v.max() + margin_y:
                y_max = pulse.v.max() + margin_y
        self.ax.set_xlim(
            x_min,
            x_max
        )
        self.ax.set_ylim(
            y_min,
            y_max
        )
        self.draw_idle()

    def refresh(self):
        """Refresh plot"""
        for i, pulse in enumerate(self._pulse):
            self._line[i].set_data(pulse.t, pulse.v)
        self.draw_idle()
    
    def add_pulse(self, pulse: PulseSequence):
        """Add a new pulse to the plot."""
        self._pulse.append(pulse)
        line,                   = self.ax.plot(pulse.t, pulse.v, "-o", picker=5)
        self._line.append(line)
        self._selected_port_idx = len(self._pulse) - 1
        self.refresh()
        self.fit_view()
    
    def get_selected_port_idx(self) -> int:
        return self._selected_port_idx
    
    def set_selected_port_idx(self, idx: int):
        self._selected_port_idx = idx

    def _locate_flat_segment(self, event) -> Optional[Tuple[int, int]]:
        """Return range (i0,i1) if click is on *flat* part, else None."""
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return None
        line_x, line_y = self._line[self._selected_port_idx].get_data()
        # Check Flat Click Events
        for i0 in range(len(self._pulse[self._selected_port_idx].t)):
            if i0 % 2 != 0:
                continue
            if i0 == len(self._pulse[self._selected_port_idx].t) - 1:
                break

            i1 = i0 + 1
            # mid-point of flat
            xm = 0.5 * (line_x[i0] + line_x[i1])
            ym = line_y[i0]
            dx = abs(x - xm)
            dy = abs(y - ym)
            if (
                dx < 0.25 * (line_x[i1] - line_x[i0])
                and dy < 0.05 * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
            ):
                return (i0, i1)
        return None

    def _locate_point_segment(self, event) -> Optional[Tuple[int, int]]:
        """Return range (i0,i1) if click is on *flat* part, else None."""
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return None
        line_x, line_y = self._line[self._selected_port_idx].get_data()
        # Check Point Click Events
        for i0 in range(len(self._pulse[self._selected_port_idx].t)):
            dx = abs(x - line_x[i0])
            dy = abs(y - line_y[i0])
            if (
                dx < 0.05 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
                and dy < 0.05 * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
            ):
                return (i0, i0+1)
        return None

    def _on_press(self, event):
        if (event.button != 1 or not event.inaxes):
            return

        seg = self._locate_flat_segment(event)
        point = self._locate_point_segment(event)
        if seg:
            self._drag_flat = seg
        elif point:
            self._drag_point = point
        else:
            self._pan_origin = (event.xdata, event.ydata)

    def _on_move(self, event):
        # tooltip
        self._annot.set_visible(False)

        # drag flat
        if self._drag_flat and event.ydata is not None:
            new_v = np.clip(
                event.ydata,
                self._pulse[self._selected_port_idx].v_bounds[0],
                self._pulse[self._selected_port_idx].v_bounds[1]
            )
            i0, i1 = self._drag_flat
            self._pulse[self._selected_port_idx].update_flat(self._drag_flat, new_v)
            self.flat_moved.emit(i0, i1, new_v)
            self.refresh()

            self._annot.xy = (
                self._pulse[self._selected_port_idx].t[i0],
                new_v
            )
            self._annot.set_text(
                f"{self._pulse[self._selected_port_idx].t[i0]:0.3g} ns\n{new_v:0.3g} mV"
            )
            self._annot.set_visible(True)

        # drag point
        if self._drag_point and event.ydata is not None:
            xmin, xmax = self.ax.get_xlim()
            new_t = np.clip(event.xdata, xmin, xmax)
            i0, i1 = self._drag_point
            self._pulse[self._selected_port_idx].update_point(self._drag_point, new_t)
            self.point_moved.emit(i0, i1, new_t)
            self.refresh()

            self._annot.xy = (
                new_t,
                self._pulse[self._selected_port_idx].v[i0]
            )
            self._annot.set_text(
                f"{new_t:0.3g} ns\n{self._pulse[self._selected_port_idx].v[i0]:0.3g} mV"
            )
            self._annot.set_visible(True)
        # pan
        if self._pan_origin and event.xdata and event.ydata:
            x0, y0 = self._pan_origin
            dx = x0 - event.xdata
            dy = y0 - event.ydata
            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            self.ax.set_xlim(xmin + dx, xmax + dx)
            self.ax.set_ylim(ymin + dy, ymax + dy)
            self.draw_idle()

    def _on_release(self, _event):
        self._drag_flat  = None
        self._drag_point = None
        self._pan_origin = None

    def _on_scroll(self, event):
        if not event.inaxes:
            return

        factor = -0.1 if event.step > 0 else +0.1
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        # Matplotlib ≥3.7: use event.modifiers; fall back to event.key otherwise
        mods = set(event.modifiers) if hasattr(event, "modifiers") else (
            {event.key} if event.key else set())

        if "ctrl" in mods:                  # Ctrl + wheel  → horizontal zoom
            new_w = (xmax - xmin) * factor
            self.ax.set_xlim(xmin - 0.5*new_w, xmax + 0.5*new_w)

        elif "shift" in mods:                  # Shift + wheel → vertical zoom
            new_h = (ymax - ymin) * factor
            self.ax.set_ylim(ymin - 0.5*new_h, ymax + 0.5*new_h)

        else:                                  # plain wheel    → isotropic zoom
            new_w = (xmax - xmin) * factor
            new_h = (ymax - ymin) * factor
            self.ax.set_xlim(xmin - 0.5*new_w, xmax + 0.5*new_w)
            self.ax.set_ylim(ymin - 0.5*new_h, ymax + 0.5*new_h)

        self.draw_idle()

class ControlPanel(QtWidgets.QWidget):
    """Input boxes plus a live table of all segments."""

    add_requested       = QtCore.pyqtSignal(float, float, float)
    update_plot         = QtCore.pyqtSignal()
    port_is_selected    = QtCore.pyqtSignal(int)
    port_idx: int       = 0

    def __init__(
            self,
            pulse: PulseSequence,
            parent=None
        ):
        super().__init__(parent)
        self._pulse         = pulse
        self.idx            = ControlPanel.port_idx
        ControlPanel.port_idx += 1

        v_splitter          = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        layout              = QtWidgets.QVBoxLayout(self)

        form                = QtWidgets.QFormLayout()
        self.edit_ramp      = QtWidgets.QLineEdit("50")
        self.edit_flat      = QtWidgets.QLineEdit("200")
        self.edit_v         = QtWidgets.QLineEdit("10")

        for w in (self.edit_ramp, self.edit_flat, self.edit_v):
            w.setValidator(QtGui.QDoubleValidator(decimals=9))

        form.addRow("Ramp [ns]:",  self.edit_ramp)
        form.addRow("Flat [ns]:",  self.edit_flat)
        form.addRow("V [mV]:",     self.edit_v)

        btn_add             = QtWidgets.QPushButton("Add segment")
        btn_add.clicked.connect(self._on_add)
        form.addRow(btn_add)

        btn_select_port     = QtWidgets.QPushButton("Select Port")
        btn_select_port.clicked.connect(self._select_port)
        form.addRow(btn_select_port)

        form_widget         = QtWidgets.QWidget()
        form_widget.setLayout(form)

        self.table          = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["#", "Ramp [ns]", "Flat [ns]", "V [mV]"]
        )
        self.table.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding
        )
        header              = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
        )
        self.table.itemChanged.connect(self._on_item_changed)

        v_splitter.addWidget(form_widget)
        v_splitter.addWidget(self.table)
        layout.addWidget(v_splitter)
        self._refresh_table()

        layout.addStretch(1)

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        row, col = item.row(), item.column()
        try:
            val = float(item.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Bad edit", "Enter a number")
            return

        # Map table row
        flat_idx = row * 2              # because every table row is (i,i+1) with i even
        if col == 1:
            ok = self._pulse.edit_ramp(flat_idx, val)
        if col == 2:
            ok = self._pulse.edit_flat(flat_idx, val)
        elif col == 3:
            ok = self._pulse.edit_voltage(flat_idx, val)
        else:
            ok = False
        self.update_plot.emit()

    def _on_add(self):
        self.port_is_selected.emit(self.idx)
        try:
            ramp = float(self.edit_ramp.text())
            flat = float(self.edit_flat.text())
            v    = float(self.edit_v.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "Bad input",
                "Please enter valid numbers."
            )
            return
        self.add_requested.emit(ramp, flat, v)

    def _refresh_table(self):
        with QtCore.QSignalBlocker(self.table):
            segs = self._pulse.flat_segments()
            self.table.setRowCount(len(segs))
            for row, (i0, i1) in enumerate(segs):
                ramp  = self._pulse.t[i0] - self._pulse.t[i0 - 1] if i0 > 0 else 0
                flat  = self._pulse.t[i1] - self._pulse.t[i0]
                v     = self._pulse.v[i0]
                for col, val in enumerate([row + 1, ramp, flat, v]):
                    item = QtWidgets.QTableWidgetItem(f"{val:.6g}")
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.table.setItem(row, col, item)
    
    def _select_port(self):
        """Emit signal to select this port."""
        self.port_is_selected.emit(self.idx)

class MultiControlPanel(QtWidgets.QWidget):
    """Multi waveform control panel"""

    def __init__(
            self,
            pulse: PulseSequence,
            initial_color: str,
            add_port: Callable,
            parent=None
        ):
        super().__init__()
        self._ctrl_pannels: List[ControlPanel]  = [ControlPanel(pulse)]
        self._color_map: List[str] = [initial_color]
        self.splitter                           = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)

        # Port Add Button
        btn_add_port                            = QtWidgets.QPushButton("Add Port")
        btn_add_port.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding
        )
        btn_widget                              = QtWidgets.QWidget()
        btn_layout                              = QtWidgets.QVBoxLayout(btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_widget.setFixedWidth(btn_add_port.sizeHint().width())
        btn_layout.addWidget(btn_add_port)
        btn_add_port.clicked.connect(add_port)

        self.splitter.addWidget(self._ctrl_pannels[0])
        self.splitter.addWidget(btn_widget)

        # Control Panels list
        self.panel_table                        = QtWidgets.QTableWidget(0, 4)
        self.panel_table.setHorizontalHeaderLabels(["#", "Color", "set_x", "set_y"])
        self.panel_table.verticalHeader().setVisible(False)
        self.panel_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.panel_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        panel_table_header = self.panel_table.horizontalHeader()
        panel_table_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        panel_table_header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        panel_table_header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        panel_table_header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)

        control_panel_v_splitter                = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        control_panel_v_splitter.addWidget(self.splitter)
        control_panel_v_splitter.addWidget(self.panel_table)

        layout                                  = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(control_panel_v_splitter)

        self._refresh_panel_table()

    def _refresh_table(self):
        """Refresh all control panels' tables."""
        for ctrl in self._ctrl_pannels:
            ctrl._refresh_table()
        self._refresh_panel_table()

    def _refresh_panel_table(self):
        self.panel_table.setRowCount(len(self._ctrl_pannels))
        for idx, ctrl in enumerate(self._ctrl_pannels):
            item_idx = QtWidgets.QTableWidgetItem(str(idx + 1))
            item_idx.setTextAlignment(QtCore.Qt.AlignCenter)
            self.panel_table.setItem(idx, 0, item_idx)

            color = self._color_map[idx]
            item_color = QtWidgets.QTableWidgetItem()
            item_color.setBackground(QtGui.QColor(color))
            self.panel_table.setItem(idx, 1, item_color)

            btn_x = QtWidgets.QPushButton("set_x")
            btn_x.clicked.connect(lambda _, i=idx: self._set_x(i))
            self.panel_table.setCellWidget(idx, 2, btn_x)

            btn_y = QtWidgets.QPushButton("set_y")
            btn_y.clicked.connect(lambda _, i=idx: self._set_y(i))
            self.panel_table.setCellWidget(idx, 3, btn_y)

    def _set_x(self, idx):
        QtWidgets.QMessageBox.information(self, "Set X", f"Set X for panel {idx+1}")

    def _set_y(self, idx):
        QtWidgets.QMessageBox.information(self, "Set Y", f"Set Y for panel {idx+1}")

class MainWindow(QtWidgets.QMainWindow):
    """Main window for the DCWaveform generator application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("QCS - DCWaveform generator")

        self._pulse: List[PulseSequence]    = [PulseSequence(0.0)]
        self._pulse[0].v_bounds             = (-2500, 2500)
        self._plot                          = MatplotWidget(self._pulse[0])
        self._selected_port_idx             = 0
        self._multi_ctrl: MultiControlPanel = MultiControlPanel(
            self._pulse[0],
            self._plot._line[0].get_color(),
            self._add_port
        )
        self._multi_ctrl._ctrl_pannels[0].add_requested.connect(self._add_segment)
        self._multi_ctrl._ctrl_pannels[0].update_plot.connect(self._plot_refresh)
        self._multi_ctrl._ctrl_pannels[0].port_is_selected.connect(self._port_select)

        splitter                            = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        splitter.addWidget(self._multi_ctrl)

        splitter.addWidget(self._plot)
        splitter.setStretchFactor(1, 3)
        self.setCentralWidget(splitter)

        # status bar
        self.statusBar().showMessage("Ready")

        self._plot.flat_moved.connect(self._flat_update)
        self._plot.point_moved.connect(self._point_update)

        self._build_menu()
        self._build_toolbar()

    def _build_menu(self):
        mb = self.menuBar()

        # File
        m_file = mb.addMenu("&File")
        m_file.addAction("Load JSON…",  self._load_json)
        m_file.addSeparator()
        m_file.addAction("E&xit",       self.close)

        # Pulse
        m_pulse = mb.addMenu("&Pulse")
        gen = m_pulse.addAction("&Generate QCS string…")
        gen.setShortcut(QtGui.QKeySequence("Ctrl+G"))
        gen.triggered.connect(self._generate_pulse)

    def _build_toolbar(self):
        tb = self.addToolBar("Tools")
        btn_fit = QtWidgets.QAction("Fit", self)
        btn_fit.setToolTip("Show the full pulse (keyboard: F)")
        btn_fit.triggered.connect(self._plot.fit_view)
        tb.addAction(btn_fit)
        # keyboard shortcut
        QtWidgets.QShortcut(QtGui.QKeySequence("F"), self, self._plot.fit_view)

    def _add_segment(self, ramp: float, flat: float, v: float) -> None:
        try:
            self._pulse[self._selected_port_idx].add_flat_ramp(
                ramp,
                flat,
                v
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid input", str(exc))
            return
        self._plot.refresh()
        self._plot.fit_view()
        self._multi_ctrl._refresh_table()

    def _flat_update(self, i0: int, i1: int, new_v: float) -> None:
        self.statusBar().showMessage(
            f"Flat {i0}-{i1} moved → {new_v:0.6g} V")
        self._multi_ctrl._refresh_table()

    def _point_update(self, i0, i1, new_t):
        self.statusBar().showMessage(
            f"Point {i0}-{i1} moved → {new_t:0.6g} V")
        self._multi_ctrl._refresh_table()

    def _load_json(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open settings JSON", "", "JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if "initial_voltage" in data:
                self._pulse[0].v[0] = float(data["initial_voltage"])
            if "voltage_bounds" in data and len(data["voltage_bounds"]) == 2:
                self._pulse[0].v_bounds = tuple(map(float, data["voltage_bounds"]))
                self._plot._pulse[0].v_bounds[0], self._plot._pulse[0].v_bounds[1] = self._pulse[0].v_bounds
            self._plot.refresh()
            self.ctrl._refresh_table()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Load failed", f"Could not read JSON:\n{exc}"
            )

    def _generate_pulse(self):
        txt = self._pulse.to_qcs_dcwaveform()
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle("Generated QCS pulse")
        dlg.setText("Copy the grammar below:")
        dlg.setDetailedText(txt)
        dlg.exec_()

    def _add_port(self):
        """Add a new control panel for a new PulseSequence."""
        new_pulse       = PulseSequence()
        self._plot.add_pulse(new_pulse)
        self._multi_ctrl._color_map.append(self._plot._line[-1].get_color())
        self._pulse.append(new_pulse)
        new_ctrl        = ControlPanel(new_pulse)
        new_ctrl.add_requested.connect(self._add_segment)
        new_ctrl.update_plot.connect(self._plot_refresh)
        new_ctrl.port_is_selected.connect(self._port_select)
        self._multi_ctrl._ctrl_pannels.append(new_ctrl)
        self._multi_ctrl.splitter.insertWidget(len(self._multi_ctrl._ctrl_pannels) - 1, new_ctrl)
        self._multi_ctrl._refresh_table()

    def _port_select(self, idx: int) -> None:
        """Activate the selected control panel."""
        self._selected_port_idx         = idx
        self._plot._selected_port_idx   = idx
        self._plot.refresh()

    def _plot_refresh(self):
        self._plot.refresh()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1500, 650)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
