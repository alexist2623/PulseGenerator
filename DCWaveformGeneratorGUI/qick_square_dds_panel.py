"""GUI editors for the autonomous square DDS and external output markers."""
from dataclasses import asdict
from PyQt5 import QtCore, QtWidgets
try:
    from .qick_square_output import SquareOutputSelector
    from .qick_front_panel import QickFrontPanelCanvas, digital_trigger_outputs
except ImportError:
    from qick_square_output import SquareOutputSelector
    from qick_front_panel import QickFrontPanelCanvas, digital_trigger_outputs
try:
    from .qick_square_dds import SquarePulseConfig, SquarePulseSweep, OutputTriggerConfig, attach_square_settings
except ImportError:
    from qick_square_dds import SquarePulseConfig, SquarePulseSweep, OutputTriggerConfig, attach_square_settings


def spin(value, suffix, minimum=0, maximum=1e6, decimals=6):
    widget=QtWidgets.QDoubleSpinBox()
    widget.setRange(minimum,maximum); widget.setDecimals(decimals)
    widget.setSuffix(suffix); widget.setValue(value)
    widget.setKeyboardTracking(False)
    return widget


class SquarePulsePanel(QtWidgets.QWidget):
    changed=QtCore.pyqtSignal()
    front_panel_requested=QtCore.pyqtSignal(object)

    def __init__(self,parent=None):
        super().__init__(parent)
        layout=QtWidgets.QVBoxLayout(self)
        self.enabled=QtWidgets.QCheckBox("Enable SquarePulse during AWG experiment")
        self.status=QtWidgets.QLabel("Identify QICK to discover SquarePulse outputs.")
        self.status.setWordWrap(True)
        layout.addWidget(self.enabled); layout.addWidget(self.status)
        self.output_selector=SquareOutputSelector(self, channel=7)
        self.channel=self.output_selector.channel
        self.output_selector.requested.connect(lambda: self.front_panel_requested.emit(self))
        layout.addWidget(self.output_selector)
        self.mute_on_finish=QtWidgets.QCheckBox("Mute SquarePulse when experiment finishes")
        self.mute_on_finish.setChecked(True)
        self.mute_on_finish.setToolTip("Unchecked: keep the final frequency, amplitude and phase running after successful completion. Stop/cancel and errors still mute output.")
        self.mute_on_finish.toggled.connect(self.changed)
        layout.addWidget(self.mute_on_finish)
        self.rc_enabled=QtWidgets.QCheckBox("RC compensation")
        self.rc_tau_us=spin(1000.0, " us", 10.0, 1_000_000.0)
        self.rc_tau_us.setEnabled(False)
        self.rc_enabled.toggled.connect(self.rc_tau_us.setEnabled)
        self.rc_enabled.toggled.connect(self.changed)
        self.rc_tau_us.valueChanged.connect(self.changed)
        layout.addWidget(self.rc_enabled)
        rc_form=QtWidgets.QFormLayout()
        rc_form.addRow("RC time constant (tau)", self.rc_tau_us)
        layout.addLayout(rc_form)
        note=QtWidgets.QLabel("Continuous 50% duty output, ±amplitude. Output full scale follows this DAC current in the shared Front Panel. "
            "Frequency and amplitude updates preserve accumulated phase. Phase sets an offset. "
            "Sweeps run in tProcessor hardware loops. Uncheck mute to keep output running after completion.")
        note.setWordWrap(True); layout.addWidget(note)
        self.rows={}
        for name,label,value,suffix,minimum in (
            ('frequency','Frequency',0.04,' MHz',0),
            ('amplitude','Peak amplitude',10,' mV',0),
            ('phase','Phase offset',0,' deg',-1e6)):
            group=QtWidgets.QGroupBox(label); row=QtWidgets.QGridLayout(group)
            fixed=spin(value,suffix,minimum); start=spin(value,suffix,minimum); stop=spin(value,suffix,minimum)
            sweep=QtWidgets.QCheckBox("Hardware sweep")
            count=QtWidgets.QSpinBox(); count.setRange(1,1000000); count.setValue(11)
            row.addWidget(QtWidgets.QLabel("Value"),0,0); row.addWidget(fixed,0,1); row.addWidget(sweep,0,2)
            for col,text,widget in ((0,'Start',start),(2,'Stop',stop),(4,'Points',count)):
                row.addWidget(QtWidgets.QLabel(text),1,col); row.addWidget(widget,1,col+1)
            self.rows[name]=dict(value=fixed,sweep=sweep,start=start,stop=stop,count=count)
            for widget in (start,stop,count): widget.setEnabled(False)
            sweep.toggled.connect(lambda checked,items=(fixed,start,stop,count): self._toggle_sweep(checked,items))
            for widget in (fixed,start,stop,count): widget.valueChanged.connect(self.changed)
            sweep.toggled.connect(self.changed)
            layout.addWidget(group)
        self.enabled.toggled.connect(self.changed); self.channel.valueChanged.connect(self.changed)
        layout.addStretch(1)
        self._available_channels=None

    @staticmethod
    def _toggle_sweep(checked,items):
        items[0].setEnabled(not checked)
        for item in items[1:]: item.setEnabled(checked)

    def set_configuration(self,configuration):
        self.output_selector.set_configuration(configuration)
        channels=tuple(getattr(configuration,'square_pulse_channels',()))
        self._available_channels=channels
        self.enabled.setEnabled(bool(channels))
        if channels:
            if self.channel.value() not in channels: self.channel.setValue(channels[0])
            self.status.setText("SquarePulse firmware detected; generator(s): "+', '.join(map(str,channels)))
        else:
            self.enabled.setChecked(False)
            self.status.setText("This firmware has no SquarePulse IP. Existing AWG and RF features remain available.")

    def settings_dict(self):
        return dict(enabled=self.enabled.isChecked(),gen_ch=self.channel.value(),
                    mute_on_finish=self.mute_on_finish.isChecked(),
                    rc_enabled=self.rc_enabled.isChecked(), rc_tau_us=self.rc_tau_us.value(),
                    parameters={name:{key:(widget.isChecked() if key=='sweep' else widget.value())
                                      for key,widget in row.items()} for name,row in self.rows.items()})

    def load_settings(self,settings):
        settings=settings or {}
        self.output_selector.load_channel(int(settings.get('gen_ch',7)), explicit=bool(settings))
        self.mute_on_finish.setChecked(settings.get('mute_on_finish',True))
        self.rc_enabled.setChecked(settings.get('rc_enabled',False))
        self.rc_tau_us.setValue(settings.get('rc_tau_us',1000.0))
        supported = self._available_channels is None or bool(self._available_channels)
        self.enabled.setChecked(bool(settings.get('enabled',False)) and supported)
        for name,row in self.rows.items():
            for key,value in settings.get('parameters',{}).get(name,{}).items():
                if key not in row: continue
                if key=='sweep': row[key].setChecked(bool(value))
                else: row[key].setValue(value)

    def attach_to_sequence(self,sequence,full_scale_mv,trigger):
        if not self.enabled.isChecked():
            return attach_square_settings(sequence,trigger=trigger)
        ch=self.channel.value()
        if self._available_channels is not None and ch not in self._available_channels:
            raise ValueError("Select a SquarePulse generator from the identified firmware")
        value=lambda key:self.rows[key]['start' if self.rows[key]['sweep'].isChecked() else 'value'].value()
        config=SquarePulseConfig(ch,value('frequency'),value('amplitude'),value('phase'),
                                 full_scale_mv,self.mute_on_finish.isChecked(), self.rc_enabled.isChecked(), self.rc_tau_us.value())
        axes=self.sweep_specs()
        return attach_square_settings(sequence,config,axes,trigger)

    def sweep_specs(self):
        if not self.enabled.isChecked(): return ()
        ch=self.channel.value()
        return tuple(SquarePulseSweep(name,row['start'].value(),row['stop'].value(),row['count'].value(),ch,
                                   output_name=f"square{ch}",segment_name=name)
                   for name,row in self.rows.items() if row['sweep'].isChecked())

    def allowed_output_channels(self, configuration):
        return self.output_selector.allowed_channels(configuration)

    def front_panel_values(self):
        return dict(output_ch=self.channel.value(), output_nqz=1)

    def apply_front_panel_settings(self, values):
        self.output_selector.apply(values)


class TriggeringPanel(QtWidgets.QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self._configuration = None
        self._available_pins = None
        self._io_to_pin = {}
        self._saved_pin = 0
        self._selected_name = None
        layout=QtWidgets.QFormLayout(self)
        self.enabled=QtWidgets.QCheckBox("Enable external output trigger")
        self.enabled.setEnabled(False)
        self.canvas = QickFrontPanelCanvas(self)
        self.canvas.set_scope("io")
        self.canvas.setMinimumSize(330, 122)
        self.canvas.setMaximumHeight(200)
        self.canvas.port_clicked.connect(self._select_sma)
        self.pin = QtWidgets.QComboBox()
        self.pin.setPlaceholderText("Identify QICK, then select a connected SMA")
        self.pin.setEnabled(False)
        self.pin.currentIndexChanged.connect(self._selection_changed)
        self.scope=QtWidgets.QComboBox()
        self.scope.addItem("Each repetition loop","loop"); self.scope.addItem("Entire experiment","experiment")
        self.edge=QtWidgets.QComboBox()
        for label,value in (("Start","start"),("End","end"),("Start and end","both")): self.edge.addItem(label,value)
        self.width=spin(1.0,' µs',0.001,1e6,6)
        self.status=QtWidgets.QLabel("Identify QICK to discover connected trigger outputs. Gray SMAs cannot be selected.")
        self.status.setWordWrap(True)
        for label,widget in (("",self.enabled),("",self.canvas),("Output SMA",self.pin),("Scope",self.scope),
                             ("Boundary",self.edge),("Width",self.width),("",self.status)):
            layout.addRow(label,widget)

    def config(self):
        pin = self.pin.currentData()
        valid = self._available_pins is None or pin in self._io_to_pin.values()
        return OutputTriggerConfig(self.enabled.isChecked() and valid,
                                   self._saved_pin if pin is None else pin,
                                   self.scope.currentData(),self.edge.currentData(),self.width.value())

    def settings_dict(self): return asdict(self.config())

    def load_settings(self,settings):
        config=OutputTriggerConfig(**(settings or {}))
        self._saved_pin = config.pin
        self._selected_name = None
        self.pin.setCurrentIndex(self.pin.findData(config.pin))
        self._selection_changed()
        supported = self._available_pins is None or config.pin in self._io_to_pin.values()
        self.enabled.setChecked(config.enabled and supported)
        self.scope.setCurrentIndex(self.scope.findData(config.scope)); self.edge.setCurrentIndex(self.edge.findData(config.edge))
        self.width.setValue(config.width_us)

    def set_configuration(self,configuration):
        self._configuration = configuration
        self._available_pins = tuple(getattr(configuration, 'output_trigger_pins', ()))
        self._io_to_pin = digital_trigger_outputs(configuration)
        self.canvas.set_configuration(configuration)
        preferred = self._saved_pin
        if self._selected_name is not None:
            # Keep the same physical output even if output_pins is reordered.
            preferred = next((pin for pin in self._io_to_pin.values()
                              if self._available_pins[pin] == self._selected_name), None)
        blocker = QtCore.QSignalBlocker(self.pin)
        self.pin.clear()
        for sma, pin in sorted(self._io_to_pin.items()):
            self.pin.addItem(f"IO{sma} — {self._available_pins[pin]} (QICK pin {pin})", pin)
        self.pin.setCurrentIndex(self.pin.findData(preferred))
        del blocker
        self.pin.setEnabled(bool(self._io_to_pin))
        self.pin.setPlaceholderText("Select a connected SMA" if self._io_to_pin else "No connected trigger SMA")
        self._selection_changed()

    def _select_sma(self, direction, sma):
        if direction == "io" and sma in self._io_to_pin:
            self.pin.setCurrentIndex(self.pin.findData(self._io_to_pin[sma]))

    def _selection_changed(self, *_args):
        pin = self.pin.currentData()
        sma = next((sma for sma, index in self._io_to_pin.items() if index == pin), None)
        self.canvas.set_selected("io", sma)
        self.enabled.setEnabled(sma is not None)
        if sma is not None:
            self._saved_pin = pin
            self._selected_name = self._available_pins[pin]
            self.status.setText(f"Selected: IO{sma} → {self._selected_name}. "
                                "Gray SMAs have no connected trigger output in this firmware.")
        elif self._available_pins is not None:
            self.enabled.setChecked(False)
            self.status.setText(
                "The saved output is unavailable. Select a connected DIGITAL I/O SMA."
                if self._io_to_pin else "No DIGITAL I/O SMA is connected to a supported tProcessor trigger output in this firmware."
            )
