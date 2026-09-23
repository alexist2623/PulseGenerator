"""GUI editors for the autonomous square DDS and external output markers."""
from dataclasses import asdict
from PyQt5 import QtCore, QtWidgets
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

    def __init__(self,parent=None):
        super().__init__(parent)
        layout=QtWidgets.QVBoxLayout(self)
        self.enabled=QtWidgets.QCheckBox("Enable SquarePulse during AWG experiment")
        self.status=QtWidgets.QLabel("Identify QICK to discover SquarePulse outputs.")
        self.status.setWordWrap(True)
        layout.addWidget(self.enabled); layout.addWidget(self.status)
        form=QtWidgets.QFormLayout(); layout.addLayout(form)
        self.channel=QtWidgets.QSpinBox(); self.channel.setRange(0,255); self.channel.setValue(7)
        form.addRow("QICK generator",self.channel)
        note=QtWidgets.QLabel("Continuous 50% duty output, ±amplitude. Output full scale comes from Experiment. "
            "Frequency and amplitude updates preserve accumulated phase. Phase sets an offset. "
            "Sweeps run in tProcessor hardware loops. The output is muted when the experiment finishes.")
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
                    parameters={name:{key:(widget.isChecked() if key=='sweep' else widget.value())
                                      for key,widget in row.items()} for name,row in self.rows.items()})

    def load_settings(self,settings):
        settings=settings or {}
        self.channel.setValue(int(settings.get('gen_ch',7)))
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
        config=SquarePulseConfig(ch,value('frequency'),value('amplitude'),value('phase'),full_scale_mv)
        axes=self.sweep_specs()
        return attach_square_settings(sequence,config,axes,trigger)

    def sweep_specs(self):
        if not self.enabled.isChecked(): return ()
        ch=self.channel.value()
        return tuple(SquarePulseSweep(name,row['start'].value(),row['stop'].value(),row['count'].value(),ch,
                                   output_name=f"square{ch}",segment_name=name)
                   for name,row in self.rows.items() if row['sweep'].isChecked())


class TriggeringPanel(QtWidgets.QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self._available_pins=None
        layout=QtWidgets.QFormLayout(self)
        self.enabled=QtWidgets.QCheckBox("Enable external output trigger")
        self.pin=QtWidgets.QSpinBox(); self.pin.setRange(0,255)
        self.scope=QtWidgets.QComboBox()
        self.scope.addItem("Each repetition loop","loop"); self.scope.addItem("Entire experiment","experiment")
        self.edge=QtWidgets.QComboBox()
        for label,value in (("Start","start"),("End","end"),("Start and end","both")): self.edge.addItem(label,value)
        self.width=spin(1.0,' µs',0.001,1e6,6)
        self.status=QtWidgets.QLabel("Output pin numbers follow the loaded firmware. Identify QICK to show available pins.")
        self.status.setWordWrap(True)
        for label,widget in (("",self.enabled),("Output pin",self.pin),("Scope",self.scope),
                             ("Boundary",self.edge),("Width",self.width),("",self.status)):
            layout.addRow(label,widget)

    def config(self):
        return OutputTriggerConfig(self.enabled.isChecked(),self.pin.value(),self.scope.currentData(),self.edge.currentData(),self.width.value())

    def settings_dict(self): return asdict(self.config())

    def load_settings(self,settings):
        config=OutputTriggerConfig(**(settings or {}))
        supported=self._available_pins is None or bool(self._available_pins)
        self.enabled.setChecked(config.enabled and supported); self.pin.setValue(config.pin)
        self.scope.setCurrentIndex(self.scope.findData(config.scope)); self.edge.setCurrentIndex(self.edge.findData(config.edge))
        self.width.setValue(config.width_us)

    def set_configuration(self,configuration):
        pins=tuple(getattr(configuration,'output_trigger_pins',()))
        self._available_pins=pins
        self.status.setText("Available outputs: "+', '.join(f"{i}: {pin}" for i,pin in enumerate(pins)) if pins else "No external trigger outputs in this firmware.")
        self.enabled.setEnabled(bool(pins))
        if not pins: self.enabled.setChecked(False)
