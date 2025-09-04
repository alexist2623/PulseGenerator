# -*- coding: utf-8 -*-
""" Dot array for QSTL"""

import logging
import threading
import json
import ast
import numpy as np
import qcodes
from functools import partial
from typing import Dict, Optional, Tuple
from qcodes import Instrument
from qcodes.utils.validators import Numbers

from qtt.instrument_drivers.virtual_gates import VirtualGates

logger = logging.getLogger(__name__)

class QSTLGate(Instrument):
    """
    Physical gate class. Note that setting voltage value does not update output voltage.
    """
    def __init__(
            self,
            name: str,
            gate_mapping: Dict[str, Tuple[int, int, int]], # Gate name : (chassis, slot, channel)
            **kwargs
        ):
        super().__init__(name, **kwargs)
        self.gates = [gate_name for gate_name, _ in gate_mapping.items()]
        self._state = {g: 0.0 for g in self.gates}
        for gate in self.gates:
            self.add_parameter(
                gate,
                label="%s" % gate,  # (\u03bcV)',
                unit="mV",
                get_cmd=partial(self._get, gate=gate),
                set_cmd=partial(self._set, gate=gate),
            )
    def _get(self, gate: str) -> float:
        return self._state[gate]

    def _set(self, value: float, gate: str) -> None:
        self._state[gate] = value
    
    def set_boundaries(self, boundaries : dict) -> None:
        """Set boundaries of physical gate"""
        for gate, bd in boundaries.items():
            self.parameters[gate].vals = Numbers(bd[0], bd[1])
    
    def ask_raw(self, cmd: str) -> str:
        """Dummy function"""
        if cmd == "*IDN?":
            return self.name

    def allvalues(self, get_latest=False):
        """Return all virtual gate voltage values in a dict."""
        if get_latest:
            vals = [(gate, self.parameters[gate].get_latest()) for gate in self.gates]
        else:
            vals = [(gate, self.get(gate)) for gate in self.gates]
        return dict(vals)

class QSTLDotModel(Instrument):
    """ 
    Quantum dot model
    """

    def __init__(
            self,
            name: str,
            verbose: int = 0,
            dot_configuration: str = None, # path of dot configuration json file
            virtual_gate: bool = False,
            crosscap_map: Optional[Dict] = None,
            **kwargs
        ):
        """
        Spin dot model

            Args:
                name  name for the instrument
                verbose : verbosity level
                dot_configuration : dot configuration
                example of dot configuration >>
                {
                    "gate_mapping" : {
                        "G0" : "(1, 7 ,3)",      # Gate Name : (chassis, slot, channel)
                        "G1" : "(1, 7, 1)",
                        "P0" : "(1, 7, 2)",
                        "P1" : "(1, 7, 4)"
                    },
                    "dig_mapping" : {
                        "Dig1" : "(1, 18, 1)",   # Digitizer Name : (chassis, slot, channel)
                        "Dig2" : "(1, 18, 2)" ,
                        "Dig3" : "(1, 18, 3)",
                        "Dig4" : "(1, 18, 4)"
                    },
                    "gate_boundary" : {
                        "G0" : "(-2250, 2450)",  # Gate name : (min, max) unit is mV
                        "G1" : "(-2250, 2450)",
                        "P0" : "(-20, 20)"
                    },
                    "rf_mapping" : {
                        "RF1" : "(1, 4, 4)",     # RF name : (chassis, slot, channel)
                        "RF2" : "(1, 4, 3)",
                        "RF3" : "(1, 4, 2)",
                        "RF4" : "(1, 4, 2)"
                    },
                    "meas_setup" : {             # Measurement setup
                        "meas_params" : {
                            "Gm0" : {            # Measurement value. (Not important)
                                "rf" : "RF1",    # RF signal generator.
                                "dig" : "Dig1"   # Digitizer for reflected RF signal measurement
                            },
                            "Gm1" : {
                                "rf" : "RF4",
                                "dig" : "Dig4"
                            }
                        },
                        "frequency" : 200e6,     # RF frequency in MHz
                        "gate_time" : 2e-6,      # DC settling time
                        "acquisition_time" : 2e-6, # RF acquisition time
                        "n_shots" : 1,           # repetition number
                        "sweepparams" : {        # Sweep parameter setup
                            "G0" : {             # Target physicla gate to sweep (X direction)
                                "range" : 200,   # sweep range in mV
                                "resolution" : 100 # number of points to sweep
                            },
                            "G1" : {             # Target physicla gate to sweep (Y direction)
                                "range" : 200,
                                "resolution" : 100
                            }
                        }
                    }
                }

        """

        super().__init__(name, **kwargs)

        if dot_configuration is None:
            raise ValueError("Dot configuration is empty...")
        else:
            with open(dot_configuration, "r", encoding="utf-8") as _config:
                _config_data = json.load(_config)

                self.gate_mapping = _config_data["gate_mapping"]
                for gate_name, dac_addr in self.gate_mapping.items():
                    self.gate_mapping[gate_name] = ast.literal_eval(dac_addr)

                self.dig_mapping = _config_data["dig_mapping"]
                for dig_name, dig_addr in self.dig_mapping.items():
                    self.dig_mapping[dig_name] = ast.literal_eval(dig_addr)
                
                self.rf_mapping = _config_data["rf_mapping"]
                for rf_name, rf_addr in self.rf_mapping.items():
                    self.rf_mapping[rf_name] = ast.literal_eval(rf_addr)

                self.gate_boundary = _config_data["gate_boundary"]
                for gate_name, gate_bound in self.gate_boundary.items():
                    self.gate_boundary[gate_name] = ast.literal_eval(gate_bound)
                
                self.meas_setup = _config_data["meas_setup"]

                if hasattr(_config_data, "crosscap_map"):
                    self.crosscap_map = _config_data["crosscap_map"]
                elif crosscap_map is not None:
                    self.crosscap_map = crosscap_map
                else:
                    # Set virtual gate matrix as identity matrix
                    self.crosscap_map = {}
                    for gate_name, _ in self.gate_mapping.items():
                        for pgate_name, _ in self.gate_mapping.items():
                            if not ("V" + gate_name) in self.crosscap_map:
                                self.crosscap_map["V" + gate_name] = {}
                            if pgate_name == gate_name:
                                self.crosscap_map["V" + gate_name][pgate_name] = 1.0
                            else:
                                self.crosscap_map["V" + gate_name][pgate_name] = 0.0

        # dictionary to hold the data of the model
        self._data: Dict = {}
        self.lock = threading.Lock()
        self.virtual_gate = virtual_gate
        # Physicla gate which contains voltage value of DAC. Setting this value does not 
        # set output voltage of QCS directly.
        self.pgates = QSTLGate(
            name = "pgates",
            gate_mapping = self.gate_mapping,
        )
        # Virtual gate instrument. This contains cross capacitance matrix.
        self.gates = VirtualGates(
            name = "gates",
            gates_instr = self.pgates,
            crosscap_map = self.crosscap_map,
        )
        # Set minimum, maximum value of physical gate.
        self.pgates.set_boundaries(self.gate_boundary)

        station = qcodes.Station(
            self.pgates,
            self.gates,
            self,
            update_snapshot=False
        )
        station.model = self
        station.gate_settle = lambda: 0
        station.jobmanager = None
        station.calib_master = None

        self.station = station
        self._initialized = True

    def _data_get(self, param):
        return self._data.get(param, 0)

    def _data_set(self, param, value):
        self._data[param] = value
        return
