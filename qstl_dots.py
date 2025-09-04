""" Dot array for QSTL

There are virtual instruments for

- DACs: several virtual IVVIs
- A virtual gates object
 """


# %% Load packages

import logging
import threading
import json
import ast
from functools import partial
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import qcodes
from qcodes import Instrument
from qcodes.utils.validators import Numbers

from qtt.instrument_drivers.virtual_gates import VirtualGates

logger = logging.getLogger(__name__)

# %%
class QSTLGate(Instrument):
    def __init__(
            self,
            name,
            gate_mapping,
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
        for gate, bd in boundaries.items():
            self.parameters[gate].vals = Numbers(bd[0], bd[1])
    
    def ask_raw(self, cmd: str) -> str:
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
    """ Simulation model for linear dot array

    The model is intended for testing the code and learning. 
    It does _not_ simulate any meaningful physics.

    """

    def __init__(
            self,
            name: str,
            verbose: int = 0,
            dot_configuration: str = None,
            virtual_gate: bool = False,
            crosscap_map: dict = None,
            **kwargs
        ):
        """
        Spin dot model

            Args:
                name  name for the instrument
                verbose : verbosity level
                dot_configuration : dot configuration
                {
                    "gate_mapping" : {
                        "G0"  : "(1, 4, 1)", # Gate Name : (chassis, slot, channel)
                        "G1"  : "(1, 4, 2)",
                        "SD0" : "(1, 4, 3)"
                    },
                    "dig_mapping" : {
                        "Dig0" : "(1, 13, 1)", # Digitizer Name : (chassis, slot, channel)
                        "Dig1" : "(1, 13, 2)"
                    },
                    "gate_boundary" : {
                        "G0" : "(-300, 450)" # Gate name : (min, max) unit is mV
                    },
                    "rf_mapping" : {
                        "RF0" : "(1,3,1)"
                    },
                    "meas_setup" : {
                        "rf" : "RF0",
                        "dig" : "Dig0",
                        "frequency" : 200e6,
                        "gate_time" : 5e-6,
                        "acquisition_time" : 5e-6,
                        "sweepparams" : {
                            "G0" : {
                                "min" : -10,
                                "max" : 15,
                                "num" : 30
                            },
                            "G1" : {
                                "min" : -20,
                                "max" : 5,
                                "num" : 30
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

        print(self.crosscap_map)
        # dictionary to hold the data of the model
        self._data: Dict = {}
        self.lock = threading.Lock()
        self.virtual_gate = virtual_gate
        self.pgates = QSTLGate(
            name = "pgates",
            gate_mapping = self.gate_mapping,
        )
        self.gates = VirtualGates(
            name = "gates",
            gates_instr = self.pgates,
            crosscap_map = self.crosscap_map,
        )
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

# %%


if __name__ == '__main__' and 1:
    pass
