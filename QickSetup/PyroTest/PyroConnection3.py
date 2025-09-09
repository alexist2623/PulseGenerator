"""Qick Pyro connection test"""
import numpy as np
import matplotlib.pyplot as plt

from qick import *
from qick.averager_program import QickSweep
from qick.pyro import make_proxy

class P1dBMeas_Exp(NDAveragerProgram):
    def initialize(self):
        cfg = self.cfg
        freq_rf     = cfg["freq_rf"]
        # Declare RF generation channel
        self.declare_gen(
            ch      = 0,        # Channel
            nqz     = 2         # Nyquist Zone
        )
        # Declare RF input channel
        self.declare_readout(
            ch      = 0,        # Channel
            length  = cfg["pulse_time"] + 100       # Readout length
        )
        self.r_freq  = self.get_gen_reg(gen_ch = 0, name = "freq")
        f_start     = self.freq2reg(
            f       = 100,      # Frequency
            gen_ch  = 0,        # Generator channel
            ro_ch   = 0         # Readout channel for round up
        )
        f_stop      = self.freq2reg(
            f       = 2000,      # Frequency
            gen_ch  = 0,        # Generator channel
            ro_ch   = 0         # Readout channel for round up
        )
        print(f_stop)
        self.add_sweep(
            QickSweep(
                prog    = self,
                reg     = self.r_freq,
                start   = 100,
                stop    = f_stop,
                expts   = 500,
                label   = "freq_sweep"
            )
        )

        # Convert RF frequency to DAC DDS register value
        freq_dac    = self.freq2reg(
            f       = freq_rf,  # Frequency
            gen_ch  = 0,        # Generator channel
            ro_ch   = 0         # Readout channel for round up
        )
        # Convert RF frequency to ADC DDS register value
        freq_adc    = self.freq2reg_adc(
            f       = freq_rf,  # Frequency
            ro_ch   = 0,        # Readout channel
            gen_ch  = 0         # Generator channel for round up
        )

        # Set DAC DDS
        self.set_pulse_registers(
            ch      = 0,        # Generator channel
            style   = "const",  # Output is gain * DDS output
            freq    = freq_dac, # Generator DDS frequency
            phase   = 0,        # Generator DDS phase
            gain    = 5000,     # Generator amplitude
            length  = cfg["pulse_time"],       # Pulse length
            phrst   = 0         # Generator DDS phase reset
        )
        # Set ADC DDS
        self.set_readout_registers(
            ch      = 0,        # Readout channel
            freq    = freq_adc, # Readout DDS frequency
            length  = cfg["pulse_time"],       # Readout DDS multiplication length
            phrst   = 0         # Readout DDS phase reset
        )
        self.synci(100)
    def body(self):
        self.pulse(
            ch      = 0,        # Generator channel
            t       = 50        # Pulse will be output @ sync_t + 100
        )
        self.readout(
            ch      = 0,        # Readout channel
            t       = 50        # Readout DDS will start multiplication
                                # @ sync_t + 100
        )
        self.trigger(
            adcs    = [0],      # Readout channels
            adc_trig_offset = 150 # Readout will capture the data @ sync_t + 50
        )
        self.sync_all(1000)

if __name__ == "__main__":
    # Qick version : 0.2.357
    (soc, soccfg) = make_proxy("192.168.2.99")

    # Set DAC Channel 0 attenuation 20 dB and 20 dB, and turn on DAC channel
    soc.rfb_set_gen_rf(0,31,31)
    # Set DAC Channel filter as bypass mode
    soc.rfb_set_gen_filter(0,fc = 2.5, ftype = "bypass")

    # Set ADC Channel attenuation 20 dB, and turn on ADC channel
    soc.rfb_set_ro_rf(0,31)
    # Set ADC Channel filter as bypass mode
    soc.rfb_set_ro_filter(0, fc = 2.5, ftype = "bypass")

    cfg = {
        # Experiment Setup
        "reps" : 100,
        "expts" : 1,
        "freq_sweep_num" : 100,
        "duration_sweep_num" : 100,
        # Parameter Setup
        "freq_rf" : 1500,
        "pulse_time" : 4000
    }
    prog = P1dBMeas_Exp(
        soccfg, # Note that it should be QickSocConfig, not the QickSoc
        cfg
    )
    expt_pts, avg_di, avg_dq = prog.acquire(soc, progress=True)
    meas_pwr = np.sqrt(avg_di[0][0] * avg_di[0][0] + avg_dq[0][0] * avg_dq[0][0])
    plt.figure()
    plt.plot(expt_pts[0], meas_pwr)
    plt.tight_layout()
    plt.show()
