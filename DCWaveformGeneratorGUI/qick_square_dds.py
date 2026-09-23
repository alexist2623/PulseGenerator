"""SquarePulse settings and exact-word tProcessor hardware sweeps."""
from dataclasses import dataclass, asdict
from math import isfinite, ceil
from numbers import Integral
import numpy as np


@dataclass(frozen=True)
class SquarePulseSweep:
    parameter: str
    start: float
    stop: float
    count: int
    gen_ch: int
    segment_name: str = "SquarePulse"
    output_name: str = "square"

    def __post_init__(self):
        if self.parameter not in ("frequency", "amplitude", "phase"):
            raise ValueError("unknown SquarePulse sweep parameter")
        if isinstance(self.count, bool) or not isinstance(self.count, Integral) or self.count < 1:
            raise ValueError("SquarePulse sweep count must be a positive integer")
        if not all(isfinite(v) for v in (self.start, self.stop)):
            raise ValueError("SquarePulse sweep endpoints must be finite")

    @property
    def points(self):
        return tuple(float(x) for x in np.linspace(self.start, self.stop, self.count))

    @property
    def axis_kind(self):
        return "square_" + self.parameter

    @property
    def coordinate_unit(self):
        return {"frequency": "MHz", "amplitude": "mV", "phase": "deg"}[self.parameter]


@dataclass(frozen=True)
class SquarePulseConfig:
    gen_ch: int
    frequency_mhz: float = 0.04
    amplitude_mv: float = 10.0
    phase_deg: float = 0.0
    full_scale_mv: float = 800.0

    def __post_init__(self):
        if isinstance(self.gen_ch, bool) or not isinstance(self.gen_ch, Integral) or self.gen_ch < 0:
            raise ValueError("SquarePulse generator must be a nonnegative integer")
        if not all(isfinite(v) for v in (self.frequency_mhz, self.amplitude_mv, self.phase_deg, self.full_scale_mv)):
            raise ValueError("SquarePulse parameters must be finite")
        if self.full_scale_mv <= 0 or not 0 <= self.amplitude_mv <= self.full_scale_mv:
            raise ValueError("SquarePulse amplitude must be between zero and output full scale")

    def word(self, parameter, value, gencfg):
        # Import only on use: older QICK installations can still open old
        # firmware and run every pre-existing GUI experiment.
        try:
            from qick.square_pulse import frequency_word, phase_word
        except ImportError as exc:
            raise RuntimeError("Update the QSTL_QICK Python library to use SquarePulse firmware") from exc
        if parameter == "frequency":
            return frequency_word(value, float(gencfg["f_dds"]))
        if parameter == "phase":
            return phase_word(value)
        if not isfinite(value) or not 0 <= value <= self.full_scale_mv:
            raise ValueError("SquarePulse amplitude exceeds output full scale")
        return min(32764, int(round(value / self.full_scale_mv * 32768 / 4)) * 4)


@dataclass(frozen=True)
class OutputTriggerConfig:
    enabled: bool = False
    pin: int = 0
    scope: str = "loop"
    edge: str = "start"
    width_us: float = 1.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("output trigger enabled must be boolean")
        if self.scope not in ("loop", "experiment") or self.edge not in ("start", "end", "both"):
            raise ValueError("invalid output trigger boundary")
        if not isfinite(self.width_us) or self.width_us <= 0:
            raise ValueError("output trigger width must be positive and finite")
        if isinstance(self.pin, bool) or not isinstance(self.pin, Integral) or self.pin < 0:
            raise ValueError("output trigger pin must be a nonnegative integer")


def attach_square_settings(sequence, config=None, sweeps=(), trigger=None):
    """Attach independent SquarePulse Cartesian axes to an AWG sequence."""
    sweeps = tuple(sweeps)
    if sweeps and config is None:
        raise ValueError("SquarePulse sweep requires an enabled generator")
    if len({axis.parameter for axis in sweeps}) != len(sweeps):
        raise ValueError("only one sweep per SquarePulse parameter is allowed")
    if any(axis.gen_ch != config.gen_ch for axis in sweeps):
        raise ValueError("SquarePulse sweep generator differs from selected output")
    sequence.sweeps = [axis for axis in sequence.sweeps if not isinstance(axis, SquarePulseSweep)] + list(sweeps)
    sequence._sweep_coordinate_cache = None
    sequence.square_pulse_config = config
    sequence.output_trigger_config = trigger or OutputTriggerConfig()
    return sequence


def decode_square_settings(settings, full_scale_mv=800.0):
    """Decode saved/exported settings independently of Qt and firmware."""
    settings = settings or {}
    if not isinstance(settings, dict):
        raise ValueError("SquarePulse settings must be a mapping")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("SquarePulse enabled must be boolean")
    if not enabled:
        return None, ()
    ch = settings.get("gen_ch", 7)
    parameters = settings.get("parameters", {})
    values = {}
    for name, default in (("frequency", .04), ("amplitude", 10.0), ("phase", 0.0)):
        row = parameters.get(name, {})
        fixed = row.get("value", default)
        values[name] = row.get("start", fixed) if row.get("sweep", False) else fixed
    config = SquarePulseConfig(ch, values["frequency"], values["amplitude"], values["phase"], full_scale_mv)
    axes = []
    for name, value in values.items():
        row = parameters.get(name, {})
        if not isinstance(row.get("sweep", False), bool):
            raise ValueError("SquarePulse sweep enabled must be boolean")
        if row.get("sweep", False):
            axes.append(SquarePulseSweep(name, row.get("start", value), row.get("stop", value),
                                        row.get("count", 11), ch, segment_name=name, output_name=f"square{ch}"))
    return config, tuple(axes)


class SquarePulseProgramMixin:
    """Use the existing DMEM pointer and Cartesian-loop allocator."""
    def _configure_square_pulse(self):
        config = getattr(self.sequence, "square_pulse_config", None)
        self.square_pulse_config = config
        self._square_initial_words = {}
        if config is None:
            if any(isinstance(axis, SquarePulseSweep) for axis in self.sequence.sweep_axes):
                raise ValueError("SquarePulse axes require generator settings")
            return
        if config.gen_ch >= len(self.soccfg["gens"]):
            raise ValueError("SquarePulse generator is absent from this firmware")
        gen = self.soccfg["gens"][config.gen_ch]
        if gen.get("type") != "axis_square_pulse_v1":
            raise ValueError("selected generator is not a SquarePulse IP in this firmware")
        if config.gen_ch in self.awg_channels or any(rf.gen_ch == config.gen_ch for rf in self.rf_pulse_configs):
            raise ValueError("SquarePulse output cannot also be an AWG or RF output")
        self.declare_gen(ch=config.gen_ch, nqz=1)
        defaults = dict(frequency=config.frequency_mhz, amplitude=config.amplitude_mv, phase=config.phase_deg)
        for axis in self.sequence.sweep_axes:
            if isinstance(axis, SquarePulseSweep):
                defaults[axis.parameter] = axis.start
        self._square_initial_words = {
            name: config.word(name, value, gen) for name, value in defaults.items()
        }

    def _square_point_table_models(self, axes):
        tables=[]
        config=self.square_pulse_config
        if config is None:
            return tables
        for index,axis in enumerate(axes):
            if not isinstance(axis,SquarePulseSweep) or axis.count <= 1:
                continue
            register={"frequency":"freq", "phase":"phase", "amplitude":"gain"}[axis.parameter]
            page,reg=self._gen_regmap[(config.gen_ch,register)]
            tables.append(dict(key=("square_point_table",config.gen_ch,axis.parameter),
                register_name="square_"+axis.parameter, gen_ch=config.gen_ch,
                event_indices=(), page=int(page),command_register=int(reg),
                axis_indices=(index,),axis_shape=(axis.count,),
                values=tuple(config.word(axis.parameter,v,self.soccfg["gens"][config.gen_ch]) for v in axis.points)))
        return tables

    def _emit_square_update(self, *, stop=False, reset_phase=False):
        config=self.square_pulse_config
        if config is None:
            return
        words=self._square_initial_words
        if stop:
            # Keep the final point's frequency/phase/amplitude registers intact.
            # Only disable output; the accumulator continues at its last rate.
            page, reg = self._gen_regmap[(config.gen_ch, "control")]
            control = int(self.soccfg["gens"][config.gen_ch].get("tmux_ch", 0)) << 24
            self.safe_regwi(page, reg, control, "mute SquarePulse without changing phase rate")
        else:
            self.set_pulse_registers(ch=config.gen_ch,style="square",freq=words["frequency"],
                phase=words["phase"],gain=words["amplitude"],enable=True,reset_phase=reset_phase)
            for table in self._rf_point_tables:
                if table["key"][0]=="square_point_table":
                    self.memr(table["page"],table["command_register"],table["pointer_register"],"load SquarePulse hardware sweep word")
        self.pulse(ch=config.gen_ch,t=0)
        # Reserve the fixed command pipeline and transport settling before
        # starting AWG/readout commands on the shared tProcessor/TMUX output.
        guard=max(8,ceil(8*self.tproc_mhz/float(self.soccfg["gens"][config.gen_ch]["f_fabric"])))
        self.synci(guard)
        self.reset_timestamps()

    def _square_settings_metadata(self):
        config=self.square_pulse_config
        return {} if config is None else dict(square_pulse={
            **asdict(config),"phase_continuous":True,"command_latency_cycles":4,
            "sweep_execution":"tProcessor Cartesian loops with exact DMEM words"})
