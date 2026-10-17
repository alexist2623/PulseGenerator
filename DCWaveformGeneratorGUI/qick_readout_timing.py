"""Minimum commands for the continuously running QSTL dynamic readout.

Command length is independent of AVG integration and DDR capture length.
The axis_dyn_readout_v1 output remains active after a oneshot command ends.
"""
from math import ceil

READOUT_COMMAND_CYCLES = 3
LEGACY_COMMAND_MAX_CYCLES = 65535
# The RTL FIFO has seven usable queued entries, plus one active command.
LEGACY_PENDING_COMMANDS = 8


def _continuous_readout(program, ch):
    cfg = program.soccfg["readouts"][ch]
    return cfg.get("ro_type", cfg.get("type")) == "axis_dyn_readout_v1"


def configure_readout(program, *, ch, freq, length, mode="oneshot", phrst=0):
    """Keep other readout IP contracts; minimize all current QSTL paths."""
    if _continuous_readout(program, ch):
        length, mode = READOUT_COMMAND_CYCLES, "oneshot"
    program.set_readout_registers(
        ch=ch, freq=freq, length=length, mode=mode, phrst=phrst,
    )


def initialize_readout(program, ch):
    """Drain old commands once before the measurement loop, entirely in ASM.

The previous GUI may have left a long periodic command or queued commands.
They cannot be interrupted by a new command. This bounded startup preamble
allows them to drain, applies the new three-cycle command, and flushes the
linear-phase FIR history. It does not change per-point pulse timestamps,
acquisition sizes, repetition intervals, or phase-reset policy.
"""
    if not _continuous_readout(program, ch):
        return
    ro = program.soccfg["readouts"][ch]
    tproc_mhz = float(program.tproccfg["f_time"])
    fabric_mhz = float(ro["f_fabric"])
    drain = ceil((LEGACY_PENDING_COMMANDS * LEGACY_COMMAND_MAX_CYCLES
                  + READOUT_COMMAND_CYCLES + 32) * tproc_mhz / fabric_mhz)
    try:
        ddr = program.soccfg["ddr4_buf"]
    except KeyError:
        ddr = {}
    history = ceil(2 * float(ddr.get("fir_group_delay_input_samples", 0))
                   * tproc_mhz / float(ddr.get("fir_input_fs_mhz", fabric_mhz)))
    # Cover the readout's own DDS/decimation pipeline as well as the DDR FIR.
    guard = drain + history + ceil(128 * tproc_mhz / fabric_mhz)
    # readout() emits nonblocking SET: a full FIFO would discard the command.
    # One maximum old period frees a slot even with seven queued entries.
    # This wait is inside the existing eight-period drain budget.
    slot_wait = ceil((LEGACY_COMMAND_MAX_CYCLES + 8) * tproc_mhz / fabric_mhz)
    program.synci(128, "readout initialization lookahead")
    program.waiti(0, slot_wait, "free one legacy readout FIFO slot")
    program.readout(ch=ch, t=slot_wait + 128)
    program.waiti(0, guard, "drain legacy readout commands and FIR history")
    program.synci(guard + 128, "measurement time origin after readout initialization")
    program.readout_initialization = dict(
        ro_ch=int(ch), command_cycles=READOUT_COMMAND_CYCLES, mode="oneshot",
        startup_wait_tproc_cycles=int(guard), startup_wait_us=guard / tproc_mhz,
        legacy_queue_bound=LEGACY_PENDING_COMMANDS, fir_history_tproc_cycles=history,
        fifo_slot_wait_tproc_cycles=slot_wait,
        measurement_timestamps_unchanged=True,
    )
