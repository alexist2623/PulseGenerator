"""Focused tests for tProcessor loop/add AWG amplitude sweeps.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import numpy as np

# Desktop imports need the PYNQ stubs installed before regular QICK modules.
from qick.sim import QickSim  # noqa: F401
from qick.awg_tuning import TProcV1BehaviorModel
from qick.qick_asm import QickConfig

from qick_fine_tune_sweep import FineTuneSequence


def _mock_soccfg(n_outputs=2):
    gens = []
    for index in range(n_outputs):
        gens.append({
            "type": "axis_awg_tuning_v1",
            "gen_type": "awg_tuning",
            "tproc_ch": 0,
            "tmux_ch": index,
            "f_fabric": 300.0,
            "n_pts": 16,
            "samps_per_clk": 16,
            "frac": 16,
            "cmd_width": 160,
            "step_width": 24,
            "duration_width": 23,
            "fixed_width": 48,
            "dac_invalid_lsb": 2,
            "maxv": 32764,
            "minv": -32768,
            "ramp_startup_latency_cycles": 7,
            "ramp_guard_cycles": 1,
            "has_mixer": False,
            "has_dds": False,
            "b_dds": 32,
            "b_phase": 32,
            "dac": f"{index:02d}",
            "interpolation": 1,
        })
    return QickConfig({
        "sw_version": "0.2.357",
        "tprocs": [{
            "type": "axis_tproc64x32_x8",
            "f_time": 300.0,
            "pmem_size": 65536,
            "dmem_size": 4096,
            "output_pins": [],
        }],
        "gens": gens,
        "readouts": [],
    })


def _command_word(command):
    return sum(
        (int(word) & 0xFFFFFFFF) << (32 * index)
        for index, word in enumerate(command.words)
    )


def _expected_words(program):
    expected = []
    for point in program.compiled_points:
        point_words = [
            _command_word(command)
            for commands in point.segment_commands
            for command in commands
        ]
        for _ in range(program.cfg["reps"]):
            expected.extend(point_words)
    return expected


def _make_cartesian_sequence():
    sequence = FineTuneSequence(("awg_0", "awg_1"))
    sequence.set_cross_capacitance(((1.0, 0.0), (0.25, 1.0)))
    sequence.add_set("start", (-0.25, 0.15), 12)
    sequence.add_ramp("to_gate", 24)
    sequence.add_set("gate", (0.1, -0.2), 18)
    sequence.add_ramp("to_measure", 20)
    sequence.add_set("measure", (0.3, 0.2), 16)
    sequence.add_amplitude_sweep("gate", "awg_0", -0.53, 0.47, 4)
    sequence.add_amplitude_sweep("measure", "awg_1", -0.31, 0.29, 3)
    return sequence


def test_set_and_ramp_words_are_updated_by_nested_loop_and_add():
    sequence = _make_cartesian_sequence()
    program = sequence.make_program(
        _mock_soccfg(),
        awg_channels=(0, 1),
        repetitions_per_sweep=3,
    )
    program.compile()

    tproc = TProcV1BehaviorModel(strict=True)
    tproc.run(program.prog_list, max_steps=1_000_000)

    assert [event.word for event in tproc.output_events] == _expected_words(program)
    assert sum(inst["name"] == "loopnz" for inst in program.prog_list) == 3
    assert any(
        inst["name"] == "mathi" and inst["args"][3] == "+"
        for inst in program.prog_list
    )
    assert not any(inst["name"] in {"memr", "memri"} for inst in program.prog_list)
    assert program.summary()["sweep_execution"] == "tproc_loop_and_add"
    assert program.summary()["sweep_uses_point_table"] is False

    swept_kinds = {
        program.compiled_points[0]
        .segment_commands[field["command_key"][0]][0]
        .kind
        for field in program._sweep_fields
    }
    assert swept_kinds == {"set", "ramp"}


def test_pmem_size_does_not_scale_with_sweep_point_count():
    instruction_counts = []
    for count in (3, 101):
        sequence = FineTuneSequence(("awg_0",))
        sequence.add_set("start", (-0.2,), 10)
        sequence.add_ramp("to_gate", 24)
        sequence.add_set("gate", (0.4,), 12)
        sequence.set_amplitude_sweep("gate", "awg_0", -0.73, 0.61, count)
        program = sequence.make_program(_mock_soccfg(1), awg_channels=(0,))
        program.compile()
        instruction_counts.append(len(program.prog_list))

    assert instruction_counts[0] == instruction_counts[1]


def test_direct_register_sweep_scales_to_eight_outputs():
    names = tuple(f"awg_{index}" for index in range(8))
    coupling = np.eye(8)
    coupling[:, 0] = np.linspace(1.0, 0.125, 8)
    sequence = FineTuneSequence(names)
    sequence.set_cross_capacitance(coupling)
    sequence.add_set("start", (0.0,) * 8, 30)
    sequence.add_ramp("to_gate", 40)
    sequence.add_set("gate", (0.1,) * 8, 30)
    sequence.set_amplitude_sweep("gate", "awg_0", -0.4, 0.4, 5)
    program = sequence.make_program(
        _mock_soccfg(8),
        awg_channels=tuple(range(8)),
    )
    program.compile()

    tproc = TProcV1BehaviorModel(strict=True)
    tproc.run(program.prog_list, max_steps=1_000_000)
    assert [event.word for event in tproc.output_events] == _expected_words(program)
    assert len(program.awg_channels) == 8
    assert not any(inst["name"] in {"memr", "memri"} for inst in program.prog_list)


def test_set_duration_excludes_startup_lead_and_hides_ramp_pipeline():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("set_0", (0.0,), 300)
    sequence.add_ramp("ramp_0", 300)
    sequence.add_set("set_1", (0.5,), 300)
    sequence.add_ramp("ramp_1", 300)
    sequence.add_set("set_2", (0.0,), 300)
    program = sequence.make_program(
        _mock_soccfg(1),
        awg_channels=(0,),
        repetitions_per_sweep=2,
        command_lead_tproc_cycles=128,
    )
    program.compile()

    for segment_index in (0, 2, 4):
        actual = (
            program.timing["segment_ends"][segment_index]
            - program.timing["segment_starts"][segment_index]
        )
        assert actual == 300

    startup = 7
    first_set_command = program.timing["command_times"][(0, 0)]
    first_ramp_command = program.timing["command_times"][(1, 0)]
    assert first_set_command == 0
    assert first_ramp_command + startup - first_set_command == 300
    assert program.summary()["startup_lead_tproc_cycles_once"] == 128

    tproc = TProcV1BehaviorModel(strict=True)
    tproc.run(program.prog_list, max_steps=100_000)
    command_cycles = [event.cycle for event in tproc.output_events]
    commands_per_point = program.summary()["commands_per_point"]
    assert command_cycles[0] == 128
    assert command_cycles[commands_per_point] - command_cycles[0] == (
        program.timing["point_end"] + program.recovery_tproc_cycles
    )
