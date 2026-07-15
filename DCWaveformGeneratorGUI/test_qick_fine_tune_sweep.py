"""Focused tests for tProcessor loop/add AWG amplitude sweeps.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import numpy as np

# Desktop imports need the PYNQ stubs installed before regular QICK modules.
from qick.sim import QickSim  # noqa: F401
from qick.awg_tuning import TProcV1BehaviorModel
from qick.qick_asm import QickConfig

from qick_fine_tune_sweep import FineTuneSequence, RfPulseConfig


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


def _shared_tmux_soccfg():
    """Normal RF gen and AWG tuning share tProc port 0 through a TMUX."""
    cfg = _mock_soccfg(1)._cfg
    awg = dict(cfg["gens"][0])
    awg.update({"tmux_ch": 1, "dac": "10"})
    signal_gen = {
        "type": "axis_signal_gen_v6",
        "tproc_ch": 0,
        "tmux_ch": 0,
        "f_fabric": 300.0,
        "f_dds": 300.0,
        "fs_mult": 1,
        "fs_div": 1,
        "fdds_div": 1,
        "samps_per_clk": 16,
        "maxlen": 16384,
        "maxv": 32766,
        "maxv_scale": 1.0,
        "complex_env": True,
        "has_mixer": False,
        "has_dds": True,
        "b_dds": 32,
        "b_phase": 32,
        "dac": "00",
        "interpolation": 1,
    }
    cfg["gens"] = [signal_gen, awg]
    cfg["refclk_freq"] = 300.0
    return QickConfig(cfg)


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


def test_manual_tproc_clock_overrides_stale_hwh_for_all_program_timing():
    soccfg = _mock_soccfg(1)
    soccfg["tprocs"][0]["f_time"] = 400.0
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("start", (0.0,), 300)
    sequence.add_ramp("to_gate", 300)
    sequence.add_set("gate", (0.5,), 300)

    program = sequence.make_program(
        soccfg,
        awg_channels=(0,),
        tproc_mhz=300.0,
    )
    program.compile()

    assert soccfg["tprocs"][0]["f_time"] == 400.0
    assert program.tproccfg["f_time"] == 300.0
    assert program.timing["segment_ends"][0] == 300
    assert program.summary()["tproc_mhz"] == 300.0
    assert program.summary()["hwh_tproc_mhz"] == 400.0
    assert program.summary()["tproc_clock_is_manual"] is True

    fallback = sequence.make_program(soccfg, awg_channels=(0,))
    fallback.compile()
    assert fallback.timing["segment_ends"][0] == 400
    assert fallback.summary()["tproc_clock_is_manual"] is False


def test_shared_tmux_commands_are_enqueued_in_timestamp_order():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("start", (0.0,), 100)
    sequence.add_ramp("to_gate", 100)
    sequence.add_set("gate", (0.4,), 400)
    sequence.add_ramp("to_end", 100)
    sequence.add_set("end", (0.0,), 100)
    rf = RfPulseConfig(
        gen_ch=0,
        at_segment="gate",
        length_cycles=30,
        gain=10000,
        freq_mhz=0.0,
        delay_tproc_cycles=20,
    )
    program = sequence.make_program(
        _shared_tmux_soccfg(),
        awg_channels=(1,),
        rf_pulse=rf,
    )
    program.compile()

    tproc = TProcV1BehaviorModel(strict=True)
    tproc.run(program.prog_list, max_steps=100_000)
    port_zero_events = [
        event for event in tproc.output_events if event.tproc_ch == 0
    ]
    event_cycles = [event.cycle for event in port_zero_events]
    assert event_cycles == sorted(event_cycles)

    tmux_channels = [(event.word >> 152) & 0xFF for event in port_zero_events]
    rf_index = tmux_channels.index(0)
    assert port_zero_events[rf_index].cycle == (
        program.command_lead_tproc_cycles + program.aux_timing["rf_start"]
    )
    assert any(
        event.cycle > port_zero_events[rf_index].cycle
        and tmux_channel == 1
        for event, tmux_channel in zip(port_zero_events, tmux_channels)
    )


def test_waveform_vertices_reconstruct_swept_virtual_and_physical_pulses():
    sequence = FineTuneSequence(("awg_0", "awg_1"))
    sequence.set_cross_capacitance(((1.0, 0.0), (0.5, 1.0)))
    sequence.add_set("start", (0.0, 0.0), 10)
    sequence.add_ramp("to_gate", 5)
    sequence.add_set("gate", (0.4, -0.2), 8)
    sequence.set_amplitude_sweep("gate", "awg_0", -0.4, 0.4, 3)

    virtual_t, virtual, _ = sequence.waveform_vertices(0, space="virtual")
    physical_t, physical, _ = sequence.waveform_vertices(0, space="physical")
    assert virtual_t.tolist() == [0.0, 10.0, 15.0, 23.0]
    assert physical_t.tolist() == virtual_t.tolist()
    assert virtual["awg_0"].tolist() == [0.0, 0.0, -0.4, -0.4]
    assert virtual["awg_1"].tolist() == [0.0, 0.0, -0.2, -0.2]
    assert physical["awg_0"].tolist() == [0.0, 0.0, -0.4, -0.4]
    assert physical["awg_1"].tolist() == [0.0, 0.0, -0.4, -0.4]

    jump = FineTuneSequence(("awg_0",))
    jump.add_set("low", (0.0,), 10)
    jump.add_set("high", (1.0,), 5)
    jump_t, jump_values, _ = jump.waveform_vertices(space="virtual")
    assert jump_t.tolist() == [0.0, 10.0, 10.0, 15.0]
    assert jump_values["awg_0"].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_counter_progress_tracks_hardware_sweep_and_repetition_count():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("start", (0.0,), 10)
    sequence.set_amplitude_sweep("start", "awg_0", -0.2, 0.2, 3)
    program = sequence.make_program(
        _mock_soccfg(1),
        awg_channels=(0,),
        repetitions_per_sweep=2,
    )
    calls = []

    class FakeSoc:
        counts = iter((0, 2, 6))

        def reload_mem(self):
            calls.append("reload")

        def clear_tproc_counter(self, addr):
            calls.append(("clear", addr))

        def start_src(self, source):
            calls.append(("source", source))

        def start_tproc(self):
            calls.append("start")

        def get_tproc_counter(self, addr):
            calls.append(("counter", addr))
            return next(self.counts)

    program.config_all = lambda *args, **kwargs: calls.append("config")
    progress = []
    program._run_rounds_with_counter_progress(
        FakeSoc(),
        lambda completed, total: progress.append((completed, total)),
        poll_interval_seconds=0.0,
    )

    assert progress == [(0, 6), (0, 6), (2, 6), (6, 6)]
    assert program.summary()["repetitions_per_sweep_point"] == 2
    assert program.summary()["total_acquisitions"] == 6
    assert calls.count("start") == 1
    assert calls[-1] == ("source", "internal")


def test_bias_t_preview_cancels_positive_and_negative_physical_area():
    sequence = FineTuneSequence(("awg_0", "awg_1"))
    sequence.add_set("hold", (0.2, -0.1), 10)
    sequence.set_bias_t_compensation(0.1)

    previews = sequence.bias_t_compensation_preview()
    assert previews[0].pulse_area == 2.0
    assert previews[0].target_amplitude == -0.1
    assert previews[0].duration_cycles == 20
    assert previews[0].residual_area == 0.0
    assert previews[1].pulse_area == -1.0
    assert previews[1].target_amplitude == 0.1
    assert previews[1].duration_cycles == 10
    assert previews[1].residual_area == 0.0

    times, values, boundaries = sequence.compensated_waveform_vertices()
    assert times[-1] == 42.0
    assert values["awg_0"][-1] == 0.0
    assert values["awg_1"][-1] == 0.0
    assert [name for name, _start, _end in boundaries[-2:]] == [
        "bias_t_comp_awg_0",
        "bias_t_comp_awg_1",
    ]


def test_bias_t_duration_is_swept_and_applied_by_tprocessor_sync():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("start", (0.1,), 20)
    sequence.add_ramp("to_gate", 100)
    sequence.add_set("gate", (0.2,), 20)
    sequence.set_amplitude_sweep("gate", "awg_0", 0.2, 0.4, 3)
    sequence.set_bias_t_compensation(0.1)
    program = sequence.make_program(
        _mock_soccfg(1),
        awg_channels=(0,),
        repetitions_per_sweep=2,
    )
    program.compile()

    assert any(inst["name"] == "sync" for inst in program.prog_list)
    assert any(
        inst["name"] == "bitwi" and inst["args"][3:] == (">>", 8)
        for inst in program.prog_list
    )
    assert program.summary()["bias_t_duration_execution"] == (
        "tproc_signed_fixed_point_sync"
    )
    assert program.summary()["bias_t_dynamic_register_fields"] == 1
    assert program._bias_t_duration_q_actual[:, 0].tolist() == sorted(
        program._bias_t_duration_q_actual[:, 0].tolist()
    )

    tproc = TProcV1BehaviorModel(strict=True).run(
        program.prog_list,
        max_steps=1_000_000,
    )
    compensation_code = program._bias_t_fields[0]["negative_code"] & 0xFFFFFFFF
    expected_durations = [
        (abs(int(value)) + (1 << 7)) >> 8
        for value in program._bias_t_duration_q_actual[:, 0]
        for _rep in range(program.cfg["reps"])
    ]
    compensation_events = [
        event
        for event in tproc.output_events
        if (event.word & 0xFFFFFFFF) == compensation_code
    ]
    assert len(compensation_events) == len(expected_durations)
    for event, duration in zip(compensation_events, expected_durations):
        following_zero = next(
            candidate
            for candidate in tproc.output_events
            if candidate.cycle > event.cycle
            and (candidate.word & 0xFFFFFFFF) == 0
        )
        assert following_zero.cycle - event.cycle == duration


def test_bias_t_compensation_registers_fit_eight_outputs():
    names = tuple(f"awg_{index}" for index in range(8))
    sequence = FineTuneSequence(names)
    sequence.add_set("hold", tuple(0.05 * (index + 1) for index in range(8)), 20)
    sequence.set_bias_t_compensation(0.1)
    program = sequence.make_program(
        _mock_soccfg(8),
        awg_channels=tuple(range(8)),
    )
    program.compile()
    assert len(program._bias_t_fields) == 8
    assert len({
        (field["page"], field["state_register"])
        for field in program._bias_t_fields
    }) == 8
    tproc = TProcV1BehaviorModel(strict=True).run(
        program.prog_list,
        max_steps=1_000_000,
    )
    assert tproc.timing_conflicts == []


def test_bias_t_tprocessor_selects_opposite_polarity_across_zero_area():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("hold", (0.0,), 20)
    sequence.set_amplitude_sweep("hold", "awg_0", -0.2, 0.2, 3)
    sequence.set_bias_t_compensation(0.1)
    program = sequence.make_program(_mock_soccfg(1), awg_channels=(0,))
    tproc = TProcV1BehaviorModel(strict=True).run(
        program.prog_list,
        max_steps=1_000_000,
    )

    positive = int(program._bias_t_fields[0]["positive_code"])
    negative = int(program._bias_t_fields[0]["negative_code"])
    compensation_targets = []
    for event in tproc.output_events:
        target = event.word & 0xFFFFFFFF
        target = target - (1 << 32) if target & (1 << 31) else target
        if target in {positive, negative}:
            compensation_targets.append(target)
    assert compensation_targets == [positive, negative]


def test_bias_t_fixed_point_range_supports_100us_full_scale_pulse():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("hold", (1.0,), 30_000)
    sequence.set_bias_t_compensation(0.1)
    program = sequence.make_program(_mock_soccfg(1), awg_channels=(0,))
    duration_cycles = (
        abs(int(program._bias_t_duration_q_actual[0, 0])) + (1 << 7)
    ) >> 8
    assert 299_900 <= duration_cycles <= 300_100
