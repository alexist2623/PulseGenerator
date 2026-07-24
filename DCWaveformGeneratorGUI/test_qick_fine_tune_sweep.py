"""Focused tests for tProcessor loop/add AWG amplitude sweeps.

Authors: Jeonghyun Park (jeonghyun.park@ubc.ca or alexist@snu.ac.kr), Farbod
"""

from __future__ import annotations

import numpy as np
import pytest

# Desktop imports need the PYNQ stubs installed before regular QICK modules.
from qick.sim import QickSim  # noqa: F401
from qick.awg_tuning import TProcV1BehaviorModel
from qick.qick_asm import QickConfig

from qick_fine_tune_sweep import (
    DdrFirReadoutConfig,
    FineTuneSequence,
    ReadoutConfig,
    RfPulseConfig,
)


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


def _independent_awg_soccfg(n_outputs=2):
    cfg = _mock_soccfg(n_outputs)
    for index, gen in enumerate(cfg["gens"]):
        gen["tproc_ch"] = index
    return cfg


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


def _fir_soccfg(*, fir_rate_profile="1_msps"):
    cfg = _mock_soccfg(1)._cfg
    is_50_ksps = fir_rate_profile == "50_ksps"
    cfg["refclk_freq"] = 300.0
    cfg["readouts"] = [{
        "type": "axis_dyn_readout_v1",
        "ro_type": "axis_dyn_readout_v1",
        "tproc_ctrl": 1,
        "tmux_ch": 0,
        "f_fabric": 300.0,
        "f_output": 300.0,
        "f_dds": 300.0,
        "fs_mult": 1,
        "fs_div": 1,
        "fdds_div": 1,
        "b_dds": 32,
        "b_phase": 32,
        "adc": "00",
        "buf_maxlen": 16384,
        "has_weights": False,
        "has_edge_counter": False,
        "trigger_type": "dport",
        "trigger_port": 0,
        "trigger_bit": 0,
    }]
    cfg["ddr4_buf"] = {
        "sample_capture": True,
        "fir_enabled": True,
        "fir_rate_profile": fir_rate_profile,
        "stored_sample_rate_hz": (
            50_000.0 if is_50_ksps else 1_000_000.0
        ),
        "fir_output_fs_mhz": 0.05 if is_50_ksps else 1.0,
        "fir_decimation": 6000 if is_50_ksps else 300,
        "fir_group_delay_input_samples": (
            296_677.0 if is_50_ksps else 8677.0
        ),
        "fir_input_fs_mhz": 300.0,
        "supports_trigger_delay": is_50_ksps,
        "trigger_delay_units": "valid_input_samples",
        "trigger_delay_default_samples": 50 if is_50_ksps else 0,
        "trigger_type": "dport",
        "trigger_port": 0,
        "trigger_bit": 1,
    }
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


def test_50_ksps_ddr_delay_stays_in_fpga_without_tproc_timing_shift(monkeypatch):
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("capture", (0.25,), 300)
    ddr = DdrFirReadoutConfig(
        ro_ch=0,
        samples_per_trigger=8,
        at_segment="capture",
        trigger_delay_tproc_cycles=17,
        margin_input_samples=0,
    )
    baseline = sequence.make_program(
        _fir_soccfg(fir_rate_profile="50_ksps"),
        awg_channels=(0,),
        repetitions_per_sweep=1,
    )
    program = sequence.make_program(
        _fir_soccfg(fir_rate_profile="50_ksps"),
        awg_channels=(0,),
        repetitions_per_sweep=1,
        ddr_readout=ddr,
    )

    assert program.timing["segment_starts"] == baseline.timing["segment_starts"]
    assert program.aux_timing["fir_warmup_tproc_cycles"] == 0
    assert program.aux_timing["ddr_readout_start"] == 0
    assert program.aux_timing["ddr_trigger_time"] == (
        baseline.timing["segment_starts"][0] + 17
    )
    assert program.summary()["fir_rate_profile"] == "50_ksps"
    assert program.summary()["fir_software_warmup_compensation"] is False

    raw = np.arange(8 * 2, dtype=np.int32).reshape(8, 2)

    class FakeSoc:
        def arm_ddr4_fir_samples(self, **kwargs):
            self.arm_kwargs = kwargs
            return 16

        def get_ddr4_fir_samples(self, **_kwargs):
            return raw

    soc = FakeSoc()
    monkeypatch.setattr(program, "run_rounds", lambda *_args, **_kwargs: None)
    result = program.acquire_fir_ddr(soc, progress=False)

    assert soc.arm_kwargs["trigger_delay_samples"] == 50
    assert result.iq.shape == (1, 1, 8, 2)
    assert result.sample_rate_hz == 50_000.0
    assert result.fir_rate_profile == "50_ksps"


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
    assert ((port_zero_events[rf_index].word >> 148) & 1) == 0
    assert port_zero_events[rf_index].cycle == (
        program.command_lead_tproc_cycles + program.aux_timing["rf_start"]
    )
    assert any(
        event.cycle > port_zero_events[rf_index].cycle
        and tmux_channel == 1
        for event, tmux_channel in zip(port_zero_events, tmux_channels)
    )


def test_long_rf_pulse_uses_periodic_start_and_timed_zero_stop():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("measure", (0.25,), 400_000)
    rf = RfPulseConfig(
        gen_ch=0,
        at_segment="measure",
        length_cycles=300_000,
        gain=12_000,
        freq_mhz=10.0,
        require_within_segment=True,
    )
    program = sequence.make_program(
        _shared_tmux_soccfg(),
        awg_channels=(1,),
        rf_pulse=rf,
        repetitions_per_sweep=2,
    )
    program.compile()

    assert program.aux_timing["rf_mode"] == "periodic_timed_stop"
    assert program.aux_timing["rf_end"] - program.aux_timing["rf_start"] == 300_000

    tproc = TProcV1BehaviorModel(strict=True)
    tproc.run(program.prog_list, max_steps=100_000)
    rf_events = [
        event
        for event in tproc.output_events
        if event.tproc_ch == 0 and ((event.word >> 152) & 0xFF) == 0
    ]
    assert len(rf_events) == 4
    assert rf_events[1].cycle - rf_events[0].cycle == 300_000
    assert rf_events[3].cycle - rf_events[2].cycle == 300_000


@pytest.mark.parametrize(
    ("segment_length_mode", "expected_ramp_offsets"),
    [
        ("fixed", [92, 92, 92]),
        ("extend_by_rf_duration", [102, 112, 122]),
    ],
)
def test_rf_duration_sweep_updates_stop_and_selected_segment_timing(
    segment_length_mode,
    expected_ramp_offsets,
):
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("gate", (0.0,), 100)
    sequence.add_ramp("to_end", 20)
    sequence.add_set("end", (0.25,), 100)
    sequence.add_rf_duration_sweep(
        "gate",
        0,
        10 / 300,
        30 / 300,
        3,
        segment_length_mode=segment_length_mode,
        sequence_fabric_mhz=300.0,
    )
    rf = RfPulseConfig(
        gen_ch=0,
        at_segment="gate",
        length_cycles=10,
        gain=12_000,
        require_within_segment=True,
    )
    program = sequence.make_program(
        _shared_tmux_soccfg(),
        awg_channels=(1,),
        rf_pulse=rf,
        command_lead_tproc_cycles=0,
        recovery_tproc_cycles=0,
    )
    program.compile()

    tproc = TProcV1BehaviorModel(strict=True)
    tproc.run(program.prog_list, max_steps=100_000)
    rf_events = [
        event
        for event in tproc.output_events
        if event.tproc_ch == 0 and ((event.word >> 152) & 0xFF) == 0
    ]
    assert len(rf_events) == 6
    assert [
        rf_events[index + 1].cycle - rf_events[index].cycle
        for index in range(0, len(rf_events), 2)
    ] == [10, 20, 30]

    awg_ramps = [
        event
        for event in tproc.output_events
        if (
            event.tproc_ch == 0
            and ((event.word >> 152) & 0xFF) == 1
            and ((event.word >> 144) & 0x3) == 0b10
        )
    ]
    assert len(awg_ramps) == 3
    assert [
        ramp.cycle - rf_events[2 * index].cycle
        for index, ramp in enumerate(awg_ramps)
    ] == expected_ramp_offsets
    expected_segment_lengths = (
        [100, 100, 100]
        if segment_length_mode == "fixed"
        else [110, 120, 130]
    )
    assert [
        sequence.segment_duration_cycles_at(point_index, 0)
        for point_index in range(3)
    ] == expected_segment_lengths


def test_rf_duration_is_a_cartesian_axis_with_amplitude_sweep():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("gate", (0.0,), 100)
    sequence.add_amplitude_sweep("gate", "awg_0", -0.25, 0.25, 2)
    sequence.add_rf_duration_sweep(
        "gate",
        0,
        10 / 300,
        30 / 300,
        3,
        sequence_fabric_mhz=300.0,
    )
    assert sequence.sweep_shape == (2, 3)
    np.testing.assert_allclose(
        sequence.sweep_coordinates,
        [
            [-0.25, 10 / 300],
            [-0.25, 20 / 300],
            [-0.25, 30 / 300],
            [0.25, 10 / 300],
            [0.25, 20 / 300],
            [0.25, 30 / 300],
        ],
    )


def test_rf_and_readout_phase_reset_are_fixed_off():
    rf = RfPulseConfig(
        gen_ch=0,
        at_segment="gate",
        length_cycles=30,
        gain=10000,
    )
    readout = ReadoutConfig(ro_ch=0, length=16)
    assert rf.phrst == 0
    assert readout.phrst == 0

    with pytest.raises(ValueError, match="RF output phrst is fixed to 0"):
        RfPulseConfig(
            gen_ch=0,
            at_segment="gate",
            length_cycles=30,
            gain=10000,
            phrst=1,
        )
    with pytest.raises(ValueError, match="readout phrst is fixed to 0"):
        ReadoutConfig(ro_ch=0, length=16, phrst=1)


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
    assert times[-1] == 95.0
    assert values["awg_0"][-1] == 0.0
    assert values["awg_1"][-1] == 0.0
    starts = {name: start for name, start, _end in boundaries[-2:]}
    ends = {name: end for name, _start, end in boundaries[-2:]}
    assert starts == {
        "bias_t_comp_awg_0": 75.0,
        "bias_t_comp_awg_1": 75.0,
    }
    assert ends == {
        "bias_t_comp_awg_0": 95.0,
        "bias_t_comp_awg_1": 85.0,
    }
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
        "tproc_simultaneous_set_and_max_sync"
    )
    assert program.summary()["bias_t_dynamic_register_fields"] == 0
    assert program.summary()["bias_t_dynamic_dmem_fields"] == 1
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


def test_bias_t_compensation_dmem_state_fits_eight_outputs():
    names = tuple(f"awg_{index}" for index in range(8))
    sequence = FineTuneSequence(names)
    sequence.add_set("hold", tuple(0.05 * (index + 1) for index in range(8)), 20)
    sequence.set_bias_t_compensation(0.1)
    program = sequence.make_program(
        _independent_awg_soccfg(8),
        awg_channels=tuple(range(8)),
    )
    program.compile()
    assert len(program._bias_t_fields) == 8
    assert len({
        field["dmem_addr"]
        for field in program._bias_t_fields
    }) == 8
    assert all("state_register" not in field for field in program._bias_t_fields)
    assert all(
        field["work_register"]
        == program._gen_regmap[(field["gen_ch"], "duration")][1]
        for field in program._bias_t_fields
    )
    tproc = TProcV1BehaviorModel(strict=True).run(
        program.prog_list,
        max_steps=1_000_000,
    )
    assert tproc.timing_conflicts == []


def test_bias_t_independent_outputs_start_together_and_stop_independently():
    sequence = FineTuneSequence(("awg_0", "awg_1"))
    sequence.add_set("hold", (0.2, -0.1), 10)
    sequence.set_bias_t_compensation(0.1)
    program = sequence.make_program(
        _independent_awg_soccfg(2),
        awg_channels=(0, 1),
        recovery_tproc_cycles=0,
    )
    tproc = TProcV1BehaviorModel(strict=True).run(
        program.prog_list,
        max_steps=1_000_000,
    )

    compensation_codes = {
        int(field["gen_ch"]): (
            int(field["negative_code"])
            if int(field["base"]) > 0
            else int(field["positive_code"])
        ) & 0xFFFFFFFF
        for field in program._bias_t_fields
    }
    starts = {}
    stops = {}
    for event in tproc.output_events:
        target = event.word & 0xFFFFFFFF
        if target == compensation_codes.get(event.tproc_ch):
            starts[event.tproc_ch] = event.cycle
        elif (
            event.tproc_ch in starts
            and target == 0
            and event.cycle > starts[event.tproc_ch]
        ):
            stops.setdefault(event.tproc_ch, event.cycle)

    assert starts[0] == starts[1]
    assert stops[0] - starts[0] == 20
    assert stops[1] - starts[1] == 10
    assert tproc.timing_conflicts == []
    assert program.summary()["bias_t_simultaneous_start"] is True
    assert program.summary()["bias_t_simultaneous_start_lead_cycles"] == 64


def test_bias_t_rejects_awgs_sharing_one_tprocessor_output():
    soccfg = _mock_soccfg(2)
    sequence = FineTuneSequence(("awg_0", "awg_1"))
    sequence.add_set("hold", (0.2, -0.1), 10)
    sequence.set_bias_t_compensation(0.1)

    with pytest.raises(ValueError, match="independent tProcessor output"):
        sequence.make_program(soccfg, awg_channels=(0, 1))


def test_bias_t_dmem_state_avoids_full_register_page_failure():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("initial", (0.0,), 16)
    for index in range(8):
        sequence.add_ramp(f"ramp_{index}", 16)
        segment_name = f"set_{index}"
        sequence.add_set(segment_name, (0.01 * (index + 1),), 16)
        sequence.add_amplitude_sweep(
            segment_name,
            "awg_0",
            0.01 * (index + 1),
            0.01 * (index + 2),
            2,
        )
    sequence.set_bias_t_compensation(0.1)

    # Generator 1 uses register page 1. Its eight swept SETs and dependent
    # RAMPs consume 24 direct target/step state registers, reproducing the
    # page-pressure case which previously left no Bias-T work register.
    program = sequence.make_program(_mock_soccfg(2), awg_channels=(1,))
    program.compile()

    direct_fields = [
        field
        for field in program._sweep_fields
        if field.get("storage") != "dmem" and field["page"] == 1
    ]
    assert len(direct_fields) == 24
    assert program._bias_t_fields[0]["page"] == 1
    assert program._bias_t_fields[0]["dmem_addr"] == 4095
    assert program._bias_t_max_duration_dmem_addr == 4094
    assert program.summary()["bias_t_dynamic_dmem_fields"] == 1
    assert any(inst["name"] == "memri" for inst in program.prog_list)
    assert any(inst["name"] == "memwi" for inst in program.prog_list)


def test_sweep_state_spills_to_dmem_after_register_page_is_full():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("initial", (0.0,), 16)
    for index in range(9):
        sequence.add_ramp(f"ramp_{index}", 16)
        segment_name = f"set_{index}"
        sequence.add_set(segment_name, (0.01 * (index + 1),), 16)
        sequence.add_amplitude_sweep(
            segment_name,
            "awg_0",
            0.01 * (index + 1),
            0.01 * (index + 2),
            2,
        )
    sequence.set_bias_t_compensation(0.1)

    program = sequence.make_program(_mock_soccfg(2), awg_channels=(1,))
    program.compile()

    register_fields = [
        field for field in program._sweep_fields
        if field.get("storage") == "register"
    ]
    dmem_fields = [
        field for field in program._sweep_fields
        if field.get("storage") == "dmem"
    ]
    assert len(register_fields) == 25
    assert len(dmem_fields) == 3
    assert len({field["dmem_addr"] for field in dmem_fields}) == 3
    assert all(field["dmem_addr"] > program.COUNTER_ADDR for field in dmem_fields)


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


def test_bias_t_fixed_time_preview_adjusts_voltage_and_keeps_duration():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("hold", (0.2,), 10)
    sequence.set_bias_t_compensation(
        0.1,
        mode="fixed_time",
        fixed_duration_cycles=20,
    )

    (preview,) = sequence.bias_t_compensation_preview()
    assert preview.pulse_area == 2.0
    assert preview.duration_cycles == 20
    assert preview.target_amplitude == pytest.approx(-0.1, abs=2 / 32768)
    assert abs(preview.residual_area) <= 40 / 32768


def test_bias_t_fixed_time_sweeps_dmem_target_and_emits_constant_duration():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("hold", (0.1,), 20)
    sequence.set_amplitude_sweep("hold", "awg_0", 0.1, 0.3, 3)
    sequence.set_bias_t_compensation(
        0.1,
        mode="fixed_time",
        fixed_duration_cycles=40,
    )
    program = sequence.make_program(
        _mock_soccfg(1),
        awg_channels=(0,),
    )
    program.compile()

    expected_targets = program._bias_t_target_code_actual[:, 0].tolist()
    assert len(set(expected_targets)) == 3
    assert all(target < 0 and target & 0b11 == 0 for target in expected_targets)
    assert program.summary()["bias_t_compensation_mode"] == "fixed_time"
    assert program.summary()["bias_t_duration_execution"] == (
        "tproc_fixed_time_dynamic_voltage"
    )
    assert program.summary()["bias_t_dynamic_dmem_fields"] == 1

    tproc = TProcV1BehaviorModel(strict=True).run(
        program.prog_list,
        max_steps=1_000_000,
    )

    def signed_target(event):
        target = event.word & 0xFFFFFFFF
        return target - (1 << 32) if target & (1 << 31) else target

    starts = [
        event
        for event in tproc.output_events
        if signed_target(event) in set(expected_targets)
    ]
    assert [signed_target(event) for event in starts] == expected_targets
    for event in starts:
        following_zero = next(
            candidate
            for candidate in tproc.output_events
            if candidate.tproc_ch == event.tproc_ch
            and candidate.cycle > event.cycle
            and (candidate.word & 0xFFFFFFFF) == 0
        )
        assert following_zero.cycle - event.cycle == 40
    assert tproc.timing_conflicts == []


def test_bias_t_fixed_time_rejects_duration_that_requires_overrange_voltage():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("hold", (1.0,), 100)
    sequence.set_bias_t_compensation(
        0.1,
        mode="fixed_time",
        fixed_duration_cycles=1,
    )

    with pytest.raises(ValueError, match="too short"):
        sequence.make_program(_mock_soccfg(1), awg_channels=(0,))


def test_bias_t_filter_flat_slew_is_target_over_tau_and_cancels_droop():
    target = 0.2
    tau_cycles = 3_000.0
    flat_cycles = 300
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("hold", (target,), flat_cycles)
    sequence.set_bias_t_filter_compensation(tau_cycles)

    (((start,), (end,)),) = sequence.filter_compensated_segment_levels()
    assert start == pytest.approx(target)
    assert (end - start) / flat_cycles == pytest.approx(target / tau_cycles)

    # First-order Bias-T model: tau*dz/dt + z = x and y = x-z. For a
    # piecewise-linear input x=a+b*t, this is the exact analytic response.
    times = np.linspace(0.0, float(flat_cycles), 101)

    def highpass_linear(a, slope):
        x = a + slope * times
        z0 = 0.0
        z = (
            a
            + slope * (times - tau_cycles)
            + (z0 - a + slope * tau_cycles) * np.exp(-times / tau_cycles)
        )
        return x - z

    compensated = highpass_linear(target, target / tau_cycles)
    uncompensated = highpass_linear(target, 0.0)
    assert np.max(np.abs(compensated - target)) < 1.0e-12
    assert uncompensated[-1] == pytest.approx(
        target * np.exp(-flat_cycles / tau_cycles)
    )
    assert uncompensated[-1] < target


def test_bias_t_filter_compiles_each_flat_as_set_then_ramp():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("hold", (0.2,), 300)
    sequence.set_bias_t_filter_compensation(3_000)
    program = sequence.make_program(_mock_soccfg(1), awg_channels=(0,))
    program.compile()

    commands = program.compiled_points[0].segment_commands[0]
    assert [(command.kind, command.command_slot) for command in commands] == [
        ("set", 0),
        ("ramp", 1),
    ]
    assert commands[1].duration_samples == 300 * 16
    assert commands[0].target_code == 6552
    assert commands[1].target_code == 7208
    assert commands[1].step > 0
    assert all(command.target_code & 0b11 == 0 for command in commands)
    actual_slew_code_per_cycle = commands[1].step * 16 / (1 << 16)
    expected_slew_code_per_cycle = commands[0].target_code / 3_000
    assert actual_slew_code_per_cycle == pytest.approx(
        expected_slew_code_per_cycle,
        rel=2.0e-3,
    )

    tproc = TProcV1BehaviorModel(strict=True).run(
        program.prog_list,
        max_steps=1_000_000,
    )
    assert [event.word for event in tproc.output_events] == _expected_words(program)
    assert tproc.timing_conflicts == []
    assert program.summary()["bias_t_duration_execution"] == (
        "awg_flat_ramp_target_over_tau"
    )


def test_bias_t_filter_retargets_transitions_from_compensated_flat_endpoint():
    sequence = FineTuneSequence(("awg_0",))
    sequence.add_set("positive", (0.2,), 100)
    sequence.add_ramp("to_negative", 50)
    sequence.add_set("negative", (-0.1,), 200)
    sequence.set_bias_t_filter_compensation(1_000)

    levels = sequence.filter_compensated_segment_levels()
    assert np.asarray(levels[0]) == pytest.approx(np.asarray(((0.2,), (0.22,))))
    assert np.asarray(levels[1]) == pytest.approx(np.asarray(((0.22,), (-0.1,))))
    assert np.asarray(levels[2]) == pytest.approx(np.asarray(((-0.1,), (-0.12,))))


def test_bias_t_filter_large_hardware_sweep_stays_below_4096_pmem_dmem():
    instruction_counts = []
    large_summary = None
    for count in (3, 5_001):
        sequence = FineTuneSequence(("awg_0",))
        sequence.add_set("hold", (0.1,), 30)
        sequence.set_amplitude_sweep("hold", "awg_0", -0.2, 0.2, count)
        sequence.set_bias_t_filter_compensation(30_000)
        soccfg = _mock_soccfg(1)
        soccfg["tprocs"][0]["pmem_size"] = 4096
        program = sequence.make_program(soccfg, awg_channels=(0,))
        program.compile()
        instruction_counts.append(len(program.prog_list))
        large_summary = program.summary()

    # Integer register-delta quantization can remove a no-op update, so the
    # exact count may differ by a few instructions. It must not scale with the
    # 5,001 sweep points or materialize one instruction block per point.
    assert max(instruction_counts) < 64
    assert abs(instruction_counts[0] - instruction_counts[1]) <= 4
    assert large_summary["cartesian_point_count"] == 5_001
    assert large_summary["sweep_uses_point_table"] is False
    assert large_summary["tproc_pmem_words_used"] < 4096
    assert large_summary["tproc_pmem_capacity"] == 4096
    assert large_summary["tproc_dmem_capacity"] == 4096
    assert large_summary["tproc_dmem_words_reserved"] < 4096
    assert large_summary["tproc_dmem_words_required"] <= 4096
    assert all(
        0 <= address < 4096
        for address in large_summary["tproc_dmem_addresses"]
    )
    assert large_summary["tproc_memory_within_4096"] is True


def test_bias_t_filter_eight_outputs_stay_within_tprocessor_memory_limits():
    names = tuple(f"awg_{index}" for index in range(8))
    sequence = FineTuneSequence(names)
    sequence.add_set("hold", tuple(0.02 * (index + 1) for index in range(8)), 20)
    sequence.set_amplitude_sweep("hold", "awg_0", -0.1, 0.1, 5)
    sequence.set_bias_t_filter_compensation(20_000)
    soccfg = _mock_soccfg(8)
    soccfg["tprocs"][0]["pmem_size"] = 4096
    program = sequence.make_program(
        soccfg,
        awg_channels=tuple(range(8)),
    )
    program.compile()
    summary = program.summary()

    assert summary["commands_per_point"] == 16
    assert summary["bias_t_compensation_type"] == "filter"
    assert summary["bias_t_simultaneous_start"] is False
    assert summary["tproc_pmem_words_used"] < 4096
    assert summary["tproc_pmem_capacity"] == 4096
    assert summary["tproc_dmem_capacity"] == 4096
    assert summary["tproc_dmem_words_reserved"] < 4096
    assert summary["tproc_dmem_words_required"] <= 4096
    assert summary["tproc_memory_within_4096"] is True
