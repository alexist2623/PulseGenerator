"""Read-only, compact coordinates of the commands compiled for an experiment.

Sequence coordinates remain the requested/sequence axes used by existing plots.
Physical AWG target codes and RF register values are separate observables: a
virtual member does not map to one DAC after matrix multiplication. These are
digital command values, not measured analog voltages or the final RC-IIR output.
"""
from dataclasses import asdict, is_dataclass

import numpy as np

from qick_dc_area import executed_commands


SCHEMA = "qick-execution-coordinates-v1"


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _column(values):
    """A scalar is broadcast over point_count; a list is a point-indexed column."""
    values = [_plain(value) for value in values]
    if values and all(value == values[0] for value in values[1:]):
        return values[0]
    return values


def _check(cancel_check):
    if cancel_check is not None:
        cancel_check()


def _event_time(program, models, key, default, point):
    model = models.get(key)
    return int(default if model is None else program._sweep_model_value(model, point))


def _rf_values(program, event, point, indices):
    runtime = program._rf_runtime[event]
    values = {name: int(runtime[name]) for name in ("freq", "phase", "gain")}
    mode = None
    for table in program._rf_point_tables_by_event.get(event, ()):
        if table.get("fine_row"):
            raise ValueError("legacy fine-duration point tables are not supported by this schema")
        row = int(np.ravel_multi_index(
            tuple(indices[int(axis)] for axis in table["axis_indices"]),
            tuple(table["axis_shape"]), order="C"))
        value = int(table["values"][row])
        name = table["register_name"]
        if name == "rf_duration":
            mode = value
        elif name == "rf_frequency":
            values["freq"] = value
        elif name == "rf_gain":
            values["gain"] = value
        else:
            raise ValueError(f"unrecognized RF register table {name!r}")
    return values, mode


def _frequency_mhz(program, channel, word):
    try:
        return float(program.reg2freq(word, gen_ch=channel))
    except KeyError as exc:
        # Compact desktop configs use f_dds directly, as does the compiler.
        if exc.args not in (("refclk_freq",), ("fs_mult",), ("fdds_div",)):
            raise
        cfg = program.soccfg["gens"][channel]
        return word * float(cfg["f_dds"]) / (1 << int(cfg["b_dds"]))


def _selected_points(count, point_indices):
    if point_indices is None:
        return range(count)
    values = tuple(point_indices)
    if not values or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                         or not 0 <= int(value) < count for value in values):
        raise ValueError("point_indices must contain valid integer point indices")
    if len(set(values)) != len(values):
        raise ValueError("point_indices must not contain duplicates")
    return tuple(map(int, values))


def _single(program, cancel_check, point_indices=None):
    sequence = program.sequence
    if any((program.soccfg["gens"][channel].get("type"),
            int(program.soccfg["gens"][channel].get("output_sum_version", 0)))
           not in (("axis_awg_tuning_v3", 2), ("axis_awg_tuning_v4", 3), ("axis_awg_tuning_v7", 7))
           for channel in program.awg_channels):
        raise ValueError("execution coordinate export requires v5 or v6 AWG/output-sum metadata")
    if point_indices is None:
        cached = getattr(program, "_execution_coordinate_snapshot", None)
        if cached is not None:
            return cached
    count = int(sequence.sweep_point_count)
    points = _selected_points(count, point_indices)
    axes = tuple(sequence.sweep_axes)
    coordinates = np.asarray(sequence.sweep_coordinates)
    models = {model["key"]: model for model in program._event_timing_models}
    first_commands = tuple(executed_commands(program, 0))
    awg_columns = [{name: [] for name in (
        "target_code_16", "target_code_14", "step", "duration_samples",
        "command_cycle", "ramp_duration_us", "target_normalized")} for _ in first_commands]
    rf_columns = [{name: [] for name in (
        "frequency_word", "frequency_mhz", "phase_word", "logical_gain",
        "emitted_start_gain", "emitted_tail_gain", "active_samples",
        "active_duration_us", "busy_cycles", "command_start_cycle",
        "tail_command_cycle", "stop_command_cycle", "periodic_plateau",
        "scheduled_stop_duration_samples")} for _ in program.rf_pulse_configs]
    group_columns = [[] for _ in sequence.loop_axis_indices]
    segment_columns = [[] for _ in sequence.segments]
    for point in points:
        _check(cancel_check)
        indices = sequence.model_indices(point)
        for column, representative in zip(group_columns, sequence.loop_axis_indices):
            column.append(int(indices[representative]))
        for index, column in enumerate(segment_columns):
            column.append(int(sequence.segment_duration_cycles_at(point, index)))
        for command, column in zip(executed_commands(program, point), awg_columns):
            cfg = program.soccfg["gens"][command.gen_ch]
            target = int(command.target_code)
            column["target_code_16"].append(target)
            column["target_code_14"].append(target >> 2)
            column["target_normalized"].append((target >> 2) / 8192.)
            column["step"].append(int(command.step))
            column["duration_samples"].append(int(command.duration_samples))
            column["ramp_duration_us"].append(
                int(command.duration_samples) / (float(cfg["f_fabric"]) * int(cfg["n_pts"]))
                if command.kind == "ramp" else None)
            column["command_cycle"].append(_event_time(program, models,
                ("event_time", "awg", *program._command_key(command)),
                program.timing["command_times"][program._timing_key(command)], point))
        for event, (rf, column) in enumerate(zip(program.rf_pulse_configs, rf_columns)):
            runtime = program._rf_runtime[event]
            cfg = program.soccfg["gens"][rf.gen_ch]
            fabric = float(cfg["f_fabric"])
            lanes = int(cfg.get("samps_per_clk", 16))
            values, mode = _rf_values(program, event, point, indices)
            frequency = int(values["freq"]) & ((1 << int(cfg["b_dds"])) - 1)
            gain = int(values["gain"])
            column["frequency_word"].append(frequency)
            column["frequency_mhz"].append(_frequency_mhz(program, rf.gen_ch, frequency))
            column["phase_word"].append(int(values["phase"]) & 0xffffffff)
            column["logical_gain"].append(gain)
            if runtime.get("fine"):
                detail = program._fine_details_at(event, indices)
                own = program._fine_rf_rows[event]["own_axis_index"]
                member_index = 0 if own is None else sequence.point_indices(point)[own]
                plan = runtime["plans"][member_index]
                start, samples = int(detail["start"]), int(detail["samples"])
                busy = int(detail["busy_end"]) - start
                tail = int(detail["tail"]) if detail["active_tail"] else None
                stop = None
                plateau = bool(plan.periodic_plateau)
                start_gain = gain if plan.long else 2 * gain
                tail_gain = 2 * gain if plan.long else None
                scheduled_samples = None
            else:
                start = _event_time(program, models, ("event_time", "rf_start", event),
                                    program.aux_timing[f"rf_{event}_start"], point)
                tail = None
                tail_gain = None
                plateau = bool(runtime["periodic"])
                stop = (_event_time(program, models, ("event_time", "rf_stop", event),
                                    program.aux_timing[f"rf_{event}_end"], point)
                        if plateau else None)
                duration = ((stop - start) * fabric / float(program.tproc_mhz)
                            if plateau else int(runtime["base_length_cycles"]) if mode is None else mode & 0xffff)
                if abs(duration - round(duration)) > 1e-9:
                    raise ValueError("RF command duration does not align with the generator fabric")
                samples = int(round(duration)) * lanes
                busy = int(round(duration * float(program.tproc_mhz) / fabric))
                start_gain = gain
                scheduled_samples = samples if plateau else None
                if plateau:
                    # A periodic SGv6 word accepts stop at its word boundary.
                    # Scheduled stop time alone is not a measured active edge.
                    samples = busy = None
            for name, value in (("emitted_start_gain", start_gain), ("emitted_tail_gain", tail_gain),
                                ("active_samples", samples), ("active_duration_us", None if samples is None else samples / (fabric * lanes)),
                                ("busy_cycles", busy), ("command_start_cycle", start),
                                ("tail_command_cycle", tail), ("stop_command_cycle", stop),
                                ("periodic_plateau", plateau),
                                ("scheduled_stop_duration_samples", scheduled_samples)):
                column[name].append(value)
    awg = []
    scales = getattr(sequence, "output_full_scales_mv", None)
    for command, columns in zip(first_commands, awg_columns):
        cfg = program.soccfg["gens"][command.gen_ch]
        item = dict(output_name=sequence.output_names[command.output_index],
                    output_index=int(command.output_index), gen_ch=int(command.gen_ch),
                    segment_name=command.segment_name, segment_index=int(command.segment_index),
                    kind=command.kind, step_fraction_bits=int(cfg["frac"]),
                    scalar_sample_rate_mhz=float(cfg["f_fabric"]) * int(cfg["n_pts"]),
                    columns={name: _column(values) for name, values in columns.items()})
        from qick_fine_ramp import is_extended
        if is_extended(command):
            item['extended_ramp_version'] = 1
            item['command_words_u32'] = list(command.words)
            item['edge_positions'] = dict(sequence.fine_ramps[command.segment_name][command.output_index])
            item['duration_note'] = 'Sample count includes final endpoint; word padding holds terminal voltage.'
        if scales is not None:
            scale = float(scales[command.output_index])
            item["full_scale_mv"] = scale
            item["columns"]["target_mv"] = _column(value * scale for value in columns["target_normalized"])
        awg.append(item)
    rf_events = [dict(event_index=index, event_id=str(rf.event_id), gen_ch=int(rf.gen_ch),
                      segment_name=str(rf.at_segment), fine_duration=bool(rf.fine_duration),
                      duration_parameter=str(rf.duration_parameter),
                      frequency_parameter=str(rf.frequency_parameter),
                      columns={name: _column(values) for name, values in columns.items()})
                 for index, (rf, columns) in enumerate(zip(program.rf_pulse_configs, rf_columns))]
    return dict(schema=SCHEMA, execution="hardware_program", point_count=len(points), total_point_count=count,
        column_encoding="A scalar broadcasts to every point; a list is indexed by point_index.",
        point_index=_column(points), sweep_shape=list(sequence.sweep_shape),
        repetitions_per_point=int(program.cfg["reps"]),
        coordinate_domain="Compiled digital commands; AWG targets are physical pre-IIR DAC values, not measured analog or final RC output.",
        requested_members=[dict(member_index=index, key=list(sequence.sweep_axis_key(axis)),
            parameter_id=sequence.sweep_member_ids[index], unit=str(axis.coordinate_unit),
            coordinates=_column(coordinates[list(points), index])) for index, axis in enumerate(axes)],
        requested_coordinate_note="Sequence coordinates are preserved: voltage/RF members retain requested values; AWG duration axes use the existing cycle-quantized grid.",
        sweep_groups=_plain(sequence.sweep_groups),
        group_indices=[dict(loop_axis_index=index, coordinates=_column(values))
                       for index, values in enumerate(group_columns)],
        cross_capacitance=_plain(sequence.cross_capacitance),
        segment_duration_fabric_cycles=[dict(segment_name=segment.name, coordinates=_column(values))
                                       for segment, values in zip(sequence.segments, segment_columns)],
        tproc_mhz=float(program.tproc_mhz), awg_commands=awg, rf_events=rf_events,
        periodic_rf_note="Coarse periodic RF records the scheduled stop interval; active duration and busy cycles remain null because stop acceptance is word-boundary aligned. Fine periodic continuation is aligned by its compiled plan.",
        hardware_dc_compensation=bool(getattr(program, "_hardware_dc_area", False)
                                      and sequence.bias_t_compensation is not None),
        compensation_note="FPGA-generated DC compensation intervals and RC-filtered waveforms are not parameter-coordinate columns.")


def build_execution_coordinates(program, cancel_check=None, point_indices=None):
    """Export register-model values without compiling or running a program.

    Optional indices select original hardware points or, for a software bundle,
    combined acquisition points. Bundle entries remain grouped by host program;
    their combined_point_indices identify the original acquisition positions.
    """
    _check(cancel_check)
    programs = getattr(program, "programs", None)
    if programs is None:
        return _single(program, cancel_check, point_indices)
    programs = tuple(programs)
    coordinates = np.asarray(program.sweep_coordinates)
    if coordinates.shape != (len(programs), len(program.sweep_axes)):
        raise ValueError("software bundle coordinates do not match programs and axes")
    total = sum(int(item.sequence.sweep_point_count) for item in programs)
    selected = _selected_points(total, point_indices)
    points = []
    offset = 0
    for index, item in enumerate(programs):
        _check(cancel_check)
        count = int(item.sequence.sweep_point_count)
        local = [point - offset for point in selected if offset <= point < offset + count]
        if not local:
            offset += count
            continue
        snapshot = _single(item, cancel_check, None if point_indices is None else local)
        points.append(dict(software_point_index=index, software_coordinate=_plain(coordinates[index]),
                           combined_point_offset=offset,
                           combined_point_indices=_column(point + offset for point in local), hardware=snapshot))
        offset += count
    return dict(schema=SCHEMA, execution="software_program_bundle", point_count=len(selected),
                total_point_count=total, point_index=_column(selected),
                software_sweep_axes=_plain(program.sweep_axes),
                software_sweep_shape=[int(axis.count) for axis in program.sweep_axes],
                software_points=points)
