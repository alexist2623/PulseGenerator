"""CPMG pulse-count hardware loops shared by GUI and program compilation."""
def enabled(spec):
    return bool(spec.predefined_n_sweep_enabled
                and spec.predefined_n_sweep_execution == "hardware")


def definition(spec):
    if spec.pulse_mode != "composite" or spec.predefined_template != "cpmg":
        raise ValueError("Hardware N sweep requires the CPMG template")
    if not spec.t2_initial_pi2 or not spec.t2_final_pi2:
        raise ValueError("Hardware CPMG N sweep requires initial and final pi/2 pulses")
    if spec.fine_duration or spec.sweep_axes or spec.software_sweep_axes:
        raise ValueError("Hardware CPMG N sweep currently requires fixed RF parameters")
    start, stop, count = (int(spec.predefined_n_sweep_start),
                          int(spec.predefined_n_sweep_stop), int(spec.predefined_n_sweep_count))
    if count > 1 and (stop - start) % (count - 1):
        raise ValueError("Hardware CPMG N values must have an exact integer step")
    step = 0 if count == 1 else (stop - start) // (count - 1)
    if min(start, stop) < 1 or (count > 1 and step == 0):
        raise ValueError("Hardware CPMG N values must be distinct positive integers")
    base = spec.with_predefined_parameters(n_pulses=start)
    events = base.pulse_events
    if len(events) != start + 2:
        raise ValueError("Hardware CPMG expects N body pulses and two pi/2 pulses")
    probe = spec.with_predefined_parameters(n_pulses=start + 1)
    period = probe.pulse_events[-1].delay_us - events[-1].delay_us
    return base, events, start, step, count, period


def runtime_spec(spec):
    return definition(spec)[0] if enabled(spec) else spec


def runtime_events(spec):
    events = spec.pulse_events
    return (events[0], events[1], events[-1])


def attach_axis(sequence, spec, fabric_mhz):
    from qick_fine_tune_sweep import CpmgNSweep, cycles_from_us
    base, events, start, step, count, period = definition(spec)
    if sequence.sweeps:
        raise ValueError("Hardware CPMG N sweep currently requires a standalone N axis")
    segment = next((item for item in sequence.segments if item.name == spec.segment_name), None)
    if segment is None or segment.kind != "set":
        raise ValueError("Hardware CPMG N sweep requires an existing SET anchor")
    initial = final = int(segment.duration_cycles)
    if spec.segment_length_mode == "extend_by_rf_duration":
        initial += sum(cycles_from_us(value, fabric_mhz)
                       for value in base.segment_extension_components_us)
        final = initial + cycles_from_us(period, fabric_mhz) * step * (count - 1)
    sequence.sweeps.append(CpmgNSweep(
        segment_name=spec.segment_name, start=float(start),
        stop=float(start + step * (count - 1)), count=count, sequence_fabric_mhz=fabric_mhz,
        n_start=start, n_step=step, gen_ch=spec.gen_ch,
        body_event_id=events[1].event_id, final_event_id=events[-1].event_id,
        period_us=period, hold_start_cycles=initial, hold_stop_cycles=final,
    ))


def loop_axes(program):
    from qick_fine_tune_sweep import CpmgNSweep
    return [(index, axis) for index, axis in enumerate(program.sequence.sweep_axes)
            if isinstance(axis, CpmgNSweep)]


def add_loop_count_fields(program, fields):
    for index, axis in loop_axes(program):
        page, _ = program._gen_regmap[(axis.gen_ch, "t")]
        deltas = [0] * len(program.sequence.sweep_axes)
        deltas[index] = axis.n_step
        fields.append(dict(key=("cpmg_n", index), register_name="cpmg_n",
                           base=axis.n_start - 1, axis_deltas=tuple(deltas),
                           page=int(page), gen_ch=axis.gen_ch,
                           allocate_command_register=True, command_register=None))


def emit_loop_start(program, key, page, time_register):
    if key[:2] != ("event_time", "rf_start"):
        return None
    rf = program.rf_pulse_configs[key[2]]
    for index, axis in loop_axes(program):
        if rf.event_id != axis.body_event_id:
            continue
        from qick_fine_tune_sweep import cycles_from_us
        period = cycles_from_us(axis.period_us, program.tproc_mhz)
        if abs(period / program.tproc_mhz - axis.period_us) > 1e-9:
            raise ValueError("CPMG pulse period must be an exact tProcessor cycle count")
        if period < 32:
            raise ValueError("Hardware CPMG pulse period must be at least 32 tProcessor cycles")
        field = program._sweep_field_by_key[("cpmg_n", index)]
        counter = int(field["command_register"])
        program._write_swept_or_static_register(
            field["key"], page, counter, axis.n_start - 1, "CPMG N-1 loop counter")
        label = f"CPMG_BODY_{index}"
        program.label(label)
        return counter, period, label
    return None


def emit_loop_end(program, loop, page, time_register):
    if loop is not None:
        counter, period, label = loop
        program.mathi(page, time_register, time_register, "+", period,
                      "next CPMG pi pulse timestamp")
        program.loopnz(page, counter, label, "repeat CPMG pi pulse N times")


def validate_loop_event_order(program, events):
    """Reject events that would be queued too late by the serial inner loop."""
    from qick_fine_tune_sweep import cycles_from_us
    models = {model["key"]: model for model in program._event_timing_models}
    for index, axis in loop_axes(program):
        body_index = next(i for i, rf in enumerate(program.rf_pulse_configs)
                          if rf.event_id == axis.body_event_id)
        first = int(program.aux_timing[f"rf_{body_index}_start"])
        period = cycles_from_us(axis.period_us, program.tproc_mhz)
        for base, _, _, kind, payload in events:
            if base <= first:
                continue
            if kind == "rf_start" and payload[1].event_id == axis.final_event_id:
                continue
            if kind == "awg":
                command, explicit_key = payload
                key = explicit_key or ("event_time", "awg", *program._command_key(command))
            elif kind in {"ddr_readout", "ddr_trigger"}:
                key = ("event_time", kind)
            else:
                key = None
            delta = models.get(key, {}).get("axis_deltas", [0] * len(program.sequence.sweep_axes))[index]
            for position in (0, axis.count - 1):
                last = first + (axis.points[position] - 1) * period
                if int(base) + int(delta) * position <= last:
                    raise ValueError("Hardware CPMG cannot enqueue another event inside its pi-pulse loop; move the readout/other event before or after the complete train")
