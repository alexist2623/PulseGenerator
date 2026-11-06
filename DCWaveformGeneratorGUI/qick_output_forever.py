"""Output-only execution: no ADC/DDR arm, acquisition buffers or dataset writes."""
import time


def execute_output_forever(soc, program, *, cancel_check, progress_callback=None):
    from qick_qcodes_experiment import ExperimentCancelled
    if program.execution_mode != 'output_forever' or program.ro_chs:
        raise ValueError('Run forever requires an output-only program without readouts')
    required = ('begin_output_forever', 'renew_output_forever', 'stop_output_forever',
                'check_output_forever_idle')
    if any(not callable(getattr(soc, name, None)) for name in required):
        raise RuntimeError('Run forever requires the updated v8 QICK server driver')
    cancel_check()
    soc.check_output_forever_idle(require_capture_idle=True)
    program.config_all(soc, reset=True, load_envelopes=True, load_mem=False)
    program._load_runtime_dmem(soc, reload_mem=True)
    cancel_check()
    token = None
    cancelled = None
    try:
        token = soc.begin_output_forever(tuple(program.awg_channels))['token']
        # No cooperative cancellation between leasing and starting: Stop is
        # serviced by the first completed shot and its zero-output epilogue.
        soc.start_src('internal')
        soc.start_tproc()
        started = time.monotonic()
        while True:
            cancel_check()
            state = soc.renew_output_forever(token)
            if state['state'] != 'running':
                raise RuntimeError(f'Output-only board session stopped: {state}')
            if progress_callback is not None:
                progress_callback(0, f'Output only — {time.monotonic()-started:.1f} s; '
                                  'repeating reps × complete sweep; no acquisition or saving')
            time.sleep(0.2)
    except ExperimentCancelled as exc:
        cancelled = exc
    finally:
        if token is not None:
            try:
                status = soc.stop_output_forever(token)
                if status.get('state') != 'stopped':
                    raise RuntimeError(str(status))
            except Exception as exc:
                raise RuntimeError(f'Output-only cleanup incomplete: {exc}') from exc
    if cancelled is not None:
        raise ExperimentCancelled('Output stopped at a shot boundary; digital zero acknowledged. '
                                  'No measurement saved.') from cancelled


def run_output_forever(*, connection_config, sequence, awg_channels,
                       repetitions_per_sweep, rf_specs=(), readout_spec=None,
                       compile_validation_mode='full', cancel_check,
                       progress_callback=None, event_callback=None, connector=None,
                       tproc_mhz=None):
    if readout_spec is not None:
        raise ValueError('Disable RF Readout before Run forever; ADC acquisition is forbidden')
    if any(spec.software_sweep_axes for spec in rf_specs):
        raise ValueError('Run forever supports hardware sweeps; host RF sweeps require finite execution')
    from qick_qcodes_experiment import connect_qick, build_qick_program, configure_rf_output
    from qick_dac_current import verify_current_settings
    cancel_check()
    soc, soccfg = connect_qick(connection_config, connector=connector)
    cancel_check()
    check_idle = getattr(soc, 'check_output_forever_idle', None)
    if not callable(check_idle):
        raise RuntimeError('Run forever requires the updated v8 QICK server driver')
    check_idle(require_capture_idle=True)
    program = build_qick_program(soccfg, sequence, awg_channels=awg_channels,
        repetitions_per_sweep=repetitions_per_sweep, rf_specs=rf_specs,
        tproc_mhz=tproc_mhz, execution_mode='output_forever',
        compile_validation_mode=compile_validation_mode, cancel_check=cancel_check)
    program.compile()
    verify_current_settings(soc, sequence.dac_current_settings)
    cancel_check()
    for spec in rf_specs:
        configure_rf_output(soc, spec)
    for square in program.square_pulse_configs:
        soc.rfb_set_gen_dc(square.gen_ch)
    if event_callback:
        guard_us = program.output_forever_guard_cycles / program.tproc_mhz
        event_callback('output_forever', 'started',
                       f'Output-only loop; no readout. Preparation gap {guard_us:.3f} µs '
                       'per shot plus per grid; pulse-internal timing unchanged.')
    execute_output_forever(soc, program, cancel_check=cancel_check,
                           progress_callback=progress_callback)
