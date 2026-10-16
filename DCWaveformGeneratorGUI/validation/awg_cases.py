"""Concrete A/B case plans and independent archive preparation, without Run.

The caller supplies verified wiring and fixed analog calibration. A dedicated
visible MainWindow is edited using normal Qt events. No board or scope is opened
here; GuiHardwareRunner subsequently owns the normal production Run path.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

from .archive import Point, Reference, RunArchive, atomic, json_file, npz_bytes
from .awg_reference import AwgReference, physical_references
from .coverage import logarithmic_durations_us


def basic_awg_cases(outputs, *, level_mv=100., sweep_limit_mv=300., point_count=5):
    """Return explicit independent runs and arithmetic/grouped hardware loops.

    Mapping/full scales come from the fixture, never presumed generator numbers.
    These specifications do not claim that every required physical category is
    implemented or measured; RF, acquisition, and failure cases have their own
    adapters. Every returned case uses three stages and distinct A/B voltages.
    """
    if (len(outputs) != 2 or type(point_count) is not int or point_count < 2
            or not all(math.isfinite(v) and v > 0 for v in (level_mv, sweep_limit_mv))):
        raise ValueError('Declare two outputs, positive finite levels, and at least two points')
    base = dict(name='set_only', category='voltage', execution='individual_gui',
        outputs=deepcopy(outputs), matrix=[[1., 0.], [0., 1.]], repetitions=1,
        stages=[dict(name='zero', ramp_us=0., hold_us=10., voltage_mv=[0., 0.]),
                dict(name='hold', ramp_us=1., hold_us=10., voltage_mv=[level_mv, -.6*level_mv]),
                dict(name='reset', ramp_us=1., hold_us=10., voltage_mv=[0., 0.])],
        axes=[], compensation=dict(dc_enabled=False, rc_enabled=False),
        required_capabilities=['sum_v2'])
    set_only = deepcopy(base)
    set_only['stages'][1]['ramp_us'] = set_only['stages'][2]['ramp_us'] = 0.
    set_only.update(adapter_status='unavailable', adapter_reason=(
        'Production PulseSequence requires positive incoming RAMP durations; '
        'this GUI adapter cannot create a SET-only multi-stage waveform.'))
    cases = [set_only]

    def variant(name, category, *, ramp=False):
        case = deepcopy(base)
        case.update(name=name, category=category)
        if ramp:
            case['stages'][1]['ramp_us'] = case['stages'][2]['ramp_us'] = 1.
        return case

    for kind in ('hold', 'ramp'):
        for index, duration in enumerate(logarithmic_durations_us()):
            case = variant(f'{kind}_log_{index:02d}', f'{kind}_duration', ramp=kind == 'ramp')
            case['execution'] = 'individual_gui_log_grid'
            case['nominal_duration_us'] = duration
            case['stages'][1][f'{kind}_us'] = duration
            cases.append(case)
        for descending in (False, True):
            case = variant(f'{kind}_arithmetic_{"descending" if descending else "ascending"}',
                           f'{kind}_duration', ramp=kind == 'ramp')
            start, stop = (10., .1) if descending else (.1, 10.)
            case.update(execution='hardware_loop', repetitions=2,
                        axes=[dict(kind=f'{kind}_duration', output=0, segment=1,
                                   start=start, stop=stop, count=point_count)])
            cases.append(case)
    grouped = variant('matrix_grouped_voltage', 'grouped_voltage', ramp=True)
    grouped.update(execution='hardware_loop', matrix=[[1., .1], [-.2, 1.]], repetitions=2,
        axes=[dict(kind='amplitude', output=0, segment=1, start=-sweep_limit_mv,
                   stop=sweep_limit_mv, count=point_count, group='voltage'),
              dict(kind='amplitude', output=1, segment=1, start=.5*sweep_limit_mv,
                   stop=-.5*sweep_limit_mv, count=point_count, group='voltage')])
    cases.append(grouped)
    mixed = deepcopy(grouped)
    mixed.update(name='matrix_grouped_voltage_independent_ramp', category='combined_axes')
    mixed['axes'].append(dict(kind='ramp_duration', output=0, segment=1,
                             start=.1, stop=1., count=3))
    cases.append(mixed)
    one = deepcopy(grouped)
    one.update(name='grouped_count_one_repetitions', category='repetition_reset', repetitions=3)
    for axis in one['axes']:
        axis['count'] = 1
    cases.append(one)
    for dc, rc, mode in ((False, False, 'fixed_time'), (True, False, 'fixed_time'),
                         (False, True, 'fixed_time'), (True, True, 'fixed_time'),
                         (True, True, 'fixed_voltage')):
        case = deepcopy(grouped)
        case.update(name=f'matrix_dc{int(dc)}_rc{int(rc)}_{mode}', category='matrix_dc_rc')
        case['compensation'] = dict(dc_enabled=dc, rc_enabled=rc, dc_mode=mode,
            dc_voltage_mv=100., dc_duration_us=10., rc_tau_us=30.)
        cases.append(case)
    # Detect unsafe mappings/scales and invalid grids before any GUI interaction.
    for case in cases:
        reference = AwgReference(case)
        for point in range(reference.point_count):
            reference.commands(point)
    return cases


def _points(reference):
    for point in range(reference.point_count):
        indices = reference.coordinates(point)
        members = []
        for index, axis in enumerate(reference.axes):
            i = indices[reference.member_groups[index]]
            value = float(axis['start']) if axis['count'] == 1 else (
                float(axis['start']) + (float(axis['stop'])-float(axis['start']))*i/(axis['count']-1))
            members.append(dict(member=index, kind=axis['kind'], segment=axis['segment'],
                output=axis.get('output', 0), requested=value,
                unit='mV' if axis['kind'] == 'amplitude' else 'us'))
        yield Point(point, dict(requested_members=members, independent_commands=reference.commands(point)),
                    repetitions=reference.spec.get('repetitions', 1),
                    groups={str(i): value for i, value in enumerate(indices)})


@dataclass(frozen=True)
class PreparedAwgCase:
    archive: RunArchive
    binding: object
    reference: AwgReference

    def create_runner(self, *, production_module, **device_callbacks):
        """Construct only. The caller must explicitly start the physical attempt."""
        from .qt_runner import GuiHardwareRunner
        return GuiHardwareRunner(window=self.binding.window, production_module=production_module,
            run_button=self.binding.window._experiment_panel.run_button,
            archive=self.archive, **device_callbacks)


def declared_compensation_builder(schedules, *, normalized_case):
    """Bind an explicit per-point timing contract to the independent DC/RC model.

    Schedules follow ``validate_case(case, fixture)`` duration-first point order.
    Bind that complete normalized case explicitly so schedules from a different
    axis order or waveform cannot be silently reused. They must come from the
    declared experiment/barriers, not fitted scope edges. No timing is inferred
    when the readout/marker/RF schedule is unknown.
    """
    from .compensation_reference import CompensationReference, CompensationSchedule
    declared = tuple(schedules)
    contract = deepcopy(normalized_case)
    if not declared or any(not isinstance(row, CompensationSchedule) for row in declared):
        raise ValueError('Declare one CompensationSchedule per sweep point')

    def build(reference, point, max_samples):
        if reference.spec != contract:
            raise ValueError('Compensation schedules belong to a different normalized case')
        if len(declared) != reference.point_count:
            raise ValueError('Compensation schedule count differs from the sweep point count')
        return CompensationReference(reference).waveforms(
            point, declared[point], max_samples=max_samples)
    return build


def prepare_awg_case(*, window, fixture, identity, case, output_models, root=None,
                     max_samples_per_point=4_800_000, waveform_builder=None):
    """Prepare intended arrays, commit real GUI events, read back, and seal.

    ``waveform_builder(reference, point, max_samples)`` may supply an explicitly
    independent DC/RC model with the same array schema. It must cover the entire
    declared compensation/reset interval. Omission supports compensation-off
    baseband cases only and raises rather than silently using an identity path.
    This function never clicks Run. Partial preparation remains an unsealed
    archive, with an explicit error, and cannot be used to execute hardware.
    """
    from .mainwindow_cases import MainWindowCaseBinding
    from .qt_runner import apply_actions
    if identity.firmware_version != 'v5':
        raise ValueError('This physical validation campaign targets verified v5 firmware')
    identity.verify(case.get('required_capabilities', ()))
    binding = MainWindowCaseBinding(window, fixture, case)
    reference = AwgReference(binding.case)
    # Keep the plan's case identifier stable. The normalized execution order is
    # a separate, explicit record; SuiteManifest hashes the original plan.
    target = RunArchive(root, case=case, fixture=fixture, identity=identity,
        points=_points(reference), execution=binding.case.get('execution', 'hardware_loop'))
    json_file(target.root/'case_requested.json', case)
    json_file(target.root/'case_normalized.json', binding.case)
    try:
        for point in range(reference.point_count):
            waveforms = (reference.waveforms(point, max_samples=max_samples_per_point)
                         if waveform_builder is None else
                         waveform_builder(reference, point, max_samples_per_point))
            physical = physical_references(waveforms, output_models)
            arrays = {key: waveforms[key] for key in ('time_s', 'requested_virtual_v',
                      'requested_physical_v', 'quantized_codes', 'pre_rc_codes') if key in waveforms}
            atomic(target.root/'independent'/f'p{point:08d}.npz', npz_bytes(**arrays))
            json_file(target.root/'independent'/f'p{point:08d}.json',
                {key: waveforms[key] for key in ('model', 'point', 'group_indices', 'commands',
                 'channels', 'schedule', 'time_origin', 'reference_plane', 'requested_reference_note')
                 if key in waveforms})
            for output, definition in enumerate(binding.case['outputs']):
                channel = str(definition['scope_channel'])
                for kind, column in (('requested', 'requested_physical_v'),
                                     ('quantized', 'quantized_physical_v')):
                    target.add_reference(point, channel, kind, Reference(
                        waveforms['time_s'], waveforms[column][:, output],
                        dict(units={'time': 's', 'voltage': 'V'}, time_origin=waveforms['time_origin'],
                             reference_plane='Physical DAC before external analog transfer',
                             model_version=waveforms['model'], transformations=[],
                             note=waveforms.get('requested_reference_note', '') if kind == 'requested' else '',
                             coordinate_domain='matrix-converted requested' if kind == 'requested'
                                               else 'independent quantized physical')))
                target.add_reference(point, channel, 'physical', physical[output])
        input_events = apply_actions(binding.actions())
        json_file(target.root/'qt_preparation_events.json', input_events)
        snapshot = binding.verify_requested_case()
        if snapshot['point_count'] != reference.point_count:
            raise ValueError('Committed GUI and independent case point counts differ')
        target.seal(gui_definition=snapshot)
    except Exception as exc:
        target.event('preparation_failed', error=f'{type(exc).__name__}: {exc}')
        target.checkpoint()
        raise
    return PreparedAwgCase(target, binding, reference)
