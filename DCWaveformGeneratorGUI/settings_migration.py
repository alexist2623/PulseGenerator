"""Import legacy waveform recipes without changing their physical definitions."""
from copy import deepcopy
from math import isfinite

RECIPE_SCHEMA = 'qick-awg-waveform-recipe-v1'


def recipe_settings(recipe, baseline, *, schema, version, awg_channels):
    """Convert the original SET/RAMP recipe to native GUI settings.

    Recipes do not contain RF, acquisition, or wiring configuration. Keep the
    GUI's connection/channel choices, but disable outputs absent from the file.
    Reject unrepresentable forms instead of silently dropping their fields.
    """
    def positive(value, name):
        value = float(value)
        if not isfinite(value) or value <= 0:
            raise ValueError(f'recipe {name} must be positive and finite')
        return value

    fabric = positive(recipe['fabric_mhz'], 'fabric_mhz')
    scale = positive(recipe['full_scale_mv'], 'full_scale_mv')
    names = recipe['output_names']
    segments = recipe['segments']
    if not isinstance(names, list) or not 1 <= len(names) <= 8 or len(set(names)) != len(names):
        raise ValueError('recipe output_names must contain 1-8 unique names')
    if not isinstance(segments, list) or not segments or len(segments) % 2 != 1:
        raise ValueError('recipe must start/end with SET and alternate SET/RAMP')
    if recipe.get('fine_ramps') or recipe.get('sweep_groups'):
        raise ValueError('this legacy recipe importer requires ungrouped SET/RAMP vertices; use settings JSON for fine packets/groups')
    channels = list(awg_channels)[:len(names)]
    if len(channels) != len(names):
        raise ValueError('configure enough AWG output channels before importing this recipe')
    times = [0.]
    values = [[] for _ in names]
    hold_names = []
    cursor = 0.
    for i, segment in enumerate(segments):
        kind = 'set' if i % 2 == 0 else 'ramp'
        if segment.get('kind') != kind:
            raise ValueError('recipe must start/end with SET and alternate SET/RAMP')
        duration = positive(segment['duration_cycles'], 'duration_cycles') * 1000 / fabric
        if kind == 'ramp':
            cursor += duration
            times.append(cursor)
            continue
        amplitudes = segment['amplitudes']
        if len(amplitudes) != len(names):
            raise ValueError('recipe amplitude count must match output_names')
        hold_names.append(segment['name'])
        for j, amplitude in enumerate(amplitudes):
            if amplitude is None:
                if not values[j]:raise ValueError('initial recipe SET needs every output voltage')
                voltage = values[j][-1]
            else:
                voltage = float(amplitude) * scale
            if not isfinite(voltage):raise ValueError('recipe voltages must be finite')
            values[j].extend((voltage, voltage))
        cursor += duration
        times.append(cursor)
    doc = deepcopy(baseline)
    doc.update(schema=schema, version=version, sweep_grouping={})
    doc['awg'] = dict(outputs=[dict(time_ns=times, voltage_mv=v,
        voltage_bounds_mv=[-scale, scale], segment_names=hold_names) for v in values],
        cross_capacitance=deepcopy(recipe['cross_capacitance']), sweeps=[])
    name_map = {name:f'awg_{i}' for i,name in enumerate(names)}
    segment_map = {name: f'set_{i}' for i, name in enumerate(hold_names)}
    for axis in recipe.get('sweep_axes', ()):
        if axis.get('axis_kind', 'amplitude') != 'amplitude' or axis.get('coordinate_unit', 'normalized') != 'normalized':
            raise ValueError('legacy recipe import currently requires normalized amplitude sweeps; use settings JSON for duration/RF sweeps')
        doc['awg']['sweeps'].append(dict(axis_kind='amplitude', segment_name=segment_map[axis['segment_name']],
            output_name=name_map[axis['output_name']], start=axis['start'], stop=axis['stop'], count=axis['count']))
    qick = doc['qick']
    qick.update(fabric_mhz=fabric, full_scale_mv=scale, awg_channels=channels)
    comp = recipe.get('bias_t_compensation')
    if comp:
        kind = comp.get('kind', 'dc')
        if kind not in ('dc', 'filter'):raise ValueError('unknown recipe compensation kind')
        qick['bias_t_compensation'] = dict(enabled=True, type=kind,
            mode=comp.get('mode', 'fixed_time'), voltage_mv=float(comp.get('amplitude', 0.1))*scale,
            duration_us=float(comp.get('fixed_duration_cycles', 300))/fabric,
            filter_tau_us=float(comp.get('tau_cycles', 30000))/fabric)
    else:qick['bias_t_compensation'] = dict(enabled=False)
    for key in ('rf_outputs', 'sparameter_rf_outputs'):
        for output in doc.get(key, ()):output['enabled'] = False
    doc['rf_readout'] = dict(enabled=False)
    # Separate experiment modes have no definition in this recipe.
    for key in ('square_wave', 'square_pulse', 'output_trigger'):
        doc.pop(key, None)
    doc['display'].update(selected_awg_output=0, waveform_point_index=0,
                          selected_awg_tuning_tab_name='Experiment')
    return doc


def migration_report(source_schema, source_version, target_version, *, recipe=False):
    changes = ['Loaded into the current settings format; missing fields use current defaults.',
               'Pulse vertices, sweep definitions and compensation settings are retained.',
               'RF duration uses automatic sample resolution. Duration sweep steps use integer DAC samples.',
               'On AWG v7, a completed RAMP holds its terminal voltage. Duplicate post-RAMP SET commands are omitted; hold time and RF delay are retained.']
    if recipe:
        changes[:2] = ['Converted the waveform recipe into editable GUI pulse vertices, voltage sweeps and compensation settings.',
            'The recipe has no wiring/RF/readout configuration: existing AWG channel assignments are retained; RF and acquisition are disabled.']
    changes.append('The source file has not been overwritten. Saving writes the current format.')
    return dict(source_schema=source_schema, source_version=source_version,
                target_version=target_version, changes=changes)
