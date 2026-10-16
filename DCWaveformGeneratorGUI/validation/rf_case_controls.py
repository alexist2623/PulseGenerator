"""Actual Qt inputs for Single RF pulses and duration/frequency sweeps.

This adapter does not click Run or touch instruments. Physical amplitude is an
independent reference input: it is never converted into a GUI gain code here.
Composite and calibration controls require additional bindings and are
explicitly unavailable rather than silently reduced to manual Single pulses.
"""
from copy import deepcopy
import math

from PyQt5 import QtWidgets

from .awg_reference import ReferenceUnavailable
from .mainwindow_cases import _text
from .qt_case_controls import reveal
from .qt_runner import QtAction


class RfCaseBinding:
    """Prepare Single pulses after the A/B waveform binding has finished.

    ``case['rf_pulses']`` requires separate ``gain`` and ``amplitude_v`` fields.
    The former is the manual signed DAC gain requested in the GUI. The latter is
    only the physical reference scale and requires independent calibration in a
    later physical comparison. ``anchor_stage`` is a zero-based SET index.
    Optional ``duration_sweep={start_us, stop_us, count}`` and
    ``frequency_sweep={start_mhz, stop_mhz, count}`` use actual RF controls.
    Omitting either disables it; these are independent Cartesian axes. This
    adapter does not infer grouping or adjust the requested anchor boundaries.
    """
    def __init__(self, window, fixture, case):
        fixture.validate()
        self.window, self.fixture = window, fixture
        self.case = deepcopy(case)
        pulses = self.case.get('rf_pulses', ())
        if not pulses or len(pulses) > window._rf_ports_panel.MAX_PORTS:
            raise ValueError('Declare one through MAX_PORTS RF pulses')
        generators = set()
        for pulse in pulses:
            if (pulse.get('pulse_mode', 'single') != 'single' or
                    pulse.get('amplitude_mode', 'manual') != 'manual' or
                    pulse.get('power_calibration_enabled', False) or
                    any(pulse.get(k) for k in ('composite_items', 'power_sweep_enabled'))):
                raise ReferenceUnavailable('RF case controls currently support Single/manual pulses only')
            for key in ('name', 'generator', 'anchor_stage', 'delay_us', 'duration_us',
                        'duration_mode', 'frequency_mhz', 'gain', 'amplitude_v', 'phase_degrees',
                        'require_within_segment'):
                if key not in pulse:
                    raise ValueError(f'RF case requires explicit {key}; physical amplitude is not a GUI gain')
            if not isinstance(pulse['name'], str) or not pulse['name'].strip():
                raise ValueError('RF pulse name must be nonempty')
            pulse['name'] = pulse['name'].strip()
            if type(pulse['generator']) is not int or not 0 <= pulse['generator'] <= 255:
                raise ValueError('Invalid RF generator index')
            if pulse['generator'] in generators:
                raise ValueError('This RF binding requires distinct generators')
            generators.add(pulse['generator'])
            if type(pulse['anchor_stage']) is not int or pulse['anchor_stage'] < 0:
                raise ValueError('anchor_stage must be a zero-based SET index')
            if type(pulse['gain']) is not int or not -32768 <= pulse['gain'] <= 32767:
                raise ValueError('Manual gain must be a signed 16-bit integer')
            if pulse['duration_mode'] not in ('fine', 'coarse'):
                raise ValueError('Declare fine or coarse duration')
            if type(pulse['require_within_segment']) is not bool:
                raise ValueError('require_within_segment must be boolean')
            for key in ('delay_us', 'duration_us', 'frequency_mhz', 'amplitude_v', 'phase_degrees'):
                if not math.isfinite(pulse[key]):
                    raise ValueError(f'RF {key} must be finite')
            if pulse['delay_us'] < 0 or pulse['duration_us'] <= 0 or pulse['amplitude_v'] < 0:
                raise ValueError('Invalid RF delay, duration, or physical reference amplitude')
            for name, unit in (('duration', 'us'), ('frequency', 'mhz')):
                key = name + '_sweep'
                sweep = pulse.get(key)
                if name + '_sweep_enabled' in pulse and bool(pulse[name + '_sweep_enabled']) != (sweep is not None):
                    raise ValueError(f'{key}_enabled must agree with the explicit {key} mapping')
                if sweep is None:
                    pulse[key] = None
                    continue
                if not isinstance(sweep, dict) or set(sweep) != {f'start_{unit}', f'stop_{unit}', 'count'}:
                    raise ValueError(f'{key} requires start_{unit}, stop_{unit}, and count')
                if type(sweep['count']) is not int or not 1 <= sweep['count'] <= 1_000_000:
                    raise ValueError(f'{key} count must be an integer from 1 through 1000000')
                bounds = [sweep[f'{end}_{unit}'] for end in ('start', 'stop')]
                if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bounds):
                    raise ValueError(f'{key} bounds must be finite numbers')
                if name == 'duration' and min(bounds) <= 0:
                    raise ValueError('duration_sweep bounds must be positive')
                if name == 'frequency' and (min(bounds) < -10000 or max(bounds) > 10000):
                    raise ValueError('frequency_sweep bounds exceed the GUI range')

    def _idle(self):
        if not self.window.isVisible() or self.window._experiment_thread is not None:
            raise ValueError('Use a dedicated visible idle MainWindow')

    def _mapping(self, pulse):
        cfg = self.window._qick_configuration
        if cfg is None:
            raise ValueError('Identify the loaded configuration before RF preparation')
        matches = [p for p in cfg.outputs if pulse['generator'] in p.qick_channels]
        if len(matches) != 1:
            raise ValueError('RF generator is absent or ambiguous in the identified configuration')
        output = matches[0]
        path = dict(zip(output.qick_channels, output.block_paths))[pulse['generator']]
        if 'axis_signal_gen_v6' not in path.lower():
            raise ReferenceUnavailable('RF case binding requires an identified SGv6 generator')
        connected = [(str(channel), item) for channel, item in self.fixture.channels.items()
                     if item['physical_connector'] == output.label]
        if len(connected) != 1:
            raise ValueError('RF connector has no unique explicitly wired scope channel')
        return dict(scope_channel=connected[0][0], physical_connector=output.label,
                    generator=pulse['generator'])

    def actions(self):
        self._idle()
        w, port = self.window, self.window._rf_ports_panel
        if len(port._panels) > len(self.case['rf_pulses']):
            raise ValueError('Do not remove unrelated RF ports; use a dedicated window')
        yield QtAction(w._control_tabs, 'tab', w._control_tabs.indexOf(w._awg_tuning_page))
        yield QtAction(w._awg_tuning_tabs, 'tab', w._awg_tuning_tabs.indexOf(port))
        while len(port._panels) < len(self.case['rf_pulses']):
            previous_count = len(port._panels)
            reveal(port.add_button)
            yield QtAction(port.add_button, 'click')
            if len(port._panels) != previous_count + 1:
                raise ValueError('Add RF Port did not append exactly one port; preparation stopped')
        for panel, pulse in zip(port._panels, self.case['rf_pulses']):
            self._mapping(pulse)
            reveal(panel)
            yield QtAction(panel, 'check', True)
            reveal(panel.single_mode)
            yield QtAction(panel.single_mode, 'click')
            yield _text(panel.gen_ch, pulse['generator'])
            anchor = panel.segment.findData(f"set_{pulse['anchor_stage']}")
            if anchor < 0:
                raise ValueError('RF anchor stage is not present in the actual AWG controls')
            reveal(panel.segment)
            yield QtAction(panel.segment, 'combo', anchor)
            for field in (panel.duration_sweep_enabled, panel.frequency_sweep_enabled,
                          panel.power_calibration_group):
                reveal(field)
                if field.isEnabled():
                    yield QtAction(field, 'check', False)
                elif field.isChecked():
                    raise ValueError('An unavailable RF control retains an enabled feature')
            reveal(panel.segment_length_mode)
            yield QtAction(panel.segment_length_mode, 'combo', panel.segment_length_mode.findData('fixed'))
            # Time units are read from the real editor; no top-level setting is changed.
            factor = {'ns': 1000., 'us': 1., 'ms': .001, 's': 1e-6}.get(panel._time_unit)
            if factor is None:
                raise ValueError('Unsupported RF editor time unit')
            for field, value in ((panel.pulse_name, pulse['name']), (panel.delay, pulse['delay_us']*factor),
                    (panel.duration, pulse['duration_us']*factor), (panel.frequency_mhz, pulse['frequency_mhz']),
                    (panel.gain, pulse['gain']), (panel.phase_degrees, pulse['phase_degrees'])):
                yield _text(field, value)
            for field, value in ((panel.fine_duration, pulse['duration_mode']=='fine'),
                                 (panel.require_within, pulse['require_within_segment'])):
                reveal(field)
                yield QtAction(field, 'check', value)
            for name, unit, scale in (('duration', 'us', factor), ('frequency', 'mhz', 1.)):
                sweep = pulse[name + '_sweep']
                if sweep is None:
                    continue
                checkbox = getattr(panel, name + '_sweep_enabled')
                reveal(checkbox)
                yield QtAction(checkbox, 'check', True)
                for end in ('start', 'stop'):
                    field_name = name + '_sweep_' + end + ('_mhz' if name == 'frequency' else '')
                    yield _text(getattr(panel, field_name), sweep[f'{end}_{unit}'] * scale)
                yield _text(getattr(panel, name + '_sweep_count'), sweep['count'])
        self.verify_requested_case()

    def snapshot(self):
        self._idle()
        panels = self.window._rf_ports_panel._panels
        if len(panels) != len(self.case['rf_pulses']):
            raise ValueError('Actual RF port count differs from the declared case')
        result = []
        for panel, requested in zip(panels, self.case['rf_pulses']):
            spec = panel.configured_spec()
            result.append(dict(**self._mapping(requested), enabled=panel.isChecked(), name=spec.pulse_name,
                anchor_stage=int(spec.segment_name.split('_')[-1]), delay_us=spec.delay_us,
                duration_us=spec.duration_us, duration_mode='fine' if spec.fine_duration else 'coarse',
                frequency_mhz=spec.frequency_mhz, gain=spec.gain, phase_degrees=spec.phase_degrees,
                require_within_segment=spec.require_within_segment, pulse_mode=spec.pulse_mode,
                power_calibration_enabled=spec.power_calibration_enabled,
                duration_sweep_enabled=spec.duration_sweep_enabled,
                frequency_sweep_enabled=spec.frequency_sweep_enabled,
                duration_sweep=(dict(start_us=spec.duration_sweep_start_us,
                    stop_us=spec.duration_sweep_stop_us, count=spec.duration_sweep_count)
                    if spec.duration_sweep_enabled else None),
                frequency_sweep=(dict(start_mhz=spec.frequency_sweep_start_mhz,
                    stop_mhz=spec.frequency_sweep_stop_mhz, count=spec.frequency_sweep_count)
                    if spec.frequency_sweep_enabled else None),
                segment_length_mode=spec.segment_length_mode, nqz=spec.nqz,
                att1_db=spec.att1_db, att2_db=spec.att2_db, filter_type=spec.filter_type,
                reference_amplitude_v=requested['amplitude_v'],
                reference_calibration_evidence=deepcopy(requested.get('amplitude_calibration_evidence'))))
            if spec.gen_ch != requested['generator']:
                raise ValueError('RF generator readback differs from the declared mapping')
        return dict(schema='qick-single-rf-controls-v2', rf_pulses=result,
            value_domain='Requested GUI RF values; gain is a code, reference amplitude is separate',
            sweep_coordinate_domain='GUI committed requested us/MHz; actual quantized coordinates are separate',
            physical_amplitude_verified=False, run_performed_by_binding=False)

    def verify_requested_case(self):
        actual = self.snapshot()
        for found, wanted in zip(actual['rf_pulses'], self.case['rf_pulses']):
            for key in ('name', 'generator', 'anchor_stage', 'duration_mode', 'gain', 'require_within_segment'):
                if found[key] != wanted[key]:
                    raise ValueError(f'RF control {key} did not commit')
            for key in ('delay_us', 'duration_us', 'frequency_mhz', 'phase_degrees'):
                if not math.isclose(found[key], wanted[key], abs_tol=1e-9, rel_tol=1e-10):
                    raise ValueError(f'RF control {key} did not commit at the requested precision')
            for name, unit in (('duration', 'us'), ('frequency', 'mhz')):
                key = name + '_sweep'
                if found[name + '_sweep_enabled'] != (wanted[key] is not None):
                    raise ValueError(f'RF {key} checkbox did not commit')
                if wanted[key] is not None:
                    if found[key]['count'] != wanted[key]['count']:
                        raise ValueError(f'RF {key} point count did not commit')
                    for end in ('start', 'stop'):
                        field = f'{end}_{unit}'
                        if not math.isclose(found[key][field], wanted[key][field], abs_tol=1e-9, rel_tol=1e-10):
                            raise ValueError(f'RF {key} {field} did not commit at the requested precision')
            if (not found['enabled'] or found['pulse_mode'] != 'single' or found['power_calibration_enabled'] or
                    found['segment_length_mode'] != 'fixed'):
                raise ValueError('RF pulse contains undeclared composite, calibration, or segment-extension behavior')
        return actual
