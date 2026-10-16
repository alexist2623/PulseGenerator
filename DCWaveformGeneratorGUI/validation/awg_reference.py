"""Independent numerical expectations for the v5 A/B baseband case adapter.

The input is the declared case, never a production preview or compiled program.
The register grid follows the documented constant endpoint-increment contract.
Time zero is the first main SET at the DAC, after the common output latency.
Compensation and analog circuits require their own models and are not silently
treated as an identity path by this reference.
"""
from copy import deepcopy
import math

import numpy as np


class ReferenceUnavailable(ValueError):
    """The declared case needs a reference plane this model cannot certify."""


def physical_references(waveforms, output_models):
    """Apply predeclared, fixed path calibration; never fit the captured trace.

    A flat DC path or a first-order high-pass path is supported. Each model must
    identify its independently obtained calibration evidence. Other circuits
    remain unavailable rather than inheriting a misleading identity model.
    """
    from .archive import Reference
    if len(output_models) != 2:
        raise ValueError('Two independently specified analog paths are required')
    references = []
    for channel, model in enumerate(output_models):
        if not model.get('calibration_evidence'):
            raise ReferenceUnavailable('Analog reference requires fixed path calibration evidence')
        kind = model.get('kind')
        if kind not in ('flat_dc', 'first_order_highpass'):
            raise ReferenceUnavailable('Unsupported analog circuit model')
        gain, offset, delay = (float(model[key]) for key in ('gain', 'offset_v', 'delay_s'))
        if not all(math.isfinite(v) for v in (gain, offset, delay)) or gain <= 0:
            raise ValueError('Fixed path gain must be positive and calibration must be finite')
        time = np.asarray(waveforms['time_s'], dtype=float)
        values = np.asarray(waveforms['quantized_physical_v'][:, channel], dtype=float)
        if kind == 'first_order_highpass':
            tau = float(model['tau_s'])
            if not math.isfinite(tau) or tau <= 0 or len(time) < 2:
                raise ValueError('High-pass reference requires positive tau and a sampled time grid')
            intervals = np.diff(time)
            if not np.allclose(intervals, intervals[0], rtol=1e-9, atol=1e-18):
                raise ValueError('High-pass reference requires uniformly spaced samples')
            if model.get('initial_state') != 'settled_zero':
                raise ReferenceUnavailable('An independent initial circuit state is required')
            alpha = math.exp(-float(intervals[0]) / tau)
            filtered = np.empty_like(values)
            previous_input = previous_output = 0.
            for index, value in enumerate(values):
                previous_output = alpha * previous_output + float(value) - previous_input
                previous_input = float(value)
                filtered[index] = previous_output
            values = filtered
        references.append(Reference(time + delay, gain * values + offset, dict(
            units={'time': 's', 'voltage': 'V'}, time_origin=waveforms['time_origin'],
            reference_plane='Scope input after declared fixed analog transfer',
            model_version=waveforms['model'] + '/' + kind,
            transformations=deepcopy(model))))
    return tuple(references)


def _nearest_ratio(numerator, denominator):
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(int(numerator)) + denominator // 2) // denominator)


def _cycles(us, mhz):
    value = float(us) * mhz
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Duration must be finite and positive')
    nearest = round(value)
    return max(1, nearest if math.isclose(value, nearest, rel_tol=1e-12, abs_tol=1e-12)
               else math.ceil(value))


class AwgReference:
    """Two physical outputs, SET/HOLD and incoming RAMP stages, shared groups."""
    def __init__(self, specification, *, fabric_mhz=300., fraction_bits=18):
        self.spec = deepcopy(specification)
        self.fabric_mhz = float(fabric_mhz)
        self.fraction_bits = int(fraction_bits)
        if self.fabric_mhz != 300. or self.fraction_bits != 18:
            raise ReferenceUnavailable('This reference is specified for v5 300 MHz / Q18 RAMP')
        self.outputs = self.spec['outputs']
        self.stages = self.spec['stages']
        self.axes = self.spec.get('axes', [])
        self.matrix = np.asarray(self.spec.get('matrix', np.eye(2)), dtype=float)
        self.scales = np.array([o['full_scale_mv'] for o in self.outputs], dtype=float)
        if (len(self.outputs) != 2 or self.matrix.shape != (2, 2)
                or not np.all(np.isfinite(self.matrix))
                or not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0)):
            raise ValueError('Two finite output scales and a finite 2x2 matrix are required')
        if not self.stages or float(self.stages[0].get('ramp_us', 0)) != 0:
            raise ValueError('The first stage must start with SET, without an incoming RAMP')
        for stage in self.stages:
            values = np.asarray(stage['voltage_mv'], dtype=float)
            if values.shape != (2,) or not np.all(np.isfinite(values)):
                raise ValueError('Every stage must specify two finite virtual voltages')
            _cycles(stage['hold_us'], self.fabric_mhz)
            if float(stage.get('ramp_us', 0)) < 0:
                raise ValueError('RAMP duration must not be negative')
        self.groups, self.member_groups = [], []
        keys, seen = {}, set()
        for member, axis in enumerate(self.axes):
            kind, stage = axis['kind'], axis['segment']
            if (kind not in ('amplitude', 'hold_duration', 'ramp_duration')
                    or type(stage) is not int or not 0 <= stage < len(self.stages)):
                raise ValueError('Unsupported AWG sweep member')
            output = axis.get('output', 0)
            if kind == 'amplitude' and output not in (0, 1):
                raise ValueError('Amplitude sweep output must be 0 or 1')
            key = (kind, stage, output if kind == 'amplitude' else None)
            if key in seen:
                raise ValueError('Duplicate sweep parameter')
            seen.add(key)
            count = axis['count']
            if type(count) is not int or count < 1:
                raise ValueError('Sweep count must be a positive integer')
            if not all(math.isfinite(float(axis[name])) for name in ('start', 'stop')):
                raise ValueError('Sweep endpoints must be finite')
            if kind != 'amplitude':
                start = _cycles(axis['start'], self.fabric_mhz)
                stop = _cycles(axis['stop'], self.fabric_mhz)
                step = _nearest_ratio(stop - start, count - 1) if count > 1 else 0
                if (count > 1 and step == 0) or start + (count - 1) * step < 1:
                    raise ValueError('Duration grid must contain distinct positive fabric cycles')
                if kind == 'ramp_duration' and (stage == 0 or not self.stages[stage].get('ramp_us', 0)):
                    raise ValueError('RAMP sweep requires an existing incoming RAMP')
            group = ('group', str(axis['group'])) if axis.get('group') else ('member', member)
            if group not in keys:
                keys[group] = len(self.groups)
                self.groups.append(dict(id=group[1], count=count, members=[]))
            index = keys[group]
            if self.groups[index]['count'] != count:
                raise ValueError('Grouped members must share their point count')
            self.groups[index]['members'].append(member)
            self.member_groups.append(index)
        self.shape = tuple(group['count'] for group in self.groups) or (1,)
        self.point_count = math.prod(self.shape)

    def coordinates(self, point):
        if type(point) is not int or not 0 <= point < self.point_count:
            raise ValueError('Point index is outside the declared sweep')
        return tuple(map(int, np.unravel_index(point, self.shape))) if self.groups else ()

    def _stages_at(self, indices, *, quantized):
        stages = deepcopy(self.stages)
        for member, axis in enumerate(self.axes):
            i, count = indices[self.member_groups[member]], axis['count']
            if quantized and axis['kind'] != 'amplitude':
                start = _cycles(axis['start'], self.fabric_mhz)
                stop = _cycles(axis['stop'], self.fabric_mhz)
                step = _nearest_ratio(stop - start, count - 1) if count > 1 else 0
                value = (start + i * step) / self.fabric_mhz
            else:
                value = float(axis['start']) if count == 1 else (
                    float(axis['start']) + (float(axis['stop']) - float(axis['start'])) * i / (count - 1))
            stage = stages[axis['segment']]
            if axis['kind'] == 'amplitude':
                stage['voltage_mv'][axis['output']] = value
            else:
                stage['ramp_us' if axis['kind'] == 'ramp_duration' else 'hold_us'] = value
        return stages

    def _rounded_codes(self, indices, stage):
        voltages = self._stages_at(indices, quantized=True)[stage]['voltage_mv']
        physical = self.matrix @ voltages
        if np.any(np.abs(physical) > self.scales):
            raise ValueError('The requested physical voltage exceeds its DAC scale')
        return tuple(max(-8192, min(8191, round(float(v / scale * 8192))))
                     for v, scale in zip(physical, self.scales))

    def _affine_field(self, function, indices, conditioned=None):
        origin = [0] * len(self.groups)
        if conditioned is not None:
            origin[conditioned] = indices[conditioned]
        base = int(function(tuple(origin)))
        value = base
        for group, details in enumerate(self.groups):
            count = details['count']
            if group == conditioned or count == 1:
                continue
            endpoint = list(origin)
            endpoint[group] = count - 1
            increment = _nearest_ratio(int(function(tuple(endpoint))) - base, count - 1)
            value += increment * indices[group]
        return value

    def commands(self, point):
        """Expected integer targets/steps derived from declared endpoint grids."""
        indices = self.coordinates(point)
        stages = self._stages_at(indices, quantized=True)
        commands = []
        for stage, values in enumerate(stages):
            targets = tuple(self._affine_field(
                lambda coordinate, ch=channel: self._rounded_codes(coordinate, stage)[ch], indices)
                for channel in range(2))
            if any(not -8192 <= code <= 8191 for code in targets):
                raise ValueError('Constant-increment target grid exceeds the DAC range')
            count = _cycles(values['ramp_us'], self.fabric_mhz) * 16 if values.get('ramp_us', 0) else 0
            steps = None
            if count:
                conditioned = next((self.member_groups[member] for member, axis in enumerate(self.axes)
                                    if axis['kind'] == 'ramp_duration' and axis['segment'] == stage), None)

                def direct_step(coordinate, channel):
                    duration = _cycles(self._stages_at(coordinate, quantized=True)[stage]['ramp_us'], self.fabric_mhz) * 16
                    difference = (self._rounded_codes(coordinate, stage)[channel]
                                  - self._rounded_codes(coordinate, stage - 1)[channel])
                    numerator = difference * (1 << (self.fraction_bits + 2))
                    return (-1 if numerator < 0 else 1) * (abs(numerator) // (duration - 1))

                steps = tuple(self._affine_field(lambda coord, ch=channel: direct_step(coord, ch),
                                                indices, conditioned) for channel in range(2))
            commands.append(dict(stage=stage, name=values['name'], target_codes=targets,
                                 ramp_samples=count, steps=steps,
                                 hold_cycles=_cycles(values['hold_us'], self.fabric_mhz)))
        return commands

    def waveforms(self, point, *, max_samples=4_800_000):
        """Full scalar arrays for a compensation-off baseband main waveform.

        Requested virtual coordinates are evaluated at the same time samples but
        retain their requested segment boundaries. Quantized values are physical
        pre-IIR codes scaled to volts. No path delay/gain/RC fit is assumed.
        """
        compensation = self.spec.get('compensation', {})
        if compensation.get('dc_enabled') or compensation.get('rc_enabled'):
            raise ReferenceUnavailable('DC/RC cases require the independent compensation/circuit reference')
        if self.spec.get('rf') or self.spec.get('rf_pulses'):
            raise ReferenceUnavailable('RF requires a carrier/envelope and output-composition reference')
        commands = self.commands(point)
        count = sum(c['hold_cycles'] * 16 + c['ramp_samples'] + (16 if c['ramp_samples'] else 0)
                    for c in commands)
        if count > max_samples:
            raise ValueError('Reference exceeds the explicit sample allocation limit')
        values = np.empty((count, 2), dtype=np.int16)
        cursor, current = 0, (0, 0)
        for command in commands:
            length = command['ramp_samples']
            target = command['target_codes']
            if length:
                indices = np.arange(length, dtype=np.int64)
                for channel in range(2):
                    ramp = ((int(current[channel]) << (self.fraction_bits + 2))
                            + indices * command['steps'][channel]) >> (self.fraction_bits + 2)
                    ramp = np.clip(ramp, -8192, 8191)
                    ramp[-1] = target[channel]
                    values[cursor:cursor + length, channel] = ramp
                cursor += length
                values[cursor:cursor + 16] = target  # Existing one-clock post-RAMP guard.
                cursor += 16
            length = command['hold_cycles'] * 16
            values[cursor:cursor + length] = target
            cursor += length
            current = target
        time = np.arange(count, dtype=float) / (self.fabric_mhz * 16 * 1e6)
        requested = self._stages_at(self.coordinates(point), quantized=False)
        virtual = np.zeros((count, 2), dtype=float)
        t0, previous = 0., np.asarray(requested[0]['voltage_mv'], dtype=float) / 1000.
        for stage in requested:
            target = np.asarray(stage['voltage_mv'], dtype=float) / 1000.
            duration = float(stage.get('ramp_us', 0)) * 1e-6
            if duration:
                mask = (time >= t0) & (time < t0 + duration)
                virtual[mask] = previous + (target - previous) * ((time[mask] - t0) / duration)[:, None]
                t0 += duration
            end = t0 + float(stage['hold_us']) * 1e-6
            virtual[(time >= t0) & (time < end)] = target
            t0, previous = end, target
        virtual[time >= t0] = previous
        return dict(model='v5-baseband-case-reference-v1', point=point,
                    group_indices=self.coordinates(point), time_s=time,
                    requested_virtual_v=virtual, requested_physical_v=virtual @ self.matrix.T,
                    quantized_codes=values, quantized_physical_v=values * self.scales / 8192000.,
                    commands=commands,
                    time_origin='First main SET at the DAC; common output latency removed.',
                    physical_reference_required=True)
