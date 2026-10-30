"""Sample HOLD sweeps: integer register adds and timed v7 edge packets.

HOLD coordinates never become FPGA point tables. RAMP coefficients retain
their existing, independently indexed coefficient storage. No device I/O.
"""
from dataclasses import replace
from itertools import product
import numpy as np


def active(sequence):
    holds = [axis for axis in sequence.sweep_axes
             if axis.axis_kind == 'hold_duration' and getattr(axis, 'samples_per_clock', 1) == 16]
    if not holds:
        return False
    if any(axis.duration_sample_points[0] % 16 or axis.duration_step_samples % 16 for axis in holds):
        return True
    # A whole-clock HOLD following a fine RAMP starts at the actual endpoint,
    # including a 16-interval edge whose endpoint occupies the next word.
    # The coarse scheduler would instead include that word's trailing padding.
    return (any(spec and not spec['rise_start'] and 'fall_start' not in spec
                for specs in sequence.fine_ramps.values() for spec in specs)
            or any(axis.axis_kind == 'ramp_duration' and axis.samples_per_clock == 16
                   and any(value % 16 for value in axis.duration_sample_points)
                   for axis in sequence.sweep_axes))


def segment_samples(sequence, point):
    indices = sequence.point_indices(point)
    result = []
    for i, segment in enumerate(sequence.segments):
        length = round(sequence.segment_duration_cycles_at(point, i) * 16)
        if segment.kind == 'ramp':
            specs = sequence.fine_ramps.get(segment.name, ())
            lengths = []
            for spec in specs:
                if spec:
                    if spec['rise_start'] or 'fall_start' in spec:
                        raise ValueError('Sample SET hold requires ordinary single-transition adjacent RAMPs')
                    lengths.append(spec['rise_end'])
            if lengths:
                if len(set(lengths)) != 1 or len(lengths) != sequence.n_outputs:
                    raise ValueError('Sample SET hold requires aligned RAMP durations across outputs')
                length = lengths[0]
            for j, axis in enumerate(sequence.sweep_axes):
                if axis.axis_kind == 'ramp_duration' and axis.segment_name == segment.name:
                    length = (axis.duration_sample_points[indices[j]]
                              * (16 // axis.samples_per_clock))
        result.append(int(length))
    return tuple(result)


def boundaries(sequence, point):
    values = segment_samples(sequence, point)
    return np.cumsum((0,) + values, dtype=np.int64)


def pairs(sequence):
    result = {}
    used = set()
    names = [s.name for s in sequence.segments]
    for axis in sequence.sweep_axes:
        if axis.axis_kind != 'hold_duration':
            continue
        i = names.index(axis.segment_name)
        if (0 < i < len(names)-1 and sequence.segments[i-1].kind == 'ramp'
                and sequence.segments[i+1].kind == 'ramp' and not ({i-1, i+1} & used)):
            result[i-1] = i+1
            used.update((i-1, i+1))
    return result


def preview(sequence, point, space):
    get = sequence.amplitudes_at if space == 'physical' else sequence.virtual_amplitudes_at
    bound = boundaries(sequence, point) / 16.
    current = np.asarray(get(point, 0), dtype=float)
    t, v, segments = [0.], [current.copy()], []
    for i, segment in enumerate(sequence.segments):
        target = np.asarray([current[j] if x is None else x for j,x in enumerate(get(point,i))])
        if segment.kind == 'set':
            t.extend((bound[i],bound[i+1]));v.extend((target.copy(),target.copy()))
        else:
            t.append(bound[i+1]);v.append(target.copy())
        segments.append((segment.name,float(bound[i]),float(bound[i+1])))
        current = target
    array = np.asarray(v).T
    return np.asarray(t),dict(zip(sequence.output_names,array)),tuple(segments)


class SampleHoldProgramMixin:
    @property
    def _sample_hold(self):
        return active(self.sequence)

    def _hold_axis_step_tproc(self, axis):
        if self._sample_hold:
            return 0  # Sample timestamp fields below own every HOLD increment.
        return super()._hold_axis_step_tproc(axis)

    def _ramp_axis_step_tproc(self, axis):
        if self._sample_hold:
            return 0
        return super()._ramp_axis_step_tproc(axis)

    def _build_timing(self):
        if not self._sample_hold:
            return super()._build_timing()
        for ch in self.awg_channels:
            cfg = self.soccfg['gens'][ch]
            if cfg.get('type') != 'axis_awg_tuning_v7' or cfg.get('extended_ramp_version') != 1:
                raise ValueError('DAC-sample SET hold sweep requires AWG v7')
            if float(cfg['f_fabric']) != self.tproc_mhz:
                raise ValueError('DAC-sample SET hold requires matching AWG and tProcessor clocks')
        if any(a.kind == b.kind == 'set' for a,b in zip(self.sequence.segments, self.sequence.segments[1:])):
            raise ValueError('Sample HOLD requires a RAMP transition between SET segments')
        self._fh_pairs = pairs(self.sequence)
        self._fh_second = {b:a for a,b in self._fh_pairs.items()}
        bound = boundaries(self.sequence,0)
        times = {(i,o):int(bound[i]//16) for i in range(len(self.sequence.segments))
                 for o in range(self.sequence.n_outputs)}
        return dict(command_times=times,segment_starts=tuple(int((x+15)//16) for x in bound[:-1]),
                    segment_ends=tuple(int((x+15)//16) for x in bound[1:]),point_end=int((bound[-1]+15)//16))

    def _fh_coordinate_point(self, coordinates):
        shape = self.sequence.model_shape
        values = tuple(coordinates[i] if size > 1 else 0 for i, size in enumerate(shape))
        return int(np.ravel_multi_index(values, shape)) if shape else 0

    def _fh_anchor_axes(self, boundary):
        names = [segment.name for segment in self.sequence.segments]
        return {self.sequence.axis_representative(i)
                for i, axis in enumerate(self.sequence.sweep_axes)
                if axis.axis_kind in ('hold_duration', 'ramp_duration')
                and names.index(axis.segment_name) < boundary and axis.count > 1}

    def _fh_anchor_delta(self, boundary, coordinates):
        first = int(boundaries(self.sequence, 0)[boundary])
        current = int(boundaries(self.sequence, self._fh_coordinate_point(coordinates))[boundary])
        return (current + 15)//16 - (first + 15)//16

    def _fh_anchor_model(self, key, boundary, base, gen):
        first = int(boundaries(self.sequence, 0)[boundary])
        offset = int(base) - (first + 15)//16
        model = self._fh_model(key, lambda p: int(boundaries(self.sequence, p)[boundary])
                               + 15 + offset*16, gen)
        model['sample_timestamp'] = True
        return model

    def _fh_model(self, key, getter, gen, register='t', indices=None):
        """Factor exact affine axes; only RAMP/voltage coefficients may use tables."""
        axes = self.sequence.compiler_sweep_axes
        shape = self.sequence.model_shape
        base = int(getter(0))
        deltas = [0]*len(axes)
        varying = []
        slope = key[0] == 'sample_edge' and key[-1] in ('low', 'high')
        for a,axis in enumerate(axes):
            if axis.count < 2 or (indices is not None and a not in indices):
                continue
            values=[]
            for j in range(axis.count):
                c=[0]*len(shape);c[a]=j
                values.append(int(getter(int(np.ravel_multi_index(tuple(c),shape)))))
            # A zero initial voltage can hide duration dependence; likewise
            # high step bits may change only after two voltage axes combine.
            potential = slope and (axis.axis_kind == 'amplitude' or
                (axis.axis_kind == 'ramp_duration' and axis.segment_name == self.sequence.segments[key[1]].name))
            if len(set(values)) == 1 and not potential:
                continue
            varying.append(a)
            delta=values[1]-base
            if values == [base+j*delta for j in range(axis.count)]:
                deltas[a]=delta
            else:
                deltas[a]=None
        model=dict(key=key,register_name='sample_hold',base=base,
                   axis_deltas=tuple(deltas),page=int(self._gen_regmap[(gen,register)][0]),
                   command_register=int(self._gen_regmap[(gen,register)][1]),gen_ch=gen)
        # Rounded signed step coefficients can differ at interior coordinates.
        # Validate their complete dependency grid before declaring them affine.
        affine=all(d is not None for d in deltas)
        if slope and np.prod([axes[a].count for a in varying]) > 1_000_000:
            raise ValueError('Sample RAMP coefficient validation exceeds one million coordinates')
        if affine:
            choices = [range(axes[a].count) if slope else (0,axes[a].count-1) for a in varying]
            for chosen in product(*choices):
                c=[0]*len(shape)
                for a,j in zip(varying,chosen):c[a]=j
                if int(getter(int(np.ravel_multi_index(tuple(c),shape)))) != base+sum(j*d for j,d in zip(c,deltas)):
                    affine=False;break
        if not affine:
            # A separate HOLD loop stays an ADD even when a RAMP coefficient
            # or sample boundary is selected from the existing duration table.
            hold_only = {a for a in varying if all(
                member.axis_kind == 'hold_duration'
                for i, member in enumerate(self.sequence.sweep_axes)
                if self.sequence.axis_representative(i) == a)}
            if any(deltas[a] is None for a in hold_only):
                raise ValueError('HOLD sample state must be affine; per-point HOLD tables are forbidden')
            table_axes = [a for a in varying if a not in hold_only]
            table_shape = tuple(axes[a].count for a in table_axes)
            if not table_axes or np.prod(table_shape)*(1+len(hold_only)) > int(self.tproccfg['dmem_size']):
                raise ValueError('RAMP coefficient table exceeds DMEM; reduce independent RAMP/voltage points')
            values = []
            for chosen in np.ndindex(table_shape):
                c = [0]*len(shape)
                for a,j in zip(table_axes,chosen):
                    c[a] = j
                value = int(getter(int(np.ravel_multi_index(tuple(c),shape))))
                values.append(value)
                for a in hold_only:
                    c[a] = axes[a].count-1
                    actual = int(getter(int(np.ravel_multi_index(tuple(c),shape))))
                    if actual != value+c[a]*deltas[a]:
                        raise ValueError('HOLD state cannot depend on a per-point table')
                    c[a] = 0
            model.update(axis_deltas=(0,)*len(axes),duration_table_bases=tuple(values),
                         duration_table_axis_deltas={},
                         duration_axis_index=table_axes[0],duration_axis_indices=tuple(table_axes),
                         duration_table_shape=table_shape)
            if hold_only:
                # Keep HOLD state outside the row so an inner RAMP row load
                # cannot overwrite an outer HOLD coordinate or expand its table.
                addend_key = (*key, 'hold_addend')
                addend = dict(model, key=addend_key, base=0,
                              axis_deltas=tuple(deltas[a] if a in hold_only else 0 for a in range(len(axes))))
                for name in ('duration_table_bases', 'duration_table_axis_deltas', 'duration_axis_index',
                             'duration_axis_indices', 'duration_table_shape'):
                    addend.pop(name, None)
                self._fh_fields[addend_key] = addend
                model['sample_addend'] = addend_key
        return model

    def _fh_target(self, point, segment, output):
        from qick_fine_tune_sweep import _target_code
        index=segment+1 if self.sequence.segments[segment].kind=='ramp' else segment
        key=(index,output,0,'target')
        model=getattr(self,'_fh_original_models',{}).get(key)
        if model is not None:
            return int(super()._sweep_model_value(model,point))
        amplitude=self.sequence._effective_physical_set_amplitudes_at(point,index)[output]
        return _target_code(self.soccfg['gens'][self.awg_channels[output]],amplitude)

    def _fh_param(self, point, segment, output, name):
        from qick.ramp_v7 import edge_step
        duration=segment_samples(self.sequence,point)[segment]
        target=self._fh_target(point,segment,output)
        start=self._fh_target(point,segment-1,output)
        step=edge_step(start,target,duration)
        return dict(target=target & 65535,low=step & 0xffffffff,
                    high=(step>>32)&7,length=duration)[name]

    def _build_event_timing_models(self):
        models=list(super()._build_event_timing_models())
        if not self._sample_hold:return tuple(models)
        models=[m for m in models if not (len(m['key'])>1 and m['key'][1] in ('awg','bias_pre_zero'))]
        self._fh_fields={}
        for i,segment in enumerate(self.sequence.segments):
            for o,ch in enumerate(self.awg_channels):
                key=('event_time','awg',i,o,0)
                offset = self.timing['command_times'][(i,o)] - int(boundaries(self.sequence,0)[i]//16)
                model=self._fh_model(key,lambda p,i=i,k=offset:int(boundaries(self.sequence,p)[i])+k*16,ch)
                model['sample_timestamp']=True
                self._fh_fields[key]=model
                models.append(model)
                if segment.kind!='ramp':continue
                for name in ('target','low','high','length'):
                    k=('sample_edge',i,o,name)
                    self._fh_fields[k]=self._fh_model(k,lambda p,i=i,o=o,n=name:self._fh_param(p,i,o,n),ch,'target')
                if i in self._fh_pairs:
                    other=self._fh_pairs[i]
                    k=('sample_gap',i,o)
                    self._fh_fields[k]=self._fh_model(k,lambda p,i=i,j=other:int(boundaries(self.sequence,p)[j]-boundaries(self.sequence,p)[i]),ch)
        for index, rf in enumerate(self.rf_pulse_configs):
            if not rf.fine_duration:
                raise ValueError('Sample HOLD with RF requires automatic fine RF duration')
            key = ('fine_start_sample', index)
            model = self._fh_anchor_model(key, self._segment_index(rf.at_segment),
                                          self.aux_timing[f'rf_{index}_start'], rf.gen_ch)
            self._fh_fields[key] = model
        if self.readout_config is not None:
            raise ValueError('Sample HOLD acquisition requires the DDR readout path')
        if self.ddr_readout_config is not None:
            ddr = self.ddr_readout_config
            boundary = self._segment_index(ddr.at_segment)
            for kind, name in (('ddr_trigger', 'ddr_trigger_time'),
                               ('ddr_readout', 'ddr_readout_start'),
                               ('ddr_capture_end', 'ddr_capture_end')):
                if kind == 'ddr_readout' and self.aux_timing[name] == 0:
                    continue
                key = ('timing_only' if kind == 'ddr_capture_end' else 'event_time', kind)
                model = self._fh_anchor_model(key, boundary, self.aux_timing[name], self.awg_channels[0])
                if kind == 'ddr_trigger':
                    model.update(page=0, command_register=None, allocate_command_register=True,
                                 allocate_output_register=True)
                elif kind == 'ddr_readout':
                    page, register = self._ro_regmap[(ddr.ro_ch, 't')]
                    model.update(page=int(page), command_register=int(register))
                else:
                    model['timing_only'] = True
                if model.get('sample_addend'):
                    addend = self._fh_fields[model['sample_addend']]
                    addend.update(page=model['page'], command_register=model['command_register'])
                    if model.get('allocate_command_register'):
                        addend['allocate_command_register'] = True
                self._fh_fields[key] = model
                models.append(model)
        for o,ch in enumerate(self.awg_channels):
            key=('event_time','bias_pre_zero',o)
            base = self.timing['segment_ends'][-1]-self.awg_output_latency_tproc
            m=self._fh_anchor_model(key,len(self.sequence.segments),base,ch)
            m['sample_timestamp']=True;self._fh_fields[key]=m;models.append(m)
        end=max(int(boundaries(self.sequence,p)[-1]) for p in self._compile_validation_point_indices)
        if end + 16*self.timing['command_times'][(0,0)] + 15 >= 1 << 31:
            raise ValueError('Sample HOLD sequence timestamp must fit a positive signed 32-bit sample register')
        self._dynamic_point_end=max(getattr(self,'_dynamic_point_end',0),(end+15)//16+self.awg_output_latency_tproc,
                                    *(self._model_maximum(m,self.sequence.sweep_axes) for m in models))
        return tuple(models)

    def _build_capture_compensation_models(self, events):
        fields=list(super()._build_capture_compensation_models(events))
        if self._sample_hold:
            if self._overlap_dc_readout():
                replacements = {}
                for item in self._capture_compensation_models:
                    key = item['key']
                    if key[1] == 'wave_end':
                        source = self._fh_anchor_model(key, len(self.sequence.segments),
                            self.timing['segment_ends'][-1], self.awg_channels[0])
                    elif key[1] in ('readout_start', 'capture_end'):
                        original = ('event_time', 'ddr_readout') if key[1] == 'readout_start' else ('timing_only', 'ddr_capture_end')
                        source = self._fh_fields.get(original)
                        if source is None:
                            continue
                    else:
                        continue
                    model = dict(source, key=key, page=item['page'], command_register=item['command_register'])
                    if model.get('sample_addend'):
                        addend_key = (*key, 'hold_addend')
                        addend = dict(self._fh_fields[model['sample_addend']], key=addend_key,
                                      page=item['page'], command_register=item['command_register'])
                        addend.pop('allocate_command_register', None)
                        self._fh_fields[addend_key] = addend
                        model['sample_addend'] = addend_key
                    self._fh_fields[key] = model
                    replacements[key] = model
                self._capture_compensation_models = tuple(replacements.get(m['key'], m)
                                                          for m in self._capture_compensation_models)
                fields = [m for m in fields if m['key'] not in replacements]
            event_keys={m['key'] for m in events}
            fields.extend(dict(m) for k,m in self._fh_fields.items() if k not in event_keys
                          and (any(m['axis_deltas']) or 'duration_table_bases' in m))
        return fields

    def _sweep_model_value(self, model, point):
        value=super()._sweep_model_value(model,point)
        if model.get('sample_addend'):
            value += super()._sweep_model_value(self._fh_fields[model['sample_addend']], point)
        return value//16 if model.get('sample_timestamp') else value

    def _model_maximum(self, model, axes):
        if 'duration_table_bases' in model:
            value = max(int(base) + sum(max(0, int(delta[row])*(axes[a].count-1))
                        for a,delta in model['duration_table_axis_deltas'].items())
                        for row,base in enumerate(model['duration_table_bases']))
        else:
            value=super()._model_maximum(model,axes)
        if model.get('sample_addend'):
            value += super()._model_maximum(self._fh_fields[model['sample_addend']], axes)
        return value//16 if model.get('sample_timestamp') else value

    def _write_swept_or_static_register(self,key,page,register,value,comment):
        model=getattr(self,'_fh_fields',{}).get(key)
        if model is not None and model.get('sample_timestamp'):
            self._fh_load(key,page,register)
            self.bitwi(page,register,register,'>>',4,'sample time to command clock')
        else:super()._write_swept_or_static_register(key,page,register,value,comment)

    def _fh_load(self,key,page,register):
        m=self._fh_fields[key]
        super()._write_swept_or_static_register(key,page,register,m['base'],'sample HOLD '+str(key))
        if m.get('sample_addend'):
            addend = self._fh_fields[m['sample_addend']]
            ch = m['gen_ch']
            names = ('reserved_start', 'duration') if (ch, 'reserved_start') in self._gen_regmap else ('freq', 'phase')
            scratch = next(int(self._gen_regmap[(ch, name)][1]) for name in names
                           if int(self._gen_regmap[(ch, name)][1]) != register)
            # DMEM[0] is the compiler's transient transfer slot. Preserve all
            # partially packed command words while loading a spilled addend.
            self.memwi(page, scratch, 0, 'save sample timestamp scratch')
            super()._write_swept_or_static_register(addend['key'], page, scratch, 0, 'HOLD register addend')
            self.math(page, register, register, '+', scratch, 'RAMP row plus independent HOLD state')
            self.memri(page, scratch, 0, 'restore sample timestamp scratch')

    def _fh_combined_test(self, first, output, page, a,b):
        self._fh_load(('event_time','awg',first,output,0),page,a)
        self.bitwi(page,a,a,'&',15,'first edge lane')
        self._fh_load(('sample_gap',first,output),page,b)
        self.math(page,a,a,'+',b,'second edge sample position')
        self.safe_regwi(page,b,16383,'two-edge packet limit')

    def _fh_packet(self, segment, output, combined):
        ch=self.awg_channels[output];page=self._gen_regmap[(ch,'target')][0]
        w0,w1,w2,w3,w4,t=[self._gen_regmap[(ch,n)][1] for n in ('target','reserved_start','duration','step','control','t')]
        second=self._fh_pairs[segment] if combined else None
        load=lambda name,r,index=segment:self._fh_load(('sample_edge',index,output,name),page,r)
        self._fh_load(('event_time','awg',segment,output,0),page,w3)
        self.bitwi(page,w3,w3,'&',15,'sample RAMP start lane')
        if combined:
            self._fh_load(('sample_gap',segment,output),page,w4)
            self.math(page,w4,w4,'+',w3,'sample falling position')
            load('length',w0,second)
            self.math(page,w0,w0,'+',w4,'sample packet last index')
            self.mathi(page,w0,w0,'+',1,'include terminal sample')
            self.bitwi(page,t,w4,'&',15)
            self.bitwi(page,t,t,'<<',10)
            self.bitwi(page,w3,w3,'<<',6)
            self.bitw(page,w3,w3,'|',t)
            self.bitwi(page,w4,w4,'>>',4)
            self.bitwi(page,w4,w4,'<<',6)
        else:
            load('length',w0)
            self.math(page,w0,w0,'+',w3)
            self.mathi(page,w0,w0,'+',1,'include terminal sample')
            self.bitwi(page,w3,w3,'<<',6)
            self.bitwi(page,w3,w3,'|',31<<10,'single-edge packet')
            self.safe_regwi(page,w4,0)
        self.bitwi(page,t,w0,'>>',17)
        self.bitw(page,w4,w4,'|',t)
        control=(2<<16)|(1<<23)|(int(self.soccfg['gens'][ch].get('tmux_ch') or 0)<<24)
        self.bitwi(page,w4,w4,'|',control)
        self.bitwi(page,w0,w0,'<<',15)
        self.bitw(page,w3,w3,'|',w0)
        load('target',w0)
        if combined:
            load('target',t,second)
            self.bitwi(page,t,t,'<<',16)
            self.bitw(page,w0,w0,'|',t)
            load('high',t,second)
            # The second signed step crosses the word boundary at bit 29.
            self.bitwi(page,t,t,'<<',3)
            load('low',w2,second)
            self.bitwi(page,w2,w2,'>>',29)
            self.bitw(page,t,t,'|',w2)
            self.bitw(page,w3,w3,'|',t)
        else:
            self.bitwi(page,t,w0,'<<',16,'single-edge terminal target')
            self.bitw(page,w0,w0,'|',t)
        load('low',w1);load('high',w2)
        if combined:
            load('low',t,second)
            self.bitwi(page,t,t,'<<',3)
            self.bitw(page,w2,w2,'|',t)
        self._fh_load(('event_time','awg',segment,output,0),page,t)
        self.bitwi(page,t,t,'>>',4,'sample RAMP command timestamp')
        self.set(int(self.soccfg['gens'][ch]['tproc_ch']),page,w0,w1,w2,w3,w4,t,
                 'sample HOLD '+('combined' if combined else 'single')+' '+self.sequence.segments[segment].name)

    def _emit_command(self,command,tproc_time,*,event_field_key=None):
        if not self._sample_hold or command.segment_index<0:
            return super()._emit_command(command,tproc_time,event_field_key=event_field_key)
        i,o=command.segment_index,command.output_index
        if command.kind=='set':
            if i and self.sequence.segments[i-1].kind=='ramp':return
            return super()._emit_command(command,tproc_time,event_field_key=event_field_key)
        ch=command.gen_ch;page=self._gen_regmap[(ch,'target')][0]
        a,b=[self._gen_regmap[(ch,n)][1] for n in ('target','reserved_start')]
        label=f'SAMPLE_HOLD_{i}_{o}'
        if i in self._fh_pairs:
            self._fh_combined_test(i,o,page,a,b)
            self.condj(page,a,'>',b,label+'_SPLIT')
            self._fh_packet(i,o,True)
            self.condj(page,0,'==',0,label+'_DONE')
            self.label(label+'_SPLIT');self._fh_packet(i,o,False)
            self.label(label+'_DONE');self.mathi(page,a,a,'+',0)
        elif i in self._fh_second:
            self._fh_combined_test(self._fh_second[i],o,page,a,b)
            self.condj(page,a,'<=',b,label+'_DONE')
            self._fh_packet(i,o,False)
            self.label(label+'_DONE');self.mathi(page,a,a,'+',0)
        else:self._fh_packet(i,o,False)

    def sample_hold_commands(self,point):
        """Independent packed-command reconstruction for area/RC/export checks."""
        from qick.ramp_v7 import ramp_packet
        from qick_fine_tune_sweep import CompiledCommand, _pack_command_words, OP_SET
        bound=boundaries(self.sequence,point);lengths=segment_samples(self.sequence,point)
        for i,segment in enumerate(self.sequence.segments):
            for o,ch in enumerate(self.awg_channels):
                if segment.kind=='set':
                    if i:continue
                    c=self.compiled_points[0].segment_commands[i][o]
                    target=self._fh_target(point,i,o)
                    yield replace(c, point_index=point, target_code=target,
                                  words=_pack_command_words(self.soccfg['gens'][ch], target, 0, 0, OP_SET))
                    continue
                first=self._fh_second.get(i)
                if first is not None and bound[i]-(bound[first]//16)*16<=16383:continue
                other=self._fh_pairs.get(i)
                combine=other is not None and bound[other]-(bound[i]//16)*16<=16383
                lane=int(bound[i]%16)
                target=self._fh_target(point,i,o)
                kw=dict(start=self._fh_target(point,i-1,o),target=target,rise_start=lane,
                        rise_end=lane+lengths[i],tmux=int(self.soccfg['gens'][ch].get('tmux_ch') or 0))
                if combine:
                    f0=int(bound[other]-(bound[i]//16)*16)
                    target=self._fh_target(point,other,o)
                    kw.update(fall_start=f0,fall_end=f0+lengths[other],end_target=target)
                pack=ramp_packet(**kw)
                yield CompiledCommand(point,i,segment.name,o,self.sequence.output_names[o],ch,
                                      'ramp',target,pack['duration_samples'],pack['rise_step'],tuple(pack['words']))

    def _validate_dynamic_event_order(self):
        if not self._sample_hold:
            return super()._validate_dynamic_event_order()
        # Preserve the actual static instruction order; sorting each point by
        # its new timestamp would hide FIFO ordering errors.
        models = {m['key']: m for m in self._event_timing_models}
        events = []
        for group in self.compiled_points[0].segment_commands:
            for command in group:
                if command.kind == 'set' and command.segment_index:
                    continue
                key = ('event_time', 'awg', *self._command_key(command))
                events.append((self.timing['command_times'][self._timing_key(command)], 10,
                               command.gen_ch, key))
        for index, rf in enumerate(self.rf_pulse_configs):
            for kind, priority, suffix in (('rf_start', 10, 'start'), ('rf_stop', 15, 'end')):
                events.append((self.aux_timing[f'rf_{index}_{suffix}'], priority, rf.gen_ch,
                               ('event_time', kind, index)))
        if self.ddr_readout_config is not None:
            port = self.soccfg['readouts'][self.ddr_readout_config.ro_ch]['tproc_ctrl']
            gen = next((ch for ch, cfg in enumerate(self.soccfg['gens']) if cfg['tproc_ch'] == port), None)
            if gen is not None:
                events.append((self.aux_timing['ddr_readout_start'], 0, gen, ('event_time', 'ddr_readout')))
        if self.sequence.bias_t_compensation or self.sequence.rc_compensation:
            for o, gen in enumerate(self.awg_channels):
                events.append((self.timing['segment_ends'][-1]-self.awg_output_latency_tproc,
                               30, gen, ('event_time', 'bias_pre_zero', o)))
        events.sort(key=lambda item: item[:2])
        axes = self.sequence.compiler_sweep_axes
        varying = tuple(sorted({self.sequence.axis_representative(i)
                                for i, axis in enumerate(self.sequence.sweep_axes)
                                if axis.axis_kind in ('hold_duration', 'ramp_duration', 'rf_duration')
                                and axis.count > 1}))
        count = int(np.prod([axes[i].count for i in varying]))
        if count > 1_000_000:
            raise ValueError('Sample HOLD timing validation exceeds one million duration coordinates')
        shape = self.sequence.model_shape
        for chosen in product(*(range(axes[i].count) for i in varying)):
            self._check_cancel()
            coordinates = [0]*len(shape)
            for i, value in zip(varying, chosen):
                coordinates[i] = value
            point = int(np.ravel_multi_index(tuple(coordinates), shape)) if shape else 0
            expanded = self.sequence.point_indices(point)
            commands = {('event_time', 'awg', *self._command_key(c)): c for c in self.sample_hold_commands(point)}
            previous, occupied = {}, {}
            for base, priority, gen, key in events:
                if key[1] == 'awg' and key not in commands:
                    continue
                if key[1] in ('rf_start', 'rf_stop'):
                    detail = self._fine_details_at(key[2], expanded)
                    if key[1] == 'rf_stop' and not detail['active_tail']:
                        continue
                    timestamp = detail['start' if key[1] == 'rf_start' else 'tail']
                    if key[1] == 'rf_start':
                        if timestamp < occupied.get(gen, -1):
                            raise ValueError('Sample HOLD sweep overlaps RF pulses on one generator')
                        occupied[gen] = detail['busy_end']
                else:
                    model = models.get(key)
                    timestamp = int(base) if model is None else int(self._sweep_model_value(model, point))
                    if key[1] == 'awg':
                        c = commands[key]
                        if timestamp < occupied.get(gen, -1):
                            raise ValueError('Sample HOLD has more transitions in one DAC word than v7 can encode')
                        # The last word may accept the next command in v7.
                        occupied[gen] = timestamp + max(1, (c.duration_samples+15)//16)
                port = int(self.soccfg['gens'][gen]['tproc_ch'])
                if port in previous and timestamp-previous[port] < self._dispatch_spacing:
                    raise ValueError('Sample HOLD sweep changes shared-port command order or spacing')
                previous[port] = timestamp

    def summary(self):
        result=super().summary()
        if self._sample_hold:
            result['sample_hold_sweep']=dict(sample_period_ns=1000/(self.tproc_mhz*16),
                increment='integer register add',duration_point_table_words=0,
                short_path='two-edge packet',long_path='two timed single-edge packets',
                branch_limit_samples=16383)
        return result
