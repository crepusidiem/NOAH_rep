"""Frozen conservative C2 mappings; unresolved source semantics remain explicit."""
from datetime import datetime
import hashlib
import math
import json
from pathlib import Path

import numpy as np

from noah_baseline.m1.data import hours
from noah_baseline.m2.adapters import adapt_row
from noah_baseline.m2.c2_io import hash_json
from noah_baseline.m2.events import Event
from noah_baseline.m2.stream_index import digest_file,precision as source_precision


def clock(value,precision=None):
    observed_precision=source_precision('charttime',str(value)) if precision is None else precision
    if value in (None,'') or observed_precision not in {'second','fractional_second'}:raise ValueError('imprecise_or_missing_occurrence')
    if len(str(value))<=10:raise ValueError('imprecise_or_missing_occurrence')
    stamp=hours(value)
    if not math.isfinite(stamp):raise ValueError('nonfinite_occurrence')
    return stamp


def deferred(row,ref,reason,category=None):
    field=next((k for k in ['charttime','starttime','chartdate','acquisition_datetime','ecg_time'] if row.get(k)),None)
    return dict(subject_id=ref['subject_id'],split=ref['split'],encounter_id=row.get('hadm_id'),
        dataset=ref['dataset'],table=ref['table'],source_path=ref['source_path'],source_sha256=ref['source_sha256'],
        source_ordinal=ref['ordinal'],source_row_sha256=ref['row_sha256'],index_path=ref['index_path'],reason=reason,
        category=category,raw_occurrence=row.get(field),precision=row.get(str(field)+'__precision','unknown' if not field else 'source_string'))


def wrap(candidate,row,ref,queues):
    event=candidate['event'];numeric=candidate.get('raw_numeric');specific=event['specific_item']
    if numeric is not None:
        # Guard the existing float32 Fourier domain before it reaches a shard.
        if abs(numeric)>np.finfo(np.float32).max/16384:raise ValueError('numeric_basis_out_of_finite_domain')
        if specific:specific=specific+' ['+event['units']+']';event=dict(event,specific_item=specific)
    skey=queues.put('specifics',hashlib.sha256(specific.encode()).hexdigest(),dict(inline_text=specific)) if specific else None
    text=candidate.get('raw_text');content=None
    if event['modality']=='text':
        content_hash=hashlib.sha256(text.encode()).hexdigest() if text is not None else row.get('text_sha256')
        if not content_hash:raise ValueError('missing_verified_text_content_hash')
        locator=dict(source_path=ref['source_path'],source_sha256=ref['source_sha256'],source_ordinal=ref['ordinal'],
            index_path=ref['index_path'],field='text',subject_id=ref['subject_id'])
        if text is not None:locator['inline_text']=text
        content=queues.put('text',content_hash,locator)
    field=candidate['timestamp_field']
    occurrence_precision=row.get(field+'__precision',source_precision(field,str(row.get(field,''))))
    if occurrence_precision not in {'second','fractional_second'}:raise ValueError('unsupported_occurrence_precision')
    if row.get('storetime') and row.get('storetime__precision',source_precision('storetime',row['storetime'])) not in {'second','fractional_second'}:
        event=dict(event,available_at=None,availability_source='unknown');candidate=dict(candidate,observed_available_at=None)
    out=dict(**{k:v for k,v in event.items() if k in {'event_id','subject_id','encounter_id','linkage_status','dataset','dataset_version','occurred_at','available_at','availability_source','category','modality','specific_item','units'}},
        split=ref['split'],stay_id=row.get('stay_id'),table=ref['table'],source_path=ref['source_path'],source_sha256=ref['source_sha256'],
        source_row_id=event['source_row_id'],source_ordinal=ref['ordinal'],source_row_sha256=ref['row_sha256'],
        index_path=ref['index_path'],raw_occurrence=row.get(field),precision=occurrence_precision,
        observed_available_at=candidate.get('observed_available_at'),raw_storetime=row.get('storetime'),raw_numeric=numeric,
        numeric_vector=None,content_key=content,specifics_key=skey,model_ready=not skey and event['modality']=='categorical',
        semantics=candidate['timestamp_semantics'],transform_revision='m2-c2-candidate-v1',aggregate_group=None)
    return out


def file_hash_stable(path):
    p=Path(path);before=p.stat();digest=digest_file(p);after=p.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('media_content_changed_during_hash')
    return digest


def media(row,ref,queues,cxr_metadata,roots):
    kind=ref['dataset'];sid=ref['subject_id'];aux=None
    if kind=='cxr':
        matches=cxr_metadata.get(row['dicom_id'],[])
        if len(matches)!=1:raise ValueError('missing_or_ambiguous_cxr_metadata')
        meta,aux=matches[0]
        if int(meta['subject_id'])!=sid or str(meta['study_id'])!=str(row['study_id']):raise ValueError('cxr_cross_patient_or_study_link')
        date=meta.get('StudyDate','');tm=meta.get('StudyTime','')
        if not date or not tm:raise ValueError('missing_cxr_date_or_time_no_midnight_fallback')
        parts=tm.split('.');digits=parts[0]
        if not digits.isdigit() or len(digits)>6:raise ValueError('unsupported_cxr_time_precision')
        value=datetime.strptime(date+digits.zfill(6)+('.'+parts[1] if len(parts)==2 else ''),
            '%Y%m%d%H%M%S.%f' if len(parts)==2 else '%Y%m%d%H%M%S').isoformat(sep=' ')
        category=1;relative=str(Path(row['path']).with_suffix('.jpg'));raw_clock=date+' '+tm
    elif kind=='ecg':category=0;relative=row['path'];value=row['ecg_time'];raw_clock=value
    else:category=2;relative=row['dicom_filepath'];value=row['acquisition_datetime'];raw_clock=value
    occurred=clock(value);path=Path(roots[kind])/relative
    if not path.resolve().is_relative_to(Path(roots[kind]).resolve()):raise ValueError('media_path_escape')
    if kind=='ecg':
        hea=path.with_suffix('.hea');dat=path.with_suffix('.dat')
        hsha=file_hash_stable(hea);dsha=file_hash_stable(dat)
        content=hashlib.sha256((hsha+dsha).encode()).hexdigest()
        lines=[line for line in hea.read_text().splitlines() if line and not line.startswith('#')]
        count=int(lines[0].split()[1]);signal_lines=lines[1:1+count]
        units=[line.split()[2].rsplit('/',1)[1] if len(line.split())>2 and '/' in line.split()[2] else None for line in signal_lines]
        locator=dict(path=str(path),header_sha256=hsha,waveform_sha256=dsha,header_units=units)
    else:content=file_hash_stable(path);locator=dict(path=str(path),file_sha256=content)
    locator.update(subject_id=sid,source_path=ref['source_path'],source_sha256=ref['source_sha256'],source_ordinal=ref['ordinal'])
    if aux:locator['metadata_source']=aux
    key=queues.put(kind,content,locator)
    event_id=f"{kind}.{ref['table']}:{ref['source_sha256']}:{ref['ordinal']}:{category}"
    return dict(event_id=event_id,subject_id=sid,split=ref['split'],encounter_id=None,stay_id=None,linkage_status='unlinked',
        dataset=kind,table=ref['table'],dataset_version=ref['version'],source_path=ref['source_path'],source_sha256=ref['source_sha256'],
        source_row_id=str(ref['ordinal']),source_ordinal=ref['ordinal'],source_row_sha256=ref['row_sha256'],index_path=ref['index_path'],
        occurred_at=occurred,raw_occurrence=raw_clock,precision='fractional_second' if '.' in raw_clock else 'second',
        available_at=None,observed_available_at=None,availability_source='unknown',raw_storetime=None,category=category,modality=kind,
        specific_item=None,units='mV' if kind=='ecg' and locator['header_units'] and all(u=='mV' for u in locator['header_units']) else None,raw_numeric=None,numeric_vector=None,content_key=key,specifics_key=None,
        model_ready=False,auxiliary_provenance=json.dumps(aux,sort_keys=True) if aux else None,semantics='source_media_clock; no encounter inference',transform_revision='m2-c2-candidate-v1',aggregate_group=None)


def map_record(row,ref,context,recipe,queues,cxr_metadata=None,roots=None):
    row=dict(row)
    for name in ['valueuom','amountuom']:
        if row.get(name) is not None:row[name]=row[name].strip()
    dataset,table=ref['dataset'],ref['table']
    if table in {'patients','mimic-cxr-2.0.0-metadata'}:return [],[],False
    if table=='pyxis':return [],[deferred(row,ref,'dispensing_has_no_verified_administration_category')],False
    if table=='chartevents' and str(row['itemid']) not in recipe['numeric']['icu_vital_itemids']:
        return [],[deferred(row,ref,'unsupported_chart_item_mapping')],False
    try:
        if dataset in {'cxr','ecg','echo'}:return [media(row,ref,queues,cxr_metadata,roots)],[],False
        if dataset=='note':
            occurred=clock(row.get('charttime'),row.get('charttime__precision'));available=None
            if row.get('storetime'):
                try:available=clock(row['storetime'],row.get('storetime__precision'))
                except ValueError:pass
            event=Event(f"note.{table}:{ref['source_sha256']}:{ref['ordinal']}",ref['subject_id'],
                int(row['hadm_id']) if row.get('hadm_id') else None,'source' if row.get('hadm_id') else 'unlinked',dataset,
                ref['version'],str(ref['ordinal']),ref['source_path'],occurred,available,'storetime' if available is not None else 'unknown',
                'text',6 if table=='discharge' else 3,row.get('note_type') or None,None,'m2-c2-candidate-v1')
            if event.encounter_id is not None and context.admission_subjects.get(event.encounter_id)!=event.subject_id:
                raise ValueError('note_cross_patient_encounter')
            event.validate()
            candidate=dict(event=event.__dict__,raw_numeric=None,raw_text=None,observed_available_at=available,
                timestamp_field='charttime',timestamp_semantics='retrospective_documentation' if table=='discharge' else 'documented_report')
            return [wrap(candidate,row,ref,queues)],[],False
        result=adapt_row(dataset,table,row,source_path=ref['source_path'],
            source_row_id=ref['source_sha256']+':'+str(ref['ordinal']),version=ref['version'],context=context)
        events=[];failures=[deferred(row,ref,d['reason'],d['category']) for d in result['deferred']]
        for candidate in result['events']:
            try:
                if candidate['event']['category'] in {15,19,28,29} and candidate['raw_numeric'] is None:
                    raise ValueError('missing_numeric_measurement_not_categorical')
                if table in {'emar','medrecon','transfers','services'} and not candidate['event']['specific_item']:
                    raise ValueError('missing_required_specific_item')
                events.append(wrap(candidate,row,ref,queues))
            except ValueError as error:failures.append(deferred(row,ref,str(error),candidate['event']['category']))
        if not events and not failures:failures=[deferred(row,ref,'source_row_has_no_supported_action')]
        if table=='chartevents':
            for event in events:event['aggregate_group']=hash_json([event[k] for k in ['subject_id','encounter_id','stay_id','specific_item','units']]+[str(row['itemid'])])
        return events,failures,table=='chartevents'
    except (ValueError,KeyError,TypeError,OverflowError,OSError) as error:
        # Only the affected source row is deferred; resource failures propagate.
        if isinstance(error,OSError) and error.errno not in {2,20}:raise
        reason=str(error) if isinstance(error,ValueError) else type(error).__name__
        return [],[deferred(row,ref,'mapping_or_content:'+reason[:160])],False
