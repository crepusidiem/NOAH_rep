"""Versioned event-to-cache/shard/window associations with independent target IDs."""
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from noah_baseline.data.numeric import DyadicFourierEncoder
from noah_baseline.data.schema import DemographicPrefix, EventRecord, Modality, PatientJourney
from noah_baseline.data.window_builder import WindowBuilder
from noah_baseline.m1.data import hours
from noah_baseline.m2.events import Event, EmbeddingCache, cutoff_view
from noah_baseline.m2.inventory import dump, sha
from noah_baseline.runtime import ArtifactNamespace

VERSION='m2-events-shard-v1'
FIELDS=('category_ids','modality_ids','positions','ages','delta_prev','delta_next','has_next_gap',
        'specifics_vec','has_specifics','value_vec','has_value','attention_mask','loss_mask')


def entry(event, *, reference=None, specifics_reference=None):
    event.validate()
    if reference is not None:
        if event.content_sha256 is None or event.content_sha256!=reference['content_sha256']:
            raise ValueError('Independent source content hash must match the cache')
        if reference['kind'] != event.modality:
            raise ValueError('Wrong value modality attached to event')
        event=replace(event,content_sha256=reference['content_sha256'],
            encoder_checkpoint_sha256=reference['encoder_checkpoint_sha256'],
            cache_recipe_sha256=reference['recipe_sha256'],embedding_ref=reference['path']).validate()
    if specifics_reference is not None and (specifics_reference['kind']!='specifics' or not event.specific_item
            or hashlib.sha256(event.specific_item.encode()).hexdigest()!=specifics_reference['content_sha256']):
        raise ValueError('Specifics require a source-linked item and specifics vector')
    return dict(schema_version=VERSION,event=asdict(event),value_reference=reference,specifics_reference=specifics_reference)


def validate_entry(record, cache, *, expected_dataset_sha256):
    if record['schema_version']!=VERSION:
        raise ValueError('Unrecognized event shard schema')
    event=Event(**record['event']).validate();reference=record['value_reference']
    if event.modality!='categorical' and reference is None:
        raise ValueError('Missing value encoding; no fabricated or silently missing modality')
    if event.embedding_ref and reference is None:
        raise ValueError('Event/cache association lost')
    for ref in [reference,record['specifics_reference']]:
        if ref is not None:
            if ref['dataset_sha256']!=expected_dataset_sha256:
                raise ValueError('Cache belongs to a different source manifest')
            cache.get(ref)
    if event.modality=='numeric' and (not event.units or reference['recipe'].get('units')!=event.units):
        raise ValueError('Encoded numeric source units are missing or inconsistent with its recipe')
    if event.modality=='ecg' and (not event.units or reference['recipe'].get('source_units')!=event.units):
        raise ValueError('ECG source units disagree with the verified waveform recipe')
    specifics=record['specifics_reference']
    if event.specific_item and specifics is None:raise ValueError('Known source item lacks its specifics encoding')
    if specifics is not None and (specifics['kind']!='specifics' or not event.specific_item
            or hashlib.sha256(event.specific_item.encode()).hexdigest()!=specifics['content_sha256']):
        raise ValueError('Specific-item content binding mismatch')
    if reference is not None:
        expected=(event.modality,event.content_sha256,event.encoder_checkpoint_sha256,event.cache_recipe_sha256,event.embedding_ref)
        actual=tuple(reference[k] for k in ['kind','content_sha256','encoder_checkpoint_sha256','recipe_sha256','path'])
        if expected!=actual:
            raise ValueError('Source event and content cache identity disagree')
    return event


def write_shards(root, out, records, splits, *, dataset_sha256):
    ArtifactNamespace(root,'reduced_pilot').owned(out)
    out.mkdir(parents=True,exist_ok=False);cache=EmbeddingCache(root);seen=set();groups={};patients={}
    for record in records:
        event=validate_entry(record,cache,expected_dataset_sha256=dataset_sha256)
        if event.event_id in seen:
            raise ValueError('Duplicate canonical event ID')
        if event.subject_id not in splits or splits[event.subject_id] not in {'train','val','test'}:
            raise ValueError('Missing patient split membership')
        seen.add(event.event_id);split=splits[event.subject_id]
        groups.setdefault(split,[]).append(record);patients.setdefault(split,set()).add(event.subject_id)
    manifest=dict(schema_version=VERSION,dataset_sha256=dataset_sha256,shards={},counts={},
                  semantics='actual canonical bounded events; not full-cohort frequencies')
    for split,rows in sorted(groups.items()):
        rows.sort(key=lambda r:(r['event']['subject_id'],r['event']['occurred_at'],r['event']['event_id']))
        path=out/f'{split}.jsonl'
        with path.open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
        manifest['shards'][split]=dict(path=path.name,sha256=sha(path))
        manifest['counts'][split]=dict(events=len(rows),patients=len(patients[split]),
            modalities=dict(Counter(r['event']['modality'] for r in rows)),
            categories=dict(Counter(str(r['event']['category']) for r in rows)),
            category_patients={str(c):len({r['event']['subject_id'] for r in rows if r['event']['category']==c})
                for c in sorted({r['event']['category'] for r in rows})},
            modality_patients={m:len({r['event']['subject_id'] for r in rows if r['event']['modality']==m})
                for m in sorted({r['event']['modality'] for r in rows})},
            distinct_specific_items=len({r['event']['specific_item'] for r in rows if r['event']['specific_item']}))
    dump(out/'manifest.json',manifest)
    return manifest


def read_shards(root, directory, splits):
    ArtifactNamespace(root,'reduced_pilot').owned(directory)
    manifest=json.loads((directory/'manifest.json').read_text());cache=EmbeddingCache(root);records=[];seen=set()
    for split,ref in sorted(manifest['shards'].items()):
        path=directory/ref['path']
        if path.resolve().parent!=directory.resolve() or sha(path)!=ref['sha256']:
            raise ValueError('Shard path/hash mismatch')
        for line in path.read_text().splitlines():
            row=json.loads(line);event=validate_entry(row,cache,expected_dataset_sha256=manifest['dataset_sha256'])
            if splits.get(event.subject_id)!=split or event.event_id in seen:
                raise ValueError('Shard split or event ownership mismatch')
            seen.add(event.event_id);records.append(row)
    return records


def model_windows(root, records, *, subject_id, birth_year, length, overlap, cutoff, profile, expected_dataset_sha256, demographic_available_at=None):
    cache=EmbeddingCache(root);selected={}
    for row in records:
        ev=validate_entry(row,cache,expected_dataset_sha256=expected_dataset_sha256)
        if ev.subject_id==subject_id:
            if ev.event_id in selected:raise ValueError('Duplicate source events before windowing')
            selected[ev.event_id]=(ev,row)
    events=cutoff_view([ev for ev,_ in selected.values()],cutoff,profile)
    if not events:return []
    if profile=='strict_prospective' and (demographic_available_at is None or not np.isfinite(demographic_available_at) or demographic_available_at>cutoff):
        raise ValueError('Strict prospective demographic availability is unknown or after cutoff')
    clinical=[];origin=hours(f'{birth_year}-01-01 00:00:00')
    for ordinal,event in enumerate(events):
        row=selected[event.event_id][1];reference=row['value_reference'];specifics=row['specifics_reference']
        clinical.append(EventRecord(subject_id=subject_id,hadm_id=event.encounter_id,source_table=event.dataset,
            source_row_id=event.source_row_id,charttime=event.occurred_at,storetime=event.available_at,
            delta_prev=0. if ordinal==0 else event.occurred_at-events[ordinal-1].occurred_at,
            delta_next=0. if ordinal==len(events)-1 else events[ordinal+1].occurred_at-event.occurred_at,
            has_next_gap=ordinal<len(events)-1,age=(event.occurred_at-origin)/8766.,ordinal=ordinal,
            category_id=event.category,modality_id=int(Modality[event.modality.upper()]),
            has_specifics=specifics is not None,specifics_vec=None if specifics is None else cache.get(specifics),
            has_value=reference is not None,value_vec=None if reference is None else cache.get(reference)))
    demo=DemographicPrefix(birth_year,*['Unknown']*5)
    builder=WindowBuilder(max_seq_len=length,overlap_len=overlap,min_events_per_patient=1)
    journey=PatientJourney(subject_id,demo,clinical);windows=[];owners=Counter()
    for window in builder.build_patient_windows(journey):
        start=window.window_idx*builder.clinical_stride
        chunk=events[start:start+builder.max_clinical_tokens]
        # Bounded window terminal never imports a gap from outside its own context.
        last=window.total_tokens-1;window.delta_next[last]=0.;window.has_next_gap[last]=False
        window.modality_ids[0]=int(Modality.NUMERIC);window.has_value[0]=True
        window.value_vec[0,:44]=DyadicFourierEncoder().encode(birth_year)
        batch={key:torch.from_numpy(getattr(window,key)).unsqueeze(0) for key in FIELDS}
        target_ids=[None]*6+[e.event_id for e in chunk]+[None]*(length-window.total_tokens)
        sidecar=[selected[e.event_id][1]['event'] for e in chunk]
        independent=batch['loss_mask'][:,1:] & batch['attention_mask'][:,1:] & batch['attention_mask'][:,:-1]
        owned=[target_ids[i+1] for i in range(length-1) if independent[0,i]]
        if None in owned or batch['loss_mask'][0,:6].any() or batch['loss_mask'][0,window.total_tokens:].any():
            raise ValueError('Prefix or padding incorrectly owns shifted target loss')
        owners.update(owned)
        windows.append(dict(batch=batch,clinical_provenance=sidecar,token_event_ids=target_ids,
            owned_target_ids=owned,subject_id=subject_id,window_idx=window.window_idx,profile=profile,
            demographic_policy='M1 static birth year; five explicit unconditioned Unknown slots',
            gap_units='hours',age_units='years',terminal_gap='masked unknown zero',
            interpretation='teacher-forced time-conditioned training windows; cutoff-safe terminal'))
    if owners!=Counter(e.event_id for e in events):
        raise ValueError('Shifted target ownership is not exactly once')
    return windows
