"""Versioned source identities and cutoff views; no legacy loaders or caches."""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch

from noah_baseline.data.numeric import DyadicFourierEncoder
from noah_baseline.data.schema import DemographicPrefix, EventCategory, EventRecord, Modality, PatientJourney
from noah_baseline.data.window_builder import WindowBuilder
from noah_baseline.evaluation.contracts import ContractError, sha256_file, validate_split_manifest
from noah_baseline.evaluation.landmarks import EventEvidence, LandmarkRequest

VERSION = 'm1-events-v1'
PREFIX = ('birth_year','race','gender','insurance','language','marital_status')


def hours(value):
    if value is None or value == '':
        return None
    if isinstance(value, (int,float)):
        return float(value)
    return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc).timestamp()/3600


@dataclass(frozen=True)
class Event:
    event_id: str
    subject_id: int
    encounter_id: int
    source_table: str
    source_row_id: str
    source_version: str
    occurred_at: float
    available_at: float | None
    category: int
    value: float | None = None
    units: str | None = None
    variable: str | None = None
    transform_version: str = VERSION

    def __post_init__(self):
        if not self.event_id or not self.source_row_id or not math.isfinite(self.occurred_at):
            raise ContractError('Missing or nonfinite source identity/time')
        if self.available_at is not None and not math.isfinite(self.available_at):
            raise ContractError('Nonfinite availability')
        if self.value is not None and (not math.isfinite(self.value) or not self.units):
            raise ContractError('Numeric observation requires finite value and units')


def select_indexes(admissions, splits):
    """Select FIRST before eligibility; never replace an excluded index with a later stay."""
    selected = {}
    for admission in admissions:
        sid = int(admission['subject_id'])
        if sid not in splits:
            raise ContractError('Admission outside explicit patient split')
        if admission['admittime'] is None:
            raise ContractError('Unknown admission ordering')
        key = (admission['admittime'], int(admission['hadm_id']))
        if sid not in selected or key < (selected[sid]['admittime'], selected[sid]['hadm_id']):
            selected[sid] = dict(admission)
    result = []
    for sid, admission in sorted(selected.items()):
        cutoff = admission['admittime']+24
        end, death, label = admission['dischtime'], admission['deathtime'], admission['hospital_expire_flag']
        reason = None
        if label not in (0,1): reason = 'unknown_mortality'
        elif end is None: reason = 'unknown_discharge_time'
        elif end < admission['admittime']: reason = 'invalid_encounter_interval'
        elif end <= cutoff: reason = 'not_in_encounter_at_24h'
        elif death is not None and death <= cutoff: reason = 'not_alive_at_24h'
        elif label == 1 and death is None: reason = 'unknown_death_timing'
        elif label == 0 and (death is not None or admission.get('discharge_location') == 'DIED'):
            reason = 'contradictory_mortality_markers'
        elif death is not None and death > end: reason = 'death_after_encounter_end'
        result.append(dict(admission, split=splits[sid], cutoff=cutoff, eligible=reason is None,
                           exclusion=reason, label=label,
                           index_rule='first_admittime_then_hadm_id_before_eligibility'))
    return result


def extract_sources(hosp: Path, ed: Path, cohort: Path):
    validate_split_manifest(cohort)
    with cohort.open(newline='') as f:
        rows = list(csv.DictReader(f))
    splits = {int(r['subject_id']):r['split'] for r in rows}
    birth = {int(r['subject_id']):int(r['anchor_year'])-int(r['anchor_age']) for r in rows}
    paths = {'admissions':hosp/'admissions.csv','patients':hosp/'patients.csv',
             'edstays':ed/'edstays.csv','vitalsign':ed/'vitalsign.csv','cohort':cohort}
    sources = {name:dict(path=str(p.resolve()),sha256=sha256_file(p),bytes=p.stat().st_size,
                         version='2.2' if name!='cohort' else 'original_pilot_1000') for name,p in paths.items()}
    admissions, events, seen = [], [], set()
    with paths['admissions'].open(newline='') as f:
        for r in csv.DictReader(f):
            sid, hadm = int(r['subject_id']),int(r['hadm_id'])
            if sid not in splits: continue
            if hadm in seen: raise ContractError('Duplicate hospital ID')
            seen.add(hadm)
            a = dict(subject_id=sid,hadm_id=hadm,admittime=hours(r['admittime']),
                     dischtime=hours(r['dischtime']),deathtime=hours(r['deathtime']),
                     hospital_expire_flag=int(r['hospital_expire_flag']),
                     discharge_location=r['discharge_location'])
            admissions.append(a)
            for key,category in [('admittime',EventCategory.ENTER_HOSPITALIZATION),
                                 ('dischtime',EventCategory.LEAVE_HOSPITALIZATION),('deathtime',EventCategory.DEATH)]:
                if a[key] is not None:
                    row_id=f'{hadm}:{key}'
                    events.append(Event(f'hosp.admissions:{row_id}',sid,hadm,'hosp.admissions',row_id,'2.2',a[key],None,int(category)))
    # Check demographic origins directly against patients, not only a copied cohort CSV.
    patient_matches=set()
    with paths['patients'].open(newline='') as f:
        for r in csv.DictReader(f):
            sid=int(r['subject_id'])
            if sid in splits:
                if sid in patient_matches or birth[sid]!=int(r['anchor_year'])-int(r['anchor_age']):
                    raise ContractError('Cohort origin disagrees with patients table')
                patient_matches.add(sid)
    if patient_matches!=set(splits): raise ContractError('Missing cohort patient source')
    links={}
    with paths['edstays'].open(newline='') as f:
        for r in csv.DictReader(f):
            sid=int(r['subject_id'])
            if sid not in splits or not r['hadm_id']: continue
            hadm=int(r['hadm_id']);stay=int(r['stay_id'])
            if hadm not in seen or not any(a['subject_id']==sid and a['hadm_id']==hadm for a in admissions):
                raise ContractError('ED/hospital cross-encounter identity mismatch')
            if stay in links: raise ContractError('Duplicate ED stay identity')
            links[stay]=(sid,hadm)
            for key,category in [('intime',EventCategory.ENTER_ED),('outtime',EventCategory.LEAVE_ED)]:
                stamp=hours(r[key])
                if stamp is not None:
                    row_id=f'{stay}:{key}'
                    events.append(Event(f'ed.edstays:{row_id}',sid,hadm,'ed.edstays',row_id,'2.2',stamp,None,int(category)))
    excluded=Counter()
    with paths['vitalsign'].open(newline='') as f:
        for ordinal,r in enumerate(csv.DictReader(f),start=1):
            stay=int(r['stay_id'])
            if stay not in links: continue
            sid,hadm=links[stay]
            if int(r['subject_id'])!=sid: raise ContractError('Vitals/ED subject join mismatch')
            if not r['charttime'] or not r['heartrate']:
                excluded['missing_heart_rate_or_time']+=1;continue
            value=float(r['heartrate'])
            if not math.isfinite(value): raise ContractError('Nonfinite source heart rate')
            row_id=f'csv_record_{ordinal}:heartrate'
            events.append(Event(f'ed.vitalsign:{row_id}',sid,hadm,'ed.vitalsign',row_id,'2.2',
                                hours(r['charttime']),None,int(EventCategory.VITALS),value,'bpm','heartrate'))
    if len({e.event_id for e in events})!=len(events): raise ContractError('Duplicate event provenance')
    indexes=select_indexes(admissions,splits)
    if len(indexes)!=len(splits): raise ContractError('Cohort patient has no source admission')
    # Rehash sources after reading: no mixed-revision cohort extraction.
    for name,p in paths.items():
        if sha256_file(p)!=sources[name]['sha256']: raise ContractError('Source changed during extraction')
    return indexes, sorted(events,key=lambda e:(e.subject_id,e.occurred_at,e.event_id)), birth, sources, dict(excluded)


def cutoff_view(events, index, *, policy='occurrence_diagnostic'):
    if not index['eligible']: raise ContractError('Ineligible index cannot be scored')
    if policy not in {'occurrence_diagnostic','observed_available'}: raise ContractError('Unknown view policy')
    result=[e for e in events if e.subject_id==index['subject_id'] and e.encounter_id==index['hadm_id']
            and e.occurred_at<=index['cutoff']]
    if policy=='observed_available':
        result=[e for e in result if e.available_at is not None and e.available_at<=index['cutoff']]
    result.sort(key=lambda e:(e.occurred_at,e.event_id))
    if not result: raise ContractError('No eligible landmark context')
    return result


@dataclass
class Window:
    batch: dict
    evidence: list[EventEvidence]
    provenance: list[dict]
    owned_target_ids: list[str]
    subject_id: int
    encounter_id: int
    window_idx: int


def build_windows(events, index, birth_year, *, length=64, overlap=16, view_policy="occurrence_diagnostic"):
    """Keep a parallel full provenance ledger; candidate model/loss tensors stay unchanged."""
    view=cutoff_view(events,index,policy=view_policy)
    origin=hours(f'{birth_year}-01-01 00:00:00')
    fourier=DyadicFourierEncoder()
    clinical=[]
    for ordinal,e in enumerate(view):
        previous=e.occurred_at-view[ordinal-1].occurred_at if ordinal else 0.
        following=view[ordinal+1].occurred_at-e.occurred_at if ordinal+1<len(view) else 0.
        record=EventRecord(subject_id=e.subject_id,hadm_id=e.encounter_id,source_table=e.source_table,
            source_row_id=e.source_row_id,charttime=e.occurred_at,storetime=e.available_at,
            category_id=e.category,modality_id=int(Modality.NUMERIC if e.value is not None else Modality.CATEGORICAL),
            age=(e.occurred_at-origin)/8766.,ordinal=ordinal,delta_prev=previous,delta_next=following,
            has_specifics=False,has_value=e.value is not None,
            value_vec=fourier.encode(e.value) if e.value is not None else None)
        record.has_next_gap=ordinal+1<len(view)
        clinical.append(record)
    # No window-end dynamic snapshots or unverified text encodings. Static birth year is numeric;
    # the other five prefix values are explicitly unconditioned/Unknown in this reduced pilot.
    demo=DemographicPrefix(birth_year,*(['Unknown']*5))
    journey=PatientJourney(index['subject_id'],demo,clinical)
    builder=WindowBuilder(max_seq_len=length,overlap_len=overlap,min_events_per_patient=1)
    result=[]
    for w in builder.build_patient_windows(journey):
        start=w.window_idx*builder.clinical_stride
        chunk=view[start:start+builder.max_clinical_tokens]
        provenance=[dict(event_id=f"prefix:{index['subject_id']}:{index['hadm_id']}:{name}",
                         source_table='patients' if k==0 else 'pilot_unknown',
                         source_row_id=str(index['subject_id']) if k==0 else 'unconditioned',
                         source_version='2.2',transform_version=VERSION,
                         occurred_at=chunk[0].occurred_at,coordinate_kind='context_alignment_not_source_timestamp',
                         units='year' if k==0 else None,
                         value=birth_year if k==0 else 'Unknown',available_at=None)
                    for k,name in enumerate(PREFIX)] + [asdict(e) for e in chunk]
        ev=[EventEvidence(p['event_id'],index['subject_id'],index['hadm_id'],chunk[0].occurred_at,None,'unknown','prefix')
            for p in provenance[:6]]
        ev.extend(EventEvidence(e.event_id,e.subject_id,e.encounter_id,e.occurred_at,e.available_at,
                                'source' if e.available_at is not None else 'unknown','clinical') for e in chunk)
        w.modality_ids[0]=int(Modality.NUMERIC);w.has_value[0]=True
        w.value_vec[0,:44]=fourier.encode(birth_year)
        # Chunk terminal gap cannot disclose unseen suffix timing even in diagnostic views.
        last=w.total_tokens-1;w.delta_next[last]=0.;w.has_next_gap[last]=False
        fields=('category_ids','modality_ids','positions','ages','delta_prev','delta_next','has_next_gap',
                'specifics_vec','has_specifics','value_vec','has_value','attention_mask','loss_mask')
        batch={key:torch.from_numpy(getattr(w,key)).unsqueeze(0) for key in fields}
        owned=[ev[t].event_id for t in range(1,len(ev)) if bool(batch['loss_mask'][0,t])]
        result.append(Window(batch,ev,provenance,owned,index['subject_id'],index['hadm_id'],w.window_idx))
    expected=Counter(e.event_id for e in view)
    actual=Counter(event for w in result for event in w.owned_target_ids)
    if actual!=expected: raise ContractError('Shifted targets are missing or multiply owned')
    for w in result:
        owned_by_shift=w.batch['loss_mask'][:,1:] & w.batch['attention_mask'][:,1:] & w.batch['attention_mask'][:,:-1]
        if int(owned_by_shift.sum())!=len(w.owned_target_ids): raise ContractError('Independent target shift mismatch')
    return result


def landmark_request(index, window):
    return LandmarkRequest(index['subject_id'],index['hadm_id'],window.evidence[-1].event_id,index['cutoff'],
        'paper_reproduction','explicit_unknown_constant',0.,window.evidence[0].occurred_at,
        {'scope':'M1 reduced modality occurrence-time diagnostic; not strict prospective',
         'gap':'constant zero with unknown final-gap mask; no realized future next-gap',
         'demographics':'static numeric birth year; five unconditioned Unknown slots',
         'context':'last bounded window ending at last observed event at/before admission+24h'})
