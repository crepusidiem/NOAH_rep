"""Canonical shard/cache associations and independent shifted target ledgers."""
from dataclasses import replace
import copy
from collections import Counter

import numpy as np
import pytest
import torch

from noah_baseline.m2.dataset import entry,validate_entry,write_shards,read_shards,model_windows
from noah_baseline.m2.events import Event,EmbeddingCache


def events():
    return [Event(f'fixture:{i}',1,10,'source','fixture','1',str(i),'fixture.csv',float(i),float(i),
        'source','categorical',22,None,None,'fixture-v1') for i in range(140)]


def test_event_to_cache_binding_cannot_be_swapped(tmp_path):
    cache=EmbeddingCache(tmp_path)
    reference=cache.put(np.ones(44),kind='numeric',content_sha256='1'*64,dataset_sha256='2'*64,
        checkpoint_sha256='3'*64,recipe={'transform_revision':'fixture','units':'bpm'})
    event=replace(events()[0],modality='numeric',category=28,units='bpm',content_sha256='1'*64)
    bound=entry(event,reference=reference)
    validate_entry(bound,cache,expected_dataset_sha256='2'*64)
    changed=copy.deepcopy(bound);changed['event']['content_sha256']='4'*64
    with pytest.raises(ValueError,match='disagree'):validate_entry(changed,cache,expected_dataset_sha256='2'*64)
    changed=copy.deepcopy(bound);changed['value_reference']['recipe']['transform_revision']='tampered'
    with pytest.raises(ValueError,match='identity'):validate_entry(changed,cache,expected_dataset_sha256='2'*64)
    with pytest.raises(ValueError):entry(replace(event,content_sha256=None),reference=reference)
    with pytest.raises(ValueError):validate_entry(bound,cache,expected_dataset_sha256='4'*64)


def test_shard_replay_duplicates_patient_split_and_hashes(tmp_path):
    rows=[entry(e) for e in events()]
    directory=tmp_path/'artifacts/reduced_pilot/shards';manifest=write_shards(tmp_path,directory,rows,{1:'train'},dataset_sha256='2'*64)
    assert manifest['counts']['train']['events']==140
    assert read_shards(tmp_path,directory,{1:'train'})==rows
    with pytest.raises(ValueError):read_shards(tmp_path,directory,{1:'test'})
    with pytest.raises(ValueError):write_shards(tmp_path,tmp_path/'artifacts/reduced_pilot/duplicate',rows+rows[:1],{1:'train'},dataset_sha256='2'*64)
    p=directory/'train.jsonl';p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='hash'):read_shards(tmp_path,directory,{1:'train'})


def test_shifted_targets_encounter_sidecars_and_future_suffix(tmp_path):
    es=events();es[70]=replace(es[70],encounter_id=11)
    rows=[entry(e) for e in es]
    args=dict(subject_id=1,birth_year=1970,length=64,overlap=16,cutoff=139.,profile='strict_prospective',expected_dataset_sha256='2'*64,demographic_available_at=0.)
    windows=model_windows(tmp_path,rows,**args)
    assert Counter(k for w in windows for k in w['owned_target_ids'])==Counter(e.event_id for e in es)
    assert any(p['encounter_id']==11 for w in windows for p in w['clinical_provenance'])
    assert len(windows)>2
    for w in windows:
        batch=w['batch'];mask=batch['loss_mask'][0];valid=batch['attention_mask'][0]
        assert not mask[:6].any() and not mask[~valid].any()
        count=int(valid.sum());assert not batch['has_next_gap'][0,count-1] and batch['delta_next'][0,count-1]==0
        assert int((mask[1:]&valid[1:]&valid[:-1]).sum())==len(w['owned_target_ids'])
    cutoff_args=dict(args,cutoff=39.)
    before=model_windows(tmp_path,rows,**cutoff_args)
    changed=[entry(replace(e,occurred_at=e.occurred_at+1000)) if e.occurred_at>39 else entry(e) for e in es]
    after=model_windows(tmp_path,changed,**cutoff_args)
    assert len(before)==len(after)
    for a,b in zip(before,after):
        for key in a['batch']:assert torch.equal(a['batch'][key],b['batch'][key])
    unknown=[entry(replace(e,available_at=None,availability_source='unknown')) for e in es]
    assert model_windows(tmp_path,unknown,**args)==[]


def test_strict_model_windows_reject_unknown_and_future_demographics(tmp_path):
    args=dict(subject_id=1,birth_year=1970,length=64,overlap=16,cutoff=139.,
        profile='strict_prospective',expected_dataset_sha256='2'*64)
    rows=[entry(e) for e in events()]
    with pytest.raises(ValueError,match='demographic availability'):
        model_windows(tmp_path,rows,**args)
    with pytest.raises(ValueError,match='demographic availability'):
        model_windows(tmp_path,rows,**args,demographic_available_at=140.)


def test_known_numeric_units_and_specific_items_cannot_silently_disappear(tmp_path):
    cache=EmbeddingCache(tmp_path)
    reference=cache.put(np.ones(44),kind='numeric',content_sha256='1'*64,dataset_sha256='2'*64,
        checkpoint_sha256='3'*64,recipe={'transform_revision':'fixture','units':'bpm'})
    event=replace(events()[0],modality='numeric',category=28,units=None,content_sha256='1'*64)
    with pytest.raises(ValueError,match='units'):
        validate_entry(entry(event,reference=reference),cache,expected_dataset_sha256='2'*64)
    wrong=replace(event,units='mmHg')
    with pytest.raises(ValueError,match='units'):
        validate_entry(entry(wrong,reference=reference),cache,expected_dataset_sha256='2'*64)
    ecg_reference=cache.put(np.ones(192),kind='ecg',content_sha256='1'*64,dataset_sha256='2'*64,
        checkpoint_sha256='3'*64,recipe={'transform_revision':'fixture','source_units':'mV'})
    ecg_event=replace(event,modality='ecg',category=0,units=None)
    with pytest.raises(ValueError,match='ECG source units'):
        validate_entry(entry(ecg_event,reference=ecg_reference),cache,expected_dataset_sha256='2'*64)
    event=replace(event,units='bpm',specific_item='heart rate')
    with pytest.raises(ValueError,match='specifics encoding'):
        validate_entry(entry(event,reference=reference),cache,expected_dataset_sha256='2'*64)


def test_shard_paths_cannot_escape_the_clean_profile(tmp_path):
    from noah_baseline.runtime import IsolationError
    with pytest.raises(IsolationError):
        write_shards(tmp_path,tmp_path/'legacy',[],{},dataset_sha256='2'*64)
    with pytest.raises(IsolationError):
        read_shards(tmp_path,tmp_path/'legacy',{})
