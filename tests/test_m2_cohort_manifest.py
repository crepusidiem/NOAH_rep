"""Actual outputs, rather than assigned cohort size, determine target eligibility."""
import copy
import json

import numpy as np
import pytest

from noah_baseline.m2.cohort_manifest import CohortManifest
from noah_baseline.m2.dataset import entry,model_windows
from noah_baseline.m2.events import Event,EmbeddingCache


def records(count=70):
    return [entry(Event(f'fixture:{i}',1,10 if i%2 else None,'source' if i%2 else 'unlinked','fixture','1',str(i),
        'fictional.csv',float(i),None,'unknown','categorical',22,None,None,'fixture-v1')) for i in range(count)]


def manifest(tmp_path,name='accounting'):
    return CohortManifest(tmp_path,tmp_path/'artifacts/reduced_pilot'/name,{1:'train',2:'val',3:'test'},dataset_sha256='2'*64)


def windows(tmp_path,rows):
    return model_windows(tmp_path,rows,subject_id=1,birth_year=1970,length=64,overlap=16,
        cutoff=100.,profile='occurrence_diagnostic',expected_dataset_sha256='2'*64)


def test_counts_retain_unencoded_patients_and_only_owned_targets_qualify(tmp_path):
    m=manifest(tmp_path);rows=records()
    try:
        for row in rows:m.event(row,source_file_sha256='1'*64)
        for i,w in enumerate(windows(tmp_path,rows)):m.window(w,window_id=str(i))
        m.defer(source_key='untimed:1',subject_id=2,reason='date_only')
        m.encoding(source_key='video:1',kind='echo',subject_id=3,status='unavailable',reason='unsupported content')
        pointer=m.out/'actual-shard.jsonl';pointer.write_text(json.dumps(rows[0])+'\n');m.artifact(pointer,role='fixture shard')
        result=m.finish()
        assert result['split_patients']=={'train':dict(assigned=1,with_canonical_event=1,eligible_owned_targets=1),
            'val':dict(assigned=1,with_canonical_event=0,eligible_owned_targets=0),
            'test':dict(assigned=1,with_canonical_event=0,eligible_owned_targets=0)}
        assert result['valid_owned_targets']=={'train':70}
        assert result['encounter_attribution']=={'encounter_attributed':35,'longitudinal_unknown_encounter':35}
        assert result['encoding_outcomes']==[dict(kind='echo',status='unavailable',source_records=1)]
        assert result['full_m2_gate']=='HOLD' and result['status']=='ENCODED_EVENTS_AND_OWNED_WINDOWS'
    finally:m.close()


def test_bad_ownership_duplicate_and_cross_patient_do_not_erase_registered_events(tmp_path):
    m=manifest(tmp_path);rows=records();ws=windows(tmp_path,rows)
    try:
        for row in rows:m.event(row,source_file_sha256='1'*64)
        bad=copy.deepcopy(ws[0]);bad['subject_id']=2
        with pytest.raises(ValueError,match='another patient'):m.window(bad,window_id='bad')
        assert m.db.execute('SELECT count(*) FROM events').fetchone()[0]==70
        assert m.db.execute('SELECT count(*) FROM windows').fetchone()[0]==0
        m.window(ws[0],window_id='first')
        with pytest.raises(ValueError,match='ownership'):m.window(ws[0],window_id='duplicate')
        changed=copy.deepcopy(ws[1]);changed['owned_target_ids']=[]
        with pytest.raises(ValueError,match='Shifted target'):m.window(changed,window_id='shift')
        with pytest.raises(ValueError,match='missing independent'):m.finish()
        assert m.finish(require_windows=False)['missing_window_ownership']>0
    finally:m.close()


def test_no_false_readiness_or_fabricated_encodings(tmp_path):
    m=manifest(tmp_path)
    try:
        with pytest.raises(ValueError,match='Duplicate canonical'):m.event(replace_record(records(1)[0],subject_id=99),source_file_sha256='1'*64)
        with pytest.raises(ValueError,match='actual dataset'):m.encoding(source_key='x',kind='text',subject_id=1,status='encoded')
        with pytest.raises(ValueError,match='reason'):m.encoding(source_key='x',kind='text',subject_id=1,status='failed')
        with pytest.raises(ValueError,match='Unknown encoding'):m.encoding(source_key='x',kind='categorical',subject_id=1,status='encoded')
        with pytest.raises(ValueError,match='outside split'):m.defer(source_key='bad',subject_id=None,reason='unknown patient')
        assert m.finish()['status']=='WINDOWS_PENDING'
    finally:m.close()


def replace_record(row,**changes):
    row=copy.deepcopy(row);row['event'].update(changes);return row


def test_success_requires_real_dataset_bound_cache(tmp_path):
    m=manifest(tmp_path);cache=EmbeddingCache(tmp_path)
    try:
        ref=cache.put(np.ones(192),kind='ecg',content_sha256='1'*64,dataset_sha256='2'*64,
            checkpoint_sha256='3'*64,recipe={'transform_revision':'fixture'})
        m.encoding(source_key='ecg:1',kind='ecg',subject_id=1,status='cached',reference=ref)
        with pytest.raises(ValueError,match='Repeated source'):m.encoding(source_key='ecg:1',kind='ecg',subject_id=1,status='cached',reference=ref)
        wrong=dict(ref,dataset_sha256='4'*64)
        with pytest.raises(ValueError,match='dataset'):m.encoding(source_key='ecg:2',kind='ecg',subject_id=1,status='cached',reference=wrong)
    finally:m.close()
