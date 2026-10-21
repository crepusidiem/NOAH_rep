"""Conservative C2 source/event/content/ownership boundaries and independent oracles."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from noah_baseline.m2.adapters import Context
from noah_baseline.m2.c2 import validate_config
from noah_baseline.m2.c2_io import EVENT_SCHEMA,Shards,Queues,source_rows
from noah_baseline.m2.c2_mapping import map_record
from noah_baseline.m2.c2_finalize import connect,aggregate_vitals,patient_windows,finalize
from noah_baseline.m2.events import aggregate_minmax
from noah_baseline.m2.stream_index import digest_file

ROOT=Path(__file__).resolve().parents[1]


def config():return json.loads((ROOT/'configs/m2_c2.json').read_text())


def setup(tmp_path):
    q=Queues(tmp_path,{k:{'fixture':k} for k in ['text','specifics','cxr','ecg','echo']})
    ctx=Context({10:1,20:2},item_labels={('hosp','1'):'fictional lab',('icu','220045'):'Heart Rate'},chart_categories={'220045':28},ed_units={'heartrate':'bpm'})
    return q,ctx


def reference(dataset='hosp',table='labevents',ordinal=1,subject=1):
    return dict(dataset=dataset,table=table,source_path='fictional.csv',source_sha256='1'*64,
        ordinal=ordinal,split='train' if subject==1 else 'val',subject_id=subject,version='fixture',index_path='fictional.parquet',row_sha256='2'*64)


def test_frozen_config_source_policy_and_caps():
    c=config();validate_config(c)
    for key,value in [('memory_GiB',64),('elapsed_seconds',7201),('output_GiB',49)]:
        bad=copy.deepcopy(c);bad['limits'][key]=value
        with pytest.raises(ValueError,match='resource'):validate_config(bad)
    bad=copy.deepcopy(c);del bad['source_mappings']['ed.pyxis']
    with pytest.raises(ValueError,match='source policy'):validate_config(bad)


def test_longitudinal_labs_units_delayed_unknown_and_invalid_values(tmp_path):
    q,ctx=setup(tmp_path)
    try:
        row=dict(subject_id='1',hadm_id='',itemid='1',charttime='2020-01-01 01:00:00',storetime='2020-01-02 00:00:00',valuenum='0',valueuom='mg/dL')
        es,ds,_=map_record(row,reference(),ctx,config(),q)
        assert not ds and es[0]['encounter_id'] is None and es[0]['raw_numeric']==0
        assert es[0]['specific_item']=='fictional lab [mg/dL]'
        other,_,_=map_record(dict(row,valueuom='mmol/L'),reference(ordinal=8),ctx,config(),q)
        assert other[0]['specifics_key']!=es[0]['specifics_key']
        assert es[0]['available_at']>es[0]['occurred_at'] and es[0]['model_ready'] is False
        changed=dict(row,storetime='');es,_,_=map_record(changed,reference(ordinal=2),ctx,config(),q)
        assert es[0]['available_at'] is None and es[0]['availability_source']=='unknown'
        for changes in [dict(valueuom=''),dict(valuenum='POSITIVE'),dict(valuenum='',value=''),dict(valuenum='NaN'),dict(charttime='2020-01-01 01:00')]:
            es,ds,_=map_record(dict(row,**changes),reference(),ctx,config(),q);assert not es and ds
        es,ds,_=map_record(dict(row,hadm_id='20'),reference(),ctx,config(),q)
        assert not es and 'crosses patients' in ds[0]['reason']
    finally:q.db.close()


def test_retrospective_note_and_untimed_records_never_use_admission_proxy(tmp_path):
    q,ctx=setup(tmp_path)
    try:
        row=dict(subject_id='1',hadm_id='10',charttime='2020-01-03 14:00:00',storetime='2020-01-04 00:00:00',text_sha256='3'*64,note_type='DS')
        es,ds,_=map_record(row,reference('note','discharge'),ctx,config(),q)
        assert es[0]['modality']=='text' and not es[0]['model_ready'] and es[0]['semantics']=='retrospective_documentation'
        assert es[0]['raw_occurrence']==row['charttime'] and es[0]['content_key']
        for table in ['diagnoses_icd','procedures_icd','prescriptions','pharmacy']:
            es,ds,_=map_record(dict(subject_id='1',hadm_id='10',chartdate='2020-01-02'),reference(table=table),ctx,config(),q)
            assert not es and ds and ds[0]['source_ordinal']==1
    finally:q.db.close()


def test_content_dedup_keeps_distinct_event_bindings_and_media_match(tmp_path):
    q,ctx=setup(tmp_path);roots={'cxr':str(tmp_path)};(tmp_path/'fictional.jpg').write_bytes(b'fictional source bytes; no encoder run')
    try:
        row=dict(subject_id='1',study_id='100',dicom_id='abc',path='fictional.dcm')
        metadata={'abc':[(dict(subject_id='1',study_id='100',StudyDate='20200101',StudyTime='120000.000'),reference('cxr','metadata'))]}
        a,_,_=map_record(row,reference('cxr','cxr-record-list'),ctx,config(),q,metadata,roots)
        b,_,_=map_record(row,reference('cxr','cxr-record-list',ordinal=2),ctx,config(),q,metadata,roots)
        assert a[0]['event_id']!=b[0]['event_id'] and a[0]['content_key']==b[0]['content_key']
        assert q.db.execute('SELECT count(*) FROM tasks').fetchone()[0]==1
        assert a[0]['auxiliary_provenance'] and a[0]['encounter_id'] is None
        metadata['abc'][0][0]['subject_id']='2'
        es,ds,_=map_record(row,reference('cxr','cxr-record-list'),ctx,config(),q,metadata,roots)
        assert not es and 'cross_patient' in ds[0]['reason']
    finally:q.db.close()


def test_real_numeric_vectors_and_aggregation_match_independent_reference(tmp_path):
    q,ctx=setup(tmp_path);raw=[];writer=Shards(tmp_path,'inputs',EVENT_SCHEMA,rows=2);events=Shards(tmp_path,'events',EVENT_SCHEMA)
    try:
        for i,(t,v) in enumerate([(0,4),(1.5,2),(2,8)]):
            row=dict(subject_id='1',hadm_id='10',stay_id='100',itemid='220045',charttime=f'2020-01-01 {int(t):02d}:{30 if t%1 else 0:02d}:00',
                storetime='' if i==1 else '2020-01-01 04:00:00',valuenum=str(v),valueuom='bpm')
            es,ds,is_vital=map_record(row,reference('icu','chartevents',i+1),ctx,config(),q)
            assert not ds and is_vital;writer.add(es[0]);raw.append(dict(subject_id=1,encounter_id=10,item='Heart Rate',units='bpm',value=v,
                source_row_id=es[0]['event_id'],occurred_at=es[0]['occurred_at'],available_at=es[0]['available_at']))
        inputs=writer.close();c=connect(tmp_path,config())
        try:proof=aggregate_vitals(c,inputs,events,tmp_path,check=lambda:None,published=lambda n:None)
        finally:c.close()
        manifest=events.close();actual=[r for s in manifest['shards'] for r in pq.read_table(s['path']).to_pylist()]
        expected=aggregate_minmax(raw,origin=np.floor(raw[0]['occurred_at']),span=2,stride=1)
        assert proof['groups']==len(expected)==3 and len(actual)==6
        for j,group in enumerate(expected):
            for offset,name in enumerate(['minimum','maximum']):
                got=actual[j*2+offset]
                # SQL grouping order is deliberately not relied on.
                got=next(r for r in actual if r['occurred_at']==group['occurred_at'] and r['event_id'].endswith(name))
                assert got['raw_numeric']==group[name] and got['available_at']==group['available_at']
                frequencies=2.**np.arange(-7,15,dtype=np.float32);arg=np.float32(got['raw_numeric'])*frequencies
                np.testing.assert_array_equal(got['numeric_vector'],np.concatenate([np.sin(arg),np.cos(arg)]))
    finally:q.db.close()


def test_shifted_ownership_equal_times_feature_completeness_and_patient_eligibility(tmp_path):
    rows=[dict(subject_id=1,split='train',event_id=f'event:{i:04d}',occurred_at=float(i//2),model_ready=True) for i in range(140)]
    ws=patient_windows(rows)
    assert [x for w in ws for x in w['owned_target_ids']]==[r['event_id'] for r in rows]
    assert all(w['last_gap_unknown'] for w in ws)
    assert [len(w['owned_target_ids']) for w in ws]==[58,42,40]
    changed=copy.deepcopy(rows);changed[60]['model_ready']=False
    bad=patient_windows(changed);assert sum(w['owned_targets'] for w in bad if w['context_features_complete'])<140
    with pytest.raises(ValueError,match='order'):patient_windows(list(reversed(rows)))
    with pytest.raises(ValueError,match='Duplicate'):patient_windows(rows+[rows[0]])
    writer=Shards(tmp_path,'events',EVENT_SCHEMA)
    for r in changed:writer.add(dict(**r,category=5,modality='categorical',source_row_id=r['event_id'],encounter_id=None))
    manifest=writer.close();c=connect(tmp_path,config())
    try:result=finalize(c,manifest,tmp_path,{1:'train',2:'val'},config(),check=lambda:None,published=lambda n:None)
    finally:c.close()
    assert result['cohort']['val']['assigned']==1 and result['cohort']['val']['with_canonical_events']==0
    assert result['cohort']['train']['eligible_64_valid_owned_targets']==1


def test_source_hash_split_identity_and_atomic_budget_failure(tmp_path):
    writer=Shards(tmp_path,'rows',EVENT_SCHEMA,rows=1)
    writer.add(dict(event_id='a',subject_id=1,split='train'));manifest=writer.close();path=Path(manifest['shards'][0]['path'])
    # Source reader uses original CSV ordinals, not filtered-row numbering.
    import pyarrow as pa
    path2=tmp_path/'source.parquet';pq.write_table(pa.Table.from_pylist([dict(subject_id='1',split='train',source_csv_record_ordinal=17)]),path2)
    source=dict(dataset='fixture',table='fixture',source_path='fixture.csv',source_sha256='1'*64,version='fixture',units=[dict(path=str(path2),sha256=digest_file(path2))])
    assert list(source_rows(source,{1:'train'}))[0][1]['ordinal']==17
    with pytest.raises(ValueError,match='split'):list(source_rows(source,{1:'val'}))
    path2.write_bytes(path2.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='content changed'):list(source_rows(source,{1:'train'}))
    def fail():raise TimeoutError('fixture cap')
    blocked=Shards(tmp_path,'blocked',EVENT_SCHEMA,rows=1,check=fail)
    with pytest.raises(TimeoutError):blocked.add(dict(event_id='b'))
    assert not list(blocked.path.glob('[0-9]*'))


def test_c2_controller_completes_fixture_without_encoding_or_extra_execution(tmp_path,monkeypatch):
    import signal
    import pyarrow as pa
    from noah_baseline.m2 import c2 as controller
    root=tmp_path;cfg=config();(root/'configs').mkdir()
    (root/'configs/m2_c2.json').write_text(json.dumps(cfg))
    for name in ['m2.json','encoder_assets.json','m2_integration.json']:
        (root/'configs'/name).write_bytes((ROOT/'configs'/name).read_bytes())
    base=root/'artifacts/reduced_pilot/m2';pre=base/'m2-c2-preflight-20261021-06';pre.mkdir(parents=True)
    digest=digest_file(root/'configs/m2_c2.json')
    (pre/'PASS.json').write_text(json.dumps(dict(status='PASS',config_sha256=digest,source_files={'configs/m2_c2.json':digest})))
    path=root/'fixture.parquet'
    pq.write_table(pa.Table.from_pylist([
        dict(subject_id='1',hadm_id='10',admittime='2020-01-01 00:00:00',dischtime='2020-01-02 00:00:00',deathtime=None,split='train',source_csv_record_ordinal=2),
        dict(subject_id='1',hadm_id='11',admittime='2020-02-01 00:00:00',dischtime='2020-02-02 00:00:00',deathtime=None,split='train',source_csv_record_ordinal=9)]),path)
    source=dict(dataset='hosp',table='admissions',expected_rows=2,source_path='fictional.csv',source_sha256='1'*64,version='fixture',units=[dict(path=str(path),sha256=digest_file(path))])
    monkeypatch.setattr(controller,'load_sources',lambda *args:({}, {1:'train',2:'val',3:'test'},[source],[]))
    monkeypatch.setattr(controller,'build_context',lambda *args:(Context({10:1,11:1}),{},[]))
    monkeypatch.setattr(controller.subprocess,'check_output',lambda args,**kwargs:'' if '--porcelain' in args else 'FIXTURE_ONLY')
    previous={s:signal.getsignal(s) for s in [signal.SIGTERM,signal.SIGXFSZ]}
    try:
        result=controller.run(root,base/'fixture-run')
        assert result['status']=='COMPLETE_PREPARATION_NOT_MODEL_READY' and result['published_events']==4
        manifest=json.loads((base/'fixture-run/dataset-manifest.json').read_text())
        assert manifest['cohort']['train']['eligible_64_valid_owned_targets']==0
        assert manifest['cohort']['val']['assigned']==1 and manifest['encoding_queues']=={}
        with pytest.raises(FileExistsError):controller.run(root,base/'forbidden-second-execution')
        assert not (base/'forbidden-second-execution').exists()
    finally:
        for s,value in previous.items():signal.signal(s,value)
