"""Independent CSV identities, crash windows and bounded Arrow chunk contracts."""
import csv
import gzip
import hashlib
import json

import pyarrow.parquet as pq
import pytest

from noah_baseline.m2.stream_index import stream_source,verify_chunk,schema_for,digest_file


def fixture(path,count=23):
    header=['subject_id','hadm_id','charttime','storetime','chartdate','value','valuenum','valueuom','text']
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'wt',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(header)
        for i in range(count):writer.writerow([1 if i%3==0 else 2 if i%3==1 else 99,'',
            '2020-01-01 01:00:00','' if i%2 else '2020-01-02 03:00:00','2020-01-01 00:00:00',
            '0' if i%2 else 'same repeated value','0','mg/dL','fictional,\nmultiline'])
    return header


def execute(source,out,**kwargs):
    return stream_source(source,out,{1:'train',2:'val'},dataset='fixture',table='labs',version='fictional',
        source_records_per_chunk=5,selected_rows_per_chunk=3,selected_bytes_per_chunk=4096,**kwargs)


def rows(out):
    return [r for p in sorted((out/'chunks').glob('*/rows.parquet')) for r in pq.read_table(p).to_pylist()]


def test_exact_global_csv_identity_split_values_precision_unknown_and_duplicates(tmp_path):
    source=tmp_path/'fixture.csv.gz';header=fixture(source);out=tmp_path/'index'
    result=execute(source,out);actual=rows(out)
    with gzip.open(source,'rt',newline='') as f:expected=[(i,r) for i,r in enumerate(csv.DictReader(f),1) if r['subject_id'] in {'1','2'}]
    assert result['source_records']==23 and result['selected_rows']==len(expected)==len(actual)
    assert result['split_patients']=={'train':1,'val':1} and result['readiness']=='HOLD'
    for got,(ordinal,want) in zip(actual,expected):
        assert got['_source_csv_record_ordinal']==ordinal and got['_split']=={1:'train',2:'val'}[int(want['subject_id'])]
        for n in header:
            if n!='text':assert got[n]==want[n]
        assert got['text_sha256']==hashlib.sha256(want['text'].encode()).hexdigest() and 'text' not in got
        assert got['chartdate__precision']=='date' and got['_encounter_attribution']=='unknown'
        assert got['storetime__precision']==('unknown' if not want['storetime'] else 'second')
        assert got['_availability_observation']==('unknown' if not want['storetime'] else 'raw_clock_present')
    assert len({r['_source_csv_record_ordinal'] for r in actual})==len(actual)
    assert len({r['_source_row_sha256'] for r in actual})<len(actual)  # Source duplicates are retained.


@pytest.mark.parametrize('stage',['before_write','before_publish','after_publish'])
def test_resume_handles_atomic_crash_windows_without_losing_or_duplicating_rows(tmp_path,stage):
    source=tmp_path/'fixture.csv';fixture(source);out=tmp_path/'index'
    def fail(point,start):
        if point==stage and start>1:raise RuntimeError('injected interruption')
    with pytest.raises(RuntimeError):execute(source,out,fault=fail)
    assert not (out/'COMPLETE.json').exists()
    original={str(p.relative_to(out)):digest_file(p) for p in out.glob('chunks/[0-9]*/*')}
    result=execute(source,out)
    assert result['selected_rows']==16
    for name,sha in original.items():assert digest_file(out/name)==sha
    assert len({r['_source_csv_record_ordinal'] for r in rows(out)})==16
    assert execute(source,out)==result


def test_resume_rejects_changed_source_splits_recipe_tampered_chunks_and_missing_ranges(tmp_path):
    source=tmp_path/'fixture.csv';header=fixture(source);out=tmp_path/'index';execute(source,out)
    with pytest.raises(ValueError,match='identity'):
        stream_source(source,out,{1:'test'},dataset='fixture',table='labs',version='fictional',source_records_per_chunk=5,
            selected_rows_per_chunk=3,selected_bytes_per_chunk=4096)
    p=sorted(out.glob('chunks/*/rows.parquet'))[0];p.chmod(0o644);p.write_bytes(p.read_bytes()+b'x')
    with pytest.raises(ValueError,match='tampered'):execute(source,out)
    other=tmp_path/'other';execute(source,other);chunk=sorted((other/'chunks').iterdir())[0]
    chunk.rename(other/'removed')
    with pytest.raises(ValueError,match='Missing'):execute(source,other)
    source.write_text(source.read_text()+'1,,,,,,,,\n')
    with pytest.raises(ValueError,match='identity'):execute(source,tmp_path/'index')


def test_empty_chunks_noncohort_records_and_byte_boundaries_are_accounted(tmp_path):
    source=tmp_path/'fixture.csv';fixture(source,30);out=tmp_path/'index'
    result=stream_source(source,out,{99:'test'},dataset='fixture',table='labs',version='fictional',
        source_records_per_chunk=2,selected_rows_per_chunk=1,selected_bytes_per_chunk=4096)
    receipts=[json.loads(p.read_text()) for p in sorted(out.glob('chunks/*/receipt.json'))]
    assert sum(r['source_records'] for r in receipts)==30
    assert result['selected_rows']==10 and any(r['selected_rows']==0 for r in receipts)
    assert all(r['selected_rows']<=1 for r in receipts)


def test_malformed_source_or_output_budget_interruption_cannot_publish_complete(tmp_path):
    source=tmp_path/'fixture.csv';fixture(source);out=tmp_path/'index'
    def cap():raise TimeoutError('budget reached')
    with pytest.raises(TimeoutError):execute(source,out,check=cap)
    assert not (out/'COMPLETE.json').exists()
    source.write_text('subject_id,value\n1,0\n1,0,extra\n')
    with pytest.raises(ValueError,match='Malformed'):execute(source,tmp_path/'malformed')
    assert not (tmp_path/'malformed/COMPLETE.json').exists()


def test_bounded_check_is_never_a_full_original_source_receipt(tmp_path):
    source=tmp_path/'fixture.csv';fixture(source);out=tmp_path/'index'
    result=execute(source,out,max_source_records=9)
    assert result['status']=='BOUNDED_CHECK_NOT_FULL_SOURCE' and result['source_records']==9
    assert not (out/'COMPLETE.json').exists()
    assert execute(source,out,max_source_records=9)==result


def test_recovery_is_disabled_before_any_output_or_source_read(tmp_path):
    from noah_baseline.m2.c1_recovery import recover
    (tmp_path/'configs').mkdir();(tmp_path/'configs/m2_c1_recovery.json').write_text(json.dumps({'execution_authorized':False,'approval_reference':None}))
    with pytest.raises(RuntimeError,match='new explicit resource approval'):recover(tmp_path,tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()


def test_independent_whole_source_reference_mismatch_blocks_acceptance():
    from noah_baseline.m2.c1_recovery import validate_reference
    refs=[dict(dataset='fixture',table='lab',split='train',source_rows=5,contributing_patients=2)]
    validate_reference(dict(split_rows={'train':5},split_patients={'train':2}),refs,'fixture','lab')
    with pytest.raises(ValueError,match='reference mismatch'):
        validate_reference(dict(split_rows={'train':5},split_patients={'train':1}),refs,'fixture','lab')
    with pytest.raises(ValueError,match='missing'):
        validate_reference(dict(split_rows={},split_patients={}),refs,'fixture','other')


def test_recovery_controller_finishes_small_sources_before_icu_failure(tmp_path,monkeypatch):
    """Execute the controller on fictional sources; no real whole-source scan."""
    from pathlib import Path
    import signal
    from noah_baseline.m2 import c1_recovery as controller
    repo=Path(__file__).resolve().parents[1];base=tmp_path/'artifacts/reduced_pilot/m2'
    cohort=base/'m2-a-20261020-01';cohort.mkdir(parents=True)
    (cohort/'patient_splits.csv').write_text('subject_id,split\n1,train\n2,val\n')
    prior=base/'m2-c1-source-index-20261020-01';prior.mkdir()
    (prior/'execution.json').write_text('{"completed_sources": []}');(prior/'source_inventory.json').write_text('[]')
    inventory=base/'m2-c1-artifact-inventory-57006471.json';inventory.write_text('{}')
    cfg=json.loads((repo/'configs/m2_c1_recovery.json').read_text())
    cfg.update(execution_authorized=True,approval_reference='fictional-fixture-only',patient_splits_sha256=digest_file(cohort/'patient_splits.csv'),reuse_inventory_sha256=digest_file(inventory))
    (tmp_path/'configs').mkdir();(tmp_path/'configs/m2_c1_recovery.json').write_text(json.dumps(cfg))
    inputs=tmp_path/'fictional';inputs.mkdir();roots={'cxr':str(inputs),'icu':str(inputs)}
    (tmp_path/'configs/m2.json').write_text(json.dumps({'roots':roots}))
    source_inventory=[];statuses=[]
    for dataset,table,body in [('icu','chartevents','subject_id,charttime\nINVALID,2020-01-01\n'),
            ('cxr','cxr-record-list','subject_id,study_id,path\n1,100,fictional.jpg\n2,200,fictional2.jpg\n'),
            ('cxr','mimic-cxr-2.0.0-metadata','subject_id,study_id,StudyDate,StudyTime\n1,100,20200101,120000\n2,200,20200101,130000\n')]:
        p=inputs/(table+'.csv');p.write_text(body);stat=p.stat()
        source_inventory.append(dict(dataset=dataset,path=str(p),version='fictional',bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,header=body.splitlines()[0].split(',')))
        statuses.append(dict(dataset=dataset,table=table,status='NOT_REACHED'))
    (cohort/'source_inventory.json').write_text(json.dumps(source_inventory))
    report=tmp_path/'reports/server/2026-10-20_m2_c1';report.mkdir(parents=True);(report/'failure.json').write_text(json.dumps({'source_status':statuses}))
    census=tmp_path/'reports/server/2026-10-20_m2';census.mkdir()
    (census/'source_counts.csv').write_text('dataset,table,split,source_rows,contributing_patients\ncxr,cxr-record-list,train,1,1\ncxr,cxr-record-list,val,1,1\n')
    for name in ['packages/noah_baseline/m2/stream_index.py','packages/noah_baseline/m2/c1_recovery.py','tools/m2_recover.py','sbatch/m2_c1_recovery.sh']:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((repo/name).read_bytes())
    monkeypatch.setattr(controller.subprocess,'check_output',lambda args,**kw:'' if '--porcelain' in args else 'fictional-commit')
    old={s:signal.getsignal(s) for s in [signal.SIGTERM,signal.SIGXFSZ]}
    out=base/'recovery'
    try:
        with pytest.raises(RuntimeError,match='incomplete'):controller.recover(tmp_path,out)
    finally:
        for s,value in old.items():signal.signal(s,value)
    execution=json.loads(next(out.glob('attempt-*/execution.json')).read_text())
    assert [r['table'] for r in execution['completed']]==['cxr-record-list','mimic-cxr-2.0.0-metadata']
    assert execution['failures']==[{'dataset':'icu','table':'chartevents','error':'ValueError'}]
    assert execution['readiness']=='HOLD' and execution['retry'] is False
    assert (out/'cxr-cxr-record-list/COMPLETE.json').exists()
    assert not (out/'icu-chartevents/COMPLETE.json').exists()
