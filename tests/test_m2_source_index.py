"""Prepared C1 source-index writer preserves identities and avoids raw-note copies."""
import csv
import hashlib
import json

import duckdb
import pytest

from noah_baseline.m2.source_index import write_source_index,source_index


def test_columnar_source_index_retains_csv_record_ordinals_precision_zero_and_splits(tmp_path):
    source=tmp_path/'fixture.csv'
    with source.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['subject_id','hadm_id','charttime','valuenum','valueuom','text','unused_payload'])
        writer.writerow([99,999,'2020-01-01 00:00:00',1,'mg/dL','outside cohort','not copied'])
        writer.writerow([1,10,'2020-01-01',0,'mg/dL','fictional,\nmultiline note','not copied'])
        writer.writerow([2,20,'2020-01-02 03:00:00',2,'mg/dL','fictional validation note','not copied'])
    connection=duckdb.connect();connection.execute('SET threads=2')
    target=tmp_path/'index.parquet'
    result=write_source_index(connection,source,target,{1:'train',2:'val'})
    rows=connection.execute('SELECT * FROM read_parquet(?) ORDER BY source_csv_record_ordinal',[str(target)]).fetchdf()
    assert list(rows['source_csv_record_ordinal'])==[2,3]
    assert list(rows['split'])==['train','val'] and rows.iloc[0]['valuenum']=='0'
    assert rows.iloc[0]['charttime']=='2020-01-01'
    assert rows.iloc[0]['text_sha256']==hashlib.sha256(b'fictional,\nmultiline note').hexdigest()
    assert 'text' not in rows.columns and 'unused_payload' not in rows.columns
    assert result['text_body_policy']=='hash_only' and sum(r['source_rows'] for r in result['counts'])==2
    clock={r['split']:r for r in result['raw_clock_completeness']['charttime']}
    assert clock['train']['date_only_strings']==1 and clock['train']['parseable_timestamp_strings']==0
    assert clock['val']['parseable_timestamp_strings']==1
    assert result['canonical_events']=='NOT_COMPUTED' and result['qualified_64_target_patients']=='NOT_COMPUTED'
    with pytest.raises(FileExistsError):write_source_index(connection,source,target,{1:'train'})


def test_full_c1_cannot_execute_before_resource_approval(tmp_path):
    config=tmp_path/'configs';config.mkdir()
    (config/'m2_c1.json').write_text(json.dumps(dict(execution_authorized=False,approval_reference=None)))
    out=tmp_path/'artifacts/reduced_pilot/forbidden'
    with pytest.raises(RuntimeError,match='explicit resource approval'):
        source_index(tmp_path,out=out,source=tmp_path/'unused')
    assert not out.exists()


def test_index_clock_completeness_keeps_missing_invalid_and_availability_distinct(tmp_path):
    source=tmp_path/'clocks.csv'
    with source.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['subject_id','charttime','storetime','valueuom'])
        writer.writerows([[1,'2020-01-01 00:00:00','',''],[1,'2020-01-01','2020-01-02 00:00:00','bpm'],
            [1,'not-a-time','2020-01-01','bpm'],[1,'','','bpm']])
    result=write_source_index(duckdb.connect(),source,tmp_path/'clocks.parquet',{1:'train'})
    chart=result['raw_clock_completeness']['charttime'][0]
    assert (chart['denominator'],chart['missing'],chart['date_only_strings'],chart['parseable_timestamp_strings'],chart['other_nonempty_strings'])==(4,1,1,1,1)
    store=result['raw_clock_completeness']['storetime'][0]
    assert store['missing']==2 and store['date_only_strings']==1 and store['parseable_timestamp_strings']==1
    assert result['raw_field_completeness']['valueuom'][0]['missing']==1
    assert result['availability_validation'].startswith('PENDING')
