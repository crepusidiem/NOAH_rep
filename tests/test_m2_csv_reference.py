"""Late quoted commas must not depend on DuckDB's initial dialect sample."""
import csv
import json

import pytest

from noah_baseline.m2.csv_reference import source_counts,attest_existing_index
from noah_baseline.m2.stream_index import stream_source,digest_file


def test_late_quoted_comma_independent_reference_and_full_existing_chunk_replay(tmp_path):
    source=tmp_path/'metadata.csv'
    with source.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['subject_id','study_id','StudyDate','StudyTime','PerformedProcedureStepDescription'])
        for i in range(25002):w.writerow([1 if i%1000==0 or i==23055 else 99,str(i),'20200101','120000',
            'fictional, quoted procedure' if i==23055 else 'fictional'])
    splits={1:'train',2:'val'};out=tmp_path/'index'
    result=stream_source(source,out,splits,dataset='cxr',table='metadata',version='fixture',
        source_records_per_chunk=10000,selected_rows_per_chunk=500,selected_bytes_per_chunk=2**20)
    # NFS unlink tombstones are OS housekeeping, not committed artifacts.
    before={str(p):digest_file(p) for p in out.rglob('*') if p.is_file() and not p.name.startswith('.nfs')}
    assert source_counts(source,splits)==dict(split_rows={'train':27},split_patients={'train':1})
    receipt=tmp_path/'attestation.json';proof=attest_existing_index(source,out,splits,receipt)
    assert proof['selected_rows']==result['selected_rows']==27 and proof['reindexed'] is False
    assert before=={str(p):digest_file(p) for p in out.rglob('*') if p.is_file() and not p.name.startswith('.nfs')}
    with pytest.raises(FileExistsError):attest_existing_index(source,out,splits,receipt)
    with pytest.raises(ValueError,match='split changed'):attest_existing_index(source,out,{1:'test'},tmp_path/'bad.json')
    assert not (tmp_path/'bad.json').exists()
