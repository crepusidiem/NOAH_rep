"""GPU budgets derive from materialized deduplicated queues, not source totals."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from noah_baseline.m2.gpu_plan import proposal
from noah_baseline.m2.stream_index import digest_file


def test_gpu_plan_counts_actual_tasks_and_stays_unapproved(tmp_path):
    queue=tmp_path/'echo.parquet';pq.write_table(pa.table({'id':['a','b']}),queue)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(config_sha256='1'*64,job_id='FIXTURE',source_commit='FIXTURE')))
    (tmp_path/'execution.json').write_text('{"status":"COMPLETE_PREPARATION_NOT_MODEL_READY"}')
    manifest=tmp_path/'dataset-manifest.json';manifest.write_text(json.dumps(dict(
        status='CANDIDATE_CANONICAL_PREPARATION_COMPLETE_EMBEDDINGS_PENDING',config_sha256='1'*64,
        patient_splits_sha256='2'*64,cohort={'train':{'assigned':10500}},
        encoding_queues={'echo':dict(path=str(queue),sha256=digest_file(queue),tasks=2)})))
    result=proposal(manifest)
    assert result['actual_deduplicated_tasks']['echo']==2 and result['actual_deduplicated_tasks']['ecg']==0
    assert result['float32_vector_payload_bytes']==2*512*4
    assert result['execution_authorized'] is False and result['submissions']==0 and result['readiness'].startswith('HOLD')
    queue.write_bytes(queue.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='changed'):proposal(manifest)


def test_incomplete_c2_cannot_generate_a_complete_cohort_gpu_proposal(tmp_path):
    p=tmp_path/'manifest.json';p.write_text('{"status":"INCOMPLETE"}')
    with pytest.raises(ValueError,match='Complete'):proposal(p)
