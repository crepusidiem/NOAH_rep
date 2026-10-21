"""Resource proposal from actual completed C2 content queues; never inference."""
from pathlib import Path
import json

import pyarrow.parquet as pq

from noah_baseline.m2.stream_index import digest_file

DIMENSIONS={'text':768,'specifics':768,'cxr':768,'ecg':192,'echo':512}


def proposal(dataset_manifest):
    path=Path(dataset_manifest);data=json.loads(path.read_text())
    if data['status']!='CANDIDATE_CANONICAL_PREPARATION_COMPLETE_EMBEDDINGS_PENDING':
        raise ValueError('Complete, source-verified C2 preparation required for the full-cohort proposal')
    execution_path=path.parent/'execution.json';identity_path=path.parent/'manifest.json'
    execution=json.loads(execution_path.read_text());identity=json.loads(identity_path.read_text())
    if execution['status']!='COMPLETE_PREPARATION_NOT_MODEL_READY' or identity['config_sha256']!=data['config_sha256']:
        raise ValueError('Terminal C2 execution/config identity mismatch')
    counts={kind:0 for kind in DIMENSIONS};identities={}
    for kind,queue in data['encoding_queues'].items():
        if kind not in DIMENSIONS:raise ValueError('Unexpected encoding queue kind')
        qpath=Path(queue['path'])
        if digest_file(qpath)!=queue['sha256']:raise ValueError('Encoding task queue changed')
        n=pq.ParquetFile(qpath).metadata.num_rows
        if n!=queue['tasks']:raise ValueError('Queue count disagrees with materialized tasks')
        counts[kind]=n;identities[kind]=dict(sha256=queue['sha256'],tasks=n)
    payload=sum(counts[k]*DIMENSIONS[k]*4 for k in counts)
    echo_low=counts['echo']*5.001698903739452/3600
    echo_high=counts['echo']*6.703587080352008/3600
    return dict(status='PROPOSED_NOT_AUTHORIZED',execution_authorized=False,submissions=0,
        c2_job_id=identity['job_id'],c2_source_commit=identity['source_commit'],
        c2_execution_sha256=digest_file(execution_path),c2_source_identity_sha256=digest_file(identity_path),
        c2_dataset_manifest_sha256=digest_file(path),c2_config_sha256=data['config_sha256'],
        patient_splits_sha256=data['patient_splits_sha256'],cohort=data['cohort'],
        actual_deduplicated_tasks=counts,queue_identities=identities,dimensions=DIMENSIONS,
        float32_vector_payload_bytes=payload,
        proposed_hard_caps=dict(jobs=1,gpus=1,gpu_type='L40S',gpu_memory_GiB=48,cpus=16,memory_GiB=64,
            wall_hours=24,new_output_GiB=48,temporary_staging_GiB=8,requeue=False,retry=False,training=False,alpha=False),
        maximum_allocated_gpu_hours=24,maximum_allocated_cpu_hours=384,
        proposed_execution=dict(cpu_preprocessing_workers=8,per_worker_threads=1,
            batch_sizes={'text':64,'specifics':64,'cxr':8,'ecg':4,'echo':4},
            phase_time_caps_hours={'text_and_specifics':3,'cxr':3,'ecg':3,'echo':14,'identity_replay_and_finalization':1},
            source_policy='recheck exact queue content/recipe/checkpoint identity; read source notes once; bounded temporary clips; source read-only',
            reuse='only verified clean cache references with exact content/checkpoint/complete recipe; retain explicit parent-dataset attestation; reject unverifiable legacy caches',
            failure='record per-task failure and all event bindings; no zero fallback, batch-size adaptation, CPU inference fallback or retry; save partial status at cap',
            verification='strict frozen loads; repeat a bounded initial batch per modality and replay saved values; verify all shapes/dtypes/content bindings; report actual modality/patient coverage'),
        conditional_echo_preprocessing=dict(serial_worker_hours=[echo_low,echo_high],
            ideal_eight_worker_hours=[echo_low/8,echo_high/8],
            limitation='Two B-mode clips from one patient; not measured full-queue throughput. I/O/unsupported clips/length variation and overhead can change runtime.'),
        other_bulk_throughput='NOT_MEASURED; phase reservations are budget caps, not predicted throughput or a completion guarantee',
        storage_note='Vector payload excludes cache headers, filesystem blocks, task/provenance/status metadata and at most8GiB temporary clips; avoid persistent duplicate AVIs',
        readiness='HOLD; successful preprocessing/encoding coverage and refreshed actual target ownership are required before any training proposal')
