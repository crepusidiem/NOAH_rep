"""Small genuine-source six-modality shard/window/model-routing verification.

Uses only authenticated clean M1/M2 evidence; no legacy caches or bulk extraction.
Separate OTIS position candidates produce separate datasets and output namespaces.
"""
from collections import Counter
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from noah_baseline.data.numeric import DyadicFourierEncoder
from noah_baseline.m2.dataset import entry,write_shards,read_shards,model_windows
from noah_baseline.m2.events import Event,EmbeddingCache
from noah_baseline.m2.inventory import dump,sha
from noah_baseline.models.noah import NOAH


def integration_smoke(root, *, out, source, media_run):
    start=time.monotonic();out.mkdir(exist_ok=False)
    config=json.loads((root/'configs/m2.json').read_text())
    recipe=json.loads((root/'configs/m2_integration.json').read_text())
    media=json.loads((media_run/'private_results.json').read_text())
    if any(media['modalities'][k].get('encoder_check')!='VERIFIED' for k in ['ecg','echo']):
        raise ValueError('Both genuine media encoder checks must pass before six-modality integration')
    if media.get('auxiliary_specifics',{}).get('status')!='VERIFIED':
        raise ValueError('Actual source-specific item vectors must be available')
    with (source/'patient_splits.csv').open(newline='') as f:
        splits={int(r['subject_id']):r['split'] for r in csv.DictReader(f)}
    if len(splits)!=15000 or sha(source/'patient_splits.csv')!='5df6b9e96b1921b94867cdacd04440de25a42e84d33c4a6fbd9ad5e26277d869':
        raise ValueError('Frozen expanded patient membership changed')
    initial=root/'artifacts/reduced_pilot/m2/m2-source-smoke-20261020-03'
    originals=[Event(**e) for e in json.loads((initial/'events.json').read_text())]
    samples=json.loads((source/'small_source_samples.json').read_text())
    prior=json.loads((root/'artifacts/reduced_pilot/m2'/recipe['verified_content_run']/'private_results.json').read_text())
    cache=EmbeddingCache(root);dataset_sha=sha(source/'source_inventory.json')
    references={};source_hashes={};specifics={}
    for row in prior['records']:
        if row['kind']=='text':
            identity='note:'+row['source']['row']['note_id']
            source_hashes[identity]=hashlib.sha256(row['source']['row']['text'].encode()).hexdigest()
        elif row['kind']=='cxr':
            identity='cxr:'+row['source']['row']['dicom_id']
            source_hashes[identity]=sha(Path(row['image_path']))
        else:
            continue
        references[identity]=row['embedding']
    checks=json.loads((initial/'private_checks.json').read_text())
    for row in checks:
        if row['kind']=='numeric':references['ed:'+row['source_id']]=row['cache']
    for sample in samples:
        if sample['dataset']=='ed' and sample['table']=='vitalsign':
            identity='ed:csv_record_'+sample['row']['source_csv_record_ordinal']+':heartrate'
            source_hashes[identity]=hashlib.sha256(str(sample['row']['heartrate']).encode()).hexdigest()
    for row in media['records']:
        if row['kind']=='specifics':specifics[row['specific_item']]=row['embedding']
    for record in samples:
        if record['dataset']=='ecg':
            path=Path(config['roots']['ecg'])/record['row']['path']
            source_hashes['ecg:'+record['row']['study_id']]=hashlib.sha256((sha(path.with_suffix('.hea'))+sha(path.with_suffix('.dat'))).encode()).hexdigest()
        elif record['dataset']=='echo':
            path=Path(config['roots']['echo'])/record['row']['dicom_filepath']
            source_hashes['echo:'+record['row']['dicom_filepath']]=sha(path)
    # Verified clean M1 histories provide real overlap stress cases, not cohort reselection.
    m1=root/'artifacts/reduced_pilot/runs/m1-real-20261019-02'
    m1_inventory=json.loads((m1/'artifact_inventory.json').read_text())
    if sha(m1/'events.jsonl')!=m1_inventory['events.jsonl']:
        raise ValueError('Immutable M1 raw event evidence changed')
    m1_rows=[json.loads(line) for line in (m1/'events.jsonl').read_text().splitlines()]
    counts=Counter(r['subject_id'] for r in m1_rows if splits.get(r['subject_id'])=='train')
    stress_subjects=sorted(s for s,n in counts.items() if n>=70)[:2]
    if len(stress_subjects)!=2:raise ValueError('Two actual overlap stress cases required')
    stress=[]
    for subject in stress_subjects:
        for raw in [r for r in m1_rows if r['subject_id']==subject][:160]:
            table=raw['source_table'];kind,name=table.split('.')
            path=Path(config['roots']['hosp' if kind=='hosp' else 'ed'])/(name+'.csv')
            event=Event(raw['event_id'],subject,raw['encounter_id'],'source',kind,raw['source_version'],
                raw['source_row_id'],str(path),raw['occurred_at'],raw['available_at'],
                'unknown' if raw['available_at'] is None else 'source',
                'numeric' if raw['value'] is not None else 'categorical',raw['category'],
                raw['variable'],raw['units'],'m2-clean-m1-stress-v1')
            reference=None
            if raw['value'] is not None:
                digest=hashlib.sha256(str(raw['value']).encode()).hexdigest();event=replace(event,content_sha256=digest)
                reference=cache.put(DyadicFourierEncoder().encode(raw['value']),kind='numeric',content_sha256=digest,
                    dataset_sha256=dataset_sha,checkpoint_sha256=sha(root/'packages/noah_baseline/data/numeric.py'),
                    recipe=dict(transform_revision='m2-clean-m1-numeric-v1',source='authenticated immutable M1 event scalar',
                        normalization='none; unchanged M1',units=event.units,phase=1.,scales=[-7,14],layout='sin then cos'))
            row=entry(event,reference=reference,specifics_reference=specifics.get(event.specific_item))
            row['upstream_event_file_sha256']=m1_inventory['events.jsonl'];stress.append(row)
    births={}
    with (Path(config['roots']['hosp'])/'patients.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            subject=int(r['subject_id'])
            if subject in splits:births[subject]=int(r['anchor_year'])-int(r['anchor_age'])
    capture=out/'executed-source';capture.mkdir()
    for p in [root/'configs/m2.json',root/'configs/m2_integration.json',root/'tools/m2_run.py',
              *sorted((root/'packages/noah_baseline/m2').glob('*.py'))]:
        target=capture/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    summaries={}
    for policy in recipe['ecg']['position_policies']:
        mapped=dict(references)
        for row in media['records']:
            if row['kind']=='ecg' and row['variant']==policy:
                mapped['ecg:'+row['source']['row']['study_id']]=row['embedding']
            elif row['kind']=='echo':mapped['echo:'+row['source']['row']['dicom_filepath']]=row['embedding']
        rows=[]
        for event in originals:
            # Actual WFDB preprocessing verified all leads are in mV; retain that known source fact.
            if event.modality=='ecg':event=replace(event,units='mV')
            reference=None if event.modality=='categorical' else mapped[event.event_id]
            if reference is not None:event=replace(event,content_sha256=source_hashes[event.event_id])
            rows.append(entry(event,reference=reference,specifics_reference=specifics.get(event.specific_item)))
        rows.extend(stress)
        if len({r['event']['event_id'] for r in rows})!=len(rows):
            raise ValueError('Smoke and stress sources overlap; explicitly reconcile source identities before deduplication')
        directory=out/policy;manifest=write_shards(root,directory,rows,splits,dataset_sha256=dataset_sha)
        replay=read_shards(root,directory,splits)
        assert sorted(json.dumps(r,sort_keys=True) for r in replay)==sorted(json.dumps(r,sort_keys=True) for r in rows)
        windows=[];strict_windows=[];strict_metadata_blocks=0
        for subject in sorted({r['event']['subject_id'] for r in replay}):
            subset=[r for r in replay if r['event']['subject_id']==subject]
            cutoff=max(r['event']['occurred_at'] for r in subset)
            windows.extend(model_windows(root,replay,subject_id=subject,birth_year=births[subject],
                length=recipe['windows']['length'],overlap=recipe['windows']['overlap'],cutoff=cutoff,
                profile='occurrence_diagnostic',expected_dataset_sha256=dataset_sha))
            try:
                strict_windows.extend(model_windows(root,replay,subject_id=subject,birth_year=births[subject],
                    length=recipe['windows']['length'],overlap=recipe['windows']['overlap'],cutoff=cutoff,
                    profile='strict_prospective',expected_dataset_sha256=dataset_sha,demographic_available_at=None))
            except ValueError as error:
                if 'Strict prospective demographic availability' not in str(error):raise
                strict_metadata_blocks+=1
        ledger=Counter(e for w in windows for e in w['owned_target_ids'])
        if ledger!=Counter(r['event']['event_id'] for r in replay):
            raise ValueError('Real-data shifted owner ledger changed during replay')
        torch.manual_seed(recipe['seed']);torch.set_num_threads(2)
        model=NOAH(**json.loads((root/'configs/m1.json').read_text())['model']).eval()
        losses=[]
        with torch.inference_mode():
            for window in windows:
                loss,_,_=model(**window['batch'],beta=0.)
                if not torch.isfinite(loss):raise ValueError('Nonfinite untrained six-modality model routing')
                losses.append(float(loss))
        # Perturb actual future suffixes and verify retained tensors AND hidden states.
        suffix_checks=[]
        for subject in stress_subjects:
            source_rows=sorted([r for r in replay if r['event']['subject_id']==subject],key=lambda r:r['event']['occurred_at'])
            cutoff=source_rows[len(source_rows)//2]['event']['occurred_at']
            args=dict(subject_id=subject,birth_year=births[subject],length=64,overlap=16,
                cutoff=cutoff,profile='occurrence_diagnostic',expected_dataset_sha256=dataset_sha)
            before=model_windows(root,replay,**args)
            changed=[]
            for row in replay:
                if row['event']['subject_id']==subject and row['event']['occurred_at']>cutoff:
                    row=dict(row,event=dict(row['event'],occurred_at=row['event']['occurred_at']+100000))
                changed.append(row)
            after=model_windows(root,changed,**args)
            if len(before)!=len(after):raise ValueError('Future suffix changed real cutoff windows')
            for a,b in zip(before,after):
                if any(not torch.equal(a['batch'][key],b['batch'][key]) for key in a['batch']):
                    raise ValueError('Future suffix changed retained model features')
                with torch.inference_mode():
                    _,_,first=model(**a['batch'],beta=0.)
                    _,_,second=model(**b['batch'],beta=0.)
                if not torch.equal(first['h'],second['h']):raise ValueError('Real cutoff hidden states changed')
            suffix_checks.append(dict(status='PASS',retained_windows=len(before),model_hidden_states='bitwise'))
        dump(directory/'real_suffix_invariance.json',dict(status='PASS',checks=suffix_checks))
        torch.save(windows,directory/'windows.pt');torch.save(strict_windows,directory/'strict-windows.pt')
        restored=torch.load(directory/'windows.pt',weights_only=True,map_location='cpu')
        if len(restored)!=len(windows):raise ValueError('Serialized window count changed')
        for before,after in zip(windows,restored):
            if before['owned_target_ids']!=after['owned_target_ids'] or before['clinical_provenance']!=after['clinical_provenance']:
                raise ValueError('Serialized window lost source identities')
            if any(not torch.equal(before['batch'][k],after['batch'][k]) for k in before['batch']):
                raise ValueError('Serialized model inputs changed')

        dump(directory/'target_ownership.json',dict(status='PASS',owners=dict(ledger)))
        summary=dict(status='PASS',events=len(replay),patients=len({r['event']['subject_id'] for r in replay}),
            windows=len(windows),strict_available_windows=len(strict_windows),strict_metadata_blocks=strict_metadata_blocks,
            strict_prospective_metadata_status='BLOCKED_UNKNOWN_SOURCE_AVAILABILITY',
            modalities=dict(Counter(r['event']['modality'] for r in replay)),
            specifics_bound=sum(r['specifics_reference'] is not None for r in replay),
            actual_overlap_stress_patients=len(stress_subjects),missing_value_embeddings=0,
            shifted_ownership='exactly_once',shard_replay='exact',serialized_window_replay='bitwise tensors and exact source sidecars',real_suffix_invariance='bitwise tensors and hidden states for two source histories',model_parameters=sum(p.numel() for p in model.parameters()),
            untrained_forward_windows=len(losses),untrained_objective_mean=float(np.mean(losses)),
            interpretation='routing/numerical diagnostic only; not learning, efficacy or prospective performance')
        dump(directory/'aggregate.json',summary);summaries[policy]=summary
    aggregate=dict(status='PASS',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        recipe_sha256=sha(root/'configs/m2_integration.json'),expanded_split_sha256=sha(source/'patient_splits.csv'),
        media_results_sha256=sha(media_run/'private_results.json'),source_inventory_sha256=dataset_sha,
        full_cohort_materialized=False,variants=summaries,elapsed_seconds=time.monotonic()-start,
        stress_sampling='two lowest-ID train patients with>=70 immutable M1 events, bounded160 each; verification only, no cohort alteration',
        training_launched=False,bulk_encoding_launched=False)
    dump(out/'aggregate.json',aggregate);print(json.dumps(aggregate,indent=2))
