"""Canonical bounded M1 event-to-metric run; outputs are private and never overwritten."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, log_loss, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from noah_baseline.evaluation.contracts import (ContractError, sha256_file, validate_release,
                                               write_validated_report)
from noah_baseline.evaluation.landmarks import extract_landmark_state, validate_prediction_record
from noah_baseline.m1.data import (Event, VERSION, extract_sources, build_windows, landmark_request)
from noah_baseline.m1.training import REVISION, Trainer, check_recovery
from noah_baseline.m1.verification import verify_source_cases, verify_real_objective
from noah_baseline.runtime import ArtifactNamespace


def dump(path,value):
    if path.exists():raise ContractError('Artifact overwrite forbidden:'+str(path))
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def source_snapshot(root,output):
    names=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=root).split(b'\0')
    with tarfile.open(output,'w') as tar:
        for name in sorted(set(n.decode() for n in names if n)):
            path=root/name
            if not path.is_file() or path.is_symlink():raise ContractError('Source snapshot requires regular files')
            content=path.read_bytes();info=tarfile.TarInfo(name);info.size=len(content);info.mtime=0
            tar.addfile(info,io.BytesIO(content))
    return sha256_file(output)


def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p,dtype=np.float64)
    return dict(n=len(y),positive=int(y.sum()),prevalence=float(y.mean()),
        auroc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        auprc=float(average_precision_score(y,p)) if y.sum() else None,
        brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),
        accuracy=float(accuracy_score(y,p>=.5)),
        balanced_accuracy=float(balanced_accuracy_score(y,p>=.5)) if len(np.unique(y))==2 else None)


def fit_probe(features, labels, splits, config):
    train=np.flatnonzero(splits=='train');val=np.flatnonzero(splits=='val');test=np.flatnonzero(splits=='test')
    if not all(len(i) for i in (train,val,test)) or len(np.unique(labels[train]))!=2:
        raise ContractError('Probe needs nonempty split populations and both train classes')
    candidates=[];models=[]
    for C in config['probe_C_grid']:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=C,solver='lbfgs',max_iter=config['probe_max_iter'],
                                                                random_state=config['seed'],class_weight=None))
        model.fit(features[train],labels[train])
        if model[-1].n_iter_.max()>=config['probe_max_iter']:raise ContractError('Unconverged probe')
        probability=model.predict_proba(features[val])[:,1]
        candidates.append(dict(C=C,validation_log_loss=float(log_loss(labels[val],probability.astype(np.float64),labels=[0,1]))))
        models.append(model)
    best=min(range(len(models)),key=lambda i:(candidates[i]['validation_log_loss'],candidates[i]['C']))
    model=models[best]
    # Parameters and preprocessing stay train-fitted; validation selects only C.
    np.testing.assert_allclose(model[0].mean_,features[train].astype(np.float64).mean(axis=0),rtol=1e-12,atol=1e-12)
    probability=model.predict_proba(features)[:,1]
    report={name:metrics(labels[mask],probability[mask]) for name,mask in [('train',train),('val',val),('test',test)]}
    return model,probability,dict(selected_C=candidates[best]['C'],selection=candidates,metrics=report,
                                  fit_split='train',selection_split='val',test_used_for_selection=False)


def execute(root,hosp,ed,cohort,run_id,config,fixture_evidence):
    started=time.time()
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
        raise ContractError('Simple unique run ID required')
    if config['max_optimizer_steps']<2 or config['max_optimizer_steps']>16:
        raise ContractError('Only2–16 bounded optimizer steps permitted')
    if sha256_file(cohort)!='0beadca1a4580c37abff93252c57b73ca5ea7dadabb26865f1e43b1d5f0cac7e':
        raise ContractError('This M1 run requires the verified original 1000-patient cohort')
    ns=ArtifactNamespace(root,'reduced_pilot');out=ns.owned(ns.root/'runs'/run_id)
    if out.exists():raise ContractError('Run directory already exists')
    out.mkdir(parents=True)
    dump(out/'config.json',config)
    dump(out/'execution.json',dict(command=sys.argv,interpreter=sys.executable,python=platform.python_version(),
        torch=torch.__version__,numpy=np.__version__,host=platform.node(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        device='cpu',threads=2,scope='bounded four-step engineering integration; no substantial training'))
    snapshot_hash=source_snapshot(root,out/'source.tar')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    # Fixture results must be from an executed pytest report for the same source contents.
    fixture=json.loads(fixture_evidence.read_text())
    current_files={str(p.relative_to(root)):sha256_file(p) for p in sorted((root/'packages').rglob('*.py'))}
    current_files.update({str(p.relative_to(root)):sha256_file(p) for p in sorted((root/'tests').glob('*.py'))})
    if fixture['source_files']!=current_files or fixture['exit_code']!=0 or not all(fixture['cases'][f'M1-{i:02d}']=='PASS' for i in range(1,9)):
        raise ContractError('Executed fixture evidence does not match source or required M1 cases')
    dump(out/'fixture_evidence.json',fixture)
    indexes,events,birth,sources,excluded=extract_sources(hosp,ed,cohort)
    if len(indexes)!=1000:raise ContractError('Verified cohort size changed')
    source_checks=verify_source_cases(indexes,events,sources)
    dump(out/'real_source_checks.json',source_checks)
    # Exact cohort copy is private; no default split membership.
    (out/'patient_splits.csv').write_bytes(cohort.read_bytes())
    dump(out/'source_manifest.json',dict(transform=VERSION,sources=sources,exclusions=excluded,
        source_date_units='UTC-naive deidentified time expressed as elapsed hours since1970; within-subject only'))
    with (out/'events.jsonl').open('x') as stream:
        for event in events:stream.write(json.dumps(asdict(event),sort_keys=True,allow_nan=False)+'\n')
    dump(out/'index_cohort.json',indexes)
    grouped=defaultdict(list)
    for event in events:grouped[event.subject_id].append(event)
    windows=[];by_index={};selected=[]
    for index in indexes:
        if not index['eligible']:continue
        made=build_windows(grouped[index['subject_id']],index,birth[index['subject_id']],
                           length=config['window_length'],overlap=config['overlap'])
        by_index[index['hadm_id']]=made;windows.extend(made);selected.append(index)
    serialized=[dict(subject_id=w.subject_id,encounter_id=w.encounter_id,window_idx=w.window_idx,
        batch=w.batch,evidence=[asdict(e) for e in w.evidence],provenance=w.provenance,owned_target_ids=w.owned_target_ids)
        for w in windows]
    torch.save(serialized,out/'windows.pt')
    train_ids=[i['subject_id'] for i in selected if i['split']=='train']
    train_windows=[w for w in windows if w.subject_id in set(train_ids)]
    identity=dict(source_sha256=snapshot_hash,data_sha256=sha256_file(out/'source_manifest.json'),
        cohort_sha256=sha256_file(out/'patient_splits.csv'),window_sha256=sha256_file(out/'windows.pt'),
        train_subjects=sorted(train_ids),config_sha256=sha256_file(out/'config.json'))
    trainer,trace,recovery=check_recovery(config,train_windows,identity,out)
    while trainer.step<config['max_optimizer_steps']:trace.append(trainer.update())
    trainer.save(out/'model.pt');dump(out/'training_trace.json',trace);dump(out/'recovery.json',recovery)
    # Independently reload the actual final checkpoint before extracting any clinical feature.
    final=Trainer(config,train_windows,identity);final.load(out/'model.pt')
    for key,value in final.model.state_dict().items():
        if not torch.equal(value,trainer.model.state_dict()[key]):raise ContractError('Final checkpoint reload mismatch')
    model=final.model.eval()
    objective_check=verify_real_objective(model,train_windows[0],trainer.clock()[1])
    dump(out/'real_objective_check.json',objective_check)
    features=[];records=[];perturbations=[]
    for index in selected:
        window=by_index[index['hadm_id']][-1];request=landmark_request(index,window)
        state=extract_landmark_state(model,window.batch,window.evidence,request)
        # Independent exact-position oracle, not a comparison to another pooling helper.
        n=len(window.evidence)
        with torch.no_grad():
            b=window.batch;e=model.projector(b['category_ids'][:,:n],b['modality_ids'][:,:n],b['specifics_vec'][:,:n],
                b['has_specifics'][:,:n],b['value_vec'][:,:n],b['has_value'][:,:n])
            h=model.backbone(e,b['positions'][:,:n],b['ages'][:,:n],b['delta_next'][:,:n],b['delta_prev'][:,:n],b['attention_mask'][:,:n])
            torch.testing.assert_close(state,h[0,-1],rtol=0,atol=0)
        features.append(state.numpy())
        records.append(dict(subject_id=index['subject_id'],index_encounter_id=index['hadm_id'],
            source_label_encounter_id=index['hadm_id'],split=index['split'],label=index['label'],
            representation_kind='single_landmark',landmark_request=asdict(request),
            context_evidence=[asdict(e) for e in window.evidence],state_event_id=request.event_id))
    # Real cases cover each split and both observed outcomes, without altering training/evaluation membership.
    for key in sorted({(i['split'],i['label']) for i in selected}):
        index=next(i for i in selected if (i['split'],i['label'])==key)
        original=by_index[index['hadm_id']][-1]
        suffix=Event('adversarial:future',index['subject_id'],index['hadm_id'],'fixture','future','fixture',
                     index['cutoff']+1,None,28,999.,'bpm','heartrate')
        modified=build_windows(grouped[index['subject_id']]+[suffix],index,birth[index['subject_id']],
                               length=config['window_length'],overlap=config['overlap'])[-1]
        if original.provenance!=modified.provenance:raise ContractError('Future suffix changed provenance')
        for name in original.batch:
            if not torch.equal(original.batch[name],modified.batch[name]):raise ContractError('Future suffix changed model input')
        torch.testing.assert_close(extract_landmark_state(model,original.batch,original.evidence,landmark_request(index,original)),
            extract_landmark_state(model,modified.batch,modified.evidence,landmark_request(index,modified)),rtol=0,atol=0)
        perturbations.append(dict(split=key[0],outcome=key[1],status='PASS'))
    dump(out/'real_suffix_checks.json',perturbations)
    features=np.stack(features);labels=np.array([r['label'] for r in records]);splits=np.array([r['split'] for r in records])
    probe,probability,probe_report=fit_probe(features,labels,splits,config)
    changed_features=features.copy();changed_features[splits=='test']+=12345
    changed_labels=labels.copy();changed_labels[splits=='test']=1-changed_labels[splits=='test']
    changed_probe,_,changed_report=fit_probe(changed_features,changed_labels,splits,config)
    if probe_report['selected_C']!=changed_report['selected_C']:raise ContractError('Test outcomes changed model selection')
    for a,b in [(probe[0].mean_,changed_probe[0].mean_),(probe[0].scale_,changed_probe[0].scale_),
                (probe[-1].coef_,changed_probe[-1].coef_),(probe[-1].intercept_,changed_probe[-1].intercept_)]:
        np.testing.assert_array_equal(a,b)
    dump(out/'real_probe_invariance.json',dict(status='PASS',test_features_and_labels_perturbed=True,
        selection_unchanged=True,preprocessing_and_probe_parameters_bitwise_unchanged=True))
    np.savez(out/'features.npz',features=features,labels=labels,splits=splits,subject_id=[r['subject_id'] for r in records],
             encounter_id=[r['index_encounter_id'] for r in records])
    with (out/'probe.pkl').open('xb') as stream:pickle.dump(probe,stream)
    with (out/'probe.pkl').open('rb') as stream:reloaded=pickle.load(stream)
    np.testing.assert_array_equal(reloaded.predict_proba(features)[:,1],probability)
    for record,p in zip(records,probability):record['prediction']=float(p)
    with (out/'predictions.jsonl').open('x') as stream:
        for record in records:
            validate_prediction_record(record,'paper_reproduction')
            stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
    # Replay from the serialized prediction file, not the in-memory labels.
    readback=[json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
    replay={split:metrics([r['label'] for r in readback if r['split']==split],
                         [r['prediction'] for r in readback if r['split']==split]) for split in ('train','val','test')}
    if replay!=probe_report['metrics']:raise ContractError('Saved metric replay differs')
    dump(out/'probe_selection.json',probe_report)
    counts=Counter((i['split'],i['label']) for i in selected)
    admission_counts=Counter(e.subject_id for e in events if e.category==5)
    aggregate=dict(source_subjects_with_multiple_admissions=sum(n>1 for n in admission_counts.values()),
        source_hospital_admissions=sum(admission_counts.values()),index_patients=len(indexes),eligible=len(selected),events=len(events),windows=len(windows),
        train_windows=len(train_windows),availability_unknown=sum(e.available_at is None for e in events),
        eligibility_exclusions=dict(Counter(i['exclusion'] for i in indexes if not i['eligible'])),
        split_counts={split:dict(n=sum(v for (s,y),v in counts.items() if s==split),deaths=counts[(split,1)]) for split in ('train','val','test')},
        real_suffix_checks=perturbations,source_checks=source_checks,objective_reference=objective_check,
        test_tuning_invariance='PASS',steps=trainer.step,elapsed_seconds=round(time.time()-started,2),
        executed_cases={f'M1-{i:02d}':dict(status='PASS',scope='executed source-like pytest fixtures plus bounded real-data assertions') for i in range(1,9)})
    modules={name:sum(p.numel() for p in getattr(model,name).parameters()) for name in ('projector','backbone','prior','posterior','decoder')}
    predictions_sha=sha256_file(out/'predictions.jsonl')
    checks={}
    procedures={'split_integrity':'verified source patient split, explicit train-only windows and probe fitting',
        'architecture_identity':'strict checkpoint reload, configuration/data/revision identities and per-module counts',
        'metric_replay':'reloaded probe probability identity and serialized prediction/metric replay',
        'endpoint_protocol':'first admission chosen before24h eligibility, exact encounter flag labels and fixture discordant-stay assertions',
        'landmark_identity':'all eligible states equal independent h[last], exact encounter/event provenance',
        'no_future_events':'all cutoff views bounded; real suffix perturbations across splits/outcomes are tensor/state invariant',
        'no_full_record_pooling':'canonical single-event extraction checked against exact h[last] for every index',
        'probe_protocol':'train-fitted scaler/logistic parameters, validation log-loss C selection, frozen held-out test'}
    for name,procedure in procedures.items():
        item=dict(check=name,status='PASS',source_sha256=snapshot_hash,predictions_sha256=predictions_sha,
                  procedure=procedure,fixture_evidence_sha256=sha256_file(out/'fixture_evidence.json'),
                  aggregate=aggregate)
        dump(out/(name+'.json'),item)
        checks[name]=dict(status='PASS',source_sha256=snapshot_hash,predictions_sha256=predictions_sha,
                         evidence_path=name+'.json',evidence_sha256=sha256_file(out/(name+'.json')))
    manifest=dict(schema_version=1,run_id=run_id,mode='paper_reproduction',protocol='landmark_retrieval',
        scope='M1 reduced-modality occurrence-time engineering diagnostic; not paper-faithful or prospective',
        code=dict(commit=commit,source_sha256=snapshot_hash,path='source.tar',tree_state='dirty snapshot' if subprocess.check_output(['git','status','--porcelain'],cwd=root) else 'clean commit plus source snapshot'),
        architecture=dict(revision=REVISION,parameter_count=sum(modules.values()),module_counts=modules),
        checkpoint=dict(path='model.pt',sha256=sha256_file(out/'model.pt'),architecture_revision=REVISION),
        encoders=[],randomness=dict(seeds=[config['seed']],policy='CPU deterministic algorithms; Python/NumPy/Torch RNG saved and restored'),
        datasets=[dict(name='MIMIC-IV hospital and MIMIC-IV-ED',version='2.2',manifest_sha256=sha256_file(out/'source_manifest.json'),encoder_recipe='no external encoders; candidate44-D numeric basis')],
        patient_splits=dict(path='patient_splits.csv',sha256=sha256_file(out/'patient_splits.csv'),unit='patient'),
        temporal_cutoff=dict(definition='index hospital admission+24h',boundary='occurrence<=cutoff; discharge/death must be>cutoff',gap_policy='explicit_unknown_constant',metadata_policy=config['demographics']),
        labels=dict(endpoint_id='in_hospital_mortality',definition='hospital_expire_flag of selected hadm_id',
            index_encounter_rule=config['index'],horizon='remaining selected hospitalization after24h',
            censoring='exclude missing/inconsistent mortality or death timing and stays ending by24h; within-hospital endpoint, no30d follow-up assumption'),
        availability_rules=dict(definition='source occurrence diagnostic; no source record has observed availability in these tables',unknown_policy='preserve null; never relabel as prospective'),
        probe_protocol=dict(definition=config['probe'],training_population='eligible train patients only',validation='original val split log-loss C selection, no test tuning',seeds=[config['seed']]),
        predictions=dict(path='predictions.jsonl',sha256=predictions_sha),
        metric_protocol=dict(definition='sklearn AUROC/AP/Brier/log-loss/accuracy/balanced accuracy; replayed per split',code_sha256=sha256_file(Path(__file__)),uncertainty='engineering demonstration; very few deaths; no inferential/clinical efficacy claims'),
        decisions={'M1_scope':dict(status='APPROVED',approval_or_evidence='researcher M1 authorization in session: categorical/numeric, first pre-outcome deterministic index,24h mortality, occurrence diagnostic, candidate objective retained'),
            'operational_choices':dict(status='RESOLVED',approval_or_evidence='configs/m1.json; documented bounded engineering choices within authorized scope')},
        deviations=config['deviations'],checks=checks,effective_config=dict(path='config.json',sha256=sha256_file(out/'config.json')),
        aggregate=aggregate,source_manifest=dict(path='source_manifest.json',sha256=sha256_file(out/'source_manifest.json')))
    manifest['execution']=dict(path='execution.json',sha256=sha256_file(out/'execution.json'))
    manifest['acceptance_fixture_evidence']=dict(path='fixture_evidence.json',sha256=sha256_file(out/'fixture_evidence.json'))
    manifest['training_artifacts']={name:sha256_file(out/name) for name in ('windows.pt','events.jsonl','index_cohort.json','config.json','training_trace.json','recovery.json','features.npz','probe.pkl','probe_selection.json')}
    dump(out/'experiment_manifest.json',manifest)
    validate_release(manifest,'retrospective',base=out)
    for forbidden in ('prospective','paper_faithful'):
        try:validate_release(manifest,forbidden,base=out)
        except ContractError:pass
        else:raise ContractError('Unsupported scientific claim passed gate')
    write_validated_report(manifest,'retrospective',probe_report['metrics'],base=out,output=out/'validated_metrics.json')
    dump(out/'acceptance.json',aggregate)
    inventory={p.name:sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}
    dump(out/'artifact_inventory.json',inventory)
    print(json.dumps(dict(run=run_id,output=str(out),aggregate=aggregate,metrics=probe_report['metrics'],selected_C=probe_report['selected_C']),sort_keys=True))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hosp',type=Path,required=True);parser.add_argument('--ed',type=Path,required=True)
    parser.add_argument('--cohort',type=Path,required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/m1.json'))
    parser.add_argument('--fixture-evidence',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    execute(root,args.hosp,args.ed,args.cohort,args.run_id,json.loads(args.config.read_text()),args.fixture_evidence)
    return 0
