"""One approved, CPU-capped learning run with immutable artifacts and held-out test quarantine."""
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch

from noah_baseline.evaluation.contracts import ContractError, sha256_file
from noah_baseline.m1.data import Window
from noah_baseline.m15.metrics import CLAIM, targets, fit_baselines, baseline_predictions, score
from noah_baseline.m15.operations import LearningTrainer, evaluate, exact, state


class BudgetStop(RuntimeError):
    """Stop the one run without expanding the approved budget."""


def dump(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


class Guard:
    def __init__(self, root, started=None):
        self.root = root; self.started = time.monotonic() if started is None else started
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self.request_stop)

    def request_stop(self, *_):
        self.stop_requested = True

    def bytes(self):
        return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def check(self):
        # Reserve120s and128MiB for a final state and receipts, before the external hard cap.
        if self.stop_requested or time.monotonic()-self.started >= 3480:
            raise BudgetStop('Elapsed-time stop; preserve checkpoint, no extension')
        if self.bytes() + 128*1024**2 >= 2*1024**3:
            raise BudgetStop('Output-cap reserve reached')


def load_data(root, config):
    run = root/config['m1_run']
    if sha256_file(run/'artifact_inventory.json') != config['m1_inventory_sha256']:
        raise ContractError('M1 inventory identity mismatch')
    inventory = json.loads((run/'artifact_inventory.json').read_text())
    for name, digest in inventory.items():
        path = run/name
        if sha256_file(path) != digest or path.stat().st_mode & 0o222:
            raise ContractError('M1 immutable artifact mismatch: '+name)
    for name, digest in config['pinned_m1_inputs'].items():
        if inventory[name] != digest:
            raise ContractError('Pinned M1 input mismatch')
    # Preserve all pre-existing mathematical, data and M1 operations exactly.
    with tarfile.open(run/'source.tar') as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.startswith(('packages/noah_baseline/models/', 'packages/noah_baseline/data/', 'packages/noah_baseline/m1/')) and member.name.endswith('.py'):
                if archive.extractfile(member).read() != (root/member.name).read_bytes():
                    raise ContractError('M1 model/data/operation source changed: '+member.name)
    cohort = json.loads((run/'index_cohort.json').read_text())
    by_subject = {r['subject_id']: r for r in cohort}
    # Loading the shared serialized container is necessary; do not inspect test tensor contents.
    saved = torch.load(run/'windows.pt', map_location='cpu', weights_only=False)
    partitions = {'train': [], 'val': []}; ordinals = {'train': [], 'val': []}
    for ordinal, item in enumerate(saved):
        row = by_subject[item['subject_id']]
        if row['split'] == 'test':
            continue
        if row['split'] not in partitions or not row['eligible'] or row['hadm_id'] != item['encounter_id']:
            raise ContractError('Window cohort/encounter mismatch')
        w = Window(**item)
        for event in w.evidence:
            if event['occurred_at'] > row['cutoff'] or event['encounter_id'] != w.encounter_id:
                raise ContractError('Future or cross-encounter evidence')
        b = w.batch
        owned = b['loss_mask'][:, 1:] & b['attention_mask'][:, 1:] & b['attention_mask'][:, :-1]
        reference = [w.evidence[t]['event_id'] for t in range(1, len(w.evidence)) if bool(b['loss_mask'][0,t])]
        if reference != w.owned_target_ids or int(owned.sum()) != len(reference):
            raise ContractError('Independent shifted ownership failed')
        last = int(b['attention_mask'].sum())-1
        if b['has_next_gap'][0,last] or b['delta_next'][0,last] != 0:
            raise ContractError('Landmark future-gap disclosure')
        partitions[row['split']].append(w); ordinals[row['split']].append(ordinal)
    assert not ({w.subject_id for w in partitions['train']} & {w.subject_id for w in partitions['val']})
    for split, display in [('train', 'train'), ('val', 'validation')]:
        windows = partitions[split]; ids = [v for w in windows for v in w.owned_target_ids]
        expected = config['partitions'][display]
        digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        t = targets(windows)
        assert digest == expected['owned_target_ledger_sha256']
        assert len(windows) == expected['windows'] and len(ids) == expected['owned_targets'] == len(set(ids))
        assert np.bincount(t['category'], minlength=31).tolist() == expected['category_count_vector']
        assert int(t['gap_known'].sum()) == expected['time_observed_zero']+expected['time_observed_positive']
    return partitions, ordinals


def make_trainer(config, partitions, identity):
    return LearningTrainer(config['candidate'], partitions['train'], identity, 5260)


def save_predictions(out, epoch, split, prediction, truth):
    path = out/f'epoch{epoch:02d}-{split}-predictions.npz'
    if path.exists():
        raise ContractError('Prediction overwrite forbidden')
    np.savez_compressed(path, **prediction, **{'target_'+k:v for k,v in truth.items()})
    # Independent serialized replay; all metric reductions use saved float64 outputs.
    with np.load(path, allow_pickle=False) as artifact:
        actual = score({k: artifact['target_'+k] for k in truth}, {k: artifact[k] for k in prediction})
    if actual != score(truth, prediction):
        raise ContractError('Metric artifact replay failed')
    return dict(path=path.name, sha256=sha256_file(path), metric_replay='PASS')


def replay_checkpoint(root, out, epoch):
    config = json.loads((out/'resolved_config.json').read_text())
    identity = json.loads((out/'identity.json').read_text())
    partitions, ordinals = load_data(root, config)
    trainer = make_trainer(config, partitions, identity); trainer.load(out/f'epoch{epoch:02d}.pt')
    for split in ('train', 'val'):
        prediction, _, _ = evaluate(trainer.model, partitions[split], ordinals[split], min(trainer.cursor/526/10, 1))
        with np.load(out/f'epoch{epoch:02d}-{split}-predictions.npz', allow_pickle=False) as artifact:
            for key, value in prediction.items():
                if not np.array_equal(value, artifact[key]):
                    raise ContractError('Fresh-process prediction mismatch: '+key)
    dump(out/f'epoch{epoch:02d}-fresh-replay.json', dict(status='PASS', tensor_equality='bitwise', test_evaluated=False))


def run(root, run_id):
    started = time.monotonic()
    config = json.loads((root/'configs/m15.json').read_text())
    if not config['execution_enabled'] or config['status'] != 'APPROVED':
        raise ContractError('M1.5 configuration is not approved')
    if os.environ.get('SLURM_CPUS_PER_TASK') != '2' or len(os.sched_getaffinity(0)) > 2:
        raise ContractError('Require an exactly two-CPU bound SLURM step')
    if torch.cuda.is_initialized():
        raise ContractError('GPU initialization forbidden')
    parent = root/'artifacts/reduced_pilot/m15'
    if Path(run_id).name != run_id:
        raise ContractError('Invalid run identifier')
    out = parent/run_id
    # One permanent claim per approval. This is not a reusable generic training launcher.
    dump(parent/'approved-run-claim.json', dict(run_id=run_id, approval='researcher2026-10-20', max_runs=1))
    out.mkdir(exist_ok=False)
    guard = Guard(parent, started)
    dump(out/'resolved_config.json', config)
    file_names = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard'], text=True).splitlines()
    file_names = sorted(set(n for n in file_names if (root/n).is_file()))
    source_files = {n: sha256_file(root/n) for n in file_names}
    dump(out/'source_files.json', source_files)
    with tarfile.open(out/'source.tar', 'x') as archive:
        for name in file_names:
            archive.add(root/name, arcname=name, recursive=False)
    source_hash = sha256_file(out/'source.tar')
    partitions, ordinals = load_data(root, config)
    identity = dict(controller='m15-v1', source_sha256=source_hash,
        config_sha256=sha256_file(out/'resolved_config.json'), windows_sha256=config['pinned_m1_inputs']['windows.pt'],
        train_subjects=[w.subject_id for w in partitions['train']])
    dump(out/'identity.json', identity)
    dump(out/'random_mapping.json', dict(seed=1729, formula='1729+100*original_window_ordinal+stream',
        streams=dict(objective=0, prior_predictive=list(range(1,9)), posterior_diagnostic=99),
        generator='torch CPU MT19937', ordinals=ordinals))
    for split in partitions:
        dump(out/(split+'-ownership.json'), [dict(subject_id=w.subject_id, encounter_id=w.encounter_id,
            target_ids=w.owned_target_ids) for w in partitions[split]])
    baseline = fit_baselines(partitions['train'])
    np.savez_compressed(out/'fitted_baselines.npz', **baseline)
    truths = {s: targets(w) for s,w in partitions.items()}
    baseline_scores = {s: {} for s in partitions}
    for split in partitions:
        for mode in ('unigram','transition'):
            prediction=baseline_predictions(truths[split],baseline,mode)
            path=out/f'baseline-{mode}-{split}.npz'
            np.savez_compressed(path,**prediction,**{'target_'+k:v for k,v in truths[split].items()})
            with np.load(path) as saved:
                replay=score({k:saved['target_'+k] for k in truths[split]}, {k:saved[k] for k in prediction})
            baseline_scores[split][mode]=score(truths[split],prediction)
            assert replay==baseline_scores[split][mode]
    for split in partitions:
        t=truths[split]; pos=t['gap_known'] & (t['gap']>0)
        baseline_scores[split]['positive_median_hours_mae'] = float(np.abs(t['gap'][pos]-baseline['median_gap']).mean())
    dump(out/'baseline_metrics.json', baseline_scores)
    trainer = make_trainer(config, partitions, identity)
    assert sum(p.numel() for p in trainer.model.parameters()) == 4760468
    histories=[]; recovery={}; physical=0; status='INCOMPLETE'; reason=None
    dump(out/'manifest.json', dict(schema_version=1, run_id=run_id, claim=CLAIM,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(), source_sha256=source_hash,
        source_files_sha256=sha256_file(out/'source_files.json'), resolved_config_sha256=identity['config_sha256'],
        architecture_revision=config['architecture_revision'], parameters=4760468,
        m1_inputs=config['pinned_m1_inputs'], source_dataset_manifest=config['pinned_m1_inputs']['source_manifest.json'],
        patient_split_manifest=config['pinned_m1_inputs']['patient_splits.csv'],
        cutoff='selected first hospital encounter admission+24h', availability='unknown preserved; occurrence diagnostic',
        target_definition='owned shifted category and outgoing gap; current M1 loss masks', probe='none',
        evaluation='fixed train/val; eight prior draws; no test evaluation',
        resources=dict(job_id=os.environ.get('SLURM_JOB_ID'),step_id=os.environ.get('SLURM_STEP_ID'),
            cpus_per_task=os.environ.get('SLURM_CPUS_PER_TASK'),affinity=sorted(os.sched_getaffinity(0)),
            memory_limit_GiB=16,elapsed_limit_seconds=3600,output_limit_bytes=2*1024**3),
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__),
        deviations=['M1 reduced modalities and dimensions','teacher-forced time conditioning','unknown availability',
            'fixed candidate weights and loss reductions','no rare-category support'],
        release_gate=dict(paper_faithful=False,prospective=False,zero_shot_rollout=False,allowed_claim=CLAIM)))
    trace_handle = (out/'training.jsonl').open('x')
    try:
        for epoch in range(21):
            guard.check()
            if epoch > 0:
                target_step=epoch*263
                while trainer.step < target_step:
                    guard.check()
                    record=trainer.update(); physical+=1
                    record['elapsed_seconds']=time.monotonic()-started
                    trace_handle.write(json.dumps(record,allow_nan=False)+'\n');trace_handle.flush()
                    if trainer.step % 50 == 0:
                        print(json.dumps(dict(step=trainer.step,epoch=epoch,elapsed=round(time.monotonic()-started,1))),flush=True)
            guard.check(); trainer.save(out/f'epoch{epoch:02d}.pt')
            before=state(trainer)
            row=dict(epoch=epoch,step=trainer.step,beta=min(trainer.cursor/526/10,1.),splits={})
            for split in partitions:
                p,m,c=evaluate(trainer.model,partitions[split],ordinals[split],row['beta'],guard.check)
                receipt=save_predictions(out,epoch,split,p,truths[split])
                row['splits'][split]=dict(metrics=m,objective=c,predictions=receipt)
                if epoch in (0,1,20):
                    repeated,_,_=evaluate(trainer.model,partitions[split],ordinals[split],row['beta'],guard.check)
                    if not exact(p,repeated):raise ContractError('Repeated prediction mismatch')
            if not exact(before,state(trainer)):raise ContractError('Evaluation mutated training/optimizer/RNG')
            row['evaluation_state_invariance']='PASS';row['elapsed_seconds']=time.monotonic()-started
            dump(out/f'epoch{epoch:02d}-metrics.json',row);histories.append(row)
            print(json.dumps(dict(epoch=epoch,step=trainer.step,train_nll=row['splits']['train']['metrics']['category_nll'],
                val_nll=row['splits']['val']['metrics']['category_nll'],elapsed=round(time.monotonic()-started,1))),flush=True)
            if epoch==1:
                # One duplicate update verifies next-step recovery AND evaluation insertion invariance.
                restored=make_trainer(config,partitions,identity);restored.load(out/'epoch01.pt')
                reference=restored.update();physical+=1;expected=state(restored)
                trainer.load(out/'epoch01.pt')
                evaluate(trainer.model,partitions['val'][:2],ordinals['val'][:2],.1,guard.check)
                actual=trainer.update();physical+=1
                if actual!=reference or not exact(expected,state(trainer)):
                    raise ContractError('Exact recovery/evaluation insertion failed')
                actual['elapsed_seconds']=time.monotonic()-started
                trace_handle.write(json.dumps(actual,allow_nan=False)+'\n');trace_handle.flush()
                recovery=dict(status='PASS',checkpoint='epoch01.pt',next_step=264,
                    model_optimizer_rng_clocks='bitwise',evaluation_insertion='PASS',duplicate_updates=1)
                dump(out/'recovery.json',recovery)
                del restored,expected
        status='COMPLETE'
    except BudgetStop as exc:
        reason=str(exc)
    except Exception as exc:
        reason=type(exc).__name__+': '+str(exc)
        status='FAILED'
    finally:
        trace_handle.close()
        # A new path preserves the latest complete optimizer boundary even for a partial epoch.
        if status!='COMPLETE':
            # On a defect, keep the last verified checkpoint rather than serialize a possibly partial update.
            if status=='INCOMPLETE':trainer.save(out/f'interrupted-step{trainer.step:05d}.pt')
        if histories:
            selected=min(histories,key=lambda r:(r['splits']['val']['metrics']['category_nll'],r['epoch']))
            dump(out/'selection.json',dict(epoch=selected['epoch'],step=selected['step'],
                metric=selected['splits']['val']['metrics']['category_nll'],definition=CLAIM+'; validation prior NLL',
                checkpoint=f"epoch{selected['epoch']:02d}.pt",checkpoint_sha256=sha256_file(out/f"epoch{selected['epoch']:02d}.pt")))
        dump(out/'history.json',histories)
        dump(out/'execution.json',dict(status='TRAINING_COMPLETE_PENDING_REPLAY' if status=='COMPLETE' else status,reason=reason,retained_optimizer_updates=trainer.step,
            physical_optimizer_calls=physical,elapsed_seconds=time.monotonic()-started,
            peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            output_bytes=guard.bytes(),test_evaluated=False,gpu_initialized=torch.cuda.is_initialized()))
    if status=='COMPLETE':
        guard.check()
        selected_epoch=json.loads((out/'selection.json').read_text())['epoch']
        trainer.load(out/f'epoch{selected_epoch:02d}.pt')
        for split in partitions:
            p,_,_=evaluate(trainer.model,partitions[split],ordinals[split],min(trainer.cursor/526/10,1.),guard.check)
            with np.load(out/f'epoch{selected_epoch:02d}-{split}-predictions.npz') as artifact:
                if any(not np.array_equal(value,artifact[key]) for key,value in p.items()):
                    raise ContractError('Selected checkpoint replay failed')
        # Fresh isolated process has the same two-core affinity and resource ceilings.
        subprocess.run([sys.executable,'-I',str(root/'tools/m15_run.py'),'--replay',str(out),str(selected_epoch)],
            check=True,timeout=max(1,3480-(time.monotonic()-started)))
        dump(out/'completion.json',dict(status='PASS',claim=CLAIM,elapsed_seconds=time.monotonic()-started,
            selected_checkpoint_replay='bitwise',metric_replay='PASS',m1_preserved=True,recovery=recovery,
            total_new_bytes=guard.bytes(),max_retained_updates=5260,max_physical_calls=5261,
            cpu_seconds_including_children=sum(getattr(resource.getrusage(who),field)
                for who in (resource.RUSAGE_SELF,resource.RUSAGE_CHILDREN) for field in ('ru_utime','ru_stime')),
            peak_parent_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            peak_child_rss_MiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024))
    dump(out/'artifact_inventory.json',{str(p.relative_to(out)):sha256_file(p) for p in sorted(out.rglob('*')) if p.is_file()})
    if status!='COMPLETE':raise SystemExit(3)
