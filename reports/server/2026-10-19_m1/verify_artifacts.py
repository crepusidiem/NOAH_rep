"""Replay M1 saved evidence and stress real-source window ownership; no training."""
from collections import Counter
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys

if not sys.flags.isolated:
    raise SystemExit('Use python -I')
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'packages'))
from noah_baseline.runtime import install_source_boundary, ArtifactNamespace
install_source_boundary(root)
from noah_baseline.evaluation.contracts import ContractError, sha256_file, validate_release
from noah_baseline.m1.data import Event, build_windows
from noah_baseline.m1.pipeline import metrics


def main():
    if len(sys.argv)!=3:raise SystemExit('verify_artifacts.py <private run directory> <new private evidence file>')
    ns=ArtifactNamespace(root,'reduced_pilot');run=ns.owned(Path(sys.argv[1]).resolve());output=ns.owned(Path(sys.argv[2]).resolve())
    if output.exists():raise ContractError('Evidence overwrite forbidden')
    inventory=json.loads((run/'artifact_inventory.json').read_text())
    for name,digest in inventory.items():
        if sha256_file(run/name)!=digest:raise ContractError('Artifact checksum mismatch:'+name)
    manifest=json.loads((run/'experiment_manifest.json').read_text())
    validate_release(manifest,'retrospective',base=run)
    rejected=[]
    for claim in ('prospective','paper_faithful'):
        try:validate_release(manifest,claim,base=run)
        except ContractError:rejected.append(claim)
        else:raise ContractError('Unsupported claim released')
    records=[json.loads(line) for line in (run/'predictions.jsonl').read_text().splitlines()]
    stored=json.loads((run/'validated_metrics.json').read_text())['metrics']
    for split in ('train','val','test'):
        subset=[r for r in records if r['split']==split]
        if metrics([r['label'] for r in subset],[r['prediction'] for r in subset])!=stored[split]:
            raise ContractError('Independent saved prediction replay differs')
    events=[Event(**json.loads(line)) for line in (run/'events.jsonl').read_text().splitlines()]
    indexes=[i for i in json.loads((run/'index_cohort.json').read_text()) if i['eligible']]
    with (run/'patient_splits.csv').open() as f:birth={int(r['subject_id']):int(r['anchor_year'])-int(r['anchor_age']) for r in csv.DictReader(f)}
    stress=[]
    for outcome in (0,1):
        candidates=[i for i in indexes if i['label']==outcome]
        index=max(candidates,key=lambda i:sum(e.encounter_id==i['hadm_id'] and e.occurred_at<=i['cutoff'] for e in events))
        # This alternate geometry tests ownership only. It is never used to train or produce metrics.
        windows=build_windows(events,index,birth[index['subject_id']],length=12,overlap=2)
        expected=Counter(e.event_id for e in events if e.subject_id==index['subject_id'] and e.encounter_id==index['hadm_id'] and e.occurred_at<=index['cutoff'])
        actual=Counter(e for w in windows for e in w.owned_target_ids)
        if actual!=expected:raise ContractError('Real stress target ownership changed')
        stress.append(dict(outcome=outcome,windows=len(windows),unique_owned_targets=len(expected),status='PASS'))
    if max(r['windows'] for r in stress)<2:raise ContractError('No real overlap case exercised')
    result=dict(status='PASS',artifact_files_verified=len(inventory),metric_replay='exact',
        eligible_predictions=len(records),unsupported_claims_rejected=rejected,
        real_window_ownership_stress=stress,stress_scope='ownership test only; original context64 training/metrics unchanged',
        manifest_sha256=sha256_file(run/'experiment_manifest.json'),recipe_sha256=sha256_file(Path(__file__)))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
