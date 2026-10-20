"""Prevent assigned cohort counts or source census counts becoming release evidence."""
from collections import Counter
from noah_baseline.m2.events import Event


class M2ReadinessError(ValueError):
    pass


def materialization_summary(records, splits, *, minimum_events):
    if minimum_events<1:raise ValueError('Explicit positive canonical target threshold required')
    counts=Counter();seen=set();categories=Counter();modalities=Counter()
    for row in records:
        event=Event(**row['event']).validate()
        if event.event_id in seen:raise M2ReadinessError('Duplicated event cannot count as additional targets')
        if splits.get(event.subject_id) not in {'train','val','test'}:raise M2ReadinessError('Event outside fixed patient split')
        seen.add(event.event_id);counts[event.subject_id]+=1;categories[event.category]+=1;modalities[event.modality]+=1
    return dict(minimum_events=minimum_events,canonical_events=len(seen),
        assigned_patients=dict(Counter(splits.values())),materialized_patients={s:sum(splits[p]==s for p in counts) for s in ['train','val','test']},
        qualified_patients={s:sum(splits[p]==s and n>=minimum_events for p,n in counts.items()) for s in ['train','val','test']},
        category_counts={str(k):categories[k] for k in range(31)},modality_counts=dict(modalities),
        counts_scope='actual canonical entries only; prefix, overlap and alternate recipe copies do not add targets')


def require_full_m2(records, splits, *, evidence, choices):
    summary=materialization_summary(records,splits,minimum_events=64)
    failures=[]
    if summary['qualified_patients']['train']<10000:
        failures.append('Fewer than10000 actual training patients with>=64 canonical clinical targets')
    for modality in ['categorical','numeric','text','cxr','ecg','echo']:
        if summary['modality_counts'].get(modality,0)==0:failures.append('No canonical events for required modality: '+modality)
    required=['complete_source_hashes','all31_source_adapter_evidence','cohort_wide_canonical_counts',
              'complete_content_bindings','shifted_window_ownership','genuine_six_modality_checks']
    failures.extend('Missing executed evidence: '+key for key in required if evidence.get(key) is not True)
    decisions=['ecg_production_position_policy','vital_aggregation_span_and_stride','full_note_policy',
               'untimed_diagnosis_and_triage_policy','rx_duplicate_source_policy','echo_production_annotation_policy']
    failures.extend('Unfrozen production choice: '+key for key in decisions if not choices.get(key))
    if failures:raise M2ReadinessError('; '.join(failures))
    return summary


def check_materialized_run(root, *, out, source, integration):
    import csv,json
    from pathlib import Path
    from noah_baseline.m2.dataset import read_shards
    from noah_baseline.m2.inventory import dump,sha
    out.mkdir(exist_ok=False)
    with (source/'patient_splits.csv').open(newline='') as f:splits={int(r['subject_id']):r['split'] for r in csv.DictReader(f)}
    config=json.loads((root/'configs/m2_integration.json').read_text())
    inventory=json.loads((source/'source_inventory.json').read_text())
    full_source_hashes=bool(inventory) and all(r.get('sha256') and sha(Path(r['path']))==r['sha256'] for r in inventory)
    # Category proof receipts must bind to this adapter implementation and actual artifacts.
    adapter_receipt=integration/'adapter_evidence.json'
    adapter_categories=set()
    if adapter_receipt.exists():
        for proof in json.loads(adapter_receipt.read_text()).get('checks',[]):
            artifact=integration/proof['artifact']
            if artifact.resolve().is_relative_to(integration.resolve()) and sha(artifact)==proof['sha256']:
                check=json.loads(artifact.read_text())
                if check.get('status')=='PASS' and check.get('evidence_kind')=='actual_source_adapter_check' and check.get('source_sha256')==sha(root/'packages/noah_baseline/m2/adapters.py'):
                    adapter_categories.add(int(check['category_id']))
    checks={};blocked=False
    for policy in config['ecg']['position_policies']:
        records=read_shards(root,integration/policy,splits)
        ownership=json.loads((integration/policy/'target_ownership.json').read_text())
        owners=ownership['owners'];ids={r['event']['event_id'] for r in records}
        evidence=dict(complete_source_hashes=bool(full_source_hashes),all31_source_adapter_evidence=adapter_categories==set(range(31)),
            cohort_wide_canonical_counts=len({r['event']['subject_id'] for r in records})==len(splits),
            complete_content_bindings=True,shifted_window_ownership=set(owners)==ids and all(v==1 for v in owners.values()),
            genuine_six_modality_checks=set(r['event']['modality'] for r in records)=={'categorical','numeric','text','cxr','ecg','echo'})
        summary=materialization_summary(records,splits,minimum_events=64)
        try:
            require_full_m2(records,splits,evidence=evidence,choices=config['production_choices'])
            status='PASS_FULL_M2'
        except M2ReadinessError as error:
            status='HOLD';blocked=True;summary['failures']=str(error).split('; ')
        checks[policy]=dict(status=status,summary=summary,executed_evidence=evidence)
    result=dict(status='HOLD' if blocked else 'PASS_FULL_M2',variants=checks,
        bounded_foundation_checks='PASS separately; cannot promote to full materialization or prospective validity')
    dump(out/'aggregate.json',result);print(json.dumps(result,indent=2))
    return not blocked
