"""Independent real-case checks for M1; no new scientific endpoint or training method."""
import csv
import math

import duckdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from noah_baseline.evaluation.contracts import ContractError


def verify_source_cases(indexes,events,sources):
    c=duckdb.connect(':memory:');c.execute('SET threads=2')
    # SQL's index selection is independent of the Python minimum loop.
    c.execute('CREATE TABLE cohort AS SELECT * FROM read_csv_auto(?)',[sources['cohort']['path']])
    reference=c.execute('''WITH ordered AS (
        SELECT a.*,row_number() OVER(PARTITION BY a.subject_id ORDER BY a.admittime,a.hadm_id) rank
        FROM read_csv_auto(?) a JOIN cohort USING(subject_id))
        SELECT subject_id,hadm_id,hospital_expire_flag,epoch(admittime)/3600.0 admit,
        dischtime>admittime+INTERVAL '24 hours'
        AND (deathtime IS NULL OR deathtime>admittime+INTERVAL '24 hours')
        AND NOT(hospital_expire_flag=1 AND deathtime IS NULL)
        AND NOT(hospital_expire_flag=0 AND(deathtime IS NOT NULL OR coalesce(discharge_location,'')='DIED'))
        AND (deathtime IS NULL OR deathtime<=dischtime) eligible
        FROM ordered WHERE rank=1''',[sources['admissions']['path']]).fetchall()
    expected={int(s):(int(h),int(y),float(a),bool(e)) for s,h,y,a,e in reference}
    for index in indexes:
        actual=(index['hadm_id'],index['label'],index['admittime'],index['eligible'])
        if actual!=expected[index['subject_id']]:raise ContractError('Independent source admission/eligibility query disagrees')
    # Check source CSV row identity/value independently for the first and last numeric
    # observation of each split/outcome's eligible index, without selecting an evaluation cohort.
    numeric={}
    for event in events:
        if event.value is not None:numeric.setdefault(event.encounter_id,[]).append(event)
    choices=[]
    for key in sorted({(i['split'],i['label']) for i in indexes if i['eligible']}):
        match=next((i for i in indexes if i['eligible'] and (i['split'],i['label'])==key and i['hadm_id'] in numeric),None)
        if match:
            group=numeric[match['hadm_id']];choices.extend([group[0],group[-1]])
    selected={int(e.source_row_id.split(':')[0].removeprefix('csv_record_')):e for e in choices}
    matched=0
    with open(sources['vitalsign']['path'],newline='') as stream:
        for ordinal,row in enumerate(csv.DictReader(stream),start=1):
            if ordinal not in selected:continue
            event=selected[ordinal]
            if int(row['subject_id'])!=event.subject_id or float(row['heartrate'])!=event.value:
                raise ContractError('Physical source-row parity failed')
            if event.available_at is not None or event.units!='bpm':raise ContractError('Source availability/units changed')
            from noah_baseline.m1.data import hours
            if hours(row['charttime'])!=event.occurred_at:raise ContractError('Source timestamp parity failed')
            matched+=1
    if matched!=len(selected) or not matched:raise ContractError('No complete real numeric source checks')
    c.close()
    return dict(indexes_compared=len(indexes),independent_admission_sql='PASS',numeric_rows_compared=matched,
                physical_csv_row_parity='PASS',unknown_availability_preserved=True)


def verify_real_objective(model,window,beta):
    predictions=[];history=[];posterior=[]
    hooks=[model.decoder.register_forward_hook(lambda m,a,o:predictions.append(o)),
           model.prior.register_forward_pre_hook(lambda m,a:history.append(a[0].detach().clone())),
           model.posterior.register_forward_pre_hook(lambda m,a:posterior.append((a[0].detach().clone(),a[1].detach().clone())))]
    model.eval()
    try:
        with torch.no_grad():actual,_,aux=model(**window.batch,beta=beta)
    finally:
        for hook in hooks:hook.remove()
    b=window.batch
    mask=b['loss_mask'][:,1:] & b['attention_mask'][:,1:] & b['attention_mask'][:,:-1]
    cat=b['category_ids'][:,1:];mod=b['modality_ids'][:,1:]
    if b['has_specifics'][:,1:][mask].any() or not set(mod[mask].tolist())<={0,1}:
        raise ContractError('Objective oracle only permits declared M1 no-specifics categorical/numeric scope')
    def weighted_ce(logits,target,weights):
        chosen=target[mask]
        nll=-logits[mask].log_softmax(-1).gather(1,chosen[:,None]).squeeze(1)
        weight=weights[chosen]
        return (nll*weight).sum()/weight.sum()
    def reconstruction(pred):
        category=weighted_ce(pred.category_logits,cat,model.category_class_weights)
        modality=weighted_ce(pred.modality_logits,mod,model.modality_class_weights)
        # All specifics are absent; element-mean weighted BCE is NOT weight-sum-normalized.
        presence=F.softplus(pred.specifics_presence_logits.squeeze(-1)[mask]).sum()*model.specifics_presence_class_weights[0]/mask.sum()
        numeric=mask & b['has_value'][:,1:] & mod.eq(1)
        value=((pred.numeric_vec[numeric]-b['value_vec'][:,1:][numeric][:,:44])**2).mean() if numeric.any() else 0.
        tm=mask & b['has_next_gap'][:,1:];gap=b['delta_next'][:,1:]
        logits=pred.time_gate_logits.squeeze(-1)
        gate=(F.softplus(logits[tm])-gap[tm].gt(0)*logits[tm]).mean() if tm.any() else 0.
        pos=tm & gap.gt(0)
        regression=((pred.log_gap_reg.squeeze(-1)[pos]-torch.log1p(gap[pos]))**2).mean() if pos.any() else 0.
        return category+modality+presence+value+gate+regression
    kl=kl_divergence(Normal(aux['mu_q'],aux['sigma_q']),Normal(aux['mu_p'],aux['sigma_p'])).sum(-1)
    kl=(kl*mask).sum()/mask.sum()
    reference=.3*reconstruction(predictions[0])+beta*kl+.1*reconstruction(predictions[1])
    torch.testing.assert_close(actual,reference,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(history[0],aux['h'][:,:-1],rtol=0,atol=0)
    with torch.no_grad():
        e=model.projector(b['category_ids'],b['modality_ids'],b['specifics_vec'],b['has_specifics'],b['value_vec'],b['has_value'])
    torch.testing.assert_close(posterior[0][0],aux['h'][:,:-1],rtol=0,atol=0)
    torch.testing.assert_close(posterior[0][1],e[:,1:],rtol=0,atol=0)
    return dict(status='PASS',actual=float(actual),independent_reference=float(reference),
                absolute_error=float(abs(actual-reference)),owned_shifted_targets=int(mask.sum()),
                gaussian_reference='torch.distributions.Normal KL',alignment='prior h[:-1], posterior target e[1:]')
