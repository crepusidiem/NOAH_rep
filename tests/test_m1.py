"""M1-01–08 independent fixtures for the bounded event-to-mortality path."""
from collections import Counter
from dataclasses import replace
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from noah_baseline.evaluation.contracts import ContractError
from noah_baseline.evaluation.landmarks import extract_landmark_state
from noah_baseline.m1.data import Event, extract_sources, select_indexes, cutoff_view, build_windows, landmark_request
from noah_baseline.m1.pipeline import fit_probe, metrics, execute
from noah_baseline.m1.training import Trainer, check_recovery
from noah_baseline.models.noah import NOAH
from noah_baseline.m1.verification import verify_source_cases, verify_real_objective


def admission(sid=1,hadm=10,start=100.,end=200.,death=None,label=0):
    return dict(subject_id=sid,hadm_id=hadm,admittime=start,dischtime=end,deathtime=death,
                hospital_expire_flag=label,discharge_location='DIED' if label else 'HOME')


def fixture_data():
    index=select_indexes([admission()],{1:'train'})[0]
    events=[Event(f'e{i}',1,10,'fixture',str(i),'fixture',100.+i,None if i%2 else 100.+i,28,
                  float(60+i),'bpm','heartrate') for i in range(40)]
    return index,events


def tiny_config():
    return dict(seed=7,model=dict(d_model=32,d_latent=16,n_layers=1,n_heads=2,head_dim=16,ffn_ratio=2,
        dropout=.1,attn_dropout=.1,drop_path=.1),learning_rate=.0006,weight_decay=.01,
        accumulation=2,gradient_clip=10.,lr_warmup_epochs=6,kl_warmup_epochs=10)


def csv_file(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def test_m1_01_source(tmp_path):
    hosp=tmp_path/'hosp';ed=tmp_path/'ed';cohort=tmp_path/'cohort.csv'
    csv_file(cohort,[dict(subject_id=1,split='train',anchor_year=2150,anchor_age=50),dict(subject_id=2,split='test',anchor_year=2150,anchor_age=50)])
    raw=[dict(subject_id=sid,hadm_id=hadm,admittime='2150-01-01 00:00:00',dischtime='2150-01-03 00:00:00',
              deathtime='2150-01-03 00:00:00' if sid==2 else '',hospital_expire_flag=int(sid==2),
              discharge_location='DIED' if sid==2 else 'HOME') for sid,hadm in [(1,10),(2,20)]]
    csv_file(hosp/'admissions.csv',raw)
    csv_file(hosp/'patients.csv',[dict(subject_id=s,anchor_year=2150,anchor_age=50) for s in (1,2)])
    csv_file(ed/'edstays.csv',[dict(subject_id=s,hadm_id=s*10,stay_id=s*100,intime='2149-12-31 23:00:00',outtime='2150-01-01 01:00:00') for s in (1,2)])
    vitals=[dict(subject_id=s,stay_id=s*100,charttime='2150-01-01 00:15:00',heartrate=70+s) for s in (1,2)]
    vitals.append(dict(vitals[0]))  # identical source records remain distinct physical rows
    csv_file(ed/'vitalsign.csv',vitals)
    a=extract_sources(hosp,ed,cohort);b=extract_sources(hosp,ed,cohort)
    assert a==b
    numeric=[e for e in a[1] if e.value is not None]
    assert len(numeric)==3 and len({e.event_id for e in numeric})==3
    assert all(e.units=='bpm' and e.available_at is None for e in numeric)
    assert {i['label'] for i in a[0]}=={0,1}
    assert verify_source_cases(a[0],a[1],a[3])['numeric_rows_compared']==3
    vitals[0]['subject_id']=2;csv_file(ed/'vitalsign.csv',vitals)
    with pytest.raises(ContractError,match='join mismatch'):extract_sources(hosp,ed,cohort)


def test_m1_02_cutoff_suffix_delayed_and_unknown():
    index,events=fixture_data();before=build_windows(events,index,1950,length=20,overlap=4)[-1]
    suffix=Event('future',1,10,'fixture','future','fixture',125.,130.,28,999.,'bpm','heartrate')
    after=build_windows(events+[suffix],index,1950,length=20,overlap=4)[-1]
    for key in before.batch:torch.testing.assert_close(before.batch[key],after.batch[key],rtol=0,atol=0)
    assert before.provenance==after.provenance
    model=NOAH(**tiny_config()['model']).eval()
    torch.testing.assert_close(extract_landmark_state(model,before.batch,before.evidence,landmark_request(index,before)),
                               extract_landmark_state(model,after.batch,after.evidence,landmark_request(index,after)),rtol=0,atol=0)
    delayed=Event('delayed',1,10,'fixture','delayed','fixture',110.,140.,28,300.,'bpm','heartrate')
    observed=build_windows(events,index,1950,length=20,overlap=4,view_policy='observed_available')[-1]
    observed_changed=build_windows(events+[delayed],index,1950,length=20,overlap=4,view_policy='observed_available')[-1]
    for key in observed.batch:assert torch.equal(observed.batch[key],observed_changed.batch[key])
    assert all(e.available_at is not None and e.available_at<=index['cutoff'] for e in cutoff_view(events,index,policy='observed_available'))
    assert any(e.available_at is None for e in cutoff_view(events,index))
    assert before.batch['delta_next'][0,len(before.evidence)-1]==0
    assert not before.batch['has_next_gap'][0,len(before.evidence)-1]
    assert [p['value'] for p in before.provenance[1:6]]==['Unknown']*5


def test_m1_03_patient_splits_and_index_before_eligibility(tmp_path):
    rows=[admission(end=110.),admission(hadm=11,start=200.,end=300.,death=290.,label=1)]
    result=select_indexes(rows,{1:'train'})
    assert result[0]['hadm_id']==10 and not result[0]['eligible']
    changed=copy.deepcopy(rows);changed[1]['hospital_expire_flag']=0
    assert select_indexes(changed,{1:'train'})[0]['hadm_id']==10
    index,events=fixture_data();windows=build_windows(events,index,1950,length=20,overlap=4)
    with pytest.raises(ContractError,match='train-only'):
        Trainer(tiny_config(),windows,{'train_subjects':[2]})
    with pytest.raises(ContractError,match='explicit patient split'):select_indexes(rows,{2:'test'})


def test_m1_04_shifted_ownership_and_padding():
    index,events=fixture_data();windows=build_windows(events,index,1950,length=20,overlap=4)
    expected=[e.event_id for e in events if e.occurred_at<=index['cutoff']]
    ledger=Counter(e for w in windows for e in w.owned_target_ids)
    assert ledger==Counter(expected)
    for w in windows:
        mask=w.batch['loss_mask'][0,1:] & w.batch['attention_mask'][0,1:] & w.batch['attention_mask'][0,:-1]
        independent=[w.evidence[t+1].event_id for t in range(len(mask)) if mask[t]]
        assert independent==w.owned_target_ids
        assert not w.batch['loss_mask'][0,:6].any()
        assert len(w.provenance)==len(w.evidence)
    model=NOAH(**tiny_config()['model']).eval();w=windows[-1]
    seen={}
    def capture(module,args):seen['history']=args[0].detach().clone()
    hook=model.prior.register_forward_pre_hook(capture)
    posterior={}
    ph=model.posterior.register_forward_pre_hook(lambda module,args:posterior.update(history=args[0].detach().clone(),target=args[1].detach().clone()))
    torch.manual_seed(1);loss,breakdown,aux=model(**w.batch);hook.remove();ph.remove()
    projected=model.projector(w.batch['category_ids'],w.batch['modality_ids'],w.batch['specifics_vec'],w.batch['has_specifics'],w.batch['value_vec'],w.batch['has_value'])
    torch.testing.assert_close(posterior['target'],projected[:,1:],rtol=0,atol=0)
    torch.testing.assert_close(posterior['history'],aux['h'][:,:-1],rtol=0,atol=0)
    effective=w.batch['loss_mask'][:,1:] & w.batch['attention_mask'][:,1:] & w.batch['attention_mask'][:,:-1]
    reference_kl=(aux['kl_per_token']*effective).sum()/effective.sum()
    torch.testing.assert_close(breakdown.loss_kl,reference_kl)
    torch.testing.assert_close(loss,.3*breakdown.loss_rec_post+reference_kl+.1*breakdown.loss_rec_prior)
    torch.testing.assert_close(seen['history'],aux['h'][:,:-1],rtol=0,atol=0)
    modified={k:v.clone() for k,v in w.batch.items()}
    pad=~modified['attention_mask'];modified['value_vec'][pad]=999.;modified['category_ids'][pad]=30
    torch.manual_seed(1);other,_,_=model(**modified)
    torch.testing.assert_close(loss,other,rtol=0,atol=0)


def test_m1_05_independent_weighted_reductions_and_accumulation():
    model=NOAH(**tiny_config()['model']);z=torch.zeros(1,3,16);dec=model.decoder(z)
    weights=torch.arange(1.,32.);model.category_class_weights=weights
    model.modality_class_weights=torch.arange(1.,7.);model.specifics_presence_class_weights=torch.tensor([2.,5.])
    mask=torch.tensor([[True,True,False]]);cats=torch.tensor([[0,3,30]]);mods=torch.ones(1,3,dtype=torch.long)
    values=torch.zeros(1,3,768);values[0,0,:44]=1.
    kw=dict(dec_out=dec,category_targets=cats,modality_targets=mods,specifics_targets=torch.zeros(1,3,768),
            has_specifics=torch.zeros(1,3,dtype=torch.bool),value_targets=values,has_value=mask,
            delta_targets=torch.tensor([[0.,2.,0.]]),has_next_gap=mask,loss_mask=mask)
    loss,parts=model._compute_reconstruction_loss(**kw)
    logp=dec.category_logits[mask].log_softmax(-1)
    ce=-(logp[0,0]*weights[0]+logp[1,3]*weights[3])/(weights[0]+weights[3])
    modlog=dec.modality_logits[mask].log_softmax(-1);modce=-modlog[:,1].mean()
    presence=(F.softplus(dec.specifics_presence_logits.squeeze(-1)[mask])*2).sum()/2
    mse=((dec.numeric_vec[mask]-values[mask][:,:44])**2).sum()/(2*44)
    gate=F.binary_cross_entropy_with_logits(dec.time_gate_logits.squeeze(-1)[mask],torch.tensor([0.,1.]))
    reg=(dec.log_gap_reg[0,1,0]-torch.log1p(torch.tensor(2.)))**2
    torch.testing.assert_close(loss,ce+modce+presence+mse+gate+reg)
    index,events=fixture_data();windows=build_windows(events,index,1950,length=20,overlap=4)
    # Existing objective averages per-microbatch component means. Verify that exact objective,
    # not an invented global-event normalization with different denominators.
    a=NOAH(**tiny_config()['model']).eval();b=copy.deepcopy(a)
    torch.manual_seed(17)
    terms=[a(**w.batch)[0] for w in windows]
    (sum(terms)/len(terms)).backward()
    torch.manual_seed(17)
    for w in windows:(b(**w.batch)[0]/len(windows)).backward()
    for x,y in zip(a.parameters(),b.parameters()):
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,rtol=2e-5,atol=1e-6)
    weighted=Trainer(tiny_config(),windows,{'train_subjects':[1]})
    assert verify_real_objective(weighted.model,windows[0],.25)['status']=='PASS'


def test_m1_06_checkpoint_rng_data_and_clock_recovery(tmp_path):
    index,events=fixture_data();windows=build_windows(events,index,1950,length=20,overlap=4)
    config=tiny_config();identity={'train_subjects':[1],'dataset':'fixture-v1'}
    restored,trace,result=check_recovery(config,windows,identity,tmp_path)
    assert result['status']=='PASS' and restored.step==2 and restored.cursor==len(windows)
    assert trace[0]['beta']==0 and trace[1]['beta']>0
    assert trace[0]['lr']!=trace[1]['lr'] and trace[1]['microbatches']==1  # short final group
    wrong=Trainer(config,windows,dict(identity,dataset='changed'))
    with pytest.raises(ContractError,match='identity mismatch'):wrong.load(tmp_path/'step1.pt')


def test_m1_07_encounter_mortality_exact_landmark_and_risk_set():
    rows=[admission(),admission(hadm=11,start=250.,end=350.,death=340.,label=1),
          admission(sid=2,hadm=20,death=150.,label=1),admission(sid=3,hadm=30,end=110.),
          admission(sid=4,hadm=40,death=120.,label=1)]
    indexes=select_indexes(rows,{1:'train',2:'test',3:'val',4:'val'})
    assert [(r['hadm_id'],r['label']) for r in indexes if r['eligible']]==[(10,0),(20,1)]
    index,events=fixture_data();window=build_windows(events,index,1950,length=20,overlap=4)[-1]
    model=NOAH(**tiny_config()['model']).eval();n=len(window.evidence);b=window.batch
    actual=extract_landmark_state(model,b,window.evidence,landmark_request(index,window))
    with torch.no_grad():
        e=model.projector(b['category_ids'][:,:n],b['modality_ids'][:,:n],b['specifics_vec'][:,:n],b['has_specifics'][:,:n],b['value_vec'][:,:n],b['has_value'][:,:n])
        h=model.backbone(e,b['positions'][:,:n],b['ages'][:,:n],b['delta_next'][:,:n],b['delta_prev'][:,:n],b['attention_mask'][:,:n])
    torch.testing.assert_close(actual,h[0,-1],rtol=0,atol=0)
    assert not torch.allclose(actual,h[0].mean(0))


def test_m1_08_probe_no_test_tuning_and_metric_replay():
    rng=np.random.default_rng(1);x=rng.normal(size=(36,4)).astype(np.float32);y=np.tile([0,1],18)
    split=np.array(['train']*20+['val']*8+['test']*8)
    config=dict(seed=42,probe_C_grid=[.01,.1,1.],probe_max_iter=2000)
    model,p,report=fit_probe(x,y,split,config)
    changed=x.copy();changed[split=='test']+=10000
    y2=y.copy();y2[split=='test']=1-y2[split=='test']
    other,_,second=fit_probe(changed,y2,split,config)
    assert report['selected_C']==second['selected_C']
    np.testing.assert_array_equal(model[0].mean_,other[0].mean_)
    np.testing.assert_array_equal(model[-1].coef_,other[-1].coef_)
    records=json.loads(json.dumps([dict(label=int(a),prediction=float(b)) for a,b in zip(y,p)]))
    assert metrics(y,p)==metrics([r['label'] for r in records],[r['prediction'] for r in records])
    with pytest.raises(ContractError,match='bounded'):
        execute(Path('.'),Path('.'),Path('.'),Path('unused'),'fixture',{'max_optimizer_steps':17},Path('unused'))
