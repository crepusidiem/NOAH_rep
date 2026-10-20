"""Inexpensive M1.5 fixtures; mathematical operations remain the M1 candidate."""
import copy
import json
import math
import random

import numpy as np
import pytest
import torch

from noah_baseline.evaluation.contracts import ContractError
from noah_baseline.m1.data import Event, build_windows, select_indexes
from noah_baseline.m1.training import Trainer
from noah_baseline.m15.metrics import targets, fit_baselines, baseline_predictions, score
from noah_baseline.m15.operations import LearningTrainer, evaluate, exact, state
from noah_baseline.m15.controller import Guard, BudgetStop


def fixture():
    index=select_indexes([dict(subject_id=1,hadm_id=10,admittime=100.,dischtime=200.,deathtime=None,
        hospital_expire_flag=0,discharge_location='HOME')],{1:'train'})[0]
    events=[Event(f'e{i}',1,10,'fixture',str(i),'fixture',100.+i,None,28 if i else 5,
        70.+i if i else None,'bpm' if i else None,'heartrate' if i else None) for i in range(6)]
    windows=build_windows(events,index,1950,length=16,overlap=2)
    config=dict(seed=42,model=dict(d_model=32,d_latent=16,n_layers=1,n_heads=2,head_dim=16,ffn_ratio=2,
        dropout=.1,attn_dropout=.1,drop_path=.1),accumulation=2,learning_rate=.0006,weight_decay=.01,
        gradient_clip=10.,lr_warmup_epochs=6,kl_warmup_epochs=10)
    identity=dict(train_subjects=[1],controller='fixture')
    return windows,config,identity


def test_update_equivalent_to_m1_and_cap(tmp_path):
    w,c,i=fixture()
    reference=Trainer(c,w,i); trace=reference.update(); expected=state(reference)
    actual=LearningTrainer(c,w,i,1); record=actual.update()
    assert {k:record[k] for k in trace}==trace
    assert exact(expected,state(actual))
    actual.save(tmp_path/'one.pt')
    restored=LearningTrainer(c,w,i,1);restored.load(tmp_path/'one.pt')
    assert exact(state(actual),state(restored))
    with pytest.raises(ContractError,match='limit'):restored.update()
    alien=copy.deepcopy(w);alien[0].subject_id=2
    with pytest.raises(ContractError,match='train-only'):LearningTrainer(c,alien,i,1)


def test_evaluation_rng_state_gradients_and_traversal():
    w,c,i=fixture();trainer=LearningTrainer(c,w,i,3);trainer.update()
    before=state(trainer);grad=[p.grad.clone() if p.grad is not None else None for p in trainer.model.parameters()]
    p,m,components=evaluate(trainer.model,w,[7],.1)
    assert exact(before,state(trainer))
    assert all(a is None and b.grad is None or a is not None and torch.equal(a,b.grad)
        for a,b in zip(grad,trainer.model.parameters()))
    repeated,again,_=evaluate(trainer.model,w,[7],.1)
    assert exact(p,repeated) and again==m
    other,_,_=evaluate(trainer.model,[w[0],w[0]],[99,7],.1)
    n=len(w[0].owned_target_ids)
    for key in p:np.testing.assert_array_equal(p[key],other[key][n:])
    assert components['fixed_beta1_total']==pytest.approx(.3*components['loss_rec_post']+components['loss_kl']+.1*components['loss_rec_prior'])
    assert all(branch+'_'+part in components for branch in ('prior','posterior')
        for part in ('category','modality','specifics_presence','specifics_vec','value','time_gate','time_reg'))


def test_train_baselines_and_independent_masks_metrics():
    w,_,_=fixture(); t=targets(w);base=fit_baselines(w)
    assert len(t['category'])==6 and t['gap_known'].sum()==5 and (~t['gap_known']).sum()==1
    assert base['category_counts'][5]==1 and base['category_counts'][28]==5
    assert base['unigram'][28]==6/37
    np.testing.assert_array_equal(base['transition'][0],base['unigram'])
    assert base['transition'][28,28]==pytest.approx((4+6/37)/5)
    pred=baseline_predictions(t,base);pred['gate']=np.full(6,.9)
    actual=score(t,pred)
    assert actual['category_nll']==pytest.approx(-(math.log(2/37)+5*math.log(6/37))/6)
    assert actual['category_accuracy']==5/6
    assert actual['positive_log1p_hours_rmse']==0
    assert actual['time_gate_log_loss']==pytest.approx(-math.log(.9))
    pred['log_gap'][-1]=999
    # A malformed prediction remains a hard failure, even on an unknown-gap row.
    with pytest.raises(ValueError,match='Nonfinite'):score(t,pred)


def test_guard_reserves_and_requested_stop(tmp_path):
    guard=Guard(tmp_path);guard.check();guard.request_stop()
    with pytest.raises(BudgetStop):guard.check()


def test_fit_excludes_validation_and_next_update_invariant(tmp_path):
    w,c,i=fixture();trainer=LearningTrainer(c,w,i,3);trainer.update();trainer.save(tmp_path/'start.pt')
    original=fit_baselines(w)
    validation=copy.deepcopy(w);validation[0].batch['category_ids'][:]=30
    trainer.load(tmp_path/'start.pt');expected_trace=trainer.update();expected=state(trainer)
    trainer.load(tmp_path/'start.pt')
    evaluate(trainer.model,validation,[23],.1)
    actual_trace=trainer.update()
    assert actual_trace==expected_trace and exact(state(trainer),expected)
    assert exact(original,fit_baselines(w))
