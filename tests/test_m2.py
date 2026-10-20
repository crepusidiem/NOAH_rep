"""M2 sampling, provenance, cache and patient-cluster inference contracts."""
from dataclasses import replace
import copy
import json

import numpy as np
import pytest

from noah_baseline.m2.cohort import select_cohort
from noah_baseline.m2.events import Event,EmbeddingCache,cutoff_view,aggregate_minmax
from noah_baseline.evaluation.bootstrap import patient_cluster_bootstrap


def test_cohort_outcome_modality_blind_and_protected():
    old={1:'train',2:'val',3:'test'};sizes=dict(train=105,val=22,test=23)
    a=select_cohort(range(1,501),old,sizes,42)
    assert a==select_cohort(reversed(range(1,501)),old,sizes,42)
    assert len(a)==150 and all(a[k]==v for k,v in old.items())
    assert {s:sum(x==s for x in a.values()) for s in sizes}==sizes
    assert a!=select_cohort(range(1,501),old,sizes,43)
    with pytest.raises(ValueError):select_cohort(range(5,501),old,sizes,42)


def event():
    return Event('row:1',1,10,'source','fixture','1','row1','fixture.csv',1.,None,'unknown',
        'numeric',28,'heartrate','bpm','fixture-v1')


def test_event_unknown_future_and_encounter_identity():
    e=event().validate()
    assert cutoff_view([e],2,'occurrence_diagnostic')==[e]
    assert cutoff_view([e],2,'strict_prospective')==[]
    delayed=replace(e,available_at=3.,availability_source='storetime')
    assert cutoff_view([delayed],2,'strict_prospective')==[]
    assert cutoff_view([e,replace(e,event_id='future',occurred_at=5.)],2,'occurrence_diagnostic')==[e]
    with pytest.raises(ValueError):replace(e,available_at=1.).validate()
    with pytest.raises(ValueError):replace(e,embedding_ref='unverified.npy').validate()
    with pytest.raises(ValueError):replace(e,encounter_id=None).validate()


def test_cache_version_collision_tamper_and_no_failed_zero(tmp_path):
    c=EmbeddingCache(tmp_path)
    args=dict(kind='numeric',content_sha256='1'*64,dataset_sha256='2'*64,checkpoint_sha256='3'*64,recipe={'transform_revision':'v1'})
    ref=c.put(np.ones(44),**args);np.testing.assert_array_equal(c.get(ref),np.ones(44))
    assert c.put(np.ones(44),**args)==ref
    with pytest.raises(ValueError):c.put(np.ones(44)*2,**args)
    with pytest.raises(ValueError):c.put(np.zeros(44),**args)
    changed=copy.deepcopy(args);changed['recipe']['transform_revision']='v2'
    assert c.put(np.ones(44),**changed)['path']!=ref['path']
    from pathlib import Path
    Path(ref['path']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash'):c.get(ref)


def test_aggregation_contributors_units_and_availability():
    rows=[dict(subject_id=1,encounter_id=10,item='HR',units='bpm',source_row_id=str(i),
        occurred_at=t,available_at=t,value=v) for i,t,v in [(1,1.,60.),(2,3.,90.)]]
    a=aggregate_minmax(rows,origin=0,span=4,stride=4)[0]
    assert a['occurred_at']==2 and a['available_at']==3 and a['last_contributor_at']==3
    assert a['minimum']==60 and a['maximum']==90 and a['contributor_ids']==['1','2']
    early=[dict(r,available_at=0.) for r in rows]
    assert aggregate_minmax(early,origin=0,span=4,stride=4)[0]['available_at']==3
    rows[1]['available_at']=None
    assert aggregate_minmax(rows,origin=0,span=4,stride=4)[0]['available_at'] is None
    rows[1]['units']='unknown'
    with pytest.raises(ValueError):aggregate_minmax(rows,origin=0,span=4,stride=4)


def predictions():
    return [dict(subject_id=i//2+1,encounter_id=i+1,cutoff='24h',label=(i//2)%2,
        probability=.9 if (i//2)%2 else .1) for i in range(40)]


def test_cluster_bootstrap_replay_and_paired_alignment(tmp_path):
    data=predictions();result=patient_cluster_bootstrap(data,n_resamples=200,seed=42)
    assert result['patients']==20 and result['predictions']==40
    assert result['metrics']['auroc']['ci']==[1.,1.]
    assert result['metrics']['brier']['point']==pytest.approx(.01)
    assert result==patient_cluster_bootstrap(list(reversed(data)),n_resamples=200,seed=42)
    paired=patient_cluster_bootstrap(data,other=list(reversed(data)),n_resamples=200,seed=42)
    assert all(m['ci']==[0.,0.] for m in paired['metrics'].values())
    worse=[dict(r,probability=.8 if r['label'] else .2) for r in data]
    assert patient_cluster_bootstrap(data,other=worse,n_resamples=200,seed=42)['metrics']['brier']['point']==pytest.approx(-.03)
    artifact=tmp_path/'predictions.json';artifact.write_text(json.dumps(data))
    assert patient_cluster_bootstrap(json.loads(artifact.read_text()),n_resamples=200,seed=42)==result
    with pytest.raises(ValueError,match='match'):patient_cluster_bootstrap(data,other=worse[:-1],n_resamples=200,seed=42)


def test_cluster_degenerate_classes_are_not_fabricated():
    data=[dict(r,label=0) for r in predictions()]
    result=patient_cluster_bootstrap(data,n_resamples=200,seed=7)
    assert result['metrics']['auroc']['point'] is None and result['metrics']['auroc']['ci'] is None
    assert result['metrics']['auprc']['valid_resamples']==0
    assert result['metrics']['brier']['valid_resamples']==200


def test_cluster_and_paired_brier_against_independent_resampling():
    data=predictions()
    data=[dict(r,probability=.05+.9*i/39) for i,r in enumerate(data)]
    other=[dict(r,probability=.3 if r['label']==0 else .7) for r in data]
    rng=np.random.Generator(np.random.PCG64(17))
    errors=np.array([(r['probability']-r['label'])**2 for r in data]).reshape(20,2).mean(1)
    other_errors=np.array([(r['probability']-r['label'])**2 for r in other]).reshape(20,2).mean(1)
    reference=[];difference=[]
    for _ in range(200):
        selected=rng.choice(np.arange(20),size=20,replace=True)
        reference.append(errors[selected].mean());difference.append((errors[selected]-other_errors[selected]).mean())
    actual=patient_cluster_bootstrap(data,n_resamples=200,seed=17)['metrics']['brier']
    paired=patient_cluster_bootstrap(data,other=other,n_resamples=200,seed=17)['metrics']['brier']
    np.testing.assert_allclose(actual['ci'],np.quantile(reference,[.025,.975]),atol=1e-14,rtol=0)
    np.testing.assert_allclose(paired['ci'],np.quantile(difference,[.025,.975]),atol=1e-14,rtol=0)
