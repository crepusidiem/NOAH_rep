"""Patient-cluster percentile intervals; training-seed variation is a separate estimand."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def aligned(records):
    result={}
    for r in records:
        key=(int(r['subject_id']),int(r['encounter_id']),str(r['cutoff']))
        if key in result:raise ValueError('Duplicate patient/encounter/landmark prediction')
        y=float(r['label']);p=float(r['probability'])
        if y not in (0.,1.) or not np.isfinite(p) or not 0<=p<=1:
            raise ValueError('Invalid binary prediction')
        result[key]=(y,p)
    if not result:raise ValueError('Empty predictions')
    keys=sorted(result)
    return keys,np.array([result[k][0] for k in keys]),np.array([result[k][1] for k in keys])


def metrics(y,p):
    return dict(auroc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        auprc=float(average_precision_score(y,p)) if np.any(y==1) else None,
        brier=float(brier_score_loss(y,p)))


def patient_cluster_bootstrap(records,*,n_resamples,seed,other=None,confidence=.95,min_valid_fraction=.8):
    if n_resamples<100 or not 0<confidence<1 or not 0<min_valid_fraction<=1:
        raise ValueError('Explicit bootstrap budget>=100 and valid confidence/retention thresholds required')
    keys,y,p=aligned(records)
    if other is not None:
        keys2,y2,q=aligned(other)
        if keys2!=keys or not np.array_equal(y,y2):raise ValueError('Paired prediction identities/labels must match exactly')
    patients=np.array([k[0] for k in keys]);clusters=np.unique(patients)
    if len(clusters)<2:raise ValueError('At least two patient clusters required')
    rows={s:np.flatnonzero(patients==s) for s in clusters}
    rng=np.random.Generator(np.random.PCG64(seed))
    samples={k:[] for k in metrics(y,p)}
    for _ in range(n_resamples):
        draw=rng.choice(clusters,size=len(clusters),replace=True)
        take=np.concatenate([rows[s] for s in draw])
        first=metrics(y[take],p[take]);second=metrics(y[take],q[take]) if other is not None else None
        for metric,v in first.items():
            if v is not None:samples[metric].append(v-second[metric] if second is not None else v)
    point=metrics(y,p);other_point=metrics(y,q) if other is not None else None
    result={}
    for key,values in samples.items():
        enough=len(values)>=max(100,int(np.ceil(n_resamples*min_valid_fraction)))
        point_estimate=point[key] if other_point is None or point[key] is None else point[key]-other_point[key]
        result[key]=dict(point=point_estimate,ci=np.quantile(values,[(1-confidence)/2,(1+confidence)/2],method='linear').tolist() if enough else None,
            valid_resamples=len(values),degenerate_resamples=n_resamples-len(values),status='OK' if enough else 'INSUFFICIENT_VALID_RESAMPLES')
    return dict(unit='patient; all aligned encounters/landmarks repeated together',method='percentile PCG64; no seed-variability pooling',
        paired_difference='first-minus-other' if other is not None else None,patients=len(clusters),predictions=len(keys),
        n_resamples=n_resamples,seed=seed,confidence=confidence,min_valid_fraction=min_valid_fraction,metrics=result)
