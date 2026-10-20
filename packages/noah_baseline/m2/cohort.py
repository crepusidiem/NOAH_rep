"""Outcome- and modality-blind sampling with frozen historical split memberships."""
import hashlib


def select_cohort(subjects, protected, sizes, seed):
    universe=set(int(s) for s in subjects)
    if set(protected)-universe or set(protected.values())-set(sizes):
        raise ValueError('Protected pilot membership is inconsistent with the admitted source universe')
    if len(universe)<sum(sizes.values()):raise ValueError('Insufficient distinct source patients')
    selected=dict(protected)
    ranked=sorted(universe-set(protected),key=lambda s:(hashlib.sha256(f'm2:{seed}:{s}'.encode()).digest(),s))
    cursor=0
    for split,n in sizes.items():
        need=n-sum(v==split for v in protected.values())
        if need<0:raise ValueError('Requested split smaller than protected pilot')
        for subject in ranked[cursor:cursor+need]:selected[subject]=split
        cursor+=need
    assert len(selected)==sum(sizes.values())
    assert all(selected[s]==split for s,split in protected.items())
    return selected
