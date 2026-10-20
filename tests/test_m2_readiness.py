"""Actual targets, rather than assigned cohort counts, control M2 release gates."""
from dataclasses import asdict
import pytest

from noah_baseline.m2.events import Event
from noah_baseline.m2.readiness import materialization_summary,require_full_m2,M2ReadinessError


def rows():
    return [dict(event=asdict(Event(f'e{i}',1,10,'source','fixture','1',str(i),'fixture.csv',float(i),None,
        'unknown','categorical',22,None,None,'fixture'))) for i in range(70)]


def test_assigned_patients_and_source_counts_cannot_replace_qualified_targets():
    splits={i:'train' for i in range(1,10501)};splits[11000]='val';splits[12000]='test'
    result=materialization_summary(rows(),splits,minimum_events=64)
    assert result['assigned_patients']['train']==10500 and result['qualified_patients']['train']==1
    evidence={k:True for k in ['complete_source_hashes','all31_source_adapter_evidence','cohort_wide_canonical_counts',
        'complete_content_bindings','shifted_window_ownership','genuine_six_modality_checks']}
    with pytest.raises(M2ReadinessError,match='10000 actual'):
        require_full_m2(rows(),splits,evidence=evidence,choices={'forged_assigned_count':10500})


def test_duplicate_targets_and_unresolved_production_choices_fail_closed():
    with pytest.raises(M2ReadinessError,match='Duplicated'):
        materialization_summary(rows()+rows(),{1:'train'},minimum_events=64)
    with pytest.raises(M2ReadinessError,match='Unfrozen production choice'):
        require_full_m2(rows(),{1:'train'},evidence={},choices={})
