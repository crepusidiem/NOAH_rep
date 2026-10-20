"""Source-shaped adapters preserve zero, clocks, deferred records and identity."""
import pytest

from noah_baseline.data.schema import EventCategory as C
from noah_baseline.m2.adapters import Context,SOURCE_CATEGORIES,adapt_row


def adapt(table,row,context=None,dataset='hosp'):
    return adapt_row(dataset,table,dict(subject_id=1,hadm_id=10,**row),source_path='fixture.csv',
        source_row_id='row1',version='2.2',context=context or Context({10:1}))


def test_taxonomy_source_registry_has_all31_without_promoting_coverage():
    assert {int(c) for categories in SOURCE_CATEGORIES.values() for c in categories}==set(range(31))
    result=adapt('diagnoses_icd',{'icd_code':'FICTIONAL'})
    assert result['events']==[] and result['deferred'][0]['category']==int(C.DIAGNOSIS)
    assert 'no_precise_occurrence_clock' in result['deferred'][0]['reason']
    with pytest.raises(ValueError):adapt('diagnoses_icd',{},Context({10:1},untimed_clock_policy='guess_discharge'))


def test_observed_numeric_zero_units_and_source_storetime():
    context=Context({10:1},item_labels={('hosp','5'):'fictional lab'})
    result=adapt('labevents',dict(itemid=5,charttime='2020-01-01 01:00:00',storetime='2020-01-01 03:00:00',
        valuenum=0,value=None,valueuom='mg/dL'),context)
    record=result['events'][0]
    assert record['raw_numeric']==0 and record['event']['modality']=='numeric'
    assert record['event']['available_at']-record['event']['occurred_at']==2
    assert record['event']['units']=='mg/dL'
    bad=adapt('labevents',dict(itemid=5,charttime='2020-01-01',valuenum=0,valueuom='mg/dL'),context)
    assert bad['events']==[] and 'date_only_precision' in bad['deferred'][0]['reason']


def test_interval_total_uses_last_contributor_and_keeps_unknown_unknown():
    context=Context({10:1},item_labels={('icu','5'):'fictional fluid'})
    row=dict(itemid=5,starttime='2020-01-01 01:00:00',endtime='2020-01-01 04:00:00',
        storetime='2020-01-01 02:00:00',amount=100,amountuom='mL')
    result=adapt('inputevents',row,context,dataset='icu')
    assert len(result['events'])==2
    first,last=result['events']
    assert first['event']['available_at']==last['event']['occurred_at']
    assert first['observed_available_at']<first['event']['available_at']
    row['storetime']=None
    assert adapt('inputevents',row,context,dataset='icu')['events'][0]['event']['available_at'] is None
    row['storetime']='2020-01-01 02:00:00';row['endtime']='2020-01-01'
    coarse=adapt('inputevents',row,context,dataset='icu')
    assert coarse['events'][0]['event']['available_at'] is None and coarse['deferred']


def test_emar_not_given_is_not_administration_and_rx_sources_do_not_duplicate():
    row=dict(charttime='2020-01-01 01:00:00',event_txt='Not Given',medication='fictional drug')
    assert adapt('emar',row)['events']==[]
    row['event_txt']='Administered'
    assert adapt('emar',row)['events'][0]['event']['category']==int(C.DRUG_START)
    row['event_txt']='Stopped'
    assert adapt('emar',row)['events'][0]['event']['category']==int(C.DRUG_STOP)
    assert adapt('prescriptions',dict(starttime='2020-01-01 01:00:00',drug='fictional'))['events']==[]


def test_multiple_vital_fields_have_unique_identities_and_cross_patient_links_fail():
    context=Context({10:1},ed_units={'heartrate':'bpm','sbp':'mmHg'})
    result=adapt('vitalsign',dict(charttime='2020-01-01 01:00:00',heartrate=60,sbp=120),context,dataset='ed')
    assert len(result['events'])==2 and len({r['event']['event_id'] for r in result['events']})==2
    with pytest.raises(ValueError):adapt('admissions',dict(admittime='2020-01-01 01:00:00'),Context({10:2}))
