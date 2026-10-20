"""Source-row adapters with explicit untimed/unresolved records, never guessed clocks.

Mappings are ASSUMED candidates from MIMIC headers and the paper taxonomy, not
NOAH author ETL. Source availability stays unknown unless a time field exists.
Full extraction must freeze item dictionaries, clock and duplicate-source rules.
"""
from dataclasses import asdict, dataclass, field
import math

from noah_baseline.data.schema import EventCategory as C
from noah_baseline.m1.data import hours
from noah_baseline.m2.events import Event


@dataclass
class Context:
    admission_subjects: dict
    ed_stays: dict = field(default_factory=dict)  # stay_id -> (subject_id,hadm_id orNone)
    item_labels: dict = field(default_factory=dict)  # (dataset,itemid) -> actual dictionary label
    chart_categories: dict = field(default_factory=dict)  # explicit itemid -> Vitals/Other Event
    ed_units: dict = field(default_factory=dict)  # explicit source-field units, never guessed
    rx_primary_source: str | None = None
    untimed_clock_policy: str = 'retain_untimed'


# Coverage declarations are not completed source verification or event frequencies.
SOURCE_CATEGORIES={
    ('hosp','admissions'):[C.ENTER_HOSPITALIZATION,C.LEAVE_HOSPITALIZATION,C.DEATH],
    ('ed','edstays'):[C.ENTER_ED,C.LEAVE_ED],('icu','icustays'):[C.ENTER_ICU,C.LEAVE_ICU],
    ('hosp','transfers'):[C.TRANSFER,C.LEAVE_TRANSFER],('hosp','services'):[C.SERVICE],
    ('hosp','labevents'):[C.LAB_TEST],('hosp','microbiologyevents'):[C.MICROBIOLOGY_TEST],
    ('icu','chartevents'):[C.VITALS,C.OTHER_EVENT],('ed','vitalsign'):[C.VITALS],
    ('icu','inputevents'):[C.BODY_INPUT,C.BODY_INPUT_END],('icu','outputevents'):[C.BODY_OUTPUT],
    ('icu','procedureevents'):[C.PROCEDURE,C.PROCEDURE_END],('hosp','procedures_icd'):[C.PROCEDURE],
    ('hosp','diagnoses_icd'):[C.DIAGNOSIS],('ed','diagnosis'):[C.DIAGNOSIS],
    ('ed','triage'):[C.COMPLAINT,C.TRIAGE_ACUITY],('ed','medrecon'):[C.DRUG_PREVIOUSLY],
    ('ed','pyxis'):[C.DRUG_START],('hosp','emar'):[C.DRUG_START,C.DRUG_STOP],
    ('hosp','prescriptions'):[C.DRUG_PRESCRIPTION],('hosp','pharmacy'):[C.DRUG_PRESCRIPTION],
    ('note','discharge'):[C.NOTE_DISCHARGESUMMARY],('note','radiology'):[C.NOTE_RADIOLOGYREPORT],
    ('cxr','cxr-record-list'):[C.X_RAY],('ecg','record_list'):[C.ECG],('echo','echo-record-list'):[C.ECHO],
}


def adapt_row(dataset, table, row, *, source_path, source_row_id, version, context):
    if (dataset,table) not in SOURCE_CATEGORIES:
        raise ValueError('Unregistered source table')
    if context.untimed_clock_policy!='retain_untimed':
        raise ValueError('Alternative ICD/triage clocks require an approved implemented protocol')
    sid=int(row['subject_id']);hadm=int(row['hadm_id']) if row.get('hadm_id') not in (None,'') else None
    if hadm is None and dataset=='ed' and row.get('stay_id') not in (None,''):
        linked=context.ed_stays.get(int(row['stay_id']))
        if linked is not None:
            if linked[0]!=sid:raise ValueError('ED source join crosses patients')
            hadm=linked[1]
    if hadm is not None and context.admission_subjects.get(hadm)!=sid:
        raise ValueError('Source hospital identity crosses patients or is unknown')
    output=[];deferred=[]
    def defer(reason, category=None):
        deferred.append(dict(dataset=dataset,table=table,source_path=source_path,source_row_id=str(source_row_id),
            subject_id=sid,encounter_id=hadm,category=None if category is None else int(category),
            reason=reason,source_row=dict(row)))
    def emit(category, clock, *, item=None, value=None, units=None, raw_text=None, semantic='source_observed',available_floor=None):
        stamp=row.get(clock)
        if stamp in (None,''):
            defer('missing_occurrence_timestamp',category);return
        if len(str(stamp))==10:
            defer('date_only_precision_no_midnight_proxy',category);return
        occurred=hours(stamp)
        availability=row.get('storetime')
        available=None if availability in (None,'') or len(str(availability))==10 else hours(availability)
        observed=available
        if available is not None and available_floor is not None:available=max(available,available_floor)
        if semantic.startswith('interval_total') and available_floor is None:available=None
        availability_source='unknown' if available is None else 'storetime' if available==observed else 'derived_storetime_and_final_contributor'
        modality='categorical';numeric=None
        if value not in (None,''):
            try:numeric=float(value)
            except (TypeError,ValueError):
                defer('nonnumeric_payload_requires_explicit_encoding',category);return
            if not math.isfinite(numeric) or not units:
                defer('nonfinite_numeric_or_unknown_units',category);return
            modality='numeric'
        if raw_text is not None:
            if not str(raw_text).strip():defer('empty_text_is_missing',category);return
            modality='text'
        slot=str(row.get('itemid',item or ''))
        identity=f'{dataset}.{table}:{source_row_id}:{int(category)}:{clock}:{slot}'
        event=Event(identity,sid,hadm,'source' if hadm is not None else 'unlinked',dataset,version,
            str(source_row_id)+':'+clock+':'+slot,source_path,occurred,available,availability_source,modality,int(category),
            item,units,'m2-source-adapters-v1').validate()
        output.append(dict(event=asdict(event),raw_numeric=numeric,raw_text=raw_text,observed_available_at=observed,
            timestamp_field=clock,timestamp_semantics=semantic,source_field_identity=str(source_row_id),
            encoding_required=modality!='categorical' or item is not None))
    def label():
        return context.item_labels.get((dataset,str(row.get('itemid'))))
    def scalar():
        return row['valuenum'] if row.get('valuenum') not in (None,'') else row.get('value')
    if table in {'admissions','edstays','icustays'}:
        fields={'admissions':[('admittime',C.ENTER_HOSPITALIZATION),('dischtime',C.LEAVE_HOSPITALIZATION),('deathtime',C.DEATH)],
                'edstays':[('intime',C.ENTER_ED),('outtime',C.LEAVE_ED)],'icustays':[('intime',C.ENTER_ICU),('outtime',C.LEAVE_ICU)]}
        for clock,category in fields[table]:
            if row.get(clock) not in (None,''):emit(category,clock)
    elif table=='transfers':
        emit(C.TRANSFER,'intime',item=row.get('careunit'))
        if row.get('outtime'):emit(C.LEAVE_TRANSFER,'outtime',item=row.get('careunit'))
    elif table=='services':emit(C.SERVICE,'transfertime',item=row.get('curr_service'))
    elif table in {'labevents','chartevents','outputevents','inputevents','procedureevents'}:
        item=label()
        if not item:defer('specific_item_dictionary_label_required')
        elif table=='labevents':emit(C.LAB_TEST,'charttime',item=item,value=scalar(),units=row.get('valueuom'))
        elif table=='chartevents':
            category=context.chart_categories.get(str(row['itemid']))
            if category not in (C.VITALS,C.OTHER_EVENT):defer('explicit_chart_item_category_required')
            else:emit(category,'charttime',item=item,value=scalar(),units=row.get('valueuom'))
        elif table=='outputevents':emit(C.BODY_OUTPUT,'charttime',item=item,value=row.get('value'),units=row.get('valueuom'))
        elif table=='procedureevents':
            emit(C.PROCEDURE,'starttime',item=item)
            if row.get('endtime'):emit(C.PROCEDURE_END,'endtime',item=item)
        else:
            end=hours(row['endtime']) if row.get('endtime') and len(str(row['endtime']))!=10 else None
            emit(C.BODY_INPUT,'starttime',item=item,value=row.get('amount'),units=row.get('amountuom'),
                semantic='interval_total; cannot be available before final contributor',available_floor=end)
            if row.get('endtime'):emit(C.BODY_INPUT_END,'endtime',item=item)
    elif table=='microbiologyevents':
        emit(C.MICROBIOLOGY_TEST,'charttime',item=row.get('test_name'),raw_text=row.get('interpretation') or row.get('org_name'))
    elif table in {'diagnoses_icd','diagnosis','procedures_icd','triage'}:
        for category in SOURCE_CATEGORIES[dataset,table]:defer('source_has_no_precise_occurrence_clock; retain_untimed',category)
    elif table in {'prescriptions','pharmacy'}:
        if context.rx_primary_source!=table:defer('rx_primary_source_or_duplicate_reconciliation_pending',C.DRUG_PRESCRIPTION)
        else:emit(C.DRUG_PRESCRIPTION,'starttime',item=row.get('drug') or row.get('medication'),semantic='scheduled_start_not_observed_order_time')
    elif table=='emar':
        kind=str(row.get('event_txt','')).strip().lower()
        if kind in {'administered','started','restarted'}:emit(C.DRUG_START,'charttime',item=row.get('medication'))
        elif kind in {'stopped','stop'}:emit(C.DRUG_STOP,'charttime',item=row.get('medication'))
        else:defer('emar_action_not_a_verified_start_or_stop')
    elif table in {'medrecon','pyxis'}:
        emit(C.DRUG_PREVIOUSLY if table=='medrecon' else C.DRUG_START,'charttime',item=row.get('name'),
            semantic='medication_reconciliation' if table=='medrecon' else 'dispensing_not_verified_administration')
    elif dataset=='ed' and table=='vitalsign':
        for variable,units in context.ed_units.items():
            if row.get(variable) not in (None,''):emit(C.VITALS,'charttime',item=variable,value=row[variable],units=units)
    elif dataset=='note':emit(SOURCE_CATEGORIES[dataset,table][0],'charttime',item=row.get('note_type'),raw_text=row.get('text'))
    else:
        defer('media_uses_verified_modality_specific_source_adapter')
    return dict(events=output,deferred=deferred,classification='ASSUMED source mapping; not author ETL')
