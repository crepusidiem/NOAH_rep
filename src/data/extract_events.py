"""
Event Extractor for Pilot Cohort.

Extracts events across MIMIC-IV (hosp, icu), MIMIC-IV-ED, MIMIC-IV-Note,
MIMIC-IV-ECG, and MIMIC-CXR for patients in the cohort manifest.
Constructs chronological lifelong patient event streams with directional deltas.
"""

import csv
import datetime
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.data.schema import (
    EventCategory,
    Modality,
    DemographicPrefix,
    EventRecord,
    PatientJourney,
)
from src.data.vitals import RawVitalMeasurement, aggregate_vitals_sliding_window
from src.data.numeric import DyadicFourierEncoder


MIMIC_BASE = Path("/home/wey371/rajpurkarlab/datasets")
HOSP_DIR = MIMIC_BASE / "mimic-iv/physionet.org/files/mimiciv/2.2/hosp"
ICU_DIR = MIMIC_BASE / "mimic-iv/physionet.org/files/mimiciv/2.2/icu"
ED_DIR = MIMIC_BASE / "mimic-iv-ed/physionet.org/files/mimic-iv-ed/2.2/ed"
NOTE_DIR = MIMIC_BASE / "mimic-iv-note/physionet.org/files/mimic-iv-note/2.2/note"
ECG_DIR = MIMIC_BASE / "mimic-iv-ecg/1.0"
CXR_DIR = MIMIC_BASE / "mimic-cxr-resized/2.0.0"


def parse_datetime(dt_str: str) -> Optional[datetime.datetime]:
    """Parse standard datetime string 'YYYY-MM-DD HH:MM:SS'."""
    if not dt_str:
        return None
    try:
        # Trim fractional seconds if any
        dt_str = dt_str.split(".")[0]
        return datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            return datetime.datetime.strptime(dt_str, "%Y-%m-%d")
        except ValueError:
            return None


def parse_cxr_datetime(study_date: str, study_time: str) -> Optional[datetime.datetime]:
    """Parse CXR StudyDate (YYYYMMDD) and StudyTime (HHMMSS.frac)."""
    if not study_date:
        return None
    try:
        y = int(study_date[:4])
        m = int(study_date[4:6])
        d = int(study_date[6:8])
        hh, mm, ss = 0, 0, 0
        if study_time:
            time_part = study_time.split(".")[0].zfill(6)
            hh = int(time_part[:2])
            mm = int(time_part[2:4])
            ss = int(time_part[4:6])
        return datetime.datetime(y, m, d, hh, mm, ss)
    except Exception:
        return None


class CohortEventExtractor:
    """Extracts and aligns multimodal events for target cohort subjects."""

    def __init__(self, cohort_manifest_path: Path):
        self.manifest_path = cohort_manifest_path
        self.patients: Dict[int, Dict[str, Any]] = {}
        self.fourier_encoder = DyadicFourierEncoder(kappa=1.0)
        self._load_manifest()

    def _load_manifest(self):
        with open(self.manifest_path, "rt") as f:
            reader = csv.DictReader(f)
            for r in reader:
                sid = int(r["subject_id"])
                anchor_age = int(r["anchor_age"]) if r["anchor_age"].isdigit() else 0
                anchor_year = int(r["anchor_year"]) if r["anchor_year"].isdigit() else 2100
                birth_year = anchor_year - anchor_age
                origin_dt = datetime.datetime(birth_year, 1, 1, 0, 0, 0)
                self.patients[sid] = {
                    "split": r["split"],
                    "gender": r["gender"],
                    "anchor_age": anchor_age,
                    "anchor_year": anchor_year,
                    "birth_year": birth_year,
                    "origin_dt": origin_dt,
                    "dod": r["dod"],
                }
        print(f"Loaded {len(self.patients):,} patients from manifest.")

    def dt_to_hours(self, sid: int, dt: datetime.datetime) -> float:
        """Calculate elapsed hours from patient's birth origin to event."""
        origin = self.patients[sid]["origin_dt"]
        delta = dt - origin
        return delta.total_seconds() / 3600.0

    def extract_admissions_and_demographics(self, target_sids: Set[int]) -> Dict[int, Tuple[DemographicPrefix, List[EventRecord]]]:
        results: Dict[int, Tuple[DemographicPrefix, List[EventRecord]]] = {}
        adm_path = HOSP_DIR / "admissions.csv"
        
        # Temporary store for latest attributes per subject
        demog_data: Dict[int, Dict[str, str]] = {}
        events_by_sub: Dict[int, List[EventRecord]] = {sid: [] for sid in target_sids}

        with open(adm_path, "rt", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = int(row["subject_id"])
                if sid not in target_sids:
                    continue
                
                hadm_id = int(row["hadm_id"]) if row["hadm_id"] else None
                # Update demographic attributes
                demog_data[sid] = {
                    "race": row.get("race", "Unknown"),
                    "insurance": row.get("insurance", "Unknown"),
                    "language": row.get("language", "Unknown"),
                    "marital_status": row.get("marital_status", "Unknown"),
                }

                # Enter Hospitalization
                admittime = parse_datetime(row["admittime"])
                if admittime:
                    hr = self.dt_to_hours(sid, admittime)
                    events_by_sub[sid].append(EventRecord(
                        subject_id=sid,
                        hadm_id=hadm_id,
                        source_table="admissions",
                        source_row_id=f"adm_{hadm_id}_enter",
                        charttime=hr,
                        storetime=hr,
                        category_id=EventCategory.ENTER_HOSPITALIZATION,
                        modality_id=Modality.CATEGORICAL,
                        has_specifics=True,
                        specifics_text=f"{row.get('admission_type', '')} {row.get('admission_location', '')}".strip(),
                    ))

                # Leave Hospitalization
                dischtime = parse_datetime(row["dischtime"])
                if dischtime:
                    hr = self.dt_to_hours(sid, dischtime)
                    events_by_sub[sid].append(EventRecord(
                        subject_id=sid,
                        hadm_id=hadm_id,
                        source_table="admissions",
                        source_row_id=f"adm_{hadm_id}_leave",
                        charttime=hr,
                        storetime=hr,
                        category_id=EventCategory.LEAVE_HOSPITALIZATION,
                        modality_id=Modality.CATEGORICAL,
                        has_specifics=True,
                        specifics_text=row.get("discharge_location", "Unknown"),
                    ))

                # Death
                deathtime = parse_datetime(row.get("deathtime", ""))
                if deathtime:
                    hr = self.dt_to_hours(sid, deathtime)
                    events_by_sub[sid].append(EventRecord(
                        subject_id=sid,
                        hadm_id=hadm_id,
                        source_table="admissions",
                        source_row_id=f"adm_{hadm_id}_death",
                        charttime=hr,
                        storetime=hr,
                        category_id=EventCategory.DEATH,
                        modality_id=Modality.CATEGORICAL,
                        has_specifics=False,
                    ))

        for sid in target_sids:
            p_info = self.patients[sid]
            d = demog_data.get(sid, {})
            prefix = DemographicPrefix(
                birth_year=p_info["birth_year"],
                race=d.get("race", "Unknown"),
                gender=p_info["gender"],
                insurance=d.get("insurance", "Unknown"),
                language=d.get("language", "Unknown"),
                marital_status=d.get("marital_status", "Unknown"),
            )
            results[sid] = (prefix, events_by_sub[sid])

        return results

    def extract_ecg_events(self, target_sids: Set[int]) -> Dict[int, List[EventRecord]]:
        events_by_sub: Dict[int, List[EventRecord]] = {sid: [] for sid in target_sids}
        ecg_path = ECG_DIR / "record_list.csv"
        if not ecg_path.exists():
            return events_by_sub

        with open(ecg_path, "rt", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = int(row["subject_id"])
                if sid not in target_sids:
                    continue
                ecg_dt = parse_datetime(row["ecg_time"])
                if ecg_dt:
                    hr = self.dt_to_hours(sid, ecg_dt)
                    study_id = row.get("study_id", "")
                    events_by_sub[sid].append(EventRecord(
                        subject_id=sid,
                        hadm_id=None,
                        source_table="mimic_iv_ecg",
                        source_row_id=f"ecg_{study_id}",
                        charttime=hr,
                        storetime=hr,
                        category_id=EventCategory.ECG,
                        modality_id=Modality.ECG,
                        has_specifics=False,
                        has_value=True,
                        value_raw=row.get("path", ""),
                    ))
        return events_by_sub

    def extract_cxr_events(self, target_sids: Set[int]) -> Dict[int, List[EventRecord]]:
        events_by_sub: Dict[int, List[EventRecord]] = {sid: [] for sid in target_sids}
        cxr_path = CXR_DIR / "mimic-cxr-2.0.0-metadata.csv.gz"
        if not cxr_path.exists():
            return events_by_sub

        with gzip.open(cxr_path, "rt", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = int(row["subject_id"])
                if sid not in target_sids:
                    continue
                cxr_dt = parse_cxr_datetime(row["StudyDate"], row.get("StudyTime", ""))
                if cxr_dt:
                    hr = self.dt_to_hours(sid, cxr_dt)
                    dicom_id = row.get("dicom_id", "")
                    events_by_sub[sid].append(EventRecord(
                        subject_id=sid,
                        hadm_id=None,
                        source_table="mimic_cxr",
                        source_row_id=f"cxr_{dicom_id}",
                        charttime=hr,
                        storetime=hr,
                        category_id=EventCategory.X_RAY,
                        modality_id=Modality.CXR,
                        has_specifics=True,
                        specifics_text=row.get("PerformedProcedureStepDescription", "CHEST").strip(),
                        has_value=True,
                        value_raw=dicom_id,
                    ))
        return events_by_sub

    def extract_lab_events(self, target_sids: Set[int]) -> Dict[int, List[EventRecord]]:
        events_by_sub: Dict[int, List[EventRecord]] = {sid: [] for sid in target_sids}
        lab_path = HOSP_DIR / "labevents.csv"
        if not lab_path.exists():
            return events_by_sub

        with open(lab_path, "rt", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = int(row["subject_id"])
                if sid not in target_sids:
                    continue
                charttime = parse_datetime(row["charttime"])
                if not charttime:
                    continue
                valuenum_str = row.get("valuenum", "")
                if not valuenum_str:
                    continue
                try:
                    val = float(valuenum_str)
                except ValueError:
                    continue

                hr = self.dt_to_hours(sid, charttime)
                store_dt = parse_datetime(row.get("storetime", ""))
                store_hr = self.dt_to_hours(sid, store_dt) if store_dt else hr
                hadm_id = int(row["hadm_id"]) if row.get("hadm_id") else None

                fourier_vec = self.fourier_encoder.encode(val)

                events_by_sub[sid].append(EventRecord(
                    subject_id=sid,
                    hadm_id=hadm_id,
                    source_table="labevents",
                    source_row_id=row.get("labevent_id", ""),
                    charttime=hr,
                    storetime=store_hr,
                    category_id=EventCategory.LAB_TEST,
                    modality_id=Modality.NUMERIC,
                    has_specifics=True,
                    specifics_text=f"item_{row.get('itemid', '')}",
                    has_value=True,
                    value_raw=val,
                    value_vec=fourier_vec,
                ))
        return events_by_sub

    def assemble_patient_journey(self, sid: int, prefix: DemographicPrefix, all_events: List[EventRecord]) -> PatientJourney:
        """
        Sort events chronologically, compute directional deltas (prev/next),
        assign ordinals and age at event.
        """
        # Sort by charttime, tie-break by category_id and source_row_id
        all_events.sort(key=lambda e: (e.charttime, e.category_id, e.source_row_id))

        n = len(all_events)
        for i, ev in enumerate(all_events):
            ev.ordinal = i
            ev.age = ev.charttime / (365.25 * 24.0)  # Convert hours to years
            ev.delta_prev = max(0.0, ev.charttime - all_events[i - 1].charttime) if i > 0 else 0.0
            ev.delta_next = max(0.0, all_events[i + 1].charttime - ev.charttime) if i < n - 1 else 0.0

        return PatientJourney(subject_id=sid, demographics=prefix, events=all_events)
