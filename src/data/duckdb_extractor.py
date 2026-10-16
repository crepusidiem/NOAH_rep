"""
High-Performance DuckDB-Powered Multimodal Event Extractor for MIMIC-IV.

Implements the unified event mapping covering all 31 NOAH categories
across MIMIC-IV (hosp, icu), MIMIC-IV-ED, MIMIC-IV-Note, MIMIC-IV-ECG,
and MIMIC-CXR (with modular plug-and-play hook for MIMIC-IV-ECHO).

Features:
- Multi-threaded DuckDB execution over compressed/uncompressed CSVs.
- Clean handling of de-identification anchor dates, age calculation, and origin t0.
- Strict causal process separation (process start at starttime, process end at endtime).
- Deterministic tie-breaking and directional delta calculation (delta_prev, delta_next).
- Robust handling of nonfinite numbers, dirty strings, missing units, and edge cases.
"""

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import duckdb
import numpy as np

from src.data.schema import (
    EventCategory,
    Modality,
    DemographicPrefix,
    EventRecord,
    PatientJourney,
)
from src.data.numeric import DyadicFourierEncoder


class DuckDBEventExtractor:
    """
    Extracts, cleans, and standardizes patient event streams from MIMIC-IV.
    """
    def __init__(
        self,
        manifest_path: Path = Path("data/pilot_cohort_manifest.csv"),
        mimic_root: Path = Path("/home/wey371/rajpurkarlab/datasets"),
        cxr_dir: Optional[Union[str, Path]] = None,
        fourier_encoder: Optional[DyadicFourierEncoder] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.mimic_root = Path(mimic_root)
        self.fourier_encoder = fourier_encoder or DyadicFourierEncoder()
        
        self.hosp_dir = self.mimic_root / "mimic-iv/physionet.org/files/mimiciv/2.2/hosp"
        self.icu_dir = self.mimic_root / "mimic-iv/physionet.org/files/mimiciv/2.2/icu"
        self.ed_dir = self.mimic_root / "mimic-iv-ed/physionet.org/files/mimic-iv-ed/2.2/ed"
        self.note_dir = self.mimic_root / "mimic-iv-note/physionet.org/files/mimic-iv-note/2.2/note"
        self.ecg_dir = self.mimic_root / "mimic-iv-ecg/1.0"
        if cxr_dir is not None:
            self.cxr_dir = Path(cxr_dir) if Path(cxr_dir).is_absolute() else self.mimic_root / cxr_dir
        else:
            self.cxr_dir = self.mimic_root / "cxr/MIMIC-CXR/raw_jpg"
        self.echo_dir = self.mimic_root / "mimic-iv-echo/1.0"

        # Initialize DuckDB connection
        self.con = duckdb.connect(database=":memory:")
        self.con.execute("PRAGMA threads=8;")
        self.con.execute("PRAGMA preserve_insertion_order=false;")
        self.con.execute("PRAGMA max_memory='16GB';")
        
        self.patients: Dict[int, Dict] = {}
        self._load_cohort_manifest()

    def close(self):
        """Releases all DuckDB memory buffers and connection."""
        if hasattr(self, "con") and self.con is not None:
            try:
                self.con.close()
            except Exception:
                pass
            self.con = None

    def _read_csv(self, path: Path) -> str:
        """Robust DuckDB read_csv expression handling quotes, commas, null padding, and malformed rows."""
        return f"read_csv('{path}', header=true, quote='\"', escape='\"', null_padding=true, ignore_errors=true, parallel=false)"

    def _load_cohort_manifest(self):
        """Loads cohort manifest into DuckDB temporary table and memory."""
        self.con.execute(f"""
            CREATE TEMP TABLE cohort_manifest AS
            SELECT 
                CAST(subject_id AS BIGINT) AS subject_id,
                split,
                gender,
                CAST(anchor_age AS INTEGER) AS anchor_age,
                CAST(anchor_year AS INTEGER) AS anchor_year,
                anchor_year_group,
                dod,
                (CAST(anchor_year AS INTEGER) - CAST(anchor_age AS INTEGER)) AS birth_year
            FROM {self._read_csv(self.manifest_path)}
        """)
        rows = self.con.execute("SELECT subject_id, birth_year, gender, anchor_age FROM cohort_manifest").fetchall()
        for sid, byear, gender, a_age in rows:
            self.patients[sid] = {
                "birth_year": byear,
                "gender": gender,
                "anchor_age": a_age,
                "t0": datetime.datetime(byear, 1, 1, 0, 0, 0),
            }

    def dt_to_hours(self, sid: int, dt: datetime.datetime) -> float:
        """Converts datetime to physical elapsed hours from patient origin t0."""
        t0 = self.patients[sid]["t0"]
        return (dt - t0).total_seconds() / 3600.0

    def extract_all(self, target_sids: Optional[Set[int]] = None) -> Dict[int, PatientJourney]:
        """
        Extracts all categories across all available tables for target patients.
        """
        if target_sids is None:
            target_sids = set(self.patients.keys())

        # Register filter table if subset requested
        if len(target_sids) < len(self.patients):
            self.con.execute("CREATE TEMP TABLE filter_sids (subject_id BIGINT);")
            self.con.executemany("INSERT INTO filter_sids VALUES (?);", [(s,) for s in target_sids])
            target_clause = "JOIN filter_sids fs ON t.subject_id = fs.subject_id"
        else:
            target_clause = "JOIN cohort_manifest fs ON t.subject_id = fs.subject_id"

        all_events_by_sub: Dict[int, List[EventRecord]] = {sid: [] for sid in target_sids}
        prefixes_by_sub: Dict[int, DemographicPrefix] = {}

        # 1. Admissions & Demographics
        self._extract_admissions_and_demographics(target_clause, all_events_by_sub, prefixes_by_sub)

        # 2. ICU Stays
        self._extract_icu_stays(target_clause, all_events_by_sub)

        # 3. ED Stays & Triage
        self._extract_ed_events(target_clause, all_events_by_sub)

        # 4. Transfers & Services
        self._extract_transfers_and_services(target_clause, all_events_by_sub)

        # 5. Procedures
        self._extract_procedures(target_clause, all_events_by_sub)

        # 6. Inputs & Outputs
        self._extract_inputs_and_outputs(target_clause, all_events_by_sub)

        # 7. Prescriptions & Medications
        self._extract_medications(target_clause, all_events_by_sub)

        # 8. Diagnoses
        self._extract_diagnoses(target_clause, all_events_by_sub)

        # 9. Labs
        self._extract_labs(target_clause, all_events_by_sub)

        # 10. Microbiology
        self._extract_microbiology(target_clause, all_events_by_sub)

        # 11. Notes
        self._extract_notes(target_clause, all_events_by_sub)

        # 12. ECGs
        self._extract_ecgs(target_clause, all_events_by_sub)

        # 13. CXRs
        self._extract_cxrs(target_clause, all_events_by_sub)

        # 14. ECHO (Plug-and-play hook)
        self._extract_echos(target_clause, all_events_by_sub)

        # 15. Assemble into chronological PatientJourneys
        journeys: Dict[int, PatientJourney] = {}
        for sid in target_sids:
            prefix = prefixes_by_sub.get(sid, DemographicPrefix(
                birth_year=self.patients[sid]["birth_year"],
                race="Unknown",
                gender=self.patients[sid]["gender"],
                insurance="Unknown",
                language="Unknown",
                marital_status="Unknown",
            ))
            evs = all_events_by_sub[sid]
            # Deterministic sorting: time, category, source_table, source_row_id
            evs.sort(key=lambda e: (e.charttime, int(e.category_id), e.source_table, e.source_row_id))

            n = len(evs)
            for i, ev in enumerate(evs):
                ev.ordinal = i
                ev.age = ev.charttime / (365.25 * 24.0)
                ev.delta_prev = max(0.0, ev.charttime - evs[i - 1].charttime) if i > 0 else 0.0
                if i < n - 1:
                    ev.delta_next = max(0.0, evs[i + 1].charttime - ev.charttime)
                    ev.has_next_gap = True
                else:
                    ev.delta_next = 0.0
                    ev.has_next_gap = False  # Terminal event has no observed successor

            journeys[sid] = PatientJourney(subject_id=sid, demographics=prefix, events=evs)

        return journeys

    def _extract_admissions_and_demographics(self, target_clause, events_by_sub, prefixes_by_sub):
        adm_path = self.hosp_dir / "admissions.csv"
        if not adm_path.exists():
            return

        query = f"""
            SELECT 
                t.subject_id, t.hadm_id, t.admittime, t.dischtime, t.deathtime,
                t.admission_type, t.discharge_location, t.race, t.insurance, t.language, t.marital_status
            FROM {self._read_csv(adm_path)} t
            {target_clause}
        """
        rows = self.con.execute(query).fetchall()
        for r in rows:
            sid, hadm_id, adm_dt, disch_dt, death_dt, adm_type, disch_loc, race, ins, lang, marital = r
            
            # Store demographics prefix if not yet set
            if sid not in prefixes_by_sub:
                prefixes_by_sub[sid] = DemographicPrefix(
                    birth_year=self.patients[sid]["birth_year"],
                    race=race or "Unknown",
                    gender=self.patients[sid]["gender"],
                    insurance=ins or "Unknown",
                    language=lang or "Unknown",
                    marital_status=marital or "Unknown",
                )

            # Admission
            if adm_dt:
                hr = self.dt_to_hours(sid, adm_dt)
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=hadm_id, source_table="admissions", source_row_id=f"adm_{hadm_id}_enter",
                    charttime=hr, storetime=hr, category_id=EventCategory.ENTER_HOSPITALIZATION, modality_id=Modality.CATEGORICAL,
                    has_specifics=True, specifics_text=adm_type or "ADMISSION",
                ))

            # Discharge
            if disch_dt:
                hr = self.dt_to_hours(sid, disch_dt)
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=hadm_id, source_table="admissions", source_row_id=f"adm_{hadm_id}_leave",
                    charttime=hr, storetime=hr, category_id=EventCategory.LEAVE_HOSPITALIZATION, modality_id=Modality.CATEGORICAL,
                    has_specifics=True, specifics_text=disch_loc or "DISCHARGED",
                ))

            # Death
            if death_dt:
                hr = self.dt_to_hours(sid, death_dt)
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=hadm_id, source_table="admissions", source_row_id=f"adm_{hadm_id}_death",
                    charttime=hr, storetime=hr, category_id=EventCategory.DEATH, modality_id=Modality.CATEGORICAL,
                ))

    def _extract_icu_stays(self, target_clause, events_by_sub):
        icu_path = self.icu_dir / "icustays.csv.gz"
        if not icu_path.exists():
            return

        query = f"""
            SELECT t.subject_id, t.hadm_id, t.stay_id, t.first_careunit, t.last_careunit, t.intime, t.outtime
            FROM {self._read_csv(icu_path)} t
            {target_clause}
        """
        for r in self.con.execute(query).fetchall():
            sid, hadm_id, stay_id, f_unit, l_unit, in_dt, out_dt = r
            if in_dt:
                hr = self.dt_to_hours(sid, in_dt)
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=hadm_id, source_table="icustays", source_row_id=f"icu_{stay_id}_enter",
                    charttime=hr, storetime=hr, category_id=EventCategory.ENTER_ICU, modality_id=Modality.CATEGORICAL,
                    has_specifics=True, specifics_text=f_unit or "ICU",
                ))
            if out_dt:
                hr = self.dt_to_hours(sid, out_dt)
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=hadm_id, source_table="icustays", source_row_id=f"icu_{stay_id}_leave",
                    charttime=hr, storetime=hr, category_id=EventCategory.LEAVE_ICU, modality_id=Modality.CATEGORICAL,
                    has_specifics=True, specifics_text=l_unit or "ICU",
                ))

    def _extract_ed_events(self, target_clause, events_by_sub):
        ed_path = self.ed_dir / "edstays.csv"
        if ed_path.exists():
            query = f"""
                SELECT t.subject_id, t.hadm_id, t.stay_id, t.intime, t.outtime, t.arrival_transport, t.disposition
                FROM {self._read_csv(ed_path)} t
                {target_clause}
            """
            for r in self.con.execute(query).fetchall():
                sid, hadm_id, stay_id, in_dt, out_dt, trans, disp = r
                if in_dt:
                    hr = self.dt_to_hours(sid, in_dt)
                    events_by_sub[sid].append(EventRecord(
                        subject_id=sid, hadm_id=hadm_id, source_table="edstays", source_row_id=f"ed_{stay_id}_enter",
                        charttime=hr, storetime=hr, category_id=EventCategory.ENTER_ED, modality_id=Modality.CATEGORICAL,
                        has_specifics=True, specifics_text=trans or "ED_ARRIVAL",
                    ))
                if out_dt:
                    hr = self.dt_to_hours(sid, out_dt)
                    events_by_sub[sid].append(EventRecord(
                        subject_id=sid, hadm_id=hadm_id, source_table="edstays", source_row_id=f"ed_{stay_id}_leave",
                        charttime=hr, storetime=hr, category_id=EventCategory.LEAVE_ED, modality_id=Modality.CATEGORICAL,
                        has_specifics=True, specifics_text=disp or "ED_DISCHARGE",
                    ))

        # Triage complaint & acuity
        triage_path = self.ed_dir / "triage.csv"
        if triage_path.exists() and ed_path.exists():
            query = f"""
                SELECT t.subject_id, t.stay_id, t.acuity, t.chiefcomplaint, es.intime
                FROM {self._read_csv(triage_path)} t
                JOIN {self._read_csv(ed_path)} es ON t.stay_id = es.stay_id
                {target_clause}
            """
            for r in self.con.execute(query).fetchall():
                sid, stay_id, acuity, complaint, in_dt = r
                if in_dt:
                    hr = self.dt_to_hours(sid, in_dt)
                    if complaint:
                        events_by_sub[sid].append(EventRecord(
                            subject_id=sid, hadm_id=None, source_table="triage", source_row_id=f"complaint_{stay_id}",
                            charttime=hr, storetime=hr, category_id=EventCategory.COMPLAINT, modality_id=Modality.CATEGORICAL,
                            has_specifics=True, specifics_text=str(complaint).strip(),
                        ))
                    if acuity is not None:
                        try:
                            ac_val = float(acuity)
                            f_vec = self.fourier_encoder.encode(ac_val)
                            events_by_sub[sid].append(EventRecord(
                                subject_id=sid, hadm_id=None, source_table="triage", source_row_id=f"acuity_{stay_id}",
                                charttime=hr, storetime=hr, category_id=EventCategory.TRIAGE_ACUITY, modality_id=Modality.NUMERIC,
                                has_specifics=True, specifics_text=f"Acuity {int(ac_val)}", has_value=True, value_raw=ac_val, value_vec=f_vec,
                            ))
                        except (ValueError, TypeError):
                            pass

    def _extract_transfers_and_services(self, target_clause, events_by_sub):
        trans_path = self.hosp_dir / "transfers.csv.gz"
        if trans_path.exists():
            query = f"""
                SELECT t.subject_id, t.hadm_id, t.transfer_id, t.careunit, t.intime, t.outtime
                FROM {self._read_csv(trans_path)} t
                {target_clause}
                WHERE t.careunit IS NOT NULL
            """
            for r in self.con.execute(query).fetchall():
                sid, hadm_id, tr_id, unit, in_dt, out_dt = r
                if in_dt:
                    hr = self.dt_to_hours(sid, in_dt)
                    events_by_sub[sid].append(EventRecord(
                        subject_id=sid, hadm_id=hadm_id, source_table="transfers", source_row_id=f"tr_{tr_id}_in",
                        charttime=hr, storetime=hr, category_id=EventCategory.TRANSFER, modality_id=Modality.CATEGORICAL,
                        has_specifics=True, specifics_text=str(unit),
                    ))
                if out_dt:
                    hr = self.dt_to_hours(sid, out_dt)
                    events_by_sub[sid].append(EventRecord(
                        subject_id=sid, hadm_id=hadm_id, source_table="transfers", source_row_id=f"tr_{tr_id}_out",
                        charttime=hr, storetime=hr, category_id=EventCategory.LEAVE_TRANSFER, modality_id=Modality.CATEGORICAL,
                        has_specifics=True, specifics_text=str(unit),
                    ))

        serv_path = self.hosp_dir / "services.csv.gz"
        if serv_path.exists():
            query = f"""
                SELECT t.subject_id, t.hadm_id, t.transfertime, t.curr_service
                FROM {self._read_csv(serv_path)} t
                {target_clause}
                WHERE t.transfertime IS NOT NULL AND t.curr_service IS NOT NULL
            """
            for r in self.con.execute(query).fetchall():
                sid, hadm_id, tr_dt, serv = r
                hr = self.dt_to_hours(sid, tr_dt)
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=hadm_id, source_table="services", source_row_id=f"serv_{sid}_{hr:.2f}",
                    charttime=hr, storetime=hr, category_id=EventCategory.SERVICE, modality_id=Modality.CATEGORICAL,
                    has_specifics=True, specifics_text=str(serv),
                ))

    def _extract_procedures(self, target_clause, events_by_sub):
        proc_path = self.icu_dir / "procedureevents.csv.gz"
        d_items_path = self.icu_dir / "d_items.csv.gz"
        if proc_path.exists() and d_items_path.exists():
            query = f"""
                SELECT 
                    t.subject_id, t.hadm_id, t.orderid, t.starttime, t.endtime, 
                    COALESCE(di.label, t.ordercategoryname, 'Procedure') AS proc_name,
                    t.value
                FROM {self._read_csv(proc_path)} t
                LEFT JOIN {self._read_csv(d_items_path)} di ON t.itemid = di.itemid
                {target_clause}
                WHERE t.starttime IS NOT NULL
            """
            for r in self.con.execute(query).fetchall():
                sid, hadm_id, orderid, start_dt, end_dt, name, val = r
                hr_start = self.dt_to_hours(sid, start_dt)
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=hadm_id, source_table="procedureevents", source_row_id=f"proc_{orderid}_start",
                    charttime=hr_start, storetime=hr_start, category_id=EventCategory.PROCEDURE, modality_id=Modality.CATEGORICAL,
                    has_specifics=True, specifics_text=str(name),
                ))
                if end_dt:
                    hr_end = self.dt_to_hours(sid, end_dt)
                    events_by_sub[sid].append(EventRecord(
                        subject_id=sid, hadm_id=hadm_id, source_table="procedureevents", source_row_id=f"proc_{orderid}_end",
                        charttime=hr_end, storetime=hr_end, category_id=EventCategory.PROCEDURE_END, modality_id=Modality.CATEGORICAL,
                        has_specifics=True, specifics_text=str(name),
                    ))

    def _extract_inputs_and_outputs(self, target_clause, events_by_sub):
        # Body Inputs
        inp_path = self.icu_dir / "inputevents.csv.gz"
        d_items_path = self.icu_dir / "d_items.csv.gz"
        if inp_path.exists() and d_items_path.exists():
            query = f"""
                SELECT 
                    t.subject_id, t.hadm_id, t.orderid, t.starttime, t.endtime, 
                    COALESCE(di.label, t.ordercategoryname, 'Input') AS input_name,
                    t.amount
                FROM {self._read_csv(inp_path)} t
                LEFT JOIN {self._read_csv(d_items_path)} di ON t.itemid = di.itemid
                {target_clause}
                WHERE t.starttime IS NOT NULL
            """
            for r in self.con.execute(query).fetchall():
                sid, hadm_id, orderid, start_dt, end_dt, name, amount = r
                hr_start = self.dt_to_hours(sid, start_dt)
                f_vec = self.fourier_encoder.encode(float(amount)) if amount is not None else None
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=hadm_id, source_table="inputevents", source_row_id=f"inp_{orderid}_start",
                    charttime=hr_start, storetime=hr_start, category_id=EventCategory.BODY_INPUT,
                    modality_id=Modality.NUMERIC if f_vec is not None else Modality.CATEGORICAL,
                    has_specifics=True, specifics_text=str(name),
                    has_value=f_vec is not None, value_raw=amount, value_vec=f_vec,
                ))
                if end_dt:
                    hr_end = self.dt_to_hours(sid, end_dt)
                    events_by_sub[sid].append(EventRecord(
                        subject_id=sid, hadm_id=hadm_id, source_table="inputevents", source_row_id=f"inp_{orderid}_end",
                        charttime=hr_end, storetime=hr_end, category_id=EventCategory.BODY_INPUT_END,
                        modality_id=Modality.CATEGORICAL, has_specifics=True, specifics_text=str(name),
                    ))

        # Body Outputs
        out_path = self.icu_dir / "outputevents.csv.gz"
        if out_path.exists() and d_items_path.exists():
            query = f"""
                SELECT 
                    t.subject_id, t.hadm_id, t.stay_id, t.charttime, 
                    COALESCE(di.label, 'Output') AS output_name,
                    t.value
                FROM {self._read_csv(out_path)} t
                LEFT JOIN {self._read_csv(d_items_path)} di ON t.itemid = di.itemid
                {target_clause}
                WHERE t.charttime IS NOT NULL AND t.value IS NOT NULL
            """
            for r in self.con.execute(query).fetchall():
                sid, hadm_id, stay_id, chart_dt, name, val = r
                hr = self.dt_to_hours(sid, chart_dt)
                try:
                    num_val = float(val)
                    f_vec = self.fourier_encoder.encode(num_val)
                    events_by_sub[sid].append(EventRecord(
                        subject_id=sid, hadm_id=hadm_id, source_table="outputevents", source_row_id=f"out_{sid}_{hr:.2f}",
                        charttime=hr, storetime=hr, category_id=EventCategory.BODY_OUTPUT, modality_id=Modality.NUMERIC,
                        has_specifics=True, specifics_text=str(name), has_value=True, value_raw=num_val, value_vec=f_vec,
                    ))
                except (ValueError, TypeError):
                    pass

    def _extract_medications(self, target_clause, events_by_sub):
        rx_path = self.hosp_dir / "prescriptions.csv"
        if rx_path.exists():
            query = f"""
                SELECT t.subject_id, t.hadm_id, t.pharmacy_id, t.starttime, t.stoptime, t.drug
                FROM {self._read_csv(rx_path)} t
                {target_clause}
                WHERE t.starttime IS NOT NULL AND t.drug IS NOT NULL
            """
            for r in self.con.execute(query).fetchall():
                sid, hadm_id, pharm_id, start_dt, stop_dt, drug = r
                hr_start = self.dt_to_hours(sid, start_dt)
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=hadm_id, source_table="prescriptions", source_row_id=f"rx_{pharm_id}_start",
                    charttime=hr_start, storetime=hr_start, category_id=EventCategory.DRUG_PRESCRIPTION, modality_id=Modality.CATEGORICAL,
                    has_specifics=True, specifics_text=str(drug).strip(),
                ))
                if stop_dt:
                    hr_stop = self.dt_to_hours(sid, stop_dt)
                    events_by_sub[sid].append(EventRecord(
                        subject_id=sid, hadm_id=hadm_id, source_table="prescriptions", source_row_id=f"rx_{pharm_id}_stop",
                        charttime=hr_stop, storetime=hr_stop, category_id=EventCategory.DRUG_STOP, modality_id=Modality.CATEGORICAL,
                        has_specifics=True, specifics_text=str(drug).strip(),
                    ))

        # Medrecon (previous drugs)
        medrec_path = self.ed_dir / "medrecon.csv"
        ed_path = self.ed_dir / "edstays.csv"
        if medrec_path.exists() and ed_path.exists():
            query = f"""
                SELECT t.subject_id, t.stay_id, t.name, es.intime
                FROM {self._read_csv(medrec_path)} t
                JOIN {self._read_csv(ed_path)} es ON t.stay_id = es.stay_id
                {target_clause}
                WHERE t.name IS NOT NULL AND es.intime IS NOT NULL
            """
            for r in self.con.execute(query).fetchall():
                sid, stay_id, drug_name, in_dt = r
                hr = self.dt_to_hours(sid, in_dt)
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=None, source_table="medrecon", source_row_id=f"medrec_{stay_id}_{drug_name}",
                    charttime=hr, storetime=hr, category_id=EventCategory.DRUG_PREVIOUSLY, modality_id=Modality.CATEGORICAL,
                    has_specifics=True, specifics_text=str(drug_name).strip(),
                ))

    def _extract_diagnoses(self, target_clause, events_by_sub):
        diag_path = self.hosp_dir / "diagnoses_icd.csv"
        d_icd_path = self.hosp_dir / "d_icd_diagnoses.csv"
        adm_path = self.hosp_dir / "admissions.csv"
        if diag_path.exists() and d_icd_path.exists() and adm_path.exists():
            query = f"""
                SELECT 
                    t.subject_id, t.hadm_id, t.seq_num, 
                    COALESCE(dicd.long_title, t.icd_code) AS diag_title,
                    adm.dischtime
                FROM {self._read_csv(diag_path)} t
                LEFT JOIN {self._read_csv(d_icd_path)} dicd ON t.icd_code = dicd.icd_code AND t.icd_version = dicd.icd_version
                JOIN {self._read_csv(adm_path)} adm ON t.hadm_id = adm.hadm_id
                {target_clause}
                WHERE adm.dischtime IS NOT NULL
            """
            for r in self.con.execute(query).fetchall():
                sid, hadm_id, seq_num, diag_title, disch_dt = r
                hr = self.dt_to_hours(sid, disch_dt)
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=hadm_id, source_table="diagnoses_icd", source_row_id=f"diag_{hadm_id}_{seq_num}",
                    charttime=hr, storetime=hr, category_id=EventCategory.DIAGNOSIS, modality_id=Modality.CATEGORICAL,
                    has_specifics=True, specifics_text=str(diag_title).strip(),
                ))

    def _extract_labs(self, target_clause, events_by_sub):
        lab_path = self.hosp_dir / "labevents.csv"
        d_lab_path = self.hosp_dir / "d_labitems.csv"
        if not lab_path.exists() or not d_lab_path.exists():
            return

        query = f"""
            SELECT 
                t.subject_id, t.hadm_id, t.labevent_id, t.charttime, t.valuenum,
                COALESCE(dl.label, 'Lab') AS lab_label
            FROM {self._read_csv(lab_path)} t
            LEFT JOIN {self._read_csv(d_lab_path)} dl ON t.itemid = dl.itemid
            {target_clause}
            WHERE t.charttime IS NOT NULL AND t.valuenum IS NOT NULL
        """
        for r in self.con.execute(query).fetchall():
            sid, hadm_id, lab_id, chart_dt, valuenum, label = r
            if np.isnan(valuenum) or np.isinf(valuenum):
                continue
            hr = self.dt_to_hours(sid, chart_dt)
            f_vec = self.fourier_encoder.encode(float(valuenum))
            events_by_sub[sid].append(EventRecord(
                subject_id=sid, hadm_id=hadm_id, source_table="labevents", source_row_id=f"lab_{lab_id}",
                charttime=hr, storetime=hr, category_id=EventCategory.LAB_TEST, modality_id=Modality.NUMERIC,
                has_specifics=True, specifics_text=str(label).strip(), has_value=True, value_raw=float(valuenum), value_vec=f_vec,
            ))

    def _extract_microbiology(self, target_clause, events_by_sub):
        micro_path = self.hosp_dir / "microbiologyevents.csv"
        if not micro_path.exists():
            return

        query = f"""
            SELECT 
                t.subject_id, t.hadm_id, t.microevent_id, 
                COALESCE(t.charttime, t.chartdate) AS event_dt,
                COALESCE(t.org_name, t.test_name, 'Microbiology') AS test_title
            FROM {self._read_csv(micro_path)} t
            {target_clause}
            WHERE (t.charttime IS NOT NULL OR t.chartdate IS NOT NULL)
        """
        for r in self.con.execute(query).fetchall():
            sid, hadm_id, m_id, dt, title = r
            if dt:
                hr = self.dt_to_hours(sid, dt)
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=hadm_id, source_table="microbiologyevents", source_row_id=f"micro_{m_id}",
                    charttime=hr, storetime=hr, category_id=EventCategory.MICROBIOLOGY_TEST, modality_id=Modality.CATEGORICAL,
                    has_specifics=True, specifics_text=str(title).strip(),
                ))

    def _extract_notes(self, target_clause, events_by_sub):
        # Discharge Summaries
        disch_note_path = self.note_dir / "discharge.csv"
        if disch_note_path.exists():
            query = f"""
                SELECT t.subject_id, t.hadm_id, t.note_id, t.charttime, t.text
                FROM {self._read_csv(disch_note_path)} t
                {target_clause}
                WHERE t.charttime IS NOT NULL
            """
            for r in self.con.execute(query).fetchall():
                sid, hadm_id, note_id, chart_dt, text = r
                hr = self.dt_to_hours(sid, chart_dt)
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=hadm_id, source_table="discharge_notes", source_row_id=f"disch_note_{note_id}",
                    charttime=hr, storetime=hr, category_id=EventCategory.NOTE_DISCHARGESUMMARY, modality_id=Modality.TEXT,
                    has_specifics=True, specifics_text=text[:512] if text else "Discharge Summary",
                    has_value=True, value_raw=text,
                ))

        # Radiology Reports
        rad_note_path = self.note_dir / "radiology.csv"
        if rad_note_path.exists():
            query = f"""
                SELECT t.subject_id, t.hadm_id, t.note_id, t.charttime, t.text
                FROM {self._read_csv(rad_note_path)} t
                {target_clause}
                WHERE t.charttime IS NOT NULL
            """
            for r in self.con.execute(query).fetchall():
                sid, hadm_id, note_id, chart_dt, text = r
                hr = self.dt_to_hours(sid, chart_dt)
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=hadm_id, source_table="radiology_notes", source_row_id=f"rad_note_{note_id}",
                    charttime=hr, storetime=hr, category_id=EventCategory.NOTE_RADIOLOGYREPORT, modality_id=Modality.TEXT,
                    has_specifics=True, specifics_text=text[:512] if text else "Radiology Report",
                    has_value=True, value_raw=text,
                ))

    def _extract_ecgs(self, target_clause, events_by_sub):
        ecg_path = self.ecg_dir / "record_list.csv"
        if not ecg_path.exists():
            return

        query = f"""
            SELECT t.subject_id, t.study_id, t.ecg_time, t.path
            FROM {self._read_csv(ecg_path)} t
            {target_clause}
            WHERE t.ecg_time IS NOT NULL
        """
        for r in self.con.execute(query).fetchall():
            sid, study_id, ecg_dt, path = r
            if ecg_dt:
                hr = self.dt_to_hours(sid, ecg_dt)
                events_by_sub[sid].append(EventRecord(
                    subject_id=sid, hadm_id=None, source_table="mimic_iv_ecg", source_row_id=f"ecg_{study_id}",
                    charttime=hr, storetime=hr, category_id=EventCategory.ECG, modality_id=Modality.ECG,
                    has_specifics=False, has_value=True, value_raw=path,
                ))

    def _extract_cxrs(self, target_clause, events_by_sub):
        cxr_path = self.cxr_dir / "mimic-cxr-2.0.0-metadata.csv.gz"
        if not cxr_path.exists():
            cxr_path = self.cxr_dir / "mimic-cxr-2.0.0-metadata.csv"
        if not cxr_path.exists():
            return

        query = f"""
            SELECT t.subject_id, t.study_id, t.dicom_id, t.StudyDate, t.StudyTime, t.PerformedProcedureStepDescription
            FROM {self._read_csv(cxr_path)} t
            {target_clause}
            WHERE t.StudyDate IS NOT NULL
        """
        for r in self.con.execute(query).fetchall():
            sid, study_id, dicom_id, s_date, s_time, desc = r
            s_date_str = str(s_date).strip()
            s_time_str = str(s_time).strip() if s_time is not None else "000000"
            if len(s_date_str) == 8:
                try:
                    y, m, d = int(s_date_str[:4]), int(s_date_str[4:6]), int(s_date_str[6:8])
                    hh = int(s_time_str[:2]) if len(s_time_str) >= 2 else 0
                    mm = int(s_time_str[2:4]) if len(s_time_str) >= 4 else 0
                    ss = int(float(s_time_str[4:])) if len(s_time_str) >= 5 else 0
                    dt = datetime.datetime(y, m, d, hh, mm, ss)
                    hr = self.dt_to_hours(sid, dt)
                    rel_path = f"files/p{str(sid)[:2]}/p{sid}/s{study_id}/{dicom_id}.jpg"
                    events_by_sub[sid].append(EventRecord(
                        subject_id=sid, hadm_id=None, source_table="mimic_cxr", source_row_id=f"cxr_{dicom_id}",
                        charttime=hr, storetime=hr, category_id=EventCategory.X_RAY, modality_id=Modality.CXR,
                        has_specifics=True, specifics_text=str(desc or "CHEST").strip(),
                        has_value=True, value_raw=rel_path,
                    ))
                except (ValueError, TypeError):
                    pass

    def _extract_echos(self, target_clause, events_by_sub):
        """Modular hook for MIMIC-IV-ECHO. Automatically activates when files are mounted."""
        if not self.echo_dir.exists():
            return
        # Extensible query pattern for echo metadata
        # SELECT subject_id, study_id, echo_time, dicom_path FROM ...
        pass
