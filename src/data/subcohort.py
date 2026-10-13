"""
Pilot Cohort Selection for NOAH Replication.

Selects a deterministic, patient-disjoint subset of multimodal patients
who have structured encounters, ECG, and Chest X-ray records.
Splits into 70% train, 15% validation, and 15% test according to DATA-05.
"""

import csv
import gzip
import os
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple


MIMIC_BASE = Path("/home/wey371/rajpurkarlab/datasets")
HOSP_DIR = MIMIC_BASE / "mimic-iv/physionet.org/files/mimiciv/2.2/hosp"
ECG_RECORD_LIST = MIMIC_BASE / "mimic-iv-ecg/1.0/record_list.csv"
CXR_METADATA = MIMIC_BASE / "mimic-cxr-resized/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz"


def load_subject_ids_with_ecg() -> Set[str]:
    """Extract set of unique subject_ids with at least one ECG."""
    subjects = set()
    if not ECG_RECORD_LIST.exists():
        return subjects
    with open(ECG_RECORD_LIST, "rt", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        sub_idx = header.index("subject_id") if header and "subject_id" in header else 0
        for row in reader:
            if row:
                subjects.add(row[sub_idx])
    return subjects


def load_subject_ids_with_cxr() -> Set[str]:
    """Extract set of unique subject_ids with at least one CXR."""
    subjects = set()
    if not CXR_METADATA.exists():
        return subjects
    with gzip.open(CXR_METADATA, "rt", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        sub_idx = header.index("subject_id") if header and "subject_id" in header else 1
        for row in reader:
            if row:
                subjects.add(row[sub_idx])
    return subjects


def count_admissions_per_subject() -> Dict[str, int]:
    """Count admissions per subject from admissions.csv."""
    counts: Dict[str, int] = {}
    adm_path = HOSP_DIR / "admissions.csv"
    if not adm_path.exists():
        return counts
    with open(adm_path, "rt", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        sub_idx = header.index("subject_id") if header and "subject_id" in header else 0
        for row in reader:
            if row:
                sid = row[sub_idx]
                counts[sid] = counts.get(sid, 0) + 1
    return counts


def generate_pilot_cohort(
    target_size: int = 1000,
    seed: int = 42,
    output_path: Path = Path("data/pilot_cohort_manifest.csv")
) -> List[Dict[str, str]]:
    """
    Generate a deterministic pilot cohort of multimodal patients.
    
    Filter criteria:
    - Present in patients.csv
    - anchor_age >= 18
    - has >= 1 admission
    - has >= 1 ECG
    - has >= 1 CXR
    """
    print("Scanning ECG records...")
    ecg_subjects = load_subject_ids_with_ecg()
    print(f"Found {len(ecg_subjects):,} subjects with ECG.")

    print("Scanning CXR records...")
    cxr_subjects = load_subject_ids_with_cxr()
    print(f"Found {len(cxr_subjects):,} subjects with CXR.")

    print("Scanning admissions...")
    adm_counts = count_admissions_per_subject()
    print(f"Found {len(adm_counts):,} subjects with hospital admissions.")

    # Intersection of all 3
    multimodal_pool = ecg_subjects.intersection(cxr_subjects).intersection(adm_counts.keys())
    print(f"Intersection (ECG + CXR + Admissions): {len(multimodal_pool):,} subjects.")

    # Read patient metadata
    patients_path = HOSP_DIR / "patients.csv"
    eligible_patients = []
    with open(patients_path, "rt", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        col = {k: idx for idx, k in enumerate(header)}
        for row in reader:
            sid = row[col["subject_id"]]
            if sid in multimodal_pool:
                age = int(row[col["anchor_age"]]) if row[col["anchor_age"]].isdigit() else 0
                if age >= 18:
                    eligible_patients.append({
                        "subject_id": sid,
                        "gender": row[col["gender"]],
                        "anchor_age": row[col["anchor_age"]],
                        "anchor_year": row[col["anchor_year"]],
                        "anchor_year_group": row[col["anchor_year_group"]],
                        "dod": row[col["dod"]],
                        "n_admissions": str(adm_counts[sid]),
                    })

    print(f"Eligible adult multimodal patients: {len(eligible_patients):,}")

    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(eligible_patients)

    selected = eligible_patients[:target_size]
    print(f"Selected {len(selected):,} patients for pilot subcohort.")

    # Partition 70% Train, 15% Val, 15% Test
    n_train = int(0.70 * len(selected))
    n_val = int(0.15 * len(selected))
    for i, p in enumerate(selected):
        if i < n_train:
            p["split"] = "train"
        elif i < n_train + n_val:
            p["split"] = "val"
        else:
            p["split"] = "test"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "split",
        "gender",
        "anchor_age",
        "anchor_year",
        "anchor_year_group",
        "dod",
        "n_admissions"
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"Manifest written to {output_path}")
    return selected


if __name__ == "__main__":
    generate_pilot_cohort(target_size=1000)
