"""
NOAH Data Contract & Event Schema.

Defines the 31 event categories, 6 modalities, demographic prefix fields,
and the structured EventRecord representation according to the NOAH specification.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional
import numpy as np


class EventCategory(IntEnum):
    ECG = 0
    X_RAY = 1
    ECHO = 2
    NOTE_RADIOLOGYREPORT = 3
    DEATH = 4
    ENTER_HOSPITALIZATION = 5
    NOTE_DISCHARGESUMMARY = 6
    ENTER_ICU = 7
    LEAVE_ICU = 8
    MICROBIOLOGY_TEST = 9
    SERVICE = 10
    LEAVE_TRANSFER = 11
    LEAVE_HOSPITALIZATION = 12
    PROCEDURE_END = 13
    LEAVE_ED = 14
    BODY_OUTPUT = 15
    TRANSFER = 16
    DRUG_PRESCRIPTION = 17
    DRUG_START = 18
    BODY_INPUT = 19
    COMPLAINT = 20
    DRUG_STOP = 21
    ENTER_ED = 22
    DRUG_PREVIOUSLY = 23
    PROCEDURE = 24
    DIAGNOSIS = 25
    OTHER_EVENT = 26
    TRIAGE_ACUITY = 27
    VITALS = 28
    LAB_TEST = 29
    BODY_INPUT_END = 30


# Canonical category string mapping matching Supplementary Table 1
CATEGORY_NAMES: List[str] = [
    "ECG",
    "X-ray",
    "ECHO",
    "Note Radiologyreport",
    "Death",
    "Enter Hospitalization",
    "Note Dischargesummary",
    "Enter ICU",
    "Leave ICU",
    "Microbiology Test",
    "Service",
    "Leave Transfer",
    "Leave Hospitalization",
    "Procedure End",
    "Leave ED",
    "Body Output",
    "Transfer",
    "Drug Prescription",
    "Drug Start",
    "Body Input",
    "Complaint",
    "Drug Stop",
    "Enter ED",
    "Drug Previously",
    "Procedure",
    "Diagnosis",
    "Other Event",
    "Triage Acuity",
    "Vitals",
    "Lab Test",
    "Body Input End",
]


class Modality(IntEnum):
    CATEGORICAL = 0
    NUMERIC = 1
    CXR = 2
    TEXT = 3
    ECG = 4
    ECHO = 5


MODALITY_NAMES: List[str] = [
    "categorical",
    "numeric",
    "cxr",
    "text",
    "ecg",
    "echo",
]

# Modality raw embedding dimensions from Table 15
MODALITY_DIMENSIONS: Dict[int, Optional[int]] = {
    Modality.CATEGORICAL: None,  # Learns category lookup
    Modality.NUMERIC: 44,        # 22 dyadic frequencies x (sin, cos)
    Modality.CXR: 768,           # RAD-DINO
    Modality.TEXT: 768,          # BioLORD
    Modality.ECG: 192,           # OTIS final patch mean
    Modality.ECHO: 512,          # EchoPrime global CLS
}


@dataclass
class DemographicPrefix:
    """The six demographic prefix attributes prepended to each window."""
    birth_year: int
    race: str
    gender: str
    insurance: str
    language: str
    marital_status: str

    def to_tokens(self) -> List[Dict[str, Any]]:
        """Return token representations for the 6 prefix items."""
        return [
            {"type": "prefix_dob", "value": str(self.birth_year)},
            {"type": "prefix_race", "value": self.race or "Unknown"},
            {"type": "prefix_gender", "value": self.gender or "Unknown"},
            {"type": "prefix_insurance", "value": self.insurance or "Unknown"},
            {"type": "prefix_language", "value": self.language or "Unknown"},
            {"type": "prefix_marital_status", "value": self.marital_status or "Unknown"},
        ]


@dataclass
class EventRecord:
    """
    Standardized atomic event record extracted from MIMIC.
    Represents a single clinical event in the lifelong patient sequence.
    """
    # 1. Identity and Provenance
    subject_id: int
    hadm_id: Optional[int]
    source_table: str
    source_row_id: str
    
    # 2. Universal Temporal Coordinates (hours from patient origin t0)
    charttime: float              # Physical clinical event time (hours)
    storetime: Optional[float]    # Charting / store time (hours, for causal audit)
    delta_prev: float = 0.0       # r_i - r_{i-1} (hours, >= 0)
    delta_next: float = 0.0       # r_{i+1} - r_i (hours, >= 0)
    age: float = 0.0              # Patient age at event (years)
    ordinal: int = 0              # Zero-indexed position in patient sequence
    
    # 3. Categorical & Modality Routing
    category_id: int = EventCategory.OTHER_EVENT
    modality_id: int = Modality.CATEGORICAL
    
    # 4. Specifics Payload (Optional fine-grained text embedded via BioLORD)
    has_specifics: bool = False
    specifics_text: Optional[str] = None
    specifics_vec: Optional[np.ndarray] = None  # [768] float16
    
    # 5. Value Payload (Numeric Fourier coefficients or frozen media embeddings)
    has_value: bool = False
    value_raw: Optional[Any] = None
    value_vec: Optional[np.ndarray] = None      # [D_m] float16/float32
    
    # 6. Loss Ownership Flags
    score_owner: bool = True     # Evaluated by loss function (False for prefix/overlap)


@dataclass
class PatientJourney:
    """A complete lifetime patient journey containing all chronological events."""
    subject_id: int
    demographics: DemographicPrefix
    events: List[EventRecord] = field(default_factory=list)

    @property
    def total_events(self) -> int:
        return len(self.events)

    def is_eligible(self, min_events: int = 64) -> bool:
        """Paper rule: patient must have at least min_events scored events."""
        return self.total_events >= min_events
