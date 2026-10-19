"""
Unit Tests for Dynamic Demographic Prefix Updating & Embedding Injection.
"""

import numpy as np
import pytest

from src.data.schema import (
    EventCategory,
    Modality,
    DemographicPrefix,
    EventRecord,
    PatientJourney,
    ModelWindow,
)
from src.data.window_builder import WindowBuilder


def test_journey_dynamic_demographics_lookup():
    """Verifies get_demographics_at retrieves latest status <= t."""
    initial_demo = DemographicPrefix(
        birth_year=1960,
        race="WHITE",
        gender="M",
        insurance="Medicare",
        language="ENGLISH",
        marital_status="SINGLE",
    )
    later_demo = DemographicPrefix(
        birth_year=1960,
        race="WHITE",
        gender="M",
        insurance="Medicaid",
        language="ENGLISH",
        marital_status="MARRIED",
    )
    
    # History: admission 1 at t=100.0, admission 2 at t=500.0
    history = [
        (100.0, initial_demo),
        (500.0, later_demo),
    ]
    
    journey = PatientJourney(
        subject_id=101,
        demographics=initial_demo,
        events=[],
        demographic_history=history,
    )

    # 1. Before admission 1 (t = 50.0): time-varying variables are 'Unknown'
    d_early = journey.get_demographics_at(50.0)
    assert d_early.birth_year == 1960
    assert d_early.gender == "M"
    assert d_early.insurance == "Unknown"
    assert d_early.marital_status == "Unknown"

    # 2. Between admission 1 and 2 (t = 250.0): initial_demo
    d_mid = journey.get_demographics_at(250.0)
    assert d_mid.insurance == "Medicare"
    assert d_mid.marital_status == "SINGLE"

    # 3. After admission 2 (t = 600.0): later_demo
    d_late = journey.get_demographics_at(600.0)
    assert d_late.insurance == "Medicaid"
    assert d_late.marital_status == "MARRIED"


def test_window_builder_dynamic_prefix_and_embeddings():
    """Verifies WindowBuilder injects updated demographics and BioLORD vectors."""
    initial_demo = DemographicPrefix(1960, "WHITE", "M", "Medicare", "ENGLISH", "SINGLE")
    later_demo = DemographicPrefix(1960, "WHITE", "M", "Medicaid", "ENGLISH", "MARRIED")
    history = [(50.0, initial_demo), (300.0, later_demo)]

    events = [
        EventRecord(
            subject_id=101, hadm_id=1, source_table="t", source_row_id=f"r_{i}",
            charttime=float(i * 10), storetime=float(i * 10), delta_prev=10.0, delta_next=10.0,
            age=50.0, ordinal=i, category_id=EventCategory.LAB_TEST, modality_id=Modality.NUMERIC,
        ) for i in range(70) # events span t = 0 to 690
    ]
    journey = PatientJourney(101, initial_demo, events, history)

    # Mock embeddings dictionary
    demo_dict = {
        "Date of birth: 1960": np.ones(768, dtype=np.float32) * 0.1,
        "Race: WHITE": np.ones(768, dtype=np.float32) * 0.2,
        "Gender: M": np.ones(768, dtype=np.float32) * 0.3,
        "Insurance: Medicare": np.ones(768, dtype=np.float32) * 0.4,
        "Insurance: Medicaid": np.ones(768, dtype=np.float32) * 0.5,
        "Language: ENGLISH": np.ones(768, dtype=np.float32) * 0.6,
        "Marital status: SINGLE": np.ones(768, dtype=np.float32) * 0.7,
        "Marital status: MARRIED": np.ones(768, dtype=np.float32) * 0.8,
    }

    # Window capacity: max_seq_len=40, prefix=6 -> 34 clinical events per window, overlap=10
    builder = WindowBuilder(
        max_seq_len=40,
        overlap_len=10,
        prefix_len=6,
        min_events_per_patient=64,
        demo_embeddings_dict=demo_dict,
    )
    windows = builder.build_patient_windows(journey)
    assert len(windows) >= 2

    # Window 0: events 0..33 (t = 0 to 330) -> at t=330, latest is later_demo (t=300)
    w0 = windows[0]
    # Check that specifics_vec is populated for prefix tokens (indices 0..5)
    assert np.all(w0.has_specifics[:6] == True)
    assert np.all(w0.loss_mask[:6] == False) # Prefix never scored
    assert np.isclose(w0.specifics_vec[0, 0], 0.1) # DOB
    assert np.isclose(w0.specifics_vec[3, 0], 0.5) # Medicaid (updated at window end t=330)
    assert np.isclose(w0.specifics_vec[5, 0], 0.8) # MARRIED (updated at window end t=330)
