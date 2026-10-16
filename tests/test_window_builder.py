"""
Unit Tests for Rolling Window Builder and Loss Ownership Masking (DATA-06..08).

Verifies:
1. U-D08: 6 Demographic prefix tokens prepended with loss_mask == False.
2. U-D09: Exactly-once scoring across all windows:
   sum of loss_mask over all windows equals total clinical events in journey.
3. U-D10: Overlap tokens remain attended (attention_mask == True) but unscored (loss_mask == False).
4. U-D07: Minimum history gate: patients with < 64 events are ineligible and yield 0 windows.
"""

import numpy as np
import pytest

from src.data.schema import (
    EventCategory,
    Modality,
    DemographicPrefix,
    EventRecord,
    PatientJourney,
)
from src.data.window_builder import WindowBuilder


def create_mock_journey(subject_id: int, num_events: int) -> PatientJourney:
    prefix = DemographicPrefix(
        birth_year=2050,
        race="WHITE",
        gender="F",
        insurance="Medicare",
        language="ENGLISH",
        marital_status="MARRIED",
    )
    events = []
    for i in range(num_events):
        events.append(EventRecord(
            subject_id=subject_id,
            hadm_id=1000 + i // 10,
            source_table="mock_table",
            source_row_id=f"row_{i}",
            charttime=float(i * 2),
            storetime=float(i * 2),
            delta_prev=2.0 if i > 0 else 0.0,
            delta_next=2.0 if i < num_events - 1 else 0.0,
            age=float(i * 2) / (365.25 * 24.0),
            ordinal=i,
            category_id=EventCategory.LAB_TEST,
            modality_id=Modality.NUMERIC,
            has_next_gap=i < num_events - 1,
            score_owner=True,
        ))
    return PatientJourney(subject_id=subject_id, demographics=prefix, events=events)


def test_ineligible_patient_filtered():
    """U-D07: Patient with fewer than 64 events produces 0 windows."""
    builder = WindowBuilder(max_seq_len=256, overlap_len=64, min_events_per_patient=64)
    short_journey = create_mock_journey(subject_id=1, num_events=63)
    windows = builder.build_patient_windows(short_journey)
    assert len(windows) == 0


def test_single_window_prefix_and_loss_mask():
    """U-D08: Window 0 prepends 6 prefix tokens with loss_mask == False."""
    builder = WindowBuilder(max_seq_len=256, overlap_len=64, min_events_per_patient=64)
    journey = create_mock_journey(subject_id=2, num_events=100)
    windows = builder.build_patient_windows(journey)
    assert len(windows) == 1
    w0 = windows[0]

    # First 6 tokens are prefix: attended but unscored
    assert np.all(w0.attention_mask[:6] == True)
    assert np.all(w0.loss_mask[:6] == False)

    # Next 100 tokens are clinical events: attended and scored
    assert np.all(w0.attention_mask[6:106] == True)
    assert np.all(w0.loss_mask[6:106] == True)

    # Remaining tokens 106..255 are padding: unattended and unscored
    assert np.all(w0.attention_mask[106:] == False)
    assert np.all(w0.loss_mask[106:] == False)


def test_multi_window_exactly_once_scoring():
    """
    U-D09: For a long sequence across multiple windows,
    each clinical event is scored EXACTLY ONCE.
    """
    max_len = 128
    overlap = 32
    prefix_len = 6
    # max_clinical = 122, stride = 90
    builder = WindowBuilder(
        max_seq_len=max_len,
        overlap_len=overlap,
        prefix_len=prefix_len,
        min_events_per_patient=64,
    )
    
    total_events = 300
    journey = create_mock_journey(subject_id=3, num_events=total_events)
    windows = builder.build_patient_windows(journey)
    
    assert len(windows) > 1

    total_scored_tokens = 0
    for w in windows:
        # Prefix tokens are never scored
        assert np.all(w.loss_mask[:prefix_len] == False)
        # Padding tokens are never scored
        assert np.all(w.loss_mask[w.total_tokens:] == False)
        # Count scored tokens in this window
        scored_in_window = np.sum(w.loss_mask)
        total_scored_tokens += scored_in_window

    # Exactly total_events scored across all windows
    assert total_scored_tokens == total_events


def test_overlap_tokens_unscored():
    """
    U-D10: In window k > 0, the first `overlap_len` clinical tokens
    have attention_mask == True but loss_mask == False.
    """
    max_len = 128
    overlap = 32
    prefix_len = 6
    builder = WindowBuilder(
        max_seq_len=max_len,
        overlap_len=overlap,
        prefix_len=prefix_len,
        min_events_per_patient=64,
    )
    journey = create_mock_journey(subject_id=4, num_events=200)
    windows = builder.build_patient_windows(journey)
    assert len(windows) >= 2
    
    w1 = windows[1]
    # In window 1, indices prefix_len .. prefix_len + overlap are overlap tokens:
    # They MUST be attended (True) but NOT scored (False)
    overlap_slice = slice(prefix_len, prefix_len + overlap)
    assert np.all(w1.attention_mask[overlap_slice] == True)
    assert np.all(w1.loss_mask[overlap_slice] == False)
    
    # Tokens after the overlap slice in window 1 should be scored
    new_clinical_slice = slice(prefix_len + overlap, w1.total_tokens)
    assert np.all(w1.loss_mask[new_clinical_slice] == True)
