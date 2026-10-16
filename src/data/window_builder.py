"""
Rolling Context Window Builder with Demographic Prefix & Loss Ownership.

Implements the exact NOAH windowing protocol:
1. 6 Demographic Prefix Tokens prepended to every window:
   [birth_year, race, gender, insurance, language, marital_status]
2. Rolling context L <= 2048 with 256 overlap tokens between consecutive windows.
3. Strict Loss Ownership Mask:
   - Demographic prefix tokens: loss_mask = False (unscored context)
   - Overlap tokens from previous window: loss_mask = False (prevent duplicate loss)
   - Newly exposed clinical events: loss_mask = True (scored exactly once)
   - Padded tokens: loss_mask = False, attention_mask = False
4. Patient Eligibility:
   - Discards patients with < 64 total scorable events (paper requirement p. 15).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.data.schema import (
    EventCategory,
    Modality,
    DemographicPrefix,
    EventRecord,
    PatientJourney,
    ModelWindow,
)


class WindowBuilder:
    """
    Constructs fixed-length ModelWindow tensors from PatientJourney sequences.
    """
    def __init__(
        self,
        max_seq_len: int = 1024,
        overlap_len: int = 256,
        prefix_len: int = 6,
        min_events_per_patient: int = 64,
        max_val_dim: int = 768,
    ):
        self.max_seq_len = max_seq_len
        self.overlap_len = overlap_len
        self.prefix_len = prefix_len
        self.min_events_per_patient = min_events_per_patient
        self.max_val_dim = max_val_dim

        # Number of clinical tokens allowed in a window
        self.max_clinical_tokens = self.max_seq_len - self.prefix_len
        assert self.max_clinical_tokens > self.overlap_len, (
            f"max_clinical_tokens ({self.max_clinical_tokens}) must exceed overlap ({self.overlap_len})"
        )
        self.clinical_stride = self.max_clinical_tokens - self.overlap_len

    def build_patient_windows(self, journey: PatientJourney) -> List[ModelWindow]:
        """
        Builds all rolling windows for a single patient journey.
        Returns empty list if patient has fewer than min_events_per_patient.
        """
        if not journey.is_eligible(self.min_events_per_patient):
            return []

        events = journey.events
        n_events = len(events)
        windows: List[ModelWindow] = []

        # Slice clinical event indices
        # Window 0: events[0 : max_clinical_tokens]
        # Window 1: events[clinical_stride : clinical_stride + max_clinical_tokens]
        # etc.
        start_idx = 0
        window_idx = 0

        while start_idx < n_events:
            end_idx = min(start_idx + self.max_clinical_tokens, n_events)
            chunk_events = events[start_idx:end_idx]

            # In window 0: all chunk events are new -> scored
            # In window k > 0: the first `overlap_len` events were already scored in window k-1
            # so they must have loss_mask = False!
            if window_idx == 0:
                scored_start_offset = 0
            else:
                scored_start_offset = self.overlap_len

            window = self._assemble_single_window(
                subject_id=journey.subject_id,
                demographics=journey.demographics,
                chunk_events=chunk_events,
                window_idx=window_idx,
                scored_start_offset=scored_start_offset,
            )
            windows.append(window)

            window_idx += 1
            if end_idx >= n_events:
                break
            start_idx += self.clinical_stride

        return windows

    def _assemble_single_window(
        self,
        subject_id: int,
        demographics: DemographicPrefix,
        chunk_events: List[EventRecord],
        window_idx: int,
        scored_start_offset: int,
    ) -> ModelWindow:
        """
        Assembles and pads a single window of length max_seq_len.
        """
        L = self.max_seq_len
        total_tokens = self.prefix_len + len(chunk_events)

        # Preallocate numpy arrays
        category_ids = np.zeros(L, dtype=np.int64)
        modality_ids = np.zeros(L, dtype=np.int64)
        positions = np.zeros(L, dtype=np.float32)
        ages = np.zeros(L, dtype=np.float32)
        delta_prev = np.zeros(L, dtype=np.float32)
        delta_next = np.zeros(L, dtype=np.float32)
        has_next_gap = np.zeros(L, dtype=bool)

        has_specifics = np.zeros(L, dtype=bool)
        specifics_vec = np.zeros((L, 768), dtype=np.float32)

        has_value = np.zeros(L, dtype=bool)
        value_vec = np.zeros((L, self.max_val_dim), dtype=np.float32)

        attention_mask = np.zeros(L, dtype=bool)
        loss_mask = np.zeros(L, dtype=bool)

        # 1. Populate Demographic Prefix (indices 0..5)
        # Prefix tokens are NOT scored: loss_mask remains False
        for p_idx in range(self.prefix_len):
            category_ids[p_idx] = EventCategory.OTHER_EVENT
            modality_ids[p_idx] = Modality.CATEGORICAL
            positions[p_idx] = p_idx
            ages[p_idx] = 0.0
            delta_prev[p_idx] = 0.0
            delta_next[p_idx] = 0.0
            has_next_gap[p_idx] = True
            attention_mask[p_idx] = True
            loss_mask[p_idx] = False  # NEVER score prefix

        # 2. Populate Clinical Events (indices 6 .. total_tokens-1)
        for c_i, ev in enumerate(chunk_events):
            t_idx = self.prefix_len + c_i
            category_ids[t_idx] = int(ev.category_id)
            modality_ids[t_idx] = int(ev.modality_id)
            positions[t_idx] = self.prefix_len + ev.ordinal
            ages[t_idx] = ev.age
            delta_prev[t_idx] = ev.delta_prev
            delta_next[t_idx] = ev.delta_next
            has_next_gap[t_idx] = ev.has_next_gap

            if ev.has_specifics and ev.specifics_vec is not None:
                has_specifics[t_idx] = True
                specifics_vec[t_idx] = ev.specifics_vec

            if ev.has_value and ev.value_vec is not None:
                has_value[t_idx] = True
                d_v = min(len(ev.value_vec), self.max_val_dim)
                value_vec[t_idx, :d_v] = ev.value_vec[:d_v]

            attention_mask[t_idx] = True

            # Score ownership rule:
            # Only score tokens at or beyond the overlap offset
            if c_i >= scored_start_offset:
                loss_mask[t_idx] = True
            else:
                loss_mask[t_idx] = False

        # Indices total_tokens .. L-1 remain 0 and attention_mask/loss_mask remain False (padding)
        return ModelWindow(
            subject_id=subject_id,
            window_idx=window_idx,
            total_tokens=total_tokens,
            category_ids=category_ids,
            modality_ids=modality_ids,
            positions=positions,
            ages=ages,
            delta_prev=delta_prev,
            delta_next=delta_next,
            has_next_gap=has_next_gap,
            has_specifics=has_specifics,
            specifics_vec=specifics_vec,
            has_value=has_value,
            value_vec=value_vec,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )
