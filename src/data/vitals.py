"""
Numeric Vitals Sliding Aggregation (DATA-04).

Reduces frequently charted numeric vitals to a minimum and maximum per vital type,
timestamped at the window mean.
Preserves maximum charting time for strict causal audit.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

from src.data.schema import EventCategory, Modality, EventRecord


@dataclass
class RawVitalMeasurement:
    charttime: float          # In hours from patient origin
    storetime: Optional[float]
    vital_name: str           # e.g., 'heart_rate', 'sbp', 'dbp', 'respiratory_rate', 'temperature', 'spo2'
    value: float
    source_row_id: str


def aggregate_vitals_sliding_window(
    measurements: List[RawVitalMeasurement],
    subject_id: int,
    hadm_id: Optional[int] = None,
    window_span_hours: float = 2.0,
    window_stride_hours: float = 1.0,
) -> List[EventRecord]:
    """
    Apply sliding window aggregation to a sequence of raw vital measurements.
    
    Each window computes min and max for each vital present.
    Both min and max are timestamped at the mean charttime of the window.
    Both min and max share the same mean timestamp (yielding delta = 0 between them).
    """
    if not measurements:
        return []

    # Sort measurements by charttime
    measurements.sort(key=lambda m: m.charttime)
    
    min_time = measurements[0].charttime
    max_time = measurements[-1].charttime
    
    events: List[EventRecord] = []
    
    current_start = min_time
    while current_start <= max_time:
        current_end = current_start + window_span_hours
        
        # Filter measurements falling into [current_start, current_end)
        window_pts = [m for m in measurements if current_start <= m.charttime < current_end]
        
        if window_pts:
            mean_charttime = float(np.mean([m.charttime for m in window_pts]))
            max_storetime = max(
                (m.storetime for m in window_pts if m.storetime is not None),
                default=mean_charttime
            )
            
            # Group by vital_name
            by_vital: Dict[str, List[RawVitalMeasurement]] = {}
            for m in window_pts:
                by_vital.setdefault(m.vital_name, []).append(m)
                
            for vname, pts in by_vital.items():
                values = [p.value for p in pts]
                min_val = float(np.min(values))
                max_val = float(np.max(values))
                
                # Emit Min Vital Event
                events.append(
                    EventRecord(
                        subject_id=subject_id,
                        hadm_id=hadm_id,
                        source_table="vitals_agg_min",
                        source_row_id=f"{pts[0].source_row_id}_min",
                        charttime=mean_charttime,
                        storetime=max_storetime,
                        category_id=EventCategory.VITALS,
                        modality_id=Modality.NUMERIC,
                        has_specifics=True,
                        specifics_text=f"{vname}_min",
                        has_value=True,
                        value_raw=min_val,
                    )
                )
                
                # Emit Max Vital Event
                events.append(
                    EventRecord(
                        subject_id=subject_id,
                        hadm_id=hadm_id,
                        source_table="vitals_agg_max",
                        source_row_id=f"{pts[0].source_row_id}_max",
                        charttime=mean_charttime,
                        storetime=max_storetime,
                        category_id=EventCategory.VITALS,
                        modality_id=Modality.NUMERIC,
                        has_specifics=True,
                        specifics_text=f"{vname}_max",
                        has_value=True,
                        value_raw=max_val,
                    )
                )
                
        current_start += window_stride_hours

    return events
