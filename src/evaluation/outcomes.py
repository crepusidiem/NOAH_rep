"""Encounter-level label primitives. Callers must approve and supply endpoint policies."""
from dataclasses import dataclass
import math
from src.evaluation.contracts import ContractError


@dataclass(frozen=True)
class Encounter:
    subject_id: int
    encounter_id: int
    admitted_at: float
    ended_at: float | None
    mortality: bool | None

    def validate(self):
        if not math.isfinite(self.admitted_at):
            raise ContractError("Missing admission timestamp")
        if self.ended_at is not None and (not math.isfinite(self.ended_at) or self.ended_at < self.admitted_at):
            raise ContractError("Encounter ends before admission or has invalid time")
        if self.mortality is not None and type(self.mortality) is not bool:
            raise ContractError("Mortality must be an observed boolean or unknown")


def encounter_labels(encounters: list[Encounter], selected: list[tuple[int, int]], *,
                     los_threshold_hours: float, los_comparison: str) -> dict:
    """No automatic last/first/maximum selection, censor imputation, or endpoint threshold."""
    if not math.isfinite(los_threshold_hours) or los_threshold_hours < 0 or los_comparison not in {">", ">="}:
        raise ContractError("Explicit LOS threshold and comparison are required")
    indexed = {}
    for e in encounters:
        e.validate()
        key = (e.subject_id, e.encounter_id)
        if key in indexed:
            raise ContractError("Duplicate encounter identity")
        indexed[key] = e
    if len(set(selected)) != len(selected):
        raise ContractError("Duplicate selected encounter")
    result = {}
    for key in selected:
        if key not in indexed:
            raise ContractError("Requested encounter is absent")
        e = indexed[key]
        duration = None if e.ended_at is None else e.ended_at - e.admitted_at
        los = None if duration is None else (duration >= los_threshold_hours if los_comparison == ">=" else duration > los_threshold_hours)
        result[key] = {"mortality": e.mortality, "los": los, "duration_hours": duration}
    return result


def readmission_label(discharge_at: float, qualifying_admissions: list[float], *,
                      horizon_hours: float, include_horizon: bool, allow_zero_gap: bool,
                      observation_end: float | None) -> bool | None:
    """Callers preselect qualifying admissions under an approved transfer/planning policy."""
    if not math.isfinite(discharge_at) or not math.isfinite(horizon_hours) or horizon_hours <= 0:
        raise ContractError("Finite discharge and positive readmission horizon are required")
    for time in qualifying_admissions:
        if not math.isfinite(time) or time < discharge_at:
            raise ContractError("Overlapping/negative-gap encounters cannot be readmissions")
        if time == discharge_at and not allow_zero_gap:
            raise ContractError("Zero-gap encounter needs an explicit transfer/exclusion decision")
    if observation_end is not None and (not math.isfinite(observation_end) or observation_end < discharge_at):
        raise ContractError("Invalid observation endpoint")
    horizon_end = discharge_at + horizon_hours
    positive = any(t <= horizon_end if include_horizon else t < horizon_end for t in qualifying_admissions)
    if positive:
        if observation_end is not None and any(t > observation_end for t in qualifying_admissions if t <= horizon_end):
            raise ContractError("Readmission falls beyond the declared observation endpoint")
        return True
    return False if observation_end is not None and observation_end >= horizon_end else None
