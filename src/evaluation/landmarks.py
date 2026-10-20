"""Single-state extraction with explicit temporal evidence and no pooling fallback."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from src.evaluation.contracts import ContractError, MODES


@dataclass(frozen=True)
class EventEvidence:
    event_id: str
    subject_id: int
    encounter_id: int | None
    occurred_at: float
    available_at: float | None
    availability_source: str
    kind: str  # clinical or demographic prefix


@dataclass(frozen=True)
class LandmarkRequest:
    subject_id: int
    encounter_id: int
    event_id: str
    cutoff: float
    mode: str
    gap_policy: str
    query_gap: float | None
    metadata_snapshot_at: float
    # This mapping holds explicitly chosen policies; missing decisions are errors.
    decisions: dict[str, Any]


def validate_context(request: LandmarkRequest, evidence: list[EventEvidence]) -> None:
    if request.mode not in MODES or not math.isfinite(request.cutoff):
        raise ContractError("An explicit evaluation mode and finite cutoff are required")
    if not request.decisions or any(v is None for v in request.decisions.values()):
        raise ContractError("Landmark policies must be supplied explicitly")
    if not math.isfinite(request.metadata_snapshot_at) or request.metadata_snapshot_at > request.cutoff:
        raise ContractError("Demographic snapshot contains future information")
    ids = [e.event_id for e in evidence]
    if len(set(ids)) != len(ids):
        raise ContractError("Duplicate source event identities in the prepared context")
    clinical = []
    for event in evidence:
        if event.subject_id != request.subject_id or not event.event_id:
            raise ContractError("Mixed patients or missing event provenance")
        if event.kind not in {"clinical", "prefix"}:
            raise ContractError("Unknown evidence kind")
        if not math.isfinite(event.occurred_at) or event.occurred_at > request.cutoff:
            raise ContractError("Landmark context contains future events")
        if event.available_at is not None and not math.isfinite(event.available_at):
            raise ContractError("Nonfinite source availability")
        if request.mode == "strict_prospective":
            if event.available_at is None or event.availability_source != "source":
                raise ContractError("Prospective evaluation requires observed source availability")
            if event.available_at > request.cutoff:
                raise ContractError("An event was unavailable at the prediction cutoff")
        if event.kind == "clinical":
            clinical.append(event)
    if any(a.occurred_at > b.occurred_at for a, b in zip(clinical, clinical[1:])):
        raise ContractError("Clinical evidence is not chronological")
    if not clinical or clinical[-1].event_id != request.event_id:
        raise ContractError("The requested landmark must be the last clinical event in its view")
    if clinical[-1].encounter_id != request.encounter_id:
        raise ContractError("Landmark and index encounter disagree")
    if request.gap_policy not in {"full_record_next_gap", "explicit_query_horizon", "explicit_unknown_constant"}:
        raise ContractError("Unspecified next-gap policy")
    if request.mode == "strict_prospective" and request.gap_policy == "full_record_next_gap":
        raise ContractError("Future next-gap information is forbidden in prospective mode")
    if request.gap_policy != "full_record_next_gap":
        if request.query_gap is None or not math.isfinite(request.query_gap) or request.query_gap < 0:
            raise ContractError("The approved unknown-gap/query-horizon value must be explicit")


def extract_landmark_state(model, batch: dict, evidence: list[EventEvidence], request: LandmarkRequest):
    """Use only the supplied audited view; reject pooled or provenance-free old shards."""
    import torch
    validate_context(request, evidence)
    mask = batch["attention_mask"]
    if mask.ndim != 2 or mask.shape[0] != 1 or mask.dtype != torch.bool:
        raise ContractError("One patient with a boolean attention mask is required")
    n = int(mask.sum().item())
    if len(evidence) != n or not mask[0, :n].all() or mask[0, n:].any():
        raise ContractError("Every valid token needs ordered evidence, including prefix tokens")
    required_inputs = ("category_ids", "modality_ids", "positions", "ages", "delta_prev", "delta_next",
                       "specifics_vec", "has_specifics", "value_vec", "has_value", "attention_mask")
    if any(batch[key].shape[:2] != mask.shape for key in required_inputs):
        raise ContractError("All token arrays must align with the evidenced attention mask")
    # Padding carries no information and is never projected as part of a landmark view.
    batch = {key: batch[key][:, :n] for key in required_inputs}
    mask = batch["attention_mask"]
    if request.mode == "strict_prospective":
        clinical_indices = [i for i, item in enumerate(evidence) if item.kind == "clinical"]
        for before, after in zip(clinical_indices, clinical_indices[1:]):
            expected = evidence[after].occurred_at - evidence[before].occurred_at
            if not math.isclose(batch["delta_next"][0, before].item(), expected, rel_tol=1e-6, abs_tol=1e-7):
                raise ContractError("A retained event next gap references an excluded event")
            if not math.isclose(batch["delta_prev"][0, after].item(), expected, rel_tol=1e-6, abs_tol=1e-7):
                raise ContractError("Prepared backward gaps do not match retained event timestamps")
    if request.gap_policy != "full_record_next_gap":
        supplied_gap = batch["delta_next"][0, n - 1].item()
        if not math.isclose(supplied_gap, request.query_gap, rel_tol=1e-6, abs_tol=1e-7):
            raise ContractError("Prepared query gap differs from the declared policy")
    for name in ("positions", "ages", "delta_prev", "delta_next", "specifics_vec", "value_vec"):
        if not torch.isfinite(batch[name][:, :n]).all():
            raise ContractError(f"Nonfinite valid inputs: {name}")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            e = model.projector(batch["category_ids"], batch["modality_ids"], batch["specifics_vec"],
                                batch["has_specifics"], batch["value_vec"], batch["has_value"])
            h = model.backbone(e, batch["positions"], batch["ages"], batch["delta_next"],
                               batch["delta_prev"], mask)
            # The state has one event identity; no averaging over states or windows is permitted.
            state = h[0, n - 1].detach().clone()
            if not torch.isfinite(state).all():
                raise ContractError("Landmark output is nonfinite")
    finally:
        model.train(was_training)
    return state


def validate_prediction_record(record: dict, mode: str) -> None:
    """Reporting repeats the context checks on the prediction artifact's provenance."""
    if record.get("representation_kind") != "single_landmark":
        raise ContractError("Landmark evaluation cannot use pooled full-record representations")
    try:
        request = LandmarkRequest(**record["landmark_request"])
        evidence = [EventEvidence(**e) for e in record["context_evidence"]]
    except (KeyError, TypeError) as exc:
        raise ContractError("Prediction artifact lacks landmark provenance") from exc
    if request.mode != mode:
        raise ContractError("Prediction artifact and manifest evaluation modes differ")
    validate_context(request, evidence)
    for key in ("prediction", "label"):
        if key not in record or not isinstance(record[key], (int, float)) or not math.isfinite(record[key]):
            raise ContractError(f"Scored prediction requires a finite {key}")
