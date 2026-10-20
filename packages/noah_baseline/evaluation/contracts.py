"""Experiment and release evidence contracts; no scientific policies are inferred."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """An evaluation lacks required evidence or violates its declared information set."""


MODES = {"paper_reproduction", "strict_prospective"}
PROTOCOLS = {"landmark_retrieval", "prospective_prediction", "teacher_forced_reconstruction",
             "free_rollout", "oracle_time_controlled", "zero_shot_mc"}
REQUIRED = {"schema_version", "run_id", "mode", "protocol", "code", "architecture", "checkpoint", "encoders", "randomness",
            "datasets", "patient_splits", "temporal_cutoff", "labels", "availability_rules",
            "probe_protocol", "predictions", "metric_protocol", "decisions", "deviations", "checks"}


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fields(value: Any, fields: set[str], name: str) -> None:
    if not isinstance(value, dict) or not fields <= value.keys():
        raise ContractError(f"{name} requires fields {sorted(fields)}")
    for field in fields:
        if value[field] is None or value[field] == "":
            raise ContractError(f"{name}.{field} is unavailable")


def validate_manifest(manifest: dict) -> None:
    """Require explicit provenance; passing this schema alone does not certify a claim."""
    _fields(manifest, REQUIRED, "manifest")
    if manifest["schema_version"] != 1 or manifest["mode"] not in MODES:
        raise ContractError("Unsupported schema version or evaluation mode")
    if manifest["protocol"] not in PROTOCOLS:
        raise ContractError("Unknown protocol; legacy pooled evaluation is not a landmark protocol")
    for name, fields in {
        "code": {"commit", "source_sha256", "path"},
        "architecture": {"revision", "parameter_count", "module_counts"},
        "checkpoint": {"path", "sha256", "architecture_revision"},
        "patient_splits": {"path", "sha256", "unit"},
        "temporal_cutoff": {"definition", "boundary", "gap_policy", "metadata_policy"},
        "labels": {"endpoint_id", "definition", "index_encounter_rule", "horizon", "censoring"},
        "availability_rules": {"definition", "unknown_policy"},
        "probe_protocol": {"definition", "training_population", "validation", "seeds"},
        "predictions": {"path", "sha256"},
        "metric_protocol": {"definition", "code_sha256", "uncertainty"},
        "randomness": {"seeds", "policy"},
    }.items():
        _fields(manifest[name], fields, name)
    if not isinstance(manifest["datasets"], list) or not manifest["datasets"]:
        raise ContractError("Source dataset versions and encoder recipes are required")
    for dataset in manifest["datasets"]:
        _fields(dataset, {"name", "version", "manifest_sha256", "encoder_recipe"}, "dataset")
    if not isinstance(manifest["encoders"], list):
        raise ContractError("Encoder identifiers must be an explicit list (empty only for a declared no-encoder scope)")
    for encoder in manifest["encoders"]:
        _fields(encoder, {"model_id", "revision", "checkpoint_sha256", "recipe_sha256", "output_selection"}, "encoder")
    if not isinstance(manifest["randomness"]["seeds"], list) or not manifest["randomness"]["seeds"]:
        raise ContractError("Explicit random seeds are required")
    if manifest["patient_splits"]["unit"] != "patient":
        raise ContractError("Patient-level split membership is required")
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["code"]["commit"]):
        raise ContractError("A complete code commit is required; dirty changes need their source artifact")
    architecture = manifest["architecture"]
    if architecture["revision"] != manifest["checkpoint"]["architecture_revision"]:
        raise ContractError("Checkpoint and architecture revisions differ")
    if sum(architecture["module_counts"].values()) != architecture["parameter_count"]:
        raise ContractError("Module counts do not reconcile to the architecture total")
    if not isinstance(manifest["checks"], dict) or not isinstance(manifest["decisions"], dict):
        raise ContractError("Checks and methodology decisions must be explicit mappings")
    if not isinstance(manifest["deviations"], list):
        raise ContractError("Deviations must be an explicit list, including when empty")
    if manifest["mode"] == "strict_prospective":
        if manifest["protocol"] in {"landmark_retrieval", "teacher_forced_reconstruction", "oracle_time_controlled"}:
            raise ContractError("Retrieval, posterior reconstruction and oracle timing cannot be prospective protocols")
        if manifest["temporal_cutoff"]["gap_policy"] == "full_record_next_gap":
            raise ContractError("True future next gaps are forbidden in prospective mode")


def validate_split_manifest(path: Path) -> None:
    """Patient membership is unique and explicit; no default train assignment."""
    import csv
    seen = set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"subject_id", "split"} <= set(reader.fieldnames):
            raise ContractError("Split artifact needs subject_id and split columns")
        for row in reader:
            patient = row["subject_id"]
            if not patient or patient in seen or row["split"] not in {"train", "val", "test"}:
                raise ContractError("Duplicate, cross-split, empty or unassigned patient membership")
            seen.add(patient)
    if not seen:
        raise ContractError("Patient split manifest is empty")


def verify_artifacts(manifest: dict, base: Path) -> None:
    """Verify actual files, not just declared artifact identifiers."""
    code_path = base / manifest["code"]["path"]
    if not code_path.is_file() or sha256_file(code_path) != manifest["code"]["source_sha256"]:
        raise ContractError("Source snapshot missing or checksum mismatch")
    for name in ("checkpoint", "patient_splits", "predictions"):
        item = manifest[name]
        path = base / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ContractError(f"{name} artifact missing or checksum mismatch")
    validate_split_manifest(base / manifest["patient_splits"]["path"])


def check_evidence(manifest: dict, name: str) -> None:
    """Required checks must bind a passing artifact to this source and predictions."""
    item = manifest["checks"].get(name)
    if not isinstance(item, dict) or item.get("status") != "PASS":
        raise ContractError(f"Required check {name} has not passed")
    for field, expected in (("source_sha256", manifest["code"]["source_sha256"]),
                            ("predictions_sha256", manifest["predictions"]["sha256"])):
        if item.get(field) != expected:
            raise ContractError(f"Check {name} belongs to different source or predictions")
    if not item.get("evidence_path") or not item.get("evidence_sha256"):
        raise ContractError(f"Check {name} lacks an execution artifact")


def validate_release(manifest: dict, claim: str, *, base: Path) -> None:
    """Fail closed for unsupported labels; never promote retrospective metrics."""
    validate_manifest(manifest)
    verify_artifacts(manifest, base)
    if not manifest["decisions"]:
        raise ContractError("A methodology decision record is required")
    for decision in manifest["decisions"].values():
        if not isinstance(decision, dict) or decision.get("status") not in {"APPROVED", "RESOLVED", "PAPER_EXPLICIT", "DERIVED"}:
            raise ContractError("Unapproved or unresolved methodology prevents release")
        if not decision.get("approval_or_evidence"):
            raise ContractError("Methodology requires an approval or paper evidence reference")
    required = {"split_integrity", "architecture_identity", "metric_replay", "endpoint_protocol"}
    if manifest["protocol"] in {"landmark_retrieval", "prospective_prediction"}:
        required |= {"landmark_identity", "no_future_events", "no_full_record_pooling", "probe_protocol"}
    if manifest["labels"]["endpoint_id"] == "news2":
        required |= {"news2_window_labels", "news2_scoring_oracle", "news2_patient_splits", "news2_probe_protocol"}
    if manifest["protocol"] in {"landmark_retrieval", "prospective_prediction"}:
        from noah_baseline.evaluation.landmarks import validate_prediction_record
        records_path = base / manifest["predictions"]["path"]
        found = False
        with records_path.open() as stream:
            for line in stream:
                if line.strip():
                    validate_prediction_record(json.loads(line), manifest["mode"])
                    found = True
        if not found:
            raise ContractError("No scored landmark predictions")
    if claim == "prospective":
        if manifest["mode"] != "strict_prospective" or manifest["protocol"] not in {
                "prospective_prediction", "free_rollout", "zero_shot_mc"}:
            raise ContractError("This protocol cannot support prospective reporting")
        required |= {"availability_provenance", "future_information_invariance", "metadata_cutoff", "gap_cutoff"}
    elif claim == "paper_faithful":
        if manifest["mode"] != "paper_reproduction" or manifest["deviations"]:
            raise ContractError("Paper-faithful reporting cannot omit or mix declared deviations")
        if any(d.get("status") != "RESOLVED" for d in manifest["decisions"].values()):
            raise ContractError("Unresolved methodology prevents a paper-faithful claim")
        if manifest["architecture"]["parameter_count"] != 150_322_970:
            raise ContractError("Architecture parameter count differs from paper Table 16")
        required |= {"paper_conformance", "encoder_provenance", "endpoint_protocol"}
    elif claim == "retrospective":
        if manifest["protocol"] != "landmark_retrieval":
            raise ContractError("Retrospective retrieval requires an explicit retrieval protocol")
    elif claim == "reconstruction":
        if manifest["protocol"] != "teacher_forced_reconstruction":
            raise ContractError("Reconstruction diagnostics require the matching protocol")
    else:
        raise ContractError("Unsupported scientific claim")
    for name in required:
        check_evidence(manifest, name)
        item = manifest["checks"][name]
        path = base / item["evidence_path"]
        if not path.is_file() or sha256_file(path) != item["evidence_sha256"]:
            raise ContractError(f"Evidence file for {name} is missing or changed")
        evidence = json.loads(path.read_text())
        if evidence.get("check") != name or evidence.get("status") != "PASS":
            raise ContractError(f"Evidence content for {name} does not certify this check")
        for key in ("source_sha256", "predictions_sha256"):
            if evidence.get(key) != item[key]:
                raise ContractError(f"Evidence content for {name} belongs to another run")


def assert_comparable(manifests: list[dict]) -> None:
    """Reject silent aggregation across architectures, labels or information protocols."""
    if not manifests:
        raise ContractError("No experiment manifests supplied")
    keys = ("mode", "protocol", "architecture", "datasets", "patient_splits", "temporal_cutoff",
            "labels", "availability_rules", "probe_protocol", "metric_protocol")
    first = manifests[0]
    for other in manifests:
        validate_manifest(other)
        for key in keys:
            # Seed changes are allowed only as an explicit comparison decision in later tooling.
            if other[key] != first[key]:
                raise ContractError(f"Cannot aggregate experiments with differing {key}")


def write_validated_report(manifest: dict, claim: str, metrics: dict, *, base: Path, output: Path) -> None:
    """The reporting boundary validates artifacts and the requested scientific interpretation."""
    validate_release(manifest, claim, base=base)
    if output.exists():
        raise ContractError("Refusing to overwrite an existing experiment report")
    if not isinstance(metrics, dict) or not metrics:
        raise ContractError("A nonempty metric mapping is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps({"claim": claim, "experiment": manifest, "metrics": metrics}, indent=2, allow_nan=False)
    with output.open("x") as stream:
        stream.write(serialized + "\n")
