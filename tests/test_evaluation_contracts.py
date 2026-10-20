"""Remediation gates use adversarial fixtures, not clinical score improvements."""
from dataclasses import asdict, replace
from pathlib import Path
import copy
import json
import subprocess
import sys

import pytest
import torch

from src.evaluation.contracts import ContractError, sha256_file, validate_manifest, validate_release, assert_comparable
from src.evaluation.landmarks import EventEvidence, LandmarkRequest, extract_landmark_state, validate_context
from src.evaluation.outcomes import Encounter, encounter_labels, readmission_label


def request(mode="strict_prospective"):
    return LandmarkRequest(1, 10, "event-b", 2., mode, "explicit_unknown_constant", 0., 0.,
                           {"unknown_gap": "synthetic fixture only; no project policy chosen"})


def evidence():
    return [EventEvidence("prefix", 1, None, 0., 0., "source", "prefix"),
            EventEvidence("event-a", 1, 10, 1., 1., "source", "clinical"),
            EventEvidence("event-b", 1, 10, 2., 2., "source", "clinical")]


def manifest_fixture(tmp_path, mode="strict_prospective", protocol="prospective_prediction"):
    for name, content in (("source.txt", "synthetic source identity"), ("model.bin", "synthetic model"),
                          ("splits.csv", "subject_id,split\n1,test\n2,train\n")):
        (tmp_path / name).write_text(content)
    record = {"representation_kind": "single_landmark", "landmark_request": asdict(request(mode)),
              "context_evidence": [asdict(e) for e in evidence()], "prediction": .2, "label": 0}
    (tmp_path / "predictions.jsonl").write_text(json.dumps(record) + "\n")
    m = dict(schema_version=1, run_id="synthetic-fixture", mode=mode, protocol=protocol,
             code={"commit": "0" * 40, "path": "source.txt", "source_sha256": sha256_file(tmp_path / "source.txt")},
             architecture={"revision": "fixture-v1", "parameter_count": 2, "module_counts": {"fixture": 2}},
             checkpoint={"path": "model.bin", "sha256": sha256_file(tmp_path / "model.bin"), "architecture_revision": "fixture-v1"},
             datasets=[{"name": "synthetic", "version": "fixture1", "manifest_sha256": "fixture", "encoder_recipe": "not applicable: synthetic values"}],
             patient_splits={"path": "splits.csv", "sha256": sha256_file(tmp_path / "splits.csv"), "unit": "patient"},
             temporal_cutoff={"definition": "fixture cutoff", "boundary": "inclusive", "gap_policy": "explicit_unknown_constant", "metadata_policy": "fixture as-of"},
             labels={"endpoint_id": "synthetic", "definition": "synthetic binary", "index_encounter_rule": "fixture explicit10", "horizon": "fixture", "censoring": "fixture complete"},
             availability_rules={"definition": "source", "unknown_policy": "reject"},
             probe_protocol={"definition": "fixture only", "training_population": "synthetic", "validation": "fixture", "seeds": [1]},
             predictions={"path": "predictions.jsonl", "sha256": sha256_file(tmp_path / "predictions.jsonl")},
             metric_protocol={"definition": "fixture", "code_sha256": "fixture", "uncertainty": "not applicable: fixture"},
             encoders=[], randomness={"seeds": [1], "policy": "fixture"},
             decisions={"synthetic": {"status": "APPROVED", "approval_or_evidence": "synthetic test fixture only"}}, deviations=["synthetic"], checks={})
    for name in ("split_integrity", "architecture_identity", "metric_replay", "endpoint_protocol", "probe_protocol", "landmark_identity", "no_future_events",
                 "no_full_record_pooling", "availability_provenance", "future_information_invariance", "metadata_cutoff", "gap_cutoff"):
        result = {"check": name, "status": "PASS", "source_sha256": m["code"]["source_sha256"],
                  "predictions_sha256": m["predictions"]["sha256"]}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(result))
        m["checks"][name] = dict(result, evidence_path=path.name, evidence_sha256=sha256_file(path))
    return m


def rewrite_predictions(tmp_path, manifest, mutate):
    path = tmp_path / "predictions.jsonl"
    r = json.loads(path.read_text())
    mutate(r)
    path.write_text(json.dumps(r) + "\n")
    manifest["predictions"]["sha256"] = sha256_file(path)


def test_release_requires_complete_manifest_and_real_artifacts(tmp_path):
    m = manifest_fixture(tmp_path)
    validate_release(m, "prospective", base=tmp_path)
    del m["availability_rules"]
    with pytest.raises(ContractError, match="manifest"):
        validate_release(m, "prospective", base=tmp_path)


@pytest.mark.parametrize("field", ["code", "architecture", "checkpoint", "datasets", "encoders", "randomness", "patient_splits", "temporal_cutoff", "labels", "availability_rules", "probe_protocol", "predictions", "metric_protocol"])
def test_each_required_manifest_field_is_mandatory(tmp_path, field):
    m = manifest_fixture(tmp_path)
    del m[field]
    with pytest.raises(ContractError):
        validate_manifest(m)


@pytest.mark.parametrize("status", ["FAIL", "SKIP", "UNKNOWN"])
def test_required_checks_cannot_be_skipped(tmp_path, status):
    m = manifest_fixture(tmp_path)
    m["checks"]["future_information_invariance"]["status"] = status
    with pytest.raises(ContractError, match="has not passed"):
        validate_release(m, "prospective", base=tmp_path)


def test_release_rejects_forged_passing_status_without_artifact(tmp_path):
    m = manifest_fixture(tmp_path)
    (tmp_path / "future_information_invariance.json").unlink()
    with pytest.raises(ContractError, match="Evidence file"):
        validate_release(m, "prospective", base=tmp_path)


def test_pooled_predictions_fail_even_when_manifest_claims_landmark(tmp_path):
    m = manifest_fixture(tmp_path)
    rewrite_predictions(tmp_path, m, lambda r: r.update(representation_kind="full_record_mean"))
    with pytest.raises(ContractError, match="pooled"):
        validate_release(m, "prospective", base=tmp_path)


def test_future_event_in_prediction_artifact_blocks_reporting(tmp_path):
    m = manifest_fixture(tmp_path)
    rewrite_predictions(tmp_path, m, lambda r: r["context_evidence"][1].update(occurred_at=10.))
    with pytest.raises(ContractError, match="future events"):
        validate_release(m, "prospective", base=tmp_path)


def test_retrospective_metrics_cannot_be_relabeled_prospective(tmp_path):
    m = manifest_fixture(tmp_path, "paper_reproduction", "landmark_retrieval")
    validate_release(m, "retrospective", base=tmp_path)
    with pytest.raises(ContractError, match="prospective reporting"):
        validate_release(m, "prospective", base=tmp_path)


def test_paper_mode_preserves_declared_time_information():
    r = replace(request("paper_reproduction"), gap_policy="full_record_next_gap", query_gap=None)
    validate_context(r, evidence())
    with pytest.raises(ContractError, match="Future next-gap"):
        validate_context(replace(r, mode="strict_prospective"), evidence())


def test_delayed_unknown_availability_and_future_metadata_rejected():
    e = evidence()
    e[-1] = replace(e[-1], available_at=3.)
    with pytest.raises(ContractError, match="unavailable"):
        validate_context(request(), e)
    e[-1] = replace(e[-1], available_at=None)
    with pytest.raises(ContractError, match="source availability"):
        validate_context(request(), e)
    with pytest.raises(ContractError, match="Demographic snapshot"):
        validate_context(replace(request(), metadata_snapshot_at=3.), evidence())


def test_encounter_specific_labels_and_boundary_policy():
    encounters = [Encounter(1, 10, 0., 72., False), Encounter(1, 11, 100., 300., True)]
    result = encounter_labels(encounters, [(1, 10)], los_threshold_hours=72., los_comparison=">=")
    assert result[(1, 10)] == {"mortality": False, "los": True, "duration_hours": 72.}
    assert encounter_labels(encounters, [(1, 10)], los_threshold_hours=72., los_comparison=">")[(1, 10)]["los"] is False
    assert encounter_labels([Encounter(1, 10, 0., None, None)], [(1, 10)], los_threshold_hours=72., los_comparison=">")[(1, 10)]["los"] is None


def test_readmission_rejects_overlap_and_preserves_censoring():
    with pytest.raises(ContractError, match="negative-gap"):
        readmission_label(10., [9.], horizon_hours=720., include_horizon=True, allow_zero_gap=False, observation_end=800.)
    assert readmission_label(10., [], horizon_hours=720., include_horizon=True, allow_zero_gap=False, observation_end=20.) is None
    assert readmission_label(10., [], horizon_hours=720., include_horizon=True, allow_zero_gap=False, observation_end=730.) is False
    assert readmission_label(10., [730.], horizon_hours=720., include_horizon=True, allow_zero_gap=False, observation_end=730.) is True


def test_no_missing_news2_imputation_or_availability_fabrication():
    from src.evaluation.probes import NEWS2Scorer
    from src.data.vitals import RawVitalMeasurement, aggregate_vitals_sliding_window
    with pytest.raises(ValueError, match="Five finite"):
        NEWS2Scorer.calculate()
    out = aggregate_vitals_sliding_window([RawVitalMeasurement(0., None, "fixture", 1., "a"),
                                          RawVitalMeasurement(1., None, "fixture", 2., "b")], 1)
    assert all(e.storetime is None for e in out)
    with pytest.raises(ValueError, match="positive"):
        aggregate_vitals_sliding_window([], 1, window_stride_hours=0)


def test_architecture_artifact_and_evidence_mixing_rejected(tmp_path):
    m = manifest_fixture(tmp_path)
    other = copy.deepcopy(m)
    other["architecture"]["revision"] = "different"
    other["checkpoint"]["architecture_revision"] = "different"
    with pytest.raises(ContractError, match="architecture"):
        assert_comparable([m, other])
    m["checks"]["split_integrity"]["source_sha256"] = "other-source"
    with pytest.raises(ContractError, match="different source"):
        validate_release(m, "prospective", base=tmp_path)


def test_legacy_evaluator_and_alpha_launch_fail_closed():
    from scripts.evaluate_clinical_probes import extract_patient_labels, extract_patient_representations_and_news2
    with pytest.raises(ContractError, match="Full-record pooling"):
        extract_patient_representations_and_news2(None, Path("unused"), "cpu")
    with pytest.raises(ContractError, match="Subject-wide"):
        extract_patient_labels("unused", "unused", "unused", [])
    proc = subprocess.run(["bash", "sbatch/run_alpha_ablation_grid.sh"], text=True, capture_output=True)
    assert proc.returncode == 78
    assert "BLOCKED" in proc.stderr


def test_release_cli_rejects_unmanifested_metrics(tmp_path):
    path = tmp_path / "old_metrics.json"
    path.write_text('{"mortality": {"auroc": 0.8}}')
    result = subprocess.run([sys.executable, "scripts/validate_experiment_release.py", str(path), "--claim", "prospective"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "RELEASE BLOCKED" in result.stderr


def test_extractor_returns_one_requested_state_and_rejects_future_before_forward():
    class Projector(torch.nn.Module):
        def forward(self, category, *args):
            return category.float().unsqueeze(-1).repeat(1, 1, 4)

    class Backbone(torch.nn.Module):
        def forward(self, e, positions, *args):
            return e + positions.unsqueeze(-1)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projector = Projector()
            self.backbone = Backbone()

    model = FakeModel()
    batch = dict(category_ids=torch.tensor([[1, 3, 8]]), modality_ids=torch.zeros(1, 3, dtype=torch.long),
                 positions=torch.tensor([[0., 1., 2.]]), ages=torch.zeros(1, 3),
                 delta_prev=torch.tensor([[0., 0., 1.]]), delta_next=torch.tensor([[0., 1., 0.]]),
                 attention_mask=torch.ones(1, 3, dtype=torch.bool), specifics_vec=torch.zeros(1, 3, 768),
                 has_specifics=torch.zeros(1, 3, dtype=torch.bool), value_vec=torch.zeros(1, 3, 768),
                 has_value=torch.zeros(1, 3, dtype=torch.bool))
    state = extract_landmark_state(model, batch, evidence(), request())
    torch.testing.assert_close(state, torch.full((4,), 10.))
    assert model.training  # inference restores the caller's state
    changed = evidence()
    changed[-1] = replace(changed[-1], occurred_at=3.)
    with pytest.raises(ContractError, match="future events"):
        extract_landmark_state(model, batch, changed, request())
    batch["delta_next"][0, 1] = 100
    with pytest.raises(ContractError, match="excluded event"):
        extract_landmark_state(model, batch, evidence(), request())


def test_paper_claim_rejects_candidate_parameter_count(tmp_path):
    m = manifest_fixture(tmp_path, "paper_reproduction", "landmark_retrieval")
    m["deviations"] = []
    m["decisions"] = {"fixture": {"status": "RESOLVED", "approval_or_evidence": "test fixture"}}
    with pytest.raises(ContractError, match="parameter count"):
        validate_release(m, "paper_faithful", base=tmp_path)


def test_reporting_cannot_promote_retrieval_or_overwrite_a_run(tmp_path):
    from src.evaluation.contracts import write_validated_report
    m = manifest_fixture(tmp_path, "paper_reproduction", "landmark_retrieval")
    path = tmp_path / "report.json"
    with pytest.raises(ContractError, match="prospective reporting"):
        write_validated_report(m, "prospective", {"auroc": .8}, base=tmp_path, output=path)
    assert not path.exists()
    write_validated_report(m, "retrospective", {"auroc": .8}, base=tmp_path, output=path)
    assert json.loads(path.read_text())["claim"] == "retrospective"
    with pytest.raises(ContractError, match="overwrite"):
        write_validated_report(m, "retrospective", {"auroc": .9}, base=tmp_path, output=path)


def test_loader_rejects_unversioned_and_conflicting_architectures(tmp_path):
    from scripts.evaluate_clinical_probes import load_model
    path = tmp_path / "legacy.pt"
    torch.save({"config": {"d_model": 768}, "model_state_dict": {}}, path)
    with pytest.raises(ContractError, match="legacy checkpoint"):
        load_model(path, {}, "cpu")
    torch.save({"config": {"architecture_revision": "2026-10-19", "d_model": 32}, "model_state_dict": {}}, path)
    with pytest.raises(ContractError, match="override checkpoint"):
        load_model(path, {"model": {"d_model": 768}}, "cpu")


def test_pending_methodology_blocks_release_despite_passing_component_checks(tmp_path):
    m = manifest_fixture(tmp_path)
    m["decisions"]["synthetic"]["status"] = "PENDING"
    with pytest.raises(ContractError, match="Unapproved or unresolved"):
        validate_release(m, "prospective", base=tmp_path)


def test_real_split_file_and_changed_checkpoint_bytes_are_checked(tmp_path):
    m = manifest_fixture(tmp_path)
    (tmp_path / "splits.csv").write_text("subject_id,split\n1,train\n1,test\n")
    m["patient_splits"]["sha256"] = sha256_file(tmp_path / "splits.csv")
    with pytest.raises(ContractError, match="cross-split"):
        validate_release(m, "prospective", base=tmp_path)
    (tmp_path / "splits.csv").write_text("subject_id,split\n1,train\n2,test\n")
    m["patient_splits"]["sha256"] = sha256_file(tmp_path / "splits.csv")
    (tmp_path / "model.bin").write_text("different checkpoint bytes")
    with pytest.raises(ContractError, match="checkpoint artifact"):
        validate_release(m, "prospective", base=tmp_path)


def test_unresolved_template_is_not_an_executable_scientific_default():
    template = json.loads(Path("configs/evaluation_manifest.template.json").read_text())
    with pytest.raises(ContractError):
        validate_manifest(template)


def test_news2_cannot_be_released_with_only_generic_component_tests(tmp_path):
    m = manifest_fixture(tmp_path, "paper_reproduction", "landmark_retrieval")
    m["labels"]["endpoint_id"] = "news2"
    with pytest.raises(ContractError, match="news2_"):
        validate_release(m, "retrospective", base=tmp_path)


def test_padding_is_excluded_before_projection():
    from src.models.noah import NOAH
    model = NOAH(d_model=32, d_latent=16, n_layers=1, n_heads=2, head_dim=16, dropout=0, attn_dropout=0)
    batch = dict(category_ids=torch.zeros(1, 3, dtype=torch.long), modality_ids=torch.zeros(1, 3, dtype=torch.long),
                 positions=torch.tensor([[0., 1., 2.]]), ages=torch.ones(1, 3),
                 delta_prev=torch.tensor([[0., 0., 1.]]), delta_next=torch.tensor([[0., 1., 0.]]),
                 attention_mask=torch.ones(1, 3, dtype=torch.bool), specifics_vec=torch.zeros(1, 3, 768),
                 has_specifics=torch.zeros(1, 3, dtype=torch.bool), value_vec=torch.zeros(1, 3, 768),
                 has_value=torch.zeros(1, 3, dtype=torch.bool))
    expected = extract_landmark_state(model, batch, evidence(), request())
    padded = {}
    for name, value in batch.items():
        tail = torch.zeros_like(value[:, :1])
        if value.dtype.is_floating_point:
            tail.fill_(float("nan"))
        padded[name] = torch.cat([value, tail], dim=1)
    actual = extract_landmark_state(model, padded, evidence(), request())
    torch.testing.assert_close(actual, expected)


def test_duplicate_legacy_rollout_cannot_run_directly():
    from scripts.verify_pilot_results import run_free_rollout_analysis
    with pytest.raises(ContractError, match="Duplicate legacy rollout"):
        run_free_rollout_analysis(None, [], "cpu")
