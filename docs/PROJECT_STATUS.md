# NOAH project status and next actions

Updated: 2026-10-14 (America/New_York). Code baseline: `30dd669325f99e4ce9fe554cae2934376f35f879`; documentation is maintained separately. Evidence: local files, paper verification and researcher-provided information. Server execution results await corresponding reports.

## 1. Goal and current position

The long-term goal is a scientifically faithful, test-driven, traceable NOAH reproduction, including paper-scale data, architecture and evaluation. The near-term goal is one complete small-scale real-data training, validation, checkpoint recovery and prior rollout using the fixed 1,000-patient candidate manifest. Verify paper-specific mechanisms before seeking paper-scale metrics.

Current stage: paper audit and partial foundation code are available; the complete data-to-training path is not connected.

| Component | Current status | Evidence |
|---|---|---|
| Architecture, traceability, assumptions and test plan | Documents and operational guide available | `REPRODUCTION_SPEC.md` and related documents; `IMPLEMENTATION_GUIDE.md` |
| Pilot candidates and split | Manifest contains 1,000 unique IDs; 700/150/150 | `data/pilot_cohort_manifest.csv` |
| Cohort selection | Code written: adult admission/ECG/CXR intersection, seed=42 | `src/data/subcohort.py`; enriched engineering cohort |
| Partial event extraction | Code written: admission/discharge/death, numeric labs, ECG/CXR indices | `src/data/extract_events.py` |
| Numeric, vital-sign and temporal modules; partial tests | Helpers and test definitions written; execution results await records | `numeric.py`, `vitals.py`, `temporal.py`, `tests/test_temporal_oracle.py` |
| Text, media decoding and frozen embeddings | Complete integration/cache artifacts not present in this repository | Index/path reading does not establish content extraction |
| Windows, prefixes and loss masks | Schema fields exist; construction/acceptance remain | No complete window builder |
| Complete NOAH, dual-path training and rollout | Implementation/integration remain | Model directory currently contains only temporal helpers |
| O2 compute platform | Researcher confirms access | Actual allocation and throughput require measurement |
| First complete server run | Completion evidence not available | GPU script has a placeholder; CPU entry reads two example patients |

These patient counts describe candidates, not the final cohort after complete event-eligibility filtering. The manifest does not determine the total server candidate pool. Presentation outputs remain local and ignored; source, documentation and reviewed summaries follow [REPOSITORY_POLICY.md](REPOSITORY_POLICY.md).

## 2. Current plan and scope decisions

| Item | Current plan |
|---|---|
| Near-term scale | Use the existing 1,000 candidates before expanding |
| Time estimate | Accepted working assumption: one primary engineer, usable assets/data/GPU, 4-6 weeks; not a measured training duration or an already-started countdown |
| Pilot modalities | Proposed categorical, numeric, text, CXR and ECG; ECHO timing remains a scope decision |
| Model scale | Proposed reduced width/layers/context for workflow validation; exact configuration not frozen |
| Required mechanisms | Directional temporal attention, causal content mask, conditional prior/informed posterior, shared latent, hierarchical decoding, dual reconstruction and KL, prior-only rollout |
| Full target | Six modalities, complete event coverage, paper architecture and evaluation; ECHO and larger compute arranged at that stage |
| Execution scope | Start the pilot at N-01/N-02/N-03 within the researcher's active task; paper-scale training has a separate scope/resource decision |

## 3. Prioritized next actions

These are pending actions. Technical steps are in [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

| Priority / ID | Next action | Completion evidence | Dependencies and execution location |
|---|---|---|---|
| P0 / N-01 | Produce a current server snapshot | Source commit/dirty state, actual repo/interpreter paths, data root/formats, aggregate sample-read results, existing checkpoints/job states | Server CLI read-only inventory; follow [report convention](../reports/README.md) |
| P0 / N-02 | Freeze initial pilot scope and minimum settings | Active modalities, event set, tiny parameters, time/target alignment, numeric basis and loss choices, each classified | Documentation/configuration work; material scope decided by researcher/PI |
| P0 / N-03 | Verify frozen encoder and source interfaces | BioLORD long-text policy, CXR provenance/processor, OTIS 1x24-compatible weights/pooling; a few real samples per modality | Available server assets; do not substitute width-100 OTIS defaults |
| P1 / N-04 | Complete and execute independent core references | Temporal/shift, mask ownership, Gaussian KL, shared latent/hierarchy, numeric basis and inference-boundary results | Mathematical/synthetic work does not require ECHO or the full clinical corpus |
| P1 / N-05 | Complete selected-event extraction, semantic cleaning, embeddings and windows | Two-patient real-content path/mask acceptance, then stage counts for 1,000 candidates | N-02/N-03; do not apply 64-event exclusions using incomplete readers |
| P1 / N-06 | Implement and connect complete tiny NOAH | Synthetic forward/backward, tiny-batch overfit, dual-path loss and prior-only rollout | Implementation task, N-02/N-04; can proceed independently of N-05 |
| P1 / N-07 | First complete real-data run | Training/validation, checkpoint save/recovery, prior rollout, component losses and resources | N-05/N-06 acceptance |
| P2 / N-08 | Decide expansion and resource requests | Natural-cohort design, ECHO path, complete event/model settings, measured time/storage/GPU budget | N-07; do not infer full cost from patient counts or script wall-time requests |

The main technical gaps are N-03/N-05/N-06 integration and implementation. Exact encoder files, directories and environments are dependencies to check, not established unavailable resources. Missing ECHO blocks the full six-modality target; its effect on the first pilot depends on agreed scope.

## 4. Acceptance of the first complete run

- Real content reading/encoding, patient-disjoint splits, windows and ownership are complete; exclusions/failures have counts and reasons.
- Core mathematical, temporal and causal tests pass; unknown final gaps are distinct from true zero gaps.
- The same run exercises posterior/prior reconstruction, KL and validation, recording components and annealing clocks.
- Checkpoints restore model, optimizer and progress, with a declared random-state/data-order recovery policy.
- Rollout generates event time/content using prior and history without target embeddings.
- Code/configuration/data-index/cache versions, job IDs/exit states, resources and test summaries are retained.

A completed job, successful forward pass or falling total loss alone does not meet these gates. Keep short-pilot configurations and metrics separate from the paper-scale target.

## 5. Maintaining this status

After server work, save the [report bundle](../reports/README.md), then update the date, evidence links, status and next action here. Use explicit states: code written, artifact generated, tests passed, run completed, pending or unknown.

Use `REPRODUCTION_PLAN.md` for the action plan, `IMPLEMENTATION_GUIDE.md` for technical steps and `ASSUMPTIONS.md` for scientific choices. Presentations summarize these sources; they are not the sole project record.

The [remote workflow](REMOTE_WORKFLOW.md) supports server CLI execution, GitHub synchronization of reviewed code/aggregate reports and local review. Direct SSH is optional.
