# NOAH_rep
This project aims to replicate the code structure and the results specified in NOAH

## Start here

For coding agents, read [AGENTS.md](AGENTS.md) before changing the repository.

Repository documentation, code annotations and execution reports are maintained in English for collaborators. The [MIMIC data guide](docs/MIMIC_DATA_GUIDE.md) introduces the source data and preparation workflow.

Read [current project status and next actions](docs/PROJECT_STATUS.md) first to distinguish completed artifacts, code in progress and pending execution. The [reproduction plan](docs/REPRODUCTION_PLAN.md) defines the current 1,000-patient pilot milestones and the later paper-scale target. For work on O2, follow the [remote workflow](docs/REMOTE_WORKFLOW.md) and [server report convention](reports/README.md).

The [technical implementation guide](docs/IMPLEMENTATION_GUIDE.md) explains the actual reproduction steps, paper-specified feature extraction, verified upstream encoder interfaces, outstanding configuration choices and acceptance checks. It complements the [full specification](docs/REPRODUCTION_SPEC.md), [paper traceability](docs/PAPER_TRACEABILITY.md), [assumption register](docs/ASSUMPTIONS.md), [test plan](docs/TEST_PLAN.md) and [implementation schedule](docs/REPRODUCTION_PLAN.md).

The [architecture audit](docs/ARCHITECTURE_AUDIT.md) connects paper requirements to current code, fixes, tests and unresolved fidelity gates.

The supplied paper is [paper/NOAH.pdf](paper/NOAH.pdf). Existing pilot experiments do not establish exact reproduction. Follow the current fidelity and integration gates in project status before expansion. Implementation and execution follow the researcher's active task; paper-scale training is a separate scope and resource decision.

Repository contents, ignored outputs and publishing checks follow [REPOSITORY_POLICY.md](docs/REPOSITORY_POLICY.md). Documentation describes the current state; Git retains version history.


Controlled remediation: [all15 findings, methodology decisions and execution order](docs/ARCHITECTURE_AUDIT.md#controlled-remediation-preserved-evidence-and-revision-separation). Historical/current checkpoint families are preserved separately; the alpha-sweep and legacy pooled clinical reporting are on hold. New experiment manifests and release gates are defined in [reports/README.md](reports/README.md#experiment-manifests-and-scientific-release).
