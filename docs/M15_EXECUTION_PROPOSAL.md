# M1.5 bounded learning experiment

Status: researcher-approved2026-10-20; controller and inexpensive verification implemented. Actual execution evidence is recorded separately from this fixed protocol. This is a reduced-modality, occurrence-time engineering experiment; the existing paper specification and assumptions remain authoritative. The experiment tests learning of this restricted event stream, not paper fidelity or clinical efficacy.

## Fixed inputs and preservation

Use `development/clean-baseline`, baseline source commit `f5e0f58c6f414c191385a1eb6e0e062d08387b93`, and the immutable M1 run `artifacts/reduced_pilot/runs/m1-real-20261019-02/`. The read-only preflight verifies all 32 inventoried M1 artifacts against SHA256 and checks their write bits. It reads model targets only in train/validation; test windows are not evaluated. Reference M1 windows and manifests in place without copying checkpoints or data. Outputs go exclusively to `artifacts/reduced_pilot/m15/<new-run-id>/`; existing paths must not be overwritten.

The proposed configuration is [configs/m15.json](../configs/m15.json). Hashes pin windows, cohort, splits, source recipe, M1 configuration, source snapshot and checkpoint. The M1 checkpoint is preserved as evidence, not a warm start: initialize the identical candidate afresh with seed42 and train-fitted weights. The executed M1 source snapshot remains immutable even when new controller code is committed. At execution, save the new commit, dirty-source inventory and source archive; require model/loss/data module identity with M1 before starting.

The verified source cohort has1000 patients with original700/150/150 patient splits. The same first-encounter/+24h eligibility yields743 windows:526 train,102 validation,115 quarantined test. Do not add excluded patients or later encounters. All windows use categorical hospital/ED boundaries and numeric ED heart rate, six fixed demographic prefixes, context64, overlap16, and the same provenance, availability, cutoff and gap policies. Unknown availability stays unknown. All actual sequences fit one window.

| Owned shifted targets | Train | Validation |
|---|---:|---:|
| All categories |2335|436|
| Enter Hospitalization |526|102|
| Leave ED |287|55|
| Enter ED |296|56|
| Vitals; numeric44-D heart-rate targets |1226|223|
| Observed outgoing gap: positive |1784|330|
| Observed outgoing gap: zero |25|4|
| Unknown outgoing gap; excluded from time scores |526|102|

Positive gaps are hours. Train median/p95/max:0.78333/3.91417/22.53333; validation:0.81667/4.66833/8.48333. Effective attention lengths including prefixes are7–28 train and7–22 validation. Both partitions use identical target ownership, prefix/overlap/padding exclusion and time masks; they have different empirical category/gap distributions, not artificially balanced targets. Train/validation owned-target ledger hashes are pinned in the configuration. Predeclare common categories as at least100 train targets and rare as1–99. All four observed categories are common; rare and unseen-category performance is not estimable here. Report this limitation rather than manufacturing a subgroup.

## Training budget and unchanged candidate

The candidate has4,760,468 trainable parameters: projector792,832; backbone3,157,504; prior131,584; posterior197,120; decoder481,428. Width256, latent128, four layers/four heads/head64 and all inactive output heads remain unchanged. No architectural pruning or new modalities.

Propose20 complete epochs:263 optimizer updates per epoch with two one-window microbatches, giving5260 retained optimizer updates and10,520 consumed windows. Preserve fixed M1 source order, without reshuffling or rebalancing. Seed42; float32 CPU; two Torch threads; deterministic algorithms. Muon/AdamW routing and settings remain those in `models/optimization.py`; LR0.0006, weight decay0.01, gradient clipping10. Preserve the M1 mean-of-microbatch objectives and train-only `(count+1)^(-0.5)` mean-one weights across the full31/6/2 vocabularies. Do not refit weights on validation or change alpha.

LR linearly warms over1578 optimizer updates (six epochs), then stays constant as in M1. KL beta is the existing consumed-window clock, increasing to1 at the start of epoch11; no clock resets on recovery. Preserve `.3 * posterior reconstruction + beta * KL + .1 * prior reconstruction` and every current component reduction. The proposed longer budget is an engineering choice, not an author-specified schedule. A separate M1.5 controller will reuse tested operations and recovery format; the existing M1 sixteen-update guard remains intact.

Approved budget: one CPU process, two allocated CPU cores,16GiB RAM, maximum60 minutes elapsed, no GPU, and at most2GiB new output. Estimate10–30 minutes, not a measured training duration. The preflight measured64 inference-only windows at7.07ms mean on two threads, approximately1003MiB peak process RSS. Training backward/optimizer overhead is unmeasured; using a4–8-fold forward-cost allowance plus evaluation/IO motivates the conservative cap. No SLURM job has been submitted. An interrupted/time-capped run is incomplete, not an excuse to extend this budget silently.

## Evaluation and checkpoint selection

Evaluate initialization and each completed epoch:21 prespecified checkpoints. Evaluate all526 train and102 validation windows, always with dropout disabled and identical target ledgers. Validation does not call backward/step or update fitted statistics. Preserve training RNG around evaluation. Evaluation seed1729 determines fixed per-window/per-draw random numbers independent of checkpoint and dataset traversal; record the generator algorithm and stable ordinal mapping. Save these identities and predictions so scores can be replayed independently.

Primary selection: smallest **validation unweighted, event-mean prior category NLL**, with earliest epoch breaking exact ties. The category probabilities are the average of eight softmax outputs from fixed prior latent draws conditioned only on shifted history. Do not average logits or use the target-informed posterior for this score. This integrates event-level latent uncertainty; it is not Monte Carlo outcome forecasting. Initialization is eligible, and a selected initialization must be reported as no demonstrated training benefit. No selection using mortality, test events, test losses or test labels. No adaptive hyperparameter search or early stopping on validation improvements.

This is teacher-forced, time-conditioned next-event evaluation under the unchanged temporal inputs: previous-event next-gap features reveal the retained target arrival time. It does not establish unconditional free-rollout prediction. No events after the landmark are used; the landmark's final next gap is unknown. Posterior reconstruction is a separate diagnostic and must never be described as predictive performance. Preserve this distinction in figures, manifests and release checks.

Report the following at initialization and every epoch:

- Train optimization traces and full train/validation evaluation curves for posterior reconstruction, prior reconstruction, unannealed KL, category, modality, specifics presence/vector, numeric value, time gate and positive time regression. Record both actual-beta total and fixed-beta1 reference total so annealing cannot masquerade as learning. Evaluate each branch's components independently through the existing reconstruction helper without changing the model return contract. Objective curves are means of window objectives, matching the candidate's reductions; predictive metrics are event means with denominators reported.
- Prior category NLL, accuracy, per-category support/recall/F1, macro-F1 over the fixed four training-supported categories, and common/rare eligibility. Keep all31 logits; do not mask unobserved classes to improve scores. Save posterior reconstruction scores separately. Report prior numeric44-D MSE against the train-fitted vector-mean baseline as an additional event-structure diagnostic; do not claim raw-bpm forecast accuracy from Fourier-space MSE.
- Outgoing time gate log loss/Brier on `owned_target & has_next_gap`, positive log1p-hours RMSE on its strictly positive subset, and hour MAE on both all observed gaps and positive gaps. For fixed prior draws, average gate probabilities and regression log outputs separately; invert the mean regressor with `expm1(max(log_prediction,0))` and apply the0.5 hard gate for the hour point prediction. This is a declared diagnostic point estimate, not the predictive mean of a continuous distribution. Preserve the loss's un-clamped regression output for log-space RMSE. Report real zero and unknown supports separately; unknown gaps never become negatives. Reject nonfinite inverse transforms rather than silently clipping.
- Preclip gradient norm per update, clipping frequency, learning rate, beta, parameter/optimizer finiteness, and train/validation divergence. Flag increasing validation NLL over three successive epochs alongside falling train NLL as possible overfitting; do not change the budget or optimizer in response.

Fit all baselines on the same owned training targets only, once, and freeze:

1. Unigram category probabilities `(count+1)/(2335+31)`; predict the largest probability for accuracy. Evaluate with the same31-category support and NLL denominators as the model.
2. Previous-category transition frequencies with one unigram-distributed pseudocount per row: `(transition_count + unigram_probability)/(row_count+1)`. The immediate previous token's category, including the fixed prefix category for the first event, is available to both models. Empty rows back off exactly to the unigram. This stronger comparator detects trivial encounter-boundary/heart-rate repetition learning. No tuning of smoothing using validation.
3. Time gate: training observed positive fraction1784/1809, including observed zero gaps. Positive log regressor: train mean log1p-hours0.6560061887; apply the same inverse/hard-gate rule for hour MAE. Also report a train positive-gap median constant as a metric-specific MAE reference, clearly separate from the log-optimal constant. Numeric: mean training44-D target vector, never validation-fitted.

Claim evidence of restricted clinical event structure only if held-out prior scores improve over initialization and the comparator evidence supports it. Beating the unigram alone does not establish contextual learning if the transition baseline remains stronger. Report inconclusive or negative outcomes without adjusting the experiment. Mortality probing is deferred from this feasibility budget; the accepted M1 probe remains preserved. Any later exploratory probe must retain its fixed encounter/landmark and development-only selection protocol.

## Artifacts, verification and stopping

Before substantial execution, implement and run inexpensive fixtures for train-only fitting/gradient participation, exact train/validation target ledgers and masks, independent metric reduction, fixed latent draws, evaluation RNG preservation and unchanged M1 operation equivalence. Verify one bounded optimizer update against the M1 reference in an isolated temporary artifact namespace only after budget approval. Do not weaken M1's safety guard.

Save initialization and all20 epoch checkpoints containing strict architecture/config/data identities, model and optimizer states, weights, clocks, cursor and Python/NumPy/Torch RNGs. Save resolved configuration, source snapshot/commit/file hashes, environment, resource allocation, split/cohort/source manifest references, target ownership ledgers, baseline fitted counts/statistics, per-update histories, per-checkpoint component histories and target-level predictions, metric definitions and selected-checkpoint receipt. Numeric vectors and all patient-linked artifacts stay ignored/private. Public reports contain aggregate evidence only. Serialize numeric predictions losslessly; compute/replay metrics in float64.

At epoch1 perform an exact checkpoint reload and one next-update recovery comparison against the uninterrupted branch, checking model, optimizer, clocks and RNG state bitwise. The duplicate verification update is not retained in the training trajectory; the cap includes at most5261 physical optimizer calls. At initialization, epoch1, selected and final checkpoint, verify identical repeated predictions and metric replay; reload the selected checkpoint in a fresh isolated process. Validation-evaluation insertion must leave the next training update identical to a branch without evaluation.

Hard stop on nonfinite loss/gradient/model/optimizer/prediction, failed source/split/ownership/cutoff invariant, recovery or prediction-replay mismatch, exhausted5260 retained updates, or the60-minute resource limit. Check remaining time before beginning each update/evaluation and reserve checkpoint/receipt time; checkpoint on termination without overwriting earlier states. Do not automatically resume past the approved limits. Save failure/incomplete status and actual compute used. A completed run reports actual learning curves and comparator results even if learning is absent. No test partition evaluation, modality expansion, new clinical endpoint or alpha sweep is authorized by this proposal.

Execution dependencies: recorded resource approval; implement the separate M1.5 controller/metrics and inexpensive contract tests; pin its source and resolved configuration; preflight hashes and independent target accounting; run the single bounded experiment; reload/replay; publish aggregate learning evidence and limitations. Scientific changes encountered during implementation require a revised proposal rather than silent substitution.
