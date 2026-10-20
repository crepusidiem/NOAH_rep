# NOAH test and scientific validation plan

Clean-worktree context: operational paths and results below describe the archived implementation unless mapped by [MIGRATION_BASELINE.md](MIGRATION_BASELINE.md). Active imports use `noah_baseline`; current stage and supported commands are in [PROJECT_STATUS.md](PROJECT_STATUS.md). Scientific requirements and paper citations are retained unchanged.

Current code verification and selected implementation candidates (2026-10-19): [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md). Paper requirements below remain distinct from implementation/test status.

Updated 2026-10-14. This document specifies required tests. Temporal/numeric test definitions exist; execution results and coverage are recorded in project status and server reports. See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for the current snapshot and encoder-specific acceptance checks. Documentation consistency checks are separate from experimental validation.

Requirements/evidence: [REPRODUCTION_SPEC.md](REPRODUCTION_SPEC.md), [PAPER_TRACEABILITY.md](PAPER_TRACEABILITY.md). Pending configurations: [ASSUMPTIONS.md](ASSUMPTIONS.md). Proposed test IDs are audit artifacts, not tests supplied by the authors.

## 1. Testing principles and fixtures

Write independent mathematical/reference oracles before implementations. Test paper-specific behavior, not only shapes or successful forward passes. Each test records requirement IDs, A-ID decisions, information profile, input fixture, expected result, precision/tolerance, and failure meaning. A test for one assumption must not certify a different assumption.

No prospective causality claim is attached to literal window-end metadata or oracle gaps. Assertions about hidden future *content*, hidden future *timing*, and hidden future *metadata* are distinct.

Proposed fixtures:

- F1: two synthetic patients, distinct IDs, clinical sources and changing insurance/language; overlapping windows, short/tail sequences and padding.
- F2: ordered timestamps [0,2,5,5,11] in declared units; category/content IDs distinct at every event. Duplicate timestamps exercise the zero gate.
- F3: all six modalities, all 31 category names, optional/missing specifics, numeric zero, absent value, ECG192/CXR768/text768/ECHO512.
- F4: stable deterministic diagonal Gaussians with known means/log-sigmas; externally fixed epsilon tensors.
- F5: long patient >two context lengths, a metadata change inside a chunk, and a next event beyond a forecast/probe cutoff.
- F6: hand-labeled synthetic clinical trajectories for target-event/threshold/invalid/competing-event cases, NEWS2 windows and censoring.
- F7: a tiny synthetic generator with a known horizon-dependent next-type rule, correlated type/modality/value, and an unpredictable innovation.

Synthetic encoders are stub *fixtures* with known dimensions, not purported replacements for frozen paper encoders. Real-small and paper-scale tests require actual frozen assets.

Proposed numerical tolerances are ASSUMED test settings, not paper hyperparameters:

| Oracle | CPU float64 | CPU/GPU float32 |
|---|---|---|
| Temporal basis/log-gap/KL values | atol1e-10, rtol1e-9 | atol1e-6, rtol1e-5, scaled for large phase |
| Masked probabilities, zero gates, ID/mask ownership | Exact zeros/booleans/IDs | Exact zeros/booleans/IDs |
| Deterministic reference attention/block | atol1e-9, rtol1e-8 | atol1e-5, rtol1e-4 |
| Analytic versus finite-difference gradients | Relative error<=1e-4 away from gates/bounds | Use float64 reference first |
| Monte Carlo moments | Fixed-seed confidence bounds, not exact sample equality | Statistical tolerance derived from sample size |

Large dyadic phases require phase-accuracy checks and a recorded scalar/precision range, not blindly loosening all tolerances. Hard gates use decision-boundary tests rather than differentiating through a binary decision.

## 2. Unit tests by module

### Data construction and windows: DATA-01..08; A-01..08,60,62

| Test | Fixture and assertion | Failure implication |
|---|---|---|
| U-D01 subject joins | Same subject across sources joins; equal row IDs from different subjects do not; one modality cannot cross patients | Patient leakage/provenance corruption |
| U-D02 event ontology | All 31 named types map to stable IDs; six modalities round-trip; special IDs/padding excluded from clinical type metrics | Wrong vocabulary or metric support |
| U-D03 termination availability | Procedure/drug start at0,end at10; prompt cutoff5 has no end event or encoded duration; end at10 appears only then | Direct future process leakage |
| U-D04 sorting/ties | F2 yields prev gaps and next gaps according to documented order; no negative gap; tie order stable under reload | Wrong adjacency or unstable zero-gap mass |
| U-D05 vital extrema | Hand window of three values yields correct min/max and chosen mean timestamp; contributors' max availability retained | Aggregation mismatch or hidden look-ahead |
| U-D06 patient splits | Train/val/test subject intersections empty, including cached encoder rows and all visits | Invalid split |
| U-D07 minimum history | Patients at 63, 64, 65 scored events obey frozen inclusion rule; overlap/prefix do not count toward 64 | Cohort drift |
| U-D08 context capacity | Boundaries around 2048 under A-06; six prefix tokens accounted explicitly; overflow fails rather than truncates silently | Wrong context configuration |
| U-D09 unique target ownership | For N=1,255,256,2041,2042,2048,4097 and randomized lengths, each eligible clinical target scores exactly once; first-window256 not lost | Duplicate/missing loss |
| U-D10 overlap context | Overlap tokens remain attended even with score0; prefix loss0 while first new clinical target uses correct history | Loss mask confused with attention mask |
| U-D11 metadata snapshots | Change insurance/language mid-window: literal window_end and strict_asof return different expected prefixes; all six fields/Unknown explicit | Accidental future metadata or silent deviation |
| U-D12 test chunks | Every uncapped clinical event has one evaluated result, metadata each chunk, no subsampling; explicit boundary/context policy | Lost events or false lifetime-attention claim |
| U-D13 missingness | Missing numeric value !=scalar0; missing specifics !=present zero embedding; no false zero-gap at unavailable tail | Bad reconstruction labels |
| U-D14 count reconciliation | Exact subject/stage counts reconcile; rounded Table15 targets have an independently specified rounding tolerance; failures reported by stage | Undocumented filtering |

### Frozen encoders and token fusion: ENC-01..05,TOK-01..03

| Test | Fixture and assertion | Failure implication |
|---|---|---|
| U-E01 dimensions | Real manifests return ECG 192, ECHO 512, CXR/text 768; specifics dimension confirmed; all projections output 768 | Wrong checkpoint/output extraction |
| U-E02 freeze contract | NOAH loss backward leaves external parameters gradient-free/unchanged; projected weights get gradient | Encoder fine-tuning deviation |
| U-E03 cache provenance | Cache key includes source/checkpoint/preprocessing hashes; changed transform/checkpoint invalidates cache | Silent stale embeddings |
| U-E04 ECG patching | Raw [12,5000] gives 2496 patch tokens, width 192; final patch-only mean matches manual oracle; tail of eight samples has an explicit policy | Wrong OTIS pipeline |
| U-E05 ECHO pipeline | Known region/overlay fixture verifies crop/removal, 16 frames at stride 2, short-video zero pad, 224x224 shape, 10% zoom, frozen normalization | Wrong EchoPrime preprocessing |
| U-E06 fusion | Known lookup and linear weights yield independently computed category+specifics+value sum; absent components obey policy | Substituted fusion |
| U-E07 temporal input | Input-time component tests separate from attention-time basis; chosen features/magnitudes match manifest | Hidden extra temporal pathway |
| U-E08 optional raw payload | Categorical and no-value events need no fabricated continuous regression target | Invented categorical decoder |

### Numerical Fourier representation: NUM-01..02; A-10,45,63

| Test | Fixture and assertion | Failure implication |
|---|---|---|
| U-N01 scales/layout | Exactly 22 scales, powers -7..14; selected sine/cosine layout and phase tested against analytic scalar oracle | Wrong basis |
| U-N02 origin/phase | Scalar 0 has sine 0/cosine 1; nonzero values discriminate kappa=1 versus 2pi; declared unit transforms tested | Silent phase/unit substitution |
| U-N03 zero/missing | Numeric 0 remains present; missing numeric produces no value loss | Wrong presence logic |
| U-N04 derivatives | Analytic derivative kappa*scale*cos (and -sin branch) agrees with autograd/finite differences over stated range | Fourier or precision bug |
| U-N05 round-trip embedding | Arbitrary decoded Fourier vector reinjected into numeric projector is unchanged, including off-manifold vector | Unspecified scalarize/re-encode/projection |
| U-N06 scalar inversion | If enabled for renal endpoints, known scalars recover within declared range and alias/off-manifold cases return uncertainty/residual/failure | Fabricated MAKE-30 scalar |
| U-N07 basis separation | Numeric dyadic encoder cannot pass the temporal-base-10000 oracle by coincidence; test distinct frequencies | Conflated two paper mechanisms |

### Temporal attention and shift: TIME-01..05,ALIGN-01..04

| Test | Fixture and assertion | Failure implication |
|---|---|---|
| U-T01 features | F2 prev/next deltas align with event IDs; log1p zero=0; age/ordinal/time units explicit | Index/unit error |
| U-T02 basis dimensions | d_h=64, F=2 gives N=16/feature; F=1 gives N=32; full banks have 64 dimensions; invalid nondivisible heads rejected | Wrong concatenation |
| U-T03 exact ordering | Sine bank then cosine bank per feature; N-dependent geometric exponent against formula pp.17-18 | Standard PE accidentally substituted |
| U-T04 directional injection | Forward-only gap perturbation changes Q bank; backward-only perturbation changes K bank; base affects both; V direct time addition absent unless input stage specified | Symmetric/time-misrouted attention |
| U-T05 no temporal normalization | Magnitude changes remain in phase inputs; no normalization across four scalar features | Violates explicit temporal mechanism |
| U-T06 full logits | Small Q/K/V with known banks and mask matches independently computed Softmax output | Wrong scaling/mask/addition |
| U-T07 causal content | Change all future content while preserving visible time/metadata; h_i and prior for i+1 remain unchanged; gradients to future content are zero | Future content leakage |
| U-T08 timing access | Change r_(i+1) only: teacher-forced/literal h_i may change as E05 predicts; cutoff-sanitized h_i must stay unchanged if its requested horizon held fixed | Information profile conflation |
| U-T09 mask stability | Padded keys have probability 0; all-masked query safely ignored; extending pad values does not change scored outputs or cause a 0*NaN loss | Padding leak/numeric failure |
| U-T10 four terms | CC+TC+CT+TT equals unexpanded dot product; ablations remove only requested term; masks held fixed | Invalid temporal ablation |
| U-T11 shift oracle | Symbolic distinct h/e IDs: x_t uses h_(t-1),e_t; first target uses chosen h0; no within-window or chunk off-by-one | Posterior/prior leakage |
| U-T12 tail semantics | Unknown patient-final outgoing gap not labeled 0; known off-window successor handled according to A-61 | Fake zero gaps or dropped targets |
| U-T13 layer range | Instrument time injection; only chosen layers 0-3 receive attention banks; input-time addition separately counted | All-layer substitution |

### Backbone and parameters: ARCH-01..04

| Test | Fixture and assertion | Failure implication |
|---|---|---|
| U-B01 module manifest | Actual 12 layers, 12 heads, 768 hidden, nominal FFN width 6144, latent width 384 match paper profile; tiny profile separately named | Wrong architecture |
| U-B02 block oracle | Small block with fixed parameters independently validates chosen norm/residual/activation ordering | Undisclosed stock block |
| U-B03 LayerScale/drop rates | Initialization 1e-4 and rates .1 match; train/eval toggles and branch placement align with A-18 | Different regularization |
| U-B04 parameter count | Full model counted by module/group; target 150,322,970 compared; explain every difference | Missing details; count matching alone not scientific proof |
| U-B05 state hook | Exported state has 768 dimensions and is before latent, with declared terminal normalization/time policy | Wrong representation used in probes |

### Prior/posterior and KL: LAT-01..05

| Test | Fixture and assertion | Failure implication |
|---|---|---|
| U-L01 conditional inputs | Prior changes when history changes, never consumes current target e; posterior consumes 1536-dimensional concatenation and produces two 384-dimensional heads | Fixed prior or wrong conditioning |
| U-L02 mean/logstd | Known head outputs give sigma=exp(logstd), not exp(.5logstd); covariance uses sigma squared | Logvariance confusion |
| U-L03 smooth prior bounds | Raw [-100,0,100] produces floor/midpoint/cap in log space, monotonically; midpoint sqrt(.05*2) under chosen log squash; gradients nonzero away from saturation | Hard clamp or wrong squash |
| U-L04 posterior freedom | q sigma can fall below .05 or above 2 under candidate; numerical stabilization separately tested | Prior bound silently imposed on posterior |
| U-L05 reparameterization | Fixed epsilon reproduces mu+sigma*epsilon exactly; gradients reach mean/scale and history/target paths as intended | Detached sample |
| U-L06 KL identity | q=p gives 0; unequal Gaussians nonnegative; direction-sensitive example distinguishes KL(q versus p) | Wrong KL |
| U-L07 known KL | One-dimensional q=N(0,1), p=N(0,4) gives log(2)-.375; 384 independent equal dimensions multiply by 384 | Mean versus sum, variance bug |
| U-L08 anti-collapse limiting case | With sigma_p>=.05 and fixed means, sigma_q decreasing raises unweighted KL without bound; beta=0 weighted KL remains 0 | Overclaim about warmup regularization |
| U-L09 shared latent | Every head in posterior pass receives exactly same z; prior pass receives its own shared z | Independent head randomness |
| U-L10 inference temperature | tau=0 yields prior mean; fixed epsilon output scales linearly with tau; Monte Carlo variance scales tau squared | Wrong temperature |
| U-L11 inference posterior absent | Posterior method raises if invoked by free/controlled rollout; instrument inference to detect calls/access | Target-assisted generation |

### Hierarchical heads and objectives: DEC-01..06,LOSS-01..04

| Test | Fixture and assertion | Failure implication |
|---|---|---|
| U-H01 cascade routing | Fixed features verify z+fcat and z+fspec additions; category influences downstream by feature path; time consumes z directly | Flat-head substitution |
| U-H02 modality dimensions | Each route outputs target encoder dim; training route pinned; inference decoded modality selects its head | Wrong head/target pairing |
| U-H03 compatibility | Illegal category/modality logits give probability 0, allowed probabilities sum to 1; empty support policy explicit | Invalid pairings or NaNs |
| U-H04 hard time gate | Negative gate gives exact 0 regardless of regression; positive gate uses declared expm1/positivity transform; threshold ties tested | Soft gate or negative time |
| U-H05 specifics gating | Missing targets contribute presence loss but zero specifics regression; present targets do both | Missingness bias |
| U-H06 time regression mask | True gap 0 contributes binary loss but zero regression gradient; positive gap contributes both; unavailable gap contributes neither | Zero-mass dilution or false labels |
| U-H07 class weights | Hand class counts reproduce selected weight formula; scope train-only/full-data explicit; weights not accidentally on all heads | Wrong loss scale/leakage |
| U-H08 no hidden clipping | Out-of-range decoded values/gaps follow chosen numeric policy and report intervention; no unrecorded normalization | Silent generated-data change |
| U-O01 objective numbers | Known rec_q=2, KL=3, rec_p=4 at beta=1 gives .3*2+3+.1*4=4; beta=0 gives 1 | Missing/wrong branch or coefficients |
| U-O02 decoder twice | Prior/posterior reconstructions share weights, target IDs and masks; prior pass uses sampled prior, not mean | Wrong auxiliary objective |
| U-O03 gradients | Rec_p reaches prior/history/decoder, rec_q reaches posterior/target projection/decoder; KL reaches both; no unreported stop-gradient | Accidental detached path |
| U-O04 ownership all branches | Masked prefix/overlap/pad targets contribute zero to rec_q/KL/rec_p and component metrics | Incomplete loss mask |
| U-O05 reduction oracle | Variable lengths/missing dimensions exercise declared per-event/per-window/per-patient denominator; adding masked padding leaves loss unchanged | Effective objective changes |
| U-O06 continuous NLL | Fixed Gaussian candidate, if chosen, matches loglikelihood including declared variance/constants/dimension sum; MSE/cosine candidates separately labeled | Unsupported exact-ELBO claim |
| U-O07 KL schedule | Fractional epochs 0,5,10,11 give 0,.5,1,1 under candidate; resume preserves progress; warmup not mistaken for the six-epoch LR warmup | Changed regularization |

### Optimizer, scheduler, and experiment state: OPT-01..03,HP-01

| Test | Fixture and assertion | Failure implication |
|---|---|---|
| U-P01 parameter partition | Every trainable parameter exactly once; hidden 2D matrices use Muon; embeddings/heads/non-2D use AdamW even when head rank is 2 | Wrong optimizer or double updates |
| U-P02 update reference | One small matrix/scalar update against pinned Muon/AdamW references and selected internals | Variant mismatch |
| U-P03 LR schedule | Chosen start, six-epoch warmup, peak 6e-4, total duration/floor/cosine match closed form | Wrong LR or unknown defaults |
| U-P04 clip/decay | Known gradient vector norm above 10 clipped in selected order; decay applies only manifest-listed groups | Distributed/AMP update mismatch |
| U-P05 resume | Checkpoint round-trips model,optimizers,schedulers,epoch,beta,RNG,sampler; one additional deterministic step matches uninterrupted run | Irreproducible training |
| U-P06 runtime manifest | Exact Python/PyTorch/CUDA/device/precision/library/checkpoint versions recorded; environment deviations named | False environment replication |

### Autoregressive rollout: ROLL-01..03,ALIGN-03

| Test | Fixture and assertion | Failure implication |
|---|---|---|
| U-R01 scheduled clock | F2 outgoing-literal candidate: seed 2 at visible time 0; generated event 1 at 2 emits gap 3; event 2 at 5 emits 0; event 3 at 5 emits 6; event 4 at 11 | Time-head incoming/outgoing shift |
| U-R02 re-encoding | Generated type/specifics/value projected back with same frozen-space dimensions/basis; generated content then changes next prior | Loop bypass/teacher forcing |
| U-R03 free boundary | Alter hidden suffix content/timestamps/metadata after fixed prompt cutoff; same seed/RNG ->identical free outputs under strict boundary profile | Held-out information used |
| U-R04 oracle boundary | Controlled mode gets only declared oracle gaps, no target content/embeddings; changing gaps changes horizon conditioning | Unmarked oracle assistance |
| U-R05 noise and coherence | Distinct trajectories/event steps get distinct epsilon; all heads share z; no resetting seed at each event | Artificial deterministic/noisy rollout |
| U-R06 zero-gap budget | Long zero-gap sequence retains equal timestamps and exhausts budget as invalid if unscorable; no forced epsilon advance | Invented time process |
| U-R07 stopping state | Target hit, threshold crossing, terminal competitor, budget exhaustion distinguishable; endpoint-specific behavior explicit | Coverage bias |
| U-R08 cache equivalence | Bounded recompute vs caching yield equal outputs with fixed noise; reseeding last query and prefix/chunk changes invalidate appropriate state | Stale temporal cache |

### Evaluation modules: EVAL-01..08

| Test | Fixture and assertion | Failure implication |
|---|---|---|
| U-V01 forecast windows | Events at anchor included in prompt; target window (anchor,anchor+H] matches Methods; persistence uses (anchor-H,anchor] | Wrong horizon labels |
| U-V02 MC aggregation | Hand rollouts reproduce occurrence frequency, mean count, count MAE, Brier/AUROC; unsupported classes flagged | Incorrect macro metrics |
| U-V03 label orientation | Early death/readmission cases compute paper-bin and clinical-positive probabilities with explicit complement mapping where needed; AUROC uses matching labels | Reversed interpretation |
| U-V04 threshold/no-target | Exact 72 h/30 d boundary and no target beyond threshold obey selected rule; budget expiry before threshold is invalid | Wrong clinical task |
| U-V05 smoothing/coverage | Three positives/four valid gives 3.5/5=.7; invalid runs excluded; zero-valid returns configured unavailable/.5 with coverage 0, never silently valid | Calibration/selection error |
| U-V06 bootstrap/ECE | Patient resampling retains all trajectories/windows; hand bins reproduce declared ECE; repeated-fold standard deviation not confused with CI | Wrong uncertainty |
| U-V07 intervention | Swap only selected fluid fields, preserve time/dose if specified, use earliest eligible episode; subsequent crystalloid conforms to arm | Different intervention |
| U-V08 paired noise | Epsilon identical by patient/trajectory/step across arms; z need not match after histories differ; paired effects use only commonly defined patients | Unpaired contrast |
| U-V09 endpoint denominators | Explicit-outcome versus all-run mortality separate; renal component requires scalar/baseline; competing hospital states and undefined LOS handled | Wrong MAKE/mortality estimand |
| U-V10 probe leakage | Fold scaler fit on training only; frozen h; end of stay labeled retrieval; labels/visible events logged by landmark | Inflated prospective claim |
| U-V11 NEWS2 labels | Hand vitals test complete/incomplete windows, one-hour span, eight-hour carry, room air/alert, unit conversion, LoC/GCS, non-overlap and patient isolation | Wrong NEWS2 labels |
| U-V12 NEWS2 splits/metrics | Patients disjoint within NOAH test subset; training statistics only; rounded bands versus raw regression metrics separate; within-patient R2 oracle | Patient/label leakage |
| U-V13 survival selection | More than five events, duration at least one hour; p=0 special case and strict fractions/ties explicit; realized T usage recorded | Retrospective landmark misselection |
| U-V14 Cox/censor | Toy tied deaths/right-censor risk sets reproduce selected partial likelihood/baseline; S(0)=1, monotonic within [0,1] | Invalid survival estimator |
| U-V15 survival metrics | Independent Antolini Ctd/IBS/D-cal references on toy censored cohort; 20 bins; tuning/evaluation patients disjoint | Metric or nested CV error |
| U-V16 explanation | Linear toy model IG completeness; CC+TC+CT+TT sum; uniform attention lift=1; graph includes chosen sequence neighbors | Invalid attribution/projection method |

## 3. Integration tests

| ID | End-to-end scope | Required assertions |
|---|---|---|
| I-01 | Source-like synthetic tables -> events -> windows -> posterior/prior -> losses | Provenance/shift/ownership survive every stage; all six modalities; prefix and tails; finite values |
| I-02 | One training update on all-modality mini-batch | Frozen encoders unchanged; expected trainable groups update; both decoder passes and KL active; statistics logged by component |
| I-03 | One-hot symbolic event shift over two overlapping windows | Same target ID has one owner; posterior target and prior predecessor align; previous h never includes current content |
| I-04 | Strict cutoff vs literal paper-profile audit | Hidden suffix content/timing/metadata perturbations: strict free outputs invariant; literal future-time/metadata dependence reported rather than incorrectly failing |
| I-05 | Training -> prior-only generation -> generated inputs | No posterior access; embedding dimension/basis preserved; clock semantics stable; fixed-seed replay |
| I-06 | Long-record inference and bounded-context rollover | Exactly one h/result per clinical event; declared memory policy; cache/recompute agreement; metadata updates |
| I-07 | Save/resume synthetic training + rollout | Model/update/RNG and subsequent rollout agree within chosen deterministic numerical tolerance |
| I-08 | Full evaluation pipeline on hand-labeled fixtures | Forecast, zero-shot, interventions, probes, NEWS2 and survival use correct information/denominators; no use of test statistics in training |
| I-09 | Small real-data encoder batch | Realassetversions, preprocessing, dimensions, missingness, cachehashes validated before training |
| I-10 | Full-size construction/dry-run after approval | Parametercount ledger,contextmemory,optimizerpartition,gradientflow and allobjectivebranches; no accuracyclaim |

## 4. Scientific correctness experiments

These are necessary scientific checks, not brittle unit tests expecting the paper's exact scores from a new split. Preregister data, configs, tolerances/decision criteria, and compute budget before evaluating the test set.

| ID | Experiment and expected evidence | Scientific failure signal |
|---|---|---|
| S-01 | Known-rule F7: vary history at fixed horizon and requested horizon at fixed history; predicted class/value follows both dependencies on held-out synthetic patients | Model ignores horizon/history despite shape-correct attention |
| S-02 | Compare predictable and stochastic innovations in F7 using unweighted KL distributions and latent variance | Posterior bypass, uninformative prior, or “surprise” unrelated to predictable structure |
| S-03 | Correlated type/modality/value generator: compare joint consistency with component marginal accuracy; diagnose impossible pairings | Shared latent/cascade does not preserve meaningful joint structure |
| S-04 | Tiny-batch overfit with nonzero KL; monitor both reconstruction losses, KL, means/scales, saturation and active dimensions; separately monitor beta=0 phase | Low posterior loss but poor prior, collapsed q, inflated prior or inactive history encoder |
| S-05 | Compare literal and strict prefix, aggregation availability, input time and probe forward-gap policies on matched cohorts | Forecast performance depends on unavailable future information |
| S-06 | Outgoing literal versus documented shifted-gap alternative; seed sensitivity; timestamp, zero-mass and horizon coverage diagnostics | Silent target shift or results dominated by first-gap choice |
| S-07 | Compare declared continuous likelihood/reduction alternatives under identical small training budget; inspect modality scales and prior/posterior gap | Objective changes mainly due to arbitrary dimension weighting |
| S-08 | Prior reconstruction on/off, KL anneal, prior bounds and flat-cascade alternatives as labeled ablations | Mechanism cannot be separated from substituted components |
| S-09 | Numeric phase/normalization/precision sensitivity; coefficient manifold diagnostics and scalar inversion residuals | Basis assumption dominates numeric forecasting or creates false renal outcomes |
| S-10 | Free versus oracle timing on identical cohort/RNG/budget and gap alignment; type/modality probabilities, counts and coverage | Oracle content leakage or unreported timing assistance |
| S-11 | Autoregressive drift across eight horizons: timing/content/count errors, zero-gap stalls, illegal pairs, embedding outliers and coverage | One-step teacher-forced quality mistaken for multistep forecasting |
| S-12 | Frozen-h retrieval probes, current-state NEWS2 and survival fractions with exact information visibility reported | Near-perfect retrieval mislabeled as future outcome prediction |
| S-13 | Outcome MC validity selection: coverage versus risk/LOS/modality; compare valid-only estimates and sensitivity bounds for invalid draws | High AUROC achieved by discarding hard trajectories |
| S-14 | Intervention factual calibration and paired contrasts together; endpoint/component missingness and denominators explicit | Effect sign presented as causal benefit despite miscalibration/selection |
| S-15 | Real-small and paper-scale data match modality/care distributions, ED Fig.5 rates/gaps/entropy, rare modalities and cohort counts | Reproduction uses a materially different event stream |
| S-16 | Independent paper-scale seeds where feasible; patient bootstrap/CV uncertainty; table-by-table comparisons under fixed protocol | Test-set tuning or cherry-picking closest metric |

For S-08, a fixed-standard-normal prior, omitted prior loss, flat heads, or changed bound are DEVIATION ablations, not faithful defaults. Synthetic criteria must come from the known generator (e.g., held-out rule accuracy/calibration beyond an independently computed baseline with confidence bounds), not arbitrary desired paper AUROC.

## 5. Paper-result acceptance and interpretation

Produce a comparison report for every figure/table family in the traceability ledger, including discrepancies and unexecuted analyses. Primary comparisons: ED Table 1 forecasting; ED Table 2 outcomes; ED Table 3/S Tables 4-6 interventions; S Tables 7-10 probes; S Tables 11-12 NEWS2; S Tables 13-14 survival; S Table 1 surprise and mechanistic Fig.6.

Matching a published CI is supportive only when cohort, representation, information set, horizon, budget, endpoint, metric and aggregation match. Published bootstrap CIs describe their cohort's sampling uncertainty; they are not a tolerance for architecture differences. Failure to match must be decomposed into data/encoder/assumption/optimization/statistical causes before changing code.

No automatic assertion requires exactly 141 specialized heads, exact projection coordinates, or exact AUROC on another split. Conversely, tests may pass while scientific fidelity remains unresolved.

Final gates:

1. **Unit gate:** all invariants pass for approved configurations; meaningful failure diagnostics.
2. **Integration gate:** masking/shift/information/gradient/rollout contracts hold together.
3. **Synthetic science gate:** known temporal/latent/joint distributions recovered; no leakage shortcut.
4. **Real-small gate:** actual encoders/data correct; training/inference diagnostics stable; assumption sensitivity recorded.
5. **Paper-scale gate:** corpus/schedule/assets/evaluation protocol evidence sufficient, unresolved choices declared, full comparison and uncertainty report.

Only the audit/document checks are in scope now. Implementation or running these experimental gates awaits explicit approval.

## Controlled remediation: executable gates and remaining tests

`tests/test_evaluation_contracts.py` implements the following independent synthetic checks:

- Complete manifest/artifact requirements, revision and module-count consistency; unchanged source/prediction evidence binding; changed or missing check artifacts; FAIL/SKIP/UNKNOWN required checks and pending methodology block release.
- One explicitly requested landmark state, no mean pooling; correct event/encounter identity; future event, delayed/unknown availability, future demographics and retained-gap inconsistency rejected; paper full-next-gap policy remains distinct from strict prospective mode.
- Explicit encounter selection and threshold comparison, unknown outcomes retained, overlapping readmission rejected and incomplete follow-up not made negative.
- Missing/nonfinite core vital values cannot produce a NEWS2 label; aggregate unknown availability remains unknown; nonpositive aggregation windows rejected.
- Unsupported retrospective-to-prospective reporting and report overwrites rejected; incompatible architecture manifests cannot be aggregated; unversioned checkpoints and conflicting YAML cannot be loaded as the current model.
- Legacy clinical evaluation fails before computation; the alpha launcher exits78 before loading modules or starting training; metric dictionaries without manifests cannot pass the release CLI.

These checks exercise contracts on fictional fixtures, not a completed real-data pipeline. Methodological parameters inside fixtures are explicitly synthetic and are not project defaults. Still pending: approved source-field/availability ingestion, provenance-preserving shard/view serialization, raw-suffix view-builder invariance, full label/source fixtures, actual patient/fold artifacts, validated NEWS2 complete-window/scoring oracle, optimizer/denominator sensitivity and exact-resume replay, clinical prediction replay and all applicable paper-scale asset checks. These remain independent release blockers; unit tests do not certify them.

Run the release boundary using:

```bash
python scripts/validate_experiment_release.py /private/run/manifest.json --claim prospective
```

The validator exits nonzero on unmet prerequisites and performs no model execution. A check artifact must identify its check name, PASS status, source SHA-256 and predictions SHA-256; the manifest also records and verifies the artifact's own checksum. Source snapshots, checkpoints, patient-split files and private prediction files must exist and match the manifest. Clinical evidence must come from executed checks; copying the synthetic fixture's PASS files is not valid execution evidence.

## Clean-worktree B0 verification

The isolated clean suite passed66 tests with no skips. Its active paths use `noah_baseline`; exact copied-test changes and legacy-only exclusions are in `migration_inventory.json`. Additional tests in `tests/test_isolation.py` reject ambient/legacy imports, nonisolated execution, external/symlink/cross-profile cache paths, unresolved profile activation and migration hash drift; meta-device module counts bind the unchanged candidate architecture. See the dated initialization report for execution evidence. M1-01–08 in `MIGRATION_BASELINE.md` are required real-data acceptance tests and remain unexecuted; B0 synthetic checks do not satisfy them.
