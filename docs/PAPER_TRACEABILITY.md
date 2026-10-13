# NOAH paper-to-requirement traceability

Phase 0, 2026-10-13. Authority: [supplied PDF](../paper/NOAH.pdf), arXiv:2609.09140v1, SHA-256 0f3a7bb049d6f7eb6cdf61c2aa9692d0695037851f95b73711a26ed803576730. Every page below is both the PDF page and the printed page. Specification: [REPRODUCTION_SPEC.md](REPRODUCTION_SPEC.md); decision register: [ASSUMPTIONS.md](ASSUMPTIONS.md); tests: [TEST_PLAN.md](TEST_PLAN.md).

E=EXPLICIT, D=DERIVED, A=ASSUMED, V=DEVIATION. Where a row combines these, only the expressly stated part is E. Evidence for an absent detail is the nearest relevant passage, not proof that a candidate choice appears in the paper.

## Equation locator convention

The displayed equations in this PDF are **unnumbered**. E01-E16 below are audit locator labels, not paper equation numbers. No invented “Eq. (n)” citations should appear in implementation comments or reports. Cite the page, subsection, and equation's left-hand side.

| Audit locator | Page | Exact subsection | Displayed formula/anchor |
|---|---:|---|---|
| E01 | 16 | Overall architecture and pretraining | Definition \(D_{KL}(q_\phi(z_t\mid h_{t-1},e_t)\Vert p_\theta(z_t\mid h_{t-1}))\) as expected log ratio |
| E02 | 17 | Bidirectional temporal attention | Inline shifted-log transform \(\widetilde\delta_t^\bullet=\log(1+\delta_t^\bullet)\) |
| E03 | 17 | Bidirectional temporal attention | \(\omega_n=T^{-n/N}\), n=0..N-1; preceding prose defines N and T |
| E04 | 18 | Bidirectional temporal attention | Two displays for \(\psi(u_i)\) and concatenated \(\Psi_F(u)\) |
| E05 | 18 | Bidirectional temporal attention | Additive updates \(q_t\leftarrow q_t+\Psi_2(v_t^{base})+\Psi_1(v_t^{fwd})\) and corresponding k update |
| E06 | 18 | Variational autoregression with a context-conditional prior | Marginal next-event integral \(p_\theta(x_t\mid x_{<t})=\int p_\theta(x_t\mid z_t)p_\theta(z_t\mid h_{t-1})dz_t\) |
| E07 | 18 | Same subsection | Posterior \(q_\phi(z_t\mid h_{t-1},e_t)=\mathcal N(\mu_q,\operatorname{diag}\sigma_q^2)\) |
| E08 | 19 | Same subsection | Conditional prior \(p_\theta(z_t\mid h_{t-1})=\mathcal N(\mu_p,\operatorname{diag}\sigma_p^2)\) |
| E09 | 19 | Same subsection | Closed-form \(\mathcal L_{KL,t}\), sum over j=1..k |
| E10 | 19 | Same subsection | Bound \(\log\sigma_p\in(\log\sigma_{\min},\log\sigma_{\max})\), .05 and 2 |
| E11 | 19 | Same subsection | Posterior reparameterization \(z_t=\mu_q+\sigma_q\odot\epsilon\) |
| E12 | 19 | Same subsection | Inference prior covariance multiplied by \(\tau^2\) |
| E13 | 20 | Multimodal output decoding | Product factorization \(p_\theta(x_t\mid z_t)\) over c,s,m,v,delta_next |
| E14 | 20-21 | Pretraining | Total \(\mathcal L(\theta,\phi)\): alpha posterior reconstruction + beta KL + gamma prior reconstruction; p.21 defines samples/terms |
| E15 | 25 | Survival analysis | Survival \(S(t\mid x)=P(T>t\mid X=x)=\exp(-\int_0^t h(u\mid x)du)\) |
| E16 | 25 | Survival analysis | Cox hazard \(h(t\mid x)=h_0(t)e^{h(x)}\) |

## Data and frozen encoders

| Requirement | Implementation obligation | Exact paper location | Equation/figure/table | Status and unresolved choices |
|---|---|---|---|---|
| DATA-01 | Six versioned MIMIC sources, join subject IDs | p.15, Dataset and preprocessing, first sentence/list | References23-28 pp.28-29 | E; source maps A-01, checkpoint/data availability A-59 |
| DATA-02 | Lifetime event sequence; charted datapoints; end events at actual termination | p.15, Dataset and preprocessing, opening paragraph; p.17, Multimodal token representation, final sentences | Fig.1f p.4 | E; birth/age, source timestamp/order A-01,02,62 |
| DATA-03 | 31 categories and six modalities; exact named event categories | p.35, Extended Data Table1 caption C=31/C=6; p.38, Supplementary Table1 complete category rows; p.49, Table15 modality rows | ED Table1; S Tables1,15 | E/D taxonomy, integer IDs/mapping A-01, categorical payload A-08 |
| DATA-04 | Reduce numeric vitals to min/max at sliding-window mean timestamp | p.15, Dataset and preprocessing, paragraph on charting frequencies | ED Fig.5 p.34 describes postprocessed temporal distribution | E; span/stride/availability A-04 |
| DATA-05 | Subject-disjoint 70/15/15; exclude patients below64 scored tokens | p.15, Dataset and preprocessing, split/window paragraph | No equation | E; stage/seed/scope A-03 |
| DATA-06 | 2048 window, overlap256; six metadata fields; window-end latter-four values; Unknown; exclude prefix/overlap losses; single ownership | p.15, Dataset and preprocessing, window and metadata sentences; p.17, Multimodal token representation, prefix sentence | Table16 context p.50 | E; budget, ownership edge cases, prefix causality A-05,06,62 |
| DATA-07 | Uncapped conceptual test sequences, every token evaluated, consecutive chunks with repeated prefix | p.15, Dataset and preprocessing, test-set sentences | No equation | E; memory/cache/overlap A-07,37 |
| DATA-08 | Cohort and per-modality reference counts/dimensions | p.37, Extended Data Table4; p.49, Supplementary Table15; Fig.1a-e p.4 | ED Table4, S Table15, Fig.1 | E rounded/exact as table labels; filtering/count stage A-60 |
| ENC-01 | Precompute frozen encoder embeddings, train projection layers | p.15, Dataset and preprocessing, precomputation sentences; p.17, Multimodal token representation | Fig.1f p.4 shows frozen/trainable legend and linear projections | E; exact manifests A-09 |
| ENC-02 | BioLORD for text and free-text specifics | p.15, Dataset and preprocessing, image/text encoders; p.17, Multimodal token representation; Fig.1f p.4 | S Table15 p.49 text dimension 768; ref39 p.29 | E encoder, D specifics dimension 768; length/pooling A-09 |
| ENC-03 | RAD-DINO CXR representation768 | p.15, Dataset and preprocessing, image-encoder sentence | S Table15 p.49; ref4 p.27 | E; image preprocessing/pooling A-09 |
| ENC-04 | Frozen OTIS, 12 leads at 500 Hz for 10 s, 1x24 patches, 2496x192 tokens, final patch mean | p.15, Dataset and preprocessing, ECG paragraph | S Table15 p.49 ECG dimension 192; ref40 p.29 | E; raw 5000 samples D, unused eight samples/pool selection A-09 |
| ENC-05 | ECHO CLS 512; DICOM crop, overlay removal, 224x224 AVI, 10% zoom, normalization, 16 frames/stride 2/zero pad | pp.15-16, Dataset and preprocessing, ECHO paragraph | S Table15 p.49; ref41 p.29 | E; pixel constants/channel/start/crop algorithms A-09 |

## Tokenization, temporal attention, and backbone

| Requirement | Implementation obligation | Exact paper location | Equation/figure/table | Status and unresolved choices |
|---|---|---|---|---|
| TOK-01 | Trainable category embedding, frozen specifics/values, projections to uniform d | p.17, Multimodal token representation, first paragraph; p.16, Overall architecture | Fig.1f p.4 | E; payload/missingness A-08, linear bias A-17 |
| TOK-02 | Add projected content components | p.4, Fig.1f diagram, plus signs between category/specifics/value rows | Fig.1f visual evidence | E figure/D algebra; fusion/missing terms A-08 |
| TOK-03 | Temporal integration also at input tokens | p.16, Overall architecture, temporal integration sentence | Fig.1f time-features row | E interface; exact basis/features/fusion A-13 |
| TIME-01 | Position, age, prev/next gaps; log1p gaps | p.17, Bidirectional temporal attention, feature definitions | E02 inline | E; time units/boundaries A-02,12,62 |
| TIME-02 | Per-feature sine then cosine banks; geometric base10000; concatenate; divisibility | p.17 final paragraph/display and p.18 first two displays | E03,E04 | E; head64, N2=16,N1=32 D |
| TIME-03 | Q base+forward, K base+backward; no cross-feature LayerNorm | p.17, transform/normalization sentences; p.18, feature-vector definitions and additive-update display | E05 | E; head broadcast and QKV details A-15 |
| TIME-04 | Preserve content causality with temporal asymmetry; masked attention | p.16, Overall architecture, causally masked transformer; pp.17-18, temporal mechanism; Table16 p.50 causal mask | E05 only adds Q/K, no full attention formula | E causal, D/A conventional scale/self-mask completion A-15 |
| TIME-05 | Decompose and ablate CC/TC/CT/TT; TV distance | pp.11,13, Noah relies on time; Fig.6b p.12 | Fig.6b | E conceptual ablation/D logit expansion; metric aggregation A-56 |
| ALIGN-01 | Decode x_t from prior h_(t-1), posterior [h_(t-1),e_t], h0 from prefix | p.16, Overall architecture, prior/posterior/h0 sentences; pp.18-19 variational subsection | E06-E08; Fig.1f inset | E/D shift; prefix output/posterior stage A-11 |
| ALIGN-02 | Identify training query's target-time access | pp.17-18 temporal gap definition/update; p.19 assertion of no target leakage | E02,E05,E08 | D consequence and unresolved paper tension; A-13,14,35 |
| ALIGN-03 | Time head is explicitly named outgoing delta_next_t | p.20, Multimodal output decoding, factorization and time-head prose | E13 | E index notation; timestamp assignment candidate A-14, final labels A-61 |
| ALIGN-04 | Separate free/controlled/probe cutoff feature policies | pp.21-22, Time-controlled/free forecasting and zero-shot readmission truncation; p.23 h-state definition | No paper cutoff-sanitization formula | D audit necessity, A/V choices A-14,35,36 |
| ARCH-01 | 12 layers at width 768, 12 heads, FFN ratio 8, k=384, context 2048, 150,322,970 parameters | p.50, Supplementary Table16, Architecture block | S Table16 | E, head/expanded dimensions D; norm/activation/MLP details A-17,19,58 |
| ARCH-02 | Temporal encoding layer range0-3 | p.50, Table16 Architecture row; Fig.6b p.12 L0-L3 panels | S Table16; Fig.6b | E range; zero-indexed attention-layer interpretation A-16 |
| ARCH-03 | LayerScale 1e-4; dropout/drop path/attention dropout .1 | p.50, Table16 Architecture/Optimization rows | S Table16 | E rates/init, placements/schedules/initialization A-17,18,34 |
| ARCH-04 | Patient embedding is h before variability/prior,768dim | p.5, Noah learns patient trajectories, first paragraph; p.23, Linear probing opening | Fig.1j p.4; Fig.5 p.10 | E; extraction future-time policy A-35 |

## Variational module, decoder, and objective

| Requirement | Implementation obligation | Exact paper location | Equation/figure/table | Status and unresolved choices |
|---|---|---|---|---|
| LAT-01 | Context prior and informed posterior diagonal Gaussians, concatenating MLP, linear mu/logsigma heads | pp.18-19, Variational autoregression | E07,E08; first sentence p.19 | E; hidden widths/activations A-19 |
| LAT-02 | Prior log-sigma smooth bound .05-2 | p.19, degeneracy-prevention paragraph and bound display | E10; Table16 p.50 | E interval/sigmoid; explicit interpolation formula D/A-20, stability A-21 |
| LAT-03 | Single posterior reparameterized sample, prior-sampled auxiliary pass, shared event latent | p.18, shared latent paragraph; p.19, reparameterization; p.21, auxiliary prior reconstruction | E11,E14 | E; noise coupling A-22 |
| LAT-04 | Analytic KL(q versus p), sum over k | p.19, closed-form display | E09 | E; dtype/stability A-21 |
| LAT-05 | Prior-only inference, temperature tau in[0,1], covariance tau^2 | p.19, inference display/temperature paragraph | E12; ED Table2 p.35 tau1 | E; uncertainty claim calibration A-57 |
| DEC-01 | Hierarchical event distribution category -> specifics -> modality -> value, plus time from z | p.20, Multimodal output decoding, factorization | E13 | E; explicit discrete-label conditioning not shown A-23 |
| DEC-02 | z+fcat specifics, z+fspec modality/value, z direct time; value head selected by modality; embedding regressions | p.20, cascading-feature-MLP paragraph | E13 plus following prose | E feature routing; residual width384 D; depths/value training route A-19,23 |
| DEC-03 | Time positive gate at all positions; loggap regression only positive; hard zero at inference | p.20, zero-inflated time paragraph | No gate likelihood equation | E; threshold, positive activation/boundaries A-25,26,61 |
| DEC-04 | Specifics presence gated identically | p.20, sentence after hard-time-gate explanation | No separate gate diagram | E; gate source/threshold A-25 |
| DEC-05 | Data-derived class weights for category, modality, specifics presence | p.20, final paragraph | No weighting formula | E; formula/scope A-28 |
| DEC-06 | Modality Softmax restricted to observed category-modality pairings during inference | p.20, final paragraph, pairings sentence | Inline conditional p(m_t given c_t), not a full mask equation | E; full/train scope, selection, empty support A-24 |
| NUM-01 | Numeric input Fourier features, 22 dyadic scales 2^-7 through 2^14 | p.17, numeric scalars sentence; p.50, Table16 Architecture numeric row | S Table16; ref42 p.29 | E scales; complete phi(v) candidate A-10 |
| NUM-02 | Numeric output coefficients same input sinusoidal basis, directly consumed next step | p.20, numeric output sentence | No scalar inverse equation | E direct loop; inverse A-45, manifold A-63 |
| LOSS-01 | alpha posterior rec + beta KL + gamma prior rec; same x target; two reparameterized paths | pp.20-21, Pretraining | E14; Table16 weights p.50 | E, prior budget .25 D |
| LOSS-02 | Sum component NLLs for modality/category/specifics/value/time | p.21, Pretraining, reconstruction definition; p.20 conditional gate supervision | E14 does not expand continuous NLLs | E conceptual sum; concrete likelihoods/scales A-27,28,29 |
| LOSS-03 | Linear beta warmup to 1 over 10 epochs | p.21, Pretraining, annealing sentence; p.50 Variational Latent rows | S Table16 | E; fractional-epoch/resume A-30 |
| LOSS-04 | All scored-event losses obey prefix/overlap/padding and conditional component masks | p.15 mask/single-event statements; p.20 conditional gate regression | E14 plus ownership derivation | E prefix/overlap, D combined branch masking; denominator/tails A-29,61 |
| OPT-01 | Muon hidden 2D matrices; AdamW embeddings, heads, non-2D | p.21, optimization sentence; p.50 optimizer row | S Table16 | E assignment, no optimizer internals A-31 |
| OPT-02 | LR6e-4,cosine,linear6epochwarmup,decay.01,clip10 | p.50, Optimization block | S Table16 | E, total duration/batch/floor A-32,33 |
| OPT-03 | Python 3.12/PyTorch 2.10/CUDA 12.8, up to eight A100 80 GB GPUs | p.26, Computing environment; p.50 Table16 caption | S Table16 caption | E historical environment; actual installation/precision A-34 |
| HP-01 | Entire reported pretraining hyperparameter table | p.50, Table16, all rows; p.17 temporal base; p.15 data-window constants | S Table16/E03 | E full reported inventory; unreported values intentionally OPEN |

## Rollout and scientific evaluation

| Requirement | Implementation obligation | Exact paper location | Equation/figure/table | Status and unresolved choices |
|---|---|---|---|---|
| ROLL-01 | Decode/re-encode/append each event; fresh common latent; output embedding reuse | p.21, Time-controlled and free forecasting, opening sentences; p.20 final sentences | Fig.1g-i p.4; E12 | E; caching A-37, decoded semantics A-64,65 |
| ROLL-02 | Compare freely predicted timing and externally supplied next-gap timing | pp.8,21, forecasting subsections; p.35 ED Table1 caption | Fig.3a,h,i p.7 | E oracle information; outgoing/incoming/seed mapping A-14,36 |
| ROLL-03 | Fixed token budget, threshold reach, invalid-budget exhaustion | pp.21-22 forecasting/zero-shot Methods | ED Fig.1 p.30; ED Table2 p.35 | E task stopping primitives; terminal/stall/context policies A-43,37 |
| EVAL-01 | Admission + 24 h, 13,551 patients, M=50, eight horizons, type/modality metrics and baselines | p.21, forecasting Methods; pp.7-8 Results; p.35 ED Table1; p.39 S Tables2-3 | Fig.3a-i; ED Table1; S Tables2-3 | E; Fig.3a 128 tokens is example evidence; eligibility/macro support/budget A-38..40 |
| EVAL-02 | Zero-shot LOS/mortality/readmission, 48 h/exclusive-discharge prompts, seeds, draw/budget/temperature, validity/smoothing/coverage | p.22, Zero-shot Monte Carlo classification; p.35 ED Table2 complete configuration | Fig.3j-l p.7; ED Fig.1 p.30; ED Table2 | E protocol; seed magnitudes/labels/boundaries/invalid patients A-14,41..43 |
| EVAL-03 | Sepsis-fluid cohort, earliest episode, 1,085 patients, two arms, preserved time, M=25, 1 h seed, common noise, fluid constraints, 30 d outcomes | pp.22-23, Counterfactuals; p.36 ED Table3; p.40 S Tables4-6 | Fig.4 p.9; ED Table3; S Tables4-6 | E; marker/token maps, budget, renal inverse/denominators A-44..47 |
| EVAL-04 | Frozen-h logistic probes, standardization, C=1/L-BFGS, five folds x three seeds, positions, outcomes, 15 ICD chapters and 29 comorbidities | pp.23-24, Linear probing; p.41 S Table7; pp.42-45 S Tables8-10 | Fig.5a-c p.10; ED Fig.2 p.31; S Fig.1 p.46 | E; maps/boundaries/CV/scaling A-48,49; timing A-35 |
| EVAL-05 | KL surprise, trajectory arclength/autocorrelation, PaCMAP/chain-UMAP | p.5, Noah learns patient trajectories; Fig.2 p.6; p.38 S Table1 | Fig.2f-n; S Table1 | E/D quantities; graph/aggregation A-56 |
| EVAL-06 | NEWS2 windows/labels; patient split; 512-256 MLP; optimizer/training/metrics/bootstrap/baselines | p.24, NEWS2 scoring; pp.47-48 S Tables11-12 | Fig.5d-e p.10; ED Fig.3 p.32; Fig.4a p.9 | E; exact score mappings/order/patience A-50,51 |
| EVAL-07 | Six fraction landmarks, 26,476 patients, DeepSurv, CV, TPE 100 trials, Ctd, D-cal 20 bins, IBS, search domains/results | p.25, Survival analysis; p.48 S Table13; p.49 S Table14 | E15,E16; Fig.5f-g p.10; S Tables13-14 | E; zero-landmark/depth/risk sets/nesting A-52..55 |
| EVAL-08 | Attention lift/specialization, CC/TC/CT/TT ablation, IG event-type baseline, history ablation M=25 | pp.11-13, Noah relies on time; p.12 Fig.6; p.33 ED Fig.4 | Fig.6a-e; ED Fig.4 | E design; steps/threshold/groups A-56; not hard replication tally targets |

## Published-result and appendix coverage ledger

| Item | Page(s) | Audit use |
|---|---:|---|
| Abstract, Introduction | 1-3 | Scope and motivation; no extra architecture inferred from novelty claims |
| Fig.1 | 4 | Encoder/fusion graph, latent inset, forecasting/state use |
| Fig.2 and accompanying Results | 5-6 | h dimension, surprise/projection/trajectory definitions |
| Fig.3 and accompanying Results | 7-8 | Forecast calibration/count trends, oracle timing distinction, outcome benchmarks |
| Fig.4 and accompanying Results | 8-9 | Intervention simulation and its substantial calibration failure |
| Fig.5 and Results | 10-11 | Retrieval/NEWS2/survival benchmarks and intended interpretation |
| Fig.6 and Results | 11-13 | Temporal four-term ablation, attention specialization, attribution/history ablation |
| Discussion | 14 | Teacher-forcing exposure bias; representations are predictive features; simulations not causal models |
| Dataset and preprocessing | 15-16 | Input sources, construction, window/prefix/mask, frozen encoders |
| Model and pretraining | 16-21 | Architecture, formulas, hierarchy, objective, optimizer |
| Evaluation framework | 21-25 | Experimental cohorts, clocks, target scoring, probes and CV |
| Computing/data/code availability | 26 | Authors' runtime/hardware; code promised, no inspected source |
| References1-52 | 27-29 | Identify encoder/dataset/method provenance; citing a method does not supply NOAH-specific settings |
| Extended Data Fig.1 | 30 | First-event bins, rollout stopping, positive-label orientation ambiguity |
| Extended Data Fig.2 | 31 | Probe positions/sample selection; retrieval versus mixed readout |
| Extended Data Fig.3 | 32 | NEWS2 held-out qualitative windows |
| Extended Data Fig.4 | 33 | Integrated-gradients baseline and temporal/value decomposition |
| Extended Data Fig.5 | 34 | Within-stay rates/gaps/composition for data and temporal QC |
| Extended Data Table1 | 35 | Forecast AUROC/Brier at1,6,24,72h; C31/M6; bootstrap |
| Extended Data Table2 | 35 | All outcome metrics and M/token-budget/temperature settings |
| Extended Data Table3 | 36 | Paired intervention contrast, explicit-outcome versus all-run denominators |
| Extended Data Table4 | 37 | Exact total299712 and cohort characteristics |
| Supplementary Table1 | 38 | All31type names and KL surprise distribution |
| Supplementary Table2 | 39 | Forecast24h LOS-quartile subgroup comparison |
| Supplementary Table3 | 39 | Per-type prevalence/AUROC/count MAE, top/bottom10 |
| Supplementary Table4 | 40 | Factual mortality decile calibration, ECE27.683pp |
| Supplementary Table5 | 40 | Arm-specific coverage and invalid-run proportions |
| Supplementary Table6 | 40 | MAKE-30 composite/components, missing renal prerequisite |
| Supplementary Table7 | 41 | Clinical probes over seven positions, prediction/retrieval mix |
| Supplementary Table8 | 42 | All15ICD chapter probe targets |
| Supplementary Table9 | 43 | All29Quan-Elixhauser probe targets |
| Supplementary Table10 | 44-45 | Age/LOS subgroup results and non-evaluable cells |
| Supplementary Fig.1 | 46 | Visual subgroup probe summary |
| Supplementary Table11 | 47 | NEWS2 full metric set, baselines, window versus patient macro |
| Supplementary Table12 | 48 | NEWS2 band and integer-score bias; rare high-score underestimation |
| Supplementary Table13 | 48 | All6survival settings, td-CI/IBS and5D-cal p-values |
| Supplementary Table14 | 49 | DeepSurv search ranges, ambiguous depth expression |
| Supplementary Table15 | 49 | Processed modality counts/embedding dimensions/coverage |
| Supplementary Table16 | 50 | Complete reported pretraining parameters; many training knobs absent |

## Evidence conflicts and negative-evidence rules

1. **p.19 no-target-leakage assertion versus pp.17-18 next-gap queries:** content causality is explicit, but target timing is also explicitly supplied. ALIGN-02 is a derivation, not an accusation of target-content leakage.
2. **p.20 time target versus event shift:** outgoing-gap naming is explicit; incoming-gap placement and general seed are not. A-14 must remain visible.
3. **p.15 window-end metadata versus prospective cutoff:** prefix masking excludes its loss, not its information. Strict-as-of alternatives are separate deviations.
4. **Fig.3h p.7 says30classes versus ED Table1 C31:** likely evaluability exclusions, but no exclusion map is given; do not guess.
5. **ED Fig.1 p.30 first-bin labels versus task-positive probability:** keep orientation mapping explicit; do not directly assume every plotted y=1 denotes mortality/readmission.
6. **p.25 zero-fraction entry description versus strict-before selection:** needs endpoint-specific clarification, not an off-by-one patch.
7. **Table14 depth[1,6]:** literal code-like expression and possible range prose interpretation are distinct candidates.
8. **Table16 exact parameter count versus missing block/MLP settings:** total is a consistency constraint, not enough evidence to reconstruct architecture by reverse fitting.

All proposed controls/tests/schedules are audit methodology, not paper requirements with a fabricated page citation. A-ID configuration keys originate in this audit; the register names nearby source passages and labels every proposed choice. Future implementation requirements added beyond these IDs require a new row and a classification before use.
