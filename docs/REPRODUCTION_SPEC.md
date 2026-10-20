# NOAH reproduction specification

Current code verification and selected implementation candidates (2026-10-19): [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md). Paper requirements below remain distinct from implementation/test status.

Updated 2026-10-14. Architecture, data, training and evaluation requirements.

For the operational steps, 2026-10-14 code snapshot and verified upstream feature-extraction interfaces, see [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md). Current implementation progress is recorded in [PROJECT_STATUS.md](PROJECT_STATUS.md).

## 1. Source, completeness, and fidelity contract

Primary evidence is the supplied [NOAH.pdf](../paper/NOAH.pdf), arXiv:2609.09140v1, dated 8 September 2026, 50 pages. SHA-256: 0f3a7bb049d6f7eb6cdf61c2aa9692d0695037851f95b73711a26ed803576730. PDF page numbers and printed page numbers coincide. The [arXiv record](https://arxiv.org/abs/2609.09140) was checked on the audit date; the local PDF metadata identifies v1.

Read coverage: abstract/introduction pp. 1-3; Results and Figs. 1-6 pp. 4-13; Discussion p. 14; Online Methods pp. 15-26; references pp. 27-29; Extended Data Figs. 1-5 pp. 30-34; Extended Data Tables 1-4 pp. 35-37; Supplementary Tables 1-16 and Supplementary Fig. 1 pp. 38-50. Formula pages, Fig. 1, the remaining research figures, and the numeric/configuration tables were also visually checked.

Classification used throughout:

- **EXPLICIT (E):** directly specified in paper text, equations, figures, or tables.
- **DERIVED (D):** follows from stated equations/dimensions; no architectural discretion.
- **ASSUMED (A):** a proposed choice for an underspecified detail; requires recording and sensitivity testing.
- **DEVIATION (V):** a deliberate alternative that changes an explicit paper mechanism or experimental information set.

Requirement IDs below are mapped to exact evidence in [PAPER_TRACEABILITY.md](PAPER_TRACEABILITY.md). A-IDs refer to separately configurable unresolved choices in [ASSUMPTIONS.md](ASSUMPTIONS.md). Unknown dimensions, losses, and hyperparameters remain unknown. This specification is complete as an evidence inventory, not an assertion that the publication fully specifies an executable model.

Three distinct run families must be recorded: a literal paper-description candidate, a strict causal audit alternative where the paper's data construction permits look-ahead, and reduced synthetic/small-scale configurations. An unresolved candidate must not be called an exact reproduction. Approval to implement does not automatically resolve missing paper details.

## 2. Data contract and preprocessing

### 2.1 Sources and event records

**DATA-01 (E):** merge by subject ID: MIMIC-IV 2.2, MIMIC-ED 2.2, MIMIC-Note 2.2, MIMIC-CXR-JPG 2.0, MIMIC-IV-ECG 1.0, MIMIC-ECHO 1.0. Preserve patient identity across all modalities and visits.

**DATA-02 (E):** one patient sequence comprises the entire available lifetime record, starting at birth, with arbitrary gaps, ED visits, hospitalizations, and ICU stays. Death need not be observed. Each charted data point is an event; ongoing processes receive distinct termination events at the charted termination time. No duration or termination information may be moved to the start event.

An audit event schema (**D/A**, serialization not specified) separates:

| Field | Shape per batch/window | Meaning |
|---|---|---|
| subject ID, source row ID, event ID | metadata | Provenance and loss-ownership identity |
| timestamp $r_i$, available-at timestamp | [B,L] scalars | Paper's event time; separate audit availability time |
| category $c_i$ | [B,L] integers | 31 high-level event categories |
| modality $m_i$ | [B,L] integers | Six modalities |
| specifics presence $b_i^s$ | [B,L] booleans | Whether optional type specifics exist |
| specifics embedding $s_i$ | [B,L,768] where present | Frozen BioLORD output, dimension D from text embedding table |
| value presence and source value | [B,L] masks + ragged payloads | Missing value must be distinct from a genuine zero |
| value embedding $v_i^{(m)}$ | [B,L,D_m] or gathered [N_m,D_m] | External modality space; dimensions below |
| ordinal $p_i$, age $a_i$, gaps | [B,L] each | Temporal features; units unresolved |
| padding, prefix, overlap, score ownership | [B,L] booleans | Attention and loss masks have separate purposes |

Category vocabulary names can be recovered from Supplementary Table 1, p. 38: ECG; X-ray; ECHO; Note Radiologyreport; Death; Enter Hospitalization; Note Dischargesummary; Enter ICU; Leave ICU; Microbiology Test; Service; Leave Transfer; Leave Hospitalization; Procedure End; Leave ED; Body Output; Transfer; Drug Prescription; Drug Start; Body Input; Complaint; Drug Stop; Enter ED; Drug Previously; Procedure; Diagnosis; Other Event; Triage Acuity; Vitals; Lab Test; Body Input End. **DATA-03 (E/D):** preserve these 31 semantic classes, without inventing their integer IDs or SQL mappings. Six modalities are categorical, numeric, imaging/CXR, text, ECG, ECHO. Whether metadata or padding introduce additional internal IDs is unspecified.

**DATA-04 (E):** a sliding aggregation window reduces frequently charted numeric vitals to a minimum and maximum per vital type, timestamped at the window mean. Its span, stride, boundaries, tie handling, and min/max token encoding are unspecified. These are not the separate one-hour NEWS2 windows. Mean-time stamping may expose later measurements at an earlier nominal time; available-at provenance is mandatory for the audit (A-04).

**DATA-05 (E):** subject-disjoint 70/15/15 train/validation/test split. Patients with fewer than 64 scored tokens are excluded. The ordering of exclusion versus splitting and exact split seed/IDs are unavailable.

**DATA-06 (E):** training and validation windows contain at most 2,048 tokens with overlap 256. Prepend six demographic tokens, one each for date of birth, race, gender, insurance, language, marital status. The latter four attributes (gender through marital status) use the most recent value available at window end, defaulting to Unknown. Race's updating/fallback rule is not stated. Prefix and overlap are excluded from the objective; every eligible clinical event contributes exactly once. Whether 2,048 includes the prefix is unresolved.

The loss ownership invariant is (**D**):

$$
\forall e\text{ eligible},\quad \sum_{w\ni e}\mathrm{score\_owner}(w,e)=1.
$$

The first window cannot blindly mask its first 256 clinical events. Loss ownership must be applied to the *target event after shift*, not the corresponding unshifted input. Prefix/overlap tokens remain available as context. All posterior reconstruction, KL, and prior reconstruction terms obey ownership and padding masks. Component-specific masks apply in addition.

**DATA-07 (E):** each test patient is one uncapped conceptual sequence; score every clinical token. Process overlength sequences in consecutive inference chunks with the metadata prefix prepended to every chunk. This does not establish unlimited attention memory: chunk overlap, state retention, caching, positions, and metadata update semantics are unspecified (A-07).

### 2.2 Frozen external encoders

**ENC-01..05 (E/D):** precompute frozen value and specifics embeddings; train NOAH's projections, not the external encoders.

| Modality | Raw input contract | Frozen representation | NOAH projection output |
|---|---|---|---|
| Specifics/text | Free text; truncation/pooling unspecified | BioLORD, 768-dimensional text space; specifics dimension inferred | [N,768] |
| CXR | Image; resizing, views, pooling unspecified | RAD-DINO, [N,768] | [N,768] |
| ECG | 12 leads, 10 s at 500 Hz; raw [N,12,5000] D | OTIS final patch mean, [N,192] | [N,768] |
| ECHO | Ultrasound video | EchoPrime global CLS, [N,512] | [N,768] |
| Numeric | Scalar with type/units | 22 dyadic scales; candidate sine/cosine [N,44] D/A | [N,768] |
| Categorical | Event/type semantics | No continuous dimension given in Table 15 | Category embedding [N,768]; payload rule unresolved |

ECG (**ENC-04 E**): non-overlapping $1\times24$ strided-convolution patches produce $12\times208=2496$ tokens of dimension 192. Mean-pool final OTIS patch tokens. $208\times24=4992$, leaving 8 samples per lead; the paper does not say whether they are discarded or otherwise handled. Frozen checkpoint, signal normalization, lead ordering, missing leads, and output token selection require a manifest.

ECHO (**ENC-05 E**): crop using DICOM SequenceOfUltrasoundRegions; remove ECG overlay and burned-in annotations; resize to 224x224; save AVI; apply further 10% zoom crop on load and resize back; normalize with EchoPrime training-set mean/std; sample 16 frames at stride 2; zero-pad shorter videos. Raw video tensor can be [N,C,16,224,224] or [N,16,C,224,224] (A), with channel count, starting frame, crop convention, and normalization constants unresolved.

Versions/checkpoint hashes, pretraining overlap with evaluation patients, text lengths, preprocessing scripts, and source-item maps are not supplied. Access/availability is a prerequisite, not assumed from the paper's availability statement.

### 2.3 Published dataset targets

**DATA-08 (E):** cohort total 299,712 patients (Extended Data Table 4), with 180,733 hospitalized, 118,979 not hospitalized, and 29,076 deceased in record. Main figure rounds to 299k patients, 431k stays, 559M events.

| Modality | Total | Train | Validation | Test | Raw embedding dim | Patient coverage |
|---|---:|---:|---:|---:|---:|---:|
| Categorical | 82M | 57M | 12M | 12M | absent | 100.0% |
| Numeric | 272M | 190M | 41M | 41M | unspecified | 94.9% |
| CXR | 357k | 250k | 53k | 53k | 768 | 20.6% |
| Text | 204M | 143M | 31M | 30M | 768 | 93.3% |
| ECG | 795k | 556k | 118k | 119k | 192 | 53.3% |
| ECHO | 281k | 198k | 39k | 43k | 512 | 1.5% |
| Total | 559M | 391M | 84M | 83M | - | 299k patients |

These are rounded targets, not exact checksum counts; their rows need not add exactly. Corpus ECG count 800,035 precedes the processed 795k figure; filtering reconciliation is unknown.

## 3. Model tensors and input representation

Notation: B=batch size, L=total model input length including metadata/padding under the chosen length policy, S=clinical event count, d=768, H=12, $d_h=d/H=64$, k=384, C=31 categories, M=6 modalities, $D_s=768$ (derived), $D_{\rm numeric}=44$ only under a full 22-frequency sine/cosine convention. $N_m$ denotes present tokens of modality m. An event index i is not a physical timestamp.

**TOK-01 (E):** learned event-category lookup [C,d]; frozen free-text specifics and value representations; trainable projections into common dimension d. Fig. 1f depicts additive fusion of category, specifics, and value. **TOK-02 (E/D):** candidate content token:

$$
e_i^{\rm content}=E_c(c_i)+\mathbf1_{b_i^s}P_s(s_i)+\mathbf1_{b_i^v}P_{m_i}(v_i).
$$

This sums to [B,L,768]. Fig. 1 supports addition; exact modality embedding, missing-value embeddings, bias, normalization, and whether numeric/categorical payloads have further terms are unspecified. Projection boxes are labeled linear in Fig. 1. Temporal features also enter the input token (**TOK-03 E**), but their selected components, basis dimensions, scale, and exact fusion are not specified. Thus $e_i=\mathrm{InputTemporalFuse}(e_i^{\rm content},\mathrm{time}_i)$ is an unresolved interface, not a filled-in standard position encoder.

Do not append a conventional RoPE or learned positional embedding as an undocumented substitute.

## 4. Bidirectional temporal attention and alignment

### 4.1 Explicit temporal mathematics

**TIME-01 (E/D):**

$$
\delta_i^{\rm prev}=r_i-r_{i-1},\quad
\delta_i^{\rm next}=r_{i+1}-r_i,\quad
\widetilde\delta_i^\bullet=\log(1+\delta_i^\bullet).
$$

Units of r, age, and ordinal origin must be pinned before encoding. A logarithm with 1 implicitly commits to a time unit; seconds and hours produce different features. Gaps must be nonnegative after a documented same-time ordering.

**TIME-02 (E):** for F features and head dimension $d_h$:

$$
N_F=d_h/(2F),\quad \omega_n=10^{-4n/N_F},\quad n=0,\ldots,N_F-1,
$$
$$
\psi_F(u_i)=[\sin(u_i\omega_0),\ldots,\sin(u_i\omega_{N_F-1}),
\cos(u_i\omega_0),\ldots,\cos(u_i\omega_{N_F-1})],
$$
$$
\Psi_F(u)=[\psi_F(u_1)\Vert\cdots\Vert\psi_F(u_F)]\in\mathbb R^{d_h}.
$$

Require $2F\mid d_h$. With $d_h=64$, $N_2=16$ per position/age feature and $N_1=32$ for each directional gap bank. Each *complete* bank has 64 dimensions; the base bank is added to the separate directional bank, not concatenated into a 3-feature bank.

**TIME-03 (E):**

$$
Q_i^*=Q_i+\Psi_2([p_i,a_i])+\Psi_1([\widetilde\delta_i^{\rm next}]),\quad
K_i^*=K_i+\Psi_2([p_i,a_i])+\Psi_1([\widetilde\delta_i^{\rm prev}]).
$$

Q/K/V shape [B,12,L,64]; added time banks [B,L,64], broadcast over heads (D candidate; learned per-head modifications not specified). No LayerNorm across temporal scalar features. Values are not explicitly given a directional temporal addition.

**TIME-04 (D/A):** a conventional scaled-dot-product completion, if approved:

$$
A_{ij}=\operatorname{softmax}_j((Q_i^*\cdot K_j^*)/\sqrt{64}+\mathcal M_{ij}),\quad O_i=\sum_jA_{ij}V_j,
$$

with output [B,L,768], causal/padding mask $\mathcal M_{ij}=-\infty$ for $j>i$ or padded keys, 0 otherwise. Scaling and self-inclusion are not explicitly written in the paper; preserve this status. “Bidirectional” refers to different temporal directions on Q and K, not access to future content.

**TIME-05 (E/D):** for ablation, before masking/Softmax:

$$
(Q_i+T_i^Q)\cdot(K_j+T_j^K)
=Q_iK_j+T_i^QK_j+Q_iT_j^K+T_i^QT_j^K.
$$

The terms are CC, TC, CT, TT, respectively. Remove an individual logit term while preserving the others and identical masks; compare attention distributions using total variation distance.

### 4.2 Shift and information boundary

**ALIGN-01 (E/D):** event $x_t$ is decoded from latent $z_t$ whose prior consumes $h_{t-1}$; informed posterior consumes $[h_{t-1}\Vert e_t]$. $h_0$ comes from metadata. For six prefix positions, the candidate first-event history state is the final prefix output (A-11). Shift along event identity, including boundaries; never use $h_t$ as prior input for reconstructing $x_t$.

**ALIGN-02 (E, unresolved causality):** $h_{t-1}$'s query uses $\delta_{t-1}^{\rm next}=r_t-r_{t-1}$. Therefore the published equations supply target *timing* during teacher-forced training; a causal content mask does not hide it. Write the operational prior as $p(z_t\mid\text{visible history},\text{query horizon})$ when describing the experiment's information set. The paper's notation $p(z_t\mid h_{t-1})$ remains intact.

**ALIGN-03 (E, unresolved indexing):** p. 20 names the decoded time component $\delta_t^{\rm next}$, the gap from generated $x_t$ to $x_{t+1}$. This is different from the incoming gap needed to place $x_t$. Do not silently shift the time target.

One coherent **literal outgoing-gap candidate (A)** is:

1. Last visible event $x_{t-1}$ receives a supplied/seeded outgoing gap $g_{t-1}$.
2. Its query uses $g_{t-1}$; transformer yields $h_{t-1}$.
3. Prior sample $z_t$ yields content of $x_t$ and its outgoing gap $g_t$.
4. Set $r_t=r_{t-1}+g_{t-1}$.
5. Re-encode $x_t$ with backward gap $g_{t-1}$ and forward query gap $g_t$; repeat.

This avoids circularly demanding the current step's unknown gap: each decoded gap schedules the *following* event. The initial gap remains unresolved for general free forecasting; the zero-shot experiments explicitly seed it, and counterfactuals specify 1 h. Another candidate decodes the incoming gap, but changes the literal time-target indexing and needs a documented strategy for the current query horizon (A-14/V).

**ALIGN-04 (D):** never retain the held-out next timestamp as the last prompt token's “free” gap. Freeze or sanitize forward features at forecast/probe cutoffs; time-controlled evaluation can explicitly supply its oracle schedule. The input token's unspecified temporal integration may also expose the target's next gap to the posterior; this must be audited.

## 5. Transformer backbone

**ARCH-01 (E):** 12 causal decoder layers, dimension 768, 12 heads, FFN expansion ratio 8, bottleneck 384, context 2,048; total reported trainable parameters 150,322,970. FFN nominal expanded width $8\times768=6144$ (**D**), but GLU versus ordinary FFN changes actual matrices and counts.

**ARCH-02 (E/A):** Table 16 says temporal encoding layers 0-3. Interpret as four zero-indexed layers for a candidate; all-12-layer temporal injection is not automatically faithful. Input temporal addition is separate. Whether 0-3 includes an embedding stage is unresolved.

**ARCH-03 (E):** LayerScale initialization $10^{-4}$; dropout, drop path, and attention dropout all 0.1. Placement, schedule, norm type, pre/post norm, activation, bias, residual layout, embedding scale, weight initialization, output normalization, and parameter tying are unspecified. These must be configured; the paper does not authorize choosing a stock Transformer block without disclosure.

Transformer output $h$: [B,L,768]. Patient state for downstream analysis is this output *before* the variational module (**ARCH-04 E**), not z, posterior mean, or decoded outputs. Representation extraction must record what future-time features it used.

## 6. Context-conditional prior and informed posterior

**LAT-01 (E):** prior input [B,S,768]; posterior concatenated input [B,S,1536]. MLPs with linear mean and log-standard-deviation heads, each output [B,S,384]. Hidden depth/width/activation and prior's precise MLP layout are not given.

$$
p_\theta(z_t\mid h_{t-1})=\mathcal N(\mu_{p,t},\operatorname{diag}(\sigma_{p,t}^2)),
\quad q_\phi(z_t\mid h_{t-1},e_t)=\mathcal N(\mu_{q,t},\operatorname{diag}(\sigma_{q,t}^2)).
$$

The paper uses $\log\sigma$, not $\log\sigma^2$; implementations must expose that distinction.

**LAT-02 (E/D/A):** smoothly squash prior log-standard-deviation into $(\log0.05,\log2)$. Candidate equation derived from the named mechanism:

$$
\log\sigma_p=\log0.05+(\log2-\log0.05)\operatorname{sigmoid}(u_p).
$$

This specific interpolation is D/A; the raw-output affine scaling is unspecified. Do not bound posterior sigma to the same range without a separate assumption. Floating-point saturation and log-sigma numerical stabilization require declared policies.

**LAT-03 (E):** one posterior sample per event during training:

$$
z_t=\mu_{q,t}+\sigma_{q,t}\odot\epsilon_t^q,\quad\epsilon_t^q\sim\mathcal N(0,I).
$$

Auxiliary prior reconstruction uses a second prior-sampled latent $\widetilde z_t=\mu_{p,t}+\sigma_{p,t}\odot\epsilon_t^p$, also reparameterized. Correlation/independence of these two draws is unspecified. All component heads in a pass share that pass's single latent.

**LAT-04 (E):**

$$
\mathcal L_{{KL},t}=\sum_{j=1}^{384}
\left[\log\frac{\sigma_{p,t,j}}{\sigma_{q,t,j}}+
\frac{\sigma_{q,t,j}^2+(\mu_{q,t,j}-\mu_{p,t,j})^2}{2\sigma_{p,t,j}^2}-\frac12\right].
$$

Sum over latent dimensions, not mean. This is $KL(q\Vert p)$, not $KL(p\Vert q)$ or KL to a standard normal.

**LAT-05 (E/D):** inference samples *only* the prior:

$$
z_t=\mu_{p,t}+\tau\sigma_{p,t}\odot\epsilon_t,\quad\tau\in[0,1].
$$

Covariance is $\tau^2\operatorname{diag}(\sigma_p^2)$. Extended Data Table 2 uses $\tau=1$; $\tau=0$ is a deterministic ablation. Fresh noise per event/trajectory, except matched counterfactual arms share noise. Future target embedding must never enter inference.

The paper's description of the variance as epistemic uncertainty and KL as inherent unpredictability is interpretive; neither statistical uncertainty decomposition nor a predictive-error upper-bound theorem is provided. Report latent variability and surprise without claiming those properties were established.

## 7. Hierarchical multimodal decoder

**DEC-01 (E):**

$$
p(x_t\mid z_t)=p(c_t\mid z_t)\,
p(s_t\mid c_t,z_t)\,
p(m_t\mid c_t,s_t,z_t)\,
p(v_t\mid c_t,s_t,m_t,z_t)\,
p(\delta_t^{\rm next}\mid z_t).
$$

**DEC-02 (E/D):** explicit feature cascade:

$$
f_t^{cat}=F_{cat}(z_t),\quad c\text{-logits}=H_{cat}(f_t^{cat}),
$$
$$
f_t^{spec}=F_{spec}(z_t+f_t^{cat}),\quad
\hat s_t=H_{spec}(f_t^{spec}),
$$
$$
m\text{-logits}=H_m(z_t+f_t^{spec}),\quad
\hat v_t^{(m)}=H_v^{(m)}(z_t+f_t^{spec}),\quad
\text{time outputs}=H_\delta(z_t).
$$

$f^{cat},f^{spec}$ must have 384 dimensions for the stated additions (D). Category logits [B,S,31], modality logits [B,S,6], specifics output [B,S,768] D, specifics-presence logit [B,S,1], time-positive logit and log-gap regressor [B,S,1] each. Value outputs [N_m,D_m] in their frozen spaces, including ECG 192 and ECHO 512; no requirement for every output to have dimension 768.

The mathematical factorization mentions discrete c/s/m conditioning; the implementation prose passes *continuous features*, not sampled category/specifics embeddings. Preserve the explicit feature cascade; whether discrete teacher forcing exists is unresolved. The modality determines the value head at inference. Training routing to the correct-dimension target is unspecified. Head depths/widths, activation, presence-gate placement, and value path's internal feature extractor are unknown.

**DEC-03 (E):** time is a hurdle output: gate $P(\delta_t^{next}>0\mid z_t)$, supervised at every valid gap position; regress $\log(1+\delta_t^{next})$ only where $\delta_t^{next}>0$. At inference a negative hard decision emits exactly zero; positive-gap inverse is $\operatorname{expm1}(\hat\ell_t)$ (D). Positive-output activation and protection against negative predictions/overflow are unspecified.

**DEC-04 (E):** specifics presence is gated identically; regress its embedding only when present. A missing specifics vector must not be trained as a zero vector without disclosure.

**DEC-05 (E):** data-derived class weights for category, modality, and specifics presence. No weighting formula or sampling policy is specified. The time gate is not explicitly included among the weighted heads.

**DEC-06 (E):** inference modality Softmax is restricted to modalities ever observed for the generated category:

$$
P(m\mid c)=\frac{\exp l_m\,\mathbf1_{(c,m)\in\mathcal A}}
{\sum_{m'\in\mathcal A(c)}\exp l_{m'}}.
$$

The support data scope (entire dataset versus train-only), unseen support policy, and categorical selection (argmax or sampling) are unresolved. A compatibility mask does not establish all outputs are clinically possible.

### Numeric representation

**NUM-01 (E/D/A):** 22 dyadic scales $2^{-7},2^{-6},\ldots,2^{14}$ are specified. If each scale has sine and cosine:

$$
\Phi(v)=[\sin(\kappa\,2^j\,g(v)),\cos(\kappa\,2^j\,g(v))]_{j=-7}^{14}\in\mathbb R^{44}.
$$

This is a candidate formula, not an explicit paper equation: phase multiplier $\kappa=1$ versus $2\pi$, bank ordering, scalar normalization $g$, additional terms, and loss metric are unavailable. It is distinct from the geometric temporal basis with base $10^4$.

**NUM-02 (E):** decoder emits numeric Fourier coefficients in the *same* input basis and reuses them directly on rollout. No scalar inversion is required for token reinjection. Do not silently scalarize/re-encode generated coefficients. Off-manifold coefficients and periodic aliasing make scalar recovery for creatinine-based MAKE-30 nontrivial; the paper supplies no inversion method.

## 8. Training objective and optimization

**LOSS-01 (E):**

$$
\mathcal L(\theta,\phi)=
\mathbb E_{x_{1:S}\sim D}\left[\sum_{t=1}^{S}
\alpha\mathcal L_{{rec},t}(z_t)+\beta\mathcal L_{{KL},t}
+\gamma\mathcal L_{{rec},t}(\widetilde z_t)\right].
$$

The line break's plus denotes addition; posterior weight $\alpha=0.3$, prior weight $\gamma=0.1$, KL target $\beta_{\max}=1$. The prior reconstruction budget fraction is $0.1/(0.3+0.1)=0.25$ (D). Decoder parameters are shared across both passes. Prior reconstruction is sampled, not a prior-mean-only pass; neither branch may be detached without a declared change.

**LOSS-02 (E/A):** reconstruction sums negative log-likelihoods of category, modality, specifics, value, time. It includes presence/time-gate supervision and conditional regression masks. Symbolically:

$$
\mathcal L_{{rec},t}=\ell_c+\ell_m+\ell_{s\text{-presence}}
+b_t^s\ell_s+\mathbf1_{\rm value\ present}\ell_v
+\ell_{\delta\text{-positive}}+\mathbf1_{\delta>0}\ell_\delta.
$$

CE/BCE are natural categorical/Bernoulli candidates, but continuous likelihoods, fixed/learned variances, regression error (MSE/cosine/Huber), feature normalization, component weights, dimension reduction, and minibatch denominators are not specified. The weighted objective is ELBO-inspired; arbitrary weighted regressions cannot be called an exact probabilistic ELBO without specifying likelihood normalization.

**LOSS-03 (E/D):** linear KL anneal from 0 to 1 over 10 epochs, separately from 6-epoch LR warmup. Candidate interpolation $\beta(e)=\min(1,\max(0,e/10))$ uses fractional epoch e (A). During beta=0 the variance floor alone does not penalize posterior collapse; monitor it rather than asserting the model structurally forbids it at every step.

**LOSS-04 (D/A):** elementwise score ownership multiplies *all* three objective branches before reduction. Unknown final-event gap must have an availability mask rather than an invented zero label. It is unresolved whether posterior $e_t$ includes that next gap, whether a window's final event has an off-window successor, and whether raw final patient tokens are scored for time.

**OPT-01 (E):** Muon for hidden 2D matrices; AdamW for embeddings, heads, and non-2D parameters. Rank alone is insufficient: head matrices and embeddings are also 2D but explicitly assigned to AdamW.

**OPT-02 (E/A):** peak LR $6\times10^{-4}$, cosine decay with 6-epoch linear warmup, weight decay 0.01, gradient clip norm 10. LR group multipliers, cosine floor, total epochs, effective batch, accumulation, Muon momentum/Newton-Schulz steps, AdamW betas/epsilon, decay exclusions, precision, and distributed setup are missing.

**OPT-03 (E):** paper computing environment Python 3.12, PyTorch 2.10, CUDA 12.8, up to 8 NVIDIA A100 80 GB. This specifies the authors' environment, not verified availability in this repository.

## 9. Complete reported pretraining hyperparameters

**HP-01 (E/D):** Table 16, p. 50:

| Group | Parameter | Value/status |
|---|---|---|
| Architecture | Trainable parameters | 150,322,970 E |
| Architecture | Hidden dim / layers / heads | 768 / 12 / 12 E |
| Architecture | Per-head dim / nominal FFN width | 64 / 6144 D |
| Architecture | FFN expansion | 8 E |
| Architecture | Latent dim | 384 E |
| Architecture | Context tokens | 2048 E; prefix accounting A |
| Architecture | Mask | Causal autoregressive E |
| Architecture | Position encoding | Bidirectional temporal delta attention E |
| Architecture | Temporal encoding layers | 0-3 E; indexing interpretation A |
| Architecture | Numeric basis | 22 dyadic scales, $2^{-7}$ through $2^{14}$ E |
| Architecture | LayerScale init | $10^{-4}$ E |
| Optimization | Optimizer | Muon hidden matrices + AdamW embeddings/heads/non-2D E |
| Optimization | Peak LR / schedule / warmup | $6\times10^{-4}$ / cosine / 6 epochs E |
| Optimization | Weight decay / gradient clip | 0.01 / norm 10 E |
| Regularization | Dropout / drop path / attention dropout | 0.1 / 0.1 / 0.1 E |
| Objective | Prior / posterior reconstruction weights | 0.1 / 0.3 E |
| Variational | KL max / linear warmup | 1 / 10 epochs E |
| Variational | Prior sigma floor / smooth cap | 0.05 / 2 E |
| Temporal math | Sinusoid base | $10^4$ E, Methods p. 17 |
| Data | Overlap / minimum scored clinical tokens | 256 / 64 E, Methods p. 15 |
| Data | Patient split / demographic tokens | 70/15/15 / 6 E |

All unreported hyperparameters are enumerated as A-IDs; no training duration, batch size, or optimizer defaults are filled in as paper facts.

## 10. Inference procedures and information sets

**ROLL-01 (E):** predict, decode, re-encode, append, then recompute/run model for each next event. Generated specifics and value embeddings remain in frozen encoder spaces; raw image/video/text/waveform generation is not described or required. No independent per-component perturbation is authorized by the shared-latent mechanism.

**ROLL-02 (E/A):** free rollout uses fresh prior samples and predicted outgoing gaps under the literal candidate in section 4.2. Time-controlled rollout supplies the true forward gap at each step through temporal attention. The latter is an oracle timing diagnostic, not an unrestricted prospective forecast. How step-indexed oracle gaps align after generated categories/counts diverge remains unspecified.

**ROLL-03 (E/A):** fixed token budgets cap generation; horizon, target-event, and budget exhaustion are distinct stopping states. Long consecutive zero-gap tokens must be allowed; adding epsilon to force time to advance changes the model. Optional stall detection may terminate and mark invalid, not change timestamps. Death/discharge stopping policies and context eviction must be explicit.

| Information | Teacher-forced pretraining | Free inference | Time-controlled evaluation | Frozen-state analysis |
|---|---|---|---|---|
| Previous event content | Ground truth | Observed then generated | Observed then generated | Ground truth through extraction point |
| Target embedding $e_t$ | Posterior only, plus future use as context | Unavailable | Unavailable | Only observed events |
| Prior input | $h_{t-1}$ | $h_{t-1}$ | $h_{t-1}$ | Not needed for h extraction |
| Last query's next gap | Ground truth as equations state | Seed initially, then predicted schedule under candidate | Ground-truth oracle | Must disclose/sanitize at extraction cutoff |
| Metadata | Window-end snapshot as described | Snapshot available at prompt cutoff under proposed causal policy | Same, unless separately disclosed | Window/chunk end could reveal future attributes |
| External encoders | Frozen cached embeddings | Generated embedding reuse | Generated embedding reuse | Frozen cached embeddings |
| Latent | Posterior and auxiliary prior samples | Prior only | Prior only | h before latent |
| Next content label | Reconstruction target | No access | No access | Retrieval/probe label separately |

## 11. Scientific evaluation specification

### 11.1 Forecasting, Figs. 3a-i, Extended Data Table 1, Supplementary Tables 2-3

**EVAL-01 (E):** 13,551 eligible test patients; anchor 24 h after most recent hospital admission; observed history through anchor; $M=50$ rollouts; horizons {1,2,4,6,12,24,48,72} h. Fig. 3a displays 128-token rollouts; the Methods specifies a fixed budget without universally fixing its number. Record the selected experiment budget and the uncertainty.

For each class and horizon, empirical rollout occurrence frequency estimates occurrence probability; mean generated count estimates event count (D). Compute macro AUROC, macro Brier, macro count MAE for event categories and modalities; ECE/reliability as in Fig. 3. Report unsupported/one-class classes, truncation coverage, and patient bootstrap 95% CIs using 1,000 resamples. Persistence marks a class present iff observed in the preceding H-window; marginal prevalence is the chance floor. Evaluation smoothing is not explicitly stated for this experiment.

Reference values (p. 35, point estimates):

| Granularity/variant | AUROC at 1/6/24/72 h | Brier at 1/6/24/72 h |
|---|---|---|
| Type, free | .747/.826/.829/.820 | .045/.073/.092/.105 |
| Type, oracle timing | .866/.854/.856/.842 | .032/.061/.089/.110 |
| Modality, free | .764/.818/.811/.813 | .097/.101/.094/.087 |
| Modality, oracle timing | .952/.894/.839/.828 | .038/.075/.092/.084 |

Use these tables rather than treating the main-text .83-.95 summary as every horizon's result. Fig. 3h says 30 evaluable types while Extended Data Table 1 says vocabulary C=31; macro exclusion policy needs clarification.

### 11.2 Zero-shot outcomes

**EVAL-02 (E):** no time control after the seeded initial gap. LOS and 72 h mortality: cutoff admission+48 h, 2,000 randomly selected test patients, 50 draws, 128 tokens/draw, tau=1. Readmission: cutoff one timestep *before* discharge to avoid inter-event-delta leakage; discharge is the reference clock, 1,000 patients, 25 draws, 1,024 tokens, tau=1.

Find first generated target event. A rollout that reaches beyond the task threshold without that event counts as prolonged LOS, or negative mortality/readmission. If token budget expires before target or threshold, it is invalid and excluded. Laplace smoothing 0.5 gives candidate $(n_++0.5)/(n_{\rm valid}+1)$ (D); all-invalid-patient handling remains unspecified. Initial task-specific seed gaps are stated but not numerically given except counterfactuals.

Report AUROC/balanced accuracy/Brier/ECE/coverage, with patient-level 95% intervals. Extended Data Table 2: LOS .741/.683/.203/.201/.804; readmission .609/.569/.232/.146/.645; mortality .972/.821/.035/.085/.822. Clarify positive-label orientation: Extended Data Fig. 1 uses first-bin class 0 even for early mortality/readmission, unlike conventional clinical-positive coding. LOS >72 versus >=72 also needs exact boundary handling.

### 11.3 Intervention-conditioned simulation

**EVAL-03 (E):** test-set sepsis markers; first saline or lactated Ringer's within 6 h of marker; earliest eligible episode; exclude records ending before intervention+24 h without terminal event and those with fewer than 256 events through that point; N=1,085. Marker definition and token IDs are not specified.

Prompt includes intervention; counterfactual changes fluid specifics while preserving factual event time. 25 free-mode trajectories per arm; initial seed 1 h; constrain subsequent crystalloid choice to arm; share latent noise across arms. Pair by patient, trajectory, and generation step, not by identical generated timestamps. Compare only patients with defined outcomes in both arms for paired effects.

Endpoints: 30-day in-hospital mortality, LOS, MAKE-30 (death, new RRT, or final creatinine >=2x baseline within 30 d), anchored to intervention. Scalar inversion, RRT dictionary, renal baseline/final selection, in-hospital-state tracking, and composite missingness rules require decisions. Table 3 distinguishes explicit-outcome mortality from all-run mortality; never merge their denominators.

Reference contrast saline minus Ringer's: mortality +8.86 percentage points among explicit outcomes, +4.30 among all runs; LOS +40.5 h; MAKE-30 +8.92 points. Matched observed versus simulated: observed in-hospital mortality 21.0% versus explicit-outcome simulated 73.7%; observed LOS 253.7 h versus simulated 170.1 h. Supplementary Table 4 reports factual ECE 27.683 points; Table 5 coverage ~68.6-71.8%. This is a reproduction of intervention-conditioned forecasts; sign agreement with SMART is not causal validation.

### 11.4 Representation and probing

**EVAL-04 (E):** use frozen 768-dimensional h. Logistic regression with feature standardization, L2 C=1, L-BFGS; stratified 5-fold x 3 seeds; one fit per target/position; each class >=5 instances per fold. Last stay; pre-admission, admission, +12/+24/+36/+48 h, discharge. Remove stays ending before each horizon. Three main outcome families, early mortality, 15 ICD chapters, 29 Quan-Elixhauser flags. Fit scaling only within training folds (D anti-leakage protocol; paper does not explicitly locate scaler fit).

Discharge is retrieval. Earlier points mix retrieval and prediction according to the paper; future gap/metadata handling adds a separate audit concern. Supplementary Tables 7-10 hold individual target and subgroup comparisons; do not interpret .998 discharge mortality AUROC as prospective prognosis.

**EVAL-05 (E/A):** state trajectories: arclength $\sum_t\|h_{t+1}-h_t\|$, cosine autocorrelation by lag; PaCMAP and chain-UMAP with sequence neighbors forced into neighbor graph. Surprise is unweighted KL(q||p), reported per category in Supplementary Table 1 (ECG median 38, CXR 37, ECHO 22.5; Vitals .0256, Lab Test .0242 nats). Exact projection seeds/graph parameters and KL aggregation need manifests.

### 11.5 NEWS2 retrieval

**EVAL-06 (E):** use separate, non-overlapping windows. Open on first of respiratory rate, oxygen saturation, temperature, systolic BP, heart rate; keep latest per parameter; close at all five, span <=1 h. Carry oxygen supplementation/consciousness up to 8 h; neither opens a window. Default room air/alert; consciousness from LoC text else GCS; exact identifiers, unit conversions, implausibility filters unspecified. Anchor label to last contributing token; this is current-state retrieval.

~1.83M windows overall, 272,698 over NOAH test patients. Split the latter's patients 70/15/15 for NEWS2 fitting; final holdout 39,952 windows/4,414 patients. Standardize from training subset. MLP 768 -> 512 -> 256 -> 1, LayerNorm/GELU/dropout .1; AdamW LR .001, decay .01, batch 4096, max 60 epochs, validation early stop, 3 seeds. Layer ordering and patience unspecified.

MAE/RMSE/R2/Spearman/within-patient R2/quadratic-weighted kappa/band macro F1/AUROC >=5 and >=7; 1,000 patient-bootstrap draws. Baselines constant mean, median, age-only ridge, 768-state ridge. Reference MLP MAE 1.336, RMSE 1.784, R2 .730, within-patient R2 .252, Spearman .862, kappa .757, AUROC >=7 .905; Fig. 5 gives >=5 .944. Retrieve rounded/clipped bands 0-4, 5-6, >=7; score range reported 0-20. Application to generated non-window states is out-of-distribution relative to supervised anchors.

### 11.6 Survival

**EVAL-07 (E):** last stay, >5 events, duration >=1 h as operationally described (text says “at least one hour”); N=26,476. At p in {0,.2,.4,.6,.8,1}, pick last clinical event strictly before fraction pT, resolve concurrent events by last event. p=0 entering-hospital description and strict-before rule conflict; resolve explicitly.

$$
S(t\mid x)=\exp[-\int_0^t h(u\mid x)du],\quad
h(t\mid x)=h_0(t)\exp h(x).
$$

DeepSurv predicts neural risk; death is event, discharge without death right-censored; endpoint from admission is unchanged across positions. This uses realized stay-duration fractions, not a deployment landmark available prospectively. Nested five-fold evaluation; reserve 10% training for tuning and additional 10% early stopping; Optuna TPE 100 trials maximizing Antolini time-dependent C-index. D-calibration 20 equal probability bins, chi-square p<.05 denotes miscalibration; IBS also reported.

Table 14 search: optimizer Adam/SGD; weight decay [1e-6,1e-2]; momentum [0,.9]; LR [1e-5,1e-2]; dropout [0,.5]; widths {32,64,128,256}; depth expression “for depth in [1,6]” is ambiguous between literal two depths and 1-6 range. Table 13 td-CI means .760/.784/.793/.813/.830/.849; IBS .023/.022/.021/.023/.023/.024. Cox loss, baseline hazard estimator, ties, time grids, censoring weights and hyperparameter sampling distributions are unspecified.

### 11.7 Explainability and ablations

**EVAL-08 (E/A):** 144 heads; attention relation lift over uniform; paper reports 141 specialized, 76 same-type longitudinal, 65 observation/reporting. Retain raw attention and label grouping definitions. Temporal ablations CC/TC/CT/TT and total-variation distance; integrated gradients with event-type-only baseline (value/specifics/time decompositions); visible-history ablation {48h,72h,1wk,1mo,3mo,1y,full}, M=25, elderly age >=75. Integration path/steps, normalization and specialization thresholds are missing. These analyses support mechanism inspection, not a test that every new run must reproduce exactly 141 heads.

## 12. Audit release gate

Critical implementation blockers: event/temporal units and mappings; temporal input encoding and last-gap handling; time-head alignment; numeric basis; backbone/MLP definitions and count reconciliation; continuous reconstruction likelihood/reduction; window length/chunk policy; effective batch/total epochs/optimizer internals. Their consequences, alternatives, tests, and resolution gates are in the other four documents.

Acceptance follows [TEST_PLAN.md](TEST_PLAN.md); current implementation and execution status is recorded in [PROJECT_STATUS.md](PROJECT_STATUS.md).


The controlled evaluation contract is operationalized in `src/evaluation/contracts.py`, `landmarks.py` and `outcomes.py`, with regression tests in `tests/test_evaluation_contracts.py`. Two distinct modes, `paper_reproduction` and `strict_prospective`, require explicit endpoint, cutoff, availability, next-gap, demographic and probe choices; a mode string does not certify conformance. The [G01–G15 remediation matrix and M01–M12 decisions](ARCHITECTURE_AUDIT.md#controlled-remediation-preserved-evidence-and-revision-separation) record implemented guards, pending real-data integration and approval-required choices. Release gates reject full-record pooling as a landmark, future/unavailable prospective evidence, unsupported prospective reporting and missing or mismatched experiment artifacts. These are engineering safeguards; no unpublished paper method is inferred and no training behavior is changed by this guard phase.
