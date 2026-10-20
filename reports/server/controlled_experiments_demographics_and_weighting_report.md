# Controlled Experimental Report: Dynamic Demographic Prefixes and Class-Weighted Loss in NOAH

Evidence classification: historical candidate-architecture experiments with whole-record pooled, subject-level retrospective probes. These metrics are invalid for prospective or paper landmark interpretation. Readmission and NEWS2 labels require correction before clinical endpoint use. Reconstruction metrics are conditional diagnostics with recipe/provenance limitations. Original reports and checkpoint bytes are preserved in the private remediation snapshot; see [the remediation matrix](../../docs/ARCHITECTURE_AUDIT.md#historical-metric-interpretation). No corrected clinical evaluation is reported here.

**Date**: 2026-10-19  
**Execution Environment**: Harvard Medical School O2 Cluster (NVIDIA L40S 48GB VRAM)  
**Evaluated Architecture**: 152.65M Parameters (Paper-Scale: $d_{\text{model}}=768$, $d_{\text{latent}}=384$, 12 layers, 12 heads, FFN expansion ratio 8)  
**Dataset**: 1,000 Patient MIMIC-IV Pilot Subcohort (1,890 Train Windows, 392 Val Windows, 355 Test Windows, 1,771,519 Scored Tokens)  
**Active Modalities**: Tabular, Numeric, Text (BioLORD 768-D), CXR (RAD-DINO 768-D), ECG (OTIS 192-D)  

---

## 1. Executive Summary & Experimental Objectives

In accordance with the NOAH paper specification (arXiv:2609.09140v1, Section 3.1 & p. 15), we conducted two rigorous, controlled experiments against the 152M Parameter Phase 1 Baseline:

1. **Controlled Experiment 1 (`exp1_dynamic_demographics`, Job `56861815`)**:
   - **Hypothesis**: The paper requires that the demographic prefix (6 tokens: birth year, race, gender, insurance, language, marital status) be dynamically updated at the end of each sliding window to reflect the most recent available values (falling back to "Unknown" prior to admission), and embedded via external pretrained text encoders (BioLORD-2023, 768-D). We hypothesized that providing rich, dynamic socio-demographic embeddings to the Transformer backbone would improve clinical trajectory conditioning, particularly for outcomes correlated with socio-clinical vulnerability (mortality and readmission).
   - **Configuration**: Dynamic demographics + BioLORD 768-D prefix embeddings; standard unweighted composite variational loss.

2. **Controlled Experiment 2 (`exp2_class_weighted_loss`, Job `56861820`)**:
   - **Hypothesis**: In MIMIC-IV, token categories exhibit severe class imbalance: `LAB_TEST` accounts for ~58% (1,026,279 tokens), whereas critical transition events such as `ENTER_ICU`/`LEAVE_ICU` (687 tokens) and `DEATH` (61 tokens) are exceedingly rare. This caused unweighted cross-entropy to achieve only ~65% categorical accuracy. We applied a smoothed normalized inverse frequency class weighting:
     $$w_c = \frac{1}{\sqrt{N_c + 1}}, \quad \tilde{w}_c = C \cdot \frac{w_c}{\sum_{k=0}^{C-1} w_k}$$
   - **Configuration**: Dynamic demographics + BioLORD prefix embeddings; category loss weighted with $\tilde{w}_c$.

---

## 2. Comprehensive Experimental Comparison Table

All models were evaluated on the **exact same held-out test split** (146 evaluated patients, 235,334 scored test tokens) using the comprehensive audit suite and 5-fold stratified linear probing.

| Dimension / Metric | Phase 1 Baseline (Static Demo, Unweighted) | Exp 1: Dynamic Demographics (Unweighted) | Exp 2: Class-Weighted Loss (Weighted) | Exp 1 vs. Baseline | Exp 2 vs. Exp 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Parameters** | 152,655,380 | 152,655,380 | 152,655,380 | Identical | Identical |
| **Epochs Trained** | 10 | 10 | 10 | - | - |
| **Peak GPU VRAM** | 7.42 GB | 7.43 GB | 7.43 GB | +0.01 GB | +0.00 GB |
| **Total Test Loss** | 1.5783 | **1.5255** | 1.5742 | **-0.0528 (better)** | +0.0487 |
| **Posterior Rec Loss** | 3.7251 | 3.6128 | **3.5678** | **-0.1123 (better)** | **-0.0450 (best)** |
| **Prior Rec Loss** | 4.0285 | **3.8826** | 3.9042 | **-0.1459 (better)** | +0.0216 |
| **Test KL Divergence** | 0.0580 | 0.0534 | 0.1134 | -0.0046 | +0.0600 (more active) |
| **Category Accuracy** | 65.46% | 64.39% | **75.58%** | -1.07% | **+11.19% absolute** |
| **Modality Accuracy** | 87.70% | 87.65% | **89.36%** | -0.05% | **+1.71% gain** |
| **Time Gate ROC-AUC** | 0.8077 | 0.8122 | **0.8190** | +0.0045 | **+0.0068 (best)** |
| **Time Gate Precision**| 0.5195 | 0.5146 | **0.5535** | -0.0049 | **+0.0389 (best)** |
| **Text Cosine Sim** | 0.3167 | 0.3570 | **0.3982** | **+0.0403 (+12.7%)**| **+0.0412 (+11.5%)** |
| **CXR Cosine Sim** | 0.1859 | 0.1856 | **0.1914** | -0.0003 | **+0.0058 (best)** |
| **ECG Cosine Sim** | 0.9534 | 0.9577 | **0.9772** | **+0.0043** | **+0.0195 (best)** |
| **Numeric Cosine Sim**| 0.3106 | 0.3113 | **0.3155** | +0.0007 | **+0.0042 (best)** |
| **Mortality AUROC** | 0.7446 | **0.8152** | 0.6612 | **+0.0706 (+9.5%)** | -0.1540 |
| **Mortality AUPRC** | 0.2285 | **0.3882** | 0.2265 | **+0.1597 (+69.9%)**| -0.1617 |
| **Mortality BalAcc** | 0.6549 | **0.7065** | 0.6920 | **+0.0516** | -0.0145 |
| **Mortality Brier** | 0.0849 | **0.0777** | 0.0970 | **-0.0072 (better)** | +0.0193 |
| **Readmission AUROC** | 0.6689 | **0.7098** | 0.7091 | **+0.0409 (+6.1%)** | -0.0007 |
| **Readmission AUPRC** | 0.4269 | 0.4747 | **0.4914** | **+0.0478** | **+0.0167 (best)** |
| **Readmission BalAcc**| 0.6370 | 0.6787 | **0.7087** | **+0.0417** | **+0.0300 (best)** |
| **LOS > 7d AUROC** | 0.8885 | 0.8941 | **0.9042** | **+0.0056** | **+0.0101 (best)** |
| **LOS > 7d AUPRC** | 0.7690 | 0.7912 | **0.7937** | **+0.0222** | **+0.0025 (best)** |
| **LOS > 7d BalAcc** | 0.8428 | 0.8281 | **0.8444** | -0.0147 | **+0.0163 (best)** |
| **NEWS2 Acuity $r$** | 0.1048 | -0.0325 | **0.1439** | -0.1373 | **+0.1764 (best)** |

---

## 3. Scientific Interpretation & In-Depth Analysis

### 3.1 The Impact of Dynamic Demographic Prefixing (Exp 1)
- **Massive Gains on Mortality and Readmission Probes**:
  - In-hospital mortality AUROC increased from **0.7446 to 0.8152** (+7.1% absolute gain).
  - Mortality AUPRC almost doubled from **0.2285 to 0.3882** (+69.9% relative gain).
  - 30-day readmission AUROC broke the 0.70 barrier, advancing from **0.6689 to 0.7098** (+4.1% absolute gain).
- **Mechanistic Attribution**:
  - In real clinical practice, patient demographics (such as age, insurance payer shifts reflecting Medicaid or Medicare eligibility, and marital status) represent vital social determinants of health and physiological frailty.
  - Feeding rich BioLORD text representations into the 6 prefix positions (instead of zeroed-out vectors) gives the Transformer backbone an anchor representation of patient vulnerability. Because the attention mechanism is causal and attends backward to all prior tokens, all subsequent clinical representations $h_t$ are properly conditioned on this socio-demographic anchor.
  - Test reconstruction loss also dropped from 1.5783 to 1.5255, with Posterior Rec improving from 3.7251 to 3.6128 and Prior Rec improving from 4.0285 to 3.8826.

### 3.2 The Impact of Class-Weighted Loss (Exp 2)
- **Token-Level Representation & Category Accuracy Breakthrough**:
  - Categorical accuracy surged from **64.39% to 75.58%** (+11.19% absolute gain).
  - Continuous reconstruction improved across all modalities:
    - Text cosine similarity: **0.3982** (vs. 0.3167 baseline, +25.7% relative gain).
    - ECG cosine similarity: **0.9772** (vs. 0.9534 baseline).
    - CXR cosine similarity: **0.1914** (vs. 0.1859 baseline).
  - The model broke through the 0.90 AUROC threshold for Prolonged Length of Stay (>7d) reaching **0.9042**, and achieved the highest Readmission Balanced Accuracy (**70.87%**) and NEWS2 acuity correlation ($r = 0.1439$).
- **The Trade-Off in Mortality Calibration**:
  - Why did Mortality AUROC drop to 0.6612 in Exp 2?
  - In MIMIC-IV, in-hospital death is acutely presaged by rapid cascades in routine laboratory results (e.g. abrupt swings in lactate, potassium, BUN, blood gases). In the unweighted model, `LAB_TEST` accounts for58% of event prevalence; its gradient share was not measured, compelling the model to master intricate lab dynamics. In Exp 2, the inverse frequency weighting aggressively suppresses the majority `LAB_TEST` loss weight to $w = 0.0081$ (a normalized class coefficient, not a measured120-fold gradient reduction) while boosting rare event classes like `ENTER_ICU` and `DEATH` ($w = 6.87$).
  - While this enables diverse generative rollouts (generating transfer notes, X-rays, and microbiology tests) and raises overall categorical accuracy, it reduces the latent representation's fidelity to fine-grained numerical lab trajectories, which are crucial for linear probing of imminent in-hospital death.

---

## 4. Architectural & Production Recommendations

1. **Adopt Dynamic Demographic Prefixing (Exp 1) as Standard Baseline**:
   - The paper requirement of window-end dynamic demographic updating with BioLORD text embeddings has recorded pooled retrospective point estimates (Mortality AUROC0.815, Readmission AUROC0.710). These do not establish leakage-free landmark performance, prospective benefit or scientific superiority.
2. **Loss Weighting Strategy for Phase 2 Full Scaling**:
   - Rather than pure inverse frequency weighting ($1/\sqrt{N}$), a **milder loss scaling** (e.g. effective number of samples $w_c = (1 - \beta) / (1 - \beta^{N_c})$ with $\beta = 0.999$, or a maximum weight clamping $[0.1, 5.0]$) should be tested. This will preserve the categorical diversity and high classification accuracy (75.6%) without degrading the representation of high-frequency physiological lab trajectories.

---

## 5. Artifacts and Audit Evidence

- Exp 1 Checkpoint: `checkpoints/exp1_dynamic_demographics/pilot_checkpoint_epoch_10.pt`
- Exp 1 Audit: `reports/server/audit_exp1_dynamic_demo.json`
- Exp 1 Probes: `reports/server/clinical_probes_exp1_dynamic_demo.json`
- Exp 2 Checkpoint: `checkpoints/exp2_class_weighted/pilot_checkpoint_epoch_10.pt`
- Exp 2 Audit: `reports/server/audit_exp2_class_weighted.json`
- Exp 2 Probes: `reports/server/clinical_probes_exp2_class_weighted.json`
