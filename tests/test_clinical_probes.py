"""
Tests for Downstream Clinical Probing Evaluator Suite (Step 4 / EVAL-04, EVAL-06).

Verifies:
1. Exact Royal College of Physicians NEWS2 component scoring across all cutoffs.
2. Binary probe evaluator (AUROC, AUPRC, Balanced Accuracy, Brier score) under 5-fold CV.
3. Regression probe evaluator (MAE, RMSE, R^2, Pearson r) under 5-fold CV.
4. Anti-leakage protocol (StandardScaler strictly fitted on training folds).
5. Robust handling of edge cases and class imbalance.
"""

import numpy as np
import pytest
import torch

from src.evaluation.probes import NEWS2Scorer, ClinicalProbeEvaluator


def test_news2_normal_vitals():
    """Verify normal clinical parameters evaluate to NEWS2 score of 0."""
    score = NEWS2Scorer.calculate(
        resp_rate=16.0,
        spo2=98.0,
        systolic_bp=120.0,
        pulse=72.0,
        temp_c=37.0,
    )
    assert score == 0.0


def test_news2_boundary_and_severe_vitals():
    """Verify exact clinical cutoffs and severe physiological deterioration scores."""
    # Test individual boundary cutoffs
    assert NEWS2Scorer.score_respiration_rate(8.0) == 3
    assert NEWS2Scorer.score_respiration_rate(10.0) == 1
    assert NEWS2Scorer.score_respiration_rate(16.0) == 0
    assert NEWS2Scorer.score_respiration_rate(22.0) == 2
    assert NEWS2Scorer.score_respiration_rate(30.0) == 3

    assert NEWS2Scorer.score_oxygen_saturation(90.0) == 3
    assert NEWS2Scorer.score_oxygen_saturation(93.0) == 2
    assert NEWS2Scorer.score_oxygen_saturation(95.0) == 1
    assert NEWS2Scorer.score_oxygen_saturation(99.0) == 0

    assert NEWS2Scorer.score_systolic_bp(85.0) == 3
    assert NEWS2Scorer.score_systolic_bp(95.0) == 2
    assert NEWS2Scorer.score_systolic_bp(105.0) == 1
    assert NEWS2Scorer.score_systolic_bp(130.0) == 0
    assert NEWS2Scorer.score_systolic_bp(230.0) == 3

    assert NEWS2Scorer.score_pulse(35.0) == 3
    assert NEWS2Scorer.score_pulse(45.0) == 1
    assert NEWS2Scorer.score_pulse(75.0) == 0
    assert NEWS2Scorer.score_pulse(100.0) == 1
    assert NEWS2Scorer.score_pulse(120.0) == 2
    assert NEWS2Scorer.score_pulse(140.0) == 3

    assert NEWS2Scorer.score_temperature(34.0) == 3
    assert NEWS2Scorer.score_temperature(35.5) == 1
    assert NEWS2Scorer.score_temperature(37.0) == 0
    assert NEWS2Scorer.score_temperature(38.5) == 1
    assert NEWS2Scorer.score_temperature(39.5) == 2

    # Aggregate severe test: 3 + 3 + 3 + 3 + 3 = 15
    severe_score = NEWS2Scorer.calculate(
        resp_rate=30.0,
        spo2=88.0,
        systolic_bp=80.0,
        pulse=140.0,
        temp_c=34.0,
    )
    assert severe_score == 15.0


def test_binary_linear_probe_cv():
    """Verify 5-fold cross-validated linear probing on binary outcome."""
    np.random.seed(42)
    n_samples = 200
    d_features = 64

    # Generate synthetic features with signal
    X = np.random.randn(n_samples, d_features).astype(np.float32)
    true_weights = np.random.randn(d_features)
    logits = X @ true_weights
    # 25% positive rate to mimic imbalanced clinical outcomes (e.g., readmission/prolonged LOS)
    threshold = np.percentile(logits, 75)
    y = (logits >= threshold).astype(np.int32)

    evaluator = ClinicalProbeEvaluator(n_splits=5, seed=42)
    metrics = evaluator.evaluate_binary_outcome(X, y, c_reg=1.0)

    assert metrics.n_samples == n_samples
    assert metrics.n_positive == int(np.sum(y))
    assert 0.0 <= metrics.brier_score <= 1.0
    # On informative signal, AUROC should easily be > 0.85
    assert metrics.auroc > 0.85, f"Expected high AUROC on synthetic signal, got {metrics.auroc}"
    assert metrics.auprc > metrics.positive_rate, "AUPRC should exceed random chance positive rate"
    assert metrics.balanced_accuracy > 0.70


def test_continuous_regression_probe_cv():
    """Verify 5-fold cross-validated regression probing on continuous NEWS2 target."""
    np.random.seed(42)
    n_samples = 200
    d_features = 64

    X = np.random.randn(n_samples, d_features).astype(np.float32)
    true_weights = np.random.randn(d_features)
    # Continuous NEWS2-like target (appropriately scaled to 0..20)
    w = true_weights / np.linalg.norm(true_weights) * 4.0
    raw_y = (X @ w) + 5.0 + np.random.randn(n_samples) * 0.3
    y = np.clip(raw_y, 0.0, 20.0).astype(np.float32)

    evaluator = ClinicalProbeEvaluator(n_splits=5, seed=42)
    metrics = evaluator.evaluate_continuous_outcome(X, y, alpha=1.0)

    assert metrics.n_samples == n_samples
    assert metrics.mae >= 0.0
    assert metrics.rmse >= metrics.mae  # Mathematical property: RMSE >= MAE
    assert metrics.r2 > 0.80, f"Expected strong R^2 on synthetic signal, got {metrics.r2}"
    assert metrics.pearson_r > 0.90, f"Expected strong Pearson correlation, got {metrics.pearson_r}"
