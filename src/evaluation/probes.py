"""
Clinical Probing Evaluator Suite for NOAH Representations (EVAL-04, EVAL-06).

Implements frozen linear probes and scoring for four core clinical downstream outcomes:
1. In-hospital mortality (Binary classification: AUROC, AUPRC, Balanced Acc, Brier)
2. 30-day readmission (Binary classification: AUROC, AUPRC, Balanced Acc, Brier)
3. Length of stay > 7 days (Binary classification: AUROC, AUPRC, Balanced Acc, Brier)
4. NEWS2 vital sign deterioration (Continuous regression: MAE, RMSE, R^2, Pearson r)

Anti-leakage protocol (D):
- Feature standardization (StandardScaler) is strictly fitted on training folds.
- Cross-validation uses StratifiedKFold (binary) and KFold (regression) with fixed seeds.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


class NEWS2Scorer:
    """
    Standard National Early Warning Score 2 (NEWS2) calculation (RCP London guidelines).
    Aggregates respiratory rate, oxygen saturation, systolic BP, pulse rate, and temperature.
    Score range: 0 to 20.
    """
    @staticmethod
    def score_respiration_rate(rate: float) -> int:
        if rate <= 8.0:
            return 3
        elif 9.0 <= rate <= 11.0:
            return 1
        elif 12.0 <= rate <= 20.0:
            return 0
        elif 21.0 <= rate <= 24.0:
            return 2
        else:  # >= 25.0
            return 3

    @staticmethod
    def score_oxygen_saturation(spo2: float) -> int:
        if spo2 <= 91.0:
            return 3
        elif 92.0 <= spo2 <= 93.0:
            return 2
        elif 94.0 <= spo2 <= 95.0:
            return 1
        else:  # >= 96.0
            return 0

    @staticmethod
    def score_systolic_bp(sbp: float) -> int:
        if sbp <= 90.0:
            return 3
        elif 91.0 <= sbp <= 100.0:
            return 2
        elif 101.0 <= sbp <= 110.0:
            return 1
        elif 111.0 <= sbp <= 219.0:
            return 0
        else:  # >= 220.0
            return 3

    @staticmethod
    def score_pulse(pulse: float) -> int:
        if pulse <= 40.0:
            return 3
        elif 41.0 <= pulse <= 50.0:
            return 1
        elif 51.0 <= pulse <= 90.0:
            return 0
        elif 91.0 <= pulse <= 110.0:
            return 1
        elif 111.0 <= pulse <= 130.0:
            return 2
        else:  # >= 131.0
            return 3

    @staticmethod
    def score_temperature(temp_c: float) -> int:
        if temp_c <= 35.0:
            return 3
        elif 35.1 <= temp_c <= 36.0:
            return 1
        elif 36.1 <= temp_c <= 38.0:
            return 0
        elif 38.1 <= temp_c <= 39.0:
            return 1
        else:  # >= 39.1
            return 2

    @classmethod
    def calculate(
        cls,
        resp_rate: Optional[float] = None,
        spo2: Optional[float] = None,
        systolic_bp: Optional[float] = None,
        pulse: Optional[float] = None,
        temp_c: Optional[float] = None,
    ) -> float:
        """Calculates total NEWS2 score for available parameters."""
        total = 0.0
        if resp_rate is not None:
            total += cls.score_respiration_rate(resp_rate)
        if spo2 is not None:
            total += cls.score_oxygen_saturation(spo2)
        if systolic_bp is not None:
            total += cls.score_systolic_bp(systolic_bp)
        if pulse is not None:
            total += cls.score_pulse(pulse)
        if temp_c is not None:
            total += cls.score_temperature(temp_c)
        return float(min(total, 20.0))


@dataclass
class BinaryProbeMetrics:
    auroc: float
    auprc: float
    balanced_accuracy: float
    brier_score: float
    n_samples: int
    n_positive: int
    positive_rate: float


@dataclass
class RegressionProbeMetrics:
    mae: float
    rmse: float
    r2: float
    pearson_r: float
    n_samples: int


class ClinicalProbeEvaluator:
    """
    Evaluates linear probing heads on frozen representations h_t.
    """
    def __init__(self, n_splits: int = 5, seed: int = 42):
        self.n_splits = n_splits
        self.seed = seed

    def evaluate_binary_outcome(
        self,
        X: np.ndarray,
        y: np.ndarray,
        c_reg: float = 1.0,
    ) -> BinaryProbeMetrics:
        """
        Stratified cross-validated linear probe evaluation for binary outcomes.
        """
        assert len(X) == len(y), "Feature and label counts must match."
        n_samples = len(y)
        n_pos = int(np.sum(y))
        pos_rate = float(np.mean(y))

        # Check for minimum positive class count per fold
        if n_pos < self.n_splits or (n_samples - n_pos) < self.n_splits:
            raise ValueError(
                f"Insufficient class samples for {self.n_splits}-fold CV: "
                f"positives={n_pos}, negatives={n_samples - n_pos}"
            )

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        y_probs = np.zeros(n_samples, dtype=np.float64)
        y_preds = np.zeros(n_samples, dtype=np.int32)

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            # Fit scaler strictly on training fold (Anti-Leakage Protocol D)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = LogisticRegression(
                C=c_reg,
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=self.seed,
            )
            clf.fit(X_train_scaled, y_train)

            # Predict probabilities
            probs = clf.predict_proba(X_test_scaled)[:, 1]
            y_probs[test_idx] = probs
            y_preds[test_idx] = (probs >= 0.5).astype(np.int32)

        auroc = float(roc_auc_score(y, y_probs))
        auprc = float(average_precision_score(y, y_probs))
        bal_acc = float(balanced_accuracy_score(y, y_preds))
        brier = float(brier_score_loss(y, y_probs))

        return BinaryProbeMetrics(
            auroc=auroc,
            auprc=auprc,
            balanced_accuracy=bal_acc,
            brier_score=brier,
            n_samples=n_samples,
            n_positive=n_pos,
            positive_rate=pos_rate,
        )

    def evaluate_continuous_outcome(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 1.0,
    ) -> RegressionProbeMetrics:
        """
        Cross-validated ridge regression probe evaluation for continuous targets (NEWS2).
        """
        assert len(X) == len(y), "Feature and label counts must match."
        n_samples = len(y)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        y_preds = np.zeros(n_samples, dtype=np.float64)

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            # Fit scaler strictly on training fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            reg = Ridge(alpha=alpha, random_state=self.seed)
            reg.fit(X_train_scaled, y_train)

            y_preds[test_idx] = reg.predict(X_test_scaled)

        mae = float(mean_absolute_error(y, y_preds))
        rmse = float(root_mean_squared_error(y, y_preds))
        r2 = float(r2_score(y, y_preds))
        pr_corr, _ = scipy.stats.pearsonr(y, y_preds)

        return RegressionProbeMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2,
            pearson_r=float(pr_corr),
            n_samples=n_samples,
        )
