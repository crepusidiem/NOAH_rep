"""
Comprehensive Probing Protocol Audit Script:
1. Repeated Stratified 5-Fold CV across 20 random seeds on Test representations (Exp 1 vs Exp 2).
2. Separate Development Set Probing: Train probe on Val (N=150) or Train (N=700), evaluate on held-out Test (N=146).
3. Downstream metrics verification (reconciling LOS AUROC 0.8941 vs 0.8251 typo).
4. Teacher-forced vs free rollout verification.
"""

import json
import yaml
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss

from scripts.evaluate_clinical_probes import load_model, extract_patient_representations_and_news2, extract_patient_labels


def run_audit(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    from src.evaluation.contracts import ContractError
    raise ContractError("Evaluation/aggregation blocked by remediation: require approved endpoint, landmark and experiment manifests; legacy source is snapshotted.")
    print("=" * 80)
    print(f"PROBING PROTOCOL AUDIT & RECONCILIATION SUITE (Device: {device})")
    print("=" * 80)

    with open("configs/paper_scale_config.yaml") as f:
        config = yaml.safe_load(f)

    data_paths = config.get("data_paths", {})
    mimic_root = data_paths.get("mimic_root", "/home/wey371/rajpurkarlab/datasets")
    mimic_hosp = Path(mimic_root) / data_paths.get("hosp_dir", "mimic-iv/physionet.org/files/mimiciv/2.2/hosp")
    mimic_ed = Path(mimic_root) / data_paths.get("ed_dir", "mimic-iv-ed/physionet.org/files/mimic-iv-ed/2.2/ed")

    # 1. Load Checkpoints
    ckpt1_p = Path("checkpoints/exp1_dynamic_demographics/pilot_checkpoint_epoch_10.pt")
    ckpt2_p = Path("checkpoints/exp2_class_weighted/pilot_checkpoint_epoch_10.pt")

    print(f"Loading Exp 1 checkpoint from {ckpt1_p}...")
    m1 = load_model(ckpt1_p, config, device)

    print(f"Loading Exp 2 checkpoint from {ckpt2_p}...")
    m2 = load_model(ckpt2_p, config, device)

    # -------------------------------------------------------------------------
    # PART 1: EXTRACT REPRESENTATIONS & LABELS
    # -------------------------------------------------------------------------
    print("\n>>> Extracting patient representations for Test and Val sets...")
    test_dir = Path("data/processed_dynamic_demo/test")
    val_dir = Path("data/processed_dynamic_demo/val")
    train_dir = Path("data/processed_dynamic_demo/train")

    reps1_test, _ = extract_patient_representations_and_news2(m1, test_dir, device)
    reps2_test, _ = extract_patient_representations_and_news2(m2, test_dir, device)

    reps1_val, _ = extract_patient_representations_and_news2(m1, val_dir, device)
    reps2_val, _ = extract_patient_representations_and_news2(m2, val_dir, device)

    reps1_train, _ = extract_patient_representations_and_news2(m1, train_dir, device)
    reps2_train, _ = extract_patient_representations_and_news2(m2, train_dir, device)

    # Sorted sids for deterministic evaluation
    sids_test = sorted(list(reps1_test.keys()))
    sids_val = sorted(list(reps1_val.keys()))
    sids_train = sorted(list(reps1_train.keys()))

    labels_test = extract_patient_labels("data/pilot_cohort_manifest.csv", str(mimic_hosp), str(mimic_ed), target_sids=sids_test)
    labels_val = extract_patient_labels("data/pilot_cohort_manifest.csv", str(mimic_hosp), str(mimic_ed), target_sids=sids_val)
    labels_train = extract_patient_labels("data/pilot_cohort_manifest.csv", str(mimic_hosp), str(mimic_ed), target_sids=sids_train)

    eval_sids_test = [s for s in sids_test if s in labels_test]
    eval_sids_val = [s for s in sids_val if s in labels_val]
    eval_sids_train = [s for s in sids_train if s in labels_train]

    print(f"Test evaluated patients:  {len(eval_sids_test)} (Mortality: {sum(labels_test[s]['mortality'] for s in eval_sids_test)})")
    print(f"Val evaluated patients:   {len(eval_sids_val)} (Mortality: {sum(labels_val[s]['mortality'] for s in eval_sids_val)})")
    print(f"Train evaluated patients: {len(eval_sids_train)} (Mortality: {sum(labels_train[s]['mortality'] for s in eval_sids_train)})")

    X1_test = np.array([reps1_test[s] for s in eval_sids_test], dtype=np.float32)
    X2_test = np.array([reps2_test[s] for s in eval_sids_test], dtype=np.float32)
    y_mort_test = np.array([labels_test[s]["mortality"] for s in eval_sids_test], dtype=np.int32)
    y_los_test = np.array([labels_test[s]["los_gt_7d"] for s in eval_sids_test], dtype=np.int32)
    y_readm_test = np.array([labels_test[s]["readmission_30d"] for s in eval_sids_test], dtype=np.int32)

    X1_val = np.array([reps1_val[s] for s in eval_sids_val], dtype=np.float32)
    X2_val = np.array([reps2_val[s] for s in eval_sids_val], dtype=np.float32)
    y_mort_val = np.array([labels_val[s]["mortality"] for s in eval_sids_val], dtype=np.int32)
    y_los_val = np.array([labels_val[s]["los_gt_7d"] for s in eval_sids_val], dtype=np.int32)
    y_readm_val = np.array([labels_val[s]["readmission_30d"] for s in eval_sids_val], dtype=np.int32)

    X1_train = np.array([reps1_train[s] for s in eval_sids_train], dtype=np.float32)
    X2_train = np.array([reps2_train[s] for s in eval_sids_train], dtype=np.float32)
    y_mort_train = np.array([labels_train[s]["mortality"] for s in eval_sids_train], dtype=np.int32)

    # -------------------------------------------------------------------------
    # PART 2: REPEATED STRATIFIED 5-FOLD CV ACROSS 20 RANDOM SEEDS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: REPEATED STRATIFIED 5-FOLD CV ACROSS 20 RANDOM SEEDS")
    print("=" * 80)

    seeds = [42, 100, 2024, 7, 12345, 999, 54321, 888, 314, 777, 1, 2, 3, 10, 21, 55, 99, 123, 789, 2026]
    
    cv_results = []
    diff_auroc_list = []
    diff_auprc_list = []
    exp1_auroc_list, exp2_auroc_list = [], []
    exp1_auprc_list, exp2_auprc_list = [], []

    print(f"{'Seed':5s} | {'Exp1 AUROC':10s} | {'Exp2 AUROC':10s} | {'Diff AUROC':11s} | {'Exp1 AUPRC':10s} | {'Exp2 AUPRC':10s} | {'Diff AUPRC':11s}")
    print("-" * 80)

    for s in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
        p1 = np.zeros(len(y_mort_test), dtype=np.float64)
        p2 = np.zeros(len(y_mort_test), dtype=np.float64)

        for tr_idx, te_idx in skf.split(X1_test, y_mort_test):
            # Exp 1
            sc1 = StandardScaler()
            X1_tr = sc1.fit_transform(X1_test[tr_idx])
            X1_te = sc1.transform(X1_test[te_idx])
            clf1 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=s)
            clf1.fit(X1_tr, y_mort_test[tr_idx])
            p1[te_idx] = clf1.predict_proba(X1_te)[:, 1]

            # Exp 2 (strictly identical folds)
            sc2 = StandardScaler()
            X2_tr = sc2.fit_transform(X2_test[tr_idx])
            X2_te = sc2.transform(X2_test[te_idx])
            clf2 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=s)
            clf2.fit(X2_tr, y_mort_test[tr_idx])
            p2[te_idx] = clf2.predict_proba(X2_te)[:, 1]

        auc1 = roc_auc_score(y_mort_test, p1)
        auc2 = roc_auc_score(y_mort_test, p2)
        pr1 = average_precision_score(y_mort_test, p1)
        pr2 = average_precision_score(y_mort_test, p2)

        d_auc = auc1 - auc2
        d_pr = pr1 - pr2

        exp1_auroc_list.append(auc1)
        exp2_auroc_list.append(auc2)
        diff_auroc_list.append(d_auc)
        exp1_auprc_list.append(pr1)
        exp2_auprc_list.append(pr2)
        diff_auprc_list.append(d_pr)

        print(f"{s:5d} | {auc1:10.4f} | {auc2:10.4f} | {d_auc:+11.4f} | {pr1:10.4f} | {pr2:10.4f} | {d_pr:+11.4f}")

    print("-" * 80)
    print(f"Exp 1 AUROC across 20 CV seeds: Mean = {np.mean(exp1_auroc_list):.4f} +/- {np.std(exp1_auroc_list):.4f} | [Min: {np.min(exp1_auroc_list):.4f}, Max: {np.max(exp1_auroc_list):.4f}]")
    print(f"Exp 2 AUROC across 20 CV seeds: Mean = {np.mean(exp2_auroc_list):.4f} +/- {np.std(exp2_auroc_list):.4f} | [Min: {np.min(exp2_auroc_list):.4f}, Max: {np.max(exp2_auroc_list):.4f}]")
    print(f"AUROC Difference (Exp1 - Exp2): Mean = {np.mean(diff_auroc_list):.4f} +/- {np.std(diff_auroc_list):.4f} | [Min: {np.min(diff_auroc_list):.4f}, Max: {np.max(diff_auroc_list):.4f}]")
    pos_diff_cnt = sum(d > 0 for d in diff_auroc_list)
    print(f"Consistency of AUROC direction: Exp 1 > Exp 2 in {pos_diff_cnt}/20 seeds ({pos_diff_cnt/20*100:.1f}%)")

    print(f"\nExp 1 AUPRC across 20 CV seeds: Mean = {np.mean(exp1_auprc_list):.4f} +/- {np.std(exp1_auprc_list):.4f} | [Min: {np.min(exp1_auprc_list):.4f}, Max: {np.max(exp1_auprc_list):.4f}]")
    print(f"Exp 2 AUPRC across 20 CV seeds: Mean = {np.mean(exp2_auprc_list):.4f} +/- {np.std(exp2_auprc_list):.4f} | [Min: {np.min(exp2_auprc_list):.4f}, Max: {np.max(exp2_auprc_list):.4f}]")
    print(f"AUPRC Difference (Exp1 - Exp2): Mean = {np.mean(diff_auprc_list):.4f} +/- {np.std(diff_auprc_list):.4f} | [Min: {np.min(diff_auprc_list):.4f}, Max: {np.max(diff_auprc_list):.4f}]")
    pos_pr_cnt = sum(d > 0 for d in diff_auprc_list)
    print(f"Consistency of AUPRC direction: Exp 1 > Exp 2 in {pos_pr_cnt}/20 seeds ({pos_pr_cnt/20*100:.1f}%)")

    # -------------------------------------------------------------------------
    # PART 3: INDEPENDENT DEVELOPMENT SET PROBING (ZERO-LEAKAGE HELD-OUT TEST)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: INDEPENDENT DEVELOPMENT SET PROBING")
    print("=" * 80)

    # Protocol A: Train probe on Val set (N=150, 6 deaths), evaluate on Test (N=146, 8 deaths)
    print("\n--- Protocol A: Train Probe on Validation Set (N=150) -> Test on Held-Out Test (N=146) ---")
    sc1_v = StandardScaler()
    sc2_v = StandardScaler()
    clf1_v = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42)
    clf2_v = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42)

    clf1_v.fit(sc1_v.fit_transform(X1_val), y_mort_val)
    clf2_v.fit(sc2_v.fit_transform(X2_val), y_mort_val)

    p1_val_test = clf1_v.predict_proba(sc1_v.transform(X1_test))[:, 1]
    p2_val_test = clf2_v.predict_proba(sc2_v.transform(X2_test))[:, 1]

    auc1_vt = roc_auc_score(y_mort_test, p1_val_test)
    auc2_vt = roc_auc_score(y_mort_test, p2_val_test)
    pr1_vt = average_precision_score(y_mort_test, p1_val_test)
    pr2_vt = average_precision_score(y_mort_test, p2_val_test)

    print(f"Exp 1 (Val->Test): AUROC = {auc1_vt:.4f} | AUPRC = {pr1_vt:.4f}")
    print(f"Exp 2 (Val->Test): AUROC = {auc2_vt:.4f} | AUPRC = {pr2_vt:.4f}")
    print(f"Val->Test AUROC Diff (Exp1 - Exp2): {auc1_vt - auc2_vt:+.4f}")
    print(f"Val->Test AUPRC Diff (Exp1 - Exp2): {pr1_vt - pr2_vt:+.4f}")

    # Protocol B: Train probe on Train set (N=700, 47 deaths) -> Test on Held-Out Test (N=146, 8 deaths)
    print("\n--- Protocol B: Train Probe on Training Set (N=700) -> Test on Held-Out Test (N=146) ---")
    sc1_tr = StandardScaler()
    sc2_tr = StandardScaler()
    clf1_tr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42)
    clf2_tr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42)

    clf1_tr.fit(sc1_tr.fit_transform(X1_train), y_mort_train)
    clf2_tr.fit(sc2_tr.fit_transform(X2_train), y_mort_train)

    p1_tr_test = clf1_tr.predict_proba(sc1_tr.transform(X1_test))[:, 1]
    p2_tr_test = clf2_tr.predict_proba(sc2_tr.transform(X2_test))[:, 1]

    auc1_tt = roc_auc_score(y_mort_test, p1_tr_test)
    auc2_tt = roc_auc_score(y_mort_test, p2_tr_test)
    pr1_tt = average_precision_score(y_mort_test, p1_tr_test)
    pr2_tt = average_precision_score(y_mort_test, p2_tr_test)

    print(f"Exp 1 (Train->Test): AUROC = {auc1_tt:.4f} | AUPRC = {pr1_tt:.4f}")
    print(f"Exp 2 (Train->Test): AUROC = {auc2_tt:.4f} | AUPRC = {pr2_tt:.4f}")
    print(f"Train->Test AUROC Diff (Exp1 - Exp2): {auc1_tt - auc2_tt:+.4f}")
    print(f"Train->Test AUPRC Diff (Exp1 - Exp2): {pr1_tt - pr2_tt:+.4f}")

    # Also evaluate LOS and Readmission under Train->Test
    print("\n--- Other Clinical Tasks (Train->Test Protocol) ---")
    y_los_train = np.array([labels_train[s]["los_gt_7d"] for s in eval_sids_train], dtype=np.int32)
    y_readm_train = np.array([labels_train[s]["readmission_30d"] for s in eval_sids_train], dtype=np.int32)

    # LOS
    clf1_los = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42)
    clf2_los = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42)
    clf1_los.fit(sc1_tr.fit_transform(X1_train), y_los_train)
    clf2_los.fit(sc2_tr.fit_transform(X2_train), y_los_train)
    p1_los = clf1_los.predict_proba(sc1_tr.transform(X1_test))[:, 1]
    p2_los = clf2_los.predict_proba(sc2_tr.transform(X2_test))[:, 1]
    print(f"LOS >7d AUROC (Train->Test): Exp 1 = {roc_auc_score(y_los_test, p1_los):.4f} | Exp 2 = {roc_auc_score(y_los_test, p2_los):.4f}")

    # Readmission
    clf1_rd = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42)
    clf2_rd = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42)
    clf1_rd.fit(sc1_tr.fit_transform(X1_train), y_readm_train)
    clf2_rd.fit(sc2_tr.fit_transform(X2_train), y_readm_train)
    p1_rd = clf1_rd.predict_proba(sc1_tr.transform(X1_test))[:, 1]
    p2_rd = clf2_rd.predict_proba(sc2_tr.transform(X2_test))[:, 1]
    print(f"Readmission AUROC (Train->Test): Exp 1 = {roc_auc_score(y_readm_test, p1_rd):.4f} | Exp 2 = {roc_auc_score(y_readm_test, p2_rd):.4f}")


if __name__ == "__main__":
    run_audit()
