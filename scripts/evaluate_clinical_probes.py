"""
Evaluates Downstream Clinical Linear Probes on Frozen NOAH Representations (EVAL-04, EVAL-06).

Tasks evaluated:
1. In-Hospital Mortality (AUROC, AUPRC, Balanced Accuracy, Brier Score)
2. 30-Day Readmission (AUROC, AUPRC, Balanced Accuracy, Brier Score)
3. Prolonged Length of Stay > 7 Days (AUROC, AUPRC, Balanced Accuracy, Brier Score)
4. NEWS2 Vital Sign Deterioration Regression (MAE, RMSE, R^2, Pearson r)

Usage:
    python scripts/evaluate_clinical_probes.py \
        --checkpoint checkpoints/pilot/pilot_checkpoint_epoch_10.pt \
        --config configs/pilot_config.yaml \
        --data_dir data/processed/test \
        --device cpu
"""

import argparse
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple, Union
import duckdb
import numpy as np
import torch
from tqdm import tqdm
import yaml

from src.data.schema import ModelWindow, EventCategory
from src.evaluation.probes import ClinicalProbeEvaluator, NEWS2Scorer
from src.models.noah import NOAH


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate downstream clinical probes on NOAH representations.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pilot/pilot_checkpoint_epoch_10.pt")
    parser.add_argument("--config", type=str, default="configs/pilot_config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/processed/test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_report", type=str, default="reports/server/clinical_probes_pilot_report.json")
    return parser.parse_args()


def load_model(checkpoint_path: Path, config: dict, device: str) -> NOAH:
    model = NOAH(
        d_model=config["model"]["d_model"],
        d_latent=config["model"]["d_latent"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        head_dim=config["model"].get("head_dim", config["model"].get("d_head", 64)),
        ffn_ratio=config["model"].get("ffn_expansion", 4),
        dropout=config["model"].get("dropout", 0.1),
        attn_dropout=config["model"].get("attention_dropout", 0.1),
        num_categories=config["feature_dimensions"]["category_vocab_size"],
        num_modalities=6,
        text_dim=config["feature_dimensions"]["content_dims"]["text"] or 768,
        cxr_dim=config["feature_dimensions"]["content_dims"]["cxr"] or 768,
        ecg_dim=config["feature_dimensions"]["content_dims"]["ecg"] or 192,
        echo_dim=config["feature_dimensions"]["content_dims"]["echo"] or 512,
        numeric_dim=config["feature_dimensions"]["content_dims"]["numeric"] or 44,
        temporal_layers=4,
        alpha_post=0.3,
        gamma_prior=0.1,
    ).to(device)

    print(f"Loading checkpoint weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def extract_patient_representations_and_news2(
    model: NOAH,
    data_dir: Path,
    device: str,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    """
    Extracts frozen patient history representations h and computes NEWS2 score from vitals.
    """
    patient_reps: Dict[int, List[np.ndarray]] = {}
    patient_news2: Dict[int, List[float]] = {}

    shard_paths = sorted(list(data_dir.glob("*.pt")))
    print(f"Processing {len(shard_paths)} shards from {data_dir}...")

    with torch.no_grad():
        for shard_p in tqdm(shard_paths, desc="Extracting representations"):
            windows: List[ModelWindow] = torch.load(shard_p, weights_only=False)
            for w in windows:
                sid = int(w.subject_id)
                # Convert to tensors
                cat_ids = torch.tensor(w.category_ids, dtype=torch.long, device=device).unsqueeze(0)
                mod_ids = torch.tensor(w.modality_ids, dtype=torch.long, device=device).unsqueeze(0)
                pos = torch.tensor(w.positions, dtype=torch.float32, device=device).unsqueeze(0)
                ages = torch.tensor(w.ages, dtype=torch.float32, device=device).unsqueeze(0)
                d_prev = torch.tensor(w.delta_prev, dtype=torch.float32, device=device).unsqueeze(0)
                d_next = torch.tensor(w.delta_next, dtype=torch.float32, device=device).unsqueeze(0)
                spec_vec = torch.tensor(w.specifics_vec, dtype=torch.float32, device=device).unsqueeze(0)
                has_spec = torch.tensor(w.has_specifics, dtype=torch.bool, device=device).unsqueeze(0)
                val_vec = torch.tensor(w.value_vec, dtype=torch.float32, device=device).unsqueeze(0)
                has_val = torch.tensor(w.has_value, dtype=torch.bool, device=device).unsqueeze(0)
                attn_mask = torch.tensor(w.attention_mask, dtype=torch.bool, device=device).unsqueeze(0)

                # Input projection
                e = model.projector(
                    category_ids=cat_ids,
                    modality_ids=mod_ids,
                    specifics_vec=spec_vec,
                    has_specifics=has_spec,
                    value_vec=val_vec,
                    has_value=has_val,
                )

                # Causal transformer hidden states
                h = model.backbone(
                    e=e,
                    positions=pos,
                    ages=ages,
                    delta_next=d_next,
                    delta_prev=d_prev,
                    attention_mask=attn_mask,
                )

                # Extract valid representation: mean pool over non-padding tokens
                valid_mask = w.attention_mask  # [L] bool
                if np.sum(valid_mask) > 0:
                    h_valid = h[0, valid_mask, :].cpu().numpy()  # [N_valid, d_model]
                    h_patient = np.mean(h_valid, axis=0)        # [d_model]
                    if sid not in patient_reps:
                        patient_reps[sid] = []
                    patient_reps[sid].append(h_patient)

                # Compute NEWS2 from available vital events in the window
                vital_cats = {EventCategory.VITALS, EventCategory.TRIAGE_ACUITY}
                vital_mask = np.isin(w.category_ids, list(vital_cats)) & valid_mask
                if np.sum(vital_mask) > 0:
                    # Estimate approximate physiological vitals from raw values
                    # Default placeholder calculation for physiological stability
                    raw_vals = np.abs(w.value_vec[vital_mask, :5].mean(axis=-1)) if np.any(w.has_value[vital_mask]) else [70.0]
                    # Map to simulated realistic vitals
                    sim_pulse = float(np.clip(raw_vals.mean() * 100.0, 50.0, 130.0))
                    n2 = NEWS2Scorer.calculate(pulse=sim_pulse, resp_rate=18.0, spo2=97.0, systolic_bp=125.0, temp_c=37.0)
                    if sid not in patient_news2:
                        patient_news2[sid] = []
                    patient_news2[sid].append(n2)

    # Average window representations per patient
    final_patient_reps: Dict[int, np.ndarray] = {
        sid: np.mean(reps, axis=0) for sid, reps in patient_reps.items()
    }
    final_patient_news2: Dict[int, float] = {
        sid: float(np.mean(n2_scores)) if n2_scores else 1.0 for sid, n2_scores in patient_news2.items()
    }

    return final_patient_reps, final_patient_news2


def extract_patient_labels(
    manifest_path: str,
    mimic_hosp_dir: str,
    mimic_ed_dir: str,
    target_sids: List[int],
) -> Dict[int, Dict[str, Union[int, float]]]:
    """
    Extracts ground truth hospital outcomes from MIMIC admissions and ED triage vitals.
    """
    con = duckdb.connect()
    adm_path = Path(mimic_hosp_dir) / "admissions.csv"
    triage_path = Path(mimic_ed_dir) / "triage.csv"

    # Register target sids
    con.execute("CREATE TEMP TABLE target_sids (subject_id BIGINT);")
    con.executemany("INSERT INTO target_sids VALUES (?);", [(s,) for s in target_sids])

    # 1. Hospital outcomes
    query_adm = f"""
        WITH ranked_adm AS (
            SELECT 
                a.subject_id, a.hadm_id, 
                try_cast(a.admittime as timestamp) as admittime,
                try_cast(a.dischtime as timestamp) as dischtime,
                a.hospital_expire_flag,
                datediff('second', try_cast(a.admittime as timestamp), try_cast(a.dischtime as timestamp)) / 86400.0 as los_days,
                lead(try_cast(a.admittime as timestamp)) over (partition by a.subject_id order by try_cast(a.admittime as timestamp)) as next_admittime
            FROM read_csv_auto('{adm_path}') a
            JOIN target_sids t ON a.subject_id = t.subject_id
        )
        SELECT 
            subject_id,
            max(hospital_expire_flag) as has_mortality,
            max(case when los_days > 7.0 then 1 else 0 end) as has_los_gt_7d,
            max(case when datediff('second', dischtime, next_admittime) <= 30*86400 then 1 else 0 end) as has_readm_30d
        FROM ranked_adm
        GROUP BY subject_id
    """
    adm_rows = con.execute(query_adm).fetchall()

    # 2. Triage vitals for NEWS2 calculation
    news2_by_sub: Dict[int, float] = {}
    if triage_path.exists():
        query_triage = f"""
            SELECT 
                t.subject_id,
                avg(t.temperature) as temp_f,
                avg(t.heartrate) as hr,
                avg(t.resprate) as rr,
                avg(t.o2sat) as spo2,
                avg(t.sbp) as sbp
            FROM read_csv_auto('{triage_path}') t
            JOIN target_sids ts ON t.subject_id = ts.subject_id
            GROUP BY t.subject_id
        """
        for sid, tf, hr, rr, spo2, sbp in con.execute(query_triage).fetchall():
            tc = (tf - 32.0) * 5.0 / 9.0 if tf is not None else 37.0
            n2 = NEWS2Scorer.calculate(pulse=hr, resp_rate=rr, spo2=spo2, systolic_bp=sbp, temp_c=tc)
            news2_by_sub[int(sid)] = float(n2)

    con.close()

    labels_by_sub: Dict[int, Dict[str, Union[int, float]]] = {}
    for sid, mort, los7, readm in adm_rows:
        sid_int = int(sid)
        labels_by_sub[sid_int] = {
            "mortality": int(mort or 0),
            "los_gt_7d": int(los7 or 0),
            "readmission_30d": int(readm or 0),
            "news2": news2_by_sub.get(sid_int, 1.0),
        }
    return labels_by_sub


def main():
    args = parse_args()
    print("=" * 70)
    print("NOAH DOWNSTREAM CLINICAL PROBES EVALUATION (EVAL-04, EVAL-06)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data Dir:   {args.data_dir}")
    print(f"Device:     {args.device}")
    print("=" * 70)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 1. Load model
    model = load_model(Path(args.checkpoint), config, args.device)

    # 2. Extract patient representations
    t0 = time.time()
    patient_reps, _ = extract_patient_representations_and_news2(
        model=model,
        data_dir=Path(args.data_dir),
        device=args.device,
    )
    print(f"Extracted representations for {len(patient_reps)} patients in {time.time() - t0:.2f} s.")

    # 3. Extract ground truth clinical labels
    mimic_hosp = Path(config["data_paths"]["mimic_root"]) / config["data_paths"]["hosp_dir"]
    mimic_ed = Path(config["data_paths"]["mimic_root"]) / config["data_paths"]["ed_dir"]
    sids = list(patient_reps.keys())
    labels = extract_patient_labels(
        manifest_path=config["cohort"]["manifest_path"],
        mimic_hosp_dir=str(mimic_hosp),
        mimic_ed_dir=str(mimic_ed),
        target_sids=sids,
    )

    # Filter patients with both representations and labels
    eval_sids = [s for s in sids if s in labels]
    print(f"Aligned {len(eval_sids)} patients with ground-truth clinical labels.")

    X = np.array([patient_reps[s] for s in eval_sids], dtype=np.float32)
    y_mort = np.array([labels[s]["mortality"] for s in eval_sids], dtype=np.int32)
    y_los7 = np.array([labels[s]["los_gt_7d"] for s in eval_sids], dtype=np.int32)
    y_readm = np.array([labels[s]["readmission_30d"] for s in eval_sids], dtype=np.int32)
    y_news2 = np.array([labels[s]["news2"] for s in eval_sids], dtype=np.float32)

    evaluator = ClinicalProbeEvaluator(n_splits=5, seed=config["experiment"]["seed"])

    results = {}
    print("\n--- Running 5-Fold Stratified Clinical Probing Evaluator ---")

    # 1. In-Hospital Mortality
    try:
        mort_metrics = evaluator.evaluate_binary_outcome(X, y_mort)
        results["mortality"] = {
            "auroc": mort_metrics.auroc,
            "auprc": mort_metrics.auprc,
            "balanced_acc": mort_metrics.balanced_accuracy,
            "brier_score": mort_metrics.brier_score,
            "n_samples": mort_metrics.n_samples,
            "n_positive": mort_metrics.n_positive,
            "positive_rate": mort_metrics.positive_rate,
        }
        print(f"1. In-Hospital Mortality: AUROC = {mort_metrics.auroc:.4f} | AUPRC = {mort_metrics.auprc:.4f} | BalAcc = {mort_metrics.balanced_accuracy:.4f}")
    except ValueError as e:
        print(f"1. In-Hospital Mortality: Skipped ({e})")

    # 2. 30-Day Readmission
    try:
        readm_metrics = evaluator.evaluate_binary_outcome(X, y_readm)
        results["readmission_30d"] = {
            "auroc": readm_metrics.auroc,
            "auprc": readm_metrics.auprc,
            "balanced_acc": readm_metrics.balanced_accuracy,
            "brier_score": readm_metrics.brier_score,
            "n_samples": readm_metrics.n_samples,
            "n_positive": readm_metrics.n_positive,
            "positive_rate": readm_metrics.positive_rate,
        }
        print(f"2. 30-Day Readmission:   AUROC = {readm_metrics.auroc:.4f} | AUPRC = {readm_metrics.auprc:.4f} | BalAcc = {readm_metrics.balanced_accuracy:.4f}")
    except ValueError as e:
        print(f"2. 30-Day Readmission: Skipped ({e})")

    # 3. Prolonged Length of Stay (> 7 days)
    try:
        los_metrics = evaluator.evaluate_binary_outcome(X, y_los7)
        results["los_gt_7d"] = {
            "auroc": los_metrics.auroc,
            "auprc": los_metrics.auprc,
            "balanced_acc": los_metrics.balanced_accuracy,
            "brier_score": los_metrics.brier_score,
            "n_samples": los_metrics.n_samples,
            "n_positive": los_metrics.n_positive,
            "positive_rate": los_metrics.positive_rate,
        }
        print(f"3. Prolonged LOS (>7d):   AUROC = {los_metrics.auroc:.4f} | AUPRC = {los_metrics.auprc:.4f} | BalAcc = {los_metrics.balanced_accuracy:.4f}")
    except ValueError as e:
        print(f"3. Prolonged LOS (>7d): Skipped ({e})")

    # 4. NEWS2 Vital Sign Deterioration Regression
    try:
        news2_metrics = evaluator.evaluate_continuous_outcome(X, y_news2)
        results["news2_regression"] = {
            "mae": news2_metrics.mae,
            "rmse": news2_metrics.rmse,
            "r2": news2_metrics.r2,
            "pearson_r": news2_metrics.pearson_r,
            "n_samples": news2_metrics.n_samples,
        }
        print(f"4. NEWS2 Deterioration:   MAE = {news2_metrics.mae:.4f} | RMSE = {news2_metrics.rmse:.4f} | R^2 = {news2_metrics.r2:.4f} | Pearson r = {news2_metrics.pearson_r:.4f}")
    except Exception as e:
        print(f"4. NEWS2 Deterioration: Skipped ({e})")

    # Save output report
    out_path = Path(args.output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport successfully saved to {out_path}")


if __name__ == "__main__":
    main()
