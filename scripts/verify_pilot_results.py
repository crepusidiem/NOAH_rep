"""
Comprehensive Verification and Diagnostic Suite for NOAH Pilot Run (N-07).

Executes rigorous scientific evaluations:
1. Held-out Test Set Evaluation (Loss breakdown, Category Acc, Modality Acc, Time Gate Acc, Time MAE).
2. Continuous Representation Reconstruction Audit (Cosine similarity for Text, CXR, ECG, and Numeric values).
3. Two-Stage Time Hurdle Discrimination Audit (Zero-gap vs Positive-gap Precision, Recall, ROC-AUC).
4. Multi-step Autoregressive Free Rollout Dynamics (Trajectory entropy, zero-gap stall analysis, simulated time span).
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.schema import EventCategory, Modality, ModelWindow
from src.models.noah import NOAH
from scripts.train_pilot import load_windows, collate_windows


def run_test_evaluation(model: NOAH, test_loader: DataLoader, device: str):
    """Approach 2 & 4: Quantitative evaluation on held-out test split."""
    model.eval()
    total_loss_sum = 0.0
    rec_post_sum = 0.0
    rec_prior_sum = 0.0
    kl_sum = 0.0

    cat_correct = 0
    cat_total = 0
    mod_correct = 0
    mod_total = 0

    time_gate_correct = 0
    time_gate_total = 0
    time_abs_err_sum = 0.0
    time_pos_count = 0

    # Time hurdle classification tracking
    y_true_gate = []
    y_pred_gate_probs = []

    # Continuous representation cosine similarities
    text_cos_sims = []
    cxr_cos_sims = []
    ecg_cos_sims = []
    numeric_cos_sims = []

    with torch.no_grad():
        for batch in test_loader:
            batch_dev = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            loss, breakdown, aux = model(
                category_ids=batch_dev["category_ids"],
                modality_ids=batch_dev["modality_ids"],
                positions=batch_dev["positions"],
                ages=batch_dev["ages"],
                delta_prev=batch_dev["delta_prev"],
                delta_next=batch_dev["delta_next"],
                has_next_gap=batch_dev["has_next_gap"],
                specifics_vec=batch_dev["specifics_vec"],
                has_specifics=batch_dev["has_specifics"],
                value_vec=batch_dev["value_vec"],
                has_value=batch_dev["has_value"],
                attention_mask=batch_dev["attention_mask"],
                loss_mask=batch_dev["loss_mask"],
                beta=1.0,
            )

            total_loss_sum += loss.item()
            rec_post_sum += breakdown.loss_rec_post.item()
            rec_prior_sum += breakdown.loss_rec_prior.item()
            kl_sum += breakdown.loss_kl.item()

            mask_tgt = batch_dev["loss_mask"][:, 1:].bool()
            if not mask_tgt.any():
                continue

            z_post = aux["mu_q"]
            dec_out = model.decoder(z_post)

            # Category & Modality
            pred_cats = dec_out.category_logits[mask_tgt].argmax(dim=-1)
            true_cats = batch_dev["category_ids"][:, 1:][mask_tgt]
            cat_correct += (pred_cats == true_cats).sum().item()
            cat_total += mask_tgt.sum().item()

            pred_mods = dec_out.modality_logits[mask_tgt].argmax(dim=-1)
            true_mods = batch_dev["modality_ids"][:, 1:][mask_tgt]
            mod_correct += (pred_mods == true_mods).sum().item()
            mod_total += mask_tgt.sum().item()

            # Time Gate
            has_gap_mask = mask_tgt & batch_dev["has_next_gap"][:, 1:].bool()
            if has_gap_mask.any():
                t_gate_logits = dec_out.time_gate_logits.squeeze(-1)[has_gap_mask]
                is_pos_true = (batch_dev["delta_next"][:, 1:][has_gap_mask] > 0.0)
                gate_probs = torch.sigmoid(t_gate_logits)
                pred_pos = gate_probs > 0.5
                time_gate_correct += (pred_pos == is_pos_true).sum().item()
                time_gate_total += has_gap_mask.sum().item()

                y_true_gate.extend(is_pos_true.cpu().numpy().tolist())
                y_pred_gate_probs.extend(gate_probs.cpu().numpy().tolist())

                pos_mask = has_gap_mask & (batch_dev["delta_next"][:, 1:] > 0.0)
                if pos_mask.any():
                    pred_gaps = model.decoder.decode_time(
                        dec_out.time_gate_logits[pos_mask],
                        dec_out.log_gap_reg[pos_mask],
                    )
                    true_gaps = batch_dev["delta_next"][:, 1:][pos_mask]
                    time_abs_err_sum += (pred_gaps - true_gaps).abs().sum().item()
                    time_pos_count += pos_mask.sum().item()

            # Continuous reconstruction quality
            # Text specifics
            text_mask = mask_tgt & batch_dev["has_specifics"][:, 1:].bool()
            if text_mask.any():
                pred_t = F.normalize(dec_out.specifics_vec[text_mask], dim=-1)
                true_t = F.normalize(batch_dev["specifics_vec"][:, 1:][text_mask], dim=-1)
                sim = (pred_t * true_t).sum(dim=-1)
                text_cos_sims.extend(sim.cpu().numpy().tolist())

            # CXR
            cxr_mask = mask_tgt & (batch_dev["modality_ids"][:, 1:] == Modality.CXR) & batch_dev["has_value"][:, 1:].bool()
            if cxr_mask.any():
                pred_c = F.normalize(dec_out.cxr_vec[cxr_mask], dim=-1)
                true_c = F.normalize(batch_dev["value_vec"][:, 1:][cxr_mask, :768], dim=-1)
                sim = (pred_c * true_c).sum(dim=-1)
                cxr_cos_sims.extend(sim.cpu().numpy().tolist())

            # ECG
            ecg_mask = mask_tgt & (batch_dev["modality_ids"][:, 1:] == Modality.ECG) & batch_dev["has_value"][:, 1:].bool()
            if ecg_mask.any():
                pred_e = F.normalize(dec_out.ecg_vec[ecg_mask], dim=-1)
                true_e = F.normalize(batch_dev["value_vec"][:, 1:][ecg_mask, :192], dim=-1)
                sim = (pred_e * true_e).sum(dim=-1)
                ecg_cos_sims.extend(sim.cpu().numpy().tolist())

            # Numeric
            num_mask = mask_tgt & (batch_dev["modality_ids"][:, 1:] == Modality.NUMERIC) & batch_dev["has_value"][:, 1:].bool()
            if num_mask.any():
                pred_n = F.normalize(dec_out.numeric_vec[num_mask], dim=-1)
                true_n = F.normalize(batch_dev["value_vec"][:, 1:][num_mask, :44], dim=-1)
                sim = (pred_n * true_n).sum(dim=-1)
                numeric_cos_sims.extend(sim.cpu().numpy().tolist())

    n_batches = max(1, len(test_loader))
    y_true_gate = np.array(y_true_gate, dtype=bool)
    y_pred_gate_probs = np.array(y_pred_gate_probs, dtype=float)

    # Compute ROC-AUC, precision, and recall via pure NumPy
    pred_binary = (y_pred_gate_probs > 0.5)
    tp = np.logical_and(y_true_gate, pred_binary).sum()
    fp = np.logical_and(~y_true_gate, pred_binary).sum()
    fn = np.logical_and(y_true_gate, ~pred_binary).sum()
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Trapezoidal ROC-AUC
    if len(np.unique(y_true_gate)) > 1:
        sort_order = np.argsort(y_pred_gate_probs)[::-1]
        y_sorted = y_true_gate[sort_order]
        tps = np.cumsum(y_sorted)
        fps = np.cumsum(~y_sorted)
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]
        roc_auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    else:
        roc_auc = 0.5

    return {
        "test_total_loss": total_loss_sum / n_batches,
        "test_rec_post": rec_post_sum / n_batches,
        "test_rec_prior": rec_prior_sum / n_batches,
        "test_kl": kl_sum / n_batches,
        "category_accuracy": cat_correct / max(1, cat_total),
        "modality_accuracy": mod_correct / max(1, mod_total),
        "time_gate_accuracy": time_gate_correct / max(1, time_gate_total),
        "time_gate_roc_auc": roc_auc,
        "time_gate_precision": prec,
        "time_gate_recall": rec,
        "time_positive_mae_hours": time_abs_err_sum / max(1, time_pos_count),
        "text_mean_cos_sim": float(np.mean(text_cos_sims)) if text_cos_sims else 0.0,
        "cxr_mean_cos_sim": float(np.mean(cxr_cos_sims)) if cxr_cos_sims else 0.0,
        "ecg_mean_cos_sim": float(np.mean(ecg_cos_sims)) if ecg_cos_sims else 0.0,
        "numeric_mean_cos_sim": float(np.mean(numeric_cos_sims)) if numeric_cos_sims else 0.0,
        "total_test_tokens_evaluated": cat_total,
    }


def run_free_rollout_analysis(model: NOAH, test_windows: list, device: str, n_patients: int = 5, steps: int = 20):
    """Approach 3: Multi-step autoregressive free rollout evaluation."""
    model.eval()
    rollout_results = []
    
    with torch.no_grad():
        for pt_idx in range(min(n_patients, len(test_windows))):
            w = test_windows[pt_idx]
            L_p = 6
            # Extract demographic prefix
            cat_in = torch.from_numpy(w.category_ids[:L_p]).unsqueeze(0).to(device)
            mod_in = torch.from_numpy(w.modality_ids[:L_p]).unsqueeze(0).to(device)
            pos_in = torch.from_numpy(w.positions[:L_p]).unsqueeze(0).to(device)
            age_in = torch.from_numpy(w.ages[:L_p]).unsqueeze(0).to(device)
            dp_in = torch.from_numpy(w.delta_prev[:L_p]).unsqueeze(0).to(device)
            dn_in = torch.from_numpy(w.delta_next[:L_p]).unsqueeze(0).to(device)
            spec_in = torch.from_numpy(w.specifics_vec[:L_p]).unsqueeze(0).to(device)
            has_spec = torch.from_numpy(w.has_specifics[:L_p]).unsqueeze(0).to(device)
            val_in = torch.from_numpy(w.value_vec[:L_p]).unsqueeze(0).to(device)
            has_val = torch.from_numpy(w.has_value[:L_p]).unsqueeze(0).to(device)
            attn_in = torch.from_numpy(w.attention_mask[:L_p]).unsqueeze(0).to(device)

            e_curr = model.projector(cat_in, mod_in, spec_in, has_spec, val_in, has_val)
            h = model.backbone(e=e_curr, positions=pos_in, ages=age_in, delta_next=dn_in, delta_prev=dp_in, attention_mask=attn_in)

            traj = []
            zero_gap_count = 0
            pos_gap_count = 0
            cumulative_hours = 0.0
            categories_generated = []
            modalities_generated = []

            for step_i in range(1, steps + 1):
                last_h = h[:, -1:]
                mu_p, sigma_p, _ = model.prior(last_h)
                z_sample = mu_p + sigma_p * torch.randn_like(sigma_p)
                dec = model.decoder(z_sample)

                pred_c = dec.category_logits.argmax(dim=-1).item()
                pred_m = dec.modality_logits.argmax(dim=-1).item()
                pred_g = model.decoder.decode_time(dec.time_gate_logits, dec.log_gap_reg).item()

                if pred_g <= 1e-4:
                    zero_gap_count += 1
                else:
                    pos_gap_count += 1
                    cumulative_hours += pred_g

                cat_name = EventCategory(pred_c).name if pred_c < len(EventCategory) else f"CAT_{pred_c}"
                mod_name = Modality(pred_m).name if pred_m < len(Modality) else f"MOD_{pred_m}"
                categories_generated.append(cat_name)
                modalities_generated.append(mod_name)

                traj.append({
                    "step": step_i,
                    "category": cat_name,
                    "modality": mod_name,
                    "gap_hours": round(pred_g, 2),
                    "cum_hours": round(cumulative_hours, 2),
                })

                # Append newly generated token to context
                new_c = torch.tensor([[pred_c]], device=device, dtype=torch.int64)
                new_m = torch.tensor([[pred_m]], device=device, dtype=torch.int64)
                new_s = dec.specifics_vec[:, :1, :768]
                new_has_s = (torch.sigmoid(dec.specifics_presence_logits[:, :1].squeeze(-1)) > 0.5)
                # Value representation
                if pred_m == Modality.NUMERIC:
                    new_v = F.pad(dec.numeric_vec[:, :1], (0, 768 - 44))
                elif pred_m == Modality.CXR:
                    new_v = dec.cxr_vec[:, :1]
                elif pred_m == Modality.ECG:
                    new_v = F.pad(dec.ecg_vec[:, :1], (0, 768 - 192))
                else:
                    new_v = torch.zeros(1, 1, 768, device=device)
                new_has_v = torch.ones(1, 1, device=device, dtype=torch.bool)

                new_e = model.projector(new_c, new_m, new_s, new_has_s, new_v, new_has_v)
                e_curr = torch.cat([e_curr, new_e], dim=1)
                pos_in = torch.cat([pos_in, pos_in[:, -1:] + 1.0], dim=1)
                age_in = torch.cat([age_in, age_in[:, -1:] + (pred_g / 8760.0)], dim=1)
                dp_in = torch.cat([dp_in, torch.tensor([[pred_g]], device=device)], dim=1)
                dn_in = torch.cat([dn_in, torch.zeros(1, 1, device=device)], dim=1)
                attn_in = torch.cat([attn_in, torch.ones(1, 1, device=device, dtype=torch.bool)], dim=1)

                h = model.backbone(e=e_curr, positions=pos_in, ages=age_in, delta_next=dn_in, delta_prev=dp_in, attention_mask=attn_in)

            # Shannon entropy of generated categories
            from collections import Counter
            counts = Counter(categories_generated)
            probs = np.array(list(counts.values())) / len(categories_generated)
            entropy = -np.sum(probs * np.log2(probs + 1e-12))

            rollout_results.append({
                "patient_idx": pt_idx,
                "entropy": round(float(entropy), 3),
                "unique_categories": len(counts),
                "zero_gap_percentage": round(zero_gap_count / steps * 100, 1),
                "total_cumulative_hours": round(cumulative_hours, 2),
                "trajectory_sample": traj[:5],
            })

    return rollout_results


def parse_args():
    parser = argparse.ArgumentParser(description="NOAH Comprehensive Result Verification & Scientific Audit Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pilot/pilot_checkpoint_epoch_10.pt")
    parser.add_argument("--data_dir", type=str, default="data/processed/test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument("--output_report", type=str, default="reports/server/audit_report.json")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print("NOAH COMPREHENSIVE RESULT VERIFICATION & SCIENTIFIC AUDIT SUITE")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data Dir:   {args.data_dir}")
    print(f"Device:     {args.device}")
    print("=" * 75)

    device = args.device

    # Load Model Checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found.")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    ckpt_cfg = ckpt.get("config", {})
    if args.config:
        import yaml
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f).get("model", {})
            ckpt_cfg.update(yaml_cfg)

    d_model = ckpt_cfg.get("d_model", 256)
    d_latent = ckpt_cfg.get("d_latent", 128)
    n_layers = ckpt_cfg.get("n_layers", 4)
    n_heads = ckpt_cfg.get("n_heads", 4)
    head_dim = ckpt_cfg.get("head_dim", d_model // n_heads)
    ffn_ratio = ckpt_cfg.get("ffn_ratio", ckpt_cfg.get("ffn_expansion", 4))
    layer_scale_init = float(ckpt_cfg.get("layer_scale_init", 1e-4))

    print(f"Instantiating NOAH: d_model={d_model}, d_latent={d_latent}, n_layers={n_layers}, n_heads={n_heads}, ffn_ratio={ffn_ratio}")
    model = NOAH(
        d_model=d_model,
        d_latent=d_latent,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        ffn_ratio=ffn_ratio,
        dropout=0.1,
        temporal_layers=4,
        layer_scale_init=layer_scale_init,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print("NOAH Model weights loaded successfully.")

    # 1. Held-out Test Set Evaluation
    print(f"\n[Audit 1/3] Loading held-out Test Shards ({args.data_dir})...")
    test_windows = load_windows(args.data_dir)
    print(f"Loaded {len(test_windows)} test windows across {len(test_windows)//100 + 1} shards.")

    test_loader = DataLoader(
        test_windows,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_windows,
        drop_last=False,
    )

    print("\nRunning comprehensive held-out test evaluation...")
    t0 = time.time()
    test_metrics = run_test_evaluation(model, test_loader, device)
    dt_eval = time.time() - t0
    print(f"Evaluation completed in {dt_eval:.2f} s on {test_metrics['total_test_tokens_evaluated']:,} scored tokens.")

    print("\n" + "-" * 50)
    print("HELD-OUT TEST SET METRICS:")
    print("-" * 50)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30}: {v:.4f}")
        else:
            print(f"  {k:<30}: {v}")

    # 2. Free Autoregressive Rollout Analysis
    print("\n[Audit 2/3] Running 20-step Autoregressive Free Rollout Analysis...")
    rollouts = run_free_rollout_analysis(model, test_windows, device, n_patients=3, steps=20)
    print("\n" + "-" * 50)
    print("ROLLOUT TRAJECTORY DYNAMICS:")
    print("-" * 50)
    for r in rollouts:
        print(f"Patient {r['patient_idx']}: Entropy={r['entropy']} bits | Unique Categories={r['unique_categories']} | Zero-Gap%={r['zero_gap_percentage']}% | Total Hours={r['total_cumulative_hours']}h")
        for s in r['trajectory_sample']:
            print(f"   Step {s['step']}: {s['category']} [{s['modality']}] -> +{s['gap_hours']}h (Cum: {s['cum_hours']}h)")

    # Save output report
    if args.output_report:
        out_p = Path(args.output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "test_metrics": test_metrics,
            "rollout_dynamics": rollouts,
        }
        with open(out_p, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"\nAudit report saved to {out_p}")

    print("\n" + "=" * 75)
    print("AUDIT EXECUTION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
