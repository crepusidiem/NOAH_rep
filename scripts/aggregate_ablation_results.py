"""
Aggregate and summarize results from the 9-run controlled class-weighting strength (alpha-sweep) ablation grid.
Computes Mean +/- Std across seeds {42, 43, 44} for each alpha in {0.00, 0.25, 0.50}.
"""

import json
from pathlib import Path
import numpy as np


def aggregate_results():
    from src.evaluation.contracts import ContractError
    raise ContractError("Evaluation/aggregation blocked by remediation: require approved endpoint, landmark and experiment manifests; legacy source is snapshotted.")
    runs = {
        "alpha_0.00": ["alpha00_seed42", "alpha00_seed43", "alpha00_seed44"],
        "alpha_0.25": ["alpha25_seed42", "alpha25_seed43", "alpha25_seed44"],
        "alpha_0.50": ["alpha50_seed42", "alpha50_seed43", "alpha50_seed44"],
    }

    server_dir = Path("reports/server")
    summary = {}

    print("=" * 80)
    print("NOAH CLASS-WEIGHTING STRENGTH (ALPHA-SWEEP) AGGREGATION")
    print("=" * 80)

    for alpha_key, run_list in runs.items():
        summary[alpha_key] = {
            "individual_runs": {},
            "metrics": {},
        }
        
        # Metrics to collect
        mort_auroc, mort_auprc, mort_balacc = [], [], []
        los_auroc, los_auprc, los_balacc = [], [], []
        readm_auroc, readm_auprc, readm_balacc = [], [], []
        news2_mae, news2_pearson = [], []
        cat_acc_post, cat_acc_prior = [], []
        mod_acc = []
        num_cos_sim = []
        time_gate_acc = []

        for rname in run_list:
            audit_p = server_dir / f"audit_{rname}.json"
            probe_p = server_dir / f"clinical_probes_{rname}.json"

            if not audit_p.exists() or not probe_p.exists():
                print(f"Warning: Results for {rname} not fully available yet ({audit_p.name}: {audit_p.exists()}, {probe_p.name}: {probe_p.exists()})")
                continue

            with open(audit_p) as f:
                audit_data = json.load(f)["test_metrics"]
            with open(probe_p) as f:
                probe_data = json.load(f)

            summary[alpha_key]["individual_runs"][rname] = {
                "audit": audit_data,
                "probes": probe_data,
            }

            # Probing metrics
            if "mortality" in probe_data:
                mort_auroc.append(probe_data["mortality"]["auroc"])
                mort_auprc.append(probe_data["mortality"]["auprc"])
                mort_balacc.append(probe_data["mortality"]["balanced_acc"])
            if "prolonged_los_7d" in probe_data:
                los_auroc.append(probe_data["prolonged_los_7d"]["auroc"])
                los_auprc.append(probe_data["prolonged_los_7d"]["auprc"])
                los_balacc.append(probe_data["prolonged_los_7d"]["balanced_acc"])
            if "readmission_30d" in probe_data:
                readm_auroc.append(probe_data["readmission_30d"]["auroc"])
                readm_auprc.append(probe_data["readmission_30d"]["auprc"])
                readm_balacc.append(probe_data["readmission_30d"]["balanced_acc"])
            if "news2_regression" in probe_data:
                news2_mae.append(probe_data["news2_regression"]["mae"])
                news2_pearson.append(probe_data["news2_regression"]["pearson_r"])

            # Audit metrics
            cat_acc_post.append(audit_data.get("category_accuracy", 0.0))
            cat_acc_prior.append(audit_data.get("prior_category_accuracy", 0.0))
            mod_acc.append(audit_data.get("modality_accuracy", 0.0))
            num_cos_sim.append(audit_data.get("numeric_cosine_similarity", 0.0))
            time_gate_acc.append(audit_data.get("time_gate_accuracy", 0.0))

        def m_s(arr):
            if not arr:
                return {"mean": None, "std": None, "n": 0}
            return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, "n": len(arr)}

        summary[alpha_key]["metrics"] = {
            "mortality_auroc": m_s(mort_auroc),
            "mortality_auprc": m_s(mort_auprc),
            "mortality_balacc": m_s(mort_balacc),
            "los_auroc": m_s(los_auroc),
            "los_auprc": m_s(los_auprc),
            "los_balacc": m_s(los_balacc),
            "readmission_auroc": m_s(readm_auroc),
            "readmission_balacc": m_s(readm_balacc),
            "news2_mae": m_s(news2_mae),
            "news2_pearson": m_s(news2_pearson),
            "category_accuracy_post": m_s(cat_acc_post),
            "category_accuracy_prior": m_s(cat_acc_prior),
            "modality_accuracy": m_s(mod_acc),
            "numeric_cosine_similarity": m_s(num_cos_sim),
            "time_gate_accuracy": m_s(time_gate_acc),
        }

    # Print summary table
    print("\n" + "=" * 95)
    print(f"{'Condition':15s} | {'Mort AUROC':16s} | {'LOS >7d AUROC':16s} | {'Readm BalAcc':16s} | {'Cat Acc (Post)':16s}")
    print("-" * 95)
    for akey, data in summary.items():
        m = data["metrics"]
        def fmt(k):
            val = m[k]
            if val["mean"] is None:
                return "N/A"
            return f"{val['mean']:.4f} +/- {val['std']:.4f}"
        print(f"{akey:15s} | {fmt('mortality_auroc'):16s} | {fmt('los_auroc'):16s} | {fmt('readmission_balacc'):16s} | {fmt('category_accuracy_post'):16s}")
    print("=" * 95)

    out_json = server_dir / "alpha_ablation_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to {out_json}")


if __name__ == "__main__":
    aggregate_results()
