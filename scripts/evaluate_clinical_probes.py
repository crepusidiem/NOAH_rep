"""Legacy clinical CLI is held pending approved provenance-complete evaluation views.

Archived implementations, checkpoints and metrics are preserved separately. The new
landmark/outcome APIs require explicit policies; no clinical methodology is inferred.
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
from src.evaluation.contracts import ContractError, validate_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate downstream clinical probes on NOAH representations.")
    parser.add_argument("--manifest", required=True, help="Experiment manifest; legacy evaluation is blocked")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pilot/pilot_checkpoint_epoch_10.pt")
    parser.add_argument("--config", type=str, default="configs/pilot_config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/processed/test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_report", type=str, default="reports/server/clinical_probes_pilot_report.json")
    return parser.parse_args()


def load_model(checkpoint_path: Path, config: dict, device: str) -> NOAH:
    print(f"Loading checkpoint weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_cfg = ckpt.get("config", {})
    model_cfg = config.get("model", {}) if config else {}
    if ckpt_cfg.get("architecture_revision") != "2026-10-19":
        raise ContractError("Unversioned/legacy checkpoint: use its preserved implementation, not the current architecture")
    aliases = {"d_head": "head_dim", "ffn_expansion": "ffn_ratio", "attention_dropout": "attn_dropout"}
    for key, value in model_cfg.items():
        saved_key = aliases.get(key, key)
        if saved_key in ckpt_cfg and value != ckpt_cfg[saved_key]:
            raise ContractError(f"Current YAML attempts to override checkpoint architecture: {key}")


    # Architecture metadata is checkpoint-authoritative; conflicting YAML is rejected.
    d_model = model_cfg.get("d_model", ckpt_cfg.get("d_model", 256))
    d_latent = model_cfg.get("d_latent", ckpt_cfg.get("d_latent", 128))
    n_layers = model_cfg.get("n_layers", ckpt_cfg.get("n_layers", 4))
    n_heads = model_cfg.get("n_heads", ckpt_cfg.get("n_heads", 4))
    head_dim = model_cfg.get("head_dim", model_cfg.get("d_head", ckpt_cfg.get("head_dim", d_model // n_heads)))
    ffn_ratio = model_cfg.get("ffn_expansion", ckpt_cfg.get("ffn_ratio", 4))
    layer_scale_init = float(model_cfg.get("layer_scale_init", ckpt_cfg.get("layer_scale_init", 1e-4)))

    print(f"Instantiating NOAH: d_model={d_model}, d_latent={d_latent}, n_layers={n_layers}, n_heads={n_heads}, ffn_ratio={ffn_ratio}")
    model = NOAH(
        d_model=d_model,
        d_latent=d_latent,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        ffn_ratio=ffn_ratio,
        dropout=model_cfg.get("dropout", 0.1),
        attn_dropout=model_cfg.get("attention_dropout", 0.1),
        num_categories=config.get("feature_dimensions", {}).get("category_vocab_size", 31) if config else 31,
        num_modalities=6,
        text_dim=768,
        cxr_dim=768,
        ecg_dim=192,
        echo_dim=512,
        numeric_dim=44,
        temporal_layers=4,
        layer_scale_init=layer_scale_init,
        alpha_post=0.3,
        gamma_prior=0.1,
    ).to(device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    for key in ("category_modality_support", "category_class_weights", "modality_class_weights", "specifics_presence_class_weights"):
        if ckpt.get(key) is not None:
            setattr(model, key, ckpt[key].to(device))
    model.eval()
    return model


def extract_patient_representations_and_news2(
    model: NOAH,
    data_dir: Path,
    device: str,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    """Reject the archived pooled extractor; it is not a landmark representation."""
    raise ContractError(
        "Full-record pooling is disabled. Use evaluation.landmarks.extract_landmark_state "
        "with an explicit encounter, cutoff, mode and source availability evidence. "
        "Historical implementation and artifacts are preserved in the reproducibility snapshot."
    )


def extract_patient_labels(
    manifest_path: str,
    mimic_hosp_dir: str,
    mimic_ed_dir: str,
    target_sids: List[int],
) -> Dict[int, Dict[str, Union[int, float]]]:
    """Reject archived subject-maximum labels and fabricated missing NEWS2 labels."""
    raise ContractError(
        "Subject-wide outcome maxima are disabled. Supply selected encounter IDs and "
        "approved endpoint parameters to evaluation.outcomes. NEWS2 evaluation remains "
        "blocked until its complete local-window label contract is approved and verified."
    )


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    validate_manifest(manifest)
    raise ContractError(
        "Legacy shard-based clinical evaluation is blocked. The new landmark API requires "
        "provenance-complete views; no scientific defaults or NEWS2 labels have been selected."
    )


if __name__ == "__main__":
    main()
