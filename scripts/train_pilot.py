"""
Pilot Training Pipeline for NOAH (N-07) with Secure W&B Logging.

Executes:
1. Load curated dataset shards (pilot_train_windows.pt, pilot_val_windows.pt).
2. Mini-batch collation into PyTorch tensors with padding & masks.
3. AdamW optimizer with cosine LR schedule and linear warmup (OPT-01, OPT-02).
4. Linear beta annealing for KL divergence (LOSS-03).
5. Dual-path reconstruction loss (0.3 L_post + beta L_KL + 0.1 L_prior).
6. Periodic validation and prior-only autoregressive rollout inspection.
7. Model checkpointing.
8. Comprehensive W&B logging (online or offline) covering all NOAH components:
   - Composite & component losses (posterior/prior rec, KL, categories, modalities, specifics, values, time gate/reg)
   - Variational latent dynamics (beta, prior sigma bounds, posterior sigma, surprise)
   - Optimizer dynamics (learning rate, gradient norm, throughput)
   - Validation classification accuracies and time MAE
   - Generative rollout table (inspecting event trajectories without logging any PHI)
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.data.schema import ModelWindow, EventCategory, Modality
from src.models.noah import NOAH, NOAHLossBreakdown


class WindowDataset(Dataset):
    def __init__(self, windows: List[ModelWindow]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> ModelWindow:
        return self.windows[idx]


def load_windows(path_str: str) -> List[ModelWindow]:
    """
    Loads ModelWindow items from either:
    1. A single non-empty .pt file
    2. A directory containing shard_*.pt files
    3. An inferred split directory (e.g. data/processed/train from data/processed/pilot_train_windows.pt)
    """
    p = Path(path_str)

    # 1. Direct directory with shards
    if p.is_dir():
        shards = sorted(p.glob("shard_*.pt"))
        if shards:
            windows = []
            for sf in shards:
                windows.extend(torch.load(sf, weights_only=False))
            return windows

    # 2. Direct valid non-empty file
    if p.is_file() and p.stat().st_size > 0:
        return torch.load(p, weights_only=False)

    # 3. Inferred directory from filename (e.g. pilot_train_windows.pt -> train/)
    candidates = []
    fname = p.name.lower()
    if "train" in fname:
        candidates = [p.parent / "train", p.parent / "pilot_train"]
    elif "val" in fname:
        candidates = [p.parent / "val", p.parent / "pilot_val"]
    elif "test" in fname:
        candidates = [p.parent / "test", p.parent / "pilot_test"]

    for cand in candidates:
        if cand.is_dir():
            shards = sorted(cand.glob("shard_*.pt"))
            if shards:
                print(f"Loading {len(shards)} shards from {cand}...")
                windows = []
                for sf in shards:
                    windows.extend(torch.load(sf, weights_only=False))
                return windows

    raise FileNotFoundError(f"Could not load windows from {p} (checked shards and fallback dirs: {candidates})")


def collate_windows(batch: List[ModelWindow]) -> Dict[str, torch.Tensor]:
    """Collates a list of ModelWindow objects into batched tensors."""
    return {
        "category_ids": torch.from_numpy(np.stack([w.category_ids for w in batch])).long(),
        "modality_ids": torch.from_numpy(np.stack([w.modality_ids for w in batch])).long(),
        "positions": torch.from_numpy(np.stack([w.positions for w in batch])).float(),
        "ages": torch.from_numpy(np.stack([w.ages for w in batch])).float(),
        "delta_prev": torch.from_numpy(np.stack([w.delta_prev for w in batch])).float(),
        "delta_next": torch.from_numpy(np.stack([w.delta_next for w in batch])).float(),
        "has_next_gap": torch.from_numpy(np.stack([w.has_next_gap for w in batch])).bool(),
        "specifics_vec": torch.from_numpy(np.stack([w.specifics_vec for w in batch])).float(),
        "has_specifics": torch.from_numpy(np.stack([w.has_specifics for w in batch])).bool(),
        "value_vec": torch.from_numpy(np.stack([w.value_vec for w in batch])).float(),
        "has_value": torch.from_numpy(np.stack([w.has_value for w in batch])).bool(),
        "attention_mask": torch.from_numpy(np.stack([w.attention_mask for w in batch])).bool(),
        "loss_mask": torch.from_numpy(np.stack([w.loss_mask for w in batch])).bool(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="NOAH Pilot Training")
    parser.add_argument("--train_data", type=str, default="data/processed/pilot_train_windows.pt")
    parser.add_argument("--val_data", type=str, default="data/processed/pilot_val_windows.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints/pilot")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--kl_warmup_epochs", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--d_model", type=int, default=None, help="Model hidden dimension (default 256 for pilot, 768 for paper)")
    parser.add_argument("--d_latent", type=int, default=None, help="Latent space dimension (default 128 for pilot, 384 for paper)")
    parser.add_argument("--n_layers", type=int, default=None, help="Number of transformer layers (default 4 for pilot, 12 for paper)")
    parser.add_argument("--n_heads", type=int, default=None, help="Number of attention heads (default 4 for pilot, 12 for paper)")
    parser.add_argument("--ffn_ratio", type=int, default=None, help="FFN expansion ratio (default 4 for pilot, 8 for paper)")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Weights & Biases Options
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="noah-pilot", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team name")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run display name")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"],
                        help="W&B operation mode. Use 'offline' if compute node has no internet access.")
    parser.add_argument("--log_interval", type=int, default=5, help="Step interval for W&B metric logging")
    parser.add_argument("--use_class_weights", action="store_true", help="Apply inverse frequency class weights to category cross-entropy loss")
    
    return parser.parse_args()


def get_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    """Cosine learning rate schedule with linear warmup (OPT-02)."""
    if step < warmup_steps:
        return base_lr * (step + 1) / (warmup_steps + 1)
    decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def get_beta(epoch: int, kl_warmup_epochs: int) -> float:
    """Linear beta warmup for KL divergence (LOSS-03)."""
    if kl_warmup_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / kl_warmup_epochs)


def evaluate(
    model: NOAH,
    val_loader: DataLoader,
    device: str,
    beta: float = 1.0,
) -> Dict[str, float]:
    """Runs full validation evaluation across all multimodal heads."""
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

    with torch.no_grad():
        for batch in val_loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
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
                beta=beta,
            )

            total_loss_sum += loss.item()
            rec_post_sum += breakdown.loss_rec_post.item()
            rec_prior_sum += breakdown.loss_rec_prior.item()
            kl_sum += breakdown.loss_kl.item()

            # Target mask
            mask_tgt = batch_dev["loss_mask"][:, 1:].bool()
            if mask_tgt.any():
                # Reconstruct with posterior latent
                z_post = aux["mu_q"]
                dec_out = model.decoder(z_post)

                # 1. Category accuracy
                pred_cats = dec_out.category_logits[mask_tgt].argmax(dim=-1)
                true_cats = batch_dev["category_ids"][:, 1:][mask_tgt]
                cat_correct += (pred_cats == true_cats).sum().item()
                cat_total += mask_tgt.sum().item()

                # 2. Modality accuracy
                pred_mods = dec_out.modality_logits[mask_tgt].argmax(dim=-1)
                true_mods = batch_dev["modality_ids"][:, 1:][mask_tgt]
                mod_correct += (pred_mods == true_mods).sum().item()
                mod_total += mask_tgt.sum().item()

                # 3. Time Gate & Continuous MAE
                has_gap_mask = mask_tgt & batch_dev["has_next_gap"][:, 1:].bool()
                if has_gap_mask.any():
                    t_gate_logits = dec_out.time_gate_logits.squeeze(-1)[has_gap_mask]
                    is_pos_true = batch_dev["delta_next"][:, 1:][has_gap_mask] > 0.0
                    pred_pos = torch.sigmoid(t_gate_logits) > 0.5
                    time_gate_correct += (pred_pos == is_pos_true).sum().item()
                    time_gate_total += has_gap_mask.sum().item()

                    # Positive gap error
                    pos_mask = has_gap_mask & (batch_dev["delta_next"][:, 1:] > 0.0)
                    if pos_mask.any():
                        pred_gaps = model.decoder.decode_time(
                            dec_out.time_gate_logits[pos_mask],
                            dec_out.log_gap_reg[pos_mask],
                        )
                        true_gaps = batch_dev["delta_next"][:, 1:][pos_mask]
                        time_abs_err_sum += (pred_gaps - true_gaps).abs().sum().item()
                        time_pos_count += pos_mask.sum().item()

    n_batches = max(1, len(val_loader))
    metrics = {
        "val/total_loss": total_loss_sum / n_batches,
        "val/loss_rec_post": rec_post_sum / n_batches,
        "val/loss_rec_prior": rec_prior_sum / n_batches,
        "val/loss_kl": kl_sum / n_batches,
        "val/category_acc": (cat_correct / max(1, cat_total)),
        "val/modality_acc": (mod_correct / max(1, mod_total)),
        "val/time_gate_acc": (time_gate_correct / max(1, time_gate_total)),
        "val/time_mae_hours": (time_abs_err_sum / max(1, time_pos_count)),
    }
    return metrics


def generate_rollout_trajectory(
    model: NOAH,
    first_batch: Dict[str, torch.Tensor],
    device: str,
    n_rollout_steps: int = 5,
) -> List[Dict[str, str]]:
    """
    Simulates prior-only autoregressive rollout (ROLL-01..03, LAT-05).
    Generates anonymized clinical event predictions from demographic prefix.
    """
    model.eval()
    trajectory = []
    with torch.no_grad():
        L_prompt = 6
        e_curr = model.projector(
            first_batch["category_ids"][:1, :L_prompt],
            first_batch["modality_ids"][:1, :L_prompt],
            first_batch["specifics_vec"][:1, :L_prompt],
            first_batch["has_specifics"][:1, :L_prompt],
            first_batch["value_vec"][:1, :L_prompt],
            first_batch["has_value"][:1, :L_prompt],
        )
        pos = first_batch["positions"][:1, :L_prompt]
        ages = first_batch["ages"][:1, :L_prompt]
        d_next = first_batch["delta_next"][:1, :L_prompt]
        d_prev = first_batch["delta_prev"][:1, :L_prompt]
        attn = first_batch["attention_mask"][:1, :L_prompt]

        h = model.backbone(e=e_curr, positions=pos, ages=ages, delta_next=d_next, delta_prev=d_prev, attention_mask=attn)

        # Rollout loop
        for step_i in range(1, n_rollout_steps + 1):
            last_h = h[:, -1:]
            mu_p, sigma_p, _ = model.prior(last_h)
            z_sample = mu_p + sigma_p * torch.randn_like(sigma_p)
            dec = model.decoder(z_sample)

            pred_cat = dec.category_logits.argmax(dim=-1).item()
            pred_mod = dec.modality_logits.argmax(dim=-1).item()
            pred_gap = model.decoder.decode_time(dec.time_gate_logits, dec.log_gap_reg).item()

            cat_name = EventCategory(pred_cat).name if pred_cat < len(EventCategory) else f"CAT_{pred_cat}"
            mod_name = Modality(pred_mod).name if pred_mod < len(Modality) else f"MOD_{pred_mod}"

            trajectory.append({
                "step": step_i,
                "category": cat_name,
                "modality": mod_name,
                "predicted_gap_hours": f"{pred_gap:.2f}",
            })

            # Append synthetic placeholder token to continue context
            new_cat = torch.tensor([[pred_cat]], device=device, dtype=torch.int64)
            new_mod = torch.tensor([[pred_mod]], device=device, dtype=torch.int64)
            new_spec = torch.zeros(1, 1, 768, device=device)
            new_has_spec = torch.zeros(1, 1, device=device, dtype=torch.bool)
            new_val = torch.zeros(1, 1, 768, device=device)
            new_has_val = torch.zeros(1, 1, device=device, dtype=torch.bool)

            new_e = model.projector(new_cat, new_mod, new_spec, new_has_spec, new_val, new_has_val)
            e_curr = torch.cat([e_curr, new_e], dim=1)
            pos = torch.cat([pos, pos[:, -1:] + 1.0], dim=1)
            ages = torch.cat([ages, ages[:, -1:] + (pred_gap / 8760.0)], dim=1)
            d_prev = torch.cat([d_prev, torch.tensor([[pred_gap]], device=device)], dim=1)
            d_next = torch.cat([d_next, torch.zeros(1, 1, device=device)], dim=1)
            attn = torch.cat([attn, torch.ones(1, 1, device=device, dtype=torch.bool)], dim=1)

            h = model.backbone(e=e_curr, positions=pos, ages=ages, delta_next=d_next, delta_prev=d_prev, attention_mask=attn)

    return trajectory


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NOAH PILOT TRAINING PIPELINE (N-07)")
    print(f"Device:       {args.device}")
    print(f"Train Data:   {args.train_data}")
    print(f"Val Data:     {args.val_data}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"Epochs:       {args.epochs}")
    print(f"Base LR:      {args.lr}")
    print(f"W&B Enabled:  {args.use_wandb} (Mode: {args.wandb_mode})")
    print("=" * 70)

    # 1. Initialize Configuration
    cfg_dict = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)

    model_cfg = cfg_dict.get("model", {})
    train_cfg = cfg_dict.get("training", {})

    d_model = args.d_model if args.d_model is not None else model_cfg.get("d_model", 256)
    d_latent = args.d_latent if args.d_latent is not None else model_cfg.get("d_latent", 128)
    n_layers = args.n_layers if args.n_layers is not None else model_cfg.get("n_layers", 4)
    n_heads = args.n_heads if args.n_heads is not None else model_cfg.get("n_heads", 4)
    d_head = model_cfg.get("d_head", d_model // n_heads)
    ffn_ratio = args.ffn_ratio if args.ffn_ratio is not None else model_cfg.get("ffn_expansion", 4)
    dropout = model_cfg.get("dropout", 0.1)
    attn_dropout = model_cfg.get("attention_dropout", 0.1)
    layer_scale_init = float(model_cfg.get("layer_scale_init", 1e-4))
    grad_accum_steps = max(1, args.grad_accum_steps if args.grad_accum_steps > 1 else train_cfg.get("gradient_accumulation_steps", 1))

    print(f"Model Architecture: d_model={d_model}, d_latent={d_latent}, n_layers={n_layers}, n_heads={n_heads}, ffn_ratio={ffn_ratio}")
    print(f"Grad Accumulation:  {grad_accum_steps} (Effective Batch Size: {args.batch_size * grad_accum_steps})")

    # 1. Initialize W&B if enabled
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            # Secure configuration: no raw API key in code
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name or f"noah_m{d_model}_l{d_latent}_e{args.epochs}_lr{args.lr}",
                mode=args.wandb_mode,
                config={
                    "d_model": d_model,
                    "d_latent": d_latent,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "head_dim": d_head,
                    "ffn_ratio": ffn_ratio,
                    "dropout": dropout,
                    "attn_dropout": attn_dropout,
                    "batch_size": args.batch_size,
                    "grad_accum_steps": grad_accum_steps,
                    "epochs": args.epochs,
                    "base_lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "warmup_epochs": args.warmup_epochs,
                    "kl_warmup_epochs": args.kl_warmup_epochs,
                    "grad_clip": args.grad_clip,
                    "alpha_post": 0.3,
                    "gamma_prior": 0.1,
                },
            )
            wandb_run = wandb.run
            print(f"-> W&B initialized successfully (Run ID: {wandb_run.id}, Mode: {args.wandb_mode})")
        except Exception as e:
            print(f"Warning: W&B initialization failed ({e}). Proceeding without W&B.")
            args.use_wandb = False

    # 2. Load Data
    try:
        train_windows = load_windows(args.train_data)
        print(f"Loaded {len(train_windows):,} training windows.")
    except Exception as e:
        print(f"Error loading training data: {e}")
        sys.exit(1)

    val_windows = []
    try:
        val_windows = load_windows(args.val_data)
        print(f"Loaded {len(val_windows):,} validation windows.")
    except Exception as e:
        print(f"Notice: validation data not found or empty ({e}). Skipping validation evaluation.")

    train_dataset = WindowDataset(train_windows)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_windows,
        drop_last=False,
    )

    val_loader = None
    if len(val_windows) > 0:
        val_dataset = WindowDataset(val_windows)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_windows,
            drop_last=False,
        )

    # 3. Instantiate Model
    category_class_weights = None
    if args.use_class_weights:
        print("Computing smoothed normalized inverse frequency class weights for category loss...")
        counts = torch.zeros(31, dtype=torch.float64)
        for w in train_windows:
            valid_cats = w.category_ids[w.loss_mask]
            for c in valid_cats:
                if 0 <= c < 31:
                    counts[c] += 1
        
        # Smoothed inverse square-root frequency: w_c = 1 / sqrt(N_c + 1)
        inv_freq = 1.0 / torch.sqrt(counts + 1.0)
        # Normalize so that weights sum to C (mean weight = 1.0)
        category_class_weights = (31.0 * (inv_freq / inv_freq.sum())).float()
        print(f"Computed weights for 31 categories. Min: {category_class_weights.min():.4f}, Max: {category_class_weights.max():.4f}")

    model = NOAH(
        d_model=d_model,
        d_latent=d_latent,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=d_head,
        ffn_ratio=ffn_ratio,
        dropout=dropout,
        attn_dropout=attn_dropout,
        num_categories=31,
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
        category_class_weights=category_class_weights,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"NOAH Model instantiated with {total_params:,} trainable parameters.")
    if args.use_wandb:
        import wandb
        wandb.summary["trainable_parameters"] = total_params

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    global_step = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        beta = get_beta(epoch, args.kl_warmup_epochs)
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_rec_post = 0.0
        epoch_rec_prior = 0.0
        epoch_start_t = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (beta={beta:.2f})")
        optimizer.zero_grad()

        for step_idx, batch in enumerate(pbar):
            step_start_t = time.time()
            lr = get_lr(global_step, total_steps, warmup_steps, args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            batch_dev = {k: v.to(args.device) for k, v in batch.items()}

            total_loss, breakdown, aux = model(
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
                beta=beta,
            )

            # Scale loss for gradient accumulation
            scaled_loss = total_loss / grad_accum_steps
            scaled_loss.backward()

            step_did_update = False
            grad_norm = 0.0

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** 0.5

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                step_did_update = True

            step_time = time.time() - step_start_t
            scored_tokens_in_step = int(batch_dev["loss_mask"].sum().item())
            tokens_per_sec = scored_tokens_in_step / max(1e-5, step_time)

            epoch_loss += total_loss.item()
            epoch_kl += breakdown.loss_kl.item()
            epoch_rec_post += breakdown.loss_rec_post.item()
            epoch_rec_prior += breakdown.loss_rec_prior.item()

            # Periodic W&B Step Logging
            if args.use_wandb and (global_step % args.log_interval == 0):
                import wandb
                sigma_p = aux["sigma_p"]
                sigma_q = aux["sigma_q"]
                wandb.log({
                    "train/step": global_step,
                    "train/epoch": epoch + 1,
                    "train/total_loss": total_loss.item(),
                    "train/loss_rec_post": breakdown.loss_rec_post.item(),
                    "train/loss_rec_prior": breakdown.loss_rec_prior.item(),
                    "train/loss_kl": breakdown.loss_kl.item(),
                    "train/loss_category": breakdown.loss_category.item(),
                    "train/loss_modality": breakdown.loss_modality.item(),
                    "train/loss_specifics_presence": breakdown.loss_specifics_presence.item(),
                    "train/loss_specifics_vec": breakdown.loss_specifics_vec.item(),
                    "train/loss_value": breakdown.loss_value.item(),
                    "train/loss_time_gate": breakdown.loss_time_gate.item(),
                    "train/loss_time_reg": breakdown.loss_time_reg.item(),
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "train/beta": beta,
                    "train/prior_sigma_mean": sigma_p.mean().item(),
                    "train/prior_sigma_min": sigma_p.min().item(),
                    "train/prior_sigma_max": sigma_p.max().item(),
                    "train/posterior_sigma_mean": sigma_q.mean().item(),
                    "train/tokens_per_sec": tokens_per_sec,
                }, step=global_step)

            global_step += 1
            pbar.set_postfix({
                "loss": f"{total_loss.item():.2f}",
                "kl": f"{breakdown.loss_kl.item():.2f}",
                "lr": f"{lr:.1e}",
            })

        avg_loss = epoch_loss / len(train_loader)
        avg_kl = epoch_kl / len(train_loader)
        avg_rec_post = epoch_rec_post / len(train_loader)
        avg_rec_prior = epoch_rec_prior / len(train_loader)
        epoch_duration = time.time() - epoch_start_t

        print(f"\n[Epoch {epoch+1} Summary] Loss: {avg_loss:.4f} | RecPost: {avg_rec_post:.4f} | KL: {avg_kl:.4f} | RecPrior: {avg_rec_prior:.4f} ({epoch_duration:.1f}s)")

        # Epoch-level logging
        if args.use_wandb:
            import wandb
            wandb.log({
                "epoch/train_loss": avg_loss,
                "epoch/rec_post": avg_rec_post,
                "epoch/rec_prior": avg_rec_prior,
                "epoch/kl": avg_kl,
                "epoch/duration_sec": epoch_duration,
                "epoch/epoch": epoch + 1,
            }, step=global_step)

        # Validation Run (if val_loader available)
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, args.device, beta=beta)
            print(f"[Val Evaluation] Loss: {val_metrics['val/total_loss']:.4f} | Cat Acc: {val_metrics['val/category_acc']*100:.1f}% | Mod Acc: {val_metrics['val/modality_acc']*100:.1f}% | Time Gate Acc: {val_metrics['val/time_gate_acc']*100:.1f}%")
            if args.use_wandb:
                import wandb
                wandb.log(val_metrics, step=global_step)

        # Free Rollout Inspection (every epoch)
        first_batch = next(iter(train_loader))
        first_batch = {k: v.to(args.device) for k, v in first_batch.items()}
        rollout_traj = generate_rollout_trajectory(model, first_batch, args.device, n_rollout_steps=5)

        print("  -> Free Rollout Sample:")
        for r in rollout_traj[:3]:
            print(f"     Step {r['step']}: {r['category']} ({r['modality']}) -> +{r['predicted_gap_hours']}h")

        if args.use_wandb:
            import wandb
            rollout_table = wandb.Table(
                columns=["step", "predicted_category", "predicted_modality", "predicted_gap_hours"],
                data=[[r["step"], r["category"], r["modality"], float(r["predicted_gap_hours"])] for r in rollout_traj]
            )
            wandb.log({f"rollout/epoch_{epoch+1}": rollout_table}, step=global_step)

        # Memory tracking
        if torch.cuda.is_available() and "cuda" in str(args.device):
            peak_vram_gb = torch.cuda.max_memory_allocated(device=args.device) / (1024 ** 3)
            print(f"[Memory] Peak VRAM: {peak_vram_gb:.2f} GB")
            if args.use_wandb:
                import wandb
                wandb.log({"system/peak_vram_gb": peak_vram_gb}, step=global_step)

        # Checkpoint every epoch
        ckpt_path = output_dir / f"pilot_checkpoint_epoch_{epoch+1}.pt"
        saved_config = {
            "d_model": d_model,
            "d_latent": d_latent,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "head_dim": d_head,
            "ffn_ratio": ffn_ratio,
            "dropout": dropout,
            "attn_dropout": attn_dropout,
            "layer_scale_init": layer_scale_init,
            "use_class_weights": args.use_class_weights,
        }
        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": saved_config,
        }, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    if args.use_wandb:
        import wandb
        wandb.finish()
        print("-> W&B run finished and finalized.")

    print("\n" + "=" * 70)
    print(f"Training completed successfully in {time.time() - t_start:.2f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
