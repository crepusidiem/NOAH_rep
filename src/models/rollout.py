"""
Autoregressive Free Rollout and Anti-Stall Sampling Engine (N-07/N-08).

Implements:
1. Temperature-scaled Nucleus (Top-p) Sampling (ROLL-01..03).
2. Anti-Stall Zero-Gap Budgeting (U-R06): Prevents degenerate mode collapse
   on simultaneous lab events by forcing temporal encounter advancement.
3. Continuous Payload Re-encoding (U-R02): Re-injects generated continuous
   features (specifics, numeric values, media vectors) into input representation space.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from src.data.schema import EventCategory, Modality
from src.models.noah import NOAH


def sample_top_p(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> int:
    """
    Samples a discrete token index using temperature scaling and nucleus (top-p) filtering.
    """
    if temperature <= 0.0 or top_p <= 0.0:
        return int(logits.argmax(dim=-1).item())

    # Flatten to 1D
    logits_1d = logits.view(-1) / max(1e-4, temperature)
    probs = F.softmax(logits_1d, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask out tokens beyond cumulative probability top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep at least the highest probability token
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False

    sorted_probs[sorted_indices_to_remove] = 0.0
    prob_sum = sorted_probs.sum()
    if prob_sum <= 0.0:
        return int(sorted_indices[0].item())

    sorted_probs = sorted_probs / prob_sum
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    sampled_token_id = sorted_indices[sampled_sorted_idx]
    return int(sampled_token_id.item())


def generate_autoregressive_rollout(
    model: NOAH,
    prefix_data: Dict[str, torch.Tensor],
    device: str,
    n_rollout_steps: int = 20,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_consecutive_zero_gaps: int = 3,
    min_forced_gap_hours: float = 0.5,
) -> List[Dict[str, any]]:
    """
    Executes free prior-only autoregressive rollout with anti-stall zero-gap budgeting.
    """
    model.eval()
    trajectory = []

    with torch.no_grad():
        L_prompt = 6  # Demographic prefix length

        e_curr = model.projector(
            prefix_data["category_ids"][:1, :L_prompt].to(device),
            prefix_data["modality_ids"][:1, :L_prompt].to(device),
            prefix_data["specifics_vec"][:1, :L_prompt].to(device),
            prefix_data["has_specifics"][:1, :L_prompt].to(device),
            prefix_data["value_vec"][:1, :L_prompt].to(device),
            prefix_data["has_value"][:1, :L_prompt].to(device),
        )
        pos = prefix_data["positions"][:1, :L_prompt].to(device)
        ages = prefix_data["ages"][:1, :L_prompt].to(device)
        d_next = prefix_data["delta_next"][:1, :L_prompt].to(device)
        d_prev = prefix_data["delta_prev"][:1, :L_prompt].to(device)
        attn = prefix_data["attention_mask"][:1, :L_prompt].to(device)

        h = model.backbone(
            e=e_curr,
            positions=pos,
            ages=ages,
            delta_next=d_next,
            delta_prev=d_prev,
            attention_mask=attn,
        )

        consecutive_zero_gaps = 0
        cumulative_hours = 0.0

        for step_i in range(1, n_rollout_steps + 1):
            last_h = h[:, -1:]
            mu_p, sigma_p, _ = model.prior(last_h)
            z_sample = mu_p + sigma_p * torch.randn_like(sigma_p)
            dec = model.decoder(z_sample)

            # Sample category and modality
            pred_cat = sample_top_p(dec.category_logits[0, 0], temperature=temperature, top_p=top_p)
            pred_mod = sample_top_p(dec.modality_logits[0, 0], temperature=temperature, top_p=top_p)

            # Time hurdle prediction
            gate_prob = torch.sigmoid(dec.time_gate_logits[0, 0, 0]).item()
            is_positive_hurdle = (gate_prob > 0.5)

            # Anti-Stall: If too many simultaneous events occur, force time advance
            if not is_positive_hurdle:
                consecutive_zero_gaps += 1
                if consecutive_zero_gaps > max_consecutive_zero_gaps:
                    is_positive_hurdle = True
                    consecutive_zero_gaps = 0
            else:
                consecutive_zero_gaps = 0

            if is_positive_hurdle:
                raw_log = torch.clamp(dec.log_gap_reg[0, 0, 0], min=0.0).item()
                pred_gap = max(min_forced_gap_hours, float(np.expm1(raw_log)))
            else:
                pred_gap = 0.0

            cumulative_hours += pred_gap

            cat_name = EventCategory(pred_cat).name if pred_cat < len(EventCategory) else f"CAT_{pred_cat}"
            mod_name = Modality(pred_mod).name if pred_mod < len(Modality) else f"MOD_{pred_mod}"

            trajectory.append({
                "step": step_i,
                "category": cat_name,
                "category_id": pred_cat,
                "modality": mod_name,
                "modality_id": pred_mod,
                "predicted_gap_hours": round(pred_gap, 2),
                "cumulative_hours": round(cumulative_hours, 2),
                "is_simultaneous": (pred_gap == 0.0),
            })

            # Re-encoding continuous payloads (U-R02)
            new_cat = torch.tensor([[pred_cat]], device=device, dtype=torch.int64)
            new_mod = torch.tensor([[pred_mod]], device=device, dtype=torch.int64)
            
            # Specifics payload
            new_has_spec = (torch.sigmoid(dec.specifics_presence_logits[:, :1].squeeze(-1)) > 0.5)
            new_spec = dec.specifics_vec[:, :1, :768]

            # Modality value payload
            new_has_val = torch.ones(1, 1, device=device, dtype=torch.bool)
            if pred_mod == Modality.NUMERIC:
                new_val = F.pad(dec.numeric_vec[:, :1], (0, 768 - 44))
            elif pred_mod == Modality.CXR:
                new_val = dec.cxr_vec[:, :1]
            elif pred_mod == Modality.ECG:
                new_val = F.pad(dec.ecg_vec[:, :1], (0, 768 - 192))
            elif pred_mod == Modality.TEXT:
                new_val = dec.text_vec[:, :1]
            else:
                new_val = torch.zeros(1, 1, 768, device=device)
                new_has_val = torch.zeros(1, 1, device=device, dtype=torch.bool)

            new_e = model.projector(new_cat, new_mod, new_spec, new_has_spec, new_val, new_has_val)

            # Append to rolling context
            e_curr = torch.cat([e_curr, new_e], dim=1)
            pos = torch.cat([pos, pos[:, -1:] + 1.0], dim=1)
            ages = torch.cat([ages, ages[:, -1:] + (pred_gap / 8760.0)], dim=1)
            d_prev = torch.cat([d_prev, torch.tensor([[pred_gap]], device=device)], dim=1)
            d_next = torch.cat([d_next, torch.zeros(1, 1, device=device)], dim=1)
            attn = torch.cat([attn, torch.ones(1, 1, device=device, dtype=torch.bool)], dim=1)

            # Recompute backbone causal history
            h = model.backbone(
                e=e_curr,
                positions=pos,
                ages=ages,
                delta_next=d_next,
                delta_prev=d_prev,
                attention_mask=attn,
            )

    return trajectory
