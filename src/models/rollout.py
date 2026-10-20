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
    max_consecutive_zero_gaps: Optional[int] = None,
    min_forced_gap_hours: float = 0.5,
    initial_gap_hours: float = 0.0,
    discrete_sampling: bool = False,
    category_modality_support: Optional[torch.Tensor] = None,

) -> List[Dict[str, any]]:
    """
    Executes free prior-only autoregressive rollout with anti-stall zero-gap budgeting.
    """
    if not 0 <= temperature <= 1 or initial_gap_hours < 0:
        raise ValueError("Latent temperature must be in [0, 1] and initial gap nonnegative")
    if category_modality_support is None:
        category_modality_support = model.category_modality_support
    if category_modality_support is None:
        raise ValueError("Rollout requires observed category-modality support fitted on training data")
    support = category_modality_support.to(device=device, dtype=torch.bool)
    if support.shape != (model.decoder.num_categories, model.decoder.num_modalities):
        raise ValueError("Invalid category-modality support shape")
    if not support.any():
        raise ValueError("Category-modality support is empty")
    model.eval()
    trajectory = []

    with torch.no_grad():
        # Consume the complete valid supplied history, including clinical events.
        valid = prefix_data["attention_mask"][0].bool()
        L_prompt = int(valid.sum().item())
        if L_prompt == 0 or not valid[:L_prompt].all() or valid[L_prompt:].any():
            raise ValueError("Prompt must have a nonempty contiguous valid prefix")

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

        d_next = d_next.clone()
        d_next[:, -1] = initial_gap_hours

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
        incoming_gap = initial_gap_hours

        for step_i in range(1, n_rollout_steps + 1):
            last_h = h[:, -1:]
            mu_p, sigma_p, _ = model.prior(last_h)
            z_sample = mu_p + temperature * sigma_p * torch.randn_like(sigma_p)
            dec = model.decoder(z_sample)

            # Sample category and modality
            cat_logits = dec.category_logits[0, 0].masked_fill(~support.any(dim=-1), -torch.inf)
            pred_cat = (sample_top_p(cat_logits, temperature=temperature, top_p=top_p)
                        if discrete_sampling else int(cat_logits.argmax().item()))
            mod_logits = dec.modality_logits[0, 0].masked_fill(~support[pred_cat], -torch.inf)
            pred_mod = (sample_top_p(mod_logits, temperature=temperature, top_p=top_p)
                        if discrete_sampling else int(mod_logits.argmax().item()))

            # Time hurdle prediction
            gate_prob = torch.sigmoid(dec.time_gate_logits[0, 0, 0]).item()
            is_positive_hurdle = (gate_prob > 0.5)

            raw_log = torch.clamp(dec.log_gap_reg[0, 0, 0], min=0.0).item()
            pred_gap = float(np.expm1(raw_log)) if is_positive_hurdle else 0.0
            forced = False
            # Optional deviation budgets actual zero outputs, including clamped regressions.
            if pred_gap == 0:
                consecutive_zero_gaps += 1
                if max_consecutive_zero_gaps is not None and consecutive_zero_gaps > max_consecutive_zero_gaps:
                    pred_gap = min_forced_gap_hours
                    forced = True
                    consecutive_zero_gaps = 0
            else:
                consecutive_zero_gaps = 0

            if not np.isfinite(pred_gap):
                raise ValueError("Nonfinite generated outgoing gap")
            # The decoded gap belongs to this event's successor (p.20).
            cumulative_hours += incoming_gap

            cat_name = EventCategory(pred_cat).name if pred_cat < len(EventCategory) else f"CAT_{pred_cat}"
            mod_name = Modality(pred_mod).name if pred_mod < len(Modality) else f"MOD_{pred_mod}"

            trajectory.append({
                "step": step_i,
                "category": cat_name,
                "category_id": pred_cat,
                "modality": mod_name,
                "modality_id": pred_mod,
                "predicted_gap_hours": pred_gap,
                "cumulative_hours": cumulative_hours,
                "is_simultaneous": (incoming_gap == 0.0),
                "incoming_gap_hours": incoming_gap,
                "forced_time_advance": forced,
                "rollout_mode": "anti_stall_deviation" if max_consecutive_zero_gaps is not None else "free",

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
            elif pred_mod == Modality.ECHO:
                new_val = F.pad(dec.echo_vec[:, :1], (0, 768 - 512))
            else:
                new_val = torch.zeros(1, 1, 768, device=device)
                new_has_val = torch.zeros(1, 1, device=device, dtype=torch.bool)

            new_e = model.projector(new_cat, new_mod, new_spec, new_has_spec, new_val, new_has_val)

            # Append to rolling context
            e_curr = torch.cat([e_curr, new_e], dim=1)
            pos = torch.cat([pos, pos[:, -1:] + 1.0], dim=1)
            ages = torch.cat([ages, ages[:, -1:] + (incoming_gap / 8760.0)], dim=1)
            d_prev = torch.cat([d_prev, torch.tensor([[incoming_gap]], device=device)], dim=1)
            d_next = torch.cat([d_next, torch.tensor([[pred_gap]], device=device)], dim=1)
            incoming_gap = pred_gap
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
