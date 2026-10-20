"""Paper optimizer routing (OPT-01); algorithm internals remain assumption A-31."""
import torch
from torch import nn


def partition_parameters(model):
    """Hidden linear matrices use Muon; embeddings, output heads and vectors use AdamW."""
    hidden_ids = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Last layers of prior/posterior are distribution output heads.
            output = (name.startswith("decoder.H_") or name in
                      {"prior.net.2", "posterior.net.2"})
            if not output:
                hidden_ids.add(id(module.weight))
    muon, adamw = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (muon if id(parameter) in hidden_ids else adamw).append((name, parameter))
    return muon, adamw


class NOAHOptimizer:
    """Two optimizers with one training-loop interface and explicit checkpoint state."""
    def __init__(self, model, lr, weight_decay, mode="muon_adamw"):
        if mode == "adamw":
            self.optimizers = [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)]
            self.parameter_manifest = {"adamw": [n for n, p in model.named_parameters() if p.requires_grad]}
        elif mode == "muon_adamw":
            if not hasattr(torch.optim, "Muon"):
                raise RuntimeError("Muon is unavailable; install a compatible PyTorch or explicitly choose the AdamW pilot deviation")
            muon, adamw = partition_parameters(model)
            self.parameter_manifest = {"muon": [n for n, _ in muon], "adamw": [n for n, _ in adamw]}
            self.optimizers = [
                torch.optim.Muon([p for _, p in muon], lr=lr, weight_decay=weight_decay,
                                 momentum=0.95, nesterov=True, ns_steps=5),
                torch.optim.AdamW([p for _, p in adamw], lr=lr, weight_decay=weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8),
            ]
        else:
            raise ValueError(f"Unknown optimizer mode: {mode}")
        self.mode = mode
        self.param_groups = [g for optimizer in self.optimizers for g in optimizer.param_groups]

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return {"mode": self.mode, "parameter_manifest": self.parameter_manifest,
                "optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state):
        if state["mode"] != self.mode or state["parameter_manifest"] != self.parameter_manifest:
            raise ValueError("Optimizer parameter routing does not match checkpoint")
        for optimizer, saved in zip(self.optimizers, state["optimizers"], strict=True):
            optimizer.load_state_dict(saved)
