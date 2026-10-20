"""Candidate-equivalent updates and RNG-isolated evaluation; M1 remains unchanged."""
from contextlib import contextmanager
import copy
import random

import numpy as np
import torch

from noah_baseline.evaluation.contracts import ContractError
from noah_baseline.m1.training import Trainer
from noah_baseline.m15.metrics import score, targets


def finite_state(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(v) for v in value)
    return True


def exact(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def state(trainer):
    return copy.deepcopy(dict(model=trainer.model.state_dict(), optimizer=trainer.optimizer.state_dict(),
        step=trainer.step, cursor=trainer.cursor, torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate()))


class LearningTrainer(Trainer):
    """Same update math as M1, with an explicit approved cap and component observations."""
    def __init__(self, config, windows, identity, max_updates):
        if not 0 < max_updates <= 5260:
            raise ContractError('M1.5 optimizer cap invalid')
        super().__init__(config, windows, identity)
        self.max_updates = max_updates

    def update(self):
        if self.step >= self.max_updates:
            raise ContractError('Approved M1.5 optimizer limit reached')
        lr, beta = self.clock()
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.zero_grad(); self.model.train()
        remaining = len(self.windows) - self.cursor % len(self.windows)
        n = min(self.config['accumulation'], remaining)
        losses, ids, components = [], [], []
        for offset in range(n):
            window = self.windows[(self.cursor + offset) % len(self.windows)]
            loss, breakdown, _ = self.model(**window.batch, beta=beta)
            if not torch.isfinite(loss):
                raise ContractError('Nonfinite M1.5 training objective')
            (loss/n).backward()
            losses.append(float(loss.detach())); ids.extend(window.owned_target_ids)
            components.append({k: float(v.detach()) for k, v in breakdown._asdict().items()})
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
        if not torch.isfinite(norm):
            raise ContractError('Nonfinite M1.5 gradient norm')
        self.optimizer.step(); self.cursor += n; self.step += 1
        if not finite_state(self.model.state_dict()) or not finite_state(self.optimizer.state_dict()):
            raise ContractError('Nonfinite model/optimizer state')
        return dict(step=self.step, cursor=self.cursor, lr=lr, beta=beta, loss=float(np.mean(losses)),
            gradient_norm=float(norm), microbatches=n, target_ids=ids,
            components={k: float(np.mean([c[k] for c in components])) for k in components[0]})


@contextmanager
def isolated_evaluation(model):
    torch_rng = torch.get_rng_state(); numpy_rng = np.random.get_state(); python_rng = random.getstate()
    training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        torch.set_rng_state(torch_rng); np.random.set_state(numpy_rng); random.setstate(python_rng)
        model.train(training)


def evaluate(model, windows, ordinals, beta, check=lambda: None):
    """Draw seed1729 + stable corpus ordinal*100 + stream; streams0,1..8,99."""
    predictions = {k: [] for k in ('probability', 'gate', 'log_gap', 'numeric')}
    posterior_probability, components = [], []
    with isolated_evaluation(model):
        for window, ordinal in zip(windows, ordinals):
            check()
            b = window.batch
            mask = b['loss_mask'][:, 1:] & b['attention_mask'][:, 1:] & b['attention_mask'][:, :-1]
            seed = 1729 + 100 * ordinal
            torch.manual_seed(seed)
            _, breakdown, aux = model(**b, beta=beta)
            record = {k: float(v) for k, v in breakdown._asdict().items()}
            record['fixed_beta1_total'] = .3*record['loss_rec_post']+record['loss_kl']+.1*record['loss_rec_prior']
            # Reconstruct identical single-sample branches to expose both sets of component means.
            torch.manual_seed(seed)
            post_z = aux['mu_q'] + aux['sigma_q'] * torch.randn_like(aux['mu_q'])
            prior_z = aux['mu_p'] + aux['sigma_p'] * torch.randn_like(aux['mu_p'])
            for branch, latent in [('posterior', post_z), ('prior', prior_z)]:
                decoded = model.decoder(latent)
                rec, parts = model._compute_reconstruction_loss(decoded, b['category_ids'][:, 1:],
                    b['modality_ids'][:, 1:], b['specifics_vec'][:, 1:], b['has_specifics'][:, 1:],
                    b['value_vec'][:, 1:], b['has_value'][:, 1:], b['delta_next'][:, 1:],
                    b['has_next_gap'][:, 1:], mask)
                if not torch.equal(rec, breakdown.loss_rec_post if branch == 'posterior' else breakdown.loss_rec_prior):
                    raise ContractError('Component reconstruction draw alignment failed')
                record.update({branch+'_'+k: float(v) for k, v in parts.items()})
            components.append(record)
            draws = {k: [] for k in predictions}
            for draw in range(1, 9):
                generator = torch.Generator(device='cpu').manual_seed(seed+draw)
                eps = torch.randn(aux['mu_p'].shape, generator=generator)
                out = model.decoder(aux['mu_p']+aux['sigma_p']*eps)
                draws['probability'].append(out.category_logits[mask].double().softmax(-1).numpy())
                draws['gate'].append(out.time_gate_logits.squeeze(-1)[mask].double().sigmoid().numpy())
                draws['log_gap'].append(out.log_gap_reg.squeeze(-1)[mask].double().numpy())
                draws['numeric'].append(out.numeric_vec[mask].double().numpy())
            for key in predictions:
                predictions[key].append(np.mean(draws[key], axis=0))
            generator = torch.Generator(device='cpu').manual_seed(seed+99)
            qout = model.decoder(aux['mu_q']+aux['sigma_q']*torch.randn(aux['mu_q'].shape, generator=generator))
            posterior_probability.append(qout.category_logits[mask].double().softmax(-1).numpy())
    prediction = {k: np.concatenate(v) for k, v in predictions.items()}
    prediction['posterior_probability'] = np.concatenate(posterior_probability)
    truth = targets(windows)
    metrics = score(truth, prediction)
    y = truth['category']
    metrics['posterior_reconstruction_category_nll'] = float(-np.log(prediction['posterior_probability'][np.arange(len(y)), y]).mean())
    objective = {key: float(np.mean([c[key] for c in components])) for key in components[0]}
    if not all(np.isfinite(v) for v in objective.values()):
        raise ContractError('Nonfinite evaluation objective')
    return prediction, metrics, objective
