"""Train-only baselines and independently replayable event-level diagnostic scores."""
import numpy as np

CLAIM = 'teacher-forced, time-conditioned category prediction'
SUPPORTED = [5, 14, 22, 28]


def targets(windows):
    rows = {key: [] for key in ('category', 'previous_category', 'gap', 'gap_known', 'numeric', 'value')}
    for window in windows:
        b = window.batch
        mask = b['loss_mask'][:, 1:] & b['attention_mask'][:, 1:] & b['attention_mask'][:, :-1]
        for key, tensor in (
            ('category', b['category_ids'][:, 1:]), ('previous_category', b['category_ids'][:, :-1]),
            ('gap', b['delta_next'][:, 1:]), ('gap_known', b['has_next_gap'][:, 1:]),
            ('numeric', b['has_value'][:, 1:] & (b['modality_ids'][:, 1:] == 1)),
            ('value', b['value_vec'][:, 1:, :44]),
        ):
            rows[key].append(tensor[mask].numpy().copy())
    return {key: np.concatenate(value) for key, value in rows.items()}


def fit_baselines(windows):
    t = targets(windows)
    counts = np.bincount(t['category'], minlength=31)
    unigram = (counts + 1) / (counts.sum() + 31)
    transitions = np.zeros((31, 31), dtype=np.int64)
    np.add.at(transitions, (t['previous_category'], t['category']), 1)
    positive = t['gap_known'] & (t['gap'] > 0)
    return dict(category_counts=counts, transition_counts=transitions, unigram=unigram,
        transition=(transitions + unigram) / (transitions.sum(1, keepdims=True) + 1),
        gate=float(positive.sum() / t['gap_known'].sum()),
        log_gap=float(np.log1p(t['gap'][positive].astype(np.float64)).mean()),
        median_gap=float(np.median(t['gap'][positive])),
        numeric_mean=t['value'][t['numeric']].astype(np.float64).mean(0))


def baseline_predictions(t, baseline, category_mode='unigram'):
    n = len(t['category'])
    probs = (np.broadcast_to(baseline['unigram'], (n, 31)).copy() if category_mode == 'unigram'
             else baseline['transition'][t['previous_category']])
    return dict(probability=probs, gate=np.full(n, baseline['gate']),
        log_gap=np.full(n, baseline['log_gap']), numeric=np.broadcast_to(baseline['numeric_mean'], (n, 44)))


def score(t, prediction):
    p = np.asarray(prediction['probability'], dtype=np.float64)
    gate = np.asarray(prediction['gate'], dtype=np.float64)
    reg = np.asarray(prediction['log_gap'], dtype=np.float64)
    num = np.asarray(prediction['numeric'], dtype=np.float64)
    if not all(np.isfinite(a).all() for a in (p, gate, reg, num)):
        raise ValueError('Nonfinite prediction')
    if (p <= 0).any() or not np.allclose(p.sum(1), 1, rtol=1e-6, atol=1e-8):
        raise ValueError('Invalid full-vocabulary category probability')
    if not ((gate > 0) & (gate < 1)).all():
        raise ValueError('Time gate must have finite log probabilities')
    y = t['category']; predicted = p.argmax(1)
    known = t['gap_known']; positive = known & (t['gap'] > 0)
    with np.errstate(over='ignore', invalid='ignore'):
        hour_pred = (gate >= .5) * np.expm1(np.maximum(reg, 0))
    if not np.isfinite(hour_pred).all():
        raise ValueError('Nonfinite hour inverse transform')
    pos_label = (t['gap'][known] > 0).astype(np.float64)
    g = gate[known]
    per_category = {}
    for category in SUPPORTED:
        tp = int(((predicted == category) & (y == category)).sum())
        fp = int(((predicted == category) & (y != category)).sum())
        fn = int(((predicted != category) & (y == category)).sum())
        per_category[str(category)] = dict(support=int((y == category).sum()),
            recall=tp / (tp + fn) if tp + fn else None,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return dict(claim=CLAIM, targets=len(y), category_nll=float(-np.log(p[np.arange(len(y)), y]).mean()),
        category_accuracy=float((predicted == y).mean()),
        macro_f1=float(np.mean([x['f1'] for x in per_category.values()])), per_category=per_category,
        rare_category_score=None, rare_category_reason='No category has1–99 owned training targets',
        gap_known=int(known.sum()), gap_positive=int(positive.sum()), gap_zero=int((known & ~positive).sum()),
        gap_unknown=int((~known).sum()), time_gate_log_loss=float(-(pos_label*np.log(g)+(1-pos_label)*np.log1p(-g)).mean()),
        time_gate_brier=float(np.mean((g-pos_label)**2)),
        positive_log1p_hours_rmse=float(np.sqrt(np.mean((reg[positive]-np.log1p(t['gap'][positive].astype(np.float64)))**2))),
        observed_gap_hours_mae=float(np.mean(np.abs(hour_pred[known]-t['gap'][known]))),
        positive_gap_hours_mae=float(np.mean(np.abs(hour_pred[positive]-t['gap'][positive]))),
        numeric_targets=int(t['numeric'].sum()), numeric44_mse=float(np.mean((num[t['numeric']]-t['value'][t['numeric']])**2)))
