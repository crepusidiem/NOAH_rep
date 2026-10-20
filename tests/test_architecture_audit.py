"""Independent architecture and information-boundary checks for the paper audit."""
import math
from unittest.mock import patch
import numpy as np
import pytest
import torch
from src.data.schema import Modality
from src.models.noah import NOAH
from src.models.optimization import NOAHOptimizer, partition_parameters
from src.models.rollout import generate_autoregressive_rollout
from src.models.transformer import TemporalAttentionLayer, DropPath


def tiny_model():
    return NOAH(d_model=32, d_latent=16, n_layers=1, n_heads=2, head_dim=16,
                ffn_ratio=2, dropout=0, attn_dropout=0)


def prompt(length=9):
    return dict(category_ids=torch.zeros(1, length, dtype=torch.long),
                modality_ids=torch.zeros(1, length, dtype=torch.long),
                specifics_vec=torch.zeros(1, length, 768),
                has_specifics=torch.zeros(1, length, dtype=torch.bool),
                value_vec=torch.zeros(1, length, 768),
                has_value=torch.zeros(1, length, dtype=torch.bool),
                positions=torch.arange(length).float()[None], ages=torch.full((1, length), 60.),
                delta_prev=torch.ones(1, length), delta_next=torch.full((1, length), 999.),
                attention_mask=torch.ones(1, length, dtype=torch.bool))


def test_attention_stays_causal_with_only_padding_mask():
    torch.manual_seed(1)
    layer = TemporalAttentionLayer(32, 2, 16, dropout=0, attn_dropout=0).eval()
    x = torch.randn(1, 5, 32)
    changed = x.clone()
    changed[:, 3:] += 100
    valid = torch.ones(1, 5, dtype=torch.bool)
    torch.testing.assert_close(layer(x, padding_mask=valid)[:, :3],
                               layer(changed, padding_mask=valid)[:, :3])


def test_input_time_and_prior_content_boundary():
    model = tiny_model().eval()
    p = prompt()
    def prior_means(p):
        e = model.projector(p['category_ids'], p['modality_ids'], p['specifics_vec'],
                            p['has_specifics'], p['value_vec'], p['has_value'])
        h = model.backbone(e, p['positions'], p['ages'], p['delta_next'],
                           p['delta_prev'], p['attention_mask'])
        return model.prior(h[:, :-1])[0]
    before = prior_means(p)
    p['category_ids'][:, 5:] = 7
    torch.testing.assert_close(before[:, :5], prior_means(p)[:, :5])
    bank = model.backbone.input_time_bank
    u = torch.tensor([[[2., 60., math.log1p(3), math.log1p(4)]]])
    expected = []
    for scalar in [2., 60., math.log1p(3), math.log1p(4)]:
        expected.extend(math.sin(scalar * 10000 ** (-n / 4)) for n in range(4))
        expected.extend(math.cos(scalar * 10000 ** (-n / 4)) for n in range(4))
    torch.testing.assert_close(bank(u)[0, 0], torch.tensor(expected))


def test_echo_loss_and_target_mask():
    model = tiny_model()
    z = torch.randn(1, 2, 16)
    dec = model.decoder(z)
    kwargs = dict(dec_out=dec, category_targets=torch.zeros(1, 2, dtype=torch.long),
                  modality_targets=torch.full((1, 2), int(Modality.ECHO)),
                  specifics_targets=torch.zeros(1, 2, 768), has_specifics=torch.zeros(1, 2, dtype=torch.bool),
                  value_targets=torch.zeros(1, 2, 768), has_value=torch.ones(1, 2, dtype=torch.bool),
                  delta_targets=torch.zeros(1, 2), has_next_gap=torch.zeros(1, 2, dtype=torch.bool),
                  loss_mask=torch.tensor([[True, False]]))
    kwargs['value_targets'][0, 0, :512] = 1
    loss, breakdown = model._compute_reconstruction_loss(**kwargs)
    reference = ((dec.echo_vec[0, 0] - 1) ** 2).mean()
    torch.testing.assert_close(breakdown['value'], reference.detach())
    kwargs['value_targets'][0, 1] = 1e6
    same_loss, _ = model._compute_reconstruction_loss(**kwargs)
    torch.testing.assert_close(loss, same_loss)
    loss.backward()
    assert model.decoder.H_v_echo.weight.grad.abs().sum() > 0


def test_rollout_outgoing_time_support_history_and_temperature():
    model = tiny_model().eval()
    p = prompt()
    support = torch.zeros(31, 6, dtype=torch.bool)
    support[2, Modality.ECHO] = True
    with torch.no_grad():
        model.decoder.H_delta_gate.weight.zero_()
        model.decoder.H_delta_gate.bias.fill_(100)
        model.decoder.H_delta_reg.weight.zero_()
        model.decoder.H_delta_reg.bias.fill_(math.log1p(2))
    calls = []
    handle = model.backbone.register_forward_pre_hook(lambda module, args, kwargs: calls.append(kwargs), with_kwargs=True)
    with patch.object(model.posterior, 'forward', side_effect=AssertionError('posterior used')):
        one = generate_autoregressive_rollout(model, p, 'cpu', 3, temperature=0,
                                              initial_gap_hours=1, category_modality_support=support)
        two = generate_autoregressive_rollout(model, p, 'cpu', 3, temperature=0,
                                              initial_gap_hours=1, category_modality_support=support)
    handle.remove()
    assert one == two
    assert calls[0]['e'].shape[1] == 9
    assert calls[0]['delta_next'][0, -1] == 1  # held-out 999 is overwritten
    assert [row['category_id'] for row in one] == [2] * 3
    assert [row['modality_id'] for row in one] == [int(Modality.ECHO)] * 3
    assert [row['cumulative_hours'] for row in one] == pytest.approx([1, 3, 5])
    assert calls[1]['delta_prev'][0, -1] == 1
    assert calls[1]['delta_next'][0, -1] == pytest.approx(2)
    assert not any(row['forced_time_advance'] for row in one)
    assert calls[1]['e'].shape[1] == 10


def test_free_zero_gaps_are_preserved():
    model = tiny_model()
    with torch.no_grad():
        model.decoder.H_delta_gate.weight.zero_()
        model.decoder.H_delta_gate.bias.fill_(-100)
    support = torch.zeros(31, 6, dtype=torch.bool)
    support[0, 0] = True
    rows = generate_autoregressive_rollout(model, prompt(), 'cpu', 5, temperature=0,
                                           category_modality_support=support)
    assert all(row['predicted_gap_hours'] == 0 for row in rows)
    assert all(row['cumulative_hours'] == 0 for row in rows)


def test_optimizer_routing_and_state_roundtrip():
    model = tiny_model()
    muon, adamw = partition_parameters(model)
    muon_names, adamw_names = {n for n, _ in muon}, {n for n, _ in adamw}
    assert 'backbone.blocks.0.attn.q_proj.weight' in muon_names
    assert 'projector.category_embed.weight' in adamw_names
    assert 'decoder.H_v_echo.weight' in adamw_names
    assert 'prior.net.2.weight' in adamw_names
    assert not muon_names & adamw_names
    assert muon_names | adamw_names == {n for n, p in model.named_parameters() if p.requires_grad}
    optimizer = NOAHOptimizer(model, 6e-4, .01)
    sum(p.square().sum() for p in model.parameters()).backward()
    optimizer.step()
    optimizer.load_state_dict(optimizer.state_dict())


def test_drop_path_is_per_example_and_disabled_at_eval():
    layer = DropPath(.5)
    x = torch.ones(128, 3, 4)
    y = layer(x)
    assert set(y.unique().tolist()) == {0., 2.}
    assert (y == y[:, :1, :1]).all()
    layer.eval()
    torch.testing.assert_close(layer(x), x)


def test_ecg_rejects_legacy_cache_without_checkpoint(tmp_path, monkeypatch):
    from src.data.embeddings_cache import ECGEmbeddingCache
    monkeypatch.delenv('NOAH_OTIS_CHECKPOINT', raising=False)
    np.save(tmp_path / 'synthetic.npy', np.zeros(192))
    cache = ECGEmbeddingCache(cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match='pretrained 1x24 OTIS checkpoint'):
        cache.get_embedding('synthetic', 'missing')


def test_failed_ecg_checkpoint_cannot_leave_random_encoder_active(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from src.data.embeddings_cache import ECGEmbeddingCache
    checkpoint = tmp_path / 'invalid.pt'
    torch.save({'bad': torch.zeros(1)}, checkpoint)
    monkeypatch.setitem(sys.modules, 'models_vit', SimpleNamespace(
        vit_baseDeep_patchX=lambda **kwargs: torch.nn.Linear(2, 2)))
    cache = ECGEmbeddingCache(cache_dir=tmp_path / 'cache', checkpoint_path=checkpoint)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            cache._lazy_load_model()
        assert cache._model is None
