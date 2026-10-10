"""Tests for the SAE milestone: model, training, resampling, caching, evals, inspection."""

import json

import numpy as np
import pytest
import torch

from tiny_gpt_sae.data import TokenDataset
from tiny_gpt_sae.model import GPT, GPTConfig
from tiny_gpt_sae.sae import SAE, SAEConfig
from tiny_gpt_sae.sae.activations import ActivationStore, cache_activations, capture_residual
from tiny_gpt_sae.sae.evals import downstream_loss, reconstruction_metrics, replace_residual
from tiny_gpt_sae.sae.inspect import render, top_activations
from tiny_gpt_sae.sae.train import SAETrainConfig, resample_dead, train_sae
from tiny_gpt_sae.train import TrainConfig, train

GPT_CFG = GPTConfig(vocab_size=64, context_length=16, n_layer=2, n_head=2, d_model=32)


def token_ds() -> TokenDataset:
    return TokenDataset((np.arange(3000) % 29).astype(np.uint16), context_length=16)


def synthetic(n_true=32, d=64, k=3, seed=0):
    """Synthetic data with a known dictionary: each sample is a sum of k random true features."""
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(n_true, d, generator=g)
    D = D / D.norm(dim=1, keepdim=True)

    def batches(bs=256):
        while True:
            idx = torch.rand(bs, n_true, generator=g).argsort(dim=1)[:, :k]
            coef = torch.zeros(bs, n_true).scatter(1, idx, torch.rand(bs, k, generator=g) + 0.5)
            yield coef @ D

    return D, batches()


# ---------- model ----------

@pytest.mark.parametrize("kind", ["relu", "topk"])
@pytest.mark.parametrize("tied", [False, True])
def test_shapes(kind, tied):
    sae = SAE(SAEConfig(d_in=16, n_features=64, kind=kind, k=4, tied=tied))
    x = torch.randn(5, 7, 16)
    xhat, f = sae(x)
    assert xhat.shape == x.shape and f.shape == (5, 7, 64)
    assert (f >= 0).all()
    if kind == "topk":
        assert ((f > 0).sum(-1) <= 4).all()


@pytest.mark.parametrize("tied", [False, True])
def test_decoder_norm_constraint_holds_after_steps(tied):
    sae = SAE(SAEConfig(d_in=16, n_features=32, tied=tied))
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    for _ in range(5):
        loss, _ = sae.loss(torch.randn(64, 16) * 3)
        opt.zero_grad()
        loss.backward()
        sae.remove_parallel_decoder_grad()
        opt.step()
        sae.normalize_decoder()
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(32), atol=1e-5)


def test_parallel_grad_removed():
    sae = SAE(SAEConfig(d_in=8, n_features=16))
    sae.loss(torch.randn(32, 8))[0].backward()
    sae.remove_parallel_decoder_grad()
    assert (sae._W_dec.grad * sae._W_dec).sum(1).abs().max() < 1e-5


def test_bad_config():
    with pytest.raises(ValueError):
        SAEConfig(d_in=8, n_features=4, kind="topk", k=10)


# ---------- training ----------

@pytest.mark.parametrize("kind", ["relu", "topk"])
def test_loss_decreases_on_synthetic_sparse_data(kind):
    _, data = synthetic()
    cfg = SAETrainConfig(sae=SAEConfig(d_in=64, n_features=64, kind=kind, k=3, l1_coeff=1e-2),
                         steps=300, lr=3e-3, l1_warmup_steps=50, resample_every=0)
    _, hist = train_sae(cfg, data, log=False)
    assert np.mean([h["mse"] for h in hist[-20:]]) < 0.3 * np.mean([h["mse"] for h in hist[:20]])


def test_recovers_known_features():
    """A TopK SAE trained on sums of k=3 known directions should learn those directions."""
    D, data = synthetic(n_true=32, d=64, k=3, seed=1)
    cfg = SAETrainConfig(sae=SAEConfig(d_in=64, n_features=64, kind="topk", k=3),
                         steps=1500, lr=3e-3, resample_every=500, resample_until=1200)
    sae, _ = train_sae(cfg, data, log=False)
    # For each true feature, cosine similarity with its best-matching learned decoder row.
    cos = (D @ sae.W_dec.detach().T).max(dim=1).values
    assert cos.mean() > 0.9, cos
    assert (cos > 0.8).float().mean() > 0.85, cos


def test_resampling_revives_dead_features():
    sae = SAE(SAEConfig(d_in=16, n_features=32))
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    x = torch.randn(256, 16)
    sae.loss(x)[0].backward()
    opt.step()
    with torch.no_grad():
        sae.b_enc[:8] = -1e3  # force features 0-7 dead
    assert (sae.encode(x)[:, :8] == 0).all()
    dead = torch.zeros(32, dtype=torch.bool)
    dead[:8] = True
    assert resample_dead(sae, opt, dead, x) == 8
    assert (sae.b_enc[:8] == 0).all()
    assert torch.allclose(sae.W_dec[:8].norm(dim=1), torch.ones(8), atol=1e-5)
    assert (opt.state[sae.b_enc]["exp_avg"][:8] == 0).all()
    assert (sae.encode(x)[:, :8] > 0).any()  # they fire again


# ---------- activation caching ----------

def test_cache_matches_block_output_and_shards(tmp_path):
    torch.manual_seed(0)
    model = GPT(GPT_CFG).eval()
    shards = cache_activations(model, token_ds(), layer=1, out_dir=tmp_path, n_tokens=1000,
                               batch_size=8, shard_tokens=300)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["n_tokens"] == 1000 and len(shards) == 4  # 300 + 300 + 300 + 100
    assert [len(torch.load(p)) for p in shards] == [300, 300, 300, 100]

    # The first cached rows must equal blocks[1]'s output computed by hand.
    x, _ = token_ds().get_batch(8, torch.Generator().manual_seed(0))
    with torch.no_grad():
        h = model.drop(model.tok_emb(x) + model.pos_emb(torch.arange(16)))
        for blk in model.blocks[:2]:
            h = blk(h)
    first = torch.load(shards[0]).float()
    assert torch.allclose(first[:128], h.reshape(-1, 32), atol=1e-2)  # first batch = 8 x 16 tokens; fp16 storage

    store = iter(ActivationStore(tmp_path, batch_size=50))
    assert next(store).shape == (50, 32)


def test_capture_hook_is_removed():
    model = GPT(GPT_CFG)
    with capture_residual(model, 0):
        pass
    assert len(model.blocks[0]._forward_hooks) == 0


# ---------- evals ----------

def test_downstream_hook_identity_and_zero():
    """Splicing in the exact activations changes nothing; zeros must match a manual zero-ablation."""
    torch.manual_seed(0)
    model = GPT(GPT_CFG).eval()
    x, y = token_ds().get_batch(4, torch.Generator().manual_seed(0))
    clean, _ = model(x)
    with torch.no_grad(), replace_residual(model, 0, lambda a: a.clone()):
        same, _ = model(x)
    assert torch.allclose(clean, same)

    with torch.no_grad(), replace_residual(model, 0, torch.zeros_like):
        zeroed, _ = model(x)
        manual = model.lm_head(model.ln_f(model.blocks[1](torch.zeros(4, 16, 32))))
    assert torch.allclose(zeroed, manual, atol=1e-6)
    assert len(model.blocks[0]._forward_hooks) == 0  # hook cleaned up


def test_downstream_loss_perfect_sae_recovers_everything(tmp_path):
    class IdentitySAE(torch.nn.Module):
        def forward(self, a):
            return a, a

    # Train briefly so clean and zero-ablated losses differ (loss_recovered is a ratio of their gap).
    cfg = TrainConfig(model=GPT_CFG, device="cpu", batch_size=8, max_steps=60, lr=3e-3, warmup_steps=5,
                      mixed_precision=False, resume=False, out_dir=str(tmp_path))
    model, _ = train(cfg, token_ds(), log=False)
    model.eval()
    g = torch.Generator().manual_seed(0)
    batches = [token_ds().get_batch(4, g) for _ in range(2)]
    r = downstream_loss(model, IdentitySAE(), 0, batches)
    assert abs(r["ce_sae"] - r["ce_clean"]) < 1e-6
    assert r["ce_zero"] > r["ce_clean"] + 0.1
    assert abs(r["loss_recovered"] - 1.0) < 1e-4


def test_reconstruction_metrics():
    sae = SAE(SAEConfig(d_in=16, n_features=32))
    with torch.no_grad():
        sae.b_enc[:4] = -1e3  # 4 features that can never fire
    m = reconstruction_metrics(sae, torch.randn(500, 16))
    assert m["n_dead"] >= 4 and m["l0"] > 0 and m["fve"] <= 1.0


# ---------- inspection ----------

def test_top_activations_and_render(tmp_path):
    torch.manual_seed(0)
    model = GPT(GPT_CFG).eval()
    sae = SAE(SAEConfig(d_in=32, n_features=16, kind="topk", k=4))
    top = top_activations(model, sae, 1, token_ds(), n_batches=3, batch_size=4, top_n=5)
    for hits in top.values():
        acts = [h[0] for h in hits]
        assert acts == sorted(acts, reverse=True) and len(acts) <= 5
    render(top, top_activations.windows, lambda ids: f"t{ids[0]}", tmp_path)
    data = json.loads((tmp_path / "features.json").read_text())
    assert set(data) == {str(i) for i in range(16)}
    assert "<b" in (tmp_path / "features.html").read_text()
