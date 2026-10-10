"""Tests for causal interventions, emergence matching and auto-interp utilities."""

import math

import numpy as np
import pytest
import torch

from tiny_gpt_sae.data import TokenDataset
from tiny_gpt_sae.model import GPT, GPTConfig
from tiny_gpt_sae.sae import SAE, SAEConfig
from tiny_gpt_sae.sae.activations import capture_residual
from tiny_gpt_sae.sae.autointerp import (
    auroc,
    balanced_accuracy,
    clean_explanation,
    explain_prompt,
    mark_context,
    score_prompt,
    split_examples,
)
from tiny_gpt_sae.sae.causal import ablate, ablate_fn, ablation_effect, clamp, clamp_fn, generate_with
from tiny_gpt_sae.sae.emergence import first_appearance, max_cosine

GPT_CFG = GPTConfig(vocab_size=64, context_length=16, n_layer=2, n_head=2, d_model=32)


def setup():
    torch.manual_seed(0)
    model = GPT(GPT_CFG).eval()
    sae = SAE(SAEConfig(d_in=32, n_features=64, kind="topk", k=8)).eval()
    x, _ = TokenDataset((np.arange(2000) % 29).astype(np.uint16), 16).get_batch(4, torch.Generator().manual_seed(0))
    return model, sae, x


# ---------- causal ----------

def test_ablate_removes_exactly_the_feature_contribution():
    _, sae, _ = setup()
    a = torch.randn(3, 5, 32)
    f = sae.encode(a)
    out = ablate_fn(sae, 7)(a)
    assert torch.allclose(a - out, f[..., 7:8] * sae.W_dec[7], atol=1e-6)


def test_clamp_sets_feature_value_along_its_direction():
    _, sae, _ = setup()
    a = torch.randn(2, 4, 32)
    out = clamp_fn(sae, 3, 5.0)(a)
    delta = out - a
    # The change is parallel to the decoder direction and has the right size.
    expected = (5.0 - sae.encode(a)[..., 3:4]) * sae.W_dec[3]
    assert torch.allclose(delta, expected, atol=1e-5)


def test_ablating_inactive_feature_is_a_noop():
    model, sae, x = setup()
    with torch.no_grad(), capture_residual(model, 0) as store:
        model(x)
    f = sae.encode(store[0])
    inactive = int((f.sum((0, 1)) == 0).nonzero()[0])  # a feature that never fires on x
    clean, _ = model(x)
    with torch.no_grad(), ablate(model, sae, 0, inactive):
        abl, _ = model(x)
    assert torch.allclose(clean, abl, atol=1e-6)
    assert len(model.blocks[0]._forward_hooks) == 0


def test_ablation_effect_metrics():
    model, sae, x = setup()
    with torch.no_grad(), capture_residual(model, 0) as store:
        model(x)
    f = sae.encode(store[0])
    feat = int(f.sum((0, 1)).argmax())  # most active feature
    pos = (f[..., feat] > 0).nonzero()[:5]
    e = ablation_effect(model, sae, 0, feat, x, pos)
    assert e["kl"].shape == (len(pos),) and (e["kl"] >= -1e-6).all()
    assert e["kl"].max() > 0  # removing an active feature changes something
    # dlogp is NaN exactly where there's no next token in the window.
    last = pos[:, 1] == x.shape[1] - 1
    assert torch.isnan(e["dlogp"][last]).all() and not torch.isnan(e["dlogp"][~last]).any()


def test_clamp_changes_generation_but_seeded_baseline_is_reproducible():
    model, sae, _ = setup()
    idx = torch.zeros(1, 1, dtype=torch.long)
    a = generate_with(model, idx, 10, seed=3)
    b = generate_with(model, idx, 10, seed=3)
    assert torch.equal(a, b)
    logits_clean, _ = model(a)
    with torch.no_grad(), clamp(model, sae, 0, 5, 50.0):
        logits_clamped, _ = model(a)
    assert not torch.allclose(logits_clean, logits_clamped)


# ---------- emergence ----------

def test_max_cosine_finds_permuted_and_rescaled_rows():
    ref = torch.randn(20, 8)
    perm = torch.randperm(20)
    other = ref[perm] * 3.0
    cos, idx = max_cosine(ref, other, chunk=7)
    assert torch.allclose(cos, torch.ones(20), atol=1e-5)
    assert torch.equal(perm[idx], torch.arange(20))


def test_first_appearance_requires_staying_matched():
    steps = [0, 100, 1000, 10000]
    cos = torch.tensor([[0.9, 0.1, 0.2],
                        [0.2, 0.8, 0.3],   # feature 0 dips: it only *stably* appears at 1000
                        [0.9, 0.8, 0.4],
                        [0.95, 0.9, 0.5]])
    assert first_appearance(steps, cos, 0.7) == [1000, 100, None]


# ---------- autointerp ----------

def test_mark_context_and_prompts():
    c = mark_context(["Once", " upon", " a", "\n", "<|endoftext|>"], 1)
    assert c == "Once<< upon>> a [END]"
    p = explain_prompt(["a <<b>>", "c <<d>>"])
    assert p[0]["role"] == "system" and "1. a <<b>>" in p[1]["content"]
    assert "Description: x" in score_prompt("x", "y <<z>>")[1]["content"]


def test_clean_explanation():
    assert clean_explanation('The neuron responds to "to" after "way".\nMore') == '"to" after "way"'
    assert clean_explanation('"the word breath"') == "the word breath"
    assert clean_explanation("  \n") == "unclear"
    assert clean_explanation("Answer: the word breath.") == "the word breath"


def test_auroc_and_balanced_accuracy():
    assert auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5], [1, 0]) == 0.5
    assert math.isnan(auroc([0.5], [1]))
    assert balanced_accuracy([0.9, 0.4, 0.1, 0.6], [1, 1, 0, 0]) == pytest.approx(0.5)


def test_split_examples_disjoint_and_interleaved():
    ex, held = split_examples(20, 10)
    assert ex == list(range(0, 20, 2)) and set(ex).isdisjoint(held) and len(held) == 10
