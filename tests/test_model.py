import pytest
import torch

from tiny_gpt_sae.model import GPT, GPTConfig

TINY = GPTConfig(vocab_size=97, context_length=16, n_layer=2, n_head=2, d_model=32)


def test_logits_shape_and_loss():
    model = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (3, 10))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 10, TINY.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_initial_loss_near_uniform():
    # A freshly initialized model should predict roughly uniformly: loss ≈ ln(V).
    torch.manual_seed(0)
    model = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (8, 16))
    _, loss = model(idx, idx)
    assert abs(loss.item() - torch.log(torch.tensor(float(TINY.vocab_size))).item()) < 0.5


def test_rejects_too_long_sequence():
    with pytest.raises(ValueError):
        GPT(TINY)(torch.zeros(1, TINY.context_length + 1, dtype=torch.long))


def test_bad_head_config():
    with pytest.raises(ValueError):
        GPTConfig(d_model=30, n_head=4)


def test_causal_masking():
    """Changing a future token must not change outputs at earlier positions."""
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    a = torch.randint(0, TINY.vocab_size, (1, 12))
    b = a.clone()
    b[0, 8] = (a[0, 8] + 1) % TINY.vocab_size  # perturb position 8
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-6)  # positions 0..7 unaffected
    assert not torch.allclose(la[:, 8:], lb[:, 8:])  # position 8 onward does change


def test_weight_tying():
    model = GPT(TINY)
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_generate_length():
    model = GPT(TINY).eval()
    out = model.generate(torch.zeros(2, 1, dtype=torch.long), max_new_tokens=20, top_k=5)
    assert out.shape == (2, 21)  # can exceed context_length thanks to cropping
