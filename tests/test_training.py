import tempfile

import numpy as np
import torch

from tiny_gpt_sae.data import TokenDataset
from tiny_gpt_sae.model import GPTConfig
from tiny_gpt_sae.train import TrainConfig, lr_at, train


def tiny_cfg(**overrides) -> TrainConfig:
    model = GPTConfig(vocab_size=64, context_length=16, n_layer=2, n_head=2, d_model=32)
    base = dict(model=model, device="cpu", batch_size=8, max_steps=150, lr=3e-3, min_lr=3e-4,
                warmup_steps=10, mixed_precision=False,
                resume=False, out_dir=tempfile.mkdtemp())
    base.update(overrides)
    return TrainConfig(**base)


def repeating_tokens(n: int = 2000) -> np.ndarray:
    # A deterministic repeating pattern a small model can memorize quickly.
    return (np.arange(n) % 23).astype(np.uint16)


def test_batch_targets_are_shifted_inputs():
    ds = TokenDataset(repeating_tokens(), context_length=16)
    x, y = ds.get_batch(4, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_overfit_tiny_dataset_loss_decreases():
    ds = TokenDataset(repeating_tokens(), context_length=16)
    _, losses = train(tiny_cfg(), ds, log=False)
    first, last = np.mean(losses[:10]), np.mean(losses[-10:])
    assert last < 0.25 * first, (first, last)
    assert last < 0.5


def test_training_is_deterministic():
    ds = TokenDataset(repeating_tokens(), context_length=16)
    m1, l1 = train(tiny_cfg(max_steps=20), ds, log=False)
    m2, l2 = train(tiny_cfg(max_steps=20), ds, log=False)
    assert l1 == l2
    for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=False):
        assert torch.equal(p1, p2)


def test_different_seeds_differ():
    ds = TokenDataset(repeating_tokens(), context_length=16)
    _, l1 = train(tiny_cfg(max_steps=5, seed=1), ds, log=False)
    _, l2 = train(tiny_cfg(max_steps=5, seed=2), ds, log=False)
    assert l1 != l2


def test_lr_schedule():
    cfg = tiny_cfg(max_steps=100, warmup_steps=10)
    assert lr_at(0, cfg) < lr_at(9, cfg) == cfg.lr
    assert abs(lr_at(99, cfg) - cfg.min_lr) < 1e-5


def test_config_yaml_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  n_layer: 3\n  d_model: 48\n  n_head: 3\nbatch_size: 4\n")
    cfg = TrainConfig.from_yaml(p)
    assert cfg.model.n_layer == 3 and cfg.batch_size == 4
