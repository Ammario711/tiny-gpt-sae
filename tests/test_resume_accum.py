"""Tests for gradient accumulation and checkpoint resume."""

from contextlib import nullcontext

import numpy as np
import torch

from tiny_gpt_sae.data import TokenDataset
from tiny_gpt_sae.model import GPT, GPTConfig
from tiny_gpt_sae.train import TrainConfig, accumulate_gradients, amp_dtype, train

MODEL = GPTConfig(vocab_size=64, context_length=16, n_layer=2, n_head=2, d_model=32)


def ds() -> TokenDataset:
    return TokenDataset((np.arange(2000) % 23).astype(np.uint16), context_length=16)


def cfg(tmp_path, **kw) -> TrainConfig:
    base = {"model": MODEL, "device": "cpu", "batch_size": 4, "max_steps": 20, "lr": 3e-3, "warmup_steps": 5,
            "mixed_precision": False, "out_dir": str(tmp_path), "checkpoint_interval": 5}
    base.update(kw)
    return TrainConfig(**base)


def test_grad_accumulation_matches_full_batch():
    """Two micro-batches of 4 must give the same gradient as one batch of 8."""
    torch.manual_seed(0)
    model = GPT(MODEL)
    x, y = ds().get_batch(8, torch.Generator().manual_seed(1))
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    model.zero_grad()
    full = accumulate_gradients(model, [(x, y)], nullcontext, scaler)
    g_full = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    acc = accumulate_gradients(model, [(x[:4], y[:4]), (x[4:], y[4:])], nullcontext, scaler)
    g_acc = [p.grad.clone() for p in model.parameters()]

    assert abs(full - acc) < 1e-5
    for a, b in zip(g_full, g_acc, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_train_with_accumulation_runs_and_learns(tmp_path):
    _, losses = train(cfg(tmp_path, grad_accum_steps=3, max_steps=60), ds(), log=False)
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def test_resume_is_bit_identical(tmp_path):
    """Training 20 steps straight == training 10 steps, stopping, and resuming to 20."""
    m_full, l_full = train(cfg(tmp_path / "a", resume=False), ds(), log=False)

    train(cfg(tmp_path / "b", checkpoint_interval=7), ds(), log=False, stop_after=10)  # "session 1" dies at step 10
    m_res, l_res = train(cfg(tmp_path / "b"), ds(), log=False)  # "session 2" resumes

    assert len(l_res) == 10  # only steps 10..19 ran
    assert l_res == l_full[10:]
    for p1, p2 in zip(m_full.parameters(), m_res.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_resume_disabled_starts_fresh(tmp_path):
    train(cfg(tmp_path, max_steps=10), ds(), log=False)
    _, losses = train(cfg(tmp_path, max_steps=10, resume=False), ds(), log=False)
    assert len(losses) == 10


def test_checkpoint_files_and_log(tmp_path):
    train(cfg(tmp_path, max_steps=10, checkpoint_steps=[3], eval_interval=5), ds(), log=True)
    assert (tmp_path / "latest.pt").exists() and (tmp_path / "ckpt_000003.pt").exists()
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) >= 2


def test_amp_dtype_cpu_is_none():
    assert amp_dtype("cpu", True) is None


def test_time_budget_stops_and_checkpoints(tmp_path):
    _, losses = train(cfg(tmp_path, max_steps=1000, max_train_hours=0.0), ds(), log=False)
    assert len(losses) == 1  # stops right after the first step
    assert (tmp_path / "latest.pt").exists()
