"""Causal tests of SAE features: ablation and steering.

Both work by editing the residual stream at the SAE's layer with a forward hook.
We edit only the feature's own contribution, so the SAE's reconstruction error
(the part of the activation the SAE doesn't explain) is left untouched:

    ablate feature i:  a  ->  a - f_i(a) * d_i              (remove what feature i writes)
    clamp feature i:   a  ->  a + (c - f_i(a)) * d_i        (force its activation to c)

where d_i is feature i's (unit-norm) decoder direction and f_i(a) its activation.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from ..model import GPT
from .evals import replace_residual
from .model import SAE


def ablate_fn(sae: SAE, feature: int):
    def fn(a: torch.Tensor) -> torch.Tensor:
        f = sae.encode(a.float())[..., feature : feature + 1]  # (B, T, 1)
        return (a.float() - f * sae.W_dec[feature]).to(a.dtype)

    return fn


def clamp_fn(sae: SAE, feature: int, value: float):
    def fn(a: torch.Tensor) -> torch.Tensor:
        f = sae.encode(a.float())[..., feature : feature + 1]
        return (a.float() + (value - f) * sae.W_dec[feature]).to(a.dtype)

    return fn


@contextmanager
def ablate(model: GPT, sae: SAE, layer: int, feature: int):
    with replace_residual(model, layer, ablate_fn(sae, feature)):
        yield


@contextmanager
def clamp(model: GPT, sae: SAE, layer: int, feature: int, value: float):
    with replace_residual(model, layer, clamp_fn(sae, feature, value)):
        yield


@torch.no_grad()
def ablation_effect(
    model: GPT, sae: SAE, layer: int, feature: int, x: torch.Tensor, positions: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Effect of ablating ``feature`` on next-token predictions at (row, pos) pairs.

    x: (B, T) token windows. positions: (N, 2) long tensor of (row, position).
    Returns per-position tensors (each (N,)):
      kl         KL(clean || ablated) of the next-token distribution
      dlogp      change in log-prob of the token that actually comes next (negative = ablation hurt)
      top_drop   id of the token whose probability dropped the most
    """
    clean = F.log_softmax(model(x)[0].float(), -1)
    with ablate(model, sae, layer, feature):
        abl = F.log_softmax(model(x)[0].float(), -1)
    r, p = positions[:, 0], positions[:, 1]
    lc, la = clean[r, p], abl[r, p]  # (N, V)
    kl = (lc.exp() * (lc - la)).sum(-1)
    nxt = x[r, (p + 1).clamp(max=x.shape[1] - 1)]
    dlogp = (la - lc).gather(1, nxt[:, None]).squeeze(1)
    valid = p + 1 < x.shape[1]  # the last position has no "actual next token" in the window
    dlogp = torch.where(valid, dlogp, torch.full_like(dlogp, float("nan")))
    top_drop = (lc.exp() - la.exp()).argmax(-1)
    return {"kl": kl, "dlogp": dlogp, "top_drop": top_drop}


@torch.no_grad()
def generate_with(model: GPT, idx: torch.Tensor, n: int, seed: int, ctx=None, **kw) -> torch.Tensor:
    """Generate with a fixed seed, optionally inside an intervention context (so samples are comparable)."""
    torch.manual_seed(seed)
    if ctx is None:
        return model.generate(idx, n, **kw)
    with ctx:
        return model.generate(idx, n, **kw)
