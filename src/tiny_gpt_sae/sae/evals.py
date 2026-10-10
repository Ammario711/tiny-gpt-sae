"""SAE evaluation metrics.

* L0: average number of active features per token (lower = sparser).
* Fraction of variance explained (FVE): 1 - ||x - xhat||^2 / ||x - mean(x)||^2.
* Downstream loss: splice the SAE reconstruction into the GPT's forward pass
  in place of the real residual stream, and measure next-token cross-entropy.
  We report it against two references:
      clean CE      (no intervention; the best you can do)
      zero-ablation (residual stream replaced by zeros; a bad baseline)
  loss_recovered = (CE_zero - CE_sae) / (CE_zero - CE_clean); 1.0 = perfect.
* Dead features never fire on the eval set; dense features fire on > ``dense_threshold`` of tokens.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from ..model import GPT
from .model import SAE


@contextmanager
def replace_residual(model: GPT, layer: int, fn: Callable[[torch.Tensor], torch.Tensor]):
    """While active, blocks[layer]'s output is replaced by fn(output) for the rest of the forward pass.

    A forward hook that *returns* a value overrides the module's output.
    """
    handle = model.blocks[layer].register_forward_hook(lambda _m, _i, out: fn(out))
    try:
        yield
    finally:
        handle.remove()


def ce_loss(model: GPT, x: torch.Tensor, y: torch.Tensor) -> float:
    logits, _ = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()


@torch.no_grad()
def downstream_loss(model: GPT, sae: SAE, layer: int, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> dict:
    """Clean vs. SAE-spliced vs. zero-ablated cross-entropy, averaged over batches."""
    model.eval()
    clean, spliced, zero = [], [], []
    for x, y in batches:
        clean.append(ce_loss(model, x, y))
        with replace_residual(model, layer, lambda a: sae(a)[0].to(a.dtype)):
            spliced.append(ce_loss(model, x, y))
        with replace_residual(model, layer, torch.zeros_like):
            zero.append(ce_loss(model, x, y))
    c, s, z = (sum(v) / len(v) for v in (clean, spliced, zero))
    # If zero-ablation barely hurts (e.g. an untrained model), the ratio is meaningless: report NaN.
    recovered = (z - s) / (z - c) if z - c > 1e-6 else float("nan")
    return {"ce_clean": c, "ce_sae": s, "ce_zero": z, "loss_recovered": recovered}


@torch.no_grad()
def reconstruction_metrics(sae: SAE, acts: torch.Tensor, dense_threshold: float = 0.1, chunk: int = 8192) -> dict:
    """acts: (N, d_in) held-out activations. Processed in chunks so wide SAEs fit in GPU memory."""
    mean = acts.mean(0)
    resid = total = l0_sum = 0.0
    fire_count = torch.zeros(sae.cfg.n_features, device=acts.device)
    for x in acts.split(chunk):
        xhat, f = sae(x)
        resid += (x - xhat).pow(2).sum().item()
        total += (x - mean).pow(2).sum().item()
        active = f > 0
        l0_sum += active.sum().item()
        fire_count += active.sum(0)
    n = len(acts)
    fire_freq = fire_count / n  # (n_features,) fraction of tokens each feature fires on
    return {
        "l0": l0_sum / n,
        "fve": 1 - resid / total,
        "mse": resid / n,
        "n_dead": int((fire_freq == 0).sum()),
        "n_dense": int((fire_freq > dense_threshold).sum()),
        "fire_freq": fire_freq,
    }

def main() -> None:
    """CLI: python -m tiny_gpt_sae.sae.evals --gpt run/latest.pt --sae runs/sae_layer3/sae.pt \\
    --data data/tinystories/val.bin --layer 3 [--out evals.json]"""
    import argparse
    import json

    from ..data import TokenDataset
    from .activations import capture_residual, load_gpt
    from .inspect import load_sae

    ap = argparse.ArgumentParser()
    for a in ("--gpt", "--sae", "--data"):
        ap.add_argument(a, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out")
    a = ap.parse_args()
    model, sae = load_gpt(a.gpt), load_sae(a.sae)
    ds = TokenDataset.from_bin(a.data, model.cfg.context_length)
    g = torch.Generator().manual_seed(123)
    batches = [ds.get_batch(a.batch_size, g) for _ in range(a.n_batches)]
    with torch.no_grad(), capture_residual(model, a.layer) as store:
        for x, _ in batches:
            model(x)
    acts = torch.cat([s.reshape(-1, s.shape[-1]) for s in store])
    rec = reconstruction_metrics(sae, acts)
    rec.pop("fire_freq")
    results = {**rec, **downstream_loss(model, sae, a.layer, batches)}
    print(json.dumps(results, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
