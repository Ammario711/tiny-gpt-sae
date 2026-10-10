"""Train an SAE on cached activations.

Usage:
    python -m tiny_gpt_sae.sae.train --config configs/sae_layer3.yaml

Dead-feature resampling (after Bricken et al. 2023, simplified): every
``resample_every`` steps, features that haven't fired in that window are
re-initialized to point at inputs the SAE currently reconstructs badly
(sampled with probability proportional to squared error). Their Adam
state is reset so stale momentum doesn't immediately undo the change.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from .activations import ActivationStore
from .model import SAE, SAEConfig


@dataclass
class SAETrainConfig:
    sae: SAEConfig
    cache_dir: str = "cache/layer3"
    out_dir: str = "runs/sae_layer3"
    seed: int = 0
    device: str = "cpu"
    batch_size: int = 4096
    steps: int = 20_000
    lr: float = 3e-4
    l1_warmup_steps: int = 1000  # ramp the L1 coefficient from 0 to avoid killing features early
    resample_every: int = 2500  # 0 disables resampling
    resample_until: int = 15_000  # stop resampling late in training
    log_every: int = 100

    @classmethod
    def from_yaml(cls, path: str | Path) -> SAETrainConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(sae=SAEConfig(**raw.pop("sae")), **raw)


@torch.no_grad()
def resample_dead(sae: SAE, opt: torch.optim.Optimizer, dead: torch.Tensor, x: torch.Tensor) -> int:
    """Re-initialize the features where ``dead`` is True using high-error inputs from batch x."""
    idx = dead.nonzero().squeeze(-1)
    if len(idx) == 0:
        return 0
    err = (sae(x)[0] - x).pow(2).sum(-1)
    probs = err.pow(2)
    if probs.sum() <= 0:
        return 0
    picks = torch.multinomial(probs / probs.sum(), len(idx), replacement=True)
    dirs = x[picks] - sae.b_dec
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
    alive = ~dead
    # Scale new encoder columns to a fraction of the typical alive encoder norm (keeps them from dominating).
    enc_norm = sae.W_enc[:, alive].norm(dim=0).mean() if alive.any() else torch.tensor(1.0)
    sae.W_enc[:, idx] = dirs.T * enc_norm * 0.2
    sae.b_enc[idx] = 0.0
    if sae._W_dec is not None:
        sae._W_dec[idx] = dirs
    # Reset Adam moments for the touched slices.
    for p, sl in ((sae.W_enc, (slice(None), idx)), (sae.b_enc, (idx,))) + (
        ((sae._W_dec, (idx,)),) if sae._W_dec is not None else ()
    ):
        state = opt.state.get(p, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                state[key][sl] = 0.0
    return len(idx)


def train_sae(cfg: SAETrainConfig, batches=None, log: bool = True) -> tuple[SAE, list[dict]]:
    """Train an SAE. ``batches`` is any iterator of (batch, d_in) tensors; defaults to the activation cache."""
    torch.manual_seed(cfg.seed)
    sae = SAE(cfg.sae).to(cfg.device)
    if batches is None:
        batches = iter(ActivationStore(cfg.cache_dir, cfg.batch_size, cfg.seed, cfg.device))
    batches = iter(batches)
    # Initialize b_dec to the data mean (approximately centres the inputs).
    first = next(batches)
    with torch.no_grad():
        sae.b_dec.copy_(first.mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    fired = torch.zeros(cfg.sae.n_features, dtype=torch.bool, device=cfg.device)
    history: list[dict] = []
    base_l1 = cfg.sae.l1_coeff

    for step in range(cfg.steps):
        x = first if step == 0 else next(batches)
        sae.cfg.l1_coeff = base_l1 * min(1.0, (step + 1) / max(1, cfg.l1_warmup_steps))
        loss, stats = sae.loss(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.remove_parallel_decoder_grad()
        opt.step()
        sae.normalize_decoder()

        with torch.no_grad():
            fired |= (sae.encode(x) > 0).any(0)
        if cfg.resample_every and (step + 1) % cfg.resample_every == 0 and step < cfg.resample_until:
            n = resample_dead(sae, opt, ~fired, x)
            stats["resampled"] = n
            fired.zero_()
        stats["step"] = step
        history.append(stats)
        if log and step % cfg.log_every == 0:
            print(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in stats.items()))
    sae.cfg.l1_coeff = base_l1
    return sae, history


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    cfg = SAETrainConfig.from_yaml(ap.parse_args().config)
    sae, history = train_sae(cfg)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"sae": sae.state_dict(), "config": asdict(cfg)}, out / "sae.pt")
    with open(out / "history.jsonl", "w") as f:
        f.writelines(json.dumps(h) + "\n" for h in history)
    print(f"saved {out / 'sae.pt'}")


if __name__ == "__main__":
    main()
