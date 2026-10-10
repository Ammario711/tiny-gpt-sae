"""Training loop: AdamW, linear warmup + cosine decay, grad clipping, periodic eval + checkpoints.

Usage:
    python -m tiny_gpt_sae.train --config configs/tinystories_20m.yaml
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch
import yaml

from .data import TokenDataset
from .model import GPT, GPTConfig
from .utils import pick_device, set_seed


@dataclass
class TrainConfig:
    model: GPTConfig = field(default_factory=GPTConfig)
    data_dir: str = "data/tinystories"
    out_dir: str = "runs/default"
    seed: int = 1337
    device: str = "auto"
    batch_size: int = 32
    max_steps: int = 5000
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 20
    checkpoint_steps: list[int] = field(default_factory=list)  # extra checkpoints (for training-dynamics study)
    mixed_precision: bool = True  # bf16 autocast on CUDA

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text())
        model = GPTConfig(**raw.pop("model", {}))
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(model=model, **raw)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to ``lr``, then cosine decay down to ``min_lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW; weight decay only on matrices (not biases, LayerNorm or embeddings' 1-D params)."""
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model: GPT, ds: TokenDataset, cfg: TrainConfig, device: str) -> float:
    model.eval()
    g = torch.Generator().manual_seed(0)  # same eval batches every time
    losses = [model(*ds.get_batch(cfg.batch_size, g, device))[1].item() for _ in range(cfg.eval_batches)]
    model.train()
    return sum(losses) / len(losses)


def train(
    cfg: TrainConfig, train_ds: TokenDataset, val_ds: TokenDataset | None = None, log: bool = True
) -> tuple[GPT, list[float]]:
    """Train a model and return it together with the per-step training losses."""
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    model = GPT(cfg.model).to(device)
    opt = make_optimizer(model, cfg)
    g = torch.Generator().manual_seed(cfg.seed)  # controls batch sampling
    use_amp = cfg.mixed_precision and device == "cuda"
    out = Path(cfg.out_dir)
    if log:
        out.mkdir(parents=True, exist_ok=True)
        print(f"device={device} params={model.num_params() / 1e6:.2f}M")

    losses: list[float] = []
    t0 = time.time()
    for step in range(cfg.max_steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        x, y = train_ds.get_batch(cfg.batch_size, g, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        losses.append(loss.item())

        if log and (step % cfg.eval_interval == 0 or step == cfg.max_steps - 1):
            msg = f"step {step:6d} | train {loss.item():.4f} | lr {lr_at(step, cfg):.2e} | {time.time() - t0:.0f}s"
            if val_ds is not None:
                msg += f" | val {estimate_loss(model, val_ds, cfg, device):.4f}"
            print(msg)
        if log and (step in cfg.checkpoint_steps or step == cfg.max_steps - 1):
            save_checkpoint(model, cfg, step, out / f"ckpt_{step:06d}.pt")
    return model, losses


def save_checkpoint(model: GPT, cfg: TrainConfig, step: int, path: Path) -> None:
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step}, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    T = cfg.model.context_length
    train_ds = TokenDataset.from_bin(Path(cfg.data_dir) / "train.bin", T)
    val_path = Path(cfg.data_dir) / "val.bin"
    val_ds = TokenDataset.from_bin(val_path, T) if val_path.exists() else None
    train(cfg, train_ds, val_ds)


if __name__ == "__main__":
    main()
