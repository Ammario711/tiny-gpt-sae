"""Training loop for the GPT.

Features:
- AdamW with linear warmup + cosine decay, gradient clipping
- gradient accumulation (simulate a large batch with several small micro-batches)
- mixed precision on CUDA: bf16 on Ampere+ GPUs, fp16 + GradScaler on older ones (e.g. Colab T4)
- checkpoint/resume: ``latest.pt`` stores model, optimizer, scaler and RNG state,
  so a resumed run continues exactly where it stopped (important on Colab, where sessions end)
- metrics appended to ``log.jsonl`` for plotting

Usage:
    python -m tiny_gpt_sae.train --config configs/colab_t4.yaml [--out-dir DIR] [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
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
    batch_size: int = 32  # micro-batch size (sequences per forward pass)
    grad_accum_steps: int = 1  # effective batch = batch_size * grad_accum_steps
    max_steps: int = 5000  # optimizer steps
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 20
    checkpoint_interval: int = 500  # how often latest.pt is overwritten
    checkpoint_steps: list[int] = field(default_factory=list)  # extra kept snapshots (training-dynamics study)
    mixed_precision: bool = True  # only used on CUDA
    resume: bool = True  # continue from out_dir/latest.pt if it exists
    max_train_hours: float | None = None  # wall-clock budget: checkpoint and stop cleanly (e.g. Kaggle's 12 h limit)

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
    """AdamW; weight decay only on matrices (not biases or LayerNorm gains)."""
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def amp_dtype(device: str, enabled: bool) -> torch.dtype | None:
    """Pick the autocast dtype: bf16 on Ampere+ (compute capability >= 8), fp16 on older GPUs, None on CPU."""
    if not enabled or not device.startswith("cuda"):
        return None
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def accumulate_gradients(model: GPT, micro_batches, autocast_ctx, scaler) -> float:
    """Forward/backward over several micro-batches, summing gradients.

    Each micro-batch loss is divided by the number of micro-batches, so the summed
    gradient equals the gradient of the mean loss over the whole effective batch.
    Returns that mean loss.
    """
    n = len(micro_batches)
    total = 0.0
    for x, y in micro_batches:
        with autocast_ctx():
            _, loss = model(x, y)
        loss = loss / n
        scaler.scale(loss).backward()  # scaler is a no-op when disabled
        total += loss.item()
    return total


@torch.no_grad()
def estimate_loss(model: GPT, ds: TokenDataset, cfg: TrainConfig, device: str, autocast_ctx=nullcontext) -> float:
    model.eval()
    g = torch.Generator().manual_seed(0)  # same eval batches every time
    losses = []
    for _ in range(cfg.eval_batches):
        with autocast_ctx():
            losses.append(model(*ds.get_batch(cfg.batch_size, g, device))[1].item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path: Path, model, opt, scaler, gen: torch.Generator, cfg: TrainConfig, step: int) -> None:
    """Save everything needed to resume. Written to a temp file first so a crash can't corrupt it."""
    state = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "batch_rng": gen.get_state(),
        "torch_rng": torch.get_rng_state(),
        "config": asdict(cfg),
        "step": step,  # number of optimizer steps completed
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(
    cfg: TrainConfig,
    train_ds: TokenDataset,
    val_ds: TokenDataset | None = None,
    log: bool = True,
    stop_after: int | None = None,
) -> tuple[GPT, list[float]]:
    """Train (or resume) and return the model and the training losses from *this* call.

    ``stop_after`` ends this call early after that many total steps (saving a checkpoint),
    without changing the LR schedule. This simulates a Colab session ending.
    """
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    model = GPT(cfg.model).to(device)
    opt = make_optimizer(model, cfg)
    gen = torch.Generator().manual_seed(cfg.seed)  # controls which batches are sampled

    dtype = amp_dtype(device, cfg.mixed_precision)
    def autocast_ctx():
        return torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()
    # GradScaler prevents fp16 gradients underflowing to zero; not needed for bf16/fp32.
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest.pt"
    start = 0
    if cfg.resume and latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        gen.set_state(ckpt["batch_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        start = ckpt["step"]
        if log:
            print(f"resumed from {latest} at step {start}")
    if log:
        eff = cfg.batch_size * cfg.grad_accum_steps
        print(f"device={device} amp={dtype} params={model.num_params() / 1e6:.2f}M effective_batch={eff}")

    losses: list[float] = []
    t0 = time.time()
    for step in range(start, cfg.max_steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        micro = [train_ds.get_batch(cfg.batch_size, gen, device) for _ in range(cfg.grad_accum_steps)]
        opt.zero_grad(set_to_none=True)
        loss = accumulate_gradients(model, micro, autocast_ctx, scaler)
        scaler.unscale_(opt)  # clip the real (unscaled) gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        losses.append(loss)
        done = step + 1

        if log and (step % cfg.eval_interval == 0 or done == cfg.max_steps):
            rec = {"step": step, "train_loss": loss, "lr": lr_at(step, cfg), "time_s": round(time.time() - t0, 1)}
            if val_ds is not None:
                rec["val_loss"] = estimate_loss(model, val_ds, cfg, device, autocast_ctx)
            print(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        if step in cfg.checkpoint_steps:
            save_checkpoint(out / f"ckpt_{step:06d}.pt", model, opt, scaler, gen, cfg, done)
        stopping = stop_after is not None and done >= stop_after
        if cfg.max_train_hours is not None and time.time() - t0 > cfg.max_train_hours * 3600:
            stopping = True
            if log:
                print(f"time budget of {cfg.max_train_hours} h reached at step {done}; saving and stopping")
        if done % cfg.checkpoint_interval == 0 or done == cfg.max_steps or stopping:
            save_checkpoint(latest, model, opt, scaler, gen, cfg, done)
        if stopping:
            break
    return model, losses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", help="override out_dir (e.g. a Google Drive folder)")
    ap.add_argument("--data-dir", help="override data_dir")
    ap.add_argument("--max-steps", type=int, help="override max_steps")
    args = ap.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.max_steps:
        cfg.max_steps = args.max_steps
    T = cfg.model.context_length
    train_ds = TokenDataset.from_bin(Path(cfg.data_dir) / "train.bin", T)
    val_path = Path(cfg.data_dir) / "val.bin"
    val_ds = TokenDataset.from_bin(val_path, T) if val_path.exists() else None
    train(cfg, train_ds, val_ds)


if __name__ == "__main__":
    main()
