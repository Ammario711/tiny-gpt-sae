"""Caching residual-stream activations from a trained GPT.

"Residual stream after layer L" = the output of ``model.blocks[L]``: the running
sum that every later block reads from. We capture it with a PyTorch forward hook,
which runs a function each time a module produces an output.

Activations are written as shards (``acts_00000.pt`` ...), each an fp16 tensor of
shape (shard_tokens, d_model), plus ``meta.json``. Sharding means we never need
all activations in memory at once.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

from ..data import TokenDataset
from ..model import GPT, GPTConfig


def load_gpt(ckpt_path: str | Path, device: str = "cpu") -> GPT:
    """Rebuild a GPT from a training checkpoint (as written by tiny_gpt_sae.train)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]["model"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model


@contextmanager
def capture_residual(model: GPT, layer: int) -> Iterator[list[torch.Tensor]]:
    """Context manager: while active, each forward pass appends blocks[layer]'s output to the list."""
    store: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        store.append(output.detach())

    handle = model.blocks[layer].register_forward_hook(hook)
    try:
        yield store
    finally:
        handle.remove()  # always remove, or the hook would leak into later runs


@torch.no_grad()
def cache_activations(
    model: GPT,
    ds: TokenDataset,
    layer: int,
    out_dir: str | Path,
    n_tokens: int,
    batch_size: int = 32,
    shard_tokens: int = 1_000_000,
    seed: int = 0,
    device: str = "cpu",
) -> list[Path]:
    """Run the model over random windows of ``ds`` and save ~n_tokens activations at ``layer``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    buf: list[torch.Tensor] = []
    buffered, written, shards = 0, 0, []

    def flush() -> None:
        nonlocal buf, buffered
        acts = torch.cat(buf)
        path = out / f"acts_{len(shards):05d}.pt"
        torch.save(acts[:shard_tokens].to(torch.float16).cpu().clone(), path)
        shards.append(path)
        rest = acts[shard_tokens:]
        buf, buffered = ([rest], len(rest)) if len(rest) else ([], 0)

    while written + buffered < n_tokens:
        x, _ = ds.get_batch(batch_size, g, device)
        with capture_residual(model, layer) as store:
            model(x)
        acts = store[0].reshape(-1, store[0].shape[-1])  # (B*T, d_model)
        acts = acts[: n_tokens - written - buffered]
        buf.append(acts)
        buffered += len(acts)
        while buffered >= shard_tokens:
            flush()
            written += shard_tokens
    if buffered:  # last, partial shard
        written += buffered
        flush()

    meta = {"layer": layer, "n_tokens": written, "d_model": model.cfg.d_model, "shards": [p.name for p in shards]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return shards


class ActivationStore:
    """Yields shuffled float32 batches from cached shards, cycling through them forever."""

    def __init__(self, cache_dir: str | Path, batch_size: int, seed: int = 0, device: str = "cpu") -> None:
        self.dir = Path(cache_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.batch_size = batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.device = device

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            order = torch.randperm(len(self.meta["shards"]), generator=self.g).tolist()
            for i in order:
                acts = torch.load(self.dir / self.meta["shards"][i])
                perm = torch.randperm(len(acts), generator=self.g)
                for j in range(0, len(acts) - self.batch_size + 1, self.batch_size):
                    yield acts[perm[j : j + self.batch_size]].float().to(self.device)
