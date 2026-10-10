"""CLI: python -m tiny_gpt_sae.sae.cache --gpt run/latest.pt --data data/tinystories/train.bin --layer 3 \\
        --out cache/layer3 --n-tokens 50000000"""

import argparse

from ..data import TokenDataset
from .activations import cache_activations, load_gpt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-tokens", type=int, default=10_000_000)
    ap.add_argument("--shard-tokens", type=int, default=1_000_000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    model = load_gpt(a.gpt, a.device)
    ds = TokenDataset.from_bin(a.data, model.cfg.context_length)
    shards = cache_activations(model, ds, a.layer, a.out, a.n_tokens, shard_tokens=a.shard_tokens, device=a.device)
    print(f"wrote {len(shards)} shards to {a.out}")
