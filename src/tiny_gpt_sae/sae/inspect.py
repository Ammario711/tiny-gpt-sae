"""Feature inspection: for each feature, find the token contexts where it activates most.

Writes ``features.json`` ({feature: [{activation, tokens, highlight_index}, ...]})
and a self-contained ``features.html`` that shows the activating token in bold.

Usage:
    python -m tiny_gpt_sae.sae.inspect --gpt run/latest.pt --sae runs/sae_layer3/sae.pt \\
        --data data/tinystories/val.bin --layer 3 --out runs/sae_layer3/inspect
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import torch

from ..data import TokenDataset, Tokenizer
from ..model import GPT
from .activations import capture_residual, load_gpt
from .model import SAE, SAEConfig


@torch.no_grad()
def top_activations(
    model: GPT, sae: SAE, layer: int, ds: TokenDataset, n_batches: int, batch_size: int = 16,
    top_n: int = 10, features: list[int] | None = None, seed: int = 0, device: str = "cpu",
) -> dict[int, list[tuple[float, int, int]]]:
    """Return {feature: [(activation, batch_row_id, position), ...]} sorted descending.

    Also stores the token windows in ``top_activations.windows`` so the caller can render contexts.
    """
    feats = torch.tensor(features) if features is not None else torch.arange(sae.cfg.n_features)
    g = torch.Generator().manual_seed(seed)
    best_vals = torch.full((len(feats), top_n), -1.0)
    best_ids = torch.zeros((len(feats), top_n, 2), dtype=torch.long)  # (window id, position)
    windows = []
    for b in range(n_batches):
        x, _ = ds.get_batch(batch_size, g, device)
        windows.append(x.cpu())
        with capture_residual(model, layer) as store:
            model(x)
        f = sae.encode(store[0].float())[..., feats.to(x.device)].cpu()  # (B, T, F)
        B, T, Fn = f.shape
        flat = f.reshape(B * T, Fn).T  # (F, B*T)
        k = min(top_n, B * T)
        vals, pos = flat.topk(k, dim=1)
        ids = torch.stack([b * batch_size + pos // T, pos % T], dim=-1)
        # Merge this batch's top-k with the running top-k.
        all_vals = torch.cat([best_vals, vals], 1)
        all_ids = torch.cat([best_ids, ids], 1)
        keep = all_vals.topk(top_n, dim=1).indices
        best_vals = all_vals.gather(1, keep)
        best_ids = all_ids.gather(1, keep.unsqueeze(-1).expand(-1, -1, 2))
    top_activations.windows = torch.cat(windows)
    result = {}
    for i, feat in enumerate(feats.tolist()):
        result[feat] = [(v, int(w), int(p)) for v, (w, p) in zip(best_vals[i].tolist(), best_ids[i].tolist(),
                                                                  strict=True) if v > 0]
    return result


def render(top: dict, windows: torch.Tensor, decode, out_dir: str | Path, context: int = 12) -> None:
    """Write features.json and features.html with ``context`` tokens before each activating token."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data, rows = {}, []
    for feat, hits in top.items():
        entries = []
        for act, w, p in hits:
            toks = windows[w, max(0, p - context) : p + 4].tolist()
            hi = min(p, context)
            pieces = [decode([t]) for t in toks]
            entries.append({"activation": round(act, 4), "tokens": pieces, "highlight_index": hi})
        data[str(feat)] = entries
        if entries:
            rows.append(f"<h3>Feature {feat}</h3><ul>")
            for e in entries:
                text = "".join(
                    f"<b style='background:#fd8'>{html.escape(t)}</b>" if j == e["highlight_index"] else html.escape(t)
                    for j, t in enumerate(e["tokens"])
                )
                rows.append(f"<li><code>{e['activation']:.3f}</code> {text}</li>")
            rows.append("</ul>")
    (out / "features.json").write_text(json.dumps(data, indent=1))
    page = "<html><head><meta charset='utf-8'><title>SAE features</title></head><body style='font-family:sans-serif'>"
    (out / "features.html").write_text(page + "\n".join(rows) + "</body></html>")


def load_sae(path: str | Path, device: str = "cpu") -> SAE:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sae = SAE(SAEConfig(**ckpt["config"]["sae"])).to(device)
    sae.load_state_dict(ckpt["sae"])
    return sae.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpt", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-batches", type=int, default=50)
    ap.add_argument("--features", type=int, nargs="*", help="subset of features (default: first 200)")
    a = ap.parse_args()
    model, sae = load_gpt(a.gpt), load_sae(a.sae)
    ds = TokenDataset.from_bin(a.data, model.cfg.context_length)
    feats = a.features or list(range(min(200, sae.cfg.n_features)))
    top = top_activations(model, sae, a.layer, ds, a.n_batches, features=feats)
    render(top, top_activations.windows, Tokenizer().decode, a.out)
    print(f"wrote {a.out}/features.json and features.html")


if __name__ == "__main__":
    main()
