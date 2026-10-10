"""End-to-end SAE sweep on a trained GPT checkpoint (designed for one GPU kernel, e.g. Kaggle T4).

Steps:
  1. cache residual-stream activations at --layer from the train split
  2. train one TopK SAE per (expansion, k) pair
  3. evaluate each on held-out val activations: L0, FVE, dead/dense, downstream CE / loss recovered
  4. plot sparsity (L0) vs fidelity (FVE and loss recovered)
  5. dump the feature inspector (JSON + HTML) for one chosen SAE

Usage:
    python scripts/sae_sweep.py --gpt gpt.pt --data-dir data/tinystories --work /tmp/sae --out runs/sae-sweep
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from tiny_gpt_sae.data import TokenDataset, Tokenizer
from tiny_gpt_sae.sae import SAEConfig
from tiny_gpt_sae.sae.activations import cache_activations, capture_residual, load_gpt
from tiny_gpt_sae.sae.evals import downstream_loss, reconstruction_metrics
from tiny_gpt_sae.sae.inspect import render, top_activations
from tiny_gpt_sae.sae.train import SAETrainConfig, train_sae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work", required=True, help="scratch dir for the activation cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--n-tokens", type=int, default=10_000_000)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--expansions", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--inspect", default="16x_k32", help="which SAE to dump the feature inspector for")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_gpt(a.gpt, device)
    d = model.cfg.d_model
    T = model.cfg.context_length
    train_ds = TokenDataset.from_bin(Path(a.data_dir) / "train.bin", T)
    val_ds = TokenDataset.from_bin(Path(a.data_dir) / "val.bin", T)

    # 1. Cache activations.
    t0 = time.time()
    cache = Path(a.work) / f"layer{a.layer}"
    if not (cache / "meta.json").exists():
        cache_activations(model, train_ds, a.layer, cache, a.n_tokens, batch_size=64, device=device)
    print(f"[timing] cache: {time.time() - t0:.0f}s", flush=True)

    # Held-out evaluation data: fixed val batches and their layer activations.
    g = torch.Generator().manual_seed(123)
    val_batches = [val_ds.get_batch(32, g, device) for _ in range(30)]  # ~245k tokens
    with torch.no_grad(), capture_residual(model, a.layer) as store:
        for x, _ in val_batches:
            model(x)
    val_acts = torch.cat([s.reshape(-1, d) for s in store]).float()

    # 2-3. Sweep.
    results, saes = [], {}
    for exp in a.expansions:
        for k in a.ks:
            name = f"{exp}x_k{k}"
            t1 = time.time()
            cfg = SAETrainConfig(sae=SAEConfig(d_in=d, n_features=exp * d, kind="topk", k=k),
                                 cache_dir=str(cache), out_dir=str(out / name), device=device,
                                 batch_size=4096, steps=a.steps, lr=3e-4,
                                 resample_every=1000, resample_until=int(a.steps * 0.75), log_every=1000)
            sae, hist = train_sae(cfg)
            sae.eval()
            rec = reconstruction_metrics(sae, val_acts)
            freq = rec.pop("fire_freq").cpu()
            (out / name).mkdir(parents=True, exist_ok=True)
            torch.save(freq, out / name / "fire_freq.pt")
            res = {"name": name, "expansion": exp, "k": k, "n_features": exp * d, **rec,
                   **downstream_loss(model, sae, a.layer, val_batches),
                   "train_final_mse": hist[-1]["mse"], "train_seconds": round(time.time() - t1)}
            print(json.dumps(res), flush=True)
            results.append(res)
            (out / name).mkdir(parents=True, exist_ok=True)
            torch.save({"sae": sae.state_dict(), "config": asdict(cfg)}, out / name / "sae.pt")
            saes[name] = sae
    (out / "results.json").write_text(json.dumps(results, indent=2))

    # 4. Plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for exp in a.expansions:
        rs = sorted([r for r in results if r["expansion"] == exp], key=lambda r: r["l0"])
        axes[0].plot([r["l0"] for r in rs], [r["fve"] for r in rs], "o-", label=f"{exp}x ({exp * d} features)")
        axes[1].plot([r["l0"] for r in rs], [r["loss_recovered"] for r in rs], "o-", label=f"{exp}x")
    axes[0].set(xlabel="L0 (active features per token)", ylabel="fraction of variance explained",
                title=f"Reconstruction, layer {a.layer}")
    axes[1].set(xlabel="L0 (active features per token)", ylabel="CE loss recovered",
                title="Downstream fidelity")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "sparsity_vs_fidelity.png", dpi=150)

    # 5. Feature inspector for the chosen SAE (first 300 features).
    sae = saes[a.inspect]
    feats = list(range(min(300, sae.cfg.n_features)))
    top = top_activations(model, sae, a.layer, val_ds, n_batches=200, batch_size=32, top_n=12,
                          features=feats, device=device)
    render(top, top_activations.windows, Tokenizer().decode, out / a.inspect / "inspect", context=16)
    # Fire frequency per inspected feature, for picking non-dead, non-dense features to review.
    freq = torch.load(out / a.inspect / "fire_freq.pt")
    (out / a.inspect / "inspect" / "fire_freq.json").write_text(json.dumps({i: freq[i].item() for i in feats}))
    print(f"[timing] total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
