"""Feature emergence: train an SAE on each training snapshot and match its features to the final SAE.

For each snapshot (and, as a baseline, the final model again with a different seed and the
same short recipe), cache layer activations, train a 16x TopK SAE, and compute for every
reference feature the best decoder cosine similarity. A feature counts as matched at >= --threshold.

Usage:
    python scripts/emergence.py --snapshots snap_*.pt --final gpt_step15000.pt --ref-sae sae.pt \\
        --data-dir data/tinystories --work /tmp/em --out runs/emergence
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch

from tiny_gpt_sae.data import TokenDataset
from tiny_gpt_sae.sae import SAEConfig
from tiny_gpt_sae.sae.activations import cache_activations, capture_residual, load_gpt
from tiny_gpt_sae.sae.emergence import first_appearance, max_cosine
from tiny_gpt_sae.sae.evals import reconstruction_metrics
from tiny_gpt_sae.sae.inspect import load_sae
from tiny_gpt_sae.sae.train import SAETrainConfig, train_sae

REVIEWED = {1: '" to" (the X thing to)', 4: '" to" (use it to)', 14: '" to" (way to)', 29: "Once (story start)",
            8: "Suddenly ... heard", 19: "learned that even", 22: "angry quote", 26: "deep breath",
            16: "became friends. They", 21: "small town", 30: "Jack"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", nargs="+", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--ref-sae", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--n-tokens", type=int, default=4_000_000)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--threshold", type=float, default=0.7)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = load_sae(a.ref_sae, device)
    ref_dec = ref.W_dec.detach()
    runs = sorted((int(re.search(r"(\d+)\.pt$", p).group(1)), p) for p in a.snapshots)
    runs.append((15000, a.final))  # same recipe on the final model = seed-to-seed baseline
    summary, cos_rows = [], []
    for step, path in runs:
        model = load_gpt(path, device)
        T = model.cfg.context_length
        cache = Path(a.work) / f"step{step}"
        cache_activations(model, TokenDataset.from_bin(Path(a.data_dir) / "train.bin", T), a.layer, cache,
                          a.n_tokens, batch_size=64, device=device, seed=step)
        cfg = SAETrainConfig(sae=SAEConfig(d_in=ref.cfg.d_in, n_features=ref.cfg.n_features, kind="topk", k=ref.cfg.k),
                             cache_dir=str(cache), device=device, batch_size=4096, steps=a.steps, lr=3e-4, seed=1,
                             resample_every=500, resample_until=int(a.steps * 0.75), log_every=10_000)
        sae, _ = train_sae(cfg, log=False)
        shutil.rmtree(cache)
        # Held-out FVE on val (so a poor match isn't just a poorly trained SAE).
        g = torch.Generator().manual_seed(123)
        val = TokenDataset.from_bin(Path(a.data_dir) / "val.bin", T)
        with torch.no_grad(), capture_residual(model, a.layer) as store:
            for _ in range(8):
                model(val.get_batch(32, g, device)[0])
        acts = torch.cat([s.reshape(-1, s.shape[-1]) for s in store]).float()
        rec = reconstruction_metrics(sae, acts)
        rec.pop("fire_freq")
        cos, _ = max_cosine(ref_dec, sae.W_dec.detach())
        cos_rows.append(cos.cpu())
        torch.save(sae.W_dec.detach().cpu().half(), out / f"W_dec_step{step}.pt")
        row = {"step": step, "baseline_final_model": path == a.final, "fve": rec["fve"], "n_dead": rec["n_dead"],
               "frac_matched": (cos >= a.threshold).float().mean().item(), "median_cos": cos.median().item(),
               "reviewed_cos": {f: round(cos[f].item(), 3) for f in REVIEWED}}
        print(json.dumps(row), flush=True)
        summary.append(row)
        del model, sae

    snap = [r for r in summary if not r["baseline_final_model"]]
    steps = [r["step"] for r in snap]
    cos_snap = torch.stack(cos_rows[: len(snap)])
    # Append the final-model baseline as the last "step" for stable-appearance computation.
    firsts = first_appearance(steps + [15000], torch.cat([cos_snap, cos_rows[-1][None]]), a.threshold)
    torch.save({"steps": steps + [15000], "cos": torch.cat([cos_snap, cos_rows[-1][None]])}, out / "cos_matrix.pt")
    result = {"threshold": a.threshold, "runs": summary,
              "reviewed_first_stable_step": {f: firsts[f] for f in REVIEWED},
              "first_stable_step_histogram": {str(s): sum(1 for x in firsts if x == s) for s in steps + [15000, None]}}
    (out / "emergence.json").write_text(json.dumps(result, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [max(s, 1) for s in steps]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(xs, [r["frac_matched"] for r in snap], "o-", label="snapshot SAE vs final SAE")
    axes[0].axhline(summary[-1]["frac_matched"], ls="--", c="gray", label="seed-to-seed baseline (final model)")
    axes[0].set(xscale="log", xlabel="training step", ylabel=f"fraction of features matched (cos ≥ {a.threshold})",
                title="Overall feature emergence (layer 3)")
    for f, name in REVIEWED.items():
        axes[1].plot(xs, [r["reviewed_cos"][f] for r in snap], "o-", ms=3, label=f"F{f} {name}")
    axes[1].axhline(a.threshold, ls="--", c="gray")
    axes[1].set(xscale="log", xlabel="training step", ylabel="best decoder cosine to final feature",
                title="Reviewed features")
    axes[1].legend(fontsize=7, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "emergence.png", dpi=150)


if __name__ == "__main__":
    main()
