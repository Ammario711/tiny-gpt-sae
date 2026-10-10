"""Ablation and steering experiments for selected SAE features.

For each feature:
  * Ablation at positions where it fires (random sample, not just top activations):
    KL(clean || ablated) of the next-token distribution, change in log-prob of the
    actual next token, and the tokens whose probability drops most.
  * Matched control: at the *same* positions, ablate a different feature that is also
    active there with the closest activation. This shows how big a "typical" single-feature
    ablation effect is, so the target feature's effect can be judged against it.
  * Sanity control: positions where the feature is inactive (ablation should do ~nothing).
  * Steering: clamp the feature to 1x and 3x its max observed activation at every position
    while generating, with the same seed as an unsteered baseline.

Usage:
    python scripts/causal_experiments.py --gpt gpt.pt --sae sae.pt --data-dir data/tinystories --out runs/causal
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from tiny_gpt_sae.data import TokenDataset, Tokenizer
from tiny_gpt_sae.sae.activations import capture_residual, load_gpt
from tiny_gpt_sae.sae.causal import ablation_effect, clamp, generate_with
from tiny_gpt_sae.sae.inspect import load_sae

FEATURES = {1: '" to" after "the X thing"', 4: '" to" after "use it"', 14: '" to"/" of" after "way"',
            29: '"Once" starting a new story', 8: '" heard" after "Suddenly, they"',
            19: '" even" after "learned that"', 22: 'opening quote after angry "said,"',
            26: '" breath" in "took a deep breath"'}
PROMPTS = ["Once upon a time", "Tom and his mom went to the store.", "The little cat"]


def nanmean(t: torch.Tensor) -> float:
    t = t[~torch.isnan(t)]
    return t.mean().item() if len(t) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--n-batches", type=int, default=150)
    ap.add_argument("--n-positions", type=int, default=150)
    ap.add_argument("--features", type=int, nargs="*", default=list(FEATURES))
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, sae = load_gpt(a.gpt, device), load_sae(a.sae, device)
    tok = Tokenizer()
    ds = TokenDataset.from_bin(Path(a.data_dir) / "val.bin", model.cfg.context_length)
    g = torch.Generator().manual_seed(7)

    # Find where each feature fires, keeping the full active set (TopK) per position for controls.
    windows, firing = [], {f: [] for f in a.features}
    with torch.no_grad():
        for b in range(a.n_batches):
            x, _ = ds.get_batch(32, g, device)
            windows.append(x.cpu())
            with capture_residual(model, a.layer) as store:
                model(x)
            acts = sae.encode(store[0].float())  # (32, T, n_features)
            for f in a.features:
                rows, pos = (acts[..., f] > 0).nonzero(as_tuple=True)
                for r, p in zip(rows.tolist(), pos.tolist(), strict=True):
                    firing[f].append((b * 32 + r, p, acts[r, p, f].item(), acts[r, p].nonzero().squeeze(1).cpu(),
                                      acts[r, p][acts[r, p] > 0].cpu()))
    windows = torch.cat(windows)

    def ctx(w, p, before=14, after=3):
        ids = windows[w, max(0, p - before): p + 1 + after].tolist()
        pieces = [tok.decode([i]) for i in ids]
        hi = min(p, before)
        return "".join(f"[[{s}]]" if i == hi else s for i, s in enumerate(pieces)).replace("\n", " ")

    rng = torch.Generator().manual_seed(0)
    report = {}
    for f in a.features:
        hits = firing[f]
        if not hits:
            report[f] = {"label": FEATURES.get(f, ""), "n_firing": 0}
            continue
        sel = torch.randperm(len(hits), generator=rng)[: a.n_positions].tolist()
        tgt_kl, tgt_dlp, ctl_kl, ctl_dlp, drops, examples = [], [], [], [], Counter(), []
        for i in sel:
            w, p, act, idx, vals = hits[i]
            x = windows[w : w + 1].to(device)
            pos = torch.tensor([[0, p]], device=device)
            e = ablation_effect(model, sae, a.layer, f, x, pos)
            tgt_kl.append(e["kl"].item())
            tgt_dlp.append(e["dlogp"].item())
            drops[tok.decode([e["top_drop"].item()])] += 1
            # Matched control: another feature active here with the closest activation.
            others = [(abs(v - act), j) for j, v in zip(idx.tolist(), vals.tolist(), strict=True) if j != f]
            if others:
                cj = min(others)[1]
                c = ablation_effect(model, sae, a.layer, cj, x, pos)
                ctl_kl.append(c["kl"].item())
                ctl_dlp.append(c["dlogp"].item())
            if len(examples) < 5:
                examples.append({"context": ctx(w, p), "activation": round(act, 3), "kl": round(e["kl"].item(), 4),
                                 "dlogp_next": round(e["dlogp"].item(), 4),
                                 "most_dropped": tok.decode([e["top_drop"].item()])})
        # Sanity: random inactive positions in the same windows.
        xs = windows[:64].to(device)
        with torch.no_grad(), capture_residual(model, a.layer) as store:
            model(xs)
        inactive = (sae.encode(store[0].float())[..., f] == 0).nonzero()
        inactive = inactive[torch.randperm(len(inactive), generator=rng)[:200].to(inactive.device)]
        z = ablation_effect(model, sae, a.layer, f, xs, inactive)
        t_kl, t_dlp = torch.tensor(tgt_kl), torch.tensor(tgt_dlp)
        report[f] = {
            "label": FEATURES.get(f, ""), "n_firing_found": len(hits), "n_tested": len(sel),
            "max_activation": max(h[2] for h in hits),
            "target": {"kl_mean": t_kl.mean().item(), "kl_median": t_kl.median().item(), "dlogp_next_mean": nanmean(t_dlp)},
            "matched_control": {"kl_mean": torch.tensor(ctl_kl).mean().item() if ctl_kl else None,
                                "kl_median": torch.tensor(ctl_kl).median().item() if ctl_kl else None,
                                "dlogp_next_mean": nanmean(torch.tensor(ctl_dlp)) if ctl_dlp else None},
            "inactive_sanity_kl_max": z["kl"].max().item(),
            "most_dropped_tokens": drops.most_common(8),
            "examples": examples,
        }
        print(json.dumps({k: v for k, v in report[f].items() if k != "examples"}), flush=True)

    # Steering.
    steer = {}
    for f in a.features:
        if not firing[f]:
            continue
        mx = report[f]["max_activation"]
        steer[f] = {}
        for prompt in PROMPTS:
            idx = torch.tensor([tok.encode(prompt)], device=device)
            row = {"baseline": tok.decode(generate_with(model, idx, 60, 0, temperature=0.8, top_k=50)[0].tolist())}
            for mult in (1, 3):
                o = generate_with(model, idx, 60, 0, clamp(model, sae, a.layer, f, mult * mx), temperature=0.8, top_k=50)
                row[f"clamp_{mult}x_max"] = tok.decode(o[0].tolist())
            steer[f][prompt] = row
    (out / "ablation.json").write_text(json.dumps(report, indent=1))
    (out / "steering.json").write_text(json.dumps(steer, indent=1))

    lines = ["| feature | label | n tested | KL target (mean / median) | KL matched control (mean / median) | "
             "Δlogp next (target / control) | top dropped tokens |", "|---|---|---:|---|---|---|---|"]
    for f, r in report.items():
        if "target" not in r:
            lines.append(f"| {f} | {r['label']} | 0 | – | – | – | – |")
            continue
        t, c = r["target"], r["matched_control"]
        drops = ", ".join(f"`{s!r}`×{n}" for s, n in r["most_dropped_tokens"][:3])
        lines.append(f"| {f} | {r['label']} | {r['n_tested']} | {t['kl_mean']:.3f} / {t['kl_median']:.3f} | "
                     f"{c['kl_mean']:.3f} / {c['kl_median']:.3f} | {t['dlogp_next_mean']:+.3f} / "
                     f"{c['dlogp_next_mean']:+.3f} | {drops} |")
    (out / "ablation_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
