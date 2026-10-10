"""Auto-label every SAE feature with an open instruct LLM, then score the labels on held-out contexts.

Writes results incrementally to labels.jsonl (one line per feature), so a run that hits the
time limit still leaves usable partial output, and a re-run skips features already done.

Usage:
    python scripts/autointerp.py --gpt gpt.pt --sae sae.pt --data-dir data/tinystories --out runs/autointerp \\
        [--llm Qwen/Qwen2.5-7B-Instruct --4bit]
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from tiny_gpt_sae.data import TokenDataset, Tokenizer
from tiny_gpt_sae.sae.activations import capture_residual, load_gpt
from tiny_gpt_sae.sae.autointerp import (
    LLMJudge,
    auroc,
    balanced_accuracy,
    clean_explanation,
    explain_prompt,
    mark_context,
    score_prompt,
    split_examples,
)
from tiny_gpt_sae.sae.inspect import load_sae, top_activations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--fallback-llm", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n-batches", type=int, default=400, help="val batches (x32 windows) to search for top contexts")
    ap.add_argument("--n-top", type=int, default=20)
    ap.add_argument("--n-explain", type=int, default=10)
    ap.add_argument("--n-score", type=int, default=6, help="held-out positives (and as many negatives) per feature")
    ap.add_argument("--max-hours", type=float, default=10.5)
    ap.add_argument("--limit", type=int, default=None, help="only the first N features (debugging)")
    a = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, sae = load_gpt(a.gpt, device), load_sae(a.sae, device)
    tok = Tokenizer()
    ds = TokenDataset.from_bin(Path(a.data_dir) / "val.bin", model.cfg.context_length)
    n_feat = a.limit or sae.cfg.n_features
    feats = list(range(n_feat))

    # 1. Top contexts for every feature.
    top = top_activations(model, sae, a.layer, ds, a.n_batches, batch_size=32, top_n=a.n_top, features=feats,
                          device=device)
    windows = top_activations.windows
    print(f"[timing] top activations: {time.time() - t0:.0f}s", flush=True)

    # 2. Pool of random positions with their active feature sets, for negatives.
    rnd = random.Random(0)
    pool = []
    with torch.no_grad():
        for start in range(0, min(256, len(windows)), 32):
            x = windows[start : start + 32].to(device)
            with capture_residual(model, a.layer) as store:
                model(x)
            f = sae.encode(store[0].float())
            for _ in range(500):
                r, p = rnd.randrange(len(x)), rnd.randrange(8, x.shape[1] - 4)
                pool.append((start + r, p, set(f[r, p].nonzero().squeeze(1).tolist())))

    def context(w: int, p: int, before: int = 24, after: int = 4) -> str:
        ids = windows[w, max(0, p - before) : p + 1 + after].tolist()
        return mark_context([tok.decode([i]) for i in ids], min(p, before))

    # 3. Explain + score, in chunks, appending to labels.jsonl.
    done = set()
    path = out / "labels.jsonl"
    if path.exists():
        done = {json.loads(line)["feature"] for line in path.open()}
    try:
        judge = LLMJudge(a.llm, load_in_4bit=a.four_bit, device=device)
        llm_name = a.llm + (" (4-bit)" if a.four_bit else "")
    except Exception as e:  # e.g. bitsandbytes unavailable or out of memory
        print(f"could not load {a.llm}: {e!r}; falling back to {a.fallback_llm}", flush=True)
        judge, llm_name = LLMJudge(a.fallback_llm, device=device), a.fallback_llm
    (out / "run_info.json").write_text(json.dumps({"llm": llm_name, **vars(a)}, indent=1))
    print(f"[timing] LLM loaded ({llm_name}): {time.time() - t0:.0f}s", flush=True)

    explain_idx, held_idx = split_examples(a.n_top, a.n_explain)
    todo = [f for f in feats if f not in done]
    chunk = 64
    with path.open("a") as fh:
        for c in range(0, len(todo), chunk):
            if time.time() - t0 > a.max_hours * 3600:
                print("time budget reached; stopping (partial results saved)", flush=True)
                break
            batch = todo[c : c + chunk]
            usable = [f for f in batch if len(top[f]) >= a.n_explain + a.n_score]
            for f in batch:
                if f not in usable:
                    fh.write(json.dumps({"feature": f, "explanation": None, "reason": "too few activations",
                                         "n_activations_found": len(top[f])}) + "\n")
            if not usable:
                continue
            convs = [explain_prompt([context(top[f][i][1], top[f][i][2]) for i in explain_idx]) for f in usable]
            exps = []
            for i in range(0, len(convs), 16):
                exps += [clean_explanation(t) for t in judge.generate(convs[i : i + 16])]
            # Scoring prompts: held-out positives + random negatives where the feature is inactive.
            items = []
            for f, e in zip(usable, exps, strict=True):
                pos = [top[f][i] for i in held_idx[: a.n_score]]
                negs = rnd.sample([q for q in pool if f not in q[2]], a.n_score)
                items += [(f, e, context(w, p), 1) for _, w, p in pos] + [(f, e, context(w, p), 0) for w, p, _ in negs]
            probs = []
            for i in range(0, len(items), 48):
                probs += judge.p_yes([score_prompt(e, ctx) for _, e, ctx, _ in items[i : i + 48]])
            for f, e in zip(usable, exps, strict=True):
                s = [p for (ff, *_), p in zip(items, probs, strict=True) if ff == f]
                y = [lab for (ff, _, _, lab) in items if ff == f]
                fh.write(json.dumps({
                    "feature": f, "explanation": e, "auroc": auroc(s, y), "balanced_acc": balanced_accuracy(s, y),
                    "max_activation": top[f][0][0], "examples": [context(top[f][i][1], top[f][i][2]) for i in explain_idx[:3]],
                }) + "\n")
            fh.flush()
            print(f"[progress] {c + len(batch)}/{len(todo)} features, {time.time() - t0:.0f}s", flush=True)

    # 4. Summary.
    rows = [json.loads(line) for line in path.open()]
    scored = [r for r in rows if r.get("explanation") and r.get("auroc") == r.get("auroc")]
    au = sorted(r["auroc"] for r in scored)
    summary = {"llm": llm_name, "n_features_total": n_feat, "n_labelled": len(rows), "n_scored": len(scored),
               "n_unclear": sum(1 for r in scored if r["explanation"].lower().startswith("unclear")),
               "auroc_median": au[len(au) // 2] if au else None,
               "frac_auroc_ge_0.8": sum(x >= 0.8 for x in au) / max(1, len(au)),
               "frac_balanced_acc_ge_0.8": sum(r["balanced_acc"] >= 0.8 for r in scored) / max(1, len(scored)),
               "hours": (time.time() - t0) / 3600}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
