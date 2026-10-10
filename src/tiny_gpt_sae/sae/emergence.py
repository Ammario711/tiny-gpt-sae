"""Feature emergence across training: match SAE features between checkpoints.

For each feature of a reference SAE (trained on the final model), find the most
similar decoder direction in an SAE trained on an earlier checkpoint. A feature
"exists" at that checkpoint if the best cosine similarity exceeds a threshold.

Caveat: two SAEs trained on the *same* model with different seeds don't match
perfectly either, so always compare against that seed-to-seed baseline.
"""

from __future__ import annotations

import torch


def max_cosine(ref_dec: torch.Tensor, other_dec: torch.Tensor, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """For each row of ref_dec (n_ref, d), best cosine similarity with any row of other_dec (n_other, d).

    Returns (best_cos, best_index), each (n_ref,).
    """
    a = ref_dec / ref_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
    b = other_dec / other_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
    vals, idxs = [], []
    for part in a.split(chunk):
        v, i = (part @ b.T).max(dim=1)
        vals.append(v)
        idxs.append(i)
    return torch.cat(vals), torch.cat(idxs)


def first_appearance(steps: list[int], cos: torch.Tensor, threshold: float = 0.7) -> list[int | None]:
    """cos: (n_steps, n_features) best-match cosines over training steps (sorted by step).

    Returns, per feature, the first step from which it stays matched (>= threshold at that
    step and every later one), or None if it never stably appears.
    """
    ok = cos >= threshold
    out: list[int | None] = []
    for j in range(cos.shape[1]):
        first = None
        for s in range(len(steps) - 1, -1, -1):  # walk backwards while still matched
            if ok[s, j]:
                first = steps[s]
            else:
                break
        out.append(first)
    return out
