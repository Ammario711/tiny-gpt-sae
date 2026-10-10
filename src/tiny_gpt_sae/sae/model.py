"""Sparse autoencoder.

An SAE rewrites a d_in-dim activation x as a sparse, non-negative combination
of n_features learned directions ("features"):

    f    = sparsify(W_enc^T (x - b_dec) + b_enc)     # (n_features,) mostly zeros
    xhat = W_dec^T f + b_dec                          # reconstruction of x

Each row of W_dec is a feature's direction in activation space. We hope that
each feature means one human-interpretable thing.

Two ways to make f sparse:
  * "relu": f = ReLU(pre), with an L1 penalty on f added to the loss.
  * "topk": keep only the k largest pre-activations (after ReLU); no L1 needed,
    and L0 is exactly <= k (Gao et al. 2024).

Decoder-norm constraint: every row of W_dec is kept at unit L2 norm. Without it,
the model could cheat the L1 penalty by shrinking f and growing W_dec.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SAEConfig:
    d_in: int  # width of the activations (the GPT's d_model)
    n_features: int  # dictionary size, usually 8x-32x d_in
    kind: str = "relu"  # "relu" (L1 penalty) or "topk"
    k: int = 32  # active features per input (topk only)
    l1_coeff: float = 1e-3  # sparsity strength (relu only)
    tied: bool = False  # if True, the decoder is the transpose of the encoder

    def __post_init__(self) -> None:
        if self.kind not in ("relu", "topk"):
            raise ValueError("kind must be 'relu' or 'topk'")
        if self.kind == "topk" and not 0 < self.k <= self.n_features:
            raise ValueError("need 0 < k <= n_features")


class SAE(nn.Module):
    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, n = cfg.d_in, cfg.n_features
        # Initialize decoder rows as random unit vectors, and the encoder as the transpose
        # (a common init that starts training from a sensible point).
        w_dec = torch.randn(n, d)
        w_dec = w_dec / w_dec.norm(dim=1, keepdim=True)
        self.W_enc = nn.Parameter(w_dec.T.clone())  # (d, n)
        self.b_enc = nn.Parameter(torch.zeros(n))
        self.b_dec = nn.Parameter(torch.zeros(d))  # also subtracted before encoding (centres the data)
        self._W_dec = None if cfg.tied else nn.Parameter(w_dec)  # (n, d)

    @property
    def W_dec(self) -> torch.Tensor:
        """(n_features, d_in) decoder matrix; each row is a feature direction."""
        if self._W_dec is None:
            # Tied: decoder = encoder^T, normalized so the unit-norm constraint holds by construction.
            return (self.W_enc / self.W_enc.norm(dim=0, keepdim=True).clamp_min(1e-8)).T
        return self._W_dec

    def pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_in) -> sparse feature activations f: (..., n_features)."""
        pre = self.pre_activations(x)
        if self.cfg.kind == "relu":
            return torch.relu(pre)
        # TopK: keep the k largest values, zero the rest. ReLU so activations stay >= 0.
        vals, idx = pre.topk(self.cfg.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, idx, torch.relu(vals))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return self.decode(f), f

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Reconstruction MSE (+ L1 penalty for relu). Returns (loss, logging dict)."""
        xhat, f = self(x)
        mse = (xhat - x).pow(2).sum(-1).mean()  # summed over dims, averaged over batch
        loss = mse
        l1 = f.abs().sum(-1).mean()
        if self.cfg.kind == "relu":
            loss = mse + self.cfg.l1_coeff * l1
        l0 = (f > 0).float().sum(-1).mean()
        return loss, {"loss": loss.item(), "mse": mse.item(), "l1": l1.item(), "l0": l0.item()}

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Project decoder rows back to unit norm. Call after every optimizer step."""
        if self._W_dec is not None:
            self._W_dec.data /= self._W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def remove_parallel_decoder_grad(self) -> None:
        """Drop the gradient component that would change decoder-row norms.

        Since rows get re-normalized anyway, this part of the gradient is wasted
        and interacts badly with Adam. Call between backward() and step().
        """
        if self._W_dec is None or self._W_dec.grad is None:
            return
        w, g = self._W_dec.data, self._W_dec.grad
        g -= (g * w).sum(dim=1, keepdim=True) * w
