"""Sparse autoencoders (SAEs) for interpreting the GPT's residual stream.

Modules:
    activations: hook a layer and cache residual-stream activations to sharded files
    model:       the SAE itself (ReLU + L1, or TopK)
    train:       SAE training loop with dead-feature resampling
    evals:       L0, variance explained, downstream loss with SAE splicing, dead/dense counts
    inspect:     top-activating token contexts per feature -> JSON / HTML
"""

from .model import SAE, SAEConfig

__all__ = ["SAE", "SAEConfig"]
