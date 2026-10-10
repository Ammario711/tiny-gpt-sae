# tiny-gpt-sae

Training a small GPT from scratch on [TinyStories](https://arxiv.org/abs/2305.07759), then using **sparse autoencoders (SAEs)** to find interpretable features in its activations and to study **when those features emerge during training**.

> Status: **Milestone 1 (model + training pipeline) complete.** No trained models or results yet. This README will only report results once they've been measured.

## Research questions
1. Which interpretable features appear in the residual stream of a ~20M-parameter language model, and in which layers?
2. How do features emerge and change across training checkpoints?
3. Do SAE features causally control behaviour (steering and ablation)?

## Repository layout
```
src/tiny_gpt_sae/
  model.py    # GPT: config dataclass, causal self-attention, MLP, pre-LN blocks, weight tying
  data.py     # GPT-2 BPE tokenizer wrapper, TinyStories -> uint16 .bin, random-window batches
  train.py    # AdamW, warmup + cosine LR, grad clipping, eval, checkpoints at chosen steps
  utils.py    # seeding, device selection
configs/      # YAML training configs (smoke run, ~20M model)
scripts/      # data preparation
tests/        # shapes, causal masking, overfitting, determinism, LR schedule, config parsing
```

## Quickstart
```bash
pip install torch            # or the CPU/CUDA wheel for your machine
pip install -e ".[data,dev]"
pytest -q                    # all tests run on CPU in under a minute

# Small data subset + CPU smoke run
python scripts/prepare_data.py --out data/tinystories_small --max-stories 20000
python -m tiny_gpt_sae.train --config configs/smoke.yaml

# Full run (single GPU)
python scripts/prepare_data.py --out data/tinystories
python -m tiny_gpt_sae.train --config configs/tinystories_20m.yaml
```

## Roadmap
- [x] **M1: Model + pipeline.** GPT from scratch, data pipeline, training loop, tests, CI
- [ ] **M2: Pretraining + evals.** Full TinyStories run, val loss curves, samples, checkpoints released on Hugging Face
- [ ] **M3: SAEs.** Activation caching; TopK and ReLU+L1 SAEs; dead-latent handling
- [ ] **M4: SAE evals.** Reconstruction (variance explained), CE loss recovered, L0, dead latents, auto-interpretability
- [ ] **M5: Experiments.** Feature emergence across checkpoints; steering and ablation
- [ ] **M6: Write-up.** Report, feature browser, reproducibility guide

## References
- Eldan & Li (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*
- Elhage et al. (2022). *Toy Models of Superposition.*
- Bricken et al. (2023). *Towards Monosemanticity.*
- Cunningham et al. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models.*
- Gao et al. (2024). *Scaling and Evaluating Sparse Autoencoders.*
- Karpathy, *nanoGPT*, which inspired the structure of the model code.

## License
MIT
