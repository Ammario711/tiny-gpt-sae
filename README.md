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
  train.py    # AdamW, warmup + cosine LR, grad clipping, grad accumulation,
              # bf16/fp16 mixed precision, exact checkpoint resume, log.jsonl
  utils.py    # seeding, device selection
  sae/
    activations.py  # forward hook on blocks[L]; cache residual stream to fp16 shards; shuffled batch store
    model.py        # SAE: ReLU+L1 or TopK, tied/untied decoder, unit-norm decoder rows
    train.py        # SAE training, L1 warmup, dead-feature resampling (python -m tiny_gpt_sae.sae.train)
    evals.py        # L0, variance explained, dead/dense counts, CE loss with SAE spliced in
    inspect.py      # top-activating contexts per feature -> features.json / features.html
    cache.py        # CLI for activation caching
configs/      # YAML training configs (smoke run, ~20M model, Colab T4)
kaggle/       # Kaggle GPU kernel script + metadata
notebooks/    # train_colab.ipynb: Drive-backed, auto-resuming Colab training
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

## SAE pipeline (after a GPT checkpoint exists)
```bash
python -m tiny_gpt_sae.sae.cache   --gpt run/latest.pt --data data/tinystories/train.bin --layer 3 --out cache/layer3 --n-tokens 50000000 --device cuda
python -m tiny_gpt_sae.sae.train   --config configs/sae_layer3.yaml
python -m tiny_gpt_sae.sae.evals   --gpt run/latest.pt --sae runs/sae_layer3/sae.pt --data data/tinystories/val.bin --layer 3
python -m tiny_gpt_sae.sae.inspect --gpt run/latest.pt --sae runs/sae_layer3/sae.pt --data data/tinystories/val.bin --layer 3 --out runs/sae_layer3/inspect
```

## Training on free Colab
Open `notebooks/train_colab.ipynb` in Colab with a T4 GPU and run all cells. The tokenized data and checkpoints are stored on Google Drive. When a session ends, re-run the notebook and training resumes from `latest.pt`. On GPUs without bf16 (like the T4), the code uses fp16 autocast with a GradScaler.

## Training on Kaggle
`kaggle/train_kaggle.py` is a GPU kernel script. It clones this repo, tokenizes TinyStories, trains with `configs/kaggle.yaml` (with a 10 h wall-clock cap, inside Kaggle's 12 h limit), and writes checkpoints, `log.jsonl`, `loss.png` and `samples.txt` to `/kaggle/working`. Push it with `kaggle kernels push -p kaggle/`.

## Roadmap
- [x] **M1: Model + pipeline.** GPT from scratch, data pipeline, training loop, tests, CI, Colab notebook with resume
- [ ] **M2: Pretraining + evals.** Full TinyStories run, val loss curves, samples, checkpoints released on Hugging Face
- [x] **M3: SAEs (code).** Activation caching, ReLU+L1 and TopK SAEs, dead-feature resampling, evals, feature inspection. Tested on synthetic data with a known dictionary; not yet run on the trained GPT
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
