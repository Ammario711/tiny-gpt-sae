"""Kaggle kernel: clone tiny-gpt-sae, prepare TinyStories, train, then save a loss plot and samples.

Everything is written to /kaggle/working, which Kaggle keeps as the kernel's output.
Push with:  kaggle kernels push -p kaggle/
"""

import json
import os
import subprocess
import sys
import time

WORK = "/kaggle/working"
REPO = "/kaggle/tmp/tiny-gpt-sae"  # code lives outside /kaggle/working so it isn't saved as output
DATA = f"{WORK}/data/tinystories"
RUN = f"{WORK}/run"


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


t0 = time.time()
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")

# 1. Code. Use Kaggle's preinstalled PyTorch (built for its GPUs), so install with --no-deps.
os.makedirs("/kaggle/tmp", exist_ok=True)
sh(f"rm -rf {REPO} && git clone --depth 1 https://github.com/Ammario711/tiny-gpt-sae {REPO}")
sh(f"{sys.executable} -m pip install -q --no-deps -e {REPO}")
sh(f"{sys.executable} -m pip install -q tiktoken datasets pyyaml")
sys.path.insert(0, f"{REPO}/src")

# 2. Data: download + tokenize TinyStories into uint16 .bin files.
os.environ.setdefault("HF_HOME", "/kaggle/tmp/hf_cache")
sh(f"{sys.executable} {REPO}/scripts/prepare_data.py --out {DATA}")
sh("rm -rf /kaggle/tmp/hf_cache")  # free disk; the .bin files are all we need
print(f"[timing] setup + data prep: {(time.time() - t0) / 60:.1f} min", flush=True)

# 3. Train (prints a line every eval_interval steps and appends to run/log.jsonl).
sh(f"cd {REPO} && {sys.executable} -u -m tiny_gpt_sae.train --config configs/kaggle.yaml "
   f"--data-dir {DATA} --out-dir {RUN}")

# 4. Loss plot.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = {r["step"]: r for r in map(json.loads, open(f"{RUN}/log.jsonl"))}
steps = sorted(rows)
plt.figure(figsize=(7, 4))
plt.plot(steps, [rows[s]["train_loss"] for s in steps], label="train")
plt.plot(steps, [rows[s].get("val_loss") for s in steps], label="val")
plt.xlabel("optimizer step")
plt.ylabel("cross-entropy loss")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"{WORK}/loss.png", dpi=150, bbox_inches="tight")

# 5. Samples from the final checkpoint.
import torch  # noqa: E402

from tiny_gpt_sae.data import Tokenizer  # noqa: E402
from tiny_gpt_sae.model import GPT, GPTConfig  # noqa: E402

ckpt = torch.load(f"{RUN}/latest.pt", map_location="cuda", weights_only=False)
model = GPT(GPTConfig(**ckpt["config"]["model"])).cuda().eval()
model.load_state_dict(ckpt["model"])
tok = Tokenizer()
torch.manual_seed(0)
with open(f"{WORK}/samples.txt", "w") as f:
    f.write(f"checkpoint step {ckpt['step']}\n\n")
    for prompt in ["Once upon a time", "Lily wanted to", "The little dog", "One day, Tom"]:
        out = model.generate(torch.tensor([tok.encode(prompt)]).cuda(), 150, temperature=0.8, top_k=50)
        f.write(tok.decode(out[0].tolist()).split("<|endoftext|>")[0] + "\n" + "-" * 60 + "\n")
print(open(f"{WORK}/samples.txt").read())

sh(f"rm -rf {WORK}/data")  # don't save ~1 GB of tokens as kernel output
print(f"[timing] total: {(time.time() - t0) / 3600:.2f} h")
