"""Tokenizer and data loading for TinyStories.

Pipeline:
  1. ``prepare_tinystories`` downloads the dataset (Hugging Face ``datasets``),
     tokenizes each story with the GPT-2 BPE tokenizer (``tiktoken``), appends
     an end-of-text token, and writes one flat array of token ids per split
     to ``<out_dir>/{train,val}.bin`` (uint16, since vocab < 65536).
  2. ``TokenDataset`` memory-maps that file and samples random windows of
     ``context_length + 1`` tokens: inputs are the first T, targets are the
     same window shifted by one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Tokenizer:
    """Thin wrapper around tiktoken's GPT-2 BPE so the rest of the code doesn't depend on it."""

    def __init__(self, name: str = "gpt2") -> None:
        import tiktoken

        self.enc = tiktoken.get_encoding(name)
        self.eot = self.enc.eot_token  # <|endoftext|>, separates stories
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


def prepare_tinystories(
    out_dir: str | Path, max_stories: int | None = None, dataset_name: str = "roneneldan/TinyStories"
) -> None:
    """Download + tokenize TinyStories into train.bin / val.bin. Requires ``pip install .[data]``."""
    from datasets import load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer()
    for split, hf_split in (("train", "train"), ("val", "validation")):
        ds = load_dataset(dataset_name, split=hf_split)
        if max_stories is not None:
            ds = ds.select(range(min(max_stories, len(ds))))
        ids: list[int] = []
        for row in ds:
            ids.extend(tok.encode(row["text"]))
            ids.append(tok.eot)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(out_dir / f"{split}.bin")
        print(f"{split}: {len(ds):,} stories -> {len(arr):,} tokens")


class TokenDataset:
    """Random-window sampler over a flat array of token ids."""

    def __init__(self, tokens: np.ndarray, context_length: int) -> None:
        if len(tokens) <= context_length:
            raise ValueError("token array shorter than context_length + 1")
        self.tokens = tokens
        self.context_length = context_length

    @classmethod
    def from_bin(cls, path: str | Path, context_length: int) -> TokenDataset:
        # memmap: the file is read lazily, so it never has to fit in RAM.
        return cls(np.memmap(path, dtype=np.uint16, mode="r"), context_length)

    def get_batch(
        self, batch_size: int, generator: torch.Generator | None = None, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y), each (batch_size, context_length); y is x shifted left by one."""
        T = self.context_length
        starts = torch.randint(0, len(self.tokens) - T - 1, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(self.tokens[s : s + T].astype(np.int64)) for s in starts.tolist()])
        y = torch.stack([torch.from_numpy(self.tokens[s + 1 : s + 1 + T].astype(np.int64)) for s in starts.tolist()])
        return x.to(device), y.to(device)
