"""Automated feature labelling with an open LLM: explain, then score.

1. Explain: show the LLM a feature's top-activating contexts (activating token marked
   with << >>) and ask for a short description of what the feature detects.
2. Score (detection): on *held-out* contexts, half where the feature fires strongly and
   half where it doesn't fire, ask the LLM "does this explanation apply to the marked
   token?" and read P(Yes) from the next-token logits. Good explanations separate the
   two sets: we report balanced accuracy and AUROC per feature.

Only the pure functions here are unit-tested; ``LLMJudge`` needs a GPU and the
``transformers`` library and is used from scripts/autointerp.py.
"""

from __future__ import annotations

import re

import torch

EXPLAIN_SYSTEM = (
    "You are an interpretability researcher. You will see text excerpts from children's stories. "
    "In each, one token is marked like <<this>>; a hidden neuron fires strongly on that token. "
    "Describe in at most 12 words what the neuron responds to, being as specific as the "
    "evidence allows (mention the token and the context that matters). "
    "If there is no clear pattern, answer exactly: unclear."
)
SCORE_SYSTEM = (
    "You judge whether a description of a neuron applies to a marked token. Answer only Yes or No."
)


def mark_context(tokens: list[str], idx: int) -> str:
    """Join token strings, wrapping the token at ``idx`` in << >>. Newlines are shown as spaces."""
    parts = [f"<<{t}>>" if i == idx else t for i, t in enumerate(tokens)]
    return "".join(parts).replace("\n", " ").replace("<|endoftext|>", "[END]")


def explain_prompt(contexts: list[str]) -> list[dict]:
    body = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(contexts))
    return [{"role": "system", "content": EXPLAIN_SYSTEM},
            {"role": "user", "content": f"Excerpts:\n{body}\n\nWhat does the neuron respond to?"}]


def score_prompt(explanation: str, context: str) -> list[dict]:
    return [{"role": "system", "content": SCORE_SYSTEM},
            {"role": "user", "content": f"Description: {explanation}\nExcerpt: {context}\n"
                                        "Does the description apply to the marked token?"}]


def clean_explanation(text: str) -> str:
    """Strip quotes/prefixes and keep the first line of a model's answer."""
    text = text.strip().split("\n")[0]
    text = re.sub(r"^(the neuron (responds to|fires on|detects)|answer:)\s*", "", text, flags=re.I).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":  # drop quotes wrapping the whole answer
        text = text[1:-1].strip()
    return text.rstrip(".").strip() or "unclear"


def auroc(scores: list[float], labels: list[int]) -> float:
    """Area under the ROC curve via the rank-sum formula (ties count half)."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def balanced_accuracy(scores: list[float], labels: list[int], threshold: float = 0.5) -> float:
    pred = [s >= threshold for s in scores]
    tpr = sum(p and y for p, y in zip(pred, labels, strict=True)) / max(1, sum(labels))
    tnr = sum((not p) and (not y) for p, y in zip(pred, labels, strict=True)) / max(1, len(labels) - sum(labels))
    return (tpr + tnr) / 2


def split_examples(n_top: int, n_explain: int) -> tuple[list[int], list[int]]:
    """Interleave ranks so explain and held-out sets both cover strong and weaker activations."""
    explain = list(range(0, n_top, 2))[:n_explain]
    held = [i for i in range(n_top) if i not in explain]
    return explain, held


class LLMJudge:
    """Wraps a Hugging Face chat model for batched explanation and Yes/No scoring."""

    def __init__(self, model_name: str, dtype=torch.float16, device: str = "cuda", load_in_4bit: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        kw = {"torch_dtype": dtype, "device_map": device}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype),
                  "device_map": device}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw).eval()
        self.device = device
        self.yes = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes")]
        self.no = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No")]

    def _encode(self, convs: list[list[dict]]):
        texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
        return self.tok(texts, return_tensors="pt", padding=True).to(self.device)

    @torch.no_grad()
    def generate(self, convs: list[list[dict]], max_new_tokens: int = 32) -> list[str]:
        enc = self._encode(convs)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def p_yes(self, convs: list[list[dict]]) -> list[float]:
        """P(Yes) / (P(Yes) + P(No)) from the first answer token's logits."""
        enc = self._encode(convs)
        logits = self.model(**enc).logits[:, -1].float()
        probs = logits.softmax(-1)
        y = probs[:, self.yes].sum(-1)
        n = probs[:, self.no].sum(-1)
        return (y / (y + n).clamp_min(1e-12)).tolist()
