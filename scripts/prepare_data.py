"""Download and tokenize TinyStories: python scripts/prepare_data.py --out data/tinystories [--max-stories N]"""

import argparse

from tiny_gpt_sae.data import prepare_tinystories

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tinystories")
    ap.add_argument("--max-stories", type=int, default=None, help="subset for quick experiments")
    a = ap.parse_args()
    prepare_tinystories(a.out, a.max_stories)
