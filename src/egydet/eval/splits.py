"""Frozen cross-validation splits.

``make_v147_splits`` reproduces exactly the outer 5-fold partition and the inner Dietterich 5x2 halves
of v1.47 (``_make_outer_splits`` + ``_run_5x2_inner_cv``) over the gold records in corpus order, and
stores them by record id so that every later experiment uses the same folds.

    python -m egydet.eval.splits            # writes data/splits/v147_seed42.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import List

from egydet.gold.io import load_gold
from egydet.kb import ROOT

DEFAULT_PATH = os.path.join(ROOT, "data", "splits", "v147_seed42.json")


def make_v147_splits(ids: List[str], seed: int = 42, n_outer: int = 5, n_inner_reps: int = 5) -> dict:
    n = len(ids)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    size, rem = n // n_outer, n % n_outer
    folds, start = [], 0
    for k in range(n_outer):
        sz = size + (1 if k < rem else 0)
        test = order[start:start + sz]
        train_val = order[:start] + order[start + sz:]
        start += sz
        rng = random.Random(seed + k * 13)
        reps = []
        for _ in range(n_inner_reps):
            shuffled = train_val[:]
            rng.shuffle(shuffled)
            mid = len(shuffled) // 2
            reps.append({"A": [ids[i] for i in shuffled[:mid]], "B": [ids[i] for i in shuffled[mid:]]})
        folds.append({"fold": k + 1, "test": [ids[i] for i in test],
                      "train_val": [ids[i] for i in train_val], "inner": reps})
    return {"name": f"v147_seed{seed}", "seed": seed, "n_outer": n_outer, "n_inner_reps": n_inner_reps,
            "source": "v1.47 _make_outer_splits / _run_5x2_inner_cv, gold records in corpus order",
            "n_records": n, "folds": folds}


def load_splits(path: str = DEFAULT_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Write the frozen v1.47 CV splits")
    ap.add_argument("--gold", default=os.path.join(ROOT, "data", "gold", "gold_v1.jsonl"))
    ap.add_argument("--out", default=DEFAULT_PATH)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    gold = sorted(load_gold(args.gold), key=lambda r: r["corpus_row"])
    splits = make_v147_splits([r["id"] for r in gold], seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(splits, fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
