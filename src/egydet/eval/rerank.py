"""Phase 2, combination 1: the verifier chooses among the rule scorer's own alternatives (offline).

    python -m egydet.eval.rerank runs/phase2_verifier_signs_gpt-6-luna [more verifier runs ...]

For every sequence the rule scorer (fold parameters of the rule-only run) produces up to ``K`` distinct
determinative sets from its beam: the main answer plus alternatives. Each set S is scored with the
verifier's per-position probabilities over the positions where the sets differ (U = union minus
intersection):

    LL(S) = sum_{i in U} [i in S] log p_i + [i not in S] log(1 - p_i)        (p clipped to [0.02, 0.98])
    total(S) = LL(S) + lam * rule_score(S)

The main rule answer is replaced only if the best alternative beats it by more than ``delta``. K, lam
and delta are tuned on the inner 5x2 halves of each outer training partition (as in
``egydet.eval.fusion``); the chosen values are applied to the outer test fold. The verifier outputs are
fold-independent, so no test label is used. No API call is made.

Also reported: the oracle (best set of the K, using the gold: the ceiling of any reranker) and a
descriptive table of fixed configurations over all 99 sequences (not a valid estimate: it is chosen
after seeing the test data).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from typing import Dict, List, Optional, Tuple

from egydet.agents.verifier import load_verifier
from egydet.eval.cv import prepare
from egydet.eval.metrics import aggregate, score_positions, summarize_folds
from egydet.eval.splits import load_splits
from egydet.gold.io import load_gold
from egydet.kb import ROOT, load_kb
from egydet.legacy_bridge.analysis import paired_comparisons
from egydet.legacy_bridge.collect import _rel, fold_best_params, latest_run
from egydet.scoring.scorer import select, select_ranked

K_GRID = (2, 3, 4, 6)
LAM_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
DELTA_GRID = (0.0, 0.5, 1.0, 2.0)
EPS = 0.02


def candidate_sets(s, p: dict, kb, k_max: int) -> List[Tuple[float, List[int]]]:
    """Main rule answer first (its beam score), then distinct alternatives from a wider beam."""
    main = sorted(select(s.structure, s.pool, p, kb))
    ranked = select_ranked(s.structure, s.pool, p, kb, beam_width=12, k=k_max + 1)
    score_of = {tuple(d): sc for sc, d in ranked}
    main_score = score_of.get(tuple(main), ranked[0][0] if ranked else 0.0)
    alts = [(sc, d) for sc, d in ranked if d != main][:k_max - 1]
    return [(main_score, main)] + alts


def loglik(S: List[int], U: set, probs: Dict[int, float]) -> float:
    ll = 0.0
    for i in U:
        q = min(1 - EPS, max(EPS, probs.get(i, 0.5)))
        ll += math.log(q) if i in S else math.log(1 - q)
    return ll


def choose(cands: List[Tuple[float, List[int]]], probs: Dict[int, float], K: int, lam: float, delta: float) -> List[int]:
    cs = cands[:K]
    if len(cs) == 1:
        return cs[0][1]
    sets = [set(d) for _, d in cs]
    U = set.union(*sets) - set.intersection(*sets)
    tot = [loglik(d, U, probs) + lam * sc for sc, d in cs]
    j = max(range(1, len(cs)), key=lambda t: tot[t])
    return cs[j][1] if tot[j] - tot[0] > delta else cs[0][1]


def _macro(ids, cands, probs, gold, cfg) -> float:
    return aggregate([score_positions(choose(cands[i], probs.get(i, {}), **cfg), gold[i]) for i in ids])["macro"]["f1"]


def run(verifier_run: str, rules_run: Optional[str] = None) -> dict:
    kb = load_kb()
    gold_recs = load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl"))
    gold = {r["id"]: r["det_idx"] for r in gold_recs}
    samples = prepare(gold_recs, kb)
    splits = load_splits()
    rules_run = rules_run or latest_run()
    best = fold_best_params(rules_run)
    probs = load_verifier(verifier_run)
    k_max = max(K_GRID)
    grid = [dict(K=K, lam=lam, delta=d) for K, lam, d in itertools.product(K_GRID, LAM_GRID, DELTA_GRID)]

    preds: Dict[str, Dict[str, List[int]]] = {"rules_only": {}, "rerank": {}, **{f"oracle_K{K}": {} for K in K_GRID}}
    chosen, cands_all = [], {}
    for f in splits["folds"]:
        k, p = f["fold"], best[f["fold"]]
        ids = f["train_val"] + f["test"]
        cands = {i: candidate_sets(samples[i], p, kb, k_max) for i in ids}   # rules use the fold's parameters
        inner = []
        for halves in f["inner"]:
            for tr, va in ((halves["A"], halves["B"]), (halves["B"], halves["A"])):
                top = max(grid, key=lambda c: _macro(tr, cands, probs, gold, c))   # first best: grid order
                inner.append((_macro(va, cands, probs, gold, top), top))
        val, cfg = max(inner, key=lambda x: x[0])
        chosen.append({"fold": k, "config": cfg, "val": val})
        for i in f["test"]:
            cands_all[i] = cands[i]
            preds["rules_only"][i] = cands[i][0][1]
            preds["rerank"][i] = choose(cands[i], probs.get(i, {}), **cfg)
            for K in K_GRID:
                preds[f"oracle_K{K}"][i] = max((d for _, d in cands[i][:K]),
                                               key=lambda d: score_positions(d, gold[i]).f1)

    fold_of = {i: f["fold"] for f in splits["folds"] for i in f["test"]}
    table = {}
    for cond, by_id in preds.items():
        per_fold: Dict[int, list] = {}
        for sid, idx in by_id.items():
            per_fold.setdefault(fold_of[sid], []).append(score_positions(idx, gold[sid]))
        aggs = [aggregate(v) for _, v in sorted(per_fold.items())]
        pooled = aggregate([x for v in per_fold.values() for x in v])
        table[cond] = {"macro_f1": summarize_folds([a["macro"]["f1"] for a in aggs]),
                       "tp": pooled["micro"]["tp"], "fp": pooled["micro"]["fp"], "fn": pooled["micro"]["fn"],
                       "changed": sum(1 for i in by_id if by_id[i] != preds["rules_only"][i])}
    comps = paired_comparisons({c: preds[c] for c in ("rules_only", "rerank")},
                               {r["id"]: r for r in gold_recs})
    ids = sorted(cands_all)
    fixed = [{"config": c, "macro_f1": _macro(ids, cands_all, probs, gold, c)} for c in grid]
    fixed.sort(key=lambda r: -r["macro_f1"])
    n_alt = [len(cands_all[i]) - 1 for i in ids]
    return {"verifier_run": _rel(verifier_run), "rules_run": _rel(rules_run), "table": table, "paired": comps,
            "chosen": chosen, "fixed_top": fixed[:10], "fixed_rules": _macro(ids, cands_all, probs, gold, dict(K=1, lam=0, delta=0)),
            "mean_alternatives": sum(n_alt) / len(n_alt), "predictions": preds}


def to_markdown(results: List[dict]) -> str:
    f = lambda s: f"{100 * s['mean']:.2f} ± {100 * (s['std_sample'] or 0):.2f}"  # noqa: E731
    L = ["# Reranking of the rule scorer's alternatives by the verifier (offline)", "",
         "Positional metric; macro-F1 mean ± sample SD over outer folds; TP/FP/FN pooled; "
         "'changed' = sequences where the answer differs from rules_only.", ""]
    for r in results:
        L += [f"## {r['verifier_run']}", "", f"Alternatives per sequence (up to K={max(K_GRID)}): "
              f"mean {r['mean_alternatives']:.2f}", "",
              "| Condition | Macro-F1 | TP | FP | FN | changed |", "|---|---|---|---|---|---|"]
        for c, t in r["table"].items():
            L.append(f"| {c} | {f(t['macro_f1'])} | {t['tp']} | {t['fp']} | {t['fn']} | {t['changed']} |")
        for c in r["paired"]:
            L += ["", f"Paired vs rules_only: {100 * c['macro_diff']:+.2f} pp, 95% CI [{100 * c['macro_ci'][0]:+.2f}, "
                  f"{100 * c['macro_ci'][1]:+.2f}], better/worse/tie {c['better']}/{c['worse']}/{c['ties']}, "
                  f"sign test p {c.get('p_sign', float('nan')):.4f}"]
        L += ["", "Inner-CV choice per fold: " + "; ".join(
            f"{c['fold']}: K={c['config']['K']} lam={c['config']['lam']} delta={c['config']['delta']}" for c in r["chosen"]),
              "", "Descriptive only (configuration chosen on the 99 test sequences, optimistic): best fixed "
              "configurations: " + "; ".join(
                  f"K={x['config']['K']} lam={x['config']['lam']} d={x['config']['delta']}: {100 * x['macro_f1']:.2f}"
                  for x in r["fixed_top"][:5]), ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Offline reranking of rule alternatives with verifier probabilities")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--rules-run", default=None)
    args = ap.parse_args(argv)
    res = [run(r, args.rules_run) for r in args.runs]
    md = to_markdown(res)
    for r_dir, r in zip(args.runs, res):
        with open(os.path.join(r_dir, "rerank.json"), "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1)
        with open(os.path.join(r_dir, "rerank.md"), "w", encoding="utf-8") as fh:
            fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
