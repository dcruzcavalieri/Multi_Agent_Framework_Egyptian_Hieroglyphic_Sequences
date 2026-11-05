"""Phase 2: fuse the verifier agent with the rule scorer, tuned inside the nested CV.

    python -m egydet.eval.fusion runs/phase2_verifier_signs_gpt-5.4-nano [--rules-run runs/<rule-only run>]

For every sign position the verifier gives a probability p. The rule candidate score s (the scorer's
per-position score with the fold's optimised parameters) is combined as

    s' = clip(s + w * (p - tau), 0, 1)

and the usual span selection runs on s'. Modes: ``veto`` adjusts only rule candidates, ``rescue`` adds
only positions the rules do not consider, ``both`` does both. ``llm_only`` predicts p >= theta. The
fusion parameters are chosen on the inner 5x2 halves of each outer training partition (same splits as
the rule-only run), the best validation choice is applied to the outer test fold. Results are compared
with ``rules_only`` by the paired tests of ``egydet.legacy_bridge.analysis``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Dict, List, Optional

from egydet.agents.verifier import load_verifier
from egydet.eval.cv import prepare
from egydet.eval.metrics import aggregate, score_positions, summarize_folds
from egydet.eval.splits import load_splits
from egydet.gold.io import load_gold
from egydet.kb import ROOT, load_kb
from egydet.legacy_bridge.analysis import paired_comparisons
from egydet.legacy_bridge.collect import _rel, fold_best_params, latest_run
from egydet.scoring.scorer import BEAM, score_candidate, score_span, select

W_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
TAU_GRID = (0.3, 0.5, 0.7, 0.9)
THETA_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MODES = ("veto", "rescue", "both")


def fused_select(s, probs: Dict[int, float], p: dict, kb, w: float, tau: float, mode: str) -> List[int]:
    st, pool = s.structure, s.pool
    occ = {i: score_candidate(c, st.signs, p, kb) for i, c in pool.items()}
    never = kb.never_determinative
    for i, sign in enumerate(st.signs):
        if sign in never:
            continue
        pi = probs.get(i, tau)
        if i in pool and mode in ("veto", "both"):
            occ[i] = min(1.0, max(0.0, occ[i] + w * (pi - tau)))
        elif i not in pool and mode in ("rescue", "both"):
            occ[i] = min(1.0, max(0.0, w * (pi - tau)))
    beam = [(0.0, [])]
    for cands in st.span_candidates:
        new_beam = []
        for score, dets in beam:
            for span in cands:
                sp = () if any(st.signs[i] in never for i in span) else span
                new_beam.append((score + score_span(sp, st.signs, occ, p, kb), dets + list(sp)))
        new_beam.sort(key=lambda x: -x[0])
        beam = new_beam[:BEAM]
    return beam[0][1] if beam else []


def _macro(samples, predict) -> float:
    return aggregate([score_positions(predict(s), s.gold_idx) for s in samples])["macro"]["f1"]


def run(verifier_run: str, rules_run: Optional[str] = None) -> dict:
    kb = load_kb()
    gold_recs = load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl"))
    samples = prepare(gold_recs, kb)
    splits = load_splits()
    folded_path = os.path.join(verifier_run, "verifier_folded.jsonl")
    if os.path.exists(folded_path):   # fold-specific outputs (few-shot from the training fold)
        from egydet.agents.fewshot import load_folded
        probs_by_fold = load_folded(verifier_run)
    else:                              # fold-independent outputs: one run serves every fold
        single = load_verifier(verifier_run)
        probs_by_fold = {f["fold"]: single for f in splits["folds"]}
    for f in splits["folds"]:
        have = probs_by_fold.get(f["fold"], {})
        missing = [i for i in f["train_val"] + f["test"] if i not in have]
        if missing:
            raise RuntimeError(f"verifier run lacks {len(missing)} sequences for fold {f['fold']}, e.g. {missing[:3]}")
    rules_run = rules_run or latest_run()
    best = fold_best_params(rules_run)

    preds: Dict[str, Dict[str, List[int]]] = {"rules_only": {}, "llm_only": {}, **{f"fusion[{m}]": {} for m in MODES}}
    chosen = []
    for f in splits["folds"]:
        k, p = f["fold"], best[f["fold"]]
        probs = probs_by_fold[k]
        test = [samples[i] for i in f["test"]]
        for s in test:
            preds["rules_only"][s.id] = select(s.structure, s.pool, p, kb)
        configs = {f"fusion[{m}]": [dict(mode=m, w=w, tau=t) for w, t in itertools.product(W_GRID, TAU_GRID)]
                   for m in MODES}
        configs["llm_only"] = [dict(theta=th) for th in THETA_GRID]

        def predictor(name, cfg):
            if name == "llm_only":
                return lambda s: sorted(i for i, v in probs[s.id].items() if v >= cfg["theta"])
            return lambda s: fused_select(s, probs[s.id], p, kb, cfg["w"], cfg["tau"], cfg["mode"])

        for name, grid in configs.items():
            inner = []
            for halves in f["inner"]:
                A, B = [samples[i] for i in halves["A"]], [samples[i] for i in halves["B"]]
                for train, val in ((A, B), (B, A)):
                    scored = [(_macro(train, predictor(name, c)), c) for c in grid]
                    top = max(scored, key=lambda x: x[0])     # first best on ties (grid order: smaller w first)
                    inner.append((_macro(val, predictor(name, top[1])), top[0], top[1]))
            val, train_v, cfg = max(inner, key=lambda x: x[0])
            chosen.append({"fold": k, "condition": name, "config": cfg, "val": val, "train": train_v})
            pred = predictor(name, cfg)
            for s in test:
                preds[name][s.id] = pred(s)

    gold = {r["id"]: r for r in gold_recs}
    fold_of = {i: f["fold"] for f in splits["folds"] for i in f["test"]}
    table = {}
    for cond, by_id in preds.items():
        per_fold = {}
        for sid, idx in by_id.items():
            per_fold.setdefault(fold_of[sid], []).append(score_positions(idx, gold[sid]["det_idx"]))
        aggs = [aggregate(v) for _, v in sorted(per_fold.items())]
        pooled = aggregate([x for v in per_fold.values() for x in v])
        table[cond] = {"macro_f1": summarize_folds([a["macro"]["f1"] for a in aggs]),
                       "micro_f1": summarize_folds([a["micro"]["f1"] for a in aggs]),
                       "tp": pooled["micro"]["tp"], "fp": pooled["micro"]["fp"], "fn": pooled["micro"]["fn"],
                       "outcomes": pooled["outcomes"]}
    comps = paired_comparisons(preds, gold)
    return {"verifier_run": _rel(verifier_run), "rules_run": _rel(rules_run), "table": table,
            "paired": comps, "chosen": chosen, "predictions": preds}


def to_markdown(res: dict) -> str:
    f = lambda s: f"{100 * s['mean']:.2f} ± {100 * (s['std_sample'] or 0):.2f}"  # noqa: E731
    L = [f"# Phase 2 fusion ({res['verifier_run']})", "",
         "Positional metric; mean ± sample SD over outer folds; TP/FP/FN pooled.", "",
         "| Condition | Macro-F1 | Micro-F1 | TP | FP | FN | FP_NULL | FN_NULL |", "|---|---|---|---|---|---|---|---|"]
    for c, r in res["table"].items():
        L.append(f"| {c} | {f(r['macro_f1'])} | {f(r['micro_f1'])} | {r['tp']} | {r['fp']} | {r['fn']} | "
                 f"{r['outcomes']['FP_NULL']} | {r['outcomes']['FN_NULL']} |")
    L += ["", "## Paired comparison with rules_only (99 sequences, bootstrap, Holm)", "",
          "| Condition | Δ macro-F1 (pp) | 95% CI | better / worse / tie | p (Holm) |", "|---|---|---|---|---|"]
    for r in sorted(res["paired"], key=lambda r: -r["macro_diff"]):
        L.append(f"| {r['condition']} | {100 * r['macro_diff']:+.2f} | [{100 * r['macro_ci'][0]:+.2f}, "
                 f"{100 * r['macro_ci'][1]:+.2f}] | {r['better']} / {r['worse']} / {r['ties']} | {r['p_holm']:.4f} |")
    L += ["", "## Parameters chosen per fold (inner CV)", "", "| Fold | Condition | Config | Val macro-F1 |", "|---|---|---|---|"]
    for c in res["chosen"]:
        L.append(f"| {c['fold']} | {c['condition']} | {json.dumps(c['config'])} | {100 * c['val']:.2f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Nested-CV fusion of the verifier agent with the rule scorer")
    ap.add_argument("verifier_run")
    ap.add_argument("--rules-run", default=None)
    args = ap.parse_args(argv)
    res = run(args.verifier_run, args.rules_run)
    with open(os.path.join(args.verifier_run, "fusion.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    md = to_markdown(res)
    with open(os.path.join(args.verifier_run, "fusion.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
