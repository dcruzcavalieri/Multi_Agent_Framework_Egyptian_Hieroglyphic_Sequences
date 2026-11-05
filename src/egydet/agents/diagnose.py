"""Diagnostics of a verifier run: does the agent's probability rank determinatives well, where, and why.

    python -m egydet.agents.diagnose runs/phase2_verifier_signs_gpt-5.4-nano [more runs ...]

For each run (offline, no API call) it reports:
* coverage (positions judged) and the AUROC of p for gold vs non-gold positions, overall and split by
  whether the sign is in the knowledge base;
* veto potential: AUROC of p separating the rule system's true positives from its false positives;
* rescue potential: where the rule system's false negatives rank among all positions the rules do not
  predict (percentile of p);
* calibration: observed determinative rate per probability bin;
* signs with the largest systematic errors (high p on non-determinatives, low p on determinatives).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional, Sequence

from egydet.agents.verifier import load_verifier
from egydet.eval.cv import prepare
from egydet.eval.splits import load_splits
from egydet.gold.io import load_gold
from egydet.kb import ROOT, load_kb
from egydet.legacy_bridge.collect import _rel, fold_best_params, latest_run
from egydet.scoring.scorer import select

BINS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)


def auroc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    ranked = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    # rank-sum with average ranks for ties
    ranks, i = {}, 0
    out_r = [0.0] * len(ranked)
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out_r[k] = r
        i = j + 1
    rsum = sum(r for r, (_, lab) in zip(out_r, ranked) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2) / (n1 * n0)


def diagnose(run_dir: str, rules_run: Optional[str] = None, restrict_ids: Optional[set] = None) -> dict:
    kb = load_kb()
    gold = load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl"))
    samples = prepare(gold, kb)
    if os.path.exists(os.path.join(run_dir, "verifier_folded.jsonl")):
        from egydet.agents.fewshot import test_outputs
        probs = test_outputs(run_dir)      # each sequence as a test item of its own fold
    else:
        probs = load_verifier(run_dir)
    best = fold_best_params(rules_run or latest_run())
    fold_of = {i: f["fold"] for f in load_splits()["folds"] for i in f["test"]}

    rows = []   # one per position
    for r in gold:
        if restrict_ids is not None and r["id"] not in restrict_ids:
            continue
        s = samples[r["id"]]
        rule_pred = set(select(s.structure, s.pool, best[fold_of[r["id"]]], kb))
        gl = set(r["det_idx"])
        for i, sign in enumerate(r["tokens"]):
            rows.append({"id": r["id"], "i": i, "sign": sign, "gold": i in gl, "in_kb": sign in kb.signs,
                         "rule": i in rule_pred, "p": probs.get(r["id"], {}).get(i)})
    judged = [x for x in rows if x["p"] is not None]

    def au(sel):
        return auroc([x["p"] for x in sel if x["gold"]], [x["p"] for x in sel if not x["gold"]])

    rule_tp = [x["p"] for x in judged if x["rule"] and x["gold"]]
    rule_fp = [x["p"] for x in judged if x["rule"] and not x["gold"]]
    outside = [x for x in judged if not x["rule"]]
    fn_p = [x["p"] for x in outside if x["gold"]]
    other_p = sorted(x["p"] for x in outside if not x["gold"])
    pct = [100 * sum(1 for v in other_p if v < p) / max(1, len(other_p)) for p in fn_p]
    calib = []
    for lo, hi in zip(BINS, BINS[1:]):
        b = [x for x in judged if lo <= x["p"] < hi]
        calib.append({"bin": f"[{lo:.1f}, {min(hi, 1.0):.1f}]", "n": len(b),
                      "det_rate": (sum(x["gold"] for x in b) / len(b)) if b else None,
                      "mean_p": mean(x["p"] for x in b) if b else None})
    by_sign = defaultdict(list)
    for x in judged:
        by_sign[x["sign"]].append(x)
    over, under = [], []
    for sign, xs in by_sign.items():
        neg = [x["p"] for x in xs if not x["gold"]]
        pos = [x["p"] for x in xs if x["gold"]]
        if len(neg) >= 3:
            over.append((sign, len(neg), mean(neg), len(pos)))
        if len(pos) >= 2:
            under.append((sign, len(pos), mean(pos), len(neg)))
    over.sort(key=lambda t: -t[1] * t[2])
    under.sort(key=lambda t: t[2])
    return {
        "run": _rel(run_dir) + (f" (restricted to {len(restrict_ids)} sequences)" if restrict_ids is not None else ""),
        "positions": len(rows), "judged": len(judged),
        "auroc_all": au(judged), "auroc_in_kb": au([x for x in judged if x["in_kb"]]),
        "auroc_not_in_kb": au([x for x in judged if not x["in_kb"]]),
        "n_gold_not_in_kb": sum(1 for x in judged if x["gold"] and not x["in_kb"]),
        "veto_auroc_tp_vs_fp": auroc(rule_tp, rule_fp), "rule_tp": len(rule_tp), "rule_fp": len(rule_fp),
        "mean_p_rule_tp": mean(rule_tp) if rule_tp else None, "mean_p_rule_fp": mean(rule_fp) if rule_fp else None,
        "rescue_fn_percentiles": sorted(round(v, 1) for v in pct), "rescue_n_fn": len(fn_p),
        "rescue_auroc": auroc(fn_p, other_p), "n_outside_rules": len(outside),
        "calibration": calib,
        "overconfident_signs": over[:12], "underconfident_signs": under[:12],
    }


def to_markdown(results: List[dict]) -> str:
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    L = ["# Verifier diagnostics", "",
         "AUROC: probability that a randomly chosen determinative gets a higher p than a randomly chosen "
         "non-determinative (0.5 = no information, 1.0 = perfect ranking).", "",
         "| Run | Judged | AUROC all | AUROC in KB | AUROC not in KB | Veto AUROC (rule TP vs FP) | mean p TP / FP | Rescue AUROC (rule FN vs other non-predicted) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['run']} | {r['judged']}/{r['positions']} | {f3(r['auroc_all'])} | {f3(r['auroc_in_kb'])} | "
                 f"{f3(r['auroc_not_in_kb'])} ({r['n_gold_not_in_kb']} gold) | {f3(r['veto_auroc_tp_vs_fp'])} "
                 f"({r['rule_tp']}/{r['rule_fp']}) | {f3(r['mean_p_rule_tp'])} / {f3(r['mean_p_rule_fp'])} | "
                 f"{f3(r['rescue_auroc'])} ({r['rescue_n_fn']} FN among {r['n_outside_rules']}) |")
    for r in results:
        L += ["", f"## {r['run']}", "",
              f"Rescue: percentile of each rule FN's p among the {r['n_outside_rules'] - r['rescue_n_fn']} other "
              f"non-predicted positions: {r['rescue_fn_percentiles']}", "",
              "Calibration:", "", "| p bin | n | mean p | observed rate |", "|---|---|---|---|"]
        for c in r["calibration"]:
            L.append(f"| {c['bin']} | {c['n']} | {f3(c['mean_p'])} | {f3(c['det_rate'])} |")
        L += ["", "Over-confident signs (non-determinative occurrences, mean p, determinative occurrences):", ""]
        L += [f"- {s}: {n} non-det, mean p {p:.2f} (det {d})" for s, n, p, d in r["overconfident_signs"]]
        L += ["", "Under-confident signs (determinative occurrences, mean p, non-determinative occurrences):", ""]
        L += [f"- {s}: {n} det, mean p {p:.2f} (non-det {d})" for s, n, p, d in r["underconfident_signs"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Diagnose verifier runs")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--rules-run", default=None)
    ap.add_argument("--restrict-to", default=None, metavar="RUN_DIR",
                    help="use only the sequences judged in this run (to compare a pilot with a full run)")
    args = ap.parse_args(argv)
    ids = None
    if args.restrict_to:
        if os.path.exists(os.path.join(args.restrict_to, "verifier_folded.jsonl")):
            from egydet.agents.fewshot import test_outputs
            ids = set(test_outputs(args.restrict_to))
        else:
            ids = set(load_verifier(args.restrict_to))
    res = [diagnose(r, args.rules_run, ids) for r in args.runs]
    md = to_markdown(res)
    suffix = "_restricted" if ids is not None else ""
    for r_dir in args.runs:
        with open(os.path.join(r_dir, f"diagnostics{suffix}.md"), "w", encoding="utf-8") as fh:
            fh.write(md)
        with open(os.path.join(r_dir, f"diagnostics{suffix}.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    print(md)


if __name__ == "__main__":
    main()
