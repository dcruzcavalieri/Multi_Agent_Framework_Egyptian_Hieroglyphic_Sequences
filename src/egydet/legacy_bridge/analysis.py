"""Phase 1 analyses, offline from a collect run.

    python -m egydet.legacy_bridge.analysis runs/<collect-run> [--extra name=dir ...]

1. Paired comparison of every condition against ``rules_only`` over the 99 sequences (all of them are
   held-out test predictions): per-sequence positional F1 differences, paired bootstrap (10,000
   resamples) of the macro-F1 and micro-F1 differences with 95% percentile intervals, an exact sign
   test on better/worse sequences, and Holm-adjusted p-values over the comparisons.
2. Error analysis of ``rules_only``: false negatives and false positives by sign, knowledge-base
   coverage, and how often each agent proposed the same position (could an agent rescue a miss, or
   veto a false alarm, and at what cost in true positives).

Writes analysis.md and analysis.json into the run directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from egydet.eval.metrics import score_positions
from egydet.kb import load_kb
from egydet.legacy_bridge.ablation import agent_json, agent_proposals, predict_conditions
from egydet.mdc import tokenize_with_separators

REFERENCE = "rules_only"
N_BOOT = 10_000
SEED = 42


def _sign_test_p(n_better: int, n_worse: int) -> float:
    n = n_better + n_worse
    if n == 0:
        return 1.0
    k = min(n_better, n_worse)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _micro_f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)


def paired_comparisons(preds: Dict[str, Dict[str, List[int]]], gold: Dict[str, dict]) -> List[dict]:
    ids = sorted(preds[REFERENCE])
    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))

    def per_seq(cond):
        s = [score_positions(preds[cond][i], gold[i]["det_idx"]) for i in ids]
        return (np.array([x.f1 for x in s]), np.array([x.tp for x in s]),
                np.array([x.fp for x in s]), np.array([x.fn for x in s]))

    ref = per_seq(REFERENCE)
    rows = []
    for cond in preds:
        if cond == REFERENCE or set(preds[cond]) != set(ids):
            continue
        cur = per_seq(cond)
        d = cur[0] - ref[0]
        macro_boot = d[boot].mean(axis=1)
        micro_c = _micro_f1(cur[1][boot].sum(1), cur[2][boot].sum(1), cur[3][boot].sum(1))
        micro_r = _micro_f1(ref[1][boot].sum(1), ref[2][boot].sum(1), ref[3][boot].sum(1))
        micro_boot = micro_c - micro_r
        better, worse = int((d > 1e-12).sum()), int((d < -1e-12).sum())
        p_boot = float(min(1.0, 2 * min((macro_boot <= 0).mean(), (macro_boot >= 0).mean())))
        rows.append({
            "condition": cond,
            "macro_diff": float(d.mean()),
            "macro_ci": [float(np.percentile(macro_boot, 2.5)), float(np.percentile(macro_boot, 97.5))],
            "micro_diff": float(_micro_f1(cur[1].sum(), cur[2].sum(), cur[3].sum())
                                - _micro_f1(ref[1].sum(), ref[2].sum(), ref[3].sum())),
            "micro_ci": [float(np.percentile(micro_boot, 2.5)), float(np.percentile(micro_boot, 97.5))],
            "better": better, "worse": worse, "ties": len(ids) - better - worse,
            "p_bootstrap": p_boot, "p_sign": _sign_test_p(better, worse),
        })
    # Holm adjustment over all comparisons (bootstrap p)
    order = sorted(range(len(rows)), key=lambda k: rows[k]["p_bootstrap"])
    m, running = len(rows), 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * rows[k]["p_bootstrap"]))
        rows[k]["p_holm"] = running
    return rows


def error_analysis(preds, gold, recs) -> dict:
    kb = load_kb()
    rec_by_id = {r["id"]: r for r in recs}
    rule = preds[REFERENCE]
    items = []   # one row per error or TP position of the rule system
    for sid, g in gold.items():
        if sid not in rule:
            continue
        rec = rec_by_id[sid]
        legacy = tokenize_with_separators(rec["mdc"])
        agents = {a: {i for i, _ in agent_proposals(agent_json(rec, a)[0], legacy)[0]}
                  for a in ("semantic", "structural")}
        pred, gl = set(rule[sid]), set(g["det_idx"])
        n = len(g["tokens"])
        for kind, idxs in (("TP", pred & gl), ("FP", pred - gl), ("FN", gl - pred)):
            for i in sorted(idxs):
                sign = g["tokens"][i]
                items.append({
                    "id": sid, "kind": kind, "index": i, "sign": sign,
                    "in_kb": sign in kb.signs,
                    "kb_category": kb.signs.get(sign, {}).get("category", ""),
                    "tier": next((t for t, ss in kb.tiers.items() if sign in ss), ""),
                    "last_sign": i == n - 1, "gold_empty": not gl,
                    "semantic": i in agents["semantic"], "structural": i in agents["structural"],
                })

    def count(kind, key=None):
        sel = [x for x in items if x["kind"] == kind and (key is None or x[key])]
        return len(sel)

    summary = {k: count(k) for k in ("TP", "FP", "FN")}
    rescue = {a: count("FN", a) for a in ("semantic", "structural")}
    rescue["either"] = sum(1 for x in items if x["kind"] == "FN" and (x["semantic"] or x["structural"]))
    fn_out_kb = sum(1 for x in items if x["kind"] == "FN" and not x["in_kb"])

    # veto simulation: drop a rule prediction unless the agent(s) also proposed it
    veto = {}
    for name, test in (("semantic", lambda x: x["semantic"]), ("structural", lambda x: x["structural"]),
                       ("either agent", lambda x: x["semantic"] or x["structural"]),
                       ("both agents", lambda x: x["semantic"] and x["structural"])):
        veto[name] = {"FP_removed": sum(1 for x in items if x["kind"] == "FP" and not test(x)),
                      "TP_lost": sum(1 for x in items if x["kind"] == "TP" and not test(x))}

    by_sign = defaultdict(Counter)
    for x in items:
        by_sign[x["sign"]][x["kind"]] += 1
    top_fn = sorted(((s, c["FN"], c["TP"]) for s, c in by_sign.items() if c["FN"]), key=lambda t: -t[1])[:15]
    top_fp = sorted(((s, c["FP"], c["TP"]) for s, c in by_sign.items() if c["FP"]), key=lambda t: -t[1])[:15]
    fp_by_tier = Counter(x["tier"] or "(none)" for x in items if x["kind"] == "FP")
    fn_by_category = Counter((x["kb_category"] or "(not in KB)") for x in items if x["kind"] == "FN")
    return {"summary": summary, "fn_not_in_kb": fn_out_kb, "fn_found_by_agent": rescue, "veto": veto,
            "top_fn_signs": top_fn, "top_fp_signs": top_fp, "fp_by_tier": dict(fp_by_tier),
            "fn_by_kb_category": dict(fn_by_category), "items": items}


def to_markdown(run: str, comps: List[dict], err: dict) -> str:
    pct = lambda v: f"{100 * v:+.2f}"  # noqa: E731
    L = [f"# Phase 1 analyses ({run})", "",
         "## 1. Paired comparison against rules_only (positional F1, 99 held-out sequences)", "",
         f"Paired bootstrap over sequences ({N_BOOT} resamples, seed {SEED}); 95% percentile intervals; "
         "sign test on sequences that got better or worse; Holm adjustment over all rows.", "",
         "| Condition | Δ macro-F1 (pp) | 95% CI | Δ micro-F1 (pp) | 95% CI | better / worse / tie | p (boot) | p (Holm) | p (sign) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(comps, key=lambda r: -r["macro_diff"]):
        L.append(f"| {r['condition']} | {pct(r['macro_diff'])} | [{pct(r['macro_ci'][0])}, {pct(r['macro_ci'][1])}] | "
                 f"{pct(r['micro_diff'])} | [{pct(r['micro_ci'][0])}, {pct(r['micro_ci'][1])}] | "
                 f"{r['better']} / {r['worse']} / {r['ties']} | {r['p_bootstrap']:.4f} | {r['p_holm']:.4f} | {r['p_sign']:.4f} |")
    s = err["summary"]
    L += ["", "## 2. Errors of rules_only", "",
          f"- Positions: TP {s['TP']}, FP {s['FP']}, FN {s['FN']}.",
          f"- False negatives whose sign is not in the knowledge base (unreachable for the rules): {err['fn_not_in_kb']}.",
          f"- False negatives proposed by an agent at the same position: semantic {err['fn_found_by_agent']['semantic']}, "
          f"structural {err['fn_found_by_agent']['structural']}, either {err['fn_found_by_agent']['either']}.",
          "", "### Agent as a veto on rule predictions (keep a rule prediction only if the agent proposed it too)", "",
          "| Veto by | FP removed | TP lost |", "|---|---|---|"]
    L += [f"| {k} | {v['FP_removed']} | {v['TP_lost']} |" for k, v in err["veto"].items()]
    L += ["", "### Signs most often missed (FN) and wrongly predicted (FP)", "",
          "| Sign | FN | TP |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in err["top_fn_signs"]]
    L += ["", "| Sign | FP | TP |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in err["top_fp_signs"]]
    L += ["", f"- FP by sign tier: {err['fp_by_tier']}", f"- FN by KB category: {err['fn_by_kb_category']}", ""]
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Paired comparisons and error analysis of a Phase 1 run")
    ap.add_argument("run_dir")
    ap.add_argument("--params-run", default=None)
    ap.add_argument("--extra", nargs="*", default=[], metavar="NAME=RUN_DIR")
    args = ap.parse_args(argv)
    preds, gold, recs, _ = predict_conditions(args.run_dir, args.params_run, dict(e.split("=", 1) for e in args.extra))
    comps = paired_comparisons(preds, gold)
    err = error_analysis(preds, gold, recs)
    with open(os.path.join(args.run_dir, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump({"comparisons": comps, "errors": err}, fh, indent=1, default=str)
    md = to_markdown(args.run_dir, comps, err)
    with open(os.path.join(args.run_dir, "analysis.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
