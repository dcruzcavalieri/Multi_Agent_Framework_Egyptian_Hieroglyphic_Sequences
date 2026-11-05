"""Evaluate the phonetic alignment veto (``egydet.rules.align``), alone and on top of a verifier fusion.

    python -m egydet.eval.align_eval [--verifier-run runs/phase2_verifier_context_gpt-6-luna] [--out DIR]

Veto: a predicted determinative is removed when the alignment gives its sign a phonetic role in its word
(it writes new letters, or it is a uniliteral complement). The veto has no tuned parameter, so it is
applied to the outer-fold test predictions as they are (rules: fold parameters of the rule-only run;
fusion: the configuration chosen by the inner CV in ``egydet.eval.fusion``). Uses the experts'
transliteration, i.e. the annotation-assistant setting (input = MdC + transliteration + translation).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from egydet.eval.metrics import aggregate, score_positions
from egydet.gold.io import load_gold
from egydet.kb import ROOT
from egydet.legacy_bridge.analysis import paired_comparisons
from egydet.legacy_bridge.collect import _rel, latest_run
from egydet.rules.align import DEFAULT_TABLE, align, load_table


def phonetic_positions(gold: List[dict], table: dict) -> Dict[str, set]:
    out = {}
    for r in gold:
        a = align(r["tokens"], r["transliteration"], table)
        out[r["id"]] = {i for i, (ro, v) in enumerate(zip(a["roles"], a["values"]))
                        if ro == "phon" or (ro == "compl" and len(v) == 1)}
    return out


def run(verifier_run: Optional[str] = None, rules_run: Optional[str] = None, table_path: str = DEFAULT_TABLE) -> dict:
    from egydet.eval import fusion
    gold = load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl"))
    G = {r["id"]: r for r in gold}
    phon = phonetic_positions(gold, load_table(table_path))
    rules_run = rules_run or latest_run()
    if verifier_run:
        res = fusion.run(verifier_run, rules_run)
        base = {c: res["predictions"][c] for c in ("rules_only", "fusion[both]", "llm_only")}
    else:
        from egydet.eval.cv import prepare
        from egydet.eval.splits import load_splits
        from egydet.kb import load_kb
        from egydet.legacy_bridge.collect import fold_best_params
        from egydet.scoring.scorer import select
        kb = load_kb()
        S = prepare(gold, kb)
        best = fold_best_params(rules_run)
        fold = {i: f["fold"] for f in load_splits()["folds"] for i in f["test"]}
        base = {"rules_only": {i: select(S[i].structure, S[i].pool, best[fold[i]], kb) for i in G}}
    preds = dict(base)
    for c, by_id in base.items():
        preds[c + "+align_veto"] = {i: [x for x in v if x not in phon[i]] for i, v in by_id.items()}
    table = {}
    for c, by_id in preds.items():
        ag = aggregate([score_positions(by_id[i], G[i]["det_idx"]) for i in G])
        table[c] = {"macro_f1": ag["macro"]["f1"], "tp": ag["micro"]["tp"], "fp": ag["micro"]["fp"],
                    "fn": ag["micro"]["fn"]}
    removed: Dict[str, Dict[str, int]] = {"fp": {}, "tp": {}}
    for i in G:
        for x in set(preds["rules_only"][i]) - set(preds["rules_only+align_veto"][i]):
            key = "tp" if x in G[i]["det_idx"] else "fp"
            removed[key][G[i]["tokens"][x]] = removed[key].get(G[i]["tokens"][x], 0) + 1
    return {"verifier_run": _rel(verifier_run) if verifier_run else None, "rules_run": _rel(rules_run),
            "table": table, "paired": paired_comparisons(preds, G), "veto_removed_from_rules": removed,
            "predictions": preds}


def to_markdown(res: dict) -> str:
    L = ["# Phonetic alignment veto", "",
         f"Verifier run: {res['verifier_run'] or 'none'}; positional metric over the 99 test predictions "
         "(macro-F1 pooled over sequences).", "",
         "| Condition | Macro-F1 | TP | FP | FN |", "|---|---|---|---|---|"]
    for c, t in res["table"].items():
        L.append(f"| {c} | {100 * t['macro_f1']:.2f} | {t['tp']} | {t['fp']} | {t['fn']} |")
    L += ["", "| vs rules_only | Δ macro-F1 (pp) | 95% CI | better / worse | p (Holm) |", "|---|---|---|---|---|"]
    for c in sorted(res["paired"], key=lambda r: -r["macro_diff"]):
        L.append(f"| {c['condition']} | {100 * c['macro_diff']:+.2f} | [{100 * c['macro_ci'][0]:+.2f}, "
                 f"{100 * c['macro_ci'][1]:+.2f}] | {c['better']} / {c['worse']} | {c['p_holm']:.4f} |")
    r = res["veto_removed_from_rules"]
    L += ["", f"Removed from the rule output: false positives {r['fp']}; true positives {r['tp']}", ""]
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Phonetic alignment veto")
    ap.add_argument("--verifier-run", default=None)
    ap.add_argument("--rules-run", default=None)
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--out", default=None, help="directory for align_veto.md/json (default: the verifier run or runs/)")
    args = ap.parse_args(argv)
    res = run(args.verifier_run, args.rules_run, args.table)
    out = args.out or args.verifier_run or os.path.join(ROOT, "runs")
    md = to_markdown(res)
    with open(os.path.join(out, "align_veto.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(os.path.join(out, "align_veto.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(md)


if __name__ == "__main__":
    main()
