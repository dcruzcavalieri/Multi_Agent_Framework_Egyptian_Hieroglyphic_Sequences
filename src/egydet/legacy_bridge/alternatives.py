"""Value of the alternative interpretations and justifications, offline from a collect run.

    python -m egydet.legacy_bridge.alternatives runs/<collect-run>

Three sources of "main + up to 3 alternatives" are compared on the same 99 held-out sequences:

* ``published``  the final v1.47 output (synthesiser answer after deterministic post-processing);
* ``llm_raw``    the synthesiser's own answer before post-processing (needs records with raw answers);
* ``rules_beam`` the next-best distinct sets kept by the rule-only scorer's beam search (no LLM).

Measures (positional): exact match of the main set, exact match of any of the k sets, best F1 over the
k sets, and how often an alternative is strictly better than the main set when the main set is wrong.
Calibration: whether simple uncertainty signals separate wrong main sets from correct ones (AUROC).
Justifications: sign codes cited in the text that are not in the sequence, and predicted signs that the
justification does not mention.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from statistics import mean
from typing import Dict, List, Optional, Sequence

from egydet.eval.metrics import legacy_position_to_sign_index, score_positions
from egydet.gold.io import load_gold
from egydet.kb import ROOT, load_kb
from egydet.legacy_bridge.ablation import load_records, parse_answer
from egydet.legacy_bridge.collect import _rel, fold_best_params
from egydet.mdc import tokenize_with_separators
from egydet.rules.baseline import rule_candidates
from egydet.rules.structure import build_structure
from egydet.scoring.scorer import select, select_ranked

K = 4                      # main + 3 alternatives, as in v1.47
SIGN_CODE = re.compile(r"(?<![A-Za-z0-9])(?:Aa|NL|NU|[A-IK-Z])\d{1,3}[A-Za-z]{0,6}\d?(?![A-Za-z0-9])")


def _items_to_idx(items, legacy) -> List[int]:
    out = set()
    for d in items or []:
        if not isinstance(d, dict):
            continue
        tok = str(d.get("token") or d.get("sign") or "").strip()
        try:
            i = int(d.get("index", d.get("position")))
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(legacy) and legacy[i] == tok:
            si = legacy_position_to_sign_index(legacy, i)
            if si is not None:
                out.add(si)
    return sorted(out)


def _interp_idx(interp: dict, legacy) -> List[int]:
    if not isinstance(interp, dict):
        return []
    dets = interp.get("determinatives")
    if isinstance(dets, list) and dets:
        return _items_to_idx(dets, legacy)
    return _items_to_idx(list(interp.get("semantic_classifiers") or []) + list(interp.get("functional_markers") or []),
                         legacy)


def _sets_from_synth(js: dict, legacy) -> List[List[int]]:
    if not js:
        return []
    sets = [_interp_idx(js.get("main_interpretation") or {}, legacy)]
    for alt in (js.get("alternative_interpretations") or [])[:K - 1]:
        sets.append(_interp_idx(alt, legacy))
    return sets


def _auroc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> Optional[float]:
    """P(score of a wrong case > score of a correct case); ties count 1/2."""
    if not scores_pos or not scores_neg:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in scores_pos for b in scores_neg)
    return wins / (len(scores_pos) * len(scores_neg))


def _consensus_level(js: dict) -> Optional[int]:
    text = json.dumps((js or {}).get("ambiguity") or {}, ensure_ascii=False).lower()
    for pat, lvl in (("all-3", 3), ("all 3", 3), ("3-of-3", 3), ("2-of-3", 2), ("2 of 3", 2), ("1-of-3", 1), ("1 of 3", 1)):
        if pat in text:
            return lvl
    return None


def evaluate(run_dir: str, params_run: Optional[str] = None) -> dict:
    kb = load_kb()
    gold = {r["id"]: r for r in load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl"))}
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    best = fold_best_params(params_run or os.path.join(ROOT, cfg["params_run"]))
    recs = load_records(run_dir)

    sources: Dict[str, Dict[str, List[List[int]]]] = {"published": {}, "llm_raw": {}, "rules_beam": {}}
    signals: Dict[str, Dict[str, float]] = {"n_alternatives_published": {}, "consensus_published(-)": {},
                                            "n_alternatives_llm_raw": {}, "beam_margin(-)": {}}
    just = Counter()
    for rec in recs:
        sid, g = rec["id"], gold[rec["id"]]
        legacy = tokenize_with_separators(rec["mdc"])
        signs_in_seq = set(g["tokens"])
        pub = rec["state"].get("synthesizer_json") or {}
        pub_sets = _sets_from_synth(pub, legacy)
        main_pub = [i for i in rec["final_idx"] if i is not None]
        sources["published"][sid] = [sorted(set(main_pub))] + pub_sets[1:]
        signals["n_alternatives_published"][sid] = len(pub_sets[1:])
        lvl = _consensus_level(pub)
        if lvl is not None:
            signals["consensus_published(-)"][sid] = -lvl

        raw = (rec.get("raw_answers") or {}).get("synthesizer", {}).get("content")
        if raw is not None:
            js = parse_answer(raw)
            sources["llm_raw"][sid] = _sets_from_synth(js, legacy) or [[]]
            signals["n_alternatives_llm_raw"][sid] = max(0, len(sources["llm_raw"][sid]) - 1)
            just["llm_raw.answers"] += 1
            _check_justification(js, sources["llm_raw"][sid][0], g, signs_in_seq, just, "llm_raw")
        _check_justification(pub, sorted(set(main_pub)), g, signs_in_seq, just, "published")

        st = build_structure(rec["mdc"], kb.block_end_signs, kb.plural_markers)
        p = best[rec["fold"]]
        pool = rule_candidates(st, kb)
        main_rules = sorted(select(st, pool, p, kb))
        ranked = select_ranked(st, pool, p, kb, beam_width=10, k=K + 1)
        alts = [s for _, s in ranked if s != main_rules][:K - 1]
        sources["rules_beam"][sid] = [main_rules] + alts
        if len(ranked) >= 2:
            signals["beam_margin(-)"][sid] = -(ranked[0][0] - ranked[1][0])

    table = {}
    main_wrong = {}
    for name, by_id in sources.items():
        if not by_id:
            continue
        rows = []
        for sid, sets in by_id.items():
            gi = gold[sid]["det_idx"]
            f1s = [score_positions(s, gi).f1 for s in sets] or [0.0]
            exact = [sorted(set(s)) == sorted(gi) for s in sets] or [False]
            rows.append({"main_f1": f1s[0], "best_f1": max(f1s), "main_exact": exact[0], "any_exact": any(exact),
                         "alt_better": any(f > f1s[0] for f in f1s[1:]), "n_alts": len(sets) - 1})
            main_wrong[(name, sid)] = not exact[0]
        wrong = [r for r in rows if not r["main_exact"]]
        table[name] = {
            "n": len(rows),
            "mean_alternatives": mean(r["n_alts"] for r in rows),
            "main_exact": mean(r["main_exact"] for r in rows),
            "any_exact@k": mean(r["any_exact"] for r in rows),
            "main_macro_f1": mean(r["main_f1"] for r in rows),
            "best_macro_f1@k": mean(r["best_f1"] for r in rows),
            "n_main_wrong": len(wrong),
            "alt_better_when_main_wrong": (mean(r["alt_better"] for r in wrong) if wrong else None),
            "alt_exact_when_main_wrong": (mean(r["any_exact"] for r in wrong) if wrong else None),
        }

    calib = {}
    for sig, vals in signals.items():
        src = "published" if "published" in sig else ("llm_raw" if "llm_raw" in sig else "rules_beam")
        pos = [v for sid, v in vals.items() if main_wrong.get((src, sid))]
        neg = [v for sid, v in vals.items() if (src, sid) in main_wrong and not main_wrong[(src, sid)]]
        calib[sig] = {"source": src, "n": len(vals), "auroc_wrong_vs_right": _auroc(pos, neg)}
    return {"run": _rel(run_dir), "k": K, "sources": table, "calibration": calib, "justifications": dict(just)}


def _check_justification(js: dict, main_idx: List[int], g: dict, signs_in_seq: set, acc: Counter, tag: str) -> None:
    mi = (js or {}).get("main_interpretation") or {}
    text = " ".join(str(mi.get(k) or "") for k in ("justification", "validation"))
    if not text.strip():
        acc[f"{tag}.no_justification"] += 1
        return
    acc[f"{tag}.with_justification"] += 1
    cited = set(SIGN_CODE.findall(text))
    foreign = {c for c in cited if c not in signs_in_seq}
    acc[f"{tag}.cited_codes"] += len(cited)
    acc[f"{tag}.cited_codes_not_in_sequence"] += len(foreign)
    acc[f"{tag}.justifications_citing_foreign_codes"] += int(bool(foreign))
    predicted = {g["tokens"][i] for i in main_idx}
    acc[f"{tag}.predicted_signs"] += len(predicted)
    acc[f"{tag}.predicted_signs_not_mentioned"] += len(predicted - cited)


def to_markdown(res: dict) -> str:
    pc = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    f2 = lambda v: f"{100 * v:.2f}"  # noqa: E731
    L = [f"# Alternatives and justifications ({res['run']})", "",
         f"k = {res['k']} (main + up to {res['k'] - 1} alternatives). Positional matching on the 99 held-out sequences.", "",
         "| Source | Alternatives / seq. | Main exact | Any exact @k | Macro-F1 main | Best macro-F1 @k | Main wrong | Alt. better when main wrong | Alt. exact when main wrong |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, r in res["sources"].items():
        L.append(f"| {name} | {r['mean_alternatives']:.2f} | {pc(r['main_exact'])} | {pc(r['any_exact@k'])} | "
                 f"{f2(r['main_macro_f1'])} | {f2(r['best_macro_f1@k'])} | {r['n_main_wrong']} | "
                 f"{pc(r['alt_better_when_main_wrong'])} | {pc(r['alt_exact_when_main_wrong'])} |")
    L += ["", "## Calibration (AUROC: does the signal rank wrong main sets above correct ones? 0.5 = no information)", "",
          "| Signal | Source | n | AUROC |", "|---|---|---|---|"]
    for sig, c in res["calibration"].items():
        a = c["auroc_wrong_vs_right"]
        L.append(f"| {sig} | {c['source']} | {c['n']} | {'n/a' if a is None else f'{a:.3f}'} |")
    L += ["", "## Justifications (automatic checks)", ""] + [f"- {k}: {v}" for k, v in sorted(res["justifications"].items())]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Value of alternatives and justifications")
    ap.add_argument("run_dir")
    ap.add_argument("--params-run", default=None)
    args = ap.parse_args(argv)
    res = evaluate(args.run_dir, args.params_run)
    with open(os.path.join(args.run_dir, "alternatives.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    md = to_markdown(res)
    with open(os.path.join(args.run_dir, "alternatives.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
