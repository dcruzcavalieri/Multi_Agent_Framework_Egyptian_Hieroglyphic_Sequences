"""Phase 1 ablation, computed offline from the records written by ``egydet.legacy_bridge.collect``.

    python -m egydet.legacy_bridge.ablation runs/<collect-run>

Conditions (all on the same outer folds, fold-specific optimised scorer parameters):

* ``rules_only``            rule candidates + scorer (no LLM)
* ``v147_full``             final output of the published pipeline (synthesiser + post-processing)
* ``semantic_raw``          determinatives proposed by the semantic agent, taken as is
* ``structural_raw``        determinatives proposed by the structural agent, taken as is
* ``v147_full[<name>]``      final output of another collect run (``--extra name=dir``); with a run made
                            in ``--mode fake`` (every LLM answer empty) this isolates the effect of the
                            v1.47 post-processing alone
* ``scorer[<sources>|<w>]`` rule candidates plus the chosen agents' proposals, ranked by the scorer;
                            ``w=v147`` uses the source weights of the v1.47 configuration,
                            ``w=equal`` gives the three sources 1/3 each

Agent proposals are read from ``main_interpretation.determinatives`` (the schema the v1.47 prompts
request); a listed sign gets confidence 1.0 because that schema has no confidence field. Proposals whose
(sign, index) pair does not match the sequence are dropped and counted. Every condition is scored positionally and as a sign multiset.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from egydet.eval.metrics import aggregate, legacy_position_to_sign_index, score_positions, score_signs, summarize_folds
from egydet.gold.io import load_gold
from egydet.kb import ROOT, load_kb
from egydet.legacy_bridge.collect import _rel, fold_best_params
from egydet.mdc import tokenize_with_separators
from egydet.rules.baseline import Candidate, rule_candidates
from egydet.rules.structure import build_structure
from egydet.scoring.scorer import select

WEIGHTS = {
    "v147": {"w_sem_llm": 0.0, "w_struct_llm": 0.0, "w_rule_llm": 1.0},
    "equal": {"w_sem_llm": 1 / 3, "w_struct_llm": 1 / 3, "w_rule_llm": 1 / 3},
}
SOURCE_SETS = {"rules+sem": ("semantic",), "rules+struct": ("structural",), "rules+sem+struct": ("semantic", "structural")}


def load_records(run_dir: str) -> List[dict]:
    with open(os.path.join(run_dir, "records.jsonl"), encoding="utf-8") as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    latest = {}
    for r in recs:            # a resumed run may contain duplicates; keep the last record per id
        latest[r["id"]] = r
    return list(latest.values())


def _agent_items(state_json: Optional[dict]) -> list:
    """Determinative list of an agent answer.

    The v1.47 prompts ask for ``main_interpretation.determinatives`` (items: token, index, gardiner_code,
    needs_lookup, word_id; no confidence). A top-level ``determinatives`` list is accepted as well.
    """
    js = state_json or {}
    mi = js.get("main_interpretation")
    if isinstance(mi, dict) and isinstance(mi.get("determinatives"), list):
        return mi["determinatives"]
    for key in ("determinatives", "candidates"):
        if isinstance(js.get(key), list):
            return js[key]
    return []


def parse_answer(content: str) -> dict:
    """Strict JSON reading of an agent answer (BEGIN_JSON/END_JSON markers or the outermost braces).

    Unlike v1.47 ``parse_llm_json`` it does not replace typographic quotes, which corrupts otherwise valid
    JSON whose string values contain them. Trailing commas are tolerated.
    """
    import re
    if not isinstance(content, str) or not content.strip():
        return {}
    t = content.strip()
    if "BEGIN_JSON" in t and "END_JSON" in t:
        t = t.split("BEGIN_JSON", 1)[1].split("END_JSON", 1)[0]
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip(), flags=re.I | re.S)
    if "{" not in t:
        return {}
    t = t[t.find("{"): t.rfind("}") + 1]
    for candidate in (t, re.sub(r",\s*([}\]])", r"\1", t)):
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else {}
        except ValueError:
            continue
    return {}


def agent_json(rec: dict, agent: str) -> Tuple[dict, str]:
    """Agent answer as v1.47 parsed it, or re-parsed from the raw answer when v1.47 got nothing."""
    js = rec["state"].get(f"{agent}_json") or {}
    if js:
        return js, "v147"
    raw = (rec.get("raw_answers") or {}).get(agent, {}).get("content")
    js = parse_answer(raw) if raw else {}
    return js, ("recovered" if js else ("no_raw" if raw is None else "unparseable"))


DEFAULT_AGENT_CONFIDENCE = 1.0   # the agents' schema has no confidence field; a listed sign counts as 1.0


def agent_proposals(state_json: Optional[dict], legacy_toks: List[str]) -> Tuple[List[Tuple[int, float]], int]:
    """Valid (sign_index, confidence) proposals and the number of invalid ones."""
    out, bad = [], 0
    for d in _agent_items(state_json):
        if not isinstance(d, dict):
            bad += 1
            continue
        tok = str(d.get("token") or "").strip()
        try:
            idx = int(d.get("index"))
        except (TypeError, ValueError):
            bad += 1
            continue
        if not tok or not (0 <= idx < len(legacy_toks)) or legacy_toks[idx] != tok:
            bad += 1
            continue
        si = legacy_position_to_sign_index(legacy_toks, idx)
        if si is None:
            bad += 1
            continue
        try:
            conf = float(d["confidence"]) if "confidence" in d else DEFAULT_AGENT_CONFIDENCE
        except (TypeError, ValueError):
            conf = DEFAULT_AGENT_CONFIDENCE
        out.append((si, conf))
    return out, bad


def build_pool(st, rec: dict, kb, sources: Iterable[str]) -> Dict[int, Candidate]:
    legacy = tokenize_with_separators(rec["mdc"])
    pool = rule_candidates(st, kb)
    phon = set()
    for pair in rec["state"].get("phonetic_complement_pairs") or []:
        si = legacy_position_to_sign_index(legacy, int(pair[1]))
        if si is not None:
            phon.add(si)

    def touch(i: int) -> Candidate:
        if i not in pool:
            pool[i] = Candidate(sign=st.signs[i], index=i, sources=frozenset(),
                                is_final=i in st.block_final,
                                is_global_final=st.final_sign is not None and i == st.final_sign,
                                is_group_end=i in st.group_ends, is_near_end=i in st.near_end,
                                is_plural_marker=i in st.plural_markers, is_plural_head=i in st.plural_heads)
        return pool[i]

    for src, attr in (("semantic", "conf_sem"), ("structural", "conf_struct")):
        if src not in sources:
            continue
        props, _ = agent_proposals(agent_json(rec, src)[0], legacy)
        for i, conf in props:
            c = touch(i)
            c.sources = c.sources | {src}
            setattr(c, attr, max(getattr(c, attr), conf))
    for i, c in pool.items():
        c.is_phonetic_complement_likely = i in phon
    return pool


def predict_conditions(run_dir: str, params_run: Optional[str] = None, extra: Optional[Dict[str, str]] = None) -> tuple:
    """Predictions of every condition: (preds[condition][id] -> sign indices, gold by id, records, diagnostics)."""
    kb = load_kb()
    gold = {r["id"]: r for r in load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl"))}
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    best = fold_best_params(params_run or os.path.join(ROOT, cfg["params_run"]))
    recs = load_records(run_dir)

    preds: Dict[str, Dict[str, List[int]]] = defaultdict(dict)   # condition -> id -> sign indices
    diag = Counter()
    for rec in recs:
        g = gold[rec["id"]]
        st = build_structure(rec["mdc"], kb.block_end_signs, kb.plural_markers)
        p = best[rec["fold"]]
        legacy = tokenize_with_separators(rec["mdc"])
        preds["rules_only"][rec["id"]] = select(st, rule_candidates(st, kb), p, kb)
        preds["v147_full"][rec["id"]] = [i for i in rec["final_idx"] if i is not None]
        for name, agent in (("semantic_raw", "semantic"), ("structural_raw", "structural")):
            js, origin = agent_json(rec, agent)
            props, bad = agent_proposals(js, legacy)
            preds[name][rec["id"]] = sorted({i for i, _ in props})
            diag[f"{name}.invalid_proposals"] += bad
            diag[f"{name}.valid_proposals"] += len(props)
            diag[f"{name}.json_{origin}"] += 1
        for sname, sources in SOURCE_SETS.items():
            for wname, w in WEIGHTS.items():
                pw = dict(p, **w)
                pool = build_pool(st, rec, kb, sources)
                preds[f"scorer[{sname}|w={wname}]"][rec["id"]] = select(st, pool, pw, kb)
        diag["records"] += 1
        diag["records_failed"] += int(not rec["success"])
        diag["records_with_llm_errors"] += int(rec["calls"]["errors"] > 0)
        diag["records_with_cache_misses"] += int(rec["calls"]["misses"] > 0)

    for name, other in (extra or {}).items():   # final outputs of other collect runs, e.g. fake-LLM mode
        ids = set(preds["rules_only"])
        for orec in load_records(other):
            if orec["id"] in ids:
                preds[f"v147_full[{name}]"][orec["id"]] = [i for i in orec["final_idx"] if i is not None]

    return preds, gold, recs, diag


def evaluate(run_dir: str, params_run: Optional[str] = None, extra: Optional[Dict[str, str]] = None) -> dict:
    preds, gold, recs, diag = predict_conditions(run_dir, params_run, extra)
    folds = sorted({r["fold"] for r in recs})
    fold_of = {r["id"]: r["fold"] for r in recs}
    table = {}
    for cond, by_id in preds.items():
        per_fold = {"positional": defaultdict(list), "signs": defaultdict(list)}
        pooled_pos, pooled_sig = [], []
        for sid, idx in by_id.items():
            g = gold[sid]
            sp = score_positions(idx, g["det_idx"])
            ss = score_signs([g["tokens"][i] for i in idx], g["det_signs"])
            per_fold["positional"][fold_of[sid]].append(sp)
            per_fold["signs"][fold_of[sid]].append(ss)
            pooled_pos.append(sp)
            pooled_sig.append(ss)
        row = {"n": len(by_id)}
        for crit in ("positional", "signs"):
            aggs = [aggregate(per_fold[crit][k]) for k in folds if per_fold[crit][k]]
            row[crit] = {"macro_f1": summarize_folds([a["macro"]["f1"] for a in aggs]),
                         "micro_f1": summarize_folds([a["micro"]["f1"] for a in aggs])}
        pooled = aggregate(pooled_pos)
        row["pooled_positional"] = {"tp": pooled["micro"]["tp"], "fp": pooled["micro"]["fp"],
                                    "fn": pooled["micro"]["fn"], "outcomes": pooled["outcomes"]}
        table[cond] = row
    return {"run": _rel(run_dir), "folds": folds, "diagnostics": dict(diag), "conditions": table}


def to_markdown(res: dict) -> str:
    lines = [f"# Phase 1 ablation ({res['run']})", "",
             f"Sequences: {res['diagnostics'].get('records', 0)}; folds: {res['folds']}. "
             "Mean ± sample SD over outer folds; TP/FP/FN pooled over all sequences (positional).", "",
             "| Condition | Macro-F1 pos. | Micro-F1 pos. | Macro-F1 signs | TP | FP | FN | FP_NULL | FN_NULL |",
             "|---|---|---|---|---|---|---|---|---|"]

    def fmt(s):
        sd = s["std_sample"]
        return f"{100 * s['mean']:.2f} ± {100 * sd:.2f}" if sd is not None else f"{100 * s['mean']:.2f}"

    for cond, row in res["conditions"].items():
        pp = row["pooled_positional"]
        lines.append(f"| {cond} | {fmt(row['positional']['macro_f1'])} | {fmt(row['positional']['micro_f1'])} | "
                     f"{fmt(row['signs']['macro_f1'])} | {pp['tp']} | {pp['fp']} | {pp['fn']} | "
                     f"{pp['outcomes']['FP_NULL']} | {pp['outcomes']['FN_NULL']} |")
    lines += ["", "## Diagnostics", ""] + [f"- {k}: {v}" for k, v in sorted(res["diagnostics"].items())]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Offline ablation over collected v1.47 agent outputs")
    ap.add_argument("run_dir")
    ap.add_argument("--params-run", default=None)
    ap.add_argument("--extra", nargs="*", default=[], metavar="NAME=RUN_DIR",
                    help="add the final output of another collect run as a condition")
    args = ap.parse_args(argv)
    extra = dict(e.split("=", 1) for e in args.extra)
    res = evaluate(args.run_dir, args.params_run, extra)
    with open(os.path.join(args.run_dir, "ablation.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    md = to_markdown(res)
    with open(os.path.join(args.run_dir, "ablation.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
