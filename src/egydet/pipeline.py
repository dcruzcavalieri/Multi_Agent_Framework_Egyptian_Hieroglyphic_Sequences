"""Frozen v2 pipeline: rules + phonetic alignment veto + determinative slots confirmed by the verifier.

Setting: annotation assistant. Input per sequence: MdC, the experts' transliteration and translation.

1. Rules: the rule scorer (``egydet.scoring``) with one parameter set tuned on all 99 annotated
   sequences (same objective, sampler and inner 5x2 procedure as the cross-validation).
2. Veto: a rule determinative is removed when the alignment of the signs with the transliteration
   (``egydet.rules.align``, JSesh phonogram values) shows that the sign writes letters of its word or is a
   uniliteral complement.
3. Rescue: an alignment slot (an unexplained sign at the end of a word) is added when the verifier agent
   (prompt ``verifier_v1``, context condition, gpt-6-luna) gives it p >= 0.5.

    python -m egydet.pipeline freeze                        # writes configs/pipeline/frozen_v2.json (once)
    python -m egydet.pipeline run --corpus-unannotated --out runs/frozen_v2_pool       # 434 sequences
    python -m egydet.pipeline run --input new.xlsx --out runs/frozen_v2_new            # id, mdc, transliteration, translation
    python -m egydet.pipeline run --gold data/gold/gold_v1.jsonl --mode offline --out runs/frozen_v2_gold
    python -m egydet.pipeline evaluate runs/frozen_v2_new --gold data/gold/<new gold>.jsonl

The freeze file records the rule parameters and the SHA-1 of every input that defines the pipeline;
``run`` refuses to start if any of them changed. Numbers on the 99 sequences used to build the pipeline
are in-sample; only new annotated sequences give a valid estimate.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import random
from typing import Dict, List, Optional

from egydet.kb import ROOT, load_kb

FREEZE_PATH = os.path.join(ROOT, "configs", "pipeline", "frozen_v2.json")
GOLD_V1 = os.path.join(ROOT, "data", "gold", "gold_v1.jsonl")
MODEL = "gpt-6-luna"
PROMPT = "verifier_v1"
CONDITION = "context"
THETA = 0.5
OBJECTIVE = "legacy_micro_signs"
N_TRIALS = 400
FINAL_SEED = 42

# files whose content defines the pipeline (relative to the repository root)
DEFINING_FILES = [
    "kb/signs.yaml", "kb/categories.yaml", "kb/jsesh_sign_values.json",
    "src/egydet/agents/prompts/verifier_v1.md",
    "src/egydet/rules/align.py", "src/egydet/rules/structure.py", "src/egydet/rules/baseline.py",
    "src/egydet/scoring/scorer.py", "src/egydet/scoring/params.py", "src/egydet/pipeline.py",
]


def _sha1(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def fingerprint() -> Dict[str, str]:
    return {f: _sha1(os.path.join(ROOT, f)) for f in DEFINING_FILES}


# ── freeze ─────────────────────────────────────────────────────────────────

def final_rule_params(log=print) -> dict:
    """Tune the rule scorer on all 99 sequences with the inner procedure of the CV: 5 random halvings
    (seed 42), Optuna on each half (seed rep*100+len(label), flat warm start), best validation wins."""
    from egydet.eval.cv import OBJECTIVES, optimise, prepare
    from egydet.gold.io import load_gold
    from egydet.scoring.params import preset
    kb = load_kb()
    samples = prepare(load_gold(GOLD_V1), kb)
    ids = sorted(samples)
    objective, init = OBJECTIVES[OBJECTIVE], preset("flat")
    rng = random.Random(FINAL_SEED)
    inner = []
    for rep in range(5):
        perm = ids[:]
        rng.shuffle(perm)
        A, B = [samples[i] for i in perm[: len(perm) // 2]], [samples[i] for i in perm[len(perm) // 2:]]
        for train, val, label in ((A, B, f"rep{rep + 1}_AB"), (B, A, f"rep{rep + 1}_BA")):
            best, opt_v, init_v, _ = optimise(train, objective, kb, N_TRIALS, rep * 100 + len(label), init)
            v = objective(val, best, kb)
            inner.append({"label": label, "opt": opt_v, "val": v, "params": best})
            log(f"  {label}: train {opt_v:.4f} val {v:.4f}")
    top = max(inner, key=lambda r: r["val"])
    return {"params": top["params"], "chosen": top["label"], "inner": [{k: r[k] for k in ("label", "opt", "val")}
                                                                          for r in inner]}


def freeze(force: bool = False) -> str:
    if os.path.exists(FREEZE_PATH) and not force:
        raise RuntimeError(f"{FREEZE_PATH} exists: the pipeline is already frozen (use --force to redo it)")
    from egydet.eval.cv import _git_hash
    rp = final_rule_params()
    rec = {"name": "frozen_v2", "created": _dt.datetime.now().isoformat(timespec="seconds"), "git": _git_hash(),
           "setting": "annotation assistant: MdC + experts' transliteration and translation",
           "rules": {"objective": OBJECTIVE, "n_trials": N_TRIALS, "trained_on": "gold_v1 (99 sequences)",
                     "gold_sha1": _sha1(GOLD_V1), **rp},
           "veto": {"table": "kb/jsesh_sign_values.json", "types": ["phonogram"]},
           "verifier": {"model": MODEL, "prompt": PROMPT, "condition": CONDITION},
           "rescue": {"theta": THETA}, "fingerprint": fingerprint()}
    os.makedirs(os.path.dirname(FREEZE_PATH), exist_ok=True)
    with open(FREEZE_PATH, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return FREEZE_PATH


def load_frozen(check: bool = True) -> dict:
    with open(FREEZE_PATH, encoding="utf-8") as fh:
        rec = json.load(fh)
    if check:
        now = fingerprint()
        changed = [f for f, h in rec["fingerprint"].items() if now.get(f) != h]
        if changed:
            raise RuntimeError(f"pipeline inputs changed since the freeze: {changed}")
    return rec


# ── run ────────────────────────────────────────────────────────────────────

def load_records(input_path: Optional[str] = None, gold_path: Optional[str] = None,
                 corpus_unannotated: bool = False) -> List[dict]:
    """Records with id, mdc, tokens, transliteration, translation (gold records keep det_idx)."""
    from egydet.mdc.tokenizer import normalize, tokenize
    if gold_path:
        from egydet.gold.io import load_gold
        return load_gold(gold_path)
    if corpus_unannotated:
        from egydet.gold.build import load_corpus
        from egydet.gold.io import load_gold
        annotated = {r["id"] for r in load_gold(GOLD_V1)}
        df = load_corpus(os.path.join(ROOT, "data", "raw", "extractrion_for_agents.xlsx"))
        rows = [r for _, r in df.iterrows() if r["id"] not in annotated]
    else:
        import pandas as pd
        df = pd.read_excel(input_path) if input_path.endswith((".xlsx", ".xls")) else pd.read_csv(input_path)
        missing = {"id", "mdc", "transliteration", "translation"} - set(df.columns)
        if missing:
            raise ValueError(f"input lacks columns {sorted(missing)}")
        rows = [r for _, r in df.iterrows()]
    out = []
    for r in rows:
        mdc = normalize(str(r["mdc"]))
        out.append({"id": str(r["id"]), "mdc": mdc, "tokens": [t.text for t in tokenize(mdc)],
                    "transliteration": str(r["transliteration"]), "translation": str(r["translation"])})
    return out


def predict(records: List[dict], frozen: dict, mode: str, out_dir: str, env_file: Optional[str] = None) -> str:
    from egydet.agents.client import ChatClient
    from egydet.agents.verifier import SCHEMA, parse_judgements, system_prompt, user_prompt
    from egydet.legacy_bridge.v147 import load_env
    from egydet.llm import CallStats, DiskCache
    from egydet.rules.align import align, load_table
    from egydet.rules.baseline import rule_candidates
    from egydet.rules.structure import build_structure
    from egydet.scoring.scorer import select

    if mode == "live":
        load_env(env_file)
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not found")
    kb = load_kb()
    params = frozen["rules"]["params"]
    table = load_table(os.path.join(ROOT, frozen["veto"]["table"]), types=frozen["veto"]["types"])
    theta = frozen["rescue"]["theta"]
    stats = CallStats()
    client = ChatClient(frozen["verifier"]["model"], mode=mode, cache=DiskCache(), stats=stats)
    system = system_prompt(frozen["verifier"]["prompt"])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictions.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"pipeline": frozen["name"], "freeze_created": frozen["created"], "mode": mode,
                   "n_records": len(records), "started": _dt.datetime.now().isoformat(timespec="seconds")},
                  fh, indent=1)
    with open(path, "a", encoding="utf-8") as out:
        for r in records:
            if r["id"] in done:
                continue
            st = build_structure(r["mdc"], kb.block_end_signs, kb.plural_markers)
            rules = sorted(select(st, rule_candidates(st, kb), params, kb))
            al = align(r["tokens"], r["transliteration"], table)
            phon = {i for i, (ro, v) in enumerate(zip(al["roles"], al["values"]))
                    if ro == "phon" or (ro == "compl" and len(v) == 1)}
            error = ""
            try:
                ans = client.complete_json(system, user_prompt(r, frozen["verifier"]["condition"], kb), SCHEMA,
                                           "determinative_judgements")
            except Exception as exc:   # recorded; the sequence then gets rules + veto only
                ans, error = {}, f"{type(exc).__name__}: {str(exc)[:300]}"
            judg = parse_judgements(ans, r)
            probs = {i: v["p"] for i, v in judg.items()}
            kept = [i for i in rules if i not in phon]
            rescued = sorted(i for i in al["slots"] if probs.get(i, 0.0) >= theta and i not in kept)
            final = sorted(set(kept) | set(rescued))
            rec = {"id": r["id"], "mdc": r["mdc"], "tokens": r["tokens"], "det_idx": final,
                   "det_signs": [r["tokens"][i] for i in final],
                   "evidence": {"rules": rules, "vetoed": sorted(set(rules) - set(kept)), "slots": al["slots"],
                                "rescued": rescued, "p": {str(i): round(p, 3) for i, p in sorted(probs.items())},
                                "reasons": {str(i): v["reason"] for i, v in sorted(judg.items()) if i in final},
                                "words": [[a, b] for a, b in al["spans"]], "alignment_ok": al["ok"],
                                "verifier_judged": f"{len(judg)}/{len(r['tokens'])}"},
                   "error": error}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            flag = "" if not error and len(judg) == len(r["tokens"]) else "  <-- check"
            print(f"{r['id']}: {' '.join(rec['det_signs']) or '(none)'}{flag}")
    s = stats.snapshot()
    with open(os.path.join(out_dir, "collect_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(s, error_messages=stats.errors[:20]), fh, indent=1)
    print(s)
    return out_dir


def write_review_sheet(out_dir: str) -> str:
    """One row per sequence for the experts: MdC with the proposed determinatives in [brackets]."""
    import pandas as pd
    rows = []
    for line in open(os.path.join(out_dir, "predictions.jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            marked = " ".join(f"[{t}]" if i in r["det_idx"] else t for i, t in enumerate(r["tokens"]))
            rows.append({"id": r["id"], "MdC": r["mdc"], "proposal": marked,
                         "determinatives": " ".join(r["det_signs"]), "check": "x" if r["error"] else ""})
    path = os.path.join(out_dir, "proposals.xlsx")
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


# ── evaluate ───────────────────────────────────────────────────────────────

def evaluate(out_dir: str, gold_path: str) -> dict:
    from egydet.eval.metrics import aggregate, score_positions
    from egydet.gold.io import load_gold
    from egydet.legacy_bridge.analysis import paired_comparisons
    gold = {r["id"]: r for r in load_gold(gold_path)}
    preds = {"rules_only": {}, "rules+veto": {}, "frozen_v2": {}}
    for line in open(os.path.join(out_dir, "predictions.jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r["id"] in gold:
                ev = r["evidence"]
                preds["rules_only"][r["id"]] = ev["rules"]
                preds["rules+veto"][r["id"]] = sorted(set(ev["rules"]) - set(ev["vetoed"]))
                preds["frozen_v2"][r["id"]] = r["det_idx"]
    ids = sorted(preds["frozen_v2"])
    res = {"n": len(ids), "gold": gold_path, "conditions": {}}
    for c, by_id in preds.items():
        ag = aggregate([score_positions(by_id[i], gold[i]["det_idx"]) for i in ids])
        res["conditions"][c] = {"macro_f1": ag["macro"]["f1"], "micro_f1": ag["micro"]["f1"],
                                "tp": ag["micro"]["tp"], "fp": ag["micro"]["fp"], "fn": ag["micro"]["fn"]}
    res["paired"] = paired_comparisons(preds, {i: gold[i] for i in ids})
    with open(os.path.join(out_dir, "evaluation.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for c, v in res["conditions"].items():
        print(f"{c:12s} macro {100 * v['macro_f1']:.2f} micro {100 * v['micro_f1']:.2f} "
              f"TP/FP/FN {v['tp']}/{v['fp']}/{v['fn']}")
    for c in res["paired"]:
        print(f"{c['condition']} vs rules_only: {100 * c['macro_diff']:+.2f} pp "
              f"[{100 * c['macro_ci'][0]:+.2f}, {100 * c['macro_ci'][1]:+.2f}] {c['better']}/{c['worse']}")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Frozen v2 pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="xlsx/csv with id, mdc, transliteration, translation")
    g.add_argument("--gold", help="gold jsonl (e.g. to re-run the 99 offline from the cache)")
    g.add_argument("--corpus-unannotated", action="store_true", help="the corpus sequences not in gold_v1")
    r.add_argument("--out", required=True)
    r.add_argument("--mode", choices=("live", "offline", "fake"), default="live")
    r.add_argument("--env-file", default=None)
    e = sub.add_parser("evaluate")
    e.add_argument("run_dir")
    e.add_argument("--gold", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "freeze":
        print("frozen ->", freeze(args.force))
    elif args.cmd == "run":
        frozen = load_frozen()
        recs = load_records(args.input, args.gold, args.corpus_unannotated)
        out = predict(recs, frozen, args.mode, args.out, args.env_file)
        print("review sheet ->", write_review_sheet(out))
    else:
        evaluate(args.run_dir, args.gold)


if __name__ == "__main__":
    main()
