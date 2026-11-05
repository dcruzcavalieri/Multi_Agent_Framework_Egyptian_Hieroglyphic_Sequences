"""Phase 1 data collection: run the v1.47 multi-agent pipeline once per annotated sequence.

Each sequence is processed as a test item of its outer fold, exactly as in the v1.47 nested CV:
few-shot examples are chosen (Jaccard similarity) from that fold's training sequences only, and the
agent guidance is built from that fold's optimised scorer parameters (taken from a rule-only run of
``python -m egydet.eval.cv``). Every LLM answer is cached, so the run can be interrupted and resumed,
and every later analysis (``egydet.legacy_bridge.ablation``) works offline from the stored records.

    python -m egydet.legacy_bridge.collect --limit 2            # pilot: 2 sequences per fold
    python -m egydet.legacy_bridge.collect                      # all 99 sequences (about 300 calls)
    python -m egydet.legacy_bridge.collect --mode offline       # rebuild records from the cache only
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional

from egydet.eval.splits import load_splits
from egydet.gold.io import load_gold
from egydet.kb import ROOT
from egydet.llm import CallStats, DiskCache
from egydet.mdc import normalize, tokenize_with_separators
from egydet.eval.metrics import legacy_position_to_sign_index

STATE_KEYS = ("semantic_json", "structural_json", "synthesizer_json", "semantic_analysis",
              "structural_analysis", "consolidated_analysis", "phonetic_complement_pairs")


def _rel(path: str) -> str:
    """Path relative to the repository when possible (on Windows a different drive makes relpath fail)."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return os.path.abspath(path)


AGENT_MARKERS = {"semantic": "SEMANTIC analysis", "structural": "STRUCTURAL analysis", "synthesizer": "FINAL ARBITER"}


def _agent_of(system_head: str) -> str:
    return next((a for a, mk in AGENT_MARKERS.items() if mk in system_head), "other")


def latest_run(pattern: str = "*_rule_only_legacy_micro_signs") -> str:
    runs = sorted(glob.glob(os.path.join(ROOT, "runs", pattern)))
    if not runs:
        raise FileNotFoundError("no rule-only run found; run `python -m egydet.eval.cv` first")
    return runs[-1]


def fold_best_params(params_run: str) -> Dict[int, dict]:
    with open(os.path.join(params_run, "folds.json"), encoding="utf-8") as fh:
        folds = json.load(fh)
    out = {}
    for f in folds:
        rec = next(r for r in f["inner"] if r["label"] == f["best_label"])
        out[int(f["fold"])] = rec["params"]
    return out


def _to_sign_idx(legacy_tokens: List[str], pos) -> Optional[int]:
    try:
        return legacy_position_to_sign_index(legacy_tokens, int(pos))
    except (TypeError, ValueError):
        return None


def _jsonable(x):
    if isinstance(x, set):
        return sorted(list(t) if isinstance(t, tuple) else t for t in x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def run(mode: str, params_run: str, out_dir: str, folds: Optional[List[int]], limit: Optional[int],
        env_file: Optional[str]) -> str:
    from egydet.legacy_bridge.v147 import load_v147

    os.makedirs(out_dir, exist_ok=True)
    stats = CallStats()
    v147 = load_v147(mode=mode, cache=DiskCache(), stats=stats, env_file=env_file,
                     log_path=os.path.join(out_dir, "v147_stdout.log"))
    gold = sorted(load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl")), key=lambda r: r["corpus_row"])
    pos_of = {r["id"]: k for k, r in enumerate(gold)}
    loo_df = v147.LOO_DF
    if [normalize(str(c)) for c in loo_df["Code"]] != [r["mdc"] for r in gold]:
        raise RuntimeError("v1.47 LOO_DF order differs from gold_v1 corpus order")
    best = fold_best_params(params_run)
    splits = load_splits()

    rec_path = os.path.join(out_dir, "records.jsonl")
    done = set()
    if os.path.exists(rec_path):
        with open(rec_path, encoding="utf-8") as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}

    meta = {"mode": mode, "params_run": _rel(params_run), "splits": splits["name"],
            "started": _dt.datetime.now().isoformat(timespec="seconds"),
            "model": "gpt-5.4-nano (as hard-coded in v1.47)", "n_few_shot": 5}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    with open(rec_path, "a", encoding="utf-8") as out:
        for f in splits["folds"]:
            k = f["fold"]
            if folds and k not in folds:
                continue
            p = best[k]
            v147.apply_optimized_params(p)
            v147._apply_params_to_score_candidate()
            train_val_df = loo_df.iloc[[pos_of[i] for i in f["train_val"]]].reset_index(drop=True)
            test_ids = f["test"][:limit] if limit else f["test"]
            for test_i, sid in enumerate(test_ids):
                if sid in done:
                    continue
                r = gold[pos_of[sid]]
                seq = str(loo_df.iloc[pos_of[sid]]["Code"]).strip()
                few = v147.select_expert_examples_loo(train_val_df, n_examples=5, test_sequence=seq)
                before = stats.snapshot()
                n_answers = len(stats.answers)
                t0 = time.time()
                with open(os.path.join(out_dir, "v147_stdout.log"), "a", encoding="utf-8") as log, \
                        contextlib.redirect_stdout(log):
                    result = v147.analyze_with_langgraph(seq, f"test_fold{k - 1}_{test_i}",
                                                         expert_examples=few, optimised_params=p)
                after = stats.snapshot()
                state = v147.LANGGRAPH_APP.last_state or {}
                legacy_toks = tokenize_with_separators(seq)
                final = result.get("determinatives", []) if result.get("success") else []
                final_idx = [_to_sign_idx(legacy_toks, d.get("position")) for d in final if isinstance(d, dict)]
                record = {
                    "id": sid, "fold": k, "mdc": r["mdc"], "success": bool(result.get("success")),
                    "error": result.get("error", ""),
                    "few_shot_mdc": [str(e[0]) for e in few],   # (MdC, translation) pairs in v1.47
                    "final_signs": [d.get("sign") for d in final if isinstance(d, dict)],
                    "final_idx": final_idx,
                    "state": {key: _jsonable(state.get(key)) for key in STATE_KEYS},
                    "calls": {key: after[key] - before[key] for key in after},
                    "raw_answers": {_agent_of(a["system_head"]): {"content": a["content"], "cache_key": a["key"]}
                                    for a in stats.answers[n_answers:]},
                    "seconds": round(time.time() - t0, 2),
                }
                out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                out.flush()
                c = record["calls"]
                flag = "" if record["success"] and not c["errors"] and not c["misses"] else "  <-- check"
                print(f"fold {k} [{test_i + 1}/{len(test_ids)}] {sid}: api={c['api_calls']} cached={c['cache_hits']} "
                      f"errors={c['errors']} misses={c['misses']} pred={record['final_signs']}{flag}")

    s = stats.snapshot()
    summary = dict(meta, finished=_dt.datetime.now().isoformat(timespec="seconds"), **s,
                   error_messages=stats.errors[:20])
    with open(os.path.join(out_dir, "collect_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\n{s}\nrecords: {rec_path}")
    return out_dir


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Collect v1.47 agent outputs for the 99 annotated sequences")
    ap.add_argument("--mode", choices=("live", "offline", "fake"), default="live")
    ap.add_argument("--params-run", default=None, help="rule-only run dir (default: latest legacy-objective run)")
    ap.add_argument("--out", default=None, help="output run dir (reuse an existing one to resume)")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--limit", type=int, default=None, help="sequences per fold (pilot runs)")
    ap.add_argument("--env-file", default=None)
    args = ap.parse_args(argv)
    params_run = args.params_run or latest_run()
    out = args.out or os.path.join(ROOT, "runs", _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_llm_v147_{args.mode}")
    run(args.mode, params_run, out, args.folds, args.limit, args.env_file)


if __name__ == "__main__":
    sys.exit(main())
