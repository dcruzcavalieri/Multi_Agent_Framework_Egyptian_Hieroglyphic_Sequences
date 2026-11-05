"""Phase 2, option A: verifier with annotated few-shot examples retrieved from the training fold.

For outer fold k, every sequence of the fold's training partition and of its test partition is judged
with ``m`` annotated examples taken from the training partition only (never the sequence itself, never a
test sequence of fold k). Test outputs therefore see no test labels; the training outputs are what the
inner CV uses to tune the fusion. (Inside the inner CV a training-half sequence may have validation-half
examples; this touches only the inner choice of fusion parameters, never the outer test estimate.)

Examples are the ``m`` training sequences with the highest Jaccard overlap of sign vocabulary with the
sequence to judge (ties broken by corpus order), as in v1.47, but shown WITH their gold determinatives.

    python -m egydet.agents.fewshot --folds 1 --limit 5      # smoke test (5 calls)
    python -m egydet.agents.fewshot                          # 5 folds x 99 sequences = 495 calls
    python -m egydet.eval.fusion runs/phase2_verifier_fewshot_signs_gpt-5.4-nano
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from typing import Dict, List, Optional

from egydet.agents.client import ChatClient
from egydet.agents.verifier import DEFAULT_MODEL, SCHEMA, parse_judgements, system_prompt, user_prompt
from egydet.eval.splits import load_splits
from egydet.gold.io import load_gold
from egydet.kb import ROOT, load_kb
from egydet.legacy_bridge.v147 import load_env
from egydet.llm import CallStats, DiskCache

PROMPT_VERSION = "verifier_v2"
FILE = "verifier_folded.jsonl"


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def select_examples(target: dict, pool: List[dict], m: int) -> List[dict]:
    cands = [r for r in pool if r["id"] != target["id"]]
    cands.sort(key=lambda r: (-jaccard(target["tokens"], r["tokens"]), r["corpus_row"]))
    return cands[:m]


def run(mode: str, model: str, out_dir: str, m: int = 5, folds: Optional[List[int]] = None,
        limit: Optional[int] = None, condition: str = "signs", env_file: Optional[str] = None,
        fake_answer=None, test_only: bool = False) -> str:
    if mode == "live":
        load_env(env_file)
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not found")
    kb = load_kb()
    gold = {r["id"]: r for r in load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl"))}
    splits = load_splits()
    stats = CallStats()
    client = ChatClient(model, mode=mode, cache=DiskCache(), stats=stats, fake_answer=fake_answer)
    system = system_prompt(PROMPT_VERSION)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, FILE)
    done = set()
    if os.path.exists(path):
        done = {(json.loads(l)["fold"], json.loads(l)["id"]) for l in open(path, encoding="utf-8") if l.strip()}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"agent": "verifier+fewshot", "prompt": PROMPT_VERSION, "condition": condition, "model": model,
                   "mode": mode, "m_examples": m, "splits": splits["name"],
                   "started": _dt.datetime.now().isoformat(timespec="seconds")}, fh, indent=1)
    with open(path, "a", encoding="utf-8") as out:
        for f in splits["folds"]:
            k = f["fold"]
            if folds and k not in folds:
                continue
            train = [gold[i] for i in f["train_val"]]
            todo = [(i, "test") for i in f["test"]] + ([] if test_only else [(i, "train") for i in f["train_val"]])
            if limit:
                todo = todo[:limit]
            test_ids = set(f["test"])
            for sid, role in todo:
                if (k, sid) in done:
                    continue
                r = gold[sid]
                ex = select_examples(r, train, m)
                assert not any(e["id"] in test_ids for e in ex), "test sequence used as example"
                before = stats.snapshot()
                error = ""
                try:
                    answer = client.complete_json(system, user_prompt(r, condition, kb, examples=ex), SCHEMA,
                                                  "determinative_judgements")
                except Exception as exc:
                    answer, error = {}, f"{type(exc).__name__}: {str(exc)[:300]}"
                judg = parse_judgements(answer, r)
                after = stats.snapshot()
                rec = {"fold": k, "id": sid, "role": role, "examples": [e["id"] for e in ex],
                       "n_signs": len(r["tokens"]), "n_judged": len(judg), "error": error,
                       "judgements": {str(i): v for i, v in sorted(judg.items())},
                       "calls": {kk: after[kk] - before[kk] for kk in after}}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                flag = "" if not error and len(judg) == len(r["tokens"]) else "  <-- check"
                print(f"fold {k} {role:5s} {sid}: judged {len(judg)}/{len(r['tokens'])} "
                      f"api={rec['calls']['api_calls']} cached={rec['calls']['cache_hits']} "
                      f"errors={rec['calls']['errors']}{flag}")
    s = stats.snapshot()
    with open(os.path.join(out_dir, "collect_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(s, error_messages=stats.errors[:20]), fh, indent=1)
    print(s)
    return out_dir


def load_folded(run_dir: str) -> Dict[int, Dict[str, Dict[int, float]]]:
    """fold -> sequence id -> index -> p (last record wins)."""
    out: Dict[int, Dict[str, Dict[int, float]]] = {}
    for line in open(os.path.join(run_dir, FILE), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out.setdefault(int(r["fold"]), {})[r["id"]] = {int(i): v["p"] for i, v in r["judgements"].items()}
    return out


def test_outputs(run_dir: str) -> Dict[str, Dict[int, float]]:
    """Each sequence judged as a test item of its own fold (for diagnostics)."""
    test_fold = {i: f["fold"] for f in load_splits()["folds"] for i in f["test"]}
    folded = load_folded(run_dir)
    return {sid: folded[k][sid] for sid, k in test_fold.items() if k in folded and sid in folded[k]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Verifier with annotated few-shot examples from the training fold")
    ap.add_argument("--mode", choices=("live", "offline", "fake"), default="live")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--m", type=int, default=5, help="examples per call")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--limit", type=int, default=None, help="calls per fold (smoke test)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--test-only", action="store_true",
                    help="judge only the test items (99 calls): enough for llm_only and diagnostics, not for fusion")
    args = ap.parse_args(argv)
    out = args.out or os.path.join(ROOT, "runs", f"phase2_verifier_fewshot_signs_{args.model}")
    run(args.mode, args.model, out, args.m, args.folds, args.limit, env_file=args.env_file, test_only=args.test_only)


if __name__ == "__main__":
    main()
