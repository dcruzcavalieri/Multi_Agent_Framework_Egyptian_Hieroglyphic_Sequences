"""Per-position verifier agent (Phase 2, prompt ``prompts/verifier_v1.md``).

One call per sequence. The agent receives the signs with explicit sign indices (no separators), the
knowledge-base description of the signs it contains, and, in the ``context`` condition only, the
transliteration and translation (an explicitly labelled oracle condition: an OCR pipeline would not
have them). It returns a probability of being a determinative for every position.

The prompt contains no sequence from the corpus and no few-shot example, so its output does not depend
on the cross-validation fold: one run serves every fold, and the fusion weights can be tuned inside the
inner CV without leakage.

    python -m egydet.agents.verifier --condition signs   --limit 5    # smoke test
    python -m egydet.agents.verifier --model <model> --pilot          # 30-sequence diagnostic pilot
    python -m egydet.agents.verifier --condition signs
    python -m egydet.agents.verifier --condition context
    python -m egydet.agents.verifier --lexicon            # + sign-list notes (prompt verifier_v3)
    python -m egydet.agents.verifier --condition context --reading runs/phase2_reader_<model>   # no oracle
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from typing import Dict, List, Optional

from egydet.agents.client import ChatClient
from egydet.gold.io import load_gold
from egydet.kb import ROOT, load_kb
from egydet.legacy_bridge.v147 import load_env
from egydet.llm import CallStats, DiskCache

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
PROMPT_VERSION = "verifier_v1"
DEFAULT_MODEL = "gpt-5.4-nano"
CONDITIONS = ("signs", "context")

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["judgements"],
    "properties": {
        "judgements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "sign", "probability", "reason"],
                "properties": {
                    "index": {"type": "integer"},
                    "sign": {"type": "string"},
                    "probability": {"type": "number"},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}


def system_prompt(version: str = PROMPT_VERSION) -> str:
    with open(os.path.join(PROMPT_DIR, f"{version}.md"), encoding="utf-8") as fh:
        return fh.read().strip()


def user_prompt(record: dict, condition: str, kb, examples: Optional[List[dict]] = None,
                lexicon: Optional[dict] = None) -> str:
    toks = record["tokens"]
    lines = []
    if examples:
        lines += ["ANNOTATED EXAMPLES (other sequences of the same corpus; determinatives in [brackets]):", ""]
        for k, ex in enumerate(examples, 1):
            marked = set(ex["det_idx"])
            lines.append(f"Example {k}: " + " ".join(f"[{t}]" if i in marked else t for i, t in enumerate(ex["tokens"])))
        lines += ["", "SEQUENCE TO JUDGE", ""]
    lines += [f"MdC: {record['mdc']}", "", "Signs (index: Gardiner code):"]
    lines += [f"{i}: {t}" for i, t in enumerate(toks)]
    known = [t for t in dict.fromkeys(toks) if t in kb.signs]
    if known:
        lines += ["", "Knowledge-base notes on signs that often act as determinatives (meaning; category; semantic field):"]
        for t in known:
            e = kb.signs[t]
            lines.append(f"- {t}: {e.get('meaning', '')}; {e.get('category', '')}; {e.get('semantics', '')}")
    if lexicon is not None:
        from egydet.agents.lexicon import lexicon_lines
        lines += ["", "Sign-list notes (description; listed uses):"] + lexicon_lines(toks, lexicon)
    if condition == "context":
        lines += ["", f"Transliteration: {record.get('transliteration', '')}", f"Translation: {record.get('translation', '')}"]
    lines += ["", f"Return one judgement for each index from 0 to {len(toks) - 1}."]
    return "\n".join(lines)


def parse_judgements(answer: dict, record: dict) -> Dict[int, dict]:
    """index -> {p, reason}; entries whose sign does not match the index are dropped."""
    out = {}
    for j in (answer or {}).get("judgements", []) or []:
        try:
            i = int(j["index"])
            p = float(j["probability"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= i < len(record["tokens"]) and str(j.get("sign", "")).strip() == record["tokens"][i]:
            out[i] = {"p": min(1.0, max(0.0, p)), "reason": str(j.get("reason", ""))[:300]}
    return out


def pilot_ids(rules_run: Optional[str] = None, n_fp: int = 6, n_clean: int = 5, seed: int = 7) -> List[str]:
    """Diagnostic pilot: every sequence where the rule scorer misses a determinative, plus a few with only
    false positives and a few without errors (seeded). Used to compare models cheaply, not to tune anything."""
    import random
    from egydet.eval.cv import prepare
    from egydet.eval.splits import load_splits
    from egydet.legacy_bridge.collect import fold_best_params, latest_run
    from egydet.scoring.scorer import select
    kb = load_kb()
    gold = sorted(load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl")), key=lambda r: r["corpus_row"])
    samples = prepare(gold, kb)
    best = fold_best_params(rules_run or latest_run())
    fold = {i: f["fold"] for f in load_splits()["folds"] for i in f["test"]}
    fn, fp_only, clean = [], [], []
    for r in gold:
        s = samples[r["id"]]
        pred, g = set(select(s.structure, s.pool, best[fold[r["id"]]], kb)), set(r["det_idx"])
        (fn if g - pred else fp_only if pred - g else clean).append(r["id"])
    rng = random.Random(seed)
    return sorted(fn + rng.sample(fp_only, min(n_fp, len(fp_only))) + rng.sample(clean, min(n_clean, len(clean))))


def run(condition: str, mode: str, model: str, out_dir: str, limit: Optional[int] = None,
        env_file: Optional[str] = None, fake_answer=None, ids: Optional[List[str]] = None,
        lexicon_path: Optional[str] = None, reading_run: Optional[str] = None) -> str:
    """``reading_run``: replace the experts' transliteration and translation (context condition) with the
    reading predicted by ``egydet.agents.reader``; the prompt text is otherwise identical."""
    if mode == "live":
        load_env(env_file)
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not found")
    kb = load_kb()
    gold = sorted(load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl")), key=lambda r: r["corpus_row"])
    if ids is not None:
        keep = set(ids)
        gold = [r for r in gold if r["id"] in keep]
    if limit:
        gold = gold[:limit]
    if reading_run:
        if condition != "context":
            raise ValueError("--reading only makes sense with --condition context")
        from egydet.agents.reader import load_reading
        reading = load_reading(reading_run)
        missing = [r["id"] for r in gold if r["id"] not in reading]
        if missing:
            raise RuntimeError(f"reading run lacks {len(missing)} sequences, e.g. {missing[:3]}")
        gold = [dict(r, transliteration=reading[r["id"]]["transliteration"],
                     translation=reading[r["id"]]["translation"]) for r in gold]
    stats = CallStats()
    client = ChatClient(model, mode=mode, cache=DiskCache(), stats=stats, fake_answer=fake_answer)
    lexicon, version, lex_sha1 = None, PROMPT_VERSION, None
    if lexicon_path:
        from egydet.agents.lexicon import lexicon_sha1, load_lexicon
        lexicon, version, lex_sha1 = load_lexicon(lexicon_path), "verifier_v3", lexicon_sha1(lexicon_path)
    system = system_prompt(version)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "verifier.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"agent": "verifier", "prompt": version, "condition": condition, "model": model,
                   "mode": mode, "lexicon_sha1": lex_sha1, "reading_run": reading_run,
                   "started": _dt.datetime.now().isoformat(timespec="seconds")}, fh, indent=1)
    with open(path, "a", encoding="utf-8") as out:
        for r in gold:
            if r["id"] in done:
                continue
            before = stats.snapshot()
            error = ""
            try:
                answer = client.complete_json(system, user_prompt(r, condition, kb, lexicon=lexicon), SCHEMA,
                                              "determinative_judgements")
            except Exception as exc:   # recorded, never silent
                answer, error = {}, f"{type(exc).__name__}: {str(exc)[:300]}"
            judg = parse_judgements(answer, r)
            after = stats.snapshot()
            rec = {"id": r["id"], "n_signs": len(r["tokens"]), "n_judged": len(judg), "error": error,
                   "judgements": {str(i): v for i, v in sorted(judg.items())},
                   "calls": {k: after[k] - before[k] for k in after}}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            flag = "" if not error and len(judg) == len(r["tokens"]) else "  <-- check"
            print(f"{r['id']}: judged {len(judg)}/{len(r['tokens'])} api={rec['calls']['api_calls']} "
                  f"cached={rec['calls']['cache_hits']} errors={rec['calls']['errors']}{flag}")
    s = stats.snapshot()
    with open(os.path.join(out_dir, "collect_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(s, error_messages=stats.errors[:20]), fh, indent=1)
    print(s)
    return out_dir


def load_verifier(run_dir: str) -> Dict[str, Dict[int, float]]:
    out = {}
    for line in open(os.path.join(run_dir, "verifier.jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = {int(i): v["p"] for i, v in r["judgements"].items()}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run the per-position verifier agent on the 99 annotated sequences")
    ap.add_argument("--condition", choices=CONDITIONS, default="signs")
    ap.add_argument("--mode", choices=("live", "offline", "fake"), default="live")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pilot", action="store_true",
                    help="only the diagnostic pilot set (all rule misses + 6 FP-only + 5 clean sequences)")
    ap.add_argument("--lexicon", nargs="?", const="default", default=None, metavar="CSV",
                    help="add sign-list notes (prompt verifier_v3); default file kb/private/lexicon_manual.csv")
    ap.add_argument("--reading", default=None, metavar="READER_RUN",
                    help="context condition with the reading predicted by egydet.agents.reader (no oracle)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--env-file", default=None)
    args = ap.parse_args(argv)
    lex = None
    if args.lexicon:
        from egydet.agents.lexicon import DEFAULT_PATH
        lex = DEFAULT_PATH if args.lexicon == "default" else args.lexicon
    tag = "lexicon_" if lex else ""
    cond = args.condition
    if args.reading:   # e.g. context_read-gpt-6-luna
        cond += "_read-" + os.path.basename(os.path.normpath(args.reading)).replace("phase2_reader_", "")
    out = args.out or os.path.join(ROOT, "runs", f"phase2_verifier_{tag}{cond}_{args.model}"
                                   + ("_pilot" if args.pilot else ""))
    run(args.condition, args.mode, args.model, out, args.limit, args.env_file,
        ids=pilot_ids() if args.pilot else None, lexicon_path=lex, reading_run=args.reading)


if __name__ == "__main__":
    main()
