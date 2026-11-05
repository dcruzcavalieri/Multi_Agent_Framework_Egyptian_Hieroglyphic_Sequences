"""Reader agent (Phase 2): MdC -> words, transliteration and translation, without any oracle.

The verifier's ``context`` condition uses the experts' transliteration and translation (an oracle). This
agent produces them from the signs alone, so that the same verifier can be run on a predicted reading:

    python -m egydet.agents.reader --model gpt-6-luna --limit 3          # smoke test
    python -m egydet.agents.reader --model gpt-6-luna                    # 99 calls
    python -m egydet.agents.reader --evaluate runs/phase2_reader_gpt-6-luna
    python -m egydet.agents.verifier --condition context --model gpt-6-luna \
        --reading runs/phase2_reader_gpt-6-luna
    python -m egydet.eval.rerank runs/phase2_verifier_context_gpt-6-luna \
        runs/phase2_verifier_context_read-gpt-6-luna_gpt-6-luna

The prompt contains no corpus sequence and asks nothing about determinatives; its output does not depend
on the cross-validation fold.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import unicodedata
from statistics import mean, median
from typing import Dict, Optional

from egydet.agents.client import ChatClient
from egydet.agents.verifier import DEFAULT_MODEL, system_prompt
from egydet.gold.io import load_gold
from egydet.kb import ROOT
from egydet.legacy_bridge.v147 import load_env
from egydet.llm import CallStats, DiskCache

PROMPT_VERSION = "reader_v1"
FILE = "reader.jsonl"

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["words", "transliteration", "translation"],
    "properties": {
        "words": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["indices", "transliteration", "gloss"],
                "properties": {
                    "indices": {"type": "array", "items": {"type": "integer"}},
                    "transliteration": {"type": "string"},
                    "gloss": {"type": "string"},
                },
            },
        },
        "transliteration": {"type": "string"},
        "translation": {"type": "string"},
    },
}


def user_prompt(record: dict) -> str:
    toks = record["tokens"]
    lines = [f"MdC: {record['mdc']}", "", "Signs (index: Gardiner code):"]
    lines += [f"{i}: {t}" for i, t in enumerate(toks)]
    lines += ["", f"Assign every index from 0 to {len(toks) - 1} to exactly one word."]
    return "\n".join(lines)


def parse_reading(answer: dict, record: dict) -> dict:
    words = []
    for w in (answer or {}).get("words", []) or []:
        try:
            idx = sorted({int(i) for i in w.get("indices", []) if 0 <= int(i) < len(record["tokens"])})
        except (TypeError, ValueError):
            idx = []
        words.append({"indices": idx, "transliteration": str(w.get("transliteration", "")).strip(),
                      "gloss": str(w.get("gloss", "")).strip()})
    covered = sorted(i for w in words for i in w["indices"])
    return {"transliteration": str((answer or {}).get("transliteration", "")).strip(),
            "translation": str((answer or {}).get("translation", "")).strip(),
            "words": words,
            "segmentation_ok": covered == list(range(len(record["tokens"])))}


def run(mode: str, model: str, out_dir: str, limit: Optional[int] = None, env_file: Optional[str] = None,
        fake_answer=None) -> str:
    if mode == "live":
        load_env(env_file)
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not found")
    gold = sorted(load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl")), key=lambda r: r["corpus_row"])
    if limit:
        gold = gold[:limit]
    stats = CallStats()
    client = ChatClient(model, mode=mode, cache=DiskCache(), stats=stats, fake_answer=fake_answer)
    system = system_prompt(PROMPT_VERSION)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, FILE)
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"agent": "reader", "prompt": PROMPT_VERSION, "model": model, "mode": mode,
                   "started": _dt.datetime.now().isoformat(timespec="seconds")}, fh, indent=1)
    with open(path, "a", encoding="utf-8") as out:
        for r in gold:
            if r["id"] in done:
                continue
            before = stats.snapshot()
            error = ""
            try:
                answer = client.complete_json(system, user_prompt(r), SCHEMA, "reading")
            except Exception as exc:   # recorded, never silent
                answer, error = {}, f"{type(exc).__name__}: {str(exc)[:300]}"
            rd = parse_reading(answer, r)
            after = stats.snapshot()
            rec = {"id": r["id"], "error": error, **rd, "calls": {k: after[k] - before[k] for k in after}}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            flag = "" if not error and rd["transliteration"] and rd["segmentation_ok"] else "  <-- check"
            print(f"{r['id']}: {rd['transliteration'][:60]!r} words={len(rd['words'])} "
                  f"seg_ok={rd['segmentation_ok']} api={rec['calls']['api_calls']}{flag}")
    s = stats.snapshot()
    with open(os.path.join(out_dir, "collect_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(s, error_messages=stats.errors[:20]), fh, indent=1)
    print(s)
    return out_dir


def load_reading(run_dir: str) -> Dict[str, dict]:
    out = {}
    for line in open(os.path.join(run_dir, FILE), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


# ---------------------------------------------------------------- evaluation against the experts' reading
_EQUIV = {"ı͗": "ꞽ", "i͗": "ꞽ", "ỉ": "ꞽ", "j": "ꞽ", "i": "ꞽ", "Ꜣ": "ꜣ", "3": "ꜣ", "ʿ": "ꜥ", "=": "⸗",
          "č": "ṯ", "ḳ": "q", "ṱ": "ṯ"}


def normalise(t: str) -> str:
    """Comparable transliteration: equivalent spellings unified, editorial marks and spaces removed."""
    t = unicodedata.normalize("NFC", t.lower())
    for a, b in sorted(_EQUIV.items(), key=lambda kv: -len(kv[0])):
        t = t.replace(a, b)
    return re.sub(r"[\s\[\]()<>{}?.\-]", "", t)


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate(run_dir: str) -> dict:
    gold = {r["id"]: r for r in load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl"))}
    pred = load_reading(run_dir)
    rows = []
    for sid, g in gold.items():
        p = pred.get(sid)
        if p is None:
            continue
        a, b = normalise(g["transliteration"]), normalise(p["transliteration"])
        cer = _lev(b, a) / max(1, len(a))
        sfx_g, sfx_p = g["transliteration"].count("⸗"), p["transliteration"].count("⸗") + p["transliteration"].count("=")
        rows.append({"id": sid, "cer": cer, "exact": a == b, "seg_ok": p["segmentation_ok"],
                     "suffix_gold": sfx_g, "suffix_pred": sfx_p, "gold": g["transliteration"],
                     "pred": p["transliteration"]})
    cers = [r["cer"] for r in rows]
    with_sfx = [r for r in rows if r["suffix_gold"]]
    res = {"run": run_dir, "n": len(rows), "cer_mean": mean(cers), "cer_median": median(cers),
           "share_cer_below_0.2": sum(c < 0.2 for c in cers) / len(cers),
           "exact": sum(r["exact"] for r in rows), "segmentation_ok": sum(r["seg_ok"] for r in rows),
           "suffix_seqs_gold": len(with_sfx), "suffix_seqs_found": sum(1 for r in with_sfx if r["suffix_pred"]),
           "suffix_false_alarms": sum(1 for r in rows if r["suffix_pred"] and not r["suffix_gold"]),
           "rows": sorted(rows, key=lambda r: -r["cer"])}
    return res


def to_markdown(res: dict) -> str:
    L = [f"# Reader evaluation ({res['run']})", "",
         "CER: character edit distance to the experts' transliteration divided by its length, after unifying "
         "equivalent spellings and removing spaces and editorial marks.", "",
         f"- sequences: {res['n']}; segmentation covers every sign: {res['segmentation_ok']}",
         f"- CER mean {res['cer_mean']:.3f}, median {res['cer_median']:.3f}; CER < 0.2 in "
         f"{100 * res['share_cer_below_0.2']:.0f}% of sequences; exact {res['exact']}",
         f"- suffix pronouns (⸗): gold has them in {res['suffix_seqs_gold']} sequences, the reader marks one in "
         f"{res['suffix_seqs_found']} of these; marks one in {res['suffix_false_alarms']} other sequences", "",
         "| id | CER | gold | predicted |", "|---|---|---|---|"]
    for r in res["rows"][:15] + res["rows"][-5:]:
        L.append(f"| {r['id']} | {r['cer']:.2f} | {r['gold']} | {r['pred']} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Reader agent: words, transliteration, translation from MdC")
    ap.add_argument("--mode", choices=("live", "offline", "fake"), default="live")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--evaluate", default=None, metavar="RUN_DIR", help="compare a reader run with the experts")
    args = ap.parse_args(argv)
    if args.evaluate:
        res = evaluate(args.evaluate)
        md = to_markdown(res)
        with open(os.path.join(args.evaluate, "reader_eval.md"), "w", encoding="utf-8") as fh:
            fh.write(md)
        with open(os.path.join(args.evaluate, "reader_eval.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)
        print(md)
        return
    out = args.out or os.path.join(ROOT, "runs", f"phase2_reader_{args.model}")
    run(args.mode, args.model, out, args.limit, args.env_file)


if __name__ == "__main__":
    main()
