"""Inspect the agent JSON captured in a Phase 1 collect run.

    python tools/inspect_phase1.py runs/phase1_llm

For the semantic and structural agents it reports which top-level keys their JSON uses, where the
determinative list lives, and whether the indices refer to the legacy token list (separators
included, as v1.47 expects) or to sign positions.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from egydet.mdc import tokenize, tokenize_with_separators  # noqa: E402

LIST_KEYS = ("determinatives", "candidates", "determinative_spans", "determinative_candidates")


def find_lists(obj, path=""):
    """Yield (path, list) for every list of dicts under a known determinative key."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if k in LIST_KEYS and isinstance(v, list):
                yield p, v
            yield from find_lists(v, p)


AGENT_MARKERS = {"semantic": "SEMANTIC analysis", "structural": "STRUCTURAL analysis", "synthesizer": "FINAL ARBITER"}


def _parse(content: str) -> str:
    """Classify an answer the way v1.47 parse_llm_json would see it."""
    import re
    if not isinstance(content, str) or not content.strip():
        return "empty response"
    t = content.strip()
    if "BEGIN_JSON" in t and "END_JSON" in t:
        t = t.split("BEGIN_JSON", 1)[1].split("END_JSON", 1)[0]
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip(), flags=re.I | re.S)
    if "{" not in t:
        return "no JSON object"
    t = t[t.find("{"): t.rfind("}") + 1]
    try:
        return "ok" if json.loads(t) else "empty object"
    except Exception:
        return "invalid JSON"


def scan_cache(cache_dir: str) -> None:
    """Outcome of every cached answer per agent (finish reason, parse result)."""
    if not os.path.isdir(cache_dir):
        print(f"\n(no cache at {cache_dir})")
        return
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    from _legacy import load_v147
    v147_parse = load_v147("_extract_json_substring", "parse_llm_json")["parse_llm_json"]
    per = {a: Counter() for a in AGENT_MARKERS}
    finish = {a: Counter() for a in AGENT_MARKERS}
    lost = {a: Counter() for a in AGENT_MARKERS}
    samples = {}
    for root, _, files in os.walk(cache_dir):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            rec = json.load(open(os.path.join(root, fn), encoding="utf-8"))
            system = next((m["content"] for m in rec["request"]["messages"] if m["role"] == "system"), "")
            agent = next((a for a, mk in AGENT_MARKERS.items() if mk in system), None)
            if agent is None:
                continue
            content = rec["response"]["content"]
            outcome = _parse(content)
            per[agent][outcome] += 1
            if outcome == "ok" and not v147_parse(content):
                curly = any(q in content for q in "\u201c\u201d")
                lost[agent]["lost by v1.47 parser" + (" (has curly quotes)" if curly else "")] += 1
            meta = rec["response"].get("response_metadata") or {}
            finish[agent][str(meta.get("finish_reason"))] += 1
            if outcome != "ok" and agent not in samples:
                samples[agent] = (outcome, str(rec["response"]["content"])[:300], rec["response"].get("usage_metadata"))
    print("\n== cached answers")
    for a in AGENT_MARKERS:
        print(f"  {a:12s} parse: {dict(per[a])}   finish_reason: {dict(finish[a])}   {dict(lost[a])}")
        if a in samples:
            print(f"  {'':12s} first problem: {samples[a][0]}; usage={samples[a][2]}; head={samples[a][1]!r}")


def main(run_dir: str) -> None:
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "records.jsonl"), encoding="utf-8") if l.strip()]
    print(f"records: {len(recs)}  failed: {sum(not r['success'] for r in recs)}")
    for agent in ("semantic_json", "structural_json"):
        top, where, item_keys, idx_kind = Counter(), Counter(), Counter(), Counter()
        empty, sample = 0, None
        for r in recs:
            js = r["state"].get(agent) or {}
            if not js:
                empty += 1
                continue
            top.update(js.keys())
            legacy = tokenize_with_separators(r["mdc"])
            signs = [t.text for t in tokenize(r["mdc"])]
            for path, items in find_lists(js):
                where[path] += 1
                for d in items:
                    if not isinstance(d, dict):
                        idx_kind["not a dict"] += 1
                        continue
                    item_keys.update(d.keys())
                    sample = sample or (path, d)
                    tok = str(d.get("token") or d.get("sign") or "").strip()
                    raw = d.get("index", d.get("position", d.get("token_index")))
                    try:
                        i = int(raw)
                    except (TypeError, ValueError):
                        idx_kind["no integer index"] += 1
                        continue
                    ok_legacy = 0 <= i < len(legacy) and legacy[i] == tok
                    ok_sign = 0 <= i < len(signs) and signs[i] == tok
                    idx_kind[{(True, True): "matches both", (True, False): "legacy index (v1.47)",
                              (False, True): "SIGN index (dropped by v1.47)",
                              (False, False): "matches neither"}[(ok_legacy, ok_sign)]] += 1
        print(f"\n== {agent}: empty {empty}/{len(recs)}")
        print("  top-level keys:", dict(top.most_common(12)))
        print("  lists found at:", dict(where.most_common(8)))
        print("  item keys:     ", dict(item_keys.most_common(10)))
        print("  index check:   ", dict(idx_kind))
        if sample:
            print("  sample item:   ", sample[0], json.dumps(sample[1], ensure_ascii=False)[:300])
    scan_cache(os.path.join(os.path.dirname(__file__), "..", ".cache", "llm"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/phase1_llm")
