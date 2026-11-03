"""Build data/gold/gold_v1.jsonl from the corpus spreadsheet and the annotated DOCX files.

    python -m egydet.gold.build            # defaults below, run from the repository root
    egydet-gold --raw data/raw --out data/gold

Outputs
    gold_v1.jsonl           one record per annotated sequence, with determinative positions
    gold_v1_review.csv      items that need an expert decision (empty while none are found)
    gold_v1_report.md       summary statistics and consistency checks
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from typing import Dict, List

import pandas as pd

from egydet.gold.docx_parser import AnnotatedSequence, parse_docx
from egydet.mdc.tokenizer import normalize, tokenize

DOCX_FILES = ["Langgraph_analyses_comments.docx", "Langgraph_analyses_comments_2.docx"]
CORPUS_FILE = "extractrion_for_agents.xlsx"
GOLD_VERSION = "gold_v1"


def _sha1(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()


def load_corpus(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["row"] = range(len(df))
    df["mdc"] = df["Code"].astype(str).map(normalize)
    df["source"] = df["Extraction"].astype(str).str.split(",").str[-1].str.strip()
    df["id"] = df.apply(lambda r: f"{r['source']}-r{r['row']:03d}", axis=1)
    return df


def build_records(corpus: pd.DataFrame, annotated: List[AnnotatedSequence]) -> tuple:
    by_mdc: Dict[str, List[int]] = {}
    for i, m in enumerate(corpus["mdc"]):
        by_mdc.setdefault(m, []).append(i)

    records, review = [], []
    seen = Counter(a.mdc for a in annotated)
    for a in annotated:
        rows = by_mdc.get(a.mdc, [])
        if len(rows) != 1:
            review.append({"mdc": a.mdc, "issue": f"matches {len(rows)} corpus rows", "detail": ""})
            continue
        if seen[a.mdc] > 1:
            review.append({"mdc": a.mdc, "issue": "annotated more than once", "detail": a.source_file})
        row = corpus.iloc[rows[0]]
        toks = tokenize(a.mdc)
        det_signs = [toks[i].text for i in a.det_idx]

        if a.partial_bold_idx:
            review.append({"id": row["id"], "mdc": a.mdc, "issue": "token partly in bold",
                           "detail": ",".join(f"{i}:{toks[i].text}" for i in a.partial_bold_idx)})
        for i in a.det_idx:
            if not toks[i].is_standard:
                review.append({"id": row["id"], "mdc": a.mdc, "issue": "non-standard sign marked as determinative",
                               "detail": f"{i}:{toks[i].text}"})
        gold_set = set(det_signs)
        for s in a.system_pred_rejected:
            if s in gold_set:
                review.append({"id": row["id"], "mdc": a.mdc, "issue": "sign marked red but also bold",
                               "detail": s})

        records.append({
            "id": row["id"],
            "source": row["source"],
            "line": str(row["Line"]),
            "corpus_row": int(row["row"]),
            "mdc": a.mdc,
            "tokens": [t.text for t in toks],
            "det_idx": a.det_idx,
            "det_signs": det_signs,
            "transliteration": "" if pd.isna(row.get("transliteration")) else str(row["transliteration"]),
            "translation": "" if pd.isna(row.get("translation")) else str(row["translation"]),
            "annotation": {
                "method": "bold_on_reviewed_system_output",
                "file": os.path.basename(a.source_file),
                "paragraph": a.paragraph_index,
                "system_pred": a.system_pred,
                "system_pred_rejected": a.system_pred_rejected,
            },
            "status": "extracted",
            "gold_version": GOLD_VERSION,
        })
    records.sort(key=lambda r: r["corpus_row"])
    return records, review


def anchoring_stats(records: List[dict]) -> dict:
    """How the expert gold relates to the system output the expert was reviewing (sign multisets)."""
    accepted = added = dropped = rejected_red = 0
    for r in records:
        gold = Counter(r["det_signs"])
        pred = Counter(r["annotation"]["system_pred"])
        accepted += sum((gold & pred).values())
        added += sum((gold - pred).values())
        dropped += sum((pred - gold).values())
        rejected_red += len(r["annotation"]["system_pred_rejected"])
    return {"accepted": accepted, "added_by_expert": added,
            "dropped_from_system": dropped, "marked_red": rejected_red}


def write_report(path: str, records: List[dict], review: List[dict], inputs: Dict[str, str]) -> None:
    n = len(records)
    n_dets = sum(len(r["det_idx"]) for r in records)
    n_tokens = sum(len(r["tokens"]) for r in records)
    empty = sum(1 for r in records if not r["det_idx"])
    single = sum(1 for r in records if len(r["det_idx"]) == 1)
    nonstd_tok = sum(1 for r in records for t in r["tokens"] if not tokenize(t)[0].is_standard)
    files = Counter(r["annotation"]["file"] for r in records)
    signs = Counter(s for r in records for s in r["det_signs"])
    anch = anchoring_stats(records)
    lines = [
        f"# {GOLD_VERSION}: extraction report",
        "",
        "Generated by `python -m egydet.gold.build`. Do not edit by hand.",
        "",
        "## Inputs",
        "",
        *[f"- `{name}` sha1 `{sha}`" for name, sha in inputs.items()],
        "",
        "## Counts",
        "",
        f"- Annotated sequences: {n} ({', '.join(f'{k}: {v}' for k, v in files.items())})",
        f"- Sign tokens: {n_tokens}; determinative positions: {n_dets} ({100 * n_dets / n_tokens:.1f}% of tokens)",
        f"- Sequences with no determinative: {empty}; with exactly one: {single}",
        f"- Distinct determinative signs: {len(signs)}; most frequent: "
        + ", ".join(f"{s} ({c})" for s, c in signs.most_common(10)),
        f"- Non-standard sign tokens anywhere in the 99 sequences: {nonstd_tok}",
        "",
        "## Consistency checks",
        "",
        "- Every annotated sequence matches exactly one corpus row: "
        + ("yes" if not any("corpus rows" in x["issue"] for x in review) else "NO, see review file"),
        "- Tokens only partly in bold: " + str(sum(1 for x in review if x["issue"] == "token partly in bold")),
        "- Items for expert review: " + str(len(review)),
        "",
        "## Relation to the reviewed system output (anchoring)",
        "",
        "The experts marked determinatives on top of the output of an earlier system version. Comparing the",
        "gold with the sign list on each `Determinativos:` line (sign multisets):",
        "",
        f"- System signs kept by the expert: {anch['accepted']}",
        f"- Determinatives added by the expert (absent from the system list): {anch['added_by_expert']}",
        f"- System signs not kept: {anch['dropped_from_system']} (of which explicitly marked red: {anch['marked_red']})",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/gold")
    args = ap.parse_args(argv)

    corpus_path = os.path.join(args.raw, CORPUS_FILE)
    docx_paths = [os.path.join(args.raw, f) for f in DOCX_FILES]
    corpus = load_corpus(corpus_path)
    annotated = [a for p in docx_paths for a in parse_docx(p)]
    records, review = build_records(corpus, annotated)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{GOLD_VERSION}.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, f"{GOLD_VERSION}_review.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "mdc", "issue", "detail"])
        w.writeheader()
        for x in review:
            w.writerow({k: x.get(k, "") for k in w.fieldnames})
    inputs = {os.path.basename(p): _sha1(p) for p in [corpus_path, *docx_paths]}
    write_report(os.path.join(args.out, f"{GOLD_VERSION}_report.md"), records, review, inputs)
    print(f"{len(records)} records, {len(review)} review items -> {args.out}")


if __name__ == "__main__":
    main()
