"""Sign-list notes for the verifier prompt (Phase 2, option B).

The lexicon file is built by ``tools/build_lexicon_manual.py`` and is not distributed with the repository
(``kb/private/`` is git-ignored). Only the content of the sign list is used: description, listed
determinative use, whether an ideographic or phonetic value is listed (not the values themselves,
whose OCR is unreliable). Corpus and gold counts in the same
file never reach a prompt.
"""
from __future__ import annotations

import csv
import hashlib
import os
from typing import Dict, List

from egydet.kb import ROOT

DEFAULT_PATH = os.path.join(ROOT, "kb", "private", "lexicon_manual.csv")
SOURCE = "Gracia Zamacona, Manual de egipcio medio (2017), Anexo I"


def _empty(v: str) -> bool:
    return (v or "").strip() in ("", "-", "—", "–")


def load_lexicon(path: str = DEFAULT_PATH) -> Dict[str, dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"lexicon not found: {path} (build it with tools/build_lexicon_manual.py)")
    with open(path, encoding="utf-8") as fh:
        return {r["corpus_code"]: r for r in csv.DictReader(fh)}


def lexicon_sha1(path: str = DEFAULT_PATH) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()


def note(code: str, lex: Dict[str, dict]) -> str:
    """One line of sign-list information for ``code`` (descriptions are in Spanish, as in the source)."""
    e = lex.get(code)
    if e is None or not e.get("manual_code"):
        return f"- {code}: not in the sign list"
    ref = "" if e["match"] == "exact" else f"entry of the base sign {e['manual_code']}; "
    parts = [f"- {code}: {ref}{e['image'].strip() or 'no description'}"]
    uses = []    # transliterated values are left out: OCR loses their diacritics
    if not _empty(e["ideogram"]):
        uses.append("ideogram")
    if not _empty(e["phonogram"]):
        uses.append("phonogram")
    if not _empty(e["determinative"]):
        uses.append(f"determinative ({e['determinative'].strip()})")
    parts.append("listed uses: " + (", ".join(uses) if uses else "none given"))
    if _empty(e["determinative"]):
        parts.append("no determinative use listed")
    return "; ".join(parts)


def lexicon_lines(tokens: List[str], lex: Dict[str, dict]) -> List[str]:
    return [note(t, lex) for t in dict.fromkeys(tokens)]
