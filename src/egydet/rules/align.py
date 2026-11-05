"""Align the signs of a sequence with the words of its transliteration, using phonetic values.

Input: the sign codes, the experts' transliteration (Egyptological characters) and a table
``sign -> phonetic values`` in Manuel de Codage ASCII (A aleph, i yod, a ayin, H, x, X, S, T, D ...).

1. The transliteration is split into words (spaces and the suffix-pronoun mark ⸗ separate words) and
   reduced to the letters that are written: dots and hyphens are dropped, letters in ( ) and < > are
   removed (not written by the scribe), letters in [ ] (restored) are kept but may lack signs.
2. A dynamic programme assigns every sign, in order, to one word (a contiguous span per word; a word may
   have no sign, at a cost, for lost or restored passages).
3. Inside its word each sign gets a role:
   ``phon``  its value writes letters of the word not written yet;
   ``compl`` its value matches letters already written (phonetic complement, or a redundant logogram);
   ``none``  no value in the table, or no value found in the word.
4. Determinative slots: the signs after the last ``phon`` sign of a word whose role is ``none``, or
   ``compl`` with a value of two or more letters (a multi-letter sign whose sound is already written acts
   as a classifier, e.g. N1 in p.t written Q3 X1 N1). A trailing uniliteral complement is not a slot,
   and a word in which no sign writes a sound (a logogram, e.g. D35 for n) has no slot.

The module uses no corpus statistics and no annotation; it has no tuned parameter except the fixed
costs below.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from egydet.kb import ROOT

DEFAULT_TABLE = os.path.join(ROOT, "kb", "private", "mdc_phonetic_values.json")
_TO_MDC = {"ꜣ": "A", "ꞽ": "i", "ı͗": "i", "j": "i", "ꜥ": "a", "ḥ": "H", "ḫ": "x", "ẖ": "X", "š": "S", "ṯ": "T",
           "ḏ": "D", "z": "s", "ḳ": "q"}
_PHONEMES = set("AiyawbpfmnrhHxXsSqkgtTdD")

# fixed costs (per letter / per sign)
LETTER_COVERED = 1.0
LETTER_MISSING = 1.0
SIGN_NONE_FINAL = 0.15      # an unexplained sign at the end of a word (a determinative slot) is cheap
SIGN_NONE_INNER = 0.8       # an unexplained sign inside a word is expensive
EMPTY_WORD = 0.6            # per letter of a word that gets no sign (lost or restored text)


def load_table(path: str = DEFAULT_TABLE, extra: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)["values"]
    table: Dict[str, List[str]] = {}
    for code, vals in list(raw.items()) + list((extra or {}).items()):
        out = table.setdefault(code, [])
        for v in vals:
            if len(v) == 1 and v.isupper() and v not in _PHONEMES:
                v = v.lower()                  # 'M' (Aa15), 'W' (Z7): variant codes for m, w
            v = v.replace("z", "s")
            if v and all(ch in _PHONEMES for ch in v) and v not in out:
                out.append(v)
    return table


def words_of(transliteration: str) -> List[str]:
    t = transliteration
    for a, b in _TO_MDC.items():
        t = t.replace(a, b)
    t = re.sub(r"\([^)]*\)|<[^>]*>", "", t)          # not written by the scribe
    t = re.sub(r"[\[\]{}?]", "", t)
    out = []
    for w in re.split(r"[\s⸗=]+", t):
        w = re.sub(r"[^AiyawbpfmnrhHxXsSqkgtTdD]", "", w)
        if w:
            out.append(w)
    return out


def _match_word(word: str, vals: Tuple[Tuple[str, ...], ...]):
    """Best role assignment of a span of signs to one word. Returns (score, roles)."""
    n, L = len(vals), len(word)

    @lru_cache(maxsize=None)
    def go(k: int, last_start: int, covered: int):
        # covered: bitmask of the letters already written; match starts never move backwards
        if k == n:
            ncov = bin(covered).count("1")
            return LETTER_COVERED * ncov - LETTER_MISSING * (L - ncov), ()
        best = None
        for v in vals[k]:
            for s in range(last_start, L - len(v) + 1):
                if word[s:s + len(v)] != v:
                    continue
                mask = ((1 << len(v)) - 1) << s
                role = ("phon" if mask & ~covered else "compl", v)
                sc, rest = go(k + 1, s, covered | mask)
                if best is None or sc > best[0]:
                    best = (sc, (role,) + rest)
        sc, rest = go(k + 1, last_start, covered)
        if best is None or sc - SIGN_NONE_INNER > best[0]:
            best = (sc - SIGN_NONE_INNER, (("none", ""),) + rest)
        return best

    score, roles = go(0, 0, 0)
    # unexplained signs after the last phonetic sign are cheap (determinative slots)
    last = max([i for i, (r, _) in enumerate(roles) if r == "phon"], default=-1)
    refund = sum(SIGN_NONE_INNER - SIGN_NONE_FINAL for i, (r, _) in enumerate(roles) if r == "none" and i > last)
    return score + refund, list(roles)


def align(tokens: Sequence[str], transliteration: str, table: Dict[str, List[str]]) -> dict:
    """Word spans, sign roles and determinative slots for one sequence."""
    words = words_of(transliteration)
    vals = [tuple(table.get(t, ())) for t in tokens]
    n, W = len(tokens), len(words)
    NEG = float("-inf")
    f = [[NEG] * (W + 1) for _ in range(n + 1)]
    back: Dict[Tuple[int, int], Tuple[int, List[str]]] = {}
    f[0][0] = 0.0
    for k in range(1, W + 1):
        w = words[k - 1]
        for i in range(0, n + 1):
            for j in range(0, i + 1):
                if f[j][k - 1] == NEG:
                    continue
                if j == i:
                    sc, roles = -EMPTY_WORD * len(w), []
                else:
                    sc, roles = _match_word(w, tuple(vals[j:i]))
                if f[j][k - 1] + sc > f[i][k]:
                    f[i][k] = f[j][k - 1] + sc
                    back[(i, k)] = (j, roles)
    if W == 0 or f[n][W] == NEG:
        return {"words": words, "spans": [], "roles": ["none"] * n, "values": [""] * n, "slots": [], "ok": False}
    spans, roles = [], [("", "")] * n
    i = n
    for k in range(W, 0, -1):
        j, r = back[(i, k)]
        spans.append((j, i))
        roles[j:i] = r
        i = j
    spans.reverse()
    slots = []
    for (a, b) in spans:
        rs = roles[a:b]
        last = max([t for t, (r, _) in enumerate(rs) if r == "phon"], default=-1)
        if last < 0:
            continue                     # no sign writes a sound: the word is written by a logogram
        for t in range(last + 1, len(rs)):
            r, v = rs[t]
            if r == "none" or (r == "compl" and len(v) >= 2):
                slots.append(a + t)
    return {"words": words, "spans": spans, "roles": [r for r, _ in roles], "values": [v for _, v in roles],
            "slots": slots, "ok": True}
