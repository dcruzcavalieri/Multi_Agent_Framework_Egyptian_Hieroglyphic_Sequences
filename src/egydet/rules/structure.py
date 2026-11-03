"""Structural view of an MdC sequence: groups, blocks ("words"), finals, plural hints, suffix spans.

Faithful port of v1.47 ``build_mdc_structure``, ``build_word_boundaries`` and
``generate_suffix_span_candidates``. The computation runs on the legacy token list (signs and
separators) so that it is bit-identical to v1.47; every index in the returned ``Structure`` is a
**sign index** (separators excluded), the index space used by the gold file and the metrics.

Blocks are closed at every ``-`` separator, as in v1.47.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence

from egydet.mdc.tokenizer import SEPARATORS, tokenize_with_separators


@dataclass(frozen=True)
class Group:
    kind: str                 # "stack" (":") or "ligature" ("*")
    sign_indices: tuple


@dataclass(frozen=True)
class Word:
    word_id: int
    sign_indices: tuple


@dataclass(frozen=True)
class Structure:
    signs: tuple                          # sign codes, index = sign index
    groups: tuple                         # Group
    words: tuple                          # Word (v1.47 "blocks")
    block_final: FrozenSet[int]
    final_sign: Optional[int]
    near_end: FrozenSet[int]              # last 8 signs
    plural_markers: FrozenSet[int]
    plural_heads: FrozenSet[int]
    group_ends: FrozenSet[int]
    span_candidates: tuple                # per word: tuple of tuples of sign indices (first = empty span)


def build_structure(mdc: str, block_end_signs: FrozenSet[str], plural_set: FrozenSet[str],
                    max_span_len: int = 6, near_end_n: int = 8) -> Structure:
    tokens = tokenize_with_separators(mdc)
    is_sep = lambda t: t in SEPARATORS  # noqa: E731
    tok2sign: Dict[int, int] = {}
    signs: List[str] = []
    for ti, t in enumerate(tokens):
        if not is_sep(t):
            tok2sign[ti] = len(signs)
            signs.append(t)

    # groups: chains joined by ':' or '*'
    groups: List[Group] = []
    i = 0
    while i < len(tokens):
        if is_sep(tokens[i]):
            i += 1
            continue
        if i + 1 < len(tokens) and tokens[i + 1] in (":", "*"):
            sep_type = tokens[i + 1]
            chain = [i]
            j = i
            while j + 1 < len(tokens) and tokens[j + 1] == sep_type:
                if j + 2 < len(tokens) and not is_sep(tokens[j + 2]):
                    chain.append(j + 2)
                    j += 2
                else:
                    break
            if len(chain) >= 2:
                groups.append(Group("stack" if sep_type == ":" else "ligature",
                                    tuple(tok2sign[c] for c in chain)))
            i = j + 1
            continue
        i += 1

    # blocks: closed at '-' and after a block-end sign followed by another sign without '-'
    blocks: List[List[int]] = []
    current: List[int] = []

    def next_sign_index(start: int) -> int:
        j = start + 1
        while j < len(tokens):
            if tokens[j] in SEPARATORS:
                if tokens[j] == "-":
                    return -1
                j += 1
                continue
            return j
        return -1

    for ti, t in enumerate(tokens):
        if is_sep(t):
            if t == "-" and current:
                blocks.append(current)
                current = []
            continue
        current.append(ti)
        if t in block_end_signs and next_sign_index(ti) != -1 and current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    words = tuple(Word(wi, tuple(tok2sign[t] for t in b)) for wi, b in enumerate(blocks))
    if not words and signs:
        words = (Word(0, tuple(range(len(signs)))),)

    final_sign = len(signs) - 1 if signs else None
    near_end = frozenset(range(max(0, len(signs) - near_end_n), len(signs)))

    plural_markers, plural_heads = [], []
    for k, s in enumerate(signs):
        if s in plural_set:
            plural_markers.append(k)
            for back in range(1, 4):
                if k - back < 0:
                    break
                if signs[k - back] not in plural_set:
                    plural_heads.append(k - back)
                    break

    spans = []
    for w in words:
        idxs = list(w.sign_indices)
        cands = [()] + [tuple(idxs[-k:]) for k in range(1, min(max_span_len, len(idxs)) + 1)]
        spans.append(tuple(cands))

    return Structure(
        signs=tuple(signs),
        groups=tuple(groups),
        words=words,
        block_final=frozenset(w.sign_indices[-1] for w in words if w.sign_indices),
        final_sign=final_sign,
        near_end=near_end,
        plural_markers=frozenset(plural_markers),
        plural_heads=frozenset(plural_heads),
        group_ends=frozenset(g.sign_indices[-1] for g in groups),
        span_candidates=tuple(spans),
    )


def legacy_tokens(mdc: str) -> List[str]:
    return tokenize_with_separators(mdc)


def sign_index_of_token(tokens: Sequence[str]) -> Dict[int, int]:
    out, k = {}, 0
    for ti, t in enumerate(tokens):
        if t not in SEPARATORS:
            out[ti] = k
            k += 1
    return out
