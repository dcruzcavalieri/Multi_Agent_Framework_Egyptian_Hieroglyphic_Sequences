"""MdC tokenizer with character offsets.

Sign tokens are maximal runs of characters between the separators ``-`` (juxtaposition), ``:``
(vertical stacking) and ``*`` (ligature). This reproduces the sign tokens of the frozen v1.47
tokenizer (``tokenize_mdc`` with separators removed), so that sign index ``i`` here is the i-th
non-separator token there. Whitespace is removed before tokenisation, as in v1.47.

Characters that MdC uses for other purposes (``&`` ligature in JSesh notation, ``.``, ``?``, ``<``,
``>``, parentheses) are deliberately kept inside tokens for compatibility; ``Token.is_standard``
flags tokens that are not a plain Gardiner-style code so they can be reviewed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

SEPARATORS = frozenset("-:*")
_WS = re.compile(r"\s+")
# Gardiner-style code: category letters, number, optional variant suffix (A1, Aa1, D54, W22VARA, N35A).
_STANDARD = re.compile(r"(?:[A-Z]|Aa|NL|NU)\d+[A-Za-z0-9]*")


@dataclass(frozen=True)
class Token:
    text: str
    index: int   # position among sign tokens (separators excluded), 0-based
    start: int   # character offset in the normalised MdC string
    end: int     # exclusive

    @property
    def is_standard(self) -> bool:
        return bool(_STANDARD.fullmatch(self.text))


def normalize(mdc: str) -> str:
    """Remove all whitespace (same rule as v1.47 ``normalize_mdc``)."""
    if not isinstance(mdc, str):
        return ""
    return _WS.sub("", mdc.strip())


def tokenize(mdc: str) -> List[Token]:
    """Split a (normalised) MdC string into sign tokens with offsets."""
    s = normalize(mdc)
    tokens: List[Token] = []
    start = None
    for i, ch in enumerate(s + "-"):
        if ch in SEPARATORS:
            if start is not None:
                tokens.append(Token(s[start:i], len(tokens), start, i))
                start = None
        elif start is None:
            start = i
    return tokens


def tokenize_with_separators(mdc: str) -> List[str]:
    """Sign tokens *and* separators, exactly as v1.47 ``tokenize_mdc`` (used by the structure builder)."""
    s = normalize(mdc)
    out: List[str] = []
    buff = ""
    for ch in s:
        if ch in SEPARATORS:
            if buff:
                out.append(buff)
                buff = ""
            out.append(ch)
        else:
            buff += ch
    if buff:
        out.append(buff)
    return out
