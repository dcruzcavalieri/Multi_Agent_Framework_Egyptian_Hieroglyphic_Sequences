"""Rule baseline: every sign present in the knowledge base is a candidate (v1.47 ``analyze_rule_based``).

Confidence: ``high`` (1.0) for the last sign of the sequence, ``medium`` (0.7) otherwise, exactly as
v1.47 maps it in ``_gather_rule_evidence``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet

from egydet.kb import KnowledgeBase
from egydet.rules.structure import Structure

CONF_HIGH, CONF_MEDIUM = 1.0, 0.7


@dataclass
class Candidate:
    sign: str
    index: int                       # sign index
    sources: FrozenSet[str]
    conf_rule: float = 0.0
    conf_sem: float = 0.0
    conf_struct: float = 0.0
    is_final: bool = False
    is_global_final: bool = False
    is_group_end: bool = False
    is_near_end: bool = False
    is_plural_marker: bool = False
    is_plural_head: bool = False
    is_phonetic_complement_likely: bool = False
    semantic_hints: str = ""
    extra: Dict = field(default_factory=dict)


def rule_candidates(st: Structure, kb: KnowledgeBase) -> Dict[int, Candidate]:
    pool: Dict[int, Candidate] = {}
    last = len(st.signs) - 1
    for i, s in enumerate(st.signs):
        if s not in kb.signs:
            continue
        pool[i] = Candidate(
            sign=s, index=i, sources=frozenset({"rules"}),
            conf_rule=CONF_HIGH if i == last else CONF_MEDIUM,
            is_final=i in st.block_final,
            is_global_final=st.final_sign is not None and i == st.final_sign,
            is_group_end=i in st.group_ends,
            is_near_end=i in st.near_end,
            is_plural_marker=i in st.plural_markers,
            is_plural_head=i in st.plural_heads,
        )
    return pool
