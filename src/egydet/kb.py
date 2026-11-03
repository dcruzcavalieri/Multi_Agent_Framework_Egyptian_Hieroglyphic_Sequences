"""Knowledge base: sign entries (kb/signs.yaml) and sign categories (kb/categories.yaml)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Tuple

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KB_DIR = os.path.join(ROOT, "kb")

# Order matters only if tiers overlapped; they are disjoint (tested). Kept in the v1.47 order.
CONTEXT_TIER_TO_FIELD: Tuple[Tuple[str, str], ...] = (
    ("context_vessel", "vessel"), ("context_divine", "divine"), ("context_vision", "vision"),
    ("context_bovine", "bovine"), ("context_aquatic", "aquatic"), ("context_food", "food"),
    ("context_animal", "animal"), ("context_insect", "insect"), ("context_tool", "tool"),
    ("context_body_action", "body_action"),
)


@dataclass(frozen=True)
class KnowledgeBase:
    signs: Dict[str, dict]
    tiers: Dict[str, FrozenSet[str]]
    semantic_fields: Dict[str, FrozenSet[str]]
    strict_context_signs: FrozenSet[str]
    recently_critical: FrozenSet[str]
    recently_minor: FrozenSet[str]
    recently_added: FrozenSet[str]
    strong_determinatives: FrozenSet[str]
    plural_markers: FrozenSet[str]
    final_location_signs: FrozenSet[str]
    location_modifier_signs: FrozenSet[str]
    contextual_only_signs: FrozenSet[str]
    block_end_signs: FrozenSet[str]
    sign_to_field: Dict[str, str] = field(default_factory=dict)

    def tier(self, name: str) -> FrozenSet[str]:
        return self.tiers.get(name, frozenset())

    @property
    def never_determinative(self) -> FrozenSet[str]:
        return self.tier("phonogram_strict")

    def context_field(self, sign: str) -> Optional[str]:
        return self.sign_to_field.get(sign)


def _fs(x) -> FrozenSet[str]:
    return frozenset(x or [])


@lru_cache(maxsize=4)
def load_kb(kb_dir: str = KB_DIR) -> KnowledgeBase:
    with open(os.path.join(kb_dir, "signs.yaml"), encoding="utf-8") as fh:
        signs = yaml.safe_load(fh)
    with open(os.path.join(kb_dir, "categories.yaml"), encoding="utf-8") as fh:
        cat = yaml.safe_load(fh)
    tiers = {k: _fs(v) for k, v in cat["tiers"].items()}
    sign_to_field: Dict[str, str] = {}
    for tier_name, fld in CONTEXT_TIER_TO_FIELD:
        for s in tiers.get(tier_name, ()):
            sign_to_field.setdefault(s, fld)
    ra = cat["recently_added"]
    return KnowledgeBase(
        signs=signs,
        tiers=tiers,
        semantic_fields={k: _fs(v) for k, v in cat["semantic_fields"].items()},
        strict_context_signs=_fs(cat["strict_context_signs"]),
        recently_critical=_fs(ra["critical"]),
        recently_minor=_fs(ra["minor"]),
        recently_added=_fs(ra["all"]),
        strong_determinatives=_fs(cat["strong_determinatives"]),
        plural_markers=_fs(cat["plural_markers"]),
        final_location_signs=_fs(cat["final_location_signs"]),
        location_modifier_signs=_fs(cat["location_modifier_signs"]),
        contextual_only_signs=_fs(cat["contextual_only_signs"]),
        block_end_signs=_fs(cat["block_end_signs"]),
        sign_to_field=sign_to_field,
    )


def context_tiers() -> List[str]:
    return [t for t, _ in CONTEXT_TIER_TO_FIELD]
