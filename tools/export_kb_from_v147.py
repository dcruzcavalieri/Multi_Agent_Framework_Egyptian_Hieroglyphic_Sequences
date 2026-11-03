"""One-off export of the knowledge base and scoring defaults from the frozen v1.47 script.

Run from the repository root:
    python tools/export_kb_from_v147.py

Writes kb/signs.yaml, kb/categories.yaml and configs/scoring/v147.yaml. The values are copied
verbatim (evaluated from the v1.47 source, so later duplicate keys win exactly as in Python).
After this export the YAML files are the source of truth; edit them, not this script.
"""
from __future__ import annotations

import ast
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V147 = os.path.join(ROOT, "legacy", "v1_47_paper", "agente_egipsio_langgraph_v1_47.py")

NAMES = [
    "DETERMINATIVES_DETAILED", "SIGN_CATEGORIES", "CONTEXT_THRESHOLDS", "SEMANTIC_FIELDS",
    "RECENTLY_ADDED_SIGNS", "STRONG_DETERMINATIVE_SIGNS", "PLURAL_MARKERS", "FINAL_LOCATION_SIGNS",
    "LOCATION_MODIFIER_SIGNS", "CONTEXTUAL_ONLY_SIGNS", "PARAM_SPACE", "CONTEXT_SIGNS_TO_OPTIMIZE",
]
FUNCS = ["_default_params", "_flat_params"]


def _load() -> dict:
    tree = ast.parse(open(V147, encoding="utf-8").read())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in NAMES for t in node.targets):
            keep.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in NAMES:
            keep.append(node)
        elif isinstance(node, ast.For) and "CONTEXT_SIGNS_TO_OPTIMIZE" in ast.unparse(node.iter):
            keep.append(node)  # adds thresh_<sign> bounds to PARAM_SPACE
        elif isinstance(node, ast.FunctionDef) and node.name in FUNCS:
            keep.append(node)
    ns: dict = {"Dict": dict, "List": list}
    exec(compile(ast.Module(body=keep, type_ignores=[]), V147, "exec"), ns)
    return ns


def _sorted(s) -> list:
    return sorted(s)


def main() -> None:
    ns = _load()
    cats = ns["SIGN_CATEGORIES"]
    th = ns["CONTEXT_THRESHOLDS"]

    signs = {}
    for code, info in ns["DETERMINATIVES_DETAILED"].items():
        entry = {"meaning": info.get("significado", ""), "category": info.get("categoria", ""),
                 "semantics": info.get("semantica", "")}
        for k, v in info.items():
            if k not in ("significado", "categoria", "semantica"):
                entry[k] = v
        signs[code] = entry

    categories = {
        "_note": ("Exported verbatim from v1.47 (2026-11-03). Lists are sets: order has no meaning. "
                  "Changes to Egyptological content require sign-off by the domain experts."),
        "tiers": {k: _sorted(v) for k, v in cats.items()},
        "semantic_fields": {k: _sorted(v) for k, v in ns["SEMANTIC_FIELDS"].items()},
        # validate_semantic_context() applies a stricter evidence rule to signs whose entry in the
        # CONTEXT_THRESHOLDS table is >= 0.85; stored here as an explicit list.
        "strict_context_signs": _sorted(s for s, v in th.items() if v >= 0.85),
        "recently_added": {
            "critical": ["I10", "I9", "L1"],
            "minor": ["D58", "I5", "I6", "L2", "U1"],
            "all": _sorted(ns["RECENTLY_ADDED_SIGNS"]),
        },
        "strong_determinatives": _sorted(ns["STRONG_DETERMINATIVE_SIGNS"]),
        "plural_markers": _sorted(ns["PLURAL_MARKERS"]),
        "final_location_signs": _sorted(ns["FINAL_LOCATION_SIGNS"]),
        "location_modifier_signs": _sorted(ns["LOCATION_MODIFIER_SIGNS"]),
        "contextual_only_signs": _sorted(ns["CONTEXTUAL_ONLY_SIGNS"]),
        "block_end_signs": ["O1", "O31", "O49"],   # local constant inside v1.47 build_mdc_structure()
    }

    scoring = {
        "_note": "Scoring parameters of v1.47: search space and the two starting points used there.",
        "context_signs": list(ns["CONTEXT_SIGNS_TO_OPTIMIZE"]),
        "param_space": {k: [float(lo), float(hi)] for k, (lo, hi) in ns["PARAM_SPACE"].items()},
        "default": {k: float(v) for k, v in ns["_default_params"]().items()},
        "flat": {k: float(v) for k, v in ns["_flat_params"]().items()},
    }

    def dump(obj, rel):
        path = os.path.join(ROOT, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            yaml.safe_dump(obj, fh, allow_unicode=True, sort_keys=False, width=110)
        print("wrote", rel)

    dump(signs, "kb/signs.yaml")
    dump(categories, "kb/categories.yaml")
    dump(scoring, "configs/scoring/v147.yaml")


if __name__ == "__main__":
    sys.exit(main())
