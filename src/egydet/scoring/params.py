"""Scoring parameters: search space and named starting points (configs/scoring/*.yaml)."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Tuple

import yaml

from egydet.kb import ROOT

DEFAULT_CONFIG = os.path.join(ROOT, "configs", "scoring", "v147.yaml")


@lru_cache(maxsize=4)
def _load(path: str = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def param_space(path: str = DEFAULT_CONFIG) -> Dict[str, Tuple[float, float]]:
    """Ordered search space (order matters for reproducing Optuna's sampling sequence)."""
    return {k: (float(lo), float(hi)) for k, (lo, hi) in _load(path)["param_space"].items()}


def preset(name: str, path: str = DEFAULT_CONFIG) -> Dict[str, float]:
    """Named parameter set: 'default' (v1.28 hand-tuned values) or 'flat' (uninformative prior)."""
    return dict(_load(path)[name])


def context_signs(path: str = DEFAULT_CONFIG) -> list:
    return list(_load(path)["context_signs"])
