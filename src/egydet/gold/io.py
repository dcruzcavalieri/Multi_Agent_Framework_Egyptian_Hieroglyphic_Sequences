"""Load the canonical gold file."""
from __future__ import annotations

import json
from typing import Dict, List


def load_gold(path: str = "data/gold/gold_v1.jsonl") -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def gold_by_mdc(records: List[dict]) -> Dict[str, dict]:
    return {r["mdc"]: r for r in records}
