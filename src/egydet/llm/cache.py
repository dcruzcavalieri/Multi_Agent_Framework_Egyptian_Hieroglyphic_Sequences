"""Content-addressed disk cache for LLM calls.

Each call is stored as one JSON file under ``<cache_dir>/<key[:2]>/<key>.json`` where ``key`` is the
SHA-256 of the canonical request (provider, model parameters, messages). API keys are never part of
the request record. Re-running an experiment with a warm cache makes zero API calls.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from egydet.kb import ROOT

DEFAULT_CACHE_DIR = os.path.join(ROOT, ".cache", "llm")


def request_key(request: dict) -> str:
    blob = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.dir = cache_dir

    def _path(self, key: str) -> str:
        return os.path.join(self.dir, key[:2], f"{key}.json")

    def get(self, key: str) -> Optional[dict]:
        p = self._path(key)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    def put(self, key: str, record: dict) -> None:
        p = self._path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=1, default=str)
        os.replace(tmp, p)

    def __contains__(self, key: str) -> bool:
        return os.path.exists(self._path(key))
