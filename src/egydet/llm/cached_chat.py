"""Drop-in replacements for LangChain chat models that go through the disk cache.

``make_cached_chat_class`` returns a class with the constructor signature of the wrapped LangChain
model (``ChatOpenAI``, ``ChatAnthropic``) and an ``invoke(messages)`` method. Modes:

* ``live``    : cache hit -> cached answer; miss -> real API call, stored in the cache.
* ``offline`` : cache hit -> cached answer; miss -> ``CacheMiss`` (never calls an API).
* ``fake``    : always answers ``{}`` without calling anything (for tests of the plumbing).

Every call, hit, miss and error is counted in ``CallStats``.
"""
from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass, field
from typing import Any, List

from egydet.llm.cache import DiskCache, request_key

SECRET_KWARGS = {"api_key", "openai_api_key", "anthropic_api_key"}
MODES = ("live", "offline", "fake")


class CacheMiss(RuntimeError):
    pass


@dataclass
class CallStats:
    api_calls: int = 0
    cache_hits: int = 0
    misses: int = 0
    errors: List[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    answers: List[dict] = field(default_factory=list)   # (system head, content) of every answer, in order

    def snapshot(self) -> dict:
        return {"api_calls": self.api_calls, "cache_hits": self.cache_hits, "misses": self.misses,
                "errors": len(self.errors), "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


def _jsonable(x: Any) -> Any:
    try:
        import json
        json.dumps(x, default=str)
        return x
    except Exception:
        return str(x)


def make_cached_chat_class(base_cls, provider: str, cache: DiskCache, mode: str, stats: CallStats):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    from langchain_core.messages import AIMessage

    class CachedChat:
        _egydet_wrapped = base_cls

        def __init__(self, **kwargs):
            self._init_kwargs = kwargs
            self.params = {k: v for k, v in kwargs.items() if k not in SECRET_KWARGS}
            self._real = None

        def _request(self, messages) -> dict:
            return {"provider": provider, "params": _jsonable(self.params),
                    "messages": [{"role": getattr(m, "type", "unknown"), "content": getattr(m, "content", str(m))}
                                 for m in messages]}

        def invoke(self, messages, **kwargs):
            req = self._request(messages)
            key = request_key(req)
            if mode == "fake":
                stats.cache_hits += 1
                return AIMessage(content="{}")
            system_head = next((m["content"][:120] for m in req["messages"] if m["role"] == "system"), "")
            hit = cache.get(key)
            if hit is not None:
                stats.cache_hits += 1
                stats.answers.append({"system_head": system_head, "content": hit["response"]["content"], "key": key})
                return AIMessage(content=hit["response"]["content"])
            if mode == "offline":
                stats.misses += 1
                raise CacheMiss(key)
            if self._real is None:
                self._real = base_cls(**self._init_kwargs)
            t0 = time.time()
            try:
                resp = self._real.invoke(messages, **kwargs)
            except Exception as exc:
                stats.errors.append(f"{type(exc).__name__}: {str(exc)[:400]}")
                raise
            usage = getattr(resp, "usage_metadata", None) or {}
            stats.api_calls += 1
            stats.input_tokens += int(usage.get("input_tokens", 0) or 0)
            stats.output_tokens += int(usage.get("output_tokens", 0) or 0)
            cache.put(key, {
                "key": key,
                "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "latency_s": round(time.time() - t0, 3),
                "request": req,
                "response": {"content": resp.content,
                             "response_metadata": _jsonable(getattr(resp, "response_metadata", {})),
                             "usage_metadata": _jsonable(usage)},
            })
            stats.answers.append({"system_head": system_head, "content": resp.content, "key": key})
            return resp

    CachedChat.__name__ = f"Cached{base_cls.__name__}"
    return CachedChat
