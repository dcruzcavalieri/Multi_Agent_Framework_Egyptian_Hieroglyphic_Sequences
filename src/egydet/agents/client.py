"""Minimal OpenAI chat client with the egydet disk cache and structured (JSON-schema) output.

Modes: ``live`` (cache, then API), ``offline`` (cache only; a miss raises CacheMiss), ``fake`` (returns
the ``fake_answer`` callable's output without any call; for tests). Every request record stores the
model, parameters, messages and response format; API keys are never stored.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
from typing import Callable, Optional

from egydet.llm import CacheMiss, CallStats, DiskCache, request_key


def _loads(content: str, stats: CallStats) -> dict:
    try:
        data = json.loads(content or "{}")
        return data if isinstance(data, dict) else {}
    except ValueError:
        stats.errors.append("answer is not valid JSON")
        return {}


class ChatClient:
    def __init__(self, model: str, mode: str = "live", cache: Optional[DiskCache] = None,
                 stats: Optional[CallStats] = None, temperature: Optional[float] = 0.0, seed: Optional[int] = 42,
                 fake_answer: Optional[Callable[[dict], str]] = None):
        self.model, self.mode = model, mode
        self.cache = cache or DiskCache()
        self.stats = stats if stats is not None else CallStats()
        self.temperature, self.seed = temperature, seed
        self.fake_answer = fake_answer
        self._client = None

    def _api(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    def complete_json(self, system: str, user: str, schema: dict, schema_name: str) -> dict:
        params = {"model": self.model, "temperature": self.temperature, "seed": self.seed,
                  "response_format": {"type": "json_schema",
                                      "json_schema": {"name": schema_name, "strict": True, "schema": schema}}}
        req = {"provider": "openai-sdk", "params": params,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        key = request_key(req)
        if self.mode == "fake":
            self.stats.cache_hits += 1
            return json.loads(self.fake_answer(req) if self.fake_answer else "{}")
        hit = self.cache.get(key)
        if hit is not None:
            self.stats.cache_hits += 1
            self.stats.answers.append({"system_head": system[:120], "content": hit["response"]["content"], "key": key})
            return _loads(hit["response"]["content"], self.stats)
        if self.mode == "offline":
            self.stats.misses += 1
            raise CacheMiss(key)
        kwargs = {k: v for k, v in params.items() if v is not None}
        dropped = []
        t0 = time.time()
        while True:
            try:
                resp = self._api().chat.completions.create(messages=req["messages"], **kwargs)
                break
            except Exception as exc:
                msg = str(exc)
                # some models accept only their default sampling parameters; retry without them, and say so
                bad = next((p for p in ("temperature", "seed", "top_p") if p in kwargs and p in msg
                            and ("unsupported" in msg.lower() or "not support" in msg.lower())), None)
                if bad is None:
                    self.stats.errors.append(f"{type(exc).__name__}: {msg[:400]}")
                    raise
                dropped.append(bad)
                kwargs.pop(bad)
        choice = resp.choices[0]
        content = choice.message.content or ""
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        self.stats.api_calls += 1
        self.stats.input_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.stats.output_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.cache.put(key, {
            "key": key, "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "latency_s": round(time.time() - t0, 3), "request": req,
            "dropped_params": dropped,
            "response": {"content": content, "finish_reason": choice.finish_reason,
                         "model": getattr(resp, "model", None),
                         "system_fingerprint": getattr(resp, "system_fingerprint", None), "usage": usage},
        })
        self.stats.answers.append({"system_head": system[:120], "content": content, "key": key})
        if choice.finish_reason != "stop":
            self.stats.errors.append(f"finish_reason={choice.finish_reason}")
        return _loads(content, self.stats)
