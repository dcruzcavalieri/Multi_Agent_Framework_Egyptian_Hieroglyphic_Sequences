"""Load the frozen v1.47 script as a module, with its LLM calls routed through the egydet cache.

The v1.47 agents (prompts, parsing, synthesiser post-processing) are used exactly as published: the
script source is executed up to its evaluation block, after replacing the hard-coded ``F:/`` paths
and the LangChain chat classes it imports inside each agent node. Nothing in the frozen file changes.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import types
from typing import Optional

from egydet.kb import ROOT
from egydet.llm import CallStats, DiskCache, make_cached_chat_class

V147_PATH = os.path.join(ROOT, "legacy", "v1_47_paper", "agente_egipsio_langgraph_v1_47.py")
MODULE_NAME = "egydet_v147_runtime"
_CUT_MARKER = "if EVAL_MODE == 'nested_cv' and LOO_DF is not None and LOO_GT_DICT:"
PLACEHOLDER_KEY = "sk-offline-placeholder"


def load_env(env_file: Optional[str] = None) -> Optional[str]:
    """Load API keys from a .env file into the process environment. Returns the file used (never the keys)."""
    candidates = [env_file] if env_file else [os.path.join(ROOT, ".env"), os.path.join(ROOT, "..", ".env")]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                from dotenv import load_dotenv
                load_dotenv(path, override=False)
            except ImportError:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return os.path.abspath(path)
    return None


def _patch_langchain(cache: DiskCache, mode: str, stats: CallStats) -> None:
    import langchain_openai
    langchain_openai.ChatOpenAI = make_cached_chat_class(
        getattr(langchain_openai.ChatOpenAI, "_egydet_wrapped", langchain_openai.ChatOpenAI),
        "openai", cache, mode, stats)
    try:
        import langchain_anthropic
        base = getattr(langchain_anthropic.ChatAnthropic, "_egydet_wrapped", langchain_anthropic.ChatAnthropic)
    except ImportError:  # v1.47 checks that the package is importable; a stub is enough (never selected)
        langchain_anthropic = types.ModuleType("langchain_anthropic")
        sys.modules["langchain_anthropic"] = langchain_anthropic
        base = object
    langchain_anthropic.ChatAnthropic = make_cached_chat_class(base, "anthropic", cache, mode, stats)


class StateCapture:
    """Proxy for the compiled LangGraph app that keeps the last final state."""

    def __init__(self, app):
        self._app = app
        self.last_state = None

    def invoke(self, state, *args, **kwargs):
        self.last_state = None
        out = self._app.invoke(state, *args, **kwargs)
        self.last_state = out
        return out


def load_v147(mode: str = "offline", cache: Optional[DiskCache] = None, stats: Optional[CallStats] = None,
              env_file: Optional[str] = None, log_path: Optional[str] = None) -> types.ModuleType:
    """Return the v1.47 runtime module. ``mode`` is 'live', 'offline' or 'fake' (see egydet.llm)."""
    cache = cache or DiskCache()
    stats = stats if stats is not None else CallStats()
    if mode == "live":
        load_env(env_file)
        if not os.environ.get("OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"] == PLACEHOLDER_KEY:
            raise RuntimeError("OPENAI_API_KEY not found (looked for .env in the repository and its parent folder)")
    elif not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = PLACEHOLDER_KEY   # only to select the OpenAI branch; never sent anywhere
    _patch_langchain(cache, mode, stats)

    src = open(V147_PATH, encoding="utf-8").read()
    out_dir = tempfile.mkdtemp(prefix="v147_out_").replace("\\", "/")
    raw_dir = os.path.join(ROOT, "data", "raw").replace("\\", "/")
    src = src.replace("F:/estancia_espanha_agentes/content/outputs", out_dir)
    src = src.replace("F:/estancia_espanha_agentes/content", raw_dir)
    cut = src.find(_CUT_MARKER)
    if cut < 0:
        raise RuntimeError("v1.47 evaluation block not found; the frozen file changed?")
    src = src[:cut]

    mod = types.ModuleType(MODULE_NAME)
    mod.__file__ = V147_PATH
    sys.modules[MODULE_NAME] = mod
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, V147_PATH, "exec"), mod.__dict__)
    output = buf.getvalue()
    if log_path:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(output)
    if not getattr(mod, "USE_LANGGRAPH", False) or getattr(mod, "LANGGRAPH_APP", None) is None:
        hints = [l.strip() for l in output.splitlines()
                 if any(k in l for k in ("Depend", "LangGraph", "langgraph", "Erro", "API key"))]
        raise RuntimeError("v1.47 did not enable LangGraph. v1.47 said:\n  " + "\n  ".join(hints[:10]))
    if len(getattr(mod, "LOO_GT_DICT", {}) or {}) != 99:
        raise RuntimeError("v1.47 could not load the 99 annotated sequences (python-docx problem?)")
    mod.LANGGRAPH_APP = StateCapture(mod.LANGGRAPH_APP)
    mod.EGYDET_STATS = stats
    return mod
