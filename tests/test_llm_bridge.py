import importlib.util
import json
import os

import pytest

from egydet.llm import CacheMiss, CallStats, DiskCache, make_cached_chat_class

HAS_LANGCHAIN = all(importlib.util.find_spec(m) for m in ("langchain_core", "langchain_openai", "langgraph"))
needs_langchain = pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain/langgraph not installed")


class _FakeResp:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {"model_name": "fake"}
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 3}


class _FakeChat:
    calls = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, messages, **kw):
        _FakeChat.calls += 1
        return _FakeResp('{"determinatives": []}')


class _Msg:
    def __init__(self, type_, content):
        self.type, self.content = type_, content


@needs_langchain
def test_cache_roundtrip_and_offline(tmp_path):
    cache, stats = DiskCache(str(tmp_path)), CallStats()
    Live = make_cached_chat_class(_FakeChat, "openai", cache, "live", stats)
    msgs = [_Msg("system", "s"), _Msg("human", "h")]
    llm = Live(model="m", temperature=0.0, api_key="sk-secret")
    assert llm.invoke(msgs).content == '{"determinatives": []}'
    assert llm.invoke(msgs).content == '{"determinatives": []}'
    assert (stats.api_calls, stats.cache_hits, _FakeChat.calls) == (1, 1, 1)
    stored = [p for p in tmp_path.rglob("*.json")]
    assert len(stored) == 1 and "sk-secret" not in stored[0].read_text(encoding="utf-8")

    Off = make_cached_chat_class(_FakeChat, "openai", cache, "offline", stats)
    assert Off(model="m", temperature=0.0, api_key="other").invoke(msgs).content.startswith("{")
    with pytest.raises(CacheMiss):
        Off(model="m", temperature=0.5).invoke(msgs)      # different parameters -> different key


@needs_langchain
def test_bridge_and_ablation_run_end_to_end_in_fake_mode(tmp_path, monkeypatch):
    """Exercises the v1.47 bridge, record writing and the ablation without any API call."""
    from egydet.legacy_bridge import ablation, collect
    from egydet.scoring.params import preset

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    params_run = tmp_path / "params"
    params_run.mkdir()
    folds = [{"fold": k, "best_label": "x", "inner": [{"label": "x", "params": preset("default")}]} for k in range(1, 6)]
    (params_run / "folds.json").write_text(json.dumps(folds), encoding="utf-8")

    out = tmp_path / "run"
    collect.run("fake", str(params_run), str(out), folds=[1], limit=2, env_file=None)
    recs = ablation.load_records(str(out))
    assert len(recs) == 2 and all(r["success"] and r["calls"]["errors"] == 0 for r in recs)

    res = ablation.evaluate(str(out), str(params_run))
    assert res["diagnostics"]["records"] == 2
    assert {"rules_only", "v147_full", "semantic_raw", "scorer[rules+sem+struct|w=v147]"} <= set(res["conditions"])

    from egydet.legacy_bridge import analysis
    preds, gold, recs2, _ = ablation.predict_conditions(str(out), str(params_run))
    gold = {k: v for k, v in gold.items() if k in preds["rules_only"]}
    comps = analysis.paired_comparisons(preds, gold)
    err = analysis.error_analysis(preds, gold, recs2)
    assert comps and all("p_holm" in c for c in comps)
    assert err["summary"]["TP"] + err["summary"]["FN"] == sum(len(g["det_idx"]) for g in gold.values())

    from openpyxl import load_workbook
    from egydet.legacy_bridge import expert_sheet
    rows = expert_sheet.build_rows(str(out))
    chosen = expert_sheet.sample(rows, seed=1)
    path = tmp_path / "eval.xlsx"
    expert_sheet.write_workbook(str(path), chosen)
    wb = load_workbook(path)
    assert wb.sheetnames == ["Instrucciones", "Evaluación", "Resumen"]
    assert wb["Evaluación"].max_row == len(chosen) + 1


def test_strict_parser_keeps_json_with_typographic_quotes():
    from egydet.legacy_bridge.ablation import parse_answer
    from _legacy import load_v147
    answer = 'BEGIN_JSON\n{"main_interpretation": {"determinatives": [{"token": "A1", "index": 0}]},\n' \
             ' "note": "classifier of “man”",}\nEND_JSON'
    assert parse_answer(answer)["main_interpretation"]["determinatives"][0]["token"] == "A1"
    v147 = load_v147("_extract_json_substring", "parse_llm_json")["parse_llm_json"]
    assert v147(answer) == {}      # v1.47 turns the typographic quotes into '"' and the JSON breaks
