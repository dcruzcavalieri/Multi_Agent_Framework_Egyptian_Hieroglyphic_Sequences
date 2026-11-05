import json
import os

from egydet.agents import verifier
from egydet.agents.verifier import SCHEMA, parse_judgements, system_prompt, user_prompt
from egydet.eval.cv import prepare
from egydet.eval.fusion import fused_select
from egydet.gold.io import load_gold
from egydet.kb import ROOT, load_kb
from egydet.scoring.params import preset
from egydet.scoring.scorer import select

GOLD = os.path.join(ROOT, "data", "gold", "gold_v1.jsonl")


def test_prompt_has_no_corpus_sequence_and_lists_every_index():
    gold = load_gold(GOLD)
    system = system_prompt()
    assert not any(r["mdc"] in system for r in gold)
    kb = load_kb()
    r = gold[1]
    u = user_prompt(r, "signs", kb)
    assert all(f"{i}: {t}" in u for i, t in enumerate(r["tokens"]))
    assert "Translation:" not in u and "Translation:" in user_prompt(r, "context", kb)
    assert SCHEMA["properties"]["judgements"]["items"]["additionalProperties"] is False


def test_parse_judgements_drops_mismatched_signs():
    r = {"tokens": ["A1", "Z1"]}
    out = parse_judgements({"judgements": [{"index": 0, "sign": "A1", "probability": 1.4, "reason": ""},
                                           {"index": 1, "sign": "N35", "probability": 0.2, "reason": ""}]}, r)
    assert out == {0: {"p": 1.0, "reason": ""}}


def test_fusion_with_zero_weight_is_the_rule_scorer():
    kb = load_kb()
    samples = prepare(load_gold(GOLD), kb)
    p = preset("default")
    for s in samples.values():
        probs = {i: 0.9 for i in range(len(s.structure.signs))}
        for mode in ("veto", "rescue", "both"):
            assert fused_select(s, probs, p, kb, 0.0, 0.5, mode) == select(s.structure, s.pool, p, kb)


def test_verifier_runs_in_fake_mode(tmp_path):
    def fake(req):
        n = sum(1 for line in req["messages"][1]["content"].splitlines() if line[:1].isdigit() and ": " in line)
        return json.dumps({"judgements": [{"index": i, "sign": "?", "probability": 0.5, "reason": ""} for i in range(n)]})
    verifier.run("signs", "fake", "fake-model", str(tmp_path), limit=3, fake_answer=fake)
    lines = (tmp_path / "verifier.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and all(json.loads(l)["error"] == "" for l in lines)


def test_auroc_basic_cases():
    from egydet.agents.diagnose import auroc
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1], [0.9]) == 0.0
    assert auroc([0.5, 0.5], [0.5]) == 0.5
    assert auroc([], [0.3]) is None


def test_fewshot_examples_come_from_training_fold_only():
    from egydet.agents.fewshot import select_examples
    from egydet.eval.splits import load_splits
    gold = {r["id"]: r for r in load_gold(GOLD)}
    for f in load_splits()["folds"]:
        train = [gold[i] for i in f["train_val"]]
        for sid in f["test"] + f["train_val"][:10]:
            ex = select_examples(gold[sid], train, 5)
            ids = {e["id"] for e in ex}
            assert len(ex) == 5 and sid not in ids and not ids & set(f["test"])


def test_user_prompt_without_examples_is_unchanged():
    kb = load_kb()
    r = load_gold(GOLD)[3]
    plain = user_prompt(r, "signs", kb)
    assert plain.startswith("MdC: ") and "Example" not in plain
    with_ex = user_prompt(r, "signs", kb, examples=[{"tokens": ["A1", "Z1"], "det_idx": [1]}])
    assert with_ex.endswith(plain) and "Example 1: A1 [Z1]" in with_ex


def test_lexicon_notes_carry_no_counts_and_cover_every_sign():
    from egydet.agents.lexicon import DEFAULT_PATH, load_lexicon
    if not os.path.exists(DEFAULT_PATH):
        import pytest
        pytest.skip("private lexicon not present")
    gold = load_gold(GOLD)
    kb, lex = load_kb(), load_lexicon()
    system = system_prompt("verifier_v3")
    assert not any(r["mdc"] in system for r in gold)
    for r in gold[:20]:
        u = user_prompt(r, "signs", kb, lexicon=lex)
        notes = u.split("Sign-list notes")[1]
        assert all(f"- {t}:" in notes for t in set(r["tokens"]))
        assert "n_gold" not in u and "in_kb" not in u
    assert "Sign-list notes" not in user_prompt(gold[0], "signs", kb)


def test_rerank_keeps_main_unless_the_verifier_prefers_an_alternative():
    from egydet.eval.rerank import choose
    cands = [(2.0, [3, 5]), (1.5, [5])]
    assert choose(cands, {3: 0.9, 5: 0.9}, K=2, lam=0.0, delta=0.0) == [3, 5]
    assert choose(cands, {3: 0.05, 5: 0.9}, K=2, lam=0.0, delta=0.0) == [5]
    assert choose(cands, {3: 0.05, 5: 0.9}, K=1, lam=0.0, delta=0.0) == [3, 5]
    assert choose(cands, {3: 0.3, 5: 0.9}, K=2, lam=0.0, delta=2.0) == [3, 5]   # margin not reached


def test_reader_prompt_and_reading_substitution(tmp_path):
    from egydet.agents import reader
    gold = load_gold(GOLD)
    system = system_prompt("reader_v1")
    assert not any(r["mdc"] in system for r in gold)
    u = reader.user_prompt(gold[1])
    assert "Translation" not in u and all(f"{i}: {t}" in u for i, t in enumerate(gold[1]["tokens"]))

    def fake_reader(req):
        n = sum(1 for line in req["messages"][1]["content"].splitlines() if line[:1].isdigit() and ": " in line)
        return json.dumps({"words": [{"indices": list(range(n)), "transliteration": "xyz", "gloss": "g"}],
                           "transliteration": "PREDICTED", "translation": "PREDICTED TR"})
    rd = tmp_path / "reader"
    reader.run("fake", "fake-model", str(rd), limit=2, fake_answer=fake_reader)
    recs = reader.load_reading(str(rd))
    assert len(recs) == 2 and all(r["segmentation_ok"] for r in recs.values())
    assert reader.normalise("ꞽr⸗ꞽ [p.t]") == reader.normalise("jr=i p.t")

    seen = []

    def fake_verifier(req):
        seen.append(req["messages"][1]["content"])
        n = sum(1 for line in req["messages"][1]["content"].splitlines() if line[:1].isdigit() and ": " in line)
        return json.dumps({"judgements": [{"index": i, "sign": "?", "probability": 0.5, "reason": ""} for i in range(n)]})
    verifier.run("context", "fake", "fake-model", str(tmp_path / "ver"), limit=2, fake_answer=fake_verifier,
                 reading_run=str(rd))
    assert seen and all("Transliteration: PREDICTED" in s for s in seen)
    assert not any(r["transliteration"] in s for r in gold[:2] for s in seen)


def test_alignment_roles_and_slots_with_a_toy_table(tmp_path):
    from egydet.rules.align import align, load_table, words_of
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"values": {"Q3": ["p"], "X1": ["t"], "N1": ["pt"], "O1": ["pr"], "D21": ["r"],
                                        "I9": ["f"]}}), encoding="utf-8")
    t = load_table(str(p))
    assert words_of("pr⸗f p.t [ꜣḫ.t] ms(.t)") == ["pr", "f", "pt", "Axt", "ms"]
    a = align(["O1", "D21", "D54", "I9", "Q3", "X1", "N1"], "pr⸗f p.t", t)
    assert a["ok"] and a["spans"] == [(0, 3), (3, 4), (4, 7)]
    assert a["roles"][0] == "phon" and a["roles"][1] == "compl"      # O1 writes pr, D21 complements r
    assert a["slots"] == [2, 6]                                        # D54 after pr; N1 redundant after p.t


def test_jsesh_table_keeps_phonogram_values_only():
    from egydet.rules.align import DEFAULT_TABLE, load_table
    t = load_table(DEFAULT_TABLE)
    assert "pr" in t["O1"] and "tA" in t["N16"] and "n" in t["N35"]
    assert "i" not in t.get("A1", [])                      # the suffix-pronoun value is typed ideogram
    assert "i" in load_table(DEFAULT_TABLE, types=None)["A1"]


def test_frozen_pipeline_runs_in_fake_mode(tmp_path):
    from egydet import pipeline
    from egydet.scoring.params import preset
    frozen = {"name": "test", "created": "now", "rules": {"params": preset("default")},
              "veto": {"table": "kb/jsesh_sign_values.json", "types": ["phonogram"]},
              "verifier": {"model": "fake-model", "prompt": "verifier_v1", "condition": "context"},
              "rescue": {"theta": 0.5}}
    recs = load_gold(GOLD)[:3]
    import egydet.agents.client as client_mod
    orig = client_mod.ChatClient.__init__

    def init(self, model, mode="live", cache=None, stats=None, fake_answer=None, **kw):
        def fake(req):
            n = sum(1 for line in req["messages"][1]["content"].splitlines() if line[:1].isdigit() and ": " in line)
            return json.dumps({"judgements": [{"index": i, "sign": "?", "probability": 0.9, "reason": ""}
                                              for i in range(n)]})
        orig(self, model, mode=mode, cache=cache, stats=stats, fake_answer=fake, **kw)
    client_mod.ChatClient.__init__ = init
    try:
        pipeline.predict(recs, frozen, "fake", str(tmp_path))
    finally:
        client_mod.ChatClient.__init__ = orig
    out = [json.loads(l) for l in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(out) == 3
    for o in out:
        ev = o["evidence"]
        assert set(o["det_idx"]) == (set(ev["rules"]) - set(ev["vetoed"])) | set(ev["rescued"])
        assert set(ev["rescued"]) <= set(ev["slots"])
    assert pipeline.write_review_sheet(str(tmp_path)).endswith("proposals.xlsx")
    res = pipeline.evaluate(str(tmp_path), GOLD)
    assert res["n"] == 3
