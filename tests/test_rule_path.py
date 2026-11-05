"""Regression of the ported rule-only path against the frozen v1.47 code."""
import json
import os
import random

import pandas as pd
import pytest

from egydet.eval.cv import legacy_micro_signs, optimise, prepare
from egydet.eval.splits import load_splits, make_v147_splits
from egydet.gold.io import load_gold
from egydet.kb import context_tiers, load_kb
from egydet.mdc import normalize
from egydet.rules.baseline import rule_candidates
from egydet.rules.structure import build_structure, legacy_tokens, sign_index_of_token
from egydet.scoring.params import param_space, preset
from egydet.scoring.scorer import select
from _legacy import ROOT, load_v147

RESULTS = os.path.join(ROOT, "legacy", "v1_47_paper", "results")

V147_FUNCS = ("normalize_mdc", "tokenize_mdc", "build_mdc_structure", "build_word_boundaries",
              "generate_suffix_span_candidates", "analyze_rule_based", "_gather_rule_evidence",
              "_score_candidate_p", "_score_span_p", "_select_dets_parametric", "validate_semantic_context",
              "_is_plural", "_is_final_location", "_is_location_modifier", "_default_params", "_flat_params")
V147_CONSTS = ("MDL_SEPARATORS", "DETERMINATIVES_DETAILED", "SIGN_CATEGORIES", "CONTEXT_THRESHOLDS",
               "SEMANTIC_FIELDS", "RECENTLY_ADDED_SIGNS", "NEVER_DETERMINATIVE_SIGNS",
               "STRONG_DETERMINATIVE_SIGNS", "PLURAL_MARKERS", "FINAL_LOCATION_SIGNS",
               "LOCATION_MODIFIER_SIGNS", "CONTEXTUAL_ONLY_SIGNS", "CONTEXT_SIGNS_TO_OPTIMIZE", "PARAM_SPACE")


@pytest.fixture(scope="module")
def v147():
    return load_v147(*V147_FUNCS, constants=V147_CONSTS)


@pytest.fixture(scope="module")
def kb():
    return load_kb()


@pytest.fixture(scope="module")
def corpus():
    df = pd.read_excel(os.path.join(ROOT, "data", "raw", "extractrion_for_agents.xlsx"))
    return [normalize(str(c)) for c in df["Code"]]


def _v147_sample(ns, mdc):
    toks = ns["tokenize_mdc"](mdc)
    st = ns["build_mdc_structure"](toks)
    words = ns["build_word_boundaries"](toks, st)
    return {"tokens": toks, "mdc_structure": st, "words": words,
            "word_span_candidates": ns["generate_suffix_span_candidates"](words, toks, max_len=6),
            "rule_evidence_pool": ns["_gather_rule_evidence"](ns["analyze_rule_based"](toks), toks, st)}


def _ours(kb, mdc, p):
    st = build_structure(mdc, kb.block_end_signs, kb.plural_markers)
    return [st.signs[i] for i in select(st, rule_candidates(st, kb), p, kb)], st


def test_kb_matches_v147(v147, kb):
    assert set(kb.signs) == set(v147["DETERMINATIVES_DETAILED"])
    assert {k: set(v) for k, v in kb.tiers.items()} == {k: set(v) for k, v in v147["SIGN_CATEGORIES"].items()}
    tiers = [kb.tier(t) for t in context_tiers()]
    assert sum(len(t) for t in tiers) == len(frozenset().union(*tiers)), "context tiers must be disjoint"
    assert preset("default") == v147["_default_params"]()
    assert preset("flat") == v147["_flat_params"]()
    assert list(param_space()) == list(v147["PARAM_SPACE"])


def test_structure_matches_v147_on_whole_corpus(v147, kb, corpus):
    for mdc in corpus:
        ref = _v147_sample(v147, mdc)
        t2s = sign_index_of_token(ref["tokens"])
        st = build_structure(mdc, kb.block_end_signs, kb.plural_markers)
        rs = ref["mdc_structure"]
        assert st.block_final == {t2s[i] for i in rs["block_final_token_indices"]}, mdc
        assert st.near_end == {t2s[i] for i in rs["last_sign_token_indices"]}, mdc
        assert st.plural_markers == {t2s[i] for i in rs["plural_marker_indices"]}, mdc
        assert st.plural_heads == {t2s[i] for i in rs["plural_head_indices"]}, mdc
        ref_spans = tuple(tuple(tuple(t2s[i] for i in c["span_token_indices"]) for c in w["candidates"])
                          for w in ref["word_span_candidates"])
        assert st.span_candidates == ref_spans, mdc


def _random_params(rng):
    p = {k: lo + rng.random() * (hi - lo) for k, (lo, hi) in param_space().items()}
    p.update({"w_sem_llm": 0.0, "w_struct_llm": 0.0, "w_rule_llm": 1.0})
    return p


def test_selection_matches_v147(v147, kb, corpus):
    rng = random.Random(7)
    param_sets = [preset("default"), preset("flat")] + [_random_params(rng) for _ in range(25)]
    for mdc in corpus:
        ref_sample = _v147_sample(v147, mdc)
        for p in param_sets:
            ours, _ = _ours(kb, mdc, p)
            assert ours == v147["_select_dets_parametric"](ref_sample, p), mdc


def test_splits_reproduce_v147_folds():
    gold = sorted(load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl")), key=lambda r: r["corpus_row"])
    frozen = load_splits()
    assert frozen == json.loads(json.dumps(make_v147_splits([r["id"] for r in gold])))
    by_id = {r["id"]: r["mdc"] for r in gold}
    archived = pd.read_csv(os.path.join(RESULTS, "nested_cv_test_samples.csv"))
    for f in frozen["folds"]:
        assert sorted(by_id[i] for i in f["test"]) == sorted(archived[archived.outer_fold == f["fold"]].sequence)


def test_inner_baseline_matches_archive_for_fold1(kb):
    """Fold 1 ran before any global state was modified in v1.47, so it is the clean reference."""
    samples = prepare(load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl")), kb)
    f1 = load_splits()["folds"][0]
    ref = pd.read_csv(os.path.join(RESULTS, "nested_cv_inner_records.csv"))
    ref = ref[ref.outer_fold == 1].set_index("label")["baseline"]
    for rep, halves in enumerate(f1["inner"]):
        for train, label in ((halves["A"], f"rep{rep + 1}_AB"), (halves["B"], f"rep{rep + 1}_BA")):
            v = legacy_micro_signs([samples[i] for i in train], preset("default"), kb)
            assert round(v, 4) == ref[label], label


@pytest.mark.skipif(not os.environ.get("EGYDET_SLOW"), reason="slow (about 3 min); set EGYDET_SLOW=1")
def test_inner_optuna_matches_archive_for_fold1(kb):
    samples = prepare(load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl")), kb)
    f1 = load_splits()["folds"][0]
    ref = pd.read_csv(os.path.join(RESULTS, "nested_cv_inner_records.csv"))
    ref = ref[ref.outer_fold == 1].set_index("label")
    hits = 0
    for rep, halves in enumerate(f1["inner"]):
        A, B = [samples[i] for i in halves["A"]], [samples[i] for i in halves["B"]]
        for train, val, label in ((A, B, f"rep{rep + 1}_AB"), (B, A, f"rep{rep + 1}_BA")):
            best, opt, _, _ = optimise(train, legacy_micro_signs, kb, 400, rep * 100 + len(label), preset("flat"))
            hits += round(opt, 4) == ref.loc[label, "opt_f1"]
            hits += round(legacy_micro_signs(val, best, kb), 4) == ref.loc[label, "val_f1"]
    assert hits >= 19   # 19/20 with Optuna 5.0; the archive was produced with an older Optuna


def test_default_params_test_scores_per_fold(kb):
    """Rule-only scorer with the v1.28 default parameters on each outer test fold (no Optuna).

    Reference values: v1.47 rule path re-executed with the default parameters on each fold.
    """
    from egydet.eval.metrics import aggregate, score_positions
    samples = prepare(load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl")), kb)
    got = []
    for f in load_splits()["folds"]:
        scores = [score_positions(select(samples[i].structure, samples[i].pool, preset("default"), kb),
                                  samples[i].gold_idx) for i in f["test"]]
        got.append(round(aggregate(scores)["macro"]["f1"], 4))
    assert got == [0.8038, 0.8974, 0.9019, 0.8239, 0.8651]


def test_ranked_selection_starts_with_the_main_set(kb):
    from egydet.scoring.scorer import select_ranked
    for r in load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl")):
        st = build_structure(r["mdc"], kb.block_end_signs, kb.plural_markers)
        pool = rule_candidates(st, kb)
        for p in (preset("default"), preset("flat")):
            ranked = select_ranked(st, pool, p, kb, beam_width=10, k=4)
            assert ranked[0][1] == sorted(select(st, pool, p, kb)), r["id"]
            assert len({tuple(s) for _, s in ranked}) == len(ranked)
