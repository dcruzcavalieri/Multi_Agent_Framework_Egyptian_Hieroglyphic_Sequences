import math
import os
import random

import pandas as pd
import pytest

from egydet.eval import (
    FN_NULL, FP_NULL, STANDARD, TRUE_NULL, aggregate, legacy_position_to_sign_index,
    per_sign_multiset, per_sign_positions, score_positions, score_signs, summarize_folds,
)
from _legacy import ROOT, load_v147

RESULTS = os.path.join(ROOT, "legacy", "v1_47_paper", "results")


# ── per-sequence outcomes ────────────────────────────────────────────────────

@pytest.mark.parametrize("scorer", [score_positions, score_signs])
def test_null_outcomes(scorer):
    empty, some = [], ([3] if scorer is score_positions else ["A1"])
    s = scorer(empty, empty)
    assert (s.outcome, s.precision, s.recall, s.f1) == (TRUE_NULL, 1.0, 1.0, 1.0)
    s = scorer(some, empty)
    assert (s.outcome, s.fp, s.precision, s.recall, s.f1) == (FP_NULL, 1, 0.0, 1.0, 0.0)
    s = scorer(empty, some)
    assert (s.outcome, s.fn, s.precision, s.recall, s.f1) == (FN_NULL, 1, 1.0, 0.0, 0.0)


def test_positional_differs_from_multiset_when_position_is_wrong():
    tokens = ["A1", "N35", "A1", "Z1"]
    gold_idx, pred_idx = [2, 3], [0, 3]          # predicts the first A1, gold is the second
    pos = score_positions(pred_idx, gold_idx)
    sig = score_signs([tokens[i] for i in pred_idx], [tokens[i] for i in gold_idx])
    assert (pos.tp, pos.fp, pos.fn) == (1, 1, 1)
    assert (sig.tp, sig.fp, sig.fn) == (2, 0, 0)
    assert pos.outcome == sig.outcome == STANDARD


def test_positional_ignores_duplicate_predictions():
    assert score_positions([1, 1, 2], [1, 2]).f1 == 1.0


def test_per_sign_positions():
    out = per_sign_positions([["A1", "N35", "A1", "Z1"]], [[0, 3]], [[2, 3]])
    assert out["A1"] == {"tp": 0, "fp": 1, "fn": 1, "support": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert out["Z1"]["f1"] == 1.0


def test_legacy_position_conversion():
    legacy = ["T18", "-", "S29", "-", "D54", "*", "A40", ":", "Z2"]
    assert [legacy_position_to_sign_index(legacy, p) for p in (0, 2, 4, 6, 8)] == [0, 1, 2, 3, 4]
    assert legacy_position_to_sign_index(legacy, 1) is None
    assert legacy_position_to_sign_index(legacy, 99) is None


def test_summarize_folds_population_std():
    s = summarize_folds([0.8, 0.9])
    assert math.isclose(s["std"], 0.05) and math.isclose(s["std_sample"], 0.0707106781, rel_tol=1e-6)


# ── equivalence with the frozen v1.47 implementation ────────────────────────

@pytest.fixture(scope="module")
def v147():
    return load_v147("_signs_to_multiset", "evaluate_single_prediction", "compute_loo_metrics")


def _random_case(rng):
    vocab = ["A1", "Z1", "Z2", "A40", "N35", "D54"]
    gold = [rng.choice(vocab) for _ in range(rng.choice([0, 0, 1, 2, 4, 6]))]
    pred = [rng.choice(vocab) for _ in range(rng.choice([0, 1, 2, 3, 5]))]
    return pred, gold


def test_score_signs_equals_v147(v147):
    rng = random.Random(0)
    for _ in range(2000):
        pred, gold = _random_case(rng)
        ours = score_signs(pred, gold)
        ref = v147["evaluate_single_prediction"](pred, gold)
        assert (ours.tp, ours.fp, ours.fn, ours.outcome) == (ref["tp"], ref["fp"], ref["fn"], ref["outcome_type"])
        assert math.isclose(ours.f1, ref["f1"]) and math.isclose(ours.precision, ref["precision"])
        assert math.isclose(ours.recall, ref["recall"])


def test_aggregate_equals_v147(v147):
    rng = random.Random(1)
    for _ in range(50):
        cases = [_random_case(rng) for _ in range(rng.randint(1, 25))]
        ours = aggregate([score_signs(p, g) for p, g in cases])
        ref = v147["compute_loo_metrics"]([
            {"predicted_signs": p, "gt_signs": g, "metrics": v147["evaluate_single_prediction"](p, g)}
            for p, g in cases])
        for block in ("micro", "macro", "macro_classic"):
            for k in ("precision", "recall", "f1"):
                assert math.isclose(ours[block][k], ref[block][k], abs_tol=1e-12), (block, k)
        assert ours["outcomes"][TRUE_NULL] == ref["null_stats"]["n_true_null"]
        assert ours["null_accuracy"] == ref["null_stats"]["null_accuracy"]
        assert per_sign_multiset([p for p, _ in cases], [g for _, g in cases]).keys() == ref["per_sign"].keys()


# ── the manuscript numbers, recomputed from the archived v1.47 predictions ──

def _split(x):
    return [] if pd.isna(x) or not str(x).strip() else str(x).split(",")


@pytest.fixture(scope="module")
def archived():
    df = pd.read_csv(os.path.join(RESULTS, "nested_cv_test_samples.csv"))
    df["pred"] = df["predicted_signs"].map(_split)
    df["gold"] = df["gt_signs"].map(_split)
    return df


def test_reproduces_table2_and_table1(archived):
    per_fold = pd.read_csv(os.path.join(RESULTS, "nested_cv_per_fold.csv"))
    folds = []
    for f, grp in archived.groupby("outer_fold"):
        agg = aggregate([score_signs(p, g) for p, g in zip(grp["pred"], grp["gold"])])
        ref = per_fold[per_fold.outer_fold == f].iloc[0]
        assert round(agg["macro"]["f1"], 4) == ref["llm_macro_f1"]
        assert round(agg["micro"]["f1"], 4) == ref["llm_micro_f1"]
        assert round(agg["macro"]["precision"], 4) == ref["llm_macro_p"]
        assert round(agg["macro"]["recall"], 4) == ref["llm_macro_r"]
        assert round(agg["macro_classic"]["f1"], 4) == ref["llm_macro_classic_f1"]
        folds.append(agg)
    macro = summarize_folds([a["macro"]["f1"] for a in folds])
    micro = summarize_folds([a["micro"]["f1"] for a in folds])
    assert (round(macro["mean"], 4), round(macro["std"], 4)) == (0.8194, 0.0219)
    assert (round(micro["mean"], 4), round(micro["std"], 4)) == (0.8750, 0.0250)
    micro_p = summarize_folds([a["micro"]["precision"] for a in folds])
    assert (round(micro_p["mean"], 4), round(micro_p["std"], 4)) == (0.8819, 0.0051)


def test_reproduces_table3_and_outcomes(archived):
    pooled = aggregate([score_signs(p, g) for p, g in zip(archived["pred"], archived["gold"])])
    assert (pooled["micro"]["tp"], pooled["micro"]["fp"], pooled["micro"]["fn"]) == (269, 36, 42)
    assert pooled["outcomes"] == {STANDARD: 88, TRUE_NULL: 5, FP_NULL: 4, FN_NULL: 2}
