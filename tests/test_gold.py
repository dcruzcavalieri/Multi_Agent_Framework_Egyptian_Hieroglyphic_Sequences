import filecmp
import os
import re
from collections import Counter

import pytest

from egydet.gold import build
from egydet.gold.io import load_gold
from egydet.mdc import tokenize
from _legacy import ROOT, load_v147

RAW = os.path.join(ROOT, "data", "raw")
GOLD = os.path.join(ROOT, "data", "gold", "gold_v1.jsonl")


@pytest.fixture(scope="module")
def gold():
    return load_gold(GOLD)


def test_size_and_totals_match_the_paper(gold):
    assert len(gold) == 99
    assert sum(len(r["det_idx"]) for r in gold) == 311          # Table 3: 311 actual positives
    assert sum(1 for r in gold if not r["det_idx"]) == 9         # §5.8: 9 sequences with empty GT


def test_records_are_internally_consistent(gold):
    ids = [r["id"] for r in gold]
    assert len(set(ids)) == len(ids)
    for r in gold:
        toks = [t.text for t in tokenize(r["mdc"])]
        assert toks == r["tokens"]
        assert r["det_idx"] == sorted(set(r["det_idx"]))
        assert all(0 <= i < len(toks) for i in r["det_idx"])
        assert [toks[i] for i in r["det_idx"]] == r["det_signs"]


def test_same_sign_lists_as_the_v147_ground_truth(gold):
    ns = load_v147("parse_ground_truth_from_docx", "load_loo_ground_truth")
    legacy = ns["load_loo_ground_truth"]([os.path.join(RAW, f) for f in build.DOCX_FILES])
    assert set(legacy) == {re.sub(r"\s+", "", r["mdc"]) for r in gold}
    for r in gold:
        assert Counter(legacy[r["mdc"]]) == Counter(r["det_signs"]), r["id"]


def test_rebuild_is_deterministic(tmp_path):
    build.main(["--raw", RAW, "--out", str(tmp_path)])
    assert filecmp.cmp(tmp_path / "gold_v1.jsonl", GOLD, shallow=False)
