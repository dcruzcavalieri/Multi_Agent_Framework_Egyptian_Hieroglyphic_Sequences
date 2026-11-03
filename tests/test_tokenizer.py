import pandas as pd
import pytest

from egydet.mdc import normalize, tokenize
from _legacy import ROOT, load_v147


def test_basic_split_and_offsets():
    toks = tokenize("T18-S29-D54*A40:Z2")
    assert [t.text for t in toks] == ["T18", "S29", "D54", "A40", "Z2"]
    assert [t.index for t in toks] == [0, 1, 2, 3, 4]
    s = normalize("T18-S29-D54*A40:Z2")
    assert all(s[t.start:t.end] == t.text for t in toks)


def test_whitespace_removed_and_empty_input():
    assert [t.text for t in tokenize(" A1 - Z1 ")] == ["A1", "Z1"]
    assert tokenize("") == []
    assert tokenize(None) == []  # type: ignore[arg-type]


def test_consecutive_separators_produce_no_empty_tokens():
    assert [t.text for t in tokenize("A1--Z1::")] == ["A1", "Z1"]


def test_standard_flag():
    flags = {t.text: t.is_standard for t in tokenize("A1-Aa1-W22VARA-N35A-US7063L36VARA-?-Aa1&G25")}
    assert flags["A1"] and flags["Aa1"] and flags["W22VARA"] and flags["N35A"]
    assert not flags["?"] and not flags["Aa1&G25"]


@pytest.fixture(scope="module")
def corpus_codes():
    df = pd.read_excel(f"{ROOT}/data/raw/extractrion_for_agents.xlsx")
    return df["Code"].astype(str).tolist()


def test_same_sign_tokens_as_v147_on_whole_corpus(corpus_codes):
    ns = load_v147("normalize_mdc", "tokenize_mdc", constants=("MDL_SEPARATORS",))
    seps = ns["MDL_SEPARATORS"]
    for code in corpus_codes:
        legacy = [t for t in ns["tokenize_mdc"](code) if t not in seps]
        assert [t.text for t in tokenize(code)] == legacy, code
