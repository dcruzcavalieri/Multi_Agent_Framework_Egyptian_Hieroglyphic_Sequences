"""Evaluation metrics for determinative detection.

Two matching criteria are provided and must always be reported together, labelled:

* **positional** (primary): a prediction is a true positive only if it points at a gold position
  (sign-token index, see ``egydet.mdc.tokenize``). This is the sequence-labelling metric.
* **sign multiset** (secondary, ``legacy``): predicted and gold *sign codes* are compared as
  multisets, ignoring positions. This is exactly the metric of the frozen v1.47 code and of the
  LRE manuscript; it is kept only for comparability. It counts a correct sign at a wrong position
  as a true positive.

Both use the same null-aware outcome categories and per-sequence conventions as v1.47:

=========  ============  ==========================================  =====  =====  ==
outcome    pred / gold   meaning                                     P      R      F1
=========  ============  ==========================================  =====  =====  ==
TRUE_NULL  empty/empty   correct abstention                          1      1      1
FP_NULL    some/empty    spurious detection                          0      1      0
FN_NULL    empty/some    everything missed                           1      0      0
STANDARD   some/some     ordinary TP/FP/FN computation               ...    ...    ...
=========  ============  ==========================================  =====  =====  ==

The P=1 for FN_NULL and R=1 for FP_NULL are vacuous-truth conventions inherited from v1.47. They
raise macro precision and macro recall slightly; report the outcome counts next to the macro values.

Aggregation (``aggregate``):
* micro: TP/FP/FN summed over sequences (TRUE_NULL contributes nothing);
* macro: mean of per-sequence P, R, F1 with TRUE_NULL included as 1 ("macro corrected" in v1.47,
  the value reported in the manuscript);
* macro_classic: same mean with TRUE_NULL sequences excluded.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import mean, pstdev, stdev
from typing import Dict, Iterable, List, Optional, Sequence

TRUE_NULL, FP_NULL, FN_NULL, STANDARD = "TRUE_NULL", "FP_NULL", "FN_NULL", "STANDARD"


@dataclass(frozen=True)
class SeqScore:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    outcome: str

    def as_dict(self) -> dict:
        return asdict(self)


def _prf(tp: int, fp: int, fn: int) -> tuple:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _score(tp: int, fp: int, fn: int, n_pred: int, n_gold: int) -> SeqScore:
    if n_pred == 0 and n_gold == 0:
        return SeqScore(0, 0, 0, 1.0, 1.0, 1.0, TRUE_NULL)
    if n_gold == 0:
        return SeqScore(0, n_pred, 0, 0.0, 1.0, 0.0, FP_NULL)
    if n_pred == 0:
        return SeqScore(0, 0, n_gold, 1.0, 0.0, 0.0, FN_NULL)
    return SeqScore(tp, fp, fn, *_prf(tp, fp, fn), STANDARD)


def score_positions(pred_idx: Iterable[int], gold_idx: Iterable[int]) -> SeqScore:
    """Positional match. Duplicated indices in the prediction are ignored."""
    pred, gold = set(pred_idx), set(gold_idx)
    tp = len(pred & gold)
    return _score(tp, len(pred - gold), len(gold - pred), len(pred), len(gold))


def score_signs(pred_signs: Sequence[str], gold_signs: Sequence[str]) -> SeqScore:
    """Sign-multiset match, identical to v1.47 ``evaluate_single_prediction``."""
    pm, gm = Counter(pred_signs), Counter(gold_signs)
    tp = sum((pm & gm).values())
    fp = sum((pm - gm).values())
    fn = sum((gm - pm).values())
    return _score(tp, fp, fn, len(pred_signs), len(gold_signs))


def aggregate(scores: Sequence[SeqScore]) -> dict:
    n = len(scores)
    counts = Counter(s.outcome for s in scores)
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    micro_p, micro_r, micro_f = _prf(tp, fp, fn)

    def _macro(subset: List[SeqScore]) -> dict:
        if not subset:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {"precision": mean(s.precision for s in subset),
                "recall": mean(s.recall for s in subset),
                "f1": mean(s.f1 for s in subset)}

    gold_empty = counts[TRUE_NULL] + counts[FP_NULL]
    gold_nonempty = counts[FN_NULL] + counts[STANDARD]
    return {
        "n": n,
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f, "tp": tp, "fp": fp, "fn": fn},
        "macro": _macro(list(scores)),
        "macro_classic": _macro([s for s in scores if s.outcome != TRUE_NULL]),
        "outcomes": {k: counts[k] for k in (STANDARD, TRUE_NULL, FP_NULL, FN_NULL)},
        "null_accuracy": counts[TRUE_NULL] / gold_empty if gold_empty else None,
        "fn_null_rate": counts[FN_NULL] / gold_nonempty if gold_nonempty else None,
    }


def per_sign_positions(tokens_list: Sequence[Sequence[str]], pred_idx_list: Sequence[Iterable[int]],
                       gold_idx_list: Sequence[Iterable[int]]) -> Dict[str, dict]:
    """Positional TP/FP/FN per sign code, pooled over sequences."""
    acc: Dict[str, Counter] = defaultdict(Counter)
    for toks, pred, gold in zip(tokens_list, pred_idx_list, gold_idx_list):
        pred, gold = set(pred), set(gold)
        for i in pred & gold:
            acc[toks[i]]["tp"] += 1
        for i in pred - gold:
            acc[toks[i]]["fp"] += 1
        for i in gold - pred:
            acc[toks[i]]["fn"] += 1
    return _finish_per_sign(acc)


def per_sign_multiset(pred_signs_list: Sequence[Sequence[str]],
                      gold_signs_list: Sequence[Sequence[str]]) -> Dict[str, dict]:
    """Sign-multiset TP/FP/FN per sign code (v1.47 ``per_sign``)."""
    acc: Dict[str, Counter] = defaultdict(Counter)
    for pred, gold in zip(pred_signs_list, gold_signs_list):
        pm, gm = Counter(pred), Counter(gold)
        for s in set(pm) | set(gm):
            acc[s]["tp"] += min(pm[s], gm[s])
            acc[s]["fp"] += max(0, pm[s] - gm[s])
            acc[s]["fn"] += max(0, gm[s] - pm[s])
    return _finish_per_sign(acc)


def _finish_per_sign(acc: Dict[str, Counter]) -> Dict[str, dict]:
    out = {}
    for s in sorted(acc):
        tp, fp, fn = acc[s]["tp"], acc[s]["fp"], acc[s]["fn"]
        p, r, f = _prf(tp, fp, fn)
        out[s] = {"tp": tp, "fp": fp, "fn": fn, "support": tp + fn, "precision": p, "recall": r, "f1": f}
    return out


def summarize_folds(values: Sequence[float]) -> dict:
    """Mean and spread over outer folds.

    ``std`` is the population standard deviation (ddof=0), as in v1.47 (``numpy.std``) and therefore in
    the manuscript; ``std_sample`` (ddof=1) is also given and is the more usual choice for reporting.
    """
    vals = [float(v) for v in values]
    return {
        "values": vals,
        "mean": mean(vals) if vals else 0.0,
        "std": pstdev(vals) if vals else 0.0,
        "std_sample": stdev(vals) if len(vals) > 1 else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def legacy_position_to_sign_index(legacy_tokens: Sequence[str], position: int,
                                  separators: Optional[set] = None) -> Optional[int]:
    """Convert a v1.47 token position (separators included) into a sign-token index.

    Returns None if the position points at a separator or is out of range.
    """
    seps = separators if separators is not None else {"-", ":", "*"}
    if not 0 <= position < len(legacy_tokens) or legacy_tokens[position] in seps:
        return None
    return sum(1 for t in legacy_tokens[:position] if t not in seps)
