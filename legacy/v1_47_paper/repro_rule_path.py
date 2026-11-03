"""Deterministic re-execution of the rule-only path of the frozen v1.47 script.

No API key is used. The script copies v1.47, rewrites its hard-coded F:/ paths to this repository,
cuts it before the main evaluation block, and then:
  * mode "baseline": runs the nested CV with default parameters only (no Optuna), fast;
  * mode "fold1":    runs the real inner 5x2 CV with Optuna for outer fold 1 (about 2-3 minutes).
Results are compared with legacy/v1_47_paper/results/nested_cv_inner_records.csv.

Usage (from the repository root):
    python legacy/v1_47_paper/repro_rule_path.py baseline
    python legacy/v1_47_paper/repro_rule_path.py fold1
Requires: pandas, numpy, openpyxl, python-docx, optuna, matplotlib, seaborn, langgraph, langchain-core.
"""
import contextlib
import io
import os
import re
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(HERE, "agente_egipsio_langgraph_v1_47.py")
REF = os.path.join(HERE, "results", "nested_cv_inner_records.csv")

HARNESS_BASELINE = '''
_dp = _default_params()
def _fake_opt(samples, n_trials=0, seed=0):
    b = _run_loo_fast(samples, _dp)
    return dict(_dp), b, b, b, 0
_optimize_on_samples = _fake_opt
def _fake_llm(test_df, train_val_df, gt_dict, best_params, n_few_shot=5, outer_fold=0):
    out = []
    for s in _precompute_loo_samples(test_df, gt_dict):
        pred = _select_dets_parametric(s, _dp)
        out.append({"loo_index": 0, "outer_fold": outer_fold, "sequence": s["sequence"],
                    "predicted_signs": pred, "gt_signs": s["gt_signs"],
                    "metrics": evaluate_single_prediction(pred, s["gt_signs"]), "few_shot_n": 0})
    return out
_evaluate_test_with_llm = _fake_llm
_CAP = {}
_orig = _aggregate_nested_cv_results
def _agg(o):
    _CAP["o"] = o
    return _orig(o)
_aggregate_nested_cv_results = _agg
run_nested_cv(LOO_DF, LOO_GT_DICT)
REPRO = [(f["outer_fold"] + 1, r["label"], None, None, r["baseline"])
         for f in _CAP["o"] for r in f["inner"]["inner_records"]]
'''

HARNESS_FOLD1 = '''
_splits = _make_outer_splits(LOO_DF, OUTER_N_SPLITS, OUTER_RANDOM_STATE)
_tv = _precompute_loo_samples(LOO_DF.iloc[_splits[0][0]].reset_index(drop=True), LOO_GT_DICT)
_inner = _run_5x2_inner_cv(_tv, INNER_N_REPS, INNER_N_TRIALS, random_state=OUTER_RANDOM_STATE)
REPRO = [(1, r["label"], r["opt_f1"], r["val_f1"], r["baseline"]) for r in _inner["inner_records"]]
'''


def build_namespace(mode: str) -> dict:
    src = open(SRC, encoding="utf-8").read()
    out_dir = tempfile.mkdtemp(prefix="v147_out_")
    src = src.replace("F:/estancia_espanha_agentes/content/outputs", out_dir.replace("\\", "/"))
    src = src.replace("F:/estancia_espanha_agentes/content",
                      os.path.join(ROOT, "data", "raw").replace("\\", "/"))
    cut = re.search(r"^if EVAL_MODE == 'nested_cv' and LOO_DF is not None", src, re.M)
    src = src[:cut.start()] + (HARNESS_BASELINE if mode == "baseline" else HARNESS_FOLD1)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.pop(key, None)
    ns = {"__name__": "__main__"}
    sys.modules["__main__"].__dict__.update(ns)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, SRC, "exec"), sys.modules["__main__"].__dict__)
    return sys.modules["__main__"].__dict__


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    ns = build_namespace(mode)
    got = pd.DataFrame(ns["REPRO"], columns=["outer_fold", "label", "opt_f1", "val_f1", "baseline"])
    ref = pd.read_csv(REF)
    m = got.merge(ref, on=["outer_fold", "label"], suffixes=("_repro", "_paper"))
    cols = ["baseline"] + (["opt_f1", "val_f1"] if mode == "fold1" else [])
    for c in cols:
        m[c + "_match"] = (m[c + "_repro"].astype(float).round(4) == m[c + "_paper"].round(4))
    print(m.to_string(index=False))
    for c in cols:
        print(f"{c}: {int(m[c + '_match'].sum())}/{len(m)} identical")


if __name__ == "__main__":
    main()
