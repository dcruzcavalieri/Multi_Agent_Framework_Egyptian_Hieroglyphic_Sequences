"""Nested cross-validation of the rule-only scorer.

    python -m egydet.eval.cv --objective legacy_micro_signs   # same objective as v1.47
    python -m egydet.eval.cv --objective macro_positional     # objective matching the metric we report

Outer folds and inner 5x2 halves come from the frozen split file. For every inner half the scorer
parameters are tuned with Optuna TPE (same sampler seeds, warm start and search space as v1.47); the
parameter set with the best validation score is applied to the outer test fold. Every test prediction is
scored both positionally and as a sign multiset. Results go to runs/<timestamp>_<name>/.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from egydet.eval.metrics import aggregate, score_positions, score_signs, summarize_folds
from egydet.eval.splits import DEFAULT_PATH as DEFAULT_SPLITS, load_splits
from egydet.gold.io import load_gold
from egydet.kb import ROOT, KnowledgeBase, load_kb
from egydet.rules.baseline import rule_candidates
from egydet.rules.structure import build_structure
from egydet.scoring.params import param_space, preset
from egydet.scoring.scorer import select


@dataclass
class Sample:
    id: str
    mdc: str
    gold_idx: List[int]
    gold_signs: List[str]
    structure: object
    pool: dict


def prepare(records: List[dict], kb: KnowledgeBase) -> Dict[str, Sample]:
    out = {}
    for r in records:
        st = build_structure(r["mdc"], kb.block_end_signs, kb.plural_markers)
        out[r["id"]] = Sample(r["id"], r["mdc"], r["det_idx"], r["det_signs"], st, rule_candidates(st, kb))
    return out


def predict(s: Sample, p: dict, kb: KnowledgeBase) -> List[int]:
    return select(s.structure, s.pool, p, kb)


# ── objectives ───────────────────────────────────────────────────────────────

def legacy_micro_signs(samples: List[Sample], p: dict, kb: KnowledgeBase) -> float:
    """v1.47 ``_run_loo_fast``: micro-F1 over sign multisets, same formula 2TP/(2TP+FP+FN)."""
    tp = fp = fn = 0
    for s in samples:
        idx = predict(s, p, kb)
        m = score_signs([s.structure.signs[i] for i in idx], s.gold_signs)
        tp += m.tp; fp += m.fp; fn += m.fn
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def macro_positional(samples: List[Sample], p: dict, kb: KnowledgeBase) -> float:
    scores = [score_positions(predict(s, p, kb), s.gold_idx) for s in samples]
    return aggregate(scores)["macro"]["f1"]


OBJECTIVES: Dict[str, Callable] = {"legacy_micro_signs": legacy_micro_signs, "macro_positional": macro_positional}


# ── inner optimisation ──────────────────────────────────────────────────────

def optimise(samples: List[Sample], objective: Callable, kb: KnowledgeBase, n_trials: int, seed: int,
             init: dict) -> tuple:
    """Optuna TPE with the v1.47 warm start. Returns (best_params, best_value, init_value, n_improvements)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = param_space()
    init_value = objective(samples, init, kb)
    n_impr = 0

    def _obj(trial):
        nonlocal n_impr
        p = {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in space.items()}
        for k in ("w_sem_llm", "w_struct_llm", "w_rule_llm"):
            p.setdefault(k, init.get(k, 0.0))
        v = objective(samples, p, kb)
        if v > init_value:
            n_impr += 1
        return v

    import warnings
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # the v1.47 warm start puts w_rule_llm=1.0 outside its range
        study.enqueue_trial({k: init.get(k, (lo + hi) / 2) for k, (lo, hi) in space.items()})
        study.optimize(_obj, n_trials=n_trials)
    best = dict(study.best_params)
    for k in ("w_sem_llm", "w_struct_llm", "w_rule_llm"):
        best[k] = init.get(k, 0.0)
    return best, study.best_value, init_value, n_impr


# ── nested CV ───────────────────────────────────────────────────────────────

def run(objective_name: str = "legacy_micro_signs", n_trials: int = 400, folds: Optional[List[int]] = None,
        splits_path: str = DEFAULT_SPLITS, init_name: str = "flat", log=print) -> dict:
    kb = load_kb()
    gold = load_gold(os.path.join(ROOT, "data", "gold", "gold_v1.jsonl"))
    samples = prepare(gold, kb)
    splits = load_splits(splits_path)
    objective = OBJECTIVES[objective_name]
    init = preset(init_name)
    default = preset("default")

    fold_results, predictions = [], []
    for f in splits["folds"]:
        k = f["fold"]
        if folds and k not in folds:
            continue
        inner = []
        for rep, halves in enumerate(f["inner"]):
            A = [samples[i] for i in halves["A"]]
            B = [samples[i] for i in halves["B"]]
            for train, val, label in ((A, B, f"rep{rep + 1}_AB"), (B, A, f"rep{rep + 1}_BA")):
                seed = rep * 100 + len(label)
                if n_trials > 0:
                    best, opt_v, init_v, n_impr = optimise(train, objective, kb, n_trials, seed, init)
                else:
                    best, opt_v, init_v, n_impr = dict(init), None, objective(train, init, kb), 0
                inner.append({"label": label, "opt": opt_v, "init": init_v, "n_improvements": n_impr,
                              "val": objective(val, best, kb), "baseline_train": legacy_micro_signs(train, default, kb),
                              "params": best})
                log(f"  fold {k} {label}: init={init_v:.4f} opt={opt_v if opt_v is None else round(opt_v, 4)} "
                    f"val={inner[-1]['val']:.4f} improved={n_impr}/{n_trials}")
        best_rec = max(inner, key=lambda r: r["val"])
        test = [samples[i] for i in f["test"]]
        res = {"fold": k, "n_train_val": len(f["train_val"]), "n_test": len(test), "inner": inner,
               "val_mean": sum(r["val"] for r in inner) / len(inner), "best_label": best_rec["label"]}
        for name, params in (("optimised", best_rec["params"]), ("default", default), ("flat", init)):
            pos, sig = [], []
            for s in test:
                idx = predict(s, params, kb)
                signs = [s.structure.signs[i] for i in idx]
                pos.append(score_positions(idx, s.gold_idx))
                sig.append(score_signs(signs, s.gold_signs))
                if name == "optimised":
                    predictions.append({"id": s.id, "fold": k, "pred_idx": idx, "pred_signs": signs,
                                        "gold_idx": s.gold_idx, "gold_signs": s.gold_signs})
            res[name] = {"positional": aggregate(pos), "signs": aggregate(sig)}
        fold_results.append(res)
        o = res["optimised"]
        log(f"fold {k}: test macro-F1 positional={o['positional']['macro']['f1']:.4f} "
            f"signs={o['signs']['macro']['f1']:.4f}")

    summary = {}
    for name in ("optimised", "default", "flat"):
        for crit in ("positional", "signs"):
            for avg in ("macro", "micro"):
                key = f"{name}.{crit}.{avg}_f1"
                summary[key] = summarize_folds([r[name][crit][avg]["f1"] for r in fold_results])
    summary["inner.val_mean"] = summarize_folds([r["val_mean"] for r in fold_results])
    return {"objective": objective_name, "n_trials": n_trials, "init": init_name,
            "splits": splits["name"], "folds": fold_results, "summary": summary, "predictions": predictions}


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def save_run(result: dict, name: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(ROOT, "runs", f"{stamp}_{name}")
    os.makedirs(out, exist_ok=True)
    meta = {k: result[k] for k in ("objective", "n_trials", "init", "splits")}
    meta["git"] = _git_hash()
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(result["summary"], fh, indent=1)
    with open(os.path.join(out, "folds.json"), "w", encoding="utf-8") as fh:
        json.dump(result["folds"], fh, indent=1)
    with open(os.path.join(out, "predictions.jsonl"), "w", encoding="utf-8") as fh:
        for p in result["predictions"]:
            fh.write(json.dumps(p) + "\n")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--objective", choices=sorted(OBJECTIVES), default="legacy_micro_signs")
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)
    result = run(args.objective, args.trials, args.folds)
    out = save_run(result, args.name or f"rule_only_{args.objective}")
    s = result["summary"]
    print(f"\nsaved to {out}")
    for key in ("optimised.positional.macro_f1", "optimised.signs.macro_f1",
                "default.positional.macro_f1", "default.signs.macro_f1"):
        v = s[key]
        print(f"{key:32s} {v['mean']:.4f} ± {v['std_sample']:.4f} (sample sd)")


if __name__ == "__main__":
    main()
